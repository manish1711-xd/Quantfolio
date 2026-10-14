"""Rule-based portfolio insights: computed numbers turned into sentences.

Nothing in this module is predicted, forecast or "AI-advised". Every sentence
is fired by one explicit numeric comparison, and every :class:`Insight`
carries an ``evidence`` string naming BOTH the observed value AND the
threshold that fired it -- so the UI can always answer "why am I being told
this?" and a reviewer can audit the rule without reading the code.

Threshold provenance
--------------------
Almost every threshold comes from ``config.INSIGHTS`` (an ``InsightThresholds``
dataclass) and is re-exported here under a module-level name so the rules read
like prose and the values stay tunable in exactly one place. Three references
have no field in ``config.InsightThresholds`` and are therefore *declared and
documented* at the top of this module (the same pattern ``analytics.technicals``
uses): ``CRITICAL_ESCALATION``, ``MEANINGFUL_R_SQUARED`` and
``RISK_WEIGHT_RATIO``. A few rules compare against a natural parity point
rather than a tunable -- zero excess return, 100% capture, a correlation of
1.0, the equal-weight share ``5/N`` -- and those parity points are named
verbatim in the evidence string.

Input contract (units matter, and they are not guessed)
-------------------------------------------------------
``positions``   ``analytics.positions.build_positions`` frame; money columns in
                the portfolio's base currency.
``summary``     ``analytics.positions.summarise`` dict. ``largest_holding`` /
                ``best`` / ``worst`` are ``(ticker, value)`` tuples;
                ``weight_pct``-family values are PERCENT; ``sectors`` maps a
                sector label to its PERCENT weight.
``risk``        ``analytics.risk.risk_summary`` dict. ``ann_vol``, ``mdd``,
                ``var``, ``cvar``, ``downside_dev`` are DECIMALS; ``var`` and
                ``cvar`` are NEGATIVE one-day losses; ``ulcer`` is PERCENT;
                ``hit_ratio`` is a fraction; the ratio family is unitless.
``bench``       ``analytics.benchmark.compare`` dict; ``ok`` gates the whole
                section. Return/vol/alpha/tracking-error values are DECIMALS.
``corr``        square correlation frame of asset daily returns.
``episodes``    ``list[DrawdownEpisode]`` (deepest first, per
                ``analytics.drawdown.drawdown_episodes``); read via ``getattr``
                so any duck-typed episode works.
``attribution`` ``analytics.attribution.attribution_summary`` dict; the
                ``return_contribution`` / ``risk_contribution`` /
                ``risk_vs_weight`` frames are used when present and the module
                falls back to ``positions`` when they are not.

Everything is read defensively: a missing key, a NaN, an empty frame or a
wrong type silently disables the rule that needed it rather than raising or
inventing a number. No input is ever mutated.

The returned list is ordered ``critical -> warning -> positive -> info``
(a stable sort, so within a level the emission order below is preserved).
None of this is investment advice; the app shows ``config.DISCLAIMER``
alongside it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import INSIGHTS, MIN_OBSERVATIONS, RISK_RUBRIC, TRADING_DAYS
from src.models import Insight

__all__ = [
    "CATEGORIES",
    "CONCENTRATION_WARN_PCT",
    "CONCENTRATION_INFO_PCT",
    "SECTOR_CONCENTRATION_PCT",
    "MIN_DIVERSIFIED_HOLDINGS",
    "HIGH_CORRELATION",
    "DEEP_DRAWDOWN",
    "SEVERE_DRAWDOWN",
    "VOL_TOLERANCE",
    "AGGRESSIVE_BETA",
    "DEFENSIVE_BETA",
    "GOOD_SHARPE",
    "GOOD_SORTINO",
    "WEAK_CALMAR",
    "HIGH_VAR_PCT",
    "FAT_TAIL_KURTOSIS",
    "LONG_RECOVERY_DAYS",
    "EFFECTIVE_HOLDINGS_RATIO",
    "HIGH_VOLATILITY",
    "CRITICAL_ESCALATION",
    "MEANINGFUL_R_SQUARED",
    "RISK_WEIGHT_RATIO",
    "generate_insights",
    # Underscore-prefixed but part of the published contract: the correlation
    # page calls it directly with threshold=-1.0 to list every pair.
    "_high_correlation_pairs",
    "insights_by_category",
    "headline_insight",
]

# ---------------------------------------------------------------------------
# thresholds sourced from config.INSIGHTS (names kept for readability)
# ---------------------------------------------------------------------------
#: Largest single position, in percent of portfolio value, above which the
#: book is called concentrated.
CONCENTRATION_WARN_PCT: float = INSIGHTS.concentration_warn_pct
CONCENTRATION_INFO_PCT: float = INSIGHTS.concentration_info_pct
SECTOR_CONCENTRATION_PCT: float = INSIGHTS.sector_concentration_pct
MIN_DIVERSIFIED_HOLDINGS: int = INSIGHTS.min_diversified_holdings
HIGH_CORRELATION: float = INSIGHTS.high_correlation
DEEP_DRAWDOWN: float = INSIGHTS.deep_drawdown
SEVERE_DRAWDOWN: float = INSIGHTS.severe_drawdown
VOL_TOLERANCE: float = INSIGHTS.vol_tolerance
AGGRESSIVE_BETA: float = INSIGHTS.aggressive_beta
DEFENSIVE_BETA: float = INSIGHTS.defensive_beta
GOOD_SHARPE: float = INSIGHTS.good_sharpe
GOOD_SORTINO: float = INSIGHTS.good_sortino
WEAK_CALMAR: float = INSIGHTS.weak_calmar
HIGH_VAR_PCT: float = INSIGHTS.high_var_pct
FAT_TAIL_KURTOSIS: float = INSIGHTS.fat_tail_kurtosis
LONG_RECOVERY_DAYS: int = INSIGHTS.long_recovery_days
EFFECTIVE_HOLDINGS_RATIO: float = INSIGHTS.effective_holdings_ratio

#: Annualised volatility that already scores FULL risk points in the published
#: rubric, reused here as the "this is a high-volatility book" line. Taken from
#: ``config.RISK_RUBRIC`` rather than restated, so the narrative and the score
#: cannot disagree.
HIGH_VOLATILITY: float = RISK_RUBRIC.vol_cap

# ---------------------------------------------------------------------------
# references declared here because config.InsightThresholds has no field
# ---------------------------------------------------------------------------
#: Multiple of a warning threshold at which a single-threshold rule escalates
#: from "warning" to "critical". A book with 60% in one name is not merely the
#: 40% rule again; the escalation factor keeps that distinction without adding
#: a second constant per rule.
CRITICAL_ESCALATION: float = 1.5

#: Benchmark R-squared below which beta, alpha and the capture ratios are
#: describing a relationship that mostly is not there. Half the variance is the
#: conventional "the regression is worth quoting" line.
MEANINGFUL_R_SQUARED: float = 0.50

#: Risk contribution divided by portfolio weight, above which a position is
#: carrying materially more risk than its size suggests.
RISK_WEIGHT_RATIO: float = 1.25

#: The vocabulary of ``Insight.category`` produced by this module, in the order
#: :func:`insights_by_category` presents them.
CATEGORIES: tuple[str, ...] = (
    "concentration",
    "diversification",
    "risk",
    "performance",
    "benchmark",
    "data",
)

#: Display order of severities; matches ``analytics.technicals``.
_LEVEL_RANK: dict[str, int] = {"critical": 0, "warning": 1, "positive": 2, "info": 3}

#: Sector labels that are an absence of data, not a sector.
_UNKNOWN_SECTORS: frozenset[str] = frozenset({"unclassified", "unknown", "n/a", ""})

#: Most correlated pairs quoted before the list stops being readable.
_MAX_CORRELATION_PAIRS: int = 3

#: Trading days per month, used only to phrase a sample size in months.
_DAYS_PER_MONTH: float = TRADING_DAYS / 12.0


# ---------------------------------------------------------------------------
# coercion helpers (private)
# ---------------------------------------------------------------------------
def _as_float(value: Any) -> float:
    """Best-effort float; NaN for anything unparseable, None or infinite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int; ``default`` for anything unparseable or non-finite."""
    number = _as_float(value)
    return int(number) if np.isfinite(number) else default


def _num(source: Any, key: str) -> float:
    """Read one numeric field out of a mapping without ever raising."""
    if not isinstance(source, Mapping):
        return float("nan")
    return _as_float(source.get(key))


def _labelled(source: Any, key: str) -> tuple[str | None, float]:
    """Read a ``(label, value)`` summary tuple such as ``largest_holding``."""
    value = source.get(key) if isinstance(source, Mapping) else None
    if isinstance(value, (tuple, list)) and len(value) >= 2 and value[0] is not None:
        return str(value[0]), _as_float(value[1])
    return None, float("nan")


def _fmt(value: float, decimals: int = 2) -> str:
    """Fixed-decimal number for evidence strings; 'n/a' when undefined."""
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


def _sfmt(value: float, decimals: int = 2) -> str:
    """Signed fixed-decimal number; 'n/a' when undefined."""
    return f"{value:+.{decimals}f}" if np.isfinite(value) else "n/a"


def _money(value: float, currency: str = "") -> str:
    """Thousands-grouped signed amount with an optional currency code."""
    if not np.isfinite(value):
        return "n/a"
    suffix = f" {currency}" if currency else ""
    return f"{value:+,.0f}{suffix}"


def _plain_money(value: float, currency: str = "") -> str:
    """Thousands-grouped unsigned amount with an optional currency code."""
    if not np.isfinite(value):
        return "n/a"
    suffix = f" {currency}" if currency else ""
    return f"{value:,.0f}{suffix}"


def _date_text(value: Any) -> str:
    """ISO date for evidence strings; 'n/a' when the episode has none."""
    if value is None:
        return "n/a"
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _as_ratio(value: float) -> float:
    """Normalise a value that may arrive as a fraction or as a percent.

    Capture ratios and R-squared are defined as fractions (``0.86``), but a
    sibling module could legitimately hand them over already multiplied by
    100. Anything above 1.5 is therefore read as a percent and divided back
    down; 1.5 is safely above any plausible R-squared and below any plausible
    percent-form capture figure.
    """
    if not np.isfinite(value):
        return float("nan")
    return value / 100.0 if abs(value) > 1.5 else value


def _numeric_column(frame: Any, column: str) -> pd.Series:
    """Coerced copy of one column, or an empty float Series when unusable."""
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _extreme_row(frame: Any, column: str, largest: bool) -> tuple[str | None, float]:
    """``(ticker, value)`` of the max/min finite entry of ``column``.

    Returns ``(None, nan)`` for a missing frame, a missing column or an
    all-NaN column, so callers never have to guard ``idxmax`` themselves.
    """
    series = _numeric_column(frame, column).dropna()
    if series.empty or "ticker" not in getattr(frame, "columns", []):
        return None, float("nan")

    # POSITIONAL lookup. idxmax/idxmin return an index *label*, and on a frame
    # with a duplicated index -- what any caller who concatenates or slices
    # positions without reset_index produces -- .loc[label] yields multiple rows
    # and float() raises. positions.summarise was hardened the same way.
    values = series.to_numpy(dtype="float64")
    position = int(values.argmax() if largest else values.argmin())
    label = series.index[position]
    try:
        row = frame.loc[label, "ticker"]
        ticker = str(row.iloc[0] if hasattr(row, "iloc") else row)
    except (KeyError, IndexError):  # pragma: no cover - defensive
        return None, float("nan")
    return ticker, float(values[position])


def _recovery_gain_pct(depth: float) -> float:
    """Gain needed to erase a drawdown: ``1 / (1 + d) - 1`` in percent.

    A 50% fall needs a 100% rise. This asymmetry is the single most useful
    thing to say about a drawdown number, so it is quoted with every one.
    """
    if not np.isfinite(depth) or depth <= -1.0:
        return float("nan")
    return float((1.0 / (1.0 + depth) - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# correlation helpers (public: the app calls _high_correlation_pairs directly)
# ---------------------------------------------------------------------------
def _high_correlation_pairs(
    corr: pd.DataFrame | None, threshold: float
) -> list[tuple[str, str, float]]:
    """Upper-triangle ``(a, b, rho)`` pairs with ``rho >= threshold``.

    Sorted strongest first. Called with ``threshold=-1.0`` the filter cannot
    bind (Pearson correlation is bounded below by -1), so the call returns
    EVERY finite pair sorted strongest-first -- which is exactly how the
    correlation page lists the whole matrix, and how
    :func:`_average_correlation` below computes its mean.

    Never mutates ``corr``; a missing, empty, non-square or single-asset frame
    yields an empty list rather than an exception. Self-pairs and duplicated
    labels are skipped, and non-finite cells are dropped rather than compared.
    """
    if not isinstance(corr, pd.DataFrame) or corr.empty:
        return []

    limit = _as_float(threshold)
    if not np.isfinite(limit):
        return []

    labels = [str(c) for c in corr.columns]
    values = corr.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = min(len(labels), values.shape[0], values.shape[1])
    if n < 2:
        return []

    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            value = values[i, j]
            if np.isfinite(value) and value >= limit:
                pairs.append((labels[i], labels[j], float(value)))
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def _average_correlation(corr: pd.DataFrame | None) -> tuple[float, int]:
    """Mean of every finite off-diagonal correlation, and the pair count."""
    pairs = _high_correlation_pairs(corr, -1.0)
    if not pairs:
        return float("nan"), 0
    return float(np.mean([p[2] for p in pairs])), len(pairs)


# ---------------------------------------------------------------------------
# rules: concentration
# ---------------------------------------------------------------------------
def _single_name_insight(summary: Mapping[str, Any], n_holdings: int) -> Insight | None:
    """Largest position against ``concentration_warn_pct`` / ``_info_pct``."""
    ticker, weight = _labelled(summary, "largest_holding")
    if ticker is None or not np.isfinite(weight):
        return None

    critical_at = CONCENTRATION_WARN_PCT * CRITICAL_ESCALATION
    others = max(n_holdings - 1, 0)

    if weight >= critical_at:
        return Insight(
            level="critical",
            text=(
                f"{ticker} is {weight:.1f}% of portfolio value. At that size a "
                f"single-name event - an earnings miss, a regulatory order, a "
                f"downgrade - sets the portfolio's outcome whatever the other "
                f"{others} position(s) do."
            ),
            category="concentration",
            evidence=(
                f"weight={weight:.1f}% >= {critical_at:.1f}% "
                f"(INSIGHTS.concentration_warn_pct={CONCENTRATION_WARN_PCT:.1f}% "
                f"x {CRITICAL_ESCALATION:.1f} critical escalation)"
            ),
        )
    if weight >= CONCENTRATION_WARN_PCT:
        return Insight(
            level="warning",
            text=(
                f"{ticker} is {weight:.1f}% of portfolio value; above the "
                f"{CONCENTRATION_WARN_PCT:.0f}% concentration threshold a single-name "
                f"shock dominates portfolio outcomes. A 20% fall in {ticker} alone "
                f"costs the portfolio {weight * 0.20:.1f}%."
            ),
            category="concentration",
            evidence=(
                f"weight={weight:.1f}% >= "
                f"INSIGHTS.concentration_warn_pct={CONCENTRATION_WARN_PCT:.1f}%"
            ),
        )
    if weight >= CONCENTRATION_INFO_PCT:
        return Insight(
            level="info",
            text=(
                f"{ticker} is the largest position at {weight:.1f}% of value - "
                f"material but not dominant. It is the name whose results move "
                f"the portfolio most."
            ),
            category="concentration",
            evidence=(
                f"weight={weight:.1f}% is in "
                f"[INSIGHTS.concentration_info_pct={CONCENTRATION_INFO_PCT:.1f}%, "
                f"INSIGHTS.concentration_warn_pct={CONCENTRATION_WARN_PCT:.1f}%)"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"No position dominates: the largest, {ticker}, is {weight:.1f}% of "
            f"value. Idiosyncratic single-name risk is contained by construction."
        ),
        category="concentration",
        evidence=(
            f"largest weight={weight:.1f}% < "
            f"INSIGHTS.concentration_info_pct={CONCENTRATION_INFO_PCT:.1f}%"
        ),
    )


def _sector_insight(summary: Mapping[str, Any]) -> Insight | None:
    """Heaviest classified sector against ``sector_concentration_pct``."""
    sectors = summary.get("sectors") if isinstance(summary, Mapping) else None
    if not isinstance(sectors, Mapping) or not sectors:
        return None

    classified = {
        str(name): _as_float(weight)
        for name, weight in sectors.items()
        if str(name).strip().lower() not in _UNKNOWN_SECTORS and np.isfinite(_as_float(weight))
    }
    if not classified:
        return None

    name, weight = max(classified.items(), key=lambda item: item[1])
    critical_at = min(SECTOR_CONCENTRATION_PCT * CRITICAL_ESCALATION, 100.0)
    n_sectors = len(classified)

    if weight >= critical_at:
        return Insight(
            level="critical",
            text=(
                f"{weight:.1f}% of the book sits in {name}. Sector-wide shocks - "
                f"a rate cycle, an input-cost spike, a policy change - hit every "
                f"one of those names at once, so the {n_sectors}-sector split is "
                f"nominal rather than real diversification."
            ),
            category="concentration",
            evidence=(
                f"{name} weight={weight:.1f}% >= {critical_at:.1f}% "
                f"(INSIGHTS.sector_concentration_pct={SECTOR_CONCENTRATION_PCT:.1f}% "
                f"x {CRITICAL_ESCALATION:.1f} critical escalation)"
            ),
        )
    if weight >= SECTOR_CONCENTRATION_PCT:
        return Insight(
            level="warning",
            text=(
                f"{name} is {weight:.1f}% of portfolio value across "
                f"{n_sectors} sector(s). Names inside one sector share their "
                f"macro driver, so this is closer to one large bet than to "
                f"several independent ones."
            ),
            category="concentration",
            evidence=(
                f"{name} weight={weight:.1f}% >= "
                f"INSIGHTS.sector_concentration_pct={SECTOR_CONCENTRATION_PCT:.1f}%"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"Sector exposure is spread: the heaviest, {name}, is {weight:.1f}% "
            f"of value across {n_sectors} sectors. No single macro driver owns "
            f"the portfolio."
        ),
        category="concentration",
        evidence=(
            f"largest sector {name}={weight:.1f}% < "
            f"INSIGHTS.sector_concentration_pct={SECTOR_CONCENTRATION_PCT:.1f}%"
        ),
    )


def _top_five_insight(summary: Mapping[str, Any], n_holdings: int) -> Insight | None:
    """Top-5 weight against the equal-weight reference ``5 / N``."""
    top5 = _num(summary, "top5_weight_pct")
    if not np.isfinite(top5) or n_holdings <= 5:
        return None

    reference = 5.0 / n_holdings * 100.0
    if reference <= 0:
        return None
    ratio = top5 / reference

    level = "warning" if ratio >= CRITICAL_ESCALATION and top5 >= CONCENTRATION_WARN_PCT else "info"
    return Insight(
        level=level,
        text=(
            f"The five largest positions carry {top5:.1f}% of value out of "
            f"{n_holdings} names - {ratio:.2f}x the {reference:.1f}% an "
            f"equal-weight book of this size would give them."
            + (
                " The tail of the portfolio is decoration; the top five are the portfolio."
                if ratio >= CRITICAL_ESCALATION
                else " The weight is spread close to evenly across the book."
            )
        ),
        category="concentration",
        evidence=(
            f"top5_weight_pct={top5:.1f}% vs equal-weight reference "
            f"5/{n_holdings}={reference:.1f}% (ratio={ratio:.2f}x vs the "
            f"{CRITICAL_ESCALATION:.1f}x escalation factor, floor "
            f"INSIGHTS.concentration_warn_pct={CONCENTRATION_WARN_PCT:.1f}%)"
        ),
    )


def _risk_hog_insight(attribution: Any, currency: str) -> Insight | None:
    """Names whose risk contribution outruns their weight by RISK_WEIGHT_RATIO."""
    if not isinstance(attribution, Mapping):
        return None

    frame = attribution.get("risk_vs_weight")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        frame = attribution.get("risk_contribution")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "ticker" not in frame.columns:
        return None

    weight = _numeric_column(frame, "weight_pct")
    risk = _numeric_column(frame, "risk_contribution_pct")
    if weight.empty or risk.empty:
        return None

    ratio = _numeric_column(frame, "ratio")
    if ratio.empty or not ratio.notna().any():
        ratio = risk.where(weight > 0) / weight.where(weight > 0)

    usable = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        return None

    idx = usable.idxmax()
    worst = float(usable.loc[idx])
    if worst < RISK_WEIGHT_RATIO:
        return None

    ticker = str(frame.loc[idx, "ticker"])
    w = _as_float(weight.get(idx))
    r = _as_float(risk.get(idx))
    return Insight(
        level="warning",
        text=(
            f"{ticker} is {_fmt(w, 1)}% of the money but {_fmt(r, 1)}% of the "
            f"risk - {worst:.2f}x its weight. Trimming it removes more variance "
            f"per rupee of turnover than trimming anything else in the book."
        ),
        category="concentration",
        evidence=(
            f"risk_contribution_pct={_fmt(r, 1)}% / weight_pct={_fmt(w, 1)}% "
            f"= {worst:.2f}x >= RISK_WEIGHT_RATIO={RISK_WEIGHT_RATIO:.2f}x "
            f"(1.00x = risk exactly proportional to weight)"
            + (f"; base currency {currency}" if currency else "")
        ),
    )


def _concentration_insights(
    positions: pd.DataFrame,
    summary: Mapping[str, Any],
    attribution: Any,
    currency: str,
) -> list[Insight]:
    """Single-name, sector, top-five and risk-vs-weight concentration rules."""
    n_holdings = _as_int(summary.get("n_holdings"), len(positions))
    candidates = (
        _single_name_insight(summary, n_holdings),
        _sector_insight(summary),
        _top_five_insight(summary, n_holdings),
        _risk_hog_insight(attribution, currency),
    )
    return [item for item in candidates if item is not None]


# ---------------------------------------------------------------------------
# rules: diversification
# ---------------------------------------------------------------------------
def _holdings_count_insight(n_holdings: int) -> Insight | None:
    """Position count against ``min_diversified_holdings``."""
    if n_holdings < 1:
        return None
    if n_holdings < MIN_DIVERSIFIED_HOLDINGS:
        return Insight(
            level="warning",
            text=(
                f"Only {n_holdings} position(s). Below "
                f"{MIN_DIVERSIFIED_HOLDINGS} names the marginal name still "
                f"removes a large slice of idiosyncratic variance, so the "
                f"portfolio is being paid for stock-picking risk it does not "
                f"have to take."
            ),
            category="diversification",
            evidence=(
                f"n_holdings={n_holdings} < "
                f"INSIGHTS.min_diversified_holdings={MIN_DIVERSIFIED_HOLDINGS}"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"{n_holdings} positions clears the {MIN_DIVERSIFIED_HOLDINGS}-name "
            f"floor where diversification benefits are still steep; what matters "
            f"from here is how the weights and correlations are distributed, not "
            f"the count."
        ),
        category="diversification",
        evidence=(
            f"n_holdings={n_holdings} >= "
            f"INSIGHTS.min_diversified_holdings={MIN_DIVERSIFIED_HOLDINGS}"
        ),
    )


def _effective_bets_insight(
    summary: Mapping[str, Any], risk: Mapping[str, Any], n_holdings: int
) -> Insight | None:
    """Effective number of bets (1/HHI) against ``effective_holdings_ratio``."""
    enb = _num(summary, "effective_holdings")
    if not np.isfinite(enb):
        enb = _num(risk, "effective_bets")
    if not np.isfinite(enb) or n_holdings < 2:
        return None

    ratio = enb / n_holdings
    hhi = _num(summary, "hhi")
    hhi_text = f", HHI={_fmt(hhi, 4)}" if np.isfinite(hhi) else ""

    if ratio < EFFECTIVE_HOLDINGS_RATIO:
        return Insight(
            level="warning",
            text=(
                f"{n_holdings} tickers but only {enb:.1f} effective bets "
                f"({ratio * 100:.0f}% of the nominal count). The weight "
                f"distribution, not the ticker list, is deciding this "
                f"portfolio's outcome."
            ),
            category="diversification",
            evidence=(
                f"effective_holdings={enb:.2f} / n_holdings={n_holdings} "
                f"= {ratio:.2f} < "
                f"INSIGHTS.effective_holdings_ratio={EFFECTIVE_HOLDINGS_RATIO:.2f}"
                f"{hhi_text}"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"{n_holdings} tickers deliver {enb:.1f} effective bets "
            f"({ratio * 100:.0f}% of the nominal count) - the weights are close "
            f"enough to even that the names actually count."
        ),
        category="diversification",
        evidence=(
            f"effective_holdings={enb:.2f} / n_holdings={n_holdings} "
            f"= {ratio:.2f} >= "
            f"INSIGHTS.effective_holdings_ratio={EFFECTIVE_HOLDINGS_RATIO:.2f}"
            f"{hhi_text}"
        ),
    )


def _correlation_insights(corr: pd.DataFrame | None) -> list[Insight]:
    """High pairwise correlations and the average correlation of the book."""
    if not isinstance(corr, pd.DataFrame) or corr.shape[0] < 2:
        return []

    out: list[Insight] = []
    pairs = _high_correlation_pairs(corr, HIGH_CORRELATION)
    for rank, (a, b, rho) in enumerate(pairs[:_MAX_CORRELATION_PAIRS]):
        shared = rho * rho * 100.0
        out.append(
            Insight(
                level="warning" if rank == 0 else "info",
                text=(
                    f"{a} and {b} move together with correlation {rho:.2f}: about "
                    f"{shared:.0f}% of their variance is shared. Holding both is "
                    f"closer to holding one at double the weight than to holding "
                    f"two independent bets."
                ),
                category="diversification",
                evidence=(
                    f"corr({a}, {b})={rho:.3f} >= "
                    f"INSIGHTS.high_correlation={HIGH_CORRELATION:.2f}; "
                    f"r^2={rho * rho:.2f} of shared variance"
                ),
            )
        )

    mean_rho, n_pairs = _average_correlation(corr)
    if np.isfinite(mean_rho) and n_pairs > 0:
        if not pairs:
            out.append(
                Insight(
                    level="positive",
                    text=(
                        f"No pair of holdings is more correlated than "
                        f"{HIGH_CORRELATION:.2f}; average pairwise correlation is "
                        f"{mean_rho:.2f} across {n_pairs} pairs. The names are "
                        f"genuinely doing different jobs."
                    ),
                    category="diversification",
                    evidence=(
                        f"max pairwise correlation < "
                        f"INSIGHTS.high_correlation={HIGH_CORRELATION:.2f}; "
                        f"mean rho={mean_rho:.3f} over {n_pairs} pairs"
                    ),
                )
            )
        else:
            out.append(
                Insight(
                    level="info",
                    text=(
                        f"Average pairwise correlation is {mean_rho:.2f} across "
                        f"{n_pairs} pairs. Diversification stops paying once the "
                        f"average correlation is high - the residual portfolio "
                        f"variance floor is roughly the average correlation itself."
                    ),
                    category="diversification",
                    evidence=(
                        f"mean pairwise correlation={mean_rho:.3f} over {n_pairs} "
                        f"pairs vs INSIGHTS.high_correlation="
                        f"{HIGH_CORRELATION:.2f}"
                    ),
                )
            )
    return out


def _diversification_insights(
    positions: pd.DataFrame,
    summary: Mapping[str, Any],
    risk: Mapping[str, Any],
    corr: pd.DataFrame | None,
) -> list[Insight]:
    """Count, effective-bets and correlation-structure rules."""
    n_holdings = _as_int(summary.get("n_holdings"), len(positions))
    out: list[Insight] = []
    for item in (
        _holdings_count_insight(n_holdings),
        _effective_bets_insight(summary, risk, n_holdings),
    ):
        if item is not None:
            out.append(item)
    out.extend(_correlation_insights(corr))
    return out


# ---------------------------------------------------------------------------
# rules: drawdown
# ---------------------------------------------------------------------------
def _max_drawdown_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Max drawdown against ``severe_drawdown`` / ``deep_drawdown``."""
    mdd = _num(risk, "mdd")
    if not np.isfinite(mdd):
        return None

    depth_pct = abs(mdd) * 100.0
    back = _recovery_gain_pct(mdd)

    if mdd <= SEVERE_DRAWDOWN:
        return Insight(
            level="critical",
            text=(
                f"The portfolio fell {depth_pct:.1f}% peak-to-trough in this "
                f"window. Recovering that requires a {_fmt(back, 1)}% gain from "
                f"the trough - drawdowns are not symmetric, and a loss this deep "
                f"changes what the portfolio can be asked to do next."
            ),
            category="risk",
            evidence=(
                f"max_drawdown={mdd * 100:.2f}% <= "
                f"INSIGHTS.severe_drawdown={SEVERE_DRAWDOWN * 100:.2f}%; "
                f"break-even gain required={_fmt(back, 2)}%"
            ),
        )
    if mdd <= DEEP_DRAWDOWN:
        return Insight(
            level="warning",
            text=(
                f"Maximum drawdown was {depth_pct:.1f}% - past the "
                f"{abs(DEEP_DRAWDOWN) * 100:.0f}% line where investors typically "
                f"stop holding. It takes a {_fmt(back, 1)}% gain from the trough "
                f"just to get back to the previous high."
            ),
            category="risk",
            evidence=(
                f"max_drawdown={mdd * 100:.2f}% <= "
                f"INSIGHTS.deep_drawdown={DEEP_DRAWDOWN * 100:.2f}%; "
                f"break-even gain required={_fmt(back, 2)}%"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"Maximum drawdown was {depth_pct:.1f}%, shallower than the "
            f"{abs(DEEP_DRAWDOWN) * 100:.0f}% deep-drawdown line; a "
            f"{_fmt(back, 1)}% gain erases it. Shallow drawdowns are what make a "
            f"strategy holdable through a cycle."
        ),
        category="risk",
        evidence=(
            f"max_drawdown={mdd * 100:.2f}% > "
            f"INSIGHTS.deep_drawdown={DEEP_DRAWDOWN * 100:.2f}%; "
            f"break-even gain required={_fmt(back, 2)}%"
        ),
    )


def _ulcer_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Ulcer index quoted against the worst single drawdown."""
    ulcer = _num(risk, "ulcer")
    mdd = _num(risk, "mdd")
    if not np.isfinite(ulcer) or not np.isfinite(mdd) or ulcer <= 0:
        return None

    depth_pct = abs(mdd) * 100.0
    share = ulcer / depth_pct if depth_pct > 0 else float("nan")
    if np.isfinite(share) and share >= 0.5:
        reading = (
            "the portfolio spent most of the window near its worst level, so the "
            "drawdown was a state rather than a moment"
        )
    else:
        reading = (
            "the typical day underwater was far shallower than the worst one, so "
            "the maximum drawdown was a spike rather than a state"
        )
    return Insight(
        level="info",
        text=(
            f"Ulcer index {ulcer:.2f} against a {depth_pct:.1f}% worst drawdown: "
            f"{reading}. The ulcer index is the root-mean-square of every daily "
            f"drawdown, so it prices duration as well as depth."
        ),
        category="risk",
        evidence=(
            f"ulcer_index={ulcer:.2f}% (RMS drawdown) vs "
            f"max_drawdown={depth_pct:.2f}%; ratio={_fmt(share, 2)}"
        ),
    )


def _episode_insights(episodes: Sequence[Any] | None) -> list[Insight]:
    """Recovery time and open drawdowns from the episode list."""
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        return []
    usable = [ep for ep in episodes if np.isfinite(_as_float(getattr(ep, "depth", None)))]
    if not usable:
        return []

    out: list[Insight] = []
    deepest = min(usable, key=lambda ep: _as_float(getattr(ep, "depth", None)))
    depth = _as_float(getattr(deepest, "depth", None))
    depth_pct = abs(depth) * 100.0
    peak = _date_text(getattr(deepest, "peak_date", None))
    trough = _date_text(getattr(deepest, "trough_date", None))
    fall_days = _as_int(getattr(deepest, "drawdown_days", None), 0)
    recovery_date = getattr(deepest, "recovery_date", None)
    days = _as_float(getattr(deepest, "recovery_days", None))

    if recovery_date is None or not np.isfinite(days):
        out.append(
            Insight(
                level="warning",
                text=(
                    f"The deepest drawdown ({depth_pct:.1f}% from the {peak} peak "
                    f"to the {trough} trough, {fall_days} calendar days down) has "
                    f"not recovered inside this window. An unrecovered loss is a "
                    f"live position, not history."
                ),
                category="risk",
                evidence=(
                    f"depth={depth * 100:.2f}%, recovery_date=None (still "
                    f"underwater) vs "
                    f"INSIGHTS.long_recovery_days={LONG_RECOVERY_DAYS} days"
                ),
            )
        )
    elif days > LONG_RECOVERY_DAYS:
        out.append(
            Insight(
                level="warning",
                text=(
                    f"Recovering the {depth_pct:.1f}% drawdown took {days:.0f} "
                    f"calendar days from the {trough} trough. Long recoveries cost "
                    f"compounding time, not just capital - money underwater is "
                    f"money not working."
                ),
                category="risk",
                evidence=(
                    f"recovery_days={days:.0f} > "
                    f"INSIGHTS.long_recovery_days={LONG_RECOVERY_DAYS}; "
                    f"depth={depth * 100:.2f}%, peak={peak}, trough={trough}"
                ),
            )
        )
    else:
        out.append(
            Insight(
                level="positive",
                text=(
                    f"The deepest drawdown ({depth_pct:.1f}%) was fully recovered "
                    f"in {days:.0f} calendar days from the {trough} trough - fast "
                    f"enough that the portfolio spent most of the window making "
                    f"new highs rather than repairing old ones."
                ),
                category="risk",
                evidence=(
                    f"recovery_days={days:.0f} <= "
                    f"INSIGHTS.long_recovery_days={LONG_RECOVERY_DAYS}; "
                    f"depth={depth * 100:.2f}%"
                ),
            )
        )

    deep = [ep for ep in usable if _as_float(getattr(ep, "depth", None)) <= DEEP_DRAWDOWN]
    if len(deep) > 1:
        out.append(
            Insight(
                level="info",
                text=(
                    f"{len(deep)} separate drawdowns beyond "
                    f"{abs(DEEP_DRAWDOWN) * 100:.0f}% occurred in this window. "
                    f"Repeated deep falls are a property of the strategy, not bad "
                    f"luck in one month."
                ),
                category="risk",
                evidence=(
                    f"episodes with depth <= "
                    f"INSIGHTS.deep_drawdown={DEEP_DRAWDOWN * 100:.2f}%: "
                    f"{len(deep)} of {len(usable)} recorded"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# rules: tail risk and volatility
# ---------------------------------------------------------------------------
def _volatility_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Annualised volatility against the risk rubric's saturation point."""
    vol = _num(risk, "ann_vol")
    if not np.isfinite(vol):
        return None

    vol_pct = vol * 100.0
    daily = vol / np.sqrt(TRADING_DAYS) * 100.0 if vol > 0 else float("nan")
    critical_at = HIGH_VOLATILITY * CRITICAL_ESCALATION

    if vol >= critical_at:
        return Insight(
            level="critical",
            text=(
                f"Annualised volatility is {vol_pct:.1f}% - a typical day moves "
                f"the book {_fmt(daily, 2)}%, and a two-standard-deviation year "
                f"spans roughly {2 * vol_pct:.0f} percentage points either side of "
                f"the mean. This is single-stock, not portfolio, volatility."
            ),
            category="risk",
            evidence=(
                f"ann_vol={vol_pct:.2f}% >= {critical_at * 100:.1f}% "
                f"(RISK_RUBRIC.vol_cap={HIGH_VOLATILITY * 100:.1f}% "
                f"x {CRITICAL_ESCALATION:.1f} critical escalation)"
            ),
        )
    if vol >= HIGH_VOLATILITY:
        return Insight(
            level="warning",
            text=(
                f"Annualised volatility is {vol_pct:.1f}%, at or beyond the "
                f"{HIGH_VOLATILITY * 100:.0f}% point where the risk score "
                f"saturates. A {_fmt(daily, 2)}% average daily move is the price "
                f"of admission for this book."
            ),
            category="risk",
            evidence=(
                f"ann_vol={vol_pct:.2f}% >= RISK_RUBRIC.vol_cap={HIGH_VOLATILITY * 100:.1f}%"
            ),
        )
    if vol <= HIGH_VOLATILITY / 2.0:
        return Insight(
            level="positive",
            text=(
                f"Annualised volatility is {vol_pct:.1f}%, under half the "
                f"{HIGH_VOLATILITY * 100:.0f}% level the risk rubric treats as "
                f"saturated. Day-to-day moves average {_fmt(daily, 2)}%, which is "
                f"the kind of ride an investor actually stays on."
            ),
            category="risk",
            evidence=(
                f"ann_vol={vol_pct:.2f}% <= RISK_RUBRIC.vol_cap/2={HIGH_VOLATILITY * 50:.1f}%"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Annualised volatility is {vol_pct:.1f}% (about {_fmt(daily, 2)}% a "
            f"day) - inside the normal equity range and below the "
            f"{HIGH_VOLATILITY * 100:.0f}% risk-rubric ceiling."
        ),
        category="risk",
        evidence=(
            f"ann_vol={vol_pct:.2f}% is in "
            f"({HIGH_VOLATILITY * 50:.1f}%, "
            f"RISK_RUBRIC.vol_cap={HIGH_VOLATILITY * 100:.1f}%)"
        ),
    )


def _var_insight(risk: Mapping[str, Any], value: float, currency: str) -> Insight | None:
    """One-day historical VaR against ``high_var_pct``."""
    var = _num(risk, "var")
    if not np.isfinite(var) or var >= 0:
        return None

    confidence = _num(risk, "confidence")
    conf_pct = confidence * 100.0 if np.isfinite(confidence) else float("nan")
    var_pct = abs(var) * 100.0
    money = abs(var) * value if np.isfinite(value) and value > 0 else float("nan")
    money_text = (
        f" - about {_plain_money(money, currency)} on today's value" if np.isfinite(money) else ""
    )
    frequency = (
        1.0 / (1.0 - confidence)
        if np.isfinite(confidence) and 0.0 < confidence < 1.0
        else float("nan")
    )
    critical_at = HIGH_VAR_PCT * CRITICAL_ESCALATION

    if var_pct >= critical_at:
        level = "critical"
        evidence = (
            f"VaR({_fmt(conf_pct, 0)}%, historical)={var * 100:.2f}% and "
            f"|VaR|={var_pct:.2f}% >= {critical_at:.2f}% "
            f"(INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}% "
            f"x {CRITICAL_ESCALATION:.1f} critical escalation)"
        )
    elif var_pct >= HIGH_VAR_PCT:
        level = "warning"
        evidence = (
            f"VaR({_fmt(conf_pct, 0)}%, historical)={var * 100:.2f}% and "
            f"|VaR|={var_pct:.2f}% >= INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}%"
        )
    else:
        level = "positive"
        evidence = (
            f"VaR({_fmt(conf_pct, 0)}%, historical)={var * 100:.2f}% and "
            f"|VaR|={var_pct:.2f}% < INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}%"
        )

    return Insight(
        level=level,
        text=(
            f"One-day {_fmt(conf_pct, 0)}% VaR is {var * 100:.2f}%: on the worst "
            f"1 session in {_fmt(frequency, 0)}, history says the portfolio "
            f"loses at least {var_pct:.2f}%{money_text}. VaR is a threshold, "
            f"never a worst case."
        ),
        category="risk",
        evidence=evidence,
    )


def _cvar_insight(risk: Mapping[str, Any], value: float, currency: str) -> Insight | None:
    """Expected shortfall against ``high_var_pct`` and against VaR itself."""
    cvar = _num(risk, "cvar")
    var = _num(risk, "var")
    if not np.isfinite(cvar) or cvar >= 0:
        return None

    confidence = _num(risk, "confidence")
    conf_pct = confidence * 100.0 if np.isfinite(confidence) else float("nan")
    cvar_pct = abs(cvar) * 100.0
    multiple = cvar / var if np.isfinite(var) and var < 0 else float("nan")
    money = abs(cvar) * value if np.isfinite(value) and value > 0 else float("nan")
    money_text = (
        f" ({_plain_money(money, currency)} on today's value)" if np.isfinite(money) else ""
    )
    critical_at = HIGH_VAR_PCT * CRITICAL_ESCALATION

    if cvar_pct >= critical_at:
        level = "critical"
        bound = (
            f">= {critical_at:.2f}% (INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}% "
            f"x {CRITICAL_ESCALATION:.1f} critical escalation)"
        )
    elif cvar_pct >= HIGH_VAR_PCT:
        level = "warning"
        bound = f">= INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}%"
    else:
        level = "positive"
        bound = f"< INSIGHTS.high_var_pct={HIGH_VAR_PCT:.2f}%"

    return Insight(
        level=level,
        text=(
            f"Expected shortfall (CVaR) at {_fmt(conf_pct, 0)}% is "
            f"{cvar * 100:.2f}%{money_text} - {_fmt(multiple, 2)}x the VaR "
            f"threshold. That is the average loss on the days that do breach VaR, "
            f"which is the number that actually describes the tail."
        ),
        category="risk",
        evidence=(
            f"CVaR({_fmt(conf_pct, 0)}%, historical)={cvar * 100:.2f}%, "
            f"|CVaR|={cvar_pct:.2f}% {bound}; "
            f"CVaR/VaR={_fmt(multiple, 2)}x"
        ),
    )


def _fat_tail_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Excess kurtosis against ``fat_tail_kurtosis``, priced through the VaR gap."""
    kurtosis = _num(risk, "kurtosis")
    if not np.isfinite(kurtosis):
        return None

    gaussian = _num(risk, "var_gaussian")
    cornish = _num(risk, "var_cornish_fisher")
    gap = (
        (abs(cornish) - abs(gaussian)) * 100.0
        if np.isfinite(gaussian) and np.isfinite(cornish)
        else float("nan")
    )
    gap_text = (
        f" A normal-distribution VaR of {gaussian * 100:.2f}% becomes "
        f"{cornish * 100:.2f}% once the Cornish-Fisher expansion prices in the "
        f"skew and the tails - {_fmt(gap, 2)} percentage points of understated "
        f"risk."
        if np.isfinite(gap)
        else ""
    )

    if kurtosis >= FAT_TAIL_KURTOSIS:
        return Insight(
            level="warning",
            text=(
                f"Daily returns have excess kurtosis {kurtosis:.2f}: extreme moves "
                f"arrive far more often than a normal distribution allows."
                f"{gap_text}"
            ),
            category="risk",
            evidence=(
                f"excess_kurtosis={kurtosis:.2f} >= "
                f"INSIGHTS.fat_tail_kurtosis={FAT_TAIL_KURTOSIS:.2f} "
                f"(0.00 = Gaussian); VaR_gaussian={_fmt(gaussian * 100, 2)}% vs "
                f"VaR_cornish_fisher={_fmt(cornish * 100, 2)}%"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Excess kurtosis is {kurtosis:.2f}, below the "
            f"{FAT_TAIL_KURTOSIS:.1f} fat-tail line, so the Gaussian VaR is not "
            f"badly misspecified for this window. Kurtosis is regime-dependent: a "
            f"calm sample understates the tails of the next crisis."
        ),
        category="risk",
        evidence=(
            f"excess_kurtosis={kurtosis:.2f} < "
            f"INSIGHTS.fat_tail_kurtosis={FAT_TAIL_KURTOSIS:.2f} (0.00 = Gaussian)"
        ),
    )


def _skew_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Return skew against the zero-skew (symmetric) reference."""
    skew = _num(risk, "skew")
    if not np.isfinite(skew):
        return None

    tail = _num(risk, "tail_ratio")
    tail_text = f" The 95th/5th percentile tail ratio is {tail:.2f}." if np.isfinite(tail) else ""
    if skew < 0:
        return Insight(
            level="info",
            text=(
                f"Return skew is {skew:+.2f}: the left tail is longer than the "
                f"right, so the big down days are bigger than the big up days."
                f"{tail_text} Negative skew is normal for long equity - it is the "
                f"risk the equity premium pays for."
            ),
            category="risk",
            evidence=(
                f"skew={skew:+.3f} < 0.000 (symmetric-distribution reference); "
                f"tail_ratio={_fmt(tail, 2)} vs 1.00 parity"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"Return skew is {skew:+.2f}: the largest moves lean positive, so the "
            f"upside days outsize the downside ones over this window.{tail_text}"
        ),
        category="risk",
        evidence=(
            f"skew={skew:+.3f} >= 0.000 (symmetric-distribution reference); "
            f"tail_ratio={_fmt(tail, 2)} vs 1.00 parity"
        ),
    )


def _consistency_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Hit ratio and gain-to-pain: frequency of winning versus size of winning."""
    hit = _num(risk, "hit_ratio")
    gain_pain = _num(risk, "gain_to_pain")
    if not np.isfinite(hit) and not np.isfinite(gain_pain):
        return None

    hit_pct = hit * 100.0 if np.isfinite(hit) else float("nan")
    if not np.isfinite(gain_pain):
        verdict = "the pain-weighted comparison is undefined (no losing sessions)"
    elif gain_pain >= 1.0:
        verdict = (
            f"every unit of loss sat through was repaid with {gain_pain:.2f} units of net return"
        )
    elif gain_pain > 0:
        verdict = (
            f"the net return was positive but only {gain_pain:.2f} units per unit of loss endured"
        )
    else:
        verdict = "the losses were not repaid at all - the net return is negative"

    return Insight(
        level="info",
        text=(
            f"Positive on {_fmt(hit_pct, 1)}% of sessions with a gain-to-pain "
            f"ratio of {_fmt(gain_pain, 2)}: {verdict}. Hit rate counts days, "
            f"gain-to-pain weighs them - a portfolio can win most days and still "
            f"lose money."
        ),
        category="risk",
        evidence=(
            f"hit_ratio={_fmt(hit_pct, 2)}% vs the 50.00% coin-flip reference; "
            f"gain_to_pain={_fmt(gain_pain, 3)} vs 0.000 break-even "
            f"(1.000 = one unit of return per unit of loss endured)"
        ),
    )


def _risk_insights(
    risk: Mapping[str, Any],
    summary: Mapping[str, Any],
    episodes: Sequence[Any] | None,
    currency: str,
) -> list[Insight]:
    """Drawdown, volatility and tail-risk rules."""
    value = _num(summary, "current_value")
    out: list[Insight] = []
    for item in (
        _max_drawdown_insight(risk),
        _volatility_insight(risk),
        _var_insight(risk, value, currency),
        _cvar_insight(risk, value, currency),
        _fat_tail_insight(risk),
        _skew_insight(risk),
        _ulcer_insight(risk),
        _consistency_insight(risk),
    ):
        if item is not None:
            out.append(item)
    out.extend(_episode_insights(episodes))
    return out


# ---------------------------------------------------------------------------
# rules: performance
# ---------------------------------------------------------------------------
def _total_return_insight(
    summary: Mapping[str, Any], risk: Mapping[str, Any], currency: str
) -> Insight | None:
    """Money-weighted P&L on the book against the zero-return reference."""
    total_pct = _num(summary, "total_return_pct")
    if not np.isfinite(total_pct):
        return None

    pnl = _num(summary, "total_pnl")
    invested = _num(summary, "total_investment")
    ann = _num(risk, "ann_return")
    ann_text = (
        f" Annualised over the price window that is {ann * 100:+.1f}%." if np.isfinite(ann) else ""
    )
    return Insight(
        level="positive" if total_pct > 0 else "warning",
        text=(
            f"The book is {total_pct:+.2f}% on cost: {_money(pnl, currency)} on "
            f"{_plain_money(invested, currency)} invested.{ann_text}"
        ),
        category="performance",
        evidence=(
            f"total_return_pct={total_pct:+.2f}% vs the 0.00% break-even "
            f"reference; total_pnl={_money(pnl, currency)}, "
            f"annualised={_sfmt(ann * 100, 2)}%"
        ),
    )


def _performer_insights(summary: Mapping[str, Any], n_holdings: int) -> list[Insight]:
    """Best and worst position by percentage return.

    Silent for a single-position book, where "best" and "worst" are the same
    row and the total-return rule has already said it.
    """
    if n_holdings < 2:
        return []
    out: list[Insight] = []
    best_ticker, best_pct = _labelled(summary, "best")
    worst_ticker, worst_pct = _labelled(summary, "worst")

    if best_ticker is not None and np.isfinite(best_pct):
        out.append(
            Insight(
                level="positive" if best_pct > 0 else "info",
                text=(f"Best position: {best_ticker} at {best_pct:+.1f}% on cost."),
                category="performance",
                evidence=(
                    f"{best_ticker} pnl_pct={best_pct:+.2f}%, the maximum across "
                    f"the book, vs the 0.00% break-even reference"
                ),
            )
        )
    if worst_ticker is not None and np.isfinite(worst_pct) and worst_ticker != best_ticker:
        out.append(
            Insight(
                level="warning" if worst_pct < 0 else "info",
                text=(
                    f"Weakest position: {worst_ticker} at {worst_pct:+.1f}% on "
                    f"cost."
                    + (
                        " A losing position is a thesis to re-test, not a number "
                        "to average down on reflexively."
                        if worst_pct < 0
                        else " Every position in the book is in profit, so this "
                        "is the laggard rather than a loser."
                    )
                ),
                category="performance",
                evidence=(
                    f"{worst_ticker} pnl_pct={worst_pct:+.2f}%, the minimum across "
                    f"the book, vs the 0.00% break-even reference"
                ),
            )
        )
    return out


def _contribution_insights(
    positions: pd.DataFrame, attribution: Any, currency: str
) -> list[Insight]:
    """Largest absolute contributor and detractor in money terms.

    Uses ``attribution["return_contribution"]`` when the attribution module has
    run (it expresses each name's P&L as a share of the portfolio's total
    return) and falls back to the raw ``pnl`` column of the positions frame.
    """
    frame: Any = None
    if isinstance(attribution, Mapping):
        candidate = attribution.get("return_contribution")
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            frame = candidate
    source = frame if frame is not None else positions
    if not isinstance(source, pd.DataFrame) or source.empty:
        return []

    out: list[Insight] = []
    top_ticker, top_pnl = _extreme_row(source, "pnl", largest=True)
    low_ticker, low_pnl = _extreme_row(source, "pnl", largest=False)
    contribution = _numeric_column(source, "contribution_pct")

    def _contribution_of(ticker: str | None) -> float:
        if ticker is None or contribution.empty or "ticker" not in source.columns:
            return float("nan")
        mask = source["ticker"].astype(str) == ticker
        picked = contribution[mask.to_numpy()].dropna()
        return float(picked.iloc[0]) if not picked.empty else float("nan")

    def _contribution_evidence(share: float) -> str:
        """Percentage-point contribution, or an honest 'unavailable'."""
        if not np.isfinite(share):
            return "contribution_pct unavailable (attribution not supplied)"
        return f"contribution_pct={share:+.2f}pp"

    if top_ticker is not None and np.isfinite(top_pnl) and top_pnl > 0:
        share = _contribution_of(top_ticker)
        share_text = (
            f" It alone adds {share:+.2f} percentage points to the portfolio's return on cost."
            if np.isfinite(share)
            else ""
        )
        out.append(
            Insight(
                level="positive",
                text=(
                    f"{top_ticker} is the largest absolute contributor at "
                    f"{_money(top_pnl, currency)}.{share_text}"
                ),
                category="performance",
                evidence=(
                    f"{top_ticker} pnl={_money(top_pnl, currency)} is the maximum "
                    f"across the book, above the 0 break-even reference; "
                    f"{_contribution_evidence(share)}"
                ),
            )
        )
    if low_ticker is not None and np.isfinite(low_pnl) and low_pnl < 0:
        share = _contribution_of(low_ticker)
        share_text = (
            f" It subtracts {abs(share):.2f} percentage points from the portfolio's return on cost."
            if np.isfinite(share)
            else ""
        )
        out.append(
            Insight(
                level="warning",
                text=(
                    f"{low_ticker} is the largest drag at {_money(low_pnl, currency)}.{share_text}"
                ),
                category="performance",
                evidence=(
                    f"{low_ticker} pnl={_money(low_pnl, currency)} is the minimum "
                    f"across the book, below the 0 break-even reference; "
                    f"{_contribution_evidence(share)}"
                ),
            )
        )
    return out


def _breadth_insight(summary: Mapping[str, Any], n_holdings: int) -> Insight | None:
    """Gainers versus losers against the 50% breadth reference."""
    gainers = _num(summary, "n_gainers")
    losers = _num(summary, "n_losers")
    if not np.isfinite(gainers) or not np.isfinite(losers) or n_holdings < 2:
        return None

    share = gainers / n_holdings * 100.0
    if share >= 50.0:
        reading = (
            "Broad breadth means the result is coming from the book, not from one or two names"
        )
    else:
        reading = (
            "Narrow breadth means a handful of names are setting the result; the "
            "portfolio return says nothing about the median position"
        )
    return Insight(
        level="info",
        text=(
            f"{int(gainers)} of {n_holdings} positions are in profit and "
            f"{int(losers)} are not ({share:.0f}% breadth). {reading}."
        ),
        category="performance",
        evidence=(
            f"n_gainers={int(gainers)} / n_holdings={n_holdings} = {share:.1f}% "
            f"vs the 50.0% even-breadth reference (n_losers={int(losers)})"
        ),
    )


def _sharpe_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Sharpe ratio against ``good_sharpe`` and against zero."""
    sharpe = _num(risk, "sharpe")
    if not np.isfinite(sharpe):
        return None

    vol = _num(risk, "ann_vol")
    vol_text = f" on {vol * 100:.1f}% annualised volatility" if np.isfinite(vol) else ""

    if sharpe >= GOOD_SHARPE:
        return Insight(
            level="positive",
            text=(
                f"Sharpe ratio {sharpe:.2f}{vol_text}: every unit of volatility "
                f"was paid for with {sharpe:.2f} units of excess return over the "
                f"risk-free rate. Above 1.0 is a genuinely well-paid risk budget."
            ),
            category="performance",
            evidence=(
                f"sharpe={sharpe:.3f} >= INSIGHTS.good_sharpe={GOOD_SHARPE:.2f}; "
                f"ann_vol={_fmt(vol * 100, 2)}%"
            ),
        )
    if sharpe < 0:
        return Insight(
            level="warning",
            text=(
                f"Sharpe ratio {sharpe:.2f}{vol_text}: the portfolio did not beat "
                f"the risk-free rate over this window, so the volatility bought "
                f"nothing. Cash would have delivered more with none of the "
                f"drawdown."
            ),
            category="performance",
            evidence=(
                f"sharpe={sharpe:.3f} < 0.00 (risk-free parity), below "
                f"INSIGHTS.good_sharpe={GOOD_SHARPE:.2f}; "
                f"ann_vol={_fmt(vol * 100, 2)}%"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Sharpe ratio {sharpe:.2f}{vol_text}: positive but short of the "
            f"{GOOD_SHARPE:.2f} mark. Sharpe over a single window is a noisy "
            f"estimate - it needs years, not months, to be a judgement."
        ),
        category="performance",
        evidence=(
            f"sharpe={sharpe:.3f} is in [0.00, "
            f"INSIGHTS.good_sharpe={GOOD_SHARPE:.2f}); "
            f"ann_vol={_fmt(vol * 100, 2)}%"
        ),
    )


def _sortino_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Sortino ratio against ``good_sortino``, quoting downside deviation."""
    sortino = _num(risk, "sortino")
    if not np.isfinite(sortino):
        return None

    downside = _num(risk, "downside_dev")
    down_text = (
        f" Downside deviation is {downside * 100:.1f}% annualised." if np.isfinite(downside) else ""
    )
    if sortino >= GOOD_SORTINO:
        return Insight(
            level="positive",
            text=(
                f"Sortino ratio {sortino:.2f}: measured against downside "
                f"deviation alone - the only volatility an investor actually "
                f"minds - the portfolio was well compensated.{down_text}"
            ),
            category="performance",
            evidence=(
                f"sortino={sortino:.3f} >= "
                f"INSIGHTS.good_sortino={GOOD_SORTINO:.2f}; "
                f"downside_dev={_fmt(downside * 100, 2)}%"
            ),
        )
    return Insight(
        level="warning" if sortino < 0 else "info",
        text=(
            f"Sortino ratio {sortino:.2f}, below the {GOOD_SORTINO:.2f} mark. "
            f"Sortino penalises only downside volatility, so falling short here "
            f"means the losses, not the swings, were the problem.{down_text}"
        ),
        category="performance",
        evidence=(
            f"sortino={sortino:.3f} < INSIGHTS.good_sortino={GOOD_SORTINO:.2f}; "
            f"downside_dev={_fmt(downside * 100, 2)}%"
        ),
    )


def _calmar_insight(risk: Mapping[str, Any]) -> Insight | None:
    """Calmar ratio against ``weak_calmar``: return earned per unit of drawdown."""
    calmar = _num(risk, "calmar")
    if not np.isfinite(calmar):
        return None

    mdd = _num(risk, "mdd")
    ann = _num(risk, "ann_return")
    detail = (
        f" ({_sfmt(ann * 100, 1)}% annualised against a {abs(mdd) * 100:.1f}% worst drawdown)"
        if np.isfinite(ann) and np.isfinite(mdd)
        else ""
    )
    if calmar < WEAK_CALMAR:
        return Insight(
            level="warning",
            text=(
                f"Calmar ratio {calmar:.2f}{detail}: less than "
                f"{WEAK_CALMAR:.2f} units of annualised return per unit of "
                f"maximum drawdown. The pain endured was not repaid by the "
                f"return earned."
            ),
            category="performance",
            evidence=(
                f"calmar={calmar:.3f} < INSIGHTS.weak_calmar={WEAK_CALMAR:.2f}; "
                f"ann_return={_sfmt(ann * 100, 2)}%, "
                f"max_drawdown={_fmt(mdd * 100, 2)}%"
            ),
        )
    return Insight(
        level="positive",
        text=(
            f"Calmar ratio {calmar:.2f}{detail}: each unit of maximum drawdown "
            f"bought {calmar:.2f} units of annualised return, clearing the "
            f"{WEAK_CALMAR:.2f} floor."
        ),
        category="performance",
        evidence=(
            f"calmar={calmar:.3f} >= INSIGHTS.weak_calmar={WEAK_CALMAR:.2f}; "
            f"ann_return={_sfmt(ann * 100, 2)}%, "
            f"max_drawdown={_fmt(mdd * 100, 2)}%"
        ),
    )


def _performance_insights(
    positions: pd.DataFrame,
    summary: Mapping[str, Any],
    risk: Mapping[str, Any],
    attribution: Any,
    currency: str,
) -> list[Insight]:
    """P&L, contribution and risk-adjusted quality rules."""
    n_holdings = _as_int(summary.get("n_holdings"), len(positions))
    out: list[Insight] = []
    total = _total_return_insight(summary, risk, currency)
    if total is not None:
        out.append(total)
    out.extend(_performer_insights(summary, n_holdings))
    out.extend(_contribution_insights(positions, attribution, currency))
    for item in (
        _sharpe_insight(risk),
        _sortino_insight(risk),
        _calmar_insight(risk),
        _breadth_insight(summary, n_holdings),
    ):
        if item is not None:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# rules: benchmark-relative
# ---------------------------------------------------------------------------
def _excess_return_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Annualised excess return against benchmark parity (zero)."""
    excess = _num(bench, "excess_return")
    if not np.isfinite(excess):
        return None

    port = _num(bench, "portfolio_return")
    index = _num(bench, "benchmark_return")
    detail = (
        f" ({port * 100:+.1f}% versus {index * 100:+.1f}%)"
        if np.isfinite(port) and np.isfinite(index)
        else ""
    )
    beat = excess > 0
    return Insight(
        level="positive" if beat else "info",
        text=(
            f"The portfolio {'outperformed' if beat else 'underperformed'} the "
            f"benchmark by {abs(excess) * 100:.2f} percentage points annualised"
            f"{detail}. One window is an observation, not a track record."
        ),
        category="benchmark",
        evidence=(
            f"excess_return={excess * 100:+.2f}pp vs the 0.00pp benchmark-parity "
            f"reference; portfolio={_sfmt(port * 100, 2)}%, "
            f"benchmark={_sfmt(index * 100, 2)}%"
        ),
    )


def _relative_volatility_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Portfolio volatility against the benchmark's, inside ``vol_tolerance``."""
    port_vol = _num(bench, "portfolio_vol")
    index_vol = _num(bench, "benchmark_vol")
    if not np.isfinite(port_vol) or not np.isfinite(index_vol) or index_vol <= 0:
        return None

    ratio = port_vol / index_vol
    band = VOL_TOLERANCE * 100.0
    if ratio > 1.0 + VOL_TOLERANCE:
        return Insight(
            level="warning",
            text=(
                f"Portfolio volatility {port_vol * 100:.1f}% is {(ratio - 1) * 100:.0f}% "
                f"above the benchmark's {index_vol * 100:.1f}%. Extra volatility "
                f"is only worth carrying if it comes with extra return - check "
                f"that against the excess-return line."
            ),
            category="benchmark",
            evidence=(
                f"portfolio_vol/benchmark_vol={ratio:.3f} > "
                f"{1 + VOL_TOLERANCE:.2f} (1 + "
                f"INSIGHTS.vol_tolerance={VOL_TOLERANCE:.2f}, "
                f"a +/-{band:.0f}% band)"
            ),
        )
    if ratio < 1.0 - VOL_TOLERANCE:
        return Insight(
            level="positive",
            text=(
                f"Portfolio volatility {port_vol * 100:.1f}% is "
                f"{(1 - ratio) * 100:.0f}% below the benchmark's "
                f"{index_vol * 100:.1f}% - the book rides calmer than the index "
                f"it is measured against."
            ),
            category="benchmark",
            evidence=(
                f"portfolio_vol/benchmark_vol={ratio:.3f} < "
                f"{1 - VOL_TOLERANCE:.2f} (1 - "
                f"INSIGHTS.vol_tolerance={VOL_TOLERANCE:.2f}, "
                f"a +/-{band:.0f}% band)"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Portfolio volatility {port_vol * 100:.1f}% sits inside the "
            f"+/-{band:.0f}% band around the benchmark's {index_vol * 100:.1f}% - "
            f"this is an index-like risk profile."
        ),
        category="benchmark",
        evidence=(
            f"portfolio_vol/benchmark_vol={ratio:.3f} inside "
            f"[{1 - VOL_TOLERANCE:.2f}, {1 + VOL_TOLERANCE:.2f}] "
            f"(1 +/- INSIGHTS.vol_tolerance={VOL_TOLERANCE:.2f})"
        ),
    )


def _beta_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Beta posture against ``aggressive_beta`` / ``defensive_beta``."""
    beta = _num(bench, "beta")
    if not np.isfinite(beta):
        return None

    r2 = _as_ratio(_num(bench, "r_squared"))
    caveat = (
        f" With an R-squared of {r2:.2f} the benchmark explains little of the "
        f"portfolio's variance, so read this beta loosely."
        if np.isfinite(r2) and r2 < MEANINGFUL_R_SQUARED
        else ""
    )
    if beta > AGGRESSIVE_BETA:
        return Insight(
            level="info",
            text=(
                f"Beta {beta:.2f}: a 10% benchmark move maps to roughly "
                f"{beta * 10:.1f}% here. The portfolio amplifies the market in "
                f"both directions.{caveat}"
            ),
            category="benchmark",
            evidence=(
                f"beta={beta:.3f} > INSIGHTS.aggressive_beta={AGGRESSIVE_BETA:.2f}; "
                f"r_squared={_fmt(r2, 3)}"
            ),
        )
    if beta < DEFENSIVE_BETA:
        return Insight(
            level="info",
            text=(
                f"Beta {beta:.2f}: a 10% benchmark move maps to roughly "
                f"{beta * 10:.1f}% here. The portfolio is defensive relative to "
                f"the index - it should lag rallies and cushion falls.{caveat}"
            ),
            category="benchmark",
            evidence=(
                f"beta={beta:.3f} < INSIGHTS.defensive_beta={DEFENSIVE_BETA:.2f}; "
                f"r_squared={_fmt(r2, 3)}"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Beta {beta:.2f} sits between the {DEFENSIVE_BETA:.2f} defensive and "
            f"{AGGRESSIVE_BETA:.2f} aggressive lines - the portfolio tracks the "
            f"benchmark's market exposure closely.{caveat}"
        ),
        category="benchmark",
        evidence=(
            f"beta={beta:.3f} is in "
            f"[INSIGHTS.defensive_beta={DEFENSIVE_BETA:.2f}, "
            f"INSIGHTS.aggressive_beta={AGGRESSIVE_BETA:.2f}]; "
            f"r_squared={_fmt(r2, 3)}"
        ),
    )


def _alpha_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Alpha against zero, gated on the regression's explanatory power."""
    alpha = _num(bench, "alpha")
    if not np.isfinite(alpha):
        return None

    r2 = _as_ratio(_num(bench, "r_squared"))
    method = str(bench.get("alpha_methodology") or "").strip()
    method_text = f" Method: {method.splitlines()[0]}" if method else ""

    if np.isfinite(r2) and r2 < MEANINGFUL_R_SQUARED:
        return Insight(
            level="info",
            text=(
                f"Alpha is {alpha * 100:+.2f}% annualised, but the benchmark "
                f"explains only {r2 * 100:.0f}% of the portfolio's variance. "
                f"Below {MEANINGFUL_R_SQUARED * 100:.0f}% explanatory power, "
                f"alpha and beta are describing a relationship that mostly is not "
                f"there.{method_text}"
            ),
            category="benchmark",
            evidence=(
                f"alpha={alpha * 100:+.2f}%, r_squared={r2:.3f} < "
                f"MEANINGFUL_R_SQUARED={MEANINGFUL_R_SQUARED:.2f}"
            ),
        )
    return Insight(
        level="positive" if alpha > 0 else "info",
        text=(
            f"Alpha is {alpha * 100:+.2f}% annualised - the return left over "
            f"after paying for the market exposure the portfolio actually took."
            f"{method_text}"
        ),
        category="benchmark",
        evidence=(
            f"alpha={alpha * 100:+.2f}% vs the 0.00% no-alpha reference; "
            f"r_squared={_fmt(r2, 3)} >= "
            f"MEANINGFUL_R_SQUARED={MEANINGFUL_R_SQUARED:.2f}"
        ),
    )


def _tracking_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Tracking error and information ratio against the zero-IR reference."""
    te = _num(bench, "tracking_error")
    ir = _num(bench, "information_ratio")
    if not np.isfinite(te) and not np.isfinite(ir):
        return None

    te_pct = te * 100.0 if np.isfinite(te) else float("nan")
    if np.isfinite(ir) and ir > 0:
        level = "positive"
        verdict = f"each unit of active risk returned {ir:.2f} units of active return"
    elif np.isfinite(ir):
        level = "warning"
        verdict = (
            "the active risk taken versus the benchmark was not rewarded - a "
            "cheaper index fund would have done better"
        )
    else:
        level = "info"
        verdict = "the information ratio is undefined for this window"

    return Insight(
        level=level,
        text=(
            f"Tracking error {_fmt(te_pct, 2)}% annualised with an information "
            f"ratio of {_fmt(ir, 2)}: {verdict}. Tracking error is how far the "
            f"portfolio is allowed to wander from the index; the information "
            f"ratio is whether the wandering paid."
        ),
        category="benchmark",
        evidence=(
            f"tracking_error={_fmt(te_pct, 2)}%, "
            f"information_ratio={_fmt(ir, 3)} vs the 0.000 no-skill reference"
        ),
    )


def _capture_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Up/down capture against 100% parity with the benchmark."""
    up = _as_ratio(_num(bench, "up_capture"))
    down = _as_ratio(_num(bench, "down_capture"))
    if not np.isfinite(up) or not np.isfinite(down):
        return None

    up_pct = up * 100.0
    down_pct = down * 100.0
    ratio = _num(bench, "capture_ratio")
    if not np.isfinite(ratio) and down != 0:
        ratio = up / down
    up_months = _num(bench, "up_months")
    down_months = _num(bench, "down_months")
    months_text = (
        f" Measured over {_fmt(up_months, 0)} up and {_fmt(down_months, 0)} down benchmark months."
        if np.isfinite(up_months) and np.isfinite(down_months)
        else ""
    )

    if up_pct >= 100.0 and down_pct <= 100.0:
        level = "positive"
        verdict = (
            "the portfolio keeps more than all of the upside and less than all of "
            "the downside, which is the whole point of active management"
        )
    elif np.isfinite(ratio) and ratio > 1.0:
        level = "positive"
        verdict = (
            "the asymmetry runs the right way - the portfolio gives up some of "
            "the rallies but takes proportionally less of the falls"
        )
    elif down_pct > up_pct:
        level = "warning"
        verdict = (
            "the portfolio takes more of the falls than it keeps of the rallies - "
            "the asymmetry runs the wrong way"
        )
    else:
        level = "info"
        verdict = "the capture profile is broadly index-like"

    return Insight(
        level=level,
        text=(
            f"Up-capture {up_pct:.0f}% and down-capture {down_pct:.0f}% "
            f"(ratio {_fmt(ratio, 2)}): {verdict}.{months_text}"
        ),
        category="benchmark",
        evidence=(
            f"up_capture={up_pct:.1f}% and down_capture={down_pct:.1f}% vs the "
            f"100.0% benchmark-parity reference; capture_ratio={_fmt(ratio, 2)} "
            f"vs 1.00 parity"
        ),
    )


def _relative_drawdown_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Portfolio maximum drawdown against the benchmark's."""
    port_dd = _num(bench, "portfolio_mdd")
    index_dd = _num(bench, "benchmark_mdd")
    if not np.isfinite(port_dd) or not np.isfinite(index_dd):
        return None

    gap = (abs(port_dd) - abs(index_dd)) * 100.0
    worse = gap > 0
    return Insight(
        level="warning" if worse else "positive",
        text=(
            f"Worst drawdown {abs(port_dd) * 100:.1f}% against the benchmark's "
            f"{abs(index_dd) * 100:.1f}% - {abs(gap):.1f} percentage points "
            f"{'deeper' if worse else 'shallower'}. Drawdown, not volatility, is "
            f"what makes an investor sell at the bottom."
        ),
        category="benchmark",
        evidence=(
            f"portfolio_mdd={port_dd * 100:.2f}% vs "
            f"benchmark_mdd={index_dd * 100:.2f}% (0.00pp parity reference); "
            f"gap={gap:+.2f}pp"
        ),
    )


def _m2_insight(bench: Mapping[str, Any]) -> Insight | None:
    """Modigliani M2: portfolio return re-levered to the benchmark's volatility."""
    m2 = _num(bench, "m2")
    index = _num(bench, "benchmark_return")
    if not np.isfinite(m2) or not np.isfinite(index):
        return None

    gap = (m2 - index) * 100.0
    return Insight(
        level="positive" if gap > 0 else "info",
        text=(
            f"Risk-adjusted to the benchmark's own volatility (M2), the "
            f"portfolio is worth {m2 * 100:+.2f}% annualised against the index's "
            f"{index * 100:+.2f}% - a {gap:+.2f} percentage point difference. M2 "
            f"is the Sharpe ratio restated in return units, so it can be compared "
            f"like a return."
        ),
        category="benchmark",
        evidence=(
            f"m2={m2 * 100:+.2f}% vs benchmark_return={index * 100:+.2f}% "
            f"(0.00pp parity reference); gap={gap:+.2f}pp"
        ),
    )


def _benchmark_insights(bench: Any) -> list[Insight]:
    """Every benchmark-relative rule; silent when no comparison ran."""
    if not isinstance(bench, Mapping) or not bench:
        return []
    if not bench.get("ok"):
        reason = str(bench.get("reason") or "no overlapping history").strip()
        return [
            Insight(
                level="info",
                text=(
                    f"No benchmark comparison was possible: {reason}. Beta, alpha "
                    f"and capture ratios are therefore unavailable for this run."
                ),
                category="benchmark",
                evidence=f"benchmark comparison ok=False, reason='{reason}'",
            )
        ]

    candidates = (
        _excess_return_insight(bench),
        _relative_volatility_insight(bench),
        _relative_drawdown_insight(bench),
        _tracking_insight(bench),
        _capture_insight(bench),
        _beta_insight(bench),
        _alpha_insight(bench),
        _m2_insight(bench),
    )
    return [item for item in candidates if item is not None]


# ---------------------------------------------------------------------------
# rules: data sufficiency and provenance
# ---------------------------------------------------------------------------
def _sample_insight(risk: Mapping[str, Any]) -> Insight:
    """Observation count against ``config.MIN_OBSERVATIONS``."""
    n_obs = _num(risk, "n_obs")
    if not np.isfinite(n_obs) or n_obs <= 1:
        return Insight(
            level="warning",
            text=(
                "No usable return history was available, so every volatility, "
                "drawdown and tail statistic on this page is undefined. The "
                "holdings table is still valid; nothing else is."
            ),
            category="data",
            evidence=(
                f"n_obs={_fmt(n_obs, 0)} < MIN_OBSERVATIONS={MIN_OBSERVATIONS} (no return series)"
            ),
        )
    if n_obs < MIN_OBSERVATIONS:
        return Insight(
            level="warning",
            text=(
                f"Risk statistics rest on only {n_obs:.0f} daily observations. "
                f"Below {MIN_OBSERVATIONS} the standard error on volatility is "
                f"roughly 1/sqrt(2n) = {1 / np.sqrt(2 * n_obs) * 100:.0f}% of the "
                f"estimate itself, and a 95% VaR expects fewer than "
                f"{n_obs * 0.05:.1f} breaches - treat every number here as "
                f"indicative."
            ),
            category="data",
            evidence=(
                f"n_obs={n_obs:.0f} < MIN_OBSERVATIONS={MIN_OBSERVATIONS}; "
                f"relative standard error of sigma "
                f"~{1 / np.sqrt(2 * n_obs) * 100:.1f}%"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Statistics are computed on {n_obs:.0f} overlapping daily "
            f"observations (about {n_obs / _DAYS_PER_MONTH:.0f} months). That clears "
            f"the "
            f"{MIN_OBSERVATIONS}-observation reliability floor, though annualised "
            f"figures extrapolated from a short window remain noisy."
        ),
        category="data",
        evidence=(
            f"n_obs={n_obs:.0f} >= MIN_OBSERVATIONS={MIN_OBSERVATIONS}; "
            f"relative standard error of sigma "
            f"~{1 / np.sqrt(2 * n_obs) * 100:.1f}%"
        ),
    )


def _currency_insight(summary: Mapping[str, Any], currency: str) -> Insight | None:
    """Multi-currency exposure against the single-currency reference."""
    currencies = summary.get("currencies") if isinstance(summary, Mapping) else None
    if not isinstance(currencies, (list, tuple)) or len(currencies) < 2:
        return None

    listed = ", ".join(str(c) for c in currencies)
    base = currency or "the base currency"
    return Insight(
        level="info",
        text=(
            f"Holdings are quoted in {len(currencies)} currencies ({listed}) and "
            f"translated into {base}. Part of the reported P&L is therefore FX, "
            f"not stock selection, and the FX leg carries its own volatility."
        ),
        category="data",
        evidence=(
            f"distinct quote currencies={len(currencies)} ({listed}) > 1 "
            f"(single-currency reference); base_currency={base}"
        ),
    )


def _unclassified_insight(summary: Mapping[str, Any]) -> Insight | None:
    """Share of the book with no sector label from the data provider."""
    sectors = summary.get("sectors") if isinstance(summary, Mapping) else None
    if not isinstance(sectors, Mapping) or not sectors:
        return None

    unknown = sum(
        _as_float(weight)
        for name, weight in sectors.items()
        if str(name).strip().lower() in _UNKNOWN_SECTORS and np.isfinite(_as_float(weight))
    )
    if unknown < SECTOR_CONCENTRATION_PCT:
        return None
    return Insight(
        level="info",
        text=(
            f"{unknown:.1f}% of portfolio value has no sector classification from "
            f"the data provider (typical for ETFs, indices and smaller listings). "
            f"Sector concentration cannot be assessed for that slice - the "
            f"absence of a label is not the absence of a sector bet."
        ),
        category="data",
        evidence=(
            f"unclassified weight={unknown:.1f}% >= "
            f"INSIGHTS.sector_concentration_pct={SECTOR_CONCENTRATION_PCT:.1f}%; "
            f"sector concentration rule suppressed for that share"
        ),
    )


def _data_insights(
    summary: Mapping[str, Any], risk: Mapping[str, Any], currency: str
) -> list[Insight]:
    """Sample size, currency mix and classification-coverage rules."""
    out: list[Insight] = [_sample_insight(risk)]
    for item in (_currency_insight(summary, currency), _unclassified_insight(summary)):
        if item is not None:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def generate_insights(
    positions: pd.DataFrame | None,
    summary: Mapping[str, Any] | None,
    risk: Mapping[str, Any] | None,
    bench: Mapping[str, Any] | None,
    corr: pd.DataFrame | None,
    episodes: Sequence[Any] | None = None,
    attribution: Mapping[str, Any] | None = None,
) -> list[Insight]:
    """Turn one analysis run into ordered, sourced, plain-English observations.

    Rules fired, grouped by the ``category`` they carry:

    * ``concentration`` -- largest single name vs ``concentration_warn_pct`` /
      ``concentration_info_pct``; heaviest sector vs
      ``sector_concentration_pct``; top-five weight vs the equal-weight
      reference ``5/N``; any name whose risk contribution exceeds its weight by
      ``RISK_WEIGHT_RATIO``.
    * ``diversification`` -- position count vs ``min_diversified_holdings``;
      effective bets (1/HHI) vs ``effective_holdings_ratio``; pairwise
      correlations vs ``high_correlation``; average pairwise correlation.
    * ``risk`` -- max drawdown vs ``severe_drawdown`` / ``deep_drawdown`` with
      the break-even gain it implies; annualised volatility vs
      ``RISK_RUBRIC.vol_cap``; one-day VaR and expected shortfall vs
      ``high_var_pct``; excess kurtosis vs ``fat_tail_kurtosis`` priced through
      the Gaussian/Cornish-Fisher VaR gap; skew; ulcer index; hit ratio and
      gain-to-pain; drawdown-episode recovery vs ``long_recovery_days``.
    * ``performance`` -- P&L on cost, best/worst position, largest contributor
      and detractor, Sharpe vs ``good_sharpe``, Sortino vs ``good_sortino``,
      Calmar vs ``weak_calmar``, gainer/loser breadth.
    * ``benchmark`` -- excess return, relative volatility inside
      ``vol_tolerance``, relative drawdown, tracking error and information
      ratio, up/down capture, beta vs ``aggressive_beta`` / ``defensive_beta``,
      alpha gated on ``MEANINGFUL_R_SQUARED``, and the Modigliani M2 measure.
    * ``data`` -- observation count vs ``config.MIN_OBSERVATIONS``, multi-
      currency exposure, and the share of the book the provider left
      unclassified.

    Ordering is ``critical -> warning -> positive -> info`` via a STABLE sort,
    so within one severity the rules appear in the deliberate order above.
    Every returned Insight has a non-empty ``evidence`` string naming the
    observed value and the threshold that fired it.

    Degenerate inputs never raise: a missing/empty ``positions`` frame returns
    exactly one warning, and any other missing argument (``None``, ``{}``, an
    empty frame, an all-NaN statistic) simply disables the rules that needed
    it. No argument is mutated.
    """
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return [
            Insight(
                level="warning",
                text=(
                    "No valid positions to analyse. Add holdings with a ticker, a "
                    "positive quantity and a positive buy price, or check the "
                    "warnings from the data fetch - every statistic on this page "
                    "is derived from the positions table."
                ),
                category="data",
                evidence=(
                    "n_holdings=0 vs the 1-holding minimum required for any portfolio statistic"
                ),
            )
        ]

    facts: Mapping[str, Any] = summary if isinstance(summary, Mapping) else {}
    stats: Mapping[str, Any] = risk if isinstance(risk, Mapping) else {}
    currency = str(facts.get("base_currency") or positions.attrs.get("base_currency") or "").strip()

    found: list[Insight] = []
    found.extend(_concentration_insights(positions, facts, attribution, currency))
    found.extend(_diversification_insights(positions, facts, stats, corr))
    found.extend(_risk_insights(stats, facts, episodes, currency))
    found.extend(_performance_insights(positions, facts, stats, attribution, currency))
    found.extend(_benchmark_insights(bench))
    found.extend(_data_insights(facts, stats, currency))

    return sorted(found, key=lambda ins: _LEVEL_RANK.get(ins.level, len(_LEVEL_RANK)))


def insights_by_category(
    insights: Sequence[Insight] | None,
) -> dict[str, list[Insight]]:
    """Group insights by ``category``, preserving their severity ordering.

    Keys appear in ``CATEGORIES`` order, followed by any unexpected category in
    first-seen order; empty categories are omitted entirely so the UI can
    iterate the result directly. Returns ``{}`` for an empty or missing list.
    """
    if not insights:
        return {}

    grouped: dict[str, list[Insight]] = {}
    for item in insights:
        category = getattr(item, "category", "") or "general"
        grouped.setdefault(str(category), []).append(item)

    ordered: dict[str, list[Insight]] = {}
    for category in CATEGORIES:
        if category in grouped:
            ordered[category] = grouped.pop(category)
    ordered.update(grouped)
    return ordered


def headline_insight(insights: Sequence[Insight] | None) -> Insight | None:
    """The single most severe insight, for the one-line banner at the top.

    Severity order is ``critical -> warning -> positive -> info``; ties are
    broken by position in the list, so the first rule to fire at the top
    severity wins. Returns ``None`` for an empty or missing list, and works on
    an unsorted list as well as on the output of :func:`generate_insights`.
    """
    if not insights:
        return None
    ranked = [
        (_LEVEL_RANK.get(getattr(item, "level", ""), len(_LEVEL_RANK)), i, item)
        for i, item in enumerate(insights)
    ]
    return min(ranked, key=lambda entry: (entry[0], entry[1]))[2]
