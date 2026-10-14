"""Rule-based portfolio insights — ``src.analytics.insights``.

This module is the last thing between a number and a sentence a user acts on,
so the failure mode that matters is not an exception: it is a rule that fires
on the wrong side of its threshold, or a sentence with no auditable number
behind it. The tests are built accordingly.

Two portfolios are constructed and then driven through the *real* upstream
functions — ``positions.build_positions``, ``positions.summarise`` and
``risk.risk_summary`` — rather than through hand-written dicts, so the input
contract this module documents (which keys, in which units) is exercised end to
end rather than assumed:

* a **deliberately broken book**: 70% in one name, every holding in one sector,
  and a value series carrying seven consecutive -25% sessions. Its risk
  statistics cross ``severe_drawdown``, ``RISK_RUBRIC.vol_cap x 1.5``,
  ``high_var_pct x 1.5`` and ``fat_tail_kurtosis``, and the assertions check
  BOTH that the statistic crossed the line AND that the sentence appeared;
* a **diversified, low-volatility book**: twelve equal-weighted names across
  six sectors on a 9.6%-volatility series. It must produce no critical and no
  warning insight at all — the negative control that stops "everything is a
  warning" from passing the tests above.

Also verified
-------------
* the ordering contract ``critical -> warning -> positive -> info``, and that
  the sort is STABLE so the deliberate emission order inside one severity
  (concentration, diversification, risk, performance, benchmark, data) survives;
* the audit contract: EVERY insight carries a non-empty ``evidence`` string and
  a ``category`` drawn from the published ``CATEGORIES`` vocabulary;
* ``_high_correlation_pairs`` — published API despite the underscore, because
  the correlation page calls it with ``threshold=-1.0`` — returns each of the
  ``n(n-1)/2`` unique pairs exactly once, strongest first, and the -1.0 filter
  genuinely cannot bind because Pearson correlation is bounded below by -1;
* hand-computable threshold arithmetic: the escalation factor (a 70% position
  is critical because ``70 >= 40 x 1.5``), the equal-weight top-five reference
  ``5/N``, the break-even gain ``1/(1+d) - 1`` quoted with every drawdown, and
  the risk-versus-weight ratio ``90/70 = 1.29x``;
* documented degradation: an empty portfolio returns exactly one warning; a
  missing ``risk``/``bench``/``corr``/``episodes`` argument silently disables
  the rules that needed it instead of raising or inventing a number; no
  argument is mutated.

Deterministic and offline: the two value series come from a seeded generator,
everything else is arithmetic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import SEED, START, make_quote
from src.analytics.insights import (
    AGGRESSIVE_BETA,
    CATEGORIES,
    CONCENTRATION_INFO_PCT,
    CONCENTRATION_WARN_PCT,
    CRITICAL_ESCALATION,
    DEEP_DRAWDOWN,
    EFFECTIVE_HOLDINGS_RATIO,
    FAT_TAIL_KURTOSIS,
    GOOD_SHARPE,
    HIGH_CORRELATION,
    HIGH_VAR_PCT,
    HIGH_VOLATILITY,
    LONG_RECOVERY_DAYS,
    MEANINGFUL_R_SQUARED,
    MIN_DIVERSIFIED_HOLDINGS,
    MIN_OBSERVATIONS,
    RISK_WEIGHT_RATIO,
    SECTOR_CONCENTRATION_PCT,
    SEVERE_DRAWDOWN,
    _high_correlation_pairs,
    generate_insights,
    headline_insight,
    insights_by_category,
)
from src.analytics.positions import build_positions, summarise
from src.analytics.risk import risk_summary
from src.models import DrawdownEpisode, Insight, SecurityProfile

#: Display order of severities, restated here so the ordering test does not
#: read its expectation out of the module under test.
LEVEL_ORDER: tuple[str, ...] = ("critical", "warning", "positive", "info")


# ---------------------------------------------------------------------------
# local builders
# ---------------------------------------------------------------------------
def value_series(
    sigma: float,
    drift: float,
    days: int,
    seed: int = SEED,
    crash: tuple[int, int] | None = None,
    crash_return: float = 0.0,
) -> pd.Series:
    """A seeded portfolio value path, optionally with a spliced-in crash.

    ``crash`` replaces a slice of the daily returns with ``crash_return``, which
    makes the resulting drawdown hand-boundable: ``k`` consecutive sessions at
    ``c`` guarantee a fall of at least ``(1 + c)**k - 1`` regardless of what
    the surrounding noise does.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, sigma, days)
    if crash is not None:
        returns[crash[0] : crash[1]] = crash_return
    index = pd.bdate_range(START, periods=days + 1)
    return pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + returns]), index=index)


def positions_frame(
    tickers: list[str],
    quantity: list[float],
    buy_price: list[float],
    price: list[float],
    sectors: list[str | None],
) -> pd.DataFrame:
    """A real ``build_positions`` frame, so the tests use the real contract."""
    holdings = pd.DataFrame({"ticker": tickers, "quantity": quantity, "buy_price": buy_price})
    quotes = {t: make_quote(t, p) for t, p in zip(tickers, price, strict=True)}
    profiles = {
        t: SecurityProfile(
            t, name=t, sector=s, industry="Industry", currency="INR", quote_type="EQUITY"
        )
        for t, s in zip(tickers, sectors, strict=True)
    }
    frame, skipped = build_positions(holdings, quotes, profiles)
    assert skipped == []
    return frame


def correlation(labels: list[str], off_diagonal: float) -> pd.DataFrame:
    """A square correlation frame with one constant off-diagonal value."""
    n = len(labels)
    values = np.full((n, n), float(off_diagonal))
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=labels, columns=labels)


def pick(
    insights: list[Insight],
    *,
    level: str | None = None,
    category: str | None = None,
    contains: str | None = None,
) -> list[Insight]:
    """Every insight matching the given level / category / substring filter."""
    return [
        item
        for item in insights
        if (level is None or item.level == level)
        and (category is None or item.category == category)
        and (contains is None or contains in item.text or contains in item.evidence)
    ]


# ---------------------------------------------------------------------------
# the two portfolios
# ---------------------------------------------------------------------------
CONCENTRATED_TICKERS = ["MEGA", "N1", "N2", "N3", "N4", "N5"]


@pytest.fixture
def concentrated_positions() -> pd.DataFrame:
    """One name at 70% of value, five at 6%, every one of them in Technology.

    MEGA  700 @ 100 = 70,000   bought at  40  ->  +150.00%
    N1     60 @ 100 =  6,000   bought at 125  ->   -20.00%
    N2     60 @ 100 =  6,000   bought at 110  ->    -9.09%
    N3     60 @ 100 =  6,000   bought at 100  ->     0.00%
    N4     60 @ 100 =  6,000   bought at  90  ->   +11.11%
    N5     60 @ 100 =  6,000   bought at  80  ->   +25.00%
    total value 100,000, so the weights are exactly 70 / 6 / 6 / 6 / 6 / 6.
    """
    return positions_frame(
        CONCENTRATED_TICKERS,
        quantity=[700.0, 60.0, 60.0, 60.0, 60.0, 60.0],
        buy_price=[40.0, 125.0, 110.0, 100.0, 90.0, 80.0],
        price=[100.0] * 6,
        sectors=["Technology"] * 6,
    )


@pytest.fixture
def crashed_values() -> pd.Series:
    """300 sessions at 2.5% daily vol with seven consecutive -25% days."""
    return value_series(sigma=0.025, drift=0.0004, days=300, crash=(150, 157), crash_return=-0.25)


@pytest.fixture
def concentrated_risk(crashed_values: pd.Series) -> dict:
    return risk_summary(
        crashed_values,
        weights_pct=[70.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        n_holdings=6,
        risk_free_rate=0.06,
    )


@pytest.fixture
def concentrated_insights(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
) -> list[Insight]:
    return generate_insights(
        concentrated_positions,
        summarise(concentrated_positions),
        concentrated_risk,
        None,
        correlation(CONCENTRATED_TICKERS, 0.95),
    )


DIVERSIFIED_TICKERS = [f"D{i:02d}" for i in range(1, 13)]
DIVERSIFIED_SECTORS = [
    "Energy",
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer",
    "Industrials",
]


@pytest.fixture
def diversified_positions() -> pd.DataFrame:
    """Twelve names of exactly 10,000 each across six sectors, all in profit.

    Every holding is 100 shares now worth 100, so every weight is 100/12 =
    8.3333% and every sector is 2/12 = 16.6667% of value. Cost runs from 80 to
    91, so the book is between +9.89% (D12) and +25.00% (D01) on cost.
    """
    return positions_frame(
        DIVERSIFIED_TICKERS,
        quantity=[100.0] * 12,
        buy_price=[80.0 + i for i in range(12)],
        price=[100.0] * 12,
        sectors=[DIVERSIFIED_SECTORS[i % 6] for i in range(12)],
    )


@pytest.fixture
def calm_values() -> pd.Series:
    """500 sessions at 0.6% daily volatility with a small positive drift."""
    return value_series(sigma=0.006, drift=0.0007, days=500)


@pytest.fixture
def diversified_risk(calm_values: pd.Series) -> dict:
    return risk_summary(
        calm_values,
        weights_pct=[100.0 / 12.0] * 12,
        n_holdings=12,
        risk_free_rate=0.06,
    )


@pytest.fixture
def diversified_insights(
    diversified_positions: pd.DataFrame, diversified_risk: dict
) -> list[Insight]:
    return generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        correlation(DIVERSIFIED_TICKERS, 0.15),
    )


# ---- _high_correlation_pairs ----------------------------------------------
FOUR_ASSET_CORRELATIONS = {
    ("A", "B"): 0.90,
    ("A", "C"): -0.60,
    ("A", "D"): 0.50,
    ("B", "C"): 0.30,
    ("B", "D"): 0.70,
    ("C", "D"): 0.20,
}


@pytest.fixture
def four_asset_corr() -> pd.DataFrame:
    """A 4x4 correlation matrix with six distinct off-diagonal values.

    Distinct on purpose: a matrix of one repeated value cannot tell a correct
    ordering from an accidental one.
    """
    labels = ["A", "B", "C", "D"]
    frame = pd.DataFrame(np.eye(4), index=labels, columns=labels)
    for (a, b), rho in FOUR_ASSET_CORRELATIONS.items():
        frame.loc[a, b] = rho
        frame.loc[b, a] = rho
    return frame


def test_a_threshold_of_minus_one_returns_every_unique_pair_exactly_once(
    four_asset_corr: pd.DataFrame,
):
    """Pearson correlation is bounded below by -1, so the filter cannot bind.

    Four assets have 4 x 3 / 2 = 6 unordered pairs. Each must appear once — not
    twice (both triangles) and not seven times (with a self-pair).
    """
    pairs = _high_correlation_pairs(four_asset_corr, -1.0)

    n = four_asset_corr.shape[0]
    assert len(pairs) == n * (n - 1) // 2 == 6
    keys = [tuple(sorted((a, b))) for a, b, _ in pairs]
    assert len(set(keys)) == len(keys)
    assert set(keys) == set(FOUR_ASSET_CORRELATIONS)
    assert all(a != b for a, b, _ in pairs)


def test_pairs_come_back_sorted_strongest_first(four_asset_corr: pd.DataFrame):
    """0.90, 0.70, 0.50, 0.30, 0.20, -0.60 — including the negative one last."""
    pairs = _high_correlation_pairs(four_asset_corr, -1.0)

    assert [(a, b) for a, b, _ in pairs] == [
        ("A", "B"),
        ("B", "D"),
        ("A", "D"),
        ("B", "C"),
        ("C", "D"),
        ("A", "C"),
    ]
    rhos = [rho for _, _, rho in pairs]
    assert rhos == pytest.approx([0.90, 0.70, 0.50, 0.30, 0.20, -0.60])
    assert rhos == sorted(rhos, reverse=True)


def test_a_real_threshold_keeps_only_the_pairs_at_or_above_it(
    four_asset_corr: pd.DataFrame,
):
    """The filter is ``>=``, so a pair sitting exactly on 0.70 is kept."""
    assert [(a, b) for a, b, _ in _high_correlation_pairs(four_asset_corr, 0.70)] == [
        ("A", "B"),
        ("B", "D"),
    ]
    assert [(a, b) for a, b, _ in _high_correlation_pairs(four_asset_corr, HIGH_CORRELATION)] == [
        ("A", "B")
    ]
    assert _high_correlation_pairs(four_asset_corr, 0.95) == []


def test_the_pair_count_is_n_choose_two_for_every_matrix_size():
    """Identity across sizes: 2 -> 1, 3 -> 3, 5 -> 10, 12 -> 66 pairs."""
    for n in (2, 3, 5, 12):
        labels = [f"X{i}" for i in range(n)]
        pairs = _high_correlation_pairs(correlation(labels, 0.4), -1.0)
        assert len(pairs) == n * (n - 1) // 2


def test_high_correlation_pairs_never_mutates_the_matrix(four_asset_corr: pd.DataFrame):
    before = four_asset_corr.copy(deep=True)
    _high_correlation_pairs(four_asset_corr, -1.0)
    pd.testing.assert_frame_equal(four_asset_corr, before)


@pytest.mark.parametrize(
    "corr",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
        "not a frame",
    ],
)
def test_an_unusable_correlation_frame_yields_no_pairs(corr: object):
    assert _high_correlation_pairs(corr, -1.0) == []


def test_a_non_finite_threshold_yields_no_pairs(four_asset_corr: pd.DataFrame):
    """NaN >= NaN is False for every pair, so returning nothing is the only
    self-consistent answer — and it must not raise."""
    assert _high_correlation_pairs(four_asset_corr, float("nan")) == []


def test_non_finite_cells_are_dropped_rather_than_compared():
    """One failed correlation must not remove the other five pairs."""
    frame = correlation(["A", "B", "C", "D"], 0.4)
    frame.loc["A", "B"] = np.nan
    frame.loc["B", "A"] = np.nan

    pairs = _high_correlation_pairs(frame, -1.0)
    assert len(pairs) == 5
    assert ("A", "B") not in [(a, b) for a, b, _ in pairs]


# ---- the concentrated, high-volatility, deep-drawdown book ---------------
def test_the_broken_book_crosses_every_threshold_it_is_meant_to(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
):
    """The trigger conditions, checked before the sentences that depend on them.

    Seven sessions at -25% compound to 0.75**7 - 1 = -86.65%, so the drawdown
    is at least that deep whatever the surrounding noise does — the bound is
    arithmetic, not a sample.
    """
    summary = summarise(concentrated_positions)

    assert summary["largest_holding"] == ("MEGA", pytest.approx(70.0))
    assert summary["sectors"] == {"Technology": pytest.approx(100.0)}
    # HHI = 0.70^2 + 5 x 0.06^2 = 0.49 + 0.018 = 0.508;  1 / 0.508 = 1.9685
    assert summary["hhi"] == pytest.approx(0.508)
    assert summary["effective_holdings"] == pytest.approx(1.0 / 0.508)

    assert concentrated_risk["mdd"] <= 0.75**7 - 1.0 <= SEVERE_DRAWDOWN
    assert concentrated_risk["ann_vol"] >= HIGH_VOLATILITY * CRITICAL_ESCALATION
    assert abs(concentrated_risk["var"]) * 100.0 >= HIGH_VAR_PCT * CRITICAL_ESCALATION
    assert abs(concentrated_risk["cvar"]) * 100.0 >= HIGH_VAR_PCT * CRITICAL_ESCALATION
    assert concentrated_risk["kurtosis"] >= FAT_TAIL_KURTOSIS


def test_a_seventy_percent_position_is_a_critical_concentration_insight(
    concentrated_insights: list[Insight],
):
    """70 >= 40 x 1.5 = 60, so the single-name rule escalates past "warning"."""
    critical = pick(concentrated_insights, level="critical", category="concentration")
    single_name = [i for i in critical if i.text.startswith("MEGA is 70.0%")]

    assert len(single_name) == 1
    assert CONCENTRATION_WARN_PCT * CRITICAL_ESCALATION == pytest.approx(60.0)
    assert "weight=70.0% >= 60.0%" in single_name[0].evidence
    assert "concentration_warn_pct=40.0%" in single_name[0].evidence


def test_a_single_sector_book_is_a_critical_concentration_insight(
    concentrated_insights: list[Insight],
):
    """100% in Technology against a 45% threshold escalated to 67.5%."""
    sector = pick(concentrated_insights, category="concentration", contains="Technology")

    assert len(sector) == 1
    assert sector[0].level == "critical"
    assert SECTOR_CONCENTRATION_PCT * CRITICAL_ESCALATION == pytest.approx(67.5)
    assert "Technology weight=100.0% >= 67.5%" in sector[0].evidence


def test_the_deep_drawdown_is_a_critical_risk_insight_with_its_break_even_gain(
    concentrated_insights: list[Insight], concentrated_risk: dict
):
    """A drawdown of d needs 1/(1 + d) - 1 to be erased — the asymmetry is the
    single most useful thing to say about it, so it is quoted every time."""
    drawdown = pick(concentrated_insights, category="risk", contains="peak-to-trough")

    assert len(drawdown) == 1
    assert drawdown[0].level == "critical"
    depth = concentrated_risk["mdd"]
    back = (1.0 / (1.0 + depth) - 1.0) * 100.0
    assert f"break-even gain required={back:.2f}%" in drawdown[0].evidence
    assert "severe_drawdown=-35.00%" in drawdown[0].evidence


def test_the_tail_risk_rules_fire_critical_on_var_and_expected_shortfall(
    concentrated_insights: list[Insight], concentrated_risk: dict
):
    """|VaR| and |CVaR| both clear 3.0% x 1.5 = 4.5%, and CVaR must be the
    worse of the two — expected shortfall is an average BEYOND the threshold."""
    var = pick(concentrated_insights, category="risk", contains="One-day 95% VaR")
    cvar = pick(concentrated_insights, category="risk", contains="Expected shortfall")

    assert len(var) == len(cvar) == 1
    assert var[0].level == "critical"
    assert cvar[0].level == "critical"
    assert HIGH_VAR_PCT * CRITICAL_ESCALATION == pytest.approx(4.5)
    assert concentrated_risk["cvar"] <= concentrated_risk["var"] < 0.0


def test_fat_tails_are_a_warning_priced_through_the_cornish_fisher_gap(
    concentrated_insights: list[Insight], concentrated_risk: dict
):
    """Excess kurtosis is quoted against 0.00 (Gaussian), and the sentence
    prices the misspecification as the gap between the two VaR estimates."""
    fat = pick(concentrated_insights, category="risk", contains="excess kurtosis")

    assert len(fat) == 1
    assert fat[0].level == "warning"
    assert "fat_tail_kurtosis=3.00" in fat[0].evidence
    assert "(0.00 = Gaussian)" in fat[0].evidence
    assert abs(concentrated_risk["var_cornish_fisher"]) > abs(concentrated_risk["var_gaussian"])


def test_a_lopsided_weight_distribution_is_a_diversification_warning(
    concentrated_insights: list[Insight],
):
    """Six tickers but 1.97 effective bets: 1.97 / 6 = 0.33 < 0.50."""
    bets = pick(concentrated_insights, category="diversification", contains="effective bets")

    assert len(bets) == 1
    assert bets[0].level == "warning"
    assert f"effective_holdings={1.0 / 0.508:.2f} / n_holdings=6" in bets[0].evidence
    assert f"effective_holdings_ratio={EFFECTIVE_HOLDINGS_RATIO:.2f}" in bets[0].evidence


def test_a_correlated_book_gets_one_warning_and_then_informational_pairs(
    concentrated_insights: list[Insight],
):
    """0.95 >= 0.80 fires the rule; only the strongest pair is a warning, the
    next two are informational, and at most three pairs are ever quoted."""
    pairs = pick(concentrated_insights, category="diversification", contains="move together")

    assert len(pairs) == 3
    assert [item.level for item in pairs] == ["warning", "info", "info"]
    assert "r^2=0.90 of shared variance" in pairs[0].evidence  # 0.95^2 = 0.9025


def test_the_broken_book_produces_critical_insights_in_both_risk_and_concentration(
    concentrated_insights: list[Insight],
):
    """The headline claim of this fixture, stated once as a whole."""
    critical = pick(concentrated_insights, level="critical")
    warning = pick(concentrated_insights, level="warning")

    assert {item.category for item in critical} == {"concentration", "risk"}
    assert "diversification" in {item.category for item in warning}
    assert headline_insight(concentrated_insights).level == "critical"


def test_the_top_five_rule_compares_against_the_equal_weight_reference(
    concentrated_insights: list[Insight],
):
    """Top-5 is 70 + 4 x 6 = 94% of a six-name book.

    An equal-weight book of six names would give its top five 5/6 = 83.3%, so
    the ratio is 94 / 83.333... = 1.128x — real but below the 1.5x escalation.
    """
    top5 = pick(concentrated_insights, category="concentration", contains="five largest")

    assert len(top5) == 1
    assert top5[0].level == "info"
    assert "top5_weight_pct=94.0%" in top5[0].evidence
    assert "5/6=83.3%" in top5[0].evidence
    assert f"(ratio={94.0 / (500.0 / 6.0):.2f}x" in top5[0].evidence


# ---- the diversified, low-volatility book -------------------------------
def test_the_calm_book_crosses_none_of_the_alarm_thresholds(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    summary = summarise(diversified_positions)

    assert summary["n_holdings"] == 12 >= MIN_DIVERSIFIED_HOLDINGS
    assert summary["largest_holding"][1] == pytest.approx(100.0 / 12.0)
    assert summary["largest_holding"][1] < CONCENTRATION_INFO_PCT
    assert max(summary["sectors"].values()) < SECTOR_CONCENTRATION_PCT
    # equal weights -> HHI = 12 x (1/12)^2 = 1/12, so effective holdings = 12
    assert summary["effective_holdings"] == pytest.approx(12.0)

    assert diversified_risk["mdd"] > DEEP_DRAWDOWN
    assert diversified_risk["ann_vol"] <= HIGH_VOLATILITY / 2.0
    assert abs(diversified_risk["var"]) * 100.0 < HIGH_VAR_PCT
    assert diversified_risk["kurtosis"] < FAT_TAIL_KURTOSIS
    assert diversified_risk["sharpe"] >= GOOD_SHARPE


def test_the_calm_book_produces_no_critical_and_no_warning_insight(
    diversified_insights: list[Insight],
):
    """The negative control. Without it, a rule that fired on everything would
    still pass every assertion made about the broken book above."""
    assert pick(diversified_insights, level="critical") == []
    assert pick(diversified_insights, level="warning") == []


def test_the_calm_book_produces_positive_insights_across_every_theme(
    diversified_insights: list[Insight],
):
    """Concentration, diversification, risk and performance all read positive."""
    positive = pick(diversified_insights, level="positive")

    assert {item.category for item in positive} >= {
        "concentration",
        "diversification",
        "risk",
        "performance",
    }
    assert headline_insight(diversified_insights).level == "positive"


def test_an_uncorrelated_book_is_told_that_no_pair_breaches_the_threshold(
    diversified_insights: list[Insight],
):
    """0.15 everywhere: the rule must say so explicitly rather than stay silent.

    12 names give 12 x 11 / 2 = 66 pairs, and the mean of a constant is that
    constant, so the reported average is exactly 0.150.
    """
    pairs = pick(diversified_insights, category="diversification", contains="No pair")

    assert len(pairs) == 1
    assert pairs[0].level == "positive"
    assert "mean rho=0.150 over 66 pairs" in pairs[0].evidence


def test_the_calm_book_reports_a_positive_sharpe_sortino_and_calmar(
    diversified_insights: list[Insight], diversified_risk: dict
):
    """Each ratio is quoted against its own configured floor, not against zero."""
    sharpe = pick(diversified_insights, contains="Sharpe ratio")
    sortino = pick(diversified_insights, contains="Sortino ratio")
    calmar = pick(diversified_insights, contains="Calmar ratio")

    assert [i.level for i in sharpe + sortino + calmar] == ["positive"] * 3
    assert f"sharpe={diversified_risk['sharpe']:.3f} >= " in sharpe[0].evidence
    assert "good_sharpe=1.00" in sharpe[0].evidence
    assert "good_sortino=1.50" in sortino[0].evidence
    assert "weak_calmar=0.50" in calmar[0].evidence


def test_every_position_in_profit_makes_the_weakest_name_a_laggard_not_a_loser(
    diversified_insights: list[Insight],
):
    """D12 cost 91 and is worth 100: +9.89%, the minimum — but still a gain, so
    the rule must not call it a losing position."""
    weakest = pick(diversified_insights, contains="Weakest position")

    assert len(weakest) == 1
    assert weakest[0].level == "info"
    assert "D12 at +9.9% on cost" in weakest[0].text
    assert "the laggard rather than a loser" in weakest[0].text


# ---- ordering, evidence and the category vocabulary ----------------------
@pytest.mark.parametrize("fixture", ["concentrated_insights", "diversified_insights"])
def test_insights_are_ordered_critical_warning_positive_info(
    fixture: str, request: pytest.FixtureRequest
):
    insights = request.getfixturevalue(fixture)
    ranks = [LEVEL_ORDER.index(item.level) for item in insights]

    assert ranks == sorted(ranks)


def test_the_sort_is_stable_so_the_emission_order_survives_inside_one_severity(
    concentrated_insights: list[Insight],
):
    """Rules are emitted concentration -> diversification -> risk -> performance
    -> benchmark -> data. A stable sort by severity must not reshuffle that, so
    the criticals arrive concentration-first and the warnings
    diversification-first.
    """
    critical = [item.category for item in pick(concentrated_insights, level="critical")]
    warning = [item.category for item in pick(concentrated_insights, level="warning")]

    assert critical == ["concentration", "concentration", "risk", "risk", "risk", "risk"]
    assert warning[0] == "diversification"
    assert warning[-1] == "performance"


@pytest.mark.parametrize("fixture", ["concentrated_insights", "diversified_insights"])
def test_every_insight_carries_evidence_and_a_documented_category(
    fixture: str, request: pytest.FixtureRequest
):
    """The audit contract: a sentence with no number behind it is an opinion."""
    insights = request.getfixturevalue(fixture)

    assert insights, "the fixture must actually produce insights"
    for item in insights:
        assert isinstance(item, Insight)
        assert item.level in LEVEL_ORDER, item
        assert item.category in CATEGORIES, item
        assert isinstance(item.evidence, str)
        assert item.evidence.strip(), item.text
        assert item.text.strip(), item.evidence


@pytest.mark.parametrize("fixture", ["concentrated_insights", "diversified_insights"])
def test_no_evidence_string_leaks_a_nan(fixture: str, request: pytest.FixtureRequest):
    """Undefined statistics are rendered 'n/a'; a literal 'nan' in the UI is a
    formatting bug, not a measurement."""
    insights = request.getfixturevalue(fixture)

    for item in insights:
        assert "nan" not in item.evidence.lower().replace("n/a", ""), item.evidence


# ---- headline_insight and insights_by_category ---------------------------
def test_the_headline_is_the_first_insight_at_the_top_severity(
    concentrated_insights: list[Insight],
):
    headline = headline_insight(concentrated_insights)

    assert headline is concentrated_insights[0]
    assert headline.level == "critical"


def test_the_headline_works_on_an_unsorted_list():
    """Severity, not position, decides — but position breaks the tie."""
    unsorted_list = [
        Insight("info", "first", "data", "e1"),
        Insight("warning", "second", "risk", "e2"),
        Insight("critical", "third", "risk", "e3"),
        Insight("critical", "fourth", "concentration", "e4"),
    ]
    assert headline_insight(unsorted_list).text == "third"


@pytest.mark.parametrize("insights", [None, []])
def test_the_headline_of_nothing_is_none(insights: object):
    assert headline_insight(insights) is None


def test_grouping_presents_categories_in_the_published_order(
    concentrated_insights: list[Insight],
):
    grouped = insights_by_category(concentrated_insights)

    assert list(grouped) == [c for c in CATEGORIES if c in grouped]
    assert "benchmark" not in grouped, "no benchmark ran, so the key is omitted"
    assert sum(len(v) for v in grouped.values()) == len(concentrated_insights)


def test_grouping_preserves_the_severity_ordering_inside_each_category(
    concentrated_insights: list[Insight],
):
    for items in insights_by_category(concentrated_insights).values():
        ranks = [LEVEL_ORDER.index(item.level) for item in items]
        assert ranks == sorted(ranks)


def test_an_unexpected_category_is_kept_and_listed_after_the_known_ones():
    grouped = insights_by_category(
        [
            Insight("info", "custom", "tax", "e1"),
            Insight("info", "known", "risk", "e2"),
        ]
    )
    assert list(grouped) == ["risk", "tax"]


@pytest.mark.parametrize("insights", [None, []])
def test_grouping_nothing_is_an_empty_mapping(insights: object):
    assert insights_by_category(insights) == {}


# ---- drawdown episodes ---------------------------------------------------
def episode(
    depth: float, recovery_days: int | None, trough: date = date(2023, 6, 1)
) -> DrawdownEpisode:
    return DrawdownEpisode(
        peak_date=date(2023, 3, 1),
        trough_date=trough,
        recovery_date=None if recovery_days is None else date(2024, 1, 1),
        depth=depth,
        drawdown_days=92,
        recovery_days=recovery_days,
    )


def test_an_unrecovered_drawdown_is_a_warning_not_history(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
):
    insights = generate_insights(
        concentrated_positions,
        summarise(concentrated_positions),
        concentrated_risk,
        None,
        None,
        episodes=[episode(-0.42, None)],
    )
    unrecovered = pick(insights, contains="has not recovered")

    assert len(unrecovered) == 1
    assert unrecovered[0].level == "warning"
    assert unrecovered[0].category == "risk"
    assert "recovery_date=None" in unrecovered[0].evidence


def test_a_recovery_longer_than_the_configured_limit_is_a_warning(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """181 > 180 calendar days, so the rule fires on the strict inequality."""
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        None,
        episodes=[episode(-0.25, LONG_RECOVERY_DAYS + 1)],
    )
    slow = pick(insights, contains="Long recoveries cost")

    assert len(slow) == 1
    assert slow[0].level == "warning"
    assert f"recovery_days=181 > INSIGHTS.long_recovery_days={LONG_RECOVERY_DAYS}" in (
        slow[0].evidence
    )


def test_a_recovery_of_exactly_the_configured_limit_is_still_positive(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """The boundary: 180 days is fast enough, 181 is not."""
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        None,
        episodes=[episode(-0.25, LONG_RECOVERY_DAYS)],
    )
    fast = pick(insights, contains="was fully recovered")

    assert len(fast) == 1
    assert fast[0].level == "positive"


def test_repeated_deep_drawdowns_are_reported_as_a_property_of_the_strategy(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """Three episodes are supplied but only the two beyond -20% are counted."""
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        None,
        episodes=[episode(-0.31, 30), episode(-0.24, 40), episode(-0.05, 10)],
    )
    repeated = pick(insights, contains="separate drawdowns beyond")

    assert len(repeated) == 1
    assert repeated[0].level == "info"
    assert "2 separate drawdowns beyond 20%" in repeated[0].text
    assert "2 of 3 recorded" in repeated[0].evidence


def test_the_deepest_episode_is_the_one_described(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """Episodes may arrive in any order; -0.31 is the one that gets the sentence."""
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        None,
        episodes=[episode(-0.09, 5), episode(-0.31, 20), episode(-0.18, 12)],
    )
    described = pick(insights, contains="was fully recovered")

    assert len(described) == 1
    assert "The deepest drawdown (31.0%)" in described[0].text


# ---- benchmark-relative rules -------------------------------------------
BENCH = {
    "ok": True,
    "excess_return": 0.04,
    "portfolio_return": 0.18,
    "benchmark_return": 0.14,
    "portfolio_vol": 0.25,
    "benchmark_vol": 0.20,
    "portfolio_mdd": -0.30,
    "benchmark_mdd": -0.25,
    "beta": 1.50,
    "r_squared": 0.81,
    "alpha": 0.03,
    "tracking_error": 0.06,
    "information_ratio": 0.67,
    "up_capture": 1.10,
    "down_capture": 0.90,
    "m2": 0.16,
}


@pytest.fixture
def benchmarked_insights(
    diversified_positions: pd.DataFrame, diversified_risk: dict
) -> list[Insight]:
    return generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        BENCH,
        None,
    )


def test_beating_the_benchmark_is_reported_in_percentage_points(
    benchmarked_insights: list[Insight],
):
    """18% against 14% is +4.00 POINTS of excess return, not +28.6%."""
    excess = pick(benchmarked_insights, category="benchmark", contains="outperformed")

    assert len(excess) == 1
    assert excess[0].level == "positive"
    assert "by 4.00 percentage points annualised" in excess[0].text
    assert "excess_return=+4.00pp" in excess[0].evidence


def test_carrying_more_volatility_than_the_index_is_a_warning(
    benchmarked_insights: list[Insight],
):
    """25 / 20 = 1.25, outside the 1 +/- 0.10 tolerance band."""
    relative = pick(benchmarked_insights, category="benchmark", contains="above the benchmark's")

    assert len(relative) == 1
    assert relative[0].level == "warning"
    assert "portfolio_vol/benchmark_vol=1.250 > 1.10" in relative[0].evidence


def test_a_deeper_drawdown_than_the_index_is_a_warning_measured_in_points(
    benchmarked_insights: list[Insight],
):
    """30% against 25% is 5.0 points deeper."""
    relative = pick(benchmarked_insights, category="benchmark", contains="Worst drawdown")

    assert relative[0].level == "warning"
    assert "5.0 percentage points deeper" in relative[0].text
    assert "gap=+5.00pp" in relative[0].evidence


def test_an_aggressive_beta_is_translated_into_a_ten_percent_market_move(
    benchmarked_insights: list[Insight],
):
    """Beta 1.50 > 1.20, and 10% of market maps to 15.0% here."""
    beta = pick(benchmarked_insights, category="benchmark", contains="Beta 1.50")

    assert len(beta) == 1
    assert f"beta=1.500 > INSIGHTS.aggressive_beta={AGGRESSIVE_BETA:.2f}" in beta[0].evidence
    assert "roughly 15.0% here" in beta[0].text
    assert "read this beta loosely" not in beta[0].text  # r^2 = 0.81 is meaningful


def test_capture_ratios_are_accepted_as_fractions_or_as_percentages(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """1.10/0.90 and 110/90 are the same claim; the module must not read the
    percent form as a 110x up-capture."""
    summary = summarise(diversified_positions)
    as_percent = dict(BENCH, up_capture=110.0, down_capture=90.0)

    fraction_text = pick(
        generate_insights(diversified_positions, summary, diversified_risk, BENCH, None),
        contains="Up-capture",
    )[0]
    percent_text = pick(
        generate_insights(diversified_positions, summary, diversified_risk, as_percent, None),
        contains="Up-capture",
    )[0]

    assert fraction_text.text == percent_text.text
    assert "Up-capture 110% and down-capture 90%" in fraction_text.text
    assert fraction_text.level == "positive"


def test_alpha_is_discounted_when_the_benchmark_explains_little_variance(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """R-squared 0.30 < 0.50, so alpha and beta describe a relationship that
    mostly is not there — the sentence has to say so rather than quote alpha."""
    weak = dict(BENCH, r_squared=0.30)
    insights = generate_insights(
        diversified_positions, summarise(diversified_positions), diversified_risk, weak, None
    )
    alpha = pick(insights, category="benchmark", contains="Alpha is")

    assert len(alpha) == 1
    assert alpha[0].level == "info"
    assert "explains only 30% of the portfolio's variance" in alpha[0].text
    assert f"MEANINGFUL_R_SQUARED={MEANINGFUL_R_SQUARED:.2f}" in alpha[0].evidence


def test_a_failed_benchmark_comparison_produces_exactly_one_explanatory_insight(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """``ok=False`` gates the whole section; the reason is quoted verbatim."""
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        {"ok": False, "reason": "no overlapping dates"},
        None,
    )
    bench = pick(insights, category="benchmark")

    assert len(bench) == 1
    assert bench[0].level == "info"
    assert "no overlapping dates" in bench[0].text
    assert bench[0].evidence == ("benchmark comparison ok=False, reason='no overlapping dates'")


@pytest.mark.parametrize("bench", [None, {}])
def test_no_benchmark_argument_means_no_benchmark_insights(
    bench: object, diversified_positions: pd.DataFrame, diversified_risk: dict
):
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        bench,
        None,
    )
    assert pick(insights, category="benchmark") == []


# ---- risk-versus-weight (attribution) -----------------------------------
def test_a_name_carrying_more_risk_than_weight_is_flagged(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
):
    """MEGA is 70% of the money and 90% of the risk: 90 / 70 = 1.29x >= 1.25x."""
    attribution = {
        "risk_vs_weight": pd.DataFrame(
            {
                "ticker": ["MEGA", "N1"],
                "weight_pct": [70.0, 6.0],
                "risk_contribution_pct": [90.0, 2.0],
            }
        )
    }
    insights = generate_insights(
        concentrated_positions,
        summarise(concentrated_positions),
        concentrated_risk,
        None,
        None,
        attribution=attribution,
    )
    hog = pick(insights, category="concentration", contains="of the risk")

    assert len(hog) == 1
    assert hog[0].level == "warning"
    assert f"{90.0 / 70.0:.2f}x >= RISK_WEIGHT_RATIO={RISK_WEIGHT_RATIO:.2f}x" in (hog[0].evidence)


def test_risk_exactly_proportional_to_weight_is_not_flagged(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    """A ratio of 1.00x is the parity point; the rule must stay silent."""
    attribution = {
        "risk_vs_weight": pd.DataFrame(
            {
                "ticker": DIVERSIFIED_TICKERS,
                "weight_pct": [100.0 / 12.0] * 12,
                "risk_contribution_pct": [100.0 / 12.0] * 12,
            }
        )
    }
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        None,
        attribution=attribution,
    )
    assert pick(insights, contains="of the risk") == []


# ---- data sufficiency ----------------------------------------------------
def test_a_short_window_warns_that_the_statistics_are_only_indicative(
    diversified_positions: pd.DataFrame,
):
    """20 daily observations is below the 30-observation floor.

    The warning quotes the relative standard error of sigma, 1/sqrt(2n), which
    for n = 20 is 1/sqrt(40) = 15.8% of the estimate itself.
    """
    short = value_series(sigma=0.01, drift=0.0005, days=20)
    risk = risk_summary(short, n_holdings=12)
    insights = generate_insights(
        diversified_positions, summarise(diversified_positions), risk, None, None
    )
    data = pick(insights, category="data")

    assert risk["n_obs"] == 20.0 < MIN_OBSERVATIONS
    assert len(data) == 1
    assert data[0].level == "warning"
    assert f"n_obs=20 < MIN_OBSERVATIONS={MIN_OBSERVATIONS}" in data[0].evidence
    assert f"~{1 / np.sqrt(40) * 100:.1f}%" in data[0].evidence


def test_a_long_window_reports_the_sample_size_as_information(
    diversified_insights: list[Insight],
):
    data = pick(diversified_insights, category="data")

    assert len(data) == 1
    assert data[0].level == "info"
    assert "500 overlapping daily observations" in data[0].text


def test_no_return_history_at_all_disables_every_statistic_with_a_warning(
    diversified_positions: pd.DataFrame,
):
    """The holdings table is still valid; nothing else is, and it says so."""
    insights = generate_insights(
        diversified_positions, summarise(diversified_positions), {}, None, None
    )
    data = pick(insights, category="data")

    assert len(data) == 1
    assert data[0].level == "warning"
    assert "No usable return history" in data[0].text
    assert pick(insights, category="risk") == []


# ---- degenerate input: degrade, never raise ------------------------------
@pytest.mark.parametrize(
    "positions", [None, pd.DataFrame(), pd.DataFrame(columns=["ticker"]), "not a frame"]
)
def test_an_empty_portfolio_returns_exactly_one_warning(positions: object):
    """Not zero (the page would be blank) and not a dozen rules firing on NaN."""
    insights = generate_insights(positions, {"n_holdings": 0}, {}, {}, None)

    assert len(insights) == 1
    assert insights[0].level == "warning"
    assert insights[0].category == "data"
    assert insights[0].evidence.strip()
    assert "No valid positions to analyse" in insights[0].text
    assert headline_insight(insights) is insights[0]


def test_a_valid_book_with_every_optional_argument_missing_still_works(
    diversified_positions: pd.DataFrame,
):
    """summary, risk, bench, corr, episodes and attribution all absent."""
    insights = generate_insights(diversified_positions, None, None, None, None)

    assert insights
    assert all(item.evidence.strip() for item in insights)
    assert {item.category for item in insights} <= set(CATEGORIES)
    # n_holdings falls back to len(positions) when no summary was supplied
    assert pick(insights, contains="12 positions clears")


def test_a_correlation_frame_with_one_asset_produces_no_pair_insights(
    diversified_positions: pd.DataFrame, diversified_risk: dict
):
    one = pd.DataFrame([[1.0]], index=["D01"], columns=["D01"])
    insights = generate_insights(
        diversified_positions,
        summarise(diversified_positions),
        diversified_risk,
        None,
        one,
    )
    assert pick(insights, contains="pairwise correlation") == []


def test_generate_insights_does_not_mutate_any_of_its_arguments(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
):
    summary = summarise(concentrated_positions)
    corr = correlation(CONCENTRATED_TICKERS, 0.95)
    positions_before = concentrated_positions.copy(deep=True)
    corr_before = corr.copy(deep=True)
    risk_before = dict(concentrated_risk)
    summary_before = dict(summary)
    bench_before = dict(BENCH)

    generate_insights(concentrated_positions, summary, concentrated_risk, BENCH, corr)

    pd.testing.assert_frame_equal(concentrated_positions, positions_before)
    pd.testing.assert_frame_equal(corr, corr_before)
    assert concentrated_risk == risk_before
    assert summary == summary_before
    assert BENCH == bench_before


# Regression guard for a fixed defect:
# insights._extreme_row does `float(series.loc[idx])` after `idxmax()`, which returns an
# index LABEL. On a positions frame with a duplicated index -- what any caller who
# concatenates or slices positions without reset_index produces -- `.loc[label]` yields a
# multi-row Series and float() raises TypeError. positions.summarise was explicitly
# hardened against exactly this (it uses a positional argmax); generate_insights was not,
# so it raises instead of degrading.
def test_a_duplicated_positions_index_degrades_instead_of_raising(
    concentrated_positions: pd.DataFrame, concentrated_risk: dict
):
    duplicated = concentrated_positions.copy()
    duplicated.index = [7] * len(duplicated)

    insights = generate_insights(duplicated, summarise(duplicated), concentrated_risk, None, None)

    assert insights
    assert all(item.evidence.strip() for item in insights)
