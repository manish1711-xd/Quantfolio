"""Benchmark -- relative performance against the chosen index.

The Overview page says how the portfolio did. This page says whether doing it
was worth the trouble: it separates the return the market simply handed over
from the return the portfolio earned on its own, and prices the risk that was
taken while trying.

Five questions, in order:

* **Did it beat the index?** Annualised return on both legs and the geometric
  excess between them, measured only on the dates both series actually traded.
* **Was the excess bought with risk or earned?** Beta says how much index
  exposure was carried; Jensen's alpha is what survives after paying for it.
* **How far can this book wander from the index?** Tracking error, and the
  information ratio that says whether the wandering was rewarded.
* **Did it behave differently in up and down months?** Up and down capture --
  the asymmetry a long-only investor is really trying to buy.
* **Is any of this stable?** Rolling beta and rolling correlation, because a
  beta quoted to two decimals over one window is a point estimate, not a
  constant.

Every statistic is read off ``result.benchmark``, which the pipeline built with
``analytics.benchmark.compare``. The one thing computed here is the
relative-strength ratio, a pure re-expression of two series the result already
carries, plus the benchmark's Sharpe ratio -- one division, done on the same
overlap and the same risk-free rate as the portfolio's, so the side-by-side
table compares like with like.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import benchmark as bm
from src.analytics.insights import MEANINGFUL_R_SQUARED
from src.config import INSIGHTS, MIN_OBSERVATIONS, PERIOD_LABELS, TRADING_DAYS
from src.ui import components as ui
from src.ui import guards
from src.ui import theme as ui_theme
from src.ui.formatters import (
    delta_tone,
    fmt_date,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    fmt_signed,
)

#: Information-ratio reference points, taken from the convention stated in
#: ``analytics.benchmark.information_ratio``: broadly 0.5 is good and 1.0 is
#: exceptional when sustained across a full market cycle. They colour a card;
#: they are not thresholds anything is judged on.
_GOOD_IR = 0.50
_EXCEPTIONAL_IR = 1.00

#: A capture ratio at this level means the portfolio gave back exactly what it
#: captured. Above it is the asymmetry a long-only book is built for.
_PARITY_CAPTURE_RATIO = 1.00

#: Capture is a monthly statistic; below this many months in a bucket the
#: average is one or two observations wearing a decimal point.
_THIN_CAPTURE_MONTHS = 6


# ---------------------------------------------------------------------------
# small local helpers (display only -- no analytics happen on this page)
# ---------------------------------------------------------------------------
def _f(value: Any) -> float:
    """Coerce to float, collapsing anything unusable to NaN.

    Every formatter renders NaN as an em dash, so a missing statistic flows
    through the whole page as "not available" rather than as a zero.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _ok(value: Any) -> bool:
    """True when ``value`` is a real, finite number worth printing."""
    return math.isfinite(_f(value))


def _sharpe(ann_return: float, ann_vol: float, risk_free_rate: float) -> float:
    """``(R_a - R_f) / sigma`` -- the same convention the risk engine uses.

    Computed here rather than read off ``result.risk`` so that both legs of the
    comparison table use the *overlap* window and one identical risk-free rate.
    A Sharpe measured on a different set of dates is not comparable with one
    measured on this set, and quietly mixing the two would be the exact kind of
    apples-to-oranges this page exists to prevent.
    """
    ret, vol, rate = _f(ann_return), _f(ann_vol), _f(risk_free_rate)
    if not (math.isfinite(ret) and math.isfinite(rate)):
        return float("nan")
    if not math.isfinite(vol) or vol == 0.0:
        return float("nan")
    return float((ret - rate) / vol)


def _rolling_column(result: Any, column: str) -> pd.Series | None:
    """One column of the rolling frame, NaN warm-up dropped; None when absent."""
    frame = result.rolling
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return series if not series.empty else None


def _span(series: pd.Series | None) -> tuple[float, float, float]:
    """``(minimum, maximum, latest)`` of a rolling series, NaN when unavailable."""
    if series is None or series.empty:
        return float("nan"), float("nan"), float("nan")
    return float(series.min()), float(series.max()), float(series.iloc[-1])


def _beta_tone(beta: Any) -> str:
    """Amber when geared to the index, green when defensive. Colour only."""
    if not _ok(beta):
        return "muted"
    value = _f(beta)
    if value >= INSIGHTS.aggressive_beta:
        return "warn"
    return "gain" if value <= INSIGHTS.defensive_beta else "neutral"


def _ir_tone(ir: Any) -> str:
    if not _ok(ir):
        return "muted"
    value = _f(ir)
    if value >= _GOOD_IR:
        return "gain"
    return "loss" if value < 0 else "warn"


def _r2_tone(r2: Any) -> str:
    """Below the meaningful line the index is a weak description of the book."""
    if not _ok(r2):
        return "muted"
    return "neutral" if _f(r2) >= MEANINGFUL_R_SQUARED else "warn"


def _capture_ratio_tone(ratio: Any, down_capture: Any) -> str:
    """A negative down-capture is a *gain* in falling months, not a failure."""
    if _ok(down_capture) and _f(down_capture) <= 0.0:
        return "gain"
    if not _ok(ratio):
        return "muted"
    return "gain" if _f(ratio) > _PARITY_CAPTURE_RATIO else "loss"


def _lead_word(value: Any, ahead: str = "ahead", behind: str = "behind") -> str:
    return ahead if _f(value) >= 0 else behind


# ---------------------------------------------------------------------------
# 1. context strip
# ---------------------------------------------------------------------------
def _context(result: Any, bench: dict, cfg: Any) -> None:
    window = PERIOD_LABELS.get(getattr(cfg, "period", ""), str(getattr(cfg, "period", "")))
    overlap = int(_f(bench.get("n_days")) if _ok(bench.get("n_days")) else 0)
    dropped = max(result.n_days - overlap, 0)

    chips: list[tuple[str, str]] = [
        (
            f"{getattr(cfg, 'benchmark_name', 'Benchmark')} ({getattr(cfg, 'benchmark_symbol', '')})",
            "accent",
        ),
        (f"Window {window}", "neutral"),
        (f"{overlap} overlapping sessions", "neutral"),
        (f"Risk-free {fmt_ratio_pct(getattr(cfg, 'risk_free_rate', float('nan')))}", "muted"),
        ("Price-only index", "warn"),
    ]
    if dropped:
        chips.append((f"{dropped} portfolio sessions unmatched", "warn"))
    if overlap and overlap < MIN_OBSERVATIONS:
        chips.append(("Short overlap", "loss"))
    ui.badges(chips)

    if dropped:
        st.caption(
            f"{dropped} of the portfolio's {result.n_days} sessions have no matching "
            f"index price and are excluded from every statistic below. That is the "
            f"correct treatment -- comparing a book against an index across misaligned "
            f"trading calendars measures the holiday schedule -- but it does mean the "
            f"window here is shorter than the one the Overview and Risk pages use."
        )

    if overlap and overlap < MIN_OBSERVATIONS:
        st.caption(
            f"Only {overlap} dates overlap, below the {MIN_OBSERVATIONS}-observation "
            f"floor this application treats as the minimum for a stable estimate. "
            f"Everything below is still computed exactly as documented, but "
            f"annualising from {overlap} sessions magnifies whatever happened to fall "
            f"inside them: read these as a description of this particular window, not "
            f"as a property of the portfolio."
        )


# ---------------------------------------------------------------------------
# 2. absolute performance against the index
# ---------------------------------------------------------------------------
def _headline_kpis(bench: dict, cfg: Any) -> None:
    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    p_ret = _f(bench.get("portfolio_return"))
    b_ret = _f(bench.get("benchmark_return"))
    excess = _f(bench.get("excess_return"))
    beta = _f(bench.get("beta"))
    r2 = _f(bench.get("r_squared"))
    overlap = int(_f(bench.get("n_days"))) if _ok(bench.get("n_days")) else 0

    cards: list[dict[str, Any]] = [
        {
            "label": "Portfolio return (ann.)",
            "value": fmt_ratio_pct(p_ret),
            "tone": delta_tone(p_ret),
            "footnote": f"Geometric, over {overlap} overlapping sessions",
            "help_text": (
                "(V_end / V_start) ** (252 / n) - 1 on the portfolio value series, "
                "restricted to the dates the index also traded."
            ),
        },
        {
            "label": f"{name} return (ann.)",
            "value": fmt_ratio_pct(b_ret),
            "tone": delta_tone(b_ret),
            "footnote": "Index level only -- dividends are not in the series",
            "help_text": (
                "Same geometric formula on the index level, same dates. A price "
                "index understates the total return an index fund would have paid."
            ),
        },
        {
            "label": "Excess return",
            "value": fmt_ratio_pct(excess, signed=True),
            "tone": delta_tone(excess),
            "footnote": "Portfolio less index, before costs and taxes",
            "help_text": (
                "The geometric gap between the two annualised returns. It is raw "
                "outperformance: it says nothing about how much index risk bought it."
            ),
        },
        {
            "label": f"Beta vs {name}",
            "value": fmt_num(beta, 2),
            "tone": _beta_tone(beta),
            "footnote": f"R-squared {fmt_ratio_pct(r2)} of daily variance explained",
            "help_text": (
                "cov(r_portfolio, r_index) / var(r_index) on overlapping daily "
                "returns. 1.00 moves with the index; above 1.20 is geared to it."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    if _ok(p_ret) and _ok(b_ret) and _ok(excess):
        line = (
            f"Over the {overlap} sessions both series traded, the portfolio compounded "
            f"at {fmt_ratio_pct(p_ret)} against {fmt_ratio_pct(b_ret)} for {name} -- "
            f"{_lead_word(excess)} by {fmt_ratio_pct(abs(excess))} annualised."
        )
        if _ok(beta) and _ok(r2):
            line += (
                f" Beta {fmt_num(beta, 2)} says a 1% session on the index moved the book "
                f"about {fmt_num(beta, 2)}% on average, and R-squared {fmt_ratio_pct(r2)} "
                f"is how much of the daily variance that explains; the remaining "
                f"{fmt_ratio_pct(1.0 - r2)} is idiosyncratic, and the index has nothing "
                f"to say about it."
            )
        st.caption(line)

    if _ok(r2) and _f(r2) < MEANINGFUL_R_SQUARED:
        st.caption(
            f"Read the beta and alpha below with care: at R-squared "
            f"{fmt_ratio_pct(r2)} the index explains less than "
            f"{fmt_ratio_pct(MEANINGFUL_R_SQUARED, 0)} of this portfolio's daily "
            f"movement, which is the level below which a CAPM decomposition against "
            f"it is an artefact of the fit rather than a measurement of skill. A "
            f"closer benchmark would describe the book better."
        )


# ---------------------------------------------------------------------------
# 3. relative risk
# ---------------------------------------------------------------------------
def _relative_risk_kpis(result: Any, bench: dict, cfg: Any) -> None:
    str(getattr(cfg, "benchmark_name", "Benchmark"))
    currency = str(getattr(cfg, "base_currency", "INR"))
    alpha = _f(bench.get("alpha"))
    beta = _f(bench.get("beta"))
    te = _f(bench.get("tracking_error"))
    ir = _f(bench.get("information_ratio"))
    r2 = _f(bench.get("r_squared"))
    correlation = _f(bench.get("correlation"))
    excess = _f(bench.get("excess_return"))
    value = _f((result.summary or {}).get("current_value"))

    te_footnote = "Annualised standard deviation of the daily active return"
    if _ok(te) and _ok(value):
        te_footnote = (
            f"About {fmt_money(te * value, currency, 0, compact=True)} of annual "
            f"wander around the index on today's book"
        )

    cards: list[dict[str, Any]] = [
        {
            "label": "Jensen's alpha",
            "value": fmt_ratio_pct(alpha, signed=True),
            "tone": delta_tone(alpha),
            "footnote": (
                f"After paying for beta {fmt_num(beta, 2)} of index exposure"
                if _ok(beta)
                else "CAPM residual over the window"
            ),
            "help_text": (
                "R_p - [R_f + beta x (R_b - R_f)], on annualised returns. What is "
                "left of the excess once the index exposure has been paid for."
            ),
        },
        {
            "label": "Tracking error",
            "value": fmt_ratio_pct(te),
            "tone": "neutral",
            "footnote": te_footnote,
            "help_text": (
                "stdev(r_portfolio - r_index) x sqrt(252). How far the book can "
                "drift from the index -- direction-blind, so beating it erratically "
                "and trailing it erratically score the same."
            ),
        },
        {
            "label": "Information ratio",
            "value": fmt_num(ir, 2),
            "tone": _ir_tone(ir),
            "footnote": "Active return per unit of tracking error",
            "help_text": (
                "mean(active) / stdev(active) x sqrt(252): the Sharpe ratio of the "
                "active bet. Roughly 0.5 is good, 1.0 exceptional, across a cycle."
            ),
        },
        {
            "label": "R-squared",
            "value": fmt_ratio_pct(r2),
            "tone": _r2_tone(r2),
            "footnote": (
                f"Correlation {fmt_num(correlation, 2)}, squared"
                if _ok(correlation)
                else "Share of daily variance the index explains"
            ),
            "help_text": (
                "corr(r_portfolio, r_index) ** 2. The honesty check on beta and "
                "alpha: both are weak descriptions when this number is low."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    if _ok(alpha) and _ok(excess) and _ok(beta):
        if _f(beta) > 1.05:
            geared = (
                f"Because beta is above 1.00, CAPM expected this book to move "
                f"{fmt_num(beta, 2)} times the index's excess over cash, so alpha "
                f"charges it for that gearing rather than crediting the extra return "
                f"the gearing alone produced"
            )
        elif _f(beta) < 0.95:
            geared = (
                f"Because beta is below 1.00, the book was only ever expected to earn "
                f"{fmt_num(beta, 2)} of the index's excess over cash, so alpha credits "
                f"it for the market risk it chose not to take"
            )
        else:
            geared = (
                "With beta close to 1.00 the two numbers nearly coincide -- there is "
                "almost no index exposure to charge for"
            )
        st.caption(
            f"Excess return and alpha answer different questions. The excess is the "
            f"raw {fmt_ratio_pct(excess, signed=True)} gap; alpha is what survives once "
            f"the portfolio has paid for the beta {fmt_num(beta, 2)} of index exposure "
            f"that produced most of the move -- {fmt_ratio_pct(alpha, signed=True)} "
            f"here. {geared}."
        )

    if _ok(ir) and _ok(te):
        if _f(ir) >= _EXCEPTIONAL_IR:
            verdict = (
                "exceptional territory by the usual convention, and almost certainly "
                "flattered by a window this short"
            )
        elif _f(ir) >= _GOOD_IR:
            verdict = "the level normally described as good when it is sustained across a cycle"
        elif _f(ir) >= 0:
            verdict = "positive but thin -- the active risk was barely paid for"
        else:
            verdict = (
                "negative: the book wandered away from the index and was not paid for doing so"
            )
        st.caption(
            f"Tracking error {fmt_ratio_pct(te)} is the size of the active bet; "
            f"information ratio {fmt_num(ir, 2)} is what it earned -- "
            f"{fmt_num(ir, 2)} units of active return per unit of active risk, {verdict}. "
            f"Note the mismatch built into the pair: the IR numerator is the "
            f"*arithmetic* annualised active return, while the excess return above is "
            f"*geometric*, so the two will not reconcile to the last decimal by design."
        )


# ---------------------------------------------------------------------------
# 4. growth of 100
# ---------------------------------------------------------------------------
def _growth(bench: dict, cfg: Any) -> None:
    ui_theme.section("GROWTH OF 100", "\U0001f4c8")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    growth = bench.get("growth")
    ui.chart(ch.benchmark_comparison(growth, name))

    if isinstance(growth, pd.DataFrame) and not growth.empty:
        try:
            last = growth.dropna().iloc[-1]
            port_level = _f(last.get("Portfolio"))
            bench_level = _f(last.get("Benchmark"))
            start = growth.dropna().index[0]
        except (IndexError, KeyError, AttributeError):  # pragma: no cover - defensive
            port_level = bench_level = float("nan")
            start = None
        if _ok(port_level) and _ok(bench_level):
            gap = port_level - bench_level
            st.caption(
                f"Both legs start at exactly 100 on {fmt_date(start)}, the first date "
                f"they share, so vertical distance *is* relative performance -- no "
                f"ruler required. A 100 invested in the portfolio ended the window at "
                f"{fmt_num(port_level, 1)} against {fmt_num(bench_level, 1)} in {name}: "
                f"{fmt_num(abs(gap), 1)} points {_lead_word(gap)}, or "
                f"{fmt_ratio_pct(port_level / bench_level - 1.0, signed=True)} of "
                f"cumulative relative performance. Shading marks the stretches the "
                f"portfolio led and the stretches it lagged; both matter, because the "
                f"end point alone hides how long the lag was lived through."
            )
    else:
        st.caption("No growth table was produced for this comparison, so there is nothing to plot.")


# ---------------------------------------------------------------------------
# 5. capture in up and down months
# ---------------------------------------------------------------------------
def _capture(bench: dict, cfg: Any) -> None:
    ui_theme.section("CAPTURE BY REGIME", "\U0001f39a")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    up = _f(bench.get("up_capture"))
    down = _f(bench.get("down_capture"))
    ratio = _f(bench.get("capture_ratio"))
    up_months = int(_f(bench.get("up_months"))) if _ok(bench.get("up_months")) else 0
    down_months = int(_f(bench.get("down_months"))) if _ok(bench.get("down_months")) else 0
    best = _f(bench.get("best_relative_month"))
    worst = _f(bench.get("worst_relative_month"))
    total_months = up_months + down_months

    cards: list[dict[str, Any]] = [
        {
            "label": "Up capture",
            "value": fmt_pct(up),
            "tone": "neutral",
            "footnote": f"Averaged over {up_months} month(s) {name} rose",
            "help_text": (
                "Portfolio's average monthly return in up months divided by the "
                "index's, x 100. 100 means it matched the index in that regime."
            ),
        },
        {
            "label": "Down capture",
            "value": fmt_pct(down),
            "tone": "neutral",
            "footnote": f"Averaged over {down_months} month(s) {name} fell",
            "help_text": (
                "Same ratio in the months the index fell. Below 100 the portfolio "
                "took less than the index's average loss; below 0 it actually rose."
            ),
        },
        {
            "label": "Capture ratio",
            "value": fmt_num(ratio, 2),
            "tone": _capture_ratio_tone(ratio, down),
            "footnote": "Up capture / down capture -- above 1.00 is what good looks like",
            "help_text": (
                "The asymmetry in a single number. Above 1.00 the portfolio kept "
                "more of the upside than it surrendered of the downside."
            ),
        },
        {
            "label": "Monthly sample",
            "value": f"{total_months}",
            "tone": "warn" if min(up_months, down_months) < _THIN_CAPTURE_MONTHS else "neutral",
            "footnote": f"{up_months} up, {down_months} down; flat months count in neither",
            "help_text": (
                "Capture is a monthly statistic, so the sample size is months, not "
                "sessions. Small buckets produce confident-looking averages."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    if _ok(up) and _ok(down):
        reading = (
            f"In the {up_months} month(s) {name} rose, the portfolio banked "
            f"{fmt_pct(up)} of the index's average gain; in the {down_months} month(s) "
            f"it fell, the portfolio took {fmt_pct(down)} of the average loss."
        )
        if _f(down) <= 0.0:
            verdict = (
                f"A negative down-capture is the rare version of this: the book was "
                f"*up* on average in the months the index was down. Treat it as a "
                f"property of {down_months} month(s), not as a hedge."
            )
        elif _f(up) >= 100.0 and _f(down) <= 100.0:
            verdict = (
                f"That is the shape every long-only investor is trying to buy -- full "
                f"participation on the way up, muted participation on the way down -- "
                f"and the capture ratio of {fmt_num(ratio, 2)}, above 1.00, is what "
                f"good looks like."
            )
        elif _ok(ratio) and _f(ratio) > _PARITY_CAPTURE_RATIO:
            verdict = (
                f"Neither leg is remarkable alone, but the ratio {fmt_num(ratio, 2)} is "
                f"above 1.00, and that is the number that matters: the book gave back "
                f"less per unit of upside than it captured. Capturing 95% of the upside "
                f"while taking only 70% of the downside is the same trade made "
                f"explicit."
            )
        else:
            verdict = (
                f"The ratio {fmt_num(ratio, 2)} sits at or below 1.00: the portfolio "
                f"surrendered at least as much in falling months as it captured in "
                f"rising ones. That is the wrong side of the asymmetry even when the "
                f"total return looks acceptable -- an investor who cannot sit through "
                f"the down months never collects the up ones."
            )
        st.caption(f"{reading} {verdict}")
    elif up_months == 0 or down_months == 0:
        missing = "down" if down_months == 0 else "up"
        st.caption(
            f"One side of the pair is empty: {name} did not close a single {missing} "
            f"month inside this window, so that leg is a dash rather than an invented "
            f"number and the ratio is undefined. Capture needs both regimes before it "
            f"says anything at all."
        )

    st.caption(
        "Both legs are computed on **calendar-month compounded returns**, not daily "
        "ones. That is the industry convention and it is not cosmetic: measured "
        "daily, capture is dominated by microstructure noise and reports the tape "
        "rather than how the book behaved in a regime."
    )

    if _ok(best) or _ok(worst):
        st.caption(
            f"The best month relative to {name} was "
            f"{fmt_ratio_pct(best, signed=True)} and the worst "
            f"{fmt_ratio_pct(worst, signed=True)} -- the honest scale of how far one "
            f"month of active positioning can move away from the index."
        )

    ui.methodology(
        "How capture is computed, and what a small sample does to it",
        f"""
{bench.get("capture_methodology") or ""}

**Sample size, stated plainly.** This window supplied {up_months} up month(s)
and {down_months} down month(s). Each capture figure is the ratio of two
averages taken inside those buckets, so with fewer than {_THIN_CAPTURE_MONTHS}
months on a side a single outlier month effectively sets the number. The card
turns amber when either bucket is that thin.

**What it does not measure.** Capture is silent about *when* the months
happened. A portfolio that captured the downside badly early and well later
scores identically to one that did the reverse, and only the rolling panels
below distinguish them.
        """,
    )


# ---------------------------------------------------------------------------
# 6. stability of the relationship
# ---------------------------------------------------------------------------
def _stability(result: Any, bench: dict, cfg: Any) -> None:
    ui_theme.section("IS THE RELATIONSHIP STABLE?", "\U0001f4c9")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    window = int(getattr(cfg, "rolling_window", 63) or 63)
    beta = _f(bench.get("beta"))
    correlation = _f(bench.get("correlation"))

    beta_series = _rolling_column(result, "beta")
    corr_series = _rolling_column(result, "correlation")

    ui.chart(
        ch.rolling_chart(
            result.rolling,
            "beta",
            title=f"Rolling beta vs {name}",
            subtitle=f"{window}-session trailing window, dotted line is the window mean",
            window=window,
        )
    )

    beta_low, beta_high, beta_last = _span(beta_series)
    if _ok(beta_low) and _ok(beta_high):
        st.caption(
            f"The headline beta of {fmt_num(beta, 2)} is one number fitted to the whole "
            f"window; this line is the same calculation on a trailing {window} sessions. "
            f"It ranged {fmt_num(beta_low, 2)} to {fmt_num(beta_high, 2)} and last read "
            f"{fmt_num(beta_last, 2)}. A beta estimated over one window is a point "
            f"estimate, not a constant -- and because var(r_index) is its denominator, "
            f"it moves when the *index* changes volatility even if the portfolio does "
            f"nothing at all. Any figure downstream that consumes beta (alpha, Treynor, "
            f"the beta-shock scenarios on the Stress page) inherits that instability."
        )
    else:
        st.caption(
            f"No rolling beta could be formed: the overlap is shorter than the "
            f"{window}-session window, so there is no trailing estimate to plot."
        )

    ui_theme.spacer(0.3)

    ui.chart(
        ch.rolling_correlation_chart(
            corr_series,
            label=f"Correlation with {name}",
            window=window,
        )
    )

    corr_low, corr_high, corr_last = _span(corr_series)
    if _ok(corr_low) and _ok(corr_high):
        st.caption(
            f"Beta answers *by how much*; correlation answers *how reliably*, and the "
            f"second question is the one that decides whether the first is worth "
            f"quoting. Correlation with {name} ranged {fmt_num(corr_low, 2)} to "
            f"{fmt_num(corr_high, 2)} against a full-window {fmt_num(correlation, 2)}, "
            f"last reading {fmt_num(corr_last, 2)}. Since R-squared is exactly the "
            f"square of this line, a drop to {fmt_num(corr_low, 2)} means the index "
            f"explained only {fmt_ratio_pct(corr_low**2)} of daily variance in that "
            f"stretch. Both panels are backward-looking: each point uses the previous "
            f"{window} sessions, so a genuine regime change shows up late by up to a "
            f"full window."
        )
    else:
        st.caption(
            f"No rolling correlation is available for this window -- the overlap is "
            f"shorter than the {window}-session look-back."
        )


# ---------------------------------------------------------------------------
# 7. side-by-side comparison
# ---------------------------------------------------------------------------
def _comparison_frame(bench: dict, cfg: Any) -> pd.DataFrame:
    """Portfolio against index on the four numbers a mandate is judged on."""
    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    rate = _f(getattr(cfg, "risk_free_rate", float("nan")))

    p_ret, b_ret = _f(bench.get("portfolio_return")), _f(bench.get("benchmark_return"))
    p_vol, b_vol = _f(bench.get("portfolio_vol")), _f(bench.get("benchmark_vol"))
    p_mdd, b_mdd = _f(bench.get("portfolio_mdd")), _f(bench.get("benchmark_mdd"))
    p_sharpe, b_sharpe = _sharpe(p_ret, p_vol, rate), _sharpe(b_ret, b_vol, rate)

    rows = [
        (
            "Annualised return",
            fmt_ratio_pct(p_ret),
            fmt_ratio_pct(b_ret),
            fmt_ratio_pct(p_ret - b_ret, signed=True),
        ),
        (
            "Annualised volatility",
            fmt_ratio_pct(p_vol),
            fmt_ratio_pct(b_vol),
            fmt_ratio_pct(p_vol - b_vol, signed=True),
        ),
        (
            "Max drawdown",
            fmt_ratio_pct(p_mdd),
            fmt_ratio_pct(b_mdd),
            fmt_ratio_pct(p_mdd - b_mdd, signed=True),
        ),
        (
            f"Sharpe ratio (rf {fmt_ratio_pct(rate)})",
            fmt_num(p_sharpe, 2),
            fmt_num(b_sharpe, 2),
            fmt_signed(p_sharpe - b_sharpe, 2),
        ),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Portfolio", name, "Difference"])


def _side_by_side(bench: dict, cfg: Any) -> None:
    ui_theme.section("SIDE BY SIDE", "\u2696")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    rate = _f(getattr(cfg, "risk_free_rate", float("nan")))
    p_ret, b_ret = _f(bench.get("portfolio_return")), _f(bench.get("benchmark_return"))
    p_vol, b_vol = _f(bench.get("portfolio_vol")), _f(bench.get("benchmark_vol"))
    p_mdd, b_mdd = _f(bench.get("portfolio_mdd")), _f(bench.get("benchmark_mdd"))
    p_sharpe, b_sharpe = _sharpe(p_ret, p_vol, rate), _sharpe(b_ret, b_vol, rate)
    m2 = _f(bench.get("m2"))

    ui.dataframe(_comparison_frame(bench, cfg))

    if _ok(p_sharpe) and _ok(b_sharpe):
        st.caption(
            f"Both Sharpe ratios are computed here from the numbers in this table -- "
            f"`(annualised return - {fmt_ratio_pct(rate)}) / annualised volatility` -- on "
            f"the same overlapping dates and the same risk-free rate, which is the only "
            f"way the comparison means anything. The portfolio's "
            f"{fmt_num(p_sharpe, 2)} against {name}'s {fmt_num(b_sharpe, 2)} says the "
            f"book was paid {fmt_num(abs(p_sharpe - b_sharpe), 2)} "
            f"{'more' if p_sharpe >= b_sharpe else 'less'} per unit of total volatility "
            f"than simply owning the index. The Sharpe on the Overview page uses the "
            f"portfolio's full price history rather than this overlap, so a small "
            f"difference between the two is expected, not an error."
        )

    if _ok(p_vol) and _ok(b_vol) and _ok(p_mdd) and _ok(b_mdd):
        st.caption(
            f"Volatility and drawdown are two different readings of the same risk. The "
            f"book ran at {fmt_ratio_pct(p_vol)} against the index's "
            f"{fmt_ratio_pct(b_vol)} day to day, and its worst peak-to-trough fall was "
            f"{fmt_ratio_pct(p_mdd)} against {fmt_ratio_pct(b_mdd)}. Both drawdowns are "
            f"measured inside this window only -- a longer window will usually find a "
            f"deeper one for both legs, so neither number travels."
        )

    stats: list[tuple[str, str]] = [
        ("Treynor ratio", fmt_ratio_pct(bench.get("treynor"))),
        ("M2 (risk-matched return)", fmt_ratio_pct(m2)),
        ("M2 alpha vs index", fmt_ratio_pct(m2 - b_ret, signed=True)),
        ("Correlation", fmt_num(bench.get("correlation"), 2)),
        (
            "Overlapping sessions",
            f"{int(_f(bench.get('n_days'))) if _ok(bench.get('n_days')) else 0:,}",
        ),
    ]
    ui.stat_grid(stats, columns=5)

    if _ok(m2) and _ok(b_ret):
        st.caption(
            f"M2 restates the portfolio's Sharpe ratio as a return: "
            f"{fmt_ratio_pct(m2)} is what the book would have earned had it been "
            f"levered or de-levered with cash until its volatility matched {name}'s. "
            f"Against the index's own {fmt_ratio_pct(b_ret)} that is an M2 alpha of "
            f"{fmt_ratio_pct(m2 - b_ret, signed=True)} -- the same information as the "
            f"Sharpe gap, expressed in units a client actually recognises."
        )


# ---------------------------------------------------------------------------
# 8. relative strength
# ---------------------------------------------------------------------------
def _relative_strength(result: Any, cfg: Any) -> None:
    ui_theme.section("RELATIVE STRENGTH", "\U0001f501")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    rs = bm.relative_strength(result.values, result.benchmark_prices)
    ui.chart(ch.relative_strength_chart(rs, name))

    if rs is None or len(rs) < 2:
        st.caption(
            "The portfolio and the index share too few priced dates to form a "
            "relative-strength ratio."
        )
        return

    last = _f(rs.iloc[-1])
    low, high = _f(rs.min()), _f(rs.max())
    st.caption(
        f"RS = (V_t / V_0) / (B_t / B_0) x 100, so the line rises whenever the "
        f"portfolio beats {name} and falls whenever it lags -- regardless of which way "
        f"the market went. In a falling market a rising line simply means the book "
        f"lost less. The level itself carries no information; only distance from the "
        f"100 parity line does. It last read {fmt_num(last, 1)}, i.e. "
        f"{fmt_ratio_pct(last / 100.0 - 1.0, signed=True)} of cumulative relative "
        f"performance since {fmt_date(rs.index[0])}, having ranged "
        f"{fmt_num(low, 1)} to {fmt_num(high, 1)} along the way."
    )
    st.caption(
        "This is the one number on the page recomputed rather than read off the "
        "analysis result -- it is a pure re-expression of the portfolio value series "
        "and the index series the run already produced, on their overlapping dates."
    )


# ---------------------------------------------------------------------------
# 9. methodology
# ---------------------------------------------------------------------------
def _methodology(bench: dict, cfg: Any) -> None:
    ui_theme.section("CONVENTIONS", "\U0001f4d0")

    name = str(getattr(cfg, "benchmark_name", "Benchmark"))
    symbol = str(getattr(cfg, "benchmark_symbol", ""))
    rate = _f(getattr(cfg, "risk_free_rate", float("nan")))
    overlap = int(_f(bench.get("n_days"))) if _ok(bench.get("n_days")) else 0

    ui.methodology(
        "Alpha, tracking error, information ratio, Treynor and M2",
        f"""
**Alpha (verbatim from the analytics layer).**
{bench.get("alpha_methodology") or "No alpha methodology string was recorded for this run."}

**Everything is computed on the overlap.** Every statistic on this page uses the
inner join of the two series -- {overlap} dates here. Comparing an Indian book
with a US index across misaligned calendars otherwise measures the holiday
schedule rather than the manager.

**Tracking error** = `stdev(a_t, ddof=1) x sqrt({TRADING_DAYS})` where
`a_t = r_portfolio,t - r_index,t` on those overlapping daily returns. It is the
risk any index-relative mandate is actually judged on, and it is
direction-blind: a manager who beats the index erratically and one who trails it
erratically score identically. `ddof=1` because a price history is a sample, not
a population. An index fund's tracking error is 0, not a small number.

**Information ratio** = `mean(a) / stdev(a, ddof=1) x sqrt({TRADING_DAYS})` --
the Sharpe ratio of the active bet. The numerator is the *arithmetic*
annualised active return, deliberately matched to the arithmetic dispersion
below it; the **excess return** card is *geometric*. The two are both correct
and will not agree to the last decimal, which is why the convention is stated
rather than left to be inferred. It is undefined (dash), not infinite, when
tracking error is zero.

**Treynor ratio** = `(R_p - R_f) / beta`, with
R_f = {fmt_ratio_pct(rate)}. Sharpe divides by *total* volatility; Treynor
divides by beta alone, which is the right denominator only when this portfolio
is one sleeve of a larger diversified whole and its idiosyncratic risk has
already been diversified away at the parent level. For a standalone book,
Sharpe is the more honest ranking. A near-zero beta makes it explode, and a
negative beta makes it near-uninterpretable; neither is suppressed here.

**M2 (Modigliani)** = `R_f + Sharpe_portfolio x sigma_index`. It answers "what
would this portfolio have returned at the index's volatility?" and is therefore
directly comparable with the index's own return -- the difference is the M2
alpha shown above. It is the Sharpe ratio re-expressed as a return, so it adds
no new information, only legibility.

**The benchmark is price-only, and that matters.** {name} ({symbol}) is an index
*level*: it excludes the dividends its constituents paid. The portfolio's value
series is built from adjusted closes but carries no cash dividends either, so
both legs understate total return -- and the comparison flatters the index
slightly for any portfolio yielding more than the index does. On an Indian
large-cap book against a price-only NIFTY, that gap is roughly one to one and a
half percentage points a year. Nothing on this page corrects for it.

**Beta, alpha and R-squared are one package.** Alpha is a CAPM residual, so it
inherits every assumption of the fit that produced the beta -- a single
full-window estimate, no confidence interval, and no correction for the fact
that the rolling panel above shows the beta moving. When R-squared is below
{fmt_ratio_pct(MEANINGFUL_R_SQUARED, 0)} the decomposition is describing noise
and should be read as such.

**Risk-free rate.** {fmt_ratio_pct(rate)}, taken from the sidebar and applied
identically to alpha, Treynor, M2 and both Sharpe ratios. Change it there and
every risk-adjusted number on this page moves together.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Benchmark",
        "What the index handed over, and what the portfolio earned on its own",
        icon="\U0001f19a",
    )

    result = guards.require_result()
    if result is None:
        return

    bench = guards.require_benchmark(result)
    if bench is None:
        return

    cfg = result.config

    _context(result, bench, cfg)

    ui_theme.section("RELATIVE PERFORMANCE", "\U0001f3af")
    _headline_kpis(bench, cfg)
    ui_theme.spacer(0.4)
    _relative_risk_kpis(result, bench, cfg)

    ui_theme.spacer(0.5)
    _growth(bench, cfg)

    ui_theme.spacer(0.4)
    _capture(bench, cfg)

    ui_theme.spacer(0.4)
    _stability(result, bench, cfg)

    ui_theme.spacer(0.4)
    _side_by_side(bench, cfg)

    ui_theme.spacer(0.4)
    _relative_strength(result, cfg)

    ui_theme.spacer(0.4)
    _methodology(bench, cfg)
