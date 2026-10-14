"""Risk analytics - what this book can lose, how often, and where it comes from.

Five questions, one per tab:

* **Summary** - how much dispersion is being carried, and what each unit of it
  actually bought once the risk-free rate is taken out.
* **Tail risk** - how bad an ordinary-bad day is, how bad the days beyond it
  get, and whether the VaR model making that claim is even calibrated.
* **Drawdowns** - how deep the holes were and, just as importantly, how long
  the portfolio had to sit in them.
* **Rolling** - whether any of the above was stable through the window, or
  whether one regime is doing all the work.
* **Contribution** - which holdings manufacture the volatility, as opposed to
  which ones merely hold the capital.

Everything is read off the completed :class:`~src.models.AnalysisResult`. Two
recomputations happen here and both are deliberate: the Kupiec
proportion-of-failures test, which is a *diagnostic on* ``result.risk["var"]``
rather than a new risk number, and the drawdown descriptives that summarise
``result.values`` against the episodes the pipeline already found.

Units are the standing trap on this page and are stated everywhere they matter:
``result.risk`` is decimals throughout (``var`` and ``cvar`` are NEGATIVE
decimals describing a one-day loss) except ``ulcer``, which is already percent,
``hit_ratio``, which is a fraction, and ``n_obs``/``confidence``. The percent
columns in ``result.attribution`` are already percent.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import drawdown as dd
from src.analytics import risk as rsk
from src.analytics.insights import RISK_WEIGHT_RATIO
from src.config import INSIGHTS, RISK_RUBRIC, TRADING_DAYS
from src.ui import components as ui
from src.ui import guards
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    fmt_date,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    humanize_days,
)

#: Trading sessions in an average calendar month, used only to translate a
#: rolling window into a duration a human recognises.
_MONTH_SESSIONS = TRADING_DAYS / 12.0

#: Rolling columns shown stacked on one shared time axis.
_ROLLING_STACK: tuple[str, ...] = ("volatility", "sharpe", "beta")

#: Gap between two VaR estimators, in basis points of a daily return, below
#: which the difference is not worth narrating. Two estimators five basis points
#: apart on a one-day loss are agreeing, whatever the sign of the difference.
_VAR_GAP_BP = 10.0

#: Absolute skewness below which the distribution is called symmetric rather
#: than left- or right-tailed. A skew of -0.01 is a rounding artefact, not a
#: long left tail, and saying otherwise is the kind of over-reading this app
#: exists to avoid.
_SKEW_NEUTRAL = 0.20


# ---------------------------------------------------------------------------
# small local helpers (display only)
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    """True when ``x`` is a real, finite number worth printing."""
    try:
        return bool(pd.notna(x)) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x: Any) -> float:
    """Best-effort float; NaN for anything unusable."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _confidence(result: Any, cfg: Any) -> float:
    """VaR confidence level actually used by this run, as a decimal."""
    level = _f((result.risk or {}).get("confidence"))
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        level = _f(getattr(cfg, "var_confidence", 0.95))
    return level if math.isfinite(level) else 0.95


def _conf_label(level: float) -> str:
    return fmt_pct(level * 100.0, 0) if _finite(level) else DASH


def _tail_label(level: float) -> str:
    return fmt_pct((1.0 - level) * 100.0, 0) if _finite(level) else DASH


def _money_of(fraction: Any, result: Any, cfg: Any) -> str:
    """A decimal return translated into base-currency money on today's book."""
    value = _f((result.summary or {}).get("current_value"))
    move = _f(fraction)
    if not (math.isfinite(value) and math.isfinite(move)):
        return DASH
    return fmt_money(abs(move) * value, getattr(cfg, "base_currency", "INR"), 0)


def _ratio_tone(value: Any, good: float, poor: float = 0.0) -> str:
    """Green at or above ``good``, red below ``poor``, amber in between."""
    if not _finite(value):
        return "muted"
    number = float(value)
    if number >= good:
        return "gain"
    return "loss" if number < poor else "warn"


def _vol_tone(ann_vol: Any) -> str:
    if not _finite(ann_vol):
        return "muted"
    value = float(ann_vol)
    if value >= RISK_RUBRIC.vol_cap:
        return "loss"
    return "warn" if value >= RISK_RUBRIC.vol_cap / 2.0 else "neutral"


def _drawdown_tone(mdd: Any) -> str:
    if not _finite(mdd):
        return "muted"
    value = float(mdd)
    if value <= INSIGHTS.severe_drawdown:
        return "loss"
    return "warn" if value <= INSIGHTS.deep_drawdown else "neutral"


def _var_tone(var: Any) -> str:
    if not _finite(var):
        return "muted"
    loss_pct = abs(float(var)) * 100.0
    if loss_pct >= INSIGHTS.high_var_pct:
        return "loss"
    return "warn" if loss_pct >= INSIGHTS.high_var_pct / 2.0 else "neutral"


def _is_frame(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame) and not obj.empty


def _has_values(frame: Any, column: str) -> bool:
    """True when ``column`` exists and holds at least two real numbers."""
    if not _is_frame(frame) or column not in frame.columns:
        return False
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum()) >= 2


def _window_phrase(window: int) -> str:
    months = float(window) / _MONTH_SESSIONS if _MONTH_SESSIONS else float("nan")
    return f"{int(window)} trading days (about {fmt_num(months, 1)} calendar months)"


# ---------------------------------------------------------------------------
# tab 1 - summary
# ---------------------------------------------------------------------------
def _dispersion_kpis(result: Any, cfg: Any) -> None:
    risk = result.risk or {}
    rate = _f(getattr(cfg, "risk_free_rate", 0.0))
    rate_label = fmt_ratio_pct(rate)

    ann_vol = risk.get("ann_vol")
    sharpe = risk.get("sharpe")
    sortino = risk.get("sortino")
    calmar = risk.get("calmar")

    ui.kpi_row(
        [
            {
                "label": "Annualised volatility",
                "value": fmt_ratio_pct(ann_vol),
                "tone": _vol_tone(ann_vol),
                "footnote": (
                    f"Daily moves scaled by sqrt({TRADING_DAYS}), over "
                    f"{int(_f(risk.get('n_obs')) or 0)} return observations"
                ),
                "help_text": (
                    "Sample (n-1) standard deviation of daily returns times "
                    f"sqrt({TRADING_DAYS}). It is dispersion in both directions, "
                    "not a loss estimate."
                ),
            },
            {
                "label": "Sharpe ratio",
                "value": fmt_num(sharpe, 2),
                "tone": _ratio_tone(sharpe, INSIGHTS.good_sharpe),
                "footnote": f"Excess over {rate_label}, per unit of TOTAL volatility",
                "help_text": (
                    "(annualised return - risk-free) / annualised volatility. It "
                    "punishes an upside spike exactly as hard as a crash."
                ),
            },
            {
                "label": "Sortino ratio",
                "value": fmt_num(sortino, 2),
                "tone": _ratio_tone(sortino, INSIGHTS.good_sortino),
                "footnote": (
                    f"Same excess, divided by downside deviation "
                    f"{fmt_ratio_pct(risk.get('downside_dev'))}"
                ),
                "help_text": (
                    "(annualised return - risk-free) / downside deviation. Only "
                    "returns below the 0% minimum acceptable return count in the "
                    "denominator."
                ),
            },
            {
                "label": "Calmar ratio",
                "value": fmt_num(calmar, 2),
                "tone": _ratio_tone(calmar, INSIGHTS.weak_calmar),
                "footnote": (
                    f"Annualised return per unit of the {fmt_ratio_pct(risk.get('mdd'))} worst fall"
                ),
                "help_text": (
                    "annualised return / |max drawdown|. Return measured against "
                    "the single worst loss the portfolio actually lived through."
                ),
            },
        ],
        columns=4,
    )


def _quality_kpis(result: Any, cfg: Any) -> None:
    risk = result.risk or {}
    rate_label = fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))

    omega = risk.get("omega")
    ulcer = risk.get("ulcer")
    martin = risk.get("martin")
    downside = risk.get("downside_dev")
    ann_vol = _f(risk.get("ann_vol"))
    down = _f(downside)
    asymmetry = (
        f"{fmt_num(down / ann_vol, 2)}x of total volatility is downside"
        if math.isfinite(down) and math.isfinite(ann_vol) and ann_vol > 0
        else "Annualised semi-deviation below a 0% threshold"
    )

    ui.kpi_row(
        [
            {
                "label": "Omega ratio",
                "value": fmt_num(omega, 2),
                "tone": _ratio_tone(omega, 1.5, 1.0),
                "footnote": "Gains above 0% divided by losses below it",
                "help_text": (
                    "Probability-weighted gain/loss ratio about a 0% threshold. "
                    "1.00 is the break-even point; it uses the whole distribution, "
                    "not just its first two moments."
                ),
            },
            {
                "label": "Ulcer index",
                "value": fmt_pct(ulcer),
                "tone": "warn" if _finite(ulcer) and float(ulcer) >= 10.0 else "neutral",
                "footnote": "Root-mean-square drawdown - already in percent",
                "help_text": (
                    "sqrt(mean(d_t^2)) over the drawdown path, in percent. Unlike "
                    "max drawdown it punishes long shallow pain, not just one deep hole."
                ),
            },
            {
                "label": "Martin ratio",
                "value": fmt_num(martin, 2),
                "tone": _ratio_tone(martin, 1.0),
                "footnote": f"Excess over {rate_label} per unit of Ulcer",
                "help_text": (
                    "The ulcer performance index: (annualised return - risk-free) "
                    "/ ulcer index. Sharpe for people who care about time spent "
                    "underwater rather than day-to-day wobble."
                ),
            },
            {
                "label": "Downside deviation",
                "value": fmt_ratio_pct(downside),
                "tone": "neutral",
                "footnote": asymmetry,
                "help_text": (
                    "Annualised standard deviation of returns below the minimum "
                    "acceptable return (0% here). The denominator of Sortino."
                ),
            },
        ],
        columns=4,
    )


def _summary_reading(result: Any, cfg: Any) -> None:
    risk = result.risk or {}
    sharpe = _f(risk.get("sharpe"))
    sortino = _f(risk.get("sortino"))
    ann_vol = _f(risk.get("ann_vol"))
    rate_label = fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))

    if not math.isfinite(sharpe):
        st.caption(
            "Sharpe could not be computed for this window - either the return or "
            "the volatility leg was undefined, and a fabricated ratio is worse "
            "than none."
        )
        return

    if sharpe >= 0:
        lead = (
            f"Sharpe {fmt_num(sharpe, 2)}: each unit of volatility bought "
            f"{fmt_num(sharpe, 2)} units of return above the {rate_label} "
            f"risk-free rate."
        )
    else:
        lead = (
            f"Sharpe {fmt_num(sharpe, 2)}: below zero, so the "
            f"{fmt_ratio_pct(ann_vol)} annualised volatility was carried for "
            f"nothing - cash at {rate_label} beat this portfolio over the window."
        )

    if math.isfinite(sortino) and math.isfinite(sharpe):
        if sortino > sharpe:
            gap = (
                f"Sortino {fmt_num(sortino, 2)} sits above Sharpe, which means a "
                "meaningful part of the volatility was upside - dispersion Sharpe "
                "penalised and Sortino correctly ignored."
            )
        else:
            gap = (
                f"Sortino {fmt_num(sortino, 2)} is no better than Sharpe, so the "
                "volatility in this window was mostly the kind that hurt: the "
                "downside leg carries the distribution."
            )
    else:
        gap = ""

    st.caption(
        f"{lead} {gap} All four ratios describe these "
        f"{int(_f(risk.get('n_obs')) or 0)} daily observations and nothing "
        "outside them - measurements, not forecasts."
    )


def _summary_methodology(result: Any, cfg: Any) -> None:
    risk = result.risk or {}
    rate = _f(getattr(cfg, "risk_free_rate", 0.0))
    rate_label = fmt_ratio_pct(rate)

    ui.methodology(
        "Every ratio on this tab, written out",
        f"""
The risk-free rate used throughout is **{rate_label} annualised**, taken from
the sidebar (`config.DEFAULT_RISK_FREE` seeds it per currency). It is a
*setting*, not a live yield - change it and every ratio below moves, which is
exactly why it is stated here rather than buried.

| Statistic | Formula | Units | Read it as |
| --- | --- | --- | --- |
| Annualised volatility | `stdev(r, ddof=1) * sqrt({TRADING_DAYS})` | decimal | width of the daily distribution |
| Sharpe | `(R_a - Rf) / sigma` | ratio | return per unit of total dispersion |
| Sortino | `(R_a - Rf) / downside deviation` | ratio | return per unit of *harmful* dispersion |
| Calmar | `R_a / abs(max drawdown)` | ratio | return per unit of worst realised loss |
| Omega | `sum(gains above MAR) / sum(losses below MAR)` | ratio | whole-distribution gain/loss odds |
| Ulcer index | `sqrt(mean(d_t^2))`, d in percent | **percent** | depth *and* duration of pain |
| Martin | `(R_a - Rf) / ulcer index` | ratio | return per unit of underwater time |
| Downside deviation | semi-deviation below MAR, annualised | decimal | the Sortino denominator |

`R_a` is the geometric annualised return of the portfolio value series
({fmt_ratio_pct(risk.get("ann_return"))} here), `sigma` its annualised
volatility ({fmt_ratio_pct(risk.get("ann_vol"))}), and the minimum acceptable
return (MAR) for Omega and the downside leg is **0%**, not the risk-free rate -
a convention worth knowing, because raising the MAR to {rate_label} would lower
both.

Caveats that matter more than the numbers:

* **The square-root-of-time scaling assumes independent daily returns.** Real
  markets trend and mean-revert, so annualised volatility is a convention, not
  a measurement of a one-year distribution.
* **Sharpe assumes roughly symmetric returns.** With skew
  {fmt_num(risk.get("skew"), 2)} and excess kurtosis
  {fmt_num(risk.get("kurtosis"), 2)} in this window, it is describing a
  distribution it does not fully fit - see the Tail risk tab.
* **Calmar and the Ulcer index depend on window length.** A longer history
  almost always finds a deeper drawdown, so neither is comparable across
  windows of different sizes.
* **Every ratio is a single point estimate from {int(_f(risk.get("n_obs")) or 0)}
  observations.** Not one of them carries a confidence interval here, and the
  shorter the window the more one violent fortnight decides the answer.
        """,
    )


def _score_block(result: Any) -> None:
    score = result.score or {}
    value = score.get("score")
    band = str(score.get("band", "") or "")
    components = score.get("components") or {}

    left, right = st.columns([1.0, 1.1], gap="large")

    with left:
        ui.chart(ch.risk_gauge(value, band, height=320))

    with right:
        if not _finite(value):
            ui.empty_state(
                "Not graded",
                "The rubric refuses to score a portfolio whose volatility or "
                "drawdown could not be measured. Nothing is substituted for the "
                "missing input.",
                icon="\U0001f6ab",
            )
        else:
            st.caption(
                f"**{fmt_num(value, 0)} / 100 - {band}.** Scored out of "
                f"{fmt_num(score.get('max_points'), 0)} *measurable* points: any "
                "component whose input was missing is dropped from the numerator "
                "and the denominator both, so a missing tail estimate cannot "
                "flatter the result. Below "
                f"{fmt_num(RISK_RUBRIC.low_band, 0)} reads Low, above "
                f"{fmt_num(RISK_RUBRIC.high_band, 0)} reads High."
            )
        if components:
            frame = pd.DataFrame(
                {
                    "Component (max points)": [str(k) for k in components],
                    "Earned": [_f(v) for v in components.values()],
                }
            )
            ui.dataframe(
                frame,
                column_config={
                    "Earned": st.column_config.NumberColumn("Earned", format="%.1f", width="small")
                },
            )
            st.caption("Raw points before rescaling, so the arithmetic stays inspectable.")

    ui.methodology(
        "The rubric, in full",
        f"""
A **transparent rule-based heuristic**, not a regulatory measure and not
calibrated against anyone's loss experience. Every weight lives in
`config.RISK_RUBRIC`:

| Component | Max points | Full points at |
| --- | --- | --- |
| Annualised volatility | {RISK_RUBRIC.vol_points:g} | {fmt_ratio_pct(RISK_RUBRIC.vol_cap, 0)} annualised |
| Max drawdown | {RISK_RUBRIC.drawdown_points:g} | {fmt_ratio_pct(RISK_RUBRIC.drawdown_cap, 0)} peak-to-trough |
| Concentration (HHI) | {RISK_RUBRIC.concentration_points:g} | one name holding everything |
| Holdings count | {RISK_RUBRIC.holdings_points:g} | a single position ({RISK_RUBRIC.holdings_floor}+ names score zero) |
| Tail loss (daily CVaR) | {RISK_RUBRIC.tail_points:g} | {fmt_ratio_pct(RISK_RUBRIC.tail_cap, 0)} expected shortfall |

Each component scales linearly between zero and its cap, and the reported score
is `earned / available * 100`.

**Higher is not "worse investment".** A concentrated, volatile book held
deliberately by someone who can sit through it scores high here and may still be
the right portfolio. The number grades *exposure*, never suitability.
        """,
    )


def _summary_tab(result: Any, cfg: Any) -> None:
    ui_theme.section("Dispersion and risk-adjusted return", "\U0001f4d0")
    _dispersion_kpis(result, cfg)
    ui_theme.spacer(0.3)
    _quality_kpis(result, cfg)
    _summary_reading(result, cfg)
    _summary_methodology(result, cfg)

    ui_theme.spacer(0.5)
    ui_theme.section("Composite risk score", "\U0001f39a")
    _score_block(result)


# ---------------------------------------------------------------------------
# tab 2 - tail risk
# ---------------------------------------------------------------------------
def _skew_tone(skew: Any) -> str:
    if not _finite(skew):
        return "muted"
    return "warn" if float(skew) <= -_SKEW_NEUTRAL else "neutral"


def _skew_note(skew: Any) -> str:
    if not _finite(skew):
        return "Undefined for this sample"
    value = float(skew)
    if value <= -_SKEW_NEUTRAL:
        return "A long left tail: many small gains against occasional large losses"
    if value >= _SKEW_NEUTRAL:
        return "The long tail is on the upside - unusual for an equity book"
    return f"Within {fmt_num(_SKEW_NEUTRAL, 2)} of zero, so effectively symmetric"


def _kurtosis_note(kurtosis: Any) -> str:
    if not _finite(kurtosis):
        return "Undefined for this sample"
    value = float(kurtosis)
    if value >= INSIGHTS.fat_tail_kurtosis:
        return (
            f"Past the {fmt_num(INSIGHTS.fat_tail_kurtosis, 0)} mark: fat enough "
            "to break normality-based risk maths"
        )
    if value > 0.0:
        return (
            f"Fatter tails than a normal curve, but short of the "
            f"{fmt_num(INSIGHTS.fat_tail_kurtosis, 0)} fat-tail mark"
        )
    return "Thinner tails than a normal curve over this particular window"


def _tail_kpis(result: Any, cfg: Any, level: float) -> None:
    risk = result.risk or {}
    var = risk.get("var")
    cvar = risk.get("cvar")
    skew = risk.get("skew")
    kurtosis = risk.get("kurtosis")

    ui.kpi_row(
        [
            {
                "label": f"Daily VaR {_conf_label(level)}",
                "value": fmt_ratio_pct(var),
                "tone": _var_tone(var),
                "footnote": (
                    f"About {_money_of(var, result, cfg)} on today's book, in one session"
                ),
                "help_text": (
                    f"Historical {_tail_label(level)} quantile of the daily returns "
                    "in this window. No distribution is assumed."
                ),
            },
            {
                "label": f"CVaR / expected shortfall {_conf_label(level)}",
                "value": fmt_ratio_pct(cvar),
                "tone": _var_tone(cvar),
                "footnote": (
                    f"Average of the worst {_tail_label(level)} of days - "
                    f"{_money_of(cvar, result, cfg)}"
                ),
                "help_text": (
                    "The mean loss GIVEN that VaR was breached. VaR says where the "
                    "tail starts; CVaR says how bad it is inside."
                ),
            },
            {
                "label": "Skewness",
                "value": fmt_num(skew, 2),
                "tone": _skew_tone(skew),
                "footnote": _skew_note(skew),
                "help_text": (
                    "Sample (Fisher-Pearson, bias-corrected) third moment of daily "
                    "returns. Equity portfolios are usually negatively skewed."
                ),
            },
            {
                "label": "Excess kurtosis",
                "value": fmt_num(kurtosis, 2),
                "tone": (
                    "loss"
                    if _finite(kurtosis) and float(kurtosis) >= INSIGHTS.fat_tail_kurtosis
                    else "neutral"
                ),
                "footnote": _kurtosis_note(kurtosis),
                "help_text": (
                    "Fisher definition, so a normal distribution scores exactly 0. "
                    "Positive means more probability mass in both tails than a "
                    "bell curve allows."
                ),
            },
        ],
        columns=4,
    )


def _var_comparison(result: Any, cfg: Any, level: float) -> None:
    risk = result.risk or {}
    hist = _f(risk.get("var"))
    gauss = _f(risk.get("var_gaussian"))
    cornish = _f(risk.get("var_cornish_fisher"))
    fell_back = (
        math.isfinite(hist)
        and math.isfinite(cornish)
        and math.isclose(hist, cornish, rel_tol=1e-9, abs_tol=1e-12)
    )

    frame = pd.DataFrame(
        {
            "Estimator": [
                "Historical (empirical quantile)",
                "Gaussian (mu + z * sigma)",
                "Cornish-Fisher (skew / kurtosis corrected)",
            ],
            # the risk dict is in decimals; the column renders as percent
            "One-day VaR": [hist * 100.0, gauss * 100.0, cornish * 100.0],
            "On today's book": [
                _money_of(hist, result, cfg),
                _money_of(gauss, result, cfg),
                _money_of(cornish, result, cfg),
            ],
            "What it assumes": [
                "Nothing about shape - but it can never report a loss worse "
                "than the worst day already observed.",
                "Returns are normal. That is the assumption equity markets break most reliably.",
                (
                    "Fell back to the historical quantile: the expansion stopped "
                    "being monotone at this quantile."
                    if fell_back
                    else "Normal quantile corrected for the sample's own third and fourth moments."
                ),
            ],
        }
    )

    ui.dataframe(
        frame,
        column_config={
            "One-day VaR": st.column_config.NumberColumn(
                f"One-day VaR ({_conf_label(level)})",
                format="%.2f%%",
                help="Negative decimals shown as percent: -2.10% is a 2.1% loss.",
            ),
            "What it assumes": st.column_config.TextColumn(width="large"),
        },
    )

    if math.isfinite(hist) and math.isfinite(gauss):
        gap_bp = abs(hist - gauss) * 10000.0
        if gap_bp < _VAR_GAP_BP:
            verdict = (
                f"The historical ({fmt_ratio_pct(hist)}) and Gaussian "
                f"({fmt_ratio_pct(gauss)}) estimates agree to within "
                f"{fmt_num(gap_bp, 0)} basis points of a daily return. At this "
                "quantile the observed left tail is about what a normal fit "
                "implies - which is a statement about the ordinary-bad days that "
                "fell inside this window, not about the extremes that did not."
            )
        elif hist < gauss:
            verdict = (
                f"The historical estimate ({fmt_ratio_pct(hist)}) is **worse** than "
                f"the Gaussian one ({fmt_ratio_pct(gauss)}), by "
                f"{fmt_num(gap_bp, 0)} basis points. That gap is direct evidence of "
                "a fat left tail: the market actually delivered losses at this "
                "frequency that a normal distribution fitted to the same mean and "
                "standard deviation says should be rarer."
            )
        else:
            verdict = (
                f"The historical estimate ({fmt_ratio_pct(hist)}) is milder than the "
                f"Gaussian one ({fmt_ratio_pct(gauss)}) by {fmt_num(gap_bp, 0)} "
                "basis points, so at this quantile the observed left tail is "
                "thinner than a normal fit implies. Read it as a fact about this "
                "window only - a 95% quantile is silent about the 1-in-500 day."
            )
    else:
        verdict = (
            "Not every estimator could be computed on this window, so the comparison is incomplete."
        )

    cf_note = (
        " Cornish-Fisher fell back to the historical quantile here: the expansion "
        "is asymptotic in skew and kurtosis and is only a valid quantile transform "
        "while it stays monotone in z. One violent session among hundreds of quiet "
        "ones is enough to push it outside that domain, where it can return a "
        "*gain* as the worst case - so the code checks monotonicity and refuses to "
        "print it."
        if fell_back
        else " Cornish-Fisher corrects the normal quantile for the observed skew "
        "and excess kurtosis, pushing it further into the tail for the negatively "
        "skewed, fat-tailed distributions equities actually produce. It falls back "
        "to the historical quantile when the sample is extreme enough that the "
        "expansion stops being monotone."
    )

    st.caption(verdict + cf_note)


def _kupiec_block(result: Any, level: float) -> None:
    returns = result.returns
    var = _f((result.risk or {}).get("var"))

    if returns is None or not math.isfinite(var):
        st.caption(
            "The Kupiec test needs both a return series and a finite VaR estimate; "
            "one of them is missing for this window."
        )
        return

    test = rsk.kupiec_pof_test(returns, var, level)
    breaches = int(test.get("breaches", 0) or 0)
    expected = _f(test.get("expected_breaches"))
    n_obs = int(test.get("n", 0) or 0)
    lr = _f(test.get("lr_statistic"))
    p_value = _f(test.get("p_value"))
    rate = _f(test.get("breach_rate"))
    reject = bool(test.get("reject", False))

    ui.kpi_row(
        [
            {
                # colour tracks the test's own verdict, not the raw count: 30
                # breaches against 30.0 expected is a match, not an exceedance.
                "label": "Breaches observed",
                "value": f"{breaches:,}" if n_obs else DASH,
                "tone": (
                    "loss"
                    if reject and math.isfinite(expected) and breaches > expected
                    else "warn"
                    if reject
                    else "neutral"
                ),
                "footnote": f"out of {n_obs:,} sessions ({fmt_ratio_pct(rate)} of days)",
                "help_text": "Days whose return fell strictly below the VaR line.",
            },
            {
                "label": "Breaches expected",
                "value": fmt_num(expected, 1),
                "tone": "neutral",
                "footnote": (f"{_tail_label(level)} of {n_obs:,} sessions, if the model is honest"),
                "help_text": "n * (1 - confidence): what a calibrated VaR promises.",
            },
            {
                "label": "LR statistic",
                "value": fmt_num(lr, 3),
                "tone": "neutral",
                "footnote": "Unconditional-coverage likelihood ratio, chi-squared with 1 df",
                "help_text": (
                    "Zero when the observed breach rate exactly matches the "
                    "promised one; it grows in either direction."
                ),
            },
            {
                "label": "p-value",
                "value": fmt_num(p_value, 3),
                "tone": "loss" if reject else "gain",
                "footnote": (
                    f"Rejected at the {_tail_label(level)} level"
                    if reject
                    else f"Not rejected at the {_tail_label(level)} level"
                ),
                "help_text": (
                    "Probability of seeing a breach count at least this far from "
                    "the promise, if the model were correctly calibrated."
                ),
            },
        ],
        columns=4,
    )

    if not math.isfinite(p_value):
        reading = (
            "The likelihood ratio is undefined for this sample - typically zero "
            "breaches, where the alternative hypothesis has no maximum-likelihood "
            "estimate. That is a data limitation, not a passing grade."
        )
    elif reject:
        direction = "too many" if math.isfinite(expected) and breaches > expected else "too few"
        consequence = (
            "the VaR is under-stating risk and the loss you are budgeting for is "
            "smaller than the one the market has been handing out"
            if direction == "too many"
            else "the VaR is over-stating risk, which is not free either: it "
            "reserves capital against a loss that has not been arriving"
        )
        reading = (
            f"**Rejected.** {breaches} breaches against {fmt_num(expected, 1)} "
            f"expected is {direction} for a {_conf_label(level)} model at this "
            f"sample size (p = {fmt_num(p_value, 3)}). Read plainly: {consequence}."
        )
    else:
        reading = (
            f"**Not rejected.** {breaches} breaches against {fmt_num(expected, 1)} "
            f"expected is consistent with a calibrated {_conf_label(level)} model "
            f"(p = {fmt_num(p_value, 3)}). Note the asymmetry of the claim: failing "
            "to reject is not proof the model is right, only that "
            f"{n_obs:,} observations were not enough to convict it."
        )

    st.caption(reading)


def _tail_methodology(result: Any, cfg: Any, level: float) -> None:
    risk = result.risk or {}
    n_obs = int(_f(risk.get("n_obs")) or 0)
    book = fmt_money(
        (result.summary or {}).get("current_value"),
        getattr(cfg, "base_currency", "INR"),
        0,
    )
    ui.methodology(
        "Tail risk: definitions, sign convention and what the backtest can prove",
        f"""
**Sign convention.** VaR and CVaR are **negative decimals** describing a
**one-day** loss: {fmt_ratio_pct(risk.get("var"))} means "a
{fmt_pct(abs(_f(risk.get("var"))) * 100.0)} loss". Nothing is clamped, so an
overwhelmingly positive window can legitimately report a positive VaR - that is
information, not a bug.

**VaR** is the {_tail_label(level)} quantile of the {n_obs} daily returns in
this window. It answers "how bad is an ordinary-bad day" and deliberately says
**nothing** about the {_tail_label(level)} of days beyond it.

**CVaR (expected shortfall)** = the mean of exactly those days,
{fmt_ratio_pct(risk.get("cvar"))} here. It is the coherent risk measure of the
two - VaR is not sub-additive, so combining two books can report a VaR larger
than the sum of the parts, which is nonsense as a risk measure and the reason
regulators moved to expected shortfall.

**The three estimators** are all fitted to the same {n_obs} observations and
differ only in what they assume about shape: historical assumes nothing,
Gaussian assumes normality,
Cornish-Fisher adjusts the normal quantile using the sample skew
({fmt_num(risk.get("skew"), 2)}) and excess kurtosis
({fmt_num(risk.get("kurtosis"), 2)}).

**Kupiec proportion-of-failures.** A VaR estimate is a claim about *frequency*:
at {_conf_label(level)} confidence, {_tail_label(level)} of days should breach
it. With `p = 1 - confidence`, `x` breaches in `n` days and `pi = x / n`:

```
LR_uc = -2 * [ (n-x)*ln(1-p)  + x*ln(p)  ]
        +2 * [ (n-x)*ln(1-pi) + x*ln(pi) ]
```

asymptotically chi-squared with one degree of freedom. Three honest limits:

* **It counts breaches, it does not look at when they happened.** A model that
  fails ten days in a row in March and never again passes Kupiec and would fail
  a conditional-coverage (Christoffersen) test. Clustered breaches are the ones
  that actually blow up an account.
* **This is an in-sample backtest.** The VaR line was estimated on the very days
  it is being tested against, which biases the test toward passing. A real
  validation re-estimates VaR on a rolling window and tests the next day out of
  sample.
* **Too few breaches fail too.** An unbreached VaR is not a safe model, it is an
  expensive one - it over-reserves against losses that never arrive.

Rejection is evaluated at a significance level of {_tail_label(level)}, the same
tail probability the VaR itself uses, so no extra constant is smuggled in.
Portfolio value used for the money translations is {book}.
        """,
    )


def _tail_tab(result: Any, cfg: Any) -> None:
    level = _confidence(result, cfg)

    ui_theme.section("Where the tail starts and how bad it gets", "\U0001f4c9")
    _tail_kpis(result, cfg, level)

    ui.chart(
        ch.return_distribution(
            result.returns,
            (result.risk or {}).get("var"),
            (result.risk or {}).get("cvar"),
            confidence=level,
            height=420,
        )
    )
    st.caption(
        "The curve is a normal distribution fitted to this sample's own mean and "
        "standard deviation. Any bar standing above it in the left tail is loss "
        "frequency a Gaussian risk model would have under-priced - which is the "
        "entire argument for not using one."
    )

    ui_theme.spacer(0.4)
    ui_theme.section("Three estimators, one window", "\U0001f9ee")
    _var_comparison(result, cfg, level)

    ui_theme.spacer(0.4)
    ui_theme.section("Is the VaR model calibrated? Kupiec POF test", "\U0001f9ea")
    _kupiec_block(result, level)

    var = (result.risk or {}).get("var")
    if _finite(var):
        ui.chart(ch.var_backtest_chart(result.returns, var, confidence=level, height=380))
        st.caption(
            f"Every bar below the {_conf_label(level)} VaR line is a breach. Count "
            "matters, but so does spacing: breaches arriving evenly through the "
            "window suggest a stable model, breaches bunched into one fortnight say "
            "the model missed a volatility regime even when the total looks "
            "respectable."
        )

    _tail_methodology(result, cfg, level)


# ---------------------------------------------------------------------------
# tab 3 - drawdowns
# ---------------------------------------------------------------------------
def _episode_table(episodes: Any) -> pd.DataFrame:
    """Episodes rendered for display: dates as dates, depth in percent."""
    frame = dd.episodes_dataframe(episodes)
    if frame.empty:
        return frame
    return pd.DataFrame(
        {
            "Peak": pd.to_datetime(frame["peak_date"], errors="coerce"),
            "Trough": pd.to_datetime(frame["trough_date"], errors="coerce"),
            "Recovered on": pd.to_datetime(frame["recovery_date"], errors="coerce"),
            "Depth": pd.to_numeric(frame["depth_pct"], errors="coerce"),
            "Peak to trough": pd.to_numeric(frame["drawdown_days"], errors="coerce"),
            "Trough to recovery": pd.to_numeric(frame["recovery_days"], errors="coerce"),
            "Total": pd.to_numeric(frame["total_days"], errors="coerce"),
        }
    )


def _drawdown_kpis(result: Any, current: dict, tuw: float, recovery: float) -> None:
    risk = result.risk or {}
    episodes = list(result.drawdowns or [])
    mdd = risk.get("mdd")

    worst_note = "No completed peak-to-trough cycle in this window"
    if episodes:
        worst = episodes[0]
        span = f"{fmt_date(worst.peak_date)} to {fmt_date(worst.trough_date)}"
        worst_note = (
            f"{span}, back in {humanize_days(worst.recovery_days)}"
            if worst.recovered
            else f"{span}, never recovered inside this window"
        )

    depth = current.get("depth")
    days_under = current.get("days_underwater")
    if _finite(depth) and float(depth) < 0:
        current_note = (
            f"Below the {fmt_date(current.get('peak_date'))} high for {humanize_days(days_under)}"
        )
        current_tone = _drawdown_tone(depth)
    else:
        current_note = "Sitting at or on a new high-water mark"
        current_tone = "gain"

    recovered = [e for e in episodes if getattr(e, "recovered", False)]

    ui.kpi_row(
        [
            {
                "label": "Max drawdown",
                "value": fmt_ratio_pct(mdd),
                "tone": _drawdown_tone(mdd),
                "footnote": worst_note,
                "help_text": (
                    "min(V_t / running_max(V) - 1) on daily closes inside this window only."
                ),
            },
            {
                "label": "Current drawdown",
                "value": fmt_ratio_pct(depth),
                "tone": current_tone,
                "footnote": current_note,
                "help_text": (
                    "Where the last valuation sits against the all-time high in "
                    "force. 0.00% means the portfolio is at its peak."
                ),
            },
            {
                "label": "Time underwater",
                "value": fmt_pct(tuw),
                "tone": "warn" if _finite(tuw) and float(tuw) >= 60.0 else "neutral",
                "footnote": (f"of {result.n_days:,} sessions spent below a prior peak"),
                "help_text": (
                    "Share of OBSERVATIONS, not calendar days, with a negative "
                    "drawdown. The first observation is a peak by construction, so "
                    "it can never reach 100%."
                ),
            },
            {
                "label": "Average recovery",
                "value": humanize_days(recovery),
                "tone": (
                    "warn"
                    if _finite(recovery) and float(recovery) >= INSIGHTS.long_recovery_days
                    else "neutral"
                ),
                "footnote": (
                    f"Trough to new high, over {len(recovered)} of "
                    f"{len(episodes)} episodes that got back"
                ),
                "help_text": (
                    "Mean trough-to-recovery span in calendar days. Episodes still "
                    "underwater are excluded rather than counted as zero."
                ),
            },
        ],
        columns=4,
    )


def _drawdown_tab(result: Any, cfg: Any) -> None:
    values = result.values
    episodes = list(result.drawdowns or [])
    current = dd.current_drawdown(values) if values is not None else {}
    tuw = dd.time_underwater_pct(values) if values is not None else float("nan")
    recovery = dd.average_recovery_days(episodes)

    ui_theme.section("Depth, duration and where the portfolio stands now", "\U0001f30a")
    _drawdown_kpis(result, current, tuw, recovery)

    st.caption(
        "Depth alone is half the story. A 20% fall that is recovered inside a "
        "month is an inconvenience; the same 20% taking three years to come back "
        "is three years of capital doing nothing, three years of compounding lost, "
        "and - the part no statistic captures - three years of temptation to sell "
        "at the bottom. That is why *time* underwater sits on the same row as depth."
    )

    ui.chart(
        ch.drawdown_episodes_chart(
            values, episodes, currency_symbol=getattr(cfg, "currency_symbol", ""), height=420
        )
    )
    st.caption(
        "Each shaded band is one complete peak-to-trough-to-recovery cycle; a band "
        "running to the right edge is a hole the portfolio is still in. The cross "
        "marks the trough."
        if episodes
        else "No episode cleared the depth filter, so the value line is drawn "
        "unshaded - this portfolio never fell far enough from a high to register "
        "a drawdown cycle worth marking."
    )

    ui.chart(ch.underwater_chart(values, height=340))
    st.caption(
        "The same history read as duration rather than depth: distance below zero "
        "is the running loss from the high-water mark, and the shaded stretch is "
        "the longest continuous period spent there."
    )

    ui_theme.spacer(0.4)
    ui_theme.section("Episode ledger", "\U0001f4d1")
    table = _episode_table(episodes)
    if table.empty:
        ui.empty_state(
            "No qualifying drawdown episodes",
            "No peak-to-trough decline in this window was deep enough to be worth "
            "tabulating - shallow single-session dips are filtered out as noise.",
            icon="\U0001f4c8",
        )
    else:
        ui.dataframe(
            table,
            column_config={
                "Peak": st.column_config.DateColumn(format="DD MMM YYYY", width="small"),
                "Trough": st.column_config.DateColumn(format="DD MMM YYYY", width="small"),
                "Recovered on": st.column_config.DateColumn(
                    format="DD MMM YYYY",
                    width="small",
                    help="Blank means the portfolio has not regained that peak.",
                ),
                "Depth": st.column_config.NumberColumn(format="%.2f%%"),
                "Peak to trough": st.column_config.NumberColumn(
                    format="%d", help="Calendar days spent falling."
                ),
                "Trough to recovery": st.column_config.NumberColumn(
                    format="%d",
                    help="Calendar days spent climbing back. Blank if still underwater.",
                ),
                "Total": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "Deepest first, not most recent first. Blank recovery cells are "
            "episodes still open - never zeros, which would read as an instant "
            "recovery."
        )

    ui.methodology(
        "How episodes are cut and what the durations count",
        f"""
**The drawdown path** is `d_t = V_t / running_max(V) - 1`, a series that is zero
at every new high and negative everywhere else.

**An episode** runs from a high-water mark, through the trough that follows, to
the day the portfolio first regains that mark. Declines shallower than
{fmt_ratio_pct(dd.DEFAULT_MIN_DEPTH, 0)} are dropped as noise - in a daily series
a single quiet session would otherwise generate dozens of rows - and the table
shows the {dd.DEFAULT_TOP_N} deepest. Both are display choices, stated here so
neither is mistaken for a statistical threshold.

**Durations are calendar days**, not sessions: a drawdown does not pause for the
weekend as far as the holder is concerned.

**Time underwater is in observations**, deliberately different, so it stays
comparable across series sharing a trading calendar. This portfolio spent
{fmt_pct(tuw)} of its {result.n_days:,} sessions below a prior peak.

**Everything here is window-bound.** Max drawdown is the worst decline *inside
this history window*; lengthen the window and it will usually get deeper. It is
therefore never comparable between two portfolios measured over different
periods, and the deepest episode being incomplete is common near the right edge.

**Marked to daily closes.** Intraday troughs are invisible: a session that fell
7% and closed down 2% is recorded as -2%. The real hole was deeper than the
chart shows.
        """,
    )


# ---------------------------------------------------------------------------
# tab 4 - rolling
# ---------------------------------------------------------------------------
def _rolling_tab(result: Any, cfg: Any) -> None:
    rolling = result.rolling
    window = int(_f(getattr(cfg, "rolling_window", 63)) or 63)
    phrase = _window_phrase(window)

    if not _is_frame(rolling):
        ui.empty_state(
            "No rolling statistics",
            f"The rolling table is empty: a {window}-session window needs more "
            "history than this analysis has, so every trailing statistic would be "
            "undefined.",
            icon="\U0001f504",
            hint="Widen the history window in the sidebar, or shorten the rolling window.",
        )
        return

    usable = [c for c in _ROLLING_STACK if _has_values(rolling, c)]
    if not usable:
        ui.empty_state(
            "Rolling statistics are all undefined",
            f"The table exists but every column in it is empty over a {window}-session "
            "window - there are not enough overlapping observations to fill even one "
            "trailing estimate.",
            icon="\U0001f504",
        )
        return

    ui_theme.section("Risk through time", "\U0001f504")
    st.caption(
        f"Every line below is a trailing **{phrase}** estimate, plotted on the day "
        "it ends. A statistic only appears once the window is full, which is why "
        f"the first {window} sessions of the history are blank, and each point "
        f"shares {window - 1} of its observations with the point before it - "
        "consecutive values are heavily autocorrelated by construction and should "
        "never be read as independent measurements."
    )

    ui.chart(ch.rolling_multi_chart(rolling, list(usable), window=window, height=640))

    missing = [c for c in _ROLLING_STACK if c not in usable]
    if missing:
        st.caption(
            "Not shown, because no value could be computed over this window: "
            + ", ".join(missing)
            + ". Beta and correlation need an overlapping benchmark series."
        )

    _rolling_reading(rolling)

    ui_theme.spacer(0.4)
    ui_theme.section("Benchmark dependence and rolling tail", "\U0001f4c8")
    left, right = st.columns(2, gap="medium")

    with left:
        if _has_values(rolling, "correlation"):
            ui.chart(
                ch.rolling_chart(
                    rolling,
                    "correlation",
                    title="Rolling correlation to the benchmark",
                    subtitle=f"Trailing {phrase}",
                    window=window,
                    height=340,
                )
            )
            st.caption(
                f"Correlation to {getattr(cfg, 'benchmark_name', 'the benchmark')} "
                "over the same trailing window. It rising toward 1.0 in a sell-off "
                "is the classic failure mode of diversification: the hedges stop "
                "hedging exactly when they are needed."
            )
        else:
            ui.empty_state(
                "No rolling correlation",
                "The benchmark series did not overlap the portfolio on enough "
                "dates to fill a single window.",
                icon="\U0001f517",
            )

    with right:
        if _has_values(rolling, "var"):
            ui.chart(
                ch.rolling_chart(
                    rolling,
                    "var",
                    title="Rolling one-day VaR",
                    subtitle=f"Trailing {phrase}",
                    window=window,
                    height=340,
                )
            )
            st.caption(
                f"The {_conf_label(_confidence(result, cfg))} historical VaR "
                "recomputed on each trailing window. Unlike the single headline "
                "number it shows when the tail widened - and because it is an "
                "empirical quantile, it steps down abruptly when one bad day "
                "finally rolls out of the window."
            )
        else:
            ui.empty_state(
                "No rolling VaR",
                "A trailing quantile needs more observations inside the window "
                "than this history provides.",
                icon="\U0001f4c9",
            )

    ui.methodology(
        "Rolling windows: construction and the traps",
        f"""
**Window.** {phrase}, taken from `AppConfig.rolling_window`. Each statistic is
computed on the last {window} daily observations and stamped on the **final**
day of that window, never the middle - so a spike marks when the window
*contained* the event, not when the event happened.

**Definitions inside the window** are identical to the headline versions:
volatility is `stdev * sqrt({TRADING_DAYS})`, Sharpe uses the same
{fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))} risk-free rate, beta is
`cov(r_p, r_b) / var(r_b)`, and VaR is the empirical
{_tail_label(_confidence(result, cfg))} quantile.

**Overlapping windows are not independent.** Two consecutive points share
{window - 1} of their {window} observations. Any apparent "trend" in a rolling
series is partly an artefact of that overlap, and standard significance tests do
not apply to it.

**Benchmark alignment.** Beta and correlation are reindexed onto the
*portfolio's* dates, so a benchmark that trades on a different calendar leaves
gaps rather than silently shifting the series by a day. Missing days are dropped
from the pair, never forward-filled.

**Both edges are ragged.** The first {window} sessions carry no value at all,
and the last point is the only one that describes the portfolio as it stands
today. The eye is drawn to the right-hand edge, where the estimate is built from
the fewest independent events.
        """,
    )


def _rolling_reading(rolling: pd.DataFrame) -> None:
    vol = pd.to_numeric(rolling.get("volatility"), errors="coerce").dropna()
    if vol.empty:
        return
    latest = float(vol.iloc[-1])
    peak = float(vol.max())
    trough = float(vol.min())
    average = float(vol.mean())

    if peak > 0 and trough > 0:
        spread = f"{fmt_num(peak / trough, 1)}x between its calmest and wildest window"
    else:
        spread = "an undefined range"

    st.caption(
        f"Trailing volatility ended at {fmt_ratio_pct(latest)} against a window "
        f"average of {fmt_ratio_pct(average)}, ranging from "
        f"{fmt_ratio_pct(trough)} to {fmt_ratio_pct(peak)} - {spread}. That "
        "spread is the honest answer to what a single annualised volatility "
        "number hides: the portfolio was never *that* risky, it was two or three "
        "different risk profiles averaged together."
    )


# ---------------------------------------------------------------------------
# tab 5 - contribution
# ---------------------------------------------------------------------------
def _contribution_reading(rvw: pd.DataFrame) -> None:
    frame = rvw.copy()
    frame["ratio"] = pd.to_numeric(frame["ratio"], errors="coerce")
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="coerce")
    frame["risk_contribution_pct"] = pd.to_numeric(frame["risk_contribution_pct"], errors="coerce")
    usable = frame.dropna(subset=["ratio"])
    if usable.empty:
        st.caption(
            "No holding has a defined risk-to-weight ratio on this run - the "
            "covariance matrix or the weights could not be used."
        )
        return

    top = usable.iloc[0]
    flagged = usable[usable["ratio"] >= RISK_WEIGHT_RATIO]
    tail = usable.iloc[-1]

    if float(top["ratio"]) >= RISK_WEIGHT_RATIO:
        lead = (
            f"**{top['ticker']} is {fmt_pct(top['weight_pct'], 1)} of the money and "
            f"{fmt_pct(top['risk_contribution_pct'], 1)} of the volatility - "
            f"{fmt_num(top['ratio'], 2)}x its weight.** That is the single most "
            "actionable line on this page: trimming it removes more variance per "
            "unit of turnover than trimming anything else in the book, because its "
            "volatility and its correlation to what you already own are both "
            "working against you."
        )
        if len(flagged) > 1:
            others = ", ".join(str(t) for t in flagged["ticker"].tolist()[1:])
            lead += (
                f" {len(flagged)} names clear the "
                f"{fmt_num(RISK_WEIGHT_RATIO, 2)}x line in total: {others} sit "
                "behind it."
            )
    else:
        lead = (
            f"No holding carries materially more risk than its weight - the widest "
            f"gap is {top['ticker']} at {fmt_num(top['ratio'], 2)}x, below the "
            f"{fmt_num(RISK_WEIGHT_RATIO, 2)}x flag. Risk in this book tracks "
            "capital closely, which usually means the holdings have similar "
            "volatility and similar correlations rather than that they are "
            "genuinely diversifying."
        )

    if _finite(tail["ratio"]) and float(tail["ratio"]) < 1.0:
        lead += (
            f" At the other end, {tail['ticker']} contributes "
            f"{fmt_pct(tail['risk_contribution_pct'], 1)} of the risk for "
            f"{fmt_pct(tail['weight_pct'], 1)} of the capital "
            f"({fmt_num(tail['ratio'], 2)}x) - it is doing real diversification "
            "work, not just sitting there."
        )

    st.caption(lead)


def _contribution_tab(result: Any, cfg: Any) -> None:
    attribution = result.attribution or {}
    rc = attribution.get("risk_contribution")
    rvw = attribution.get("risk_vs_weight")

    if not _is_frame(rc):
        ui.empty_state(
            "No risk decomposition",
            "Splitting portfolio volatility across holdings needs a usable "
            "covariance matrix and a full set of weights; this run produced "
            "neither in a form the decomposition could accept.",
            icon="\U0001f9e9",
            hint="It needs at least two holdings with overlapping price history.",
        )
        return

    ui_theme.section("Who manufactures the volatility", "\U0001f9e9")

    ui.chart(ch.risk_contribution_chart(rc, height=max(340, 46 * len(rc) + 160)))

    total = pd.to_numeric(rc.get("risk_contribution_pct"), errors="coerce").sum()
    sigma_p = _f(rc.attrs.get("portfolio_volatility"))
    st.caption(
        f"Capital weight against risk share, name by name. The risk column sums to "
        f"{fmt_pct(total, 1)}"
        + (
            f" and the underlying contributions sum to a portfolio volatility of "
            f"{fmt_ratio_pct(sigma_p)}"
            if math.isfinite(sigma_p)
            else ""
        )
        + " - that exactness is what makes the split comparable across names "
        "instead of merely suggestive."
    )

    if _is_frame(rvw):
        ui_theme.spacer(0.3)
        _contribution_reading(rvw)

        display = pd.DataFrame(
            {
                "Ticker": rvw["ticker"].astype(str),
                "Weight": pd.to_numeric(rvw["weight_pct"], errors="coerce"),
                "Risk share": pd.to_numeric(rvw["risk_contribution_pct"], errors="coerce"),
                "Risk / weight": pd.to_numeric(rvw["ratio"], errors="coerce"),
            }
        )
        ui.dataframe(
            display,
            column_config={
                "Ticker": st.column_config.TextColumn(width="small", pinned=True),
                "Weight": st.column_config.ProgressColumn(
                    "Weight",
                    format="%.2f%%",
                    min_value=0.0,
                    max_value=100.0,
                    help="Share of current market value.",
                ),
                "Risk share": st.column_config.ProgressColumn(
                    "Risk share",
                    format="%.2f%%",
                    min_value=0.0,
                    max_value=100.0,
                    help="Share of portfolio volatility (CCTR / sigma_p).",
                ),
                "Risk / weight": st.column_config.NumberColumn(
                    format="%.2fx",
                    help=f"Above {RISK_WEIGHT_RATIO:.2f}x the name punches above "
                    f"its capital weight in risk terms; below 1.00x it "
                    f"diversifies.",
                ),
            },
        )
        st.caption(
            f"Sorted by risk-to-weight, worst first. 1.00x means risk exactly "
            f"proportional to capital; the {fmt_num(RISK_WEIGHT_RATIO, 2)}x line "
            "is `analytics.insights.RISK_WEIGHT_RATIO`, a stated judgement call "
            "rather than a statistical result."
        )

    asset_returns = result.asset_returns
    n_assets = int(getattr(asset_returns, "shape", (0, 0))[1]) if asset_returns is not None else 0
    realised_vol = fmt_ratio_pct((result.risk or {}).get("ann_vol"))

    ui.methodology(
        "MCTR, CCTR and why the contributions have to add up",
        f"""
With `w` the current weight vector (normalised to sum to 1) and `S` the
annualised covariance matrix of daily asset returns:

```
sigma_p = sqrt(w' S w)                 portfolio volatility
MCTR_i  = (S w)_i / sigma_p            marginal contribution to risk
CCTR_i  = w_i * MCTR_i                 component contribution to risk
rc_i    = CCTR_i / sigma_p * 100       risk share, in percent
```

**MCTR is a derivative, not a share.** It answers "add one more unit of weight
to this name, funded pro rata from everything else - how much does portfolio
volatility move?" It carries the covariance matrix's units (annualised decimals
here) and it does **not** sum to anything meaningful.

**CCTR is the additive version, and its additivity is not an approximation.**
Because `sigma_p` is homogeneous of degree one in `w`, Euler's theorem makes the
decomposition exact: `sum_i CCTR_i = w' S w / sigma_p = sigma_p`, and therefore
`sum_i rc_i = 100`. That identity is the whole reason the numbers are worth
printing. An approximate split - one where the parts did not reconstruct the
whole - would let you attribute risk twice and would collapse the moment two
names were correlated.

**Why a 5% position can carry 15% of the risk.** `MCTR_i` contains `(S w)_i`,
the covariance of that asset with the *portfolio*, not just its own variance. A
name that is volatile **and** moves with what you already own gets punished
twice; a volatile name that moves against the book can carry less risk than its
weight, and occasionally a negative contribution.

**Where the matrix comes from.** `S` is the annualised sample covariance of the
{n_assets} holdings with usable price history. Two consequences worth stating: the split is
only as stable as that estimate - sample covariance is notoriously noisy when
the number of names approaches the number of observations - and it is computed
on **today's** weights, so it describes the portfolio you hold now, not the one
that produced the historical return series.

**The totals will not tie exactly to the headline volatility.** `sigma_p` from
this decomposition is the covariance-implied volatility of today's weights;
{realised_vol} on the Summary tab is the realised volatility of the actual value
series, whose weights drifted throughout the window. The gap between them is
drift, not an error.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Risk analytics",
        "Dispersion, tails, drawdowns and where the volatility is manufactured",
        icon="\U0001f6e1",
    )

    result = guards.require_result()
    if result is None:
        return
    if not guards.require_observations(result, what="Risk analytics"):
        return

    cfg = result.config
    risk = result.risk or {}
    level = _confidence(result, cfg)

    ui.badges(
        [
            (f"{int(_f(risk.get('n_obs')) or 0):,} daily observations", "accent"),
            (f"VaR confidence {_conf_label(level)}", "neutral"),
            (f"Risk-free {fmt_ratio_pct(getattr(cfg, 'risk_free_rate', 0.0))}", "muted"),
            (
                f"Rolling window {int(_f(getattr(cfg, 'rolling_window', 63)) or 63)} sessions",
                "neutral",
            ),
            (f"Reported in {getattr(cfg, 'base_currency', '')}", "muted"),
        ]
    )

    summary_tab, tail_tab, drawdown_tab, rolling_tab, contribution_tab = st.tabs(
        ["Summary", "Tail risk", "Drawdowns", "Rolling", "Contribution"]
    )

    with summary_tab:
        _summary_tab(result, cfg)

    with tail_tab:
        _tail_tab(result, cfg)

    with drawdown_tab:
        _drawdown_tab(result, cfg)

    with rolling_tab:
        _rolling_tab(result, cfg)

    with contribution_tab:
        _contribution_tab(result, cfg)
