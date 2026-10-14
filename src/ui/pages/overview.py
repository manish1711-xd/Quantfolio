"""Overview -- the command centre.

The first screen after Analyze, and the one that has to answer two questions in
about five seconds: *how am I doing* and *how much risk am I carrying to do it*.

Everything here is read straight off the :class:`src.models.AnalysisResult` the
pipeline produced -- value and P&L from ``summary``, the risk statistics from
``risk``, the 0-100 heuristic from ``score``, the rule-derived commentary from
``insights``. This page computes nothing of its own; it decides what deserves
the top of the screen, states the units, and says where each number came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.config import (
    DATA_SOURCE,
    INSIGHTS,
    PERIOD_LABELS,
    RISK_RUBRIC,
    TRADING_DAYS,
)
from src.ui import components as ui
from src.ui import guards
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    delta_tone,
    fmt_date,
    fmt_delta,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    humanize_days,
)

#: A quote older than this is flagged on the provenance strip. Yahoo's NSE/BSE
#: feed is itself delayed ~15 minutes, so anything beyond 20 is stale twice over.
_STALE_AFTER_MINUTES = 20.0

#: Sparkline window on the headline value card, in trading sessions.
_SPARK_SESSIONS = 90


# ---------------------------------------------------------------------------
# small local helpers (display only -- no analytics happen on this page)
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    """True when ``x`` is a real, finite number worth printing."""
    try:
        return bool(pd.notna(x)) and float(x) == float(x) and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def _latest_fetch(result: Any) -> Any:
    """Newest quote timestamp, falling back to when the run itself finished."""
    stamps = [
        q.fetched_at
        for q in (result.quotes or {}).values()
        if getattr(q, "fetched_at", None) is not None
    ]
    try:
        return max(stamps) if stamps else result.generated_at
    except (TypeError, ValueError):  # pragma: no cover - mixed tz-awareness
        return result.generated_at


def _age_minutes(stamp: Any) -> float:
    """Minutes since ``stamp``; naive timestamps are read as UTC."""
    if stamp is None:
        return float("nan")
    try:
        moment = stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - moment).total_seconds() / 60.0, 0.0)
    except Exception:  # noqa: BLE001 - a provenance strip must never crash a page
        return float("nan")


def _sparkline(result: Any) -> list[float] | None:
    """The last ~90 sessions of portfolio value, for the headline card."""
    values = result.values
    if values is None or len(values) < 3:
        return None
    try:
        return [float(v) for v in values.tail(_SPARK_SESSIONS).tolist()]
    except Exception:  # noqa: BLE001 - decoration must never break the page
        return None


def _sharpe_tone(sharpe: Any) -> str:
    if not _finite(sharpe):
        return "muted"
    if float(sharpe) >= INSIGHTS.good_sharpe:
        return "gain"
    return "loss" if float(sharpe) < 0 else "warn"


def _vol_tone(ann_vol: Any) -> str:
    """Amber at half the rubric's saturation point, red at it. Colour only."""
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


def _beta_tone(beta: Any) -> str:
    if not _finite(beta):
        return "muted"
    value = float(beta)
    if value >= INSIGHTS.aggressive_beta:
        return "warn"
    return "gain" if value <= INSIGHTS.defensive_beta else "neutral"


# ---------------------------------------------------------------------------
# 1-2. provenance and data notes
# ---------------------------------------------------------------------------
def _provenance(result: Any, cfg: Any) -> None:
    fetched = _latest_fetch(result)
    age = _age_minutes(fetched)
    intraday = [q for q in (result.quotes or {}).values() if getattr(q, "is_intraday", False)]

    extra = [
        f"{len(result.quotes or {})} of {int(result.summary.get('n_holdings', 0) or 0)} "
        f"holdings priced",
        "Live-ish quotes (delayed ~15 min)" if intraday else "Latest available closes",
    ]
    if _finite(age):
        extra.append(f"{int(round(age))} min old")

    ui.freshness_bar(
        fetched,
        DATA_SOURCE,
        extra=tuple(extra),
        stale=bool(_finite(age) and age > _STALE_AFTER_MINUTES),
    )

    window = PERIOD_LABELS.get(cfg.period, cfg.period)
    chips: list[tuple[str, str]] = [
        (f"Window {window} ({result.n_days} sessions)", "accent"),
        (f"Benchmark {cfg.benchmark_name} ({cfg.benchmark_symbol})", "neutral"),
        (f"Reported in {cfg.base_currency}", "neutral"),
        (f"Risk-free {fmt_ratio_pct(cfg.risk_free_rate)}", "muted"),
    ]
    if result.fx:
        chips.append((f"FX applied to {len(result.fx)} currencies", "warn"))
    if result.ledger.get("transactions"):
        chips.append((f"{len(result.ledger['transactions'])}-trade ledger loaded", "gain"))
    ui.badges(chips)

    ui.warning_list(result.warnings)


# ---------------------------------------------------------------------------
# 3. headline KPI row
# ---------------------------------------------------------------------------
def _headline_kpis(result: Any, cfg: Any) -> None:
    summary = result.summary
    risk = result.risk
    bench = result.benchmark or {}
    ccy = cfg.base_currency

    day_pct = summary.get("day_change_pct")
    day_money = summary.get("day_change")
    total_pnl = summary.get("total_pnl")
    total_pct = summary.get("total_return_pct")
    ann_return = risk.get("ann_return")
    sharpe = risk.get("sharpe")
    excess = bench.get("excess_return") if bench.get("ok") else None

    value_card = {
        "label": "Current value",
        "value": fmt_money(summary.get("current_value"), ccy, 0),
        "tone": "accent",
        "sparkline": _sparkline(result),
        "help_text": (
            "Today's share counts at the latest available price, converted into "
            f"{ccy} at spot FX where a holding trades in another currency."
        ),
    }
    if _finite(day_pct):
        value_card["delta"] = f"{fmt_delta(day_pct)} today"
        value_card["delta_tone"] = delta_tone(day_pct)
        value_card["footnote"] = (
            f"{fmt_money(day_money, ccy, 0, signed=True)} versus the previous close"
        )
    else:
        value_card["footnote"] = "No previous close supplied for these instruments"

    pnl_card = {
        "label": "Total P&L",
        "value": fmt_money(total_pnl, ccy, 0, signed=True),
        "tone": delta_tone(total_pnl),
        "delta": fmt_delta(total_pct),
        "delta_tone": delta_tone(total_pct),
        "footnote": (
            f"On {fmt_money(summary.get('total_investment'), ccy, 0)} invested - "
            f"{summary.get('n_gainers', 0)} up, {summary.get('n_losers', 0)} down"
        ),
        "help_text": (
            "Unrealised, measured against the average buy price you entered. "
            "It ignores dividends, brokerage and taxes."
        ),
    }

    ann_card = {
        "label": "Annualised return",
        "value": fmt_ratio_pct(ann_return),
        "tone": delta_tone(ann_return),
        "footnote": f"Geometric, from {result.n_days} sessions of daily closes",
        "help_text": (
            "Compound rate that would turn the first value in the window into the "
            f"last one, scaled to {TRADING_DAYS} trading days."
        ),
    }
    if _finite(excess):
        ann_card["delta"] = f"{fmt_ratio_pct(excess, signed=True)} vs {cfg.benchmark_name}"
        ann_card["delta_tone"] = delta_tone(excess)

    sharpe_card = {
        "label": "Sharpe ratio",
        "value": fmt_num(sharpe, 2),
        "tone": _sharpe_tone(sharpe),
        "footnote": (
            f"Excess over a {fmt_ratio_pct(cfg.risk_free_rate)} risk-free rate, "
            "per unit of total volatility"
        ),
        "help_text": (
            "(annualised return - risk-free) / annualised volatility. Below zero "
            "the portfolio lost to cash; the market's own long-run number is "
            "roughly 0.4 to 0.6."
        ),
    }

    ui.kpi_row([value_card, pnl_card, ann_card, sharpe_card], columns=4)

    if _finite(sharpe) and _finite(ann_return):
        if float(sharpe) >= 0:
            reading = (
                f"each unit of volatility bought {fmt_num(sharpe, 2)} units of return "
                f"above the {fmt_ratio_pct(cfg.risk_free_rate)} risk-free rate"
            )
        else:
            reading = (
                f"below zero, so the {fmt_ratio_pct(ann_return)} annualised return lost "
                f"to a {fmt_ratio_pct(cfg.risk_free_rate)} risk-free rate -- the volatility "
                "was carried for nothing over this window"
            )
        st.caption(
            f"Sharpe {fmt_num(sharpe, 2)}: {reading}. It describes these "
            f"{result.n_days} sessions and nothing else -- a measurement, not a forecast."
        )

    ui.methodology(
        "How the headline numbers are built",
        f"""
**Current value** = `sum(quantity x latest price)` per holding, converted into
{cfg.base_currency} at the spot FX rate when the instrument trades elsewhere.
Prices are the latest available (intraday quote when the venue is open, last
close otherwise) -- never a mid, never an estimate.

**Day change** is *value-weighted* across the holdings that reported a previous
close. Names without one are excluded from both the numerator and the
denominator rather than counted as flat, so the percentage never gets diluted
by silent zeros.

**Total P&L** is unrealised and measured against the average buy price you
entered: `current_value - invested`. It excludes dividends, brokerage, STT and
taxes. If you supplied a trade ledger, the realised side lives on
**Trades & lots**.

**Annualised return** is geometric: `(V_end / V_start) ** (252 / n_obs) - 1`,
with n = {result.n_days} observations here and {TRADING_DAYS} trading days per
year. It is a *time-weighted* figure computed on today's share counts, so it is
not the same thing as your money-weighted (XIRR) return.

**Sharpe** = `(annualised return - {fmt_ratio_pct(cfg.risk_free_rate)}) / annualised
volatility`, using the risk-free rate set in the sidebar. Caveats worth stating:
Sharpe assumes returns are roughly symmetric, it punishes upside volatility as
hard as downside, and over a short window it is dominated by whichever regime
happened to fall inside the window.
        """,
    )


# ---------------------------------------------------------------------------
# 4. risk KPI row
# ---------------------------------------------------------------------------
def _risk_kpis(result: Any, cfg: Any) -> None:
    risk = result.risk
    bench = result.benchmark or {}
    ccy = cfg.base_currency

    ann_vol = risk.get("ann_vol")
    mdd = risk.get("mdd")
    var = risk.get("var")
    confidence = risk.get("confidence", cfg.var_confidence)
    beta = bench.get("beta") if bench.get("ok") else None

    vol_card = {
        "label": "Annualised volatility",
        "value": fmt_ratio_pct(ann_vol),
        "tone": _vol_tone(ann_vol),
        "footnote": (
            f"{cfg.benchmark_name} {fmt_ratio_pct(bench.get('benchmark_vol'))} over the same window"
            if bench.get("ok")
            else "Daily standard deviation scaled by sqrt(252)"
        ),
        "help_text": "Sample standard deviation of daily returns x sqrt(252).",
    }

    dd_card = {
        "label": "Max drawdown",
        "value": fmt_ratio_pct(mdd),
        "tone": _drawdown_tone(mdd),
        "footnote": _drawdown_footnote(result),
        "help_text": (
            "Deepest peak-to-trough fall in portfolio value inside this window, "
            "measured on daily closes."
        ),
    }

    var_card = {
        "label": f"Daily VaR {fmt_pct(float(confidence) * 100.0, 0)}",
        "value": fmt_ratio_pct(var),
        "tone": _var_tone(var),
        "footnote": _var_footnote(result, ccy),
        "help_text": (
            "Historical (non-parametric) one-day loss that was exceeded on "
            f"{fmt_pct((1.0 - float(confidence)) * 100.0, 0)} of the days in this window."
        ),
    }

    beta_card = {
        "label": f"Beta vs {cfg.benchmark_name}",
        "value": fmt_num(beta, 2) if _finite(beta) else DASH,
        "tone": _beta_tone(beta),
        "footnote": (
            f"R-squared {fmt_ratio_pct(bench.get('r_squared'))} of daily variance explained"
            if bench.get("ok")
            else (bench.get("reason") or "Benchmark comparison unavailable")
        ),
        "help_text": (
            "cov(portfolio, benchmark) / var(benchmark) on overlapping dates. "
            "1.0 moves with the index; above 1.2 is geared to it."
        ),
    }

    ui.kpi_row([vol_card, dd_card, var_card, beta_card], columns=4)

    if _finite(var) and _finite(ann_vol):
        st.caption(
            f"Volatility {fmt_ratio_pct(ann_vol)} and a "
            f"{fmt_pct(float(confidence) * 100.0, 0)} daily VaR of "
            f"{fmt_ratio_pct(var)} describe two different risks: the first is how "
            "much the book normally moves, the second is how bad a single ordinary-bad "
            "session looks. Neither bounds the loss -- a tail day goes past VaR by "
            "definition, and expected shortfall beyond it is "
            f"{fmt_ratio_pct(risk.get('cvar'))}."
        )

    if bench.get("ok"):
        beta_note = (
            f"on the {int(bench.get('n_days', 0) or 0)} dates the portfolio and "
            f"{cfg.benchmark_name} both traded. It only means something to the extent "
            "the index explains the portfolio at all, so read it next to R-squared "
            f"({fmt_ratio_pct(bench.get('r_squared'))}): a low R-squared says most of "
            "the movement is idiosyncratic and the beta is a weak description of it"
        )
    else:
        beta_note = (
            f"-- unavailable on this run, because the {cfg.benchmark_name} series could "
            f"not be aligned with the portfolio ({bench.get('reason') or 'no reason recorded'})"
        )

    ui.methodology(
        "Volatility, drawdown, VaR and beta -- conventions and caveats",
        f"""
**Annualised volatility** = `stdev(daily returns) x sqrt({TRADING_DAYS})`, sample
(n-1) standard deviation. Scaling by the square root of time assumes daily
returns are independent; in a trending or mean-reverting market that assumption
understates and overstates risk respectively.

**Max drawdown** = `min(V_t / running_max(V) - 1)`, a negative decimal, measured
on daily closes inside this window only. A longer window will usually find a
deeper one -- drawdown is not comparable across windows of different lengths.

**Value at Risk** is the empirical
{fmt_pct((1.0 - float(confidence)) * 100.0, 0)} quantile of the
{int(risk.get("n_obs", 0) or 0)} daily returns in this window (historical
simulation -- no normality assumed). It is a *one-day* loss at
{fmt_pct(float(confidence) * 100.0, 0)} confidence and it says nothing about how
much worse the remaining {fmt_pct((1.0 - float(confidence)) * 100.0, 0)} of days
get; that is what conditional VaR (expected shortfall,
{fmt_ratio_pct(risk.get("cvar"))}) is for. Gaussian and Cornish-Fisher variants
are on the Risk analytics page for comparison.

**Beta** = `cov(r_portfolio, r_benchmark) / var(r_benchmark)`, measured
{beta_note}.
        """,
    )


def _drawdown_footnote(result: Any) -> str:
    """Dates and recovery status of the deepest episode the pipeline found."""
    episodes = result.drawdowns or []
    if not episodes:
        return "No completed peak-to-trough cycle in this window"
    worst = episodes[0]
    span = f"{fmt_date(worst.peak_date)} to {fmt_date(worst.trough_date)}"
    if worst.recovered:
        return f"{span}, recovered in {humanize_days(worst.recovery_days)}"
    return f"{span}, still below that high-water mark"


def _var_footnote(result: Any, ccy: str) -> str:
    var = result.risk.get("var")
    value = result.summary.get("current_value")
    if not (_finite(var) and _finite(value)):
        return "Historical simulation on daily returns"
    money = abs(float(var)) * float(value)
    return f"About {fmt_money(money, ccy, 0)} on today's book, in one session"


# ---------------------------------------------------------------------------
# 5. risk score and allocation map
# ---------------------------------------------------------------------------
def _score_and_allocation(result: Any, cfg: Any) -> None:
    left, right = st.columns([1.0, 1.25], gap="large")

    with left:
        st.markdown("**Portfolio risk score**")
        score = result.score or {}
        ui.chart(ch.risk_gauge(score.get("score"), score.get("band", ""), height=330))
        _score_caption(score)
        _score_methodology(score)

    with right:
        st.markdown("**Where the money actually sits**")
        ui.chart(
            ch.allocation_treemap(result.positions, currency_symbol=cfg.currency_symbol, height=330)
        )
        _allocation_caption(result, cfg)


def _score_caption(score: dict) -> None:
    value = score.get("score")
    band = score.get("band", "")
    if not _finite(value):
        st.caption(
            "The rubric refused to grade this portfolio: volatility or drawdown "
            "could not be measured from the available history."
        )
        return
    st.caption(
        f"{fmt_num(value, 0)} / 100 -- **{band}** on a rubric scored out of "
        f"{fmt_num(score.get('max_points'), 0)} measurable points. Below "
        f"{fmt_num(RISK_RUBRIC.low_band, 0)} is Low, above "
        f"{fmt_num(RISK_RUBRIC.high_band, 0)} is High. It grades the *shape* of "
        "the book, not the quality of the names in it."
    )


def _score_methodology(score: dict) -> None:
    components = score.get("components") or {}
    body = f"""
The score is a **transparent rule-based heuristic**, published here in full so
you can disagree with it. It is **not** a regulatory measure, not a credit-style
rating, and not calibrated against anyone's loss experience. Every weight and
saturation point lives in `config.RISK_RUBRIC`:

| Component | Max points | Full points at |
| --- | --- | --- |
| Annualised volatility | {RISK_RUBRIC.vol_points:g} | {fmt_ratio_pct(RISK_RUBRIC.vol_cap, 0)} annualised |
| Max drawdown | {RISK_RUBRIC.drawdown_points:g} | {fmt_ratio_pct(RISK_RUBRIC.drawdown_cap, 0)} peak-to-trough |
| Concentration (HHI) | {RISK_RUBRIC.concentration_points:g} | one name holding everything |
| Holdings count | {RISK_RUBRIC.holdings_points:g} | a single position ({RISK_RUBRIC.holdings_floor}+ names score zero) |
| Tail loss (daily CVaR) | {RISK_RUBRIC.tail_points:g} | {fmt_ratio_pct(RISK_RUBRIC.tail_cap, 0)} expected shortfall |

Each component is scaled linearly between zero and its cap. A component whose
input is missing is dropped from **both** the earned points and the available
points, and the result is reported as `earned / available x 100` -- so a
portfolio with no tail estimate is scored out of
{fmt_num(score.get("max_points"), 0)} rather than being silently flattered by
the {RISK_RUBRIC.tail_points:g} points it could not be measured on.

Caveat: higher is not "worse investment". A concentrated, volatile book that you
hold deliberately and can sit through scores high here and may still be the right
portfolio for you. The number describes exposure, not suitability.
"""
    with st.expander("How the risk score is computed", expanded=False):
        st.markdown(body)
        if components:
            frame = pd.DataFrame(
                {
                    "Component": list(components.keys()),
                    "Points earned": [fmt_num(v, 1) for v in components.values()],
                }
            )
            st.caption("Raw points earned on this run, before rescaling:")
            ui.dataframe(frame)


def _allocation_caption(result: Any, cfg: Any) -> None:
    summary = result.summary
    largest = summary.get("largest_holding")
    top5 = summary.get("top5_weight_pct")
    effective = summary.get("effective_holdings")
    parts = ["Tile area is market value; tile colour is unrealised return against your cost."]
    if largest:
        parts.append(f"Largest position {largest[0]} at {fmt_pct(largest[1])} of the book.")
    if _finite(top5):
        parts.append(f"Top five names carry {fmt_pct(top5)}.")
    if _finite(effective):
        parts.append(
            f"Effective holdings {fmt_num(effective, 1)} against "
            f"{summary.get('n_holdings', 0)} actual -- the diversification you are "
            "really getting after weights are taken into account."
        )
    st.caption(" ".join(parts))


# ---------------------------------------------------------------------------
# 6. value timeline
# ---------------------------------------------------------------------------
def _timeline(result: Any, cfg: Any) -> None:
    ui_theme.section("VALUE THROUGH TIME", "📈")

    if not guards.require_history(result, "The value timeline"):
        return

    ui.chart(
        ch.growth_line(
            result.values,
            cfg.currency_symbol,
            result.benchmark_prices,
            cfg.benchmark_name,
        )
    )

    ledger = result.ledger or {}
    if ledger.get("transactions"):
        xirr = ledger.get("xirr")
        money_weighted = (
            f" Your money-weighted return (XIRR) on those trades is {fmt_ratio_pct(xirr)}."
            if _finite(xirr)
            else ""
        )
        st.caption(
            f"This line holds **today's** share counts constant back through the "
            f"whole window -- it is the history of the portfolio you own now, not a "
            f"record of what you actually held on each date. You supplied a "
            f"{len(ledger['transactions'])}-trade ledger, so the money-weighted view "
            f"that accounts for *when* the cash went in lives on **Trades & lots**."
            f"{money_weighted}"
        )
    else:
        st.caption(
            "Assumption stated plainly: **today's share counts are held constant "
            "across the whole window**. Any position you added or trimmed mid-window "
            "is back-projected at its current size, which flatters or penalises the "
            "line depending on what you did. Add a trade ledger on the Portfolio page "
            "to unlock the money-weighted (XIRR) view that fixes this."
            + (
                f" The benchmark leg is {cfg.benchmark_name}, indexed to the same starting scale."
                if result.benchmark_prices is not None
                else ""
            )
        )


# ---------------------------------------------------------------------------
# 7. narrative
# ---------------------------------------------------------------------------
def _narrative(result: Any) -> None:
    ui_theme.section("WHAT THE NUMBERS SAY", "🧭")
    st.caption(
        "Each line is fired by a documented threshold in `config.InsightThresholds` "
        f"-- concentration above {fmt_pct(INSIGHTS.concentration_warn_pct, 0)}, "
        f"drawdown past {fmt_ratio_pct(INSIGHTS.deep_drawdown, 0)}, correlation above "
        f"{fmt_num(INSIGHTS.high_correlation, 2)}, and so on -- and carries the "
        "evidence that fired it. These are observations about what already happened. "
        "Nothing here is a prediction, a recommendation or advice."
    )
    ui.insight_list(result.insights, limit=8)
    if len(result.insights or []) > 8:
        st.caption(
            f"{len(result.insights) - 8} further observations are grouped by theme "
            "on the Risk analytics and Correlation pages."
        )


# ---------------------------------------------------------------------------
# 8. secondary statistics
# ---------------------------------------------------------------------------
def _secondary_stats(result: Any) -> None:
    ui_theme.section("SECONDARY STATISTICS", "🔬")
    risk = result.risk

    stats: list[tuple[str, str]] = [
        ("Sortino ratio", fmt_num(risk.get("sortino"), 2)),
        ("Calmar ratio", fmt_num(risk.get("calmar"), 2)),
        ("Ulcer index", fmt_pct(risk.get("ulcer"))),
        ("Skew", fmt_num(risk.get("skew"), 2)),
        ("Excess kurtosis", fmt_num(risk.get("kurtosis"), 2)),
        ("Hit ratio", fmt_ratio_pct(risk.get("hit_ratio"), 1)),
        ("Best day", fmt_ratio_pct(risk.get("best_day"), 2, signed=True)),
        ("Worst day", fmt_ratio_pct(risk.get("worst_day"), 2, signed=True)),
        ("Effective bets", fmt_num(risk.get("effective_bets"), 1)),
        ("Sessions analysed", f"{result.n_days:,}"),
    ]
    ui.stat_grid(stats, columns=5)

    ui.methodology(
        "Definitions, units and what each one is actually for",
        f"""
* **Sortino** = `(annualised return - risk-free) / downside deviation`. Same
  shape as Sharpe but the denominator only counts returns below the minimum
  acceptable return (0% here), so upside volatility stops being punished.
* **Calmar** = `annualised return / |max drawdown|`. Return per unit of the
  worst loss actually lived through. Below
  {fmt_num(INSIGHTS.weak_calmar, 2)} the pain has not been paid for.
* **Ulcer index** is in **percent**, not a ratio: the root-mean-square of the
  drawdown series. Unlike max drawdown it punishes *long* underwater stretches,
  not just deep ones.
* **Skew** below zero means the left tail is longer -- occasional large losses
  against many small gains, which is the typical shape of equity risk.
* **Excess kurtosis** uses the Fisher definition, so a normal distribution
  scores 0. Above {fmt_num(INSIGHTS.fat_tail_kurtosis, 0)} the tails are fat
  enough that any normality-based risk number (Gaussian VaR, most option maths)
  understates the extremes.
* **Hit ratio** is the share of sessions that closed positive. It is not a
  quality measure on its own: many small wins and one catastrophic loss produce
  an excellent hit ratio.
* **Best / worst day** are the extreme single-session returns in this window --
  the honest scale of what one day can do.
* **Effective bets** = `1 / sum(w_i ** 2)` on current weights. It answers "how
  many *independent-sized* positions do I really have"; far below the holdings
  count means the book is lopsided.
* **Sessions analysed** is {result.n_days} overlapping trading days, giving
  {int(risk.get("n_obs", 0) or 0)} daily return observations. Every statistic on
  this page describes that window and nothing outside it.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Overview",
        "Value, performance and the risk being carried to earn it",
        icon="📊",
    )

    result = guards.require_result()
    if result is None:
        return

    cfg = result.config

    _provenance(result, cfg)
    _headline_kpis(result, cfg)
    ui_theme.spacer(0.4)
    _risk_kpis(result, cfg)

    ui_theme.spacer(0.6)
    ui_theme.section("RISK PROFILE AND ALLOCATION", "🧮")
    _score_and_allocation(result, cfg)

    ui_theme.spacer(0.4)
    _timeline(result, cfg)

    ui_theme.spacer(0.4)
    _narrative(result)

    ui_theme.spacer(0.4)
    _secondary_stats(result)
