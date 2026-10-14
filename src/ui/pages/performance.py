"""Performance - how the return was earned, and when.

This page answers four questions the headline number cannot:

* **How did the portfolio grow?** Indexed against the benchmark, so the chart
  compares growth rates rather than a money value against an index level.
* **When did it earn?** A calendar-month grid exposes seasonality, streaks and
  the hit rate - a 20% year made of one explosive month is a different animal
  from a 20% year made of twelve quiet ones.
* **Who earned it?** Return attribution splits the headline percentage across
  holdings so the parts actually sum to the whole, which per-holding percentage
  returns never do.
* **What convention is this?** Everything here is TIME-WEIGHTED: it measures the
  strategy, not the timing of the deposits. The methodology blocks say so out
  loud, and point at the money-weighted alternative.

Every number is read off the ``AnalysisResult`` the pipeline already produced;
this module computes no analytics of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.config import PERIOD_LABELS, TRADING_DAYS
from src.ui import components as ui
from src.ui import guards
from src.ui import theme as ui_theme
from src.ui.formatters import (
    delta_tone,
    fmt_date,
    fmt_money,
    fmt_pct,
    fmt_ratio_pct,
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Callout:
    """Duck-typed stand-in for ``models.Insight`` so ``ui.insight_card`` works."""

    level: str
    text: str
    evidence: str = ""
    category: str = "performance"


def _finite(value: Any) -> float | None:
    """Float when the value is a real, finite number; ``None`` otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_frame(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame) and not obj.empty


def _window_dates(values: Any) -> tuple[Any, Any]:
    index = getattr(values, "index", None)
    if index is None or len(index) == 0:
        return None, None
    return index[0], index[-1]


def _contribution_for(contrib: Any, ticker: str) -> float | None:
    """One holding's contribution in percentage points, or ``None``."""
    if not _is_frame(contrib) or "ticker" not in contrib.columns:
        return None
    row = contrib.loc[contrib["ticker"].astype(str) == str(ticker)]
    if row.empty:
        return None
    return _finite(row["contribution_pct"].iloc[0])


# ---------------------------------------------------------------------------
# headline
# ---------------------------------------------------------------------------
def _kpis(result: Any, cfg: Any) -> None:
    stats = result.return_stats or {}
    start, end = _window_dates(result.values)

    cumulative = _finite(stats.get("cumulative_pct"))
    annualized = _finite(stats.get("annualized_pct"))
    best_month = _finite(stats.get("best_month_pct"))
    worst_month = _finite(stats.get("worst_month_pct"))

    rf_pct = (_finite(getattr(cfg, "risk_free_rate", None)) or 0.0) * 100.0
    over_cash = None if annualized is None else annualized - rf_pct

    cards = [
        {
            "label": "Cumulative return",
            "value": fmt_pct(cumulative, signed=True),
            "tone": delta_tone(cumulative),
            "delta": f"over {result.n_days} trading days",
            "delta_tone": "muted",
            "footnote": f"{fmt_date(start)} to {fmt_date(end)}",
            "sparkline": result.values,
            "help_text": "Total return of the value series: V_end / V_start - 1.",
        },
        {
            "label": "Annualised return",
            "value": fmt_pct(annualized, signed=True),
            "tone": delta_tone(annualized),
            "delta": f"{fmt_pct(over_cash, signed=True)} vs cash",
            "delta_tone": delta_tone(over_cash),
            "footnote": f"geometric, {TRADING_DAYS}-day year; "
            f"cash = {fmt_ratio_pct(getattr(cfg, 'risk_free_rate', None))}",
            "help_text": "Compounded growth rate, not the average of daily returns.",
        },
        {
            "label": "Best month",
            "value": fmt_pct(best_month, signed=True),
            "tone": "gain" if (best_month or 0.0) > 0 else "muted",
            "footnote": "strongest completed calendar month",
            "help_text": "Calendar month, compounded from daily returns.",
        },
        {
            "label": "Worst month",
            "value": fmt_pct(worst_month, signed=True),
            "tone": "loss" if (worst_month or 0.0) < 0 else "muted",
            "footnote": "the month you had to sit through",
            "help_text": "The realised monthly tail - not a modelled one.",
        },
    ]
    ui.kpi_row(cards, columns=4)

    positive = stats.get("positive_months")
    total_months = stats.get("total_months")
    hit = None
    if _finite(total_months) and float(total_months) > 0:
        hit = float(positive or 0) / float(total_months) * 100.0

    win_streak = _finite(stats.get("longest_win_streak"))
    loss_streak = _finite(stats.get("longest_loss_streak"))

    ui_theme.spacer(0.5)
    ui.stat_grid(
        [
            (
                "Positive months",
                f"{int(positive or 0)} of {int(total_months or 0)}"
                if _finite(total_months)
                else "-",
            ),
            ("Monthly hit rate", fmt_pct(hit, decimals=1)),
            ("Best day", fmt_pct(_finite(stats.get("best_day_pct")), signed=True)),
            ("Worst day", fmt_pct(_finite(stats.get("worst_day_pct")), signed=True)),
            (
                "Longest winning run",
                f"{int(win_streak)} days" if win_streak is not None else "-",
            ),
            (
                "Longest losing run",
                f"{int(loss_streak)} days" if loss_streak is not None else "-",
            ),
            ("Trading days", f"{result.n_days:,d}"),
            ("Months observed", f"{int(total_months or 0)}"),
        ],
        columns=4,
    )


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------
def _growth(result: Any, cfg: Any) -> None:
    ui_theme.section("Growth", "\U0001f4c8")

    bench_name = str(getattr(cfg, "benchmark_name", "") or "Benchmark")
    bench_prices = result.benchmark_prices
    bench = result.benchmark or {}
    paired = bool(bench.get("ok")) and bench_prices is not None

    ui.chart(
        ch.growth_line(
            result.values,
            getattr(cfg, "currency_symbol", ""),
            bench_prices,
            bench_name,
        )
    )

    if paired:
        port_ann = _finite(bench.get("portfolio_return"))
        bench_ann = _finite(bench.get("benchmark_return"))
        excess = _finite(bench.get("excess_return"))
        st.caption(
            f"Both legs are rebased to 100 on the first common date, so the slopes "
            f"are comparable even though one is a rupee value and the other an index "
            f"level. The portfolio compounded at {fmt_ratio_pct(port_ann, signed=True)} "
            f"a year against {fmt_ratio_pct(bench_ann, signed=True)} for {bench_name} - "
            f"an annualised excess of {fmt_ratio_pct(excess, signed=True)}. Beating the "
            f"index on return alone says nothing about the risk taken to do it; the "
            f"Benchmark page carries beta, tracking error and capture."
        )
    else:
        _, end = _window_dates(result.values)
        last_value = None
        try:
            last_value = _finite(result.values.iloc[-1])
        except Exception:  # noqa: BLE001 - a malformed series must not break the page
            last_value = None
        st.caption(
            f"Mark-to-market value of the current holdings priced back through the "
            f"window, ending at "
            f"{fmt_money(last_value, getattr(cfg, 'base_currency', 'INR'))} on "
            f"{fmt_date(end)}. No benchmark overlay: the index series could not be "
            f"aligned to this window."
        )

    ui.chart(ch.cumulative_return_chart(result.values, bench_prices, bench_name))

    growth = bench.get("growth") if paired else None
    if _is_frame(growth) and {"Portfolio", "Benchmark"} <= set(growth.columns):
        port_cum = _finite(growth["Portfolio"].iloc[-1])
        bench_cum = _finite(growth["Benchmark"].iloc[-1])
        port_cum = None if port_cum is None else port_cum - 100.0
        bench_cum = None if bench_cum is None else bench_cum - 100.0
        gap = None if (port_cum is None or bench_cum is None) else port_cum - bench_cum
        st.caption(
            f"Both curves start at 0%, so the vertical gap on any date IS the "
            f"cumulative excess return to that point. Over the overlap the portfolio "
            f"finished {fmt_pct(port_cum, signed=True)} against "
            f"{fmt_pct(bench_cum, signed=True)} for {bench_name}, a gap of "
            f"{fmt_pct(gap, signed=True)}. Watch where the gap opens: relative "
            f"performance earned in a single week is a different proposition from "
            f"relative performance ground out month after month."
        )
    else:
        st.caption(
            "Both curves start at 0%, so the vertical distance between any two "
            "points is the return earned between those dates - the shape of the "
            "compounding, not just its endpoint."
        )

    ui.methodology(
        "What the value series actually is (read this before trusting the curve)",
        """
**Construction.** `V_t = sum_i (quantity_i * adjusted_close_(i,t))`, using the
share counts you hold **today**, held constant across the whole window.

**What that means.** This is a look-through backtest of the current book, not a
record of the trades you actually made. If you doubled a position last month,
the chart credits that larger position with the whole window's move. If you sold
a name entirely, it never appears - even though its P&L was real.

**Dividends.** Prices are split- and dividend-adjusted closes, so cash dividends
are treated as reinvested on the ex-date. The curve is therefore a **total
return**, not a price return.

**Costs.** No brokerage, STT, stamp duty, slippage or tax is deducted anywhere on
this page. Real returns are lower, and the gap widens with turnover.

**Window.** All holdings are cut to their overlapping dates. One recently listed
name shortens the window for the entire portfolio - if the window looks shorter
than the history period you asked for, that is why.

**When it is accurate.** For a buy-and-hold book that did not change during the
window, this series equals the real time-weighted return. The further your actual
trading is from that, the more this is a simulation of today's portfolio rather
than a report on yesterday's.
        """,
    )


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def _calendar(result: Any) -> None:
    ui_theme.section("Calendar returns", "\U0001f5d3")

    monthly = result.monthly
    if not _is_frame(monthly):
        ui.empty_state(
            "No completed calendar month",
            "The monthly grid needs at least one full calendar month inside the "
            "window. Widen the history period in the sidebar and the seasonality "
            "view appears.",
            icon="\U0001f5d3",
        )
        return

    ui.chart(ch.monthly_heatmap(monthly))

    stats = result.return_stats or {}
    positive = int(stats.get("positive_months") or 0)
    total = int(stats.get("total_months") or 0)
    hit = (positive / total * 100.0) if total else None
    n_years = int(len(monthly.index))

    st.caption(
        f"{positive} of {total} months finished positive - a "
        f"{fmt_pct(hit, decimals=1)} hit rate across {n_years} calendar "
        f"{'year' if n_years == 1 else 'years'}. A high hit rate with a modest total "
        f"means the portfolio ground out its return; a low hit rate with a strong "
        f"total means a handful of months carried it, and missing those months is "
        f"the whole risk. Blank cells are months the window does not cover - never "
        f"zeroes."
    )

    ui.methodology(
        "Monthly return convention",
        """
**Bucketing.** Daily returns are compounded into calendar months:
`R_m = prod(1 + r_d) - 1` over the days in month `m`. Months are calendar months
(month-end frequency), not rolling 21-day windows.

**Partial months.** The first and last months of the window are included as-is
and are **not** scaled up to a full month. A window starting on the 20th shows
that month's last seven sessions, labelled as the month.

**The `Year` column** compounds only the months actually present in that row:
`prod(1 + R_m) - 1`. A partial year is therefore a partial-year return, **not**
an annualised one - do not read a half-year `Year` cell as a CAGR.

**Colour.** The scale is blue/red diverging and centred on zero, deliberately not
green/red: green and red are reserved for gain/loss elsewhere in the app, and the
pair is unreadable with red-green colour-vision deficiency. `Year` sits on its own
scale because annual figures span a far wider range than single months and would
otherwise wash the monthly grid flat.
        """,
    )


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def _contribution_table(contrib: pd.DataFrame, currency: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": contrib["ticker"].astype(str),
            "Weight": [fmt_pct(v) for v in contrib["weight_pct"]],
            "P&L": [fmt_money(v, currency) for v in contrib["pnl"]],
            "Own return": [fmt_pct(v, signed=True) for v in contrib["pnl_pct"]],
            "Contribution": [fmt_pct(v, signed=True) for v in contrib["contribution_pct"]],
            "Share of P&L": [fmt_pct(v, decimals=1) for v in contrib["contribution_share_pct"]],
        }
    )


def _attribution(result: Any, cfg: Any) -> None:
    ui_theme.section("Return attribution", "\U0001f9ee")

    contrib = (result.attribution or {}).get("return_contribution")
    if not _is_frame(contrib):
        ui.empty_state(
            "Attribution unavailable",
            "Splitting the return across holdings needs a positive cost base and a "
            "computable P&L for at least one position.",
            icon="\U0001f9ee",
        )
        return

    currency = str(getattr(cfg, "base_currency", "INR"))
    ui.chart(ch.contribution_waterfall(contrib, getattr(cfg, "currency_symbol", "")))

    total_return_pct = _finite(contrib.attrs.get("total_return_pct"))
    total_investment = _finite(contrib.attrs.get("total_investment"))
    leader = contrib.iloc[0]

    st.caption(
        f"The steps sum to {fmt_pct(total_return_pct, signed=True)} - the "
        f"portfolio's return on {fmt_money(total_investment, currency)} of invested "
        f"capital - because each bar is that holding's P&L over the **total** cost "
        f"base, not over its own. {leader['ticker']} led with "
        f"{fmt_pct(_finite(leader['contribution_pct']), signed=True)} of the total, "
        f"off a {fmt_pct(_finite(leader['pnl_pct']), signed=True)} own return at a "
        f"{fmt_pct(_finite(leader['weight_pct']))} weight. That gap between own "
        f"return and contribution is the whole point of the chart: conviction only "
        f"pays in proportion to position size."
    )
    st.caption(
        "**This total will not match the cumulative return above, and should not.** "
        "Attribution starts the clock at *your purchase prices*; the growth chart "
        "starts it at the beginning of the *price window*. Two different questions - "
        '"how am I doing on what I paid?" and "how did the book move over this '
        'period?" - and therefore two different answers.'
    )

    ui.dataframe(
        _contribution_table(contrib, currency),
        height=min(38 * (len(contrib) + 1) + 8, 420),
    )

    ui.methodology(
        "Contribution vs return - why the column sums to the portfolio",
        """
**Formula.** `contribution_pct_i = pnl_i / total_investment * 100`, where `pnl_i`
is the holding's unrealised profit in base currency and `total_investment` is the
**whole portfolio's** cost base.

**Why the denominator is shared.** Per-holding returns (`pnl_i / investment_i`,
the *Own return* column) cannot be added or averaged into the portfolio's number:
a 60% gain on 2% of the book and a 3% gain on 40% of the book live on different
denominators. Contributions can be added, because they all share one:

    sum_i contribution_pct_i = total_pnl / total_investment * 100 = portfolio return

Every bar therefore reads as "this name put X percentage points on, or took X off,
the headline return". That identity is what turns a ranking into an attribution.

**Share of P&L** = `pnl_i / total_pnl * 100` splits the profit itself into 100%.
It is undefined - shown as a dash - when net P&L is exactly zero, and it can
exceed 100% or turn negative when winners and losers offset each other.

**Caveats, and they matter.**
- **Different baseline from the growth chart.** The sum here is P&L against your
  *purchase prices*; the cumulative return above is measured from the *start of the
  price window*. Unless you happened to buy everything on the window's first day,
  the two totals differ - legitimately. Neither is the "wrong" number; they answer
  different questions.
- This is a **cost-based, point-in-time** attribution of **unrealised** P&L on the
  book as it stands today. It is not a Brinson attribution and carries no
  allocation/selection split against a benchmark.
- It says nothing about **when** you bought. A name held three weeks and a name
  held three years are compared on the same footing.
- **Closed positions are invisible.** Anything you sold contributed real return
  that this table cannot see. The Trades page covers realised P&L.
- Weights are today's market-value weights; contributions are cost-based. The two
  columns answer different questions and are not expected to rank identically.
        """,
    )


# ---------------------------------------------------------------------------
# per-holding
# ---------------------------------------------------------------------------
def _per_holding(result: Any, cfg: Any) -> None:
    ui_theme.section("Winners and losers", "\U0001f4ca")

    left, right = st.columns(2, gap="medium")
    with left:
        ui.chart(ch.return_bar(result.positions))
        st.caption(
            "Return on cost per holding, ignoring position size entirely. This is "
            "the scoreboard for stock picking."
        )
    with right:
        ui.chart(ch.pnl_bar(result.positions, getattr(cfg, "currency_symbol", "")))
        st.caption(
            "The same book measured in money. This is the scoreboard for portfolio construction."
        )

    st.caption(
        "Read the two together. Wherever the rank order differs, a large position "
        "moved a little or a small one moved a lot - and the waterfall above is "
        "exactly the reconciliation between them."
    )


def _extremes(result: Any) -> None:
    ui_theme.section("Extremes", "\U0001f3c1")

    summary = result.summary or {}
    best = summary.get("best")
    worst = summary.get("worst")
    contrib = (result.attribution or {}).get("return_contribution")
    total_return_pct = (
        _finite(contrib.attrs.get("total_return_pct")) if _is_frame(contrib) else None
    )

    if not best and not worst:
        st.caption(
            "No holding has a computable return on cost yet - every position is "
            "missing either a buy price or a live quote."
        )
        return

    def _evidence(ticker: str) -> str:
        share = _contribution_for(contrib, ticker)
        if share is None:
            return "Unrealised, marked at the latest available price."
        return (
            f"Contributed {fmt_pct(share, signed=True)} of the portfolio's "
            f"{fmt_pct(total_return_pct, signed=True)} total return."
        )

    left, right = st.columns(2, gap="medium")

    with left:
        if best:
            ticker, pct = str(best[0]), _finite(best[1])
            ui.insight_card(
                _Callout(
                    level="positive" if (pct or 0.0) >= 0 else "warning",
                    text=(f"Best performer: {ticker} at {fmt_pct(pct, signed=True)} on cost."),
                    evidence=_evidence(ticker),
                )
            )
        else:
            st.caption("No best performer could be ranked.")

    with right:
        if worst:
            ticker, pct = str(worst[0]), _finite(worst[1])
            ui.insight_card(
                _Callout(
                    level="critical" if (pct or 0.0) < 0 else "positive",
                    text=(f"Worst performer: {ticker} at {fmt_pct(pct, signed=True)} on cost."),
                    evidence=_evidence(ticker),
                )
            )
        else:
            st.caption("No worst performer could be ranked.")

    st.caption(
        "Both are ranked on unrealised return since purchase, so a name bought last "
        "week is compared against one bought years ago. When the worst performer is "
        "still positive, the whole book is in profit."
    )


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
def _conventions(result: Any, cfg: Any) -> None:
    stats = result.return_stats or {}
    annualized = _finite(stats.get("annualized_pct"))
    xirr_pct = None
    ledger = result.ledger or {}
    if ledger.get("transactions"):
        raw_xirr = _finite(ledger.get("xirr"))
        xirr_pct = None if raw_xirr is None else raw_xirr * 100.0

    if xirr_pct is not None and annualized is not None:
        gap = annualized - xirr_pct
        timing = (
            "Your money-weighted return is **below** the time-weighted one, which "
            "means capital was, on average, added before the weaker stretches."
            if gap > 0
            else "Your money-weighted return is **above** the time-weighted one, "
            "which means capital was, on average, added before the stronger "
            "stretches."
        )
        comparison = (
            f"\n\n**Your two numbers.** Time-weighted "
            f"{fmt_pct(annualized, signed=True)} a year against a ledger XIRR of "
            f"{fmt_pct(xirr_pct, signed=True)} - a gap of "
            f"{fmt_pct(abs(gap))} attributable to deposit timing, not to the "
            f"holdings. {timing}"
        )
    else:
        comparison = (
            "\n\n**Your money-weighted number.** Not available: XIRR needs the dated "
            "trades, which average cost alone cannot supply. Add a trade ledger on "
            "the Portfolio page and the Trades page reports it."
        )

    ui.methodology(
        "Return conventions - time-weighted vs money-weighted",
        f"""
**Time-weighted return (TWR) - what this whole page reports.** Computed from the
portfolio's value series, which holds share counts constant. It measures the
compounded growth of one unit of capital left invested for the entire window,
**independent of when money went in or came out**. It is the only fair basis for
comparison against an index, because an index takes no deposits either, and it is
why GIPS and every fund factsheet quote it. In short: TWR grades the **strategy**.

**Money-weighted return (MWR, reported as XIRR) - what your statement feels like.**
Solves for the single annual discount rate `r` that sets the net present value of
your actual dated cash flows to zero, with today's portfolio value appended as a
terminal inflow. A large deposit made just before a rally lifts it; the same
deposit made just before a sell-off sinks it. In short: MWR grades **your timing**
as much as the portfolio.

**Neither is wrong, and the gap between them is the message.** TWR above MWR means
the average rupee arrived at a bad moment; MWR above TWR means it arrived at a good
one. A fund manager should be judged on TWR because they do not control the flows;
you should look at both, because you do.{comparison}

**Conventions used on this page**
- Annualised return is **geometric** (CAGR-style), computed from the endpoints and
  the calendar span - not the arithmetic mean of daily returns, which is always
  higher and is not a return anyone earned.
- Annualisation uses **{TRADING_DAYS} trading days** per year, and the risk-free
  rate used for the "vs cash" comparison is
  **{fmt_ratio_pct(getattr(cfg, "risk_free_rate", None))}**, set in the sidebar.
- Monthly figures are **calendar** months compounded from daily returns; partial
  months at either end are included unscaled.
- Prices are **split- and dividend-adjusted closes**, so this is a total return
  with dividends reinvested on the ex-date.
- **Gross of everything**: no brokerage, taxes, stamp duty or slippage.
- The value series is a **look-through** of today's holdings. Positions opened or
  closed inside the window are not reflected in it, so where your book changed a
  lot, treat these figures as a backtest of the portfolio you hold now.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Performance",
        "Growth, seasonality and who actually earned the return",
        icon="\U0001f4c8",
    )

    result = guards.require_result()
    if result is None:
        return
    if not guards.require_history(result, "Performance analysis"):
        return

    cfg = result.config
    summary = result.summary or {}

    ui.badges(
        [
            (
                PERIOD_LABELS.get(getattr(cfg, "period", ""), str(getattr(cfg, "period", ""))),
                "accent",
            ),
            (f"vs {getattr(cfg, 'benchmark_name', 'benchmark')}", "neutral"),
            (f"{int(summary.get('n_holdings', 0))} holdings", "neutral"),
            (str(getattr(cfg, "base_currency", "INR")), "neutral"),
            ("Time-weighted", "neutral"),
        ]
    )

    _kpis(result, cfg)
    _growth(result, cfg)
    _calendar(result)
    _attribution(result, cfg)
    _per_holding(result, cfg)
    _extremes(result)

    ui_theme.section("Conventions", "\U0001f4d0")
    _conventions(result, cfg)
