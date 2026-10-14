"""Transactions, FIFO tax lots and the money-weighted return.

This page answers the questions a value-weighted price screen cannot: what did
the trades actually BOOK, and what did *my* money earn given when I put it in.

Concretely:

* XIRR - the money-weighted return of the real cash-flow schedule - shown
  against the time-weighted annualised return so the timing effect is visible
  as a number rather than implied;
* every closed position matched FIFO, one row per matched slice, split by
  holding period (India treats listed equity held more than 365 days as
  long-term);
* the open tax lots in the exact order a future sale will consume them;
* the aggregate realised statistics (win rate, profit factor, payoff);
* the raw ledger and its cash-flow timeline.

Nothing here is recomputed: every number is read off ``result.ledger``, which
the pipeline built with :mod:`src.analytics.lots`. The one exception is the
cash-flow bar chart, which is assembled inline because no chart module owns it.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts as ch
from src.analytics import lots
from src.config import CURRENCY_SYMBOLS, DEFAULT_BASE_CURRENCY, PERIOD_LABELS
from src.ui import components as ui
from src.ui import formatters as fmt
from src.ui import guards, tax_panel
from src.ui import theme as ui_theme

#: India: listed-equity gains are long-term beyond 12 months. The engine uses
#: ``models.RealisedTrade.is_long_term`` (strictly more than 365 days); this
#: constant only labels the UI copy so the two can never drift apart silently.
_LONG_TERM_DAYS = 365

_TERM_ALL = "All closed slices"
_TERM_SHORT = "Short-term (365 days or fewer)"
_TERM_LONG = "Long-term (over 365 days)"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _num(value: Any) -> float:
    """Anything to a float, with NaN for whatever will not convert."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _ok(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _as_of(result: Any) -> date:
    """The valuation date the pipeline stamped on this run."""
    stamp = getattr(result, "generated_at", None)
    try:
        return stamp.date()
    except AttributeError:
        return date.today()


def _money(value: Any, currency: str, decimals: int = 0, signed: bool = False) -> str:
    return fmt.fmt_money(value, currency, decimals=decimals, signed=signed)


def _age_days(when: Any, as_of: date) -> float:
    """Calendar days from a lot's open date to the valuation date."""
    try:
        return float((as_of - when).days)
    except (TypeError, AttributeError):
        return float("nan")


def _table_height(n_rows: int, cap: int = 460) -> int:
    return int(min(38 * (n_rows + 1) + 10, cap))


# ---------------------------------------------------------------------------
# 1. money-weighted return
# ---------------------------------------------------------------------------
def _headline(ledger: dict, result: Any, currency: str, as_of: date) -> None:
    ui_theme.section("Money-weighted return", "\U0001f4b8")

    summary = ledger.get("realised_summary") or {}
    txns = list(ledger.get("transactions") or [])
    xirr = _num(ledger.get("xirr"))
    twr = _num((getattr(result, "risk", None) or {}).get("ann_return"))
    realised = _num(summary.get("total_realised"))
    win_rate = _num(summary.get("win_rate"))
    profit_factor = _num(summary.get("profit_factor"))
    n_trades = int(_num(summary.get("n_trades")) or 0)

    gap_pp = (xirr - twr) * 100.0 if _ok(xirr) and _ok(twr) else float("nan")
    gap_delta = f"{fmt.fmt_delta(gap_pp, 2, ' pp')} vs time-weighted" if _ok(gap_pp) else None

    ui.kpi_row(
        [
            {
                "label": "XIRR (money-weighted)",
                "value": fmt.fmt_ratio_pct(xirr),
                "tone": fmt.delta_tone(xirr),
                "delta": gap_delta,
                "delta_tone": fmt.delta_tone(gap_pp),
                "footnote": "ACT/365 IRR of every trade, plus today's value as a terminal inflow",
                "help_text": "What your rupees earned given when you actually committed them.",
            },
            {
                "label": "Realised P&L",
                "value": _money(realised, currency, signed=True),
                "tone": fmt.delta_tone(realised),
                "footnote": f"{n_trades} closed FIFO slices, net of all fees",
                "help_text": "Booked, not marked. Open positions are excluded.",
            },
            {
                "label": "Win rate",
                "value": fmt.fmt_pct(win_rate, 1),
                "tone": "neutral",
                "footnote": "slices closed above their fee-inclusive cost",
                "help_text": "Frequency, not magnitude - read it with the payoff ratio below.",
            },
            {
                "label": "Profit factor",
                "value": fmt.fmt_multiple(profit_factor),
                "tone": fmt.delta_tone(profit_factor - 1.0) if _ok(profit_factor) else "muted",
                "footnote": "gross gains per unit of gross loss",
                "help_text": "Undefined (em dash) when nothing has been closed at a loss.",
            },
        ]
    )

    first_trade = _first_trade_date(txns)
    span = fmt.humanize_days((as_of - first_trade).days) if first_trade is not None else fmt.DASH
    ui.stat_grid(
        [
            ("Money-weighted (XIRR)", fmt.fmt_ratio_pct(xirr)),
            ("Time-weighted (annualised)", fmt.fmt_ratio_pct(twr)),
            (
                "Timing gap",
                f"{fmt.fmt_signed(gap_pp, 2)} pp" if _ok(gap_pp) else fmt.DASH,
            ),
            ("Ledger span", span),
        ],
        columns=4,
    )

    st.caption(_timing_sentence(xirr, twr, gap_pp))
    window = _window_sentence(result, first_trade, as_of)
    if window:
        st.caption(window)

    ui.methodology(
        "Money-weighted (XIRR) vs time-weighted return",
        _XIRR_METHODOLOGY.format(
            xirr=fmt.fmt_ratio_pct(xirr),
            twr=fmt.fmt_ratio_pct(twr),
            long_term=_LONG_TERM_DAYS,
        ),
    )


def _first_trade_date(txns: list) -> date | None:
    dates = [getattr(t, "trade_date", None) for t in txns]
    dates = [d for d in dates if isinstance(d, date)]
    return min(dates) if dates else None


def _timing_sentence(xirr: float, twr: float, gap_pp: float) -> str:
    """One sentence that teaches the XIRR/TWR distinction with this ledger's numbers."""
    if not _ok(xirr) and not _ok(twr):
        return (
            "Neither return could be solved: XIRR needs at least one buy and one "
            "sell (or an open position with a value), and the time-weighted figure "
            "needs a usable price history."
        )
    if not _ok(xirr):
        return (
            "XIRR is undefined for this schedule - the cash flows never change "
            "sign, so there is no rate to solve for. The time-weighted return "
            f"({fmt.fmt_ratio_pct(twr)}) still describes how the holdings behaved."
        )
    if not _ok(twr):
        return (
            f"XIRR {fmt.fmt_ratio_pct(xirr)} is the annualised rate your own cash "
            "schedule earned. The time-weighted comparison is unavailable because "
            "the analysis window has no usable value series."
        )
    if abs(gap_pp) < 0.25:
        return (
            f"XIRR {fmt.fmt_ratio_pct(xirr)} against a time-weighted "
            f"{fmt.fmt_ratio_pct(twr)}: the two agree to within "
            f"{abs(gap_pp):.2f} pp, which is what happens when capital went in as "
            "one lump near the start. Timing added nothing and cost nothing."
        )
    if gap_pp > 0:
        return (
            f"XIRR {fmt.fmt_ratio_pct(xirr)} beats the time-weighted "
            f"{fmt.fmt_ratio_pct(twr)} by {abs(gap_pp):.2f} pp: the weight of your "
            "capital landed in the better stretches. Timing helped - the strategy "
            "alone does not explain the outcome."
        )
    return (
        f"XIRR {fmt.fmt_ratio_pct(xirr)} trails the time-weighted "
        f"{fmt.fmt_ratio_pct(twr)} by {abs(gap_pp):.2f} pp: the biggest cheques "
        "landed before the weakest stretches. The strategy earned more than the "
        "investor did, which is a scheduling problem, not a selection problem."
    )


def _window_sentence(result: Any, first_trade: date | None, as_of: date) -> str:
    """State plainly that the two returns are measured over different windows."""
    values = getattr(result, "values", None)
    if values is None or not getattr(result, "has_history", False):
        return ""
    try:
        start = values.index[0]
        end = values.index[-1]
    except (AttributeError, IndexError, KeyError):
        return ""
    cfg = getattr(result, "config", None)
    period = PERIOD_LABELS.get(str(getattr(cfg, "period", "")), "the selected window")
    ledger_from = fmt.fmt_date(first_trade) if first_trade else fmt.DASH
    return (
        f"The windows are not the same: the time-weighted figure covers "
        f"{period.lower()} of prices ({result.n_days} sessions, "
        f"{fmt.fmt_date(start)} to {fmt.fmt_date(end)}), while XIRR spans the whole "
        f"ledger, {ledger_from} to {fmt.fmt_date(as_of)}. Part of any gap is "
        "calendar, not skill."
    )


_XIRR_METHODOLOGY = """
**XIRR - money-weighted, and it is about you.** The single annualised rate `r`
that solves

    sum_t  CF_t / (1 + r) ** (d_t / 365)  =  0

over every dated cash flow in the ledger, with the portfolio's current value
appended as a terminal inflow "as if liquidated today". Sign convention is the
investor's pocket: a BUY is negative, a SELL positive. Day count is ACT/365.
The solver is bracketed (Brent, with a Newton fallback) over
`(-99.9%, +10000%)` and every candidate root is substituted back before it is
returned - a schedule with no sign change yields an em dash, never a guess.
**This run: {xirr}.**

**Time-weighted - it is about the strategy.** `risk["ann_return"]` is the
geometric annualisation of the portfolio value series,
`(V_end / V_start) ** (252 / (n - 1)) - 1`. It behaves as though one rupee had
been invested on the first day and left alone, which makes it blind to the size
and timing of contributions by construction. That blindness is the point: it is
the number you use to judge a strategy, a fund or a manager, because they do not
control when you wire money. **This run: {twr}.**

**Why they differ.** They coincide only when capital went in as a single lump at
the start and never moved. Add to a winner before it runs and XIRR rises above
TWR; add just before a drawdown and it falls below. The gap is therefore a
measure of *contribution timing*, not of selection.

**Caveats worth stating.**

* The two are annualised on different day counts - ACT/365 calendar days for
  XIRR, 252 trading days for the time-weighted figure - so a small residual gap
  is convention, not economics.
* They also cover different windows unless your ledger happens to start exactly
  where the price history does.
* XIRR only knows the flows you entered. Dividends, bonus issues, buybacks and
  any cash held outside these trades are invisible to it unless you record them
  as transactions.
* The terminal inflow is today's mark. An illiquid or stale-priced holding
  flatters or punishes the rate directly.
* A schedule with several sign reversals can admit more than one mathematically
  valid IRR; the bracketed solver returns the one inside the search domain and
  verifies it, but the ambiguity is inherent to IRR, not to this implementation.
"""


# ---------------------------------------------------------------------------
# 2. realised trades
# ---------------------------------------------------------------------------
def _realised_block(ledger: dict, currency: str) -> None:
    ui_theme.section("Realised trades", "\U0001f4d5")

    trades = list(ledger.get("realised") or [])
    frame = lots.realised_dataframe(trades)
    if frame.empty:
        ui.empty_state(
            "Nothing has been closed yet",
            "Every lot in this ledger is still open, so there is no realised P&L "
            "and no holding-period split to report. Unrealised performance lives "
            "on the Holdings and Performance pages.",
            icon="\U0001f4d6",
        )
        return

    n_long = int((frame["term"] == "Long").sum())
    n_short = int(len(frame) - n_long)
    long_basis = float(frame.loc[frame["term"] == "Long", "cost_basis"].sum())
    total_basis = float(frame["cost_basis"].sum())
    long_share = long_basis / total_basis * 100.0 if total_basis > 0 else float("nan")

    ui.stat_grid(
        [
            ("Closed slices", f"{len(frame):,d}"),
            ("Short-term slices", f"{n_short:,d}"),
            ("Long-term slices", f"{n_long:,d}"),
            ("Long-term share of closed basis", fmt.fmt_pct(long_share, 1)),
        ],
        columns=4,
    )

    choice = st.radio(
        "Holding-period filter",
        (_TERM_ALL, _TERM_SHORT, _TERM_LONG),
        horizontal=True,
        key="txn_term_filter",
        label_visibility="collapsed",
    )
    view = frame
    if choice == _TERM_SHORT:
        view = frame.loc[frame["term"] == "Short"]
    elif choice == _TERM_LONG:
        view = frame.loc[frame["term"] == "Long"]

    if view.empty:
        st.caption("No closed slices in that holding-period bucket.")
    else:
        shown = float(view["realised_pnl"].sum())
        st.caption(
            f"{len(view):,d} slice(s) shown, {_money(shown, currency, signed=True)} "
            "realised between them. Sorted by close date, oldest first; one row per "
            "FIFO slice, so a single sell order that ate three lots appears three "
            "times."
        )
        ui.dataframe(_realised_display(view, currency), height=_table_height(len(view)))

    st.caption(
        f"**Holding period, not tax advice.** A slice is labelled long-term when it "
        f"was held strictly more than {_LONG_TERM_DAYS} calendar days, which is the "
        "threshold India applies to listed equity and equity mutual funds; "
        "everything else is short-term. That is a classification of *your own trade "
        "dates* and nothing more. It applies no rate, no exemption, no "
        "grandfathering, no set-off of losses, no indexation, no securities "
        "transaction tax logic, and it does not know whether an instrument is "
        "listed equity, debt, gold or foreign. Use it to see the shape of the book, "
        "then take the actual numbers to a tax professional."
    )

    ui.methodology(
        "How a realised slice is matched and priced",
        _REALISED_METHODOLOGY.format(long_term=_LONG_TERM_DAYS),
    )

    ui.download_row(
        [
            (
                "Download realised trades (CSV)",
                frame.to_csv(index=False).encode("utf-8"),
                "realised-trades.csv",
                "text/csv",
            )
        ]
    )


def _realised_display(frame: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Formatted view of ``lots.realised_dataframe`` output."""
    return pd.DataFrame(
        {
            "Ticker": frame["ticker"],
            "Opened": frame["open_date"].map(fmt.fmt_date),
            "Closed": frame["close_date"].map(fmt.fmt_date),
            "Qty": frame["quantity"].map(fmt.fmt_shares),
            "Cost/share": frame["cost_per_share"].map(lambda v: _money(v, currency, decimals=2)),
            "Sale/share": frame["sale_per_share"].map(lambda v: _money(v, currency, decimals=2)),
            "Fees": frame["fees"].map(lambda v: _money(v, currency, decimals=2)),
            "Cost basis": frame["cost_basis"].map(lambda v: _money(v, currency)),
            "Proceeds": frame["proceeds"].map(lambda v: _money(v, currency)),
            "Realised P&L": frame["realised_pnl"].map(lambda v: _money(v, currency, signed=True)),
            "Return": frame["realised_pct"].map(lambda v: fmt.fmt_pct(v, 2, signed=True)),
            "Held": frame["holding_days"].map(fmt.humanize_days),
            "Term": frame["term"]
            .map({"Long": "Long-term", "Short": "Short-term"})
            .fillna(fmt.DASH),
        }
    )


_REALISED_METHODOLOGY = """
**Matching: FIFO, per ticker.** A sale consumes the oldest open lot first, then
the next, until it is filled. This is the default cost-basis method for listed
equity in India and the usual fallback elsewhere. Specific-lot and average-cost
matching are deliberately not offered - quietly applying the wrong method is
worse than not offering the choice.

**One row per matched slice.** A sell of 100 shares that spans three purchase
lots produces three rows, each with its own open date, cost and holding period.
That is not padding: those three pieces genuinely have different tax characters,
and collapsing them would hide a short-term slice inside a long-term sale.

**Fees, both directions.**

* Buy fees CAPITALISE: a lot opened by `BUY q @ p` with `f` of brokerage carries
  `cost_per_share = p + f / q`, so the fee has to be earned back before the
  trade counts as a win.
* Sell fees are APPORTIONED pro-rata across the slices the sale consumed, by
  matched quantity, with the last slice absorbing the rounding residual. The
  slice fees therefore sum exactly to the fee paid - anything else either
  double-counts it or loses it.

**Arithmetic per slice.**

    cost_basis   = q * (buy_price + buy_fees / buy_quantity)
    proceeds     = q * sell_price - apportioned_sell_fees
    realised_pnl = proceeds - cost_basis
    realised_pct = realised_pnl / cost_basis * 100        (NaN when basis <= 0)

**Term.** `holding_days = close_date - open_date` in calendar days; long-term is
strictly greater than {long_term}. Trade dates, not settlement dates - if your
broker reports on a T+1 settlement basis, a slice sitting exactly on the
boundary can differ by a day.

**Limits.** No FX conversion is applied: the engine books each slice in the
currency the trade was entered in. Corporate actions (splits, bonuses, mergers)
are not adjusted for - if a split happened mid-holding, restate the affected
rows yourself. Selling more than you hold never creates a negative lot; the
excess is dropped and reported as a data note instead.
"""


# ---------------------------------------------------------------------------
# 3. realised statistics
# ---------------------------------------------------------------------------
def _realised_stats(ledger: dict, currency: str) -> None:
    summary = ledger.get("realised_summary") or {}
    n_trades = int(_num(summary.get("n_trades")) or 0)
    if n_trades <= 0:
        return

    ui_theme.section("Realised P&L statistics", "\U0001f4d0")

    avg_win = _num(summary.get("avg_win"))
    avg_loss = _num(summary.get("avg_loss"))
    win_rate = _num(summary.get("win_rate"))
    profit_factor = _num(summary.get("profit_factor"))

    ui.stat_grid(
        [
            ("Closed slices", f"{n_trades:,d}"),
            ("Total realised P&L", _money(summary.get("total_realised"), currency, signed=True)),
            ("Total cost basis", _money(summary.get("total_cost_basis"), currency)),
            ("Win rate", fmt.fmt_pct(win_rate, 1)),
            ("Average win", _money(avg_win, currency, signed=True)),
            ("Average loss", _money(avg_loss, currency, signed=True)),
            ("Profit factor", fmt.fmt_multiple(profit_factor)),
            ("Average holding period", fmt.humanize_days(summary.get("avg_holding_days"))),
            ("Best slice", _money(summary.get("best"), currency, signed=True)),
            ("Worst slice", _money(summary.get("worst"), currency, signed=True)),
            ("Short-term P&L", _money(summary.get("short_term_pnl"), currency, signed=True)),
            ("Long-term P&L", _money(summary.get("long_term_pnl"), currency, signed=True)),
        ],
        columns=4,
    )

    st.caption(_payoff_sentence(win_rate, avg_win, avg_loss, profit_factor, currency))
    ui.methodology(
        "Realised statistics: definitions and conventions",
        _STATS_METHODOLOGY.format(long_term=_LONG_TERM_DAYS),
    )


def _payoff_sentence(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    profit_factor: float,
    currency: str,
) -> str:
    """Tie win rate, payoff and profit factor into one honest reading."""
    if not _ok(avg_win) or not _ok(avg_loss) or avg_loss >= 0.0:
        if _ok(profit_factor):
            return (
                f"Profit factor {fmt.fmt_multiple(profit_factor)}: every unit of "
                "gross loss was covered by that much gross gain. Below 1.00x the "
                "closed book destroyed capital."
            )
        return (
            "Nothing has been closed at a loss yet, so profit factor and the "
            "breakeven hit rate are undefined rather than infinite. One losing "
            "exit will define them."
        )

    payoff = avg_win / abs(avg_loss)
    breakeven = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100.0
    verdict = (
        "comfortably above the bar"
        if _ok(win_rate) and win_rate - breakeven > 5.0
        else "only just clearing the bar"
        if _ok(win_rate) and win_rate >= breakeven
        else "below the bar"
    )
    return (
        f"Average win {_money(avg_win, currency)} against an average loss of "
        f"{_money(abs(avg_loss), currency)} is a payoff of "
        f"{fmt.fmt_multiple(payoff)}, so this book needs to win "
        f"{fmt.fmt_pct(breakeven, 1)} of its slices just to break even - it wins "
        f"{fmt.fmt_pct(win_rate, 1)}, {verdict}. A high win rate with a payoff "
        "under 1.00x is the classic pattern of cutting winners and holding losers."
    )


_STATS_METHODOLOGY = """
Every figure below is computed over CLOSED slices only. Open positions
contribute nothing here - their gains are marks, not money.

* **Closed slices** (`n_trades`) counts matched FIFO slices, not sell orders. A
  sale spanning three lots counts three.
* **Total realised P&L** is the sum of `proceeds - cost_basis`, net of buy fees
  (capitalised into the basis) and sell fees (apportioned across the slices).
* **Total cost basis** is the fee-inclusive capital those closed slices tied up.
* **Win rate** is `wins / all slices * 100`. A slice that closed exactly flat is
  neither a win nor a loss but stays in the denominator, so the win rate is not
  `wins / (wins + losses)`.
* **Average win / average loss** are the plain means of the positive and the
  negative slices. The average loss is negative by construction.
* **Profit factor** is `gross wins / |gross losses|`. With no losing slice it is
  reported as an em dash - the ratio is undefined, and printing "infinite" would
  read as a result rather than an absence.
* **Best / worst** are single-slice extremes, in money. A large slice can
  dominate them; read them next to the average, not instead of it.
* **Short- / long-term P&L** split at strictly more than {long_term} calendar
  days held. Either side is an em dash when no slice falls in it.
* **Average holding period** is an UNWEIGHTED mean of calendar days across
  slices - a 1-share slice counts as much as a 1,000-share one. It describes
  behaviour, not exposure.

Two honest limitations. First, none of these statistics is annualised or
risk-adjusted: a 40% gain over four years and a 40% gain over four months look
identical here. Second, the derived breakeven hit rate quoted above,
`|avg_loss| / (avg_win + |avg_loss|)`, assumes the next slice is drawn from the
same payoff distribution as the past ones, which a changing position size or
strategy quietly breaks.
"""


# ---------------------------------------------------------------------------
# 4. open tax lots
# ---------------------------------------------------------------------------
def _open_lots_block(ledger: dict, currency: str, as_of: date) -> None:
    ui_theme.section("Open tax lots", "\U0001f4e6")

    open_lots = list(ledger.get("open_lots") or [])
    frame = lots.lots_dataframe(open_lots)
    if frame.empty:
        ui.empty_state(
            "No open lots",
            "The ledger closes every position it opens, so there is nothing left "
            "to hold and no future sale to match against.",
            icon="\U0001f4ed",
        )
        return

    ages = frame["open_date"].map(lambda d: _age_days(d, as_of))
    is_long = ages > float(_LONG_TERM_DAYS)
    basis = pd.to_numeric(frame["cost_basis"], errors="coerce")
    total_basis = float(basis.sum())
    long_basis = float(basis.where(is_long, 0.0).sum())
    long_share = long_basis / total_basis * 100.0 if total_basis > 0 else float("nan")

    ui.stat_grid(
        [
            ("Open lots", f"{len(frame):,d}"),
            ("Tickers", f"{frame['ticker'].nunique():,d}"),
            ("Open cost basis", _money(total_basis, currency)),
            ("Already long-term, by basis", fmt.fmt_pct(long_share, 1)),
        ],
        columns=4,
    )

    display = pd.DataFrame(
        {
            "Ticker": frame["ticker"],
            "Opened": frame["open_date"].map(fmt.fmt_date),
            "Age": ages.map(fmt.humanize_days),
            "Qty": frame["quantity"].map(fmt.fmt_shares),
            "Cost/share": frame["cost_per_share"].map(lambda v: _money(v, currency, decimals=2)),
            "Cost basis": basis.map(lambda v: _money(v, currency)),
            "If sold today": ages.map(
                lambda a: (
                    fmt.DASH
                    if not _ok(a)
                    else ("Long-term" if a > _LONG_TERM_DAYS else "Short-term")
                )
            ),
            "Days to long-term": ages.map(_days_to_long_term),
        }
    )
    ui.dataframe(display, height=_table_height(len(display)))

    st.caption(
        f"{len(frame):,d} open lot(s) across {frame['ticker'].nunique():,d} ticker(s), "
        f"{_money(total_basis, currency)} of fee-inclusive cost still at risk, of "
        f"which {fmt.fmt_pct(long_share, 1)} has already crossed the "
        f"{_LONG_TERM_DAYS}-day line. Rows are in FIFO order: the top lot for a "
        "ticker is the one your next sale of it will consume."
    )

    ui.methodology(
        "What an open lot is, and what this table deliberately omits",
        _LOTS_METHODOLOGY.format(long_term=_LONG_TERM_DAYS),
    )

    ui.download_row(
        [
            (
                "Download open lots (CSV)",
                frame.to_csv(index=False).encode("utf-8"),
                "open-tax-lots.csv",
                "text/csv",
            )
        ]
    )


def _days_to_long_term(age: float) -> str:
    if not _ok(age):
        return fmt.DASH
    remaining = _LONG_TERM_DAYS + 1 - int(age)
    return f"{remaining:,d}" if remaining > 0 else fmt.DASH


_LOTS_METHODOLOGY = """
**A lot is one purchase that has not yet been sold.** Every BUY opens a lot at
`cost_per_share = price + fees / quantity`; a SELL consumes the oldest lot for
that ticker first and can leave a partial lot behind. What survives is this
table, in exactly the order a future sale will eat it.

**Age and the term column** are measured to the run's valuation date, in
calendar days. "If sold today" applies the same rule the realised table uses -
strictly more than {long_term} days is long-term - so you can see which lots are
short of the line and by how many days. "Days to long-term" is that shortfall,
`{long_term} + 1 - age`.

**What is NOT here, on purpose.** These are cost figures only: no current price,
no market value, no unrealised P&L. The lot engine never touches the price feed,
which is what keeps it deterministic and testable. Unrealised performance lives
on the Holdings and Performance pages, computed from live quotes.

**Caveats.** Ageing a lot toward a tax threshold is a description, not a
recommendation - holding an unwanted position for a date on the calendar is a
real risk decision. Corporate actions are not adjusted for, and no FX conversion
is applied to lots opened in another currency.
"""


# ---------------------------------------------------------------------------
# 5. cash-flow timeline
# ---------------------------------------------------------------------------
def _cash_flow_block(ledger: dict, currency: str, symbol: str) -> None:
    ui_theme.section("Cash-flow timeline", "\U0001f30a")

    flows = list(ledger.get("cash_flows") or [])
    if not flows:
        st.caption("No dated cash flows in this ledger.")
        return

    frame = _flow_frame(flows)
    if frame.empty:
        st.caption("No usable dated cash flows in this ledger.")
        return

    ui.chart(_cash_flow_figure(frame, symbol))

    out = float(-frame.loc[frame["amount"] < 0, "amount"].sum())
    back = float(frame.loc[frame["amount"] > 0, "amount"].sum())
    net = back - out
    st.caption(
        f"{_money(out, currency)} left your pocket across {int((frame['amount'] < 0).sum()):,d} "
        f"buying day(s); {_money(back, currency)} came back on "
        f"{int((frame['amount'] > 0).sum()):,d} selling day(s), for a net "
        f"{_money(net, currency, signed=True)} still committed to the market. The "
        "right-hand line is that running commitment - it is the capital base XIRR "
        "weights every return by, which is why a return earned when the line is "
        "high matters far more than the same percentage earned when it is low. "
        "Colour marks direction of cash, not quality of trade."
    )

    ui.methodology(
        "How this timeline is built",
        _FLOWS_METHODOLOGY,
    )


def _flow_frame(flows: list) -> pd.DataFrame:
    """``[(date, amount)]`` to a clean, date-aggregated frame."""
    frame = pd.DataFrame(list(flows), columns=["date", "amount"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "amount"])
    if frame.empty:
        return frame
    daily = frame.groupby("date", as_index=False)["amount"].sum()
    return daily.sort_values("date").reset_index(drop=True)


def _cash_flow_figure(frame: pd.DataFrame, symbol: str) -> go.Figure:
    """Sign-coloured daily net flows with cumulative capital deployed.

    Built inline rather than in ``src.charts`` because no chart module owns the
    ledger; it still goes through ``charts.theme`` so it matches every other
    figure in the app.
    """
    palette = ch.theme.palette()
    deployed = (-frame["amount"]).cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["amount"],
            name="Net cash flow",
            marker_color=ch.theme.sign_colors(frame["amount"]),
            hovertemplate="%{x|%d %b %Y}<br>Net flow " + symbol + "%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=deployed,
            name="Capital deployed (right)",
            mode="lines",
            line={"color": palette["accent"], "width": 1.6, "shape": "hv"},
            yaxis="y2",
            hovertemplate="%{x|%d %b %Y}<br>Deployed " + symbol + "%{y:,.0f}<extra></extra>",
        )
    )

    ch.theme.apply(
        fig,
        title="Cash flows and cumulative capital deployed",
        subtitle="Investor sign convention: money leaving your pocket is negative. "
        "Trades on the same date are netted.",
        height=380,
    )
    ch.theme.money_axis(fig, symbol, "y")
    ch.theme.add_zero_line(fig, "y")
    fig.update_layout(
        bargap=0.55,
        yaxis2={
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickprefix": symbol,
            "tickformat": "~s",
            "separatethousands": True,
            "title": {"text": "Net deployed"},
        },
    )
    return fig


_FLOWS_METHODOLOGY = """
**The bars** are `Transaction.cash_flow` summed per calendar date:
`-(quantity * price + fees)` for a BUY, `+(quantity * price - fees)` for a SELL.
Sign is from the investor's pocket, which is the convention XIRR requires. Two
trades on the same date are netted into one bar, so a same-day round trip shows
its net, not its gross.

**The line** is the negative running sum of those flows - the capital you have
committed and not yet taken back. It falls when you sell and rises when you buy;
it is not a valuation and it is not marked to market, so it can sit above or
below what the portfolio is actually worth.

**What is missing.** Only recorded trades appear. Dividends, interest, deposits
that were never invested, and any cash swept in or out of the account are absent
unless you entered them as transactions - and each of them would move XIRR.
"""


# ---------------------------------------------------------------------------
# 6. raw ledger
# ---------------------------------------------------------------------------
def _ledger_table(ledger: dict, currency: str) -> None:
    ui_theme.section("Trade ledger", "\U0001f5d2")

    txns = list(ledger.get("transactions") or [])
    if not txns:
        st.caption("No parsed transactions.")
        return

    frame = _transactions_frame(txns, currency)
    n_buys = sum(1 for t in txns if str(getattr(t, "action", "")).upper() == "BUY")
    n_sells = len(txns) - n_buys
    first = _first_trade_date(txns)
    last = max(
        (t.trade_date for t in txns if isinstance(getattr(t, "trade_date", None), date)),
        default=None,
    )
    tickers = sorted({str(getattr(t, "ticker", "")).upper() for t in txns} - {""})
    fees = sum(_num(getattr(t, "fees", 0.0)) or 0.0 for t in txns)

    ui.stat_grid(
        [
            ("Transactions", f"{len(txns):,d}"),
            ("Buys / sells", f"{n_buys:,d} / {n_sells:,d}"),
            ("Tickers traded", f"{len(tickers):,d}"),
            ("Fees paid", _money(fees, currency, decimals=2)),
        ],
        columns=4,
    )
    ui.dataframe(frame, height=_table_height(len(frame)))
    st.caption(
        f"Every row the parser accepted, {fmt.fmt_date(first)} to "
        f"{fmt.fmt_date(last)}, sorted chronologically with buys settling ahead of "
        "sells on a shared date - the order FIFO matching requires so that a "
        "same-day round trip is not mistaken for an oversell. Rows the parser "
        "rejected are listed under Data notes in the sidebar, not silently dropped."
    )


def _transactions_frame(txns: list, currency: str) -> pd.DataFrame:
    has_note = any(str(getattr(t, "note", "") or "").strip() for t in txns)
    data = {
        "Date": [fmt.fmt_date(getattr(t, "trade_date", None)) for t in txns],
        "Ticker": [str(getattr(t, "ticker", "")) for t in txns],
        "Action": [str(getattr(t, "action", "")) for t in txns],
        "Qty": [fmt.fmt_shares(getattr(t, "quantity", None)) for t in txns],
        "Price": [_money(getattr(t, "price", None), currency, decimals=2) for t in txns],
        "Fees": [_money(getattr(t, "fees", None), currency, decimals=2) for t in txns],
        "Gross value": [_money(getattr(t, "gross_value", None), currency) for t in txns],
        "Cash flow": [_money(getattr(t, "cash_flow", None), currency, signed=True) for t in txns],
    }
    if has_note:
        data["Note"] = [str(getattr(t, "note", "") or "") for t in txns]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# data-integrity notes
# ---------------------------------------------------------------------------
def _integrity_notes(ledger: dict, result: Any) -> None:
    """Two limitations that silently corrupt ledger numbers if left unsaid."""
    txns = list(ledger.get("transactions") or [])
    messages: list[str] = []

    currencies = sorted({str(getattr(t, "currency", "") or "").upper() for t in txns} - {""})
    if len(currencies) > 1:
        messages.append(
            "This ledger mixes " + ", ".join(currencies) + ". Lot accounting books "
            "each trade in its own currency and applies no FX conversion, so the "
            "realised totals, cost bases and XIRR on this page are adding different "
            "units together. Split the ledger by currency to read them safely."
        )

    positions = getattr(result, "positions", None)
    try:
        held = {str(t).upper() for t in positions["ticker"]}
    except (TypeError, KeyError, IndexError):
        held = set()
    traded = {str(getattr(t, "ticker", "")).upper() for t in txns} - {""}
    untraded = sorted(held - traded)
    if held and untraded:
        messages.append(
            f"{len(untraded)} holding(s) have no trades in the ledger "
            f"({', '.join(untraded[:6])}{'...' if len(untraded) > 6 else ''}). Their "
            "cost basis comes from the average buy price you typed, so they "
            "contribute to today's value in XIRR's terminal inflow but never appear "
            "as a dated outflow. That flatters the rate; enter their trades to fix it."
        )

    if messages:
        ui.warning_list(messages, "Ledger coverage")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Transactions & Tax Lots",
        "FIFO lot accounting, realised P&L and money-weighted return",
        icon="\U0001f9fe",
    )

    result = guards.require_result()
    if result is None:
        return
    ledger = guards.require_ledger(result)
    if ledger is None:
        return

    cfg = getattr(result, "config", None)
    currency = str(getattr(cfg, "base_currency", DEFAULT_BASE_CURRENCY) or DEFAULT_BASE_CURRENCY)
    symbol = str(getattr(cfg, "currency_symbol", "") or CURRENCY_SYMBOLS.get(currency, ""))
    as_of = _as_of(result)

    _integrity_notes(ledger, result)
    _headline(ledger, result, currency, as_of)
    _realised_block(ledger, currency)
    _realised_stats(ledger, currency)
    _open_lots_block(ledger, currency, as_of)
    _cash_flow_block(ledger, currency, symbol)

    ui_theme.spacer(0.4)
    ui_theme.section("TAX-LOSS HARVESTING", "\U0001f9fe")
    st.caption(
        "What each unrealised loss is actually worth under Indian capital-gains "
        "rules — which is rarely the loss times a tax rate, because the set-off "
        "hierarchy and the long-term exemption both bind."
    )
    tax_panel.render(result, ledger)

    _ledger_table(ledger, currency)
