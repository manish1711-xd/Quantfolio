"""Rebalance -- from drift to an executable order ticket.

This page answers three questions in the order a trader actually asks them:
how far has the book drifted from its policy weights, which of those
deviations are large enough to be worth paying to fix, and what exactly gets
sent to the broker.

Targets come from one of four sources -- equal weight, today's weights, the
weights handed over by the Optimiser page, or hand-entered -- and everything
below them is produced by :mod:`src.analytics.rebalance`: ``drift_table`` for
the worksheet, ``rebalance_trades`` for the order list, ``rebalance_summary``
for the economics. Nothing is sized here. This page chooses the inputs, states
the conventions and is honest about what a trade list computed from closing
prices cannot know.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import rebalance as rb
from src.config import DATA_SOURCE, DEFAULT_REBALANCE_BAND_PCT
from src.ui import components as ui
from src.ui import guards, session
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    delta_tone,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_shares,
    fmt_signed,
)

# ---------------------------------------------------------------------------
# target sources
# ---------------------------------------------------------------------------
_SOURCE_EQUAL = "Equal weight"
_SOURCE_CURRENT = "Current weights"
_SOURCE_OPTIMISER = "From optimiser"
_SOURCE_MANUAL = "Manual"

_SOURCES: tuple[str, ...] = (
    _SOURCE_EQUAL,
    _SOURCE_CURRENT,
    _SOURCE_OPTIMISER,
    _SOURCE_MANUAL,
)

# ---------------------------------------------------------------------------
# control ranges (display only -- the analytics carry their own defaults)
# ---------------------------------------------------------------------------
#: Widest band offered. Past 20 percentage points a "policy" stops describing
#: anything: a 20% target with a 20pp band permits everything from 0% to 40%.
_BAND_MAX_PCT = 20.0
_BAND_STEP_PCT = 0.5

#: Typing granularity for the cash box. The value itself is free-form.
_CASH_STEP = 1000.0

#: Two-way turnover above this share of the book turns the KPI card amber.
#: A prompt to check the cost of the list, not a rule about it.
_HIGH_TURNOVER_PCT = 25.0

#: How far a hand-entered target vector may sit from 100% before the sum
#: indicator stops calling it balanced.
_SUM_TOLERANCE_PCT = 0.05


# ---------------------------------------------------------------------------
# small display helpers
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    """True when ``x`` is a real, finite number worth printing."""
    try:
        return bool(pd.notna(x)) and float(x) == float(x) and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def _tickers(positions: pd.DataFrame) -> list[str]:
    """Held symbols, in the order the positions table already ranked them."""
    if "ticker" not in positions.columns:
        return []
    seen: list[str] = []
    for raw in positions["ticker"].tolist():
        ticker = str(raw).strip().upper()
        if ticker and ticker not in {"NAN", "NONE", "NAT"} and ticker not in seen:
            seen.append(ticker)
    return seen


def _current_weights(positions: pd.DataFrame) -> dict[str, float]:
    """Ticker -> current weight in PERCENT, straight off the positions table."""
    tickers = _tickers(positions)
    if "weight_pct" not in positions.columns:
        return rb.equal_weight_targets(tickers)
    out: dict[str, float] = {}
    for raw, weight in zip(positions["ticker"], positions["weight_pct"]):
        ticker = str(raw).strip().upper()
        if ticker and _finite(weight):
            out[ticker] = out.get(ticker, 0.0) + float(weight)
    return out or rb.equal_weight_targets(tickers)


def _editor_key(tickers: list[str]) -> str:
    """Widget key that changes when the holdings do.

    ``st.data_editor`` keeps its edits per key, so reusing one key across a
    re-analysis with different holdings would show a stale grid of targets for
    names that are no longer in the book.
    """
    digest = hashlib.md5("|".join(tickers).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"rebalance_manual_targets_{digest}"


def _money_format(cfg: Any) -> str:
    return f"{getattr(cfg, 'currency_symbol', '')}%.0f"


# ---------------------------------------------------------------------------
# 1. target policy
# ---------------------------------------------------------------------------
def _manual_targets(positions: pd.DataFrame, current: dict[str, float]) -> dict[str, float]:
    """Hand-entered target vector, seeded from the weights held today."""
    tickers = _tickers(positions)
    seed = pd.DataFrame(
        {
            "ticker": tickers,
            "target_weight_pct": [round(float(current.get(t, 0.0)), 2) for t in tickers],
        }
    )

    edited = st.data_editor(
        seed,
        num_rows="dynamic",
        width="stretch",
        key=_editor_key(tickers),
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker",
                required=True,
                width="medium",
                help="A symbol you do not hold yet is allowed: it becomes a buy. "
                "Deleting a row sets that name's target to 0% -- a sell.",
            ),
            "target_weight_pct": st.column_config.NumberColumn(
                "Target weight",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                format="%.2f%%",
                help="Share of the whole portfolio, in percent.",
            ),
        },
    )

    targets: dict[str, float] = {}
    tick_col = edited["ticker"] if "ticker" in edited.columns else []
    weight_col = edited["target_weight_pct"] if "target_weight_pct" in edited.columns else []
    for raw, weight in zip(tick_col, weight_col):
        ticker = str(raw).strip().upper()
        if not ticker or ticker in {"NAN", "NONE", "NAT"} or not _finite(weight):
            continue
        if float(weight) < 0:
            continue
        targets[ticker] = targets.get(ticker, 0.0) + float(weight)

    if not targets:
        st.caption(
            "No usable target rows yet. Every row needs a symbol and a "
            "non-negative weight; blank and negative rows are ignored rather "
            "than guessed at."
        )
        return {}

    total = float(sum(targets.values()))
    balanced = abs(total - 100.0) <= _SUM_TOLERANCE_PCT
    ui.badges(
        [
            (f"Targets sum to {fmt_pct(total)}", "gain" if balanced else "warn"),
            (f"{len(targets)} names", "neutral"),
        ]
    )
    if not balanced:
        st.caption(
            f"That is {fmt_signed(total - 100.0)} pp away from a full allocation. "
            f"The vector is renormalised to 100% before anything is sized, so the "
            f"weights below are read as *relative* -- their shape is honoured, "
            f"their scale is not."
        )
    return targets


def _optimiser_targets() -> dict[str, float]:
    """Weights sent over from the Optimiser page, converted to percentages."""
    raw = st.session_state.get(session.TARGET_WEIGHTS) or {}
    targets = rb.targets_from_optimizer(raw)
    if targets:
        st.caption(
            f"{len(targets)} names carried over from the Optimiser page. Optimiser "
            f"output arrives as weight *fractions* summing to 1, so it is scaled to "
            f"percentages here; dust below 0.0001% of the book is numerical noise "
            f"from the solver rather than a position, and is dropped before the "
            f"remainder is renormalised to 100%."
        )
        return targets

    ui.empty_state(
        "No optimiser weights in this session",
        "Nothing has been sent to this page yet. Open Analytics > Optimiser, pick "
        "an objective -- maximum Sharpe, minimum variance, risk parity or maximum "
        "diversification -- and use its 'Send to Rebalance' action. The chosen "
        "weights are stored in session state and land here as target percentages.",
        icon="\U0001f9ee",
        hint="Optimiser weights are estimated from this window's returns and are "
        "only as stable as those estimates. Compare any of them against the "
        "Equal weight option before trading on the difference.",
    )
    return {}


def _target_policy(positions: pd.DataFrame, cfg: Any) -> tuple[dict[str, float], str]:
    """Render the source picker and return ``(targets_pct, source_label)``."""
    source = st.radio(
        "Target allocation",
        options=_SOURCES,
        horizontal=True,
        key="rebalance_target_source",
        help="The policy the book is measured against. A target vector describes "
        "the WHOLE portfolio: a held name that is absent from it has a "
        "target of 0%, not 'no opinion'.",
    )
    current = _current_weights(positions)

    if source == _SOURCE_EQUAL:
        targets = rb.equal_weight_targets(list(_tickers(positions)))
        if targets:
            each = 100.0 / float(len(targets))
            st.caption(
                f"1/N across {len(targets)} names -- {fmt_pct(each)} each. Equal "
                f"weight estimates nothing, so it has no estimation error to be "
                f"wrong about, which is why it remains an awkwardly hard benchmark "
                f"for an optimiser to beat out of sample. It also quietly tilts the "
                f"book toward the smallest, least liquid names in it."
            )
        return targets, source

    if source == _SOURCE_CURRENT:
        st.caption(
            "Policy equals the book as it stands, so every drift is zero by "
            "construction and nothing trades on its own. Pair it with Cash to "
            "invest below to deploy new money pro-rata and hold the current shape "
            "fixed -- the cheapest way to put cash to work when you are happy with "
            "the allocation you already have."
        )
        return dict(current), source

    if source == _SOURCE_OPTIMISER:
        return _optimiser_targets(), source

    return _manual_targets(positions, current), source


def _execution_controls(cfg: Any) -> tuple[float, float, bool]:
    """Band, new cash and the fractional-share switch."""
    ccy = getattr(cfg, "base_currency", "INR")
    left, middle, right = st.columns([1.4, 1.1, 0.9], gap="large")

    with left:
        band = st.slider(
            "Rebalancing band (pp)",
            min_value=0.0,
            max_value=_BAND_MAX_PCT,
            value=float(DEFAULT_REBALANCE_BAND_PCT),
            step=_BAND_STEP_PCT,
            key="rebalance_band",
            help="A name trades only once |current weight - target weight| "
            "exceeds this many percentage points. At 0 every name trades on "
            "every look, which is the calendar rebalance a band exists to "
            "avoid.",
        )
    with middle:
        cash = st.number_input(
            f"Cash to invest ({ccy})",
            min_value=0.0,
            value=0.0,
            step=_CASH_STEP,
            format="%.2f",
            key="rebalance_cash",
            help="New money to deploy. It is added to the portfolio value before "
            "targets are sized, and it is put to work even in names that are "
            "already inside their band.",
        )
    with right:
        fractional = st.toggle(
            "Fractional shares",
            value=False,
            key="rebalance_fractional",
            help="Off: every order is truncated toward zero, so it stops short of "
            "its target and never overshoots. On: sizes are exact, which "
            "most Indian brokers cannot execute on equities.",
        )

    return float(band), float(cash), bool(fractional)


# ---------------------------------------------------------------------------
# 2. drift versus policy
# ---------------------------------------------------------------------------
def _drift_caption(drift: pd.DataFrame, band: float, cfg: Any) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    n = len(drift)
    out_of_band = int((~drift["in_band"].astype(bool)).sum())

    if out_of_band == 0:
        st.caption(
            f"All {n} names sit inside the +/-{fmt_num(band, 1)} pp band. On a "
            f"banded policy that is a finished decision, not an absence of one: "
            f"the drift that exists is not worth the spread and the tax to remove."
        )
        return

    worst = drift.iloc[0]
    direction = "over" if _finite(worst["drift_pct"]) and worst["drift_pct"] > 0 else "under"
    side = "trimming" if direction == "over" else "topping up"
    st.caption(
        f"{out_of_band} of {n} names sit outside the +/-{fmt_num(band, 1)} pp band. "
        f"{worst['ticker']} is furthest from policy at {fmt_pct(worst['current_weight_pct'])} "
        f"against a {fmt_pct(worst['target_weight_pct'])} target -- "
        f"{fmt_num(abs(worst['drift_pct']), 2)} pp {direction}weight, or about "
        f"{fmt_money(abs(worst['value_gap']), ccy, 0)} of {side} to close it. The "
        f"solid bar is where the money is; the outlined bar behind it is where the "
        f"policy says it should be."
    )


def _drift_table(drift: pd.DataFrame, cfg: Any) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    money = _money_format(cfg)
    frame = drift.rename(
        columns={
            "ticker": "Ticker",
            "current_weight_pct": "Current",
            "target_weight_pct": "Target",
            "drift_pct": "Drift (pp)",
            "current_value": f"Market value ({ccy})",
            "target_value": f"Target value ({ccy})",
            "value_gap": f"Gap ({ccy})",
            "in_band": "In band",
        }
    )
    ui.dataframe(
        frame,
        height=min(38 * (len(frame) + 1) + 10, 520),
        column_config={
            "Ticker": st.column_config.TextColumn(width="small", pinned=True),
            "Current": st.column_config.NumberColumn(
                format="%.2f%%", help="Share of today's market value."
            ),
            "Target": st.column_config.NumberColumn(
                format="%.2f%%", help="Policy weight, renormalised to sum to 100%."
            ),
            "Drift (pp)": st.column_config.NumberColumn(
                format="%+.2f",
                help="Current minus target, in percentage POINTS. Positive is overweight.",
            ),
            f"Market value ({ccy})": st.column_config.NumberColumn(format=money),
            f"Target value ({ccy})": st.column_config.NumberColumn(format=money),
            f"Gap ({ccy})": st.column_config.NumberColumn(
                format=money, help="Target value less market value: + buy, - sell."
            ),
            "In band": st.column_config.CheckboxColumn(
                help="|drift| <= band. An undefined drift is reported as out of "
                "band, because unknown is not the same as fine.",
            ),
        },
    )


def _drift_block(drift: pd.DataFrame, band: float, cfg: Any) -> None:
    ui_theme.section("DRIFT VERSUS POLICY", "\U0001f4d0")

    if drift is None or drift.empty:
        ui.empty_state(
            "Nothing to compare",
            "The target vector and the priced positions produced no overlapping "
            "rows, so there is no drift to measure.",
            icon="\U0001f4ad",
        )
        return

    attrs = dict(drift.attrs)
    if attrs.get("targets_renormalised"):
        st.caption(
            f"Targets as supplied summed to {fmt_pct(attrs.get('target_sum_pct'))} "
            f"and were rescaled to 100%. Their relative shape is untouched."
        )
    dropped = list(attrs.get("targets_dropped") or [])
    if dropped:
        st.caption(
            "Ignored as unusable target weights (negative or non-numeric): "
            + ", ".join(str(t) for t in dropped)
            + "."
        )

    ui.chart(ch.weight_vs_target_bar(drift, band_pct=band, height=max(340, 34 * len(drift) + 120)))
    _drift_caption(drift, band, cfg)
    _drift_table(drift, cfg)

    ccy = getattr(cfg, "base_currency", "INR")
    ui.methodology(
        "How drift is measured, and why it is in points",
        f"""
With `V` the current market value of the whole book:

```
current_weight_pct = current_value / V * 100
drift_pct          = current_weight_pct - target_weight_pct
target_value       = target_weight_pct / 100 * V
value_gap          = target_value - current_value
in_band            = |drift_pct| <= {fmt_num(band, 1)}
```

**Drift is a difference of two percentages, so its unit is percentage POINTS,
not percent.** A name sitting at 18% against a 12% target has drifted
+6 pp -- it is not "50% overweight", even though it is half again as large as
policy. The distinction matters the moment you compare a 2 pp deviation in a
40% core holding against the same 2 pp in a 3% satellite: the first is a rounding
error, the second is most of the position.

**The row set is the union of what you hold and what you target.** A held name
absent from the target vector is shown against a 0% target, because a target
allocation describes the entire portfolio rather than a shortlist. A target on a
name you do not own yet appears with a 0% current weight and a positive gap.

**This table is sized against `V` alone -- new cash is deliberately excluded
here.** Cash enters on the order ticket below, where the target book value
becomes `V + cash`. That is why a target value in this table can be smaller than
the one the trade list works toward: the worksheet answers "how far from policy
am I today", the ticket answers "what do I buy with everything I am about to
have".

**Weights are market-value weights on the latest prices in this run**, expressed
in {ccy}. They drift with the market between refreshes, so a drift table computed
before an open is a different table from the one computed after it.
        """,
    )


# ---------------------------------------------------------------------------
# 3. the order ticket
# ---------------------------------------------------------------------------
def _ticket_kpis(summary: dict, trades: pd.DataFrame, cfg: Any, band: float) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    actions = (
        trades["action"].astype(str) if "action" in trades.columns else pd.Series(dtype=object)
    )
    n_buys = int((actions == rb.BUY).sum())
    n_sells = int((actions == rb.SELL).sum())

    buy_value = summary.get("buy_value")
    sell_value = summary.get("sell_value")
    net_cash = summary.get("net_cash")
    turnover = summary.get("turnover_pct")

    cards: list[dict[str, Any]] = [
        {
            "label": "Total to buy",
            "value": fmt_money(buy_value, ccy, 0),
            "tone": "neutral",
            "footnote": f"{n_buys} buy order(s)",
            "help_text": "Sum of the buy legs, in base currency, before any cost.",
        },
        {
            "label": "Total to sell",
            "value": fmt_money(sell_value, ccy, 0),
            "tone": "neutral",
            "footnote": f"{n_sells} sell order(s)",
            "help_text": "Sum of the sell legs. Every one of these is a taxable "
            "event if it is in profit.",
        },
        {
            "label": "Net cash",
            "value": fmt_money(net_cash, ccy, 0, signed=True),
            "tone": delta_tone(net_cash),
            "footnote": (
                "released by the list"
                if _finite(net_cash) and float(net_cash) > 0
                else "needed to fund the list"
                if _finite(net_cash) and float(net_cash) < 0
                else "self-funding"
            ),
            "help_text": "Sells less buys. Whole-share rounding means a "
            "self-funding rebalance rarely nets to exactly zero.",
        },
        {
            "label": "Turnover",
            "value": fmt_pct(turnover),
            "tone": (
                "warn" if _finite(turnover) and float(turnover) >= _HIGH_TURNOVER_PCT else "neutral"
            ),
            "footnote": f"two-way notional over a {fmt_money(trades.attrs.get('total_value'), ccy, 0)} book",
            "help_text": "(buys + sells) / pre-trade value. Swapping the entire "
            "portfolio scores 200%, not 100%.",
        },
    ]
    ui.kpi_row(cards, columns=4)

    stats: list[tuple[str, str]] = [
        ("Orders", f"{int(summary.get('n_trades', 0))}"),
        ("Inside band", f"{int(summary.get('in_band_count', 0))} of {len(trades)}"),
        (
            "Largest drift",
            f"{fmt_num(summary.get('max_drift_pct'), 2)} pp"
            if _finite(summary.get("max_drift_pct"))
            else DASH,
        ),
        ("Band in force", f"{fmt_num(band, 1)} pp"),
        ("Cash added", fmt_money(trades.attrs.get("cash_to_invest"), ccy, 0)),
        ("Target book value", fmt_money(trades.attrs.get("target_value_total"), ccy, 0)),
    ]
    ui.stat_grid(stats, columns=6)


def _trade_display(
    actionable: pd.DataFrame, positions: pd.DataFrame, cfg: Any
) -> tuple[pd.DataFrame, str]:
    """Formatted order ticket plus the name of its base-currency value column."""
    ccy = getattr(cfg, "base_currency", "INR")
    value_col = f"Value ({ccy})"

    native: dict[str, str] = {}
    if "currency" in positions.columns and "ticker" in positions.columns:
        for raw, quote_ccy in zip(positions["ticker"], positions["currency"]):
            if isinstance(quote_ccy, str) and quote_ccy.strip():
                native[str(raw).strip().upper()] = quote_ccy.strip()

    rows: list[dict[str, str]] = []
    for row in actionable.itertuples(index=False):
        ticker = str(row.ticker)
        quote_ccy = native.get(ticker.upper(), ccy)
        rows.append(
            {
                "Side": str(row.action),
                "Ticker": ticker,
                "Shares": fmt_shares(abs(float(row.shares))) if _finite(row.shares) else DASH,
                "Price": fmt_money(row.price, quote_ccy, 2),
                value_col: fmt_money(abs(float(row.trade_value)), ccy, 0)
                if _finite(row.trade_value)
                else DASH,
                "Current": fmt_pct(row.current_weight_pct),
                "Target": fmt_pct(row.target_weight_pct),
                "Drift (pp)": fmt_signed(row.drift_pct, 2),
            }
        )
    return pd.DataFrame(rows), value_col


def _render_ticket_table(frame: pd.DataFrame, value_col: str) -> None:
    """Colour-coded order list.

    Rendered through ``st.dataframe`` directly rather than ``ui.dataframe``
    because a pandas ``Styler`` -- which is what carries the BUY/SELL colour --
    has no ``len()`` for the component's own empty check.
    """
    tokens = ui_theme.tokens()
    palette = {
        rb.BUY: f"color:{tokens['gain']};font-weight:600",
        rb.SELL: f"color:{tokens['loss']};font-weight:600",
    }

    def _row_style(row: pd.Series) -> list[str]:
        style = palette.get(str(row.get("Side", "")), f"color:{tokens['text_muted']}")
        return [style if col in {"Side", value_col} else "" for col in row.index]

    try:
        rendered: Any = frame.style.apply(_row_style, axis=1)
    except Exception:  # noqa: BLE001 - colour is decoration, the list is the point
        rendered = frame

    st.dataframe(
        rendered,
        hide_index=True,
        width="stretch",
        height=min(38 * (len(frame) + 1) + 10, 520),
    )


def _download_name(result: Any) -> str:
    stamp = getattr(result, "generated_at", None)
    try:
        suffix = stamp.strftime("%Y%m%d-%H%M")
    except AttributeError:
        suffix = "latest"
    return f"quantfolio-trades-{suffix}.csv"


def _ticket_notes(trades: pd.DataFrame, cash: float, cfg: Any) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    no_price = list(trades.attrs.get("no_price") or [])
    below_lot = list(trades.attrs.get("below_lot_size") or [])

    if no_price:
        st.warning(
            "No price is available for "
            + ", ".join(str(t) for t in no_price)
            + ". Those are target names you do not hold yet, so the list can say "
            "how much money to deploy but not how many shares that buys. Size them "
            "at the broker's live quote."
        )
    if below_lot:
        st.caption(
            "Breached the band but rounded down to zero shares: "
            + ", ".join(str(t) for t in below_lot)
            + ". One share is worth more than the gap, so the drift is real and "
            "stays. It is reported rather than papered over -- turning on "
            "fractional shares is the only way to close it."
        )
    if cash > 0:
        st.caption(
            f"{fmt_money(cash, ccy, 0)} of new cash is included in the target book "
            f"value, and is deployed into names below target even where they are "
            f"already inside the band. Fresh money should be put to work; that is "
            f"the point of supplying it."
        )


def _nothing_to_do(band: float, cash: float, trades: pd.DataFrame, cfg: Any) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    below_lot = list(trades.attrs.get("below_lot_size") or [])

    if below_lot:
        message = (
            "Every breached name rounds down to zero whole shares at current "
            "prices (" + ", ".join(str(t) for t in below_lot) + "), so there is "
            "no executable order. Turn on fractional shares, or wait for the drift "
            "to grow past one share."
        )
    elif cash > 0:
        message = (
            f"The {fmt_money(cash, ccy, 0)} of new cash does not buy a whole share "
            f"in any underweight name at current prices."
        )
    else:
        message = (
            f"No name has drifted more than {fmt_num(band, 1)} percentage points "
            f"from its target, so the band is doing exactly what it exists to do: "
            f"stopping you from paying spread, brokerage and tax to correct noise."
        )

    ui.empty_state("No trades required", message, icon="\U0001f4a4")


def _ticket_block(
    trades: pd.DataFrame,
    summary: dict,
    positions: pd.DataFrame,
    result: Any,
    cfg: Any,
    band: float,
    cash: float,
) -> None:
    ui_theme.section("ORDER TICKET", "\U0001f9fe")

    if trades is None or trades.empty:
        ui.empty_state(
            "No order list",
            "The target vector produced no sizable rows against these positions.",
            icon="\U0001f4ad",
        )
        return

    _ticket_kpis(summary, trades, cfg, band)

    actions = (
        trades["action"].astype(str) if "action" in trades.columns else pd.Series(dtype=object)
    )
    actionable = trades.loc[actions != rb.HOLD].reset_index(drop=True)

    if actionable.empty:
        _nothing_to_do(band, cash, trades, cfg)
        _ticket_notes(trades, cash, cfg)
        return

    frame, value_col = _trade_display(actionable, positions, cfg)
    _render_ticket_table(frame, value_col)

    ccy = getattr(cfg, "base_currency", "INR")
    st.caption(
        f"Rows are in execution order -- sells first, because they fund the buys, "
        f"then largest ticket to smallest. Sizes are shown as absolute quantities "
        f"with the side in the first column; the CSV keeps the signed convention "
        f"instead (positive shares and value = buy), so `current value + "
        f"trade_value` is the post-trade value of each line. **Price is the "
        f"instrument's own quote currency, value is {ccy}** -- for a single-currency "
        f"book they differ by nothing, for a cross-currency one they differ by the "
        f"FX rate on the Holdings blotter."
    )

    st.download_button(
        "Download trade list (CSV)",
        data=actionable.to_csv(index=False).encode("utf-8"),
        file_name=_download_name(result),
        mime="text/csv",
        key="rebalance_download_trades",
        help="Signed shares and values, plus the current, target and drift "
        "weights behind each order. Prices are as of this analysis run, not "
        "of the moment you place the order.",
    )

    _ticket_notes(trades, cash, cfg)


# ---------------------------------------------------------------------------
# 4. bands, cost and tax
# ---------------------------------------------------------------------------
def _band_notes(band: float, summary: dict, cfg: Any) -> None:
    ui_theme.section("BANDS, COST AND TAX", "⚖️")

    turnover = summary.get("turnover_pct")
    turnover_line = (
        f"This list turns over {fmt_pct(turnover)} of the book two-way. "
        if _finite(turnover) and float(turnover) > 0
        else ""
    )

    st.markdown(
        f"""
**Why a band and not a calendar.** A quarterly or annual rebalance trades
because the date said so. On the day it fires, most names are a point or two
from policy and the orders it generates buy nothing except transaction costs and
a tax bill. A banded policy inverts the trigger: it looks continuously and
trades only the names whose drift has actually breached the tolerance -- here
+/-{fmt_num(band, 1)} percentage points -- and leaves everything else to mean-revert
on its own. The published work on tolerance-band rebalancing is consistent on
the result: the risk control is essentially as good as calendar rebalancing
while the number of trades, and therefore the cost drag, falls sharply, because
most calendar trades were never needed. {turnover_line}The band width is the
real decision. Tight bands hold the portfolio close to policy and pay for that
in turnover; wide bands accept more deviation from your own policy and trade
far less. At 0 pp this page degenerates into precisely the calendar rebalance
the band exists to avoid.

**What this trade list does not know.** It ignores brokerage, exchange and
regulatory fees, STT, stamp duty and GST; it ignores the bid-ask spread and
market impact, which on a mid- or small-cap order sized as a share of daily
volume routinely costs more than the commission does; it ignores lot and tick
sizes, which is safe for Indian cash equities but not for futures, several ETFs
or anything in the bond market; and it ignores settlement timing, margin and any
holding you cannot actually sell. Every price here is from this analysis run, so
re-run before you send anything.

**Tax is the reason to hesitate before a sell.** In India, listed equity sold
within 12 months of purchase is a short-term capital gain and is taxed
materially harder than the long-term treatment that applies after 12 months.
Trimming a few percentage points of overweight can therefore convert an entire
holding period from long-term to short-term for the lots that get matched, and
FIFO decides which lots those are -- your oldest, cheapest, most-appreciated
shares go first. This page has no view of your lots. **Trades & lots** does,
once a trade ledger is supplied, and that is where the question belongs.

Nothing here is investment, tax or execution advice. It is arithmetic on the
numbers you supplied.
        """
    )

    ui.methodology(
        "Sizing arithmetic, the whole-share rule and the turnover definition",
        f"""
With `V` the current market value and `C` the cash to invest, the target book
value is `V_t = V + C` and, per name:

```
target_value = target_weight_pct / 100 * V_t
gap          = target_value - current_value
shares       = gap / (price * fx_rate)
```

**A name trades when `|drift| > {fmt_num(band, 1)} pp`, or when `C > 0` and it sits
below target.** New money is deployed even into names already inside their band,
because refusing to invest fresh cash in an on-policy portfolio would be absurd.
Everything else is returned as HOLD with zero shares and zero value.

**Whole-share truncation is toward zero, always.** An order therefore stops
*short* of its target and can never overshoot it, which is what guarantees that
applying the entire list leaves every name inside the band (measured against
`V_t`) except where a single share is worth more than the band itself. Those
names are reported explicitly rather than hidden. Sells are additionally capped
at the quantity held: this page never produces a short.

**Signs and currencies.** `shares` and `trade_value` are signed -- positive is a
buy -- so `current_value + trade_value` is the post-trade value of the line.
`price` is the instrument's native quote price, `trade_value` is
{getattr(cfg, "base_currency", "the base currency")}; the two differ by the FX
rate and coincide for a single-currency book.

**Turnover is two-way**: `(buys + sells) / V * 100`. Selling everything and
buying a completely different portfolio scores 200%, not 100%. Half of a
two-way turnover figure is the conventional one-way number if you need to
compare against a fund fact sheet.

**Net cash** is `sells - buys`. It is rarely exactly zero on a self-funding
rebalance, because every order was rounded down to a whole share.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Rebalance",
        "How far the book has drifted from policy, and the exact orders that fix it",
        icon="⚖️",
    )

    result = guards.require_result()
    if result is None:
        return

    positions = result.positions
    if positions is None or len(positions) == 0 or "ticker" not in positions.columns:
        ui.empty_state(
            "No priced positions to rebalance",
            "The analysis ran but no holding produced a usable quote, so there are "
            "no weights to compare against a target and nothing to size an order "
            "from.",
            icon="\U0001f4ed",
            hint="Check the ticker symbols on the Portfolio page -- Indian equities "
            "need the .NS or .BO suffix.",
        )
        ui.warning_list(result.warnings, "Data notes from this run")
        return

    cfg = result.config
    ccy = getattr(cfg, "base_currency", "INR")

    ui.freshness_bar(
        result.generated_at,
        DATA_SOURCE,
        extra=[
            f"{len(positions)} priced positions",
            f"Sized in {ccy}",
            "Order sizes are struck against these prices",
        ],
    )

    ui_theme.section("POLICY TARGETS", "\U0001f3af")
    targets, source = _target_policy(positions, cfg)
    band, cash, fractional = _execution_controls(cfg)

    ui.badges(
        [
            (f"Source: {source}", "accent"),
            (f"Band +/-{fmt_num(band, 1)} pp", "neutral"),
            (
                f"Cash {fmt_money(cash, ccy, 0)}",
                "gain" if cash > 0 else "muted",
            ),
            ("Fractional shares" if fractional else "Whole shares only", "neutral"),
        ]
    )

    if not targets:
        st.caption(
            "No usable target allocation, so no drift and no orders. An empty "
            "target vector is treated as 'no instruction' -- never as an "
            "instruction to liquidate."
        )
        return

    drift = rb.drift_table(positions, targets, band_pct=band)
    trades = rb.rebalance_trades(
        positions,
        targets,
        band_pct=band,
        cash_to_invest=cash,
        allow_fractional=fractional,
    )
    summary = rb.rebalance_summary(trades)

    ui_theme.spacer(0.4)
    _drift_block(drift, band, cfg)

    ui_theme.spacer(0.4)
    _ticket_block(trades, summary, positions, result, cfg, band, cash)

    ui_theme.spacer(0.4)
    _band_notes(band, summary, cfg)
