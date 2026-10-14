"""FIFO tax-lot accounting: open lots, realised trades and realised P&L.

Portfolio accounting lives or dies on two details that toy trackers get wrong,
so both are stated up front:

1. Buy fees CAPITALISE into the cost basis. A lot opened by ``BUY q @ p`` with
   ``f`` of brokerage carries ``cost_per_share = p + f / q``, so the fee has to
   be earned back before the trade counts as a win.
2. Sell fees are APPORTIONED PRO-RATA across the lot slices the sale consumed.
   A sale that eats three FIFO slices produces three ``RealisedTrade`` rows
   whose ``fees`` sum back to the fee actually paid, split by matched quantity.
   Anything else either double-counts the fee or loses it.

Matching convention
-------------------
FIFO (first in, first out) per ticker -- the default cost-basis method for
listed equity in India and the usual fallback elsewhere. Specific-lot and
average-cost matching are deliberately NOT offered: silently applying the wrong
method is worse than not offering the choice at all.

Realised P&L for one matched slice
----------------------------------
    cost_basis   = q * (buy_price + buy_fees / buy_quantity)
    proceeds     = q * sell_price - apportioned_sell_fees
    realised_pnl = proceeds - cost_basis
    realised_pct = realised_pnl / cost_basis * 100          (guarded: basis > 0)

Selling more than is held never raises and never creates a negative lot: the
available quantity is matched, the excess is dropped, and a problem message is
returned for the UI to surface. The long-term / short-term split comes from
``models.RealisedTrade.is_long_term`` (India: strictly more than 365 calendar
days), not from a threshold duplicated in this module.

Pure pandas/numpy. No network, no Streamlit, no input mutated.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.models import Lot, RealisedTrade, Transaction

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
#: Canonical transaction schema (after column normalisation).
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "action",
    "quantity",
    "price",
    "fees",
)

#: Columns a transaction table must actually supply; ``fees`` defaults to 0.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "action",
    "quantity",
    "price",
)

#: Understood but optional. ``fees`` -> 0.0, ``currency`` -> the model default.
OPTIONAL_COLUMNS: tuple[str, ...] = ("fees", "currency", "note")

#: Tolerated spellings of the canonical column names (broker CSV exports vary).
_COLUMN_ALIASES: dict[str, str] = {
    "symbol": "ticker",
    "scrip": "ticker",
    "instrument": "ticker",
    "date": "trade_date",
    "trade_dt": "trade_date",
    "transaction_date": "trade_date",
    "tradedate": "trade_date",
    "type": "action",
    "side": "action",
    "transaction_type": "action",
    "trade_type": "action",
    "qty": "quantity",
    "shares": "quantity",
    "units": "quantity",
    "rate": "price",
    "trade_price": "price",
    "unit_price": "price",
    "fee": "fees",
    "charges": "fees",
    "brokerage": "fees",
    "commission": "fees",
    "notes": "note",
    "remark": "note",
    "remarks": "note",
    "ccy": "currency",
}

#: action spellings -> canonical TradeAction.
_ACTION_ALIASES: dict[str, str] = {
    "B": "BUY",
    "BUY": "BUY",
    "BOUGHT": "BUY",
    "PURCHASE": "BUY",
    "PURCHASED": "BUY",
    "S": "SELL",
    "SELL": "SELL",
    "SALE": "SELL",
    "SOLD": "SELL",
}

#: BUY settles before SELL when both land on the same calendar date, otherwise
#: a same-day round trip would look like an oversell.
_ACTION_ORDER: dict[str, int] = {"BUY": 0, "SELL": 1}

#: Share counts below this are treated as zero (float accumulation guard).
_QTY_TOLERANCE: float = 1e-9

_LOT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "open_date",
    "quantity",
    "cost_per_share",
    "cost_basis",
)

_REALISED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "open_date",
    "close_date",
    "quantity",
    "cost_per_share",
    "sale_per_share",
    "fees",
    "cost_basis",
    "proceeds",
    "realised_pnl",
    "realised_pct",
    "holding_days",
    "term",
)

_HOLDING_COLUMNS: tuple[str, ...] = ("ticker", "quantity", "buy_price")

_SUMMARY_KEYS: tuple[str, ...] = (
    "n_trades",
    "total_realised",
    "total_cost_basis",
    "win_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "best",
    "worst",
    "short_term_pnl",
    "long_term_pnl",
    "avg_holding_days",
)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower/snake-case the columns and resolve known broker aliases."""
    out = frame.copy()
    renamed: list[str] = []
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed.append(_COLUMN_ALIASES.get(key, key))
    out.columns = renamed
    # A duplicate after aliasing (e.g. both "date" and "trade_date") keeps the
    # first occurrence rather than raising later on assignment.
    return out.loc[:, ~pd.Index(renamed).duplicated()]


def _to_date(value: object) -> date | None:
    """Coerce anything date-like to a plain ``datetime.date`` (None on failure).

    Parsing follows pandas' default, which is MONTH-first for ambiguous
    ``05/02/2024`` style strings. ISO-8601 (``2024-02-05``) is the recommended
    input format; pandas emits its own dayfirst warning when it has to guess.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    stamp = pd.to_datetime(value, errors="coerce")
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    try:
        return stamp.to_pydatetime().date()
    except AttributeError:  # pragma: no cover - defensive
        return None


def _fmt_quantity(value: float) -> str:
    """Compact share count for problem messages: 15 not 15.000000."""
    if not np.isfinite(value):
        return "?"
    if abs(value - round(value)) < _QTY_TOLERANCE:
        return f"{int(round(value)):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _sort_key(txn: Transaction) -> tuple[date, int, str]:
    """Chronological, BUY before SELL on the same date, then ticker."""
    when = _to_date(txn.trade_date) or date.min
    # .strip() matters: build_lots normalises the action the same way, and a
    # whitespace-padded " BUY " that misses _ACTION_ORDER here would sort AFTER
    # a same-date SELL, fabricating an oversell out of a clean round trip.
    return (when, _ACTION_ORDER.get(str(txn.action).strip().upper(), 2), str(txn.ticker))


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean that returns NaN (not a warning) for an empty input."""
    return float(np.mean(values)) if len(values) else _NAN


@dataclass
class _OpenLot:
    """Mutable FIFO book entry. Never exposed -- ``Lot`` is the public form."""

    ticker: str
    open_date: date
    quantity: float
    cost_per_share: float


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_transactions(
    raw: pd.DataFrame | None,
) -> tuple[list[Transaction], list[str]]:
    """Validate a raw transaction table into ``Transaction`` objects.

    Rules applied, each rejected row reported once in ``problems``:
      * columns are snake-cased and common broker aliases resolved
        ("Trade Date" -> ``trade_date``, "Qty" -> ``quantity``)
      * ``ticker`` stripped and upper-cased; blank / "NAN" / "NONE" dropped
      * ``action`` upper-cased and mapped: B/BUY/BOUGHT/PURCHASE -> BUY,
        S/SELL/SOLD/SALE -> SELL; anything else is rejected
      * ``trade_date`` via ``pd.to_datetime(..., errors="coerce")`` then
        ``.date()``; NaT is rejected
      * ``quantity`` and ``price`` must parse and be strictly positive
      * ``fees`` is optional; missing, unparseable or negative fees become 0.0
        (a fee is a cost, never a credit)

    The result is sorted by ``(trade_date, action)`` with BUY ahead of SELL on
    the same date, which is the order :func:`build_lots` needs so that a
    same-day round trip is not mistaken for an oversell.

    Returns ``(transactions, problems)`` -- never raises on bad input.
    """
    problems: list[str] = []
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return [], ["No transactions supplied."]

    df = _normalise_columns(raw).reset_index(drop=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [], [f"Missing column(s): {', '.join(missing)}."]

    tickers = df["ticker"].astype(str).str.strip().str.upper()
    actions = df["action"].astype(str).str.strip().str.upper()
    quantities = pd.to_numeric(df["quantity"], errors="coerce")
    prices = pd.to_numeric(df["price"], errors="coerce")

    if "fees" in df.columns:
        fees = pd.to_numeric(df["fees"], errors="coerce")
    else:
        fees = pd.Series(0.0, index=df.index)

    if "currency" in df.columns:
        currencies = df["currency"].astype(str).str.strip().str.upper()
    else:
        currencies = pd.Series("", index=df.index)

    notes = df["note"].astype(str) if "note" in df.columns else pd.Series("", index=df.index)

    transactions: list[Transaction] = []
    for idx in df.index:
        ticker = tickers.at[idx]
        if ticker in ("", "NAN", "NONE"):
            continue  # blank grid row, not an error worth reporting

        label = f"{ticker} (row {int(idx) + 1})"
        action = _ACTION_ALIASES.get(actions.at[idx])
        if action is None:
            problems.append(f"{label}: unknown action '{actions.at[idx]}' - row skipped.")
            continue

        trade_date = _to_date(df.at[idx, "trade_date"])
        if trade_date is None:
            problems.append(f"{label}: unparseable trade date - row skipped.")
            continue

        quantity = float(quantities.at[idx]) if pd.notna(quantities.at[idx]) else _NAN
        price = float(prices.at[idx]) if pd.notna(prices.at[idx]) else _NAN
        if not np.isfinite(quantity) or quantity <= 0 or not np.isfinite(price) or price <= 0:
            problems.append(f"{label}: quantity and price must be positive numbers - row skipped.")
            continue

        fee = float(fees.at[idx]) if pd.notna(fees.at[idx]) else 0.0
        if not np.isfinite(fee) or fee < 0:
            fee = 0.0

        currency = currencies.at[idx]
        note = notes.at[idx]
        kwargs = {} if currency in ("", "NAN", "NONE") else {"currency": currency}

        transactions.append(
            Transaction(
                ticker=ticker,
                trade_date=trade_date,
                action=action,  # type: ignore[arg-type]
                quantity=quantity,
                price=price,
                fees=fee,
                note="" if note in ("nan", "NaN", "None") else note,
                **kwargs,
            )
        )

    if not transactions:
        problems.append("No valid transactions after validation.")

    transactions.sort(key=_sort_key)
    return transactions, problems


# ---------------------------------------------------------------------------
# FIFO engine
# ---------------------------------------------------------------------------
def build_lots(
    transactions: Sequence[Transaction],
) -> tuple[list[Lot], list[RealisedTrade], list[str]]:
    """Run FIFO matching and return ``(open_lots, realised_trades, problems)``.

    Per ticker:
      * a BUY appends an open lot with ``cost_per_share = price + fees /
        quantity`` -- the brokerage is capitalised into the basis;
      * a SELL consumes the OLDEST open lots first. Each consumed slice emits
        one :class:`~src.models.RealisedTrade` whose ``fees`` is the sale's fee
        apportioned pro-rata by matched quantity, so the slice fees sum exactly
        back to the fee paid (the last slice absorbs the rounding residual).

    Selling more than is held consumes what exists, emits
    ``"<TICKER>: sell of X exceeds holding of Y on <date> - excess ignored."``
    and continues. No exception is raised and no lot can go negative.

    The input sequence is sorted defensively (chronological, BUY before SELL on
    the same date) so the engine is correct even if handed unsorted data. The
    input is never mutated.
    """
    problems: list[str] = []
    if not transactions:
        return [], [], problems

    ordered = sorted(transactions, key=_sort_key)
    book: dict[str, deque[_OpenLot]] = {}
    realised: list[RealisedTrade] = []

    for txn in ordered:
        ticker = str(txn.ticker).strip().upper()
        action = str(txn.action).strip().upper()
        trade_date = _to_date(txn.trade_date)
        quantity = float(txn.quantity)
        price = float(txn.price)
        fees = float(txn.fees or 0.0)

        if trade_date is None:
            problems.append(f"{ticker}: transaction with no valid date - ignored.")
            continue
        if not np.isfinite(quantity) or quantity <= 0:
            problems.append(
                f"{ticker}: non-positive quantity on {trade_date} - transaction ignored."
            )
            continue
        if not np.isfinite(price) or price <= 0:
            problems.append(f"{ticker}: non-positive price on {trade_date} - transaction ignored.")
            continue
        if not np.isfinite(fees) or fees < 0:
            fees = 0.0
        if action not in _ACTION_ORDER:
            problems.append(f"{ticker}: unknown action '{action}' - transaction ignored.")
            continue

        lots = book.setdefault(ticker, deque())

        if action == "BUY":
            lots.append(
                _OpenLot(
                    ticker=ticker,
                    open_date=trade_date,
                    quantity=quantity,
                    cost_per_share=price + fees / quantity,
                )
            )
            continue

        # ---- SELL: FIFO consumption ----
        available = float(sum(lot.quantity for lot in lots))
        matched = min(quantity, available)
        if quantity - matched > _QTY_TOLERANCE:
            problems.append(
                f"{ticker}: sell of {_fmt_quantity(quantity)} exceeds holding of "
                f"{_fmt_quantity(max(available, 0.0))} on {trade_date} - excess ignored."
            )
        if matched <= _QTY_TOLERANCE:
            continue

        slices: list[tuple[_OpenLot, float]] = []
        remaining = matched
        while remaining > _QTY_TOLERANCE and lots:
            lot = lots[0]
            take = min(lot.quantity, remaining)
            slices.append((lot, take))
            lot.quantity -= take
            remaining -= take
            if lot.quantity <= _QTY_TOLERANCE:
                lots.popleft()

        allocated = 0.0
        last = len(slices) - 1
        for i, (lot, take) in enumerate(slices):
            if i == last:
                slice_fee = fees - allocated  # residual: fees sum exactly
            else:
                slice_fee = fees * (take / matched)
                allocated += slice_fee
            realised.append(
                RealisedTrade(
                    ticker=ticker,
                    open_date=lot.open_date,
                    close_date=trade_date,
                    quantity=take,
                    cost_per_share=lot.cost_per_share,
                    sale_per_share=price,
                    fees=max(slice_fee, 0.0),
                )
            )

    open_lots = [
        Lot(
            ticker=lot.ticker,
            open_date=lot.open_date,
            quantity=lot.quantity,
            cost_per_share=lot.cost_per_share,
        )
        for ticker in sorted(book)
        for lot in book[ticker]
        if lot.quantity > _QTY_TOLERANCE
    ]
    return open_lots, realised, problems


def holdings_from_transactions(transactions: Sequence[Transaction]) -> pd.DataFrame:
    """Current holdings implied by the OPEN lots after FIFO matching.

    ``quantity`` is the sum of the open lot quantities and ``buy_price`` the
    quantity-weighted average cost per share,
    ``sum(q_i * c_i) / sum(q_i)`` -- fee-inclusive, because the lots are.
    Tickers whose position is fully closed are excluded (they are history, not
    holdings). Columns match ``positions.REQUIRED_COLUMNS`` so the frame can be
    fed straight into ``clean_holdings``.

    Returns an empty frame with the right columns when nothing is open.
    """
    empty = pd.DataFrame(columns=list(_HOLDING_COLUMNS)).astype(
        {"ticker": "object", "quantity": "float64", "buy_price": "float64"}
    )
    lots, _realised, _problems = build_lots(transactions)
    if not lots:
        return empty

    frame = pd.DataFrame(
        {
            "ticker": [lot.ticker for lot in lots],
            "quantity": [float(lot.quantity) for lot in lots],
            "_cost": [float(lot.cost_basis) for lot in lots],
        }
    )
    grouped = frame.groupby("ticker", as_index=False).agg(
        quantity=("quantity", "sum"), _cost=("_cost", "sum")
    )
    grouped = grouped.loc[grouped["quantity"] > _QTY_TOLERANCE]
    if grouped.empty:
        return empty

    grouped["buy_price"] = grouped["_cost"] / grouped["quantity"]
    out = grouped[list(_HOLDING_COLUMNS)].sort_values("ticker")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def lots_dataframe(lots: Sequence[Lot]) -> pd.DataFrame:
    """Open lots as a table: ticker, open_date, quantity, cost_per_share, cost_basis.

    ``cost_basis = quantity * cost_per_share`` (fee-inclusive). Sorted by
    ticker then open date, which is also the FIFO consumption order. Returns an
    empty frame with those columns when there are no open lots.
    """
    if not lots:
        return pd.DataFrame(columns=list(_LOT_COLUMNS))

    frame = pd.DataFrame(
        {
            "ticker": [lot.ticker for lot in lots],
            "open_date": [lot.open_date for lot in lots],
            "quantity": [float(lot.quantity) for lot in lots],
            "cost_per_share": [float(lot.cost_per_share) for lot in lots],
            "cost_basis": [float(lot.cost_basis) for lot in lots],
        }
    )
    frame = frame.sort_values(["ticker", "open_date"], kind="stable")
    return frame[list(_LOT_COLUMNS)].reset_index(drop=True)


def realised_dataframe(trades: Sequence[RealisedTrade]) -> pd.DataFrame:
    """Closed (matched) trades as a table, one row per FIFO slice.

    Columns: ticker, open_date, close_date, quantity, cost_per_share,
    sale_per_share, fees, cost_basis, proceeds, realised_pnl, realised_pct,
    holding_days, term. ``proceeds`` is net of the apportioned sell fee,
    ``cost_basis`` includes the capitalised buy fee, and ``term`` is
    "Long" / "Short" per ``RealisedTrade.is_long_term`` (> 365 calendar days).
    Sorted by close date then ticker. Empty frame with those columns when there
    are no closed trades.
    """
    if not trades:
        return pd.DataFrame(columns=list(_REALISED_COLUMNS))

    frame = pd.DataFrame(
        {
            "ticker": [t.ticker for t in trades],
            "open_date": [t.open_date for t in trades],
            "close_date": [t.close_date for t in trades],
            "quantity": [float(t.quantity) for t in trades],
            "cost_per_share": [float(t.cost_per_share) for t in trades],
            "sale_per_share": [float(t.sale_per_share) for t in trades],
            "fees": [float(t.fees) for t in trades],
            "cost_basis": [float(t.cost_basis) for t in trades],
            "proceeds": [float(t.proceeds) for t in trades],
            "realised_pnl": [float(t.realised_pnl) for t in trades],
            "realised_pct": [float(t.realised_pct) for t in trades],
            "holding_days": [int(t.holding_days) for t in trades],
            "term": ["Long" if t.is_long_term else "Short" for t in trades],
        }
    )
    frame = frame.sort_values(["close_date", "ticker"], kind="stable")
    return frame[list(_REALISED_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# realised summary
# ---------------------------------------------------------------------------
def realised_summary(trades: Sequence[RealisedTrade]) -> dict[str, float | int]:
    """Aggregate statistics over closed trades.

    Keys and conventions
    --------------------
    n_trades          number of matched FIFO slices (int)
    total_realised    sum of realised P&L, net of all fees
    total_cost_basis  sum of fee-inclusive cost bases
    win_rate          PERCENT (0-100) of slices with realised_pnl > 0
    avg_win           mean P&L of winners       (NaN if none)
    avg_loss          mean P&L of losers, negative by construction (NaN if none)
    profit_factor     gross wins / |gross losses|; NaN when there are no
                      losses (an undefined ratio, not "infinitely good")
    best / worst      largest and smallest single-slice realised P&L
    short_term_pnl    sum over slices held <= 365 days
    long_term_pnl     sum over slices held  > 365 days
    avg_holding_days  unweighted mean holding period in calendar days

    A slice with exactly zero P&L counts as neither a win nor a loss, so
    ``win_rate`` is wins / all slices, not wins / (wins + losses).

    With no closed trades every float is NaN (nothing has been realised, and a
    fabricated 0.0 would read as a real result); ``n_trades`` is 0.
    """
    if not trades:
        out: dict[str, float | int] = dict.fromkeys(_SUMMARY_KEYS, _NAN)
        out["n_trades"] = 0
        return out

    pnl = np.array([float(t.realised_pnl) for t in trades], dtype=float)
    basis = np.array([float(t.cost_basis) for t in trades], dtype=float)
    days = np.array([int(t.holding_days) for t in trades], dtype=float)
    long_term = np.array([bool(t.is_long_term) for t in trades], dtype=bool)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = float(np.abs(losses.sum()))

    n = int(pnl.size)
    return {
        "n_trades": n,
        "total_realised": float(pnl.sum()),
        "total_cost_basis": float(basis.sum()),
        "win_rate": float(wins.size) / n * 100.0,
        "avg_win": _mean(wins),
        "avg_loss": _mean(losses),
        "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else _NAN,
        "best": float(pnl.max()),
        "worst": float(pnl.min()),
        "short_term_pnl": float(pnl[~long_term].sum()) if (~long_term).any() else _NAN,
        "long_term_pnl": float(pnl[long_term].sum()) if long_term.any() else _NAN,
        "avg_holding_days": _mean(days),
    }


# ---------------------------------------------------------------------------
# cash flows
# ---------------------------------------------------------------------------
def cash_flows(transactions: Sequence[Transaction]) -> list[tuple[date, float]]:
    """External cash flows implied by the trades, for XIRR / money-weighted return.

    One ``(trade_date, cash_flow)`` pair per transaction using
    ``Transaction.cash_flow``:  BUY -> ``-(quantity * price + fees)``,
    SELL -> ``+(quantity * price - fees)``. Sign convention is the investor's
    pocket: money out is negative.

    Sorted by date (BUY ahead of SELL on the same date). Transactions with an
    unparseable date or a non-finite flow are skipped rather than poisoning the
    solver. Returns an empty list for empty input.
    """
    if not transactions:
        return []

    flows: list[tuple[date, float]] = []
    for txn in sorted(transactions, key=_sort_key):
        when = _to_date(txn.trade_date)
        if when is None:
            continue
        try:
            amount = float(txn.cash_flow)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if not np.isfinite(amount):
            continue
        flows.append((when, amount))
    return flows
