"""Rebalancing: how far the book has drifted, and the exact trades that fix it.

A drift report is only useful if it ends in an order ticket. This module turns
a target allocation into (a) an auditable drift table and (b) a concrete,
executable trade list sized in whole shares.

Definitions
-----------
::

    V        = current invested value of the portfolio      (base currency)
    C        = cash_to_invest, >= 0                         (base currency)
    V_t      = V + C                       target portfolio value
    w_i      = target weight of name i, in PERCENT, renormalised to sum to 100
    v_i      = current value of name i                      (base currency)
    cw_i     = v_i / V * 100               current weight, PERCENT
    drift_i  = cw_i - w_i                  PERCENTAGE POINTS (not a ratio)
    t_i      = w_i / 100 * V_t             target value
    gap_i    = t_i - v_i                   money to buy (+) or to sell (-)

Trade rule
----------
A name trades when it has breached its tolerance band,

    ``|drift_i| > band_pct``

or when there is new cash to deploy and the name is short of its target
(``C > 0 and gap_i > 0``). Fresh cash is put to work even in a book that is
already inside its bands -- which is the whole point of supplying it -- while
a name that is merely drifting inside the band is left alone, because
harvesting a taxable gain to fix a 2% deviation is not risk management.

Share sizing
------------
::

    shares_i = gap_i / (price_i * fx_rate_i)

truncated TOWARD ZERO when ``allow_fractional`` is False, so a whole-share
trade always stops short of its target and never overshoots it. Sell size is
additionally capped at the quantity held: this module never shorts.

What the trade list guarantees
------------------------------
Measure post-trade weights against ``V_t`` (i.e. count any cash left over as
part of the portfolio). Then for every name:

* a name that traded lands on its target, up to a one-share rounding residue
  bounded by ``price_i * fx_rate_i / V_t``;
* a name that did NOT trade is inside its band already. With ``C = 0`` its
  weight is unchanged. With ``C > 0`` it must have had ``gap_i <= 0`` (it was
  at or above target), and diluting it by ``V / V_t`` only moves it toward
  target -- so its post-trade drift lies in ``[0, band_pct]``.

Hence after applying the whole list every weight is within ``band_pct`` of
its target, except where a single share is worth more than the band. That is
a real constraint of trading in whole units, and it is reported in
``trades.attrs["below_lot_size"]`` rather than papered over.

Conventions
-----------
* ``shares`` and ``trade_value`` are SIGNED: positive = BUY, negative = SELL,
  so ``v_i + trade_value_i`` is the post-trade value. ``action`` carries the
  human-readable side.
* ``price`` is the instrument's NATIVE quote price (the number shown in the
  positions table); ``trade_value`` is in the portfolio's BASE currency. The
  two differ by ``fx_rate`` and coincide for a single-currency book.
* Targets that do not sum to 100 are renormalised and the fact is recorded in
  ``.attrs["targets_renormalised"]`` / ``["target_sum_pct"]`` -- never
  silently.
* A held name absent from ``targets`` has a target of 0%: a target allocation
  describes the WHOLE portfolio. An empty or unusable ``targets`` mapping is
  a no-op, never an instruction to liquidate.
* A target on a name that is not held yet has no price in the positions
  frame, so ``shares`` is NaN while ``trade_value`` still says how much to
  deploy. Those tickers are listed in ``trades.attrs["no_price"]``.
* Nothing here mutates its input and nothing raises: degenerate input yields
  an empty frame or NaN.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_REBALANCE_BAND_PCT

__all__ = [
    "BUY",
    "DRIFT_COLUMNS",
    "HOLD",
    "SELL",
    "TRADE_COLUMNS",
    "drift_table",
    "equal_weight_targets",
    "rebalance_summary",
    "rebalance_trades",
    "targets_from_optimizer",
]

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

#: Exact output schema of :func:`drift_table`, in order.
DRIFT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "current_weight_pct",
    "target_weight_pct",
    "drift_pct",
    "current_value",
    "target_value",
    "value_gap",
    "in_band",
)

#: Exact output schema of :func:`rebalance_trades`, in order.
TRADE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "action",
    "shares",
    "price",
    "trade_value",
    "current_weight_pct",
    "target_weight_pct",
    "drift_pct",
)

#: Weights are expressed in percent throughout this module.
_FULL_ALLOCATION_PCT = 100.0

#: Tolerance below which a target vector is considered to already sum to 100.
_SUM_TOLERANCE_PCT = 1e-9

#: Optimiser output below this weight FRACTION is numerical noise, not a
#: position anybody would place, and is dropped by ``targets_from_optimizer``.
_MIN_OPTIMIZER_WEIGHT = 1e-6

#: Execution order of the trade list: raise the cash before you spend it.
_ACTION_RANK: dict[str, int] = {SELL: 0, BUY: 1, HOLD: 2}

_BLANK_TICKERS: frozenset[str] = frozenset({"", "NAN", "NONE", "NAT", "NULL"})


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _nan() -> float:
    return float("nan")


def _as_float(value: object) -> float:
    """Best-effort float conversion; anything unparseable becomes NaN."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _nan()
    return out if np.isfinite(out) else _nan()


def _clean_ticker(value: object) -> str | None:
    """Strip and upper-case a ticker; blanks and placeholders become ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    return None if text in _BLANK_TICKERS else text


def _non_negative(value: object, default: float = 0.0) -> float:
    """Coerce to a finite, non-negative float, falling back to ``default``."""
    out = _as_float(value)
    if not np.isfinite(out) or out < 0:
        return default
    return float(out)


def _pairs(mapping: Mapping[str, float] | pd.Series | Iterable | None) -> list[tuple]:
    """Flatten any accepted target container into ``(key, value)`` pairs."""
    if mapping is None:
        return []
    if isinstance(mapping, pd.Series):
        return list(mapping.items())
    if isinstance(mapping, pd.DataFrame):
        if mapping.shape[1] == 0:
            return []
        return list(mapping.iloc[:, 0].items())
    if isinstance(mapping, Mapping):
        return list(mapping.items())
    try:
        return [(key, value) for key, value in mapping]  # type: ignore[misc]
    except (TypeError, ValueError):
        return []


def _numeric_column(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    """A float Series for ``name``, or a constant ``default`` when absent."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _sum_min_count(values: pd.Series) -> float:
    """Sum that stays NaN when every input is NaN (unlike ``Series.sum()``)."""
    return float(values.sum(min_count=1))


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def equal_weight_targets(tickers: Iterable[str] | None) -> dict[str, float]:
    """Equal-weight target allocation in PERCENT: ``100 / n`` for each name.

    The naive 1/N portfolio is a genuinely hard benchmark to beat out of
    sample -- it has no estimation error because it estimates nothing -- so it
    is the natural default target here.

    Tickers are stripped and upper-cased, blanks and duplicates are removed,
    and the returned weights sum to 100 (up to floating-point). An empty or
    unusable input returns ``{}`` rather than raising.
    """
    seen: list[str] = []
    try:
        candidates = [] if tickers is None else list(tickers)
    except TypeError:
        return {}
    for raw in candidates:
        ticker = _clean_ticker(raw)
        if ticker is not None and ticker not in seen:
            seen.append(ticker)
    if not seen:
        return {}
    weight = _FULL_ALLOCATION_PCT / float(len(seen))
    return dict.fromkeys(seen, weight)


def targets_from_optimizer(
    weights: Mapping[str, float] | pd.Series | None,
) -> dict[str, float]:
    """Convert optimiser weight FRACTIONS into target PERCENTAGES.

    ``{"AAA": 0.6, "BBB": 0.4}`` becomes ``{"AAA": 60.0, "BBB": 40.0}``.

    Optimisers routinely emit long-only weights that sum to ``1 - 1e-12`` and
    dust weights of order ``1e-17``; both are numerical artefacts. Weights
    below ``_MIN_OPTIMIZER_WEIGHT`` (1e-6 of the book, i.e. 0.0001%) and any
    negative weight are dropped, and what remains is renormalised to sum to
    exactly 100. That also makes the function idempotent-ish for input that is
    already in percent, since renormalisation removes the scale.

    Returns ``{}`` for empty, non-finite or all-zero input.
    """
    cleaned: dict[str, float] = {}
    for key, value in _pairs(weights):
        ticker = _clean_ticker(key)
        if ticker is None:
            continue
        weight = _as_float(value)
        if not np.isfinite(weight) or weight < _MIN_OPTIMIZER_WEIGHT:
            continue
        cleaned[ticker] = cleaned.get(ticker, 0.0) + weight * _FULL_ALLOCATION_PCT

    total = float(sum(cleaned.values()))
    if not cleaned or not np.isfinite(total) or total <= 0:
        return {}
    scale = _FULL_ALLOCATION_PCT / total
    return {ticker: weight * scale for ticker, weight in cleaned.items()}


def _normalise_targets(
    targets: Mapping[str, float] | pd.Series | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Clean a target mapping and renormalise it to sum to 100 percent.

    Returns ``(targets_pct, meta)``. ``meta`` carries ``target_sum_pct`` (the
    sum as supplied), ``targets_renormalised`` (bool) and a human-readable
    ``note``. Duplicate tickers are summed, negative and non-finite weights are
    dropped, and a target of exactly 0 is KEPT because "hold none of this" is a
    meaningful instruction. A vector that sums to zero cannot be renormalised
    and yields ``{}`` -- which downstream is a no-op, never a liquidation.
    """
    meta: dict[str, Any] = {
        "target_sum_pct": _nan(),
        "targets_renormalised": False,
        "targets_dropped": [],
        "note": "",
    }

    cleaned: dict[str, float] = {}
    dropped: list[str] = []
    for key, value in _pairs(targets):
        ticker = _clean_ticker(key)
        if ticker is None:
            continue
        weight = _as_float(value)
        if not np.isfinite(weight) or weight < 0:
            dropped.append(ticker)
            continue
        cleaned[ticker] = cleaned.get(ticker, 0.0) + weight

    meta["targets_dropped"] = sorted(set(dropped))
    if not cleaned:
        meta["note"] = "No usable target weights supplied - nothing to rebalance."
        return {}, meta

    total = float(sum(cleaned.values()))
    meta["target_sum_pct"] = total
    if not np.isfinite(total) or total <= 0:
        meta["note"] = "Target weights sum to zero - nothing to rebalance."
        return {}, meta

    if abs(total - _FULL_ALLOCATION_PCT) > _SUM_TOLERANCE_PCT:
        scale = _FULL_ALLOCATION_PCT / total
        cleaned = {ticker: w * scale for ticker, w in cleaned.items()}
        meta["targets_renormalised"] = True
        meta["note"] = f"Target weights summed to {total:.4f}% and were renormalised to 100%."
    return cleaned, meta


# ---------------------------------------------------------------------------
# positions snapshot
# ---------------------------------------------------------------------------
_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "current_value",
    "quantity",
    "price",
    "fx_rate",
    "price_base",
)


def _empty_snapshot() -> pd.DataFrame:
    frame = pd.DataFrame(
        {name: pd.Series(dtype=float) for name in _SNAPSHOT_COLUMNS},
    )
    frame.index = pd.Index([], dtype=object, name="ticker")
    return frame


def _position_snapshot(positions: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse a positions frame to one row per ticker with what trading needs.

    Columns: ``current_value`` (base currency), ``quantity``, ``price``
    (native currency), ``fx_rate`` and ``price_base = price * fx_rate``, the
    base-currency price used to turn a money gap into a share count. When the
    native price or the FX rate is missing, ``price_base`` falls back to
    ``current_value / quantity``, which is base-currency by construction.

    Duplicate ticker rows are merged (values and quantities summed, price and
    rate averaged); rows without a finite ``current_value`` are unusable and
    are dropped. The input is never mutated.
    """
    if (
        positions is None
        or not isinstance(positions, pd.DataFrame)
        or positions.empty
        or "ticker" not in positions.columns
    ):
        return _empty_snapshot()

    # A fresh RangeIndex on a copy: the caller's frame is never touched, and a
    # duplicated index cannot scramble the column-wise assembly below.
    source = positions.reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "ticker": source["ticker"].map(_clean_ticker),
            "quantity": _numeric_column(source, "quantity", _nan()),
            "price": _numeric_column(source, "current_price", _nan()),
            "fx_rate": _numeric_column(source, "fx_rate", 1.0),
        }
    )
    if "current_value" in source.columns:
        frame["current_value"] = _numeric_column(source, "current_value", _nan())
    else:
        frame["current_value"] = frame["quantity"] * frame["price"] * frame["fx_rate"]

    frame = frame.dropna(subset=["ticker", "current_value"])
    if frame.empty:
        return _empty_snapshot()

    grouped = frame.groupby("ticker", sort=False).agg(
        current_value=("current_value", "sum"),
        quantity=("quantity", _sum_min_count),
        price=("price", "mean"),
        fx_rate=("fx_rate", "mean"),
    )

    price = grouped["price"].to_numpy(dtype=float)
    rate = grouped["fx_rate"].to_numpy(dtype=float)
    quantity = grouped["quantity"].to_numpy(dtype=float)
    value = grouped["current_value"].to_numpy(dtype=float)

    price_base = price * rate
    usable = np.isfinite(price_base) & (price_base > 0)
    fallback = np.full(price_base.shape, _nan())
    np.divide(
        value,
        quantity,
        out=fallback,
        where=np.isfinite(quantity) & (quantity > 0) & np.isfinite(value),
    )
    grouped["price_base"] = np.where(usable, price_base, fallback)
    return grouped[list(_SNAPSHOT_COLUMNS)]


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------
def _empty_drift() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype=object),
            "current_weight_pct": pd.Series(dtype=float),
            "target_weight_pct": pd.Series(dtype=float),
            "drift_pct": pd.Series(dtype=float),
            "current_value": pd.Series(dtype=float),
            "target_value": pd.Series(dtype=float),
            "value_gap": pd.Series(dtype=float),
            "in_band": pd.Series(dtype=bool),
            "quantity": pd.Series(dtype=float),
            "price": pd.Series(dtype=float),
            "fx_rate": pd.Series(dtype=float),
            "price_base": pd.Series(dtype=float),
        }
    )


def _working_frame(
    positions: pd.DataFrame | None,
    targets: Mapping[str, float] | pd.Series | None,
    band_pct: float,
    cash_to_invest: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the shared drift + execution frame used by both public entry points.

    The row set is the UNION of held tickers and target tickers: a held name
    with no target gets a target of 0% (sell it down), a target on a name that
    is not held gets a current weight of 0% (buy it). Current weights are
    recomputed from ``current_value`` rather than trusting an incoming
    ``weight_pct`` column, so a filtered positions frame still yields a
    self-consistent table.
    """
    band = _non_negative(band_pct, DEFAULT_REBALANCE_BAND_PCT)
    cash = _non_negative(cash_to_invest, 0.0)
    snapshot = _position_snapshot(positions)
    target_map, meta = _normalise_targets(targets)

    total_value = float(snapshot["current_value"].sum()) if len(snapshot) else 0.0
    if not np.isfinite(total_value):
        total_value = 0.0
    target_total = total_value + cash

    meta.update(
        {
            "band_pct": band,
            "cash_to_invest": cash,
            "total_value": total_value,
            "target_value_total": target_total,
            "n_targets": len(target_map),
            "n_positions": int(len(snapshot)),
        }
    )

    held = list(snapshot.index)
    extra = sorted(t for t in target_map if t not in snapshot.index)
    tickers = held + extra
    if not tickers:
        return _empty_drift(), meta

    reindexed = snapshot.reindex(tickers)
    current_value = reindexed["current_value"].to_numpy(dtype=float)
    current_value = np.where(np.isfinite(current_value), current_value, 0.0)

    if target_map:
        target_weight = np.array([float(target_map.get(t, 0.0)) for t in tickers], dtype=float)
    else:
        # No usable targets: every comparison is undefined, so the whole table
        # is NaN and downstream nothing trades.
        target_weight = np.full(len(tickers), _nan())

    if total_value > 0:
        current_weight = current_value / total_value * _FULL_ALLOCATION_PCT
    else:
        # Nothing invested yet: a zero position genuinely IS a zero weight.
        current_weight = np.where(current_value == 0.0, 0.0, _nan())

    drift = current_weight - target_weight
    target_value = target_weight / _FULL_ALLOCATION_PCT * target_total
    value_gap = target_value - current_value
    with np.errstate(invalid="ignore"):
        in_band = np.abs(drift) <= band

    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "current_weight_pct": current_weight,
            "target_weight_pct": target_weight,
            "drift_pct": drift,
            "current_value": current_value,
            "target_value": target_value,
            "value_gap": value_gap,
            "in_band": in_band,
            "quantity": reindexed["quantity"].to_numpy(dtype=float),
            "price": reindexed["price"].to_numpy(dtype=float),
            "fx_rate": reindexed["fx_rate"].to_numpy(dtype=float),
            "price_base": reindexed["price_base"].to_numpy(dtype=float),
        }
    )
    return frame.reset_index(drop=True), meta


def drift_table(
    positions: pd.DataFrame | None,
    targets: Mapping[str, float] | pd.Series | None,
    band_pct: float = DEFAULT_REBALANCE_BAND_PCT,
) -> pd.DataFrame:
    """Current versus target allocation, name by name.

    Columns::

        ticker, current_weight_pct, target_weight_pct, drift_pct,
        current_value, target_value, value_gap, in_band

    with

        current_weight_pct = current_value / total_value * 100
        drift_pct          = current_weight_pct - target_weight_pct   (points)
        target_value       = target_weight_pct / 100 * total_value
        value_gap          = target_value - current_value   (+ = buy, - = sell)
        in_band            = |drift_pct| <= band_pct

    ``drift_pct`` is a difference of percentages, i.e. PERCENTAGE POINTS: a
    name at 18% against a 12% target has drifted +6, not +50%.

    Row set: held names plus target names. A held name absent from ``targets``
    is shown with a 0% target (the target vector describes the entire
    portfolio); a target on a name that is not held is shown with a 0% current
    weight. Targets are renormalised to 100 when they do not sum to it, and
    ``.attrs`` records ``target_sum_pct``, ``targets_renormalised``,
    ``band_pct``, ``total_value``, ``cash_to_invest`` (always 0 here) and a
    human-readable ``note``.

    Rows are sorted by absolute drift descending, so the names that need
    attention are at the top. Sorting NaN drift last is deliberate: an
    undefined drift (no usable target, or a portfolio with no value) is
    reported as ``in_band = False`` because "unknown" is not "fine", yet it
    still generates no trade.

    An empty portfolio and an unusable target mapping both yield an empty
    frame with the documented columns rather than an exception.
    """
    frame, meta = _working_frame(positions, targets, band_pct, cash_to_invest=0.0)
    out = frame.loc[:, list(DRIFT_COLUMNS)].copy()
    if not out.empty:
        order = out["drift_pct"].abs()
        out = (
            out.assign(_abs_drift=order.fillna(-1.0))
            .sort_values(["_abs_drift", "ticker"], ascending=[False, True], kind="mergesort")
            .drop(columns="_abs_drift")
            .reset_index(drop=True)
        )
    out["in_band"] = out["in_band"].astype(bool)
    out.attrs.update(_public_meta(meta))
    return out


# ---------------------------------------------------------------------------
# trade list
# ---------------------------------------------------------------------------
def _public_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the working metadata that is published on ``.attrs``."""
    keys = (
        "band_pct",
        "cash_to_invest",
        "total_value",
        "target_value_total",
        "target_sum_pct",
        "targets_renormalised",
        "targets_dropped",
        "n_targets",
        "n_positions",
        "note",
    )
    return {key: meta[key] for key in keys if key in meta}


def _empty_trades(meta: Mapping[str, Any] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": pd.Series(dtype=object),
            "action": pd.Series(dtype=object),
            "shares": pd.Series(dtype=float),
            "price": pd.Series(dtype=float),
            "trade_value": pd.Series(dtype=float),
            "current_weight_pct": pd.Series(dtype=float),
            "target_weight_pct": pd.Series(dtype=float),
            "drift_pct": pd.Series(dtype=float),
        }
    )
    frame.attrs.update(_public_meta(meta or {}))
    frame.attrs.setdefault("no_price", [])
    frame.attrs.setdefault("below_lot_size", [])
    return frame


def rebalance_trades(
    positions: pd.DataFrame | None,
    targets: Mapping[str, float] | pd.Series | None,
    band_pct: float = DEFAULT_REBALANCE_BAND_PCT,
    cash_to_invest: float = 0.0,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """Turn a drift table into an executable order list.

    Columns::

        ticker, action, shares, price, trade_value,
        current_weight_pct, target_weight_pct, drift_pct

    Sizing, for a target portfolio value ``V_t = total_value +
    cash_to_invest``::

        target_value_i = target_weight_pct_i / 100 * V_t
        gap_i          = target_value_i - current_value_i
        shares_i       = gap_i / (price_i * fx_rate_i)

    ``shares_i`` is truncated toward zero unless ``allow_fractional`` is True,
    so a whole-share order always stops SHORT of its target and can never
    overshoot it, and a sell is additionally capped at the quantity held so
    the list never goes short. ``shares`` and ``trade_value`` are signed
    (positive = BUY, negative = SELL); ``price`` is the native quote price
    while ``trade_value`` is base currency, so they differ by ``fx_rate``.

    A name trades only when ``|drift_pct| > band_pct``, or when
    ``cash_to_invest > 0`` and the name sits below its target -- new money is
    deployed even into a book that is already inside its bands. Everything
    else is returned with ``action = "HOLD"``, ``shares = 0`` and
    ``trade_value = 0``, including a breached name whose order rounds down to
    zero shares (its ticker is listed in ``attrs["below_lot_size"]``, since
    that residual drift is real and must not be hidden).

    A target on a name that is not held has no price, so ``shares`` is NaN
    while ``trade_value`` reports the money to deploy; those tickers are in
    ``attrs["no_price"]``.

    Rows are ordered SELL, then BUY, then HOLD -- the order you would actually
    execute in, since the sells fund the buys -- and by descending trade size
    within each side. ``.attrs`` carries the same metadata as
    :func:`drift_table` plus ``no_price`` and ``below_lot_size``.

    Applying the whole list leaves every name within ``band_pct`` of its
    target when weights are measured against ``V_t`` (see the module
    docstring for the proof sketch and the one-share caveat).
    """
    frame, meta = _working_frame(positions, targets, band_pct, cash_to_invest)
    if frame.empty:
        return _empty_trades(meta)

    band = float(meta["band_pct"])
    cash = float(meta["cash_to_invest"])

    drift = frame["drift_pct"].to_numpy(dtype=float)
    gap = frame["value_gap"].to_numpy(dtype=float)
    price_base = frame["price_base"].to_numpy(dtype=float)
    quantity = frame["quantity"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore"):
        breached = np.abs(drift) > band  # NaN drift -> False, no trade
        deploy_cash = (cash > 0) & (gap > 0)  # new money always gets put to work
        tradeable = (breached | deploy_cash) & np.isfinite(gap) & (gap != 0.0)

    # money gap -> share count, in base currency
    priced = np.isfinite(price_base) & (price_base > 0) & np.isfinite(gap)
    raw_shares = np.full(gap.shape, _nan())
    np.divide(gap, price_base, out=raw_shares, where=priced)

    shares = raw_shares if allow_fractional else np.trunc(raw_shares)
    # never sell more than is held (no shorting); NaN quantity imposes no cap
    sell_floor = np.where(np.isfinite(quantity), -np.abs(quantity), -np.inf)
    shares = np.maximum(shares, sell_floor)

    no_price = tradeable & ~priced
    executes = tradeable & (no_price | (np.isfinite(shares) & (shares != 0.0)))

    shares = np.where(executes, shares, 0.0)
    shares = np.where(no_price, _nan(), shares)
    trade_value = np.where(no_price, gap, shares * price_base)
    trade_value = np.where(executes, trade_value, 0.0)

    with np.errstate(invalid="ignore"):
        action = np.where(executes & (gap > 0), BUY, np.where(executes & (gap < 0), SELL, HOLD))

    out = pd.DataFrame(
        {
            "ticker": frame["ticker"].to_numpy(),
            "action": action.astype(object),
            "shares": shares.astype(float),
            "price": frame["price"].to_numpy(dtype=float),
            "trade_value": trade_value.astype(float),
            "current_weight_pct": frame["current_weight_pct"].to_numpy(dtype=float),
            "target_weight_pct": frame["target_weight_pct"].to_numpy(dtype=float),
            "drift_pct": drift,
        }
    )

    rank = out["action"].map(_ACTION_RANK).fillna(len(_ACTION_RANK)).astype(int)
    size = out["trade_value"].abs().fillna(0.0)
    out = (
        out.assign(_rank=rank, _size=size)
        .sort_values(
            ["_rank", "_size", "ticker"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop(columns=["_rank", "_size"])
        .reset_index(drop=True)
    )

    tickers = frame["ticker"].to_numpy()
    out.attrs.update(_public_meta(meta))
    out.attrs["no_price"] = sorted(str(t) for t in tickers[no_price])
    out.attrs["below_lot_size"] = sorted(str(t) for t in tickers[tradeable & ~executes])
    return out[list(TRADE_COLUMNS)]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def rebalance_summary(trades: pd.DataFrame | None) -> dict:
    """Headline numbers for a trade list produced by :func:`rebalance_trades`.

    Keys::

        n_trades       rows whose action is not HOLD
        buy_value      total value bought   (positive, base currency)
        sell_value     total value sold     (positive, base currency)
        turnover_pct   (buy_value + sell_value) / total_value * 100
        net_cash       sell_value - buy_value  (+ released, - required)
        max_drift_pct  largest |drift_pct| in the list
        in_band_count  rows with |drift_pct| <= band_pct

    ``turnover_pct`` is TWO-WAY traded notional over the pre-trade portfolio
    value, so swapping the entire book scores 200%, and it is NaN when the
    frame does not carry a ``total_value`` in ``.attrs``. ``band_pct`` and
    ``total_value`` are read from ``.attrs``; ``band_pct`` falls back to
    ``config.DEFAULT_REBALANCE_BAND_PCT`` if the frame was rebuilt without it.

    A missing or empty frame returns zeros with NaN ratios rather than
    raising.
    """
    summary: dict = {
        "n_trades": 0,
        "buy_value": 0.0,
        "sell_value": 0.0,
        "turnover_pct": _nan(),
        "net_cash": 0.0,
        "max_drift_pct": _nan(),
        "in_band_count": 0,
    }
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return summary

    value = (
        pd.to_numeric(trades["trade_value"], errors="coerce")
        if "trade_value" in trades.columns
        else pd.Series(dtype=float)
    ).to_numpy(dtype=float)
    value = value[np.isfinite(value)] if value.size else value

    buy_value = float(np.abs(value[value > 0]).sum()) if value.size else 0.0
    sell_value = float(np.abs(value[value < 0]).sum()) if value.size else 0.0
    summary["buy_value"] = buy_value
    summary["sell_value"] = sell_value
    summary["net_cash"] = sell_value - buy_value

    if "action" in trades.columns:
        summary["n_trades"] = int((trades["action"].astype(str) != HOLD).sum())
    else:
        summary["n_trades"] = int(np.count_nonzero(value)) if value.size else 0

    total_value = _as_float(trades.attrs.get("total_value"))
    if np.isfinite(total_value) and total_value > 0:
        summary["turnover_pct"] = (buy_value + sell_value) / total_value * 100.0

    if "drift_pct" in trades.columns:
        drift = pd.to_numeric(trades["drift_pct"], errors="coerce").to_numpy(float)
        finite = np.abs(drift[np.isfinite(drift)])
        if finite.size:
            summary["max_drift_pct"] = float(finite.max())
        band = _non_negative(trades.attrs.get("band_pct"), DEFAULT_REBALANCE_BAND_PCT)
        with np.errstate(invalid="ignore"):
            summary["in_band_count"] = int(np.count_nonzero(np.abs(drift) <= band))

    return summary
