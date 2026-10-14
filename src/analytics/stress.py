"""Stress testing: what this book would have lost, and what it could lose.

Two different machines live here and they answer two different questions.

    1. HISTORICAL REPLAY (``historical_stress``, ``scenario_from_returns``)
       Take the portfolio's CURRENT weights, drop them into a documented
       crisis window, and compound each holding's actual price history over
       the dates that actually overlap that window. Nothing is modelled: the
       only assumption is "these weights, that market".

    2. FACTOR SHOCK (``beta_shock``)
       A first-order CAPM projection: a benchmark move of ``s`` is assumed to
       drag the portfolio by ``beta * s``. One number, one line of algebra,
       and a large pile of caveats stated on the function itself.

The coverage problem (the reason this module exists)
----------------------------------------------------
A portfolio built in 2024 has no 2008 in it. A naive stress table hides that
by treating every missing holding as flat, which quietly converts "I have no
idea what this name did in the GFC" into "this name did nothing in the GFC" --
a fabricated 0% that makes the whole portfolio look more robust than it is.

So every replayed row carries a ``coverage`` column: the fraction of portfolio
weight that genuinely had price data inside the window. The reported return is
the return of that COVERED SLEEVE, renormalised so the weights sum to one
across the names that were actually there, and a window with zero coverage
reports ``coverage = 0`` and a NaN return with an explanatory note. It is never
dropped silently and never given a number.

Read the pair together: "-31.4% at 62% coverage" means the 62% of the book
that existed fell 31.4%; the other 38% is unknown, not unharmed.

Conventions
-----------
Weights         accepted as percent or fraction (dict / Series / sequence) and
                normalised internally by their own sum. Long-only is assumed;
                weights that sum to zero yield NaN rather than a division blow-up.
Asset return    total return over the window, ``P_last / P_first - 1``, using
                the first and last VALID observation inside the window. At
                least two observations and a positive opening price are
                required, otherwise the name counts as uncovered.
Portfolio       ``r_p = sum_i(w_i * r_i) / sum_i(w_i)`` over covered names i,
                i.e. the covered sleeve's return. Coverage is the denominator
                and is reported alongside it.
Percent columns anything named ``*_pct`` is in PERCENT (-31.4 = a 31.4% fall);
                ``coverage`` is a FRACTION in [0, 1].
Value columns   ``value_change`` and ``projected_value`` are in the same
                currency as the ``portfolio_value`` passed in, and are NaN
                when no portfolio value was supplied.
Undefined       ``float("nan")`` / empty frames. Nothing here raises on
                degenerate input and nothing substitutes zero for "unknown".

Scenario windows, factor shocks and the default rolling horizon all come from
``src.config``; none is written inline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.risk import max_drawdown
from src.config import (
    FACTOR_SHOCKS,
    HISTORICAL_STRESS,
    MIN_OBSERVATIONS,
    PERIOD_TRADING_DAYS,
    StressWindow,
)

__all__ = [
    "HISTORICAL_COLUMNS",
    "BETA_SHOCK_COLUMNS",
    "WORST_WINDOW_COLUMNS",
    "DEFAULT_STRESS_WINDOW_DAYS",
    "MIN_WINDOW_OBSERVATIONS",
    "historical_stress",
    "beta_shock",
    "worst_rolling_windows",
    "scenario_from_returns",
    "stress_summary",
]


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
#: Output schema of ``historical_stress``.
HISTORICAL_COLUMNS: tuple[str, ...] = (
    "scenario",
    "start",
    "end",
    "days",
    "coverage",
    "portfolio_return_pct",
    "value_change",
    "worst_asset",
    "worst_asset_pct",
    "note",
)

#: Output schema of ``beta_shock``.
BETA_SHOCK_COLUMNS: tuple[str, ...] = (
    "scenario",
    "shock_pct",
    "implied_portfolio_pct",
    "value_change",
    "projected_value",
)

#: Output schema of ``worst_rolling_windows``.
WORST_WINDOW_COLUMNS: tuple[str, ...] = (
    "rank",
    "start",
    "end",
    "days",
    "calendar_days",
    "return_pct",
    "start_value",
    "end_value",
)

#: Default holding period for the worst-window scan: one month of trading,
#: taken from the period table rather than written as a bare 21.
DEFAULT_STRESS_WINDOW_DAYS: int = PERIOD_TRADING_DAYS["1mo"]

#: A window return needs an opening AND a closing price. One observation is a
#: price, not a return, and reporting 0% for it would be a fabricated number.
MIN_WINDOW_OBSERVATIONS: int = 2

#: Coverage within this of 1.0 counts as fully covered (float tolerance only).
_COVERAGE_TOLERANCE: float = 1e-9

#: Rebasing level for the notional stress index when no portfolio value is
#: known. Window RETURNS are scale-invariant, so the base is cosmetic.
_NOTIONAL_BASE: float = 100.0

#: Column dtypes used when materialising the output frames.
_TEXT_COLUMNS: frozenset[str] = frozenset({"scenario", "note", "worst_asset"})
_INT_COLUMNS: frozenset[str] = frozenset({"days", "rank"})


# ---------------------------------------------------------------------------
# internal helpers -- coercion
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> float:
    """Best-effort float coercion; NaN for anything unparseable or infinite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _to_int(value: Any, default: int) -> int:
    """Best-effort int coercion falling back to ``default``."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse a date-like to a tz-naive ``Timestamp``; None when unparseable."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    if stamp.tz is not None:
        stamp = stamp.tz_localize(None)
    return stamp


def _fmt_date(stamp: pd.Timestamp | None) -> str:
    """ISO date for note text; ``?`` when the stamp is missing."""
    if stamp is None or pd.isna(stamp):
        return "?"
    return str(pd.Timestamp(stamp).date())


def _join_note(*parts: str) -> str:
    """Join non-empty note fragments with a single space."""
    return " ".join(p.strip() for p in parts if p and p.strip())


def _clean_values(values: Any) -> pd.Series:
    """Copy-coerce a value series to float, drop NaN/inf, sort a date index.

    Never mutates the caller's object. An unusable input yields an EMPTY
    Series rather than an exception.
    """
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    try:
        series = pd.Series(values, copy=True)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    if len(series) == 0:
        return pd.Series(dtype=float)
    if not pd.api.types.is_numeric_dtype(series):
        series = pd.to_numeric(series, errors="coerce")
    try:
        series = series.astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    series = series[np.isfinite(series.to_numpy(dtype="float64"))]
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
    return series


def _clean_frame(frame: Any) -> pd.DataFrame:
    """Copy-coerce a wide price/return frame to float with a sorted date index.

    Non-numeric cells become NaN, infinities become NaN, duplicate column
    labels keep their first occurrence, and an object index that parses as
    dates is promoted to a tz-naive ``DatetimeIndex``. A numeric (positional)
    index is left alone -- silently reading 0, 1, 2 as 1970 epoch nanoseconds
    would be worse than having no dates at all. Never mutates the input.
    """
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        obj = frame.to_frame()
    elif isinstance(frame, pd.DataFrame):
        obj = frame.copy()
    else:
        try:
            obj = pd.DataFrame(frame)
        except (TypeError, ValueError):
            return pd.DataFrame()
    if obj.shape[0] == 0 or obj.shape[1] == 0:
        return pd.DataFrame()

    labels = pd.Index([str(c) for c in obj.columns])
    obj = obj.loc[:, ~labels.duplicated()]
    obj.columns = [str(c) for c in obj.columns]
    for col in obj.columns:
        if not pd.api.types.is_numeric_dtype(obj[col]):
            obj[col] = pd.to_numeric(obj[col], errors="coerce")
    try:
        obj = obj.astype("float64")
    except (TypeError, ValueError):
        return pd.DataFrame()
    obj = obj.where(np.isfinite(obj))

    if not isinstance(obj.index, pd.DatetimeIndex) and obj.index.dtype == object:
        converted = pd.to_datetime(obj.index, errors="coerce")
        if isinstance(converted, pd.DatetimeIndex) and bool(converted.notna().all()):
            obj.index = converted
    if isinstance(obj.index, pd.DatetimeIndex):
        if obj.index.tz is not None:
            obj.index = obj.index.tz_localize(None)
        if not obj.index.is_monotonic_increasing:
            obj = obj.sort_index()
    return obj


def _normalise(weights: pd.Series) -> pd.Series:
    """Scale weights so they sum to 1; EMPTY when the sum is zero/non-finite."""
    if weights.empty:
        return pd.Series(dtype=float)
    total = float(weights.sum())
    if not math.isfinite(total) or total == 0.0:
        return pd.Series(dtype=float)
    return (weights / total).astype("float64")


def _weight_map(weights: Any, columns: Sequence[str]) -> pd.Series:
    """Align weights to ``columns`` and normalise them to sum to 1.

    Accepts a mapping, a Series, or a bare sequence whose length matches
    ``columns`` (positional alignment). Percent and fraction inputs both work
    because the series is divided by its own sum. Names absent from the price
    frame, zero weights and unparseable weights are dropped. Returns an EMPTY
    Series when nothing usable survives -- that is what makes every downstream
    statistic degrade to NaN instead of raising.
    """
    if weights is None:
        return pd.Series(dtype=float)
    try:
        if isinstance(weights, Mapping):
            series = pd.Series(dict(weights))
        elif isinstance(weights, pd.Series):
            series = pd.Series(weights, copy=True)
        elif isinstance(weights, pd.DataFrame):
            return pd.Series(dtype=float)
        else:
            series = pd.Series(list(weights))
            if len(series) != len(columns):
                return pd.Series(dtype=float)
            series.index = pd.Index([str(c) for c in columns])
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    if len(series) == 0:
        return pd.Series(dtype=float)

    series.index = pd.Index([str(i) for i in series.index])
    series = series[~series.index.duplicated()]
    series = pd.to_numeric(series, errors="coerce")
    series = series[np.isfinite(series.to_numpy(dtype="float64"))]
    if series.empty:
        return pd.Series(dtype=float)

    series = series[series != 0.0]
    if series.empty:
        return pd.Series(dtype=float)

    # Normalise over the WHOLE book FIRST, then restrict to the priced names.
    # Doing it the other way round renormalises the survivors back to 1.0, so a
    # holding with no price column at all silently vanishes from the denominator
    # and a 70%-priced book reports coverage 1.00. Unpriced weight is exactly
    # what this module exists to disclose, so the returned series is allowed to
    # sum to less than 1 -- that shortfall IS the uncovered weight.
    series = _normalise(series)

    keep = [c for c in columns if c in series.index]
    if not keep:
        return pd.Series(dtype=float)
    return series.reindex(keep)


def _window_fields(window: Any) -> tuple[str, Any, Any, str]:
    """Unpack a scenario definition into ``(name, start, end, note)``.

    Accepts ``config.StressWindow``, any object with those attributes, a
    mapping, or a 3/4-element sequence, so callers can pass ad-hoc windows.
    """
    if isinstance(window, Mapping):
        return (
            str(window.get("name", "")),
            window.get("start"),
            window.get("end"),
            str(window.get("note", "") or ""),
        )
    if hasattr(window, "name") and hasattr(window, "start") and hasattr(window, "end"):
        return (
            str(getattr(window, "name", "")),
            getattr(window, "start", None),
            getattr(window, "end", None),
            str(getattr(window, "note", "") or ""),
        )
    try:
        parts = tuple(window)
    except TypeError:
        return ("", None, None, "")
    if len(parts) < 3:
        return ("", None, None, "")
    note = str(parts[3]) if len(parts) > 3 and parts[3] is not None else ""
    return (str(parts[0]), parts[1], parts[2], note)


def _shock_pairs(shocks: Any) -> list[tuple[str, float]]:
    """Unpack factor shocks into ``(label, decimal shock)`` pairs."""
    if shocks is None:
        return []
    if isinstance(shocks, Mapping):
        items: list[Any] = list(shocks.items())
    elif isinstance(shocks, (str, bytes)):
        return []
    else:
        try:
            items = list(shocks)
        except TypeError:
            return []
    pairs: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pairs.append((str(item[0]), _to_float(item[1])))
            continue
        shock = _to_float(item)
        if math.isfinite(shock):
            pairs.append((f"Benchmark {shock * 100.0:+.0f}%", shock))
    return pairs


# ---------------------------------------------------------------------------
# internal helpers -- frame construction
# ---------------------------------------------------------------------------
def _cast_column(frame: pd.DataFrame, col: str, datetime_columns: Sequence[str]) -> None:
    """Coerce one output column to its declared dtype, in place on a local frame."""
    if col in datetime_columns:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    elif col in _TEXT_COLUMNS:
        frame[col] = frame[col].astype("object")
    elif col in _INT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype("int64")
    else:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")


def _empty_frame(columns: Sequence[str], datetime_columns: Sequence[str] = ()) -> pd.DataFrame:
    """A zero-row frame carrying the full schema, so the UI never key-errors."""
    data: dict[str, pd.Series] = {}
    for col in columns:
        if col in datetime_columns:
            dtype: Any = "datetime64[ns]"
        elif col in _TEXT_COLUMNS:
            dtype = "object"
        elif col in _INT_COLUMNS:
            dtype = "int64"
        else:
            dtype = "float64"
        data[col] = pd.Series(dtype=dtype)
    return pd.DataFrame(data)


def _frame(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    datetime_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Materialise result rows with a stable schema and stable dtypes."""
    if not rows:
        return _empty_frame(columns, datetime_columns)
    frame = pd.DataFrame(list(rows), columns=list(columns))
    for col in columns:
        _cast_column(frame, col, datetime_columns)
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# internal helpers -- window arithmetic
# ---------------------------------------------------------------------------
def _total_returns(window: pd.DataFrame) -> pd.Series:
    """Per-asset total return over a price window: ``P_last / P_first - 1``.

    Only columns with at least ``MIN_WINDOW_OBSERVATIONS`` valid prices and a
    strictly positive opening price appear in the result. Everything else is
    OMITTED (not zero-filled) -- absence from this Series is exactly what
    "uncovered" means downstream.
    """
    if window.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for col in window.columns:
        prices = window[col].dropna()
        if len(prices) < MIN_WINDOW_OBSERVATIONS:
            continue
        first = float(prices.iloc[0])
        last = float(prices.iloc[-1])
        if not math.isfinite(first) or not math.isfinite(last) or first <= 0.0:
            continue
        out[col] = last / first - 1.0
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out, dtype="float64")


def _compounded_returns(window: pd.DataFrame) -> pd.Series:
    """Per-asset compounded return over a RETURN window: ``prod(1 + r) - 1``.

    One observation is enough here (unlike prices, a single return is already
    a return). Columns with no observations in the window are omitted.
    """
    if window.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for col in window.columns:
        rets = window[col].dropna()
        if len(rets) < 1:
            continue
        growth = float(np.prod(1.0 + rets.to_numpy(dtype="float64")))
        if not math.isfinite(growth):
            continue
        out[col] = growth - 1.0
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out, dtype="float64")


def _sleeve_return(weights: pd.Series, returns: pd.Series, covered: Sequence[str]) -> float:
    """Weight-renormalised return of the covered sleeve.

        r_p = sum_i(w_i * r_i) / sum_i(w_i)   over covered names i

    Dividing by the covered weight is the whole point: it refuses to treat a
    name with no history as a 0% return.
    """
    if not len(covered):
        return float("nan")
    covered_weight = float(weights.loc[list(covered)].sum())
    if not math.isfinite(covered_weight) or covered_weight <= 0.0:
        return float("nan")
    numerator = float((weights.loc[list(covered)] * returns.loc[list(covered)]).sum())
    if not math.isfinite(numerator):
        return float("nan")
    return numerator / covered_weight


def _coverage_of(weights: pd.Series, covered: Sequence[str]) -> float:
    """Fraction of (normalised) portfolio weight that had data, clamped to [0, 1]."""
    if not len(covered):
        return 0.0
    total = float(weights.loc[list(covered)].sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    return min(total, 1.0)


def _coverage_note(coverage: float, n_covered: int, n_held: int) -> str:
    """Plain-English caveat appended to a partially covered scenario."""
    if coverage >= 1.0 - _COVERAGE_TOLERANCE:
        return ""
    return (
        f"Only {coverage * 100.0:.0f}% of portfolio weight had price history here "
        f"({n_covered} of {n_held} holdings); the return shown is that covered "
        "sleeve alone, not the whole portfolio."
    )


def _weighted_index(prices: pd.DataFrame, weights: pd.Series, base_value: float) -> pd.Series:
    """Notional value path of the CURRENT weights held constant.

        I_t = base * sum_i( w_i * P_(i,t) / P_(i,0) )

    Each holding is rebased to its own first price on the common window (all
    holdings present after a forward fill), so the path answers "how would
    today's book have travelled?" rather than "what did I actually earn?".
    Returns an EMPTY Series when fewer than two common dates exist.
    """
    if prices.empty or weights.empty:
        return pd.Series(dtype=float)
    held = prices.loc[:, list(weights.index)].ffill().dropna(how="any")
    if len(held) < MIN_WINDOW_OBSERVATIONS:
        return pd.Series(dtype=float)
    first = held.iloc[0]
    usable = [c for c in held.columns if math.isfinite(float(first[c])) and float(first[c]) > 0.0]
    if not usable:
        return pd.Series(dtype=float)
    sleeve = _normalise(weights.loc[usable])
    if sleeve.empty:
        return pd.Series(dtype=float)
    usable_base = math.isfinite(base_value) and base_value > 0.0
    base = base_value if usable_base else _NOTIONAL_BASE
    rebased = held.loc[:, usable].div(first[usable], axis=1)
    series = rebased.mul(sleeve, axis=1).sum(axis=1) * base
    series.name = "stress_index"
    return series


# ---------------------------------------------------------------------------
# historical scenario replay
# ---------------------------------------------------------------------------
def historical_stress(
    asset_prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
    windows: Sequence[StressWindow] = HISTORICAL_STRESS,
    portfolio_value: float = float("nan"),
) -> pd.DataFrame:
    """Replay documented crisis windows against the portfolio's current weights.

    For every window the price frame is sliced to the dates that actually fall
    inside it, each holding's total return is computed as
    ``P_last / P_first - 1`` over its own first and last valid observation in
    that slice, and the portfolio impact is the weight-renormalised return of
    the holdings that had data:

        coverage = sum_(i in covered) w_i                     (fraction, <= 1)
        r_p      = sum_(i in covered) w_i * r_i / coverage    (decimal)

    Weights are normalised by their own sum, so percent or fraction input both
    work; ``r_p`` is therefore the return of the COVERED SLEEVE, and
    ``coverage`` tells the reader how much of the book that sleeve is. A name
    listed in 2021 simply has no COVID crash, and this function says so rather
    than scoring it flat.

    Columns
    -------
    ``scenario``              window name
    ``start`` / ``end``       the dates ACTUALLY replayed (NaT when none)
    ``days``                  trading observations used in the replay
    ``coverage``              FRACTION in [0, 1] of weight that had data
    ``portfolio_return_pct``  covered-sleeve return, PERCENT (NaN if uncovered)
    ``value_change``          ``portfolio_value * r_p`` (NaN without a value)
    ``worst_asset``           covered holding with the lowest window return
    ``worst_asset_pct``       that holding's return, PERCENT
    ``note``                  the scenario's context plus any coverage caveat

    Windows with zero coverage are RETAINED with ``coverage = 0``, a NaN return
    and an explanatory note (deviating from "skip them" deliberately: a silently
    missing row reads as a scenario that did no damage). Their names are also
    listed in ``.attrs["skipped"]``. Row order follows ``windows``.

    ``portfolio_value`` is optional and defaults to NaN, in which case
    ``value_change`` is NaN throughout and only the percentage columns carry
    information.

    Degenerate input -- empty/None prices, no usable weights, a non-dated
    index, a reversed window -- yields the full schema with NaN metrics and a
    note saying why, never an exception.
    """
    prices = _clean_frame(asset_prices)
    weight = _weight_map(weights, list(prices.columns))
    value = _to_float(portfolio_value)
    n_held = int(len(weight))
    dated = (not prices.empty) and isinstance(prices.index, pd.DatetimeIndex)

    try:
        window_list = list(windows) if windows is not None else []
    except TypeError:
        window_list = []

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    covered_scenarios: list[str] = []

    for window in window_list:
        name, raw_start, raw_end, base_note = _window_fields(window)
        start_ts = _as_timestamp(raw_start)
        end_ts = _as_timestamp(raw_end)
        row: dict[str, Any] = {
            "scenario": name,
            "start": pd.NaT,
            "end": pd.NaT,
            "days": 0,
            "coverage": 0.0,
            "portfolio_return_pct": float("nan"),
            "value_change": float("nan"),
            "worst_asset": None,
            "worst_asset_pct": float("nan"),
            "note": base_note,
        }

        blocker = ""
        if not dated:
            blocker = "no dated price history was supplied"
        elif weight.empty:
            blocker = "no usable portfolio weights were supplied"
        elif start_ts is None or end_ts is None or end_ts < start_ts:
            blocker = "the scenario window is not a valid date range"
        if blocker:
            row["note"] = _join_note(base_note, f"Not replayed: {blocker}.")
            rows.append(row)
            skipped.append(name)
            continue

        mask = (prices.index >= start_ts) & (prices.index <= end_ts)
        sliced = prices.loc[mask, list(weight.index)]
        returns = _total_returns(sliced)
        covered = [t for t in weight.index if t in returns.index]
        coverage = _coverage_of(weight, covered)

        if not covered or coverage <= 0.0:
            row["note"] = _join_note(
                base_note,
                f"No overlapping price history between {_fmt_date(start_ts)} and "
                f"{_fmt_date(end_ts)}: 0% of portfolio weight is covered, so no "
                "return is reported for this scenario.",
            )
            rows.append(row)
            skipped.append(name)
            continue

        used = sliced.loc[:, covered].dropna(how="all")
        sleeve = _sleeve_return(weight, returns, covered)
        worst_name = str(returns.loc[covered].idxmin())
        row["start"] = used.index[0] if len(used) else pd.NaT
        row["end"] = used.index[-1] if len(used) else pd.NaT
        row["days"] = int(len(used))
        row["coverage"] = coverage
        row["portfolio_return_pct"] = sleeve * 100.0 if math.isfinite(sleeve) else float("nan")
        row["value_change"] = value * sleeve if math.isfinite(sleeve) else float("nan")
        row["worst_asset"] = worst_name
        row["worst_asset_pct"] = float(returns.loc[worst_name]) * 100.0
        row["note"] = _join_note(base_note, _coverage_note(coverage, len(covered), n_held))
        rows.append(row)
        covered_scenarios.append(name)

    frame = _frame(rows, HISTORICAL_COLUMNS, datetime_columns=("start", "end"))
    frame.attrs["skipped"] = skipped
    frame.attrs["covered"] = covered_scenarios
    frame.attrs["n_skipped"] = len(skipped)
    frame.attrs["n_holdings"] = n_held
    frame.attrs["portfolio_value"] = value
    frame.attrs["history_start"] = prices.index[0] if dated else pd.NaT
    frame.attrs["history_end"] = prices.index[-1] if dated else pd.NaT
    frame.attrs["coverage_note"] = (
        "coverage is the fraction of portfolio weight with price history inside "
        "the window; returns are for that covered sleeve only, and scenarios with "
        "zero coverage report NaN rather than a fabricated 0%."
    )
    return frame


def scenario_from_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
    start: Any = None,
    end: Any = None,
) -> dict:
    """Replay an ARBITRARY date range from a per-asset daily-return frame.

    The custom-window twin of ``historical_stress``: each asset's return over
    ``[start, end]`` is the geometric compound ``prod(1 + r_t) - 1`` of its
    observations inside the range, and the portfolio figure is again the
    weight-renormalised return of the covered names,
    ``sum(w_i * r_i) / sum(w_i)``, reported next to that coverage fraction.

    ``start`` / ``end`` accept anything ``pd.Timestamp`` parses (str, date,
    Timestamp) and either may be None for an open-ended side.

    Returns a dict with keys ``ok, reason, start, end, requested_start,
    requested_end, days, n_assets, n_covered, coverage, portfolio_return_pct,
    asset_return_pct, worst_asset, worst_asset_pct, best_asset,
    best_asset_pct, note``. ``ok`` is False -- with ``reason`` set and every
    numeric NaN -- whenever the range holds no usable data; nothing raises.
    """
    frame = _clean_frame(asset_returns)
    weight = _weight_map(weights, list(frame.columns))
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)

    result: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "start": None,
        "end": None,
        "requested_start": start_ts,
        "requested_end": end_ts,
        "days": 0,
        "n_assets": int(len(weight)),
        "n_covered": 0,
        "coverage": 0.0,
        "portfolio_return_pct": float("nan"),
        "asset_return_pct": {},
        "worst_asset": None,
        "worst_asset_pct": float("nan"),
        "best_asset": None,
        "best_asset_pct": float("nan"),
        "note": (
            "Compounded replay of the observed daily returns over the requested "
            "range at the portfolio's current weights. The return covers only the "
            "holdings that had data; read it together with 'coverage'."
        ),
    }

    if frame.empty:
        result["reason"] = "no usable return history supplied"
        return result
    if weight.empty:
        result["reason"] = "no usable portfolio weights supplied"
        return result
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        result["reason"] = "the requested window ends before it starts"
        return result
    if (start_ts is not None or end_ts is not None) and not isinstance(
        frame.index, pd.DatetimeIndex
    ):
        result["reason"] = "return history has no dated index to slice"
        return result

    mask = pd.Series(True, index=frame.index)
    if isinstance(frame.index, pd.DatetimeIndex):
        if start_ts is not None:
            mask &= frame.index >= start_ts
        if end_ts is not None:
            mask &= frame.index <= end_ts
    sliced = frame.loc[mask.to_numpy(), list(weight.index)]

    returns = _compounded_returns(sliced)
    covered = [t for t in weight.index if t in returns.index]
    coverage = _coverage_of(weight, covered)
    if not covered or coverage <= 0.0:
        result["reason"] = (
            f"no overlapping return history between {_fmt_date(start_ts)} and {_fmt_date(end_ts)}"
        )
        return result

    used = sliced.loc[:, covered].dropna(how="all")
    sleeve = _sleeve_return(weight, returns, covered)
    covered_returns = returns.loc[covered]
    worst_name = str(covered_returns.idxmin())
    best_name = str(covered_returns.idxmax())

    result["ok"] = True
    result["start"] = used.index[0] if len(used) else None
    result["end"] = used.index[-1] if len(used) else None
    result["days"] = int(len(used))
    result["n_covered"] = len(covered)
    result["coverage"] = coverage
    result["portfolio_return_pct"] = sleeve * 100.0 if math.isfinite(sleeve) else float("nan")
    result["asset_return_pct"] = {str(k): float(v) * 100.0 for k, v in covered_returns.items()}
    result["worst_asset"] = worst_name
    result["worst_asset_pct"] = float(covered_returns.loc[worst_name]) * 100.0
    result["best_asset"] = best_name
    result["best_asset_pct"] = float(covered_returns.loc[best_name]) * 100.0
    caveat = _coverage_note(coverage, len(covered), int(len(weight)))
    if caveat:
        result["note"] = _join_note(result["note"], caveat)
    return result


# ---------------------------------------------------------------------------
# beta-implied factor shocks
# ---------------------------------------------------------------------------
def beta_shock(
    portfolio_value: float,
    beta: float,
    shocks: Iterable[tuple[str, float]] | Mapping[str, float] = FACTOR_SHOCKS,
) -> pd.DataFrame:
    """First-order CAPM translation of a benchmark move into portfolio money.

        implied_portfolio_return = beta * shock
        value_change             = beta * shock * portfolio_value
        projected_value          = portfolio_value + value_change

    That is the entire model, and its limitations are as important as its
    arithmetic:

    * It is FIRST ORDER only. Idiosyncratic (stock-specific) risk is assumed
      away, so a portfolio can lose far more than ``beta * shock`` from a
      single-name blow-up that has nothing to do with the index.
    * BETA IS NOT STABLE IN A CRISIS. Cross-sectional correlations converge
      toward one in a sell-off, so a 0.8-beta book routinely behaves like a
      1.0-beta book exactly when it matters. Treat the downside rows as an
      optimistic bound.
    * It is INSTANTANEOUS and linear: no path, no convexity, no rebalancing,
      no time for the beta estimate to be re-estimated.
    * Beta is estimated against ONE benchmark over ONE past window. A
      portfolio whose factor exposure differs from that benchmark (small cap,
      sector concentration, another currency) is not described by it.

    Columns: ``scenario, shock_pct, implied_portfolio_pct, value_change,
    projected_value``. Percentages are in PERCENT, ``value_change`` and
    ``projected_value`` in the currency of ``portfolio_value``.

    A non-finite beta or portfolio value leaves the derived columns NaN while
    the scenario rows are still returned, so the UI can show the shock ladder
    with blanks instead of nothing at all.
    """
    beta_f = _to_float(beta)
    value = _to_float(portfolio_value)
    pairs = _shock_pairs(shocks)

    rows: list[dict[str, Any]] = []
    for label, shock in pairs:
        implied = beta_f * shock
        change = beta_f * shock * value
        rows.append(
            {
                "scenario": label,
                "shock_pct": shock * 100.0 if math.isfinite(shock) else float("nan"),
                "implied_portfolio_pct": (
                    implied * 100.0 if math.isfinite(implied) else float("nan")
                ),
                "value_change": change if math.isfinite(change) else float("nan"),
                "projected_value": (value + change if math.isfinite(change) else float("nan")),
            }
        )

    frame = _frame(rows, BETA_SHOCK_COLUMNS)
    frame.attrs["beta"] = beta_f
    frame.attrs["portfolio_value"] = value
    frame.attrs["method"] = (
        "first-order CAPM: impact = beta * benchmark shock; ignores "
        "idiosyncratic risk and assumes beta is stable in a crisis"
    )
    return frame


# ---------------------------------------------------------------------------
# empirical worst windows
# ---------------------------------------------------------------------------
def worst_rolling_windows(
    values: pd.Series,
    window_days: int = DEFAULT_STRESS_WINDOW_DAYS,
    top_n: int = 5,
) -> pd.DataFrame:
    """The worst NON-OVERLAPPING ``window_days`` holding periods actually lived.

    Every window return ``V_t / V_(t - window_days) - 1`` is evaluated, then
    windows are taken greedily worst-first, skipping any that overlaps one
    already taken. Overlap is measured in RETURN space: a window covering
    observations ``(a, b]`` clashes with ``(c, d]`` unless ``b <= c`` or
    ``d <= a``, so two windows may share a single boundary price (the close of
    one is the open of the next) without double-counting a single day's move.

    Greedy selection is not guaranteed to be the globally worst non-overlapping
    SET -- it is the standard, deterministic and explainable choice, and it
    guarantees the property that matters: the same bad fortnight can never be
    reported five times with the start date nudged by a day.

    Rows are sorted worst-first. Columns: ``rank`` (1 = worst), ``start``,
    ``end``, ``days`` (the window length in observations), ``calendar_days``,
    ``return_pct`` (PERCENT), ``start_value``, ``end_value``.

    Returns an EMPTY frame with the full schema when the series is shorter
    than ``window_days + 1``, or when ``window_days``/``top_n`` are not
    positive.
    """
    span = _to_int(window_days, DEFAULT_STRESS_WINDOW_DAYS)
    count = _to_int(top_n, 0)
    series = _clean_values(values)
    dated = isinstance(series.index, pd.DatetimeIndex)

    if span < 1 or count < 1 or len(series) < span + 1:
        return _empty_frame(WORST_WINDOW_COLUMNS, ("start", "end"))

    prices = series.to_numpy(dtype="float64")
    starts = np.arange(0, len(prices) - span, dtype=int)
    ends = starts + span
    base = prices[starts]
    with np.errstate(divide="ignore", invalid="ignore"):
        window_returns = np.where(base > 0.0, prices[ends] / base - 1.0, np.nan)
    window_returns = np.where(np.isfinite(window_returns), window_returns, np.nan)

    order = np.argsort(window_returns, kind="stable")
    taken: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for position in order:
        ret = float(window_returns[position])
        if not math.isfinite(ret):
            continue
        lo, hi = int(starts[position]), int(ends[position])
        if any(not (hi <= t_lo or t_hi <= lo) for t_lo, t_hi in taken):
            continue
        taken.append((lo, hi))
        start_label = series.index[lo]
        end_label = series.index[hi]
        if dated:
            calendar = float((pd.Timestamp(end_label) - pd.Timestamp(start_label)).days)
        else:
            calendar = float("nan")
        rows.append(
            {
                "rank": len(rows) + 1,
                "start": start_label if dated else pd.NaT,
                "end": end_label if dated else pd.NaT,
                "days": span,
                "calendar_days": calendar,
                "return_pct": ret * 100.0,
                "start_value": float(prices[lo]),
                "end_value": float(prices[hi]),
            }
        )
        if len(rows) >= count:
            break

    frame = _frame(rows, WORST_WINDOW_COLUMNS, datetime_columns=("start", "end"))
    frame.attrs["window_days"] = span
    frame.attrs["n_observations"] = int(len(series))
    frame.attrs["dated_index"] = bool(dated)
    frame.attrs["selection"] = (
        "greedy worst-first over non-overlapping windows; windows may share a "
        "single boundary observation but never a return day"
    )
    if not dated:
        frame.attrs["note"] = "value series had no date index, so start/end are NaT"
    return frame


# ---------------------------------------------------------------------------
# combined report
# ---------------------------------------------------------------------------
def stress_summary(
    asset_prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None,
    portfolio_value: float,
    beta: float,
    windows: Sequence[StressWindow] = HISTORICAL_STRESS,
    shocks: Iterable[tuple[str, float]] | Mapping[str, float] = FACTOR_SHOCKS,
) -> dict:
    """Assemble the three stress views into one dict for the UI and the report.

    Combines the historical replay, the beta shock ladder and the empirical
    worst-window scan, and adds the one comparison that keeps the table
    honest: the portfolio's own realised maximum drawdown over the supplied
    history (``sample_max_drawdown_pct``), computed on a notional index of the
    current weights held constant. A stress table whose worst scenario is
    milder than what the book has already survived is a coverage artefact, not
    a clean bill of health.

    Keys
    ----
    ``ok``, ``reason``                      whether anything could be replayed
    ``historical``                          ``historical_stress`` frame
    ``beta_shocks``                         ``beta_shock`` frame
    ``worst_windows``                       ``worst_rolling_windows`` frame
    ``worst_scenario`` / ``_pct`` / ``_value_change`` / ``_coverage``
                                            deepest covered scenario
    ``mean_coverage``                       mean coverage across ALL scenarios
    ``n_scenarios``, ``n_covered``, ``n_skipped``, ``skipped``
    ``sample_max_drawdown_pct``             realised MDD of the notional index
    ``n_observations``, ``sufficient_history``, ``history_start``, ``history_end``
    ``n_holdings``, ``portfolio_value``, ``beta``, ``note``

    Every numeric key is a float and is NaN when undefined. Degenerate input
    returns the same shape with empty frames and ``ok=False``.
    """
    prices = _clean_frame(asset_prices)
    weight = _weight_map(weights, list(prices.columns))
    value = _to_float(portfolio_value)
    beta_f = _to_float(beta)

    historical = historical_stress(prices, weight, windows, portfolio_value=value)
    shock_table = beta_shock(value, beta_f, shocks)
    index = _weighted_index(prices, weight, value)
    worst_windows = worst_rolling_windows(index, DEFAULT_STRESS_WINDOW_DAYS)

    covered = historical.loc[historical["coverage"] > 0.0]
    scored = covered.loc[covered["portfolio_return_pct"].notna()]
    if scored.empty:
        worst_row: pd.Series | None = None
    else:
        worst_row = scored.loc[scored["portfolio_return_pct"].idxmin()]

    dated = (not prices.empty) and isinstance(prices.index, pd.DatetimeIndex)
    n_obs = int(len(index))
    reason = ""
    if prices.empty:
        reason = "no usable price history supplied"
    elif weight.empty:
        reason = "no usable portfolio weights supplied"
    elif worst_row is None:
        reason = "no scenario window overlapped the available price history"

    coverage_column = historical["coverage"]
    mean_coverage = float(coverage_column.mean()) if len(coverage_column) else float("nan")

    return {
        "ok": worst_row is not None,
        "reason": reason,
        "historical": historical,
        "beta_shocks": shock_table,
        "worst_windows": worst_windows,
        "worst_scenario": (str(worst_row["scenario"]) if worst_row is not None else None),
        "worst_scenario_pct": (
            float(worst_row["portfolio_return_pct"]) if worst_row is not None else float("nan")
        ),
        "worst_scenario_value_change": (
            float(worst_row["value_change"]) if worst_row is not None else float("nan")
        ),
        "worst_scenario_coverage": (
            float(worst_row["coverage"]) if worst_row is not None else float("nan")
        ),
        "mean_coverage": mean_coverage,
        "n_scenarios": int(len(historical)),
        "n_covered": int(len(covered)),
        "n_skipped": int(historical.attrs.get("n_skipped", 0)),
        "skipped": list(historical.attrs.get("skipped", [])),
        "sample_max_drawdown_pct": (float(max_drawdown(index)) * 100.0 if n_obs else float("nan")),
        "n_observations": n_obs,
        "sufficient_history": bool(n_obs >= MIN_OBSERVATIONS),
        "history_start": prices.index[0] if dated else None,
        "history_end": prices.index[-1] if dated else None,
        "n_holdings": int(len(weight)),
        "portfolio_value": value,
        "beta": beta_f,
        "note": (
            "Historical rows replay actual prices at today's weights and are "
            "reported alongside the fraction of weight that had data; scenarios "
            "with no overlapping history report no return at all. Beta rows are a "
            "first-order CAPM approximation that ignores idiosyncratic risk and "
            "assumes beta holds up in a crisis, which it historically does not. "
            "Worst-window rows are the deepest non-overlapping "
            f"{DEFAULT_STRESS_WINDOW_DAYS}-day stretches this book actually lived "
            "through."
        ),
    }
