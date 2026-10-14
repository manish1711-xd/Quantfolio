"""Return engine: money-weighted (XIRR) and time-weighted (TWR) performance.

Two questions get conflated constantly, so this module answers both explicitly
and labels which is which:

* "How did the *assets* perform?"  -> time-weighted return (TWR). External
  contributions and withdrawals are neutralised, which is why TWR is the basis
  of every published fund track record.
* "How did the *investor* do?"     -> money-weighted return (MWR / XIRR): the
  internal rate of return of the actual dated cash flows, so a large
  contribution made at the top of the market genuinely hurts it.

Conventions
-----------
Day count       ACT/365 fixed (``DAY_COUNT``) for everything driven by dated
                cash flows: XIRR, NPV, CAGR, Modified Dietz.
Annualisation   ``TRADING_DAYS / (n - 1)`` for statistics driven by a value
                series, so this module and ``src.analytics.risk`` agree to the
                last decimal.
Flow sign       ``npv``/``xirr`` sign flows from the *investor's pocket*: money
                paid in is negative, money received is positive.
                ``time_weighted_return``/``modified_dietz`` sign flows from the
                *portfolio's* perspective: a contribution in is positive.
                Both conventions are restated on the functions themselves.
Degenerate in   Every public function returns ``float("nan")`` (or an empty
                Series/DataFrame). Nothing here raises, nothing loops without a
                bound, and nothing substitutes zero for "undefined".
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton

from src.config import TRADING_DAYS

__all__ = [
    "DAY_COUNT",
    "RATE_LOWER",
    "RATE_UPPER",
    "MONTH_LABELS",
    "npv",
    "xirr",
    "cagr",
    "annualized_return",
    "cumulative_return",
    "time_weighted_return",
    "modified_dietz",
    "periodic_returns",
    "monthly_return_table",
    "rolling_returns",
    "portfolio_xirr",
    "return_stats",
]

# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
#: ACT/365 fixed. Dated cash-flow maths must use calendar time -- an investor's
#: money is at risk on weekends too -- so 365 is the divisor, not TRADING_DAYS.
DAY_COUNT: float = 365.0

#: XIRR search domain. Below -100% the discount factor (1 + r) ** t is
#: undefined for fractional t; +10000% is far beyond any credible portfolio.
RATE_LOWER: float = -0.999
RATE_UPPER: float = 100.0

#: A candidate root is accepted when |NPV(r)| is within this fraction of the
#: largest single cash flow (an absolute tolerance is meaningless when flows
#: can be either 1e2 or 1e9).
_NPV_RTOL: float = 1e-6

#: Fixed ASCII month labels -- deliberately not ``calendar.month_abbr``, which
#: is locale-dependent and would change the report's column names by machine.
MONTH_LABELS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> float:
    """Best-effort float coercion; NaN for anything unparseable or infinite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_date(value: Any) -> date | None:
    """Coerce a date / datetime / Timestamp / string to ``datetime.date``.

    Returns None (never raises) when the value cannot be interpreted, which is
    what lets the cleaners drop bad rows instead of failing a whole report.
    """
    if value is None:
        return None
    if isinstance(value, datetime):  # covers pd.Timestamp
        return value.date()
    if isinstance(value, date):
        return value
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return stamp.to_pydatetime().date()


def _clean_flows(flows: Any) -> list[tuple[date, float]]:
    """Normalise a flow schedule to a date-sorted list of (date, amount).

    Accepts a sequence of 2-tuples or a ``pd.Series`` indexed by date. Rows
    with an uninterpretable date or a non-finite amount are dropped rather
    than poisoning the solve.
    """
    if flows is None:
        return []
    if isinstance(flows, pd.Series):
        pairs: Any = list(flows.items())
    elif isinstance(flows, pd.DataFrame):
        return []
    else:
        try:
            pairs = list(flows)
        except TypeError:
            return []

    items: list[tuple[date, float]] = []
    for item in pairs:
        try:
            raw_date, raw_amount = item
        except (TypeError, ValueError):
            continue
        when = _as_date(raw_date)
        amount = _to_float(raw_amount)
        if when is not None and math.isfinite(amount):
            items.append((when, amount))
    items.sort(key=lambda pair: pair[0])
    return items


def _clean_values(values: Any) -> pd.Series:
    """Copy-coerce any value series to a float Series with NaNs dropped.

    Never mutates the caller object (``copy=True``) and sorts a non-monotonic
    DatetimeIndex so period bucketing is well defined.
    """
    if values is None:
        return pd.Series(dtype=float)
    try:
        series = pd.Series(values, copy=True)
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    series = series.dropna()
    index = series.index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def _simple_returns(values: pd.Series) -> pd.Series:
    """r_t = V_t / V_(t-1) - 1 on an already-cleaned series.

    Written out rather than using ``pct_change`` so the result never depends
    on the pandas fill-method default, which has changed across versions.
    Points whose previous value is non-positive become NaN, not +/-inf.
    """
    if len(values) < 2:
        return pd.Series(dtype=float)
    previous = values.shift(1)
    return (values / previous.where(previous > 0) - 1.0).iloc[1:]


def _discount_factor(rate: float, years: float) -> float:
    """(1 + r) ** t with an overflow guard.

    An overflowed factor is reported as +inf, which discounts that flow to
    zero -- the mathematically correct limit rather than an exception.
    """
    base = 1.0 + rate
    if base <= 0.0:
        return float("nan")
    try:
        return float(base**years)
    except (OverflowError, ValueError):
        return float("inf")


def _same_sign(a: float, b: float) -> bool:
    """Sign comparison that cannot overflow the way ``a * b < 0`` can."""
    return (a < 0.0) == (b < 0.0)


def _find_bracket(func: Any, guess: float, max_expansions: int = 64) -> tuple[float, float] | None:
    """Locate an interval on which the NPV changes sign.

    Two bounded strategies, in order: (1) a two-sided interval around ``guess``
    whose half-width doubles until it reaches the domain edges, and (2) a
    coarse fixed scan of the whole domain looking for an adjacent-node sign
    change (which catches the non-conventional schedules the widening search
    can straddle). Both loops are finite by construction: this cannot spin.
    """
    anchor = func(guess)
    if not math.isfinite(anchor):
        return None

    low = high = guess
    step = 0.05
    for _ in range(max_expansions):
        low = max(RATE_LOWER, low - step)
        high = min(RATE_UPPER, high + step)
        f_low = func(low)
        if math.isfinite(f_low) and not _same_sign(f_low, anchor):
            return (low, guess)
        f_high = func(high)
        if math.isfinite(f_high) and not _same_sign(f_high, anchor):
            return (guess, high)
        if low <= RATE_LOWER and high >= RATE_UPPER:
            break
        step *= 2.0

    grid = (
        RATE_LOWER,
        -0.95,
        -0.9,
        -0.75,
        -0.5,
        -0.3,
        -0.15,
        -0.05,
        0.0,
        0.05,
        0.15,
        0.3,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        RATE_UPPER,
    )
    previous_x: float | None = None
    previous_f = float("nan")
    for node in grid:
        value = func(node)
        if not math.isfinite(value):
            continue
        if previous_x is not None and not _same_sign(value, previous_f):
            return (previous_x, node)
        previous_x, previous_f = node, value
    return None


def _accept_root(root: float, func: Any, tolerance: float) -> bool:
    """A root counts only if it is in-domain and genuinely zeroes the NPV."""
    if root is None or not math.isfinite(root):
        return False
    if not RATE_LOWER <= root <= RATE_UPPER:
        return False
    residual = func(root)
    return math.isfinite(residual) and abs(residual) <= tolerance


def _transaction_flows(transactions: Any) -> list[tuple[date, float]]:
    """Extract (trade_date, cash_flow) pairs from Transactions or 2-tuples."""
    if transactions is None:
        return []
    try:
        sequence = list(transactions)
    except TypeError:
        return []

    items: list[tuple[date, float]] = []
    for item in sequence:
        if hasattr(item, "trade_date") and hasattr(item, "cash_flow"):
            when = _as_date(item.trade_date)
            amount = _to_float(item.cash_flow)
        else:
            try:
                raw_date, raw_amount = item
            except (TypeError, ValueError):
                continue
            when = _as_date(raw_date)
            amount = _to_float(raw_amount)
        if when is not None and math.isfinite(amount):
            items.append((when, amount))
    items.sort(key=lambda pair: pair[0])
    return items


def _longest_streak(returns: pd.Series, positive: bool) -> int:
    """Longest run of consecutive strictly positive (or negative) returns."""
    best = run = 0
    for value in returns.to_numpy(dtype=float):
        hit = value > 0.0 if positive else value < 0.0
        run = run + 1 if hit else 0
        best = max(best, run)
    return int(best)


# ---------------------------------------------------------------------------
# money-weighted returns (dated cash flows)
# ---------------------------------------------------------------------------
def npv(rate: float, flows: Sequence[tuple[date, float]]) -> float:
    """Net present value of dated cash flows, discounted ACT/365.

        NPV(r) = sum_i  CF_i / (1 + r) ** ((d_i - d_0) / 365)

    ``d_0`` is the earliest flow date, so the NPV is stated as of the first
    flow. Flows are signed from the investor pocket (paid in = negative).

    Returns NaN for an empty schedule, a non-finite rate, or ``rate <= -1``,
    where the fractional-power discount factor is undefined.
    """
    items = _clean_flows(flows)
    discount_rate = _to_float(rate)
    if not items or not math.isfinite(discount_rate) or discount_rate <= -1.0:
        return float("nan")

    origin = items[0][0]
    total = 0.0
    for when, amount in items:
        years = (when - origin).days / DAY_COUNT
        factor = _discount_factor(discount_rate, years)
        if factor == float("inf"):
            continue  # discounted to zero
        if not math.isfinite(factor) or factor == 0.0:
            return float("nan")
        total += amount / factor
    return float(total) if math.isfinite(total) else float("nan")


def xirr(flows: Sequence[tuple[date, float]], guess: float = 0.1) -> float:
    """Money-weighted (internal) rate of return of irregular dated flows.

    Solves ``npv(r, flows) = 0`` for the annualised r over the domain
    ``(RATE_LOWER, RATE_UPPER)``, using in order:

    1. Brent's method (``scipy.optimize.brentq``) on a bracket found by a
       widening two-sided search around ``guess`` and then a coarse scan of
       the whole domain. Brent is tried first precisely because it is
       bracketed and so cannot diverge or wander off to a spurious root;
    2. Newton-Raphson (``scipy.optimize.newton``) from ``guess`` as a fallback
       for the rare schedule whose sign change cannot be bracketed on the grid.

    Day count is ACT/365. Any candidate root is verified by substitution back
    into :func:`npv` before it is returned, so a "converged" but wrong answer
    from either solver is rejected.

    Returns NaN -- never raises, never iterates unbounded -- when

    * fewer than two usable flows survive cleaning,
    * the flows lack both a negative and a positive amount (with no sign
      change there is no rate of return: you cannot earn a return on money
      that was never put in),
    * every flow falls on the same calendar date (zero elapsed time), or
    * no verified root exists inside the search domain.
    """
    items = _clean_flows(flows)
    if len(items) < 2:
        return float("nan")

    amounts = [amount for _, amount in items]
    if not min(amounts) < 0.0 < max(amounts):
        return float("nan")
    if items[0][0] == items[-1][0]:
        return float("nan")

    def objective(rate: float) -> float:
        return npv(rate, items)

    start = _to_float(guess)
    if not math.isfinite(start) or not RATE_LOWER < start < RATE_UPPER:
        start = 0.1

    scale = max(abs(amount) for amount in amounts) or 1.0
    tolerance = max(_NPV_RTOL * scale, 1e-9)

    # NOTE: there is deliberately no "NPV at the guess is already small enough,
    # return the guess" short circuit here. That tolerance scales with the
    # largest flow, so on an institution-sized schedule (1e9) an NPV of ~1000
    # counts as zero and xirr would hand back the raw guess -- making the answer
    # depend on the guess, which is precisely what a root finder must not do.
    # The scale-relative tolerance is still the right test for REJECTING a
    # candidate root below; it is not a valid test for accepting one.
    bracket = _find_bracket(objective, start)
    if bracket is not None:
        low, high = min(bracket), max(bracket)
        if low < high:
            try:
                root = float(brentq(objective, low, high, xtol=1e-12, maxiter=200))
            except (ValueError, RuntimeError, OverflowError):
                root = float("nan")
            if _accept_root(root, objective, tolerance):
                return root

    with warnings.catch_warnings():
        # Newton signals non-convergence with a RuntimeWarning; here that is
        # an expected outcome, not a defect -- the result is verified below.
        warnings.simplefilter("ignore")
        try:
            root = float(newton(objective, start, tol=1e-10, maxiter=100))
        except (
            RuntimeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
            FloatingPointError,
            TypeError,
        ):
            root = float("nan")
    if _accept_root(root, objective, tolerance):
        return root
    return float("nan")


def portfolio_xirr(
    transactions: Sequence[Any],
    current_value: float,
    as_of: date,
) -> float:
    """XIRR of the whole portfolio, including the still-open position.

    Builds the investor cash-flow schedule from the transaction list --
    ``Transaction.cash_flow`` is negative for a BUY (money out of pocket) and
    positive for a SELL -- and appends ``current_value`` at ``as_of`` as the
    terminal inflow. That "as if liquidated today" terminal flow is what turns
    an open portfolio into a solvable IRR problem, and is the standard
    convention for reporting a personal money-weighted return.

    ``transactions`` may be ``models.Transaction`` objects or plain
    ``(date, amount)`` pairs (e.g. the output of ``lots.cash_flows``).

    Returns NaN when there are no usable flows, ``as_of`` is unparseable,
    ``current_value`` is not finite, or the flows have no sign change.
    """
    items = _transaction_flows(transactions)
    value = _to_float(current_value)
    valuation_date = _as_date(as_of)
    if not items or valuation_date is None or not math.isfinite(value):
        return float("nan")
    if value > 0.0:
        items.append((valuation_date, value))
    return xirr(items)


def modified_dietz(
    begin_value: float,
    end_value: float,
    flows: Sequence[tuple[date, float]],
    start: date,
    end: date,
) -> float:
    """Modified Dietz money-weighted return for one period (decimal).

        MD  = (EV - BV - F) / (BV + sum_i w_i * F_i)
        w_i = (T - t_i) / T,   T   = (end - start) in calendar days
                               t_i = (flow_date - start) in calendar days

    Each flow is weighted by the fraction of the period it was actually
    invested for, which makes MD a closed-form approximation to :func:`xirr`
    that needs no root finding -- the industry workhorse from before daily
    valuations were universal, and still the GIPS fallback.

    Flows are signed from the PORTFOLIO perspective here: a contribution into
    the portfolio is POSITIVE (the opposite of :func:`xirr`). Flows dated
    outside ``[start, end]`` are ignored.

    Returns NaN when either date is unparseable, the period has non-positive
    length, either value is non-finite, or the average invested capital (the
    denominator) is non-positive.
    """
    begin = _to_float(begin_value)
    final = _to_float(end_value)
    first_day = _as_date(start)
    last_day = _as_date(end)
    if not math.isfinite(begin) or not math.isfinite(final):
        return float("nan")
    if first_day is None or last_day is None:
        return float("nan")

    total_days = (last_day - first_day).days
    if total_days <= 0:
        return float("nan")

    in_period = [
        (when, amount) for when, amount in _clean_flows(flows) if first_day <= when <= last_day
    ]
    net_flow = math.fsum(amount for _, amount in in_period)
    weighted_flow = math.fsum(
        amount * ((total_days - (when - first_day).days) / total_days) for when, amount in in_period
    )

    denominator = begin + weighted_flow
    if not math.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    result = (final - begin - net_flow) / denominator
    return float(result) if math.isfinite(result) else float("nan")


# ---------------------------------------------------------------------------
# time-weighted returns (value series)
# ---------------------------------------------------------------------------
def cagr(values: pd.Series) -> float:
    """Compound annual growth rate over the CALENDAR span of the series.

        CAGR = (V_end / V_start) ** (365 / elapsed_days) - 1        (ACT/365)

    Unlike :func:`annualized_return` this reads the DatetimeIndex, so it is
    unaffected by how many trading days happened to fall inside the window --
    and it is the figure that reconciles with :func:`xirr` on an account with
    a single opening flow. Falls back to :func:`annualized_return` (trading-day
    annualisation) when the index carries no dates.

    NaN for fewer than two observations, a non-positive start value,
    non-positive total growth, or a zero-length calendar span.
    """
    series = _clean_values(values)
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")

    growth = float(series.iloc[-1] / series.iloc[0])
    if not math.isfinite(growth) or growth <= 0.0:
        return float("nan")

    if not isinstance(series.index, pd.DatetimeIndex):
        return annualized_return(series)

    elapsed = int((series.index[-1] - series.index[0]).days)
    if elapsed <= 0:
        return float("nan")
    return float(growth ** (DAY_COUNT / elapsed) - 1.0)


def annualized_return(values: pd.Series) -> float:
    """Geometric annualised return (CAGR-style) from a value series.

        R_a = (V_end / V_start) ** (TRADING_DAYS / (n - 1)) - 1

    ``n - 1`` is the number of return intervals, so exactly one year of daily
    closes annualises to its own total return. Trading-day annualisation (not
    calendar) is used so this stays identical to the numerator of the risk
    module's Sharpe/Sortino/Calmar ratios.

    NaN for fewer than two points, a non-positive start value, or non-positive
    total growth -- a portfolio that reached zero has no geometric mean.
    """
    series = _clean_values(values)
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")
    total_growth = float(series.iloc[-1] / series.iloc[0])
    if not math.isfinite(total_growth) or total_growth <= 0.0:
        return float("nan")
    n_intervals = len(series) - 1
    return float(total_growth ** (TRADING_DAYS / n_intervals) - 1.0)


def cumulative_return(values: pd.Series) -> float:
    """Total return over the whole window as a decimal: V_end / V_start - 1.

    NaN for fewer than two observations or a non-positive start value.
    """
    series = _clean_values(values)
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")
    result = float(series.iloc[-1] / series.iloc[0] - 1.0)
    return result if math.isfinite(result) else float("nan")


def time_weighted_return(values: pd.Series, flows: pd.Series | None = None) -> float:
    """Annualised time-weighted return, chain-linked across external flows.

        r_t = (V_t - F_t) / V_(t-1) - 1     F_t = external flow during period t
        TWR = prod_t (1 + r_t) ** (TRADING_DAYS / (n - 1)) - 1

    Removing F_t from the closing value before dividing by the opening value
    is what strips the timing and size of contributions out of the result:
    TWR measures the assets, XIRR measures the investor.

    ``flows`` is a Series aligned to the index of ``values``, signed from the
    PORTFOLIO perspective (a contribution in is positive) and assumed to occur
    at the END of its period. Duplicate flow dates are summed; dates absent
    from ``values`` contribute nothing.

    With ``flows=None`` every F_t is zero, the product telescopes to
    V_end / V_start, and the result equals :func:`annualized_return` exactly.

    NaN for fewer than two observations, any non-positive opening value, or a
    non-positive chained growth factor (a wipe-out has no geometric mean).
    """
    series = _clean_values(values)
    if len(series) < 2:
        return float("nan")

    opening = series.to_numpy(dtype=float)[:-1]
    closing = series.to_numpy(dtype=float)[1:]
    if np.any(opening <= 0.0):
        return float("nan")

    external = np.zeros(closing.shape, dtype=float)
    if flows is not None:
        flow_series = _clean_values(flows)
        if not flow_series.empty:
            if not flow_series.index.is_unique:
                flow_series = flow_series.groupby(level=0).sum()
            try:
                aligned = flow_series.reindex(series.index).fillna(0.0)
            except (ValueError, TypeError):
                return float("nan")
            external = aligned.to_numpy(dtype=float)[1:]

    growth_factors = (closing - external) / opening
    if not np.all(np.isfinite(growth_factors)) or np.any(growth_factors <= 0.0):
        return float("nan")

    chained = float(np.prod(growth_factors))
    if not math.isfinite(chained) or chained <= 0.0:
        return float("nan")
    n_intervals = len(series) - 1
    return float(chained ** (TRADING_DAYS / n_intervals) - 1.0)


# ---------------------------------------------------------------------------
# periodic decomposition
# ---------------------------------------------------------------------------
#: Resample-style aliases mapped to their Period equivalents. pandas 2.x
#: spells period-end offsets "ME"/"QE"/"YE" for resampling but "M"/"Q"/"Y" for
#: Periods. Bucketing via Periods (instead of ``Series.resample``) is used
#: deliberately: it never invents an empty bucket for a month the market was
#: shut, and it avoids a pandas/numpy deprecation in the resample bin-edge code.
_PERIOD_ALIASES: dict[str, str] = {
    "ME": "M",
    "MS": "M",
    "M": "M",
    "QE": "Q",
    "QS": "Q",
    "Q": "Q",
    "YE": "Y",
    "YS": "Y",
    "A": "Y",
    "AS": "Y",
    "Y": "Y",
    "W": "W",
    "D": "D",
}


def _period_alias(freq: str) -> str:
    """Translate a resample offset alias to the Period alias, if it has one.

    Returns the UPPER-CASED alias even when it is not in the table. pandas
    deprecated lowercase Period aliases, so passing a bare "m" straight through
    emits a FutureWarning -- which this project's own pytest config promotes to
    an error, breaking the module's documented "never raises" contract. Every
    accepted spelling is normalised the same way.
    """
    return _PERIOD_ALIASES.get(str(freq).strip().upper(), str(freq).strip().upper())


def periodic_returns(values: pd.Series, freq: str = "ME") -> pd.Series:
    """Compounded returns bucketed to a calendar frequency (decimals).

        R_p = prod_{t in p} (1 + r_t) - 1

    Compounding the daily returns inside each bucket -- rather than taking
    period-end closes and differencing them -- keeps the FIRST (partial)
    period honest: it is measured from the first observation instead of being
    silently discarded. For complete periods the two definitions agree
    exactly.

    ``freq`` accepts either spelling of the period-end aliases ("ME"/"M",
    "QE"/"Q", "YE"/"Y", "W"). The result is indexed by period-END dates, and
    periods with no observations are simply absent rather than reported as a
    0% return.

    Returns an EMPTY Series (never raises) for fewer than two observations, a
    non-datetime index, or an unrecognised frequency.
    """
    series = _clean_values(values)
    if len(series) < 2 or not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)

    daily = _simple_returns(series).dropna()
    if daily.empty:
        return pd.Series(dtype=float)

    try:
        periods = daily.index.to_period(_period_alias(freq))
    except (ValueError, TypeError, AttributeError):
        return pd.Series(dtype=float)

    compounded = (1.0 + daily).groupby(periods).prod() - 1.0
    if compounded.empty:
        return pd.Series(dtype=float)

    out = pd.Series(
        compounded.to_numpy(dtype=float),
        index=compounded.index.to_timestamp(how="end").normalize(),
        name="periodic_return",
    )
    return out.dropna()


def monthly_return_table(values: pd.Series) -> pd.DataFrame:
    """Calendar-month return table in PERCENT: rows=year, cols=Jan..Dec, Year.

    ``Year`` is the compounded return of the months actually present in that
    row, ``(prod(1 + R_m) - 1) * 100``, so a partial year compounds only the
    months it has and nothing is imputed. Absent months stay NaN so the UI can
    render an em dash rather than a fabricated zero.

    Returns an EMPTY DataFrame when the series yields no monthly returns.
    """
    monthly = periodic_returns(values, "ME")
    if monthly.empty:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "value": monthly.to_numpy(dtype=float),
        }
    )
    table = frame.pivot_table(index="year", columns="month", values="value", aggfunc="last")
    table = table.reindex(columns=range(1, 13)) * 100.0
    table.columns = list(MONTH_LABELS)

    annual = (1.0 + monthly).groupby(monthly.index.year).prod() - 1.0
    table["Year"] = (annual * 100.0).reindex(table.index)
    table.index.name = "year"
    table.columns.name = None
    return table.sort_index()


def rolling_returns(values: pd.Series, window_days: int) -> pd.Series:
    """Trailing total return over ``window_days`` OBSERVATIONS (not annualised).

        rr_t = V_t / V_(t - window) - 1

    The window counts observations (trading days for a daily series) rather
    than calendar days, so it lines up with the rolling risk statistics. The
    first ``window_days`` entries are NaN by construction, as is any point
    whose lagged value is non-positive.

    Returns an EMPTY Series for a non-positive or unparseable window, or when
    the series is not longer than the window.
    """
    series = _clean_values(values)
    window = _to_float(window_days)
    if not math.isfinite(window) or window < 1:
        return pd.Series(dtype=float)
    window = int(window)
    if len(series) <= window:
        return pd.Series(dtype=float)

    lagged = series.shift(window)
    out = series / lagged.where(lagged > 0) - 1.0
    out.name = f"rolling_{window}d_return"
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def return_stats(values: pd.Series) -> dict:
    """Headline return statistics for one value series.

    Percentage entries are PERCENT (2.5 means +2.5%); counts are ints. Any
    statistic that is undefined comes back NaN -- in particular every monthly
    figure is NaN/0 when the index carries no dates, because calendar months
    cannot be inferred from a positional index.

    Keys: ``cumulative_pct, annualized_pct, best_month_pct, worst_month_pct,
    positive_months, total_months, best_day_pct, worst_day_pct,
    longest_win_streak, longest_loss_streak``.
    """
    series = _clean_values(values)
    monthly = periodic_returns(series, "ME")
    daily = _simple_returns(series).dropna()

    def _extreme(sample: pd.Series, worst: bool) -> float:
        if sample.empty:
            return float("nan")
        value = float(sample.min() if worst else sample.max())
        return value * 100.0 if math.isfinite(value) else float("nan")

    return {
        "cumulative_pct": cumulative_return(series) * 100.0,
        "annualized_pct": annualized_return(series) * 100.0,
        "best_month_pct": _extreme(monthly, worst=False),
        "worst_month_pct": _extreme(monthly, worst=True),
        "positive_months": int((monthly > 0).sum()),
        "total_months": int(len(monthly)),
        "best_day_pct": _extreme(daily, worst=False),
        "worst_day_pct": _extreme(daily, worst=True),
        "longest_win_streak": _longest_streak(daily, positive=True),
        "longest_loss_streak": _longest_streak(daily, positive=False),
    }
