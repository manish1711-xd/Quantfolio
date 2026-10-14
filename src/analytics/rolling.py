"""Rolling risk statistics: how the portfolio's character changed over time.

A single Sharpe ratio for a five-year history is an average of five very
different market regimes. These functions recompute the headline statistics
over a moving window so the UI can show when volatility regime-shifted, when
beta drifted, and whether a good full-period number came from one lucky
quarter.

Conventions
-----------
Window          counted in OBSERVATIONS (trading days for a daily series),
                never calendar days, so every statistic in ``rolling_frame``
                shares one denominator.
Warm-up         the first ``window - 1`` points are NaN by construction
                (``min_periods == window``); nothing is back-filled and no
                partial window is ever reported as if it were complete.
Too little data  fewer than ``config.MIN_ROLLING_WINDOW`` usable observations
                returns an EMPTY Series/DataFrame rather than a line made of
                two points. A window LONGER than the sample returns a
                correctly indexed all-NaN series (the window simply never
                fills).
Annualisation   ``TRADING_DAYS`` throughout, matching ``src.analytics.risk``:
                volatility by ``sqrt``, window return geometrically by
                ``TRADING_DAYS / window``.
Risk-free rate  an ANNUALISED decimal, subtracted from the ANNUALISED window
                return -- the same convention as ``risk.sharpe_ratio``.
Degenerate in   NaN inside the series, empty containers outside. Nothing here
                raises and nothing substitutes zero for "undefined".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.risk import MIN_VAR_OBSERVATIONS
from src.config import DEFAULT_VAR_CONFIDENCE, MIN_ROLLING_WINDOW, TRADING_DAYS, AppConfig

__all__ = [
    "DEFAULT_ROLLING_WINDOW",
    "MIN_ROLLING_WINDOW",
    "ROLLING_COLUMNS",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_beta",
    "rolling_correlation",
    "rolling_var",
    "rolling_frame",
]

# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
#: Default look-back, taken from the app's own default (~one quarter) rather
#: than written inline here.
DEFAULT_ROLLING_WINDOW: int = AppConfig().rolling_window

#: Column order produced by :func:`rolling_frame`.
ROLLING_COLUMNS: tuple[str, ...] = (
    "volatility",
    "sharpe",
    "sortino",
    "beta",
    "correlation",
    "var",
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


def _clean_returns(returns: Any) -> pd.Series:
    """Copy-coerce a return series to float, drop NaN/inf, sort a date index.

    Never mutates the caller's object (``copy=True``). Mirrors the helper in
    ``src.analytics.risk`` so both modules digest the same inputs the same way.
    """
    if returns is None:
        return pd.Series(dtype=float)
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            return pd.Series(dtype=float)
        returns = returns.iloc[:, 0]
    try:
        series = pd.Series(returns, copy=True)
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    index = series.index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def _valid_window(window: Any) -> int:
    """Coerce a window to an int >= 2; 0 signals "unusable, return empty".

    Two observations are the hard floor: every statistic here is a dispersion
    or co-dispersion measure and ``ddof=1`` needs at least two points.
    """
    size = _to_float(window)
    if not math.isfinite(size) or size < 2:
        return 0
    return int(size)


def _empty(name: str) -> pd.Series:
    """A named, float-typed empty Series -- the "not enough data" answer."""
    return pd.Series(dtype=float, name=name)


def _prepare(returns: Any, window: Any) -> tuple[pd.Series, int] | None:
    """Clean the returns and validate the window; None when unusable.

    ``None`` means the caller must return :func:`_empty`: either the window is
    nonsense or the sample is shorter than ``MIN_ROLLING_WINDOW``.
    """
    r = _clean_returns(returns)
    size = _valid_window(window)
    if size == 0 or len(r) < MIN_ROLLING_WINDOW:
        return None
    return r, size


def _join(port_returns: Any, bench_returns: Any) -> pd.DataFrame:
    """Inner-join two return series on their common dates.

    Rolling beta and correlation are meaningless across misaligned calendars,
    so the join happens BEFORE the window is applied: the window then always
    contains ``window`` genuinely paired observations.
    """
    left = _clean_returns(port_returns)
    right = _clean_returns(bench_returns)
    if left.empty or right.empty:
        return pd.DataFrame(columns=["port", "bench"], dtype=float)
    # Collapse duplicate timestamps FIRST. pd.concat(join="inner") raises
    # ValueError("cannot reindex on an axis with duplicate labels") the moment a
    # repeated date appears -- and a repeated date is easy to produce, e.g. by
    # concatenating two overlapping history pulls. This module promises never to
    # raise, so keep the last observation for any duplicated timestamp, which is
    # the same convention the fetcher uses when it de-duplicates history.
    left = left[~left.index.duplicated(keep="last")].rename("port")
    right = right[~right.index.duplicated(keep="last")].rename("bench")
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    joined.columns = ["port", "bench"]
    return joined


def _window_total_return(returns: pd.Series, window: int) -> pd.Series:
    """Compounded total return over each trailing window: prod(1 + r) - 1.

    Computed in log space (exact and O(n)) whenever every gross return is
    positive; a series containing a total wipeout (``r <= -1``) falls back to
    a rolling product so no ``log(0)`` warning is ever emitted.
    """
    growth = 1.0 + returns
    if bool((growth <= 0.0).any()):
        return growth.rolling(window).apply(np.prod, raw=True) - 1.0
    return np.expm1(np.log(growth).rolling(window).sum())


def _annualized_window_return(returns: pd.Series, window: int) -> pd.Series:
    """Window total return annualised geometrically.

        R_a = (prod(1 + r) ) ** (TRADING_DAYS / window) - 1

    A window that lost everything (gross growth 0) annualises to exactly
    -100%; a negative gross growth is not raisable to a fractional power and
    yields NaN rather than a complex number.
    """
    base = 1.0 + _window_total_return(returns, window)
    exponent = float(TRADING_DAYS) / float(window)
    values = base.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0.0
    out[positive] = np.power(values[positive], exponent) - 1.0
    out[values == 0.0] = -1.0
    return pd.Series(out, index=base.index)


def _positive(series: pd.Series) -> pd.Series:
    """Mask non-positive denominators to NaN so a ratio degrades, not explodes."""
    return series.where(series > 0.0)


# ---------------------------------------------------------------------------
# dispersion
# ---------------------------------------------------------------------------
def rolling_volatility(returns: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """Annualised volatility over a trailing window of daily returns.

        sigma_t = std(r_(t-window+1 .. t), ddof=1) * sqrt(TRADING_DAYS)

    ``ddof=1`` matches ``risk.annualized_volatility``, so the last value of
    this series equals the full-period figure when ``window == len(returns)``.

    The first ``window - 1`` points are NaN. Returns an EMPTY Series when the
    window is unusable (< 2) or the sample is shorter than
    ``MIN_ROLLING_WINDOW``.
    """
    prepared = _prepare(returns, window)
    if prepared is None:
        return _empty("volatility")
    r, size = prepared
    out = r.rolling(size).std(ddof=1) * math.sqrt(TRADING_DAYS)
    out.name = "volatility"
    return out


def rolling_sharpe(
    returns: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Annualised excess return per unit of total volatility, rolling.

        S_t = (R_a,t - R_f) / sigma_t

    with ``R_a,t`` the window's compounded return annualised by
    ``TRADING_DAYS / window`` and ``sigma_t`` from :func:`rolling_volatility`.
    ``risk_free_rate`` is an ANNUALISED decimal (0.065 = 6.5%).

    Windows with exactly zero volatility give NaN, not an infinite Sharpe.
    EMPTY Series under the usual insufficient-data conditions.
    """
    prepared = _prepare(returns, window)
    if prepared is None:
        return _empty("sharpe")
    r, size = prepared
    rate = _to_float(risk_free_rate)
    if not math.isfinite(rate):
        rate = 0.0

    ann_return = _annualized_window_return(r, size)
    ann_vol = r.rolling(size).std(ddof=1) * math.sqrt(TRADING_DAYS)
    out = (ann_return - rate) / _positive(ann_vol)
    out.name = "sharpe"
    return out


def rolling_sortino(
    returns: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Annualised excess return per unit of DOWNSIDE deviation, rolling.

        DD_t  = sqrt( mean( min(r, 0) ** 2 ) over the window ) * sqrt(TRADING_DAYS)
        So_t  = (R_a,t - R_f) / DD_t

    The minimum acceptable return inside the deviation is 0 (capital
    preservation), matching ``risk.downside_deviation``'s default, while the
    risk-free rate is subtracted in the numerator, matching
    ``risk.sortino_ratio``. The mean is taken over EVERY observation in the
    window, not just the losing ones -- the standard Sortino denominator.

    A window with no losing day has ``DD_t = 0`` and yields NaN (undefined,
    not infinite). EMPTY Series under the usual insufficient-data conditions.
    """
    prepared = _prepare(returns, window)
    if prepared is None:
        return _empty("sortino")
    r, size = prepared
    rate = _to_float(risk_free_rate)
    if not math.isfinite(rate):
        rate = 0.0

    ann_return = _annualized_window_return(r, size)
    shortfall = r.clip(upper=0.0) ** 2
    downside = np.sqrt(shortfall.rolling(size).mean()) * math.sqrt(TRADING_DAYS)
    out = (ann_return - rate) / _positive(downside)
    out.name = "sortino"
    return out


# ---------------------------------------------------------------------------
# relationship to a benchmark
# ---------------------------------------------------------------------------
def rolling_beta(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """Rolling market sensitivity: cov(r_p, r_b) / var(r_b) over the window.

    Both series are inner-joined on their common dates FIRST, so each window
    holds ``window`` genuinely paired observations rather than a mixture of
    two trading calendars. ``ddof=1`` on both moments, matching ``risk.beta``.

    Identity: the rolling beta of a series against itself is exactly 1.0 once
    the window has filled. Zero-variance benchmark windows give NaN. EMPTY
    Series when the overlap is shorter than ``MIN_ROLLING_WINDOW``.
    """
    joined = _join(port_returns, bench_returns)
    size = _valid_window(window)
    if size == 0 or len(joined) < MIN_ROLLING_WINDOW:
        return _empty("beta")

    port, bench = joined["port"], joined["bench"]
    covariance = port.rolling(size).cov(bench)
    variance = bench.rolling(size).var(ddof=1)
    out = covariance / _positive(variance)
    out.name = "beta"
    return out


def rolling_correlation(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """Rolling Pearson correlation between portfolio and benchmark returns.

        rho_t = cov(r_p, r_b) / (sigma_p * sigma_b)   over the window

    Computed from the explicit ratio rather than ``Series.rolling().corr()`` so
    a zero-variance window becomes NaN silently instead of a divide-by-zero
    warning or a spurious +/-inf. The result is clipped to [-1, 1] to absorb
    floating-point overshoot.

    Same inner-join and EMPTY-Series rules as :func:`rolling_beta`.
    """
    joined = _join(port_returns, bench_returns)
    size = _valid_window(window)
    if size == 0 or len(joined) < MIN_ROLLING_WINDOW:
        return _empty("correlation")

    port, bench = joined["port"], joined["bench"]
    covariance = port.rolling(size).cov(bench)
    spread = port.rolling(size).std(ddof=1) * bench.rolling(size).std(ddof=1)
    out = (covariance / _positive(spread)).clip(-1.0, 1.0)
    out.name = "correlation"
    return out


# ---------------------------------------------------------------------------
# tail risk
# ---------------------------------------------------------------------------
def rolling_var(
    returns: pd.Series,
    window: int = DEFAULT_ROLLING_WINDOW,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> pd.Series:
    """Rolling one-day historical Value at Risk, as a NEGATIVE decimal loss.

        VaR_t = empirical quantile of the window at (1 - confidence)

    Same sign convention as ``risk.value_at_risk``: -0.021 reads "on the worst
    5% of days in this window the portfolio lost at least 2.1%". Linear
    interpolation between order statistics, matching ``np.percentile``.

    A tail quantile estimated from fewer than ``risk.MIN_VAR_OBSERVATIONS``
    points is noise, so a window shorter than that yields an all-NaN series
    (mirroring the guard in ``risk.value_at_risk``). A confidence level
    outside (0, 1) also gives all-NaN. EMPTY Series under the usual
    insufficient-data conditions.
    """
    prepared = _prepare(returns, window)
    if prepared is None:
        return _empty("var")
    r, size = prepared

    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0 or size < MIN_VAR_OBSERVATIONS:
        out = pd.Series(np.nan, index=r.index, dtype=float)
    else:
        out = r.rolling(size).quantile(1.0 - level)
    out.name = "var"
    return out


# ---------------------------------------------------------------------------
# combined frame
# ---------------------------------------------------------------------------
def rolling_frame(
    port_returns: pd.Series,
    bench_returns: pd.Series | None = None,
    window: int = DEFAULT_ROLLING_WINDOW,
    risk_free_rate: float = 0.0,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> pd.DataFrame:
    """Every rolling statistic on one index: the risk-evolution table.

    Columns are :data:`ROLLING_COLUMNS` -- ``volatility, sharpe, sortino,
    beta, correlation, var`` -- indexed by the portfolio's own dates. The
    benchmark-relative columns are reindexed onto that index, so a benchmark
    that trades on a different calendar leaves NaN on the non-overlapping days
    instead of silently shifting the portfolio series.

    ``bench_returns=None`` (or a benchmark with too little overlap) still
    returns the full column set with ``beta`` and ``correlation`` all NaN, so
    the UI never has to test for a missing column.

    Returns an EMPTY frame carrying the full column set when the portfolio has
    fewer than ``MIN_ROLLING_WINDOW`` observations or the window is unusable.
    Never raises.
    """
    prepared = _prepare(port_returns, window)
    if prepared is None:
        return pd.DataFrame(columns=list(ROLLING_COLUMNS), dtype=float)
    r, size = prepared

    frame = pd.DataFrame(index=r.index)
    frame["volatility"] = rolling_volatility(r, size)
    frame["sharpe"] = rolling_sharpe(r, size, risk_free_rate)
    frame["sortino"] = rolling_sortino(r, size, risk_free_rate)

    beta = rolling_beta(r, bench_returns, size)
    correlation = rolling_correlation(r, bench_returns, size)
    frame["beta"] = beta.reindex(r.index) if not beta.empty else np.nan
    frame["correlation"] = correlation.reindex(r.index) if not correlation.empty else np.nan

    frame["var"] = rolling_var(r, size, confidence)
    return frame[list(ROLLING_COLUMNS)].astype("float64")
