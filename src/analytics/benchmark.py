"""Portfolio versus benchmark: relative return, relative risk and capture.

Supersedes ``src/benchmark.py``. Every existing key of :func:`compare` and the
public names ``normalize_to_100`` / ``ALPHA_METHODOLOGY`` / ``DEFAULT_BENCHMARKS``
are preserved verbatim; what is new is the relative-risk block (tracking error,
information ratio, R-squared) and the regime block (up/down capture).

Conventions (restated on each function, collected here so a reviewer can audit
them in one place)
------------------------------------------------------------------------
Overlap             Every comparison is computed on the INNER JOIN of the two
                    series. Comparing an Indian book with a US index over
                    misaligned calendars otherwise measures the holiday
                    schedule rather than the manager.
Annualised return   R_a = (V_end / V_start) ** (TRADING_DAYS / (n - 1)) - 1,
                    imported from ``src.analytics.returns`` so this module and
                    the risk engine cannot drift apart.
Active return       a_t = r_p,t - r_b,t on the overlapping DAILY returns.
Tracking error      TE  = std(a, ddof=1) * sqrt(TRADING_DAYS)
Capture             Computed on MONTHLY compounded returns (the industry
                    convention: daily capture is dominated by noise), reported
                    in PERCENT, where 100 means "matched the index".
Growth / RS         Rebased to exactly ``REBASE_BASE`` (100) on the first
                    common date, so the two lines start together by
                    construction.
Undefined values    ``float("nan")`` (or an empty Series/DataFrame). Nothing
                    here raises on degenerate input and nothing substitutes
                    zero for "undefined".

Identities this module guarantees, and which the tests assert: a series
compared with ITSELF has beta 1, correlation 1, R-squared 1, tracking error 0
and zero excess return; a portfolio whose daily returns are exactly k times
the benchmark's has beta exactly k.

Every constant comes from ``src.config``; none is written inline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import risk as rk
from src.analytics.returns import annualized_return
from src.config import DEFAULT_BENCHMARKS, TRADING_DAYS

__all__ = [
    "DEFAULT_BENCHMARKS",
    "ALPHA_METHODOLOGY",
    "CAPTURE_METHODOLOGY",
    "REBASE_BASE",
    "MONTH_PERIOD",
    "normalize_to_100",
    "relative_strength",
    "tracking_error",
    "information_ratio",
    "r_squared",
    "capture_ratios",
    "treynor_ratio",
    "m2_measure",
    "compare",
]

# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
#: Index level both sides of a growth chart start from. Rebasing is what lets a
#: portfolio worth 1,240,000 and an index at 24,300 share one axis: afterwards,
#: vertical distance IS relative performance.
REBASE_BASE: float = 100.0

#: pandas Period alias for calendar-month bucketing. Capture ratios are a
#: MONTHLY statistic by convention; measured daily they mostly report
#: microstructure noise rather than the manager's behaviour in a regime.
MONTH_PERIOD: str = "M"

ALPHA_METHODOLOGY = (
    "Alpha here is Jensen's alpha from the CAPM: "
    "alpha = R_p - [R_f + beta * (R_b - R_f)], computed with geometrically "
    "annualised returns over the selected window. This is a simplified "
    "approximation -- a rigorous treatment would regress daily excess returns "
    "and report confidence intervals."
)

CAPTURE_METHODOLOGY = (
    "Capture ratios aggregate both series to calendar-month compounded "
    "returns, split the months by the sign of the BENCHMARK, and divide the "
    "portfolio's average by the benchmark's average inside each bucket. They "
    "are reported in percent: 100 means the portfolio matched the index in "
    "that regime, and up-capture above 100 with down-capture below 100 is the "
    "asymmetry every manager is trying to sell. Months in which the benchmark "
    "was exactly flat belong to neither bucket, and a bucket with no months in "
    "it reports a blank rather than a fabricated number."
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


def _clean_series(data: Any) -> pd.Series:
    """Copy-coerce any series-like input to a float Series fit for joining.

    Drops NaN and +/-inf, keeps the LAST value of a duplicated index label so
    an inner join can never trip over a repeated date, and sorts a
    non-monotonic DatetimeIndex so calendar bucketing is well defined. Never
    mutates the caller's object (``copy=True``); unusable input yields an
    EMPTY Series rather than an exception.
    """
    if data is None:
        return pd.Series(dtype=float)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            return pd.Series(dtype=float)
        data = data.iloc[:, 0]
    try:
        series = pd.Series(data, copy=True)
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype=float)

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if not series.index.is_unique:
        series = series[~series.index.duplicated(keep="last")]
    index = series.index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def _empty_pair() -> pd.DataFrame:
    """The canonical empty two-column frame every aligner degrades to."""
    return pd.DataFrame({"portfolio": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float)})


def _align(left: Any, right: Any) -> pd.DataFrame:
    """Inner-join two series into columns ``portfolio`` / ``benchmark``.

    The join is what enforces the "same days on both sides" convention stated
    in the module docstring. Returns an EMPTY frame (never raises) when either
    input is unusable or the indexes share nothing -- including the case where
    the two indexes have incompatible dtypes.
    """
    a = _clean_series(left)
    b = _clean_series(right)
    if a.empty or b.empty:
        return _empty_pair()
    try:
        joined = pd.concat([a.rename("portfolio"), b.rename("benchmark")], axis=1, join="inner")
    except (TypeError, ValueError):
        return _empty_pair()
    joined = joined.dropna()
    if joined.empty:
        return _empty_pair()
    return joined


def _monthly_returns(daily: Any) -> pd.Series:
    """Compound daily returns into calendar months: R_m = prod(1 + r_t) - 1.

    Bucketing by pandas Period (rather than ``resample``) never invents an
    empty month for a period the market was shut. The result is indexed by
    month-END dates, matching ``returns.periodic_returns`` exactly.

    Returns an EMPTY Series when the index carries no dates: calendar months
    cannot be inferred from a positional index, and guessing would be worse
    than a blank.
    """
    r = _clean_series(daily)
    if r.empty or not isinstance(r.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    try:
        buckets = r.index.to_period(MONTH_PERIOD)
    except (ValueError, TypeError, AttributeError):
        return pd.Series(dtype=float)

    compounded = (1.0 + r).groupby(buckets).prod() - 1.0
    if compounded.empty:
        return pd.Series(dtype=float)
    out = pd.Series(
        compounded.to_numpy(dtype=float),
        index=compounded.index.to_timestamp(how="end").normalize(),
        name="monthly_return",
    )
    return out.dropna()


def _monthly_pair(port_returns: Any, bench_returns: Any) -> pd.DataFrame:
    """Both return series aggregated to the SAME calendar months.

    The daily series are aligned FIRST and only then compounded, so every
    month is built from identical days on both sides even when the two
    calendars disagree. Returns an EMPTY two-column frame when monthly
    aggregation is impossible.
    """
    pair = _align(port_returns, bench_returns)
    if pair.empty:
        return _empty_pair()
    port_m = _monthly_returns(pair["portfolio"])
    bench_m = _monthly_returns(pair["benchmark"])
    if port_m.empty or bench_m.empty:
        return _empty_pair()
    try:
        frame = pd.concat(
            [port_m.rename("portfolio"), bench_m.rename("benchmark")],
            axis=1,
            join="inner",
        )
    except (TypeError, ValueError):
        return _empty_pair()
    frame = frame.dropna()
    if frame.empty:
        return _empty_pair()
    return frame


def _pearson(pair: pd.DataFrame) -> float:
    """Pearson correlation of an aligned pair; NaN if either leg is constant."""
    if pair.empty or len(pair) < 2:
        return float("nan")
    a = pair["portfolio"]
    b = pair["benchmark"]
    var_a = _to_float(a.var(ddof=1))
    var_b = _to_float(b.var(ddof=1))
    if not math.isfinite(var_a) or not math.isfinite(var_b):
        return float("nan")
    if var_a == 0.0 or var_b == 0.0:
        return float("nan")
    return _to_float(a.corr(b))


def _jensen_alpha(
    portfolio_return: float, benchmark_return: float, beta: float, rate: float
) -> float:
    """alpha = R_p - [R_f + beta * (R_b - R_f)]; NaN if any input is missing."""
    inputs = (portfolio_return, benchmark_return, beta, rate)
    if not all(math.isfinite(value) for value in inputs):
        return float("nan")
    return float(portfolio_return - (rate + beta * (benchmark_return - rate)))


def _bucket_capture(frame: pd.DataFrame, positive: bool) -> tuple[float, int]:
    """Average portfolio return / average benchmark return in one regime.

    Returns ``(capture_pct, n_months)``. The capture is NaN when the bucket is
    empty or the benchmark's average inside it is exactly zero.
    """
    mask = frame["benchmark"] > 0.0 if positive else frame["benchmark"] < 0.0
    bucket = frame.loc[mask]
    count = int(len(bucket))
    if count == 0:
        return float("nan"), 0
    bench_mean = _to_float(bucket["benchmark"].mean())
    port_mean = _to_float(bucket["portfolio"].mean())
    if not math.isfinite(bench_mean) or bench_mean == 0.0:
        return float("nan"), count
    if not math.isfinite(port_mean):
        return float("nan"), count
    return float(port_mean / bench_mean * 100.0), count


# ---------------------------------------------------------------------------
# rebasing
# ---------------------------------------------------------------------------
def normalize_to_100(series: pd.Series) -> pd.Series:
    """Rebase a series so its first value is exactly 100.

        N_t = V_t / V_0 * REBASE_BASE

    After rebasing, vertical distance between two lines IS relative
    performance, which is the only way a portfolio and an index of different
    magnitudes can honestly share an axis. The first element is exactly
    ``100.0`` by construction -- the integration test asserts it.

    Returns an EMPTY Series (never raises) for empty input or a zero /
    non-finite first observation, where the rebasing divisor is undefined.
    """
    s = _clean_series(series)
    if s.empty:
        return pd.Series(dtype=float)
    first = float(s.iloc[0])
    if not math.isfinite(first) or first == 0.0:
        return pd.Series(dtype=float)
    return s / first * REBASE_BASE


def relative_strength(portfolio_values: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
    """Portfolio-to-benchmark ratio, rebased to 100 on the first common date.

        RS_t = (V_t / V_0) / (B_t / B_0) * REBASE_BASE

    A purely RELATIVE view: the line rises whenever the portfolio beats the
    index regardless of which way the market went, so a rising line in a
    falling market means the book simply lost less. The level carries no
    information -- only the distance from 100 does.

    Computed on the overlapping dates only. Returns an EMPTY Series when the
    two series share no dates, or when either starts at a non-positive level
    (the rebasing divisor is undefined there).
    """
    pair = _align(portfolio_values, benchmark_prices)
    if pair.empty:
        return pd.Series(dtype=float)
    if float(pair["portfolio"].iloc[0]) <= 0.0:
        return pd.Series(dtype=float)
    if float(pair["benchmark"].iloc[0]) <= 0.0:
        return pd.Series(dtype=float)

    port = normalize_to_100(pair["portfolio"])
    bench = normalize_to_100(pair["benchmark"])
    if port.empty or bench.empty:
        return pd.Series(dtype=float)

    ratio = port / bench.where(bench > 0.0) * REBASE_BASE
    ratio.name = "relative_strength"
    return ratio


# ---------------------------------------------------------------------------
# relative risk
# ---------------------------------------------------------------------------
def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualised standard deviation of the active return series.

        a_t = r_p,t - r_b,t                        (on OVERLAPPING dates)
        TE  = std(a, ddof=1) * sqrt(TRADING_DAYS)

    Tracking error answers "how far can this portfolio wander from the index?"
    -- the risk any index-relative mandate is actually measured on. It is
    silent about direction: a manager who beats the index erratically and one
    who trails it erratically score identically.

    ``ddof=1`` because a price history is a sample, not a population.

    Identity: the tracking error of a series against itself is exactly 0.0.
    NaN for fewer than two overlapping observations.
    """
    pair = _align(port_returns, bench_returns)
    if len(pair) < 2:
        return float("nan")
    active = pair["portfolio"] - pair["benchmark"]
    return _to_float(active.std(ddof=1) * math.sqrt(TRADING_DAYS))


def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Active return earned per unit of tracking error.

        a_t = r_p,t - r_b,t
        IR  = mean(a) * TRADING_DAYS / (std(a, ddof=1) * sqrt(TRADING_DAYS))
            = mean(a) / std(a, ddof=1) * sqrt(TRADING_DAYS)

    The numerator is the ARITHMETIC annualised active return, deliberately
    matched to the arithmetic dispersion in the denominator so that the ratio
    is dimensionally consistent. The geometric ``excess_return`` reported by
    :func:`compare` is a different (and equally legitimate) numerator, so the
    two will not agree to the last decimal -- which is why the convention is
    stated here rather than left to be inferred.

    IR is the Sharpe ratio of the active bet: broadly, 0.5 is good and 1.0 is
    exceptional sustained over a full cycle.

    NaN for fewer than two overlapping observations or a zero tracking error
    (an index fund has no information ratio, not an infinite one).
    """
    pair = _align(port_returns, bench_returns)
    if len(pair) < 2:
        return float("nan")
    active = pair["portfolio"] - pair["benchmark"]
    dispersion = _to_float(active.std(ddof=1))
    mean_active = _to_float(active.mean())
    if not math.isfinite(dispersion) or dispersion == 0.0:
        return float("nan")
    if not math.isfinite(mean_active):
        return float("nan")
    return float(mean_active / dispersion * math.sqrt(TRADING_DAYS))


def r_squared(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Fraction of the portfolio's return variance explained by the benchmark.

        R2 = corr(r_p, r_b) ** 2                   (0 to 1)

    This is the honesty check on beta and alpha: a CAPM alpha measured against
    an index that explains 15% of the variance is an artefact, not a skill
    measurement. A broad equity book against its own market index typically
    sits above 0.85.

    Identity: the R-squared of a series against itself is exactly 1.0. NaN for
    fewer than two overlapping observations or a zero-variance leg.
    """
    correlation = _pearson(_align(port_returns, bench_returns))
    if not math.isfinite(correlation):
        return float("nan")
    return float(correlation**2)


def capture_ratios(port_returns: pd.Series, bench_returns: pd.Series) -> dict:
    """Up / down market capture on MONTHLY compounded returns, in PERCENT.

        R_m      = prod_{t in month} (1 + r_t) - 1         (both series)
        up   (%) = mean(R_p | R_b > 0) / mean(R_b | R_b > 0) * 100
        down (%) = mean(R_p | R_b < 0) / mean(R_b | R_b < 0) * 100
        ratio    = up / down                               (dimensionless)

    Monthly aggregation is the convention: measured daily, capture is
    dominated by microstructure noise rather than by how the portfolio behaves
    in a regime. 100 means "matched the index" in that regime, so the shape
    every manager is selling is up above 100 with down below it, and
    ``capture_ratio > 1`` is the one-number summary of that asymmetry.

    Months in which the benchmark was exactly flat belong to NEITHER bucket.
    An empty bucket -- or one whose benchmark average is exactly zero --
    reports NaN rather than a fabricated number: a portfolio measured over a
    window with no down months has no down-capture, which is a fact about the
    sample, not a virtue of the portfolio.

    Returns ``{up_capture, down_capture, capture_ratio, up_months,
    down_months, n_months}``. Never raises; every statistic is NaN and every
    count 0 when the index carries no dates, because calendar months cannot be
    inferred from a positional index.
    """
    blank = {
        "up_capture": float("nan"),
        "down_capture": float("nan"),
        "capture_ratio": float("nan"),
        "up_months": 0,
        "down_months": 0,
        "n_months": 0,
    }

    frame = _monthly_pair(port_returns, bench_returns)
    if frame.empty:
        return blank

    up_capture, up_months = _bucket_capture(frame, positive=True)
    down_capture, down_months = _bucket_capture(frame, positive=False)

    if math.isfinite(up_capture) and math.isfinite(down_capture) and down_capture != 0.0:
        ratio = float(up_capture / down_capture)
    else:
        ratio = float("nan")

    return {
        "up_capture": up_capture,
        "down_capture": down_capture,
        "capture_ratio": ratio,
        "up_months": up_months,
        "down_months": down_months,
        "n_months": int(len(frame)),
    }


# ---------------------------------------------------------------------------
# risk-adjusted rankings against the benchmark
# ---------------------------------------------------------------------------
def treynor_ratio(ann_return: float, beta: float, risk_free_rate: float) -> float:
    """Excess return per unit of SYSTEMATIC risk.

        Treynor = (R_a - R_f) / beta

    Sharpe divides by total volatility; Treynor divides by beta alone, which
    is the right denominator when the portfolio is one sleeve of a larger
    diversified whole and its idiosyncratic risk has already been diversified
    away at the parent level.

    ``risk_free_rate`` is an ANNUALISED decimal (0.065 = 6.5%). A negative beta
    produces a mathematically valid but near-uninterpretable ratio; it is
    returned rather than suppressed, because hiding it would also hide the
    fact that the portfolio is inversely exposed.

    NaN when any input is non-finite or beta is exactly zero (a market-neutral
    book has no Treynor ratio, not an infinite one).
    """
    ret = _to_float(ann_return)
    systematic = _to_float(beta)
    rate = _to_float(risk_free_rate)
    if not math.isfinite(ret) or not math.isfinite(rate):
        return float("nan")
    if not math.isfinite(systematic) or systematic == 0.0:
        return float("nan")
    return float((ret - rate) / systematic)


def m2_measure(
    ann_return: float,
    ann_vol: float,
    bench_return: float,  # noqa: ARG001 - see the note below; part of the published signature
    bench_vol: float,
    risk_free_rate: float,
) -> float:
    """Modigliani risk-adjusted performance (M2), in the units of a return.

    ``bench_return`` is accepted but deliberately unused: M2 rescales the
    portfolio to the BENCHMARK'S VOLATILITY, so only ``bench_vol`` enters the
    formula. It stays in the signature because every caller passes the
    benchmark pair together and dropping it would invite someone to "fix" the
    formula by adding it back in.

        S_p = (R_a - R_f) / sigma_p
        M2  = R_f + S_p * sigma_b

    M2 restates the Sharpe ratio as a RETURN: what the portfolio would have
    earned had it been levered (or de-levered) with cash until its volatility
    matched the benchmark's. That makes it directly comparable with the
    index's own return -- the "M2 alpha" a reader usually wants is
    ``M2 - R_b`` -- whereas a bare Sharpe ratio of 0.94 is comparable with
    nothing a client recognises.

    ``bench_return`` is accepted so a caller holds the whole comparison in one
    call and can form that M2 alpha; by the definition above it does not enter
    the level itself. All rates are ANNUALISED decimals.

    NaN when ``ann_return``, ``bench_vol`` or ``risk_free_rate`` is non-finite,
    or when portfolio volatility is non-finite or exactly zero (the leverage
    factor is undefined there).
    """
    ret = _to_float(ann_return)
    vol = _to_float(ann_vol)
    bench_sigma = _to_float(bench_vol)
    rate = _to_float(risk_free_rate)
    if not math.isfinite(ret) or not math.isfinite(rate):
        return float("nan")
    if not math.isfinite(vol) or vol == 0.0:
        return float("nan")
    if not math.isfinite(bench_sigma):
        return float("nan")
    sharpe = (ret - rate) / vol
    return float(rate + sharpe * bench_sigma)


# ---------------------------------------------------------------------------
# full comparison
# ---------------------------------------------------------------------------
def compare(
    portfolio_values: pd.Series,
    benchmark_prices: pd.Series,
    risk_free_rate: float,
) -> dict:
    """Full portfolio-vs-benchmark comparison on overlapping dates.

    Returns a dict with ``ok=True`` and the metrics below, or exactly
    ``{"ok": False, "reason": <str>}`` when fewer than two dates overlap.

    ``risk_free_rate`` is an ANNUALISED decimal and deliberately has no
    default: an alpha or a Treynor ratio quoted without its risk-free
    assumption is not a number, it is a rumour.

    Keys on success
    ---------------
    Levels (annualised decimals; drawdowns negative decimals):
        ``portfolio_return``, ``benchmark_return``, ``portfolio_vol``,
        ``benchmark_vol``, ``portfolio_mdd``, ``benchmark_mdd``
    Relative performance:
        ``excess_return``  -- portfolio_return - benchmark_return, the
                              geometric relative performance
        ``alpha``          -- Jensen's alpha, per ``ALPHA_METHODOLOGY``
        ``beta``           -- cov(r_p, r_b) / var(r_b)
        ``treynor``, ``m2``
    Relative risk:
        ``tracking_error``, ``information_ratio``, ``correlation``,
        ``r_squared``
    Regime behaviour (monthly; captures in PERCENT):
        ``up_capture``, ``down_capture``, ``capture_ratio``, ``up_months``,
        ``down_months``, ``best_relative_month``, ``worst_relative_month``
        (the last two are monthly active returns as DECIMALS)
    Series and provenance:
        ``growth``               -- DataFrame, both legs rebased to 100
        ``active_return_series`` -- daily r_p - r_b over the overlap
        ``n_days``, ``alpha_methodology``, ``capture_methodology``

    Identities: comparing a series with itself gives beta 1, correlation 1,
    R-squared 1, tracking error 0 and excess return 0, and both ``growth``
    columns start at exactly 100.0. Nothing here raises and nothing is
    fabricated -- every statistic that the available overlap cannot support is
    NaN.
    """
    df = _align(portfolio_values, benchmark_prices)

    if len(df) < 2:
        return {
            "ok": False,
            "reason": "Not enough overlapping dates between portfolio and benchmark.",
        }

    rate = _to_float(risk_free_rate)

    # ---- levels ---------------------------------------------------------
    portfolio_return = annualized_return(df["portfolio"])
    benchmark_return = annualized_return(df["benchmark"])

    rets = rk.daily_returns(df)
    port_r = _clean_series(rets["portfolio"])
    bench_r = _clean_series(rets["benchmark"])

    portfolio_vol = rk.annualized_volatility(port_r)
    benchmark_vol = rk.annualized_volatility(bench_r)

    # ---- relative performance -------------------------------------------
    beta = rk.beta(port_r, bench_r)
    alpha = _jensen_alpha(portfolio_return, benchmark_return, beta, rate)
    if math.isfinite(portfolio_return) and math.isfinite(benchmark_return):
        excess_return = float(portfolio_return - benchmark_return)
    else:
        excess_return = float("nan")

    # ---- relative risk --------------------------------------------------
    pair = _align(port_r, bench_r)
    correlation = _pearson(pair)
    r2 = float(correlation**2) if math.isfinite(correlation) else float("nan")

    if pair.empty:
        active = pd.Series(dtype=float)
    else:
        active = pair["portfolio"] - pair["benchmark"]
    active.name = "active_return"

    # ---- regime behaviour -----------------------------------------------
    capture = capture_ratios(port_r, bench_r)
    monthly = _monthly_pair(port_r, bench_r)
    if monthly.empty:
        best_relative = float("nan")
        worst_relative = float("nan")
    else:
        relative = monthly["portfolio"] - monthly["benchmark"]
        best_relative = _to_float(relative.max())
        worst_relative = _to_float(relative.min())

    # normalize_to_100 returns an EMPTY series when a first value is zero, and
    # dropping an empty series into a DataFrame constructor silently back-fills
    # that whole column with NaN -- producing a blank benchmark line on the
    # growth chart that looks like real data. Rebasing is the one thing this
    # function promises ("both series start at exactly 100"), so if it cannot be
    # done, say so in ok/reason rather than shipping a plausible-looking void.
    port_growth = normalize_to_100(df["portfolio"])
    bench_growth = normalize_to_100(df["benchmark"])
    if port_growth.empty or bench_growth.empty:
        return {
            "ok": False,
            "reason": (
                "Cannot rebase to 100: the first overlapping observation is zero "
                "for the portfolio or the benchmark."
            ),
        }

    growth = pd.DataFrame({"Portfolio": port_growth, "Benchmark": bench_growth})

    return {
        "ok": True,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "beta": beta,
        "alpha": alpha,
        "portfolio_vol": portfolio_vol,
        "benchmark_vol": benchmark_vol,
        "portfolio_mdd": rk.max_drawdown(df["portfolio"]),
        "benchmark_mdd": rk.max_drawdown(df["benchmark"]),
        "tracking_error": tracking_error(port_r, bench_r),
        "information_ratio": information_ratio(port_r, bench_r),
        "treynor": treynor_ratio(portfolio_return, beta, rate),
        "m2": m2_measure(portfolio_return, portfolio_vol, benchmark_return, benchmark_vol, rate),
        "correlation": correlation,
        "r_squared": r2,
        "up_capture": capture["up_capture"],
        "down_capture": capture["down_capture"],
        "capture_ratio": capture["capture_ratio"],
        "up_months": capture["up_months"],
        "down_months": capture["down_months"],
        "best_relative_month": best_relative,
        "worst_relative_month": worst_relative,
        "active_return_series": active,
        "growth": growth,
        "n_days": len(df),
        "alpha_methodology": ALPHA_METHODOLOGY,
        "capture_methodology": CAPTURE_METHODOLOGY,
    }
