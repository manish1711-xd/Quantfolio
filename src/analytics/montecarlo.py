"""Forward simulation: the distribution of outcomes a portfolio *could* print.

Four generators are offered, deliberately, because the honest answer to "what
happens next?" depends entirely on which assumption you are willing to make.
Each is stated below with the assumption it buys and the failure it hides.

    1. GBM (``simulate_gbm``)
       S_{t+1} = S_t * exp((mu - sigma^2/2) + sigma * z),   z ~ N(0, 1)

       ``mu`` and ``sigma`` are DAILY simple-return moments. The ``-sigma^2/2``
       term is the Ito convexity adjustment: without it the simulated arithmetic
       mean would drift above ``mu`` because exp() is convex. With it,

           E[S_T]      = S_0 * exp(mu * T)                (arithmetic drift)
           median[S_T] = S_0 * exp((mu - sigma^2/2) * T)  (log-return drift)

       and the gap between those two lines is the volatility drag. GBM's flaw
       is its Normal tail: it cannot produce the crashes markets actually have.

    2. IID bootstrap (``simulate_bootstrap`` with ``block_size <= 1``)
       Each simulated day is drawn with replacement from the observed daily
       returns, so the empirical skew and fat tails survive exactly. What dies
       is the ORDER: independent draws destroy volatility clustering, so
       multi-day drawdowns are understated. The worst observed day is also a
       hard ceiling - the bootstrap can never invent a loss it has not seen.

    3. Moving-block bootstrap (``simulate_bootstrap`` with ``block_size > 1``)
       Contiguous runs of ``b`` days are resampled, so dependence up to ``b``
       days (vol clustering, short-run momentum/reversal) survives inside a
       block and is broken at the joins. Blocks are non-circular, which
       under-weights the first and last ``b - 1`` observations of the sample.

    4. Student-t (``simulate_student_t``)
       GBM with the Normal innovation replaced by a t(df) standardised to unit
       variance, ``z = t_df / sqrt(df / (df - 2))``, so ``sigma`` still means
       what it says. Fat tails, but symmetric ones: real equity returns are
       left-skewed, and this model is not.

What every method above assumes, and what none of them can do
-------------------------------------------------------------
* i.i.d. draws from a STATIONARY distribution. Markets are neither: volatility
  clusters, correlations converge in a crisis, and the mean is not constant.
* No regime change. The sample cannot contain a crisis it has never seen, so a
  book with only post-2021 history has no COVID crash in its future either.
* Frozen composition: no rebalancing, no contributions, no withdrawals, and no
  transaction costs or taxes.
* Parameters known exactly. They are not - they are estimates. The standard
  error of an annualised drift estimated from one year of daily data is
  typically wider than the drift itself, which is why the fan chart's centre
  line deserves far less attention than its width.

Conventions
-----------
Shape           every generator returns ``(n_paths, horizon_days + 1)`` with
                column 0 equal to ``start_value``. Column ``t`` is the value
                after ``t`` simulated days.
Determinism     identical ``seed`` -> identical paths, on any platform, via
                ``numpy.random.default_rng``. ``seed=None`` draws fresh entropy.
Bankruptcy      simple-return paths floor growth at zero: a portfolio cannot go
                below zero, and a path that reaches zero stays there.
Degenerate in   generators return an EMPTY ``(0, horizon + 1)`` array, summary
                statistics return ``float("nan")``, and ``run_simulation``
                returns ``{"ok": False, "reason": ...}``. Nothing raises and
                nothing invents a number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_VAR_CONFIDENCE,
    MIN_OBSERVATIONS,
    SIMULATION,
    TRADING_DAYS,
    SimulationConfig,
)

__all__ = [
    "GAIN_THRESHOLD",
    "MAX_STUDENT_T_DF",
    "METHODS",
    "MIN_STUDENT_T_DF",
    "TERMINAL_HIGH_PCT",
    "TERMINAL_LOW_PCT",
    "resolve_method",
    "run_simulation",
    "simulate_bootstrap",
    "simulate_gbm",
    "simulate_student_t",
    "summarise_paths",
]

# ---------------------------------------------------------------------------
# module constants (config carries no simulation thresholds beyond SIMULATION)
# ---------------------------------------------------------------------------
#: Upside move reported by the ``prob_gain_10pct`` key. Named here rather than
#: inlined so the key and the threshold can never drift apart.
GAIN_THRESHOLD: float = 0.10

#: Percentiles behind the headline ``p5_terminal`` / ``p95_terminal`` keys.
TERMINAL_LOW_PCT: float = 5.0
TERMINAL_HIGH_PCT: float = 95.0

#: A Student-t has finite variance only for df > 2, and the standardisation
#: 1 / sqrt(df / (df - 2)) explodes as df approaches 2. Below this floor the
#: "volatility" input would be meaningless, so it is rejected.
MIN_STUDENT_T_DF: float = 2.5

#: Above this the t is indistinguishable from a Normal; clamping keeps the
#: method-of-moments estimate from returning absurd values on thin-tailed data.
MAX_STUDENT_T_DF: float = 50.0

#: Fallback degrees of freedom when the sample kurtosis cannot pin one down.
#: 5 is the usual empirical ballpark for daily equity index returns.
DEFAULT_STUDENT_T_DF: float = 5.0

#: Canonical method names accepted by :func:`run_simulation`.
METHODS: tuple[str, ...] = ("gbm", "bootstrap", "block_bootstrap", "student_t")

#: Spellings the UI (or a saved config) might hand us, mapped to a canonical
#: method. "historical" is an alias for the bootstrap because that is what a
#: historical simulation is.
_METHOD_ALIASES: dict[str, str] = {
    "gbm": "gbm",
    "geometric_brownian_motion": "gbm",
    "lognormal": "gbm",
    "normal": "gbm",
    "gaussian": "gbm",
    "parametric": "gbm",
    "bootstrap": "bootstrap",
    "iid": "bootstrap",
    "iid_bootstrap": "bootstrap",
    "historical": "bootstrap",
    "empirical": "bootstrap",
    "block": "block_bootstrap",
    "block_bootstrap": "block_bootstrap",
    "moving_block": "block_bootstrap",
    "moving_block_bootstrap": "block_bootstrap",
    "student_t": "student_t",
    "student-t": "student_t",
    "studentt": "student_t",
    "t": "student_t",
    "fat_tail": "student_t",
}

#: Excess kurtosis of a Student-t is 6 / (df - 4) for df > 4, so a sample
#: kurtosis inverts to df = 4 + 6 / k. Below this k the estimate is noise.
_MIN_KURTOSIS_FOR_DF: float = 1e-3


# ---------------------------------------------------------------------------
# input hygiene
# ---------------------------------------------------------------------------
def _empty_paths(horizon_days: Any) -> np.ndarray:
    """The "no answer" array: zero paths, but the promised column count."""
    steps = _as_int(horizon_days, 0)
    return np.empty((0, max(steps, 0) + 1), dtype=float)


def _as_int(value: Any, default: int) -> int:
    """Best-effort int coercion; ``default`` for None/NaN/garbage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return int(number)


def _as_float(value: Any, default: float = float("nan")) -> float:
    """Best-effort float coercion; ``default`` for None/non-finite/garbage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clean_returns(returns: Any) -> np.ndarray:
    """Finite daily simple returns as a fresh 1-D float array.

    Accepts a Series, a single-column frame, an array or any sequence.
    Non-numeric entries, NaN and +/-inf are dropped rather than filled: a
    missing day is missing, and imputing zero would understate volatility.
    The caller's object is copied and never modified.
    """
    if returns is None:
        return np.empty(0, dtype=float)
    try:
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] != 1:
                return np.empty(0, dtype=float)
            series = returns.iloc[:, 0]
        elif isinstance(returns, pd.Series):
            series = returns
        else:
            values = np.asarray(returns, dtype="float64").ravel()
            return np.array(values[np.isfinite(values)], dtype=float)
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    return np.array(values[np.isfinite(values)], dtype=float)


def _start_value_from(values: Any) -> float:
    """Last finite observation of a value series (or the scalar itself).

    ``run_simulation`` is normally handed a portfolio value Series; a bare
    number is accepted too so the simulation page can be driven from a single
    "current value" figure. NaN when nothing usable is present.
    """
    if values is None:
        return float("nan")
    if isinstance(values, (int, float, np.integer, np.floating)) and not isinstance(values, bool):
        return _as_float(values)
    cleaned = _clean_returns(values)  # same coercion rules, different meaning
    if cleaned.size == 0:
        return float("nan")
    return float(cleaned[-1])


def _percentile_labels(percentiles: Any) -> list[tuple[float, str]]:
    """Sorted, de-duplicated ``(percentile, column_label)`` pairs in (0, 100)."""
    if percentiles is None:
        percentiles = SIMULATION.percentiles
    if isinstance(percentiles, (int, float, np.integer, np.floating)):
        percentiles = (percentiles,)
    seen: dict[float, str] = {}
    try:
        candidates = list(percentiles)
    except TypeError:
        candidates = list(SIMULATION.percentiles)
    for raw in candidates:
        value = _as_float(raw)
        if not np.isfinite(value) or value <= 0.0 or value >= 100.0:
            continue
        label = f"p{int(value)}" if float(value).is_integer() else f"p{value:g}"
        seen[float(value)] = label
    if not seen:
        seen = {float(p): f"p{int(p)}" for p in SIMULATION.percentiles}
    return [(p, seen[p]) for p in sorted(seen)]


# ---------------------------------------------------------------------------
# path construction
# ---------------------------------------------------------------------------
def _paths_from_log_steps(start_value: float, log_steps: np.ndarray) -> np.ndarray:
    """Compound per-day LOG returns into a value matrix with a start column.

    ``S_t = S_0 * exp(sum_{i<=t} l_i)``. Log space cannot produce a negative
    value, so no bankruptcy floor is needed here.
    """
    n_paths, horizon = log_steps.shape
    paths = np.empty((n_paths, horizon + 1), dtype=float)
    paths[:, 0] = start_value
    # An extreme (but legitimate) draw can overflow exp(); numpy yields inf,
    # which is a truthful "this path exploded" rather than an exception.
    with np.errstate(over="ignore"):
        paths[:, 1:] = start_value * np.exp(np.cumsum(log_steps, axis=1))
    return paths


def _paths_from_simple_returns(start_value: float, steps: np.ndarray) -> np.ndarray:
    """Compound per-day SIMPLE returns into a value matrix with a start column.

    ``S_t = S_0 * prod_{i<=t} (1 + r_i)`` with the growth factor floored at
    zero: a long-only portfolio cannot be worth less than nothing, and a path
    that hits zero is absorbed there.
    """
    n_paths, horizon = steps.shape
    growth = np.clip(1.0 + steps, 0.0, None)
    paths = np.empty((n_paths, horizon + 1), dtype=float)
    paths[:, 0] = start_value
    with np.errstate(over="ignore"):
        paths[:, 1:] = start_value * np.cumprod(growth, axis=1)
    return paths


def _valid_shape(start_value: float, horizon_days: int, n_paths: int) -> bool:
    """True when a simulation of this size is meaningful at all."""
    return np.isfinite(start_value) and start_value > 0.0 and horizon_days >= 1 and n_paths >= 1


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def simulate_gbm(
    start_value: float,
    mu_daily: float,
    sigma_daily: float,
    horizon_days: int = SIMULATION.horizon_days,
    n_paths: int = SIMULATION.n_paths,
    seed: int | None = SIMULATION.seed,
) -> np.ndarray:
    """Geometric Brownian motion paths, shape ``(n_paths, horizon_days + 1)``.

    Formula (daily step, ``mu`` and ``sigma`` are DAILY simple-return moments)::

        S_{t+1} = S_t * exp((mu - sigma^2 / 2) + sigma * z),   z ~ N(0, 1)

    The ``-sigma^2 / 2`` term is the Ito / Jensen adjustment. Applying the
    arithmetic drift inside exp() without it would bias the simulated mean
    upward by ``exp(sigma^2 T / 2)``. With it the two central tendencies are::

        E[S_T]      = S_0 * exp(mu * T)
        median[S_T] = S_0 * exp((mu - sigma^2 / 2) * T)

    the difference being volatility drag - which is the whole reason a 50/50
    coin flip on +/-10% loses money.

    Assumes i.i.d. Normal log-returns: no volatility clustering, no
    autocorrelation, no regime change, and tails far thinner than real
    markets'. Returns an empty ``(0, horizon + 1)`` array when
    ``start_value`` is not a positive finite number, when ``sigma_daily`` is
    negative or non-finite, or when the requested shape is degenerate.
    """
    start = _as_float(start_value)
    mu = _as_float(mu_daily, 0.0)
    sigma = _as_float(sigma_daily)
    horizon = _as_int(horizon_days, 0)
    paths_n = _as_int(n_paths, 0)
    if not _valid_shape(start, horizon, paths_n):
        return _empty_paths(horizon)
    if not np.isfinite(sigma) or sigma < 0.0:
        return _empty_paths(horizon)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths_n, horizon))
    log_steps = (mu - 0.5 * sigma * sigma) + sigma * shocks
    return _paths_from_log_steps(start, log_steps)


def simulate_bootstrap(
    returns: pd.Series | np.ndarray | Any,
    start_value: float,
    horizon_days: int = SIMULATION.horizon_days,
    n_paths: int = SIMULATION.n_paths,
    seed: int | None = SIMULATION.seed,
    block_size: int = 1,
) -> np.ndarray:
    """Historical bootstrap paths, shape ``(n_paths, horizon_days + 1)``.

    ``block_size <= 1`` -> IID bootstrap: every simulated day is an independent
    draw with replacement from the observed daily returns. The empirical mean,
    variance, skew and kurtosis of a single day are reproduced exactly, so the
    fat left tail of the sample survives; the ordering does not, which means
    volatility clustering and any autocorrelation are destroyed and multi-day
    drawdowns come out too shallow.

    ``block_size = b > 1`` -> moving-block bootstrap: ``ceil(horizon / b)``
    contiguous blocks of ``b`` consecutive observed days are drawn with
    replacement (start index uniform on ``[0, n - b]``), concatenated and
    truncated to the horizon. Dependence up to ``b`` days survives inside each
    block and is broken at the joins, so the result sits between the IID
    bootstrap and the historical path itself. Blocks are non-circular, so the
    first and last ``b - 1`` observations appear in fewer blocks than interior
    ones and are correspondingly under-weighted.

    Compounds simple returns: ``S_t = S_0 * prod(1 + r_i)``, growth floored at
    zero. ``b`` is clamped to the sample length. Returns an empty
    ``(0, horizon + 1)`` array when there are no finite returns or the
    requested shape is degenerate.
    """
    sample = _clean_returns(returns)
    start = _as_float(start_value)
    horizon = _as_int(horizon_days, 0)
    paths_n = _as_int(n_paths, 0)
    if not _valid_shape(start, horizon, paths_n) or sample.size == 0:
        return _empty_paths(horizon)

    rng = np.random.default_rng(seed)
    block = _as_int(block_size, 1)
    block = min(max(block, 1), int(sample.size))

    if block <= 1:
        picks = rng.integers(0, sample.size, size=(paths_n, horizon))
        steps = sample[picks]
    else:
        n_blocks = int(np.ceil(horizon / block))
        max_start = int(sample.size - block)  # inclusive
        starts = rng.integers(0, max_start + 1, size=(paths_n, n_blocks))
        offsets = np.arange(block, dtype=np.int64)
        picks = (starts[:, :, None] + offsets[None, None, :]).reshape(paths_n, -1)
        steps = sample[picks[:, :horizon]]

    return _paths_from_simple_returns(start, steps)


def simulate_student_t(
    start_value: float,
    mu_daily: float,
    sigma_daily: float,
    df: float = DEFAULT_STUDENT_T_DF,
    horizon_days: int = SIMULATION.horizon_days,
    n_paths: int = SIMULATION.n_paths,
    seed: int | None = SIMULATION.seed,
) -> np.ndarray:
    """Fat-tailed GBM with Student-t innovations, ``(n_paths, horizon + 1)``.

    Formula (``mu``, ``sigma`` daily; ``t_df`` a Student-t variate)::

        z     = t_df / sqrt(df / (df - 2))          standardised to Var[z] = 1
        S_t+1 = S_t * exp((mu - sigma^2 / 2) + sigma * z)

    The standardisation matters: a raw t(df) has variance ``df / (df - 2)``, so
    feeding it in unscaled would silently inflate the volatility by 29% at
    df = 5. After scaling, ``sigma`` still means the daily volatility you
    passed, while the excess kurtosis ``6 / (df - 4)`` (df > 4) fattens both
    tails relative to GBM.

    Honest limits: the t is SYMMETRIC, and equity returns are left-skewed, so
    this model buys fat tails on the wrong side too. It remains i.i.d. and
    stationary - fat tails without volatility clustering. Requires
    ``df > MIN_STUDENT_T_DF`` (variance is infinite at df <= 2); anything
    lower, or a degenerate shape, returns an empty ``(0, horizon + 1)`` array.
    """
    start = _as_float(start_value)
    mu = _as_float(mu_daily, 0.0)
    sigma = _as_float(sigma_daily)
    nu = _as_float(df)
    horizon = _as_int(horizon_days, 0)
    paths_n = _as_int(n_paths, 0)
    if not _valid_shape(start, horizon, paths_n):
        return _empty_paths(horizon)
    if not np.isfinite(sigma) or sigma < 0.0:
        return _empty_paths(horizon)
    if not np.isfinite(nu) or nu <= MIN_STUDENT_T_DF:
        return _empty_paths(horizon)

    rng = np.random.default_rng(seed)
    raw = rng.standard_t(nu, size=(paths_n, horizon))
    shocks = raw / np.sqrt(nu / (nu - 2.0))
    log_steps = (mu - 0.5 * sigma * sigma) + sigma * shocks
    return _paths_from_log_steps(start, log_steps)


# ---------------------------------------------------------------------------
# summary statistics
# ---------------------------------------------------------------------------
def _nan_summary(percentiles: Any) -> dict[str, Any]:
    """The "no answer" summary: empty fan chart, NaN everywhere else."""
    labels = [label for _, label in _percentile_labels(percentiles)]
    frame = pd.DataFrame({label: pd.Series(dtype=float) for label in labels})
    frame.index.name = "day"
    return {
        "percentile_paths": frame,
        "terminal": np.empty(0, dtype=float),
        "mean_terminal": float("nan"),
        "median_terminal": float("nan"),
        "prob_loss": float("nan"),
        "prob_gain_10pct": float("nan"),
        "var_terminal": float("nan"),
        "cvar_terminal": float("nan"),
        "p5_terminal": float("nan"),
        "p95_terminal": float("nan"),
        "expected_return_pct": float("nan"),
        "worst_terminal": float("nan"),
        "best_terminal": float("nan"),
        "confidence": float("nan"),
        "n_paths": 0,
        "horizon_days": 0,
    }


def summarise_paths(
    paths: np.ndarray,
    start_value: float,
    percentiles: tuple[int, ...] = SIMULATION.percentiles,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> dict[str, Any]:
    """Collapse a path matrix into the numbers a tearsheet actually shows.

    Parameters
    ----------
    paths:
        ``(n_paths, horizon + 1)`` value matrix from any generator above.
    start_value:
        The value every path started from; all ratios are taken against it.
    percentiles:
        Percentiles of the CROSS-SECTION of paths, evaluated at every day, that
        make up the fan chart.
    confidence:
        Confidence level of the terminal VaR / CVaR (default from
        ``config.DEFAULT_VAR_CONFIDENCE``).

    Returns a dict with

    ``percentile_paths``
        DataFrame indexed by day 0..horizon, one column per percentile
        (``"p5"`` .. ``"p95"``). NOTE these are per-day quantiles, NOT paths:
        no single simulated portfolio ever traced the p5 line.
    ``terminal``
        1-D array of terminal values (last column).
    ``mean_terminal`` / ``median_terminal`` / ``worst_terminal`` / ``best_terminal``
        The mean sits above the median whenever the terminal distribution is
        right-skewed, which it always is under multiplicative compounding.
    ``prob_loss``
        Fraction of paths finishing BELOW ``start_value``; always in [0, 1].
    ``prob_gain_10pct``
        Fraction finishing at or above ``start_value * (1 + GAIN_THRESHOLD)``.
    ``var_terminal`` / ``cvar_terminal``
        HORIZON (not daily) loss as a negative decimal versus ``start_value``:
        the ``1 - confidence`` quantile of the terminal distribution, and the
        mean of the terminal values at or below it (expected shortfall).
    ``p5_terminal`` / ``p95_terminal``
        Terminal values at the 5th/95th percentile, in currency units.
    ``expected_return_pct``
        ``mean_terminal / start_value - 1`` in percent.

    Every value is ``float("nan")`` when the input cannot support it. Nothing
    raises, and the input array is never modified.
    """
    summary = _nan_summary(percentiles)
    start = _as_float(start_value)
    try:
        matrix = np.asarray(paths, dtype="float64")
    except (TypeError, ValueError):
        return summary
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return summary

    horizon = int(matrix.shape[1] - 1)
    summary["n_paths"] = int(matrix.shape[0])
    summary["horizon_days"] = horizon

    pairs = _percentile_labels(percentiles)
    levels = [level for level, _ in pairs]
    labels = [label for _, label in pairs]
    finite_cols = np.isfinite(matrix).all(axis=0)
    if finite_cols.all():
        grid = np.percentile(matrix, levels, axis=0)
    else:
        # A path that overflowed to inf must not poison every other day.
        grid = np.full((len(levels), matrix.shape[1]), np.nan, dtype=float)
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            values = values[np.isfinite(values)]
            if values.size:
                grid[:, column] = np.percentile(values, levels)
    fan = pd.DataFrame(grid.T, columns=labels, index=pd.RangeIndex(horizon + 1))
    fan.index.name = "day"
    summary["percentile_paths"] = fan

    terminal = np.array(matrix[:, -1], dtype=float)
    summary["terminal"] = terminal
    usable = terminal[np.isfinite(terminal)]
    if usable.size == 0:
        return summary

    summary["mean_terminal"] = float(np.mean(usable))
    summary["median_terminal"] = float(np.median(usable))
    summary["worst_terminal"] = float(np.min(usable))
    summary["best_terminal"] = float(np.max(usable))
    summary["p5_terminal"] = float(np.percentile(usable, TERMINAL_LOW_PCT))
    summary["p95_terminal"] = float(np.percentile(usable, TERMINAL_HIGH_PCT))

    if not np.isfinite(start) or start <= 0.0:
        return summary

    summary["prob_loss"] = float(np.mean(usable < start))
    summary["prob_gain_10pct"] = float(np.mean(usable >= start * (1.0 + GAIN_THRESHOLD)))
    summary["expected_return_pct"] = float(summary["mean_terminal"] / start - 1.0) * 100.0

    level = _as_float(confidence)
    if np.isfinite(level) and 0.0 < level < 1.0:
        summary["confidence"] = float(level)
        cutoff = float(np.percentile(usable, (1.0 - level) * 100.0))
        summary["var_terminal"] = float(cutoff / start - 1.0)
        tail = usable[usable <= cutoff]
        if tail.size:
            summary["cvar_terminal"] = float(np.mean(tail) / start - 1.0)
    return summary


# ---------------------------------------------------------------------------
# model wiring
# ---------------------------------------------------------------------------
def resolve_method(method: str | None) -> str:
    """Map a user/UI spelling onto one of :data:`METHODS`.

    Unknown spellings fall back to ``"bootstrap"`` - the method that assumes
    least about the return distribution.
    """
    if method is None:
        return "bootstrap"
    key = str(method).strip().lower().replace(" ", "_").replace("-", "_")
    return _METHOD_ALIASES.get(key, "bootstrap")


def _estimate_df(sample: np.ndarray) -> float:
    """Degrees of freedom implied by the sample kurtosis (method of moments).

    A Student-t with df > 4 has excess kurtosis ``k = 6 / (df - 4)``, which
    inverts to ``df = 4 + 6 / k``. Thin-tailed or too-short samples give no
    usable signal and fall back to :data:`DEFAULT_STUDENT_T_DF`; the result is
    clamped to ``(MIN_STUDENT_T_DF, MAX_STUDENT_T_DF]``.
    """
    if sample.size < 4:
        return DEFAULT_STUDENT_T_DF
    excess = _as_float(pd.Series(sample).kurt())
    if not np.isfinite(excess) or excess <= _MIN_KURTOSIS_FOR_DF:
        return MAX_STUDENT_T_DF
    estimate = 4.0 + 6.0 / excess
    return float(min(max(estimate, MIN_STUDENT_T_DF + 0.5), MAX_STUDENT_T_DF))


def _drift_standard_error_pct(sigma_daily: float, n_obs: int) -> float:
    """Standard error of the ANNUALISED drift estimate, in percent.

    ``se(mu_hat) = sigma / sqrt(n)`` per day, annualised by ``TRADING_DAYS``.
    Printed in the note because it is usually larger than the drift itself -
    the single most under-appreciated fact about return forecasting.
    """
    if not np.isfinite(sigma_daily) or sigma_daily <= 0.0 or n_obs < 2:
        return float("nan")
    return float(sigma_daily / np.sqrt(n_obs) * TRADING_DAYS * 100.0)


def _build_note(
    method: str,
    sample: np.ndarray,
    sigma_daily: float,
    block_size: int,
    df: float,
    horizon_days: int,
) -> str:
    """Plain-English statement of what this model assumes and where it breaks."""
    n_obs = int(sample.size)
    skew = _as_float(pd.Series(sample).skew()) if n_obs >= 3 else float("nan")
    excess = _as_float(pd.Series(sample).kurt()) if n_obs >= 4 else float("nan")
    worst = float(np.min(sample) * 100.0) if n_obs else float("nan")
    se_pct = _drift_standard_error_pct(sigma_daily, n_obs)

    if method == "gbm":
        head = (
            "Geometric Brownian motion: log-returns are i.i.d. Normal with the "
            f"daily drift and volatility estimated from {n_obs} observations. "
            "The -sigma^2/2 term makes the arithmetic mean of the simulated "
            "terminal values equal start * exp(mu * T); the median sits below "
            "it by the volatility drag."
        )
        weakness = (
            "Its central weakness is the Normal tail. The sample's excess "
            f"kurtosis is {excess:.2f} (0 for a Normal) and its worst single "
            f"day was {worst:.2f}%, so this model materially understates the "
            "chance of an extreme loss."
        )
    elif method == "bootstrap":
        head = (
            f"IID bootstrap: each simulated day is drawn with replacement from "
            f"the {n_obs} observed daily returns, so the empirical skew "
            f"({skew:.2f}) and excess kurtosis ({excess:.2f}) are reproduced "
            "exactly rather than assumed away."
        )
        weakness = (
            "What it destroys is the ORDER of returns: independent draws remove "
            "volatility clustering, so sustained drawdowns are understated. It "
            f"also cannot invent a day worse than the worst observed ({worst:.2f}%) "
            "- the historical sample is a hard floor on the simulated tail."
        )
    elif method == "block_bootstrap":
        head = (
            f"Moving-block bootstrap: contiguous {block_size}-day runs are "
            f"resampled from the {n_obs} observed returns, so dependence up to "
            f"{block_size} days - volatility clustering, short-run momentum - "
            "survives inside a block, along with the empirical skew "
            f"({skew:.2f}) and excess kurtosis ({excess:.2f})."
        )
        weakness = (
            "Dependence is still broken at every block join, non-circular blocks "
            f"under-weight the first and last {max(block_size - 1, 0)} observations, "
            f"and no day worse than the observed {worst:.2f}% can ever appear."
        )
    else:
        head = (
            f"Student-t innovations with df = {df:.1f}, standardised to unit "
            "variance so the daily volatility still means what it says. Excess "
            f"kurtosis of the innovation is {6.0 / (df - 4.0):.2f} where df > 4, "
            "which produces the extreme days a Normal model cannot."
            if df > 4.0
            else (
                f"Student-t innovations with df = {df:.1f}, standardised to unit "
                "variance; at this df the kurtosis is infinite, i.e. deliberately "
                "extreme tails."
            )
        )
        weakness = (
            "But the t is SYMMETRIC and equity returns are not (sample skew "
            f"{skew:.2f}), so it fattens the upside tail just as much as the "
            "crash tail, and it is still i.i.d. - fat tails with no volatility "
            "clustering."
        )

    shared = (
        f"All of it rests on {n_obs} days of history being a stationary, "
        "representative sample of the next "
        f"{horizon_days} trading days: no regime change, no rebalancing, no "
        "contributions, no costs or taxes."
    )
    if np.isfinite(se_pct):
        shared += (
            " The standard error of the annualised drift estimate alone is "
            f"+/-{se_pct:.1f}%, so treat the centre of the fan as noise and read "
            "the width."
        )
    return " ".join((head, weakness, shared))


def run_simulation(
    values: pd.Series | float | None,
    returns: pd.Series | np.ndarray | None,
    method: str = "bootstrap",
    cfg: SimulationConfig = SIMULATION,
    horizon_days: int | None = None,
    n_paths: int | None = None,
    seed: int | None = None,
    block_size: int | None = None,
    df: float | None = None,
) -> dict[str, Any]:
    """Run one forward simulation end to end and summarise it honestly.

    Parameters
    ----------
    values:
        Portfolio value Series (the last finite level is the starting value); a
        bare number is also accepted.
    returns:
        Daily simple returns used to estimate ``mu``/``sigma`` (GBM,
        Student-t) or to resample from (bootstraps). When empty, they are
        derived from ``values``.
    method:
        One of :data:`METHODS`, or any alias in ``_METHOD_ALIASES``
        (``"historical"``, ``"block"``, ``"t"``, ...). Unknown values fall back
        to the IID bootstrap.
    cfg:
        ``config.SIMULATION`` defaults for horizon, path count, seed, block
        size and fan-chart percentiles. The explicit arguments override it.
    df:
        Student-t degrees of freedom. ``None`` estimates it from the sample
        excess kurtosis via ``df = 4 + 6 / k``.

    Returns the ``summarise_paths`` dict plus ``ok, reason, method,
    horizon_days, horizon_years, n_paths, n_obs, mu_daily, sigma_daily,
    start_value, seed, block_size, df, paths, note``.

    ``mu_daily`` is the arithmetic mean of the sample returns and
    ``sigma_daily`` their sample standard deviation (ddof=1); both are used
    only by the parametric methods. Insufficient or unusable data returns
    ``{"ok": False, "reason": ...}`` rather than a plausible-looking fiction:
    fewer than ``config.MIN_OBSERVATIONS`` finite returns cannot support any of
    these models. Deterministic for a given seed; inputs are never modified.
    """
    settings = cfg if isinstance(cfg, SimulationConfig) else SIMULATION
    canonical = resolve_method(method)
    horizon = _as_int(horizon_days, settings.horizon_days)
    paths_n = _as_int(n_paths, settings.n_paths)
    resolved_seed = settings.seed if seed is None else seed

    failure: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "method": canonical,
        "horizon_days": horizon,
        "n_paths": paths_n,
        "n_obs": 0,
        "start_value": float("nan"),
        "seed": resolved_seed,
    }

    sample = _clean_returns(returns)
    start = _start_value_from(values)

    if not np.isfinite(start) or start <= 0.0:
        failure["reason"] = (
            "No positive starting portfolio value - cannot scale a simulation "
            "from an unknown or non-positive base."
        )
        return failure

    if sample.size == 0:
        # Fall back to deriving returns from the value series itself.
        levels = _clean_returns(values)
        if levels.size >= 2 and np.all(levels[:-1] > 0.0):
            sample = levels[1:] / levels[:-1] - 1.0
            sample = sample[np.isfinite(sample)]

    failure["n_obs"] = int(sample.size)
    failure["start_value"] = start

    if sample.size < MIN_OBSERVATIONS:
        failure["reason"] = (
            f"Only {sample.size} usable daily returns; "
            f"{MIN_OBSERVATIONS} are required before a simulated distribution "
            "means anything. Extend the history window."
        )
        return failure
    if horizon < 1 or paths_n < 1:
        failure["reason"] = (
            f"Degenerate simulation size (horizon {horizon} days, "
            f"{paths_n} paths); both must be at least 1."
        )
        return failure

    mu_daily = float(np.mean(sample))
    sigma_daily = float(np.std(sample, ddof=1)) if sample.size > 1 else float("nan")
    default_block = settings.block_size if canonical == "block_bootstrap" else 1
    block = _as_int(block_size, default_block)
    block = min(max(block, 1), int(sample.size))
    # The block size is what actually distinguishes the two bootstraps, so let
    # it have the last word on the label the result reports.
    if canonical in ("bootstrap", "block_bootstrap"):
        canonical = "block_bootstrap" if block > 1 else "bootstrap"
        failure["method"] = canonical
    nu = _as_float(df) if df is not None else _estimate_df(sample)

    if canonical == "gbm":
        paths = simulate_gbm(start, mu_daily, sigma_daily, horizon, paths_n, resolved_seed)
    elif canonical == "student_t":
        if not np.isfinite(nu) or nu <= MIN_STUDENT_T_DF:
            failure["reason"] = (
                f"Student-t degrees of freedom {nu} is not usable; variance is "
                f"infinite at or below {MIN_STUDENT_T_DF}."
            )
            return failure
        paths = simulate_student_t(
            start, mu_daily, sigma_daily, nu, horizon, paths_n, resolved_seed
        )
    else:
        paths = simulate_bootstrap(sample, start, horizon, paths_n, resolved_seed, block_size=block)

    if paths.shape[0] == 0:
        failure["reason"] = (
            "The generator could not produce paths from these parameters "
            f"(sigma={sigma_daily}, horizon={horizon}, paths={paths_n})."
        )
        return failure

    result: dict[str, Any] = {
        "ok": True,
        "reason": "",
        "method": canonical,
        "horizon_days": horizon,
        "horizon_years": float(horizon / TRADING_DAYS),
        "n_paths": int(paths.shape[0]),
        "n_obs": int(sample.size),
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "start_value": start,
        "seed": resolved_seed,
        "block_size": int(block) if canonical == "block_bootstrap" else 1,
        "df": float(nu) if canonical == "student_t" else float("nan"),
        "paths": paths,
    }
    result.update(summarise_paths(paths, start, percentiles=settings.percentiles))
    # summarise_paths reports the shape it actually saw; keep those values.
    result["note"] = _build_note(canonical, sample, sigma_daily, block, nu, horizon)
    if not np.isfinite(sigma_daily) or sigma_daily <= 0.0:
        result["note"] = (
            "WARNING: the sample has zero (or undefined) volatility, so every "
            "path is identical and no distribution is being explored. "
        ) + result["note"]
    return result
