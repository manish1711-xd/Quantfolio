"""Mean-variance optimisation: the efficient frontier and four ways onto it.

Markowitz in its practical, long-only form. Given a matrix of daily asset
returns this module estimates the two inputs every allocation argument really
turns on -- an expected-return vector ``mu`` and a covariance matrix ``Sigma``
-- and then solves a sequence of small constrained quadratic programmes with
``scipy.optimize.minimize(method="SLSQP")``:

    minimise    f(w)
    subject to  sum(w) = 1                    (fully invested)
                min_weight <= w_i <= max_weight   (long-only, per-name cap)

with ``f`` one of

    variance                w' Sigma w                     -> min_variance
    negative Sharpe        -(w'mu - rf) / sqrt(w' Sigma w)  -> max_sharpe
    risk-budget error       sum_i (rc_i - sigma_p^2/n)^2    -> risk_parity
    negative diversification ratio -(w'sigma)/sqrt(w'Sigma w) -> max_diversification

where ``rc_i = w_i * (Sigma w)_i`` is asset i's contribution to portfolio
variance (these sum to ``sigma_p^2`` exactly, which is what makes risk parity
well posed).

Conventions and guarantees
--------------------------
Annualisation   Both estimators are annualised with ``config.TRADING_DAYS``.
                ``mu`` is a geometric annualisation of realised daily
                compounding; ``Sigma`` is the sample covariance of daily
                returns scaled by ``TRADING_DAYS``.
Estimation risk The sample covariance is notoriously unstable when the number
                of assets approaches the number of observations, so it is
                shrunk toward a scaled identity (Ledoit-Wolf style target,
                intensity ``config.OPTIMIZER.shrinkage``). ``optimize_all``
                warns when ``n_obs < 2 * n_assets``.
Portfolio maths ``r_p = w . mu`` (the standard MPT convention: expected
                returns aggregate arithmetically across weights),
                ``sigma_p = sqrt(w' Sigma w)``, ``Sharpe = (r_p - rf)/sigma_p``.
Feasibility     A cap that cannot be met (``max_weight * n < 1``) is widened to
                ``1/n`` and the widening is recorded in
                ``OptimizedPortfolio.note`` -- never raised, never silently
                ignored. Same for an infeasible floor.
Robustness      Every optimiser is a best-of over several deterministic
                starting points (equal weight, inverse volatility, inverse
                variance, and objective-specific seeds), and the returned
                weights are the best FEASIBLE candidate seen -- including the
                starting points themselves. That is what makes the structural
                identities hold by construction rather than by luck:

                    Sharpe(max_sharpe)   >= Sharpe(equal_weight)
                    vol(min_variance)    <= vol(equal_weight)
                    vol(min_variance)    <= vol(any frontier point)
                    frontier volatility is non-decreasing in target return

                The frontier constrains ``w . mu >= target`` (an inequality, so
                the feasible sets are nested and the efficient branch is the
                only one reachable) rather than ``== target``.
Determinism     No random restarts. ``random_portfolios`` is seeded. The same
                inputs always produce the same weights.
Degenerate in   Empty frames, a single asset, a singular or zero covariance,
                all-NaN columns and non-finite expected returns are handled by
                returning NaN metrics / empty frames, never by raising.
Purity          Nothing here mutates its inputs; every frame is copied first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import MIN_OBSERVATIONS, OPTIMIZER, TRADING_DAYS
from src.models import OptimizedPortfolio

__all__ = [
    "SOLVER",
    "MAX_ITERATIONS",
    "SOLVER_FTOL",
    "WEIGHT_TOLERANCE",
    "COVARIANCE_OBSERVATION_RATIO",
    "EXPECTED_RETURN_METHODS",
    "DEFAULT_RANDOM_PORTFOLIOS",
    "DEFAULT_SEED",
    "FRONTIER_COLUMNS",
    "LABEL_MAX_SHARPE",
    "LABEL_MIN_VARIANCE",
    "LABEL_RISK_PARITY",
    "LABEL_MAX_DIVERSIFICATION",
    "LABEL_EQUAL_WEIGHT",
    "LABEL_CURRENT",
    "PORTFOLIO_LABELS",
    "expected_returns",
    "covariance",
    "portfolio_performance",
    "min_variance",
    "max_sharpe",
    "risk_parity",
    "max_diversification",
    "equal_weight",
    "target_return_portfolio",
    "efficient_frontier",
    "random_portfolios",
    "optimize_all",
]

# ---------------------------------------------------------------------------
# solver settings and module thresholds
# ---------------------------------------------------------------------------
#: Sequential Least SQuares Programming: the right tool here because the
#: problems are smooth, low dimensional (<= a few hundred names) and carry both
#: an equality constraint and box bounds.
SOLVER: str = "SLSQP"

MAX_ITERATIONS: int = 300

#: SLSQP stops when the objective improves by less than this. The objectives
#: below are all scaled to O(1) so this is effectively a relative tolerance.
SOLVER_FTOL: float = 1e-12

#: Constraint violation treated as numerical noise rather than infeasibility.
FEASIBILITY_TOL: float = 1e-9

#: Guaranteed accuracy of ``sum(weights) == 1`` in every returned portfolio.
WEIGHT_TOLERANCE: float = 1e-6

#: Weights below this are reported as exactly zero (they are solver dust; the
#: worst-case effect on the budget constraint is n * 1e-12, i.e. far inside
#: ``WEIGHT_TOLERANCE``).
WEIGHT_FLOOR: float = 1e-12

#: Variance floor used only to keep 1/sigma finite for a degenerate (zero
#: variance) portfolio. 1e-12 annualised variance is a 1e-6 annualised
#: volatility -- six orders of magnitude below anything real.
VARIANCE_FLOOR: float = 1e-12

#: ``n_obs < COVARIANCE_OBSERVATION_RATIO * n_assets`` -> the sample covariance
#: is close to singular and the optimiser will happily fit its own noise.
COVARIANCE_OBSERVATION_RATIO: float = 2.0

#: Supported ``expected_returns`` methods.
EXPECTED_RETURN_METHODS: frozenset[str] = frozenset({"historical", "mean", "capm"})

DEFAULT_RANDOM_PORTFOLIOS: int = 3000
DEFAULT_SEED: int = 42

#: Dirichlet concentration for the random cloud. alpha = 1 is the uniform
#: distribution over the long-only simplex.
DIRICHLET_ALPHA: float = 1.0

#: Hard ceiling on the random cloud so a UI slip cannot allocate gigabytes.
MAX_RANDOM_PORTFOLIOS: int = 200_000

#: Statistic columns of the frontier frame, in order, before the weight
#: columns (one per ticker).
FRONTIER_COLUMNS: tuple[str, str, str] = ("return", "volatility", "sharpe")

LABEL_MAX_SHARPE: str = "Max Sharpe"
LABEL_MIN_VARIANCE: str = "Min Variance"
LABEL_RISK_PARITY: str = "Risk Parity"
LABEL_MAX_DIVERSIFICATION: str = "Max Diversification"
LABEL_EQUAL_WEIGHT: str = "Equal Weight"
LABEL_CURRENT: str = "Current"

#: The optimised portfolios ``optimize_all`` always attempts, in display order.
PORTFOLIO_LABELS: tuple[str, ...] = (
    LABEL_MAX_SHARPE,
    LABEL_MIN_VARIANCE,
    LABEL_RISK_PARITY,
    LABEL_MAX_DIVERSIFICATION,
    LABEL_EQUAL_WEIGHT,
)

WeightsLike = Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None


# ---------------------------------------------------------------------------
# input hygiene
# ---------------------------------------------------------------------------
def _empty_frame() -> pd.DataFrame:
    """The typed "no answer" frame."""
    return pd.DataFrame(dtype=float)


def _empty_series(name: str) -> pd.Series:
    """The typed "no answer" series."""
    return pd.Series(dtype=float, name=name)


def _prepare_returns(asset_returns: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    """Numeric, finite, de-duplicated daily return columns.

    Coerces to float (non-numeric -> NaN), voids infinities and drops duplicate
    columns plus any column with fewer than two observations. Zero-variance
    columns are KEPT: a constant series is a legitimate (cash-like) asset and
    the optimiser has to cope with the singular covariance it produces. The
    caller's object is copied first and is never modified.
    """
    if asset_returns is None:
        return _empty_frame()
    try:
        frame = pd.DataFrame(asset_returns).copy()
    except (TypeError, ValueError):
        return _empty_frame()
    if frame.empty or frame.shape[1] == 0:
        return _empty_frame()
    try:
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    except (TypeError, ValueError):
        return _empty_frame()
    frame = frame.where(np.isfinite(frame))
    keep = [col for col in frame.columns if int(frame[col].notna().sum()) >= 2]
    return frame.loc[:, keep]


def _unusable_columns(
    asset_returns: pd.DataFrame | pd.Series | None, prepared: pd.DataFrame
) -> list[str]:
    """Columns the caller supplied that could not support an estimate.

    Named rather than dropped quietly: a holding that silently vanished from
    an optimisation is exactly the kind of thing a portfolio tool must say out
    loud.
    """
    if asset_returns is None:
        return []
    try:
        supplied = list(pd.DataFrame(asset_returns).columns)
    except (TypeError, ValueError):
        return []
    kept = set(prepared.columns)
    return sorted({str(col) for col in supplied if col not in kept})


def _complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows on which every asset traded.

    Estimating a covariance on complete cases costs a few observations but
    guarantees a positive semi-definite matrix; pairwise-complete estimation
    does not, and a non-PSD Sigma lets an optimiser manufacture a portfolio
    with negative variance.
    """
    if frame.empty:
        return frame
    return frame.dropna(axis=0, how="any")


def _as_series(values: WeightsLike, index: Sequence[Any], name: str) -> pd.Series:
    """Coerce dict / Series / array-like to a float Series aligned to ``index``.

    Unknown labels are dropped, missing labels come back as NaN, and anything
    uncoercible yields an all-NaN series. Never mutates ``values``.
    """
    idx = pd.Index(list(index))
    if values is None:
        return pd.Series(np.nan, index=idx, dtype=float, name=name)
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, Mapping):
        series = pd.Series(dict(values), dtype="object")
    else:
        try:
            array = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return pd.Series(np.nan, index=idx, dtype=float, name=name)
        if array.size != len(idx):
            return pd.Series(np.nan, index=idx, dtype=float, name=name)
        return pd.Series(array, index=idx, dtype=float, name=name)
    if series.index.has_duplicates:
        series = series[~series.index.duplicated()]
    series = pd.to_numeric(series, errors="coerce")
    series = series.reindex(idx).astype(float)
    series.name = name
    return series


def _prepare_covariance(cov: pd.DataFrame | None) -> tuple[list[Any], np.ndarray]:
    """Clean, square, symmetric covariance -> (labels, matrix).

    Keeps labels present in both the index and the columns, coerces to float,
    drops any label whose row or column is not fully finite (worst offender
    first, so one broken name does not delete the matrix) and averages away
    residual asymmetry. Unusable input gives ``([], empty array)``.
    """
    if cov is None:
        return [], np.zeros((0, 0), dtype=float)
    try:
        frame = pd.DataFrame(cov).copy()
    except (TypeError, ValueError):
        return [], np.zeros((0, 0), dtype=float)
    if frame.empty or frame.shape[1] == 0:
        return [], np.zeros((0, 0), dtype=float)

    frame = frame.loc[~frame.index.duplicated(), ~frame.columns.duplicated()]
    column_set = set(frame.columns)
    labels = [label for label in frame.index if label in column_set]
    if not labels:
        return [], np.zeros((0, 0), dtype=float)
    try:
        frame = frame.loc[labels, labels].apply(pd.to_numeric, errors="coerce")
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return [], np.zeros((0, 0), dtype=float)

    while values.shape[0] > 0:
        broken = ~np.isfinite(values)
        if not broken.any():
            break
        worst = int(np.argmax(broken.sum(axis=1)))
        keep = [i for i in range(values.shape[0]) if i != worst]
        labels = [labels[i] for i in keep]
        values = values[np.ix_(keep, keep)]
    if values.shape[0] == 0:
        return [], np.zeros((0, 0), dtype=float)

    values = (values + values.T) / 2.0
    return list(labels), values


def _prepare_problem(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Align an expected-return vector and a covariance matrix.

    The covariance decides the asset universe (it is the only input every
    optimiser needs); ``mu`` is reindexed onto it and may legitimately contain
    NaN -- ``min_variance`` and ``risk_parity`` do not use it, and the
    return-dependent optimisers drop those names explicitly via
    :func:`_drop_unpriced`.
    """
    labels, cov_arr = _prepare_covariance(cov)
    if not labels:
        return [], np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    mu_arr = _as_series(mu, labels, "expected_return").to_numpy(dtype=float)
    mu_arr = np.where(np.isfinite(mu_arr), mu_arr, np.nan)
    return labels, mu_arr, cov_arr


def _drop_unpriced(
    labels: list[Any], mu_arr: np.ndarray, cov_arr: np.ndarray
) -> tuple[list[Any], np.ndarray, np.ndarray, str]:
    """Remove assets with no usable expected return, reporting which.

    An asset whose realised growth was non-positive (a total wipeout) has no
    finite geometric expected return; carrying it into a return-driven
    objective would poison every candidate, so it is dropped and named in the
    note rather than being assigned an invented number.
    """
    finite = np.isfinite(mu_arr)
    if bool(finite.all()):
        return labels, mu_arr, cov_arr, ""
    keep = [i for i in range(len(labels)) if finite[i]]
    dropped = [str(labels[i]) for i in range(len(labels)) if not finite[i]]
    note = (
        "No finite expected return for " + ", ".join(dropped) + " - excluded from this objective."
    )
    if not keep:
        return [], np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float), note
    return (
        [labels[i] for i in keep],
        mu_arr[keep],
        cov_arr[np.ix_(keep, keep)],
        note,
    )


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------
def _resolve_bounds(
    n: int, max_weight: float | None, min_weight: float | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-name box bounds, widened until the budget constraint is reachable.

    Defaults come from ``config.OPTIMIZER``. A cap with ``max_weight * n < 1``
    makes ``sum(w) = 1`` unreachable, so it is widened to ``1/n`` (the tightest
    feasible cap) and the widening is returned as a note; a floor with
    ``min_weight * n > 1`` is dropped to zero the same way. Nothing raises.
    """
    notes: list[str] = []
    upper_value = OPTIMIZER.max_weight if max_weight is None else float(max_weight)
    lower_value = OPTIMIZER.min_weight if min_weight is None else float(min_weight)

    if not np.isfinite(upper_value) or upper_value <= 0.0:
        upper_value = 1.0
    upper_value = min(upper_value, 1.0)
    if not np.isfinite(lower_value) or lower_value < 0.0:
        lower_value = 0.0
    lower_value = min(lower_value, 1.0)

    if n <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), notes

    equal = 1.0 / n
    if lower_value > upper_value:
        notes.append(
            f"Floor of {lower_value:.1%} exceeds the cap of {upper_value:.1%}"
            " - floor dropped to 0%."
        )
        lower_value = 0.0
    if lower_value * n > 1.0 + 1e-12:
        notes.append(
            f"Floor of {lower_value:.1%} across {n} names needs more than 100%"
            " of the portfolio - floor dropped to 0%."
        )
        lower_value = 0.0
    if upper_value * n < 1.0 - 1e-12:
        notes.append(
            f"Cap of {upper_value:.1%} across {n} names cannot reach 100% - widened to {equal:.1%}."
        )
        upper_value = equal

    lower = np.full(n, lower_value, dtype=float)
    upper = np.full(n, upper_value, dtype=float)
    return lower, upper, notes


def _repair_weights(weights: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project a weight vector back onto ``{sum(w)=1, lower <= w <= upper}``.

    Water-filling: clip into the box, then spread the residual budget across
    the names that still have room, proportionally to that room. Converges in
    a handful of passes and is exact for the box-plus-budget geometry, which
    is what lets the caller rely on ``WEIGHT_TOLERANCE``.
    """
    n = int(lower.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    try:
        w = np.asarray(weights, dtype=float).ravel().copy()
    except (TypeError, ValueError):
        w = np.full(n, 1.0 / n, dtype=float)
    if w.size != n:
        w = np.full(n, 1.0 / n, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, lower, upper)

    for _ in range(64):
        gap = 1.0 - float(np.sum(w))
        if abs(gap) <= 1e-14:
            break
        room = (upper - w) if gap > 0.0 else (w - lower)
        total = float(np.sum(room))
        if total <= 1e-15:
            break
        w = np.clip(w + gap * room / total, lower, upper)
    return w


def _weights_dict(labels: Sequence[Any], weights: np.ndarray) -> dict[str, float]:
    """Ticker -> weight fraction, with solver dust snapped to exactly zero."""
    cleaned = np.where(np.abs(weights) < WEIGHT_FLOOR, 0.0, weights)
    return {str(label): float(value) for label, value in zip(labels, cleaned)}


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------
def expected_returns(
    asset_returns: pd.DataFrame | pd.Series | None, method: str = "historical"
) -> pd.Series:
    """Annualised expected return per asset.

    ``"historical"`` (default) compounds the realised daily path and
    annualises it geometrically::

        mu_i = prod(1 + r_it) ** (TRADING_DAYS / n_i) - 1

    which is what the asset actually delivered per year over the sample.
    ``"mean"`` is the arithmetic convention ``mean(r_i) * TRADING_DAYS``; it
    sits above the geometric figure by roughly ``sigma^2 / 2`` and is the
    consistent input when a mean-variance objective is read literally, so both
    are offered. ``"capm"`` is not implemented (it needs a market proxy and a
    risk-free curve this module is not given).

    An asset whose cumulative growth is non-positive (a total wipeout) has no
    real geometric annualisation and returns NaN rather than a fabricated
    number. Unknown methods raise ``ValueError`` (a caller bug, not degenerate
    data). Empty input returns an empty Series; the input is never mutated.
    """
    key = str(method).strip().lower()
    if key == "capm":
        raise NotImplementedError(
            "CAPM expected returns need a market proxy and a risk-free curve; "
            "use method='historical' or 'mean'."
        )
    if key not in EXPECTED_RETURN_METHODS:
        raise ValueError(
            f"Unknown expected-return method {method!r}; "
            f"expected one of {sorted(EXPECTED_RETURN_METHODS)}."
        )

    frame = _prepare_returns(asset_returns)
    if frame.empty:
        return _empty_series("expected_return")

    values: list[float] = []
    for col in frame.columns:
        series = frame[col].dropna()
        n = int(series.size)
        if n < 1:
            values.append(float("nan"))
            continue
        if key == "mean":
            values.append(float(series.mean()) * TRADING_DAYS)
            continue
        growth = float(np.prod(1.0 + series.to_numpy(dtype=float)))
        if not np.isfinite(growth) or growth <= 0.0:
            values.append(float("nan"))
            continue
        values.append(float(growth ** (TRADING_DAYS / n) - 1.0))

    result = pd.Series(values, index=frame.columns, dtype=float)
    result.name = "expected_return"
    return result


def covariance(
    asset_returns: pd.DataFrame | pd.Series | None, shrinkage: float | None = None
) -> pd.DataFrame:
    """Annualised covariance matrix with Ledoit-Wolf-style shrinkage.

    The sample covariance (``ddof=1``) is estimated on COMPLETE CASES -- dates
    on which every asset traded -- which guarantees a positive semi-definite
    matrix, then annualised by ``TRADING_DAYS`` (daily returns are treated as
    serially uncorrelated, the standard square-root-of-time assumption)::

        S = TRADING_DAYS * cov(r)
        Sigma = (1 - a) * S + a * (trace(S) / n) * I

    The shrinkage target is the scaled identity: same average variance, zero
    correlation. Intensity ``a`` defaults to ``config.OPTIMIZER.shrinkage`` and
    is clipped to [0, 1]. Shrinking is not cosmetic -- with n assets and T
    observations the sample matrix has rank min(n, T-1), so at ``T < n`` it is
    singular and the unshrunk optimiser will happily bet the portfolio on a
    spurious zero-variance combination.

    Returns an empty frame when there are fewer than two complete observations
    or no usable column. The input is never mutated.
    """
    frame = _complete_cases(_prepare_returns(asset_returns))
    if frame.empty or frame.shape[0] < 2 or frame.shape[1] == 0:
        return _empty_frame()

    intensity = OPTIMIZER.shrinkage if shrinkage is None else float(shrinkage)
    if not np.isfinite(intensity):
        intensity = OPTIMIZER.shrinkage
    intensity = float(min(max(intensity, 0.0), 1.0))

    labels = list(frame.columns)
    n = len(labels)
    sample = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=1)
    sample = np.atleast_2d(np.asarray(sample, dtype=float))
    if sample.shape != (n, n):
        return _empty_frame()
    sample = sample * float(TRADING_DAYS)
    sample = np.where(np.isfinite(sample), sample, 0.0)
    sample = (sample + sample.T) / 2.0

    target = float(np.trace(sample)) / n if n else 0.0
    if not np.isfinite(target):
        target = 0.0
    shrunk = (1.0 - intensity) * sample + intensity * target * np.eye(n)
    return pd.DataFrame(shrunk, index=pd.Index(labels), columns=pd.Index(labels))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _stats(
    weights: np.ndarray, mu_arr: np.ndarray, cov_arr: np.ndarray, risk_free_rate: float
) -> tuple[float, float, float]:
    """(return, volatility, Sharpe) for an already-aligned weight vector."""
    if weights.size == 0:
        return float("nan"), float("nan"), float("nan")

    ret = float(weights @ mu_arr) if np.all(np.isfinite(mu_arr)) else float("nan")

    variance = float(weights @ cov_arr @ weights)
    if np.isfinite(variance) and variance > 0.0:
        vol = math.sqrt(variance)
    elif np.isfinite(variance):
        vol = 0.0
    else:
        vol = float("nan")

    rf = float(risk_free_rate) if np.isfinite(risk_free_rate) else float("nan")
    if np.isfinite(ret) and np.isfinite(vol) and vol > 0.0 and np.isfinite(rf):
        sharpe = (ret - rf) / vol
    else:
        sharpe = float("nan")
    return ret, vol, float(sharpe)


def portfolio_performance(
    weights: WeightsLike,
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Score any weight vector on a given mu / Sigma: (return, vol, Sharpe).

    ``r_p = w . mu``, ``sigma_p = sqrt(w' Sigma w)``,
    ``Sharpe = (r_p - rf) / sigma_p``, all annualised decimals.

    Weights are aligned to the covariance labels, missing names are treated as
    zero weight, and the vector is normalised by its own sum -- so percent
    (sums to 100) and fraction (sums to 1) inputs both work. Returns
    ``(nan, nan, nan)`` when the universe is empty or the weights do not sum to
    anything positive, and a NaN Sharpe when volatility is zero (an infinite
    Sharpe is not a number a portfolio report should print). Nothing is
    mutated.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    if not labels:
        return float("nan"), float("nan"), float("nan")

    raw = _as_series(weights, labels, "weight").to_numpy(dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = float(np.sum(raw))
    if not np.isfinite(total) or abs(total) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    return _stats(raw / total, mu_arr, cov_arr, risk_free_rate)


# ---------------------------------------------------------------------------
# objectives (value and analytic gradient -- SLSQP is far faster and steadier
# with a real Jacobian than with finite differences)
# ---------------------------------------------------------------------------
def _variance_objective(
    cov_arr: np.ndarray,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """f(w) = w' Sigma w, grad = 2 Sigma w."""

    def value(w: np.ndarray) -> float:
        return float(w @ cov_arr @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_arr @ w)

    return value, grad


def _neg_sharpe_objective(
    mu_arr: np.ndarray, cov_arr: np.ndarray, risk_free_rate: float
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """f(w) = -(w'mu - rf) / sqrt(w' Sigma w).

    grad = -mu / sigma + (w'mu - rf) * Sigma w / sigma^3, with sigma floored at
    sqrt(VARIANCE_FLOOR) so a degenerate zero-variance portfolio produces a
    large but finite gradient instead of a NaN.
    """

    def value(w: np.ndarray) -> float:
        sigma = math.sqrt(max(float(w @ cov_arr @ w), VARIANCE_FLOOR))
        return -float(w @ mu_arr - risk_free_rate) / sigma

    def grad(w: np.ndarray) -> np.ndarray:
        sigma = math.sqrt(max(float(w @ cov_arr @ w), VARIANCE_FLOOR))
        excess = float(w @ mu_arr - risk_free_rate)
        return -mu_arr / sigma + excess * (cov_arr @ w) / (sigma**3)

    return value, grad


def _neg_diversification_objective(
    cov_arr: np.ndarray,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """f(w) = -(w . sigma) / sqrt(w' Sigma w), sigma = sqrt(diag(Sigma)).

    The diversification ratio is the weighted average stand-alone volatility
    divided by the realised portfolio volatility: 1.0 when every name is the
    same bet, higher the more the correlations work for you.
    """
    stand_alone = np.sqrt(np.clip(np.diag(cov_arr), 0.0, None))

    def value(w: np.ndarray) -> float:
        sigma = math.sqrt(max(float(w @ cov_arr @ w), VARIANCE_FLOOR))
        return -float(w @ stand_alone) / sigma

    def grad(w: np.ndarray) -> np.ndarray:
        sigma = math.sqrt(max(float(w @ cov_arr @ w), VARIANCE_FLOOR))
        weighted = float(w @ stand_alone)
        return -stand_alone / sigma + weighted * (cov_arr @ w) / (sigma**3)

    return value, grad


def _risk_budget_objective(
    cov_arr: np.ndarray, scale: float
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """f(w) = sum_i (rc_i - sigma_p^2 / n)^2 with rc_i = w_i (Sigma w)_i.

    ``scale`` is a positive constant (independent of w) that lifts the residual
    from variance-squared units to O(1) so SLSQP's ``ftol`` means what it says;
    scaling an objective by a constant does not move its argmin.

    Gradient: with d_i = rc_i - t and t = sigma_p^2/n, sum_i d_i = 0
    identically, so the derivative of the target cancels and

        grad_j = 2 * scale * [ d_j (Sigma w)_j + (Sigma (d * w))_j ].
    """
    n = int(cov_arr.shape[0])

    def value(w: np.ndarray) -> float:
        marginal = cov_arr @ w
        contribution = w * marginal
        target = float(np.sum(contribution)) / n
        residual = contribution - target
        return float(residual @ residual) * scale

    def grad(w: np.ndarray) -> np.ndarray:
        marginal = cov_arr @ w
        contribution = w * marginal
        target = float(np.sum(contribution)) / n
        residual = contribution - target
        return 2.0 * scale * (residual * marginal + cov_arr @ (residual * w))

    return value, grad


# ---------------------------------------------------------------------------
# solver core
# ---------------------------------------------------------------------------
def _default_starts(lower: np.ndarray, cov_arr: np.ndarray) -> list[np.ndarray]:
    """Deterministic seeds every optimiser starts from.

    Equal weight (always feasible once the bounds are resolved, and the
    benchmark every optimiser must beat), inverse volatility and inverse
    variance (the two risk-based allocations that sit near the min-variance /
    risk-parity solutions). Feeding the solver the answer's neighbourhood is
    what keeps a 30-asset problem in the millisecond range.
    """
    n = int(lower.size)
    starts = [np.full(n, 1.0 / n, dtype=float)]
    variances = np.clip(np.diag(cov_arr), 0.0, None)
    positive = variances > 0.0
    if bool(positive.all()):
        inverse_vol = 1.0 / np.sqrt(variances)
        starts.append(inverse_vol / float(np.sum(inverse_vol)))
        inverse_var = 1.0 / variances
        starts.append(inverse_var / float(np.sum(inverse_var)))
    elif bool(positive.any()):
        # Zero-variance names are cash-like: seed the whole budget into them.
        seed = np.where(positive, 0.0, 1.0)
        starts.append(seed / float(np.sum(seed)))
    return starts


def _violation(
    w: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    constraints: Sequence[dict[str, Any]],
) -> float:
    """Total constraint breach: budget + box + every inequality constraint."""
    total = abs(float(np.sum(w)) - 1.0)
    total += float(np.sum(np.clip(lower - w, 0.0, None)))
    total += float(np.sum(np.clip(w - upper, 0.0, None)))
    for constraint in constraints:
        if constraint.get("type") != "ineq":
            continue
        try:
            values = np.atleast_1d(np.asarray(constraint["fun"](w), dtype=float))
        except (TypeError, ValueError):
            return float("inf")
        total += float(np.sum(np.clip(-values, 0.0, None)))
    return total


def _solve(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray] | None,
    lower: np.ndarray,
    upper: np.ndarray,
    starts: Sequence[np.ndarray],
    extra_constraints: Sequence[dict[str, Any]] = (),
    stop_on_success: bool = False,
) -> tuple[np.ndarray, bool]:
    """Best-of SLSQP over several starting points.

    Candidates are ranked lexicographically by ``(constraint violation beyond
    FEASIBILITY_TOL, objective, 1 if the candidate is only a starting point)``,
    so a feasible point always beats an infeasible one, ties go to a genuine
    solve, and -- crucially -- the starting points themselves compete. A
    solver that fails to improve on equal weight therefore returns equal
    weight, which is why ``max_sharpe`` can never be worse than
    ``equal_weight``.

    ``stop_on_success`` stops at the first start that converges to a feasible
    point instead of trying them all. It is used for the frontier sweep, where
    each point is warm-started from its neighbour and the later starts are
    only a fallback -- 60 points times three starts is the difference between
    a two-second and a half-second frontier.

    Returns ``(weights, converged)``; ``converged`` is False when the winning
    candidate did not come from a successful SLSQP run.
    """
    n = int(lower.size)
    if n == 0:
        return np.zeros(0, dtype=float), False
    if n == 1:
        return np.ones(1, dtype=float), True

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda _w: np.ones(n, dtype=float),
        }
    ]
    constraints.extend(extra_constraints)
    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]

    def score(w: np.ndarray, solved: bool) -> tuple[float, float, int]:
        breach = max(0.0, _violation(w, lower, upper, constraints) - FEASIBILITY_TOL)
        try:
            value = float(objective(w))
        except (TypeError, ValueError, FloatingPointError):
            value = float("inf")
        if not np.isfinite(value):
            value = float("inf")
        return breach, value, 0 if solved else 1

    best_w = _repair_weights(starts[0] if starts else np.zeros(n), lower, upper)
    best_key = score(best_w, solved=False)
    best_converged = False

    for start in starts:
        w0 = _repair_weights(start, lower, upper)
        key = score(w0, solved=False)
        if key < best_key:
            best_key, best_w, best_converged = key, w0, False
        try:
            result = minimize(
                objective,
                w0,
                jac=gradient,
                method=SOLVER,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": MAX_ITERATIONS, "ftol": SOLVER_FTOL},
            )
        except (ValueError, TypeError, np.linalg.LinAlgError):
            continue
        candidate = np.asarray(getattr(result, "x", np.zeros(n)), dtype=float)
        if candidate.size != n or not np.all(np.isfinite(candidate)):
            continue
        candidate = _repair_weights(candidate, lower, upper)
        key = score(candidate, solved=True)
        solved_ok = bool(getattr(result, "success", False))
        if key < best_key:
            best_key, best_w = key, candidate
            best_converged = solved_ok
        if stop_on_success and solved_ok and best_converged and best_key[0] <= 0.0:
            break

    return best_w, best_converged


def _build(
    label: str,
    labels: Sequence[Any],
    weights: np.ndarray,
    mu_arr: np.ndarray,
    cov_arr: np.ndarray,
    risk_free_rate: float,
    converged: bool,
    notes: Sequence[str],
) -> OptimizedPortfolio:
    """Package a solved weight vector as an :class:`OptimizedPortfolio`."""
    ret, vol, sharpe = _stats(weights, mu_arr, cov_arr, risk_free_rate)
    note = " ".join(part for part in notes if part).strip()
    return OptimizedPortfolio(
        label=label,
        weights=_weights_dict(labels, weights),
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=bool(converged),
        note=note,
    )


def _degenerate(label: str, reason: str) -> OptimizedPortfolio:
    """The "there is nothing to optimise" result: empty weights, NaN metrics."""
    return OptimizedPortfolio(
        label=label,
        weights={},
        expected_return=float("nan"),
        volatility=float("nan"),
        sharpe=float("nan"),
        converged=False,
        note=reason,
    )


# ---------------------------------------------------------------------------
# named optimisers
# ---------------------------------------------------------------------------
def min_variance(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
    min_weight: float | None = None,
) -> OptimizedPortfolio:
    """Global minimum-variance portfolio: minimise ``w' Sigma w``.

    Long-only, fully invested, ``min_weight <= w_i <= max_weight``. The one
    portfolio on the frontier that needs no expected-return forecast, which is
    why it is the most robust point on it -- ``mu`` is used only to score the
    result. Its volatility is, by construction, no higher than equal weight's
    or any point on :func:`efficient_frontier`.

    Returns an unconverged empty portfolio when the covariance is unusable.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    n = len(labels)
    if n == 0:
        return _degenerate(LABEL_MIN_VARIANCE, "No usable covariance matrix.")

    lower, upper, notes = _resolve_bounds(n, max_weight, min_weight)
    value, grad = _variance_objective(cov_arr)
    weights, converged = _solve(value, grad, lower, upper, _default_starts(lower, cov_arr))
    return _build(
        LABEL_MIN_VARIANCE,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        converged,
        notes,
    )


def max_sharpe(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
    min_weight: float | None = None,
) -> OptimizedPortfolio:
    """Tangency portfolio: maximise ``(w'mu - rf) / sqrt(w' Sigma w)``.

    The point where the capital market line touches the efficient frontier --
    the best reward per unit of risk available from this universe under the
    given bounds. Solved as a minimisation of the negative Sharpe ratio with
    an analytic gradient, seeded from equal weight, the risk-based
    allocations, and a return-tilted start.

    Because equal weight is one of the candidates the solver is scored
    against, the returned Sharpe is never below the equal-weight Sharpe.
    Assets with no finite expected return are excluded and named in ``note``.
    Returns an unconverged empty portfolio when nothing is left to optimise.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    if not labels:
        return _degenerate(LABEL_MAX_SHARPE, "No usable covariance matrix.")
    labels, mu_arr, cov_arr, dropped = _drop_unpriced(labels, mu_arr, cov_arr)
    n = len(labels)
    if n == 0:
        return _degenerate(LABEL_MAX_SHARPE, dropped or "No usable expected returns.")

    lower, upper, notes = _resolve_bounds(n, max_weight, min_weight)
    starts = _default_starts(lower, cov_arr)
    excess = mu_arr - float(risk_free_rate)
    tilt = np.clip(excess, 0.0, None)
    if float(np.sum(tilt)) > 0.0:
        starts.append(tilt / float(np.sum(tilt)))

    value, grad = _neg_sharpe_objective(mu_arr, cov_arr, float(risk_free_rate))
    weights, converged = _solve(value, grad, lower, upper, starts)
    return _build(
        LABEL_MAX_SHARPE,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        converged,
        [dropped, *notes],
    )


def _risk_parity_fixed_point(
    cov_arr: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray | None:
    """Cyclical fixed point for the equal-risk-contribution portfolio.

    At an ERC solution ``w_i (Sigma w)_i`` is the same for every i, i.e.
    ``w_i is proportional to 1 / (Sigma w)_i``. Iterating that relation and
    renormalising converges monotonically for a positive-definite Sigma and is
    both faster and far more accurate than a least-squares solve. Iteration
    count and tolerance come from ``config.OPTIMIZER``.

    Returns None when the iteration cannot run (a name with zero marginal
    risk) or when the fixed point violates the caller's box bounds -- the
    SLSQP formulation then takes over.
    """
    int(cov_arr.shape[0])
    variances = np.clip(np.diag(cov_arr), 0.0, None)
    if not bool((variances > 0.0).all()):
        return None
    w = 1.0 / np.sqrt(variances)
    w = w / float(np.sum(w))

    # DAMPED update. The undamped map w <- normalise(1/(Sigma w)) has a
    # marginally unstable period-2 mode on a near-diagonal Sigma: it oscillates
    # with growing amplitude until a marginal risk goes negative, the iteration
    # bails, and the caller falls back to SLSQP while reporting a diagnostic
    # whose stated causes are both false. Averaging each step with the current
    # point kills that mode and leaves the fixed point itself unchanged, since
    # w = (1-d)*w + d*w holds at any solution.
    damping = 0.5
    for _ in range(int(OPTIMIZER.risk_parity_iterations)):
        marginal = cov_arr @ w
        if not np.all(np.isfinite(marginal)) or float(np.min(marginal)) <= 0.0:
            return None
        candidate = 1.0 / marginal
        candidate = candidate / float(np.sum(candidate))
        updated = (1.0 - damping) * w + damping * candidate
        updated = updated / float(np.sum(updated))
        shift = float(np.max(np.abs(updated - w)))
        w = updated
        if shift <= float(OPTIMIZER.risk_parity_tolerance):
            break

    if float(np.min(w - lower)) < -FEASIBILITY_TOL:
        return None
    if float(np.max(w - upper)) > FEASIBILITY_TOL:
        return None
    return _repair_weights(w, lower, upper)


def risk_parity(
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    mu: pd.Series | Mapping[str, float] | None = None,
    max_weight: float | None = None,
) -> OptimizedPortfolio:
    """Equal-risk-contribution portfolio: every name adds the same risk.

    Asset i's contribution to portfolio variance is ``rc_i = w_i (Sigma w)_i``
    and these sum to ``sigma_p^2`` exactly. Risk parity minimises the squared
    dispersion of those contributions around their equal budget::

        minimise sum_i (rc_i - sigma_p^2 / n)^2,  sum(w) = 1,  0 <= w_i <= cap

    Solved with the ERC fixed point ``w_i proportional to 1/(Sigma w)_i`` when
    the cap does not bind (exact to ``OPTIMIZER.risk_parity_tolerance``), and
    with SLSQP on the least-squares residual otherwise -- best of both is
    returned. Needs no expected returns at all; ``mu`` is optional and used
    only to report the resulting return and Sharpe (NaN when it is None).
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    n = len(labels)
    if n == 0:
        return _degenerate(LABEL_RISK_PARITY, "No usable covariance matrix.")

    lower, upper, notes = _resolve_bounds(n, max_weight, OPTIMIZER.min_weight)
    average_variance = float(np.mean(np.clip(np.diag(cov_arr), 0.0, None)))
    scale = 1.0 / max(average_variance**2, 1e-24)
    value, grad = _risk_budget_objective(cov_arr, scale)

    starts = _default_starts(lower, cov_arr)
    fixed_point = _risk_parity_fixed_point(cov_arr, lower, upper)
    extra_note = ""
    if fixed_point is not None:
        starts.insert(0, fixed_point)
        if value(fixed_point) <= SOLVER_FTOL:
            # Exact ERC already: no cap binds, so SLSQP has nothing to add.
            return _build(
                LABEL_RISK_PARITY,
                labels,
                fixed_point,
                mu_arr,
                cov_arr,
                risk_free_rate,
                True,
                notes,
            )
    else:
        extra_note = (
            "Equal-risk fixed point unavailable (a name carries no marginal "
            "risk or the cap binds); solved as a least-squares risk budget."
        )

    weights, converged = _solve(value, grad, lower, upper, starts)
    return _build(
        LABEL_RISK_PARITY,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        converged,
        [extra_note, *notes],
    )


def max_diversification(
    cov: pd.DataFrame | None,
    mu: pd.Series | Mapping[str, float] | None = None,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
) -> OptimizedPortfolio:
    """Most-diversified portfolio: maximise ``(w . sigma) / sqrt(w' Sigma w)``.

    The diversification ratio compares the volatility you would carry if the
    names were perfectly correlated (the weighted average of stand-alone
    volatilities) with the volatility you actually carry. It is 1.0 for a
    single bet and rises as correlations fall, so maximising it buys
    correlation structure rather than forecast returns. ``mu`` is optional and
    used only for scoring. Long-only, fully invested, capped per name.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    n = len(labels)
    if n == 0:
        return _degenerate(LABEL_MAX_DIVERSIFICATION, "No usable covariance matrix.")

    lower, upper, notes = _resolve_bounds(n, max_weight, OPTIMIZER.min_weight)
    starts = _default_starts(lower, cov_arr)
    value, grad = _neg_diversification_objective(cov_arr)
    weights, converged = _solve(value, grad, lower, upper, starts)
    return _build(
        LABEL_MAX_DIVERSIFICATION,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        converged,
        notes,
    )


def equal_weight(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
) -> OptimizedPortfolio:
    """The 1/N portfolio -- the benchmark every optimiser has to beat.

    ``w_i = 1/n`` for every asset in the covariance matrix. No estimation, no
    solver, no estimation error: DeMiguel, Garlappi and Uppal (2009) showed
    1/N is hard to beat out of sample precisely because optimised weights
    inherit the noise in ``mu``. Included here so the UI can always show the
    optimiser's edge over doing nothing clever.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    n = len(labels)
    if n == 0:
        return _degenerate(LABEL_EQUAL_WEIGHT, "No usable covariance matrix.")
    weights = np.full(n, 1.0 / n, dtype=float)
    return _build(
        LABEL_EQUAL_WEIGHT,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        True,
        [],
    )


# ---------------------------------------------------------------------------
# the frontier
# ---------------------------------------------------------------------------
def _achievable_returns(
    mu_arr: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    """(lowest, highest) portfolio return reachable inside the box.

    A continuous-knapsack argument: with a linear objective, one budget
    equality and box bounds, the optimum starts everyone at their floor and
    pours the remaining budget into the best (or worst) names in order until
    they hit their caps. Exact, and it keeps the frontier sweep from asking
    for targets no long-only capped portfolio could ever deliver.
    """
    if mu_arr.size == 0 or not np.all(np.isfinite(mu_arr)):
        return float("nan"), float("nan")

    def fill(order: np.ndarray) -> float:
        w = lower.copy()
        remaining = 1.0 - float(np.sum(w))
        for i in order:
            if remaining <= 1e-15:
                break
            room = float(upper[i] - w[i])
            add = min(room, remaining)
            if add > 0.0:
                w[i] += add
                remaining -= add
        return float(w @ mu_arr)

    best = fill(np.argsort(-mu_arr, kind="stable"))
    worst = fill(np.argsort(mu_arr, kind="stable"))
    return worst, best


def target_return_portfolio(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    target: float,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
) -> OptimizedPortfolio:
    """Cheapest risk for at least ``target`` annualised return.

        minimise w' Sigma w   s.t.   w . mu >= target,  sum(w) = 1,  bounds

    The constraint is an INEQUALITY on purpose. With ``>=`` the feasible sets
    are nested as the target rises, so the solution is always on the efficient
    (upper) branch and its volatility is monotone in the target; with ``==``
    the solver can land on the inefficient lower branch and the frontier comes
    out non-monotone. Below the minimum-variance return the constraint is
    simply slack and you get the minimum-variance portfolio.

    A target above the highest return the bounds allow is infeasible; it is
    clipped to that maximum and the clipping is recorded in ``note`` rather
    than returning garbage.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    if not labels:
        return _degenerate("Target Return", "No usable covariance matrix.")
    labels, mu_arr, cov_arr, dropped = _drop_unpriced(labels, mu_arr, cov_arr)
    n = len(labels)
    if n == 0:
        return _degenerate("Target Return", dropped or "No usable expected returns.")

    lower, upper, notes = _resolve_bounds(n, max_weight, OPTIMIZER.min_weight)
    try:
        wanted = float(target)
    except (TypeError, ValueError):
        wanted = float("nan")
    if not np.isfinite(wanted):
        return _degenerate("Target Return", "Target return is not a number.")

    _, highest = _achievable_returns(mu_arr, lower, upper)
    clip_note = ""
    if np.isfinite(highest) and wanted > highest:
        clip_note = (
            f"Target of {wanted:.2%} is above the {highest:.2%} maximum these "
            "names can deliver under the weight cap - clipped to the maximum."
        )
        wanted = highest

    label = f"Target {wanted:.2%}"
    value, grad = _variance_objective(cov_arr)
    weights, converged = _solve(
        value,
        grad,
        lower,
        upper,
        _default_starts(lower, cov_arr),
        extra_constraints=[_return_constraint(mu_arr, wanted)],
    )
    return _build(
        label,
        labels,
        weights,
        mu_arr,
        cov_arr,
        risk_free_rate,
        converged,
        [clip_note, dropped, *notes],
    )


def _return_constraint(mu_arr: np.ndarray, target: float) -> dict[str, Any]:
    """SLSQP inequality ``w . mu - target >= 0`` with its analytic Jacobian."""
    return {
        "type": "ineq",
        "fun": lambda w: float(w @ mu_arr - target),
        "jac": lambda _w: mu_arr.copy(),
    }


def efficient_frontier(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    n_points: int | None = None,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """The efficient frontier as a frame: return, volatility, sharpe + weights.

    The sweep runs from the minimum-variance portfolio's return (the frontier's
    left-hand anchor, which is row 0 exactly) to the highest return the weight
    cap allows -- not to ``mu.max()``, which a capped long-only portfolio
    generally cannot reach and which would only produce infeasible points.
    Each step is a :func:`target_return_portfolio` solve with ``w . mu >=
    target``, warm-started from the previous solution.

    Two structural guarantees, both enforced explicitly:

    * no point sits left of the minimum-variance portfolio (points that come
      back below it are numerical duplicates of the anchor and are dropped);
    * volatility is non-decreasing along the frame. Where a solve underperforms
      a later one, the later -- strictly better, and feasible for the earlier
      target -- solution replaces it.

    ``n_points`` defaults to ``config.OPTIMIZER.frontier_points``. Columns are
    ``return, volatility, sharpe`` followed by one weight column per ticker (a
    ticker colliding with a statistic name is suffixed ``_w``). Returns an
    empty frame when the universe is unusable.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    if not labels:
        return _empty_frame()
    labels, mu_arr, cov_arr, _ = _drop_unpriced(labels, mu_arr, cov_arr)
    n = len(labels)
    if n == 0:
        return _empty_frame()

    points = OPTIMIZER.frontier_points if n_points is None else int(n_points)
    points = max(2, min(int(points), 500))

    lower, upper, _ = _resolve_bounds(n, max_weight, OPTIMIZER.min_weight)
    anchor = min_variance(
        pd.Series(mu_arr, index=pd.Index(labels)),
        pd.DataFrame(cov_arr, index=pd.Index(labels), columns=pd.Index(labels)),
        risk_free_rate,
        max_weight=max_weight,
        min_weight=OPTIMIZER.min_weight,
    )
    anchor_w = np.array([anchor.weights.get(str(label), 0.0) for label in labels])
    anchor_w = _repair_weights(anchor_w, lower, upper)
    anchor_return, anchor_vol, anchor_sharpe = _stats(anchor_w, mu_arr, cov_arr, risk_free_rate)

    _, highest = _achievable_returns(mu_arr, lower, upper)
    if not np.isfinite(anchor_return) or not np.isfinite(highest):
        return _empty_frame()
    highest = max(highest, anchor_return)

    value, grad = _variance_objective(cov_arr)
    rows: list[tuple[float, float, float, np.ndarray]] = [
        (anchor_return, anchor_vol, anchor_sharpe, anchor_w)
    ]
    previous = anchor_w
    # A single asset, or a universe whose names all share one expected return,
    # has no span to sweep: the frontier is the anchor and nothing else.
    span = highest - anchor_return
    targets = (
        np.linspace(anchor_return, highest, points)[1:]
        if span > abs(anchor_return) * 1e-12 + 1e-15
        else np.zeros(0, dtype=float)
    )
    for target in targets:
        starts = [previous, anchor_w, np.full(n, 1.0 / n, dtype=float)]
        weights, _ = _solve(
            value,
            grad,
            lower,
            upper,
            starts,
            extra_constraints=[_return_constraint(mu_arr, float(target))],
            stop_on_success=True,
        )
        achieved = float(weights @ mu_arr)
        if achieved < float(target) - 1e-6:
            continue  # infeasible under these bounds - drop the point
        ret, vol, sharpe = _stats(weights, mu_arr, cov_arr, risk_free_rate)
        if not np.isfinite(vol) or vol < anchor_vol:
            continue  # numerically indistinguishable from the anchor
        rows.append((ret, vol, sharpe, weights))
        previous = weights

    # Backward sweep: a later point with LOWER volatility is feasible for the
    # earlier (smaller) target and strictly better there, so it replaces it.
    for i in range(len(rows) - 2, 0, -1):
        if rows[i][1] > rows[i + 1][1] + 1e-12:
            rows[i] = rows[i + 1]

    stats = pd.DataFrame([(row[0], row[1], row[2]) for row in rows], columns=list(FRONTIER_COLUMNS))
    weight_matrix = np.vstack([row[3] for row in rows])
    columns = [f"{label}_w" if str(label) in FRONTIER_COLUMNS else str(label) for label in labels]
    weights_frame = pd.DataFrame(weight_matrix, columns=columns)
    frontier = pd.concat([stats, weights_frame], axis=1)
    frontier.attrs["tickers"] = columns
    frontier.attrs["min_variance_volatility"] = anchor_vol
    return frontier.reset_index(drop=True)


def random_portfolios(
    mu: pd.Series | Mapping[str, float] | None,
    cov: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    n: int = DEFAULT_RANDOM_PORTFOLIOS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """A cloud of random long-only portfolios: ``return, volatility, sharpe``.

    Weights are drawn from a Dirichlet with ``alpha = 1``, i.e. uniformly over
    the fully-invested long-only simplex (no weight cap is applied -- the cloud
    is a backdrop that shows how much of the risk/return plane is reachable at
    all, and the frontier is drawn over it). Deterministic for a given
    ``seed``. Returns an empty frame with the right columns when the universe
    is unusable or ``n <= 0``.
    """
    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    labels, mu_arr, cov_arr, _ = _drop_unpriced(labels, mu_arr, cov_arr)
    columns = list(FRONTIER_COLUMNS)
    try:
        count = int(n)
    except (TypeError, ValueError):
        count = 0
    count = max(0, min(count, MAX_RANDOM_PORTFOLIOS))
    if not labels or count == 0:
        return pd.DataFrame(columns=columns, dtype=float)

    rng = np.random.default_rng(int(seed) if np.isfinite(seed) else DEFAULT_SEED)
    weights = rng.dirichlet(np.full(len(labels), DIRICHLET_ALPHA), size=count)
    returns = weights @ mu_arr
    variances = np.einsum("ij,jk,ik->i", weights, cov_arr, weights)
    vols = np.sqrt(np.clip(variances, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(vols > 0.0, (returns - float(risk_free_rate)) / vols, np.nan)
    return pd.DataFrame({"return": returns, "volatility": vols, "sharpe": sharpe}, columns=columns)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _current_portfolio(
    current_weights: WeightsLike,
    labels: Sequence[Any],
    mu_arr: np.ndarray,
    cov_arr: np.ndarray,
    risk_free_rate: float,
) -> tuple[OptimizedPortfolio | None, list[str]]:
    """Score the portfolio the investor actually holds on the same mu/Sigma.

    Weights are normalised by their sum (percent or fraction both work) and
    restricted to names that have return history, so the current dot can be
    plotted on the same frontier as the optimised ones. Returns
    ``(None, warnings)`` when nothing scoreable was supplied.
    """
    warnings: list[str] = []
    if current_weights is None:
        return None, warnings
    if isinstance(current_weights, Mapping) and not current_weights:
        return None, warnings

    known = {str(label) for label in labels}
    if isinstance(current_weights, Mapping):
        missing = [str(k) for k in current_weights if str(k) not in known]
        if missing:
            warnings.append(
                "Current portfolio holds "
                + ", ".join(sorted(missing))
                + " with no usable price history - excluded from the comparison."
            )
    raw = _as_series(current_weights, labels, "weight").to_numpy(dtype=float)
    values = np.where(np.isfinite(raw), raw, 0.0)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        warnings.append("Current portfolio weights do not sum to anything positive.")
        return None, warnings

    weights = values / total
    ret, vol, sharpe = _stats(weights, mu_arr, cov_arr, risk_free_rate)
    covered = float(np.sum(values))
    note = "Weights normalised to 100% over the names with price history."
    if abs(covered - 1.0) > 1e-6 and abs(covered - 100.0) > 1e-4:
        note += f" Supplied weights summed to {covered:.4g}."
    return (
        OptimizedPortfolio(
            label=LABEL_CURRENT,
            weights=_weights_dict(labels, weights),
            expected_return=ret,
            volatility=vol,
            sharpe=sharpe,
            converged=True,
            note=note,
        ),
        warnings,
    )


def optimize_all(
    asset_returns: pd.DataFrame | None,
    risk_free_rate: float = 0.0,
    current_weights: WeightsLike = None,
    max_weight: float | None = None,
) -> dict[str, Any]:
    """Run the whole optimisation suite on one panel of daily asset returns.

    Estimates ``mu`` and ``Sigma`` ONCE, on the complete-case rows, so every
    portfolio in the result is scored on identical inputs and the frontier
    chart is internally consistent. Then solves the four objectives plus equal
    weight, builds the frontier and the random cloud, and scores the caller's
    current weights on the same inputs.

    Returns::

        {"mu": Series, "cov": DataFrame, "frontier": DataFrame,
         "random": DataFrame, "portfolios": {label: OptimizedPortfolio},
         "current": OptimizedPortfolio | None, "n_assets": int,
         "warnings": list[str]}

    ``warnings`` is where the honesty lives: too few observations for the
    covariance to be meaningful (``n_obs < 2 * n_assets``), a widened weight
    cap, names dropped for want of history, and any optimiser that did not
    converge. Degenerate input yields empty structures and a reason, never an
    exception.
    """
    warnings: list[str] = []
    prepared = _prepare_returns(asset_returns)
    clean = _complete_cases(prepared)
    n_obs = int(clean.shape[0])
    n_assets = int(clean.shape[1])

    dropped = _unusable_columns(asset_returns, prepared)
    if dropped:
        warnings.append(
            "No usable return history for "
            + ", ".join(dropped)
            + " - excluded from the optimisation."
        )

    empty: dict[str, Any] = {
        "mu": _empty_series("expected_return"),
        "cov": _empty_frame(),
        "frontier": _empty_frame(),
        "random": pd.DataFrame(columns=list(FRONTIER_COLUMNS), dtype=float),
        "portfolios": {},
        "current": None,
        "n_assets": n_assets,
        "warnings": warnings,
    }
    if n_assets == 0 or n_obs < 2:
        warnings.append(
            "Not enough overlapping return history to estimate a covariance "
            "matrix - optimisation skipped."
        )
        return empty

    if n_obs < MIN_OBSERVATIONS:
        warnings.append(
            f"Only {n_obs} overlapping observations (below the "
            f"{MIN_OBSERVATIONS} minimum): every estimate here is noisy."
        )
    if n_obs < COVARIANCE_OBSERVATION_RATIO * n_assets:
        warnings.append(
            f"{n_obs} observations for {n_assets} assets is below the "
            f"{COVARIANCE_OBSERVATION_RATIO:.0f}x rule of thumb - the sample "
            "covariance is unstable and the optimiser will fit its own noise. "
            f"Shrinkage of {OPTIMIZER.shrinkage:.0%} is applied but cannot "
            "manufacture data."
        )

    mu = expected_returns(clean, method="historical")
    cov = covariance(clean)
    if cov.empty:
        warnings.append("Covariance matrix could not be estimated.")
        return empty
    mu = mu.reindex(cov.columns)

    labels, mu_arr, cov_arr = _prepare_problem(mu, cov)
    rf = float(risk_free_rate) if np.isfinite(risk_free_rate) else 0.0

    portfolios: dict[str, OptimizedPortfolio] = {
        LABEL_MAX_SHARPE: max_sharpe(mu, cov, rf, max_weight=max_weight),
        LABEL_MIN_VARIANCE: min_variance(mu, cov, rf, max_weight=max_weight),
        LABEL_RISK_PARITY: risk_parity(cov, rf, mu=mu, max_weight=max_weight),
        LABEL_MAX_DIVERSIFICATION: max_diversification(
            cov, mu=mu, risk_free_rate=rf, max_weight=max_weight
        ),
        LABEL_EQUAL_WEIGHT: equal_weight(mu, cov, rf),
    }

    seen: set[str] = set()
    for label, portfolio in portfolios.items():
        if portfolio.note and portfolio.note not in seen:
            seen.add(portfolio.note)
            warnings.append(portfolio.note)
        if not portfolio.converged and portfolio.weights:
            warnings.append(
                f"{label}: the solver did not fully converge; the best "
                "feasible weights found are shown."
            )

    current, current_warnings = _current_portfolio(current_weights, labels, mu_arr, cov_arr, rf)
    warnings.extend(current_warnings)

    return {
        "mu": mu,
        "cov": cov,
        "frontier": efficient_frontier(mu, cov, rf, max_weight=max_weight),
        "random": random_portfolios(mu, cov, rf),
        "portfolios": portfolios,
        "current": current,
        "n_assets": n_assets,
        "warnings": warnings,
    }
