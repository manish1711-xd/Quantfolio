"""Multi-factor (Fama-French style) regression of portfolio excess returns.

What this answers
-----------------
A single-factor beta says how much of a portfolio moves with the market. It
cannot say *why* the rest moves. This module regresses portfolio excess return
on several explanatory factors at once::

    r_p - r_f  =  alpha  +  b_mkt * MKT  +  b_smb * SMB  +  b_hml * HML  +  ...  +  e

and reports each loading with a standard error, so the honest question -- "is
this exposure actually distinguishable from zero?" -- has an answer. The
intercept is the part of the return the supplied factors do not explain.

Relationship to the sibling modules
-----------------------------------
* :mod:`src.analytics.factors` decomposes the return *covariance* with PCA and
  clustering: unsupervised, no external data, "how many independent bets is
  this?". This module is supervised: it needs factor return series and answers
  "how much of the return does each NAMED factor explain?".
* :mod:`src.analytics.benchmark` computes a CAPM Jensen alpha from ANNUALISED
  returns. The alpha here is a regression INTERCEPT annualised afterwards.
  The two are different estimators of related quantities and will not agree to
  the last basis point; neither is wrong. See :func:`regress` for the
  annualisation convention used here.
* A one-factor regression's beta is *exactly* ``risk.beta`` (both are
  cov/var), which the test suite asserts as a cross-module identity.

Statistical care taken, and why each piece is not optional
----------------------------------------------------------
* **Newey-West (HAC) standard errors by default.** Return residuals are
  heteroskedastic, and both factors and residuals are often serially
  correlated. Plain OLS standard errors assume neither, so they are too small,
  t-statistics come out too large, and noise is reported as "significant"
  factor exposure. On persistent data the error is not marginal -- a t-stat of
  16 becomes 8 once the correction is applied. This is the single most common
  way a factor regression misleads its reader.
* **SVD least squares**, not the normal equations. Factor series are highly
  correlated (value and size proxies especially), and ``(X'X)^-1`` on a
  near-singular design is numerically unstable.
* **Variance inflation factors are reported.** With correlated factors the
  individual loadings become unstable even while R^2 stays high. A user reading
  betas without seeing the collinearity is being misled by omission.
* **Rank deficiency is reported, not crashed on.** Two identical factors is a
  user error the UI should explain.
* **Undefined is never zero.** Too few observations, a degenerate factor, a
  singular design -- all yield NaN and a stated reason.

What this module does NOT do
----------------------------
It does not supply factor data. Real Fama-French factors come from Kenneth
French's data library (US) or the IIM Ahmedabad dataset (India), and the
honest ways to get them into this application are to load such a file
(:func:`parse_factor_csv`) or to build clearly-labelled *proxies* from
tradeable ETF returns (:func:`build_proxy_factors`). A proxy is not the
academic factor and this module never calls it one.

Pure module: numpy/pandas/scipy only. No I/O, no network, no Streamlit.
"""

from __future__ import annotations

import io
import logging
import math
import warnings as warnings_module
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MIN_OBSERVATIONS, TRADING_DAYS, FactorSpec

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_FACTOR_OBSERVATIONS",
    "DEFAULT_SIGNIFICANCE",
    "VIF_WARN",
    "FactorLoading",
    "RegressionResult",
    "newey_west_lags",
    "newey_west_cov",
    "variance_inflation_factors",
    "to_period_rate",
    "prepare",
    "regress",
    "factor_regression",
    "rolling_betas",
    "factor_contributions",
    "build_proxy_factors",
    "parse_factor_csv",
]

#: Below this many overlapping observations a multi-factor regression is not
#: reported at all. With k factors the design needs meaningfully more than k+1
#: points before the loadings mean anything, and 30 is already generous for a
#: five-factor model.
MIN_FACTOR_OBSERVATIONS: int = MIN_OBSERVATIONS

#: Two-sided significance level used by the ``significant`` flags.
DEFAULT_SIGNIFICANCE: float = 0.05

#: Variance inflation factor above which collinearity is flagged. 5 is the
#: conventional caution line, 10 the alarm line; the UI shows the number so the
#: reader can apply their own judgement.
VIF_WARN: float = 5.0

_NAN = float("nan")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _clean_series(data: object) -> pd.Series:
    """A float Series on a sorted, unique, non-null index. Empty on junk."""
    if data is None:
        return pd.Series(dtype="float64")
    try:
        series = pd.Series(data).astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype="float64")
    series = series.dropna()
    if series.empty:
        return series
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def _clean_frame(data: object) -> pd.DataFrame:
    """A float DataFrame on a sorted, unique index, all-NaN columns dropped."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except (TypeError, ValueError):
            return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()

    frame = data.copy()
    frame.columns = [str(c) for c in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def to_period_rate(annual_rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    """De-annualise a rate GEOMETRICALLY: ``(1 + r) ** (1 / P) - 1``.

    Matches :mod:`src.analytics.risk`, which de-annualises the minimum
    acceptable return the same way. Using ``r / P`` instead would be a
    different convention, and two conventions inside one application is how
    Sharpe ratios stop reconciling.
    """
    rate = _to_float(annual_rate)
    if not math.isfinite(rate) or periods_per_year <= 0:
        return _NAN
    if rate <= -1.0:
        return _NAN
    return float((1.0 + rate) ** (1.0 / periods_per_year) - 1.0)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorLoading:
    """One factor's estimated exposure, with its uncertainty.

    ``beta`` without ``std_error`` is a number without a claim attached, which
    is why they are one object: a loading of 0.30 means something entirely
    different at a standard error of 0.03 than at 0.30.
    """

    name: str
    beta: float
    std_error: float = _NAN
    t_stat: float = _NAN
    p_value: float = _NAN
    vif: float = _NAN

    def significant(self, level: float = DEFAULT_SIGNIFICANCE) -> bool:
        """True when the loading is distinguishable from zero at ``level``."""
        return bool(math.isfinite(self.p_value) and self.p_value < level)

    @property
    def collinear(self) -> bool:
        """True when this factor's VIF is above the caution line."""
        return bool(math.isfinite(self.vif) and self.vif >= VIF_WARN)

    def confidence_interval(self, level: float = 0.95, dof: int = 0) -> tuple[float, float]:
        """Two-sided interval for the loading. ``(nan, nan)`` when undefined."""
        if not math.isfinite(self.beta) or not math.isfinite(self.std_error) or dof <= 0:
            return (_NAN, _NAN)
        critical = float(stats.t.ppf(0.5 + level / 2.0, dof))
        return (self.beta - critical * self.std_error, self.beta + critical * self.std_error)


@dataclass(frozen=True)
class RegressionResult:
    """Everything one factor regression produced.

    ``ok`` is False when the regression could not be estimated at all; every
    numeric field is then NaN and ``warnings`` says why. Callers must check it
    rather than reading a zero as a finding.
    """

    loadings: tuple[FactorLoading, ...] = ()
    alpha: float = _NAN
    alpha_annual: float = _NAN
    alpha_std_error: float = _NAN
    alpha_t_stat: float = _NAN
    alpha_p_value: float = _NAN
    r_squared: float = _NAN
    adj_r_squared: float = _NAN
    f_stat: float = _NAN
    f_p_value: float = _NAN
    residual_vol_annual: float = _NAN
    durbin_watson: float = _NAN
    n_observations: int = 0
    n_factors: int = 0
    dof: int = 0
    se_method: str = ""
    lags: int = 0
    periods_per_year: int = TRADING_DAYS
    start: object = None
    end: object = None
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # -- status ----------------------------------------------------------
    @property
    def ok(self) -> bool:
        return bool(self.loadings) and math.isfinite(self.r_squared)

    @property
    def reliable(self) -> bool:
        """Enough observations for the loadings to carry meaning."""
        return self.ok and self.n_observations >= MIN_FACTOR_OBSERVATIONS

    @property
    def explained_pct(self) -> float:
        """Share of return variation the factors account for, in percent."""
        return self.r_squared * 100.0 if math.isfinite(self.r_squared) else _NAN

    @property
    def unexplained_pct(self) -> float:
        """Idiosyncratic share -- what the factors do NOT account for."""
        return (1.0 - self.r_squared) * 100.0 if math.isfinite(self.r_squared) else _NAN

    def alpha_significant(self, level: float = DEFAULT_SIGNIFICANCE) -> bool:
        return bool(math.isfinite(self.alpha_p_value) and self.alpha_p_value < level)

    def loading(self, name: str) -> FactorLoading | None:
        return next((load for load in self.loadings if load.name == name), None)

    def betas(self) -> dict[str, float]:
        return {load.name: load.beta for load in self.loadings}

    @property
    def collinear_factors(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.loadings if load.collinear)

    # -- presentation ----------------------------------------------------
    def summary_frame(self) -> pd.DataFrame:
        """One row per term, alpha first, for direct display."""
        rows: list[dict[str, object]] = [
            {
                "Term": "Alpha (annualised)",
                "Estimate": self.alpha_annual,
                "Std error": self.alpha_std_error * self.periods_per_year
                if math.isfinite(self.alpha_std_error)
                else _NAN,
                "t-stat": self.alpha_t_stat,
                "p-value": self.alpha_p_value,
                "VIF": _NAN,
            }
        ]
        rows.extend(
            {
                "Term": load.name,
                "Estimate": load.beta,
                "Std error": load.std_error,
                "t-stat": load.t_stat,
                "p-value": load.p_value,
                "VIF": load.vif,
            }
            for load in self.loadings
        )
        return pd.DataFrame(
            rows, columns=["Term", "Estimate", "Std error", "t-stat", "p-value", "VIF"]
        )


# ---------------------------------------------------------------------------
# standard errors
# ---------------------------------------------------------------------------
def newey_west_lags(n_observations: int) -> int:
    """Newey-West truncation lag by the usual rule ``floor(4 * (n/100)^(2/9))``.

    A data-driven default rather than a magic constant: it grows slowly with
    the sample, which is what the theory requires for consistency. Returns 0
    for a sample too small to estimate any autocovariance.
    """
    if n_observations < 4:
        return 0
    return int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(design: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """Heteroskedasticity- and autocorrelation-consistent covariance of the
    OLS coefficients, using the Bartlett (Newey-West) kernel.

        Var(b) = (X'X)^-1 [ S0 + sum_l w_l (G_l + G_l') ] (X'X)^-1 * n/(n-k)

    where ``S0 = sum_t u_t^2 x_t x_t'``, ``G_l = sum_t (u_t x_t)(u_{t-l} x_{t-l})'``
    and ``w_l = 1 - l/(lags+1)``.

    Why this and not OLS standard errors: with ``lags=0`` this reduces to
    White's HC1 estimator, and with homoskedastic iid residuals HC1 in turn
    agrees with the classical OLS formula -- both of which the test suite
    pins. So nothing is lost by using it by default.

    What it corrects, stated precisely: the autocorrelation of the SCORE
    ``u_t * x_t``, not of the residual alone. When the regressor is itself
    serially uncorrelated the two are nearly the same thing and HAC barely
    moves the standard error; when both the factor and the residual are
    persistent -- the realistic case, and the case overlapping or
    slow-moving factor series produce -- the correction is large. On an AR(1)
    regressor and AR(1) residual at rho = 0.85 it roughly DOUBLES the standard
    error, taking a t-statistic of 16 down to 8. An OLS standard error there
    is not slightly optimistic; it is wrong by a factor of two.

    The ``n/(n-k)`` factor is the standard small-sample correction.
    """
    n, k = design.shape
    if n <= k:
        return np.full((k, k), _NAN)

    gram_inv = np.linalg.pinv(design.T @ design)
    weighted = design * residuals.reshape(-1, 1)  # rows are u_t * x_t

    sandwich = weighted.T @ weighted
    for lag in range(1, max(int(lags), 0) + 1):
        if lag >= n:
            break
        weight = 1.0 - lag / (lags + 1.0)
        gamma = weighted[lag:].T @ weighted[:-lag]
        sandwich = sandwich + weight * (gamma + gamma.T)

    correction = n / float(n - k)
    return gram_inv @ sandwich @ gram_inv * correction


def variance_inflation_factors(factors: pd.DataFrame) -> dict[str, float]:
    """VIF per factor: ``1 / (1 - R^2)`` from regressing it on the others.

    Reported because correlated factors make individual loadings unstable
    while R^2 stays high -- a size proxy and a value proxy built from
    overlapping ETFs can easily sit above 5. Showing a beta without this is
    misleading by omission.

    A single factor has no collinearity and returns 1.0. A perfectly collinear
    factor returns ``inf``.
    """
    frame = _clean_frame(factors).dropna()
    if frame.empty:
        return {}
    if frame.shape[1] < 2:
        return {str(column): 1.0 for column in frame.columns}

    out: dict[str, float] = {}
    for column in frame.columns:
        target = frame[column].to_numpy(dtype="float64")
        others = frame.drop(columns=[column]).to_numpy(dtype="float64")
        design = np.column_stack([np.ones(len(others)), others])
        try:
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            out[str(column)] = _NAN
            continue
        fitted = design @ coef
        total = float(np.sum((target - target.mean()) ** 2))
        if total <= 0.0:
            out[str(column)] = _NAN
            continue
        r2 = 1.0 - float(np.sum((target - fitted) ** 2)) / total
        out[str(column)] = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
    return out


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def prepare(
    portfolio_returns: object,
    factors: object,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    market_factor: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Align returns and factors onto their common dates. ``(y, X, warnings)``.

    ``y`` is the portfolio EXCESS return (``r_p`` minus the per-period
    risk-free rate). Regressing raw returns on excess-return factors would
    load the risk-free rate into the intercept and quietly inflate alpha, so
    the subtraction is not optional.

    Inner-joins on the index for the same reason
    :func:`src.analytics.risk.beta` does: a portfolio on the NSE calendar and
    a factor series on the NYSE calendar share far fewer dates than either
    has, and regressing across the mismatch measures the holiday schedule.

    ``market_factor`` names a column that is ALREADY an excess return (the
    classic ``Mkt-RF``); it is passed through untouched and only recorded in
    the warnings if absent, since nothing here can verify the claim.
    """
    warnings: list[str] = []

    returns = _clean_series(portfolio_returns)
    frame = _clean_frame(factors)
    if returns.empty:
        return pd.Series(dtype="float64"), pd.DataFrame(), ["No portfolio returns supplied."]
    if frame.empty:
        return pd.Series(dtype="float64"), pd.DataFrame(), ["No factor returns supplied."]

    joined = pd.concat([returns.rename("__y__"), frame], axis=1, join="inner").dropna()
    if joined.empty:
        return (
            pd.Series(dtype="float64"),
            pd.DataFrame(),
            [
                "The portfolio and the factor series share no dates. Check that both "
                "are daily (or both monthly) and cover the same window."
            ],
        )

    overlap_loss = len(returns) - len(joined)
    if overlap_loss > 0:
        warnings.append(
            f"{len(joined)} of {len(returns)} portfolio observations overlap the factor "
            f"data; the regression uses the overlap only."
        )

    period_rf = to_period_rate(risk_free_rate, periods_per_year)
    if not math.isfinite(period_rf):
        period_rf = 0.0
        warnings.append("Risk-free rate was unusable; excess returns computed against zero.")

    y = (joined["__y__"] - period_rf).rename("excess_return")
    design = joined.drop(columns="__y__")

    if market_factor and market_factor not in design.columns:
        warnings.append(
            f"'{market_factor}' was named as the market excess-return factor but is not "
            "among the supplied columns."
        )

    degenerate = [str(c) for c in design.columns if float(design[c].std(ddof=1) or 0.0) == 0.0]
    if degenerate:
        design = design.drop(columns=degenerate)
        warnings.append(
            f"Factor(s) {', '.join(degenerate)} never vary over this window and were "
            "dropped — a constant factor carries no information."
        )

    return y, design, warnings


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------
def regress(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    robust: bool = True,
    lags: int | None = None,
    periods_per_year: int = TRADING_DAYS,
    extra_warnings: Sequence[str] = (),
) -> RegressionResult:
    """OLS of ``excess_returns`` on ``factors`` with an intercept.

    ``robust=True`` (the default) uses Newey-West HAC standard errors; set it
    False for the classical OLS formula, which assumes residuals that daily
    return data does not have.

    **Alpha annualisation.** ``alpha`` is the per-period intercept;
    ``alpha_annual`` is ``alpha * periods_per_year``. Linear scaling is the
    convention in the factor literature and keeps the alpha t-statistic
    meaningful (a t-stat is scale-invariant, so annualising the estimate and
    its standard error by the same factor leaves it unchanged). Note this
    differs from :mod:`src.analytics.benchmark`, whose Jensen alpha is built
    from compounded annualised returns; both are defensible and they will not
    agree exactly.

    Never raises. An unusable design returns ``ok=False`` with the reason.
    """
    warnings = list(extra_warnings)

    y_series = _clean_series(excess_returns)
    x_frame = _clean_frame(factors)
    if y_series.empty or x_frame.empty:
        return RegressionResult(
            warnings=(*warnings, "Nothing to regress: empty returns or factors."),
            periods_per_year=periods_per_year,
        )

    joined = pd.concat([y_series.rename("__y__"), x_frame], axis=1, join="inner").dropna()
    names = [str(c) for c in joined.columns if c != "__y__"]
    n = int(len(joined))
    k = len(names)

    if k == 0:
        return RegressionResult(
            warnings=(*warnings, "No usable factors remain."),
            periods_per_year=periods_per_year,
        )
    if n <= k + 1:
        return RegressionResult(
            n_observations=n,
            n_factors=k,
            warnings=(
                *warnings,
                f"{n} observations cannot support {k} factor(s) plus an intercept — "
                f"at least {k + 2} are needed for any estimate, and "
                f"{MIN_FACTOR_OBSERVATIONS} before it means anything.",
            ),
            periods_per_year=periods_per_year,
        )

    y = joined["__y__"].to_numpy(dtype="float64")
    raw = joined[names].to_numpy(dtype="float64")
    design = np.column_stack([np.ones(n), raw])

    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return RegressionResult(
            n_observations=n,
            n_factors=k,
            warnings=(
                *warnings,
                "The factor design is rank-deficient: at least two factors are exact "
                "linear combinations of each other, so their separate loadings are not "
                "identified. Remove the duplicate factor.",
            ),
            periods_per_year=periods_per_year,
        )

    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        return RegressionResult(
            n_observations=n,
            n_factors=k,
            warnings=(*warnings, f"The regression did not converge ({exc})."),
            periods_per_year=periods_per_year,
        )

    fitted = design @ coefficients
    residuals = y - fitted
    dof = n - k - 1

    total_ss = float(np.sum((y - y.mean()) ** 2))
    resid_ss = float(np.sum(residuals**2))
    r_squared = 1.0 - resid_ss / total_ss if total_ss > 0.0 else _NAN
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / dof if math.isfinite(r_squared) and dof > 0 else _NAN
    )

    if robust:
        chosen_lags = newey_west_lags(n) if lags is None else max(int(lags), 0)
        cov = newey_west_cov(design, residuals, chosen_lags)
        se_method = f"Newey-West HAC, {chosen_lags} lag(s)"
    else:
        chosen_lags = 0
        sigma2 = resid_ss / dof if dof > 0 else _NAN
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se_method = "OLS (assumes iid homoskedastic residuals)"

    variances = np.diag(cov)
    std_errors = np.sqrt(np.where(variances > 0.0, variances, np.nan))

    def _t_and_p(estimate: float, error: float) -> tuple[float, float]:
        if not math.isfinite(error) or error <= 0.0 or dof <= 0:
            return (_NAN, _NAN)
        t_stat = estimate / error
        return (float(t_stat), float(2.0 * stats.t.sf(abs(t_stat), dof)))

    vifs = variance_inflation_factors(joined[names])
    alpha_t, alpha_p = _t_and_p(float(coefficients[0]), float(std_errors[0]))

    loadings = []
    for index, name in enumerate(names, start=1):
        beta = float(coefficients[index])
        error = float(std_errors[index])
        t_stat, p_value = _t_and_p(beta, error)
        loadings.append(
            FactorLoading(
                name=name,
                beta=beta,
                std_error=error,
                t_stat=t_stat,
                p_value=p_value,
                vif=_to_float(vifs.get(name, _NAN)),
            )
        )

    # Joint significance of the factors (the intercept is not under test).
    if math.isfinite(r_squared) and 0.0 <= r_squared < 1.0 and dof > 0 and k > 0:
        f_stat = (r_squared / k) / ((1.0 - r_squared) / dof)
        f_p_value = float(stats.f.sf(f_stat, k, dof))
    else:
        f_stat, f_p_value = _NAN, _NAN

    residual_vol = float(np.std(residuals, ddof=1) * math.sqrt(periods_per_year)) if n > 1 else _NAN

    if n > 1:
        diffs = np.diff(residuals)
        durbin_watson = float(np.sum(diffs**2) / resid_ss) if resid_ss > 0.0 else _NAN
    else:  # pragma: no cover - n > k + 1 already excludes this
        durbin_watson = _NAN

    if n < MIN_FACTOR_OBSERVATIONS:
        warnings.append(
            f"Only {n} overlapping observations — the loadings are indicative at best "
            f"({MIN_FACTOR_OBSERVATIONS}+ recommended)."
        )
    collinear = [name for name, value in vifs.items() if math.isfinite(value) and value >= VIF_WARN]
    if collinear:
        warnings.append(
            f"Factor(s) {', '.join(sorted(collinear))} have a variance inflation factor "
            f"above {VIF_WARN:g}: they overlap heavily, so their INDIVIDUAL loadings are "
            "unstable even though the model's overall fit is not."
        )

    return RegressionResult(
        loadings=tuple(loadings),
        alpha=float(coefficients[0]),
        alpha_annual=float(coefficients[0]) * periods_per_year,
        alpha_std_error=float(std_errors[0]),
        alpha_t_stat=alpha_t,
        alpha_p_value=alpha_p,
        r_squared=_to_float(r_squared),
        adj_r_squared=_to_float(adj_r_squared),
        f_stat=_to_float(f_stat),
        f_p_value=_to_float(f_p_value),
        residual_vol_annual=residual_vol,
        durbin_watson=durbin_watson,
        n_observations=n,
        n_factors=k,
        dof=dof,
        se_method=se_method,
        lags=chosen_lags,
        periods_per_year=periods_per_year,
        start=joined.index[0],
        end=joined.index[-1],
        warnings=tuple(dict.fromkeys(warnings)),
    )


def factor_regression(
    portfolio_returns: object,
    factors: object,
    risk_free_rate: float = 0.0,
    robust: bool = True,
    lags: int | None = None,
    periods_per_year: int = TRADING_DAYS,
    market_factor: str | None = None,
) -> RegressionResult:
    """Align, then regress. The one call a page needs."""
    y, design, warnings = prepare(
        portfolio_returns,
        factors,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        market_factor=market_factor,
    )
    if y.empty or design.empty:
        return RegressionResult(warnings=tuple(warnings), periods_per_year=periods_per_year)
    return regress(
        y,
        design,
        robust=robust,
        lags=lags,
        periods_per_year=periods_per_year,
        extra_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# how the exposures move
# ---------------------------------------------------------------------------
def rolling_betas(
    portfolio_returns: object,
    factors: object,
    window: int = 126,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Factor loadings re-estimated over a rolling window.

    Often the most useful output on the page: a single full-sample loading
    averages away the fact that exposures DRIFT, and a portfolio whose value
    loading has quietly gone from +0.4 to -0.2 is a different portfolio from
    the one the full-sample number describes.

    Deliberately reports only the point estimates and ``alpha``/``r_squared``
    -- not standard errors. Overlapping windows make a strip of t-statistics
    look far more informative than it is, and the honest use of this frame is
    to look at the SHAPE of the exposure over time.

    Returns an empty frame when the window cannot be filled.
    """
    y, design, _warnings = prepare(
        portfolio_returns,
        factors,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    if y.empty or design.empty:
        return pd.DataFrame()

    window = max(int(window), 2)
    needed = window if min_periods is None else max(int(min_periods), 2)
    names = [str(c) for c in design.columns]
    if len(y) < needed or len(y) <= len(names) + 1:
        return pd.DataFrame()

    joined = pd.concat([y.rename("__y__"), design], axis=1, join="inner").dropna()
    y_values = joined["__y__"].to_numpy(dtype="float64")
    x_values = joined[names].to_numpy(dtype="float64")

    rows: list[dict[str, float]] = []
    index: list[object] = []
    for end in range(needed, len(joined) + 1):
        start = max(0, end - window)
        y_window = y_values[start:end]
        x_window = x_values[start:end]
        if len(y_window) <= len(names) + 1:
            continue
        block = np.column_stack([np.ones(len(x_window)), x_window])
        if int(np.linalg.matrix_rank(block)) < block.shape[1]:
            continue
        try:
            coef, *_ = np.linalg.lstsq(block, y_window, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            continue
        residuals = y_window - block @ coef
        total = float(np.sum((y_window - y_window.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residuals**2)) / total if total > 0.0 else _NAN
        row = {"alpha_annual": float(coef[0]) * periods_per_year, "r_squared": _to_float(r2)}
        row.update({name: float(coef[i + 1]) for i, name in enumerate(names)})
        rows.append(row)
        index.append(joined.index[end - 1])

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, index=pd.Index(index, name=joined.index.name or "Date"))
    return frame[[*names, "alpha_annual", "r_squared"]]


def factor_contributions(
    result: RegressionResult,
    factors: object,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Decompose the fitted annual excess return into per-factor pieces.

    ``contribution_i = beta_i * mean(factor_i) * periods_per_year``, plus the
    intercept, which sums to the model's fitted annual excess return. This is
    the *model's* attribution of return, not a realised P&L attribution --
    :mod:`src.analytics.attribution` does the latter from actual positions,
    and the two answer different questions.

    Columns: ``factor, beta, mean_annual, contribution_annual, share_pct``.
    Empty frame when the regression did not estimate.
    """
    if not result.ok:
        return pd.DataFrame(
            columns=["factor", "beta", "mean_annual", "contribution_annual", "share_pct"]
        )

    frame = _clean_frame(factors)
    rows: list[dict[str, object]] = []
    for load in result.loadings:
        if load.name not in frame.columns:
            continue
        mean_period = _to_float(frame[load.name].mean())
        mean_annual = mean_period * periods_per_year if math.isfinite(mean_period) else _NAN
        contribution = (
            load.beta * mean_annual
            if math.isfinite(load.beta) and math.isfinite(mean_annual)
            else _NAN
        )
        rows.append(
            {
                "factor": load.name,
                "beta": load.beta,
                "mean_annual": mean_annual,
                "contribution_annual": contribution,
            }
        )

    rows.append(
        {
            "factor": "Alpha (unexplained)",
            "beta": _NAN,
            "mean_annual": _NAN,
            "contribution_annual": result.alpha_annual,
        }
    )

    out = pd.DataFrame(rows)
    finite = out["contribution_annual"].apply(lambda v: v if math.isfinite(_to_float(v)) else 0.0)
    gross = float(finite.abs().sum())
    out["share_pct"] = finite.abs() / gross * 100.0 if gross > 0.0 else pd.Series([_NAN] * len(out))
    return out[["factor", "beta", "mean_annual", "contribution_annual", "share_pct"]]


# ---------------------------------------------------------------------------
# factor sources
# ---------------------------------------------------------------------------
def build_proxy_factors(
    prices: pd.DataFrame | None,
    specs: Sequence[FactorSpec],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build long/short factor return series from tradeable proxy prices.

    ``(factors, warnings)``. Each spec becomes one column:

    * with a ``short_symbol``: ``return(long) - return(short)`` — a
      self-financing long/short spread, e.g. small-cap ETF minus large-cap ETF
      as a size proxy;
    * without one: ``return(long) - period_risk_free`` — an excess return, used
      for the market factor.

    **These are proxies, not the Fama-French factors.** The academic factors
    are built from the full cross-section of stocks sorted on book-to-market,
    size and so on, rebalanced annually, and are not investable. An ETF spread
    is investable, correlated with the real factor, and different from it —
    the tracking difference includes fund fees, the index provider's own
    screening rules and the ETF's inception date. Every caller is expected to
    label them as proxies; nothing in this module calls them anything else.

    Specs whose symbols are absent from ``prices`` are SKIPPED with a warning
    rather than silently omitted, because a five-factor model that quietly
    became a two-factor model is a different model.
    """
    warnings: list[str] = []
    frame = _clean_frame(prices)
    if frame.empty:
        return pd.DataFrame(), ["No proxy price history supplied."]

    returns = frame.pct_change().iloc[1:]
    period_rf = to_period_rate(risk_free_rate, periods_per_year)
    if not math.isfinite(period_rf):
        period_rf = 0.0
        warnings.append("Risk-free rate was unusable; the market proxy uses a zero rate.")

    columns: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.long_symbol not in returns.columns:
            warnings.append(
                f"Factor '{spec.name}' skipped: no price history for {spec.long_symbol}."
            )
            continue
        if spec.short_symbol and spec.short_symbol not in returns.columns:
            warnings.append(
                f"Factor '{spec.name}' skipped: no price history for its short leg "
                f"{spec.short_symbol}."
            )
            continue
        long_leg = returns[spec.long_symbol]
        if spec.short_symbol:
            columns[spec.name] = (long_leg - returns[spec.short_symbol]).rename(spec.name)
        else:
            columns[spec.name] = (long_leg - period_rf).rename(spec.name)

    if not columns:
        return pd.DataFrame(), [*warnings, "No factor could be built from the supplied prices."]
    return pd.DataFrame(columns).dropna(), warnings


#: Column-name aliases in published factor files. Kenneth French's library
#: writes "Mkt-RF"; the IIM Ahmedabad Indian dataset writes "MF" and "WML".
_FACTOR_ALIASES: dict[str, str] = {
    "mkt_rf": "Mkt-RF",
    "mktrf": "Mkt-RF",
    "mkt": "Mkt-RF",
    "mf": "Mkt-RF",
    "market": "Mkt-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
    "wml": "WML",
    "mom": "WML",
    "umd": "WML",
    "momentum": "WML",
    "rf": "RF",
}

#: Above this mean absolute daily value, a factor file is almost certainly
#: quoted in PERCENT rather than decimals. A real daily factor return averages
#: a few basis points; 0.05 (5%) per day is not a return series.
_PERCENT_CUTOFF: float = 0.05


def parse_factor_csv(
    source: object,
    scale: str = "auto",
    date_column: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read a published factor file (Kenneth French, IIM Ahmedabad, or custom).

    ``(factors, warnings)``, never raising. The frame is indexed by date with
    one column per factor, in DECIMAL returns.

    The trap this exists to handle: published factor files are almost always
    in **percent** (``0.53`` means 0.53%), while everything in this
    application is decimal. Getting it wrong scales every loading by 100 and
    the mistake is invisible — the betas simply look absurd. ``scale="auto"``
    therefore inspects the magnitude and REPORTS the decision it made; pass
    ``"percent"`` or ``"decimal"`` to state it yourself.

    A ``RF`` column is kept, since the caller may want to use the file's own
    risk-free series rather than the sidebar's rate.
    """
    warnings: list[str] = []

    try:
        if isinstance(source, pd.DataFrame):
            raw = source.copy()
        else:
            handle = io.BytesIO(source) if isinstance(source, bytes) else source
            raw = pd.read_csv(handle, encoding="utf-8-sig", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001 - a bad upload is a normal user event
        return pd.DataFrame(), [f"Could not read the factor file: {exc}"]

    if raw is None or raw.empty:
        return pd.DataFrame(), ["The factor file contains no rows."]

    raw.columns = [str(c).strip() for c in raw.columns]
    chosen_date = date_column or str(raw.columns[0])
    if chosen_date not in raw.columns:
        return pd.DataFrame(), [f"Column '{chosen_date}' is not in the factor file."]

    # Published files write dates as YYYYMMDD (French daily), YYYYMM (French
    # monthly) or ISO. The explicit formats are tried first, in that order:
    # letting pandas infer would work too, but it emits a "could not infer
    # format" warning and falls back to per-element dateutil parsing, which is
    # both noisy and needlessly slow on a multi-decade factor file.
    stamps = raw[chosen_date].astype(str).str.strip()
    dates = pd.Series(pd.NaT, index=stamps.index, dtype="datetime64[ns]")
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y%m", "%d-%m-%Y", "%d/%m/%Y"):
        attempt = pd.to_datetime(stamps, errors="coerce", format=fmt)
        if attempt.notna().sum() > dates.notna().sum():
            dates = attempt
        if dates.notna().all():
            break
    if dates.isna().all():
        # Last resort: let pandas infer. Its "could not infer format" warning
        # is expected here (we have already tried every published layout), and
        # surfacing it would just be noise in the caller's log.
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("ignore", UserWarning)
            dates = pd.to_datetime(stamps, errors="coerce")
    if dates.isna().all():
        return pd.DataFrame(), [
            f"Could not read dates from column '{chosen_date}'. Expected ISO dates, "
            "YYYYMMDD or YYYYMM."
        ]

    values = raw.drop(columns=[chosen_date]).apply(pd.to_numeric, errors="coerce")
    frame = values.set_index(pd.DatetimeIndex(dates, name="Date"))
    frame = frame.loc[frame.index.notna()].dropna(axis=1, how="all").dropna(how="all")
    if frame.empty:
        return pd.DataFrame(), ["No numeric factor columns were found."]

    renamed = {
        column: _FACTOR_ALIASES.get(
            str(column).strip().lower().replace(" ", "_").replace("-", "_"), str(column).strip()
        )
        for column in frame.columns
    }
    frame = frame.rename(columns=renamed)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]

    magnitude = float(frame.abs().mean().mean())
    if scale == "auto":
        as_percent = math.isfinite(magnitude) and magnitude > _PERCENT_CUTOFF
        warnings.append(
            f"Values average {magnitude:.4g} in absolute terms, read as "
            + ("PERCENT and divided by 100." if as_percent else "DECIMAL returns.")
            + " Pass scale='percent' or scale='decimal' to override."
        )
    else:
        as_percent = scale.strip().lower() == "percent"

    if as_percent:
        frame = frame / 100.0

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame, warnings
