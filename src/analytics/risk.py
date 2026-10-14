"""Risk engine: dispersion, tail risk, drawdown and a transparent risk score.

This module is the quantitative centre of QuantFolio. Everything here is a
pure function of a price/return series -- no I/O, no global state, no
Streamlit -- so every number on screen can be traced back to one formula
stated in a docstring below.

Conventions (restated on each function, collected here so a reviewer can
audit them in one place)
------------------------------------------------------------------------
Daily return        r_t   = P_t / P_(t-1) - 1
Annualised return   R_a   = (V_end / V_start) ** (TRADING_DAYS / (n - 1)) - 1
                    (imported from ``src.analytics.returns`` and re-exported
                    here so the two modules cannot drift apart)
Annualised vol      sigma = std(r, ddof=1) * sqrt(TRADING_DAYS)
Sharpe              S     = (R_a - R_f) / sigma
Sortino             So    = (R_a - R_f) / downside_deviation
Drawdown            d_t   = V_t / running_max(V) - 1                (<= 0)
Beta                beta  = cov(r_p, r_b) / var(r_b)
VaR / CVaR          NEGATIVE decimals describing a ONE-DAY loss:
                    -0.021 means "a 2.1% loss". Nothing is clamped or
                    sign-flipped, so a series of overwhelmingly positive
                    returns can legitimately report a positive VaR.
Risk-free rate      always an ANNUALISED decimal (0.065 = 6.5%).
Undefined values    ``float("nan")`` (or an empty Series/DataFrame). Nothing
                    here raises on degenerate input and nothing substitutes
                    zero for "undefined" -- a fabricated zero is worse than a
                    blank in a risk report.

Every constant (annualisation factor, minimum observation counts, default
confidence level, risk-score rubric) comes from ``src.config``; none is
written inline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.analytics.returns import annualized_return
from src.config import (
    DEFAULT_MAR,
    DEFAULT_VAR_CONFIDENCE,
    MIN_OBSERVATIONS,
    RISK_RUBRIC,
    TRADING_DAYS,
)

__all__ = [
    "TRADING_DAYS",
    "MIN_OBSERVATIONS",
    "MIN_VAR_OBSERVATIONS",
    "DEFAULT_VAR_CONFIDENCE",
    "VAR_METHODS",
    "daily_returns",
    "portfolio_value_series",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "drawdown_series",
    "max_drawdown",
    "beta",
    "correlation_matrix",
    "herfindahl_index",
    "downside_deviation",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "ulcer_index",
    "martin_ratio",
    "value_at_risk",
    "conditional_var",
    "skewness",
    "excess_kurtosis",
    "tail_ratio",
    "hit_ratio",
    "gain_to_pain",
    "covariance_matrix",
    "effective_number_of_bets",
    "diversification_ratio",
    "kupiec_pof_test",
    "risk_score",
    "risk_summary",
]

# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
#: Supported quantile estimators for VaR / CVaR.
VAR_METHODS: tuple[str, ...] = ("historical", "gaussian", "cornish_fisher")

#: A tail statistic estimated from fewer observations than this is noise, not
#: information, so it is reported as NaN. Half the general reliability floor:
#: at the 95% level a 15-day sample contains less than one expected breach.
MIN_VAR_OBSERVATIONS: int = MIN_OBSERVATIONS // 2


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


def _clip01(value: float) -> float:
    """Clamp to the unit interval; NaN in gives 0.0 out (no points awarded)."""
    number = _to_float(value)
    if not math.isfinite(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


def _clean_values(values: Any) -> pd.Series:
    """Copy-coerce a value series to float, drop NaN/inf, sort a date index.

    Never mutates the caller's object (``copy=True``). Mirrors the helper in
    ``src.analytics.returns`` so both modules digest the same inputs the same
    way.
    """
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    try:
        series = pd.Series(values, copy=True)
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


def _clean_returns(returns: Any) -> pd.Series:
    """Copy-coerce a return series to float and drop NaN/inf observations."""
    return _clean_values(returns)


def _weight_series(weights: Any, index: pd.Index | None = None) -> pd.Series:
    """Coerce weights (dict / Series / sequence) to a float Series.

    When ``index`` is given the weights are aligned to it: by label when the
    labels overlap (missing names become 0.0), otherwise positionally when the
    lengths match. An unusable input yields an EMPTY Series, never an
    exception.
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
    except (TypeError, ValueError):
        return pd.Series(dtype=float)

    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)

    if index is not None:
        if len(set(series.index) & set(index)) > 0:
            series = series.reindex(index)
        elif len(series) == len(index):
            series = pd.Series(series.to_numpy(dtype=float), index=index)
        else:
            return pd.Series(dtype=float)
        # Fill AFTER both branches. A NaN weight means "no known exposure" and
        # must read as zero however it was supplied; left in place on the
        # positional path it makes w' Sigma w non-finite and NaNs out every
        # weight-driven statistic, so the same portfolio spelled two ways would
        # return two different answers. (attribution._weight_series matches.)
        return series.fillna(0.0).astype("float64")
    return series.dropna().astype("float64")


def _normalise_weights(weights: pd.Series) -> pd.Series:
    """Scale weights so they sum to 1 (accepts percent or fraction input).

    Returns an EMPTY Series when the weights are empty or sum to zero, which
    is what makes every weight-driven statistic degrade to NaN instead of
    dividing by zero.
    """
    if weights.empty:
        return pd.Series(dtype=float)
    total = float(weights.sum())
    if not math.isfinite(total) or total == 0.0:
        return pd.Series(dtype=float)
    return weights / total


def _z_quantile(confidence: float) -> float:
    """Standard-normal quantile at the LOSS tail: z = Phi^-1(1 - confidence).

    Negative for any sensible confidence level (z = -1.645 at 95%).
    """
    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return float("nan")
    return float(norm.ppf(1.0 - level))


def _xlogy(count: float, probability: float) -> float:
    """count * log(probability) with the convention 0 * log(0) = 0."""
    if count == 0.0:
        return 0.0
    if probability <= 0.0:
        return float("-inf")
    return float(count * math.log(probability))


# ---------------------------------------------------------------------------
# return series
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily returns, r_t = P_t / P_(t-1) - 1.

    Gaps are forward-filled first (assets trade on different exchange
    calendars, so a holiday in one market must not read as a -100% day), and
    the first row -- always NaN by construction -- is dropped.
    ``fill_method=None`` is passed explicitly because forward-filling has
    already happened; relying on the pandas default would be version-dependent.

    The input is never mutated: ``ffill`` returns a new object.
    """
    if prices is None:
        return pd.Series(dtype=float)
    return prices.ffill().pct_change(fill_method=None).iloc[1:]


def portfolio_value_series(prices: pd.DataFrame, quantities: dict[str, float]) -> pd.Series:
    """Historical portfolio value:  V_t = sum_i( quantity_i * price_(i,t) ).

    Modelling assumption (surfaced in the UI and the report): *today's* share
    counts are held constant across the whole window, so this answers "how
    would my CURRENT portfolio have behaved?" rather than "what did I
    actually earn?". That is the standard basis for volatility, drawdown and
    beta; the money-weighted answer is ``returns.portfolio_xirr``.

    Tickers absent from ``prices`` are ignored. Returns an EMPTY Series when
    nothing overlaps.
    """
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.Series(dtype=float)
    try:
        qty = pd.Series(quantities, dtype=float)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    common = [c for c in prices.columns if c in qty.index]
    if not common:
        return pd.Series(dtype=float)
    values = prices[common].mul(qty[common], axis=1).sum(axis=1)
    values.name = "portfolio_value"
    return values


# ---------------------------------------------------------------------------
# headline dispersion metrics
# ---------------------------------------------------------------------------
def annualized_volatility(returns: pd.Series) -> float:
    """Sample std-dev of daily returns scaled by sqrt(TRADING_DAYS).

        sigma = std(r, ddof=1) * sqrt(TRADING_DAYS)

    ``ddof=1`` (Bessel-corrected) because a price history is a sample, not the
    population. NaN for fewer than two observations.
    """
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    result = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    return result if math.isfinite(result) else float("nan")


def sharpe_ratio(ann_return: float, ann_vol: float, risk_free_rate: float) -> float:
    """Excess return per unit of TOTAL volatility: (R_a - R_f) / sigma.

    ``risk_free_rate`` is an annualised decimal (0.065 = 6.5%) and is
    user-configurable, because a Sharpe ratio quoted without its risk-free
    assumption is not a number, it is a rumour.

    NaN when either input is non-finite or volatility is exactly zero (a
    riskless series has no risk-adjusted return, and dividing by zero here is
    the classic way to print an infinite Sharpe).
    """
    excess = _to_float(ann_return)
    vol = _to_float(ann_vol)
    rate = _to_float(risk_free_rate)
    if not math.isfinite(excess) or not math.isfinite(vol) or vol == 0.0:
        return float("nan")
    if not math.isfinite(rate):
        return float("nan")
    return float((excess - rate) / vol)


def downside_deviation(
    returns: pd.Series, mar: float = DEFAULT_MAR, periods: int = TRADING_DAYS
) -> float:
    """Annualised semi-deviation below a minimum acceptable return.

        mar_daily = (1 + mar) ** (1 / periods) - 1
        DD        = sqrt( mean_t( min(r_t - mar_daily, 0) ** 2 ) ) * sqrt(periods)

    ``mar`` is an ANNUALISED decimal (0.0 by default, i.e. capital
    preservation). The mean is taken over EVERY observation, not just the
    losing ones -- that is the standard Sortino denominator, and it is why
    downside deviation is normally smaller than total volatility rather than
    merely a volatility of the losses.

    Returns 0.0 when no observation falls below the MAR (the honest answer:
    there was no measured downside); ``sortino_ratio`` then guards the
    division. NaN for fewer than two observations, a non-positive ``periods``,
    or ``mar <= -1``.
    """
    r = _clean_returns(returns)
    span = _to_float(periods)
    threshold = _to_float(mar)
    if len(r) < 2 or not math.isfinite(span) or span < 1:
        return float("nan")
    if not math.isfinite(threshold) or threshold <= -1.0:
        return float("nan")

    span = float(int(span))
    mar_daily = (1.0 + threshold) ** (1.0 / span) - 1.0
    shortfall = np.minimum(r.to_numpy(dtype=float) - mar_daily, 0.0)
    result = float(math.sqrt(float(np.mean(shortfall**2))) * math.sqrt(span))
    return result if math.isfinite(result) else float("nan")


def sortino_ratio(ann_return: float, downside_dev: float, risk_free_rate: float) -> float:
    """Excess return per unit of DOWNSIDE volatility: (R_a - R_f) / DD.

    Identical to the Sharpe ratio except that upside dispersion is not
    penalised. Because ``DD <= sigma`` for any series whose gains are not all
    losses, a positive-excess-return portfolio always scores at least as well
    on Sortino as on Sharpe.

    NaN when either input is non-finite or the downside deviation is zero.
    """
    excess = _to_float(ann_return)
    dev = _to_float(downside_dev)
    rate = _to_float(risk_free_rate)
    if not math.isfinite(excess) or not math.isfinite(dev) or dev == 0.0:
        return float("nan")
    if not math.isfinite(rate):
        return float("nan")
    return float((excess - rate) / dev)


def calmar_ratio(ann_return: float, mdd: float) -> float:
    """Annualised return per unit of worst peak-to-trough loss.

        Calmar = R_a / |MDD|

    No risk-free deduction, by convention. NaN when either input is
    non-finite or the drawdown is exactly zero (an undrawn series has an
    undefined, not infinite, Calmar).
    """
    ret = _to_float(ann_return)
    drawdown = _to_float(mdd)
    if not math.isfinite(ret) or not math.isfinite(drawdown) or drawdown == 0.0:
        return float("nan")
    return float(ret / abs(drawdown))


def omega_ratio(returns: pd.Series, mar_annual: float = DEFAULT_MAR) -> float:
    """Probability-weighted ratio of gains to losses about a threshold.

        mar_daily = (1 + mar_annual) ** (1 / TRADING_DAYS) - 1
        Omega     = sum_t max(r_t - mar_daily, 0) / sum_t max(mar_daily - r_t, 0)

    Omega uses the WHOLE return distribution rather than its first two
    moments, so unlike Sharpe it responds to skew and fat tails. Omega > 1
    means the gain area above the threshold exceeds the loss area below it.

    NaN when there are fewer than two observations, when ``mar_annual <= -1``,
    or when nothing fell below the threshold (a zero denominator is
    undefined, not infinite).
    """
    r = _clean_returns(returns)
    threshold = _to_float(mar_annual)
    if len(r) < 2 or not math.isfinite(threshold) or threshold <= -1.0:
        return float("nan")

    mar_daily = (1.0 + threshold) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r.to_numpy(dtype=float) - mar_daily
    gains = float(np.sum(np.maximum(excess, 0.0)))
    pains = float(np.sum(np.maximum(-excess, 0.0)))
    if not math.isfinite(gains) or not math.isfinite(pains) or pains == 0.0:
        return float("nan")
    return float(gains / pains)


# ---------------------------------------------------------------------------
# drawdown
# ---------------------------------------------------------------------------
def drawdown_series(values: pd.Series) -> pd.Series:
    """Running peak-to-current loss:  d_t = V_t / running_max(V) - 1  (<= 0).

    Periods whose running peak is non-positive yield NaN rather than +/-inf,
    so a value series that touched zero degrades to a blank instead of a
    fabricated number. Returns an EMPTY Series for empty input.
    """
    v = _clean_values(values)
    if v.empty:
        return pd.Series(dtype=float)
    peak = v.cummax()
    return v / peak.where(peak > 0) - 1.0


def max_drawdown(values: pd.Series) -> float:
    """Deepest peak-to-trough decline as a negative decimal (-0.25 = -25%).

    NaN for an empty series; 0.0 for a monotonically rising one.
    """
    dd = drawdown_series(values)
    if dd.empty:
        return float("nan")
    return _to_float(dd.min())


def ulcer_index(values: pd.Series) -> float:
    """Root-mean-square drawdown, in PERCENT.

        UI = sqrt( mean_t( d_t_pct ** 2 ) ),   d_t_pct = drawdown in percent

    Unlike max drawdown -- a single unlucky day -- the Ulcer Index penalises
    both the DEPTH and the DURATION of every decline, which is much closer to
    what an investor actually experiences. Lower is better; a series at new
    highs throughout scores 0.

    NaN for an empty series.
    """
    dd = drawdown_series(values).dropna()
    if dd.empty:
        return float("nan")
    squared = (dd.to_numpy(dtype=float) * 100.0) ** 2
    result = float(math.sqrt(float(np.mean(squared))))
    return result if math.isfinite(result) else float("nan")


def martin_ratio(ann_return: float, ulcer: float, risk_free_rate: float) -> float:
    """Excess return per unit of Ulcer Index (the "ulcer performance index").

        Martin = (R_a - R_f) / (UI / 100)

    ``ulcer`` arrives in PERCENT from :func:`ulcer_index` and is converted to
    a decimal here so both legs share units -- mixing a decimal return with a
    percentage denominator would silently scale the ratio by 100.

    NaN when any input is non-finite or the Ulcer Index is zero.
    """
    ret = _to_float(ann_return)
    pain = _to_float(ulcer)
    rate = _to_float(risk_free_rate)
    if not math.isfinite(ret) or not math.isfinite(pain) or pain == 0.0:
        return float("nan")
    if not math.isfinite(rate):
        return float("nan")
    return float((ret - rate) / (pain / 100.0))


# ---------------------------------------------------------------------------
# tail risk
# ---------------------------------------------------------------------------
def value_at_risk(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    method: str = "historical",
) -> float:
    """One-day Value at Risk as a NEGATIVE decimal loss (-0.021 = -2.1%).

    SIGN CONVENTION: this function returns the return quantile itself, not its
    absolute value. A 95% VaR of -0.021 reads "on the worst 5% of days the
    portfolio lost at least 2.1%". Nothing is clamped, so a series that is
    overwhelmingly positive can legitimately report a positive number; that is
    information, not an error.

    ``method``:

    * ``"historical"``     -- the empirical quantile at ``1 - confidence``.
      Assumption-free about the shape of the distribution, but it can never
      report a loss worse than the worst day already observed.
    * ``"gaussian"``       -- ``mu + z * sigma`` with ``z = Phi^-1(1 - c)``.
      Closed form, but normality understates equity tails.
    * ``"cornish_fisher"`` -- the Gaussian quantile corrected for the sample's
      third and fourth moments::

            z_cf = z + (z^2 - 1) * S / 6
                     + (z^3 - 3z) * K / 24
                     - (2z^3 - 5z) * S^2 / 36
            VaR  = mu + z_cf * sigma

      with ``S`` the sample skewness and ``K`` the sample EXCESS kurtosis. For
      the negatively skewed, fat-tailed distributions equities actually
      produce this pushes the quantile further into the tail than Gaussian.

    NaN for fewer than ``MIN_VAR_OBSERVATIONS`` observations, a confidence
    level outside (0, 1), or an unrecognised method (never raises).
    """
    r = _clean_returns(returns)
    if len(r) < MIN_VAR_OBSERVATIONS:
        return float("nan")

    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return float("nan")

    name = str(method).strip().lower()
    sample = r.to_numpy(dtype=float)

    if name == "historical":
        return float(np.percentile(sample, (1.0 - level) * 100.0))

    if name in ("gaussian", "normal", "parametric", "cornish_fisher", "cornish-fisher"):
        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        z = _z_quantile(level)
        if not math.isfinite(mu) or not math.isfinite(sigma) or not math.isfinite(z):
            return float("nan")
        if name in ("gaussian", "normal", "parametric"):
            return float(mu + z * sigma)

        skew = skewness(r)
        kurt = excess_kurtosis(r)
        if not math.isfinite(skew) or not math.isfinite(kurt):
            return float("nan")
        z_cf = (
            z
            + (z**2 - 1.0) * skew / 6.0
            + (z**3 - 3.0 * z) * kurt / 24.0
            - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
        )
        # Cornish-Fisher is an ASYMPTOTIC expansion in the third and fourth
        # moments, valid only while it stays monotone in z. Outside that domain
        # it is not a quantile transform at all, and it can return a *gain* as
        # the 99% worst case -- a single -20% day among 400 quiet ones is enough
        # (skew ~ -19, excess kurtosis ~ 370). Reporting that would be far worse
        # than reporting nothing, so we check monotonicity at the requested
        # quantile and fall back to the assumption-free historical estimate.
        dz_cf = (
            1.0
            + 2.0 * z * skew / 6.0
            + (3.0 * z**2 - 3.0) * kurt / 24.0
            - (6.0 * z**2 - 5.0) * skew**2 / 36.0
        )
        if not math.isfinite(dz_cf) or dz_cf <= 0.0:
            return float(np.percentile(sample, (1.0 - level) * 100.0))
        return float(mu + z_cf * sigma)

    return float("nan")


def conditional_var(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    method: str = "historical",
) -> float:
    """Expected shortfall: the average loss GIVEN that VaR was breached.

    Same NEGATIVE-decimal convention as :func:`value_at_risk`, and always at
    least as bad (``CVaR <= VaR``), because it averages the tail beyond the
    quantile instead of stopping at it. CVaR is the coherent risk measure --
    unlike VaR it is sub-additive, so merging two portfolios can never make it
    look better than the sum of its parts.

    * ``"historical"`` / ``"cornish_fisher"`` -- the empirical mean of the
      returns at or below that method's VaR threshold. In the degenerate case
      where no observation reaches the threshold (possible only for the
      Cornish-Fisher quantile, which can extrapolate past the worst observed
      day) the threshold itself is returned, which is the limiting value of
      the conditional mean and keeps ``CVaR <= VaR`` true by construction.
    * ``"gaussian"`` -- the closed form

          ES = mu - sigma * phi(z_alpha) / (1 - c)

      where ``phi`` is the standard-normal PDF and ``z_alpha = Phi^-1(1 - c)``.

    NaN under exactly the same conditions as :func:`value_at_risk`.
    """
    r = _clean_returns(returns)
    if len(r) < MIN_VAR_OBSERVATIONS:
        return float("nan")

    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return float("nan")

    name = str(method).strip().lower()
    sample = r.to_numpy(dtype=float)

    if name in ("gaussian", "normal", "parametric"):
        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        z = _z_quantile(level)
        if not math.isfinite(mu) or not math.isfinite(sigma) or not math.isfinite(z):
            return float("nan")
        return float(mu - sigma * float(norm.pdf(z)) / (1.0 - level))

    threshold = value_at_risk(r, level, name)
    if not math.isfinite(threshold):
        return float("nan")
    tail = sample[sample <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(np.mean(tail))


def skewness(returns: pd.Series) -> float:
    """Sample skewness (Fisher-Pearson, bias-corrected) of daily returns.

    Negative skew -- the norm for equities -- means the left tail is longer:
    losses are rarer but bigger than the symmetric Gaussian assumption behind
    Sharpe and parametric VaR admits. NaN for fewer than three observations.
    """
    r = _clean_returns(returns)
    if len(r) < 3:
        return float("nan")
    return _to_float(r.skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """Sample EXCESS kurtosis (Fisher definition: normal = 0).

    Positive excess kurtosis means fatter tails and a taller peak than the
    normal distribution, i.e. more days of "that shouldn't happen" than a
    Gaussian VaR model prices in. NaN for fewer than four observations.
    """
    r = _clean_returns(returns)
    if len(r) < 4:
        return float("nan")
    return _to_float(r.kurt())


def tail_ratio(returns: pd.Series) -> float:
    """Size of the right tail relative to the left tail.

        tail_ratio = |quantile(r, c)| / |quantile(r, 1 - c)|

    with ``c = DEFAULT_VAR_CONFIDENCE`` (the 95th versus the 5th percentile).
    Above 1 the best days are larger than the worst days; below 1 the
    portfolio gives back more in a crisis than it earns in a rally.

    NaN for fewer than ``MIN_VAR_OBSERVATIONS`` observations or a zero left
    tail.
    """
    r = _clean_returns(returns)
    if len(r) < MIN_VAR_OBSERVATIONS:
        return float("nan")
    sample = r.to_numpy(dtype=float)
    upper = float(np.percentile(sample, DEFAULT_VAR_CONFIDENCE * 100.0))
    lower = float(np.percentile(sample, (1.0 - DEFAULT_VAR_CONFIDENCE) * 100.0))
    if not math.isfinite(upper) or not math.isfinite(lower) or lower == 0.0:
        return float("nan")
    return float(abs(upper) / abs(lower))


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of observations that were strictly positive (0 to 1).

        hit = count(r_t > 0) / n

    A DECIMAL fraction, not a percentage. Flat days count in the denominator
    but not the numerator. NaN for an empty series.
    """
    r = _clean_returns(returns)
    if r.empty:
        return float("nan")
    return float((r > 0.0).sum() / len(r))


def gain_to_pain(returns: pd.Series) -> float:
    """Sum of all returns divided by the absolute sum of the losing ones.

        GPR = sum_t r_t / |sum_t min(r_t, 0)|

    A pain-weighted profit factor: it asks how much total return the investor
    was paid for each unit of loss they had to sit through. NaN when there
    were no losses (a zero denominator) or the series is empty.
    """
    r = _clean_returns(returns)
    if r.empty:
        return float("nan")
    sample = r.to_numpy(dtype=float)
    pain = float(np.sum(np.minimum(sample, 0.0)))
    if pain == 0.0 or not math.isfinite(pain):
        return float("nan")
    total = float(np.sum(sample))
    if not math.isfinite(total):
        return float("nan")
    return float(total / abs(pain))


# ---------------------------------------------------------------------------
# relationship to a benchmark and to other assets
# ---------------------------------------------------------------------------
def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """cov(r_asset, r_benchmark) / var(r_benchmark) on the OVERLAPPING dates.

    Inner-joining first matters: comparing an Indian portfolio with a US index
    over misaligned calendars produces a beta that is mostly an artefact of
    the holiday schedule.

    Identity: the beta of any series against itself is exactly 1.0. NaN for
    fewer than two overlapping points or a zero-variance benchmark.
    """
    left = _clean_returns(asset_returns)
    right = _clean_returns(benchmark_returns)
    if left.empty or right.empty:
        return float("nan")
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]
    bench_var = _to_float(b.var(ddof=1))
    if not math.isfinite(bench_var) or bench_var == 0.0:
        return float("nan")
    return _to_float(a.cov(b) / bench_var)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns between all portfolio assets.

    Returns an EMPTY DataFrame for empty or non-numeric input. Correlations
    computed from fewer than two overlapping observations come back as NaN
    rather than as a fabricated 0.
    """
    if returns is None:
        return pd.DataFrame()
    frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr()


def covariance_matrix(
    returns: pd.DataFrame, shrinkage: float = 0.0, annualize: bool = True
) -> pd.DataFrame:
    """Asset covariance matrix, optionally annualised and shrunk.

        S_sample = cov(r, ddof=1)                 (pairwise-complete)
        S_ann    = S_sample * TRADING_DAYS        (when annualize=True)
        S        = (1 - a) * S_ann + a * (trace(S_ann) / n) * I

    Shrinking the sample covariance toward a scaled identity (Ledoit-Wolf in
    spirit, with the intensity supplied rather than estimated) is what keeps a
    mean-variance optimiser from betting the portfolio on the noisiest
    eigenvector when the number of assets approaches the number of
    observations.

    The shrinkage target is chosen so that ``trace(S) == trace(S_ann)`` for
    ANY intensity: total variance is preserved and only the cross-sectional
    structure is pulled toward independence.

    ``shrinkage`` is clamped to [0, 1]. Returns an EMPTY DataFrame for input
    with fewer than two rows or no numeric columns. Never mutates the input.
    """
    if returns is None:
        return pd.DataFrame()
    frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.empty or len(numeric) < 2:
        return pd.DataFrame()

    cov = numeric.cov()
    if annualize:
        cov = cov * float(TRADING_DAYS)

    intensity = _clip01(shrinkage)
    if intensity <= 0.0:
        return cov

    matrix = cov.to_numpy(dtype=float)
    n = matrix.shape[0]
    trace = float(np.trace(matrix))
    if n == 0 or not math.isfinite(trace):
        return cov
    target = np.eye(n) * (trace / n)
    blended = (1.0 - intensity) * matrix + intensity * target
    return pd.DataFrame(blended, index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# concentration & diversification
# ---------------------------------------------------------------------------
def herfindahl_index(weights_pct: Any) -> float:
    """HHI = sum(w_i ** 2) with weights as FRACTIONS of the portfolio.

    Ranges from 1/n for a perfectly equal-weight book to 1.0 for a single
    stock, so it is the cleanest one-number answer to "how concentrated am
    I?". Input is expected in PERCENT (it is divided by 100); a dict or Series
    contributes its values, not its keys.

    NaN for empty or unparseable input.
    """
    if weights_pct is None:
        return float("nan")
    if isinstance(weights_pct, (Mapping, pd.Series)):
        raw: Any = list(pd.Series(weights_pct).to_numpy())
    else:
        try:
            raw = list(weights_pct)
        except TypeError:
            return float("nan")
    try:
        w = np.asarray(raw, dtype=float) / 100.0
    except (TypeError, ValueError):
        return float("nan")
    w = w[np.isfinite(w)]
    if w.size == 0:
        return float("nan")
    return float(np.sum(w**2))


def effective_number_of_bets(weights: Any) -> float:
    """Inverse Herfindahl: 1 / sum(w_i ** 2) with weights normalised to sum 1.

    Reads as "how many equally sized positions would carry the same
    concentration?". A 20-name book where one position is 60% has an
    effective count near 2, not 20. Weights may be supplied as percentages or
    as fractions -- they are normalised internally by their own sum.

    NaN for empty input or weights that sum to zero.
    """
    w = _normalise_weights(_weight_series(weights))
    if w.empty:
        return float("nan")
    total = float(np.sum(w.to_numpy(dtype=float) ** 2))
    if not math.isfinite(total) or total == 0.0:
        return float("nan")
    return float(1.0 / total)


def diversification_ratio(weights: Any, cov: pd.DataFrame) -> float:
    """Weighted average asset volatility divided by portfolio volatility.

        DR = (w . sigma) / sqrt(w' S w),   sigma_i = sqrt(S_ii)

    DR = 1 means the holdings move as one asset and diversification bought
    nothing; DR > 1 quantifies exactly how much volatility the correlation
    structure cancelled out. Weights are aligned to ``cov.columns`` by label
    (missing names count as zero) or positionally when the lengths match, and
    are normalised to sum to 1.

    NaN for an empty covariance matrix, unusable weights, or a non-positive
    portfolio variance.
    """
    if cov is None or not isinstance(cov, pd.DataFrame) or cov.empty:
        return float("nan")
    w = _normalise_weights(_weight_series(weights, cov.columns))
    if w.empty:
        return float("nan")

    matrix = cov.to_numpy(dtype=float)
    vector = w.to_numpy(dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != vector.size:
        return float("nan")

    variances = np.diag(matrix)
    if np.any(~np.isfinite(variances)) or np.any(variances < 0.0):
        return float("nan")
    weighted_vol = float(vector @ np.sqrt(variances))
    portfolio_var = float(vector @ matrix @ vector)
    if not math.isfinite(portfolio_var) or portfolio_var <= 0.0:
        return float("nan")
    if not math.isfinite(weighted_vol):
        return float("nan")
    return float(weighted_vol / math.sqrt(portfolio_var))


# ---------------------------------------------------------------------------
# VaR backtesting
# ---------------------------------------------------------------------------
def kupiec_pof_test(
    returns: pd.Series,
    var_threshold: float,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> dict:
    """Kupiec proportion-of-failures test: is the VaR model calibrated?

    A VaR estimate is a claim about frequency: at 95% confidence, exactly 5%
    of days should breach it. The unconditional-coverage likelihood ratio

        p     = 1 - confidence                       (expected breach rate)
        x     = count(r_t < var_threshold),   n = observations
        pi    = x / n                                (observed breach rate)
        LR_uc = -2 * [ (n-x)*ln(1-p) + x*ln(p) ]
                +2 * [ (n-x)*ln(1-pi) + x*ln(pi) ]

    is asymptotically chi-squared with 1 degree of freedom under the null
    "the model is correctly calibrated". Too FEW breaches fail the test as
    well as too many: a VaR that is never breached is over-stating risk and
    wasting capital.

    ``reject`` is evaluated at a significance level of ``1 - confidence`` (the
    same tail probability the VaR itself uses, so no extra constant is
    introduced). The test only checks the COUNT of breaches, not their
    clustering -- a model that fails all its breaches in one week passes
    Kupiec but not a conditional-coverage test.

    Returns ``{n, breaches, expected_breaches, breach_rate, lr_statistic,
    p_value, reject}``, with NaN statistics and ``reject=False`` when the
    inputs are unusable. Never raises.
    """
    blank = {
        "n": 0,
        "breaches": 0,
        "expected_breaches": float("nan"),
        "breach_rate": float("nan"),
        "lr_statistic": float("nan"),
        "p_value": float("nan"),
        "reject": False,
    }

    r = _clean_returns(returns)
    threshold = _to_float(var_threshold)
    level = _to_float(confidence)
    if r.empty or not math.isfinite(threshold):
        return blank
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return blank

    n = int(len(r))
    breaches = int((r.to_numpy(dtype=float) < threshold).sum())
    p = 1.0 - level
    observed = breaches / n

    log_null = _xlogy(n - breaches, 1.0 - p) + _xlogy(breaches, p)
    log_alt = _xlogy(n - breaches, 1.0 - observed) + _xlogy(breaches, observed)
    if not math.isfinite(log_null) or not math.isfinite(log_alt):
        statistic = float("nan")
        p_value = float("nan")
        reject = False
    else:
        statistic = float(max(-2.0 * log_null + 2.0 * log_alt, 0.0))
        p_value = float(chi2.sf(statistic, 1))
        reject = bool(p_value < p)

    return {
        "n": n,
        "breaches": breaches,
        "expected_breaches": float(n * p),
        "breach_rate": float(observed),
        "lr_statistic": statistic,
        "p_value": p_value,
        "reject": reject,
    }


# ---------------------------------------------------------------------------
# transparent rule-based risk score (NOT financial advice)
# ---------------------------------------------------------------------------
def risk_score(
    ann_vol: float,
    mdd: float,
    weights_pct: Any,
    n_holdings: int,
    cvar: float = float("nan"),
) -> dict:
    """Heuristic 0-100 portfolio risk score with a fully documented rubric.

    Every weight and saturation point comes from ``config.RISK_RUBRIC``, so
    the rubric can be audited and tuned in one place. Each component is
    linearly scaled and capped:

    * volatility     -> ``vol_points``           (0 at 0% annualised vol,
                                                  full at ``vol_cap``)
    * max drawdown   -> ``drawdown_points``      (0 at 0%, full at
                                                  ``drawdown_cap``)
    * concentration  -> ``concentration_points`` (HHI rescaled so a perfectly
                                                  equal-weight book scores 0
                                                  and a single stock scores
                                                  full points)
    * holdings count -> ``holdings_points``      (full at 1 name, 0 at
                                                  ``holdings_floor`` names)
    * tail risk      -> ``tail_points``          (0 at 0% daily CVaR, full at
                                                  ``tail_cap``)

    RESCALING (important): a component whose input is missing is dropped from
    BOTH the numerator and the denominator, and the score is reported as
    ``earned / available * 100``. So a portfolio with no CVaR estimate is
    scored out of the 85 points that were measurable rather than being
    silently flattered by 15 points of missing tail risk. ``max_points`` in
    the returned dict is that denominator, and ``components`` holds the RAW
    points earned (pre-rescaling) so the arithmetic stays inspectable.
    Concentration is treated as unavailable when the weights cannot be parsed
    and more than one holding is claimed.

    Bands come from ``RISK_RUBRIC.low_band`` / ``high_band``: below the low
    band "Low", up to and including the high band "Moderate", above it
    "High".

    Returns ``{"score": float | None, "band": str, "components": dict,
    "max_points": float}``. When annualised volatility or max drawdown is not
    finite, or ``n_holdings < 1``, the score is ``None`` and the band is
    ``"Insufficient data"`` -- the tool refuses to grade a portfolio it could
    not measure.

    The thresholds are transparent judgement calls, NOT a calibrated
    regulatory standard; the app shows this rubric verbatim and labels the
    result as educational only.
    """
    rubric = RISK_RUBRIC
    vol = _to_float(ann_vol)
    drawdown = _to_float(mdd)
    try:
        count = int(n_holdings) if n_holdings is not None else 0
    except (TypeError, ValueError):
        count = 0

    if not math.isfinite(vol) or not math.isfinite(drawdown) or count < 1:
        return {
            "score": None,
            "band": "Insufficient data",
            "components": {},
            "max_points": 0.0,
        }

    components: dict[str, float] = {}
    earned = 0.0
    available = 0.0

    # volatility ----------------------------------------------------------
    vol_cap = rubric.vol_cap if rubric.vol_cap > 0 else 1.0
    vol_pts = rubric.vol_points * _clip01(vol / vol_cap)
    components[f"volatility (max {rubric.vol_points:g})"] = round(vol_pts, 1)
    earned += vol_pts
    available += rubric.vol_points

    # max drawdown --------------------------------------------------------
    dd_cap = rubric.drawdown_cap if rubric.drawdown_cap > 0 else 1.0
    dd_pts = rubric.drawdown_points * _clip01(abs(drawdown) / dd_cap)
    components[f"max drawdown (max {rubric.drawdown_points:g})"] = round(dd_pts, 1)
    earned += dd_pts
    available += rubric.drawdown_points

    # concentration -------------------------------------------------------
    hhi = herfindahl_index(weights_pct)
    if count == 1:
        conc_pts: float | None = rubric.concentration_points
    elif not math.isfinite(hhi):
        conc_pts = None  # unmeasurable -> rescale it away
    else:
        floor = 1.0 / count  # HHI of a perfect equal-weight book
        span = 1.0 - floor
        share = 0.0 if span <= 0 else (hhi - floor) / span
        conc_pts = rubric.concentration_points * _clip01(share)
    if conc_pts is not None:
        label = f"concentration (max {rubric.concentration_points:g})"
        components[label] = round(conc_pts, 1)
        earned += conc_pts
        available += rubric.concentration_points

    # holdings count ------------------------------------------------------
    hold_span = float(rubric.holdings_floor - 1)
    hold_share = 1.0 if hold_span <= 0 else (rubric.holdings_floor - count) / hold_span
    hold_pts = rubric.holdings_points * _clip01(hold_share)
    components[f"holdings count (max {rubric.holdings_points:g})"] = round(hold_pts, 1)
    earned += hold_pts
    available += rubric.holdings_points

    # tail risk (CVaR) ----------------------------------------------------
    tail = _to_float(cvar)
    if math.isfinite(tail):
        tail_cap = rubric.tail_cap if rubric.tail_cap > 0 else 1.0
        tail_pts = rubric.tail_points * _clip01(abs(tail) / tail_cap)
        components[f"tail risk (max {rubric.tail_points:g})"] = round(tail_pts, 1)
        earned += tail_pts
        available += rubric.tail_points

    if available <= 0.0:
        return {
            "score": None,
            "band": "Insufficient data",
            "components": {},
            "max_points": 0.0,
        }

    score = round(min(max(earned / available * 100.0, 0.0), 100.0), 1)
    if score < rubric.low_band:
        band = "Low"
    elif score <= rubric.high_band:
        band = "Moderate"
    else:
        band = "High"

    return {
        "score": float(score),
        "band": band,
        "components": components,
        "max_points": float(available),
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def risk_summary(
    values: pd.Series,
    returns: pd.Series | None = None,
    weights_pct: Any = None,
    n_holdings: int = 0,
    risk_free_rate: float = 0.0,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> dict:
    """Every headline risk statistic for one portfolio, as a flat float dict.

    ``values`` is the portfolio value series; ``returns`` its daily returns
    (derived from ``values`` when omitted). ``risk_free_rate`` is an
    ANNUALISED decimal.

    Units, stated once so the UI never has to guess:

    * ``ann_return``, ``ann_vol``, ``downside_dev``, ``mdd``, ``var``,
      ``cvar``, ``var_gaussian``, ``var_cornish_fisher``, ``best_day``,
      ``worst_day`` -- DECIMALS (0.021 = 2.1%). The VaR/CVaR family is
      negative by convention (see :func:`value_at_risk`) and describes a
      ONE-DAY loss.
    * ``ulcer`` -- PERCENT (it is a root-mean-square of percentage drawdowns).
    * ``hit_ratio`` -- a fraction between 0 and 1.
    * ``sharpe``, ``sortino``, ``calmar``, ``omega``, ``martin``,
      ``tail_ratio``, ``gain_to_pain``, ``skew``, ``kurtosis`` -- pure ratios.
    * ``n_obs``, ``n_holdings`` -- counts, as floats.
    * ``hhi``, ``effective_bets`` -- concentration of ``weights_pct``.

    Every value is a float and every undefined statistic is NaN, so the UI can
    render an em dash without special-casing missing sections. Nothing raises.
    """
    v = _clean_values(values)
    r = _clean_returns(returns) if returns is not None else _clean_returns(daily_returns(v))

    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        level = DEFAULT_VAR_CONFIDENCE
    rate = _to_float(risk_free_rate)
    if not math.isfinite(rate):
        rate = 0.0

    ann_ret = annualized_return(v)
    ann_vol = annualized_volatility(r)
    mdd = max_drawdown(v)
    down_dev = downside_deviation(r)
    ulcer = ulcer_index(v)
    var = value_at_risk(r, level, "historical")
    cvar = conditional_var(r, level, "historical")

    best = _to_float(r.max()) if not r.empty else float("nan")
    worst = _to_float(r.min()) if not r.empty else float("nan")

    try:
        holdings = int(n_holdings) if n_holdings is not None else 0
    except (TypeError, ValueError):
        holdings = 0

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe_ratio(ann_ret, ann_vol, rate),
        "sortino": sortino_ratio(ann_ret, down_dev, rate),
        "calmar": calmar_ratio(ann_ret, mdd),
        "omega": omega_ratio(r),
        "ulcer": ulcer,
        "martin": martin_ratio(ann_ret, ulcer, rate),
        "mdd": mdd,
        "var": var,
        "cvar": cvar,
        "var_gaussian": value_at_risk(r, level, "gaussian"),
        "var_cornish_fisher": value_at_risk(r, level, "cornish_fisher"),
        "skew": skewness(r),
        "kurtosis": excess_kurtosis(r),
        "tail_ratio": tail_ratio(r),
        "hit_ratio": hit_ratio(r),
        "gain_to_pain": gain_to_pain(r),
        "downside_dev": down_dev,
        "n_obs": float(len(r)),
        "best_day": best,
        "worst_day": worst,
        "confidence": float(level),
        "hhi": herfindahl_index(weights_pct),
        "effective_bets": effective_number_of_bets(weights_pct),
        "n_holdings": float(holdings),
    }
