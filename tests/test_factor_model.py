"""Multi-factor regression — ``src.analytics.factor_model``, offline and seeded.

A factor regression is easy to write and easy to get subtly, invisibly wrong,
so these tests are mostly *identities* and *exact recoveries* rather than
smoke checks.

Verified here
-------------
**Exact recovery.** When ``y`` is built as an exact linear combination of the
factors, the regression must return those coefficients, that intercept, and
R² = 1 to floating-point precision. Nothing subtler catches a sign error, a
mis-stacked design matrix or an intercept in the wrong column.

**Cross-module identity.** A one-factor regression's beta is exactly
``risk.beta`` (both are cov/var), asserted to 1e-12. Two modules computing the
same quantity by different routes and agreeing is the strongest evidence
available that neither has drifted, and it is what makes the factor page's
market beta reconcile with the Risk page's.

**The standard-error hierarchy.** HAC at zero lags is White's HC1, and on iid
homoskedastic residuals HC1 agrees with the classical OLS formula — so nothing
is lost by defaulting to robust. On persistent data (AR(1) factor *and* AR(1)
residual) HAC roughly doubles the standard error, cutting a t-statistic of ~16
to ~8. That gap is the whole reason the default is robust, and it is pinned
here so a future "simplification" to plain OLS fails loudly.

**Statistical plumbing.** p-values in [0, 1]; a t-statistic invariant to
annualising alpha; adjusted R² asserted against its algebraic definition and
shown to fall for a factor made *exactly* orthogonal to the target (a merely
random column is not enough — 25 of them can still raise it by luck);
Durbin-Watson near 2 for iid residuals and near 0 for strongly positively
autocorrelated ones; VIF = 1 for orthogonal factors and exploding for
near-duplicates.

**Refusals.** Too few observations, a constant factor, perfectly collinear
factors, no overlapping dates — each returns ``ok=False`` with a stated reason
rather than a fabricated zero.

**Percent-vs-decimal.** Published factor files are quoted in percent while
this application is decimal. Getting it wrong scales every loading by 100 and
looks like nothing at all. The auto-detection is pinned in both directions and
always reports the decision it made.

Deterministic: every RNG is seeded. No network, no Streamlit.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from src.analytics import factor_model as fm
from src.analytics import risk as rsk
from src.config import TRADING_DAYS, FactorSpec

SEED = 20240201
N = 500


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _index(n: int = N) -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=n)


def _ar1(rho: float, sd: float, seed: int, n: int = N) -> np.ndarray:
    """An AR(1) path — the shape that makes HAC standard errors matter."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + rng.normal(0.0, sd)
    return out


@pytest.fixture
def factors() -> pd.DataFrame:
    """Two near-orthogonal factor series."""
    rng = np.random.default_rng(SEED)
    idx = _index()
    return pd.DataFrame(
        {
            "MKT": rng.normal(0.0004, 0.011, N),
            "SMB": rng.normal(0.0001, 0.006, N),
        },
        index=idx,
    )


@pytest.fixture
def exact(factors: pd.DataFrame) -> pd.Series:
    """y = 0.0002 + 1.3*MKT - 0.4*SMB, with no noise at all."""
    return 0.0002 + 1.3 * factors["MKT"] - 0.4 * factors["SMB"]


@pytest.fixture
def noisy(factors: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED + 1)
    return 0.8 * factors["MKT"] - 0.3 * factors["SMB"] + rng.normal(0.0, 0.005, N)


# ---------------------------------------------------------------------------
# exact recovery
# ---------------------------------------------------------------------------
def test_an_exact_linear_combination_is_recovered_exactly(
    exact: pd.Series, factors: pd.DataFrame
) -> None:
    """The test that catches a sign error or a mis-stacked design matrix."""
    result = fm.regress(exact, factors, robust=False)

    assert result.ok
    assert result.alpha == pytest.approx(0.0002, abs=1e-12)
    assert result.loading("MKT").beta == pytest.approx(1.3, abs=1e-10)
    assert result.loading("SMB").beta == pytest.approx(-0.4, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)


def test_a_perfect_fit_leaves_no_residual_volatility(
    exact: pd.Series, factors: pd.DataFrame
) -> None:
    result = fm.regress(exact, factors, robust=False)

    assert result.residual_vol_annual == pytest.approx(0.0, abs=1e-9)
    assert result.unexplained_pct == pytest.approx(0.0, abs=1e-9)


def test_regressing_a_factor_on_itself_gives_beta_one_and_zero_alpha(
    factors: pd.DataFrame,
) -> None:
    """True for any correct implementation, forever."""
    result = fm.regress(factors["MKT"], factors[["MKT"]], robust=False)

    assert result.loading("MKT").beta == pytest.approx(1.0, abs=1e-12)
    assert result.alpha == pytest.approx(0.0, abs=1e-14)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)


def test_a_scaled_factor_scales_the_loading_inversely(factors: pd.DataFrame) -> None:
    """Doubling a factor must halve its loading; the fit is unchanged."""
    y = 0.6 * factors["MKT"]
    plain = fm.regress(y, factors[["MKT"]], robust=False)
    doubled = fm.regress(y, (factors[["MKT"]] * 2.0), robust=False)

    assert doubled.loading("MKT").beta == pytest.approx(plain.loading("MKT").beta / 2.0)
    assert doubled.r_squared == pytest.approx(plain.r_squared)


# ---------------------------------------------------------------------------
# cross-module identity
# ---------------------------------------------------------------------------
def test_a_one_factor_beta_equals_risk_beta_exactly(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """Both are cov/var. Two modules agreeing by different routes is the
    strongest evidence available that neither has drifted."""
    regression = fm.regress(noisy, factors[["MKT"]], robust=False)
    covariance = rsk.beta(noisy, factors["MKT"])

    assert regression.loading("MKT").beta == pytest.approx(covariance, abs=1e-12)


def test_the_identity_survives_a_risk_free_rate(noisy: pd.Series, factors: pd.DataFrame) -> None:
    """Subtracting a CONSTANT from y moves the intercept, never the slope."""
    plain = fm.regress(noisy, factors[["MKT"]], robust=False)
    excess = fm.factor_regression(noisy, factors[["MKT"]], risk_free_rate=0.065, robust=False)

    assert excess.loading("MKT").beta == pytest.approx(plain.loading("MKT").beta, abs=1e-12)
    assert excess.alpha < plain.alpha, "the risk-free rate must reduce alpha"


# ---------------------------------------------------------------------------
# standard errors
# ---------------------------------------------------------------------------
def test_hac_at_zero_lags_matches_ols_on_iid_residuals(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """HC1 and the classical formula agree when the assumptions hold, so
    defaulting to robust costs nothing."""
    ols = fm.regress(noisy, factors, robust=False)
    hc1 = fm.regress(noisy, factors, robust=True, lags=0)

    for name in ("MKT", "SMB"):
        assert hc1.loading(name).std_error == pytest.approx(ols.loading(name).std_error, rel=0.10)


def test_hac_roughly_doubles_the_standard_error_on_persistent_data() -> None:
    """The reason robust is the default. An AR(1) factor with an AR(1)
    residual is the realistic case, and OLS is wrong there by ~2x — not
    marginally optimistic."""
    idx = _index()
    factor = pd.Series(_ar1(0.85, 0.01, 1), index=idx, name="F")
    residual = pd.Series(_ar1(0.85, 0.01, 2), index=idx)
    y = 0.5 * factor + residual

    ols = fm.regress(y, factor.to_frame(), robust=False)
    hac = fm.regress(y, factor.to_frame(), robust=True)

    ratio = hac.loading("F").std_error / ols.loading("F").std_error
    assert ratio > 1.5, f"HAC should widen materially, got {ratio:.2f}x"
    assert abs(hac.loading("F").t_stat) < abs(ols.loading("F").t_stat) / 1.5


def test_the_default_is_robust(noisy: pd.Series, factors: pd.DataFrame) -> None:
    assert "Newey-West" in fm.regress(noisy, factors).se_method


def test_the_se_method_states_the_lag_count(noisy: pd.Series, factors: pd.DataFrame) -> None:
    """The reader must be able to see what was assumed."""
    result = fm.regress(noisy, factors, robust=True, lags=7)

    assert result.lags == 7
    assert "7 lag" in result.se_method


def test_the_ols_method_states_its_own_assumption(noisy: pd.Series, factors: pd.DataFrame) -> None:
    assert "iid" in fm.regress(noisy, factors, robust=False).se_method


@pytest.mark.parametrize(
    "n,expected",
    [
        (3, 0),  # too short to estimate any autocovariance
        (100, 4),  # the formula's reference point: 4 * 1 = 4
        (500, 5),  # 4 * 5^(2/9) = 5.72
        (1260, 7),  # 4 * 12.6^(2/9) = 7.02
        (2520, 8),  # 4 * 25.2^(2/9) = 8.19
    ],
)
def test_the_newey_west_lag_rule_matches_the_formula(n: int, expected: int) -> None:
    """floor(4 * (n/100)^(2/9)) — a data-driven default, not a magic number."""
    assert fm.newey_west_lags(n) == expected


def test_the_lag_count_grows_with_the_sample() -> None:
    """Consistency requires the truncation lag to grow (slowly) with n."""
    counts = [fm.newey_west_lags(n) for n in (50, 250, 1000, 5000)]

    assert counts == sorted(counts)


def test_standard_errors_are_positive(noisy: pd.Series, factors: pd.DataFrame) -> None:
    result = fm.regress(noisy, factors)

    assert result.alpha_std_error > 0
    assert all(load.std_error > 0 for load in result.loadings)


# ---------------------------------------------------------------------------
# statistical plumbing
# ---------------------------------------------------------------------------
def test_every_p_value_is_a_probability(noisy: pd.Series, factors: pd.DataFrame) -> None:
    result = fm.regress(noisy, factors)

    assert 0.0 <= result.alpha_p_value <= 1.0
    assert 0.0 <= result.f_p_value <= 1.0
    for load in result.loadings:
        assert 0.0 <= load.p_value <= 1.0, load.name


def test_a_real_loading_is_significant_and_a_noise_loading_is_not(
    factors: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(SEED + 2)
    noise = pd.Series(rng.normal(0.0, 0.004, N), index=factors.index, name="NOISE")
    design = factors.assign(NOISE=noise)
    y = 0.9 * factors["MKT"] + rng.normal(0.0, 0.004, N)

    result = fm.regress(pd.Series(y, index=factors.index), design)

    assert result.loading("MKT").significant(), "a real exposure must be detected"
    assert not result.loading("NOISE").significant(), "noise must not be"


def test_annualising_alpha_leaves_its_t_statistic_unchanged(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """A t-stat is scale-invariant, which is exactly why alpha is annualised
    linearly: the significance claim survives the units change."""
    daily = fm.regress(noisy, factors, periods_per_year=TRADING_DAYS)

    assert daily.alpha_annual == pytest.approx(daily.alpha * TRADING_DAYS)
    monthly = fm.regress(noisy, factors, periods_per_year=12)
    assert monthly.alpha_t_stat == pytest.approx(daily.alpha_t_stat, rel=1e-12)


def test_adjusted_r_squared_matches_its_definition(noisy: pd.Series, factors: pd.DataFrame) -> None:
    """adj = 1 - (1 - R²)(n - 1)/(n - k - 1). Asserted algebraically rather
    than by a directional claim, because "adjusted R² falls when a factor is
    added" is NOT a theorem — it falls only when the new factor's |t| is below
    roughly 1, and a random column can exceed that by chance."""
    result = fm.regress(noisy, factors, robust=False)
    n, k = result.n_observations, result.n_factors

    expected = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - k - 1)
    assert result.adj_r_squared == pytest.approx(expected)


def test_adjusted_r_squared_never_exceeds_the_raw_value(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """This one IS a theorem for k >= 1."""
    result = fm.regress(noisy, factors, robust=False)

    assert result.adj_r_squared <= result.r_squared


def test_raw_r_squared_cannot_fall_when_a_factor_is_added(factors: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED + 3)
    y = pd.Series(0.9 * factors["MKT"] + rng.normal(0.0, 0.006, N), index=factors.index)
    junk = pd.Series(rng.normal(0.0, 0.01, N), index=factors.index, name="JUNK")

    lean = fm.regress(y, factors[["MKT"]], robust=False)
    padded = fm.regress(y, factors[["MKT"]].assign(JUNK=junk), robust=False)

    assert padded.r_squared >= lean.r_squared - 1e-12


def test_a_factor_orthogonal_to_the_target_lowers_adjusted_r_squared(
    factors: pd.DataFrame,
) -> None:
    """The deterministic version of "adjusted R² penalises padding".

    A merely random column is not enough: 25 of them can still raise adjusted
    R² by explaining part of the residual through luck. So the added factor is
    made EXACTLY orthogonal to the target and to the existing design by
    Gram-Schmidt. Its beta is then exactly zero, its contribution to R² exactly
    zero, and the degrees-of-freedom penalty has to bite. A theorem, not a
    coin flip."""
    rng = np.random.default_rng(SEED + 3)
    y = pd.Series(0.9 * factors["MKT"] + rng.normal(0.0, 0.006, N), index=factors.index)

    # Project a random vector off the span of {1, MKT, y}.
    basis = np.column_stack([np.ones(N), factors["MKT"].to_numpy(), y.to_numpy()])
    candidate = rng.normal(0.0, 0.01, N)
    coef, *_ = np.linalg.lstsq(basis, candidate, rcond=None)
    orthogonal = candidate - basis @ coef

    design = factors[["MKT"]].assign(ORTHO=orthogonal)
    lean = fm.regress(y, factors[["MKT"]], robust=False)
    padded = fm.regress(y, design, robust=False)

    assert padded.loading("ORTHO").beta == pytest.approx(0.0, abs=1e-12)
    assert padded.r_squared == pytest.approx(lean.r_squared, abs=1e-12), "no R² gained"
    assert padded.adj_r_squared < lean.adj_r_squared, "so adjusted R² must fall"


def test_the_dof_penalty_grows_with_the_number_of_factors(
    factors: pd.DataFrame,
) -> None:
    """At a FIXED fit, more factors must mean a larger gap between raw and
    adjusted R². This is the penalty itself, isolated from any randomness."""
    rng = np.random.default_rng(SEED + 11)
    y = pd.Series(0.9 * factors["MKT"] + rng.normal(0.0, 0.006, N), index=factors.index)

    gaps = []
    for extra in (0, 5, 20):
        design = factors[["MKT"]].copy()
        for i in range(extra):
            design[f"X{i}"] = rng.normal(0.0, 0.01, N)
        result = fm.regress(y, design, robust=False)
        gaps.append(result.r_squared - result.adj_r_squared)

    assert gaps == sorted(gaps), gaps
    assert gaps[-1] > gaps[0]


def test_r_squared_is_bounded(noisy: pd.Series, factors: pd.DataFrame) -> None:
    result = fm.regress(noisy, factors)

    assert 0.0 <= result.r_squared <= 1.0
    assert result.explained_pct + result.unexplained_pct == pytest.approx(100.0)


def test_durbin_watson_is_near_two_for_iid_residuals(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    assert fm.regress(noisy, factors).durbin_watson == pytest.approx(2.0, abs=0.25)


def test_durbin_watson_collapses_for_autocorrelated_residuals() -> None:
    """The diagnostic must actually diagnose."""
    idx = _index()
    factor = pd.Series(np.random.default_rng(4).normal(0, 0.01, N), index=idx, name="F")
    y = 0.5 * factor + pd.Series(_ar1(0.9, 0.004, 5), index=idx)

    assert fm.regress(y, factor.to_frame()).durbin_watson < 0.75


def test_the_degrees_of_freedom_are_n_minus_k_minus_one(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    result = fm.regress(noisy, factors)

    assert result.dof == result.n_observations - result.n_factors - 1


def test_the_window_endpoints_are_reported(noisy: pd.Series, factors: pd.DataFrame) -> None:
    result = fm.regress(noisy, factors)

    assert result.start == factors.index[0]
    assert result.end == factors.index[-1]


# ---------------------------------------------------------------------------
# collinearity
# ---------------------------------------------------------------------------
def test_orthogonal_factors_have_a_variance_inflation_factor_near_one(
    factors: pd.DataFrame,
) -> None:
    vifs = fm.variance_inflation_factors(factors)

    for name, value in vifs.items():
        assert value == pytest.approx(1.0, abs=0.15), f"{name} = {value}"


def test_a_single_factor_has_no_collinearity(factors: pd.DataFrame) -> None:
    assert fm.variance_inflation_factors(factors[["MKT"]]) == {"MKT": 1.0}


def test_a_near_duplicate_factor_has_a_huge_variance_inflation_factor(
    factors: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(SEED + 4)
    almost = factors["MKT"] + rng.normal(0.0, 1e-5, N)
    design = factors.assign(MKT2=almost)

    vifs = fm.variance_inflation_factors(design)

    assert vifs["MKT"] > 50.0
    assert vifs["MKT2"] > 50.0


def test_collinearity_is_reported_on_the_result(factors: pd.DataFrame) -> None:
    """A beta shown without its collinearity is misleading by omission."""
    rng = np.random.default_rng(SEED + 5)
    design = factors.assign(MKT2=factors["MKT"] + rng.normal(0.0, 1e-4, N))
    y = pd.Series(0.7 * factors["MKT"] + rng.normal(0, 0.004, N), index=factors.index)

    result = fm.regress(y, design)

    assert "MKT" in result.collinear_factors
    assert any("variance inflation" in w for w in result.warnings)


def test_perfectly_collinear_factors_are_refused_with_an_explanation(
    factors: pd.DataFrame, noisy: pd.Series
) -> None:
    """Two identical factors is a user error the UI must be able to explain."""
    design = factors.assign(COPY=factors["MKT"])

    result = fm.regress(noisy, design)

    assert not result.ok
    assert any("rank-deficient" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_too_few_observations_is_refused_with_the_count_needed(
    factors: pd.DataFrame,
) -> None:
    short = factors.iloc[:3]
    y = pd.Series([0.01, -0.02, 0.005], index=short.index)

    result = fm.regress(y, short)

    assert not result.ok
    assert any("cannot support" in w for w in result.warnings)


def test_a_short_but_estimable_window_is_flagged_as_indicative(
    factors: pd.DataFrame,
) -> None:
    """Estimated, but honestly labelled — not silently presented as solid."""
    window = factors.iloc[:12]
    rng = np.random.default_rng(SEED + 6)
    y = pd.Series(0.8 * window["MKT"] + rng.normal(0, 0.003, len(window)), index=window.index)

    result = fm.regress(y, window)

    assert result.ok
    assert not result.reliable
    assert any("indicative at best" in w for w in result.warnings)


def test_a_constant_factor_is_dropped_and_reported(factors: pd.DataFrame, noisy: pd.Series) -> None:
    design = factors.assign(FLAT=0.01)

    _y, prepared, warnings = fm.prepare(noisy, design)

    assert "FLAT" not in prepared.columns
    assert any("never vary" in w for w in warnings)


def test_non_overlapping_dates_are_refused_with_a_readable_reason(
    factors: pd.DataFrame,
) -> None:
    """Regressing across a calendar mismatch measures the holiday schedule."""
    elsewhere = pd.Series(
        np.random.default_rng(7).normal(0, 0.01, 50),
        index=pd.bdate_range("2010-01-04", periods=50),
    )

    result = fm.factor_regression(elsewhere, factors)

    assert not result.ok
    assert any("share no dates" in w for w in result.warnings)


def test_partial_overlap_is_used_and_the_loss_reported(factors: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED + 7)
    longer_index = pd.bdate_range("2021-01-04", periods=N + 200)
    longer = pd.Series(rng.normal(0.0005, 0.01, N + 200), index=longer_index)

    result = fm.factor_regression(longer, factors)

    assert result.ok
    assert result.n_observations <= N
    assert any("overlap" in w for w in result.warnings)


@pytest.mark.parametrize(
    "returns,design",
    [
        (None, None),
        (pd.Series(dtype="float64"), pd.DataFrame()),
    ],
)
def test_empty_input_is_refused_rather_than_raising(returns: object, design: object) -> None:
    result = fm.factor_regression(returns, design)

    assert not result.ok
    assert result.warnings


def test_an_unusable_result_reports_nan_not_zero() -> None:
    result = fm.factor_regression(None, None)

    assert math.isnan(result.alpha)
    assert math.isnan(result.r_squared)
    assert result.loadings == ()
    assert result.betas() == {}


def test_a_missing_market_factor_name_is_reported(factors: pd.DataFrame, noisy: pd.Series) -> None:
    _y, _x, warnings = fm.prepare(noisy, factors, market_factor="Mkt-RF")

    assert any("Mkt-RF" in w for w in warnings)


# ---------------------------------------------------------------------------
# loadings & presentation
# ---------------------------------------------------------------------------
def test_the_summary_frame_puts_alpha_first(noisy: pd.Series, factors: pd.DataFrame) -> None:
    frame = fm.regress(noisy, factors).summary_frame()

    assert frame["Term"].iloc[0].startswith("Alpha")
    assert list(frame["Term"])[1:] == ["MKT", "SMB"]


def test_the_summary_frame_has_stable_columns(noisy: pd.Series, factors: pd.DataFrame) -> None:
    frame = fm.regress(noisy, factors).summary_frame()

    assert list(frame.columns) == ["Term", "Estimate", "Std error", "t-stat", "p-value", "VIF"]


def test_a_confidence_interval_brackets_the_estimate(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    result = fm.regress(noisy, factors)
    load = result.loading("MKT")

    low, high = load.confidence_interval(0.95, result.dof)
    assert low < load.beta < high


def test_a_confidence_interval_is_undefined_without_degrees_of_freedom() -> None:
    load = fm.FactorLoading("X", 0.5, 0.1)

    assert all(math.isnan(v) for v in load.confidence_interval(0.95, dof=0))


def test_an_unknown_loading_is_none(noisy: pd.Series, factors: pd.DataFrame) -> None:
    assert fm.regress(noisy, factors).loading("NOPE") is None


# ---------------------------------------------------------------------------
# rolling loadings
# ---------------------------------------------------------------------------
def test_rolling_betas_returns_one_row_per_window_end(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    frame = fm.rolling_betas(noisy, factors, window=100)

    assert not frame.empty
    assert len(frame) == N - 100 + 1
    assert list(frame.columns) == ["MKT", "SMB", "alpha_annual", "r_squared"]


def test_rolling_betas_track_a_deliberate_regime_change(factors: pd.DataFrame) -> None:
    """The point of the chart: a full-sample loading averages away drift."""
    first = 0.3 * factors["MKT"].iloc[: N // 2]
    second = 1.7 * factors["MKT"].iloc[N // 2 :]
    y = pd.concat([first, second])

    frame = fm.rolling_betas(y, factors[["MKT"]], window=80)

    assert frame["MKT"].iloc[0] == pytest.approx(0.3, abs=0.05)
    assert frame["MKT"].iloc[-1] == pytest.approx(1.7, abs=0.05)


def test_rolling_betas_of_too_short_a_series_is_empty(factors: pd.DataFrame) -> None:
    short = factors.iloc[:10]
    y = pd.Series(short["MKT"].to_numpy() * 0.5, index=short.index)

    assert fm.rolling_betas(y, short, window=200).empty


def test_rolling_betas_of_nothing_is_empty() -> None:
    assert fm.rolling_betas(None, None).empty


# ---------------------------------------------------------------------------
# contributions
# ---------------------------------------------------------------------------
def test_contributions_sum_to_the_fitted_annual_excess_return(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """The decomposition identity: betas times factor means, plus alpha."""
    result = fm.regress(noisy, factors)
    frame = fm.factor_contributions(result, factors)

    expected = result.alpha_annual + sum(
        load.beta * float(factors[load.name].mean()) * TRADING_DAYS for load in result.loadings
    )
    assert float(frame["contribution_annual"].sum()) == pytest.approx(expected)


def test_contributions_include_an_explicit_alpha_row(
    noisy: pd.Series, factors: pd.DataFrame
) -> None:
    """Alpha is part of the decomposition, not a residual to hide."""
    frame = fm.factor_contributions(fm.regress(noisy, factors), factors)

    assert "Alpha (unexplained)" in set(frame["factor"])


def test_contribution_shares_are_percentages(noisy: pd.Series, factors: pd.DataFrame) -> None:
    frame = fm.factor_contributions(fm.regress(noisy, factors), factors)

    assert float(frame["share_pct"].sum()) == pytest.approx(100.0, abs=1e-6)


def test_contributions_of_an_unusable_regression_are_empty(factors: pd.DataFrame) -> None:
    frame = fm.factor_contributions(fm.RegressionResult(), factors)

    assert frame.empty
    assert list(frame.columns) == [
        "factor",
        "beta",
        "mean_annual",
        "contribution_annual",
        "share_pct",
    ]


# ---------------------------------------------------------------------------
# proxy construction
# ---------------------------------------------------------------------------
@pytest.fixture
def proxy_prices() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 8)
    idx = _index(300)
    market = rng.normal(0.0004, 0.01, 300)
    return pd.DataFrame(
        {
            "VTI": 100.0 * np.cumprod(1.0 + market),
            "IWM": 90.0 * np.cumprod(1.0 + market + rng.normal(0, 0.004, 300)),
            "VTV": 80.0 * np.cumprod(1.0 + market + rng.normal(0, 0.003, 300)),
            "VUG": 120.0 * np.cumprod(1.0 + market + rng.normal(0, 0.003, 300)),
        },
        index=idx,
    )


def test_a_spread_factor_is_the_difference_of_two_returns(
    proxy_prices: pd.DataFrame,
) -> None:
    spec = FactorSpec("SMB", "Size", "IWM", "VTI")

    built, warnings = fm.build_proxy_factors(proxy_prices, [spec])

    expected = proxy_prices["IWM"].pct_change() - proxy_prices["VTI"].pct_change()
    assert warnings == []
    pd.testing.assert_series_equal(built["SMB"], expected.dropna().rename("SMB"), check_freq=False)


def test_a_market_factor_is_an_excess_return(proxy_prices: pd.DataFrame) -> None:
    spec = FactorSpec("MKT", "Market", "VTI", None)

    built, _warnings = fm.build_proxy_factors(proxy_prices, [spec], risk_free_rate=0.05)

    period_rf = fm.to_period_rate(0.05)
    expected = proxy_prices["VTI"].pct_change().dropna() - period_rf
    assert built["MKT"].to_numpy() == pytest.approx(expected.to_numpy())


def test_a_market_factor_with_no_rate_is_just_the_return(
    proxy_prices: pd.DataFrame,
) -> None:
    spec = FactorSpec("MKT", "Market", "VTI", None)

    built, _ = fm.build_proxy_factors(proxy_prices, [spec], risk_free_rate=0.0)

    assert built["MKT"].to_numpy() == pytest.approx(
        proxy_prices["VTI"].pct_change().dropna().to_numpy()
    )


def test_a_spec_whose_symbol_is_missing_is_skipped_and_reported(
    proxy_prices: pd.DataFrame,
) -> None:
    """A five-factor model that quietly became a two-factor model is a
    different model, so the omission must be visible."""
    specs = [
        FactorSpec("MKT", "Market", "VTI", None),
        FactorSpec("WML", "Momentum", "MTUM", "VTI"),
    ]

    built, warnings = fm.build_proxy_factors(proxy_prices, specs)

    assert list(built.columns) == ["MKT"]
    assert any("MTUM" in w for w in warnings)


def test_proxies_from_no_prices_are_refused(proxy_prices: pd.DataFrame) -> None:
    built, warnings = fm.build_proxy_factors(None, [FactorSpec("MKT", "M", "VTI", None)])

    assert built.empty
    assert warnings


def test_the_shipped_us_proxies_all_build(proxy_prices: pd.DataFrame) -> None:
    """Guards the config against naming a symbol the builder cannot use."""
    from src.config import FACTOR_PROXIES

    rng = np.random.default_rng(SEED + 9)
    symbols = sorted({s for spec in FACTOR_PROXIES["US"] for s in spec.symbols})
    prices = pd.DataFrame(
        {s: 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, 200)) for s in symbols},
        index=_index(200),
    )

    built, warnings = fm.build_proxy_factors(prices, FACTOR_PROXIES["US"])

    assert warnings == []
    assert list(built.columns) == [spec.name for spec in FACTOR_PROXIES["US"]]


def test_a_factor_spec_describes_its_own_formula() -> None:
    assert FactorSpec("SMB", "Size", "IWM", "VTI").formula == "IWM − VTI"
    assert FactorSpec("MKT", "Market", "VTI", None).formula == "VTI − risk-free"
    assert FactorSpec("MKT", "Market", "VTI", None).is_excess_return
    assert not FactorSpec("SMB", "Size", "IWM", "VTI").is_excess_return


# ---------------------------------------------------------------------------
# published factor files
# ---------------------------------------------------------------------------
def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.strip().encode("utf-8"))


FRENCH_STYLE = """
Date,Mkt-RF,SMB,HML,RF
20240102,-0.53,0.31,0.12,0.021
20240103,0.41,-0.18,0.05,0.021
20240104,0.12,0.09,-0.22,0.021
20240105,-0.31,0.14,0.31,0.021
"""


def test_a_french_style_file_parses_with_yyyymmdd_dates() -> None:
    frame, _warnings = fm.parse_factor_csv(_csv(FRENCH_STYLE))

    assert not frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index[0] == pd.Timestamp("2024-01-02")


def test_percent_values_are_detected_and_converted() -> None:
    """Getting this wrong scales every loading by 100 and looks like nothing."""
    frame, warnings = fm.parse_factor_csv(_csv(FRENCH_STYLE))

    assert frame["Mkt-RF"].iloc[0] == pytest.approx(-0.0053)
    assert any("PERCENT" in w for w in warnings)


def test_decimal_values_are_left_alone() -> None:
    body = "Date,Mkt-RF,SMB\n2024-01-02,-0.0053,0.0031\n2024-01-03,0.0041,-0.0018\n"

    frame, warnings = fm.parse_factor_csv(_csv(body))

    assert frame["Mkt-RF"].iloc[0] == pytest.approx(-0.0053)
    assert any("DECIMAL" in w for w in warnings)


def test_the_scale_decision_is_always_reported() -> None:
    """An invisible unit conversion is unacceptable; it is stated either way."""
    _frame, warnings = fm.parse_factor_csv(_csv(FRENCH_STYLE))

    assert warnings, "auto-detection must say what it decided"


@pytest.mark.parametrize("scale,expected", [("percent", -0.0053), ("decimal", -0.53)])
def test_the_scale_can_be_stated_explicitly(scale: str, expected: float) -> None:
    frame, _warnings = fm.parse_factor_csv(_csv(FRENCH_STYLE), scale=scale)

    assert frame["Mkt-RF"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "header,canonical",
    [("MF", "Mkt-RF"), ("mkt_rf", "Mkt-RF"), ("UMD", "WML"), ("MOM", "WML"), ("smb", "SMB")],
)
def test_published_column_aliases_are_normalised(header: str, canonical: str) -> None:
    """French writes "Mkt-RF"; the IIM Ahmedabad Indian dataset writes "MF"."""
    frame, _warnings = fm.parse_factor_csv(
        _csv(f"Date,{header}\n2024-01-02,0.0012\n2024-01-03,-0.0004\n")
    )

    assert canonical in frame.columns


def test_a_monthly_yyyymm_file_parses() -> None:
    frame, _warnings = fm.parse_factor_csv(_csv("Date,Mkt-RF\n202401,0.0123\n202402,-0.0045\n"))

    assert not frame.empty
    assert frame.index[0] == pd.Timestamp("2024-01-01")


def test_an_rf_column_is_preserved() -> None:
    """The caller may prefer the file's own risk-free series."""
    frame, _warnings = fm.parse_factor_csv(_csv(FRENCH_STYLE))

    assert "RF" in frame.columns


def test_a_file_with_unreadable_dates_is_refused() -> None:
    frame, warnings = fm.parse_factor_csv(_csv("Date,Mkt-RF\nwhenever,0.01\nlater,0.02\n"))

    assert frame.empty
    assert any("dates" in w for w in warnings)


def test_a_file_with_no_numeric_columns_is_refused() -> None:
    frame, warnings = fm.parse_factor_csv(_csv("Date,Note\n2024-01-02,hello\n"))

    assert frame.empty
    assert warnings


def test_an_unreadable_upload_is_refused_rather_than_raising() -> None:
    frame, warnings = fm.parse_factor_csv(io.BytesIO(b"\x00\x01\x02\xff"))

    assert frame.empty
    assert warnings


def test_a_dataframe_can_be_passed_directly() -> None:
    source = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "Mkt-RF": [0.001, -0.002]})

    frame, _warnings = fm.parse_factor_csv(source)

    assert "Mkt-RF" in frame.columns


def test_a_parsed_file_feeds_the_regression_end_to_end() -> None:
    """The whole point of the parser: rigorous factor data, straight in."""
    rng = np.random.default_rng(SEED + 10)
    idx = pd.bdate_range("2023-01-02", periods=200)
    body = pd.DataFrame(
        {
            "Date": idx.strftime("%Y%m%d"),
            "Mkt-RF": rng.normal(0.04, 1.0, 200),
            "SMB": rng.normal(0.0, 0.6, 200),
        }
    )
    frame, _warnings = fm.parse_factor_csv(body)
    y = pd.Series(0.9 * frame["Mkt-RF"] + rng.normal(0, 0.004, 200), index=frame.index)

    result = fm.factor_regression(
        y, frame.drop(columns=["Mkt-RF"]).assign(**{"Mkt-RF": frame["Mkt-RF"]})
    )

    assert result.ok
    assert result.loading("Mkt-RF") is not None


# ---------------------------------------------------------------------------
# rate conversion
# ---------------------------------------------------------------------------
def test_the_risk_free_rate_is_de_annualised_geometrically() -> None:
    """Matches src.analytics.risk; two conventions in one app is how Sharpe
    ratios stop reconciling."""
    daily = fm.to_period_rate(0.065, TRADING_DAYS)

    assert (1.0 + daily) ** TRADING_DAYS == pytest.approx(1.065)


def test_a_zero_rate_converts_to_zero() -> None:
    assert fm.to_period_rate(0.0) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, -2.0])
def test_an_unusable_rate_is_nan_not_zero(bad: float) -> None:
    assert math.isnan(fm.to_period_rate(bad))
