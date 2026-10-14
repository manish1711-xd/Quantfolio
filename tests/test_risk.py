"""Verification of ``src.analytics.risk`` -- the quantitative centre of QuantFolio.

Every number a QuantFolio risk report prints comes from this module, so the
tests below are written so that a reviewer can audit the arithmetic without
running anything: wherever a golden value appears, the hand computation that
produced it appears next to it in a comment.

Three kinds of assertion are used, deliberately:

1. **Hand-computed goldens.** ``max_drawdown([100, 120, 90, 130]) == -0.25``
   because 90/120 - 1 = -0.25. A test that re-derives the expected value with
   the same code path proves nothing at all.
2. **Mathematical identities**, which cannot rot as the implementation evolves:
   beta against itself is 1, CVaR <= VaR at every confidence level, the
   correlation diagonal is 1, ``HHI * effective_bets == 1``, shrinkage
   preserves the covariance trace, ``gain_to_pain == omega - 1`` at a zero MAR.
3. **Documented degradation.** The module contract is that nothing raises and
   nothing fabricates a zero: undefined statistics are NaN. Empty, single-row,
   all-NaN, zero-variance, unsorted and duplicated inputs each get a test that
   pins the NaN / empty / ``None`` answer rather than an exception.

Sign and unit conventions are treated as regression guards in their own right,
because reversing one of them would silently invert an entire risk report:
VaR and CVaR are NEGATIVE decimals describing a one-day loss, volatility is
annualised with sqrt(252), the Ulcer Index is in PERCENT, drawdowns are <= 0,
and the risk-free rate is an annualised decimal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import (
    DEFAULT_VAR_CONFIDENCE,
    MIN_VAR_OBSERVATIONS,
    TRADING_DAYS,
    VAR_METHODS,
    annualized_volatility,
    beta,
    calmar_ratio,
    conditional_var,
    correlation_matrix,
    covariance_matrix,
    daily_returns,
    diversification_ratio,
    downside_deviation,
    drawdown_series,
    effective_number_of_bets,
    excess_kurtosis,
    gain_to_pain,
    herfindahl_index,
    hit_ratio,
    kupiec_pof_test,
    martin_ratio,
    max_drawdown,
    omega_ratio,
    portfolio_value_series,
    risk_score,
    risk_summary,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    ulcer_index,
    value_at_risk,
)

# The 5% standard-normal quantile, hard-coded so the Gaussian-VaR tests do not
# simply re-run the same scipy call the implementation uses.
Z_95 = -1.6448536269514722
#: phi(z_95) / 0.05 -- the textbook standard-normal expected-shortfall factor.
#: phi(Phi^-1(0.05)) / 0.05 -- the textbook 95% expected-shortfall multiplier.
#: Written out to full double precision on purpose: a value rounded at the 7th
#: digit silently fails a 1e-8 comparison and looks like an implementation bug.
ES_FACTOR_95 = 2.0627128075074253

DATE_INDEX = pd.bdate_range("2023-01-02", periods=4)


# ---------------------------------------------------------------------------
# local, deterministic samples (the shared fixtures cover everything else)
# ---------------------------------------------------------------------------
def ramp_returns(n: int = 1000) -> pd.Series:
    """Returns spaced exactly evenly from -5% to +5%.

    Chosen because every percentile of it is computable by hand: element ``i``
    is ``-0.05 + i * (0.10 / 999)``, so the numpy linear-interpolated 5th
    percentile lands on ``-0.05 + 49.95 * 0.10/999 == -0.045`` exactly.
    """
    return pd.Series(np.linspace(-0.05, 0.05, n))


def normal_returns(n: int = 1000, mu: float = 0.0005, sigma: float = 0.01) -> pd.Series:
    """A seeded Gaussian sample: near-zero skew and near-zero excess kurtosis."""
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(mu, sigma, n))


def mirrored_returns(n: int = 500, sigma: float = 0.01) -> pd.Series:
    """``[x, -x]`` -- exactly symmetric about zero, so skew is exactly 0."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, sigma, n)
    return pd.Series(np.concatenate([x, -x]))


@pytest.fixture
def port_returns(portfolio_values: pd.Series) -> pd.Series:
    """Daily returns of the shared portfolio value series (499 observations)."""
    return daily_returns(portfolio_values)


@pytest.fixture
def four_day_values() -> pd.Series:
    """100 -> 120 -> 90 -> 130 on a business-day index. Peak 120, trough 90."""
    return pd.Series([100.0, 120.0, 90.0, 130.0], index=DATE_INDEX)


# ---- daily returns -------------------------------------------------------
def test_daily_returns_of_100_110_121_are_exactly_ten_percent_each():
    # 110/100 - 1 = 0.10 ; 121/110 - 1 = 0.10
    result = daily_returns(pd.Series([100.0, 110.0, 121.0]))
    assert list(result) == pytest.approx([0.10, 0.10], abs=1e-12)


def test_daily_returns_forward_fills_gaps_instead_of_reading_them_as_a_total_loss():
    # [100, NaN, 110] ffills to [100, 100, 110]:
    #   100/100 - 1 = 0.00   (a holiday is a flat day, NOT a -100% day)
    #   110/100 - 1 = 0.10
    result = daily_returns(pd.Series([100.0, np.nan, 110.0]))
    assert list(result) == pytest.approx([0.0, 0.10], abs=1e-12)


def test_daily_returns_drops_the_undefined_first_row(prices: pd.DataFrame):
    # r_1 = P_1/P_0 - 1 needs a predecessor, so n prices give n-1 returns.
    result = daily_returns(prices)
    assert len(result) == len(prices) - 1
    assert list(result.columns) == list(prices.columns)
    assert not result.isna().to_numpy().any()


def test_daily_returns_of_none_degrades_to_an_empty_series():
    assert daily_returns(None).empty


def test_daily_returns_does_not_mutate_the_caller(prices: pd.DataFrame):
    before = prices.copy(deep=True)
    daily_returns(prices)
    pd.testing.assert_frame_equal(prices, before)


def test_daily_returns_of_a_single_row_is_empty(single_value_series: pd.Series):
    assert daily_returns(single_value_series).empty


# ---- portfolio value series ----------------------------------------------
def test_portfolio_value_series_is_the_quantity_weighted_sum(prices: pd.DataFrame):
    quantities = {"AAA": 10.0, "BBB": 5.0, "CCC": 8.0}
    values = portfolio_value_series(prices, quantities)
    row = prices.iloc[0]
    # V_0 = 10*P(AAA,0) + 5*P(BBB,0) + 8*P(CCC,0)
    expected = 10.0 * row["AAA"] + 5.0 * row["BBB"] + 8.0 * row["CCC"]
    assert values.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert len(values) == len(prices)
    assert values.name == "portfolio_value"


def test_portfolio_value_series_ignores_tickers_absent_from_the_price_frame(
    prices: pd.DataFrame,
):
    with_ghost = portfolio_value_series(prices, {"AAA": 10.0, "NOT_LISTED": 999.0})
    only_real = portfolio_value_series(prices, {"AAA": 10.0})
    pd.testing.assert_series_equal(with_ghost, only_real)


def test_portfolio_value_series_is_empty_when_no_ticker_overlaps(prices: pd.DataFrame):
    assert portfolio_value_series(prices, {"NOT_LISTED": 1.0}).empty


def test_portfolio_value_series_is_empty_for_an_empty_price_frame():
    assert portfolio_value_series(pd.DataFrame(), {"AAA": 1.0}).empty


def test_portfolio_value_series_is_linear_in_quantities(prices: pd.DataFrame):
    # Identity: doubling every share count doubles the value at every date.
    single = portfolio_value_series(prices, {"AAA": 10.0, "BBB": 5.0})
    double = portfolio_value_series(prices, {"AAA": 20.0, "BBB": 10.0})
    assert double.to_numpy() == pytest.approx(2.0 * single.to_numpy(), rel=1e-12)


# ---- volatility, Sharpe, Sortino, Calmar, Omega --------------------------
def test_annualized_volatility_is_sample_std_times_sqrt_252():
    # r = [0.01, -0.01, 0.02, -0.02]; mean = 0
    # sum of squares = 1e-4 + 1e-4 + 4e-4 + 4e-4 = 1.0e-3
    # var(ddof=1) = 1.0e-3 / 3 ; std = 0.018257418...
    # sigma = std * sqrt(252) = 0.018257418 * 15.8745079 = 0.28982753
    result = annualized_volatility(pd.Series([0.01, -0.01, 0.02, -0.02]))
    assert result == pytest.approx(math.sqrt(1.0e-3 / 3.0) * math.sqrt(252.0), rel=1e-12)
    assert result == pytest.approx(0.28982753, abs=1e-8)


def test_annualized_volatility_uses_the_252_day_convention(port_returns: pd.Series):
    # Regression guard: switching to 260 (calendar weeks) or 365 would move
    # every Sharpe ratio in the product by ~2% and ~20% respectively.
    assert TRADING_DAYS == 252
    expected = float(port_returns.std(ddof=1)) * math.sqrt(252.0)
    assert annualized_volatility(port_returns) == pytest.approx(expected, rel=1e-12)


def test_annualized_volatility_uses_the_bessel_corrected_sample_std():
    # Population std of [0.01, -0.01] is 0.01; the SAMPLE std (ddof=1) is
    # sqrt(2e-4/1) = 0.0141421356 -- a factor sqrt(2) larger for n = 2.
    result = annualized_volatility(pd.Series([0.01, -0.01]))
    assert result == pytest.approx(0.01 * math.sqrt(2.0) * math.sqrt(252.0), rel=1e-12)


def test_constant_returns_have_zero_volatility_and_a_nan_sharpe(flat_series: pd.Series):
    returns = daily_returns(flat_series)
    vol = annualized_volatility(returns)
    assert vol == 0.0
    # No ZeroDivisionError, no infinite Sharpe: a riskless series has an
    # UNDEFINED risk-adjusted return, and NaN is how the module says so.
    assert math.isnan(sharpe_ratio(0.05, vol, 0.0))


def test_annualized_volatility_needs_at_least_two_observations():
    assert math.isnan(annualized_volatility(pd.Series([0.01])))
    assert math.isnan(annualized_volatility(pd.Series(dtype=float)))


def test_sharpe_ratio_is_excess_return_over_volatility():
    # (0.12 - 0.06) / 0.20 = 0.06 / 0.20 = 0.30
    assert sharpe_ratio(0.12, 0.20, 0.06) == pytest.approx(0.30, abs=1e-12)


def test_sharpe_ratio_is_nan_rather_than_infinite_for_zero_volatility():
    assert math.isnan(sharpe_ratio(0.12, 0.0, 0.06))


def test_sharpe_ratio_is_nan_for_non_finite_inputs():
    assert math.isnan(sharpe_ratio(float("nan"), 0.20, 0.06))
    assert math.isnan(sharpe_ratio(0.12, float("nan"), 0.06))
    assert math.isnan(sharpe_ratio(0.12, 0.20, float("nan")))


def test_downside_deviation_is_hand_computable():
    # r = [0.02, -0.01, 0.03, -0.02], MAR = 0 so mar_daily = 0 exactly.
    # shortfalls  min(r, 0)  = [0, -0.01, 0, -0.02]
    # squares                = [0, 1e-4, 0, 4e-4] -> sum 5e-4
    # mean over ALL FOUR obs = 1.25e-4 ; sqrt = 0.0111803399
    # DD = 0.0111803399 * sqrt(252) = 0.17748239
    result = downside_deviation(pd.Series([0.02, -0.01, 0.03, -0.02]))
    assert result == pytest.approx(math.sqrt(5.0e-4 / 4.0) * math.sqrt(252.0), rel=1e-12)
    assert result == pytest.approx(0.17748239, abs=1e-8)


def test_downside_deviation_is_at_most_total_volatility(port_returns: pd.Series):
    # The Sortino denominator only counts losses, and it averages them over
    # every observation, so for a rising series it must sit below total vol.
    assert downside_deviation(port_returns) <= annualized_volatility(port_returns)


def test_downside_deviation_is_zero_when_nothing_falls_below_the_mar():
    # Honest answer: there was no measured downside. Not NaN, not a fudge.
    assert downside_deviation(pd.Series([0.01] * 30)) == 0.0


def test_sortino_exceeds_sharpe_when_downside_deviation_is_below_volatility(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    from src.analytics.returns import annualized_return

    ann_ret = annualized_return(portfolio_values)
    vol = annualized_volatility(port_returns)
    dev = downside_deviation(port_returns)
    assert ann_ret > 0.0 and dev < vol  # premise of the identity
    assert sortino_ratio(ann_ret, dev, 0.0) >= sharpe_ratio(ann_ret, vol, 0.0)


def test_sortino_is_nan_when_there_was_no_downside():
    dev = downside_deviation(pd.Series([0.01] * 30))
    assert math.isnan(sortino_ratio(0.20, dev, 0.0))


def test_calmar_is_annual_return_over_the_absolute_drawdown():
    # 0.15 / |-0.25| = 0.60
    assert calmar_ratio(0.15, -0.25) == pytest.approx(0.60, abs=1e-12)


def test_calmar_is_nan_for_an_undrawn_series():
    assert math.isnan(calmar_ratio(0.15, 0.0))


def test_martin_ratio_converts_the_percent_ulcer_index_to_a_decimal():
    # (0.20 - 0.06) / (10.0 / 100) = 0.14 / 0.10 = 1.4
    # Forgetting the /100 would report 0.014 -- wrong by two orders of magnitude.
    assert martin_ratio(0.20, 10.0, 0.06) == pytest.approx(1.4, abs=1e-12)


def test_martin_ratio_is_nan_for_a_zero_ulcer_index():
    assert math.isnan(martin_ratio(0.20, 0.0, 0.06))


def test_omega_of_a_symmetric_sample_at_a_zero_mar_is_one():
    # MAR = 0 -> mar_daily = (1+0)**(1/252) - 1 = 0 exactly, so Omega is
    # sum(gains)/sum(losses); for the mirrored sample [x, -x] both legs are
    # sum(|x|) and the ratio is exactly 1.
    assert omega_ratio(mirrored_returns(), 0.0) == pytest.approx(1.0, abs=1e-12)


def test_omega_exceeds_one_for_a_series_with_more_upside_than_downside():
    # gains 0.03 + 0.02 = 0.05 ; losses 0.01 + 0.01 = 0.02 ; Omega = 2.5
    r = pd.Series([0.03, -0.01, 0.02, -0.01])
    assert omega_ratio(r, 0.0) == pytest.approx(2.5, abs=1e-12)


def test_omega_is_nan_when_nothing_fell_below_the_threshold():
    assert math.isnan(omega_ratio(pd.Series([0.01] * 30), 0.0))


def test_gain_to_pain_is_omega_minus_one_at_a_zero_mar(port_returns: pd.Series):
    # Identity: sum(r) = gains - pains and |sum(min(r,0))| = pains, so
    # GPR = (gains - pains)/pains = Omega - 1 whenever the threshold is zero.
    assert gain_to_pain(port_returns) == pytest.approx(
        omega_ratio(port_returns, 0.0) - 1.0, rel=1e-9
    )


def test_gain_to_pain_is_nan_when_there_were_no_losses():
    assert math.isnan(gain_to_pain(pd.Series([0.01] * 10)))


# ---- drawdown ------------------------------------------------------------
def test_max_drawdown_of_100_120_90_130_is_exactly_minus_25_percent(
    four_day_values: pd.Series,
):
    # Running peak at the trough is 120; 90/120 - 1 = -0.25
    assert max_drawdown(four_day_values) == pytest.approx(-0.25, abs=1e-12)


def test_max_drawdown_is_zero_for_monotonic_growth():
    # Every day is a new high, so the running peak IS the current value.
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0, 130.0])) == 0.0


def test_max_drawdown_is_nan_for_an_empty_series(empty_series: pd.Series):
    assert math.isnan(max_drawdown(empty_series))


def test_max_drawdown_is_nan_for_an_all_nan_series(nan_series: pd.Series):
    assert math.isnan(max_drawdown(nan_series))


def test_drawdown_series_starts_at_zero_and_is_never_positive(
    portfolio_values: pd.Series,
):
    dd = drawdown_series(portfolio_values)
    assert dd.iloc[0] == 0.0  # the first point is its own running peak
    assert (dd.dropna() <= 0.0).all()  # d_t = V_t/max(V) - 1 <= 0 by construction


def test_drawdown_series_is_hand_computable(four_day_values: pd.Series):
    # peaks  = [100, 120, 120, 130]
    # d      = [100/100-1, 120/120-1, 90/120-1, 130/130-1]
    assert list(drawdown_series(four_day_values)) == pytest.approx(
        [0.0, 0.0, -0.25, 0.0], abs=1e-12
    )


def test_max_drawdown_is_the_minimum_of_the_drawdown_series(
    portfolio_values: pd.Series,
):
    dd = drawdown_series(portfolio_values)
    assert max_drawdown(portfolio_values) == pytest.approx(float(dd.min()), rel=1e-12)


def test_drawdown_series_is_empty_for_an_empty_series(empty_series: pd.Series):
    assert drawdown_series(empty_series).empty


def test_drawdown_of_a_single_observation_is_zero(single_value_series: pd.Series):
    assert list(drawdown_series(single_value_series)) == [0.0]
    assert max_drawdown(single_value_series) == 0.0


def test_drawdown_is_nan_rather_than_infinite_while_the_peak_is_non_positive():
    # A series opening at 0 has no meaningful high-water mark on day one;
    # 0/0 must read as a blank, never as +/-inf or a fabricated number.
    dd = drawdown_series(pd.Series([0.0, 50.0, 100.0]))
    assert math.isnan(dd.iloc[0])
    assert list(dd.iloc[1:]) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert np.isfinite(dd.dropna()).all()


def test_max_drawdown_is_independent_of_row_order_on_a_date_index(
    four_day_values: pd.Series,
):
    # An unsorted DatetimeIndex is sorted before measurement; otherwise a CSV
    # in descending date order would report a drawdown that never happened.
    shuffled = four_day_values.iloc[[2, 0, 3, 1]]
    assert max_drawdown(shuffled) == pytest.approx(-0.25, abs=1e-12)


def test_duplicate_index_entries_do_not_raise(four_day_values: pd.Series):
    duplicated = pd.Series(
        four_day_values.to_numpy(),
        index=[DATE_INDEX[0], DATE_INDEX[0], DATE_INDEX[1], DATE_INDEX[1]],
    )
    assert max_drawdown(duplicated) == pytest.approx(-0.25, abs=1e-12)
    assert len(drawdown_series(duplicated)) == 4


def test_ulcer_index_is_zero_for_a_monotonically_rising_series():
    assert ulcer_index(pd.Series([100.0, 110.0, 120.0, 130.0])) == 0.0


def test_ulcer_index_is_hand_computable(four_day_values: pd.Series):
    # drawdowns in PERCENT = [0, 0, -25, 0]
    # mean of squares = (0 + 0 + 625 + 0) / 4 = 156.25 ; sqrt = 12.5
    assert ulcer_index(four_day_values) == pytest.approx(12.5, abs=1e-12)


def test_ulcer_index_is_positive_whenever_a_drawdown_occurred(
    portfolio_values: pd.Series,
):
    assert max_drawdown(portfolio_values) < 0.0
    assert ulcer_index(portfolio_values) > 0.0


def test_ulcer_index_is_nan_for_an_empty_series(empty_series: pd.Series):
    assert math.isnan(ulcer_index(empty_series))


# ---- tail risk: VaR and CVaR ---------------------------------------------
@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        # linspace(-0.05, 0.05, 1000): element i = -0.05 + i*(0.10/999)
        # numpy's linear percentile at q lands on index q/100 * 999:
        #   q=10 -> i =  99.90 -> -0.05 +  9.99/999 = -0.040
        #   q= 5 -> i =  49.95 -> -0.05 +  4.995/999 = -0.045
        #   q= 1 -> i =   9.99 -> -0.05 +  0.999/999 = -0.049
        (0.90, -0.040),
        (0.95, -0.045),
        (0.99, -0.049),
    ],
)
def test_historical_var_is_the_empirical_loss_quantile(confidence, expected):
    assert value_at_risk(ramp_returns(), confidence, "historical") == pytest.approx(
        expected, abs=1e-9
    )


def test_value_at_risk_is_a_negative_decimal(port_returns: pd.Series):
    # SIGN REGRESSION GUARD. -0.021 means "a 2.1% loss". If this ever returns
    # +0.021 every VaR chart, every CVaR and the whole tail leg of the risk
    # score silently flip sign.
    var = value_at_risk(port_returns, 0.95, "historical")
    assert var < 0.0
    assert -1.0 < var < 0.0  # a decimal, never a percent


def test_var_deepens_monotonically_with_confidence(port_returns: pd.Series):
    v90 = value_at_risk(port_returns, 0.90)
    v95 = value_at_risk(port_returns, 0.95)
    v99 = value_at_risk(port_returns, 0.99)
    assert v99 <= v95 <= v90


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("method", ["historical", "gaussian", "cornish_fisher"])
def test_cvar_is_never_better_than_var(port_returns: pd.Series, confidence: float, method: str):
    # CVaR averages the tail BEYOND the quantile, so it can never be less bad
    # than the quantile itself. This is the coherence property that makes CVaR
    # sub-additive and VaR not.
    var = value_at_risk(port_returns, confidence, method)
    cvar = conditional_var(port_returns, confidence, method)
    assert math.isfinite(var) and math.isfinite(cvar)
    assert cvar <= var


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        # Mean of the elements at or below the historical threshold.
        # q=10 -> 100 elements (i = 0..99): -0.05 + (0.10/999)*49.5 = -0.045045045
        # q= 5 ->  50 elements (i = 0..49): -0.05 + (0.10/999)*24.5 = -0.047547548
        # q= 1 ->  10 elements (i = 0.. 9): -0.05 + (0.10/999)* 4.5 = -0.049549550
        (0.90, -0.05 + (0.10 / 999.0) * 49.5),
        (0.95, -0.05 + (0.10 / 999.0) * 24.5),
        (0.99, -0.05 + (0.10 / 999.0) * 4.5),
    ],
)
def test_historical_cvar_is_the_mean_of_the_breaching_tail(confidence, expected):
    assert conditional_var(ramp_returns(), confidence, "historical") == pytest.approx(
        expected, abs=1e-9
    )


def test_gaussian_var_of_a_normal_sample_is_close_to_mu_plus_z_sigma():
    # Closed form: VaR = mu + z*sigma with z = Phi^-1(0.05) = -1.6448536.
    # The sample's own mu/sigma differ from the generating parameters by
    # O(sigma/sqrt(n)) ~ 3e-4, hence the 1e-3 tolerance on n = 1000 draws.
    sample = normal_returns(n=1000, mu=0.0005, sigma=0.01)
    assert value_at_risk(sample, 0.95, "gaussian") == pytest.approx(0.0005 + Z_95 * 0.01, abs=1e-3)


def test_gaussian_var_matches_the_closed_form_on_the_samples_own_moments():
    sample = normal_returns()
    mu = float(np.mean(sample.to_numpy()))
    sigma = float(np.std(sample.to_numpy(), ddof=1))
    assert value_at_risk(sample, 0.95, "gaussian") == pytest.approx(mu + Z_95 * sigma, rel=1e-9)


def test_gaussian_cvar_is_the_expected_shortfall_closed_form():
    # ES = mu - sigma * phi(z)/(1-c); at 95% the factor phi(-1.6449)/0.05 is
    # the textbook constant 2.0627134.
    sample = normal_returns()
    mu = float(np.mean(sample.to_numpy()))
    sigma = float(np.std(sample.to_numpy(), ddof=1))
    assert conditional_var(sample, 0.95, "gaussian") == pytest.approx(
        mu - sigma * ES_FACTOR_95, rel=1e-8
    )


def test_cornish_fisher_equals_gaussian_when_skew_and_excess_kurtosis_vanish():
    # z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36 collapses to z
    # when S = K = 0. On a Gaussian sample both moments are within ~0.03 of
    # zero, so the two quantiles must agree to a few basis points of sigma.
    sample = normal_returns(n=1000, mu=0.0005, sigma=0.01)
    assert abs(skewness(sample)) < 0.05
    assert abs(excess_kurtosis(sample)) < 0.05
    gaussian = value_at_risk(sample, 0.95, "gaussian")
    cornish = value_at_risk(sample, 0.95, "cornish_fisher")
    assert cornish == pytest.approx(gaussian, abs=1e-4)


def test_cornish_fisher_digs_deeper_than_gaussian_for_a_negatively_skewed_sample():
    # Eight moderately bad days in an otherwise mild sample: S ~ -1.6 and
    # excess K ~ 6.3, which is realistic for equities and comfortably inside
    # the expansion's domain. (z^2-1)*S/6 < 0 then pushes the quantile further
    # into the loss tail -- exactly the correction that makes CF worth having.
    base = normal_returns(n=400, mu=0.0, sigma=0.005).to_numpy()
    skewed = pd.Series(np.concatenate([base, [-0.03] * 8]))
    assert skewness(skewed) < -1.0
    assert excess_kurtosis(skewed) < 10.0  # still inside the valid domain
    assert value_at_risk(skewed, 0.99, "cornish_fisher") < value_at_risk(skewed, 0.99, "gaussian")


def test_cornish_fisher_falls_back_when_the_expansion_stops_being_a_quantile():
    """A single catastrophic day pushes CF outside its domain of validity.

    Cornish-Fisher is an asymptotic expansion in the third and fourth moments.
    One -20% day among 400 quiet ones gives skew ~ -14 and excess kurtosis
    ~ 250, at which point the polynomial is no longer monotone in z -- it stops
    being a quantile transform and, evaluated naively, reports a *gain* as the
    99% worst case. The implementation must detect that and fall back to the
    assumption-free historical quantile rather than print nonsense.
    """
    base = normal_returns(n=400, mu=0.0, sigma=0.005).to_numpy()
    pathological = pd.Series(np.concatenate([base, [-0.20]]))
    assert skewness(pathological) < -5.0
    assert excess_kurtosis(pathological) > 100.0

    cf = value_at_risk(pathological, 0.99, "cornish_fisher")
    assert cf < 0.0, "a 99% VaR must be a loss, never a gain"
    assert cf == pytest.approx(value_at_risk(pathological, 0.99, "historical"))


def test_var_is_nan_below_the_minimum_observation_floor():
    # MIN_VAR_OBSERVATIONS = MIN_OBSERVATIONS // 2 = 15. A tail estimated from
    # fewer points is noise, and the module reports noise as NaN.
    assert MIN_VAR_OBSERVATIONS == 15
    too_few = pd.Series(np.linspace(-0.01, 0.01, MIN_VAR_OBSERVATIONS - 1))
    just_enough = pd.Series(np.linspace(-0.01, 0.01, MIN_VAR_OBSERVATIONS))
    assert math.isnan(value_at_risk(too_few))
    assert math.isnan(conditional_var(too_few))
    assert math.isfinite(value_at_risk(just_enough))


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_var_is_nan_for_a_confidence_outside_the_open_unit_interval(confidence):
    assert math.isnan(value_at_risk(ramp_returns(), confidence))
    assert math.isnan(conditional_var(ramp_returns(), confidence))


def test_var_is_nan_for_an_unrecognised_method_instead_of_raising():
    assert math.isnan(value_at_risk(ramp_returns(), 0.95, "monte-carlo"))
    assert math.isnan(conditional_var(ramp_returns(), 0.95, "monte-carlo"))


def test_var_of_a_zero_variance_series_is_zero_not_a_division_error(
    flat_series: pd.Series,
):
    returns = daily_returns(flat_series)
    assert value_at_risk(returns, 0.95, "historical") == 0.0
    assert value_at_risk(returns, 0.95, "gaussian") == 0.0
    assert conditional_var(returns, 0.95, "historical") == 0.0


def test_var_methods_constant_names_exactly_the_supported_estimators():
    assert VAR_METHODS == ("historical", "gaussian", "cornish_fisher")
    for method in VAR_METHODS:
        assert math.isfinite(value_at_risk(ramp_returns(), 0.95, method))


# ---- distribution shape --------------------------------------------------
def test_skewness_of_a_mirrored_sample_is_zero():
    assert skewness(mirrored_returns()) == pytest.approx(0.0, abs=1e-12)


def test_skewness_needs_three_observations():
    assert math.isnan(skewness(pd.Series([0.01, 0.02])))


def test_excess_kurtosis_of_a_gaussian_sample_is_near_zero():
    # Fisher definition: the normal distribution scores 0, not 3.
    assert excess_kurtosis(normal_returns()) == pytest.approx(0.0, abs=0.2)


def test_excess_kurtosis_is_large_and_positive_for_a_fat_tailed_sample():
    base = normal_returns(n=400, mu=0.0, sigma=0.005).to_numpy()
    fat = pd.Series(np.concatenate([base, [-0.20, 0.20]]))
    assert excess_kurtosis(fat) > 5.0


def test_excess_kurtosis_needs_four_observations():
    assert math.isnan(excess_kurtosis(pd.Series([0.01, 0.02, 0.03])))


def test_tail_ratio_of_a_symmetric_sample_is_one():
    # |q95| / |q5| with q95 = -q5 for an exactly symmetric sample.
    assert tail_ratio(mirrored_returns()) == pytest.approx(1.0, abs=1e-9)


def test_tail_ratio_is_nan_below_the_minimum_observation_floor():
    assert math.isnan(tail_ratio(pd.Series([0.01] * 10)))


def test_hit_ratio_counts_only_strictly_positive_days():
    # [0.01, -0.01, 0.00, 0.02] -> 2 winners out of 4 observations = 0.5.
    # The flat day is in the denominator but not the numerator.
    assert hit_ratio(pd.Series([0.01, -0.01, 0.0, 0.02])) == pytest.approx(0.5, abs=1e-12)


def test_hit_ratio_is_a_fraction_not_a_percentage(port_returns: pd.Series):
    assert 0.0 <= hit_ratio(port_returns) <= 1.0


def test_hit_ratio_is_nan_for_an_empty_series(empty_series: pd.Series):
    assert math.isnan(hit_ratio(empty_series))


# ---- beta and correlation ------------------------------------------------
def test_beta_of_a_series_against_itself_is_one(port_returns: pd.Series):
    # cov(r, r)/var(r) = var(r)/var(r) = 1, whatever the data.
    assert beta(port_returns, port_returns) == pytest.approx(1.0, rel=1e-12)


def test_beta_of_twice_a_series_against_it_is_two(port_returns: pd.Series):
    # cov(2r, r)/var(r) = 2*var(r)/var(r) = 2
    assert beta(2.0 * port_returns, port_returns) == pytest.approx(2.0, rel=1e-12)


def test_beta_is_unchanged_by_adding_a_constant(port_returns: pd.Series):
    # Covariance is translation-invariant: alpha does not leak into beta.
    assert beta(port_returns + 0.01, port_returns) == pytest.approx(1.0, rel=1e-12)


def test_beta_against_a_zero_variance_benchmark_is_nan(port_returns: pd.Series):
    flat_benchmark = pd.Series(0.0, index=port_returns.index)
    assert math.isnan(beta(port_returns, flat_benchmark))


def test_beta_uses_only_the_overlapping_dates(port_returns: pd.Series):
    # The benchmark covers a 10-day slice and equals HALF the portfolio there,
    # so on the overlap beta = cov(2b, b)/var(b) = 2 exactly. If the join were
    # outer (or positional) this would not come out to a clean 2.
    overlap = 0.5 * port_returns.iloc[5:15]
    assert beta(port_returns, overlap) == pytest.approx(2.0, rel=1e-9)


def test_beta_is_nan_for_empty_or_non_overlapping_input(
    port_returns: pd.Series, empty_series: pd.Series
):
    assert math.isnan(beta(empty_series, port_returns))
    assert math.isnan(beta(port_returns, empty_series))
    disjoint = pd.Series([0.01, 0.02], index=pd.bdate_range("2099-01-04", periods=2))
    assert math.isnan(beta(port_returns, disjoint))


def test_correlation_matrix_diagonal_is_one(asset_returns: pd.DataFrame):
    corr = correlation_matrix(asset_returns)
    assert np.diag(corr.to_numpy()) == pytest.approx(np.ones(corr.shape[0]), abs=1e-12)


def test_correlation_matrix_is_symmetric_and_bounded(asset_returns: pd.DataFrame):
    corr = correlation_matrix(asset_returns)
    assert corr.to_numpy() == pytest.approx(corr.to_numpy().T, abs=1e-12)
    assert np.all(np.abs(corr.to_numpy()) <= 1.0 + 1e-12)


def test_correlation_matrix_is_empty_for_empty_or_non_numeric_input():
    assert correlation_matrix(pd.DataFrame()).empty
    assert correlation_matrix(pd.DataFrame({"name": ["AAA", "BBB"]})).empty
    assert correlation_matrix(None).empty


def test_correlation_from_a_single_row_is_nan_not_a_fabricated_zero(
    asset_returns: pd.DataFrame,
):
    corr = correlation_matrix(asset_returns.iloc[:1])
    assert corr.shape == (3, 3)
    assert corr.isna().to_numpy().all()


# ---- covariance, shrinkage and diversification ---------------------------
def test_covariance_annualisation_multiplies_by_exactly_252(
    asset_returns: pd.DataFrame,
):
    annualised = covariance_matrix(asset_returns, annualize=True)
    raw = covariance_matrix(asset_returns, annualize=False)
    assert annualised.to_numpy() == pytest.approx(252.0 * raw.to_numpy(), rel=1e-12)


def test_covariance_diagonal_is_the_square_of_annualised_volatility(
    asset_returns: pd.DataFrame,
):
    # Identity tying the covariance matrix to the scalar volatility function:
    # sqrt(S_ii) must equal annualized_volatility of column i, or the optimiser
    # and the risk report would be quoting two different volatilities.
    cov = covariance_matrix(asset_returns)
    for ticker in asset_returns.columns:
        expected = annualized_volatility(asset_returns[ticker])
        assert math.sqrt(cov.loc[ticker, ticker]) == pytest.approx(expected, rel=1e-12)


def test_covariance_shrinkage_preserves_the_trace(asset_returns: pd.DataFrame):
    # The target is (trace/n) * I precisely so that total variance survives
    # ANY shrinkage intensity; only the cross-sectional structure is pulled in.
    plain = covariance_matrix(asset_returns, shrinkage=0.0)
    baseline = float(np.trace(plain.to_numpy()))
    for intensity in (0.1, 0.5, 0.9, 1.0):
        shrunk = covariance_matrix(asset_returns, shrinkage=intensity)
        assert float(np.trace(shrunk.to_numpy())) == pytest.approx(baseline, rel=1e-12)


def test_covariance_shrinkage_improves_the_condition_number(
    asset_returns: pd.DataFrame,
):
    # Better conditioning is the entire point: it stops a mean-variance
    # optimiser betting the book on the noisiest eigenvector.
    plain = covariance_matrix(asset_returns, shrinkage=0.0).to_numpy()
    shrunk = covariance_matrix(asset_returns, shrinkage=0.5).to_numpy()
    assert np.linalg.cond(shrunk) < np.linalg.cond(plain)


def test_full_shrinkage_produces_a_scaled_identity(asset_returns: pd.DataFrame):
    full = covariance_matrix(asset_returns, shrinkage=1.0).to_numpy()
    off_diagonal = full - np.diag(np.diag(full))
    assert off_diagonal == pytest.approx(np.zeros_like(off_diagonal), abs=1e-15)
    # Every diagonal entry is trace/n, so they are all identical.
    assert np.diag(full) == pytest.approx(np.full(full.shape[0], full[0, 0]), rel=1e-12)


def test_covariance_shrinkage_is_clamped_to_the_unit_interval(
    asset_returns: pd.DataFrame,
):
    assert covariance_matrix(asset_returns, shrinkage=5.0).to_numpy() == pytest.approx(
        covariance_matrix(asset_returns, shrinkage=1.0).to_numpy(), rel=1e-12
    )
    assert covariance_matrix(asset_returns, shrinkage=-3.0).to_numpy() == pytest.approx(
        covariance_matrix(asset_returns, shrinkage=0.0).to_numpy(), rel=1e-12
    )


def test_covariance_is_empty_for_fewer_than_two_rows(asset_returns: pd.DataFrame):
    assert covariance_matrix(asset_returns.iloc[:1]).empty
    assert covariance_matrix(pd.DataFrame()).empty
    assert covariance_matrix(None).empty


def test_diversification_ratio_of_a_single_asset_is_exactly_one():
    # DR = (w.sigma)/sqrt(w'Sw) = sqrt(0.04)/sqrt(0.04) = 1: one asset cannot
    # diversify itself.
    cov = pd.DataFrame([[0.04]], index=["AAA"], columns=["AAA"])
    assert diversification_ratio({"AAA": 1.0}, cov) == pytest.approx(1.0, rel=1e-12)


def test_diversification_ratio_of_perfectly_correlated_assets_is_one():
    # S = [[s^2, s^2], [s^2, s^2]] with s = 0.2 and w = (0.5, 0.5):
    #   w.sigma      = 0.5*0.2 + 0.5*0.2 = 0.2
    #   w'Sw         = 0.04  -> sqrt = 0.2
    #   DR           = 0.2 / 0.2 = 1
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    assert diversification_ratio([50.0, 50.0], cov) == pytest.approx(1.0, rel=1e-12)


def test_diversification_ratio_of_two_uncorrelated_equal_vol_assets_is_root_two():
    # S = diag(0.04, 0.04), w = (0.5, 0.5):
    #   w.sigma      = 0.5*0.2 + 0.5*0.2 = 0.2
    #   w'Sw         = 0.25*0.04 + 0.25*0.04 = 0.02 -> sqrt = 0.1414214
    #   DR           = 0.2 / 0.1414214 = sqrt(2) = 1.41421356
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    assert diversification_ratio([0.5, 0.5], cov) == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_diversification_ratio_exceeds_one_for_imperfectly_correlated_assets(
    asset_returns: pd.DataFrame,
):
    cov = covariance_matrix(asset_returns)
    corr = correlation_matrix(asset_returns)
    off = corr.to_numpy()[np.triu_indices(3, k=1)]
    assert np.all(off < 0.99)  # premise: the assets are not one asset
    assert diversification_ratio([1 / 3, 1 / 3, 1 / 3], cov) > 1.0


def test_diversification_ratio_accepts_percent_or_fraction_weights_identically(
    asset_returns: pd.DataFrame,
):
    cov = covariance_matrix(asset_returns)
    as_percent = diversification_ratio({"AAA": 50.0, "BBB": 30.0, "CCC": 20.0}, cov)
    as_fraction = diversification_ratio({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, cov)
    assert as_percent == pytest.approx(as_fraction, rel=1e-12)


def test_diversification_ratio_is_nan_for_unusable_input(asset_returns: pd.DataFrame):
    cov = covariance_matrix(asset_returns)
    assert math.isnan(diversification_ratio([1 / 3] * 3, pd.DataFrame()))
    assert math.isnan(diversification_ratio(None, cov))
    assert math.isnan(diversification_ratio([0.0, 0.0, 0.0], cov))


# ---- concentration -------------------------------------------------------
def test_herfindahl_of_four_equal_weights_is_one_quarter():
    # 4 * (25/100)^2 = 4 * 0.0625 = 0.25 -- also the 1/n floor for n = 4.
    assert herfindahl_index([25, 25, 25, 25]) == pytest.approx(0.25, abs=1e-12)


def test_herfindahl_of_a_single_holding_is_one():
    # (100/100)^2 = 1.0 -- maximum possible concentration.
    assert herfindahl_index([100]) == pytest.approx(1.0, abs=1e-12)


def test_herfindahl_is_hand_computable_for_an_uneven_book():
    # (0.5)^2 + (0.3)^2 + (0.2)^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert herfindahl_index([50, 30, 20]) == pytest.approx(0.38, abs=1e-12)


def test_herfindahl_reads_values_not_keys_from_mappings_and_series():
    expected = herfindahl_index([25, 25, 25, 25])
    assert herfindahl_index({"A": 25, "B": 25, "C": 25, "D": 25}) == pytest.approx(
        expected, abs=1e-12
    )
    assert herfindahl_index(pd.Series([25.0] * 4)) == pytest.approx(expected, abs=1e-12)


def test_herfindahl_is_nan_for_empty_or_unparseable_input():
    assert math.isnan(herfindahl_index(None))
    assert math.isnan(herfindahl_index([]))
    assert math.isnan(herfindahl_index(42.0))  # not iterable
    assert math.isnan(herfindahl_index(["a", "b"]))  # not numeric


def test_effective_number_of_bets_accepts_percent_or_fraction_weights_identically():
    # Weights are normalised by their own sum, so 25/25/25/25 and
    # 0.25/0.25/0.25/0.25 must give the same answer: 1/0.25 = 4 equal bets.
    as_percent = effective_number_of_bets([25, 25, 25, 25])
    as_fraction = effective_number_of_bets([0.25, 0.25, 0.25, 0.25])
    assert as_percent == pytest.approx(4.0, rel=1e-12)
    assert as_percent == pytest.approx(as_fraction, rel=1e-12)


def test_effective_number_of_bets_is_the_reciprocal_of_hhi_for_percent_weights():
    # Identity: with weights summing to 100, HHI = sum(f_i^2) and the effective
    # count is 1/sum(f_i^2), so their product is exactly 1.
    weights = [40, 30, 20, 10]
    assert herfindahl_index(weights) * effective_number_of_bets(weights) == pytest.approx(
        1.0, rel=1e-12
    )


def test_effective_number_of_bets_collapses_toward_one_for_a_dominant_position():
    # 60/10/10/10/10: HHI = 0.36 + 4*0.01 = 0.40 -> 1/0.40 = 2.5 effective
    # names, not the 5 the position count claims.
    assert effective_number_of_bets([60, 10, 10, 10, 10]) == pytest.approx(2.5, rel=1e-12)


def test_effective_number_of_bets_is_nan_for_empty_or_zero_summing_weights():
    assert math.isnan(effective_number_of_bets([]))
    assert math.isnan(effective_number_of_bets([0.0, 0.0]))
    assert math.isnan(effective_number_of_bets(None))


# ---- Kupiec proportion-of-failures backtest ------------------------------
def test_kupiec_p_value_is_a_probability(port_returns: pd.Series):
    result = kupiec_pof_test(port_returns, value_at_risk(port_returns, 0.95), 0.95)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["lr_statistic"] >= 0.0


def test_kupiec_does_not_reject_when_the_breach_rate_matches_the_confidence():
    # 5 breaches in 100 days at 95% is EXACTLY the expected rate, so the
    # observed and null likelihoods coincide: LR = 0 and p = 1.
    returns = pd.Series([-0.10] * 5 + [0.01] * 95)
    result = kupiec_pof_test(returns, -0.05, 0.95)
    assert result["n"] == 100
    assert result["breaches"] == 5
    assert result["breach_rate"] == pytest.approx(0.05, abs=1e-12)
    assert result["expected_breaches"] == pytest.approx(5.0, abs=1e-9)
    assert result["lr_statistic"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-12)
    assert result["reject"] is False


def test_kupiec_also_rejects_a_var_that_is_never_breached():
    # Too FEW breaches is a failure too -- it means the model overstates risk.
    # With x = 0, n = 100, p = 0.05:
    #   LR = -2 * n * ln(1 - p) = -2 * 100 * ln(0.95) = 10.2586589
    returns = pd.Series([0.001] * 100)
    result = kupiec_pof_test(returns, -0.05, 0.95)
    assert result["breaches"] == 0
    assert result["lr_statistic"] == pytest.approx(-2.0 * 100.0 * math.log(0.95), abs=1e-9)
    assert result["lr_statistic"] == pytest.approx(10.2586589, abs=1e-6)
    assert result["p_value"] < 0.05
    assert result["reject"] is True


def test_kupiec_counts_breaches_strictly_below_the_threshold():
    # A return sitting exactly ON the VaR line is not a breach.
    returns = pd.Series([-0.05] * 10 + [0.01] * 10)
    assert kupiec_pof_test(returns, -0.05, 0.95)["breaches"] == 0
    assert kupiec_pof_test(returns, -0.04, 0.95)["breaches"] == 10


def test_kupiec_expected_breaches_is_n_times_one_minus_confidence(
    port_returns: pd.Series,
):
    result = kupiec_pof_test(port_returns, -0.02, 0.99)
    assert result["expected_breaches"] == pytest.approx(len(port_returns) * 0.01, rel=1e-9)


def test_kupiec_does_not_reject_a_var_fitted_on_the_same_sample(
    port_returns: pd.Series,
):
    # In-sample, the historical VaR is the empirical quantile by construction,
    # so the breach count should land on the expected rate and pass.
    var = value_at_risk(port_returns, 0.95, "historical")
    result = kupiec_pof_test(port_returns, var, 0.95)
    assert result["breach_rate"] == pytest.approx(0.05, abs=0.01)
    assert result["reject"] is False


def test_kupiec_returns_a_blank_result_for_unusable_input(empty_series: pd.Series):
    blank = kupiec_pof_test(empty_series, -0.02, 0.95)
    assert blank["n"] == 0
    assert blank["breaches"] == 0
    assert math.isnan(blank["p_value"])
    assert blank["reject"] is False
    assert kupiec_pof_test(pd.Series([0.01] * 50), float("nan"))["n"] == 0
    assert kupiec_pof_test(pd.Series([0.01] * 50), -0.02, 1.5)["n"] == 0


# ---- risk score ----------------------------------------------------------
def test_risk_score_is_hand_computable_for_a_balanced_book():
    # Rubric: vol 30 pts (cap 40%), drawdown 25 pts (cap 50%), concentration
    # 18 pts, holdings 12 pts (floor 10 names), tail 15 pts (cap 5%).
    #   volatility     30 * (0.20 / 0.40)        = 15.0
    #   max drawdown   25 * (0.25 / 0.50)        = 12.5
    #   concentration  HHI 0.25 == 1/4 floor      =  0.0  (perfectly equal)
    #   holdings       12 * ((10 - 4) / 9)        =  8.0
    #   tail           CVaR missing -> DROPPED from both sides
    #   earned 35.5 out of an available 30+25+18+12 = 85
    #   score = 35.5 / 85 * 100 = 41.76 -> 41.8, which sits in [35, 65]
    result = risk_score(0.20, -0.25, [25, 25, 25, 25], 4)
    assert result["score"] == pytest.approx(41.8, abs=0.05)
    assert result["band"] == "Moderate"
    assert result["max_points"] == pytest.approx(85.0, abs=1e-12)
    assert result["components"]["volatility (max 30)"] == pytest.approx(15.0, abs=0.05)
    assert result["components"]["max drawdown (max 25)"] == pytest.approx(12.5, abs=0.05)
    assert result["components"]["concentration (max 18)"] == pytest.approx(0.0, abs=0.05)
    assert result["components"]["holdings count (max 12)"] == pytest.approx(8.0, abs=0.05)
    assert "tail risk (max 15)" not in result["components"]


def test_risk_score_bands_a_quiet_diversified_book_as_low():
    #   volatility     30 * (0.05 / 0.40) = 3.75
    #   max drawdown   25 * (0.05 / 0.50) = 2.5
    #   concentration  HHI 0.10 == 1/10 floor = 0.0
    #   holdings       12 * ((10 - 10)/9)  = 0.0
    #   score = 6.25 / 85 * 100 = 7.35 -> 7.4, below the 35-point low band
    result = risk_score(0.05, -0.05, [10] * 10, 10)
    assert result["score"] == pytest.approx(7.4, abs=0.05)
    assert result["band"] == "Low"


def test_risk_score_saturates_at_100_for_a_single_volatile_holding():
    #   volatility     0.60 > cap 0.40 -> full 30
    #   max drawdown   0.80 > cap 0.50 -> full 25
    #   concentration  one holding      -> full 18
    #   holdings       12 * ((10-1)/9)  -> full 12
    #   tail           |−0.08| > cap 0.05 -> full 15
    #   100 / 100 * 100 = 100.0, above the 65-point high band
    result = risk_score(0.60, -0.80, [100], 1, cvar=-0.08)
    assert result["score"] == pytest.approx(100.0, abs=1e-9)
    assert result["band"] == "High"
    assert result["max_points"] == pytest.approx(100.0, abs=1e-12)


@pytest.mark.parametrize("vol", [0.0, 0.05, 0.20, 0.45, 2.0])
@pytest.mark.parametrize("mdd", [0.0, -0.10, -0.55, -1.0])
@pytest.mark.parametrize("n_holdings", [1, 5, 25])
def test_risk_score_is_bounded_between_zero_and_one_hundred(vol, mdd, n_holdings):
    weights = [100.0 / n_holdings] * n_holdings
    result = risk_score(vol, mdd, weights, n_holdings, cvar=-0.03)
    assert 0.0 <= result["score"] <= 100.0
    assert result["band"] in {"Low", "Moderate", "High"}


def test_risk_score_components_reproduce_the_reported_score():
    # Identity keeping the rubric inspectable: score is earned/available*100,
    # and ``components`` holds the raw per-component points (rounded to 1dp).
    result = risk_score(0.31, -0.37, [45, 25, 20, 10], 4, cvar=-0.026)
    earned = sum(result["components"].values())
    assert earned / result["max_points"] * 100.0 == pytest.approx(result["score"], abs=0.1)


def test_risk_score_rescales_instead_of_flattering_a_missing_component():
    # Without a CVaR estimate the 15 tail points leave the DENOMINATOR too,
    # so the portfolio is graded out of 85 rather than quietly gifted 15
    # points of "no tail risk".
    with_tail = risk_score(0.20, -0.25, [25, 25, 25, 25], 4, cvar=-0.02)
    without_tail = risk_score(0.20, -0.25, [25, 25, 25, 25], 4)
    assert with_tail["max_points"] == pytest.approx(100.0, abs=1e-12)
    assert without_tail["max_points"] == pytest.approx(85.0, abs=1e-12)


def test_risk_score_rises_monotonically_with_volatility():
    scores = [
        risk_score(vol, -0.20, [25, 25, 25, 25], 4)["score"]
        for vol in (0.05, 0.10, 0.20, 0.30, 0.40)
    ]
    assert scores == sorted(scores)


def test_risk_score_is_none_when_the_inputs_could_not_be_measured():
    for bad in (
        risk_score(float("nan"), -0.25, [100], 1),
        risk_score(0.20, float("nan"), [100], 1),
        risk_score(float("inf"), -0.25, [100], 1),
    ):
        assert bad["score"] is None
        assert bad["band"] == "Insufficient data"
        assert bad["components"] == {}
        assert bad["max_points"] == 0.0


def test_risk_score_is_none_for_a_portfolio_with_no_holdings():
    result = risk_score(0.20, -0.25, [], 0)
    assert result["score"] is None
    assert result["band"] == "Insufficient data"


def test_risk_score_drops_concentration_when_weights_cannot_be_parsed():
    # Unmeasurable, so it leaves both sides of the fraction: 85 - 18 = 67.
    result = risk_score(0.20, -0.25, None, 4)
    assert result["max_points"] == pytest.approx(67.0, abs=1e-12)
    assert not any(k.startswith("concentration") for k in result["components"])


# ---- risk_summary --------------------------------------------------------
def test_risk_summary_agrees_with_every_underlying_function(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    from src.analytics.returns import annualized_return

    summary = risk_summary(
        portfolio_values,
        weights_pct=[40, 30, 30],
        n_holdings=3,
        risk_free_rate=0.06,
        confidence=0.95,
    )
    ann_ret = annualized_return(portfolio_values)
    vol = annualized_volatility(port_returns)
    dev = downside_deviation(port_returns)
    mdd = max_drawdown(portfolio_values)
    ulcer = ulcer_index(portfolio_values)

    assert summary["ann_return"] == pytest.approx(ann_ret, rel=1e-12)
    assert summary["ann_vol"] == pytest.approx(vol, rel=1e-12)
    assert summary["mdd"] == pytest.approx(mdd, rel=1e-12)
    assert summary["ulcer"] == pytest.approx(ulcer, rel=1e-12)
    assert summary["downside_dev"] == pytest.approx(dev, rel=1e-12)
    assert summary["sharpe"] == pytest.approx(sharpe_ratio(ann_ret, vol, 0.06), rel=1e-12)
    assert summary["sortino"] == pytest.approx(sortino_ratio(ann_ret, dev, 0.06), rel=1e-12)
    assert summary["calmar"] == pytest.approx(calmar_ratio(ann_ret, mdd), rel=1e-12)
    assert summary["martin"] == pytest.approx(martin_ratio(ann_ret, ulcer, 0.06), rel=1e-12)
    assert summary["var"] == pytest.approx(
        value_at_risk(port_returns, 0.95, "historical"), rel=1e-12
    )
    assert summary["cvar"] == pytest.approx(
        conditional_var(port_returns, 0.95, "historical"), rel=1e-12
    )
    assert summary["hhi"] == pytest.approx(herfindahl_index([40, 30, 30]), rel=1e-12)
    assert summary["effective_bets"] == pytest.approx(
        effective_number_of_bets([40, 30, 30]), rel=1e-12
    )
    assert summary["n_obs"] == float(len(port_returns))
    assert summary["n_holdings"] == 3.0


def test_risk_summary_keeps_its_sign_and_ordering_conventions(
    portfolio_values: pd.Series,
):
    summary = risk_summary(portfolio_values, n_holdings=3)
    assert summary["mdd"] <= 0.0  # drawdown is negative
    assert summary["cvar"] <= summary["var"] < 0  # loss quantiles, both negative
    assert summary["ulcer"] > 0.0  # PERCENT, so a whole number scale
    assert 0.0 <= summary["hit_ratio"] <= 1.0  # a fraction, not a percentage
    assert summary["worst_day"] <= summary["best_day"]


def test_risk_summary_returns_only_floats(portfolio_values: pd.Series):
    summary = risk_summary(portfolio_values, weights_pct=[50, 50], n_holdings=2)
    assert all(isinstance(v, float) for v in summary.values())


def test_risk_summary_falls_back_to_the_default_confidence(
    portfolio_values: pd.Series,
):
    assert risk_summary(portfolio_values, confidence=5.0)["confidence"] == pytest.approx(
        DEFAULT_VAR_CONFIDENCE, abs=1e-12
    )


def test_risk_summary_derives_returns_from_values_when_none_are_supplied(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    derived = risk_summary(portfolio_values)
    explicit = risk_summary(portfolio_values, returns=port_returns)
    assert derived["ann_vol"] == pytest.approx(explicit["ann_vol"], rel=1e-12)
    assert derived["n_obs"] == explicit["n_obs"]


def test_risk_summary_survives_degenerate_input(
    empty_series: pd.Series, nan_series: pd.Series, single_value_series: pd.Series
):
    for series in (empty_series, nan_series, single_value_series):
        summary = risk_summary(series)
        assert summary["n_obs"] == 0.0
        assert math.isnan(summary["ann_vol"])
        assert math.isnan(summary["var"])
        assert math.isnan(summary["sharpe"])


# ---- degenerate inputs across the whole module ---------------------------
def test_every_scalar_statistic_is_nan_for_an_all_nan_series(nan_series: pd.Series):
    returns = daily_returns(nan_series)
    for value in (
        annualized_volatility(returns),
        downside_deviation(returns),
        max_drawdown(nan_series),
        ulcer_index(nan_series),
        value_at_risk(returns),
        conditional_var(returns),
        skewness(returns),
        excess_kurtosis(returns),
        tail_ratio(returns),
        hit_ratio(returns),
        gain_to_pain(returns),
        omega_ratio(returns),
        beta(returns, returns),
    ):
        assert math.isnan(value)


def test_a_single_observation_never_raises(single_value_series: pd.Series):
    returns = daily_returns(single_value_series)
    assert returns.empty
    assert math.isnan(annualized_volatility(returns))
    assert max_drawdown(single_value_series) == 0.0
    assert ulcer_index(single_value_series) == 0.0
    assert math.isnan(value_at_risk(returns))


def test_a_zero_variance_series_never_divides_by_zero(flat_series: pd.Series):
    returns = daily_returns(flat_series)
    assert annualized_volatility(returns) == 0.0
    assert downside_deviation(returns) == 0.0
    assert max_drawdown(flat_series) == 0.0
    assert ulcer_index(flat_series) == 0.0
    assert math.isnan(sharpe_ratio(0.0, annualized_volatility(returns), 0.0))
    assert math.isnan(sortino_ratio(0.0, downside_deviation(returns), 0.0))
    assert math.isnan(calmar_ratio(0.0, max_drawdown(flat_series)))
    assert math.isnan(martin_ratio(0.0, ulcer_index(flat_series), 0.0))
    assert math.isnan(omega_ratio(returns))
    assert math.isnan(gain_to_pain(returns))


def test_unsorted_dates_are_sorted_before_any_path_dependent_measurement(
    portfolio_values: pd.Series,
):
    rng = np.random.default_rng(3)
    shuffled = portfolio_values.iloc[rng.permutation(len(portfolio_values))]
    assert not shuffled.index.is_monotonic_increasing
    assert max_drawdown(shuffled) == pytest.approx(max_drawdown(portfolio_values), rel=1e-12)
    assert ulcer_index(shuffled) == pytest.approx(ulcer_index(portfolio_values), rel=1e-12)
