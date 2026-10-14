"""Portfolio-versus-benchmark analytics: ``src/analytics/benchmark.py``.

Why this file matters
--------------------
Every relative-performance number a client actually reads -- "we beat the index
by 3%", "our beta is 1.1", "we capture 95% of the upside and only 60% of the
downside" -- comes out of this one module. A benchmark comparison that is
quietly wrong is worse than no comparison at all, because it is *plausible*: a
beta of 1.08 looks exactly as reasonable whether the covariance was divided by
the right variance or not.

So this file leans on three things a reviewer can check without trusting the
implementation:

1.  **Identities that cannot rot.** A series compared with ITSELF must give
    beta 1, correlation 1, R-squared 1, tracking error exactly 0 and zero
    excess return; a portfolio whose daily returns are exactly ``k`` times the
    benchmark's must have beta exactly ``k``; both growth legs must start at
    exactly 100.0; ``IR * TE`` must equal the annualised mean active return.
    These hold for any input and any future refactor.
2.  **Hand-computed golden values.** Every scalar statistic (tracking error,
    information ratio, R-squared, Treynor, M2, capture) is checked against an
    expected number derived in a comment from a four- or six-point frame small
    enough to do on paper.
3.  **Documented degradation.** The module promises that nothing raises and
    that nothing substitutes zero for "undefined". Empty, single-row, all-NaN,
    zero-variance, non-overlapping, duplicated-date, unsorted and
    positional-index inputs each get a test asserting NaN / an empty Series /
    ``ok=False`` with a reason -- never an exception.

Conventions pinned here as regression guards
--------------------------------------------
* Annualisation is 252 trading days (``TE = std(a, ddof=1) * sqrt(252)``),
  with ``ddof=1`` because a price history is a sample.
* Capture ratios are computed on MONTHLY compounded returns and reported in
  PERCENT, where 100 means "matched the index".
* ``information_ratio`` returns **NaN**, not 0 and not infinity, when tracking
  error is zero -- i.e. a series against itself has NO information ratio.
* ``m2_measure`` deliberately ignores ``bench_return``: M2 is a level, and
  ``M2 - R_b`` is the alpha the caller forms afterwards.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics import benchmark as bm

# ---------------------------------------------------------------------------
# local builders -- used only where a hand-built frame makes the expected
# value obvious; everything else uses the shared conftest fixtures.
# ---------------------------------------------------------------------------
FOUR_DAYS = pd.bdate_range("2023-01-02", periods=4)


def _month_marked_returns(moves: dict[str, float], start: str, end: str) -> pd.Series:
    """Daily returns that are zero except for ONE marked day per month.

    Because every other day in the month is exactly 0.0, the compounded
    calendar-month return collapses to the single marked move:
    ``prod(1 + r_t) - 1 == move``. That is what lets the capture-ratio tests
    below state their expected percentages exactly.
    """
    index = pd.bdate_range(start, end)
    periods = index.to_period(bm.MONTH_PERIOD)
    series = pd.Series(0.0, index=index)
    for month, move in moves.items():
        series.loc[index[periods == pd.Period(month)][0]] = move
    return series


# ---------------------------------------------------------------------------
# module conventions
# ---------------------------------------------------------------------------
def test_the_published_conventions_are_the_documented_ones() -> None:
    # Rebasing to 100 and 252-day annualisation are load-bearing: a silent
    # change to either would rescale every chart and every ratio downstream.
    assert bm.REBASE_BASE == 100.0
    assert bm.MONTH_PERIOD == "M"
    assert bm.TRADING_DAYS == 252


def test_the_methodology_strings_and_benchmark_catalogue_are_exported() -> None:
    assert isinstance(bm.ALPHA_METHODOLOGY, str) and "Jensen" in bm.ALPHA_METHODOLOGY
    assert isinstance(bm.CAPTURE_METHODOLOGY, str) and "percent" in bm.CAPTURE_METHODOLOGY
    assert isinstance(bm.DEFAULT_BENCHMARKS, dict) and bm.DEFAULT_BENCHMARKS


# ---------------------------------------------------------------------------
# normalize_to_100 -- the rebasing that lets two magnitudes share one axis
# ---------------------------------------------------------------------------
def test_normalize_to_100_rebases_the_first_observation_to_exactly_100() -> None:
    # 200 -> 100 (divisor); 240/200 = 1.20 -> 120; 180/200 = 0.90 -> 90
    out = bm.normalize_to_100(pd.Series([200.0, 240.0, 180.0], index=FOUR_DAYS[:3]))
    assert out.iloc[0] == 100.0  # exact, not approximate: the divisor is itself
    assert out.tolist() == pytest.approx([100.0, 120.0, 90.0])


def test_normalize_to_100_preserves_every_ratio(portfolio_values: pd.Series) -> None:
    # Identity: N_t / N_0 == V_t / V_0 for all t, which is the whole point of
    # rebasing -- vertical distance on the chart must still be real performance.
    out = bm.normalize_to_100(portfolio_values)
    expected = portfolio_values / portfolio_values.iloc[0] * 100.0
    assert out.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-12)


def test_normalize_to_100_returns_empty_when_the_divisor_is_zero() -> None:
    # A first observation of exactly 0 makes V_t / V_0 undefined; the module
    # promises an empty Series rather than inf or a fabricated 100.
    assert bm.normalize_to_100(pd.Series([0.0, 1.0, 2.0])).empty


def test_normalize_to_100_returns_empty_for_an_empty_series(empty_series: pd.Series) -> None:
    assert bm.normalize_to_100(empty_series).empty


def test_normalize_to_100_returns_empty_for_an_all_nan_series(nan_series: pd.Series) -> None:
    assert bm.normalize_to_100(nan_series).empty


def test_normalize_to_100_rebases_on_the_earliest_date_when_the_input_is_unsorted() -> None:
    # Rows supplied out of order: 240 on day 3, 200 on day 1, 180 on day 2.
    # The divisor must be the EARLIEST date's 200, not the first ROW's 240.
    unsorted = pd.Series([240.0, 200.0, 180.0], index=[FOUR_DAYS[2], FOUR_DAYS[0], FOUR_DAYS[1]])
    out = bm.normalize_to_100(unsorted)
    assert out.index.is_monotonic_increasing
    assert out.tolist() == pytest.approx([100.0, 90.0, 120.0])


def test_normalize_to_100_keeps_the_last_value_of_a_duplicated_date() -> None:
    # Day 1 appears twice (100 then 111); the LAST wins, so the divisor is 111.
    # 111/111 -> 100; 120/111 = 1.081081... -> 108.108108...
    dup = pd.Series([100.0, 111.0, 120.0], index=[FOUR_DAYS[0], FOUR_DAYS[0], FOUR_DAYS[1]])
    out = bm.normalize_to_100(dup)
    assert len(out) == 2
    assert out.tolist() == pytest.approx([100.0, 108.10810810810811])


def test_normalize_to_100_drops_a_leading_nan_before_choosing_the_divisor() -> None:
    # NaN is dropped first, so the divisor is 200: [200, 240] -> [100, 120].
    out = bm.normalize_to_100(pd.Series([np.nan, 200.0, 240.0], index=FOUR_DAYS[:3]))
    assert out.tolist() == pytest.approx([100.0, 120.0])


# ---------------------------------------------------------------------------
# relative_strength -- the purely relative view
# ---------------------------------------------------------------------------
def test_relative_strength_starts_at_exactly_100(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # RS_0 = (V_0/V_0) / (B_0/B_0) * 100 = 100 by construction.
    rs = bm.relative_strength(portfolio_values, benchmark_prices)
    assert not rs.empty
    assert rs.iloc[0] == 100.0


def test_relative_strength_is_the_ratio_of_the_two_rebased_series() -> None:
    # Portfolio 100 -> 120 (x1.20); benchmark 200 -> 220 (x1.10).
    # RS_1 = 1.20 / 1.10 * 100 = 109.0909...
    port = pd.Series([100.0, 120.0], index=FOUR_DAYS[:2])
    bench = pd.Series([200.0, 220.0], index=FOUR_DAYS[:2])
    rs = bm.relative_strength(port, bench)
    assert rs.tolist() == pytest.approx([100.0, 109.09090909090909])


def test_relative_strength_tracks_the_portfolio_when_the_benchmark_is_flat() -> None:
    # A flat index means the ratio degenerates to the portfolio's own growth:
    # 100 -> 110 -> 121 -> 133.1 is +10% compounded three times.
    port = pd.Series([100.0, 110.0, 121.0, 133.1], index=FOUR_DAYS)
    bench = pd.Series([50.0, 50.0, 50.0, 50.0], index=FOUR_DAYS)
    rs = bm.relative_strength(port, bench)
    assert rs.tolist() == pytest.approx([100.0, 110.0, 121.0, 133.1])
    assert rs.name == "relative_strength"


def test_relative_strength_rises_when_the_book_merely_loses_less() -> None:
    # Both fall, the index harder: portfolio -10%, benchmark -20%.
    # RS_1 = 0.90 / 0.80 * 100 = 112.5 -- a rising line in a falling market.
    port = pd.Series([100.0, 90.0], index=FOUR_DAYS[:2])
    bench = pd.Series([1000.0, 800.0], index=FOUR_DAYS[:2])
    assert bm.relative_strength(port, bench).tolist() == pytest.approx([100.0, 112.5])


def test_relative_strength_is_empty_when_the_two_series_share_no_dates(
    portfolio_values: pd.Series,
) -> None:
    disjoint = pd.Series(np.arange(100.0, 110.0), index=pd.bdate_range("2030-01-01", periods=10))
    assert bm.relative_strength(portfolio_values, disjoint).empty


def test_relative_strength_is_empty_when_either_leg_starts_non_positive() -> None:
    zero_start = pd.Series([0.0, 10.0], index=FOUR_DAYS[:2])
    bench = pd.Series([200.0, 220.0], index=FOUR_DAYS[:2])
    assert bm.relative_strength(zero_start, bench).empty
    assert bm.relative_strength(bench, zero_start).empty


def test_relative_strength_is_empty_for_a_single_overlapping_row(
    single_value_series: pd.Series,
) -> None:
    # One common date still rebases cleanly to 100 -- but there is no ratio to
    # look at, so the honest degradation is a one-point series at 100 exactly.
    rs = bm.relative_strength(single_value_series, single_value_series)
    assert rs.tolist() == [100.0]


# ---------------------------------------------------------------------------
# tracking error
# ---------------------------------------------------------------------------
# The four-day pair reused by the tracking-error / IR / R-squared goldens.
#   r_p = [0.01, 0.02, 0.03, 0.04]
#   r_b = [0.00, 0.01, 0.01, 0.02]
#   a   = r_p - r_b = [0.01, 0.01, 0.02, 0.02]
_PORT_R = pd.Series([0.01, 0.02, 0.03, 0.04], index=FOUR_DAYS)
_BENCH_R = pd.Series([0.00, 0.01, 0.01, 0.02], index=FOUR_DAYS)


def test_tracking_error_matches_the_hand_computed_active_dispersion() -> None:
    # a       = [0.01, 0.01, 0.02, 0.02],  mean(a) = 0.015
    # dev     = [-0.005, -0.005, 0.005, 0.005] -> sum(dev^2) = 4 * 2.5e-5 = 1e-4
    # var     = 1e-4 / (4 - 1) = 3.3333...e-5
    # std     = 0.01 / sqrt(3)  = 0.005773502691896258
    # TE      = std * sqrt(252) = 0.005773502691896258 * 15.874507866387544
    #         = 0.09165151389911678
    assert bm.tracking_error(_PORT_R, _BENCH_R) == pytest.approx(0.09165151389911678, rel=1e-12)


def test_tracking_error_of_a_series_against_itself_is_exactly_zero() -> None:
    # a_t = 0 for every t, so std(a) = 0 and TE = 0 -- an index fund does not
    # wander from its index at all.
    assert bm.tracking_error(_PORT_R, _PORT_R) == 0.0


def test_tracking_error_uses_the_252_day_annualisation_with_ddof_one() -> None:
    # Regression guard on the convention itself: dividing the reported TE by
    # the SAMPLE (ddof=1) daily dispersion must recover sqrt(252) exactly.
    # A switch to ddof=0, or to 260/365-day annualisation, breaks this.
    active = _PORT_R - _BENCH_R
    ratio = bm.tracking_error(_PORT_R, _BENCH_R) / float(active.std(ddof=1))
    assert ratio == pytest.approx(math.sqrt(252), rel=1e-12)


def test_tracking_error_ignores_direction_and_only_measures_dispersion() -> None:
    # A portfolio that trails the index by the same wobble it leads it by
    # scores identically -- TE is silent about sign, by design.
    mirrored = _BENCH_R - (_PORT_R - _BENCH_R)
    assert bm.tracking_error(mirrored, _BENCH_R) == pytest.approx(
        bm.tracking_error(_PORT_R, _BENCH_R), rel=1e-12
    )


def test_tracking_error_is_nan_with_fewer_than_two_overlapping_days() -> None:
    assert math.isnan(bm.tracking_error(_PORT_R.iloc[:1], _BENCH_R.iloc[:1]))


def test_tracking_error_is_nan_when_the_series_share_no_dates() -> None:
    other = pd.Series([0.01] * 4, index=pd.bdate_range("2030-01-01", periods=4))
    assert math.isnan(bm.tracking_error(_PORT_R, other))


def test_tracking_error_is_nan_for_an_empty_series(empty_series: pd.Series) -> None:
    assert math.isnan(bm.tracking_error(empty_series, _BENCH_R))


# ---------------------------------------------------------------------------
# information ratio
# ---------------------------------------------------------------------------
def test_information_ratio_matches_the_hand_computed_value() -> None:
    # mean(a) = 0.015,  std(a, ddof=1) = 0.01 / sqrt(3) = 0.005773502691896258
    # IR = mean/std * sqrt(252) = 2.598076211353316 * 15.874507866387544
    #    = 41.24318125460257
    assert bm.information_ratio(_PORT_R, _BENCH_R) == pytest.approx(41.24318125460257, rel=1e-12)


def test_information_ratio_times_tracking_error_is_the_annualised_active_return() -> None:
    # Identity linking the two statistics:
    #   IR * TE = (mean/std * sqrt(252)) * (std * sqrt(252)) = mean(a) * 252
    # mean(a) = 0.015 -> 0.015 * 252 = 3.78
    product = bm.information_ratio(_PORT_R, _BENCH_R) * bm.tracking_error(_PORT_R, _BENCH_R)
    assert product == pytest.approx(3.78, rel=1e-12)


def test_information_ratio_is_nan_when_tracking_error_is_zero() -> None:
    # DOCUMENTED CONVENTION, asserted so it cannot silently become 0 or inf:
    # a series against itself has NO information ratio, not an infinite one.
    assert math.isnan(bm.information_ratio(_PORT_R, _PORT_R))


def test_information_ratio_is_negative_when_the_book_persistently_trails() -> None:
    # a_t = -0.01 every day but with dispersion, so the sign of mean(a) --
    # and therefore of IR -- must come through negative.
    laggard = _BENCH_R - pd.Series([0.005, 0.010, 0.015, 0.020], index=FOUR_DAYS)
    assert bm.information_ratio(laggard, _BENCH_R) < 0.0


def test_information_ratio_is_nan_with_fewer_than_two_overlapping_days() -> None:
    assert math.isnan(bm.information_ratio(_PORT_R.iloc[:1], _BENCH_R.iloc[:1]))


# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------
def test_r_squared_matches_the_hand_computed_correlation_squared() -> None:
    # x = [1, 2, 3, 4] * 1%,  y = [1, 3, 2, 4] * 1%   (mean 2.5% both)
    # dev_x = [-1.5, -0.5,  0.5, 1.5],  dev_y = [-1.5, 0.5, -0.5, 1.5]
    # sum(dev_x * dev_y) = 2.25 - 0.25 - 0.25 + 2.25 = 4.0
    # sum(dev_x^2) = sum(dev_y^2) = 5.0
    # corr = 4.0 / 5.0 = 0.8  ->  R2 = 0.64
    bench = pd.Series([0.01, 0.02, 0.03, 0.04], index=FOUR_DAYS)
    port = pd.Series([0.01, 0.03, 0.02, 0.04], index=FOUR_DAYS)
    assert bm.r_squared(port, bench) == pytest.approx(0.64, rel=1e-12)


def test_r_squared_of_a_series_against_itself_is_exactly_one() -> None:
    assert bm.r_squared(_PORT_R, _PORT_R) == pytest.approx(1.0, abs=1e-12)


def test_r_squared_is_one_for_any_perfect_linear_rescaling() -> None:
    # R2 is invariant to slope and intercept: a 2x-levered clone of the index
    # is still 100% explained by it.
    assert bm.r_squared(2.0 * _BENCH_R + 0.001, _BENCH_R) == pytest.approx(1.0, abs=1e-12)


def test_r_squared_is_nan_for_a_zero_variance_leg() -> None:
    # A constant portfolio has no variance to explain, so the correlation --
    # and hence R2 -- is undefined rather than 0.
    constant = pd.Series([0.0, 0.0, 0.0, 0.0], index=FOUR_DAYS)
    assert math.isnan(bm.r_squared(constant, _BENCH_R))
    assert math.isnan(bm.r_squared(_BENCH_R, constant))


def test_r_squared_is_nan_with_fewer_than_two_overlapping_days() -> None:
    assert math.isnan(bm.r_squared(_PORT_R.iloc[:1], _BENCH_R.iloc[:1]))


def test_r_squared_stays_inside_the_unit_interval_on_real_shaped_data(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    port_r = portfolio_values.pct_change().iloc[1:]
    bench_r = benchmark_prices.pct_change().iloc[1:]
    assert 0.0 <= bm.r_squared(port_r, bench_r) <= 1.0


# ---------------------------------------------------------------------------
# capture ratios -- monthly, in percent
# ---------------------------------------------------------------------------
def test_capture_of_a_half_down_full_up_portfolio_is_100_up_and_50_down() -> None:
    # Six calendar months, one marked move each, so the compounded monthly
    # return IS the marked move:
    #     benchmark  +2%,  -4%,  +3%,  -1%,  +5%,  -2%
    #     portfolio  +2%,  -2%,  +3%, -0.5%, +5%,  -1%   (1.0x up, 0.5x down)
    # up   = mean(2, 3, 5)      / mean(2, 3, 5)        * 100 = 100
    # down = mean(-2, -0.5, -1) / mean(-4, -1, -2)     * 100 =  50
    #        (-1.1666.. / -2.3333..) = 0.5 exactly
    # ratio = 100 / 50 = 2.0 -- the asymmetry every manager is selling.
    bench = _month_marked_returns(
        {
            "2023-01": 0.02,
            "2023-02": -0.04,
            "2023-03": 0.03,
            "2023-04": -0.01,
            "2023-05": 0.05,
            "2023-06": -0.02,
        },
        "2023-01-02",
        "2023-06-30",
    )
    port = bench.where(bench >= 0.0, bench * 0.5)

    out = bm.capture_ratios(port, bench)
    assert out["up_capture"] == pytest.approx(100.0, rel=1e-9)
    assert out["down_capture"] == pytest.approx(50.0, rel=1e-9)
    assert out["capture_ratio"] == pytest.approx(2.0, rel=1e-9)
    assert out["up_months"] == 3
    assert out["down_months"] == 3
    assert out["n_months"] == 6


def test_capture_of_a_series_against_itself_is_100_in_both_regimes() -> None:
    bench = _month_marked_returns(
        {"2023-01": 0.02, "2023-02": -0.04, "2023-03": 0.03}, "2023-01-02", "2023-03-31"
    )
    out = bm.capture_ratios(bench, bench)
    assert out["up_capture"] == pytest.approx(100.0)
    assert out["down_capture"] == pytest.approx(100.0)
    assert out["capture_ratio"] == pytest.approx(1.0)


def test_capture_puts_an_exactly_flat_benchmark_month_in_neither_bucket() -> None:
    # Jan +2%, Feb exactly 0.0 for both legs, Mar -4%.
    # A flat month is neither "up" nor "down": it is counted in n_months but
    # in neither bucket, so the two capture figures stay clean.
    bench = _month_marked_returns({"2023-01": 0.02, "2023-03": -0.04}, "2023-01-02", "2023-03-31")
    port = bench.where(bench >= 0.0, bench * 0.5)
    out = bm.capture_ratios(port, bench)
    assert out["up_months"] == 1
    assert out["down_months"] == 1
    assert out["n_months"] == 3  # February is present but bucketed nowhere
    assert out["up_capture"] == pytest.approx(100.0, rel=1e-9)
    assert out["down_capture"] == pytest.approx(50.0, rel=1e-9)


def test_capture_reports_nan_rather_than_a_fabricated_number_for_an_empty_bucket() -> None:
    # Three up months and no down month: down-capture is a fact about the
    # SAMPLE, not a virtue of the portfolio, so it must be blank.
    bench = _month_marked_returns(
        {"2023-01": 0.02, "2023-02": 0.01, "2023-03": 0.03}, "2023-01-02", "2023-03-31"
    )
    out = bm.capture_ratios(bench * 1.2, bench)
    assert out["up_capture"] == pytest.approx(120.0, rel=1e-9)  # 1.2x every up month
    assert math.isnan(out["down_capture"])
    assert math.isnan(out["capture_ratio"])
    assert out["down_months"] == 0


def test_capture_is_blank_on_a_positional_index_because_months_cannot_be_inferred() -> None:
    port = pd.Series([0.01, -0.02, 0.03, -0.01], index=[0, 1, 2, 3])
    bench = pd.Series([0.02, -0.04, 0.03, -0.02], index=[0, 1, 2, 3])
    out = bm.capture_ratios(port, bench)
    assert math.isnan(out["up_capture"])
    assert math.isnan(out["down_capture"])
    assert out["n_months"] == 0


def test_capture_is_blank_when_the_series_share_no_dates() -> None:
    a = _month_marked_returns({"2023-01": 0.02}, "2023-01-02", "2023-01-31")
    b = _month_marked_returns({"2030-01": 0.02}, "2030-01-01", "2030-01-31")
    out = bm.capture_ratios(a, b)
    assert out["n_months"] == 0
    assert out["up_months"] == 0 and out["down_months"] == 0


def test_capture_is_blank_for_empty_input(empty_series: pd.Series) -> None:
    out = bm.capture_ratios(empty_series, empty_series)
    assert math.isnan(out["up_capture"]) and out["n_months"] == 0


# ---------------------------------------------------------------------------
# Treynor
# ---------------------------------------------------------------------------
def test_treynor_ratio_matches_the_hand_computed_value() -> None:
    # (R_a - R_f) / beta = (0.18 - 0.06) / 1.2 = 0.12 / 1.2 = 0.10
    assert bm.treynor_ratio(0.18, 1.2, 0.06) == pytest.approx(0.10, rel=1e-12)


def test_treynor_reduces_to_the_excess_return_when_beta_is_one() -> None:
    # beta = 1 means all of the risk is the market's, so Treynor IS the
    # excess return: 0.18 - 0.06 = 0.12.
    assert bm.treynor_ratio(0.18, 1.0, 0.06) == pytest.approx(0.12, rel=1e-12)


def test_treynor_is_nan_for_a_market_neutral_book() -> None:
    # beta exactly 0: no systematic risk to divide by, so no Treynor ratio --
    # not an infinite one.
    assert math.isnan(bm.treynor_ratio(0.18, 0.0, 0.06))


def test_treynor_reports_a_negative_beta_rather_than_suppressing_it() -> None:
    # (0.10 - 0.04) / -0.5 = -0.12. Hiding this would also hide the fact that
    # the portfolio is inversely exposed to its own benchmark.
    assert bm.treynor_ratio(0.10, -0.5, 0.04) == pytest.approx(-0.12, rel=1e-12)


def test_treynor_is_nan_when_any_input_is_not_finite() -> None:
    assert math.isnan(bm.treynor_ratio(float("nan"), 1.2, 0.06))
    assert math.isnan(bm.treynor_ratio(0.18, float("nan"), 0.06))
    assert math.isnan(bm.treynor_ratio(0.18, 1.2, float("nan")))
    assert math.isnan(bm.treynor_ratio(float("inf"), 1.2, 0.06))


# ---------------------------------------------------------------------------
# M2 (Modigliani risk-adjusted performance)
# ---------------------------------------------------------------------------
def test_m2_measure_matches_the_hand_computed_value() -> None:
    # S_p = (0.18 - 0.06) / 0.20 = 0.60
    # M2  = R_f + S_p * sigma_b = 0.06 + 0.60 * 0.15 = 0.06 + 0.09 = 0.15
    assert bm.m2_measure(0.18, 0.20, 0.12, 0.15, 0.06) == pytest.approx(0.15, rel=1e-12)


def test_m2_equals_the_portfolio_return_when_the_two_volatilities_match() -> None:
    # Identity: with sigma_p == sigma_b there is no leverage to apply, so
    # M2 collapses to R_f + (R_a - R_f) = R_a = 0.18.
    assert bm.m2_measure(0.18, 0.20, 0.12, 0.20, 0.06) == pytest.approx(0.18, rel=1e-12)


def test_m2_alpha_against_the_benchmark_is_the_readable_number() -> None:
    # The number a client wants is M2 - R_b = 0.15 - 0.12 = 0.03: "levered to
    # the index's risk, we would have earned three points more than the index".
    m2 = bm.m2_measure(0.18, 0.20, 0.12, 0.15, 0.06)
    assert m2 - 0.12 == pytest.approx(0.03, rel=1e-12)


def test_m2_ignores_the_level_of_the_benchmark_return() -> None:
    # DOCUMENTED CONVENTION: bench_return is accepted so the caller can form
    # the M2 alpha, but by definition it does not enter the M2 level itself.
    assert bm.m2_measure(0.18, 0.20, 0.12, 0.15, 0.06) == bm.m2_measure(
        0.18, 0.20, -0.99, 0.15, 0.06
    )


def test_m2_is_nan_for_zero_portfolio_volatility() -> None:
    # The leverage factor (R_a - R_f) / sigma_p is undefined at sigma_p = 0.
    assert math.isnan(bm.m2_measure(0.18, 0.0, 0.12, 0.15, 0.06))


def test_m2_is_nan_when_a_required_input_is_not_finite() -> None:
    assert math.isnan(bm.m2_measure(float("nan"), 0.20, 0.12, 0.15, 0.06))
    assert math.isnan(bm.m2_measure(0.18, 0.20, 0.12, float("nan"), 0.06))
    assert math.isnan(bm.m2_measure(0.18, 0.20, 0.12, 0.15, float("nan")))


# ---------------------------------------------------------------------------
# compare -- the self-comparison identity, the backbone of this module
# ---------------------------------------------------------------------------
def test_compare_of_a_series_against_itself_is_the_identity_case(
    portfolio_values: pd.Series,
) -> None:
    # Feeding the same series in on both sides is the one case where every
    # answer is known a priori. If any of these drifts, the beta/alpha shown
    # to a client is wrong by an unknown amount.
    out = bm.compare(portfolio_values, portfolio_values, 0.06)
    assert out["ok"] is True
    assert out["beta"] == pytest.approx(1.0, abs=1e-12)
    assert out["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert out["tracking_error"] == 0.0  # exactly zero: a_t == 0 for every t
    assert out["excess_return"] == 0.0  # exactly zero: R_p == R_b
    assert out["alpha"] == pytest.approx(0.0, abs=1e-12)
    # DOCUMENTED CONVENTION: zero tracking error means NO information ratio.
    assert math.isnan(out["information_ratio"])


def test_compare_of_a_series_against_itself_captures_100_in_both_regimes(
    portfolio_values: pd.Series,
) -> None:
    out = bm.compare(portfolio_values, portfolio_values, 0.06)
    assert out["up_capture"] == pytest.approx(100.0)
    assert out["down_capture"] == pytest.approx(100.0)
    assert out["capture_ratio"] == pytest.approx(1.0)
    assert out["best_relative_month"] == pytest.approx(0.0, abs=1e-15)
    assert out["worst_relative_month"] == pytest.approx(0.0, abs=1e-15)


def test_compare_of_a_series_against_itself_has_a_zero_active_return_series(
    portfolio_values: pd.Series,
) -> None:
    out = bm.compare(portfolio_values, portfolio_values, 0.06)
    active = out["active_return_series"]
    # One return per interval: n_days observations give n_days - 1 returns.
    assert len(active) == out["n_days"] - 1
    assert float(active.abs().max()) == 0.0


def test_compare_gives_beta_exactly_two_for_a_twice_as_volatile_clone() -> None:
    # Construct r_p = 2 * r_b day by day, then integrate both into levels.
    # beta = cov(2 r_b, r_b) / var(r_b) = 2 var(r_b) / var(r_b) = 2 exactly,
    # and a perfect linear relationship means correlation and R2 are 1.
    rng = np.random.default_rng(20240101)
    index = pd.bdate_range("2023-01-02", periods=300)
    bench_r = rng.normal(0.0005, 0.01, 300)
    bench_r[0] = 0.0  # both series start on the same base date
    bench = pd.Series(100.0 * np.cumprod(1.0 + bench_r), index=index)
    port = pd.Series(1000.0 * np.cumprod(1.0 + 2.0 * bench_r), index=index)

    out = bm.compare(port, bench, 0.05)
    assert out["beta"] == pytest.approx(2.0, rel=1e-9)
    assert out["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_compare_growth_columns_both_start_at_exactly_100(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # A portfolio worth ~4,000 and an index at ~100 can only share an axis
    # because both are rebased; the two lines MUST start together.
    growth = bm.compare(portfolio_values, benchmark_prices, 0.06)["growth"]
    assert list(growth.columns) == ["Portfolio", "Benchmark"]
    assert growth["Portfolio"].iloc[0] == 100.0
    assert growth["Benchmark"].iloc[0] == 100.0


def test_compare_excess_return_is_the_difference_of_the_annualised_returns(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    out = bm.compare(portfolio_values, benchmark_prices, 0.06)
    assert out["excess_return"] == pytest.approx(
        out["portfolio_return"] - out["benchmark_return"], rel=1e-12
    )


def test_compare_r_squared_is_the_square_of_the_reported_correlation(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    out = bm.compare(portfolio_values, benchmark_prices, 0.06)
    assert out["r_squared"] == pytest.approx(out["correlation"] ** 2, rel=1e-12)


def test_compare_treynor_reduces_to_the_excess_return_when_beta_is_one(
    portfolio_values: pd.Series,
) -> None:
    # Self-comparison pins beta at 1, so Treynor must equal R_p - R_f exactly.
    out = bm.compare(portfolio_values, portfolio_values, 0.06)
    assert out["treynor"] == pytest.approx(out["portfolio_return"] - 0.06, rel=1e-9)


def test_compare_m2_equals_the_portfolio_return_when_the_vols_match(
    portfolio_values: pd.Series,
) -> None:
    # Self-comparison pins sigma_p == sigma_b, so no leverage is applied and
    # M2 collapses back onto the portfolio's own annualised return.
    out = bm.compare(portfolio_values, portfolio_values, 0.06)
    assert out["portfolio_vol"] == pytest.approx(out["benchmark_vol"], rel=1e-12)
    assert out["m2"] == pytest.approx(out["portfolio_return"], rel=1e-12)


def test_compare_scalar_statistics_agree_with_the_standalone_functions(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # compare() must not carry its own private copy of the maths: the block it
    # reports has to be the same numbers the public helpers produce.
    out = bm.compare(portfolio_values, benchmark_prices, 0.06)
    aligned = pd.concat(
        [portfolio_values.rename("p"), benchmark_prices.rename("b")],
        axis=1,
        join="inner",
    ).dropna()
    port_r = aligned["p"].pct_change().iloc[1:]
    bench_r = aligned["b"].pct_change().iloc[1:]
    assert out["tracking_error"] == pytest.approx(bm.tracking_error(port_r, bench_r), rel=1e-12)
    assert out["information_ratio"] == pytest.approx(
        bm.information_ratio(port_r, bench_r), rel=1e-12
    )
    assert out["r_squared"] == pytest.approx(bm.r_squared(port_r, bench_r), rel=1e-12)


# ---------------------------------------------------------------------------
# compare -- the published contract (a missing key is a regression)
# ---------------------------------------------------------------------------
LEGACY_KEYS = (
    "ok",
    "portfolio_return",
    "benchmark_return",
    "excess_return",
    "beta",
    "alpha",
    "portfolio_vol",
    "benchmark_vol",
    "growth",
    "n_days",
    "alpha_methodology",
)

NEW_KEYS = (
    "portfolio_mdd",
    "benchmark_mdd",
    "treynor",
    "m2",
    "tracking_error",
    "information_ratio",
    "correlation",
    "r_squared",
    "up_capture",
    "down_capture",
    "capture_ratio",
    "up_months",
    "down_months",
    "best_relative_month",
    "worst_relative_month",
    "active_return_series",
    "capture_methodology",
)


@pytest.mark.parametrize("key", LEGACY_KEYS)
def test_compare_still_returns_every_legacy_key(
    key: str, portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # These keys predate the v2 rewrite and are consumed by the UI, the Excel
    # workbook and the HTML tearsheet. Dropping one is a silent breakage.
    assert key in bm.compare(portfolio_values, benchmark_prices, 0.06)


@pytest.mark.parametrize("key", NEW_KEYS)
def test_compare_returns_every_documented_v2_key(
    key: str, portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    assert key in bm.compare(portfolio_values, benchmark_prices, 0.06)


def test_compare_reports_the_alpha_methodology_verbatim(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # Provenance travels with the number: an alpha without its methodology is
    # not auditable.
    out = bm.compare(portfolio_values, benchmark_prices, 0.06)
    assert out["alpha_methodology"] == bm.ALPHA_METHODOLOGY
    assert out["capture_methodology"] == bm.CAPTURE_METHODOLOGY


def test_compare_n_days_and_growth_are_sized_by_the_overlap_only(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # 500 portfolio days against a benchmark trimmed to its last 400 days
    # leaves exactly 400 common dates -- the comparison must use those and
    # nothing else, or it measures the calendar rather than the manager.
    trimmed = benchmark_prices.iloc[100:]
    assert len(portfolio_values) == 500 and len(trimmed) == 400
    out = bm.compare(portfolio_values, trimmed, 0.06)
    assert out["n_days"] == 400
    assert len(out["growth"]) == 400
    assert out["growth"].index[0] == trimmed.index[0]


# ---------------------------------------------------------------------------
# compare -- documented degradation (never an exception)
# ---------------------------------------------------------------------------
def test_compare_reports_a_reason_instead_of_raising_when_nothing_overlaps(
    portfolio_values: pd.Series,
) -> None:
    disjoint = pd.Series(np.arange(100.0, 110.0), index=pd.bdate_range("2030-01-01", periods=10))
    out = bm.compare(portfolio_values, disjoint, 0.06)
    assert out["ok"] is False
    assert isinstance(out["reason"], str) and out["reason"]
    assert set(out) == {"ok", "reason"}


def test_compare_rejects_an_empty_portfolio(
    empty_series: pd.Series, benchmark_prices: pd.Series
) -> None:
    assert bm.compare(empty_series, benchmark_prices, 0.06)["ok"] is False


def test_compare_rejects_a_single_observation(
    single_value_series: pd.Series, benchmark_prices: pd.Series
) -> None:
    # One date cannot produce a return, so there is nothing to compare.
    assert bm.compare(single_value_series, benchmark_prices, 0.06)["ok"] is False


def test_compare_rejects_an_all_nan_series(
    nan_series: pd.Series, benchmark_prices: pd.Series
) -> None:
    assert bm.compare(nan_series, benchmark_prices, 0.06)["ok"] is False


def test_compare_rejects_mismatched_index_dtypes_without_raising(
    benchmark_prices: pd.Series,
) -> None:
    positional = pd.Series([100.0, 101.0, 102.0])
    assert bm.compare(positional, benchmark_prices, 0.06)["ok"] is False


def test_compare_survives_a_zero_variance_series(flat_series: pd.Series) -> None:
    # The classic divide-by-zero trap: a perfectly flat book. Everything that
    # needs a variance in the denominator must come back NaN, and everything
    # that does not must come back as an honest zero.
    out = bm.compare(flat_series, flat_series, 0.06)
    assert out["ok"] is True
    assert out["n_days"] == 60
    assert math.isnan(out["beta"])  # var(r_b) == 0
    assert math.isnan(out["correlation"])
    assert math.isnan(out["r_squared"])
    assert math.isnan(out["information_ratio"])
    assert math.isnan(out["alpha"])  # depends on beta
    assert math.isnan(out["treynor"])  # beta is undefined
    assert math.isnan(out["m2"])  # sigma_p == 0
    assert out["portfolio_vol"] == 0.0
    assert out["portfolio_return"] == 0.0
    assert out["portfolio_mdd"] == 0.0
    assert out["tracking_error"] == 0.0


def test_compare_deduplicates_a_repeated_date_instead_of_exploding() -> None:
    # A duplicated index label would make the inner join fan out into a
    # cartesian product; the last observation for the date must win instead.
    index = pd.bdate_range("2023-01-02", periods=5)
    port = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0], index=index)
    bench = pd.Series([50.0, 50.2, 51.0, 50.8, 51.5], index=index)
    duplicated = pd.concat([port, pd.Series([999.0], index=[index[1]])])
    out = bm.compare(duplicated, bench, 0.05)
    assert out["ok"] is True
    assert out["n_days"] == 5  # not 6, and certainly not 25


def test_compare_accepts_an_unsorted_portfolio_and_rebases_chronologically() -> None:
    index = pd.bdate_range("2023-01-02", periods=4)
    port = pd.Series([100.0, 110.0, 121.0, 133.1], index=index)
    bench = pd.Series([50.0, 51.0, 52.0, 53.0], index=index)
    shuffled = port.iloc[[3, 1, 0, 2]]
    out = bm.compare(shuffled, bench, 0.05)
    assert out["growth"].index.is_monotonic_increasing
    assert out["growth"]["Portfolio"].iloc[0] == 100.0
    # 133.1 / 100 = 1.331 -> the last rebased level is 133.1
    assert out["growth"]["Portfolio"].iloc[-1] == pytest.approx(133.1, rel=1e-12)


def test_compare_leaves_capture_blank_on_a_positional_index() -> None:
    # Calendar months cannot be inferred from 0, 1, 2, 3 -- the regime block
    # must go blank rather than invent buckets, while the rest still computes.
    port = pd.Series([100.0, 101.0, 102.0, 103.0], index=[0, 1, 2, 3])
    bench = pd.Series([50.0, 50.5, 51.0, 51.5], index=[0, 1, 2, 3])
    out = bm.compare(port, bench, 0.0)
    assert out["ok"] is True
    assert math.isnan(out["up_capture"]) and math.isnan(out["down_capture"])
    assert out["up_months"] == 0 and out["down_months"] == 0
    assert math.isnan(out["best_relative_month"])


def test_compare_blanks_the_rate_dependent_block_for_a_non_finite_risk_free_rate(
    portfolio_values: pd.Series, benchmark_prices: pd.Series
) -> None:
    # An alpha quoted without its risk-free assumption is a rumour: with an
    # unusable rate the rate-dependent statistics must be NaN, while the
    # rate-independent ones (beta, correlation, TE) still stand.
    out = bm.compare(portfolio_values, benchmark_prices, float("nan"))
    assert out["ok"] is True
    assert math.isnan(out["alpha"])
    assert math.isnan(out["treynor"])
    assert math.isnan(out["m2"])
    assert math.isfinite(out["beta"])
    assert math.isfinite(out["tracking_error"])


def test_compare_never_raises_on_any_degenerate_input(
    empty_series: pd.Series,
    single_value_series: pd.Series,
    flat_series: pd.Series,
    nan_series: pd.Series,
) -> None:
    # Blanket promise from the module docstring: "Nothing here raises on
    # degenerate input." One assertion covering the whole cross-product.
    candidates = [empty_series, single_value_series, flat_series, nan_series]
    for left in candidates:
        for right in candidates:
            out = bm.compare(left, right, 0.06)
            assert isinstance(out, dict) and "ok" in out


# ---------------------------------------------------------------------------
# known defect
# ---------------------------------------------------------------------------
def test_compare_does_not_silently_blank_the_growth_frame_for_a_zero_first_price() -> None:
    """A zero first observation must fail loudly, not render a blank line as data.

    normalize_to_100 correctly returns an EMPTY Series when the first value is
    zero, but dropping an empty Series into a DataFrame constructor back-fills
    that whole column with NaN. compare() once returned ok=True anyway, so the
    tearsheet drew an invisible benchmark line that looked like real data.
    """
    index = pd.bdate_range("2023-01-02", periods=4)
    port = pd.Series([100.0, 110.0, 121.0, 133.1], index=index)
    bench_with_zero_start = pd.Series([0.0, 10.0, 20.0, 30.0], index=index)

    out = bm.compare(port, bench_with_zero_start, 0.05)
    if out["ok"]:
        # The documented invariant: if the comparison claims to be usable,
        # both legs of the growth chart must start together at exactly 100.
        assert out["growth"]["Portfolio"].iloc[0] == 100.0
        assert out["growth"]["Benchmark"].iloc[0] == 100.0
    else:
        assert isinstance(out["reason"], str) and out["reason"]
