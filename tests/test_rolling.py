"""Rolling risk statistics — warm-up, annualisation, identities and degradation.

``src.analytics.rolling`` recomputes the headline risk numbers over a moving
window so the UI can show *when* the portfolio's character changed. Three
things make it easy to get subtly wrong, and all three are pinned here:

1. **Warm-up.** ``min_periods == window``, so the first ``window - 1`` points
   must be NaN. A partial window silently reported as a complete one is the
   classic way a rolling chart grows a fake spike at the left edge.
2. **Annualisation.** ``sqrt(252)`` on volatility and ``252 / window`` on the
   window's compounded return. Both conventions are asserted against
   hand-computed arithmetic and cross-checked against the full-period
   functions in ``src.analytics.risk``, so the two modules cannot drift.
3. **Insufficient data.** Fewer than ``MIN_ROLLING_WINDOW`` (20) usable
   observations returns an EMPTY Series — never a two-point line pretending to
   be a trend. A window LONGER than the sample returns a correctly indexed
   all-NaN series instead.

On top of that: the mathematical identities that cannot rot as the code
evolves (rolling beta of a series against itself is exactly 1; correlation is
bounded to [-1, 1] and is exactly ±1 against itself and its negative;
volatility of a constant-return series is 0), the NEGATIVE sign convention on
rolling VaR, and the exact six-column contract of ``rolling_frame``.

The golden values come from a deliberately periodic return series:

    r = [+1%, -1%, +2%, -2%] repeated six times   (24 observations)

Every trailing window of 4 holds the same four numbers in a different order,
so every defined point of every statistic is the SAME hand-computable value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import annualized_volatility, value_at_risk
from src.analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    MIN_ROLLING_WINDOW,
    MIN_VAR_OBSERVATIONS,
    ROLLING_COLUMNS,
    TRADING_DAYS,
    rolling_beta,
    rolling_correlation,
    rolling_frame,
    rolling_sharpe,
    rolling_sortino,
    rolling_var,
    rolling_volatility,
)

# ---------------------------------------------------------------------------
# fixtures — small, periodic, hand-computable
# ---------------------------------------------------------------------------
CYCLE = [0.01, -0.01, 0.02, -0.02]
CYCLE_WINDOW = 4


@pytest.fixture
def cycle_returns() -> pd.Series:
    """24 daily returns; every trailing window of 4 is a rotation of CYCLE."""
    return pd.Series(CYCLE * 6, index=pd.bdate_range("2023-01-02", periods=24))


@pytest.fixture
def walk_returns() -> pd.Series:
    """120 seeded daily returns — long enough for a 30-day window to breathe."""
    rng = np.random.default_rng(20240101)
    return pd.Series(
        rng.normal(0.0004, 0.012, 120), index=pd.bdate_range("2023-01-02", periods=120)
    )


@pytest.fixture
def constant_returns() -> pd.Series:
    """Zero-variance returns — the divide-by-zero trap for every ratio here."""
    return pd.Series([0.001] * 30, index=pd.bdate_range("2023-01-02", periods=30))


SINGLE_SERIES_FUNCS = (rolling_volatility, rolling_sharpe, rolling_sortino, rolling_var)
PAIR_SERIES_FUNCS = (rolling_beta, rolling_correlation)


# ---------------------------------------------------------------------------
# warm-up: NaN before the window fills, a value after
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("func", SINGLE_SERIES_FUNCS)
def test_every_rolling_statistic_is_nan_until_the_window_fills(walk_returns, func):
    window = 30
    out = func(walk_returns, window)
    # positions 0..28 are partial windows and must be blank...
    assert out.iloc[: window - 1].isna().all()
    # ...position 29 is the first complete window and must carry a number.
    assert np.isfinite(out.iloc[window - 1])
    # exactly len - window + 1 defined points: 120 - 30 + 1 = 91
    assert int(out.notna().sum()) == len(walk_returns) - window + 1 == 91


@pytest.mark.parametrize("func", PAIR_SERIES_FUNCS)
def test_benchmark_relative_statistics_share_the_same_warm_up(walk_returns, func):
    window = 30
    out = func(walk_returns, walk_returns * 1.5 + 0.0001, window)
    assert out.iloc[: window - 1].isna().all()
    assert np.isfinite(out.iloc[window - 1])
    assert int(out.notna().sum()) == 91


@pytest.mark.parametrize(
    ("func", "name"),
    [
        (rolling_volatility, "volatility"),
        (rolling_sharpe, "sharpe"),
        (rolling_sortino, "sortino"),
        (rolling_var, "var"),
    ],
)
def test_single_series_output_keeps_the_input_index_and_its_column_name(walk_returns, func, name):
    out = func(walk_returns, 30)
    assert out.index.equals(walk_returns.index)
    assert out.name == name


# ---------------------------------------------------------------------------
# hand-computed golden values
# ---------------------------------------------------------------------------
def test_rolling_volatility_matches_the_hand_computed_annualised_sigma(cycle_returns):
    # Window = {+0.01, -0.01, +0.02, -0.02}; mean = 0 exactly.
    #   sum of squares = 1e-4 + 1e-4 + 4e-4 + 4e-4 = 1.0e-3
    #   var (ddof=1)   = 1.0e-3 / 3      = 3.333333e-4
    #   sigma          = sqrt(3.3333e-4) = 0.018257418583505537
    #   annualised     = sigma * sqrt(252) = 0.28982753492378877
    out = rolling_volatility(cycle_returns, CYCLE_WINDOW)
    assert out.iloc[CYCLE_WINDOW - 1] == pytest.approx(0.28982753492378877, rel=1e-12)
    # every rotation of the same four numbers gives the same sigma
    assert out.dropna().nunique() == 1
    assert len(out.dropna()) == 24 - CYCLE_WINDOW + 1


def test_rolling_volatility_annualises_by_sqrt_252_exactly(walk_returns):
    # Regression guard on the 252-day convention: change the constant and this
    # test fails loudly instead of every downstream chart shifting quietly.
    assert TRADING_DAYS == 252
    raw = walk_returns.rolling(30).std(ddof=1)
    scaled = rolling_volatility(walk_returns, 30)
    ratio = (scaled / raw).dropna()
    assert ratio.to_numpy() == pytest.approx(math.sqrt(252), rel=1e-12)


def test_rolling_volatility_over_the_whole_sample_equals_the_full_period_figure(
    walk_returns,
):
    # Identity with src.analytics.risk: window == len(returns) means the last
    # rolling point IS the full-period annualised volatility.
    last = rolling_volatility(walk_returns, len(walk_returns)).iloc[-1]
    assert last == pytest.approx(annualized_volatility(walk_returns), rel=1e-12)


def test_rolling_volatility_of_a_constant_return_series_is_zero(constant_returns):
    # Zero dispersion must be reported as zero, not as a floating-point smear.
    out = rolling_volatility(constant_returns, 10).dropna()
    assert len(out) == 21
    assert out.to_numpy() == pytest.approx(0.0, abs=1e-12)


def test_rolling_sharpe_matches_the_hand_computed_excess_over_sigma(cycle_returns):
    # Window growth = 1.01 * 0.99 * 1.02 * 0.98 = 0.9995000399999999
    # Annualised    = 0.99950004 ** (252 / 4) - 1
    #               = 0.99950004 ** 63 - 1 = -0.031014233802001412
    # Sharpe        = (-0.031014233802 - 0) / 0.28982753492378877
    #               = -0.10700927298
    out = rolling_sharpe(cycle_returns, CYCLE_WINDOW, risk_free_rate=0.0)
    assert out.iloc[CYCLE_WINDOW - 1] == pytest.approx(-0.10700927298, rel=1e-9)
    assert out.dropna().nunique() == 1


def test_rolling_sharpe_annualises_the_window_return_by_252_over_window(cycle_returns):
    # The exponent is 252 / window, so the implied annual growth of a 4-day
    # window is the window's gross growth raised to the 63rd power.
    ann_excess = rolling_sharpe(cycle_returns, CYCLE_WINDOW) * rolling_volatility(
        cycle_returns, CYCLE_WINDOW
    )
    expected = (1.01 * 0.99 * 1.02 * 0.98) ** (TRADING_DAYS / CYCLE_WINDOW) - 1.0
    assert ann_excess.dropna().to_numpy() == pytest.approx(expected, rel=1e-9)


def test_the_risk_free_rate_is_subtracted_from_the_annualised_return(walk_returns):
    # Identity: S(rf) = S(0) - rf / sigma, for every defined point.
    vol = rolling_volatility(walk_returns, 30)
    base = rolling_sharpe(walk_returns, 30, risk_free_rate=0.0)
    charged = rolling_sharpe(walk_returns, 30, risk_free_rate=0.065)
    expected = (base - 0.065 / vol).dropna()
    assert charged.dropna().to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-10)


def test_rolling_sharpe_is_nan_when_the_window_has_zero_volatility(constant_returns):
    # A constant return series has a perfectly real positive return and zero
    # risk. That is undefined, not infinite.
    out = rolling_sharpe(constant_returns, 10)
    assert out.isna().all()


def test_rolling_sortino_matches_the_hand_computed_downside_denominator(cycle_returns):
    # min(r, 0) ** 2 over the window = {0, 1e-4, 0, 4e-4}
    #   mean over ALL four observations = 5e-4 / 4 = 1.25e-4
    #   downside deviation = sqrt(1.25e-4) * sqrt(252)
    #                      = 0.011180339887 * 15.874507866 = 0.1774823934929885
    #   sortino = -0.031014233802 / 0.1774823934929885 = -0.174745411033
    out = rolling_sortino(cycle_returns, CYCLE_WINDOW, risk_free_rate=0.0)
    assert out.iloc[CYCLE_WINDOW - 1] == pytest.approx(-0.174745411033, rel=1e-9)


def test_rolling_sortino_divides_by_a_smaller_denominator_than_sharpe(cycle_returns):
    # Only two of the four days lose, so downside deviation (0.1775) is below
    # total volatility (0.2898) and |sortino| must exceed |sharpe|.
    sharpe = rolling_sharpe(cycle_returns, CYCLE_WINDOW).dropna()
    sortino = rolling_sortino(cycle_returns, CYCLE_WINDOW).dropna()
    assert (sortino.abs() > sharpe.abs()).all()


def test_rolling_sortino_is_nan_for_a_window_with_no_losing_day(constant_returns):
    # Downside deviation is exactly 0 — undefined, not an infinite Sortino.
    assert rolling_sortino(constant_returns, 10).isna().all()


# ---------------------------------------------------------------------------
# identities against a benchmark
# ---------------------------------------------------------------------------
def test_rolling_beta_of_a_series_against_itself_is_one(walk_returns):
    # cov(x, x) / var(x) == 1 by construction, for every filled window.
    out = rolling_beta(walk_returns, walk_returns, 30)
    defined = out.dropna()
    assert len(defined) == 91
    assert defined.to_numpy() == pytest.approx(1.0, abs=1e-12)


def test_rolling_beta_of_a_series_against_twice_itself_is_one_half(walk_returns):
    # cov(x, 2x) / var(2x) = 2 var(x) / 4 var(x) = 0.5
    out = rolling_beta(walk_returns, walk_returns * 2.0, 30).dropna()
    assert out.to_numpy() == pytest.approx(0.5, abs=1e-12)


def test_rolling_beta_is_nan_when_the_benchmark_window_has_no_variance(walk_returns):
    flat_bench = pd.Series(0.001, index=walk_returns.index)
    out = rolling_beta(walk_returns, flat_bench, 30)
    # Correctly indexed, entirely undefined — never ±inf.
    assert out.index.equals(walk_returns.index)
    assert out.isna().all()


def test_rolling_correlation_of_a_series_against_itself_is_one(walk_returns):
    out = rolling_correlation(walk_returns, walk_returns, 30).dropna()
    assert out.to_numpy() == pytest.approx(1.0, abs=1e-12)


def test_rolling_correlation_of_a_series_against_its_negative_is_minus_one(walk_returns):
    out = rolling_correlation(walk_returns, -walk_returns, 30).dropna()
    assert out.to_numpy() == pytest.approx(-1.0, abs=1e-12)


def test_rolling_correlation_is_bounded_to_minus_one_and_one(asset_returns):
    out = rolling_correlation(asset_returns["AAA"], asset_returns["BBB"], 63).dropna()
    assert len(out) > 0
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_rolling_correlation_is_nan_for_a_zero_variance_benchmark(walk_returns):
    flat_bench = pd.Series(0.001, index=walk_returns.index)
    assert rolling_correlation(walk_returns, flat_bench, 30).isna().all()


def test_beta_and_correlation_are_computed_on_the_overlapping_dates_only(walk_returns):
    # The benchmark starts 30 sessions late; the join happens BEFORE the
    # window, so the result lives on the 90 shared dates, not on 120.
    bench = walk_returns.iloc[30:]
    out = rolling_beta(walk_returns, bench, 30)
    assert out.index.equals(bench.index)
    assert len(out) == 90
    # ...and the first filled window is the 30th SHARED observation.
    assert out.iloc[:29].isna().all()
    assert np.isfinite(out.iloc[29])


@pytest.mark.parametrize("func", PAIR_SERIES_FUNCS)
def test_a_missing_benchmark_gives_an_empty_series(walk_returns, func):
    assert func(walk_returns, None, 30).empty


# ---------------------------------------------------------------------------
# rolling VaR — the sign convention is a regression guard
# ---------------------------------------------------------------------------
def test_rolling_var_matches_the_hand_computed_fifth_percentile():
    # 20 returns, evenly spaced: -1.0%, -0.9%, ..., +0.9%.
    # np.percentile(..., 5) interpolates linearly between order statistics:
    #   position = 0.05 * (20 - 1) = 0.95
    #   value    = -0.010 + 0.95 * (-0.009 - -0.010) = -0.00905
    ladder = pd.Series(
        (np.arange(20) - 10) / 1000.0, index=pd.bdate_range("2023-01-02", periods=20)
    )
    out = rolling_var(ladder, 20, confidence=0.95)
    assert out.iloc[-1] == pytest.approx(-0.00905, rel=1e-12)
    # only the single complete window is defined
    assert int(out.notna().sum()) == 1


def test_rolling_var_is_a_negative_decimal_loss(walk_returns):
    # SIGN CONVENTION: -0.021 means "lost at least 2.1%", not "+2.1% of risk".
    out = rolling_var(walk_returns, 30, confidence=0.95).dropna()
    assert len(out) == 91
    assert (out < 0.0).all()


def test_rolling_var_over_the_whole_sample_equals_the_full_period_var(walk_returns):
    # Identity with src.analytics.risk.value_at_risk (historical method).
    last = rolling_var(walk_returns, len(walk_returns), confidence=0.95).iloc[-1]
    assert last == pytest.approx(value_at_risk(walk_returns, 0.95), rel=1e-12)


def test_a_deeper_confidence_level_pushes_var_further_into_the_tail(walk_returns):
    at_95 = rolling_var(walk_returns, 60, 0.95).dropna()
    at_99 = rolling_var(walk_returns, 60, 0.99).dropna()
    assert (at_99 <= at_95).all()


def test_rolling_var_is_all_nan_when_the_window_is_too_short_for_a_tail(walk_returns):
    # A 5%-tail quantile from 10 points is noise; the guard mirrors
    # risk.value_at_risk's MIN_VAR_OBSERVATIONS floor.
    assert MIN_VAR_OBSERVATIONS == 15
    out = rolling_var(walk_returns, 10, 0.95)
    assert out.index.equals(walk_returns.index)
    assert out.isna().all()
    # one observation above the floor and it starts producing numbers again
    assert rolling_var(walk_returns, MIN_VAR_OBSERVATIONS, 0.95).notna().any()


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2, float("nan")])
def test_rolling_var_is_all_nan_for_a_confidence_outside_zero_to_one(walk_returns, bad):
    out = rolling_var(walk_returns, 30, confidence=bad)
    assert out.index.equals(walk_returns.index)
    assert out.isna().all()


# ---------------------------------------------------------------------------
# insufficient data — EMPTY, not a two-point line
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("func", SINGLE_SERIES_FUNCS)
def test_fewer_observations_than_min_rolling_window_returns_an_empty_series(func):
    assert MIN_ROLLING_WINDOW == 20
    short = pd.Series(
        np.linspace(-0.01, 0.01, MIN_ROLLING_WINDOW - 1),
        index=pd.bdate_range("2023-01-02", periods=MIN_ROLLING_WINDOW - 1),
    )
    out = func(short, 5)
    assert out.empty
    assert out.dtype == np.dtype("float64")


@pytest.mark.parametrize("func", PAIR_SERIES_FUNCS)
def test_too_little_overlap_returns_an_empty_series(func):
    short = pd.Series(
        np.linspace(-0.01, 0.01, MIN_ROLLING_WINDOW - 1),
        index=pd.bdate_range("2023-01-02", periods=MIN_ROLLING_WINDOW - 1),
    )
    assert func(short, short, 5).empty


def test_exactly_min_rolling_window_observations_is_enough():
    # 19 is empty (above); 20 must produce a real, correctly indexed series.
    just_enough = pd.Series(
        np.linspace(-0.01, 0.01, MIN_ROLLING_WINDOW),
        index=pd.bdate_range("2023-01-02", periods=MIN_ROLLING_WINDOW),
    )
    out = rolling_volatility(just_enough, 5)
    assert not out.empty
    assert len(out) == MIN_ROLLING_WINDOW


@pytest.mark.parametrize("func", SINGLE_SERIES_FUNCS)
def test_an_empty_series_gives_an_empty_series(empty_series, func):
    assert func(empty_series, 30).empty


@pytest.mark.parametrize("func", SINGLE_SERIES_FUNCS)
def test_an_all_nan_series_gives_an_empty_series(nan_series, func):
    # Every value is dropped by the cleaner, leaving nothing to roll over.
    assert func(nan_series, 5).empty


@pytest.mark.parametrize("window", [1, 0, -5, None, float("nan"), "abc"])
def test_an_unusable_window_gives_an_empty_series(walk_returns, window):
    # ddof=1 needs at least two points, so 2 is the hard floor.
    assert rolling_volatility(walk_returns, window).empty
    assert rolling_beta(walk_returns, walk_returns, window).empty


@pytest.mark.parametrize("func", SINGLE_SERIES_FUNCS)
def test_a_window_longer_than_the_sample_gives_an_indexed_all_nan_series(walk_returns, func):
    # Documented behaviour: the window never fills, so the series is NaN — but
    # it still carries the right index, so a chart draws an empty panel rather
    # than crashing on a missing axis.
    out = func(walk_returns, 500)
    assert len(out) == len(walk_returns)
    assert out.index.equals(walk_returns.index)
    assert out.isna().all()


def test_a_flat_price_derived_return_series_still_produces_zero_volatility(flat_series):
    # flat_series is 60 identical values → 59 returns of exactly 0.0.
    returns = flat_series.pct_change().dropna()
    out = rolling_volatility(returns, 20).dropna()
    assert len(out) == 40
    assert out.to_numpy() == pytest.approx(0.0, abs=1e-15)


def test_an_unsorted_date_index_is_sorted_before_the_window_is_applied(walk_returns):
    shuffled = walk_returns.sample(frac=1.0, random_state=1)
    assert not shuffled.index.is_monotonic_increasing
    out = rolling_volatility(shuffled, 30)
    assert out.index.is_monotonic_increasing
    assert out.equals(rolling_volatility(walk_returns, 30))


def test_duplicate_index_entries_do_not_raise(walk_returns):
    """A repeated timestamp must not blow up the join.

    _join once used pd.concat(join="inner"), which raises
    ValueError("cannot reindex on an axis with duplicate labels") on a repeated
    date -- easy to produce by concatenating two overlapping history pulls.
    This module promises never to raise.
    """
    doubled_stamp = pd.DatetimeIndex(
        [*list(walk_returns.index[:60]), walk_returns.index[59], *list(walk_returns.index[60:119])]
    )
    duplicated = pd.Series(walk_returns.to_numpy()[:120], index=doubled_stamp)
    # The single-series statistics already survive this...
    assert rolling_volatility(duplicated, 30).notna().any()
    # ...and the benchmark-relative ones must too.
    assert rolling_beta(duplicated, duplicated, 30).notna().any()


# ---------------------------------------------------------------------------
# rolling_frame — the risk-evolution table
# ---------------------------------------------------------------------------
def test_rolling_frame_carries_exactly_the_six_documented_columns(asset_returns, benchmark_prices):
    bench = benchmark_prices.pct_change().dropna()
    frame = rolling_frame(asset_returns["AAA"], bench, 63)
    assert tuple(frame.columns) == ROLLING_COLUMNS
    assert tuple(frame.columns) == (
        "volatility",
        "sharpe",
        "sortino",
        "beta",
        "correlation",
        "var",
    )
    assert len(frame.columns) == 6
    assert (frame.dtypes == np.dtype("float64")).all()


def test_rolling_frame_is_indexed_by_the_portfolios_own_dates(walk_returns):
    # A benchmark on a shorter calendar must not shift the portfolio series.
    frame = rolling_frame(walk_returns, walk_returns.iloc[30:], 30)
    assert frame.index.equals(walk_returns.index)
    # 90 shared dates, 30-day window → 61 filled beta points on a 120-row frame
    assert int(frame["beta"].notna().sum()) == 61
    assert frame["beta"].iloc[:59].isna().all()


def test_rolling_frame_columns_agree_with_the_standalone_functions(walk_returns):
    frame = rolling_frame(walk_returns, walk_returns, 30, risk_free_rate=0.05)
    assert frame["volatility"].equals(rolling_volatility(walk_returns, 30).rename("volatility"))
    assert frame["var"].equals(rolling_var(walk_returns, 30).rename("var"))
    expected_beta = rolling_beta(walk_returns, walk_returns, 30).dropna()
    assert frame["beta"].dropna().to_numpy() == pytest.approx(expected_beta.to_numpy(), abs=1e-12)


def test_rolling_frame_without_a_benchmark_keeps_the_columns_but_blanks_them(
    walk_returns,
):
    # The UI never has to test for a missing column.
    frame = rolling_frame(walk_returns, None, 30)
    assert tuple(frame.columns) == ROLLING_COLUMNS
    assert frame["beta"].isna().all()
    assert frame["correlation"].isna().all()
    # ...while the portfolio-only statistics are still fully populated.
    assert int(frame["volatility"].notna().sum()) == 91


def test_rolling_frame_with_a_non_overlapping_benchmark_blanks_beta_only(walk_returns):
    # A benchmark trading on a completely different calendar: the overlap is
    # empty, so beta/correlation are NaN and nothing else is disturbed.
    alien = pd.Series(walk_returns.to_numpy(), index=pd.bdate_range("2030-01-01", periods=120))
    frame = rolling_frame(walk_returns, alien, 30)
    assert frame["beta"].isna().all()
    assert frame["correlation"].isna().all()
    assert int(frame["sharpe"].notna().sum()) == 91


@pytest.mark.parametrize("degenerate", ["empty", "nan", "short"])
def test_rolling_frame_degrades_to_an_empty_frame_with_the_full_schema(degenerate):
    if degenerate == "empty":
        port = pd.Series(dtype="float64")
    elif degenerate == "nan":
        port = pd.Series([np.nan] * 40, index=pd.bdate_range("2023-01-02", periods=40))
    else:
        port = pd.Series(
            np.linspace(-0.01, 0.01, MIN_ROLLING_WINDOW - 1),
            index=pd.bdate_range("2023-01-02", periods=MIN_ROLLING_WINDOW - 1),
        )
    frame = rolling_frame(port, None, 10)
    assert frame.empty
    assert tuple(frame.columns) == ROLLING_COLUMNS


def test_rolling_frame_with_an_unusable_window_is_empty(walk_returns):
    assert rolling_frame(walk_returns, walk_returns, 1).empty
    assert tuple(rolling_frame(walk_returns, walk_returns, 1).columns) == ROLLING_COLUMNS


def test_the_default_window_is_the_apps_own_quarterly_look_back(asset_returns):
    # ~one quarter of trading days; changing it silently would re-scale every
    # rolling chart in the product.
    assert DEFAULT_ROLLING_WINDOW == 63
    frame = rolling_frame(asset_returns["AAA"])
    assert int(frame["volatility"].notna().sum()) == len(asset_returns) - 63 + 1
