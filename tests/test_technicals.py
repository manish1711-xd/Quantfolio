"""Technical indicators — ``src.analytics.technicals``.

Technical indicators are the part of a quant stack most likely to be
*plausibly* wrong: every formula has three popular variants, they all produce
a smooth line that looks right on a chart, and nobody notices the difference
until a signal fires a week late. So the tests below concentrate on the
choices that separate a correct implementation from a convincing one, and
every expected number is worked out by hand in a comment at the assertion.

Verified here
-------------
* **Wilder's smoothing, not a rolling mean.** RSI and ATR are defined with
  ``ewm(alpha = 1/period, adjust=False)``. The behavioural fingerprint is at
  the boundary: a monotonically RISING series drives RSI to exactly 100 and a
  falling one to exactly 0, and a 14-period Wilder average is identical to a
  27-span EMA (``alpha = 1/p <=> span = 2p - 1``) and provably different from
  ``rolling(14).mean()``. Those three assertions together are what catch a
  swapped smoother.
* **Bollinger's POPULATION standard deviation (ddof=0).** On ``[1, 3, 2]`` the
  band is 3.633 wide, not the 4.0 a sample standard deviation would give — a
  10% error in every band and every %B in the product.
* **Identities.** ``histogram == macd - signal`` everywhere both are defined;
  ``percent_b == (P - lower) / (upper - lower)``, which is exactly 0.5 on the
  middle band, 1.0 on the upper and 0.0 on the lower; ``upper > middle >
  lower`` always; ATR is non-negative; RSI never leaves [0, 100];
  ``position_pct`` never leaves [0, 100].
* **Leading NaNs are load-bearing.** A half-formed average is not an average:
  SMA(n) and Bollinger(n) are NaN for ``n - 1`` rows, ATR(p) for ``p - 1``,
  RSI(p) for ``p``. Publishing a partial window would print a signal before
  the data supporting it exists.
* **Documented degradation.** Empty, single-row, all-NaN, flat (zero-variance),
  close-only and nonsense-parameter inputs each get a test pinning the
  NaN / empty / fixed-layout answer rather than an exception.

Two tests are ``xfail`` — they assert order-invariance this module does not
have. See the "chronological order" section for the full description.

Deterministic and offline: hand-built ladders plus the seeded shared fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.technicals import (
    CATEGORY,
    CROSS_LOOKBACK_SESSIONS,
    WEEKS_PER_YEAR,
    atr,
    bollinger_bands,
    distance_from_52w,
    ema,
    indicator_columns,
    indicator_frame,
    macd,
    rsi,
    sma,
    technical_signals,
    true_range,
    wilder_smooth,
)
from src.config import TECHNICALS, TRADING_DAYS, TechnicalConfig
from src.models import Insight

#: sqrt(2/3) -- the POPULATION standard deviation of [1, 3, 2], written to full
#: double precision so a 1e-12 comparison is testing the code, not the constant.
SD_1_3_2 = 0.816496580927726

#: Levels ``technical_signals`` may emit, most actionable first.
LEVEL_ORDER = ("critical", "warning", "positive", "info")


# ---------------------------------------------------------------------------
# local, hand-checkable ladders (the shared fixtures cover realistic data)
# ---------------------------------------------------------------------------
def dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n)


@pytest.fixture
def rising() -> pd.Series:
    """1, 2, 3, ... 40 — every session up, so there is never a loss to average."""
    return pd.Series(np.arange(1.0, 41.0), index=dates(40))


@pytest.fixture
def falling() -> pd.Series:
    """40, 39, ... 1 — every session down, so there is never a gain to average."""
    return pd.Series(np.arange(40.0, 0.0, -1.0), index=dates(40))


@pytest.fixture
def ohlc_four_bars() -> pd.DataFrame:
    """Four bars whose true ranges are exactly 2, 3, 1 and 3.5.

    bar 0: H 10 L  8 C  9    -> no previous close, TR = H - L         = 2
    bar 1: H 12 L  9 C 11    -> max(3, |12-9|=3,    |9-9|=0)          = 3
    bar 2: H 11 L 10 C 10.5  -> max(1, |11-11|=0,   |10-11|=1)        = 1
    bar 3: H 14 L 13 C 13.5  -> max(1, |14-10.5|=3.5, |13-10.5|=2.5)  = 3.5
    """
    return pd.DataFrame(
        {
            "Open": [9.0, 10.0, 11.0, 13.0],
            "High": [10.0, 12.0, 11.0, 14.0],
            "Low": [8.0, 9.0, 10.0, 13.0],
            "Close": [9.0, 11.0, 10.5, 13.5],
            "Volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=dates(4),
    )


@pytest.fixture
def uptrend_ohlcv() -> pd.DataFrame:
    """260 sessions compounding at +1% a day — a deliberately extreme trend.

    Extreme on purpose: it pins RSI at 100 and pushes price far above the
    200-day average, so the overbought / trend-regime rules are guaranteed to
    fire and their ordering can be asserted.
    """
    close = pd.Series(100.0 * 1.01 ** np.arange(260), index=dates(260))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": 1.0,
        }
    )


# ---- simple and exponential moving averages ------------------------------
def test_the_sma_of_a_constant_series_is_that_constant(flat_series: pd.Series):
    # mean(100, 100, ..., 100) = 100 -- the smoother must add nothing of its own.
    result = sma(flat_series, 5)
    assert result.dropna().to_numpy() == pytest.approx(np.full(56, 100.0), abs=1e-12)
    assert result.name == "SMA5"


def test_the_sma_is_the_mean_of_the_last_window_closes():
    # closes 1..10, window 3:
    #   SMA_2 = (1 + 2 + 3)/3 = 2.0     first date with a full window
    #   SMA_9 = (8 + 9 + 10)/3 = 9.0
    series = pd.Series(np.arange(1.0, 11.0), index=dates(10))
    result = sma(series, 3)
    assert result.iloc[2] == pytest.approx(2.0, abs=1e-12)
    assert result.iloc[9] == pytest.approx(9.0, abs=1e-12)


def test_the_sma_needs_a_full_window_before_it_prints_anything():
    # window - 1 = 2 leading NaNs. A two-day average labelled "SMA3" would be
    # a different indicator wearing the same name.
    series = pd.Series(np.arange(1.0, 11.0), index=dates(10))
    result = sma(series, 3)
    assert result.head(2).isna().all()
    assert np.isfinite(result.iloc[2])
    assert result.index.equals(series.index)


def test_the_ema_recursion_is_hand_computable():
    # span 3 -> a = 2/(3+1) = 0.5, adjust=False:
    #   EMA_0 = 2                        (suppressed: min_periods = span = 3)
    #   EMA_1 = 0.5*4 + 0.5*2   = 3.0    (suppressed)
    #   EMA_2 = 0.5*6 + 0.5*3.0 = 4.5
    result = ema(pd.Series([2.0, 4.0, 6.0]), 3)
    assert result.head(2).isna().all()
    assert result.iloc[2] == pytest.approx(4.5, abs=1e-12)
    assert result.name == "EMA3"


def test_the_ema_of_a_constant_series_is_that_constant(flat_series: pd.Series):
    assert ema(flat_series, 12).dropna().to_numpy() == pytest.approx(np.full(49, 100.0), abs=1e-12)


def test_the_ema_weights_the_latest_close_more_heavily_than_the_sma():
    # A step from 100 to 200: after one session the 10-span EMA has moved
    # 2/11 of the way (a = 2/11 = 0.1818...) while the 10-day SMA has moved
    # only 1/10. The EMA must sit ABOVE the SMA on a rising step.
    series = pd.Series([100.0] * 20 + [200.0], index=dates(21))
    assert float(ema(series, 10).iloc[-1]) > float(sma(series, 10).iloc[-1])
    # 100 + (200 - 100) * 2/11 = 118.1818...
    assert float(ema(series, 10).iloc[-1]) == pytest.approx(100.0 + 100.0 * 2.0 / 11.0, abs=1e-12)


def test_wilder_smoothing_is_hand_computable():
    # alpha = 1/2, adjust=False, min_periods = 2:
    #   X_0 = 1                          (suppressed)
    #   X_1 = 0.5*2 + 0.5*1     = 1.5
    #   X_2 = 0.5*3 + 0.5*1.5   = 2.25
    #   X_3 = 0.5*4 + 0.5*2.25  = 3.125
    result = wilder_smooth(pd.Series([1.0, 2.0, 3.0, 4.0]), 2)
    assert math.isnan(result.iloc[0])
    assert list(result.iloc[1:]) == pytest.approx([1.5, 2.25, 3.125], abs=1e-12)


def test_wilder_smoothing_equals_an_ema_of_span_two_p_minus_one(ohlcv: pd.DataFrame):
    # THE definition check: Wilder's alpha = 1/p is the EMA alpha 2/(s+1) with
    # s = 2p - 1. A 14-period Wilder average is a 27-span EMA, NOT a 14-span
    # one -- the single most common technical-analysis implementation error.
    close = ohlcv["Close"]
    wilder = wilder_smooth(close, 14)
    equivalent = ema(close, 27)
    both = wilder.notna() & equivalent.notna()
    assert int(both.sum()) > 300
    assert wilder[both].to_numpy() == pytest.approx(equivalent[both].to_numpy(), rel=1e-12)
    # ... and emphatically not the 14-span EMA it is so often confused with.
    assert not np.allclose(wilder[both], ema(close, 14)[both])


@pytest.mark.parametrize("bad", [0, -3, None, "nonsense", float("nan")])
def test_the_moving_averages_degrade_to_nan_for_an_unusable_window(flat_series: pd.Series, bad):
    for function in (sma, ema, wilder_smooth):
        result = function(flat_series, bad)
        assert result.isna().all()
        assert result.index.equals(flat_series.index)


def test_the_moving_averages_of_an_empty_series_are_empty(empty_series: pd.Series):
    assert sma(empty_series, 5).empty
    assert ema(empty_series, 5).empty
    assert wilder_smooth(empty_series, 5).empty


def test_the_moving_averages_of_an_all_nan_series_are_all_nan(nan_series: pd.Series):
    assert sma(nan_series, 5).isna().all()
    assert ema(nan_series, 5).isna().all()
    assert len(sma(nan_series, 5)) == len(nan_series)


# ---- RSI -----------------------------------------------------------------
def test_rsi_of_a_monotonically_rising_series_is_exactly_one_hundred(rising: pd.Series):
    # REGRESSION GUARD, and the reason this test exists at all.
    #
    # With no down session, Wilder's average loss is exactly 0, RS = avg_gain/0
    # is infinite and RSI = 100 - 100/(1 + inf) = 100 -- and it STAYS at 100
    # for as long as the streak lasts. Swap Wilder's smoothing for a simple
    # rolling mean and the value still climbs into the 90s, so a spot check of
    # "is it high?" passes either way. Only the exact 100.0 boundary
    # distinguishes the two smoothers, so assert the boundary.
    result = rsi(rising, 14)
    assert result.dropna().to_numpy() == pytest.approx(np.full(26, 100.0), abs=1e-12)
    assert result.name == "RSI"


def test_rsi_of_a_monotonically_falling_series_is_exactly_zero(falling: pd.Series):
    # Mirror image: no up session -> avg_gain = 0 -> RS = 0 -> RSI = 100 - 100 = 0.
    assert rsi(falling, 14).dropna().to_numpy() == pytest.approx(np.zeros(26), abs=1e-12)


def test_rsi_never_leaves_the_zero_to_one_hundred_band(ohlcv: pd.DataFrame):
    values = rsi(ohlcv["Close"]).dropna()
    assert not values.empty
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rsi_of_a_flat_series_is_fifty_rather_than_zero_over_zero(flat_series: pd.Series):
    # No gains AND no losses is 0/0. A flat market is neither strong nor weak,
    # so the honest answer is the neutral midpoint -- not 0, not NaN.
    assert rsi(flat_series, 14).dropna().to_numpy() == pytest.approx(np.full(46, 50.0), abs=1e-12)


def test_rsi_is_hand_computable_over_a_short_ladder():
    # closes 10, 11, 10.5, 12, 11 with period 2 (alpha = 1/2):
    #   delta      =  -,  +1,  -0.5,  +1.5,  -1
    #   gain       =  -,   1,     0,   1.5,   0
    #   loss       =  -,   0,   0.5,     0,   1
    #   avg_gain   =  -,   -,   0.5,   1.0,  0.5     (min_periods = 2)
    #   avg_loss   =  -,   -,  0.25, 0.125, 0.5625
    #   RS         =  -,   -,     2,     8,  0.888..
    #   RSI        =  -,   -, 100-100/3, 100-100/9, 100-900/17
    #              =  -,   -,   200/3,  800/9,  800/17
    result = rsi(pd.Series([10.0, 11.0, 10.5, 12.0, 11.0]), 2)
    assert result.head(2).isna().all()
    assert list(result.iloc[2:]) == pytest.approx(
        [200.0 / 3.0, 800.0 / 9.0, 800.0 / 17.0], abs=1e-12
    )
    assert result.iloc[2] == pytest.approx(66.66666666666667, abs=1e-9)


def test_rsi_is_nan_for_the_first_period_rows(rising: pd.Series):
    # The first difference is undefined, then period Wilder observations are
    # needed, so RSI(14) starts at row 14 -- 14 NaNs, not 13.
    result = rsi(rising, 14)
    assert result.head(14).isna().all()
    assert np.isfinite(result.iloc[14])


def test_rsi_uses_the_configured_default_period(ohlcv: pd.DataFrame):
    assert TECHNICALS.rsi_period == 14
    pd.testing.assert_series_equal(rsi(ohlcv["Close"]), rsi(ohlcv["Close"], 14))


def test_rsi_degrades_for_empty_all_nan_and_nonsense_periods(
    empty_series: pd.Series, nan_series: pd.Series, rising: pd.Series
):
    assert rsi(empty_series).empty
    assert rsi(nan_series).isna().all()
    assert rsi(rising, 0).isna().all()
    assert rsi(rising, None).isna().all()
    assert rsi(rising, "nonsense").isna().all()


def test_rsi_does_not_mutate_the_caller(ohlcv: pd.DataFrame):
    before = ohlcv.copy(deep=True)
    rsi(ohlcv["Close"])
    pd.testing.assert_frame_equal(ohlcv, before)


# ---- MACD ----------------------------------------------------------------
def test_the_macd_histogram_is_exactly_macd_minus_signal_everywhere(ohlcv: pd.DataFrame):
    # Identity, not an approximation: the histogram is computed once from the
    # two lines and never recomputed from a differently-rounded source. If it
    # ever drifts, a bar can cross zero on the chart while the lines have not.
    frame = macd(ohlcv["Close"])
    both = frame["macd"].notna() & frame["signal"].notna()
    assert int(both.sum()) > 300
    difference = (frame["macd"] - frame["signal"] - frame["histogram"])[both]
    assert float(difference.abs().max()) == 0.0
    # ... and where the histogram is undefined, so is at least one input.
    assert frame.loc[~both, "histogram"].isna().all()


def test_the_macd_line_is_the_fast_ema_minus_the_slow_ema(ohlcv: pd.DataFrame):
    close = ohlcv["Close"]
    frame = macd(close, 12, 26, 9)
    expected = ema(close, 12) - ema(close, 26)
    assert frame["macd"].dropna().to_numpy() == pytest.approx(
        expected.dropna().to_numpy(), rel=1e-12
    )


def test_the_macd_signal_is_an_ema_of_the_macd_line(ohlcv: pd.DataFrame):
    frame = macd(ohlcv["Close"], 12, 26, 9)
    expected = ema(frame["macd"], 9)
    assert frame["signal"].dropna().to_numpy() == pytest.approx(
        expected.dropna().to_numpy(), rel=1e-12
    )


def test_the_macd_of_a_constant_series_is_flat_zero(flat_series: pd.Series):
    # Two EMAs of the same constant are the same constant, so their difference
    # is 0 and so is the histogram: a still market generates no momentum.
    frame = macd(flat_series, 3, 6, 2)
    assert frame["macd"].dropna().to_numpy() == pytest.approx(np.zeros(55), abs=1e-12)
    assert frame["histogram"].dropna().to_numpy() == pytest.approx(np.zeros(54), abs=1e-12)


def test_the_macd_leads_with_nan_until_both_emas_are_formed(ohlcv: pd.DataFrame):
    # The slow EMA needs 26 closes, so macd starts at row 25; the signal EMA
    # then needs 9 macd values, so it starts at row 25 + 8 = 33.
    frame = macd(ohlcv["Close"], 12, 26, 9)
    assert frame["macd"].head(25).isna().all()
    assert np.isfinite(frame["macd"].iloc[25])
    assert frame["signal"].head(33).isna().all()
    assert np.isfinite(frame["signal"].iloc[33])


def test_the_macd_columns_are_stable_for_empty_and_nonsense_input(
    empty_series: pd.Series, ohlcv: pd.DataFrame
):
    columns = ["macd", "signal", "histogram"]
    empty = macd(empty_series)
    assert list(empty.columns) == columns
    assert empty.empty
    broken = macd(ohlcv["Close"], "nonsense", 26, 9)
    assert list(broken.columns) == columns
    assert broken.isna().to_numpy().all()
    assert broken.index.equals(ohlcv.index)


def test_the_macd_index_matches_the_input_index(ohlcv: pd.DataFrame):
    assert macd(ohlcv["Close"]).index.equals(ohlcv.index)


# ---- Bollinger bands -----------------------------------------------------
def test_percent_b_is_exactly_one_half_on_the_middle_band():
    # window [1, 3, 2]: middle = (1+3+2)/3 = 2, which is also the last close.
    # sd(ddof=0) = sqrt(((1-2)^2 + (3-2)^2 + (2-2)^2)/3) = sqrt(2/3)
    # upper = 2 + 2*sqrt(2/3), lower = 2 - 2*sqrt(2/3)
    # %B = (2 - lower)/(upper - lower) = (2*sqrt(2/3)) / (4*sqrt(2/3)) = 0.5
    bands = bollinger_bands(pd.Series([1.0, 3.0, 2.0]), period=3, n_std=2.0)
    assert bands["middle"].iloc[2] == pytest.approx(2.0, abs=1e-12)
    assert bands["upper"].iloc[2] == pytest.approx(2.0 + 2.0 * SD_1_3_2, abs=1e-12)
    assert bands["lower"].iloc[2] == pytest.approx(2.0 - 2.0 * SD_1_3_2, abs=1e-12)
    assert bands["percent_b"].iloc[2] == pytest.approx(0.5, abs=1e-12)


def test_percent_b_is_exactly_one_when_the_close_sits_on_the_upper_band():
    # window [0, 0, 3]: middle = 1, deviations (-1, -1, +2),
    # sd(ddof=0) = sqrt((1+1+4)/3) = sqrt(2). With n_std = sqrt(2):
    #   upper = 1 + sqrt(2)*sqrt(2) = 3  == the close
    #   lower = 1 - 2 = -1
    #   %B = (3 - (-1)) / (3 - (-1)) = 1
    bands = bollinger_bands(pd.Series([0.0, 0.0, 3.0]), period=3, n_std=math.sqrt(2.0))
    assert bands["upper"].iloc[2] == pytest.approx(3.0, abs=1e-12)
    assert bands["lower"].iloc[2] == pytest.approx(-1.0, abs=1e-12)
    assert bands["percent_b"].iloc[2] == pytest.approx(1.0, abs=1e-12)


def test_percent_b_is_exactly_zero_when_the_close_sits_on_the_lower_band():
    # Mirror image: window [3, 3, 0], middle = 2, sd = sqrt(2),
    #   lower = 2 - sqrt(2)*sqrt(2) = 0 == the close, upper = 4, %B = 0.
    bands = bollinger_bands(pd.Series([3.0, 3.0, 0.0]), period=3, n_std=math.sqrt(2.0))
    assert bands["upper"].iloc[2] == pytest.approx(4.0, abs=1e-12)
    assert bands["lower"].iloc[2] == pytest.approx(0.0, abs=1e-12)
    assert bands["percent_b"].iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_bollinger_uses_the_population_standard_deviation():
    # REGRESSION GUARD on ddof. For [1, 3, 2]:
    #   population (ddof=0): sd = sqrt(2/3) = 0.8165 -> upper = 3.632993...
    #   sample     (ddof=1): sd = 1.0             -> upper = 4.0
    # Bollinger's own construction is the population form; the sample form is
    # ~22% wider here and would suppress every band breakout signal.
    upper = float(bollinger_bands(pd.Series([1.0, 3.0, 2.0]), 3, 2.0)["upper"].iloc[2])
    assert upper == pytest.approx(2.0 + 2.0 * SD_1_3_2, abs=1e-12)
    assert upper == pytest.approx(3.632993161855452, abs=1e-12)
    assert upper != pytest.approx(4.0, abs=1e-6)


def test_the_bands_are_strictly_ordered_upper_above_middle_above_lower(
    ohlcv: pd.DataFrame,
):
    bands = bollinger_bands(ohlcv["Close"]).dropna(subset=["middle"])
    assert len(bands) == 381
    assert (bands["upper"] > bands["middle"]).all()
    assert (bands["middle"] > bands["lower"]).all()
    # The middle band IS the SMA over the same window -- not a separate fit.
    assert bands["middle"].to_numpy() == pytest.approx(
        sma(ohlcv["Close"], 20).dropna().to_numpy(), rel=1e-12
    )


def test_percent_b_is_the_close_expressed_in_band_widths(ohlcv: pd.DataFrame):
    # Identity: %B = (P - lower) / (upper - lower), so it is 0.5 at the middle
    # band by construction and leaves [0, 1] exactly when price leaves the band.
    close = ohlcv["Close"]
    bands = bollinger_bands(close)
    defined = bands["percent_b"].notna()
    expected = (close - bands["lower"]) / (bands["upper"] - bands["lower"])
    assert bands.loc[defined, "percent_b"].to_numpy() == pytest.approx(
        expected[defined].to_numpy(), rel=1e-12
    )
    above = defined & (close > bands["upper"])
    below = defined & (close < bands["lower"])
    assert int(above.sum()) > 0 and int(below.sum()) > 0
    assert (bands.loc[above, "percent_b"] > 1.0).all()
    assert (bands.loc[below, "percent_b"] < 0.0).all()


def test_the_bandwidth_is_the_band_span_as_a_percentage_of_the_middle_band():
    # (upper - lower)/middle * 100 = (4*sqrt(2/3)) / 2 * 100 = 163.2993161855452
    bands = bollinger_bands(pd.Series([1.0, 3.0, 2.0]), 3, 2.0)
    assert bands["bandwidth"].iloc[2] == pytest.approx(4.0 * SD_1_3_2 / 2.0 * 100.0, rel=1e-12)
    assert bands["bandwidth"].iloc[2] == pytest.approx(163.2993161855452, abs=1e-10)


def test_a_zero_width_band_reports_an_undefined_percent_b(flat_series: pd.Series):
    # A flat window has no dispersion: upper == middle == lower, so the
    # position inside the band is genuinely 0/0. NaN, not a fabricated 0.5.
    bands = bollinger_bands(flat_series, 20, 2.0)
    formed = bands.dropna(subset=["middle"])
    assert len(formed) == 41
    assert (formed["upper"] == formed["lower"]).all()
    assert formed["percent_b"].isna().all()
    assert formed["bandwidth"].to_numpy() == pytest.approx(np.zeros(41), abs=1e-12)


def test_the_bands_need_a_full_window_before_they_print(ohlcv: pd.DataFrame):
    bands = bollinger_bands(ohlcv["Close"], 20, 2.0)
    assert bands.head(19).isna().to_numpy().all()
    assert np.isfinite(bands["middle"].iloc[19])
    assert bands.index.equals(ohlcv.index)


def test_the_bollinger_columns_are_stable_for_empty_and_nonsense_input(
    empty_series: pd.Series, ohlcv: pd.DataFrame
):
    columns = ["middle", "upper", "lower", "bandwidth", "percent_b"]
    empty = bollinger_bands(empty_series)
    assert list(empty.columns) == columns
    assert empty.empty
    broken = bollinger_bands(ohlcv["Close"], "nonsense", 2.0)
    assert list(broken.columns) == columns
    assert broken.isna().to_numpy().all()


# ---- true range and ATR --------------------------------------------------
def test_the_true_range_is_hand_computable(ohlc_four_bars: pd.DataFrame):
    # See the fixture docstring for the arithmetic of each of the four bars.
    result = true_range(ohlc_four_bars)
    assert list(result) == pytest.approx([2.0, 3.0, 1.0, 3.5], abs=1e-12)
    assert result.name == "true_range"


def test_the_first_true_range_falls_back_to_the_high_low_range(ohlcv: pd.DataFrame):
    # There is no previous close on the first bar, so the gap terms are
    # undefined; using them would produce a NaN and poison the whole ATR.
    first = float(true_range(ohlcv).iloc[0])
    assert first == pytest.approx(float(ohlcv["High"].iloc[0] - ohlcv["Low"].iloc[0]), rel=1e-12)


def test_the_true_range_is_never_negative(ohlcv: pd.DataFrame):
    assert float(true_range(ohlcv).min()) >= 0.0


def test_the_atr_is_hand_computable(ohlc_four_bars: pd.DataFrame):
    # Wilder smoothing of TR = [2, 3, 1, 3.5] with period 3 (alpha = 1/3):
    #   X_0 = 2                              (suppressed: min_periods = 3)
    #   X_1 = (3   + 2*2    )/3 = 7/3        (suppressed)
    #   X_2 = (1   + 2*7/3  )/3 = 17/9  = 1.8888...
    #   X_3 = (3.5 + 2*17/9 )/3 = 65.5/27 = 2.4259...
    result = atr(ohlc_four_bars, 3)
    assert result.head(2).isna().all()
    assert result.iloc[2] == pytest.approx(17.0 / 9.0, abs=1e-12)
    assert result.iloc[3] == pytest.approx(65.5 / 27.0, abs=1e-12)
    assert result.name == "ATR"


def test_the_atr_is_non_negative_with_nan_for_the_first_period_minus_one_rows(
    ohlcv: pd.DataFrame,
):
    result = atr(ohlcv, 14)
    assert result.head(13).isna().all()
    assert np.isfinite(result.iloc[13])
    assert float(result.dropna().min()) >= 0.0
    assert result.index.equals(ohlcv.index)


def test_the_atr_is_wilder_smoothed_not_a_rolling_mean(ohlcv: pd.DataFrame):
    # Same guard as for RSI, on the other Wilder-smoothed indicator: an ATR
    # built on rolling(14).mean() reacts twice as fast to a single wide bar.
    tr = true_range(ohlcv)
    result = atr(ohlcv, 14)
    both = result.notna()
    assert result[both].to_numpy() == pytest.approx(
        wilder_smooth(tr, 14)[both].to_numpy(), rel=1e-12
    )
    rolling_version = tr.rolling(14).mean()
    aligned = result.notna() & rolling_version.notna()
    assert not np.allclose(result[aligned], rolling_version[aligned])


def test_the_atr_of_a_close_only_frame_is_all_nan_rather_than_an_error(
    ohlcv: pd.DataFrame,
):
    # High and Low are structurally required; without them there is no true
    # range to average, and the honest answer is "unknown", aligned to the
    # input so the chart layer can still bind to it.
    close_only = ohlcv[["Close"]]
    result = atr(close_only, 14)
    assert result.isna().all()
    assert len(result) == len(close_only)


def test_the_atr_of_an_empty_frame_is_empty():
    assert atr(pd.DataFrame()).empty
    assert true_range(pd.DataFrame()).empty


def test_the_atr_degrades_for_a_nonsense_period(ohlcv: pd.DataFrame):
    assert atr(ohlcv, 0).isna().all()
    assert atr(ohlcv, "nonsense").isna().all()


def test_the_atr_does_not_mutate_the_caller(ohlcv: pd.DataFrame):
    before = ohlcv.copy(deep=True)
    atr(ohlcv)
    true_range(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


# ---- 52-week range positioning -------------------------------------------
def test_distance_from_52w_is_hand_computable():
    # last = 120, high = 150, low = 100 over the whole (short) window:
    #   pct_from_high = (120/150 - 1)*100 = -20
    #   pct_from_low  = (120/100 - 1)*100 = +20
    #   position_pct  = (120 - 100)/(150 - 100)*100 = 40
    result = distance_from_52w(pd.Series([100.0, 150.0, 120.0]))
    assert result["last"] == pytest.approx(120.0, abs=1e-12)
    assert result["high"] == pytest.approx(150.0, abs=1e-12)
    assert result["low"] == pytest.approx(100.0, abs=1e-12)
    assert result["pct_from_high"] == pytest.approx(-20.0, abs=1e-9)
    assert result["pct_from_low"] == pytest.approx(20.0, abs=1e-9)
    assert result["position_pct"] == pytest.approx(40.0, abs=1e-9)
    assert result["n_obs"] == 3


def test_position_pct_stays_inside_zero_and_one_hundred(ohlcv: pd.DataFrame):
    # The last close is a member of its own window, so it can never sit
    # outside that window's own high/low. A value outside [0, 100] would mean
    # the range and the price were measured over different samples.
    result = distance_from_52w(ohlcv["Close"])
    assert 0.0 <= result["position_pct"] <= 100.0
    assert result["pct_from_high"] <= 0.0
    assert result["pct_from_low"] >= 0.0
    assert result["low"] <= result["last"] <= result["high"]


def test_position_pct_is_one_hundred_at_a_new_high_and_zero_at_a_new_low():
    assert distance_from_52w(pd.Series([1.0, 2.0, 3.0]))["position_pct"] == pytest.approx(
        100.0, abs=1e-12
    )
    assert distance_from_52w(pd.Series([3.0, 2.0, 1.0]))["position_pct"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_the_window_is_the_trailing_fifty_two_weeks_not_the_whole_history():
    # REGRESSION GUARD. A spike two years ago is not part of the 52-week
    # range; including it would understate position_pct forever.
    assert WEEKS_PER_YEAR == 52
    index = pd.bdate_range("2022-01-03", periods=700)
    series = pd.Series(np.linspace(100.0, 200.0, 700), index=index)
    series.iloc[0] = 1000.0  # an ancient high, far outside the trailing year
    result = distance_from_52w(series)
    assert result["high"] == pytest.approx(200.0, abs=1e-9)
    assert result["n_obs"] == 261
    assert result["n_obs"] < len(series)


def test_a_non_datetime_index_falls_back_to_the_trading_day_count():
    assert TRADING_DAYS == 252
    series = pd.Series(np.arange(1.0, 401.0))  # RangeIndex, 400 observations
    assert distance_from_52w(series)["n_obs"] == 252


def test_a_flat_range_reports_an_undefined_position(flat_series: pd.Series):
    # high == low, so (last - low)/(high - low) is 0/0. The percentages from
    # the high and the low are still honestly 0.
    result = distance_from_52w(flat_series)
    assert math.isnan(result["position_pct"])
    assert result["pct_from_high"] == pytest.approx(0.0, abs=1e-12)
    assert result["pct_from_low"] == pytest.approx(0.0, abs=1e-12)


def test_distance_from_52w_degrades_for_empty_and_all_nan_input(
    empty_series: pd.Series, nan_series: pd.Series
):
    for source in (empty_series, nan_series, None):
        result = distance_from_52w(source)
        assert result["n_obs"] == 0
        assert math.isnan(result["last"])
        assert math.isnan(result["position_pct"])


def test_distance_from_52w_ignores_gaps_rather_than_reading_them_as_zero():
    # A NaN is a non-trading day, not a price of 0. If it were treated as a
    # price, the low would collapse to 0 and position_pct would jump to ~100.
    series = pd.Series([100.0, np.nan, 150.0, 120.0], index=dates(4))
    result = distance_from_52w(series)
    assert result["low"] == pytest.approx(100.0, abs=1e-12)
    assert result["n_obs"] == 3


# ---- the assembled indicator frame ---------------------------------------
def test_the_indicator_frame_carries_exactly_the_documented_columns(ohlcv: pd.DataFrame):
    frame = indicator_frame(ohlcv)
    assert list(frame.columns) == indicator_columns()
    assert list(frame.columns) == [
        "close",
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA12",
        "EMA26",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "BB_BANDWIDTH",
        "BB_PERCENT_B",
        "ATR",
    ]


def test_the_indicator_frame_is_aligned_to_the_input_index(ohlcv: pd.DataFrame):
    frame = indicator_frame(ohlcv)
    assert frame.index.equals(ohlcv.index)
    assert len(frame) == len(ohlcv)
    assert frame["close"].to_numpy() == pytest.approx(ohlcv["Close"].to_numpy(), rel=1e-15)


def test_every_indicator_frame_column_matches_its_standalone_function(
    ohlcv: pd.DataFrame,
):
    # The frame is a convenience wrapper, never a second implementation.
    close = ohlcv["Close"]
    frame = indicator_frame(ohlcv)
    bands = bollinger_bands(close, 20, 2.0)
    macd_frame = macd(close, 12, 26, 9)
    for column, expected in (
        ("SMA20", sma(close, 20)),
        ("SMA200", sma(close, 200)),
        ("EMA12", ema(close, 12)),
        ("RSI", rsi(close, 14)),
        ("MACD", macd_frame["macd"]),
        ("MACD_SIGNAL", macd_frame["signal"]),
        ("MACD_HIST", macd_frame["histogram"]),
        ("BB_UPPER", bands["upper"]),
        ("BB_PERCENT_B", bands["percent_b"]),
        ("ATR", atr(ohlcv, 14)),
    ):
        left = frame[column]
        assert left.notna().any(), column
        assert left.dropna().to_numpy() == pytest.approx(expected.dropna().to_numpy(), rel=1e-12), (
            column
        )


def test_the_indicator_frame_layout_follows_the_config(ohlcv: pd.DataFrame):
    # Periods are config-driven so the assumptions live in one auditable place;
    # the column NAMES have to follow, or the chart layer binds to nothing.
    cfg = TechnicalConfig(
        sma_windows=(5,),
        ema_windows=(3,),
        rsi_period=4,
        macd_fast=3,
        macd_slow=6,
        macd_signal=2,
        bollinger_period=5,
        bollinger_std=1.5,
        atr_period=4,
    )
    assert indicator_columns(cfg)[:3] == ["close", "SMA5", "EMA3"]
    frame = indicator_frame(ohlcv, cfg)
    assert list(frame.columns) == indicator_columns(cfg)
    assert "SMA20" not in frame.columns
    # SMA5 needs 5 closes -> first value at row 4; ATR4 needs 4 true ranges.
    assert int(frame["SMA5"].notna().argmax()) == 4
    assert int(frame["ATR"].notna().argmax()) == 3


def test_the_indicator_frame_keeps_the_atr_column_for_a_close_only_frame(
    ohlcv: pd.DataFrame,
):
    # Fixed layout is the contract: the column set never changes with the
    # data, so the UI never has to branch on which columns turned up.
    frame = indicator_frame(ohlcv[["Close"]])
    assert list(frame.columns) == indicator_columns()
    assert frame["ATR"].isna().all()
    assert frame["RSI"].notna().any()


def test_the_indicator_frame_accepts_a_bare_close_series(ohlcv: pd.DataFrame):
    frame = indicator_frame(ohlcv["Close"])
    assert list(frame.columns) == indicator_columns()
    assert frame.index.equals(ohlcv.index)
    assert frame["ATR"].isna().all()


def test_the_indicator_frame_of_empty_input_keeps_its_columns():
    frame = indicator_frame(pd.DataFrame())
    assert list(frame.columns) == indicator_columns()
    assert frame.empty


def test_the_indicator_frame_does_not_mutate_the_caller(ohlcv: pd.DataFrame):
    before = ohlcv.copy(deep=True)
    indicator_frame(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


# ---- rule-based signals --------------------------------------------------
def test_every_signal_is_an_insight_tagged_technical_with_real_evidence(
    ohlcv: pd.DataFrame,
):
    signals = technical_signals(ohlcv)
    assert len(signals) == 5  # trend regime, cross, RSI, Bollinger, MACD
    for insight in signals:
        assert isinstance(insight, Insight)
        assert insight.category == CATEGORY == "technical"
        assert insight.evidence.strip(), insight.text
        assert insight.text.strip()
        assert insight.level in LEVEL_ORDER


def test_the_evidence_string_carries_both_the_observed_value_and_its_threshold(
    uptrend_ohlcv: pd.DataFrame,
):
    # An insight without the number that fired it is an opinion. RSI is pinned
    # at 100 on this series, so the overbought rule must fire and must quote
    # both 100 and the configured 70 level.
    signals = technical_signals(uptrend_ohlcv)
    rsi_insight = next(i for i in signals if i.evidence.startswith("RSI("))
    assert rsi_insight.level == "warning"
    assert "RSI(14)=100.0" in rsi_insight.evidence
    assert f"overbought {TECHNICALS.rsi_overbought:.1f}" in rsi_insight.evidence


def test_signals_are_ordered_most_actionable_first(uptrend_ohlcv: pd.DataFrame):
    # warning before positive before info -- the same ordering the rest of the
    # app uses, so the banner at the top of the page is the right sentence.
    signals = technical_signals(uptrend_ohlcv)
    ranks = [LEVEL_ORDER.index(i.level) for i in signals]
    assert ranks == sorted(ranks)
    assert "warning" in {i.level for i in signals}
    assert "positive" in {i.level for i in signals}


def test_a_strong_uptrend_reads_as_constructive_against_the_regime_filter(
    uptrend_ohlcv: pd.DataFrame,
):
    signals = technical_signals(uptrend_ohlcv)
    trend = next(i for i in signals if "moving average" in i.text and "cross" not in i.text)
    assert trend.level == "positive"
    assert "SMA200=" in trend.evidence


def test_a_short_history_yields_no_signals_rather_than_wrong_ones(ohlcv: pd.DataFrame):
    # Rules with insufficient history are skipped, not guessed at.
    assert technical_signals(ohlcv.head(1)) == []
    assert technical_signals(pd.DataFrame()) == []
    short = technical_signals(ohlcv.head(30))
    assert len(short) == 2  # only RSI(14) and Bollinger(20) can be formed
    assert all(i.category == CATEGORY for i in short)


def test_a_flat_market_reports_an_undefined_band_position_rather_than_a_number(
    flat_series: pd.Series,
):
    signals = technical_signals(pd.DataFrame({"Close": flat_series}))
    bollinger = next(i for i in signals if "Bollinger" in i.text)
    assert bollinger.level == "info"
    assert "collapsed to zero" in bollinger.text
    assert "%B undefined (0/0)" in bollinger.evidence


def test_the_cross_lookback_window_is_a_documented_constant(ohlcv: pd.DataFrame):
    assert CROSS_LOOKBACK_SESSIONS == 5
    cross = next(i for i in technical_signals(ohlcv) if "crossover" in i.text or "cross:" in i.text)
    assert f"{CROSS_LOOKBACK_SESSIONS} sessions" in cross.evidence


def test_technical_signals_do_not_mutate_the_caller(ohlcv: pd.DataFrame):
    before = ohlcv.copy(deep=True)
    technical_signals(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


# ---- chronological order (known gap) -------------------------------------
# Every function in this module is POSITIONAL: it reads row n-1 as "yesterday"
# and row n as "today" without ever sorting the index. src.analytics.risk, by
# contrast, sorts before any path-dependent measurement. A price table handed
# over newest-first -- the default export order of most broker and exchange
# CSVs -- therefore produces silently wrong indicators instead of an error.
# The two tests below assert the behaviour that would make the module safe.


# Regression guard for a fixed defect:
# technicals never sorts its index: sma/ema/rsi/atr are positional, so a newest-first
# price table yields silently wrong indicators. src.analytics.risk sorts first; this
# module does not.
def test_the_moving_average_should_not_depend_on_the_row_order(ohlcv: pd.DataFrame):
    close = ohlcv["Close"]
    reversed_close = close.iloc[::-1]
    assert float(sma(reversed_close, 20).dropna().iloc[-1]) == pytest.approx(
        float(sma(close, 20).dropna().iloc[-1]), rel=1e-12
    )


# Regression guard for a fixed defect:
# distance_from_52w anchors its trailing window on values.index[-1], the last ROW rather
# than the latest DATE. On a newest-first series that is the OLDEST bar, so 'last' is the
# oldest close and the window spans the whole history instead of 52 weeks.
def test_the_52_week_window_should_anchor_on_the_latest_date():
    index = pd.bdate_range("2022-01-03", periods=700)
    series = pd.Series(np.linspace(100.0, 200.0, 700), index=index)
    ascending = distance_from_52w(series)
    descending = distance_from_52w(series.iloc[::-1])
    assert descending["last"] == pytest.approx(ascending["last"], rel=1e-12)
    assert descending["n_obs"] == ascending["n_obs"]


# ---- module conventions (regression guards) ------------------------------
def test_the_shipped_indicator_periods_are_the_industry_defaults():
    # Changing any of these silently changes every signal the product emits.
    assert TECHNICALS.sma_windows == (20, 50, 200)
    assert TECHNICALS.ema_windows == (12, 26)
    assert TECHNICALS.rsi_period == 14
    assert (TECHNICALS.macd_fast, TECHNICALS.macd_slow, TECHNICALS.macd_signal) == (12, 26, 9)
    assert TECHNICALS.bollinger_period == 20
    assert TECHNICALS.bollinger_std == 2.0
    assert TECHNICALS.atr_period == 14
    assert (TECHNICALS.rsi_oversold, TECHNICALS.rsi_overbought) == (30.0, 70.0)


def test_every_insight_this_module_emits_shares_one_category():
    assert CATEGORY == "technical"
