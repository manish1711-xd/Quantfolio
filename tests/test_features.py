"""Feature construction: the properties a forecaster's inputs must have.

Three of the tests here are the whole reason this module exists as a separate
layer, and each one guards a failure that is invisible in the output:

* :func:`test_truncating_the_history_does_not_change_the_last_feature_row` is
  the **lookahead test**. If any feature peeks at a future bar, computing it on
  ``prices[:k]`` gives a different answer from computing it on the full series
  and reading row ``k - 1``. A model with lookahead reports superb accuracy and
  is worthless, and nothing downstream can detect it.

* :func:`test_every_price_feature_is_invariant_to_the_price_level` pins the
  scale-free claim. A feature that carries the price level lets a model learn
  "the number went up, so it goes up", which fits a trend perfectly in sample.

* :func:`test_a_newest_first_price_table_is_sorted_before_anything_is_computed`
  is the same trap the technicals module already has a regression guard for:
  most brokers export newest-first, and an unsorted momentum feature is
  confidently backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv, make_prices
from src.analytics import features as feat
from src.analytics import technicals as tech
from src.config import FEATURES, TECHNICALS

#: Long enough for the 252-session lookbacks to fill with room to spare.
DAYS = 900


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_ohlcv("AAA", days=DAYS, seed=4242)


@pytest.fixture(scope="module")
def matrix(bars: pd.DataFrame) -> pd.DataFrame:
    return feat.build_features(bars)


# ---------------------------------------------------------------------------
# shape and completeness
# ---------------------------------------------------------------------------
def test_the_matrix_spans_the_input_index_and_names_every_column(
    bars: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    assert len(matrix) == len(bars)
    assert matrix.index.equals(bars.index)
    assert matrix.shape[1] > 30, "a full OHLCV history should build the whole catalogue"
    assert all(isinstance(c, str) and c for c in matrix.columns)
    assert matrix.dtypes.eq("float64").all()


def test_early_rows_are_missing_rather_than_filled(matrix: pd.DataFrame) -> None:
    """A window that has not filled yet is NaN, never back-filled or zeroed.

    Forward-filling an unavailable 200-day average would invent a trend signal
    out of the first observation, which is worse than having no feature.
    """
    assert matrix.iloc[0].isna().any()
    # The longest lookback is the 252-session momentum/range pair, so every
    # row before that must carry at least one missing value.
    assert matrix.iloc[: FEATURES.range_window - 1].isna().any(axis=1).all()


def test_the_supervised_matrix_carries_no_missing_feature_at_all(bars: pd.DataFrame) -> None:
    x, y = feat.supervised_matrix(bars, 21)
    assert not x.empty
    assert x.isna().sum().sum() == 0
    assert len(x) == len(y)
    assert x.index.equals(y.index)


def test_the_last_horizon_rows_are_predictable_from_but_not_trainable_on(
    bars: pd.DataFrame,
) -> None:
    """``y.notna()`` is the boundary, and it must be exactly ``horizon`` rows wide."""
    horizon = 13
    x, y = feat.supervised_matrix(bars, horizon)
    assert int(y.isna().sum()) == horizon
    assert y.iloc[-horizon:].isna().all()
    assert y.iloc[:-horizon].notna().all()
    # The final row has complete features and no label: that is the live
    # forecast row, and dropping it would leave nothing to predict from.
    assert not feat.latest_row(x).empty
    assert feat.latest_row(x).index[0] == x.index[-1]


# ---------------------------------------------------------------------------
# THE lookahead test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [400, 600, 850])
def test_truncating_the_history_does_not_change_the_last_feature_row(
    bars: pd.DataFrame, matrix: pd.DataFrame, cut: int
) -> None:
    """No feature may depend on a bar that had not happened yet.

    Computed on the whole series, row ``cut - 1`` must equal the last row of
    the matrix computed on ``bars[:cut]``. Any centred window, any ``bfill``,
    any statistic fitted on the full sample breaks this immediately -- and
    nothing downstream could tell, because a leaking model simply looks good.
    """
    truncated = feat.build_features(bars.iloc[:cut])
    full_row = matrix.iloc[cut - 1]
    cut_row = truncated.iloc[-1]

    assert list(cut_row.index) == [c for c in matrix.columns if c in cut_row.index]
    pd.testing.assert_series_equal(
        cut_row.dropna(),
        full_row[cut_row.dropna().index],
        check_names=False,
        rtol=1e-12,
        atol=0.0,
    )


def test_the_label_never_leaks_into_the_feature_matrix(bars: pd.DataFrame) -> None:
    x, _ = feat.supervised_matrix(bars, 5)
    assert feat.TARGET_NAME not in x.columns


# ---------------------------------------------------------------------------
# the label itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_the_label_is_exactly_the_forward_log_return(bars: pd.DataFrame, horizon: int) -> None:
    target = feat.forward_log_return(bars, horizon)
    close = bars["Close"]
    for position in (0, 100, len(close) - horizon - 1):
        expected = np.log(float(close.iloc[position + horizon]) / float(close.iloc[position]))
        assert target.iloc[position] == pytest.approx(expected, rel=1e-12)


def test_log_labels_are_additive_across_time(bars: pd.DataFrame) -> None:
    """A 5-day log label is the sum of its five daily parts, exactly.

    That additivity is the whole reason the target is a log return rather than
    a simple one, so it is pinned rather than assumed.
    """
    daily = feat.forward_log_return(bars, 1)
    five = feat.forward_log_return(bars, 5)
    rebuilt = sum(daily.shift(-step) for step in range(5))
    pd.testing.assert_series_equal(five.dropna(), rebuilt.dropna(), check_names=False, rtol=1e-12)


# ---------------------------------------------------------------------------
# scale invariance
# ---------------------------------------------------------------------------
def test_every_price_feature_is_invariant_to_the_price_level(bars: pd.DataFrame) -> None:
    """Doubling every price must not move a single price-derived feature.

    This is the property that lets a model trained on a 200-rupee stock mean
    anything on a 4,000-rupee one -- and its absence is how a model learns the
    trend of the price axis instead of the behaviour of the instrument.
    """
    scaled = bars.copy()
    for column in ("Open", "High", "Low", "Close"):
        scaled[column] = scaled[column] * 7.5

    original = feat.build_features(bars).dropna(how="any")
    rescaled = feat.build_features(scaled).dropna(how="any")
    assert not original.empty

    pd.testing.assert_frame_equal(original, rescaled, rtol=1e-9, atol=1e-12)


def test_volume_features_are_invariant_to_the_volume_level(bars: pd.DataFrame) -> None:
    """A z-score and a ratio of averages are both unit-free. Pin it."""
    scaled = bars.copy()
    scaled["Volume"] = scaled["Volume"] * 1000.0
    original = feat.build_features(bars).dropna(how="any")
    rescaled = feat.build_features(scaled).dropna(how="any")
    volume_columns = [c for c in original.columns if feat.family_of(c) == "volume"]
    assert volume_columns
    pd.testing.assert_frame_equal(
        original[volume_columns], rescaled[volume_columns], rtol=1e-9, atol=1e-12
    )


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------
def test_a_newest_first_price_table_is_sorted_before_anything_is_computed(
    bars: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    """Most brokers export newest-first. An unsorted momentum feature is backwards."""
    reversed_bars = bars.iloc[::-1]
    assert reversed_bars.index[0] > reversed_bars.index[-1]
    pd.testing.assert_frame_equal(feat.build_features(reversed_bars), matrix)


def test_a_duplicated_session_keeps_its_last_observation(bars: pd.DataFrame) -> None:
    doubled = pd.concat([bars, bars.iloc[[-1]]])
    built = feat.build_features(doubled)
    assert len(built) == len(bars)


# ---------------------------------------------------------------------------
# cross-module agreement
# ---------------------------------------------------------------------------
def test_the_rsi_feature_is_the_same_number_the_security_page_draws(
    bars: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    """Two modules, two implementations of Wilder smoothing, one answer.

    ``features`` reimplements the smoothing rather than importing ``technicals``
    so it stays a leaf of the dependency graph. That choice is only defensible
    while the two agree, so the agreement is a test.
    """
    name = f"rsi_{TECHNICALS.rsi_period}"
    recentred = matrix[name] * 50.0 + 50.0
    expected = tech.rsi(bars["Close"])
    pd.testing.assert_series_equal(
        recentred.dropna(),
        expected.reindex(recentred.dropna().index),
        check_names=False,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# what an input can support
# ---------------------------------------------------------------------------
def test_a_close_only_series_drops_the_volume_family_rather_than_inventing_it() -> None:
    """A portfolio value series has no bar and no volume. Say so; do not fake it."""
    close = make_prices(("AAA",), days=DAYS).iloc[:, 0]
    families = feat.available_families(close)
    assert "volume" not in families
    assert "momentum" in families

    built = feat.build_features(close)
    assert not any(feat.family_of(c) == "volume" for c in built.columns)
    assert "bar_range" not in built.columns  # needs High/Low
    assert "mom_21" in built.columns


def test_a_positional_index_drops_the_calendar_family() -> None:
    close = pd.Series(make_prices(("AAA",), days=DAYS).iloc[:, 0].to_numpy())
    assert "calendar" not in feat.available_families(close)
    built = feat.build_features(close)
    assert not any(feat.family_of(c) == "calendar" for c in built.columns)


def test_every_generated_column_has_a_family_and_a_description(matrix: pd.DataFrame) -> None:
    """No feature may reach the importance table as an unexplained string."""
    for name in matrix.columns:
        assert feat.family_of(name) != "other", name
        assert feat.family_of(name) in feat.FEATURE_FAMILIES, name
        description = feat.describe_feature(name)
        assert description and description != "Derived price feature.", name


def test_the_catalogue_describes_exactly_the_columns_that_get_built(
    bars: pd.DataFrame, matrix: pd.DataFrame
) -> None:
    catalogue = feat.feature_catalogue(bars)
    assert set(catalogue["feature"]) == set(matrix.columns)
    assert list(catalogue.columns) == ["feature", "family", "description"]


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        None,
        pd.DataFrame(),
        pd.Series(dtype="float64"),
        pd.Series([100.0]),
        pd.Series([np.nan] * 40, index=pd.bdate_range("2023-01-02", periods=40)),
        "not a price series",
    ],
    ids=["none", "empty-frame", "empty-series", "one-row", "all-nan", "garbage"],
)
def test_unusable_input_yields_an_empty_matrix_rather_than_raising(bad: object) -> None:
    built = feat.build_features(bad)
    assert isinstance(built, pd.DataFrame)
    assert built.shape[1] == 0 or built.empty

    x, y = feat.supervised_matrix(bad, 5)
    assert x.empty
    assert y.empty
    assert feat.latest_row(x).empty


def test_a_short_history_drops_the_long_windows_instead_of_padding_them(
    matrix: pd.DataFrame,
) -> None:
    """A 252-session feature on 120 rows is dropped, never back-filled.

    The matrix therefore gets NARROWER on a short history rather than getting
    fabricated columns — and it is the forecaster, not this layer, that refuses
    to fit when what remains is too little to validate.
    """
    short = make_ohlcv("AAA", days=120)
    x, _ = feat.supervised_matrix(short, 5)

    assert x.shape[1] < matrix.shape[1]
    for dropped in ("mom_252", "range_position_252", "mom_12_1"):
        assert dropped in matrix.columns
        assert dropped not in x.columns
    assert x.isna().sum().sum() == 0, "what survives must still be complete"

    # And the layer that owns the decision refuses, with a reason that names
    # the shortfall rather than a silent empty chart.
    from src.analytics import forecast as fcst

    run = fcst.run_forecast(short, 5, "ridge")
    assert run["ok"] is False
    assert "rows" in run["reason"]


def test_a_zero_or_negative_price_is_dropped_not_logged() -> None:
    """``log(0)`` is not a feature value. The bar goes, quietly and completely."""
    bars = make_ohlcv("AAA", days=400)
    bars.iloc[50, bars.columns.get_loc("Close")] = 0.0
    bars.iloc[51, bars.columns.get_loc("Close")] = -5.0
    built = feat.build_features(bars)
    assert len(built) == len(bars) - 2
    assert np.isfinite(built.dropna(how="any").to_numpy()).all()


def test_the_caller_s_frame_is_never_modified(bars: pd.DataFrame) -> None:
    before = bars.copy()
    feat.build_features(bars)
    feat.supervised_matrix(bars, 21)
    feat.forward_log_return(bars, 21)
    pd.testing.assert_frame_equal(bars, before)
