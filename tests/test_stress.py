"""Stress testing — ``src.analytics.stress``.

A stress table is read by someone deciding how much risk they can live with, so
the way it handles what it does NOT know matters as much as its arithmetic. The
failure mode this module exists to prevent is a specific and seductive one: a
portfolio bought in 2021 has no 2008 in it, and a naive replay scores every
missing holding as flat, quietly converting "I have no idea what this name did
in the GFC" into "this name did nothing in the GFC" — a fabricated 0% that makes
the book look sturdier than it is.

Verified here
-------------
* **Coverage honesty.** A window entirely outside the price history reports
  ``coverage = 0``, a NaN return and a note saying so, and is RETAINED in the
  table (a silently dropped row reads as a scenario that did no damage). A
  partially covered window reports the covered sleeve's renormalised return
  next to the weight fraction that sleeve represents, and says in words that
  the rest is unknown rather than unharmed.
* **Replay arithmetic**, hand-checked against the price frame: a single-asset
  book replays exactly ``P_last / P_first - 1`` over the window's own first and
  last observation; a multi-asset book is the weight-renormalised average of
  those; ``value_change`` is ``portfolio_value * r_p``.
* **The beta ladder is pure algebra**: ``value_change == beta * shock * value``
  and ``projected_value == value + value_change``, asserted exactly, plus
  linearity in beta.
* **Non-overlap of the worst windows.** The greedy scan must never report the
  same bad fortnight five times with the start date nudged by a day, so the
  selected windows are asserted pairwise disjoint in RETURN space — they may
  share a single boundary price (one window's close is the next one's open) but
  never a return day.
* **Documented degradation**: empty prices, a positional index, a reversed
  window, weights that sum to zero, a series shorter than the window, an
  undated series and a non-finite beta each get a test pinning the NaN / empty
  / explanatory-note answer rather than an exception.

Sign and unit conventions are treated as regression guards: anything ``*_pct``
is in PERCENT, ``coverage`` is a FRACTION in [0, 1], and value columns are in
the currency of the ``portfolio_value`` passed in.

Everything runs on the seeded synthetic ``long_prices`` fixture (2015-01-01
onward), which covers five of the seven configured windows and misses the other
two — exactly the mix needed to test both halves of the coverage contract.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.stress import (
    BETA_SHOCK_COLUMNS,
    DEFAULT_STRESS_WINDOW_DAYS,
    HISTORICAL_COLUMNS,
    MIN_WINDOW_OBSERVATIONS,
    WORST_WINDOW_COLUMNS,
    beta_shock,
    historical_stress,
    scenario_from_returns,
    stress_summary,
    worst_rolling_windows,
)
from src.config import FACTOR_SHOCKS, HISTORICAL_STRESS, StressWindow

# ---------------------------------------------------------------------------
# the two windows this file leans on, pulled from config rather than retyped
# ---------------------------------------------------------------------------
#: 2020-02-19 -> 2020-03-23. Inside the ``long_prices`` history.
COVID = HISTORICAL_STRESS[0]
#: 2008-01-01 -> 2009-03-09. Seven years BEFORE the history starts.
GFC = HISTORICAL_STRESS[2]

WEIGHTS = {"AAA": 60.0, "BBB": 30.0, "CCC": 10.0}
VALUE = 1_000_000.0


# ---------------------------------------------------------------------------
# local helpers
# ---------------------------------------------------------------------------
def window_slice(prices: pd.DataFrame, window: StressWindow) -> pd.DataFrame:
    """The rows of ``prices`` that fall inside ``window``, inclusive."""
    mask = (prices.index >= pd.Timestamp(window.start)) & (prices.index <= pd.Timestamp(window.end))
    return prices.loc[mask]


def total_return_pct(prices: pd.DataFrame, window: StressWindow, ticker: str) -> float:
    """``P_last / P_first - 1`` in PERCENT, computed straight off the frame."""
    column = window_slice(prices, window)[ticker].dropna()
    return float(column.iloc[-1] / column.iloc[0] - 1.0) * 100.0


@pytest.fixture
def covid_only(long_prices: pd.DataFrame) -> pd.DataFrame:
    """A one-row historical table for the COVID window at the shared weights."""
    return historical_stress(long_prices, WEIGHTS, windows=(COVID,), portfolio_value=VALUE)


@pytest.fixture
def blanked_prices(long_prices: pd.DataFrame) -> pd.DataFrame:
    """``long_prices`` with BBB's prices removed inside the COVID window only.

    Models the realistic case of a holding that had not listed yet: it exists
    in the frame but has no observations in the crisis.
    """
    frame = long_prices.copy()
    mask = (frame.index >= pd.Timestamp(COVID.start)) & (frame.index <= pd.Timestamp(COVID.end))
    frame.loc[mask, "BBB"] = np.nan
    return frame


@pytest.fixture
def five_point_series() -> pd.Series:
    """100 -> 90 -> 95 -> 80 -> 120 on five consecutive business days."""
    return pd.Series(
        [100.0, 90.0, 95.0, 80.0, 120.0], index=pd.bdate_range("2023-01-02", periods=5)
    )


# ---- coverage: the honesty contract --------------------------------------
def test_a_window_outside_the_price_history_reports_zero_coverage_and_no_return(
    long_prices: pd.DataFrame,
):
    # long_prices begins 2015-01-01; the GFC window ended 2009-03-09, so not a
    # single observation overlaps. The honest answer is "0% of this book is
    # covered", NOT "the book was unharmed by the GFC".
    table = historical_stress(long_prices, WEIGHTS, windows=(GFC,), portfolio_value=VALUE)
    row = table.iloc[0]
    assert row["coverage"] == 0.0
    assert math.isnan(row["portfolio_return_pct"])
    assert math.isnan(row["value_change"])
    assert math.isnan(row["worst_asset_pct"])
    assert row["worst_asset"] is None
    assert row["days"] == 0
    assert pd.isna(row["start"]) and pd.isna(row["end"])


def test_an_uncovered_window_says_why_in_words_and_keeps_its_row(
    long_prices: pd.DataFrame,
):
    # Retained deliberately: a missing row reads as a scenario that did no
    # damage, which is the same fabrication in a different shape.
    table = historical_stress(long_prices, WEIGHTS, windows=(GFC,))
    assert len(table) == 1
    note = str(table.loc[0, "note"])
    assert "0% of portfolio weight is covered" in note
    assert "2008-01-01" in note and "2009-03-09" in note
    assert table.attrs["skipped"] == [GFC.name]
    assert table.attrs["n_skipped"] == 1
    assert table.attrs["covered"] == []


def test_the_default_window_set_splits_into_covered_and_uncovered_honestly(
    long_prices: pd.DataFrame,
):
    # Of the seven configured windows, the 2015-2024 history covers five; the
    # GFC (2008) and the taper tantrum (2013) predate it entirely.
    table = historical_stress(long_prices, WEIGHTS, portfolio_value=VALUE)
    assert len(table) == len(HISTORICAL_STRESS)
    assert table.attrs["skipped"] == ["Global financial crisis", "Taper tantrum"]
    assert table.attrs["n_skipped"] == 2
    assert len(table.attrs["covered"]) == 5
    # Every scored row is scored; every skipped row is blank. No middle ground.
    scored = table.loc[table["coverage"] > 0.0]
    blank = table.loc[table["coverage"] == 0.0]
    assert scored["portfolio_return_pct"].notna().all()
    assert blank["portfolio_return_pct"].isna().all()


def test_coverage_is_one_when_every_holding_has_data(covid_only: pd.DataFrame):
    assert covid_only.loc[0, "coverage"] == pytest.approx(1.0, abs=1e-12)
    # A fully covered row carries no caveat beyond the scenario's own note.
    assert str(covid_only.loc[0, "note"]) == COVID.note


def test_coverage_is_a_fraction_in_the_unit_interval(long_prices: pd.DataFrame):
    # UNIT REGRESSION GUARD. coverage is a FRACTION, not a percentage: a table
    # reporting 70 where the UI expects 0.7 would render "7000% covered".
    table = historical_stress(long_prices, WEIGHTS, portfolio_value=VALUE)
    assert ((table["coverage"] >= 0.0) & (table["coverage"] <= 1.0)).all()


def test_coverage_is_the_weight_fraction_that_actually_had_data(
    blanked_prices: pd.DataFrame,
):
    # Weights 60/30/10 normalise to 0.6/0.3/0.1. BBB has no price inside the
    # window, so the covered weight is 0.6 + 0.1 = 0.7.
    table = historical_stress(blanked_prices, WEIGHTS, windows=(COVID,), portfolio_value=VALUE)
    assert table.loc[0, "coverage"] == pytest.approx(0.70, abs=1e-12)


def test_a_partially_covered_return_is_the_renormalised_covered_sleeve(
    blanked_prices: pd.DataFrame,
):
    # r_p = (w_AAA * r_AAA + w_CCC * r_CCC) / (w_AAA + w_CCC)
    #     = (0.6 * r_AAA + 0.1 * r_CCC) / 0.7
    # Dividing by 0.7 rather than 1.0 is the whole point: BBB is excluded from
    # the denominator instead of being scored as a flat 0%.
    r_aaa = total_return_pct(blanked_prices, COVID, "AAA")
    r_ccc = total_return_pct(blanked_prices, COVID, "CCC")
    expected = (0.6 * r_aaa + 0.1 * r_ccc) / 0.7
    table = historical_stress(blanked_prices, WEIGHTS, windows=(COVID,))
    assert table.loc[0, "portfolio_return_pct"] == pytest.approx(expected, rel=1e-12)
    # Scoring the missing name as flat would have divided by 1.0 instead:
    naive = 0.6 * r_aaa + 0.1 * r_ccc
    assert table.loc[0, "portfolio_return_pct"] != pytest.approx(naive, rel=1e-6)


def test_a_partially_covered_row_states_the_caveat_in_plain_english(
    blanked_prices: pd.DataFrame,
):
    table = historical_stress(blanked_prices, WEIGHTS, windows=(COVID,))
    note = str(table.loc[0, "note"])
    assert COVID.note in note
    assert "Only 70% of portfolio weight had price history here" in note
    assert "(2 of 3 holdings)" in note


# Regression guard for a fixed defect:
# A holding with NO column at all in the price frame (a failed fetch, a newly listed name)
# is dropped by _weight_map and the survivors are renormalised, so coverage reports 1.0
# for a book that is only 70% priced. The module's stated contract is that unpriced weight
# is reported as uncovered, never silently removed from the denominator.
def test_a_holding_missing_from_the_price_frame_counts_as_uncovered(
    long_prices: pd.DataFrame,
):
    # AAA 35 / BBB 35 / ZZZ 30, but ZZZ was never fetched. 30% of the book has
    # no price history anywhere, so the covered fraction is 0.70 — the same
    # answer the module gives when a name is present but blank in the window.
    table = historical_stress(
        long_prices[["AAA", "BBB"]],
        {"AAA": 35.0, "BBB": 35.0, "ZZZ": 30.0},
        windows=(COVID,),
    )
    assert table.loc[0, "coverage"] == pytest.approx(0.70, abs=1e-12)


# ---- replay arithmetic ----------------------------------------------------
def test_a_single_asset_portfolio_replays_that_assets_exact_total_return(
    long_prices: pd.DataFrame,
):
    """One holding means no weighting at all: the portfolio IS the asset.

    Worked from the fixture: over 2020-02-19 -> 2020-03-23 the AAA column has
    24 observations, opening at 271.64759462706814 and closing at
    292.4372237479921, so

        292.4372237479921 / 271.64759462706814 - 1 = 0.076531614975148

    i.e. +7.6531614975% — reported in PERCENT, and multiplied by the portfolio
    value to give the money column.
    """
    expected_pct = total_return_pct(long_prices, COVID, "AAA")
    table = historical_stress(
        long_prices, {"AAA": 1.0}, windows=(COVID,), portfolio_value=250_000.0
    )
    row = table.iloc[0]
    assert row["portfolio_return_pct"] == pytest.approx(expected_pct, rel=1e-12)
    assert row["portfolio_return_pct"] == pytest.approx(7.6531614975, abs=1e-8)
    assert row["coverage"] == pytest.approx(1.0, abs=1e-12)
    # With a single holding the worst asset is that holding, at the same number.
    assert row["worst_asset"] == "AAA"
    assert row["worst_asset_pct"] == pytest.approx(expected_pct, rel=1e-12)


def test_value_change_is_the_portfolio_value_times_the_sleeve_return(
    covid_only: pd.DataFrame,
):
    # value_change = 1,000,000 * r_p, with r_p the decimal behind the percent
    # column. Getting the /100 wrong here would inflate the money column 100x.
    row = covid_only.iloc[0]
    assert row["value_change"] == pytest.approx(
        VALUE * row["portfolio_return_pct"] / 100.0, rel=1e-12
    )


def test_value_change_is_nan_when_no_portfolio_value_was_supplied(
    long_prices: pd.DataFrame,
):
    # Honest blank rather than a zero that reads as "this cost you nothing".
    table = historical_stress(long_prices, WEIGHTS, windows=(COVID,))
    assert math.isnan(table.loc[0, "value_change"])
    assert math.isfinite(table.loc[0, "portfolio_return_pct"])


def test_the_multi_asset_return_is_the_weighted_average_of_its_holdings(
    long_prices: pd.DataFrame,
):
    # Full coverage, so the renormalisation denominator is 1 and the result is
    # the plain weighted mean:  0.6*r_AAA + 0.3*r_BBB + 0.1*r_CCC
    parts = {t: total_return_pct(long_prices, COVID, t) for t in ("AAA", "BBB", "CCC")}
    expected = 0.6 * parts["AAA"] + 0.3 * parts["BBB"] + 0.1 * parts["CCC"]
    table = historical_stress(long_prices, WEIGHTS, windows=(COVID,))
    assert table.loc[0, "portfolio_return_pct"] == pytest.approx(expected, rel=1e-12)


def test_the_worst_asset_is_the_covered_holding_with_the_lowest_return(
    long_prices: pd.DataFrame,
):
    parts = {t: total_return_pct(long_prices, COVID, t) for t in ("AAA", "BBB", "CCC")}
    worst = min(parts, key=lambda t: parts[t])
    table = historical_stress(long_prices, WEIGHTS, windows=(COVID,))
    assert table.loc[0, "worst_asset"] == worst
    assert table.loc[0, "worst_asset_pct"] == pytest.approx(parts[worst], rel=1e-12)


def test_the_replayed_dates_are_the_ones_actually_present_not_the_ones_requested(
    long_prices: pd.DataFrame, covid_only: pd.DataFrame
):
    # The window asks for 2020-02-19 -> 2020-03-23; the table must report the
    # first and last observation it really used, and count them.
    used = window_slice(long_prices, COVID)
    row = covid_only.iloc[0]
    assert row["start"] == used.index[0]
    assert row["end"] == used.index[-1]
    assert row["days"] == len(used) == 24


def test_percent_and_fraction_weights_replay_identically(long_prices: pd.DataFrame):
    # Weights are normalised by their own sum, so the caller never has to know
    # which convention the module wants.
    as_percent = historical_stress(long_prices, {"AAA": 60, "BBB": 30, "CCC": 10}, windows=(COVID,))
    as_fraction = historical_stress(
        long_prices, {"AAA": 0.6, "BBB": 0.3, "CCC": 0.1}, windows=(COVID,)
    )
    assert as_percent.loc[0, "portfolio_return_pct"] == pytest.approx(
        as_fraction.loc[0, "portfolio_return_pct"], rel=1e-12
    )


def test_a_sequence_of_weights_aligns_positionally_to_the_price_columns(
    long_prices: pd.DataFrame,
):
    positional = historical_stress(long_prices, [60.0, 30.0, 10.0], windows=(COVID,))
    named = historical_stress(long_prices, WEIGHTS, windows=(COVID,))
    assert positional.loc[0, "portfolio_return_pct"] == pytest.approx(
        named.loc[0, "portfolio_return_pct"], rel=1e-12
    )


def test_an_unsorted_price_frame_is_sorted_before_the_window_is_sliced(
    long_prices: pd.DataFrame, covid_only: pd.DataFrame
):
    # A CSV in descending date order must not invert the replay: P_first and
    # P_last are defined by the calendar, not by row position.
    rng = np.random.default_rng(3)
    shuffled = long_prices.iloc[rng.permutation(len(long_prices))]
    assert not shuffled.index.is_monotonic_increasing
    table = historical_stress(shuffled, WEIGHTS, windows=(COVID,), portfolio_value=VALUE)
    assert table.loc[0, "portfolio_return_pct"] == pytest.approx(
        covid_only.loc[0, "portfolio_return_pct"], rel=1e-12
    )
    assert table.loc[0, "start"] == covid_only.loc[0, "start"]


def test_historical_stress_does_not_mutate_the_price_frame(long_prices: pd.DataFrame):
    before = long_prices.copy(deep=True)
    historical_stress(long_prices, WEIGHTS, portfolio_value=VALUE)
    pd.testing.assert_frame_equal(long_prices, before)


# ---- schema and degradation ----------------------------------------------
def test_row_order_follows_the_window_order_not_the_severity(
    long_prices: pd.DataFrame,
):
    # The UI sorts for display; the function must not reorder underneath it,
    # or a caller pairing rows with the config list positionally would misread
    # every scenario.
    table = historical_stress(long_prices, WEIGHTS, portfolio_value=VALUE)
    assert list(table["scenario"]) == [w.name for w in HISTORICAL_STRESS]


def test_the_output_schema_is_stable_and_typed(covid_only: pd.DataFrame):
    assert list(covid_only.columns) == list(HISTORICAL_COLUMNS)
    assert covid_only["days"].dtype == np.int64
    assert covid_only["coverage"].dtype == np.float64
    assert str(covid_only["start"].dtype).startswith("datetime64")


@pytest.mark.parametrize(
    "prices",
    [None, pd.DataFrame(), pd.DataFrame({"AAA": []})],
    ids=["none", "empty", "no-rows"],
)
def test_empty_price_history_yields_the_full_schema_with_a_reason(prices):
    # The UI must never key-error on a missing column, so the schema survives
    # even when there is nothing at all to report.
    table = historical_stress(prices, WEIGHTS, windows=(COVID,))
    assert list(table.columns) == list(HISTORICAL_COLUMNS)
    assert len(table) == 1
    assert table.loc[0, "coverage"] == 0.0
    assert "no dated price history was supplied" in str(table.loc[0, "note"])
    assert pd.isna(table.attrs["history_start"])


def test_a_positional_index_is_refused_rather_than_read_as_epoch_dates(
    long_prices: pd.DataFrame,
):
    # Reading 0, 1, 2 as nanoseconds since 1970 would place the whole history
    # in January 1970 and "replay" every crisis against nothing.
    table = historical_stress(long_prices.reset_index(drop=True), WEIGHTS, windows=(COVID,))
    assert table.loc[0, "coverage"] == 0.0
    assert "no dated price history was supplied" in str(table.loc[0, "note"])


def test_a_reversed_window_is_reported_as_an_invalid_range(long_prices: pd.DataFrame):
    backwards = StressWindow("Backwards", "2020-03-23", "2020-02-19", "typo in config")
    table = historical_stress(long_prices, WEIGHTS, windows=(backwards,))
    assert table.loc[0, "coverage"] == 0.0
    assert "not a valid date range" in str(table.loc[0, "note"])


@pytest.mark.parametrize(
    "weights",
    [None, {}, {"AAA": 0.0, "BBB": 0.0}, {"ZZZ": 1.0}, ["a", "b", "c"]],
    ids=["none", "empty", "zero-sum", "unknown-names", "non-numeric"],
)
def test_unusable_weights_degrade_to_no_replay(long_prices: pd.DataFrame, weights):
    table = historical_stress(long_prices, weights, windows=(COVID,))
    assert table.loc[0, "coverage"] == 0.0
    assert math.isnan(table.loc[0, "portfolio_return_pct"])
    assert "no usable portfolio weights" in str(table.loc[0, "note"])
    assert table.attrs["n_holdings"] == 0


def test_no_windows_at_all_yields_an_empty_table_with_the_schema(
    long_prices: pd.DataFrame,
):
    table = historical_stress(long_prices, WEIGHTS, windows=())
    assert table.empty
    assert list(table.columns) == list(HISTORICAL_COLUMNS)


def test_a_window_holding_a_single_observation_is_not_a_return(
    long_prices: pd.DataFrame,
):
    # MIN_WINDOW_OBSERVATIONS = 2: one price is a price, not a return, and
    # reporting 0% for it would be a fabricated number.
    assert MIN_WINDOW_OBSERVATIONS == 2
    day = long_prices.index[100]
    one_day = StressWindow("Single day", str(day.date()), str(day.date()), "")
    table = historical_stress(long_prices, WEIGHTS, windows=(one_day,))
    assert table.loc[0, "coverage"] == 0.0
    assert math.isnan(table.loc[0, "portfolio_return_pct"])


def test_an_ad_hoc_window_may_be_supplied_as_a_mapping_or_a_tuple(
    long_prices: pd.DataFrame, covid_only: pd.DataFrame
):
    expected = covid_only.loc[0, "portfolio_return_pct"]
    as_mapping = historical_stress(
        long_prices,
        WEIGHTS,
        windows=({"name": COVID.name, "start": COVID.start, "end": COVID.end},),
    )
    as_tuple = historical_stress(
        long_prices, WEIGHTS, windows=((COVID.name, COVID.start, COVID.end),)
    )
    assert as_mapping.loc[0, "portfolio_return_pct"] == pytest.approx(expected, rel=1e-12)
    assert as_tuple.loc[0, "portfolio_return_pct"] == pytest.approx(expected, rel=1e-12)


# ---- beta shock: first-order CAPM algebra ---------------------------------
@pytest.mark.parametrize(("label", "shock"), FACTOR_SHOCKS)
def test_value_change_is_exactly_beta_times_shock_times_value(label, shock):
    # implied = beta * shock ; value_change = beta * shock * portfolio_value
    # With beta = 0.85 and a -20% index move on a 1,000,000 book:
    #   0.85 * -0.20            = -0.17      -> -17.00 %
    #   0.85 * -0.20 * 1_000_000 = -170,000
    table = beta_shock(VALUE, 0.85)
    row = table.loc[table["scenario"] == label].iloc[0]
    assert row["shock_pct"] == pytest.approx(shock * 100.0, rel=1e-12)
    assert row["implied_portfolio_pct"] == pytest.approx(0.85 * shock * 100.0, rel=1e-12)
    assert row["value_change"] == pytest.approx(0.85 * shock * VALUE, rel=1e-12)


def test_the_minus_twenty_percent_row_is_hand_computable():
    table = beta_shock(1_000_000.0, 0.85)
    row = table.loc[table["scenario"] == "Benchmark -20%"].iloc[0]
    assert row["implied_portfolio_pct"] == pytest.approx(-17.0, abs=1e-12)
    assert row["value_change"] == pytest.approx(-170_000.0, abs=1e-9)
    assert row["projected_value"] == pytest.approx(830_000.0, abs=1e-9)


def test_projected_value_is_the_value_plus_the_change():
    table = beta_shock(VALUE, 1.15)
    assert table["projected_value"].to_numpy() == pytest.approx(
        VALUE + table["value_change"].to_numpy(), rel=1e-12
    )


def test_the_shock_ladder_is_linear_in_beta():
    # Identity: doubling beta doubles every impact, because the model is one
    # multiplication. A stress table that is not linear in beta is not CAPM.
    single = beta_shock(VALUE, 0.60)
    double = beta_shock(VALUE, 1.20)
    assert double["value_change"].to_numpy() == pytest.approx(
        2.0 * single["value_change"].to_numpy(), rel=1e-12
    )


def test_a_zero_beta_book_is_untouched_by_any_benchmark_move():
    table = beta_shock(VALUE, 0.0)
    assert table["implied_portfolio_pct"].to_numpy() == pytest.approx(
        np.zeros(len(table)), abs=1e-12
    )
    assert table["projected_value"].to_numpy() == pytest.approx(
        np.full(len(table), VALUE), rel=1e-12
    )


def test_a_negative_beta_book_gains_when_the_benchmark_falls():
    row = beta_shock(VALUE, -0.40)
    minus_20 = row.loc[row["scenario"] == "Benchmark -20%"].iloc[0]
    assert minus_20["value_change"] == pytest.approx(+80_000.0, abs=1e-9)


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), None, "high"])
def test_an_unusable_beta_leaves_the_ladder_visible_with_blank_impacts(beta):
    # The scenario rows still render so the UI can show the ladder with blanks
    # rather than nothing at all; only the derived columns go NaN.
    table = beta_shock(VALUE, beta)
    assert len(table) == len(FACTOR_SHOCKS)
    assert table["shock_pct"].notna().all()
    assert table["implied_portfolio_pct"].isna().all()
    assert table["value_change"].isna().all()


def test_bare_shock_numbers_are_labelled_from_the_shock_itself():
    table = beta_shock(VALUE, 1.0, shocks=[-0.05, 0.10])
    assert list(table["scenario"]) == ["Benchmark -5%", "Benchmark +10%"]
    assert table["value_change"].to_numpy() == pytest.approx([-50_000.0, 100_000.0], rel=1e-12)


def test_shocks_may_be_supplied_as_a_mapping():
    table = beta_shock(VALUE, 0.5, shocks={"Custom -8%": -0.08})
    assert list(table["scenario"]) == ["Custom -8%"]
    assert table.loc[0, "value_change"] == pytest.approx(-40_000.0, abs=1e-9)


@pytest.mark.parametrize(
    "shocks",
    [None, {}, [], "minus ten"],
    ids=["none", "empty-map", "empty-list", "string"],
)
def test_no_shocks_yields_an_empty_ladder_with_the_schema(shocks):
    table = beta_shock(VALUE, 1.0, shocks=shocks)
    assert table.empty
    assert list(table.columns) == list(BETA_SHOCK_COLUMNS)


def test_the_beta_shock_frame_documents_its_own_method():
    table = beta_shock(VALUE, 0.85)
    assert table.attrs["beta"] == pytest.approx(0.85, abs=1e-12)
    assert table.attrs["portfolio_value"] == pytest.approx(VALUE, rel=1e-12)
    assert "first-order CAPM" in table.attrs["method"]


# ---- worst rolling windows ------------------------------------------------
def test_worst_rolling_windows_are_hand_computable(five_point_series: pd.Series):
    # 100 -> 90 -> 95 -> 80 -> 120 with a one-day holding period:
    #   (0,1)  90/100 - 1 = -10.000000 %
    #   (1,2)  95/ 90 - 1 =  +5.555556 %
    #   (2,3)  80/ 95 - 1 = -15.789474 %
    #   (3,4) 120/ 80 - 1 = +50.000000 %
    # Worst first: (2,3) then (0,1). They do not overlap, so both are reported.
    table = worst_rolling_windows(five_point_series, window_days=1, top_n=2)
    assert list(table["rank"]) == [1, 2]
    assert table["return_pct"].to_numpy() == pytest.approx([-15.789473684210527, -10.0], abs=1e-12)
    assert table["start_value"].to_numpy() == pytest.approx([95.0, 100.0], abs=1e-12)
    assert table["end_value"].to_numpy() == pytest.approx([80.0, 90.0], abs=1e-12)
    assert list(table["start"]) == [pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-02")]
    assert list(table["end"]) == [pd.Timestamp("2023-01-05"), pd.Timestamp("2023-01-03")]


def test_worst_rolling_windows_are_sorted_worst_first(long_prices: pd.DataFrame):
    values = long_prices.mul(pd.Series({"AAA": 10.0, "BBB": 5.0, "CCC": 8.0}), axis=1).sum(axis=1)
    table = worst_rolling_windows(values, window_days=21, top_n=5)
    assert list(table["rank"]) == [1, 2, 3, 4, 5]
    assert list(table["return_pct"]) == sorted(table["return_pct"])
    assert table.loc[0, "return_pct"] < 0.0


def test_selected_windows_never_share_a_return_day(long_prices: pd.DataFrame):
    """The non-overlap guarantee, asserted pairwise on the reported dates.

    Overlap is measured in RETURN space: a window covering observations
    ``(a, b]`` clashes with ``(c, d]`` unless ``b <= c`` or ``d <= a``. So two
    windows may touch at a single boundary price — one window's closing level
    is the next one's opening level — but they can never both contain the same
    day's move. Without this the same bad fortnight would be reported five
    times with the start date nudged by a day, and the table would claim five
    independent shocks where there was one.
    """
    values = long_prices.mul(pd.Series({"AAA": 10.0, "BBB": 5.0, "CCC": 8.0}), axis=1).sum(axis=1)
    table = worst_rolling_windows(values, window_days=21, top_n=5)
    assert len(table) == 5
    spans = list(zip(table["start"], table["end"]))
    for i, (start_i, end_i) in enumerate(spans):
        assert start_i < end_i
        for start_j, end_j in spans[i + 1 :]:
            assert end_i <= start_j or end_j <= start_i


def test_adjacent_windows_may_share_a_single_boundary_price():
    # 100 -> 80 -> 60 -> 100. The two worst one-day windows are (1,2) at -25%
    # and (0,1) at -20%; they meet at the 80 on 2023-01-03, which is the close
    # of one and the open of the other. Sharing a PRICE double-counts nothing;
    # sharing a RETURN would.
    values = pd.Series([100.0, 80.0, 60.0, 100.0], index=pd.bdate_range("2023-01-02", periods=4))
    table = worst_rolling_windows(values, window_days=1, top_n=2)
    assert table["return_pct"].to_numpy() == pytest.approx([-25.0, -20.0], abs=1e-12)
    assert table.loc[0, "start"] == table.loc[1, "end"] == pd.Timestamp("2023-01-03")


def test_return_pct_agrees_with_the_start_and_end_values_reported(
    long_prices: pd.DataFrame,
):
    # Identity keeping the row self-consistent: the percentage and the two
    # levels in the same row must describe the same window.
    values = long_prices["AAA"]
    table = worst_rolling_windows(values, window_days=21, top_n=5)
    expected = (table["end_value"] / table["start_value"] - 1.0) * 100.0
    assert table["return_pct"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-12)


def test_the_window_length_and_calendar_span_are_reported_separately(
    long_prices: pd.DataFrame,
):
    # 21 TRADING days is about 29 CALENDAR days; conflating the two would
    # misdate every stress window by a month a year.
    table = worst_rolling_windows(long_prices["AAA"], window_days=21, top_n=3)
    assert (table["days"] == 21).all()
    assert (table["calendar_days"] >= 21).all()
    for _, row in table.iterrows():
        assert row["calendar_days"] == pytest.approx(
            float((row["end"] - row["start"]).days), abs=1e-12
        )


def test_worst_rolling_windows_returns_at_most_top_n_rows(long_prices: pd.DataFrame):
    for n in (1, 3, 7):
        assert len(worst_rolling_windows(long_prices["AAA"], 21, top_n=n)) == n


def test_the_default_holding_period_is_one_trading_month():
    # Taken from the period table rather than written as a bare 21.
    assert DEFAULT_STRESS_WINDOW_DAYS == 21


@pytest.mark.parametrize(
    ("values", "window_days", "top_n"),
    [
        (pd.Series(dtype=float), 21, 5),  # nothing at all
        (pd.Series([1.0, 2.0, 3.0]), 5, 5),  # shorter than the window
        (pd.Series([1.0, 2.0, 3.0]), 3, 5),  # exactly one short of usable
        (pd.Series([1.0, 2.0, 3.0, 4.0]), 0, 5),  # non-positive window
        (pd.Series([1.0, 2.0, 3.0, 4.0]), 1, 0),  # nothing requested
        (None, 21, 5),  # no series
    ],
    ids=["empty", "too-short", "boundary", "zero-window", "zero-top-n", "none"],
)
def test_worst_rolling_windows_degrades_to_an_empty_table(values, window_days, top_n):
    table = worst_rolling_windows(values, window_days, top_n)
    assert table.empty
    assert list(table.columns) == list(WORST_WINDOW_COLUMNS)


def test_a_series_without_dates_reports_nat_and_says_why(
    five_point_series: pd.Series,
):
    # Honest blank dates plus a note, rather than inventing an index.
    table = worst_rolling_windows(pd.Series(five_point_series.to_numpy()), window_days=1, top_n=2)
    assert table["start"].isna().all()
    assert table["end"].isna().all()
    assert table["calendar_days"].isna().all()
    assert table.attrs["dated_index"] is False
    assert "no date index" in table.attrs["note"]
    # The returns themselves are unaffected — only the labelling is.
    assert table["return_pct"].to_numpy() == pytest.approx([-15.789473684210527, -10.0], abs=1e-12)


def test_an_unsorted_value_series_is_sorted_before_the_scan(long_prices: pd.DataFrame):
    # A value series handed over in descending date order must produce the same
    # worst windows: the scan is path-dependent, so row order cannot decide it.
    shuffled = long_prices["AAA"].iloc[np.random.default_rng(5).permutation(len(long_prices))]
    assert not shuffled.index.is_monotonic_increasing
    ordered = worst_rolling_windows(long_prices["AAA"], 21, top_n=3)
    from_shuffled = worst_rolling_windows(shuffled, 21, top_n=3)
    assert from_shuffled["return_pct"].to_numpy() == pytest.approx(
        ordered["return_pct"].to_numpy(), rel=1e-12
    )


# ---- scenario_from_returns ------------------------------------------------
@pytest.fixture
def tiny_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {"AAA": [0.01, -0.02, 0.03], "BBB": [0.0, 0.05, -0.01]},
        index=pd.bdate_range("2023-01-02", periods=3),
    )


def test_scenario_from_returns_compounds_geometrically(tiny_returns: pd.DataFrame):
    # AAA: 1.01 * 0.98 * 1.03 - 1 = 0.019494  -> +1.9494 %
    # BBB: 1.00 * 1.05 * 0.99 - 1 = 0.0395    -> +3.9500 %
    # Adding the returns instead (0.02 and 0.04) would be wrong by 26 bp on the
    # first name alone, and the error compounds with the window length.
    result = scenario_from_returns(tiny_returns, {"AAA": 0.5, "BBB": 0.5})
    assert result["ok"] is True
    assert result["asset_return_pct"]["AAA"] == pytest.approx(1.9494, abs=1e-9)
    assert result["asset_return_pct"]["BBB"] == pytest.approx(3.95, abs=1e-9)


def test_scenario_from_returns_is_the_weighted_average_of_covered_assets(
    tiny_returns: pd.DataFrame,
):
    # (1.9494 + 3.95) / 2 = 2.9497 %
    result = scenario_from_returns(tiny_returns, {"AAA": 50, "BBB": 50})
    assert result["portfolio_return_pct"] == pytest.approx(2.9497, abs=1e-9)
    assert result["coverage"] == pytest.approx(1.0, abs=1e-12)
    assert result["n_covered"] == 2
    assert result["days"] == 3


def test_scenario_from_returns_ranks_the_best_and_worst_holding(
    tiny_returns: pd.DataFrame,
):
    result = scenario_from_returns(tiny_returns, {"AAA": 0.5, "BBB": 0.5})
    assert result["worst_asset"] == "AAA"
    assert result["best_asset"] == "BBB"
    assert result["worst_asset_pct"] <= result["best_asset_pct"]


def test_scenario_from_returns_honours_an_explicit_date_range(
    tiny_returns: pd.DataFrame,
):
    # Restricted to the first two days: AAA is 1.01 * 0.98 - 1 = -0.0102.
    result = scenario_from_returns(tiny_returns, {"AAA": 1.0}, start="2023-01-02", end="2023-01-03")
    assert result["days"] == 2
    assert result["asset_return_pct"]["AAA"] == pytest.approx(-1.02, abs=1e-9)
    assert result["start"] == pd.Timestamp("2023-01-02")
    assert result["end"] == pd.Timestamp("2023-01-03")


def test_a_range_with_no_observations_is_reported_as_not_ok(
    tiny_returns: pd.DataFrame,
):
    result = scenario_from_returns(tiny_returns, {"AAA": 1.0}, start="2099-01-01", end="2099-02-01")
    assert result["ok"] is False
    assert result["coverage"] == 0.0
    assert math.isnan(result["portfolio_return_pct"])
    assert "no overlapping return history" in result["reason"]


@pytest.mark.parametrize(
    ("returns", "weights", "fragment"),
    [
        (pd.DataFrame(), {"AAA": 1.0}, "no usable return history"),
        (None, {"AAA": 1.0}, "no usable return history"),
        (pd.DataFrame({"AAA": [0.01, 0.02]}), None, "no usable portfolio weights"),
    ],
)
def test_scenario_from_returns_degrades_with_a_stated_reason(returns, weights, fragment):
    result = scenario_from_returns(returns, weights)
    assert result["ok"] is False
    assert fragment in result["reason"]
    assert math.isnan(result["portfolio_return_pct"])


def test_a_backwards_custom_range_is_refused(tiny_returns: pd.DataFrame):
    result = scenario_from_returns(tiny_returns, {"AAA": 1.0}, start="2023-01-04", end="2023-01-02")
    assert result["ok"] is False
    assert "ends before it starts" in result["reason"]


# ---- stress_summary -------------------------------------------------------
def test_stress_summary_worst_scenario_is_the_deepest_covered_row(
    long_prices: pd.DataFrame,
):
    summary = stress_summary(long_prices, WEIGHTS, VALUE, 0.9)
    historical = summary["historical"]
    covered = historical.loc[historical["portfolio_return_pct"].notna()]
    worst = covered.loc[covered["portfolio_return_pct"].idxmin()]
    assert summary["ok"] is True
    assert summary["worst_scenario"] == worst["scenario"]
    assert summary["worst_scenario_pct"] == pytest.approx(
        float(worst["portfolio_return_pct"]), rel=1e-12
    )
    assert summary["worst_scenario_value_change"] == pytest.approx(
        VALUE * summary["worst_scenario_pct"] / 100.0, rel=1e-9
    )


def test_mean_coverage_averages_every_scenario_including_the_uncovered_ones(
    long_prices: pd.DataFrame,
):
    # Five covered windows at coverage 1.0 and two at 0.0 average to 5/7.
    # Averaging only the covered rows would report a flattering 1.0 and hide
    # the two crises this book has never lived through.
    summary = stress_summary(long_prices, WEIGHTS, VALUE, 0.9)
    assert summary["n_scenarios"] == 7
    assert summary["n_covered"] == 5
    assert summary["n_skipped"] == 2
    assert summary["mean_coverage"] == pytest.approx(5.0 / 7.0, rel=1e-12)


def test_stress_summary_reports_the_books_own_realised_drawdown_for_comparison(
    long_prices: pd.DataFrame,
):
    # The keeps-it-honest number: a stress table whose worst scenario is milder
    # than what the book already survived is a coverage artefact, not safety.
    summary = stress_summary(long_prices, WEIGHTS, VALUE, 0.9)
    assert summary["sample_max_drawdown_pct"] < 0.0
    assert summary["n_observations"] == len(long_prices)
    assert summary["sufficient_history"] is True
    assert summary["history_start"] == long_prices.index[0]
    assert summary["history_end"] == long_prices.index[-1]


def test_stress_summary_carries_the_three_views_it_promises(long_prices: pd.DataFrame):
    summary = stress_summary(long_prices, WEIGHTS, VALUE, 0.9)
    assert list(summary["historical"].columns) == list(HISTORICAL_COLUMNS)
    assert list(summary["beta_shocks"].columns) == list(BETA_SHOCK_COLUMNS)
    assert list(summary["worst_windows"].columns) == list(WORST_WINDOW_COLUMNS)
    assert summary["beta"] == pytest.approx(0.9, abs=1e-12)
    assert summary["portfolio_value"] == pytest.approx(VALUE, rel=1e-12)
    assert summary["n_holdings"] == 3


def test_stress_summary_worst_windows_are_the_non_overlapping_scan(
    long_prices: pd.DataFrame,
):
    summary = stress_summary(long_prices, WEIGHTS, VALUE, 0.9)
    windows = summary["worst_windows"]
    assert len(windows) == 5
    spans = list(zip(windows["start"], windows["end"]))
    for i, (start_i, end_i) in enumerate(spans):
        for start_j, end_j in spans[i + 1 :]:
            assert end_i <= start_j or end_j <= start_i


@pytest.mark.parametrize(
    ("prices", "weights", "fragment"),
    [
        (pd.DataFrame(), WEIGHTS, "no usable price history"),
        (None, WEIGHTS, "no usable price history"),
    ],
)
def test_stress_summary_degrades_to_not_ok_with_a_reason(prices, weights, fragment):
    summary = stress_summary(prices, weights, VALUE, 0.9)
    assert summary["ok"] is False
    assert fragment in summary["reason"]
    assert summary["worst_scenario"] is None
    assert math.isnan(summary["worst_scenario_pct"])
    assert summary["worst_windows"].empty
    assert math.isnan(summary["sample_max_drawdown_pct"])


def test_stress_summary_says_so_when_no_window_overlaps_the_history(
    prices: pd.DataFrame,
):
    # The shared ``prices`` fixture starts in 2023, after every configured
    # crisis window. ok=False with a reason beats a table of blank rows.
    summary = stress_summary(prices, WEIGHTS, VALUE, 0.9)
    assert summary["ok"] is False
    assert "no scenario window overlapped" in summary["reason"]
    assert summary["n_covered"] == 0
    assert summary["mean_coverage"] == pytest.approx(0.0, abs=1e-12)
    # ...but the beta ladder does not need history, so it is still populated.
    assert len(summary["beta_shocks"]) == len(FACTOR_SHOCKS)
