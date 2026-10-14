"""Drawdown analytics — the peak → trough → recovery state machine.

``src.analytics.drawdown`` is the module that turns an equity curve into the
story an investor actually reads: *how deep*, *how long to get there*, *how
long to climb out*, and *am I still in the hole right now*. Every one of those
numbers is a date arithmetic problem layered on a small state machine, and a
state machine is exactly the kind of code that keeps working on the happy path
while quietly mis-dating the boundaries.

So this file does not sample-check. It pins the WHOLE machine against one
hand-built series whose every episode was worked out on paper:

    2023-01-02  100  ← first observation is a peak by definition
    2023-01-03  120  ← new high; peak of episode 1
    2023-01-04   90  ← trough of episode 1        90/120 - 1 = -25.0000%
    2023-01-05  130  ← new high: episode 1 recovers, peak of episode 2
    2023-01-06  100
    2023-01-09   80  ← trough of episode 2        80/130 - 1 = -38.4615%
    2023-01-10  140  ← new high: episode 2 recovers

Peak/trough/recovery DATES are asserted, not just depths — a depth that is
right on a wrong date is a chart that lies. Alongside that: the deepest-first
ordering contract, the censoring rule for an episode still underwater at the
end of the sample (``recovery_date=None``, ``recovered=False`` — never a
flattering zero), the ``min_depth`` noise filter, the [0, 100] bound on time
underwater, and the documented degradation of every function on empty,
single-row, all-NaN, zero-valued, positional-index, unsorted and duplicated
input. Nothing in this module is allowed to raise.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics.drawdown import (
    DEFAULT_MIN_DEPTH,
    DEFAULT_TOP_N,
    EPISODE_COLUMNS,
    average_recovery_days,
    current_drawdown,
    drawdown_episodes,
    episodes_dataframe,
    time_underwater_pct,
    underwater_days,
)
from src.analytics.risk import max_drawdown

# ---------------------------------------------------------------------------
# the reference series — every expected value below is derived from THIS table
# ---------------------------------------------------------------------------
# pd.bdate_range("2023-01-02", periods=7) skips the weekend of the 7th/8th:
#   Mon 02, Tue 03, Wed 04, Thu 05, Fri 06, Mon 09, Tue 10
REFERENCE_INDEX = pd.bdate_range("2023-01-02", periods=7)
REFERENCE_VALUES = [100.0, 120.0, 90.0, 130.0, 100.0, 80.0, 140.0]

D02, D03, D04, D05, D06, D09, D10 = (
    date(2023, 1, 2),
    date(2023, 1, 3),
    date(2023, 1, 4),
    date(2023, 1, 5),
    date(2023, 1, 6),
    date(2023, 1, 9),
    date(2023, 1, 10),
)


@pytest.fixture
def episode_values() -> pd.Series:
    """The seven-point curve documented in the module docstring."""
    return pd.Series(REFERENCE_VALUES, index=REFERENCE_INDEX, name="portfolio_value")


@pytest.fixture
def still_underwater() -> pd.Series:
    """100 → 120 → 90 and the sample ends: one episode, never recovered."""
    return pd.Series([100.0, 120.0, 90.0], index=pd.bdate_range("2023-01-02", periods=3))


# ---------------------------------------------------------------------------
# the episode state machine, asserted by hand
# ---------------------------------------------------------------------------
def test_the_reference_series_contains_exactly_two_episodes(episode_values):
    # Peaks in force are 100 → 120 → 130 → 140. Two of those (120 and 130) are
    # followed by a decline, so exactly two episodes open and both close.
    assert len(drawdown_episodes(episode_values)) == 2


def test_deepest_episode_runs_from_the_130_peak_to_the_80_trough(episode_values):
    deepest = drawdown_episodes(episode_values)[0]
    # peak 2023-01-05 @ 130, trough 2023-01-09 @ 80, new high 2023-01-10 @ 140
    assert deepest.peak_date == D05
    assert deepest.trough_date == D09
    assert deepest.recovery_date == D10
    # 80 / 130 - 1 = -50/130 = -0.384615384615...  →  -38.4615%
    assert deepest.depth == pytest.approx(-0.38461538461538464, rel=1e-12)
    # calendar days, read off the index: Thu 05 → Mon 09 is 4 days
    assert deepest.drawdown_days == 4
    # Mon 09 → Tue 10 is 1 day
    assert deepest.recovery_days == 1
    assert deepest.recovered is True


def test_shallower_episode_runs_from_the_120_peak_to_the_90_trough(episode_values):
    shallower = drawdown_episodes(episode_values)[1]
    # peak 2023-01-03 @ 120, trough 2023-01-04 @ 90, new high 2023-01-05 @ 130
    assert shallower.peak_date == D03
    assert shallower.trough_date == D04
    assert shallower.recovery_date == D05
    # 90 / 120 - 1 = -0.25 exactly
    assert shallower.depth == pytest.approx(-0.25, abs=1e-15)
    assert shallower.drawdown_days == 1  # Tue 03 → Wed 04
    assert shallower.recovery_days == 1  # Wed 04 → Thu 05
    assert shallower.recovered is True


def test_episodes_come_back_deepest_first(episode_values):
    episodes = drawdown_episodes(episode_values)
    # depth is a NEGATIVE decimal, so "deepest first" means ascending:
    # -0.3846 (the 130→80 leg) must precede -0.25 (the 120→90 leg).
    assert [ep.depth for ep in episodes] == sorted(ep.depth for ep in episodes)
    assert episodes[0].depth < episodes[1].depth


def test_total_days_is_the_sum_of_the_two_legs(episode_values):
    # Identity that must hold for every recovered episode.
    for ep in drawdown_episodes(episode_values):
        assert ep.total_days == ep.drawdown_days + ep.recovery_days
    # 4 + 1 = 5 for the deepest, 1 + 1 = 2 for the other.
    assert [ep.total_days for ep in drawdown_episodes(episode_values)] == [5, 2]


def test_deepest_episode_depth_equals_the_headline_max_drawdown(episode_values):
    # Identity across modules: the deepest episode IS the max drawdown, so
    # drawdown.py and risk.py can never disagree on the headline number.
    assert drawdown_episodes(episode_values)[0].depth == pytest.approx(
        max_drawdown(episode_values), rel=1e-12
    )


def test_an_episode_still_underwater_at_the_end_is_censored_not_recovered(
    still_underwater,
):
    (episode,) = drawdown_episodes(still_underwater)
    assert episode.peak_date == D03
    assert episode.trough_date == D04
    assert episode.depth == pytest.approx(-0.25, abs=1e-15)  # 90/120 - 1
    # Nothing is invented for a hole the portfolio has not climbed out of.
    assert episode.recovery_date is None
    assert episode.recovery_days is None
    assert episode.total_days is None
    assert episode.recovered is False


def test_a_tie_at_the_trough_keeps_the_earliest_date():
    # 100 → 90 → 95 → 90 → 110: the value 90 is touched twice. The documented
    # rule is that ties keep the EARLIEST trough, i.e. 2023-01-03 not 01-05.
    values = pd.Series(
        [100.0, 90.0, 95.0, 90.0, 110.0], index=pd.bdate_range("2023-01-02", periods=5)
    )
    (episode,) = drawdown_episodes(values)
    assert episode.trough_date == D03
    assert episode.depth == pytest.approx(-0.10, abs=1e-12)  # 90/100 - 1


# ---------------------------------------------------------------------------
# min_depth — the noise filter
# ---------------------------------------------------------------------------
def test_min_depth_filters_shallow_noise():
    # 100 → 120 → 90 → 130 → 129.35 → 131
    #   episode A: 120 → 90     =  -25.0%   (real)
    #   episode B: 130 → 129.35 =   -0.5%   (a single quiet session)
    values = pd.Series(
        [100.0, 120.0, 90.0, 130.0, 129.35, 131.0],
        index=pd.bdate_range("2023-01-02", periods=6),
    )
    # DEFAULT_MIN_DEPTH is 1%, so only the -25% leg survives.
    assert DEFAULT_MIN_DEPTH == pytest.approx(0.01)
    kept = drawdown_episodes(values)
    assert len(kept) == 1
    assert kept[0].depth == pytest.approx(-0.25, abs=1e-15)

    # Drop the floor to 0.1% and the -0.5% blip is reported too.
    both = drawdown_episodes(values, min_depth=0.001)
    assert len(both) == 2
    assert both[1].depth == pytest.approx(-0.005, rel=1e-9)  # 129.35/130 - 1


def test_min_depth_boundary_keeps_a_decline_exactly_at_the_floor():
    # 100 → 99 → 101: depth is exactly -1%, and the filter is `<= -floor`,
    # so an exactly-at-the-floor decline is KEPT rather than rounded away.
    values = pd.Series([100.0, 99.0, 101.0], index=pd.bdate_range("2023-01-02", periods=3))
    (episode,) = drawdown_episodes(values, min_depth=0.01)
    assert episode.depth == pytest.approx(-0.01, rel=1e-9)


def test_min_depth_sign_is_ignored():
    values = pd.Series(
        [100.0, 120.0, 90.0, 130.0, 129.35, 131.0],
        index=pd.bdate_range("2023-01-02", periods=6),
    )
    # abs(min_depth) is used, so -0.01 and +0.01 must filter identically.
    assert drawdown_episodes(values, min_depth=-0.01) == drawdown_episodes(values, min_depth=0.01)


# ---------------------------------------------------------------------------
# top_n — a display limit, not a statistical one
# ---------------------------------------------------------------------------
@pytest.fixture
def many_episodes() -> pd.Series:
    """Twelve saw-teeth whose depths deepen monotonically: -2%, -4%, ... -24%."""
    rows: list[float] = []
    peak = 100.0
    for k in range(12):
        rows += [peak, peak * (1.0 - 0.02 * (k + 1)), peak * 1.05]
        peak *= 1.05
    return pd.Series(rows, index=pd.bdate_range("2023-01-02", periods=len(rows)))


def test_top_n_returns_the_deepest_episodes_only(many_episodes):
    assert DEFAULT_TOP_N == 5
    top = drawdown_episodes(many_episodes)
    assert len(top) == 5
    # Deepest five of -2% .. -24% are -24, -22, -20, -18, -16.
    assert [round(ep.depth, 4) for ep in top] == [-0.24, -0.22, -0.20, -0.18, -0.16]


@pytest.mark.parametrize("limit", [0, -3, None])
def test_a_non_positive_top_n_returns_every_episode(many_episodes, limit):
    assert len(drawdown_episodes(many_episodes, top_n=limit)) == 12


# ---------------------------------------------------------------------------
# underwater profile
# ---------------------------------------------------------------------------
def test_underwater_days_counts_calendar_days_since_the_running_peak(episode_values):
    # peak in force per row: 02, 03, 03, 05, 05, 05, 10
    #   02→02 = 0 | 03→03 = 0 | 03→04 = 1 | 05→05 = 0
    #   05→06 = 1 | 05→09 = 4 (the weekend counts) | 10→10 = 0
    assert underwater_days(episode_values).tolist() == [0, 0, 1, 0, 1, 4, 0]


def test_underwater_days_is_aligned_to_the_input_index(episode_values):
    out = underwater_days(episode_values)
    assert out.index.equals(episode_values.index)
    assert out.name == "underwater_days"


def test_underwater_days_degrades_to_observations_over_a_positional_index():
    # Same values, integer index: 2023-01-06 → 2023-01-09 is 4 CALENDAR days
    # but only 2 OBSERVATIONS, and the positional answer must say 2.
    positional = pd.Series(REFERENCE_VALUES)
    assert underwater_days(positional).tolist() == [0, 0, 1, 0, 1, 2, 0]


def test_underwater_days_is_zero_everywhere_for_a_monotonic_series():
    rising = pd.Series(np.arange(100.0, 130.0), index=pd.bdate_range("2023-01-02", periods=30))
    assert underwater_days(rising).abs().sum() == 0


def test_time_underwater_pct_counts_observations_below_the_running_peak(episode_values):
    # Drawdown is strictly negative on 3 of the 7 rows (01-04, 01-06, 01-09);
    # the other four sit at a running high.  3 / 7 * 100 = 42.857142857%
    assert time_underwater_pct(episode_values) == pytest.approx(300.0 / 7.0, rel=1e-12)


def test_time_underwater_pct_is_zero_for_a_monotonic_series():
    rising = pd.Series(np.arange(100.0, 130.0), index=pd.bdate_range("2023-01-02", periods=30))
    assert time_underwater_pct(rising) == pytest.approx(0.0, abs=1e-15)


def test_time_underwater_pct_stays_inside_zero_to_one_hundred(portfolio_values):
    share = time_underwater_pct(portfolio_values)
    assert 0.0 <= share <= 100.0
    # The first observation is a peak by construction, so a full 100 is
    # unreachable; a real 500-day random walk must also be strictly above 0.
    assert 0.0 < share < 100.0


def test_time_underwater_pct_never_exceeds_one_hundred_for_a_falling_series():
    # Worst possible case: every observation after the first is below the peak.
    falling = pd.Series(
        np.linspace(100.0, 50.0, 40), index=pd.bdate_range("2023-01-02", periods=40)
    )
    # 39 of 40 rows underwater → 97.5%, and never 100.
    assert time_underwater_pct(falling) == pytest.approx(97.5, rel=1e-12)


# ---------------------------------------------------------------------------
# current_drawdown — where the portfolio stands right now
# ---------------------------------------------------------------------------
def test_current_drawdown_reports_not_underwater_at_a_new_high(episode_values):
    # The series ends at 140, its all-time high, so there is no hole at all.
    now = current_drawdown(episode_values)
    assert now["underwater"] is False
    assert now["depth"] == pytest.approx(0.0, abs=1e-15)
    assert now["peak_date"] == D10
    assert now["peak_value"] == pytest.approx(140.0)
    assert now["days_underwater"] == 0


def test_current_drawdown_reports_the_live_hole_when_below_the_peak(still_underwater):
    # 100 → 120 → 90: the high in force is 120 on 2023-01-03.
    now = current_drawdown(still_underwater)
    assert now["underwater"] is True
    assert now["depth"] == pytest.approx(-0.25, abs=1e-15)  # 90/120 - 1
    assert now["peak_date"] == D03
    assert now["peak_value"] == pytest.approx(120.0)
    assert now["days_underwater"] == 1  # Tue 03 → Wed 04


def test_current_drawdown_has_no_peak_date_over_a_positional_index():
    now = current_drawdown(pd.Series(REFERENCE_VALUES))
    assert now["peak_date"] is None
    # ...and the day count degrades to observations since the peak (0 at a high).
    assert now["days_underwater"] == 0


# ---------------------------------------------------------------------------
# episodes_dataframe — the UI/Excel table
# ---------------------------------------------------------------------------
def test_episodes_dataframe_carries_exactly_the_documented_columns(episode_values):
    frame = episodes_dataframe(drawdown_episodes(episode_values))
    assert tuple(frame.columns) == EPISODE_COLUMNS
    assert len(frame) == 2


def test_episodes_dataframe_preserves_the_order_it_was_given(episode_values):
    episodes = drawdown_episodes(episode_values)
    frame = episodes_dataframe(episodes)
    assert frame["trough_date"].tolist() == [D09, D04]
    assert frame["recovered"].tolist() == [True, True]


def test_depth_pct_is_depth_expressed_in_percent(episode_values):
    frame = episodes_dataframe(drawdown_episodes(episode_values))
    # -0.384615... × 100 = -38.4615...%, and -0.25 × 100 = -25%.
    assert frame["depth_pct"].iloc[0] == pytest.approx(-38.46153846153846, rel=1e-12)
    assert frame["depth_pct"].iloc[1] == pytest.approx(-25.0, abs=1e-12)
    assert (frame["depth_pct"] == frame["depth"] * 100.0).all()


def test_episodes_dataframe_marks_an_unrecovered_episode_na_not_zero(still_underwater):
    frame = episodes_dataframe(drawdown_episodes(still_underwater))
    # pd.NA, NOT 0 — a zero here would read as "recovered instantly".
    assert frame["recovery_days"].iloc[0] is pd.NA
    assert frame["total_days"].iloc[0] is pd.NA
    assert frame["recovery_date"].iloc[0] is None
    assert bool(frame["recovered"].iloc[0]) is False


@pytest.mark.parametrize("supplied", [None, []])
def test_episodes_dataframe_of_nothing_is_empty_but_keeps_its_columns(supplied):
    frame = episodes_dataframe(supplied)
    assert frame.empty
    # Callers index columns unconditionally, so the schema must survive.
    assert tuple(frame.columns) == EPISODE_COLUMNS


# ---------------------------------------------------------------------------
# average_recovery_days
# ---------------------------------------------------------------------------
def test_average_recovery_days_is_the_mean_over_recovered_episodes():
    # 2023-01-02 100 | 01-03 90 | 01-04 100 | 01-05 95 | 01-06 96 | 01-09 100
    #   episode A: peak 01-02, trough 01-03, recovery 01-04 → 1 calendar day
    #   episode B: peak 01-04, trough 01-05, recovery 01-09 → 4 calendar days
    #   mean = (1 + 4) / 2 = 2.5
    values = pd.Series(
        [100.0, 90.0, 100.0, 95.0, 96.0, 100.0],
        index=pd.bdate_range("2023-01-02", periods=6),
    )
    episodes = drawdown_episodes(values)
    assert sorted(ep.recovery_days for ep in episodes) == [1, 4]
    assert average_recovery_days(episodes) == pytest.approx(2.5)


def test_average_recovery_days_ignores_an_episode_still_underwater():
    # 100 → 90 → 100 → 80: one recovered episode (1 day) and one still open.
    # Counting the open one as zero would flatter the average to 0.5.
    values = pd.Series([100.0, 90.0, 100.0, 80.0], index=pd.bdate_range("2023-01-02", periods=4))
    episodes = drawdown_episodes(values, top_n=0)
    assert [ep.recovered for ep in episodes] == [False, True]
    assert average_recovery_days(episodes) == pytest.approx(1.0)


@pytest.mark.parametrize("supplied", [None, []])
def test_average_recovery_days_is_nan_with_nothing_to_average(supplied):
    assert np.isnan(average_recovery_days(supplied))


def test_average_recovery_days_is_nan_when_no_episode_has_recovered(still_underwater):
    assert np.isnan(average_recovery_days(drawdown_episodes(still_underwater)))


# ---------------------------------------------------------------------------
# edge cases — documented degradation, never an exception
# ---------------------------------------------------------------------------
def test_empty_input_degrades_everywhere(empty_series):
    assert drawdown_episodes(empty_series) == []
    assert underwater_days(empty_series).empty
    assert np.isnan(time_underwater_pct(empty_series))
    now = current_drawdown(empty_series)
    assert np.isnan(now["depth"])
    assert np.isnan(now["peak_value"])
    assert now["peak_date"] is None
    assert now["days_underwater"] is None
    assert now["underwater"] is False


def test_a_single_observation_has_no_episode_and_no_hole(single_value_series):
    # Two points are the minimum for a decline to exist.
    assert drawdown_episodes(single_value_series) == []
    assert time_underwater_pct(single_value_series) == pytest.approx(0.0, abs=1e-15)
    now = current_drawdown(single_value_series)
    assert now["depth"] == pytest.approx(0.0, abs=1e-15)
    assert now["underwater"] is False
    assert underwater_days(single_value_series).tolist() == [0]


def test_an_all_nan_series_degrades_like_an_empty_one(nan_series):
    # Every value is dropped by the cleaner, so nothing is left to analyse.
    assert drawdown_episodes(nan_series) == []
    assert underwater_days(nan_series).empty
    assert np.isnan(time_underwater_pct(nan_series))
    assert current_drawdown(nan_series)["underwater"] is False


def test_a_flat_zero_variance_series_never_goes_underwater(flat_series):
    # 60 identical values: `>=` on the running max means every point is a peak.
    assert drawdown_episodes(flat_series) == []
    assert time_underwater_pct(flat_series) == pytest.approx(0.0, abs=1e-15)
    assert current_drawdown(flat_series)["depth"] == pytest.approx(0.0, abs=1e-15)
    assert underwater_days(flat_series).abs().sum() == 0


def test_a_series_pinned_at_zero_gives_nan_rather_than_a_fabricated_ratio():
    # A non-positive running peak makes V/peak - 1 undefined; the documented
    # answer is NaN, not ±inf and not 0.
    zeros = pd.Series([0.0, 0.0, 0.0], index=pd.bdate_range("2023-01-02", periods=3))
    assert np.isnan(time_underwater_pct(zeros))
    now = current_drawdown(zeros)
    assert np.isnan(now["depth"])
    assert now["underwater"] is False


def test_a_positional_index_returns_no_episodes_rather_than_epoch_dates():
    # Calendar-day counts off an integer index would be fiction, so the
    # module refuses to produce them.
    assert drawdown_episodes(pd.Series(REFERENCE_VALUES)) == []


def test_an_unsorted_date_index_is_sorted_before_anything_is_computed(episode_values):
    shuffled = episode_values.iloc[[3, 0, 6, 2, 5, 1, 4]]
    assert not shuffled.index.is_monotonic_increasing
    assert drawdown_episodes(shuffled) == drawdown_episodes(episode_values)
    assert time_underwater_pct(shuffled) == pytest.approx(
        time_underwater_pct(episode_values), rel=1e-12
    )


def test_duplicate_index_entries_do_not_raise():
    # Two prints stamped on the same day: the machine still runs positionally
    # and simply reports a zero-calendar-day leg.
    values = pd.Series(
        [100.0, 120.0, 90.0, 130.0],
        index=pd.DatetimeIndex(["2023-01-02", "2023-01-03", "2023-01-03", "2023-01-04"]),
    )
    (episode,) = drawdown_episodes(values)
    assert episode.depth == pytest.approx(-0.25, abs=1e-15)  # 90/120 - 1
    assert episode.drawdown_days == 0
    assert len(underwater_days(values)) == 4
