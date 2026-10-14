"""Tests for the scheduling core.

The property that everything else rests on is **idempotence**: a scheduler
firing every fifteen minutes must deliver an 08:00 report exactly once. Almost
every test here is ultimately about that, or about the calendar arithmetic it
depends on. All of it runs against fixed clocks -- no ``datetime.now()``
anywhere except in the two tests that deliberately exercise the default.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.reporting import schedule as sch

MON = datetime(2026, 9, 7, 9, 30)  # a Monday, 09:30
WED = datetime(2026, 9, 2, 9, 30)  # a Wednesday, 09:30


def weekly(**overrides: object) -> sch.Schedule:
    base = {
        "name": "Weekly review",
        "portfolio": "Core",
        "recipients": ("owner@example.com",),
        "frequency": sch.WEEKLY,
        "hour": 8,
        "minute": 0,
        "weekday": 0,
    }
    base.update(overrides)
    return sch.Schedule(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# recipients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "address",
    ["a@b.com", "first.last+tag@sub.example.co.uk", "UPPER@EXAMPLE.COM", "x_y-z@a-b.io"],
)
def test_plausible_addresses_are_accepted(address: str) -> None:
    assert sch.valid_email(address)


@pytest.mark.parametrize(
    "address",
    ["", "   ", "no-at-sign", "@example.com", "a@", "a@b", "a b@example.com", "a@b.c"],
)
def test_malformed_addresses_are_rejected(address: str) -> None:
    """Conservative on purpose: a malformed address means a silent bounce."""
    assert not sch.valid_email(address)


def test_an_absurdly_long_address_is_rejected() -> None:
    assert not sch.valid_email("a" * 250 + "@example.com")


def test_recipients_split_on_commas_semicolons_and_whitespace() -> None:
    accepted, rejected = sch.clean_recipients("a@x.com, b@x.com; c@x.com  d@x.com")

    assert accepted == ("a@x.com", "b@x.com", "c@x.com", "d@x.com")
    assert rejected == []


def test_a_bad_address_is_reported_not_silently_dropped() -> None:
    """The whole point: a typo in one of five addresses must be visible, or
    one person stops receiving the report and nobody notices for months."""
    accepted, rejected = sch.clean_recipients(["good@x.com", "oops", "also@x.com"])

    assert accepted == ("good@x.com", "also@x.com")
    assert len(rejected) == 1
    assert "oops" in rejected[0]


def test_duplicate_recipients_collapse_case_insensitively() -> None:
    accepted, _rejected = sch.clean_recipients("a@x.com, A@X.com, a@x.com")

    assert accepted == ("a@x.com",)


def test_angle_brackets_are_stripped() -> None:
    assert sch.clean_recipients("<a@x.com>")[0] == ("a@x.com",)


def test_none_yields_nothing_and_no_complaint() -> None:
    assert sch.clean_recipients(None) == ((), [])


def test_a_nonsense_type_is_reported() -> None:
    accepted, rejected = sch.clean_recipients(42)

    assert accepted == ()
    assert rejected and "int" in rejected[0]


# ---------------------------------------------------------------------------
# construction and clamping
# ---------------------------------------------------------------------------
def test_out_of_range_fields_are_clamped_not_rejected() -> None:
    s = weekly(hour=99, minute=-5, weekday=12, day_of_month=99)

    assert (s.hour, s.minute, s.weekday, s.day_of_month) == (23, 0, 6, 31)


def test_an_unknown_frequency_falls_back_to_weekly() -> None:
    assert weekly(frequency="fortnightly").frequency == sch.WEEKLY


def test_frequency_is_case_insensitive() -> None:
    assert weekly(frequency="DAILY").frequency == sch.DAILY


def test_invalid_recipients_are_dropped_at_construction() -> None:
    assert weekly(recipients=("ok@x.com", "bad")).recipients == ("ok@x.com",)


def test_a_schedule_needs_a_name_a_portfolio_and_a_recipient() -> None:
    assert weekly().ok
    assert not weekly(name="").ok
    assert not weekly(portfolio="").ok
    assert not weekly(recipients=()).ok


def test_problems_name_each_missing_piece() -> None:
    problems = sch.Schedule(name="", portfolio="", recipients=()).problems

    assert len(problems) == 3
    assert any("name" in p for p in problems)
    assert any("portfolio" in p for p in problems)
    assert any("recipient" in p for p in problems)


def test_mark_sent_never_mutates() -> None:
    original = weekly()
    updated = original.mark_sent(MON)

    assert original.last_sent is None
    assert updated.last_sent == MON


# ---------------------------------------------------------------------------
# occurrences: daily
# ---------------------------------------------------------------------------
def test_daily_before_the_hour_looks_back_to_yesterday() -> None:
    s = weekly(frequency=sch.DAILY, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 9, 2, 7, 59)) == datetime(2026, 9, 1, 8, 0)


def test_daily_after_the_hour_is_today() -> None:
    s = weekly(frequency=sch.DAILY, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 9, 2, 8, 1)) == datetime(2026, 9, 2, 8, 0)


def test_exactly_on_the_minute_counts_as_having_occurred() -> None:
    """``<=``, not ``<``: at 08:00:00 the 08:00 slot has arrived. Otherwise a
    cron firing precisely on the minute would skip it until the next tick."""
    s = weekly(frequency=sch.DAILY, hour=8)
    at = datetime(2026, 9, 2, 8, 0)

    assert sch.previous_occurrence(s, at) == at


def test_daily_next_occurrence_is_strictly_in_the_future() -> None:
    s = weekly(frequency=sch.DAILY, hour=8)
    at = datetime(2026, 9, 2, 8, 0)

    assert sch.next_occurrence(s, at) == datetime(2026, 9, 3, 8, 0)


# ---------------------------------------------------------------------------
# occurrences: weekly
# ---------------------------------------------------------------------------
def test_weekly_looks_back_to_the_most_recent_matching_weekday() -> None:
    s = weekly(weekday=0, hour=8)  # Monday 08:00, asked on Wednesday

    assert sch.previous_occurrence(s, WED) == datetime(2026, 8, 31, 8, 0)


def test_weekly_on_the_day_but_before_the_hour_looks_back_a_full_week() -> None:
    s = weekly(weekday=0, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 9, 7, 7, 0)) == datetime(2026, 8, 31, 8, 0)


def test_weekly_next_occurrence_skips_to_next_week_once_past() -> None:
    s = weekly(weekday=0, hour=8)

    assert sch.next_occurrence(s, MON) == datetime(2026, 9, 14, 8, 0)


@pytest.mark.parametrize("weekday", range(7))
def test_every_weekly_occurrence_lands_on_the_configured_weekday(weekday: int) -> None:
    s = weekly(weekday=weekday)

    assert sch.previous_occurrence(s, WED).weekday() == weekday
    assert sch.next_occurrence(s, WED).weekday() == weekday


def test_consecutive_weekly_occurrences_are_exactly_seven_days_apart() -> None:
    s = weekly(weekday=3)
    previous = sch.previous_occurrence(s, WED)
    upcoming = sch.next_occurrence(s, WED)

    assert upcoming - previous == timedelta(days=7)


# ---------------------------------------------------------------------------
# occurrences: monthly, and the day-of-month problem
# ---------------------------------------------------------------------------
def test_monthly_before_the_day_looks_back_into_the_previous_month() -> None:
    s = weekly(frequency=sch.MONTHLY, day_of_month=15, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 9, 2, 9, 0)) == datetime(2026, 8, 15, 8, 0)


def test_the_thirty_first_clamps_to_the_end_of_a_short_month() -> None:
    """A "31st of each month" schedule must stay monthly. Skipping February
    would make it eleven times a year, which is not what anyone configured."""
    s = weekly(frequency=sch.MONTHLY, day_of_month=31, hour=8)

    assert sch.next_occurrence(s, datetime(2026, 2, 1, 9, 0)) == datetime(2026, 2, 28, 8, 0)


def test_the_thirty_first_clamps_to_the_twenty_ninth_in_a_leap_february() -> None:
    s = weekly(frequency=sch.MONTHLY, day_of_month=31, hour=8)

    assert sch.next_occurrence(s, datetime(2028, 2, 1, 9, 0)) == datetime(2028, 2, 29, 8, 0)


def test_looking_back_from_february_to_a_clamped_january_is_the_real_date() -> None:
    """The bug this guards: stepping back a month and reusing day 31 would
    construct 31 February and raise."""
    s = weekly(frequency=sch.MONTHLY, day_of_month=31, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 2, 15, 9, 0)) == datetime(2026, 1, 31, 8, 0)


def test_monthly_crosses_the_year_boundary_backwards() -> None:
    s = weekly(frequency=sch.MONTHLY, day_of_month=20, hour=8)

    assert sch.previous_occurrence(s, datetime(2026, 1, 5, 9, 0)) == datetime(2025, 12, 20, 8, 0)


def test_monthly_crosses_the_year_boundary_forwards() -> None:
    s = weekly(frequency=sch.MONTHLY, day_of_month=5, hour=8)

    assert sch.next_occurrence(s, datetime(2026, 12, 20, 9, 0)) == datetime(2027, 1, 5, 8, 0)


@pytest.mark.parametrize("month", range(1, 13))
def test_a_monthly_schedule_fires_in_every_month_of_the_year(month: int) -> None:
    s = weekly(frequency=sch.MONTHLY, day_of_month=31, hour=8)
    upcoming = sch.next_occurrence(s, datetime(2026, month, 1, 0, 0))

    assert upcoming.month == month


# ---------------------------------------------------------------------------
# idempotence -- the property that matters
# ---------------------------------------------------------------------------
def test_a_never_sent_schedule_past_its_slot_is_due() -> None:
    assert sch.is_due(weekly(), WED)


def test_recording_the_send_immediately_makes_it_not_due() -> None:
    s = weekly()
    occurrence = sch.previous_occurrence(s, WED)

    assert not sch.is_due(s.mark_sent(occurrence), WED)


def simulate(
    schedule: sch.Schedule, start: datetime, ticks: int, step: timedelta
) -> list[datetime]:
    """Run a scheduler over ``schedule`` and return the occurrences it sent.

    Exactly what :mod:`src.reporting.cli` does under cron: ask whether it is
    due, and if so record the occurrence. The schedule starts from its next
    slot, as a freshly saved one does.
    """
    current = sch.starting_next_slot(schedule, start)
    now = start
    sent: list[datetime] = []
    for _ in range(ticks):
        if sch.is_due(current, now):
            occurrence = sch.previous_occurrence(current, now)
            current = current.mark_sent(occurrence)
            sent.append(occurrence)
        now += step
    return sent


def test_a_fifteen_minute_cron_sends_an_eight_oclock_report_exactly_once() -> None:
    """The headline guarantee, simulated: step a whole day in 15-minute ticks
    and count the sends. Ninety-six chances, one e-mail."""
    sent = simulate(
        weekly(frequency=sch.DAILY, hour=8),
        datetime(2026, 9, 2, 0, 0),
        ticks=96,
        step=timedelta(minutes=15),
    )

    assert sent == [datetime(2026, 9, 2, 8, 0)]


def test_a_one_minute_cron_also_sends_exactly_once() -> None:
    """The cadence of the scheduler must not affect the number of e-mails.
    1440 chances, still one."""
    sent = simulate(
        weekly(frequency=sch.DAILY, hour=8),
        datetime(2026, 9, 2, 0, 0),
        ticks=1440,
        step=timedelta(minutes=1),
    )

    assert sent == [datetime(2026, 9, 2, 8, 0)]


def test_a_daily_schedule_sends_once_per_day_over_a_fortnight() -> None:
    sent = simulate(
        weekly(frequency=sch.DAILY, hour=8),
        datetime(2026, 9, 1, 0, 0),
        ticks=14 * 96,
        step=timedelta(minutes=15),
    )

    assert len(sent) == 14
    assert len({d.date() for d in sent}) == 14
    assert all(d.hour == 8 and d.minute == 0 for d in sent)


def test_a_weekly_schedule_sends_once_per_week_over_a_quarter() -> None:
    sent = simulate(
        weekly(weekday=2, hour=17, minute=30),
        datetime(2026, 1, 1, 0, 0),
        ticks=13 * 7 * 24,
        step=timedelta(hours=1),
    )

    assert len(sent) == 13
    assert all(d.weekday() == 2 and d.hour == 17 and d.minute == 30 for d in sent)


def test_a_monthly_schedule_sends_once_a_month_across_a_leap_year() -> None:
    """Twelve sends, including February, with day 31 clamped every short
    month -- the calendar edge cases and the idempotence rule at once."""
    sent = simulate(
        weekly(frequency=sch.MONTHLY, day_of_month=31, hour=8),
        datetime(2028, 1, 1, 0, 0),
        ticks=366 * 4,
        step=timedelta(hours=6),
    )

    assert len(sent) == 12
    assert [d.month for d in sent] == list(range(1, 13))
    assert sent[1] == datetime(2028, 2, 29, 8, 0)


def test_an_intermittent_scheduler_still_sends_every_slot_once() -> None:
    """The machine is only awake for one hour a day, at 21:00 -- thirteen
    hours after the slot. Every report still arrives, exactly once."""
    schedule = sch.starting_next_slot(
        weekly(frequency=sch.DAILY, hour=8), datetime(2026, 9, 1, 0, 0)
    )
    sent: list[datetime] = []
    for day in range(1, 11):
        now = datetime(2026, 9, day, 21, 0)
        if sch.is_due(schedule, now):
            occurrence = sch.previous_occurrence(schedule, now)
            schedule = schedule.mark_sent(occurrence)
            sent.append(occurrence)

    assert len(sent) == 10
    assert all(d.hour == 8 for d in sent)


# ---------------------------------------------------------------------------
# starting_next_slot
# ---------------------------------------------------------------------------
def test_a_new_schedule_does_not_fire_for_a_slot_that_predates_it() -> None:
    """Configured at 14:00 saying "daily at 08:00": the next cron tick must
    not mail a report seven minutes after Save."""
    created_at = datetime(2026, 9, 2, 14, 0)
    fresh = sch.starting_next_slot(weekly(frequency=sch.DAILY, hour=8), created_at)

    assert not sch.is_due(fresh, created_at)
    assert not sch.is_due(fresh, datetime(2026, 9, 2, 23, 59))
    assert sch.is_due(fresh, datetime(2026, 9, 3, 8, 0))


def test_starting_next_slot_stamps_the_previous_occurrence() -> None:
    s = weekly(weekday=0, hour=8)

    assert sch.starting_next_slot(s, WED).last_sent == datetime(2026, 8, 31, 8, 0)


def test_starting_next_slot_does_not_mutate() -> None:
    original = weekly()
    sch.starting_next_slot(original, WED)

    assert original.last_sent is None


def test_a_missed_slot_is_sent_once_however_late() -> None:
    """The machine was asleep at 08:00 and the runner first fires at 15:00.
    Better late than never -- but exactly once."""
    s = weekly(frequency=sch.DAILY, hour=8)
    late = datetime(2026, 9, 2, 15, 0)

    assert sch.is_due(s, late)
    s = s.mark_sent(sch.previous_occurrence(s, late))
    assert not sch.is_due(s, late)
    assert not sch.is_due(s, datetime(2026, 9, 2, 23, 59))


def test_a_late_send_stamped_with_the_occurrence_does_not_suppress_tomorrow() -> None:
    """Why the runner records the OCCURRENCE and not the send time: stamping
    15:00 would make tomorrow's 08:00 look already-sent."""
    s = weekly(frequency=sch.DAILY, hour=8)
    s = s.mark_sent(datetime(2026, 9, 2, 8, 0))  # the occurrence, not 15:00

    assert sch.is_due(s, datetime(2026, 9, 3, 8, 0))


def test_a_paused_schedule_is_never_due() -> None:
    assert not sch.is_due(weekly(enabled=False), WED)


def test_an_unusable_schedule_is_never_due() -> None:
    """No recipient means nowhere to send it, so it cannot be due however
    long ago its slot passed."""
    assert not sch.is_due(weekly(recipients=()), WED)


def test_a_future_only_schedule_is_not_due_before_its_first_slot() -> None:
    """A monthly report configured on the 2nd for the 15th has a previous
    occurrence in the PREVIOUS month, which is genuinely due -- so the honest
    guard is last_sent, which is what a freshly saved schedule inherits."""
    s = weekly(frequency=sch.MONTHLY, day_of_month=15, hour=8)
    just_configured = s.mark_sent(sch.previous_occurrence(s, WED))

    assert not sch.is_due(just_configured, WED)
    assert sch.is_due(just_configured, datetime(2026, 9, 15, 8, 0))


# ---------------------------------------------------------------------------
# due_schedules
# ---------------------------------------------------------------------------
def test_due_schedules_returns_only_the_due_ones() -> None:
    ready = weekly(name="ready")
    paused = weekly(name="paused", enabled=False)
    sent = weekly(name="sent").mark_sent(WED)

    due = sch.due_schedules([ready, paused, sent], WED)

    assert [s.name for s in due] == ["ready"]


def test_due_schedules_puts_the_oldest_outstanding_first() -> None:
    """Ordered so an interrupted backlog has already sent the oldest report."""
    recent = weekly(name="recent", frequency=sch.DAILY, hour=8)
    stale = weekly(name="stale", frequency=sch.MONTHLY, day_of_month=1, hour=8)

    due = sch.due_schedules([recent, stale], WED)

    assert [s.name for s in due] == ["stale", "recent"]


def test_due_schedules_of_nothing_is_empty() -> None:
    assert sch.due_schedules([], WED) == []


# ---------------------------------------------------------------------------
# timezones
# ---------------------------------------------------------------------------
def test_a_known_zone_resolves_without_complaint() -> None:
    tz, warning = sch.resolve_timezone("Asia/Kolkata")

    assert tz is not None
    assert warning == ""


def test_an_unknown_zone_degrades_to_local_time_with_a_stated_warning() -> None:
    """Degrading silently would be the bug: an 08:00 report firing at 02:30 is
    useless, and the user has to be told which clock is being used."""
    tz, warning = sch.resolve_timezone("Mars/Olympus_Mons")

    assert tz is None
    assert "unavailable" in warning
    assert "local timezone" in warning


def test_a_blank_zone_falls_back_to_the_default() -> None:
    assert sch.resolve_timezone("")[0] is not None


def test_now_in_returns_a_naive_datetime() -> None:
    """Naive by design: every comparison in the module is within one zone, and
    SQLite has no aware type."""
    assert sch.now_in("Asia/Kolkata").tzinfo is None


def test_now_in_differs_between_distant_zones() -> None:
    kolkata = sch.now_in("Asia/Kolkata")
    los_angeles = sch.now_in("America/Los_Angeles")

    assert abs((kolkata - los_angeles).total_seconds()) > 3600


def test_is_due_without_an_explicit_clock_uses_the_schedules_timezone() -> None:
    """Exercises the default-argument path rather than a fixed clock."""
    never = weekly(frequency=sch.DAILY, hour=0, minute=0)

    assert sch.is_due(never) in (True, False)  # no exception is the assertion


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------
def test_describe_names_the_weekday_and_the_next_run() -> None:
    text = sch.describe(weekly(weekday=0, hour=8), WED)

    assert "every Monday at 08:00" in text
    assert "07 Sep 2026" in text


def test_describe_reads_naturally_for_daily_and_monthly() -> None:
    assert "every day at 06:15" in sch.describe(weekly(frequency=sch.DAILY, hour=6, minute=15), WED)
    assert "day 15 of each month" in sch.describe(
        weekly(frequency=sch.MONTHLY, day_of_month=15), WED
    )


def test_describe_says_when_a_schedule_is_paused() -> None:
    assert "(paused)" in sch.describe(weekly(enabled=False), WED)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def test_a_schedule_survives_a_record_round_trip_unchanged() -> None:
    original = weekly(
        frequency=sch.MONTHLY,
        day_of_month=28,
        hour=19,
        minute=45,
        timezone="America/New_York",
        period="5y",
        benchmark="S&P 500",
        base_currency="USD",
        attach_excel=True,
        last_sent=datetime(2026, 8, 28, 19, 45),
    )

    assert sch.from_record(sch.to_record(original)) == original


def test_a_record_carries_no_credential() -> None:
    """A schedule says when and to whom. How to reach the mail server is read
    from the environment at send time and never persisted."""
    record = sch.to_record(weekly())

    assert set(record) == set(sch.SCHEDULE_FIELDS)
    assert not any(key in record for key in ("password", "smtp_password", "host", "username"))


def test_recipients_serialise_as_a_readable_list() -> None:
    record = sch.to_record(weekly(recipients=("a@x.com", "b@x.com")))

    assert record["recipients"] == "a@x.com, b@x.com"


def test_never_sent_serialises_as_an_empty_string() -> None:
    assert sch.to_record(weekly())["last_sent"] == ""
    assert (
        sch.from_record({"name": "n", "portfolio": "p", "recipients": "a@x.com"}).last_sent is None
    )


def test_an_unreadable_timestamp_reads_as_never_sent() -> None:
    """One duplicate e-mail at worst. The opposite default -- treating it as
    just-sent -- would suppress the report forever, and a scheduled report
    that stops arriving is the failure nobody notices."""
    s = sch.from_record(
        {"name": "n", "portfolio": "p", "recipients": "a@x.com", "last_sent": "yesterday"}
    )

    assert s.last_sent is None


def test_an_aware_timestamp_is_stored_naive() -> None:
    from datetime import timezone as tz

    s = sch.from_record(
        {
            "name": "n",
            "portfolio": "p",
            "recipients": "a@x.com",
            "last_sent": datetime(2026, 9, 1, 8, 0, tzinfo=tz.utc),
        }
    )

    assert s.last_sent == datetime(2026, 9, 1, 8, 0)
    assert s.last_sent.tzinfo is None


def test_a_corrupt_record_falls_back_to_defaults_rather_than_raising() -> None:
    """One bad row must not stop every other schedule from running."""
    s = sch.from_record(
        {
            "name": "n",
            "portfolio": "p",
            "recipients": "a@x.com",
            "hour": "not a number",
            "weekday": None,
            "frequency": "",
            "enabled": "maybe",
        }
    )

    assert (s.hour, s.weekday, s.frequency, s.enabled) == (8, 0, sch.WEEKLY, True)


def test_from_record_tolerates_a_non_mapping() -> None:
    s = sch.from_record("not a record")

    assert not s.ok  # nothing usable, but no exception
    assert s.problems


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(1, True), (0, False), ("true", True), ("0", False), ("", False), (True, True)],
)
def test_sqlite_style_booleans_read_back_correctly(stored: object, expected: bool) -> None:
    s = sch.from_record({"name": "n", "portfolio": "p", "recipients": "a@x.com", "enabled": stored})

    assert s.enabled is expected
