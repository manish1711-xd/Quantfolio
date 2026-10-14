"""Tests for the price-alert state machine.

Two properties carry the feature, and most of this file is about them:

* **Fire once per crossing.** A runner polling every five minutes must send
  one e-mail when a level is reached, not one per poll. Asserted by simulating
  a poll loop, not by inspecting the implementation.
* **Never silently fail to evaluate.** A rule with no quote or no direction
  produces silence that looks exactly like "your level was not reached", so
  every un-evaluated rule must come back in ``skipped`` with a reason.

Everything runs at a fixed clock against hand-built quotes: no network, no
market hours, no ``datetime.now()`` except where the default is the thing under
test.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src import alerts as al
from src import watchlist as wl
from src.models import Quote

NOW = datetime(2026, 9, 3, 10, 30)


def quote(
    ticker: str = "AAA.NS",
    price: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    currency: str = "INR",
    as_of: datetime | None = None,
    intraday: bool = True,
) -> Quote:
    stamp = as_of or datetime(2026, 9, 3, 10, 15)
    return Quote(
        ticker,
        price,
        currency,
        stamp,
        stamp,
        intraday,
        day_high=high,
        day_low=low,
    )


def rule(**overrides: object) -> wl.WatchItem:
    """A watchlist item with an active, unambiguous alert rule."""
    base: dict[str, object] = {
        "ticker": "AAA.NS",
        "target_price": 110.0,
        "direction": al.ABOVE,
        "alert_enabled": True,
        "reference_price": 100.0,
    }
    base.update(overrides)
    return wl.WatchItem(**base)  # type: ignore[arg-type]


def run(
    items, quotes, settings: al.AlertSettings | None = None, now: datetime | None = None
) -> al.AlertOutcome:
    return al.evaluate(
        items,
        quotes,
        settings
        if settings is not None
        else al.AlertSettings(enabled=True, recipients=["a@x.com"]),
        now=now or NOW,
    )


# ---------------------------------------------------------------------------
# direction: the fact the watchlist never had
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["above", "ABOVE", "up", "rise", "rises", "over", ">=", ">"])
def test_words_meaning_upward_all_normalise_to_above(text: str) -> None:
    assert al.normalise_direction(text) == al.ABOVE


@pytest.mark.parametrize("text", ["below", "BELOW", "down", "fall", "falls", "under", "<=", "<"])
def test_words_meaning_downward_all_normalise_to_below(text: str) -> None:
    assert al.normalise_direction(text) == al.BELOW


@pytest.mark.parametrize("text", ["", "   ", "sideways", "maybe", None, 42])
def test_an_unrecognised_direction_is_blank_not_guessed(text: object) -> None:
    """A direction that fails to parse disarms the rule, and the rule is then
    reported as inactive. Guessing would attach behaviour the user never chose."""
    assert al.normalise_direction(text) == ""


def test_a_target_above_the_reference_price_infers_above() -> None:
    assert al.infer_direction(110.0, 100.0) == al.ABOVE


def test_a_target_below_the_reference_price_infers_below() -> None:
    assert al.infer_direction(90.0, 100.0) == al.BELOW


def test_a_target_exactly_on_the_reference_price_infers_nothing() -> None:
    """Genuinely ambiguous, so it stays blank and is reported rather than
    silently becoming a rule that never fires."""
    assert al.infer_direction(100.0, 100.0) == ""


@pytest.mark.parametrize(
    ("level", "reference"),
    [(None, 100.0), (110.0, None), (0.0, 100.0), (110.0, 0.0), ("x", 100.0)],
)
def test_direction_cannot_be_inferred_without_two_usable_prices(
    level: object, reference: object
) -> None:
    assert al.infer_direction(level, reference) == ""


# ---------------------------------------------------------------------------
# crossing
# ---------------------------------------------------------------------------
def test_a_price_above_an_upward_level_has_crossed() -> None:
    hit, evidence = al.crossing(al.ABOVE, 110.0, 111.0)

    assert hit
    assert evidence == al.EVIDENCE_LAST


def test_a_price_below_an_upward_level_has_not() -> None:
    assert al.crossing(al.ABOVE, 110.0, 109.0) == (False, "")


def test_a_price_below_a_downward_level_has_crossed() -> None:
    hit, evidence = al.crossing(al.BELOW, 90.0, 89.0)

    assert hit
    assert evidence == al.EVIDENCE_LAST


def test_a_price_exactly_on_the_level_counts_as_reached() -> None:
    """A user who typed a round number is describing the most likely print of
    the day; ``>`` rather than ``>=`` would skip exactly that case."""
    assert al.crossing(al.ABOVE, 110.0, 110.0)[0]
    assert al.crossing(al.BELOW, 90.0, 90.0)[0]


def test_the_session_high_catches_a_crossing_the_last_price_missed() -> None:
    """The whole reason the session range is consulted: between two checks the
    price may have gone through the level and come back."""
    hit, evidence = al.crossing(al.ABOVE, 110.0, 105.0, day_high=112.0)

    assert hit
    assert evidence == al.EVIDENCE_HIGH


def test_the_session_low_catches_a_downward_crossing() -> None:
    hit, evidence = al.crossing(al.BELOW, 90.0, 95.0, day_low=88.0)

    assert hit
    assert evidence == al.EVIDENCE_LOW


def test_the_last_price_is_preferred_over_the_session_range() -> None:
    """ "It is there now" is the stronger statement, so it is reported when
    both are true."""
    _hit, evidence = al.crossing(al.ABOVE, 110.0, 115.0, day_high=120.0)

    assert evidence == al.EVIDENCE_LAST


def test_the_session_range_can_be_switched_off() -> None:
    assert al.crossing(al.ABOVE, 110.0, 105.0, day_high=112.0, use_session_range=False) == (
        False,
        "",
    )


def test_the_wrong_side_of_the_range_is_not_evidence() -> None:
    """An upward rule must not be triggered by the session LOW."""
    assert al.crossing(al.ABOVE, 110.0, 105.0, day_low=90.0) == (False, "")
    assert al.crossing(al.BELOW, 90.0, 95.0, day_high=120.0) == (False, "")


@pytest.mark.parametrize(
    ("direction", "level", "price"),
    [("", 110.0, 120.0), ("sideways", 110.0, 120.0), (al.ABOVE, None, 120.0), (al.ABOVE, 0.0, 1.0)],
)
def test_an_unusable_rule_never_reports_a_crossing(
    direction: object, level: object, price: object
) -> None:
    assert al.crossing(direction, level, price) == (False, "")


def test_a_missing_price_with_no_range_is_not_a_crossing() -> None:
    assert al.crossing(al.ABOVE, 110.0, None) == (False, "")


# ---------------------------------------------------------------------------
# re-arming
# ---------------------------------------------------------------------------
def test_a_price_just_below_the_level_does_not_rearm() -> None:
    """The anti-flap property: without a band, a price wobbling on a round
    number re-arms and re-fires until the user stops reading the alerts."""
    assert not al.rearmed(al.ABOVE, 110.0, 109.9, rearm_pct=1.0)


def test_a_price_past_the_band_rearms() -> None:
    assert al.rearmed(al.ABOVE, 110.0, 108.0, rearm_pct=1.0)


def test_a_downward_rule_rearms_when_the_price_recovers_past_the_band() -> None:
    assert al.rearmed(al.BELOW, 90.0, 92.0, rearm_pct=1.0)
    assert not al.rearmed(al.BELOW, 90.0, 90.5, rearm_pct=1.0)


def test_a_zero_band_rearms_on_any_retreat() -> None:
    assert al.rearmed(al.ABOVE, 110.0, 109.99, rearm_pct=0.0)
    assert not al.rearmed(al.ABOVE, 110.0, 110.0, rearm_pct=0.0)


def test_the_band_scales_with_the_level() -> None:
    """A percentage, not an absolute: 1% of 30,000 is 300, and a fixed rupee
    band would be either useless on an index or absurd on a penny stock."""
    assert al.rearmed(al.ABOVE, 30000.0, 29600.0, rearm_pct=1.0)
    assert not al.rearmed(al.ABOVE, 30000.0, 29900.0, rearm_pct=1.0)


def test_rearming_never_uses_the_session_range() -> None:
    """Re-arming on a session low would undo a trigger the same session
    produced, so only the last price counts. Asserted through evaluate,
    because that is where the two could get crossed."""
    triggered = rule(alert_state=al.TRIGGERED)
    out = run([triggered], {"AAA.NS": quote(price=112.0, low=100.0)})

    assert out.rearmed == ()
    assert out.items[0].alert_state == al.TRIGGERED


# ---------------------------------------------------------------------------
# arming a new rule
# ---------------------------------------------------------------------------
def test_arming_records_the_reference_price() -> None:
    """The whole point: without it, a direction can never be inferred later."""
    armed = al.arm(wl.WatchItem("AAA.NS", target_price=110.0), 100.0)

    assert armed.reference_price == 100.0
    assert armed.direction == al.ABOVE


def test_arming_a_rule_already_through_its_level_starts_it_triggered() -> None:
    """Mirrors the tearsheet fix: a rule created now describes the future. If
    the price is already past the level, announcing it would report a crossing
    that happened before the alert existed."""
    armed = al.arm(wl.WatchItem("AAA.NS", target_price=110.0, alert_enabled=True), 120.0)

    assert armed.direction == al.BELOW  # 110 is below 120, so it is an entry level
    assert armed.alert_state == al.ARMED  # ...and 120 has not fallen to 110

    explicit = al.arm(
        wl.WatchItem("AAA.NS", target_price=110.0, alert_enabled=True), 120.0, al.ABOVE
    )
    assert explicit.alert_state == al.TRIGGERED


def test_arming_does_not_fire_on_the_first_evaluation() -> None:
    """The behavioural consequence of the test above, end to end."""
    armed = al.arm(wl.WatchItem("AAA.NS", target_price=110.0, alert_enabled=True), 120.0, al.ABOVE)

    out = run([armed], {"AAA.NS": quote(price=121.0)})

    assert out.events == ()


def test_an_explicit_direction_overrides_the_inference() -> None:
    armed = al.arm(wl.WatchItem("AAA.NS", target_price=110.0), 100.0, al.BELOW)

    assert armed.direction == al.BELOW


def test_arming_clears_any_previous_fire() -> None:
    stale = rule(alert_state=al.TRIGGERED, last_fired=NOW)

    assert al.arm(stale, 100.0).last_fired is None


# ---------------------------------------------------------------------------
# evaluate: firing
# ---------------------------------------------------------------------------
def test_a_crossed_rule_produces_one_event() -> None:
    out = run([rule()], {"AAA.NS": quote(price=111.0)})

    assert len(out.events) == 1
    event = out.events[0]
    assert event.ticker == "AAA.NS"
    assert event.direction == al.ABOVE
    assert event.level == 110.0
    assert event.price == 111.0
    assert event.currency == "INR"


def test_firing_advances_the_state_and_stamps_the_time() -> None:
    out = run([rule()], {"AAA.NS": quote(price=111.0)})

    assert out.items[0].alert_state == al.TRIGGERED
    assert out.items[0].last_fired == NOW
    assert out.changed


def test_the_second_check_sends_nothing() -> None:
    """The headline guarantee."""
    first = run([rule()], {"AAA.NS": quote(price=111.0)})
    second = run(first.items, {"AAA.NS": quote(price=111.0)}, now=NOW + timedelta(minutes=5))

    assert len(first.events) == 1
    assert second.events == ()


def test_an_uncrossed_rule_is_left_exactly_as_it_was() -> None:
    original = rule()
    out = run([original], {"AAA.NS": quote(price=105.0)})

    assert out.events == ()
    assert out.items[0] == original
    assert not out.changed


def test_through_by_pct_is_positive_for_both_directions() -> None:
    """Signed in the direction of the rule, so "past your level" always reads
    as a positive number whichever way the price moved."""
    up = run([rule()], {"AAA.NS": quote(price=110.0 * 1.02)})
    down = run([rule(target_price=90.0, direction=al.BELOW)], {"AAA.NS": quote(price=90.0 * 0.98)})

    assert up.events[0].through_by_pct == pytest.approx(2.0)
    assert down.events[0].through_by_pct == pytest.approx(2.0)


def test_an_event_carries_the_quote_timestamp() -> None:
    """So the e-mail can say the level *had been* reached as of a time, rather
    than implying a live cross against a ~15-minute-delayed feed."""
    stamp = datetime(2026, 9, 3, 10, 15)
    out = run([rule()], {"AAA.NS": quote(price=111.0, as_of=stamp)})

    assert out.events[0].as_of == stamp
    assert out.events[0].is_intraday


def test_an_aware_quote_timestamp_is_stored_naive() -> None:
    aware = datetime(2026, 9, 3, 4, 45, tzinfo=timezone.utc)
    out = run([rule()], {"AAA.NS": quote(price=111.0, as_of=aware)})

    assert out.events[0].as_of is not None
    assert out.events[0].as_of.tzinfo is None


def test_an_event_carries_the_users_own_note() -> None:
    out = run([rule(note="breakout watch")], {"AAA.NS": quote(price=111.0)})

    assert out.events[0].note == "breakout watch"


def test_a_session_high_trigger_is_labelled_as_such() -> None:
    """A user acting on this needs to know the price may have come back."""
    out = run([rule()], {"AAA.NS": quote(price=105.0, high=112.0)})

    assert out.events[0].evidence == al.EVIDENCE_HIGH
    assert out.events[0].price == 112.0  # the observation that crossed, not the last


def test_an_event_headline_names_the_evidence() -> None:
    out = run([rule()], {"AAA.NS": quote(price=105.0, high=112.0)})

    assert "session high" in out.events[0].headline
    assert "rises to or above" in out.events[0].headline


def test_a_blank_direction_is_inferred_from_the_reference_price() -> None:
    """A rule saved before directions existed still works, as long as a
    reference price was captured."""
    out = run([rule(direction="")], {"AAA.NS": quote(price=111.0)})

    assert len(out.events) == 1
    assert out.items[0].direction == al.ABOVE  # and is persisted, so it stops inferring


# ---------------------------------------------------------------------------
# evaluate: re-arming and cooldown
# ---------------------------------------------------------------------------
def test_a_retreat_past_the_band_rearms_without_sending() -> None:
    fired = run([rule()], {"AAA.NS": quote(price=111.0)})
    back = run(fired.items, {"AAA.NS": quote(price=105.0)}, now=NOW + timedelta(hours=1))

    assert back.events == ()
    assert back.rearmed == ("AAA.NS",)
    assert back.items[0].alert_state == al.ARMED
    assert back.changed


def test_a_rearmed_rule_can_fire_again() -> None:
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=0.0)
    fired = run([rule()], {"AAA.NS": quote(price=111.0)}, settings)
    back = run(fired.items, {"AAA.NS": quote(price=105.0)}, settings, NOW + timedelta(hours=1))
    again = run(back.items, {"AAA.NS": quote(price=112.0)}, settings, NOW + timedelta(hours=2))

    assert len(again.events) == 1


def test_the_cooldown_suppresses_a_second_alert_and_says_so() -> None:
    """Suppressed, not lost: the notice travels so the user knows the level was
    reached again rather than being told nothing."""
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=12.0)
    fired = run([rule()], {"AAA.NS": quote(price=111.0)}, settings)
    back = run(fired.items, {"AAA.NS": quote(price=105.0)}, settings, NOW + timedelta(hours=1))
    again = run(back.items, {"AAA.NS": quote(price=112.0)}, settings, NOW + timedelta(hours=2))

    assert again.events == ()
    assert any("within 12h" in notice for notice in again.notices)


def test_a_cooldown_suppressed_crossing_still_advances_the_state() -> None:
    """Otherwise it would stay armed and fire the moment the cooldown lapsed,
    hours after the crossing stopped being news."""
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=12.0)
    fired = run([rule()], {"AAA.NS": quote(price=111.0)}, settings)
    back = run(fired.items, {"AAA.NS": quote(price=105.0)}, settings, NOW + timedelta(hours=1))
    again = run(back.items, {"AAA.NS": quote(price=112.0)}, settings, NOW + timedelta(hours=2))

    assert again.items[0].alert_state == al.TRIGGERED


def test_the_cooldown_lapses() -> None:
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=12.0)
    fired = run([rule()], {"AAA.NS": quote(price=111.0)}, settings)
    back = run(fired.items, {"AAA.NS": quote(price=105.0)}, settings, NOW + timedelta(hours=13))
    again = run(back.items, {"AAA.NS": quote(price=112.0)}, settings, NOW + timedelta(hours=14))

    assert len(again.events) == 1


# ---------------------------------------------------------------------------
# evaluate: what it refuses to do silently
# ---------------------------------------------------------------------------
def test_a_rule_with_no_quote_is_reported_not_treated_as_quiet() -> None:
    """The failure an alert system must never have: silence that looks exactly
    like "your level was not reached"."""
    out = run([rule()], {})

    assert out.events == ()
    assert len(out.skipped) == 1
    ticker, reason = out.skipped[0]
    assert ticker == "AAA.NS"
    assert "no quote" in reason


def test_a_quote_with_no_usable_price_is_reported() -> None:
    out = run([rule()], {"AAA.NS": quote(price=float("nan"))})

    assert out.skipped and "no usable price" in out.skipped[0][1]


def test_a_rule_with_no_direction_and_no_reference_is_reported() -> None:
    out = run([rule(direction="", reference_price=None)], {"AAA.NS": quote(price=111.0)})

    assert out.events == ()
    assert out.skipped
    assert "no direction" in out.skipped[0][1]


def test_a_rule_that_cannot_be_evaluated_keeps_its_state() -> None:
    original = rule()
    out = run([original], {})

    assert out.items[0] == original


def test_alerting_switched_off_is_passed_through_untouched() -> None:
    off = rule(alert_enabled=False)
    out = run([off], {"AAA.NS": quote(price=111.0)})

    assert out.events == ()
    assert out.skipped == ()
    assert out.items[0] == off


def test_an_item_with_no_target_is_passed_through_untouched() -> None:
    plain = wl.WatchItem("AAA.NS", note="just watching", alert_enabled=True)
    out = run([plain], {"AAA.NS": quote(price=111.0)})

    assert out.events == ()
    assert out.skipped == ()
    assert out.items[0] == plain


def test_every_item_comes_back_exactly_once() -> None:
    """The caller persists ``items``; dropping one would delete a watchlist
    entry, and duplicating one would resurrect a deleted rule."""
    items = [
        rule(ticker="AAA.NS"),
        rule(ticker="BBB.NS", target_price=50.0, direction=al.BELOW),
        wl.WatchItem("CCC.NS"),
    ]
    out = run(items, {"AAA.NS": quote(price=111.0)})

    assert [i.ticker for i in out.items] == ["AAA.NS", "BBB.NS", "CCC.NS"]


def test_quotes_are_matched_case_insensitively() -> None:
    out = run([rule()], {"aaa.ns": quote(price=111.0)})

    assert len(out.events) == 1


def test_evaluating_nothing_is_an_empty_outcome() -> None:
    out = run([], {})

    assert out.events == ()
    assert out.items == ()
    assert not out.changed


def test_evaluate_without_an_explicit_clock_does_not_raise() -> None:
    out = al.evaluate([rule()], {"AAA.NS": quote(price=111.0)}, al.AlertSettings())

    assert len(out.events) == 1
    assert out.items[0].last_fired is not None


# ---------------------------------------------------------------------------
# fire-once, by simulation
# ---------------------------------------------------------------------------
def simulate(item: wl.WatchItem, prices, settings=None, step=timedelta(minutes=5)):
    """Poll ``prices`` one at a time, as the cron runner does."""
    config = settings or al.AlertSettings(enabled=True, recipients=["a@x.com"])
    items = (item,)
    moment = NOW
    fired: list[al.AlertEvent] = []
    for price in prices:
        out = al.evaluate(items, {item.ticker: quote(price=price)}, config, now=moment)
        fired.extend(out.events)
        items = out.items
        moment += step
    return fired, items


def test_a_price_sitting_on_the_level_produces_one_email_not_ninety() -> None:
    """Ninety-six five-minute polls with the price parked above the level."""
    fired, _items = simulate(rule(), [111.0] * 96)

    assert len(fired) == 1


def test_a_price_drifting_far_past_the_level_still_produces_one() -> None:
    fired, _items = simulate(rule(), [110.0 + i for i in range(50)])

    assert len(fired) == 1


def test_one_genuine_round_trip_produces_two_emails() -> None:
    """Up through the level, well back down, up again — that is two pieces of
    news, and the rule should say so."""
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=0.0)
    fired, _items = simulate(rule(), [105.0, 111.0, 111.0, 105.0, 104.0, 112.0, 112.0], settings)

    assert len(fired) == 2


def test_noise_on_the_level_does_not_produce_a_burst() -> None:
    """A price oscillating inside the re-arm band around 110: one alert, not
    one per tick. This is the reason the band exists."""
    settings = al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=0.0)
    wobble = [110.2, 109.8, 110.4, 109.6, 110.1, 109.9] * 10
    fired, _items = simulate(rule(), wobble, settings)

    assert len(fired) == 1


def test_the_cooldown_bounds_a_violently_oscillating_price() -> None:
    """Why the cooldown exists as well as the band.

    A price swinging 100 -> 115 -> 100 clears the 1% re-arm band on every leg,
    so hysteresis alone lets every crossing through. Over a simulated day of
    five-minute polls that is 144 e-mails; a 12-hour cooldown makes it 2. Both
    numbers are pinned, because the contrast *is* the property under test.
    """
    swings = [115.0, 100.0] * 144  # 288 polls = 24 hours

    unbounded, _ = simulate(
        rule(), swings, al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=0.0)
    )
    bounded, _ = simulate(
        rule(), swings, al.AlertSettings(enabled=True, recipients=["a@x.com"], cooldown_hours=12.0)
    )

    assert len(unbounded) == 144
    assert len(bounded) == 2


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_alerts_are_off_by_default() -> None:
    """An e-mail nobody asked for is worse than no e-mail."""
    config = al.AlertSettings()

    assert not config.enabled
    assert config.recipients == ()
    assert not config.ok


def test_settings_are_ok_only_when_on_and_addressed() -> None:
    assert al.AlertSettings(enabled=True, recipients=["a@x.com"]).ok
    assert not al.AlertSettings(enabled=True).ok
    assert not al.AlertSettings(recipients=["a@x.com"]).ok


def test_settings_problems_name_what_is_missing() -> None:
    problems = al.AlertSettings().problems

    assert any("switched off" in p for p in problems)
    assert any("recipient" in p for p in problems)


def test_invalid_recipients_are_dropped_at_construction() -> None:
    config = al.AlertSettings(recipients="good@x.com, oops")

    assert config.recipients == ("good@x.com",)


def test_a_comma_separated_string_is_accepted() -> None:
    config = al.AlertSettings(recipients="a@x.com, b@x.com")

    assert config.recipients == ("a@x.com", "b@x.com")


@pytest.mark.parametrize(
    ("field", "given", "expected"),
    [
        ("rearm_pct", -5.0, 0.0),
        ("rearm_pct", 500.0, 50.0),
        ("rearm_pct", "nonsense", al.DEFAULT_REARM_PCT),
        ("cooldown_hours", -1.0, 0.0),
        ("cooldown_hours", 10_000.0, 720.0),
        ("cooldown_hours", None, al.DEFAULT_COOLDOWN_HOURS),
    ],
)
def test_out_of_range_settings_are_clamped_not_rejected(
    field: str, given: object, expected: float
) -> None:
    config = al.AlertSettings(**{field: given})  # type: ignore[arg-type]

    assert getattr(config, field) == pytest.approx(expected)


def test_settings_carry_no_credential() -> None:
    """Like Schedule: who to notify, never how to reach the mail server."""
    fields = set(al.AlertSettings().__dataclass_fields__)

    assert not fields & {"password", "smtp_password", "host", "username"}


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------
def test_events_frame_has_the_documented_columns() -> None:
    out = run([rule()], {"AAA.NS": quote(price=111.0)})
    frame = al.events_frame(out.events)

    assert tuple(frame.columns) == al.EVENT_COLUMNS
    assert len(frame) == 1


def test_an_empty_events_frame_is_still_typed() -> None:
    frame = al.events_frame([])

    assert tuple(frame.columns) == al.EVENT_COLUMNS
    assert frame.empty


def test_describe_rule_says_what_will_happen() -> None:
    assert "rises to or above" in al.describe_rule(rule())
    assert "Waiting" in al.describe_rule(rule())


def test_describe_rule_explains_an_already_triggered_rule() -> None:
    text = al.describe_rule(rule(alert_state=al.TRIGGERED))

    assert "Already through" in text
    assert "retreats" in text


def test_describe_rule_names_a_missing_direction_as_the_blocker() -> None:
    text = al.describe_rule(rule(direction="", reference_price=None))

    assert "no direction set" in text


def test_describe_rule_distinguishes_alerting_off_from_broken() -> None:
    text = al.describe_rule(rule(alert_enabled=False))

    assert "alerting off" in text


def test_describe_rule_handles_no_target() -> None:
    assert al.describe_rule(wl.WatchItem("AAA.NS")) == "No target set."


def test_the_state_table_flags_only_rules_that_can_fire() -> None:
    items = [
        rule(ticker="AAA.NS"),
        rule(ticker="BBB.NS", alert_enabled=False),
        wl.WatchItem("CCC.NS", alert_enabled=True),  # no target
    ]
    table = al.alert_state_table(items)

    assert list(table["alert"]) == [True, False, False]


def test_an_empty_state_table_is_still_typed() -> None:
    table = al.alert_state_table([])

    assert list(table.columns) == [
        "ticker",
        "alert",
        "direction",
        "level",
        "state",
        "last_fired",
        "rule",
    ]
    assert table.empty


# ---------------------------------------------------------------------------
# the seam back into the watchlist
# ---------------------------------------------------------------------------
def test_merging_state_back_keeps_edits_the_user_made_meanwhile() -> None:
    """A cron run must not overwrite a note or a target changed in the browser
    while it was fetching quotes — the same reasoning as the targeted UPDATE
    in mark_schedule_sent."""
    stored = rule(note="edited in the browser", target_price=115.0)
    evaluated = run([rule()], {"AAA.NS": quote(price=111.0)}).items

    merged = wl.with_alert([stored], evaluated)

    assert merged[0].note == "edited in the browser"
    assert merged[0].target_price == 115.0
    assert merged[0].alert_state == al.TRIGGERED


def test_merging_ignores_items_the_evaluation_did_not_see() -> None:
    stored = [rule(ticker="AAA.NS"), rule(ticker="BBB.NS")]
    merged = wl.with_alert(stored, [replace(stored[0], alert_state=al.TRIGGERED)])

    assert merged[0].alert_state == al.TRIGGERED
    assert merged[1].alert_state == al.ARMED
