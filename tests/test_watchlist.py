"""The watchlist — ``src.watchlist``, pure and offline.

Verified here
-------------
**A watchlist entry is not a holding.** It carries no quantity and no cost
basis, and the table it produces has no P&L, no weight and no investment
column. That separation is the reason this is its own module: a watched name
must never reach position P&L, portfolio weights, risk statistics or
attribution.

**No target is not a zero target.** ``target_price=None`` and
``target_price=0`` are different facts, and conflating them would render "no
opinion" as "expecting a 100% fall to zero". Construction normalises 0, "" and
NaN to ``None``, and only a positive number is kept.

**Sign convention of the distance to target.** Positive means the target is
ABOVE the current price. Pinned in both directions against hand-computed
values, because a distance-to-target with an ambiguous sign is worse than no
number.

**Missing data stays missing.** No quote means NaN price, not 0. No 52-week
range means NaN range position, not 50 — a fabricated midpoint would read as
"fairly valued". A watched instrument whose quote FAILED still appears as a
row, because dropping it would look to the user like it had been removed.

**Every collection operation is a replacement.** ``add``/``remove``/
``update_item`` return new tuples and never mutate their input, so a Streamlit
rerun cannot observe a half-applied edit.

**Frame round-trip is lossless and order-stable.** ``to_frame`` then
``from_frame`` returns the same items in the same order — the property that
makes SQLite persistence safe, since saving is a wholesale replace.

Deterministic, offline, no RNG, no network. Quotes and profiles are hand-built
fixtures.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src import watchlist as wl
from src.models import Quote, SecurityProfile

NOW = datetime(2025, 9, 1, 10, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def quote(ticker: str, price: float, previous: float | None = None, ccy: str = "INR") -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        currency=ccy,
        as_of=NOW,
        fetched_at=NOW,
        is_intraday=False,
        previous_close=previous,
    )


def profile(ticker: str, low: float | None = None, high: float | None = None) -> SecurityProfile:
    return SecurityProfile(
        ticker=ticker,
        name=f"{ticker} Industries Ltd",
        sector="Energy",
        currency="INR",
        quote_type="EQUITY",
        fifty_two_week_low=low,
        fifty_two_week_high=high,
    )


@pytest.fixture
def items() -> tuple[wl.WatchItem, ...]:
    return (
        wl.WatchItem(
            "RELIANCE.NS",
            note="core holding candidate",
            target_price=1500.0,
            added_at=date(2025, 1, 10),
        ),
        wl.WatchItem("TCS.NS", note="", target_price=None, added_at=date(2025, 2, 20)),
        wl.WatchItem("AAPL", note="fx exposure", target_price=150.0, added_at=date(2025, 3, 5)),
    )


@pytest.fixture
def market() -> tuple[dict[str, Quote], dict[str, SecurityProfile]]:
    quotes = {
        "RELIANCE.NS": quote("RELIANCE.NS", 1200.0, previous=1180.0),
        "TCS.NS": quote("TCS.NS", 3000.0, previous=3060.0),
        "AAPL": quote("AAPL", 200.0, previous=200.0, ccy="USD"),
    }
    profiles = {
        "RELIANCE.NS": profile("RELIANCE.NS", low=1000.0, high=1400.0),
        "TCS.NS": profile("TCS.NS", low=2800.0, high=3200.0),
        "AAPL": profile("AAPL"),  # no 52-week range reported
    }
    return quotes, profiles


# ---------------------------------------------------------------------------
# WatchItem construction
# ---------------------------------------------------------------------------
def test_the_ticker_is_upper_cased_and_stripped_on_construction() -> None:
    assert wl.WatchItem("  reliance.ns  ").ticker == "RELIANCE.NS"


@pytest.mark.parametrize("blank", [None, 0, 0.0, "", "nan", float("nan"), -5.0])
def test_no_target_and_a_zero_or_negative_target_are_both_none(blank: object) -> None:
    """0.0 would render as "expecting a 100% fall to zero"."""
    assert wl.WatchItem("X", target_price=blank).target_price is None  # type: ignore[arg-type]
    assert not wl.WatchItem("X", target_price=blank).has_target  # type: ignore[arg-type]


def test_a_positive_target_is_kept_as_a_float() -> None:
    item = wl.WatchItem("X", target_price="1500")  # type: ignore[arg-type]
    assert item.target_price == 1500.0
    assert item.has_target


def test_a_note_is_truncated_rather_than_rejected() -> None:
    item = wl.WatchItem("X", note="a" * 500)
    assert len(item.note) == wl.MAX_NOTE_LENGTH


@pytest.mark.parametrize("junk", [None, float("nan"), "nan", "None", "  "])
def test_a_missing_note_becomes_an_empty_string_not_the_text_nan(junk: object) -> None:
    assert wl.WatchItem("X", note=junk).note == ""  # type: ignore[arg-type]


def test_a_watch_item_is_immutable() -> None:
    item = wl.WatchItem("X")
    with pytest.raises((AttributeError, TypeError)):
        item.ticker = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# distance to target — the sign convention
# ---------------------------------------------------------------------------
def test_a_target_above_the_price_is_a_positive_distance() -> None:
    """1500 target, 1200 price -> +25%. Hand-computed."""
    item = wl.WatchItem("X", target_price=1500.0)
    assert item.distance_to_target(1200.0) == pytest.approx(25.0)


def test_a_target_below_the_price_is_a_negative_distance() -> None:
    """1000 target, 1250 price -> -20%. Hand-computed."""
    item = wl.WatchItem("X", target_price=1000.0)
    assert item.distance_to_target(1250.0) == pytest.approx(-20.0)


def test_a_target_at_the_price_is_exactly_zero() -> None:
    assert wl.WatchItem("X", target_price=1200.0).distance_to_target(1200.0) == pytest.approx(0.0)


@pytest.mark.parametrize("price", [None, 0.0, -10.0, float("nan")])
def test_an_unusable_price_makes_the_distance_undefined_not_zero(price: object) -> None:
    item = wl.WatchItem("X", target_price=1500.0)
    assert math.isnan(item.distance_to_target(price))  # type: ignore[arg-type]


def test_no_target_makes_the_distance_undefined() -> None:
    assert math.isnan(wl.WatchItem("X").distance_to_target(1200.0))


# ---------------------------------------------------------------------------
# collection operations
# ---------------------------------------------------------------------------
def test_add_appends_and_reports_no_problem() -> None:
    items, problem = wl.add((), "RELIANCE.NS")
    assert problem == ""
    assert [i.ticker for i in items] == ["RELIANCE.NS"]


def test_add_resolves_a_company_name_to_a_symbol() -> None:
    items, problem = wl.add((), "reliance")
    assert problem == ""
    assert items[0].ticker == "RELIANCE.NS"


def test_add_can_be_told_not_to_resolve() -> None:
    items, _ = wl.add((), "reliance", resolve=False)
    assert items[0].ticker == "RELIANCE"


def test_add_refuses_a_duplicate_and_says_why() -> None:
    first, _ = wl.add((), "RELIANCE.NS")
    second, problem = wl.add(first, "RELIANCE.NS")
    assert second == first
    assert "already" in problem


def test_add_refuses_a_duplicate_reached_by_a_different_spelling() -> None:
    first, _ = wl.add((), "RELIANCE.NS")
    _, problem = wl.add(first, "reliance")
    assert "already" in problem


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_add_refuses_a_blank_ticker_and_says_why(blank: object) -> None:
    items, problem = wl.add((), blank)
    assert items == ()
    assert problem


def test_add_refuses_past_the_size_cap_and_names_the_cap() -> None:
    full = tuple(wl.WatchItem(f"T{i}.NS") for i in range(wl.MAX_ITEMS))
    items, problem = wl.add(full, "RELIANCE.NS")
    assert len(items) == wl.MAX_ITEMS
    assert str(wl.MAX_ITEMS) in problem


def test_add_stamps_todays_date_when_none_is_given() -> None:
    items, _ = wl.add((), "RELIANCE.NS")
    assert items[0].added_at == datetime.now(timezone.utc).date()


def test_add_does_not_mutate_the_input(items: tuple[wl.WatchItem, ...]) -> None:
    before = len(items)
    wl.add(items, "INFY.NS")
    assert len(items) == before


def test_remove_drops_only_the_named_ticker(items: tuple[wl.WatchItem, ...]) -> None:
    remaining = wl.remove(items, "TCS.NS")
    assert [i.ticker for i in remaining] == ["RELIANCE.NS", "AAPL"]


def test_remove_of_an_absent_ticker_is_a_no_op(items: tuple[wl.WatchItem, ...]) -> None:
    assert wl.remove(items, "NOTTHERE") == items


def test_remove_is_case_insensitive(items: tuple[wl.WatchItem, ...]) -> None:
    assert len(wl.remove(items, "tcs.ns")) == len(items) - 1


def test_contains_matches_case_insensitively(items: tuple[wl.WatchItem, ...]) -> None:
    assert wl.contains(items, "tcs.ns")
    assert not wl.contains(items, "INFY.NS")


def test_update_item_changes_the_note_and_preserves_order(
    items: tuple[wl.WatchItem, ...],
) -> None:
    updated = wl.update_item(items, "TCS.NS", note="new note")
    assert [i.ticker for i in updated] == [i.ticker for i in items]
    assert updated[1].note == "new note"


def test_update_item_changes_the_target(items: tuple[wl.WatchItem, ...]) -> None:
    updated = wl.update_item(items, "TCS.NS", target_price=3500.0)
    assert updated[1].target_price == 3500.0


def test_update_item_leaves_the_target_alone_when_none_is_passed(
    items: tuple[wl.WatchItem, ...],
) -> None:
    """None means "no change"; clearing needs an explicit flag."""
    updated = wl.update_item(items, "RELIANCE.NS", note="edited", target_price=None)
    assert updated[0].target_price == 1500.0


def test_update_item_clears_the_target_only_when_told_to(
    items: tuple[wl.WatchItem, ...],
) -> None:
    updated = wl.update_item(items, "RELIANCE.NS", clear_target=True)
    assert updated[0].target_price is None


def test_update_item_of_an_absent_ticker_changes_nothing(
    items: tuple[wl.WatchItem, ...],
) -> None:
    assert wl.update_item(items, "NOTTHERE", note="x") == items


# ---------------------------------------------------------------------------
# frame round-trip
# ---------------------------------------------------------------------------
def test_to_frame_has_the_documented_column_contract(items: tuple[wl.WatchItem, ...]) -> None:
    assert tuple(wl.to_frame(items).columns) == wl.WATCHLIST_COLUMNS


def test_an_empty_watchlist_yields_a_typed_empty_frame() -> None:
    frame = wl.to_frame(())
    assert frame.empty
    assert tuple(frame.columns) == wl.WATCHLIST_COLUMNS
    assert frame["target_price"].dtype == "float64"


def test_the_frame_round_trip_is_lossless_and_order_stable(
    items: tuple[wl.WatchItem, ...],
) -> None:
    """The property that makes wholesale-replace persistence safe."""
    assert wl.from_frame(wl.to_frame(items)) == items


def test_a_missing_target_round_trips_as_none_not_zero(
    items: tuple[wl.WatchItem, ...],
) -> None:
    restored = wl.from_frame(wl.to_frame(items))
    assert restored[1].target_price is None


def test_from_frame_of_none_or_empty_is_an_empty_tuple() -> None:
    assert wl.from_frame(None) == ()
    assert wl.from_frame(pd.DataFrame()) == ()


def test_from_frame_without_a_ticker_column_is_an_empty_tuple() -> None:
    assert wl.from_frame(pd.DataFrame({"note": ["x"]})) == ()


def test_from_frame_matches_column_names_case_insensitively() -> None:
    frame = pd.DataFrame({"Ticker": ["RELIANCE.NS"], "Target Price": [1500.0]})
    restored = wl.from_frame(frame)
    assert restored[0].ticker == "RELIANCE.NS"
    assert restored[0].target_price == 1500.0


def test_from_frame_drops_blank_rows_from_a_grid_editor() -> None:
    """Empty data_editor rows are not data the user lost."""
    frame = pd.DataFrame({"ticker": ["RELIANCE.NS", "", None, "TCS.NS"]})
    assert [i.ticker for i in wl.from_frame(frame)] == ["RELIANCE.NS", "TCS.NS"]


def test_from_frame_keeps_the_first_of_a_duplicated_ticker() -> None:
    frame = pd.DataFrame({"ticker": ["X", "X"], "note": ["first", "second"]})
    restored = wl.from_frame(frame)
    assert len(restored) == 1
    assert restored[0].note == "first"


# ---------------------------------------------------------------------------
# the display table
# ---------------------------------------------------------------------------
def test_the_table_has_exactly_the_documented_columns(items, market) -> None:
    quotes, profiles = market
    assert tuple(wl.watch_table(items, quotes, profiles).columns) == wl.TABLE_COLUMNS


def test_the_table_carries_no_pnl_weight_or_investment_column(items, market) -> None:
    """A watched name has no cost basis; these columns must not exist."""
    quotes, profiles = market
    columns = set(wl.watch_table(items, quotes, profiles).columns)
    assert not columns & {"pnl", "pnl_pct", "weight_pct", "investment", "current_value"}


def test_an_empty_watchlist_yields_a_typed_empty_table() -> None:
    table = wl.watch_table(())
    assert table.empty
    assert tuple(table.columns) == wl.TABLE_COLUMNS


def test_one_row_per_watched_instrument(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert len(table) == len(items)
    assert list(table["ticker"]) == [i.ticker for i in items]


def test_the_price_comes_from_the_quote(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert float(table.loc[0, "price"]) == 1200.0


def test_the_day_change_is_computed_against_the_previous_close(items, market) -> None:
    """1200 from 1180 -> +1.6949%. Hand-computed."""
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert float(table.loc[0, "day_change_pct"]) == pytest.approx((1200.0 / 1180.0 - 1) * 100.0)


def test_the_range_position_is_zero_at_the_low_and_one_hundred_at_the_high() -> None:
    items = (wl.WatchItem("A"), wl.WatchItem("B"), wl.WatchItem("C"))
    quotes = {
        "A": quote("A", 1000.0),
        "B": quote("B", 1400.0),
        "C": quote("C", 1200.0),
    }
    profiles = {
        "A": profile("A", low=1000.0, high=1400.0),
        "B": profile("B", low=1000.0, high=1400.0),
        "C": profile("C", low=1000.0, high=1400.0),
    }
    table = wl.watch_table(items, quotes, profiles)
    assert float(table.loc[0, "range_position_pct"]) == pytest.approx(0.0)
    assert float(table.loc[1, "range_position_pct"]) == pytest.approx(100.0)
    assert float(table.loc[2, "range_position_pct"]) == pytest.approx(50.0)


def test_a_missing_52_week_range_leaves_the_position_undefined_not_mid_range(items, market) -> None:
    """A fabricated 50 would read as "fairly valued"."""
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert math.isnan(float(table.loc[2, "range_position_pct"])), "AAPL reports no range"


def test_a_degenerate_range_leaves_the_position_undefined() -> None:
    items = (wl.WatchItem("A"),)
    table = wl.watch_table(
        items, {"A": quote("A", 100.0)}, {"A": profile("A", low=100.0, high=100.0)}
    )
    assert math.isnan(float(table.loc[0, "range_position_pct"]))


def test_the_distance_to_target_appears_in_the_table(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert float(table.loc[0, "to_target_pct"]) == pytest.approx(25.0)
    assert math.isnan(float(table.loc[1, "to_target_pct"])), "TCS has no target"


def test_an_unpriced_instrument_still_appears_as_a_row_with_nan_numbers(items) -> None:
    """Dropping it would look to the user like the entry had been removed."""
    table = wl.watch_table(items, quotes={}, profiles={})
    assert len(table) == len(items)
    assert table["price"].isna().all()
    assert table["day_change_pct"].isna().all()


def test_an_unpriced_instrument_falls_back_to_its_ticker_as_a_name(items) -> None:
    table = wl.watch_table(items, quotes={}, profiles={})
    assert table.loc[0, "name"] == "RELIANCE.NS"


def test_holdings_are_flagged_and_others_are_not(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles, held_tickers=["TCS.NS"])
    assert list(table["in_portfolio"]) == [False, True, False]


def test_the_held_flag_is_matched_case_insensitively(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles, held_tickers=["tcs.ns"])
    assert bool(table.loc[1, "in_portfolio"])


def test_no_holdings_means_nothing_is_flagged(items, market) -> None:
    quotes, profiles = market
    assert not wl.watch_table(items, quotes, profiles)["in_portfolio"].any()


def test_the_numeric_columns_are_float_dtype(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    for column in ("price", "day_change_pct", "target_price", "to_target_pct"):
        assert table[column].dtype == "float64", column


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
def test_the_summary_counts_items_priced_rows_and_targets(items, market) -> None:
    quotes, profiles = market
    stats = wl.summarise(wl.watch_table(items, quotes, profiles))
    assert stats["n_items"] == 3
    assert stats["n_priced"] == 3
    assert stats["n_with_target"] == 2


def test_the_summary_names_the_best_and_worst_movers(items, market) -> None:
    quotes, profiles = market
    stats = wl.summarise(wl.watch_table(items, quotes, profiles))
    assert stats["best_ticker"] == "RELIANCE.NS", "the only riser"
    assert stats["worst_ticker"] == "TCS.NS", "3000 from 3060 is the biggest fall"


def test_the_summary_average_is_the_unweighted_mean_of_priced_rows(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    expected = table["day_change_pct"].dropna().mean()
    assert wl.summarise(table)["avg_day_change_pct"] == pytest.approx(expected)


def test_the_summary_of_an_empty_table_is_zeros_and_nans_not_an_exception() -> None:
    stats = wl.summarise(wl.watch_table(()))
    assert stats["n_items"] == 0
    assert math.isnan(stats["avg_day_change_pct"])
    assert stats["best_ticker"] is None


def test_the_summary_of_none_is_the_empty_summary() -> None:
    assert wl.summarise(None)["n_items"] == 0


def test_an_all_unpriced_table_reports_an_undefined_average_not_zero(items) -> None:
    """A zero average would read as a flat market."""
    stats = wl.summarise(wl.watch_table(items, {}, {}))
    assert stats["n_priced"] == 0
    assert math.isnan(stats["avg_day_change_pct"])


def test_the_summary_counts_held_names(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles, held_tickers=["TCS.NS", "AAPL"])
    assert wl.summarise(table)["n_in_portfolio"] == 2


# ---------------------------------------------------------------------------
# proximity to target
# ---------------------------------------------------------------------------
def test_only_rows_within_the_tolerance_are_reported(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    # RELIANCE is +25% from target, AAPL is -25% from target.
    assert wl.triggered(table, tolerance_pct=2.0).empty
    assert len(wl.triggered(table, tolerance_pct=30.0)) == 2


def test_a_row_exactly_at_its_target_is_reported() -> None:
    items = (wl.WatchItem("A", target_price=1200.0),)
    table = wl.watch_table(items, {"A": quote("A", 1200.0)}, {"A": profile("A")})
    near = wl.triggered(table, tolerance_pct=0.0)
    assert len(near) == 1
    assert near["ticker"].iloc[0] == "A"


def test_a_row_with_no_target_is_never_reported() -> None:
    items = (wl.WatchItem("A"),)
    table = wl.watch_table(items, {"A": quote("A", 1200.0)}, {"A": profile("A")})
    assert wl.triggered(table, tolerance_pct=100.0).empty


def test_triggered_of_an_empty_or_none_table_is_empty() -> None:
    assert wl.triggered(None).empty
    assert wl.triggered(wl.watch_table(())).empty


def test_a_negative_tolerance_is_treated_as_zero() -> None:
    items = (wl.WatchItem("A", target_price=1200.0),)
    table = wl.watch_table(items, {"A": quote("A", 1200.0)}, {"A": profile("A")})
    assert len(wl.triggered(table, tolerance_pct=-5.0)) == 1


def test_triggered_preserves_the_table_columns(items, market) -> None:
    quotes, profiles = market
    table = wl.watch_table(items, quotes, profiles)
    assert tuple(wl.triggered(table, tolerance_pct=100.0).columns) == wl.TABLE_COLUMNS


# ---------------------------------------------------------------------------
# ticker normalisation
# ---------------------------------------------------------------------------
def test_normalise_ticker_resolves_a_company_name() -> None:
    assert wl.normalise_ticker("reliance") == "RELIANCE.NS"


def test_normalise_ticker_prefers_the_indian_line_of_an_ambiguous_symbol() -> None:
    """India-first, matching the importer, so one watchlist entry means one thing."""
    assert wl.normalise_ticker("INFY") == "INFY.NS"


def test_normalise_ticker_passes_an_unknown_symbol_through() -> None:
    assert wl.normalise_ticker("somesmallcap") == "SOMESMALLCAP"


@pytest.mark.parametrize("blank", ["", "   ", None, float("nan")])
def test_normalise_ticker_of_blank_input_is_empty(blank: object) -> None:
    assert wl.normalise_ticker(blank) == ""


# ---------------------------------------------------------------------------
# the alert rule carried on a WatchItem
# ---------------------------------------------------------------------------
def test_a_new_item_has_no_alert() -> None:
    """Opt-in: a target is a reminder, an e-mail is an interruption, and one
    must never imply the other."""
    item = wl.WatchItem("AAA.NS", target_price=110.0)

    assert item.alert_enabled is False
    assert item.direction == ""
    assert item.alert_state == "armed"
    assert item.last_fired is None
    assert not item.alerting


def test_alerting_needs_a_target_a_switch_and_a_direction() -> None:
    """``alerting`` is the honest "will this ever fire?" question, so a rule
    missing any one of the three must read as inactive rather than armed."""
    assert wl.WatchItem("A", target_price=110.0, alert_enabled=True, direction="above").alerting
    assert not wl.WatchItem("A", alert_enabled=True, direction="above").alerting
    assert not wl.WatchItem("A", target_price=110.0, direction="above").alerting
    assert not wl.WatchItem("A", target_price=110.0, alert_enabled=True).alerting


def test_a_reference_price_is_enough_for_alerting_without_a_direction() -> None:
    """The direction can be inferred from it, so the rule is genuinely live."""
    item = wl.WatchItem("A", target_price=110.0, alert_enabled=True, reference_price=100.0)

    assert item.alerting


@pytest.mark.parametrize(
    ("given", "expected"),
    [(True, True), (False, False), (1, True), (0, False), ("true", True), ("0", False)],
)
def test_alert_enabled_accepts_the_spellings_a_store_returns(given: object, expected: bool) -> None:
    assert wl.WatchItem("A", alert_enabled=given).alert_enabled is expected


def test_a_nan_alert_flag_reads_as_off() -> None:
    """Regression guard. ``bool(float("nan"))`` is **True**, so a frame column
    that exists but holds no value once switched alerting ON for every row."""
    assert wl.WatchItem("A", alert_enabled=float("nan")).alert_enabled is False


def test_an_unrecognised_alert_state_falls_back_to_armed() -> None:
    """A corrupt state must not leave a rule permanently silent."""
    assert wl.WatchItem("A", alert_state="exploded").alert_state == "armed"
    assert wl.WatchItem("A", alert_state="").alert_state == "armed"


def test_the_direction_is_stored_lower_case() -> None:
    assert wl.WatchItem("A", direction="ABOVE").direction == "above"


def test_a_non_positive_reference_price_is_dropped() -> None:
    assert wl.WatchItem("A", reference_price=0.0).reference_price is None
    assert wl.WatchItem("A", reference_price=-5.0).reference_price is None


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("2026-09-03T10:30:00", datetime(2026, 9, 3, 10, 30)),
        ("", None),
        ("not a date", None),
        (None, None),
        (True, None),
    ],
)
def test_last_fired_parses_what_sqlite_returns(stored: object, expected: object) -> None:
    """An unreadable timestamp reads as "never fired", which risks one extra
    alert rather than silently suppressing every future one."""
    assert wl.WatchItem("A", last_fired=stored).last_fired == expected


def test_an_aware_last_fired_is_stored_naive() -> None:
    item = wl.WatchItem("A", last_fired=datetime(2026, 9, 3, 5, 0, tzinfo=timezone.utc))

    assert item.last_fired == datetime(2026, 9, 3, 5, 0)
    assert item.last_fired.tzinfo is None


def test_an_iso_string_with_an_offset_is_stored_naive() -> None:
    item = wl.WatchItem("A", last_fired="2026-09-03T10:30:00+05:30")

    assert item.last_fired is not None
    assert item.last_fired.tzinfo is None


# ---------------------------------------------------------------------------
# add / update
# ---------------------------------------------------------------------------
def test_add_can_set_an_alert_rule_up_front() -> None:
    items, problem = wl.add(
        (),
        "AAA.NS",
        target_price=110.0,
        direction="above",
        alert_enabled=True,
        reference_price=100.0,
        resolve=False,
    )

    assert problem == ""
    assert items[0].alerting


def test_updating_the_target_rearms_the_rule() -> None:
    """A new level has not been reached yet, whatever the old one had done.
    Without the reset an edited rule would inherit a stale "triggered" and go
    permanently silent."""
    items = (
        wl.WatchItem(
            "A",
            target_price=110.0,
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
        ),
    )

    updated = wl.update_item(items, "A", target_price=120.0)

    assert updated[0].alert_state == "armed"


def test_re_saving_the_same_target_does_not_rearm() -> None:
    """Pressing Update without changing the level is not news."""
    items = (
        wl.WatchItem(
            "A",
            target_price=110.0,
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
        ),
    )

    updated = wl.update_item(items, "A", target_price=110.0, note="new note")

    assert updated[0].alert_state == "triggered"
    assert updated[0].note == "new note"


def test_changing_the_direction_rearms_the_rule() -> None:
    items = (
        wl.WatchItem(
            "A",
            target_price=110.0,
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
        ),
    )

    updated = wl.update_item(items, "A", direction="below")

    assert updated[0].direction == "below"
    assert updated[0].alert_state == "armed"


def test_re_saving_the_same_direction_does_not_rearm() -> None:
    items = (
        wl.WatchItem(
            "A",
            target_price=110.0,
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
        ),
    )

    assert wl.update_item(items, "A", direction="above")[0].alert_state == "triggered"


def test_clearing_the_target_also_clears_the_alert() -> None:
    """A rule with no level cannot fire, and leaving the switch on would show
    an active alert that never triggers."""
    items = (
        wl.WatchItem(
            "A",
            target_price=110.0,
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
            last_fired=datetime(2026, 9, 3),
        ),
    )

    updated = wl.update_item(items, "A", clear_target=True)

    assert updated[0].target_price is None
    assert updated[0].alert_enabled is False
    assert updated[0].direction == ""
    assert updated[0].alert_state == "armed"
    assert updated[0].last_fired is None


def test_switching_the_alert_off_keeps_the_target() -> None:
    items = (wl.WatchItem("A", target_price=110.0, direction="above", alert_enabled=True),)

    updated = wl.update_item(items, "A", alert_enabled=False)

    assert updated[0].alert_enabled is False
    assert updated[0].target_price == 110.0


def test_update_leaves_alert_fields_alone_when_not_given() -> None:
    """``None`` means "leave it", the same convention as the note."""
    original = wl.WatchItem(
        "A",
        target_price=110.0,
        direction="above",
        alert_enabled=True,
        alert_state="triggered",
    )

    updated = wl.update_item((original,), "A", note="just a note")

    assert updated[0].direction == "above"
    assert updated[0].alert_enabled is True
    assert updated[0].alert_state == "triggered"


def test_a_reference_price_can_be_captured_later() -> None:
    items = (wl.WatchItem("A", target_price=110.0),)

    updated = wl.update_item(items, "A", reference_price=100.0)

    assert updated[0].reference_price == 100.0


# ---------------------------------------------------------------------------
# frame round trip
# ---------------------------------------------------------------------------
def test_the_alert_rule_survives_a_frame_round_trip() -> None:
    original = (
        wl.WatchItem(
            "AAA.NS",
            note="keep",
            target_price=110.0,
            added_at=date(2026, 1, 15),
            direction="above",
            alert_enabled=True,
            alert_state="triggered",
            last_fired=datetime(2026, 9, 3, 10, 30),
            reference_price=100.0,
        ),
    )

    assert wl.from_frame(wl.to_frame(original)) == original


def test_a_frame_written_before_alerts_existed_loads_as_no_alert() -> None:
    """An upgrade must not start e-mailing about targets set years ago."""
    legacy = pd.DataFrame(
        [{"ticker": "AAA.NS", "note": "old", "target_price": 110.0, "added_at": "2026-01-15"}]
    )

    item = wl.from_frame(legacy)[0]

    assert item.target_price == 110.0
    assert item.alert_enabled is False
    assert item.direction == ""


def test_an_empty_frame_still_carries_the_alert_columns() -> None:
    frame = wl.to_frame([])

    assert tuple(frame.columns) == wl.WATCHLIST_COLUMNS
    assert frame.empty


def test_the_table_reports_only_rules_that_can_fire() -> None:
    """``alert_enabled`` in the display table is ``alerting``, not the raw
    switch: a rule with no direction must not render as armed."""
    items = (
        wl.WatchItem("AAA.NS", target_price=110.0, alert_enabled=True, direction="above"),
        wl.WatchItem("BBB.NS", target_price=110.0, alert_enabled=True),  # no direction
    )

    table = wl.watch_table(items)

    assert list(table["alert_enabled"]) == [True, False]


def test_the_table_carries_the_direction_and_state() -> None:
    items = (
        wl.WatchItem(
            "AAA.NS",
            target_price=110.0,
            alert_enabled=True,
            direction="below",
            alert_state="triggered",
        ),
    )

    table = wl.watch_table(items)

    assert table["direction"].iloc[0] == "below"
    assert table["alert_state"].iloc[0] == "triggered"


# ---------------------------------------------------------------------------
# with_alert: the seam back from the state machine
# ---------------------------------------------------------------------------
def test_with_alert_copies_only_the_alert_fields() -> None:
    """A cron run must not overwrite a note or a target the user edited in the
    browser while it was fetching quotes."""
    stored = (wl.WatchItem("A", note="edited", target_price=115.0, alert_enabled=True),)
    evaluated = (
        wl.WatchItem(
            "A",
            note="stale",
            target_price=110.0,
            alert_enabled=True,
            direction="above",
            alert_state="triggered",
            last_fired=datetime(2026, 9, 3, 10, 30),
        ),
    )

    merged = wl.with_alert(stored, evaluated)

    assert merged[0].note == "edited"
    assert merged[0].target_price == 115.0
    assert merged[0].alert_state == "triggered"
    assert merged[0].last_fired == datetime(2026, 9, 3, 10, 30)


def test_with_alert_leaves_unmatched_items_untouched() -> None:
    stored = (wl.WatchItem("A", target_price=1.0), wl.WatchItem("B", target_price=2.0))

    merged = wl.with_alert(stored, (wl.WatchItem("A", alert_state="triggered"),))

    assert merged[1] == stored[1]


def test_with_alert_preserves_order_and_count() -> None:
    stored = tuple(wl.WatchItem(f"T{i}.NS") for i in range(5))

    merged = wl.with_alert(stored, ())

    assert [i.ticker for i in merged] == [i.ticker for i in stored]
