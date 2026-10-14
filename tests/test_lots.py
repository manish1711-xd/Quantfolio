"""FIFO tax-lot accounting — ``src.analytics.lots``.

Everything downstream of this module (realised P&L, XIRR, the holdings that feed
the positions table, the tax split on the report) is only as trustworthy as the
matching engine underneath it. So these tests are deliberately arithmetic-first:
almost every expected number is worked out by hand in a comment above the
assertion, and the properties that must hold for *any* ledger are asserted as
identities that cannot rot as the implementation changes.

Verified here
-------------
* the central worked example — BUY 10@100, BUY 10@120, SELL 15@150 realises 10
  units at cost 100 and 5 units at cost 120 and leaves one open lot of 5 @ 120;
* buy fees CAPITALISE into ``cost_per_share`` (10 shares at 100 with 10 of
  brokerage -> 101.0 per share), sell fees are APPORTIONED PRO-RATA across the
  slices the sale consumed and sum back exactly to the fee paid;
* conservation identities: shares in = shares matched + shares open, and
  gross buy outlay = realised cost basis + open cost basis;
* overselling degrades — it emits a problem message, consumes only what exists
  and never creates a negative lot;
* ``holdings_from_transactions`` drops fully-closed tickers and reports the
  QUANTITY-weighted average cost of the open lots (not a naive mean);
* ``realised_summary`` conventions: win_rate is wins / all slices (a zero-P&L
  slice is neither), profit_factor is NaN — not infinity — with no losers, and
  the India long-term boundary is *strictly* more than 365 days;
* ``parse_transactions`` validation: broker action/column aliases, bad dates,
  non-positive quantity/price, unknown actions, BUY-before-SELL on the same day;
* ``cash_flows`` sign convention: money out of the investor's pocket is
  negative, money in is positive.

Deterministic, offline, no RNG beyond the seeded shared fixtures.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics.lots import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    TRANSACTION_COLUMNS,
    build_lots,
    cash_flows,
    holdings_from_transactions,
    lots_dataframe,
    parse_transactions,
    realised_dataframe,
    realised_summary,
)
from src.analytics.positions import REQUIRED_COLUMNS as POSITION_REQUIRED_COLUMNS
from src.models import RealisedTrade, Transaction

# ---------------------------------------------------------------------------
# local builders — used only where a hand-built ledger makes the arithmetic
# obvious at a glance; everything else uses the shared conftest fixtures.
# ---------------------------------------------------------------------------


def _txn(
    ticker: str,
    day: date,
    action: str,
    quantity: float,
    price: float,
    fees: float = 0.0,
) -> Transaction:
    return Transaction(
        ticker=ticker,
        trade_date=day,
        action=action,  # type: ignore[arg-type]
        quantity=quantity,
        price=price,
        fees=fees,
    )


def _realised(
    pnl_price: float,
    *,
    open_day: date = date(2023, 1, 1),
    close_day: date = date(2023, 4, 1),
    quantity: float = 10.0,
    cost: float = 100.0,
) -> RealisedTrade:
    """A fee-free slice of ``quantity`` shares bought at ``cost``, sold at ``pnl_price``."""
    return RealisedTrade(
        ticker="AAA",
        open_date=open_day,
        close_date=close_day,
        quantity=quantity,
        cost_per_share=cost,
        sale_per_share=pnl_price,
        fees=0.0,
    )


@pytest.fixture
def core_ledger() -> list[Transaction]:
    """The central worked example, fee-free so the arithmetic is unclouded.

    BUY 10 @ 100  ->  lot A: 10 shares, cost 100/share
    BUY 10 @ 120  ->  lot B: 10 shares, cost 120/share
    SELL 15 @ 150 ->  eats all 10 of lot A, then 5 of lot B
    """
    return [
        _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0),
        _txn("AAA", date(2023, 6, 15), "BUY", 10, 120.0),
        _txn("AAA", date(2024, 8, 20), "SELL", 15, 150.0),
    ]


# ---------------------------------------------------------------------------
# the central worked example: BUY 10@100, BUY 10@120, SELL 15@150
# ---------------------------------------------------------------------------
def test_a_sell_consumes_the_oldest_lot_first(core_ledger: list[Transaction]) -> None:
    """FIFO: the 2023-01-10 lot is exhausted before the 2023-06-15 lot is touched."""
    _lots, realised, problems = build_lots(core_ledger)

    assert problems == []
    assert len(realised) == 2, "one RealisedTrade per FIFO slice consumed"

    # slice 1 — the whole of the older lot: 10 shares at cost 100
    assert realised[0].open_date == date(2023, 1, 10)
    assert realised[0].quantity == pytest.approx(10.0)
    assert realised[0].cost_per_share == pytest.approx(100.0)

    # slice 2 — the remainder comes out of the newer lot: 5 shares at cost 120
    assert realised[1].open_date == date(2023, 6, 15)
    assert realised[1].quantity == pytest.approx(5.0)
    assert realised[1].cost_per_share == pytest.approx(120.0)


def test_the_worked_example_leaves_exactly_one_open_lot_of_five_at_one_twenty(
    core_ledger: list[Transaction],
) -> None:
    """20 bought - 15 sold = 5 left, and they must be the newer, dearer shares."""
    lots, _realised, _problems = build_lots(core_ledger)

    assert len(lots) == 1
    lot = lots[0]
    assert lot.ticker == "AAA"
    assert lot.open_date == date(2023, 6, 15)
    assert lot.quantity == pytest.approx(5.0)
    assert lot.cost_per_share == pytest.approx(120.0)
    # 5 shares x 120 = 600 of cost basis still at risk
    assert lot.cost_basis == pytest.approx(600.0)


def test_the_worked_example_realises_six_hundred_and_fifty(
    core_ledger: list[Transaction],
) -> None:
    """slice 1: 10 x (150 - 100) = 500;  slice 2: 5 x (150 - 120) = 150;  total 650."""
    _lots, realised, _problems = build_lots(core_ledger)

    assert realised[0].realised_pnl == pytest.approx(500.0)
    assert realised[1].realised_pnl == pytest.approx(150.0)
    assert sum(t.realised_pnl for t in realised) == pytest.approx(650.0)

    # and as percentages of their own basis: 500/1000 = 50%, 150/600 = 25%
    assert realised[0].realised_pct == pytest.approx(50.0)
    assert realised[1].realised_pct == pytest.approx(25.0)


def test_average_cost_matching_is_not_used(core_ledger: list[Transaction]) -> None:
    """Guard against a silent switch to average cost.

    Average cost would price all 15 sold shares at (10*100 + 10*120)/20 = 110,
    realising 15 x (150 - 110) = 600 and leaving 5 shares at 110. FIFO gives 650
    and leaves them at 120. The two answers differ, so this pins the convention.
    """
    lots, realised, _problems = build_lots(core_ledger)

    assert sum(t.realised_pnl for t in realised) == pytest.approx(650.0)
    assert sum(t.realised_pnl for t in realised) != pytest.approx(600.0)
    assert lots[0].cost_per_share == pytest.approx(120.0)
    assert lots[0].cost_per_share != pytest.approx(110.0)


# ---------------------------------------------------------------------------
# conservation identities — true for ANY ledger, so they cannot rot
# ---------------------------------------------------------------------------
def test_shares_bought_equal_shares_matched_plus_shares_still_open(
    transactions: list[Transaction],
) -> None:
    """Nothing is created or destroyed by the matcher.

    conftest ledger: bought 10 + 10 (AAA) + 5 (BBB) = 25; sold 15; open 10.
    """
    lots, realised, _problems = build_lots(transactions)

    bought = sum(t.quantity for t in transactions if t.action == "BUY")
    matched = sum(t.quantity for t in realised)
    still_open = sum(lot.quantity for lot in lots)

    assert bought == pytest.approx(25.0)
    assert matched == pytest.approx(15.0)
    assert still_open == pytest.approx(10.0)
    assert matched + still_open == pytest.approx(bought)


def test_gross_buy_outlay_equals_realised_basis_plus_open_basis(
    transactions: list[Transaction],
) -> None:
    """Cost basis is conserved, fees included.

    Buys:  10*100 + 10 = 1010;  10*120 + 12 = 1212;  5*140 + 7 = 707  -> 2929
    Realised basis: 10*101.0 = 1010  and  5*121.2 = 606               -> 1616
    Open basis:      5*121.2 = 606   and  5*141.4 = 707               -> 1313
    1616 + 1313 = 2929
    """
    lots, realised, _problems = build_lots(transactions)

    outlay = sum(t.quantity * t.price + t.fees for t in transactions if t.action == "BUY")
    assert outlay == pytest.approx(2929.0)

    realised_basis = sum(t.cost_basis for t in realised)
    open_basis = sum(lot.cost_basis for lot in lots)
    assert realised_basis == pytest.approx(1616.0)
    assert open_basis == pytest.approx(1313.0)
    assert realised_basis + open_basis == pytest.approx(outlay)


def test_the_matcher_does_not_mutate_its_input(core_ledger: list[Transaction]) -> None:
    """build_lots sorts defensively — into a copy, never in place."""
    shuffled = [core_ledger[2], core_ledger[0], core_ledger[1]]
    snapshot = list(shuffled)

    build_lots(shuffled)

    assert shuffled == snapshot


def test_unsorted_input_gives_the_same_answer_as_sorted_input(
    core_ledger: list[Transaction],
) -> None:
    """The engine is order-insensitive because it sorts internally."""
    reversed_ledger = list(reversed(core_ledger))

    lots_a, realised_a, problems_a = build_lots(core_ledger)
    lots_b, realised_b, problems_b = build_lots(reversed_ledger)

    assert lots_a == lots_b
    assert realised_a == realised_b
    assert problems_a == problems_b == []


# ---------------------------------------------------------------------------
# fees: buys capitalise, sells apportion pro-rata
# ---------------------------------------------------------------------------
def test_a_buy_fee_capitalises_into_cost_per_share() -> None:
    """10 shares at 100 with 10 of brokerage: 1010 / 10 = 101.0 per share."""
    lots, _realised, _problems = build_lots(
        [_txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=10.0)]
    )

    assert lots[0].cost_per_share == pytest.approx(101.0)
    assert lots[0].cost_basis == pytest.approx(1010.0)


def test_a_zero_fee_buy_leaves_cost_per_share_equal_to_the_traded_price() -> None:
    lots, _realised, _problems = build_lots(
        [_txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=0.0)]
    )
    assert lots[0].cost_per_share == pytest.approx(100.0)


def test_the_capitalised_buy_fee_must_be_earned_back_before_a_trade_is_a_win() -> None:
    """BUY 10 @ 100 + 10 fees, SELL 10 @ 100.5 with no sell fee.

    basis    = 10 * 101.0 = 1010
    proceeds = 10 * 100.5 = 1005
    P&L      = -5  ->  a LOSS despite selling above the traded buy price.
    """
    _lots, realised, _problems = build_lots(
        [
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=10.0),
            _txn("AAA", date(2023, 3, 10), "SELL", 10, 100.5),
        ]
    )

    assert realised[0].realised_pnl == pytest.approx(-5.0)


def test_a_sell_fee_is_apportioned_pro_rata_across_the_slices_it_consumed() -> None:
    """SELL 15 with 30 of fees against slices of 10 and 5.

    slice 1: 30 * 10/15 = 20.0
    slice 2: 30 *  5/15 = 10.0
    """
    _lots, realised, _problems = build_lots(
        [
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 6, 15), "BUY", 10, 120.0),
            _txn("AAA", date(2024, 8, 20), "SELL", 15, 150.0, fees=30.0),
        ]
    )

    assert realised[0].fees == pytest.approx(20.0)
    assert realised[1].fees == pytest.approx(10.0)


def test_slice_fees_sum_exactly_back_to_the_fee_actually_paid() -> None:
    """A fee that does not divide evenly: 7.00 across slices of 3, 3 and 4 of 10.

    3/10 * 7 = 2.10, 3/10 * 7 = 2.10, and the last slice takes the residual
    7 - 4.20 = 2.80. Anything else either loses or invents money.
    """
    _lots, realised, _problems = build_lots(
        [
            _txn("AAA", date(2023, 1, 1), "BUY", 3, 100.0),
            _txn("AAA", date(2023, 1, 2), "BUY", 3, 100.0),
            _txn("AAA", date(2023, 1, 3), "BUY", 4, 100.0),
            _txn("AAA", date(2023, 6, 1), "SELL", 10, 150.0, fees=7.0),
        ]
    )

    assert [t.quantity for t in realised] == pytest.approx([3.0, 3.0, 4.0])
    assert [t.fees for t in realised] == pytest.approx([2.1, 2.1, 2.8])
    assert sum(t.fees for t in realised) == pytest.approx(7.0, abs=1e-12)


def test_proceeds_are_net_of_the_apportioned_sell_fee() -> None:
    """conftest AAA sale: SELL 15 @ 150 with 15 of fees -> 10.0 and 5.0 per slice.

    slice 1 proceeds = 10 * 150 - 10 = 1490; basis 10 * 101.0 = 1010; P&L 480
    slice 2 proceeds =  5 * 150 -  5 =  745; basis  5 * 121.2 =  606; P&L 139
    """
    txns = [
        _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=10.0),
        _txn("AAA", date(2023, 6, 15), "BUY", 10, 120.0, fees=12.0),
        _txn("AAA", date(2024, 8, 20), "SELL", 15, 150.0, fees=15.0),
    ]
    _lots, realised, _problems = build_lots(txns)

    assert realised[0].fees == pytest.approx(10.0)
    assert realised[0].proceeds == pytest.approx(1490.0)
    assert realised[0].realised_pnl == pytest.approx(480.0)

    assert realised[1].fees == pytest.approx(5.0)
    assert realised[1].proceeds == pytest.approx(745.0)
    assert realised[1].realised_pnl == pytest.approx(139.0)


def test_a_negative_fee_is_treated_as_zero_never_as_a_credit() -> None:
    """A fee is a cost. A negative one is bad data, not free money."""
    lots, _realised, _problems = build_lots(
        [_txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=-50.0)]
    )
    assert lots[0].cost_per_share == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# overselling — degrade with a message, never with an exception or a short lot
# ---------------------------------------------------------------------------
def test_overselling_consumes_only_what_exists_and_reports_the_problem() -> None:
    """BUY 10 @ 100 then SELL 15 @ 150: only 10 can be matched."""
    lots, realised, problems = build_lots(
        [
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 6, 1), "SELL", 15, 150.0),
        ]
    )

    assert lots == [], "the position is flat, not short"
    assert len(realised) == 1
    assert realised[0].quantity == pytest.approx(10.0)
    # 10 x (150 - 100) = 500 realised; the phantom 5 shares realise nothing
    assert realised[0].realised_pnl == pytest.approx(500.0)

    assert len(problems) == 1
    assert problems[0] == ("AAA: sell of 15 exceeds holding of 10 on 2023-06-01 - excess ignored.")


def test_overselling_never_creates_a_negative_lot() -> None:
    """Every lot the engine hands back holds a strictly positive share count."""
    lots, _realised, problems = build_lots(
        [
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 2, 1), "SELL", 4, 130.0),
            _txn("AAA", date(2023, 3, 1), "SELL", 40, 140.0),
            _txn("AAA", date(2023, 4, 1), "SELL", 1, 150.0),
        ]
    )

    assert all(lot.quantity > 0 for lot in lots)
    assert sum(lot.quantity for lot in lots) == pytest.approx(0.0)
    # 4 matched, then the remaining 6, then nothing left for the last sell
    assert len(problems) == 2


def test_selling_a_ticker_that_was_never_bought_realises_nothing() -> None:
    lots, realised, problems = build_lots([_txn("ZZZ", date(2023, 6, 1), "SELL", 3, 50.0)])

    assert lots == []
    assert realised == []
    assert problems == ["ZZZ: sell of 3 exceeds holding of 0 on 2023-06-01 - excess ignored."]


def test_a_same_day_round_trip_is_not_mistaken_for_an_oversell() -> None:
    """BUY settles before SELL on the same calendar date, even if listed after it."""
    day = date(2023, 5, 1)
    lots, realised, problems = build_lots(
        [_txn("AAA", day, "SELL", 10, 150.0), _txn("AAA", day, "BUY", 10, 100.0)]
    )

    assert problems == []
    assert lots == []
    assert len(realised) == 1
    assert realised[0].realised_pnl == pytest.approx(500.0)  # 10 x (150 - 100)
    assert realised[0].holding_days == 0


def test_a_whitespace_padded_action_still_sorts_buy_before_sell() -> None:
    """Regression guard: _sort_key must normalise the action the same way build_lots does.

    _sort_key once called only ``.upper()`` while build_lots called
    ``.strip().upper()``. A padded " BUY " then missed _ACTION_ORDER, sorted
    after a same-date SELL, and fabricated a spurious oversell out of a clean
    round trip. A CSV with a trailing space in one cell is not exotic.
    """
    day = date(2023, 5, 1)
    lots, realised, problems = build_lots(
        [_txn("AAA", day, "SELL", 10, 150.0), _txn("AAA", day, " BUY ", 10, 100.0)]
    )

    assert problems == []
    assert lots == []
    assert len(realised) == 1


# ---------------------------------------------------------------------------
# malformed transactions reaching the engine directly
# ---------------------------------------------------------------------------
def test_a_transaction_with_no_valid_date_is_ignored_with_a_message() -> None:
    lots, realised, problems = build_lots([_txn("AAA", "not-a-date", "BUY", 10, 100.0)])

    assert (lots, realised) == ([], [])
    assert problems == ["AAA: transaction with no valid date - ignored."]


@pytest.mark.parametrize("quantity", [0.0, -5.0, float("nan")])
def test_a_non_positive_quantity_is_ignored_with_a_message(quantity: float) -> None:
    lots, realised, problems = build_lots([_txn("AAA", date(2023, 1, 1), "BUY", quantity, 100.0)])

    assert (lots, realised) == ([], [])
    assert problems == ["AAA: non-positive quantity on 2023-01-01 - transaction ignored."]


@pytest.mark.parametrize("price", [0.0, -100.0, float("nan")])
def test_a_non_positive_price_is_ignored_with_a_message(price: float) -> None:
    lots, realised, problems = build_lots([_txn("AAA", date(2023, 1, 1), "BUY", 10, price)])

    assert (lots, realised) == ([], [])
    assert problems == ["AAA: non-positive price on 2023-01-01 - transaction ignored."]


def test_an_unknown_action_is_ignored_with_a_message() -> None:
    lots, realised, problems = build_lots([_txn("AAA", date(2023, 1, 1), "DIVIDEND", 10, 100.0)])

    assert (lots, realised) == ([], [])
    assert problems == ["AAA: unknown action 'DIVIDEND' - transaction ignored."]


def test_tickers_are_folded_to_upper_case_so_one_book_is_kept_per_name() -> None:
    """'aaa' and 'AAA' are the same instrument, not two half-positions."""
    lots, realised, problems = build_lots(
        [
            _txn("aaa", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 6, 1), "SELL", 10, 150.0),
        ]
    )

    assert problems == []
    assert lots == []
    assert len(realised) == 1
    assert realised[0].ticker == "AAA"


def test_an_empty_ledger_produces_empty_output_and_no_problems() -> None:
    assert build_lots([]) == ([], [], [])


# ---------------------------------------------------------------------------
# holdings_from_transactions
# ---------------------------------------------------------------------------
def test_holdings_use_the_quantity_weighted_average_of_the_open_lots() -> None:
    """BUY 10 @ 100, BUY 30 @ 200, SELL 5 @ 150 (no fees).

    FIFO leaves 5 shares at 100 and 30 shares at 200.
    quantity  = 35
    cost      = 5*100 + 30*200 = 500 + 6000 = 6500
    buy_price = 6500 / 35 = 185.714285...
    A naive mean of the lot prices would give (100 + 200)/2 = 150, so the two
    answers are far apart and this genuinely pins the weighting.
    """
    frame = holdings_from_transactions(
        [
            _txn("AAA", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 2, 1), "BUY", 30, 200.0),
            _txn("AAA", date(2023, 3, 1), "SELL", 5, 150.0),
        ]
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["quantity"] == pytest.approx(35.0)
    assert row["buy_price"] == pytest.approx(6500.0 / 35.0)
    assert row["buy_price"] == pytest.approx(185.7142857142857, rel=1e-12)


def test_holdings_exclude_fully_closed_tickers() -> None:
    """CLOSED is bought and sold out entirely; only OPEN should survive."""
    frame = holdings_from_transactions(
        [
            _txn("CLOSED", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("CLOSED", date(2023, 6, 1), "SELL", 10, 150.0),
            _txn("OPEN", date(2023, 1, 1), "BUY", 4, 50.0),
        ]
    )

    assert list(frame["ticker"]) == ["OPEN"]
    assert frame.iloc[0]["quantity"] == pytest.approx(4.0)


def test_holdings_carry_the_capitalised_fee_into_buy_price(
    transactions: list[Transaction],
) -> None:
    """conftest ledger leaves AAA 5 @ 121.2 (120 + 12/10) and BBB 5 @ 141.4 (140 + 7/5)."""
    frame = holdings_from_transactions(transactions)

    assert list(frame["ticker"]) == ["AAA", "BBB"]
    assert list(frame["quantity"]) == pytest.approx([5.0, 5.0])
    assert list(frame["buy_price"]) == pytest.approx([121.2, 141.4])


def test_holdings_columns_match_the_positions_input_contract() -> None:
    """The frame must feed straight into positions.clean_holdings."""
    frame = holdings_from_transactions([_txn("AAA", date(2023, 1, 1), "BUY", 10, 100.0)])
    assert tuple(frame.columns) == POSITION_REQUIRED_COLUMNS


def test_holdings_from_an_empty_ledger_is_an_empty_frame_with_the_right_columns() -> None:
    frame = holdings_from_transactions([])

    assert frame.empty
    assert tuple(frame.columns) == POSITION_REQUIRED_COLUMNS
    assert frame["quantity"].dtype == np.dtype("float64")
    assert frame["buy_price"].dtype == np.dtype("float64")


def test_holdings_from_a_fully_closed_book_is_an_empty_frame() -> None:
    frame = holdings_from_transactions(
        [
            _txn("AAA", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("AAA", date(2023, 6, 1), "SELL", 10, 150.0),
        ]
    )
    assert frame.empty
    assert tuple(frame.columns) == POSITION_REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# lots_dataframe / realised_dataframe
# ---------------------------------------------------------------------------
def test_lots_dataframe_cost_basis_is_quantity_times_cost_per_share(
    transactions: list[Transaction],
) -> None:
    """AAA: 5 x 121.2 = 606.0;  BBB: 5 x 141.4 = 707.0"""
    lots, _realised, _problems = build_lots(transactions)
    frame = lots_dataframe(lots)

    assert list(frame.columns) == [
        "ticker",
        "open_date",
        "quantity",
        "cost_per_share",
        "cost_basis",
    ]
    assert list(frame["ticker"]) == ["AAA", "BBB"]
    assert list(frame["cost_basis"]) == pytest.approx([606.0, 707.0])
    assert list(frame["cost_basis"]) == pytest.approx(
        list(frame["quantity"] * frame["cost_per_share"])
    )


def test_lots_dataframe_of_nothing_is_an_empty_frame_with_the_right_columns() -> None:
    frame = lots_dataframe([])
    assert frame.empty
    assert list(frame.columns) == [
        "ticker",
        "open_date",
        "quantity",
        "cost_per_share",
        "cost_basis",
    ]


def test_realised_dataframe_reports_the_hand_computed_slice_arithmetic(
    transactions: list[Transaction],
) -> None:
    """conftest ledger, both slices closed on 2024-08-20.

    slice 1: q 10, cost 101.0, sale 150, fee 10 -> basis 1010, proceeds 1490,
             P&L 480, pct 480/1010*100 = 47.5247524752...,
             held 2023-01-10 -> 2024-08-20 = 588 days -> Long
    slice 2: q  5, cost 121.2, sale 150, fee  5 -> basis  606, proceeds  745,
             P&L 139, pct 139/606*100 = 22.9372937294...,
             held 2023-06-15 -> 2024-08-20 = 432 days -> Long
    """
    _lots, realised, _problems = build_lots(transactions)
    frame = realised_dataframe(realised)

    assert len(frame) == 2
    assert list(frame["cost_basis"]) == pytest.approx([1010.0, 606.0])
    assert list(frame["proceeds"]) == pytest.approx([1490.0, 745.0])
    assert list(frame["realised_pnl"]) == pytest.approx([480.0, 139.0])
    assert list(frame["realised_pct"]) == pytest.approx(
        [480.0 / 1010.0 * 100.0, 139.0 / 606.0 * 100.0]
    )
    assert list(frame["holding_days"]) == [588, 432]
    assert list(frame["term"]) == ["Long", "Long"]

    # identity: realised_pnl == proceeds - cost_basis, row by row
    assert list(frame["realised_pnl"]) == pytest.approx(
        list(frame["proceeds"] - frame["cost_basis"])
    )


def test_realised_dataframe_of_nothing_is_an_empty_frame_with_the_right_columns() -> None:
    frame = realised_dataframe([])
    assert frame.empty
    assert "realised_pnl" in frame.columns
    assert "term" in frame.columns
    assert len(frame.columns) == 13


# ---------------------------------------------------------------------------
# realised_summary
# ---------------------------------------------------------------------------
@pytest.fixture
def mixed_trades() -> list[RealisedTrade]:
    """Four fee-free slices of 10 shares bought at 100 — deliberately one of each.

    sold at 150 -> P&L 10 * (150 - 100) = +500   win
    sold at  90 -> P&L 10 * ( 90 - 100) = -100   loss
    sold at 100 -> P&L 10 * (100 - 100) =    0   neither
    sold at 120 -> P&L 10 * (120 - 100) = +200   win
    """
    return [
        _realised(150.0),
        _realised(90.0),
        _realised(100.0),
        _realised(120.0),
    ]


def test_win_rate_is_winners_over_every_slice_not_over_winners_plus_losers(
    mixed_trades: list[RealisedTrade],
) -> None:
    """2 winners out of 4 slices = 50.0%.

    If the flat trade were excluded from the denominator the answer would be
    2/3 = 66.67%, so this discriminates between the two conventions.
    """
    summary = realised_summary(mixed_trades)

    assert summary["n_trades"] == 4
    assert summary["win_rate"] == pytest.approx(50.0)


def test_profit_factor_is_gross_wins_over_gross_losses(
    mixed_trades: list[RealisedTrade],
) -> None:
    """gross wins 500 + 200 = 700; gross losses |-100| = 100; 700 / 100 = 7.0"""
    summary = realised_summary(mixed_trades)

    assert summary["profit_factor"] == pytest.approx(7.0)
    assert summary["avg_win"] == pytest.approx(350.0)  # (500 + 200) / 2
    assert summary["avg_loss"] == pytest.approx(-100.0)  # only one loser
    assert summary["best"] == pytest.approx(500.0)
    assert summary["worst"] == pytest.approx(-100.0)
    assert summary["total_realised"] == pytest.approx(600.0)  # 500 - 100 + 0 + 200
    assert summary["total_cost_basis"] == pytest.approx(4000.0)  # 4 x 10 x 100


def test_profit_factor_is_nan_when_there_are_no_losing_trades() -> None:
    """An undefined ratio, not 'infinitely good' and not a fabricated 0.0."""
    summary = realised_summary([_realised(150.0), _realised(120.0)])

    assert np.isnan(summary["profit_factor"])
    assert summary["win_rate"] == pytest.approx(100.0)
    assert np.isnan(summary["avg_loss"])
    assert summary["avg_win"] == pytest.approx(350.0)  # (500 + 200) / 2


def test_avg_win_and_avg_loss_reconstruct_the_total(
    mixed_trades: list[RealisedTrade],
) -> None:
    """Identity: 2 wins x 350 + 1 loss x -100 + one flat trade = 600 realised."""
    summary = realised_summary(mixed_trades)

    n_wins = 2
    n_losses = 1
    assert summary["avg_win"] * n_wins + summary["avg_loss"] * n_losses == pytest.approx(
        summary["total_realised"]
    )


def test_avg_holding_days_is_the_unweighted_mean_of_the_slice_holding_periods() -> None:
    """2023-01-01 -> 2023-04-01 is 90 days; -> 2023-01-11 is 10; -> 2023-03-02 is 60.

    mean = (90 + 10 + 60) / 3 = 160 / 3 = 53.333...
    """
    trades = [
        _realised(150.0, close_day=date(2023, 4, 1)),
        _realised(150.0, close_day=date(2023, 1, 11)),
        _realised(150.0, close_day=date(2023, 3, 2)),
    ]
    summary = realised_summary(trades)

    assert summary["avg_holding_days"] == pytest.approx(160.0 / 3.0)


def test_exactly_365_days_is_short_term_and_366_days_is_long_term() -> None:
    """India: listed equity turns long-term STRICTLY beyond 365 calendar days.

    2023-01-01 -> 2024-01-01 is 365 days (2023 is not a leap year) -> Short
    2023-01-01 -> 2024-01-02 is 366 days                            -> Long
    """
    short = _realised(110.0, open_day=date(2023, 1, 1), close_day=date(2024, 1, 1))
    long = _realised(130.0, open_day=date(2023, 1, 1), close_day=date(2024, 1, 2))

    assert short.holding_days == 365
    assert short.is_long_term is False
    assert long.holding_days == 366
    assert long.is_long_term is True

    # short P&L 10 * (110 - 100) = +100;  long P&L 10 * (130 - 100) = +300
    summary = realised_summary([short, long])
    assert summary["short_term_pnl"] == pytest.approx(100.0)
    assert summary["long_term_pnl"] == pytest.approx(300.0)
    assert summary["short_term_pnl"] + summary["long_term_pnl"] == pytest.approx(
        summary["total_realised"]
    )
    assert summary["avg_holding_days"] == pytest.approx(365.5)  # (365 + 366) / 2


def test_the_365_day_boundary_survives_the_whole_engine_end_to_end() -> None:
    """Same boundary, but reached through build_lots -> realised_dataframe."""
    _lots, realised, problems = build_lots(
        [
            _txn("STX", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("STX", date(2024, 1, 1), "SELL", 10, 110.0),
            _txn("LTX", date(2023, 1, 1), "BUY", 10, 100.0),
            _txn("LTX", date(2024, 1, 2), "SELL", 10, 130.0),
        ]
    )
    assert problems == []

    frame = realised_dataframe(realised)  # sorted by close_date: STX then LTX
    assert list(frame["ticker"]) == ["STX", "LTX"]
    assert list(frame["holding_days"]) == [365, 366]
    assert list(frame["term"]) == ["Short", "Long"]


def test_short_term_pnl_is_nan_when_every_slice_is_long_term(
    transactions: list[Transaction],
) -> None:
    """conftest slices were held 588 and 432 days — nothing short-term happened,
    and a fabricated 0.0 would read as 'realised nothing short-term', which is
    a different claim."""
    _lots, realised, _problems = build_lots(transactions)
    summary = realised_summary(realised)

    assert np.isnan(summary["short_term_pnl"])
    assert summary["long_term_pnl"] == pytest.approx(619.0)  # 480 + 139


def test_the_conftest_ledger_summary_matches_hand_arithmetic(
    transactions: list[Transaction],
) -> None:
    """Two winning slices: +480 and +139.

    n_trades         2
    total_realised   619
    total_cost_basis 1010 + 606 = 1616
    win_rate         2/2 = 100%
    avg_win          619 / 2 = 309.5
    best / worst     480 / 139
    avg_holding_days (588 + 432) / 2 = 510
    """
    _lots, realised, _problems = build_lots(transactions)
    summary = realised_summary(realised)

    assert summary["n_trades"] == 2
    assert summary["total_realised"] == pytest.approx(619.0)
    assert summary["total_cost_basis"] == pytest.approx(1616.0)
    assert summary["win_rate"] == pytest.approx(100.0)
    assert summary["avg_win"] == pytest.approx(309.5)
    assert summary["best"] == pytest.approx(480.0)
    assert summary["worst"] == pytest.approx(139.0)
    assert summary["avg_holding_days"] == pytest.approx(510.0)
    assert np.isnan(summary["avg_loss"])
    assert np.isnan(summary["profit_factor"])


def test_a_summary_of_no_trades_is_every_float_nan_with_a_zero_trade_count() -> None:
    """Nothing has been realised; 0.0 would read as a real, measured result."""
    summary = realised_summary([])

    assert summary["n_trades"] == 0
    floats = {k: v for k, v in summary.items() if k != "n_trades"}
    assert len(floats) == 11
    assert all(np.isnan(v) for v in floats.values()), floats


def test_best_is_never_below_worst(mixed_trades: list[RealisedTrade]) -> None:
    summary = realised_summary(mixed_trades)
    assert summary["best"] >= summary["worst"]


# ---------------------------------------------------------------------------
# parse_transactions — validation of raw user / broker input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("B", "BUY"),
        ("BUY", "BUY"),
        ("buy", "BUY"),
        ("Buy", "BUY"),
        ("PURCHASE", "BUY"),
        ("purchased", "BUY"),
        ("BOUGHT", "BUY"),
        ("S", "SELL"),
        ("SELL", "SELL"),
        ("sell", "SELL"),
        ("SALE", "SELL"),
        ("sold", "SELL"),
    ],
)
def test_broker_action_spellings_map_onto_buy_and_sell(spelling: str, expected: str) -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "trade_date": ["2023-01-10"],
            "action": [spelling],
            "quantity": [10],
            "price": [100.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert len(txns) == 1
    assert txns[0].action == expected


def test_an_unknown_action_is_rejected_with_a_row_numbered_message() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "trade_date": ["2023-01-10", "2023-01-11"],
            "action": ["BUY", "DIVIDEND"],
            "quantity": [10, 5],
            "price": [100.0, 50.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert [t.ticker for t in txns] == ["AAA"]
    assert problems == ["BBB (row 2): unknown action 'DIVIDEND' - row skipped."]


def test_an_unparseable_trade_date_is_rejected_not_silently_coerced() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "trade_date": ["2023-01-10", "not-a-date"],
            "action": ["BUY", "BUY"],
            "quantity": [10, 5],
            "price": [100.0, 50.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert [t.ticker for t in txns] == ["AAA"]
    assert problems == ["BBB (row 2): unparseable trade date - row skipped."]


@pytest.mark.parametrize(
    ("quantity", "price"),
    [(0, 100.0), (-5, 100.0), (10, 0.0), (10, -100.0), ("abc", 100.0), (10, "abc")],
)
def test_non_positive_or_unparseable_quantity_and_price_are_rejected(
    quantity: object, price: object
) -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "trade_date": ["2023-01-10"],
            "action": ["BUY"],
            "quantity": [quantity],
            "price": [price],
        }
    )
    txns, problems = parse_transactions(raw)

    assert txns == []
    assert problems[0] == (
        "AAA (row 1): quantity and price must be positive numbers - row skipped."
    )
    assert "No valid transactions after validation." in problems


def test_buy_is_ordered_before_sell_on_the_same_calendar_date() -> None:
    """The rows arrive SELL-first; the parser must hand back BUY-first so the
    FIFO engine does not read a same-day round trip as an oversell."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "trade_date": ["2023-03-01", "2023-03-01"],
            "action": ["SELL", "BUY"],
            "quantity": [5, 10],
            "price": [150.0, 100.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert [t.action for t in txns] == ["BUY", "SELL"]

    # and the engine agrees: 10 bought, 5 sold, 5 open at 100, no problems
    lots, realised, engine_problems = build_lots(txns)
    assert engine_problems == []
    assert lots[0].quantity == pytest.approx(5.0)
    assert realised[0].realised_pnl == pytest.approx(250.0)  # 5 x (150 - 100)


def test_rows_are_returned_in_chronological_order() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "trade_date": ["2024-08-20", "2023-01-10", "2023-06-15"],
            "action": ["SELL", "BUY", "BUY"],
            "quantity": [15, 10, 10],
            "price": [150.0, 100.0, 120.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert [t.trade_date for t in txns] == [
        date(2023, 1, 10),
        date(2023, 6, 15),
        date(2024, 8, 20),
    ]


def test_common_broker_column_aliases_are_resolved() -> None:
    """Symbol/Trade Date/Side/Qty/Rate/Brokerage is a typical export header."""
    raw = pd.DataFrame(
        {
            "Symbol": ["aaa "],
            "Trade Date": ["2023-01-10"],
            "Side": ["b"],
            "Qty": [10],
            "Rate": [100.0],
            "Brokerage": [10.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert len(txns) == 1
    txn = txns[0]
    assert txn.ticker == "AAA"  # stripped and upper-cased
    assert txn.action == "BUY"
    assert txn.quantity == pytest.approx(10.0)
    assert txn.price == pytest.approx(100.0)
    assert txn.fees == pytest.approx(10.0)


def test_a_missing_fees_column_defaults_every_fee_to_zero() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "trade_date": ["2023-01-10"],
            "action": ["BUY"],
            "quantity": [10],
            "price": [100.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert txns[0].fees == pytest.approx(0.0)


@pytest.mark.parametrize("bad_fee", [-25.0, "abc", None])
def test_negative_or_unparseable_fees_become_zero_never_a_credit(
    bad_fee: object,
) -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "trade_date": ["2023-01-10"],
            "action": ["BUY"],
            "quantity": [10],
            "price": [100.0],
            "fees": [bad_fee],
        }
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert txns[0].fees == pytest.approx(0.0)


def test_a_blank_ticker_row_is_dropped_without_being_called_a_problem() -> None:
    """An empty row in the Streamlit editor grid is not a data error."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "   ", None],
            "trade_date": ["2023-01-10", "", ""],
            "action": ["BUY", "", ""],
            "quantity": [10, np.nan, np.nan],
            "price": [100.0, np.nan, np.nan],
        }
    )
    txns, problems = parse_transactions(raw)

    assert [t.ticker for t in txns] == ["AAA"]
    assert problems == []


def test_duplicate_index_labels_do_not_confuse_the_row_validator() -> None:
    """A concatenated CSV can arrive with a repeated index; parsing must not
    raise and must not lose rows."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "trade_date": ["2023-01-10", "2023-01-11", "2023-01-12"],
            "action": ["BUY", "BUY", "BUY"],
            "quantity": [10, 5, 3],
            "price": [100.0, 50.0, 20.0],
        },
        index=[7, 7, 7],
    )
    txns, problems = parse_transactions(raw)

    assert problems == []
    assert [t.ticker for t in txns] == ["AAA", "BBB", "CCC"]


def test_parsing_none_reports_a_problem_rather_than_raising() -> None:
    assert parse_transactions(None) == ([], ["No transactions supplied."])


def test_parsing_an_empty_frame_reports_a_problem_rather_than_raising() -> None:
    assert parse_transactions(pd.DataFrame()) == ([], ["No transactions supplied."])


def test_missing_required_columns_are_named_in_the_problem_message() -> None:
    raw = pd.DataFrame({"ticker": ["AAA"], "quantity": [10]})
    txns, problems = parse_transactions(raw)

    assert txns == []
    assert problems == ["Missing column(s): trade_date, action, price."]


def test_a_frame_where_every_row_is_invalid_says_so_explicitly() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "trade_date": ["nope", "2023-01-11"],
            "action": ["BUY", "SPLIT"],
            "quantity": [10, 5],
            "price": [100.0, 50.0],
        }
    )
    txns, problems = parse_transactions(raw)

    assert txns == []
    assert problems[-1] == "No valid transactions after validation."


def test_the_shared_transactions_frame_parses_into_the_shared_transactions_list(
    transactions_frame: pd.DataFrame, transactions: list[Transaction]
) -> None:
    """The two conftest fixtures are two spellings of one ledger; they must agree.

    The frame carries the mixed spellings BUY / buy / B, which is exactly the
    normalisation this parser exists to perform.
    """
    parsed, problems = parse_transactions(transactions_frame)

    assert problems == []
    assert len(parsed) == len(transactions)
    for got, want in zip(parsed, sorted(transactions, key=lambda t: (t.trade_date,)), strict=True):
        assert got.ticker == want.ticker
        assert got.trade_date == want.trade_date
        assert got.action == want.action
        assert got.quantity == pytest.approx(float(want.quantity))
        assert got.price == pytest.approx(want.price)
        assert got.fees == pytest.approx(want.fees)


def test_the_input_frame_is_not_mutated_by_parsing(
    transactions_frame: pd.DataFrame,
) -> None:
    before = transactions_frame.copy(deep=True)
    parse_transactions(transactions_frame)
    pd.testing.assert_frame_equal(transactions_frame, before)


def test_the_declared_schema_constants_are_self_consistent() -> None:
    """REQUIRED + OPTIONAL must cover the canonical column list."""
    assert set(REQUIRED_COLUMNS).issubset(set(TRANSACTION_COLUMNS))
    assert "fees" in OPTIONAL_COLUMNS
    assert "fees" not in REQUIRED_COLUMNS
    assert set(TRANSACTION_COLUMNS) - set(REQUIRED_COLUMNS) == {"fees"}


# ---------------------------------------------------------------------------
# cash_flows — the investor's pocket, for XIRR
# ---------------------------------------------------------------------------
def test_a_buy_is_a_negative_cash_flow_and_a_sell_a_positive_one() -> None:
    """BUY 10 @ 100 with 10 fees -> -(1000 + 10) = -1010
    SELL 15 @ 150 with 15 fees ->  (2250 - 15) = +2235
    """
    flows = cash_flows(
        [
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=10.0),
            _txn("AAA", date(2024, 8, 20), "SELL", 15, 150.0, fees=15.0),
        ]
    )

    assert [d for d, _ in flows] == [date(2023, 1, 10), date(2024, 8, 20)]
    assert flows[0][1] == pytest.approx(-1010.0)
    assert flows[1][1] == pytest.approx(2235.0)


def test_fees_push_a_buy_further_negative_and_pull_a_sell_less_positive() -> None:
    """Fees always work against the investor, whichever side of the trade."""
    buy_free = cash_flows([_txn("AAA", date(2023, 1, 1), "BUY", 10, 100.0)])[0][1]
    buy_fee = cash_flows([_txn("AAA", date(2023, 1, 1), "BUY", 10, 100.0, fees=10.0)])[0][1]
    sell_free = cash_flows([_txn("AAA", date(2023, 1, 1), "SELL", 10, 100.0)])[0][1]
    sell_fee = cash_flows([_txn("AAA", date(2023, 1, 1), "SELL", 10, 100.0, fees=10.0)])[0][1]

    assert buy_free == pytest.approx(-1000.0)
    assert buy_fee == pytest.approx(-1010.0)
    assert sell_free == pytest.approx(1000.0)
    assert sell_fee == pytest.approx(990.0)


def test_the_conftest_ledger_cash_flows_match_hand_arithmetic(
    transactions: list[Transaction],
) -> None:
    """-(1000 + 10), -(1200 + 12), -(700 + 7), +(2250 - 15)
    = -1010, -1212, -707, +2235   -> net -694 still at risk in the market.
    """
    flows = cash_flows(transactions)

    assert [d for d, _ in flows] == [
        date(2023, 1, 10),
        date(2023, 6, 15),
        date(2023, 7, 1),
        date(2024, 8, 20),
    ]
    assert [amount for _, amount in flows] == pytest.approx([-1010.0, -1212.0, -707.0, 2235.0])
    assert sum(amount for _, amount in flows) == pytest.approx(-694.0)


def test_cash_flows_are_dated_in_order_with_buy_ahead_of_sell(
    transactions: list[Transaction],
) -> None:
    day = date(2023, 5, 1)
    flows = cash_flows(
        [
            _txn("AAA", day, "SELL", 10, 150.0),
            _txn("AAA", day, "BUY", 10, 100.0),
        ]
    )

    assert [amount for _, amount in flows] == pytest.approx([-1000.0, 1500.0])

    dates = [d for d, _ in cash_flows(transactions)]
    assert dates == sorted(dates)


def test_cash_flows_of_an_empty_ledger_is_an_empty_list() -> None:
    assert cash_flows([]) == []


def test_a_transaction_with_an_unparseable_date_is_skipped_not_poisoned() -> None:
    """A NaT date would silently destroy any XIRR solve it reached."""
    flows = cash_flows(
        [
            _txn("AAA", "not-a-date", "BUY", 10, 100.0),
            _txn("AAA", date(2023, 1, 10), "BUY", 10, 100.0),
        ]
    )

    assert flows == [(date(2023, 1, 10), pytest.approx(-1000.0))]
