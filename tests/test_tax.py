"""Indian capital-gains tax and harvesting — ``src.analytics.tax``, offline.

Tax arithmetic is the one place in this application where a plausible wrong
answer has consequences outside the screen, so these tests are written against
the statute rather than against the implementation.

Verified here
-------------
**Set-off quantum is MANDATORY, not elective.** Sections 70/71 say a loss
*shall* be set off against available gains of the permitted character in the
same year. The painful consequence is pinned directly: a long-term loss set
off against a long-term gain the annual exemption already shelters is consumed
for **no tax benefit**, and cannot be held back for a better year. An earlier
draft "optimised" around that and would have reported a lower tax than the law
permits — the single most dangerous bug this module could carry — so the
mandatory behaviour and the resulting ``wasted_on_exempt_gains`` figure are
both asserted.

**Set-off ordering IS elective**, and the beneficial order is taken: STCL to
STCG first (highest rate), LTCL to LTCG next (it can go nowhere else), then
remaining STCL to LTCG.

**A loss is worth what it absorbs, not loss x rate.** Harvesting into a year
with no gains saves exactly zero and creates a carry-forward. Asserted,
because reporting the notional figure would be a fabrication.

**The July 2024 regime change.** A sale on 22 July 2024 is taxed at 15%/10%
with a ₹1 lakh exemption; two days later at 20%/12.5% with ₹1.25 lakh. Both
sides of the boundary are pinned, as is the exact effective date.

**The 12-month boundary cuts both ways.** For a gain, crossing it lowers the
rate — wait. For a loss, crossing it makes the loss *less* useful — harvest
first. That inversion is the feature's most valuable insight and is asserted
in both directions.

**Grandfathering is flagged, never faked.** Section 112A needs the highest
quoted price on 31 January 2018, which this application does not hold. A
pre-cutoff lot is reported as overstated rather than valued on a number that
is not the statutory one.

Deterministic: every date is explicit, no ``date.today()`` in an assertion.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.analytics import tax
from src.config import INDIAN_TAX_REGIMES
from src.models import Lot, RealisedTrade

#: A date inside FY2024-25 and after the July 2024 rate change.
TODAY = date(2025, 3, 1)
NEW = tax.regime_for(TODAY)
OLD = INDIAN_TAX_REGIMES[0]


def lot(ticker: str, opened: date, quantity: float, cost: float) -> Lot:
    return Lot(ticker=ticker, open_date=opened, quantity=quantity, cost_per_share=cost)


def closed(
    ticker: str, opened: date, close: date, quantity: float, cost: float, sale: float
) -> RealisedTrade:
    return RealisedTrade(
        ticker=ticker,
        open_date=opened,
        close_date=close,
        quantity=quantity,
        cost_per_share=cost,
        sale_per_share=sale,
    )


# ---------------------------------------------------------------------------
# the financial year
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "when,label",
    [
        (date(2024, 4, 1), "FY2024-25"),
        (date(2024, 12, 31), "FY2024-25"),
        (date(2025, 3, 31), "FY2024-25"),
        (date(2025, 4, 1), "FY2025-26"),
        (date(2024, 3, 31), "FY2023-24"),
    ],
)
def test_the_financial_year_runs_april_to_march(when: date, label: str) -> None:
    """A January sale belongs to the year that STARTED the previous April.
    Using the calendar year would put Q4 gains in the wrong year and misstate
    every exemption calculation."""
    assert tax.financial_year_label(when) == label


def test_the_financial_year_window_is_inclusive() -> None:
    start, end = tax.financial_year(date(2024, 12, 1))

    assert start == date(2024, 4, 1)
    assert end == date(2025, 3, 31)


# ---------------------------------------------------------------------------
# the July 2024 regime change
# ---------------------------------------------------------------------------
def test_the_pre_july_2024_regime_applies_before_the_cutover() -> None:
    regime = tax.regime_for(date(2024, 7, 22))

    assert regime.stcg_rate == 0.15
    assert regime.ltcg_rate == 0.10
    assert regime.ltcg_exemption == 100_000.0


def test_the_new_regime_applies_from_23_july_2024() -> None:
    """The exact effective date matters: a sale either side is taxed
    differently, and a tool using one rate for the whole year would be wrong
    for anyone reviewing FY2024-25."""
    regime = tax.regime_for(date(2024, 7, 23))

    assert regime.stcg_rate == 0.20
    assert regime.ltcg_rate == 0.125
    assert regime.ltcg_exemption == 125_000.0


def test_the_regimes_are_listed_in_ascending_date_order() -> None:
    dates = [r.effective_from for r in INDIAN_TAX_REGIMES]

    assert dates == sorted(dates)


def test_a_date_before_every_regime_falls_back_to_the_earliest() -> None:
    assert tax.regime_for(date(2010, 1, 1)) is INDIAN_TAX_REGIMES[0]


# ---------------------------------------------------------------------------
# the 12-month boundary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "days,expected",
    [(0, False), (100, False), (364, False), (365, False), (366, True), (500, True)],
)
def test_long_term_needs_more_than_365_days(days: int, expected: bool) -> None:
    """Strictly greater. A position sold on its 365th day is still
    short-term, and rounding that the other way would misclassify every
    anniversary sale."""
    assert tax.is_long_term(days, NEW) is expected


def test_the_boundary_matches_the_realised_trade_model() -> None:
    """Two modules must not disagree about what long-term means."""
    trade = closed("A", date(2023, 1, 1), date(2024, 1, 1), 10, 100, 120)

    assert trade.is_long_term == tax.is_long_term(trade.holding_days, NEW)


@pytest.mark.parametrize("days,remaining", [(0, 366), (300, 66), (365, 1), (366, 0), (400, 0)])
def test_days_to_long_term_counts_down_to_the_boundary(days: int, remaining: int) -> None:
    assert tax.days_to_long_term(days, NEW) == remaining


def test_holding_days_of_an_unusable_date_is_negative_not_zero() -> None:
    assert tax.holding_days(None, TODAY) == -1


# ---------------------------------------------------------------------------
# tax on net gains
# ---------------------------------------------------------------------------
def test_short_term_gains_are_taxed_at_the_flat_rate() -> None:
    amount, _exemption = tax.tax_on(200_000, 0, NEW)

    assert amount == pytest.approx(200_000 * 0.20)


def test_long_term_gains_inside_the_exemption_are_untaxed() -> None:
    amount, exemption = tax.tax_on(0, 100_000, NEW)

    assert amount == 0.0
    assert exemption == 100_000.0


def test_only_long_term_gains_above_the_exemption_are_taxed() -> None:
    """3,00,000 minus 1,25,000 = 1,75,000 at 12.5% = 21,875."""
    amount, exemption = tax.tax_on(0, 300_000, NEW)

    assert exemption == 125_000.0
    assert amount == pytest.approx(175_000 * 0.125)


def test_the_exemption_is_capped_at_the_gain() -> None:
    _amount, exemption = tax.tax_on(0, 40_000, NEW)

    assert exemption == 40_000.0


def test_negative_gains_are_floored_at_zero() -> None:
    amount, _exemption = tax.tax_on(-50_000, -50_000, NEW)

    assert amount == 0.0


def test_the_old_regime_taxes_the_same_gains_less() -> None:
    new_tax, _ = tax.tax_on(200_000, 300_000, NEW)
    old_tax, _ = tax.tax_on(200_000, 300_000, OLD)

    assert old_tax < new_tax


# ---------------------------------------------------------------------------
# set-off: MANDATORY quantum
# ---------------------------------------------------------------------------
def test_a_long_term_loss_against_an_exempt_gain_is_consumed_for_nothing() -> None:
    """The statutory trap, and the reason this module exists in this shape.

    Set-off is compulsory under Section 70, so the loss is absorbed against a
    gain the exemption already shelters. It buys no tax reduction AND it is
    gone — it cannot be carried forward to a year where it would have
    mattered."""
    result = tax.set_off(tax.GainsPosition(ltcg=100_000, ltcl=50_000), NEW)

    assert result.tax == 0.0
    assert result.ltcl_used_against_ltcg == 50_000.0
    assert result.ltcl_carried_forward == 0.0, "compulsory set-off leaves nothing to carry"
    assert result.wasted_on_exempt_gains == 50_000.0
    assert result.has_waste


def test_short_term_loss_is_also_set_off_against_exempt_long_term_gains() -> None:
    """Section 70(2) gives no opt-out either."""
    result = tax.set_off(tax.GainsPosition(ltcg=100_000, stcl=80_000), NEW)

    assert result.stcl_used_against_ltcg == 80_000.0
    assert result.stcl_carried_forward == 0.0
    assert result.wasted_on_exempt_gains == 80_000.0


def test_waste_is_only_the_portion_below_the_exemption() -> None:
    """A gain straddling the threshold: 1,50,000 gain, 1,00,000 loss. Only the
    25,000 above the exemption did any work."""
    result = tax.set_off(tax.GainsPosition(ltcg=150_000, ltcl=100_000), NEW)

    assert result.wasted_on_exempt_gains == pytest.approx(75_000.0)


def test_a_loss_fully_above_the_exemption_wastes_nothing() -> None:
    result = tax.set_off(tax.GainsPosition(ltcg=300_000, ltcl=100_000), NEW)

    assert result.wasted_on_exempt_gains == 0.0
    assert result.taxable_ltcg == pytest.approx(75_000.0)
    assert result.tax == pytest.approx(75_000 * 0.125)


def test_short_term_set_off_never_counts_as_waste() -> None:
    """There is no exemption on short-term gains, so every rupee absorbed
    there saves tax."""
    result = tax.set_off(tax.GainsPosition(stcg=200_000, stcl=150_000), NEW)

    assert result.wasted_on_exempt_gains == 0.0
    assert result.tax == pytest.approx(50_000 * 0.20)


# ---------------------------------------------------------------------------
# set-off: ELECTIVE ordering
# ---------------------------------------------------------------------------
def test_short_term_loss_is_applied_to_short_term_gains_first() -> None:
    """The beneficial election: 20% beats 12.5%."""
    result = tax.set_off(tax.GainsPosition(stcg=100_000, ltcg=400_000, stcl=100_000), NEW)

    assert result.stcl_used_against_stcg == 100_000.0
    assert result.stcl_used_against_ltcg == 0.0
    assert result.taxable_stcg == 0.0


def test_long_term_loss_takes_first_claim_on_long_term_gains() -> None:
    """LTCL can go nowhere else. Letting STCL consume the long-term gains
    first would strand a usable LTCL while spending an STCL that could have
    absorbed short-term gains."""
    result = tax.set_off(
        tax.GainsPosition(stcg=100_000, ltcg=300_000, stcl=100_000, ltcl=100_000), NEW
    )

    assert result.ltcl_used_against_ltcg == 100_000.0
    assert result.stcl_used_against_stcg == 100_000.0
    assert result.stcl_used_against_ltcg == 0.0
    assert result.tax == pytest.approx(75_000 * 0.125)


def test_a_long_term_loss_can_never_offset_a_short_term_gain() -> None:
    """Section 70(3): LTCL is set off against long-term gains ONLY."""
    result = tax.set_off(tax.GainsPosition(stcg=200_000, ltcl=200_000), NEW)

    assert result.taxable_stcg == 200_000.0
    assert result.tax == pytest.approx(200_000 * 0.20)
    assert result.ltcl_carried_forward == 200_000.0


# ---------------------------------------------------------------------------
# set-off: carry-forward
# ---------------------------------------------------------------------------
def test_an_unabsorbed_loss_is_carried_forward() -> None:
    result = tax.set_off(tax.GainsPosition(stcg=50_000, stcl=200_000), NEW)

    assert result.stcl_used_against_stcg == 50_000.0
    assert result.stcl_carried_forward == 150_000.0
    assert result.tax == 0.0


def test_with_no_gains_at_all_every_loss_carries_forward() -> None:
    result = tax.set_off(tax.GainsPosition(stcl=100_000, ltcl=80_000), NEW)

    assert result.tax == 0.0
    assert result.stcl_carried_forward == 100_000.0
    assert result.ltcl_carried_forward == 80_000.0
    assert result.total_carried_forward == 180_000.0
    assert not result.has_waste, "nothing was absorbed, so nothing was wasted"


def test_tax_is_never_negative() -> None:
    for position in (
        tax.GainsPosition(stcl=500_000),
        tax.GainsPosition(ltcl=500_000),
        tax.GainsPosition(stcg=10_000, stcl=500_000, ltcl=500_000),
    ):
        assert tax.set_off(position, NEW).tax >= 0.0


def test_the_regime_label_is_recorded_on_the_result() -> None:
    """So a stored figure can be traced to the rules that produced it."""
    assert tax.set_off(tax.GainsPosition(stcg=1000), NEW).regime == NEW.label


# ---------------------------------------------------------------------------
# realised gains for the year
# ---------------------------------------------------------------------------
def test_realised_trades_are_split_into_the_four_buckets() -> None:
    trades = [
        closed("A", date(2024, 5, 1), date(2024, 9, 1), 10, 100, 150),  # ST gain 500
        closed("B", date(2024, 5, 1), date(2024, 9, 1), 10, 100, 80),  # ST loss 200
        closed("C", date(2022, 1, 1), date(2024, 9, 1), 10, 100, 200),  # LT gain 1000
        closed("D", date(2022, 1, 1), date(2024, 9, 1), 10, 100, 60),  # LT loss 400
    ]

    position = tax.realised_gains(trades, TODAY, NEW)

    assert position.stcg == pytest.approx(500.0)
    assert position.stcl == pytest.approx(200.0)
    assert position.ltcg == pytest.approx(1000.0)
    assert position.ltcl == pytest.approx(400.0)
    assert position.n_trades == 4


def test_only_trades_closed_inside_the_financial_year_count() -> None:
    """Capital gains are assessed year by year: last year's profit cannot
    absorb this year's harvested loss."""
    trades = [
        closed("A", date(2023, 1, 1), date(2023, 6, 1), 10, 100, 200),  # FY2023-24
        closed("B", date(2024, 5, 1), date(2024, 9, 1), 10, 100, 150),  # FY2024-25
    ]

    position = tax.realised_gains(trades, TODAY, NEW)

    assert position.n_trades == 1
    assert position.stcg == pytest.approx(500.0)


def test_the_year_label_is_recorded() -> None:
    assert tax.realised_gains([], TODAY, NEW).year_label == "FY2024-25"


def test_realised_gains_of_nothing_is_an_empty_position() -> None:
    position = tax.realised_gains(None, TODAY, NEW)

    assert position.stcg == position.ltcg == position.stcl == position.ltcl == 0.0
    assert not position.has_gains


def test_net_gains_are_reported() -> None:
    position = tax.GainsPosition(stcg=100.0, stcl=40.0, ltcg=200.0, ltcl=50.0)

    assert position.net_stcg == 60.0
    assert position.net_ltcg == 150.0


# ---------------------------------------------------------------------------
# lot classification
# ---------------------------------------------------------------------------
def test_lots_are_classified_by_their_own_open_date() -> None:
    """Per LOT, not per position. Two FIFO tranches of one ticker can have
    different tax characters, and harvesting is a lot-level decision."""
    lots, _warnings = tax.classify_lots(
        [
            lot("A", date(2022, 1, 1), 10, 100),  # long-term
            lot("A", date(2024, 12, 1), 10, 100),  # short-term
        ],
        {"A": 90.0},
        TODAY,
        NEW,
    )

    assert [item.character for item in lots] == [tax.LONG_TERM, tax.SHORT_TERM]


def test_a_lot_carries_its_unrealised_gain_and_percentage() -> None:
    lots, _ = tax.classify_lots([lot("A", date(2024, 12, 1), 10, 100)], {"A": 80.0}, TODAY, NEW)
    item = lots[0]

    assert item.cost_basis == 1000.0
    assert item.market_value == 800.0
    assert item.unrealised == -200.0
    assert item.unrealised_pct == pytest.approx(-20.0)
    assert item.is_loss


def test_an_unpriced_lot_is_excluded_not_valued_at_cost() -> None:
    """Pretending it is flat would put a zero-gain lot into the harvest
    ranking as though it had been analysed."""
    lots, warnings = tax.classify_lots(
        [lot("A", date(2024, 1, 1), 10, 100), lot("NOPRICE", date(2024, 1, 1), 5, 50)],
        {"A": 90.0},
        TODAY,
        NEW,
    )

    assert [item.ticker for item in lots] == ["A"]
    assert any("NOPRICE" in w for w in warnings)


@pytest.mark.parametrize(
    "bad_lot",
    [
        Lot(ticker="", open_date=date(2024, 1, 1), quantity=10, cost_per_share=100),
        Lot(ticker="A", open_date=date(2024, 1, 1), quantity=0, cost_per_share=100),
        Lot(ticker="A", open_date=date(2024, 1, 1), quantity=-5, cost_per_share=100),
    ],
)
def test_an_unusable_lot_is_reported_not_silently_dropped(bad_lot: Lot) -> None:
    lots, warnings = tax.classify_lots([bad_lot], {"A": 90.0}, TODAY, NEW)

    assert lots == []
    assert warnings


def test_classifying_nothing_yields_nothing() -> None:
    lots, warnings = tax.classify_lots(None, None, TODAY, NEW)

    assert lots == []
    assert warnings == []


# ---------------------------------------------------------------------------
# grandfathering
# ---------------------------------------------------------------------------
def test_a_pre_2018_lot_is_flagged_as_grandfathered() -> None:
    lots, _warnings = tax.classify_lots(
        [lot("A", date(2017, 6, 1), 10, 100)], {"A": 500.0}, TODAY, NEW
    )

    assert lots[0].grandfathered


def test_a_post_cutoff_lot_is_not_grandfathered() -> None:
    lots, _warnings = tax.classify_lots(
        [lot("A", date(2018, 2, 1), 10, 100)], {"A": 500.0}, TODAY, NEW
    )

    assert not lots[0].grandfathered


def test_grandfathering_is_reported_as_overstating_the_gain_not_silently_applied() -> None:
    """The statutory FMV is the HIGHEST QUOTED PRICE on 31 Jan 2018, which this
    application does not hold. Using an adjusted close would be a number that
    is not the statutory one, so the limitation is stated instead."""
    _lots, warnings = tax.classify_lots(
        [lot("A", date(2017, 6, 1), 10, 100)], {"A": 500.0}, TODAY, NEW
    )

    joined = " ".join(warnings)
    assert "112A" in joined
    assert "HIGHEST QUOTED PRICE" in joined
    assert "OVERSTATED" in joined


# ---------------------------------------------------------------------------
# harvest candidates
# ---------------------------------------------------------------------------
def test_only_losing_lots_are_harvest_candidates() -> None:
    lots, _ = tax.classify_lots(
        [lot("WIN", date(2024, 12, 1), 10, 100), lot("LOSE", date(2024, 12, 1), 10, 100)],
        {"WIN": 150.0, "LOSE": 60.0},
        TODAY,
        NEW,
    )

    candidates = tax.harvest_candidates(lots, tax.GainsPosition(stcg=100_000), NEW)

    assert [c.ticker for c in candidates] == ["LOSE"]


def test_a_loss_is_priced_by_what_it_absorbs_not_by_the_rate() -> None:
    """The central correction. A 40,000 short-term loss against 40,000 of
    short-term gains saves 20% of it; the same loss with no gains saves
    nothing."""
    lots, _ = tax.classify_lots([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, TODAY, NEW)

    with_gains = tax.harvest_candidates(lots, tax.GainsPosition(stcg=40_000), NEW)[0]
    without = tax.harvest_candidates(lots, tax.GainsPosition(), NEW)[0]

    assert with_gains.loss == pytest.approx(4_000.0)
    assert with_gains.tax_saved_now == pytest.approx(4_000 * 0.20)
    assert without.tax_saved_now == 0.0
    assert without.saves_nothing_now
    assert without.carried_forward == pytest.approx(4_000.0)


def test_a_long_term_loss_against_an_exempt_gain_is_priced_at_zero() -> None:
    """Not at 12.5% of the loss. The exemption means it absorbs a gain that
    was never taxable."""
    lots, _ = tax.classify_lots([lot("A", date(2022, 1, 1), 100, 100)], {"A": 60.0}, TODAY, NEW)

    candidate = tax.harvest_candidates(lots, tax.GainsPosition(ltcg=100_000), NEW)[0]

    assert candidate.character == tax.LONG_TERM
    assert candidate.tax_saved_now == 0.0


def test_candidates_are_ranked_by_tax_saved() -> None:
    lots, _ = tax.classify_lots(
        [
            lot("SMALL", date(2024, 12, 1), 10, 100),
            lot("BIG", date(2024, 12, 1), 100, 100),
        ],
        {"SMALL": 90.0, "BIG": 50.0},
        TODAY,
        NEW,
    )

    candidates = tax.harvest_candidates(lots, tax.GainsPosition(stcg=500_000), NEW)

    assert [c.ticker for c in candidates] == ["BIG", "SMALL"]
    assert candidates[0].tax_saved_now >= candidates[1].tax_saved_now


def test_the_ranking_is_deterministic() -> None:
    """The same portfolio must always render the same list."""
    lots, _ = tax.classify_lots(
        [lot("A", date(2024, 12, 1), 10, 100), lot("B", date(2024, 12, 1), 10, 100)],
        {"A": 90.0, "B": 90.0},
        TODAY,
        NEW,
    )
    position = tax.GainsPosition(stcg=100_000)

    first = [c.ticker for c in tax.harvest_candidates(lots, position, NEW)]
    second = [c.ticker for c in tax.harvest_candidates(lots, position, NEW)]

    assert first == second


def test_the_effective_rate_is_tax_saved_per_rupee_of_loss() -> None:
    lots, _ = tax.classify_lots([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, TODAY, NEW)

    candidate = tax.harvest_candidates(lots, tax.GainsPosition(stcg=500_000), NEW)[0]

    assert candidate.effective_rate == pytest.approx(0.20)


def test_the_effective_rate_of_a_worthless_harvest_is_zero_not_undefined() -> None:
    lots, _ = tax.classify_lots([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, TODAY, NEW)

    assert tax.harvest_candidates(lots, tax.GainsPosition(), NEW)[0].effective_rate == 0.0


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_the_plan_selects_every_candidate_that_saves_tax_by_default() -> None:
    plan = tax.harvest_plan(
        [lot("A", date(2024, 12, 1), 100, 100), lot("B", date(2024, 12, 1), 100, 100)],
        {"A": 60.0, "B": 150.0},
        [closed("X", date(2024, 5, 1), date(2024, 9, 1), 100, 100, 600)],
        as_of=TODAY,
    )

    assert plan.ok
    assert [c.ticker for c in plan.selected] == ["A"], "B is a gain, not a candidate"
    assert plan.tax_saved > 0


def test_the_plan_reduces_the_tax_bill_by_the_reported_amount() -> None:
    """before.tax minus after.tax must equal what the page claims."""
    plan = tax.harvest_plan(
        [lot("A", date(2024, 12, 1), 100, 100)],
        {"A": 60.0},
        [closed("X", date(2024, 5, 1), date(2024, 9, 1), 100, 100, 600)],
        as_of=TODAY,
    )

    assert plan.tax_saved == pytest.approx(plan.before.tax - plan.after.tax)
    assert plan.tax_saved == pytest.approx(4_000 * 0.20)


def test_an_explicit_selection_is_honoured() -> None:
    plan = tax.harvest_plan(
        [lot("A", date(2024, 12, 1), 100, 100), lot("B", date(2024, 11, 1), 100, 100)],
        {"A": 60.0, "B": 70.0},
        [closed("X", date(2024, 5, 1), date(2024, 9, 1), 100, 100, 600)],
        as_of=TODAY,
        select=["B@2024-11-01"],
    )

    assert [c.ticker for c in plan.selected] == ["B"]


def test_an_empty_selection_harvests_nothing() -> None:
    plan = tax.harvest_plan(
        [lot("A", date(2024, 12, 1), 100, 100)],
        {"A": 60.0},
        [closed("X", date(2024, 5, 1), date(2024, 9, 1), 100, 100, 600)],
        as_of=TODAY,
        select=[],
    )

    assert plan.selected == ()
    assert plan.tax_saved == 0.0
    assert plan.after.tax == plan.before.tax


def test_the_plan_warns_when_there_are_no_gains_to_absorb_the_loss() -> None:
    """Reporting a saving that does not exist would be a fabrication."""
    plan = tax.harvest_plan([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, [], as_of=TODAY)

    assert plan.tax_saved == 0.0
    joined = " ".join(plan.notes)
    assert "saves nothing THIS year" in joined
    assert "carry-forward" in joined


def test_the_plan_always_states_the_wash_sale_position() -> None:
    """India has no general wash-sale rule, which is a real advantage — and
    narrower than 'anything goes'."""
    plan = tax.harvest_plan([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, [], as_of=TODAY)

    joined = " ".join(plan.notes)
    assert "no general wash-sale rule" in joined
    assert "94(7)" in joined


def test_the_plan_warns_when_the_harvest_would_waste_a_loss() -> None:
    plan = tax.harvest_plan(
        [lot("A", date(2022, 1, 1), 100, 100)],
        {"A": 60.0},
        [closed("X", date(2020, 1, 1), date(2024, 9, 1), 100, 100, 600)],
        as_of=TODAY,
        select=["A@2022-01-01"],
    )

    assert plan.after.has_waste
    assert any("compulsory, not elective" in note for note in plan.notes)


def test_the_plan_records_the_regime_and_year_it_used() -> None:
    plan = tax.harvest_plan([lot("A", date(2024, 12, 1), 10, 100)], {"A": 90.0}, [], as_of=TODAY)

    assert plan.year_label == "FY2024-25"
    assert plan.regime.label == NEW.label
    assert plan.as_of == TODAY


def test_a_plan_with_no_lots_is_not_ok() -> None:
    plan = tax.harvest_plan(None, None, None, as_of=TODAY)

    assert not plan.ok
    assert plan.candidates == ()
    assert plan.tax_saved == 0.0


def test_the_candidates_frame_has_a_stable_schema() -> None:
    plan = tax.harvest_plan([lot("A", date(2024, 12, 1), 100, 100)], {"A": 60.0}, [], as_of=TODAY)
    frame = plan.candidates_frame()

    assert "tax_saved_now" in frame.columns
    assert "carried_forward" in frame.columns
    assert "days_to_long_term" in frame.columns
    assert len(frame) == 1


def test_the_candidates_frame_is_empty_but_typed_when_nothing_qualifies() -> None:
    plan = tax.harvest_plan([lot("A", date(2024, 12, 1), 10, 100)], {"A": 150.0}, [], as_of=TODAY)
    frame = plan.candidates_frame()

    assert frame.empty
    assert "tax_saved_now" in frame.columns


def test_the_lots_frame_carries_gains_as_well_as_losses() -> None:
    plan = tax.harvest_plan(
        [lot("WIN", date(2024, 12, 1), 10, 100), lot("LOSE", date(2024, 12, 1), 10, 100)],
        {"WIN": 150.0, "LOSE": 60.0},
        [],
        as_of=TODAY,
    )

    assert len(plan.lots_frame()) == 2
    assert len(plan.candidates_frame()) == 1


# ---------------------------------------------------------------------------
# the boundary alerts — the inversion
# ---------------------------------------------------------------------------
def test_a_gain_near_the_boundary_is_told_to_wait() -> None:
    """Crossing it moves the gain from 20% to 12.5%."""
    lots, _ = tax.classify_lots([lot("A", date(2024, 4, 1), 10, 100)], {"A": 150.0}, TODAY, NEW)
    assert lots[0].days_to_boundary(NEW) <= 45

    alerts = tax.boundary_alerts(lots, NEW)

    assert len(alerts) == 1
    assert "waiting" in alerts["action"].iloc[0].lower()


def test_a_loss_near_the_boundary_is_told_to_harvest_first() -> None:
    """The inversion, and the feature's most valuable insight: a short-term
    loss offsets a wider set of gains at a higher rate, so holding it for
    'long-term treatment' makes it LESS useful."""
    lots, _ = tax.classify_lots([lot("A", date(2024, 4, 1), 10, 100)], {"A": 60.0}, TODAY, NEW)

    alerts = tax.boundary_alerts(lots, NEW)

    assert len(alerts) == 1
    assert "BEFORE" in alerts["action"].iloc[0]
    assert "less useful" in alerts["reason"].iloc[0]


def test_the_two_alert_directions_are_genuinely_opposite() -> None:
    lots, _ = tax.classify_lots(
        [lot("WIN", date(2024, 4, 1), 10, 100), lot("LOSE", date(2024, 4, 1), 10, 100)],
        {"WIN": 150.0, "LOSE": 60.0},
        TODAY,
        NEW,
    )

    alerts = tax.boundary_alerts(lots, NEW).set_index("ticker")

    assert "waiting" in alerts.loc["WIN", "action"].lower()
    assert "BEFORE" in alerts.loc["LOSE", "action"]


def test_a_lot_already_long_term_raises_no_alert() -> None:
    lots, _ = tax.classify_lots([lot("A", date(2020, 1, 1), 10, 100)], {"A": 60.0}, TODAY, NEW)

    assert tax.boundary_alerts(lots, NEW).empty


def test_a_lot_far_from_the_boundary_raises_no_alert() -> None:
    lots, _ = tax.classify_lots([lot("A", date(2025, 2, 1), 10, 100)], {"A": 60.0}, TODAY, NEW)

    assert tax.boundary_alerts(lots, NEW, window_days=45).empty


def test_alerts_are_sorted_most_urgent_first() -> None:
    lots, _ = tax.classify_lots(
        [lot("SOON", date(2024, 3, 15), 10, 100), lot("LATER", date(2024, 4, 20), 10, 100)],
        {"SOON": 60.0, "LATER": 60.0},
        TODAY,
        NEW,
    )

    alerts = tax.boundary_alerts(lots, NEW, window_days=90)

    assert list(alerts["days_to_long_term"]) == sorted(alerts["days_to_long_term"])


def test_boundary_alerts_of_nothing_is_an_empty_typed_frame() -> None:
    frame = tax.boundary_alerts([], NEW)

    assert frame.empty
    assert "action" in frame.columns
    assert "reason" in frame.columns


# ---------------------------------------------------------------------------
# end to end, through the real lots engine
# ---------------------------------------------------------------------------
def test_the_plan_runs_on_lots_from_the_fifo_engine() -> None:
    """The integration that matters: the harvesting analysis consumes exactly
    what src.analytics.lots produces, with no adapter."""
    from src.analytics import lots as lots_module

    ledger = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "A"],
            "trade_date": ["2022-01-10", "2024-12-01", "2024-11-15", "2024-09-20"],
            "action": ["BUY", "BUY", "BUY", "SELL"],
            "quantity": [100, 50, 80, 40],
            "price": [100.0, 120.0, 200.0, 180.0],
            "fees": [10.0, 6.0, 16.0, 7.2],
        }
    )
    parsed, _problems = lots_module.parse_transactions(ledger)
    open_lots, realised, _issues = lots_module.build_lots(parsed)

    plan = tax.harvest_plan(open_lots, {"A": 90.0, "B": 150.0}, realised, as_of=TODAY)

    assert plan.ok
    assert plan.position.n_trades >= 1
    assert plan.candidates, "B is at a loss and should be harvestable"
    assert plan.tax_saved >= 0.0
    # Every candidate must trace to a real open lot.
    keys = {f"{item.ticker}@{item.open_date.isoformat()}" for item in plan.lots}
    for candidate in plan.candidates:
        assert f"{candidate.ticker}@{candidate.lot.open_date.isoformat()}" in keys
