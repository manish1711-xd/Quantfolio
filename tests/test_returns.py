"""Verification of ``src.analytics.returns`` — the performance engine.

This module answers the two questions a portfolio owner actually asks, and the
whole point of the file is that it answers them *separately*:

* **How did the assets do?** — time-weighted return. Contributions and
  withdrawals are neutralised, so the number describes the manager.
* **How did I do?** — money-weighted return (XIRR). The actual dated cash
  flows drive it, so a big contribution made at the top genuinely hurts.

Getting either one subtly wrong is invisible on a chart, which is why the tests
below are arithmetic-first: wherever a golden value appears, the hand
computation that produced it appears in a comment next to the assertion, so a
reviewer can audit the numbers without running anything.

Three kinds of assertion, deliberately:

1. **Hand-computed goldens.** ``xirr([-1000 today, +1100 in 365 days]) == 0.10``
   because ``-1000 + 1100/1.1 == 0``. A test that re-derives the expectation
   through the same code path proves nothing.
2. **Mathematical identities**, which cannot rot: ``npv(xirr(f), f) == 0``;
   ``(1 + cumulative)**(252/(n-1)) == 1 + annualized``; the compounded monthly
   returns of a row equal that row's ``Year`` cell; the product of every
   periodic return equals the total return at *any* bucketing frequency; the
   answer is invariant to the solver's starting guess.
3. **Documented degradation.** The module contract is that nothing raises,
   nothing loops without a bound and nothing substitutes zero for "undefined".
   Empty, single-point, all-NaN, flat, unsorted, duplicate-index, no-sign-
   change, single-flow and all-same-date inputs each get a test pinning the
   NaN / empty answer rather than an exception.

Sign and unit conventions are treated as regression guards in their own right,
because inverting one silently inverts a whole performance report:

* ``npv`` / ``xirr`` / ``portfolio_xirr`` sign flows from the **investor's**
  pocket — money paid in is NEGATIVE;
* ``time_weighted_return`` / ``modified_dietz`` sign them from the
  **portfolio's** perspective — a contribution in is POSITIVE.

Feeding a schedule to the wrong one does not raise. It returns a plausible,
wrong number — 120% instead of 10% in the worked example below — so the
distinction is asserted explicitly rather than left to the docstrings.

``cumulative_return`` is a DECIMAL, ``monthly_return_table`` and the
``*_pct`` keys of ``return_stats`` are PERCENT; both are pinned here too.

Deterministic and offline: the only randomness is the seeded shared fixtures.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analytics.lots import cash_flows
from src.analytics.returns import (
    DAY_COUNT,
    MONTH_LABELS,
    RATE_LOWER,
    RATE_UPPER,
    TRADING_DAYS,
    annualized_return,
    cagr,
    cumulative_return,
    modified_dietz,
    monthly_return_table,
    npv,
    periodic_returns,
    portfolio_xirr,
    return_stats,
    rolling_returns,
    time_weighted_return,
    xirr,
)
from src.models import Transaction

# ---------------------------------------------------------------------------
# local, deterministic scaffolding
# ---------------------------------------------------------------------------
#: 2021 and 2022 both have 365 days, so ``day(365)`` and ``day(730)`` are
#: exactly one and two ACT/365 years after the origin with no leap-day fudge.
#: Every XIRR golden below depends on that, so do not move this date.
D0 = date(2021, 1, 1)


def day(offset: int) -> date:
    """``offset`` calendar days after :data:`D0`."""
    return D0 + timedelta(days=offset)


#: The ten keys :func:`return_stats` documents. Frozen here so that adding or
#: renaming one has to be a deliberate act with a matching test edit — the
#: report template and the Excel workbook both index this dict by name.
RETURN_STAT_KEYS = frozenset(
    {
        "cumulative_pct",
        "annualized_pct",
        "best_month_pct",
        "worst_month_pct",
        "positive_months",
        "total_months",
        "best_day_pct",
        "worst_day_pct",
        "longest_win_streak",
        "longest_loss_streak",
    }
)

# One economic story, told twice — used by the sign-convention section.
#   2024-01-01  the account is worth 1,000
#   2024-07-01  the investor pays IN another 500      (182 days in of 365)
#   2024-12-31  the account is worth 1,650
STORY_START = date(2024, 1, 1)
STORY_MID = date(2024, 7, 1)
STORY_END = date(2024, 12, 31)


@pytest.fixture
def sip_flows() -> list[tuple[date, float]]:
    """A twelve-instalment SIP: 10,000 a month, worth 130,000 after a year.

    Twelve payments 30 days apart from :data:`D0` (120,000 paid in) and a
    single 130,000 redemption on day 365. Deliberately irregular against the
    calendar — 30-day spacing is not a calendar month — because that is
    exactly the schedule a closed-form annualisation cannot handle and XIRR
    can.
    """
    return [*((day(30 * i), -10_000.0) for i in range(12)), (day(365), 130_000.0)]


@pytest.fixture
def quarterly_series() -> pd.Series:
    """Four month-end valuations whose monthly returns are +10%, -10%, +50%.

    100 -> 110 -> 99 -> 148.5, so the compounded total is
    ``1.10 * 0.90 * 1.50 = 1.485`` — i.e. +48.5% — and every monthly figure in
    the calendar table can be checked by eye.
    """
    return pd.Series(
        [100.0, 110.0, 99.0, 148.5],
        index=pd.DatetimeIndex(["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-28"]),
    )


@pytest.fixture
def contribution_account() -> tuple[pd.Series, pd.Series]:
    """253 daily valuations containing one contribution and one real gain.

    positions 0..125    V = 1000     nothing happens at all
    position  126       V = 1500     500 paid IN at the end of that day
    positions 127..252  V = 1650     the assets then gained 10% on 1500

    253 points is exactly 252 return intervals, so the annualisation exponent
    ``TRADING_DAYS / (n - 1)`` is 1 and every number below is a raw, unscaled
    return that can be read straight off the series.
    """
    index = pd.bdate_range("2024-01-01", periods=253)
    values = np.full(253, 1000.0)
    values[126] = 1500.0
    values[127:] = 1650.0
    flows = pd.Series([500.0], index=[index[126]])
    return pd.Series(values, index=index), flows


# ---------------------------------------------------------------------------
# npv — the discounting kernel every money-weighted number is built on
# ---------------------------------------------------------------------------
def test_npv_discounts_a_single_year_flow_by_exactly_one_plus_the_rate():
    # -1000 at t=0 and +1100 at t=365/365 = 1 year:
    #   NPV(0.10) = -1000 + 1100 / 1.10 = -1000 + 1000 = 0
    assert npv(0.10, [(D0, -1000.0), (day(365), 1100.0)]) == pytest.approx(0.0, abs=1e-9)


def test_npv_at_a_zero_rate_is_the_plain_sum_of_the_flows():
    # (1 + 0) ** t == 1 for every t, so no flow is discounted at all.
    flows = [(D0, -1000.0), (day(365), -1000.0), (day(730), 2310.0)]
    assert npv(0.0, flows) == pytest.approx(310.0, abs=1e-9)


def test_npv_is_positive_below_the_irr_and_negative_above_it():
    # SIGN GUARD. Flows are signed from the INVESTOR's pocket, so a project
    # that beats the discount rate has a POSITIVE present value:
    #   NPV(0.05) = -1000 + 1100/1.05 = -1000 + 1047.619048 = +47.619048
    #   NPV(0.20) = -1000 + 1100/1.20 = -1000 +  916.666667 = -83.333333
    flows = [(D0, -1000.0), (day(365), 1100.0)]
    assert npv(0.05, flows) == pytest.approx(1100.0 / 1.05 - 1000.0, rel=1e-12)
    assert npv(0.05, flows) == pytest.approx(47.61904762, abs=1e-7)
    assert npv(0.20, flows) == pytest.approx(-83.33333333, abs=1e-7)


def test_npv_is_stated_as_of_the_first_flow_so_shifting_the_calendar_changes_nothing():
    # Identity: NPV depends only on the day OFFSETS from d_0, never on where
    # d_0 sits in the calendar. Moving the whole schedule 100 days later must
    # not move the answer by a paisa.
    flows = [(D0, -1000.0), (day(365), -1000.0), (day(730), 2310.0)]
    shifted = [(when + timedelta(days=100), amount) for when, amount in flows]
    assert npv(0.07, shifted) == pytest.approx(npv(0.07, flows), rel=1e-12)


def test_npv_uses_act_365_calendar_days_not_trading_days():
    # REGRESSION GUARD on the documented day count. A flow 730 calendar days
    # out is discounted by (1+r)**2, not (1+r)**(730/252).
    assert DAY_COUNT == 365.0
    flows = [(D0, -1000.0), (day(730), 1210.0)]
    assert npv(0.10, flows) == pytest.approx(-1000.0 + 1210.0 / 1.21, abs=1e-9)


def test_npv_accepts_a_date_indexed_series_as_well_as_tuples():
    pairs = [(D0, -1000.0), (day(365), -1000.0), (day(730), 2310.0)]
    series = pd.Series([-1000.0, -1000.0, 2310.0], index=pd.DatetimeIndex([d for d, _ in pairs]))
    assert npv(0.10, series) == pytest.approx(npv(0.10, pairs), abs=1e-9)


def test_npv_sorts_the_schedule_before_choosing_the_origin():
    # A CSV in descending date order must not make the LAST flow the origin.
    forward = [(D0, -1000.0), (day(365), 1100.0)]
    reversed_rows = list(reversed(forward))
    assert npv(0.03, reversed_rows) == pytest.approx(npv(0.03, forward), rel=1e-12)


def test_npv_ignores_rows_with_an_unusable_date_or_amount():
    flows = [(D0, -1000.0), ("not-a-date", 999.0), (day(365), float("nan")), (day(365), 1100.0)]
    assert npv(0.10, flows) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("rate", [-1.0, -1.5, float("nan"), float("inf")])
def test_npv_is_nan_where_the_discount_factor_is_undefined(rate):
    # (1 + r) ** t has no real value for a fractional t once 1 + r <= 0.
    assert math.isnan(npv(rate, [(D0, -1000.0), (day(365), 1100.0)]))


def test_npv_of_an_empty_schedule_is_nan_not_zero():
    # "No flows" is unmeasurable, not "worth nothing".
    assert math.isnan(npv(0.10, []))
    assert math.isnan(npv(0.10, None))


# ---------------------------------------------------------------------------
# xirr — the money-weighted centrepiece
# ---------------------------------------------------------------------------
def test_xirr_of_minus_1000_today_and_plus_1100_in_a_year_is_ten_percent():
    # THE canonical case. 1100/1000 = 1.10 over exactly one ACT/365 year.
    assert xirr([(D0, -1000.0), (day(365), 1100.0)]) == pytest.approx(0.10, abs=1e-6)


def test_xirr_of_a_loss_making_year_is_negative_ten_percent():
    # 900/1000 = 0.90 over one year, so the annualised rate is -0.10 exactly.
    assert xirr([(D0, -1000.0), (day(365), 900.0)]) == pytest.approx(-0.10, abs=1e-6)


def test_xirr_of_money_that_merely_came_back_is_zero():
    # -5000 out, +5000 back a year later: the investor earned nothing at all.
    # Not NaN, not a rounding artefact — a genuine, measurable 0%.
    assert xirr([(D0, -5000.0), (day(365), 5000.0)]) == pytest.approx(0.0, abs=1e-9)


def test_xirr_annualises_a_doubling_over_730_days_to_the_square_root_of_two():
    # r = 2 ** (365/730) - 1 = sqrt(2) - 1 = 0.41421356237309515 .
    # Written to full double precision on purpose: 0.4142 would fail a 1e-9
    # comparison and look like an implementation defect.
    expected = 2.0 ** (365.0 / 730.0) - 1.0
    assert expected == pytest.approx(0.41421356237309515, abs=1e-15)
    assert xirr([(D0, -1000.0), (day(730), 2000.0)]) == pytest.approx(expected, abs=1e-9)


def test_xirr_of_a_three_flow_schedule_hits_a_hand_checkable_ten_percent():
    # -1000 at t=0, -1000 at t=1y, +2310 at t=2y. At r = 0.10:
    #   -1000 - 1000/1.10 + 2310/1.21 = -1000 - 909.090909 + 1909.090909 = 0
    flows = [(D0, -1000.0), (day(365), -1000.0), (day(730), 2310.0)]
    assert xirr(flows) == pytest.approx(0.10, abs=1e-9)


def test_a_monthly_sip_schedule_satisfies_the_npv_identity(sip_flows):
    # THE defining identity: whatever rate comes back, discounting the real
    # flows at it must zero the present value. This is what makes XIRR a rate
    # of return rather than a plausible-looking number.
    rate = xirr(sip_flows)
    assert math.isfinite(rate)
    scale = max(abs(amount) for _, amount in sip_flows)
    assert abs(npv(rate, sip_flows)) <= 1e-6 * scale


def test_a_monthly_sip_earns_a_rate_between_its_simple_and_lump_sum_returns(sip_flows):
    # 12 monthly instalments of 10,000 (120,000 paid in) worth 130,000 after a
    # year. The simple return on the total paid in is 130/120 - 1 = 8.33%; a
    # single lump sum on day one would have had to earn that same 8.33%. Money
    # invested for an average of ~6 months must have worked HARDER than that,
    # so the money-weighted rate has to sit above it — and below the 30% that
    # the FINAL instalment alone appears to have made.
    rate = xirr(sip_flows)
    assert 130_000.0 / 120_000.0 - 1.0 < rate < 0.30


@pytest.mark.parametrize("guess", [-0.5, 0.0, 5.0, 1e9])
def test_xirr_is_invariant_to_the_starting_guess(guess, sip_flows):
    # A root finder whose answer depends on where it started is not reporting
    # a property of the cash flows. 1e9 is deliberately outside the search
    # domain (RATE_UPPER = 100) and must be clamped, not obeyed.
    assert RATE_LOWER < 0.0 < RATE_UPPER < 1e9
    assert xirr(sip_flows, guess) == pytest.approx(xirr(sip_flows, 0.1), abs=1e-6)


@pytest.mark.parametrize("guess", [-0.5, 0.0, 5.0, 1e9])
def test_the_guess_cannot_move_a_hand_computed_ten_percent(guess):
    assert xirr([(D0, -1000.0), (day(365), 1100.0)], guess) == pytest.approx(0.10, abs=1e-6)


def test_xirr_returns_a_verified_root_when_the_schedule_has_two_of_them():
    # Non-conventional flows: -1000, +2500, -1540 one year apart. Writing
    # x = 1/(1+r), the NPV is -1000 + 2500x - 1540x**2 = 0, i.e.
    #   1540x**2 - 2500x + 1000 = 0
    #   x = (2500 +/- sqrt(6_250_000 - 6_160_000)) / 3080 = (2500 +/- 300)/3080
    #   x = 0.909090... -> r = 0.10   |   x = 0.714285... -> r = 0.40
    # Descartes' rule allows both. The contract is not "pick the nice one" but
    # "return something that genuinely zeroes the NPV", so that is what is
    # asserted — plus the substitution check the docstring promises.
    flows = [(D0, -1000.0), (day(365), 2500.0), (day(730), -1540.0)]
    rate = xirr(flows)
    assert rate == pytest.approx(0.10, abs=1e-6) or rate == pytest.approx(0.40, abs=1e-6)
    assert abs(npv(rate, flows)) <= 1e-6 * 2500.0


def test_xirr_accepts_a_date_indexed_series_of_flows():
    dates = pd.DatetimeIndex([D0, day(365)])
    assert xirr(pd.Series([-1000.0, 1100.0], index=dates)) == pytest.approx(0.10, abs=1e-6)


def test_xirr_ignores_the_row_order_of_the_schedule():
    forward = [(D0, -1000.0), (day(365), -1000.0), (day(730), 2310.0)]
    assert xirr(list(reversed(forward))) == pytest.approx(xirr(forward), abs=1e-9)


# Regression guard for a fixed defect:
# xirr short-circuits at `guess` whenever |npv(guess)| falls inside a tolerance that
# scales with the LARGEST flow (_NPV_RTOL * scale), so on a 1e9-sized schedule guess=0.1
# returns 0.1 while guess=0.0 solves to the true 0.10000110 — a 1.1e-6 guess-dependent
# error.
def test_xirr_is_guess_invariant_even_for_institution_sized_cash_flows():
    # -1e9 out, +1.1e9 * 1.000001 back one year later, so the exact rate is
    #   1.1 * 1.000001 - 1 = 0.10000110000000006
    # The absolute NPV tolerance at this scale is 1e-6 * 1.1e9 = 1100, and the
    # NPV slope near the root is about -9.1e8 per unit of rate, so an answer
    # accepted purely on |NPV| can be a full 1.2e-6 away from the truth.
    flows = [(D0, -1e9), (day(365), 1.1e9 * 1.000001)]
    exact = 1.1 * 1.000001 - 1.0
    assert xirr(flows, 0.0) == pytest.approx(exact, abs=1e-9)
    assert xirr(flows, 0.1) == pytest.approx(exact, abs=1e-9)


# ---- xirr: documented degradation ----------------------------------------
def test_xirr_of_an_empty_schedule_is_nan():
    assert math.isnan(xirr([]))
    assert math.isnan(xirr(None))


def test_xirr_of_a_single_flow_is_nan():
    # One dated amount is a balance, not a return.
    assert math.isnan(xirr([(D0, -1000.0)]))


@pytest.mark.parametrize(
    "flows",
    [
        pytest.param([(D0, -1000.0), (day(365), -500.0)], id="all-outflows"),
        pytest.param([(D0, 1000.0), (day(365), 500.0)], id="all-inflows"),
        pytest.param([(D0, -1000.0), (day(365), 0.0)], id="zero-terminal-flow"),
    ],
)
def test_xirr_is_nan_without_a_sign_change(flows):
    # You cannot earn a return on money that was never put in (or never came
    # back). NaN says "undefined"; a zero here would read as "broke even".
    assert math.isnan(xirr(flows))


def test_xirr_is_nan_when_every_flow_falls_on_the_same_day():
    # Zero elapsed time: (1 + r) ** 0 == 1 for every r, so the NPV is constant
    # and there is nothing to solve. The bounded search must say so, not spin.
    assert math.isnan(xirr([(D0, -1000.0), (D0, 1100.0), (D0, -50.0)]))


def test_xirr_is_nan_for_a_wipe_out_below_the_search_domain():
    # -1000 in, 0.0001 back: the true rate is about -99.99999%, below
    # RATE_LOWER = -0.999 where the fractional-power discount factor stops
    # being meaningful. The module reports "no verified root", not a guess.
    assert RATE_LOWER == -0.999
    assert math.isnan(xirr([(D0, -1000.0), (day(365), 0.0001)]))


def test_xirr_survives_an_adversarial_alternating_schedule_without_hanging():
    # 40 flows alternating -1000 / +1000 every 30 days: many sign changes, so
    # many candidate roots. Both search strategies are finite by construction,
    # so this must return promptly with either a verified root or NaN.
    flows = [(day(30 * i), (-1.0) ** i * 1000.0) for i in range(40)]
    result = xirr(flows)
    assert math.isnan(result) or abs(npv(result, flows)) <= 1e-6 * 1000.0


# ---------------------------------------------------------------------------
# portfolio_xirr — the whole book, including the still-open position
# ---------------------------------------------------------------------------
def test_portfolio_xirr_accepts_transaction_objects():
    # BUY 10 @ 100 with no fees -> cash_flow = -(10*100) = -1000 out of pocket;
    # worth 1100 a year later -> 1100/1000 - 1 = 0.10 .
    buy = Transaction("AAA", D0, "BUY", 10, 100.0)
    assert buy.cash_flow == pytest.approx(-1000.0, abs=1e-12)
    assert portfolio_xirr([buy], 1100.0, day(365)) == pytest.approx(0.10, abs=1e-6)


def test_portfolio_xirr_accepts_plain_date_amount_pairs():
    assert portfolio_xirr([(D0, -1000.0)], 1100.0, day(365)) == pytest.approx(0.10, abs=1e-6)


def test_portfolio_xirr_reads_transactions_and_pairs_identically(
    transactions: list[Transaction],
):
    # ``lots.cash_flows`` turns the same ledger into (date, amount) pairs, so
    # the two entry points must agree to the last bit — one of them feeds the
    # tearsheet and the other feeds the Excel workbook.
    from_objects = portfolio_xirr(transactions, 2000.0, date(2025, 1, 1))
    from_pairs = portfolio_xirr(cash_flows(transactions), 2000.0, date(2025, 1, 1))
    assert math.isfinite(from_objects)
    assert from_objects == pytest.approx(from_pairs, rel=1e-12)


def test_portfolio_xirr_appends_the_current_value_as_a_terminal_inflow():
    # SIGN GUARD. The "as if liquidated today" flow is money coming BACK, so
    # it must be positive. Were it appended negative there would be no sign
    # change at all and the answer would collapse to NaN.
    buys = [(D0, -1000.0), (day(180), -1000.0)]
    assert math.isnan(xirr(buys))  # premise
    assert portfolio_xirr(buys, 2500.0, day(365)) > 0.0


def test_portfolio_xirr_matches_the_explicit_schedule_it_builds():
    buys = [(D0, -1000.0), (day(180), -1000.0)]
    explicit = xirr([*buys, (day(365), 2500.0)])
    assert portfolio_xirr(buys, 2500.0, day(365)) == pytest.approx(explicit, rel=1e-12)


def test_portfolio_xirr_of_a_profitable_ledger_is_positive(
    transactions: list[Transaction],
):
    # Fixture ledger: -1010, -1212, -707 paid in, +2235 realised, plus 2,000
    # still on the books = 4,225 back against 2,929 paid in.
    assert portfolio_xirr(transactions, 2000.0, date(2025, 1, 1)) > 0.0


def test_portfolio_xirr_omits_a_non_positive_current_value():
    # Nothing left to liquidate, so no terminal flow is appended; a lone BUY
    # then has no sign change and the answer is NaN rather than an invention.
    buy = Transaction("AAA", D0, "BUY", 10, 100.0)
    assert math.isnan(portfolio_xirr([buy], 0.0, day(365)))
    assert math.isnan(portfolio_xirr([buy], -500.0, day(365)))


@pytest.mark.parametrize(
    ("txns", "value", "as_of"),
    [
        pytest.param([], 1100.0, day(365), id="no-transactions"),
        pytest.param(None, 1100.0, day(365), id="none-transactions"),
        pytest.param([(D0, -1000.0)], 1100.0, "not-a-date", id="unparseable-as-of"),
        pytest.param([(D0, -1000.0)], float("nan"), day(365), id="non-finite-value"),
    ],
)
def test_portfolio_xirr_degrades_to_nan_rather_than_raising(txns, value, as_of):
    assert math.isnan(portfolio_xirr(txns, value, as_of))


# ---------------------------------------------------------------------------
# modified_dietz
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("begin", "end", "expected"),
    [
        (1000.0, 1100.0, 0.10),  # 1100/1000 - 1
        (1000.0, 900.0, -0.10),  #  900/1000 - 1
        (2500.0, 2500.0, 0.0),  # untouched and unchanged
        (400.0, 1000.0, 1.50),  # 1000/400 - 1
    ],
)
def test_modified_dietz_with_no_flows_is_simply_ev_over_bv_minus_one(begin, end, expected):
    # With F = 0 the formula collapses to (EV - BV)/BV, so the money-weighted
    # and time-weighted answers must coincide on an untouched account.
    result = modified_dietz(begin, end, [], date(2024, 1, 1), date(2024, 12, 31))
    assert result == pytest.approx(expected, abs=1e-12)


def test_modified_dietz_weights_a_mid_period_contribution_by_its_time_invested():
    # BV 1000 on 1 Jan, +500 on 1 Jul (182 days into a 365-day period), EV 1650.
    #   w   = (365 - 182)/365 = 183/365 = 0.501369863
    #   MD  = (1650 - 1000 - 500) / (1000 + 500 * 183/365)
    #       = 150 / 1250.684931506849 = 0.11993428258488499
    result = modified_dietz(1000.0, 1650.0, [(STORY_MID, 500.0)], STORY_START, STORY_END)
    assert (STORY_END - STORY_START).days == 365
    assert (STORY_MID - STORY_START).days == 182
    assert result == pytest.approx(150.0 / (1000.0 + 500.0 * 183.0 / 365.0), rel=1e-12)
    assert result == pytest.approx(0.11993428, abs=1e-8)


def test_modified_dietz_gives_a_day_one_contribution_full_weight():
    # A flow on the OPENING day was invested for the whole period, w = 1:
    #   MD = (2200 - 1000 - 1000) / (1000 + 1000 * 1) = 200 / 2000 = 0.10
    # Dividing by BV alone would report 0.20 — double the real return.
    result = modified_dietz(
        1000.0, 2200.0, [(date(2024, 1, 1), 1000.0)], date(2024, 1, 1), date(2024, 1, 11)
    )
    assert result == pytest.approx(0.10, abs=1e-12)


def test_modified_dietz_gives_a_final_day_contribution_no_weight():
    # A flow on the CLOSING day earned nothing and must not dilute the base:
    #   MD = (2100 - 1000 - 1000) / (1000 + 1000 * 0) = 100 / 1000 = 0.10
    result = modified_dietz(
        1000.0, 2100.0, [(date(2024, 1, 11), 1000.0)], date(2024, 1, 1), date(2024, 1, 11)
    )
    assert result == pytest.approx(0.10, abs=1e-12)


def test_modified_dietz_ignores_flows_dated_outside_the_period():
    # A 100,000 contribution a year before the window would otherwise swamp
    # both the numerator and the denominator: (1200-1000)/1000 = 0.20 .
    result = modified_dietz(
        1000.0, 1200.0, [(date(2022, 1, 1), 100_000.0)], date(2024, 1, 1), date(2024, 1, 11)
    )
    assert result == pytest.approx(0.20, abs=1e-12)


def test_modified_dietz_approximates_the_xirr_of_the_same_story():
    # MD is the closed-form, no-root-finding cousin of XIRR. On a single
    # mid-period flow the two should land within a few tens of basis points:
    #   MD   = 0.119934...  XIRR = 0.120620...
    md = modified_dietz(1000.0, 1650.0, [(STORY_MID, 500.0)], STORY_START, STORY_END)
    mwr = xirr([(STORY_START, -1000.0), (STORY_MID, -500.0), (STORY_END, 1650.0)])
    assert mwr == pytest.approx(0.12062058, abs=1e-8)
    assert abs(mwr - md) < 0.005


@pytest.mark.parametrize(
    ("begin", "end", "flows", "start", "end_date", "why"),
    [
        (1000.0, 1100.0, [], date(2024, 1, 1), date(2024, 1, 1), "zero-length period"),
        (1000.0, 1100.0, [], date(2024, 12, 31), date(2024, 1, 1), "reversed dates"),
        (1000.0, 1100.0, [], "nope", date(2024, 1, 11), "unparseable start"),
        (float("nan"), 1100.0, [], date(2024, 1, 1), date(2024, 1, 11), "non-finite BV"),
        (1000.0, float("inf"), [], date(2024, 1, 1), date(2024, 1, 11), "non-finite EV"),
        (0.0, 100.0, [], date(2024, 1, 1), date(2024, 2, 1), "zero average capital"),
        (
            100.0,
            50.0,
            [(date(2024, 1, 1), -200.0)],
            date(2024, 1, 1),
            date(2024, 2, 1),
            "negative average capital",
        ),
    ],
)
def test_modified_dietz_is_nan_for_an_undefined_period(begin, end, flows, start, end_date, why):
    assert math.isnan(modified_dietz(begin, end, flows, start, end_date)), why


# ---------------------------------------------------------------------------
# the two flow sign conventions — the thing that silently inverts a report
# ---------------------------------------------------------------------------
def test_the_investor_convention_signs_money_paid_in_as_negative():
    # xirr / npv / portfolio_xirr: paid in = NEGATIVE, received = POSITIVE.
    # Told the investor's way, the story solves; told the portfolio's way
    # (everything positive) there is no sign change and the answer is NaN.
    investor = [(STORY_START, -1000.0), (STORY_MID, -500.0), (STORY_END, 1650.0)]
    portfolio = [(STORY_START, 1000.0), (STORY_MID, 500.0), (STORY_END, 1650.0)]
    assert xirr(investor) == pytest.approx(0.12062058, abs=1e-8)
    assert math.isnan(xirr(portfolio))


def test_the_portfolio_convention_signs_a_contribution_as_positive():
    # modified_dietz: a contribution INTO the portfolio is POSITIVE. Feeding
    # it the investor's -500 does not raise — it silently reports 153% where
    # the truth is 12%:
    #   right: (1650 - 1000 - 500)/(1000 + 500*183/365) = 150/1250.68 = 0.1199
    #   wrong: (1650 - 1000 + 500)/(1000 - 500*183/365) = 1150/749.32 = 1.5347
    right = modified_dietz(1000.0, 1650.0, [(STORY_MID, 500.0)], STORY_START, STORY_END)
    wrong = modified_dietz(1000.0, 1650.0, [(STORY_MID, -500.0)], STORY_START, STORY_END)
    assert right == pytest.approx(0.11993428, abs=1e-8)
    assert wrong == pytest.approx(1150.0 / (1000.0 - 500.0 * 183.0 / 365.0), rel=1e-12)
    assert wrong == pytest.approx(1.53473492, abs=1e-8)
    assert wrong > 12.0 * right  # an order of magnitude, from one minus sign


def test_time_weighted_return_also_uses_the_portfolio_convention(
    contribution_account: tuple[pd.Series, pd.Series],
):
    # Same trap, different function. The account is worth 1650 after a 500
    # contribution and a 10% gain on the 1500 that resulted:
    #   right: growth at the contribution day = (1500 - 500)/1000 = 1.00
    #          then 1650/1500 = 1.10 -> TWR = 0.10
    #   wrong: (1500 + 500)/1000 = 2.00, then 1.10 -> TWR = 1.20
    values, flows = contribution_account
    assert time_weighted_return(values, flows) == pytest.approx(0.10, abs=1e-9)
    assert time_weighted_return(values, -flows) == pytest.approx(1.20, abs=1e-9)


# ---------------------------------------------------------------------------
# annualized_return, cumulative_return, cagr
# ---------------------------------------------------------------------------
def test_annualized_return_over_253_points_from_100_to_112_is_exactly_twelve_percent():
    # 253 observations are 252 return intervals, so the exponent
    # TRADING_DAYS / (n - 1) = 252/252 = 1 and one year of daily closes
    # annualises to its own total return: 112/100 - 1 = 0.12 .
    assert TRADING_DAYS == 252
    line = pd.Series(np.linspace(100.0, 112.0, 253))
    assert annualized_return(line) == pytest.approx(0.12, abs=1e-12)


def test_annualized_return_uses_intervals_not_observations():
    # REGRESSION GUARD. Over 127 points (126 intervals) the exponent is
    # 252/126 = 2, so a +12% window annualises to 1.12**2 - 1 = 0.2544 .
    # Using n instead of n-1 would give 1.12**(252/127) - 1 = 0.25276... —
    # a 17-basis-point error that nobody would ever spot by eye.
    line = pd.Series(np.linspace(100.0, 112.0, 127))
    assert annualized_return(line) == pytest.approx(1.12**2 - 1.0, rel=1e-12)
    assert annualized_return(line) == pytest.approx(0.2544, abs=1e-9)


def test_cumulative_return_is_a_decimal_not_a_percent(quarterly_series: pd.Series):
    # UNIT GUARD. 148.5/100 - 1 = 0.485, NOT 48.5. Everything downstream
    # multiplies by 100 exactly once.
    assert cumulative_return(quarterly_series) == pytest.approx(0.485, abs=1e-12)


def test_cumulative_and_annualized_return_are_two_views_of_one_number(
    portfolio_values: pd.Series,
):
    # Identity: (1 + total) ** (252 / (n - 1)) == 1 + annualised, by
    # definition. If these two ever drift apart, one of them has changed its
    # annualisation convention behind the other's back.
    n = len(portfolio_values)
    total = cumulative_return(portfolio_values)
    annual = annualized_return(portfolio_values)
    assert (1.0 + total) ** (TRADING_DAYS / (n - 1)) == pytest.approx(1.0 + annual, rel=1e-12)


def test_a_flat_series_returns_exactly_zero_not_nan(flat_series: pd.Series):
    # Zero is the honest answer here: the portfolio measurably went nowhere.
    assert cumulative_return(flat_series) == 0.0
    assert annualized_return(flat_series) == 0.0
    assert time_weighted_return(flat_series) == 0.0
    assert cagr(flat_series) == 0.0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([100.0], id="one-point"),
        pytest.param([], id="no-points"),
    ],
)
def test_annualized_return_needs_at_least_two_points(values):
    series = pd.Series(values, dtype=float)
    assert math.isnan(annualized_return(series))
    assert math.isnan(cumulative_return(series))


def test_annualized_return_is_nan_for_a_non_positive_start_value():
    # V_end / V_start is meaningless from zero, and a geometric mean of a
    # negative base is not real.
    assert math.isnan(annualized_return(pd.Series([0.0, 120.0])))
    assert math.isnan(annualized_return(pd.Series([-100.0, 120.0])))
    assert math.isnan(cumulative_return(pd.Series([0.0, 120.0])))


def test_annualized_return_is_nan_for_non_positive_growth():
    # A portfolio that reached zero (or went negative) has no geometric mean:
    # (-0.5) ** (252/1) is not a real number.
    assert math.isnan(annualized_return(pd.Series([100.0, 0.0])))
    assert math.isnan(annualized_return(pd.Series([100.0, -50.0])))


def test_cumulative_return_reads_a_datetime_index_in_date_order(
    quarterly_series: pd.Series,
):
    # A CSV in descending date order must report +48.5%, not the reciprocal.
    shuffled = quarterly_series.iloc[[2, 0, 3, 1]]
    assert cumulative_return(shuffled) == pytest.approx(0.485, abs=1e-12)
    assert annualized_return(shuffled) == pytest.approx(
        annualized_return(quarterly_series), rel=1e-12
    )


def test_cagr_annualises_on_calendar_days_not_trading_days():
    # 1000 -> 1300 over exactly 730 calendar days:
    #   CAGR = 1.3 ** (365/730) - 1 = sqrt(1.3) - 1 = 0.14017542509913805
    values = pd.Series([1000.0, 1300.0], index=pd.DatetimeIndex([D0, day(730)]))
    assert cagr(values) == pytest.approx(math.sqrt(1.3) - 1.0, rel=1e-12)
    assert cagr(values) == pytest.approx(0.14017543, abs=1e-8)


def test_cagr_reconciles_with_xirr_on_an_account_with_a_single_opening_flow():
    # Cross-module identity. One payment in, one valuation out: the calendar
    # CAGR of the value series and the money-weighted rate of the equivalent
    # two-flow schedule are the same number, because they are the same maths.
    values = pd.Series([1000.0, 1300.0], index=pd.DatetimeIndex([D0, day(730)]))
    schedule = [(D0, -1000.0), (day(730), 1300.0)]
    assert cagr(values) == pytest.approx(xirr(schedule), abs=1e-9)


def test_cagr_falls_back_to_trading_day_annualisation_without_dates():
    # No DatetimeIndex means no calendar span to measure, so the documented
    # fallback is annualized_return — identical, not merely close.
    positional = pd.Series([100.0, 112.0])
    assert cagr(positional) == pytest.approx(annualized_return(positional), rel=1e-12)


def test_cagr_is_nan_for_a_zero_length_calendar_span():
    same_day = pd.Series([100.0, 112.0], index=pd.DatetimeIndex(["2024-01-01"] * 2))
    assert math.isnan(cagr(same_day))


# ---------------------------------------------------------------------------
# time_weighted_return
# ---------------------------------------------------------------------------
def test_time_weighted_return_without_flows_equals_annualized_return(
    portfolio_values: pd.Series,
):
    # With every F_t = 0 the chain prod((V_t)/(V_(t-1))) telescopes to
    # V_end / V_start, so the two must agree to floating-point noise.
    assert time_weighted_return(portfolio_values) == pytest.approx(
        annualized_return(portfolio_values), rel=1e-12
    )


def test_time_weighted_return_strips_out_a_contribution_that_annualized_return_keeps(
    contribution_account: tuple[pd.Series, pd.Series],
):
    # This is the entire reason TWR exists. The account went 1000 -> 1650, but
    # 500 of that was paid in, not earned:
    #   annualized_return -> 1650/1000 - 1 = 0.65   (flattered by the deposit)
    #   time_weighted     -> 1.00 * 1.10 - 1 = 0.10 (what the assets did)
    values, flows = contribution_account
    assert annualized_return(values) == pytest.approx(0.65, abs=1e-9)
    assert time_weighted_return(values, flows) == pytest.approx(0.10, abs=1e-9)


def test_a_pure_contribution_earns_a_time_weighted_return_of_exactly_zero():
    # Money in is not performance. 253 points so the exponent is 1.
    index = pd.bdate_range("2024-01-01", periods=253)
    values = np.full(253, 1000.0)
    values[126:] = 1500.0
    flows = pd.Series([500.0], index=[index[126]])
    assert time_weighted_return(pd.Series(values, index=index), flows) == pytest.approx(
        0.0, abs=1e-12
    )


def test_time_weighted_return_sums_duplicate_flow_dates(
    contribution_account: tuple[pd.Series, pd.Series],
):
    # Two 250 contributions booked on the same day are one 500 contribution.
    values, flows = contribution_account
    split = pd.Series([250.0, 250.0], index=[flows.index[0], flows.index[0]])
    assert time_weighted_return(values, split) == pytest.approx(
        time_weighted_return(values, flows), rel=1e-12
    )


def test_time_weighted_return_ignores_flow_dates_absent_from_the_value_series(
    contribution_account: tuple[pd.Series, pd.Series],
):
    # Documented: a flow the valuation series never saw contributes nothing,
    # so the result collapses back to the no-flows answer.
    values, _ = contribution_account
    stray = pd.Series([500.0], index=pd.DatetimeIndex(["1999-01-01"]))
    assert time_weighted_return(values, stray) == pytest.approx(
        time_weighted_return(values), rel=1e-12
    )


def test_a_flow_dated_on_the_very_first_observation_is_not_a_period_flow(
    contribution_account: tuple[pd.Series, pd.Series],
):
    # REGRESSION GUARD on the end-of-period convention: F_t is subtracted from
    # a CLOSING value, and the first observation has no period before it, so
    # a flow there is opening capital rather than an external flow.
    values, _ = contribution_account
    opening = pd.Series([500.0], index=[values.index[0]])
    assert time_weighted_return(values, opening) == pytest.approx(
        time_weighted_return(values), rel=1e-12
    )


def test_time_weighted_return_is_nan_when_a_flow_wipes_the_period_out():
    # (V_t - F_t) <= 0 is not a -100% period, it is a broken input: the
    # chained growth factor would be zero or negative and has no root.
    index = pd.bdate_range("2024-01-01", periods=4)
    values = pd.Series([1000.0, 1000.0, 1000.0, 1000.0], index=index)
    flows = pd.Series([2000.0], index=[index[2]])
    assert math.isnan(time_weighted_return(values, flows))


def test_time_weighted_return_is_nan_for_a_non_positive_opening_value():
    values = pd.Series([100.0, 0.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert math.isnan(time_weighted_return(values))


# ---------------------------------------------------------------------------
# periodic_returns
# ---------------------------------------------------------------------------
def test_periodic_returns_compounds_the_days_inside_each_bucket():
    # Two +10% moves inside February compound, they do not add:
    #   1.10 * 1.10 - 1 = 0.21   (0.20 would be the naive sum)
    values = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.DatetimeIndex(["2023-01-31", "2023-02-10", "2023-02-28"]),
    )
    monthly = periodic_returns(values, "ME")
    assert len(monthly) == 1
    assert float(monthly.iloc[0]) == pytest.approx(0.21, abs=1e-12)


def test_periodic_returns_is_indexed_by_period_end_dates():
    # The April observation is dated the 28th; the bucket is stamped 30 April.
    values = pd.Series([100.0, 110.0], index=pd.DatetimeIndex(["2023-03-31", "2023-04-28"]))
    monthly = periodic_returns(values, "ME")
    assert list(monthly.index) == [pd.Timestamp("2023-04-30")]
    assert monthly.name == "periodic_return"


def test_periodic_returns_are_decimals(quarterly_series: pd.Series):
    # +10%, -10%, +50% as 0.10 / -0.10 / 0.50 — the PERCENT conversion happens
    # once, in monthly_return_table, and must not happen twice.
    monthly = periodic_returns(quarterly_series, "ME")
    assert list(monthly) == pytest.approx([0.10, -0.10, 0.50], abs=1e-12)


def test_periodic_returns_omit_periods_with_no_observations_instead_of_zeroing_them():
    # March and April were never observed. Reporting them as 0% would claim
    # the portfolio was measured and flat; absence says it was not measured.
    values = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.DatetimeIndex(["2023-01-31", "2023-02-28", "2023-05-31"]),
    )
    monthly = periodic_returns(values, "ME")
    assert [stamp.month for stamp in monthly.index] == [2, 5]


@pytest.mark.parametrize("freq", ["ME", "QE", "YE"])
def test_every_bucketing_frequency_compounds_back_to_the_same_total(
    portfolio_values: pd.Series, freq: str
):
    # Identity: every daily return falls in exactly one bucket at every
    # frequency, so prod(1 + R_p) - 1 is the total return whatever the
    # bucketing. This is what makes the monthly heatmap reconcilable with the
    # headline number on the same page.
    buckets = periodic_returns(portfolio_values, freq)
    assert not buckets.empty
    assert float((1.0 + buckets).prod() - 1.0) == pytest.approx(
        cumulative_return(portfolio_values), rel=1e-12
    )


@pytest.mark.parametrize(
    ("resample_alias", "period_alias"), [("ME", "M"), ("QE", "Q"), ("YE", "Y")]
)
def test_both_spellings_of_a_period_alias_give_the_same_buckets(
    portfolio_values: pd.Series, resample_alias: str, period_alias: str
):
    # pandas 2.x spells period-END offsets "ME"/"QE"/"YE" for resampling but
    # "M"/"Q"/"Y" for Periods. The UI passes either; both must work.
    pd.testing.assert_series_equal(
        periodic_returns(portfolio_values, resample_alias),
        periodic_returns(portfolio_values, period_alias),
    )


def test_periodic_returns_is_empty_for_an_unrecognised_frequency(portfolio_values: pd.Series):
    assert periodic_returns(portfolio_values, "fortnightly").empty


def test_periodic_returns_is_empty_without_a_datetime_index():
    # Calendar buckets cannot be inferred from 0, 1, 2, ...
    assert periodic_returns(pd.Series([100.0, 110.0, 121.0])).empty


# Regression guard for a fixed defect:
# _period_alias() upper-cases only for the DICT LOOKUP and returns the raw text otherwise,
# so 'me'/'qe'/'ye' are accepted case-insensitively while the bare 'm'/'q'/'y' fall
# through to pandas, which emits a FutureWarning about the deprecated lowercase Period
# alias. Under this project's own `error::FutureWarning` policy that breaks the module's
# documented 'never raises' contract.
def test_a_lowercase_period_alias_degrades_like_every_other_spelling(
    portfolio_values: pd.Series,
):
    pd.testing.assert_series_equal(
        periodic_returns(portfolio_values, "m"),
        periodic_returns(portfolio_values, "M"),
    )


# ---------------------------------------------------------------------------
# monthly_return_table
# ---------------------------------------------------------------------------
def test_monthly_return_table_is_in_percent(quarterly_series: pd.Series):
    # UNIT GUARD. +10% is the cell value 10.0, not 0.10 — the heatmap's colour
    # scale and the report's number format both assume PERCENT here, while
    # cumulative_return next door is a DECIMAL.
    table = monthly_return_table(quarterly_series)
    assert table.loc[2023, "Feb"] == pytest.approx(10.0, abs=1e-12)
    assert table.loc[2023, "Mar"] == pytest.approx(-10.0, abs=1e-12)
    assert table.loc[2023, "Apr"] == pytest.approx(50.0, abs=1e-12)
    assert cumulative_return(quarterly_series) == pytest.approx(0.485, abs=1e-12)


def test_the_year_column_is_the_compounded_months_of_that_row(quarterly_series: pd.Series):
    # 1.10 * 0.90 * 1.50 = 1.485 -> +48.5% for 2023. Adding the months
    # instead would print 10 - 10 + 50 = 50.0, which is 1.5 points too high.
    table = monthly_return_table(quarterly_series)
    assert table.loc[2023, "Year"] == pytest.approx(48.5, abs=1e-9)
    assert table.loc[2023, list(MONTH_LABELS)].sum() == pytest.approx(50.0, abs=1e-9)


def test_the_year_column_compounds_its_row_on_real_data(portfolio_values: pd.Series):
    # Identity form of the same rule, across every row of a 500-day series
    # that spans two calendar years and a partial one.
    table = monthly_return_table(portfolio_values)
    assert len(table) >= 2
    for year, row in table.iterrows():
        compounded = float((1.0 + row[list(MONTH_LABELS)] / 100.0).prod() - 1.0)
        assert compounded == pytest.approx(row["Year"] / 100.0, rel=1e-12), year


def test_the_year_column_chains_back_to_the_total_return(portfolio_values: pd.Series):
    # Identity: compounding the annual figures reproduces the whole-window
    # return, so the table and the headline number cannot disagree.
    table = monthly_return_table(portfolio_values)
    chained = float((1.0 + table["Year"] / 100.0).prod() - 1.0)
    assert chained == pytest.approx(cumulative_return(portfolio_values), rel=1e-9)


def test_the_year_column_compounds_only_the_months_that_exist(quarterly_series: pd.Series):
    # A partial year must not be grossed up to twelve months, and the missing
    # months must stay NaN so the UI can render an em dash.
    table = monthly_return_table(quarterly_series)
    present = [label for label in MONTH_LABELS if not math.isnan(table.loc[2023, label])]
    assert present == ["Feb", "Mar", "Apr"]
    assert table.loc[2023, "Year"] == pytest.approx(48.5, abs=1e-9)


def test_monthly_return_table_has_one_row_per_calendar_year():
    # 2023: +10%, -10%, then a flat December -> 1.10*0.90*1.00 - 1 = -1.0%
    # 2024: +10% in January alone                                 -> +10.0%
    values = pd.Series(
        [100.0, 110.0, 99.0, 99.0, 108.9],
        index=pd.DatetimeIndex(
            ["2023-01-31", "2023-02-28", "2023-03-31", "2023-12-29", "2024-01-31"]
        ),
    )
    table = monthly_return_table(values)
    assert list(table.index) == [2023, 2024]
    assert table.loc[2023, "Year"] == pytest.approx(-1.0, abs=1e-9)
    assert table.loc[2024, "Year"] == pytest.approx(10.0, abs=1e-9)


def test_monthly_return_table_always_has_thirteen_ascii_columns(portfolio_values: pd.Series):
    # Column names are frozen ASCII, not calendar.month_abbr: a locale change
    # must never rename a column in a saved report.
    table = monthly_return_table(portfolio_values)
    assert list(table.columns) == [*MONTH_LABELS, "Year"]
    assert MONTH_LABELS[0] == "Jan" and MONTH_LABELS[-1] == "Dec"
    assert table.index.name == "year"
    assert list(table.index) == sorted(table.index)


def test_monthly_return_table_is_empty_when_no_monthly_return_exists(
    empty_series: pd.Series, single_value_series: pd.Series
):
    assert monthly_return_table(empty_series).empty
    assert monthly_return_table(single_value_series).empty
    assert monthly_return_table(pd.Series([100.0, 110.0])).empty  # no dates


# ---------------------------------------------------------------------------
# rolling_returns
# ---------------------------------------------------------------------------
def test_rolling_returns_measure_over_observations_not_calendar_days():
    # 100 -> 110 -> 121 -> 133.1 -> 146.41 is +10% per step, so a two-step
    # trailing return is 1.10 ** 2 - 1 = 0.21 at every measurable point.
    values = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    rolling = rolling_returns(values, 2)
    assert len(rolling) == len(values)
    assert rolling.iloc[:2].isna().all()  # no predecessor yet
    assert list(rolling.iloc[2:]) == pytest.approx([0.21, 0.21, 0.21], abs=1e-12)
    assert rolling.name == "rolling_2d_return"


def test_rolling_returns_leave_exactly_window_leading_nans(portfolio_values: pd.Series):
    rolling = rolling_returns(portfolio_values, 21)
    assert int(rolling.isna().sum()) == 21
    assert rolling.index.equals(portfolio_values.index)


def test_the_last_rolling_return_over_the_whole_window_is_the_total_return(
    portfolio_values: pd.Series,
):
    # Identity: with window = n - 1 the final point spans the entire series.
    window = len(portfolio_values) - 1
    rolling = rolling_returns(portfolio_values, window)
    assert float(rolling.iloc[-1]) == pytest.approx(cumulative_return(portfolio_values), rel=1e-12)


def test_rolling_returns_are_empty_when_the_window_is_not_shorter_than_the_series():
    assert rolling_returns(pd.Series([1.0, 2.0]), 2).empty
    assert rolling_returns(pd.Series([1.0, 2.0]), 5).empty


@pytest.mark.parametrize("window", [0, -1, float("nan"), "twenty-one"])
def test_rolling_returns_are_empty_for_an_unusable_window(portfolio_values: pd.Series, window):
    assert rolling_returns(portfolio_values, window).empty


def test_rolling_returns_are_nan_where_the_lagged_value_is_non_positive():
    # V_t / 0 must read as a blank, never as +inf.
    values = pd.Series([0.0, 50.0, 100.0, 150.0])
    rolling = rolling_returns(values, 2)
    assert math.isnan(rolling.iloc[2])
    assert float(rolling.iloc[3]) == pytest.approx(2.0, abs=1e-12)  # 150/50 - 1


# ---------------------------------------------------------------------------
# return_stats
# ---------------------------------------------------------------------------
def test_return_stats_always_returns_its_ten_documented_keys(
    portfolio_values: pd.Series, empty_series: pd.Series, nan_series: pd.Series
):
    # The report template and the Excel writer both index this dict by name,
    # so a missing key is a KeyError in a user's download, not a test failure.
    for series in (portfolio_values, empty_series, nan_series, pd.Series([100.0, 110.0])):
        stats = return_stats(series)
        assert set(stats) == RETURN_STAT_KEYS
        assert len(stats) == 10


def test_return_stats_is_hand_computable_on_the_quarterly_series(
    quarterly_series: pd.Series,
):
    # months  +10%, -10%, +50%   ->  best 50.0, worst -10.0, 2 of 3 positive
    # days    the same three moves (one observation per month)
    # streaks +, -, + -> longest win run 1, longest loss run 1
    # total   1.10*0.90*1.50 - 1 = 0.485 -> 48.5 PERCENT
    stats = return_stats(quarterly_series)
    assert stats["cumulative_pct"] == pytest.approx(48.5, abs=1e-9)
    assert stats["best_month_pct"] == pytest.approx(50.0, abs=1e-9)
    assert stats["worst_month_pct"] == pytest.approx(-10.0, abs=1e-9)
    assert stats["positive_months"] == 2
    assert stats["total_months"] == 3
    assert stats["best_day_pct"] == pytest.approx(50.0, abs=1e-9)
    assert stats["worst_day_pct"] == pytest.approx(-10.0, abs=1e-9)
    assert stats["longest_win_streak"] == 1
    assert stats["longest_loss_streak"] == 1


def test_return_stats_percentages_are_the_decimal_functions_times_one_hundred(
    portfolio_values: pd.Series,
):
    # UNIT GUARD across the module boundary.
    stats = return_stats(portfolio_values)
    assert stats["cumulative_pct"] == pytest.approx(
        cumulative_return(portfolio_values) * 100.0, rel=1e-12
    )
    assert stats["annualized_pct"] == pytest.approx(
        annualized_return(portfolio_values) * 100.0, rel=1e-12
    )
    monthly = periodic_returns(portfolio_values, "ME")
    assert stats["best_month_pct"] == pytest.approx(float(monthly.max()) * 100.0, rel=1e-12)
    assert stats["total_months"] == len(monthly)


def test_return_stats_counts_are_ints_not_numpy_scalars(portfolio_values: pd.Series):
    # np.int64 survives arithmetic but breaks `json.dumps` and the Excel
    # writer's type dispatch, so the contract is plain Python ints.
    stats = return_stats(portfolio_values)
    for key in ("positive_months", "total_months", "longest_win_streak", "longest_loss_streak"):
        assert type(stats[key]) is int


def test_return_stats_orders_its_extremes_consistently(portfolio_values: pd.Series):
    stats = return_stats(portfolio_values)
    assert stats["worst_day_pct"] <= stats["best_day_pct"]
    assert stats["worst_month_pct"] <= stats["best_month_pct"]
    assert 0 <= stats["positive_months"] <= stats["total_months"]
    assert stats["longest_win_streak"] >= 1  # a rising series has some run


def test_return_stats_reports_nan_monthly_figures_without_a_calendar():
    # Documented: calendar months cannot be inferred from a positional index,
    # so the monthly block is NaN/0 while the daily block is still computed.
    stats = return_stats(pd.Series(np.linspace(100.0, 112.0, 253)))
    assert math.isnan(stats["best_month_pct"])
    assert math.isnan(stats["worst_month_pct"])
    assert stats["total_months"] == 0
    assert stats["positive_months"] == 0
    assert stats["cumulative_pct"] == pytest.approx(12.0, abs=1e-9)
    assert stats["longest_win_streak"] == 252  # a straight line never falls


def test_return_stats_of_an_empty_series_is_nan_and_zero_not_an_exception(
    empty_series: pd.Series,
):
    stats = return_stats(empty_series)
    assert math.isnan(stats["cumulative_pct"])
    assert math.isnan(stats["annualized_pct"])
    assert math.isnan(stats["best_day_pct"])
    assert stats["total_months"] == 0
    assert stats["longest_win_streak"] == 0


# ---------------------------------------------------------------------------
# degenerate inputs, across the whole module
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name", ["empty_series", "single_value_series", "nan_series"])
def test_every_scalar_statistic_is_nan_for_a_degenerate_series(fixture_name, request):
    # The module contract: nothing raises, nothing fabricates a zero.
    series = request.getfixturevalue(fixture_name)
    for value in (
        cumulative_return(series),
        annualized_return(series),
        cagr(series),
        time_weighted_return(series),
    ):
        assert math.isnan(value)


@pytest.mark.parametrize("fixture_name", ["empty_series", "single_value_series", "nan_series"])
def test_every_series_returning_function_is_empty_for_a_degenerate_series(fixture_name, request):
    series = request.getfixturevalue(fixture_name)
    assert periodic_returns(series).empty
    assert monthly_return_table(series).empty
    assert rolling_returns(series, 5).empty


def test_none_is_treated_as_no_data_everywhere():
    assert math.isnan(cumulative_return(None))
    assert math.isnan(annualized_return(None))
    assert math.isnan(time_weighted_return(None))
    assert periodic_returns(None).empty
    assert monthly_return_table(None).empty


def test_a_duplicated_date_index_does_not_raise(quarterly_series: pd.Series):
    # Two rows stamped with the same date is a real CSV, not a hypothetical.
    duplicated = pd.Series(
        quarterly_series.to_numpy(),
        index=pd.DatetimeIndex(["2023-01-31", "2023-01-31", "2023-03-31", "2023-03-31"]),
    )
    assert cumulative_return(duplicated) == pytest.approx(0.485, abs=1e-12)
    assert not monthly_return_table(duplicated).empty
    assert math.isfinite(annualized_return(duplicated))


def test_the_value_series_is_never_mutated(portfolio_values: pd.Series):
    before = portfolio_values.copy(deep=True)
    return_stats(portfolio_values)
    monthly_return_table(portfolio_values)
    rolling_returns(portfolio_values, 21)
    time_weighted_return(portfolio_values)
    pd.testing.assert_series_equal(portfolio_values, before)


def test_nan_observations_are_dropped_rather_than_read_as_a_wipe_out():
    # A missing valuation is a gap in the record, not a day the account went
    # to zero: 121/100 - 1 = 0.21 over the two surviving observations.
    values = pd.Series(
        [100.0, np.nan, 121.0],
        index=pd.DatetimeIndex(["2023-01-31", "2023-02-28", "2023-03-31"]),
    )
    assert cumulative_return(values) == pytest.approx(0.21, abs=1e-12)
