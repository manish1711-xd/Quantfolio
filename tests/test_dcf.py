"""The DCF, checked against arithmetic rather than against its own output.

Testing philosophy for this module
----------------------------------
A snapshot test of a valuation is close to worthless: it locks in whatever the
model happened to produce, including its bugs, and a change that silently
breaks the arithmetic still passes because the recorded number was broken too.

So almost every test here asserts against a value that can be **derived by
hand**. The flat-perpetuity case is the backbone of that: with zero growth, no
capital expenditure, no working-capital drag and no tax friction beyond a flat
rate, a firm's enterprise value is exactly ``FCFF / WACC`` -- a number anyone
can check on a napkin. If the discounting, the terminal value or the mid-year
convention is wrong, that identity fails immediately.

The second class of test asserts a *direction* rather than a magnitude, where
no closed form exists: more capex must lower the value, mid-year discounting
must raise it, a higher tax rate must lower it. Directional tests catch sign
errors, which are the failure mode that matters most here because a flipped
capex sign turns a cash outflow into a cash inflow and makes every company look
wonderful.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from src.analytics import dcf

NAN = float("nan")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _flat(**overrides: object) -> dcf.DcfAssumptions:
    """A forecast whose value is analytically known: EV == FCFF / WACC.

    Revenue 1,000 flat, 10% EBIT margin, 25% tax, no D&A, no capex, no working
    capital, zero terminal growth, 10% discount rate, no mid-year convention.
    Free cash flow is therefore a flat 75 a year and the firm is worth exactly
    750.
    """
    base: dict[str, object] = {
        "years": 5,
        "revenue_growth": 0.0,
        "ebit_margin": 0.10,
        "tax_rate": 0.25,
        "da_pct_revenue": 0.0,
        "capex_pct_revenue": 0.0,
        "nwc_pct_incremental_revenue": 0.0,
        "wacc_value": 0.10,
        "terminal_growth": 0.0,
        "mid_year": False,
    }
    base.update(overrides)
    return dcf.DcfAssumptions.constant(**base)  # type: ignore[arg-type]


def _company(**overrides: object) -> dcf.CompanyFinancials:
    base: dict[str, object] = {
        "base_revenue": 1000.0,
        "net_debt": 0.0,
        "shares": 100.0,
        "current_price": 7.50,
    }
    base.update(overrides)
    return dcf.CompanyFinancials(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the analytic backbone
# ---------------------------------------------------------------------------
def test_flat_perpetuity_enterprise_value_is_free_cash_flow_over_wacc() -> None:
    """The identity the whole module rests on: EV == FCFF / WACC with no growth."""
    result = dcf.run_dcf(_flat(), _company())
    assert result.enterprise_value == pytest.approx(750.0, rel=1e-9)


def test_flat_perpetuity_free_cash_flow_is_ebit_times_one_minus_tax() -> None:
    result = dcf.run_dcf(_flat(), _company())
    assert result.forecast["fcff"] == pytest.approx([75.0] * 5, rel=1e-9)


def test_the_explicit_and_terminal_pieces_sum_to_enterprise_value() -> None:
    """No value may be created or lost in the join between the two halves."""
    result = dcf.run_dcf(_flat(), _company())
    assert result.pv_explicit + result.pv_terminal == pytest.approx(
        result.enterprise_value, rel=1e-9
    )


def test_explicit_period_is_the_annuity_factor_times_the_cash_flow() -> None:
    """75 x (1 - 1.1^-5)/0.1, computed independently of the module."""
    annuity = (1.0 - 1.10**-5) / 0.10
    result = dcf.run_dcf(_flat(), _company())
    assert result.pv_explicit == pytest.approx(75.0 * annuity, rel=1e-9)


def test_per_share_value_divides_equity_by_the_share_count() -> None:
    result = dcf.run_dcf(_flat(), _company())
    assert result.value_per_share == pytest.approx(7.50, rel=1e-9)
    assert result.equity_value == pytest.approx(750.0, rel=1e-9)


def test_net_debt_is_subtracted_from_enterprise_value() -> None:
    """Equity holders own what is left after the lenders are repaid."""
    result = dcf.run_dcf(_flat(), _company(net_debt=250.0))
    assert result.equity_value == pytest.approx(500.0, rel=1e-9)
    assert result.value_per_share == pytest.approx(5.00, rel=1e-9)


def test_net_cash_is_added_to_enterprise_value() -> None:
    """A negative net debt is a net cash position and must raise equity value."""
    result = dcf.run_dcf(_flat(), _company(net_debt=-150.0))
    assert result.equity_value == pytest.approx(900.0, rel=1e-9)


# ---------------------------------------------------------------------------
# discounting conventions
# ---------------------------------------------------------------------------
def test_mid_year_convention_raises_the_value() -> None:
    """Cash arriving through the year is worth more than cash arriving at year end."""
    year_end = dcf.run_dcf(_flat(mid_year=False), _company())
    mid = dcf.run_dcf(_flat(mid_year=True), _company())
    assert mid.enterprise_value > year_end.enterprise_value


def test_mid_year_discounts_the_first_year_by_half_a_period() -> None:
    """The first year's cash flow is discounted at 10%^-0.5, not at 10%^-1."""
    result = dcf.run_dcf(_flat(mid_year=True), _company())
    expected_first = 75.0 / (1.10**0.5)
    # PV explicit is the sum of all five; subtracting the rest leaves the first.
    rest = sum(75.0 / (1.10 ** (i - 0.5)) for i in range(2, 6))
    assert result.pv_explicit - rest == pytest.approx(expected_first, rel=1e-9)


def test_a_higher_discount_rate_lowers_the_value() -> None:
    cheap = dcf.run_dcf(_flat(wacc_value=0.09), _company())
    dear = dcf.run_dcf(_flat(wacc_value=0.14), _company())
    assert dear.enterprise_value < cheap.enterprise_value


# ---------------------------------------------------------------------------
# sign and sensitivity correctness
# ---------------------------------------------------------------------------
def test_capital_expenditure_reduces_value() -> None:
    """Guards the sign convention: capex is a magnitude, and it is subtracted."""
    no_capex = dcf.run_dcf(_flat(capex_pct_revenue=0.0), _company())
    with_capex = dcf.run_dcf(_flat(capex_pct_revenue=0.05), _company())
    assert with_capex.enterprise_value < no_capex.enterprise_value


def test_depreciation_adds_back_to_free_cash_flow() -> None:
    """D&A is a non-cash charge, so adding it back must raise free cash flow."""
    none = dcf.run_dcf(_flat(da_pct_revenue=0.0), _company())
    some = dcf.run_dcf(_flat(da_pct_revenue=0.04), _company())
    assert some.enterprise_value > none.enterprise_value


def test_working_capital_is_charged_only_on_incremental_revenue() -> None:
    """A flat business ties up no extra cash, however high the intensity."""
    flat = dcf.run_dcf(_flat(nwc_pct_incremental_revenue=0.50), _company())
    assert flat.enterprise_value == pytest.approx(750.0, rel=1e-9)


def test_growth_with_working_capital_drag_is_worth_less_than_growth_alone() -> None:
    free = dcf.run_dcf(_flat(revenue_growth=0.10, nwc_pct_incremental_revenue=0.0), _company())
    dragged = dcf.run_dcf(_flat(revenue_growth=0.10, nwc_pct_incremental_revenue=0.20), _company())
    assert dragged.enterprise_value < free.enterprise_value


def test_a_higher_tax_rate_lowers_the_value() -> None:
    low = dcf.run_dcf(_flat(tax_rate=0.15), _company())
    high = dcf.run_dcf(_flat(tax_rate=0.35), _company())
    assert high.enterprise_value < low.enterprise_value


def test_the_effective_tax_rate_from_the_accounts_overrides_the_default() -> None:
    """A company paying 12% is worth more than one paying 25% on the same EBIT."""
    statutory = dcf.run_dcf(_flat(), _company())
    observed = dcf.run_dcf(_flat(), _company(), effective_tax_rate=0.12)
    assert observed.enterprise_value > statutory.enterprise_value
    assert any("effective tax rate" in n for n in observed.notices)


# ---------------------------------------------------------------------------
# refusing to answer
# ---------------------------------------------------------------------------
def test_missing_revenue_yields_no_value_and_says_why() -> None:
    result = dcf.run_dcf(_flat(), _company(base_revenue=NAN))
    assert math.isnan(result.value_per_share)
    assert not result.ok
    assert any("revenue" in n for n in result.notices)


def test_missing_share_count_yields_no_value() -> None:
    result = dcf.run_dcf(_flat(), _company(shares=NAN))
    assert math.isnan(result.value_per_share)
    assert any("share count" in n for n in result.notices)


def test_a_zero_share_count_is_refused_rather_than_dividing_by_it() -> None:
    result = dcf.run_dcf(_flat(), _company(shares=0.0))
    assert math.isnan(result.value_per_share)


def test_a_discount_rate_below_terminal_growth_is_refused() -> None:
    """The perpetuity would divide by a negative and return a negative value."""
    result = dcf.run_dcf(_flat(wacc_value=0.04, terminal_growth=0.045), _company())
    assert math.isnan(result.value_per_share)
    assert any("discount rate" in n.lower() for n in result.notices)


def test_mismatched_growth_and_margin_paths_are_refused() -> None:
    broken = dcf.DcfAssumptions(revenue_growth=(0.10, 0.10, 0.10), ebit_margin=(0.15, 0.15))
    assert not broken.usable()
    assert any("different lengths" in p for p in broken.problems)


def test_net_debt_above_enterprise_value_is_flagged_as_worthless_equity() -> None:
    result = dcf.run_dcf(_flat(), _company(net_debt=900.0))
    assert result.equity_value < 0
    assert any("worthless" in n for n in result.notices)


# ---------------------------------------------------------------------------
# terminal value transparency
# ---------------------------------------------------------------------------
def test_terminal_share_is_reported_and_a_dominant_one_is_flagged() -> None:
    """A valuation that is mostly terminal value is a statement about the terminal."""
    result = dcf.run_dcf(_flat(), _company())
    assert 0.0 < result.terminal_share < 1.0
    # 9% terminal growth against a 10% discount rate leaves only a 1pp spread,
    # so almost all of the 750 is the perpetuity rather than the five years.
    # (0.099 would be refused outright: the spread collapses below the 0.5pp
    # minimum and the perpetuity divides by nearly zero.)
    heavy = dcf.run_dcf(_flat(terminal_growth=0.09), _company())
    assert heavy.terminal_share > dcf.MAX_TERMINAL_VALUE_SHARE
    assert any("terminal value" in n for n in heavy.notices)


def test_the_exit_multiple_terminal_value_agrees_when_assumptions_are_coherent() -> None:
    """Gordon and an exit multiple should reconcile; a gap means one is wrong.

    The two methods must measure the same quantity to be comparable at all: an
    exit-EBITDA multiple values the perpetuity on EBITDA, Gordon values it on
    free cash flow, and those differ by tax and by D&A. So the fixture removes
    both -- a zero tax rate and no D&A make EBITDA exactly equal to FCFF. The
    valuation then discounts a flat 100 a year at 10%, so the perpetuity is
    worth 1,000, and a 10x EBITDA exit multiple must land on the same 1,000.
    Any disagreement here is an arithmetic error in one of the two paths.
    """
    result = dcf.run_dcf(
        _flat(tax_rate=0.0, exit_ebitda_multiple=10.0),
        _company(base_ebitda=100.0, base_revenue=1000.0),
    )
    assert result.terminal_value == pytest.approx(1000.0, rel=1e-9)
    assert not math.isnan(result.terminal_value_exit)
    assert result.terminal_disagreement == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sensitivity grid
# ---------------------------------------------------------------------------
def test_sensitivity_grid_locates_the_base_case() -> None:
    """The grid holds per-share values, so it can be read against the price."""
    grid = dcf.sensitivity_grid(
        _flat(), _company(), row_values=(0.08, 0.10, 0.12), col_values=(0.0, 0.02)
    )
    assert grid.ok
    assert grid.rows[grid.base_row] == pytest.approx(0.10)
    assert grid.cols[grid.base_col] == pytest.approx(0.00)
    assert grid.base_value == pytest.approx(7.50, rel=1e-9)  # 750 over 100 shares


def test_sensitivity_grid_is_monotonic_in_the_discount_rate() -> None:
    """Every higher discount rate must produce a strictly lower value."""
    rates = (0.08, 0.10, 0.12, 0.14)
    grid = dcf.sensitivity_grid(_flat(), _company(), row_values=rates, col_values=(0.0,))
    column = [row[0] for row in grid.values]
    assert column == sorted(column, reverse=True)


def test_sensitivity_grid_reports_how_elastic_the_answer_is() -> None:
    grid = dcf.sensitivity_grid(
        _flat(), _company(), row_values=(0.08, 0.10, 0.13), col_values=(0.0,)
    )
    assert grid.spread > 0.3  # a 3pp move in WACC moves the value by >30%


def test_sensitivity_grid_reports_the_current_price_for_comparison() -> None:
    grid = dcf.sensitivity_grid(_flat(), _company(), row_values=(0.10,), col_values=(0.0,))
    assert grid.current_price == pytest.approx(7.50)


# ---------------------------------------------------------------------------
# reverse DCF
# ---------------------------------------------------------------------------
def test_reverse_dcf_round_trips_the_growth_it_was_given() -> None:
    """The strongest correctness check available: invert the model and recover
    the input. Building a flat-perpetuity DCF at a price of 7.50 means the
    growth consistent with that price is exactly zero."""
    result = dcf.reverse_dcf(_flat(), _company(), "revenue_cagr")
    assert result.implied_growth == pytest.approx(0.0, abs=1e-6)


def test_reverse_dcf_recovers_a_positive_growth_that_was_built_in() -> None:
    """Value at 10% growth, then ask what growth that price implies."""
    grown = dcf.run_dcf(_flat(revenue_growth=0.10), _company())
    priced = _company(current_price=grown.value_per_share)
    implied = dcf.reverse_dcf(_flat(revenue_growth=0.10), priced, "revenue_cagr")
    assert implied.implied_growth == pytest.approx(0.10, abs=1e-5)


def test_reverse_dcf_refuses_a_price_the_model_cannot_reach() -> None:
    """Out of bracket is reported with the bracket, never as a boundary value."""
    absurd = _company(current_price=1_000_000.0)
    result = dcf.reverse_dcf(_flat(), absurd, "revenue_cagr", lo=-0.2, hi=0.6)
    assert math.isnan(result.implied_growth)
    assert any("outside" in n for n in result.notices)


def test_reverse_dcf_needs_a_price() -> None:
    result = dcf.reverse_dcf(_flat(), _company(current_price=NAN), "revenue_cagr")
    assert not result.ok
    assert any("price" in n for n in result.notices)


# ---------------------------------------------------------------------------
# cost of capital
# ---------------------------------------------------------------------------
def test_cost_of_equity_is_capm() -> None:
    assert dcf.cost_of_equity(0.07, 1.2, 0.07) == pytest.approx(0.07 + 1.2 * 0.07)


def test_cost_of_equity_adds_an_optional_size_premium() -> None:
    with_premium = dcf.cost_of_equity(0.07, 1.0, 0.07, size_premium=0.02)
    assert with_premium == pytest.approx(0.16)


def test_after_tax_cost_of_debt_shields_interest() -> None:
    assert dcf.after_tax_cost_of_debt(0.09, 0.25) == pytest.approx(0.0675)


def test_wacc_is_the_weighted_average() -> None:
    result = dcf.wacc(
        dcf.WaccInputs(
            risk_free=0.07,
            equity_risk_premium=0.07,
            beta=1.0,
            pretax_cost_of_debt=0.09,
            tax_rate=0.25,
            equity_weight=0.80,
            debt_weight=0.20,
        )
    )
    expected = 0.80 * 0.14 + 0.20 * 0.0675
    assert result.wacc == pytest.approx(expected)


def test_wacc_weights_are_normalised_whatever_scale_they_arrive_in() -> None:
    """60/40 and 0.6/0.4 must give the same answer."""
    fraction = dcf.wacc(
        dcf.WaccInputs(beta=1.0, pretax_cost_of_debt=0.08, equity_weight=0.6, debt_weight=0.4)
    )
    percent = dcf.wacc(
        dcf.WaccInputs(beta=1.0, pretax_cost_of_debt=0.08, equity_weight=60.0, debt_weight=40.0)
    )
    assert percent.wacc == pytest.approx(fraction.wacc)


def test_wacc_infers_the_other_weight_when_only_one_is_given() -> None:
    result = dcf.wacc(dcf.WaccInputs(beta=1.0, pretax_cost_of_debt=0.08, equity_weight=0.75))
    assert result.debt_weight == pytest.approx(0.25)


def test_wacc_without_a_cost_of_debt_is_computed_on_equity_alone_and_says_so() -> None:
    result = dcf.wacc(dcf.WaccInputs(beta=1.0, equity_weight=1.0, debt_weight=0.0))
    assert result.ok
    assert any("Cost of debt" in n for n in result.notices)


def test_wacc_build_up_is_readable_arithmetic() -> None:
    result = dcf.wacc(
        dcf.WaccInputs(beta=1.0, pretax_cost_of_debt=0.08, equity_weight=0.5, debt_weight=0.5)
    )
    assert "%" in result.build_up
    assert "=" in result.build_up


def test_wacc_returns_nan_when_no_weights_are_supplied() -> None:
    result = dcf.wacc(dcf.WaccInputs(beta=1.0))
    assert not result.ok


# ---------------------------------------------------------------------------
# seeding assumptions from history
# ---------------------------------------------------------------------------
def test_default_assumptions_fade_growth_towards_the_terminal_rate() -> None:
    assumptions = dcf.default_assumptions(historical_growth=(0.20, 0.22, 0.24), years=5)
    path = assumptions.revenue_growth
    assert len(path) == 5
    assert path[0] > path[-1]
    assert path[-1] == pytest.approx(dcf.DEFAULT_TERMINAL_GROWTH)
    assert all(a >= b for a, b in pairwise(path))  # monotonically fading


def test_default_assumptions_never_extrapolate_the_latest_rate_forever() -> None:
    """A company that grew 40% does not grow 40% for the whole horizon."""
    assumptions = dcf.default_assumptions(historical_growth=(0.05, 0.40), years=5)
    assert assumptions.revenue_growth[1] < 0.40


def test_default_assumptions_cope_with_no_history_at_all() -> None:
    assumptions = dcf.default_assumptions()
    assert assumptions.usable()
    assert len(assumptions.revenue_growth) == 5


def test_default_assumptions_use_the_latest_observed_margin() -> None:
    assumptions = dcf.default_assumptions(historical_margin=(0.10, 0.12, 0.18), years=3)
    assert assumptions.ebit_margin == pytest.approx((0.18, 0.18, 0.18))


# ---------------------------------------------------------------------------
# history from statements
# ---------------------------------------------------------------------------
def test_historical_rates_are_year_over_year_growth() -> None:
    from src.analytics import fundamentals as fu

    periods = tuple(fu.FiscalPeriod(None, {"revenue": v}) for v in (100.0, 110.0, 132.0))
    statements = fu.Statements(periods=tuple(reversed(periods)))
    rates = dcf.historical_rates(statements, "revenue")
    # series() runs oldest first, so growth is 10% then 20%.
    assert rates == pytest.approx((0.10, 0.20))


def test_historical_rates_are_empty_without_two_periods() -> None:
    from src.analytics import fundamentals as fu

    statements = fu.Statements(periods=(fu.FiscalPeriod(None, {"revenue": 100.0}),))
    assert dcf.historical_rates(statements, "revenue") == ()


def test_historical_rates_tolerate_an_object_that_is_not_statements() -> None:
    """The page may hand this anything; it must degrade, not raise."""
    assert dcf.historical_rates(object(), "revenue") == ()
