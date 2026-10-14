"""Option pricing, Greeks and overlays — ``src.analytics.options``, offline.

Option maths has an unusual property for a test author: it is almost entirely
pinned by identities and limits. Nothing here needs a golden number copied from
a spreadsheet.

Verified here
-------------
**Put-call parity**, ``C - P == S e^(-qT) - K e^(-rT)``, to 1e-12. This single
identity constrains both formulas simultaneously — a sign error or a misplaced
discount factor in either one breaks it.

**Every Greek against a numerical derivative** of the pricing function by
central differences. Delta, vega, theta and rho match to ~1e-10; gamma to
~1e-6 with a properly-scaled step (a second difference at h=1e-5 loses most of
its precision to cancellation, which is a property of the test, not the code).

**The Greek cross-identities.** Gamma and vega are identical for a call and a
put at the same strike, and ``delta_call - delta_put == e^(-qT)`` exactly.
Cheap, and they catch a whole class of copy-paste error between the two
branches.

**Implied volatility round-trips** from 10% to 200% to ~1e-11, and REFUSES
rather than guessing below the no-arbitrage floor or above the underlying.

**Limits**: deep ITM to the discounted forward, deep OTM to zero, ``T=0`` to
intrinsic, ``sigma=0`` to the *discounted forward* intrinsic (which is not the
same number — conflating them is a common slip and is pinned separately).

**Overlay payoffs against their closed forms** at, above and below every
strike: a covered call is ``min(S_T, K) - S_0 + premium``, a protective put is
``max(S_T, K) - S_0 - premium``, and a collar is exactly the sum of its legs.

**Coverage arithmetic.** ``max_contracts`` floors, and writing more contracts
than the shares cover produces an explicit NAKED warning — the one error in
this module that could cost a user unbounded money.

Deterministic: fixed inputs, no RNG, no network.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from src.analytics import options as op

S, K, T, V, R, Q = 100.0, 105.0, 0.5, 0.25, 0.05, 0.02
TODAY = date(2025, 1, 15)
EXPIRY = TODAY + timedelta(days=30)


def price(spot=S, strike=K, time=T, vol=V, rate=R, div=Q, kind=op.CALL) -> float:
    return op.black_scholes_price(spot, strike, time, vol, rate, div, kind)


def call_quote(strike: float = 105.0, premium: float = 2.10, **kwargs) -> op.OptionQuote:
    kwargs.setdefault("expiry", EXPIRY)
    kwargs.setdefault("implied_vol", 0.25)
    kwargs.setdefault("contract_size", 100)
    return op.OptionQuote(op.CALL, strike, premium, **kwargs)


def put_quote(strike: float = 95.0, premium: float = 1.80, **kwargs) -> op.OptionQuote:
    kwargs.setdefault("expiry", EXPIRY)
    kwargs.setdefault("implied_vol", 0.28)
    kwargs.setdefault("contract_size", 100)
    return op.OptionQuote(op.PUT, strike, premium, **kwargs)


# ---------------------------------------------------------------------------
# put-call parity
# ---------------------------------------------------------------------------
def test_put_call_parity_holds() -> None:
    """One identity constraining both formulas at once."""
    parity = price(kind=op.CALL) - price(kind=op.PUT)
    forward = S * math.exp(-Q * T) - K * math.exp(-R * T)

    assert parity == pytest.approx(forward, abs=1e-12)


@pytest.mark.parametrize("spot", [50.0, 90.0, 100.0, 105.0, 150.0, 400.0])
@pytest.mark.parametrize("vol", [0.05, 0.25, 0.80])
def test_parity_holds_across_moneyness_and_volatility(spot: float, vol: float) -> None:
    parity = price(spot=spot, vol=vol, kind=op.CALL) - price(spot=spot, vol=vol, kind=op.PUT)
    forward = spot * math.exp(-Q * T) - K * math.exp(-R * T)

    assert parity == pytest.approx(forward, abs=1e-10)


def test_parity_holds_with_no_rate_or_dividend() -> None:
    parity = price(rate=0.0, div=0.0, kind=op.CALL) - price(rate=0.0, div=0.0, kind=op.PUT)

    assert parity == pytest.approx(S - K, abs=1e-12)


# ---------------------------------------------------------------------------
# Greeks against numerical derivatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_delta_matches_the_numerical_derivative(kind: str) -> None:
    h = 1e-5
    numeric = (price(spot=S + h, kind=kind) - price(spot=S - h, kind=kind)) / (2 * h)

    assert op.greeks(S, K, T, V, R, Q, kind).delta == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_gamma_matches_the_numerical_second_derivative(kind: str) -> None:
    """A larger step than the first-derivative checks, deliberately: a second
    difference at h=1e-5 loses most of its precision to cancellation, so a
    tight tolerance there would be testing floating-point noise."""
    h = 0.05
    numeric = (
        price(spot=S + h, kind=kind) - 2 * price(kind=kind) + price(spot=S - h, kind=kind)
    ) / (h * h)

    assert op.greeks(S, K, T, V, R, Q, kind).gamma == pytest.approx(numeric, rel=1e-4)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_vega_matches_the_numerical_derivative(kind: str) -> None:
    h = 1e-5
    numeric = (price(vol=V + h, kind=kind) - price(vol=V - h, kind=kind)) / (2 * h)

    assert op.greeks(S, K, T, V, R, Q, kind).vega_raw == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_theta_matches_the_numerical_time_derivative(kind: str) -> None:
    """Theta is the NEGATIVE derivative with respect to time to expiry: value
    decays as expiry approaches."""
    h = 1e-5
    numeric = -(price(time=T + h, kind=kind) - price(time=T - h, kind=kind)) / (2 * h)

    assert op.greeks(S, K, T, V, R, Q, kind).theta_raw == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_rho_matches_the_numerical_rate_derivative(kind: str) -> None:
    h = 1e-5
    numeric = (price(rate=R + h, kind=kind) - price(rate=R - h, kind=kind)) / (2 * h)

    assert op.greeks(S, K, T, V, R, Q, kind).rho * 100.0 == pytest.approx(numeric, rel=1e-6)


# ---------------------------------------------------------------------------
# Greek cross-identities
# ---------------------------------------------------------------------------
def test_gamma_is_identical_for_a_call_and_a_put() -> None:
    """Both come from the same phi(d1) term. Catches a copy-paste error
    between the two branches for free."""
    assert op.greeks(S, K, T, V, R, Q, op.CALL).gamma == pytest.approx(
        op.greeks(S, K, T, V, R, Q, op.PUT).gamma, abs=1e-15
    )


def test_vega_is_identical_for_a_call_and_a_put() -> None:
    assert op.greeks(S, K, T, V, R, Q, op.CALL).vega == pytest.approx(
        op.greeks(S, K, T, V, R, Q, op.PUT).vega, abs=1e-15
    )


def test_the_delta_difference_is_exactly_the_dividend_discount() -> None:
    """delta_call - delta_put == e^(-qT), the differentiated form of parity."""
    call_delta = op.greeks(S, K, T, V, R, Q, op.CALL).delta
    put_delta = op.greeks(S, K, T, V, R, Q, op.PUT).delta

    assert call_delta - put_delta == pytest.approx(math.exp(-Q * T), abs=1e-12)


def test_a_call_delta_is_between_zero_and_the_dividend_discount() -> None:
    for spot in (10.0, 90.0, 105.0, 500.0):
        delta = op.greeks(spot, K, T, V, R, Q, op.CALL).delta
        assert 0.0 <= delta <= math.exp(-Q * T) + 1e-12, spot


def test_a_put_delta_is_never_positive() -> None:
    for spot in (10.0, 90.0, 105.0, 500.0):
        assert op.greeks(spot, K, T, V, R, Q, op.PUT).delta <= 0.0, spot


def test_gamma_and_vega_are_never_negative() -> None:
    for spot in (50.0, 100.0, 200.0):
        greeks = op.greeks(spot, K, T, V, R, Q, op.CALL)
        assert greeks.gamma >= 0.0
        assert greeks.vega >= 0.0


def test_gamma_peaks_near_the_money() -> None:
    at_money = op.greeks(105.0, K, T, V, R, Q, op.CALL).gamma
    for away in (60.0, 180.0):
        assert op.greeks(away, K, T, V, R, Q, op.CALL).gamma < at_money


def test_theta_is_quoted_per_calendar_day_and_vega_per_vol_point() -> None:
    """The units a desk reasons in, not the raw partials."""
    greeks = op.greeks(S, K, T, V, R, Q, op.CALL)

    assert greeks.theta == pytest.approx(greeks.theta_raw / op.DAYS_PER_YEAR)
    assert greeks.vega == pytest.approx(greeks.vega_raw / 100.0)


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------
def test_a_call_is_worth_more_as_the_spot_rises() -> None:
    prices = [price(spot=s) for s in (80.0, 95.0, 105.0, 120.0)]

    assert prices == sorted(prices)


def test_a_call_is_worth_less_as_the_strike_rises() -> None:
    prices = [price(strike=k) for k in (80.0, 95.0, 105.0, 120.0)]

    assert prices == sorted(prices, reverse=True)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_both_are_worth_more_with_more_volatility(kind: str) -> None:
    prices = [price(vol=v, kind=kind) for v in (0.05, 0.15, 0.35, 0.90)]

    assert prices == sorted(prices)


@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_both_are_worth_more_with_more_time(kind: str) -> None:
    """True for a European option on a non-dividend-paying underlying; with
    q = 0 it holds for puts too."""
    prices = [price(time=t, div=0.0, kind=kind) for t in (0.05, 0.25, 1.0, 2.0)]

    assert prices == sorted(prices)


def test_a_price_is_never_negative() -> None:
    for spot in (1.0, 100.0, 10_000.0):
        for kind in (op.CALL, op.PUT):
            assert price(spot=spot, kind=kind) >= 0.0


# ---------------------------------------------------------------------------
# limits and degenerate inputs
# ---------------------------------------------------------------------------
def test_a_deep_in_the_money_call_approaches_the_discounted_forward() -> None:
    assert price(spot=10_000.0, strike=10.0) == pytest.approx(
        10_000.0 * math.exp(-Q * T) - 10.0 * math.exp(-R * T), rel=1e-9
    )


def test_a_deep_out_of_the_money_call_is_worthless() -> None:
    assert price(spot=1.0, strike=10_000.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "spot,strike,kind,expected",
    [
        (110.0, 100.0, op.CALL, 10.0),
        (90.0, 100.0, op.CALL, 0.0),
        (90.0, 100.0, op.PUT, 10.0),
        (110.0, 100.0, op.PUT, 0.0),
    ],
)
def test_at_expiry_the_price_is_intrinsic(
    spot: float, strike: float, kind: str, expected: float
) -> None:
    assert price(spot=spot, strike=strike, time=0.0, kind=kind) == pytest.approx(expected)


def test_zero_volatility_gives_the_DISCOUNTED_forward_intrinsic() -> None:
    """NOT the plain intrinsic. With certainty the payoff is known but still
    has to be discounted, and conflating the two is a common slip — so the
    distinction gets its own test."""
    certain = price(spot=120.0, strike=100.0, vol=0.0)
    forward_intrinsic = 120.0 * math.exp(-Q * T) - 100.0 * math.exp(-R * T)

    assert certain == pytest.approx(forward_intrinsic, abs=1e-12)
    assert certain != pytest.approx(20.0, abs=1e-6), "plain intrinsic would be wrong"


@pytest.mark.parametrize(
    "spot,strike,vol,kind",
    [(0.0, 100.0, 0.25, op.CALL), (100.0, 0.0, 0.25, op.CALL), (100.0, 100.0, -0.1, op.CALL)],
)
def test_an_unusable_input_yields_nan_not_a_number(
    spot: float, strike: float, vol: float, kind: str
) -> None:
    assert math.isnan(price(spot=spot, strike=strike, vol=vol, kind=kind))


def test_an_unknown_option_type_yields_nan() -> None:
    assert math.isnan(op.black_scholes_price(S, K, T, V, R, Q, "straddle"))
    assert math.isnan(op.greeks(S, K, T, V, R, Q, "straddle").delta)


def test_greeks_collapse_at_expiry_rather_than_dividing_by_zero() -> None:
    itm = op.greeks(110.0, 100.0, 0.0, V, R, Q, op.CALL)

    assert itm.delta == 1.0
    assert itm.gamma == 0.0
    assert itm.theta == 0.0


# ---------------------------------------------------------------------------
# implied volatility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", [0.05, 0.10, 0.25, 0.60, 1.20, 2.00])
@pytest.mark.parametrize("kind", [op.CALL, op.PUT])
def test_implied_volatility_round_trips(target: float, kind: str) -> None:
    solved = op.implied_volatility(price(vol=target, kind=kind), S, K, T, R, Q, kind)

    assert solved == pytest.approx(target, abs=1e-8)


def test_implied_volatility_round_trips_away_from_the_money() -> None:
    for spot in (70.0, 90.0, 120.0, 160.0):
        target = 0.35
        solved = op.implied_volatility(price(spot=spot, vol=target), spot, K, T, R, Q, op.CALL)
        assert solved == pytest.approx(target, abs=1e-6), spot


def test_a_price_below_the_no_arbitrage_floor_has_no_implied_volatility() -> None:
    """Which means the quote is stale, crossed, or carries an American
    early-exercise premium this European model cannot represent. Refusing is
    the only honest answer."""
    floor = price(spot=150.0, strike=100.0, vol=0.0)

    assert math.isnan(op.implied_volatility(floor - 1.0, 150.0, 100.0, T, R, Q, op.CALL))


def test_a_price_above_the_underlying_has_no_implied_volatility() -> None:
    assert math.isnan(op.implied_volatility(S * 2, S, K, T, R, Q, op.CALL))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_an_unusable_target_price_yields_nan(bad: float) -> None:
    assert math.isnan(op.implied_volatility(bad, S, K, T, R, Q, op.CALL))


def test_implied_volatility_at_expiry_is_undefined() -> None:
    assert math.isnan(op.implied_volatility(5.0, S, K, 0.0, R, Q, op.CALL))


# ---------------------------------------------------------------------------
# probability of finishing in the money
# ---------------------------------------------------------------------------
def test_the_itm_probability_is_a_probability() -> None:
    for spot in (50.0, 100.0, 200.0):
        for kind in (op.CALL, op.PUT):
            value = op.probability_itm(spot, K, T, V, R, Q, kind)
            assert 0.0 <= value <= 1.0


def test_call_and_put_itm_probabilities_sum_to_one() -> None:
    """N(d2) + N(-d2) == 1 by construction."""
    total = op.probability_itm(S, K, T, V, R, Q, op.CALL) + op.probability_itm(
        S, K, T, V, R, Q, op.PUT
    )

    assert total == pytest.approx(1.0, abs=1e-12)


def test_a_further_out_of_the_money_call_is_less_likely_to_finish_in() -> None:
    probabilities = [op.probability_itm(S, k, T, V, R, Q, op.CALL) for k in (90.0, 100.0, 130.0)]

    assert probabilities == sorted(probabilities, reverse=True)


def test_at_expiry_the_probability_is_zero_or_one() -> None:
    assert op.probability_itm(110.0, 100.0, 0.0, V, R, Q, op.CALL) == 1.0
    assert op.probability_itm(90.0, 100.0, 0.0, V, R, Q, op.CALL) == 0.0


# ---------------------------------------------------------------------------
# contracts and coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shares,size,expected",
    [(500, 100, 5), (450, 100, 4), (99, 100, 0), (0, 100, 0), (1000, 250, 4), (500, 0, 0)],
)
def test_max_contracts_floors_never_rounds(shares: float, size: int, expected: int) -> None:
    """Rounding up would leave part of the position naked."""
    assert op.max_contracts(shares, size) == expected


def test_writing_more_contracts_than_the_shares_cover_is_warned_about() -> None:
    """The one error in this module that could cost unbounded money."""
    result = op.covered_call(S, 450, call_quote(), contracts=5, as_of=TODAY)

    assert any("NAKED" in w for w in result.warnings)


def test_a_fully_covered_write_raises_no_coverage_warning() -> None:
    result = op.covered_call(S, 500, call_quote(), contracts=5, as_of=TODAY)

    assert not any("NAKED" in w for w in result.warnings)


def test_uncovered_shares_are_reported() -> None:
    result = op.covered_call(S, 450, call_quote(), as_of=TODAY)

    assert result.contracts == 4
    assert result.covered_shares == 400.0
    assert result.uncovered_shares == 50.0


# ---------------------------------------------------------------------------
# covered call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("terminal", [70.0, 100.0, 104.99, 105.0, 130.0, 500.0])
def test_the_covered_call_payoff_matches_its_closed_form(terminal: float) -> None:
    """payoff = shares * (min(S_T, K) - S_0 + premium)."""
    quote = call_quote(105.0, 2.10)
    expected = 500 * (min(terminal, 105.0) - 100.0 + 2.10)

    got = op.payoff_at_expiry(terminal, 100.0, 500, short_call=quote, contracts=5)

    assert got == pytest.approx(expected)


def test_the_covered_call_payoff_is_capped_above_the_strike() -> None:
    quote = call_quote(105.0, 2.10)
    at_strike = op.payoff_at_expiry(105.0, 100.0, 500, short_call=quote, contracts=5)
    far_above = op.payoff_at_expiry(1_000.0, 100.0, 500, short_call=quote, contracts=5)

    assert far_above == pytest.approx(at_strike)


def test_the_covered_call_figures_are_arithmetically_right() -> None:
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10), as_of=TODAY)

    assert result.ok
    assert result.net_premium_per_share == pytest.approx(2.10)
    assert result.net_premium_total == pytest.approx(2.10 * 500)
    assert result.breakeven == pytest.approx(97.90)
    assert result.max_profit == pytest.approx(7.10)
    assert result.static_return == pytest.approx(0.021)
    assert result.if_called_return == pytest.approx(0.071)
    assert result.upside_cap == pytest.approx(0.05)


def test_the_covered_call_max_loss_is_the_full_position_net_of_premium() -> None:
    """A covered call is not "limited risk": the underlying can go to zero.
    Reporting anything smaller would be false comfort."""
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10), as_of=TODAY)

    assert result.max_loss == pytest.approx(97.90)


def test_the_covered_call_is_an_income_structure() -> None:
    result = op.covered_call(100.0, 500, call_quote(), as_of=TODAY)

    assert result.is_income
    assert result.net_premium_per_share > 0


def test_a_put_supplied_as_a_covered_call_is_refused() -> None:
    result = op.covered_call(100.0, 500, put_quote(), as_of=TODAY)

    assert not result.ok
    assert any("CALL" in w for w in result.warnings)


def test_a_covered_call_with_too_few_shares_is_refused() -> None:
    result = op.covered_call(100.0, 50, call_quote(), as_of=TODAY)

    assert not result.ok
    assert result.warnings


# ---------------------------------------------------------------------------
# protective put
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("terminal", [50.0, 94.99, 95.0, 100.0, 130.0])
def test_the_protective_put_payoff_matches_its_closed_form(terminal: float) -> None:
    """payoff = shares * (max(S_T, K) - S_0 - premium)."""
    quote = put_quote(95.0, 1.80)
    expected = 500 * (max(terminal, 95.0) - 100.0 - 1.80)

    got = op.payoff_at_expiry(terminal, 100.0, 500, long_put=quote, contracts=5)

    assert got == pytest.approx(expected)


def test_the_protective_put_payoff_is_floored_below_the_strike() -> None:
    quote = put_quote(95.0, 1.80)
    at_strike = op.payoff_at_expiry(95.0, 100.0, 500, long_put=quote, contracts=5)
    at_zero = op.payoff_at_expiry(0.01, 100.0, 500, long_put=quote, contracts=5)

    assert at_zero == pytest.approx(at_strike)


def test_the_protective_put_figures_are_arithmetically_right() -> None:
    result = op.protective_put(100.0, 500, put_quote(95.0, 1.80), as_of=TODAY)

    assert result.ok
    assert result.net_premium_per_share == pytest.approx(-1.80)
    assert result.breakeven == pytest.approx(101.80)
    assert result.floor_price == pytest.approx(93.20)
    assert result.max_loss == pytest.approx(6.80)
    assert result.downside_protection == pytest.approx(0.05)


def test_the_protective_puts_static_return_is_negative() -> None:
    """The insurance costs money every period it is not needed, and saying so
    is the whole point of reporting it as a return."""
    result = op.protective_put(100.0, 500, put_quote(), as_of=TODAY)

    assert result.static_return < 0
    assert result.static_return_annual < 0
    assert not result.is_income


def test_the_protective_put_upside_is_unbounded() -> None:
    result = op.protective_put(100.0, 500, put_quote(), as_of=TODAY)

    assert result.max_profit == float("inf")


def test_a_call_supplied_as_a_protective_put_is_refused() -> None:
    result = op.protective_put(100.0, 500, call_quote(), as_of=TODAY)

    assert not result.ok
    assert any("PUT" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# collar
# ---------------------------------------------------------------------------
def test_the_collar_net_premium_is_the_difference_of_the_legs() -> None:
    result = op.collar(100.0, 500, call_quote(105.0, 2.10), put_quote(95.0, 1.80), as_of=TODAY)

    assert result.net_premium_per_share == pytest.approx(0.30)


def test_the_collar_payoff_is_exactly_the_sum_of_its_legs() -> None:
    """Written as one function over components, so this cannot drift."""
    call, put = call_quote(105.0, 2.10), put_quote(95.0, 1.80)
    for terminal in (60.0, 95.0, 100.0, 105.0, 200.0):
        both = op.payoff_at_expiry(terminal, 100.0, 500, short_call=call, long_put=put, contracts=5)
        stock = 500 * (terminal - 100.0)
        legs = (
            op.payoff_at_expiry(terminal, 100.0, 500, short_call=call, contracts=5)
            + op.payoff_at_expiry(terminal, 100.0, 500, long_put=put, contracts=5)
            - stock
        )
        assert both == pytest.approx(legs), terminal


def test_the_collar_is_bounded_in_both_directions() -> None:
    """The only structure here with a genuinely bounded outcome either way."""
    result = op.collar(100.0, 500, call_quote(105.0, 2.10), put_quote(95.0, 1.80), as_of=TODAY)

    assert result.max_profit == pytest.approx(5.30)
    assert result.max_loss == pytest.approx(4.70)
    assert result.floor_price == pytest.approx(95.30)
    assert result.upside_cap == pytest.approx(0.05)


def test_an_inverted_collar_is_flagged() -> None:
    """A put strike above the call strike locks in a loss whatever happens."""
    result = op.collar(100.0, 500, call_quote(95.0, 6.0), put_quote(105.0, 7.0), as_of=TODAY)

    assert any("above the call strike" in w for w in result.warnings)


def test_mismatched_expiries_are_flagged_as_a_diagonal() -> None:
    call = call_quote(105.0, 2.10, expiry=EXPIRY)
    put = put_quote(95.0, 1.80, expiry=EXPIRY + timedelta(days=30))

    result = op.collar(100.0, 500, call, put, as_of=TODAY)

    assert any("diagonal" in w for w in result.warnings)


def test_the_legs_must_be_supplied_in_the_right_order() -> None:
    result = op.collar(100.0, 500, put_quote(), call_quote(), as_of=TODAY)

    assert not result.ok
    assert any("one call and one put" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# annualisation
# ---------------------------------------------------------------------------
def test_returns_are_annualised_by_compounding_not_multiplying() -> None:
    """A 2.1% premium over 30 days is 28.8% a year compounded, not 25.6%
    multiplied — and for a structure written repeatedly the difference is the
    reason to write it."""
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10), as_of=TODAY)

    compounded = (1.021) ** (op.DAYS_PER_YEAR / 30) - 1
    assert result.static_return_annual == pytest.approx(compounded, rel=1e-9)
    assert result.static_return_annual > result.static_return * (365 / 30) * 0.9


def test_pricing_uses_calendar_days_not_trading_days() -> None:
    """An option decays over a weekend, so the 252-day convention used
    everywhere else in this package would understate decay by ~30%."""
    assert op.DAYS_PER_YEAR == 365.0
    assert op.year_fraction(TODAY + timedelta(days=365), TODAY) == pytest.approx(1.0)


def test_year_fraction_floors_at_zero_for_an_expired_option() -> None:
    assert op.year_fraction(TODAY - timedelta(days=10), TODAY) == 0.0


def test_year_fraction_of_an_unusable_date_is_nan() -> None:
    assert math.isnan(op.year_fraction(None, TODAY))


# ---------------------------------------------------------------------------
# quotes
# ---------------------------------------------------------------------------
def test_the_quote_midpoint_prefers_the_bid_ask() -> None:
    quote = call_quote(105.0, 2.10, bid=2.00, ask=2.20)

    assert quote.mid == pytest.approx(2.10)


def test_the_quote_midpoint_falls_back_to_the_premium_when_unquoted() -> None:
    quote = call_quote(105.0, 2.10, bid=float("nan"), ask=float("nan"))

    assert quote.mid == pytest.approx(2.10)


def test_the_spread_percentage_is_measured_against_the_midpoint() -> None:
    quote = call_quote(105.0, 2.10, bid=2.00, ask=2.20)

    assert quote.spread_pct == pytest.approx(0.20 / 2.10 * 100.0)


def test_an_unquoted_spread_is_nan_not_zero() -> None:
    assert math.isnan(call_quote(bid=float("nan"), ask=float("nan")).spread_pct)


def test_a_wide_spread_is_warned_about() -> None:
    """An overlay attractive at the midpoint can be unattractive at the price
    actually available."""
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10, bid=1.80, ask=2.40), as_of=TODAY)

    assert any("bid-ask spread" in w for w in result.warnings)


def test_thin_open_interest_is_warned_about() -> None:
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10, open_interest=3), as_of=TODAY)

    assert any("Open interest" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# payoff curves and comparison
# ---------------------------------------------------------------------------
def test_the_payoff_curve_compares_the_overlay_against_stock_alone() -> None:
    """An overlay payoff shown on its own says nothing about what it cost."""
    result = op.covered_call(100.0, 500, call_quote(), as_of=TODAY)
    curve = op.payoff_curve(result)

    assert list(curve.columns) == ["price", "stock_only", "overlay", "difference"]
    assert not curve.empty
    assert (curve["difference"] == curve["overlay"] - curve["stock_only"]).all()


def test_the_payoff_curve_puts_the_strike_exactly_on_the_grid() -> None:
    """A grid straddling the strike would round the kink off and make a capped
    payoff look smoothly rising."""
    result = op.covered_call(100.0, 500, call_quote(105.0), as_of=TODAY)
    curve = op.payoff_curve(result)

    assert 105.0 in set(curve["price"])


def test_the_covered_call_curve_is_flat_above_the_strike() -> None:
    result = op.covered_call(100.0, 500, call_quote(105.0, 2.10), as_of=TODAY)
    curve = op.payoff_curve(result, span=0.5)
    above = curve[curve["price"] > 105.0]["overlay"]

    # Compared with a tolerance, not for bit-equality: the payoff is computed
    # as 500*(S_T-100) + 500*(2.10 - (S_T-105)), in which a large intermediate
    # cancels. The residue varies in the last bits with S_T even though the
    # value is mathematically constant, so nunique() would report noise.
    assert above.max() - above.min() < 1e-6, "the payoff must be capped, not merely flattening"
    assert above.iloc[0] == pytest.approx(3550.0)


def test_the_protective_put_curve_is_flat_below_the_strike() -> None:
    result = op.protective_put(100.0, 500, put_quote(95.0, 1.80), as_of=TODAY)
    curve = op.payoff_curve(result, span=0.5)
    below = curve[curve["price"] < 95.0]["overlay"]

    assert below.max() - below.min() < 1e-6, "the floor must be flat"
    assert below.iloc[0] == pytest.approx(-3400.0)


def test_the_payoff_curve_of_an_unusable_result_is_empty_but_typed() -> None:
    curve = op.payoff_curve(op.OverlayResult())

    assert curve.empty
    assert list(curve.columns) == ["price", "stock_only", "overlay", "difference"]


def test_the_comparison_table_puts_max_loss_beside_the_yields() -> None:
    """An income table showing only premium invites picking the biggest number,
    and the biggest number usually has the least protection."""
    results = [
        op.covered_call(100.0, 500, call_quote(), as_of=TODAY),
        op.protective_put(100.0, 500, put_quote(), as_of=TODAY),
        op.collar(100.0, 500, call_quote(), put_quote(), as_of=TODAY),
    ]

    frame = op.overlay_comparison(results)

    assert len(frame) == 3
    assert "max_loss" in frame.columns
    assert "static_return_annual_pct" in frame.columns
    assert set(frame["strategy"]) == {"Covered call", "Protective put", "Collar"}


def test_the_comparison_table_skips_unusable_results() -> None:
    frame = op.overlay_comparison(
        [op.OverlayResult(), op.covered_call(100.0, 500, call_quote(), as_of=TODAY)]
    )

    assert len(frame) == 1


def test_the_comparison_table_of_nothing_is_empty_but_typed() -> None:
    frame = op.overlay_comparison([])

    assert frame.empty
    assert "strategy" in frame.columns


# ---------------------------------------------------------------------------
# the summary frame
# ---------------------------------------------------------------------------
def test_the_summary_frame_omits_measures_that_do_not_apply() -> None:
    """A covered call has no floor price; a protective put has no if-called
    return. Showing either as a blank row would imply the concept applies."""
    call_rows = set(
        op.covered_call(100.0, 500, call_quote(), as_of=TODAY).summary_frame()["Measure"]
    )
    put_rows = set(
        op.protective_put(100.0, 500, put_quote(), as_of=TODAY).summary_frame()["Measure"]
    )

    assert "If-called return %" in call_rows
    assert "Floor price" not in call_rows
    assert "Floor price" in put_rows
    assert "If-called return %" not in put_rows


def test_the_summary_frame_of_a_collar_carries_both_bounds() -> None:
    rows = set(
        op.collar(100.0, 500, call_quote(), put_quote(), as_of=TODAY).summary_frame()["Measure"]
    )

    assert "Floor price" in rows
    assert "Upside cap %" in rows


def test_greeks_are_attached_to_the_overlay() -> None:
    result = op.covered_call(100.0, 500, call_quote(), as_of=TODAY)

    assert result.greeks_call is not None
    assert math.isfinite(result.greeks_call.delta)


def test_the_implied_volatility_is_solved_when_the_quote_omits_it() -> None:
    """So the assignment probability and Greeks still have a volatility to
    work from."""
    quote = op.OptionQuote(
        op.CALL, 105.0, 2.10, EXPIRY, implied_vol=float("nan"), contract_size=100
    )

    result = op.covered_call(100.0, 500, quote, rate=R, dividend=Q, as_of=TODAY)

    assert math.isfinite(result.probability_assigned)
    assert 0.0 <= result.probability_assigned <= 1.0


# ---------------------------------------------------------------------------
# a stock leg with no options at all
# ---------------------------------------------------------------------------
def test_the_payoff_with_no_contracts_is_just_the_stock() -> None:
    assert op.payoff_at_expiry(120.0, 100.0, 500, contracts=0) == pytest.approx(10_000.0)


def test_the_payoff_ignores_a_leg_of_the_wrong_type() -> None:
    """A put passed as the short-call leg must be ignored, not applied as a
    call — silently mispricing the structure would be far worse."""
    got = op.payoff_at_expiry(120.0, 100.0, 500, short_call=put_quote(), contracts=5)

    assert got == pytest.approx(10_000.0)


@pytest.mark.parametrize("bad", [float("nan"), None])
def test_an_unusable_terminal_price_yields_nan(bad: object) -> None:
    assert math.isnan(op.payoff_at_expiry(bad, 100.0, 500))  # type: ignore[arg-type]


def test_moneyness_is_spot_over_strike() -> None:
    assert op.moneyness(110.0, 100.0) == pytest.approx(1.1)
    assert math.isnan(op.moneyness(110.0, 0.0))


def test_an_unusable_overlay_result_is_not_ok() -> None:
    result = op.OverlayResult()

    assert not result.ok
    assert result.summary_frame() is not None
    assert isinstance(result.summary_frame(), pd.DataFrame)
