"""Option pricing, Greeks and covered-call / protective-put overlay analytics.

Scope, and a data limitation stated first because it shapes the design
--------------------------------------------------------------------
Everything in this module is pure mathematics over inputs the caller supplies:
spot, strike, time, volatility, rate, dividend yield. It therefore works for
any option on any exchange.

What does NOT work is fetching an option chain for Indian underlyings. Yahoo
Finance -- the only market-data source this application has -- publishes chains
for US listings and nothing for NSE::

    AAPL          -> 21 expiries
    RELIANCE.NS   ->  0 expiries
    ^NSEI         ->  0 expiries

For an India-first tool that is a significant hole, and the honest response is
not to approximate a chain from somewhere. It is to make the analytics
complete and let the user type in the contract they are actually looking at --
strike, premium, expiry -- from their broker's screen. The premium is the only
number that genuinely needs a market, and the user already has it in front of
them. :mod:`src.data.fetcher` reports chain availability per symbol rather than
returning an empty frame that reads like "no options exist".

European vs American, and the irony
-----------------------------------
Black-Scholes prices a EUROPEAN option. NSE stock and index options are
European-style, so the model is exactly right for the market this application
targets. US equity options are AMERICAN, so for the market whose chains we CAN
fetch the model is an approximation -- it undervalues an American put, and a
call on a stock about to pay a large dividend, because both can be worth
exercising early. That gap is stated wherever a US chain is priced rather than
left for the reader to know already.

What the overlay analytics answer
---------------------------------
*Covered call* (own the shares, sell a call): premium income now against
upside surrendered above the strike. The numbers that matter are the static
return if nothing happens, the return if called away, the breakeven, and the
chance of assignment.

*Protective put* (own the shares, buy a put): a floor under the position,
bought with a premium that is a permanent drag if it is never needed. The
numbers are the floor, the breakeven, and the annualised cost of the
insurance.

*Collar* (both at once): sell the call to pay for the put. Sometimes free,
sometimes better than free, and always at the cost of the upside.

On "probability of assignment"
------------------------------
Reported as the RISK-NEUTRAL probability of finishing in the money -- the
``N(d2)`` that falls out of the same model that priced the option. It is not a
forecast, and under the real-world measure the number would differ by the
equity risk premium. Labelled as such everywhere it appears, because a
probability presented without its measure invites being read as a prediction.

Pure module: numpy/pandas/scipy only. No I/O, no network, no Streamlit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)

__all__ = [
    "CALL",
    "PUT",
    "DAYS_PER_YEAR",
    "US_CONTRACT_SIZE",
    "MIN_TIME",
    "MAX_IMPLIED_VOL",
    "Greeks",
    "OptionQuote",
    "OverlayResult",
    "year_fraction",
    "black_scholes_price",
    "greeks",
    "implied_volatility",
    "moneyness",
    "probability_itm",
    "payoff_at_expiry",
    "max_contracts",
    "covered_call",
    "protective_put",
    "collar",
    "payoff_curve",
    "overlay_comparison",
]

CALL = "call"
PUT = "put"

#: Option pricing uses CALENDAR days, not trading days: an option decays over
#: weekends. This is the one place in the codebase where 252 would be wrong.
DAYS_PER_YEAR: float = 365.0

#: Shares per contract on US equity options. NSE lot sizes vary per underlying
#: and are revised periodically by the exchange, so there is no safe default
#: for India -- the caller must supply it.
US_CONTRACT_SIZE: int = 100

#: Below this many years to expiry an option is treated as expiring: the
#: Black-Scholes formula divides by sqrt(T) and degenerates.
MIN_TIME: float = 1e-9

#: Implied-volatility search ceiling. Above ~500% annualised the quote is
#: almost certainly stale or crossed rather than genuinely that volatile.
MAX_IMPLIED_VOL: float = 5.0

_NAN = float("nan")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _f(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001 - exotic input
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).date()


def _kind(option_type: str) -> str:
    text = str(option_type or "").strip().lower()
    if text.startswith("c"):
        return CALL
    if text.startswith("p"):
        return PUT
    return ""


def year_fraction(expiry: object, as_of: object = None) -> float:
    """Years to expiry on a 365-day calendar. ``0.0`` once expired.

    Calendar days, deliberately: an option loses value across a weekend, so
    the 252-day convention every other module in this package uses would
    understate decay by about 30%.
    """
    end = _as_date(expiry)
    start = _as_date(as_of) or date.today()
    if end is None:
        return _NAN
    return max(0.0, (end - start).days / DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def _d1_d2(
    spot: float, strike: float, time: float, vol: float, rate: float, dividend: float
) -> tuple[float, float]:
    sigma_root_t = vol * math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * time) / sigma_root_t
    return d1, d1 - sigma_root_t


def black_scholes_price(
    spot: float,
    strike: float,
    time: float,
    vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    option_type: str = CALL,
) -> float:
    """Black-Scholes-Merton price of a EUROPEAN option.

        d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)
        call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
        put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    Two degenerate cases are handled explicitly rather than left to divide by
    zero, because both are reachable from a real UI:

    * ``time <= 0`` -- an expired or expiring option is worth its intrinsic
      value, ``max(S - K, 0)`` or ``max(K - S, 0)``.
    * ``vol <= 0`` -- a certain payoff, so the DISCOUNTED forward intrinsic
      ``max(S e^(-qT) - K e^(-rT), 0)``. Note this is not the same as the
      undiscounted intrinsic above; conflating them is a common slip.

    Returns NaN for unusable inputs (non-positive spot or strike) rather than
    a number, so a caller cannot mistake garbage for a price.
    """
    kind = _kind(option_type)
    s, k, t, v = _f(spot), _f(strike), _f(time), _f(vol)
    r, q = _f(rate), _f(dividend)
    if not kind or not all(math.isfinite(x) for x in (s, k, t, v, r, q)):
        return _NAN
    if s <= 0.0 or k <= 0.0 or t < 0.0 or v < 0.0:
        return _NAN

    if t <= MIN_TIME:
        return max(s - k, 0.0) if kind == CALL else max(k - s, 0.0)

    discounted_spot = s * math.exp(-q * t)
    discounted_strike = k * math.exp(-r * t)
    if v <= 0.0:
        forward_intrinsic = discounted_spot - discounted_strike
        return max(forward_intrinsic, 0.0) if kind == CALL else max(-forward_intrinsic, 0.0)

    d1, d2 = _d1_d2(s, k, t, v, r, q)
    if kind == CALL:
        return discounted_spot * stats.norm.cdf(d1) - discounted_strike * stats.norm.cdf(d2)
    return discounted_strike * stats.norm.cdf(-d2) - discounted_spot * stats.norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    """Option sensitivities, in the units a desk actually quotes them in.

    ``vega`` is per 1 PERCENTAGE POINT of volatility and ``theta`` per
    CALENDAR DAY, not per unit and per year. The raw partial derivatives are
    kept alongside as ``vega_raw`` and ``theta_raw`` so nothing is lost, but
    the quoted units are the defaults because "vega 0.23" meaning 23 paise per
    vol point is the number a human reasons with.
    """

    delta: float = _NAN
    gamma: float = _NAN
    vega: float = _NAN
    theta: float = _NAN
    rho: float = _NAN
    vega_raw: float = _NAN
    theta_raw: float = _NAN

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def greeks(
    spot: float,
    strike: float,
    time: float,
    vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    option_type: str = CALL,
) -> Greeks:
    """Analytic Black-Scholes Greeks.

    Gamma and vega are identical for a call and a put at the same strike --
    both come from the same ``phi(d1)`` term -- which is one of the cheapest
    correctness checks available and is asserted in the test suite. Delta
    differs by exactly ``e^(-qT)``, another.

    All NaN for an unusable input, and zero-ish limits at expiry rather than a
    division by zero.
    """
    kind = _kind(option_type)
    s, k, t, v = _f(spot), _f(strike), _f(time), _f(vol)
    r, q = _f(rate), _f(dividend)
    if not kind or not all(math.isfinite(x) for x in (s, k, t, v, r, q)):
        return Greeks()
    if s <= 0.0 or k <= 0.0 or t <= MIN_TIME or v <= 0.0:
        # At expiry delta is a step function and every other Greek collapses.
        if s > 0 and k > 0 and t <= MIN_TIME:
            in_money = s > k if kind == CALL else s < k
            delta = (1.0 if kind == CALL else -1.0) if in_money else 0.0
            return Greeks(
                delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, vega_raw=0.0, theta_raw=0.0
            )
        return Greeks()

    d1, d2 = _d1_d2(s, k, t, v, r, q)
    pdf = float(stats.norm.pdf(d1))
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    root_t = math.sqrt(t)

    gamma = disc_q * pdf / (s * v * root_t)
    vega_raw = s * disc_q * pdf * root_t
    decay = -s * disc_q * pdf * v / (2.0 * root_t)

    if kind == CALL:
        delta = disc_q * float(stats.norm.cdf(d1))
        theta_raw = (
            decay
            - r * k * disc_r * float(stats.norm.cdf(d2))
            + q * s * disc_q * float(stats.norm.cdf(d1))
        )
        rho = k * t * disc_r * float(stats.norm.cdf(d2))
    else:
        delta = -disc_q * float(stats.norm.cdf(-d1))
        theta_raw = (
            decay
            + r * k * disc_r * float(stats.norm.cdf(-d2))
            - q * s * disc_q * float(stats.norm.cdf(-d1))
        )
        rho = -k * t * disc_r * float(stats.norm.cdf(-d2))

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega_raw / 100.0,
        theta=theta_raw / DAYS_PER_YEAR,
        rho=rho / 100.0,
        vega_raw=vega_raw,
        theta_raw=theta_raw,
    )


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    option_type: str = CALL,
) -> float:
    """Volatility that reproduces ``price``, by Brent bracketing.

    Refuses rather than guessing in the two cases where no answer exists:

    * a price BELOW the no-arbitrage floor (the discounted forward intrinsic),
      which means the quote is stale, crossed, or an American early-exercise
      premium the European model cannot represent;
    * a price above the underlying itself, which is not a legitimate quote.

    Brent rather than Newton: vega goes to zero for deep in- or out-of-the-money
    options, and Newton divides by it. Bracketing cannot diverge.
    """
    kind = _kind(option_type)
    target, s, k, t = _f(price), _f(spot), _f(strike), _f(time)
    r, q = _f(rate), _f(dividend)
    if not kind or not all(math.isfinite(x) for x in (target, s, k, t, r, q)):
        return _NAN
    if s <= 0.0 or k <= 0.0 or t <= MIN_TIME or target <= 0.0:
        return _NAN

    floor = black_scholes_price(s, k, t, 0.0, r, q, kind)
    if target < floor - 1e-9:
        logger.info(
            "Option price %.4f is below the no-arbitrage floor %.4f; no implied volatility exists.",
            target,
            floor,
        )
        return _NAN
    ceiling = s * math.exp(-q * t) if kind == CALL else k * math.exp(-r * t)
    if target > ceiling + 1e-9:
        return _NAN

    def residual(sigma: float) -> float:
        return black_scholes_price(s, k, t, sigma, r, q, kind) - target

    low, high = 1e-6, MAX_IMPLIED_VOL
    if residual(low) > 0 or residual(high) < 0:
        return _NAN
    try:
        return float(optimize.brentq(residual, low, high, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - bracket checked above
        logger.warning("Implied volatility solve failed: %s", exc)
        return _NAN


def moneyness(spot: float, strike: float) -> float:
    """``S / K``. Above 1 a call is in the money, a put out of it."""
    s, k = _f(spot), _f(strike)
    if not math.isfinite(s) or not math.isfinite(k) or k <= 0:
        return _NAN
    return s / k


def probability_itm(
    spot: float,
    strike: float,
    time: float,
    vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    option_type: str = CALL,
) -> float:
    """RISK-NEUTRAL probability of finishing in the money: ``N(d2)``.

    Not a forecast. This is the probability under the measure that priced the
    option, in which every asset drifts at the risk-free rate. Under the
    real-world measure an equity drifts faster, so the true chance of a call
    finishing in the money is HIGHER than this number and the chance for a put
    is lower. Every caller is expected to say "risk-neutral"; the name of this
    function does not, so its docstring has to.
    """
    kind = _kind(option_type)
    s, k, t, v = _f(spot), _f(strike), _f(time), _f(vol)
    r, q = _f(rate), _f(dividend)
    if not kind or not all(math.isfinite(x) for x in (s, k, t, v, r, q)):
        return _NAN
    if s <= 0 or k <= 0:
        return _NAN
    if t <= MIN_TIME:
        in_money = s > k if kind == CALL else s < k
        return 1.0 if in_money else 0.0
    if v <= 0.0:
        forward = s * math.exp((r - q) * t)
        in_money = forward > k if kind == CALL else forward < k
        return 1.0 if in_money else 0.0

    _d1, d2 = _d1_d2(s, k, t, v, r, q)
    return float(stats.norm.cdf(d2) if kind == CALL else stats.norm.cdf(-d2))


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------
def max_contracts(shares: float, contract_size: int = US_CONTRACT_SIZE) -> int:
    """How many contracts the holding actually covers.

    A covered call is only covered if the shares are there: writing more
    contracts than ``floor(shares / contract_size)`` leaves part of the
    position naked, with unbounded loss. Floored, never rounded.
    """
    held, size = _f(shares), int(contract_size or 0)
    if not math.isfinite(held) or held <= 0 or size <= 0:
        return 0
    return int(held // size)


@dataclass(frozen=True)
class OptionQuote:
    """One option contract as the user or a chain describes it."""

    option_type: str
    strike: float
    premium: float
    expiry: date | None = None
    implied_vol: float = _NAN
    bid: float = _NAN
    ask: float = _NAN
    open_interest: float = _NAN
    volume: float = _NAN
    contract_size: int = US_CONTRACT_SIZE
    source: str = ""

    @property
    def kind(self) -> str:
        return _kind(self.option_type)

    @property
    def is_call(self) -> bool:
        return self.kind == CALL

    @property
    def mid(self) -> float:
        """Bid-ask midpoint, falling back to the premium when unquoted."""
        bid, ask = _f(self.bid), _f(self.ask)
        if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid:
            return (bid + ask) / 2.0
        return _f(self.premium)

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a percentage of the midpoint. NaN when unquoted.

        Worth surfacing: an overlay that looks attractive at the midpoint can
        be unattractive at the price actually available, and a 20%-wide spread
        on a thin strike is a real cost the payoff diagram does not show.
        """
        bid, ask = _f(self.bid), _f(self.ask)
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask < bid:
            return _NAN
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 100.0 if mid > 0 else _NAN


# ---------------------------------------------------------------------------
# payoffs
# ---------------------------------------------------------------------------
def payoff_at_expiry(
    terminal_price: float,
    entry_price: float,
    shares: float = 1.0,
    short_call: OptionQuote | None = None,
    long_put: OptionQuote | None = None,
    contracts: int = 0,
) -> float:
    """Profit and loss of the whole overlaid position at expiry.

        stock          : shares * (S_T - S_0)
        short call     : contracts * size * (premium - max(S_T - K, 0))
        long put       : contracts * size * (max(K - S_T, 0) - premium)

    Written as one function over the components rather than three, so a collar
    is automatically the sum and cannot disagree with its own legs.
    """
    terminal, entry, held = _f(terminal_price), _f(entry_price), _f(shares)
    if not all(math.isfinite(x) for x in (terminal, entry, held)):
        return _NAN

    total = held * (terminal - entry)
    n = max(int(contracts or 0), 0)
    if n <= 0:
        return total

    if short_call is not None and short_call.is_call:
        size = max(int(short_call.contract_size or 0), 0)
        intrinsic = max(terminal - _f(short_call.strike), 0.0)
        total += n * size * (_f(short_call.premium) - intrinsic)
    if long_put is not None and not long_put.is_call:
        size = max(int(long_put.contract_size or 0), 0)
        intrinsic = max(_f(long_put.strike) - terminal, 0.0)
        total += n * size * (intrinsic - _f(long_put.premium))
    return total


# ---------------------------------------------------------------------------
# overlay analytics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OverlayResult:
    """Everything one overlay structure implies, per share and in total.

    Per-share figures are the ones to reason with ("this yields 1.8% a month");
    totals are what actually lands in the account. Both are kept because
    quoting only one is how a user ends up off by a factor of the contract
    size.
    """

    strategy: str = ""
    spot: float = _NAN
    shares: float = _NAN
    contracts: int = 0
    contract_size: int = US_CONTRACT_SIZE
    covered_shares: float = _NAN
    uncovered_shares: float = _NAN
    days_to_expiry: int = 0
    #: Net premium per share: positive is received, negative is paid.
    net_premium_per_share: float = _NAN
    net_premium_total: float = _NAN
    breakeven: float = _NAN
    max_profit: float = _NAN
    max_loss: float = _NAN
    #: Return if the underlying is unchanged at expiry (decimal).
    static_return: float = _NAN
    static_return_annual: float = _NAN
    #: Return if the short call is assigned (covered call / collar only).
    if_called_return: float = _NAN
    if_called_return_annual: float = _NAN
    #: Worst realisable value per share (protective put / collar only).
    floor_price: float = _NAN
    downside_protection: float = _NAN
    upside_cap: float = _NAN
    probability_assigned: float = _NAN
    probability_exercised: float = _NAN
    call: OptionQuote | None = None
    put: OptionQuote | None = None
    greeks_call: Greeks | None = None
    greeks_put: Greeks | None = None
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.contracts > 0 and math.isfinite(self.net_premium_per_share)

    @property
    def is_income(self) -> bool:
        return math.isfinite(self.net_premium_per_share) and self.net_premium_per_share > 0

    def summary_frame(self) -> pd.DataFrame:
        """Label/value pairs for display, skipping what does not apply."""
        rows: list[tuple[str, float]] = [
            ("Spot", self.spot),
            ("Contracts", float(self.contracts)),
            ("Covered shares", self.covered_shares),
            ("Net premium per share", self.net_premium_per_share),
            ("Net premium total", self.net_premium_total),
            ("Breakeven", self.breakeven),
            ("Max profit", self.max_profit),
            ("Max loss", self.max_loss),
            ("Static return %", self.static_return * 100.0),
            ("Static return % p.a.", self.static_return_annual * 100.0),
        ]
        if math.isfinite(self.if_called_return):
            rows.append(("If-called return %", self.if_called_return * 100.0))
            rows.append(("If-called return % p.a.", self.if_called_return_annual * 100.0))
        if math.isfinite(self.floor_price):
            rows.append(("Floor price", self.floor_price))
            rows.append(("Downside protection %", self.downside_protection * 100.0))
        if math.isfinite(self.upside_cap):
            rows.append(("Upside cap %", self.upside_cap * 100.0))
        if math.isfinite(self.probability_assigned):
            rows.append(("P(assigned), risk-neutral %", self.probability_assigned * 100.0))
        if math.isfinite(self.probability_exercised):
            rows.append(("P(exercised), risk-neutral %", self.probability_exercised * 100.0))
        return pd.DataFrame(rows, columns=["Measure", "Value"])


def _annualise(period_return: float, days: int) -> float:
    """Compound a holding-period return to a year. NaN when undefined.

    Compounded, not multiplied: a 2% monthly premium is 26.8% a year, not 24%,
    and for short-dated overlays the difference is the whole point of writing
    them repeatedly.
    """
    value, horizon = _f(period_return), int(days or 0)
    if not math.isfinite(value) or horizon <= 0 or value <= -1.0:
        return _NAN
    periods = DAYS_PER_YEAR / horizon
    try:
        return float((1.0 + value) ** periods - 1.0)
    except OverflowError:  # pragma: no cover - absurdly short horizon
        return _NAN


def _overlay_warnings(
    shares: float, contracts: int, size: int, quote: OptionQuote | None
) -> list[str]:
    out: list[str] = []
    covered = contracts * size
    if math.isfinite(shares) and covered > shares + 1e-9:
        out.append(
            f"{contracts} contract(s) of {size} covers {covered:,.0f} shares but only "
            f"{shares:,.0f} are held — the excess would be NAKED, with unbounded loss "
            "on a short call. Reduce the contract count."
        )
    if quote is not None:
        spread = quote.spread_pct
        if math.isfinite(spread) and spread > 10.0:
            out.append(
                f"The bid-ask spread on this strike is {spread:.1f}% of the midpoint. "
                "Every figure here uses the premium supplied; at the price actually "
                "available the economics are materially worse."
            )
        if math.isfinite(_f(quote.open_interest)) and _f(quote.open_interest) < 10:
            out.append(
                "Open interest on this strike is in single digits — expect to trade "
                "well away from the theoretical price, if at all."
            )
    return out


def covered_call(
    spot: float,
    shares: float,
    call: OptionQuote,
    contracts: int | None = None,
    entry_price: float | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    as_of: object = None,
) -> OverlayResult:
    """Long stock plus a short call: premium income for capped upside.

    ``entry_price`` defaults to ``spot``, which is the right default for
    "should I write this today". Passing the actual cost basis instead answers
    a different and also useful question -- total return from inception -- and
    the caller chooses.

    * breakeven = entry - premium
    * max profit = (K - entry) + premium, reached at or above the strike
    * static return = premium / spot, if the shares do not move
    * if-called return = (K - spot + premium) / spot

    ``max_loss`` is the entry price net of premium: a covered call still loses
    everything but the premium if the underlying goes to zero. Reporting it as
    "limited risk" would be false comfort, so the number is the full amount.
    """
    s, held = _f(spot), _f(shares)
    entry = _f(entry_price) if entry_price is not None else s
    size = max(int(call.contract_size or 0), 0)
    n = max_contracts(held, size) if contracts is None else max(int(contracts), 0)
    days = (
        max(0, (_as_date(call.expiry) - (_as_date(as_of) or date.today())).days)
        if call.expiry
        else 0
    )
    time = days / DAYS_PER_YEAR
    strike, premium = _f(call.strike), _f(call.premium)

    if not call.is_call:
        return OverlayResult(
            strategy="Covered call",
            warnings=("A covered call needs a CALL option; a put was supplied.",),
        )
    if not all(math.isfinite(x) for x in (s, held, strike, premium)) or s <= 0 or n <= 0:
        return OverlayResult(
            strategy="Covered call",
            warnings=(
                "Not enough usable inputs: a covered call needs a positive spot, the "
                "shares held, a strike and a premium, and enough shares to cover at "
                "least one contract.",
            ),
        )

    vol = _f(call.implied_vol)
    if not math.isfinite(vol) or vol <= 0:
        vol = implied_volatility(premium, s, strike, time, rate, dividend, CALL)

    static = premium / s
    if_called = (strike - s + premium) / s
    return OverlayResult(
        strategy="Covered call",
        spot=s,
        shares=held,
        contracts=n,
        contract_size=size,
        covered_shares=float(n * size),
        uncovered_shares=max(0.0, held - n * size),
        days_to_expiry=days,
        net_premium_per_share=premium,
        net_premium_total=premium * n * size,
        breakeven=entry - premium,
        max_profit=(strike - entry) + premium,
        max_loss=entry - premium,
        static_return=static,
        static_return_annual=_annualise(static, days),
        if_called_return=if_called,
        if_called_return_annual=_annualise(if_called, days),
        upside_cap=(strike - s) / s,
        probability_assigned=probability_itm(s, strike, time, vol, rate, dividend, CALL),
        call=call,
        greeks_call=greeks(s, strike, time, vol, rate, dividend, CALL),
        warnings=tuple(_overlay_warnings(held, n, size, call)),
    )


def protective_put(
    spot: float,
    shares: float,
    put: OptionQuote,
    contracts: int | None = None,
    entry_price: float | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    as_of: object = None,
) -> OverlayResult:
    """Long stock plus a long put: a floor, bought with a premium.

    * breakeven = entry + premium
    * floor = K - premium, the worst realisable value per share
    * max loss = entry - K + premium
    * static return = -premium / spot, the drag if the insurance is never used

    ``static_return`` is NEGATIVE by construction and that is the point: a
    protective put costs money every period it is not needed. Presenting the
    cost as anything other than a negative return would be flattering the
    structure.
    """
    s, held = _f(spot), _f(shares)
    entry = _f(entry_price) if entry_price is not None else s
    size = max(int(put.contract_size or 0), 0)
    n = max_contracts(held, size) if contracts is None else max(int(contracts), 0)
    days = (
        max(0, (_as_date(put.expiry) - (_as_date(as_of) or date.today())).days) if put.expiry else 0
    )
    time = days / DAYS_PER_YEAR
    strike, premium = _f(put.strike), _f(put.premium)

    if put.is_call:
        return OverlayResult(
            strategy="Protective put",
            warnings=("A protective put needs a PUT option; a call was supplied.",),
        )
    if not all(math.isfinite(x) for x in (s, held, strike, premium)) or s <= 0 or n <= 0:
        return OverlayResult(
            strategy="Protective put",
            warnings=(
                "Not enough usable inputs: a protective put needs a positive spot, the "
                "shares held, a strike and a premium, and enough shares for at least "
                "one contract.",
            ),
        )

    vol = _f(put.implied_vol)
    if not math.isfinite(vol) or vol <= 0:
        vol = implied_volatility(premium, s, strike, time, rate, dividend, PUT)

    static = -premium / s
    return OverlayResult(
        strategy="Protective put",
        spot=s,
        shares=held,
        contracts=n,
        contract_size=size,
        covered_shares=float(n * size),
        uncovered_shares=max(0.0, held - n * size),
        days_to_expiry=days,
        net_premium_per_share=-premium,
        net_premium_total=-premium * n * size,
        breakeven=entry + premium,
        max_profit=float("inf"),
        max_loss=entry - strike + premium,
        static_return=static,
        static_return_annual=_annualise(static, days),
        floor_price=strike - premium,
        downside_protection=(s - strike) / s if s > 0 else _NAN,
        probability_exercised=probability_itm(s, strike, time, vol, rate, dividend, PUT),
        put=put,
        greeks_put=greeks(s, strike, time, vol, rate, dividend, PUT),
        warnings=tuple(_overlay_warnings(held, n, size, put)),
    )


def collar(
    spot: float,
    shares: float,
    call: OptionQuote,
    put: OptionQuote,
    contracts: int | None = None,
    entry_price: float | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    as_of: object = None,
) -> OverlayResult:
    """Long stock, short call, long put: the call pays for the put.

    Net premium can be positive (a credit collar), negative, or near zero (a
    "costless" collar -- costless in cash, never in upside). Both bounds are
    real: the position cannot make more than the call strike allows nor lose
    more than the put strike permits, which is the only structure here with a
    genuinely bounded outcome in both directions.
    """
    s, held = _f(spot), _f(shares)
    entry = _f(entry_price) if entry_price is not None else s
    size = max(int(call.contract_size or 0), 0)
    n = max_contracts(held, size) if contracts is None else max(int(contracts), 0)

    if not call.is_call or put.is_call:
        return OverlayResult(
            strategy="Collar",
            warnings=("A collar needs one call and one put, in that order.",),
        )
    call_leg = covered_call(s, held, call, n, entry, rate, dividend, as_of)
    put_leg = protective_put(s, held, put, n, entry, rate, dividend, as_of)
    if not call_leg.ok or not put_leg.ok:
        return OverlayResult(
            strategy="Collar",
            warnings=tuple(dict.fromkeys([*call_leg.warnings, *put_leg.warnings])),
        )

    call_strike, put_strike = _f(call.strike), _f(put.strike)
    net = _f(call.premium) - _f(put.premium)
    days = call_leg.days_to_expiry
    static = net / s
    if_called = (call_strike - s + net) / s

    warnings = list(dict.fromkeys([*call_leg.warnings, *put_leg.warnings]))
    if put_strike > call_strike:
        warnings.append(
            f"The put strike ({put_strike:,.2f}) is above the call strike "
            f"({call_strike:,.2f}), which locks in a loss on the stock leg whatever "
            "happens. Check the strikes."
        )
    if call_leg.days_to_expiry != put_leg.days_to_expiry:
        warnings.append(
            "The two legs expire on different dates, so this is a diagonal, not a "
            "collar. Every figure below assumes the call's expiry."
        )

    return OverlayResult(
        strategy="Collar",
        spot=s,
        shares=held,
        contracts=n,
        contract_size=size,
        covered_shares=float(n * size),
        uncovered_shares=max(0.0, held - n * size),
        days_to_expiry=days,
        net_premium_per_share=net,
        net_premium_total=net * n * size,
        breakeven=entry - net,
        max_profit=(call_strike - entry) + net,
        max_loss=entry - put_strike - net,
        static_return=static,
        static_return_annual=_annualise(static, days),
        if_called_return=if_called,
        if_called_return_annual=_annualise(if_called, days),
        floor_price=put_strike + net,
        downside_protection=(s - put_strike) / s if s > 0 else _NAN,
        upside_cap=(call_strike - s) / s,
        probability_assigned=call_leg.probability_assigned,
        probability_exercised=put_leg.probability_exercised,
        call=call,
        put=put,
        greeks_call=call_leg.greeks_call,
        greeks_put=put_leg.greeks_put,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# payoff curves
# ---------------------------------------------------------------------------
def payoff_curve(
    result: OverlayResult,
    entry_price: float | None = None,
    span: float = 0.4,
    points: int = 121,
) -> pd.DataFrame:
    """Profit at expiry across a price grid: stock alone vs the overlay.

    Columns ``price, stock_only, overlay, difference``. The comparison is the
    whole point -- an overlay payoff shown on its own tells the reader nothing
    about what it cost them.

    Strikes are forced onto the grid so the kink lands exactly on them; a grid
    that straddled a strike would round the corner off and make a capped payoff
    look smoothly rising.
    """
    if not result.ok:
        return pd.DataFrame(columns=["price", "stock_only", "overlay", "difference"])

    spot = _f(result.spot)
    entry = _f(entry_price) if entry_price is not None else spot
    if not math.isfinite(spot) or spot <= 0 or not math.isfinite(entry):
        return pd.DataFrame(columns=["price", "stock_only", "overlay", "difference"])

    width = max(min(_f(span), 0.9), 0.05)
    grid = list(np.linspace(spot * (1.0 - width), spot * (1.0 + width), max(int(points), 11)))
    for quote in (result.call, result.put):
        if quote is not None and math.isfinite(_f(quote.strike)):
            grid.append(_f(quote.strike))
    prices = sorted({round(float(p), 6) for p in grid if p > 0})

    rows = [
        {
            "price": price,
            "stock_only": result.shares * (price - entry),
            "overlay": payoff_at_expiry(
                price,
                entry,
                result.shares,
                short_call=result.call,
                long_put=result.put,
                contracts=result.contracts,
            ),
        }
        for price in prices
    ]
    frame = pd.DataFrame(rows)
    frame["difference"] = frame["overlay"] - frame["stock_only"]
    return frame


def overlay_comparison(results: list[OverlayResult]) -> pd.DataFrame:
    """One row per structure, for a side-by-side decision.

    Deliberately includes ``max_loss`` next to the yield figures. An income
    table showing only premium and static return invites picking the biggest
    number, and the biggest number usually belongs to the structure with the
    least protection.
    """
    usable = [r for r in results if r is not None and r.ok]
    columns = [
        "strategy",
        "net_premium_per_share",
        "net_premium_total",
        "static_return_pct",
        "static_return_annual_pct",
        "breakeven",
        "max_profit",
        "max_loss",
        "floor_price",
        "upside_cap_pct",
        "probability_assigned_pct",
        "days_to_expiry",
    ]
    if not usable:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [
            {
                "strategy": r.strategy,
                "net_premium_per_share": r.net_premium_per_share,
                "net_premium_total": r.net_premium_total,
                "static_return_pct": r.static_return * 100.0,
                "static_return_annual_pct": r.static_return_annual * 100.0,
                "breakeven": r.breakeven,
                "max_profit": r.max_profit,
                "max_loss": r.max_loss,
                "floor_price": r.floor_price,
                "upside_cap_pct": r.upside_cap * 100.0 if math.isfinite(r.upside_cap) else _NAN,
                "probability_assigned_pct": r.probability_assigned * 100.0
                if math.isfinite(r.probability_assigned)
                else _NAN,
                "days_to_expiry": r.days_to_expiry,
            }
            for r in usable
        ],
        columns=columns,
    )
