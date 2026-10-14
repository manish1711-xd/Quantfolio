"""Options overlay UI: price a covered call or protective put on a holding.

Mounted by the Security deep dive page, which already has a ticker selected
and its price loaded.

The shape of this panel is dictated by one fact: **Yahoo Finance publishes no
option chain for NSE or BSE listings.** AAPL returns 21 expiries;
``RELIANCE.NS`` and ``^NSEI`` return zero. For an India-first application the
primary market has no chain data at all, so manual entry is not a fallback
here — it is the main path, and the chain picker is the convenience that
happens to work for US names.

That is why the panel opens on "enter the contract" rather than "pick from the
chain", and why the unavailability message names the reason instead of showing
an empty table that would read as "this stock has no options".
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics import options as op
from src.data.fetcher import option_chain_available
from src.ui import components as ui
from src.ui import data_access
from src.ui import formatters as fmt

__all__ = ["render"]

_MANUAL = "Enter the contract manually"
_CHAIN = "Pick from the live chain"

_COVERED_CALL = "Covered call"
_PROTECTIVE_PUT = "Protective put"
_COLLAR = "Collar (both)"


def _f(value: object) -> float:
    """Finite float, or NaN. Local so the UI never reaches into a private
    helper of the analytics module."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _shares_held(result: object, ticker: str) -> float:
    """Shares of ``ticker`` in the portfolio, 0.0 when not held."""
    positions = getattr(result, "positions", None)
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return 0.0
    if "ticker" not in positions.columns or "quantity" not in positions.columns:
        return 0.0
    match = positions.loc[positions["ticker"].astype(str) == str(ticker), "quantity"]
    if match.empty:
        return 0.0
    try:
        value = float(match.iloc[0])
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _chain_source(
    ticker: str, spot: float, contract_size: int
) -> tuple[op.OptionQuote | None, op.OptionQuote | None, list[str]]:
    """Let the user pick a call and a put from the live chain, if there is one."""
    expiries, reason = data_access.load_option_expiries(ticker)
    if not expiries:
        st.info(reason or f"No option expiries are available for {ticker}.")
        st.caption(
            "Switch to **Enter the contract manually** above — every calculation on "
            "this panel works from a strike, a premium and an expiry, which your "
            "broker's option screen already shows you. The premium is the only figure "
            "that genuinely needs a live market."
        )
        return None, None, []

    expiry = st.selectbox("Expiry", expiries, key="opt_expiry")
    with st.spinner(f"Loading the {expiry} chain…"):
        chain, chain_reason = data_access.load_option_chain(ticker, str(expiry))
    if chain.empty:
        st.warning(chain_reason or "The chain came back empty.")
        return None, None, []

    st.caption(
        "Premia below are the provider's own bid/ask and last-trade figures, "
        "untouched — no midpoint substitution and no filling a missing bid with a "
        "stale trade. **US equity options are American-style**, so the Black-Scholes "
        "Greeks shown here slightly understate the value of an in-the-money put and "
        "of a call on a stock about to pay a large dividend, both of which can be "
        "worth exercising early."
    )

    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]

    def pick(side: pd.DataFrame, label: str, kind: str) -> op.OptionQuote | None:
        if side.empty:
            return None
        strikes = [float(s) for s in side["strike"]]
        # Default to the first strike above spot for a call, below for a put:
        # the out-of-the-money contract is what an overlay normally uses.
        if kind == op.CALL:
            candidates = [s for s in strikes if s >= spot] or strikes
            default = strikes.index(candidates[0])
        else:
            candidates = [s for s in strikes if s <= spot] or strikes
            default = strikes.index(candidates[-1])

        strike = st.selectbox(
            label,
            strikes,
            index=default,
            key=f"opt_strike_{kind}",
            format_func=lambda s: f"{s:,.2f}",
        )
        row = side.loc[side["strike"] == strike].iloc[0]
        premium = _f(row.get("last_price"))
        bid, ask = _f(row.get("bid")), _f(row.get("ask"))
        if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid:
            premium = (bid + ask) / 2.0
        return op.OptionQuote(
            option_type=kind,
            strike=float(strike),
            premium=premium,
            expiry=pd.Timestamp(str(expiry)).date(),
            implied_vol=_f(row.get("implied_vol")),
            bid=bid,
            ask=ask,
            open_interest=_f(row.get("open_interest")),
            volume=_f(row.get("volume")),
            contract_size=contract_size,
            source="yahoo-chain",
        )

    left, right = st.columns(2)
    with left:
        call = pick(calls, "Call strike", op.CALL)
    with right:
        put = pick(puts, "Put strike", op.PUT)
    return call, put, ["Premia taken as the bid-ask midpoint where both are quoted."]


def _manual_source(
    spot: float, contract_size: int, want_call: bool, want_put: bool
) -> tuple[op.OptionQuote | None, op.OptionQuote | None]:
    """Type the contract in from a broker screen. The primary path for NSE."""
    st.caption(
        "Take these four numbers off your broker's option chain. Everything below is "
        "computed from them — no market data required."
    )
    expiry = st.date_input(
        "Expiry",
        value=date.today() + timedelta(days=30),
        min_value=date.today(),
        key="opt_manual_expiry",
        help="Indian equity and index options are EUROPEAN-style, so the "
        "Black-Scholes model used here is exactly the right one for them.",
    )

    call = put = None
    if want_call:
        c1, c2 = st.columns(2)
        strike = c1.number_input(
            "Call strike",
            min_value=0.01,
            value=float(round(spot * 1.05, 2)) if spot > 0 else 100.0,
            step=1.0,
            format="%.2f",
            key="opt_manual_call_strike",
        )
        premium = c2.number_input(
            "Call premium received (per share)",
            min_value=0.0,
            value=float(round(max(spot, 1.0) * 0.02, 2)),
            step=0.05,
            format="%.2f",
            key="opt_manual_call_premium",
        )
        call = op.OptionQuote(
            op.CALL,
            float(strike),
            float(premium),
            expiry,
            contract_size=contract_size,
            source="manual",
        )

    if want_put:
        p1, p2 = st.columns(2)
        strike = p1.number_input(
            "Put strike",
            min_value=0.01,
            value=float(round(spot * 0.95, 2)) if spot > 0 else 100.0,
            step=1.0,
            format="%.2f",
            key="opt_manual_put_strike",
        )
        premium = p2.number_input(
            "Put premium paid (per share)",
            min_value=0.0,
            value=float(round(max(spot, 1.0) * 0.018, 2)),
            step=0.05,
            format="%.2f",
            key="opt_manual_put_premium",
        )
        put = op.OptionQuote(
            op.PUT,
            float(strike),
            float(premium),
            expiry,
            contract_size=contract_size,
            source="manual",
        )
    return call, put


def _kpis(result: op.OverlayResult, currency: str) -> None:
    premium_tone = "gain" if result.is_income else "warn"
    cards = [
        {
            "label": "Net premium" if result.is_income else "Net cost",
            "value": fmt.fmt_money(abs(result.net_premium_total), currency),
            "tone": premium_tone,
            "footnote": f"{fmt.fmt_money(abs(result.net_premium_per_share), currency)} "
            f"per share × {result.contracts} contract(s)",
        },
        {
            "label": "Static return",
            "value": fmt.fmt_ratio_pct(result.static_return, signed=True),
            "tone": premium_tone,
            "delta": f"{fmt.fmt_ratio_pct(result.static_return_annual, signed=True)} p.a.",
            "delta_tone": premium_tone,
            "footnote": f"if unchanged over {result.days_to_expiry} days",
            "help_text": "Annualised by COMPOUNDING the holding-period return, which is "
            "the right convention for a structure written repeatedly.",
        },
        {
            "label": "Breakeven",
            "value": fmt.fmt_money(result.breakeven, currency),
            "tone": "neutral",
            "footnote": f"spot {fmt.fmt_money(result.spot, currency)}",
        },
        {
            "label": "Max loss",
            "value": fmt.fmt_money(result.max_loss, currency),
            "tone": "loss",
            "footnote": "per share",
            "help_text": "A covered call is NOT limited risk: the underlying can go to "
            "zero and the premium is all you keep.",
        },
    ]
    ui.kpi_row(cards, columns=4)


def _verdict(result: op.OverlayResult, currency: str) -> None:
    """One paragraph naming the trade-off in plain terms."""
    if result.strategy == _COVERED_CALL:
        st.markdown(
            f"Writing this call collects **{fmt.fmt_money(abs(result.net_premium_total), currency)}** "
            f"today. In exchange the position stops participating above "
            f"**{fmt.fmt_money(result.call.strike, currency)}** — "
            f"{fmt.fmt_ratio_pct(result.upside_cap)} above spot — and the risk-neutral "
            f"chance of being assigned is **{fmt.fmt_ratio_pct(result.probability_assigned)}**. "
            f"If assigned, the total return is "
            f"{fmt.fmt_ratio_pct(result.if_called_return, signed=True)} "
            f"({fmt.fmt_ratio_pct(result.if_called_return_annual, signed=True)} annualised)."
        )
    elif result.strategy == _PROTECTIVE_PUT:
        st.markdown(
            f"This put costs **{fmt.fmt_money(abs(result.net_premium_total), currency)}** and "
            f"puts a floor of **{fmt.fmt_money(result.floor_price, currency)}** per share "
            f"under the position — {fmt.fmt_ratio_pct(result.downside_protection)} below "
            f"spot. The upside stays unbounded. The cost is a permanent drag of "
            f"{fmt.fmt_ratio_pct(abs(result.static_return_annual))} a year if the "
            "protection is never needed, which is what insurance is."
        )
    else:
        st.markdown(
            f"The call pays for the put: net "
            f"{'credit' if result.is_income else 'cost'} of "
            f"**{fmt.fmt_money(abs(result.net_premium_total), currency)}**. The position "
            f"is bounded at both ends — floor "
            f"{fmt.fmt_money(result.floor_price, currency)}, cap "
            f"{fmt.fmt_money(result.call.strike, currency)} — which is the only structure "
            "here with a limited outcome in both directions. The price of that is the "
            f"{fmt.fmt_ratio_pct(result.upside_cap)} of upside above the call strike."
        )


def _payoff_chart(result: op.OverlayResult, entry: float) -> None:
    curve = op.payoff_curve(result, entry_price=entry)
    if curve.empty:
        return
    st.markdown("**Profit at expiry: overlay vs holding the shares alone**")
    st.line_chart(
        curve.set_index("price")[["stock_only", "overlay"]],
        height=320,
    )
    st.caption(
        "The comparison is the point — an overlay payoff on its own says nothing about "
        "what it cost. Where the lines diverge is exactly what the premium bought or "
        "sold. Strikes are forced onto the grid so the kink lands on them."
    )


def _greeks_table(result: op.OverlayResult) -> None:
    rows: list[dict[str, object]] = []
    for label, greeks, quote in (
        ("Short call", result.greeks_call, result.call),
        ("Long put", result.greeks_put, result.put),
    ):
        if greeks is None or quote is None:
            continue
        rows.append(
            {
                "Leg": label,
                "Strike": quote.strike,
                "Delta": greeks.delta,
                "Gamma": greeks.gamma,
                "Vega (per 1% vol)": greeks.vega,
                "Theta (per day)": greeks.theta,
                "Rho (per 1% rate)": greeks.rho,
            }
        )
    if not rows:
        return
    with st.expander("Greeks", expanded=False):
        ui.dataframe(
            pd.DataFrame(rows),
            column_config={
                "Strike": st.column_config.NumberColumn(format="%.2f"),
                "Delta": st.column_config.NumberColumn(format="%.4f"),
                "Gamma": st.column_config.NumberColumn(format="%.5f"),
                "Vega (per 1% vol)": st.column_config.NumberColumn(format="%.4f"),
                "Theta (per day)": st.column_config.NumberColumn(format="%.4f"),
                "Rho (per 1% rate)": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "Per SHARE and per CONTRACT of one share — multiply by the contract size "
            "and the contract count for the position's exposure. Delta on a short call "
            "works against the stock leg: a delta of 0.35 means the overlay has cut the "
            "position's effective share exposure by 35% of the covered quantity."
        )


def _methodology(result: op.OverlayResult) -> None:
    ui.methodology(
        "The model, and where it is wrong",
        f"""
**Pricing.** Black-Scholes-Merton for a EUROPEAN option:

```
d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),   d2 = d1 − σ√T
call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

Time is measured in **calendar days over 365**, not the 252 trading days every
other module here uses — an option decays over a weekend.

**European vs American, and the irony.** NSE stock and index options are
European-style, so this model is *exactly* right for the market this application
targets. US equity options are American, so for the market whose chains Yahoo
actually publishes it is an approximation: it undervalues an in-the-money put and
a call on a stock about to pay a large dividend, because both can be worth
exercising early.

**"Probability of assignment" is risk-neutral.** It is the `N(d2)` that falls out
of the same model that priced the option, under the measure in which every asset
drifts at the risk-free rate. It is **not a forecast**. Under the real-world
measure an equity drifts faster, so the true chance of a call finishing in the
money is higher than the figure shown.

**Volatility.** {"Taken from the chain's implied volatility." if result.call and result.call.source == "yahoo-chain" else "Solved from the premium you entered by Brent bracketing, so the Greeks and the assignment probability are consistent with the price you are actually being quoted."}

**What is not modelled**

- Early exercise (American options), and dividend-driven early assignment
- The volatility smile: one implied volatility per contract, no surface
- Transaction costs, STT/CTT, exchange fees and the bid-ask spread you actually cross
- Margin: a short call against stock is covered, but your broker may still block
  the position or demand margin depending on how the holding is pledged
- Assignment mechanics, physical settlement and the corporate actions that adjust
  strikes
- Rolling: every figure is for holding this contract to expiry
""",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(result: object, ticker: str, spot: float, currency: str = "INR") -> None:
    """Mount the panel for one underlying."""
    price = _f(spot)
    if not math.isfinite(price) or price <= 0:
        st.caption("No current price is available for this instrument.")
        return

    held = _shares_held(result, ticker)
    available, reason = option_chain_available(ticker)

    st.caption(
        f"Overlay analytics for **{ticker}** at {fmt.fmt_money(price, currency)}"
        + (
            f" · you hold {held:,.0f} shares"
            if held > 0
            else " · you do not hold this instrument, so the figures below are per "
            "hypothetical share block"
        )
    )
    if not available:
        st.warning(reason)

    top = st.columns(3)
    strategy = top[0].selectbox(
        "Structure",
        [_COVERED_CALL, _PROTECTIVE_PUT, _COLLAR],
        key="opt_strategy",
    )
    contract_size = top[1].number_input(
        "Shares per contract",
        min_value=1,
        value=op.US_CONTRACT_SIZE if available else 1,
        step=1,
        key="opt_contract_size",
        help="US equity options are 100 shares. **NSE lot sizes vary per underlying "
        "and are revised periodically by the exchange**, so there is no safe default "
        "for India — take it from the contract specification.",
    )
    shares = top[2].number_input(
        "Shares available",
        min_value=0.0,
        value=float(held) if held > 0 else float(contract_size),
        step=float(contract_size),
        key="opt_shares",
        help="Defaults to your holding. A covered call is only covered if the shares "
        "are actually there.",
    )

    source = st.radio(
        "Contract details",
        [_MANUAL, _CHAIN] if not available else [_CHAIN, _MANUAL],
        horizontal=True,
        key="opt_source",
        help="Manual entry works for every market. The live chain works for US "
        "listings only — this provider carries no NSE or BSE chains.",
    )

    want_call = strategy in (_COVERED_CALL, _COLLAR)
    want_put = strategy in (_PROTECTIVE_PUT, _COLLAR)

    if source == _CHAIN:
        call, put, notes = _chain_source(ticker, price, int(contract_size))
        for note in notes:
            st.caption(note)
    else:
        call, put = _manual_source(price, int(contract_size), want_call, want_put)

    if want_call and call is None:
        return
    if want_put and put is None:
        return

    rate = float(getattr(getattr(result, "config", None), "risk_free_rate", 0.0) or 0.0)
    dividend = 0.0

    if strategy == _COVERED_CALL:
        overlay = op.covered_call(price, shares, call, rate=rate, dividend=dividend)
    elif strategy == _PROTECTIVE_PUT:
        overlay = op.protective_put(price, shares, put, rate=rate, dividend=dividend)
    else:
        overlay = op.collar(price, shares, call, put, rate=rate, dividend=dividend)

    if not overlay.ok:
        ui.empty_state(
            "Overlay not priced",
            "; ".join(overlay.warnings) or "The contract details are incomplete.",
            icon="🧮",
            hint=f"A single contract of {contract_size} needs at least that many shares.",
        )
        return

    ui.warning_list(list(overlay.warnings), "Things to check before trading this")
    _verdict(overlay, currency)
    _kpis(overlay, currency)
    ui.dataframe(overlay.summary_frame(), height=None)
    _payoff_chart(overlay, price)
    _greeks_table(overlay)
    _methodology(overlay)
    st.caption(
        "⚠️ Educational analytics. Options can lose more than their premium, assignment "
        "can be early and involuntary, and none of this is a recommendation to trade."
    )
