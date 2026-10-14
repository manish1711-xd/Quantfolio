"""Security deep dive -- one instrument, read off its own price history.

Every other page reasons about the portfolio as a whole. This one zooms all the
way in and answers four questions about a single name: what the instrument is
and what the provider says about its fundamentals, what the position is doing
inside the book, what the price itself has actually been doing, and which
mechanical chart patterns are currently firing.

This is the only page permitted to fetch its own market data, and it fetches
exactly one thing the pipeline never collects: full OHLCV bars for a single
ticker. Candles, RSI, MACD, Bollinger bands and ATR all need High/Low/Volume,
and the portfolio pipeline stores adjusted closes only. That same allowance is
what lets this page research a symbol the portfolio does not own.

Everything portfolio-level -- weight, P&L, risk contribution, the profile of a
held name -- is still read straight off the :class:`~src.models.AnalysisResult`.
Indicator maths is delegated to :mod:`src.analytics.technicals`; nothing beyond
trivial display arithmetic on the returned indicator frame happens here.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import technicals as tec
from src.config import (
    CURRENCY_SYMBOLS,
    DATA_SOURCE,
    PERIOD_LABELS,
    QUOTE_TTL_SECONDS,
    TECHNICALS,
    TRADING_DAYS,
)
from src.ui import components as ui
from src.ui import data_access, fundamentals_panel, guards, options_panel, session
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    delta_tone,
    fmt_compact,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    fmt_shares,
)

#: Height of the stacked price / RSI / MACD figure. Three panels need the room;
#: the chart layer's own default (680) squeezes the oscillators.
_PANEL_HEIGHT = 760

#: Sessions used for the "volume against its own norm" comparison in the
#: caption. Matches the averaging window ``charts.volume_chart`` overlays.
_VOLUME_WINDOW = 20

#: Below this many sessions the trailing "52-week" range is not really a year,
#: and the page says so instead of quietly relabelling the window.
_FULL_YEAR_SESSIONS = 200


# ---------------------------------------------------------------------------
# small display helpers (no analytics happen in this module)
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    """True when ``x`` is a real, finite number worth printing."""
    try:
        return bool(pd.notna(x)) and float(x) == float(x) and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def _text(value: object) -> str:
    """Stringify a possibly-missing reference field, collapsing blanks to a dash."""
    if value is None:
        return DASH
    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return DASH
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.upper() not in {"NAN", "NONE", "NAT"} else DASH


def _last(frame: pd.DataFrame | None, column: str) -> float:
    """Last finite value in ``column``, or NaN when the column never resolved."""
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return float("nan")
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series.iloc[-1]) if not series.empty else float("nan")


def _gap_pct(price: Any, level: Any) -> float:
    """Distance of ``price`` above/below ``level`` in percent."""
    if not (_finite(price) and _finite(level)) or float(level) == 0.0:
        return float("nan")
    return (float(price) / float(level) - 1.0) * 100.0


def _owned_tickers(result: Any) -> list[str]:
    positions = result.positions
    if positions is None or len(positions) == 0 or "ticker" not in positions.columns:
        return []
    return [str(t) for t in positions["ticker"].tolist()]


def _row_for(result: Any, ticker: str) -> pd.Series | None:
    """The positions row for ``ticker``, or None when it is not held/priced."""
    positions = result.positions
    if positions is None or len(positions) == 0 or "ticker" not in positions.columns:
        return None
    match = positions.loc[positions["ticker"].astype(str) == str(ticker)]
    return None if match.empty else match.iloc[0]


def _native_currency(profile: Any, row: pd.Series | None, cfg: Any) -> str:
    """Currency the instrument itself quotes in; base currency as last resort."""
    for candidate in (
        getattr(profile, "currency", None),
        row.get("currency") if row is not None else None,
    ):
        text = _text(candidate)
        if text != DASH:
            return text.upper()
    return str(getattr(cfg, "base_currency", "INR"))


# ---------------------------------------------------------------------------
# 1. ticker selection (the one place this page is allowed to touch the network)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=QUOTE_TTL_SECONDS, show_spinner=False)
def _validate(symbol: str) -> tuple[bool, str]:
    """Cached ``fetcher.validate_ticker`` -- one quote request per symbol per TTL."""
    from src.data import fetcher

    return fetcher.validate_ticker(symbol)


def _pick_ticker(result: Any) -> str | None:
    """Resolve the security under study, persisting it for the other pages."""
    owned = _owned_tickers(result)
    left, right = st.columns([1, 1], gap="large")

    picked: str | None = None
    with left:
        if owned:
            preferred = session.selected_ticker(owned[0])
            index = owned.index(preferred) if preferred in owned else 0
            picked = str(
                st.selectbox(
                    "Holding",
                    options=owned,
                    index=index,
                    key="security_holding",
                    help="Any name from the analysed portfolio. Position economics "
                    "and the risk split are shown for these.",
                )
            )
        else:
            st.caption(
                "This run produced no priced holdings, so there is nothing to pick "
                "from. Type a symbol on the right instead."
            )

    with right:
        typed = st.text_input(
            "Or research any symbol",
            value="",
            placeholder="AAPL, TCS.NS, GOLDBEES.NS",
            key="security_symbol",
            help="Anything the data provider prices, held or not. Indian equities "
            "need a .NS (NSE) or .BO (BSE) suffix; US symbols take none.",
        )

    symbol = str(typed or "").strip().upper()
    if symbol:
        ok, message = _validate(symbol)
        if ok:
            picked = symbol
            st.caption(f"{message}")
        else:
            st.error(f"Cannot research {symbol}. {message}")
            if picked:
                st.caption(
                    f"Falling back to {picked} from the portfolio. Clear the search "
                    f"box to dismiss this."
                )

    if picked:
        session.select_ticker(picked)
    return picked


# ---------------------------------------------------------------------------
# 2. identity strip
# ---------------------------------------------------------------------------
def _profile_for(result: Any, ticker: str) -> Any:
    """The pipeline's profile for a held name; a fresh fetch for anything else."""
    profile = (result.profiles or {}).get(ticker)
    if profile is not None:
        return profile
    try:
        return data_access.load_profiles((ticker,)).get(ticker)
    except Exception:  # noqa: BLE001 - reference data must never break the page
        return None


def _identity(ticker: str, profile: Any, ohlcv: pd.DataFrame, cfg: Any, held: bool) -> None:
    name = _text(getattr(profile, "display_name", None))
    st.markdown(f"### {ticker}" + (f" - {name}" if name != DASH else ""))

    window = PERIOD_LABELS.get(cfg.period, cfg.period)
    first, last = ohlcv.index[0], ohlcv.index[-1]
    chips: list[tuple[str, str]] = [
        (f"Sector {_text(getattr(profile, 'sector_label', None))}", "neutral"),
        (f"Industry {_text(getattr(profile, 'industry', None))}", "neutral"),
        (f"Exchange {_text(getattr(profile, 'exchange', None))}", "neutral"),
        (f"Instrument {_text(getattr(profile, 'quote_type', None))}", "muted"),
        (
            "Held in this portfolio" if held else "Not held - research only",
            "accent" if held else "warn",
        ),
        (f"{len(ohlcv):,} sessions, {window}", "muted"),
    ]
    ui.badges(chips)

    st.caption(
        f"Bars from {first:%d %b %Y} to {last:%d %b %Y}, split- and "
        f"dividend-adjusted. Everything below this line is computed from those "
        f"{len(ohlcv):,} sessions and nothing else -- change the history window in "
        f"the sidebar and every indicator on the page changes with it."
    )


# ---------------------------------------------------------------------------
# 3. fundamentals
# ---------------------------------------------------------------------------
def _fundamentals(profile: Any, ticker: str, native_ccy: str) -> None:
    ui_theme.section("FUNDAMENTALS", "🏛")

    if profile is None:
        ui.empty_state(
            "No reference data returned",
            f"The provider gave back nothing for {ticker}. Fundamentals are routinely "
            "absent for ETFs, index funds, newly listed names and thinly covered "
            "small caps -- the price history below is unaffected.",
            icon="📭",
        )
        return

    stats: list[tuple[str, str]] = [
        ("Market cap", fmt_compact(getattr(profile, "market_cap", None), native_ccy)),
        ("Trailing P/E", fmt_num(getattr(profile, "trailing_pe", None), 1)),
        ("Forward P/E", fmt_num(getattr(profile, "forward_pe", None), 1)),
        ("Price / book", fmt_num(getattr(profile, "price_to_book", None), 2)),
        ("Return on equity", fmt_ratio_pct(getattr(profile, "return_on_equity", None), 1)),
        ("Debt / equity (%)", fmt_num(getattr(profile, "debt_to_equity", None), 1)),
        ("Profit margin", fmt_ratio_pct(getattr(profile, "profit_margin", None), 1)),
        ("Revenue growth", fmt_ratio_pct(getattr(profile, "revenue_growth", None), 1, signed=True)),
        ("Dividend yield", fmt_ratio_pct(getattr(profile, "dividend_yield", None))),
        ("Beta (provider)", fmt_num(getattr(profile, "beta", None), 2)),
    ]
    ui.stat_grid(stats, columns=5)

    missing = sum(1 for _, value in stats if value == DASH)
    if missing:
        st.caption(
            f"{missing} of {len(stats)} fields came back empty and render as a dash. "
            "A dash is the honest answer: a P/E of 0.00 on a loss-making company, or "
            "a 0.00% yield on a fund that simply does not report one, would be a "
            "fabricated number sitting next to real ones."
        )

    summary_text = getattr(profile, "summary", None)
    if summary_text:
        with st.expander("Business description (provider text, unedited)"):
            st.write(str(summary_text))

    ui.methodology(
        "What each fundamental is, and its units",
        f"""
Every figure above is the data provider's, point in time, as-is -- nothing is
derived, interpolated or back-filled here. Values are quoted in
**{native_ccy}**, the instrument's own reporting currency, not the portfolio's
base currency: mixing a market cap into a rupee book at today's spot rate would
imply an FX view the balance sheet never had.

* **Market cap** is shares outstanding x last price. For an ETF the provider
  substitutes total net assets, which is a fund size, not an equity valuation.
* **Trailing P/E** uses the last four reported quarters of earnings and is
  undefined (dash) for a loss-making company. **Forward P/E** uses the analyst
  consensus estimate -- a forecast, and the only forward-looking number on this
  page. Neither means anything for a fund or an index.
* **Price / book** compares market value with accounting equity. Informative for
  banks and asset-heavy businesses; close to meaningless for asset-light ones
  whose value is intangible and therefore off the balance sheet.
* **Return on equity** and **profit margin** are stored as decimals and shown as
  percentages. ROE is flattered by leverage -- read it beside debt / equity, not
  instead of it.
* **Debt / equity** is left exactly as the provider reports it: total debt as a
  **percentage of equity**, so 82.4 means 0.82x, not 82x. It is a snapshot of
  the last reported balance sheet, which can be several months stale.
* **Revenue growth** is the latest reported quarter against the same quarter a
  year earlier -- one observation, not a trend.
* **Dividend yield** is stored as a decimal (0.012) and rendered as 1.20%. It is
  the trailing twelve-month cash yield on today's price, not a declared or
  forward rate; a falling price mechanically raises it.
* **Beta (provider)** is the vendor's own five-year monthly regression against
  its own chosen index. It is *not* the beta on the Benchmark page, which is
  measured on daily returns against the benchmark you selected over this
  analysis window. When the two disagree, they are answering different
  questions.
        """,
    )


# ---------------------------------------------------------------------------
# 4. position economics (held names only)
# ---------------------------------------------------------------------------
def _risk_row(result: Any, ticker: str) -> pd.Series | None:
    frame = (result.attribution or {}).get("risk_contribution")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "ticker" not in frame.columns:
        return None
    match = frame.loc[frame["ticker"].astype(str) == str(ticker)]
    return None if match.empty else match.iloc[0]


def _position_economics(result: Any, ticker: str, row: pd.Series, cfg: Any, native: str) -> None:
    ui_theme.section("THIS POSITION IN THE BOOK", "💼")

    base = str(getattr(cfg, "base_currency", "INR"))

    stats: list[tuple[str, str]] = [
        ("Quantity", fmt_shares(row.get("quantity"))),
        ("Average cost", fmt_money(row.get("buy_price"), native, 2)),
        ("Last price", fmt_money(row.get("current_price"), native, 2)),
        ("Day move", fmt_pct(row.get("day_change_pct"), signed=True)),
        ("Invested", fmt_money(row.get("investment"), base)),
        ("Market value", fmt_money(row.get("current_value"), base)),
        ("Unrealised P&L", fmt_money(row.get("pnl"), base, signed=True)),
        ("Return", fmt_pct(row.get("pnl_pct"), signed=True)),
        ("Portfolio weight", fmt_pct(row.get("weight_pct"))),
        ("FX rate to base", fmt_num(row.get("fx_rate"), 4)),
    ]
    ui.stat_grid(stats, columns=5)

    risk = _risk_row(result, ticker)
    if risk is None:
        st.caption(
            "No risk decomposition is available for this name -- the covariance "
            "matrix needs overlapping daily history for every holding, and this run "
            "did not produce one."
        )
        return

    weight = risk.get("weight_pct")
    contribution = risk.get("risk_contribution_pct")
    ratio = (
        float(contribution) / float(weight)
        if _finite(weight) and _finite(contribution) and float(weight) != 0.0
        else float("nan")
    )

    ui.kpi_row(
        [
            {
                "label": "Capital weight",
                "value": fmt_pct(weight),
                "tone": "neutral",
                "footnote": "Share of portfolio market value",
                "help_text": "Market value of this position over total market value.",
            },
            {
                "label": "Risk contribution",
                "value": fmt_pct(contribution),
                "tone": "neutral",
                "footnote": "Share of portfolio volatility",
                "help_text": "Euler decomposition of portfolio volatility. The column "
                "sums to exactly 100% across holdings.",
            },
            {
                "label": "Risk / weight",
                "value": fmt_num(ratio, 2) if _finite(ratio) else DASH,
                "tone": ("muted" if not _finite(ratio) else ("warn" if ratio > 1.0 else "gain")),
                "footnote": "Above 1.0 = costs more risk than its size",
                "help_text": "Risk contribution divided by capital weight.",
            },
            {
                "label": "Marginal contribution",
                "value": fmt_ratio_pct(risk.get("mctr")),
                "tone": "neutral",
                "footnote": "Annualised vol added per unit of extra weight",
                "help_text": "d(sigma_p) / d(w_i) -- the sensitivity of portfolio "
                "volatility to one more unit of this name.",
            },
        ],
        columns=4,
    )

    if _finite(ratio):
        if ratio > 1.0:
            reading = (
                f"carries {fmt_pct(contribution)} of the portfolio's volatility on "
                f"{fmt_pct(weight)} of its capital -- a ratio of {fmt_num(ratio, 2)}, so "
                f"this name is punching above its size. That happens when it is more "
                f"volatile than the book, more correlated with the rest of it, or both"
            )
        else:
            reading = (
                f"carries {fmt_pct(contribution)} of the portfolio's volatility on "
                f"{fmt_pct(weight)} of its capital -- a ratio of {fmt_num(ratio, 2)}, so "
                f"it is diluting risk rather than adding it, either through lower "
                f"volatility or through low correlation with everything else"
            )
        st.caption(f"{ticker} {reading}.")

    if native == base:
        currency_note = (
            f"**One currency throughout.** The instrument quotes in {native} and the "
            f"portfolio reports in {native}, so `fx` is 1.0 on every line and nothing "
            f"above was converted. **Return** is `last price / average cost - 1` "
            f"exactly."
        )
    else:
        currency_note = (
            f"**Prices stay native, money converts.** Average cost and last price are "
            f"shown in {native} because that is the number the holder recognises; only "
            f"*Invested*, *Market value* and *Unrealised P&L* are translated into "
            f"{base}. Both P&L legs convert at the same rate, so **Return** is "
            f"FX-neutral and equals `last price / average cost - 1` exactly."
        )

    ui.methodology(
        "Position arithmetic and the risk split",
        f"""
With `fx` the value of one unit of {native} in {base}:

```
invested      = quantity x average cost x fx
market value  = quantity x last price   x fx
unrealised    = market value - invested
return %      = unrealised / invested x 100
weight %      = market value / total market value x 100
```

{currency_note}

**The risk split is exact, not approximate.** With `S` the annualised covariance
matrix and `w` the normalised weight vector:

```
sigma_p = sqrt(w' S w)
MCTR_i  = (S w)_i / sigma_p          marginal contribution
CCTR_i  = w_i x MCTR_i               component contribution
rc_i    = CCTR_i / sigma_p x 100     [%], and sum(rc_i) = 100
```

Because `sigma_p` is homogeneous of degree one in `w`, Euler's theorem makes the
decomposition add up exactly -- a 5% position in a volatile, highly correlated
name genuinely can own 15% of the risk, and that gap is the entire point of the
table.

**Caveats.** The covariance matrix is estimated from the daily returns inside
this analysis window, so the split is backward-looking and it moves when the
window moves. It also describes *volatility*, which is a symmetric measure: a
name that only ever gapped upward still books risk here. Unrealised P&L excludes
dividends, brokerage, STT and taxes.
        """,
    )


# ---------------------------------------------------------------------------
# 5. price, momentum and trend
# ---------------------------------------------------------------------------
def _indicator_readings(ind: pd.DataFrame, close: float, native_ccy: str) -> None:
    rsi = _last(ind, "RSI")
    hist = _last(ind, "MACD_HIST")
    percent_b = _last(ind, "BB_PERCENT_B")
    bandwidth = _last(ind, "BB_BANDWIDTH")
    atr = _last(ind, "ATR")

    atr_pct = (
        float(atr) / float(close) * 100.0
        if _finite(atr) and _finite(close) and float(close) != 0.0
        else float("nan")
    )

    stats: list[tuple[str, str]] = [
        (f"RSI ({TECHNICALS.rsi_period})", fmt_num(rsi, 1)),
        ("MACD histogram", fmt_num(hist, 3)),
        ("Bollinger %B", fmt_ratio_pct(percent_b, 0)),
        ("Bollinger bandwidth", fmt_pct(bandwidth, 1)),
        (f"ATR ({TECHNICALS.atr_period})", fmt_money(atr, native_ccy, 2)),
        ("ATR / price", fmt_pct(atr_pct, 2)),
    ]
    for window in TECHNICALS.sma_windows:
        stats.append(
            (
                f"Price vs SMA{window}",
                fmt_pct(_gap_pct(close, _last(ind, f"SMA{window}")), 1, signed=True),
            )
        )
    ui.stat_grid(stats, columns=5)


def _rsi_caption(rsi: float) -> str:
    if not _finite(rsi) or not 0.0 < float(rsi) < 100.0:
        return (
            f"RSI is undefined for this window -- it needs at least "
            f"{TECHNICALS.rsi_period} sessions of price changes with both gains and "
            f"losses in them."
        )
    strength = float(rsi) / (100.0 - float(rsi))
    if float(rsi) >= TECHNICALS.rsi_overbought:
        posture = (
            f"above the conventional overbought line at {fmt_num(TECHNICALS.rsi_overbought, 0)}"
        )
    elif float(rsi) <= TECHNICALS.rsi_oversold:
        posture = f"below the conventional oversold line at {fmt_num(TECHNICALS.rsi_oversold, 0)}"
    else:
        posture = (
            f"between the conventional {fmt_num(TECHNICALS.rsi_oversold, 0)} / "
            f"{fmt_num(TECHNICALS.rsi_overbought, 0)} lines"
        )
    return (
        f"RSI {fmt_num(rsi, 1)}, {posture}: over the last "
        f"{TECHNICALS.rsi_period} sessions the Wilder-smoothed average up-move is "
        f"{fmt_num(strength, 2)}x the average down-move. Those lines are a 1978 "
        f"convention, not a law -- a strong trend can sit above 70 for months while "
        f"it keeps working."
    )


def _macd_caption(ind: pd.DataFrame) -> str:
    macd = _last(ind, "MACD")
    signal = _last(ind, "MACD_SIGNAL")
    hist = _last(ind, "MACD_HIST")
    if not (_finite(macd) and _finite(signal) and _finite(hist)):
        return (
            f"MACD is undefined here: the {TECHNICALS.macd_slow}-day EMA and its "
            f"{TECHNICALS.macd_signal}-day signal line both need more sessions than "
            f"this window contains before either settles."
        )
    trend = "above" if macd >= 0 else "below"
    momentum = "above" if hist >= 0 else "below"
    agreement = (
        "so momentum is turning against the trend"
        if (macd < 0) != (hist < 0)
        else "so momentum is running with the trend"
    )
    return (
        f"MACD {fmt_num(macd, 3)}: the {TECHNICALS.macd_fast}-day EMA sits "
        f"{fmt_num(abs(macd), 3)} {trend} the {TECHNICALS.macd_slow}-day one, in price "
        f"units -- that line is the trend. The histogram is {fmt_num(hist, 3)}, which "
        f"puts MACD {momentum} its own {TECHNICALS.macd_signal}-day signal line at "
        f"{fmt_num(signal, 3)} -- that is the momentum, {agreement}. Both are moving "
        f"averages differenced, so both lag by construction: they confirm a move that "
        f"has already happened rather than anticipating one."
    )


def _technical_panels(ohlcv: pd.DataFrame, ind: pd.DataFrame, ticker: str, native_ccy: str) -> None:
    ui_theme.section("PRICE, MOMENTUM AND TREND", "📈")

    symbol = CURRENCY_SYMBOLS.get(native_ccy, "")
    ui.chart(
        ch.price_with_panels(
            ohlcv,
            ind,
            currency_symbol=symbol,
            ticker=ticker,
            height=_PANEL_HEIGHT,
        )
    )

    close = _last(ind, "close")
    _indicator_readings(ind, close, native_ccy)

    st.caption(_rsi_caption(_last(ind, "RSI")))
    st.caption(_macd_caption(ind))

    percent_b = _last(ind, "BB_PERCENT_B")
    bandwidth = _last(ind, "BB_BANDWIDTH")
    if _finite(percent_b) and _finite(bandwidth):
        st.caption(
            f"Bollinger %B {fmt_ratio_pct(percent_b, 0)} places the close inside the "
            f"{TECHNICALS.bollinger_period}-day envelope, where 0% is the lower band "
            f"and 100% the upper. Bandwidth {fmt_pct(bandwidth, 1)} is the width of "
            f"that envelope as a share of its own middle band -- the security's "
            f"current volatility measured against itself, which is the only reason "
            f"the number is comparable across names at different prices."
        )


# ---------------------------------------------------------------------------
# 6. the 52-week range
# ---------------------------------------------------------------------------
def _range_52w(ohlcv: pd.DataFrame, ticker: str, cfg: Any, native_ccy: str) -> None:
    ui_theme.section("POSITION IN THE 52-WEEK RANGE", "🎯")

    band = tec.distance_from_52w(ohlcv["Close"])
    n_obs = int(band.get("n_obs", 0) or 0)
    if n_obs == 0:
        ui.empty_state(
            "No usable closes",
            f"Every close in the loaded window for {ticker} is missing, so the range "
            "cannot be measured.",
            icon="📉",
        )
        return

    last = band.get("last")
    high = band.get("high")
    low = band.get("low")
    from_high = band.get("pct_from_high")
    from_low = band.get("pct_from_low")
    placement = band.get("position_pct")

    ui.kpi_row(
        [
            {
                "label": "Last close",
                "value": fmt_money(last, native_ccy, 2),
                "tone": "accent",
                "footnote": f"Final bar of the {n_obs:,}-session window",
                "help_text": "The last adjusted close, not an intraday quote.",
            },
            {
                "label": "Range high",
                "value": fmt_money(high, native_ccy, 2),
                "tone": "neutral",
                "delta": f"{fmt_pct(from_high, 1, signed=True)} from here",
                "delta_tone": delta_tone(from_high),
                "footnote": "Highest close in the trailing window",
                "help_text": "Highest closing price, not the highest intraday print.",
            },
            {
                "label": "Range low",
                "value": fmt_money(low, native_ccy, 2),
                "tone": "neutral",
                "delta": f"{fmt_pct(from_low, 1, signed=True)} from here",
                "delta_tone": delta_tone(from_low),
                "footnote": "Lowest close in the trailing window",
                "help_text": "Lowest closing price, not the lowest intraday print.",
            },
            {
                "label": "Range position",
                "value": fmt_pct(placement, 0),
                "tone": "neutral",
                "footnote": "0% at the low, 100% at the high",
                "help_text": "(last - low) / (high - low). Undefined for a flat range.",
            },
        ],
        columns=4,
    )

    ui.chart(
        ch.range_52w_bar(
            band,
            ticker=ticker,
            currency_symbol=CURRENCY_SYMBOLS.get(native_ccy, ""),
            height=240,
        )
    )

    if _finite(placement):
        st.caption(
            f"{ticker} closed at {fmt_money(last, native_ccy, 2)}, "
            f"{fmt_pct(placement, 0)} of the way up its range: "
            f"{fmt_pct(abs(float(from_high)) if _finite(from_high) else None, 1)} below "
            f"the high of {fmt_money(high, native_ccy, 2)} and "
            f"{fmt_pct(from_low, 1)} above the low of "
            f"{fmt_money(low, native_ccy, 2)}. Where a price sits in its own range is "
            f"a momentum observation and never a valuation one -- a name at 5% of its "
            f"range is cheap against its own year and can still be expensive against "
            f"its earnings."
        )

    if n_obs < _FULL_YEAR_SESSIONS:
        st.warning(
            f"The range above spans {n_obs:,} sessions, not 52 weeks. Either the "
            f"history window is set to "
            f"{PERIOD_LABELS.get(cfg.period, cfg.period)} and never reaches back a "
            f"year, or the provider simply has less history than that for {ticker} -- "
            f"a recently listed name has no year to look back on. The high and low "
            f"above are the honest extremes of the sessions that do exist, but they "
            f"are not a 52-week high and low and should not be read as one."
        )


# ---------------------------------------------------------------------------
# 7. mechanical signals
# ---------------------------------------------------------------------------
def _signals(ohlcv: pd.DataFrame, ticker: str) -> None:
    ui_theme.section("MECHANICAL SIGNALS", "🚦")

    st.caption(
        f"These are pattern flags computed from {ticker}'s price alone -- a moving "
        f"average cross, an RSI level, a close outside the Bollinger envelope, a MACD "
        f"sign flip. Every one of them is decades old, ships in every charting package "
        f"on the market, and is therefore already reflected in the price. They describe "
        f"what the chart has done. They are not predictions, they carry no view on the "
        f"business, and nothing here is a recommendation to trade."
    )
    ui.insight_list(
        tec.technical_signals(ohlcv),
        empty_message=(
            "No rule fired: either the patterns are quiet or this window is too short "
            "for the longer moving averages to have resolved. A rule with insufficient "
            "history is skipped rather than reported on partial data."
        ),
    )


# ---------------------------------------------------------------------------
# 8. volume
# ---------------------------------------------------------------------------
def _volume(ohlcv: pd.DataFrame, ticker: str, native_ccy: str) -> None:
    ui_theme.section("TRADED VOLUME", "📊")

    ui.chart(
        ch.volume_chart(
            ohlcv,
            currency_symbol=CURRENCY_SYMBOLS.get(native_ccy, ""),
            ticker=ticker,
            height=320,
        )
    )

    volume = (
        pd.to_numeric(ohlcv["Volume"], errors="coerce").dropna()
        if "Volume" in ohlcv.columns
        else pd.Series(dtype="float64")
    )
    if len(volume) >= 2:
        latest = float(volume.iloc[-1])
        average = float(volume.tail(_VOLUME_WINDOW).mean())
        multiple = latest / average if average > 0 else float("nan")
        st.caption(
            f"The last session traded {fmt_compact(latest, '')} shares against a "
            f"{_VOLUME_WINDOW}-session average of {fmt_compact(average, '')} -- "
            f"{fmt_num(multiple, 2)}x its own norm. Volume is a share count, not "
            f"turnover, so it is comparable with this security's own history and with "
            f"nothing else: a 1,000-share day in a "
            f"{fmt_money(float(ohlcv['Close'].iloc[-1]), native_ccy, 0)} stock is not "
            f"the same trade as a 1,000-share day in a penny stock. Bars are coloured "
            f"by whether the close beat the previous one, which is the standard "
            f"reading of who was in control of the session."
        )
    else:
        st.caption(
            "The provider returned no usable volume for this instrument. That is "
            "normal for index series and for some mutual-fund share classes."
        )


# ---------------------------------------------------------------------------
# 9. indicator methodology
# ---------------------------------------------------------------------------
def _methodology(ohlcv: pd.DataFrame, cfg: Any) -> None:
    smas = ", ".join(str(w) for w in TECHNICALS.sma_windows)
    emas = ", ".join(str(s) for s in TECHNICALS.ema_windows)

    ui.methodology(
        "Every indicator on this page, defined",
        f"""
All of it is computed by `analytics.technicals` on the **{len(ohlcv):,} adjusted
daily closes** in the {PERIOD_LABELS.get(cfg.period, cfg.period)} window. Every
period and level below comes from `config.TECHNICALS`, so the assumptions are
auditable in one file rather than scattered through the chart code.

| Indicator | Definition | Settings |
| --- | --- | --- |
| SMA(n) | mean of the last `n` closes | {smas} |
| EMA(s) | `a*P_t + (1-a)*EMA_(t-1)`, `a = 2/(s+1)` | {emas} |
| RSI | `100 - 100/(1 + avg_gain/avg_loss)` | {TECHNICALS.rsi_period}, lines at {TECHNICALS.rsi_oversold:g}/{TECHNICALS.rsi_overbought:g} |
| MACD | `EMA(fast) - EMA(slow)`, signal `= EMA(macd)` | {TECHNICALS.macd_fast}/{TECHNICALS.macd_slow}/{TECHNICALS.macd_signal} |
| Bollinger | `SMA +/- n_std x sd`, `%B = (P-lower)/(upper-lower)` | {TECHNICALS.bollinger_period}, {TECHNICALS.bollinger_std:g} sd |
| ATR | Wilder smoothing of `max(H-L, abs(H-C_prev), abs(L-C_prev))` | {TECHNICALS.atr_period} |

**RSI and ATR use Wilder's smoothing, not a rolling mean.** Both apply
`alpha = 1 / period` exponential smoothing (`ewm(alpha=1/period, adjust=False)`),
which is the definition from Wilder's 1978 *New Concepts in Technical Trading
Systems* and the one every serious charting package implements. A rolling-mean
RSI produces visibly different numbers and is the classic tell of a formula
copied from a blog post. If you cross-check these readings against a broker
terminal, this is the convention they will match.

**Bollinger bands use the population standard deviation** (`ddof=0`) over the
same window as the middle band, because that is Bollinger's own construction.
The rest of QuantFolio uses the sample convention (`ddof=1`) for risk statistics,
where it is the correct one -- the two are deliberately different and the
difference is documented at each site.

**A cross is treated as news for {tec.CROSS_LOOKBACK_SESSIONS} sessions.** Beyond
that the signal engine reports the resulting posture rather than the event.
There is no field for this in `config.TECHNICALS`; it is declared and documented
in `analytics.technicals.CROSS_LOOKBACK_SESSIONS`.

**Adjusted prices, stated plainly.** Bars are split- and dividend-adjusted, so a
historical moving-average cross is drawn at the adjusted price, not at the price
that actually printed that day. That is the right basis for measuring returns and
the wrong basis for reconstructing the exact level a chart showed at the time.

**The window is the whole story.** Annualisation elsewhere in the app uses
{TRADING_DAYS} trading days; nothing on this page is annualised. Every reading
above describes these {len(ohlcv):,} sessions. Load a different window and the
same security produces different indicators -- not because anything is unstable,
but because a moving average is a statement about a window, not about a company.

**Source.** {DATA_SOURCE}.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _market_cap(result, ticker: str) -> float | None:
    """Market capitalisation in the instrument's OWN currency, or None.

    Native, not converted: Altman's X4 divides it by total liabilities taken
    from the same statements, so a rupee-converted numerator over a dollar
    denominator would be wrong by the FX rate. ``None`` rather than a guess --
    the Z-Score reports itself unavailable instead of substituting book
    equity, which measures something else.
    """
    profile = (getattr(result, "profiles", None) or {}).get(ticker)
    raw = getattr(profile, "market_cap", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value and value > 0 else None


def _country_of(result, ticker: str) -> str:
    """The issuer's country, which selects the Altman variant."""
    profile = (getattr(result, "profiles", None) or {}).get(ticker)
    country = str(getattr(profile, "country", "") or "").strip()
    if country:
        return country
    from src import stock_universe as universe

    record = universe.by_yahoo_symbol(ticker)
    return record.country if record is not None else ""


def _overlay_spot(result, ticker: str) -> float:
    """Native current price for ``ticker`` from the positions table.

    Native rather than converted: an option strike is quoted in the currency
    the underlying trades in, so a rupee-converted price would misprice every
    strike comparison on a US holding.
    """
    positions = getattr(result, "positions", None)
    columns = getattr(positions, "columns", ())
    if "ticker" in columns and "current_price" in columns:
        match = positions.loc[positions["ticker"].astype(str) == str(ticker), "current_price"]
        if not match.empty:
            try:
                value = float(match.iloc[0])
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value) and value > 0:
                return value
    quote = (getattr(result, "quotes", None) or {}).get(ticker)
    return float(getattr(quote, "price", float("nan")) or float("nan"))


def render() -> None:
    ui_theme.page_header(
        "Security deep dive",
        "One instrument: what it is, what it costs you, and what its price has done",
        icon="🔬",
    )

    result = guards.require_result()
    if result is None:
        return

    cfg = result.config

    ticker = _pick_ticker(result)
    if not ticker:
        ui.empty_state(
            "No security selected",
            "Pick a holding from the analysed portfolio, or type any symbol the data "
            "provider prices, to load its candles, indicators and fundamentals.",
            icon="🔎",
            hint="Indian equities need a .NS (NSE) or .BO (BSE) suffix -- RELIANCE.NS, "
            "not RELIANCE.",
        )
        return

    with ui.loading(f"Loading {ticker} daily bars..."):
        try:
            ohlcv = data_access.load_ohlcv(ticker, cfg.period)
        except Exception as exc:  # noqa: BLE001 - a data failure is a page state
            ohlcv = pd.DataFrame()
            st.caption(f"The provider request failed: {exc}")

    if ohlcv is None or len(ohlcv) == 0 or "Close" not in ohlcv.columns:
        ui.empty_state(
            f"No price history for {ticker}",
            f"The provider returned no usable daily bars over the "
            f"{PERIOD_LABELS.get(cfg.period, cfg.period)} window. A bar without a close "
            f"is dropped rather than filled, so a symbol that is delisted, suspended or "
            f"simply mistyped arrives here empty rather than wrong.",
            icon="📭",
            hint="Check the suffix (.NS / .BO for India, none for US listings), or "
            "widen the history window in the sidebar.",
        )
        return

    row = _row_for(result, ticker)
    held = row is not None
    profile = _profile_for(result, ticker)
    native_ccy = _native_currency(profile, row, cfg)

    _identity(ticker, profile, ohlcv, cfg, held)

    ui_theme.spacer(0.4)
    _fundamentals(profile, ticker, native_ccy)

    if held and row is not None:
        ui_theme.spacer(0.4)
        _position_economics(result, ticker, row, cfg, native_ccy)
    else:
        ui_theme.spacer(0.4)
        st.caption(
            f"{ticker} is not in the analysed portfolio, so there is no weight, no P&L "
            f"and no risk contribution to show. Everything below is instrument "
            f"research, independent of what you own."
        )

    ind = tec.indicator_frame(ohlcv)

    ui_theme.spacer(0.4)
    _technical_panels(ohlcv, ind, ticker, native_ccy)

    ui_theme.spacer(0.4)
    _range_52w(ohlcv, ticker, cfg, native_ccy)

    ui_theme.spacer(0.4)
    _signals(ohlcv, ticker)

    ui_theme.spacer(0.4)
    _volume(ohlcv, ticker, native_ccy)

    ui_theme.spacer(0.4)
    _methodology(ohlcv, cfg)

    ui_theme.spacer(0.4)
    ui_theme.section("OPTIONS OVERLAY", "\U0001f9ee")
    st.caption(
        "What a covered call or a protective put would do to this holding: the "
        "premium, the payoff, the breakeven and what it costs in upside or drag."
    )
    options_panel.render(result, ticker, spot=_overlay_spot(result, ticker), currency=native_ccy)

    ui_theme.spacer(0.4)
    ui_theme.section("FUNDAMENTAL QUALITY", "📑")
    st.caption(
        "Published scores computed from this company's own balance sheet, income "
        "statement and cash-flow statement — with the return horizons that cannot "
        "be modelled from them stated plainly."
    )
    fundamentals_panel.render(
        ticker,
        market_cap=_market_cap(result, ticker),
        country=_country_of(result, ticker),
        price=_overlay_spot(result, ticker),
    )
