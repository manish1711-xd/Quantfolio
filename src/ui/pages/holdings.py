"""Holdings - the position blotter and the shape of the book.

This page answers what the headline numbers cannot: what exactly is owned, how
lopsided the capital is once weights are taken into account, how the money
splits across sectors and names, and how stale the price sitting behind every
row actually is.

Everything is read off the completed :class:`~src.models.AnalysisResult`. The
only computation performed here is
:func:`src.analytics.positions.allocation_by`, which is a pure regrouping of
the positions table that the sector charts consume.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import positions as pos
from src.config import DATA_SOURCE, INSIGHTS
from src.ui import components as ui
from src.ui import guards, session
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    fmt_compact,
    fmt_datetime,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    fmt_shares,
)

# ---------------------------------------------------------------------------
# blotter column vocabulary
# ---------------------------------------------------------------------------
#: positions column -> human label shown in the blotter and the column picker.
_LABELS: dict[str, str] = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "industry": "Industry",
    "quantity": "Qty",
    "buy_price": "Avg cost",
    "current_price": "Last price",
    "currency": "Ccy",
    "fx_rate": "FX rate",
    "day_change_pct": "Day %",
    "investment": "Invested",
    "current_value": "Market value",
    "pnl": "Unrealised P&L",
    "pnl_pct": "Return %",
    "weight_pct": "Weight",
    "buy_date": "First bought",
    "target_weight_pct": "Target weight",
}

#: What a position blotter should show before the user customises it.
_DEFAULT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "name",
    "sector",
    "quantity",
    "buy_price",
    "current_price",
    "day_change_pct",
    "current_value",
    "pnl",
    "pnl_pct",
    "weight_pct",
)


# ---------------------------------------------------------------------------
# small display helpers
# ---------------------------------------------------------------------------
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


def _row_for(positions: pd.DataFrame, ticker: str) -> pd.Series | None:
    """The single positions row for ``ticker``, or None when it was not priced."""
    if "ticker" not in positions.columns:
        return None
    match = positions.loc[positions["ticker"].astype(str) == str(ticker)]
    return None if match.empty else match.iloc[0]


def _progress_ceiling(weights: pd.Series) -> float:
    """Upper bound for the weight bar, rounded up to the next 5 percentage points.

    Scaling the bar to 100% makes every position in a 25-name book look
    identical; scaling it to the largest holding keeps the ranking readable.
    The caption states which one is in force so the bar is never mistaken for
    a share of 100.
    """
    numeric = pd.to_numeric(weights, errors="coerce").dropna()
    if numeric.empty:
        return 100.0
    largest = float(numeric.max())
    if largest <= 0:
        return 100.0
    ceiling = 5.0 * (int(largest / 5.0) + 1)
    return float(min(max(ceiling, 5.0), 100.0))


def _range_position_pct(price: Any, low: Any, high: Any) -> float:
    """Where ``price`` sits inside the [low, high] band, in percent."""
    try:
        p, lo, hi = float(price), float(low), float(high)
    except (TypeError, ValueError):
        return float("nan")
    if not (pd.notna(p) and pd.notna(lo) and pd.notna(hi)) or hi <= lo:
        return float("nan")
    return (p - lo) / (hi - lo) * 100.0


# ---------------------------------------------------------------------------
# 1. concentration KPIs
# ---------------------------------------------------------------------------
def _kpis(summary: dict, n_holdings: int) -> None:
    effective = summary.get("effective_holdings")
    hhi = summary.get("hhi")
    largest = summary.get("largest_holding")
    top5 = summary.get("top5_weight_pct")

    largest_ticker, largest_weight = largest if largest else (None, float("nan"))
    ratio = (
        float(effective) / n_holdings
        if effective is not None and pd.notna(effective) and n_holdings > 0
        else float("nan")
    )

    cards: list[dict[str, Any]] = [
        {
            "label": "Holdings",
            "value": str(n_holdings),
            "tone": "neutral",
            "footnote": (
                f"{summary.get('n_gainers', 0)} in profit / "
                f"{summary.get('n_losers', 0)} under water"
            ),
            "help_text": "Priced positions that survived validation and had a usable quote.",
        },
        {
            "label": "Effective bets",
            "value": fmt_num(effective, 1),
            "tone": (
                "warn"
                if pd.notna(ratio) and ratio < INSIGHTS.effective_holdings_ratio
                else "neutral"
            ),
            "delta": (
                f"{fmt_pct(ratio * 100.0, 0)} of {n_holdings} names" if pd.notna(ratio) else DASH
            ),
            "delta_tone": (
                "loss" if pd.notna(ratio) and ratio < INSIGHTS.effective_holdings_ratio else "muted"
            ),
            "footnote": f"1 / HHI, HHI = {fmt_num(hhi, 4)}",
            "help_text": "Equally-weighted positions that would carry the same concentration.",
        },
        {
            "label": "Largest holding",
            "value": fmt_pct(largest_weight),
            "tone": (
                "warn"
                if pd.notna(largest_weight)
                and float(largest_weight) >= INSIGHTS.concentration_warn_pct
                else "neutral"
            ),
            "footnote": _text(largest_ticker),
            "help_text": "Share of current market value in the single biggest position.",
        },
        {
            "label": "Top 5 weight",
            "value": fmt_pct(top5),
            "tone": "neutral",
            "footnote": (
                f"the other {n_holdings - 5} names hold {fmt_pct(100.0 - float(top5))}"
                if pd.notna(top5) and n_holdings > 5
                else "the whole book"
            ),
            "help_text": "Combined weight of the five largest positions.",
        },
    ]
    ui.kpi_row(cards, columns=4)

    ui.methodology(
        "How concentration is measured",
        f"""
**Herfindahl-Hirschman Index.** `HHI = sum(w_i^2)` with every weight `w_i`
expressed as a *fraction* of current market value. This portfolio scores
**{fmt_num(hhi, 4)}**. An equal-weighted book of {n_holdings} names would score
{fmt_num(1.0 / n_holdings if n_holdings else float("nan"), 4)}; a single-stock
book scores 1.0000.

**Effective number of bets.** `ENB = 1 / HHI` = **{fmt_num(effective, 2)}**. It
answers one question: *how many equally-weighted positions would carry this much
concentration?* Listing {n_holdings} tickers means nothing if two of them are
70% of the money - the ENB reports the number the capital actually behaves like,
and it is always between 1 and the holding count.

**The flag.** The card turns amber when `ENB / N` falls below
{fmt_ratio_pct(INSIGHTS.effective_holdings_ratio, 0)}
(`config.INSIGHTS.effective_holdings_ratio`) - i.e. when more than half the
nominal diversification is an illusion. Here `ENB / N` =
**{fmt_pct(ratio * 100.0) if pd.notna(ratio) else DASH}**.

**Caveat.** All three numbers are counts of *capital*, not of *risk*. Five
equal-weighted PSU banks score ENB = 5 and are still one bet. Correlation is
what settles that question - see Risk > Correlation and the risk-contribution
split on the Attribution page.
        """,
    )


# ---------------------------------------------------------------------------
# 2. the blotter
# ---------------------------------------------------------------------------
def _blotter(positions: pd.DataFrame, cfg: Any) -> None:
    ui_theme.section("Position blotter", "\U0001f4d2")

    available = [c for c in _LABELS if c in positions.columns]
    defaults = [c for c in _DEFAULT_COLUMNS if c in available]

    chosen = st.multiselect(
        "Columns",
        options=available,
        default=defaults,
        format_func=lambda c: _LABELS.get(c, c),
        key="holdings_blotter_columns",
        help="Prices and average cost stay in each instrument's own quote "
        "currency; every money column is converted to the base currency.",
    )
    ordered = [c for c in available if c in set(chosen)] or defaults
    if "ticker" in available and "ticker" not in ordered:
        ordered = ["ticker", *ordered]

    symbol = getattr(cfg, "currency_symbol", "")
    money_format = f"{symbol}%.0f"
    ceiling = _progress_ceiling(positions.get("weight_pct", pd.Series(dtype="float64")))

    column_config: dict[str, Any] = {
        "ticker": st.column_config.TextColumn("Ticker", width="small", pinned=True),
        "name": st.column_config.TextColumn("Name", width="medium"),
        "sector": st.column_config.TextColumn("Sector", width="small"),
        "industry": st.column_config.TextColumn("Industry", width="medium"),
        "quantity": st.column_config.NumberColumn(
            "Qty", format="%.2f", width="small", help="Units held today."
        ),
        "buy_price": st.column_config.NumberColumn(
            "Avg cost",
            format="%.2f",
            width="small",
            help="Quantity-weighted average purchase price, in the instrument's native currency.",
        ),
        "current_price": st.column_config.NumberColumn(
            "Last price",
            format="%.2f",
            width="small",
            help="Latest quote in the instrument's native currency. See the "
            "freshness table for its timestamp.",
        ),
        "currency": st.column_config.TextColumn("Ccy", width="small"),
        "fx_rate": st.column_config.NumberColumn(
            "FX rate",
            format="%.4f",
            width="small",
            help=f"Value of one unit of the quote currency in "
            f"{getattr(cfg, 'base_currency', '')}. 1.0000 means no conversion "
            f"was applied.",
        ),
        "day_change_pct": st.column_config.NumberColumn(
            "Day %",
            format="%.2f%%",
            width="small",
            help="Move against the previous close. Blank when the provider "
            "supplied no previous close - never assumed flat.",
        ),
        "investment": st.column_config.NumberColumn(
            "Invested", format=money_format, help="quantity x avg cost x FX rate."
        ),
        "current_value": st.column_config.NumberColumn(
            "Market value",
            format=money_format,
            help="quantity x last price x FX rate.",
        ),
        "pnl": st.column_config.NumberColumn(
            "Unrealised P&L",
            format=money_format,
            help="Market value less invested. Realised gains live on the Ledger page.",
        ),
        "pnl_pct": st.column_config.NumberColumn(
            "Return %",
            format="%.2f%%",
            help="P&L over cost. FX-neutral: both legs convert at the same rate.",
        ),
        "weight_pct": st.column_config.ProgressColumn(
            "Weight",
            format="%.2f%%",
            min_value=0.0,
            max_value=ceiling,
            help=f"Share of total market value. The bar is scaled to "
            f"{fmt_pct(ceiling, 0)}, not to 100%.",
        ),
        "buy_date": st.column_config.DateColumn(
            "First bought",
            format="DD MMM YYYY",
            width="small",
            help="Optional. Display only unless a trade ledger is supplied.",
        ),
        "target_weight_pct": st.column_config.NumberColumn(
            "Target weight",
            format="%.2f%%",
            help="Optional column carried through from your holdings input; "
            "drives the Rebalance page.",
        ),
    }

    frame = positions.loc[:, ordered]
    ui.dataframe(
        frame,
        height=min(38 * (len(frame) + 1) + 10, 640),
        column_config={k: v for k, v in column_config.items() if k in ordered},
    )

    largest = positions["weight_pct"].max() if "weight_pct" in positions else float("nan")
    st.caption(
        f"Sorted by market value, largest first. Weight bars are scaled to "
        f"{fmt_pct(ceiling, 0)} (the largest position is {fmt_pct(largest)}), so the "
        f"bar shows rank, not a share of 100. Blank cells are missing data, never "
        f"zeros."
    )

    ui.methodology(
        "Position arithmetic and the currency convention",
        f"""
Per holding, with `fx` the value of one unit of the quote currency in
{getattr(cfg, "base_currency", "the base currency")}:

```
invested      = quantity x avg cost   x fx
market value  = quantity x last price x fx
unrealised    = market value - invested
return %      = unrealised / invested x 100     (invested > 0)
weight %      = market value / total market value x 100
```

**Prices stay native, money converts.** A US name inside a rupee book still
shows its dollar price, because that is the number the holder recognises. Only
*Invested*, *Market value* and *Unrealised P&L* are translated. Because both
legs of the P&L convert at the same rate, **Return %** is FX-neutral and equals
`last price / avg cost - 1` exactly.

**Average cost, not tax lots.** *Avg cost* is the quantity-weighted average of
the rows you entered; duplicate tickers were merged as
`sum(q_i p_i) / sum(q_i)`. It is a cost basis, not a FIFO lot ledger - holding
periods, realised P&L and the short/long-term split need the trade ledger and
live on the Ledger page.

**Unrealised only.** Nothing here includes dividends received, brokerage, STT
or stamp duty. Cash returns are not in the price series either, so total return
on a high-yield name is understated by roughly its yield.
        """,
    )


# ---------------------------------------------------------------------------
# 3. allocation structure
# ---------------------------------------------------------------------------
def _allocation(positions: pd.DataFrame, cfg: Any) -> None:
    ui_theme.section("Allocation structure", "\U0001f9ee")

    symbol = getattr(cfg, "currency_symbol", "")
    currency = getattr(cfg, "base_currency", "INR")
    n = len(positions)

    treemap_tab, sector_tab, holding_tab, hierarchy_tab = st.tabs(
        ["Treemap", "By sector", "By holding", "Hierarchy"]
    )

    with treemap_tab:
        ui.chart(ch.allocation_treemap(positions, currency_symbol=symbol, height=520))
        st.caption(
            "Tile area is market value, tile colour is unrealised return against "
            "cost. A large red tile is where the money and the pain sit together - "
            "that is the position whose recovery the portfolio is really waiting on."
        )

    with sector_tab:
        allocation = pos.allocation_by(positions, "sector")
        left, right = st.columns(2, gap="medium")
        with left:
            ui.chart(ch.sector_bar(allocation, currency_symbol=symbol, height=400))
        with right:
            ui.chart(ch.allocation_donut(positions, "sector", currency_symbol=symbol, height=400))
        if not allocation.empty:
            top = allocation.iloc[0]
            st.caption(
                f"{_text(top['sector'])} is the heaviest sleeve at "
                f"{fmt_pct(top['weight_pct'])} of market value "
                f"({fmt_money(top['current_value'], currency, compact=True)}) across "
                f"{int(top['n'])} name(s), out of {len(allocation)} sector(s). Above "
                f"{fmt_pct(INSIGHTS.sector_concentration_pct, 0)} in one sector the "
                f"portfolio is a sector view wearing a diversified label."
            )
            ui.dataframe(
                allocation.rename(
                    columns={
                        "sector": "Sector",
                        "current_value": "Market value",
                        "weight_pct": "Weight",
                        "n": "Names",
                    }
                ),
                column_config={
                    "Market value": st.column_config.NumberColumn(format=f"{symbol}%.0f"),
                    "Weight": st.column_config.ProgressColumn(
                        format="%.2f%%", min_value=0.0, max_value=100.0
                    ),
                    "Names": st.column_config.NumberColumn(format="%d", width="small"),
                },
            )
        else:
            st.caption("No sector labels were available for these holdings.")

    with holding_tab:
        ui.chart(ch.weight_bar(positions, currency_symbol=symbol, height=max(340, 30 * n)))
        st.caption(
            f"Ranked weights against the equal-weight line at "
            f"{fmt_pct(100.0 / n if n else float('nan'))} per name. Bars to the right "
            f"of the line are active overweights - the positions that decide the "
            f"portfolio's result; the tail to the left is capital that cannot move "
            f"the outcome either way."
        )

    with hierarchy_tab:
        ui.chart(ch.allocation_sunburst(positions, currency_symbol=symbol, height=560))
        st.caption(
            "Inner ring sectors, outer ring holdings, each ring summing to the whole "
            "book. A thin sector wedge filled by a single holding is a one-name "
            "sector bet; a wide wedge split evenly is a genuine sector allocation."
        )

    ui.methodology(
        "How the buckets are formed",
        """
**Sector labels** come from the data provider's classification for each ticker,
falling back to a `sector` column on your holdings input, then to the quote type
(*Fund / ETF*, *Index*) and finally to **Unclassified**. Coverage is patchy for
Indian small caps and for most ETFs, so an *Unclassified* wedge is missing
metadata, not a real sector.

**Weights are market-value weights** - `bucket value / total market value x 100`
- computed from today's prices, so every allocation view is a snapshot that
drifts with the market between refreshes.

**A fund is not its holdings.** An index ETF counts as one line item in one
bucket. If a broad-market ETF sits next to five individual large caps, the true
look-through sector exposure is more concentrated than any chart on this tab can
show.

**A sector is not a factor.** Two names in different sectors can still share a
single driver - rates, the rupee, oil. Correlation and PCA on the Risk pages are
where that question gets answered.
        """,
    )


# ---------------------------------------------------------------------------
# 4. per-holding detail
# ---------------------------------------------------------------------------
def _reference_stats(profile: Any, row: pd.Series | None, cfg: Any) -> list[tuple[str, str]]:
    native = _text(getattr(profile, "currency", None))
    if native == DASH and row is not None:
        native = _text(row.get("currency"))
    native_ccy = native if native != DASH else getattr(cfg, "base_currency", "INR")

    low = getattr(profile, "fifty_two_week_low", None)
    high = getattr(profile, "fifty_two_week_high", None)
    band = (
        f"{fmt_money(low, native_ccy, 2)} to {fmt_money(high, native_ccy, 2)}"
        if low is not None or high is not None
        else DASH
    )

    return [
        ("Name", _text(getattr(profile, "display_name", None))),
        ("Sector", _text(getattr(profile, "sector_label", None))),
        ("Industry", _text(getattr(profile, "industry", None))),
        ("Exchange", _text(getattr(profile, "exchange", None))),
        ("Instrument", _text(getattr(profile, "quote_type", None))),
        ("Market cap", fmt_compact(getattr(profile, "market_cap", None), native_ccy)),
        ("Trailing P/E", fmt_num(getattr(profile, "trailing_pe", None), 1)),
        ("Price / book", fmt_num(getattr(profile, "price_to_book", None), 2)),
        ("Dividend yield", fmt_ratio_pct(getattr(profile, "dividend_yield", None))),
        ("52-week range", band),
    ]


def _economics_stats(row: pd.Series, cfg: Any) -> list[tuple[str, str]]:
    currency = getattr(cfg, "base_currency", "INR")
    native = _text(row.get("currency"))
    native_ccy = native if native != DASH else currency

    return [
        ("Quantity", fmt_shares(row.get("quantity"))),
        ("Average cost", fmt_money(row.get("buy_price"), native_ccy, 2)),
        ("Last price", fmt_money(row.get("current_price"), native_ccy, 2)),
        ("Day move", fmt_pct(row.get("day_change_pct"), signed=True)),
        ("Invested", fmt_money(row.get("investment"), currency)),
        ("Market value", fmt_money(row.get("current_value"), currency)),
        ("Unrealised P&L", fmt_money(row.get("pnl"), currency, signed=True)),
        ("Return", fmt_pct(row.get("pnl_pct"), signed=True)),
        ("Portfolio weight", fmt_pct(row.get("weight_pct"))),
        ("FX rate to base", fmt_num(row.get("fx_rate"), 4)),
    ]


def _detail(result: Any, positions: pd.DataFrame, cfg: Any) -> None:
    ui_theme.section("Holding detail", "\U0001f50e")

    tickers = [str(t) for t in positions["ticker"].tolist()]
    if not tickers:
        return

    preferred = session.selected_ticker(tickers[0])
    index = tickers.index(preferred) if preferred in tickers else 0
    ticker = st.selectbox("Holding", options=tickers, index=index, key="holdings_detail_ticker")

    row = _row_for(positions, ticker)
    profile = (result.profiles or {}).get(ticker)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("**Reference data**")
        ui.stat_grid(_reference_stats(profile, row, cfg), columns=3)

        low = getattr(profile, "fifty_two_week_low", None)
        high = getattr(profile, "fifty_two_week_high", None)
        price = row.get("current_price") if row is not None else None
        placement = _range_position_pct(price, low, high)
        if pd.notna(placement):
            st.caption(
                f"The last price sits {fmt_pct(placement, 0)} of the way up the "
                f"provider's 52-week band - 0% is the year's low, 100% the year's "
                f"high. That band is the provider's rolling year, not this "
                f"analysis window."
            )
        elif profile is None:
            st.caption(
                "No reference data was returned for this symbol. Fundamentals are "
                "routinely absent for ETFs, index funds and thinly covered small caps."
            )

    with right:
        st.markdown("**Position economics**")
        if row is None:
            ui.empty_state(
                "Position not priced",
                f"{ticker} has no usable quote in this run, so no economics can be shown for it.",
                icon="\U0001f4ad",
            )
        else:
            ui.stat_grid(_economics_stats(row, cfg), columns=2)

    if st.button("Analyse this security", key="holdings_analyse_security", width="stretch"):
        session.select_ticker(ticker)
        st.success(
            f"{ticker} is now the selected security. Open **Research > Security "
            f"deep dive** in the sidebar for candles, moving averages, RSI, MACD "
            f"and the 52-week bullet."
        )

    summary_text = getattr(profile, "summary", None)
    if summary_text:
        with st.expander("Business description (provider text)"):
            st.write(str(summary_text))

    ui.methodology(
        "What the reference figures do and do not mean",
        """
**Provider-sourced, point-in-time.** Every field on the left comes from the
market-data provider as-is. Nothing is derived, interpolated or back-filled, and
a missing field renders as an em dash rather than a zero - a P/E of 0.00 is a
lie, a dash is the truth.

**Trailing P/E** uses the last four reported quarters of earnings; it is
undefined (dash) for a loss-making company, and meaningless for an ETF or index
fund. **Price / book** is most informative for financials and asset-heavy
businesses, least so for asset-light ones.

**Dividend yield** is stored as a decimal (0.012) and rendered as a percentage
(1.20%). It is the provider's trailing twelve-month yield on the current price,
not a forward or declared rate.

**The 52-week band** is the provider's rolling year, which will not match the
history window this analysis used. Placement inside the band is a momentum
observation, never a valuation one: a name at 5% of its band is cheap against
its own year and can still be expensive against its earnings.
        """,
    )


# ---------------------------------------------------------------------------
# 5. data freshness
# ---------------------------------------------------------------------------
def _freshness_frame(result: Any, positions: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    quotes = result.quotes or {}
    ordered = [str(t) for t in positions["ticker"].tolist()]
    extras = [str(t) for t in quotes if str(t) not in set(ordered)]

    rows: list[dict[str, str]] = []
    for ticker in [*ordered, *sorted(extras)]:
        quote = quotes.get(ticker)
        if quote is None:
            rows.append(
                {
                    "Ticker": ticker,
                    "Last price": DASH,
                    "Ccy": DASH,
                    "Freshness": "No quote returned - position skipped",
                    "Price as of": DASH,
                    "Fetched": DASH,
                }
            )
            continue
        currency = _text(getattr(quote, "currency", None))
        rows.append(
            {
                "Ticker": ticker,
                "Last price": fmt_money(
                    getattr(quote, "price", None),
                    currency if currency != DASH else getattr(cfg, "base_currency", "INR"),
                    2,
                ),
                "Ccy": currency,
                "Freshness": _text(getattr(quote, "freshness_label", None)),
                "Price as of": fmt_datetime(getattr(quote, "as_of", None)),
                "Fetched": fmt_datetime(getattr(quote, "fetched_at", None)),
            }
        )
    return pd.DataFrame(rows)


def _freshness(result: Any, positions: pd.DataFrame, cfg: Any) -> None:
    ui_theme.section("Data freshness", "\U0001f4e1")

    frame = _freshness_frame(result, positions, cfg)
    ui.dataframe(frame, height=min(38 * (len(frame) + 1) + 10, 420))

    st.caption(
        "Why the label matters: a *latest traded price* is a live-ish quote and can "
        "still be delayed by roughly fifteen minutes on NSE/BSE, while a *latest "
        "available closing price* is the last settled close and will not move again "
        "until the next session. Mixing the two across markets - an Indian close "
        "beside a US name still trading - means the portfolio total is a blend of "
        "two moments, not one snapshot. The timestamp on each row is the honest "
        "answer to 'as of when?'."
    )

    unconverted = list(positions.attrs.get("fx_unconverted", []) or [])
    if unconverted:
        st.warning(
            "No usable FX rate for "
            + ", ".join(str(c) for c in unconverted)
            + ". Those rows were left unconverted at a rate of 1.0, so the "
            "aggregate totals mix currencies and are indicative only."
        )

    ui.warning_list(result.warnings, "Data notes from this run")

    ui.methodology(
        "Provenance and caching",
        f"""
**Source.** {DATA_SOURCE}.

**Two timestamps, deliberately.** *Price as of* is when the price itself was
struck by the market; *Fetched* is when this application asked for it. They
diverge outside market hours - a Friday close fetched on Sunday is correct data
and stale news at the same time. Both are shown in UTC.

**Caching.** Quotes are cached briefly, daily history longer, and fundamentals
for a day - so a refreshed page can legitimately return the same numbers. Re-run
the analysis to force a fresh pull.

**Skipped symbols.** A ticker with no usable quote (missing, non-finite or
non-positive price) is dropped from the analysis rather than defaulted to zero,
and appears here with a dash so it is never silently lost. One dead symbol
therefore shrinks the portfolio being analysed - it does not corrupt it.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Holdings",
        "The position blotter, what each name weighs, and how fresh its price is",
        icon="\U0001f4cb",
    )

    result = guards.require_result()
    if result is None:
        return

    positions = result.positions
    if positions is None or len(positions) == 0:
        ui.empty_state(
            "No priced positions",
            "The analysis ran but no holding produced a usable quote, so there is "
            "no blotter to show.",
            icon="\U0001f4ed",
            hint="Check the ticker symbols on the Portfolio page - Indian equities "
            "need the .NS or .BO suffix.",
        )
        ui.warning_list(result.warnings, "Data notes from this run")
        return

    cfg = result.config
    summary = result.summary or {}
    n_holdings = int(summary.get("n_holdings", len(positions)))

    ui.freshness_bar(
        result.generated_at,
        DATA_SOURCE,
        extra=[
            f"Base currency {getattr(cfg, 'base_currency', '')}",
            "FX conversion on" if getattr(cfg, "convert_fx", False) else "FX conversion off",
            f"{n_holdings} priced positions",
        ],
    )

    _kpis(summary, n_holdings)
    _blotter(positions, cfg)
    _allocation(positions, cfg)
    _detail(result, positions, cfg)
    _freshness(result, positions, cfg)
