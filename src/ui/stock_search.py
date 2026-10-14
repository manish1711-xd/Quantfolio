"""Stock-search widget: type a company name, get a resolved Yahoo symbol.

Used by the Portfolio builder and the Watchlist page, so it lives here rather
than in either one. Two search tiers, in this order:

1. **Local** — :mod:`src.stock_universe` over the bundled master CSVs. Instant,
   offline, runs on every keystroke, covers NSE large/mid caps, popular Indian
   ETFs, US large caps, the major US ETFs and the global indices.
2. **Online** — :func:`src.data.fetcher.search_symbols` via the provider, and
   ONLY on an explicit button press. Covers the long tail. Results are labelled
   as provider hits because, unlike a curated row, they carry no verified
   country or currency.

The widget never adds a holding by itself. It returns the record the user
picked and the caller decides what to do with it, which keeps "search" and
"mutate the portfolio" separable and testable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from src import stock_universe as universe
from src.config import MAX_SEARCH_RESULTS, PREFERRED_EXCHANGE
from src.ui import data_access
from src.ui import session as sess

__all__ = ["search_box", "add_holding_form", "add_to_watchlist_form", "coverage_caption"]

#: Session key holding the currently-picked ``yahoo_symbol`` per widget instance.
_PICK_SUFFIX = "_pick"
#: Session key holding the last online-search result list per widget instance.
_ONLINE_SUFFIX = "_online"

#: Badge tone per instrument type, so an ETF never reads as a single stock.
_TYPE_TONE: dict[str, str] = {
    "EQUITY": "neutral",
    "ETF": "accent",
    "MUTUALFUND": "accent",
    "INDEX": "muted",
}


def coverage_caption() -> str:
    """Honest one-line statement of what the local file actually contains.

    Quotes the real row count and, when the file was generated from the
    exchange listings, the date its market data came from. The date is not
    decoration: the listing is an end-of-day file, so a company that listed
    this morning is not in it, and a UI that omits the date implies otherwise.

    The US and index files remain a curated selection -- no equivalent
    listing is fetched for them -- so the sentence distinguishes the two
    rather than claiming completeness for everything.
    """
    summary = universe.universe_summary()
    total = summary.get("total", 0)
    venues = [f"{code} {summary[code]}" for code in universe.exchanges() if code in summary]
    stamp = universe.as_of()

    if stamp:
        try:
            pretty = datetime.fromisoformat(stamp).strftime("%d %b %Y")
        except ValueError:
            pretty = stamp
        currency = (
            f"Indian listings are the full NSE and BSE equity lists as of **{pretty}**; "
            "US names and indices are a curated selection. "
        )
    else:
        currency = ""

    return (
        f"Offline search covers {total:,} instruments ({', '.join(venues)}). "
        f"{currency}"
        "Anything else — a very recent listing, another exchange — is one "
        "**Search online** click away."
    )


def _format_hit(record: universe.StockRecord) -> str:
    """Selectbox label: name, symbol, exchange, country, and the type when notable."""
    bits = [record.name, f"·  {record.yahoo_symbol}"]
    if record.exchange:
        bits.append(f"·  {record.exchange}")
    if record.country:
        bits.append(f"·  {record.country}")
    kind = record.instrument_type.upper()
    if kind and kind != "EQUITY":
        bits.append(f"·  {kind}")
    return "  ".join(bits)


def _online_record(hit: dict[str, str]) -> universe.StockRecord:
    """Adapt a provider search hit onto :class:`StockRecord`.

    ``country`` and ``currency`` are left EMPTY on purpose. The provider's
    search endpoint does not return them, and filling them with a plausible
    guess would put unverified data into the same shape the curated file uses
    — after which nothing downstream could tell the two apart.
    """
    symbol = str(hit.get("symbol", "")).strip().upper()
    return universe.StockRecord(
        symbol=symbol.split(".")[0],
        yahoo_symbol=symbol,
        name=str(hit.get("name") or symbol),
        exchange=str(hit.get("exchange") or ""),
        country="",
        currency="",
        instrument_type=str(hit.get("quote_type") or "EQUITY").upper(),
        sector="",
    )


# ---------------------------------------------------------------------------
# the widget
# ---------------------------------------------------------------------------
def search_box(
    key_prefix: str,
    label: str = "Search stock",
    placeholder: str = "Reliance, TCS, HDFC Bank, AAPL, Nifty 50…",
    limit: int = MAX_SEARCH_RESULTS,
    instrument_types: tuple[str, ...] | None = None,
    allow_online: bool = True,
) -> universe.StockRecord | None:
    """Render the search input and result picker; return the chosen record.

    ``None`` means nothing is selected yet — an empty query, no matches, or the
    user has not picked a row. Callers must treat that as "do nothing", not as
    an error.

    Every widget key is namespaced by ``key_prefix`` so two instances (the
    builder and the watchlist) can coexist on one screen without sharing state.
    """
    query = st.text_input(
        label,
        key=f"{key_prefix}_query",
        placeholder=placeholder,
        help="Company name, ticker, or a nickname like HUL, Bank Nifty or Fevicol. "
        "Typos are tolerated.",
    )

    online_key = f"{key_prefix}{_ONLINE_SUFFIX}"
    text = (query or "").strip()
    if not text:
        st.session_state.pop(online_key, None)
        st.caption(coverage_caption())
        return None

    hits = universe.search(text, limit=limit, instrument_types=instrument_types)
    records: list[universe.StockRecord] = [h.record for h in hits]
    reasons = {h.record.yahoo_symbol: h.matched_on for h in hits}

    # Merge in whatever the last online search returned, without displacing
    # local hits: curated rows are better data and should stay on top.
    for hit in st.session_state.get(online_key, []) or []:
        record = _online_record(hit)
        if record.yahoo_symbol and all(r.yahoo_symbol != record.yahoo_symbol for r in records):
            records.append(record)
            reasons[record.yahoo_symbol] = "provider search"

    if not records:
        st.info(
            f"No local match for “{text}”. The bundled files cover common Indian and "
            "US listings only — try **Search online** for everything else, or type the "
            "exact Yahoo Finance symbol into the grid below."
        )
    else:
        st.caption(f"{len(records)} match{'es' if len(records) != 1 else ''}")

    chosen: universe.StockRecord | None = None
    if records:
        options = [r.yahoo_symbol for r in records]
        by_symbol = {r.yahoo_symbol: r for r in records}
        picked = st.radio(
            "Results",
            options,
            key=f"{key_prefix}{_PICK_SUFFIX}",
            format_func=lambda s: _format_hit(by_symbol[s]),
            label_visibility="collapsed",
        )
        chosen = by_symbol.get(picked)
        if chosen is not None:
            reason = reasons.get(chosen.yahoo_symbol, "")
            provenance = (
                "Provider search result — country and currency are confirmed when the "
                "price is fetched."
                if reason == "provider search"
                else f"Matched on {reason} in the bundled stock-master file."
            )
            st.caption(provenance)

    if allow_online and st.button(
        "Search online",
        key=f"{key_prefix}_online_btn",
        help="Query Yahoo Finance directly for symbols the bundled files do not list. "
        "Needs an internet connection.",
    ):
        with st.spinner(f"Searching for “{text}”…"):
            found = data_access.search_online(text, int(limit))
        st.session_state[online_key] = found
        if not found:
            st.warning(
                f"The provider returned no instruments for “{text}”. Check the "
                "spelling, or enter the Yahoo Finance symbol directly."
            )
        st.rerun()

    return chosen


# ---------------------------------------------------------------------------
# add-a-holding form
# ---------------------------------------------------------------------------
def add_holding_form(
    record: universe.StockRecord,
    key_prefix: str,
    on_add: Any,
) -> None:
    """Quantity / average-buy-price inputs for a picked instrument.

    ``on_add(symbol, quantity, buy_price)`` is invoked on submit. Prices are
    entered in the instrument's OWN currency — matching
    :mod:`src.analytics.positions`, which keeps ``buy_price`` native and
    converts only the money columns — so the label states which currency that
    is whenever the universe knows it.
    """
    currency = record.currency or ""
    price_label = f"Average buy price{f' ({currency})' if currency else ''}"

    with st.form(key=f"{key_prefix}_add_form", clear_on_submit=True):
        st.markdown(f"**{record.name}** · `{record.yahoo_symbol}`")
        c1, c2, c3 = st.columns([1, 1, 1])
        quantity = c1.number_input(
            "Quantity",
            min_value=0.0,
            value=10.0,
            step=1.0,
            format="%.4f",
            key=f"{key_prefix}_qty",
        )
        buy_price = c2.number_input(
            price_label,
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_price",
            help="Per share, in the instrument's own quote currency — not your reporting currency.",
        )
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        submitted = c3.form_submit_button("Add to portfolio", type="primary", width="stretch")

    if not submitted:
        return
    if quantity <= 0 or buy_price <= 0:
        st.error(
            "Quantity and average buy price must both be greater than zero — a row "
            "with either at zero is dropped by validation, so it would silently "
            "vanish from the analysis."
        )
        return
    on_add(record.yahoo_symbol, float(quantity), float(buy_price))


def add_to_watchlist_form(
    record: universe.StockRecord,
    key_prefix: str,
    on_add: Any,
) -> None:
    """Optional note + target price for a watchlist entry.

    ``on_add(symbol, note, target_price)`` runs on submit. ``target_price`` is
    ``None`` when left at zero: a watchlist entry with no target is normal, and
    storing 0.0 would render as a 100% downside target on the watchlist table.
    """
    currency = record.currency or ""
    with st.form(key=f"{key_prefix}_watch_form", clear_on_submit=True):
        st.markdown(f"**{record.name}** · `{record.yahoo_symbol}`")
        c1, c2 = st.columns([2, 1])
        note = c1.text_input(
            "Note (optional)",
            key=f"{key_prefix}_note",
            placeholder="Why you are watching it",
            max_chars=200,
        )
        target = c2.number_input(
            f"Target price{f' ({currency})' if currency else ''} (optional)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_target",
            help="Leave at 0 for no target. Shown as distance-to-target on the "
            "watchlist — it is a reminder you set, not a forecast.",
        )
        submitted = st.form_submit_button("Add to watchlist", type="primary")

    if submitted:
        on_add(record.yahoo_symbol, note.strip(), float(target) if target > 0 else None)


# ---------------------------------------------------------------------------
# convenience: append a searched symbol to the holdings grid
# ---------------------------------------------------------------------------
def append_holding(symbol: str, quantity: float, buy_price: float) -> bool:
    """Append one row to the session holdings grid. ``False`` if already present.

    Duplicate tickers are legal downstream —
    :func:`src.analytics.positions.clean_holdings` merges them at a
    quantity-weighted average cost — but appending a second row from a search
    box is almost always a mis-click, so this refuses and says so.
    """
    import pandas as pd

    wanted = universe.normalise_symbol(symbol, prefer_exchange=PREFERRED_EXCHANGE)
    current = sess.get_holdings()
    if current is not None and "ticker" in getattr(current, "columns", []):
        existing = {str(t).strip().upper() for t in current["ticker"].dropna()}
        if wanted in existing:
            return False
    row = pd.DataFrame({"ticker": [wanted], "quantity": [quantity], "buy_price": [buy_price]})
    sess.replace_holdings(pd.concat([current, row], ignore_index=True))
    return True
