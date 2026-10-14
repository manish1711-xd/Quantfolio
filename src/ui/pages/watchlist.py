"""Watchlist — instruments tracked without owning them.

The only page in the application that does NOT require a completed analysis:
watching a stock is independent of whether you have run the portfolio, and
gating it behind Analyze would make the feature useless on a first visit.

It fetches its own quotes and profiles through the cached
:mod:`src.ui.data_access` layer rather than reading them off
:class:`~src.models.AnalysisResult`, because a watched name is by definition
not in the positions table. Every number is a live quote or an em dash — the
page never fills a gap with a plausible value.

Persistence is explicit. Edits live in session state and reach SQLite when the
user presses Save, so a stray click cannot rewrite a saved list.

Price alerts hang off the targets set here (:mod:`src.ui.alerts_panel` for the
configuration, :mod:`src.alerts` for the rules). This page never *checks* them:
the same ``python -m src.reporting.cli`` command that sends scheduled
tearsheets does that, because Streamlit has no background worker. What the page
does offer is a preview computed from the prices it has already fetched, which
sends nothing and records nothing -- it exists so "is my rule set up right?"
does not have to wait for the next cron run.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import alerts as al
from src import watchlist as wl
from src.config import DATA_SOURCE, DISCLAIMER
from src.data.store import PortfolioStore
from src.ui import alerts_panel, data_access, session, stock_search
from src.ui import components as ui
from src.ui import formatters as fmt
from src.ui import theme as ui_theme

#: How close to a target counts as "near your level" on this page.
_NEAR_TARGET_PCT = 3.0


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _store() -> PortfolioStore:
    """One long-lived store per server process (shared with the sidebar)."""
    return PortfolioStore()


def _load_once() -> None:
    """Hydrate the watchlist from SQLite on the first visit of a session.

    Guarded by a flag rather than by emptiness: without it, deliberately
    clearing the watchlist would be undone by the next rerun reloading the
    saved rows, which is maddening and looks like a bug.
    """
    if st.session_state.get(session.WATCHLIST_LOADED):
        return
    st.session_state[session.WATCHLIST_LOADED] = True
    try:
        session.set_watchlist(wl.from_frame(_store().load_watchlist()))
    except Exception as exc:  # noqa: BLE001 - persistence is a convenience
        st.caption(f"Saved watchlist unavailable: {exc}")


def _save() -> None:
    try:
        stored = _store().save_watchlist(wl.to_frame(session.get_watchlist()))
        st.toast(f"Watchlist saved ({stored} instrument(s)).", icon="💾")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save the watchlist: {exc}")


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
def _add_section() -> None:
    ui_theme.section("Add an instrument", "🔍")
    record = stock_search.search_box(
        "watch",
        label="Search stock, ETF or index",
        placeholder="Reliance, HDFC Bank, AAPL, Nifty 50…",
    )
    if record is None:
        return

    def _add(symbol: str, note: str, target: float | None) -> None:
        # The direction is left unset here on purpose. This form has no price
        # for the instrument yet -- it has only just been searched for -- and
        # guessing a direction from nothing would attach a rule the user never
        # chose. The reference price is captured on the first table build
        # instead, and the alert is switched on from "Edit or remove" where
        # both the price and an explicit direction are available.
        items, problem = wl.add(session.get_watchlist(), symbol, note=note, target_price=target)
        if problem:
            st.warning(problem)
            return
        session.set_watchlist(items)
        st.toast(f"{symbol} added to the watchlist.", icon="👁")
        if target:
            st.info(
                f"Target {target:,.2f} saved for {symbol}. To be **e-mailed** when "
                "it is reached, open *Edit or remove* below and set the direction "
                "— a target on its own never sends anything."
            )
        st.rerun()

    stock_search.add_to_watchlist_form(record, "watch", _add)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def _build_table(items) -> pd.DataFrame:
    """Fetch quotes and profiles for the watched names and build the table."""
    tickers = tuple(item.ticker for item in items)
    quotes, errors = data_access.load_quotes(tickers)
    profiles = data_access.load_profiles(tickers)

    result = session.get_result()
    held: list[str] = []
    if result is not None and getattr(result, "positions", None) is not None:
        positions = result.positions
        if hasattr(positions, "columns") and "ticker" in positions.columns:
            held = [str(t) for t in positions["ticker"]]

    if errors:
        ui.warning_list(
            [f"{ticker}: {message}" for ticker, message in errors.items()],
            "Instruments that could not be priced",
        )

    _capture_reference_prices(items, quotes)
    return wl.watch_table(items, quotes, profiles, held_tickers=held)


def _capture_reference_prices(items, quotes) -> None:
    """Record the price against which a target was set, once, if unknown.

    This is what lets :func:`src.alerts.infer_direction` later tell an upside
    level from an entry level. It is written only when ``reference_price`` is
    still empty, and never overwritten: re-capturing it against today's price
    could silently invert the meaning of a rule the user set months ago.
    """
    missing = [
        item
        for item in items
        if item.has_target and item.reference_price is None and item.ticker in quotes
    ]
    if not missing:
        return
    updated = items
    for item in missing:
        price = getattr(quotes.get(item.ticker), "price", None)
        if price is None:
            continue
        updated = wl.update_item(updated, item.ticker, reference_price=float(price))
    if updated is not items:
        session.set_watchlist(updated)


def _kpis(table: pd.DataFrame) -> None:
    stats = wl.summarise(table)
    ui.kpi_row(
        [
            {
                "label": "Watching",
                "value": str(stats["n_items"]),
                "tone": "neutral",
                "footnote": f"{stats['n_priced']} priced · {stats['n_with_target']} with a target",
            },
            {
                "label": "Average day change",
                "value": fmt.fmt_pct(stats["avg_day_change_pct"], signed=True),
                "tone": fmt.delta_tone(stats["avg_day_change_pct"]),
                "footnote": "unweighted mean across priced names",
            },
            {
                "label": "Best today",
                "value": fmt.safe(stats["best_ticker"]),
                "tone": "gain",
                "delta": fmt.fmt_pct(stats["best_day_change_pct"], signed=True),
                "delta_tone": "gain",
            },
            {
                "label": "Weakest today",
                "value": fmt.safe(stats["worst_ticker"]),
                "tone": "loss",
                "delta": fmt.fmt_pct(stats["worst_day_change_pct"], signed=True),
                "delta_tone": "loss",
            },
        ],
        columns=4,
    )
    if stats["n_in_portfolio"]:
        st.caption(
            f"ℹ {stats['n_in_portfolio']} watched name(s) are also holdings — flagged "
            "in the table so you do not double-count a position you already own."
        )


def _alert_cell(row) -> str:
    """One short phrase per row describing the alert, or "" for none.

    Three raw columns (``direction``, ``alert_enabled``, ``alert_state``) are
    the storage shape; a reader wants one cell that says whether they will be
    told and what they are waiting for.
    """
    if not bool(row.get("alert_enabled")):
        return ""
    arrow = "≥" if al.normalise_direction(row.get("direction")) == al.ABOVE else "≤"
    level = row.get("target_price")
    try:
        shown = f"{float(level):,.2f}"
    except (TypeError, ValueError):
        shown = "?"
    state = str(row.get("alert_state") or al.ARMED).lower()
    mark = "waiting" if state == al.ARMED else "sent"
    return f"{arrow} {shown} · {mark}"


def _render_table(table: pd.DataFrame) -> None:
    ui_theme.section("Watched instruments", "👁")
    table = table.copy()
    table["alert_state"] = table.apply(_alert_cell, axis=1)
    table = table.drop(columns=["direction", "alert_enabled"], errors="ignore")
    display = table.rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "sector": "Sector",
            "currency": "Ccy",
            "price": "Price",
            "day_change_pct": "Day %",
            "week52_low": "52w low",
            "week52_high": "52w high",
            "range_position_pct": "In 52w range %",
            "target_price": "Target",
            "alert_state": "Alert",
            "to_target_pct": "To target %",
            "in_portfolio": "Held",
            "note": "Note",
            "added_at": "Added",
        }
    ).drop(columns=["as_of"], errors="ignore")

    ui.dataframe(
        display,
        height=min(38 * (len(display) + 1) + 8, 520),
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Day %": st.column_config.NumberColumn(format="%+.2f%%"),
            "52w low": st.column_config.NumberColumn(format="%.2f"),
            "52w high": st.column_config.NumberColumn(format="%.2f"),
            "In 52w range %": st.column_config.ProgressColumn(
                format="%.0f%%",
                min_value=0,
                max_value=100,
                help="0% = at the 52-week low, 100% = at the 52-week high. Blank when "
                "the provider reports no 52-week range.",
            ),
            "Target": st.column_config.NumberColumn(format="%.2f"),
            "To target %": st.column_config.NumberColumn(
                format="%+.2f%%",
                help="Move required to reach the target you set. Positive means the "
                "target is above the current price.",
            ),
            "Held": st.column_config.CheckboxColumn(help="Also a holding in your portfolio"),
            "Alert": st.column_config.TextColumn(
                help="Blank unless you have switched an e-mail alert on for this "
                "level. 'waiting' means the level has not been reported yet; "
                "'sent' means it has, and will not be reported again until the "
                "price retreats past your level.",
            ),
        },
    )

    near = wl.triggered(table, tolerance_pct=_NEAR_TARGET_PCT)
    if not near.empty:
        st.info(
            "**Near your level:** "
            + ", ".join(
                f"`{row.ticker}` ({fmt.fmt_pct(row.to_target_pct, signed=True)} to target)"
                for row in near.itertuples()
            )
            + f" — within {_NEAR_TARGET_PCT:.0f}% of the target *you* set. This is "
            "arithmetic against your own number, not a signal."
        )

    ui.download_row(
        [
            (
                "Download watchlist (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                "watchlist.csv",
                "text/csv",
            )
        ]
    )


# ---------------------------------------------------------------------------
# manage
# ---------------------------------------------------------------------------
def _manage_section(items, table: pd.DataFrame | None = None) -> None:
    def price_for(ticker: str) -> float | None:
        """Latest price for one ticker, from the table already on screen.

        Used to pre-select the alert direction. Passing it in rather than
        re-fetching keeps this section free of a second round of quote
        requests for prices the page has already paid for.
        """
        if table is None or table.empty or "ticker" not in table.columns:
            return None
        match = table.loc[table["ticker"].astype(str) == str(ticker), "price"]
        if match.empty:
            return None
        try:
            value = float(match.iloc[0])
        except (TypeError, ValueError):
            return None
        return value if value == value and value > 0 else None

    with st.expander("Edit or remove", expanded=False):
        symbols = [item.ticker for item in items]
        chosen = st.selectbox("Instrument", symbols, key="watch_manage_pick")
        current = next((i for i in items if i.ticker == chosen), None)
        if current is None:  # pragma: no cover - selectbox cannot return an absent key
            return

        c1, c2 = st.columns([2, 1])
        note = c1.text_input(
            "Note", value=current.note, key="watch_edit_note", max_chars=wl.MAX_NOTE_LENGTH
        )
        target = c2.number_input(
            "Target price (0 clears it)",
            min_value=0.0,
            value=float(current.target_price or 0.0),
            step=1.0,
            format="%.2f",
            key="watch_edit_target",
        )

        rule = alerts_panel.rule_editor(current, price_for(chosen), "watch_edit")

        b1, b2, b3 = st.columns(3)
        if b1.button("Update", width="stretch", key="watch_update"):
            alert_enabled, direction = rule if rule is not None else (None, None)
            session.set_watchlist(
                wl.update_item(
                    items,
                    chosen,
                    note=note,
                    target_price=target if target > 0 else None,
                    clear_target=target <= 0,
                    direction=direction,
                    alert_enabled=alert_enabled,
                )
            )
            st.toast(f"{chosen} updated.", icon="✏️")
            st.rerun()
        if b2.button("Remove", width="stretch", key="watch_remove"):
            session.set_watchlist(wl.remove(items, chosen))
            st.toast(f"{chosen} removed.", icon="🗑")
            st.rerun()
        if b3.button("Add to portfolio", width="stretch", key="watch_promote"):
            st.session_state["build_query"] = chosen
            st.info(
                f"Open **Portfolio › Build** and search `{chosen}` to add it with a "
                "quantity and cost basis. A watchlist entry has neither, so it cannot "
                "become a holding on its own."
            )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Watchlist",
        "Track instruments you do not own — prices, 52-week position and your own targets",
        icon="👁",
    )
    st.caption(
        "A watchlist is intentionally separate from the portfolio: a watched name has "
        "no quantity and no cost basis, so it never enters P&L, weights, risk "
        "statistics or attribution."
    )

    _load_once()
    _add_section()

    items = session.get_watchlist()
    st.divider()

    if not items:
        ui.empty_state(
            "Nothing on the watchlist yet",
            "Search above for a company, ETF or index and add it. Prices, the "
            "52-week range position and the distance to any target you set appear here.",
            icon="👁",
            hint="The watchlist is saved locally and is shared across every portfolio you model.",
        )
        return

    with ui.loading("Fetching prices for watched instruments…"):
        table = _build_table(items)

    _kpis(table)
    _render_table(table)
    _manage_section(items, table)

    st.divider()
    ui_theme.section("PRICE ALERTS", "🔔")
    st.caption(
        "Be e-mailed when a watched instrument reaches a level you set. The "
        "rules live here; the checking is done by a command your operating "
        "system runs, because Streamlit has no background worker."
    )
    alerts_panel.render(items, table)

    st.divider()
    left, right = st.columns([1, 1])
    if left.button("Save watchlist", type="primary", width="stretch", key="watch_save"):
        _save()
    if right.button(
        "Clear watchlist",
        width="stretch",
        key="watch_clear",
        help="Empties the list here. Press Save afterwards to make it permanent.",
    ):
        session.set_watchlist(())
        st.rerun()

    ui.freshness_bar(
        max((v for v in table["as_of"] if pd.notna(v)), default=None),
        DATA_SOURCE,
        extra=[f"{len(items)} instrument(s)"],
    )
    st.caption(
        f"⚠️ {DISCLAIMER} A target price is a reminder you set, not a forecast or a recommendation."
    )
