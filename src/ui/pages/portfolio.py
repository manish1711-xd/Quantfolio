"""Portfolio builder — the input surface and the app's landing screen.

Everything the user types lives here: stock search, the smart CSV importer,
the holdings grid, the optional trade ledger, and quick-add chips for common
Indian and US symbols. Validation runs live against
:func:`src.analytics.positions.clean_holdings`, so problems are visible
*before* the user clicks Analyze rather than as a wall of warnings afterwards.

Four ways in, in the order most people want them:

* **Search** — type "Reliance", pick the row, enter quantity and cost. No need
  to know that the Yahoo symbol is ``RELIANCE.NS``.
* **Import** — a broker CSV whose columns are named whatever that broker
  names them; :mod:`src.portfolio_importer` maps them and shows a preview.
* **Grid** — type or paste directly, for people who already know their symbols.
* **Quick add** — one-click chips for the usual large caps and ETFs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import positions as pos
from src.config import APP_NAME, APP_TAGLINE, DISCLAIMER, TICKER_SUGGESTIONS
from src.ui import components as ui
from src.ui import csv_import, session, stock_search
from src.ui import theme as ui_theme


# ---------------------------------------------------------------------------
# search & add
# ---------------------------------------------------------------------------
def _search_and_add() -> None:
    """Name-based instrument search, then a quantity / cost form."""
    ui_theme.section("Search & add", "🔍")
    record = stock_search.search_box("build", label="Search stock")
    if record is None:
        return

    def _add(symbol: str, quantity: float, buy_price: float) -> None:
        if stock_search.append_holding(symbol, quantity, buy_price):
            st.toast(f"Added {symbol}.", icon="➕")
            st.rerun()
        else:
            st.warning(
                f"{symbol} is already in the holdings grid. Edit its quantity there "
                "instead of adding a second row."
            )

    stock_search.add_holding_form(record, "build", _add)


# ---------------------------------------------------------------------------
# holdings grid
# ---------------------------------------------------------------------------
def _holdings_editor() -> None:
    ui_theme.section("Holdings", "💼")
    st.caption(
        "Indian equities use the Yahoo suffix — `RELIANCE.NS`, `TCS.NS`, `INFY.NS` "
        "(NSE) or `.BO` (BSE). ETFs (`NIFTYBEES.NS`, `GOLDBEES.NS`) and "
        "international tickers (`AAPL`, `MSFT`) work too. Duplicate rows are "
        "merged with a quantity-weighted average buy price."
    )

    edited = st.data_editor(
        session.get_holdings(),
        num_rows="dynamic",
        width="stretch",
        key=f"holdings_editor_{st.session_state[session.EDITOR_VERSION]}",
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker",
                required=True,
                width="medium",
                help="Yahoo Finance symbol",
            ),
            "quantity": st.column_config.NumberColumn(
                "Quantity",
                min_value=0.0,
                format="%.4f",
                width="small",
            ),
            "buy_price": st.column_config.NumberColumn(
                "Avg buy price",
                min_value=0.0,
                format="%.2f",
                width="small",
                help="Per share, in the instrument's own currency",
            ),
            "buy_date": st.column_config.TextColumn(
                "First bought",
                width="small",
                help="Optional. Used only for display unless a trade ledger is supplied.",
            ),
        },
    )
    st.session_state[session.HOLDINGS] = edited


def _import_section() -> None:
    """Full-width broker-CSV import with a mapping and reject preview."""
    ui_theme.section("Import from a broker export", "📄")
    st.caption(
        "Upload a CSV or Excel export from Zerodha, Groww, Upstox, an international "
        "broker, or your own spreadsheet. Column names are detected automatically — "
        "you review the mapping and every rejected row **before** anything is imported."
    )
    holdings_tab, ledger_tab = st.tabs(["Holdings", "Trade ledger"])
    with holdings_tab:
        csv_import.import_panel("holdings", "page_holdings")
    with ledger_tab:
        csv_import.import_panel("transactions", "page_txns")


def _quick_add() -> None:
    with st.expander("Quick add", expanded=False):
        st.caption("Adds the symbol with a placeholder quantity and price — edit them in the grid.")
        for market, tickers in TICKER_SUGGESTIONS.items():
            st.markdown(f"**{market}**")
            cols = st.columns(min(len(tickers), 7))
            for i, ticker in enumerate(tickers):
                if cols[i % len(cols)].button(ticker, key=f"add_{ticker}", width="stretch"):
                    _append_ticker(ticker)
                    st.rerun()


def _append_ticker(ticker: str) -> None:
    current = session.get_holdings()
    if current is not None and "ticker" in current.columns:
        existing = {str(t).strip().upper() for t in current["ticker"].dropna()}
        if ticker.upper() in existing:
            st.toast(f"{ticker} is already in the portfolio.", icon="ℹ️")
            return
    row = pd.DataFrame({"ticker": [ticker], "quantity": [1.0], "buy_price": [100.0]})
    session.replace_holdings(pd.concat([current, row], ignore_index=True))


# ---------------------------------------------------------------------------
# trade ledger
# ---------------------------------------------------------------------------
def _transactions_editor() -> None:
    ui_theme.section("Trade ledger", "🧾")
    st.caption(
        "Optional. With a ledger the app runs proper FIFO lot accounting — "
        "realised vs unrealised P&L, holding periods, short/long-term split, and "
        "a money-weighted return (XIRR) that accounts for *when* you invested. "
        "Without one, every statistic assumes today's share counts across the "
        "whole window."
    )

    frame = session.get_transactions()
    if frame is None or frame.empty:
        frame = session.empty_transactions()

    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        width="stretch",
        key=f"txn_editor_{st.session_state[session.TXN_EDITOR_VERSION]}",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="medium"),
            "trade_date": st.column_config.TextColumn("Date", width="small", help="YYYY-MM-DD"),
            "action": st.column_config.SelectboxColumn(
                "Action", options=["BUY", "SELL"], width="small"
            ),
            "quantity": st.column_config.NumberColumn(
                "Qty", min_value=0.0, format="%.4f", width="small"
            ),
            "price": st.column_config.NumberColumn(
                "Price", min_value=0.0, format="%.2f", width="small"
            ),
            "fees": st.column_config.NumberColumn(
                "Fees",
                min_value=0.0,
                format="%.2f",
                width="small",
                help="Brokerage, STT, stamp duty. Buy fees capitalise into cost basis; "
                "sell fees reduce proceeds.",
            ),
        },
    )
    st.session_state[session.TRANSACTIONS] = edited

    if not edited.empty:
        c1, _c2 = st.columns([1, 3])
        if c1.button("Derive holdings from ledger", width="stretch"):
            _holdings_from_ledger(edited)


def _holdings_from_ledger(frame: pd.DataFrame) -> None:
    """Replace the holdings grid with the open FIFO position implied by the ledger."""
    from src.analytics import lots

    txns, problems = lots.parse_transactions(frame)
    if problems:
        ui.warning_list(problems, "Ledger issues")
    if not txns:
        st.error("No valid transactions to derive holdings from.")
        return
    derived = lots.holdings_from_transactions(txns)
    if derived.empty:
        st.warning("The ledger closes every position — there is nothing left to hold.")
        return
    session.replace_holdings(derived)
    st.toast(f"Derived {len(derived)} open positions from the ledger.", icon="🧮")
    st.rerun()


# ---------------------------------------------------------------------------
# live validation
# ---------------------------------------------------------------------------
def _validation_panel() -> None:
    clean, problems = pos.clean_holdings(session.get_holdings())

    ui_theme.section("Validation", "✅")
    cards = [
        {
            "label": "Valid holdings",
            "value": str(len(clean)),
            "tone": "gain" if len(clean) else "loss",
            "footnote": "after merging duplicates",
        },
        {
            "label": "Issues found",
            "value": str(len(problems)),
            "tone": "warn" if problems else "neutral",
            "footnote": "rows that will be skipped",
        },
        {
            "label": "Ledger rows",
            "value": str(len(session.get_transactions())),
            "tone": "neutral",
            "footnote": "unlocks XIRR & realised P&L",
        },
    ]
    ui.kpi_row(cards, columns=3)

    if problems:
        ui.warning_list(problems, "Rows that will be skipped")

    if not clean.empty:
        st.caption("Cleaned input the analysis will actually use:")
        ui.dataframe(clean, height=min(38 * (len(clean) + 1) + 8, 320))


# ---------------------------------------------------------------------------
# welcome
# ---------------------------------------------------------------------------
def _welcome() -> None:
    st.markdown(
        f"""
Welcome to **{APP_NAME}** — {APP_TAGLINE.lower()}.

**Three steps:** enter holdings below (or hit **Load sample** in the sidebar) →
pick a history window and benchmark → click **Analyze portfolio**.

**What you get:** position-level P&L, thirteen risk statistics with their
formulas shown, drawdown episode analysis, benchmark attribution, mean-variance
optimisation with an efficient frontier, Monte Carlo projections, historical
stress replay, a rebalancing trade list, and a **machine-learning price
forecast** reported alongside its own out-of-sample record against a random
walk — because a forecast is worth exactly what its backtest says it is.

**About the data:** prices come from Yahoo Finance via `yfinance`. NSE/BSE
quotes are typically **delayed ~15 minutes**, and outside market hours you see
the **latest available close**. Every price in this app carries its own
timestamp and freshness label — it never claims to be real-time, because it
isn't.
        """
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Portfolio",
        "Enter holdings and an optional trade ledger, then analyse",
        icon="📥",
    )

    if not session.has_result():
        _welcome()

    _search_and_add()
    _holdings_editor()
    _quick_add()

    with st.expander("Import from CSV / broker export", expanded=False):
        _import_section()

    _validation_panel()

    with st.expander("Trade ledger (optional — unlocks XIRR and realised P&L)", expanded=False):
        _transactions_editor()

    st.divider()
    left, right = st.columns([1, 2])
    if left.button("Analyze portfolio", type="primary", width="stretch", key="analyze_main"):
        session.request_run()
        st.rerun()
    right.caption(f"⚠️ {DISCLAIMER}")
