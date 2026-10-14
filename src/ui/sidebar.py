"""The control rail: portfolio source, analysis settings, saved portfolios.

Design note — the editable holdings grid deliberately does **not** live here.
A sidebar is ~330px wide; a five-column data editor squeezed into it is
miserable to use. The sidebar owns the things that are genuinely controls
(where the portfolio comes from, what window to analyse, which benchmark), and
the grid gets the full page width on the Portfolio screen where it belongs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    APP_NAME,
    APP_VERSION,
    DATA_SOURCE,
    DEFAULT_BENCHMARKS,
    DEFAULT_RISK_FREE,
    PERIOD_LABELS,
    SUPPORTED_BASE_CURRENCIES,
    VALID_PERIODS,
)
from src.data.store import PortfolioStore
from src.ui import csv_import, data_access, session
from src.ui import theme as ui_theme

SAMPLE_HOLDINGS = Path(__file__).resolve().parents[2] / "data" / "sample_portfolio.csv"
SAMPLE_TRANSACTIONS = Path(__file__).resolve().parents[2] / "data" / "sample_transactions.csv"


# ---------------------------------------------------------------------------
# portfolio source
# ---------------------------------------------------------------------------
def _load_sample() -> None:
    try:
        frame = pd.read_csv(SAMPLE_HOLDINGS)
        session.replace_holdings(frame)
        txns = pd.read_csv(SAMPLE_TRANSACTIONS)
        session.replace_transactions(txns)
        st.toast("Sample portfolio and trade ledger loaded.", icon="📁")
    except Exception as exc:  # noqa: BLE001 - a bad sample must not kill the app
        st.error(f"Could not load the sample portfolio: {exc}")


def _portfolio_source() -> None:
    ui_theme.section("Portfolio")

    left, right = st.columns(2)
    if left.button(
        "Load sample",
        width="stretch",
        help="A diversified 9-holding Indian portfolio with a US name for FX",
    ):
        _load_sample()
    if right.button("Clear", width="stretch", help="Empty the holdings table"):
        session.replace_holdings(session.default_holdings().iloc[0:0])
        session.replace_transactions(session.empty_transactions())

    # Import now goes through src.portfolio_importer, so a broker export whose
    # columns are named "Tradingsymbol / Qty / Average Price" works without the
    # user renaming anything -- and nothing is committed until they confirm.
    # Kept compact here: the sidebar is too narrow to review a mapping table
    # in, so it shows the verdict and the Portfolio page shows the detail.
    with st.expander("Import CSV", expanded=False):
        csv_import.import_panel("holdings", "side_holdings", compact=True)
        csv_import.import_panel(
            "transactions", "side_txns", label="Trade ledger (optional)", compact=True
        )
        st.caption(
            "Column names are detected automatically. For the full mapping and "
            "reject report, use **Import from CSV** on the Portfolio › Build page."
        )

    holdings = session.get_holdings()
    n = 0 if holdings is None else len(holdings)
    txn_note = f" · {len(session.get_transactions())} trades" if session.has_transactions() else ""
    st.caption(f"{n} row{'s' if n != 1 else ''} in the table{txn_note}")


# ---------------------------------------------------------------------------
# saved portfolios (SQLite)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _store() -> PortfolioStore:
    """One long-lived store per server process."""
    return PortfolioStore()


def _saved_portfolios() -> None:
    with st.expander("Saved portfolios", expanded=False):
        try:
            store = _store()
            saved = store.list_portfolios()
        except Exception as exc:  # noqa: BLE001 - persistence is a convenience, not a dependency
            st.caption(f"Local database unavailable: {exc}")
            return

        if saved:
            names = [row["name"] for row in saved]
            chosen = st.selectbox("Open", names, key="saved_pick", index=0)
            open_col, del_col = st.columns(2)
            if open_col.button("Open", width="stretch", key="saved_open"):
                holdings, txns = store.load_portfolio(chosen)
                session.replace_holdings(holdings)
                session.replace_transactions(txns)
                st.toast(f"Opened '{chosen}'.", icon="📂")
                st.rerun()
            if del_col.button("Delete", width="stretch", key="saved_delete"):
                store.delete_portfolio(chosen)
                st.toast(f"Deleted '{chosen}'.", icon="🗑")
                st.rerun()
        else:
            st.caption("No saved portfolios yet.")

        name = st.text_input("Save current as", placeholder="My long-term portfolio")
        if st.button("Save", width="stretch", disabled=not name.strip(), key="saved_save"):
            try:
                store.save_portfolio(
                    name.strip(),
                    session.get_holdings(),
                    base_currency=session.get_config().base_currency,
                    transactions=session.get_transactions(),
                )
                st.toast(f"Saved '{name.strip()}'.", icon="💾")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save: {exc}")


# ---------------------------------------------------------------------------
# analysis settings
# ---------------------------------------------------------------------------
def _settings(cfg):
    ui_theme.section("Analysis")

    period = st.selectbox(
        "History window",
        VALID_PERIODS,
        index=VALID_PERIODS.index(cfg.period) if cfg.period in VALID_PERIODS else 3,
        format_func=lambda p: PERIOD_LABELS.get(p, p),
        help="Drives volatility, drawdown, beta, correlation and the optimiser. "
        "Longer windows are statistically sturdier but blend market regimes.",
    )

    benchmark_names = list(DEFAULT_BENCHMARKS)
    benchmark = st.selectbox(
        "Benchmark",
        benchmark_names,
        index=benchmark_names.index(cfg.benchmark_name)
        if cfg.benchmark_name in benchmark_names
        else 0,
        help="Beta, alpha, tracking error and capture ratios are all measured against this index.",
    )

    currency = st.selectbox(
        "Reporting currency",
        SUPPORTED_BASE_CURRENCIES,
        index=SUPPORTED_BASE_CURRENCIES.index(cfg.base_currency)
        if cfg.base_currency in SUPPORTED_BASE_CURRENCIES
        else 0,
        help="Holdings quoted in other currencies are converted at the latest spot rate.",
    )

    # Re-default the risk-free rate when the reporting currency changes: 6.5%
    # is a sensible Indian G-Sec ballpark and a nonsense US Treasury one.
    suggested = DEFAULT_RISK_FREE.get(currency, 0.05)
    default_rf = suggested if currency != cfg.base_currency else cfg.risk_free_rate

    with st.expander("Advanced", expanded=False):
        rf_pct = st.number_input(
            "Risk-free rate (% p.a.)",
            min_value=0.0,
            max_value=25.0,
            value=round(default_rf * 100.0, 2),
            step=0.05,
            help=f"Used in Sharpe, Sortino, alpha, Treynor and the capital market line. "
            f"The {suggested * 100:.1f}% default is an illustrative {currency} sovereign "
            f"yield — replace it with today's actual number.",
        )
        confidence = st.select_slider(
            "VaR / CVaR confidence",
            options=[0.90, 0.95, 0.99],
            value=cfg.var_confidence if cfg.var_confidence in (0.90, 0.95, 0.99) else 0.95,
            format_func=lambda c: f"{c:.0%}",
            help="95% daily VaR is the loss exceeded on roughly one day in twenty.",
        )
        rolling_window = st.select_slider(
            "Rolling window (trading days)",
            options=[21, 42, 63, 126, 252],
            value=cfg.rolling_window if cfg.rolling_window in (21, 42, 63, 126, 252) else 63,
            help="Window for rolling volatility, Sharpe, beta and correlation. "
            "63 days is about one quarter.",
        )
        convert_fx = st.toggle(
            "Convert foreign holdings to reporting currency",
            value=cfg.convert_fx,
            help="Off: totals mix currencies and are only indicative.",
        )

    return replace(
        cfg,
        period=period,
        benchmark_name=benchmark,
        base_currency=currency,
        risk_free_rate=rf_pct / 100.0,
        var_confidence=float(confidence),
        rolling_window=int(rolling_window),
        convert_fx=bool(convert_fx),
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render():
    """Draw the sidebar and return the (possibly updated) :class:`AppConfig`."""
    cfg = session.get_config()

    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:.5rem;margin-bottom:.2rem">'
            f'<span style="font-size:1.15rem;font-weight:700;letter-spacing:-.02em">'
            f"{APP_NAME}</span>"
            f'<span style="font-size:.7rem;color:var(--qf-faint)">v{APP_VERSION}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Portfolio risk, performance & optimisation")

        _portfolio_source()
        _saved_portfolios()
        cfg = _settings(cfg)

        st.divider()
        if st.button("Analyze portfolio", type="primary", width="stretch"):
            session.request_run()
        if st.button(
            "Refresh market data",
            width="stretch",
            help="Drop every cached quote and price series, then refetch.",
        ):
            data_access.clear_market_cache()
            session.request_run()
            st.toast("Market cache cleared — refetching.", icon="♻️")

        st.caption(DATA_SOURCE)

    session.set_config(cfg)
    return cfg
