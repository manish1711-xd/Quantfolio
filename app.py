"""QuantFolio — portfolio risk, performance and optimisation terminal.

This file is deliberately a *shell*: page configuration, theme injection,
navigation, and a last-resort error boundary. It contains no formulas and no
layout beyond the chrome.

    computation  ->  src/analytics/   (pure, offline-testable, no Streamlit)
    market data  ->  src/data/        (the only network boundary)
    figures      ->  src/charts/      (pure Plotly, no Streamlit)
    presentation ->  src/ui/          (this layer)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import traceback

import streamlit as st

from src import charts
from src.config import APP_NAME, APP_TAGLINE, DISCLAIMER
from src.ui import pipeline, session, sidebar
from src.ui import theme as ui_theme

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"**{APP_NAME}** — {APP_TAGLINE}.\n\n{DISCLAIMER}\n\n"
            "Market data: Yahoo Finance via yfinance (delayed, not real-time)."
        )
    },
)


# ---------------------------------------------------------------------------
# page registry
# ---------------------------------------------------------------------------
def _pages() -> dict[str, list]:
    """Build the navigation tree.

    Imported lazily inside the function so a syntax error in one page module
    surfaces in the error boundary below rather than as a blank screen at
    import time.
    """
    from src.ui.pages import (
        benchmark,
        correlation,
        forecast,
        holdings,
        news,
        optimizer,
        overview,
        performance,
        portfolio,
        rebalance,
        reports,
        research,
        risk,
        security,
        simulation,
        stress,
        transactions,
        watchlist,
    )

    def page(fn, title: str, icon: str, url: str, default: bool = False):
        return st.Page(fn, title=title, icon=icon, url_path=url, default=default)

    return {
        "Portfolio": [
            page(portfolio.render, "Build", ":material/edit_note:", "portfolio", default=True),
            page(overview.render, "Overview", ":material/dashboard:", "overview"),
            page(holdings.render, "Holdings", ":material/table_rows:", "holdings"),
            page(performance.render, "Performance", ":material/trending_up:", "performance"),
            page(transactions.render, "Trades & lots", ":material/receipt_long:", "trades"),
        ],
        "Risk": [
            page(risk.render, "Risk analytics", ":material/warning:", "risk"),
            page(benchmark.render, "Benchmark", ":material/compare_arrows:", "benchmark"),
            page(correlation.render, "Correlation & factors", ":material/hub:", "correlation"),
            page(stress.render, "Stress testing", ":material/bolt:", "stress"),
        ],
        "Strategy": [
            page(optimizer.render, "Optimiser", ":material/tune:", "optimiser"),
            page(simulation.render, "Monte Carlo", ":material/casino:", "simulation"),
            page(rebalance.render, "Rebalance", ":material/balance:", "rebalance"),
        ],
        "Research": [
            page(research.render, "Company research", ":material/manage_search:", "research"),
            page(security.render, "Security deep dive", ":material/query_stats:", "security"),
            page(forecast.render, "Price forecast", ":material/auto_graph:", "forecast"),
            page(watchlist.render, "Watchlist", ":material/visibility:", "watchlist"),
            page(news.render, "News & calendar", ":material/newspaper:", "news"),
            page(reports.render, "Export", ":material/download:", "reports"),
        ],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ui_theme.inject_css()
    charts.set_mode(ui_theme.active_base())
    session.init()

    navigation = st.navigation(_pages(), position="sidebar")

    cfg = sidebar.render()
    pipeline.ensure_result(cfg)

    navigation.run()

    st.divider()
    st.caption(f"⚠️ {DISCLAIMER}")


try:
    main()
except Exception:  # noqa: BLE001 - last-resort guard: never show a blank crash
    st.error(
        "Something unexpected went wrong. This is a bug — the details below are "
        "everything needed to reproduce it."
    )
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
