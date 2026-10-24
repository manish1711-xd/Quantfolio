"""Equity Research Dashboard - Institutional Stock Analysis and Report Generator.

Presentation layer: zero formula calculations, zero direct network I/O.
Consumes pure computation modules in src.equity_research:
- data_access: unified fetcher
- fundamental: margins, ROE, ROCE, EPS, debt ratios, cash flows
- technical: Wilder RSI, MACD, SMAs, EMAs, Volatility, Beta, Sharpe, Max Drawdown
- dcf: 5-year DCF projections, Gordon Growth, Sensitivity Heatmap, Bull/Base/Bear scenarios
- peer_comparison: peer multiples, sector averages, percentile rankings
- ai_report: institutional equity research narrative generator
- export: PDF tearsheet and formatted multi-tab Excel models
- history: SQLite persistence for equity research teardowns
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts
from src.charts import theme as ch_theme
from src.config import DISCLAIMER
from src.equity_research.ai_report import generate_report_sections
from src.equity_research.data_access import fetch_research_bundle
from src.equity_research.dcf import (
    DCFInputs,
    calculate_dcf_valuation,
    generate_scenario_analysis,
)
from src.equity_research.export import (
    export_equity_report_pdf,
    export_equity_research_excel,
)
from src.equity_research.fundamental import analyze_fundamentals
from src.equity_research.history import (
    delete_research_record,
    load_research_history,
    save_research_record,
)
from src.equity_research.models import (
    EquityResearchReport,
    FundamentalData,
    TechnicalData,
    ValuationResult,
)
from src.equity_research.peer_comparison import analyze_peer_comparison
from src.equity_research.technical import analyze_technicals
from src.equity_research.market_segments import compute_market_segment_ratios, get_nse_market_snapshot
from src.ui import components as ui
from src.ui import theme as ui_theme
from src.ui.formatters import DASH, fmt_date, fmt_money, fmt_num, fmt_ratio_pct

logger = logging.getLogger(__name__)

POPULAR_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "TATAMOTORS.NS",
    "ITC.NS",
    "AAPL",
    "MSFT",
    "GOOGL",
]


def _format_val(val: Any, formatter=fmt_num, default: str = DASH) -> str:
    """Format numeric values safely."""
    if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
        return default
    try:
        return formatter(val)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Interactive Plotly Chart Builders (Presentation Layer)
# ---------------------------------------------------------------------------
def _build_price_technical_chart(ohlcv: pd.DataFrame, tech: TechnicalData, ticker: str) -> go.Figure:
    """Candlestick/Line price history with overlay SMA 20/50/200 and EMA 20/50."""
    fig = go.Figure()
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return fig

    # Close price line
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=ohlcv["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#0EA5E9", width=2),
        )
    )

    # Moving averages calculated over the series
    close = ohlcv["Close"]
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=sma20,
                mode="lines",
                name="SMA 20",
                line=dict(color="#F59E0B", width=1.2, dash="dash"),
            )
        )
    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=sma50,
                mode="lines",
                name="SMA 50",
                line=dict(color="#10B981", width=1.2, dash="dot"),
            )
        )
    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=sma200,
                mode="lines",
                name="SMA 200",
                line=dict(color="#8B5CF6", width=1.5),
            )
        )

    fig.update_layout(
        title=dict(text=f"{ticker} — Price History & Moving Averages"),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return ch_theme.apply(fig)


def _build_rsi_chart(ohlcv: pd.DataFrame, ticker: str) -> go.Figure:
    """14-Period Wilder RSI chart with 70/30 thresholds."""
    fig = go.Figure()
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return fig

    close = ohlcv["Close"].dropna()
    if len(close) < 15:
        return fig

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))

    fig.add_trace(
        go.Scatter(
            x=rsi_series.index,
            y=rsi_series,
            mode="lines",
            name="14-Day RSI",
            line=dict(color="#8B5CF6", width=1.8),
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold (30)")

    fig.update_layout(
        title=dict(text="Relative Strength Index (14-Day Wilder)"),
        yaxis=dict(range=[0, 100]),
        xaxis_title="Date",
        yaxis_title="RSI",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return ch_theme.apply(fig)


def _build_macd_chart(ohlcv: pd.DataFrame, ticker: str) -> go.Figure:
    """MACD Line, Signal line, and color-coded histogram."""
    fig = go.Figure()
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return fig

    close = ohlcv["Close"].dropna()
    if len(close) < 35:
        return fig

    fast_ema = close.ewm(span=12, adjust=False).mean()
    slow_ema = close.ewm(span=26, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    colors = ["#10B981" if val >= 0 else "#EF4444" for val in histogram]

    fig.add_trace(
        go.Bar(
            x=histogram.index,
            y=histogram,
            name="Histogram",
            marker_color=colors,
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macd_line.index,
            y=macd_line,
            mode="lines",
            name="MACD (12, 26)",
            line=dict(color="#0EA5E9", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=signal_line.index,
            y=signal_line,
            mode="lines",
            name="Signal (9)",
            line=dict(color="#F59E0B", width=1.5),
        )
    )

    fig.update_layout(
        title=dict(text="MACD (12, 26, 9)"),
        xaxis_title="Date",
        yaxis_title="MACD",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return ch_theme.apply(fig)


def _build_dcf_cashflow_chart(val: ValuationResult, curr_symbol: str) -> go.Figure:
    """Discounted Cash Flow 5-year present value contributions."""
    fig = go.Figure()
    if not val.dcf_interpolated:
        return fig

    labels = [k.replace("_", " ").title() for k in val.dcf_interpolated.keys()]
    values = list(val.dcf_interpolated.values())

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#0EA5E9",
            text=[f"{curr_symbol}{v:,.0f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(text="5-Year Projected Cash Flow Present Values (PV)"),
        xaxis_title="Projection Horizon",
        yaxis_title="Discounted PV",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return ch_theme.apply(fig)


def _build_sensitivity_heatmap(sens_df: pd.DataFrame, curr_symbol: str) -> go.Figure:
    """Plotly heatmap of DCF sensitivity table (WACC vs Growth)."""
    fig = go.Figure()
    if sens_df is None or sens_df.empty:
        return fig

    z_vals = sens_df.values
    x_labels = list(sens_df.columns)
    y_labels = list(sens_df.index)

    fig.add_trace(
        go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            texttemplate=f"{curr_symbol}%{{z:,.1f}}",
            hoverongaps=False,
            colorbar=dict(title="Intrinsic Target"),
        )
    )

    fig.update_layout(
        title=dict(text="DCF Intrinsic Value Sensitivity (WACC vs. Terminal Growth)"),
        xaxis_title="Terminal Growth Rate",
        yaxis_title="Discount Rate (WACC)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return ch_theme.apply(fig)


def _build_peer_multiples_chart(peers_df: pd.DataFrame, target_ticker: str, target_pe: float) -> go.Figure:
    """Peer comparison bar chart for valuation multiples."""
    fig = go.Figure()
    if peers_df is None or peers_df.empty or "pe_ratio" not in peers_df.columns:
        return fig

    clean_peers = peers_df.dropna(subset=["pe_ratio"]).copy()
    tickers = list(clean_peers["ticker"])
    pes = list(clean_peers["pe_ratio"])
    colors = ["#94A3B8"] * len(tickers)

    if not np.isnan(target_pe) and target_pe > 0:
        tickers.insert(0, f"{target_ticker} (Target)")
        pes.insert(0, target_pe)
        colors.insert(0, "#0EA5E9")

    fig.add_trace(
        go.Bar(
            x=tickers,
            y=pes,
            marker_color=colors,
            text=[f"{p:.1f}x" for p in pes],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(text="Price-to-Earnings (P/E) Multiple vs. Industry Peers"),
        xaxis_title="Company",
        yaxis_title="P/E Ratio",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return ch_theme.apply(fig)


# ---------------------------------------------------------------------------
# Main Page Render Function
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the Equity Research Report Generator Streamlit Dashboard."""
    ui_theme.page_header(
        "Equity Research Terminal",
        "AI-Powered Institutional Stock Research & DCF Valuation",
        icon="📑",
    )

    # 1. Search & Configuration Bar
    with st.container():
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1.5], vertical_alignment="bottom")
        with c1:
            ticker_input = st.text_input(
                "Stock Ticker",
                value="RELIANCE.NS",
                placeholder="e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS, AAPL",
                help="Enter any listed stock ticker. Indian NSE stocks end in .NS, BSE in .BO.",
            ).strip().upper()
        with c2:
            period_opt = st.selectbox(
                "Price History",
                options=["6mo", "1y", "2y", "5y"],
                index=1,
            )
        with c3:
            benchmark_opt = st.selectbox(
                "Benchmark",
                options=["^NSEI", "^BSESN", "^GSPC"],
                index=0,
                help="^NSEI = Nifty 50, ^BSESN = Sensex, ^GSPC = S&P 500",
            )
        with c4:
            run_btn = st.button("Generate Report", type="primary", use_container_width=True)

    if not ticker_input:
        ui.empty_state("No Ticker Selected", "Please enter a stock ticker above to generate a research report.", icon="🔍")
        return

    # Cache report state in session
    if "equity_report_bundle" not in st.session_state or st.session_state.get("current_research_ticker") != ticker_input or run_btn:
        with st.spinner(f"Fetching public filings, market data, and peer metrics for {ticker_input}..."):
            bundle, warnings = fetch_research_bundle(ticker_input, period=period_opt, benchmark=benchmark_opt)
            st.session_state["equity_report_bundle"] = bundle
            st.session_state["equity_report_warnings"] = warnings
            st.session_state["current_research_ticker"] = ticker_input

    bundle = st.session_state.get("equity_report_bundle")
    warnings = st.session_state.get("equity_report_warnings", [])
    if not bundle:
        st.error(f"Failed to acquire research bundle for {ticker_input}.")
        return

    quote = bundle["quote"]
    price = quote.get("price")
    if not quote or price is None or np.isnan(price) or price <= 0:
        st.error(f"Could not retrieve a valid price or quote for ticker '{ticker_input}'. Please check the symbol (e.g. RELIANCE.NS, AAPL, TCS.NS) and ensure your connection or ticker format is valid.")
        return
    statements = bundle["statements"]
    ohlcv = bundle["ohlcv"]
    bench_hist = bundle["bench_hist"]
    news_items = bundle["news"]
    peers_df = bundle["peers"]

    # 2. Compute Core Analytics
    fundamentals = analyze_fundamentals(statements, quote_info=quote)
    technicals = analyze_technicals(ohlcv, bench_df=bench_hist)

    # Valuation Knobs (Overrides) in Sidebar/Expander
    with st.expander("🛠️ Valuation Modeling Controls & Assumptions", expanded=False):
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            wacc_input = st.slider(
                "Discount Rate (WACC)",
                min_value=0.05,
                max_value=0.20,
                value=0.10,
                step=0.005,
                format="%.1f%%",
                help="Weighted Average Cost of Capital used to discount future cash flows.",
            )
        with col_w2:
            growth_input = st.slider(
                "Forecast FCF Growth Rate",
                min_value=0.00,
                max_value=0.30,
                value=0.08,
                step=0.01,
                format="%.1f%%",
                help="Annual geometric growth rate projected for the explicit 5-year horizon.",
            )
        with col_w3:
            term_g_input = st.slider(
                "Terminal Growth Rate",
                min_value=0.01,
                max_value=0.05,
                value=0.03,
                step=0.005,
                format="%.1f%%",
                help="Perpetual growth rate into infinity. Must be strictly lower than WACC.",
            )

    # 3. Compute DCF Valuation
    base_fcf = fundamentals.free_cash_flow if not np.isnan(fundamentals.free_cash_flow) and fundamentals.free_cash_flow > 0 else (
        quote.get("market_cap", 0.0) * 0.04 if quote.get("market_cap") and not np.isnan(quote["market_cap"]) else 1000000.0
    )
    shares = quote.get("shares_outstanding") or fundamentals.shares_outstanding

    dcf_in = DCFInputs(
        fcffree=base_fcf,
        wacc=wacc_input,
        growth_rate=growth_input,
        terminal_growth_rate=term_g_input,
        shares_outstanding=shares,
        net_debt=0.0,
        projection_years=5,
    )
    valuation = calculate_dcf_valuation(dcf_in, current_price=quote.get("price"))
    scenarios = generate_scenario_analysis(
        base_fcf=base_fcf,
        base_wacc=wacc_input,
        base_growth=growth_input,
        terminal_growth=term_g_input,
        shares_outstanding=shares or 1.0,
        current_price=quote.get("price"),
    )

    # 4. Compute Peer Multiples Comparison
    target_multiples = {
        "pe_ratio": quote.get("pe_ratio"),
        "pb_ratio": quote.get("pb_ratio"),
        "beta": quote.get("beta") or technicals.beta,
        "dividend_yield": quote.get("dividend_yield"),
    }
    peer_results = analyze_peer_comparison(ticker_input, peers_df, target_multiples)

    # 5. Generate AI Research Report Sections
    report_sections = generate_report_sections(
        quote=quote,
        fundamental=fundamentals,
        technical=technicals,
        valuation=valuation,
        peer_comparison=peer_results,
        news_items=news_items,
    )

    full_report = EquityResearchReport(
        ticker=ticker_input,
        as_of_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        sections=report_sections,
        quote=quote,
        fundamental=fundamentals,
        technical=technicals,
        valuation=valuation,
        peer_comparison=peer_results,
    )

    curr = quote.get("currency", "INR")
    curr_sym = "₹" if curr == "INR" else "$" if curr == "USD" else f"{curr} "
    price = quote.get("price")
    target_val = valuation.intrinsic_value

    # --- Header Strip ---
    st.markdown(f"### {quote.get('name', ticker_input)} ({ticker_input})")
    badge_items = [
        (ticker_input, "accent"),
        (quote.get("sector", "Unknown Sector"), "neutral"),
        (quote.get("industry", "Unknown Industry"), "neutral"),
        (f"Currency: {curr}", "neutral"),
        (f"Data as of: {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}", "neutral"),
    ]
    ui.badges(badge_items)

    # Warning Notices if any
    if warnings or fundamentals.missing_fields:
        with st.expander("⚠️ Data Integrity & Filing Disclosures", expanded=False):
            if warnings:
                st.caption("**Provider Fetch Notices:**")
                for w in warnings:
                    st.caption(f"- {w}")
            if fundamentals.missing_fields:
                st.caption(f"**Missing Statement Items:** {', '.join(fundamentals.missing_fields)} (reported as missing, never fabricated).")

    # --- KPI Row ---
    rec = valuation.recommendation or "Hold"
    rec_tone = "gain" if rec.upper() == "BUY" else "loss" if rec.upper() == "SELL" else "warn"

    upside_str = DASH
    if target_val and price and price > 0:
        up = (target_val - price) / price
        upside_str = f"{'+' if up >= 0 else ''}{up * 100:.1f}%"

    kpis = [
        {"label": "Spot Price", "value": f"{curr_sym}{price:,.2f}" if price and not np.isnan(price) else DASH},
        {"label": "DCF Fair Value", "value": f"{curr_sym}{target_val:,.2f}" if target_val and not np.isnan(target_val) else DASH, "tone": rec_tone},
        {"label": "Implied Upside", "value": upside_str, "tone": rec_tone},
        {"label": "Model Assessment", "value": rec.upper(), "tone": rec_tone},
        {"label": "Operating Margin", "value": fmt_ratio_pct(fundamentals.operating_margin)},
        {"label": "14-Day RSI", "value": fmt_num(technicals.rsi, 1)},
    ]
    ui.kpi_row(kpis)

    # Action Toolbar: Download PDF, Excel, and Save to History
    st.divider()
    act_col1, act_col2, act_col3 = st.columns([1, 1, 1])

    with act_col1:
        pdf_bytes = export_equity_report_pdf(full_report)
        st.download_button(
            label="📥 Download PDF Tearsheet",
            data=pdf_bytes,
            file_name=f"{ticker_input}_research_report.pdf" if len(pdf_bytes) > 2000 else f"{ticker_input}_report.html",
            mime="application/pdf" if len(pdf_bytes) > 2000 else "text/html",
            use_container_width=True,
            help="Download the formal, CSS-formatted Equity Research Tearsheet.",
        )

    with act_col2:
        excel_bytes = export_equity_research_excel(full_report)
        st.download_button(
            label="📊 Export Financial Model (Excel)",
            data=excel_bytes,
            file_name=f"{ticker_input}_financial_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Download complete multi-tab financial model: DCF cash flows, fundamentals, technicals, and peers.",
        )

    with act_col3:
        if st.button("💾 Save Report to History", use_container_width=True):
            rec_id = save_research_record(
                ticker=ticker_input,
                company_name=quote.get("name", ticker_input),
                price=price,
                intrinsic_value=target_val,
                recommendation=rec,
                summary=report_sections.executive_summary[:300],
            )
            st.success(f"Report saved to database (Record #{rec_id})!")

    # --- Main Detail Tabs ---
    tabs = st.tabs([
        "📖 Executive Report",
        "💵 DCF Valuation & Scenarios",
        "📊 Fundamentals & Ratios",
        "📈 Technicals & Risk",
        "👥 Peer Multiples",
        "🌐 NSE Market Segments",
        "📚 Saved History",
    ])

    # Tab 1: Executive AI Report
    with tabs[0]:
        st.markdown("## Equity Research Report Preview")
        st.markdown(report_sections.executive_summary)
        st.markdown(report_sections.company_overview)
        st.markdown(report_sections.financial_analysis)
        st.markdown(report_sections.valuation)
        st.markdown(report_sections.risks)
        st.markdown(report_sections.conclusion)
        st.markdown(report_sections.sources)
        st.markdown(report_sections.assumptions)

    # Tab 2: DCF Valuation & Scenarios
    with tabs[1]:
        st.markdown("### Discounted Cash Flow (DCF) Valuation Model")
        st.caption("5-Year Explicit Projection Horizon with Gordon Growth Terminal Value.")

        val_col1, val_col2 = st.columns([1.2, 1])
        with val_col1:
            cf_fig = _build_dcf_cashflow_chart(valuation, curr_sym)
            ui.chart(cf_fig)
        with val_col2:
            st.markdown("**Valuation Bridge Summary**")
            val_summary = [
                ("WACC Applied", fmt_ratio_pct(valuation.wacc_used)),
                ("Terminal Growth (g)", fmt_ratio_pct(dcf_in.terminal_growth_rate)),
                ("Enterprise Value", f"{curr_sym}{valuation.enterprise_value:,.0f}" if not np.isnan(valuation.enterprise_value) else DASH),
                ("Terminal Value (PV)", f"{curr_sym}{valuation.terminal_value:,.0f}" if not np.isnan(valuation.terminal_value) else DASH),
                ("Equity Value", f"{curr_sym}{valuation.equity_value:,.0f}" if not np.isnan(valuation.equity_value) else DASH),
                ("Shares Outstanding", fmt_num(shares, 0)),
                ("Intrinsic Target / Share", f"{curr_sym}{valuation.intrinsic_value:,.2f}" if valuation.intrinsic_value else DASH),
                ("Fair Value Band", f"{curr_sym}{valuation.fair_value_low:,.1f} – {curr_sym}{valuation.fair_value_high:,.1f}" if valuation.fair_value_low else DASH),
            ]
            ui.stat_grid(val_summary, columns=2)

        # Sensitivity Matrix Heatmap
        st.divider()
        st.markdown("### Valuation Sensitivity Matrix (WACC vs Growth)")
        if valuation.sensitivity_table is not None and not valuation.sensitivity_table.empty:
            sens_fig = _build_sensitivity_heatmap(valuation.sensitivity_table, curr_sym)
            ui.chart(sens_fig)

        # Bull / Base / Bear Scenarios
        st.divider()
        st.markdown("### Scenario Analysis (Bull, Base, Bear)")
        sc_cols = st.columns(3)
        for idx, (sc_name, sc_data) in enumerate(scenarios.items()):
            with sc_cols[idx]:
                iv = sc_data["intrinsic_value"]
                up = sc_data["upside_pct"]
                tone = "gain" if sc_name == "Bull" else "loss" if sc_name == "Bear" else "warn"
                ui.kpi_card(
                    label=f"{sc_name} Scenario",
                    value=f"{curr_sym}{iv:,.2f}" if iv else DASH,
                    delta=f"Upside: {'+' if up >= 0 else ''}{up * 100:.1f}%" if not np.isnan(up) else None,
                    delta_tone=tone,
                    tone=tone,
                    footnote=f"WACC: {sc_data['wacc']:.1%} | g: {sc_data['growth_rate']:.1%}",
                )

    # Tab 3: Fundamentals & Financial Statements
    with tabs[2]:
        st.markdown("### Fundamental Ratios & Profitability")
        f_stats = [
            ("Gross Margin", fmt_ratio_pct(fundamentals.gross_margin)),
            ("Operating Margin", fmt_ratio_pct(fundamentals.operating_margin)),
            ("Net Profit Margin", fmt_ratio_pct(fundamentals.net_margin)),
            ("Earnings Per Share (EPS)", f"{curr_sym}{fundamentals.eps:,.2f}" if not np.isnan(fundamentals.eps) else DASH),
            ("Return on Equity (ROE)", fmt_ratio_pct(fundamentals.roe)),
            ("ROCE", fmt_ratio_pct(fundamentals.roce)),
            ("Debt-to-Equity", fmt_num(fundamentals.debt_to_equity, 2)),
            ("Debt-to-Asset", fmt_num(fundamentals.debt_to_asset, 2)),
            ("Latest Period Revenue", f"{curr_sym}{fundamentals.revenue_latest:,.0f}" if not np.isnan(fundamentals.revenue_latest) else DASH),
            ("Latest Period Net Profit", f"{curr_sym}{fundamentals.profit_latest:,.0f}" if not np.isnan(fundamentals.profit_latest) else DASH),
            ("Free Cash Flow", f"{curr_sym}{fundamentals.free_cash_flow:,.0f}" if not np.isnan(fundamentals.free_cash_flow) else DASH),
            ("Currency", fundamentals.currency),
        ]
        ui.stat_grid(f_stats, columns=4)

        if news_items:
            st.divider()
            st.markdown("### Recent Company Headlines & Catalysts")
            for item in news_items[:5]:
                t = item.get("title", "") if isinstance(item, dict) else getattr(item, "title", "")
                s = item.get("source", "News") if isinstance(item, dict) else getattr(item, "source", "News")
                p = item.get("published", "") if isinstance(item, dict) else getattr(item, "published", "")
                st.markdown(f"- **{t}** — *{s}* ({p})")

    # Tab 4: Technicals & Risk
    with tabs[3]:
        st.markdown("### Technical Analysis & Risk Indicators")
        t_fig = _build_price_technical_chart(ohlcv, technicals, ticker_input)
        ui.chart(t_fig)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            rsi_fig = _build_rsi_chart(ohlcv, ticker_input)
            ui.chart(rsi_fig)
        with t_col2:
            macd_fig = _build_macd_chart(ohlcv, ticker_input)
            ui.chart(macd_fig)

        st.markdown("### Statistical Risk Profile")
        t_stats = [
            ("Annualized Volatility (ddof=1)", fmt_ratio_pct(technicals.volatility)),
            ("Beta vs Benchmark", fmt_num(technicals.beta, 2)),
            ("Sharpe Ratio (Annualized)", fmt_num(technicals.sharpe_ratio, 2)),
            ("Maximum Drawdown", fmt_ratio_pct(technicals.max_drawdown)),
            ("20-Day SMA", f"{curr_sym}{technicals.sma.get('20', np.nan):,.2f}" if '20' in technicals.sma else DASH),
            ("50-Day SMA", f"{curr_sym}{technicals.sma.get('50', np.nan):,.2f}" if '50' in technicals.sma else DASH),
            ("200-Day SMA", f"{curr_sym}{technicals.sma.get('200', np.nan):,.2f}" if '200' in technicals.sma else DASH),
            ("Period Trading Range", f"{curr_sym}{technicals.period_low:,.1f} – {curr_sym}{technicals.period_high:,.1f}"),
        ]
        ui.stat_grid(t_stats, columns=4)

    # Tab 5: Peer Comparison
    with tabs[4]:
        st.markdown("### Relative Valuation vs. Sector Peers")
        if peers_df is not None and not peers_df.empty:
            peer_fig = _build_peer_multiples_chart(peers_df, ticker_input, quote.get("pe_ratio", np.nan))
            ui.chart(peer_fig)

            st.markdown("#### Peer Group Multiples Table")
            display_peers = peers_df.copy()
            if "market_cap" in display_peers.columns:
                display_peers["market_cap"] = display_peers["market_cap"].apply(lambda v: f"{curr_sym}{v:,.0f}" if pd.notnull(v) and v > 0 else DASH)
            ui.dataframe(display_peers, height=220)

            st.markdown("#### Target Company Percentile Ranks in Sector")
            p_ranks = [
                ("P/E Percentile", fmt_ratio_pct(peer_results.percentile_rank.get("pe_ratio"))),
                ("P/B Percentile", fmt_ratio_pct(peer_results.percentile_rank.get("pb_ratio"))),
                ("Beta Percentile", fmt_ratio_pct(peer_results.percentile_rank.get("beta"))),
                ("Dividend Yield Percentile", fmt_ratio_pct(peer_results.percentile_rank.get("dividend_yield"))),
            ]
            ui.stat_grid(p_ranks, columns=4)
        else:
            ui.empty_state("No Peer Data Available", "Peer comparison requires sector catalog data. None found for this listing.", icon="👥")

    # Tab 6: NSE Market Segments & Ratios
    with tabs[5]:
        st.markdown("### NSE Market Segments & Turnover Ratios")
        st.caption("Aggregated market statistics, turnover share, and segment ratios across all NSE listed equities.")

        df_nse, meta_nse = get_nse_market_snapshot()
        seg_df = compute_market_segment_ratios(df_nse)

        kpi_nse = [
            {"label": "Total NSE Listed Equities", "value": f"{meta_nse.get('total_nse_stocks', 0):,}"},
            {"label": "Total Traded Turnover", "value": f"₹{meta_nse.get('total_turnover', 0.0):,.0f}"},
            {"label": "Mapped Segments", "value": f"{len(seg_df):,}"},
        ]
        ui.stat_grid(kpi_nse, columns=3)

        if not seg_df.empty:
            st.divider()
            st.markdown("#### Market Segment Turnover Share (%)")
            fig_seg = go.Figure(
                go.Bar(
                    x=seg_df["sector"],
                    y=seg_df["turnover_share_pct"],
                    marker_color="#0EA5E9",
                    text=[f"{v:.1f}%" for v in seg_df["turnover_share_pct"]],
                    textposition="outside",
                )
            )
            fig_seg.update_layout(
                xaxis_title="Market Segment / Sector",
                yaxis_title="Turnover Share (%)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            ui.chart(ch_theme.apply(fig_seg))

            st.markdown("#### Segment Breakdown Table")
            disp_seg = seg_df.copy()
            disp_seg["total_turnover"] = disp_seg["total_turnover"].apply(lambda v: f"₹{v:,.0f}")
            disp_seg["mean_turnover"] = disp_seg["mean_turnover"].apply(lambda v: f"₹{v:,.0f}")
            disp_seg["median_turnover"] = disp_seg["median_turnover"].apply(lambda v: f"₹{v:,.0f}")
            disp_seg["max_turnover"] = disp_seg["max_turnover"].apply(lambda v: f"₹{v:,.0f}")
            disp_seg["turnover_share_pct"] = disp_seg["turnover_share_pct"].apply(lambda v: f"{v:.2f}%")
            ui.dataframe(disp_seg, height=300)
        else:
            ui.empty_state("No Segment Data", "Could not load NSE market snapshot.", icon="📊")

    # Tab 7: Saved Research History
    with tabs[6]:
        st.markdown("### Saved Equity Research Reports")
        history = load_research_history()
        if history:
            h_df = pd.DataFrame(history)
            ui.dataframe(h_df[["id", "ticker", "company_name", "price", "intrinsic_value", "recommendation", "created_at"]], height=240)

            # Option to delete an entry
            del_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1)
            if st.button("Delete Entry"):
                if delete_research_record(int(del_id)):
                    st.success(f"Record #{del_id} deleted.")
                    st.rerun()
                else:
                    st.warning(f"Record #{del_id} not found.")
        else:
            ui.empty_state("No Saved Reports", "Click 'Save Report to History' on any generated report to persist it here.", icon="📚")

    # Compliance Disclaimer Footer
    st.divider()
    st.caption(f"⚖️ **Regulatory Compliance Notice:** {DISCLAIMER} All ratings (Buy/Hold/Sell) are automated quantitative model assessments, not guaranteed forecasts or personalized investment advice. Source: Yahoo Finance API.")


# Support both render() and equity_research_page() for app.py
def equity_research_page() -> None:
    """Streamlit entry point."""
    render()
