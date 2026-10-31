"""Export module for Equity Research Reports: PDF and formatted Excel workbooks.

No Streamlit imports; zero direct network calls.
Uses headless Edge/Chrome via src.reporting.pdf for PDF rendering and XlsxWriter
for professional multi-tab financial modeling workbooks.
"""

from __future__ import annotations

import io
import html
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
import xlsxwriter

from src.reporting.pdf import pdf_available, html_to_pdf
from src.equity_research.models import EquityResearchReport

logger = logging.getLogger(__name__)


def _escape(text: Any) -> str:
    """Safe HTML escaping for markdown-like string chunks."""
    if text is None:
        return ""
    # Convert markdown headers and bolding to simple clean html
    s = html.escape(str(text))
    s = s.replace("### ", "<h3>").replace("## ", "<h2>").replace("# ", "<h1>")
    s = s.replace("**", "<b>").replace("*", "<i>")
    s = s.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return s


def build_html_report(report: EquityResearchReport) -> str:
    """Render a standalone, CSS-styled HTML equity research tearsheet."""
    quote = report.quote or {}
    ticker = report.ticker
    name = quote.get("name", ticker)
    price = quote.get("price")
    curr = quote.get("currency", "INR")
    curr_sym = "₹" if curr == "INR" else "$" if curr == "USD" else f"{curr} "

    rec = report.valuation.recommendation if report.valuation else "Hold"
    rec_color = (
        "#10B981" if rec.upper() == "BUY"
        else "#EF4444" if rec.upper() == "SELL"
        else "#F59E0B"
    )

    intrinsic = report.valuation.intrinsic_value if report.valuation else None
    upside_str = ""
    if intrinsic and price and price > 0:
        up = (intrinsic - price) / price
        upside_str = f"({'+' if up >= 0 else ''}{up * 100:.1f}%)"

    sections = report.sections

    # Build dedicated Multi-Model Summary HTML table if available
    mm_html = ""
    if report.multi_valuation and report.multi_valuation.ok:
        mv = report.multi_valuation
        p_val = price or 0.0
        mm_rows_html = ""
        models_data = [
            ("FCFF", "5-Year FCF + TV", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.fcff_intrinsic, f"{curr_sym}{mv.fcff_fair_low:,.1f} – {curr_sym}{mv.fcff_fair_high:,.1f}" if not np.isnan(mv.fcff_fair_low) else "—", ((mv.fcff_intrinsic - p_val) / p_val * 100) if p_val > 0 else np.nan),
            ("FCFE", "Equity Cash Flow", "Cost of Equity", f"{mv.terminal_growth_applied:.1%}", mv.fcfe_intrinsic, f"{curr_sym}{mv.fcfe_fair_low:,.1f} – {curr_sym}{mv.fcfe_fair_high:,.1f}" if not np.isnan(mv.fcfe_fair_low) else "—", ((mv.fcfe_intrinsic - p_val) / p_val * 100) if p_val > 0 else np.nan),
            ("DDM", "Dividends", "Cost of Equity", f"{mv.terminal_growth_applied:.1%}", mv.ddm_value, "—", ((mv.ddm_value - p_val) / p_val * 100) if p_val > 0 and not np.isnan(mv.ddm_value) else np.nan),
            ("Gordon Growth", "Next-period FCF", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.gordon_value, "—", ((mv.gordon_value - p_val) / p_val * 100) if p_val > 0 and not np.isnan(mv.gordon_value) else np.nan),
            ("Blended", "Multi-model Avg", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.blended_intrinsic, f"{curr_sym}{mv.blended_fair_low:,.1f} – {curr_sym}{mv.blended_fair_high:,.1f}" if not np.isnan(mv.blended_fair_low) else "—", (mv.premium_to_price * 100) if not np.isnan(mv.premium_to_price) else np.nan),
        ]
        for name, cf_base, disc, tg, iv, band, up in models_data:
            if not np.isnan(iv) and iv > 0:
                up_str = f"{up:+.1f}%" if not np.isnan(up) else "—"
                mm_rows_html += f"""
                <tr>
                  <td><b>{name}</b></td>
                  <td>{cf_base}</td>
                  <td>{disc}</td>
                  <td>{tg}</td>
                  <td><b>{curr_sym}{iv:,.2f}</b></td>
                  <td>{band}</td>
                  <td><b>{up_str}</b></td>
                </tr>
                """
        mm_html = f"""
<div class="section">
  <h2>Multi-Model Valuation Summary Matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Cash Flow Base</th>
        <th>Discount Rate</th>
        <th>Growth Rate</th>
        <th>Intrinsic Value</th>
        <th>Fair Value Band</th>
        <th>Implied Upside</th>
      </tr>
    </thead>
    <tbody>
      {mm_rows_html}
    </tbody>
  </table>
  <div style="font-size: 9pt; color: #64748B; margin-top: 6px;">
    <b>Institutional Guardrails:</b> TV Contribution: {mv.tv_contribution_pct:.1%} (Capped ≤80%) &bull; Cross-Model Gap: {mv.cross_model_gap_pct:.1%} &bull; Confidence Score: {mv.confidence_score:.0%} &bull; Spread: {(mv.wacc_applied - mv.terminal_growth_applied):.1%}
  </div>
</div>
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Equity Research Report - {html.escape(ticker)}</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 15mm 15mm 15mm 15mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1E293B;
    background: #FFFFFF;
    line-height: 1.5;
    font-size: 11pt;
    margin: 0;
    padding: 0;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #0EA5E9;
    padding-bottom: 12px;
    margin-bottom: 20px;
  }}
  .header h1 {{
    margin: 0;
    font-size: 20pt;
    color: #0F172A;
  }}
  .header .meta {{
    font-size: 10pt;
    color: #64748B;
    margin-top: 4px;
  }}
  .badge {{
    background: {rec_color};
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 12pt;
    text-align: right;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 10px;
    text-align: center;
  }}
  .kpi-title {{
    font-size: 8.5pt;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
  }}
  .kpi-value {{
    font-size: 13pt;
    font-weight: 700;
    color: #0F172A;
    margin-top: 4px;
  }}
  .section {{
    margin-bottom: 20px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 14px;
  }}
  .section h2 {{
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 13pt;
    color: #0F172A;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 6px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 10pt;
  }}
  th, td {{
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid #E2E8F0;
  }}
  th {{
    background: #F1F5F9;
    color: #475569;
    font-weight: 600;
  }}
  .disclaimer {{
    font-size: 8pt;
    color: #94A3B8;
    line-height: 1.4;
    margin-top: 25px;
    border-top: 1px solid #E2E8F0;
    padding-top: 10px;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{html.escape(name)} ({html.escape(ticker)})</h1>
    <div class="meta">
      Equity Research Tearsheet &bull; Date: {html.escape(report.as_of_date)} &bull; Currency: {html.escape(curr)}
    </div>
  </div>
  <div>
    <div class="badge">{html.escape(rec.upper())}</div>
    <div style="font-size: 9pt; color: #64748B; margin-top: 4px; text-align: right;">Model Assessment</div>
  </div>
</div>

<div class="grid">
  <div class="kpi-card">
    <div class="kpi-title">Current Price</div>
    <div class="kpi-value">{curr_sym}{price:,.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">DCF Intrinsic Target</div>
    <div class="kpi-value">{curr_sym}{intrinsic:,.2f} {upside_str}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Operating Margin</div>
    <div class="kpi-value">{report.fundamental.operating_margin * 100:.1f}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">14-Day RSI</div>
    <div class="kpi-value">{report.technical.rsi:.1f}</div>
  </div>
</div>

<div class="section">
  <h2>Executive Summary</h2>
  <div>{_escape(sections.executive_summary)}</div>
</div>

<div class="section">
  <h2>Company Overview</h2>
  <div>{_escape(sections.company_overview)}</div>
</div>

<div class="section">
  <h2>Financial Analysis & Metrics</h2>
  <div>{_escape(sections.financial_analysis)}</div>
</div>

<div class="section">
  <h2>DCF Valuation & Sensitivity</h2>
  <div>{_escape(sections.valuation)}</div>
</div>

{mm_html}

<div class="section">
  <h2>Risks, Catalysts & Scenarios</h2>
  <div>{_escape(sections.risks)}</div>
</div>

<div class="section">
  <h2>Conclusion & Stance</h2>
  <div>{_escape(sections.conclusion)}</div>
</div>

<div class="disclaimer">
  <strong>Regulatory & Compliance Disclaimer:</strong><br>
  This Equity Research Report is an automated quantitative artifact generated for educational, analytical,
  and portfolio research purposes. All recommendations (Buy/Hold/Sell) are model-generated assessments
  derived from mathematical algorithms and historical financial filings, not guaranteed forecasts or personalized
  investment advice. Historical performance is not indicative of future results. Source: Yahoo Finance via yfinance.
</div>

</body>
</html>
"""
    return html_content


def export_equity_report_pdf(report: EquityResearchReport) -> bytes:
    """Export equity research report to PDF bytes.
    Leverages src.reporting.pdf (headless Edge/Chrome) or returns clean HTML fallback.
    """
    html_str = build_html_report(report)
    if pdf_available():
        try:
            return html_to_pdf(html_str)
        except Exception as exc:
            logger.warning("Headless PDF rendering failed: %s; returning HTML document bytes.", exc)

    return html_str.encode("utf-8")


def _safe_write(ws, r, c, val, fmt, empty_fmt):
    """Write number or blank cell if NaN/None."""
    if val is None or (isinstance(val, (float, int)) and (np.isnan(val) or np.isinf(val))):
        ws.write(r, c, "", empty_fmt)
    else:
        ws.write(r, c, val, fmt)


def export_equity_research_excel(report: EquityResearchReport) -> bytes:
    """Generate a multi-tab institutional Excel financial model workbook."""
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})

    # Formats
    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 14,
        "font_color": "#FFFFFF",
        "bg_color": "#0F172A",
        "align": "left",
        "valign": "vcenter",
    })
    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#1E293B",
        "font_color": "#FFFFFF",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })
    label_fmt = workbook.add_format({"bold": True, "border": 1, "bg_color": "#F8FAFC"})
    num_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
    curr_fmt = workbook.add_format({"num_format": "₹#,##0.00", "border": 1})
    pct_fmt = workbook.add_format({"num_format": "0.0%", "border": 1})
    text_fmt = workbook.add_format({"border": 1})

    # Tab 1: Summary & Valuation
    ws_val = workbook.add_worksheet("Summary & DCF")
    ws_val.set_column(0, 0, 28)
    ws_val.set_column(1, 4, 18)

    ws_val.merge_range("A1:E1", f"Equity Research Model: {report.ticker}", title_fmt)
    ws_val.write("A3", "Company Name", label_fmt)
    ws_val.write("B3", report.quote.get("name", report.ticker), text_fmt)
    ws_val.write("A4", "Spot Price", label_fmt)
    ws_val.write("B4", report.quote.get("price", np.nan), curr_fmt)
    ws_val.write("A5", "Model Rating", label_fmt)
    ws_val.write("B5", report.valuation.recommendation if report.valuation else "N/A", text_fmt)
    ws_val.write("A6", "DCF Intrinsic Target", label_fmt)
    ws_val.write("B6", report.valuation.intrinsic_value if report.valuation else np.nan, curr_fmt)
    ws_val.write("A7", "Fair Value Range", label_fmt)
    ws_val.write("B7", f"{report.valuation.fair_value_low:,.1f} - {report.valuation.fair_value_high:,.1f}" if report.valuation and report.valuation.fair_value_low else "N/A", text_fmt)

    # 5-Year DCF Cash Flows Table
    ws_val.write("A9", "5-Year FCF Projections", header_fmt)
    ws_val.write("B9", "Discounted PV", header_fmt)
    row = 10
    if report.valuation and report.valuation.dcf_interpolated:
        for year, pv_val in report.valuation.dcf_interpolated.items():
            ws_val.write(row, 0, year.replace("_", " ").title(), label_fmt)
            ws_val.write(row, 1, pv_val, curr_fmt)
            row += 1

    # Sensitivity Table
    if report.valuation and report.valuation.sensitivity_table is not None and not report.valuation.sensitivity_table.empty:
        row += 2
        ws_val.write(row, 0, "DCF Sensitivity Matrix (WACC vs Growth)", header_fmt)
        row += 1
        sens = report.valuation.sensitivity_table
        # Header row
        ws_val.write(row, 0, "WACC \\ Growth", header_fmt)
        for col_idx, col_name in enumerate(sens.columns, start=1):
            ws_val.write(row, col_idx, col_name, header_fmt)
        row += 1
        for idx_label, s_row in sens.iterrows():
            ws_val.write(row, 0, str(idx_label), label_fmt)
            for c_idx, val in enumerate(s_row, start=1):
                ws_val.write(row, c_idx, val, curr_fmt)
            row += 1

    # Multi-Model Valuation Worksheet
    if report.multi_valuation and report.multi_valuation.ok:
        ws_mm = workbook.add_worksheet("Multi-Model Valuation")
        ws_mm.set_column(0, 0, 20)
        ws_mm.set_column(1, 5, 18)
        ws_mm.merge_range("A1:F1", "Multi-Model Valuation Summary Matrix", title_fmt)

        mv = report.multi_valuation
        p_val = report.quote.get("price", np.nan)

        mm_headers = ["Model", "Cash Flow Base", "Discount Rate", "Terminal Growth", "Intrinsic Value", "Implied Upside"]
        for c_idx, h in enumerate(mm_headers):
            ws_mm.write(2, c_idx, h, header_fmt)

        mm_rows = []
        if not np.isnan(mv.fcff_intrinsic):
            fcff_up = (mv.fcff_intrinsic - p_val) / p_val if p_val and p_val > 0 else np.nan
            mm_rows.append(("FCFF", "5-Year FCF + TV", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.fcff_intrinsic, fcff_up))
        if not np.isnan(mv.fcfe_intrinsic):
            fcfe_up = (mv.fcfe_intrinsic - p_val) / p_val if p_val and p_val > 0 else np.nan
            mm_rows.append(("FCFE", "Equity Cash Flow", "Cost of Equity", f"{mv.terminal_growth_applied:.1%}", mv.fcfe_intrinsic, fcfe_up))
        if not np.isnan(mv.ddm_value):
            ddm_up = (mv.ddm_value - p_val) / p_val if p_val and p_val > 0 else np.nan
            mm_rows.append(("DDM", "Dividends", "Cost of Equity", f"{mv.terminal_growth_applied:.1%}", mv.ddm_value, ddm_up))
        if not np.isnan(mv.gordon_value):
            gordon_up = (mv.gordon_value - p_val) / p_val if p_val and p_val > 0 else np.nan
            mm_rows.append(("Gordon Growth", "Next-period FCF", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.gordon_value, gordon_up))
        if not np.isnan(mv.blended_intrinsic):
            mm_rows.append(("Blended", "Multi-model Avg", f"WACC {mv.wacc_applied:.1%}", f"{mv.terminal_growth_applied:.1%}", mv.blended_intrinsic, mv.premium_to_price))

        r_mm = 3
        for m_name, base_cf, disc_rate, tg_rate, iv, up in mm_rows:
            ws_mm.write(r_mm, 0, m_name, label_fmt)
            ws_mm.write(r_mm, 1, base_cf, text_fmt)
            ws_mm.write(r_mm, 2, disc_rate, text_fmt)
            ws_mm.write(r_mm, 3, tg_rate, text_fmt)
            _safe_write(ws_mm, r_mm, 4, iv, curr_fmt, text_fmt)
            _safe_write(ws_mm, r_mm, 5, up, pct_fmt, text_fmt)
            r_mm += 1

        # Institutional Validation Metrics in Excel
        r_mm += 2
        ws_mm.merge_range(f"A{r_mm+1}:D{r_mm+1}", "Institutional Validation & Guardrails", header_fmt)
        r_mm += 1
        val_metrics = [
            ("Terminal Value Contribution %", mv.tv_contribution_pct, pct_fmt),
            ("Cross-Model Gap %", mv.cross_model_gap_pct, pct_fmt),
            ("Valuation Confidence Score", mv.confidence_score, pct_fmt),
            ("WACC Applied", mv.wacc_applied, pct_fmt),
            ("Terminal Growth Applied", mv.terminal_growth_applied, pct_fmt),
            ("Assumptions Rejected & Recalculated", "Yes" if mv.assumptions_rejected else "No", text_fmt),
        ]
        for v_lbl, v_val, v_fmt in val_metrics:
            ws_mm.write(r_mm, 0, v_lbl, label_fmt)
            if isinstance(v_val, str):
                ws_mm.write(r_mm, 1, v_val, v_fmt)
            else:
                _safe_write(ws_mm, r_mm, 1, v_val, v_fmt, text_fmt)
            r_mm += 1

    # Tab 2: Fundamentals
    ws_fund = workbook.add_worksheet("Fundamentals")
    ws_fund.set_column(0, 0, 30)
    ws_fund.set_column(1, 1, 20)
    ws_fund.merge_range("A1:B1", "Key Financial Ratios & Statement Summary", title_fmt)

    f = report.fundamental
    fund_rows = [
        ("Gross Margin", getattr(f, "gross_margin", np.nan), pct_fmt),
        ("Operating Margin", getattr(f, "operating_margin", np.nan), pct_fmt),
        ("Net Profit Margin", getattr(f, "net_margin", np.nan), pct_fmt),
        ("Earnings Per Share (EPS)", getattr(f, "eps", np.nan), curr_fmt),
        ("Return on Equity (ROE)", getattr(f, "roe", np.nan), pct_fmt),
        ("Return on Capital Employed (ROCE)", getattr(f, "roce", np.nan), pct_fmt),
        ("Debt-to-Equity Ratio", getattr(f, "debt_to_equity", np.nan), num_fmt),
        ("Debt-to-Asset Ratio", getattr(f, "debt_to_asset", np.nan), num_fmt),
        ("Latest Period Revenue", getattr(f, "revenue_latest", np.nan), curr_fmt),
        ("Latest Period Net Profit", getattr(f, "profit_latest", np.nan), curr_fmt),
        ("Free Cash Flow", getattr(f, "free_cash_flow", np.nan), curr_fmt),
    ]

    r_idx = 2
    for label, val, fmt in fund_rows:
        ws_fund.write(r_idx, 0, label, label_fmt)
        ws_fund.write(r_idx, 1, val if not np.isnan(val) else "", fmt)
        r_idx += 1

    # Tab 3: Technicals
    ws_tech = workbook.add_worksheet("Technicals & Risk")
    ws_tech.set_column(0, 0, 25)
    ws_tech.set_column(1, 1, 20)
    ws_tech.merge_range("A1:B1", "Technical Indicators & Market Risk", title_fmt)

    t = report.technical
    tech_rows = [
        ("Current Close Price", getattr(t, "current_price", np.nan), curr_fmt),
        ("20-Day SMA", t.sma.get("20", np.nan) if t else np.nan, curr_fmt),
        ("50-Day SMA", t.sma.get("50", np.nan) if t else np.nan, curr_fmt),
        ("200-Day SMA", t.sma.get("200", np.nan) if t else np.nan, curr_fmt),
        ("14-Day Wilder RSI", getattr(t, "rsi", np.nan), num_fmt),
        ("MACD Line", t.macd.get("macd", np.nan) if t else np.nan, num_fmt),
        ("MACD Signal Line", t.macd.get("signal", np.nan) if t else np.nan, num_fmt),
        ("MACD Histogram", t.macd.get("histogram", np.nan) if t else np.nan, num_fmt),
        ("Annualized Volatility", getattr(t, "volatility", np.nan), pct_fmt),
        ("Beta vs Benchmark", getattr(t, "beta", np.nan), num_fmt),
        ("Sharpe Ratio", getattr(t, "sharpe_ratio", np.nan), num_fmt),
        ("Maximum Drawdown", getattr(t, "max_drawdown", np.nan), pct_fmt),
    ]

    t_idx = 2
    for label, val, fmt in tech_rows:
        ws_tech.write(t_idx, 0, label, label_fmt)
        ws_tech.write(t_idx, 1, val if not np.isnan(val) else "", fmt)
        t_idx += 1

    # Tab 4: Peers
    if report.peer_comparison and report.peer_comparison.peers is not None and not report.peer_comparison.peers.empty:
        ws_peers = workbook.add_worksheet("Peer Comparison")
        peers_df = report.peer_comparison.peers
        ws_peers.merge_range("A1:F1", "Sector Peer Multiples & Relative Valuation", title_fmt)

        headers = ["Ticker", "Company Name", "Price", "P/E Ratio", "P/B Ratio", "Market Cap"]
        for c_idx, h in enumerate(headers):
            ws_peers.write(2, c_idx, h, header_fmt)

        p_idx = 3
        for _, p_row in peers_df.iterrows():
            ws_peers.write(p_idx, 0, p_row.get("ticker", ""), text_fmt)
            ws_peers.write(p_idx, 1, p_row.get("name", ""), text_fmt)
            _safe_write(ws_peers, p_idx, 2, p_row.get("price"), curr_fmt, text_fmt)
            _safe_write(ws_peers, p_idx, 3, p_row.get("pe_ratio"), num_fmt, text_fmt)
            _safe_write(ws_peers, p_idx, 4, p_row.get("pb_ratio"), num_fmt, text_fmt)
            _safe_write(ws_peers, p_idx, 5, p_row.get("market_cap"), curr_fmt, text_fmt)
            p_idx += 1

    workbook.close()
    output.seek(0)
    return output.read()
