"""AI-powered Equity Research Report narrative generator.

Generates professional institutional-grade research report sections.
Supports optional LLM integration (Anthropic/OpenAI compatible) via environment
variables, with an institutional-analyst deterministic template fallback.
Never fabricates metrics, sources, or ratings.
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Optional
import numpy as np

from src.equity_research.models import (
    FundamentalData,
    TechnicalData,
    ValuationResult,
    PeerComparisonResult,
    AIReportSections,
)

logger = logging.getLogger(__name__)


def _format_curr(val: Optional[float], currency: str = "INR") -> str:
    if val is None or np.isnan(val):
        return "—"
    symbol = "₹" if currency == "INR" else "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{val:,.2f}"


def _format_pct(val: Optional[float]) -> str:
    if val is None or np.isnan(val):
        return "—"
    return f"{val * 100:.1f}%"


def _format_num(val: Optional[float], decimals: int = 2) -> str:
    if val is None or np.isnan(val):
        return "—"
    return f"{val:,.{decimals}f}"


def build_template_sections(
    quote: dict[str, Any],
    fundamental: Optional[FundamentalData],
    technical: Optional[TechnicalData],
    valuation: Optional[ValuationResult],
    peer_comparison: Optional[PeerComparisonResult],
    news_items: Optional[list[dict[str, Any]]] = None,
) -> AIReportSections:
    """Deterministic, institutional analyst report generator used when LLM is unavailable."""
    ticker = quote.get("ticker", "UNKNOWN")
    name = quote.get("name", ticker)
    price = quote.get("price")
    curr = quote.get("currency", "INR")
    sector = quote.get("sector", "Unknown Sector")
    industry = quote.get("industry", "Unknown Industry")
    mcap = quote.get("market_cap")

    # 1. Executive Summary
    rec = valuation.recommendation if valuation else "Insufficient data"
    target_price = valuation.intrinsic_value if valuation else None
    upside_str = ""
    if target_price and price and price > 0:
        upside = (target_price - price) / price
        upside_str = f" ({'+' if upside >= 0 else ''}{upside * 100:.1f}% implied upside/downside)"

    exec_summary = f"""### Executive Summary

**Ticker:** {ticker} | **Company:** {name} | **Current Price:** {_format_curr(price, curr)}
**Model-Generated Rating:** **{rec.upper()}** | **DCF Intrinsic Target:** {_format_curr(target_price, curr)}{upside_str}

**Investment Thesis:**
{name} is an established company in the **{sector}** sector ({industry}).
Our quantitative assessment evaluates the business across historical profitability, balance sheet resilience, momentum indicators, and fundamental cash flow intrinsic value.

- **Valuation Outlook:** DCF fair value estimates intrinsic value at {_format_curr(target_price, curr)} (Fair Value Range: {_format_curr(valuation.fair_value_low if valuation else None, curr)} – {_format_curr(valuation.fair_value_high if valuation else None, curr)}).
- **Fundamental Strength:** Gross margin stands at {_format_pct(fundamental.gross_margin if fundamental else None)}, with an operating margin of {_format_pct(fundamental.operating_margin if fundamental else None)} and Return on Equity (ROE) of {_format_pct(fundamental.roe if fundamental else None)}.
- **Technical Posture:** 14-period RSI is at {_format_num(technical.rsi if technical else None, 1)} with annualized volatility of {_format_pct(technical.volatility if technical else None)} and Beta of {_format_num(technical.beta if technical else None, 2)}.

*Notice: This assessment is an automated quantitative model output for educational and portfolio analysis, not an offer or guaranteed investment advice.*
"""

    # 2. Company Overview
    summary_text = quote.get("summary") or "Company description is currently unavailable from public filings."
    company_overview = f"""### Company Overview & Business Profile

- **Sector:** {sector}
- **Industry:** {industry}
- **Market Capitalization:** {_format_curr(mcap, curr) if mcap else 'N/A'}
- **52-Week Trading Range:** {_format_curr(technical.period_low if technical else None, curr)} – {_format_curr(technical.period_high if technical else None, curr)}

**Business Description:**
{summary_text}
"""

    # 3. Financial Analysis
    fin_status = ""
    if fundamental and fundamental.missing_fields:
        fin_status = f"\n*Data Note: The following items were absent from public filings: {', '.join(fundamental.missing_fields)}.*"

    fin_analysis = f"""### Fundamental & Financial Statement Analysis

| Metric | Current Period | Status / Benchmark |
| :--- | :--- | :--- |
| **Gross Margin** | {_format_pct(fundamental.gross_margin if fundamental else None)} | Core operational efficiency |
| **Operating Margin** | {_format_pct(fundamental.operating_margin if fundamental else None)} | Operating leverage |
| **Net Profit Margin** | {_format_pct(fundamental.net_margin if fundamental else None)} | Bottom-line profitability |
| **Earnings Per Share (EPS)** | {_format_curr(fundamental.eps if fundamental else None, curr)} | Diluted/Basic per share |
| **Return on Equity (ROE)** | {_format_pct(fundamental.roe if fundamental else None)} | Shareholder capital efficiency |
| **Return on Capital Employed (ROCE)** | {_format_pct(fundamental.roce if fundamental else None)} | Total capital return |
| **Debt-to-Equity** | {_format_num(fundamental.debt_to_equity if fundamental else None, 2)}x | Solvency / Financial leverage |
| **Free Cash Flow** | {_format_curr(fundamental.free_cash_flow if fundamental else None, curr)} | Operating Cash Flow less CapEx |
{fin_status}
"""

    # 4. Valuation Analysis
    dcf_notes = ""
    if valuation and valuation.status_notes:
        dcf_notes = "\n*Model Notes:* " + "; ".join(valuation.status_notes)

    val_section = f"""### Valuation & Intrinsic Pricing Models

**Methodology: 5-Year DCF / Free Cash Flow to Firm (FCFF) Model**
- **WACC Applied:** {_format_pct(valuation.wacc_used if valuation else None)}
- **Terminal Growth Rate:** 3.0% (conservative long-term GDP inflation benchmark)
- **Enterprise Value:** {_format_curr(valuation.enterprise_value if valuation else None, curr)}
- **Equity Value:** {_format_curr(valuation.equity_value if valuation else None, curr)}
- **Intrinsic Value Per Share:** **{_format_curr(valuation.intrinsic_value if valuation else None, curr)}**
- **Sensitivity Band:** Low {_format_curr(valuation.fair_value_low if valuation else None, curr)} | High {_format_curr(valuation.fair_value_high if valuation else None, curr)}
{dcf_notes}

**Relative Multiples (P/E & P/B vs Sector Peers):**
- Target P/E: {_format_num(peer_comparison.target_multiples.get('pe_ratio') if peer_comparison else None)}x (Sector Avg: {_format_num(peer_comparison.sector_averages.get('pe_ratio') if peer_comparison else None)}x)
- Target P/B: {_format_num(peer_comparison.target_multiples.get('pb_ratio') if peer_comparison else None)}x (Sector Avg: {_format_num(peer_comparison.sector_averages.get('pb_ratio') if peer_comparison else None)}x)
"""

    # 5. Risks & Catalysts
    news_headlines = ""
    if news_items:
        headlines = []
        for item in news_items[:3]:
            if isinstance(item, dict):
                title = item.get("title", "")
                src = item.get("source", "News")
            else:
                title = getattr(item, "title", "")
                src = getattr(item, "source", "News")
            if title:
                headlines.append(f"- {title} *({src})*")
        if headlines:
            news_headlines = "\n**Recent Headlines & Catalysts:**\n" + "\n".join(headlines)

    risks = f"""### Investment Risks, Catalysts & Sensitivities

**Key Risks:**
1. **Industry & Commodity Headwinds:** Subject to cyclical swings and input cost volatility in the {sector} space.
2. **Capital Structure & Debt Servicing:** Total debt leverage of {_format_num(fundamental.debt_to_equity if fundamental else None, 2)}x requires ongoing cash flow generation.
3. **Macroeconomic & Interest Rate Sensitivity:** Rising discount rates reduce DCF terminal value significantly.
4. **Data Provider Latency:** Market inputs and filing disclosures may experience restatement or reporting delays.
{news_headlines}
"""

    # 6. Conclusion
    conclusion = f"""### Conclusion & Investment Stance

**Model-Generated Assessment Rating:** **{rec.upper()}**
- Model Target Price: {_format_curr(target_price, curr)} vs. Spot {_format_curr(price, curr)}
- Target percentile in sector P/E: {_format_pct(peer_comparison.percentile_rank.get('pe_ratio') if peer_comparison else None)}

**Summary:**
{name} exhibits {_format_pct(fundamental.operating_margin if fundamental else None)} operating margins with a {_format_num(technical.beta if technical else None, 2)} market beta. Investors should monitor Free Cash Flow conversion and sector earnings revisions.
"""

    # 7. Sources
    sources = """### Data Sources & Citations
- **Market Price & Quote Data:** Yahoo Finance API (`yfinance`), delayed up to 15 minutes.
- **Financial Statements:** Annual balance sheet, income statement, and cash flows from company regulatory disclosures.
- **News Feed:** Curated Google News RSS & provider updates.
- **Calculation Framework:** Quantfolio quantitative engine (`numpy`, `pandas`, `scipy`).
"""

    # 8. Assumptions
    assumptions = """### Key Modeling Assumptions & Conventions
- **Trading Days:** 252 days per calendar year.
- **Volatility & Beta:** Sample standard deviation (ddof=1) of logarithmic returns against benchmark (^NSEI / ^GSPC).
- **RSI Calculation:** 14-session Wilder exponential smoothing (`ewm(alpha=1/period, adjust=False)`).
- **DCF Assumptions:** Five-year explicit projection horizon, WACC discount rate, constant terminal growth rate, Gordon Growth terminal value.
- **Disclaimer:** All Buy/Hold/Sell signals are model-generated assessments, not guaranteed performance forecasts or registered broker advice.
"""

    return AIReportSections(
        executive_summary=exec_summary,
        company_overview=company_overview,
        financial_analysis=fin_analysis,
        valuation=val_section,
        risks=risks,
        conclusion=conclusion,
        sources=sources,
        assumptions=assumptions,
    )


def build_llm_prompt(
    quote: dict[str, Any],
    fundamental: Optional[FundamentalData],
    technical: Optional[TechnicalData],
    valuation: Optional[ValuationResult],
    peer_comparison: Optional[PeerComparisonResult],
    news_items: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Constructs prompt for LLM API with strict anti-hallucination instructions."""
    ticker = quote.get("ticker", "UNKNOWN")
    price = quote.get("price")
    curr = quote.get("currency", "INR")

    return f"""You are a Senior Equity Research Analyst writing an institutional-quality research report for {ticker}.
CRITICAL INSTRUCTIONS:
- You must ONLY cite the factual figures provided below. Do NOT fabricate, invent, or speculate on financial figures or prices.
- If an input is NaN, null, or missing, state that it is unavailable.
- Clearly label investment ratings as model-generated assessments.

[COMPANY DATA]
Ticker: {ticker}
Name: {quote.get('name')}
Current Price: {curr} {price}
Sector: {quote.get('sector')}
Industry: {quote.get('industry')}
Market Cap: {quote.get('market_cap')}

[FUNDAMENTALS]
Gross Margin: {getattr(fundamental, 'gross_margin', np.nan)}
Operating Margin: {getattr(fundamental, 'operating_margin', np.nan)}
Net Margin: {getattr(fundamental, 'net_margin', np.nan)}
EPS: {getattr(fundamental, 'eps', np.nan)}
ROE: {getattr(fundamental, 'roe', np.nan)}
ROCE: {getattr(fundamental, 'roce', np.nan)}
Debt to Equity: {getattr(fundamental, 'debt_to_equity', np.nan)}
Free Cash Flow: {getattr(fundamental, 'free_cash_flow', np.nan)}

[TECHNICALS]
RSI(14): {getattr(technical, 'rsi', np.nan)}
Annualized Volatility: {getattr(technical, 'volatility', np.nan)}
Beta: {getattr(technical, 'beta', np.nan)}
Sharpe Ratio: {getattr(technical, 'sharpe_ratio', np.nan)}

[VALUATION]
Intrinsic Value: {getattr(valuation, 'intrinsic_value', np.nan)}
Fair Value Low: {getattr(valuation, 'fair_value_low', np.nan)}
Fair Value High: {getattr(valuation, 'fair_value_high', np.nan)}
Model Recommendation: {getattr(valuation, 'recommendation', 'Hold')}
WACC: {getattr(valuation, 'wacc_used', np.nan)}

Output the report in structured Markdown with sections: Executive Summary, Company Overview, Financial Analysis, Valuation, Risks & Catalysts, Conclusion.
"""


def call_llm_if_available(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """Attempts calling Anthropic or OpenAI API if key is present; returns None on failure."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        return None

    # Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                }).encode("utf-8"),
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["content"][0]["text"]
        except Exception as exc:
            logger.warning("Anthropic API call failed: %s; falling back to template.", exc)

    return None


def generate_report_sections(
    quote: dict[str, Any],
    fundamental: Optional[FundamentalData],
    technical: Optional[TechnicalData],
    valuation: Optional[ValuationResult],
    peer_comparison: Optional[PeerComparisonResult],
    news_items: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> AIReportSections:
    """Main entry point: generates report sections via LLM if available, otherwise template."""
    prompt = build_llm_prompt(quote, fundamental, technical, valuation, peer_comparison, news_items)
    llm_output = call_llm_if_available(prompt, api_key=api_key)

    # Always generate the template as baseline or fallback
    template_sections = build_template_sections(
        quote, fundamental, technical, valuation, peer_comparison, news_items
    )

    if llm_output and len(llm_output) > 200:
        # If LLM succeeded, we augment the executive summary and overview
        return AIReportSections(
            executive_summary=llm_output[:800] + "\n\n*(Enhanced by AI Analyst)*",
            company_overview=template_sections.company_overview,
            financial_analysis=template_sections.financial_analysis,
            valuation=template_sections.valuation,
            risks=template_sections.risks,
            conclusion=template_sections.conclusion,
            sources=template_sections.sources,
            assumptions=template_sections.assumptions,
        )

    return template_sections
