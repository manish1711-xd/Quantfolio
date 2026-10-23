import numpy as np
import pytest

from src.equity_research.ai_report import (
    generate_report_sections,
    build_llm_prompt,
    build_template_sections,
    call_llm_if_available,
)
from src.equity_research.models import (
    FundamentalData,
    TechnicalData,
    ValuationResult,
    PeerComparisonResult,
    AIReportSections,
)


@pytest.fixture
def mock_components():
    quote = {
        "ticker": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "price": 2500.0,
        "currency": "INR",
        "sector": "Energy",
        "industry": "Oil & Gas Refining",
        "market_cap": 16900000000000.0,
        "summary": "Diversified energy, retail, and telecommunications conglomerate in India.",
    }
    fundamental = FundamentalData(
        gross_margin=0.35,
        operating_margin=0.18,
        net_margin=0.09,
        eps=105.0,
        roe=0.12,
        roce=0.10,
        debt_to_equity=0.45,
        debt_to_asset=0.25,
        revenue_latest=900000.0,
        profit_latest=80000.0,
        free_cash_flow=45000.0,
        currency="INR",
    )
    technical = TechnicalData(
        sma={"20": 2480.0, "50": 2420.0, "200": 2300.0},
        ema={"20": 2485.0, "50": 2430.0},
        rsi=56.0,
        macd={"macd": 15.0, "signal": 12.0, "histogram": 3.0},
        volatility=0.19,
        beta=1.05,
        sharpe_ratio=0.85,
        max_drawdown=-0.14,
        current_price=2500.0,
    )
    valuation = ValuationResult(
        intrinsic_value=2850.0,
        fair_value_low=2550.0,
        fair_value_high=3150.0,
        recommendation="Buy",
        wacc_used=0.10,
        terminal_value=600000.0,
        enterprise_value=850000.0,
        equity_value=750000.0,
    )
    peer_comp = PeerComparisonResult(
        target_ticker="RELIANCE.NS",
        target_multiples={"pe_ratio": 24.0, "pb_ratio": 2.2},
        sector_averages={"pe_ratio": 21.0, "pb_ratio": 1.8},
        percentile_rank={"pe_ratio": 0.65},
        sufficient_data=True,
    )
    return quote, fundamental, technical, valuation, peer_comp


def test_build_template_sections(mock_components):
    quote, fundamental, technical, valuation, peer_comp = mock_components
    sections = build_template_sections(quote, fundamental, technical, valuation, peer_comp)
    assert isinstance(sections, AIReportSections)
    assert "RELIANCE.NS" in sections.executive_summary
    assert "₹2,850.00" in sections.executive_summary or "2850" in sections.executive_summary
    assert "Energy" in sections.company_overview
    assert "0.35" in sections.financial_analysis or "35.0%" in sections.financial_analysis
    assert "DCF" in sections.valuation
    assert "Model-Generated" in sections.conclusion or "Assessment" in sections.conclusion
    assert "Yahoo Finance" in sections.sources
    assert "WACC" in sections.assumptions


def test_build_llm_prompt(mock_components):
    quote, fundamental, technical, valuation, peer_comp = mock_components
    prompt = build_llm_prompt(quote, fundamental, technical, valuation, peer_comp)
    assert "RELIANCE.NS" in prompt
    assert "Do NOT fabricate" in prompt
    assert "2500.0" in prompt
    assert "2850.0" in prompt


def test_generate_report_sections_fallback(mock_components):
    # Without any API keys configured, it should smoothly fall back to deterministic template
    quote, fundamental, technical, valuation, peer_comp = mock_components
    sections = generate_report_sections(
        quote, fundamental, technical, valuation, peer_comp, news_items=[], api_key=None
    )
    assert isinstance(sections, AIReportSections)
    assert len(sections.executive_summary) > 50
    assert len(sections.financial_analysis) > 50
    assert len(sections.valuation) > 50
