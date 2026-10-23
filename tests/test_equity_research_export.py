import io
import pandas as pd
import numpy as np
import pytest

from src.equity_research.export import (
    build_html_report,
    export_equity_report_pdf,
    export_equity_research_excel,
)
from src.equity_research.models import (
    EquityResearchReport,
    AIReportSections,
    FundamentalData,
    TechnicalData,
    ValuationResult,
    PeerComparisonResult,
)


@pytest.fixture
def sample_report():
    sections = AIReportSections(
        executive_summary="### Executive Summary\nStrong operational momentum.",
        company_overview="### Company Overview\nLeading energy and tech conglomerate.",
        financial_analysis="### Financial Analysis\nOperating margin 18.5%, ROE 14.2%.",
        valuation="### Valuation\nDCF fair value ₹2,900 per share.",
        risks="### Risks\nRegulatory and commodity volatility.",
        conclusion="### Conclusion\nModel assessment: BUY.",
        sources="### Sources\nYahoo Finance via yfinance.",
        assumptions="### Assumptions\nDiscount rate WACC 10.0%, terminal growth 3.0%.",
    )
    quote = {
        "ticker": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "price": 2500.0,
        "currency": "INR",
        "sector": "Energy",
        "industry": "Oil & Gas",
        "market_cap": 16900000000000.0,
    }
    fundamental = FundamentalData(
        gross_margin=0.35,
        operating_margin=0.18,
        net_margin=0.09,
        eps=105.0,
        roe=0.14,
        roce=0.12,
        debt_to_equity=0.45,
        debt_to_asset=0.25,
        revenue_latest=900000.0,
        profit_latest=80000.0,
        free_cash_flow=45000.0,
    )
    technical = TechnicalData(
        sma={"20": 2480.0, "50": 2420.0, "200": 2300.0},
        rsi=55.0,
        macd={"macd": 12.0, "signal": 10.0, "histogram": 2.0},
        volatility=0.18,
        beta=1.05,
        current_price=2500.0,
    )
    valuation = ValuationResult(
        intrinsic_value=2900.0,
        fair_value_low=2600.0,
        fair_value_high=3200.0,
        recommendation="Buy",
        wacc_used=0.10,
        enterprise_value=850000.0,
        equity_value=750000.0,
        sensitivity_table=pd.DataFrame(
            {"g=2.0%": [2700.0, 2500.0], "g=3.0%": [2900.0, 2700.0]},
            index=["WACC=9.0%", "WACC=10.0%"],
        ),
    )
    peer_comp = PeerComparisonResult(
        target_ticker="RELIANCE.NS",
        peers=pd.DataFrame([
            {"ticker": "ONGC.NS", "name": "ONGC", "pe_ratio": 7.5, "pb_ratio": 1.1},
            {"ticker": "BPCL.NS", "name": "BPCL", "pe_ratio": 8.0, "pb_ratio": 1.4},
        ]),
        target_multiples={"pe_ratio": 24.0, "pb_ratio": 2.2},
        sector_averages={"pe_ratio": 7.75, "pb_ratio": 1.25},
        percentile_rank={"pe_ratio": 1.0},
        sufficient_data=True,
    )
    return EquityResearchReport(
        ticker="RELIANCE.NS",
        as_of_date="2026-10-23",
        sections=sections,
        quote=quote,
        fundamental=fundamental,
        technical=technical,
        valuation=valuation,
        peer_comparison=peer_comp,
    )


def test_build_html_report(sample_report):
    html = build_html_report(sample_report)
    assert "<!DOCTYPE html>" in html
    assert "RELIANCE.NS" in html
    assert "Reliance Industries Limited" in html
    assert "Executive Summary" in html
    assert "DCF Valuation" in html or "Valuation" in html
    assert "Disclaimer" in html or "disclaimer" in html


def test_export_equity_research_excel(sample_report):
    excel_bytes = export_equity_research_excel(sample_report)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000
    # Header of valid zip/xlsx file is PK\x03\x04
    assert excel_bytes.startswith(b"PK")


def test_export_equity_report_pdf(sample_report):
    pdf_bytes = export_equity_report_pdf(sample_report)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
