import numpy as np
import pytest
from src.equity_research.models import (
    EquityResearchInputs,
    FundamentalData,
    TechnicalData,
    DCFInputs,
    ValuationResult,
    PeerComparisonResult,
    AIReportSections,
    EquityResearchReport,
)


def test_equity_research_inputs_defaults():
    inp = EquityResearchInputs(ticker="RELIANCE.NS")
    assert inp.ticker == "RELIANCE.NS"
    assert inp.period == "1y"
    assert inp.benchmark == "^NSEI"


def test_fundamental_data_creation_and_defaults():
    fund = FundamentalData(
        gross_margin=0.45,
        operating_margin=0.20,
        net_margin=0.12,
        eps=125.5,
        roe=0.18,
        roce=0.15,
        debt_to_equity=0.55,
        debt_to_asset=0.35,
        revenue_latest=250000.0,
        profit_latest=30000.0,
    )
    assert fund.gross_margin == 0.45
    assert fund.net_margin == 0.12
    assert fund.eps == 125.5


def test_technical_data_creation():
    tech = TechnicalData(
        sma={"20": 2400.0, "50": 2350.0, "200": 2200.0},
        ema={"20": 2410.0, "50": 2360.0},
        rsi=58.5,
        macd={"macd": 12.5, "signal": 10.0, "histogram": 2.5},
        volatility=0.22,
        beta=1.05,
    )
    assert tech.rsi == 58.5
    assert tech.macd["histogram"] == 2.5
    assert tech.sma["200"] == 2200.0


def test_dcf_inputs_and_valuation_result():
    dcf_in = DCFInputs(
        fcffree=100.0,
        wacc=0.10,
        growth_rate=0.05,
        terminal_growth_rate=0.03,
        shares_outstanding=10.0,
    )
    val = ValuationResult(
        intrinsic_value=125.0,
        fair_value_low=110.0,
        fair_value_high=140.0,
        recommendation="Buy",
        dcf_interpolated={"year_1": 15.0},
        wacc_used=0.10,
        terminal_value=800.0,
        enterprise_value=950.0,
    )
    assert dcf_in.fcffree == 100.0
    assert val.intrinsic_value == 125.0
    assert val.recommendation == "Buy"


def test_ai_report_sections_and_full_report():
    sections = AIReportSections(
        executive_summary="Executive Summary text",
        company_overview="Company Overview text",
        financial_analysis="Financial Analysis text",
        valuation="Valuation text",
        risks="Risks text",
        conclusion="Conclusion text",
        sources="Sources text",
        assumptions="Assumptions text",
    )
    assert sections.executive_summary == "Executive Summary text"

    report = EquityResearchReport(
        ticker="TCS.NS",
        as_of_date="2026-10-23",
        sections=sections,
    )
    assert report.ticker == "TCS.NS"
    assert report.sections.conclusion == "Conclusion text"
