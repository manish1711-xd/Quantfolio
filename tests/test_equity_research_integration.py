"""End-to-end integration tests for Equity Research Report Generator."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from src.equity_research.models import (
    DCFInputs,
    EquityResearchReport,
    FundamentalData,
    TechnicalData,
    ValuationResult,
    PeerComparisonResult,
)
from src.equity_research.fundamental import analyze_fundamentals
from src.equity_research.technical import analyze_technicals
from src.equity_research.dcf import calculate_dcf_valuation, generate_scenario_analysis
from src.equity_research.peer_comparison import analyze_peer_comparison
from src.equity_research.ai_report import generate_report_sections
from src.equity_research.export import (
    build_html_report,
    export_equity_report_pdf,
    export_equity_research_excel,
)
from src.equity_research.history import (
    save_research_record,
    load_research_history,
    delete_research_record,
    init_history_db,
)


@pytest.fixture
def mock_bundle():
    """Create complete mock research bundle for RELIANCE.NS."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    prices = 2400.0 + np.cumsum(np.random.normal(0.5, 5.0, size=252))
    ohlcv = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 10.0,
            "Low": prices - 10.0,
            "Close": prices,
            "Volume": 1000000,
        },
        index=dates,
    )
    bench_prices = 18000.0 + np.cumsum(np.random.normal(1.0, 15.0, size=252))
    bench_hist = pd.DataFrame({"Close": bench_prices}, index=dates)

    income = pd.DataFrame(
        {
            "2024-03-31": [900000.0, 315000.0, 160000.0, 80000.0],
        },
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    balance = pd.DataFrame(
        {
            "2024-03-31": [1200000.0, 600000.0, 270000.0],
        },
        index=["Total Assets", "Stockholders Equity", "Total Debt"],
    )
    cash_flow = pd.DataFrame(
        {
            "2024-03-31": [120000.0, -75000.0, 45000.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
    )

    quote = {
        "ticker": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "price": 2550.0,
        "currency": "INR",
        "sector": "Energy",
        "industry": "Oil & Gas Refining",
        "market_cap": 17200000000000.0,
        "shares_outstanding": 6760000000.0,
        "pe_ratio": 24.5,
        "pb_ratio": 2.2,
        "summary": "Conglomerate with presence across energy, retail, and telecommunications.",
    }

    peers = pd.DataFrame([
        {"ticker": "ONGC.NS", "name": "ONGC", "pe_ratio": 7.5, "pb_ratio": 1.1, "beta": 0.85},
        {"ticker": "BPCL.NS", "name": "BPCL", "pe_ratio": 8.0, "pb_ratio": 1.4, "beta": 1.10},
        {"ticker": "IOC.NS", "name": "Indian Oil", "pe_ratio": 6.8, "pb_ratio": 1.0, "beta": 0.95},
    ])

    news = [
        {"title": "Reliance expanding renewable energy portfolio", "source": "Reuters", "published": "2026-10-20"},
        {"title": "Strong quarterly subscriber growth at Jio", "source": "Bloomberg", "published": "2026-10-18"},
    ]

    return {
        "quote": quote,
        "statements": {"income": income, "balance_sheet": balance, "cash_flow": cash_flow},
        "ohlcv": ohlcv,
        "bench_hist": bench_hist,
        "peers": peers,
        "news": news,
    }


def test_end_to_end_equity_research_pipeline(mock_bundle, tmp_path):
    """Test full pipeline: data ingestion -> fundamental & technical -> DCF -> peers -> report -> export -> persistence."""
    quote = mock_bundle["quote"]
    stmts = mock_bundle["statements"]
    ohlcv = mock_bundle["ohlcv"]
    bench_hist = mock_bundle["bench_hist"]
    peers_df = mock_bundle["peers"]
    news = mock_bundle["news"]

    # 1. Fundamental analysis
    fundamental = analyze_fundamentals(stmts, quote_info=quote)
    assert fundamental.gross_margin == pytest.approx(315000.0 / 900000.0)
    assert fundamental.operating_margin == pytest.approx(160000.0 / 900000.0)
    assert fundamental.roe == pytest.approx(80000.0 / 600000.0)
    assert fundamental.debt_to_equity == pytest.approx(270000.0 / 600000.0)
    assert fundamental.free_cash_flow == 45000.0

    # 2. Technical analysis
    technical = analyze_technicals(ohlcv, bench_df=bench_hist)
    assert not np.isnan(technical.rsi)
    assert 0 <= technical.rsi <= 100
    assert not np.isnan(technical.volatility)
    assert technical.volatility > 0
    assert not np.isnan(technical.beta)

    # 3. DCF valuation
    dcf_in = DCFInputs(
        fcffree=fundamental.free_cash_flow,
        wacc=0.10,
        growth_rate=0.08,
        terminal_growth_rate=0.03,
        shares_outstanding=quote["shares_outstanding"],
        net_debt=0.0,
    )
    val_res = calculate_dcf_valuation(dcf_in, current_price=quote["price"])
    assert val_res.intrinsic_value is not None
    assert val_res.intrinsic_value > 0
    assert val_res.recommendation in {"Buy", "Hold", "Sell"}

    # Scenarios
    scenarios = generate_scenario_analysis(
        base_fcf=fundamental.free_cash_flow,
        base_wacc=0.10,
        base_growth=0.08,
        terminal_growth=0.03,
        shares_outstanding=quote["shares_outstanding"],
        current_price=quote["price"],
    )
    assert "Bull" in scenarios and "Base" in scenarios and "Bear" in scenarios
    assert scenarios["Bull"]["intrinsic_value"] > scenarios["Bear"]["intrinsic_value"]

    # 4. Peer comparison
    target_multiples = {"pe_ratio": quote["pe_ratio"], "pb_ratio": quote["pb_ratio"]}
    peer_res = analyze_peer_comparison("RELIANCE.NS", peers_df, target_multiples)
    assert peer_res.sufficient_data is True
    assert "pe_ratio" in peer_res.sector_averages

    # 5. Report generation
    sections = generate_report_sections(
        quote=quote,
        fundamental=fundamental,
        technical=technical,
        valuation=val_res,
        peer_comparison=peer_res,
        news_items=news,
    )
    assert "RELIANCE.NS" in sections.executive_summary
    assert "Valuation" in sections.valuation
    assert "Risks" in sections.risks

    full_report = EquityResearchReport(
        ticker="RELIANCE.NS",
        as_of_date="2026-10-23",
        sections=sections,
        quote=quote,
        fundamental=fundamental,
        technical=technical,
        valuation=val_res,
        peer_comparison=peer_res,
    )

    # 6. HTML & PDF export
    html_text = build_html_report(full_report)
    assert "RELIANCE.NS" in html_text
    assert "Executive Summary" in html_text

    pdf_bytes = export_equity_report_pdf(full_report)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500

    # 7. Excel export
    excel_bytes = export_equity_research_excel(full_report)
    assert isinstance(excel_bytes, bytes)
    assert excel_bytes.startswith(b"PK")  # valid zip/xlsx magic bytes

    # 8. SQLite History Persistence
    test_db = str(tmp_path / "integration_history.db")
    rec_id = save_research_record(
        ticker="RELIANCE.NS",
        company_name=quote["name"],
        price=quote["price"],
        intrinsic_value=val_res.intrinsic_value,
        recommendation=val_res.recommendation,
        summary=sections.executive_summary[:200],
        db_path=test_db,
    )
    assert rec_id > 0

    history = load_research_history(db_path=test_db)
    assert len(history) == 1
    assert history[0]["ticker"] == "RELIANCE.NS"
    assert history[0]["intrinsic_value"] == pytest.approx(val_res.intrinsic_value)
