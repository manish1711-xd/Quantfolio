from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest

from src.equity_research.data_access import (
    fetch_research_bundle,
    fetch_equity_quote,
    fetch_equity_fundamentals,
    fetch_equity_history,
    fetch_peer_quotes,
)


def test_fetch_equity_quote_with_mock():
    with patch("src.equity_research.data_access.fetch_profile") as mock_prof, \
         patch("src.equity_research.data_access.fetch_quote") as mock_q:
        mock_q.return_value = MagicMock(
            ticker="RELIANCE.NS",
            price=2500.0,
            currency="INR",
            previous_close=2480.0,
            day_high=2520.0,
            day_low=2475.0,
            volume=5000000,
        )
        mock_prof.return_value = MagicMock(
            ticker="RELIANCE.NS",
            name="Reliance Industries Limited",
            sector="Energy",
            industry="Oil & Gas Refining & Marketing",
            currency="INR",
            country="India",
            shares_outstanding=6765000000,
            pe_ratio=25.4,
            pb_ratio=2.1,
            beta=1.1,
            dividend_yield=0.004,
            summary="Conglomerate engaged in refining, telecom, and retail.",
        )
        quote_dict, warnings = fetch_equity_quote("RELIANCE.NS")
        assert quote_dict["ticker"] == "RELIANCE.NS"
        assert quote_dict["price"] == 2500.0
        assert quote_dict["sector"] == "Energy"
        assert quote_dict["shares_outstanding"] == 6765000000
        assert len(warnings) == 0


def test_fetch_equity_fundamentals_mocked():
    with patch("src.equity_research.data_access.fetch_statements") as mock_stmt:
        income_df = pd.DataFrame(
            {"2025": [1000000, 200000, 150000, 100000]},
            index=["Total Revenue", "Operating Income", "EBIT", "Net Income"]
        )
        balance_df = pd.DataFrame(
            {"2025": [2000000, 800000, 1200000, 300000]},
            index=["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Long Term Debt"]
        )
        cf_df = pd.DataFrame(
            {"2025": [180000, -80000, 100000]},
            index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]
        )
        mock_stmt.return_value = (
            {"income": income_df, "balance_sheet": balance_df, "cash_flow": cf_df},
            ""
        )
        stmts, warnings = fetch_equity_fundamentals("RELIANCE.NS")
        assert "income" in stmts
        assert not stmts["income"].empty
        assert "balance_sheet" in stmts
        assert "cash_flow" in stmts


def test_fetch_research_bundle_integration_mocked():
    with patch("src.equity_research.data_access.fetch_equity_quote") as mock_q, \
         patch("src.equity_research.data_access.fetch_equity_fundamentals") as mock_f, \
         patch("src.equity_research.data_access.fetch_equity_history") as mock_h, \
         patch("src.equity_research.data_access.fetch_company_news") as mock_news, \
         patch("src.equity_research.data_access.fetch_peer_quotes") as mock_peers:

        mock_q.return_value = ({"ticker": "TCS.NS", "price": 3500.0, "sector": "Technology", "name": "Tata Consultancy"}, [])
        mock_f.return_value = ({"income": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cash_flow": pd.DataFrame()}, ["Missing statements"])
        mock_h.return_value = (pd.DataFrame({"Close": [3400.0, 3500.0]}), pd.DataFrame({"Close": [22000.0, 22100.0]}), [])
        mock_news.return_value = []
        mock_peers.return_value = (pd.DataFrame(), [])

        bundle, warnings = fetch_research_bundle("TCS.NS", period="1y", benchmark="^NSEI")
        assert bundle["ticker"] == "TCS.NS"
        assert bundle["quote"]["price"] == 3500.0
        assert "Missing statements" in warnings
