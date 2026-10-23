"""Test suite for Equity Research Streamlit Dashboard helpers and chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity_research.models import (
    FundamentalData,
    TechnicalData,
    ValuationResult,
    PeerComparisonResult,
    AIReportSections,
    EquityResearchReport,
)
from src.ui.pages.equity_research import (
    _build_price_technical_chart,
    _build_rsi_chart,
    _build_macd_chart,
    _build_dcf_cashflow_chart,
    _build_sensitivity_heatmap,
    _build_peer_multiples_chart,
)


@pytest.fixture
def mock_ohlcv():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.normal(0.1, 1.0, size=100))
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )


def test_build_price_technical_chart(mock_ohlcv):
    tech = TechnicalData(rsi=55.0, volatility=0.20)
    fig = _build_price_technical_chart(mock_ohlcv, tech, "TEST.NS")
    assert fig is not None
    assert len(fig.data) >= 1


def test_build_rsi_chart(mock_ohlcv):
    fig = _build_rsi_chart(mock_ohlcv, "TEST.NS")
    assert fig is not None
    assert len(fig.data) >= 1


def test_build_macd_chart(mock_ohlcv):
    fig = _build_macd_chart(mock_ohlcv, "TEST.NS")
    assert fig is not None
    assert len(fig.data) >= 3  # Histogram, MACD, Signal


def test_build_dcf_cashflow_chart():
    val = ValuationResult(
        intrinsic_value=2500.0,
        fair_value_low=2200.0,
        fair_value_high=2800.0,
        recommendation="Buy",
        dcf_interpolated={"year_1": 100.0, "year_2": 110.0, "year_3": 121.0},
    )
    fig = _build_dcf_cashflow_chart(val, "₹")
    assert fig is not None
    assert len(fig.data) == 1


def test_build_sensitivity_heatmap():
    sens = pd.DataFrame(
        {"g=2.0%": [2100.0, 1900.0], "g=3.0%": [2300.0, 2050.0]},
        index=["9.0%", "10.0%"],
    )
    fig = _build_sensitivity_heatmap(sens, "₹")
    assert fig is not None
    assert len(fig.data) == 1


def test_build_peer_multiples_chart():
    peers = pd.DataFrame([
        {"ticker": "PEER1.NS", "pe_ratio": 25.0},
        {"ticker": "PEER2.NS", "pe_ratio": 30.0},
    ])
    fig = _build_peer_multiples_chart(peers, "TARGET.NS", 22.0)
    assert fig is not None
    assert len(fig.data) == 1
