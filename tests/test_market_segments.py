"""Tests for NSE market segments and ratios analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.equity_research.market_segments import (
    compute_market_segment_ratios,
    get_nse_market_snapshot,
)


def test_compute_market_segment_ratios() -> None:
    data = pd.DataFrame([
        {"symbol": "RELIANCE", "yahoo_symbol": "RELIANCE.NS", "name": "Reliance", "exchange": "NSE", "sector": "Energy", "turnover": 1000.0},
        {"symbol": "ONGC", "yahoo_symbol": "ONGC.NS", "name": "ONGC", "exchange": "NSE", "sector": "Energy", "turnover": 500.0},
        {"symbol": "TCS", "yahoo_symbol": "TCS.NS", "name": "TCS", "exchange": "NSE", "sector": "Technology", "turnover": 800.0},
    ])
    res = compute_market_segment_ratios(data)

    assert not res.empty
    assert "Energy" in list(res["sector"])
    assert "Technology" in list(res["sector"])
    energy_row = res[res["sector"] == "Energy"].iloc[0]
    assert energy_row["stock_count"] == 2
    assert energy_row["total_turnover"] == 1500.0
    assert energy_row["turnover_share_pct"] == pytest.approx(1500.0 / 2300.0 * 100.0)
