import numpy as np
import pandas as pd
import pytest

from src.equity_research.peer_comparison import (
    analyze_peer_comparison,
    compute_sector_averages,
    compute_percentiles,
)


@pytest.fixture
def sample_peers_df():
    return pd.DataFrame([
        {"ticker": "INFY.NS", "name": "Infosys", "pe_ratio": 24.5, "pb_ratio": 7.2, "beta": 0.95, "dividend_yield": 0.024},
        {"ticker": "WIPRO.NS", "name": "Wipro", "pe_ratio": 20.0, "pb_ratio": 3.1, "beta": 0.85, "dividend_yield": 0.015},
        {"ticker": "HCLTECH.NS", "name": "HCL Tech", "pe_ratio": 26.0, "pb_ratio": 5.5, "beta": 1.05, "dividend_yield": 0.030},
    ])


def test_compute_sector_averages(sample_peers_df):
    avgs = compute_sector_averages(sample_peers_df)
    assert np.isclose(avgs["pe_ratio"], (24.5 + 20.0 + 26.0) / 3.0)
    assert np.isclose(avgs["pb_ratio"], (7.2 + 3.1 + 5.5) / 3.0)


def test_compute_percentiles(sample_peers_df):
    target = {"pe_ratio": 25.0, "pb_ratio": 6.0}
    pctiles = compute_percentiles(target, sample_peers_df)
    # Target PE is 25.0. Peers are 20.0, 24.5, 26.0.
    # 2 peers are below target, 1 above -> percentile is 2/3 = 0.6667
    assert np.isclose(pctiles["pe_ratio"], 2.0 / 3.0)


def test_analyze_peer_comparison_sufficient_data(sample_peers_df):
    target_multiples = {"pe_ratio": 28.0, "pb_ratio": 8.0, "beta": 1.1, "dividend_yield": 0.018}
    res = analyze_peer_comparison("TCS.NS", sample_peers_df, target_multiples)
    assert res.target_ticker == "TCS.NS"
    assert res.sufficient_data is True
    assert len(res.sector_averages) > 0
    assert res.percentile_rank["pe_ratio"] == 1.0  # TCS PE 28 > all peers


def test_analyze_peer_comparison_insufficient_data():
    empty_peers = pd.DataFrame()
    res = analyze_peer_comparison("UNKNOWN.NS", empty_peers, {"pe_ratio": 15.0})
    assert res.sufficient_data is False
    assert any("insufficient" in n.lower() or "no peer" in n.lower() for n in res.notes)
