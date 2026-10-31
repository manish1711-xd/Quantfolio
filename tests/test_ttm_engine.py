"""Tests for Quarterly TTM & Seasonality Engine."""

import pytest
import numpy as np
import pandas as pd
from src.analytics.ttm_engine import (
    aggregate_ttm_metrics,
    compute_seasonality_indices,
)


def _make_sample_quarterly_statements():
    cols = ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    # Income stmt
    income_data = {
        "Total Revenue": [2500.0, 2400.0, 2300.0, 2200.0, 2100.0],
        "Operating Income": [500.0, 480.0, 460.0, 440.0, 420.0],
        "Net Income": [375.0, 360.0, 345.0, 330.0, 315.0],
        "EBITDA": [600.0, 580.0, 560.0, 540.0, 520.0],
    }
    income_df = pd.DataFrame(income_data, index=cols).T

    # Cash flow
    cf_data = {
        "Operating Cash Flow": [450.0, 440.0, 420.0, 400.0, 390.0],
        "Capital Expenditure": [-100.0, -90.0, -80.0, -70.0, -60.0],
    }
    cf_df = pd.DataFrame(cf_data, index=cols).T

    # Balance sheet
    bs_data = {
        "Total Assets": [15000.0, 14500.0, 14000.0, 13500.0, 13000.0],
        "Total Debt": [3000.0, 3100.0, 3200.0, 3300.0, 3400.0],
        "Cash And Cash Equivalents": [1200.0, 1100.0, 1000.0, 900.0, 800.0],
        "Stockholders Equity": [8000.0, 7800.0, 7600.0, 7400.0, 7200.0],
    }
    bs_df = pd.DataFrame(bs_data, index=cols).T

    return income_df, cf_df, bs_df


def test_aggregate_ttm_metrics_sums_trailing_4_quarters():
    inc_df, cf_df, bs_df = _make_sample_quarterly_statements()
    ttm = aggregate_ttm_metrics(inc_df, cf_df, bs_df)

    # Revenue sum of first 4 cols: 2500 + 2400 + 2300 + 2200 = 9400
    assert ttm["ttm_revenue"] == 9400.0
    # Operating Income: 500 + 480 + 460 + 440 = 1880
    assert ttm["ttm_operating_income"] == 1880.0
    # Net Income: 375 + 360 + 345 + 330 = 1410
    assert ttm["ttm_net_income"] == 1410.0
    # EBITDA: 600 + 580 + 560 + 540 = 2280
    assert ttm["ttm_ebitda"] == 2280.0
    # OCF: 450 + 440 + 420 + 400 = 1710
    assert ttm["ttm_operating_cash_flow"] == 1710.0
    # Capex: |-100| + |-90| + |-80| + |-70| = 340
    assert ttm["ttm_capex"] == 340.0
    # Free cash flow: 1710 - 340 = 1370
    assert ttm["ttm_free_cash_flow"] == 1370.0

    # Margins
    assert round(ttm["ttm_operating_margin"], 4) == round(1880.0 / 9400.0, 4)
    assert round(ttm["ttm_ebitda_margin"], 4) == round(2280.0 / 9400.0, 4)

    # Balance sheet: takes latest period (2026-09-30)
    assert ttm["latest_total_debt"] == 3000.0
    assert ttm["latest_cash"] == 1200.0
    assert ttm["latest_net_debt"] == 1800.0
    assert ttm["quarters_aggregated"] == 4


def test_compute_seasonality_indices():
    # 8 quarters spanning 2 full calendar years
    dates = pd.date_range(start="2024-01-01", periods=8, freq="QE")
    # Simulate seasonal pattern: Q1 normal, Q2 low, Q3 holiday peak, Q4 strong
    rev_vals = [100.0, 80.0, 140.0, 120.0, 105.0, 85.0, 145.0, 125.0]
    rev_series = pd.Series(rev_vals, index=dates)

    indices = compute_seasonality_indices(rev_series)
    assert "Q1" in indices
    assert "Q2" in indices
    assert "Q3" in indices
    assert "Q4" in indices
    # Q3 should be the highest seasonal factor
    assert indices["Q3"] > indices["Q2"]
    # Mean of seasonal factors should be close to 1.0
    factors = list(indices.values())
    assert abs(np.mean(factors) - 1.0) < 0.1
