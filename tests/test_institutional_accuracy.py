"""Comprehensive Unit Tests for All 5 Institutional Accuracy Modules."""

import pytest
import numpy as np
import pandas as pd
from src.analytics.wacc_engine import calculate_dynamic_wacc, compute_dynamic_cost_of_equity, calculate_synthetic_credit_spread
from src.analytics.ttm_engine import aggregate_ttm_metrics, compute_seasonality_indices
from src.analytics.ml_forecast import forecast_revenue_growth, forecast_margin_trajectory
from src.analytics.backtest_engine import evaluate_historical_prediction, calculate_bias_correction_factor, apply_bias_correction
from src.analytics.macro_overlay import adjust_wacc_for_macro, adjust_growth_for_macro, compute_macro_overlay


def test_all_institutional_accuracy_modules_integrated_and_functional():
    # 1. WACC
    ke = compute_dynamic_cost_of_equity(0.07, 1.1, 0.055)
    kd = 0.07 + calculate_synthetic_credit_spread(4.0)
    wacc = calculate_dynamic_wacc(10000.0, 2500.0, ke, kd, 0.25)
    assert 0.05 <= wacc <= 0.25

    # 2. TTM
    inc = pd.DataFrame({"Total Revenue": [100.0, 100.0, 100.0, 100.0]}, index=["Q1", "Q2", "Q3", "Q4"]).T
    cf = pd.DataFrame({"Operating Cash Flow": [20.0, 20.0, 20.0, 20.0]}, index=["Q1", "Q2", "Q3", "Q4"]).T
    bs = pd.DataFrame({"Total Assets": [500.0], "Total Debt": [100.0]}, index=["Q1"]).T
    ttm = aggregate_ttm_metrics(inc, cf, bs)
    assert ttm["ttm_revenue"] == 400.0

    # 3. ML Forecast
    fc = forecast_revenue_growth([100.0, 110.0, 121.0], horizon_years=3)
    assert len(fc["forecasts"]) == 3

    # 4. Backtest
    bias = calculate_bias_correction_factor([120.0], [100.0])
    corr = apply_bias_correction(100.0, bias)
    assert corr <= 100.0

    # 5. Macro
    macro = compute_macro_overlay(0.10, 0.08, 0.06, 0.075, 0.03)
    assert macro["adjusted_wacc"] > 0
