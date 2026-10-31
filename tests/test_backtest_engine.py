"""Tests for Valuation Backtesting & Prediction Error Attribution Engine."""

import pytest
import numpy as np
from src.analytics.backtest_engine import (
    calculate_bias_correction_factor,
    apply_bias_correction,
    evaluate_historical_prediction,
)


def test_calculate_bias_correction_factor():
    # Historical predictions vs actual realizations
    # If model consistently overpredicted by 10%, bias factor is < 1.0
    predictions = [120.0, 150.0, 200.0]
    actuals = [100.0, 135.0, 180.0]

    factor = calculate_bias_correction_factor(predictions, actuals)
    assert 0.7 <= factor <= 1.0


def test_apply_bias_correction():
    base_forecast = 100.0
    bias_factor = 0.90
    corrected = apply_bias_correction(base_forecast, bias_factor)
    assert corrected == 90.0


def test_evaluate_historical_prediction():
    res = evaluate_historical_prediction(intrinsic_prediction=110.0, realized_price=100.0)
    assert "error_pct" in res
    assert "bias_direction" in res
    assert res["bias_direction"] == "overpredicted"
    assert res["error_pct"] == 0.10
