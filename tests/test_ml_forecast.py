"""Tests for ML & Statistical Revenue/Margin Forecasting Engine."""

import pytest
import numpy as np
import pandas as pd
from src.analytics.ml_forecast import (
    forecast_revenue_growth,
    forecast_margin_trajectory,
)


def test_forecast_revenue_growth_returns_interval():
    # 5 years of historical revenue growth series
    series = pd.Series([1000.0, 1100.0, 1210.0, 1330.0, 1460.0])
    res = forecast_revenue_growth(series, horizon_years=5, confidence_level=0.80)

    assert "forecasts" in res
    assert "lower_bound" in res
    assert "upper_bound" in res
    assert len(res["forecasts"]) == 5
    assert len(res["lower_bound"]) == 5
    assert len(res["upper_bound"]) == 5

    # Check bounds order: lower <= forecast <= upper
    for f, l, u in zip(res["forecasts"], res["lower_bound"], res["upper_bound"]):
        assert l <= f <= u


def test_forecast_margin_trajectory():
    margin_series = pd.Series([0.15, 0.16, 0.155, 0.165, 0.17])
    margins = forecast_margin_trajectory(margin_series, horizon_years=5)

    assert len(margins) == 5
    for m in margins:
        assert 0.0 <= m <= 1.0
