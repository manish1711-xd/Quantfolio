"""Tests for Dynamic WACC & Cost of Capital Engine."""

import math
import pytest
import numpy as np
from src.analytics.wacc_engine import (
    calculate_synthetic_credit_spread,
    compute_dynamic_cost_of_equity,
    calculate_dynamic_wacc,
    estimate_company_wacc,
)


def test_calculate_synthetic_credit_spread():
    # AAA rating (interest coverage > 8.5)
    spread_aaa = calculate_synthetic_credit_spread(10.0)
    assert 0.005 <= spread_aaa <= 0.012

    # BBB rating (coverage ~ 3.5)
    spread_bbb = calculate_synthetic_credit_spread(3.5)
    assert 0.015 <= spread_bbb <= 0.028

    # Distressed / Low coverage (< 1.5)
    spread_low = calculate_synthetic_credit_spread(0.8)
    assert spread_low >= 0.04

    # Negative / NaN coverage handles safely
    spread_nan = calculate_synthetic_credit_spread(np.nan)
    assert spread_nan > 0
    spread_neg = calculate_synthetic_credit_spread(-2.0)
    assert spread_neg >= 0.04


def test_compute_dynamic_cost_of_equity():
    rf = 0.07  # 7% 10Y Indian G-Sec proxy
    beta = 1.2
    erp = 0.055  # 5.5% ERP
    ke = compute_dynamic_cost_of_equity(rf, beta, erp)
    expected = 0.07 + 1.2 * 0.055
    assert math.isclose(ke, expected, rel_tol=1e-4)


def test_calculate_dynamic_wacc():
    market_cap = 8000.0
    total_debt = 2000.0
    cost_of_equity = 0.136
    cost_of_debt = 0.085
    tax_rate = 0.25

    wacc = calculate_dynamic_wacc(
        market_cap=market_cap,
        total_debt=total_debt,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
    )
    # E_weight = 0.8, D_weight = 0.2
    # After-tax cost of debt = 0.085 * (1 - 0.25) = 0.06375
    # WACC = 0.8 * 0.136 + 0.2 * 0.06375 = 0.1088 + 0.01275 = 0.12155
    expected = 0.8 * 0.136 + 0.2 * 0.085 * (1.0 - 0.25)
    assert math.isclose(wacc, expected, rel_tol=1e-4)


def test_estimate_company_wacc_orchestration():
    quote_info = {
        "market_cap": 1000000.0,
        "total_debt": 250000.0,
        "currency": "INR",
    }
    result = estimate_company_wacc(
        ticker="RELIANCE.NS",
        quote_info=quote_info,
        beta=1.1,
        interest_coverage=4.5,
        risk_free_rate=0.071,
    )
    assert "wacc" in result
    assert "cost_of_equity" in result
    assert "cost_of_debt" in result
    assert "synthetic_rating" in result
    assert 0.05 <= result["wacc"] <= 0.25
