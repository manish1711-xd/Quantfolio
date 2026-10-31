"""Tests for Macro & Industry Cyclicality Overlay."""

import pytest
import numpy as np
from src.analytics.macro_overlay import (
    adjust_wacc_for_macro,
    adjust_growth_for_macro,
    compute_macro_overlay,
)


def test_macro_adjustment_applies_to_wacc():
    wacc = 0.10
    inflation = 0.06
    interest_rate = 0.075
    adjusted = adjust_wacc_for_macro(wacc, inflation, interest_rate)
    # High inflation/rates should increase WACC
    assert adjusted > wacc


def test_adjust_growth_for_macro():
    growth = 0.08
    inflation = 0.06
    adjusted = adjust_growth_for_macro(growth, inflation)
    # High inflation erodes real growth or dampens nominal momentum
    assert isinstance(adjusted, float)


def test_compute_macro_overlay():
    res = compute_macro_overlay(
        base_wacc=0.10,
        base_growth=0.08,
        inflation_rate=0.055,
        policy_rate=0.065,
        currency_volatility=0.02,
    )
    assert "adjusted_wacc" in res
    assert "adjusted_growth" in res
    assert "macro_penalty" in res
    assert res["adjusted_wacc"] > 0
