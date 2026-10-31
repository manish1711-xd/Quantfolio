"""Macro & Industry Cyclicality Overlay.

Applies macroeconomic and monetary regime overlays to valuation models:
- Inflation rate, central bank repo rate, and currency exchange rate volatility
- Macro penalty/bonus multipliers adjusting discount rates (WACC) and cash flow growth
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def adjust_wacc_for_macro(
    base_wacc: float,
    inflation_rate: float = 0.05,
    policy_rate: float = 0.065,
    normal_inflation: float = 0.04,
    normal_policy_rate: float = 0.06,
) -> float:
    """Adjust discount rate (WACC) for macroeconomic tightening or easing."""
    w = float(base_wacc) if base_wacc and not np.isnan(base_wacc) else 0.10
    inf = float(inflation_rate) if inflation_rate and not np.isnan(inflation_rate) else 0.05
    pol = float(policy_rate) if policy_rate and not np.isnan(policy_rate) else 0.065

    inf_spread = inf - normal_inflation
    pol_spread = pol - normal_policy_rate

    # If inflation or interest rates exceed baseline, add a macro tightening penalty
    penalty = max(0.0, inf_spread * 0.5 + pol_spread * 0.5)
    adjusted = w + penalty

    return float(max(0.04, min(0.35, adjusted)))


def adjust_growth_for_macro(
    base_growth: float,
    inflation_rate: float = 0.05,
    normal_inflation: float = 0.04,
) -> float:
    """Adjust nominal growth rate for inflationary or deflationary pressure."""
    g = float(base_growth) if base_growth and not np.isnan(base_growth) else 0.08
    inf = float(inflation_rate) if inflation_rate and not np.isnan(inflation_rate) else 0.05

    # Moderate inflation adds to nominal growth; hyperinflation or stagnation depresses real momentum
    inf_diff = inf - normal_inflation
    adjusted = g + (inf_diff * 0.2)

    return float(max(-0.10, min(0.30, adjusted)))


def compute_macro_overlay(
    base_wacc: float,
    base_growth: float,
    inflation_rate: float = 0.05,
    policy_rate: float = 0.065,
    currency_volatility: float = 0.02,
) -> dict[str, Any]:
    """Compute comprehensive macroeconomic overlay parameters."""
    adj_wacc = adjust_wacc_for_macro(base_wacc, inflation_rate, policy_rate)
    adj_growth = adjust_growth_for_macro(base_growth, inflation_rate)
    macro_penalty = adj_wacc - base_wacc

    # Currency risk premium if volatility is high (> 4%)
    fx_penalty = max(0.0, (currency_volatility - 0.04) * 0.5) if currency_volatility and not np.isnan(currency_volatility) else 0.0
    if fx_penalty > 0:
        adj_wacc += fx_penalty

    return {
        "base_wacc": round(base_wacc, 4),
        "adjusted_wacc": round(adj_wacc, 4),
        "base_growth": round(base_growth, 4),
        "adjusted_growth": round(adj_growth, 4),
        "macro_penalty": round(macro_penalty, 4),
        "fx_penalty": round(fx_penalty, 4),
        "inflation_rate": round(inflation_rate, 4),
        "policy_rate": round(policy_rate, 4),
    }
