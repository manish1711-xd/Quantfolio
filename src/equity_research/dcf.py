"""Discounted Cash Flow (DCF) Valuation and Sensitivity Engine.

Features:
- Five-year free cash flow projections
- WACC-discounted enterprise value and equity value
- Gordon Growth terminal value modeling
- Sensitivity matrix across WACC and terminal growth rates
- Bull / Base / Bear scenario analysis
- Clear model-generated ratings with transparent assumptions and missing-data guards.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.equity_research.models import DCFInputs, ValuationResult

logger = logging.getLogger(__name__)


def project_free_cash_flows(base_fcf: float, growth_rate: float, years: int = 5) -> list[float]:
    """Project Free Cash Flows for N forward years with geometric growth."""
    if base_fcf is None or np.isnan(base_fcf) or base_fcf <= 0:
        return []
    projections: list[float] = []
    current = base_fcf
    for _ in range(1, years + 1):
        current *= (1.0 + growth_rate)
        projections.append(float(current))
    return projections


def compute_terminal_value(final_fcf: float, wacc: float, terminal_growth: float) -> float:
    """Gordon Growth terminal value = Final FCF * (1 + g) / (WACC - g).
    Returns np.nan if WACC <= terminal_growth or WACC <= 0.
    """
    if wacc is None or terminal_growth is None or np.isnan(wacc) or np.isnan(terminal_growth):
        return np.nan
    if wacc <= terminal_growth or wacc <= 0:
        return np.nan
    numerator = final_fcf * (1.0 + terminal_growth)
    denominator = wacc - terminal_growth
    return float(numerator / denominator)


def calculate_dcf_valuation(
    inputs: DCFInputs,
    current_price: Optional[float] = None,
) -> ValuationResult:
    """Run full five-year DCF valuation with terminal value and sensitivity."""
    status_notes: list[str] = []

    # Guard 1: FCF validation
    if inputs.fcffree is None or np.isnan(inputs.fcffree) or inputs.fcffree <= 0:
        status_notes.append("Base Free Cash Flow is negative, non-positive or unavailable; DCF requires positive cash flows.")
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=status_notes,
        )

    # Guard 2: WACC validation
    if inputs.wacc is None or np.isnan(inputs.wacc) or inputs.wacc <= 0:
        status_notes.append("WACC must be positive and non-zero.")
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=status_notes,
        )

    # Guard 3: WACC vs terminal growth
    term_g = inputs.terminal_growth_rate if inputs.terminal_growth_rate is not None else 0.03
    if inputs.wacc <= term_g:
        status_notes.append(f"WACC ({inputs.wacc:.1%}) must exceed terminal growth rate ({term_g:.1%}).")
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=status_notes,
        )

    years = inputs.projection_years or 5
    fcf_projected = project_free_cash_flows(inputs.fcffree, inputs.growth_rate, years=years)
    if not fcf_projected:
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=["Failed to project cash flows."],
        )

    # Discounted cash flows
    pv_fcfs: dict[str, float] = {}
    sum_pv_fcfs = 0.0
    for idx, fcf in enumerate(fcf_projected, start=1):
        discount_factor = (1.0 + inputs.wacc) ** idx
        pv = fcf / discount_factor
        pv_fcfs[f"year_{idx}"] = float(pv)
        sum_pv_fcfs += pv

    # Terminal value
    tv = compute_terminal_value(fcf_projected[-1], inputs.wacc, term_g)
    pv_tv = tv / ((1.0 + inputs.wacc) ** years) if not np.isnan(tv) else np.nan

    enterprise_value = sum_pv_fcfs + (pv_tv if not np.isnan(pv_tv) else 0.0)
    net_debt = inputs.net_debt or 0.0
    equity_value = enterprise_value - net_debt

    # Per share value
    intrinsic_val: Optional[float] = None
    if inputs.shares_outstanding and inputs.shares_outstanding > 0:
        intrinsic_val = float(max(0.0, equity_value / inputs.shares_outstanding))
    else:
        status_notes.append("Shares outstanding unavailable; equity value computed in aggregate.")

    # High / Low valuation scenarios
    # Low: Growth - 1%, WACC + 1%
    low_wacc = inputs.wacc + 0.01
    low_growth = max(0.0, inputs.growth_rate - 0.01)
    low_proj = project_free_cash_flows(inputs.fcffree, low_growth, years=years)
    low_pv = sum(f / ((1.0 + low_wacc) ** i) for i, f in enumerate(low_proj, start=1))
    low_tv = compute_terminal_value(low_proj[-1], low_wacc, term_g)
    low_ev = low_pv + (low_tv / ((1.0 + low_wacc) ** years)) if not np.isnan(low_tv) else low_pv
    low_eq = max(0.0, low_ev - net_debt)
    has_shares = inputs.shares_outstanding is not None and not np.isnan(inputs.shares_outstanding) and inputs.shares_outstanding > 0
    fair_low = float(low_eq / inputs.shares_outstanding) if has_shares else None

    # High: Growth + 1%, WACC - 1%
    high_wacc = max(term_g + 0.005, inputs.wacc - 0.01)
    high_growth = inputs.growth_rate + 0.01
    high_proj = project_free_cash_flows(inputs.fcffree, high_growth, years=years)
    high_pv = sum(f / ((1.0 + high_wacc) ** i) for i, f in enumerate(high_proj, start=1))
    high_tv = compute_terminal_value(high_proj[-1], high_wacc, term_g)
    high_ev = high_pv + (high_tv / ((1.0 + high_wacc) ** years)) if not np.isnan(high_tv) else high_pv
    high_eq = max(0.0, high_ev - net_debt)
    fair_high = float(high_eq / inputs.shares_outstanding) if has_shares else None

    # Recommendation
    recommendation = "Hold"
    if intrinsic_val is not None and current_price is not None and current_price > 0:
        upside = (intrinsic_val - current_price) / current_price
        if upside > 0.15:
            recommendation = "Buy"
        elif upside < -0.15:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
    elif intrinsic_val is None:
        recommendation = "Insufficient data"

    # Sensitivity matrix
    sens_matrix = None
    if inputs.shares_outstanding and inputs.shares_outstanding > 0:
        sens_matrix = generate_sensitivity_matrix(
            base_fcf=inputs.fcffree,
            base_wacc=inputs.wacc,
            base_growth=inputs.growth_rate,
            terminal_growth=term_g,
            shares_outstanding=inputs.shares_outstanding,
            net_debt=net_debt,
        )

    return ValuationResult(
        intrinsic_value=intrinsic_val,
        fair_value_low=fair_low,
        fair_value_high=fair_high,
        recommendation=recommendation,
        dcf_interpolated=pv_fcfs,
        wacc_used=inputs.wacc,
        terminal_value=float(tv) if not np.isnan(tv) else np.nan,
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        sensitivity_table=sens_matrix,
        status_notes=status_notes,
    )


def generate_sensitivity_matrix(
    base_fcf: float,
    base_wacc: float,
    base_growth: float,
    terminal_growth: float = 0.03,
    shares_outstanding: float = 1.0,
    net_debt: float = 0.0,
    wacc_steps: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
    growth_steps: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
) -> pd.DataFrame:
    """Generate a 2D sensitivity table: WACC on rows, FCF growth rate on columns."""
    if base_fcf <= 0 or shares_outstanding <= 0:
        return pd.DataFrame()

    matrix: dict[str, list[float]] = {}
    row_labels: list[str] = []

    for w_step in wacc_steps:
        w = base_wacc + w_step
        if w <= terminal_growth or w <= 0:
            continue
        row_labels.append(f"{w:.1%}")

        for g_step in growth_steps:
            g = max(0.0, base_growth + g_step)
            col_name = f"g={g:.1%}"
            if col_name not in matrix:
                matrix[col_name] = []

            proj = project_free_cash_flows(base_fcf, g, years=5)
            pv_sum = sum(f / ((1.0 + w) ** i) for i, f in enumerate(proj, start=1))
            tv = compute_terminal_value(proj[-1], w, terminal_growth)
            pv_tv = tv / ((1.0 + w) ** 5) if not np.isnan(tv) else 0.0
            ev = pv_sum + pv_tv
            eq = max(0.0, ev - net_debt)
            per_share = eq / shares_outstanding
            matrix[col_name].append(round(per_share, 2))

    df = pd.DataFrame(matrix, index=row_labels)
    df.index.name = "WACC"
    return df


def generate_scenario_analysis(
    base_fcf: float,
    base_wacc: float,
    base_growth: float,
    terminal_growth: float = 0.03,
    shares_outstanding: float = 1.0,
    net_debt: float = 0.0,
    current_price: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Generate Base, Bull, and Bear scenarios."""
    scenarios_config = {
        "Bear": {
            "wacc": base_wacc + 0.015,
            "growth": max(0.0, base_growth - 0.02),
            "term_g": max(0.01, terminal_growth - 0.005),
        },
        "Base": {
            "wacc": base_wacc,
            "growth": base_growth,
            "term_g": terminal_growth,
        },
        "Bull": {
            "wacc": max(terminal_growth + 0.01, base_wacc - 0.015),
            "growth": base_growth + 0.02,
            "term_g": terminal_growth + 0.005,
        },
    }

    results: dict[str, dict[str, Any]] = {}
    for name, params in scenarios_config.items():
        inp = DCFInputs(
            fcffree=base_fcf,
            wacc=params["wacc"],
            growth_rate=params["growth"],
            terminal_growth_rate=params["term_g"],
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        val = calculate_dcf_valuation(inp, current_price=current_price)
        upside = np.nan
        if val.intrinsic_value and current_price and current_price > 0:
            upside = (val.intrinsic_value - current_price) / current_price

        results[name] = {
            "intrinsic_value": val.intrinsic_value,
            "wacc": params["wacc"],
            "growth_rate": params["growth"],
            "terminal_growth": params["term_g"],
            "upside_pct": upside,
            "recommendation": val.recommendation,
        }

    return results
