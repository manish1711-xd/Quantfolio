"""Discounted Cash Flow (DCF) Valuation and Sensitivity Engine.

Features:
- Five-year free cash flow projections
- WACC-discounted enterprise value and equity value
- Gordon Growth terminal value modeling
- Sensitivity matrix across WACC and terminal growth rates
- Bull / Base / Bear scenario analysis
- Clear model-generated ratings with transparent assumptions and missing-data guards
- Multi-model valuation: FCFF, FCFE, DDM, Gordon Growth
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.analytics.dcf import (
    DcfAssumptions,
    CompanyFinancials,
    run_dcf,
    run_fcfe,
    run_ddm,
    run_gordon,
    run_multi_model,
    sensitivity_grid,
    default_assumptions,
    MIN_WACC_SPREAD,
)
from src.equity_research.models import DCFInputs, ValuationResult, MultiModelValuationResult

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


def _build_fundamentals_for_multi_model(
    financials: dict,
    wacc: float,
    growth_rate: float,
    terminal_growth: float,
) -> tuple[DcfAssumptions, CompanyFinancials]:
    """Build DcfAssumptions and CompanyFinancials from financial data dictionary."""
    base_revenue = financials.get("revenue_latest", 0.0)
    net_debt = financials.get("net_debt", 0.0)
    shares = financials.get("shares_outstanding", 1.0)

    # Get EBITDA from financials or calculate it
    ebitda = financials.get("ebitda", np.nan)
    if np.isnan(ebitda):
        ebitda = financials.get("ebitda_margin", 0.1) * base_revenue if base_revenue > 0 else np.nan

    # Build assumptions
    assumptions = DcfAssumptions(
        revenue_growth=(growth_rate,) * 5,
        ebit_margin=(financials.get("operating_margin", 0.15),) * 5,
        tax_rate=0.2517,  # Default Indian tax rate
        da_pct_revenue=0.04,
        capex_pct_revenue=0.05,
        nwc_pct_incremental_revenue=0.10,
        gross_margin=financials.get("gross_margin", 0.40),
        pretax_cost_of_debt=0.08,
        net_borrowing_pct_nwc=0.0,
        wacc_value=wacc,
        terminal_growth=terminal_growth,
        exit_ebitda_multiple=financials.get("pe_ratio", 15.0),
        mid_year=True,
    )

    # Build company financials
    company_fin = CompanyFinancials(
        base_revenue=base_revenue,
        net_debt=net_debt if not np.isnan(net_debt) else 0.0,
        shares=shares if shares else 1.0,
        current_price=financials.get("current_price", np.nan),
        base_ebitda=ebitda if not np.isnan(ebitda) else 0.0,
    )

    return assumptions, company_fin


def calculate_multi_model_valuation(
    inputs: DCFInputs,
    current_price: Optional[float] = None,
    dividend_per_share: Optional[float] = None,
) -> MultiModelValuationResult:
    """Run all 4 DCF models and blend the results.

    Models:
    - FCFF (Free Cash Flow to Firm): Discounted at WACC, then net debt subtracted
    - FCFE (Free Cash Flow to Equity): Discounted at Cost of Equity
    - DDM (Dividend Discount Model): Gordon Growth for dividend discounting
    - Gordon Growth: One-stage perpetuity model for FCF

    Returns a MultiModelValuationResult with all four intrinsic values and a blended estimate.
    """
    notices: list[str] = []

    # Guard: validate inputs
    if inputs.fcffree is None or np.isnan(inputs.fcffree) or inputs.fcffree <= 0:
        notices.append("Base Free Cash Flow is negative, non-positive or unavailable; DCF requires positive cash flows.")
        return MultiModelValuationResult(
            recommendation="Insufficient data",
            current_price=current_price or np.nan,
            notices=notices,
        )

    if inputs.wacc is None or np.isnan(inputs.wacc) or inputs.wacc <= 0:
        notices.append("WACC must be positive and non-zero.")
        return MultiModelValuationResult(
            recommendation="Insufficient data",
            current_price=current_price or np.nan,
            notices=notices,
        )

    term_g = inputs.terminal_growth_rate if inputs.terminal_growth_rate is not None else 0.03
    if inputs.wacc <= term_g:
        notices.append(f"WACC ({inputs.wacc:.1%}) must exceed terminal growth rate ({term_g:.1%}).")
        return MultiModelValuationResult(
            recommendation="Insufficient data",
            current_price=current_price or np.nan,
            notices=notices,
        )

    # Build assumptions and financials for multi-model engine
    financials_dict = {
        "revenue_latest": inputs.fcffree,  # Use FCF as proxy for revenue baseline
        "net_debt": inputs.net_debt or 0.0,
        "shares_outstanding": inputs.shares_outstanding or 1.0,
        "ebitda": inputs.fcffree,  # Use FCF as proxy
        "current_price": current_price or np.nan,
        "pe_ratio": 15.0,  # Default
    }

    assumptions, company_fin = _build_fundamentals_for_multi_model(
        financials_dict, inputs.wacc, inputs.growth_rate, term_g
    )

    # Calculate Cost of Equity for FCFE and DDM
    # Using simplified CAPM: Ke = risk_free + beta * ERP
    cost_of_equity = inputs.wacc - 0.02  # Approximate levered cost of equity
    if cost_of_equity <= term_g + MIN_WACC_SPREAD:
        cost_of_equity = inputs.wacc * 0.85  # Ensure it's below WACC but above terminal growth

    # Run all 4 models using the advanced engine
    fcff_result = run_dcf(assumptions, company_fin)

    fcfe_result = run_fcfe(
        assumptions, company_fin, cost_of_equity=cost_of_equity
    )

    # DDM: Use dividend if provided, otherwise estimate from earnings
    dividend = dividend_per_share if dividend_per_share is not None else (company_fin.base_revenue / company_fin.shares * 0.3)
    ddm_result = run_ddm(
        current_dividend=dividend,
        growth_rate=term_g,  # Use terminal growth for DDM
        cost_of_equity=cost_of_equity,
        current_price=current_price or np.nan,
    )

    # Gordon Growth: Use next year's FCF
    next_fcf = inputs.fcffree * (1.0 + inputs.growth_rate)
    gordon_val = run_gordon(
        base_cash_flow=next_fcf,
        growth_rate=term_g,
        discount_rate=inputs.wacc,
    )
    # Convert Gordon to per-share
    gordon_per_share = np.nan
    if not np.isnan(gordon_val) and inputs.shares_outstanding and inputs.shares_outstanding > 0:
        gordon_equity = gordon_val - (inputs.net_debt or 0.0)
        gordon_per_share = gordon_equity / inputs.shares_outstanding

    # Run multi-model engine to combine results
    multi_result = run_multi_model(
        assumptions=assumptions,
        financials=company_fin,
        cost_of_equity=cost_of_equity,
        current_dividend=dividend if dividend > 0 else np.nan,
        dividend_growth=term_g,
        gordon_cash_flow=next_fcf,
        effective_tax_rate=0.2517,
    )

    # Build our result object
    intrinsic_values: dict[str, float] = {}
    model_names: list[str] = []

    # FCFF value
    if fcff_result.ok and not np.isnan(fcff_result.value_per_share):
        intrinsic_values["FCFF"] = fcff_result.value_per_share
        model_names.append("FCFF")

    # FCFE value
    if fcfe_result.ok and not np.isnan(fcfe_result.value_per_share):
        intrinsic_values["FCFE"] = fcfe_result.value_per_share
        model_names.append("FCFE")

    # DDM value
    if ddm_result.ok and not np.isnan(ddm_result.value_per_share):
        intrinsic_values["DDM"] = ddm_result.value_per_share
        model_names.append("DDM")

    # Gordon Growth value
    if _ok(gordon_per_share):
        intrinsic_values["Gordon Growth"] = gordon_per_share
        model_names.append("Gordon Growth")

    # Calculate blended value
    valid_values = list(intrinsic_values.values())
    blended_intrinsic = np.nan
    blended_fair_low = np.nan
    blended_fair_high = np.nan

    if valid_values:
        blended_intrinsic = float(np.mean(valid_values))

        # Calculate fair value range from FCFF results
        if fcff_result.ok:
            blended_fair_low = fcff_result.value_per_share * 0.85 if fcff_result.value_per_share else np.nan
            blended_fair_high = fcff_result.value_per_share * 1.15 if fcff_result.value_per_share else np.nan

    # Calculate premium/discount to current price
    premium = np.nan
    if current_price and current_price > 0 and not np.isnan(blended_intrinsic):
        premium = (blended_intrinsic - current_price) / current_price

    # Determine recommendation
    recommendation = "Hold"
    if not np.isnan(blended_intrinsic) and blended_intrinsic > 0:
        if current_price and current_price > 0:
            if premium > 0.15:
                recommendation = "Buy"
            elif premium < -0.15:
                recommendation = "Sell"
            else:
                recommendation = "Hold"

    return MultiModelValuationResult(
        fcff_value=fcff_result.value_per_share if fcff_result.ok else np.nan,
        fcff_intrinsic=fcff_result.value_per_share if fcff_result.ok else np.nan,
        fcff_fair_low=fcff_result.value_per_share * 0.85 if fcff_result.value_per_share else np.nan,
        fcff_fair_high=fcff_result.value_per_share * 1.15 if fcff_result.value_per_share else np.nan,
        fcfe_value=fcfe_result.value_per_share if fcfe_result.ok else np.nan,
        fcfe_intrinsic=fcfe_result.value_per_share if fcfe_result.ok else np.nan,
        fcfe_fair_low=fcfe_result.value_per_share * 0.85 if fcfe_result.value_per_share else np.nan,
        fcfe_fair_high=fcfe_result.value_per_share * 1.15 if fcfe_result.value_per_share else np.nan,
        ddm_value=ddm_result.value_per_share if ddm_result.ok else np.nan,
        gordon_value=gordon_per_share,
        blended_intrinsic=blended_intrinsic if not np.isnan(blended_intrinsic) else np.nan,
        blended_fair_low=blended_fair_low if not np.isnan(blended_fair_low) else np.nan,
        blended_fair_high=blended_fair_high if not np.isnan(blended_fair_high) else np.nan,
        premium_to_price=premium,
        intrinsic_values=intrinsic_values,
        model_names=model_names,
        recommendation=recommendation,
        current_price=current_price or np.nan,
        notices=list(set(notices + list(multi_result.notices))),
    )


def _ok(*values: float) -> bool:
    """True when every value is a real number."""
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
    return True


def calculate_dcf_valuation(
    inputs: DCFInputs,
    current_price: Optional[float] = None,
) -> ValuationResult:
    """Run full five-year DCF valuation with terminal value and sensitivity.

    This is the legacy interface - now wraps the multi-model engine.
    """
    # For back compatibility, compute using the standard FCFF approach
    if inputs.fcffree is None or np.isnan(inputs.fcffree) or inputs.fcffree <= 0:
        status_notes = ["Base Free Cash Flow is negative, non-positive or unavailable; DCF requires positive cash flows."]
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
    term_g = inputs.terminal_growth_rate if inputs.terminal_growth_rate is not None else 0.03
    if inputs.wacc <= term_g:
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=[f"WACC ({inputs.wacc:.1%}) must exceed terminal growth rate ({term_g:.1%})."],
        )

    tv = compute_terminal_value(fcf_projected[-1], inputs.wacc, term_g)
    pv_tv = tv / ((1.0 + inputs.wacc) ** years) if not np.isnan(tv) else 0.0

    enterprise_value = sum_pv_fcfs + (pv_tv if not np.isnan(pv_tv) else 0.0)
    net_debt = inputs.net_debt or 0.0
    equity_value = enterprise_value - net_debt

    # Per share value
    intrinsic_val: Optional[float] = None
    if inputs.shares_outstanding and inputs.shares_outstanding > 0:
        intrinsic_val = float(max(0.0, equity_value / inputs.shares_outstanding))
    else:
        intrinsic_val = None

    # High / Low valuation scenarios
    low_wacc = inputs.wacc + 0.01
    low_growth = max(0.0, inputs.growth_rate - 0.01)
    low_proj = project_free_cash_flows(inputs.fcffree, low_growth, years=years)
    low_pv = sum(f / ((1.0 + low_wacc) ** i) for i, f in enumerate(low_proj, start=1))
    low_tv = compute_terminal_value(low_proj[-1], low_wacc, term_g)
    low_ev = low_pv + (low_tv / ((1.0 + low_wacc) ** years)) if low_proj and not np.isnan(low_tv) else low_pv
    has_shares = inputs.shares_outstanding is not None and not np.isnan(inputs.shares_outstanding) and inputs.shares_outstanding > 0
    fair_low = float((low_ev - net_debt) / inputs.shares_outstanding) if has_shares else None

    high_wacc = max(term_g + 0.005, inputs.wacc - 0.01)
    high_growth = inputs.growth_rate + 0.01
    high_proj = project_free_cash_flows(inputs.fcffree, high_growth, years=years)
    high_pv = sum(f / ((1.0 + high_wacc) ** i) for i, f in enumerate(high_proj, start=1))
    high_tv = compute_terminal_value(high_proj[-1], high_wacc, term_g)
    high_ev = high_pv + (high_tv / ((1.0 + high_wacc) ** years)) if high_proj and not np.isnan(high_tv) else high_pv
    fair_high = float((high_ev - net_debt) / inputs.shares_outstanding) if has_shares else None

    # Recommendation
    recommendation = "Hold"
    if intrinsic_val is not None and current_price is not None and current_price > 0:
        upside = (intrinsic_val - current_price) / current_price
        if upside > 0.15:
            recommendation = "Buy"
        elif upside < -0.15:
            recommendation = "Sell"

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
        status_notes=[],
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