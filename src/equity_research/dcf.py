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


# ---------------------------------------------------------------------------
# Constants for DCF Fixes and Institutional Accuracy (11 Requirements)
# ---------------------------------------------------------------------------
#: Conservative defaults (Requirement 10)
DEFAULT_CONSERVATIVE_WACC: float = 0.10  # 10%
DEFAULT_CONSERVATIVE_TERMINAL_GROWTH: float = 0.03  # 3%

#: Realistic WACC range (Requirement 5)
WACC_MIN_REALISTIC: float = 0.07  # 7.0%
WACC_MAX_REALISTIC: float = 0.15  # 15.0%

#: Terminal growth spread and caps (Requirements 2, 3, 4)
MIN_TERMINAL_SPREAD: float = 0.02  # g must be <= WACC - 2% (2% to 4% spread)
TERMINAL_GROWTH_MIN: float = 0.02  # 2.0% minimum floor
TERMINAL_GROWTH_MAX_MATURE: float = 0.06  # 6.0% maximum for mature companies
TERMINAL_GROWTH_MAX_HIGH_GROWTH: float = 0.08  # 8.0% maximum for high-growth companies
MATURE_GROWTH_THRESHOLD: float = 0.10  # Companies with growth rate <= 10% are mature

#: Terminal value contribution cap (Requirement 6)
MAX_TERMINAL_VALUE_CONTRIBUTION: float = 0.80  # Max 80% of Enterprise Value

#: Cross-model difference thresholds (Requirement 8)
CROSS_MODEL_WARN_THRESHOLD: float = 0.25  # > 25%: Warning
CROSS_MODEL_MAJOR_WARN_THRESHOLD: float = 0.50  # > 50%: Major Warning
CROSS_MODEL_REJECT_THRESHOLD: float = 0.75  # > 75%: Reject assumptions and recalculate

#: Unrealistic upside threshold (Requirement 11)
UNREALISTIC_UPSIDE_THRESHOLD: float = 2.00  # > 200% upside


def validate_dcf_assumptions(
    inputs: DCFInputs,
    is_mature: Optional[bool] = None,
) -> tuple[bool, list[str], DCFInputs]:
    """Validate all DCF assumptions according to institutional standards (Requirements 1-5, 10).

    Returns:
        (is_valid, notices, adjusted_inputs)
    """
    notices: list[str] = []
    is_valid = True

    # Check Base FCF
    if inputs.fcffree is None or np.isnan(inputs.fcffree) or inputs.fcffree <= 0:
        notices.append("Base Free Cash Flow is negative, non-positive or unavailable; DCF requires positive cash flows.")
        is_valid = False

    # Validate WACC (Requirement 5 & 10)
    wacc = inputs.wacc
    if wacc is None or np.isnan(wacc) or wacc <= 0:
        wacc = DEFAULT_CONSERVATIVE_WACC
        notices.append(f"WACC was missing or non-positive; defaulted to conservative {DEFAULT_CONSERVATIVE_WACC:.1%}.")
    else:
        orig_wacc = float(wacc)
        if orig_wacc < WACC_MIN_REALISTIC or orig_wacc > WACC_MAX_REALISTIC:
            notices.append(
                f"WACC ({orig_wacc:.1%}) is outside the realistic range of 7%–15% (0.07–0.15)."
            )
            # Bound WACC to realistic range
            wacc = max(WACC_MIN_REALISTIC, min(WACC_MAX_REALISTIC, orig_wacc))

    # Determine company growth classification (Requirement 4)
    growth_rate = inputs.growth_rate if inputs.growth_rate is not None and not np.isnan(inputs.growth_rate) else 0.08
    if is_mature is None:
        is_mature = growth_rate <= MATURE_GROWTH_THRESHOLD

    # Validate and cap terminal growth rate (Requirements 2, 3, 4, 10)
    term_g = inputs.terminal_growth_rate
    if term_g is None or np.isnan(term_g):
        term_g = DEFAULT_CONSERVATIVE_TERMINAL_GROWTH
        notices.append(f"Terminal growth rate was missing; defaulted to conservative {DEFAULT_CONSERVATIVE_TERMINAL_GROWTH:.1%}.")
    else:
        orig_g = float(term_g)
        term_g = orig_g

        # Enforce terminal growth < WACC checks (Requirements 2 & 3)
        if orig_g >= wacc:
            notices.append(
                f"Terminal growth rate ({orig_g:.1%}) was >= WACC ({wacc:.1%}); automatically adjusted to realistic level below WACC."
            )
        elif (wacc - orig_g) < MIN_TERMINAL_SPREAD:
            notices.append(
                f"Terminal growth rate ({orig_g:.1%}) must be less than WACC ({wacc:.1%}) by at least 2%–4% spread; auto-adjusted to maintain minimum spread."
            )

        # Cap based on company maturity (Requirement 4)
        if is_mature:
            if orig_g > TERMINAL_GROWTH_MAX_MATURE:
                notices.append(
                    f"Terminal growth rate ({orig_g:.1%}) capped at 6% (0.06) for mature company (growth rate <= 10%)."
                )
                term_g = TERMINAL_GROWTH_MAX_MATURE
            elif orig_g < TERMINAL_GROWTH_MIN:
                notices.append(f"Terminal growth rate ({orig_g:.1%}) floored at minimum 2% (0.02) for mature company.")
                term_g = TERMINAL_GROWTH_MIN
        else:
            if orig_g > TERMINAL_GROWTH_MAX_HIGH_GROWTH:
                notices.append(
                    f"Terminal growth rate ({orig_g:.1%}) capped at 8% (0.08) for high-growth company."
                )
                term_g = TERMINAL_GROWTH_MAX_HIGH_GROWTH
            elif orig_g < TERMINAL_GROWTH_MIN:
                term_g = TERMINAL_GROWTH_MIN

        # Ensure spread after maturity cap
        if term_g > (wacc - MIN_TERMINAL_SPREAD):
            term_g = max(TERMINAL_GROWTH_MIN, wacc - MIN_TERMINAL_SPREAD)

    adjusted_inputs = inputs.copy_with(
        wacc=float(wacc),
        growth_rate=float(growth_rate),
        terminal_growth_rate=float(term_g),
    )
    return is_valid, notices, adjusted_inputs


def calculate_terminal_value_contribution(terminal_val: float, enterprise_val: float) -> float:
    """Calculate terminal value contribution percentage = PV(TV) / Enterprise Value."""
    if enterprise_val is None or np.isnan(enterprise_val) or enterprise_val <= 0:
        return 0.0
    if terminal_val is None or np.isnan(terminal_val):
        return 0.0
    return float(terminal_val / enterprise_val)


def apply_terminal_value_cap(
    terminal_val: float,
    enterprise_val: float,
    max_share: float = MAX_TERMINAL_VALUE_CONTRIBUTION,
) -> float:
    """Cap terminal value so that terminal_val / enterprise_val <= max_share (Requirement 6)."""
    if enterprise_val is None or np.isnan(enterprise_val) or enterprise_val <= 0:
        return terminal_val
    if terminal_val is None or np.isnan(terminal_val):
        return terminal_val
    current_share = terminal_val / enterprise_val
    if current_share > max_share:
        return float(enterprise_val * max_share)
    return float(terminal_val)


def calculate_cross_model_gap(fcff_value: float, gordon_value: float) -> float:
    """Calculate percentage gap between FCFF and Gordon Growth intrinsic values."""
    if fcff_value is None or gordon_value is None or np.isnan(fcff_value) or np.isnan(gordon_value):
        return np.nan
    if fcff_value <= 0 and gordon_value <= 0:
        return np.nan
    diff = abs(fcff_value - gordon_value)
    gap_fcff = diff / fcff_value if fcff_value > 0 else 0.0
    gap_gordon = diff / gordon_value if gordon_value > 0 else 0.0
    return float(max(gap_fcff, gap_gordon))


def classify_cross_model_gap(gap: float) -> str:
    """Classify cross-model gap into alignment tiers (Requirement 8):
    - gap <= 25%: OK
    - 25% < gap <= 50%: Warning
    - 50% < gap <= 75%: Major Warning
    - gap > 75%: Reject
    """
    if gap is None or np.isnan(gap):
        return "Unknown"
    if gap > CROSS_MODEL_REJECT_THRESHOLD:
        return "Reject"
    if gap > CROSS_MODEL_MAJOR_WARN_THRESHOLD:
        return "Major Warning"
    if gap > CROSS_MODEL_WARN_THRESHOLD:
        return "Warning"
    return "OK"


def calculate_valuation_confidence(
    wacc: float,
    terminal_growth: float,
    tv_contribution: float,
    cross_model_gap: float,
    upside: float,
) -> float:
    """Calculate an institutional valuation confidence score from 0.0 to 1.0 (Requirement 9)."""
    score = 0.95

    # WACC penalties
    if wacc < WACC_MIN_REALISTIC or wacc > WACC_MAX_REALISTIC:
        score -= 0.20

    # Spread penalty
    if (wacc - terminal_growth) < 0.03:
        score -= 0.10

    # Terminal value contribution penalties
    if tv_contribution > 0.80:
        score -= 0.20
    elif tv_contribution > 0.65:
        score -= 0.10

    # Cross-model gap penalties
    if not np.isnan(cross_model_gap):
        if cross_model_gap > CROSS_MODEL_MAJOR_WARN_THRESHOLD:
            score -= 0.25
        elif cross_model_gap > CROSS_MODEL_WARN_THRESHOLD:
            score -= 0.15

    # Unrealistic upside penalty
    if not np.isnan(upside) and upside > UNREALISTIC_UPSIDE_THRESHOLD:
        score -= 0.25

    return float(max(0.10, min(0.98, round(score, 2))))


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
    is_recalculating_after_rejection: bool = False,
) -> MultiModelValuationResult:
    """Run all 4 DCF models with institutional guardrails and cross-model validation.

    Models:
    - FCFF (Free Cash Flow to Firm): Discounted at WACC with TV capped at 80%
    - FCFE (Free Cash Flow to Equity): Discounted at Cost of Equity
    - DDM (Dividend Discount Model): Gordon Growth for dividend discounting
    - Gordon Growth: One-stage perpetuity model for FCF

    Validates assumptions (Req 1-5, 10), applies 80% TV contribution cap (Req 6),
    enforces TV formula (Req 7), checks cross-model gap (Req 8), flags upside >200% (Req 11),
    and computes Valuation Confidence Score (Req 9).
    """
    notices: list[str] = []

    # 1. Validate all assumptions before calculating valuation (Requirements 1-5, 10)
    is_valid, validation_notices, valid_inputs = validate_dcf_assumptions(inputs)
    notices.extend(validation_notices)

    if not is_valid:
        return MultiModelValuationResult(
            recommendation="Insufficient data",
            current_price=current_price or np.nan,
            notices=notices,
            wacc_applied=valid_inputs.wacc,
            terminal_growth_applied=valid_inputs.terminal_growth_rate,
        )

    wacc = valid_inputs.wacc
    term_g = valid_inputs.terminal_growth_rate
    growth_rate = valid_inputs.growth_rate
    fcffree = valid_inputs.fcffree
    shares = valid_inputs.shares_outstanding or 1.0
    net_debt = valid_inputs.net_debt or 0.0

    # Build assumptions and financials for multi-model engine
    financials_dict = {
        "revenue_latest": fcffree,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "ebitda": fcffree,
        "current_price": current_price or np.nan,
        "pe_ratio": 15.0,
    }

    assumptions, company_fin = _build_fundamentals_for_multi_model(
        financials_dict, wacc, growth_rate, term_g
    )

    # Cost of Equity for FCFE and DDM
    cost_of_equity = max(term_g + MIN_TERMINAL_SPREAD, wacc - 0.02)

    # 1. FCFF Model with Terminal Value contribution guardrail (Requirement 6 & 7)
    years = valid_inputs.projection_years or 5
    fcf_projected = project_free_cash_flows(fcffree, growth_rate, years=years)
    pv_explicit = 0.0
    if fcf_projected:
        for idx, f in enumerate(fcf_projected, start=1):
            pv_explicit += f / ((1.0 + wacc) ** idx)

    # Terminal Value: TV = FCF * (1 + g) / (WACC - g) (Requirement 7)
    final_fcf = fcf_projected[-1] if fcf_projected else fcffree * ((1.0 + growth_rate) ** years)
    tv = compute_terminal_value(final_fcf, wacc, term_g)
    pv_tv = tv / ((1.0 + wacc) ** years) if not np.isnan(tv) else 0.0

    unadjusted_ev = pv_explicit + pv_tv
    tv_contribution = calculate_terminal_value_contribution(pv_tv, unadjusted_ev)

    # Prevent terminal value from contributing more than 80% of EV (Requirement 6)
    if tv_contribution > MAX_TERMINAL_VALUE_CONTRIBUTION:
        notices.append(
            f"Terminal value contributed {tv_contribution:.1%} of Enterprise Value; capped at 80.0% to prevent terminal dominance."
        )
        pv_tv = apply_terminal_value_cap(pv_tv, unadjusted_ev, max_share=MAX_TERMINAL_VALUE_CONTRIBUTION)
        adjusted_ev = pv_explicit + pv_tv
        tv_contribution = calculate_terminal_value_contribution(pv_tv, adjusted_ev)
    else:
        adjusted_ev = unadjusted_ev

    fcff_equity_val = max(0.0, adjusted_ev - net_debt)
    fcff_per_share = fcff_equity_val / shares if shares > 0 else np.nan

    # 2. Run FCFE
    fcfe_result = run_fcfe(assumptions, company_fin, cost_of_equity=cost_of_equity)

    # 3. Run DDM
    dividend = dividend_per_share if dividend_per_share is not None else (company_fin.base_revenue / company_fin.shares * 0.3)
    ddm_result = run_ddm(
        current_dividend=dividend,
        growth_rate=term_g,
        cost_of_equity=cost_of_equity,
        current_price=current_price or np.nan,
    )

    # 4. Run Gordon Growth Model (Requirement 7)
    next_fcf = fcffree * (1.0 + growth_rate)
    gordon_ev = run_gordon(
        base_cash_flow=next_fcf,
        growth_rate=term_g,
        discount_rate=wacc,
    )
    gordon_per_share = np.nan
    if _ok(gordon_ev) and shares > 0:
        gordon_equity = max(0.0, gordon_ev - net_debt)
        gordon_per_share = gordon_equity / shares

    # 5. Sanity Check Cross-Model Difference between FCFF and Gordon Growth (Requirement 8)
    cross_gap = calculate_cross_model_gap(fcff_per_share, gordon_per_share)
    gap_tier = classify_cross_model_gap(cross_gap)
    assumptions_rejected = False

    if gap_tier == "Warning":
        notices.append(
            f"⚠️ Cross-model difference of {cross_gap:.1%} between FCFF ({fcff_per_share:,.2f}) and Gordon Growth ({gordon_per_share:,.2f}) exceeds 25% threshold."
        )
    elif gap_tier == "Major Warning":
        notices.append(
            f"🚨 Major Warning: Cross-model difference of {cross_gap:.1%} between FCFF ({fcff_per_share:,.2f}) and Gordon Growth ({gordon_per_share:,.2f}) exceeds 50% threshold. Terminal assumptions require review."
        )
    elif gap_tier == "Reject" and not is_recalculating_after_rejection:
        assumptions_rejected = True
        notices.append(
            f"🛑 Rejected assumptions: Cross-model difference of {cross_gap:.1%} exceeds 75% tolerance. Recalculating using conservative defaults (WACC={DEFAULT_CONSERVATIVE_WACC:.1%}, Terminal Growth={DEFAULT_CONSERVATIVE_TERMINAL_GROWTH:.1%})."
        )
        # Recalculate using conservative defaults (Requirement 8 & 10)
        conservative_inputs = valid_inputs.copy_with(
            wacc=DEFAULT_CONSERVATIVE_WACC,
            terminal_growth_rate=DEFAULT_CONSERVATIVE_TERMINAL_GROWTH,
        )
        recalculated_result = calculate_multi_model_valuation(
            conservative_inputs,
            current_price=current_price,
            dividend_per_share=dividend_per_share,
            is_recalculating_after_rejection=True,
        )
        merged_notices = list(dict.fromkeys(notices + recalculated_result.notices))
        return MultiModelValuationResult(
            fcff_value=recalculated_result.fcff_value,
            fcff_intrinsic=recalculated_result.fcff_intrinsic,
            fcff_fair_low=recalculated_result.fcff_fair_low,
            fcff_fair_high=recalculated_result.fcff_fair_high,
            fcfe_value=recalculated_result.fcfe_value,
            fcfe_intrinsic=recalculated_result.fcfe_intrinsic,
            fcfe_fair_low=recalculated_result.fcfe_fair_low,
            fcfe_fair_high=recalculated_result.fcfe_fair_high,
            ddm_value=recalculated_result.ddm_value,
            gordon_value=recalculated_result.gordon_value,
            blended_intrinsic=recalculated_result.blended_intrinsic,
            blended_fair_low=recalculated_result.blended_fair_low,
            blended_fair_high=recalculated_result.blended_fair_high,
            premium_to_price=recalculated_result.premium_to_price,
            intrinsic_values=recalculated_result.intrinsic_values,
            model_names=recalculated_result.model_names,
            recommendation=recalculated_result.recommendation,
            current_price=current_price or np.nan,
            notices=merged_notices,
            tv_contribution_pct=recalculated_result.tv_contribution_pct,
            cross_model_gap_pct=recalculated_result.cross_model_gap_pct,
            confidence_score=recalculated_result.confidence_score,
            wacc_applied=recalculated_result.wacc_applied,
            terminal_growth_applied=recalculated_result.terminal_growth_applied,
            assumptions_rejected=True,
        )

    # Collect usable model values for blended intrinsic value
    intrinsic_values: dict[str, float] = {}
    model_names: list[str] = []

    if _ok(fcff_per_share) and fcff_per_share > 0:
        intrinsic_values["FCFF"] = fcff_per_share
        model_names.append("FCFF")
    if fcfe_result.ok and not np.isnan(fcfe_result.value_per_share):
        intrinsic_values["FCFE"] = fcfe_result.value_per_share
        model_names.append("FCFE")
    if ddm_result.ok and not np.isnan(ddm_result.value_per_share):
        intrinsic_values["DDM"] = ddm_result.value_per_share
        model_names.append("DDM")
    if _ok(gordon_per_share) and gordon_per_share > 0:
        intrinsic_values["Gordon Growth"] = gordon_per_share
        model_names.append("Gordon Growth")

    valid_values = list(intrinsic_values.values())
    blended_intrinsic = float(np.mean(valid_values)) if valid_values else np.nan
    blended_fair_low = fcff_per_share * 0.85 if _ok(fcff_per_share) else np.nan
    blended_fair_high = fcff_per_share * 1.15 if _ok(fcff_per_share) else np.nan

    # Implied premium / upside
    premium = np.nan
    if current_price and current_price > 0 and not np.isnan(blended_intrinsic):
        premium = (blended_intrinsic - current_price) / current_price

    # Requirement 11: If upside > 200%, flag as 'Potentially Unrealistic Assumptions'
    if not np.isnan(premium) and premium > UNREALISTIC_UPSIDE_THRESHOLD:
        notices.append(
            f"⚠️ Potentially Unrealistic Assumptions: Implied upside of {premium:.1%} exceeds 200% threshold. Verify growth and cash flow estimates."
        )

    # Recommendation
    recommendation = "Hold"
    if not np.isnan(blended_intrinsic) and blended_intrinsic > 0:
        if current_price and current_price > 0:
            if premium > 0.15:
                recommendation = "Buy"
            elif premium < -0.15:
                recommendation = "Sell"
            else:
                recommendation = "Hold"

    # Valuation Confidence Score (Requirement 9)
    confidence = calculate_valuation_confidence(
        wacc=wacc,
        terminal_growth=term_g,
        tv_contribution=tv_contribution,
        cross_model_gap=cross_gap if not np.isnan(cross_gap) else 0.0,
        upside=premium if not np.isnan(premium) else 0.0,
    )

    return MultiModelValuationResult(
        fcff_value=fcff_per_share,
        fcff_intrinsic=fcff_per_share,
        fcff_fair_low=blended_fair_low,
        fcff_fair_high=blended_fair_high,
        fcfe_value=fcfe_result.value_per_share if fcfe_result.ok else np.nan,
        fcfe_intrinsic=fcfe_result.value_per_share if fcfe_result.ok else np.nan,
        fcfe_fair_low=fcfe_result.value_per_share * 0.85 if fcfe_result.ok else np.nan,
        fcfe_fair_high=fcfe_result.value_per_share * 1.15 if fcfe_result.ok else np.nan,
        ddm_value=ddm_result.value_per_share if ddm_result.ok else np.nan,
        gordon_value=gordon_per_share,
        blended_intrinsic=blended_intrinsic,
        blended_fair_low=blended_fair_low,
        blended_fair_high=blended_fair_high,
        premium_to_price=premium,
        intrinsic_values=intrinsic_values,
        model_names=model_names,
        recommendation=recommendation,
        current_price=current_price or np.nan,
        notices=list(dict.fromkeys(notices)),
        tv_contribution_pct=tv_contribution,
        cross_model_gap_pct=cross_gap,
        confidence_score=confidence,
        wacc_applied=wacc,
        terminal_growth_applied=term_g,
        assumptions_rejected=assumptions_rejected,
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

    Applies institutional validation, terminal value cap (80%), and confidence scoring.
    """
    is_valid, validation_notices, valid_inputs = validate_dcf_assumptions(inputs)
    status_notes = list(validation_notices)

    if not is_valid:
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=status_notes,
        )

    wacc = valid_inputs.wacc
    term_g = valid_inputs.terminal_growth_rate
    growth = valid_inputs.growth_rate
    fcf = valid_inputs.fcffree
    shares = valid_inputs.shares_outstanding or 1.0
    net_debt = valid_inputs.net_debt or 0.0
    years = valid_inputs.projection_years or 5

    fcf_projected = project_free_cash_flows(fcf, growth, years=years)
    if not fcf_projected:
        status_notes.append("Failed to project cash flows.")
        return ValuationResult(
            intrinsic_value=None,
            fair_value_low=None,
            fair_value_high=None,
            recommendation="Insufficient data",
            status_notes=status_notes,
        )

    # Discounted cash flows
    pv_fcfs: dict[str, float] = {}
    sum_pv_fcfs = 0.0
    for idx, cf in enumerate(fcf_projected, start=1):
        pv = cf / ((1.0 + wacc) ** idx)
        pv_fcfs[f"year_{idx}"] = float(pv)
        sum_pv_fcfs += pv

    # Terminal value: TV = FCF * (1 + g) / (WACC - g) (Req 7)
    tv = compute_terminal_value(fcf_projected[-1], wacc, term_g)
    pv_tv = tv / ((1.0 + wacc) ** years) if not np.isnan(tv) else 0.0

    unadjusted_ev = sum_pv_fcfs + pv_tv
    tv_contribution = calculate_terminal_value_contribution(pv_tv, unadjusted_ev)

    # Prevent terminal value from contributing more than 80% of EV (Req 6)
    if tv_contribution > MAX_TERMINAL_VALUE_CONTRIBUTION:
        status_notes.append(
            f"Terminal value contributed {tv_contribution:.1%} of Enterprise Value; capped at 80.0% to prevent terminal dominance."
        )
        pv_tv = apply_terminal_value_cap(pv_tv, unadjusted_ev, max_share=MAX_TERMINAL_VALUE_CONTRIBUTION)
        enterprise_value = sum_pv_fcfs + pv_tv
        tv_contribution = calculate_terminal_value_contribution(pv_tv, enterprise_value)
    else:
        enterprise_value = unadjusted_ev

    equity_value = max(0.0, enterprise_value - net_debt)
    intrinsic_val: Optional[float] = float(equity_value / shares) if shares > 0 else None

    # Fair value low / high
    fair_low = float(intrinsic_val * 0.85) if intrinsic_val is not None else None
    fair_high = float(intrinsic_val * 1.15) if intrinsic_val is not None else None

    # Recommendation & upside check (Req 11)
    recommendation = "Hold"
    upside = np.nan
    if intrinsic_val is not None and current_price is not None and current_price > 0:
        upside = (intrinsic_val - current_price) / current_price
        if upside > 0.15:
            recommendation = "Buy"
        elif upside < -0.15:
            recommendation = "Sell"
        if upside > UNREALISTIC_UPSIDE_THRESHOLD:
            status_notes.append(
                f"⚠️ Potentially Unrealistic Assumptions: Implied upside of {upside:.1%} exceeds 200% threshold."
            )

    # Confidence score (Req 9)
    confidence = calculate_valuation_confidence(
        wacc=wacc,
        terminal_growth=term_g,
        tv_contribution=tv_contribution,
        cross_model_gap=0.0,
        upside=upside if not np.isnan(upside) else 0.0,
    )

    sens_matrix = None
    if shares > 0:
        sens_matrix = generate_sensitivity_matrix(
            base_fcf=fcf,
            base_wacc=wacc,
            base_growth=growth,
            terminal_growth=term_g,
            shares_outstanding=shares,
            net_debt=net_debt,
        )

    return ValuationResult(
        intrinsic_value=intrinsic_val,
        fair_value_low=fair_low,
        fair_value_high=fair_high,
        recommendation=recommendation,
        dcf_interpolated=pv_fcfs,
        wacc_used=wacc,
        terminal_value=float(tv) if not np.isnan(tv) else np.nan,
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        sensitivity_table=sens_matrix,
        status_notes=status_notes,
        tv_contribution_pct=tv_contribution,
        confidence_score=confidence,
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