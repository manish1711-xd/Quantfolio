"""Dynamic WACC & Real-Time Cost of Capital Engine.

Calculates company-specific dynamic cost of capital:
- Dynamic risk-free rate (10Y G-Sec / Treasury proxy)
- Rolling or estimated beta
- Synthetic credit rating and credit spread based on Interest Coverage Ratio
- Capital structure weighting based on market capitalization and net/total debt
- Realized Weighted Average Cost of Capital (WACC) and Cost of Equity (Ke)
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

# Synthetic Credit Rating Table based on Damodaran's Large Cap Interest Coverage Brackets
# (Interest Coverage Min, Rating, Credit Spread bps)
SPREAD_TABLE: list[tuple[float, str, float]] = [
    (8.5, "AAA", 0.0075),   # 75 bps
    (6.5, "AA", 0.0100),    # 100 bps
    (5.5, "A+", 0.0125),    # 125 bps
    (4.25, "A", 0.0150),    # 150 bps
    (3.0, "A-", 0.0175),    # 175 bps
    (2.5, "BBB", 0.0225),   # 225 bps
    (2.0, "BB+", 0.0300),   # 300 bps
    (1.75, "BB", 0.0375),   # 375 bps
    (1.5, "B+", 0.0450),    # 450 bps
    (1.25, "B", 0.0550),    # 550 bps
    (0.8, "B-", 0.0650),    # 650 bps
    (0.5, "CCC", 0.0800),   # 800 bps
    (-np.inf, "D", 0.1050), # 1050 bps (Distressed)
]

DEFAULT_INDIAN_RISK_FREE_RATE = 0.0705  # 7.05% 10-year G-Sec benchmark
DEFAULT_EQUITY_RISK_PREMIUM = 0.0550    # 5.50% Indian ERP
DEFAULT_TAX_RATE = 0.2517              # 25.17% Base corporate tax rate in India


def get_synthetic_rating(interest_coverage: float) -> str:
    """Determine synthetic bond credit rating from interest coverage ratio."""
    if interest_coverage is None or np.isnan(interest_coverage):
        return "BBB"  # Conservative investment-grade default
    for min_cov, rating, _ in SPREAD_TABLE:
        if interest_coverage >= min_cov:
            return rating
    return "D"


def calculate_synthetic_credit_spread(interest_coverage: float) -> float:
    """Map interest coverage ratio (EBIT / Interest Expense) to debt credit spread."""
    if interest_coverage is None or np.isnan(interest_coverage):
        return 0.0225  # BBB default spread (225 bps)
    for min_cov, _, spread in SPREAD_TABLE:
        if interest_coverage >= min_cov:
            return spread
    return 0.1050


def compute_dynamic_cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
) -> float:
    """CAPM Cost of Equity: Ke = Rf + Beta * ERP."""
    rf = risk_free_rate if not np.isnan(risk_free_rate) and risk_free_rate > 0 else DEFAULT_INDIAN_RISK_FREE_RATE
    b = beta if not np.isnan(beta) and beta > 0 else 1.0
    erp = equity_risk_premium if not np.isnan(equity_risk_premium) and equity_risk_premium > 0 else DEFAULT_EQUITY_RISK_PREMIUM
    return float(rf + b * erp)


def calculate_dynamic_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> float:
    """Calculate Weighted Average Cost of Capital (WACC).

    WACC = (E / (D + E)) * Ke + (D / (D + E)) * Kd * (1 - t)
    """
    e = max(0.0, market_cap) if market_cap and not np.isnan(market_cap) else 0.0
    d = max(0.0, total_debt) if total_debt and not np.isnan(total_debt) else 0.0
    total_capital = e + d

    if total_capital <= 0.0:
        return float(cost_of_equity)

    weight_e = e / total_capital
    weight_d = d / total_capital

    t = max(0.0, min(0.5, tax_rate)) if not np.isnan(tax_rate) else DEFAULT_TAX_RATE
    after_tax_kd = cost_of_debt * (1.0 - t)

    wacc = weight_e * cost_of_equity + weight_d * after_tax_kd
    return float(max(0.03, min(0.35, wacc)))


def estimate_company_wacc(
    ticker: str,
    quote_info: dict[str, Any],
    beta: float = 1.0,
    interest_coverage: float = np.nan,
    risk_free_rate: float = DEFAULT_INDIAN_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> dict[str, Any]:
    """Orchestrate dynamic WACC estimation for a company."""
    market_cap = quote_info.get("market_cap", 0.0)
    total_debt = quote_info.get("total_debt", 0.0)

    # If currency is USD, default Rf to 10Y US Treasury ~4.2%
    curr = quote_info.get("currency", "INR")
    rf = risk_free_rate
    if curr == "USD" and math.isclose(rf, DEFAULT_INDIAN_RISK_FREE_RATE):
        rf = 0.042

    effective_beta = beta if not np.isnan(beta) and beta > 0 else 1.0
    ke = compute_dynamic_cost_of_equity(rf, effective_beta, equity_risk_premium)

    rating = get_synthetic_rating(interest_coverage)
    credit_spread = calculate_synthetic_credit_spread(interest_coverage)
    kd = rf + credit_spread

    wacc = calculate_dynamic_wacc(
        market_cap=market_cap,
        total_debt=total_debt,
        cost_of_equity=ke,
        cost_of_debt=kd,
        tax_rate=tax_rate,
    )

    tot_cap = (market_cap or 0.0) + (total_debt or 0.0)
    we = (market_cap / tot_cap) if tot_cap > 0 else 1.0
    wd = (total_debt / tot_cap) if tot_cap > 0 else 0.0

    return {
        "ticker": ticker,
        "wacc": round(wacc, 4),
        "cost_of_equity": round(ke, 4),
        "cost_of_debt": round(kd, 4),
        "credit_spread": round(credit_spread, 4),
        "synthetic_rating": rating,
        "risk_free_rate": round(rf, 4),
        "beta": round(effective_beta, 3),
        "equity_risk_premium": round(equity_risk_premium, 4),
        "weight_equity": round(we, 4),
        "weight_debt": round(wd, 4),
        "tax_rate": round(tax_rate, 4),
    }
