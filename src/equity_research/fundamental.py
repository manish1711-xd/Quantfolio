"""Fundamental financial analysis: margins, ratios, earnings, debt and cash flow.

Pure calculation module: accepts raw statement DataFrames, computes standard
financial ratios, handles missing lines honestly, and returns FundamentalData.
No Streamlit imports; zero network operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.equity_research.models import FundamentalData

logger = logging.getLogger(__name__)

# Row aliases for Yahoo Finance / standard filings
REVENUE_KEYS = ["Total Revenue", "Operating Revenue", "Revenue"]
GROSS_PROFIT_KEYS = ["Gross Profit"]
OPERATING_INCOME_KEYS = ["Operating Income", "Operating Profit"]
EBIT_KEYS = ["EBIT", "Operating Income", "Normalized EBITDA"]
NET_INCOME_KEYS = [
    "Net Income Common Stockholders",
    "Net Income",
    "Net Income Applicable To Common Shares",
    "Net Income Available to Stockholders",
]
TOTAL_ASSETS_KEYS = ["Total Assets"]
EQUITY_KEYS = [
    "Stockholders Equity",
    "Total Equity",
    "Total Stockholders Equity",
    "Common Stock Equity",
]
TOTAL_DEBT_KEYS = ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]
CASH_KEYS = [
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Short Term Investments",
]
LIABILITIES_KEYS = [
    "Total Liabilities Net Minority Interest",
    "Total Liabilities",
    "Current Liabilities",
]
OPERATING_CF_KEYS = [
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
]
CAPEX_KEYS = ["Capital Expenditure", "Capital Expenditures"]
FCF_KEYS = ["Free Cash Flow"]


def _get_latest_item(df: pd.DataFrame, candidate_keys: list[str]) -> tuple[float, Optional[str]]:
    """Look up the most recent period's value for the first matching line item.
    Returns (value, matched_key). Returns (np.nan, None) if absent.
    """
    if df is None or df.empty:
        return np.nan, None

    for key in candidate_keys:
        if key in df.index:
            row = df.loc[key]
            # yfinance columns are periods (e.g. '2024-03-31'); first column is usually latest
            val = row.iloc[0] if isinstance(row, pd.Series) else row
            try:
                num = float(val)
                if not np.isnan(num):
                    return num, key
            except (ValueError, TypeError):
                continue
    return np.nan, None


def compute_margins(income_df: pd.DataFrame) -> dict[str, float]:
    """Calculate gross, operating, and net profit margins."""
    rev, _ = _get_latest_item(income_df, REVENUE_KEYS)
    gp, _ = _get_latest_item(income_df, GROSS_PROFIT_KEYS)
    op_inc, _ = _get_latest_item(income_df, OPERATING_INCOME_KEYS)
    net_inc, _ = _get_latest_item(income_df, NET_INCOME_KEYS)

    if np.isnan(rev) or rev <= 0:
        return {
            "gross_margin": np.nan,
            "operating_margin": np.nan,
            "net_margin": np.nan,
            "revenue": np.nan,
            "net_income": np.nan,
        }

    return {
        "gross_margin": float(gp / rev) if not np.isnan(gp) else np.nan,
        "operating_margin": float(op_inc / rev) if not np.isnan(op_inc) else np.nan,
        "net_margin": float(net_inc / rev) if not np.isnan(net_inc) else np.nan,
        "revenue": float(rev),
        "net_income": float(net_inc) if not np.isnan(net_inc) else np.nan,
    }


def compute_returns_and_earnings(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    shares_outstanding: Optional[float] = None,
) -> dict[str, float]:
    """Calculate ROE, ROCE, and EPS."""
    net_inc, _ = _get_latest_item(income_df, NET_INCOME_KEYS)
    ebit, _ = _get_latest_item(income_df, EBIT_KEYS)
    equity, _ = _get_latest_item(balance_df, EQUITY_KEYS)
    debt, _ = _get_latest_item(balance_df, TOTAL_DEBT_KEYS)
    assets, _ = _get_latest_item(balance_df, TOTAL_ASSETS_KEYS)

    # ROE = Net Income / Equity
    roe = np.nan
    if not np.isnan(net_inc) and not np.isnan(equity) and equity > 0:
        roe = float(net_inc / equity)

    # ROCE = EBIT / Capital Employed (Equity + Total Debt)
    roce = np.nan
    cap_employed = np.nan
    if not np.isnan(equity):
        d_val = debt if not np.isnan(debt) else 0.0
        cap_employed = equity + d_val

    if not np.isnan(ebit) and not np.isnan(cap_employed) and cap_employed > 0:
        roce = float(ebit / cap_employed)
    elif not np.isnan(ebit) and not np.isnan(assets) and assets > 0:
        # Fallback to total assets if capital employed is unavailable
        roce = float(ebit / assets)

    # EPS = Net Income / Shares Outstanding
    eps = np.nan
    if not np.isnan(net_inc) and shares_outstanding and shares_outstanding > 0:
        eps = float(net_inc / shares_outstanding)
    else:
        # Check if income statement has reported Basic EPS
        reported_eps, _ = _get_latest_item(income_df, ["Basic EPS", "Diluted EPS"])
        if not np.isnan(reported_eps):
            eps = float(reported_eps)

    return {
        "roe": roe,
        "roce": roce,
        "eps": eps,
        "equity": float(equity) if not np.isnan(equity) else np.nan,
        "capital_employed": float(cap_employed) if not np.isnan(cap_employed) else np.nan,
    }


def compute_debt_ratios(balance_df: pd.DataFrame) -> dict[str, float]:
    """Calculate Debt-to-Equity and Debt-to-Asset ratios."""
    debt, _ = _get_latest_item(balance_df, TOTAL_DEBT_KEYS)
    equity, _ = _get_latest_item(balance_df, EQUITY_KEYS)
    liabilities, _ = _get_latest_item(balance_df, LIABILITIES_KEYS)
    assets, _ = _get_latest_item(balance_df, TOTAL_ASSETS_KEYS)

    d_to_e = np.nan
    if not np.isnan(debt) and not np.isnan(equity) and equity > 0:
        d_to_e = float(debt / equity)

    d_to_a = np.nan
    if not np.isnan(liabilities) and not np.isnan(assets) and assets > 0:
        d_to_a = float(liabilities / assets)
    elif not np.isnan(debt) and not np.isnan(assets) and assets > 0:
        d_to_a = float(debt / assets)

    return {
        "debt_to_equity": d_to_e,
        "debt_to_asset": d_to_a,
        "total_debt": float(debt) if not np.isnan(debt) else np.nan,
        "total_assets": float(assets) if not np.isnan(assets) else np.nan,
    }


def extract_cash_flows(cash_flow_df: pd.DataFrame) -> dict[str, float]:
    """Extract operating cash flow, capex, and free cash flow."""
    ocf, _ = _get_latest_item(cash_flow_df, OPERATING_CF_KEYS)
    capex, _ = _get_latest_item(cash_flow_df, CAPEX_KEYS)
    fcf, _ = _get_latest_item(cash_flow_df, FCF_KEYS)

    if np.isnan(fcf) and not np.isnan(ocf) and not np.isnan(capex):
        # FCF = OCF - CapEx (capex in yfinance is usually negative)
        fcf = float(ocf + capex) if capex < 0 else float(ocf - capex)

    return {
        "operating_cash_flow": float(ocf) if not np.isnan(ocf) else np.nan,
        "capital_expenditure": float(capex) if not np.isnan(capex) else np.nan,
        "free_cash_flow": float(fcf) if not np.isnan(fcf) else np.nan,
    }


def analyze_fundamentals(
    statements: dict[str, pd.DataFrame],
    quote_info: Optional[dict[str, Any]] = None,
) -> FundamentalData:
    """Run comprehensive fundamental analysis on financial statement frames."""
    quote_info = quote_info or {}
    income = statements.get("income", pd.DataFrame())
    balance = statements.get("balance_sheet", pd.DataFrame())
    cash_flow = statements.get("cash_flow", pd.DataFrame())

    stmts_avail = {
        "income": isinstance(income, pd.DataFrame) and not income.empty,
        "balance_sheet": isinstance(balance, pd.DataFrame) and not balance.empty,
        "cash_flow": isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty,
    }

    missing_fields: list[str] = []
    if not stmts_avail["income"]:
        missing_fields.append("income statement")
    if not stmts_avail["balance_sheet"]:
        missing_fields.append("balance sheet")
    if not stmts_avail["cash_flow"]:
        missing_fields.append("cash flow statement")

    shares_out = quote_info.get("shares_outstanding")

    # 1. Margins
    margins = compute_margins(income) if stmts_avail["income"] else {}
    # 2. Returns & Earnings
    returns_eps = (
        compute_returns_and_earnings(income, balance, shares_outstanding=shares_out)
        if stmts_avail["income"] and stmts_avail["balance_sheet"]
        else {}
    )
    # 3. Debt
    debt_metrics = compute_debt_ratios(balance) if stmts_avail["balance_sheet"] else {}
    # 4. Cash flows
    cf_metrics = extract_cash_flows(cash_flow) if stmts_avail["cash_flow"] else {}

    # Check for specific missing components
    if np.isnan(margins.get("gross_margin", np.nan)):
        missing_fields.append("gross margin")
    if np.isnan(returns_eps.get("roe", np.nan)):
        missing_fields.append("ROE")
    if np.isnan(debt_metrics.get("debt_to_equity", np.nan)):
        missing_fields.append("Debt/Equity")
    if np.isnan(cf_metrics.get("free_cash_flow", np.nan)):
        missing_fields.append("Free Cash Flow")

    return FundamentalData(
        gross_margin=margins.get("gross_margin", np.nan),
        operating_margin=margins.get("operating_margin", np.nan),
        net_margin=margins.get("net_margin", np.nan),
        eps=returns_eps.get("eps", np.nan),
        roe=returns_eps.get("roe", np.nan),
        roce=returns_eps.get("roce", np.nan),
        debt_to_equity=debt_metrics.get("debt_to_equity", np.nan),
        debt_to_asset=debt_metrics.get("debt_to_asset", np.nan),
        revenue_latest=margins.get("revenue", np.nan),
        profit_latest=margins.get("net_income", np.nan),
        free_cash_flow=cf_metrics.get("free_cash_flow", np.nan),
        shares_outstanding=float(shares_out) if shares_out else np.nan,
        currency=quote_info.get("currency", "INR"),
        statements_available=stmts_avail,
        missing_fields=missing_fields,
    )
