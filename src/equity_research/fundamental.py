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

# ---------------------------------------------------------------------------
# Normalized key matching — survives yfinance label renames (0.2.x spaced
# "Total Revenue" vs 1.x camelCase "TotalRevenue") and banking-specific
# line item names without silent NaN.
# ---------------------------------------------------------------------------

def _normalize_key(label: str) -> str:
    """Reduce a row label to lowercase alphanumerics for tolerant matching.

    yfinance 0.2.x returned ``"Total Revenue"`` and 1.x returns
    ``"TotalRevenue"``.  Folding away case, spaces, and underscores makes
    the alias lists survive this rename and the next one.
    """
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _get_latest_item(df: pd.DataFrame, candidate_keys: list[str]) -> tuple[float, Optional[str]]:
    """Look up the most recent period's value for the first matching line item.

    Tries exact match first (cheap), then a normalized fallback (survives
    provider renames).  Returns (value, matched_key) or (np.nan, None).
    """
    if df is None or df.empty:
        return np.nan, None

    # Pass 1 — exact match
    for key in candidate_keys:
        if key in df.index:
            row = df.loc[key]
            val = row.iloc[0] if isinstance(row, pd.Series) else row
            try:
                num = float(val)
                if not np.isnan(num):
                    return num, key
            except (ValueError, TypeError):
                continue

    # Pass 2 — normalized (case-/punctuation-insensitive) match
    wanted = {_normalize_key(k) for k in candidate_keys}
    for label in df.index:
        if _normalize_key(label) in wanted:
            row = df.loc[label]
            val = row.iloc[0] if isinstance(row, pd.Series) else row
            try:
                num = float(val)
                if not np.isnan(num):
                    return num, str(label)
            except (ValueError, TypeError):
                continue

    return np.nan, None


# ---------------------------------------------------------------------------
# Row aliases for Yahoo Finance / standard filings
# ---------------------------------------------------------------------------
# Each list includes spaced labels (yfinance 0.2.x), camelCase labels
# (yfinance 1.x), and banking/financial-sector alternates.

REVENUE_KEYS = [
    "Total Revenue", "TotalRevenue",
    "Operating Revenue", "OperatingRevenue",
    "Revenue",
    "Net Interest Income",         # banking
    "Total Interest Income",       # banking
]
GROSS_PROFIT_KEYS = [
    "Gross Profit", "GrossProfit",
]
COST_OF_REVENUE_KEYS = [
    "Cost Of Revenue", "CostOfRevenue",
    "Cost of Goods Sold", "CostOfGoodsSold",
    "Total Cost Of Revenue",
]
OPERATING_INCOME_KEYS = [
    "Operating Income", "OperatingIncome",
    "Operating Profit", "OperatingProfit",
    "Operating Income EBIT",
]
EBIT_KEYS = [
    "EBIT",
    "Operating Income", "OperatingIncome",
    "Operating Profit", "OperatingProfit",
    "Normalized EBITDA",
]
PRETAX_INCOME_KEYS = [
    "Pretax Income", "PretaxIncome",
    "Income Before Tax",
    "Earnings Before Tax",
]
INTEREST_EXPENSE_KEYS = [
    "Interest Expense", "InterestExpense",
    "Interest Expense Non Operating",
    "Net Interest Income",  # sometimes negative = expense for non-banks
]
NET_INCOME_KEYS = [
    "Net Income Common Stockholders", "NetIncomeCommonStockholders",
    "Net Income", "NetIncome",
    "Net Income Applicable To Common Shares",
    "Net Income Available to Stockholders",
]
TOTAL_ASSETS_KEYS = ["Total Assets", "TotalAssets"]
EQUITY_KEYS = [
    "Stockholders Equity", "StockholdersEquity",
    "Total Equity", "TotalEquity",
    "Total Stockholders Equity", "TotalStockholdersEquity",
    "Common Stock Equity", "CommonStockEquity",
]
TOTAL_DEBT_KEYS = [
    "Total Debt", "TotalDebt",
    "Long Term Debt", "LongTermDebt",
    "Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest",
]
CURRENT_LIABILITIES_KEYS = [
    "Current Liabilities", "CurrentLiabilities",
    "Total Current Liabilities",
]
CASH_KEYS = [
    "Cash And Cash Equivalents", "CashAndCashEquivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Short Term Investments",
]
LIABILITIES_KEYS = [
    "Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest",
    "Total Liabilities", "TotalLiabilities",
    "Current Liabilities", "CurrentLiabilities",
]
OPERATING_CF_KEYS = [
    "Operating Cash Flow", "OperatingCashFlow",
    "Cash Flow From Continuing Operating Activities",
]
CAPEX_KEYS = ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"]
FCF_KEYS = ["Free Cash Flow", "FreeCashFlow"]


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
    """Calculate gross, operating, net profit, and EBITDA margins with fallback derivations."""
    rev, _ = _get_latest_item(income_df, REVENUE_KEYS)
    gp, _ = _get_latest_item(income_df, GROSS_PROFIT_KEYS)
    op_inc, _ = _get_latest_item(income_df, OPERATING_INCOME_KEYS)
    net_inc, _ = _get_latest_item(income_df, NET_INCOME_KEYS)

    if np.isnan(rev) or rev <= 0:
        return {
            "gross_margin": np.nan,
            "operating_margin": np.nan,
            "net_margin": np.nan,
            "ebitda_margin": np.nan,
            "ebitda": np.nan,
            "revenue": np.nan,
            "net_income": np.nan,
        }

    # Fallback derivation for Gross Profit if absent: Revenue - Cost of Revenue
    if np.isnan(gp):
        cor, _ = _get_latest_item(income_df, COST_OF_REVENUE_KEYS)
        if not np.isnan(cor):
            gp = rev - cor
        elif not np.isnan(op_inc):
            gp = op_inc

    if np.isnan(op_inc) and not np.isnan(gp):
        op_inc = gp

    # EBITDA calculation: Net Income + Interest + Tax + Depreciation (+ Amortization)
    ebitda, _ = _get_latest_item(income_df, ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA", "Ebitda"])
    depr_val, _ = _get_latest_item(income_df, [
        "Depreciation And Amortization", "DepreciationAmortizationDepletion",
        "Depreciation", "Amortization", "Reconciled Depreciation"
    ])
    if np.isnan(ebitda):
        ebit_val, _ = _get_latest_item(income_df, EBIT_KEYS)
        if not np.isnan(ebit_val) and not np.isnan(depr_val):
            ebitda = ebit_val + depr_val
        elif not np.isnan(net_inc):
            tax_val, _ = _get_latest_item(income_df, ["Tax Provision", "TaxProvision", "Income Tax Expense"])
            int_val, _ = _get_latest_item(income_df, INTEREST_EXPENSE_KEYS)
            t_val = tax_val if not np.isnan(tax_val) else 0.0
            i_val = int_val if not np.isnan(int_val) else 0.0
            d_val = depr_val if not np.isnan(depr_val) else 0.0
            ebitda = net_inc + i_val + t_val + d_val
        elif not np.isnan(op_inc):
            ebitda = op_inc + (depr_val if not np.isnan(depr_val) else 0.0)

    ebitda_margin = float(ebitda / rev) if not np.isnan(ebitda) and rev > 0 else np.nan

    return {
        "gross_margin": float(gp / rev) if not np.isnan(gp) and rev > 0 else np.nan,
        "operating_margin": float(op_inc / rev) if not np.isnan(op_inc) and rev > 0 else np.nan,
        "net_margin": float(net_inc / rev) if not np.isnan(net_inc) and rev > 0 else np.nan,
        "ebitda_margin": ebitda_margin,
        "ebitda": float(ebitda) if not np.isnan(ebitda) else np.nan,
        "revenue": float(rev),
        "net_income": float(net_inc) if not np.isnan(net_inc) else np.nan,
    }


def compute_returns_and_earnings(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    shares_outstanding: Optional[float] = None,
) -> dict[str, float]:
    """Calculate ROE, ROCE, and EPS with robust fallbacks for financial/banking stocks."""
    net_inc, _ = _get_latest_item(income_df, NET_INCOME_KEYS)
    ebit, _ = _get_latest_item(income_df, EBIT_KEYS)
    if np.isnan(ebit):
        op_inc, _ = _get_latest_item(income_df, OPERATING_INCOME_KEYS)
        if not np.isnan(op_inc):
            ebit = op_inc
        else:
            pretax, _ = _get_latest_item(income_df, PRETAX_INCOME_KEYS)
            interest, _ = _get_latest_item(income_df, INTEREST_EXPENSE_KEYS)
            if not np.isnan(pretax) and not np.isnan(interest):
                ebit = pretax + interest

    equity, _ = _get_latest_item(balance_df, EQUITY_KEYS)
    debt, _ = _get_latest_item(balance_df, TOTAL_DEBT_KEYS)
    assets, _ = _get_latest_item(balance_df, TOTAL_ASSETS_KEYS)
    current_liab, _ = _get_latest_item(balance_df, CURRENT_LIABILITIES_KEYS)

    # ROE = Net Income / Equity
    roe = np.nan
    if not np.isnan(net_inc) and not np.isnan(equity) and equity > 0:
        roe = float(net_inc / equity)
    elif not np.isnan(net_inc) and not np.isnan(assets) and assets > 0:
        # Fallback ROE using assets if equity is absent
        roe = float(net_inc / assets)

    # ROCE = EBIT / Capital Employed
    # Capital Employed = Equity + Total Debt (or Total Assets - Current Liabilities)
    roce = np.nan
    cap_employed = np.nan
    if not np.isnan(equity):
        d_val = debt if not np.isnan(debt) else 0.0
        cap_employed = equity + d_val
    elif not np.isnan(assets) and not np.isnan(current_liab):
        cap_employed = assets - current_liab
    elif not np.isnan(assets):
        cap_employed = assets

    if not np.isnan(ebit) and not np.isnan(cap_employed) and cap_employed > 0:
        roce = float(ebit / cap_employed)
    elif not np.isnan(net_inc) and not np.isnan(cap_employed) and cap_employed > 0:
        # Fallback using net income if EBIT is absent
        roce = float(net_inc / cap_employed)

    # EPS = Net Income / Shares Outstanding
    eps = np.nan
    if not np.isnan(net_inc) and shares_outstanding and shares_outstanding > 0:
        eps = float(net_inc / shares_outstanding)
    else:
        reported_eps, _ = _get_latest_item(income_df, ["Basic EPS", "Diluted EPS", "BasicEPS", "DilutedEPS"])
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
        ebitda_margin=margins.get("ebitda_margin", np.nan),
        ebitda=margins.get("ebitda", np.nan),
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
