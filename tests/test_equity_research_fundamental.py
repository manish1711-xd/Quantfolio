import numpy as np
import pandas as pd
import pytest

from src.equity_research.fundamental import (
    compute_margins,
    compute_returns_and_earnings,
    compute_debt_ratios,
    extract_cash_flows,
    analyze_fundamentals,
)


@pytest.fixture
def sample_statements():
    income_df = pd.DataFrame(
        {
            "2024-03-31": [100000.0, 60000.0, 20000.0, 18000.0, 14000.0, 1000.0],
            "2023-03-31": [85000.0, 52000.0, 16000.0, 15000.0, 11000.0, 1000.0],
        },
        index=[
            "Total Revenue",
            "Gross Profit",
            "Operating Income",
            "EBIT",
            "Net Income Common Stockholders",
            "Basic EPS",
        ],
    )
    balance_df = pd.DataFrame(
        {
            "2024-03-31": [150000.0, 90000.0, 30000.0, 15000.0, 60000.0],
            "2023-03-31": [130000.0, 80000.0, 25000.0, 12000.0, 50000.0],
        },
        index=[
            "Total Assets",
            "Stockholders Equity",
            "Total Debt",
            "Cash And Cash Equivalents",
            "Total Liabilities Net Minority Interest",
        ],
    )
    cash_df = pd.DataFrame(
        {
            "2024-03-31": [22000.0, -7000.0, 15000.0],
            "2023-03-31": [19000.0, -6000.0, 13000.0],
        },
        index=[
            "Operating Cash Flow",
            "Capital Expenditure",
            "Free Cash Flow",
        ],
    )
    return {
        "income": income_df,
        "balance_sheet": balance_df,
        "cash_flow": cash_df,
    }


def test_compute_margins_valid(sample_statements):
    margins = compute_margins(sample_statements["income"])
    assert np.isclose(margins["gross_margin"], 60000.0 / 100000.0)
    assert np.isclose(margins["operating_margin"], 20000.0 / 100000.0)
    assert np.isclose(margins["net_margin"], 14000.0 / 100000.0)


def test_compute_margins_zero_revenue():
    income_zero = pd.DataFrame(
        {"2024-03-31": [0.0, 0.0, 0.0, 0.0]},
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    margins = compute_margins(income_zero)
    assert np.isnan(margins["gross_margin"])
    assert np.isnan(margins["operating_margin"])
    assert np.isnan(margins["net_margin"])


def test_compute_returns_and_earnings(sample_statements):
    metrics = compute_returns_and_earnings(
        sample_statements["income"], sample_statements["balance_sheet"], shares_outstanding=1000.0
    )
    # ROE = Net Income / Equity = 14000 / 90000
    assert np.isclose(metrics["roe"], 14000.0 / 90000.0)
    # ROCE = EBIT / (Total Assets - Current Liabilities or Capital Employed = Equity + Debt)
    # Capital Employed = 90000 + 30000 = 120000; EBIT = 18000 -> 18000/120000 = 0.15
    assert np.isclose(metrics["roce"], 18000.0 / 120000.0)
    assert np.isclose(metrics["eps"], 14.0)


def test_compute_debt_ratios(sample_statements):
    debt_ratios = compute_debt_ratios(sample_statements["balance_sheet"])
    # Total Debt / Equity = 30000 / 90000 = 0.3333
    assert np.isclose(debt_ratios["debt_to_equity"], 30000.0 / 90000.0)
    # Total Liabilities / Assets = 60000 / 150000 = 0.40
    assert np.isclose(debt_ratios["debt_to_asset"], 60000.0 / 150000.0)


def test_extract_cash_flows(sample_statements):
    cf = extract_cash_flows(sample_statements["cash_flow"])
    assert cf["operating_cash_flow"] == 22000.0
    assert cf["capital_expenditure"] == -7000.0
    assert cf["free_cash_flow"] == 15000.0


def test_analyze_fundamentals_empty_statements():
    empty_stmts = {"income": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cash_flow": pd.DataFrame()}
    fund_data = analyze_fundamentals(empty_stmts, quote_info={})
    assert np.isnan(fund_data.net_margin)
    assert np.isnan(fund_data.roe)
    assert "income statement" in fund_data.missing_fields or "balance sheet" in fund_data.missing_fields
