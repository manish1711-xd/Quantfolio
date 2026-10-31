"""Quarterly TTM (Trailing Twelve Months) & Seasonality Engine.

Extracts real-time financial trajectory by aggregating trailing 4 quarters:
- Trailing 4-quarter sums for income & cash flow statement metrics
- Most recent balance sheet point-in-time metrics
- Trailing margins: TTM Operating Margin, TTM Net Margin, TTM EBITDA Margin, TTM FCF Margin
- Seasonality decomposition across fiscal quarters (Q1-Q4)
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def _clean_key(name: object) -> str:
    """Normalize row label for robust alphanumeric matching."""
    return "".join(c for c in str(name).lower() if c.isalnum())


def _extract_metric(df: pd.DataFrame, target_labels: list[str], col: object) -> float:
    """Extract a numeric value for target labels from a statement column."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.nan
    wanted = {_clean_key(l) for l in target_labels}
    for idx in df.index:
        if _clean_key(idx) in wanted:
            val = df.at[idx, col]
            try:
                f_val = float(val)
                if math.isfinite(f_val):
                    return f_val
            except (ValueError, TypeError):
                continue
    return np.nan


def aggregate_ttm_metrics(
    quarterly_income: pd.DataFrame | None,
    quarterly_cashflow: pd.DataFrame | None,
    quarterly_balance_sheet: pd.DataFrame | None,
    max_quarters: int = 4,
) -> dict[str, Any]:
    """Aggregate trailing 4 quarters into a unified TTM dataset."""
    result: dict[str, Any] = {
        "ttm_revenue": np.nan,
        "ttm_operating_income": np.nan,
        "ttm_net_income": np.nan,
        "ttm_ebitda": np.nan,
        "ttm_operating_cash_flow": np.nan,
        "ttm_capex": np.nan,
        "ttm_free_cash_flow": np.nan,
        "ttm_gross_margin": np.nan,
        "ttm_operating_margin": np.nan,
        "ttm_net_margin": np.nan,
        "ttm_ebitda_margin": np.nan,
        "ttm_fcf_margin": np.nan,
        "latest_total_debt": np.nan,
        "latest_cash": np.nan,
        "latest_net_debt": np.nan,
        "latest_equity": np.nan,
        "latest_total_assets": np.nan,
        "quarters_aggregated": 0,
        "latest_quarter_date": "",
    }

    # Extract available quarter columns sorted newest first
    cols_set: set[object] = set()
    for frame in (quarterly_income, quarterly_cashflow, quarterly_balance_sheet):
        if frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
            cols_set.update(frame.columns)

    if not cols_set:
        return result

    def _sort_col(col: object):
        try:
            return pd.Timestamp(col)
        except Exception:
            return pd.Timestamp.min

    sorted_cols = sorted(list(cols_set), key=_sort_col, reverse=True)[:max_quarters]
    n_q = len(sorted_cols)
    result["quarters_aggregated"] = n_q
    if sorted_cols:
        result["latest_quarter_date"] = str(sorted_cols[0])

    # Income Statement Trailing Flow Items
    rev_vals: list[float] = []
    op_inc_vals: list[float] = []
    net_inc_vals: list[float] = []
    ebitda_vals: list[float] = []
    gp_vals: list[float] = []

    for col in sorted_cols:
        r = _extract_metric(quarterly_income, ["Total Revenue", "Operating Revenue", "Revenue"], col)
        if not np.isnan(r):
            rev_vals.append(r)

        op = _extract_metric(quarterly_income, ["Operating Income", "Operating Profit", "EBIT"], col)
        if not np.isnan(op):
            op_inc_vals.append(op)

        ni = _extract_metric(quarterly_income, ["Net Income", "Net Income Common Stockholders"], col)
        if not np.isnan(ni):
            net_inc_vals.append(ni)

        eb = _extract_metric(quarterly_income, ["EBITDA", "Normalized EBITDA"], col)
        if not np.isnan(eb):
            ebitda_vals.append(eb)

        gp = _extract_metric(quarterly_income, ["Gross Profit"], col)
        if not np.isnan(gp):
            gp_vals.append(gp)

    # Cash Flow Trailing Flow Items
    ocf_vals: list[float] = []
    capex_vals: list[float] = []

    for col in sorted_cols:
        ocf = _extract_metric(quarterly_cashflow, ["Operating Cash Flow", "Cash Flow From Operating Activities"], col)
        if not np.isnan(ocf):
            ocf_vals.append(ocf)

        cap = _extract_metric(quarterly_cashflow, ["Capital Expenditure", "Capital Expenditures"], col)
        if not np.isnan(cap):
            capex_vals.append(abs(cap))

    # Aggregations (with scale factor if < 4 quarters available)
    scale = (4.0 / n_q) if (0 < n_q < 4) else 1.0

    if rev_vals:
        result["ttm_revenue"] = float(sum(rev_vals) * scale)
    if op_inc_vals:
        result["ttm_operating_income"] = float(sum(op_inc_vals) * scale)
    if net_inc_vals:
        result["ttm_net_income"] = float(sum(net_inc_vals) * scale)

    # EBITDA fallback: if not reported, Net Income + Tax + Interest + D&A or Operating Income + D&A
    if ebitda_vals:
        result["ttm_ebitda"] = float(sum(ebitda_vals) * scale)
    elif not np.isnan(result["ttm_operating_income"]):
        result["ttm_ebitda"] = float(result["ttm_operating_income"] * 1.15)  # D&A proxy

    if ocf_vals:
        result["ttm_operating_cash_flow"] = float(sum(ocf_vals) * scale)
    if capex_vals:
        result["ttm_capex"] = float(sum(capex_vals) * scale)

    # Free cash flow = OCF - Capex
    if not np.isnan(result["ttm_operating_cash_flow"]):
        c = result["ttm_capex"] if not np.isnan(result["ttm_capex"]) else 0.0
        result["ttm_free_cash_flow"] = float(result["ttm_operating_cash_flow"] - c)

    # Margin Calculations
    r_ttm = result["ttm_revenue"]
    if not np.isnan(r_ttm) and r_ttm > 0:
        if gp_vals:
            result["ttm_gross_margin"] = float((sum(gp_vals) * scale) / r_ttm)
        if not np.isnan(result["ttm_operating_income"]):
            result["ttm_operating_margin"] = float(result["ttm_operating_income"] / r_ttm)
        if not np.isnan(result["ttm_net_income"]):
            result["ttm_net_margin"] = float(result["ttm_net_income"] / r_ttm)
        if not np.isnan(result["ttm_ebitda"]):
            result["ttm_ebitda_margin"] = float(result["ttm_ebitda"] / r_ttm)
        if not np.isnan(result["ttm_free_cash_flow"]):
            result["ttm_fcf_margin"] = float(result["ttm_free_cash_flow"] / r_ttm)

    # Latest Balance Sheet Stock Items (From newest available quarter)
    if sorted_cols:
        latest_col = sorted_cols[0]
        debt = _extract_metric(quarterly_balance_sheet, ["Total Debt", "Long Term Debt"], latest_col)
        cash = _extract_metric(quarterly_balance_sheet, ["Cash And Cash Equivalents", "Cash"], latest_col)
        equity = _extract_metric(quarterly_balance_sheet, ["Stockholders Equity", "Common Stock Equity"], latest_col)
        assets = _extract_metric(quarterly_balance_sheet, ["Total Assets"], latest_col)

        result["latest_total_debt"] = debt if not np.isnan(debt) else 0.0
        result["latest_cash"] = cash if not np.isnan(cash) else 0.0
        result["latest_net_debt"] = float(result["latest_total_debt"] - result["latest_cash"])
        result["latest_equity"] = equity
        result["latest_total_assets"] = assets

    return result


def compute_seasonality_indices(quarterly_series: pd.Series) -> dict[str, float]:
    """Calculate seasonal indices (Q1-Q4) normalized to average 1.0."""
    default_indices = {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0}
    if quarterly_series is None or quarterly_series.empty or len(quarterly_series) < 4:
        return default_indices

    s = quarterly_series.dropna()
    if len(s) < 4:
        return default_indices

    # Map timestamps to quarters
    q_map: dict[str, list[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    overall_mean = float(s.mean())
    if overall_mean <= 0:
        return default_indices

    for idx, val in s.items():
        try:
            ts = pd.Timestamp(idx)
            q_label = f"Q{ts.quarter}"
            q_map[q_label].append(float(val))
        except Exception:
            continue

    indices: dict[str, float] = {}
    for q_label in ["Q1", "Q2", "Q3", "Q4"]:
        vals = q_map[q_label]
        if vals:
            indices[q_label] = float(np.mean(vals) / overall_mean)
        else:
            indices[q_label] = 1.0

    # Normalize so average is exactly 1.0
    avg_factor = float(np.mean(list(indices.values())))
    if avg_factor > 0:
        for k in indices:
            indices[k] = round(indices[k] / avg_factor, 4)

    return indices
