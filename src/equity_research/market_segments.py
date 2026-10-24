"""Market Segment Ratios and Analytics from NSE India Data.

Computes comprehensive market segment ratios, turnover distribution, sector
market shares, and valuation/activity metrics from official NSE listings
and end-of-day bhavcopies via src.data.exchange and src.stock_universe.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.stock_universe import load_universe
from src.data.exchange import fetch_nse_listing

logger = logging.getLogger(__name__)


def get_nse_market_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch or load NSE market universe and bhavcopy turnover data.

    Returns a DataFrame of records and a metadata summary dict.
    """
    universe = load_universe()
    nse_records = [r for r in universe if r.exchange.upper() == "NSE"]

    if not nse_records:
        listing = fetch_nse_listing()
        if listing.ok:
            records = []
            for row in listing.rows:
                sym = row.get("SYMBOL", "")
                if sym:
                    records.append({
                        "symbol": sym,
                        "yahoo_symbol": f"{sym}.NS",
                        "name": row.get("NAME OF COMPANY", sym),
                        "exchange": "NSE",
                        "sector": "Unmapped",
                        "turnover": 0.0,
                    })
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame()
    else:
        rows = []
        for r in nse_records:
            rows.append({
                "symbol": r.symbol,
                "yahoo_symbol": r.yahoo_symbol,
                "name": r.name,
                "exchange": r.exchange,
                "sector": r.sector or "Unmapped",
                "turnover": r.turnover,
                "isin": r.isin,
                "instrument_type": r.instrument_type,
            })
        df = pd.DataFrame(rows)

    meta = {
        "total_nse_stocks": len(df),
        "total_turnover": float(df["turnover"].sum()) if not df.empty and "turnover" in df.columns else 0.0,
        "as_of": "Latest available exchange trade file",
    }
    return df, meta


def compute_market_segment_ratios(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute aggregated ratios and analytics for each market segment / sector on NSE."""
    if df is None or df.empty:
        df, _ = get_nse_market_snapshot()

    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["sector"] = df["sector"].replace("", "Unmapped").fillna("Unmapped")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)

    total_market_turnover = df["turnover"].sum()

    grouped = df.groupby("sector").agg(
        stock_count=("symbol", "count"),
        total_turnover=("turnover", "sum"),
        mean_turnover=("turnover", "mean"),
        median_turnover=("turnover", "median"),
        max_turnover=("turnover", "max"),
    ).reset_index()

    if total_market_turnover > 0:
        grouped["turnover_share_pct"] = (grouped["total_turnover"] / total_market_turnover) * 100.0
    else:
        grouped["turnover_share_pct"] = 0.0

    grouped = grouped.sort_values(by="total_turnover", ascending=False).reset_index(drop=True)
    return grouped
