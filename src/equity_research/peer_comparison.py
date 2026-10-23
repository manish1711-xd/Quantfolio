"""Peer Comparison Analysis for Equity Research.

Calculates comparative valuation multiples (P/E, P/B, Dividend Yield, Beta),
benchmark sector averages, and target company percentile rankings.
Never fabricates missing peer data; reports data sufficiency honestly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.equity_research.models import PeerComparisonResult

logger = logging.getLogger(__name__)

COMPARISON_METRICS = ["pe_ratio", "pb_ratio", "beta", "dividend_yield", "market_cap"]


def compute_sector_averages(peers_df: pd.DataFrame) -> dict[str, float]:
    """Compute arithmetic mean of peer multiples, ignoring NaNs."""
    if peers_df is None or peers_df.empty:
        return {}

    averages: dict[str, float] = {}
    for col in COMPARISON_METRICS:
        if col in peers_df.columns:
            valid = peers_df[col].dropna()
            if len(valid) > 0:
                averages[col] = float(valid.mean())
    return averages


def compute_percentiles(
    target_multiples: dict[str, Optional[float]],
    peers_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute target company percentile rank within the peer group.
    Percentile = fraction of peers whose metric is strictly lower than the target.
    """
    if peers_df is None or peers_df.empty:
        return {}

    ranks: dict[str, float] = {}
    for metric, target_val in target_multiples.items():
        if target_val is None or np.isnan(target_val):
            continue
        if metric in peers_df.columns:
            valid_peers = peers_df[metric].dropna()
            if len(valid_peers) > 0:
                lower_count = (valid_peers < target_val).sum()
                ranks[metric] = float(lower_count / len(valid_peers))
    return ranks


def analyze_peer_comparison(
    ticker: str,
    peers_df: pd.DataFrame,
    target_multiples: dict[str, Optional[float]],
) -> PeerComparisonResult:
    """Analyze target company position relative to industry peers."""
    notes: list[str] = []

    if peers_df is None or peers_df.empty:
        notes.append("No peer company data available for comparison.")
        return PeerComparisonResult(
            target_ticker=ticker,
            peers=pd.DataFrame(),
            target_multiples=target_multiples,
            sector_averages={},
            percentile_rank={},
            sufficient_data=False,
            notes=notes,
        )

    sector_averages = compute_sector_averages(peers_df)
    percentiles = compute_percentiles(target_multiples, peers_df)

    # Need at least 2 valid peer records for meaningful comparative analysis
    sufficient = len(peers_df) >= 2
    if not sufficient:
        notes.append(f"Only {len(peers_df)} peer(s) found; minimum 2 required for reliable sector comparison.")

    return PeerComparisonResult(
        target_ticker=ticker,
        peers=peers_df,
        target_multiples=target_multiples,
        sector_averages=sector_averages,
        percentile_rank=percentiles,
        sufficient_data=sufficient,
        notes=notes,
    )
