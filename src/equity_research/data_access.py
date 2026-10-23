"""Data access layer for the Equity Research Report Generator.

Interacts with src.data.fetcher and src.data.news to assemble a unified
research bundle while preserving offline testability and network encapsulation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import pandas as pd
import numpy as np

from src.data.fetcher import (
    fetch_quote,
    fetch_profile,
    fetch_statements,
    fetch_ohlcv,
    fetch_benchmark_history,
    DataFetchError,
)
from src.data.news import fetch_company_news

logger = logging.getLogger(__name__)

# Sector peers catalog for Indian & Global common sectors
SECTOR_PEERS: dict[str, list[str]] = {
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
    "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Consumer Cyclical": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "TITAN.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS"],
    "Basic Materials": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS"],
    "Industrials": ["LT.NS", "ADANIENT.NS", "BEL.NS", "SIEMENS.NS"],
    "Utilities": ["NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TATAPOWER.NS"],
    "Communication Services": ["BHARTIARTL.NS"],
}


def fetch_equity_quote(ticker: str) -> tuple[dict[str, Any], list[str]]:
    """Fetch current quote and security profile for a ticker."""
    warnings: list[str] = []
    quote_data: dict[str, Any] = {
        "ticker": ticker,
        "price": np.nan,
        "currency": "INR",
        "name": ticker,
        "sector": "Unknown",
        "industry": "Unknown",
        "market_cap": np.nan,
        "shares_outstanding": np.nan,
        "pe_ratio": np.nan,
        "pb_ratio": np.nan,
        "beta": np.nan,
        "dividend_yield": np.nan,
        "summary": "",
        "day_high": np.nan,
        "day_low": np.nan,
        "previous_close": np.nan,
        "volume": 0,
    }

    try:
        q = fetch_quote(ticker)
        quote_data.update({
            "price": q.price,
            "currency": q.currency,
            "previous_close": q.previous_close,
            "day_high": q.day_high,
            "day_low": q.day_low,
            "volume": q.volume,
            "as_of": q.as_of,
        })
    except Exception as exc:
        warnings.append(f"Failed to fetch live quote for {ticker}: {exc}")

    try:
        prof = fetch_profile(ticker)
        mcap = getattr(prof, "market_cap", None)
        price = quote_data.get("price")

        shares = getattr(prof, "shares_outstanding", None)
        if shares is None and mcap is not None and price and not np.isnan(price) and price > 0:
            shares = mcap / price

        pe = getattr(prof, "pe_ratio", None)
        if pe is None:
            pe = getattr(prof, "trailing_pe", None)
            if pe is None:
                pe = getattr(prof, "forward_pe", None)

        pb = getattr(prof, "pb_ratio", None)
        if pb is None:
            pb = getattr(prof, "price_to_book", None)

        quote_data.update({
            "name": getattr(prof, "name", None) or ticker,
            "sector": getattr(prof, "sector", None) or "Unknown",
            "industry": getattr(prof, "industry", None) or "Unknown",
            "market_cap": mcap if mcap is not None else np.nan,
            "shares_outstanding": shares if shares is not None else np.nan,
            "pe_ratio": pe if pe is not None else np.nan,
            "pb_ratio": pb if pb is not None else np.nan,
            "beta": getattr(prof, "beta", np.nan) if getattr(prof, "beta", None) is not None else np.nan,
            "dividend_yield": getattr(prof, "dividend_yield", np.nan) if getattr(prof, "dividend_yield", None) is not None else np.nan,
            "summary": getattr(prof, "summary", "") or "",
        })
    except Exception as exc:
        warnings.append(f"Failed to fetch profile details for {ticker}: {exc}")

    return quote_data, warnings


def fetch_equity_fundamentals(ticker: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch annual balance sheet, income statement, and cash flow."""
    warnings: list[str] = []
    try:
        stmts, notice = fetch_statements(ticker)
        if notice:
            warnings.append(notice)
        return stmts, warnings
    except Exception as exc:
        warnings.append(f"Statements fetch error for {ticker}: {exc}")
        return {
            "income": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }, warnings


def fetch_equity_history(
    ticker: str,
    period: str = "1y",
    benchmark: str = "^NSEI",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fetch historical OHLCV data for target stock and benchmark price history."""
    warnings: list[str] = []
    ohlcv = pd.DataFrame()
    bench_hist = pd.DataFrame()

    try:
        ohlcv = fetch_ohlcv(ticker, period=period)
        if ohlcv.empty:
            warnings.append(f"No price history returned for {ticker} ({period}).")
    except Exception as exc:
        warnings.append(f"Price history error for {ticker}: {exc}")

    try:
        bench_hist = fetch_benchmark_history(benchmark, period=period)
        if bench_hist.empty:
            warnings.append(f"No benchmark history returned for {benchmark}.")
    except Exception as exc:
        warnings.append(f"Benchmark fetch error for {benchmark}: {exc}")

    return ohlcv, bench_hist, warnings


def fetch_peer_quotes(ticker: str, sector: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """Fetch profile metrics and market multiples for sector peers."""
    warnings: list[str] = []
    peers_list: list[str] = []

    if sector and sector in SECTOR_PEERS:
        peers_list = [p for p in SECTOR_PEERS[sector] if p.upper() != ticker.upper()]
    else:
        # Fallback to general Indian large caps if sector is unmapped
        peers_list = [p for p in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"] if p.upper() != ticker.upper()][:4]

    rows: list[dict[str, Any]] = []
    for peer in peers_list:
        try:
            prof = fetch_profile(peer)
            q = fetch_quote(peer)
            mcap = getattr(prof, "market_cap", None)
            price = getattr(q, "price", np.nan)
            if mcap is None and getattr(prof, "shares_outstanding", None) and price and not np.isnan(price):
                mcap = prof.shares_outstanding * price

            pe = getattr(prof, "pe_ratio", None)
            if pe is None:
                pe = getattr(prof, "trailing_pe", None) or getattr(prof, "forward_pe", None)

            pb = getattr(prof, "pb_ratio", None)
            if pb is None:
                pb = getattr(prof, "price_to_book", None)

            rows.append({
                "ticker": peer,
                "name": getattr(prof, "name", None) or peer,
                "price": price,
                "market_cap": mcap if mcap is not None else np.nan,
                "pe_ratio": pe if pe is not None else np.nan,
                "pb_ratio": pb if pb is not None else np.nan,
                "beta": getattr(prof, "beta", np.nan) if getattr(prof, "beta", None) is not None else np.nan,
                "dividend_yield": getattr(prof, "dividend_yield", np.nan) if getattr(prof, "dividend_yield", None) is not None else np.nan,
            })
        except Exception:
            continue

    if not rows:
        warnings.append(f"No peer data could be collected for {ticker} (sector: {sector}).")
        return pd.DataFrame(), warnings

    return pd.DataFrame(rows), warnings


def fetch_research_bundle(
    ticker: str,
    period: str = "1y",
    benchmark: str = "^NSEI",
) -> tuple[dict[str, Any], list[str]]:
    """Assembles all data required to generate an equity research report."""
    warnings: list[str] = []

    # 1. Quote & Profile
    quote_data, q_warn = fetch_equity_quote(ticker)
    warnings.extend(q_warn)

    # 2. Fundamentals
    statements, f_warn = fetch_equity_fundamentals(ticker)
    warnings.extend(f_warn)

    # 3. Price History
    ohlcv, bench_hist, h_warn = fetch_equity_history(ticker, period=period, benchmark=benchmark)
    warnings.extend(h_warn)

    # 4. News
    company_name = quote_data.get("name", ticker)
    news_items: list[dict[str, Any]] = []
    try:
        raw_news = fetch_company_news(company_name)
        for item in raw_news:
            if hasattr(item, "title"):
                news_items.append({
                    "title": item.title,
                    "link": getattr(item, "link", ""),
                    "published": str(item.published) if getattr(item, "published", None) else "",
                    "source": getattr(item, "source", "News"),
                    "summary": getattr(item, "summary", ""),
                })
            elif isinstance(item, dict):
                news_items.append(item)
    except Exception as exc:
        warnings.append(f"News fetch error for {ticker}: {exc}")

    # 5. Peers
    sector = quote_data.get("sector")
    peers_df, p_warn = fetch_peer_quotes(ticker, sector=sector)
    warnings.extend(p_warn)

    bundle = {
        "ticker": ticker,
        "period": period,
        "benchmark": benchmark,
        "quote": quote_data,
        "statements": statements,
        "ohlcv": ohlcv,
        "bench_hist": bench_hist,
        "news": news_items,
        "peers": peers_df,
    }

    return bundle, warnings
