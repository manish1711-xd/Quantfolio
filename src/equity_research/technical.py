"""Technical analysis module for Equity Research.

Calculates moving averages (SMA, EMA), momentum (Wilder's RSI), trend (MACD),
risk (annualized volatility, max drawdown, beta) and risk-adjusted return (Sharpe ratio).
Pure calculation functions with no external network calls and no Streamlit imports.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.config import TRADING_DAYS, MIN_OBSERVATIONS
from src.equity_research.models import TechnicalData

logger = logging.getLogger(__name__)


def compute_sma(series: pd.Series, period: int) -> float:
    """Simple moving average at the latest observation.
    Returns np.nan if series contains fewer observations than period.
    """
    clean = series.dropna()
    if len(clean) < period or period <= 0:
        return np.nan
    return float(clean.iloc[-period:].mean())


def compute_ema(series: pd.Series, period: int) -> float:
    """Exponential moving average with Wilder/standard smoothing (alpha = 2 / (period + 1))."""
    clean = series.dropna()
    if len(clean) < 2 or period <= 0:
        return np.nan
    ema_series = clean.ewm(span=period, adjust=False).mean()
    return float(ema_series.iloc[-1])


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Relative Strength Index using Wilder's original exponential smoothing.
    Wilder: ewm(alpha = 1 / period, adjust=False).
    """
    clean = series.dropna()
    if len(clean) < period + 1:
        return np.nan

    delta = clean.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()

    latest_gain = float(avg_gain.iloc[-1])
    latest_loss = float(avg_loss.iloc[-1])

    if latest_loss == 0.0:
        return 100.0 if latest_gain > 0 else 50.0
    if latest_gain == 0.0:
        return 0.0

    rs = latest_gain / latest_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """Moving Average Convergence Divergence (MACD)."""
    clean = series.dropna()
    if len(clean) < slow + signal:
        return {"macd": np.nan, "signal": np.nan, "histogram": np.nan}

    fast_ema = clean.ewm(span=fast, adjust=False).mean()
    slow_ema = clean.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
    }


def compute_volatility(series: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualized volatility computed from daily log returns using sample standard deviation (ddof=1)."""
    clean = series.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return np.nan

    log_returns = np.log(clean / clean.shift(1)).dropna()
    if len(log_returns) < 2:
        return np.nan

    daily_std = float(log_returns.std(ddof=1))
    return float(daily_std * np.sqrt(trading_days))


def compute_max_drawdown(series: pd.Series) -> float:
    """Maximum peak-to-trough decline expressed as a negative decimal."""
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan

    cummax = clean.cummax()
    drawdown = (clean - cummax) / cummax
    return float(drawdown.min())


def compute_sharpe_ratio(
    series: pd.Series,
    risk_free_rate: float = 0.06,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio = (Annualized Return - Risk Free Rate) / Annualized Volatility."""
    clean = series.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return np.nan

    daily_returns = clean.pct_change().dropna()
    if len(daily_returns) < 2:
        return np.nan

    mean_daily = float(daily_returns.mean())
    ann_return = mean_daily * trading_days
    ann_vol = float(daily_returns.std(ddof=1)) * np.sqrt(trading_days)

    if ann_vol <= 0 or np.isnan(ann_vol):
        return np.nan

    return float((ann_return - risk_free_rate) / ann_vol)


def compute_beta(
    series: pd.Series,
    benchmark_series: pd.Series,
    min_observations: int = MIN_OBSERVATIONS,
) -> float:
    """Beta against benchmark = Cov(R_asset, R_bench) / Var(R_bench)."""
    s_clean = series.dropna()
    b_clean = benchmark_series.dropna()

    s_ret = s_clean.pct_change().dropna()
    b_ret = b_clean.pct_change().dropna()

    combined = pd.DataFrame({"asset": s_ret, "bench": b_ret}).dropna()
    if len(combined) < min_observations:
        return np.nan

    cov_matrix = np.cov(combined["asset"], combined["bench"], ddof=1)
    var_bench = cov_matrix[1, 1]
    cov_asset_bench = cov_matrix[0, 1]

    if var_bench <= 0 or np.isnan(var_bench):
        return np.nan

    return float(cov_asset_bench / var_bench)


def analyze_technicals(
    ohlcv_df: pd.DataFrame,
    bench_df: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.06,
) -> TechnicalData:
    """Run full technical indicator and risk analysis from price history."""
    if ohlcv_df is None or ohlcv_df.empty or "Close" not in ohlcv_df.columns:
        return TechnicalData()

    close = ohlcv_df["Close"].dropna()
    if close.empty:
        return TechnicalData()

    # Moving averages
    sma_dict = {
        "20": compute_sma(close, 20),
        "50": compute_sma(close, 50),
        "200": compute_sma(close, 200),
    }
    ema_dict = {
        "20": compute_ema(close, 20),
        "50": compute_ema(close, 50),
    }

    # Momentum & Trend
    rsi_val = compute_rsi(close, 14)
    macd_val = compute_macd(close, 12, 26, 9)

    # Risk metrics
    vol = compute_volatility(close, TRADING_DAYS)
    mdd = compute_max_drawdown(close)
    sharpe = compute_sharpe_ratio(close, risk_free_rate=risk_free_rate, trading_days=TRADING_DAYS)

    # Beta
    beta_val = np.nan
    if bench_df is not None and not bench_df.empty and "Close" in bench_df.columns:
        bench_close = bench_df["Close"].dropna()
        beta_val = compute_beta(close, bench_close)

    current_p = float(close.iloc[-1])
    high_p = float(ohlcv_df["High"].max()) if "High" in ohlcv_df.columns else float(close.max())
    low_p = float(ohlcv_df["Low"].min()) if "Low" in ohlcv_df.columns else float(close.min())

    return TechnicalData(
        sma=sma_dict,
        ema=ema_dict,
        rsi=rsi_val,
        macd=macd_val,
        volatility=vol,
        beta=beta_val,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        current_price=current_p,
        period_high=high_p,
        period_low=low_p,
    )
