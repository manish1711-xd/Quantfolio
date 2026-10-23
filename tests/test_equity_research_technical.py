import numpy as np
import pandas as pd
import pytest

from src.equity_research.technical import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_volatility,
    compute_beta,
    compute_sharpe_ratio,
    compute_max_drawdown,
    analyze_technicals,
)


@pytest.fixture
def sample_price_series():
    # 250 daily prices with an upward trend
    dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
    np.random.seed(42)
    # Drift 0.05% per day, vol 1%
    returns = np.random.normal(0.0005, 0.01, size=250)
    prices = 1000.0 * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=dates)


def test_compute_sma(sample_price_series):
    sma20 = compute_sma(sample_price_series, 20)
    expected = sample_price_series.iloc[-20:].mean()
    assert np.isclose(sma20, expected)

    sma_short = compute_sma(sample_price_series.iloc[:10], 20)
    assert np.isnan(sma_short)


def test_compute_ema(sample_price_series):
    ema20 = compute_ema(sample_price_series, 20)
    assert not np.isnan(ema20)
    assert ema20 > 0


def test_compute_rsi(sample_price_series):
    rsi = compute_rsi(sample_price_series, 14)
    assert not np.isnan(rsi)
    assert 0 <= rsi <= 100

    # Test monotonic series
    rising = pd.Series(range(1, 30), index=pd.date_range("2024-01-01", periods=29))
    rsi_rising = compute_rsi(rising, 14)
    assert rsi_rising > 90

    falling = pd.Series(range(30, 1, -1), index=pd.date_range("2024-01-01", periods=29))
    rsi_falling = compute_rsi(falling, 14)
    assert rsi_falling < 10


def test_compute_macd(sample_price_series):
    macd = compute_macd(sample_price_series, 12, 26, 9)
    assert "macd" in macd
    assert "signal" in macd
    assert "histogram" in macd
    assert np.isclose(macd["histogram"], macd["macd"] - macd["signal"])


def test_compute_volatility_and_sharpe(sample_price_series):
    vol = compute_volatility(sample_price_series, trading_days=252)
    assert not np.isnan(vol)
    assert 0.05 < vol < 0.40

    sharpe = compute_sharpe_ratio(sample_price_series, risk_free_rate=0.06, trading_days=252)
    assert not np.isnan(sharpe)


def test_compute_max_drawdown():
    # Peak at 100, drops to 70, recovers to 90 -> drawdown is (70 - 100) / 100 = -0.30
    prices = pd.Series([80, 90, 100, 85, 70, 75, 90])
    mdd = compute_max_drawdown(prices)
    assert np.isclose(mdd, -0.30)


def test_compute_beta_against_benchmark(sample_price_series):
    # Perfect correlation: beta of series against itself is 1.0
    beta_self = compute_beta(sample_price_series, sample_price_series)
    assert np.isclose(beta_self, 1.0)

    # 2x leverage series: beta should be 2.0
    levered = sample_price_series * 2.0
    # Note: returns are identical, so beta is 1.0. Let's make 2x daily return:
    r = sample_price_series.pct_change()
    levered_returns = r * 2.0
    levered_series = (1 + levered_returns).cumprod()
    beta_2x = compute_beta(levered_series, sample_price_series)
    assert np.isclose(beta_2x, 2.0, atol=0.05)


def test_analyze_technicals_empty():
    tech = analyze_technicals(pd.DataFrame(), pd.DataFrame())
    assert np.isnan(tech.rsi)
    assert np.isnan(tech.volatility)
    assert np.isnan(tech.beta)
