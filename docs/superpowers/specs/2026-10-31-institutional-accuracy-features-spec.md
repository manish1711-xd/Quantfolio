# Institutional Accuracy & Valuation Engines Specification

**Date:** 2026-10-31  
**Target:** Elevate DCF and Intrinsic Valuation Accuracy to 85%–95% Institutional Grade  

## 1. Overview & Objectives
Quantfolio's Equity Research and DCF valuation engines currently provide multi-model outputs (FCFF, FCFE, DDM, Gordon Growth). To elevate prediction accuracy and operational fidelity to institutional standards (85%–95% predictive alignment), five specialized quantitative engines are introduced:
1. **Dynamic WACC & Real-Time Cost of Capital Engine (`src/analytics/wacc_engine.py`)**: Live 10Y sovereign yields (Indian 10Y G-Sec / US 10Y Treasury proxy `^TNX`), rolling 2Y beta against benchmark, and interest coverage to synthetic credit rating spreads (`Kd = Rf + spread`).
2. **Quarterly TTM & Seasonality Engine (`src/analytics/ttm_engine.py`)**: Real-time aggregation of trailing 4 quarters from `quarterly_income_stmt` and `quarterly_cashflow`, seasonal adjustment factors, and updated balance sheet figures.
3. **ML & Statistical Revenue/Margin Forecasting Engine (`src/analytics/ml_forecast.py`)**: Time-series exponential smoothing (Holt-Winters) and auto-regressive trend projections for 1–5 forward years with conformal prediction intervals (80% / 95% confidence bands).
4. **Valuation Backtesting & Prediction Error Attribution Engine (`src/analytics/backtest_engine.py`)**: Historical point-in-time DCF valuation compared with realized price realizations to compute systematic bias and apply a ground-truth error correction coefficient.
5. **Macroeconomic & Industry Cyclicality Overlay (`src/analytics/macro_overlay.py`)**: Macroeconomic factor adjustments (inflation rate, policy repo rate cycle, currency exchange rate volatility) acting as multipliers on WACC and terminal growth rates.

## 2. Architecture & File Layout
- `src/analytics/wacc_engine.py`: Cost of Capital & Dynamic WACC computations.
- `src/analytics/ttm_engine.py`: Trailing Twelve Month quarterly aggregation and seasonality analysis.
- `src/analytics/ml_forecast.py`: Statistical and ML forecasting for revenue and margin trajectories.
- `src/analytics/backtest_engine.py`: Point-in-time historical DCF backtesting and error calibration.
- `src/analytics/macro_overlay.py`: Cyclical regime and macroeconomic factor overlays.
- `src/equity_research/dcf.py`: Integration with multi-model valuation and dynamic parameters.
- `src/equity_research/data_access.py`: Integration with quarterly financial statements and macro indicators.
- `src/ui/pages/equity_research.py`: Interactive Streamlit visualization panels for institutional metrics.
- `tests/test_institutional_accuracy.py`: Comprehensive test suite verifying all 5 engines and their integrations.

## 3. Interfaces & Contracts

### 3.1 WACC Engine
```python
def calculate_synthetic_credit_spread(interest_coverage: float) -> float: ...
def compute_dynamic_cost_of_equity(risk_free_rate: float, beta: float, equity_risk_premium: float = 0.055) -> float: ...
def calculate_dynamic_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.2517,
) -> float: ...
```

### 3.2 TTM Engine
```python
def aggregate_ttm_metrics(
    quarterly_income: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
) -> dict[str, float]: ...
def compute_seasonality_indices(quarterly_series: pd.Series) -> dict[int, float]: ...
```

### 3.3 ML Forecast Engine
```python
def forecast_time_series_growth(
    historical_values: list[float] | pd.Series,
    horizon_years: int = 5,
    confidence_level: float = 0.80,
) -> dict[str, Any]: ...
```

### 3.4 Backtest Engine
```python
def backtest_historical_dcf(
    historical_periods: list[dict[str, Any]],
    actual_realized_price: float,
) -> dict[str, Any]: ...
def calculate_bias_correction_factor(backtest_results: list[dict[str, Any]]) -> float: ...
```

### 3.5 Macro Overlay
```python
def compute_macro_adjustments(
    base_wacc: float,
    base_growth: float,
    inflation_rate: float = 0.05,
    policy_rate: float = 0.065,
    currency_volatility: float = 0.03,
) -> dict[str, float]: ...
```
