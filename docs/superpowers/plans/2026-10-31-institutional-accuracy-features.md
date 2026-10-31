# Institutional Accuracy Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five advanced quantitative modules to elevate Quantfolio's valuation accuracy to 85%–95% institutional tier, integrating them into the Equity Research Terminal and DCF model.

**Architecture:** Create five pure analytics modules in `src/analytics/`, integrate them into data access and DCF calculation, add UI panels, and write comprehensive unit tests. Each module will be test-driven with isolated interfaces.

**Tech Stack:** Python 3.13, pandas<3.0, numpy, pandas, pytest, Streamlit, yfinance, pandas-datareader, pandas-ta, and any required financial APIs.

**Spec:** `docs/superpowers/specs/2026-10-31-institutional-accuracy-features-spec.md` (the spec travels with this plan; executors read both).

## Global Constraints

- Use Python 3.13 virtual environment with pandas<3.0 pin.
- All modules must be pure analytics functions with clear interfaces.
- Integration points: `src/equity_research/data_access.py` (fetch_research_bundle), `src/equity_research/dcf.py` (calculate_multi_model_valuation).
- UI: Add panels in `src/ui/pages/equity_research.py` for WACC breakdown, TTM comparison, ML forecast intervals, backtest accuracy, macro overlay.
- Unit tests: Write tests in `tests/test_institutional_accuracy.py` covering each module and integration points.
- Maintain 185/185 existing test coverage; new tests must pass.
- Follow TDD: write failing test first, minimal code, verify pass.
- Commit each task with descriptive messages following the plan's commit steps.

## Review Focus

- WACC Engine: Dynamic risk-free rate integration (10Y yield), rolling beta, interest coverage credit spread mapping.
- TTM Engine: Accurate aggregation of quarterly statements, handling missing periods, preserving balance sheet totals.
- ML Forecast Engine: Accurate time-series forecasting with confidence intervals, avoiding overfitting.
- Backtest Engine: Historical DCF backtesting against actual price realization to compute bias correction.
- Macro Overlay: Macro factor adjustments (inflation, interest rates, currency) to discount rates and growth assumptions.

---

### Task 1: Dynamic WACC & Cost of Capital Engine (`src/analytics/wacc_engine.py`)

**Files:**
- Create: `src/analytics/wacc_engine.py`
- Modify: `src/analytics/__init__.py` to expose WACC functions
- Test: `tests/test_wacc_engine.py`

**Interfaces:**
- Consumes: `src/analytics/forecast_engine.py` (beta, risk-free rate), `src/analytics/macro_overlay.py` (macro adjustments)
- Produces: `wacc_calculation` function returning WACC and cost of debt components

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_wacc_calculation_with_live_rates():
    wacc, kd = calculate_wacc(ticker="AAPL", period="5y")
    assert not np.isnan(wacc), "WACC should be computed"
    assert kd > 0, "Cost of debt should be positive"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_wacc_engine.py::test_wacc_calculation_with_live_rates -v`
Expected: FAIL with "calculate_wacc not defined"
- [ ] **Step 3: Implement `calculate_wacc(ticker: str, period: str) -> Tuple[float, float]` in `src/analytics/wacc_engine.py`**
One line on approach: fetch live 10Y yield, compute rolling beta, map interest coverage to credit spread, calculate WACC using market cap weights.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_wacc_engine.py::test_wacc_calculation_with_live_rates -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/analytics/wacc_engine.py tests/test_wacc_engine.py
git commit -m "feat: add Dynamic WACC & Cost of Capital Engine"
```

### Task 2: Quarterly TTM & Seasonality Engine (`src/analytics/ttm_engine.py`)

**Files:**
- Create: `src/analytics/ttm_engine.py`
- Modify: `src/analytics/__init__.py` to expose TTM functions
- Test: `tests/test_ttm_engine.py`

**Interfaces:**
- Consumes: `src/analytics/data_access.py` (quarterly statements), `src/analytics/wacc_engine.py` (WACC for cash flow discounting)
- Produces: `ttm_income`, `ttm_cashflow`, `ttm_balance_sheet` functions returning aggregated 4-quarter data

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_ttm_income_aggregation():
    ttm_rev = aggregate_ttm_income(statement_data)
    assert ttm_rev == sum([q.revenue for q in last_4_quarters]), "TTM revenue should be sum of last 4 quarters"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ttm_engine.py::test_ttm_income_aggregation -v`
Expected: FAIL with "aggregate_ttm_income not defined"
- [ ] **Step 3: Implement `aggregate_ttm_income(statement_data: dict) -> dict` in `src/analytics/ttm_engine.py`**
Approach: sum revenue, operating income, net income, operating cash flow across the last 4 quarters; handle missing quarters with zero or interpolation.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ttm_engine.py::test_ttm_income_aggregation -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/analytics/ttm_engine.py tests/test_ttm_engine.py
git commit -m "feat: add Quarterly TTM & Seasonality Engine"
```

### Task 3: ML & Statistical Growth & Margin Forecasting Engine (`src/analytics/ml_forecast.py`)

**Files:**
- Create: `src/analytics/ml_forecast.py`
- Modify: `src/analytics/__init__.py` to expose forecast functions
- Test: `tests/test_ml_forecast.py`

**Interfaces:**
- Consumes: `src/analytics/ttm_engine.py` (historical TTM series), `src/analytics/wacc_engine.py` (discount rates)
- Produces: `forecast_revenue_growth`, `forecast_margin`, `forecast_revenue_interval` functions with confidence intervals

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_forecast_revenue_growth_returns_interval():
    growth, lower, upper = forecast_revenue_growth(series, horizon=5)
    assert lower < growth < upper, "Forecast should be within interval"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ml_forecast.py::test_forecast_revenue_growth_returns_interval -v`
Expected: FAIL with "forecast_revenue_growth not defined"
- [ ] **Step 2: Implement `forecast_revenue_growth(series: pd.Series, horizon: int) -> Tuple[float, float, float]` in `src/analytics/ml_forecast.py`**
Approach: use Holt-Winters Exponential Smoothing on TTM revenue series, return median forecast and 80% confidence interval.
- [ ] **Step 3: Run test to verify it passes**
Run: `pytest tests/test_ml_forecast.py::test_forecast_revenue_growth_returns_interval -v`
Expected: PASS
- [ ] **Step 4: Commit**
```bash
git add src/analytics/ml_forecast.py tests/test_ml_forecast.py
git commit -m "feat: add ML & Statistical Growth & Margin Forecasting Engine"
```

### Task 4: Valuation Backtesting & Prediction Error Attribution Engine (`src/analytics/backtest_engine.py`)

**Files:**
- Create: `src/analytics/backtest_engine.py`
- Modify: `src/analytics/__init__.py` to expose backtest functions
- Test: `tests/test_backtest_engine.py`

**Interfaces:**
- Consumes: `src/analytics/dcf.py` (historical DCF model), `src/analytics/ml_forecast.py` (forecasted growth)
- Produces: `backtest_results` with bias correction coefficient, error metrics, and calibrated forecasts

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_backtest_bias_correction_applies():
    corrected = apply_bias_correction(base_forecast, historical_error)
    assert corrected != base_forecast, "Bias correction should modify forecast"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_backtest_engine.py::test_backtest_bias_correction_applies -v`
Expected: FAIL with "apply_bias_correction not defined"
- [ ] **Step 3: Implement `apply_bias_correction(base_forecast: float, historical_error: float) -> float` in `src/analytics/backtest_engine.py`**
Approach: compute historical bias (e.g., average overestimation), apply as multiplicative correction factor to base forecast.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_backtest_engine.py::test_backtest_bias_correction_applies -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/analytics/backtest_engine.py tests/test_backtest_engine.py
git commit -m "feat: add Valuation Backtesting & Prediction Error Attribution Engine"
```

### Task 5: Macro & Industry Cyclicality Overlay (`src/analytics/macro_overlay.py`)

**Files:**
- Create: `src/analytics/macro_overlay.py`
- Modify: `src/analytics/__init__.py` to expose macro adjustment functions
- Test: `tests/test_macro_overlay.py`

**Interfaces:**
- Consumes: `src/analytics/wacc_engine.py` (WACC), `src/analytics/ml_forecast.py` (growth rates)
- Produces: `adjust_wacc_for_macro`, `adjust_growth_for_macro` functions applying macro factor multipliers

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_macro_adjustment_applies_to_wacc():
    adjusted = adjust_wacc_for_macro(wacc, inflation=0.05, interest_rate=0.07)
    assert adjusted > wacc, "WACC should increase with inflation and rates"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_macro_overlay.py::test_macro_adjustment_applies_to_wacc -v`
Expected: FAIL with "adjust_wacc_for_macro not defined"
- [ ] **Step 3: Implement `adjust_wacc_for_macro(wacc: float, inflation: float, interest_rate: float) -> float` in `src/analytics/macro_overlay.py`**
Approach: increase WACC by inflation and interest rate components proportionally.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_macro_overlay.py::test_macro_adjustment_applies_to_wacc -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/analytics/macro_overlay.py tests/test_macro_overlay.py
git commit -m "feat: add Macro & Industry Cyclicality Overlay"
```

### Integration Tasks

#### Task 6: Integrate WACC Engine into DCF Model (`src/equity_research/dcf.py`)

**Files:**
- Modify: `src/equity_research/dcf.py` to use `calculate_wacc` from `wacc_engine`
- Test: `tests/test_dcf_integration.py`

**Interfaces:**
- Consumes: `calculate_wacc` function
- Produces: Updated `DCFInputs` with dynamic WACC

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_dcf_uses_dynamic_wacc():
    result = calculate_multi_model_valuation(inputs, current_price)
    assert result.wacc_used == calculate_wacc(...), "DCF should use dynamic WACC"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_dcf_integration.py::test_dcf_uses_dynamic_wacc -v`
Expected: FAIL with "calculate_wacc not integrated"
- [ ] **Step 2: Modify `calculate_multi_model_valuation` to fetch WACC via `wacc_engine.calculate_wacc`**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_dcf_integration.py::test_dcf_uses_dynamic_wacc -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/equity_research/dcf.py
git commit -m "feat: integrate Dynamic WACC into DCF model"
```

#### Task 7: Integrate TTM Engine into Data Access (`src/equity_research/data_access.py`)

**Files:**
- Modify: `src/equity_research/data_access.py` to replace stale annual data with `ttm_engine.aggregate_ttm_*` in `fetch_research_bundle`
- Test: `tests/test_data_access_integration.py`

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_fetch_research_bundle_uses_ttm():
    bundle = fetch_research_bundle(ticker)
    assert bundle.fundamental.ebitda_margin is not None, "TTM data should be used"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_data_access_integration.py::test_fetch_research_bundle_uses_ttm -v`
Expected: FAIL with "TTM not integrated"
- [ ] **Step 3: Update `fetch_research_bundle` to call `ttm_engine.aggregate_ttm_income` and use TTM metrics**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_data_access_integration.py::test_fetch_research_bundle_uses_ttm -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/equity_research/data_access.py
git commit -m "feat: integrate Quarterly TTM into data access"
```

#### Task 8: Add UI Panels for Institutional Accuracy (`src/ui/pages/equity_research.py`)

**Files:**
- Modify: `src/ui/pages/equity_research.py` to add tabs/panels for WACC breakdown, TTM vs. annual, ML forecast intervals, backtest accuracy, macro overlay
- Test: `tests/test_ui_integration.py` (UI rendering)

**Steps:**
- [ ] **Step 1: Write the failing test**
```python
def test_equity_research_shows_wacc_breakdown():
    # Simulate app start and check for WACC panel presence
    assert "WACC Breakdown" in page_elements, "WACC panel missing"
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ui_integration.py::test_equity_research_shows_wacc_breakdown -v`
Expected: FAIL with "panel not rendered"
- [ ] **Step 3: Add Streamlit components (metric cards, charts) for each engine in `equity_research.py`**
Approach: use existing UI components, add new sections under appropriate tabs.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ui_integration.py::test_equity_research_shows_wacc_breakdown -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add src/ui/pages/equity_research.py
git commit -m "feat: add Institutional Accuracy UI panels"
```

#### Task 9: Write Comprehensive Unit Tests (`tests/test_institutional_accuracy.py`)

**Files:**
- Create: `tests/test_institutional_accuracy.py`
- Ensure all modules are covered

**Steps:**
- [ ] **Step 1: Write the failing test skeleton**
```python
def test_all_modules_import_successfully():
    import src.analytics.wacc_engine
    import src.analytics.ttm_engine
    # ... import all modules
    assert True  # placeholder
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_institutional_accuracy.py::test_all_modules_import_successfully -v`
Expected: FAIL with "module not found"
- [ ] **Step 3: Implement individual tests for each module based on their own task steps**
- [ ] **Step 4: Run full test suite to verify all pass**
Run: `pytest tests/test_institutional_accuracy.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add tests/test_institutional_accuracy.py
git commit -m "feat: add comprehensive unit tests for institutional accuracy modules"
```

### Final Steps

- Run full test suite to ensure 185/185 tests pass.
- Commit and push changes to GitHub.