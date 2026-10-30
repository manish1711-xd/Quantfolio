# DCF, FCFE, DDM, and Gordon Growth Valuation Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement comprehensive valuation models (FCFF, FCFE, DDM, Gordon Growth) and financial metric projections (Sales, Gross Profit, EBITDA, EBIT, Net Profit, Premium/Discount vs CMP) in Quantfolio's equity research engine and reporting modules.

**Architecture:** Extend `src/analytics/dcf.py` to support FCFE, DDM, and Gordon Growth valuation functions, expand `src/analytics/company.py` and `src/equity_research/` to compute and structure financial statement metrics (Sales, Gross Profit, EBITDA, EBIT, Net Profit), and update research reports, UI panels, and export engines.

**Tech Stack:** Python 3.13, Pandas, NumPy, SciPy, Streamlit, XlsxWriter.

**Spec:** `docs/superpowers/specs/2026-10-30-dcf-valuation-models-design.md`

## Global Constraints
- Python 3.13 compatibility with pandas < 3.0 pin.
- Pure analytics modules (`src/analytics/`) must have zero Streamlit or network imports.
- Missing data propagates as `NaN` (rendered as em-dash in UI/reports), never zero or fabricated numbers.

## Review Focus
1. **Negative FCFE / Dividends**: How models handle negative cash flows or zero dividends (must return NaN or valid fallback notices rather than crashing).
2. **Cost of Equity vs WACC alignment**: Ensuring FCFE and DDM properly discount at Cost of Equity ($Ke$), while FCFF discounts at $WACC$.
3. **Gordon Growth Convergence**: Preventing division by zero or negative denominators when $Ke$ or $WACC \le g$.
4. **Unit consistency**: Ensuring margins, growth rates, and percentages are consistently represented as decimals.
5. **Statement metric mapping**: Ensuring Sales, Gross Profit, EBITDA, EBIT, and Net Profit are correctly extracted from statements across varying corporate financial statement formats.

---

## Tasks

### Task 1: Valuation Engine Extensions (`src/analytics/dcf.py`)

**Files:**
- Modify: `Quantfolio/src/analytics/dcf.py`
- Test: `Quantfolio/tests/test_dcf_models.py`

**Interfaces:**
- Consumes: Existing `CompanyFinancials`, `DcfAssumptions`, `DcfResult`, `wacc`, `cost_of_equity`.
- Produces: `run_fcfe()`, `run_ddm()`, `run_gordon()`, enhanced `CompanyFinancials` including Sales, Gross Profit, EBITDA, EBIT, Net Profit.

- [ ] **Step 1: Write failing tests for FCFE, DDM, and Gordon Growth models in `Quantfolio/tests/test_dcf_models.py`**

```python
from src.analytics.dcf import DcfAssumptions, CompanyFinancials, run_fcfe, run_ddm, run_gordon

def test_fcfe_valuation():
    fin = CompanyFinancials(base_revenue=1000.0, net_debt=200.0, shares=10.0, current_price=100.0)
    assump = DcfAssumptions.constant(years=3, revenue_growth=0.08, ebit_margin=0.20)
    res = run_fcfe(assump, fin, cost_of_equity=0.10)
    assert res is not None
    assert not math.isnan(res.value_per_share)

def test_ddm_valuation():
    res = run_ddm(current_dividend=5.0, growth_rate=0.05, cost_of_equity=0.10, current_price=100.0)
    assert res is not None
    assert res.value_per_share == 105.0 # 5 * (1+0.05) / (0.10 - 0.05) = 5.25 / 0.05 = 105
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest Quantfolio/tests/test_dcf_models.py -v`
Expected: FAIL due to missing functions `run_fcfe`, `run_ddm`, `run_gordon`.

- [ ] **Step 3: Implement `run_fcfe`, `run_ddm`, and `run_gordon` in `Quantfolio/src/analytics/dcf.py`**

Add FCFE projection and discounting, DDM Gordon Growth formula, and standalone Gordon Growth model with robust NaN guards and clear result dataclasses.

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest Quantfolio/tests/test_dcf_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Quantfolio/src/analytics/dcf.py Quantfolio/tests/test_dcf_models.py
git commit -m "feat: add FCFE, DDM, and Gordon Growth models to analytics dcf"
```

---

### Task 2: Equity Research Report & Exporters (`src/equity_research/` & `src/reporting/`)

**Files:**
- Modify: `Quantfolio/src/equity_research/ai_report.py`
- Modify: `Quantfolio/src/equity_research/export.py`
- Test: `Quantfolio/tests/test_equity_research.py`

**Interfaces:**
- Consumes: `src.analytics.dcf` valuation results (FCFF, FCFE, DDM, Gordon Growth, Premium/Discount).
- Produces: Enhanced equity research markdown reports, HTML tearsheets, and Excel workbook sheets featuring Sales, Gross Profit, EBITDA, EBIT, Net Profit, and Multi-Model Valuations.

- [ ] **Step 1: Write test for equity research multi-model valuation generation**

```python
from src.equity_research.ai_report import generate_research_report

def test_research_report_valuation_models():
    # Test that research report incorporates FCFF, FCFE, DDM, Gordon Growth and metrics
    ...
```

- [ ] **Step 2: Implement multi-model valuation and financial metrics table in `ai_report.py` and `export.py`**

- [ ] **Step 3: Run tests to verify passing**

Run: `pytest Quantfolio/tests/test_equity_research.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Quantfolio/src/equity_research/ai_report.py Quantfolio/src/equity_research/export.py Quantfolio/tests/test_equity_research.py
git commit -m "feat: integrate FCFE, DDM, Gordon growth and key metrics in research reports and exporters"
```

---

### Task 3: Streamlit UI Integration (`Quantfolio/src/ui/pages/research.py`)

**Files:**
- Modify: `Quantfolio/src/ui/pages/research.py`

**Interfaces:**
- Consumes: Updated analytics and reporting modules.
- Produces: Streamlit UI sections displaying FCFF, FCFE, DDM, Gordon Growth intrinsic values, premium/discount values, and Sales/Gross Profit/EBITDA/EBIT/Net Profit breakdown.

- [ ] **Step 1: Update Streamlit research page to render FCFF, FCFE, DDM, Gordon Growth, and financial metrics**

- [ ] **Step 2: Run test suite to verify no regressions**

Run: `pytest Quantfolio/tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add Quantfolio/src/ui/pages/research.py
git commit -m "feat: add multi-model valuation and financial metrics to equity research UI page"
```
