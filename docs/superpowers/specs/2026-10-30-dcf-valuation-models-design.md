# Architectural Design: Comprehensive DCF, FCFE, DDM, and Gordon Growth Valuation Models in Quantfolio Equity Research

## 1. Overview & Objectives
This design specifies the implementation of a institutional-grade valuation suite and research reporting upgrade within **Quantfolio**, integrating:
- **Core Financial Projections**: Sales (Revenue), Gross Profit, EBITDA, EBIT, and Net Profit across historical and projected horizons.
- **Multi-Model Valuation Framework**:
  - **Free Cash Flow to Firm (FCFF)** model.
  - **Free Cash Flow to Equity (FCFE)** model.
  - **Dividend Discount Model (DDM)** (Gordon Growth / Multi-stage).
  - **Gordon Growth Model** (perpetuity valuation).
  - **Premium / Discount Analysis** (Implied valuation vs Current Market Price, Target Upside/Margin of Safety).
- **Quantfolio & Research Report Integration**: Streamlit UI panels, export engines (HTML/PDF/Excel), and professional research report generation.

---

## 2. Component Architecture

### A. Analytics Module (`src/analytics/dcf.py` & new valuation helpers)
- **FCFF**: $EV = \sum \frac{FCFF_t}{(1+WACC)^t} + \frac{TerminalValue}{(1+WACC)^n}$, $EquityValue = EV - NetDebt$, $ValuePerShare = \frac{EquityValue}{Shares}$.
- **FCFE**: $EquityValue = \sum \frac{FCFE_t}{(1+Ke)^t} + \frac{TerminalValue_{equity}}{(1+Ke)^n}$, where $FCFE = NetIncome + D\&A - Capex - \Delta NWC + NetBorrowing$.
- **DDM**: $ValuePerShare = \frac{D_1}{Ke - g}$ (Gordon Growth) or multi-stage dividend discount model based on sustainable growth $g = ROE \times (1 - Payout)$.
- **Gordon Growth Model**: General perpetuity formulation for cash flows / dividends.
- **Financial Metric Breakdown**: Sales, Gross Profit, EBITDA, EBIT, Net Profit explicit projections.

### B. Equity Research Report (`src/equity_research/ai_report.py` & templates)
- Add valuation reconciliation table showing FCFF, FCFE, DDM, and Gordon Growth intrinsic values alongside CMP and premium/discount percentage.
- Detailed financial forecast table featuring Sales, Gross Profit, EBITDA, EBIT, and Net Profit.

### C. UI & Export (`src/ui/pages/equity_research.py` & `src/equity_research/export.py`)
- Interactive valuation dashboard with selector for valuation model (FCFF, FCFE, DDM, Gordon Growth, or Blended SOTP).
- Excel workbook export tabs with complete DCF / FCFE / DDM schedules.
- Standalone HTML / PDF tear sheets.

---

## 3. Verification Plan
- Unit tests in `tests/` verifying mathematical correctness of FCFF, FCFE, DDM, and Gordon Growth implementations against known analytical benchmarks.
- Verification instructions for users to confirm input correctness.
