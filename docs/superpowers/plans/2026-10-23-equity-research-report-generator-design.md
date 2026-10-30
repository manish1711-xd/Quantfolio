# Equity Research Report Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete AI-powered Stock Equity Research Report Generator using Python and Streamlit, integrated with the existing Quantfolio project architecture. The user enters a stock ticker and gets a professional equity research report with fundamental analysis, technical analysis, DCF valuation, peer comparison, and AI-generated content.

**Architecture:** This extension follows Quantfolio's strict architectural rules: computation never imports Streamlit, every network call lives in `src/data/`, and pure math lives in `src/analytics/`. The new equity research modules will be packaged under `src/equity_research/` as a sibling to the existing packages, sharing `src/config/` conventions and `src/models/` dataclasses. A new `src/ui/data_access.py` will handle TTL-cached data fetching, and the UI layer in `src/ui/pages/` will render the report.

**Tech Stack:** Python 3.11+, Pandas, NumPy, yfinance, Plotly, Streamlit 1.50+, SQLite (existing), XlsxWriter for Excel export, an LLM API (OpenAI-compatible) for AI-generated content.

**Spec:** This plan implements the feature described in the user's request. No separate spec file needed — the user's requirements serve as both spec and design brief.

## Global Constraints (copied verbatim from project context)

- Computation never imports Streamlit (pure modules in `src/analytics/`, `src/data/`, `src/models/`)
- Every network call lives in `src/data/` (fetcher module, no urllib outside it)
- yfinance is a potentially incomplete data source; use reliable filings or alternative APIs when configured
- API keys in .env, provide .env.example, never hardcode secrets
- Treat yfinance as potentially incomplete; label Buy/Hold/Sell as model-generated assessments
- All financial formulas transparent; prevent look-ahead bias
- Handle missing data honestly; never force price target or investment rating when inputs insufficient
- 252 trading days convention, ddof=1 for risk statistics, ddof=0 for Bollinger Bands
- Minimum 30 observations before risk statistics are emitted as reliable
- DEFAULT_VAR_CONFIDENCE: 0.95
- Valid periods: ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")

## Review Focus

- **Missing financial data**: When yfinance returns incomplete statements, the app should report which fields are available and which are missing, rather than fabricating values
- **Insufficient inputs for valuation**: DCF and relative valuation should show warnings when free cash flow or comparables data is insufficient, never force a price target
- **Look-ahead bias**: All technical indicators and features must use only data available at or before the current bar (Wilder smoothing, forward-looking windows respected)
- **yfinance rate limiting**: Batch fetches return (results, errors) pairs; one dead ticker degrades to a warning
- **LLM API unavailability**: AI-generated report sections should have fallback text stating "AI generation unavailable" rather than blank sections or fabricated content
- **Delisted/unpriceable symbols**: UI should display appropriate warnings rather than crashing

---