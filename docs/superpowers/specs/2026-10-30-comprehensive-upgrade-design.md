# Quantfolio Comprehensive Institutional Upgrade Design

## Overview
This design covers the implementation of four major institutional modules:
1. **PDF & Excel Export Engine**: Professional PDF research tear sheets and multi-tab Excel financial models (`XlsxWriter`).
2. **Portfolio Backtesting Engine**: Historical strategy backtester with performance metrics (Sharpe, Sortino, Max Drawdown, Alpha, Beta).
3. **ML Signal Enhancements**: Walk-forward cross-validation forecasting models.
4. **Monte Carlo Simulation**: Path simulation for portfolio risk and return distributions.

## Architecture
- `src/reporting/pdf_export.py`: ReportLab / headless browser PDF generation.
- `src/analytics/backtest.py`: Backtesting engine with cash flow tracking and rebalancing.
- `src/analytics/ml_signal.py`: Enhanced ML signal pipeline.
- `src/analytics/montecarlo.py`: Monte Carlo simulation runner.
