"""Portfolio analytics — every formula in the application lives here.

Twenty-three pure modules, no I/O and no Streamlit, which is what makes each
one directly unit-testable offline against hand-computed values.

============== =============================================================
module         responsibility
============== =============================================================
positions      position P&L, weights, FX conversion, portfolio summary
lots           FIFO tax lots, realised vs unrealised P&L, holding periods
returns        XIRR, time-weighted return, Modified Dietz, monthly tables
risk           volatility, Sharpe/Sortino/Calmar/Omega, VaR/CVaR, risk score
drawdown       drawdown episodes with peak/trough/recovery dates
rolling        rolling volatility, Sharpe, beta, correlation, VaR
benchmark      beta, alpha, tracking error, information ratio, capture ratios
attribution    return contribution and marginal/component risk contribution
black_litterman  equilibrium returns blended with explicit investor views
optimize       mean-variance frontier, min-variance, max-Sharpe, risk parity
options        Black-Scholes, Greeks, covered-call / protective-put overlays
montecarlo     GBM, bootstrap and Student-t forward simulation
stress         historical scenario replay and beta-implied shocks
tax            Indian STCG/LTCG rules, set-off waterfall, loss harvesting
factors        PCA of returns, correlation-distance clustering (unsupervised)
factor_model   Fama-French style regression on named factors, HAC errors
technicals     SMA/EMA, Wilder RSI, MACD, Bollinger bands, ATR
features       leak-free, scale-free supervised feature matrix built from a
               price history, plus the forward-log-return label for it
forecast       supervised forward-return models, purged walk-forward
               validation against a random walk, conformal intervals
rebalance      target weights, drift bands, executable trade list
insights       rule-based plain-English observations with cited evidence
============== =============================================================

Import the submodules directly (``from src.analytics import risk``) rather than
re-exporting every symbol here: the module name is meaningful context at the
call site, and ``risk.value_at_risk`` reads better than a bare
``value_at_risk``.
"""

from __future__ import annotations

__all__ = [
    "attribution",
    "benchmark",
    "black_litterman",
    "drawdown",
    "factor_model",
    "factors",
    "features",
    "forecast",
    "fundamentals",
    "insights",
    "lots",
    "montecarlo",
    "news",
    "optimize",
    "options",
    "positions",
    "rebalance",
    "returns",
    "risk",
    "rolling",
    "stress",
    "tax",
    "technicals",
]
