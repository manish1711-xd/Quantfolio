"""Plotly figure builders.

One flat namespace over nine modules, so page code reads
``charts.efficient_frontier_chart(...)`` instead of threading imports through
every page. :mod:`src.charts.theme` stays reachable as ``charts.theme`` for the
palette and layout helpers.

Nothing in this package imports Streamlit — every figure can be built in a
notebook, a test, or the HTML report writer.
"""

from __future__ import annotations

from src.charts import theme
from src.charts.allocation import (
    allocation_donut,
    allocation_sunburst,
    allocation_treemap,
    sector_bar,
    weight_bar,
    weight_vs_target_bar,
)
from src.charts.correlation import (
    correlation_heatmap,
    dendrogram,
    pca_loadings_heatmap,
    pca_scree,
    rolling_correlation_chart,
)
from src.charts.forecast import (
    backtest_scatter,
    feature_importance_chart,
    forecast_cone_chart,
    horizon_skill_chart,
    model_leaderboard_chart,
    prediction_timeline_chart,
    residual_distribution_chart,
)
from src.charts.optimizer import (
    efficient_frontier_chart,
    frontier_weights_area,
    weights_comparison,
    weights_donut,
)
from src.charts.performance import (
    benchmark_comparison,
    contribution_waterfall,
    cumulative_return_chart,
    growth_line,
    monthly_heatmap,
    pnl_bar,
    relative_strength_chart,
    return_bar,
    value_bar,
)
from src.charts.research import (
    common_size_chart,
    financial_trend_chart,
    peer_median_chart,
    scorecard_breakdown_chart,
    sensitivity_heatmap,
    sentiment_scatter,
    volatility_regime_chart,
)
from src.charts.risk import (
    drawdown_chart,
    drawdown_episodes_chart,
    return_distribution,
    risk_contribution_chart,
    risk_gauge,
    risk_return_scatter,
    rolling_chart,
    rolling_multi_chart,
    stress_bar,
    underwater_chart,
    var_backtest_chart,
)
from src.charts.simulation import fan_chart, sample_paths_chart, terminal_distribution
from src.charts.statements import balance_sheet_chart
from src.charts.technicals import (
    macd_chart,
    price_chart,
    price_with_panels,
    range_52w_bar,
    rsi_chart,
    volume_chart,
)

# Convenience aliases used by the page layer.
empty_figure = theme.empty_figure
set_mode = theme.set_mode

__all__ = [
    "theme",
    "empty_figure",
    "set_mode",
    # allocation
    "allocation_donut",
    "allocation_sunburst",
    "allocation_treemap",
    "sector_bar",
    "weight_bar",
    "weight_vs_target_bar",
    # performance
    "benchmark_comparison",
    "contribution_waterfall",
    "cumulative_return_chart",
    "growth_line",
    "monthly_heatmap",
    "pnl_bar",
    "relative_strength_chart",
    "return_bar",
    "value_bar",
    # risk
    "drawdown_chart",
    "drawdown_episodes_chart",
    "return_distribution",
    "risk_contribution_chart",
    "risk_gauge",
    "risk_return_scatter",
    "rolling_chart",
    "rolling_multi_chart",
    "stress_bar",
    "underwater_chart",
    "var_backtest_chart",
    # correlation & factors
    "correlation_heatmap",
    "dendrogram",
    "pca_loadings_heatmap",
    "pca_scree",
    "rolling_correlation_chart",
    # forecast
    "backtest_scatter",
    "feature_importance_chart",
    "forecast_cone_chart",
    "horizon_skill_chart",
    "model_leaderboard_chart",
    "prediction_timeline_chart",
    "residual_distribution_chart",
    # optimiser
    "efficient_frontier_chart",
    "frontier_weights_area",
    "weights_comparison",
    "weights_donut",
    # statements
    "balance_sheet_chart",
    # simulation
    "fan_chart",
    "sample_paths_chart",
    "terminal_distribution",
    # company research
    "common_size_chart",
    "financial_trend_chart",
    "peer_median_chart",
    "scorecard_breakdown_chart",
    "sensitivity_heatmap",
    "sentiment_scatter",
    "volatility_regime_chart",
    # technicals
    "macd_chart",
    "price_chart",
    "price_with_panels",
    "range_52w_bar",
    "rsi_chart",
    "volume_chart",
]
