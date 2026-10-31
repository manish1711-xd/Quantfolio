"""Valuation Backtesting & Prediction Error Attribution Engine.

Implements a closed-loop ground-truth calibration feedback loop:
- Point-in-time historical DCF backtesting against actual stock price realization
- Systematic error and bias attribution (overestimation vs underestimation)
- Multiplicative Error Correction Coefficient applied to future model forecasts
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def evaluate_historical_prediction(
    intrinsic_prediction: float,
    realized_price: float,
) -> dict[str, Any]:
    """Evaluate a single historical valuation prediction against realized outcome."""
    pred = float(intrinsic_prediction) if intrinsic_prediction and not np.isnan(intrinsic_prediction) else 0.0
    actual = float(realized_price) if realized_price and not np.isnan(realized_price) else 0.0

    if actual <= 0 or pred <= 0:
        return {
            "error_pct": 0.0,
            "absolute_error": 0.0,
            "bias_direction": "neutral",
        }

    abs_err = abs(pred - actual)
    error_pct = abs_err / actual
    diff = pred - actual
    direction = "overpredicted" if diff > 0 else ("underpredicted" if diff < 0 else "exact")

    return {
        "error_pct": round(error_pct, 4),
        "absolute_error": round(abs_err, 2),
        "bias_direction": direction,
    }


def calculate_bias_correction_factor(
    predictions: list[float],
    actuals: list[float],
) -> float:
    """Calculate empirical error correction factor from historical backtests.

    Factor = Mean(Actual / Prediction). If model consistently overpredicts,
    Actual / Prediction < 1.0, which scales future forecasts downward.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 1.0

    ratios: list[float] = []
    for p, a in zip(predictions, actuals):
        if p > 0 and a > 0 and not np.isnan(p) and not np.isnan(a):
            ratios.append(a / p)

    if not ratios:
        return 1.0

    mean_ratio = float(np.mean(ratios))
    # Bound correction factor between 0.75 and 1.25 for stability
    return float(max(0.75, min(1.25, mean_ratio)))


def apply_bias_correction(base_forecast: float, bias_factor: float) -> float:
    """Apply empirical bias correction coefficient to a base forecast value."""
    bf = float(base_forecast) if base_forecast and not np.isnan(base_forecast) else 0.0
    factor = float(bias_factor) if bias_factor and not np.isnan(bias_factor) else 1.0
    return float(bf * factor)
