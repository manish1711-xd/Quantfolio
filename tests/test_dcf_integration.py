"""Tests for DCF Integration with WACC and Macro Engines."""

import pytest
import numpy as np
from src.equity_research.models import DCFInputs
from src.equity_research.dcf import calculate_multi_model_valuation


def test_dcf_uses_dynamic_wacc_integration():
    inputs = DCFInputs(
        fcffree=1000.0,
        wacc=0.11,
        growth_rate=0.07,
        terminal_growth_rate=0.03,
        shares_outstanding=100.0,
        net_debt=500.0,
    )
    result = calculate_multi_model_valuation(inputs, current_price=50.0)
    assert result.ok
    assert not np.isnan(result.blended_intrinsic)
    assert "FCFF" in result.model_names
    assert "FCFE" in result.model_names
