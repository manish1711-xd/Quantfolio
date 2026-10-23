import numpy as np
import pandas as pd
import pytest

from src.equity_research.dcf import (
    project_free_cash_flows,
    compute_terminal_value,
    calculate_dcf_valuation,
    generate_sensitivity_matrix,
    generate_scenario_analysis,
)
from src.equity_research.models import DCFInputs


def test_project_free_cash_flows():
    # Base FCF = 100, growth = 10%, 5 years
    fcf_proj = project_free_cash_flows(base_fcf=100.0, growth_rate=0.10, years=5)
    assert len(fcf_proj) == 5
    assert np.isclose(fcf_proj[0], 110.0)
    assert np.isclose(fcf_proj[1], 121.0)
    assert np.isclose(fcf_proj[4], 100.0 * (1.10 ** 5))


def test_compute_terminal_value_valid():
    # Final FCF = 161.051, WACC = 10%, terminal growth = 3%
    # TV = 161.051 * 1.03 / (0.10 - 0.03) = 165.8825 / 0.07 = 2369.75
    tv = compute_terminal_value(final_fcf=161.051, wacc=0.10, terminal_growth=0.03)
    expected = (161.051 * 1.03) / (0.10 - 0.03)
    assert np.isclose(tv, expected)


def test_compute_terminal_value_wacc_le_growth():
    # When WACC <= terminal growth, denominator <= 0, TV should return np.nan
    tv = compute_terminal_value(final_fcf=100.0, wacc=0.03, terminal_growth=0.03)
    assert np.isnan(tv)
    tv_neg = compute_terminal_value(final_fcf=100.0, wacc=0.02, terminal_growth=0.03)
    assert np.isnan(tv_neg)


def test_calculate_dcf_valuation_full():
    inputs = DCFInputs(
        fcffree=1000.0,
        wacc=0.10,
        growth_rate=0.08,
        terminal_growth_rate=0.03,
        shares_outstanding=100.0,
        net_debt=2000.0,
        projection_years=5,
    )
    res = calculate_dcf_valuation(inputs, current_price=150.0)
    assert res.intrinsic_value is not None
    assert res.intrinsic_value > 0
    assert res.fair_value_low is not None
    assert res.fair_value_high is not None
    assert res.fair_value_low < res.intrinsic_value < res.fair_value_high
    assert res.recommendation in ["Buy", "Hold", "Sell"]
    assert "year_1" in res.dcf_interpolated


def test_calculate_dcf_valuation_missing_or_negative_inputs():
    inputs_neg = DCFInputs(
        fcffree=-500.0,
        wacc=0.10,
        growth_rate=0.05,
    )
    res = calculate_dcf_valuation(inputs_neg, current_price=100.0)
    assert res.intrinsic_value is None
    assert res.recommendation == "Insufficient data"
    assert any("negative" in note.lower() or "insufficient" in note.lower() for note in res.status_notes)


def test_generate_sensitivity_matrix():
    matrix = generate_sensitivity_matrix(
        base_fcf=1000.0,
        base_wacc=0.10,
        base_growth=0.08,
        terminal_growth=0.03,
        shares_outstanding=100.0,
        net_debt=2000.0,
    )
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] > 1
    assert matrix.shape[1] > 1


def test_generate_scenario_analysis():
    scenarios = generate_scenario_analysis(
        base_fcf=1000.0,
        base_wacc=0.10,
        base_growth=0.08,
        terminal_growth=0.03,
        shares_outstanding=100.0,
        net_debt=2000.0,
        current_price=150.0,
    )
    assert "Base" in scenarios
    assert "Bull" in scenarios
    assert "Bear" in scenarios
    assert scenarios["Bear"]["intrinsic_value"] < scenarios["Base"]["intrinsic_value"] < scenarios["Bull"]["intrinsic_value"]
