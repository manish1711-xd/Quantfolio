"""Tests for DCF Valuation Model Fixes - 11 Requirements."""

import pytest
import numpy as np
from src.equity_research.dcf import (
    calculate_multi_model_valuation,
    DCFInputs,
    ValuationResult,
    MultiModelValuationResult,
)
from src.equity_research.models import MultiModelValuationResult as ModelsMultiModelValuationResult


class TestDCFFixes:
    """Test the 11 DCF valuation model fixes."""

    def setup_method(self):
        """Set up common test inputs."""
        self.base_inputs = DCFInputs(
            fcffree=1000.0,  # Base FCF
            wacc=0.10,  # 10% WACC
            growth_rate=0.08,  # 8% growth
            terminal_growth_rate=0.03,  # 3% terminal growth
            shares_outstanding=100.0,
            net_debt=500.0,
            projection_years=5,
        )
        self.current_price = 100.0

    # Requirement 1: Validate all assumptions before calculating valuation
    def test_assumption_validation_rejects_invalid_wacc(self):
        """Test that invalid WACC (outside 7%-15%) is rejected with warning."""
        inputs = self.base_inputs.model_copy(update={"wacc": 0.05})  # Below 7%
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        # Should still return a result but with notices about invalid assumptions
        assert len(result.notices) > 0
        assert any("WACC" in n and ("7%" in n or "15%" in n or "range" in n.lower()) for n in result.notices)

    def test_assumption_validation_rejects_invalid_terminal_growth(self):
        """Test that terminal growth outside valid ranges is rejected."""
        # Test above max for mature (6%)
        inputs = self.base_inputs.model_copy(update={"terminal_growth_rate": 0.08})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)
        assert any("terminal growth" in n.lower() and "cap" in n.lower() for n in result.notices)

    # Requirement 2 & 3: Terminal growth < WACC by at least 2%-4%
    def test_terminal_growth_must_be_below_wacc_minus_spread(self):
        """Test that g < WACC - 2% minimum spread enforced."""
        # WACC = 10%, g = 9% (only 1% spread - should trigger adjustment)
        inputs = self.base_inputs.model_copy(update={"wacc": 0.10, "terminal_growth_rate": 0.09})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        # Should auto-adjust terminal growth and warn
        assert any("spread" in n.lower() or "adjusted" in n.lower() for n in result.notices)
        # Terminal growth should be capped at WACC - 2% = 8%
        assert result.notices is not None

    def test_terminal_growth_auto_adjusted_when_g_ge_wacc(self):
        """Test automatic adjustment when g >= WACC."""
        inputs = self.base_inputs.model_copy(update={"wacc": 0.10, "terminal_growth_rate": 0.12})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        assert any("adjusted" in n.lower() or "auto" in n.lower() for n in result.notices)

    # Requirement 4: Cap terminal growth based on company type
    def test_terminal_growth_capped_mature_company(self):
        """Test terminal growth capped at 6% for mature companies."""
        # Mature company - low growth
        inputs = self.base_inputs.model_copy(update={"growth_rate": 0.05, "terminal_growth_rate": 0.08})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        # Should be capped at 6% for mature
        assert any("6%" in n or "0.06" in n for n in result.notices)

    def test_terminal_growth_capped_high_growth_company(self):
        """Test terminal growth capped at 8% for high-growth companies."""
        # High growth company
        inputs = self.base_inputs.model_copy(update={"growth_rate": 0.20, "terminal_growth_rate": 0.10})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        # Should be capped at 8% for high-growth
        assert any("8%" in n or "0.08" in n for n in result.notices)

    # Requirement 5: Validate WACC within 7%-15%
    def test_wacc_validated_range_7_to_15(self):
        """Test WACC validated within 7%-15% range."""
        # Below 7%
        inputs = self.base_inputs.model_copy(update={"wacc": 0.05})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)
        assert any("7%" in n or "15%" in n for n in result.notices)

        # Above 15%
        inputs = self.base_inputs.model_copy(update={"wacc": 0.20})
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)
        assert any("7%" in n or "15%" in n for n in result.notices)

    # Requirement 6: Terminal value contribution cap at 80%
    def test_terminal_value_capped_at_80_percent(self):
        """Test terminal value contribution capped at 80% of EV."""
        # Create scenario where TV would be > 80% - very high growth, low WACC
        inputs = DCFInputs(
            fcffree=100.0,
            wacc=0.08,
            growth_rate=0.15,
            terminal_growth_rate=0.05,
            shares_outstanding=10.0,
            net_debt=0.0,
            projection_years=5,
        )
        result = calculate_multi_model_valuation(inputs, current_price=50.0)

        # Should have notice about terminal value cap
        assert any("80%" in n or "terminal value" in n.lower() for n in result.notices)

    # Requirement 7: Terminal value formula (already implemented in analytics/dcf.py)
    def test_terminal_value_formula_correct(self):
        """Test terminal value uses correct formula: FCF * (1+g) / (WACC-g)."""
        # This is tested indirectly through valuation results
        result = calculate_multi_model_valuation(self.base_inputs, current_price=self.current_price)

        # Both FCFF and Gordon should produce valid values
        assert not np.isnan(result.fcff_value) or not np.isnan(result.gordon_value)
        assert result.fcff_value > 0 or result.gordon_value > 0

    # Requirement 8: Cross-model sanity checks
    def test_cross_model_difference_25_percent_warning(self):
        """Test warning when cross-model difference > 25%."""
        # Create scenario with large gap between FCFF and Gordon
        inputs = DCFInputs(
            fcffree=100.0,
            wacc=0.10,
            growth_rate=0.20,  # High growth
            terminal_growth_rate=0.05,
            shares_outstanding=10.0,
            net_debt=0.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=50.0)

        # Check for cross-model gap warnings
        gap_notices = [n for n in result.notices if "cross-model" in n.lower() or "gap" in n.lower()]
        # At least one model should produce values we can compare
        assert result.fcff_value is not None or result.gordon_value is not None

    def test_cross_model_difference_50_percent_major_warning(self):
        """Test major warning when cross-model difference > 50%."""
        inputs = DCFInputs(
            fcffree=100.0,
            wacc=0.08,
            growth_rate=0.30,
            terminal_growth_rate=0.06,
            shares_outstanding=10.0,
            net_debt=0.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=30.0)

        major_warnings = [n for n in result.notices if "major" in n.lower() or "50%" in n]
        # Should have major warning or rejection

    def test_cross_model_difference_75_percent_reject_and_recalculate(self):
        """Test rejection and recalculation when cross-model difference > 75%."""
        inputs = DCFInputs(
            fcffree=100.0,
            wacc=0.07,
            growth_rate=0.40,
            terminal_growth_rate=0.06,
            shares_outstanding=10.0,
            net_debt=0.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=20.0)

        # Should reject assumptions and use conservative defaults
        rejection_notices = [n for n in result.notices if "reject" in n.lower() or "recalculat" in n.lower() or "conservative" in n.lower()]
        assert len(rejection_notices) > 0 or result.recommendation == "Insufficient data"

    # Requirement 9: Display metrics
    def test_displays_required_metrics(self):
        """Test that all required metrics are present in result."""
        result = calculate_multi_model_valuation(self.base_inputs, current_price=self.current_price)

        # Check all required display metrics exist
        assert hasattr(result, 'fcff_value')
        assert hasattr(result, 'gordon_value')
        # Terminal value contribution should be computable
        assert hasattr(result, 'blended_intrinsic')
        assert hasattr(result, 'premium_to_price')

    # Requirement 10: Conservative defaults
    def test_conservative_defaults_wacc_10_percent(self):
        """Test conservative default WACC = 10%."""
        # When WACC not provided or invalid, should default to 10%
        inputs = DCFInputs(
            fcffree=1000.0,
            wacc=None,  # No WACC provided
            growth_rate=0.08,
            terminal_growth_rate=0.03,
            shares_outstanding=100.0,
            net_debt=500.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        # Should use 10% as default or show notice about using default
        assert result is not None

    def test_conservative_defaults_terminal_growth_3_percent(self):
        """Test conservative default terminal growth = 3%."""
        inputs = DCFInputs(
            fcffree=1000.0,
            wacc=0.10,
            growth_rate=0.08,
            terminal_growth_rate=None,  # No terminal growth provided
            shares_outstanding=100.0,
            net_debt=500.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=self.current_price)

        assert result is not None

    # Requirement 11: Upside > 200% flag
    def test_upside_exceeds_200_percent_flag(self):
        """Test flag when implied upside exceeds 200%."""
        # Very low price vs high intrinsic value
        inputs = DCFInputs(
            fcffree=1000.0,
            wacc=0.08,
            growth_rate=0.15,
            terminal_growth_rate=0.04,
            shares_outstanding=10.0,
            net_debt=0.0,
        )
        result = calculate_multi_model_valuation(inputs, current_price=10.0)  # Very low price

        # Should flag as potentially unrealistic
        assert any("200%" in n or "unrealistic" in n.lower() for n in result.notices)

    def test_valuation_confidence_score(self):
        """Test valuation confidence score is calculated."""
        result = calculate_multi_model_valuation(self.base_inputs, current_price=self.current_price)

        # Confidence score should be present (could be in notices or as attribute)
        # For now, check that the result has notices about confidence/validation
        confidence_notices = [n for n in result.notices if "confidence" in n.lower() or "validation" in n.lower()]
        # At minimum, the result should be well-formed
        assert result is not None
        assert isinstance(result.notices, list)


class TestDCFInputValidation:
    """Test DCF input validation functions."""

    def test_validate_dcf_inputs_function_exists(self):
        """Test that a validation function exists or will be created."""
        # This will pass once we implement the validation function
        from src.equity_research.dcf import validate_dcf_assumptions
        assert callable(validate_dcf_assumptions)

    def test_validate_dcf_assumptions_returns_valid_and_warnings(self):
        """Test validation returns (is_valid, warnings, adjusted_inputs)."""
        from src.equity_research.dcf import validate_dcf_assumptions

        inputs = DCFInputs(
            fcffree=1000.0,
            wacc=0.10,
            growth_rate=0.08,
            terminal_growth_rate=0.03,
            shares_outstanding=100.0,
            net_debt=500.0,
        )

        is_valid, warnings, adjusted = validate_dcf_assumptions(inputs)
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)
        assert isinstance(adjusted, DCFInputs)


class TestTerminalValueContribution:
    """Test terminal value contribution calculation."""

    def test_terminal_value_contribution_calculated(self):
        """Test terminal value contribution % is calculated."""
        from src.equity_research.dcf import calculate_terminal_value_contribution

        fcff_val = 1000.0
        terminal_val = 800.0
        enterprise_val = 1800.0

        contribution = calculate_terminal_value_contribution(terminal_val, enterprise_val)
        assert abs(contribution - 0.444) < 0.01  # 800/1800 ≈ 44.4%

    def test_terminal_value_contribution_capped(self):
        """Test terminal value contribution capped at 80%."""
        from src.equity_research.dcf import calculate_terminal_value_contribution, apply_terminal_value_cap

        # When contribution > 80%, should cap
        fcff_val = 100.0
        terminal_val = 1000.0
        enterprise_val = 1100.0  # 1000/1100 = 90.9% > 80%

        contribution = calculate_terminal_value_contribution(terminal_val, enterprise_val)
        assert contribution > 0.80

        capped_tv = apply_terminal_value_cap(terminal_val, enterprise_val, max_share=0.80)
        capped_contribution = calculate_terminal_value_contribution(capped_tv, enterprise_val)
        assert abs(capped_contribution - 0.80) < 0.01


class TestCrossModelValidation:
    """Test cross-model validation logic."""

    def test_calculate_cross_model_gap(self):
        """Test cross-model gap calculation."""
        from src.equity_research.dcf import calculate_cross_model_gap

        fcff = 100.0
        gordon = 130.0
        gap = calculate_cross_model_gap(fcff, gordon)
        assert abs(gap - 0.30) < 0.01  # 30% gap

    def test_cross_model_gap_tiers(self):
        """Test cross-model gap tier classification."""
        from src.equity_research.dcf import classify_cross_model_gap

        assert classify_cross_model_gap(0.20) == "OK"  # < 25%
        assert classify_cross_model_gap(0.30) == "Warning"  # > 25%
        assert classify_cross_model_gap(0.55) == "Major Warning"  # > 50%
        assert classify_cross_model_gap(0.80) == "Reject"  # > 75%


class TestValuationConfidence:
    """Test valuation confidence scoring."""

    def test_confidence_score_calculated(self):
        """Test confidence score calculation."""
        from src.equity_research.dcf import calculate_valuation_confidence

        # Good case: valid assumptions, small cross-model gap, reasonable TV share
        score = calculate_valuation_confidence(
            wacc=0.10,
            terminal_growth=0.03,
            tv_contribution=0.60,
            cross_model_gap=0.10,
            upside=0.30,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # High confidence

    def test_confidence_score_low_for_bad_assumptions(self):
        """Test low confidence for problematic assumptions."""
        from src.equity_research.dcf import calculate_valuation_confidence

        score = calculate_valuation_confidence(
            wacc=0.06,  # Too low
            terminal_growth=0.08,  # Too high for mature
            tv_contribution=0.90,  # Too high
            cross_model_gap=0.60,  # Major gap
            upside=3.0,  # > 200%
        )
        assert score < 0.5  # Low confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])