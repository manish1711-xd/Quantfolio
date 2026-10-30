import math
import pytest
from src.analytics.dcf import (
    DcfAssumptions,
    CompanyFinancials,
    run_fcfe,
    run_ddm,
    run_gordon,
)

def test_fcfe_valuation():
    fin = CompanyFinancials(base_revenue=1000.0, net_debt=200.0, shares=10.0, current_price=100.0)
    assump = DcfAssumptions.constant(years=3, revenue_growth=0.08, ebit_margin=0.20)
    res = run_fcfe(assump, fin, cost_of_equity=0.10)
    assert res is not None
    assert not math.isnan(res.value_per_share)

def test_ddm_valuation():
    res = run_ddm(current_dividend=5.0, growth_rate=0.05, cost_of_equity=0.10, current_price=100.0)
    assert res is not None
    assert math.isclose(res.value_per_share, 105.0, rel_tol=1e-5)

def test_gordon_growth():
    val = run_gordon(base_cash_flow=10.0, growth_rate=0.04, discount_rate=0.10)
    assert not math.isnan(val)
    assert math.isclose(val, 10.0 * (1.0 + 0.04) / (0.10 - 0.04), rel_tol=1e-5)
