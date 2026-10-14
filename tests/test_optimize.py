"""Mean-variance optimisation — ``src.analytics.optimize``.

This is the module a user *acts* on: it does not describe the portfolio they
have, it tells them what to buy. So the tests below are written to be auditable
without running them, and they lean on three things a reviewer can check by eye.

**Closed-form goldens.** Most of the file optimises against a *diagonal*
covariance matrix, because every Markowitz answer for uncorrelated assets has a
one-line closed form, written out in the comment beside the assertion:

===========================  ==================================  =====================
portfolio                    weights when Sigma is diagonal      worked example
===========================  ==================================  =====================
global minimum variance      ``w_i prop. 1 / sigma_i^2``         ``(16, 4, 1) / 21``
tangency / max Sharpe        ``w prop. Sigma^-1 mu``             ``(4, 3, 1.125) / 8.125``
equal risk contribution      ``w_i prop. 1 / sigma_i``           ``(4, 2, 1) / 7``
maximum diversification      ``w_i prop. 1 / sigma_i``           ``DR = sqrt(n)``
===========================  ==================================  =====================

and the frontier itself is checked against the textbook two-fund result
``sigma_p^2(m) = (A m^2 - 2 B m + C) / (A C - B^2)`` with ``A = 1' Sigma^-1 1``,
``B = 1' Sigma^-1 mu`` and ``C = mu' Sigma^-1 mu``.

**Optimality identities**, which cannot rot as the solver changes:

* every returned weight vector sums to 1 within ``WEIGHT_TOLERANCE`` and sits
  inside its box bounds;
* ``Sharpe(max_sharpe) >= Sharpe(equal_weight)`` — the 1/N benchmark is one of
  the solver's own candidates, so this holds by construction, not by luck;
* ``vol(min_variance) <= vol(equal_weight)`` and ``<= vol(every frontier point)``;
* frontier volatility is non-decreasing as the target return rises;
* risk parity equalises ``rc_i = w_i (Sigma w)_i``, and those contributions sum
  to ``sigma_p^2`` exactly;
* no portfolio in the random cloud beats the uncapped optimisers.

**Documented degradation.** The module contract is that nothing raises: an empty
universe, a single asset, a singular covariance, an unreachable weight cap, a
target return nobody can deliver, and a panel with fewer observations than
assets each get a test pinning the NaN / empty / *note* / *warning* answer.

Deterministic and offline: ``random_portfolios`` is seeded, no optimiser uses a
random restart, and the only sampled data is a locally seeded return panel.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.optimize import (
    COVARIANCE_OBSERVATION_RATIO,
    DEFAULT_RANDOM_PORTFOLIOS,
    DEFAULT_SEED,
    EXPECTED_RETURN_METHODS,
    FRONTIER_COLUMNS,
    LABEL_CURRENT,
    LABEL_EQUAL_WEIGHT,
    LABEL_MAX_DIVERSIFICATION,
    LABEL_MAX_SHARPE,
    LABEL_MIN_VARIANCE,
    LABEL_RISK_PARITY,
    MAX_RANDOM_PORTFOLIOS,
    PORTFOLIO_LABELS,
    TRADING_DAYS,
    WEIGHT_TOLERANCE,
    covariance,
    efficient_frontier,
    equal_weight,
    expected_returns,
    max_diversification,
    max_sharpe,
    min_variance,
    optimize_all,
    portfolio_performance,
    random_portfolios,
    risk_parity,
    target_return_portfolio,
)
from src.config import MIN_OBSERVATIONS, OPTIMIZER
from src.models import OptimizedPortfolio

# ---------------------------------------------------------------------------
# the diagonal three-asset laboratory
#
# Sigma = diag(0.01, 0.04, 0.16)  ->  stand-alone vols 10%, 20%, 40%
# mu    = (0.04, 0.12, 0.18)      ->  picked so that no two optimisers happen
#                                     to coincide and the tangency Sharpe comes
#                                     out at exactly 0.85.
# ---------------------------------------------------------------------------
DIAG_LABELS = ("LOW", "MID", "HIGH")
DIAG_VARIANCES = (0.01, 0.04, 0.16)
DIAG_MU = (0.04, 0.12, 0.18)

#: A = 1' Sigma^-1 1 = 1/0.01 + 1/0.04 + 1/0.16 = 100 + 25 + 6.25
FRONTIER_A = 131.25
#: B = 1' Sigma^-1 mu = 0.04/0.01 + 0.12/0.04 + 0.18/0.16 = 4 + 3 + 1.125
FRONTIER_B = 8.125
#: C = mu' Sigma^-1 mu = 0.04*4 + 0.12*3 + 0.18*1.125 = 0.16 + 0.36 + 0.2025
FRONTIER_C = 0.7225
#: D = A*C - B^2 = 94.828125 - 66.015625
FRONTIER_D = 28.8125

#: Sharpe of the tangency portfolio at rf = 0 is sqrt(mu' Sigma^-1 mu), i.e.
#: sqrt(0.7225) = 0.85 exactly. Kept as the closed form rather than a decimal
#: so the comparison tolerance can be as tight as the solver allows.
TANGENCY_SHARPE = math.sqrt(FRONTIER_C)

# ---------------------------------------------------------------------------
# a four-asset, weakly and mixed-sign correlated laboratory
#
# Built from round vols and round correlations so the matrix stays auditable:
#   vols   ALFA 18%, BRVO 20%, CHRL 19%, DLTA 17%
#   rho    AB +0.05   AC -0.02   AD -0.06   BC +0.01   BD -0.01   CD +0.04
# Deliberately close to — but not exactly — diagonal: that is the regime where
# the risk-parity fixed point is least well behaved.
# ---------------------------------------------------------------------------
QUAD_LABELS = ("ALFA", "BRVO", "CHRL", "DLTA")
QUAD_VOLS = (0.18, 0.20, 0.19, 0.17)
QUAD_MU = (0.06, 0.11, 0.09, 0.05)
QUAD_CORRELATIONS = {
    (0, 1): 0.05,
    (0, 2): -0.02,
    (0, 3): -0.06,
    (1, 2): 0.01,
    (1, 3): -0.01,
    (2, 3): 0.04,
}


# ---------------------------------------------------------------------------
# local, deterministic builders (the shared fixtures cover everything else)
# ---------------------------------------------------------------------------
def diag_cov() -> pd.DataFrame:
    """``diag(0.01, 0.04, 0.16)`` labelled LOW / MID / HIGH."""
    index = pd.Index(list(DIAG_LABELS))
    return pd.DataFrame(np.diag(DIAG_VARIANCES), index=index, columns=index)


def diag_mu() -> pd.Series:
    """``(4%, 12%, 18%)`` annualised expected returns for the diagonal lab."""
    return pd.Series(dict(zip(DIAG_LABELS, DIAG_MU)), dtype=float)


def quad_cov() -> pd.DataFrame:
    """``Sigma_ij = rho_ij * sigma_i * sigma_j`` for the four-asset lab."""
    corr = np.eye(len(QUAD_LABELS))
    for (i, j), rho in QUAD_CORRELATIONS.items():
        corr[i, j] = corr[j, i] = rho
    vols = np.asarray(QUAD_VOLS, dtype=float)
    index = pd.Index(list(QUAD_LABELS))
    return pd.DataFrame(corr * np.outer(vols, vols), index=index, columns=index)


def quad_mu() -> pd.Series:
    return pd.Series(dict(zip(QUAD_LABELS, QUAD_MU)), dtype=float)


def returns_panel(n_obs: int = 260, n_assets: int = 4, seed: int = 20240101) -> pd.DataFrame:
    """A seeded daily-return panel for the end-to-end ``optimize_all`` tests.

    A shared market factor is mixed into every name so the covariance carries
    real structure; a toy uncorrelated panel would hide exactly the bugs an
    optimiser test is supposed to catch.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.008, n_obs)
    columns = [f"S{i}" for i in range(n_assets)]
    data = {
        name: 0.5 * market + 0.5 * rng.normal(0.0003 + 0.0001 * i, 0.010 + 0.002 * i, n_obs)
        for i, name in enumerate(columns)
    }
    return pd.DataFrame(data, index=pd.bdate_range("2023-01-02", periods=n_obs))


def weight_vector(portfolio: OptimizedPortfolio, labels=DIAG_LABELS) -> np.ndarray:
    """Weights as a numpy array in the caller's label order."""
    return np.array([portfolio.weights[str(label)] for label in labels], dtype=float)


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame) -> np.ndarray:
    """``rc_i = w_i (Sigma w)_i`` — the exact additive split of ``sigma_p^2``."""
    matrix = cov.to_numpy(dtype=float)
    return weights * (matrix @ weights)


def all_optimisers(mu: pd.Series, cov: pd.DataFrame, max_weight: float | None):
    """Every named optimiser, run over the same universe and the same cap."""
    return [
        min_variance(mu, cov, 0.0, max_weight=max_weight),
        max_sharpe(mu, cov, 0.0, max_weight=max_weight),
        risk_parity(cov, 0.0, mu=mu, max_weight=max_weight),
        max_diversification(cov, mu=mu, max_weight=max_weight),
        equal_weight(mu, cov, 0.0),
    ]


def single_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """One name: 10% expected return, 4% variance (a 20% volatility)."""
    index = pd.Index(["SOLO"])
    return pd.Series({"SOLO": 0.10}), pd.DataFrame([[0.04]], index=index, columns=index)


def singular_cov() -> pd.DataFrame:
    """Rank-2 3x3 covariance: TWIN_A and TWIN_B are literally the same bet::

        Sigma = [[0.04, 0.04, 0.00],
                 [0.04, 0.04, 0.00],
                 [0.00, 0.00, 0.01]]

    Perfect correlation plus equal variance makes the top-left block singular,
    so ``Sigma`` has an exact zero eigenvalue and cannot be inverted.
    """
    index = pd.Index(["TWIN_A", "TWIN_B", "SOLO"])
    values = [[0.04, 0.04, 0.0], [0.04, 0.04, 0.0], [0.0, 0.0, 0.01]]
    return pd.DataFrame(values, index=index, columns=index)


# ---- estimation: expected returns -----------------------------------------
def test_expected_returns_geometrically_annualises_the_realised_path():
    # r = [+10%, -10%, +10%, -10%] compounds to 1.1*0.9*1.1*0.9 = 0.9801 over
    # 4 observations, so mu = 0.9801 ** (252/4) - 1 = 0.9801 ** 63 - 1.
    result = expected_returns(pd.DataFrame({"X": [0.10, -0.10, 0.10, -0.10]}))
    assert result.iloc[0] == pytest.approx(0.9801**63 - 1.0, rel=1e-12)
    assert result.iloc[0] == pytest.approx(-0.71813930, abs=1e-8)


def test_expected_returns_mean_method_is_the_arithmetic_annualisation():
    # mean([+10%, -10%, +10%, -10%]) = 0 exactly, so 0 * 252 = 0. The gap to
    # the -71.8% geometric figure above IS the volatility drag; confusing the
    # two overstates a choppy asset by seventy percentage points.
    result = expected_returns(pd.DataFrame({"X": [0.10, -0.10, 0.10, -0.10]}), method="mean")
    assert result.iloc[0] == pytest.approx(0.0, abs=1e-15)


def test_expected_returns_of_a_constant_drift_is_the_compounded_year():
    # 252 sessions of exactly +0.1%: growth = 1.001**252 over n = 252, so
    # mu = (1.001**252) ** (252/252) - 1 = 1.001**252 - 1 = 0.28643404...
    result = expected_returns(pd.DataFrame({"X": [0.001] * TRADING_DAYS}))
    assert result.iloc[0] == pytest.approx(1.001**252 - 1.0, rel=1e-12)
    assert result.iloc[0] == pytest.approx(0.28643404, abs=1e-8)


def test_the_annualisation_convention_is_252_trading_days():
    # Regression guard: 260 (calendar weeks) or 365 would rescale every single
    # optimiser input, and therefore every recommended weight.
    assert TRADING_DAYS == 252


def test_expected_returns_is_nan_for_a_total_wipeout_rather_than_a_fabrication():
    # A -100% day drives cumulative growth to zero, and 0 ** (252/n) - 1 = -1
    # is not a meaningful annualisation of "the asset ceased to exist".
    result = expected_returns(pd.DataFrame({"OK": [0.01, 0.02, -0.01], "DEAD": [-1.0, 0.5, 0.5]}))
    assert math.isnan(result["DEAD"])
    assert np.isfinite(result["OK"])


def test_expected_returns_of_no_data_is_an_empty_named_series():
    assert expected_returns(None).empty
    assert expected_returns(pd.DataFrame()).empty
    assert expected_returns(None).name == "expected_return"


def test_expected_returns_rejects_an_unknown_method_as_a_caller_bug():
    # Degenerate DATA degrades to NaN; a mistyped METHOD is a programming error
    # and has to be loud.
    with pytest.raises(ValueError):
        expected_returns(pd.DataFrame({"A": [0.01, 0.02]}), method="ewma")


def test_expected_returns_capm_is_declared_unimplemented_not_silently_wrong():
    assert "capm" in EXPECTED_RETURN_METHODS
    with pytest.raises(NotImplementedError):
        expected_returns(pd.DataFrame({"A": [0.01, 0.02]}), method="capm")


def test_expected_returns_is_invariant_to_the_order_of_the_rows():
    # A product does not care about ordering; if this ever fails the estimator
    # has grown a path dependency it must not have.
    frame = returns_panel(n_obs=120, n_assets=3)
    shuffled = frame.iloc[np.random.default_rng(5).permutation(len(frame))]
    pd.testing.assert_series_equal(expected_returns(frame), expected_returns(shuffled), rtol=1e-12)


# ---- estimation: covariance ----------------------------------------------
def test_covariance_is_the_sample_covariance_scaled_by_252():
    # r_A = [0.01, -0.01, 0.02, -0.02]: mean 0, sum of squares 1.0e-3,
    # var(ddof=1) = 1.0e-3/3, annualised = 1.0e-3/3 * 252 = 0.084.
    # r_B = 2 * r_A, so var_B = 4 * var_A = 0.336 and cov_AB = 2 * var_A = 0.168.
    frame = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02], "B": [0.02, -0.02, 0.04, -0.04]})
    cov = covariance(frame, shrinkage=0.0)
    assert cov.loc["A", "A"] == pytest.approx(0.084, rel=1e-12)
    assert cov.loc["B", "B"] == pytest.approx(0.336, rel=1e-12)
    assert cov.loc["A", "B"] == pytest.approx(0.168, rel=1e-12)


def test_covariance_shrinkage_preserves_the_trace_and_shrinks_the_off_diagonal():
    # Sigma = (1-a) S + a (trace(S)/n) I keeps trace(Sigma) = trace(S) for any
    # a, and multiplies every off-diagonal by (1-a). At a = 0.5:
    #   Sigma_AB = 0.5 * 0.168 = 0.084
    #   Sigma_AA = 0.5 * 0.084 + 0.5 * (0.42/2) = 0.042 + 0.105 = 0.147
    frame = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02], "B": [0.02, -0.02, 0.04, -0.04]})
    sample = covariance(frame, shrinkage=0.0)
    shrunk = covariance(frame, shrinkage=0.5)
    assert np.trace(shrunk.to_numpy()) == pytest.approx(np.trace(sample.to_numpy()), rel=1e-12)
    assert shrunk.loc["A", "B"] == pytest.approx(0.084, rel=1e-12)
    assert shrunk.loc["A", "A"] == pytest.approx(0.147, rel=1e-12)


def test_covariance_at_full_shrinkage_is_the_scaled_identity():
    # a = 1 -> Sigma = (trace(S)/n) I = (0.42/2) I = diag(0.21, 0.21): the same
    # average variance, zero correlation.
    frame = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02], "B": [0.02, -0.02, 0.04, -0.04]})
    cov = covariance(frame, shrinkage=1.0)
    assert cov.to_numpy() == pytest.approx(np.diag([0.21, 0.21]), abs=1e-12)


def test_covariance_clips_an_out_of_range_shrinkage_intensity():
    frame = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02], "B": [0.02, -0.02, 0.04, -0.04]})
    pd.testing.assert_frame_equal(
        covariance(frame, shrinkage=5.0), covariance(frame, shrinkage=1.0)
    )
    pd.testing.assert_frame_equal(
        covariance(frame, shrinkage=-2.0), covariance(frame, shrinkage=0.0)
    )


def test_covariance_falls_back_to_the_configured_shrinkage_for_a_nan_intensity():
    # NaN is not a request for zero shrinkage; it is no request at all.
    frame = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02], "B": [0.02, -0.02, 0.04, -0.04]})
    pd.testing.assert_frame_equal(
        covariance(frame, shrinkage=float("nan")),
        covariance(frame, shrinkage=OPTIMIZER.shrinkage),
    )


def test_covariance_is_symmetric_and_positive_semi_definite(asset_returns: pd.DataFrame):
    # Estimating on COMPLETE CASES is what buys positive semi-definiteness; a
    # non-PSD Sigma would let the optimiser manufacture a portfolio with
    # negative variance and an infinite Sharpe ratio.
    matrix = covariance(asset_returns).to_numpy()
    assert matrix == pytest.approx(matrix.T, abs=1e-15)
    assert float(np.min(np.linalg.eigvalsh(matrix))) >= -1e-12


def test_covariance_needs_two_complete_observations():
    assert covariance(None).empty
    assert covariance(pd.DataFrame()).empty
    assert covariance(pd.DataFrame({"A": [0.01], "B": [0.02]})).empty


def test_covariance_keeps_a_zero_variance_cash_like_column():
    # r = [0.02, -0.01, 0.03, -0.02, 0.01, 0.0]: mean 0.005, deviations
    # +/-0.015, +/-0.025, +/-0.005 -> sum of squares 1.75e-3, var(ddof=1)
    # = 3.5e-4, annualised 3.5e-4 * 252 = 0.0882. CASH never moves, so its
    # whole row is exactly zero — a legitimate, if degenerate, asset.
    frame = pd.DataFrame({"RISKY": [0.02, -0.01, 0.03, -0.02, 0.01, 0.0], "CASH": [0.0] * 6})
    cov = covariance(frame, shrinkage=0.0)
    assert cov.loc["RISKY", "RISKY"] == pytest.approx(0.0882, rel=1e-12)
    assert cov.loc["CASH", "CASH"] == 0.0
    assert cov.loc["RISKY", "CASH"] == 0.0


def test_covariance_does_not_mutate_the_caller(asset_returns: pd.DataFrame):
    before = asset_returns.copy(deep=True)
    covariance(asset_returns)
    pd.testing.assert_frame_equal(asset_returns, before)


# ---- scoring: portfolio_performance --------------------------------------
def test_portfolio_performance_of_a_single_name_is_that_name_alone():
    # 100% LOW: r_p = 0.04, sigma_p = sqrt(0.01) = 0.10, Sharpe = 0.04/0.10.
    ret, vol, sharpe = portfolio_performance({"LOW": 1.0}, diag_mu(), diag_cov())
    assert ret == pytest.approx(0.04, rel=1e-12)
    assert vol == pytest.approx(0.10, rel=1e-12)
    assert sharpe == pytest.approx(0.40, rel=1e-12)


def test_portfolio_performance_of_equal_weights_is_hand_computable():
    # w = (1/3, 1/3, 1/3):  r_p = 0.34/3 = 0.113333...
    # Sigma is diagonal, so sigma_p^2 = (0.01 + 0.04 + 0.16)/9 = 0.21/9 and
    # sigma_p = sqrt(0.21)/3 = 0.15275252...
    ret, vol, sharpe = portfolio_performance(
        dict.fromkeys(DIAG_LABELS, 1 / 3), diag_mu(), diag_cov()
    )
    assert ret == pytest.approx(0.34 / 3.0, rel=1e-12)
    assert vol == pytest.approx(math.sqrt(0.21) / 3.0, rel=1e-12)
    assert sharpe == pytest.approx(0.34 / math.sqrt(0.21), rel=1e-12)


def test_portfolio_performance_normalises_percent_and_fraction_weights_alike():
    # Regression guard: a UI that hands over 25/25/25/25 must not be scored as
    # a hundred-times-levered book.
    percent = portfolio_performance(dict.fromkeys(QUAD_LABELS, 25.0), quad_mu(), quad_cov())
    fraction = portfolio_performance(dict.fromkeys(QUAD_LABELS, 0.25), quad_mu(), quad_cov())
    assert percent == pytest.approx(fraction, rel=1e-12)


def test_portfolio_performance_subtracts_the_risk_free_rate_from_the_return_only():
    # Sharpe = (r_p - rf)/sigma_p, so only the numerator moves with rf:
    #   (0.12 - 0.00)/0.20 = 0.60  against  (0.12 - 0.02)/0.20 = 0.50
    _, vol_a, sharpe_a = portfolio_performance({"MID": 1.0}, diag_mu(), diag_cov(), 0.0)
    ret_b, vol_b, sharpe_b = portfolio_performance({"MID": 1.0}, diag_mu(), diag_cov(), 0.02)
    assert ret_b == pytest.approx(0.12, rel=1e-12)
    assert vol_b == pytest.approx(vol_a, rel=1e-12)
    assert sharpe_a == pytest.approx(0.60, rel=1e-12)
    assert sharpe_b == pytest.approx(0.50, rel=1e-12)


def test_portfolio_performance_treats_a_missing_name_as_zero_weight():
    # Only LOW is supplied: MID and HIGH come back NaN, are read as zero, and
    # the survivor is normalised to the whole book.
    assert portfolio_performance({"LOW": 0.4}, diag_mu(), diag_cov()) == pytest.approx(
        portfolio_performance({"LOW": 1.0}, diag_mu(), diag_cov()), rel=1e-12
    )


@pytest.mark.parametrize(
    "weights", [None, {}, {"UNKNOWN": 1.0}, {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0}]
)
def test_portfolio_performance_is_all_nan_when_nothing_scoreable_was_supplied(weights):
    result = portfolio_performance(weights, diag_mu(), diag_cov())
    assert all(math.isnan(value) for value in result)


def test_portfolio_performance_is_all_nan_without_a_covariance_universe():
    assert all(math.isnan(x) for x in portfolio_performance({"LOW": 1.0}, diag_mu(), None))


def test_portfolio_performance_sharpe_is_nan_rather_than_infinite_at_zero_volatility():
    index = pd.Index(["P", "Q"])
    zero = pd.DataFrame(np.zeros((2, 2)), index=index, columns=index)
    ret, vol, sharpe = portfolio_performance(
        {"P": 0.5, "Q": 0.5}, pd.Series({"P": 0.10, "Q": 0.20}), zero
    )
    assert ret == pytest.approx(0.15, rel=1e-12)
    assert vol == 0.0
    assert math.isnan(sharpe)


# ---- equal weight ---------------------------------------------------------
def test_equal_weight_is_one_over_n_and_needs_no_solver():
    portfolio = equal_weight(diag_mu(), diag_cov())
    assert portfolio.label == LABEL_EQUAL_WEIGHT
    assert portfolio.weights == pytest.approx(dict.fromkeys(DIAG_LABELS, 1 / 3), abs=1e-15)
    assert portfolio.converged is True
    assert portfolio.note == ""


def test_equal_weight_metrics_match_the_hand_computed_scores():
    # r_p = (0.04 + 0.12 + 0.18)/3 = 0.34/3 ; sigma_p = sqrt(0.21)/3 ;
    # Sharpe = (0.34/3) / (sqrt(0.21)/3) = 0.34/sqrt(0.21) = 0.74194083...
    portfolio = equal_weight(diag_mu(), diag_cov())
    assert portfolio.expected_return == pytest.approx(0.34 / 3.0, rel=1e-12)
    assert portfolio.volatility == pytest.approx(math.sqrt(0.21) / 3.0, rel=1e-12)
    assert portfolio.sharpe == pytest.approx(0.34 / math.sqrt(0.21), rel=1e-12)
    assert portfolio.sharpe == pytest.approx(0.74194083, abs=1e-8)


def test_equal_weight_ignores_the_weight_cap_because_one_over_n_always_fits():
    # 1/n is feasible under every cap the module accepts, so there is nothing
    # to widen and no note to write.
    assert equal_weight(quad_mu(), quad_cov()).note == ""


# ---- minimum variance ------------------------------------------------------
def test_min_variance_matches_the_inverse_variance_closed_form():
    # Sigma diagonal -> w_i prop. 1/sigma_i^2 = (100, 25, 6.25). The total is
    # A = 131.25, so w = (16, 4, 1)/21 and sigma_p = 1/sqrt(A) = 0.08728716.
    portfolio = min_variance(diag_mu(), diag_cov(), max_weight=1.0)
    assert portfolio.label == LABEL_MIN_VARIANCE
    assert portfolio.weights == pytest.approx(
        {"LOW": 16 / 21, "MID": 4 / 21, "HIGH": 1 / 21}, abs=1e-8
    )
    assert portfolio.volatility == pytest.approx(1.0 / math.sqrt(FRONTIER_A), rel=1e-9)
    assert portfolio.volatility == pytest.approx(0.08728716, abs=1e-8)
    # r_p = (16*0.04 + 4*0.12 + 1*0.18)/21 = (0.64 + 0.48 + 0.18)/21 = 1.30/21
    assert portfolio.expected_return == pytest.approx(1.30 / 21.0, rel=1e-8)
    assert portfolio.converged is True


def test_min_variance_volatility_is_never_above_equal_weights():
    # The identity that makes the optimiser worth running at all: equal weight
    # is one of the solver's own candidates, so it can only be improved on.
    for mu, cov in ((diag_mu(), diag_cov()), (quad_mu(), quad_cov())):
        assert min_variance(mu, cov).volatility <= equal_weight(mu, cov).volatility + 1e-12


def test_min_variance_does_not_need_expected_returns_at_all():
    # mu is used only to SCORE the answer, so the weights must be identical
    # with and without it, and the score must go NaN rather than invent one.
    with_mu = min_variance(diag_mu(), diag_cov(), max_weight=1.0)
    without_mu = min_variance(None, diag_cov(), max_weight=1.0)
    assert without_mu.weights == pytest.approx(with_mu.weights, abs=1e-12)
    assert without_mu.volatility == pytest.approx(with_mu.volatility, rel=1e-12)
    assert math.isnan(without_mu.expected_return)
    assert math.isnan(without_mu.sharpe)


def test_min_variance_keeps_a_name_whose_expected_return_is_unusable():
    # min_variance is the one optimiser that can still act on a name with no
    # finite mu, so it must not drop it the way max_sharpe has to.
    frame = pd.DataFrame({"OK": [0.01, 0.02, -0.01, 0.005], "DEAD": [-1.0, 0.5, 0.5, 0.1]})
    portfolio = min_variance(expected_returns(frame), covariance(frame))
    assert set(portfolio.weights) == {"OK", "DEAD"}
    assert math.isnan(portfolio.expected_return)  # the score degrades, not the weights


# ---- maximum Sharpe --------------------------------------------------------
def test_max_sharpe_matches_the_tangency_closed_form():
    # w prop. Sigma^-1 mu = (0.04/0.01, 0.12/0.04, 0.18/0.16) = (4, 3, 1.125),
    # normalised by B = 8.125. Then
    #   r_p     = C / B      = 0.7225 / 8.125 = 0.08892308...
    #   sigma_p = sqrt(C) / B = 0.85   / 8.125 = 0.10461538...
    #   Sharpe  = sqrt(C)     = sqrt(0.7225)   = 0.85 exactly
    portfolio = max_sharpe(diag_mu(), diag_cov(), 0.0, max_weight=1.0)
    assert portfolio.label == LABEL_MAX_SHARPE
    assert portfolio.weights == pytest.approx(
        {"LOW": 4 / FRONTIER_B, "MID": 3 / FRONTIER_B, "HIGH": 1.125 / FRONTIER_B},
        abs=1e-6,
    )
    assert portfolio.expected_return == pytest.approx(FRONTIER_C / FRONTIER_B, rel=1e-8)
    assert portfolio.volatility == pytest.approx(TANGENCY_SHARPE / FRONTIER_B, rel=1e-8)
    assert portfolio.sharpe == pytest.approx(TANGENCY_SHARPE, rel=1e-9)
    assert portfolio.sharpe == pytest.approx(0.85, abs=1e-9)


@pytest.mark.parametrize("max_weight", [1.0, OPTIMIZER.max_weight])
def test_max_sharpe_is_never_worse_than_equal_weight(max_weight):
    # The headline structural guarantee, checked both unconstrained and under
    # the shipped 35% cap — a binding cap is exactly where a naive solver would
    # fall behind 1/N.
    for mu, cov in ((diag_mu(), diag_cov()), (quad_mu(), quad_cov())):
        optimised = max_sharpe(mu, cov, 0.0, max_weight=max_weight)
        assert optimised.sharpe >= equal_weight(mu, cov, 0.0).sharpe - 1e-9


def test_max_sharpe_tilts_towards_the_high_return_name_as_the_risk_free_rate_rises():
    # Raising rf shrinks every excess return by the same amount, which hurts
    # the low-return / low-vol names proportionally more, so the tangency
    # portfolio slides up the frontier.
    low_rf = max_sharpe(diag_mu(), diag_cov(), 0.0, max_weight=1.0)
    high_rf = max_sharpe(diag_mu(), diag_cov(), 0.03, max_weight=1.0)
    assert high_rf.weights["HIGH"] > low_rf.weights["HIGH"]
    assert high_rf.expected_return > low_rf.expected_return


def test_max_sharpe_drops_a_name_with_no_finite_expected_return_and_names_it():
    frame = pd.DataFrame({"OK": [0.01, 0.02, -0.01, 0.005], "DEAD": [-1.0, 0.5, 0.5, 0.1]})
    portfolio = max_sharpe(expected_returns(frame), covariance(frame))
    assert set(portfolio.weights) == {"OK"}
    assert "DEAD" in portfolio.note  # named out loud, never silently vanished


def test_max_sharpe_reports_a_self_consistent_return_volatility_sharpe_triple():
    portfolio = max_sharpe(quad_mu(), quad_cov(), 0.015, max_weight=1.0)
    assert portfolio.sharpe == pytest.approx(
        (portfolio.expected_return - 0.015) / portfolio.volatility, rel=1e-10
    )


# ---- risk parity -----------------------------------------------------------
def test_risk_parity_matches_the_inverse_volatility_closed_form_when_uncorrelated():
    # With Sigma diagonal, rc_i = w_i^2 sigma_i^2, so equal contributions mean
    # w_i prop. 1/sigma_i = (10, 5, 2.5), i.e. w = (4, 2, 1)/7. Then
    # sigma_p^2 = (16*0.01 + 4*0.04 + 1*0.16)/49 = 0.48/49.
    portfolio = risk_parity(diag_cov(), 0.0, mu=diag_mu(), max_weight=1.0)
    assert portfolio.label == LABEL_RISK_PARITY
    assert portfolio.weights == pytest.approx({"LOW": 4 / 7, "MID": 2 / 7, "HIGH": 1 / 7}, abs=1e-9)
    assert portfolio.volatility == pytest.approx(math.sqrt(0.48 / 49.0), rel=1e-9)


def test_risk_parity_equalises_the_risk_contributions_on_four_assets():
    # Computed directly from w and Sigma rather than read back off the
    # optimiser: rc_i = w_i (Sigma w)_i must be the same for every name.
    cov = quad_cov()
    portfolio = risk_parity(cov, 0.0, max_weight=1.0)
    contributions = risk_contributions(weight_vector(portfolio, QUAD_LABELS), cov)
    assert float(np.min(contributions)) > 0.0
    assert float(np.max(contributions)) / float(np.min(contributions)) < 1.05


def test_risk_contributions_sum_to_the_portfolio_variance_exactly():
    # sum_i w_i (Sigma w)_i = w' Sigma w = sigma_p^2. This identity is what
    # makes an equal-risk-contribution budget well posed in the first place.
    cov = quad_cov()
    portfolio = risk_parity(cov, 0.0, mu=quad_mu(), max_weight=1.0)
    contributions = risk_contributions(weight_vector(portfolio, QUAD_LABELS), cov)
    assert float(np.sum(contributions)) == pytest.approx(portfolio.volatility**2, rel=1e-12)


def test_risk_parity_sits_between_minimum_variance_and_equal_weight_in_volatility():
    # ERC is the classic middle ground: it uses the correlation structure (so
    # it beats 1/N) without chasing the variance minimum.
    cov = quad_cov()
    mv = min_variance(quad_mu(), cov, max_weight=1.0).volatility
    rp = risk_parity(cov, 0.0, mu=quad_mu(), max_weight=1.0).volatility
    ew = equal_weight(quad_mu(), cov).volatility
    assert mv <= rp + 1e-12
    assert rp <= ew + 1e-12


def test_risk_parity_needs_no_expected_returns_and_says_so_with_nan():
    portfolio = risk_parity(diag_cov(), 0.0, mu=None, max_weight=1.0)
    assert np.isfinite(portfolio.volatility)
    assert math.isnan(portfolio.expected_return)
    assert math.isnan(portfolio.sharpe)


def test_risk_parity_uses_the_exact_fixed_point_when_it_is_available():
    # A diagonal Sigma is the case the ERC iteration handles perfectly, so no
    # fallback note should be written. Positive control for the two tests below.
    assert risk_parity(diag_cov(), max_weight=1.0).note == ""


def test_risk_parity_reports_the_fixed_point_note_when_a_name_carries_no_risk():
    # The note's first documented cause, exercised honestly: a cash-like column
    # has zero marginal risk, so w_i prop. 1/(Sigma w)_i cannot even be formed.
    frame = pd.DataFrame({"RISKY": [0.02, -0.01, 0.03, -0.02, 0.01, 0.0], "CASH": [0.0] * 6})
    portfolio = risk_parity(covariance(frame, shrinkage=0.0), max_weight=1.0)
    assert "fixed point unavailable" in portfolio.note


# Regression guard for a fixed defect:
# the ERC fixed point w <- normalise(1/(Sigma w)) has a marginally unstable period-2 mode.
# On a near-diagonal Sigma it oscillates with growing amplitude until a marginal risk
# turns negative, so _risk_parity_fixed_point bails out, risk_parity falls back to SLSQP
# and emits a note whose two stated causes are both false here. The weights are still
# correct; the diagnostic is not, and it reaches the user through
# optimize_all()['warnings'].
def test_risk_parity_writes_no_note_when_no_cap_binds_and_every_name_carries_risk():
    cov = quad_cov()
    portfolio = risk_parity(cov, 0.0, max_weight=1.0)
    weights = weight_vector(portfolio, QUAD_LABELS)
    # The premises of the note's two stated causes, checked before the claim:
    assert float(np.min(np.linalg.eigvalsh(cov.to_numpy()))) > 0.0  # Sigma is PD
    assert float(np.min(cov.to_numpy() @ weights)) > 0.0  # every marginal > 0
    assert float(np.max(weights)) < 1.0  # the cap is slack
    assert portfolio.note == ""


# ---- maximum diversification ----------------------------------------------
def test_max_diversification_of_uncorrelated_assets_is_inverse_volatility():
    # For a diagonal Sigma the ratio (sum w_i sigma_i)/sqrt(sum w_i^2 sigma_i^2)
    # is maximised at w_i prop. 1/sigma_i = (4, 2, 1)/7.
    portfolio = max_diversification(diag_cov(), mu=diag_mu(), max_weight=1.0)
    assert portfolio.label == LABEL_MAX_DIVERSIFICATION
    assert portfolio.weights == pytest.approx({"LOW": 4 / 7, "MID": 2 / 7, "HIGH": 1 / 7}, abs=1e-9)


def test_max_diversification_reaches_the_root_n_diversification_ratio():
    # At w_i = c/sigma_i the numerator is n*c and the denominator sqrt(n)*c, so
    # DR = sqrt(n) = sqrt(3) = 1.7320508075688772 for three uncorrelated names.
    cov = diag_cov()
    portfolio = max_diversification(cov, max_weight=1.0)
    stand_alone = np.sqrt(np.diag(cov.to_numpy()))
    ratio = float(weight_vector(portfolio) @ stand_alone) / portfolio.volatility
    assert ratio == pytest.approx(math.sqrt(3.0), rel=1e-9)


def test_max_diversification_beats_equal_weight_on_its_own_objective():
    # DR = 1 would mean every name is the same bet; anything above 1 is the
    # correlation structure paying for itself.
    cov = quad_cov()
    stand_alone = np.sqrt(np.diag(cov.to_numpy()))
    optimised = max_diversification(cov, max_weight=1.0)
    benchmark = equal_weight(quad_mu(), cov)
    best = float(weight_vector(optimised, QUAD_LABELS) @ stand_alone) / optimised.volatility
    naive = float(weight_vector(benchmark, QUAD_LABELS) @ stand_alone) / benchmark.volatility
    assert best >= naive - 1e-9
    assert best > 1.0


# ---- target-return portfolio ----------------------------------------------
def test_target_return_portfolio_hits_the_two_fund_closed_form():
    # Minimising w'Sigma w subject to w.mu = m and sum(w) = 1 gives
    #   sigma_p^2(m) = (A m^2 - 2 B m + C) / (A C - B^2)
    # with A = 131.25, B = 8.125, C = 0.7225, D = 28.8125. At m = 0.12:
    #   (131.25*0.0144 - 2*8.125*0.12 + 0.7225)/28.8125
    #   = (1.89 - 1.95 + 0.7225)/28.8125 = 0.6625/28.8125 = 0.022993...
    #   sigma_p = 0.15163605...
    portfolio = target_return_portfolio(diag_mu(), diag_cov(), 0.12, max_weight=1.0)
    expected_variance = (FRONTIER_A * 0.12**2 - 2 * FRONTIER_B * 0.12 + FRONTIER_C) / FRONTIER_D
    assert portfolio.expected_return == pytest.approx(0.12, abs=1e-9)
    assert portfolio.volatility == pytest.approx(math.sqrt(expected_variance), rel=1e-7)
    assert portfolio.volatility == pytest.approx(0.15163605, abs=1e-7)


def test_a_target_below_the_minimum_variance_return_returns_the_minimum_variance_mix():
    # The constraint is w.mu >= target, so below the anchor it is simply slack.
    relaxed = target_return_portfolio(diag_mu(), diag_cov(), 0.01, max_weight=1.0)
    anchor = min_variance(diag_mu(), diag_cov(), max_weight=1.0)
    assert relaxed.weights == pytest.approx(anchor.weights, abs=1e-8)
    assert relaxed.volatility == pytest.approx(anchor.volatility, rel=1e-9)


def test_an_unreachable_target_is_clipped_to_the_best_the_bounds_allow():
    # No long-only mix of (4%, 12%, 18%) delivers 50%. The honest answer is the
    # 18% all-HIGH corner plus a note, not a fabricated portfolio.
    portfolio = target_return_portfolio(diag_mu(), diag_cov(), 0.50, max_weight=1.0)
    assert portfolio.expected_return == pytest.approx(0.18, abs=1e-9)
    assert portfolio.weights["HIGH"] == pytest.approx(1.0, abs=1e-8)
    assert "clipped to the maximum" in portfolio.note


def test_a_weight_cap_lowers_the_reachable_maximum_return():
    # Under a 35% cap the greedy fill is 35% HIGH + 35% MID + 30% LOW, so the
    # best reachable return is 0.35*0.18 + 0.35*0.12 + 0.30*0.04
    # = 0.063 + 0.042 + 0.012 = 0.117, not 0.18.
    portfolio = target_return_portfolio(diag_mu(), diag_cov(), 0.50, max_weight=0.35)
    assert portfolio.expected_return == pytest.approx(0.117, abs=1e-8)
    assert "clipped to the maximum" in portfolio.note


def test_a_non_numeric_target_degrades_to_an_empty_portfolio():
    portfolio = target_return_portfolio(diag_mu(), diag_cov(), float("nan"))
    assert portfolio.weights == {}
    assert math.isnan(portfolio.volatility)
    assert "not a number" in portfolio.note


# ---- the efficient frontier ------------------------------------------------
def test_the_frontier_starts_exactly_at_the_minimum_variance_portfolio():
    frontier = efficient_frontier(diag_mu(), diag_cov(), 0.0, n_points=12, max_weight=1.0)
    anchor = min_variance(diag_mu(), diag_cov(), max_weight=1.0)
    assert frontier["volatility"].iloc[0] == pytest.approx(anchor.volatility, rel=1e-12)
    assert frontier.attrs["min_variance_volatility"] == pytest.approx(anchor.volatility, rel=1e-12)
    assert frontier["volatility"].iloc[0] == pytest.approx(1.0 / math.sqrt(FRONTIER_A), rel=1e-9)


def test_no_frontier_point_has_a_lower_volatility_than_the_minimum_variance_point():
    # The defining property of the GLOBAL minimum: nothing on the frontier can
    # sit to its left.
    anchor = min_variance(diag_mu(), diag_cov(), max_weight=1.0).volatility
    frontier = efficient_frontier(diag_mu(), diag_cov(), 0.0, n_points=20, max_weight=1.0)
    assert float(frontier["volatility"].min()) >= anchor - 1e-12


def test_frontier_volatility_is_non_decreasing_as_the_target_return_rises():
    # Nested feasible sets (the constraint is >=, not ==) make this monotone; a
    # non-monotone frontier is the classic symptom of an equality constraint
    # letting the solver onto the inefficient lower branch.
    frontier = efficient_frontier(diag_mu(), diag_cov(), 0.0, n_points=25, max_weight=1.0)
    assert (frontier["volatility"].diff().dropna() >= -1e-12).all()
    assert (frontier["return"].diff().dropna() >= -1e-12).all()


def test_every_frontier_row_is_a_fully_invested_long_only_capped_portfolio():
    frontier = efficient_frontier(quad_mu(), quad_cov(), 0.0, n_points=15, max_weight=0.4)
    weights = frontier[frontier.attrs["tickers"]].to_numpy()
    assert weights.sum(axis=1) == pytest.approx(np.ones(len(frontier)), abs=WEIGHT_TOLERANCE)
    assert float(weights.min()) >= -1e-9
    assert float(weights.max()) <= 0.4 + 1e-9


def test_the_frontiers_last_point_is_the_highest_return_corner():
    # Uncapped, the top of the frontier is 100% HIGH: r = 0.18, sigma = 0.40,
    # Sharpe = 0.18/0.40 = 0.45.
    last = efficient_frontier(diag_mu(), diag_cov(), 0.0, n_points=12, max_weight=1.0).iloc[-1]
    assert last["return"] == pytest.approx(0.18, abs=1e-8)
    assert last["volatility"] == pytest.approx(0.40, rel=1e-8)
    assert last["sharpe"] == pytest.approx(0.45, rel=1e-8)


def test_no_frontier_point_beats_the_tangency_portfolios_sharpe():
    # The frontier is a discretisation of the same feasible set, so the best
    # Sharpe it can show is bounded above by the max-Sharpe solve — and by the
    # closed-form sqrt(mu' Sigma^-1 mu) = 0.85.
    frontier = efficient_frontier(diag_mu(), diag_cov(), 0.0, n_points=40, max_weight=1.0)
    best = float(frontier["sharpe"].max())
    assert best <= max_sharpe(diag_mu(), diag_cov(), 0.0, max_weight=1.0).sharpe + 1e-9
    assert best <= TANGENCY_SHARPE + 1e-9


def test_frontier_columns_lead_with_return_volatility_sharpe():
    # Regression guard: the optimiser chart reads these positionally and the
    # weight columns are expected to follow in ticker order.
    assert FRONTIER_COLUMNS == ("return", "volatility", "sharpe")
    frontier = efficient_frontier(diag_mu(), diag_cov(), n_points=5, max_weight=1.0)
    assert list(frontier.columns[:3]) == list(FRONTIER_COLUMNS)
    assert list(frontier.columns[3:]) == list(DIAG_LABELS)
    assert frontier.attrs["tickers"] == list(DIAG_LABELS)


def test_a_ticker_colliding_with_a_statistic_name_is_suffixed():
    index = pd.Index(["return", "OTHER"])
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=index, columns=index)
    frontier = efficient_frontier(
        pd.Series({"return": 0.05, "OTHER": 0.10}), cov, n_points=4, max_weight=1.0
    )
    assert "return_w" in frontier.columns
    assert list(frontier.columns).count("return") == 1


def test_the_frontier_point_count_is_clamped_to_a_sane_range():
    for requested in (0, 1, 2):
        frontier = efficient_frontier(diag_mu(), diag_cov(), n_points=requested, max_weight=1.0)
        assert len(frontier) == 2
    assert len(efficient_frontier(diag_mu(), diag_cov(), n_points=1000, max_weight=1.0)) <= 500


def test_a_universe_with_no_return_spread_collapses_to_the_anchor_alone():
    # Every name promises 10%: there is nothing to sweep, so the frontier is
    # the minimum-variance point and nothing else. Not an error, not 60 copies.
    flat = pd.Series(dict.fromkeys(DIAG_LABELS, 0.1))
    frontier = efficient_frontier(flat, diag_cov(), n_points=10, max_weight=1.0)
    assert len(frontier) == 1
    assert frontier["return"].iloc[0] == pytest.approx(0.10, abs=1e-12)


def test_the_frontier_of_an_unusable_universe_is_empty():
    assert efficient_frontier(None, None).empty
    assert efficient_frontier(diag_mu(), pd.DataFrame()).empty


# ---- the random cloud ------------------------------------------------------
def test_random_portfolios_is_reproducible_for_a_fixed_seed():
    first = random_portfolios(quad_mu(), quad_cov(), 0.0, n=500, seed=7)
    second = random_portfolios(quad_mu(), quad_cov(), 0.0, n=500, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(random_portfolios(quad_mu(), quad_cov(), 0.0, n=500, seed=8))


def test_random_portfolios_defaults_are_the_documented_ones():
    # Regression guard: the default cloud must not silently change size or seed,
    # because the chart it draws is compared across releases.
    assert DEFAULT_SEED == 42
    assert DEFAULT_RANDOM_PORTFOLIOS == 3000
    pd.testing.assert_frame_equal(
        random_portfolios(diag_mu(), diag_cov()),
        random_portfolios(diag_mu(), diag_cov(), 0.0, n=3000, seed=42),
    )


def test_random_portfolios_has_the_frontier_statistic_columns_and_the_asked_for_length():
    cloud = random_portfolios(quad_mu(), quad_cov(), 0.0, n=250, seed=3)
    assert list(cloud.columns) == list(FRONTIER_COLUMNS)
    assert len(cloud) == 250


def test_every_random_portfolios_sharpe_is_its_own_excess_return_over_its_own_volatility():
    cloud = random_portfolios(quad_mu(), quad_cov(), 0.02, n=400, seed=3)
    assert cloud["sharpe"].to_numpy() == pytest.approx(
        ((cloud["return"] - 0.02) / cloud["volatility"]).to_numpy(), rel=1e-12
    )


def test_no_random_portfolio_beats_the_uncapped_optimisers():
    # The cloud is drawn uniformly from the same long-only simplex the
    # optimisers search, so it bounds them from the wrong side: nothing sampled
    # can be less volatile than the minimum-variance point or sharper than the
    # tangency point.
    cloud = random_portfolios(quad_mu(), quad_cov(), 0.0, n=4000, seed=3)
    floor = min_variance(quad_mu(), quad_cov(), max_weight=1.0).volatility
    ceiling = max_sharpe(quad_mu(), quad_cov(), 0.0, max_weight=1.0).sharpe
    assert float(cloud["volatility"].min()) >= floor - 1e-9
    assert float(cloud["sharpe"].max()) <= ceiling + 1e-9


@pytest.mark.parametrize("count", [0, -5])
def test_random_portfolios_returns_an_empty_typed_frame_for_a_non_positive_count(count):
    cloud = random_portfolios(quad_mu(), quad_cov(), n=count)
    assert cloud.empty
    assert list(cloud.columns) == list(FRONTIER_COLUMNS)


def test_random_portfolios_declares_a_hard_ceiling_on_the_cloud_size():
    # A UI slip must not be able to allocate gigabytes.
    assert MAX_RANDOM_PORTFOLIOS == 200_000
    assert len(random_portfolios(quad_mu(), quad_cov(), n=10, seed=1)) == 10


def test_random_portfolios_of_an_unusable_universe_is_empty():
    assert random_portfolios(None, None).empty
    assert random_portfolios(diag_mu(), pd.DataFrame()).empty


# ---- bounds and feasibility -------------------------------------------------
@pytest.mark.parametrize("max_weight", [None, 0.30, 0.50, 1.0])
def test_every_optimiser_returns_a_fully_invested_long_only_portfolio(max_weight):
    for portfolio in all_optimisers(quad_mu(), quad_cov(), max_weight):
        total = sum(portfolio.weights.values())
        assert total == pytest.approx(1.0, abs=WEIGHT_TOLERANCE), portfolio.label
        assert min(portfolio.weights.values()) >= -1e-12, portfolio.label


@pytest.mark.parametrize("max_weight", [0.30, 0.50])
def test_every_optimiser_respects_a_feasible_weight_cap(max_weight):
    # 0.30 * 4 = 1.2 and 0.50 * 4 = 2.0 both clear 100%, so neither cap needs
    # widening and every name must honour it.
    for portfolio in all_optimisers(quad_mu(), quad_cov(), max_weight):
        if portfolio.label == LABEL_EQUAL_WEIGHT:
            continue  # 1/N is defined without reference to the cap
        assert max(portfolio.weights.values()) <= max_weight + 1e-9, portfolio.label


def test_an_unreachable_cap_is_widened_to_one_over_n_and_the_widening_is_noted():
    # A 10% cap across four names caps the book at 40%, so sum(w) = 1 is
    # unreachable; the cap is lifted to the tightest feasible value, 1/4 = 25%,
    # which pins every weight at 25% exactly.
    portfolio = min_variance(quad_mu(), quad_cov(), max_weight=0.1)
    assert portfolio.weights == pytest.approx(dict.fromkeys(QUAD_LABELS, 0.25), abs=1e-9)
    assert "Cap of 10.0% across 4 names cannot reach 100%" in portfolio.note
    assert "widened to 25.0%" in portfolio.note


@pytest.mark.parametrize(
    "optimiser",
    [
        lambda cov: min_variance(quad_mu(), cov, max_weight=0.1),
        lambda cov: max_sharpe(quad_mu(), cov, max_weight=0.1),
        lambda cov: risk_parity(cov, mu=quad_mu(), max_weight=0.1),
        lambda cov: max_diversification(cov, mu=quad_mu(), max_weight=0.1),
        lambda cov: target_return_portfolio(quad_mu(), cov, 0.08, max_weight=0.1),
    ],
    ids=["min_variance", "max_sharpe", "risk_parity", "max_diversification", "target"],
)
def test_an_infeasible_cap_is_widened_and_noted_in_every_optimiser(optimiser):
    portfolio = optimiser(quad_cov())
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)
    assert "widened" in portfolio.note


def test_a_feasible_floor_is_honoured_by_every_name():
    # A 20% floor and a 30% cap across four names bracket the budget
    # (4*0.20 = 0.8 <= 1 <= 1.2 = 4*0.30), so the programme is feasible as
    # stated and neither bound needs adjusting.
    portfolio = min_variance(quad_mu(), quad_cov(), max_weight=0.30, min_weight=0.20)
    assert min(portfolio.weights.values()) >= 0.20 - 1e-9
    assert max(portfolio.weights.values()) <= 0.30 + 1e-9
    assert portfolio.note == ""


def test_an_impossible_floor_is_dropped_to_zero_and_the_drop_is_noted():
    # A 40% floor sits above the shipped 35% cap, so it is dropped rather than
    # handed to the solver as an unsolvable programme.
    portfolio = min_variance(quad_mu(), quad_cov(), min_weight=0.40)
    assert "floor dropped to 0%" in portfolio.note
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)


def test_a_floor_above_a_cap_widens_the_cap_too_and_reports_both():
    portfolio = min_variance(quad_mu(), quad_cov(), max_weight=0.2, min_weight=0.3)
    assert "floor dropped to 0%" in portfolio.note
    assert "widened to 25.0%" in portfolio.note
    assert portfolio.weights == pytest.approx(dict.fromkeys(QUAD_LABELS, 0.25), abs=1e-9)


# ---- degenerate universes ---------------------------------------------------
@pytest.mark.parametrize("label", list(PORTFOLIO_LABELS))
def test_a_single_asset_universe_gets_the_whole_portfolio(label):
    # There is nothing to allocate: w == [1.0]. Note that the shipped 35% cap
    # has to be widened to 100% for that to even be expressible.
    mu, cov = single_asset_inputs()
    built = {
        LABEL_MIN_VARIANCE: min_variance(mu, cov),
        LABEL_MAX_SHARPE: max_sharpe(mu, cov),
        LABEL_RISK_PARITY: risk_parity(cov, mu=mu),
        LABEL_MAX_DIVERSIFICATION: max_diversification(cov, mu=mu),
        LABEL_EQUAL_WEIGHT: equal_weight(mu, cov),
    }[label]
    assert built.weights == pytest.approx({"SOLO": 1.0}, abs=1e-12)
    assert sum(built.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)


def test_a_single_asset_portfolio_is_scored_as_that_asset():
    # r = 0.10, sigma = sqrt(0.04) = 0.20, Sharpe = 0.10/0.20 = 0.50.
    mu, cov = single_asset_inputs()
    portfolio = max_sharpe(mu, cov)
    assert portfolio.expected_return == pytest.approx(0.10, rel=1e-12)
    assert portfolio.volatility == pytest.approx(0.20, rel=1e-12)
    assert portfolio.sharpe == pytest.approx(0.50, rel=1e-12)


def test_a_single_asset_frontier_is_a_single_row():
    mu, cov = single_asset_inputs()
    frontier = efficient_frontier(mu, cov, n_points=20)
    assert len(frontier) == 1
    assert frontier["volatility"].iloc[0] == pytest.approx(0.20, rel=1e-12)
    assert frontier["SOLO"].iloc[0] == pytest.approx(1.0, abs=1e-12)


def test_a_singular_covariance_still_gives_the_right_minimum_variance_answer():
    # The twins behave as one asset of variance 0.04 against SOLO at 0.01, so
    # the inverse-variance split is 25 : 100 — 20% across the twins, 80% in
    # SOLO — for sigma_p = 1/sqrt(25 + 100) = 1/sqrt(125) = 0.08944272.
    cov = singular_cov()
    assert float(np.min(np.linalg.eigvalsh(cov.to_numpy()))) == pytest.approx(0.0, abs=1e-15)
    portfolio = min_variance(None, cov, max_weight=1.0)
    twins = portfolio.weights["TWIN_A"] + portfolio.weights["TWIN_B"]
    assert twins == pytest.approx(0.20, abs=1e-6)
    assert portfolio.weights["SOLO"] == pytest.approx(0.80, abs=1e-6)
    assert portfolio.volatility == pytest.approx(1.0 / math.sqrt(125.0), rel=1e-7)


def test_a_singular_covariance_does_not_break_risk_parity():
    cov = singular_cov()
    portfolio = risk_parity(cov, max_weight=1.0)
    contributions = risk_contributions(weight_vector(portfolio, ("TWIN_A", "TWIN_B", "SOLO")), cov)
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)
    assert float(np.max(contributions)) / float(np.min(contributions)) < 1.05
    # Interchangeable assets must be treated interchangeably.
    assert portfolio.weights["TWIN_A"] == pytest.approx(portfolio.weights["TWIN_B"], abs=1e-6)


def test_an_all_zero_covariance_gives_zero_volatility_and_a_nan_sharpe():
    index = pd.Index(["P", "Q"])
    zero = pd.DataFrame(np.zeros((2, 2)), index=index, columns=index)
    portfolio = min_variance(pd.Series({"P": 0.10, "Q": 0.20}), zero, max_weight=1.0)
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)
    assert portfolio.volatility == 0.0
    assert math.isnan(portfolio.sharpe)  # never an infinite Sharpe on a report


@pytest.mark.parametrize(
    "optimiser",
    [
        lambda: min_variance(None, None),
        lambda: max_sharpe(None, None),
        lambda: risk_parity(None),
        lambda: max_diversification(None),
        lambda: equal_weight(None, None),
        lambda: target_return_portfolio(None, None, 0.10),
    ],
    ids=["min_variance", "max_sharpe", "risk_parity", "max_div", "equal_weight", "target"],
)
def test_an_empty_universe_degrades_to_a_reasoned_empty_portfolio(optimiser):
    portfolio = optimiser()
    assert portfolio.weights == {}
    assert math.isnan(portfolio.volatility)
    assert math.isnan(portfolio.sharpe)
    assert portfolio.converged is False
    assert portfolio.note  # a stated reason, never a silent blank


def test_duplicate_covariance_labels_are_de_duplicated_rather_than_raising():
    cov = quad_cov()
    cov.index = pd.Index(["ALFA", "ALFA", "CHRL", "DLTA"])
    cov.columns = pd.Index(["ALFA", "ALFA", "CHRL", "DLTA"])
    portfolio = min_variance(None, cov, max_weight=1.0)
    assert set(portfolio.weights) == {"ALFA", "CHRL", "DLTA"}
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)


def test_a_name_with_a_non_finite_covariance_row_is_dropped_not_fatal():
    cov = quad_cov()
    cov.iloc[1, :] = np.nan
    cov.iloc[:, 1] = np.nan
    portfolio = min_variance(quad_mu(), cov, max_weight=1.0)
    assert "BRVO" not in portfolio.weights
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)


def test_the_optimisers_do_not_mutate_the_inputs_they_are_handed():
    mu, cov = quad_mu(), quad_cov()
    mu_before, cov_before = mu.copy(deep=True), cov.copy(deep=True)
    all_optimisers(mu, cov, 0.4)
    efficient_frontier(mu, cov, n_points=6)
    random_portfolios(mu, cov, n=100)
    pd.testing.assert_series_equal(mu, mu_before)
    pd.testing.assert_frame_equal(cov, cov_before)


# ---- optimize_all: the orchestration --------------------------------------
def test_optimize_all_returns_every_documented_key():
    result = optimize_all(
        returns_panel(), 0.02, current_weights={"S0": 40, "S1": 30, "S2": 20, "S3": 10}
    )
    assert set(result) == {
        "mu",
        "cov",
        "frontier",
        "random",
        "portfolios",
        "current",
        "n_assets",
        "warnings",
    }
    assert set(result["portfolios"]) == set(PORTFOLIO_LABELS)
    assert result["n_assets"] == 4


def test_optimize_all_scores_every_portfolio_on_one_shared_mu_and_sigma():
    # The chart is only honest if every dot is scored on identical inputs, so
    # re-score the returned weights against the returned mu / Sigma and demand
    # the reported triple back.
    result = optimize_all(returns_panel(), 0.01)
    for label, portfolio in result["portfolios"].items():
        ret, vol, sharpe = portfolio_performance(
            portfolio.weights, result["mu"], result["cov"], 0.01
        )
        assert ret == pytest.approx(portfolio.expected_return, rel=1e-9), label
        assert vol == pytest.approx(portfolio.volatility, rel=1e-9), label
        assert sharpe == pytest.approx(portfolio.sharpe, rel=1e-9), label


def test_optimize_all_preserves_the_headline_optimality_identities():
    result = optimize_all(returns_panel(), 0.0, max_weight=0.5)
    portfolios = result["portfolios"]
    benchmark = portfolios[LABEL_EQUAL_WEIGHT]
    assert portfolios[LABEL_MAX_SHARPE].sharpe >= benchmark.sharpe - 1e-9
    assert portfolios[LABEL_MIN_VARIANCE].volatility <= benchmark.volatility + 1e-12
    assert float(result["frontier"]["volatility"].min()) >= (
        portfolios[LABEL_MIN_VARIANCE].volatility - 1e-9
    )


def test_optimize_all_weights_all_sum_to_one_within_the_documented_tolerance():
    result = optimize_all(returns_panel(), 0.0, max_weight=0.35)
    for label, portfolio in result["portfolios"].items():
        assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE), label
        assert max(portfolio.weights.values()) <= 0.35 + 1e-9 or label == LABEL_EQUAL_WEIGHT


def test_optimize_all_scores_the_current_book_on_the_same_inputs():
    result = optimize_all(
        returns_panel(), 0.0, current_weights={"S0": 40, "S1": 30, "S2": 20, "S3": 10}
    )
    current = result["current"]
    assert current.label == LABEL_CURRENT
    # 40/30/20/10 PERCENT normalises to fractions 0.4/0.3/0.2/0.1.
    assert current.weights == pytest.approx({"S0": 0.4, "S1": 0.3, "S2": 0.2, "S3": 0.1}, abs=1e-12)
    assert current.expected_return == pytest.approx(
        float(np.array([0.4, 0.3, 0.2, 0.1]) @ result["mu"].to_numpy()), rel=1e-12
    )


def test_optimize_all_names_a_current_holding_it_cannot_price():
    result = optimize_all(returns_panel(), current_weights={"S0": 0.5, "GHOST": 0.5})
    assert any("GHOST" in warning for warning in result["warnings"])


def test_optimize_all_reports_current_weights_that_sum_to_nothing():
    result = optimize_all(returns_panel(), current_weights={"S0": 0.0, "S1": 0.0})
    assert result["current"] is None
    assert any("do not sum to anything positive" in w for w in result["warnings"])


def test_optimize_all_warns_when_there_are_fewer_observations_than_the_rule_of_thumb():
    # Six rows for four assets is below the 2x rule: the sample covariance is
    # unstable and the module must SAY so in `warnings` rather than raising or
    # quietly optimising noise.
    assert COVARIANCE_OBSERVATION_RATIO == 2.0
    result = optimize_all(returns_panel(n_obs=6, n_assets=4))
    joined = " ".join(result["warnings"])
    assert "below the 2x rule of thumb" in joined
    assert f"below the {MIN_OBSERVATIONS} minimum" in joined
    # Degradation, not failure: a full suite of portfolios still comes back.
    assert set(result["portfolios"]) == set(PORTFOLIO_LABELS)


def test_optimize_all_warns_when_a_holding_has_no_usable_history():
    frame = pd.DataFrame({"GOOD": [0.01, -0.01, 0.02, -0.02, 0.01], "BLANK": [np.nan] * 5})
    result = optimize_all(frame)
    assert any("BLANK" in warning for warning in result["warnings"])
    assert result["n_assets"] == 1


@pytest.mark.parametrize(
    "panel",
    [None, pd.DataFrame(), pd.DataFrame({"A": [0.01]})],
    ids=["none", "empty", "single_row"],
)
def test_optimize_all_of_an_unusable_panel_returns_empty_structures_with_a_reason(panel):
    result = optimize_all(panel)
    assert result["portfolios"] == {}
    assert result["frontier"].empty
    assert result["mu"].empty
    assert result["cov"].empty
    assert list(result["random"].columns) == list(FRONTIER_COLUMNS)
    assert result["warnings"]


def test_optimize_all_is_deterministic_and_leaves_the_panel_untouched():
    panel = returns_panel()
    before = panel.copy(deep=True)
    first = optimize_all(panel, 0.01, max_weight=0.4)
    second = optimize_all(panel, 0.01, max_weight=0.4)
    pd.testing.assert_frame_equal(panel, before)
    pd.testing.assert_frame_equal(first["random"], second["random"])
    pd.testing.assert_frame_equal(first["frontier"], second["frontier"])
    for label in PORTFOLIO_LABELS:
        assert first["portfolios"][label].weights == second["portfolios"][label].weights


def test_optimize_all_is_invariant_to_the_order_of_the_observations():
    # mu is a product and Sigma a second moment; neither depends on row order,
    # so an unsorted panel must not move a single weight.
    panel = returns_panel(n_obs=200, n_assets=4)
    shuffled = panel.iloc[np.random.default_rng(9).permutation(len(panel))]
    ordered = optimize_all(panel, 0.0, max_weight=0.5)["portfolios"]
    scrambled = optimize_all(shuffled, 0.0, max_weight=0.5)["portfolios"]
    for label in PORTFOLIO_LABELS:
        assert scrambled[label].weights == pytest.approx(ordered[label].weights, abs=1e-9), label
