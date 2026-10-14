"""Black-Litterman — ``src.analytics.black_litterman``, offline and exact.

Black-Litterman has three properties that pin the implementation completely,
and they are worth more than any amount of smoke testing:

Verified here
-------------
**Reverse optimisation round-trips exactly.** ``Pi = delta*Sigma*w`` and
``w = (delta*Sigma)^-1 Pi`` are inverses, so market weights pushed through the
equilibrium step and back must return unchanged — asserted to 1e-12. This is
the *defining* property of equilibrium returns; an implementation that fails it
is not computing an equilibrium at all.

**No views leaves the prior untouched.** Not approximately: the posterior
returns equal the prior *exactly* and the covariance is unmodified. This is
what makes the model safe to leave switched on, since an asset nobody has an
opinion about keeps its equilibrium return instead of inheriting a historical
average's noise.

**Both posterior formulas agree.** The Woodbury form used in production and
the textbook master formula are algebraically identical; asserting they match
numerically (to ~1e-16) is strong evidence neither has drifted, and it is the
justification for using the form that never inverts ``Omega``.

Plus the behaviour that makes the model worth having
----------------------------------------------------
* A view propagates to assets it does not mention, through the covariance — the
  single feature that distinguishes Black-Litterman from overwriting a return.
* Confidence is monotone: a more confident view moves the posterior further,
  ``c -> 1`` enforces the view almost exactly, ``c -> 0`` ignores it, and
  ``c = 0.5`` reproduces He-Litterman's ``Omega``.
* A view that merely restates the equilibrium changes nothing, at any
  confidence.
* ``tau`` scales how far views move the answer.
* The posterior covariance stays symmetric and positive semi-definite, so the
  optimiser it feeds cannot receive an invalid matrix.

**Refusals.** A malformed covariance, all-zero weights, views naming unknown
assets, self-cancelling views — each reported, never a fabricated number.

Deterministic: a hand-built covariance, no RNG in the identities. No network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics import black_litterman as bl
from src.analytics import optimize as opt

NAMES = ["A", "B", "C", "D"]

#: A hand-built correlation matrix with deliberately uneven structure: C and D
#: are the most correlated pair (0.5), which is what lets the propagation test
#: below have something to detect.
CORR = np.array(
    [
        [1.00, 0.60, 0.30, 0.10],
        [0.60, 1.00, 0.40, 0.20],
        [0.30, 0.40, 1.00, 0.50],
        [0.10, 0.20, 0.50, 1.00],
    ]
)
VOLS = np.array([0.18, 0.22, 0.15, 0.28])
DELTA = 2.5


@pytest.fixture
def cov() -> pd.DataFrame:
    """Annualised covariance, as the optimiser and pipeline produce it."""
    return pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=NAMES, columns=NAMES)


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series([0.40, 0.25, 0.20, 0.15], index=NAMES)


# ---------------------------------------------------------------------------
# the defining identity
# ---------------------------------------------------------------------------
def test_reverse_optimisation_round_trips_market_weights(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Pi = delta*Sigma*w and w = (delta*Sigma)^-1 Pi are inverses. An
    implementation that fails this is not computing an equilibrium."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    recovered = bl.reverse_optimise_weights(cov, prior, DELTA)

    assert recovered.to_numpy() == pytest.approx(weights.to_numpy(), abs=1e-12)


def test_the_round_trip_holds_for_any_risk_aversion(cov: pd.DataFrame, weights: pd.Series) -> None:
    for delta in (0.5, 1.0, 2.5, 7.0, 25.0):
        prior = bl.market_implied_returns(cov, weights, delta)
        recovered = bl.reverse_optimise_weights(cov, prior, delta)
        assert recovered.to_numpy() == pytest.approx(weights.to_numpy(), abs=1e-10), delta


def test_equilibrium_returns_scale_linearly_with_risk_aversion(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Pi is linear in delta, so doubling aversion doubles the premium."""
    single = bl.market_implied_returns(cov, weights, 1.0)
    double = bl.market_implied_returns(cov, weights, 2.0)

    assert double.to_numpy() == pytest.approx(2.0 * single.to_numpy())


def test_the_riskiest_most_held_asset_earns_the_highest_equilibrium_premium(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Sanity on direction: Pi is Sigma*w, so premium follows covariance with
    the market portfolio, not standalone volatility."""
    prior = bl.market_implied_returns(cov, weights, DELTA)

    assert (prior > 0).all(), "a long-only market must imply positive premia"
    assert prior["B"] > prior["C"], "B is more volatile and more correlated with A"


def test_weights_are_normalised_before_reverse_optimisation(cov: pd.DataFrame) -> None:
    """Unnormalised weights must not scale the equilibrium."""
    normalised = bl.market_implied_returns(
        cov, pd.Series([0.4, 0.25, 0.2, 0.15], index=NAMES), DELTA
    )
    scaled = bl.market_implied_returns(cov, pd.Series([40.0, 25.0, 20.0, 15.0], index=NAMES), DELTA)

    assert scaled.to_numpy() == pytest.approx(normalised.to_numpy())


# ---------------------------------------------------------------------------
# risk aversion calibration
# ---------------------------------------------------------------------------
def test_implied_risk_aversion_is_premium_over_variance() -> None:
    """delta = 0.06 / 0.15^2 = 2.667 — the textbook illustration."""
    assert bl.implied_risk_aversion(0.06, 0.15**2) == pytest.approx(0.06 / 0.0225)


@pytest.mark.parametrize(
    "premium,variance", [(float("nan"), 0.02), (0.06, float("nan")), (0.06, 0.0), (0.06, -0.01)]
)
def test_an_uncalibratable_risk_aversion_is_nan_not_a_default(
    premium: float, variance: float
) -> None:
    """NaN so the CALLER decides whether to assume a default and say so, rather
    than this function quietly inventing an equilibrium."""
    assert math.isnan(bl.implied_risk_aversion(premium, variance))


def test_an_uncalibrated_delta_falls_back_and_says_so(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(cov, weights, [], risk_aversion=float("nan"))

    assert result.risk_aversion == bl.DEFAULT_RISK_AVERSION
    assert any("could not be calibrated" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# no views
# ---------------------------------------------------------------------------
def test_with_no_views_the_posterior_is_exactly_the_prior(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Exactly, not approximately. This is what makes the model safe to leave
    switched on."""
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA)

    assert result.ok
    assert result.posterior_returns.to_numpy() == pytest.approx(
        result.prior_returns.to_numpy(), abs=0.0
    )
    assert result.tilt().abs().max() == 0.0


def test_with_no_views_the_covariance_is_untouched(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA)

    assert result.posterior_covariance.to_numpy() == pytest.approx(cov.to_numpy())


def test_with_no_views_the_implied_weights_are_the_market_weights(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """The round-trip identity, surfaced on the result object."""
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA)

    assert result.implied_weights.to_numpy() == pytest.approx(weights.to_numpy(), abs=1e-10)


# ---------------------------------------------------------------------------
# the two formulas
# ---------------------------------------------------------------------------
def test_the_woodbury_and_textbook_posteriors_agree(cov: pd.DataFrame, weights: pd.Series) -> None:
    """Algebraically identical; agreeing numerically is what justifies using
    the form that never inverts Omega."""
    views = [bl.relative_view("A", "B", 0.05, 0.6), bl.absolute_view("D", 0.14, 0.8)]
    result = bl.black_litterman(cov, weights, views, risk_aversion=DELTA)

    textbook = bl.posterior_master_formula(
        result.prior_returns.to_numpy(),
        cov.to_numpy(),
        result.pick.to_numpy(),
        np.array([v.expected_return for v in result.views]),
        result.omega.to_numpy(),
        result.tau,
    )

    assert textbook == pytest.approx(result.posterior_returns.to_numpy(), abs=1e-12)


def test_the_two_formulas_agree_at_high_confidence(cov: pd.DataFrame, weights: pd.Series) -> None:
    """The case that breaks the textbook form in naive implementations: a
    confident view drives Omega towards singular."""
    views = [bl.absolute_view("A", 0.20, 0.99)]
    result = bl.black_litterman(cov, weights, views, risk_aversion=DELTA)

    textbook = bl.posterior_master_formula(
        result.prior_returns.to_numpy(),
        cov.to_numpy(),
        result.pick.to_numpy(),
        np.array([v.expected_return for v in result.views]),
        result.omega.to_numpy(),
        result.tau,
    )

    assert textbook == pytest.approx(result.posterior_returns.to_numpy(), rel=1e-6)


# ---------------------------------------------------------------------------
# what views actually do
# ---------------------------------------------------------------------------
def test_an_absolute_view_moves_its_asset_towards_the_view(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    prior = bl.market_implied_returns(cov, weights, DELTA)
    bullish = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", prior["A"] + 0.10, 0.8)], risk_aversion=DELTA
    )
    bearish = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", prior["A"] - 0.10, 0.8)], risk_aversion=DELTA
    )

    assert bullish.posterior_returns["A"] > prior["A"]
    assert bearish.posterior_returns["A"] < prior["A"]


def test_a_relative_view_moves_the_two_legs_in_opposite_directions(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.relative_view("A", "B", 0.08, 0.8)], risk_aversion=DELTA
    )

    tilt = result.tilt()
    assert tilt["A"] > 0
    assert tilt["B"] < 0


def test_a_view_propagates_to_an_asset_it_never_mentions(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """THE feature that distinguishes Black-Litterman from overwriting a
    return. C carries no view, but is 0.5-correlated with D, so a bullish view
    on D must lift C too."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("D", prior["D"] + 0.10, 0.9)], risk_aversion=DELTA
    )

    tilt = result.tilt()
    assert tilt["D"] > 0
    assert tilt["C"] > 0, "C is correlated 0.5 with D and must move with it"
    assert tilt["C"] < tilt["D"], "but by less than the asset actually viewed"


def test_propagation_is_ordered_by_correlation(cov: pd.DataFrame, weights: pd.Series) -> None:
    """D correlates 0.5 with C and 0.1 with A, so the spillover must rank the
    same way."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    tilt = bl.black_litterman(
        cov, weights, [bl.absolute_view("D", prior["D"] + 0.10, 0.9)], risk_aversion=DELTA
    ).tilt()

    assert tilt["C"] > tilt["A"]


def test_a_view_that_restates_the_equilibrium_changes_nothing(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """At any confidence. The model moves on SURPRISE, not on assertion."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    for confidence in (0.1, 0.5, 0.95):
        result = bl.black_litterman(
            cov,
            weights,
            [bl.absolute_view("A", float(prior["A"]), confidence)],
            risk_aversion=DELTA,
        )
        assert result.tilt().abs().max() == pytest.approx(0.0, abs=1e-12), confidence


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------
def test_more_confidence_moves_the_posterior_further(cov: pd.DataFrame, weights: pd.Series) -> None:
    prior = bl.market_implied_returns(cov, weights, DELTA)
    target = float(prior["A"]) + 0.10

    tilts = [
        bl.black_litterman(
            cov, weights, [bl.absolute_view("A", target, c)], risk_aversion=DELTA
        ).tilt()["A"]
        for c in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]

    assert tilts == sorted(tilts), tilts


def test_near_total_confidence_enforces_the_view_almost_exactly(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", 0.25, 1.0)], risk_aversion=DELTA
    )

    assert result.posterior_returns["A"] == pytest.approx(0.25, abs=1e-6)


def test_a_fully_confident_relative_view_is_enforced_as_a_spread(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """P E[R] = Q holds, which is the constraint a relative view states."""
    result = bl.black_litterman(
        cov, weights, [bl.relative_view("A", "B", 0.06, 1.0)], risk_aversion=DELTA
    )
    posterior = result.posterior_returns

    assert posterior["A"] - posterior["B"] == pytest.approx(0.06, abs=1e-6)


def test_zero_confidence_drops_the_view_and_says_so(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", 0.99, 0.0)], risk_aversion=DELTA
    )

    assert result.n_views == 0
    assert result.tilt().abs().max() == 0.0
    assert any("confidence is zero" in w for w in result.warnings)


def test_confidence_is_clamped_to_the_unit_interval() -> None:
    assert bl.absolute_view("A", 0.1, 5.0).confidence == 1.0
    assert bl.absolute_view("A", 0.1, -3.0).confidence == 0.0
    assert bl.absolute_view("A", 0.1, float("nan")).confidence == bl.DEFAULT_CONFIDENCE


def test_half_confidence_reproduces_the_he_litterman_omega(cov: pd.DataFrame) -> None:
    """0.5 is the neutral point, and it is neutral by construction rather than
    by convention: (1-0.5)/0.5 = 1."""
    pick = np.array([[1.0, -1.0, 0.0, 0.0]])
    matrix = cov.to_numpy()

    he = bl.omega_he_litterman(pick, matrix, bl.DEFAULT_TAU)
    idzorek = bl.omega_from_confidence(pick, matrix, [0.5], bl.DEFAULT_TAU)

    assert idzorek == pytest.approx(he)


def test_omega_is_monotone_decreasing_in_confidence(cov: pd.DataFrame) -> None:
    pick = np.array([[1.0, 0.0, 0.0, 0.0]])
    matrix = cov.to_numpy()

    variances = [
        float(bl.omega_from_confidence(pick, matrix, [c], bl.DEFAULT_TAU)[0, 0])
        for c in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]

    assert variances == sorted(variances, reverse=True), variances


def test_he_litterman_omega_is_proportional_to_view_variance(cov: pd.DataFrame) -> None:
    """A view on a volatile spread is automatically treated as less precise."""
    matrix = cov.to_numpy()
    quiet = bl.omega_he_litterman(np.array([[0.0, 0.0, 1.0, 0.0]]), matrix)  # C, 15% vol
    loud = bl.omega_he_litterman(np.array([[0.0, 0.0, 0.0, 1.0]]), matrix)  # D, 28% vol

    assert loud[0, 0] > quiet[0, 0]


# ---------------------------------------------------------------------------
# tau
# ---------------------------------------------------------------------------
def test_a_larger_tau_lets_views_move_the_answer_further(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """tau scales the prior's own uncertainty, so a bigger tau means the
    equilibrium is held less firmly. It has no agreed value, which is why it is
    exposed rather than buried."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    view = bl.absolute_view("A", float(prior["A"]) + 0.10, 0.5)

    # Omega is derived from tau too, so hold it fixed to isolate tau's effect.
    pick = np.array([[1.0, 0.0, 0.0, 0.0]])
    omega = bl.omega_he_litterman(pick, cov.to_numpy(), 0.05)

    tilts = [
        bl.black_litterman(cov, weights, [view], risk_aversion=DELTA, tau=t, omega=omega).tilt()[
            "A"
        ]
        for t in (0.01, 0.05, 0.20)
    ]

    assert tilts == sorted(tilts), tilts


def test_an_unusable_tau_falls_back_and_reports_it(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA, tau=-1.0)

    assert result.tau == bl.DEFAULT_TAU
    assert any("tau" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# the posterior covariance
# ---------------------------------------------------------------------------
def test_the_posterior_covariance_stays_symmetric(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.relative_view("A", "B", 0.05, 0.7)], risk_aversion=DELTA
    )
    matrix = result.posterior_covariance.to_numpy()

    assert matrix == pytest.approx(matrix.T)


def test_the_posterior_covariance_stays_positive_semi_definite(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """It is handed straight to the optimiser, which cannot accept an invalid
    matrix."""
    result = bl.black_litterman(
        cov,
        weights,
        [bl.relative_view("A", "B", 0.05, 0.9), bl.absolute_view("D", 0.18, 0.6)],
        risk_aversion=DELTA,
    )

    eigenvalues = np.linalg.eigvalsh(result.posterior_covariance.to_numpy())
    assert eigenvalues.min() > -1e-12, eigenvalues.min()


def test_the_posterior_covariance_is_at_least_the_prior(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Sigma_post = Sigma + M widens the matrix, because the posterior MEAN is
    itself uncertain. Variances must not shrink."""
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", 0.20, 0.5)], risk_aversion=DELTA
    )

    prior_var = np.diag(cov.to_numpy())
    posterior_var = np.diag(result.posterior_covariance.to_numpy())
    assert (posterior_var >= prior_var - 1e-15).all()


def test_the_prior_covariance_is_preserved_for_comparison(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA)

    assert result.prior_covariance.to_numpy() == pytest.approx(cov.to_numpy())


# ---------------------------------------------------------------------------
# views: construction and validation
# ---------------------------------------------------------------------------
def test_an_absolute_view_is_not_relative() -> None:
    view = bl.absolute_view("A", 0.12)

    assert not view.is_relative
    assert view.weights == (1.0,)
    assert "A returns 12.00%" in view.description


def test_a_relative_views_legs_sum_to_zero() -> None:
    view = bl.relative_view("A", "B", 0.03)

    assert view.is_relative
    assert sum(view.weights) == pytest.approx(0.0)
    assert "beats" in view.description


def test_a_custom_label_wins_over_the_generated_description() -> None:
    assert bl.relative_view("A", "B", 0.03, label="IT over banks").description == ("IT over banks")


@pytest.mark.parametrize(
    "view",
    [
        bl.View((), (), 0.05),
        bl.View(("A",), (1.0, 2.0), 0.05),
        bl.View(("A",), (1.0,), float("nan")),
        bl.View(("A",), (float("nan"),), 0.05),
        bl.View(("A",), (0.0,), 0.05),
    ],
)
def test_a_malformed_view_is_not_ok(view: bl.View) -> None:
    assert not view.ok


def test_a_view_naming_an_unknown_asset_is_dropped_and_explained(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """The user's missing opinion is the thing they would most want told."""
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("NOTHELD", 0.20, 0.8)], risk_aversion=DELTA
    )

    assert result.n_views == 0
    assert any("NOTHELD" in w and "not in the analysed universe" in w for w in result.warnings)


def test_a_self_cancelling_view_is_rejected(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.View(("A", "A"), (1.0, -1.0), 0.05, 0.8)], risk_aversion=DELTA
    )

    assert result.n_views == 0
    assert any("cancel out" in w for w in result.warnings)


def test_surviving_views_are_kept_when_others_are_dropped(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(
        cov,
        weights,
        [bl.absolute_view("NOTHELD", 0.2, 0.8), bl.absolute_view("A", 0.15, 0.8)],
        risk_aversion=DELTA,
    )

    assert result.n_views == 1
    assert result.warnings


def test_the_pick_matrix_has_one_row_per_view_and_one_column_per_asset(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    views = [bl.relative_view("A", "B", 0.05), bl.absolute_view("D", 0.14)]
    result = bl.black_litterman(cov, weights, views, risk_aversion=DELTA)

    assert result.pick.shape == (2, 4)
    assert list(result.pick.columns) == NAMES
    assert result.pick.iloc[0].to_numpy() == pytest.approx([1.0, -1.0, 0.0, 0.0])


def test_repeated_legs_accumulate_in_the_pick_row(cov: pd.DataFrame, weights: pd.Series) -> None:
    """A two-for-one bet on the same name is legitimate and must not be
    silently deduplicated."""
    result = bl.black_litterman(
        cov, weights, [bl.View(("A", "A", "B"), (1.0, 1.0, -1.0), 0.05, 0.7)], risk_aversion=DELTA
    )

    assert result.pick.iloc[0].to_numpy() == pytest.approx([2.0, -1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_the_view_impacts_table_shows_the_gap_the_model_acts_on(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """The most informative table in the model: what equilibrium implied, what
    the user said, and the difference the posterior is driven by."""
    prior = bl.market_implied_returns(cov, weights, DELTA)
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", 0.15, 0.7)], risk_aversion=DELTA
    )

    frame = result.view_impacts()
    assert len(frame) == 1
    assert float(frame["equilibrium"].iloc[0]) == pytest.approx(float(prior["A"]))
    assert float(frame["your view"].iloc[0]) == pytest.approx(0.15)
    assert float(frame["difference"].iloc[0]) == pytest.approx(0.15 - float(prior["A"]))


def test_the_view_impact_of_a_relative_view_is_the_equilibrium_spread(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    prior = bl.market_implied_returns(cov, weights, DELTA)
    result = bl.black_litterman(
        cov, weights, [bl.relative_view("A", "B", 0.05, 0.7)], risk_aversion=DELTA
    )

    expected = float(prior["A"] - prior["B"])
    assert float(result.view_impacts()["equilibrium"].iloc[0]) == pytest.approx(expected)


def test_the_summary_frame_carries_prior_posterior_and_tilt(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("A", 0.15, 0.7)], risk_aversion=DELTA
    )
    frame = result.summary_frame()

    assert list(frame.columns) == ["asset", "market_weight", "equilibrium", "posterior", "tilt"]
    assert list(frame["asset"]) == NAMES
    assert frame["tilt"].to_numpy() == pytest.approx(
        (frame["posterior"] - frame["equilibrium"]).to_numpy()
    )


def test_the_posterior_mu_adds_the_risk_free_rate_back(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """Black-Litterman works in excess returns; the optimiser wants total
    returns. Forgetting the rate understates every expected return by it."""
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA, risk_free_rate=0.065)

    assert result.posterior_mu().to_numpy() == pytest.approx(
        result.posterior_returns.to_numpy() + 0.065
    )
    assert result.prior_mu().to_numpy() == pytest.approx(result.prior_returns.to_numpy() + 0.065)


def test_the_posterior_mu_rate_can_be_overridden(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(cov, weights, [], risk_aversion=DELTA, risk_free_rate=0.065)

    assert result.posterior_mu(0.0).to_numpy() == pytest.approx(result.posterior_returns.to_numpy())


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame([[1.0, 2.0]], index=["A"], columns=["A", "B"]),  # not square
        pd.DataFrame([[np.nan, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"]),
    ],
)
def test_a_malformed_covariance_is_refused_with_a_reason(bad: object) -> None:
    result = bl.black_litterman(bad, pd.Series([0.5, 0.5], index=["A", "B"]))

    assert not result.ok
    assert result.warnings
    assert result.posterior_returns.empty


@pytest.mark.parametrize(
    "bad", [None, pd.Series(dtype="float64"), pd.Series([0.0, 0.0, 0.0, 0.0], index=NAMES)]
)
def test_unusable_prior_weights_are_refused(cov: pd.DataFrame, bad: object) -> None:
    result = bl.black_litterman(cov, bad, [])

    assert not result.ok
    assert result.warnings


def test_a_mismatched_covariance_index_is_refused() -> None:
    """A frame whose index and columns disagree is not a covariance matrix."""
    broken = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["C", "D"])

    assert not bl.black_litterman(broken, pd.Series([0.5, 0.5], index=["C", "D"])).ok


def test_an_unusable_result_reports_empty_not_zero() -> None:
    result = bl.black_litterman(None, None)

    assert not result.ok
    assert result.tilt().empty
    assert result.posterior_mu().empty
    assert result.view_impacts().empty
    assert result.summary_frame().empty


def test_market_implied_returns_of_nothing_is_empty() -> None:
    assert bl.market_implied_returns(None, None).empty
    assert bl.reverse_optimise_weights(None, None).empty


# ---------------------------------------------------------------------------
# cap weights
# ---------------------------------------------------------------------------
def test_cap_weights_are_proportional_to_market_cap() -> None:
    positions = pd.DataFrame({"ticker": ["A", "B", "C"], "market_cap": [6e11, 3e11, 1e11]})

    weights, warnings = bl.market_cap_weights(positions)

    assert weights.to_numpy() == pytest.approx([0.6, 0.3, 0.1])
    assert warnings == []


def test_cap_weights_sum_to_one() -> None:
    positions = pd.DataFrame({"ticker": ["A", "B"], "market_cap": [7.3e11, 2.1e11]})

    assert float(bl.market_cap_weights(positions)[0].sum()) == pytest.approx(1.0)


def test_a_holding_with_no_market_cap_is_excluded_and_the_coverage_reported() -> None:
    """Yahoo reports no market cap for most ETFs, and a cap weighting computed
    from half the book is not a cap weighting."""
    positions = pd.DataFrame({"ticker": ["A", "ETF", "B"], "market_cap": [6e11, np.nan, 4e11]})

    weights, warnings = bl.market_cap_weights(positions)

    assert weights["ETF"] == 0.0
    assert weights["A"] == pytest.approx(0.6)
    assert any("ETF" in w and "2 of 3" in w for w in warnings)


def test_cap_weights_fall_back_to_the_profiles() -> None:
    from src.models import SecurityProfile

    positions = pd.DataFrame({"ticker": ["A", "B"]})
    profiles = {
        "A": SecurityProfile("A", market_cap=8e11),
        "B": SecurityProfile("B", market_cap=2e11),
    }

    weights, _warnings = bl.market_cap_weights(positions, profiles)

    assert weights.to_numpy() == pytest.approx([0.8, 0.2])


def test_no_market_cap_anywhere_is_refused_with_an_alternative_suggested() -> None:
    positions = pd.DataFrame({"ticker": ["A", "B"], "market_cap": [np.nan, np.nan]})

    weights, warnings = bl.market_cap_weights(positions)

    assert weights.empty
    assert any("current weights or equal weights" in w for w in warnings)


@pytest.mark.parametrize("positions", [None, pd.DataFrame(), pd.DataFrame({"nope": [1]})])
def test_cap_weights_of_unusable_positions_are_refused(positions: object) -> None:
    weights, warnings = bl.market_cap_weights(positions)

    assert weights.empty
    assert warnings


# ---------------------------------------------------------------------------
# handoff to the optimiser
# ---------------------------------------------------------------------------
def test_the_posterior_feeds_the_optimiser_unchanged(cov: pd.DataFrame, weights: pd.Series) -> None:
    """The whole point: better inputs for the existing mean-variance code."""
    result = bl.black_litterman(
        cov,
        weights,
        [bl.relative_view("A", "B", 0.06, 0.7)],
        risk_aversion=DELTA,
        risk_free_rate=0.05,
    )

    portfolio = opt.max_sharpe(
        result.posterior_mu(), result.posterior_covariance, risk_free_rate=0.05
    )

    assert portfolio.converged
    assert sum(portfolio.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_a_bullish_view_raises_that_assets_optimised_weight(
    cov: pd.DataFrame, weights: pd.Series
) -> None:
    """End to end, the behaviour a user expects to see."""
    neutral = bl.black_litterman(cov, weights, [], risk_aversion=DELTA, risk_free_rate=0.05)
    bullish = bl.black_litterman(
        cov,
        weights,
        [bl.absolute_view("C", float(neutral.prior_returns["C"]) + 0.12, 0.9)],
        risk_aversion=DELTA,
        risk_free_rate=0.05,
    )

    before = opt.max_sharpe(neutral.posterior_mu(), neutral.posterior_covariance, 0.05)
    after = opt.max_sharpe(bullish.posterior_mu(), bullish.posterior_covariance, 0.05)

    assert after.weights["C"] > before.weights["C"]


def test_the_posterior_also_feeds_min_variance(cov: pd.DataFrame, weights: pd.Series) -> None:
    result = bl.black_litterman(
        cov, weights, [bl.absolute_view("D", 0.18, 0.6)], risk_aversion=DELTA
    )

    portfolio = opt.min_variance(result.posterior_mu(), result.posterior_covariance)

    assert portfolio.converged
