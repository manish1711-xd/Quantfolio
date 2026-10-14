"""Attribution — ``src.analytics.attribution``: who earned it, who carries the risk.

A weight column tells you what you own. This module tells you what that
ownership actually *did*, and it makes two promises a reviewer should be able to
check without trusting the implementation:

1. **Return contributions add up.** ``sum_i contribution_pct_i`` is exactly the
   portfolio's total return percentage, so every row reads as "this name put X
   percentage points on (or took X off) the headline number". A ranking that
   does not sum to the total is a league table, not an attribution.
2. **Risk contributions add up too**, which is the surprising one. Volatility is
   not additive across holdings, but because ``sigma_p = sqrt(w' S w)`` is
   homogeneous of degree one in ``w``, Euler's theorem splits it EXACTLY:
   ``sum_i CCTR_i == sigma_p`` and ``sum_i risk_contribution_pct_i == 100``.
   Both are asserted here at 1e-12, on hand-built matrices and on a sample
   covariance estimated from the shared price fixture.

The anchor case is arithmetic a reader can do in their head: three assets, equal
weights, equal variances, zero correlation. ``S = s^2 I`` and ``w_i = 1/3`` give
``MCTR_i = s/sqrt(3)``, ``CCTR_i = s/(3 sqrt(3))`` and ``rc_i = 100/3`` — exactly
one third each, with no special case anywhere in the module to produce it. Give
one of the three double the volatility and its risk share jumps to 66.67% while
its capital share stays at 33.33%; that gap is the entire point of the table.

Also verified
-------------
* WEIGHT HANDLING is a correctness issue, not a convenience: percent and
  fraction inputs must give identical answers, and weights must be aligned to
  ``cov.columns`` BY LABEL — a shuffled dict is fed in and the answer must not
  move, because a positional read would silently attribute one holding's risk
  to another and never say so.
* Concentration from four angles: the Gini is 0 for equal weights and climbs to
  its population supremum ``(n-1)/n`` as one name takes over; HHI and effective
  holdings are exact reciprocals; ``top1 <= top3 <= top5`` always.
* Documented degradation, one test each — empty, single asset, all-NaN
  covariance, zero variance, a singular matrix, a name held but not covered by
  the covariance matrix, duplicate ticker labels. Every one returns a typed
  empty frame or NaN. Nothing here raises, and nothing fabricates a zero.
* Sign and unit conventions as regression guards: component VaR is a NEGATIVE
  decimal loss, ``mctr``/``cctr`` carry the covariance matrix's own units, and
  ``weight_pct`` / ``risk_contribution_pct`` are PERCENTAGES summing to 100.

Deterministic and offline: every matrix is hand-built or derived from the seeded
shared fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.attribution import (
    COMPONENT_VAR_COLUMNS,
    CONCENTRATION_KEYS,
    RETURN_CONTRIBUTION_COLUMNS,
    RISK_CONTRIBUTION_COLUMNS,
    RISK_VS_WEIGHT_COLUMNS,
    TOP_N,
    attribution_summary,
    component_var,
    concentration_metrics,
    gini_coefficient,
    marginal_contribution_to_risk,
    return_contribution,
    risk_contribution,
)
from src.analytics.risk import covariance_matrix

# ---------------------------------------------------------------------------
# golden constants — written to full double precision on purpose. A constant
# rounded at the 8th digit silently fails a 1e-12 comparison and then looks
# like an implementation bug rather than a typo in the test.
# ---------------------------------------------------------------------------
#: Standard-normal quantile at the 95% loss tail, Phi^-1(0.05).
Z_95 = -1.6448536269514722
#: Standard-normal quantile at the 99% loss tail, Phi^-1(0.01).
Z_99 = -2.3263478740408408

#: Per-asset annualised volatility in the equal-variance anchor case.
IID_ASSET_VOL = 0.20
#: ...and its variance, the diagonal entry of ``S = s^2 I``.
IID_ASSET_VARIANCE = 0.04
#: sigma_p = 0.20 / sqrt(3) for three equal-weighted uncorrelated assets.
IID_PORTFOLIO_VOL = 0.11547005383792516
#: CCTR_i = 0.20 / (3 * sqrt(3)); three of these sum back to IID_PORTFOLIO_VOL.
IID_CCTR = 0.03849001794597505

TICKERS = ("AAA", "BBB", "CCC")


# ---------------------------------------------------------------------------
# local builders — hand-built matrices whose answers are computable by eye.
# Everything else rides on the seeded shared fixtures.
# ---------------------------------------------------------------------------
def diagonal_cov(variances: tuple[float, ...], tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    """A zero-correlation covariance matrix: ``S = diag(variances)``.

    Zero correlation is chosen deliberately for the golden cases. With a zero
    off-diagonal the Euler split collapses to

        rc_i = w_i^2 * var_i / sum_j (w_j^2 * var_j)

    which is a one-line hand computation, so every expected number below can be
    checked without a calculator — and without re-running the code under test.
    """
    return pd.DataFrame(
        np.diag(np.asarray(variances, dtype=float)),
        index=list(tickers),
        columns=list(tickers),
    )


def by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-key an output frame on ticker so two sort orders can be compared."""
    return frame.set_index("ticker").sort_index()


@pytest.fixture
def iid_cov() -> pd.DataFrame:
    """Three assets, 20% vol each, zero correlation — the anchor case."""
    return diagonal_cov((IID_ASSET_VARIANCE,) * 3)


@pytest.fixture
def loud_cov() -> pd.DataFrame:
    """AAA and BBB at 10% vol, CCC at 20% — one name with double the volatility."""
    return diagonal_cov((0.01, 0.01, 0.04))


@pytest.fixture
def sample_cov(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """The annualised, unshrunk sample covariance ``attribution_summary`` uses."""
    return covariance_matrix(asset_returns, shrinkage=0.0, annualize=True)


@pytest.fixture
def hand_positions() -> pd.DataFrame:
    """A three-name book whose whole attribution is checkable by eye.

    AAA  investment 1000, pnl  +200   -> +20% on its own cost
    BBB  investment 3000, pnl  -300   -> -10%
    CCC  investment 1000, pnl  +250   -> +25%
    -------------------------------------------------------------
    total investment 5000, total pnl +150 -> +3.0% for the portfolio
    """
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "investment": [1000.0, 3000.0, 1000.0],
            "pnl": [200.0, -300.0, 250.0],
            "pnl_pct": [20.0, -10.0, 25.0],
            "weight_pct": [24.0, 54.0, 22.0],
        }
    )


# ---------------------------------------------------------------------------
# return attribution — the additive identity
# ---------------------------------------------------------------------------
def test_return_contribution_emits_exactly_the_documented_schema(hand_positions):
    frame = return_contribution(hand_positions)
    assert list(frame.columns) == list(RETURN_CONTRIBUTION_COLUMNS)
    assert len(frame) == 3


def test_return_contribution_is_pnl_over_total_investment_in_percentage_points(
    hand_positions,
):
    # total investment = 1000 + 3000 + 1000 = 5000
    #   AAA:  +200 / 5000 * 100 = +4.0 pp
    #   BBB:  -300 / 5000 * 100 = -6.0 pp
    #   CCC:  +250 / 5000 * 100 = +5.0 pp
    frame = by_ticker(return_contribution(hand_positions))
    assert frame.loc["AAA", "contribution_pct"] == pytest.approx(4.0, abs=1e-12)
    assert frame.loc["BBB", "contribution_pct"] == pytest.approx(-6.0, abs=1e-12)
    assert frame.loc["CCC", "contribution_pct"] == pytest.approx(5.0, abs=1e-12)


def test_return_contributions_sum_to_the_portfolios_total_return_percentage(
    hand_positions,
):
    # THE identity. +4.0 - 6.0 + 5.0 = +3.0, and independently
    # total_pnl / total_investment * 100 = 150 / 5000 * 100 = 3.0.
    frame = return_contribution(hand_positions)
    assert frame["contribution_pct"].sum() == pytest.approx(3.0, abs=1e-12)
    assert frame.attrs["total_return_pct"] == pytest.approx(3.0, abs=1e-12)
    assert frame["contribution_pct"].sum() == pytest.approx(
        frame.attrs["total_return_pct"], abs=1e-12
    )


def test_the_summation_identity_holds_on_a_book_built_by_the_real_pipeline(
    holdings_frame, quotes, profiles
):
    # Same identity, but on positions produced by build_positions rather than
    # by hand: whatever it computes, the contributions must still reconstruct
    # its headline return exactly.
    from src.analytics.positions import build_positions

    positions, _problems = build_positions(holdings_frame, quotes, profiles)
    frame = return_contribution(positions)
    assert not frame.empty
    assert frame["contribution_pct"].sum() == pytest.approx(
        frame.attrs["total_return_pct"], rel=1e-12
    )
    assert frame.attrs["total_return_pct"] == pytest.approx(
        frame.attrs["total_pnl"] / frame.attrs["total_investment"] * 100.0, rel=1e-12
    )


def test_return_contribution_attrs_carry_the_totals_the_identity_sums_to(
    hand_positions,
):
    frame = return_contribution(hand_positions)
    assert frame.attrs["total_investment"] == pytest.approx(5000.0, abs=1e-12)
    assert frame.attrs["total_pnl"] == pytest.approx(150.0, abs=1e-12)


def test_contribution_shares_split_the_net_pnl_into_exactly_one_hundred_percent(
    hand_positions,
):
    # share_i = pnl_i / total_pnl * 100, total_pnl = +150:
    #   AAA   200 / 150 * 100 =  133.333...
    #   BBB  -300 / 150 * 100 = -200.0      (a detractor can exceed the whole)
    #   CCC   250 / 150 * 100 =  166.666...
    # 133.333 - 200 + 166.667 = 100 exactly.
    frame = by_ticker(return_contribution(hand_positions))
    assert frame.loc["AAA", "contribution_share_pct"] == pytest.approx(
        200.0 / 150.0 * 100.0, rel=1e-12
    )
    assert frame.loc["BBB", "contribution_share_pct"] == pytest.approx(-200.0, abs=1e-12)
    assert frame["contribution_share_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_return_contribution_ranks_the_biggest_contributor_first(hand_positions):
    # CCC (+5.0 pp) > AAA (+4.0 pp) > BBB (-6.0 pp)
    frame = return_contribution(hand_positions)
    assert list(frame["ticker"]) == ["CCC", "AAA", "BBB"]


def test_return_contribution_passes_weight_and_pnl_pct_through_untouched(
    hand_positions,
):
    # Regression guard: these two columns are build_positions' numbers, not
    # attribution's. Recomputing them here would let two definitions of
    # "weight" drift apart between the allocation page and this one.
    frame = by_ticker(return_contribution(hand_positions))
    assert frame.loc["BBB", "weight_pct"] == pytest.approx(54.0, abs=1e-12)
    assert frame.loc["BBB", "pnl_pct"] == pytest.approx(-10.0, abs=1e-12)


# ---------------------------------------------------------------------------
# return attribution — documented degradation
# ---------------------------------------------------------------------------
def test_contribution_share_is_nan_rather_than_a_fabricated_zero_when_pnl_nets_out():
    # +10 and -10 net to zero: the SPLIT of a zero P&L is undefined, not 0%.
    # The contribution against cost is still perfectly well defined:
    #   +10 / 200 * 100 = +5.0 and -10 / 200 * 100 = -5.0, summing to 0.0.
    frame = return_contribution(
        pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "pnl": [10.0, -10.0],
                "investment": [100.0, 100.0],
            }
        )
    )
    assert frame["contribution_share_pct"].isna().all()
    assert frame["contribution_pct"].to_numpy() == pytest.approx([5.0, -5.0], abs=1e-12)
    assert frame.attrs["total_return_pct"] == pytest.approx(0.0, abs=1e-12)


def test_contribution_is_nan_without_a_cost_base_but_the_share_still_works():
    # No ``investment`` column -> no denominator -> NaN, never a silent zero.
    frame = return_contribution(pd.DataFrame({"ticker": ["A", "B"], "pnl": [10.0, -5.0]}))
    assert frame["contribution_pct"].isna().all()
    assert math.isnan(frame.attrs["total_return_pct"])
    # 10 / 5 * 100 = 200 and -5 / 5 * 100 = -100, which still sum to 100.
    assert frame["contribution_share_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_a_row_with_an_unmeasurable_pnl_is_nan_and_leaves_the_identity_intact():
    # B's P&L could not be measured. Both the numerator and the total skip it,
    # so the surviving contributions still sum to the reported total return:
    #   total_pnl = 10 + 5 = 15 over investment 300 -> 5.0%
    #   10/300*100 + 5/300*100 = 3.333... + 1.666... = 5.0
    frame = return_contribution(
        pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "pnl": [10.0, np.nan, 5.0],
                "investment": [100.0, 100.0, 100.0],
            }
        )
    )
    assert frame["contribution_pct"].isna().sum() == 1
    assert frame["contribution_pct"].sum() == pytest.approx(
        frame.attrs["total_return_pct"], abs=1e-12
    )
    assert frame.attrs["total_return_pct"] == pytest.approx(5.0, abs=1e-12)


def test_a_non_numeric_pnl_becomes_nan_instead_of_stopping_the_pipeline():
    # A hand-typed holdings grid can contain anything; attribution coerces
    # rather than raising, because one bad cell must not blank a whole report.
    frame = return_contribution(
        pd.DataFrame({"ticker": ["A", "B"], "pnl": ["oops", 5.0], "investment": [100.0, 100.0]})
    )
    assert frame["contribution_pct"].isna().sum() == 1
    assert frame.attrs["total_pnl"] == pytest.approx(5.0, abs=1e-12)


@pytest.mark.parametrize(
    "positions",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"pnl": [1.0]}),  # no ticker column
        pd.DataFrame({"ticker": ["A"]}),  # no pnl column
    ],
)
def test_return_contribution_degrades_to_a_typed_empty_frame(positions):
    frame = return_contribution(positions)
    assert frame.empty
    assert list(frame.columns) == list(RETURN_CONTRIBUTION_COLUMNS)
    assert frame["contribution_pct"].dtype == np.dtype("float64")
    assert frame["ticker"].dtype == np.dtype("O")


def test_return_contribution_keeps_duplicate_ticker_rows_separate():
    # Two tax lots of the same name are two rows of the book. Collapsing them
    # here would hide a losing lot inside a profitable aggregate.
    frame = return_contribution(
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "pnl": [100.0, 50.0, -30.0],
                "investment": [300.0, 200.0, 500.0],
            }
        )
    )
    assert len(frame) == 3
    assert list(frame["ticker"]).count("AAA") == 2
    # (100 + 50 - 30) / 1000 * 100 = 12.0% of a 1000 cost base
    assert frame["contribution_pct"].sum() == pytest.approx(12.0, abs=1e-12)


def test_return_contribution_does_not_mutate_its_input(hand_positions):
    before = hand_positions.copy(deep=True)
    return_contribution(hand_positions)
    pd.testing.assert_frame_equal(hand_positions, before)


# ---------------------------------------------------------------------------
# marginal contribution to risk
# ---------------------------------------------------------------------------
def test_mctr_for_three_equal_uncorrelated_assets_is_sigma_over_sqrt_n(iid_cov):
    # S = 0.04 I, w = 1/3:  (S w)_i = 0.04/3 and sigma_p = sqrt(0.04/3),
    # so MCTR_i = (0.04/3) / sqrt(0.04/3) = sqrt(0.04/3) = 0.20 / sqrt(3).
    mctr = marginal_contribution_to_risk([1, 1, 1], iid_cov)
    assert list(mctr.index) == list(TICKERS)
    assert mctr.to_numpy() == pytest.approx([IID_PORTFOLIO_VOL] * 3, rel=1e-12)
    assert mctr.to_numpy() == pytest.approx([IID_ASSET_VOL / math.sqrt(3.0)] * 3, rel=1e-12)


def test_the_weighted_sum_of_mctr_is_the_portfolio_volatility(sample_cov):
    # Euler's theorem on a degree-one homogeneous function:
    #   sum_i w_i * d(sigma_p)/d(w_i) = sigma_p
    # Asserted on an estimated covariance matrix, where nothing is round.
    weights = pd.Series({"AAA": 40.0, "BBB": 35.0, "CCC": 25.0})
    mctr = marginal_contribution_to_risk(weights, sample_cov)
    fractions = (weights / weights.sum()).reindex(mctr.index)
    sigma_p = risk_contribution(weights, sample_cov).attrs["portfolio_volatility"]
    assert float((fractions * mctr).sum()) == pytest.approx(sigma_p, rel=1e-12)


def test_mctr_is_all_nan_for_a_zero_variance_book(iid_cov):
    # A riskless book has an UNDEFINED derivative of volatility, not a zero one.
    mctr = marginal_contribution_to_risk([1, 1, 1], iid_cov * 0.0)
    assert list(mctr.index) == list(TICKERS)
    assert mctr.isna().all()


@pytest.mark.parametrize("cov", [None, pd.DataFrame(), "not a matrix", 42])
def test_mctr_is_empty_for_an_unusable_covariance_matrix(cov):
    assert marginal_contribution_to_risk([1, 1, 1], cov).empty


# ---------------------------------------------------------------------------
# risk attribution — the Euler identity
# ---------------------------------------------------------------------------
def test_risk_contribution_emits_exactly_the_documented_schema(iid_cov):
    frame = risk_contribution([1, 1, 1], iid_cov)
    assert list(frame.columns) == list(RISK_CONTRIBUTION_COLUMNS)
    assert len(frame) == 3


def test_three_equal_uncorrelated_assets_each_carry_exactly_one_third_of_the_risk(
    iid_cov,
):
    # The sanity anchor, with no special case anywhere in the module:
    #   sigma_p = sqrt((1/3)^2 * 0.04 * 3) = sqrt(0.04/3) = 0.115470053837925
    #   MCTR_i  = 0.20 / sqrt(3)           = 0.115470053837925
    #   CCTR_i  = (1/3) * MCTR_i           = 0.038490017945975
    #   rc_i    = CCTR_i / sigma_p * 100   = 100/3 = 33.3333... exactly
    frame = risk_contribution([100 / 3, 100 / 3, 100 / 3], iid_cov)
    assert frame["mctr"].to_numpy() == pytest.approx([IID_PORTFOLIO_VOL] * 3, rel=1e-12)
    assert frame["cctr"].to_numpy() == pytest.approx([IID_CCTR] * 3, rel=1e-12)
    assert frame["risk_contribution_pct"].to_numpy() == pytest.approx([100.0 / 3.0] * 3, rel=1e-12)
    assert frame.attrs["portfolio_volatility"] == pytest.approx(IID_PORTFOLIO_VOL, rel=1e-12)


def test_component_risk_contributions_sum_to_the_portfolio_volatility(iid_cov):
    # sum_i CCTR_i = (w' S w) / sigma_p = sigma_p. Exact, not approximate:
    #   3 * 0.038490017945975 = 0.115470053837925
    frame = risk_contribution([1, 1, 1], iid_cov)
    assert frame["cctr"].sum() == pytest.approx(IID_PORTFOLIO_VOL, rel=1e-12)
    assert frame["cctr"].sum() == pytest.approx(frame.attrs["portfolio_volatility"], rel=1e-12)


def test_the_euler_identity_also_holds_on_an_estimated_covariance_matrix(sample_cov):
    # Correlated, unrounded, 499 observations of the seeded price fixture. If
    # the decomposition were an approximation, this is where it would show.
    frame = risk_contribution({"AAA": 40.0, "BBB": 35.0, "CCC": 25.0}, sample_cov)
    sigma_p = frame.attrs["portfolio_volatility"]
    assert sigma_p > 0.0
    assert frame["cctr"].sum() == pytest.approx(sigma_p, rel=1e-12)
    assert frame["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


@pytest.mark.parametrize(
    "weights",
    [
        [1, 1, 1],
        [70, 20, 10],
        {"AAA": 5.0, "BBB": 90.0, "CCC": 5.0},
        [0.34, 0.33, 0.33],
    ],
)
def test_risk_contribution_pct_always_sums_to_one_hundred(weights, sample_cov):
    frame = risk_contribution(weights, sample_cov)
    assert frame["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_normalised_weights_are_reported_in_percent_and_sum_to_one_hundred(loud_cov):
    # Input weights of 2/1/1 are 50%/25%/25% of the book once normalised.
    frame = by_ticker(risk_contribution([2, 1, 1], loud_cov))
    assert frame["weight_pct"].sum() == pytest.approx(100.0, abs=1e-10)
    assert frame.loc["AAA", "weight_pct"] == pytest.approx(50.0, abs=1e-12)
    assert frame.loc["BBB", "weight_pct"] == pytest.approx(25.0, abs=1e-12)


def test_the_portfolio_variance_attr_is_the_square_of_the_volatility_attr(sample_cov):
    frame = risk_contribution([40, 35, 25], sample_cov)
    assert frame.attrs["portfolio_variance"] == pytest.approx(
        frame.attrs["portfolio_volatility"] ** 2, rel=1e-12
    )


def test_the_asset_with_double_the_volatility_carries_more_risk_than_its_weight(
    loud_cov,
):
    # Equal weights (1/3 each) but CCC's variance is 0.04 against 0.01 for the
    # other two. With zero correlation rc_i = w_i^2 var_i / sum_j(w_j^2 var_j),
    # and the equal w_i^2 cancels, leaving the plain variance shares:
    #   CCC: 0.04 / (0.01 + 0.01 + 0.04) = 0.04/0.06 = 2/3 -> 66.667%
    #   AAA: 0.01 / 0.06                 = 1/6        -> 16.667%
    # So a third of the money carries two thirds of the risk.
    frame = by_ticker(risk_contribution([1, 1, 1], loud_cov))
    assert frame.loc["CCC", "weight_pct"] == pytest.approx(100.0 / 3.0, rel=1e-12)
    assert frame.loc["CCC", "risk_contribution_pct"] == pytest.approx(200.0 / 3.0, rel=1e-12)
    assert frame.loc["AAA", "risk_contribution_pct"] == pytest.approx(100.0 / 6.0, rel=1e-12)
    assert frame.loc["CCC", "risk_contribution_pct"] > frame.loc["CCC", "weight_pct"]
    # ...and the quiet names must give risk back, or the split would not close.
    assert frame.loc["AAA", "risk_contribution_pct"] < frame.loc["AAA", "weight_pct"]


def test_risk_contribution_ranks_the_biggest_risk_taker_first(loud_cov):
    frame = risk_contribution([1, 1, 1], loud_cov)
    assert frame["ticker"].iloc[0] == "CCC"
    assert frame["risk_contribution_pct"].is_monotonic_decreasing


def test_doubling_every_weight_changes_nothing_because_weights_are_normalised(
    sample_cov,
):
    # Scale invariance: only relative sizes can matter to a risk SHARE.
    single = risk_contribution([40, 35, 25], sample_cov)
    double = risk_contribution([80, 70, 50], sample_cov)
    pd.testing.assert_frame_equal(single, double)


# ---------------------------------------------------------------------------
# risk attribution — weight handling (percent, fractions, label alignment)
# ---------------------------------------------------------------------------
def test_percent_and_fraction_weights_give_identical_answers(loud_cov):
    # 50/30/20 and 0.50/0.30/0.20 are the same portfolio. If they were not
    # treated identically, a UI that stores fractions and a CSV that stores
    # percentages would disagree about the same book.
    as_percent = risk_contribution([50, 30, 20], loud_cov)
    as_fraction = risk_contribution([0.5, 0.3, 0.2], loud_cov)
    pd.testing.assert_frame_equal(as_percent, as_fraction)
    # Hand check, since both must be RIGHT and not merely equal:
    #   w^2 var = (0.25*0.01, 0.09*0.01, 0.04*0.04) = (0.0025, 0.0009, 0.0016)
    #   sum = 0.0050 -> rc = 50%, 18%, 32%
    frame = by_ticker(as_percent)
    assert frame.loc["AAA", "risk_contribution_pct"] == pytest.approx(50.0, rel=1e-12)
    assert frame.loc["BBB", "risk_contribution_pct"] == pytest.approx(18.0, rel=1e-12)
    assert frame.loc["CCC", "risk_contribution_pct"] == pytest.approx(32.0, rel=1e-12)


def test_weights_are_aligned_to_cov_columns_by_label_not_by_position(loud_cov):
    # The weights arrive DELIBERATELY SHUFFLED: index order CCC, AAA, BBB
    # against a covariance matrix ordered AAA, BBB, CCC. A positional read
    # would hand AAA the 20 that belongs to CCC and report 6.2/38.5/55.4
    # instead of 50/18/32 — a wrong answer with no error message anywhere.
    ordered = risk_contribution({"AAA": 50, "BBB": 30, "CCC": 20}, loud_cov)
    shuffled = risk_contribution({"CCC": 20, "AAA": 50, "BBB": 30}, loud_cov)
    pd.testing.assert_frame_equal(ordered, shuffled)
    assert by_ticker(shuffled).loc["AAA", "risk_contribution_pct"] == pytest.approx(50.0, rel=1e-12)


def test_a_shuffled_weight_series_is_aligned_by_label_too(loud_cov):
    shuffled = pd.Series([20.0, 50.0, 30.0], index=["CCC", "AAA", "BBB"])
    pd.testing.assert_frame_equal(
        risk_contribution(shuffled, loud_cov),
        risk_contribution({"AAA": 50, "BBB": 30, "CCC": 20}, loud_cov),
    )


def test_a_bare_sequence_with_no_shared_labels_is_aligned_positionally(loud_cov):
    # A plain list carries no names, so position is the only information there
    # is — and it must line up with cov.columns, in cov.columns' own order.
    positional = risk_contribution([50, 30, 20], loud_cov)
    labelled = risk_contribution({"AAA": 50, "BBB": 30, "CCC": 20}, loud_cov)
    pd.testing.assert_frame_equal(positional, labelled)


def test_reordering_the_covariance_matrix_does_not_move_any_names_risk(sample_cov):
    # Permuting BOTH axes consistently describes the same joint distribution,
    # so every per-name number must be unchanged.
    order = ["CCC", "AAA", "BBB"]
    weights = {"AAA": 40.0, "BBB": 35.0, "CCC": 25.0}
    permuted = risk_contribution(weights, sample_cov.loc[order, order])
    original = risk_contribution(weights, sample_cov)
    pd.testing.assert_frame_equal(by_ticker(permuted), by_ticker(original))


def test_a_name_missing_from_the_weights_counts_as_zero(loud_cov):
    # CCC is not mentioned; it is held at zero, not dropped, so the frame still
    # documents the whole covered universe.
    frame = by_ticker(risk_contribution({"AAA": 50, "BBB": 50}, loud_cov))
    assert frame.loc["CCC", "weight_pct"] == pytest.approx(0.0, abs=1e-12)
    assert frame.loc["CCC", "risk_contribution_pct"] == pytest.approx(0.0, abs=1e-12)
    assert frame["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_a_holding_the_covariance_does_not_cover_is_dropped_and_the_rest_renormalised(
    loud_cov,
):
    # ZZZ has no risk estimate, so it cannot be attributed. The remaining 80
    # points of weight are renormalised over the covered universe:
    #   w = (50/80, 30/80, 0) = (0.625, 0.375, 0)
    #   w^2 var = (0.390625*0.01, 0.140625*0.01, 0) = (0.00390625, 0.00140625)
    #   sum = 0.0053125 -> rc = 73.529412%, 26.470588%
    frame = by_ticker(risk_contribution({"AAA": 50, "BBB": 30, "ZZZ": 20}, loud_cov))
    assert "ZZZ" not in frame.index
    assert frame.loc["AAA", "weight_pct"] == pytest.approx(62.5, rel=1e-12)
    assert frame.loc["AAA", "risk_contribution_pct"] == pytest.approx(
        0.00390625 / 0.0053125 * 100.0, rel=1e-12
    )
    assert frame.loc["AAA", "risk_contribution_pct"] == pytest.approx(73.529411764705884, rel=1e-12)
    assert frame["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_duplicate_weight_labels_are_summed_into_one_exposure(loud_cov):
    # Two lots of AAA are one exposure of 50, not two competing entries.
    lots = pd.Series([30.0, 20.0, 50.0], index=["AAA", "AAA", "BBB"])
    merged = risk_contribution({"AAA": 50.0, "BBB": 50.0}, loud_cov)
    pd.testing.assert_frame_equal(risk_contribution(lots, loud_cov), merged)


@pytest.mark.parametrize(
    "weights",
    [
        None,
        [],
        [1, 2],  # wrong length, no labels to align on
        [0.0, 0.0, 0.0],  # sums to zero -> no portfolio to split
        pd.DataFrame({"w": [1, 1, 1]}),
    ],
)
def test_unusable_weights_degrade_to_a_typed_empty_frame(weights, loud_cov):
    frame = risk_contribution(weights, loud_cov)
    assert frame.empty
    assert list(frame.columns) == list(RISK_CONTRIBUTION_COLUMNS)


# Regression guard for a fixed defect:
# a NaN weight is filled with 0.0 on the label-alignment path but survives the positional
# path, where it makes the portfolio variance non-finite and NaNs out the entire
# decomposition. The same portfolio spelled two ways returns two different answers.
def test_a_nan_weight_is_read_the_same_way_however_the_weights_are_supplied(loud_cov):
    labelled = risk_contribution({"AAA": np.nan, "BBB": 50.0, "CCC": 50.0}, loud_cov)
    positional = risk_contribution([np.nan, 50.0, 50.0], loud_cov)
    pd.testing.assert_frame_equal(by_ticker(labelled), by_ticker(positional))


def test_risk_contribution_does_not_mutate_its_inputs(loud_cov):
    weights = pd.Series({"AAA": 50.0, "BBB": 30.0, "CCC": 20.0})
    cov_before = loud_cov.copy(deep=True)
    weights_before = weights.copy(deep=True)
    risk_contribution(weights, loud_cov)
    pd.testing.assert_frame_equal(loud_cov, cov_before)
    pd.testing.assert_series_equal(weights, weights_before)


# ---------------------------------------------------------------------------
# risk attribution — degenerate covariance matrices
# ---------------------------------------------------------------------------
def test_a_single_asset_carries_all_of_the_risk_and_no_diversification():
    # One name: sigma_p = sqrt(1 * 0.04) = 0.20, MCTR = 0.04 / 0.20 = 0.20,
    # CCTR = 1 * 0.20 = 0.20, rc = 100%.
    solo = pd.DataFrame([[0.04]], index=["AAA"], columns=["AAA"])
    frame = risk_contribution([100], solo)
    assert len(frame) == 1
    assert frame.attrs["portfolio_volatility"] == pytest.approx(0.20, rel=1e-12)
    assert frame["mctr"].iloc[0] == pytest.approx(0.20, rel=1e-12)
    assert frame["cctr"].iloc[0] == pytest.approx(0.20, rel=1e-12)
    assert frame["risk_contribution_pct"].iloc[0] == pytest.approx(100.0, abs=1e-12)


def test_a_singular_perfectly_correlated_pair_still_decomposes_exactly():
    # S = [[0.04, 0.04], [0.04, 0.04]] is singular (rho = 1, det = 0). Euler
    # needs no matrix inverse, so the split is still exact — and correct: two
    # identical assets at equal weight ARE one asset, sigma_p = 0.20, and each
    # carries half. An implementation that solved a linear system here would
    # raise or return garbage.
    singular = pd.DataFrame(
        [[0.04, 0.04], [0.04, 0.04]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    frame = risk_contribution([50, 50], singular)
    assert frame.attrs["portfolio_volatility"] == pytest.approx(0.20, rel=1e-12)
    assert frame["cctr"].to_numpy() == pytest.approx([0.10, 0.10], rel=1e-12)
    assert frame["risk_contribution_pct"].to_numpy() == pytest.approx([50.0, 50.0], rel=1e-12)
    assert frame["cctr"].sum() == pytest.approx(frame.attrs["portfolio_volatility"], rel=1e-12)


def test_a_singular_matrix_from_a_rank_deficient_return_panel_still_sums_to_sigma():
    # Two assets whose returns are an exact linear copy of one another, so the
    # sample covariance is rank 1 by construction.
    rng = np.random.default_rng(20240101)
    base = rng.normal(0.0004, 0.012, 260)
    panel = pd.DataFrame(
        {"AAA": base, "BBB": 2.0 * base},
        index=pd.bdate_range("2023-01-02", periods=260),
    )
    cov = covariance_matrix(panel, shrinkage=0.0, annualize=True)
    frame = risk_contribution([60, 40], cov)
    assert frame["cctr"].sum() == pytest.approx(frame.attrs["portfolio_volatility"], rel=1e-12)
    assert frame["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_a_zero_variance_book_reports_its_weights_with_nan_risk(iid_cov):
    # The honest answer: show WHAT is held, say the risk split is undefined.
    # A fabricated 33.3% each would claim a decomposition of nothing.
    frame = risk_contribution([1, 1, 1], iid_cov * 0.0)
    assert len(frame) == 3
    assert frame["weight_pct"].sum() == pytest.approx(100.0, abs=1e-10)
    assert frame[["mctr", "cctr", "risk_contribution_pct"]].isna().all().all()
    assert math.isnan(frame.attrs["portfolio_volatility"])
    assert math.isnan(frame.attrs["portfolio_variance"])


def test_a_covariance_matrix_carrying_a_nan_is_rejected_as_a_whole(sample_cov):
    # One NaN poisons every quadratic form built from the matrix, so patching
    # it with a zero would quietly understate the portfolio's risk.
    poisoned = sample_cov.copy()
    poisoned.iloc[0, 1] = np.nan
    assert risk_contribution([40, 35, 25], poisoned).empty


@pytest.mark.parametrize(
    "cov",
    [
        None,
        pd.DataFrame(),
        "not a matrix",
        np.zeros((0, 0)),
        np.array([1.0, 2.0, 3.0]),  # one-dimensional
    ],
)
def test_an_unusable_covariance_matrix_yields_a_typed_empty_frame(cov):
    frame = risk_contribution([1, 1, 1], cov)
    assert frame.empty
    assert list(frame.columns) == list(RISK_CONTRIBUTION_COLUMNS)


def test_a_non_square_covariance_matrix_is_rejected(sample_cov):
    assert risk_contribution([40, 35], sample_cov.iloc[:, :2]).empty


def test_a_bare_ndarray_covariance_matrix_is_labelled_positionally():
    # Accepted, but the tickers are then 0..n-1 — a documented fallback, not a
    # silent guess at names.
    frame = risk_contribution([50, 50], np.diag([0.04, 0.01]))
    assert list(frame["ticker"]) == ["0", "1"]
    # w^2 var = (0.25*0.04, 0.25*0.01) = (0.01, 0.0025); sum = 0.0125
    #   -> rc = 80% / 20%,  sigma_p = sqrt(0.0125) = 0.111803398874989
    assert frame["risk_contribution_pct"].to_numpy() == pytest.approx([80.0, 20.0], rel=1e-12)
    assert frame.attrs["portfolio_volatility"] == pytest.approx(math.sqrt(0.0125), rel=1e-12)


# ---------------------------------------------------------------------------
# component VaR
# ---------------------------------------------------------------------------
def test_component_var_emits_exactly_the_documented_schema(iid_cov):
    frame = component_var([1, 1, 1], iid_cov)
    assert list(frame.columns) == list(COMPONENT_VAR_COLUMNS)


def test_component_var_is_the_cctr_scaled_by_the_normal_loss_quantile(iid_cov):
    # z = Phi^-1(1 - 0.95) = -1.6448536269514722, horizon 1:
    #   CVaR_i = CCTR_i * z = 0.038490017945975 * -1.6448536269514722
    #          = -0.063310445619864
    frame = component_var([1, 1, 1], iid_cov, confidence=0.95, horizon_days=1)
    assert frame.attrs["z"] == pytest.approx(Z_95, abs=1e-12)
    assert frame["component_var"].to_numpy() == pytest.approx([IID_CCTR * Z_95] * 3, rel=1e-12)
    assert frame["component_var"].iloc[0] == pytest.approx(-0.06331044561986432, rel=1e-12)


def test_component_vars_sum_to_the_parametric_portfolio_var(iid_cov):
    # THE identity: sum_i CVaR_i = sigma_p * z * sqrt(h) = VaR_p
    #   0.115470053837925 * -1.6448536269514722 = -0.189931336859593
    frame = component_var([1, 1, 1], iid_cov)
    assert frame["component_var"].sum() == pytest.approx(frame.attrs["portfolio_var"], rel=1e-12)
    assert frame.attrs["portfolio_var"] == pytest.approx(IID_PORTFOLIO_VOL * Z_95, rel=1e-12)
    assert frame["component_var"].sum() == pytest.approx(-0.18993133685959296, rel=1e-12)


def test_component_vars_sum_to_the_portfolio_var_on_an_estimated_matrix(sample_cov):
    frame = component_var({"AAA": 40, "BBB": 35, "CCC": 25}, sample_cov, 0.99, 5)
    assert frame["component_var"].sum() == pytest.approx(frame.attrs["portfolio_var"], rel=1e-12)
    assert frame["component_var_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_component_var_is_a_negative_decimal_loss(sample_cov):
    # Sign convention shared with risk.value_at_risk: -0.021 reads "2.1% of
    # portfolio value". Flipping it would turn every VaR chart upside down.
    frame = component_var([40, 35, 25], sample_cov)
    assert (frame["component_var"] < 0.0).all()
    assert frame.attrs["portfolio_var"] < 0.0
    assert frame.attrs["z"] < 0.0


def test_component_var_pct_is_numerically_identical_to_risk_contribution_pct(
    sample_cov,
):
    # The documented limitation of a Gaussian decomposition: the z * sqrt(h)
    # scaling is common to every name, so this "tail share" is only the
    # volatility share wearing a different label. It cannot see a name whose
    # risk is skew or jump risk rather than variance.
    weights = {"AAA": 40, "BBB": 35, "CCC": 25}
    cvar = by_ticker(component_var(weights, sample_cov, 0.99, 10))
    rc = by_ticker(risk_contribution(weights, sample_cov))
    assert cvar["component_var_pct"].to_numpy() == pytest.approx(
        rc["risk_contribution_pct"].to_numpy(), rel=1e-12
    )


def test_component_var_scales_with_the_square_root_of_the_horizon(iid_cov):
    # sqrt(4) = 2 exactly, so a four-period VaR is twice a one-period VaR.
    one = component_var([1, 1, 1], iid_cov, horizon_days=1)
    four = component_var([1, 1, 1], iid_cov, horizon_days=4)
    assert four["component_var"].to_numpy() == pytest.approx(
        2.0 * one["component_var"].to_numpy(), rel=1e-12
    )
    assert four.attrs["portfolio_var"] == pytest.approx(2.0 * one.attrs["portfolio_var"], rel=1e-12)


def test_a_higher_confidence_deepens_the_loss_without_moving_the_shares(iid_cov):
    # |Phi^-1(0.01)| = 2.3263478740408408 > |Phi^-1(0.05)| = 1.6448536269514722
    at_95 = component_var([1, 1, 1], iid_cov, confidence=0.95)
    at_99 = component_var([1, 1, 1], iid_cov, confidence=0.99)
    assert at_99.attrs["z"] == pytest.approx(Z_99, abs=1e-12)
    assert at_99.attrs["portfolio_var"] < at_95.attrs["portfolio_var"] < 0.0
    assert at_99["component_var_pct"].to_numpy() == pytest.approx(
        at_95["component_var_pct"].to_numpy(), rel=1e-12
    )


def test_component_var_records_the_unit_convention_a_report_has_to_print(iid_cov):
    # Regression guard: the horizon scales cov's OWN frequency. Passing an
    # annualised matrix with horizon_days=1 reports a one-YEAR VaR, and the
    # attrs are what lets a report say so out loud.
    frame = component_var([1, 1, 1], iid_cov, confidence=0.95, horizon_days=1)
    assert "horizon_days" in frame.attrs["convention"]
    assert frame.attrs["confidence"] == pytest.approx(0.95, abs=1e-12)
    assert frame.attrs["horizon_days"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_component_var_rejects_a_confidence_outside_the_unit_interval(confidence, iid_cov):
    frame = component_var([1, 1, 1], iid_cov, confidence=confidence)
    assert frame.empty
    assert list(frame.columns) == list(COMPONENT_VAR_COLUMNS)


@pytest.mark.parametrize("horizon", [0, -5, float("nan")])
def test_component_var_rejects_a_non_positive_horizon(horizon, iid_cov):
    assert component_var([1, 1, 1], iid_cov, horizon_days=horizon).empty


def test_component_var_of_a_zero_variance_book_is_weights_only(iid_cov):
    frame = component_var([1, 1, 1], iid_cov * 0.0)
    assert len(frame) == 3
    assert frame["weight_pct"].sum() == pytest.approx(100.0, abs=1e-10)
    assert frame["component_var"].isna().all()
    assert math.isnan(frame.attrs["portfolio_var"])


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------
def test_gini_of_equal_weights_is_exactly_zero():
    # Every pairwise |w_i - w_j| is 0, so the Lorenz numerator is 0 for any n.
    assert gini_coefficient([25, 25, 25, 25]) == pytest.approx(0.0, abs=1e-15)
    assert gini_coefficient([10] * 10) == pytest.approx(0.0, abs=1e-15)
    assert gini_coefficient([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-15)


def test_gini_of_a_dominated_book_is_hand_computable():
    # Weights 97/1/1/1 -> fractions 0.97, 0.01, 0.01, 0.01.
    # Double-sum form: |0.97 - 0.01| = 0.96 for 6 ordered pairs -> 5.76;
    # denominator 2 * n^2 * mean(w) = 2 * 16 * 0.25 = 8.
    #   G = 5.76 / 8 = 0.72
    assert gini_coefficient([97, 1, 1, 1]) == pytest.approx(0.72, abs=1e-12)


def test_gini_of_a_one_name_book_reaches_its_population_supremum():
    # This is the POPULATION Gini, so its supremum for n names is (n-1)/n, not
    # 1. Two names with all the money in one:
    #   sorted [0, 1]; G = 2*(1*0 + 2*1)/(2*1) - 3/2 = 2 - 1.5 = 0.5
    assert gini_coefficient([0, 100]) == pytest.approx(0.5, abs=1e-12)
    assert gini_coefficient([0, 0, 0, 100]) == pytest.approx(0.75, abs=1e-12)


@pytest.mark.parametrize("n", [4, 10, 100])
def test_gini_approaches_one_as_a_single_name_takes_over(n):
    # One name a billion times larger than each of the others: the Gini sits a
    # rounding error below (n-1)/n, which is what "approaches 1" means for a
    # population Gini over a finite book — 0.75 for four names, 0.99 for a
    # hundred. It never actually reaches 1.
    weights = [1e9] + [1.0] * (n - 1)
    supremum = (n - 1) / n
    result = gini_coefficient(weights)
    assert result == pytest.approx(supremum, abs=1e-6)
    assert result < supremum


def test_gini_rises_monotonically_as_the_book_concentrates():
    scores = [gini_coefficient([x, 1, 1, 1]) for x in (1, 2, 5, 20, 100, 1e6)]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.0, abs=1e-15)
    assert scores[-1] < 0.75  # still below the four-name supremum


def test_gini_is_scale_free_between_percent_and_fraction():
    # Only relative sizes matter, so 60/40 and 0.6/0.4 must score identically.
    assert gini_coefficient([60, 40]) == pytest.approx(gini_coefficient([0.6, 0.4]), abs=1e-15)
    assert gini_coefficient([600, 400]) == pytest.approx(gini_coefficient([60, 40]), abs=1e-15)
    # sorted [0.4, 0.6]; G = 2*(1*0.4 + 2*0.6)/(2*1) - 3/2 = 1.6 - 1.5 = 0.1
    assert gini_coefficient([60, 40]) == pytest.approx(0.1, abs=1e-12)


def test_a_single_holding_scores_zero_which_is_why_hhi_is_reported_beside_it():
    # A one-element distribution is trivially "equal", so the Gini alone would
    # rate a single-stock portfolio as perfectly diversified. The HHI (1.0) is
    # what catches it — hence the pair, never one on its own.
    assert gini_coefficient([100]) == pytest.approx(0.0, abs=1e-15)
    solo = concentration_metrics(pd.DataFrame({"weight_pct": [100.0]}))
    assert solo["gini"] == pytest.approx(0.0, abs=1e-15)
    assert solo["hhi"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        None,
        [],
        [0.0, 0.0, 0.0],  # non-positive total
        [-10.0, 110.0],  # a short leg makes the Lorenz curve meaningless
        pd.DataFrame({"w": [1, 2]}),
    ],
)
def test_gini_is_nan_for_input_it_cannot_rank(weights):
    assert math.isnan(gini_coefficient(weights))


def test_gini_ignores_unmeasurable_weights_rather_than_failing():
    # NaN entries are dropped; the surviving 50/50 is perfectly equal.
    assert gini_coefficient([50.0, np.nan, 50.0]) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# concentration metrics
# ---------------------------------------------------------------------------
def test_concentration_returns_exactly_the_documented_keys(hand_positions):
    result = concentration_metrics(hand_positions)
    assert set(result) == set(CONCENTRATION_KEYS)
    assert isinstance(result["n_holdings"], int)


def test_equal_weights_score_zero_gini_and_one_over_n_hhi():
    # Ten equal names: HHI = 10 * 0.1^2 = 0.10, effective holdings = 10,
    # top1 = 10%, top3 = 30%, top5 = 50%, Gini = 0.
    result = concentration_metrics(pd.DataFrame({"weight_pct": [10.0] * 10}))
    assert result["hhi"] == pytest.approx(0.10, rel=1e-12)
    assert result["effective_holdings"] == pytest.approx(10.0, rel=1e-12)
    assert result["gini"] == pytest.approx(0.0, abs=1e-12)
    assert result["top1_pct"] == pytest.approx(10.0, rel=1e-12)
    assert result["top3_pct"] == pytest.approx(30.0, rel=1e-12)
    assert result["top5_pct"] == pytest.approx(50.0, rel=1e-12)
    assert result["n_holdings"] == 10


def test_hhi_and_effective_holdings_are_exact_reciprocals(hand_positions):
    # Identity by definition: effective_holdings = 1 / HHI. If these two ever
    # drift apart, one of them is being computed on un-normalised weights.
    result = concentration_metrics(hand_positions)
    assert result["hhi"] * result["effective_holdings"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        [100.0],
        [50.0, 50.0],
        [40.0, 30.0, 20.0, 10.0],
        [30, 20, 15, 15, 10, 5, 3, 2],
        [1.0] * 25,
    ],
)
def test_top1_is_at_most_top3_is_at_most_top5(weights):
    result = concentration_metrics(pd.DataFrame({"weight_pct": weights}))
    assert result["top1_pct"] <= result["top3_pct"] + 1e-12
    assert result["top3_pct"] <= result["top5_pct"] + 1e-12
    assert result["top5_pct"] <= 100.0 + 1e-12
    # ...and the effective count can never exceed the raw count.
    assert result["effective_holdings"] <= result["n_holdings"] + 1e-12


def test_topk_is_the_running_total_when_fewer_than_k_names_are_held():
    # Three names: top3 and top5 are both the whole book, not NaN and not a
    # padded 60%.
    result = concentration_metrics(pd.DataFrame({"weight_pct": [50.0, 30.0, 20.0]}))
    assert result["top1_pct"] == pytest.approx(50.0, rel=1e-12)
    assert result["top3_pct"] == pytest.approx(100.0, rel=1e-12)
    assert result["top5_pct"] == pytest.approx(100.0, rel=1e-12)


def test_concentration_is_hand_computable_for_an_uneven_book(hand_positions):
    # weights 24 / 54 / 22 (already summing to 100, so fractions are /100):
    #   HHI = 0.24^2 + 0.54^2 + 0.22^2 = 0.0576 + 0.2916 + 0.0484 = 0.3976
    #   eff = 1 / 0.3976 = 2.5150905432595575
    #   Gini: sorted [0.22, 0.24, 0.54]
    #         2*(1*0.22 + 2*0.24 + 3*0.54)/(3*1) - 4/3
    #       = 2*2.32/3 - 1.333333 = 1.546667 - 1.333333 = 0.2133333
    result = concentration_metrics(hand_positions)
    assert result["hhi"] == pytest.approx(0.3976, rel=1e-12)
    assert result["effective_holdings"] == pytest.approx(1.0 / 0.3976, rel=1e-12)
    assert result["gini"] == pytest.approx(2.0 * 2.32 / 3.0 - 4.0 / 3.0, abs=1e-12)
    assert result["gini"] == pytest.approx(0.21333333333333, abs=1e-9)
    assert result["top1_pct"] == pytest.approx(54.0, rel=1e-12)
    assert result["n_holdings"] == 3


def test_a_book_whose_weights_do_not_sum_to_one_hundred_is_renormalised():
    # A filtered subset (say, only the equity sleeve) must still be
    # self-consistent: two 30% names are 50/50 of what is being measured.
    partial = concentration_metrics(pd.DataFrame({"weight_pct": [30.0, 30.0]}))
    whole = concentration_metrics(pd.DataFrame({"weight_pct": [50.0, 50.0]}))
    assert partial["hhi"] == pytest.approx(whole["hhi"], rel=1e-12)
    assert partial["top1_pct"] == pytest.approx(50.0, rel=1e-12)


def test_negative_weights_are_dropped_before_anything_is_measured():
    # A short leg is excluded and the two remaining longs are renormalised to
    # 50/50 — so HHI is 0.5 and the Gini is 0, not NaN.
    result = concentration_metrics(pd.DataFrame({"weight_pct": [50.0, -10.0, 50.0]}))
    assert result["n_holdings"] == 2
    assert result["hhi"] == pytest.approx(0.5, rel=1e-12)
    assert result["gini"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "positions",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"ticker": ["AAA"]}),  # no weight_pct column
        pd.DataFrame({"weight_pct": [np.nan, np.nan]}),  # nothing measurable
    ],
)
def test_concentration_degrades_to_nan_without_raising(positions):
    result = concentration_metrics(positions)
    assert set(result) == set(CONCENTRATION_KEYS)
    assert result["n_holdings"] == 0
    assert all(math.isnan(result[key]) for key in CONCENTRATION_KEYS if key != "n_holdings")


def test_an_all_zero_book_reports_the_count_but_refuses_to_invent_ratios():
    # Weights summing to zero give no denominator, so every ratio is NaN — but
    # the holdings are still there and are still counted.
    result = concentration_metrics(pd.DataFrame({"weight_pct": [0.0, 0.0]}))
    assert result["n_holdings"] == 2
    assert math.isnan(result["hhi"])
    assert math.isnan(result["gini"])
    assert math.isnan(result["top1_pct"])


def test_concentration_does_not_mutate_its_input(hand_positions):
    before = hand_positions.copy(deep=True)
    concentration_metrics(hand_positions)
    pd.testing.assert_frame_equal(hand_positions, before)


# ---------------------------------------------------------------------------
# attribution_summary
# ---------------------------------------------------------------------------
def test_summary_carries_every_documented_section(hand_positions, sample_cov):
    summary = attribution_summary(hand_positions, cov=sample_cov)
    assert set(summary) == {
        "return_contribution",
        "risk_contribution",
        "concentration",
        "top_contributors",
        "top_detractors",
        "risk_vs_weight",
    }
    assert list(summary["risk_vs_weight"].columns) == list(RISK_VS_WEIGHT_COLUMNS)
    assert summary["concentration"] == concentration_metrics(hand_positions)


def test_summary_estimates_an_annualised_unshrunk_covariance_from_asset_returns(
    hand_positions, asset_returns, sample_cov
):
    # Regression guard on a documented modelling choice: attribution describes
    # the risk actually run, so the estimate is left UNSHRUNK here, unlike the
    # optimiser's (which inverts the matrix and therefore shrinks it).
    from_returns = attribution_summary(hand_positions, asset_returns=asset_returns)
    from_cov = attribution_summary(hand_positions, cov=sample_cov)
    pd.testing.assert_frame_equal(from_returns["risk_contribution"], from_cov["risk_contribution"])


def test_an_explicit_covariance_matrix_overrides_the_returns_panel(
    hand_positions, asset_returns, iid_cov
):
    summary = attribution_summary(hand_positions, asset_returns=asset_returns, cov=iid_cov)
    # With S = 0.04 I the risk shares collapse to w_i^2 / sum_j(w_j^2), which
    # the correlated sample covariance of the price fixture cannot reproduce.
    frame = by_ticker(summary["risk_contribution"])
    squares = np.array([0.24, 0.54, 0.22]) ** 2
    assert frame.loc["BBB", "risk_contribution_pct"] == pytest.approx(
        squares[1] / squares.sum() * 100.0, rel=1e-12
    )


def test_the_summation_identities_survive_the_summary(hand_positions, sample_cov):
    summary = attribution_summary(hand_positions, cov=sample_cov)
    returns = summary["return_contribution"]
    risk = summary["risk_contribution"]
    assert returns["contribution_pct"].sum() == pytest.approx(
        returns.attrs["total_return_pct"], abs=1e-12
    )
    assert risk["cctr"].sum() == pytest.approx(risk.attrs["portfolio_volatility"], rel=1e-12)
    assert risk["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)


def test_every_section_degrades_independently_of_the_others(hand_positions):
    # No covariance matrix and no returns panel: the risk half goes empty, the
    # return half and the concentration metrics are untouched.
    summary = attribution_summary(hand_positions)
    assert summary["risk_contribution"].empty
    assert summary["risk_vs_weight"].empty
    assert not summary["return_contribution"].empty
    assert summary["concentration"]["n_holdings"] == 3
    assert summary["return_contribution"]["contribution_pct"].sum() == pytest.approx(3.0, abs=1e-12)


def test_leaderboards_are_split_by_sign_and_capped_at_top_n():
    # Eight winners and three losers; only TOP_N of each may be shown, and a
    # winner must never appear among the detractors.
    positions = pd.DataFrame(
        {
            "ticker": list("ABCDEFGHIJK"),
            "pnl": [80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, -5.0, -15.0, -25.0],
            "investment": [100.0] * 11,
            "weight_pct": [100.0 / 11] * 11,
        }
    )
    summary = attribution_summary(positions)
    winners = summary["top_contributors"]
    losers = summary["top_detractors"]
    assert TOP_N == 5
    assert len(winners) == TOP_N
    assert len(losers) == 3
    assert list(winners["ticker"]) == ["A", "B", "C", "D", "E"]
    assert list(losers["ticker"]) == ["K", "J", "I"]
    assert (winners["contribution_pct"] > 0.0).all()
    assert (losers["contribution_pct"] < 0.0).all()


def test_leaderboards_fall_back_to_raw_pnl_when_there_is_no_cost_base():
    # Without ``investment`` there is no contribution to rank on, so ranking by
    # money made is the honest fallback — an empty table would be worse.
    positions = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "pnl": [8.0, -3.0, 5.0],
            "weight_pct": [40.0, 30.0, 30.0],
        }
    )
    summary = attribution_summary(positions)
    assert list(summary["top_contributors"]["ticker"]) == ["A", "C"]
    assert list(summary["top_detractors"]["ticker"]) == ["B"]
    assert summary["top_contributors"]["contribution_pct"].isna().all()


def test_a_book_with_no_losers_gets_an_empty_but_typed_detractor_table():
    positions = pd.DataFrame(
        {"ticker": ["A", "B"], "pnl": [5.0, 3.0], "investment": [100.0, 100.0]}
    )
    losers = attribution_summary(positions)["top_detractors"]
    assert losers.empty
    assert list(losers.columns) == list(RETURN_CONTRIBUTION_COLUMNS)


def test_risk_vs_weight_ratio_is_risk_share_over_capital_share(hand_positions, loud_cov):
    # ratio_i = risk_contribution_pct_i / weight_pct_i. Because the risk shares
    # sum to 100, sum_i(weight_pct_i * ratio_i) == 100 too — the ratio only
    # redistributes the same 100 points, it does not create new ones.
    rvw = attribution_summary(hand_positions, cov=loud_cov)["risk_vs_weight"]
    assert (rvw["weight_pct"] * rvw["ratio"]).sum() == pytest.approx(100.0, abs=1e-10)
    assert rvw["ratio"].is_monotonic_decreasing
    merged = by_ticker(rvw)
    assert merged.loc["AAA", "ratio"] == pytest.approx(
        merged.loc["AAA", "risk_contribution_pct"] / merged.loc["AAA", "weight_pct"],
        rel=1e-12,
    )


def test_risk_vs_weight_flags_the_volatile_name_above_one_and_the_diversifier_below(
    loud_cov,
):
    # Equal weights with CCC at double the volatility: its ratio is
    # 66.667 / 33.333 = 2.0, while the quiet names sit at 16.667/33.333 = 0.5.
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "weight_pct": [100 / 3, 100 / 3, 100 / 3],
            "pnl": [1.0, 1.0, 1.0],
            "investment": [100.0, 100.0, 100.0],
        }
    )
    rvw = by_ticker(attribution_summary(positions, cov=loud_cov)["risk_vs_weight"])
    assert rvw.loc["CCC", "ratio"] == pytest.approx(2.0, rel=1e-12)
    assert rvw.loc["AAA", "ratio"] == pytest.approx(0.5, rel=1e-12)


def test_a_zero_weight_holding_gets_a_nan_ratio_rather_than_an_infinity(loud_cov):
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "weight_pct": [60.0, 40.0, 0.0],
            "pnl": [1.0, 1.0, 0.0],
            "investment": [100.0, 100.0, 100.0],
        }
    )
    rvw = by_ticker(attribution_summary(positions, cov=loud_cov)["risk_vs_weight"])
    assert math.isnan(rvw.loc["CCC", "ratio"])


def test_a_holding_absent_from_the_covariance_matrix_drops_out_of_the_risk_split():
    # DDD is held but has no risk estimate. The covered names are renormalised
    # over 90 points of weight — w = (4/9, 3/9, 2/9) — and with
    # S = diag(0.04, 0.01, 0.01) the variance shares are, everything x81:
    #   AAA 16*0.04 = 0.64 | BBB 9*0.01 = 0.09 | CCC 4*0.01 = 0.04, sum 0.77
    #   -> 83.116883%, 11.688312%, 5.194805%
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "weight_pct": [40.0, 30.0, 20.0, 10.0],
            "pnl": [1.0, 2.0, 3.0, 4.0],
            "investment": [100.0, 100.0, 100.0, 100.0],
        }
    )
    summary = attribution_summary(positions, cov=diagonal_cov((0.04, 0.01, 0.01)))
    risk = by_ticker(summary["risk_contribution"])
    assert list(risk.index) == ["AAA", "BBB", "CCC"]
    assert risk.loc["AAA", "weight_pct"] == pytest.approx(400.0 / 9.0, rel=1e-12)
    assert risk.loc["AAA", "risk_contribution_pct"] == pytest.approx(0.64 / 0.77 * 100.0, rel=1e-12)
    assert risk.loc["CCC", "risk_contribution_pct"] == pytest.approx(0.04 / 0.77 * 100.0, rel=1e-12)
    assert risk["risk_contribution_pct"].sum() == pytest.approx(100.0, abs=1e-10)
    # The RETURN side still sees all four names — only the risk split shrinks.
    assert len(summary["return_contribution"]) == 4
    assert summary["concentration"]["n_holdings"] == 4


def test_duplicate_ticker_rows_are_one_risk_exposure_but_two_return_lots():
    # 30 + 20 of AAA is a single 50% exposure to AAA's volatility, while the
    # return attribution keeps the lots apart so a bad lot stays visible.
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "weight_pct": [30.0, 20.0, 50.0],
            "pnl": [100.0, 50.0, -30.0],
            "investment": [300.0, 200.0, 500.0],
        }
    )
    summary = attribution_summary(positions, cov=diagonal_cov((0.04, 0.01), ("AAA", "BBB")))
    risk = by_ticker(summary["risk_contribution"])
    assert len(risk) == 2
    assert risk.loc["AAA", "weight_pct"] == pytest.approx(50.0, rel=1e-12)
    # w^2 var = (0.25*0.04, 0.25*0.01) = (0.01, 0.0025) -> 80% / 20%
    assert risk.loc["AAA", "risk_contribution_pct"] == pytest.approx(80.0, rel=1e-12)
    assert len(summary["return_contribution"]) == 3


def test_positions_without_a_weight_column_still_produce_the_return_attribution(
    sample_cov,
):
    positions = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "pnl": [10.0, -5.0], "investment": [100.0, 100.0]}
    )
    summary = attribution_summary(positions, cov=sample_cov)
    assert summary["risk_contribution"].empty
    assert summary["concentration"]["n_holdings"] == 0
    # (10 - 5) / 200 * 100 = 2.5
    assert summary["return_contribution"]["contribution_pct"].sum() == pytest.approx(2.5, abs=1e-12)


def test_a_summary_of_nothing_is_empty_everywhere_and_still_the_right_shape():
    summary = attribution_summary(None)
    assert summary["return_contribution"].empty
    assert summary["risk_contribution"].empty
    assert summary["top_contributors"].empty
    assert summary["top_detractors"].empty
    assert summary["risk_vs_weight"].empty
    assert list(summary["risk_vs_weight"].columns) == list(RISK_VS_WEIGHT_COLUMNS)
    assert summary["concentration"]["n_holdings"] == 0
    assert math.isnan(summary["concentration"]["hhi"])


def test_attribution_summary_does_not_mutate_its_inputs(hand_positions, asset_returns, sample_cov):
    positions_before = hand_positions.copy(deep=True)
    returns_before = asset_returns.copy(deep=True)
    cov_before = sample_cov.copy(deep=True)
    attribution_summary(hand_positions, asset_returns=asset_returns, cov=sample_cov)
    attribution_summary(hand_positions, asset_returns=asset_returns)
    pd.testing.assert_frame_equal(hand_positions, positions_before)
    pd.testing.assert_frame_equal(asset_returns, returns_before)
    pd.testing.assert_frame_equal(sample_cov, cov_before)


def test_the_whole_module_survives_a_positions_frame_of_one_row(sample_cov):
    # One holding: 100% of the weight, 100% of the risk, a Gini of 0 but an
    # HHI of 1 — the two concentration numbers disagreeing on purpose.
    positions = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "weight_pct": [100.0],
            "pnl": [250.0],
            "investment": [1000.0],
        }
    )
    summary = attribution_summary(positions, cov=sample_cov.iloc[:1, :1])
    assert summary["return_contribution"]["contribution_pct"].iloc[0] == pytest.approx(
        25.0, abs=1e-12
    )
    assert summary["risk_contribution"]["risk_contribution_pct"].iloc[0] == pytest.approx(
        100.0, abs=1e-10
    )
    assert summary["concentration"]["hhi"] == pytest.approx(1.0, rel=1e-12)
    assert summary["concentration"]["effective_holdings"] == pytest.approx(1.0, rel=1e-12)
    assert summary["concentration"]["gini"] == pytest.approx(0.0, abs=1e-12)
