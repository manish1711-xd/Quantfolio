"""Factor structure — ``src.analytics.factors``.

This module answers the only question that matters about diversification: *how
many independent bets does the book actually own?* It answers it twice, with a
PCA of the correlation matrix and with hierarchical clustering on the
correlation distance. Both answers are pure linear algebra, which means both
can be pinned down exactly — so almost every expected number below is worked
out by hand in a comment beside the assertion.

Verified here
-------------
* **Spectrum identities that cannot rot.** ``trace(C) = k`` for a correlation
  matrix, so the eigenvalues sum to the number of assets and the explained
  variance ratios sum to exactly 1.0 and come back sorted descending. The
  cumulative curve is their running sum and ends at 1.0.
* **Hand-computed spectra.** A 2x2 correlation matrix with ``rho`` has
  eigenvalues ``1 +/- rho``: at ``rho = 0.6`` the split is exactly 80/20 and
  the loadings are ``sqrt(0.8)`` and ``sqrt(0.2)``. Three *perfectly*
  correlated assets give ``PC1 = 1.0`` and one effective bet; three *exactly
  uncorrelated* assets give ``1/3`` each and three effective bets. Those two
  extremes bracket everything the function can ever return.
* **Loadings contract.** One row per asset, ``L L' == C``, each asset's
  squared loadings summing to 1 across the full spectrum, and PC1 oriented so
  its dominant loading is positive (PC1 reads as "the market", never as
  "minus the market").
* **The correlation distance** ``d = sqrt(0.5 * (1 - rho))``: exactly 0 on the
  diagonal, ``sqrt(0.2)`` at ``rho = 0.6``, ``sqrt(0.5)`` at ``rho = 0`` and
  1.0 at ``rho = -1``. Reversing or rescaling that map would silently reorder
  every correlation heatmap in the product.
* **Clustering.** Two perfectly correlated names land in the same cluster; the
  default cut sits at distance 0.5, which *is* ``rho = 0.5``, so 0.6 merges
  and 0.4 does not. ``clustered_order`` is always a permutation of the input
  columns, so ``corr.loc[order, order]`` is safe.
* **Documented degradation.** One asset, an empty frame, ``None``, a single
  row, an all-NaN panel, a zero-variance column and assets that never traded
  on the same day each get a test pinning the empty / NaN / ``ok=False``
  answer rather than an exception.

Deterministic and offline: the only randomness is a locally seeded generator
and the seeded shared fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.factors import (
    CLUSTER_DISTANCE_THRESHOLD,
    CUMULATIVE_TARGET,
    DEFAULT_LINKAGE,
    EIGENVALUE_FLOOR,
    HIGH_CORRELATION,
    LINKAGE_METHODS,
    average_correlation,
    cluster_assets,
    clustered_order,
    correlation_distance,
    pca_decomposition,
    rolling_average_correlation,
)
from src.config import INSIGHTS, MIN_ROLLING_WINDOW

# Full double precision on purpose: a constant rounded at the 7th digit fails a
# 1e-12 comparison and then looks like an implementation bug.
SQRT_0_2 = 0.4472135954999579  # math.sqrt(0.2)  -- d at rho = +0.6
SQRT_0_5 = 0.7071067811865476  # math.sqrt(0.5)  -- d at rho =  0.0
SQRT_0_8 = 0.8944271909999159  # math.sqrt(0.8)  -- PC1 loading at rho = +0.6

INDEX_4 = pd.bdate_range("2023-01-02", periods=4)
INDEX_8 = pd.bdate_range("2023-01-02", periods=8)


# ---------------------------------------------------------------------------
# local, exactly-constructible samples
#
# u = [1, -1, 1, -1] and v = [1, 1, -1, -1] both sum to zero, are orthogonal
# (u . v = 0) and have the same norm (2). So for y = rho*u + sqrt(1-rho^2)*v,
#     corr(u, y) = (u . y) / (|u| |y|) = 4*rho / (2 * 2) = rho
# exactly -- no estimation error, no seed, no tolerance games.
# ---------------------------------------------------------------------------
U4 = np.array([1.0, -1.0, 1.0, -1.0])
V4 = np.array([1.0, 1.0, -1.0, -1.0])
W4 = np.array([1.0, -1.0, -1.0, 1.0])


def pair_with_correlation(rho: float, index: pd.Index = INDEX_4) -> pd.DataFrame:
    """Two return columns whose SAMPLE Pearson correlation is exactly ``rho``."""
    return pd.DataFrame({"AAA": U4, "BBB": rho * U4 + math.sqrt(1.0 - rho**2) * V4}, index=index)


def orthogonal_returns() -> pd.DataFrame:
    """Three columns whose sample correlation matrix is exactly the identity.

    u, v and w are mutually orthogonal zero-sum vectors, so every off-diagonal
    correlation is 0 by construction, not by luck.
    """
    return pd.DataFrame({"AAA": U4, "BBB": V4, "CCC": W4}, index=INDEX_4)


def perfectly_correlated_returns(days: int = 60) -> pd.DataFrame:
    """Three assets that are affine functions of ONE driver: rho = 1 everywhere.

    Scale and level differ (1x, 2x, 3x plus an offset) precisely so the test
    proves the decomposition runs on the CORRELATION matrix — a covariance PCA
    would hand back a different spectrum for the same three names.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.01, days)
    return pd.DataFrame(
        {"AAA": base, "BBB": 2.0 * base, "CCC": 3.0 * base + 0.001},
        index=pd.bdate_range("2023-01-02", periods=days),
    )


def three_block_returns() -> pd.DataFrame:
    """AAA/BBB correlated at exactly 0.9; CCC exactly uncorrelated with both."""
    u = np.concatenate([U4, U4])
    v = np.concatenate([V4, V4])
    w = np.concatenate([W4, W4])
    return pd.DataFrame(
        {"AAA": u, "BBB": 0.9 * u + math.sqrt(1.0 - 0.81) * v, "CCC": w},
        index=INDEX_8,
    )


@pytest.fixture
def corr_3x3() -> pd.DataFrame:
    """A hand-written correlation matrix: pairs are 0.6, 0.3 and 0.9."""
    return pd.DataFrame(
        [[1.0, 0.6, 0.3], [0.6, 1.0, 0.9], [0.3, 0.9, 1.0]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )


@pytest.fixture
def asset_corr(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the shared three-asset return panel (499 observations)."""
    return asset_returns.corr()


# ---- correlation distance ------------------------------------------------
def test_correlation_distance_is_zero_on_the_diagonal_and_sqrt_half_one_minus_rho_off_it():
    # rho = 0.6 -> d = sqrt(0.5 * (1 - 0.6)) = sqrt(0.2) = 0.4472135954999579
    distance = correlation_distance(pair_with_correlation(0.6).corr())
    assert distance.loc["AAA", "AAA"] == 0.0
    assert distance.loc["BBB", "BBB"] == 0.0
    assert distance.loc["AAA", "BBB"] == pytest.approx(SQRT_0_2, abs=1e-12)
    assert distance.loc["AAA", "BBB"] == pytest.approx(math.sqrt(0.2), rel=1e-15)


def test_correlation_distance_maps_the_three_reference_correlations_exactly():
    # The whole point of the metric, as three fixed points:
    #   rho = +1 -> sqrt(0.5 * 0) = 0     (same bet, zero distance)
    #   rho =  0 -> sqrt(0.5 * 1) = 0.70710678...
    #   rho = -1 -> sqrt(0.5 * 2) = 1     (perfect hedge, maximal distance)
    labels = ["AAA", "BBB"]
    for rho, expected in ((1.0, 0.0), (0.0, SQRT_0_5), (-1.0, 1.0)):
        matrix = pd.DataFrame([[1.0, rho], [rho, 1.0]], index=labels, columns=labels)
        distance = correlation_distance(matrix)
        assert distance.loc["AAA", "BBB"] == pytest.approx(expected, abs=1e-12)


def test_correlation_distance_is_symmetric_and_bounded_in_the_unit_interval(
    asset_corr: pd.DataFrame,
):
    distance = correlation_distance(asset_corr)
    values = distance.to_numpy(dtype=float)
    assert values == pytest.approx(values.T, abs=1e-15)
    assert values.min() >= 0.0
    assert values.max() <= 1.0
    assert np.diag(values) == pytest.approx(np.zeros(3), abs=1e-15)


def test_correlation_distance_is_monotonically_decreasing_in_correlation():
    # Identity: a more correlated pair is always a CLOSER pair. If this ever
    # inverts, every dendrogram and every heatmap ordering flips with it.
    labels = ["AAA", "BBB"]
    previous = None
    for rho in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0):
        matrix = pd.DataFrame([[1.0, rho], [rho, 1.0]], index=labels, columns=labels)
        d = float(correlation_distance(matrix).loc["AAA", "BBB"])
        if previous is not None:
            assert d < previous
        previous = d


def test_correlation_distance_preserves_the_label_order_of_the_input(
    corr_3x3: pd.DataFrame,
):
    distance = correlation_distance(corr_3x3)
    assert list(distance.index) == ["AAA", "BBB", "CCC"]
    assert list(distance.columns) == ["AAA", "BBB", "CCC"]


def test_correlation_distance_clips_an_impossible_correlation_and_mirrors_a_missing_cell():
    # Input is doubly broken: 2.5 is not a correlation, and the lower triangle
    # is missing. The contract is to mirror the supplied cell into the hole,
    # clip it to +1, and report d = sqrt(0.5 * (1 - 1)) = 0 -- never a NaN
    # cascade and never sqrt of a negative number.
    broken = pd.DataFrame([[1.0, 2.5], [np.nan, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    distance = correlation_distance(broken)
    assert distance.loc["AAA", "BBB"] == pytest.approx(0.0, abs=1e-15)
    assert distance.loc["BBB", "AAA"] == pytest.approx(0.0, abs=1e-15)


def test_correlation_distance_of_a_single_asset_is_a_one_by_one_zero():
    single = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    distance = correlation_distance(single)
    assert distance.shape == (1, 1)
    assert distance.iloc[0, 0] == 0.0


def test_correlation_distance_of_an_empty_frame_is_empty():
    assert correlation_distance(pd.DataFrame()).empty


def test_correlation_distance_of_none_is_empty():
    assert correlation_distance(None).empty


def test_correlation_distance_does_not_mutate_the_caller(corr_3x3: pd.DataFrame):
    before = corr_3x3.copy(deep=True)
    correlation_distance(corr_3x3)
    pd.testing.assert_frame_equal(corr_3x3, before)


# ---- average correlation -------------------------------------------------
def test_average_correlation_is_the_mean_of_the_strict_upper_triangle(
    corr_3x3: pd.DataFrame,
):
    # Pairs are (AAA,BBB)=0.6, (AAA,CCC)=0.3, (BBB,CCC)=0.9.
    # mean = (0.6 + 0.3 + 0.9) / 3 = 1.8 / 3 = 0.60
    # The diagonal 1.0s must NOT be counted; including them would give
    # (1.8 + 3) / 6 = 0.80 and flatter a concentrated book by a third.
    assert average_correlation(corr_3x3) == pytest.approx(0.60, abs=1e-12)


def test_average_correlation_of_perfectly_correlated_assets_is_one():
    corr = perfectly_correlated_returns().corr()
    assert average_correlation(corr) == pytest.approx(1.0, abs=1e-12)


def test_average_correlation_of_an_orthogonal_book_is_zero():
    assert average_correlation(orthogonal_returns().corr()) == pytest.approx(0.0, abs=1e-12)


def test_average_correlation_matches_the_value_pca_reports(
    asset_returns: pd.DataFrame, asset_corr: pd.DataFrame
):
    # Identity: the PCA result carries the same summary statistic, computed on
    # the same complete-case matrix, so the two must never disagree.
    result = pca_decomposition(asset_returns)
    assert result["average_correlation"] == pytest.approx(
        average_correlation(asset_corr), rel=1e-12
    )


def test_average_correlation_is_nan_for_fewer_than_two_assets():
    single = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    assert math.isnan(average_correlation(single))
    assert math.isnan(average_correlation(pd.DataFrame()))
    assert math.isnan(average_correlation(None))


def test_average_correlation_ignores_duplicate_labels():
    # The duplicated "AAA" column is dropped, leaving the single AAA/BBB pair
    # at 0.5 rather than triple-counting one name into the average.
    duplicated = pd.DataFrame(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        index=["AAA", "AAA", "BBB"],
        columns=["AAA", "AAA", "BBB"],
    )
    assert average_correlation(duplicated) == pytest.approx(0.5, abs=1e-12)


# ---- PCA: the spectrum ---------------------------------------------------
def test_explained_variance_ratios_sum_to_exactly_one(asset_returns: pd.DataFrame):
    # lambda_i / sum(lambda_j) is a partition of unity by construction; if this
    # ever drifts, the scree plot is lying about how much is left unexplained.
    ratios = pca_decomposition(asset_returns)["explained_variance_ratio"]
    assert ratios.sum() == pytest.approx(1.0, abs=1e-12)


def test_explained_variance_ratios_are_sorted_descending(asset_returns: pd.DataFrame):
    ratios = pca_decomposition(asset_returns)["explained_variance_ratio"]
    assert len(ratios) == 3
    assert list(ratios) == sorted(ratios, reverse=True)
    assert float(ratios.min()) >= 0.0


def test_cumulative_is_the_running_sum_of_the_ratios_and_ends_at_one(
    asset_returns: pd.DataFrame,
):
    result = pca_decomposition(asset_returns)
    ratios = result["explained_variance_ratio"]
    assert result["cumulative"] == pytest.approx(np.cumsum(ratios), abs=1e-15)
    assert float(result["cumulative"][-1]) == pytest.approx(1.0, abs=1e-12)


def test_eigenvalues_sum_to_the_number_of_assets(asset_returns: pd.DataFrame):
    # trace(C) = k for any correlation matrix (the diagonal is all ones), and
    # the trace is the sum of the eigenvalues. This is the single identity that
    # proves the decomposition ran on the CORRELATION and not the covariance.
    result = pca_decomposition(asset_returns)
    assert float(result["eigenvalues"].sum()) == pytest.approx(3.0, rel=1e-12)


def test_two_assets_correlated_at_zero_point_six_split_the_variance_eighty_twenty():
    # For a 2x2 correlation matrix [[1, r], [r, 1]] the eigenvalues are exactly
    # 1 + r and 1 - r. At r = 0.6:  lambda = 1.6 and 0.4, trace = 2.0, so
    #   ratio_1 = 1.6 / 2 = 0.80  and  ratio_2 = 0.4 / 2 = 0.20.
    result = pca_decomposition(pair_with_correlation(0.6))
    assert result["ok"] is True
    assert result["eigenvalues"] == pytest.approx([1.6, 0.4], abs=1e-12)
    assert result["explained_variance_ratio"] == pytest.approx([0.80, 0.20], abs=1e-12)
    assert result["pc1_ratio"] == pytest.approx(0.80, abs=1e-12)


def test_three_perfectly_correlated_assets_are_a_single_factor():
    # rho = 1 everywhere -> C is the all-ones matrix, rank 1, eigenvalues
    # (3, 0, 0). So PC1 explains 3/3 = 100% of the co-movement and the
    # participation ratio 1 / sum(ratio_i^2) = 1 / 1 = 1.0: three tickers,
    # one bet. This is the headline claim the diversification page makes.
    result = pca_decomposition(perfectly_correlated_returns())
    assert result["ok"] is True
    assert result["pc1_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert result["explained_variance_ratio"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert result["eigenvalues"] == pytest.approx([3.0, 0.0, 0.0], abs=1e-9)
    assert result["effective_bets"] == pytest.approx(1.0, abs=1e-9)
    assert result["components_for_target"] == 1


def test_three_uncorrelated_assets_split_the_variance_evenly():
    # C = I -> every eigenvalue is exactly 1, every ratio is 1/3, and the
    # participation ratio is 1 / (3 * (1/3)^2) = 3 -- the opposite pole from
    # the perfectly correlated case above.
    result = pca_decomposition(orthogonal_returns())
    assert result["explained_variance_ratio"] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)
    assert result["eigenvalues"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    assert result["effective_bets"] == pytest.approx(3.0, abs=1e-12)
    assert result["average_correlation"] == pytest.approx(0.0, abs=1e-12)


def test_effective_bets_is_the_inverse_participation_ratio(asset_returns: pd.DataFrame):
    result = pca_decomposition(asset_returns)
    ratios = result["explained_variance_ratio"]
    assert result["effective_bets"] == pytest.approx(1.0 / float(np.sum(ratios**2)), rel=1e-12)


def test_effective_bets_is_bracketed_by_one_and_the_asset_count(
    asset_returns: pd.DataFrame,
):
    # Identity: the participation ratio of a k-vector of non-negative weights
    # summing to 1 lies in [1, k] -- 1 when one component takes everything,
    # k when they share equally. Anything outside that is arithmetic failure.
    result = pca_decomposition(asset_returns)
    assert 1.0 <= result["effective_bets"] <= float(result["n_assets"])


def test_components_for_target_is_the_first_count_reaching_ninety_percent(
    asset_returns: pd.DataFrame,
):
    result = pca_decomposition(asset_returns)
    assert result["cumulative_target"] == pytest.approx(CUMULATIVE_TARGET, abs=1e-12)
    k = result["components_for_target"]
    cumulative = result["cumulative"]
    assert float(cumulative[k - 1]) >= CUMULATIVE_TARGET
    if k > 1:
        assert float(cumulative[k - 2]) < CUMULATIVE_TARGET


def test_n_components_truncates_the_loadings_but_never_the_spectrum(
    asset_returns: pd.DataFrame,
):
    # Contract: asking for one component must not make the scree plot claim
    # that one component explains everything.
    result = pca_decomposition(asset_returns, n_components=1)
    assert result["n_components"] == 1
    assert list(result["loadings"].columns) == ["PC1"]
    assert len(result["explained_variance_ratio"]) == 3
    assert result["explained_variance_ratio"].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("requested", "expected"), [(0, 1), (-4, 1), (2, 2), (99, 3), ("nonsense", 3), (None, 3)]
)
def test_n_components_is_clamped_into_one_to_n_assets(
    asset_returns: pd.DataFrame, requested, expected: int
):
    result = pca_decomposition(asset_returns, n_components=requested)
    assert result["n_components"] == expected
    assert result["loadings"].shape[1] == expected


# ---- PCA: the loadings ---------------------------------------------------
def test_loadings_have_exactly_one_row_per_asset(asset_returns: pd.DataFrame):
    result = pca_decomposition(asset_returns)
    assert list(result["loadings"].index) == result["assets"]
    assert result["loadings"].shape == (3, 3)
    assert result["loadings"].index.name == "ticker"
    assert list(result["loadings"].columns) == ["PC1", "PC2", "PC3"]


def test_each_assets_squared_loadings_sum_to_one_across_the_full_spectrum(
    asset_returns: pd.DataFrame,
):
    # loadings = V sqrt(L), so row i of (L L') is diag(C)_i = 1: each asset is
    # fully accounted for by the components. This is what makes a loading
    # readable as a CORRELATION between the asset and the component.
    loadings = pca_decomposition(asset_returns)["loadings"]
    assert (loadings**2).sum(axis=1).to_numpy() == pytest.approx(np.ones(3), abs=1e-12)
    assert loadings.abs().to_numpy().max() <= 1.0 + 1e-12


def test_the_loadings_reconstruct_the_correlation_matrix(asset_returns: pd.DataFrame):
    # C = V L V' = (V sqrt(L)) (V sqrt(L))' -- the full spectrum loses nothing.
    result = pca_decomposition(asset_returns)
    loadings = result["loadings"].to_numpy(dtype=float)
    rebuilt = loadings @ loadings.T
    original = asset_returns[result["assets"]].corr().to_numpy(dtype=float)
    assert rebuilt == pytest.approx(original, abs=1e-12)


def test_two_asset_loadings_are_hand_computable():
    # Eigenvectors of [[1, .6], [.6, 1]] are (1, 1)/sqrt(2) and (1, -1)/sqrt(2)
    # with eigenvalues 1.6 and 0.4, so
    #   PC1 loading = sqrt(1.6)/sqrt(2) = sqrt(0.8) = 0.8944271909999159
    #   PC2 loading = sqrt(0.4)/sqrt(2) = sqrt(0.2) = 0.4472135954999579
    # and 0.8 + 0.2 = 1 per asset, as the previous test requires in general.
    loadings = pca_decomposition(pair_with_correlation(0.6))["loadings"]
    assert loadings.loc["AAA", "PC1"] == pytest.approx(SQRT_0_8, abs=1e-12)
    assert loadings.loc["BBB", "PC1"] == pytest.approx(SQRT_0_8, abs=1e-12)
    assert abs(loadings.loc["AAA", "PC2"]) == pytest.approx(SQRT_0_2, abs=1e-12)
    # PC2 is the spread trade: the two assets must load on it with OPPOSITE
    # signs, otherwise it would be a second copy of PC1.
    assert loadings.loc["AAA", "PC2"] * loadings.loc["BBB", "PC2"] < 0.0


def test_pc1_is_oriented_so_its_largest_loading_is_positive(asset_returns: pd.DataFrame):
    # Eigenvectors are only defined up to sign. Pinning the dominant loading
    # positive is what makes PC1 read as "the market" instead of "minus the
    # market" -- and keeps the sign stable across LAPACK builds.
    for frame in (asset_returns, pair_with_correlation(0.6), three_block_returns()):
        loadings = pca_decomposition(frame)["loadings"]
        for column in loadings.columns:
            values = loadings[column].to_numpy(dtype=float)
            assert values[int(np.argmax(np.abs(values)))] >= 0.0


def test_a_positively_correlated_book_loads_on_pc1_with_one_sign(
    asset_returns: pd.DataFrame,
):
    # Regression guard on the shared fixture, whose assets share a common
    # market factor: PC1 must be the "everything together" component.
    result = pca_decomposition(asset_returns)
    assert result["average_correlation"] > 0.5
    assert (result["loadings"]["PC1"] > 0.5).all()


# ---- PCA: reported context ----------------------------------------------
def test_pca_reports_the_number_of_complete_case_observations(
    asset_returns: pd.DataFrame,
):
    # 500 prices -> 499 returns, and every asset trades on every date, so the
    # complete-case count is the full 499.
    result = pca_decomposition(asset_returns)
    assert result["n_obs"] == 499
    assert result["n_assets"] == 3
    assert result["assets"] == ["AAA", "BBB", "CCC"]


def test_pca_interpretation_quotes_the_pc1_share_and_the_observation_count(
    asset_returns: pd.DataFrame,
):
    result = pca_decomposition(asset_returns)
    text = result["interpretation"]
    assert text
    assert f"PC1 explains {result['pc1_ratio'] * 100:.0f}%" in text
    assert f"{result['n_obs']} overlapping daily observations" in text
    assert result["reason"] == ""


def test_pca_warns_when_the_sample_is_below_the_observation_minimum():
    # Four observations is enough for the algebra and nowhere near enough for
    # the inference; the interpretation has to say so out loud.
    result = pca_decomposition(pair_with_correlation(0.6))
    assert result["ok"] is True
    assert "indicative only" in result["interpretation"]


def test_pca_calls_a_perfectly_correlated_book_one_market_bet():
    result = pca_decomposition(perfectly_correlated_returns())
    assert "one market bet" in result["interpretation"]


# ---- PCA: degenerate input ----------------------------------------------
def test_pca_of_a_single_asset_degrades_to_an_empty_spectrum_with_a_reason(
    asset_returns: pd.DataFrame,
):
    result = pca_decomposition(asset_returns[["AAA"]])
    assert result["ok"] is False
    assert "at least two assets" in result["reason"]
    assert len(result["explained_variance_ratio"]) == 0
    assert len(result["cumulative"]) == 0
    assert result["n_components"] == 0
    assert math.isnan(result["pc1_ratio"])
    assert math.isnan(result["effective_bets"])
    assert math.isnan(result["average_correlation"])
    # The loadings frame still carries one row per asset -- the UI can bind to
    # it without a special case.
    assert list(result["loadings"].index) == ["AAA"]


def test_pca_of_an_empty_frame_degrades_without_raising():
    result = pca_decomposition(pd.DataFrame())
    assert result["ok"] is False
    assert result["assets"] == []
    assert result["n_assets"] == 0
    assert result["loadings"].empty


def test_pca_of_none_degrades_without_raising():
    result = pca_decomposition(None)
    assert result["ok"] is False
    assert result["reason"]


def test_pca_of_an_all_nan_panel_degrades_without_raising(nan_series: pd.Series):
    frame = pd.DataFrame({"AAA": nan_series, "BBB": nan_series})
    result = pca_decomposition(frame)
    assert result["ok"] is False
    assert result["assets"] == []


def test_pca_of_a_single_row_degrades_without_raising(asset_returns: pd.DataFrame):
    result = pca_decomposition(asset_returns.head(1))
    assert result["ok"] is False
    # One row cannot support a standard deviation, so the columns are dropped
    # before the overlap check ever runs.
    assert len(result["explained_variance_ratio"]) == 0


def test_pca_drops_a_zero_variance_column_rather_than_inventing_a_correlation(
    asset_returns: pd.DataFrame,
):
    # A constant series has no correlation with anything. Imputing one (0, or
    # worse, 1) would silently add a fake independent bet to the count.
    frame = asset_returns.copy()
    frame["FLAT"] = 0.0
    result = pca_decomposition(frame)
    assert result["assets"] == ["AAA", "BBB", "CCC"]
    assert result["n_assets"] == 3


def test_pca_of_assets_that_never_traded_on_the_same_day_degrades(
    asset_returns: pd.DataFrame,
):
    frame = asset_returns[["AAA", "BBB"]].copy()
    frame.iloc[:250, 0] = np.nan
    frame.iloc[250:, 1] = np.nan
    result = pca_decomposition(frame)
    assert result["ok"] is False
    assert "overlapping observations" in result["reason"]
    assert result["n_obs"] == 0


def test_pca_of_non_numeric_columns_degrades_without_raising():
    frame = pd.DataFrame({"AAA": ["a", "b", "c", "d"], "BBB": ["e", "f", "g", "h"]}, index=INDEX_4)
    result = pca_decomposition(frame)
    assert result["ok"] is False


def test_pca_ignores_infinities_instead_of_propagating_them(asset_returns: pd.DataFrame):
    frame = asset_returns.copy()
    frame.iloc[10, 0] = np.inf
    result = pca_decomposition(frame)
    assert result["ok"] is True
    # The infinite cell is voided to NaN and its whole row leaves the
    # complete-case sample; the other 498 dates still decompose.
    assert result["n_obs"] == 498
    assert np.isfinite(result["explained_variance_ratio"]).all()


def test_pca_is_invariant_to_row_order(asset_returns: pd.DataFrame):
    # Regression guard: a correlation is a sum over dates, so shuffling the
    # calendar must not move a single eigenvalue. (Contrast with the path
    # dependent statistics in src.analytics.risk, which sort first.)
    rng = np.random.default_rng(3)
    shuffled = asset_returns.iloc[rng.permutation(len(asset_returns))]
    assert not shuffled.index.is_monotonic_increasing
    assert pca_decomposition(shuffled)["explained_variance_ratio"] == pytest.approx(
        pca_decomposition(asset_returns)["explained_variance_ratio"], abs=1e-12
    )


def test_pca_does_not_mutate_the_caller(asset_returns: pd.DataFrame):
    before = asset_returns.copy(deep=True)
    pca_decomposition(asset_returns)
    pd.testing.assert_frame_equal(asset_returns, before)


def test_eigenvalue_floor_removes_numerical_noise_rather_than_reporting_it():
    # eigh of a singular matrix returns eigenvalues a few 1e-17 below zero.
    # A negative eigenvalue would produce a NaN loading via sqrt(); the floor
    # is what keeps the degenerate case reportable.
    assert EIGENVALUE_FLOOR > 0.0
    eigenvalues = pca_decomposition(perfectly_correlated_returns())["eigenvalues"]
    assert (eigenvalues >= 0.0).all()
    assert eigenvalues[1] == 0.0
    assert eigenvalues[2] == 0.0


# ---- hierarchical clustering --------------------------------------------
def test_two_perfectly_correlated_assets_land_in_the_same_cluster():
    # rho = 1 -> d = 0, so the pair merges at the very first (zero-height)
    # link no matter where the tree is cut. CCC, at rho = 0, cannot join them.
    rng = np.random.default_rng(11)
    driver = rng.normal(0.0, 0.01, 80)
    independent = rng.normal(0.0, 0.01, 80)
    frame = pd.DataFrame(
        {"AAA": driver, "BBB": 2.0 * driver, "CCC": independent},
        index=pd.bdate_range("2023-01-02", periods=80),
    )
    result = cluster_assets(frame.corr())
    labels = result["labels"]
    assert labels["AAA"] == labels["BBB"]
    assert labels["CCC"] != labels["AAA"]
    assert result["n_clusters"] == 2
    assert result["n_assets"] == 3


def test_the_default_cut_sits_exactly_at_rho_equals_one_half():
    # CLUSTER_DISTANCE_THRESHOLD = 0.5 and d = sqrt(0.5 * (1 - rho)), so
    #   rho = 0.5 -> d = sqrt(0.25) = 0.50  (exactly on the cut, merged)
    #   rho = 0.6 -> d = sqrt(0.20) = 0.447 (inside  the cut, merged)
    #   rho = 0.4 -> d = sqrt(0.30) = 0.548 (outside the cut, separate)
    assert CLUSTER_DISTANCE_THRESHOLD == 0.5
    assert 1.0 - 2.0 * CLUSTER_DISTANCE_THRESHOLD**2 == pytest.approx(0.5, abs=1e-15)
    for rho, expected_clusters in ((0.6, 1), (0.5, 1), (0.4, 2)):
        result = cluster_assets(pair_with_correlation(rho).corr())
        assert result["n_clusters"] == expected_clusters, rho
        assert result["threshold"] == pytest.approx(0.5, abs=1e-15)


def test_perfectly_anticorrelated_assets_are_never_the_same_cluster():
    # rho = -1 -> d = 1.0, the maximum possible distance. A hedge is not a
    # duplicate, however tightly the two prices move together in magnitude.
    labels = ["AAA", "BBB"]
    matrix = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], index=labels, columns=labels)
    result = cluster_assets(matrix)
    assert result["labels"]["AAA"] != result["labels"]["BBB"]
    assert result["n_clusters"] == 2


def test_requesting_n_clusters_forces_exactly_that_many_groups(asset_corr: pd.DataFrame):
    for wanted in (1, 2, 3):
        result = cluster_assets(asset_corr, n_clusters=wanted)
        assert result["n_clusters"] == wanted
        assert result["threshold"] == pytest.approx(float(wanted), abs=1e-15)
        assert len(set(result["labels"].values())) == wanted


def test_requesting_more_clusters_than_assets_is_clamped(asset_corr: pd.DataFrame):
    result = cluster_assets(asset_corr, n_clusters=99)
    assert result["n_clusters"] == 3


def test_cluster_labels_partition_every_asset_exactly_once(asset_corr: pd.DataFrame):
    # Identity: the clusters are a partition. Every asset carries exactly one
    # label, and the members listed under the labels are the assets again.
    result = cluster_assets(asset_corr)
    assert sorted(result["labels"]) == sorted(asset_corr.columns)
    flattened = [asset for members in result["clusters"].values() for asset in members]
    assert sorted(flattened) == sorted(asset_corr.columns)
    assert len(flattened) == len(set(flattened))
    assert set(result["clusters"]) == set(result["labels"].values())
    assert min(result["labels"].values()) == 1  # 1-based ids, not 0-based


def test_cluster_members_are_contiguous_in_the_leaf_order():
    # The whole reason `order` exists: drawing the heatmap in leaf order puts
    # each cluster on the diagonal as an unbroken block.
    result = cluster_assets(three_block_returns().corr())
    order = result["order"]
    labels = result["labels"]
    sequence = [labels[asset] for asset in order]
    seen: list[int] = []
    for label in sequence:
        if not seen or seen[-1] != label:
            assert label not in seen, "a cluster was split across the leaf order"
            seen.append(label)


def test_the_linkage_matrix_records_n_minus_one_merges(asset_corr: pd.DataFrame):
    result = cluster_assets(asset_corr)
    assert result["linkage"].shape == (2, 4)  # 3 assets -> 2 merges, 4 columns


def test_an_unknown_linkage_method_falls_back_to_average_instead_of_raising(
    asset_corr: pd.DataFrame,
):
    result = cluster_assets(asset_corr, method="not-a-linkage")
    assert result["method"] == DEFAULT_LINKAGE == "average"
    assert result["n_assets"] == 3


def test_every_declared_linkage_method_actually_runs(asset_corr: pd.DataFrame):
    for method in sorted(LINKAGE_METHODS):
        result = cluster_assets(asset_corr, method=method)
        assert result["method"] == method
        assert sorted(result["labels"]) == ["AAA", "BBB", "CCC"]


def test_cluster_assets_of_a_single_asset_is_one_trivial_cluster():
    single = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    result = cluster_assets(single)
    assert result["labels"] == {"AAA": 1}
    assert result["order"] == ["AAA"]
    assert result["n_clusters"] == 1
    assert result["clusters"] == {1: ["AAA"]}
    assert result["linkage"].shape == (0, 4)
    assert math.isnan(result["threshold"])
    assert "single holding is a single bet" in result["interpretation"]


def test_cluster_assets_of_an_empty_matrix_degrades_to_nothing():
    result = cluster_assets(pd.DataFrame())
    assert result["labels"] == {}
    assert result["order"] == []
    assert result["n_clusters"] == 0
    assert result["n_assets"] == 0
    assert result["linkage"].shape == (0, 4)
    assert "No usable correlation matrix" in result["interpretation"]


def test_cluster_assets_of_none_degrades_without_raising():
    assert cluster_assets(None)["n_clusters"] == 0


def test_cluster_interpretation_quotes_the_cut_and_the_average_correlation(
    asset_corr: pd.DataFrame,
):
    text = cluster_assets(asset_corr)["interpretation"]
    assert "distance cut of 0.50 (rho = 0.50)" in text
    assert f"{average_correlation(asset_corr):.2f}" in text


def test_cluster_assets_does_not_mutate_the_caller(asset_corr: pd.DataFrame):
    before = asset_corr.copy(deep=True)
    cluster_assets(asset_corr)
    pd.testing.assert_frame_equal(asset_corr, before)


# ---- clustered order -----------------------------------------------------
def test_clustered_order_is_a_permutation_of_the_input_columns(asset_corr: pd.DataFrame):
    order = clustered_order(asset_corr)
    assert sorted(order) == sorted(str(c) for c in asset_corr.columns)
    assert len(order) == len(set(order)) == asset_corr.shape[1]


def test_clustered_order_can_be_used_to_reindex_the_matrix(asset_corr: pd.DataFrame):
    # The documented calling contract for the heatmap: corr.loc[order, order].
    order = clustered_order(asset_corr)
    reordered = asset_corr.loc[order, order]
    assert reordered.shape == asset_corr.shape
    assert np.diag(reordered.to_numpy(dtype=float)) == pytest.approx(np.ones(3))


def test_clustered_order_places_the_correlated_pair_side_by_side():
    # AAA/BBB at rho = 0.9, CCC at rho = 0: whatever the leaf order, the pair
    # must be adjacent, otherwise the heatmap block is broken in two.
    order = clustered_order(three_block_returns().corr())
    assert abs(order.index("AAA") - order.index("BBB")) == 1


def test_clustered_order_of_a_single_asset_is_that_asset():
    single = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    assert clustered_order(single) == ["AAA"]


def test_clustered_order_of_an_empty_or_missing_matrix_is_empty():
    assert clustered_order(pd.DataFrame()) == []
    assert clustered_order(None) == []


# ---- rolling average correlation ----------------------------------------
def test_a_full_width_window_reproduces_the_static_average_correlation(
    asset_returns: pd.DataFrame, asset_corr: pd.DataFrame
):
    # Identity: one window covering the whole sample IS the static estimate.
    # The rolling path recomputes the correlations from rolling sums, so this
    # is a genuine cross-check of two different implementations.
    window = len(asset_returns)
    rolling = rolling_average_correlation(asset_returns, window)
    assert rolling.iloc[-1] == pytest.approx(average_correlation(asset_corr), rel=1e-9)


def test_the_first_window_minus_one_points_are_nan(asset_returns: pd.DataFrame):
    rolling = rolling_average_correlation(asset_returns, 63)
    assert rolling.head(62).isna().all()
    assert np.isfinite(rolling.iloc[62])
    assert len(rolling) == len(asset_returns)
    assert rolling.name == "avg_correlation"


def test_each_rolling_point_equals_the_average_correlation_of_its_own_window(
    asset_returns: pd.DataFrame,
):
    window = 63
    rolling = rolling_average_correlation(asset_returns, window)
    for position in (62, 200, len(asset_returns) - 1):
        slice_ = asset_returns.iloc[position - window + 1 : position + 1]
        assert rolling.iloc[position] == pytest.approx(average_correlation(slice_.corr()), rel=1e-9)


def test_rolling_average_correlation_is_bounded_by_plus_and_minus_one(
    asset_returns: pd.DataFrame,
):
    rolling = rolling_average_correlation(asset_returns, 63).dropna()
    assert not rolling.empty
    assert rolling.min() >= -1.0
    assert rolling.max() <= 1.0


def test_rolling_average_correlation_of_a_perfectly_correlated_book_is_one():
    rolling = rolling_average_correlation(perfectly_correlated_returns(), 20).dropna()
    assert rolling.to_numpy() == pytest.approx(np.ones(len(rolling)), abs=1e-9)


def test_rolling_average_correlation_needs_the_minimum_rolling_window(
    asset_returns: pd.DataFrame,
):
    # Even a short window is refused on a short sample: fewer than
    # MIN_ROLLING_WINDOW rows is not enough history to plot anything.
    assert MIN_ROLLING_WINDOW == 20
    assert rolling_average_correlation(asset_returns.head(19), 5).empty
    assert not rolling_average_correlation(asset_returns.head(20), 5).empty


@pytest.mark.parametrize("window", [0, 1, -5, "nonsense", None])
def test_rolling_average_correlation_refuses_a_meaningless_window(
    asset_returns: pd.DataFrame, window
):
    assert rolling_average_correlation(asset_returns, window).empty


def test_rolling_average_correlation_degrades_for_fewer_than_two_assets(
    asset_returns: pd.DataFrame,
):
    assert rolling_average_correlation(asset_returns[["AAA"]], 20).empty
    assert rolling_average_correlation(pd.DataFrame(), 20).empty
    assert rolling_average_correlation(None, 20).empty


def test_rolling_average_correlation_does_not_mutate_the_caller(
    asset_returns: pd.DataFrame,
):
    before = asset_returns.copy(deep=True)
    rolling_average_correlation(asset_returns, 63)
    pd.testing.assert_frame_equal(asset_returns, before)


# ---- module conventions (regression guards) ------------------------------
def test_the_high_correlation_threshold_matches_the_insight_engine():
    # Wording in the cluster interpretation has to fire off the same number
    # the insight engine uses, or the report contradicts itself.
    assert HIGH_CORRELATION == INSIGHTS.high_correlation == 0.80


def test_the_cumulative_target_is_ninety_percent():
    assert CUMULATIVE_TARGET == 0.90


def test_the_default_linkage_is_one_of_the_accepted_methods():
    assert DEFAULT_LINKAGE in LINKAGE_METHODS
    assert "ward" in LINKAGE_METHODS
