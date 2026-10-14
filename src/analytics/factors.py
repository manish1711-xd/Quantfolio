"""Factor structure: how many independent bets a portfolio actually owns.

A weight table shows what you hold; it does not show what you are *exposed
to*. Twenty names that all load on the same market factor are one bet wearing
twenty name tags. This module makes that visible two ways.

1. PCA on the CORRELATION matrix (i.e. on standardised returns, so the
   decomposition describes co-movement and not volatility scale):

       C = V L V'                     C symmetric, positive semi-definite
       explained_variance_ratio_i   = lambda_i / sum_j(lambda_j)
       loadings                     = V * sqrt(L)   (asset-to-PC correlation)
       effective bets               = 1 / sum_i(ratio_i^2)

   For a k-asset correlation matrix trace(C) = k, so the eigenvalues sum to k
   and ratio_1 is the share of common movement a single latent factor
   explains. The effective-bets participation ratio equals k for an identity
   correlation matrix (k genuinely independent assets) and 1.0 when every
   asset is the same bet.

2. Hierarchical clustering on the standard correlation distance

       d_ij = sqrt(0.5 * (1 - rho_ij))          in [0, 1]

   which is a true metric on standardised return vectors: d = 0 at rho = +1,
   0.5 at rho = +0.5, 0.707 at rho = 0 and 1.0 at rho = -1. Average linkage on
   the condensed form of that matrix yields both the cluster labels and the
   leaf order the correlation heatmap should be drawn in.

Conventions and guarantees
--------------------------
* Nothing here mutates its input; every frame is copied before use.
* Correlations for the PCA are estimated on COMPLETE CASES (dates on which
  every asset traded). That costs a few observations but guarantees a
  positive semi-definite matrix, so the eigenvalues are real and >= 0.
* Assets that cannot support a correlation estimate (all-NaN, fewer than two
  observations, zero variance) are dropped rather than imputed: a constant
  series has no correlation with anything, and inventing one would be a lie.
* Degenerate input (empty, one asset, one row, all-NaN) returns empty
  structures and NaN - never an exception.
* Thresholds that ``src.config`` does not carry are declared as named module
  constants below rather than being buried in the code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from src.config import INSIGHTS, MIN_OBSERVATIONS, MIN_ROLLING_WINDOW

__all__ = [
    "CLUSTER_DISTANCE_THRESHOLD",
    "CUMULATIVE_TARGET",
    "DEFAULT_LINKAGE",
    "HIGH_CORRELATION",
    "LINKAGE_METHODS",
    "average_correlation",
    "cluster_assets",
    "clustered_order",
    "correlation_distance",
    "pca_decomposition",
    "rolling_average_correlation",
]

# ---------------------------------------------------------------------------
# module thresholds (config carries no factor-analysis constants)
# ---------------------------------------------------------------------------
#: Cophenetic distance at which the dendrogram is cut when the caller does not
#: ask for a specific number of clusters. d = 0.5 <=> rho = 0.5, i.e. names
#: that share at least half their correlation are treated as one bet.
CLUSTER_DISTANCE_THRESHOLD: float = 0.5

#: Linkage rules accepted by :func:`cluster_assets`. "average" (UPGMA) is the
#: default because it is stable on correlation distances and, unlike "ward",
#: adds no extra Euclidean-variance assumption.
LINKAGE_METHODS: frozenset[str] = frozenset(
    {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
)
DEFAULT_LINKAGE: str = "average"

#: Share of variance used for the "how many factors do I really need" reading.
CUMULATIVE_TARGET: float = 0.90

#: Qualitative bands for the PC1 share, used only to word the interpretation.
PC1_DOMINANT: float = 0.70
PC1_MEANINGFUL: float = 0.50
PC1_BALANCED: float = 0.30

#: Correlation above which a book is called "highly correlated" (from config,
#: so the wording here matches the insight engine's threshold exactly).
HIGH_CORRELATION: float = INSIGHTS.high_correlation

#: Eigenvalues below this are numerical noise around zero and are floored.
EIGENVALUE_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# input hygiene
# ---------------------------------------------------------------------------
def _empty_frame() -> pd.DataFrame:
    """A typed empty frame - the "no answer" return value of this module."""
    return pd.DataFrame(dtype=float)


def _prepare_returns(asset_returns: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    """Numeric, de-duplicated return columns that can support a correlation.

    Coerces to float (non-numeric -> NaN), voids infinities, and drops
    duplicate columns, all-NaN columns, columns with fewer than two
    observations and zero-variance columns. The caller's object is copied
    first and is never modified.
    """
    if asset_returns is None:
        return _empty_frame()
    try:
        df = pd.DataFrame(asset_returns).copy()
    except (TypeError, ValueError):
        return _empty_frame()
    if df.empty or df.shape[1] == 0:
        return _empty_frame()
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    except (TypeError, ValueError):
        return _empty_frame()
    df = df.where(np.isfinite(df))

    keep: list[Any] = []
    for col in df.columns:
        series = df[col]
        if int(series.notna().sum()) < 2:
            continue
        sd = float(series.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            continue
        keep.append(col)
    return df.loc[:, keep]


def _prepare_corr(corr: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a correlation matrix to a clean, square, symmetric frame.

    Keeps the labels present in BOTH the index and the columns (in index
    order), drops duplicates, coerces to float, clips to [-1, 1], mirrors
    one-sided values across the diagonal, averages any residual asymmetry and
    forces a unit diagonal. Unusable input yields an empty frame. The input is
    copied, never modified.
    """
    if corr is None:
        return _empty_frame()
    try:
        c = pd.DataFrame(corr).copy()
    except (TypeError, ValueError):
        return _empty_frame()
    if c.empty or c.shape[1] == 0:
        return _empty_frame()

    c = c.loc[~c.index.duplicated(), ~c.columns.duplicated()]
    column_set = set(c.columns)
    labels = [label for label in c.index if label in column_set]
    if len(labels) == 0:
        return _empty_frame()
    try:
        c = c.loc[labels, labels].apply(pd.to_numeric, errors="coerce")
        values = c.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return _empty_frame()

    # Mirror a value across the diagonal when only one side was supplied,
    # clamp out-of-range entries BEFORE averaging (so one bad cell cannot drag
    # its mirror image), then average away any asymmetry that survives.
    values = np.where(np.isnan(values), values.T, values)
    values = np.clip(values, -1.0, 1.0)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=labels, columns=labels)


def _drop_nan_labels(corr: pd.DataFrame) -> pd.DataFrame:
    """Drop assets, worst offender first, until no NaN correlation remains.

    One asset with no overlapping history would otherwise poison the whole
    matrix; removing it keeps the decomposition available for the rest.
    """
    c = corr
    while c.shape[0] > 0:
        missing = np.isnan(c.to_numpy(dtype=float))
        if not missing.any():
            return c
        worst = int(np.argmax(missing.sum(axis=1)))
        keep = [label for i, label in enumerate(c.columns) if i != worst]
        if not keep:
            return _empty_frame()
        c = c.loc[keep, keep]
    return _empty_frame()


# ---------------------------------------------------------------------------
# correlation summaries
# ---------------------------------------------------------------------------
def average_correlation(corr: pd.DataFrame | None) -> float:
    """Mean off-diagonal correlation: mean(rho_ij) over all pairs i < j.

    The one number that says how close a book is to being a single bet: 0.0
    means the names move independently, 1.0 means they are the same position.
    Returns NaN for fewer than two assets, or when no pair has a finite
    correlation.
    """
    c = _prepare_corr(corr)
    n = int(c.shape[0])
    if n < 2:
        return float("nan")
    values = c.to_numpy(dtype=float)
    upper = values[np.triu_indices(n, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return float("nan")
    return float(np.mean(upper))


def correlation_distance(corr: pd.DataFrame | None) -> pd.DataFrame:
    """Correlation distance d_ij = sqrt(0.5 * (1 - rho_ij)), bounded [0, 1].

    A proper metric (the Euclidean distance between standardised return
    vectors, up to a constant factor): 0.0 at rho = +1, 0.5 at rho = +0.5,
    0.707 at rho = 0 and 1.0 at rho = -1. The diagonal is exactly zero.
    Returns an empty frame for unusable input; NaN correlations stay NaN.
    """
    c = _prepare_corr(corr)
    if c.empty:
        return _empty_frame()
    values = c.to_numpy(dtype=float)
    distance = np.sqrt(np.clip(0.5 * (1.0 - values), 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=c.index, columns=c.columns)


# ---------------------------------------------------------------------------
# principal component analysis on the correlation matrix
# ---------------------------------------------------------------------------
def _empty_pca(assets: list[Any], n_obs: int, reason: str) -> dict[str, Any]:
    """The degenerate PCA result: empty spectrum, NaN scalars, honest reason."""
    return {
        "explained_variance_ratio": np.zeros(0, dtype=float),
        "cumulative": np.zeros(0, dtype=float),
        "eigenvalues": np.zeros(0, dtype=float),
        "loadings": pd.DataFrame(index=pd.Index(assets, name="ticker"), dtype=float),
        "n_components": 0,
        "interpretation": reason,
        "assets": list(assets),
        "n_assets": len(assets),
        "n_obs": int(n_obs),
        "pc1_ratio": float("nan"),
        "effective_bets": float("nan"),
        "components_for_target": 0,
        "cumulative_target": CUMULATIVE_TARGET,
        "average_correlation": float("nan"),
        "ok": False,
        "reason": reason,
    }


def _orient(loadings: np.ndarray) -> np.ndarray:
    """Fix each component's arbitrary sign so its largest loading is positive.

    Eigenvectors are only defined up to sign; pinning the dominant loading
    positive makes PC1 read as "the market" rather than "minus the market",
    and makes the output reproducible across LAPACK builds.
    """
    oriented = np.array(loadings, dtype=float, copy=True)
    for j in range(oriented.shape[1]):
        column = oriented[:, j]
        if column[int(np.argmax(np.abs(column)))] < 0.0:
            oriented[:, j] = -column
    return oriented


def _pca_interpretation(
    ratios: np.ndarray,
    assets: list[Any],
    n_obs: int,
    effective_bets: float,
    components_for_target: int,
    pc1_loadings: pd.Series,
) -> str:
    """Plain-English reading of the spectrum - the headline the UI shows."""
    n = len(assets)
    pc1 = float(ratios[0])
    if pc1 >= PC1_DOMINANT:
        verdict = f"these {n} holdings are effectively one market bet"
    elif pc1 >= PC1_MEANINGFUL:
        verdict = "one common factor drives most of the movement, with a second theme underneath it"
    elif pc1 >= PC1_BALANCED:
        verdict = "a common factor is present but does not dominate"
    else:
        verdict = "no single factor dominates - the exposure is genuinely broad"

    verb = "covers" if components_for_target == 1 else "cover"
    parts = [
        f"PC1 explains {pc1 * 100:.0f}% of variance: {verdict}.",
        f"{components_for_target} of {n} components {verb} "
        f"{CUMULATIVE_TARGET * 100:.0f}% of it, worth about "
        f"{effective_bets:.1f} independent bets across {n} holdings.",
    ]
    top = pc1_loadings.abs().sort_values(ascending=False).head(3)
    if len(top) > 0:
        detail = ", ".join(f"{label} {pc1_loadings[label]:+.2f}" for label in top.index)
        parts.append(f"Heaviest PC1 loadings: {detail}.")
    parts.append(f"Estimated from {n_obs} overlapping daily observations.")
    if n_obs < MIN_OBSERVATIONS:
        parts.append(
            f"That is below the {MIN_OBSERVATIONS}-observation minimum, so treat "
            "the structure as indicative only."
        )
    return " ".join(parts)


def pca_decomposition(
    asset_returns: pd.DataFrame | None, n_components: int | None = None
) -> dict[str, Any]:
    """Principal components of the asset CORRELATION matrix.

    Standardising the assets (correlation rather than covariance) strips out
    the volatility scale, so the spectrum describes co-movement alone:

        C = V L V'                    (symmetric eigendecomposition, eigh)
        explained_variance_ratio_i  = lambda_i / sum_j(lambda_j)
        loadings                    = V * sqrt(L)
        effective bets              = 1 / sum_i(ratio_i^2)

    Because trace(C) = k, the eigenvalues sum to k, the ratios sum to 1 by
    construction and are returned in descending order. ``loadings.iloc[i, j]``
    is the correlation of asset i with component j, so each asset's squared
    loadings sum to 1 across the FULL spectrum. Correlations are estimated on
    complete cases, which keeps C positive semi-definite; eigenvalues within
    ``EIGENVALUE_FLOOR`` of zero are floored to exactly 0.

    Parameters
    ----------
    asset_returns:
        Daily (or any fixed-frequency) returns, one column per asset.
    n_components:
        How many components to return LOADINGS for; ``None`` means all.
        ``explained_variance_ratio``, ``cumulative`` and ``eigenvalues``
        always describe the full spectrum, so the ratios still sum to 1.

    Returns
    -------
    dict with the contract keys ``explained_variance_ratio``, ``cumulative``,
    ``loadings``, ``eigenvalues``, ``n_components`` and ``interpretation``,
    plus ``assets``, ``n_assets``, ``n_obs``, ``pc1_ratio``, ``effective_bets``,
    ``components_for_target``, ``cumulative_target``, ``average_correlation``,
    ``ok`` and ``reason``. Fewer than two usable assets, or fewer than two
    overlapping dates, returns the empty shape with ``ok=False`` and a reason
    (never an exception).
    """
    returns = _prepare_returns(asset_returns)
    assets = list(returns.columns)
    if len(assets) < 2:
        return _empty_pca(
            assets,
            int(returns.shape[0]),
            "Not enough usable assets for a factor decomposition: PCA needs at "
            f"least two assets with non-zero variance, found {len(assets)}.",
        )

    complete = returns.dropna(how="any")
    n_obs = int(complete.shape[0])
    if n_obs < 2:
        return _empty_pca(
            assets,
            n_obs,
            f"Only {n_obs} date(s) on which every asset traded: a correlation "
            "matrix needs at least two overlapping observations.",
        )

    corr = _drop_nan_labels(_prepare_corr(complete.corr()))
    assets = list(corr.columns)
    if len(assets) < 2:
        return _empty_pca(
            assets,
            n_obs,
            "Correlations could not be estimated for at least two assets on the overlapping dates.",
        )

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return _empty_pca(assets, n_obs, "Eigendecomposition of the correlation matrix failed.")

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.where(eigenvalues[order] > EIGENVALUE_FLOOR, eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _empty_pca(assets, n_obs, "The correlation matrix carries no variance to decompose.")

    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    n_assets = len(assets)

    loadings_all = _orient(eigenvectors * np.sqrt(eigenvalues))
    names = [f"PC{i + 1}" for i in range(n_assets)]

    keep = n_assets
    if n_components is not None:
        try:
            keep = int(n_components)
        except (TypeError, ValueError):
            keep = n_assets
        keep = int(min(max(keep, 1), n_assets))
    loadings = pd.DataFrame(
        loadings_all[:, :keep],
        index=pd.Index(assets, name="ticker"),
        columns=names[:keep],
    )

    squared = float(np.sum(ratios**2))
    effective_bets = float(1.0 / squared) if squared > 0.0 else float("nan")
    components_for_target = int(
        min(int(np.searchsorted(cumulative, CUMULATIVE_TARGET)) + 1, n_assets)
    )
    pc1_loadings = pd.Series(loadings_all[:, 0], index=pd.Index(assets, name="ticker"), dtype=float)

    return {
        "explained_variance_ratio": ratios,
        "cumulative": cumulative,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "n_components": int(keep),
        "interpretation": _pca_interpretation(
            ratios, assets, n_obs, effective_bets, components_for_target, pc1_loadings
        ),
        "assets": assets,
        "n_assets": n_assets,
        "n_obs": n_obs,
        "pc1_ratio": float(ratios[0]),
        "effective_bets": effective_bets,
        "components_for_target": components_for_target,
        "cumulative_target": CUMULATIVE_TARGET,
        "average_correlation": average_correlation(corr),
        "ok": True,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# hierarchical clustering by correlation distance
# ---------------------------------------------------------------------------
def _empty_clusters(assets: list[Any], method: str) -> dict[str, Any]:
    """Degenerate clustering result: one trivial cluster, or nothing at all."""
    if len(assets) == 1:
        label = assets[0]
        return {
            "labels": {label: 1},
            "order": [label],
            "linkage": np.zeros((0, 4), dtype=float),
            "n_clusters": 1,
            "clusters": {1: [label]},
            "method": method,
            "threshold": float("nan"),
            "n_assets": 1,
            "interpretation": (
                "A single holding is a single bet - clustering needs at least "
                "two assets to say anything."
            ),
        }
    return {
        "labels": {},
        "order": [],
        "linkage": np.zeros((0, 4), dtype=float),
        "n_clusters": 0,
        "clusters": {},
        "method": method,
        "threshold": float("nan"),
        "n_assets": 0,
        "interpretation": "No usable correlation matrix to cluster.",
    }


def _cluster_interpretation(
    clusters: dict[int, list[Any]],
    corr: pd.DataFrame,
    threshold: float,
    by_distance: bool,
) -> str:
    """Plain-English reading of the cluster structure."""
    n_assets = int(corr.shape[0])
    n_clusters = len(clusters)
    plural = "" if n_clusters == 1 else "s"
    if by_distance:
        rho_cut = 1.0 - 2.0 * threshold**2
        head = (
            f"{n_assets} holdings form {n_clusters} correlation cluster{plural} "
            f"at a distance cut of {threshold:.2f} (rho = {rho_cut:.2f})."
        )
    else:
        head = f"{n_assets} holdings split into the {n_clusters} cluster{plural} requested."
    parts = [head]

    if n_clusters == n_assets:
        parts.append("Every name sits in its own cluster - no redundancy at this cut.")
    else:
        largest = max(clusters.values(), key=len)
        shown = ", ".join(str(label) for label in largest[:4])
        if len(largest) > 4:
            shown += ", ..."
        parts.append(
            f"The largest cluster holds {len(largest)} of them ({shown}) - those "
            "move together and count as roughly one bet."
        )

    avg = average_correlation(corr)
    if np.isfinite(avg):
        tail = (
            f" - above the {HIGH_CORRELATION:.2f} high-correlation threshold, so "
            "the diversification here is mostly cosmetic."
            if avg >= HIGH_CORRELATION
            else "."
        )
        parts.append(f"Average pairwise correlation {avg:.2f}{tail}")
    return " ".join(parts)


def cluster_assets(
    corr: pd.DataFrame | None,
    n_clusters: int | None = None,
    method: str = DEFAULT_LINKAGE,
) -> dict[str, Any]:
    """Group assets by correlation distance with hierarchical clustering.

    ``scipy.cluster.hierarchy.linkage`` (default average linkage / UPGMA) runs
    on the condensed form of ``d_ij = sqrt(0.5 * (1 - rho_ij))``. The tree is
    then cut either into ``n_clusters`` groups (``criterion="maxclust"``) or,
    when ``n_clusters is None``, at the ``CLUSTER_DISTANCE_THRESHOLD``
    cophenetic distance of 0.5 - which is exactly rho = 0.5, so names sharing
    at least half their correlation land in the same bucket.

    ``order`` comes from ``optimal_leaf_ordering``, so drawing the correlation
    heatmap in that order puts co-moving blocks on the diagonal. Leaf
    reordering never changes cluster membership. An unknown ``method`` falls
    back to ``DEFAULT_LINKAGE`` rather than raising.

    Returns
    -------
    dict with ``labels`` (ticker -> 1-based cluster id), ``order``,
    ``linkage`` (the (n-1) x 4 matrix), ``n_clusters`` and ``clusters``
    (cluster id -> tickers in leaf order), plus ``method``, ``threshold``,
    ``n_assets`` and ``interpretation``. Fewer than two usable assets degrades
    to a single trivial cluster (or empties) rather than raising.
    """
    used_method = method if method in LINKAGE_METHODS else DEFAULT_LINKAGE
    clean = _drop_nan_labels(_prepare_corr(corr))
    assets = list(clean.columns)
    if len(assets) < 2:
        return _empty_clusters(assets, used_method)

    distance = correlation_distance(clean).to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if condensed.size == 0 or not bool(np.isfinite(condensed).all()):
        return _empty_clusters(assets, used_method)  # pragma: no cover - defensive

    try:
        matrix = linkage(condensed, method=used_method)
        matrix = optimal_leaf_ordering(matrix, condensed)
    except (ValueError, MemoryError):  # pragma: no cover - defensive
        return _empty_clusters(assets, used_method)

    by_distance = n_clusters is None
    if by_distance:
        threshold = CLUSTER_DISTANCE_THRESHOLD
        flat = fcluster(matrix, t=threshold, criterion="distance")
    else:
        try:
            wanted = int(n_clusters)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            wanted = len(assets)
        wanted = int(min(max(wanted, 1), len(assets)))
        threshold = float(wanted)
        flat = fcluster(matrix, t=wanted, criterion="maxclust")

    order = [assets[i] for i in leaves_list(matrix)]
    labels = {asset: int(flat[i]) for i, asset in enumerate(assets)}

    clusters: dict[int, list[Any]] = {}
    for asset in order:  # leaf order keeps cluster members adjacent
        clusters.setdefault(labels[asset], []).append(asset)
    clusters = {key: clusters[key] for key in sorted(clusters)}

    return {
        "labels": labels,
        "order": order,
        "linkage": matrix,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "method": used_method,
        "threshold": threshold,
        "n_assets": len(assets),
        "interpretation": _cluster_interpretation(clusters, clean, threshold, by_distance),
    }


def clustered_order(corr: pd.DataFrame | None) -> list[str]:
    """Asset order that groups correlated names together on a heatmap.

    Thin wrapper over :func:`cluster_assets` returning only the leaf order of
    the average-linkage dendrogram (optimal-leaf-ordered). Returns ``[]`` for
    an unusable matrix and ``[asset]`` for a single asset, so a caller can
    always do ``corr.loc[order, order]`` when the list is non-empty.
    """
    return [str(label) for label in cluster_assets(corr)["order"]]


# ---------------------------------------------------------------------------
# rolling co-movement
# ---------------------------------------------------------------------------
def rolling_average_correlation(asset_returns: pd.DataFrame | None, window: int) -> pd.Series:
    """Rolling mean of all pairwise correlations - diversification over time.

    For every window of ``window`` observations the full pairwise Pearson
    correlation matrix is formed from rolling sums,

        cov_ij = (S_xy - S_x * S_y / w) / (w - 1)
        rho_ij = cov_ij / sqrt(var_i * var_j)

    and the mean of its upper triangle (i < j) is reported - the same quantity
    :func:`average_correlation` returns for a static matrix. Rising values mean
    diversification is evaporating exactly when it is needed: correlations go
    to one in a crisis.

    Rows where any asset is missing are dropped first, so every window spans
    the same dates. The first ``window - 1`` points are NaN by construction,
    and a window in which an asset is constant yields NaN for that asset's
    pairs only (the remaining pairs still average).

    Returns an EMPTY series (never an exception) when there are fewer than two
    usable assets, when ``window < 2``, or when the sample is shorter than
    ``max(window, MIN_ROLLING_WINDOW)``.
    """
    empty = pd.Series(dtype=float, name="avg_correlation")
    try:
        w = int(window)
    except (TypeError, ValueError):
        return empty
    if w < 2:
        return empty

    frame = _prepare_returns(asset_returns).dropna(how="any")
    n_rows, n_assets = frame.shape
    if n_assets < 2 or n_rows < max(w, MIN_ROLLING_WINDOW):
        return empty

    values = frame.to_numpy(dtype=float)
    left, right = np.triu_indices(n_assets, k=1)
    products = pd.DataFrame(values[:, left] * values[:, right], index=frame.index)

    sums = frame.rolling(w, min_periods=w).sum().to_numpy(dtype=float)
    squares = (frame**2).rolling(w, min_periods=w).sum().to_numpy(dtype=float)
    cross = products.rolling(w, min_periods=w).sum().to_numpy(dtype=float)

    variance = (squares - sums**2 / w) / (w - 1)
    variance = np.where(variance > 0.0, variance, np.nan)
    covariance = (cross - sums[:, left] * sums[:, right] / w) / (w - 1)
    correlations = np.clip(covariance / np.sqrt(variance[:, left] * variance[:, right]), -1.0, 1.0)

    finite = np.isfinite(correlations)
    counts = finite.sum(axis=1)
    totals = np.where(finite, correlations, 0.0).sum(axis=1)
    averages = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return pd.Series(averages, index=frame.index, name="avg_correlation")
