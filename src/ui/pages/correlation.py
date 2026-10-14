"""Correlation & factors -- how many independent bets this book really owns.

A weight table says what is *held*; it cannot say what is *owned*. Ten large-cap
Indian banks are one position wearing ten ticker symbols, and no amount of
counting rows will ever reveal that. This page answers the question the blotter
cannot: once co-movement is taken into account, how many genuinely separate
exposures are in the portfolio, which names are near-substitutes for one
another, and how much of the daily movement is a single shared factor.

Everything is read off the completed :class:`~src.models.AnalysisResult` --
``correlation`` for the matrix, ``attribution["factors"]`` for the PCA and the
clustering, ``summary`` for the capital-based effective-bets count, ``rolling``
for the through-time view. ``attribution["factors"]`` is legitimately an empty
dict on a degenerate run, so every access into it is guarded and each dependent
block degrades to an honest empty state rather than a traceback.

The only analytics call is :func:`src.analytics.insights._high_correlation_pairs`
with a threshold of -1.0, which is a pure re-listing of the matrix the pipeline
already produced (its own docstring documents that call as this page's use of
it). Factor thresholds are imported from :mod:`src.analytics.factors` rather
than restated, so the page and the engine cannot disagree about where a cut sits.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import insights as ins
from src.analytics.factors import (
    CLUSTER_DISTANCE_THRESHOLD,
    CUMULATIVE_TARGET,
    PC1_BALANCED,
    PC1_DOMINANT,
    PC1_MEANINGFUL,
)
from src.config import INSIGHTS, MIN_OBSERVATIONS, TRADING_DAYS
from src.ui import components as ui
from src.ui import factor_panel, guards
from src.ui import theme as ui_theme
from src.ui.formatters import DASH, fmt_num, fmt_pct, fmt_ratio_pct

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
#: Correlation bands used ONLY to word the verdict column of the pairwise
#: table. The top band is ``config.INSIGHTS.high_correlation`` so the wording
#: matches the threshold that fires the diversification insights; the two below
#: it are declared here because config carries no equivalent field.
_SAME_THEME: float = 0.60
_LOOSELY_LINKED: float = 0.30
_OFFSETTING: float = -0.10

#: Badge tones cycled across clusters so adjacent groups stay visually distinct.
#: Colour here is an index, not a judgement -- cluster 2 is not "worse" than 1.
_CLUSTER_TONES: tuple[str, ...] = ("accent", "gain", "warn", "neutral", "muted")

#: Rows the pairwise table shows before it starts scrolling inside itself.
_MAX_TABLE_ROWS: int = 14

#: Height shared by the scree plot and the loadings heatmap so they line up.
_PCA_PANEL_HEIGHT: int = 400


# ---------------------------------------------------------------------------
# small display helpers (no analytics happen on this page)
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    """True when ``x`` is a real, finite number worth printing."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def _int_or_none(x: Any) -> int | None:
    """Coerce to int, or None when the value is missing or not a number."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _floats(values: Any) -> list[float]:
    """Coerce a numpy vector / sequence to a plain list of floats.

    Written out rather than using ``values or []`` because the PCA block hands
    back numpy arrays, whose truth value is ambiguous for more than one element.
    """
    if values is None:
        return []
    try:
        return [float(v) for v in list(values)]
    except (TypeError, ValueError):
        return []


def _factor_block(result: Any) -> dict:
    """``attribution["factors"]``, always a dict -- it may legitimately be empty."""
    block = (result.attribution or {}).get("factors")
    return block if isinstance(block, dict) else {}


def _sub_block(factors: dict, key: str) -> dict:
    """One sub-block of the factor dict, always a dict."""
    block = factors.get(key)
    return block if isinstance(block, dict) else {}


def _labels(corr: Any) -> list[str]:
    """Column labels of the correlation matrix, as strings."""
    if not isinstance(corr, pd.DataFrame) or corr.empty:
        return []
    return [str(c) for c in corr.columns]


def _pairs(corr: Any) -> list[tuple[str, str, float]]:
    """Every finite ``(a, b, rho)`` pair, strongest first.

    ``threshold=-1.0`` cannot bind, because Pearson correlation is bounded
    below by -1 -- so this returns the whole upper triangle, sorted.
    """
    found = ins._high_correlation_pairs(corr, -1.0)
    return [(str(a), str(b), float(rho)) for a, b, rho in found if _finite(rho)]


def _average_correlation(factors: dict, pairs: list[tuple[str, str, float]]) -> tuple[float, str]:
    """The mean off-diagonal correlation, plus where the number came from.

    Prefers the pipeline's own ``average_correlation``. When the factor block
    is empty the identical quantity is averaged off the pairwise list, and the
    provenance string says so rather than presenting it as the engine's figure.
    """
    value = factors.get("average_correlation")
    if _finite(value):
        return float(value), "factor decomposition"
    if pairs:
        return sum(rho for _, _, rho in pairs) / len(pairs), "pairwise list"
    return float("nan"), "unavailable"


def _cut_rho(clusters: dict) -> float:
    """Correlation at which the dendrogram was cut: ``rho = 1 - 2 * d^2``.

    Returns NaN when the tree was cut into a fixed number of clusters instead,
    because then ``threshold`` carries a cluster count, not a distance.
    """
    threshold = clusters.get("threshold")
    if not _finite(threshold):
        return float("nan")
    distance = float(threshold)
    if distance > 1.0:
        return float("nan")
    return 1.0 - 2.0 * distance**2


def _verdict(rho: float) -> str:
    """Plain-English band for one pairwise correlation."""
    if rho >= INSIGHTS.high_correlation:
        return "Near-substitutes"
    if rho >= _SAME_THEME:
        return "Same theme"
    if rho >= _LOOSELY_LINKED:
        return "Loosely linked"
    if rho > _OFFSETTING:
        return "Largely independent"
    return "Offsetting"


def _tightest_within(
    members: list[Any], pairs: list[tuple[str, str, float]]
) -> tuple[str, str, float] | None:
    """The most-correlated pair whose both legs sit inside one cluster."""
    inside = {str(m) for m in members}
    for a, b, rho in pairs:  # already sorted strongest first
        if a in inside and b in inside:
            return (a, b, rho)
    return None


def _corr_tone(avg: Any) -> str:
    if not _finite(avg):
        return "muted"
    value = float(avg)
    if value >= INSIGHTS.high_correlation:
        return "loss"
    if value >= _SAME_THEME:
        return "warn"
    return "gain" if value < _LOOSELY_LINKED else "neutral"


def _pc1_tone(ratio: Any) -> str:
    if not _finite(ratio):
        return "muted"
    value = float(ratio)
    if value >= PC1_DOMINANT:
        return "loss"
    if value >= PC1_MEANINGFUL:
        return "warn"
    return "gain" if value < PC1_BALANCED else "neutral"


def _bets_tone(effective: Any, n_holdings: int) -> str:
    if not (_finite(effective) and n_holdings > 0):
        return "muted"
    ratio = float(effective) / n_holdings
    return "warn" if ratio < INSIGHTS.effective_holdings_ratio else "neutral"


# ---------------------------------------------------------------------------
# 1. headline KPIs
# ---------------------------------------------------------------------------
def _kpis(
    result: Any,
    factors: dict,
    pairs: list[tuple[str, str, float]],
    avg: float,
    avg_source: str,
    n_assets: int,
) -> None:
    pca = _sub_block(factors, "pca")
    clusters = _sub_block(factors, "clusters")
    summary = result.summary or {}

    n_holdings = int(summary.get("n_holdings", 0) or 0)
    n_clusters = _int_or_none(clusters.get("n_clusters"))
    pc1 = pca.get("pc1_ratio")
    pca_bets = pca.get("effective_bets")
    capital_bets = summary.get("effective_holdings")
    cut = _cut_rho(clusters)
    needed = _int_or_none(pca.get("components_for_target"))

    corr_card = {
        "label": "Average pairwise correlation",
        "value": fmt_num(avg, 2),
        "tone": _corr_tone(avg),
        "footnote": (
            f"{len(pairs):,} pairs across {n_assets} priced holdings"
            if pairs
            else "No finite pair could be measured"
        ),
        "help_text": (
            "Mean of the off-diagonal Pearson correlations of daily returns "
            f"(source: {avg_source}). Dimensionless, bounded -1 to +1."
        ),
    }

    cluster_card = {
        "label": "Correlation clusters",
        "value": str(n_clusters) if n_clusters is not None else DASH,
        "tone": (
            "warn"
            if n_clusters is not None and n_assets > 0 and n_clusters < max(2, n_assets // 2)
            else "neutral"
        ),
        "footnote": (
            f"Average linkage, cut where rho = {fmt_num(cut, 2)}"
            if _finite(cut)
            else "Hierarchical clustering unavailable on this run"
        ),
        "help_text": (
            "Groups produced by cutting the correlation-distance dendrogram. "
            "Names in one group are treated as a single bet."
        ),
    }

    pc1_card = {
        "label": "PC1 explained variance",
        "value": fmt_ratio_pct(pc1, 1),
        "tone": _pc1_tone(pc1),
        "footnote": (
            f"{needed} of {n_assets} components carry "
            f"{fmt_ratio_pct(CUMULATIVE_TARGET, 0)} of the variance"
            if needed is not None
            else "Principal-component decomposition unavailable"
        ),
        "help_text": (
            "Share of total return variance explained by the first principal "
            "component of the correlation matrix -- in equities, almost always "
            "the market itself."
        ),
    }

    bets_card = {
        "label": "Effective number of bets",
        "value": fmt_num(capital_bets, 1),
        "tone": _bets_tone(capital_bets, n_holdings),
        "footnote": (
            f"1 / HHI on today's weights, against {n_holdings} names held"
            if n_holdings
            else "1 / HHI on today's weights"
        ),
        "help_text": (
            "Equally-weighted positions that would carry this much concentration. "
            "It counts capital, not risk -- the PCA number below counts risk."
        ),
    }
    if _finite(pca_bets):
        bets_card["delta"] = f"PCA counts {fmt_num(pca_bets, 1)} independent directions"
        bets_card["delta_tone"] = (
            "loss"
            if _finite(capital_bets) and float(pca_bets) < float(capital_bets) * 0.6
            else "muted"
        )

    ui.kpi_row([corr_card, cluster_card, pc1_card, bets_card], columns=4)

    _kpi_caption(avg, n_assets, capital_bets, pca_bets)
    _kpi_methodology(result, avg, avg_source, n_assets, pairs, clusters, cut)


def _kpi_caption(avg: float, n_assets: int, capital_bets: Any, pca_bets: Any) -> None:
    """The one sentence this whole page exists to make concrete."""
    if not _finite(avg):
        st.caption(
            "No average pairwise correlation could be measured, so nothing on this "
            "page can say whether the holdings are separate bets. That happens when "
            "the names share too few trading dates for a correlation to be estimated."
        )
        return

    floor = math.sqrt(avg) if avg > 0 else float("nan")
    if _finite(floor):
        limit = (
            f"Push the holding count to infinity at this correlation and portfolio "
            f"volatility still stops falling at about {fmt_ratio_pct(floor, 0)} of the "
            f"average single-name volatility (sqrt of the average correlation, for "
            f"equally weighted names of similar volatility). That floor -- not the "
            f"number of tickers -- is what average correlation buys or denies you."
        )
    else:
        limit = (
            "At or below zero average correlation there is no volatility floor from "
            "co-movement at all: every additional name keeps reducing portfolio "
            "volatility, which is as good as diversification gets."
        )

    st.caption(
        f"Average pairwise correlation **{fmt_num(avg, 2)}** across {n_assets} "
        f"holdings. This is the number that decides whether adding the eleventh "
        f"stock diversifies anything at all. {limit} By contrast the capital-based "
        f"effective-bets count ({fmt_num(capital_bets, 1)}) knows nothing about "
        f"correlation -- it only sees weights. The PCA count of independent "
        f"directions ({fmt_num(pca_bets, 1)}) is the risk-side answer to the same "
        f"question, and it is the one that falls when the names start moving together."
    )


def _kpi_methodology(
    result: Any,
    avg: float,
    avg_source: str,
    n_assets: int,
    pairs: list[tuple[str, str, float]],
    clusters: dict,
    cut: float,
) -> None:
    method = str(clusters.get("method") or "average")
    ui.methodology(
        "What each headline number is, and what it is not",
        f"""
**Average pairwise correlation** = `mean(rho_ij)` over the {len(pairs):,}
distinct pairs `i < j` of the correlation matrix -- the diagonal is excluded,
because an asset's correlation with itself is 1.00 by construction and would
flatter every book equally. This run reports **{fmt_num(avg, 2)}**, taken from
the {avg_source}. The high-correlation line the insight engine uses is
{fmt_num(INSIGHTS.high_correlation, 2)} (`config.INSIGHTS.high_correlation`).

**Why it is the number that matters.** For N equally weighted names of similar
volatility `sigma`, portfolio variance is
`sigma^2 * (1/N + (1 - 1/N) * rho)`. The `1/N` term -- the idiosyncratic part --
is the only piece extra names can remove; the `rho` term is a floor no holding
count can cross. That is why a twentieth stock at rho = 0.8 changes almost
nothing, and a second stock at rho = 0.1 changes a great deal.

**Correlation clusters** come from {method}-linkage hierarchical clustering on
the correlation distance `d_ij = sqrt(0.5 * (1 - rho_ij))`, a true metric:
d = 0.00 at rho = +1, 0.50 at rho = +0.5, 0.71 at rho = 0 and 1.00 at rho = -1.
The tree is cut at a cophenetic distance of {CLUSTER_DISTANCE_THRESHOLD:.2f},
which is exactly rho = {fmt_num(cut, 2)} -- names sharing at least that much
correlation land in one bucket.

**PC1 explained variance** is `lambda_1 / sum(lambda_j)` from the
eigendecomposition of the correlation matrix. Because a k-asset correlation
matrix has trace k, the eigenvalues sum to k and the ratios sum to 1 exactly.

**Effective number of bets** = `1 / sum(w_i^2)` on today's market-value weights.
It is a concentration statistic and nothing more: five equally weighted PSU
banks score 5.0 here and remain one bet. Every number to its left on this page
exists precisely because that card cannot see correlation.

**Window.** All four are measured on the {result.n_days:,} sessions in this
history window, on overlapping dates only, and describe that window alone.
        """,
    )


# ---------------------------------------------------------------------------
# 2. the matrix
# ---------------------------------------------------------------------------
def _matrix(result: Any, clusters: dict, avg: float, n_assets: int) -> None:
    ui_theme.section("CORRELATION MATRIX", "\U0001f9ee")

    order = clusters.get("order")
    ordered = [str(label) for label in order] if order else None
    ui.chart(ch.correlation_heatmap(result.correlation, ordered))

    if ordered:
        arrangement = (
            "Rows and columns are in dendrogram leaf order, so co-moving names sit "
            "next to each other and blocks of related holdings appear on the "
            "diagonal. Reordering moves labels only -- not one correlation changes."
        )
    else:
        arrangement = (
            "Holdings are in their original order: the clustering that would group "
            "related names together was not available on this run."
        )

    st.caption(
        f"{arrangement} The colour scale is hard-pinned to -1.00 .. +1.00, so the "
        f"same shade always means the same number across portfolios and across "
        f"screenshots -- an auto-ranged scale would make a well-diversified book "
        f"look identical to a perfectly correlated one. The diagonal is 1.00 by "
        f"definition and carries no information; the {n_assets * (n_assets - 1) // 2:,} "
        f"off-diagonal cells average {fmt_num(avg, 2)}."
    )

    ui.methodology(
        "How these correlations are estimated",
        f"""
**Definition.** Pearson correlation of **daily arithmetic returns**,
`r_t = P_t / P_(t-1) - 1`, computed on the {result.n_days:,} sessions of adjusted
closes in this window. Adjusted closes mean splits and dividends do not appear
as spurious one-day moves.

**Overlapping dates only.** Each pair is estimated on the dates *both* names
traded. A recently listed holding, or one that trades on another exchange
calendar, is therefore measured over a shorter sample than the rest of the
matrix -- its row is noisier, not wrong.

**Three honest limitations.**

* *Linear.* Pearson measures straight-line co-movement. Two names that both
  collapse only in a crash, and drift independently otherwise, will show a
  modest correlation that badly understates their joint tail behaviour.
* *Unconditional.* One number is fitted to the whole window, calm days and
  panic days together. Correlations are not stable: the through-time chart at
  the foot of this page is the check on that.
* *In-sample.* With {result.n_days:,} observations and {n_assets} assets there
  are {n_assets * (n_assets - 1) // 2:,} pairs being estimated at once, so some
  of the extreme cells are sampling noise. Treat the block structure as the
  signal and any single cell as an estimate with error bars.

**Not a hedge ratio.** Correlation is scale-free; it says two names move
together, not that they move by the same amount. Sizing that co-movement in
money is what the covariance-based risk contributions on the Attribution page
are for.
        """,
    )


# ---------------------------------------------------------------------------
# 3. every pair
# ---------------------------------------------------------------------------
def _pairwise(pairs: list[tuple[str, str, float]], n_assets: int) -> None:
    ui_theme.section("EVERY PAIR, STRONGEST FIRST", "\U0001f4cb")

    if not pairs:
        ui.empty_state(
            "No pairwise correlations",
            "Not one pair of holdings produced a finite correlation, so there is nothing to rank.",
            icon="\U0001f9ff",
            hint="Two names need overlapping trading dates and non-zero price "
            "movement before a correlation exists.",
        )
        return

    frame = pd.DataFrame(
        {
            "Holding A": [a for a, _, _ in pairs],
            "Holding B": [b for _, b, _ in pairs],
            "Correlation": [rho for _, _, rho in pairs],
            "Shared variance": [rho * rho * 100.0 for _, _, rho in pairs],
            "Reading": [_verdict(rho) for _, _, rho in pairs],
        }
    )

    _pairwise_prose(pairs, n_assets)

    ui.dataframe(
        frame,
        height=min(38 * (min(len(frame), _MAX_TABLE_ROWS) + 1) + 10, 600),
        column_config={
            "Holding A": st.column_config.TextColumn("Holding A", width="small"),
            "Holding B": st.column_config.TextColumn("Holding B", width="small"),
            "Correlation": st.column_config.ProgressColumn(
                "Correlation (rho)",
                format="%.3f",
                min_value=-1.0,
                max_value=1.0,
                help="Pearson correlation of daily returns. The bar spans the full "
                "-1 to +1 range, so a half-filled bar is rho = 0.",
            ),
            "Shared variance": st.column_config.NumberColumn(
                "Shared variance",
                format="%.1f%%",
                help="rho squared: the share of each name's daily variance the other one explains.",
            ),
            "Reading": st.column_config.TextColumn("Reading", width="small"),
        },
    )

    st.caption(
        f"All {len(pairs):,} pairs, strongest first -- the whole upper triangle of "
        f"the matrix, nothing filtered out. Sorting descending is deliberate: the "
        f"pairs that cost you diversification sit at the top, and the pairs that "
        f"earn it sit at the bottom."
    )

    ui.methodology(
        "Reading the pair table",
        f"""
**Shared variance** is `rho^2`, the coefficient of determination: the fraction
of one name's daily variance that the other one accounts for. It is the number
that makes correlation intuitive -- rho = 0.70 sounds like a strong link and
explains {0.70**2 * 100:.0f}% of the variance, while rho = 0.90 explains
{0.90**2 * 100:.0f}%. Note that it is symmetric and sign-blind: rho = -0.80
and rho = +0.80 both explain {0.80**2 * 100:.0f}%, in opposite directions.

**The Reading column** is wording, not a rating, and its cuts are declared in
this module:

| Band | Correlation |
| --- | --- |
| Near-substitutes | rho >= {fmt_num(INSIGHTS.high_correlation, 2)} (`config.INSIGHTS.high_correlation`) |
| Same theme | {fmt_num(_SAME_THEME, 2)} <= rho < {fmt_num(INSIGHTS.high_correlation, 2)} |
| Loosely linked | {fmt_num(_LOOSELY_LINKED, 2)} <= rho < {fmt_num(_SAME_THEME, 2)} |
| Largely independent | {fmt_num(_OFFSETTING, 2)} < rho < {fmt_num(_LOOSELY_LINKED, 2)} |
| Offsetting | rho <= {fmt_num(_OFFSETTING, 2)} |

**What a high pair does and does not imply.** Two near-substitutes are one
exposure held twice, so trimming either releases risk budget without changing
what the portfolio is betting on. It does *not* follow that one of them should
be sold: correlation says nothing about which name is the better business, and
two highly correlated holdings with very different volatilities are not equal
contributors to portfolio risk. The money-weighted version of this question is
answered by the risk-contribution split on the Attribution page.

**Negative pairs are rare and precious.** In a long-only equity book, genuinely
negative correlations are usually a short sample, a currency effect or a
sector-rotation pair rather than a structural hedge. Verify one before relying
on it -- and check whether it survives on the through-time chart below.
        """,
    )


def _pairwise_prose(pairs: list[tuple[str, str, float]], n_assets: int) -> None:
    """Call out the tightest and the most diversifying pairs in words."""
    top_a, top_b, top_rho = pairs[0]
    lines: list[str] = [
        f"The tightest pair in the book is **{top_a}** and **{top_b}** at rho "
        f"{fmt_num(top_rho, 2)}: they share {fmt_pct(top_rho * top_rho * 100.0, 0)} "
        f"of their daily variance, so holding both is closer to holding one "
        f"position at double size than to holding two separate ones."
    ]

    if len(pairs) >= 3:
        runners = ", ".join(f"{a} / {b} at {fmt_num(rho, 2)}" for a, b, rho in pairs[1:3])
        lines.append(f"Behind it: {runners}.")

    n_high = sum(1 for _, _, rho in pairs if rho >= INSIGHTS.high_correlation)
    if n_high:
        lines.append(
            f"{n_high} of {len(pairs):,} pairs sit at or above "
            f"{fmt_num(INSIGHTS.high_correlation, 2)}, the level at which two names "
            f"stop being separate decisions."
        )
    else:
        lines.append(
            f"No pair reaches {fmt_num(INSIGHTS.high_correlation, 2)}, so no two of "
            f"these {n_assets} holdings are outright substitutes for one another."
        )

    if len(pairs) >= 2:
        low_a, low_b, low_rho = pairs[-1]
        if low_rho < 0:
            lines.append(
                f"At the other end, **{low_a}** and **{low_b}** are actually "
                f"offsetting at rho {fmt_num(low_rho, 2)} -- the one place in this "
                f"book where a bad day for one is, on average, not a bad day for "
                f"the other."
            )
        else:
            lines.append(
                f"The most diversifying pair is **{low_a}** and **{low_b}** at rho "
                f"{fmt_num(low_rho, 2)}, and it is still positive -- everything here "
                f"moves the same way, only by different amounts."
            )

    st.caption(" ".join(lines))


# ---------------------------------------------------------------------------
# 4. clustering
# ---------------------------------------------------------------------------
def _clustering(result: Any, clusters: dict, pairs: list[tuple[str, str, float]]) -> None:
    ui_theme.section("CORRELATION CLUSTERS", "\U0001f333")

    membership = clusters.get("clusters")
    if not clusters or not isinstance(membership, dict) or not membership:
        ui.empty_state(
            "Clustering unavailable",
            str(clusters.get("interpretation") or "")
            or "The factor block was not produced on this run, so the holdings "
            "could not be grouped by correlation distance.",
            icon="\U0001f333",
            hint="Hierarchical clustering needs at least two holdings whose "
            "correlations are all defined.",
        )
        return

    labels = _labels(result.correlation)
    left, right = st.columns([1.3, 1.0], gap="large")

    with left:
        ui.chart(ch.dendrogram(clusters.get("linkage"), labels))
        st.caption(
            "The distance axis is the correlation distance at which two branches "
            "join: a merge near 0.00 is a pair of near-perfect substitutes, 0.50 is "
            "rho = 0.50, 0.71 is an uncorrelated pair and 1.00 is perfect inverse "
            "correlation. Read the merge heights, not the shape -- hierarchical "
            "clustering returns a tree even when there is no real structure in the "
            "data to find."
        )

    with right:
        st.markdown("**Cluster membership**")
        for position, (cluster_id, members) in enumerate(sorted(membership.items())):
            names = [str(m) for m in members]
            tone = _CLUSTER_TONES[position % len(_CLUSTER_TONES)]
            st.markdown(
                f"**Cluster {cluster_id}** - {len(names)} name{'s' if len(names) != 1 else ''}"
            )
            ui.badges([(name, tone) for name in names])
            tightest = _tightest_within(names, pairs) if len(names) > 1 else None
            if tightest is not None:
                a, b, rho = tightest
                st.caption(f"Tightest inside: {a} / {b} at rho {fmt_num(rho, 2)}.")

    interpretation = str(clusters.get("interpretation") or "").strip()
    if interpretation:
        st.markdown(f"> {interpretation}")

    _cluster_reading(membership, len(labels))

    ui.methodology(
        "How the groups are formed, and what a group licenses you to do",
        f"""
**The distance.** Correlation is turned into a genuine metric,
`d_ij = sqrt(0.5 * (1 - rho_ij))`, bounded in [0, 1]. It satisfies the triangle
inequality -- ordinary `1 - rho` does not -- which is what makes the resulting
tree meaningful rather than merely decorative.

**The linkage.** `{clusters.get("method", "average")}` linkage (UPGMA by
default): the distance between two groups is the mean distance between their
members. It is used here in preference to Ward linkage, which would import an
extra Euclidean-variance assumption that correlation distance does not satisfy.

**The cut.** The tree is cut at a cophenetic distance of
{CLUSTER_DISTANCE_THRESHOLD:.2f}, i.e. rho = {fmt_num(_cut_rho(clusters), 2)},
which produced {clusters.get("n_clusters", DASH)} cluster(s) from
{clusters.get("n_assets", DASH)} holdings. Leaf ordering is optimised so the
heatmap above shows blocks on its diagonal; reordering leaves never changes
which cluster a name belongs to.

**What membership means.** Names in one cluster are **near-substitutes for risk
purposes**. For sizing, stress and diversification the cluster -- not the ticker
-- is the unit of exposure: three names in a cluster at 8% each is a 24% bet,
not three 8% bets. That is the practical use of this section. Concretely, it
means an equal-weight rule applied at ticker level quietly overweights whichever
cluster happens to contain the most names.

**What it does not mean.** A cluster is not a sector, not a recommendation and
not a permission to treat its members as interchangeable businesses. It is a
statement about how their prices moved over these {result.n_days:,} sessions,
estimated in-sample; a different window can move a borderline name across the
cut. Where two names sit in one cluster and one sector, the sector label is the
weaker of the two facts -- correlation is what the portfolio actually feels.
        """,
    )


def _cluster_reading(membership: dict, n_assets: int) -> None:
    """One sentence on what the cluster structure implies for sizing."""
    sizes = [len(v) for v in membership.values()]
    if not sizes:
        return
    n_clusters = len(sizes)
    largest = max(sizes)
    singletons = sum(1 for s in sizes if s == 1)

    if n_clusters == n_assets:
        st.caption(
            f"Every one of the {n_assets} holdings sits in its own cluster at this "
            f"cut: no pair shares enough correlation to be treated as one bet. That "
            f"is the best structure this page can report, and it is uncommon in a "
            f"long-only equity book."
        )
        return

    alone = ""
    if singletons == 1:
        alone = " and one name standing alone"
    elif singletons > 1:
        alone = f" and {singletons} names standing alone"

    st.caption(
        f"{n_assets} holdings collapse into {n_clusters} correlation groups at this "
        f"cut, the largest holding {largest} of them{alone}. Size the portfolio at "
        f"the group level: that {largest}-name cluster is one exposure however many "
        f"tickers carry it, and the diversification actually on offer here is closer "
        f"to {n_clusters} bets than to {n_assets}."
    )


# ---------------------------------------------------------------------------
# 5. principal components
# ---------------------------------------------------------------------------
def _pca(pca: dict, n_assets: int) -> None:
    ui_theme.section("PRINCIPAL COMPONENTS", "\U0001f9ec")

    if not pca or not pca.get("ok"):
        ui.empty_state(
            "Factor decomposition unavailable",
            str(pca.get("reason") or "")
            or "The factor block was not produced on this run, so the portfolio's "
            "independent directions of movement could not be extracted.",
            icon="\U0001f9ec",
            hint="PCA needs at least two holdings with non-zero variance and at "
            "least two dates on which every one of them traded.",
        )
        return

    interpretation = str(pca.get("interpretation") or "").strip()
    if interpretation:
        st.markdown(f"> **{interpretation}**")

    ratios = _floats(pca.get("explained_variance_ratio"))
    cumulative = _floats(pca.get("cumulative"))

    left, right = st.columns(2, gap="large")
    with left:
        ui.chart(
            ch.pca_scree(
                pca.get("explained_variance_ratio"),
                pca.get("cumulative"),
                height=_PCA_PANEL_HEIGHT,
            )
        )
        st.caption(_scree_caption(ratios, cumulative, pca, n_assets))
    with right:
        ui.chart(ch.pca_loadings_heatmap(pca.get("loadings"), height=_PCA_PANEL_HEIGHT))
        st.caption(
            "Each cell is a holding's correlation with a component, so it reads on "
            "the same -1 to +1 scale as the matrix above. A column where every cell "
            "is large and the same sign is a common factor everything shares; a "
            "column that splits the book into positives and negatives is a spread "
            "-- one group up while the other is down. Squared loadings across the "
            "full spectrum sum to 1 for every holding."
        )

    _pca_methodology(pca, ratios, n_assets)


def _scree_caption(ratios: list[float], cumulative: list[float], pca: dict, n_assets: int) -> str:
    if not ratios:
        return "No spectrum was produced for this portfolio."

    pc1 = ratios[0]
    parts = [
        f"PC1 explains {fmt_ratio_pct(pc1, 1)} of the variance in these "
        f"{n_assets} standardised return series"
    ]
    if len(ratios) > 1:
        parts.append(f", PC2 a further {fmt_ratio_pct(ratios[1], 1)}")
    parts.append(". ")

    needed = _int_or_none(pca.get("components_for_target"))
    if needed is not None:
        verb = "component covers" if needed == 1 else "components cover"
        parts.append(
            f"{needed} {verb} {fmt_ratio_pct(CUMULATIVE_TARGET, 0)} of it, against "
            f"{n_assets} tickers held. "
        )
    if cumulative and len(cumulative) > 1:
        parts.append(f"The first two together reach {fmt_ratio_pct(cumulative[1], 0)}. ")
    parts.append(
        "Eigenvalues sum to the number of assets, so the bars are shares of a fixed "
        "total: a tall first bar is not a scaling artefact, it is genuinely one "
        "direction absorbing the book."
    )
    return "".join(parts)


def _pca_methodology(pca: dict, ratios: list[float], n_assets: int) -> None:
    n_obs = _int_or_none(pca.get("n_obs")) or 0
    pc1 = ratios[0] if ratios else float("nan")
    per_asset = float(n_obs) / n_assets if n_assets else float("nan")
    ratio_note = (
        f"Here {n_obs:,} observations support {n_assets} assets, a ratio of "
        f"{fmt_num(per_asset, 1)} to 1."
    )
    if n_obs < MIN_OBSERVATIONS:
        ratio_note += (
            f" That is below the {MIN_OBSERVATIONS}-observation minimum this app "
            "applies elsewhere, so treat the structure as indicative only."
        )

    ui.methodology(
        "PCA on the correlation matrix -- the formula, the convention, the caveat",
        f"""
**What it does.** Principal component analysis on the **correlation** matrix
finds the independent directions of shared movement in the portfolio. It
decomposes `C = V L V'`, where `L` holds the eigenvalues in descending order and
each column of `V` is a direction along which the holdings move together. The
share of variance each direction carries is
`explained_variance_ratio_i = lambda_i / sum_j(lambda_j)`, and the reported
loadings are `V * sqrt(L)` -- which makes every cell of the loadings heatmap a
plain correlation between a holding and a component.

**Why the correlation matrix and not the covariance matrix.** Correlation
standardises every holding to unit variance first. On raw covariance, the single
most volatile name would capture PC1 on the strength of its own variance alone,
and the decomposition would describe volatility rather than co-movement.

**PC1 is almost always "the market".** In a long-only equity book the first
component nearly always loads positively on everything, because the dominant
shared driver is the direction of equities as an asset class. That is not a flaw
in the method; it is the fact the method was built to expose. The practical
reading is blunt: **a portfolio whose PC1 explains 80% or more of variance owns
one bet wearing many tickers** -- the ticker count is bookkeeping, the exposure
is singular. This portfolio's PC1 is at {fmt_ratio_pct(pc1, 0)}; the bands used
to word the interpretation above are {fmt_ratio_pct(PC1_DOMINANT, 0)} (dominant),
{fmt_ratio_pct(PC1_MEANINGFUL, 0)} (meaningful) and
{fmt_ratio_pct(PC1_BALANCED, 0)} (present but not dominant).

**Sign convention.** Eigenvectors are only defined up to sign, so each component
is oriented to make its largest loading positive. Without that, PC1 could
legitimately print as "minus the market" and flip between runs.

**Independent bets.** `1 / sum(ratio_i^2)`, the participation ratio of the
spectrum. It equals the asset count for a perfectly diagonal correlation matrix
and 1.0 when every name is the same bet -- the risk-side counterpart to the
capital-side `1 / HHI`.

**Caveats, stated plainly.**

* *In-sample and unstable.* Eigenvalues need the observation count to be
  comfortably larger than the asset count. {ratio_note}
* *Complete cases only.* Correlations feeding the decomposition use dates on
  which every asset traded. That costs observations but guarantees a positive
  semi-definite matrix, so the eigenvalues are real and non-negative.
* *Statistical, not economic.* PC2 and PC3 are whatever is left after PC1, in
  whatever mixture the sample produced. They are not "rates" or "the rupee"
  unless you can read that story off the loadings yourself, and the labels are
  not stable across windows. Only PC1 reliably means something.
* *Linear.* Like the correlations underneath it, the decomposition sees
  straight-line co-movement and is blind to tail dependence.
        """,
    )


# ---------------------------------------------------------------------------
# 6. correlation through time
# ---------------------------------------------------------------------------
def _through_time(result: Any, cfg: Any, avg: float) -> None:
    ui_theme.section("CORRELATION THROUGH TIME", "\U0001f4c8")

    rolling = result.rolling
    series = (
        rolling["correlation"]
        if isinstance(rolling, pd.DataFrame) and "correlation" in rolling.columns
        else None
    )
    clean = (
        pd.to_numeric(series, errors="coerce").dropna()
        if series is not None
        else pd.Series(dtype="float64")
    )

    window = int(getattr(cfg, "rolling_window", 63) or 63)
    benchmark_name = str(getattr(cfg, "benchmark_name", "the benchmark"))

    if len(clean) < 2:
        ui.empty_state(
            "No rolling correlation series",
            f"A trailing {window}-session correlation against {benchmark_name} needs "
            f"more overlapping history than this window contains, or the benchmark "
            f"series could not be aligned with the portfolio at all.",
            icon="\U0001f4c8",
            hint="Widen the history window in the sidebar, or pick a benchmark that "
            "trades on the same calendar as your holdings.",
        )
        return

    ui.chart(
        ch.rolling_correlation_chart(
            series,
            label=f"Portfolio vs {benchmark_name}",
            window=window,
            threshold=INSIGHTS.high_correlation,
        )
    )

    last = float(clean.iloc[-1])
    mean = float(clean.mean())
    lowest = float(clean.min())
    highest = float(clean.max())

    st.caption(
        f"Latest {fmt_num(last, 2)} against a window mean of {fmt_num(mean, 2)}, "
        f"having ranged from {fmt_num(lowest, 2)} to {fmt_num(highest, 2)} across "
        f"{len(clean):,} rolling windows. **Correlations rise in crises -- exactly "
        f"when diversification is needed most.** In a risk-off session almost "
        f"everything liquid is sold at once, so the co-movement measured on calm "
        f"days is precisely the co-movement you do not get on the day it would have "
        f"paid. A portfolio should be sized against the peak of this line "
        f"({fmt_num(highest, 2)}), not its average. Note what this series is: the "
        f"**portfolio against {benchmark_name}**, not the average correlation among "
        f"your own holdings -- that figure is estimated once over the full window "
        f"and is the {fmt_num(avg, 2)} at the top of this page."
    )

    ui.methodology(
        "The rolling estimate, and how to read a line that lags",
        f"""
**Construction.** Each point is the Pearson correlation between the portfolio's
daily returns and {benchmark_name}'s over the trailing **{window} sessions**
(about {fmt_num(window / TRADING_DAYS * 12.0, 0)} months of trading), taken from
`result.rolling["correlation"]`. The benchmark is reindexed onto the portfolio's
own dates, so a day the index traded and the portfolio did not becomes a gap
rather than silently shifting the series.

**It lags by construction.** A regime change enters the window on day one and
does not fully own it until day {window}. The line therefore turns *after* the
event, which is why the peak matters more than the latest reading. The first
{window - 1} points are undefined and are dropped, never drawn as zero.

**Overlapping windows.** Consecutive points share {window - 1} of their
{window} observations, so the series is smooth by design. Neither the smoothness
nor a run of similar values is evidence of stability.

**The threshold line** is drawn at {fmt_num(INSIGHTS.high_correlation, 2)}
(`config.INSIGHTS.high_correlation`) and only appears when the series actually
reaches it. Above that level the portfolio is, for risk purposes, a geared or
de-geared holding of the index rather than an independent set of decisions --
read it next to beta and R-squared on the Benchmark page, which say by how much.
        """,
    )


# ---------------------------------------------------------------------------
# 7. narrative
# ---------------------------------------------------------------------------
def _narrative(result: Any) -> None:
    ui_theme.section("WHAT THE STRUCTURE IMPLIES", "\U0001f9ed")
    st.caption(
        f"Diversification and concentration observations only, each fired by a "
        f"documented threshold in `config.InsightThresholds` -- correlation above "
        f"{fmt_num(INSIGHTS.high_correlation, 2)}, a largest position above "
        f"{fmt_pct(INSIGHTS.concentration_warn_pct, 0)}, effective bets below "
        f"{fmt_ratio_pct(INSIGHTS.effective_holdings_ratio, 0)} of the holding "
        f"count -- and carrying the evidence that fired it. Observations about what "
        f"already happened; not predictions, not recommendations, not advice."
    )
    ui.insight_list(
        result.insights,
        categories=("diversification", "concentration"),
        empty_message="No diversification or concentration rule fired for this portfolio.",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Correlation & factors",
        "Whether these holdings are many bets, or one bet wearing many tickers",
        icon="\U0001f517",
    )

    result = guards.require_result()
    if result is None:
        return
    if not guards.require_multiple_assets(result, what="Correlation analysis"):
        return

    cfg = result.config
    correlation = result.correlation
    if not isinstance(correlation, pd.DataFrame) or correlation.empty:
        ui.empty_state(
            "No correlation matrix",
            "The analysis ran, but fewer than two holdings produced overlapping "
            "price history, so no pair of names can be compared.",
            icon="\U0001f9ff",
            hint="Correlation needs at least two holdings quoted on the same trading dates.",
        )
        ui.warning_list(result.warnings, "Data notes from this run")
        return

    factors = _factor_block(result)
    clusters = _sub_block(factors, "clusters")
    pca = _sub_block(factors, "pca")
    pairs = _pairs(correlation)
    n_assets = len(_labels(correlation))
    avg, avg_source = _average_correlation(factors, pairs)

    if not factors:
        st.info(
            "The factor block is empty for this run: the clustering and the "
            "principal-component decomposition were not produced, so those sections "
            "below are unavailable. The correlation matrix itself is unaffected."
        )

    _kpis(result, factors, pairs, avg, avg_source, n_assets)

    ui_theme.spacer(0.4)
    _matrix(result, clusters, avg, n_assets)

    ui_theme.spacer(0.4)
    _pairwise(pairs, n_assets)

    ui_theme.spacer(0.4)
    _clustering(result, clusters, pairs)

    ui_theme.spacer(0.4)
    _pca(pca, n_assets)

    ui_theme.spacer(0.4)
    ui_theme.section("Factor regression", "📐")
    st.caption(
        "The PCA above is unsupervised — it finds independent directions without "
        "naming them. This regresses the portfolio on NAMED factors instead, so the "
        "question becomes 'how much size, value and momentum exposure is in here, and "
        "is any of it distinguishable from zero?'"
    )
    factor_panel.render(result)

    ui_theme.spacer(0.4)
    _through_time(result, cfg, avg)

    ui_theme.spacer(0.4)
    _narrative(result)
