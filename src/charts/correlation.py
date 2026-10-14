"""Correlation, principal-component and clustering figures.

These charts answer one question a holdings table cannot: *how many
independent bets does this portfolio actually contain?* A ten-name portfolio
of large-cap Indian banks is one bet wearing ten costumes, and the figures
here exist to make that obvious at a glance.

Conventions used throughout this module
---------------------------------------
* Correlation is **Pearson correlation of daily arithmetic returns** over the
  loaded history window. It is a linear, in-sample, unconditional measure: it
  says nothing about tail dependence, and empirically it rises in crises -
  exactly when the diversification it implies is most needed. The captions on
  these figures say so rather than letting the colour do the lying.
* The correlation heatmap's colour scale is **hard-pinned to [-1, +1]**. A
  scale that auto-ranged to the data would make a well-diversified portfolio
  look identical to a perfectly correlated one, and would make two portfolios
  impossible to compare side by side.
* PCA is run on the **correlation** matrix (i.e. standardised returns), so a
  single volatile holding cannot capture the first component on the strength
  of its variance alone.

No Streamlit import: every builder here is a pure ``go.Figure`` factory.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import theme
from src.config import INSIGHTS

__all__ = [
    "correlation_heatmap",
    "rolling_correlation_chart",
    "pca_scree",
    "pca_loadings_heatmap",
    "dendrogram",
]

#: Above this many assets per side, per-cell text collides with its
#: neighbours; the hover carries the exact number instead.
MAX_ANNOTATED_CELLS = 20


# ---------------------------------------------------------------------------
# defensive plumbing
# ---------------------------------------------------------------------------
def _never_raises(
    reason: str,
) -> Callable[[Callable[..., go.Figure]], Callable[..., go.Figure]]:
    """Turn any unexpected exception into a styled placeholder figure.

    A dashboard that raises out of a chart builder takes the whole page down
    with it. Every public function in this module is wrapped so the worst case
    is one honest placeholder tile that names what went wrong.
    """

    def decorate(fn: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
                detail = str(exc).replace("\n", " ")[:120]
                return theme.empty_figure(f"{reason} ({type(exc).__name__}: {detail})")

        return wrapper

    return decorate


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame and coerce every column to float, never touching the input."""
    out = frame.copy()
    out.columns = [str(c) for c in out.columns]
    out.index = [str(i) for i in out.index]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _square_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict a matrix to the labels present on both axes, in row order."""
    cols = set(frame.columns)
    common = [i for i in frame.index if i in cols]
    return frame.loc[common, common]


def _reorder(frame: pd.DataFrame, order: Sequence[str] | None) -> tuple[pd.DataFrame, bool]:
    """Reorder rows and columns by ``order``; unknown labels keep their place.

    Returns the reordered frame plus a flag saying whether the supplied order
    actually changed anything, so the subtitle can be honest about it.
    """
    if not order:
        return frame, False
    original = list(frame.columns)
    wanted: list[str] = []
    seen: set[str] = set()
    for label in order:
        key = str(label)
        if key in frame.columns and key not in seen:
            wanted.append(key)
            seen.add(key)
    if not wanted:
        return frame, False
    cols = wanted + [c for c in original if c not in seen]
    return frame.loc[cols, cols], cols != original


def _cell_text_color(value: float) -> str:
    """Readable text colour for one cell of the diverging scale.

    ``theme.DIVERGING`` is dark at both ends and near-white in the middle, so
    the contrast colour has to flip with the *magnitude* of the value rather
    than with the app's light/dark mode. Both colours are named by the theme.
    """
    if not np.isfinite(value):
        return theme.palette()["subtext"]
    return theme.DARK_PALETTE["text"] if abs(value) >= 0.55 else theme.LIGHT_PALETTE["text"]


def _annotate_matrix(
    fig: go.Figure,
    values: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    font_size: float,
) -> None:
    """Write each cell's value onto a heatmap with per-cell contrast."""
    for r, row_label in enumerate(y_labels):
        for c, col_label in enumerate(x_labels):
            v = float(values[r, c])
            fig.add_annotation(
                x=col_label,
                y=row_label,
                text="-" if not np.isfinite(v) else f"{v:.2f}",
                showarrow=False,
                font={"size": float(font_size), "color": _cell_text_color(v)},
            )


def _diverging_colorbar(title: str, tickvals: Sequence[float]) -> dict[str, Any]:
    """Slim, low-chrome colourbar shared by both heatmaps."""
    p = theme.palette()
    return {
        "title": {"text": title, "font": {"size": 11, "color": p["subtext"]}},
        "thickness": 12,
        "len": 0.86,
        "outlinewidth": 0,
        "tickvals": [float(v) for v in tickvals],
        "tickfont": {"size": 10, "color": p["subtext"]},
    }


# ---------------------------------------------------------------------------
# correlation matrix
# ---------------------------------------------------------------------------
@_never_raises("Correlation heatmap could not be rendered")
def correlation_heatmap(
    corr: pd.DataFrame | None,
    order: Sequence[str] | None = None,
    height: int | None = None,
    max_annotated: int = MAX_ANNOTATED_CELLS,
) -> go.Figure:
    """Pairwise correlation matrix of daily returns, as an annotated heatmap.

    Shows every pairwise Pearson correlation between holdings with the colour
    scale **hard-pinned to [-1, +1]**, so the same colour always means the same
    number - across rebalances, across portfolios and across screenshots. Cells
    carry the value to two decimals whenever the matrix is small enough for the
    text to fit; above ``max_annotated`` assets the value moves into the hover
    rather than overlapping its neighbours.

    Parameters
    ----------
    corr:
        Square correlation matrix indexed and columned by ticker. Values are
        expected in [-1, 1]; NaN renders as a blank cell, never as zero.
    order:
        Optional label order - typically ``factors.clustered_order(corr)`` - so
        similar assets sit next to each other and blocks of related names
        become visible. Labels absent from ``corr`` are ignored; labels missing
        from ``order`` keep their original relative position at the end.
    height:
        Caller override. Otherwise the figure grows with the number of assets
        so cells stay roughly square.

    Assumptions
    -----------
    Correlation here is linear, in-sample and unconditional, computed on
    overlapping calendar dates only - so a recently listed holding is measured
    over a shorter window than the rest of the portfolio.
    """
    if corr is None or not isinstance(corr, pd.DataFrame) or corr.empty:
        return theme.empty_figure(
            "No correlation matrix: fewer than two holdings had overlapping price history."
        )

    frame = _square_common(_numeric_frame(corr))
    if frame.shape[0] and not np.isfinite(frame.to_numpy(dtype=float)).any():
        return theme.empty_figure(
            "Every pairwise correlation is undefined - the holdings have no overlapping "
            "returns, or their prices never moved."
        )

    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame = _square_common(frame)
    if frame.shape[0] < 2:
        return theme.empty_figure(
            "A correlation matrix needs at least two holdings with overlapping returns."
        )

    frame, reordered = _reorder(frame, order)
    labels = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=float)
    n = len(labels)

    upper = values[np.triu_indices(n, k=1)]
    avg = float(np.nanmean(upper)) if upper.size and np.isfinite(upper).any() else float("nan")

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale=theme.DIVERGING,
            xgap=1,
            ygap=1,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation  %{z:.2f}<extra></extra>",
            colorbar=_diverging_colorbar("Pearson rho", [-1, -0.5, 0, 0.5, 1]),
            name="",
        )
    )

    annotated = n <= max_annotated
    if annotated:
        _annotate_matrix(fig, values, labels, labels, font_size=max(7.5, 13.0 - 0.28 * n))

    subtitle = "Daily returns; colour scale pinned to -1.00 .. +1.00"
    if np.isfinite(avg):
        subtitle += f" - average pairwise rho {avg:.2f}"
    if reordered:
        subtitle += " - grouped by hierarchical clustering"

    theme.apply(
        fig,
        title="Correlation matrix",
        subtitle=subtitle,
        height=height or int(np.clip(30 * n + 170, 320, 780)),
        legend=False,
    )
    fig.update_xaxes(
        title_text="Asset",
        side="bottom",
        tickangle=-45 if n > 6 else 0,
        showgrid=False,
    )
    fig.update_yaxes(title_text="Asset", autorange="reversed", showgrid=False)

    note = (
        "Pearson correlation of daily returns on overlapping dates. It is a linear, "
        "in-sample measure and it rises in crises - the diversification it implies is "
        "weakest exactly when it is needed most."
    )
    if not annotated:
        note += f" Cell values are hidden above {max_annotated} holdings; hover for the figure."
    theme.caption(fig, note)
    return fig


# ---------------------------------------------------------------------------
# rolling correlation
# ---------------------------------------------------------------------------
@_never_raises("Rolling correlation chart could not be rendered")
def rolling_correlation_chart(
    series: pd.Series | None,
    label: str = "Rolling correlation",
    window: int | None = None,
    threshold: float | None = INSIGHTS.high_correlation,
    height: int | None = None,
) -> go.Figure:
    """Correlation through time, with the diversification threshold marked.

    Shows how a correlation - portfolio versus benchmark, or the average
    pairwise correlation across holdings - has evolved. The y-axis is fixed to
    [-1, +1] so two periods or two portfolios can be compared without
    re-reading the scale; the zero line is emphasised because the sign is the
    whole point; and the series mean is drawn as a reference so the reader can
    judge whether the latest value is actually unusual.

    Parameters
    ----------
    series:
        Correlation values indexed by date. NaN warm-up values at the start of
        the rolling window are dropped, never plotted as zero.
    label:
        Series name, used in the hover box.
    window:
        Rolling window length in trading days, for the subtitle.
    threshold:
        Correlation above which diversification is treated as broken down
        (defaults to ``config.INSIGHTS.high_correlation``). Drawn only when the
        series actually reaches it, so the chart stays clean when it does not.

    Assumptions
    -----------
    Each point uses the trailing ``window`` observations, so the series is
    backward-looking and reacts to a regime change with a lag of up to one
    full window.
    """
    if series is None:
        return theme.empty_figure(
            "No rolling correlation series was supplied - the analysis has not been run."
        )
    data = pd.to_numeric(pd.Series(series).copy(), errors="coerce").dropna()
    if data.empty:
        return theme.empty_figure(
            "Rolling correlation is empty - the rolling window is longer than the "
            "overlapping price history."
        )
    if len(data) < 2:
        return theme.empty_figure(
            "Only one rolling window could be computed - a line needs at least two points."
        )

    p = theme.palette()
    mean_rho = float(data.mean())
    last_rho = float(data.iloc[-1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.to_numpy(dtype=float),
            mode="lines",
            name=label,
            line={"color": p["accent"], "width": 2.0},
            fill="tozeroy",
            fillcolor=p["accent_fill"],
            hovertemplate=f"{label}  %{{y:.2f}}<extra></extra>",
        )
    )

    subtitle = "Pearson rho of daily returns on a trailing window"
    if window:
        subtitle = f"Pearson rho of daily returns, trailing {int(window)}-day window"
    subtitle += f" - latest {last_rho:.2f}"

    theme.apply(
        fig,
        title="Correlation through time",
        subtitle=subtitle,
        height=height or 320,
        unified_hover=True,
        legend=False,
    )
    theme.add_zero_line(fig)
    theme.add_reference_line(fig, mean_rho, f"Mean {mean_rho:.2f}", color=p["benchmark"])
    if (
        threshold is not None
        and np.isfinite(float(threshold))
        and float(data.max()) >= float(threshold)
    ):
        theme.add_reference_line(
            fig,
            float(threshold),
            f"Diversification breaks down above {float(threshold):.2f}",
            color=p["warning"],
            dash="dash",
        )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(
        title_text="Correlation (rho, dimensionless)",
        range=[-1.05, 1.05],
        tickformat=".2f",
        tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    return fig


# ---------------------------------------------------------------------------
# principal components
# ---------------------------------------------------------------------------
def _as_percent_vector(values: Any) -> np.ndarray:
    """Coerce an explained-variance vector to percent units.

    Accepts fractions or percentages and detects which from the LARGEST entry,
    so callers can pass ``explained_variance_ratio`` or ``cumulative`` straight
    through without knowing the convention.

    Detecting from the sum instead would be wrong for a CUMULATIVE vector:
    ``[0.60, 0.85, 0.93, 0.97, 1.00]`` sums to 4.35, which reads as
    "already percent", and the line would then be drawn between 0.6 and 1.0 on
    an axis fixed to [0, 105] -- invisible, on every portfolio with two or more
    holdings. The maximum is unambiguous for both shapes: a fraction vector
    never exceeds 1, a percent vector's leading component always does.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    peak = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else float("nan")
    return arr * 100.0 if np.isfinite(peak) and peak <= 1.5 else arr


@_never_raises("Scree plot could not be rendered")
def pca_scree(
    explained: np.ndarray | Sequence[float] | None,
    cumulative: np.ndarray | Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    height: int | None = None,
) -> go.Figure:
    """Scree plot: variance explained per principal component, plus cumulative.

    Bars give each component's share of total return variance (left axis); the
    line gives the running total (right axis). The first bar is highlighted
    because it is the headline number - a portfolio whose PC1 explains 80% of
    variance is, statistically, one position. The 80% cumulative marker is the
    practical read of "how many independent bets do I actually own".

    Parameters
    ----------
    explained:
        ``explained_variance_ratio`` from ``factors.pca_decomposition``, as
        fractions summing to ~1 or as percentages; units are auto-detected.
    cumulative:
        Optional running total. Derived from ``explained`` when omitted or
        when it does not line up with ``explained``.
    labels:
        Optional component names; defaults to ``PC1 .. PCk``.

    Assumptions
    -----------
    The decomposition runs on the *correlation* matrix, so components are
    scale-free and one high-volatility holding cannot own PC1 on variance
    alone. Eigenvalues are in-sample and unstable when the number of
    observations is not comfortably larger than the number of assets.
    """
    if explained is None:
        return theme.empty_figure(
            "No PCA result was supplied - the factor decomposition has not been run."
        )
    pct = _as_percent_vector(explained)
    if pct.size == 0:
        return theme.empty_figure(
            "PCA produced no components - at least two holdings with overlapping "
            "returns are required."
        )
    finite = np.isfinite(pct)
    if not finite.any():
        return theme.empty_figure(
            "Explained variance is undefined - the return matrix is singular or all-NaN."
        )
    pct = np.where(finite, pct, 0.0)

    fallback = np.cumsum(pct)
    if cumulative is None:
        cum = fallback
    else:
        supplied = _as_percent_vector(cumulative)
        if supplied.size != pct.size or not np.isfinite(supplied).any():
            cum = fallback
        else:
            cum = np.where(np.isfinite(supplied), supplied, fallback)

    n = int(pct.size)
    names = [f"PC{i + 1}" for i in range(n)]
    if labels is not None:
        supplied_names = [str(x) for x in labels][:n]
        if len(supplied_names) == n:
            names = supplied_names

    p = theme.palette()
    bar_colors = [p["accent_2"] if i == 0 else p["accent"] for i in range(n)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=names,
            y=pct,
            name="Explained variance",
            marker={"color": bar_colors, "line": {"width": 0}},
            hovertemplate="<b>%{x}</b><br>Explains %{y:.1f}% of return variance<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=names,
            y=cum,
            name="Cumulative",
            mode="lines+markers",
            line={"color": p["warning"], "width": 2.2},
            marker={"size": 7, "color": p["warning"]},
            hovertemplate="<b>%{x}</b><br>Cumulative %{y:.1f}% of variance<extra></extra>",
        ),
        secondary_y=True,
    )

    first = float(pct[0])
    reached = int(np.searchsorted(cum, 80.0) + 1) if float(np.nanmax(cum)) >= 80.0 else 0
    subtitle = "Eigen-decomposition of the daily-return correlation matrix"
    subtitle += f" - PC1 alone explains {first:.0f}% of variance"

    theme.apply(
        fig,
        title="Principal components of return variance",
        subtitle=subtitle,
        height=height or 340,
    )
    theme.pct_axis(fig, axis="y")
    fig.update_yaxes(
        title_text="Explained variance (%)",
        range=[0.0, float(max(np.nanmax(pct) * 1.18, 5.0))],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Cumulative variance (%)",
        range=[0.0, 105.0],
        showgrid=False,
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Principal component (ordered by eigenvalue)")
    # theme.add_reference_line targets the primary axis only; the 80% marker
    # belongs to the cumulative (secondary) scale, so it is drawn directly with
    # theme colours.
    fig.add_hline(
        y=80.0,
        secondary_y=True,
        line_width=1.2,
        line_dash="dot",
        line_color=p["muted"],
        annotation_text="80% of variance",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=p["subtext"],
    )

    note = (
        "A dominant first component means the holdings are largely one bet on the same "
        "market factor."
    )
    if reached:
        note += f" {reached} component(s) are needed to reach 80% of variance."
    theme.caption(fig, note)
    return fig


@_never_raises("Loadings heatmap could not be rendered")
def pca_loadings_heatmap(
    loadings: pd.DataFrame | None,
    max_components: int = 6,
    height: int | None = None,
    max_annotated: int = MAX_ANNOTATED_CELLS,
) -> go.Figure:
    """Heatmap of each holding's loading on each principal component.

    Rows are holdings sorted by absolute PC1 loading - so the names driving the
    common market factor sit at the top - and columns are components. The
    diverging scale is symmetric about zero so +0.40 and -0.40 read as equal
    and opposite, with the limit taken from the data because eigenvector
    entries are rarely anywhere near +/-1 and a pinned scale would wash the
    whole matrix out.

    Parameters
    ----------
    loadings:
        ``factors.pca_decomposition()["loadings"]`` - assets on the index,
        ``PC1..PCk`` on the columns.
    max_components:
        Only the leading components are shown; the tail is noise for a
        retail-sized portfolio.

    Assumptions
    -----------
    Loadings are entries of unit-norm eigenvectors of the correlation matrix.
    The **sign of a whole component is arbitrary** - only the relative signs of
    assets within one component carry meaning (opposite signs mean the two
    names move against each other on that factor).
    """
    if loadings is None or not isinstance(loadings, pd.DataFrame) or loadings.empty:
        return theme.empty_figure(
            "No PCA loadings were supplied - the factor decomposition has not been run."
        )

    frame = _numeric_frame(loadings)
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.empty or frame.shape[1] == 0:
        return theme.empty_figure(
            "PCA loadings are entirely undefined - the return matrix is singular or all-NaN."
        )
    if frame.shape[0] < 2:
        return theme.empty_figure(
            "Loadings need at least two holdings - a single asset has no factor structure."
        )

    if max_components and frame.shape[1] > max_components:
        frame = frame.iloc[:, :max_components]

    lead = frame.iloc[:, 0].abs()
    frame = frame.loc[lead.sort_values(ascending=False, na_position="last").index]

    values = frame.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    limit = float(min(1.0, max(limit, 0.2)))

    x_labels = [str(c) for c in frame.columns]
    y_labels = [str(i) for i in frame.index]
    n_rows, n_cols = values.shape

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=x_labels,
            y=y_labels,
            zmin=-limit,
            zmax=limit,
            colorscale=theme.DIVERGING,
            xgap=1,
            ygap=1,
            hovertemplate="<b>%{y}</b> on <b>%{x}</b><br>Loading  %{z:+.2f}<extra></extra>",
            colorbar=_diverging_colorbar(
                "Loading", [-limit, -limit / 2.0, 0.0, limit / 2.0, limit]
            ),
            name="",
        )
    )

    if n_rows <= max_annotated and n_cols <= max_annotated:
        _annotate_matrix(fig, values, x_labels, y_labels, font_size=max(8.0, 12.0 - 0.2 * n_rows))

    theme.apply(
        fig,
        title="Factor loadings",
        subtitle=(
            "Eigenvectors of the daily-return correlation matrix - scale symmetric at "
            f"+/-{limit:.2f}"
        ),
        height=height or int(np.clip(30 * n_rows + 170, 300, 720)),
        legend=False,
    )
    fig.update_xaxes(title_text="Principal component", side="bottom", showgrid=False)
    fig.update_yaxes(title_text="Asset", autorange="reversed", showgrid=False)
    theme.caption(
        fig,
        "Rows are sorted by absolute PC1 loading. The sign of a component is arbitrary; "
        "only the relative signs of assets within the same component are meaningful.",
    )
    return fig


# ---------------------------------------------------------------------------
# hierarchical clustering
# ---------------------------------------------------------------------------
def _dendrogram_colorscale() -> list[str]:
    """Eight colours in the order ``create_dendrogram`` consumes them.

    The factory maps its colour list onto SciPy's link colours in alphabetical
    key order (b, c, g, k, m, r, w, y), and SciPy's ``C0`` - the
    above-threshold colour - resolves to ``b``. Sending the muted grey to that
    slot and the house categorical ramp to the cluster slots is what turns the
    factory's default green-and-magenta tree into a chart that belongs to the
    rest of the app.
    """
    p = theme.palette()
    cat = theme.CATEGORICAL
    return [
        p["muted"],  # b -> C0, links above the colour threshold
        cat[2],  # c -> C3 / C9
        cat[0],  # g -> C1 / C7
        cat[5],  # k -> C6
        cat[3],  # m -> C4
        cat[1],  # r -> C2 / C8
        p["muted"],  # w, unused by SciPy
        cat[4],  # y -> C5
    ]


@_never_raises("Dendrogram could not be rendered")
def dendrogram(
    linkage_matrix: np.ndarray | None,
    labels: Sequence[str] | None = None,
    orientation: str = "left",
    color_threshold: float | None = None,
    height: int | None = None,
) -> go.Figure:
    """Hierarchical clustering tree of the holdings, restyled to the house theme.

    Shows which holdings merge first - that is, which names the market treats
    as the same trade. The distance axis is the correlation distance
    ``sqrt(0.5 * (1 - rho))`` at which two branches join: a join near 0.00 is a
    pair of near-perfect substitutes, a join near 0.71 is an uncorrelated pair,
    and 1.00 is perfect inverse correlation.

    Parameters
    ----------
    linkage_matrix:
        SciPy linkage matrix of shape ``(n_assets - 1, 4)``, e.g.
        ``factors.cluster_assets()["linkage"]``.
    labels:
        Tickers in the same order as the observations that produced the
        linkage; must contain exactly ``linkage_matrix.shape[0] + 1`` entries.
        Generic ``Asset n`` labels are used when omitted.
    orientation:
        ``"left"`` (default - leaf labels read horizontally, which suits long
        tickers such as ``HDFCBANK.NS``) or ``"bottom"``. Any other value falls
        back to ``"left"``, because the factory renders the mirrored
        orientations with negated coordinates.
    color_threshold:
        Distance at which the tree is cut into coloured clusters; SciPy's
        default of ``0.7 * max(distance)`` applies when omitted.

    Assumptions
    -----------
    The tree reflects the linkage method chosen upstream (average linkage by
    default). Hierarchical clustering always returns a tree even when there is
    no real cluster structure, so read the merge distances, not just the shape.
    """
    if linkage_matrix is None:
        return theme.empty_figure(
            "No linkage matrix was supplied - hierarchical clustering has not been run."
        )
    matrix = np.asarray(linkage_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or matrix.shape[0] < 1:
        return theme.empty_figure(
            "Linkage matrix is malformed - expected an (n_assets - 1) x 4 SciPy linkage array."
        )
    if not np.isfinite(matrix).all():
        return theme.empty_figure(
            "Linkage matrix holds non-finite distances - the correlation matrix is degenerate."
        )

    n_assets = int(matrix.shape[0]) + 1
    if labels is None:
        leaf_labels = [f"Asset {i + 1}" for i in range(n_assets)]
    else:
        leaf_labels = [str(x) for x in labels]
    if len(leaf_labels) != n_assets:
        return theme.empty_figure(
            f"Linkage implies {n_assets} assets but {len(leaf_labels)} labels were supplied."
        )
    if n_assets < 2:
        return theme.empty_figure(
            "Clustering needs at least two holdings - a single asset has no tree."
        )

    side = orientation if orientation in {"left", "bottom"} else "left"
    fig = ff.create_dendrogram(
        np.zeros((n_assets, 1)),
        orientation=side,
        labels=leaf_labels,
        linkagefun=lambda _distances: matrix,
        colorscale=_dendrogram_colorscale(),
        color_threshold=color_threshold,
    )

    p = theme.palette()
    dist_axis = "x" if side == "left" else "y"
    hover = "Branches merge at correlation distance %{" + dist_axis + ":.3f}<extra></extra>"
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        colour = getattr(marker, "color", None) if marker is not None else None
        trace.update(
            line={"color": colour or p["accent"], "width": 1.6},
            hovertemplate=hover,
            showlegend=False,
        )

    max_distance = float(np.nanmax(matrix[:, 2])) if matrix.size else 1.0
    theme.apply(
        fig,
        title="Holdings clustered by correlation",
        subtitle=(
            "Hierarchical clustering of daily-return correlation distances - deepest "
            f"merge at {max_distance:.2f}"
        ),
        height=height or int(np.clip(26 * n_assets + 170, 300, 760)),
        legend=False,
    )
    axis_reset: dict[str, Any] = {
        "showline": False,
        "mirror": False,
        "ticks": "",
        "zeroline": False,
        "linecolor": p["axis"],
    }
    dist_title = "Correlation distance   sqrt(0.5 * (1 - rho))"
    if side == "left":
        fig.update_xaxes(title_text=dist_title, showgrid=True, gridcolor=p["grid"], **axis_reset)
        fig.update_yaxes(title_text="", showgrid=False, **axis_reset)
    else:
        fig.update_yaxes(title_text=dist_title, showgrid=True, gridcolor=p["grid"], **axis_reset)
        fig.update_xaxes(title_text="", showgrid=False, tickangle=-45, **axis_reset)

    theme.caption(
        fig,
        "Distance is sqrt(0.5 * (1 - rho)): 0.00 is perfectly correlated, 0.71 is "
        "uncorrelated, 1.00 is perfectly inversely correlated.",
    )
    return fig
