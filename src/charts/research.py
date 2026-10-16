"""Figures for the company research page.

Built on the shared theme so every chart on the page matches the rest of the
app, and so the same builders can be reused verbatim by the HTML report — which
is the reason each function takes plain frames and numbers rather than page
objects. Nothing here imports Streamlit or fetches anything.

Every builder satisfies the package-wide contract the chart tests enforce: it
always returns a ``go.Figure``. When there is nothing honest to draw it returns
``theme.empty_figure`` carrying the *specific* reason, in the chart, rather than
``None``. A chart axis with no data on it is worse than no chart — it implies a
measurement was made — but a blank rectangle is worse still, because a reader
cannot tell a quiet quarter from a broken feed.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = [
    "financial_trend_chart",
    "common_size_chart",
    "sensitivity_heatmap",
    "peer_median_chart",
    "volatility_regime_chart",
    "scorecard_breakdown_chart",
    "sentiment_scatter",
]

#: Height for the full-width charts on the page.
_TALL = 420
_STANDARD = 360

#: Sessions of history below which a volatility reading says more about the
#: window than about the company.
_MIN_VOL_SESSIONS = 5

#: Days of coverage below which a tone/return relationship is not measurable.
_MIN_SENTIMENT_DAYS = 3


# ---------------------------------------------------------------------------
# safety net
# ---------------------------------------------------------------------------
_F = TypeVar("_F", bound=Callable[..., go.Figure])


def _never_raises(fn: _F) -> _F:
    """Enforce the chart contract: a builder always yields a Figure.

    Each function below names every degenerate shape it can anticipate and
    explains it in the figure. This is the backstop for the shapes it cannot,
    because a dashboard that raises inside a render loop takes the whole page
    down with it — and on this page the loop is a report a reader is midway
    through.
    """

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive  # pragma: no cover
            return theme.empty_figure(
                f"This figure could not be built from the data supplied ({type(exc).__name__})."
            )

    return cast(_F, _wrapped)


def _clean(values: Sequence[float]) -> list[float | None]:
    """NaN -> None so Plotly breaks the line instead of drawing through zero."""
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in values
    ]


# ---------------------------------------------------------------------------
# financial history
# ---------------------------------------------------------------------------
@_never_raises
def financial_trend_chart(
    periods: Sequence[str],
    revenue: Sequence[float],
    profit: Sequence[float],
    *,
    currency_symbol: str = "",
    height: int = _STANDARD,
) -> go.Figure:
    """Revenue and net profit side by side, oldest first.

    Two bars rather than two lines: the levels are what a reader compares, and
    a line implies a continuous process between annual observations that does
    not exist. A company does not grow smoothly between 31 March and 31 March.
    """
    if not periods or len(revenue) != len(periods):
        return theme.empty_figure(
            "No aligned annual periods came back for this company, so there is no "
            "trend to draw.",
            height=height,
        )
    if all(v is None or math.isnan(float(v)) for v in revenue):
        return theme.empty_figure(
            "Every revenue figure returned for these years was blank.", height=height
        )

    palette = theme.palette()
    fig = go.Figure()
    fig.add_bar(
        x=list(periods),
        y=_clean(revenue),
        name="Revenue",
        marker_color=palette["accent"],
        hovertemplate=f"Revenue  {currency_symbol}%{{y:,.0f}}<extra></extra>",
    )
    if profit and len(profit) == len(periods):
        fig.add_bar(
            x=list(periods),
            y=_clean(profit),
            name="Net profit",
            marker_color=palette["gain"],
            hovertemplate=f"Net profit  {currency_symbol}%{{y:,.0f}}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.35)
    theme.apply(fig, height=height)
    theme.money_axis(fig, currency_symbol)
    return fig


@_never_raises
def common_size_chart(
    frame: pd.DataFrame,
    *,
    height: int = _TALL,
) -> go.Figure:
    """Common-size lines: every row expressed as a share of its own basis.

    The point of the chart is the *shape*, not the level -- a margin that
    quietly slid from 14% to 9% over four years is invisible in an absolute
    bar chart and obvious here.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return theme.empty_figure(
            "No common-size statement was available for this company.", height=height
        )

    colours = theme.CATEGORICAL
    fig = go.Figure()
    for index, label in enumerate(frame.index):
        values = pd.to_numeric(frame.loc[label], errors="coerce")
        if values.dropna().empty:
            continue
        fig.add_scatter(
            x=[str(c) for c in frame.columns],
            y=[None if pd.isna(v) else float(v) * 100.0 for v in values],
            mode="lines+markers",
            name=str(label),
            line={"color": colours[index % len(colours)], "width": 2},
            hovertemplate=f"{label}  %{{y:.1f}}%<extra></extra>",
        )
    if not fig.data:
        return theme.empty_figure(
            "Every line in the common-size statement was blank.", height=height
        )
    theme.apply(fig, height=height)
    theme.pct_axis(fig)
    theme.add_zero_line(fig)
    return fig


# ---------------------------------------------------------------------------
# valuation
# ---------------------------------------------------------------------------
@_never_raises
def sensitivity_heatmap(
    row_values: Sequence[float],
    col_values: Sequence[float],
    grid: Sequence[Sequence[float]],
    *,
    row_title: str = "",
    col_title: str = "",
    highlight: tuple[int, int] | None = None,
    height: int = _STANDARD,
) -> go.Figure:
    """Two-way sensitivity of value per share, as a heatmap.

    The diagonal that matters is the one through the current price: a valuation
    is only interesting where it disagrees with the market, and the colour scale
    is centred there rather than on the model's own average.

    ``highlight`` marks the (row, column) cell matching the base assumptions, so
    a reader can see at a glance how far the answer moves for a small change in
    either input.
    """
    if not row_values or not col_values or not grid:
        return theme.empty_figure("No sensitivity grid was supplied.", height=height)
    if len(grid) != len(row_values) or any(len(r) != len(col_values) for r in grid):
        return theme.empty_figure(
            "The sensitivity grid does not line up with its own axes, so no cell "
            "could be labelled honestly.",
            height=height,
        )

    z: list[list[float | None]] = [
        [None if v is None or math.isnan(float(v)) else float(v) for v in row] for row in grid
    ]
    if all(v is None for row in z for v in row):
        return theme.empty_figure(
            "Every cell of the sensitivity grid was blank.", height=height
        )

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{v:.1%}" for v in col_values],
            y=[f"{v:.1%}" for v in row_values],
            colorscale=theme.DIVERGING,
            colorbar={"title": "per share"},
            hovertemplate="%{y} / %{x}<br>value %{z:,.2f}<extra></extra>",
        )
    )

    if highlight is not None:
        row_index, col_index = highlight
        if 0 <= row_index < len(row_values) and 0 <= col_index < len(col_values):
            fig.add_shape(
                type="rect",
                x0=col_index - 0.5,
                x1=col_index + 0.5,
                y0=row_index - 0.5,
                y1=row_index + 0.5,
                line={"color": theme.palette()["accent"], "width": 3},
                fillcolor="rgba(0,0,0,0)",
            )

    theme.apply(fig, height=height)
    axis = {"title": f"{row_title} \\ {col_title}" if row_title and col_title else ""}
    fig.update_yaxes(**axis)
    fig.update_xaxes(title=col_title or "")
    return fig


# ---------------------------------------------------------------------------
# peers
# ---------------------------------------------------------------------------
@_never_raises
def peer_median_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    height: int = _STANDARD,
) -> go.Figure:
    """Each metric's subject value against its peer median, as a diverging bar.

    Bars are the *relative* gap to the median and are coloured by whether that
    gap is favourable, which is a property of the metric rather than of the
    sign. Being 40% above the peer median on ROE is green; being 40% above it
    on P/E is red, and a chart that coloured both by sign alone would be worse
    than useless.
    """
    usable = [
        row
        for row in rows
        if isinstance(row.get("relative"), (int, float)) and not math.isnan(float(row["relative"]))  # type: ignore[arg-type]
    ]
    if not usable:
        return theme.empty_figure(
            "No peer metric had both a figure for this company and a median to "
            "compare it against.",
            height=height,
        )

    labels = [str(row.get("label", "")) for row in usable]
    values = [float(row["relative"]) * 100.0 for row in usable]  # type: ignore[arg-type]
    colours = [
        theme.gain() if row.get("favourable") else theme.loss()  # type: ignore[arg-type]
        for row in usable
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            customdata=[
                [
                    row.get("subject_display", ""),
                    row.get("median_display", ""),
                    int(row.get("n_peers", 0) or 0),
                ]
                for row in usable
            ],
            hovertemplate=(
                "This company %{customdata[0]} vs peer median %{customdata[1]}"
                "<br>%{x:+.1f}% relative, %{customdata[2]} peer(s)<extra></extra>"
            ),
        )
    )
    theme.apply(fig, height=height)
    theme.add_zero_line(fig, axis="x")
    fig.update_xaxes(title="gap to peer median (%)")
    fig.update_layout(margin={"l": 150})
    return fig


# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------
@_never_raises
def volatility_regime_chart(
    series: Sequence[float],
    *,
    current: float = float("nan"),
    median: float = float("nan"),
    height: int = _STANDARD,
) -> go.Figure:
    """Realised volatility through time, with the median and today marked.

    The band between the median line and the current reading is the whole
    message: it shows at once whether today is ordinary for this company or not.
    A raw volatility number means nothing without that comparison, because 30%
    is calm for a small cap and alarming for a utility.
    """
    values = [v for v in series if v is not None and not math.isnan(float(v))]
    if len(values) < _MIN_VOL_SESSIONS:
        return theme.empty_figure(
            f"Fewer than {_MIN_VOL_SESSIONS} sessions of returns were available, and a "
            "volatility measured over a shorter window describes the window rather "
            "than the company.",
            height=height,
        )

    palette = theme.palette()
    fig = go.Figure()
    fig.add_scatter(
        y=[float(v) * 100.0 for v in values],
        mode="lines",
        name="EWMA volatility",
        line={"color": palette["accent"], "width": 2},
        hovertemplate="%{y:.1f}% annualised<extra></extra>",
    )
    if not math.isnan(median):
        theme.add_reference_line(fig, median * 100.0, label="median", axis="y")
    if not math.isnan(current):
        fig.add_scatter(
            x=[len(values) - 1],
            y=[current * 100.0],
            mode="markers",
            name="now",
            marker={"color": palette["warning"], "size": 12, "symbol": "diamond"},
            hovertemplate="now %{y:.1f}%<extra></extra>",
        )
    theme.apply(fig, height=height)
    theme.pct_axis(fig)
    fig.update_xaxes(title="sessions")
    fig.update_yaxes(title="annualised volatility")
    return fig


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
@_never_raises
def scorecard_breakdown_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    composite: float = float("nan"),
    height: int = _TALL,
) -> go.Figure:
    """How far each input pulled the composite from neutral, signed.

    Signed by *influence* rather than by contribution, because a maximally
    adverse input contributes nothing to the total while being the reason the
    verdict reads as it does. Plotting contributions would draw that factor as
    an empty bar and hide the objection.
    """
    usable = [
        row
        for row in rows
        if isinstance(row.get("influence"), (int, float))
        and not math.isnan(float(row["influence"]))  # type: ignore[arg-type]
    ]
    if not usable:
        return theme.empty_figure(
            "No scorecard input could be scored from the data available, so there is "
            "nothing to show pulling on the composite.",
            height=height,
        )

    usable = sorted(usable, key=lambda r: abs(float(r["influence"])), reverse=True)  # type: ignore[arg-type]
    labels = [str(row.get("label", "")) for row in usable]
    values = [
        float(row["influence"]) * (1 if float(row.get("score", 0.5)) >= 0.5 else -1)  # type: ignore[arg-type]
        for row in usable
    ]
    colours = [
        theme.gain() if v >= 0 else theme.loss()  # type: ignore[assignment]
        for v in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            customdata=[
                [row.get("display", ""), float(row.get("effective_weight", 0.0) or 0.0) * 100.0]
                for row in usable
            ],
            hovertemplate=(
                "%{customdata[0]} &middot; weight %{customdata[1]:.1f}%"
                "<br>pull %{x:+.3f}<extra></extra>"
            ),
        )
    )
    theme.apply(fig, height=height)
    theme.add_zero_line(fig, axis="x")
    fig.update_xaxes(title="pull on the composite (points, out of 1)")
    fig.update_layout(margin={"l": 190})
    if not math.isnan(composite):
        theme.caption(
            fig, f"Composite {composite:.3f} on a 0–1 scale, shown above each input's pull."
        )
    return fig


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------
@_never_raises
def sentiment_scatter(
    clusters: Sequence[tuple[object, float, float]],
    *,
    rank_ic: float = float("nan"),
    horizon: int = 20,
    height: int = _STANDARD,
) -> go.Figure:
    """Daily tone against the return that followed it, one point per day.

    The scatter *is* the measurement -- a reader can see the absence of a
    relationship, which a single correlation coefficient would hide. The trend
    line is dashed and faint precisely because it is usually meaningless at this
    sample size.
    """
    points = [(tone, ret) for _, tone, ret in clusters]
    if len(points) < _MIN_SENTIMENT_DAYS:
        return theme.empty_figure(
            f"Fewer than {_MIN_SENTIMENT_DAYS} days of dated coverage were available, which "
            "is too few to set beside returns.",
            height=height,
        )

    palette = theme.palette()
    fig = go.Figure()
    fig.add_scatter(
        x=[p[0] for p in points],
        y=[p[1] * 100.0 for p in points],
        mode="markers",
        marker={"color": palette["accent"], "size": 10, "opacity": 0.75},
        name="one day of coverage",
        hovertemplate="tone %{x:+.2f}<br>next return %{y:+.1f}%<extra></extra>",
    )
    if not math.isnan(rank_ic):
        theme.caption(
            fig,
            f"Rank correlation {rank_ic:+.2f} between tone and the following "
            f"{horizon} sessions. No fitted line is drawn: at this sample size a "
            f"trend line would suggest a relationship the data does not support.",
        )
    theme.apply(fig, height=height)
    theme.add_zero_line(fig)
    theme.pct_axis(fig)
    fig.update_xaxes(title="mean headline tone that day (−1 to +1)")
    fig.update_yaxes(title=f"return over the next {horizon} sessions")
    return fig
