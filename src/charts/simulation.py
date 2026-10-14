"""Monte Carlo charts: percentile fan, terminal distribution, sample paths.

These three figures are the visual half of ``src.analytics.montecarlo``. They
consume exactly what the simulator emits - the percentile frame, the terminal
value vector and the raw ``(n_paths, horizon + 1)`` path matrix - and add no
statistics of their own beyond the cross-sectional median used as an overlay.

Reading conventions this module commits to:

* Day 0 of every simulation is the CURRENT portfolio value, so the fan starts
  as a point and widens; the dashed "starting value" line is the break-even
  reference every number on the chart is quoted against.
* A percentile band is a band of *outcomes at a point in time*, not a path any
  single portfolio would follow. That caveat is printed on the figure rather
  than left to the reader, because reading the P95 edge as "a good scenario"
  is the most common misuse of a fan chart.
* Gain/loss colour is reserved for its semantic meaning: finishing above or
  below the starting value. Percentile bands are drawn in the neutral accent
  so no band is implicitly labelled "good" or "bad".

Nothing here imports Streamlit; every function returns a ``go.Figure`` and
never raises, so a degenerate simulation degrades to a titled placeholder
instead of breaking the page.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = ["fan_chart", "terminal_distribution", "sample_paths_chart"]

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Prefix of the last-resort placeholder message. Any figure carrying it is a
#: bug, not a data condition - the tests assert it never appears.
_FALLBACK_PREFIX = "Chart unavailable: "

#: Opacity range of the nested confidence bands, outermost -> innermost.
_BAND_ALPHA_MIN = 0.09
_BAND_ALPHA_MAX = 0.26

#: Default figure heights (px).
_FAN_HEIGHT = 420
_HIST_HEIGHT = 380
_PATHS_HEIGHT = 400

#: Hard ceiling on drawn spaghetti lines - beyond this the plot is mud and the
#: browser starts to feel it.
_MAX_PATHS_DRAWN = 300


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _never_raises(func: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Guarantee the chart contract: a Figure comes back, always.

    Each function below validates its own inputs and returns a *specific*
    :func:`theme.empty_figure` for every anticipated degenerate case. This
    decorator only catches the unanticipated, so a single malformed frame can
    never take a dashboard page down with it.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
            return theme.empty_figure(
                f"{_FALLBACK_PREFIX}{type(exc).__name__} while drawing this figure."
            )

    return wrapper


def _rgba(color: str, alpha: float) -> str:
    """Re-express a theme colour at a given opacity.

    Keeps every tint derived from ``theme.palette()`` rather than hardcoded,
    which is what allows the bands to re-theme with the rest of the app.
    """
    text = str(color).strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) >= 6:
            red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
    if text.startswith("rgb"):
        inner = text[text.find("(") + 1 : text.rfind(")")]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 3:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha:.3f})"
    return text


def _is_finite(value: Any) -> bool:
    """True only for a real, finite number (None/NaN/str -> False)."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _footnote(fig: go.Figure, text: str) -> None:
    """``theme.caption`` nudged clear of the x-axis title.

    The theme pins a caption 0.16 of the plot height below the axis, which is
    exactly where an axis title also sits. Every figure here titles its x
    axis, so the note is shifted down and the bottom margin opened to match -
    a caption that lands on top of the axis title is worse than no caption.
    """
    theme.caption(fig, text)
    notes = fig.layout.annotations or ()
    if notes:
        notes[-1].update(yshift=-26)
    fig.update_layout(margin={"b": 104})


def _value_axis_title(currency_symbol: str) -> str:
    """Axis title carrying the unit, per the house rule that every axis names one."""
    symbol = (currency_symbol or "").strip()
    return f"Portfolio value ({symbol})" if symbol else "Portfolio value"


def _percentile_levels(frame: pd.DataFrame) -> list[tuple[float, Any]]:
    """Parse ``p5``/``P95``/``5`` style column labels into ``(level, column)``.

    Sorted ascending by level so the caller can pair the outermost columns
    inwards without assuming a particular column order in the input.
    """
    levels: list[tuple[float, Any]] = []
    for column in frame.columns:
        label = str(column).strip().lower()
        candidate = label[1:] if label.startswith("p") else label
        try:
            level = float(candidate)
        except ValueError:
            continue
        if 0.0 <= level <= 100.0:
            levels.append((level, column))
    levels.sort(key=lambda item: item[0])
    return levels


def _pct_customdata(values: np.ndarray, start_value: float) -> np.ndarray | None:
    """Per-point percent change versus the starting value, shaped for hover."""
    if not _is_finite(start_value) or float(start_value) == 0.0:
        return None
    return np.column_stack([(values / float(start_value) - 1.0) * 100.0])


def _hover(label: str, currency_symbol: str, with_pct: bool) -> str:
    """Hand-written hover row: label, money value, and move versus start."""
    symbol = currency_symbol or ""
    body = f"{label}  {symbol}%{{y:,.0f}}"
    if with_pct:
        body += "  (%{customdata[0]:+.1f}% vs start)"
    return body + "<extra></extra>"


def _level_label(level: float) -> str:
    """``5.0 -> 'P5'`` - terminal shorthand that survives a narrow legend."""
    return f"P{level:g}"


def _step_axis_title(index: pd.Index) -> str:
    """Name the x unit honestly: calendar dates if given, else trading steps."""
    return "Date" if isinstance(index, pd.DatetimeIndex) else "Trading days ahead"


# ---------------------------------------------------------------------------
# fan chart
# ---------------------------------------------------------------------------
@_never_raises
def fan_chart(
    percentile_paths: pd.DataFrame,
    start_value: float,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Nested confidence bands of simulated portfolio value through time.

    Shows the cross-sectional percentiles of the simulated distribution at
    each step: the outermost filled band spans the extreme percentile pair
    (P5-P95 for the default configuration), each inner band is drawn at a
    higher opacity, and the median is a solid line. A dashed horizontal rule
    marks the starting value, so the whole chart reads as "where could this
    portfolio be, relative to today".

    Parameters
    ----------
    percentile_paths:
        Frame indexed by step (0 = today) whose columns are percentile labels
        (``p5``, ``p25``, ``p50``, ``p75``, ``p95``). Column order does not
        matter; any ``p<number>`` set is paired outermost-inwards.
    start_value:
        Portfolio value at step 0, used for the reference line and for the
        "vs start" figure in every hover row. Non-finite values suppress both.
    currency_symbol:
        Prefix for the money axis and hover text (``""`` for a bare number).
    height:
        Caller override; defaults to 420 px.

    Assumes the percentiles were computed across paths at each step, from a
    single simulation run with a fixed horizon. It therefore says nothing
    about the *sequence* of returns any one portfolio experiences - a point
    made in the figure's own caption.
    """
    if not isinstance(percentile_paths, pd.DataFrame) or percentile_paths.empty:
        return theme.empty_figure("The simulation returned no percentile paths to plot.")

    frame = percentile_paths.copy()
    levels = _percentile_levels(frame)
    if not levels:
        return theme.empty_figure(
            "No percentile columns found - expected labels like p5, p50, p95."
        )

    frame = frame[[column for _, column in levels]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(how="all")
    if frame.empty or frame.shape[1] == 0:
        return theme.empty_figure("Every simulated percentile path is missing or non-numeric.")
    if len(frame) < 2:
        return theme.empty_figure(
            "The simulation horizon is a single step - a fan needs at least two."
        )

    levels = [(level, column) for level, column in levels if column in frame.columns]
    median_at = min(range(len(levels)), key=lambda i: abs(levels[i][0] - 50.0))
    median_level, median_column = levels[median_at]
    outer = [item for index, item in enumerate(levels) if index != median_at]

    bands: list[tuple[tuple[float, Any], tuple[float, Any]]] = []
    low, high = 0, len(outer) - 1
    while low < high:
        bands.append((outer[low], outer[high]))
        low += 1
        high -= 1

    palette = theme.palette()
    band_color = palette["accent"]
    x_values = frame.index
    show_pct = _is_finite(start_value) and float(start_value) != 0.0
    fig = go.Figure()

    # Outermost band first so the tighter, more opaque bands paint on top.
    for position, ((lo_level, lo_column), (hi_level, hi_column)) in enumerate(bands):
        alpha = _BAND_ALPHA_MIN
        if len(bands) > 1:
            step = (_BAND_ALPHA_MAX - _BAND_ALPHA_MIN) / (len(bands) - 1)
            alpha = _BAND_ALPHA_MIN + step * position
        elif len(bands) == 1:
            alpha = (_BAND_ALPHA_MIN + _BAND_ALPHA_MAX) / 2.0

        lower = frame[lo_column].to_numpy(dtype=float)
        upper = frame[hi_column].to_numpy(dtype=float)
        band_name = f"{_level_label(lo_level)}-{_level_label(hi_level)} band"

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower,
                mode="lines",
                name=band_name,
                legendgroup=band_name,
                showlegend=False,
                line={"width": 0.8, "color": _rgba(band_color, 0.45)},
                customdata=_pct_customdata(lower, start_value),
                hovertemplate=_hover(_level_label(lo_level), currency_symbol, show_pct),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper,
                mode="lines",
                name=band_name,
                legendgroup=band_name,
                showlegend=True,
                fill="tonexty",
                fillcolor=_rgba(band_color, alpha),
                line={"width": 0.8, "color": _rgba(band_color, 0.45)},
                customdata=_pct_customdata(upper, start_value),
                hovertemplate=_hover(_level_label(hi_level), currency_symbol, show_pct),
            )
        )

    median = frame[median_column].to_numpy(dtype=float)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=median,
            mode="lines",
            name=f"Median ({_level_label(median_level)})",
            line={"width": 2.4, "color": band_color},
            customdata=_pct_customdata(median, start_value),
            hovertemplate=_hover("Median", currency_symbol, show_pct),
        )
    )

    if _is_finite(start_value):
        theme.add_reference_line(
            fig,
            float(start_value),
            "starting value",
            axis="y",
            color=palette["muted"],
            dash="dash",
        )

    steps = len(frame) - 1
    if bands:
        widest = f"{_level_label(bands[0][0][0])}-{_level_label(bands[0][1][0])}"
        tightest = f"{_level_label(bands[-1][0][0])}-{_level_label(bands[-1][1][0])}"
        band_note = (
            f"outer band {widest}, inner band {tightest}" if len(bands) > 1 else f"band {widest}"
        )
    else:
        band_note = "median path only"
    subtitle = f"{steps:,} steps ahead - {band_note}"

    theme.apply(
        fig,
        "Simulated portfolio value",
        subtitle=subtitle,
        height=height or _FAN_HEIGHT,
        unified_hover=True,
    )
    fig.update_xaxes(title_text=_step_axis_title(frame.index))
    fig.update_yaxes(title_text=_value_axis_title(currency_symbol))
    theme.money_axis(fig, currency_symbol, axis="y")
    _footnote(
        fig,
        "Each band is a percentile of the simulated distribution at that step, "
        "not a path a portfolio would follow. Simulated outcomes, not a forecast.",
    )
    return fig


# ---------------------------------------------------------------------------
# terminal distribution
# ---------------------------------------------------------------------------
def _histogram_bins(sample: np.ndarray) -> int:
    """Square-root rule, clamped so the shape reads at any sample size."""
    return int(np.clip(round(float(np.sqrt(sample.size))), 20, 60))


@_never_raises
def terminal_distribution(
    terminal: np.ndarray,
    start_value: float,
    currency_symbol: str = "",
    height: int | None = None,
    bins: int | None = None,
) -> go.Figure:
    """Distribution of simulated end-of-horizon portfolio values.

    Bars are shaded gain/loss by whether the bucket finishes above or below
    the starting value, the loss region is additionally shaded so the
    downside reads at a glance, and two vertical rules mark the starting
    value and the median outcome. The y axis is the share of simulated paths
    landing in each bucket, which keeps the chart comparable across runs with
    different path counts.

    Parameters
    ----------
    terminal:
        1-D array of terminal values, one per simulated path (the
        ``terminal`` key of ``montecarlo.summarise_paths``).
    start_value:
        Break-even reference; the probability of loss printed in the subtitle
        is the share of paths strictly below it.
    currency_symbol:
        Prefix for the money axis and hover text.
    height:
        Caller override; defaults to 380 px.
    bins:
        Bin count override; defaults to a clamped square-root rule.

    Assumes every path shares one horizon and one starting value, so the
    spread is attributable to the simulated return process alone.
    """
    try:
        values = np.asarray(terminal, dtype=float).ravel()
    except (TypeError, ValueError):
        return theme.empty_figure("Terminal values could not be read as numbers.")

    if values.size == 0:
        return theme.empty_figure("The simulation produced no terminal values.")
    sample = values[np.isfinite(values)]
    if sample.size == 0:
        return theme.empty_figure("Every simulated terminal value is missing or infinite.")

    palette = theme.palette()
    n_bins = int(bins) if bins and int(bins) > 0 else _histogram_bins(sample)
    counts, edges = np.histogram(sample, bins=n_bins)
    if counts.sum() <= 0:
        return theme.empty_figure("Terminal values could not be binned into a histogram.")

    centres = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    share = counts / float(sample.size) * 100.0
    has_start = _is_finite(start_value)
    start = float(start_value) if has_start else float("nan")

    colors = (
        [palette["gain"] if centre >= start else palette["loss"] for centre in centres]
        if has_start
        else [palette["accent"]] * centres.size
    )
    symbol = currency_symbol or ""
    customdata = np.column_stack([edges[:-1], edges[1:], counts.astype(float)])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centres,
            y=share,
            width=widths,
            marker={"color": colors, "line": {"width": 0}},
            opacity=0.88,
            name="Simulated paths",
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                f"{symbol}%{{customdata[0]:,.0f}} to {symbol}%{{customdata[1]:,.0f}}"
                "<br>%{y:.1f}% of paths (%{customdata[2]:,.0f})"
                "<extra></extra>"
            ),
        )
    )

    prob_loss = float("nan")
    if has_start:
        prob_loss = float((sample < start).mean() * 100.0)
        # Shade the whole loss region, clipped to the plotted range.
        loss_right = min(start, float(edges[-1]))
        if loss_right > float(edges[0]):
            fig.add_vrect(
                x0=float(edges[0]),
                x1=loss_right,
                fillcolor=_rgba(palette["loss"], 0.10),
                line_width=0,
                layer="below",
            )
        theme.add_reference_line(
            fig,
            start,
            "starting value",
            axis="x",
            color=palette["muted"],
            dash="dash",
        )

    median = float(np.median(sample))
    # theme.add_reference_line pins x-axis labels to "top"; the median rule
    # gets its own anchor so the two labels can never collide.
    fig.add_vline(
        x=median,
        line_width=1.4,
        line_dash="solid",
        line_color=palette["accent"],
        annotation_text="median",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=palette["accent"],
    )

    loss_note = f"{prob_loss:.1f}% finish below the starting value" if has_start else ""
    subtitle = f"{sample.size:,} simulated paths"
    if loss_note:
        subtitle = f"{subtitle} - {loss_note}"

    theme.apply(
        fig,
        "Distribution of terminal values",
        subtitle=subtitle,
        height=height or _HIST_HEIGHT,
        legend=False,
        bargap=0.02,
    )
    fig.update_xaxes(title_text=f"Terminal {_value_axis_title(currency_symbol).lower()}")
    fig.update_yaxes(title_text="Share of simulated paths (%)")
    theme.money_axis(fig, currency_symbol, axis="x")
    theme.pct_axis(fig, axis="y", decimals=0)
    _footnote(
        fig,
        "Buckets are shaded by outcome against the starting value; bar height is "
        "the share of paths, so runs with different path counts stay comparable.",
    )
    return fig


# ---------------------------------------------------------------------------
# sample paths
# ---------------------------------------------------------------------------
def _sample_indices(n_rows: int, n_show: int) -> np.ndarray:
    """Evenly spaced, deterministic row picks - no RNG, so screenshots repeat."""
    take = int(np.clip(n_show, 1, min(n_rows, _MAX_PATHS_DRAWN)))
    if take >= n_rows:
        return np.arange(n_rows)
    return np.unique(np.linspace(0, n_rows - 1, take).round().astype(int))


@_never_raises
def sample_paths_chart(
    paths: np.ndarray,
    n_show: int = 100,
    start_value: float | None = None,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """A readable sample of individual simulated trajectories.

    Draws up to ``n_show`` thin translucent paths - coloured by whether they
    finish above or below the starting value - with the median of ALL
    simulated paths overlaid as a solid line. The fan chart shows the shape of
    the distribution; this one shows what a single future actually looks like,
    which is the honest antidote to reading a smooth percentile band as a
    forecast.

    Parameters
    ----------
    paths:
        ``(n_paths, horizon + 1)`` matrix; column 0 is the starting value.
    n_show:
        Maximum number of trajectories drawn. Rows are picked at even
        intervals (never randomly), so the same input always yields the same
        picture. Capped at 300 - beyond that the plot is unreadable anyway.
    start_value:
        Break-even reference. Defaults to the median of column 0.
    currency_symbol:
        Prefix for the money axis and hover text.
    height:
        Caller override; defaults to 400 px.

    Hover is per-path ("closest") rather than the house x-unified mode: a
    unified tooltip over a hundred overlapping series is a hundred-row wall of
    numbers, which is worse than no tooltip at all.
    """
    try:
        matrix = np.asarray(paths, dtype=float)
    except (TypeError, ValueError):
        return theme.empty_figure("Simulated paths could not be read as numbers.")

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.size == 0:
        return theme.empty_figure("Simulated paths are not a (paths x steps) matrix.")
    if matrix.shape[1] < 2:
        return theme.empty_figure(
            "Each simulated path holds a single point - nothing to trace through time."
        )
    if not np.isfinite(matrix).any():
        return theme.empty_figure("Simulated paths contain no finite values.")

    palette = theme.palette()
    symbol = currency_symbol or ""
    steps = np.arange(matrix.shape[1])

    start = float(start_value) if _is_finite(start_value) else float("nan")
    if not np.isfinite(start):
        first_column = matrix[:, 0]
        if np.isfinite(first_column).any():
            start = float(np.nanmedian(first_column))

    picks = _sample_indices(matrix.shape[0], n_show)
    fig = go.Figure()
    seen_groups: set[str] = set()

    for row in picks:
        series = matrix[row]
        if not np.isfinite(series).any():
            continue
        finals = series[np.isfinite(series)]
        ends_up = np.isfinite(start) and float(finals[-1]) >= start
        group = "up" if ends_up else "down"
        color = palette["gain"] if ends_up else palette["loss"]
        if not np.isfinite(start):
            group, color = "path", palette["accent"]
        label = {
            "up": "Ends at or above start",
            "down": "Ends below start",
            "path": "Simulated path",
        }[group]
        first_of_group = group not in seen_groups
        seen_groups.add(group)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=series,
                mode="lines",
                name=label,
                legendgroup=group,
                showlegend=first_of_group,
                line={"width": 1.0, "color": _rgba(color, 0.30)},
                hovertemplate=(
                    f"Path {int(row) + 1}<br>day %{{x:,.0f}}: {symbol}%{{y:,.0f}}<extra></extra>"
                ),
            )
        )

    median_path = np.nanmedian(matrix, axis=0)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=median_path,
            mode="lines",
            name="Median of all paths",
            line={"width": 2.4, "color": palette["accent"]},
            hovertemplate=(
                f"Median of all paths<br>day %{{x:,.0f}}: {symbol}%{{y:,.0f}}<extra></extra>"
            ),
        )
    )

    if np.isfinite(start):
        theme.add_reference_line(
            fig, start, "starting value", axis="y", color=palette["muted"], dash="dash"
        )

    drawn = len(picks)
    subtitle = f"{drawn:,} of {matrix.shape[0]:,} simulated paths"
    if np.isfinite(start):
        finals = matrix[:, -1]
        finite_finals = finals[np.isfinite(finals)]
        if finite_finals.size:
            up_share = float((finite_finals >= start).mean() * 100.0)
            subtitle = f"{subtitle} - {up_share:.0f}% of all paths finish at or above start"

    theme.apply(
        fig,
        "Individual simulated paths",
        subtitle=subtitle,
        height=height or _PATHS_HEIGHT,
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Trading days ahead")
    fig.update_yaxes(title_text=_value_axis_title(currency_symbol))
    theme.money_axis(fig, currency_symbol, axis="y")
    _footnote(
        fig,
        "Paths are an evenly spaced sample of the run, not the extremes; the solid "
        "line is the median across every simulated path, which no single path follows.",
    )
    return fig
