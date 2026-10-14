"""Performance charts: growth, benchmark relatives, attribution and seasonality.

What lives here
---------------
* ``growth_line`` / ``cumulative_return_chart`` - the headline "how has this
  portfolio done" views, in money and in percent respectively.
* ``benchmark_comparison`` / ``relative_strength_chart`` - the same story told
  against an index, with the out/under-performance shaded rather than left for
  the reader to measure with a ruler.
* ``value_bar`` / ``pnl_bar`` / ``return_bar`` - per-holding cross sections.
* ``contribution_waterfall`` - how each name added up to the portfolio return.
* ``monthly_heatmap`` - the calendar-month seasonality grid.

House conventions honoured by every builder
-------------------------------------------
1. **Never raises.** Degenerate input returns ``theme.empty_figure`` naming the
   *specific* reason; an unexpected exception is caught by ``_never_raises``
   and rendered as a placeholder rather than breaking the page around it.
2. **Never mutates.** Inputs are copied and coerced before anything is read.
3. **Hand-written hover.** Every trace carries an explicit ``hovertemplate``
   with formatted numbers and ``<extra></extra>``; default Plotly hover text
   never reaches a user.
4. **Assumptions travel with the figure.** The window and the valuation
   convention go in ``theme.apply(subtitle=...)``; methodology footnotes go in
   ``theme.caption`` so they survive into screenshots and PDF exports.
5. **Colour is semantic.** Red/green is reserved for gain/loss; qualitative
   series use ``theme.CATEGORICAL``; signed quantities always get
   ``theme.sign_colors`` plus ``theme.add_zero_line``.

Percent convention: ``pnl_pct``, ``contribution_pct`` and the monthly return
table all arrive already scaled to PERCENT (12.5 means +12.5%), matching
``src.analytics``. Nothing is re-scaled here.

No Streamlit import - these builders work in the HTML report writer and in a
notebook exactly as they do in the dashboard.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import theme

__all__ = [
    "growth_line",
    "benchmark_comparison",
    "value_bar",
    "pnl_bar",
    "return_bar",
    "contribution_waterfall",
    "monthly_heatmap",
    "relative_strength_chart",
    "cumulative_return_chart",
]

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Fixed ASCII month labels, matching ``analytics.returns.MONTH_LABELS``.
MONTH_LABELS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

#: Name of the compounded annual column in ``returns.monthly_return_table``.
YEAR_COLUMN = "Year"

#: A line needs two points; one observation is a dot, not a series.
MIN_POINTS = 2

#: Index level both a portfolio and its benchmark are rebased to.
REBASE_BASE = 100.0

#: Height envelope for auto-sized figures (callers may always override).
MIN_HEIGHT = 280
MAX_HEIGHT = 460


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _never_raises(fn: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Last-resort net: a chart must never take the page down with it.

    Degenerate input is handled explicitly inside each builder, with a named
    reason. This decorator only catches the genuinely unexpected.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
            return theme.empty_figure(f"Chart could not be built ({type(exc).__name__}: {exc}).")

    return wrapper


def _clean_series(values: Any) -> pd.Series:
    """Copy -> numeric -> drop non-finite -> sort by index. Never mutates."""
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return pd.Series(dtype=float)
        series = values.iloc[:, 0].copy()
    elif isinstance(values, pd.Series):
        series = values.copy()
    else:
        try:
            series = pd.Series(values)
        except (TypeError, ValueError):
            return pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.Series(dtype=float)
    # An unsortable mixed index keeps its insertion order rather than raising.
    with contextlib.suppress(TypeError):
        series = series.sort_index()
    return series.astype(float)


def _rebase(series: pd.Series, base: float = REBASE_BASE) -> pd.Series:
    """Index a series so its first observation equals ``base``.

    Returns an EMPTY series when the first value is non-positive: a ratio to a
    zero or negative base is meaningless rather than merely large.
    """
    if series.empty:
        return pd.Series(dtype=float)
    first = float(series.iloc[0])
    if not math.isfinite(first) or first <= 0.0:
        return pd.Series(dtype=float)
    return series / first * base


def _format_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return str(value)


def _window_label(index: pd.Index) -> str:
    """'03 Jan 2024 to 29 Aug 2025 - 412 observations', for a subtitle."""
    if len(index) == 0:
        return ""
    return f"{_format_date(index[0])} to {_format_date(index[-1])} - {len(index):,} observations"


def _is_dates(index: pd.Index) -> bool:
    return isinstance(index, pd.DatetimeIndex)


def _date_hover(fig: go.Figure, index: pd.Index) -> go.Figure:
    """Readable date in the hover box instead of a raw timestamp."""
    if _is_dates(index):
        fig.update_xaxes(hoverformat="%d %b %Y")
    return fig


def _height(explicit: int | None, default: int) -> int:
    """Respect a caller-supplied height, else use the builder's default."""
    if explicit:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            return default
    return default


def _bar_height(n_bars: int, explicit: int | None, per_bar: int = 26, base: int = 140) -> int:
    """Grow with the number of categories, clamped to the house envelope."""
    if explicit:
        return _height(explicit, MIN_HEIGHT)
    wanted = base + per_bar * max(int(n_bars), 1)
    return int(min(MAX_HEIGHT, max(MIN_HEIGHT, wanted)))


def _crowded(n_bars: int, height: int) -> bool:
    """True when rows are too tight to carry an end-of-bar label legibly.

    Roughly 130px of a bar figure goes to the title block and the value axis;
    below ~18px of remaining space per row a 10.5px label would collide with
    its neighbour, so the number moves to the hover instead of being drawn.
    """
    usable = max(int(height) - 130, 1)
    return usable / max(int(n_bars), 1) < 18.0


def _missing(frame: Any, columns: Sequence[str]) -> list[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return list(columns)
    return [c for c in columns if c not in frame.columns]


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Copy of ``frame`` with ``columns`` coerced to float (NaN when unusable)."""
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = float("nan")
    return out


def _labels(frame: pd.DataFrame, column: str = "ticker") -> list[str]:
    return [str(v) for v in frame[column].tolist()]


def _padded_range(values: np.ndarray, pad_frac: float = 0.18) -> list[float]:
    """Bar-axis range with headroom for outside labels, anchored at zero."""
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return [0.0, 1.0]
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = high - low
    if span <= 0:
        span = abs(high) or 1.0
    pad = span * pad_frac
    return [low - pad if low < 0 else low, high + pad if high > 0 else high]


def _series_range(values: np.ndarray, pad_frac: float = 0.08) -> list[float]:
    """Tight range for a value/level series so the filled area has shape."""
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return [0.0, 1.0]
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    if span <= 0:
        span = abs(high) * 0.05 or 1.0
    pad = span * pad_frac
    lower = low - pad
    if low >= 0.0 and lower < 0.0:
        lower = 0.0
    return [lower, high + pad]


def _signed_pct(value: Any, decimals: int = 1) -> str:
    """'+12.4%' / '-3.1%' / '' for anything not finite."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(num):
        return ""
    return f"{num:+.{decimals}f}%"


def _plain_pct(value: Any, decimals: int = 1) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(num):
        return ""
    return f"{num:.{decimals}f}%"


def _money_unit(currency_symbol: str) -> str:
    """Axis-title suffix such as ' (S$)' - blank when no symbol is supplied."""
    symbol = (currency_symbol or "").strip()
    return f" ({symbol})" if symbol else ""


def _split_fill(
    fig: go.Figure,
    x: Sequence[Any],
    y: np.ndarray,
    base: np.ndarray,
    gain_color: str,
    loss_color: str,
) -> go.Figure:
    """Shade the gap between ``y`` and ``base``: green above, red below.

    Implemented as two invisible baseline/band pairs with ``fill="tonexty"``.
    Plotly has no native two-sided fill, and clipping the series against the
    baseline is the only way to keep the SIGN of the gap legible at a glance.
    All four helper traces skip hover so the unified tooltip stays clean.
    """
    for band, color in ((np.maximum(y, base), gain_color), (np.minimum(y, base), loss_color)):
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=base.tolist(),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=band.tolist(),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=color,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return fig


def _hbar(
    labels: Sequence[str],
    values: np.ndarray,
    colors: list[str],
    hovertemplate: str,
    customdata: list[list[float]] | None = None,
    texts: Sequence[str] | None = None,
    compact: bool = False,
) -> go.Figure:
    """Horizontal bar trace with value labels sitting outside the bar ends.

    Horizontal because ticker and holding names collide when rotated under a
    vertical axis; one label per row can never overlap another. ``compact``
    shrinks the category ticks for a long holdings list.
    """
    bar = go.Bar(
        x=values.tolist(),
        y=list(labels),
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        hovertemplate=hovertemplate,
        showlegend=False,
        cliponaxis=False,
    )
    if customdata is not None:
        bar.customdata = customdata
    if texts is not None:
        bar.text = list(texts)
        bar.textposition = "outside"
        bar.textfont = {"size": 10.5, "color": theme.palette()["subtext"]}
    fig = go.Figure(bar)
    fig.update_yaxes(automargin=True, type="category")
    if compact:
        fig.update_yaxes(tickfont={"size": 9.5})
    return fig


# ---------------------------------------------------------------------------
# growth & cumulative performance
# ---------------------------------------------------------------------------
@_never_raises
def growth_line(
    values: pd.Series,
    currency_symbol: str = "",
    benchmark: pd.Series | None = None,
    benchmark_name: str = "",
    height: int | None = None,
) -> go.Figure:
    """Portfolio value through time, optionally indexed against a benchmark.

    With ``values`` alone the chart shows the portfolio's mark-to-market value
    on every date in the window as a filled line in base currency. When
    ``benchmark`` is supplied BOTH series are cut to their overlapping dates and
    rebased to 100, so the chart answers "which grew faster" instead of
    comparing a money value with an index level; the benchmark is drawn dashed
    in the palette's reserved benchmark grey.

    Assumes ``values`` is a portfolio value series indexed by date, built by
    valuing the CURRENT holdings at historical prices - a look-through of
    today's portfolio, not a record of the trades actually made in the window.
    """
    series = _clean_series(values)
    if series.empty:
        return theme.empty_figure("Portfolio value history is empty - no priced dates to plot.")
    if len(series) < MIN_POINTS:
        return theme.empty_figure(
            "Portfolio value history holds a single observation - "
            "a growth line needs at least two dates."
        )

    p = theme.palette()
    bench_raw = _clean_series(benchmark)
    bench_label = benchmark_name or "Benchmark"
    port_rebased = pd.Series(dtype=float)
    bench_rebased = pd.Series(dtype=float)
    fallback_note = ""

    if not bench_raw.empty:
        common = series.index.intersection(bench_raw.index).sort_values()
        if len(common) >= MIN_POINTS:
            port_rebased = _rebase(series.loc[common])
            bench_rebased = _rebase(bench_raw.loc[common])
            if port_rebased.empty or bench_rebased.empty:
                fallback_note = "benchmark omitted, a series starts at a non-positive value"
        else:
            fallback_note = f"{bench_label} omitted, fewer than two dates overlap the portfolio"

    paired = not port_rebased.empty and not bench_rebased.empty
    fig = go.Figure()

    if paired:
        x = list(port_rebased.index)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=port_rebased.tolist(),
                mode="lines",
                name="Portfolio",
                line={"color": p["accent"], "width": 2.2},
                fill="tozeroy",
                fillcolor=p["accent_fill"],
                hovertemplate="Portfolio  %{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bench_rebased.tolist(),
                mode="lines",
                name=bench_label,
                line={"color": p["benchmark"], "width": 1.8, "dash": "dash"},
                hovertemplate=bench_label + "  %{y:,.2f}<extra></extra>",
            )
        )
        levels = np.concatenate(
            [port_rebased.to_numpy(dtype=float), bench_rebased.to_numpy(dtype=float)]
        )
        theme.apply(
            fig,
            f"Growth of {REBASE_BASE:,.0f} vs {bench_label}",
            subtitle=f"Both rebased to {REBASE_BASE:,.0f} - {_window_label(port_rebased.index)}",
            height=_height(height, 380),
            unified_hover=True,
        )
        fig.update_yaxes(
            title=f"Index level (start = {REBASE_BASE:,.0f})",
            range=_series_range(levels),
            tickformat=",.0f",
        )
        theme.add_reference_line(fig, REBASE_BASE, "Start", color=p["muted"])
    else:
        x = list(series.index)
        y = series.to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y.tolist(),
                mode="lines",
                name="Portfolio value",
                line={"color": p["accent"], "width": 2.2},
                fill="tozeroy",
                fillcolor=p["accent_fill"],
                hovertemplate="Value  " + (currency_symbol or "") + "%{y:,.0f}<extra></extra>",
            )
        )
        subtitle = _window_label(series.index)
        if fallback_note:
            subtitle = f"{subtitle} - {fallback_note}"
        theme.apply(
            fig,
            "Portfolio value over time",
            subtitle=subtitle,
            height=_height(height, 380),
            legend=False,
            unified_hover=True,
        )
        fig.update_yaxes(
            title=f"Portfolio value{_money_unit(currency_symbol)}",
            range=_series_range(y),
        )
        theme.money_axis(fig, currency_symbol or "")
        theme.annotate(
            fig,
            f"{currency_symbol or ''}{float(series.iloc[-1]):,.0f}",
            x=series.index[-1],
            y=float(series.iloc[-1]),
            xanchor="left",
            xshift=8,
            size=11,
            color=p["accent"],
        )
        fig.update_layout(margin={"r": 88})

    fig.update_xaxes(title="Date")
    _date_hover(fig, series.index)
    theme.caption(
        fig,
        "Current holdings valued at historical prices; excludes dividends, "
        "cash flows and any trades made inside the window.",
    )
    return fig


@_never_raises
def cumulative_return_chart(
    values: pd.Series,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "",
    height: int | None = None,
) -> go.Figure:
    """Cumulative total return since the start of the window, in percent.

    Plots ``(V_t / V_0 - 1) * 100`` for the portfolio and, when supplied, for
    the benchmark over the same overlapping dates. Both curves therefore start
    at 0% and the vertical gap on any date IS the cumulative excess return.

    Assumes the first observation is a valid, positive starting value; when it
    is not the chart is refused rather than drawn against a meaningless base.
    """
    series = _clean_series(values)
    if series.empty:
        return theme.empty_figure(
            "Portfolio value history is empty - no dates to accumulate a return over."
        )
    if len(series) < MIN_POINTS:
        return theme.empty_figure(
            "Portfolio value history holds a single observation - "
            "cumulative return needs a start and an end."
        )

    bench_raw = _clean_series(benchmark)
    bench_label = benchmark_name or "Benchmark"
    if not bench_raw.empty:
        common = series.index.intersection(bench_raw.index).sort_values()
        if len(common) >= MIN_POINTS:
            series = series.loc[common]
            bench_raw = bench_raw.loc[common]
        else:
            bench_raw = pd.Series(dtype=float)

    port = _rebase(series)
    if port.empty:
        return theme.empty_figure(
            "Starting portfolio value is not positive - cumulative return is undefined."
        )
    port_pct = port - REBASE_BASE
    bench_pct = _rebase(bench_raw)
    if not bench_pct.empty:
        bench_pct = bench_pct - REBASE_BASE

    p = theme.palette()
    end = float(port_pct.iloc[-1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(port_pct.index),
            y=port_pct.tolist(),
            mode="lines",
            name="Portfolio",
            line={"color": p["accent"], "width": 2.2},
            fill="tozeroy",
            fillcolor=p["gain_fill"] if end >= 0 else p["loss_fill"],
            hovertemplate="Portfolio  %{y:+.2f}%<extra></extra>",
        )
    )
    if not bench_pct.empty:
        fig.add_trace(
            go.Scatter(
                x=list(bench_pct.index),
                y=bench_pct.tolist(),
                mode="lines",
                name=bench_label,
                line={"color": p["benchmark"], "width": 1.8, "dash": "dash"},
                hovertemplate=bench_label + "  %{y:+.2f}%<extra></extra>",
            )
        )
        excess = end - float(bench_pct.iloc[-1])
        subtitle = (
            f"{_window_label(port_pct.index)} - portfolio {end:+.2f}%, excess {excess:+.2f} pp"
        )
    else:
        subtitle = f"{_window_label(port_pct.index)} - portfolio {end:+.2f}%"

    theme.apply(
        fig,
        "Cumulative return",
        subtitle=subtitle,
        height=_height(height, 360),
        legend=not bench_pct.empty,
        unified_hover=True,
    )
    fig.update_yaxes(title="Cumulative return (%)")
    fig.update_xaxes(title="Date")
    theme.pct_axis(fig)
    theme.add_zero_line(fig)
    _date_hover(fig, port_pct.index)
    return fig


@_never_raises
def benchmark_comparison(
    growth: pd.DataFrame,
    benchmark_name: str = "Benchmark",
    height: int | None = None,
) -> go.Figure:
    """Portfolio against its benchmark, both indexed to 100, with the gap shaded.

    Takes the ``growth`` frame from ``analytics.benchmark.compare`` - columns
    ``Portfolio`` and ``Benchmark`` on a shared date index. Both columns are
    (re)based to 100 at the first date where both exist, which is a no-op when
    the caller already normalised them. The area between the curves is shaded
    green where the portfolio leads and red where it lags, so relative
    performance is read by area rather than measured with a ruler.

    Assumes index levels are price-only: an index that excludes dividends is a
    slightly conservative bar for a portfolio measured the same way.
    """
    if not isinstance(growth, pd.DataFrame) or growth.empty:
        return theme.empty_figure(
            "Benchmark growth table is empty - no overlapping history to compare."
        )

    frame = growth.copy()
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    port_col = lookup.get("portfolio")
    bench_col = lookup.get("benchmark") or lookup.get(str(benchmark_name).strip().lower())
    if port_col is None or bench_col is None:
        available = ", ".join(str(c) for c in frame.columns) or "none"
        return theme.empty_figure(
            "Benchmark comparison needs a 'Portfolio' and a 'Benchmark' column - "
            f"found: {available}."
        )

    pair = frame.loc[:, [port_col, bench_col]].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    with contextlib.suppress(TypeError):
        pair = pair.sort_index()
    if len(pair) < MIN_POINTS:
        return theme.empty_figure(
            "Fewer than two dates carry both a portfolio and a benchmark level - "
            "nothing to compare."
        )

    port = _rebase(pair[port_col])
    bench = _rebase(pair[bench_col])
    if port.empty or bench.empty:
        return theme.empty_figure(
            "A series starts at a non-positive level, so it cannot be rebased to 100."
        )

    p = theme.palette()
    label = str(benchmark_name) or "Benchmark"
    x = list(port.index)
    y_port = port.to_numpy(dtype=float)
    y_bench = bench.to_numpy(dtype=float)

    fig = go.Figure()
    _split_fill(fig, x, y_port, y_bench, p["gain_fill"], p["loss_fill"])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_bench.tolist(),
            mode="lines",
            name=label,
            line={"color": p["benchmark"], "width": 1.8, "dash": "dash"},
            hovertemplate=label + "  %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_port.tolist(),
            mode="lines",
            name="Portfolio",
            line={"color": p["accent"], "width": 2.2},
            hovertemplate="Portfolio  %{y:,.2f}<extra></extra>",
        )
    )

    spread = float(y_port[-1] - y_bench[-1])
    theme.apply(
        fig,
        f"Portfolio vs {label}",
        subtitle=(
            f"Both rebased to {REBASE_BASE:,.0f} on "
            f"{_format_date(port.index[0])} - final spread {spread:+,.1f} pts"
        ),
        height=_height(height, 400),
        unified_hover=True,
    )
    fig.update_yaxes(
        title=f"Index level (start = {REBASE_BASE:,.0f})",
        range=_series_range(np.concatenate([y_port, y_bench])),
        tickformat=",.0f",
    )
    fig.update_xaxes(title="Date")
    theme.add_reference_line(fig, REBASE_BASE, "Start", color=p["muted"])
    _date_hover(fig, port.index)
    theme.caption(
        fig,
        "Shaded area is the cumulative lead (green) or lag (red) versus the "
        "index. Index levels are price-only and exclude dividends.",
    )
    return fig


@_never_raises
def relative_strength_chart(
    rs: pd.Series,
    benchmark_name: str = "Benchmark",
    height: int | None = None,
) -> go.Figure:
    """Portfolio-to-benchmark ratio rebased to 100 - a purely relative view.

    Takes the series from ``analytics.benchmark.relative_strength``. A RISING
    line means the portfolio is beating the index regardless of which way the
    market went; the level alone is meaningless, so the chart is anchored on a
    labelled parity line at 100 with the region above shaded green and the
    region below red.

    Assumes ``rs`` is already the rebased ratio (start = 100). It is rebased
    defensively here, so a raw portfolio/benchmark ratio renders identically.
    """
    series = _clean_series(rs)
    if series.empty:
        return theme.empty_figure(
            "Relative strength series is empty - the portfolio and the benchmark "
            "share no priced dates."
        )
    if len(series) < MIN_POINTS:
        return theme.empty_figure(
            "Relative strength series holds a single observation - at least two "
            "overlapping dates are needed."
        )

    rebased = _rebase(series)
    if rebased.empty:
        return theme.empty_figure(
            "Relative strength starts at a non-positive level and cannot be rebased to 100."
        )

    p = theme.palette()
    label = str(benchmark_name) or "Benchmark"
    x = list(rebased.index)
    y = rebased.to_numpy(dtype=float)
    base = np.full_like(y, REBASE_BASE)

    fig = go.Figure()
    _split_fill(fig, x, y, base, p["gain_fill"], p["loss_fill"])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y.tolist(),
            mode="lines",
            name="Relative strength",
            line={"color": p["accent"], "width": 2.2},
            hovertemplate="Relative strength  %{y:,.2f}<extra></extra>",
        )
    )

    end = float(y[-1])
    verdict = "ahead of" if end >= REBASE_BASE else "behind"
    theme.apply(
        fig,
        f"Relative strength vs {label}",
        subtitle=(
            f"{_window_label(rebased.index)} - currently "
            f"{abs(end - REBASE_BASE):,.1f} pts {verdict} the index"
        ),
        height=_height(height, 340),
        legend=False,
        unified_hover=True,
    )
    fig.update_yaxes(
        title=f"Portfolio / {label} (start = {REBASE_BASE:,.0f})",
        range=_series_range(np.concatenate([y, base])),
        tickformat=",.0f",
    )
    fig.update_xaxes(title="Date")
    theme.add_reference_line(fig, REBASE_BASE, "Parity", color=p["muted"])
    _date_hover(fig, rebased.index)
    theme.caption(
        fig,
        "Ratio of portfolio value to index level, rebased to 100 at the start. "
        "A rising line is outperformance, not necessarily a positive return.",
    )
    return fig


# ---------------------------------------------------------------------------
# per-holding cross sections
# ---------------------------------------------------------------------------
@_never_raises
def value_bar(
    positions: pd.DataFrame,
    currency_symbol: str = "",
    top_n: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Current market value of each holding, largest at the top.

    Value is unsigned, so the bars use the neutral accent colour - green and
    red stay reserved for gain/loss. Every bar is labelled with its portfolio
    weight, so the chart carries the absolute and the relative reading at once.

    Assumes ``positions`` is the frame from
    ``analytics.positions.build_positions``, with money columns already
    converted into the portfolio's base currency.
    """
    missing = _missing(positions, ("ticker", "current_value"))
    if missing:
        return theme.empty_figure(
            "Holdings table is empty or missing the "
            f"{', '.join(repr(m) for m in missing)} column(s) needed for values."
        )

    frame = _numeric_columns(positions, ["current_value", "weight_pct"])
    frame = frame.loc[np.isfinite(frame["current_value"].to_numpy(dtype=float))]
    frame = frame.loc[frame["current_value"] > 0]
    if frame.empty:
        return theme.empty_figure(
            "No holding has a positive current value - nothing to size the bars by."
        )

    n_total = len(frame)
    frame = frame.sort_values("current_value", ascending=True)
    if top_n:
        frame = frame.tail(int(top_n))

    total = float(frame["current_value"].sum())
    weights = frame["weight_pct"].to_numpy(dtype=float)
    if not np.isfinite(weights).any() and total > 0:
        weights = frame["current_value"].to_numpy(dtype=float) / total * 100.0

    values = frame["current_value"].to_numpy(dtype=float)
    p = theme.palette()
    bar_h = _bar_height(len(frame), height)
    dense = _crowded(len(frame), bar_h)
    fig = _hbar(
        labels=_labels(frame),
        values=values,
        colors=[p["accent"]] * len(frame),
        hovertemplate=(
            "<b>%{y}</b><br>Value  " + (currency_symbol or "") + "%{x:,.0f}"
            "<br>Weight  %{customdata[0]:.2f}%<extra></extra>"
        ),
        customdata=[[float(w)] for w in weights],
        texts=None if dense else [_plain_pct(w) for w in weights],
        compact=dense,
    )

    scope = (
        f"top {len(frame)} of {n_total} holdings" if len(frame) < n_total else f"{n_total} holdings"
    )
    theme.apply(
        fig,
        "Value by holding",
        subtitle=(
            f"Latest available prices - {scope}, {(currency_symbol or '')}{total:,.0f} shown"
        ),
        height=bar_h,
        legend=False,
    )
    fig.update_xaxes(
        title=f"Current value{_money_unit(currency_symbol)}",
        range=_padded_range(values),
    )
    fig.update_yaxes(title="Holding")
    theme.money_axis(fig, currency_symbol or "", axis="x")
    return fig


@_never_raises
def pnl_bar(
    positions: pd.DataFrame,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Unrealised profit and loss per holding in base currency, sign-coloured.

    Sorted by P&L so the largest winners sit at the top and the largest losers
    at the bottom, with the zero line emphasised because the sign is the whole
    point. Bar labels carry each name's percentage return, which stops a big
    position with a small move from reading as a great decision.

    Assumes ``pnl`` is unrealised, mark-to-market against average cost, and
    already in the portfolio's base currency.
    """
    missing = _missing(positions, ("ticker", "pnl"))
    if missing:
        return theme.empty_figure(
            "Holdings table is empty or missing the "
            f"{', '.join(repr(m) for m in missing)} column(s) needed for P&L."
        )

    frame = _numeric_columns(positions, ["pnl", "pnl_pct"])
    frame = frame.loc[np.isfinite(frame["pnl"].to_numpy(dtype=float))]
    if frame.empty:
        return theme.empty_figure(
            "No holding has a computable profit or loss - every P&L value is missing."
        )

    frame = frame.sort_values("pnl", ascending=True)
    values = frame["pnl"].to_numpy(dtype=float)
    pcts = frame["pnl_pct"].to_numpy(dtype=float)
    net = float(values.sum())
    winners = int((values > 0).sum())
    bar_h = _bar_height(len(frame), height)
    dense = _crowded(len(frame), bar_h)

    fig = _hbar(
        labels=_labels(frame),
        values=values,
        colors=theme.sign_colors(values),
        hovertemplate=(
            "<b>%{y}</b><br>P&L  " + (currency_symbol or "") + "%{x:,.0f}"
            "<br>Return  %{customdata[0]:+.2f}%<extra></extra>"
        ),
        customdata=[[float(v)] for v in pcts],
        texts=None if dense else [_signed_pct(v) for v in pcts],
        compact=dense,
    )

    theme.apply(
        fig,
        "Unrealised profit & loss by holding",
        subtitle=(
            f"{winners} of {len(frame)} names in profit - net {(currency_symbol or '')}{net:,.0f}"
        ),
        height=bar_h,
        legend=False,
    )
    fig.update_xaxes(
        title=f"Unrealised P&L{_money_unit(currency_symbol)}",
        range=_padded_range(values),
    )
    fig.update_yaxes(title="Holding")
    theme.money_axis(fig, currency_symbol or "", axis="x")
    theme.add_zero_line(fig, axis="x")
    theme.caption(
        fig,
        "Mark-to-market against average cost; excludes dividends, fees and "
        "realised trades. Hover for each holding's percentage return.",
    )
    return fig


@_never_raises
def return_bar(positions: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Percentage return per holding since purchase, sorted and sign-coloured.

    The percentage view deliberately ignores position size: it ranks the
    QUALITY of each decision where ``pnl_bar`` ranks its impact. Read the two
    together - a 60% gain on a 1% weight moves nothing.

    Assumes ``pnl_pct`` arrives already in percent units and is FX-neutral
    (both legs converted at the same rate), per ``analytics.positions``.
    """
    missing = _missing(positions, ("ticker", "pnl_pct"))
    if missing:
        return theme.empty_figure(
            "Holdings table is empty or missing the "
            f"{', '.join(repr(m) for m in missing)} column(s) needed for returns."
        )

    frame = _numeric_columns(positions, ["pnl_pct", "weight_pct"])
    frame = frame.loc[np.isfinite(frame["pnl_pct"].to_numpy(dtype=float))]
    if frame.empty:
        return theme.empty_figure(
            "No holding has a computable return - every buy price is missing or zero."
        )

    frame = frame.sort_values("pnl_pct", ascending=True)
    values = frame["pnl_pct"].to_numpy(dtype=float)
    weights = frame["weight_pct"].to_numpy(dtype=float)
    best = float(values.max())
    worst = float(values.min())
    bar_h = _bar_height(len(frame), height)
    dense = _crowded(len(frame), bar_h)

    fig = _hbar(
        labels=_labels(frame),
        values=values,
        colors=theme.sign_colors(values),
        hovertemplate=(
            "<b>%{y}</b><br>Return  %{x:+.2f}%<br>Weight  %{customdata[0]:.2f}%<extra></extra>"
        ),
        customdata=[[float(w)] for w in weights],
        texts=None if dense else [_signed_pct(v) for v in values],
        compact=dense,
    )

    theme.apply(
        fig,
        "Return by holding",
        subtitle=f"Unrealised, since purchase - best {best:+.1f}%, worst {worst:+.1f}%",
        height=bar_h,
        legend=False,
    )
    fig.update_xaxes(title="Return since purchase (%)", range=_padded_range(values))
    fig.update_yaxes(title="Holding")
    theme.pct_axis(fig, axis="x")
    theme.add_zero_line(fig, axis="x")
    return fig


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
@_never_raises
def contribution_waterfall(
    contribution: pd.DataFrame,
    currency_symbol: str = "",
    max_bars: int = 14,
    height: int | None = None,
) -> go.Figure:
    """How each holding built the portfolio's total return, as a waterfall.

    Takes the frame from ``analytics.attribution.return_contribution`` with
    columns ``ticker, weight_pct, pnl, pnl_pct, contribution_pct``. Each step is
    one holding's P&L expressed as a share of TOTAL invested capital, which is
    why the bars can be added at all: percentage returns per holding do not
    sum, contributions do - and the closing "Total" bar is the portfolio return.

    Beyond ``max_bars`` names the smallest absolute contributors are collapsed
    into one honest "Other (n names)" step rather than being dropped.
    """
    missing = _missing(contribution, ("ticker", "contribution_pct"))
    if missing:
        return theme.empty_figure(
            "Contribution table is empty or missing the "
            f"{', '.join(repr(m) for m in missing)} column(s)."
        )

    frame = _numeric_columns(contribution, ["contribution_pct", "pnl", "weight_pct"])
    frame = frame.loc[np.isfinite(frame["contribution_pct"].to_numpy(dtype=float))]
    if frame.empty:
        return theme.empty_figure(
            "No holding has a computable contribution - invested capital is zero "
            "or every P&L is missing."
        )

    frame = frame.sort_values("contribution_pct", ascending=False)
    has_pnl = bool(np.isfinite(frame["pnl"].to_numpy(dtype=float)).any())
    n_holdings = len(frame)

    labels = _labels(frame)
    contrib = frame["contribution_pct"].to_numpy(dtype=float)
    pnl = frame["pnl"].to_numpy(dtype=float)
    weight = frame["weight_pct"].to_numpy(dtype=float)

    limit = max(int(max_bars), 3)
    if n_holdings > limit:
        keep = np.zeros(n_holdings, dtype=bool)
        keep[np.argsort(-np.abs(contrib))[: limit - 1]] = True
        rest = ~keep
        order = np.argsort(-contrib[keep])
        kept_labels = [str(v) for v in np.asarray(labels, dtype=object)[keep][order]]
        labels = [*kept_labels, f"Other ({int(rest.sum())} names)"]
        contrib = np.append(contrib[keep][order], float(np.nansum(contrib[rest])))
        pnl = np.append(pnl[keep][order], float(np.nansum(pnl[rest])))
        weight = np.append(weight[keep][order], float(np.nansum(weight[rest])))

    total_contrib = float(np.nansum(contrib))
    total_pnl = float(np.nansum(pnl))
    total_weight = float(np.nansum(weight))

    p = theme.palette()
    x_labels = [*labels, "Total"]
    measures = ["relative"] * len(labels) + ["total"]
    # A "total" step draws the running sum, so its own y is ignored by Plotly;
    # the true figure travels in customdata where the hover can reach it.
    y_values = np.append(contrib, 0.0)
    custom = [
        [float(c), float(v), float(w)]
        for c, v, w in zip(
            np.append(contrib, total_contrib),
            np.append(pnl, total_pnl),
            np.append(weight, total_weight),
        )
    ]

    hover_lines = ["<b>%{x}</b>", "Contribution  %{customdata[0]:+.2f} pp"]
    if has_pnl:
        hover_lines.append("P&L  " + (currency_symbol or "") + "%{customdata[1]:,.0f}")
    hover_lines.append("Weight  %{customdata[2]:.2f}%")

    # Past ~18 steps an outside label per bar starts colliding with the next
    # one, so the numbers retreat into the hover rather than overprinting.
    labelled = len(x_labels) <= 18
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values.tolist(),
            customdata=custom,
            text=(
                [_signed_pct(v, 2) for v in contrib] + [f"{total_contrib:+.2f}%"]
                if labelled
                else None
            ),
            textposition="outside" if labelled else "none",
            textfont={"size": 10.5, "color": p["subtext"]},
            cliponaxis=False,
            increasing={"marker": {"color": p["gain"]}},
            decreasing={"marker": {"color": p["loss"]}},
            totals={"marker": {"color": p["accent"]}},
            connector={"line": {"color": p["muted"], "width": 1, "dash": "dot"}},
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            showlegend=False,
        )
    )

    theme.apply(
        fig,
        "Return contribution by holding",
        subtitle=(f"{n_holdings} holdings summing to {total_contrib:+.2f}% of invested capital"),
        height=_height(height, 420),
        legend=False,
    )
    running = np.cumsum(np.nan_to_num(contrib))
    fig.update_yaxes(
        title="Contribution to total return (pp)",
        range=_padded_range(np.append(running, [0.0, total_contrib]), 0.22),
    )
    fig.update_xaxes(
        title="Holding",
        type="category",
        tickangle=0 if len(x_labels) <= 6 else (-35 if labelled else -55),
        tickfont={"size": 11 if labelled else 9},
    )
    theme.pct_axis(fig, decimals=1)
    theme.add_zero_line(fig)
    theme.caption(
        fig,
        "Contribution = holding P&L / total invested capital, so the steps sum "
        "to the portfolio return. Unrealised and excludes dividends.",
    )
    return fig


# ---------------------------------------------------------------------------
# seasonality
# ---------------------------------------------------------------------------
def _year_labels(index: pd.Index) -> list[str]:
    """Row labels as plain strings so the axis stays categorical, not numeric."""
    out: list[str] = []
    for value in index:
        try:
            out.append(str(int(value)))
        except (TypeError, ValueError):
            out.append(str(value))
    return out


def _cell_text_color(value: float, scale_max: float) -> str:
    """Dark ink on pale cells, light ink on saturated ones.

    The cell colour comes from the fixed DIVERGING ramp rather than from the
    app theme, so the label colour has to be chosen against the RAMP - hence
    the light-mode text colour on pale cells and the dark-mode one on deep
    cells, both still taken from the design system.
    """
    if not math.isfinite(value) or scale_max <= 0:
        return theme.LIGHT_PALETTE["text"]
    return (
        theme.DARK_PALETTE["text"]
        if abs(value) >= 0.55 * scale_max
        else theme.LIGHT_PALETTE["text"]
    )


@_never_raises
def monthly_heatmap(table: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Calendar-month returns as a year x month grid with an annual column.

    Takes the frame from ``analytics.returns.monthly_return_table``: index =
    year, columns ``Jan..Dec`` plus ``Year``, values already in PERCENT and NaN
    for months that have not happened. Colour is the blue/red DIVERGING ramp
    centred on zero - deliberately not green/red, which is reserved for
    gain/loss and fails for red-green colour-vision deficiency - and every cell
    is annotated with its own number, so the grid doubles as a table.

    The ``Year`` column sits in a separate panel on its own colour scale:
    compounded annual returns span a far wider range than single months, and
    forcing both onto one scale would wash the monthly grid out. Blank cells
    are months with no data, never zeroes.
    """
    if not isinstance(table, pd.DataFrame) or table.empty:
        return theme.empty_figure(
            "Monthly return table is empty - the window contains no completed calendar month."
        )

    frame = table.copy()
    months = [m for m in MONTH_LABELS if m in frame.columns]
    if not months:
        return theme.empty_figure("Monthly return table has no month columns (expected Jan..Dec).")

    month_frame = frame.loc[:, months].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    z_months = month_frame.to_numpy(dtype=float)
    if not np.isfinite(z_months).any():
        return theme.empty_figure(
            "Every month in the return table is blank - no calendar month has "
            "completed inside this window."
        )
    z_months = np.where(np.isfinite(z_months), z_months, np.nan)

    years = _year_labels(frame.index)
    n_years = len(years)
    month_max = float(np.nanmax(np.abs(z_months))) or 1.0

    if YEAR_COLUMN in frame.columns:
        year_values = pd.to_numeric(frame[YEAR_COLUMN], errors="coerce").to_numpy(dtype=float)
        year_values = np.where(np.isfinite(year_values), year_values, np.nan)
        finite = year_values[np.isfinite(year_values)]
        year_max = (float(np.max(np.abs(finite))) if finite.size else 0.0) or month_max
    else:
        year_values = np.full(n_years, np.nan)
        year_max = month_max

    p = theme.palette()
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.885, 0.115],
        horizontal_spacing=0.055,
        shared_yaxes=False,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_months.tolist(),
            x=list(months),
            y=years,
            coloraxis="coloraxis",
            xgap=3,
            ygap=3,
            hoverongaps=False,
            hovertemplate="<b>%{x} %{y}</b><br>Return  %{z:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[[v] for v in year_values.tolist()],
            x=[YEAR_COLUMN],
            y=years,
            coloraxis="coloraxis2",
            xgap=3,
            ygap=3,
            hoverongaps=False,
            hovertemplate="<b>%{y} full year</b><br>Return  %{z:+.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    font_size = 10 if n_years <= 8 else 9
    for r, year in enumerate(years):
        for c, month in enumerate(months):
            value = float(z_months[r, c])
            if not math.isfinite(value):
                continue
            fig.add_annotation(
                x=month,
                y=year,
                xref="x",
                yref="y",
                text=f"{value:.1f}",
                showarrow=False,
                font={"size": font_size, "color": _cell_text_color(value, month_max)},
            )
        annual = float(year_values[r])
        if math.isfinite(annual):
            fig.add_annotation(
                x=YEAR_COLUMN,
                y=year,
                xref="x2",
                yref="y2",
                text=f"{annual:.1f}",
                showarrow=False,
                font={"size": font_size, "color": _cell_text_color(annual, year_max)},
            )

    covered = int(np.isfinite(z_months).sum())
    theme.apply(
        fig,
        "Monthly returns",
        subtitle=(
            f"{years[0]} to {years[-1]} - {covered} completed months, blank where "
            "a month has not finished"
        ),
        height=_height(height, int(min(MAX_HEIGHT, max(MIN_HEIGHT, 150 + 46 * n_years)))),
        legend=False,
        margin={"r": 96},
    )
    fig.update_layout(
        coloraxis={
            "colorscale": theme.DIVERGING,
            "cmin": -month_max,
            "cmid": 0.0,
            "cmax": month_max,
            "colorbar": {
                "title": {"text": "Month", "font": {"size": 11}},
                "ticksuffix": "%",
                "thickness": 12,
                "len": 0.8,
                "x": 1.02,
                "outlinewidth": 0,
            },
        },
        coloraxis2={
            "colorscale": theme.DIVERGING,
            "cmin": -year_max,
            "cmid": 0.0,
            "cmax": year_max,
            "showscale": False,
        },
    )
    fig.update_xaxes(showgrid=False, ticks="", type="category")
    fig.update_yaxes(showgrid=False, ticks="", type="category")
    fig.update_yaxes(title="Year", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(title="Month", row=1, col=1)
    fig.update_xaxes(tickfont={"color": p["text"], "size": 11}, row=1, col=2)
    theme.caption(
        fig,
        "Month cells share one colour scale; the annual column is scaled "
        "separately because compounded years span a much wider range.",
    )
    return fig
