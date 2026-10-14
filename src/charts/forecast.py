"""Forecast charts: the cone, the backtest, the leaderboard and the residuals.

These six figures are the visual half of :mod:`src.analytics.forecast`, and
between them they are designed to make one thing hard to avoid noticing:
**whether the model is any good**. A forecasting page that plots only the
forward cone is a page that always looks confident. So the cone is one figure
of six, and the other five are all about error.

Reading conventions this module commits to:

* **The forward cone starts at today's actual close**, so it opens from a point
  rather than from a band. Its width is the conformal prediction interval --
  the model's own out-of-sample residual quantiles -- which means a model with
  no skill draws a very wide cone and says so visually before anyone reads a
  number.
* **A node is a fit; the line between nodes is drawn.** Every horizon is a
  separate model, so the markers are real and the segments joining them are
  interpolation. They are drawn differently for exactly that reason.
* **Zero is the benchmark on the leaderboard**, not the left edge. A bar to the
  left of the zero line is a model that lost to a random walk, which is the
  normal outcome and must be legible as such.
* **Gain/loss colour keeps its semantic meaning** -- above or below today's
  price, right or wrong about direction. The cone bands use the neutral accent
  so no part of the distribution is implicitly labelled good.

Nothing here imports Streamlit; every function returns a ``go.Figure`` and
never raises, so a failed model degrades to a titled placeholder rather than
breaking the page.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = [
    "backtest_scatter",
    "feature_importance_chart",
    "forecast_cone_chart",
    "horizon_skill_chart",
    "model_leaderboard_chart",
    "prediction_timeline_chart",
    "residual_distribution_chart",
]

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Prefix of the last-resort placeholder. Any figure carrying it is a bug.
_FALLBACK_PREFIX = "Chart unavailable: "

#: Default figure heights (px).
_CONE_HEIGHT = 480
_SCATTER_HEIGHT = 420
_BOARD_HEIGHT = 360
_IMPORTANCE_HEIGHT = 420
_RESIDUAL_HEIGHT = 360
_SKILL_HEIGHT = 340
_TIMELINE_HEIGHT = 380

#: Trailing sessions of realised price drawn behind the cone. A year of
#: context is enough to read the forecast against; more compresses the cone
#: into the right-hand margin where its width stops being visible.
_HISTORY_SESSIONS: int = 252

#: Opacity of the conformal band fill.
_BAND_ALPHA: float = 0.16

#: Features listed on the importance chart before it becomes unreadable.
_MAX_IMPORTANCE_ROWS: int = 15

#: Histogram bins for the residual distribution.
_RESIDUAL_BINS: int = 40

#: Family -> categorical colour index, so a family keeps one colour across
#: every chart on the page.
_FAMILY_ORDER: tuple[str, ...] = (
    "returns",
    "momentum",
    "volatility",
    "trend",
    "oscillator",
    "range",
    "volume",
    "calendar",
    "other",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _never_raises(func: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Guarantee the chart contract: a Figure comes back, always.

    Each function validates its own inputs and returns a *specific* placeholder
    for every anticipated degenerate case. This decorator catches only the
    unanticipated, so one malformed frame cannot take the page down.
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
    """Re-express a theme colour at a given opacity, so tints stay themed."""
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
    """True only for a real, finite number."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _family_color(family: str) -> str:
    """Stable colour per feature family, from the colourblind-safe ramp."""
    try:
        position = _FAMILY_ORDER.index(str(family))
    except ValueError:
        position = len(_FAMILY_ORDER) - 1
    return theme.CATEGORICAL[position % len(theme.CATEGORICAL)]


def _footnote(fig: go.Figure, text: str) -> None:
    """``theme.caption`` nudged clear of an axis title."""
    theme.caption(fig, text)
    notes = fig.layout.annotations or ()
    if notes:
        notes[-1].update(yshift=-26)
    fig.update_layout(margin={"b": 104})


def _price_axis_title(currency_symbol: str) -> str:
    symbol = (currency_symbol or "").strip()
    return f"Price ({symbol})" if symbol else "Price"


def _numeric_series(frame: Any, column: str) -> pd.Series | None:
    """One numeric column of a frame, or ``None`` when it is not usable."""
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    return series if series.notna().any() else None


def _optional_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Like :func:`_numeric_series`, but an absent column becomes all-NaN.

    For hover payloads, where a missing column and an unmeasurable value are
    the same fact and both must render as an em dash rather than crash the
    column stack.
    """
    series = _numeric_series(frame, column)
    if series is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return series


# ---------------------------------------------------------------------------
# 1. the forward cone
# ---------------------------------------------------------------------------
@_never_raises
def forecast_cone_chart(
    history: Any,
    path: pd.DataFrame,
    nodes: pd.DataFrame | None = None,
    currency_symbol: str = "",
    confidence: float = 0.80,
    height: int | None = None,
    history_sessions: int = _HISTORY_SESSIONS,
) -> go.Figure:
    """Realised price, then the forecast path inside its conformal interval.

    Parameters
    ----------
    history:
        Close Series (or a frame carrying ``Close``) of the realised price. The
        last ``history_sessions`` observations are drawn.
    path:
        Frame from ``forecast.forecast_path`` with ``point``, ``lower``,
        ``upper`` and ``is_node`` columns on a forward index.
    nodes:
        The per-horizon frame, used only to mark which points were fitted.
    confidence:
        Nominal coverage of the band, for the legend and the caption.

    The band is the model's out-of-sample residual quantile range carried
    forward, so it is **marginal** coverage: it holds on average over the
    backtest and says nothing about whether today is a calm or a violent day.
    That is stated in the figure's own caption rather than left to the reader.
    """
    if not isinstance(path, pd.DataFrame) or path.empty:
        return theme.empty_figure("No forecast path was produced for this instrument.")
    point = _numeric_series(path, "point")
    if point is None:
        return theme.empty_figure("The forecast path carries no usable point forecast.")

    lower = _numeric_series(path, "lower")
    upper = _numeric_series(path, "upper")
    palette = theme.palette()
    symbol = currency_symbol or ""
    fig = go.Figure()

    close = _history_close(history)
    if close is not None and len(close) >= 2:
        window = close.iloc[-max(int(history_sessions), 2) :]
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window.to_numpy(dtype="float64"),
                mode="lines",
                name="Realised price",
                line={"width": 1.9, "color": palette["text"]},
                hovertemplate=f"Realised  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )

    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=path.index,
                y=lower.to_numpy(dtype="float64"),
                mode="lines",
                name=f"{confidence * 100.0:.0f}% interval",
                legendgroup="band",
                showlegend=False,
                line={"width": 0.9, "color": _rgba(palette["accent"], 0.45)},
                hovertemplate=f"Lower  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path.index,
                y=upper.to_numpy(dtype="float64"),
                mode="lines",
                name=f"{confidence * 100.0:.0f}% prediction interval",
                legendgroup="band",
                showlegend=True,
                fill="tonexty",
                fillcolor=_rgba(palette["accent"], _BAND_ALPHA),
                line={"width": 0.9, "color": _rgba(palette["accent"], 0.45)},
                hovertemplate=f"Upper  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=path.index,
            y=point.to_numpy(dtype="float64"),
            mode="lines",
            name="Forecast (interpolated)",
            line={"width": 2.4, "color": palette["accent"], "dash": "dot"},
            hovertemplate=f"Forecast  {symbol}%{{y:,.2f}}<extra></extra>",
        )
    )

    marked = path[path.get("is_node", pd.Series(False, index=path.index)).astype(bool)]
    if not marked.empty:
        fig.add_trace(
            go.Scatter(
                x=marked.index,
                y=pd.to_numeric(marked["point"], errors="coerce").to_numpy(dtype="float64"),
                mode="markers",
                name="Fitted horizon",
                marker={
                    "size": 9,
                    "color": palette["accent"],
                    "line": {"width": 1.4, "color": palette["hover_bg"]},
                    "symbol": "circle",
                },
                customdata=_node_customdata(marked, nodes),
                hovertemplate=(
                    f"%{{customdata[0]}} sessions ahead<br>Forecast  {symbol}%{{y:,.2f}}"
                    "<br>Skill vs random walk %{customdata[1]}"
                    "<br>P(higher) %{customdata[2]}<extra></extra>"
                ),
            )
        )

    start = float(point.iloc[0]) if _is_finite(point.iloc[0]) else float("nan")
    if _is_finite(start):
        theme.add_reference_line(
            fig, start, "today's close", axis="y", color=palette["muted"], dash="dash"
        )

    theme.apply(
        fig,
        title="Forecast cone",
        subtitle=(
            f"{confidence * 100.0:.0f}% conformal prediction interval from the model's own "
            f"out-of-sample residuals"
        ),
        height=height or _CONE_HEIGHT,
        unified_hover=True,
        xaxis_title="Date",
        yaxis_title=_price_axis_title(currency_symbol),
    )
    theme.money_axis(fig, currency_symbol)
    _footnote(
        fig,
        "Each marker is a separately fitted model for that horizon; the dotted line between "
        "them is interpolation, not a prediction of the route. The band is marginal coverage — "
        "it holds on average across the backtest, not conditionally on today.",
    )
    return fig


def _node_customdata(marked: pd.DataFrame, nodes: pd.DataFrame | None) -> np.ndarray:
    """Hover payload for the fitted-horizon markers: sessions, skill, P(up).

    The skill of the model behind each marker belongs on the marker itself.
    Without it a reader hovers a point on a confident-looking cone and learns
    only what the model claims, never how often it has been right. Missing
    values render as an em dash rather than a fabricated number.
    """
    steps = marked.get("sessions_ahead", pd.Series(0, index=marked.index)).astype(int)
    lookup: dict[int, tuple[str, str]] = {}
    if isinstance(nodes, pd.DataFrame) and "horizon" in nodes.columns:
        for row in nodes.itertuples():
            skill = getattr(row, "skill", float("nan"))
            probability = getattr(row, "prob_up", float("nan"))
            lookup[int(row.horizon)] = (
                f"{float(skill) * 100.0:+.2f}%" if _is_finite(skill) else "—",
                f"{float(probability) * 100.0:.0f}%" if _is_finite(probability) else "—",
            )
    pairs = [lookup.get(int(step), ("—", "—")) for step in steps]
    return np.column_stack(
        [steps.to_numpy(dtype="int64"), [p[0] for p in pairs], [p[1] for p in pairs]]
    )


def _history_close(history: Any) -> pd.Series | None:
    """Close series from a Series, a frame, or nothing usable."""
    if history is None:
        return None
    try:
        if isinstance(history, pd.DataFrame):
            if "Close" in history.columns:
                series = history["Close"]
            elif history.shape[1] == 1:
                series = history.iloc[:, 0]
            else:
                return None
        elif isinstance(history, pd.Series):
            series = history
        else:
            return None
    except (TypeError, ValueError):
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return numeric if len(numeric) else None


# ---------------------------------------------------------------------------
# 2. predicted vs realised
# ---------------------------------------------------------------------------
@_never_raises
def backtest_scatter(
    predictions: pd.DataFrame,
    height: int | None = None,
) -> go.Figure:
    """Every out-of-sample prediction against what actually happened.

    The 45-degree line is a perfect forecast. The two shaded quadrants are the
    ones where the SIGN was right; a model with directional skill puts more of
    its mass there, and a model without puts an even cloud across all four --
    which is the picture most price forecasters actually produce.

    A near-vertical cloud hugging zero on the x axis is the other diagnostic
    worth knowing: it means the model barely varies its prediction at all, and
    its apparent accuracy is the accuracy of predicting the mean.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return theme.empty_figure("The walk-forward produced no out-of-sample predictions.")
    truth = _numeric_series(predictions, "y_true")
    predicted = _numeric_series(predictions, "y_pred")
    if truth is None or predicted is None:
        return theme.empty_figure("The prediction frame is missing y_true or y_pred.")

    frame = pd.DataFrame({"y_true": truth, "y_pred": predicted}).dropna()
    if frame.empty:
        return theme.empty_figure("Every out-of-sample prediction is missing.")

    palette = theme.palette()
    x = frame["y_pred"].to_numpy(dtype="float64") * 100.0
    y = frame["y_true"].to_numpy(dtype="float64") * 100.0
    correct = np.sign(x) == np.sign(y)
    colors = np.where(correct, palette["gain"], palette["loss"])

    limit = float(max(np.abs(np.concatenate([x, y])).max(), 1e-6)) * 1.08
    fig = go.Figure()

    # Shade the two "sign was right" quadrants before anything is plotted.
    for x0, y0 in ((0.0, 0.0), (-limit, -limit)):
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x0 + limit,
            y1=y0 + limit,
            fillcolor=_rgba(palette["gain"], 0.05),
            line={"width": 0},
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Out-of-sample prediction",
            marker={"size": 5.5, "color": colors, "opacity": 0.62},
            customdata=np.column_stack([np.where(correct, "direction right", "direction wrong")]),
            hovertemplate=(
                "Predicted %{x:+.2f}%<br>Realised %{y:+.2f}%<br>%{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-limit, limit],
            y=[-limit, limit],
            mode="lines",
            name="Perfect forecast",
            line={"width": 1.2, "color": palette["muted"], "dash": "dash"},
            hoverinfo="skip",
        )
    )

    theme.apply(
        fig,
        title="Predicted against realised",
        subtitle="Every held-out observation from the purged walk-forward",
        height=height or _SCATTER_HEIGHT,
        xaxis_title="Predicted return over the horizon (%)",
        yaxis_title="Realised return over the horizon (%)",
    )
    theme.add_zero_line(fig, "y")
    theme.add_zero_line(fig, "x")
    fig.update_xaxes(range=[-limit, limit], ticksuffix="%")
    fig.update_yaxes(range=[-limit, limit], ticksuffix="%")
    _footnote(
        fig,
        "Shaded quadrants are where the sign was right. Watch the horizontal spread: a cloud "
        "squeezed near x = 0 means the model is predicting the average and little else.",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. leaderboard
# ---------------------------------------------------------------------------
@_never_raises
def model_leaderboard_chart(
    board: pd.DataFrame,
    height: int | None = None,
) -> go.Figure:
    """Out-of-sample skill by model, with the random walk pinned at zero.

    Skill is ``1 - MAE_model / MAE_random_walk``, so the zero line is not the
    axis origin by accident -- it is the benchmark. Bars to its left are models
    that would have been better off predicting no change at all, and colouring
    them as losses is the point of the figure.
    """
    if not isinstance(board, pd.DataFrame) or board.empty:
        return theme.empty_figure("No model could be scored on this history.")
    skill = _numeric_series(board, "skill")
    if skill is None or "label" not in board.columns:
        return theme.empty_figure("The leaderboard carries no skill column to plot.")

    frame = board.copy()
    frame["skill_pct"] = skill * 100.0
    frame = frame[frame["skill_pct"].notna()].sort_values("skill_pct")
    if frame.empty:
        return theme.empty_figure("Every model failed to produce a skill estimate.")

    palette = theme.palette()
    colors = [
        palette["gain"] if value > 0 else (palette["muted"] if value == 0 else palette["loss"])
        for value in frame["skill_pct"]
    ]
    # An all-NaN column and an absent one are the same thing to a hover row, so
    # both collapse to a NaN series rather than to ``None``.
    directional = _optional_series(frame, "directional")
    p_values = _optional_series(frame, "dm_p")

    fig = go.Figure(
        go.Bar(
            x=frame["skill_pct"].to_numpy(dtype="float64"),
            y=frame["label"].astype(str),
            orientation="h",
            marker={"color": colors},
            name="Skill",
            customdata=np.column_stack(
                [
                    (directional * 100.0).to_numpy(dtype="float64"),
                    p_values.to_numpy(dtype="float64"),
                ]
            ),
            hovertemplate=(
                "%{y}<br>Skill %{x:+.2f}% vs random walk"
                "<br>Direction %{customdata[0]:.0f}%"
                "<br>Diebold-Mariano p %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )

    theme.apply(
        fig,
        title="Model leaderboard",
        subtitle="Accuracy gain over a random walk, measured on identical held-out blocks",
        height=height or _BOARD_HEIGHT,
        legend=False,
        xaxis_title="Skill: 1 − MAE / MAE of a random walk (%)",
        yaxis_title="",
    )
    theme.add_zero_line(fig, "x")
    fig.update_xaxes(ticksuffix="%")
    _footnote(
        fig,
        "Zero is a random walk — predicting no change at all. A negative bar is a model that "
        "was worse than doing nothing, which is the ordinary result at these horizons.",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. feature importance
# ---------------------------------------------------------------------------
@_never_raises
def feature_importance_chart(
    importance: pd.DataFrame,
    top_n: int = _MAX_IMPORTANCE_ROWS,
    height: int | None = None,
) -> go.Figure:
    """Permutation importance of the top features, coloured by family.

    Importance here is the increase in held-out mean absolute error when a
    single feature's values are shuffled. Measured out of sample deliberately:
    an in-sample importance ranks whatever the model overfitted on.

    A near-zero (or negative) importance is not a rendering failure. It means
    shuffling the feature did not hurt -- the model was not using it -- and on
    a genuinely unforecastable series the whole chart looks like that.
    """
    if not isinstance(importance, pd.DataFrame) or importance.empty:
        return theme.empty_figure("Permutation importance could not be computed for this model.")
    values = _numeric_series(importance, "importance")
    if values is None or "feature" not in importance.columns:
        return theme.empty_figure("The importance frame carries no usable values.")

    frame = importance.copy()
    frame["importance"] = values
    frame = frame.dropna(subset=["importance"]).head(max(int(top_n), 1))
    if frame.empty:
        return theme.empty_figure("No feature produced a measurable importance.")
    frame = frame.iloc[::-1]  # largest at the top of a horizontal bar chart

    families = (
        frame["family"].astype(str)
        if "family" in frame.columns
        else pd.Series("other", index=frame.index)
    )
    descriptions = (
        frame["description"].astype(str)
        if "description" in frame.columns
        else pd.Series("", index=frame.index)
    )
    errors = _numeric_series(frame, "std")

    fig = go.Figure(
        go.Bar(
            x=frame["importance"].to_numpy(dtype="float64") * 10_000.0,
            y=frame["feature"].astype(str),
            orientation="h",
            marker={"color": [_family_color(f) for f in families]},
            error_x=(
                {
                    "type": "data",
                    "array": errors.to_numpy(dtype="float64") * 10_000.0,
                    "thickness": 1.0,
                    "width": 3,
                    "color": theme.palette()["muted"],
                }
                if errors is not None
                else None
            ),
            customdata=np.column_stack([families, descriptions]),
            hovertemplate=(
                "%{y}  (%{customdata[0]})<br>+%{x:.2f} bp of error when shuffled"
                "<br>%{customdata[1]}<extra></extra>"
            ),
            name="Importance",
        )
    )

    theme.apply(
        fig,
        title="What the model actually used",
        subtitle="Permutation importance on held-out rows, in basis points of added error",
        height=height or _IMPORTANCE_HEIGHT,
        legend=False,
        xaxis_title="Increase in held-out mean absolute error (bp)",
        yaxis_title="",
    )
    theme.add_zero_line(fig, "x")
    _footnote(
        fig,
        "A bar near zero means shuffling that feature cost the model nothing — it was not "
        "being used. Error bars are the spread across repeats of the shuffle.",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. residual distribution
# ---------------------------------------------------------------------------
@_never_raises
def residual_distribution_chart(
    residuals: Any,
    alpha: float = 0.20,
    height: int | None = None,
) -> go.Figure:
    """Out-of-sample errors, with the conformal quantiles marked.

    This distribution IS the prediction interval: the band drawn on the cone is
    the two dashed lines here, shifted onto the point forecast. Its width is
    therefore the honest statement of what the model does not know, and its
    skew is why the band is asymmetric.
    """
    sample = np.asarray(residuals, dtype="float64").ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size < 8:
        return theme.empty_figure(
            "Fewer than eight out-of-sample residuals — too few to draw a distribution."
        )

    palette = theme.palette()
    percent = sample * 100.0
    level = float(alpha) if 0.0 < float(alpha) < 1.0 else 0.20
    low = float(np.quantile(percent, level / 2.0))
    high = float(np.quantile(percent, 1.0 - level / 2.0))
    median = float(np.median(percent))

    fig = go.Figure(
        go.Histogram(
            x=percent,
            nbinsx=_RESIDUAL_BINS,
            marker={"color": _rgba(palette["accent"], 0.55), "line": {"width": 0}},
            name="Out-of-sample error",
            hovertemplate="Error %{x:+.2f}%<br>%{y} observations<extra></extra>",
        )
    )
    theme.add_reference_line(
        fig, low, f"P{level / 2 * 100:g}", axis="x", color=palette["loss"], dash="dash"
    )
    theme.add_reference_line(
        fig,
        high,
        f"P{(1 - level / 2) * 100:g}",
        axis="x",
        color=palette["gain"],
        dash="dash",
    )
    theme.add_reference_line(
        fig, median, "median error", axis="x", color=palette["warning"], dash="dot"
    )

    theme.apply(
        fig,
        title="Where the model was wrong",
        subtitle=f"Realised minus predicted, on {sample.size:,} held-out observations",
        height=height or _RESIDUAL_HEIGHT,
        legend=False,
        xaxis_title="Realised − predicted return (%)",
        yaxis_title="Observations",
    )
    fig.update_xaxes(ticksuffix="%")
    _footnote(
        fig,
        "The two dashed lines are the conformal interval, shifted onto the point forecast to "
        "draw the cone. A median away from zero is the model's measured bias.",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. skill by horizon
# ---------------------------------------------------------------------------
@_never_raises
def horizon_skill_chart(
    nodes: pd.DataFrame,
    height: int | None = None,
) -> go.Figure:
    """Skill and directional accuracy against forecast horizon.

    Two axes because the units are unrelated: skill is a percentage gain over a
    random walk (zero is the benchmark) and directional accuracy is a hit rate
    (fifty is the benchmark). Both reference levels are drawn, on their own
    axis, rather than one line asked to mean two things.
    """
    if not isinstance(nodes, pd.DataFrame) or nodes.empty or "horizon" not in nodes.columns:
        return theme.empty_figure("No per-horizon results were produced.")
    skill = _numeric_series(nodes, "skill")
    if skill is None:
        return theme.empty_figure("The per-horizon frame carries no skill estimate.")

    frame = nodes.copy().sort_values("horizon")
    palette = theme.palette()
    horizons = frame["horizon"].astype(int).astype(str) + "d"
    values = (skill * 100.0).to_numpy(dtype="float64")
    colors = [palette["gain"] if v > 0 else palette["loss"] for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=horizons,
            y=values,
            name="Skill vs random walk",
            marker={"color": colors},
            hovertemplate="%{x} horizon<br>Skill %{y:+.2f}%<extra></extra>",
        )
    )

    directional = _numeric_series(frame, "directional")
    if directional is not None:
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=(directional * 100.0).to_numpy(dtype="float64"),
                name="Directional accuracy",
                yaxis="y2",
                mode="lines+markers",
                line={"width": 2.0, "color": palette["accent_2"]},
                marker={"size": 7},
                hovertemplate="%{x} horizon<br>Direction %{y:.1f}%<extra></extra>",
            )
        )

    theme.apply(
        fig,
        title="Does the edge survive the horizon?",
        subtitle="One independently fitted and validated model per horizon",
        height=height or _SKILL_HEIGHT,
        xaxis_title="Forecast horizon (trading sessions)",
        yaxis_title="Skill vs random walk (%)",
        yaxis2={
            "title": {"text": "Directional accuracy (%)", "font": {"size": 12}},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, 100],
            "ticksuffix": "%",
        },
    )
    theme.add_zero_line(fig, "y")
    # Only the primary axis: y2 sets its own suffix in the layout above, and
    # ``update_yaxes`` would otherwise overwrite the 0-100 range with it.
    fig.update_layout(yaxis={"ticksuffix": "%"})
    _footnote(
        fig,
        "Skill is read against zero on the left axis; directional accuracy against 50% on the "
        "right. A model can be right about direction while losing money on magnitude.",
    )
    return fig


# ---------------------------------------------------------------------------
# 7. prediction timeline
# ---------------------------------------------------------------------------
@_never_raises
def prediction_timeline_chart(
    predictions: pd.DataFrame,
    height: int | None = None,
) -> go.Figure:
    """Predicted and realised forward returns through the held-out period.

    The figure that exposes a model which worked until a regime change and
    stopped -- an average skill number cannot show that, and a scatter plot
    throws the dates away. Fold boundaries are marked so a break that lines up
    with a refit is visible as such.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return theme.empty_figure("The walk-forward produced no out-of-sample predictions.")
    truth = _numeric_series(predictions, "y_true")
    predicted = _numeric_series(predictions, "y_pred")
    if truth is None or predicted is None:
        return theme.empty_figure("The prediction frame is missing y_true or y_pred.")

    palette = theme.palette()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=truth.to_numpy(dtype="float64") * 100.0,
            mode="lines",
            name="Realised",
            line={"width": 1.6, "color": palette["text"]},
            hovertemplate="Realised %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predicted.to_numpy(dtype="float64") * 100.0,
            mode="lines",
            name="Predicted",
            line={"width": 2.0, "color": palette["accent"]},
            hovertemplate="Predicted %{y:+.2f}%<extra></extra>",
        )
    )

    if "fold" in predictions.columns:
        folds = pd.to_numeric(predictions["fold"], errors="coerce")
        changes = folds.ne(folds.shift()).fillna(False)
        for position in predictions.index[changes][1:]:
            fig.add_vline(
                x=position,
                line_width=1.0,
                line_dash="dot",
                line_color=palette["muted"],
                opacity=0.7,
            )

    theme.apply(
        fig,
        title="Forecast against outcome, through time",
        subtitle="Held-out blocks only — the model never saw these rows when it was fitted",
        height=height or _TIMELINE_HEIGHT,
        unified_hover=True,
        xaxis_title="Date the forecast was made",
        yaxis_title="Return over the horizon (%)",
    )
    theme.add_zero_line(fig, "y")
    fig.update_yaxes(ticksuffix="%")
    _footnote(
        fig,
        "Dotted verticals are fold boundaries, where the model was refitted on everything up "
        "to that point. A prediction line far flatter than the realised one is a model "
        "correctly admitting it does not know.",
    )
    return fig
