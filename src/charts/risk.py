"""Risk visuals: drawdowns, tails, rolling stability, contribution and stress.

This is the densest chart module in the application, and deliberately so -
performance charts show what happened, these show what it cost and what could
happen next. Four ideas run through all of them.

1. **The tail is the story.** :func:`return_distribution` overlays a normal
   density fitted to the same sample and colours every bar left of VaR in the
   loss colour, so the gap between "what a normal says should happen" and
   "what actually happened" is visible rather than asserted. The same honesty
   drives :func:`var_backtest_chart`, which shows every day the model was
   wrong instead of a single summary number.
2. **Duration matters as much as depth.** A -20% drawdown recovered in six
   weeks and a -20% drawdown that took three years are different experiences,
   so :func:`underwater_chart` and :func:`drawdown_episodes_chart` draw the
   *width* of a drawdown as prominently as its depth.
3. **Weight is not risk.** :func:`risk_contribution_chart` puts allocation and
   risk contribution on the same axis; a 10% position carrying 30% of the
   portfolio's volatility is the single most useful thing a concentration
   chart can say, and it is flagged explicitly.
4. **Say the assumption out loud.** Every figure carries its window, sample
   size and convention in the subtitle, and a methodology caption where the
   number would otherwise be easy to misread. Screenshots travel; caveats
   should travel with them.

Conventions
-----------
* Drawdowns, VaR and CVaR arrive as **negative decimals** (-0.021 = -2.1%) and
  are rendered as percent. Where a caller plausibly passes percent already,
  the rule used to detect it is documented on the function.
* Every function returns a :class:`plotly.graph_objects.Figure` and **never
  raises**. Degenerate input returns :func:`theme.empty_figure` naming the
  actual reason, so the UI shows "only 3 observations" rather than "no data".
* Nothing mutates its input: frames and series are copied before use.
* No Streamlit import - these builders work in a notebook, the HTML report
  writer or a test just as well as in the dashboard.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import theme
from src.config import DEFAULT_VAR_CONFIDENCE, RISK_RUBRIC
from src.models import DrawdownEpisode

__all__ = [
    "MIN_DISTRIBUTION_POINTS",
    "MAX_DISTRIBUTION_BINS",
    "MAX_SCATTER_LABELS",
    "RISK_WEIGHT_FLAG_RATIO",
    "drawdown_chart",
    "underwater_chart",
    "drawdown_episodes_chart",
    "return_distribution",
    "rolling_chart",
    "rolling_multi_chart",
    "risk_gauge",
    "risk_contribution_chart",
    "risk_return_scatter",
    "var_backtest_chart",
    "stress_bar",
]

# ---------------------------------------------------------------------------
# chart-level constants (thresholds src.config does not carry)
# ---------------------------------------------------------------------------
#: Below this many daily returns a histogram is shape-less noise, not a
#: distribution, so the tail comparison would be misleading.
MIN_DISTRIBUTION_POINTS: int = 10

#: Freedman-Diaconis can ask for hundreds of bins on a long, fat-tailed
#: sample; cap it so bars stay wide enough to read.
MAX_DISTRIBUTION_BINS: int = 60

#: Point labels stop being labels and start being noise past this count.
MAX_SCATTER_LABELS: int = 12

#: Risk contribution / weight above which a holding is called out as carrying
#: more risk than its allocation implies.
RISK_WEIGHT_FLAG_RATIO: float = 1.35

#: Row height used by the stacked rolling panel.
ROLLING_ROW_HEIGHT: int = 150

_DATE_FMT = "%d %b %Y"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _never_raises(fn: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Last-resort guard so a malformed frame degrades to a placeholder.

    Every function below handles its own degenerate cases explicitly and names
    them; this only catches the genuinely unexpected (a caller passing an
    object that quacks like a frame but is not) and keeps the promise that a
    chart builder never raises into the UI.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
            return theme.empty_figure(
                f"Chart unavailable - unexpected {type(exc).__name__} in the input."
            )

    return wrapper


def _rgba(color: str, alpha: float) -> str:
    """Return ``color`` at ``alpha`` opacity; accepts ``#rrggbb`` or rgb(a)."""
    c = str(color or "").strip()
    if c.startswith("#") and len(c) == 7:
        try:
            r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
            return f"rgba({r}, {g}, {b}, {alpha:.3f})"
        except ValueError:
            return c
    for prefix, n in (("rgba(", 4), ("rgb(", 3)):
        if c.startswith(prefix):
            parts = [p.strip() for p in c[len(prefix) : -1].split(",")]
            if len(parts) >= n:
                return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha:.3f})"
    return c


def _numeric_series(obj: Any) -> pd.Series:
    """Copy ``obj`` to a float Series with infinities and NaN removed."""
    if obj is None:
        return pd.Series(dtype="float64")
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, 0].copy() if obj.shape[1] else pd.Series(dtype="float64")
    else:
        try:
            s = pd.Series(obj).copy()
        except Exception:  # noqa: BLE001 - must degrade to a placeholder, never raise
            return pd.Series(dtype="float64")
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")


def _frame(obj: Any) -> pd.DataFrame:
    """Copy ``obj`` to a DataFrame, or an empty frame when it is not one."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame()


def _is_time(index: Any) -> bool:
    """True when the index is dated, so hovers and shading can use dates."""
    return isinstance(index, pd.DatetimeIndex)


def _x_values(s: pd.Series) -> list[Any]:
    """X coordinates as plain Python objects (Plotly-JSON safe)."""
    if _is_time(s.index):
        return [ts.to_pydatetime() for ts in s.index]
    return list(s.index)


def _y_values(s: pd.Series) -> list[float]:
    """Y values as plain floats - numpy scalars are not Plotly-JSON safe."""
    return [float(v) for v in s.to_numpy(dtype="float64")]


def _finite(x: Any) -> bool:
    """True when ``x`` coerces to a finite float (None, NaN and text are not)."""
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _fmt_date(value: Any) -> str:
    """Format a date-ish value for a subtitle or hover, never raising."""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "-"
        return ts.strftime(_DATE_FMT)
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: Any, symbol: str) -> str:
    """Grouped money string with the caller's symbol, em-dash when unknown."""
    if not _finite(value):
        return "-"
    return f"{symbol}{float(value):,.0f}"


def _fmt_pct(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Percent string for a value already in percent units; "-" when unknown."""
    if not _finite(value):
        return "-"
    spec = f"{{:+,.{decimals}f}}%" if signed else f"{{:,.{decimals}f}}%"
    return spec.format(float(value))


def _wrap(text: str, width: int = 56) -> str:
    """Soft-wrap a sentence with ``<br>`` so hover boxes stay narrow."""
    words = str(text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _value_at(series: pd.Series, when: Any) -> float:
    """Series value on or before ``when``; NaN when it cannot be resolved."""
    try:
        ts = pd.Timestamp(when)
        if ts in series.index:
            return float(series.loc[ts])
        return float(series.asof(ts))
    except Exception:  # noqa: BLE001 - must degrade to a placeholder, never raise
        return float("nan")


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian density - written out so the charts layer needs no SciPy."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian CDF via the error function."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _longest_true_run(mask: pd.Series) -> tuple[int, Any, Any]:
    """Longest consecutive True run: (length, first index, last index)."""
    best = 0
    best_span: tuple[Any, Any] = (None, None)
    run = 0
    start: Any = None
    for idx, flag in mask.items():
        if bool(flag):
            if run == 0:
                start = idx
            run += 1
            if run > best:
                best = run
                best_span = (start, idx)
        else:
            run = 0
    return best, best_span[0], best_span[1]


def _span_days(start: Any, end: Any, fallback: int) -> int:
    """Calendar days between two index labels, falling back to a count."""
    try:
        return int((pd.Timestamp(end) - pd.Timestamp(start)).days)
    except Exception:  # noqa: BLE001 - must degrade to a placeholder, never raise
        return int(fallback)


# ---------------------------------------------------------------------------
# drawdown family
# ---------------------------------------------------------------------------
@_never_raises
def drawdown_chart(dd: pd.Series, height: int | None = None) -> go.Figure:
    """Drawdown path: the running loss from the portfolio's high-water mark.

    Shows how far below its previous peak the portfolio has been on every day
    of the window, with the worst point marked. The area hangs from zero, so
    depth reads downward the way a loss should.

    Assumes ``dd`` is a drawdown **series in decimals** (-0.23 = -23%), as
    produced by ``analytics.risk.drawdown_series``. If the minimum is below
    -1.0 the series is assumed to already be in percent and is not rescaled,
    since a decimal drawdown can never be worse than -100%.
    """
    s = _numeric_series(dd)
    if s.empty:
        return theme.empty_figure("No drawdown series - the portfolio has no usable price history.")
    if len(s) < 2:
        return theme.empty_figure(
            "Only one observation - a drawdown path needs at least two dates."
        )

    already_pct = float(s.min()) < -1.0
    pct = s if already_pct else s * 100.0
    p = theme.palette()

    trough_idx = pct.idxmin()
    worst = float(pct.min())
    trough_x = trough_idx.to_pydatetime() if isinstance(trough_idx, pd.Timestamp) else trough_idx

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_values(pct),
            y=_y_values(pct),
            mode="lines",
            name="Drawdown",
            line={"color": p["loss"], "width": 1.4},
            fill="tozeroy",
            fillcolor=_rgba(p["loss"], 0.18),
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trough_x],
            y=[worst],
            mode="markers",
            name="Worst point",
            marker={
                "size": 9,
                "color": p["loss"],
                "line": {"color": p["text"], "width": 1.2},
                "symbol": "circle",
            },
            hovertemplate=f"Deepest drawdown {worst:,.2f}%<extra></extra>",
        )
    )

    span = ""
    if _is_time(pct.index):
        span = f" | {_fmt_date(pct.index[0])} to {_fmt_date(pct.index[-1])}"
    subtitle = (
        f"Loss from the running peak | worst {worst:,.2f}% on "
        f"{_fmt_date(trough_idx)} | {len(pct):,} observations{span}"
    )

    theme.apply(
        fig,
        "Drawdown",
        subtitle=subtitle,
        height=height or 320,
        legend=False,
        unified_hover=True,
    )
    theme.pct_axis(fig, "y")
    theme.add_zero_line(fig, "y")
    fig.update_yaxes(title_text="Drawdown (%)", range=[min(worst, -0.01) * 1.18, 0])
    fig.update_xaxes(title_text="Date" if _is_time(pct.index) else "Observation")
    return fig


@_never_raises
def underwater_chart(values: pd.Series, height: int | None = None) -> go.Figure:
    """Time spent below the high-water mark, with the longest stretch shaded.

    Takes the **portfolio value series** (not a drawdown) and derives the
    underwater curve as ``value / running_peak - 1``. Depth answers "how much
    did it cost"; the shaded band answers "how long did it stay lost", which
    is the part investors actually live through.

    Assumes values are positive levels in a single currency; non-positive and
    missing observations are dropped before the running peak is taken.
    """
    v = _numeric_series(values)
    v = v[v > 0]
    if v.empty:
        return theme.empty_figure(
            "No positive portfolio values - an underwater curve needs a value series."
        )
    if len(v) < 2:
        return theme.empty_figure(
            "Only one valuation date - the underwater curve needs at least two."
        )

    peak = v.cummax()
    dd_pct = (v / peak - 1.0) * 100.0
    underwater = dd_pct < -1e-9
    p = theme.palette()

    worst = float(dd_pct.min())
    uw_share = float(underwater.mean() * 100.0)
    run_len, run_start, run_end = _longest_true_run(underwater)
    run_days = _span_days(run_start, run_end, run_len) if run_len else 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_values(dd_pct),
            y=_y_values(dd_pct),
            mode="lines",
            name="Underwater",
            line={"color": p["loss"], "width": 1.3},
            fill="tozeroy",
            fillcolor=_rgba(p["loss"], 0.16),
            customdata=[[float(a), float(b)] for a, b in zip(v.to_numpy(), peak.to_numpy())],
            hovertemplate=(
                "Below peak: %{y:.2f}%<br>"
                "Value: %{customdata[0]:,.0f}<br>"
                "Prior peak: %{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    if run_len > 1 and run_start is not None and _is_time(dd_pct.index):
        fig.add_vrect(
            x0=pd.Timestamp(run_start).to_pydatetime(),
            x1=pd.Timestamp(run_end).to_pydatetime(),
            fillcolor=_rgba(p["warning"], 0.12),
            line_width=0,
            layer="below",
            annotation_text=f"Longest stretch: {run_days:,} days",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=p["warning"],
        )

    subtitle = (
        f"{uw_share:.0f}% of days below the prior high | deepest {worst:,.2f}% | "
        f"longest stretch {run_days:,} calendar days"
    )
    theme.apply(
        fig,
        "Underwater curve",
        subtitle=subtitle,
        height=height or 320,
        legend=False,
        unified_hover=True,
    )
    theme.pct_axis(fig, "y")
    theme.add_zero_line(fig, "y")
    fig.update_yaxes(title_text="Below high-water mark (%)", range=[min(worst, -0.01) * 1.18, 0])
    fig.update_xaxes(title_text="Date" if _is_time(dd_pct.index) else "Observation")
    theme.caption(
        fig,
        "Underwater = the portfolio is below its previous high. Zero means a new "
        "high-water mark; width is the time taken to earn the loss back.",
    )
    return fig


@_never_raises
def drawdown_episodes_chart(
    values: pd.Series,
    episodes: Sequence[DrawdownEpisode] | None = None,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Portfolio value with each drawdown episode shaded peak-to-recovery.

    Every shaded band is one complete peak -> trough -> recovery cycle; a band
    that runs to the right edge is a drawdown the portfolio has not yet
    recovered from. The cross marks each trough and carries the depth and the
    day counts in its hover.

    Assumes ``values`` is a dated portfolio value series and ``episodes`` are
    :class:`~src.models.DrawdownEpisode` records (deepest first) whose dates
    fall inside that series. Depth is a negative decimal.
    """
    v = _numeric_series(values)
    if v.empty:
        return theme.empty_figure(
            "No portfolio value series - drawdown episodes need a value history."
        )
    if len(v) < 2:
        return theme.empty_figure("Only one valuation date - drawdown episodes need at least two.")

    p = theme.palette()
    eps = [e for e in (episodes or []) if e is not None]
    dated = _is_time(v.index)
    x_end = v.index[-1]
    total_span = 0.0
    if dated and len(v) > 1:
        total_span = float((v.index[-1] - v.index[0]).days) or 0.0

    fig = go.Figure()
    # No area fill here: filling to zero would force the value axis to include
    # zero and flatten the path the episodes are meant to explain.
    fig.add_trace(
        go.Scatter(
            x=_x_values(v),
            y=_y_values(v),
            mode="lines",
            name="Portfolio value",
            line={"color": p["accent"], "width": 1.7},
            hovertemplate=f"Value: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        )
    )

    trough_x: list[Any] = []
    trough_y: list[float] = []
    trough_meta: list[list[Any]] = []
    shaded = 0

    if dated:
        for ep in eps:
            peak_date = getattr(ep, "peak_date", None)
            trough_date = getattr(ep, "trough_date", None)
            recovery_date = getattr(ep, "recovery_date", None)
            depth = getattr(ep, "depth", float("nan"))
            if peak_date is None:
                continue
            try:
                x0 = pd.Timestamp(peak_date)
                x1 = pd.Timestamp(recovery_date) if recovery_date else pd.Timestamp(x_end)
            except (TypeError, ValueError):
                continue
            if pd.isna(x0) or pd.isna(x1):
                continue

            recovered = bool(getattr(ep, "recovered", recovery_date is not None))
            colour = p["loss"] if recovered else p["warning"]
            width_share = float((x1 - x0).days) / total_span if total_span > 0 else 0.0
            label = _fmt_pct(float(depth) * 100.0 if _finite(depth) else float("nan"))
            annotate_this = width_share >= 0.06

            fig.add_vrect(
                x0=x0.to_pydatetime(),
                x1=x1.to_pydatetime(),
                fillcolor=_rgba(colour, 0.13),
                line_width=0,
                layer="below",
                **(
                    {
                        "annotation_text": label,
                        "annotation_position": "top left",
                        "annotation_font_size": 10,
                        "annotation_font_color": colour,
                    }
                    if annotate_this
                    else {}
                ),
            )
            shaded += 1

            if trough_date is not None:
                ts = pd.Timestamp(trough_date)
                y_val = _value_at(v, ts)
                if _finite(y_val):
                    trough_x.append(ts.to_pydatetime())
                    trough_y.append(float(y_val))
                    rec_days = getattr(ep, "recovery_days", None)
                    recovered_on = (
                        _fmt_date(recovery_date) if recovery_date else "not yet recovered"
                    )
                    took = f"{int(rec_days):,} days" if rec_days is not None else "still underwater"
                    trough_meta.append(
                        [
                            label,
                            _fmt_date(peak_date),
                            int(getattr(ep, "drawdown_days", 0) or 0),
                            recovered_on,
                            took,
                        ]
                    )

    if trough_x:
        fig.add_trace(
            go.Scatter(
                x=trough_x,
                y=trough_y,
                mode="markers",
                name="Trough",
                marker={
                    "size": 9,
                    "symbol": "x-thin",
                    "line": {"color": p["loss"], "width": 2},
                },
                customdata=trough_meta,
                hovertemplate=(
                    "Trough %{customdata[0]}<br>"
                    f"Value: {currency_symbol}%{{y:,.0f}}<br>"
                    "Peak: %{customdata[1]} (%{customdata[2]:,} days to trough)<br>"
                    "Recovered: %{customdata[3]} (%{customdata[4]})"
                    "<extra></extra>"
                ),
            )
        )

    if not dated:
        subtitle = "Episode shading needs a dated index - showing the value path only"
    elif shaded:
        unrecovered = sum(1 for e in eps if not getattr(e, "recovered", True))
        subtitle = (
            f"{shaded} drawdown episode{'s' if shaded != 1 else ''} shaded peak to recovery"
            + (f" | {unrecovered} still underwater" if unrecovered else " | all recovered")
        )
    else:
        subtitle = "No drawdown deep enough to qualify as an episode in this window"

    theme.apply(
        fig,
        "Drawdown episodes",
        subtitle=subtitle,
        height=height or 380,
        legend=bool(trough_x),
        unified_hover=True,
    )
    theme.money_axis(fig, currency_symbol, "y")
    unit = currency_symbol.strip() or "base currency"
    fig.update_yaxes(title_text=f"Portfolio value ({unit})")
    fig.update_xaxes(title_text="Date" if dated else "Observation")
    if shaded:
        theme.caption(
            fig,
            "Shaded bands run from the prior peak to the day the peak was regained; "
            "amber bands are drawdowns still open at the end of the window.",
        )
    return fig


# ---------------------------------------------------------------------------
# return distribution & VaR
# ---------------------------------------------------------------------------
@_never_raises
def return_distribution(
    returns: pd.Series,
    var: float | None = None,
    cvar: float | None = None,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    height: int | None = None,
) -> go.Figure:
    """Daily return histogram against a fitted normal, with VaR and CVaR.

    The normal curve is fitted with the sample's own mean and standard
    deviation and scaled to the histogram's counts, so any bar standing above
    the curve in the left tail is empirical loss frequency a Gaussian risk
    model would have under-priced. Bars at or below VaR are drawn in the loss
    colour, and the annotation compares observed tail days with the number the
    fitted normal predicts.

    Assumes ``returns`` are **daily decimal** returns and that ``var`` and
    ``cvar`` are **negative decimals** at ``confidence`` (as returned by
    ``analytics.risk.value_at_risk`` / ``conditional_var``). Either may be
    None or NaN, in which case its line is omitted.
    """
    r = _numeric_series(returns)
    if r.empty:
        return theme.empty_figure("No daily returns - a distribution needs a return series.")
    if len(r) < MIN_DISTRIBUTION_POINTS:
        return theme.empty_figure(
            f"Only {len(r)} daily returns - at least {MIN_DISTRIBUTION_POINTS} are "
            "needed before a histogram means anything."
        )

    pct = (r * 100.0).to_numpy(dtype="float64")
    sd = float(np.std(pct, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return theme.empty_figure(
            "Every daily return is identical - a distribution needs variation."
        )
    mu = float(np.mean(pct))
    n = int(pct.size)

    edges = np.histogram_bin_edges(pct, bins="auto")
    if edges.size - 1 > MAX_DISTRIBUTION_BINS:
        edges = np.histogram_bin_edges(pct, bins=MAX_DISTRIBUTION_BINS)
    counts, edges = np.histogram(pct, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    p = theme.palette()
    var_pct = float(var) * 100.0 if _finite(var) else float("nan")
    cvar_pct = float(cvar) * 100.0 if _finite(cvar) else float("nan")

    if np.isfinite(var_pct):
        tail_mask = centres <= var_pct
    else:
        tail_mask = np.zeros_like(centres, dtype=bool)
    bar_colors = [p["loss"] if flag else _rgba(p["accent"], 0.62) for flag in tail_mask]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[float(c) for c in centres],
            y=[float(c) for c in counts],
            width=[float(w) * 0.96 for w in widths],
            name="Observed days",
            marker={"color": bar_colors, "line": {"width": 0}},
            customdata=[
                [float(lo), float(hi), float(c) / n * 100.0]
                for lo, hi, c in zip(edges[:-1], edges[1:], counts)
            ],
            hovertemplate=(
                "Return %{customdata[0]:.2f}% to %{customdata[1]:.2f}%<br>"
                "%{y:,.0f} days (%{customdata[2]:.1f}% of sample)"
                "<extra></extra>"
            ),
        )
    )

    grid = np.linspace(float(edges[0]), float(edges[-1]), 240)
    density = _normal_pdf(grid, mu, sd) * n * float(np.mean(widths))
    fig.add_trace(
        go.Scatter(
            x=[float(x) for x in grid],
            y=[float(y) for y in density],
            mode="lines",
            name=f"Normal fit (mean {mu:.2f}%, sd {sd:.2f}%)",
            line={"color": p["accent_2"], "width": 2, "dash": "solid"},
            hovertemplate="Normal fit: %{y:,.1f} days at %{x:.2f}%<extra></extra>",
        )
    )

    conf_label = f"{confidence * 100:.0f}%" if _finite(confidence) else "95%"
    if np.isfinite(var_pct):
        theme.add_reference_line(
            fig,
            var_pct,
            label=f"VaR {conf_label} {var_pct:,.2f}%",
            axis="x",
            color=p["warning"],
            dash="dash",
        )
    if np.isfinite(cvar_pct):
        theme.add_reference_line(
            fig,
            cvar_pct,
            label=f"CVaR {conf_label} {cvar_pct:,.2f}%",
            axis="x",
            color=p["loss"],
            dash="dot",
        )

    skew = float(pd.Series(pct).skew()) if n >= 3 else float("nan")
    kurt = float(pd.Series(pct).kurt()) if n >= 4 else float("nan")
    subtitle = (
        f"{n:,} daily returns | mean {mu:.2f}% | sd {sd:.2f}% | "
        f"skew {skew:.2f} | excess kurtosis {kurt:.2f}"
    )

    if np.isfinite(var_pct):
        observed_tail = int(np.sum(pct <= var_pct))
        expected_tail = _normal_cdf(var_pct, mu, sd) * n
        verdict = "fatter" if observed_tail > expected_tail else "thinner"
        theme.annotate(
            fig,
            (
                f"<b>{observed_tail:,}</b> days at or beyond VaR vs "
                f"<b>{expected_tail:,.1f}</b> expected under the normal fit "
                f"- left tail is {verdict}"
            ),
            x=0.0,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            size=10.5,
            color=p["loss"] if observed_tail > expected_tail else p["subtext"],
        )

    theme.apply(
        fig,
        "Distribution of daily returns",
        subtitle=subtitle,
        height=height or 380,
        legend=True,
        bargap=0.02,
    )
    theme.pct_axis(fig, "x", decimals=1)
    fig.update_xaxes(title_text="Daily return (%)")
    fig.update_yaxes(title_text="Number of days", rangemode="tozero")
    theme.caption(
        fig,
        "Normal density fitted on the sample mean and standard deviation, scaled to "
        "the histogram. Bars left of VaR are the loss tail: mass above the curve "
        "there is exactly the risk a Gaussian model under-states.",
    )
    return fig


@_never_raises
def var_backtest_chart(
    returns: pd.Series,
    var_threshold: float,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    height: int | None = None,
) -> go.Figure:
    """Daily returns against the VaR line, with every breach highlighted.

    A breach is a day whose return fell at or below the VaR threshold. A
    well-calibrated 95% VaR should be breached on roughly 5% of days; a much
    higher rate means the model under-states risk, and breaches arriving in
    clusters mean it misses volatility regimes even when the count looks fine.

    Assumes ``returns`` are daily decimals and ``var_threshold`` is the
    **negative decimal** VaR at ``confidence`` estimated over the same window
    (this is an in-sample backtest, stated in the caption).
    """
    r = _numeric_series(returns)
    if r.empty:
        return theme.empty_figure("No daily returns - a VaR backtest needs a return series.")
    if not _finite(var_threshold):
        return theme.empty_figure(
            "VaR is not available for this window, so there is no threshold to test."
        )

    p = theme.palette()
    pct = r * 100.0
    var_pct = float(var_threshold) * 100.0
    breaches = pct[pct <= var_pct]
    n = len(pct)
    rate = len(breaches) / n * 100.0 if n else float("nan")
    expected_rate = (1.0 - float(confidence)) * 100.0 if _finite(confidence) else float("nan")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=_x_values(pct),
            y=_y_values(pct),
            name="Daily return",
            marker={"color": _rgba(p["muted"], 0.55), "line": {"width": 0}},
            hovertemplate="Return: %{y:.2f}%<extra></extra>",
        )
    )
    if len(breaches):
        fig.add_trace(
            go.Bar(
                x=_x_values(breaches),
                y=_y_values(breaches),
                name=f"VaR breach ({len(breaches)})",
                marker={"color": p["loss"], "line": {"width": 0}},
                hovertemplate="Breach: %{y:.2f}%<extra></extra>",
            )
        )

    conf_label = f"{float(confidence) * 100:.0f}%" if _finite(confidence) else "95%"
    theme.add_reference_line(
        fig,
        var_pct,
        label=f"{conf_label} VaR {var_pct:,.2f}%",
        axis="y",
        color=p["warning"],
        dash="dash",
    )

    subtitle = (
        f"{len(breaches):,} breach{'es' if len(breaches) != 1 else ''} in {n:,} days "
        f"({rate:.1f}% observed vs {expected_rate:.1f}% expected at {conf_label})"
    )
    theme.apply(
        fig,
        "VaR backtest",
        subtitle=subtitle,
        height=height or 340,
        legend=True,
        unified_hover=True,
        barmode="overlay",
        bargap=0.15,
    )
    theme.pct_axis(fig, "y")
    theme.add_zero_line(fig, "y")
    fig.update_yaxes(title_text="Daily return (%)")
    fig.update_xaxes(title_text="Date" if _is_time(pct.index) else "Observation")
    theme.caption(
        fig,
        "In-sample test: the threshold is estimated on the same window it is tested "
        "against, so it flatters the model. Clustered breaches signal volatility "
        "regimes a single static VaR cannot capture.",
    )
    return fig


# ---------------------------------------------------------------------------
# rolling stability
# ---------------------------------------------------------------------------
#: column -> (axis label, unit, reference line). ``pct`` columns are decimals
#: that get multiplied by 100 for display; ``ratio`` columns are plotted raw.
_ROLLING_META: dict[str, tuple[str, str, float | None]] = {
    "volatility": ("Annualised volatility (%)", "pct", None),
    "sharpe": ("Rolling Sharpe", "ratio", 0.0),
    "sortino": ("Rolling Sortino", "ratio", 0.0),
    "beta": ("Rolling beta", "ratio", 1.0),
    "correlation": ("Rolling correlation", "ratio", 0.0),
    "var": ("Rolling daily VaR (%)", "pct", None),
    "downside_deviation": ("Downside deviation (%)", "pct", None),
    "tracking_error": ("Tracking error (%)", "pct", None),
}

_ROLLING_TITLES: dict[str, str] = {
    "volatility": "Rolling volatility",
    "sharpe": "Rolling Sharpe ratio",
    "sortino": "Rolling Sortino ratio",
    "beta": "Rolling beta vs benchmark",
    "correlation": "Rolling correlation vs benchmark",
    "var": "Rolling Value-at-Risk",
    "downside_deviation": "Rolling downside deviation",
    "tracking_error": "Rolling tracking error",
}


def _rolling_meta(column: str) -> tuple[str, str, float | None]:
    """(axis label, unit, reference line) for a rolling column, with fallbacks."""
    key = str(column).strip().lower()
    if key in _ROLLING_META:
        return _ROLLING_META[key]
    return (str(column).replace("_", " ").capitalize(), "ratio", None)


def _rolling_display(series: pd.Series, unit: str) -> pd.Series:
    """Scale a rolling series into its display unit (decimals -> percent)."""
    return series * 100.0 if unit == "pct" else series


@_never_raises
def rolling_chart(
    frame: pd.DataFrame,
    column: str,
    title: str | None = None,
    subtitle: str | None = None,
    window: int | None = None,
    color: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """One rolling risk statistic through time, with its mean as a reference.

    A single number for volatility or Sharpe hides whether the portfolio was
    steadily that risky or calm-then-violent. The dotted line is the window
    mean, so drift away from it reads immediately; beta additionally carries a
    reference at 1.0 (market exposure) and signed ratios carry a zero line.

    Assumes ``frame`` is the output of ``analytics.rolling.rolling_frame``:
    ``volatility`` and ``var`` are **decimals** and are shown as percent;
    ``sharpe``, ``sortino``, ``beta`` and ``correlation`` are plotted raw.
    """
    df = _frame(frame)
    if df.empty:
        return theme.empty_figure(
            "The rolling frame is empty - the history is shorter than the window."
        )
    if column not in df.columns:
        available = ", ".join(str(c) for c in df.columns[:6]) or "none"
        return theme.empty_figure(
            f"Column '{column}' is not in the rolling frame (available: {available})."
        )

    raw = _numeric_series(df[column])
    if raw.empty:
        return theme.empty_figure(
            f"'{column}' has no values in this window - the rolling window is "
            "longer than the available history."
        )
    if len(raw) < 2:
        return theme.empty_figure(
            f"'{column}' has a single value - a rolling line needs at least two."
        )

    label, unit, reference = _rolling_meta(column)
    s = _rolling_display(raw, unit)
    p = theme.palette()
    line_color = color or p["accent"]
    mean_val = float(s.mean())
    suffix = "%" if unit == "pct" else ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_values(s),
            y=_y_values(s),
            mode="lines",
            name=label,
            line={"color": line_color, "width": 1.8},
            fill="tozeroy",
            fillcolor=_rgba(line_color, 0.12),
            hovertemplate=f"{label}: %{{y:,.2f}}{suffix}<extra></extra>",
        )
    )

    theme.add_reference_line(
        fig,
        mean_val,
        label=f"mean {mean_val:,.2f}{suffix}",
        axis="y",
        color=p["muted"],
        dash="dot",
    )
    if reference is not None and abs(reference - mean_val) > 1e-9:
        ref_label = "market beta 1.0" if str(column).lower() == "beta" else f"{reference:,.0f}"
        theme.add_reference_line(
            fig, float(reference), label=ref_label, axis="y", color=p["benchmark"], dash="dash"
        )

    window_txt = f"{int(window)}-day window | " if window else ""
    auto_subtitle = (
        f"{window_txt}{len(s):,} points | latest {float(s.iloc[-1]):,.2f}{suffix} | "
        f"mean {mean_val:,.2f}{suffix} | range {float(s.min()):,.2f}{suffix} to "
        f"{float(s.max()):,.2f}{suffix}"
    )

    theme.apply(
        fig,
        title or _ROLLING_TITLES.get(str(column).lower(), f"Rolling {column}"),
        subtitle=subtitle or auto_subtitle,
        height=height or 300,
        legend=False,
        unified_hover=True,
    )
    if unit == "pct":
        theme.pct_axis(fig, "y")
    fig.update_yaxes(title_text=label)
    fig.update_xaxes(title_text="Date" if _is_time(s.index) else "Observation")
    if reference is not None or float(s.min()) < 0:
        theme.add_zero_line(fig, "y")
    return fig


@_never_raises
def rolling_multi_chart(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    window: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Several rolling statistics stacked on a shared time axis.

    Sharing the x axis is the whole point: it lets the eye line up the month
    volatility spiked with the month Sharpe collapsed and beta jumped, which a
    grid of separate charts cannot do.

    Assumes ``frame`` is the output of ``analytics.rolling.rolling_frame``.
    Columns absent or entirely missing are skipped rather than drawn empty;
    percent-unit rows (volatility, VaR) are scaled from decimals.
    """
    df = _frame(frame)
    if df.empty:
        return theme.empty_figure(
            "The rolling frame is empty - the history is shorter than the window."
        )

    wanted = list(columns) if columns else [c for c in _ROLLING_META if c in df.columns]
    if not wanted:
        wanted = [str(c) for c in df.columns]

    usable: list[tuple[str, pd.Series, str, str, float | None]] = []
    for col in wanted:
        if col not in df.columns:
            continue
        raw = _numeric_series(df[col])
        if len(raw) < 2:
            continue
        label, unit, reference = _rolling_meta(col)
        usable.append((str(col), _rolling_display(raw, unit), label, unit, reference))

    if not usable:
        asked = ", ".join(str(c) for c in wanted[:6])
        return theme.empty_figure(
            f"None of the requested rolling series ({asked}) has two or more values in this window."
        )

    p = theme.palette()
    rows = len(usable)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.075, 0.30 / rows),
        subplot_titles=[label for _, _, label, _, _ in usable],
    )

    for i, (_col, s, label, unit, reference) in enumerate(usable, start=1):
        colour = theme.CATEGORICAL[(i - 1) % len(theme.CATEGORICAL)]
        suffix = "%" if unit == "pct" else ""
        fig.add_trace(
            go.Scatter(
                x=_x_values(s),
                y=_y_values(s),
                mode="lines",
                name=label,
                line={"color": colour, "width": 1.6},
                fill="tozeroy",
                fillcolor=_rgba(colour, 0.10),
                hovertemplate=f"{label}: %{{y:,.2f}}{suffix}<extra></extra>",
            ),
            row=i,
            col=1,
        )
        if reference is not None:
            fig.add_hline(
                y=float(reference),
                row=i,
                col=1,
                line_width=1,
                line_dash="dot",
                line_color=p["benchmark"],
            )
        if unit == "pct":
            fig.update_yaxes(ticksuffix="%", tickformat=".1f", row=i, col=1)
        fig.update_yaxes(
            title_text=label if rows <= 3 else None,
            title_font={"size": 11, "color": p["subtext"]},
            row=i,
            col=1,
        )

    last = usable[-1][1]
    fig.update_xaxes(title_text="Date" if _is_time(last.index) else "Observation", row=rows, col=1)

    window_txt = f"{int(window)}-day window | " if window else ""
    span = ""
    if _is_time(last.index):
        span = f" | {_fmt_date(last.index[0])} to {_fmt_date(last.index[-1])}"
    theme.apply(
        fig,
        "Rolling risk profile",
        subtitle=f"{window_txt}{rows} statistics on a shared time axis{span}",
        height=height or max(320, ROLLING_ROW_HEIGHT * rows + 120),
        legend=False,
        unified_hover=True,
    )
    # Left-align the subplot titles so they read as row labels, not headings.
    for ann in fig.layout.annotations[:rows]:
        ann.update(x=0, xanchor="left", font={"size": 11.5, "color": p["subtext"]})
    return fig


# ---------------------------------------------------------------------------
# score, contribution, risk/return
# ---------------------------------------------------------------------------
@_never_raises
def risk_gauge(
    score: float | None,
    band: str = "",
    height: int | None = None,
) -> go.Figure:
    """0-100 portfolio risk score on a banded gauge.

    The score is a transparent rubric (volatility, drawdown, concentration,
    holdings count and tail loss weighted by ``config.RISK_RUBRIC``), not a
    regulatory measure. Bands come from the same rubric: below
    ``low_band`` is conservative, above ``high_band`` is aggressive, and the
    needle marks the exact score.

    Assumes ``score`` is already on the 0-100 scale; None or NaN (the rubric's
    "insufficient data" case) returns a placeholder rather than a zero needle.
    """
    if not _finite(score):
        return theme.empty_figure(
            "Risk score unavailable - volatility or drawdown could not be computed."
        )

    value = max(0.0, min(100.0, float(score)))
    low = float(RISK_RUBRIC.low_band)
    high = float(RISK_RUBRIC.high_band)
    p = theme.palette()

    if value < low:
        band_color = p["gain"]
    elif value < high:
        band_color = p["warning"]
    else:
        band_color = p["loss"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "font": {"size": 40, "color": band_color, "family": theme.FONT_FAMILY},
                "suffix": " / 100",
                "valueformat": ".0f",
            },
            domain={"x": [0.0, 1.0], "y": [0.08, 1.0]},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, low, high, 100],
                    "tickwidth": 1,
                    "tickcolor": p["axis"],
                    "tickfont": {"size": 10, "color": p["subtext"]},
                },
                "bar": {"color": _rgba(band_color, 0.85), "thickness": 0.26},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, low], "color": _rgba(p["gain"], 0.20)},
                    {"range": [low, high], "color": _rgba(p["warning"], 0.20)},
                    {"range": [high, 100], "color": _rgba(p["loss"], 0.20)},
                ],
                "threshold": {
                    "line": {"color": p["text"], "width": 3},
                    "thickness": 0.82,
                    "value": value,
                },
            },
        )
    )

    if band:
        theme.annotate(
            fig,
            f"<b>{band}</b>",
            x=0.5,
            y=0.02,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            size=13,
            color=band_color,
        )

    theme.apply(
        fig,
        "Portfolio risk score",
        subtitle=(
            f"Rubric-weighted 0-100 | under {low:.0f} conservative, over {high:.0f} "
            "aggressive | not a regulatory rating"
        ),
        height=height or 300,
        legend=False,
    )
    return fig


@_never_raises
def risk_contribution_chart(
    rc: pd.DataFrame,
    top_n: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Capital weight against risk contribution, per holding.

    The chart every concentration discussion needs: a name can be 10% of the
    money and 30% of the volatility. Where risk contribution outruns weight by
    more than ``RISK_WEIGHT_FLAG_RATIO`` the multiple is printed at the end of
    the bar.

    Assumes ``rc`` is ``analytics.attribution.risk_contribution`` output with
    columns ``ticker, weight_pct, mctr, cctr, risk_contribution_pct``, both
    percent columns summing to ~100. ``mctr`` and ``cctr`` are annualised
    volatility units and are shown in the hover only.
    """
    df = _frame(rc)
    if df.empty:
        return theme.empty_figure(
            "No risk contribution table - it needs a covariance matrix and weights."
        )
    needed = {"ticker", "weight_pct", "risk_contribution_pct"}
    missing = needed - set(df.columns)
    if missing:
        return theme.empty_figure(
            "Risk contribution table is missing " + ", ".join(sorted(missing)) + "."
        )

    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")
    df["risk_contribution_pct"] = pd.to_numeric(df["risk_contribution_pct"], errors="coerce")
    for col in ("mctr", "cctr"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df = df.dropna(subset=["risk_contribution_pct", "weight_pct"])
    if df.empty:
        return theme.empty_figure(
            "Every risk contribution is undefined - portfolio volatility is zero or NaN."
        )

    df = df.sort_values("risk_contribution_pct", ascending=False)
    if top_n:
        df = df.head(int(top_n))
    df = df.iloc[::-1]  # plotly draws the first category at the bottom

    tickers = [str(t) for t in df["ticker"]]
    weights = [float(v) for v in df["weight_pct"]]
    risks = [float(v) for v in df["risk_contribution_pct"]]
    ratios = [
        (r / w) if (_finite(w) and abs(w) > 1e-9) else float("nan") for r, w in zip(risks, weights)
    ]
    p = theme.palette()
    weight_color = theme.CATEGORICAL[0]
    risk_color = theme.CATEGORICAL[5]

    meta = [
        [
            float(m) if _finite(m) else float("nan"),
            float(c) if _finite(c) else float("nan"),
            float(ratio) if _finite(ratio) else float("nan"),
        ]
        for m, c, ratio in zip(df["mctr"], df["cctr"], ratios)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weights,
            y=tickers,
            orientation="h",
            name="Capital weight",
            marker={"color": weight_color, "line": {"width": 0}},
            hovertemplate="Capital weight: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=risks,
            y=tickers,
            orientation="h",
            name="Risk contribution",
            marker={"color": risk_color, "line": {"width": 0}},
            customdata=meta,
            hovertemplate=(
                "Risk contribution: %{x:.2f}%<br>"
                "Risk / weight: %{customdata[2]:.2f}x<br>"
                "MCTR: %{customdata[0]:.4f}<br>"
                "CCTR: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    flagged = 0
    span = max(risks + weights) if (risks or weights) else 1.0
    for ticker, risk, ratio in zip(tickers, risks, ratios):
        if _finite(ratio) and ratio >= RISK_WEIGHT_FLAG_RATIO:
            flagged += 1
            theme.annotate(
                fig,
                f"{ratio:.1f}x",
                x=risk + span * 0.012,
                y=ticker,
                xanchor="left",
                yanchor="middle",
                size=10,
                color=p["warning"],
            )

    subtitle = f"{len(tickers)} holdings | bars above their weight carry more risk than capital" + (
        f" | {flagged} flagged over {RISK_WEIGHT_FLAG_RATIO:.2f}x" if flagged else ""
    )
    theme.apply(
        fig,
        "Weight vs risk contribution",
        subtitle=subtitle,
        height=height or max(300, 34 * len(tickers) + 140),
        legend=True,
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
    )
    theme.pct_axis(fig, "x")
    # Headroom so the "x" multiple printed past the bar end never clips.
    upper = max([v for v in (risks + weights) if _finite(v)] or [1.0])
    lower = min([v for v in (risks + weights) if _finite(v)] or [0.0])
    fig.update_xaxes(
        title_text="Share of portfolio (%)",
        range=[min(0.0, lower * 1.15), upper * 1.18],
    )
    fig.update_yaxes(title_text="", automargin=True)
    theme.caption(
        fig,
        "MCTR is the marginal contribution to risk (d sigma_p / d w_i); CCTR = w_i x "
        "MCTR_i and sums to portfolio volatility. Risk share = CCTR_i / sigma_p.",
    )
    return fig


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """First matching column name, case-insensitively."""
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _to_percent(series: pd.Series, column_name: str) -> pd.Series:
    """Scale to percent when a column looks like decimals.

    A column whose name carries ``pct`` is taken as percent already; otherwise
    a series whose largest absolute value is at most 2.5 is read as decimals
    (a 250% annualised return is possible but a 2.5% one is far likelier) and
    multiplied by 100. The rule is documented on the caller.
    """
    if "pct" in column_name.lower() or "percent" in column_name.lower():
        return series
    finite = series[np.isfinite(series)]
    if finite.empty:
        return series
    return series * 100.0 if float(finite.abs().max()) <= 2.5 else series


@_never_raises
def risk_return_scatter(
    positions: pd.DataFrame,
    asset_stats: pd.DataFrame,
    height: int | None = None,
) -> go.Figure:
    """Annualised return against annualised volatility, sized by weight.

    Each bubble is a holding: up is better, right is riskier, and bubble area
    is the capital allocated to it. The quadrant lines sit at the
    weight-averaged return and volatility, so the top-left quadrant is
    everything delivering more return per unit of risk than the portfolio as a
    whole - and a large bubble in the bottom-right is the position to defend.

    Assumes ``positions`` carries ``ticker`` (and ideally ``weight_pct``) and
    ``asset_stats`` is indexed by ticker (or carries a ``ticker`` column) with
    an annualised return column and an annualised volatility column. Columns
    named ``*_pct`` are read as percent; otherwise values whose magnitude is
    at most 2.5 are read as decimals and scaled.
    """
    pos = _frame(positions)
    stats = _frame(asset_stats)
    if stats.empty:
        return theme.empty_figure(
            "No per-asset return/volatility statistics - they need overlapping price history."
        )

    if "ticker" in stats.columns:
        stats = stats.set_index("ticker")
    stats.index = [str(i).strip().upper() for i in stats.index]

    ret_col = _pick_column(
        stats,
        (
            "ann_return",
            "annualized_return",
            "annualised_return",
            "return",
            "cagr",
            "ann_return_pct",
            "return_pct",
            "annual_return",
        ),
    )
    vol_col = _pick_column(
        stats,
        (
            "ann_vol",
            "annualized_volatility",
            "annualised_volatility",
            "volatility",
            "vol",
            "ann_vol_pct",
            "volatility_pct",
            "annual_volatility",
            "std",
        ),
    )
    if ret_col is None or vol_col is None:
        return theme.empty_figure(
            "Asset statistics need an annualised return column and a volatility "
            "column; neither pair was found."
        )

    ret = _to_percent(pd.to_numeric(stats[ret_col], errors="coerce"), ret_col)
    vol = _to_percent(pd.to_numeric(stats[vol_col], errors="coerce"), vol_col)
    data = pd.DataFrame({"ret": ret, "vol": vol}).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return theme.empty_figure(
            "Every asset's return or volatility is missing - the window is too short."
        )

    if not pos.empty and "ticker" in pos.columns:
        pos = pos.copy()
        pos["ticker"] = pos["ticker"].astype(str).str.strip().str.upper()
        weights = (
            pd.to_numeric(pos["weight_pct"], errors="coerce")
            if "weight_pct" in pos.columns
            else None
        )
        if weights is not None:
            wmap = pd.Series(weights.to_numpy(), index=pos["ticker"].to_numpy())
            wmap = wmap[~wmap.index.duplicated(keep="first")]
            data = data.join(wmap.rename("weight_pct"), how="inner")
        else:
            data = data.loc[data.index.intersection(pos["ticker"].unique())]
            data["weight_pct"] = 100.0 / max(len(data), 1)
        if data.empty:
            return theme.empty_figure(
                "No overlap between the holdings and the return/volatility statistics."
            )
    else:
        data["weight_pct"] = 100.0 / len(data)

    data["weight_pct"] = pd.to_numeric(data["weight_pct"], errors="coerce").fillna(0.0)
    data = data.sort_values("weight_pct", ascending=False)

    p = theme.palette()
    tickers = [str(t) for t in data.index]
    xs = [float(v) for v in data["vol"]]
    ys = [float(v) for v in data["ret"]]
    ws = [float(v) for v in data["weight_pct"]]
    colors = theme.sign_colors(ys)

    total_w = sum(w for w in ws if _finite(w))
    if total_w > 0:
        avg_vol = sum(x * w for x, w in zip(xs, ws)) / total_w
        avg_ret = sum(y * w for y, w in zip(ys, ws)) / total_w
    else:
        avg_vol = float(np.mean(xs))
        avg_ret = float(np.mean(ys))

    max_w = max(ws) if ws and max(ws) > 0 else 1.0
    labels = (
        tickers
        if len(tickers) <= MAX_SCATTER_LABELS
        else [t if i < MAX_SCATTER_LABELS else "" for i, t in enumerate(tickers)]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            name="Holdings",
            text=labels,
            textposition="top center",
            textfont={"size": 10, "color": p["subtext"]},
            marker={
                "size": ws,
                "sizemode": "area",
                "sizeref": 2.0 * max_w / (38.0**2),
                "sizemin": 6,
                "color": colors,
                "opacity": 0.85,
                "line": {"width": 1, "color": _rgba(p["text"], 0.35)},
            },
            customdata=[
                [t, float(w), (r / v) if abs(v) > 1e-9 else float("nan")]
                for t, w, r, v in zip(tickers, ws, ys, xs)
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Annualised return: %{y:,.2f}%<br>"
                "Annualised volatility: %{x:,.2f}%<br>"
                "Weight: %{customdata[1]:.2f}%<br>"
                "Return per unit of risk: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    theme.add_reference_line(
        fig, avg_vol, label=f"portfolio vol {avg_vol:,.1f}%", axis="x", color=p["muted"]
    )
    theme.add_reference_line(
        fig, avg_ret, label=f"portfolio return {avg_ret:,.1f}%", axis="y", color=p["muted"]
    )

    hidden = max(0, len(tickers) - MAX_SCATTER_LABELS)
    subtitle = (
        f"{len(tickers)} holdings | bubble area = capital weight | quadrants split at "
        f"the weighted average"
        + (f" | {hidden} smaller names labelled on hover only" if hidden else "")
    )
    theme.apply(
        fig,
        "Risk vs return by holding",
        subtitle=subtitle,
        height=height or 420,
        legend=False,
    )
    theme.pct_axis(fig, "x")
    theme.pct_axis(fig, "y")
    theme.add_zero_line(fig, "y")
    fig.update_xaxes(title_text="Annualised volatility (%)", rangemode="tozero")
    fig.update_yaxes(title_text="Annualised return (%)")
    theme.caption(
        fig,
        "Statistics are realised over the selected window, not forecasts; a short "
        "window makes a lucky quarter look like skill.",
    )
    return fig


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
@_never_raises
def stress_bar(
    stress: pd.DataFrame,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Portfolio impact of each historical stress window, worst first.

    Today's weights are replayed through each crisis window's actual asset
    returns. Coverage is stated on every scenario that is only partly
    covered - a stock listed in 2021 has no COVID-crash history, and a
    scenario computed on 40% of the book is not a scenario for the book.

    Assumes ``stress`` is ``analytics.stress.historical_stress`` output with
    columns ``scenario, start, end, days, coverage, portfolio_return_pct,
    value_change, worst_asset, worst_asset_pct, note``; ``coverage`` is a
    fraction of current weight and ``portfolio_return_pct`` is percent.
    """
    df = _frame(stress)
    if df.empty:
        return theme.empty_figure("No stress scenarios overlap the available price history.")
    if "portfolio_return_pct" not in df.columns:
        return theme.empty_figure("Stress table has no 'portfolio_return_pct' column to plot.")

    df["portfolio_return_pct"] = pd.to_numeric(df["portfolio_return_pct"], errors="coerce")
    df = df.dropna(subset=["portfolio_return_pct"])
    if df.empty:
        return theme.empty_figure(
            "Every stress scenario returned NaN - no window overlaps the price history."
        )

    if "scenario" not in df.columns:
        df["scenario"] = [f"Scenario {i + 1}" for i in range(len(df))]
    df["scenario"] = df["scenario"].astype(str)

    coverage = (
        pd.to_numeric(df["coverage"], errors="coerce")
        if "coverage" in df.columns
        else pd.Series(1.0, index=df.index)
    ).fillna(1.0)
    df = df.assign(_coverage=coverage)

    # Descending sort puts the worst scenario at the TOP of a horizontal axis.
    df = df.sort_values("portfolio_return_pct", ascending=False)

    p = theme.palette()
    values = [float(v) for v in df["portfolio_return_pct"]]
    colors = theme.sign_colors(values)

    tick_text: list[str] = []
    partial = 0
    for scenario, cov in zip(df["scenario"], df["_coverage"]):
        if _finite(cov) and cov < 0.999:
            partial += 1
            tick_text.append(
                f"{scenario} <span style='color:{p['warning']};font-size:10px'>"
                f"({cov * 100:.0f}% covered)</span>"
            )
        else:
            tick_text.append(str(scenario))

    def _col(name: str) -> list[Any]:
        """Optional stress column as a list, padded with None when absent."""
        return list(df[name]) if name in df.columns else [None] * len(df)

    meta: list[list[str]] = []
    for scen, start, end, days, cov, change, worst, worst_pct, note in zip(
        df["scenario"],
        _col("start"),
        _col("end"),
        _col("days"),
        df["_coverage"],
        _col("value_change"),
        _col("worst_asset"),
        _col("worst_asset_pct"),
        _col("note"),
    ):
        meta.append(
            [
                str(scen),
                _fmt_money(change, currency_symbol),
                f"{_fmt_date(start)} to {_fmt_date(end)}",
                f"{int(days):,} days" if _finite(days) else "-",
                f"{float(cov) * 100:.0f}%" if _finite(cov) else "-",
                str(worst) if worst is not None and str(worst) != "nan" else "-",
                _fmt_pct(worst_pct, 2, signed=True),
                _wrap(str(note)) if note is not None and str(note) != "nan" else "",
            ]
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=tick_text,
            orientation="h",
            name="Scenario impact",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{v:,.1f}%" for v in values],
            textposition="outside",
            textfont={"size": 10.5, "color": p["subtext"]},
            cliponaxis=False,
            customdata=meta,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Portfolio impact: %{x:.2f}%<br>"
                "Value change: %{customdata[1]}<br>"
                "Window: %{customdata[2]} (%{customdata[3]})<br>"
                "Weight covered: %{customdata[4]}<br>"
                "Worst holding: %{customdata[5]} %{customdata[6]}<br>"
                "<i>%{customdata[7]}</i>"
                "<extra></extra>"
            ),
        )
    )

    worst_row = df.iloc[-1]
    subtitle = (
        f"{len(df)} scenarios on today's weights | worst: "
        f"{worst_row['scenario']} {float(worst_row['portfolio_return_pct']):,.1f}%"
        + (f" | {partial} with partial history" if partial else " | full history for all")
    )
    theme.apply(
        fig,
        "Historical stress scenarios",
        subtitle=subtitle,
        height=height or max(300, 40 * len(df) + 140),
        legend=False,
        bargap=0.35,
    )
    theme.pct_axis(fig, "x")
    theme.add_zero_line(fig, "x")
    # Pad both ends so the value printed outside each bar stays inside the plot.
    lo, hi = min([*values, 0.0]), max([*values, 0.0])
    pad = max(abs(lo), abs(hi), 1.0) * 0.16
    fig.update_xaxes(title_text="Portfolio return in scenario (%)", range=[lo - pad, hi + pad])
    fig.update_yaxes(
        title_text="",
        categoryorder="array",
        categoryarray=tick_text,
        automargin=True,
    )
    theme.caption(
        fig,
        "Current weights replayed through each window's realised asset returns - not "
        "a forecast. Coverage is the share of today's weight that actually traded in "
        "that window; a partly covered scenario understates the true impact.",
    )
    return fig
