"""Mean-variance optimisation figures: the efficient frontier and its weights.

The efficient frontier chart is the centrepiece of the application, so it is
built in explicit layers, back to front, the way a printed exhibit would be:

1. a faint cloud of randomly sampled long-only portfolios, coloured by Sharpe
   ratio, which shows the *feasible set* and makes the frontier's edge legible;
2. the efficient frontier itself as a bold line with a background halo, so it
   survives being drawn on top of several thousand scatter points;
3. the capital market line from the risk-free rate through the tangency
   (maximum-Sharpe) portfolio, extended beyond it into the leveraged region;
4. each named optimal portfolio as a large directly-labelled marker - direct
   labelling rather than a legend, because a legend forces the reader to
   look away from the data to decode it;
5. the **current** portfolio as an amber star, plus the return it is giving up
   at its own level of risk. That last number is the entire point of the page.

Units and conventions
---------------------
* x is **annualised volatility in percent**, y is **annualised return in
  percent**. Inputs are accepted as decimals (0.18) or as percent (18.0); the
  units are detected once from the frontier and applied consistently to every
  layer, so the chart can never mix the two.
* Everything is long-only and fully invested unless the upstream optimiser
  said otherwise; a portfolio that failed to converge says so in its hover.
* Expected returns are estimates. The frontier is an in-sample object and is
  notoriously unstable out of sample - the caption says so on the figure.

No Streamlit import: every builder here is a pure ``go.Figure`` factory.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = [
    "efficient_frontier_chart",
    "weights_comparison",
    "weights_donut",
    "frontier_weights_area",
]

#: Columns of a frontier frame that are statistics rather than asset weights.
STAT_COLUMNS = frozenset({"return", "volatility", "sharpe"})

#: Marker shapes cycled across the named optimal portfolios.
_OPTIMAL_SYMBOLS = (
    "diamond",
    "square",
    "circle",
    "triangle-up",
    "hexagon",
    "pentagon",
    "triangle-down",
    "cross",
)


# ---------------------------------------------------------------------------
# defensive plumbing
# ---------------------------------------------------------------------------
def _never_raises(
    reason: str,
) -> Callable[[Callable[..., go.Figure]], Callable[..., go.Figure]]:
    """Turn any unexpected exception into a styled placeholder figure.

    A chart builder that raises takes the whole Streamlit page with it. Every
    public function here is wrapped so the worst case is one honest tile.
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


def _finite(value: Any) -> float:
    """Best-effort float, NaN for anything that will not convert."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _scale_factor(volatility: np.ndarray, returns: np.ndarray) -> float:
    """Detect whether inputs are decimals or already percent.

    An annualised volatility above 3.0 can only be percent units (300% is not
    a real portfolio); below that it can only be a decimal. The factor is
    resolved once from the frontier and reused for every other layer so the
    chart can never plot decimals and percentages on the same axis.
    """
    pool = np.concatenate([np.abs(volatility), np.abs(returns)])
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return 100.0
    return 1.0 if float(np.nanmax(pool)) > 3.0 else 100.0


def _weight_columns(frame: pd.DataFrame) -> list[str]:
    """Asset-weight columns of a frontier frame (everything but the stats)."""
    return [c for c in frame.columns if str(c).strip().lower() not in STAT_COLUMNS]


def _weights_to_percent(values: np.ndarray) -> np.ndarray:
    """Scale a weight vector to percent, accepting fractions or percentages."""
    total = float(np.nansum(np.abs(values)))
    if not np.isfinite(total) or total <= 0:
        return values
    return values * 100.0 if total <= 1.5 else values


def _as_weight_mapping(obj: Any) -> dict[str, float]:
    """Coerce weights held as a dict, Series or OptimizedPortfolio to a dict."""
    if obj is None:
        return {}
    if hasattr(obj, "weights"):
        obj = obj.weights
    if isinstance(obj, pd.Series):
        obj = obj.to_dict()
    if not isinstance(obj, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in obj.items():
        v = _finite(value)
        if np.isfinite(v):
            out[str(key)] = v
    return out


def _top_weights_text(weights: Mapping[str, float], top_n: int = 3) -> str:
    """``"TCS 34%, INFY 21%, ITC 12%"`` - the hover's one-line composition."""
    if not weights:
        return "n/a"
    pairs = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    scale = 100.0 if max(abs(v) for v in weights.values()) <= 1.5 else 1.0
    return ", ".join(f"{k} {v * scale:.0f}%" for k, v in pairs)


def _rgba(color: str, alpha: float) -> str:
    """Add alpha to a ``#rrggbb`` theme colour; pass rgb/rgba strings through."""
    text = str(color).strip()
    if text.startswith("#") and len(text) == 7:
        r, g, b = (int(text[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r}, {g}, {b}, {alpha})"
    return text


def _portfolio_point(obj: Any, fallback_label: str = "Portfolio") -> dict[str, Any] | None:
    """Read an ``OptimizedPortfolio`` (or an equivalent dict) defensively."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):

        def get(key: str, default: Any = None) -> Any:
            return obj.get(key, default)

    else:

        def get(key: str, default: Any = None) -> Any:
            return getattr(obj, key, default)

    ret = _finite(get("expected_return"))
    vol = _finite(get("volatility"))
    if not (np.isfinite(ret) and np.isfinite(vol)):
        return None
    return {
        "label": str(get("label") or fallback_label),
        "expected_return": ret,
        "volatility": vol,
        "sharpe": _finite(get("sharpe")),
        "weights": _as_weight_mapping(get("weights", {})),
        "converged": bool(get("converged", True)),
        "note": str(get("note", "") or ""),
    }


def _hover_suffix(point: Mapping[str, Any]) -> str:
    """Convergence / caveat line appended to an optimal portfolio's hover."""
    bits: list[str] = []
    if not point.get("converged", True):
        bits.append("did not fully converge")
    note = str(point.get("note", "") or "").strip()
    if note:
        bits.append(note)
    return "<br><i>" + "; ".join(bits) + "</i>" if bits else ""


# ---------------------------------------------------------------------------
# the marquee figure
# ---------------------------------------------------------------------------
@_never_raises("Efficient frontier could not be rendered")
def efficient_frontier_chart(
    frontier: pd.DataFrame | None,
    random: pd.DataFrame | None = None,
    portfolios: Mapping[str, Any] | None = None,
    current: Any = None,
    risk_free_rate: float = 0.0,
    show_cml: bool = True,
    height: int | None = None,
) -> go.Figure:
    """The efficient frontier with the feasible set, the CML and the portfolio.

    Plots annualised return against annualised volatility in five layers: a
    faint Sharpe-coloured cloud of random long-only portfolios (the feasible
    set), the efficient frontier as a bold line, the capital market line drawn
    from the risk-free rate through the tangency portfolio and extended past
    it, each named optimal portfolio as a directly-labelled marker, and the
    current portfolio as an amber star. When the current portfolio sits below
    the frontier, the chart also draws the vertical gap and states how much
    return is being left on the table *at the same level of risk*.

    Parameters
    ----------
    frontier:
        Columns ``return``, ``volatility``, ``sharpe`` plus one column per
        ticker holding that point's weight (extra weight columns are used only
        to enrich the hover).
    random:
        Columns ``return``, ``volatility``, ``sharpe`` - the Dirichlet-sampled
        feasible set. Optional; the chart is complete without it.
    portfolios:
        ``{label: OptimizedPortfolio}``. Marker colours are assigned in
        dictionary order and drawn from ``theme.CATEGORICAL``, matching the
        order :func:`weights_comparison` uses, so a strategy keeps one colour
        across the whole optimiser page.
    current:
        The investor's actual portfolio scored against the same mu/cov, as an
        ``OptimizedPortfolio``. Drawn as a star so it is never mistaken for a
        model portfolio.
    risk_free_rate:
        Annualised decimal (0.065) or percent (6.5); matched automatically to
        the units detected from ``frontier``.
    show_cml:
        Draw the capital market line. Requires a positive-volatility
        maximum-Sharpe point; it is skipped silently when one is unavailable.

    Assumptions
    -----------
    Mean-variance optimisation assumes returns are adequately described by
    their first two moments and that the historical mu and covariance are the
    forward-looking ones. The frontier is an **in-sample** object: it is fitted
    on the same history it is plotted against, so the gap between the current
    portfolio and the frontier is an upper bound on the improvement actually
    achievable out of sample.
    """
    if frontier is None or not isinstance(frontier, pd.DataFrame) or frontier.empty:
        return theme.empty_figure(
            "The efficient frontier is empty - the optimiser could not solve for any "
            "feasible target return."
        )
    front = frontier.copy()
    front.columns = [str(c) for c in front.columns]
    if "return" not in front.columns or "volatility" not in front.columns:
        return theme.empty_figure(
            "Frontier data is missing the 'return' and/or 'volatility' columns."
        )

    front["return"] = pd.to_numeric(front["return"], errors="coerce")
    front["volatility"] = pd.to_numeric(front["volatility"], errors="coerce")
    if "sharpe" in front.columns:
        front["sharpe"] = pd.to_numeric(front["sharpe"], errors="coerce")
    else:
        front["sharpe"] = np.nan
    front = front.dropna(subset=["return", "volatility"])
    if front.empty:
        return theme.empty_figure(
            "Every frontier point is non-finite - the covariance matrix is singular or "
            "the expected returns are undefined."
        )
    front = front.sort_values("volatility").reset_index(drop=True)

    p = theme.palette()
    scale = _scale_factor(
        front["volatility"].to_numpy(dtype=float), front["return"].to_numpy(dtype=float)
    )
    f_vol = front["volatility"].to_numpy(dtype=float) * scale
    f_ret = front["return"].to_numpy(dtype=float) * scale
    weight_cols = _weight_columns(front)

    fig = go.Figure()

    # -- layer 1: the feasible set -----------------------------------------
    n_random = 0
    if isinstance(random, pd.DataFrame) and not random.empty:
        cloud = random.copy()
        cloud.columns = [str(c) for c in cloud.columns]
        if {"return", "volatility"}.issubset(cloud.columns):
            cloud["return"] = pd.to_numeric(cloud["return"], errors="coerce")
            cloud["volatility"] = pd.to_numeric(cloud["volatility"], errors="coerce")
            cloud["sharpe"] = (
                pd.to_numeric(cloud["sharpe"], errors="coerce")
                if "sharpe" in cloud.columns
                else np.nan
            )
            cloud = cloud.dropna(subset=["return", "volatility"])
            n_random = int(len(cloud))
            if n_random:
                sharpe = cloud["sharpe"].to_numpy(dtype=float)
                fig.add_trace(
                    go.Scattergl(
                        x=cloud["volatility"].to_numpy(dtype=float) * scale,
                        y=cloud["return"].to_numpy(dtype=float) * scale,
                        mode="markers",
                        name="Random portfolios",
                        showlegend=False,
                        customdata=np.nan_to_num(sharpe, nan=0.0),
                        marker={
                            "size": 4.5,
                            "opacity": 0.45,
                            # trimmed of its lightest steps so points stay
                            # visible on a light page as well as a dark one
                            "colorscale": theme.SEQUENTIAL[1:9],
                            "color": sharpe,
                            "showscale": True,
                            "colorbar": {
                                "title": {
                                    "text": "Sharpe",
                                    "font": {"size": 11, "color": p["subtext"]},
                                },
                                "thickness": 12,
                                "len": 0.62,
                                "y": 0.42,
                                "x": 1.015,
                                "outlinewidth": 0,
                                "tickfont": {"size": 10, "color": p["subtext"]},
                            },
                        },
                        hovertemplate=(
                            "<b>Random portfolio</b><br>"
                            "Volatility  %{x:.2f}%<br>"
                            "Return  %{y:.2f}%<br>"
                            "Sharpe  %{customdata:.2f}<extra></extra>"
                        ),
                    )
                )

    # -- layer 2: the frontier, with a halo so it reads over the cloud -----
    if len(f_vol) > 1:
        fig.add_trace(
            go.Scatter(
                x=f_vol,
                y=f_ret,
                mode="lines",
                name="frontier halo",
                showlegend=False,
                hoverinfo="skip",
                line={"color": p["hover_bg"], "width": 6.5},
                opacity=0.75,
            )
        )
    top_weights = [
        _top_weights_text({c: row[c] for c in weight_cols}) if weight_cols else "n/a"
        for _, row in front.iterrows()
    ]
    frontier_hover = (
        "<b>Efficient frontier</b><br>"
        "Volatility  %{x:.2f}%<br>"
        "Return  %{y:.2f}%<br>"
        "Sharpe  %{customdata[0]:.2f}"
    )
    if weight_cols:
        frontier_hover += "<br>Largest weights: %{customdata[1]}"
    frontier_hover += "<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=f_vol,
            y=f_ret,
            mode="lines" if len(f_vol) > 1 else "markers",
            name="Efficient frontier",
            line={"color": p["accent"], "width": 3.0},
            marker={"color": p["accent"], "size": 9},
            customdata=[
                [float(s) if np.isfinite(s) else 0.0, w]
                for s, w in zip(front["sharpe"].to_numpy(dtype=float), top_weights)
            ],
            hovertemplate=frontier_hover,
        )
    )

    # -- layer 3: the capital market line ----------------------------------
    points: list[dict[str, Any]] = []
    for key, value in (portfolios or {}).items():
        point = _portfolio_point(value, str(key))
        if point is not None:
            point["label"] = str(point.get("label") or key)
            points.append(point)

    rf_plot = float(risk_free_rate) * scale if np.isfinite(_finite(risk_free_rate)) else 0.0
    tangency = _tangency_point(points, front, scale)
    x_max = float(np.nanmax(f_vol)) if f_vol.size else 0.0
    if isinstance(random, pd.DataFrame) and n_random:
        x_max = max(x_max, float(np.nanmax(cloud["volatility"].to_numpy(dtype=float) * scale)))

    if show_cml and tangency is not None and tangency[0] > 0:
        t_vol, t_ret = tangency
        slope = (t_ret - rf_plot) / t_vol
        x_end = max(x_max * 1.06, t_vol * 1.25)
        fig.add_trace(
            go.Scatter(
                x=[0.0, t_vol, x_end],
                y=[rf_plot, t_ret, rf_plot + slope * x_end],
                mode="lines",
                name="Capital market line",
                line={"color": p["accent_2"], "width": 1.8, "dash": "dash"},
                hovertemplate=(
                    "<b>Capital market line</b><br>"
                    "Volatility  %{x:.2f}%<br>"
                    "Return  %{y:.2f}%<br>"
                    f"Slope = Sharpe of the tangency portfolio  {slope:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[rf_plot],
                mode="markers",
                name="Risk-free rate",
                showlegend=False,
                marker={
                    "size": 9,
                    "symbol": "circle",
                    "color": p["accent_2"],
                    "line": {"width": 1.2, "color": p["hover_bg"]},
                },
                hovertemplate=(
                    f"<b>Risk-free rate</b><br>Return  {rf_plot:.2f}%<br>"
                    "Volatility  0.00%<extra></extra>"
                ),
            )
        )

    # -- layer 4: the named optimal portfolios, directly labelled ----------
    for i, pt in enumerate(points):
        colour = theme.CATEGORICAL[i % len(theme.CATEGORICAL)]
        fig.add_trace(
            go.Scatter(
                x=[pt["volatility"] * scale],
                y=[pt["expected_return"] * scale],
                mode="markers+text",
                name=pt["label"],
                showlegend=False,
                text=[pt["label"]],
                textposition="top center" if i % 2 == 0 else "bottom center",
                textfont={"size": 11, "color": p["text"]},
                marker={
                    "size": 15,
                    "symbol": _OPTIMAL_SYMBOLS[i % len(_OPTIMAL_SYMBOLS)],
                    "color": colour,
                    "line": {"width": 1.6, "color": p["hover_bg"]},
                },
                hovertemplate=(
                    f"<b>{pt['label']}</b><br>"
                    "Volatility  %{x:.2f}%<br>"
                    "Return  %{y:.2f}%<br>"
                    f"Sharpe  {pt['sharpe']:.2f}<br>"
                    f"Largest weights: {_top_weights_text(pt['weights'])}"
                    f"{_hover_suffix(pt)}<extra></extra>"
                ),
            )
        )

    # -- layer 5: where the investor actually is ---------------------------
    cur = _portfolio_point(current, "Current portfolio")
    gap_note = ""
    if cur is not None:
        c_vol = cur["volatility"] * scale
        c_ret = cur["expected_return"] * scale
        if f_vol.size >= 2 and f_vol.min() <= c_vol <= f_vol.max():
            reachable = float(np.interp(c_vol, f_vol, f_ret))
            gap = reachable - c_ret
            if gap > 0.05:
                fig.add_trace(
                    go.Scatter(
                        x=[c_vol, c_vol],
                        y=[c_ret, reachable],
                        mode="lines",
                        name="Return forgone",
                        showlegend=False,
                        hoverinfo="skip",
                        line={"color": p["muted"], "width": 1.4, "dash": "dot"},
                    )
                )
                theme.annotate(
                    fig,
                    f"+{gap:.2f} pp available at the same risk",
                    x=c_vol,
                    y=(c_ret + reachable) / 2.0,
                    xanchor="left",
                    xshift=8,
                    size=10.5,
                    color=p["warning"],
                )
                gap_note = (
                    f" The current portfolio sits {gap:.2f} pp of annualised return "
                    "below the frontier at its own level of risk."
                )
        fig.add_trace(
            go.Scatter(
                x=[c_vol],
                y=[c_ret],
                mode="markers+text",
                name="Current portfolio",
                text=["Current"],
                textposition="bottom right",
                textfont={"size": 11.5, "color": p["warning"]},
                marker={
                    "size": 22,
                    "symbol": "star",
                    "color": p["warning"],
                    "line": {"width": 1.6, "color": p["hover_bg"]},
                },
                hovertemplate=(
                    "<b>Current portfolio</b><br>"
                    "Volatility  %{x:.2f}%<br>"
                    "Return  %{y:.2f}%<br>"
                    f"Sharpe  {cur['sharpe']:.2f}<br>"
                    f"Largest weights: {_top_weights_text(cur['weights'])}"
                    "<extra></extra>"
                ),
            )
        )

    subtitle = "Annualised, long-only, fully invested"
    if n_random:
        subtitle += f" - {n_random:,} sampled portfolios"
    if np.isfinite(rf_plot):
        subtitle += f" - risk-free {rf_plot:.2f}%"

    theme.apply(
        fig,
        title="Efficient frontier",
        subtitle=subtitle,
        height=height or 460,
        hovermode="closest",
    )
    fig.update_layout(margin_r=96)
    fig.update_xaxes(title_text="Annualised volatility (%)", rangemode="tozero")
    fig.update_yaxes(title_text="Annualised return (%)")
    theme.pct_axis(fig, axis="x", decimals=0)
    theme.pct_axis(fig, axis="y", decimals=0)
    theme.caption(
        fig,
        "Mean-variance frontier fitted in sample on the loaded history; expected returns "
        "and covariances are estimates, and out-of-sample results are typically worse." + gap_note,
    )
    return fig


def _tangency_point(
    points: Sequence[Mapping[str, Any]], front: pd.DataFrame, scale: float
) -> tuple[float, float] | None:
    """Locate the maximum-Sharpe point the capital market line runs through.

    Prefers a named optimal portfolio (the optimiser's own tangency solution)
    and falls back to the best Sharpe ratio found along the frontier, so the
    line is still drawn when only a frontier was supplied.
    """
    best: tuple[float, float, float] | None = None
    for pt in points:
        sharpe = _finite(pt.get("sharpe"))
        vol = _finite(pt.get("volatility"))
        ret = _finite(pt.get("expected_return"))
        if (
            np.isfinite(sharpe)
            and np.isfinite(vol)
            and vol > 0
            and np.isfinite(ret)
            and (best is None or sharpe > best[0])
        ):
            best = (sharpe, vol * scale, ret * scale)
    if best is not None:
        return best[1], best[2]

    sharpe_col = front["sharpe"].to_numpy(dtype=float)
    if not np.isfinite(sharpe_col).any():
        return None
    idx = int(np.nanargmax(sharpe_col))
    vol = float(front["volatility"].iloc[idx]) * scale
    ret = float(front["return"].iloc[idx]) * scale
    return (vol, ret) if vol > 0 else None


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
@_never_raises("Weights comparison could not be rendered")
def weights_comparison(
    weights_by_label: Mapping[str, Any] | None,
    top_n: int | None = None,
    min_weight_pct: float = 0.10,
    height: int | None = None,
) -> go.Figure:
    """Grouped bars comparing what each optimisation strategy would hold.

    One bar group per holding, one bar per strategy, so the reader can see at
    a glance which names every objective agrees on and which are an artefact
    of a single objective. Bars are horizontal because tickers such as
    ``HDFCBANK.NS`` do not fit under a vertical bar, and holdings are sorted
    by their average weight across strategies rather than alphabetically, so
    the names that matter are at the top.

    Parameters
    ----------
    weights_by_label:
        ``{strategy_label: {ticker: weight}}``. Weights are accepted as
        fractions summing to 1 or as percentages, detected per strategy.
        ``OptimizedPortfolio`` objects are also accepted in place of dicts.
    top_n:
        Keep only the ``top_n`` holdings by average weight.
    min_weight_pct:
        Drop holdings that no strategy gives at least this weight (in percent),
        so a long tail of rounding dust does not crowd the axis.

    Assumptions
    -----------
    All strategies are assumed to have been optimised over the same asset
    universe; a ticker missing from one strategy is shown as a zero weight,
    which is exactly what an optimiser that excluded it meant.
    """
    if not weights_by_label:
        return theme.empty_figure(
            "No optimised portfolios to compare - the optimiser has not produced any solutions yet."
        )

    columns: dict[str, dict[str, float]] = {}
    for label, payload in weights_by_label.items():
        mapping = _as_weight_mapping(payload)
        if mapping:
            columns[str(label)] = mapping
    if not columns:
        return theme.empty_figure(
            "Every optimised portfolio came back with empty weights - nothing to compare."
        )

    frame = pd.DataFrame(columns).fillna(0.0)
    if frame.empty:
        return theme.empty_figure("No overlapping holdings across the optimised portfolios.")
    for col in frame.columns:
        frame[col] = _weights_to_percent(frame[col].to_numpy(dtype=float))

    frame = frame.loc[frame.max(axis=1) >= float(min_weight_pct)]
    if frame.empty:
        return theme.empty_figure(
            f"No holding reaches {min_weight_pct:.2f}% in any strategy - the optimiser "
            "spread the portfolio into dust."
        )
    frame = frame.loc[frame.mean(axis=1).sort_values(ascending=True).index]
    if top_n:
        frame = frame.iloc[-int(top_n) :]

    tickers = [str(i) for i in frame.index]
    fig = go.Figure()
    for i, label in enumerate(frame.columns):
        fig.add_trace(
            go.Bar(
                y=tickers,
                x=frame[label].to_numpy(dtype=float),
                name=str(label),
                orientation="h",
                marker={"color": theme.CATEGORICAL[i % len(theme.CATEGORICAL)]},
                hovertemplate=(f"<b>{label}</b><br>%{{y}}<br>Weight  %{{x:.2f}}%<extra></extra>"),
            )
        )

    n_names = len(tickers)
    n_strats = int(frame.shape[1])
    block = 13.0 * n_strats + 16.0
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    theme.apply(
        fig,
        title="Weights by strategy",
        subtitle=(
            f"{n_names} holdings across {n_strats} objectives - long-only weights, "
            "sorted by average allocation"
        ),
        height=height or int(np.clip(block * n_names + 170, 320, 900)),
    )
    fig.update_xaxes(title_text="Weight (%)", rangemode="tozero")
    fig.update_yaxes(title_text="Holding", showgrid=False)
    theme.pct_axis(fig, axis="x", decimals=0)
    theme.add_zero_line(fig, axis="x")
    return fig


@_never_raises("Weights donut could not be rendered")
def weights_donut(
    weights: Mapping[str, float] | None,
    title: str = "Portfolio weights",
    min_slice_pct: float = 1.5,
    height: int | None = None,
) -> go.Figure:
    """Donut of one portfolio's target weights, with the tail folded together.

    Slices are ordered largest first so the colour ramp doubles as a ranking,
    and every slice below ``min_slice_pct`` is folded into a single muted
    "Other" wedge - a donut with twenty two-percent slivers communicates
    nothing but effort. The hole carries the holding count and the largest
    single weight, which is the concentration read most people actually want.

    Parameters
    ----------
    weights:
        ``{ticker: weight}`` as fractions summing to 1 or as percentages;
        detected from the total. An ``OptimizedPortfolio`` is also accepted.
    title:
        Figure title - normally the strategy label.
    min_slice_pct:
        Slices smaller than this (in percent) are folded into "Other".

    Assumptions
    -----------
    Weights are treated as long-only target allocations. Negative weights are
    plotted by magnitude, which a donut cannot represent honestly, so they are
    dropped and reported in the subtitle instead.
    """
    mapping = _as_weight_mapping(weights)
    if not mapping:
        return theme.empty_figure("No weights to plot - this portfolio has no allocation.")

    keys = list(mapping.keys())
    values = _weights_to_percent(np.array([mapping[k] for k in keys], dtype=float))
    negative = int(np.sum(values < 0))
    keep = values > 0
    keys = [k for k, ok in zip(keys, keep) if ok]
    values = values[keep]
    if values.size == 0:
        return theme.empty_figure(
            "Every weight in this portfolio is zero or negative - nothing to plot."
        )

    order = np.argsort(-values)
    keys = [keys[i] for i in order]
    values = values[order]

    small = values < float(min_slice_pct)
    n_small = int(small.sum())
    if n_small > 1:
        labels = [str(k) for k, ok in zip(keys, ~small) if ok]
        sizes = list(values[~small])
        labels.append(f"Other ({n_small} names)")
        sizes.append(float(values[small].sum()))
    else:
        labels = [str(k) for k in keys]
        sizes = list(values)

    p = theme.palette()
    colours = [theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i in range(len(labels))]
    if n_small > 1:
        colours[-1] = p["muted"]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[float(v) for v in sizes],
            sort=False,
            hole=0.62,
            direction="clockwise",
            marker={"colors": colours, "line": {"color": p["hover_bg"], "width": 1.4}},
            texttemplate="%{label}<br>%{value:.1f}%",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont={"size": 10.5},
            hovertemplate="<b>%{label}</b><br>Weight  %{value:.2f}%<extra></extra>",
        )
    )
    theme.annotate(
        fig,
        f"<b style='font-size:17px'>{len(keys)}</b><br>holdings",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        size=11,
        color=p["subtext"],
    )

    subtitle = f"Largest position {values[0]:.1f}% - top 3 hold {values[:3].sum():.1f}%"
    if negative:
        subtitle += f" - {negative} negative weight(s) omitted"
    theme.apply(
        fig,
        title=title,
        subtitle=subtitle,
        height=height or 380,
        uniformtext={"minsize": 9, "mode": "hide"},
    )
    return fig


@_never_raises("Frontier weights chart could not be rendered")
def frontier_weights_area(
    frontier: pd.DataFrame | None,
    min_mean_weight_pct: float = 0.50,
    height: int | None = None,
) -> go.Figure:
    """Stacked area showing how the optimal mix changes along the frontier.

    Reading left to right is a tour from the minimum-variance portfolio to the
    highest-return one, and the bands show exactly which names the optimiser
    sells and which it buys to get there. It is the fastest way to see whether
    a frontier is a smooth reallocation or a knife-edge that dumps everything
    into one holding at the right-hand end.

    Parameters
    ----------
    frontier:
        Frontier frame with ``volatility`` plus one column per ticker holding
        that point's weight (``return`` and ``sharpe`` are ignored here).
    min_mean_weight_pct:
        Holdings whose average weight along the frontier is below this are
        folded into a muted "Other" band.

    Assumptions
    -----------
    Weights are long-only and sum to 100% at each frontier point; the bands are
    plotted at their raw values rather than normalised, so a frontier whose
    weights do not sum to one is visible as a gap under 100% instead of being
    silently rescaled.
    """
    if frontier is None or not isinstance(frontier, pd.DataFrame) or frontier.empty:
        return theme.empty_figure(
            "The efficient frontier is empty - there are no weights to trace."
        )
    front = frontier.copy()
    front.columns = [str(c) for c in front.columns]
    if "volatility" not in front.columns:
        return theme.empty_figure("Frontier data is missing the 'volatility' column.")

    weight_cols = _weight_columns(front)
    if not weight_cols:
        return theme.empty_figure(
            "The frontier carries no per-asset weight columns - only its summary "
            "statistics were stored."
        )

    front["volatility"] = pd.to_numeric(front["volatility"], errors="coerce")
    for col in weight_cols:
        front[col] = pd.to_numeric(front[col], errors="coerce")
    front = front.dropna(subset=["volatility"]).sort_values("volatility")
    front = front.drop_duplicates(subset=["volatility"], keep="first")
    if len(front) < 2:
        return theme.empty_figure("A weights profile needs at least two distinct frontier points.")

    weights = front[weight_cols].fillna(0.0)
    if float(np.nansum(np.abs(weights.to_numpy(dtype=float)))) <= 0:
        return theme.empty_figure(
            "Every weight along the frontier is zero or undefined - the optimiser "
            "returned no allocations to trace."
        )
    ret_scale = _scale_factor(
        front["volatility"].to_numpy(dtype=float),
        pd.to_numeric(front.get("return", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=float
        ),
    )
    x = front["volatility"].to_numpy(dtype=float) * ret_scale

    row_total = float(np.nanmean(weights.to_numpy(dtype=float).sum(axis=1)))
    w_scale = 100.0 if np.isfinite(row_total) and row_total <= 1.5 else 1.0
    weights = weights * w_scale

    means = weights.mean(axis=0).sort_values(ascending=False)
    major = [c for c in means.index if means[c] >= float(min_mean_weight_pct)]
    minor = [c for c in means.index if c not in major]
    if not major:
        major = list(means.index[:1])
        minor = [c for c in means.index if c not in major]

    p = theme.palette()
    fig = go.Figure()
    for i, col in enumerate(major):
        colour = theme.CATEGORICAL[i % len(theme.CATEGORICAL)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=weights[col].to_numpy(dtype=float),
                name=str(col),
                mode="lines",
                stackgroup="weights",
                line={"width": 0.6, "color": colour},
                fillcolor=_rgba(colour, 0.82),
                hovertemplate=f"{col}  %{{y:.1f}}%<extra></extra>",
            )
        )
    if minor:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=weights[minor].sum(axis=1).to_numpy(dtype=float),
                name=f"Other ({len(minor)} names)",
                mode="lines",
                stackgroup="weights",
                line={"width": 0.6, "color": p["muted"]},
                fillcolor=_rgba(p["muted"], 0.72),
                hovertemplate=f"Other ({len(minor)} names)  %{{y:.1f}}%<extra></extra>",
            )
        )

    theme.apply(
        fig,
        title="Weights along the efficient frontier",
        subtitle=(
            f"{len(front)} frontier points from minimum variance to maximum return - "
            f"{len(major)} holdings shown individually"
        ),
        height=height or 400,
        unified_hover=True,
    )
    fig.update_xaxes(title_text="Annualised volatility (%)")
    fig.update_yaxes(title_text="Weight (%)", range=[0, 100.5])
    theme.pct_axis(fig, axis="x", decimals=0)
    theme.pct_axis(fig, axis="y", decimals=0)
    theme.caption(
        fig,
        "Each vertical slice is one fully invested long-only portfolio. Sharp steps mean "
        "the optimiser is switching between near-identical solutions - a sign that the "
        "expected returns are estimated too precisely to trust.",
    )
    return fig
