"""Plotly design system: palette, registered template and layout helpers.

Every figure in the application is built through :func:`apply`, which means the
whole chart language - typography, grid weight, hover behaviour, margins,
legend placement - is changed in exactly one place.

Two deliberate choices worth naming:

* **Transparent backgrounds.** Figures paint no background of their own, so
  they sit on the app's card surface and re-theme for free when the app
  switches between dark and light.
* **Colourblind-safe categorical colours.** The qualitative ramp is derived
  from Paul Tol's bright/light schemes, which stay distinguishable under
  deuteranopia and protanopia. Red/green is used *only* for semantic gain/loss,
  never to distinguish one holding from another - and gain/loss are always
  reinforced by position or sign, never by hue alone.

No Streamlit import: these builders work in a notebook, a Flask app or the
HTML report writer just as well as in the dashboard.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE_NAME = "quantfolio"

# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------
DARK_PALETTE: dict[str, str] = {
    "gain": "#22c55e",
    "gain_fill": "rgba(34, 197, 94, 0.16)",
    "loss": "#f43f5e",
    "loss_fill": "rgba(244, 63, 94, 0.16)",
    "accent": "#6366f1",
    "accent_fill": "rgba(99, 102, 241, 0.16)",
    "accent_2": "#22d3ee",
    "warning": "#f59e0b",
    "benchmark": "#94a3b8",
    "muted": "#64748b",
    "text": "#e6edf7",
    "subtext": "#94a3b8",
    "grid": "rgba(148, 163, 184, 0.14)",
    "axis": "rgba(148, 163, 184, 0.35)",
    "surface": "rgba(0,0,0,0)",
    "hover_bg": "#1b2231",
}

LIGHT_PALETTE: dict[str, str] = {
    "gain": "#15803d",
    "gain_fill": "rgba(21, 128, 61, 0.14)",
    "loss": "#be123c",
    "loss_fill": "rgba(190, 18, 60, 0.14)",
    "accent": "#4f46e5",
    "accent_fill": "rgba(79, 70, 229, 0.14)",
    "accent_2": "#0891b2",
    "warning": "#b45309",
    "benchmark": "#64748b",
    "muted": "#94a3b8",
    "text": "#0f172a",
    "subtext": "#52627a",
    "grid": "rgba(15, 23, 42, 0.08)",
    "axis": "rgba(15, 23, 42, 0.25)",
    "surface": "rgba(0,0,0,0)",
    "hover_bg": "#ffffff",
}

#: Qualitative ramp (Paul Tol bright + light). Safe for the common forms of
#: colour-vision deficiency and legible on both dark and light surfaces.
CATEGORICAL: list[str] = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#EE8866",  # orange
    "#44AA99",  # teal
    "#99DDFF",  # pale blue
    "#BBCC33",  # olive
    "#DDCC77",  # sand
    "#AAAAAA",  # grey
]

#: Monotone ramp for magnitude (weights, exposures, volume).
SEQUENTIAL: list[str] = [
    "#0b1b3a",
    "#123159",
    "#194878",
    "#216098",
    "#2b79b8",
    "#4a93cd",
    "#74acdb",
    "#a0c6e8",
    "#cddff4",
    "#eef5fc",
]

#: Diverging ramp for signed quantities (correlation, monthly returns).
#: Blue-white-red rather than green-red so it is readable for the ~8% of men
#: with red-green colour-vision deficiency.
DIVERGING: list[list[Any]] = [
    [0.00, "#67001f"],
    [0.15, "#b2182b"],
    [0.30, "#d6604d"],
    [0.42, "#f4a582"],
    [0.50, "#f7f7f7"],
    [0.58, "#92c5de"],
    [0.70, "#4393c3"],
    [0.85, "#2166ac"],
    [1.00, "#053061"],
]

FONT_FAMILY = 'Inter, "Segoe UI", Roboto, system-ui, -apple-system, sans-serif'
MONO_FAMILY = '"JetBrains Mono", ui-monospace, Consolas, monospace'

_mode = "dark"


# ---------------------------------------------------------------------------
# mode & palette access
# ---------------------------------------------------------------------------
def set_mode(mode: str) -> None:
    """Switch the chart palette between ``"dark"`` and ``"light"``.

    The UI layer calls this once per render from the resolved Streamlit theme,
    then every figure built afterwards matches the page.
    """
    global _mode
    if mode in {"dark", "light"} and mode != _mode:
        _mode = mode
        register_template(force=True)
    elif mode in {"dark", "light"}:
        _mode = mode


def mode() -> str:
    return _mode


def palette() -> dict[str, str]:
    """Semantic colours for the active mode."""
    return DARK_PALETTE if _mode == "dark" else LIGHT_PALETTE


#: Backwards-compatible aliases used throughout the chart modules.
def gain() -> str:
    return palette()["gain"]


def loss() -> str:
    return palette()["loss"]


def accent() -> str:
    return palette()["accent"]


def sign_colors(values) -> list[str]:
    """Per-bar gain/loss colours for a sequence of signed numbers."""
    p = palette()
    out: list[str] = []
    for v in values:
        try:
            out.append(p["gain"] if float(v) >= 0 else p["loss"])
        except (TypeError, ValueError):
            out.append(p["muted"])
    return out


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------
def build_template() -> go.layout.Template:
    """Construct the Plotly template for the active mode."""
    p = palette()
    axis = {
        "gridcolor": p["grid"],
        "linecolor": p["axis"],
        "zerolinecolor": p["grid"],
        "tickfont": {"size": 11, "color": p["subtext"]},
        "title": {"font": {"size": 12, "color": p["subtext"]}},
        "showline": False,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": "rgba(0,0,0,0)",
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_FAMILY, "size": 12, "color": p["text"]},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": CATEGORICAL,
            "colorscale": {"sequential": SEQUENTIAL, "diverging": DIVERGING},
            "title": {
                "font": {"size": 15, "color": p["text"]},
                "x": 0.01,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
            },
            "margin": {"l": 56, "r": 24, "t": 62, "b": 44},
            "hoverlabel": {
                "bgcolor": p["hover_bg"],
                "bordercolor": p["grid"],
                "font": {"family": FONT_FAMILY, "size": 12, "color": p["text"]},
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0,
                "font": {"size": 11, "color": p["subtext"]},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "xaxis": {**axis, "showgrid": False},
            "yaxis": {**axis, "showgrid": True},
            "separators": ".,",
        }
    )


def register_template(force: bool = False) -> None:
    """Register (or refresh) the template with Plotly. Idempotent."""
    if force or TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = build_template()


register_template()


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------
def apply(
    fig: go.Figure,
    title: str | None = None,
    subtitle: str | None = None,
    height: int | None = None,
    legend: bool = True,
    unified_hover: bool = False,
    **layout: Any,
) -> go.Figure:
    """Apply the house template plus a title/subtitle block to any figure.

    ``subtitle`` renders as a muted second line under the title - the right
    place for the assumption or window a chart depends on, so the caveat
    travels with the chart into exports and screenshots.
    """
    register_template()
    p = palette()

    title_spec: dict[str, Any] | None = None
    if title:
        text = title
        if subtitle:
            text = (
                f"{title}<br><span style='font-size:11.5px;color:{p['subtext']};"
                f"font-weight:400'>{subtitle}</span>"
            )
        title_spec = {"text": text}

    fig.update_layout(
        template=TEMPLATE_NAME,
        showlegend=legend,
        **({"title": title_spec} if title_spec else {}),
        **({"height": height} if height else {}),
        **({"hovermode": "x unified"} if unified_hover else {}),
        **layout,
    )
    if subtitle and title:
        # give the two-line title room to breathe
        fig.update_layout(margin={"t": 82})
    return fig


def money_axis(fig: go.Figure, currency_symbol: str = "", axis: str = "y") -> go.Figure:
    """Prefix an axis with a currency symbol and use SI-style compaction."""
    spec = {"tickprefix": currency_symbol, "tickformat": "~s", "separatethousands": True}
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**spec)
    return fig


def pct_axis(fig: go.Figure, axis: str = "y", decimals: int = 1) -> go.Figure:
    """Suffix an axis with % (values are expected already in percent units)."""
    spec = {"ticksuffix": "%", "tickformat": f".{decimals}f"}
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**spec)
    return fig


def add_zero_line(fig: go.Figure, axis: str = "y") -> go.Figure:
    """Emphasise the zero line - essential on any signed bar or area chart."""
    p = palette()
    spec = {"zeroline": True, "zerolinecolor": p["axis"], "zerolinewidth": 1}
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**spec)
    return fig


def add_reference_line(
    fig: go.Figure,
    value: float,
    label: str = "",
    axis: str = "y",
    color: str | None = None,
    dash: str = "dot",
) -> go.Figure:
    """Horizontal (or vertical) annotated reference line."""
    p = palette()
    kwargs: dict[str, Any] = {
        "line_width": 1.2,
        "line_dash": dash,
        "line_color": color or p["muted"],
    }
    if label:
        kwargs |= {
            "annotation_text": label,
            "annotation_position": "top left" if axis == "y" else "top",
            "annotation_font_size": 10,
            "annotation_font_color": color or p["subtext"],
        }
    if axis == "y":
        fig.add_hline(y=value, **kwargs)
    else:
        fig.add_vline(x=value, **kwargs)
    return fig


def annotate(fig: go.Figure, text: str, x: Any, y: Any, **kw: Any) -> go.Figure:
    """Small muted annotation anchored to data coordinates."""
    p = palette()
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=kw.pop("showarrow", False),
        font={"size": kw.pop("size", 11), "color": kw.pop("color", p["subtext"])},
        **kw,
    )
    return fig


def caption(fig: go.Figure, text: str) -> go.Figure:
    """Footnote pinned below the plotting area (methodology, assumptions)."""
    p = palette()
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 10.5, "color": p["subtext"]},
    )
    fig.update_layout(margin={"b": 76})
    return fig


def empty_figure(
    message: str = "Not enough data to plot this yet.", height: int = 260
) -> go.Figure:
    """Styled placeholder used wherever a chart cannot honestly be drawn."""
    p = palette()
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 13, "color": p["subtext"], "family": FONT_FAMILY},
    )
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=height,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        showlegend=False,
    )
    return fig
