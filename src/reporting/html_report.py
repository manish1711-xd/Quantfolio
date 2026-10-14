"""Self-contained HTML tearsheet for one analysis run.

The output is a single ``.html`` string with no external dependency other than
the Plotly bundle the embedded figures need: all styling lives in one inline
``<style>`` block, so the file can be e-mailed, committed to a repo or opened
from a USB stick and still render identically.

Three decisions worth naming, because they are what separates a report from a
screenshot of a dashboard:

* **Dark-on-light.** The terminal is dark; a document is not. Every figure
  handed in is *copied* and recoloured through the light half of
  :mod:`src.charts.theme` before it is embedded, so a chart authored for the
  app's dark surface does not arrive as pale grey text on white paper.
* **It prints as a PDF.** ``@media print`` sets an A4 page, repeats table
  headers across page boundaries and keeps cards, insight rows and figures
  whole via ``page-break-inside: avoid``. "Save as PDF" in any browser is the
  intended export path.
* **Everything data-derived is escaped.** Tickers, security names, sector
  labels, warnings and insight text all originate in a user-supplied CSV.
  They pass through ``html.escape`` without exception - a portfolio file is an
  untrusted document, and a report that renders it raw is one upload away from
  script injection.

Number formatting is delegated to :mod:`src.ui.formatters`, so the report and
the dashboard round, group and dash-out missing values in exactly the same way;
an undefined statistic renders as an em dash here too, never as ``0.00``.

No Streamlit import: this module runs in a cron job or a notebook just as well
as it does behind a download button.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from string import Template
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.charts import theme
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    CURRENCY_SYMBOLS,
    DATA_SOURCE,
    DEFAULT_BASE_CURRENCY,
    DEFAULT_VAR_CONFIDENCE,
    DISCLAIMER,
    INSIGHTS,
    PERIOD_LABELS,
    TRADING_DAYS,
    AppConfig,
)
from src.models import AnalysisResult
from src.ui import formatters as fmt

__all__ = ["build_html_report", "report_bytes", "report_filename"]

#: Height applied to an embedded figure that carries none of its own. Plotly's
#: default height of "100%" collapses to nothing inside a flow-layout document.
DEFAULT_FIGURE_HEIGHT = 420

#: A printed page has no mode bar and no Plotly logo; responsive resizing stays
#: on so the report still reflows on screen before it is printed.
PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

#: Paper neutrals. Everything semantic (gain, loss, accent, grid, ...) comes
#: from ``theme.LIGHT_PALETTE``; only the page/card substrate is defined here,
#: because a transparent chart background has no printed equivalent.
_PAPER: dict[str, str] = {
    "paper": "#ffffff",
    "canvas": "#f5f7fa",
    "surface_2": "#f8fafc",
    "warn_soft": "rgba(180, 83, 9, 0.10)",
    "shadow": "0 1px 2px rgba(15, 23, 42, 0.05), 0 10px 28px -20px rgba(15, 23, 42, 0.35)",
}

_LEVEL_LABELS: dict[str, str] = {
    "critical": "Critical",
    "warning": "Watch",
    "positive": "Strength",
    "info": "Note",
}
_LEVEL_ORDER: tuple[str, ...] = ("critical", "warning", "positive", "info")

#: Plotly lays a chart out for the screen it was rendered on. A print sheet is
#: narrower than a browser window, so the figures are re-laid-out on the
#: ``beforeprint`` event and restored afterwards - without this, charts print
#: cropped or overflowing the page box. Guarded so the page degrades quietly
#: when the CDN bundle is unreachable (offline printing still works).
_PRINT_RESIZE_SCRIPT = (
    "<script>"
    "(function(){"
    "function qfResize(){"
    "if(!window.Plotly){return;}"
    "var nodes=document.querySelectorAll('.plotly-graph-div');"
    "for(var i=0;i<nodes.length;i++){try{window.Plotly.Plots.resize(nodes[i]);}"
    "catch(e){}}"
    "}"
    "window.addEventListener('beforeprint',qfResize);"
    "window.addEventListener('afterprint',qfResize);"
    "})();"
    "</script>"
)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------
_CSS = Template(
    """
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: $canvas;
  color: $ink;
  font-family: $font;
  font-size: 14px;
  line-height: 1.5;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.page { max-width: 1160px; margin: 0 auto; padding: 26px 24px 56px; }
h1, h2, h3 { margin: 0; font-weight: 640; letter-spacing: -0.01em; }
p { margin: 0; }
ul { margin: 0; padding: 0; list-style: none; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.c-gain { color: $gain; }
.c-loss { color: $loss; }
.c-warn { color: $warn; }
.c-accent { color: $accent; }
.c-neutral, .c-muted { color: $faint; }

/* ---- masthead ---- */
.masthead { border-bottom: 2px solid $ink; padding-bottom: 15px; }
.brandline { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.brand { font-size: 11px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: $accent; }
.brand .ver { color: $faint; font-weight: 500; letter-spacing: 0.10em; }
.masthead h1 { font-size: 27px; margin-top: 9px; }
.masthead .tagline { margin-top: 3px; font-size: 13px; color: $muted; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 26px; margin-top: 14px; }
.meta li { display: flex; flex-direction: column; gap: 1px; }
.meta .k { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.09em; color: $faint; }
.meta .v { font-size: 13px; font-weight: 620; font-variant-numeric: tabular-nums; }
.printbtn {
  font: inherit; font-size: 11.5px; color: $muted; cursor: pointer;
  background: $paper; border: 1px solid $line; border-radius: 8px; padding: 6px 12px;
}
.printbtn:hover { color: $ink; border-color: $ink; }

/* ---- sections ---- */
.section { margin-top: 26px; }
.section > h2 {
  display: flex; align-items: center; gap: 10px; margin-bottom: 11px;
  font-size: 11.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.11em; color: $faint;
}
.section > h2::after { content: ""; flex: 1; height: 1px; background: $grid; }
.grid { display: grid; gap: 12px; }

/* ---- cards ---- */
.card {
  background: $paper; border: 1px solid $grid; border-radius: 12px;
  box-shadow: $shadow; padding: 13px 15px;
  page-break-inside: avoid; break-inside: avoid;
}
.kpi-grid { grid-template-columns: repeat(auto-fit, minmax(176px, 1fr)); }
.kpi { border-top: 3px solid $accent; }
.kpi.t-gain { border-top-color: $gain; }
.kpi.t-loss { border-top-color: $loss; }
.kpi.t-warn { border-top-color: $warn; }
.kpi.t-neutral { border-top-color: $line; }
.kpi .k { font-size: 9.5px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: $faint; }
.kpi .v { margin-top: 5px; font-size: 21px; font-weight: 660; font-variant-numeric: tabular-nums; line-height: 1.15; }
.kpi .d { margin-top: 3px; font-size: 12px; font-weight: 620; font-variant-numeric: tabular-nums; }
.kpi .f { margin-top: 5px; font-size: 10.5px; color: $faint; }

/* ---- statistic cards ---- */
.stats-grid { grid-template-columns: repeat(auto-fit, minmax(262px, 1fr)); }
.stat-title { margin-bottom: 7px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: $accent; }
.stat-row { display: flex; justify-content: space-between; gap: 14px; padding: 4px 0; border-bottom: 1px dashed $grid; font-size: 12.5px; }
.stat-row:last-child { border-bottom: 0; }
.stat-row .k { color: $muted; }
.stat-row .v { font-weight: 620; font-variant-numeric: tabular-nums; white-space: nowrap; }

/* ---- insights ---- */
.insights { display: grid; gap: 8px; }
.insight {
  display: grid; grid-template-columns: auto 1fr; gap: 11px; align-items: start;
  padding: 9px 13px; border: 1px solid $grid; border-left-width: 3px; border-radius: 10px;
  background: $paper; page-break-inside: avoid; break-inside: avoid;
}
.insight .lvl {
  font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 999px; white-space: nowrap; color: $paper;
}
.insight .txt { font-size: 13px; }
.insight .ev { display: block; margin-top: 3px; font-family: $mono; font-size: 11px; color: $muted; }
.i-critical { border-left-color: $loss; background: $loss_soft; }
.i-critical .lvl { background: $loss; }
.i-warning { border-left-color: $warn; background: $warn_soft; }
.i-warning .lvl { background: $warn; }
.i-positive { border-left-color: $gain; background: $gain_soft; }
.i-positive .lvl { background: $gain; }
.i-info { border-left-color: $accent; background: $accent_soft; }
.i-info .lvl { background: $accent; }
.notes { display: grid; gap: 5px; font-size: 12.5px; color: $muted; }
.notes li { padding-left: 16px; text-indent: -16px; }
.notes li::before { content: "! "; color: $warn; font-weight: 700; }

/* ---- tables ---- */
.table-wrap { overflow-x: auto; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.tbl caption { caption-side: bottom; text-align: left; padding-top: 9px; font-size: 11px; color: $faint; }
.tbl th {
  padding: 8px 10px; text-align: left; white-space: nowrap;
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: $faint; border-bottom: 1px solid $line;
}
.tbl td { padding: 6.5px 10px; border-bottom: 1px solid $grid; }
.tbl th.num, .tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
.tbl tbody tr:nth-child(even) { background: $surface2; }
.tbl tfoot td { padding-top: 8px; font-weight: 700; border-top: 1px solid $line; border-bottom: 0; background: $paper; }
.tbl .tk { font-family: $mono; font-weight: 620; white-space: nowrap; }
.tbl .nm { color: $muted; max-width: 230px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tbl .idx { color: $faint; font-variant-numeric: tabular-nums; }

/* ---- figures ---- */
.figures { display: grid; gap: 14px; }
.figure { padding: 12px 14px 6px; }
.figure h3 { font-size: 13.5px; }
.figure .sub { margin-top: 2px; font-size: 11px; color: $faint; }
.fig-fallback { padding: 30px 0; text-align: center; font-size: 12.5px; color: $muted; }

/* ---- footer ---- */
.footer { display: grid; gap: 6px; margin-top: 32px; padding-top: 14px; border-top: 1px solid $grid; font-size: 11.5px; color: $muted; }
.footer strong { color: $ink; }

@media (max-width: 640px) {
  .page { padding: 18px 14px 40px; }
  .masthead h1 { font-size: 22px; }
}

/* ---- print ---- */
@media print {
  @page { size: A4 portrait; margin: 12mm; }
  html, body { background: $paper; font-size: 10.5px; }
  .page { max-width: none; padding: 0; }
  .no-print { display: none; }
  .card { box-shadow: none; }
  .card, .kpi, .insight, .figure, .masthead, .footer {
    page-break-inside: avoid; break-inside: avoid;
  }
  .section > h2, h1, h3 { page-break-after: avoid; break-after: avoid; }
  .kpi-grid { grid-template-columns: repeat(4, 1fr); }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .table-wrap { overflow: visible; }
  thead { display: table-header-group; }
  tfoot { display: table-footer-group; }
  tr { page-break-inside: avoid; break-inside: avoid; }
}
"""
)


def _css() -> str:
    """Render the stylesheet against the light half of the chart palette."""
    p = theme.LIGHT_PALETTE
    return _CSS.substitute(
        font=theme.FONT_FAMILY,
        mono=theme.MONO_FAMILY,
        ink=p["text"],
        muted=p["subtext"],
        faint=p["muted"],
        gain=p["gain"],
        gain_soft=p["gain_fill"],
        loss=p["loss"],
        loss_soft=p["loss_fill"],
        accent=p["accent"],
        accent_soft=p["accent_fill"],
        warn=p["warning"],
        warn_soft=_PAPER["warn_soft"],
        grid=p["grid"],
        line=p["axis"],
        paper=_PAPER["paper"],
        canvas=_PAPER["canvas"],
        surface2=_PAPER["surface_2"],
        shadow=_PAPER["shadow"],
    )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _esc(value: Any) -> str:
    """HTML-escape anything, including quotes. ``None`` becomes an empty string."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _get(source: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an object attribute, defensively."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _dict(source: Any) -> dict:
    """Coerce a mapping-ish attribute into a plain dict (never raises)."""
    if isinstance(source, Mapping):
        return dict(source)
    return {}


def _frame(source: Any) -> pd.DataFrame:
    """Copy of ``source`` as a DataFrame; an empty frame when that is impossible.

    Always a copy: the report must never mutate the caller's positions frame.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if source is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(source)
    except Exception:  # noqa: BLE001 - a report must render whatever it is handed
        return pd.DataFrame()


def _num(value: Any) -> float:
    """Best-effort float conversion; NaN for anything unparseable."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _NAN
    return out


def _finite(value: Any) -> bool:
    """True only for a real, finite number."""
    return math.isfinite(_num(value))


def _col_sum(df: pd.DataFrame, column: str) -> float:
    """Column total, NaN when the column is absent or entirely missing."""
    if column not in df.columns:
        return _NAN
    return _num(pd.to_numeric(df[column], errors="coerce").sum(min_count=1))


def _tone(value: Any, invert: bool = False) -> str:
    """``fmt.delta_tone`` mapped onto this report's CSS tone names."""
    tone = fmt.delta_tone(value, invert=invert)
    return "neutral" if tone == "muted" else tone


def _text(value: Any) -> str:
    """Displayable label for a table cell; empty for None, NaN and NaT.

    ``str(float("nan"))`` is ``"nan"``, and a table full of the word "nan" is
    the classic tell of a report that never met real data.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    label = str(value).strip()
    return "" if label.lower() in {"nan", "none", "nat", "<na>"} else label


def _plain_text(markup: Any) -> str:
    """Strip HTML tags out of a Plotly title fragment."""
    return re.sub(r"<[^>]+>", "", str(markup)).strip()


def _timestamp(value: Any) -> str:
    """Human timestamp with its timezone label when the value carries one."""
    label = fmt.fmt_datetime(value)
    try:
        zone = value.tzname()
    except (AttributeError, TypeError, ValueError):
        zone = None
    return f"{label} {zone}" if zone else label


def _safe(builder: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Render one block, degrading to an HTML comment if it cannot be built.

    A tearsheet that loses its correlation table is still useful; a tearsheet
    that raises on a download click is not.
    """
    try:
        return builder(*args, **kwargs)
    except Exception:  # noqa: BLE001 - deliberate: no single block may kill the report
        return f"<!-- section unavailable: {builder.__name__} -->"


# ---------------------------------------------------------------------------
# figure preparation (dark app chart -> light printable chart)
# ---------------------------------------------------------------------------
def _build_colour_remap() -> tuple[re.Pattern[str], dict[str, str]]:
    """Map every dark-palette colour onto its light-palette counterpart."""
    mapping: dict[str, str] = {}
    for key, dark in theme.DARK_PALETTE.items():
        light = theme.LIGHT_PALETTE.get(key)
        if light and light.lower() != dark.lower():
            mapping[dark.lower()] = light
    # An empty alternation would match the empty string everywhere, so fall
    # back to a pattern that can never match if the two palettes ever converge.
    alternation = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    pattern = re.compile(alternation or r"(?!)", re.IGNORECASE)
    return pattern, mapping


_COLOUR_PATTERN, _COLOUR_MAP = _build_colour_remap()


def _recolour(value: Any) -> Any:
    """Recursively swap dark-palette colours for light ones in a figure spec.

    The substitution is a SINGLE regex pass, not a chain of ``str.replace``
    calls: the dark palette maps ``muted -> #94a3b8`` and ``#94a3b8 -> subtext``,
    so sequential replacement would rewrite its own output.
    """
    if isinstance(value, str):
        return _COLOUR_PATTERN.sub(lambda m: _COLOUR_MAP[m.group(0).lower()], value)
    if isinstance(value, Mapping):
        return {k: _recolour(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recolour(v) for v in value]
    return value


def _light_template() -> go.layout.Template:
    """The house template rendered in light mode, without leaving it enabled.

    ``theme.build_template`` reads the module-level mode, so the mode is
    flipped, read and restored inside a ``finally`` - the app keeps whatever
    mode it was rendering in.
    """
    previous = theme.mode()
    if previous == "light":
        return theme.build_template()
    theme.set_mode("light")
    try:
        return theme.build_template()
    finally:
        theme.set_mode(previous)


def _title_parts(fig: go.Figure) -> tuple[str, str]:
    """Plain-text ``(title, subtitle)`` of a figure titled through ``theme.apply``.

    ``theme.apply`` packs the subtitle into the title as a second ``<br>`` line
    wrapped in a styled span, so both halves are recovered by splitting on the
    break and stripping the markup.
    """
    title = getattr(getattr(fig.layout, "title", None), "text", None) or ""
    head, _, tail = str(title).partition("<br>")
    return _plain_text(head), _plain_text(tail)


def _print_ready(fig: go.Figure, height: int | None = None, drop_title: bool = False) -> go.Figure:
    """Return a light-mode copy of ``fig`` sized for a printed page.

    The input figure is never touched: the copy is rebuilt from
    ``fig.to_plotly_json()``, recoloured, and given the light template plus an
    explicit height (Plotly's ``100%`` default collapses inside a document).
    ``drop_title`` removes the in-figure title when the surrounding card
    already carries it, and reclaims the top margin it was using.
    """
    spec = _recolour(fig.to_plotly_json())
    out = go.Figure(spec)
    out.update_layout(
        template=_light_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if drop_title:
        out.update_layout(title=None, margin={"t": 28})
    if out.layout.height is None:
        out.update_layout(height=int(height or DEFAULT_FIGURE_HEIGHT))
    return out


def _figure_html(fig: go.Figure, include_js: bool, div_id: str, drop_title: bool = False) -> str:
    """Embed one figure, loading the Plotly CDN bundle only when asked to."""
    prepared = _print_ready(fig, drop_title=drop_title)
    return prepared.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config=PLOTLY_CONFIG,
        default_width="100%",
        default_height=f"{DEFAULT_FIGURE_HEIGHT}px",
        div_id=div_id,
        validate=False,
    )


# ---------------------------------------------------------------------------
# fragment builders
# ---------------------------------------------------------------------------
def _section(title: str, body: str, extra_class: str = "") -> str:
    """Wrap a rendered body in a titled section; empty bodies vanish.

    A body that is only the HTML comment left by :func:`_safe` is passed
    through unwrapped, so a failed block leaves a trace in the source but never
    a heading with nothing under it.
    """
    text = body.strip()
    if not text:
        return ""
    if text.startswith("<!--") and text.endswith("-->"):
        return text
    cls = f"section {extra_class}".strip()
    return f'<section class="{cls}">\n<h2>{_esc(title)}</h2>\n{text}\n</section>'


def _kpi(
    label: str,
    value: str,
    delta: str | None = None,
    tone: str = "neutral",
    delta_tone: str | None = None,
    foot: str | None = None,
) -> str:
    """One KPI tile: label, large value, optional delta and footnote."""
    parts = [
        f'<div class="card kpi t-{_esc(tone)}">',
        f'<div class="k">{_esc(label)}</div>',
        f'<div class="v">{_esc(value)}</div>',
    ]
    if delta:
        parts.append(f'<div class="d c-{_esc(delta_tone or tone)}">{_esc(delta)}</div>')
    if foot:
        parts.append(f'<div class="f">{_esc(foot)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _stat_card(title: str, rows: Sequence[tuple[str, str]]) -> str:
    """A titled card of label/value rows; returns "" when every row is missing."""
    body = "".join(
        f'<div class="stat-row"><span class="k">{_esc(k)}</span>'
        f'<span class="v">{_esc(v)}</span></div>'
        for k, v in rows
        if v
    )
    if not body:
        return ""
    return f'<div class="card">\n<div class="stat-title">{_esc(title)}</div>\n{body}\n</div>'


def _masthead(
    name: str,
    generated: Any,
    cfg: Any,
    summary: Mapping[str, Any],
    currency: str,
) -> str:
    """Report identity block: who, what, when and under which assumptions."""
    period = _get(cfg, "period")
    window = PERIOD_LABELS.get(str(period), str(period)) if period else None
    symbol = CURRENCY_SYMBOLS.get(currency, "")
    rate = _get(cfg, "risk_free_rate")

    meta: list[tuple[str, str]] = [("Generated", _timestamp(generated))]
    if window:
        meta.append(("History window", window))
    benchmark = _get(cfg, "benchmark_name")
    if benchmark:
        meta.append(("Benchmark", str(benchmark)))
    meta.append(("Reporting currency", f"{currency} {symbol}".strip()))
    meta.append(("Holdings", fmt.fmt_num(summary.get("n_holdings"), 0)))
    if rate is not None:
        meta.append(("Risk-free rate", fmt.fmt_ratio_pct(rate, 2)))

    items = "".join(
        f'<li><span class="k">{_esc(k)}</span><span class="v">{_esc(v)}</span></li>'
        for k, v in meta
    )
    return (
        '<header class="masthead">\n'
        '<div class="brandline">'
        f'<div class="brand">{_esc(APP_NAME)} <span class="ver">v{_esc(APP_VERSION)}</span></div>'
        '<button class="printbtn no-print" type="button" onclick="window.print()">'
        "Print / Save as PDF</button>"
        "</div>\n"
        f"<h1>{_esc(name)}</h1>\n"
        f'<p class="tagline">{_esc(APP_TAGLINE)}</p>\n'
        f'<ul class="meta">{items}</ul>\n'
        "</header>"
    )


def _kpi_grid(
    summary: Mapping[str, Any],
    risk: Mapping[str, Any],
    bench: Mapping[str, Any],
    score: Mapping[str, Any],
    ledger: Mapping[str, Any],
    cfg: Any,
    currency: str,
) -> str:
    """The twelve headline numbers, in the order a reviewer reads them."""

    def money(value: Any, decimals: int = 0, signed: bool = False) -> str:
        return fmt.fmt_money(value, currency, decimals, signed=signed)

    total_pnl = summary.get("total_pnl")
    total_ret = summary.get("total_return_pct")
    day_change = summary.get("day_change")
    day_pct = summary.get("day_change_pct")
    confidence = _num(risk.get("confidence", DEFAULT_VAR_CONFIDENCE))
    if not math.isfinite(confidence):
        confidence = DEFAULT_VAR_CONFIDENCE
    conf_label = fmt.fmt_pct(confidence * 100.0, 0)

    sharpe = _num(risk.get("sharpe"))
    sortino = _num(risk.get("sortino"))
    mdd = _num(risk.get("mdd"))
    xirr = _num(ledger.get("xirr")) if ledger else _NAN

    cards = [
        _kpi(
            "Portfolio value",
            money(summary.get("current_value")),
            delta=f"{fmt.fmt_delta(day_pct)} today" if _finite(day_pct) else None,
            tone="accent",
            delta_tone=_tone(day_pct),
            foot=f"Invested {money(summary.get('total_investment'))}",
        ),
        _kpi(
            "Unrealised P&L",
            money(total_pnl, signed=True),
            delta=fmt.fmt_pct(total_ret, 2, signed=True),
            tone=_tone(total_pnl),
            foot=(
                f"{fmt.fmt_num(summary.get('n_gainers'), 0)} up / "
                f"{fmt.fmt_num(summary.get('n_losers'), 0)} down"
            ),
        ),
        _kpi(
            "Change today",
            money(day_change, signed=True),
            delta=fmt.fmt_pct(day_pct, 2, signed=True),
            tone=_tone(day_change),
            foot="Versus the previous close, where reported",
        ),
        _kpi(
            "Annualised return",
            fmt.fmt_ratio_pct(risk.get("ann_return")),
            tone=_tone(risk.get("ann_return")),
            foot=f"Geometric, {TRADING_DAYS} trading days",
        ),
        _kpi(
            "Annualised volatility",
            fmt.fmt_ratio_pct(risk.get("ann_vol")),
            tone="neutral",
            foot=f"{fmt.fmt_num(risk.get('n_obs'), 0)} daily observations",
        ),
        _kpi(
            "Sharpe ratio",
            fmt.fmt_num(sharpe),
            tone="gain" if sharpe >= INSIGHTS.good_sharpe else "neutral",
            foot=f"Excess return per unit of total risk (>{INSIGHTS.good_sharpe:.2f} is good)",
        ),
        _kpi(
            "Sortino ratio",
            fmt.fmt_num(sortino),
            tone="gain" if sortino >= INSIGHTS.good_sortino else "neutral",
            foot=f"Downside deviation {fmt.fmt_ratio_pct(risk.get('downside_dev'))}",
        ),
        _kpi(
            "Max drawdown",
            fmt.fmt_ratio_pct(mdd),
            tone="warn" if mdd <= INSIGHTS.deep_drawdown else "neutral",
            foot=f"Ulcer index {fmt.fmt_pct(risk.get('ulcer'))}",
        ),
        _kpi(
            f"Daily VaR ({conf_label})",
            fmt.fmt_ratio_pct(risk.get("var")),
            tone="warn",
            foot=f"Historical; expected shortfall {fmt.fmt_ratio_pct(risk.get('cvar'))}",
        ),
        _kpi(
            "Risk score",
            f"{fmt.fmt_num(score.get('score'), 0)} / {fmt.fmt_num(score.get('max_points'), 0)}",
            delta=str(score.get("band") or ""),
            tone=(
                "neutral"
                if fmt.band_tone(str(score.get("band"))) == "muted"
                else fmt.band_tone(str(score.get("band")))
            ),
            foot="Rule-based rubric, not a regulatory measure",
        ),
        _kpi(
            f"Beta vs {_get(cfg, 'benchmark_name') or 'benchmark'}",
            fmt.fmt_num(bench.get("beta")),
            tone="neutral",
            foot=f"Jensen alpha {fmt.fmt_ratio_pct(bench.get('alpha'))}",
        ),
        (
            _kpi(
                "Money-weighted return",
                fmt.fmt_ratio_pct(xirr),
                tone=_tone(xirr),
                foot="XIRR over the trade ledger, ACT/365",
            )
            if math.isfinite(xirr)
            else _kpi(
                "Calmar ratio",
                fmt.fmt_num(risk.get("calmar")),
                tone="neutral",
                foot="Annualised return per unit of max drawdown",
            )
        ),
    ]
    return f'<div class="grid kpi-grid">{"".join(cards)}</div>'


def _stats_grid(
    summary: Mapping[str, Any],
    risk: Mapping[str, Any],
    stats: Mapping[str, Any],
    bench: Mapping[str, Any],
    ledger: Mapping[str, Any],
    cfg: Any,
    currency: str,
) -> str:
    """Full statistic tables: return, risk, ratios, concentration, benchmark.

    Units follow ``risk.risk_summary``: decimals for return/volatility/VaR,
    percent for the Ulcer index, pure ratios for Sharpe and friends.
    """
    realised = _dict(ledger.get("realised_summary"))
    xirr = ledger.get("xirr")

    returns_rows: list[tuple[str, str]] = [
        ("Cumulative return", fmt.fmt_pct(stats.get("cumulative_pct"))),
        ("Annualised return", fmt.fmt_ratio_pct(risk.get("ann_return"))),
        ("Best month", fmt.fmt_pct(stats.get("best_month_pct"), 2, signed=True)),
        ("Worst month", fmt.fmt_pct(stats.get("worst_month_pct"), 2, signed=True)),
        ("Best day", fmt.fmt_ratio_pct(risk.get("best_day"), 2, signed=True)),
        ("Worst day", fmt.fmt_ratio_pct(risk.get("worst_day"), 2, signed=True)),
    ]
    total_months = stats.get("total_months")
    if total_months:
        returns_rows.append(
            (
                "Positive months",
                f"{fmt.fmt_num(stats.get('positive_months'), 0)} of {fmt.fmt_num(total_months, 0)}",
            )
        )
    if stats.get("longest_win_streak") is not None:
        returns_rows.append(
            ("Longest winning streak", f"{fmt.fmt_num(stats.get('longest_win_streak'), 0)} days")
        )
    if xirr is not None:
        returns_rows.append(("Money-weighted return (XIRR)", fmt.fmt_ratio_pct(xirr)))
    if realised:
        realised_pnl = fmt.fmt_money(realised.get("total_realised"), currency, 0, signed=True)
        returns_rows.append(("Realised P&L", realised_pnl))
        returns_rows.append(
            ("Win rate on closed trades", fmt.fmt_ratio_pct(realised.get("win_rate")))
        )

    risk_rows: list[tuple[str, str]] = [
        ("Annualised volatility", fmt.fmt_ratio_pct(risk.get("ann_vol"))),
        ("Downside deviation", fmt.fmt_ratio_pct(risk.get("downside_dev"))),
        ("Maximum drawdown", fmt.fmt_ratio_pct(risk.get("mdd"))),
        ("Ulcer index", fmt.fmt_pct(risk.get("ulcer"))),
        ("Daily VaR (historical)", fmt.fmt_ratio_pct(risk.get("var"))),
        ("Daily CVaR (expected shortfall)", fmt.fmt_ratio_pct(risk.get("cvar"))),
        ("Daily VaR (Gaussian)", fmt.fmt_ratio_pct(risk.get("var_gaussian"))),
        ("Daily VaR (Cornish-Fisher)", fmt.fmt_ratio_pct(risk.get("var_cornish_fisher"))),
        ("Skewness", fmt.fmt_num(risk.get("skew"))),
        ("Excess kurtosis", fmt.fmt_num(risk.get("kurtosis"))),
        ("Observations", fmt.fmt_num(risk.get("n_obs"), 0)),
    ]

    ratio_rows: list[tuple[str, str]] = [
        ("Sharpe", fmt.fmt_num(risk.get("sharpe"))),
        ("Sortino", fmt.fmt_num(risk.get("sortino"))),
        ("Calmar", fmt.fmt_num(risk.get("calmar"))),
        ("Omega", fmt.fmt_num(risk.get("omega"))),
        ("Martin (return / ulcer)", fmt.fmt_num(risk.get("martin"))),
        ("Tail ratio", fmt.fmt_num(risk.get("tail_ratio"))),
        ("Hit ratio (positive days)", fmt.fmt_ratio_pct(risk.get("hit_ratio"), 1)),
        ("Gain to pain", fmt.fmt_num(risk.get("gain_to_pain"))),
    ]

    best = summary.get("best")
    worst = summary.get("worst")
    largest = summary.get("largest_holding")
    concentration_rows: list[tuple[str, str]] = [
        ("Holdings", fmt.fmt_num(summary.get("n_holdings"), 0)),
        ("Effective holdings (1 / HHI)", fmt.fmt_num(summary.get("effective_holdings"), 1)),
        ("Herfindahl index", fmt.fmt_num(summary.get("hhi"), 3)),
        ("Top 5 weight", fmt.fmt_pct(summary.get("top5_weight_pct"), 1)),
    ]
    if isinstance(largest, (tuple, list)) and len(largest) == 2:
        concentration_rows.append(
            ("Largest position", f"{largest[0]} {fmt.fmt_pct(largest[1], 1)}")
        )
    if isinstance(best, (tuple, list)) and len(best) == 2:
        concentration_rows.append(
            ("Best performer", f"{best[0]} {fmt.fmt_pct(best[1], 2, signed=True)}")
        )
    if isinstance(worst, (tuple, list)) and len(worst) == 2:
        concentration_rows.append(
            ("Worst performer", f"{worst[0]} {fmt.fmt_pct(worst[1], 2, signed=True)}")
        )

    cards = [
        _stat_card("Return", returns_rows),
        _stat_card("Risk", risk_rows),
        _stat_card("Risk-adjusted ratios", ratio_rows),
        _stat_card("Concentration", concentration_rows),
    ]

    if bench.get("ok"):
        name = str(_get(cfg, "benchmark_name") or "Benchmark")
        bench_rows: list[tuple[str, str]] = [
            ("Benchmark", name),
            ("Benchmark return", fmt.fmt_ratio_pct(bench.get("benchmark_return"))),
            ("Portfolio return", fmt.fmt_ratio_pct(bench.get("portfolio_return"))),
            ("Excess return", fmt.fmt_ratio_pct(bench.get("excess_return"), 2, signed=True)),
            ("Beta", fmt.fmt_num(bench.get("beta"))),
            ("Jensen alpha", fmt.fmt_ratio_pct(bench.get("alpha"), 2, signed=True)),
        ]
        optional: tuple[tuple[str, str, Callable[[Any], str]], ...] = (
            ("tracking_error", "Tracking error", lambda v: fmt.fmt_ratio_pct(v)),
            ("information_ratio", "Information ratio", lambda v: fmt.fmt_num(v)),
            ("treynor", "Treynor ratio", lambda v: fmt.fmt_num(v)),
            ("m2", "M-squared", lambda v: fmt.fmt_ratio_pct(v)),
            ("correlation", "Correlation", lambda v: fmt.fmt_num(v)),
            ("r_squared", "R-squared", lambda v: fmt.fmt_num(v)),
            ("up_capture", "Up capture", lambda v: fmt.fmt_multiple(v)),
            ("down_capture", "Down capture", lambda v: fmt.fmt_multiple(v)),
            ("capture_ratio", "Capture ratio", lambda v: fmt.fmt_multiple(v)),
            ("benchmark_mdd", "Benchmark max drawdown", lambda v: fmt.fmt_ratio_pct(v)),
            ("n_days", "Overlapping days", lambda v: fmt.fmt_num(v, 0)),
        )
        for key, label, formatter in optional:
            if key in bench:
                bench_rows.append((label, formatter(bench.get(key))))
        cards.append(_stat_card(f"Versus {name}", bench_rows))
    elif bench.get("reason"):
        cards.append(_stat_card("Benchmark", [("Not compared", str(bench["reason"]))]))

    body = "".join(c for c in cards if c)
    return f'<div class="grid stats-grid">{body}</div>' if body else ""


def _insight_list(insights: Any) -> str:
    """Rule-derived observations, ordered critical -> warning -> positive -> info."""
    items = list(insights) if isinstance(insights, (list, tuple)) else []
    if not items:
        return ""

    def rank(item: Any) -> int:
        level = str(_get(item, "level", "info") or "info").lower()
        return _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else len(_LEVEL_ORDER)

    rendered: list[str] = []
    for item in sorted(items, key=rank):
        level = str(_get(item, "level", "info") or "info").lower()
        if level not in _LEVEL_LABELS:
            level = "info"
        text = str(_get(item, "text", "") or "")
        if not text:
            continue
        evidence = str(_get(item, "evidence", "") or "")
        evidence_html = f'<span class="ev">{_esc(evidence)}</span>' if evidence else ""
        rendered.append(
            f'<li class="insight i-{level}">'
            f'<span class="lvl">{_esc(_LEVEL_LABELS[level])}</span>'
            f'<span class="txt">{_esc(text)}{evidence_html}</span>'
            "</li>"
        )
    return f'<ul class="insights">{"".join(rendered)}</ul>' if rendered else ""


def _warning_list(warnings: Any) -> str:
    """Data-quality caveats raised by the pipeline, verbatim and escaped."""
    if not isinstance(warnings, (list, tuple)):
        return ""
    items = [str(w) for w in warnings if str(w).strip()]
    if not items:
        return ""
    body = "".join(f"<li>{_esc(w)}</li>" for w in items)
    return f'<div class="card"><ul class="notes">{body}</ul></div>'


def _holdings_table(positions: Any, currency: str) -> str:
    """Position-level table, heaviest holding first, with a totals row.

    Prices stay in each instrument's native currency (the convention set by
    ``analytics.positions.build_positions``); value, P&L and weight are in the
    reporting currency.
    """
    df = _frame(positions)
    if df.empty or "ticker" not in df.columns:
        return ""

    if "current_value" in df.columns:
        df = df.sort_values("current_value", ascending=False, na_position="last")

    currencies = (
        sorted({str(c) for c in df["currency"].dropna().unique()})
        if "currency" in df.columns
        else []
    )
    show_ccy = len(currencies) > 1

    text_headers = {"#", "Ticker", "Security", "Sector", "Ccy"}
    headers = ["#", "Ticker", "Security", "Sector"]
    if show_ccy:
        headers.append("Ccy")
    headers += ["Quantity", "Avg cost", "Last price", "Value", "Weight", "P&L", "Return"]
    head = "".join(
        f"<th>{_esc(h)}</th>" if h in text_headers else f'<th class="num">{_esc(h)}</th>'
        for h in headers
    )

    rows: list[str] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        native = _text(row.get("currency")) or currency
        pnl = row.get("pnl")
        pnl_pct = row.get("pnl_pct")
        cells = [
            f'<td class="idx">{i}</td>',
            f'<td class="tk">{_esc(_text(row.get("ticker")))}</td>',
            f'<td class="nm">{_esc(_text(row.get("name")))}</td>',
            f'<td class="nm">{_esc(_text(row.get("sector")))}</td>',
        ]
        if show_ccy:
            cells.append(f'<td class="nm">{_esc(native)}</td>')
        cells += [
            f'<td class="num">{_esc(fmt.fmt_shares(row.get("quantity")))}</td>',
            f'<td class="num">{_esc(fmt.fmt_money(row.get("buy_price"), native, 2))}</td>',
            f'<td class="num">{_esc(fmt.fmt_money(row.get("current_price"), native, 2))}</td>',
            f'<td class="num">{_esc(fmt.fmt_money(row.get("current_value"), currency, 0))}</td>',
            f'<td class="num">{_esc(fmt.fmt_pct(row.get("weight_pct"), 1))}</td>',
            f'<td class="num c-{_tone(pnl)}">'
            f"{_esc(fmt.fmt_money(pnl, currency, 0, signed=True))}</td>",
            f'<td class="num c-{_tone(pnl_pct)}">{_esc(fmt.fmt_pct(pnl_pct, 2, signed=True))}</td>',
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")

    total_investment = _col_sum(df, "investment")
    total_value = _col_sum(df, "current_value")
    total_pnl = total_value - total_investment
    total_return = (
        total_pnl / total_investment * 100.0
        if math.isfinite(total_investment) and total_investment > 0
        else _NAN
    )
    span = 4 + (1 if show_ccy else 0) + 3
    foot = (
        f'<tr><td colspan="{span}">Total ({len(df)} holdings)</td>'
        f'<td class="num">{_esc(fmt.fmt_money(total_value, currency, 0))}</td>'
        f'<td class="num">{_esc(fmt.fmt_pct(_col_sum(df, "weight_pct"), 1))}</td>'
        f'<td class="num c-{_tone(total_pnl)}">'
        f"{_esc(fmt.fmt_money(total_pnl, currency, 0, signed=True))}</td>"
        f'<td class="num c-{_tone(total_return)}">'
        f"{_esc(fmt.fmt_pct(total_return, 2, signed=True))}</td></tr>"
    )

    caption = (
        "Quantity, average cost and last price are in each instrument's native "
        f"currency; value, P&amp;L and weight are in {_esc(currency)}. Weights "
        "are shares of current market value."
    )
    return (
        '<div class="card"><div class="table-wrap">'
        '<table class="tbl">'
        f"<caption>{caption}</caption>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"<tfoot>{foot}</tfoot>"
        "</table></div></div>"
    )


def _figures(figures: Any) -> str:
    """Embed each supplied figure under its heading.

    The Plotly CDN bundle is requested by the FIRST figure that embeds
    successfully and by no other, so the script tag appears exactly once.
    """
    if not figures:
        return ""
    if isinstance(figures, Mapping):
        items = list(figures.items())
    else:
        try:
            items = [(k, v) for k, v in figures]
        except (TypeError, ValueError):
            return ""

    blocks: list[str] = []
    needs_js = True
    for index, (heading, fig) in enumerate(items):
        if not isinstance(fig, go.Figure):
            continue
        label = str(heading)
        title, subtitle = _title_parts(fig)
        duplicate = bool(title) and title.casefold() == label.casefold()
        caption = f'<p class="sub">{_esc(subtitle)}</p>' if duplicate and subtitle else ""
        try:
            fragment = _figure_html(fig, needs_js, f"qf-figure-{index}", duplicate)
            needs_js = False
        except Exception:  # noqa: BLE001 - one bad figure must not lose the report
            fragment = '<p class="fig-fallback">This chart could not be rendered.</p>'
        blocks.append(
            f'<figure class="card figure"><h3>{_esc(label)}</h3>{caption}{fragment}</figure>'
        )
    if not blocks:
        return ""
    return f'<div class="figures">{"".join(blocks)}</div>{_PRINT_RESIZE_SCRIPT}'


def _footer(cfg: Any, risk: Mapping[str, Any], bench: Mapping[str, Any], generated: Any) -> str:
    """Disclaimer, data source and the conventions every number above assumes."""
    confidence = _num(risk.get("confidence", DEFAULT_VAR_CONFIDENCE))
    if not math.isfinite(confidence):
        confidence = DEFAULT_VAR_CONFIDENCE
    conventions = (
        f"Returns are geometrically annualised over {TRADING_DAYS} trading days. "
        "VaR and expected shortfall are one-day "
        f"{fmt.fmt_pct(confidence * 100.0, 0)} historical estimates, stated as "
        "negative decimals. Drawdowns are peak-to-trough on daily closes. "
        f"Risk-free rate {fmt.fmt_ratio_pct(_get(cfg, 'risk_free_rate'), 2)}."
    )
    lines = [
        f"<p><strong>Disclaimer.</strong> {_esc(DISCLAIMER)}</p>",
        f"<p><strong>Data source.</strong> {_esc(DATA_SOURCE)}</p>",
        f"<p><strong>Conventions.</strong> {_esc(conventions)}</p>",
    ]
    methodology = bench.get("alpha_methodology")
    if methodology:
        lines.append(f"<p><strong>Alpha.</strong> {_esc(methodology)}</p>")
    lines.append(
        f"<p>{_esc(APP_NAME)} {_esc(APP_VERSION)} &middot; {_esc(APP_TAGLINE)} "
        f"&middot; generated {_esc(_timestamp(generated))}</p>"
    )
    return f'<footer class="footer">{"".join(lines)}</footer>'


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _base_currency(cfg: Any, summary: Mapping[str, Any]) -> str:
    """Reporting currency: the config's, then the summary's, then the default."""
    for candidate in (
        _get(cfg, "base_currency"),
        summary.get("base_currency"),
        DEFAULT_BASE_CURRENCY,
    ):
        if candidate:
            return str(candidate).upper()
    return DEFAULT_BASE_CURRENCY


def _default_name(result: Any) -> str:
    """Portfolio name from the result (or its positions frame) if it carries one."""
    for candidate in (
        _get(result, "portfolio_name"),
        _get(result, "name"),
    ):
        if candidate:
            return str(candidate)
    positions = _get(result, "positions")
    attrs = getattr(positions, "attrs", None)
    if isinstance(attrs, Mapping):
        for key in ("portfolio_name", "name"):
            if attrs.get(key):
                return str(attrs[key])
    return "Untitled portfolio"


def build_html_report(
    result: AnalysisResult,
    config: AppConfig | None = None,
    figures: Mapping[str, go.Figure] | None = None,
    portfolio_name: str | None = None,
) -> str:
    """Render one analysis run as a standalone, printable HTML tearsheet.

    Shows, in order: an identity header (portfolio name, generation timestamp,
    history window, benchmark, reporting currency, risk-free rate), any
    data-quality warnings the pipeline raised, a twelve-tile KPI grid, full
    return / risk / ratio / concentration / benchmark statistic tables, the
    rule-derived insight list, the holdings table, then every supplied Plotly
    figure under its dictionary key as a heading, and a footer carrying the
    disclaimer, the data source and the conventions behind every number.

    Assumptions
    -----------
    * ``result`` follows :class:`~src.models.AnalysisResult`; every section is
      optional and silently omitted when its data is absent, so a run with no
      benchmark or no trade ledger still produces a complete document.
    * Units follow ``analytics.risk.risk_summary`` - decimals for returns,
      volatility, drawdown and VaR; percent for the Ulcer index and for
      ``summary``/``return_stats`` percentages.
    * ``config`` is an :class:`~src.config.AppConfig` (or anything exposing
      ``period``, ``benchmark_name``, ``base_currency`` and
      ``risk_free_rate``); when omitted, ``result.config`` is used.
    * ``figures`` maps a heading to a ``go.Figure``. The first figure embeds
      the Plotly CDN bundle and the rest do not, so the script loads once.
      Figures are copied and recoloured for light paper; the caller's figures
      are not modified.
    * Nothing about the input is trusted: every data-derived string is escaped.

    Returns a complete ``<!DOCTYPE html>`` document as a ``str``.
    """
    cfg = config if config is not None else _get(result, "config")
    summary = _dict(_get(result, "summary"))
    risk = _dict(_get(result, "risk"))
    bench = _dict(_get(result, "benchmark"))
    score = _dict(_get(result, "score"))
    stats = _dict(_get(result, "return_stats"))
    ledger = _dict(_get(result, "ledger"))
    generated = _get(result, "generated_at") or datetime.now(timezone.utc)
    currency = _base_currency(cfg, summary)
    name = portfolio_name or _default_name(result)

    blocks = [
        _safe(_masthead, name, generated, cfg, summary, currency),
        _section("Data notes", _safe(_warning_list, _get(result, "warnings"))),
        _section(
            "Key figures",
            _safe(_kpi_grid, summary, risk, bench, score, ledger, cfg, currency),
        ),
        _section(
            "Statistics",
            _safe(_stats_grid, summary, risk, stats, bench, ledger, cfg, currency),
        ),
        _section("What the numbers say", _safe(_insight_list, _get(result, "insights"))),
        _section("Holdings", _safe(_holdings_table, _get(result, "positions"), currency)),
        _section("Charts", _safe(_figures, figures)),
        _safe(_footer, cfg, risk, bench, generated),
    ]
    body = "\n".join(block for block in blocks if block)

    title = f"{name} - {APP_NAME} tearsheet"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="generator" content="{_esc(APP_NAME)} {_esc(APP_VERSION)}">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_css()}</style>\n"
        "</head>\n"
        "<body>\n"
        f'<main class="page">\n{body}\n</main>\n'
        "</body>\n"
        "</html>\n"
    )


def report_bytes(
    result: AnalysisResult,
    config: AppConfig | None = None,
    figures: Mapping[str, go.Figure] | None = None,
    portfolio_name: str | None = None,
) -> bytes:
    """UTF-8 encoded :func:`build_html_report`, ready for a download button.

    The document declares ``<meta charset="utf-8">``, so currency symbols and
    the em dashes used for missing values survive the round trip to disk.
    """
    return build_html_report(result, config, figures, portfolio_name).encode("utf-8")


def report_filename(portfolio_name: str | None = None, generated_at: Any = None) -> str:
    """Filesystem-safe download name, e.g. ``quantfolio-core-equity-20260901.html``.

    The portfolio name is slugified to ASCII word characters so the file lands
    intact on Windows, macOS and Linux; the date stamp keeps successive
    downloads from overwriting one another.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(portfolio_name or "portfolio")).strip("-")
    slug = slug.lower() or "portfolio"
    stamp = generated_at if isinstance(generated_at, datetime) else datetime.now(timezone.utc)
    return f"quantfolio-{slug}-{stamp.strftime('%Y%m%d')}.html"
