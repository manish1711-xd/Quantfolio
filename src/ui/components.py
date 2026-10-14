"""Reusable Streamlit building blocks.

Streamlit's stock widgets are functional but visually flat, and ``st.metric``
cannot show a trend. These components wrap the design tokens from
:mod:`src.ui.theme` into a small vocabulary - KPI card, insight card, badge,
empty state - so every page looks like one product.

Security note: every value that reaches ``unsafe_allow_html`` is passed through
``html.escape`` first. Tickers and notes are user-supplied, and a portfolio tool
that renders them raw is one CSV upload away from script injection.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import pandas as pd
import streamlit as st

from src.ui import theme
from src.ui.formatters import DASH, arrow

# level -> (css class, glyph)
_INSIGHT_STYLE: dict[str, tuple[str, str]] = {
    "critical": ("qf-critical", "✖"),
    "warning": ("qf-warning", "⚠"),
    "positive": ("qf-positive", "✔"),
    "info": ("qf-info", "ℹ"),
}

_TONE_CLASSES = {"gain", "loss", "warn", "accent", "neutral", "muted"}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _tone_class(tone: str) -> str:
    return tone if tone in _TONE_CLASSES else "neutral"


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------
def sparkline_svg(
    values: Sequence[float] | pd.Series | None,
    width: int = 132,
    height: int = 26,
    color: str | None = None,
) -> str:
    """Inline SVG sparkline; returns "" when there is nothing meaningful to draw.

    Deliberately dependency-free: an ``<svg>`` string costs nothing to render
    and, unlike a Plotly figure, does not fight the KPI card for layout.
    """
    if values is None:
        return ""
    series = pd.Series(list(values), dtype="float64").dropna()
    if len(series) < 3:
        return ""

    lo, hi = float(series.min()), float(series.max())
    span = hi - lo
    n = len(series)
    pad = 2.0
    usable = height - 2 * pad

    if span <= 0:  # a flat line still deserves to be shown, centred
        points = [(i / (n - 1) * width, height / 2) for i in range(n)]
    else:
        points = [
            (i / (n - 1) * width, pad + (1 - (v - lo) / span) * usable)
            for i, v in enumerate(series)
        ]

    stroke = color or (
        theme.tokens()["gain"] if series.iloc[-1] >= series.iloc[0] else theme.tokens()["loss"]
    )
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"0,{height} {path} {width},{height}"

    return (
        f'<svg class="qf-spark" width="100%" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'role="img" aria-hidden="true">'
        f'<polygon points="{area}" fill="{stroke}" opacity="0.12"/>'
        f'<polyline points="{path}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    tone: str = "neutral",
    delta_tone: str | None = None,
    footnote: str | None = None,
    sparkline: Sequence[float] | pd.Series | None = None,
    help_text: str | None = None,
) -> None:
    """One KPI tile: label, large value, optional delta / footnote / sparkline.

    ``tone`` colours the card's left accent rail; ``delta_tone`` colours the
    delta text independently, because "volatility down 3%" is a *good* delta on
    a neutral card.
    """
    card_tone = _tone_class(tone)
    dtone = _tone_class(delta_tone or tone)
    title_attr = f' title="{_esc(help_text)}"' if help_text else ""
    hint = ' <span style="opacity:.55">ⓘ</span>' if help_text else ""

    parts = [
        f'<div class="qf-kpi qf-{card_tone}"{title_attr}>',
        f'<div class="qf-kpi-label">{_esc(label)}{hint}</div>',
        f'<div class="qf-kpi-value">{_esc(value)}</div>',
    ]
    if delta:
        parts.append(f'<div class="qf-kpi-delta qf-{dtone}">{_esc(delta)}</div>')
    if footnote:
        parts.append(f'<div class="qf-kpi-foot">{_esc(footnote)}</div>')
    spark = sparkline_svg(sparkline)
    if spark:
        parts.append(spark)
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_row(cards: Sequence[dict[str, Any]], columns: int | None = None) -> None:
    """Lay a list of ``kpi_card`` kwargs dicts out in an even grid."""
    if not cards:
        return
    n = columns or min(len(cards), 4)
    for start in range(0, len(cards), n):
        chunk = cards[start : start + n]
        cols = st.columns(n, gap="small")
        for col, card in zip(cols, chunk):
            with col:
                kpi_card(**card)
        # keep the grid rectangular so trailing cards do not stretch
        for col in cols[len(chunk) :]:
            with col:
                st.empty()


def stat_grid(stats: Sequence[tuple[str, str]], columns: int = 4) -> None:
    """Compact label/value pairs for secondary statistics."""
    if not stats:
        return
    for start in range(0, len(stats), columns):
        cols = st.columns(columns, gap="small")
        for col, (label, value) in zip(cols, stats[start : start + columns]):
            col.markdown(
                f'<div style="padding:.15rem 0">'
                f'<div class="qf-kpi-label">{_esc(label)}</div>'
                f'<div style="font-size:1rem;font-weight:600;'
                f'font-variant-numeric:tabular-nums">{_esc(value)}</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# badges & tags
# ---------------------------------------------------------------------------
def badge(text: str, tone: str = "neutral") -> str:
    """Return a pill as an HTML string (compose several, then render once)."""
    return f'<span class="qf-badge qf-{_tone_class(tone)}">{_esc(text)}</span>'


def badges(items: Iterable[tuple[str, str]]) -> None:
    """Render ``(text, tone)`` pairs as a row of pills."""
    html_parts = " ".join(badge(text, tone) for text, tone in items)
    if html_parts:
        st.markdown(
            f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:.2rem 0 .6rem">'
            f"{html_parts}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# insights
# ---------------------------------------------------------------------------
def insight_card(insight: Any) -> None:
    """Render one :class:`src.models.Insight` with its evidence line."""
    level = getattr(insight, "level", "info")
    css, glyph = _INSIGHT_STYLE.get(level, _INSIGHT_STYLE["info"])
    evidence = getattr(insight, "evidence", "") or ""
    evidence_html = f'<span class="qf-insight-evidence">{_esc(evidence)}</span>' if evidence else ""
    st.markdown(
        f'<div class="qf-insight {css}">'
        f'<span class="qf-insight-icon">{glyph}</span>'
        f'<span class="qf-insight-body">{_esc(getattr(insight, "text", ""))}'
        f"{evidence_html}</span></div>",
        unsafe_allow_html=True,
    )


def insight_list(
    insights: Sequence[Any],
    categories: Sequence[str] | None = None,
    limit: int | None = None,
    empty_message: str = "No observations for this portfolio yet.",
) -> None:
    """Render a filtered, optionally truncated list of insights."""
    items = list(insights or [])
    if categories:
        allowed = set(categories)
        items = [i for i in items if getattr(i, "category", "general") in allowed]
    if limit is not None:
        items = items[:limit]
    if not items:
        st.caption(empty_message)
        return
    for item in items:
        insight_card(item)


# ---------------------------------------------------------------------------
# hero, verdict and meter
# ---------------------------------------------------------------------------
def hero(eyebrow: str, title: str, body: str) -> None:
    """Framing panel for a page whose subject needs stating before the numbers.

    One per page. It exists for the case where rendering a chart immediately
    would let a reader draw the wrong conclusion from it -- a forecast cone
    being the obvious example.
    """
    st.markdown(
        f'<div class="qf-hero">'
        f'<div class="qf-hero-eyebrow">{_esc(eyebrow)}</div>'
        f'<div class="qf-hero-title">{_esc(title)}</div>'
        f'<div class="qf-hero-body">{_esc(body)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


#: Verdict tone -> (css class, glyph).
_VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "pass": ("qf-pass", "✔"),
    "fail": ("qf-fail", "✖"),
    "caution": ("qf-caution", "⚠"),
    "neutral": ("qf-neutral", "ℹ"),
}


def verdict(title: str, text: str, tone: str = "neutral", evidence: str | None = None) -> None:
    """The one sentence a page most needs its reader to see, stated first.

    Distinct from :func:`insight_card`: an insight is one of many observations,
    a verdict is the page's headline conclusion and there is exactly one.
    """
    css, glyph = _VERDICT_STYLE.get(tone, _VERDICT_STYLE["neutral"])
    evidence_html = f'<span class="qf-verdict-evidence">{_esc(evidence)}</span>' if evidence else ""
    st.markdown(
        f'<div class="qf-verdict {css}">'
        f'<span class="qf-verdict-glyph">{glyph}</span>'
        f'<span class="qf-verdict-body">'
        f'<span class="qf-verdict-title">{_esc(title)}</span>'
        f'<div class="qf-verdict-text">{_esc(text)}</div>{evidence_html}'
        f"</span></div>",
        unsafe_allow_html=True,
    )


def meter(
    label: str,
    value: float | None,
    display: str | None = None,
    reference: float | None = 0.5,
    tone: str | None = None,
    footnote: str | None = None,
) -> None:
    """Horizontal 0-1 bar with an optional reference mark.

    ``reference`` draws the level the bar should be read against -- 0.5 for a
    probability, so "58%" is visibly barely above a coin flip rather than
    looking like a large number on its own. Pass ``None`` to omit it.

    A missing value renders the track empty with an em dash, never a zero-width
    bar that reads as "0%".
    """
    if value is None or not pd.notna(value):
        fraction, shown = 0.0, DASH
    else:
        fraction = min(max(float(value), 0.0), 1.0)
        shown = display if display is not None else f"{fraction * 100:.0f}%"

    fill_tone = _tone_class(tone) if tone else "accent"
    mark = ""
    if reference is not None and pd.notna(reference):
        position = min(max(float(reference), 0.0), 1.0) * 100.0
        mark = f'<span class="qf-meter-mark" style="left:{position:.2f}%"></span>'
    foot = f'<div class="qf-meter-foot">{_esc(footnote)}</div>' if footnote else ""

    st.markdown(
        f'<div class="qf-meter-label"><span>{_esc(label)}</span><b>{_esc(shown)}</b></div>'
        f'<div class="qf-meter">'
        f'<span class="qf-meter-fill qf-{fill_tone}" style="width:{fraction * 100:.2f}%"></span>'
        f"{mark}</div>{foot}",
        unsafe_allow_html=True,
    )


def model_card(
    name: str,
    description: str,
    rows: Sequence[tuple[str, str]] = (),
    badge_text: str | None = None,
    badge_tone: str = "accent",
) -> None:
    """A named model with its description and its labelled properties.

    ``rows`` is where the assumption and the weakness go. A model presented
    without both is a model being sold rather than explained.
    """
    pill = f" {badge(badge_text, badge_tone)}" if badge_text else ""
    body = "".join(
        f'<div class="qf-model-row"><span>{_esc(label)}</span> {_esc(text)}</div>'
        for label, text in rows
        if text
    )
    st.markdown(
        f'<div class="qf-model">'
        f'<div class="qf-model-name">{_esc(name)}{pill}</div>'
        f'<div class="qf-model-desc">{_esc(description)}</div>'
        f"{body}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# empty / error states
# ---------------------------------------------------------------------------
def empty_state(
    title: str, message: str, icon: str = "\U0001f4ad", hint: str | None = None
) -> None:
    """A styled placeholder - far friendlier than a bare ``st.info``."""
    hint_html = f'<div class="qf-empty-hint">{_esc(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="qf-empty">'
        f'<div class="qf-empty-icon">{_esc(icon)}</div>'
        f'<div class="qf-empty-title">{_esc(title)}</div>'
        f'<div class="qf-empty-msg">{_esc(message)}</div>'
        f"{hint_html}</div>",
        unsafe_allow_html=True,
    )


def warning_list(messages: Sequence[str], title: str = "Data notes") -> None:
    """Collapse a pile of per-ticker warnings into one expander.

    Ten stacked ``st.warning`` calls push the actual analysis below the fold;
    one honest expander keeps the problems visible without burying the content.
    """
    items = [m for m in (messages or []) if m]
    if not items:
        return
    with st.expander(f"{title} ({len(items)})", expanded=len(items) <= 2):
        for message in items:
            st.markdown(f"- {message}")


# ---------------------------------------------------------------------------
# data provenance
# ---------------------------------------------------------------------------
def freshness_bar(
    fetched_at: Any,
    source: str,
    extra: Sequence[str] = (),
    stale: bool = False,
) -> None:
    """One-line provenance strip: when the data was fetched and from where."""
    from src.ui.formatters import fmt_datetime

    dot = '<span class="qf-dot qf-stale"></span>' if stale else '<span class="qf-dot"></span>'
    chips = "".join(f'<span style="opacity:.85">{_esc(item)}</span>' for item in extra if item)
    st.markdown(
        f'<div class="qf-freshness">{dot}'
        f"<span>Fetched {_esc(fmt_datetime(fetched_at))} UTC</span>"
        f"<span>{_esc(source)}</span>{chips}</div>",
        unsafe_allow_html=True,
    )


def methodology(title: str, body_md: str, expanded: bool = False) -> None:
    """Collapsible 'how this number is computed' block.

    Used everywhere a judgement call was made. Showing the formula and its
    caveats is the difference between an analytics tool and a black box.
    """
    with st.expander(f"ℹ {title}", expanded=expanded):
        st.markdown(body_md)


# ---------------------------------------------------------------------------
# tables & downloads
# ---------------------------------------------------------------------------
def dataframe(df: pd.DataFrame | None, height: int | None = None, **kwargs: Any) -> None:
    """``st.dataframe`` with the defaults this app always wants."""
    if df is None or len(df) == 0:
        st.caption("No rows to display.")
        return
    kwargs.setdefault("hide_index", True)
    kwargs.setdefault("width", "stretch")
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def download_row(items: Sequence[tuple[str, Any, str, str]]) -> None:
    """A row of download buttons: ``(label, data, filename, mime)``."""
    items = [i for i in items if i[1] is not None]
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, data, filename, mime) in zip(cols, items):
        col.download_button(
            label,
            data=data,
            file_name=filename,
            mime=mime,
            width="stretch",
        )


@contextmanager
def loading(message: str):
    """Spinner context manager, so page code reads as prose."""
    with st.spinner(message):
        yield


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def delta_text(value: float | None, suffix: str = "%", decimals: int = 2) -> str:
    """'▲ 2.41%' / '▼ 1.10%' / em dash when unknown."""
    if value is None or not pd.notna(value):
        return DASH
    return f"{arrow(value)} {abs(float(value)):,.{decimals}f}{suffix}"


def chart(fig: Any, **kwargs: Any) -> None:
    """Render a Plotly figure with this app's standard options."""
    if fig is None:
        return
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("config", {"displaylogo": False, "responsive": True})
    st.plotly_chart(fig, **kwargs)
