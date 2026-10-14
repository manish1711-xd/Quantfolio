"""Visual system: design tokens, injected CSS and page chrome.

The look is a professional analyst terminal - dense, quiet, high-contrast, with
colour reserved for meaning (gain / loss / warning) rather than decoration.

Streamlit does not expose its active theme to CSS, so the palette is resolved in
Python at render time and emitted as CSS custom properties. Everything
downstream (cards, badges, tables, charts) reads those properties, which means
the entire application re-themes from the token table below.
"""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

# ---------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------
DARK_TOKENS: dict[str, str] = {
    "bg": "#0b0f19",
    "surface": "#141a26",
    "surface_2": "#1b2231",
    "surface_hover": "#212a3b",
    "border": "rgba(148, 163, 184, 0.16)",
    "border_strong": "rgba(148, 163, 184, 0.30)",
    "text": "#e6edf7",
    "text_muted": "#94a3b8",
    "text_faint": "#64748b",
    "accent": "#6366f1",
    "accent_soft": "rgba(99, 102, 241, 0.16)",
    "accent_2": "#22d3ee",
    "gain": "#22c55e",
    "gain_soft": "rgba(34, 197, 94, 0.14)",
    "loss": "#f43f5e",
    "loss_soft": "rgba(244, 63, 94, 0.14)",
    "warn": "#f59e0b",
    "warn_soft": "rgba(245, 158, 11, 0.14)",
    "info_soft": "rgba(56, 189, 248, 0.12)",
    "shadow": "0 1px 2px rgba(0,0,0,0.40), 0 8px 24px -12px rgba(0,0,0,0.60)",
    "grid": "rgba(148, 163, 184, 0.14)",
}

LIGHT_TOKENS: dict[str, str] = {
    "bg": "#f7f8fb",
    "surface": "#ffffff",
    "surface_2": "#f2f4f8",
    "surface_hover": "#eaeef5",
    "border": "rgba(15, 23, 42, 0.10)",
    "border_strong": "rgba(15, 23, 42, 0.20)",
    "text": "#0f172a",
    "text_muted": "#52627a",
    "text_faint": "#8494a8",
    "accent": "#4f46e5",
    "accent_soft": "rgba(79, 70, 229, 0.10)",
    "accent_2": "#0891b2",
    "gain": "#15803d",
    "gain_soft": "rgba(21, 128, 61, 0.10)",
    "loss": "#be123c",
    "loss_soft": "rgba(190, 18, 60, 0.10)",
    "warn": "#b45309",
    "warn_soft": "rgba(180, 83, 9, 0.12)",
    "info_soft": "rgba(2, 132, 199, 0.10)",
    "shadow": "0 1px 2px rgba(15,23,42,0.06), 0 8px 24px -16px rgba(15,23,42,0.24)",
    "grid": "rgba(15, 23, 42, 0.08)",
}

#: Records which theme's stylesheet the last run emitted. Informational --
#: nothing branches on it any more, because branching on it was the bug.
_STATE_KEY = "_quantfolio_css_theme"


# ---------------------------------------------------------------------------
# theme resolution
# ---------------------------------------------------------------------------
def active_base() -> str:
    """Return "dark" or "light" for the theme Streamlit is actually rendering.

    The CONFIGURED base wins, and that ordering is the whole point of this
    function. ``.streamlit/config.toml`` pins ``theme.base = "dark"`` along with
    an explicit palette, and Streamlit honours it regardless of the viewer's
    system preference — so config is a statement about what is on screen.

    ``st.context.theme.type`` is not. It reports the *browser's* resolved
    appearance, which on a machine set to light mode says "light" while
    Streamlit is painting the configured dark chrome behind it. Trusting it
    first is what produced white cards on a near-black background: the app
    chrome dark, every design-system surface light, on every viewer whose OS
    preference happened to disagree with the config.

    So the order is: an explicitly configured base, then the viewer's context
    (which is the right answer only when nothing is configured and Streamlit
    genuinely is following the system preference), then dark — the terminal
    look this app was designed around.
    """
    try:
        base = st.get_option("theme.base")
        if base in {"dark", "light"}:
            return str(base)
    except Exception:  # noqa: BLE001 - get_option can fail outside a runtime
        pass
    try:
        context_theme = getattr(st, "context", None)
        theme_type = getattr(getattr(context_theme, "theme", None), "type", None)
        if theme_type in {"dark", "light"}:
            return str(theme_type)
    except Exception:  # noqa: BLE001 - context is not available in every runtime
        pass
    return "dark"


def tokens(base: str | None = None) -> dict[str, str]:
    """Design tokens for the active (or requested) theme."""
    return DARK_TOKENS if (base or active_base()) == "dark" else LIGHT_TOKENS


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------
_CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --qf-bg: {bg};
  --qf-surface: {surface};
  --qf-surface-2: {surface_2};
  --qf-surface-hover: {surface_hover};
  --qf-border: {border};
  --qf-border-strong: {border_strong};
  --qf-text: {text};
  --qf-muted: {text_muted};
  --qf-faint: {text_faint};
  --qf-accent: {accent};
  --qf-accent-soft: {accent_soft};
  --qf-accent-2: {accent_2};
  --qf-gain: {gain};
  --qf-gain-soft: {gain_soft};
  --qf-loss: {loss};
  --qf-loss-soft: {loss_soft};
  --qf-warn: {warn};
  --qf-warn-soft: {warn_soft};
  --qf-info-soft: {info_soft};
  --qf-shadow: {shadow};
  --qf-radius: 12px;
  --qf-radius-sm: 8px;
  --qf-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif;
  --qf-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace;
}}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
  font-family: var(--qf-font);
  font-feature-settings: 'cv02','cv03','cv04','tnum';
}}

/* ---------- layout density ---------- */
[data-testid="stMain"] .block-container {{
  padding-top: 2.2rem;
  padding-bottom: 3.5rem;
  max-width: 1560px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 0.5rem; }}
[data-testid="stMain"] hr {{ border-color: var(--qf-border); margin: 1.4rem 0; }}

/* ---------- typography ---------- */
h1, h2, h3, h4 {{ letter-spacing: -0.018em; font-weight: 650; }}
[data-testid="stMain"] h1 {{ font-size: 1.85rem; margin-bottom: 0.15rem; }}
[data-testid="stMain"] h2 {{ font-size: 1.28rem; margin-top: 0.4rem; }}
[data-testid="stMain"] h3 {{ font-size: 1.06rem; }}
code, pre, .qf-mono {{ font-family: var(--qf-mono); font-size: 0.86em; }}

/* ---------- page header ---------- */
.qf-header {{
  display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap;
  padding-bottom: 0.55rem; margin-bottom: 1.1rem;
  border-bottom: 1px solid var(--qf-border);
}}
.qf-header .qf-title {{
  font-size: 1.5rem; font-weight: 680; letter-spacing: -0.02em;
  color: var(--qf-text); line-height: 1.2;
}}
.qf-header .qf-subtitle {{ font-size: 0.86rem; color: var(--qf-muted); }}

/* ---------- section label ---------- */
.qf-section {{
  display: flex; align-items: center; gap: 0.5rem;
  margin: 1.6rem 0 0.7rem; font-size: 0.74rem; font-weight: 650;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--qf-faint);
}}
.qf-section::after {{
  content: ""; flex: 1; height: 1px; background: var(--qf-border);
}}

/* ---------- KPI cards ---------- */
.qf-kpi {{
  background: var(--qf-surface);
  border: 1px solid var(--qf-border);
  border-radius: var(--qf-radius);
  padding: 0.85rem 1rem 0.9rem;
  height: 100%;
  box-shadow: var(--qf-shadow);
  transition: border-color 140ms ease, transform 140ms ease;
  position: relative; overflow: hidden;
}}
.qf-kpi:hover {{ border-color: var(--qf-border-strong); transform: translateY(-1px); }}
.qf-kpi::before {{
  content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: var(--qf-accent); opacity: 0.75;
}}
.qf-kpi.qf-gain::before {{ background: var(--qf-gain); }}
.qf-kpi.qf-loss::before {{ background: var(--qf-loss); }}
.qf-kpi.qf-warn::before {{ background: var(--qf-warn); }}
.qf-kpi-label {{
  font-size: 0.685rem; font-weight: 600; letter-spacing: 0.075em;
  text-transform: uppercase; color: var(--qf-faint);
  display: flex; align-items: center; gap: 0.3rem;
}}
.qf-kpi-value {{
  font-size: 1.52rem; font-weight: 660; color: var(--qf-text);
  margin-top: 0.28rem; line-height: 1.15; font-variant-numeric: tabular-nums;
  word-break: break-word;
}}
.qf-kpi-delta {{
  font-size: 0.79rem; font-weight: 570; margin-top: 0.22rem;
  font-variant-numeric: tabular-nums;
}}
.qf-kpi-delta.qf-gain {{ color: var(--qf-gain); }}
.qf-kpi-delta.qf-loss {{ color: var(--qf-loss); }}
.qf-kpi-delta.qf-warn {{ color: var(--qf-warn); }}
.qf-kpi-delta.qf-muted {{ color: var(--qf-faint); }}
.qf-kpi-foot {{ font-size: 0.7rem; color: var(--qf-faint); margin-top: 0.3rem; }}
.qf-spark {{ display: block; margin-top: 0.5rem; opacity: 0.9; }}

/* ---------- badges / pills ---------- */
.qf-badge {{
  display: inline-flex; align-items: center; gap: 0.28rem;
  padding: 0.12rem 0.5rem; border-radius: 999px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.015em;
  border: 1px solid transparent; white-space: nowrap;
}}
.qf-badge.qf-gain {{ background: var(--qf-gain-soft); color: var(--qf-gain); border-color: var(--qf-gain); }}
.qf-badge.qf-loss {{ background: var(--qf-loss-soft); color: var(--qf-loss); border-color: var(--qf-loss); }}
.qf-badge.qf-warn {{ background: var(--qf-warn-soft); color: var(--qf-warn); border-color: var(--qf-warn); }}
.qf-badge.qf-accent {{ background: var(--qf-accent-soft); color: var(--qf-accent); border-color: var(--qf-accent); }}
.qf-badge.qf-neutral {{ background: var(--qf-surface-2); color: var(--qf-muted); border-color: var(--qf-border); }}

/* ---------- insight cards ---------- */
.qf-insight {{
  display: flex; gap: 0.65rem; align-items: flex-start;
  padding: 0.62rem 0.8rem; margin-bottom: 0.45rem;
  border-radius: var(--qf-radius-sm);
  border: 1px solid var(--qf-border);
  border-left-width: 3px;
  background: var(--qf-surface);
  font-size: 0.885rem; line-height: 1.45; color: var(--qf-text);
}}
.qf-insight.qf-positive {{ border-left-color: var(--qf-gain); background: var(--qf-gain-soft); }}
.qf-insight.qf-warning  {{ border-left-color: var(--qf-warn); background: var(--qf-warn-soft); }}
.qf-insight.qf-critical {{ border-left-color: var(--qf-loss); background: var(--qf-loss-soft); }}
.qf-insight.qf-info     {{ border-left-color: var(--qf-accent); background: var(--qf-info-soft); }}
.qf-insight-icon {{ flex: none; font-size: 0.95rem; line-height: 1.4; }}
.qf-insight-body {{ flex: 1; }}
.qf-insight-evidence {{
  display: block; margin-top: 0.22rem; font-size: 0.75rem;
  color: var(--qf-muted); font-family: var(--qf-mono);
}}

/* ---------- hero panel ---------- */
/* Used at the top of a page whose subject needs framing before any number
   appears. One per page, never more: a second hero is just a box. */
.qf-hero {{
  position: relative; overflow: hidden;
  border: 1px solid var(--qf-border); border-radius: var(--qf-radius);
  background:
    radial-gradient(110% 140% at 100% 0%, var(--qf-accent-soft) 0%, transparent 58%),
    linear-gradient(180deg, var(--qf-surface) 0%, var(--qf-surface-2) 100%);
  padding: 1.15rem 1.35rem 1.2rem; margin-bottom: 1rem;
  box-shadow: var(--qf-shadow);
}}
.qf-hero::before {{
  content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: linear-gradient(180deg, var(--qf-accent), var(--qf-accent-2));
}}
.qf-hero-eyebrow {{
  font-size: 0.68rem; font-weight: 650; letter-spacing: 0.11em;
  text-transform: uppercase; color: var(--qf-accent);
}}
.qf-hero-title {{
  font-size: 1.18rem; font-weight: 660; color: var(--qf-text);
  margin-top: 0.3rem; letter-spacing: -0.015em;
}}
.qf-hero-body {{
  font-size: 0.875rem; line-height: 1.55; color: var(--qf-muted);
  margin-top: 0.4rem; max-width: 78ch;
}}

/* ---------- verdict banner ---------- */
/* The single most important sentence on a page, stated before the charts.
   Tone is semantic: pass / fail / caution, never decorative. */
.qf-verdict {{
  display: flex; gap: 0.9rem; align-items: flex-start;
  border: 1px solid var(--qf-border); border-left-width: 4px;
  border-radius: var(--qf-radius); padding: 0.95rem 1.15rem;
  background: var(--qf-surface); margin-bottom: 0.9rem;
  box-shadow: var(--qf-shadow);
}}
.qf-verdict.qf-pass {{ border-left-color: var(--qf-gain); background: var(--qf-gain-soft); }}
.qf-verdict.qf-fail {{ border-left-color: var(--qf-loss); background: var(--qf-loss-soft); }}
.qf-verdict.qf-caution {{ border-left-color: var(--qf-warn); background: var(--qf-warn-soft); }}
.qf-verdict.qf-neutral {{ border-left-color: var(--qf-accent); background: var(--qf-info-soft); }}
.qf-verdict-glyph {{ flex: none; font-size: 1.5rem; line-height: 1.25; opacity: 0.9; }}
.qf-verdict-body {{ flex: 1; min-width: 0; }}
.qf-verdict-title {{
  font-size: 1.02rem; font-weight: 650; color: var(--qf-text); letter-spacing: -0.01em;
}}
.qf-verdict-text {{
  font-size: 0.885rem; line-height: 1.5; color: var(--qf-text);
  opacity: 0.92; margin-top: 0.25rem;
}}
.qf-verdict-evidence {{
  display: block; margin-top: 0.4rem; font-family: var(--qf-mono);
  font-size: 0.735rem; color: var(--qf-muted);
}}

/* ---------- probability meter ---------- */
.qf-meter-label {{
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--qf-faint); margin-bottom: 0.28rem;
}}
.qf-meter-label b {{
  font-size: 0.95rem; font-weight: 660; letter-spacing: 0;
  text-transform: none; color: var(--qf-text); font-variant-numeric: tabular-nums;
}}
.qf-meter {{
  position: relative; height: 9px; border-radius: 999px;
  background: var(--qf-surface-2); border: 1px solid var(--qf-border);
  overflow: hidden;
}}
.qf-meter-fill {{
  position: absolute; inset: 0 auto 0 0; border-radius: 999px;
  background: linear-gradient(90deg, var(--qf-accent), var(--qf-accent-2));
}}
.qf-meter-fill.qf-gain {{ background: linear-gradient(90deg, var(--qf-gain), var(--qf-accent-2)); }}
.qf-meter-fill.qf-loss {{ background: linear-gradient(90deg, var(--qf-loss), var(--qf-warn)); }}
/* The reference level a meter is read against (50% for a coin flip). */
.qf-meter-mark {{
  position: absolute; top: -2px; bottom: -2px; width: 2px;
  background: var(--qf-text); opacity: 0.45;
}}
.qf-meter-foot {{ font-size: 0.72rem; color: var(--qf-faint); margin-top: 0.3rem; }}

/* ---------- model card ---------- */
.qf-model {{
  border: 1px solid var(--qf-border); border-radius: var(--qf-radius);
  background: var(--qf-surface); padding: 0.85rem 1rem; height: 100%;
}}
.qf-model-name {{
  font-size: 0.97rem; font-weight: 640; color: var(--qf-text);
  display: flex; align-items: center; gap: 0.45rem;
}}
.qf-model-desc {{ font-size: 0.82rem; color: var(--qf-muted); margin-top: 0.3rem; line-height: 1.45; }}
.qf-model-row {{ font-size: 0.8rem; line-height: 1.45; margin-top: 0.5rem; color: var(--qf-text); }}
.qf-model-row span {{
  display: inline-block; min-width: 5.6rem; color: var(--qf-faint);
  font-size: 0.68rem; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase;
}}

/* ---------- empty state ---------- */
.qf-empty {{
  text-align: center; padding: 2.6rem 1.4rem;
  border: 1px dashed var(--qf-border-strong); border-radius: var(--qf-radius);
  background: var(--qf-surface); color: var(--qf-muted);
}}
.qf-empty-icon {{ font-size: 1.9rem; opacity: 0.7; }}
.qf-empty-title {{ font-size: 1rem; font-weight: 620; color: var(--qf-text); margin-top: 0.45rem; }}
.qf-empty-msg {{ font-size: 0.87rem; margin-top: 0.3rem; }}
.qf-empty-hint {{ font-size: 0.79rem; margin-top: 0.7rem; color: var(--qf-faint); }}

/* ---------- freshness strip ---------- */
.qf-freshness {{
  display: flex; flex-wrap: wrap; gap: 0.45rem 0.9rem; align-items: center;
  padding: 0.42rem 0.75rem; border-radius: var(--qf-radius-sm);
  background: var(--qf-surface-2); border: 1px solid var(--qf-border);
  font-size: 0.74rem; color: var(--qf-muted); margin-bottom: 0.9rem;
}}
.qf-dot {{
  width: 7px; height: 7px; border-radius: 50%; display: inline-block;
  background: var(--qf-gain); margin-right: 0.35rem;
  box-shadow: 0 0 0 3px var(--qf-gain-soft);
}}
.qf-dot.qf-stale {{ background: var(--qf-warn); box-shadow: 0 0 0 3px var(--qf-warn-soft); }}

/* ---------- streamlit widget polish ---------- */
[data-testid="stMetric"] {{
  background: var(--qf-surface); border: 1px solid var(--qf-border);
  border-radius: var(--qf-radius); padding: 0.7rem 0.9rem;
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.7rem !important; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--qf-faint);
}}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-size: 1.4rem; }}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.15rem; border-bottom: 1px solid var(--qf-border); padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
  height: 2.4rem; padding: 0 0.85rem; background: transparent;
  font-size: 0.87rem; font-weight: 550; color: var(--qf-muted);
  border-radius: var(--qf-radius-sm) var(--qf-radius-sm) 0 0;
}}
.stTabs [data-baseweb="tab"]:hover {{ background: var(--qf-surface-2); color: var(--qf-text); }}
.stTabs [aria-selected="true"] {{ color: var(--qf-text) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--qf-accent); height: 2px; }}

[data-testid="stExpander"] details {{
  border: 1px solid var(--qf-border); border-radius: var(--qf-radius);
  background: var(--qf-surface); overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-size: 0.86rem; font-weight: 570; }}
[data-testid="stExpander"] summary:hover {{ color: var(--qf-accent); }}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: var(--qf-radius-sm); font-weight: 570; font-size: 0.86rem;
  border: 1px solid var(--qf-border-strong); transition: all 130ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--qf-accent); color: var(--qf-accent);
  transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{ border-color: transparent; }}

[data-testid="stSidebar"] {{
  border-right: 1px solid var(--qf-border);
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}
[data-testid="stSidebar"] h1 {{ font-size: 1.05rem; }}
[data-testid="stSidebar"] .qf-section {{ margin-top: 1rem; }}

[data-testid="stDataFrame"] {{ border-radius: var(--qf-radius-sm); }}
[data-testid="stDataFrame"] div[role="grid"] {{ font-variant-numeric: tabular-nums; font-size: 0.84rem; }}

[data-testid="stAlert"] {{ border-radius: var(--qf-radius-sm); font-size: 0.87rem; }}

[data-testid="stElementToolbar"] {{ backdrop-filter: blur(6px); }}

/* plotly charts sit on the app background, not a white card */
[data-testid="stPlotlyChart"] {{
  border: 1px solid var(--qf-border); border-radius: var(--qf-radius);
  padding: 0.35rem 0.35rem 0.1rem; background: var(--qf-surface);
}}

/* ---------- accessibility & print ---------- */
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
  .qf-kpi:hover, .stButton > button:hover {{ transform: none; }}
}}
@media print {{
  [data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stHeader"] {{ display: none !important; }}
  [data-testid="stMain"] .block-container {{ padding: 0; max-width: 100%; }}
  .qf-kpi, [data-testid="stPlotlyChart"] {{ break-inside: avoid; box-shadow: none; }}
}}
@media (max-width: 900px) {{
  [data-testid="stMain"] .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .qf-kpi-value {{ font-size: 1.3rem; }}
}}
"""


@lru_cache(maxsize=2)
def _css_for(base: str) -> str:
    """The rendered stylesheet for one theme, formatted at most once."""
    return _CSS_TEMPLATE.format(**tokens(base))


def build_css(base: str | None = None) -> str:
    """Render the stylesheet for one theme."""
    return _css_for(base or active_base())


def inject_css() -> None:
    """Emit the stylesheet. Runs on EVERY script run, deliberately.

    Streamlit rebuilds the page from whatever the *current* script run emits.
    A ``<style>`` block is not mounted once and kept: it exists only while the
    run that produced it is the latest one.

    This function used to skip re-emitting when a session flag said the theme's
    CSS was "already mounted", on the reasoning that re-emitting a large style
    block would bloat the DOM. The premise was wrong in both directions.
    Streamlit *replaces* the element at this position on each run rather than
    appending, so emitting every time yields exactly one ``<style>`` tag — and
    skipping it yields **none**, which left the entire application unstyled
    from the second interaction onward: no cards, no KPI tiles, no layout
    width, every page reduced to unstyled text.

    The cost of getting it right is one cached dictionary lookup: the rendered
    stylesheet is memoised per theme by :func:`_css_for`, so the template is
    formatted at most twice for the life of the process.
    """
    base = active_base()
    st.markdown(f"<style>{_css_for(base)}</style>", unsafe_allow_html=True)
    st.session_state[_STATE_KEY] = base


# ---------------------------------------------------------------------------
# page chrome
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    """Title row with a hairline rule, used at the top of every page."""
    import html as _html

    icon_html = f"{_html.escape(icon)} " if icon else ""
    sub_html = f'<span class="qf-subtitle">{_html.escape(subtitle)}</span>' if subtitle else ""
    st.markdown(
        f'<div class="qf-header">'
        f'<span class="qf-title">{icon_html}{_html.escape(title)}</span>{sub_html}'
        f"</div>",
        unsafe_allow_html=True,
    )


def section(title: str, icon: str = "") -> None:
    """Small uppercase section label with a trailing rule."""
    import html as _html

    label = f"{icon} {title}".strip() if icon else title
    st.markdown(f'<div class="qf-section">{_html.escape(label)}</div>', unsafe_allow_html=True)


def spacer(height_rem: float = 1.0) -> None:
    st.markdown(f'<div style="height:{height_rem}rem"></div>', unsafe_allow_html=True)
