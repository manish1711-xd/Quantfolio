"""Export - the run leaves the terminal.

This page answers three questions: *what can I hand to someone else*, *what
exactly is inside each file*, and *what settings produced these numbers*.

Three artefacts are offered, all built from the completed
:class:`~src.models.AnalysisResult` and nothing else: a formatted multi-sheet
Excel workbook (:mod:`src.reporting.excel`), a self-contained HTML tearsheet
with the charts embedded (:mod:`src.reporting.html_report`), and the raw
positions table as CSV. Nothing is computed here - the page assembles the
figure set the tearsheet embeds, serialises what the pipeline already produced,
and prints the run parameters beside it, because a report without its
assumptions is a screenshot with extra steps.

The bundle is built **on demand** and cached in session state against
``result.generated_at``: a workbook is tens of kilobytes of formatting work and
Streamlit reruns this script on every keystroke elsewhere in the app.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from html import unescape
from io import BytesIO
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts as ch
from src.config import (
    APP_NAME,
    APP_VERSION,
    DATA_SOURCE,
    DISCLAIMER,
    PERIOD_LABELS,
    TRADING_DAYS,
)
from src.reporting import excel, html_report
from src.ui import components as ui
from src.ui import email_panel, guards
from src.ui import theme as ui_theme
from src.ui.formatters import DASH, fmt_datetime, fmt_ratio_pct

#: Session key holding the built artefacts for one analysis run.
_BUNDLE_KEY = "reports_bundle"

#: A return histogram fitted against a normal needs at least this many points
#: before the shape means anything (``charts.risk.MIN_DISTRIBUTION_POINTS``).
_MIN_DISTRIBUTION_POINTS = 10

#: MIME types, spelled out once so the filename and the type never drift apart.
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_HTML_MIME = "text/html"
_CSV_MIME = "text/csv"

#: What every workbook sheet holds, in :data:`src.reporting.excel.SHEETS` order.
_SHEET_NOTES: dict[str, str] = {
    "Summary": (
        "Title block, the disclaimer, the portfolio snapshot, the run "
        "parameters and the full insight log - plus any section that failed "
        "to write, reported rather than hidden."
    ),
    "Holdings": (
        "One row per priced position: quantity, average cost, last price, "
        "quote currency, FX rate, invested, market value, P&L and weight, "
        "closed by a SUBTOTAL row that recalculates when you filter."
    ),
    "Risk": (
        "Every headline risk statistic with its unit and a one-line meaning, "
        "followed by the 0-100 risk score and the points each component "
        "earned."
    ),
    "Benchmark": (
        "Portfolio against the index on overlapping dates only: excess "
        "return, alpha, beta, R-squared, tracking error, information ratio "
        "and up/down capture."
    ),
    "Contribution": (
        "Who moved the portfolio and who carries the risk that moved it - "
        "return contributions summing to the portfolio return, risk "
        "contributions summing to 100% of portfolio volatility."
    ),
    "Correlation": (
        "Pairwise Pearson correlation of daily returns on a diverging colour "
        "scale, so the blocks of names that move together are visible at a "
        "glance."
    ),
    "Drawdowns": (
        "Every peak to trough to recovery episode, deepest first, with "
        "calendar-day counts and a blank recovery date where the portfolio "
        "was still under water at the end of the window."
    ),
    "Transactions": (
        "The FIFO ledger: the trades you supplied, open tax lots, realised "
        "P&L per matched slice and the dated cash flows behind XIRR."
    ),
}

#: Sections the HTML tearsheet emits, in document order.
_REPORT_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Masthead",
        "Portfolio name, generation timestamp, window, benchmark, "
        "reporting currency and risk-free rate.",
    ),
    ("Data notes", "Every warning the pipeline raised on this run, verbatim."),
    ("Key figures", "The twelve headline tiles, in the order a reviewer reads them."),
    ("Statistics", "Return, risk, ratio, concentration and benchmark tables in full."),
    (
        "What the numbers say",
        "The rule-derived insights, critical first, each with the evidence that fired it.",
    ),
    ("Holdings", "The position table as printed rows, not a screenshot of a grid."),
    ("Charts", "The figures listed below, recoloured for white paper."),
    ("Footer", "Disclaimer, data source and the conventions behind every number."),
)


# ---------------------------------------------------------------------------
# small local helpers (serialisation only - no analytics happen on this page)
# ---------------------------------------------------------------------------
def _is_frame(obj: Any) -> bool:
    """True for a non-empty DataFrame."""
    return isinstance(obj, pd.DataFrame) and not obj.empty


def _run_stamp(result: Any) -> datetime:
    """The run's timestamp, falling back to now when the pipeline left none."""
    stamp = getattr(result, "generated_at", None)
    return stamp if isinstance(stamp, datetime) else datetime.now(timezone.utc)


def _cache_key(result: Any) -> str:
    """Identity of one analysis run, used to invalidate the cached bundle."""
    stamp = getattr(result, "generated_at", None)
    return stamp.isoformat() if isinstance(stamp, datetime) else "unstamped"


def _filesize(blob: bytes | None) -> str:
    """Byte count as KB / MB, or a dash when the artefact was not built."""
    if not blob:
        return DASH
    kb = len(blob) / 1024.0
    return f"{kb:,.1f} KB" if kb < 1024.0 else f"{kb / 1024.0:,.2f} MB"


def _workbook_sheets(blob: bytes | None) -> list[str]:
    """Sheet names actually present in a built workbook, in tab order.

    An ``.xlsx`` is a zip archive, so the truthful answer to "which sheets did
    this run produce?" is read back out of the bytes that were built rather
    than re-derived from the analysis. Returns an empty list if the archive
    cannot be inspected - the preview then falls back to the documented order.
    """
    if not blob:
        return []
    try:
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            xml = archive.read("xl/workbook.xml").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a preview must never break the page
        return []
    return [unescape(name) for name in re.findall(r'<sheet[^>]*\sname="([^"]*)"', xml)]


def _safe_build(builder: Callable[[], bytes]) -> tuple[bytes | None, str]:
    """Run one artefact builder, returning ``(bytes, error)``.

    One artefact failing must not cost the user the other two, and a silent
    failure would be worse than either: the error text is surfaced on the page.
    """
    try:
        return builder(), ""
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 1. the figure set the tearsheet embeds
# ---------------------------------------------------------------------------
def _figure_plan(
    result: Any, cfg: Any
) -> list[tuple[str, str, Callable[[], go.Figure] | None, str]]:
    """``(heading, what it shows, builder or None, why it is missing)``.

    One list drives both the preview panel and the document, so the panel can
    never promise a chart the tearsheet does not contain. A panel whose input
    the pipeline did not produce is skipped rather than embedded empty - a
    blank chart in a report reads as a bug, not as missing data.
    """
    symbol = getattr(cfg, "currency_symbol", "")
    risk = result.risk or {}
    confidence = risk.get("confidence", getattr(cfg, "var_confidence", 0.95))

    values = result.values
    positions = result.positions
    returns = result.returns
    correlation = result.correlation
    monthly = result.monthly
    contribution = (result.attribution or {}).get("return_contribution")

    has_history = bool(result.has_history)
    n_returns = int(len(returns.dropna())) if isinstance(returns, pd.Series) else 0
    corr_assets = (
        int(min(correlation.shape))
        if isinstance(correlation, pd.DataFrame) and correlation.ndim == 2
        else 0
    )

    plan: list[tuple[str, str, Callable[[], go.Figure] | None, str]] = [
        (
            "Portfolio value through time",
            f"Value in {getattr(cfg, 'base_currency', '')} against "
            f"{getattr(cfg, 'benchmark_name', 'the benchmark')}, indexed to the "
            "same starting scale.",
            (
                lambda: ch.growth_line(
                    values,
                    symbol,
                    result.benchmark_prices,
                    getattr(cfg, "benchmark_name", ""),
                )
            )
            if has_history
            else None,
            "needs at least two overlapping trading days",
        ),
        (
            "Where the money sits",
            "Treemap of market value by holding, tinted by unrealised return against cost.",
            (lambda: ch.allocation_treemap(positions, currency_symbol=symbol))
            if _is_frame(positions)
            else None,
            "no priced position survived validation",
        ),
        (
            "Drawdown from the high-water mark",
            "The underwater curve: how far below the running peak the book sat, and for how long.",
            (lambda: ch.underwater_chart(values)) if has_history else None,
            "needs at least two valuation dates",
        ),
        (
            "Daily return distribution",
            f"Histogram of daily returns against a fitted normal, with the "
            f"{fmt_ratio_pct(confidence, 0)} VaR and expected-shortfall lines "
            "marked.",
            (
                lambda: ch.return_distribution(
                    returns,
                    risk.get("var"),
                    risk.get("cvar"),
                    confidence=float(confidence),
                )
            )
            if n_returns >= _MIN_DISTRIBUTION_POINTS
            else None,
            f"needs {_MIN_DISTRIBUTION_POINTS} daily returns, this run has {n_returns}",
        ),
        (
            "Correlation of daily returns",
            "Pairwise Pearson correlation between holdings - the check on "
            "whether the diversification is real.",
            (lambda: ch.correlation_heatmap(correlation)) if corr_assets >= 2 else None,
            "needs two holdings with overlapping price history",
        ),
        (
            "Calendar month returns",
            "Year by month grid with an annual column, so a bad quarter is "
            "visible rather than averaged away.",
            (lambda: ch.monthly_heatmap(monthly)) if _is_frame(monthly) else None,
            "no complete calendar month inside the window",
        ),
        (
            "Return contribution by holding",
            "Waterfall from zero to the portfolio's return, one step per name, "
            "each measured against the total cost base.",
            (lambda: ch.contribution_waterfall(contribution, symbol))
            if _is_frame(contribution)
            else None,
            "attribution needs a positive cost base and a computable P&L",
        ),
    ]
    return plan


def _tearsheet_figures(result: Any, cfg: Any) -> dict[str, go.Figure]:
    """Build every available panel, keyed by the heading it prints under."""
    figures: dict[str, go.Figure] = {}
    for heading, _shows, builder, _reason in _figure_plan(result, cfg):
        if builder is None:
            continue
        try:
            figures[heading] = builder()
        except Exception:  # noqa: BLE001 - one bad panel must not lose the report
            continue
    return figures


# ---------------------------------------------------------------------------
# 2. build and cache
# ---------------------------------------------------------------------------
def _build_bundle(result: Any, cfg: Any) -> dict[str, Any]:
    """Serialise all three artefacts once, recording sizes and any failure."""
    figures = _tearsheet_figures(result, cfg)

    workbook, xlsx_error = _safe_build(lambda: excel.build_workbook(result, cfg))
    tearsheet, html_error = _safe_build(
        lambda: html_report.build_html_report(result, cfg, figures).encode("utf-8")
    )
    positions_csv, csv_error = _safe_build(lambda: excel.positions_csv(result.positions))

    return {
        "key": _cache_key(result),
        "built_at": datetime.now(timezone.utc),
        "xlsx": workbook,
        "html": tearsheet,
        "csv": positions_csv,
        "figures": list(figures),
        "sheets": _workbook_sheets(workbook),
        "errors": [
            (label, message)
            for label, message in (
                ("Excel workbook", xlsx_error),
                ("HTML tearsheet", html_error),
                ("Positions CSV", csv_error),
            )
            if message
        ],
    }


def _cached_bundle(result: Any) -> dict[str, Any] | None:
    """The bundle built for *this* run, or None when it must be rebuilt."""
    bundle = st.session_state.get(_BUNDLE_KEY)
    if isinstance(bundle, dict) and bundle.get("key") == _cache_key(result):
        return bundle
    return None


# ---------------------------------------------------------------------------
# 3. the download block
# ---------------------------------------------------------------------------
def _downloads(result: Any, cfg: Any) -> dict[str, Any] | None:
    ui_theme.section("BUILD AND DOWNLOAD", "\U0001f4e6")

    bundle = _cached_bundle(result)
    if bundle is None:
        st.markdown(
            "Three files, all built from this run and nothing else: a formatted "
            "**Excel workbook** for further analysis, a self-contained **HTML "
            "tearsheet** that prints straight to PDF, and the **positions CSV** "
            "for a database or a spreadsheet of your own."
        )
        st.caption(
            "Nothing is generated until you ask for it. Building the workbook "
            "and rendering the tearsheet charts costs a second or two, and "
            "Streamlit re-runs this script on every interaction - so the bytes "
            "are built once and held for this analysis run."
        )
        if not st.button("Build export bundle", type="primary", key="reports_build_bundle"):
            return None
        with ui.loading("Rendering charts, formatting the workbook, writing the tearsheet..."):
            bundle = _build_bundle(result, cfg)
        st.session_state[_BUNDLE_KEY] = bundle

    stamp = _run_stamp(result)
    base = f"quantfolio_{stamp:%Y%m%d}"

    ui.download_row(
        [
            ("Excel workbook (.xlsx)", bundle.get("xlsx"), f"{base}.xlsx", _XLSX_MIME),
            ("HTML tearsheet (.html)", bundle.get("html"), f"{base}.html", _HTML_MIME),
            ("Positions (.csv)", bundle.get("csv"), f"{base}_positions.csv", _CSV_MIME),
        ]
    )

    positions = result.positions
    n_rows = len(positions) if isinstance(positions, pd.DataFrame) else 0
    n_cols = len(positions.columns) if isinstance(positions, pd.DataFrame) else 0
    sheets = bundle.get("sheets") or []

    ui.stat_grid(
        [
            ("Workbook size", _filesize(bundle.get("xlsx"))),
            ("Sheets written", str(len(sheets)) if sheets else DASH),
            ("Tearsheet size", _filesize(bundle.get("html"))),
            ("Charts embedded", str(len(bundle.get("figures") or []))),
            ("CSV size", _filesize(bundle.get("csv"))),
            ("Rows x columns", f"{n_rows} x {n_cols}" if n_rows else DASH),
            ("Built at", f"{fmt_datetime(bundle.get('built_at'))} UTC"),
            ("Analysis stamp", f"{fmt_datetime(stamp)} UTC"),
        ],
        columns=4,
    )

    for label, message in bundle.get("errors") or []:
        st.error(f"{label} could not be built - {message}")

    st.caption(
        f"All three filenames carry the analysis date ({stamp:%Y%m%d}), so "
        "successive downloads do not overwrite one another and a file found "
        "later still says which run produced it. The bundle is cached against "
        "this run; press Analyze again and it is rebuilt from the new numbers."
    )
    return bundle


# ---------------------------------------------------------------------------
# 4. what is inside each file
# ---------------------------------------------------------------------------
def _workbook_preview(bundle: dict[str, Any] | None) -> None:
    with st.expander("Inside the Excel workbook - sheet by sheet", expanded=False):
        st.markdown(
            "Sheets are written in a fixed order; a sheet whose section of the "
            "analysis is missing is **omitted rather than left blank**, so the "
            "workbook is always a subsequence of the list below. Money columns "
            "are in the base currency, per-share prices stay in each "
            "instrument's own currency, and an undefined statistic is written "
            "as an empty cell - never as a zero."
        )
        # Read back out of the archive that was actually built, so the status
        # column reports the file rather than a second guess at the analysis.
        written = set(bundle.get("sheets") or []) if bundle else set()
        if bundle is None:
            status = dict.fromkeys(excel.SHEETS, "Build the bundle to check")
        elif not written:
            status = dict.fromkeys(excel.SHEETS, "Workbook could not be inspected")
        else:
            status = {
                name: ("Written" if name in written else "Omitted - no data")
                for name in excel.SHEETS
            }
        rows = [
            {
                "Sheet": name,
                "Contents": _SHEET_NOTES.get(name, ""),
                "This run": status[name],
            }
            for name in excel.SHEETS
        ]
        ui.dataframe(
            pd.DataFrame(rows),
            column_config={
                "Sheet": st.column_config.TextColumn(width="small"),
                "Contents": st.column_config.TextColumn(width="large"),
                "This run": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(
            "Numbers are written as numbers with an Excel number format, not "
            "as pre-formatted text, so every column sorts, filters and charts "
            "natively. Percentages carry their unit in the header because the "
            "cell itself cannot: risk statistics are stored as decimals, the "
            "Ulcer index and the position table's return columns as percent."
        )


def _chart_status(
    bundle: dict[str, Any] | None,
    heading: str,
    builder: Callable[[], go.Figure] | None,
    reason: str,
) -> str:
    """Whether one panel made it into the document, and why it did not."""
    if bundle is None:
        return "Available" if builder is not None else f"Skipped - {reason}"
    if heading in set(bundle.get("figures") or []):
        return "Embedded"
    if builder is None:
        return f"Skipped - {reason}"
    return "Skipped - the chart could not be rendered"


def _tearsheet_preview(result: Any, cfg: Any, bundle: dict[str, Any] | None) -> None:
    with st.expander("Inside the HTML tearsheet - section by section", expanded=False):
        st.markdown(
            "One file, no assets: the stylesheet is inlined and the charts are "
            "embedded, so the document opens correctly from an email "
            "attachment, a USB stick or a folder with no network. The only "
            "external request is the Plotly bundle the interactive charts need "
            "- open it once online and it renders everywhere after."
        )
        ui.dataframe(
            pd.DataFrame([{"Section": name, "Contents": note} for name, note in _REPORT_SECTIONS]),
            column_config={
                "Section": st.column_config.TextColumn(width="small"),
                "Contents": st.column_config.TextColumn(width="large"),
            },
        )

        chart_rows = [
            {
                "Chart": heading,
                "Shows": shows,
                "This run": _chart_status(bundle, heading, builder, reason),
            }
            for heading, shows, builder, reason in _figure_plan(result, cfg)
        ]
        st.caption("Charts, in the order they appear in the document:")
        ui.dataframe(
            pd.DataFrame(chart_rows),
            column_config={
                "Chart": st.column_config.TextColumn(width="medium"),
                "Shows": st.column_config.TextColumn(width="large"),
                "This run": st.column_config.TextColumn(width="medium"),
            },
        )
        st.caption(
            "Figures are copied and recoloured for white paper before they are "
            "embedded - the dark terminal palette prints as a solid black "
            "rectangle and costs a fortune in toner. The copy is what changes; "
            "the charts on screen are untouched."
        )


def _csv_preview(result: Any) -> None:
    with st.expander("Inside the positions CSV - column by column", expanded=False):
        positions = result.positions
        columns = [str(c) for c in positions.columns] if isinstance(positions, pd.DataFrame) else []
        st.markdown(
            "The positions table exactly as the analytics layer built it - the "
            "input to every allocation, concentration and attribution number "
            "in the other two files. It is the raw feed, not a report: no "
            "totals row, no formatting, one row per priced holding."
        )
        if columns:
            st.markdown("Columns in this run: " + ", ".join(f"`{name}`" for name in columns) + ".")
        else:
            st.markdown(
                "No priced position survived this run, so the file is written "
                "with its header row and no data - openable, and honest about "
                "being empty."
            )
        st.caption(
            "Encoded UTF-8 with a byte-order mark, which is what makes Excel "
            "on Windows read non-ASCII tickers and company names correctly. "
            "Floats are written in fixed notation so nothing arrives as "
            "1.23e+06, and a missing value is an empty field rather than a "
            "zero - the distinction survives the round trip into your own "
            "tools."
        )


# ---------------------------------------------------------------------------
# 5. share and reproduce
# ---------------------------------------------------------------------------
def _reproducibility(result: Any, cfg: Any) -> None:
    ui_theme.section("SHARE AND REPRODUCE", "\U0001f9ea")

    risk = result.risk or {}
    summary = result.summary or {}
    stamp = _run_stamp(result)
    window = PERIOD_LABELS.get(getattr(cfg, "period", ""), getattr(cfg, "period", ""))
    n_holdings = int(summary.get("n_holdings", 0) or 0)
    n_obs = int(risk.get("n_obs", 0) or 0)

    ui.stat_grid(
        [
            ("Engine", f"{APP_NAME} {APP_VERSION}"),
            ("Generated", f"{fmt_datetime(stamp)} UTC"),
            ("History window", f"{window} ({getattr(cfg, 'period', DASH)})"),
            ("Trading days analysed", f"{result.n_days:,}"),
            ("Daily return observations", f"{n_obs:,}" if n_obs else DASH),
            ("Holdings priced", f"{n_holdings:,}" if n_holdings else DASH),
            (
                "Benchmark",
                f"{getattr(cfg, 'benchmark_name', DASH)} "
                f"({getattr(cfg, 'benchmark_symbol', DASH)})",
            ),
            ("Base currency", str(getattr(cfg, "base_currency", DASH))),
            ("FX conversion", "On" if getattr(cfg, "convert_fx", False) else "Off"),
            ("Risk-free rate", fmt_ratio_pct(getattr(cfg, "risk_free_rate", None))),
            (
                "VaR confidence",
                fmt_ratio_pct(risk.get("confidence", getattr(cfg, "var_confidence", None)), 0),
            ),
            (
                "Rolling window",
                f"{int(getattr(cfg, 'rolling_window', 0) or 0)} sessions",
            ),
        ],
        columns=4,
    )

    st.caption(
        f"These twelve lines are the run's fingerprint. Change any one of them "
        f"- a {window} window for a two-year one, "
        f"{getattr(cfg, 'benchmark_name', 'one index')} for another, the "
        f"{fmt_ratio_pct(getattr(cfg, 'risk_free_rate', None))} risk-free rate "
        f"for today's actual yield - and every ratio on the page moves. Both "
        f"artefacts carry this block, which is why a tearsheet can be argued "
        f"with and a screenshot cannot."
    )

    ui.methodology(
        "What reproduces exactly, and what does not",
        f"""
**Frozen inside the files.** The workbook and the tearsheet are a *record*.
Every number in them was computed at {fmt_datetime(stamp)} UTC from
{result.n_days:,} overlapping sessions and {n_obs:,} daily return observations,
and they will still say that in a year. Re-running the analysis will not
reproduce them, and it is not supposed to.

**Not frozen: the market.** Prices move, so a re-run tomorrow with identical
settings gives different numbers. Providers also revise history - splits,
dividends and adjusted closes are restated after the fact - so even the *past*
window can change underneath you. Source: {DATA_SOURCE}.

**Conventions applied to every statistic here.** {TRADING_DAYS} trading days per
year for annualisation; sample (n-1) standard deviations; historical simulation
for VaR at {fmt_ratio_pct(risk.get("confidence", getattr(cfg, "var_confidence", None)), 0)}
confidence; a {int(getattr(cfg, "rolling_window", 0) or 0)}-session rolling
window; and a {fmt_ratio_pct(getattr(cfg, "risk_free_rate", None))} risk-free
rate that you set - it is an input, not a live yield, and every Sharpe, Sortino,
Treynor and alpha in both files is measured against it.

**What the value series assumes.** Today's share counts are held constant back
through the whole window, so the history is of the portfolio you own now rather
than the trades you actually made. Where a trade ledger was supplied, the
money-weighted (XIRR) view that fixes this travels with the workbook's
Transactions sheet.

**What is not in any file.** Dividends received, brokerage, STT, stamp duty and
taxes. P&L is unrealised and measured against the average cost you entered.

**To reproduce this run:** same holdings, same window, same benchmark, same base
currency, same risk-free rate, same VaR confidence, same rolling window - and
accept that the prices underneath will have moved. The workbook is the only
copy that will not.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Export",
        "Take the analysis with you - workbook, printable tearsheet, raw positions",
        icon="\U0001f4e4",
    )

    result = guards.require_result()
    if result is None:
        return

    cfg = result.config

    ui.freshness_bar(
        result.generated_at,
        DATA_SOURCE,
        extra=[
            f"Base currency {getattr(cfg, 'base_currency', '')}",
            f"{result.n_days:,} sessions analysed",
            f"{len(result.warnings or [])} data note(s) carried into the files",
        ],
    )
    ui.warning_list(
        result.warnings,
        "Data notes travelling with these exports",
    )

    bundle = _downloads(result, cfg)

    ui_theme.spacer(0.4)
    ui_theme.section("WHAT IS IN EACH FILE", "\U0001f50e")
    _workbook_preview(bundle)
    _tearsheet_preview(result, cfg, bundle)
    _csv_preview(result)

    ui_theme.spacer(0.4)
    ui_theme.section("SCHEDULED E-MAIL", "✉")
    st.caption(
        "Have the tearsheet built and e-mailed on a cadence. The schedule is "
        "stored here; the clock is your operating system's, because Streamlit "
        "has none."
    )
    email_panel.render()

    ui_theme.spacer(0.4)
    _reproducibility(result, cfg)

    ui_theme.spacer(0.4)
    ui_theme.section("DISCLAIMER", "⚖")
    st.caption(DISCLAIMER)
    st.caption(
        "That sentence is stamped into both artefacts - the workbook's Summary "
        "sheet and document properties, and the tearsheet's footer - so it "
        "travels with the file to whoever opens it next."
    )
