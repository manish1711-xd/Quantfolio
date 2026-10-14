"""The two artefacts that leave the machine -- ``src.reporting.excel`` and
``src.reporting.html_report``.

Everything else in QuantFolio is read on the screen of the person who ran it.
These two writers produce files that get e-mailed, printed and opened on
somebody else's laptop, so their failure modes are different in kind: a
workbook that is *almost* a zip, a CSV that will not reload, a tearsheet that
loads the Plotly bundle four times, or a ticker from an uploaded CSV that
reaches the page as live markup. None of those raise an exception at build
time; all of them are only discovered by the recipient.

Verified here
-------------
* **The workbook is a real .xlsx.** ``build_workbook`` returns ``bytes`` that
  begin with the ``PK`` local-file-header magic and that ``zipfile`` opens with
  ``testzip() is None`` -- a CRC check of every member, which is what "valid"
  actually means for an OOXML file. The workbook is inspected through
  ``zipfile`` and the OOXML parts rather than through ``openpyxl``, which emits
  a warning about the xlsxwriter extension these workbooks carry.
* **Sheet order is the documented ``SHEETS`` order**, and a fully populated run
  produces all eight; a result whose sections are all missing still produces an
  openable workbook with ``Summary`` alone. A shorter workbook, never an error.
* **Missing is blank, never zero.** A NaN P&L writes no cell at all. Pinned by
  building the same portfolio twice -- once with ``NaN``, once with ``0.0`` --
  and showing the zero version carries exactly two extra numeric cells.
* **positions_csv round-trips through ``pandas.read_csv``** with its column
  order, its values and its NaNs intact, and is encoded ``utf-8-sig`` so Excel
  on Windows reads non-ASCII names correctly. ``None`` yields a header-only
  file, so a download button always produces something openable.
* **The HTML report is a complete, printable document**: a doctype, a closing
  ``</html>``, an ``@media print`` block, and the Plotly CDN bundle loaded
  **exactly once** no matter how many figures are embedded -- the regression
  that silently quadruples a report's weight.
* **Nothing from the user's CSV is trusted.** A holding named
  ``<script>alert(1)</script>`` must appear escaped in the HTML and
  XML-escaped in the workbook's shared-string table. A portfolio file is an
  untrusted document.

Deterministic and offline: the analysis is built once from the shared seeded
fixtures through the real analytics pipeline, exactly as ``app.py`` builds it.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import src.charts as charts

# ``tests/`` is on sys.path under pytest's default (prepend) import mode; the
# analysis below is expensive enough to want session scope, which rules out
# depending on the function-scoped conftest fixtures directly.
from conftest import make_prices, make_quote
from src.analytics import attribution as attr
from src.analytics import benchmark as bm
from src.analytics import drawdown as dd
from src.analytics import insights as ins
from src.analytics import lots as lot
from src.analytics import positions as pos
from src.analytics import returns as ret
from src.analytics import risk as rsk
from src.analytics import rolling as roll
from src.config import AppConfig
from src.models import AnalysisResult, SecurityProfile
from src.reporting import excel, html_report

TICKERS: tuple[str, ...] = ("AAA", "BBB", "CCC")
SECTORS: tuple[str | None, ...] = ("Energy", "Technology", None)
QUANTITIES: tuple[float, ...] = (10.0, 5.0, 8.0)
BUY_PRICES: tuple[float, ...] = (100.0, 140.0, 190.0)
NOW = datetime(2025, 9, 1, 10, 30, tzinfo=timezone.utc)

#: A ticker straight out of an uploaded CSV, written by somebody unfriendly.
EVIL_TICKER = "<script>alert(1)</script>"
EVIL_NAME = '<img src=x onerror="alert(2)">'
EVIL_WARNING = "<b>feed degraded</b>"

_PRICES = make_prices(TICKERS)
_BENCH = make_prices(("BMK",), seed=20240200).iloc[:, 0]


# ---------------------------------------------------------------------------
# a complete analysis, built the way app.py builds it
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def analysis() -> AnalysisResult:
    """A fully populated :class:`AnalysisResult` -- every section present."""
    cfg = AppConfig()
    prices = _PRICES
    quotes = {t: make_quote(t, float(prices[t].iloc[-1])) for t in TICKERS}
    profiles = {
        t: SecurityProfile(
            t,
            name=f"{t} Industries",
            sector=s,
            currency="INR",
            quote_type="ETF" if s is None else "EQUITY",
        )
        for t, s in zip(TICKERS, SECTORS)
    }
    holdings, _ = pos.clean_holdings(
        pd.DataFrame(
            {"ticker": list(TICKERS), "quantity": list(QUANTITIES), "buy_price": list(BUY_PRICES)}
        )
    )
    positions, _ = pos.build_positions(holdings, quotes, profiles, None, "INR")
    summary = pos.summarise(positions)

    values = rsk.portfolio_value_series(
        prices, dict(zip(positions["ticker"], positions["quantity"]))
    )
    returns = rsk.daily_returns(values)
    asset_returns = rsk.daily_returns(prices)
    bench_returns = rsk.daily_returns(_BENCH)

    risk = rsk.risk_summary(
        values,
        returns,
        positions["weight_pct"],
        len(positions),
        cfg.risk_free_rate,
        cfg.var_confidence,
    )
    correlation = rsk.correlation_matrix(asset_returns)
    covariance = rsk.covariance_matrix(asset_returns, shrinkage=cfg.optimizer.shrinkage)
    comparison = bm.compare(values, _BENCH, cfg.risk_free_rate)
    episodes = dd.drawdown_episodes(values)
    attribution = attr.attribution_summary(positions, asset_returns, covariance)

    txns, _ = lot.parse_transactions(
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "AAA"],
                "trade_date": ["2023-01-10", "2023-06-15", "2023-07-03", "2024-08-20"],
                "action": ["BUY", "BUY", "BUY", "SELL"],
                "quantity": [10, 15, 12, 5],
                "price": [95.0, 110.0, 150.0, 140.0],
                "fees": [9.5, 11.0, 15.0, 14.0],
            }
        )
    )
    open_lots, realised, _ = lot.build_lots(txns)
    ledger = {
        "transactions": txns,
        "open_lots": open_lots,
        "realised": realised,
        "realised_summary": lot.realised_summary(realised),
        "cash_flows": lot.cash_flows(txns),
        "xirr": ret.portfolio_xirr(txns, float(summary["current_value"]), NOW.date()),
    }
    score = rsk.risk_score(
        risk["ann_vol"], risk["mdd"], positions["weight_pct"], len(positions), risk["cvar"]
    )

    return AnalysisResult(
        positions=positions,
        summary=summary,
        quotes=quotes,
        profiles=profiles,
        prices=prices,
        values=values,
        returns=returns,
        asset_returns=asset_returns,
        benchmark_prices=_BENCH,
        risk=risk,
        benchmark=comparison,
        attribution=attribution,
        drawdowns=episodes,
        correlation=correlation,
        covariance=covariance,
        rolling=roll.rolling_frame(
            returns, bench_returns, cfg.rolling_window, cfg.risk_free_rate, cfg.var_confidence
        ),
        monthly=ret.monthly_return_table(values),
        return_stats=ret.return_stats(values),
        ledger=ledger,
        score=score,
        insights=ins.generate_insights(
            positions, summary, risk, comparison, correlation, episodes, attribution
        ),
        fx={},
        warnings=["synthetic prices"],
        config=cfg,
        generated_at=NOW,
    )


@pytest.fixture(scope="session")
def workbook(analysis: AnalysisResult) -> bytes:
    return excel.build_workbook(analysis, analysis.config)


@pytest.fixture(scope="session")
def report(analysis: AnalysisResult) -> str:
    return html_report.build_html_report(analysis, analysis.config, None, "Core Equity")


@pytest.fixture
def hostile_positions() -> pd.DataFrame:
    """A positions table carrying markup in every string column it has."""
    return pd.DataFrame(
        {
            "ticker": [EVIL_TICKER, "BBB"],
            "name": [EVIL_NAME, "Beta & Company"],
            "sector": ["Energy", None],
            "quantity": [10.0, 5.0],
            "buy_price": [100.0, 140.0],
            "current_price": [120.0, 130.0],
            "currency": ["INR", "INR"],
            "fx_rate": [1.0, 1.0],
            "investment": [1000.0, 700.0],
            "current_value": [1200.0, 650.0],
            "pnl": [200.0, np.nan],
            "pnl_pct": [20.0, np.nan],
            "weight_pct": [64.86, 35.14],
        }
    )


@pytest.fixture
def empty_result() -> AnalysisResult:
    """Every section missing -- the state after an analysis that found nothing."""
    return AnalysisResult(positions=pd.DataFrame())


# ---------------------------------------------------------------------------
# helpers -- the workbook is inspected as OOXML, never through openpyxl
# ---------------------------------------------------------------------------
def _archive(book: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(book))


def _sheet_names(book: bytes) -> list[str]:
    xml = _archive(book).read("xl/workbook.xml").decode("utf-8")
    return re.findall(r'<sheet name="([^"]+)"', xml)


def _shared_strings(book: bytes) -> str:
    archive = _archive(book)
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ""
    return archive.read("xl/sharedStrings.xml").decode("utf-8")


def _numeric_cells(book: bytes, sheet_index: int) -> list[str]:
    xml = _archive(book).read(f"xl/worksheets/sheet{sheet_index}.xml").decode("utf-8")
    return re.findall(r"<v>([^<]*)</v>", xml)


def _positions_only(frame: pd.DataFrame) -> AnalysisResult:
    return AnalysisResult(positions=frame, config=AppConfig(), generated_at=NOW)


def _sample_figures(n: int) -> dict[str, go.Figure]:
    """``n`` genuine figures, built from a tiny deterministic value series."""
    index = pd.bdate_range("2023-01-02", periods=40)
    values = pd.Series(np.linspace(100.0, 130.0, 40), index=index)
    built = {
        "Growth of the portfolio": charts.growth_line(values),
        "Drawdown": charts.drawdown_chart(values / values.cummax() - 1.0),
        "Underwater": charts.underwater_chart(values),
        "Cumulative return": charts.cumulative_return_chart(values),
    }
    return dict(list(built.items())[:n])


# ---------------------------------------------------------------------------
# excel: the file is a real .xlsx
# ---------------------------------------------------------------------------
def test_the_workbook_is_bytes_beginning_with_the_zip_local_header_magic(
    workbook: bytes,
) -> None:
    """``PK\\x03\\x04`` is the first thing every OOXML reader looks for.

    Returning ``str`` or a ``BytesIO`` would sail through a download button and
    fail in the recipient's Excel, so the type is part of the contract.
    """
    assert isinstance(workbook, bytes)
    assert workbook[:4] == b"PK\x03\x04"
    assert len(workbook) > 5_000, "a formatted eight-sheet workbook is not tiny"


def test_every_member_of_the_workbook_passes_its_crc_check(workbook: bytes) -> None:
    """``testzip()`` returns the first corrupt member, or ``None`` if all are sound."""
    archive = _archive(workbook)

    assert archive.testzip() is None
    names = set(archive.namelist())
    assert {"[Content_Types].xml", "xl/workbook.xml", "docProps/core.xml"} <= names


def test_a_fully_populated_run_writes_every_documented_sheet_in_order(
    workbook: bytes,
) -> None:
    """``SHEETS`` is the published tab order; this result has data for all eight."""
    assert _sheet_names(workbook) == list(excel.SHEETS)


def test_the_sheets_written_are_always_a_subsequence_of_the_documented_order(
    analysis: AnalysisResult,
) -> None:
    """Sections without data are omitted; the survivors keep their relative order.

    Stripping the benchmark, the ledger and the correlation matrix must remove
    exactly those three tabs and leave the rest in ``SHEETS`` order -- never
    reshuffle the tab strip.
    """
    stripped = AnalysisResult(
        positions=analysis.positions,
        summary=analysis.summary,
        risk=analysis.risk,
        score=analysis.score,
        attribution=analysis.attribution,
        drawdowns=analysis.drawdowns,
        config=analysis.config,
        generated_at=NOW,
    )
    names = _sheet_names(excel.build_workbook(stripped, analysis.config))

    assert "Benchmark" not in names
    assert "Transactions" not in names
    assert "Correlation" not in names
    assert names == [s for s in excel.SHEETS if s in names]
    assert names[0] == "Summary", "Summary always leads the tab strip"


def test_an_entirely_empty_analysis_still_yields_an_openable_workbook(
    empty_result: AnalysisResult,
) -> None:
    """A run that found nothing must still download as a file the user can open."""
    book = excel.build_workbook(empty_result)

    assert book[:2] == b"PK"
    assert _archive(book).testzip() is None
    assert _sheet_names(book) == ["Summary"]


def test_a_missing_result_object_is_survivable() -> None:
    """``build_workbook`` accepts anything exposing the field names, including nothing."""
    book = excel.build_workbook(None)

    assert _archive(book).testzip() is None
    assert _sheet_names(book) == ["Summary"]


def test_a_missing_number_is_written_as_a_blank_cell_and_never_as_a_zero(
    hostile_positions: pd.DataFrame,
) -> None:
    """ "Unknown P&L" and "broke even" are different facts about a position.

    The same portfolio is written twice -- once with the second holding's P&L
    and return as NaN, once with both as 0.0. The zero version must carry
    exactly two more numeric cells, and both of them must be zeros.
    """
    with_nan = hostile_positions.copy()
    with_zero = hostile_positions.copy()
    with_zero.loc[with_zero.index[1], ["pnl", "pnl_pct"]] = 0.0

    nan_cells = _numeric_cells(excel.build_workbook(_positions_only(with_nan)), 2)
    zero_cells = _numeric_cells(excel.build_workbook(_positions_only(with_zero)), 2)

    extra = Counter(zero_cells) - Counter(nan_cells)
    assert len(zero_cells) - len(nan_cells) == 2
    assert list(extra.elements()) == ["0", "0"]


def test_a_hostile_ticker_reaches_the_workbook_as_xml_escaped_text(
    hostile_positions: pd.DataFrame,
) -> None:
    """A sheet is XML. An unescaped ``<script>`` would not be dangerous there --
    it would simply corrupt the part and make the file unopenable.
    """
    strings = _shared_strings(excel.build_workbook(_positions_only(hostile_positions)))

    assert EVIL_TICKER not in strings
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in strings


def test_two_exports_of_one_analysis_differ_only_in_the_file_creation_stamp(
    analysis: AnalysisResult, workbook: bytes
) -> None:
    """A fixed ``generated_at`` must give a reproducible file.

    Every OOXML part that carries data -- worksheets, shared strings, styles,
    the workbook itself -- must be byte-identical across two exports of one
    analysis. Anything else would mean wall-clock time, dict ordering or a
    palette lookup had leaked into the numbers, and would make a versioned
    artefact diff against itself.

    ``docProps/core.xml`` is the documented exception: xlsxwriter stamps
    ``dcterms:created`` with the wall clock, which QuantFolio does not set and
    which says nothing about the analysis.
    """
    again = excel.build_workbook(analysis, analysis.config)
    first, second = _archive(workbook), _archive(again)

    assert first.namelist() == second.namelist()
    for part in first.namelist():
        if part == "docProps/core.xml":
            continue
        assert first.read(part) == second.read(part), part


# ---------------------------------------------------------------------------
# excel: the raw-data CSV
# ---------------------------------------------------------------------------
def test_the_positions_csv_round_trips_through_pandas(analysis: AnalysisResult) -> None:
    """The file exists to be re-loaded, so re-loading it is the whole test."""
    payload = excel.positions_csv(analysis.positions)
    reloaded = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")

    assert isinstance(payload, bytes)
    assert list(reloaded.columns) == list(analysis.positions.columns)
    assert len(reloaded) == len(analysis.positions)
    assert list(reloaded["ticker"]) == list(analysis.positions["ticker"])
    for column in ("quantity", "investment", "current_value", "pnl", "weight_pct"):
        assert reloaded[column].to_numpy() == pytest.approx(
            analysis.positions[column].to_numpy(), rel=1e-5
        ), column


def test_the_positions_csv_carries_the_byte_order_mark_excel_needs() -> None:
    """``utf-8-sig``: without the BOM, Excel on Windows mangles non-ASCII names."""
    payload = excel.positions_csv(
        pd.DataFrame({"ticker": ["TATAMOTORS.NS"], "name": ["Tata Motors ₹"]})
    )

    assert payload.startswith(b"\xef\xbb\xbf")
    assert payload.decode("utf-8-sig").startswith("ticker,")
    assert "₹" in payload.decode("utf-8-sig")


def test_a_missing_value_survives_the_csv_round_trip_as_a_missing_value(
    hostile_positions: pd.DataFrame,
) -> None:
    """A NaN P&L is written as an empty field, so it reloads as NaN, not as 0."""
    reloaded = pd.read_csv(io.BytesIO(excel.positions_csv(hostile_positions)), encoding="utf-8-sig")

    assert reloaded["pnl"].isna().tolist() == [False, True]
    assert reloaded["pnl"].iloc[0] == pytest.approx(200.0)


def test_an_absent_positions_table_still_produces_a_header_only_csv() -> None:
    """A download always produces something openable, never a zero-byte file."""
    text = excel.positions_csv(None).decode("utf-8-sig")

    header = text.splitlines()[0].split(",")
    assert header[:5] == ["ticker", "name", "sector", "industry", "quantity"]
    assert "weight_pct" in header
    assert len(text.splitlines()) == 1, "header only -- no fabricated rows"


def test_the_csv_keeps_the_documented_column_order_and_appends_the_rest() -> None:
    """``build_positions`` order first; anything the caller added stays on the right."""
    frame = pd.DataFrame(
        {
            "weight_pct": [100.0],
            "ticker": ["AAA"],
            "custom_note": ["kept"],
            "quantity": [1.0],
        }
    )
    header = excel.positions_csv(frame).decode("utf-8-sig").splitlines()[0].split(",")

    assert header.index("ticker") < header.index("quantity") < header.index("weight_pct")
    assert header[-1] == "custom_note"


# ---------------------------------------------------------------------------
# html: a complete, printable document
# ---------------------------------------------------------------------------
def test_the_report_is_a_complete_html_document(report: str) -> None:
    """Doctype in, ``</html>`` out. Anything less is a fragment, not a file."""
    assert isinstance(report, str)
    assert report[:15].lower() == "<!doctype html>"
    assert report.rstrip().endswith("</html>")
    assert '<meta charset="utf-8">' in report
    assert '<html lang="en">' in report


def test_the_report_carries_a_print_stylesheet(report: str) -> None:
    """ "Save as PDF" is the intended export path, so the print rules must ship."""
    assert report.count("@media print") == 1
    assert "page-break-inside" in report


def test_the_report_is_self_contained_apart_from_the_plotly_bundle(report: str) -> None:
    """All styling is inline, so the file renders from a USB stick.

    With no figures there is no external reference at all.
    """
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', report)

    assert external == [], external
    assert "<style>" in report


@pytest.mark.parametrize("n_figures", [1, 2, 3, 4])
def test_the_plotly_bundle_is_requested_exactly_once_however_many_figures(
    analysis: AnalysisResult, n_figures: int
) -> None:
    """Only the FIRST figure embeds the CDN bundle; the rest reuse it.

    Getting this wrong costs a multi-megabyte script tag per chart and is
    invisible on a fast connection -- exactly the kind of regression that
    reaches a recipient before it reaches a developer.
    """
    figures = _sample_figures(n_figures)
    document = html_report.build_html_report(analysis, analysis.config, figures)

    assert document.count("cdn.plot.ly") == 1
    assert len(re.findall(r'id="qf-figure-\d+"', document)) == n_figures
    for heading in figures:
        assert f"<h3>{heading}</h3>" in document


def test_a_report_with_no_figures_loads_no_plotly_bundle_at_all(
    analysis: AnalysisResult,
) -> None:
    """A statistics-only tearsheet has nothing to run, so it should download nothing."""
    document = html_report.build_html_report(analysis, analysis.config, None)

    assert document.count("cdn.plot.ly") == 0
    assert "<h2>Charts</h2>" not in document


def test_embedding_a_figure_does_not_mutate_the_caller_s_figure(
    analysis: AnalysisResult,
) -> None:
    """The report recolours a *copy* for light paper.

    Recolouring in place would leave the dashboard's own figure repainted for
    print the moment a user clicked Export.
    """
    figure = _sample_figures(1)["Growth of the portfolio"]
    before = figure.to_json()

    html_report.build_html_report(analysis, analysis.config, {"Growth": figure})

    assert figure.to_json() == before


def test_the_report_survives_a_completely_empty_analysis(
    empty_result: AnalysisResult,
) -> None:
    """No positions, no risk, no benchmark -- still a valid, printable document."""
    document = html_report.build_html_report(empty_result)

    assert document[:15].lower() == "<!doctype html>"
    assert document.rstrip().endswith("</html>")
    assert "@media print" in document


def test_report_bytes_is_the_document_encoded_as_utf_eight(
    analysis: AnalysisResult, report: str
) -> None:
    """The document declares ``charset=utf-8``, so the bytes must match it."""
    payload = html_report.report_bytes(analysis, analysis.config, None, "Core Equity")

    assert isinstance(payload, bytes)
    assert payload == report.encode("utf-8")


# ---------------------------------------------------------------------------
# html: nothing data-derived is trusted
# ---------------------------------------------------------------------------
def test_a_hostile_ticker_is_escaped_and_never_reaches_the_page_as_markup(
    hostile_positions: pd.DataFrame,
) -> None:
    """A portfolio CSV is an untrusted document.

    ``<script>alert(1)</script>`` in the ticker column must appear only in its
    escaped form; the literal string must not survive anywhere in the output,
    or the report is one upload away from script injection.
    """
    result = _positions_only(hostile_positions)
    result.warnings = [EVIL_WARNING]
    document = html_report.build_html_report(result, AppConfig())

    assert EVIL_TICKER not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert EVIL_NAME not in document
    assert EVIL_WARNING not in document
    assert "&lt;b&gt;feed degraded&lt;/b&gt;" in document


def test_an_ampersand_in_a_security_name_is_escaped_into_an_entity(
    hostile_positions: pd.DataFrame,
) -> None:
    """ "Beta & Company" must not open a stray entity in the middle of a table cell."""
    document = html_report.build_html_report(_positions_only(hostile_positions), AppConfig())

    assert "Beta &amp; Company" in document
    assert "Beta & Company" not in document


def test_a_hostile_portfolio_name_is_escaped_in_the_document_title(
    analysis: AnalysisResult,
) -> None:
    """The title is the browser tab and the default PDF filename."""
    document = html_report.build_html_report(analysis, analysis.config, None, "My <b>Book</b>")

    title = re.findall(r"<title>(.*?)</title>", document)[0]
    assert "<b>" not in title
    assert title.startswith("My &lt;b&gt;Book&lt;/b&gt;")


def test_a_non_figure_value_in_the_figures_mapping_is_skipped_not_embedded(
    analysis: AnalysisResult,
) -> None:
    """One bad entry must not cost the report, and must not be rendered as text."""
    document = html_report.build_html_report(analysis, analysis.config, {"Broken": "not a figure"})

    assert document.count("cdn.plot.ly") == 0
    assert document.rstrip().endswith("</html>")


# ---------------------------------------------------------------------------
# html: the download filename
# ---------------------------------------------------------------------------
def test_the_filename_slugifies_the_portfolio_name_and_stamps_the_date() -> None:
    """ "Core Equity" on 2025-09-01 -> ``quantfolio-core-equity-20250901.html``."""
    assert html_report.report_filename("Core Equity", NOW) == (
        "quantfolio-core-equity-20250901.html"
    )


@pytest.mark.parametrize("name", [None, "", "   ", "///", "!!!"])
def test_a_nameless_portfolio_still_gets_a_usable_filename(name: str | None) -> None:
    """A slug that reduces to nothing falls back to ``portfolio``, never to ``--``."""
    filename = html_report.report_filename(name, NOW)

    assert filename == "quantfolio-portfolio-20250901.html"


def test_the_filename_keeps_only_characters_every_filesystem_accepts() -> None:
    """Windows, macOS and Linux all take ``[a-z0-9-]``; nothing else is guaranteed."""
    filename = html_report.report_filename("Ünïcode / Portfolio!!", NOW)

    assert re.fullmatch(r"quantfolio-[a-z0-9-]+-\d{8}\.html", filename), filename
    assert "//" not in filename and not filename.startswith("quantfolio--")
