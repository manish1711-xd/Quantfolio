"""Tests for the exchange-file download layer.

No socket is opened: ``urllib.request.build_opener`` is replaced by a fake
that serves canned bytes, which is the only way to assert the things that
matter about this module — the User-Agent it sends, that it walks back over
non-trading days, and that it refuses a 200 response that is not actually a
file.

The last of those is the whole reason the schema contract exists. Asked for a
future date, BSE serves an HTML error page with status 200, and
``csv.DictReader`` parses it into rows of ``None``. Accepting that built the
turnover signal from nothing and left ~4,900 symbols ranked at zero.
"""

from __future__ import annotations

import io
import urllib.error
import zipfile
from datetime import date
from typing import Any

import pytest

from src.data import exchange as ex

BHAVCOPY_CSV = (
    "TradDt,BizDt,Sgmt,ISIN,TckrSymb,SctySrs,FinInstrmNm,TtlTrfVal\n"
    "2026-09-02,2026-09-02,CM,INE002A01018,RELIANCE,EQ,RELIANCE INDUSTRIES,17668058961.00\n"
    "2026-09-02,2026-09-02,CM,INE467B01029,TCS,EQ,TATA CONSULTANCY,5923000000.00\n"
)

NSE_LISTING_CSV = (
    "SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, ISIN NUMBER\n"
    "RELIANCE,Reliance Industries Limited, EQ, 29-NOV-1995, INE002A01018\n"
    "TCS,Tata Consultancy Services Limited, EQ, 25-AUG-2004, INE467B01029\n"
)

BSE_LISTING_JSON = (
    '[{"SCRIP_CD":"500325","scrip_id":"RELIANCE","Scrip_Name":"RELIANCE",'
    '"Issuer_Name":"Reliance Industries Ltd","ISIN_NUMBER":"INE002A01018",'
    '"INDUSTRY":"Refineries","Status":"Active","GROUP":"A"}]'
)

#: What BSE actually returns for a date that has not happened yet. Kept
#: multi-line on purpose: that is what makes ``csv.DictReader`` yield ROWS
#: (71 of them, from the real page) instead of nothing, which is precisely why
#: a row-count check was not enough and the column contract exists.
HTML_ERROR_PAGE = """<!DOCTYPE html><html lang="en" data-critters-container="">
<head><title>Error</title></head>
<body>
<div>Not found</div>
</body>
</html>"""


def zipped(payload: str, name: str = "BhavCopy.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"Content-Type": "text/csv"}
        self.status = 200

    def read(self, amount: int | None = None) -> bytes:
        return self._body if amount is None else self._body[:amount]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    """Serves canned bytes and records every request it was asked to make."""

    def __init__(self, routes: dict[str, Any], default: Any = None) -> None:
        self.routes = routes
        self.default = default
        self.addheaders: list[tuple[str, str]] = []
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float | None = None) -> FakeResponse:
        url = request if isinstance(request, str) else request.full_url
        self.requests.append(request)
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                body = payload.encode("utf-8") if isinstance(payload, str) else payload
                return FakeResponse(body)
        if self.default is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        if isinstance(self.default, Exception):
            raise self.default
        body = self.default.encode("utf-8") if isinstance(self.default, str) else self.default
        return FakeResponse(body)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Install a fake opener; the test supplies the routes."""
    holder: dict[str, FakeOpener] = {}

    def install(routes: dict[str, Any], default: Any = None) -> FakeOpener:
        opener = FakeOpener(routes, default)
        holder["opener"] = opener
        monkeypatch.setattr(ex.urllib.request, "build_opener", lambda *_a, **_k: opener)
        return opener

    return install


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def test_a_browser_user_agent_is_sent(wire) -> None:
    """Not a disguise: NSE's CDN answers 403 to ``Python-urllib``, so without
    this the feature simply does not work."""
    opener = wire({"EQUITY_L.csv": NSE_LISTING_CSV})

    ex.fetch_nse_listing()

    headers = dict(opener.addheaders)
    assert "Mozilla" in headers["User-Agent"]
    assert headers["User-Agent"] == ex.USER_AGENT


def test_the_home_page_is_fetched_first_for_cookies(wire) -> None:
    opener = wire({"EQUITY_L.csv": NSE_LISTING_CSV, "www.nseindia.com/": "<html/>"})

    ex.fetch_nse_listing()
    urls = [r if isinstance(r, str) else r.full_url for r in opener.requests]

    assert urls[0] == ex.NSE_HOME


def test_a_refused_warm_up_does_not_abort_the_download(wire) -> None:
    """Measured on the real endpoints: the warm-up is sometimes refused while
    the file download then succeeds anyway."""
    wire(
        {
            "www.nseindia.com/": urllib.error.HTTPError(ex.NSE_HOME, 403, "Forbidden", {}, None),  # type: ignore[arg-type]
            "EQUITY_L.csv": NSE_LISTING_CSV,
        }
    )

    result = ex.fetch_nse_listing()

    assert result.ok
    assert len(result.rows) == 2


def test_the_bse_listing_sends_a_referer(wire) -> None:
    opener = wire({"ListofScripData": BSE_LISTING_JSON})

    ex.fetch_bse_listing()
    referers = [
        r.get_header("Referer")
        for r in opener.requests
        if not isinstance(r, str) and r.get_header("Referer")
    ]

    assert referers and "bseindia.com" in referers[0]


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("no route to host"),
        TimeoutError("timed out"),
        OSError("connection reset"),
    ],
)
def test_an_unreachable_host_is_reported_not_raised(wire, failure: Exception) -> None:
    """A refresh that fails must leave the existing universe alone rather than
    crash the build script."""
    wire({"EQUITY_L.csv": failure})

    result = ex.fetch_nse_listing()

    assert not result.ok
    assert "could not be reached" in result.notice


def test_an_http_error_names_the_status(wire) -> None:
    wire(
        {
            "EQUITY_L.csv": urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
        }
    )

    result = ex.fetch_nse_listing()

    assert "HTTP 503" in result.notice


def test_a_404_is_described_as_not_published(wire) -> None:
    """Overwhelmingly a non-trading day, not a broken URL -- said plainly so
    the walk-back is not mistaken for a retry storm."""
    wire({})

    result = ex.fetch_nse_bhavcopy(date(2026, 9, 2))

    assert "not published for that date" in result.notice


def test_an_empty_response_is_reported(wire) -> None:
    wire({"EQUITY_L.csv": b""})

    assert "empty response" in ex.fetch_nse_listing().notice


def test_an_oversized_response_is_discarded(wire, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 64 MB body means something went wrong upstream, and streaming it into
    memory is the wrong way to find out."""
    monkeypatch.setattr(ex, "MAX_DOWNLOAD_BYTES", 100)
    wire({"EQUITY_L.csv": "x" * 500})

    result = ex.fetch_nse_listing()

    assert not result.ok
    assert "exceeded" in result.notice


# ---------------------------------------------------------------------------
# the schema contract
# ---------------------------------------------------------------------------
def test_an_html_error_page_served_as_200_is_refused(wire) -> None:
    """The trap this guard exists for. BSE returns this for a future date, and
    csv.DictReader parses it into 71 rows of None."""
    wire({"BhavCopy_BSE": HTML_ERROR_PAGE})

    result = ex.fetch_bse_bhavcopy(date(2026, 9, 2))

    assert not result.ok
    assert "does not look like the expected file" in result.notice
    assert "TckrSymb" in result.notice


def test_the_contract_names_the_missing_columns(wire) -> None:
    wire({"EQUITY_L.csv": "SYMBOL,SOMETHING\nAAA,1\n"})

    notice = ex.fetch_nse_listing().notice

    assert "NAME OF COMPANY" in notice
    assert "ISIN NUMBER" in notice


def test_a_correctly_shaped_file_passes_the_contract(wire) -> None:
    wire({"EQUITY_L.csv": NSE_LISTING_CSV})

    assert ex.fetch_nse_listing().ok


def test_the_bse_listing_must_carry_a_trading_symbol(wire) -> None:
    """``scrip_id`` is load-bearing: Yahoo addresses a BSE line by it, and by
    nothing else."""
    wire({"ListofScripData": '[{"SCRIP_CD":"500325","Issuer_Name":"Reliance"}]'})

    notice = ex.fetch_bse_listing().notice

    assert "scrip_id" in notice


def test_json_that_is_not_a_list_is_reported(wire) -> None:
    wire({"ListofScripData": '{"error": "nope"}'})

    assert "no scrip records" in ex.fetch_bse_listing().notice


def test_unreadable_json_is_reported(wire) -> None:
    wire({"ListofScripData": "not json at all"})

    assert "not readable JSON" in ex.fetch_bse_listing().notice


# ---------------------------------------------------------------------------
# headers with leading spaces
# ---------------------------------------------------------------------------
def test_the_nse_header_spaces_are_stripped(wire) -> None:
    """NSE's own header is ``SYMBOL,NAME OF COMPANY, SERIES, ISIN NUMBER`` --
    with a leading space on every column but the first, which silently breaks
    any lookup by name."""
    wire({"EQUITY_L.csv": NSE_LISTING_CSV})

    row = ex.fetch_nse_listing().rows[0]

    assert row["SERIES"] == "EQ"
    assert row["ISIN NUMBER"] == "INE002A01018"


def test_values_are_stripped_too(wire) -> None:
    wire({"EQUITY_L.csv": NSE_LISTING_CSV})

    assert ex.fetch_nse_listing().rows[0]["NAME OF COMPANY"] == "Reliance Industries Limited"


# ---------------------------------------------------------------------------
# zipped bhavcopy
# ---------------------------------------------------------------------------
def test_a_zipped_nse_bhavcopy_is_unpacked(wire) -> None:
    wire({"BhavCopy_NSE": zipped(BHAVCOPY_CSV)})

    result = ex.fetch_nse_bhavcopy(date(2026, 9, 2))

    assert result.ok
    assert result.rows[0]["TckrSymb"] == "RELIANCE"
    assert result.as_of == date(2026, 9, 2)


def test_a_bse_bhavcopy_is_plain_csv(wire) -> None:
    """The two exchanges publish the same schema and different packaging."""
    wire({"BhavCopy_BSE": BHAVCOPY_CSV})

    assert ex.fetch_bse_bhavcopy(date(2026, 9, 2)).ok


def test_a_corrupt_zip_is_reported(wire) -> None:
    wire({"BhavCopy_NSE": b"PK\x03\x04 truncated"})

    assert "not a readable zip" in ex.fetch_nse_bhavcopy(date(2026, 9, 2)).notice


def test_a_zip_with_no_csv_is_reported(wire) -> None:
    wire({"BhavCopy_NSE": zipped("nothing", name="readme.txt")})

    assert "held no CSV" in ex.fetch_nse_bhavcopy(date(2026, 9, 2)).notice


def test_the_url_carries_the_requested_date() -> None:
    assert "20260902" in ex.nse_bhavcopy_url(date(2026, 9, 2))
    assert "20260902" in ex.bse_bhavcopy_url(date(2026, 9, 2))


# ---------------------------------------------------------------------------
# walking back to the last trading day
# ---------------------------------------------------------------------------
def test_a_weekend_is_skipped_without_a_request(wire) -> None:
    """Saturday and Sunday are never trading days on either exchange, so
    asking costs a round trip for nothing.

    Deliberately a weekend in the PAST: the future-date clamp runs first, so
    a forward-dated weekend would never reach the weekday rule at all.
    """
    opener = wire({"BhavCopy_NSE_CM_0_0_0_20260828": zipped(BHAVCOPY_CSV)})
    log: list[str] = []

    # 30 Aug 2026 is a Sunday, 29 Aug a Saturday, 28 Aug the Friday before.
    result = ex.latest_bhavcopy("NSE", on=date(2026, 8, 30), attempts=log)

    assert result.as_of == date(2026, 8, 28)
    assert sum("weekend, not requested" in line for line in log) == 2
    requested = [r.full_url for r in opener.requests if not isinstance(r, str)]
    assert not any("20260829" in u or "20260830" in u for u in requested)


def test_a_holiday_costs_one_404_and_steps_back(wire) -> None:
    """There is no published trading calendar in these files, so the only
    honest way to find the last trading day is to ask."""
    wire({"BhavCopy_NSE_CM_0_0_0_20260902": zipped(BHAVCOPY_CSV)})
    log: list[str] = []

    result = ex.latest_bhavcopy("NSE", on=date(2026, 9, 4), attempts=log)

    assert result.as_of == date(2026, 9, 2)
    assert any("20260904" in line or "04 Sep" in line for line in log)


def test_the_date_that_worked_is_reported_not_the_date_asked_for(wire) -> None:
    """So the universe records the day it was really built from."""
    wire({"BhavCopy_NSE_CM_0_0_0_20260902": zipped(BHAVCOPY_CSV)})

    result = ex.latest_bhavcopy("NSE", on=date(2026, 9, 4))

    assert result.as_of == date(2026, 9, 2)


def test_a_future_date_is_never_requested(wire) -> None:
    """Besides being pointless, BSE answers a future date with an HTML error
    page at status 200."""
    opener = wire({"BhavCopy_NSE": zipped(BHAVCOPY_CSV)})
    log: list[str] = []

    ex.latest_bhavcopy("NSE", on=date(2999, 1, 1), attempts=log)
    requested = [r.full_url for r in opener.requests if not isinstance(r, str)]

    assert any("in the future" in line for line in log)
    assert not any("29990101" in u for u in requested)


def test_giving_up_explains_how_far_it_looked(wire) -> None:
    wire({})

    result = ex.latest_bhavcopy("NSE", on=date(2026, 9, 2), look_back=3)

    assert not result.ok
    assert "no bhavcopy was published between" in result.notice
    assert "candidate(s) tried" in result.notice


def test_the_look_back_is_bounded(wire) -> None:
    """A failing endpoint must not turn into an unbounded retry loop."""
    opener = wire({})

    ex.latest_bhavcopy("NSE", on=date(2026, 9, 2), look_back=2)

    assert len(opener.requests) <= 3 * 2  # at most one warm-up + one fetch each


def test_a_bhavcopy_failing_the_contract_is_walked_past(wire) -> None:
    """The two guards compose: a 200-with-HTML on one day does not stop the
    walk finding a real file on an earlier one."""
    wire(
        {
            "BhavCopy_BSE_CM_0_0_0_20260902": BHAVCOPY_CSV,
            "BhavCopy_BSE": HTML_ERROR_PAGE,
        }
    )

    result = ex.latest_bhavcopy("BSE", on=date(2026, 9, 3))

    assert result.ok
    assert result.as_of == date(2026, 9, 2)


# ---------------------------------------------------------------------------
# the result type
# ---------------------------------------------------------------------------
def test_a_failure_is_falsy_and_carries_a_reason() -> None:
    result = ex.SourceFile(notice="the CDN refused")

    assert not result.ok
    assert "refused" in result.summary


def test_a_success_summarises_its_rows_and_date() -> None:
    result = ex.SourceFile(rows=({"a": "1"},), as_of=date(2026, 9, 2), source="NSE bhavcopy")

    assert "1 rows" in result.summary
    assert "02 Sep 2026" in result.summary
