"""Official NSE and BSE files: the second — and last — network boundary.

Why this is a separate module from :mod:`src.data.fetcher`
----------------------------------------------------------
``fetcher.py`` used to say "every network call in the whole project lives in
this module". That is no longer true, and the invariant has been restated
rather than quietly broken: **every network call lives in ``src/data/``, in
exactly two modules, and nothing outside this package may import ``urllib`` or
any HTTP client.**

The two are split because they are genuinely different problems:

===================  ==============================  ============================
                     ``fetcher.py``                  this module
===================  ==============================  ============================
provider             Yahoo Finance, via ``yfinance``  NSE and BSE, directly
what                 prices for symbols you name      the list of what exists
cadence              every few minutes                once a day at most
auth                 none                             a browser User-Agent
failure              one ticker has no data           the whole file is missing
consumed by          the analysis pipeline            an offline build step
===================  ==============================  ============================

Folding an exchange-file downloader into the yfinance wrapper would have made
the file that matters most harder to read, and it would not have preserved the
property the invariant exists for — that swapping the price provider is a
rewrite of one file. That property still holds.

No new dependency
-----------------
Everything here is stdlib: ``urllib.request`` with a cookie jar, ``zipfile``,
``csv``, ``json``. The project deliberately has no ``requests``, and one
User-Agent header plus a cookie jar is all these endpoints need — measured, not
assumed.

What is actually fetched, and why each one
------------------------------------------
* **The listings** (``EQUITY_L.csv``, BSE ``ListOfScripData``) are the
  authoritative "what is listed" files. They carry the **company name** and the
  **ISIN**, which is what name search and ISIN import need.
* **The bhavcopies** are the end-of-day trade files. They carry **turnover**,
  which is the only prominence signal available anywhere in this data — and
  without it a full-market universe ranks "reliance" alphabetically and puts
  Reliance Communications (₹0.3 cr traded) above Reliance Industries
  (₹1,767 cr). Measured on real files; see :mod:`src.universe_build`.

The bhavcopy alone is *not* sufficient for a universe: its ``FinInstrmNm`` is a
squashed trading name ("21ST CENTURY MGMT SERVICE") and it includes gold bonds
and government securities. The listings alone are not sufficient either,
because they carry no measure of size. Both are needed.

Honesty about what this returns
-------------------------------
* A bhavcopy exists only for a **trading day**. A request for a weekend or a
  holiday is a 404, so :func:`latest_bhavcopy` walks backwards day by day and
  **reports the date it actually got**. Nothing is interpolated, and no date is
  assumed to be a trading day.
* Every result carries its ``as_of`` date and the URL it came from, so the
  universe can state how stale it is instead of implying it is current.
* Nothing here raises for an absent file. A missing day, a 403 or a timeout
  comes back as an empty result with a reason, because a refresh that fails
  must leave the existing universe alone rather than truncate it.
* **A 200 is not a file.** Asked for a future date, BSE returns an HTML error
  page with status 200, which ``csv.DictReader`` happily parses into 71 rows
  of ``None``. Accepting that would have built the turnover signal from
  nothing and left ~4,900 symbols ranked at zero. So every download is
  checked against the columns it must have (:data:`BHAVCOPY_COLUMNS` and
  friends) and a future date is never requested at all. A schema contract
  rather than a row-count threshold: it also catches a changed URL, a
  redirect to a login page and a silently altered format.
"""

from __future__ import annotations

import csv
import http.cookiejar
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta

logger = logging.getLogger(__name__)

__all__ = [
    "USER_AGENT",
    "MAX_DOWNLOAD_BYTES",
    "DEFAULT_TIMEOUT",
    "BHAVCOPY_LOOK_BACK_DAYS",
    "BHAVCOPY_COLUMNS",
    "NSE_LISTING_COLUMNS",
    "BSE_LISTING_COLUMNS",
    "NSE_LISTING_URL",
    "BSE_LISTING_URL",
    "SourceFile",
    "nse_bhavcopy_url",
    "bse_bhavcopy_url",
    "fetch_nse_listing",
    "fetch_bse_listing",
    "fetch_nse_bhavcopy",
    "fetch_bse_bhavcopy",
    "latest_bhavcopy",
]

#: A browser User-Agent. Not a disguise: NSE's CDN answers 403 to the default
#: ``Python-urllib/3.x`` string, and these are public files a browser
#: downloads freely from the exchange's own "Downloads" page. Without it the
#: feature simply does not work.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: Refuse a response larger than this. The real files are 180 KB - 1.8 MB; a
#: 64 MB body means something has gone wrong upstream (an error page, a
#: redirect loop, a captive portal) and streaming it into memory would be the
#: wrong way to find out.
MAX_DOWNLOAD_BYTES: int = 64 * 1024 * 1024

#: Seconds. Generous, because these files are fetched at most daily by a build
#: step, and an exchange CDN under load is slow rather than broken.
DEFAULT_TIMEOUT: float = 60.0

#: How many days :func:`latest_bhavcopy` walks back before giving up. Long
#: enough to clear Diwali, which can close the market for several consecutive
#: days around a weekend.
BHAVCOPY_LOOK_BACK_DAYS: int = 10

#: Columns a UDiFF bhavcopy must carry to be usable. Checked on every
#: download, because a 200 response is not proof of a file.
BHAVCOPY_COLUMNS: tuple[str, ...] = ("TckrSymb", "SctySrs", "ISIN", "TtlTrfVal")

#: Columns the NSE listing must carry. The header has leading spaces on every
#: column but the first, which ``_csv_rows`` strips.
NSE_LISTING_COLUMNS: tuple[str, ...] = ("SYMBOL", "NAME OF COMPANY", "SERIES", "ISIN NUMBER")

#: Keys the BSE scrip master must carry. ``scrip_id`` is the load-bearing one:
#: it is the symbol Yahoo addresses a BSE line by.
BSE_LISTING_COLUMNS: tuple[str, ...] = ("scrip_id", "ISIN_NUMBER")

NSE_HOME = "https://www.nseindia.com/"
BSE_HOME = "https://www.bseindia.com/"

#: The full list of NSE-listed equities: SYMBOL, NAME OF COMPANY, SERIES,
#: DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE.
NSE_LISTING_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

#: BSE's active-equity scrip master. JSON, and the only place the BSE trading
#: symbol (``scrip_id``) is published alongside the name and ISIN -- which
#: matters because Yahoo addresses BSE lines by that symbol (``RELIANCE.BO``),
#: NOT by the numeric scrip code (``500325.BO`` resolves to nothing).
BSE_LISTING_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
)

_BSE_LISTING_REFERER = "https://www.bseindia.com/corporates/List_Scrips.html"

#: The UDiFF ("unified diff format") bhavcopy both exchanges publish since
#: 2024. One schema for both, which is why one parser serves them.
_NSE_BHAVCOPY = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv.zip"
)
_BSE_BHAVCOPY = (
    "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{stamp}_F_0000.CSV"
)


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFile:
    """One downloaded exchange file, parsed into records.

    Empty ``rows`` with a non-empty ``notice`` is the normal shape of a
    failure: a caller must be able to tell "the market was closed" from "the
    download was refused" from "there is nothing listed", and none of those
    should raise in a build script.
    """

    rows: tuple[dict[str, str], ...] = ()
    as_of: date | None = None
    exchange: str = ""
    source: str = ""
    url: str = ""
    notice: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.rows)

    @property
    def summary(self) -> str:
        if not self.rows:
            return f"{self.source or 'file'}: {self.notice or 'nothing returned'}"
        stamp = f" as of {self.as_of:%d %b %Y}" if self.as_of else ""
        return f"{self.source}: {len(self.rows):,} rows{stamp}"


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def _opener() -> urllib.request.OpenerDirector:
    """A fresh opener with its own cookie jar.

    Fresh per call rather than module-global: these are occasional downloads,
    and a long-lived cookie jar in a build tool is state that outlives its
    usefulness and can only cause confusion when a session expires.
    """
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Accept", "text/csv,application/json,application/zip,*/*"),
        ("Accept-Language", "en-US,en;q=0.9"),
    ]
    return opener


def _download(
    url: str, referer: str = "", warm_up: str = "", timeout: float = DEFAULT_TIMEOUT
) -> tuple[bytes, str]:
    """``(body, notice)``. Empty body plus a reason on any failure.

    ``warm_up`` fetches the exchange's home page first so the CDN sets its
    cookies. It is **best effort**: measured on the real endpoints, the warm-up
    itself is sometimes refused while the file download then succeeds anyway,
    so a failed warm-up is logged at debug level and never aborts the attempt.
    """
    opener = _opener()
    if warm_up:
        try:
            opener.open(urllib.request.Request(warm_up), timeout=timeout).read(2048)
        except Exception as exc:  # noqa: BLE001 - best effort by design
            logger.debug("Cookie warm-up on %s was refused (%s); continuing.", warm_up, exc)

    request = urllib.request.Request(url)
    if referer:
        request.add_header("Referer", referer)

    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Overwhelmingly a non-trading day, not a broken URL. Said plainly
            # so the caller's walk-back is not mistaken for a retry storm.
            return b"", f"not published for that date (HTTP 404) at {url}"
        return b"", f"HTTP {exc.code} {exc.reason} from {url}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return b"", f"could not be reached ({type(exc).__name__}: {exc}) at {url}"

    if len(body) > MAX_DOWNLOAD_BYTES:
        return b"", (
            f"the response from {url} exceeded "
            f"{MAX_DOWNLOAD_BYTES // 1024 // 1024} MB and was discarded"
        )
    if not body:
        return b"", f"an empty response from {url}"
    return body, ""


def _csv_rows(body: bytes) -> tuple[dict[str, str], ...]:
    """Parse CSV bytes into records with stripped keys and values.

    Keys are stripped because NSE's ``EQUITY_L.csv`` header is
    ``SYMBOL,NAME OF COMPANY, SERIES, ISIN NUMBER`` -- with leading spaces on
    every column but the first, which silently breaks any lookup by name.
    """
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    out: list[dict[str, str]] = []
    for record in reader:
        out.append(
            {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in record.items()
                if key is not None
            }
        )
    return tuple(out)


def _missing_columns(rows: tuple[dict[str, str], ...], needed: tuple[str, ...]) -> str:
    """A notice naming the absent columns, or ``""`` when the shape is right.

    The guard that catches an error page served with status 200. Checked
    against the first row's keys, since ``csv.DictReader`` gives every row the
    same fields.
    """
    if not rows:
        return "the file parsed to zero rows"
    present = set(rows[0])
    absent = [column for column in needed if column not in present]
    if not absent:
        return ""
    got = ", ".join(sorted(present)[:6]) or "nothing"
    return (
        f"the download does not look like the expected file: missing "
        f"{', '.join(absent)} (found: {got}...). This usually means an error "
        f"page was served with status 200, or the format changed."
    )


def _unzip_single_csv(body: bytes) -> tuple[bytes, str]:
    """The one CSV inside a zip. ``(payload, notice)``."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            return b"", f"the archive held no CSV (members: {archive.namelist()})"
        return archive.read(members[0]), ""
    except zipfile.BadZipFile as exc:
        return b"", f"the download was not a readable zip ({exc})"


# ---------------------------------------------------------------------------
# listings
# ---------------------------------------------------------------------------
def fetch_nse_listing(timeout: float = DEFAULT_TIMEOUT) -> SourceFile:
    """The full list of NSE-listed equities.

    ~2,500 rows across the EQ, BE and BZ series, each with a company name and
    an ISIN. No date: it is a current-state file, so ``as_of`` is left ``None``
    and the caller stamps its own download time.
    """
    body, notice = _download(NSE_LISTING_URL, warm_up=NSE_HOME, timeout=timeout)
    if notice:
        return SourceFile(
            exchange="NSE", source="NSE equity listing", url=NSE_LISTING_URL, notice=notice
        )

    rows = _csv_rows(body)
    problem = _missing_columns(rows, NSE_LISTING_COLUMNS)
    if problem:
        return SourceFile(
            exchange="NSE", source="NSE equity listing", url=NSE_LISTING_URL, notice=problem
        )
    return SourceFile(rows=rows, exchange="NSE", source="NSE equity listing", url=NSE_LISTING_URL)


def fetch_bse_listing(timeout: float = DEFAULT_TIMEOUT) -> SourceFile:
    """BSE's active-equity scrip master (JSON).

    ~5,000 rows, each with ``scrip_id`` (the symbol Yahoo uses), the issuer
    name, the ISIN, an industry label and a market capitalisation.
    """
    body, notice = _download(
        BSE_LISTING_URL, referer=_BSE_LISTING_REFERER, warm_up=BSE_HOME, timeout=timeout
    )
    if notice:
        return SourceFile(
            exchange="BSE", source="BSE scrip master", url=BSE_LISTING_URL, notice=notice
        )

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        return SourceFile(
            exchange="BSE",
            source="BSE scrip master",
            url=BSE_LISTING_URL,
            notice=f"the response was not readable JSON ({exc})",
        )

    if not isinstance(payload, list) or not payload:
        return SourceFile(
            exchange="BSE",
            source="BSE scrip master",
            url=BSE_LISTING_URL,
            notice="the response held no scrip records",
        )

    rows = tuple(
        {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in item.items()}
        for item in payload
        if isinstance(item, dict)
    )
    problem = _missing_columns(rows, BSE_LISTING_COLUMNS)
    if problem:
        return SourceFile(
            exchange="BSE", source="BSE scrip master", url=BSE_LISTING_URL, notice=problem
        )
    return SourceFile(rows=rows, exchange="BSE", source="BSE scrip master", url=BSE_LISTING_URL)


# ---------------------------------------------------------------------------
# bhavcopies
# ---------------------------------------------------------------------------
def nse_bhavcopy_url(on: date) -> str:
    return _NSE_BHAVCOPY.format(stamp=on.strftime("%Y%m%d"))


def bse_bhavcopy_url(on: date) -> str:
    return _BSE_BHAVCOPY.format(stamp=on.strftime("%Y%m%d"))


def fetch_nse_bhavcopy(on: date, timeout: float = DEFAULT_TIMEOUT) -> SourceFile:
    """The NSE end-of-day trade file for one date (zipped CSV)."""
    url = nse_bhavcopy_url(on)
    body, notice = _download(url, warm_up=NSE_HOME, timeout=timeout)
    if notice:
        return SourceFile(as_of=on, exchange="NSE", source="NSE bhavcopy", url=url, notice=notice)

    payload, unzip_notice = _unzip_single_csv(body)
    if unzip_notice:
        return SourceFile(
            as_of=on, exchange="NSE", source="NSE bhavcopy", url=url, notice=unzip_notice
        )

    rows = _csv_rows(payload)
    problem = _missing_columns(rows, BHAVCOPY_COLUMNS)
    if problem:
        return SourceFile(as_of=on, exchange="NSE", source="NSE bhavcopy", url=url, notice=problem)
    return SourceFile(rows=rows, as_of=on, exchange="NSE", source="NSE bhavcopy", url=url)


def fetch_bse_bhavcopy(on: date, timeout: float = DEFAULT_TIMEOUT) -> SourceFile:
    """The BSE end-of-day trade file for one date (plain CSV, not zipped)."""
    url = bse_bhavcopy_url(on)
    body, notice = _download(url, referer=BSE_HOME, warm_up=BSE_HOME, timeout=timeout)
    if notice:
        return SourceFile(as_of=on, exchange="BSE", source="BSE bhavcopy", url=url, notice=notice)

    rows = _csv_rows(body)
    problem = _missing_columns(rows, BHAVCOPY_COLUMNS)
    if problem:
        return SourceFile(as_of=on, exchange="BSE", source="BSE bhavcopy", url=url, notice=problem)
    return SourceFile(rows=rows, as_of=on, exchange="BSE", source="BSE bhavcopy", url=url)


@dataclass(frozen=True)
class _Attempt:
    on: date
    notice: str = ""


def latest_bhavcopy(
    exchange: str,
    on: date | None = None,
    look_back: int = BHAVCOPY_LOOK_BACK_DAYS,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: list[str] | None = None,
) -> SourceFile:
    """The most recent bhavcopy at or before ``on``.

    There is no published trading calendar in these files, so the only honest
    way to find the last trading day is to ask for one and see. Weekends are
    skipped without a request (they are never trading days on either
    exchange); every other miss costs one 404 and steps back a day.

    The returned :class:`SourceFile` carries the date that actually worked, so
    the universe records the day it was really built from rather than the day
    the build ran. ``attempts``, if given, collects a line per try for a
    verbose log.
    """
    fetch = fetch_nse_bhavcopy if str(exchange).upper() == "NSE" else fetch_bse_bhavcopy
    label = f"{str(exchange).upper()} bhavcopy"

    # Never ask for a date that has not happened. Besides being pointless, BSE
    # answers a future date with an HTML error page at status 200 -- which the
    # column contract now rejects, but not requesting it at all is cheaper and
    # keeps the attempt log honest.
    today = date.today()
    day = min(on, today) if on is not None else today
    if on is not None and on > today and attempts is not None:
        attempts.append(f"{on:%d %b %Y} is in the future; starting from {today:%d %b %Y}")

    tried: list[_Attempt] = []

    for _ in range(max(int(look_back), 1) + 1):
        if day.weekday() >= 5:  # Saturday / Sunday
            if attempts is not None:
                attempts.append(f"{day:%d %b %Y}: weekend, not requested")
            day -= timedelta(days=1)
            continue

        result = fetch(day, timeout=timeout)
        if result.ok:
            if attempts is not None:
                attempts.append(f"{day:%d %b %Y}: {len(result.rows):,} rows")
            return result

        tried.append(_Attempt(day, result.notice))
        if attempts is not None:
            attempts.append(f"{day:%d %b %Y}: {result.notice}")
        day -= timedelta(days=1)

    first = tried[0].on if tried else (on or date.today())
    last = tried[-1].on if tried else first
    return SourceFile(
        exchange=str(exchange).upper(),
        source=label,
        notice=(
            f"no bhavcopy was published between {last:%d %b %Y} and "
            f"{first:%d %b %Y} ({len(tried)} trading-day candidate(s) tried). "
            "Either the exchange has not posted it yet or the files are "
            "unreachable from here."
        ),
    )
