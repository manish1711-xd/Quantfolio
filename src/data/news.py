"""News feeds and the economic calendar — with a staleness guard that matters.

The finding that shaped this module
-----------------------------------
MoneyControl's RSS endpoints **still answer HTTP 200 with valid XML, and every
one of them is frozen**. Measured on 3 September 2026:

===============================  ==================  ==========
feed                             newest item         age
===============================  ==================  ==========
``rss/latestnews.xml``           23 Apr 2024         863 days
``rss/business.xml``             23 Apr 2024         862 days
``rss/results.xml``              23 Apr 2024         863 days
``rss/marketreports.xml``        23 Apr 2024         863 days
``rss/MCtopnews.xml``            2016                3620 days
===============================  ==================  ==========

A naive implementation would have shipped a "Latest news" page confidently
presenting April-2024 headlines as today's market news. That is worse than
having no news page at all: a reader would act on it.

So **freshness is checked, not assumed**. Every fetch measures the age of the
newest item and a feed past :data:`MAX_FEED_AGE_DAYS` is *rejected* with its
age stated. MoneyControl stays in the registry precisely so that guard is
exercised — and so the feature lights up again by itself if the publisher ever
restores it. Nothing about this is specific to MoneyControl; any feed can rot,
usually silently.

What is live, and therefore what the page runs on
-------------------------------------------------
Economic Times, Business Standard and LiveMint market feeds (all age 0 days
when measured), Yahoo Finance per-ticker headlines, and the ForexFactory
economic calendar.

Per-company news is the one thing the market feeds above cannot supply -- a
market feed names whichever companies it names -- so it comes from Google News
RSS search, scoped to the company. That source is what the company page
actually runs on: Yahoo Finance's ``.news`` attribute, the earlier per-company
source, now returns an empty list for every ticker. See
:func:`fetch_company_news`.

The calendar's real limitation
------------------------------
``ff_calendar_thisweek.json`` is the only horizon published — ``nextweek``
returns 404. So "important news days coming up" means *the rest of this week*,
not the rest of the month, and :func:`fetch_calendar` reports the window it
actually covers rather than implying more.

Scope
-----
This module fetches and parses wire formats into plain records. It classifies
nothing: matching a headline to an instrument and reading sentiment off it are
:mod:`src.analytics.news`' job, so both are testable offline against fixtures.

Every network call in the project lives in ``src/data/``; nothing outside that
package imports ``urllib`` or any HTTP client. This is the third module in it,
after ``fetcher`` (Yahoo prices) and ``exchange`` (NSE/BSE files), and the
rule is the package boundary rather than a module count.
"""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

__all__ = [
    "USER_AGENT",
    "DEFAULT_TIMEOUT",
    "MAX_DOWNLOAD_BYTES",
    "MAX_FEED_AGE_DAYS",
    "IMPACT_ORDER",
    "FEEDS",
    "CALENDAR_URL",
    "GOOGLE_NEWS_URL",
    "NewsItem",
    "FeedResult",
    "CalendarEvent",
    "CalendarResult",
    "fetch_feed",
    "fetch_feeds",
    "fetch_ticker_news",
    "fetch_company_news",
    "company_query",
    "fetch_calendar",
]

#: NSE's CDN and several publishers refuse the default ``Python-urllib``
#: string. These are public RSS feeds a browser reads freely.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT: float = 25.0

#: An RSS feed is tens of kilobytes. Past this something is wrong upstream.
MAX_DOWNLOAD_BYTES: int = 8 * 1024 * 1024

#: A feed whose newest item is older than this is treated as **abandoned** and
#: its items are not returned. Seven days is loose enough for a weekly
#: publication and a long holiday, and tight enough to catch a feed that has
#: silently stopped updating -- which is the failure this whole module is
#: arranged around.
MAX_FEED_AGE_DAYS: int = 7

#: Calendar impact, strongest first. ForexFactory's own vocabulary.
IMPACT_ORDER: tuple[str, ...] = ("High", "Medium", "Low", "Holiday")

#: ``key -> (label, url, region)``. Ordered most useful first.
#:
#: MoneyControl is listed on purpose despite being frozen since April 2024:
#: the user asked for it, the staleness guard is what makes including it safe,
#: and it will start working again on its own if the publisher resumes. Its
#: entry is the regression test for the guard, in production.
FEEDS: dict[str, tuple[str, str, str]] = {
    "et_markets": (
        "Economic Times — Markets",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "India",
    ),
    "bs_markets": (
        "Business Standard — Markets",
        "https://www.business-standard.com/rss/markets-106.rss",
        "India",
    ),
    "mint_markets": (
        "LiveMint — Markets",
        "https://www.livemint.com/rss/markets",
        "India",
    ),
    "moneycontrol_latest": (
        "MoneyControl — Latest",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "India",
    ),
    "moneycontrol_results": (
        "MoneyControl — Results",
        "https://www.moneycontrol.com/rss/results.xml",
        "India",
    ),
}

#: ForexFactory's own JSON feed, the one their public widget reads. Only the
#: current week is published; ``ff_calendar_nextweek.json`` is a 404.
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

#: Google News' RSS search endpoint. Query parameters are appended per request
#: rather than here, because the query is the whole point of the call.
#:
#: This is the per-company source the app actually runs on: Yahoo Finance's
#: ``.news`` attribute -- the previous one -- now returns an empty list for
#: every ticker, so a company page built on it silently showed nothing. Google
#: News answers with real dated items from Indian outlets and needs no key,
#: which is what keeps it inside the app's public-feed contract.
GOOGLE_NEWS_URL = "https://news.google.com/rss/search"

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsItem:
    """One headline, with enough provenance to judge it.

    ``published`` is tz-naive UTC, matching every other timestamp in the
    project. ``None`` means the feed gave no parseable date, which is reported
    rather than replaced with "now" -- a headline stamped with the fetch time
    would sort to the top and read as breaking news.
    """

    title: str
    link: str = ""
    published: datetime | None = None
    source: str = ""
    summary: str = ""
    region: str = ""

    @property
    def age_days(self) -> float:
        if self.published is None:
            return float("nan")
        return (datetime.now(timezone.utc).replace(tzinfo=None) - self.published).days

    @property
    def key(self) -> str:
        """Identity for de-duplication: the link, else the normalised title.

        The link first because three publishers syndicating one agency story
        keep distinct titles but often share a canonical URL; the title is the
        fallback for feeds that rewrite links per source.
        """
        return self.link.strip().lower() or _SPACE.sub(" ", self.title.strip().lower())


@dataclass(frozen=True)
class FeedResult:
    """One feed's items, or the reason there are none.

    ``stale`` is distinct from a failure: the feed answered correctly and its
    content is simply too old to present as news. Conflating the two would
    lose the only diagnosis a reader can act on.
    """

    key: str = ""
    label: str = ""
    url: str = ""
    region: str = ""
    items: tuple[NewsItem, ...] = ()
    newest: datetime | None = None
    stale: bool = False
    notice: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.items)

    @property
    def age_days(self) -> float:
        if self.newest is None:
            return float("nan")
        return (datetime.now(timezone.utc).replace(tzinfo=None) - self.newest).days

    @property
    def summary(self) -> str:
        if self.stale:
            return f"{self.label}: abandoned — newest item is {self.age_days:.0f} days old"
        if not self.items:
            return f"{self.label}: {self.notice or 'nothing returned'}"
        return f"{self.label}: {len(self.items)} item(s), newest {self.newest:%d %b %Y}"


@dataclass(frozen=True)
class CalendarEvent:
    """One scheduled economic release."""

    title: str
    country: str = ""
    when: datetime | None = None
    impact: str = ""
    forecast: str = ""
    previous: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def label(self) -> str:
        stamp = f"{self.when:%d %b %H:%M}" if self.when else "time unknown"
        return f"{stamp} · {self.country} · {self.title}"


@dataclass(frozen=True)
class CalendarResult:
    """The calendar window that was actually published."""

    events: tuple[CalendarEvent, ...] = ()
    covers_from: datetime | None = None
    covers_to: datetime | None = None
    notice: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.events)

    @property
    def window(self) -> str:
        if not (self.covers_from and self.covers_to):
            return "no window"
        return f"{self.covers_from:%d %b} to {self.covers_to:%d %b %Y}"

    def upcoming(self, now: datetime | None = None) -> tuple[CalendarEvent, ...]:
        """Events still ahead of ``now``, earliest first."""
        moment = now or datetime.now(timezone.utc).replace(tzinfo=None)
        ahead = [e for e in self.events if e.when is not None and e.when >= moment]
        return tuple(sorted(ahead, key=lambda e: e.when or moment))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def _download(url: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[bytes, str]:
    """``(body, notice)``. Empty body plus a reason on any failure."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/xml,application/json,*/*"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return b"", f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return b"", f"could not be reached ({type(exc).__name__}: {exc})"

    if len(body) > MAX_DOWNLOAD_BYTES:
        return b"", f"the response exceeded {MAX_DOWNLOAD_BYTES // 1024 // 1024} MB"
    if not body:
        return b"", "an empty response"
    return body, ""


def _clean_text(raw: object) -> str:
    """Plain text from an RSS field: tags stripped, entities decoded.

    RSS descriptions routinely carry markup -- MoneyControl embeds an
    ``<img>`` before every summary -- and rendering that raw would put HTML
    into the page from an untrusted feed.
    """
    text = html.unescape(str(raw or ""))
    text = _TAG.sub(" ", text)
    return _SPACE.sub(" ", html.unescape(text)).strip()


def _parse_when(raw: object) -> datetime | None:
    """An RSS or ISO timestamp as tz-naive UTC, or ``None``.

    Normalised to UTC before dropping the zone: Indian feeds stamp
    ``+0530`` and a naive comparison against them would be five and a half
    hours out, which is enough to sort today's headlines behind yesterday's.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    for parse in (parsedate_to_datetime, datetime.fromisoformat):
        try:
            stamp = parse(text)
        except (TypeError, ValueError, IndexError):
            continue
        if stamp.tzinfo is not None:
            return stamp.astimezone(timezone.utc).replace(tzinfo=None)
        return stamp
    return None


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------
def fetch_feed(
    key: str,
    max_age_days: int = MAX_FEED_AGE_DAYS,
    timeout: float = DEFAULT_TIMEOUT,
    now: datetime | None = None,
) -> FeedResult:
    """One registered feed, rejected if its content is too old to be news.

    The staleness check is the point. A feed that answers 200 with valid XML
    from two years ago is the failure mode this module exists to prevent, so
    items are withheld and the age is reported instead. Pass
    ``max_age_days=0`` to disable the check and see whatever is there.
    """
    entry = FEEDS.get(key)
    if entry is None:
        return FeedResult(key=key, notice=f"{key!r} is not a registered feed.")
    label, url, region = entry

    body, notice = _download(url, timeout)
    if notice:
        return FeedResult(key=key, label=label, url=url, region=region, notice=notice)

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return FeedResult(
            key=key,
            label=label,
            url=url,
            region=region,
            notice=f"the feed is not readable XML ({exc})",
        )

    items: list[NewsItem] = []
    for node in root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = _clean_text(
            node.findtext("title") or node.findtext("{http://www.w3.org/2005/Atom}title")
        )
        if not title:
            continue
        items.append(
            NewsItem(
                title=title,
                link=_clean_text(node.findtext("link")),
                published=_parse_when(
                    node.findtext("pubDate")
                    or node.findtext("published")
                    or node.findtext("{http://purl.org/dc/elements/1.1/}date")
                ),
                source=label,
                summary=_clean_text(node.findtext("description")),
                region=region,
            )
        )

    if not items:
        return FeedResult(
            key=key, label=label, url=url, region=region, notice="the feed carried no items"
        )

    dated = [i.published for i in items if i.published is not None]
    newest = max(dated) if dated else None

    if max_age_days > 0 and newest is not None:
        moment = now or datetime.now(timezone.utc).replace(tzinfo=None)
        if moment - newest > timedelta(days=max_age_days):
            age = (moment - newest).days
            return FeedResult(
                key=key,
                label=label,
                url=url,
                region=region,
                newest=newest,
                stale=True,
                notice=(
                    f"the feed still answers but its newest item is {age:,} days old "
                    f"({newest:%d %b %Y}), so it is treated as abandoned rather than "
                    "presented as news."
                ),
            )

    if newest is None:
        return FeedResult(
            key=key,
            label=label,
            url=url,
            region=region,
            notice="no item carried a readable date, so freshness could not be checked",
        )

    items.sort(key=lambda i: i.published or datetime.min, reverse=True)
    return FeedResult(
        key=key, label=label, url=url, region=region, items=tuple(items), newest=newest
    )


def fetch_feeds(
    keys: Iterable[str] | None = None,
    max_age_days: int = MAX_FEED_AGE_DAYS,
    timeout: float = DEFAULT_TIMEOUT,
    now: datetime | None = None,
) -> list[FeedResult]:
    """Every requested feed. One failing feed never stops the others."""
    wanted = list(keys) if keys is not None else list(FEEDS)
    return [fetch_feed(key, max_age_days, timeout, now) for key in wanted]


# ---------------------------------------------------------------------------
# company news -- Google News RSS
# ---------------------------------------------------------------------------
def company_query(name: str, symbol: str = "") -> str:
    """A Google News search string for one company.

    A multi-word name is quoted: without the quotes Google treats "Polycab
    India" as two loose terms and returns any company that happens to share a
    word with it, which is noise dressed as company coverage. The symbol is a
    *fallback* for the caller that holds only a ticker, not something appended
    to the name -- a bare Indian ticker in the query pulls in headlines that
    merely quote the symbol, and the quoted name is the stronger signal.
    """
    label = _clean_text(name)
    ticker = _clean_text(symbol)
    if not label:
        return ticker
    return f'"{label}"' if " " in label else label


def _google_news_url(query: str) -> str:
    """The search URL for ``query``, pinned to the Indian edition.

    ``hl``, ``gl`` and ``ceid`` are what make this the Indian edition; drop them
    and the same query is answered from the US edition, where the mid- and
    small-cap companies this app covers barely appear.
    """
    return f"{GOOGLE_NEWS_URL}?q={urllib.parse.quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"


def _split_publisher(title: str, source: str) -> tuple[str, str]:
    """``(headline, publisher)`` from Google's ``"Headline - Publisher"`` title.

    Google appends the outlet to every title, so leaving it on would render
    "Q2 profit rises 40% - The Economic Times" directly beneath a source line
    that already names The Economic Times. The ``<source>`` element is preferred
    when present; the title's trailing segment is the fallback for the shape
    where it is absent, and either way the suffix comes off the headline shown.
    """
    head, sep, tail = title.rpartition(" - ")
    if sep and head.strip():
        title = head.strip()
        source = source or tail.strip()
    return title, source


def _parse_company_feed(body: bytes, limit: int = 40) -> tuple[NewsItem, ...]:
    """Items from a Google News RSS body, newest first, at most ``limit``.

    Split out from the fetch so the parse can be exercised offline against
    literal XML. Unreadable bytes yield ``()`` rather than an exception: a
    malformed response is a "no news found" state for the caller, not a reason
    to take the page down with it.
    """
    try:
        root = ET.fromstring(body)
    except (ET.ParseError, ValueError):
        return ()

    items: list[NewsItem] = []
    for node in root.findall(".//item"):
        raw_title = _clean_text(node.findtext("title"))
        if not raw_title:
            continue
        title, source = _split_publisher(raw_title, _clean_text(node.findtext("source")))
        if not title:
            continue
        items.append(
            NewsItem(
                title=title,
                link=_clean_text(node.findtext("link")),
                published=_parse_when(node.findtext("pubDate")),
                source=source,
                summary=_clean_text(node.findtext("description")),
                region="IN",
            )
        )

    # Undated items sort last rather than presenting as the newest thing.
    items.sort(key=lambda i: i.published or datetime.min, reverse=True)
    return tuple(items[: max(int(limit), 0)])


def fetch_company_news(query: str, limit: int = 40) -> tuple[NewsItem, ...]:
    """Google News headlines for one company, or ``()``.

    **This never raises.** A company page renders a "no news found" state when
    the tuple is empty, and that page must survive Google being unreachable,
    answering an error, or returning bytes that are not XML -- so every one of
    those collapses to ``()`` here instead of propagating. Callers therefore
    check ``if not items`` and do not guard the call.

    No staleness guard, unlike the registered feeds: a search result is scoped
    to one company and each item carries its own visible date, so an old
    headline is labelled rather than withheld. ``limit`` is a ceiling; fewer
    items come back when Google has fewer.
    """
    text = (query or "").strip()
    if not text:
        return ()

    try:
        body, notice = _download(_google_news_url(text))
        if notice:
            logger.warning("Company news for %r %s", text, notice)
            return ()
        return _parse_company_feed(body, limit)
    except Exception as exc:  # noqa: BLE001 - a news failure must not break a page
        logger.warning("Company news for %r failed: %s", text, exc)
        return ()


# ---------------------------------------------------------------------------
# per-ticker headlines
# ---------------------------------------------------------------------------
def fetch_ticker_news(ticker: str, limit: int = 10) -> tuple[tuple[NewsItem, ...], str]:
    """Yahoo Finance headlines for one symbol. ``(items, notice)``.

    A different shape from the RSS feeds: the provider returns a nested
    ``content`` object whose key names have changed at least once, so every
    field is read defensively and a missing one yields "" rather than raising.
    """
    from src.data.fetcher import _clean_ticker

    symbol = _clean_ticker(ticker)
    if not symbol:
        return (), "No ticker was given."

    try:
        import yfinance as yf

        raw = yf.Ticker(symbol).news or []
    except Exception as exc:  # noqa: BLE001 - a news failure must not break a page
        logger.warning("News for %s failed: %s", symbol, exc)
        return (), f"{symbol}: headlines could not be fetched ({exc})"

    items: list[NewsItem] = []
    for entry in raw:
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        if not isinstance(content, dict):
            continue
        title = _clean_text(content.get("title"))
        if not title:
            continue

        provider = content.get("provider")
        source = (
            _clean_text(provider.get("displayName"))
            if isinstance(provider, dict)
            else _clean_text(provider)
        )
        canonical = content.get("canonicalUrl")
        link = (
            _clean_text(canonical.get("url"))
            if isinstance(canonical, dict)
            else _clean_text(content.get("link"))
        )
        stamp = content.get("pubDate") or content.get("providerPublishTime")
        published: datetime | None
        if isinstance(stamp, (int, float)):
            # The provider has used both an ISO string and a Unix epoch for
            # this field, so both are handled rather than assumed.
            published = datetime.fromtimestamp(float(stamp), tz=timezone.utc).replace(tzinfo=None)
        else:
            published = _parse_when(stamp)

        items.append(
            NewsItem(
                title=title,
                link=link,
                published=published,
                source=source or "Yahoo Finance",
                summary=_clean_text(content.get("summary")),
            )
        )

    if not items:
        return (), f"{symbol}: the provider published no headlines."

    items.sort(key=lambda i: i.published or datetime.min, reverse=True)
    return tuple(items[: max(int(limit), 1)]), ""


# ---------------------------------------------------------------------------
# economic calendar
# ---------------------------------------------------------------------------
def fetch_calendar(timeout: float = DEFAULT_TIMEOUT) -> CalendarResult:
    """The ForexFactory economic calendar for the current week.

    Only one week is published, so the result carries the window it actually
    covers. Presenting it as "upcoming events" without that would imply a
    month of visibility the source does not offer.
    """
    body, notice = _download(CALENDAR_URL, timeout)
    if notice:
        return CalendarResult(notice=f"The economic calendar {notice}.")

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError as exc:
        return CalendarResult(notice=f"The calendar was not readable JSON ({exc}).")

    if not isinstance(payload, list) or not payload:
        return CalendarResult(notice="The calendar held no events.")

    events: list[CalendarEvent] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        title = _clean_text(entry.get("title"))
        if not title:
            continue
        events.append(
            CalendarEvent(
                title=title,
                country=_clean_text(entry.get("country")),
                when=_parse_when(entry.get("date")),
                impact=_clean_text(entry.get("impact")),
                forecast=_clean_text(entry.get("forecast")),
                previous=_clean_text(entry.get("previous")),
            )
        )

    if not events:
        return CalendarResult(notice="No calendar entry carried a usable title.")

    stamps = [e.when for e in events if e.when is not None]
    events.sort(key=lambda e: e.when or datetime.max)
    return CalendarResult(
        events=tuple(events),
        covers_from=min(stamps) if stamps else None,
        covers_to=max(stamps) if stamps else None,
    )
