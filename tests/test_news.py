"""Tests for the news pipeline.

No socket is opened: ``urllib.request.urlopen`` is replaced with a fake that
serves canned bytes.

The property this file is mostly about is **freshness**. MoneyControl's RSS
endpoints answer HTTP 200 with valid XML frozen at April 2024, so "the request
succeeded" is not evidence that the content is news. A page that trusted the
status code would present two-year-old headlines as today's market news, and
a reader would act on them.

The rest is precision: a tone label nobody can interrogate is a random number
in a confident font, and an instrument tag on a headline that visibly does not
mention the company reads as a bug.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta
from typing import ClassVar

import pytest

from src.analytics import news as na
from src.data import news as nw

NOW = datetime(2026, 9, 3, 12, 0)


def rss(items: list[tuple[str, str, str]], title: str = "Feed") -> bytes:
    """An RSS document from ``(title, pubDate, link)`` triples."""
    body = "".join(
        f"<item><title>{t}</title><pubDate>{d}</pubDate><link>{link}</link>"
        f"<description>&lt;img src='x.jpg'/&gt;Summary of {t}</description></item>"
        for t, d, link in items
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>{title}</title>{body}</channel></rss>"
    ).encode()


FRESH = rss(
    [
        ("TCS beats estimates and upgrades guidance", "Wed, 02 Sep 2026 09:30:00 +0530", "u1"),
        (
            "Reliance Industries profit falls on weak demand",
            "Wed, 02 Sep 2026 08:00:00 +0530",
            "u2",
        ),
        ("Sensex ends flat ahead of policy meeting", "Tue, 01 Sep 2026 16:00:00 +0530", "u3"),
    ]
)

#: The MoneyControl shape: perfectly valid, and two years old.
STALE = rss([("Old market wrap", "Tue, 23 Apr 2024 11:23:30 +0530", "old1")])

CALENDAR = json.dumps(
    [
        {
            "title": "ISM Manufacturing PMI",
            "country": "USD",
            "date": "2026-09-04T14:00:00-04:00",
            "impact": "High",
            "forecast": "51.2",
            "previous": "50.8",
        },
        {
            "title": "RBI Rate Decision",
            "country": "INR",
            "date": "2026-09-02T05:00:00-04:00",
            "impact": "High",
            "forecast": "",
            "previous": "6.50%",
        },
        {
            "title": "Retail Sales y/y",
            "country": "JPY",
            "date": "2026-09-05T19:50:00-04:00",
            "impact": "Low",
            "forecast": "3.2%",
            "previous": "0.5%",
        },
    ]
).encode()


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, amount: int | None = None) -> bytes:
        return self._body if amount is None else self._body[:amount]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Route URL fragments to canned bodies or exceptions."""

    def install(routes: dict[str, object]) -> list[str]:
        asked: list[str] = []

        def fake_open(request, timeout=None):
            url = request if isinstance(request, str) else request.full_url
            asked.append(url)
            for fragment, payload in routes.items():
                if fragment in url:
                    if isinstance(payload, Exception):
                        raise payload
                    return FakeResponse(payload)
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(nw.urllib.request, "urlopen", fake_open)
        return asked

    return install


# ---------------------------------------------------------------------------
# freshness -- the property that shaped the module
# ---------------------------------------------------------------------------
def test_a_fresh_feed_returns_its_items(wire) -> None:
    wire({"economictimes": FRESH})

    result = nw.fetch_feed("et_markets", now=NOW)

    assert result.ok
    assert len(result.items) == 3
    assert not result.stale


def test_a_feed_frozen_two_years_ago_is_rejected(wire) -> None:
    """The MoneyControl case. Valid XML, HTTP 200, and not news."""
    wire({"moneycontrol": STALE})

    result = nw.fetch_feed("moneycontrol_latest", now=NOW)

    assert not result.ok
    assert result.stale
    assert result.items == ()


def test_a_stale_feed_reports_its_age_in_days(wire) -> None:
    """So the reader can tell an abandoned feed from a broken one."""
    wire({"moneycontrol": STALE})

    result = nw.fetch_feed("moneycontrol_latest", now=NOW)

    assert "863 days old" in result.notice
    assert "23 Apr 2024" in result.notice
    assert "abandoned" in result.summary


def test_staleness_is_distinct_from_failure(wire) -> None:
    """Conflating them loses the only diagnosis a reader can act on."""
    wire(
        {"moneycontrol": STALE, "economictimes": urllib.error.HTTPError("u", 500, "err", {}, None)}
    )

    stale = nw.fetch_feed("moneycontrol_latest", now=NOW)
    broken = nw.fetch_feed("et_markets", now=NOW)

    assert stale.stale and not stale.ok
    assert broken.ok is False and broken.stale is False
    assert "HTTP 500" in broken.notice


def test_a_feed_just_inside_the_window_is_kept(wire) -> None:
    recent = (NOW - timedelta(days=nw.MAX_FEED_AGE_DAYS - 1)).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    wire({"economictimes": rss([("Recent story", recent, "u")])})

    assert nw.fetch_feed("et_markets", now=NOW).ok


def test_a_feed_just_outside_the_window_is_rejected(wire) -> None:
    old = (NOW - timedelta(days=nw.MAX_FEED_AGE_DAYS + 1)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    wire({"economictimes": rss([("Older story", old, "u")])})

    assert nw.fetch_feed("et_markets", now=NOW).stale


def test_the_check_can_be_disabled_deliberately(wire) -> None:
    wire({"moneycontrol": STALE})

    assert nw.fetch_feed("moneycontrol_latest", max_age_days=0, now=NOW).ok


def test_a_feed_with_no_dates_says_freshness_was_unverifiable(wire) -> None:
    """Silently trusting an undated feed would defeat the whole guard."""
    wire({"economictimes": b"<rss><channel><item><title>Undated</title></item></channel></rss>"})

    result = nw.fetch_feed("et_markets", now=NOW)

    assert not result.ok
    assert "freshness could not be checked" in result.notice


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_html_is_stripped_from_a_summary(wire) -> None:
    """Feeds embed markup -- MoneyControl puts an <img> before every summary --
    and rendering it raw would put untrusted HTML into the page."""
    wire({"economictimes": FRESH})

    summary = nw.fetch_feed("et_markets", now=NOW).items[0].summary

    assert "<img" not in summary
    assert "Summary of" in summary


def test_a_timezone_is_normalised_to_utc(wire) -> None:
    """Indian feeds stamp +0530. Comparing that naively is five and a half
    hours out, which sorts today's headlines behind yesterday's."""
    wire({"economictimes": FRESH})

    newest = nw.fetch_feed("et_markets", now=NOW).items[0].published

    assert newest == datetime(2026, 9, 2, 4, 0)  # 09:30 IST
    assert newest.tzinfo is None


def test_items_come_back_newest_first(wire) -> None:
    wire({"economictimes": FRESH})

    items = nw.fetch_feed("et_markets", now=NOW).items

    assert [i.published for i in items] == sorted((i.published for i in items), reverse=True)


def test_unreadable_xml_is_reported_not_raised(wire) -> None:
    wire({"economictimes": b"<rss><channel><item>unclosed"})

    result = nw.fetch_feed("et_markets", now=NOW)

    assert not result.ok
    assert "not readable XML" in result.notice


def test_an_unknown_feed_key_is_refused(wire) -> None:
    wire({})

    assert "not a registered feed" in nw.fetch_feed("nope", now=NOW).notice


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("down"), TimeoutError("slow"), OSError("reset")],
)
def test_an_unreachable_feed_is_reported(wire, failure: Exception) -> None:
    wire({"economictimes": failure})

    assert "could not be reached" in nw.fetch_feed("et_markets", now=NOW).notice


def test_one_failing_feed_does_not_stop_the_others(wire) -> None:
    wire({"economictimes": FRESH, "moneycontrol": STALE})

    results = nw.fetch_feeds(["et_markets", "moneycontrol_latest"], now=NOW)

    assert results[0].ok
    assert results[1].stale


def test_moneycontrol_stays_registered(wire) -> None:
    """It is the guard's regression test in production, and it will start
    working again on its own if the publisher resumes."""
    assert any(key.startswith("moneycontrol") for key in nw.FEEDS)


# ---------------------------------------------------------------------------
# the calendar
# ---------------------------------------------------------------------------
def test_the_calendar_parses_with_impact_ratings(wire) -> None:
    wire({"ff_calendar": CALENDAR})

    result = nw.fetch_calendar()

    assert result.ok
    assert len(result.events) == 3
    assert sum(1 for e in result.events if e.is_high_impact) == 2


def test_the_calendar_reports_the_window_it_covers(wire) -> None:
    """Only one week is published, so implying more would overstate it."""
    wire({"ff_calendar": CALENDAR})

    result = nw.fetch_calendar()

    assert "02 Sep" in result.window
    assert "05 Sep 2026" in result.window


def test_calendar_events_are_sorted_by_time(wire) -> None:
    wire({"ff_calendar": CALENDAR})

    stamps = [e.when for e in nw.fetch_calendar().events]

    assert stamps == sorted(stamps)


def test_only_events_still_ahead_are_upcoming(wire) -> None:
    wire({"ff_calendar": CALENDAR})

    upcoming = nw.fetch_calendar().upcoming(now=datetime(2026, 9, 4, 12, 0))

    assert [e.title for e in upcoming] == ["ISM Manufacturing PMI", "Retail Sales y/y"]


def test_an_unreachable_calendar_is_reported(wire) -> None:
    wire({})

    result = nw.fetch_calendar()

    assert not result.ok
    assert result.notice


def test_unreadable_calendar_json_is_reported(wire) -> None:
    wire({"ff_calendar": b"not json"})

    assert "not readable JSON" in nw.fetch_calendar().notice


# ---------------------------------------------------------------------------
# tone
# ---------------------------------------------------------------------------
def test_a_clearly_good_headline_is_positive() -> None:
    result = na.tone("TCS beats estimates and upgrades guidance")

    assert result.label == "positive"
    assert "beats" in result.positive


def test_a_clearly_bad_headline_is_negative() -> None:
    result = na.tone("Profit falls 40% on weak demand")

    assert result.label == "negative"
    assert "falls" in result.negative


def test_a_negated_term_does_not_count() -> None:
    """ "No fall in profit" is not bad news, and a plain word count says it is."""
    result = na.tone("No fall in profit despite headwinds")

    assert result.label == "neutral"
    assert "fall" in result.negated


def test_a_dampened_gain_is_not_good_news() -> None:
    """Found in live output: "India bonds pare gains as oil spike tames FCNR
    boost" was classified positive on "gains" and "boost" while saying the
    opposite."""
    result = na.tone("India bonds pare gains as oil spike tames FCNR boost")

    assert result.label == "neutral"
    assert "gains" in result.negated


def test_non_directional_nouns_do_not_drive_the_label() -> None:
    """An earlier lexicon counted "profit", "record", "high", "order" and
    "buy" as positive, which classified 64% of a live market feed positive --
    and scored "profit falls" as merely mixed, because the noun cancelled the
    verb."""
    assert na.tone("Company reports record profit and a high order book").label == "neutral"
    assert na.tone("Profit falls sharply").label == "negative"


def test_a_headline_with_nothing_matched_is_neutral_not_balanced() -> None:
    """0.0 with no matches is not the same as one positive and one negative
    term cancelling, and the label distinguishes them."""
    nothing = na.tone("Sensex ends flat ahead of policy meeting")
    balanced = na.tone("Shares rise then fall")

    assert nothing.label == "neutral"
    assert nothing.matched == ()
    assert balanced.label == "mixed"
    assert balanced.matched


def test_the_matched_terms_are_always_reported() -> None:
    """A tone label nobody can interrogate is a number in a confident font."""
    result = na.tone("Stock surges after upgrade but warns on margins")

    assert "positive:" in result.explained
    assert "negative:" in result.explained


def test_empty_text_is_neutral() -> None:
    assert na.tone("").label == "neutral"
    assert na.tone(None).label == "neutral"


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------
def test_the_same_link_from_two_feeds_is_one_story() -> None:
    a = nw.NewsItem(title="Agency story", link="https://x.com/a", source="ET")
    b = nw.NewsItem(title="Agency story rewritten", link="https://x.com/a", source="Mint")

    unique, removed = na.dedupe([a, b])

    assert len(unique) == 1
    assert removed == 1
    assert unique[0].source == "ET"  # first wins, callers pass best-first


def test_an_identical_title_without_a_link_is_deduped() -> None:
    items = [nw.NewsItem(title="Same words"), nw.NewsItem(title="  same   WORDS ")]

    unique, removed = na.dedupe(items)

    assert len(unique) == 1
    assert removed == 1


def test_distinct_stories_are_kept() -> None:
    items = [nw.NewsItem(title="A", link="1"), nw.NewsItem(title="B", link="2")]

    assert len(na.dedupe(items)[0]) == 2


# ---------------------------------------------------------------------------
# instrument matching
# ---------------------------------------------------------------------------
class Record:
    """The subset of a universe record that matching reads."""

    def __init__(self, symbol: str, yahoo: str, name: str, aliases=(), prominence=0.5) -> None:
        self.symbol = symbol
        self.yahoo_symbol = yahoo
        self.name = name
        self.aliases = tuple(aliases)
        self.prominence = prominence


RELIANCE = Record(
    "RELIANCE", "RELIANCE.NS", "Reliance Industries Limited", ("RIL", "Reliance"), 1.0
)
TCS = Record("TCS", "TCS.NS", "Tata Consultancy Services Limited", ("Tata Consultancy",), 0.9)
TATAPOWER = Record("TATAPOWER", "TATAPOWER.NS", "Tata Power Company Limited", (), 0.8)
INFY = Record("INFY", "INFY.NS", "Infosys Limited", ("Infosys",), 0.95)


def test_a_ticker_in_a_headline_is_matched() -> None:
    item = nw.NewsItem(title="TCS wins a large order")

    mentions = na.match_instruments([item], [TCS, INFY])

    assert [m.ticker for m in mentions] == ["TCS.NS"]


def test_a_curated_alias_is_matched() -> None:
    item = nw.NewsItem(title="RIL to invest in new energy")

    assert na.match_instruments([item], [RELIANCE])[0].term == "ril"


def test_a_group_prefix_alone_never_identifies_a_company() -> None:
    """The bug this rule exists for. "Tata Consultancy Services" and "Tata
    Power Company" both reduce to a leading word of "tata", so matching on it
    tagged every Tata headline to both -- and to Tata Steel and Tata Motors
    besides."""
    item = nw.NewsItem(title="Tata group to expand its retail footprint")

    mentions = na.match_instruments([item], [TCS, TATAPOWER])

    assert mentions == []


def test_a_leading_word_is_used_when_no_other_record_shares_it() -> None:
    """ "tata" identifies nothing when two Tata companies are tracked;
    "infosys" identifies Infosys."""
    item = nw.NewsItem(title="Infosys raises its full-year guidance")

    mentions = na.match_instruments([item], [INFY, TCS, TATAPOWER])

    assert [m.ticker for m in mentions] == ["INFY.NS"]


def test_the_two_word_form_still_matches_a_group_company() -> None:
    item = nw.NewsItem(title="Tata Power commissions a new solar plant")

    mentions = na.match_instruments([item], [TCS, TATAPOWER])

    assert [m.ticker for m in mentions] == ["TATAPOWER.NS"]


def test_generic_name_words_never_match() -> None:
    """Without this filter a company named "... India Industries" matches
    every headline containing "India"."""
    generic = Record("XYZ", "XYZ.NS", "India Industries Limited")
    item = nw.NewsItem(title="India industries brace for a slowdown")

    assert na.match_instruments([item], [generic]) == []


def test_a_match_found_only_in_the_body_is_labelled() -> None:
    """Otherwise a tag appears beside a headline that visibly does not mention
    the company, which reads as a bug."""
    item = nw.NewsItem(title="Market wrap for Wednesday", summary="Infosys led the gainers")

    mentions = na.match_instruments([item], [INFY])

    assert "in body" in mentions[0].matched_on


def test_a_headline_match_is_not_labelled_as_body() -> None:
    item = nw.NewsItem(title="Infosys led the gainers", summary="Infosys again")

    assert "in body" not in na.match_instruments([item], [INFY])[0].matched_on


def test_a_word_boundary_is_required() -> None:
    """ "TCS" must not match inside another word."""
    item = nw.NewsItem(title="WSTCSTPAPR announces results")

    assert na.match_instruments([item], [TCS]) == []


def test_matching_nothing_returns_nothing() -> None:
    assert na.match_instruments([], [RELIANCE]) == []
    assert na.match_instruments([nw.NewsItem(title="x")], []) == []


def test_the_matches_per_headline_are_capped() -> None:
    item = nw.NewsItem(title="Reliance, Infosys and Tata Power all gain")

    mentions = na.match_instruments([item], [RELIANCE, INFY, TATAPOWER], limit_per_item=2)

    assert len(mentions) == 2


# ---------------------------------------------------------------------------
# frames and summaries
# ---------------------------------------------------------------------------
def test_the_items_frame_has_the_documented_columns() -> None:
    frame = na.items_frame([nw.NewsItem(title="Stock surges", published=NOW)])

    assert list(frame.columns) == ["published", "title", "source", "tone", "why", "link"]
    assert frame["tone"].iloc[0] == "positive"


def test_an_empty_items_frame_is_still_typed() -> None:
    assert list(na.items_frame([]).columns) == [
        "published",
        "title",
        "source",
        "tone",
        "why",
        "link",
    ]


def test_the_mentions_frame_names_what_matched() -> None:
    item = nw.NewsItem(title="Infosys upgrades guidance", published=NOW)
    frame = na.mentions_frame(na.match_instruments([item], [INFY]))

    assert frame["ticker"].iloc[0] == "INFY.NS"
    assert "infosys" in frame["matched_on"].iloc[0]


def test_the_calendar_frame_is_sorted_by_time(wire) -> None:
    wire({"ff_calendar": CALENDAR})
    frame = na.calendar_frame(nw.fetch_calendar().events)

    assert list(frame["when"]) == sorted(frame["when"])


def test_events_group_with_high_impact_first(wire) -> None:
    wire({"ff_calendar": CALENDAR})

    groups = na.group_by_impact(nw.fetch_calendar().events)

    assert list(groups) == ["High", "Low"]
    assert len(groups["High"]) == 2


def test_the_summary_separates_unclassified_from_mixed() -> None:
    """Collapsing them would overstate how much of the feed was actually
    read."""
    items = [
        nw.NewsItem(title="Stock surges on upgrade"),
        nw.NewsItem(title="Shares rise then fall"),
        nw.NewsItem(title="Sensex ends flat"),
    ]

    stats = na.summarise(items)

    assert stats.n_positive == 1
    assert stats.n_mixed == 1
    assert stats.n_neutral == 1
    assert stats.share_classified == pytest.approx(2 / 3)


def test_the_summary_of_nothing_does_not_divide_by_zero() -> None:
    stats = na.summarise([])

    assert stats.n_items == 0
    assert stats.newest is None


# ---------------------------------------------------------------------------
# per-ticker headlines
# ---------------------------------------------------------------------------
class FakeNewsTicker:
    """Minimal ``yfinance.Ticker`` exposing only ``.news``."""

    payload: ClassVar[list] = []
    explode: ClassVar[bool] = False

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def news(self) -> list:
        if type(self).explode:
            raise RuntimeError("provider refused the news request")
        return type(self).payload


@pytest.fixture
def fake_ticker_news(monkeypatch: pytest.MonkeyPatch):
    import yfinance

    FakeNewsTicker.explode = False
    FakeNewsTicker.payload = [
        {
            "id": "1",
            "content": {
                "title": "Reliance wins a large order",
                "pubDate": "2026-09-02T09:30:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.test/a"},
                "summary": "Order book expands",
            },
        },
        {
            "id": "2",
            "content": {
                "title": "Older story",
                "pubDate": "2026-08-20T05:00:00Z",
                "provider": {"displayName": "PTI"},
                "canonicalUrl": {"url": "https://example.test/b"},
            },
        },
    ]
    monkeypatch.setattr(yfinance, "Ticker", FakeNewsTicker)
    return FakeNewsTicker


def test_ticker_headlines_are_normalised(fake_ticker_news) -> None:
    items, notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert notice == ""
    assert items[0].title == "Reliance wins a large order"
    assert items[0].source == "Reuters"
    assert items[0].link == "https://example.test/a"
    assert items[0].published == datetime(2026, 9, 2, 9, 30)


def test_ticker_headlines_come_back_newest_first(fake_ticker_news) -> None:
    items, _notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert [i.title for i in items] == ["Reliance wins a large order", "Older story"]


def test_a_unix_timestamp_is_accepted(fake_ticker_news) -> None:
    """The provider has used both an ISO string and an epoch for this field."""
    fake_ticker_news.payload = [
        {"content": {"title": "Epoch story", "providerPublishTime": 1788000000}}
    ]

    items, _notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert items[0].published is not None
    assert items[0].published.tzinfo is None


def test_a_flat_payload_without_content_still_parses(fake_ticker_news) -> None:
    """The response shape has changed at least once, so every field is read
    defensively rather than assumed."""
    fake_ticker_news.payload = [{"title": "Flat shape", "link": "https://example.test/c"}]

    items, _notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert items[0].title == "Flat shape"
    assert items[0].source == "Yahoo Finance"


def test_an_item_without_a_title_is_skipped(fake_ticker_news) -> None:
    fake_ticker_news.payload = [{"content": {"title": "", "pubDate": "2026-09-01T00:00:00Z"}}]

    items, notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert items == ()
    assert "published no headlines" in notice


def test_no_headlines_is_reported(fake_ticker_news) -> None:
    fake_ticker_news.payload = []

    items, notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert items == ()
    assert "published no headlines" in notice


def test_an_exploding_provider_is_reported_not_raised(fake_ticker_news) -> None:
    """A news failure must not break the page it sits on."""
    fake_ticker_news.explode = True

    items, notice = nw.fetch_ticker_news("RELIANCE.NS")

    assert items == ()
    assert "could not be fetched" in notice


def test_a_blank_ticker_is_refused(fake_ticker_news) -> None:
    items, notice = nw.fetch_ticker_news("   ")

    assert items == ()
    assert "No ticker" in notice


def test_the_limit_is_honoured(fake_ticker_news) -> None:
    items, _notice = nw.fetch_ticker_news("RELIANCE.NS", limit=1)

    assert len(items) == 1


# ---------------------------------------------------------------------------
# company news -- Google News RSS
# ---------------------------------------------------------------------------
def google_rss(items: list[tuple[str, str, str, str]]) -> bytes:
    """A Google News shaped RSS document from ``(title, pubDate, link, source)``.

    ``source`` of "" omits the ``<source>`` element, which is the shape that
    forces the publisher to be split off the title instead.
    """
    body = ""
    for title, stamp, link, source in items:
        element = f'<source url="https://example.test">{source}</source>' if source else ""
        body += (
            f"<item><title>{title}</title><link>{link}</link><pubDate>{stamp}</pubDate>"
            f"{element}<description>&lt;b&gt;{title}&lt;/b&gt;</description></item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>Company - Google News</title>{body}</channel></rss>"
    ).encode()


GOOGLE = google_rss(
    [
        (
            "Polycab India Q2 profit rises 40% - The Economic Times",
            "Wed, 02 Sep 2026 09:30:00 GMT",
            "https://news.google.com/rss/articles/aaa",
            "The Economic Times",
        ),
        (
            "Cables demand lifts Polycab order book - Moneycontrol.com",
            "Tue, 01 Sep 2026 05:00:00 GMT",
            "https://news.google.com/rss/articles/bbb",
            "",
        ),
    ]
)


def test_a_multi_word_company_name_is_quoted() -> None:
    """Unquoted, "Polycab India" is two loose terms and matches any other
    company sharing a word with it."""
    assert nw.company_query("Polycab India") == '"Polycab India"'


def test_a_single_word_company_name_is_left_bare() -> None:
    assert nw.company_query("Infosys") == "Infosys"


def test_the_symbol_is_a_fallback_when_no_name_is_given() -> None:
    """A caller holding only a ticker still gets a usable query, and appending
    the ticker to a real name would only add ticker-noise headlines."""
    assert nw.company_query("", "POLYCAB.NS") == "POLYCAB.NS"
    assert nw.company_query("   ", "INFY") == "INFY"
    assert nw.company_query("", "") == ""


def test_the_publisher_element_supplies_the_source() -> None:
    """Google appends the outlet to the title; leaving it there would repeat
    it beside a source line that already names it."""
    items = nw._parse_company_feed(GOOGLE)

    assert items[0].title == "Polycab India Q2 profit rises 40%"
    assert items[0].source == "The Economic Times"


def test_the_publisher_is_split_off_the_title_when_the_element_is_absent() -> None:
    items = nw._parse_company_feed(GOOGLE)

    assert items[1].title == "Cables demand lifts Polycab order book"
    assert items[1].source == "Moneycontrol.com"


def test_a_company_item_is_dated_and_scoped_to_india() -> None:
    items = nw._parse_company_feed(GOOGLE)

    assert items[0].published == datetime(2026, 9, 2, 9, 30)
    assert items[0].published.tzinfo is None
    assert items[0].region == "IN"


def test_company_items_come_back_newest_first() -> None:
    items = nw._parse_company_feed(GOOGLE)

    assert [i.published for i in items] == sorted((i.published for i in items), reverse=True)


def test_company_markup_is_stripped_from_the_summary() -> None:
    items = nw._parse_company_feed(GOOGLE)

    assert "<b>" not in items[0].summary
    assert "Polycab India Q2 profit rises 40%" in items[0].summary


def test_the_company_limit_is_honoured() -> None:
    assert len(nw._parse_company_feed(GOOGLE, limit=1)) == 1


def test_unreadable_company_xml_returns_nothing_rather_than_raising() -> None:
    """A malformed response is a "no news found" state, not a page crash."""
    assert nw._parse_company_feed(b"<rss><channel><item>unclosed") == ()
    assert nw._parse_company_feed(b"") == ()
    assert nw._parse_company_feed(b"not xml at all") == ()


def test_company_news_returns_nothing_when_the_provider_is_unreachable(wire) -> None:
    """The hard requirement: the page must not break when Google is down."""
    wire({"news.google": urllib.error.URLError("down")})

    assert nw.fetch_company_news("Polycab India") == ()


def test_company_news_returns_nothing_on_an_http_error(wire) -> None:
    wire({"news.google": urllib.error.HTTPError("u", 503, "err", {}, None)})

    assert nw.fetch_company_news("Polycab India") == ()


def test_company_news_returns_nothing_when_no_query_is_given(wire) -> None:
    asked = wire({"news.google": GOOGLE})

    assert nw.fetch_company_news("   ") == ()
    assert asked == []  # and no request was made


def test_company_news_parses_a_wire_response(wire) -> None:
    wire({"news.google": GOOGLE})

    items = nw.fetch_company_news("Polycab India")

    assert len(items) == 2
    assert items[0].title == "Polycab India Q2 profit rises 40%"


@pytest.mark.network
@pytest.mark.slow
def test_the_live_company_search_still_answers() -> None:
    """Deselect with ``-m "not network"``; CI does.

    The offline tests prove the parse; only this proves Google still serves the
    RSS search the parser is written against. It SKIPS rather than fails when
    the provider is unreachable, so an outage or a blocked network is never
    reported as a defect in this repository.
    """
    items = nw.fetch_company_news(nw.company_query("Polycab India"), limit=10)

    if not items:
        pytest.skip("Google News unreachable or returned nothing (offline or throttled)")

    assert all(i.region == "IN" for i in items)
    assert all(i.title for i in items)
    dated = [i for i in items if i.published is not None]
    assert [i.published for i in dated] == sorted((i.published for i in dated), reverse=True)
