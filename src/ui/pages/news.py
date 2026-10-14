"""News — market headlines, what they mention of yours, and the week's calendar.

Like the Watchlist, this page does **not** require a completed analysis:
reading the news is independent of whether you have run the portfolio.

Three things it is careful about, because each is a way a news page can
mislead:

* **A stale feed is not news.** MoneyControl's RSS endpoints still answer with
  valid XML frozen at April 2024. They are requested, measured, and rejected
  with their age shown — never rendered as today's headlines.
* **Tone is a reading of the words, not a forecast.** Every classification
  shows the terms that produced it, so a wrong one is visibly wrong rather
  than authoritative.
* **The calendar covers one week, because that is all that is published.**
  The window it actually covers is printed rather than implied.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import news as na
from src.config import DISCLAIMER
from src.data import news as nw
from src.ui import components as ui
from src.ui import formatters as fmt
from src.ui import session
from src.ui import theme as ui_theme

#: Feeds are polled no more often than this. News moves in minutes, not
#: seconds, and re-fetching five publishers on every widget interaction would
#: be rude to them and slow for the reader.
_FEED_TTL = 600

_TONE_ICON = {"positive": "▲", "negative": "▼", "mixed": "◆", "neutral": "·"}


@st.cache_data(ttl=_FEED_TTL, show_spinner=False)
def _load_feeds(keys: tuple[str, ...]) -> list[nw.FeedResult]:
    return nw.fetch_feeds(keys)


@st.cache_data(ttl=_FEED_TTL, show_spinner=False)
def _load_calendar() -> nw.CalendarResult:
    return nw.fetch_calendar()


@st.cache_data(ttl=_FEED_TTL, show_spinner=False)
def _load_ticker_news(ticker: str) -> tuple[tuple[nw.NewsItem, ...], str]:
    return nw.fetch_ticker_news(ticker)


def _tracked_records() -> list:
    """Universe records for everything the user holds or watches.

    The set headlines are matched against. Deliberately small: "which of my
    positions is in the news" is the question worth answering, and scanning
    7,500 company names produces false positives faster than matches.
    """
    from src import stock_universe as universe

    tickers: list[str] = []
    holdings = session.get_holdings()
    if isinstance(holdings, pd.DataFrame) and "ticker" in holdings.columns:
        tickers += [str(t) for t in holdings["ticker"]]
    tickers += [item.ticker for item in session.get_watchlist()]

    records = []
    for ticker in dict.fromkeys(t for t in tickers if t):
        record = universe.by_yahoo_symbol(ticker)
        if record is not None:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _feed_health(results: list[nw.FeedResult]) -> None:
    """Which feeds answered, and which are abandoned. Stated, not hidden."""
    live = [r for r in results if r.ok]
    stale = [r for r in results if r.stale]
    broken = [r for r in results if not r.ok and not r.stale]

    badges = [(f"{r.label}: {len(r.items)}", "gain") for r in live]
    badges += [(f"{r.label}: stale", "warn") for r in stale]
    badges += [(f"{r.label}: unavailable", "loss") for r in broken]
    ui.badges(badges)

    if stale:
        st.warning(
            "**These feeds still respond but have stopped publishing**, so their "
            "items are withheld rather than shown as current news:\n\n"
            + "\n".join(
                f"- **{r.label}** — newest item {r.newest:%d %b %Y}, {r.age_days:.0f} days old"
                for r in stale
            )
            + f"\n\nA feed is rejected past {nw.MAX_FEED_AGE_DAYS} days. They will "
            "reappear here automatically if the publisher resumes."
        )
    for result in broken:
        st.caption(f"⚠ {result.label}: {result.notice}")


def _headlines(items: list, records: list) -> None:
    ui_theme.section("MARKET HEADLINES", "\U0001f4f0")
    if not items:
        ui.empty_state(
            "No current headlines",
            "Every configured feed is either unreachable or has stopped "
            "publishing. Nothing is shown rather than showing old news.",
            icon="📰",
        )
        return

    stats = na.summarise(items)
    ui.kpi_row(
        [
            {
                "label": "Headlines",
                "value": f"{stats.n_items:,}",
                "footnote": f"newest {stats.newest:%d %b %H:%M}" if stats.newest else "",
            },
            {"label": "Positive tone", "value": f"{stats.n_positive:,}", "tone": "gain"},
            {"label": "Negative tone", "value": f"{stats.n_negative:,}", "tone": "loss"},
            {
                "label": "Unclassified",
                "value": f"{stats.n_neutral:,}",
                "tone": "neutral",
                "footnote": f"{fmt.fmt_pct((1 - stats.share_classified) * 100, 0)} of the feed",
            },
        ],
        columns=4,
    )
    st.caption(
        "Tone is a reading of the **words in the headline** by a small "
        "documented lexicon — not a forecast, and not a view on whether the "
        "market had already priced the story in. Hover the *why* column to see "
        "which terms produced each label."
    )

    frame = na.items_frame(items)
    frame["tone"] = frame["tone"].map(lambda label: f"{_TONE_ICON.get(label, '')} {label}")
    ui.dataframe(
        frame.rename(
            columns={
                "published": "Published",
                "title": "Headline",
                "source": "Source",
                "tone": "Tone",
                "why": "Why",
                "link": "Link",
            }
        ),
        height=520,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")},
    )


def _your_positions(items: list, records: list) -> None:
    ui_theme.section("YOUR POSITIONS IN THE NEWS", "\U0001f3af")
    if not records:
        st.info(
            "Add holdings or watchlist entries and any headline mentioning them "
            "is picked out here. Matching is scoped to what you track on "
            "purpose — scanning every listed company against every headline "
            "produces false positives faster than matches."
        )
        return

    mentions = na.match_instruments(items, records)
    if not mentions:
        st.caption(
            f"None of your {len(records)} tracked instrument(s) appears in the "
            f"{len(items)} current headlines. That is the common case on any given day."
        )
        return

    frame = na.mentions_frame(mentions)
    frame["tone"] = frame["tone"].map(lambda label: f"{_TONE_ICON.get(label, '')} {label}")
    ui.dataframe(
        frame.rename(
            columns={
                "ticker": "Instrument",
                "name": "Name",
                "published": "Published",
                "title": "Headline",
                "matched_on": "Matched on",
                "tone": "Tone",
                "why": "Why",
                "link": "Link",
            }
        ),
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")},
    )
    st.caption(
        "**Matched on** names the term that produced the match, so a wrong tag "
        "is visibly wrong. A match found only in the article body — not the "
        "headline you can see — is labelled *(in body)*. A shared group name "
        "like “Tata” is never used alone when you track more than one company "
        "from that group, because it identifies neither."
    )


def _per_ticker(records: list) -> None:
    with st.expander("Headlines for one instrument", expanded=False):
        if not records:
            st.caption("Add a holding or a watchlist entry first.")
            return
        symbols = [r.yahoo_symbol for r in records]
        chosen = st.selectbox("Instrument", symbols, key="news_ticker_pick")
        items, notice = _load_ticker_news(chosen)
        if notice:
            st.caption(f"⚠ {notice}")
        if not items:
            return
        frame = na.items_frame(items)
        frame["tone"] = frame["tone"].map(lambda label: f"{_TONE_ICON.get(label, '')} {label}")
        ui.dataframe(
            frame.rename(
                columns={
                    "published": "Published",
                    "title": "Headline",
                    "source": "Source",
                    "tone": "Tone",
                    "why": "Why",
                    "link": "Link",
                }
            ),
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")},
        )


def _calendar() -> None:
    ui_theme.section("ECONOMIC CALENDAR", "\U0001f5d3")
    result = _load_calendar()
    if not result.ok:
        ui.empty_state(
            "No economic calendar",
            result.notice or "The calendar could not be fetched.",
            icon="🗓",
        )
        return

    upcoming = result.upcoming()
    high = [e for e in upcoming if e.is_high_impact]

    st.caption(
        f"Covers **{result.window}** — {len(result.events)} events, "
        f"{len(upcoming)} still ahead. Only the current week is published by the "
        "source, so this is not a month's visibility."
    )

    if high:
        st.warning(
            "**High-impact releases still to come this week** — the days a "
            "broad move is most likely:\n\n" + "\n".join(f"- {event.label}" for event in high[:10])
        )
    elif upcoming:
        st.info("No high-impact releases left this week; the remaining entries are medium or low.")

    tabs = st.tabs(["Still ahead", "Whole week by impact"])
    with tabs[0]:
        if upcoming:
            ui.dataframe(
                na.calendar_frame(upcoming).rename(
                    columns={
                        "when": "When",
                        "country": "Currency",
                        "impact": "Impact",
                        "event": "Event",
                        "forecast": "Forecast",
                        "previous": "Previous",
                    }
                )
            )
        else:
            st.caption("Nothing left in the published window.")
    with tabs[1]:
        for impact, events in na.group_by_impact(result.events).items():
            st.markdown(f"**{impact}** — {len(events)} event(s)")
            ui.dataframe(
                na.calendar_frame(events).rename(
                    columns={
                        "when": "When",
                        "country": "Currency",
                        "impact": "Impact",
                        "event": "Event",
                        "forecast": "Forecast",
                        "previous": "Previous",
                    }
                )
            )

    st.caption(
        "`Currency` is the currency the release affects, which is how the "
        "source labels country. A high-impact rating means the *release* "
        "historically moves markets — not that it will move **your** holdings, "
        "and not in a stated direction."
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "News",
        "Market headlines, what they mention of yours, and the week's scheduled releases",
        icon="📰",
    )

    with st.sidebar:
        st.caption("News sources")
        chosen = st.multiselect(
            "Feeds",
            options=list(nw.FEEDS),
            default=list(nw.FEEDS),
            format_func=lambda key: nw.FEEDS[key][0],
            key="news_feed_pick",
        )

    with ui.loading("Fetching feeds…"):
        results = _load_feeds(tuple(chosen))

    _feed_health(results)

    items, removed = na.dedupe([item for result in results for item in result.items])
    if removed:
        st.caption(
            f"{removed} duplicate headline(s) removed — publishers syndicating one "
            "agency story is several items and one piece of news."
        )

    records = _tracked_records()

    st.divider()
    _your_positions(items, records)
    _per_ticker(records)

    st.divider()
    _headlines(items, records)

    st.divider()
    _calendar()

    ui.methodology(
        "What this page can and cannot tell you",
        """
**A feed that answers is not a feed that works.** Every source is checked for
freshness and one whose newest item is more than a week old is withheld with
its age shown. MoneyControl's RSS endpoints are the live example: they return
valid XML frozen at April 2024, and a page that trusted an HTTP 200 would
present two-year-old headlines as today's market news.

**Tone reads the text, and nothing else.** A small hand-built lexicon of
directional finance terms, with the matched words always displayed. It is not
Loughran-McDonald and does not pretend to be. It has no idea what was already
priced in — "profit falls 40%" is negative *text* and the stock can rise on it
if the market feared worse. Around a third of a market feed comes back
unclassified, which is the honest answer for "Sensex ends flat".

**Matching is scoped to what you track**, using tickers, curated aliases and
the distinctive words of a company name with generic ones ("India",
"Industries", "Bank") removed. A group prefix shared by two of your holdings
is never used on its own, because "Tata" identifies neither Tata Power nor
TCS.

**No impact is predicted.** The calendar tells you which releases have
historically moved markets and when they land; it does not say what will
happen or in which direction. Measuring what news *did* to a price is an event
study on realised returns around past announcements — a separate calculation,
and not one this page performs.
""",
    )

    newest = max(
        (r.newest for r in results if r.newest is not None),
        default=None,
    )
    ui.freshness_bar(
        newest,
        "RSS feeds + ForexFactory + Yahoo Finance",
        extra=[
            f"{len(items)} headline(s) from {sum(1 for r in results if r.ok)} live feed(s)",
            f"cached for {_FEED_TTL // 60} minutes",
        ],
    )
    st.caption(f"⚠️ {DISCLAIMER} Headlines are third-party content, reproduced as published.")
