"""Reading headlines: de-duplication, instrument matching, and tone.

Three jobs, all pure, all offline-testable:

1. **De-duplicate** across feeds. Three publishers syndicating one agency
   story is three items and one piece of news.
2. **Match a headline to instruments you actually hold or watch.** "Which of
   my positions is in the news" is the question worth answering; scanning
   7,500 company names against every headline is both slow and a
   false-positive machine, so matching is deliberately scoped to a supplied
   list of records.
3. **Read the tone of the text** with a small, documented lexicon — and report
   *which words matched*, so the reader can judge the classification instead
   of trusting it.

What tone is, and is not
------------------------
:func:`tone` classifies **the words in a headline**. That is all. It is not a
prediction of the price, it does not know whether the market had already
priced the story in, and a headline is a sub-twenty-word summary written to be
clicked on. "Profit falls 40%" is negative *text*; the stock can rise on it if
the expectation was worse.

So every result carries the matched terms and the app shows them. A tone score
you cannot interrogate is indistinguishable from a random number with a
confident label, and the one thing this module must not do is imply a
forecast. Measuring what a headline *did* to a price is an event study on
realised returns — a separate, honest calculation, and not this function.

The lexicon is hand-built and small
-----------------------------------
:data:`POSITIVE` and :data:`NEGATIVE` are a compact list of directional
finance terms, chosen for precision over recall: it is better to classify
fewer headlines confidently than to guess at all of them. It is **not**
Loughran-McDonald (a licensed research word list of ~4,000 terms), and the
module says so rather than implying academic provenance. Roughly half of a
market feed's headlines come back ``neutral``, which is the honest answer for
"Sensex ends flat" and for anything the lexicon does not cover.

Pure module: stdlib and pandas only, no Streamlit and no network.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

__all__ = [
    "POSITIVE",
    "NEGATIVE",
    "NEGATORS",
    "DAMPENERS",
    "GENERIC_NAME_WORDS",
    "MIN_MATCH_LENGTH",
    "ToneResult",
    "Mention",
    "dedupe",
    "tone",
    "match_instruments",
    "mentions_frame",
    "items_frame",
    "calendar_frame",
    "group_by_impact",
    "FeedSummary",
    "summarise",
]

#: Directional terms that read as good news for the subject of a headline.
#:
#: **Verbs and adjectives only.** An earlier version included the nouns every
#: market headline contains regardless of direction -- "profit", "high",
#: "record", "order", "stake", "contract", "dividend", "growth", "buy" -- and
#: it classified 64% of a live feed positive, including "profit falls 40%"
#: (scored mixed, because the noun cancelled the verb) and "bonds pare gains".
#: Precision over recall: a headline the lexicon cannot read confidently is
#: better left neutral than guessed at.
POSITIVE: frozenset[str] = frozenset(
    {
        "beats",
        "beat",
        "upgrade",
        "upgrades",
        "upgraded",
        "outperform",
        "outperforms",
        "surge",
        "surges",
        "surged",
        "jump",
        "jumps",
        "jumped",
        "rally",
        "rallies",
        "rallied",
        "gain",
        "gains",
        "gained",
        "rise",
        "rises",
        "rose",
        "soar",
        "soars",
        "soared",
        "climb",
        "climbs",
        "climbed",
        "expands",
        "expanded",
        "wins",
        "won",
        "awarded",
        "approved",
        "buyback",
        "acquires",
        "acquired",
        "robust",
        "rebound",
        "rebounds",
        "rebounded",
        "bullish",
        "accumulate",
        "raises",
        "raised",
        "turnaround",
        "boosts",
        "boosted",
        "strengthens",
        "strengthened",
        "advances",
        "advanced",
        "recovers",
        "recovered",
        "doubles",
        "tops",
    }
)

#: Directional terms that read as bad news.
NEGATIVE: frozenset[str] = frozenset(
    {
        "misses",
        "miss",
        "missed",
        "downgrade",
        "downgrades",
        "downgraded",
        "underperform",
        "plunge",
        "plunges",
        "plunged",
        "slump",
        "slumps",
        "slumped",
        "fall",
        "falls",
        "fell",
        "drop",
        "drops",
        "dropped",
        "decline",
        "declines",
        "declined",
        "slide",
        "slides",
        "slid",
        "tumble",
        "tumbles",
        "tumbled",
        "crash",
        "crashes",
        "crashed",
        "loss",
        "losses",
        "lossmaking",
        "weak",
        "weakness",
        "warning",
        "warns",
        "warned",
        "probe",
        "probes",
        "raid",
        "raids",
        "fraud",
        "scam",
        "penalty",
        "fine",
        "fined",
        "default",
        "defaults",
        "downturn",
        "recession",
        "layoff",
        "layoffs",
        "cuts",
        "cut",
        "resigns",
        "resigned",
        "exits",
        "delisting",
        "insolvency",
        "bankruptcy",
        "lawsuit",
        "sues",
        "bearish",
        "sell",
        "reduce",
        "halt",
        "halted",
        "suspended",
        "recall",
        "shortfall",
        "slowdown",
    }
)

#: Words that invert the term that follows. "Profit falls" is negative;
#: "no fall in profit" is not, and a bag-of-words score gets that backwards.
#: Handled with a two-word window rather than real parsing, which is honest
#: about being a heuristic.
NEGATORS: frozenset[str] = frozenset(
    {"no", "not", "never", "without", "avoids", "avoid", "despite"}
)

#: Verbs that REDUCE a following positive: "pare gains", "trim gains", "erase
#: a rally" are not good news. Found in live output -- "India bonds pare gains
#: as oil spike tames FCNR boost" was classified positive on "gains" and
#: "boost" while saying the opposite. Treated as negators of the term they
#: precede rather than as negatives in their own right, because "pare losses"
#: is genuinely good news and a flat negative would invert that too.
DAMPENERS: frozenset[str] = frozenset(
    {
        "pare",
        "pares",
        "pared",
        "paring",
        "trim",
        "trims",
        "trimmed",
        "erase",
        "erases",
        "erased",
        "reverse",
        "reverses",
        "reversed",
        "tames",
        "tamed",
        "limits",
        "limited",
        "caps",
        "capped",
    }
)

#: Name words too generic to identify a company on their own. Matching on any
#: of these would tag every India-related headline to a company whose name
#: happens to contain "India".
GENERIC_NAME_WORDS: frozenset[str] = frozenset(
    {
        "india",
        "indian",
        "bharat",
        "national",
        "state",
        "central",
        "global",
        "international",
        "asia",
        "asian",
        "world",
        "united",
        "general",
        "standard",
        "premier",
        "first",
        "new",
        "modern",
        "supreme",
        "royal",
        "capital",
        "finance",
        "financial",
        "investments",
        "investment",
        "industries",
        "industrial",
        "enterprises",
        "enterprise",
        "trading",
        "traders",
        "services",
        "service",
        "solutions",
        "systems",
        "technologies",
        "technology",
        "tech",
        "power",
        "energy",
        "steel",
        "cement",
        "motors",
        "bank",
        "banks",
        "insurance",
        "life",
        "housing",
        "infra",
        "infrastructure",
        "projects",
        "products",
        "mills",
        "textiles",
        "chemicals",
        "pharma",
        "labs",
        "laboratories",
        "healthcare",
        "hospital",
        "hotels",
        "resorts",
        "foods",
        "agro",
        "sugar",
        "paper",
        "packaging",
        "logistics",
        "transport",
        "telecom",
        "media",
        "entertainment",
        "retail",
        "corporation",
        "company",
        "limited",
        "ltd",
        "group",
        "holdings",
    }
)

#: Shortest token that may identify an instrument. Two-character matches are
#: noise; three is the floor at which a ticker is meaningful ("TCS", "ITC").
MIN_MATCH_LENGTH = 3

_WORD = re.compile(r"[a-z0-9&]+")
_TICKER = re.compile(r"\b[A-Z][A-Z0-9&\-]{2,}\b")


def _words(text: object) -> list[str]:
    return _WORD.findall(str(text or "").lower())


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------
def dedupe(items: Iterable[object]) -> tuple[list[object], int]:
    """``(unique items, how many duplicates were removed)``.

    Keyed on each item's own ``key`` (link, else normalised title). The FIRST
    occurrence wins, and callers pass feeds most-trusted-first, so the kept
    copy is the one from the better source.

    The count is returned rather than logged because "42 headlines, 9
    duplicates removed" tells a reader something about their feed mix.
    """
    seen: set[str] = set()
    unique: list[object] = []
    removed = 0
    for item in items:
        key = str(getattr(item, "key", "") or getattr(item, "title", "")).strip().lower()
        if not key:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(item)
    return unique, removed


# ---------------------------------------------------------------------------
# tone
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToneResult:
    """The tone of a piece of text, and the evidence for it.

    ``matched`` exists so the UI can show *why*. A label the reader cannot
    interrogate is a number with a confident font.
    """

    label: str = "neutral"
    score: float = 0.0
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()
    negated: tuple[str, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> tuple[str, ...]:
        return (*self.positive, *self.negative)

    @property
    def explained(self) -> str:
        if not self.matched:
            return "no directional terms matched"
        parts = []
        if self.positive:
            parts.append("positive: " + ", ".join(self.positive))
        if self.negative:
            parts.append("negative: " + ", ".join(self.negative))
        if self.negated:
            parts.append("negated: " + ", ".join(self.negated))
        return "; ".join(parts)


def tone(text: object) -> ToneResult:
    """Classify the words in ``text``. Never a price prediction.

    Bag-of-words against :data:`POSITIVE` and :data:`NEGATIVE`, with one
    refinement: a term immediately preceded (within two words) by a
    :data:`NEGATORS` entry is dropped rather than counted, because "no fall in
    profit" is not bad news and a plain word count says it is.

    ``score`` is ``(positive - negative) / (positive + negative)`` in
    ``[-1, 1]``, and ``0.0`` with an empty ``matched`` means *nothing matched*
    — which is not the same as balanced. The label distinguishes them.
    """
    words = _words(text)
    if not words:
        return ToneResult()

    positive: list[str] = []
    negative: list[str] = []
    negated: list[str] = []

    for index, word in enumerate(words):
        bucket = POSITIVE if word in POSITIVE else (NEGATIVE if word in NEGATIVE else None)
        if bucket is None:
            continue
        window = words[max(index - 2, 0) : index]
        if any(w in NEGATORS or w in DAMPENERS for w in window):
            negated.append(word)
            continue
        (positive if bucket is POSITIVE else negative).append(word)

    total = len(positive) + len(negative)
    if not total:
        return ToneResult(negated=tuple(negated))

    score = (len(positive) - len(negative)) / total
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "mixed"

    return ToneResult(
        label=label,
        score=float(score),
        positive=tuple(dict.fromkeys(positive)),
        negative=tuple(dict.fromkeys(negative)),
        negated=tuple(dict.fromkeys(negated)),
    )


# ---------------------------------------------------------------------------
# instrument matching
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mention:
    """One instrument found in one headline, with the term that found it."""

    ticker: str
    name: str
    item: object
    matched_on: str = ""
    term: str = ""

    @property
    def title(self) -> str:
        return str(getattr(self.item, "title", ""))

    @property
    def published(self) -> datetime | None:
        stamp = getattr(self.item, "published", None)
        return stamp if isinstance(stamp, datetime) else None


def _leading_words(records: Sequence[object]) -> dict[str, int]:
    """How many records share each leading distinctive name word.

    The fix for group names. "Tata Consultancy Services" and "Tata Power
    Company" both reduce to a leading distinctive word of "tata", so matching
    on it tagged every Tata headline to *both* -- and to Tata Steel, Tata
    Motors and the rest. Same for Adani, Bajaj, Birla and Mahindra.

    A leading word shared by more than one supplied record is therefore not
    specific enough to identify any of them, and :func:`_match_keys` requires
    the two-word form instead. Computed from the records actually passed in,
    so it adapts to whatever the user holds rather than needing a hand-kept
    list of Indian business houses.
    """
    counts: dict[str, int] = {}
    for record in records:
        lead = next(
            (
                w
                for w in _words(getattr(record, "name", ""))
                if w not in GENERIC_NAME_WORDS and len(w) >= 4
            ),
            "",
        )
        if lead:
            counts[lead] = counts.get(lead, 0) + 1
    return counts


def _match_keys(record: object, shared: Mapping[str, int] | None = None) -> list[tuple[str, str]]:
    """``(key, how)`` pairs a headline could mention this instrument by.

    Three sources, each filtered to avoid the false positives that make
    headline matching useless:

    * the **symbol**, if at least :data:`MIN_MATCH_LENGTH` characters,
    * each **curated alias** ("RIL", "HUL") — these are the names people
      actually write,
    * the **distinctive words of the company name**, with
      :data:`GENERIC_NAME_WORDS` removed. Without that filter a company named
      "... India Industries Ltd" matches every headline containing "India".
    """
    keys: list[tuple[str, str]] = []

    symbol = str(getattr(record, "symbol", "") or "").strip().lower()
    if len(symbol) >= MIN_MATCH_LENGTH:
        keys.append((symbol, "ticker"))

    for alias in getattr(record, "aliases", ()) or ():
        text = str(alias or "").strip().lower()
        if len(text) >= MIN_MATCH_LENGTH:
            keys.append((text, "alias"))

    raw = _words(getattr(record, "name", ""))
    lead = next((w for w in raw if w not in GENERIC_NAME_WORDS and len(w) >= 4), "")
    if lead:
        # The two-word form is built from the RAW name, not from the
        # generic-filtered list. "Tata Power Company Limited" has every word
        # after "Tata" in GENERIC_NAME_WORDS, so filtering first left it with
        # no name key at all and made a major company unmatchable. The next
        # word as written -- "tata power" -- is both specific and how
        # headlines actually name it.
        position = raw.index(lead)
        if position + 1 < len(raw):
            keys.append((f"{lead} {raw[position + 1]}", "name"))
        # The single leading word only when no OTHER supplied record shares
        # it: "tata" identifies nothing when two Tata companies are tracked,
        # but "infosys" identifies Infosys.
        if (shared or {}).get(lead, 1) <= 1:
            keys.append((lead, "name"))

    return keys


def match_instruments(
    items: Sequence[object], records: Sequence[object], limit_per_item: int = 3
) -> list[Mention]:
    """Find which of ``records`` each headline mentions.

    ``records`` is expected to be a **small, relevant** list -- the user's
    holdings and watchlist -- not the whole universe. Two reasons, and both
    are about the result being useful rather than about speed:

    * "which of my positions is in the news" is the question a portfolio tool
      should answer; "which of 7,500 listed companies shares a word with this
      headline" is not,
    * a market-wide scan produces false positives faster than it produces
      matches, and a page of wrong tags is worse than no tags.

    A longer key wins where several match one headline, so "reliance
    industries" beats "reliance", and where two records tie the more heavily
    traded one wins (``prominence``, set by the universe loader).
    """
    if not items or not records:
        return []

    shared = _leading_words(records)
    index: list[tuple[str, str, object]] = []
    for record in records:
        for key, how in _match_keys(record, shared):
            index.append((key, how, record))
    if not index:
        return []

    # Longest key first: a match on "reliance industries" should be preferred
    # over the substring "reliance" pointing at a different company.
    index.sort(key=lambda entry: -len(entry[0]))

    mentions: list[Mention] = []
    for item in items:
        title_words = " ".join(_words(getattr(item, "title", "")))
        summary_words = " ".join(_words(getattr(item, "summary", "")))
        if not title_words and not summary_words:
            continue
        found: dict[str, Mention] = {}
        for key, how, record in index:
            ticker = str(getattr(record, "yahoo_symbol", "") or getattr(record, "symbol", ""))
            if ticker in found:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])"
            in_title = bool(re.search(pattern, title_words))
            # A match found only in the body is reported as such: otherwise a
            # tag appears next to a headline that visibly does not mention the
            # company, which reads as a bug.
            if not in_title and not re.search(pattern, summary_words):
                continue
            found[ticker] = Mention(
                ticker=ticker,
                name=str(getattr(record, "name", "") or ticker),
                item=item,
                matched_on=how if in_title else f"{how} (in body)",
                term=key,
            )
        ordered = sorted(
            found.values(),
            key=lambda m: (-len(m.term), -_prominence_of(records, m.ticker), m.ticker),
        )
        mentions.extend(ordered[: max(int(limit_per_item), 1)])
    return mentions


def _prominence_of(records: Sequence[object], ticker: str) -> float:
    for record in records:
        if str(getattr(record, "yahoo_symbol", "")) == ticker:
            try:
                return float(getattr(record, "prominence", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# frames for display
# ---------------------------------------------------------------------------
def items_frame(items: Iterable[object]) -> pd.DataFrame:
    """Headlines as a frame, tone included, newest first."""
    rows = []
    for item in items:
        result = tone(f"{getattr(item, 'title', '')} {getattr(item, 'summary', '')}")
        rows.append(
            {
                "published": getattr(item, "published", None),
                "title": str(getattr(item, "title", "")),
                "source": str(getattr(item, "source", "")),
                "tone": result.label,
                "why": result.explained,
                "link": str(getattr(item, "link", "")),
            }
        )
    columns = ["published", "title", "source", "tone", "why", "link"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("published", ascending=False, na_position="last").reset_index(
        drop=True
    )


def mentions_frame(mentions: Iterable[Mention]) -> pd.DataFrame:
    """One row per (instrument, headline) pair."""
    rows = []
    for mention in mentions:
        result = tone(mention.title)
        rows.append(
            {
                "ticker": mention.ticker,
                "name": mention.name,
                "published": mention.published,
                "title": mention.title,
                "matched_on": f"{mention.matched_on} “{mention.term}”",
                "tone": result.label,
                "why": result.explained,
                "link": str(getattr(mention.item, "link", "")),
            }
        )
    columns = ["ticker", "name", "published", "title", "matched_on", "tone", "why", "link"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("published", ascending=False, na_position="last").reset_index(
        drop=True
    )


def calendar_frame(events: Iterable[object]) -> pd.DataFrame:
    """Calendar events as a frame, soonest first."""
    rows = [
        {
            "when": getattr(event, "when", None),
            "country": str(getattr(event, "country", "")),
            "impact": str(getattr(event, "impact", "")),
            "event": str(getattr(event, "title", "")),
            "forecast": str(getattr(event, "forecast", "")),
            "previous": str(getattr(event, "previous", "")),
        }
        for event in events
    ]
    columns = ["when", "country", "impact", "event", "forecast", "previous"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("when", na_position="last")
        .reset_index(drop=True)
    )


def group_by_impact(events: Iterable[object]) -> dict[str, list[object]]:
    """Events keyed by impact rating, strongest bucket first."""
    out: dict[str, list[object]] = {}
    for event in events:
        out.setdefault(str(getattr(event, "impact", "") or "Unrated"), []).append(event)
    order = ["High", "Medium", "Low", "Holiday", "Unrated"]
    return {key: out[key] for key in order if key in out} | {
        key: value for key, value in out.items() if key not in order
    }


@dataclass(frozen=True)
class FeedSummary:
    """Headline counts and tone mix, typed.

    A ``dict[str, object]`` was tried first here and in two other modules,
    and it costs every caller its types: the UI could not do arithmetic on
    ``share_classified`` without the checker objecting, correctly.
    """

    n_items: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_mixed: int = 0
    n_neutral: int = 0
    share_classified: float = float("nan")
    newest: datetime | None = None
    oldest: datetime | None = None


def summarise(items: Sequence[object]) -> FeedSummary:
    """Headline counts and tone mix, for a KPI row.

    ``n_neutral`` counts headlines the lexicon did not classify at all, kept
    separate from "mixed" (which matched terms on both sides). Collapsing them
    would overstate how much of the feed was actually read.
    """
    counts = {"positive": 0, "negative": 0, "mixed": 0, "neutral": 0}
    for item in items:
        counts[tone(f"{getattr(item, 'title', '')} {getattr(item, 'summary', '')}").label] += 1

    dated: list[datetime] = [
        stamp
        for stamp in (getattr(item, "published", None) for item in items)
        if isinstance(stamp, datetime)
    ]
    classified = counts["positive"] + counts["negative"] + counts["mixed"]
    return FeedSummary(
        n_items=len(items),
        n_positive=counts["positive"],
        n_negative=counts["negative"],
        n_mixed=counts["mixed"],
        n_neutral=counts["neutral"],
        share_classified=(classified / len(items)) if items else float("nan"),
        newest=max(dated) if dated else None,
        oldest=min(dated) if dated else None,
    )
