"""Resolving a typed company name to the ticker the user actually meant.

This is the riskiest step in the whole research page, and it is unsafe in a way
that is easy to miss: **the wrong company produces a completely plausible
report.** A P/E of 24, a 12% margin, a Hold verdict -- none of it looks wrong,
because none of it *is* wrong. It is simply about a different business. There is
no downstream check that catches this, so the resolution step has to.

Two measured facts shaped the design, both from live probing:

**The provider's own search does not understand full names.** ``search("reliance
industries")`` returns nothing, and neither does ``"hd fc bank"`` -- the query
is matched against symbols and short aliases, not against the long name a
person would type. So the offline universe, which does carry full names, is
tried first and the provider is the fallback rather than the reverse. The
opposite order would fail on the most ordinary input there is.

**The provider mis-resolves ambiguous short names.** ``search("tcs")`` returns
``TCS.TO`` -- Tecsys Inc of Canada -- alongside ``TCS.NS``, Tata Consultancy
Services, and the Canadian one can rank first. A page that took the top hit
would generate a full research report on Tecsys for someone who typed "TCS".
That is the failure this module exists to prevent: :func:`resolve_candidates`
therefore returns a *list* and there is deliberately no function here that
returns a single answer. The page must ask.

Both sources are merged and de-duplicated by symbol, the offline hits first
because they carry a verified country, currency and exchange while a provider
hit carries none. Every candidate states *why* it surfaced, because a result a
reader cannot explain is a result they will not check.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src import stock_universe as universe

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LIMIT",
    "Candidate",
    "Resolution",
    "resolve_candidates",
    "looks_like_a_symbol",
]

#: How many candidates to offer. Long enough to cover the realistic ambiguity
#: ("tcs" genuinely has a handful of meanings) and short enough that the list is
#: read rather than skimmed -- a disambiguation list nobody reads is the same as
#: no disambiguation list.
DEFAULT_LIMIT = 8


@dataclass(frozen=True, slots=True)
class Candidate:
    """One company the typed text might have meant, and where it came from."""

    symbol: str
    name: str = ""
    exchange: str = ""
    currency: str = ""
    country: str = ""
    instrument_type: str = ""
    sector: str = ""
    #: ``"universe"`` or ``"provider"``. Not cosmetic: a universe row has a
    #: verified currency and country behind it and a provider row does not, so
    #: the page labels which is which rather than presenting them as equals.
    source: str = "universe"
    #: Why this row surfaced -- "ticker", "name", "alias", "fuzzy name", or the
    #: provider's own search. Shown in the UI.
    matched_on: str = ""
    score: float = 0.0

    @property
    def label(self) -> str:
        """One line a reader can choose from without opening anything."""
        parts = [self.name or self.symbol, self.symbol]
        if self.exchange:
            parts.append(self.exchange)
        if self.country:
            parts.append(self.country)
        return " · ".join(parts)

    @property
    def from_universe(self) -> bool:
        return self.source == "universe"

    @property
    def is_fund(self) -> bool:
        return self.instrument_type.upper() in {"ETF", "MUTUALFUND", "FUND"}


@dataclass(frozen=True, slots=True)
class Resolution:
    """The candidates for one query, plus what the search did and did not reach.

    ``offline_only`` is set when the provider was never asked -- either because
    the query resolved offline, or because the caller asked for an offline-only
    resolution. It matters: an offline-only result cannot have found a company
    missing from the bundled files, and the page says so rather than implying
    the list is exhaustive.
    """

    query: str = ""
    candidates: tuple[Candidate, ...] = ()
    offline_only: bool = False
    notices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.candidates)

    @property
    def unambiguous(self) -> bool:
        """Exactly one candidate. Still shown as a choice, never auto-picked."""
        return len(self.candidates) == 1

    @property
    def best(self) -> Candidate | None:
        """The top-ranked candidate -- for *highlighting*, never for selecting."""
        return self.candidates[0] if self.candidates else None

    @property
    def ambiguous(self) -> bool:
        """True when more than one company answers to the query."""
        return len(self.candidates) > 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def looks_like_a_symbol(text: str) -> bool:
    """True when the text could be a ticker rather than a company name.

    A ticker is short, has no spaces, and is mostly upper-case letters once the
    exchange suffix is peeled off. Used to explain an empty result ("that looks
    like a symbol -- try it directly") rather than to change the search, because
    a rule that silently reinterprets a query is the same class of bug as
    silently picking a company.
    """
    token = (text or "").strip()
    if not token or " " in token:
        return False
    stem = token.split(".", 1)[0]
    if not 1 <= len(stem) <= 12:
        return False
    return stem.replace("&", "").replace("-", "").isalnum() and stem.upper() == stem


def _from_record(record: object, matched_on: str, score: float) -> Candidate:
    return Candidate(
        symbol=str(getattr(record, "yahoo_symbol", "") or "").strip().upper(),
        name=str(getattr(record, "name", "") or "").strip(),
        exchange=str(getattr(record, "exchange", "") or "").strip(),
        currency=str(getattr(record, "currency", "") or "").strip(),
        country=str(getattr(record, "country", "") or "").strip(),
        instrument_type=str(getattr(record, "instrument_type", "") or "").strip(),
        sector=str(getattr(record, "sector", "") or "").strip(),
        source="universe",
        matched_on=matched_on,
        score=score,
    )


def _from_provider(hit: dict, rank: int) -> Candidate:
    """A provider search hit as a candidate.

    Currency, country and sector are left empty rather than guessed. A provider
    hit carries none of them, and inferring a currency from the exchange suffix
    is exactly the kind of plausible-looking assumption this module refuses.
    """
    return Candidate(
        symbol=str(hit.get("symbol", "") or "").strip().upper(),
        name=str(hit.get("name", "") or "").strip(),
        exchange=str(hit.get("exchange", "") or "").strip(),
        source="provider",
        matched_on=str(hit.get("source", "") or "provider search"),
        # Decays with rank so offline hits and provider hits stay in their own
        # bands and a provider hit never outranks an offline one on score alone.
        score=max(0.0, 0.4 - rank * 0.01),
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def _offline(text: str, limit: int) -> tuple[list[Candidate], str]:
    """Universe hits for ``text``, best-first. Never raises."""
    try:
        hits = universe.search(text, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a search failure is an empty result
        logger.warning("Universe search for %r failed: %s", text, exc)
        return [], "The offline universe could not be searched."
    return [
        _from_record(hit.record, getattr(hit, "matched_on", "") or "name", float(hit.score))
        for hit in hits
    ], ""


def _online(text: str, limit: int) -> tuple[list[Candidate], str]:
    """Provider hits for ``text``, best-first. Never raises."""
    try:
        from src.data import fetcher

        hits = fetcher.search_symbols(text, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a search failure is an empty result
        logger.warning("Provider search for %r failed: %s", text, exc)
        return [], f"The provider search could not be reached ({type(exc).__name__})."
    return [_from_provider(hit, rank) for rank, hit in enumerate(hits)], ""


def _merge(
    offline: Sequence[Candidate], online: Sequence[Candidate], limit: int
) -> tuple[Candidate, ...]:
    """Offline first, provider hits appended, de-duplicated by symbol.

    De-duplication keeps the first occurrence, so a company present in both
    keeps its universe row -- and therefore its verified currency and country --
    rather than being downgraded to the provider row that carries neither.
    """
    seen: set[str] = set()
    merged: list[Candidate] = []
    for candidate in (*offline, *online):
        if not candidate.symbol or candidate.symbol in seen:
            continue
        seen.add(candidate.symbol)
        merged.append(candidate)
    return tuple(merged[:limit])


def resolve_candidates(
    text: str,
    limit: int = DEFAULT_LIMIT,
    *,
    online: bool = True,
) -> Resolution:
    """Every company ``text`` might refer to, best-first, **never just one**.

    There is no single-answer variant of this function on purpose. The provider
    resolves ``"tcs"`` to Tecsys Inc as readily as to Tata Consultancy Services,
    and the caller has no way to tell which of those it received -- so the
    choice is returned to the user, who does.

    ``online=False`` skips the provider entirely, which the tests use and the
    page uses when the offline list already looks decisive.
    """
    query = (text or "").strip()
    if not query:
        return Resolution(query=query, notices=("Enter a company name or ticker.",))

    offline, offline_notice = _offline(query, limit)
    notices: list[str] = []
    if offline_notice:
        notices.append(offline_notice)

    online_hits: list[Candidate] = []
    offline_only = not online
    if online and not _looks_decisive(offline):
        online_hits, online_notice = _online(query, limit)
        if online_notice:
            notices.append(online_notice)
    elif online:
        offline_only = True

    candidates = _merge(offline, online_hits, limit)

    if not candidates:
        notices.append(
            f"Nothing matched {query!r}. "
            + (
                "That looks like a ticker, so try adding the exchange suffix "
                "(.NS for NSE, .BO for BSE) or check the spelling."
                if looks_like_a_symbol(query)
                else "Check the spelling, or try the ticker with its exchange "
                "suffix (.NS for NSE, .BO for BSE)."
            )
        )

    return Resolution(
        query=query,
        candidates=candidates,
        offline_only=offline_only,
        notices=tuple(notices),
    )


def _looks_decisive(offline: Sequence[Candidate]) -> bool:
    """True when the offline hits are good enough that the provider adds nothing.

    The rule is narrow on purpose. A single exact-ticker hit is decisive. A
    single *name* hit is not: the universe holds only curated listings, and a
    company missing from it -- a recent IPO, a small cap, a foreign listing --
    would be silently unreachable if one offline hit ended the search.
    """
    if len(offline) != 1:
        return False
    only = offline[0]
    return only.matched_on.strip().lower() in {"ticker", "symbol"}
