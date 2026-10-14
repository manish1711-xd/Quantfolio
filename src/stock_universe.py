"""Local stock-master reference data: search by name, resolve to a Yahoo symbol.

Why this module exists
----------------------
Until now the only way to add a holding was to already know its Yahoo Finance
symbol -- ``RELIANCE.NS``, not "Reliance". That is a real usability wall: the
suffix convention is invisible to anyone who has not used yfinance, ``.NS``
versus ``.BO`` is a coin flip, and a US-listed ADR shares its root symbol with
the Indian listing (``INFY`` is the NYSE ADR; ``INFY.NS`` is the NSE line).
This module turns "reliance" into a concrete, disambiguated instrument.

Design constraints
------------------
* **Offline and pure.** No network, no Streamlit, no pandas dependency in the
  hot path -- the universe is read once from bundled CSVs with ``csv`` from the
  standard library and cached in a module-level singleton. Downloading a
  full exchange listing on every app start would be slow, fragile and rude to
  the provider; a curated local file is instant and works on a plane.
* **The local file is a convenience, never a gate.** A symbol absent from the
  universe is still perfectly usable -- :func:`resolve` returns ``None`` and
  the caller falls back to passing the raw text through to the data layer,
  which is the only thing that can actually confirm a symbol prices. Nothing
  here validates existence and nothing here fabricates one.
* **Reference data only.** ``sector`` in these files is an indicative label for
  search and filtering. Authoritative sector/industry classification comes
  from the provider via :class:`src.models.SecurityProfile`; never let a
  universe row overwrite a fetched profile.

Coverage
--------
Bundled: NSE large/mid caps and popular Indian ETFs, US large caps and the
major US ETFs, plus the global indices the benchmark selector offers. It is a
curated subset, deliberately -- roughly the instruments a retail or analyst
portfolio actually holds -- and the honest way to describe it in the UI is
"common Indian and US listings", not "every stock". Extend it by dropping more
rows into ``data/*.csv`` and adding the filename to :data:`UNIVERSE_FILES`.

File format (``data/indian_stocks.csv``, ``data/us_stocks.csv``, ...)::

    symbol,yahoo_symbol,name,exchange,country,currency,instrument_type,sector,aliases
    RELIANCE,RELIANCE.NS,Reliance Industries Ltd,NSE,India,INR,EQUITY,Energy,RIL|Reliance

``aliases`` is pipe-separated and exists for the names people actually type:
"HUL" for Hindustan Unilever, "Bank Nifty" for ^NSEBANK, "Fevicol" for Pidilite.
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_DIR",
    "UNIVERSE_FILES",
    "UNIVERSE_COLUMNS",
    "EXCHANGE_PRIORITY",
    "StockRecord",
    "SearchHit",
    "load_universe",
    "refresh",
    "search",
    "resolve",
    "candidates_for",
    "by_yahoo_symbol",
    "normalise_symbol",
    "suffix_hint",
    "exchanges",
    "countries",
    "instrument_types",
    "universe_summary",
    "universe_meta",
    "as_of",
]

#: ``<project root>/data`` -- this file is ``<root>/src/stock_universe.py``.
DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"

#: Master files loaded, in order. A missing file is skipped with a log line
#: rather than raising: the app must still start if someone deletes one.
UNIVERSE_FILES: tuple[str, ...] = (
    "indian_stocks.csv",
    "us_stocks.csv",
    "global_indices.csv",
)

#: Required header of a universe CSV.
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "yahoo_symbol",
    "name",
    "exchange",
    "country",
    "currency",
    "instrument_type",
    "sector",
    "aliases",
    # Appended when the full NSE/BSE listing landed. Rows are read by column
    # NAME, so a file written before these existed still loads and simply has
    # no ISIN and no turnover.
    "isin",
    "turnover",
)

#: Tie-break order when one root symbol exists on several exchanges. India
#: first because this application is India-first: a user typing "INFY" in a
#: rupee-denominated portfolio almost certainly means the NSE line, not the
#: NYSE ADR. The ADR is still returned as a lower-ranked, clearly-labelled
#: second result, so the choice stays the user's.
EXCHANGE_PRIORITY: tuple[str, ...] = ("NSE", "BSE", "NASDAQ", "NYSE", "NYSEARCA", "INDEX")

#: Score floor for a fuzzy (difflib) name match to be offered at all. Below
#: this the "match" is noise and showing it would erode trust in the top hit.
_FUZZY_FLOOR = 0.62

#: Yahoo suffixes that mean "this symbol already names its exchange".
_KNOWN_SUFFIXES: frozenset[str] = frozenset(
    {".NS", ".BO", ".L", ".TO", ".HK", ".AX", ".DE", ".PA", ".SI", ".T"}
)

_ALIAS_SEPARATOR = "|"
_WORD_SPLIT = re.compile(r"[^a-z0-9&]+")
_MULTISPACE = re.compile(r"\s+")

#: Words carrying no discriminating power in a company name. Dropped when
#: building the token set so "Reliance Industries Ltd" and "reliance" match on
#: the tokens that matter.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "ltd",
        "limited",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "plc",
        "the",
        "and",
        "of",
        "sa",
        "nv",
        "ag",
        "group",
        "holdings",
        "holding",
    }
)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def _norm(text: object) -> str:
    """Lower-case, collapse whitespace, strip. "" for missing values."""
    if text is None:
        return ""
    return _MULTISPACE.sub(" ", str(text).strip().lower())


def _tokens(text: str) -> frozenset[str]:
    """Meaningful lower-case word tokens of a name, stopwords removed.

    Falls back to the raw tokens when every one of them is a stopword, so a
    company literally called "The Group" is still searchable.
    """
    raw = {t for t in _WORD_SPLIT.split(_norm(text)) if t}
    meaningful = raw - _STOPWORDS
    return frozenset(meaningful or raw)


def _clean_symbol(text: object) -> str:
    """Upper-case, strip whitespace and any stray quoting from a symbol."""
    return str(text or "").strip().strip("'\"").upper()


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StockRecord:
    """One instrument in the local master file.

    ``symbol`` is the native trading symbol as the exchange lists it
    (``RELIANCE``); ``yahoo_symbol`` is what the data layer must be handed
    (``RELIANCE.NS``). They differ for every non-US listing, and conflating
    them is precisely the bug this module exists to remove.
    """

    symbol: str
    yahoo_symbol: str
    name: str
    exchange: str = ""
    country: str = ""
    currency: str = ""
    instrument_type: str = "EQUITY"
    sector: str = ""
    aliases: tuple[str, ...] = ()
    #: ISO 6166 identifier from the exchange listing, "" when unknown. Lets
    #: :mod:`src.isin` resolve a broker export that names only ISINs.
    isin: str = ""
    #: Traded value in rupees on the bhavcopy date the universe was built
    #: from, 0 when the instrument did not trade (or the file has none). The
    #: MEASURED number is stored in the CSV rather than a score, so the
    #: ranking rule can change without re-downloading anything.
    turnover: float = 0.0
    #: ``turnover`` on a 0-1 scale, filled in by :func:`load_universe` --
    #: the only place the whole distribution is known. 0.0 for anything that
    #: did not trade and for every file with no turnover data at all.
    #:
    #: See :func:`_scale_prominence` for why this is a log of the value
    #: normalised over the observed range, rather than either a raw log or a
    #: percentile. Both were tried on the real 7,573-row file and both failed
    #: on the cases that matter.
    prominence: float = 0.0

    @property
    def label(self) -> str:
        """One-line human label: ``Reliance Industries Ltd · RELIANCE.NS · NSE``."""
        parts = [self.name or self.yahoo_symbol, self.yahoo_symbol]
        if self.exchange:
            parts.append(self.exchange)
        return " · ".join(parts)

    @property
    def is_fund(self) -> bool:
        return self.instrument_type.upper() in {"ETF", "MUTUALFUND", "FUND"}

    @property
    def is_index(self) -> bool:
        return self.instrument_type.upper() == "INDEX"

    @property
    def priority(self) -> int:
        """Exchange rank used only to break score ties (lower sorts first)."""
        try:
            return EXCHANGE_PRIORITY.index(self.exchange.upper())
        except ValueError:
            return len(EXCHANGE_PRIORITY)


@dataclass(frozen=True)
class SearchHit:
    """A scored search result.

    ``matched_on`` names *why* the row surfaced ("ticker", "name", "alias",
    "fuzzy name"). The UI shows it, because a result the user cannot explain
    is a result they will not trust.
    """

    record: StockRecord
    score: float
    matched_on: str

    # Convenience passthroughs so callers can treat a hit like a record.
    @property
    def yahoo_symbol(self) -> str:
        return self.record.yahoo_symbol

    @property
    def label(self) -> str:
        return self.record.label


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _turnover(value: object) -> float:
    """Traded value as a non-negative float. 0.0 for anything unreadable.

    Never raises and never returns NaN: this feeds a ranking comparison, and
    a NaN there would make the sort order depend on the input sequence.
    """
    try:
        out = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) and out > 0.0 else 0.0


def _read_file(path: Path) -> list[StockRecord]:
    """Parse one universe CSV. Returns [] (and logs) on any problem.

    Rows missing a ``yahoo_symbol`` are dropped -- that field is the only one
    the rest of the application cannot work without.
    """
    if not path.is_file():
        logger.info("Stock universe file not found, skipping: %s", path)
        return []

    records: list[StockRecord] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in ("yahoo_symbol", "name") if c not in (reader.fieldnames or [])]
            if missing:
                logger.warning("%s is missing required column(s) %s; skipped.", path.name, missing)
                return []
            for row in reader:
                yahoo = _clean_symbol(row.get("yahoo_symbol"))
                if not yahoo:
                    continue
                native = _clean_symbol(row.get("symbol")) or yahoo.split(".")[0]
                aliases = tuple(
                    a.strip()
                    for a in str(row.get("aliases") or "").split(_ALIAS_SEPARATOR)
                    if a.strip()
                )
                records.append(
                    StockRecord(
                        symbol=native,
                        yahoo_symbol=yahoo,
                        name=str(row.get("name") or "").strip() or yahoo,
                        exchange=str(row.get("exchange") or "").strip(),
                        country=str(row.get("country") or "").strip(),
                        currency=str(row.get("currency") or "").strip().upper(),
                        instrument_type=(
                            str(row.get("instrument_type") or "EQUITY").strip().upper()
                        ),
                        sector=str(row.get("sector") or "").strip(),
                        aliases=aliases,
                        isin=str(row.get("isin") or "").strip().upper(),
                        turnover=_turnover(row.get("turnover")),
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Could not read stock universe file %s: %s", path, exc)
        return []
    return records


#: Score floor for an instrument that traded at all, so 0.0 is reserved for
#: "did not trade". Small, because the distinction between the quietest traded
#: name and an untraded one is real but slight.
_PROMINENCE_FLOOR = 0.05


def _scale_prominence(records: Sequence[StockRecord]) -> tuple[StockRecord, ...]:
    """Fill in ``prominence``: log turnover, normalised over the observed range.

        prominence = (log10(t) - log10(min)) / (log10(max) - log10(min))

    Two simpler transforms were tried first on the real 7,573-row file, and
    the reason each failed is worth keeping:

    * **A raw log over a fixed divisor** (``log10(t)/11``) put a ₹86 cr name
      and a ₹25 cr name 0.05 apart -- less than the cost of one extra symbol
      character in the prefix rule -- so search still preferred the shorter
      symbol regardless of size.
    * **A percentile rank** fixed the mid-cap spread but *saturates at the
      top*, which is exactly where search collisions happen: HDFC Bank
      (₹2,595 cr) and HDFC AMC (₹241 cr) came out 0.016 apart, so "hdfc" still
      did not return HDFC Bank first. Worse, ranked *within* each exchange it
      inverted every dual listing, because BSE's ~5,000 mostly-thin names put
      a mid-cap at a higher percentile there than the same company's busier
      NSE line reaches among NSE's more uniformly liquid population.

    Normalising the log over the range the data actually spans keeps the ratio
    information at both ends: the same HDFC pair lands 0.12 apart, and the
    Tata cluster spreads across 0.15. One pool for all exchanges, so the
    busier line of a dual-listed company wins on its own merits.

    Instruments that did not trade keep 0.0 and are excluded from the range,
    so one illiquid name cannot drag the scale. A universe with no turnover
    anywhere (the US and index files carry none) leaves every row at 0.0 --
    exactly the ordering those files had before turnover existed.

    A traded instrument is mapped into ``[_PROMINENCE_FLOOR, 1]`` rather than
    ``[0, 1]``, so 0.0 means "did not trade" and nothing else. Without the
    floor the *least* traded instrument scored exactly the same as one that
    never traded at all, and "changed hands today" is real information about
    an instrument a user might be about to buy.
    """
    traded = [record.turnover for record in records if record.turnover > 0.0]
    if not traded:
        return tuple(records)

    low = math.log10(min(traded))
    high = math.log10(max(traded))
    span = high - low
    if span <= 0.0:
        # Every traded instrument shows the same turnover: nothing to rank on,
        # so say so with a flat 1.0 rather than dividing by zero.
        return tuple(
            replace(record, prominence=1.0) if record.turnover > 0.0 else record
            for record in records
        )

    out: list[StockRecord] = []
    for record in records:
        if record.turnover <= 0.0:
            out.append(record)
            continue
        scaled = (math.log10(record.turnover) - low) / span
        ranged = _PROMINENCE_FLOOR + (1.0 - _PROMINENCE_FLOOR) * scaled
        out.append(replace(record, prominence=min(max(ranged, _PROMINENCE_FLOOR), 1.0)))
    return tuple(out)


@lru_cache(maxsize=8)
def load_universe(data_dir: str | None = None) -> tuple[StockRecord, ...]:
    """Every instrument in the bundled master files, de-duplicated.

    Cached for the life of the process: the files are static reference data,
    so re-parsing them on every keystroke of a search box would be pure waste.
    Call :func:`refresh` after editing a CSV.

    De-duplication is by ``yahoo_symbol``, first file wins, so listing a
    symbol in two files (or twice in one) cannot produce duplicate search
    results -- the requirement that made this worth centralising.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    seen: dict[str, StockRecord] = {}
    for filename in UNIVERSE_FILES:
        for record in _read_file(root / filename):
            seen.setdefault(record.yahoo_symbol, record)
    # Ranked here rather than in _read_file: a percentile needs the whole
    # distribution, and per exchange that spans more than one file.
    return _scale_prominence(list(seen.values()))


def refresh() -> None:
    """Drop the cached universe so the next call re-reads the CSVs."""
    load_universe.cache_clear()
    _indices.cache_clear()
    universe_meta.cache_clear()


@lru_cache(maxsize=8)
def _indices(data_dir: str | None = None) -> dict[str, dict]:
    """Pre-built lookup tables over the universe.

    Built once alongside the universe itself because every search needs them
    and rebuilding per query would make the search O(n) in dict construction
    rather than O(n) in comparisons.

    * ``by_yahoo``  -- exact Yahoo symbol -> record
    * ``by_symbol`` -- native symbol -> [records] (several exchanges possible)
    * ``by_alias``  -- normalised alias / name -> [records]
    * ``tokens``    -- yahoo symbol -> token set of its name and aliases
    * ``fuzzy``     -- short comparison key -> [records], the difflib pool
    """
    universe = load_universe(data_dir)
    by_yahoo: dict[str, StockRecord] = {}
    by_symbol: dict[str, list[StockRecord]] = {}
    by_alias: dict[str, list[StockRecord]] = {}
    tokens: dict[str, frozenset[str]] = {}
    fuzzy: dict[str, list[StockRecord]] = {}

    for record in universe:
        by_yahoo[record.yahoo_symbol] = record
        by_symbol.setdefault(record.symbol, []).append(record)
        # A suffixed symbol is also addressable by its root ("RELIANCE.NS"
        # typed as "RELIANCE"), which is how brokers export Indian holdings.
        root = record.yahoo_symbol.split(".")[0].lstrip("^")
        if root and root != record.symbol:
            by_symbol.setdefault(root, []).append(record)

        name_key = _norm(record.name)
        if name_key:
            by_alias.setdefault(name_key, []).append(record)
        for alias in record.aliases:
            alias_key = _norm(alias)
            if alias_key:
                by_alias.setdefault(alias_key, []).append(record)

        blob = frozenset().union(_tokens(record.name), *(_tokens(a) for a in record.aliases))
        tokens[record.yahoo_symbol] = blob

        # Fuzzy pool. Comparing a typo against the FULL name is close to
        # useless -- difflib scores "relaince" against "reliance industries
        # ltd" at ~0.42, below any sane cutoff, while an unrelated short name
        # scores higher purely by being short. So the pool holds the SHORT
        # keys a person actually types: the symbol, each alias, and the first
        # meaningful word of the name. "relaince" then hits "reliance" at 0.88.
        candidates = {_norm(record.symbol), name_key}
        candidates.update(_norm(a) for a in record.aliases)
        first_word = next((t for t in _WORD_SPLIT.split(name_key) if t and t not in _STOPWORDS), "")
        candidates.add(first_word)
        for key in candidates:
            if key:
                fuzzy.setdefault(key, []).append(record)

    for bucket in (*by_symbol.values(), *by_alias.values(), *fuzzy.values()):
        bucket.sort(key=lambda r: (r.priority, r.yahoo_symbol))

    return {
        "by_yahoo": by_yahoo,
        "by_symbol": by_symbol,
        "by_alias": by_alias,
        "tokens": tokens,
        "fuzzy": fuzzy,
    }


# ---------------------------------------------------------------------------
# direct lookup
# ---------------------------------------------------------------------------
def by_yahoo_symbol(symbol: str, data_dir: str | None = None) -> StockRecord | None:
    """The record for an exact Yahoo symbol, or ``None`` if not in the universe."""
    return _indices(data_dir)["by_yahoo"].get(_clean_symbol(symbol))


def candidates_for(text: str, data_dir: str | None = None) -> tuple[StockRecord, ...]:
    """Every record an exact identifier could refer to, best-first.

    Usually length 0 or 1. Length > 1 means a genuine ambiguity worth showing
    the user: ``"INFY"`` is the NYSE ADR *and* the root of ``INFY.NS`` on the
    NSE. The importer surfaces these as "needs review" rather than picking
    one behind the user's back.
    """
    key = _clean_symbol(text)
    if not key:
        return ()
    idx = _indices(data_dir)

    found: list[StockRecord] = []
    exact = idx["by_yahoo"].get(key)
    if exact is not None:
        found.append(exact)
    for bucket in (idx["by_symbol"].get(key), idx["by_alias"].get(_norm(text))):
        for record in bucket or ():
            if record not in found:
                found.append(record)
    return tuple(found)


def resolve(
    text: str, data_dir: str | None = None, prefer_exchange: str | None = None
) -> StockRecord | None:
    """Best single record for an unambiguous identifier, else ``None``.

    Resolution order -- deliberately conservative, because this function's
    output is used to *rewrite* what the user typed:

    1. exact Yahoo symbol (``RELIANCE.NS``, ``^NSEI``)
    2. exact native symbol (``RELIANCE``) -- highest-priority exchange wins
    3. exact company name or alias (``reliance industries ltd``, ``hul``)

    ``prefer_exchange`` overrides that order for one exchange only, and only
    among candidates that already matched exactly. It exists for CSV import:
    a Zerodha file whose other rows are all ``.NS`` should resolve its bare
    ``INFY`` row to ``INFY.NS``, not to the NYSE ADR that happens to own that
    exact Yahoo symbol. It can never invent a match that was not already
    there.

    Fuzzy matching is intentionally NOT part of resolution. Guessing that
    "RELAINCE" meant "RELIANCE" and silently buying the latter is the kind of
    helpfulness a portfolio tool must not have. Fuzzy candidates are offered
    through :func:`search`, where the user picks.
    """
    found = candidates_for(text, data_dir)
    if not found:
        return None
    if prefer_exchange:
        wanted = prefer_exchange.strip().upper()
        preferred = next((r for r in found if r.exchange.upper() == wanted), None)
        if preferred is not None:
            return preferred
    return found[0]


def normalise_symbol(
    text: str, data_dir: str | None = None, prefer_exchange: str | None = None
) -> str:
    """Turn user-typed text into the symbol the data layer should be given.

    ``"reliance"`` -> ``"RELIANCE.NS"``, ``"  aapl "`` -> ``"AAPL"``,
    ``"Reliance Industries Ltd"`` -> ``"RELIANCE.NS"``.

    Unknown input is returned upper-cased and stripped but otherwise
    UNCHANGED. That matters: a symbol missing from the curated file is not a
    symbol that does not exist, and inventing a ``.NS`` suffix for it would
    turn a working ticker into a broken one.
    """
    record = resolve(text, data_dir, prefer_exchange)
    if record is not None:
        return record.yahoo_symbol
    return _clean_symbol(text)


def suffix_hint(symbol: str) -> str | None:
    """Advice for a bare symbol that looks like it is missing an exchange suffix.

    Returns ``None`` when the symbol already carries a known suffix, is an
    index (``^``), or resolves in the universe -- i.e. only speaks up when it
    has something useful to say.
    """
    key = _clean_symbol(symbol)
    if not key or key.startswith("^"):
        return None
    if any(key.endswith(suffix) for suffix in _KNOWN_SUFFIXES):
        return None
    if resolve(key) is not None:
        return None
    return (
        f"{key} has no exchange suffix. Indian listings need one — "
        f"{key}.NS for NSE or {key}.BO for BSE. US symbols are used as-is."
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
#: Score awarded by each match rule. Ordered so that an identifier match can
#: never be outranked by a name match: someone who types a ticker means it.
_SCORE_EXACT_YAHOO = 100.0
_SCORE_EXACT_SYMBOL = 96.0
_SCORE_EXACT_NAME = 92.0
#: Equal to an exact name match, deliberately. An alias IS a name the
#: instrument is known by -- "HUL", "RIL", or a previous official spelling --
#: and scoring it four points lower had a concrete cost once the universe held
#: both listings of every dual-listed company: BSE's own spelling of Reliance
#: happens to be the curated "Reliance Industries Ltd", so that query matched
#: the BSE row on `name` (92) and the NSE row on `alias` (88), and the less
#: liquid line won. Tied, the India-first EXCHANGE_PRIORITY tie-break settles
#: it, which is what that constant exists for.
_SCORE_EXACT_ALIAS = 92.0
_SCORE_SYMBOL_PREFIX = 80.0
_SCORE_NAME_PREFIX = 76.0
_SCORE_ALL_TOKENS = 68.0
_SCORE_NAME_SUBSTRING = 60.0
_SCORE_SYMBOL_SUBSTRING = 52.0
_SCORE_FUZZY_BASE = 50.0

#: Smallest gap between two adjacent score tiers above. Load-bearing: the
#: prominence bonus must stay strictly under it, or a weaker kind of match
#: could outrank a stronger one purely for being a big company. A test pins
#: both this value and that inequality.
_MIN_TIER_GAP = 4.0

#: Cost per extra character when a symbol merely STARTS WITH the query.
_PREFIX_LENGTH_PENALTY = 0.1

#: Largest penalty the symbol-prefix rule can apply (8 characters).
_MAX_PREFIX_PENALTY = 8 * _PREFIX_LENGTH_PENALTY

#: Most a record can gain from being heavily traded. Bounded by
#: ``_MIN_TIER_GAP - _MAX_PREFIX_PENALTY`` so the tiers stay strictly ordered:
#: the weakest possible member of a tier still beats the strongest possible
#: member of the tier below, whatever either one traded. Turnover therefore
#: only ever reorders hits that matched the same WAY -- it never promotes a
#: substring match over a name match. A test pins the inequality.
#:
#: Why any bonus at all: with the curated 243-row universe, ranking within a
#: tier alphabetically was fine. Across the full ~7,700-instrument market it
#: is actively bad. Measured on the real files, "tata" returned TATACAP
#: (₹25 cr traded) above TCS (₹592 cr) and TATAPOWER (₹439 cr), because the
#: prefix rule prefers a shorter symbol; and the typo "relaince" returned
#: Reliance Communications (₹0.3 cr) above Reliance Industries (₹1,767 cr),
#: because RCOM.NS sorts first alphabetically. Both are fixed by giving the
#: busier instrument a small, bounded edge.
_PROMINENCE_BONUS = 2.0


def _prominence_bonus(record: StockRecord) -> float:
    """A record's turnover as a bounded score bonus, in [0, _PROMINENCE_BONUS].

    Zero for anything that did not trade on the universe's bhavcopy date, and
    for every file built without turnover -- so an older universe, or one
    built with ``--no-bhavcopy``, ranks exactly as it did before.
    """
    return _PROMINENCE_BONUS * record.prominence


def _score(
    record: StockRecord, query: str, query_tokens: frozenset[str], token_blob: frozenset[str]
) -> tuple[float, str]:
    """Score one record against a normalised query. ``(0.0, "")`` for no match.

    Rules are checked strongest-first and the first hit wins, so the reported
    ``matched_on`` is always the strongest reason the row qualified.

    An exact identifier match is returned at its full tier score with NO
    prominence bonus: typing a ticker means that ticker, and a bigger company
    must never displace an exact hit. Every weaker tier gets the bonus, which
    is what makes a full-market universe rank usefully instead of
    alphabetically.
    """
    upper = query.upper()
    name_key = _norm(record.name)

    if record.yahoo_symbol == upper:
        return _SCORE_EXACT_YAHOO, "ticker"
    if record.symbol == upper or record.yahoo_symbol.split(".")[0].lstrip("^") == upper:
        return _SCORE_EXACT_SYMBOL, "ticker"
    if name_key == query:
        return _SCORE_EXACT_NAME, "name"
    if any(_norm(a) == query for a in record.aliases):
        return _SCORE_EXACT_ALIAS, "alias"
    bonus = _prominence_bonus(record)
    if record.symbol.startswith(upper):
        # Shorter symbols are the weaker signal now that turnover exists.
        # This rule was sized at 0.5/character when nothing else could break a
        # prefix tie, and at that weight it outvoted a 10x difference in
        # traded value: "hdfc" returned HDFC AMC (7 characters, ₹241 cr) ahead
        # of HDFC Bank (8 characters, ₹2,595 cr). Its original justification
        # -- "TCS should beat TCSPLUS for the query TCS" -- is handled by the
        # exact-symbol tier 16 points above, so the penalty only needs to
        # order names that are otherwise indistinguishable, which is what
        # 0.1/character does.
        penalty = min(len(record.symbol) - len(upper), 8) * _PREFIX_LENGTH_PENALTY
        return _SCORE_SYMBOL_PREFIX - penalty + bonus, "ticker"
    if name_key.startswith(query):
        return _SCORE_NAME_PREFIX + bonus, "name"
    if any(_norm(a).startswith(query) for a in record.aliases):
        return _SCORE_NAME_PREFIX - 2.0 + bonus, "alias"
    if query_tokens and query_tokens <= token_blob:
        return _SCORE_ALL_TOKENS + bonus, "name"
    if query in name_key:
        return _SCORE_NAME_SUBSTRING + bonus, "name"
    if any(query in _norm(a) for a in record.aliases):
        return _SCORE_NAME_SUBSTRING - 2.0 + bonus, "alias"
    if len(upper) >= 3 and upper in record.symbol:
        return _SCORE_SYMBOL_SUBSTRING + bonus, "ticker"
    return 0.0, ""


def search(
    query: str,
    limit: int = 10,
    exchanges: tuple[str, ...] | None = None,
    instrument_types: tuple[str, ...] | None = None,
    data_dir: str | None = None,
) -> list[SearchHit]:
    """Rank the local universe against ``query``.

    Matches on ticker, company name and alias, then falls back to a fuzzy
    name comparison (``difflib``) above :data:`_FUZZY_FLOOR` so a typo still
    surfaces the intended company as a *suggestion*.

    Results are unique by Yahoo symbol -- the requirement that duplicates never
    appear -- and ties break on exchange priority then symbol, so the ordering
    is deterministic and the same query always renders the same list.

    ``exchanges`` / ``instrument_types`` are case-insensitive filters
    ("NSE" only, or "EQUITY" and "ETF" but no indices). ``limit <= 0`` returns
    everything that matched.
    """
    normalised = _norm(query)
    if not normalised:
        return []

    idx = _indices(data_dir)
    token_map: dict[str, frozenset[str]] = idx["tokens"]
    query_tokens = _tokens(normalised)

    allowed_exchanges = {e.upper() for e in exchanges} if exchanges else None
    allowed_types = {t.upper() for t in instrument_types} if instrument_types else None

    best: dict[str, SearchHit] = {}
    for record in load_universe(data_dir):
        if allowed_exchanges is not None and record.exchange.upper() not in allowed_exchanges:
            continue
        if allowed_types is not None and record.instrument_type.upper() not in allowed_types:
            continue
        score, reason = _score(
            record, normalised, query_tokens, token_map.get(record.yahoo_symbol, frozenset())
        )
        if score <= 0.0:
            continue
        current = best.get(record.yahoo_symbol)
        if current is None or score > current.score:
            best[record.yahoo_symbol] = SearchHit(record, score, reason)

    # Fuzzy fallback: only when the exact rules found little AND nothing they
    # found was a confident identifier/name/alias hit. Padding a confident
    # "reliance -> RELIANCE.NS" with a 37-point difflib guess at "Elevance
    # Health" makes the result list look broken and teaches the user to
    # distrust the top row, which is the one that was right.
    strong = max((h.score for h in best.values()), default=0.0) >= _SCORE_EXACT_ALIAS
    if not strong and len(best) < max(limit, 1):
        pool: dict[str, list[StockRecord]] = idx["fuzzy"]
        for key in difflib.get_close_matches(
            normalised, list(pool), n=max(limit, 1) * 3, cutoff=_FUZZY_FLOOR
        ):
            ratio = difflib.SequenceMatcher(None, normalised, key).ratio()
            for record in pool[key]:
                if (
                    allowed_exchanges is not None
                    and record.exchange.upper() not in allowed_exchanges
                ):
                    continue
                if (
                    allowed_types is not None
                    and record.instrument_type.upper() not in allowed_types
                ):
                    continue
                if record.yahoo_symbol in best:
                    continue
                # Without the bonus every record sharing a fuzzy key scores
                # identically and the winner is whichever symbol sorts first --
                # which is how "relaince" returned Reliance Communications.
                best[record.yahoo_symbol] = SearchHit(
                    record,
                    _SCORE_FUZZY_BASE * ratio + _prominence_bonus(record),
                    "fuzzy match",
                )

    # Exchange priority ahead of turnover: India-first is a stated product
    # decision (see EXCHANGE_PRIORITY), so an exact hit on the NSE line is not
    # displaced by a BSE line that happened to trade more. Turnover breaks the
    # remaining ties, and the symbol is last so the order is fully
    # deterministic for a given file.
    hits = sorted(
        best.values(),
        key=lambda h: (
            -h.score,
            h.record.priority,
            -h.record.prominence,
            h.record.yahoo_symbol,
        ),
    )
    return hits if limit <= 0 else hits[:limit]


# ---------------------------------------------------------------------------
# facets & metadata
# ---------------------------------------------------------------------------
def exchanges(data_dir: str | None = None) -> tuple[str, ...]:
    """Distinct exchange codes present, in :data:`EXCHANGE_PRIORITY` order."""
    found = {r.exchange for r in load_universe(data_dir) if r.exchange}
    ranked = [e for e in EXCHANGE_PRIORITY if e in found]
    return tuple(ranked + sorted(found - set(ranked)))


def countries(data_dir: str | None = None) -> tuple[str, ...]:
    """Distinct countries present, alphabetically."""
    return tuple(sorted({r.country for r in load_universe(data_dir) if r.country}))


def instrument_types(data_dir: str | None = None) -> tuple[str, ...]:
    """Distinct instrument types present, alphabetically."""
    return tuple(sorted({r.instrument_type for r in load_universe(data_dir) if r.instrument_type}))


#: Provenance written by :mod:`src.universe_build`. Absent for a hand-edited
#: universe, which is why every reader treats a missing file as "unknown"
#: rather than as an error.
META_FILE = "universe_meta.json"


@lru_cache(maxsize=8)
def universe_meta(data_dir: str | None = None) -> dict:
    """Provenance of the generated universe, or ``{}`` if there is none.

    Carries the exchange files each row came from and, crucially, the
    ``as_of`` date of the bhavcopy the turnover was taken from. A universe
    presented without its date implies it is current, and for an end-of-day
    file that is never quite true.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    path = root / META_FILE
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def as_of(data_dir: str | None = None) -> str:
    """ISO date the universe's market data was taken from, or "".

    "" means the file was not generated by :mod:`src.universe_build` -- a
    hand-maintained or older universe. Callers say "date unknown" rather than
    guessing today.
    """
    return str(universe_meta(data_dir).get("as_of") or "")


def universe_summary(data_dir: str | None = None) -> dict[str, int]:
    """Row counts by exchange plus a ``total``, for the UI's coverage note.

    Stating the size honestly is the point: it lets the interface say "512
    instruments across NSE, NASDAQ and NYSE" rather than implying it knows
    every listed company in the world.
    """
    universe = load_universe(data_dir)
    counts: dict[str, int] = {"total": len(universe)}
    for record in universe:
        key = record.exchange or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts
