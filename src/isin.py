"""ISIN validation and ISIN -> Yahoo symbol resolution.

Why this is delicate
--------------------
An ISIN (ISO 6166) identifies a security exactly, and a *wrong* mapping puts
somebody else's company in your portfolio while everything on screen still
looks plausible. That makes fabricated data the one unacceptable failure mode
here, and it shapes every decision in this module:

* **Nothing is guessed.** There is no fuzzy matching, no "close enough" prefix
  logic and no inference from the ISIN body. An ISIN either resolves through a
  table or it does not resolve.
* **The check digit is enforced.** ISO 6166 carries a mod-10 (Luhn) check
  digit over the first eleven characters, so a typo or a truncation is
  *arithmetically* detectable rather than a matter of judgement. Every entry in
  the shipped table is verified by that arithmetic in the test suite, which
  makes a mistyped seed row a build failure instead of a silent mis-holding.
* **The country prefix is cross-checked.** ``INE...`` is an Indian security and
  must not resolve to a NASDAQ listing. This catches a whole class of
  mis-pairing that the check digit cannot.

Where the mappings come from
----------------------------
Three sources, in descending order of trust, because none of them alone is
enough:

1. **Learned from the user's own files** (``data/isin_learned.csv``). Most
   broker exports carry an ISIN column *and* a symbol column. When both are
   present the importer records the pairing, so a later ISIN-only export from
   the same broker resolves. This is the best source available: it is the
   user's own authoritative data, not anybody's guess.
2. **A small bundled seed** (``data/isin_seed.csv``). Deliberately small and
   restricted to widely-held names. It exists so the feature works on a fresh
   install, NOT as a claim to comprehensive coverage — there are tens of
   thousands of Indian ISINs alone and this file holds a few dozen.
3. **The provider**, via :func:`src.data.fetcher.search_symbols`, on explicit
   request. Covers the long tail; results are labelled as provider hits and
   can be learned once confirmed.

An ISIN that resolves through none of them is reported as unresolved with that
explanation. That is a worse user experience than a fabricated answer and a far
better product.

Pure module: stdlib plus pandas for the file round-trip. No network (the
provider path lives in the data layer, where every network call belongs) and no
Streamlit.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_DIR",
    "SEED_FILE",
    "LEARNED_FILE",
    "MAP_COLUMNS",
    "COUNTRY_PREFIXES",
    "IsinRecord",
    "normalise",
    "is_well_formed",
    "check_digit",
    "is_valid",
    "country_code",
    "country_name",
    "load_map",
    "refresh",
    "resolve",
    "learn",
    "pairs_from_frame",
    "map_summary",
]

#: ``<project root>/data`` -- this file is ``<root>/src/isin.py``.
DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"

#: Bundled reference pairings. Committed; small on purpose.
SEED_FILE = "isin_seed.csv"

#: Pairings learned from the user's own imports. NOT committed (it describes
#: what they hold), and safe to delete -- it is a cache, never a source of
#: truth the application depends on.
LEARNED_FILE = "isin_learned.csv"

#: Column contract of both files.
MAP_COLUMNS: tuple[str, ...] = ("isin", "yahoo_symbol", "name", "source")

#: ISO 6166 shape: 2-letter country prefix, 9 alphanumerics, 1 check digit.
_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

#: Characters people paste around an ISIN that carry no meaning.
_NOISE = re.compile(r"[\s\-_.]")

#: ISIN country prefix -> the ``country`` label the stock universe uses. Only
#: the prefixes this application actually has instruments for; an unknown
#: prefix yields ``None`` rather than a guess, and the cross-check is then
#: skipped rather than failed.
COUNTRY_PREFIXES: dict[str, str] = {
    "IN": "India",
    "US": "United States",
    "GB": "United Kingdom",
    "IE": "Ireland",
    "NL": "Netherlands",
    "CH": "Switzerland",
    "DE": "Germany",
    "FR": "France",
    "LU": "Luxembourg",
    "CA": "Canada",
    "JP": "Japan",
    "HK": "Hong Kong",
    "SG": "Singapore",
    "AU": "Australia",
    "KY": "Cayman Islands",
    "BM": "Bermuda",
    "TW": "Taiwan",
}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def normalise(text: object) -> str:
    """Upper-case and strip an ISIN of the punctuation people paste with it.

    ``" in e002a010 18 "`` -> ``"INE002A01018"``. Returns "" for missing input.
    Does NOT validate: use :func:`is_valid` for that, so a caller can report a
    malformed identifier rather than having it silently become "".
    """
    if text is None:
        return ""
    if isinstance(text, float) and not pd.notna(text):
        return ""
    return _NOISE.sub("", str(text).strip()).upper()


def _clean_symbol(value: object) -> str:
    """A Yahoo symbol from a spreadsheet cell, or "" when there is none.

    Written out rather than inlined as ``str(value or "")`` because that idiom
    is WRONG here and the bug it caused was subtle: ``float("nan")`` is truthy,
    so a blank cell survived the ``or``, stringified to ``"nan"`` and became
    the literal symbol ``"NAN"`` — which then got *learned* into the ISIN
    table as a permanent wrong mapping.
    """
    if value is None:
        return ""
    if isinstance(value, float) and not pd.notna(value):
        return ""
    out = str(value).strip().upper()
    return "" if out in {"NAN", "NONE", "NAT", "N/A", "NULL", ""} else out


def is_well_formed(text: object) -> bool:
    """True when ``text`` has the ISO 6166 SHAPE, check digit unverified."""
    return bool(_ISIN_PATTERN.match(normalise(text)))


def check_digit(body: object) -> str:
    """The ISO 6166 check digit for the first eleven characters of an ISIN.

    The algorithm, because it is easy to get subtly wrong and impossible to
    review without it written down:

    1. Expand every letter to its 0-based alphabet position plus 10 (``A`` ->
       ``10``, ``Z`` -> ``35``) and concatenate, so the body becomes a pure
       digit string.
    2. Apply Luhn over that string: counting from the RIGHTMOST digit as
       position 1, double every digit in an odd position, and add the digits of
       any product above 9.
    3. The check digit is ``(10 - sum % 10) % 10``.

    Worked example (Apple, ``US0378331005``): body ``US037833100`` expands to
    ``3028037833100``, the Luhn sum is 45, and the check digit is 5 — which is
    the last character of the real ISIN.

    Returns "" when ``body`` is not eleven valid characters, so a caller never
    compares against a fabricated digit.
    """
    text = normalise(body)
    if len(text) != 11 or not text.isalnum():
        return ""

    digits: list[int] = []
    for char in text:
        if char.isdigit():
            digits.append(int(char))
        elif char.isalpha():
            digits.extend(int(d) for d in str(ord(char) - ord("A") + 10))
        else:  # pragma: no cover - isalnum() already excluded this
            return ""

    total = 0
    for position, digit in enumerate(reversed(digits), start=1):
        if position % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    return str((10 - total % 10) % 10)


def is_valid(text: object) -> bool:
    """True when ``text`` is a structurally valid ISIN with a correct check digit.

    This is the whole reason a mistyped ISIN cannot reach the portfolio: the
    check digit makes a single-character error detectable by arithmetic, not by
    opinion.
    """
    isin = normalise(text)
    if not _ISIN_PATTERN.match(isin):
        return False
    expected = check_digit(isin[:11])
    return bool(expected) and expected == isin[11]


def country_code(text: object) -> str:
    """The two-letter ISO country prefix, or "" when the ISIN is malformed."""
    isin = normalise(text)
    return isin[:2] if _ISIN_PATTERN.match(isin) else ""


def country_name(text: object) -> str | None:
    """The country this ISIN was issued in, or ``None`` for an unknown prefix.

    ``None`` means "this module has no opinion", which is why the cross-check
    in :func:`resolve` skips rather than rejects on it. Prefixes also include
    the special ``XS`` (Euroclear/Clearstream) and CIS codes that name no
    single country at all.
    """
    return COUNTRY_PREFIXES.get(country_code(text))


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IsinRecord:
    """One ISIN -> Yahoo symbol pairing, with its provenance.

    ``source`` is load-bearing, not decoration: the UI states where a mapping
    came from, and ``learned`` (from the user's own broker file) is a stronger
    claim than ``seed`` (bundled) or ``provider`` (a search hit).
    """

    isin: str
    yahoo_symbol: str
    name: str = ""
    source: str = "seed"

    @property
    def country(self) -> str | None:
        return country_name(self.isin)

    @property
    def label(self) -> str:
        return f"{self.name or self.yahoo_symbol} ({self.yahoo_symbol})"


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def _read_file(path: Path, default_source: str) -> dict[str, IsinRecord]:
    """Parse one ISIN map file. Returns {} (and logs) on any problem.

    Rows whose ISIN fails validation are DROPPED with a warning rather than
    loaded: a mapping table with a bad check digit is corrupt, and honouring it
    would defeat the point of validating at all.
    """
    if not path.is_file():
        logger.debug("ISIN map file not present: %s", path)
        return {}

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
        logger.warning("Could not read ISIN map %s: %s", path, exc)
        return {}

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "isin" not in frame.columns or "yahoo_symbol" not in frame.columns:
        logger.warning("%s is missing isin/yahoo_symbol columns; skipped.", path.name)
        return {}

    out: dict[str, IsinRecord] = {}
    for record in frame.to_dict("records"):
        isin = normalise(record.get("isin"))
        symbol = _clean_symbol(record.get("yahoo_symbol"))
        if not isin or not symbol:
            continue
        if not is_valid(isin):
            logger.warning(
                "Dropping %s from %s: check digit is wrong, so the row is corrupt.",
                isin,
                path.name,
            )
            continue
        name = str(record.get("name") or "").strip()
        source = str(record.get("source") or "").strip() or default_source
        out[isin] = IsinRecord(isin=isin, yahoo_symbol=symbol, name=name, source=source)
    return out


def _from_universe(data_dir: str | None = None) -> dict[str, IsinRecord]:
    """ISIN pairings carried by the stock universe itself.

    The exchange listings publish an ISIN per instrument, so once the universe
    is generated from them (:mod:`src.universe_build`) it holds thousands of
    pairings -- 7,557 on the file this was written against, against 114 in the
    hand-curated seed. Reading them here is what turns ISIN-only broker
    imports from "works for the seeded names" into "works for the market".

    Lowest precedence of the three sources, deliberately. It is the broadest
    and the least reviewed: nobody has checked these by hand, and the file is
    regenerated by a build step. A curated seed row or a pairing learned from
    the user's own export still wins.

    Imported inside the function because :mod:`src.stock_universe` does not
    depend on this module and must not start to.
    """
    from src import stock_universe as su

    out: dict[str, IsinRecord] = {}
    for record in su.load_universe(data_dir):
        code = normalise(record.isin)
        if not code or not is_valid(code):
            continue
        # First listing wins, and load_universe orders NSE before BSE, so a
        # dual-listed company resolves to its NSE line -- the same India-first
        # rule EXCHANGE_PRIORITY applies to search.
        out.setdefault(
            code,
            IsinRecord(
                isin=code, yahoo_symbol=record.yahoo_symbol, name=record.name, source="universe"
            ),
        )
    return out


@lru_cache(maxsize=8)
def load_map(data_dir: str | None = None) -> dict[str, IsinRecord]:
    """Every known ISIN pairing, in increasing order of authority.

    Three sources, each overriding the one before:

    1. **the stock universe** -- thousands of pairings straight from the
       exchange listings, unreviewed,
    2. **the curated seed** -- hand-checked, every row validated by a test,
    3. **learned** -- taken from the user's own broker export.

    Learned wins deliberately. A pairing from the user's own file is
    authoritative for that user; a bundled seed row is this repository's best
    effort; a universe row is whatever the exchange published. If they
    disagree, the more specific source is right.

    Cached, since all three are static reference data. Call :func:`refresh`
    after :func:`learn` writes to disk.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    # The RESOLVED root, not the argument: `load_map(data_dir=X)` must read
    # every source from X. Passing the bare argument through meant a caller
    # that redirected only this module's DATA_DIR still picked up the global
    # universe file, so an isolated map was not isolated at all.
    merged = _from_universe(str(root))
    merged.update(_read_file(root / SEED_FILE, "seed"))
    merged.update(_read_file(root / LEARNED_FILE, "learned"))
    return merged


def refresh() -> None:
    """Drop the cached map so the next call re-reads both files."""
    load_map.cache_clear()


def resolve(
    text: object,
    data_dir: str | None = None,
    expected_country: str | None = None,
) -> IsinRecord | None:
    """The instrument an ISIN identifies, or ``None``.

    ``None`` covers three different situations, all of which the caller should
    report rather than paper over: the input is not an ISIN at all, the check
    digit is wrong (a typo), or the ISIN is valid but simply not in the table.

    ``expected_country`` adds the cross-check: pass the country of the symbol
    you are about to accept and a mapping that disagrees with the ISIN's own
    prefix is REFUSED. ``INE...`` resolving to a NASDAQ listing is a corrupt
    row, not a quirk, and this is what catches it. The check is skipped when
    either country is unknown, so it can never reject on ignorance.
    """
    isin = normalise(text)
    if not isin:
        return None
    if not is_valid(isin):
        return None

    record = load_map(data_dir).get(isin)
    if record is None:
        return None

    if expected_country:
        issued = country_name(isin)
        if issued and issued.strip().lower() != expected_country.strip().lower():
            logger.warning(
                "Refusing %s -> %s: the ISIN is issued in %s but the symbol is %s.",
                isin,
                record.yahoo_symbol,
                issued,
                expected_country,
            )
            return None
    return record


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------
def pairs_from_frame(
    frame: pd.DataFrame | None,
    isin_column: str,
    symbol_column: str,
) -> dict[str, str]:
    """Extract ISIN -> symbol pairs from a file that carries both columns.

    The best source of ISIN mappings there is, and it costs nothing: a broker
    export that lists ``ISIN`` alongside ``Tradingsymbol`` has already told us
    the pairing for every row. Only well-formed, check-digit-valid ISINs with a
    non-blank symbol are returned, and the first spelling of a repeated ISIN
    wins.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    if isin_column not in frame.columns or symbol_column not in frame.columns:
        return {}

    pairs: dict[str, str] = {}
    for isin_value, symbol_value in zip(frame[isin_column], frame[symbol_column]):
        isin = normalise(isin_value)
        symbol = _clean_symbol(symbol_value)
        if not isin or not symbol or not is_valid(isin):
            continue
        pairs.setdefault(isin, symbol)
    return pairs


def learn(
    pairs: Mapping[str, str] | Iterable[tuple[str, str]],
    source: str = "learned",
    names: Mapping[str, str] | None = None,
    data_dir: str | None = None,
    persist: bool = True,
) -> tuple[int, list[str]]:
    """Record ISIN -> symbol pairings. Returns ``(stored, rejected_messages)``.

    Every pair is validated before it is stored, and a rejection is REPORTED
    rather than dropped: silently discarding half a broker file's mappings
    would make the next ISIN-only import fail for no visible reason.

    Writes ``data/isin_learned.csv`` (created if absent, merged if present) and
    refreshes the cache. ``persist=False`` validates and returns without
    touching the disk, which is what the tests and a dry-run UI use.

    Never raises on a write failure — an unwritable cache directory degrades to
    "this session knows the mapping, the next one will not", which is a
    nuisance, not an error worth taking the import down for.
    """
    items = dict(pairs)
    names = names or {}

    accepted: dict[str, IsinRecord] = {}
    rejected: list[str] = []
    for raw_isin, raw_symbol in items.items():
        isin = normalise(raw_isin)
        symbol = _clean_symbol(raw_symbol)
        if not isin:
            rejected.append("blank ISIN skipped")
            continue
        if not symbol:
            rejected.append(f"{isin}: no symbol supplied")
            continue
        if not is_well_formed(isin):
            rejected.append(
                f"{isin}: not the shape of an ISIN (2 letters, 9 alphanumerics, 1 digit)"
            )
            continue
        if not is_valid(isin):
            rejected.append(
                f"{isin}: check digit is wrong — expected "
                f"{check_digit(isin[:11]) or '?'}, got {isin[11]}. Likely a typo."
            )
            continue
        accepted[isin] = IsinRecord(
            isin=isin,
            yahoo_symbol=symbol,
            name=str(names.get(isin, "") or "").strip(),
            source=source,
        )

    if not accepted or not persist:
        return len(accepted), rejected

    root = Path(data_dir) if data_dir else DATA_DIR
    path = root / LEARNED_FILE
    existing = _read_file(path, "learned")
    existing.update(accepted)

    try:
        root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "isin": r.isin,
                    "yahoo_symbol": r.yahoo_symbol,
                    "name": r.name,
                    "source": r.source,
                }
                for r in sorted(existing.values(), key=lambda r: r.isin)
            ],
            columns=list(MAP_COLUMNS),
        )
        frame.to_csv(path, index=False, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist learned ISIN mappings to %s: %s", path, exc)
        rejected.append(
            f"mappings could not be saved to {path.name} ({exc}); they apply to this session only."
        )
        return len(accepted), rejected

    refresh()
    return len(accepted), rejected


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def map_summary(data_dir: str | None = None) -> dict[str, int]:
    """Counts by ``source`` plus a ``total``, for the UI's coverage note.

    Quoting the real size is the point: a few dozen seeded mappings must not be
    presented as an ISIN directory.
    """
    records = load_map(data_dir)
    counts: dict[str, int] = {"total": len(records)}
    for record in records.values():
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts
