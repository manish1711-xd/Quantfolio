"""Build the stock-master CSVs from the official NSE and BSE files.

This is the *write* path for the data :mod:`src.stock_universe` reads. It runs
offline, on demand:

    python -m src.universe_build            # refresh data/indian_stocks.csv
    python -m src.universe_build --dry-run  # report what would change

Nothing in the running application calls it. The app reads committed CSVs, so
search stays instant, offline and deterministic — which is the whole reason the
universe is a file rather than a live query.

Where each field comes from, and why it takes two kinds of file
--------------------------------------------------------------
========================  ==========================================
``symbol`` / ``name``      the exchange **listing** (authoritative)
``isin``                   the listing
``sector``                 BSE's ``INDUSTRY``; curated for NSE
``turnover``               the **bhavcopy** (last trading day)
``aliases``                the curated file, always preserved
========================  ==========================================

The bhavcopy on its own cannot build a universe: its ``FinInstrmNm`` is a
squashed trading name and it carries gold bonds and government securities
alongside equities. The listings on their own rank badly, because they contain
no measure of size. Together they work.

Turnover is not decoration
--------------------------
A curated 243-row universe could rank by hand. A 7,700-row one cannot, and
without a prominence signal a full-market file makes search **worse**. Measured
on the real files:

* ``relaince`` (a typo) matched six companies whose name starts "Reliance" and
  returned **Reliance Communications** first, purely because ``RCOM.NS`` sorts
  before ``RELIANCE.NS`` alphabetically. Turnover that day: ₹0.3 cr versus
  ₹1,767 cr.
* ``tata`` returned ``TATACAP.NS`` (₹25 cr) ahead of ``TCS`` (₹592 cr) and
  ``TATAPOWER`` (₹439 cr), because the existing rule prefers a *shorter*
  symbol for a prefix match.

So the raw turnover is stored per row and :mod:`src.stock_universe` turns it
into a bounded ranking bonus. The CSV keeps the **measured number**, not a
derived score, so the ranking rule can change without re-downloading anything.

What this refuses to do
-----------------------
* **It never drops a curated row.** ``data/indian_stocks_curated.csv`` carries
  hand-written aliases ("RIL", "Reliance") and eleven NSE ETFs that appear in
  no equity listing. A wholesale replace would silently delete both, so the
  merge is additive and every curated alias survives.
* **It never writes to its own input.** The curated seed and the generated
  ``data/indian_stocks.csv`` are separate files. They were briefly the same
  file, which meant running the build twice replaced the pristine seed's
  backup with a generated one and lost the hand-written aliases irrecoverably.
  Reading one file and writing another makes the build idempotent and the
  curated data permanent.
* **It keeps the old name as an alias.** The exchange's spelling wins for
  ``name`` -- "Reliance Industries Limited" over "Reliance Industries Ltd" --
  but the previous name stays searchable. Dropping it removed a lookup key
  that worked, which broke CSV import for anyone whose broker used the old
  spelling.
* **It stops offering symbols that have stopped existing.** A curated row the
  listing no longer contains *and* that the bhavcopy shows no turnover for is
  excluded from the output and named in the report. Two independent sources
  have to agree before anything is dropped, and the curated seed is never
  edited -- the operator sees the list and decides. This is not tidiness: it
  caught ``TATAMOTORS.NS`` (demerged into ``TMCV``/``TMPV``), ``LTIM.NS``,
  ``AKZOINDIA.NS`` and ``GUJGASLTD.NS``, all four of which Yahoo also refuses
  to price. Offering a searchable symbol that cannot be priced is the same
  harm as inventing one. The eleven curated NSE ETFs are kept by exactly the
  same rule: no listing entry, but real bhavcopy turnover.
* **It never writes a truncated universe.** If a download fails the existing
  file is left exactly as it was. A refresh that half-worked is worse than one
  that did not run, because search would quietly stop finding things.
* **It backs the file up first.** This repository is not under version
  control, so an overwrite would otherwise be unrecoverable.
* **It records the date it really used.** ``data/universe_meta.json`` carries
  the bhavcopy date per exchange and the row counts, so the UI can say "as of
  2 Sep 2026" instead of implying the list is current.

Pure, apart from one entry point: every parse/merge function takes rows and
returns rows, so the whole build is testable against fixtures. Only
:func:`main` reaches the network, through :mod:`src.data.exchange`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger("quantfolio.universe")

__all__ = [
    "CSV_COLUMNS",
    "META_FILE",
    "INDIA_FILE",
    "CURATED_FILE",
    "NSE_SERIES",
    "BHAVCOPY_SERIES",
    "BuildReport",
    "parse_nse_listing",
    "parse_bse_listing",
    "parse_turnover",
    "load_curated",
    "merge",
    "write_universe",
    "build_meta",
    "main",
]

#: Column contract of the stock-master files. ``isin`` and ``turnover`` were
#: appended when the full listing landed; readers match by name, so an older
#: file without them still loads (they read as blank).
CSV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "yahoo_symbol",
    "name",
    "exchange",
    "country",
    "currency",
    "instrument_type",
    "sector",
    "aliases",
    "isin",
    "turnover",
)

#: Where the provenance of the generated files is recorded.
META_FILE = "universe_meta.json"

#: Generated output: the curated seed merged with the exchange listings.
INDIA_FILE = "indian_stocks.csv"

#: Hand-maintained input, never written by this module. Holds the aliases and
#: sectors no exchange file supplies, plus instruments (NSE ETFs) that appear
#: in no equity listing.
CURATED_FILE = "indian_stocks_curated.csv"

#: NSE series kept from the listing. ``EQ`` is the rolling-settlement main
#: board, ``BE`` trade-for-trade, ``BZ`` trade-for-trade under surveillance.
#: All three are real equities a user can hold, and all three price on Yahoo
#: (measured: 30/30 sampled symbols resolved). Anything else in a bhavcopy --
#: ``GS`` government securities, ``GB`` sovereign gold bonds, ``SM``/``ST`` SME
#: boards, ``N0`` debt -- is excluded because it is either not an equity or not
#: reliably priced.
NSE_SERIES: frozenset[str] = frozenset({"EQ", "BE", "BZ"})

#: Bhavcopy series/groups whose turnover counts towards prominence. NSE uses
#: series codes, BSE uses group letters, and one set covers both because they
#: do not collide.
BHAVCOPY_SERIES: frozenset[str] = frozenset(
    {"EQ", "BE", "BZ", "A", "B", "X", "XT", "T", "M", "MT", "Z", "P", "R"}
)

#: Legal-form suffixes stripped when deriving an alias from a company name, so
#: "Reliance Industries Limited" also answers to "Reliance Industries".
_NAME_TAIL: tuple[str, ...] = (
    " limited",
    " ltd.",
    " ltd",
    " private",
    " pvt",
    " corporation",
    " corp.",
    " corp",
    " company",
    " co.",
    " & co",
    " plc",
    " inc.",
    " inc",
)

_ALIAS_SEPARATOR = "|"


def _text(value: object) -> str:
    """Trimmed string, with the usual missing-value sentinels as ""."""
    if value is None:
        return ""
    out = str(value).strip()
    return "" if out.upper() in {"NAN", "NONE", "NAT", "N/A", "NULL", "-"} else out


def _float(value: object) -> float:
    try:
        out = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) and out > 0 else 0.0


def _int(value: object) -> int:
    """A non-negative row count, or 0. Never raises on a stored string."""
    try:
        return max(int(float(str(value).strip())), 0)
    except (TypeError, ValueError):
        return 0


def _short_name(name: str) -> str:
    """A company name with its legal-form tail removed.

    Used only to derive an alias. The full name is always kept as ``name``;
    this is an extra way in, not a replacement.
    """
    lowered = f" {name.strip().lower()}"
    for tail in _NAME_TAIL:
        if lowered.endswith(tail):
            return name.strip()[: len(name.strip()) - len(tail) + 1].strip(" ,-")
    return name.strip()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildReport:
    """What one merge did, in enough detail to review before writing."""

    rows: tuple[dict[str, str], ...] = ()
    curated_kept: int = 0
    curated_enriched: int = 0
    added: int = 0
    #: Curated rows excluded because two sources agree they no longer trade.
    stale: tuple[tuple[str, str], ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    with_turnover: int = 0
    with_isin: int = 0
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.rows)

    def describe(self) -> str:
        lines = [
            f"{self.total:,} instruments",
            f"  {self.curated_kept:,} curated rows kept "
            f"({self.curated_enriched:,} enriched from the exchange files)",
            f"  {self.added:,} added from the exchange listings",
            f"  {self.with_isin:,} carry an ISIN, {self.with_turnover:,} traded on the "
            "bhavcopy date",
        ]
        if self.stale:
            lines.append(
                f"  {len(self.stale):,} curated row(s) dropped - not in the listing "
                "and no turnover, so no longer traded:"
            )
            lines.extend(f"    {symbol}  ({why})" for symbol, why in self.stale)
        if self.skipped:
            reasons: dict[str, int] = {}
            for _symbol, reason in self.skipped:
                reasons[reason] = reasons.get(reason, 0) + 1
            lines.append(f"  {len(self.skipped):,} listing rows skipped:")
            lines.extend(
                f"    {count:,} x {reason}"
                for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
            )
        lines.extend(f"  ! {notice}" for notice in self.notices)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# parsing the sources (pure)
# ---------------------------------------------------------------------------
def parse_nse_listing(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """NSE ``EQUITY_L.csv`` rows -> universe rows. ``(records, skipped)``.

    Skips are returned rather than dropped: a listing row this build cannot
    use is a coverage gap the operator should be able to see, not a silent
    omission.
    """
    records: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for row in rows:
        symbol = _text(row.get("SYMBOL")).upper()
        if not symbol:
            skipped.append(("", "no symbol"))
            continue
        series = _text(row.get("SERIES")).upper()
        if series and series not in NSE_SERIES:
            skipped.append((symbol, f"NSE series {series} is not a priced equity series"))
            continue

        name = _text(row.get("NAME OF COMPANY"))
        aliases: list[str] = []
        short = _short_name(name)
        if short and short.lower() != name.lower():
            aliases.append(short)

        records.append(
            {
                "symbol": symbol,
                "yahoo_symbol": f"{symbol}.NS",
                "name": name or symbol,
                "exchange": "NSE",
                "country": "India",
                "currency": "INR",
                "instrument_type": "EQUITY",
                "sector": "",
                "aliases": _ALIAS_SEPARATOR.join(aliases),
                "isin": _text(row.get("ISIN NUMBER")).upper(),
                "turnover": "0",
            }
        )
    return records, skipped


def parse_bse_listing(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """BSE scrip-master rows -> universe rows. ``(records, skipped)``.

    ``scrip_id`` is used as the symbol, never ``SCRIP_CD``: Yahoo addresses a
    BSE line as ``RELIANCE.BO`` and returns nothing for ``500325.BO``. That was
    measured, and getting it wrong would have produced thousands of rows whose
    prices never load -- a universe worse than no universe.
    """
    records: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for row in rows:
        symbol = _text(row.get("scrip_id")).upper()
        code = _text(row.get("SCRIP_CD"))
        if not symbol:
            skipped.append((code, "no BSE trading symbol (scrip_id), so Yahoo cannot address it"))
            continue
        status = _text(row.get("Status")).lower()
        if status and status != "active":
            skipped.append((symbol, f"BSE status is {status}, not active"))
            continue

        name = _text(row.get("Issuer_Name")) or _text(row.get("Scrip_Name"))
        aliases: list[str] = []
        scrip_name = _text(row.get("Scrip_Name"))
        if scrip_name and scrip_name.lower() != name.lower():
            aliases.append(scrip_name)
        short = _short_name(name)
        if short and short.lower() not in {name.lower(), scrip_name.lower()}:
            aliases.append(short)

        records.append(
            {
                "symbol": symbol,
                "yahoo_symbol": f"{symbol}.BO",
                "name": name or symbol,
                "exchange": "BSE",
                "country": "India",
                "currency": "INR",
                "instrument_type": "EQUITY",
                "sector": _text(row.get("INDUSTRY")),
                "aliases": _ALIAS_SEPARATOR.join(dict.fromkeys(aliases)),
                "isin": _text(row.get("ISIN_NUMBER")).upper(),
                "turnover": "0",
            }
        )
    return records, skipped


def parse_turnover(
    rows: Iterable[Mapping[str, object]], series: Iterable[str] = BHAVCOPY_SERIES
) -> dict[str, float]:
    """UDiFF bhavcopy rows -> ``{symbol: turnover in rupees}``.

    ``TtlTrfVal`` is the day's traded value, which is the prominence signal.
    Only the equity series count: a sovereign gold bond's turnover says
    nothing about how prominent a *company* is, and including it would let a
    government security outrank a listed business in a company-name search.

    A symbol appearing twice (it can, across sessions) keeps the LARGER value
    rather than summing: this is a ranking signal, not an accounting figure,
    and the largest observation is the safest reading of "how much interest".
    """
    allowed = {str(s).upper() for s in series}
    out: dict[str, float] = {}
    for row in rows:
        symbol = _text(row.get("TckrSymb")).upper()
        if not symbol:
            continue
        if allowed and _text(row.get("SctySrs")).upper() not in allowed:
            continue
        value = _float(row.get("TtlTrfVal"))
        if value > out.get(symbol, 0.0):
            out[symbol] = value
    return out


def load_curated(path: Path) -> list[dict[str, str]]:
    """The existing stock-master file, or ``[]`` if there is none.

    Read with :data:`CSV_COLUMNS` defaults so a file written before ``isin``
    and ``turnover`` existed loads cleanly.
    """
    if not path.is_file():
        return []
    out: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = {column: _text(row.get(column)) for column in CSV_COLUMNS}
            if record["symbol"] or record["yahoo_symbol"]:
                out.append(record)
    return out


# ---------------------------------------------------------------------------
# the merge (pure)
# ---------------------------------------------------------------------------
def _merge_aliases(curated: str, fresh: str) -> str:
    """Curated aliases first, then any new ones, de-duplicated.

    Order matters for readability only, but curated-first states the priority:
    a hand-written "RIL" is worth more than a mechanically derived
    "Reliance Industries", and it must never be displaced by one.
    """
    seen: dict[str, str] = {}
    for blob in (curated, fresh):
        for alias in str(blob or "").split(_ALIAS_SEPARATOR):
            text = alias.strip()
            if text and text.lower() not in seen:
                seen[text.lower()] = text
    return _ALIAS_SEPARATOR.join(seen.values())


def merge(
    curated: Sequence[Mapping[str, str]],
    listings: Sequence[Mapping[str, str]],
    turnover: Mapping[str, Mapping[str, float]] | None = None,
    skipped: Sequence[tuple[str, str]] = (),
    notices: Sequence[str] = (),
    keep_stale: bool = False,
) -> BuildReport:
    """Combine the curated file with the exchange listings.

    Rules, in the order they matter:

    1. **Every curated row survives.** Keyed on ``yahoo_symbol``. The eleven
       bundled NSE ETFs appear in no equity listing, and the hand-written
       aliases exist nowhere else; losing either would be a regression
       disguised as an upgrade.
    2. **The exchange is authoritative for identity.** Where a curated row and
       a listing row are the same instrument, the listing supplies ``name``
       and ``isin`` -- it is the exchange's own spelling -- while the curated
       ``sector`` and ``aliases`` are kept, because the listings have no
       aliases and NSE's has no sector.
    3. **Turnover is attached per exchange**, so ``RELIANCE.NS`` and
       ``RELIANCE.BO`` are ranked on their own volumes rather than sharing one.
    4. **A curated row no longer traded is dropped.** Only when BOTH the
       listing has no entry for it and the bhavcopy shows no turnover -- two
       independent sources agreeing. That is what separates a renamed equity
       (``TATAMOTORS`` after the demerger: absent, no turnover, and Yahoo will
       not price it) from a curated ETF (absent from an *equity* listing by
       definition, but ₹210 cr traded). ``keep_stale=True`` overrides it.
    5. **Order is deterministic**: exchange, then symbol. The file is committed,
       so a stable order keeps a refresh's diff readable.
    """
    turnover = turnover or {}
    by_symbol: dict[str, dict[str, str]] = {}

    for row in curated:
        record = {column: _text(row.get(column)) for column in CSV_COLUMNS}
        key = record["yahoo_symbol"].upper()
        if key:
            by_symbol[key] = record

    curated_keys = set(by_symbol)
    enriched = 0
    added = 0

    for row in listings:
        record = {column: _text(row.get(column)) for column in CSV_COLUMNS}
        key = record["yahoo_symbol"].upper()
        if not key:
            continue

        existing = by_symbol.get(key)
        if existing is None:
            by_symbol[key] = record
            added += 1
            continue

        # A curated row, refreshed. Identity from the exchange; everything the
        # exchange does not know stays as curated.
        merged = dict(existing)
        merged["name"] = record["name"] or existing["name"]
        merged["isin"] = record["isin"] or existing["isin"]
        merged["sector"] = existing["sector"] or record["sector"]

        # The old name becomes an alias when the exchange spells it
        # differently. Without this, replacing "Reliance Industries Ltd" with
        # "Reliance Industries Limited" silently deleted a lookup key that
        # worked: a broker CSV naming the old spelling stopped resolving, and
        # -- because the BSE row happened to carry the old spelling as ITS
        # name -- the importer started rewriting NSE holdings to .BO symbols.
        superseded = existing["name"]
        keep = superseded if superseded and superseded != merged["name"] else ""
        merged["aliases"] = _merge_aliases(
            existing["aliases"],
            _ALIAS_SEPARATOR.join(x for x in (keep, record["aliases"]) if x),
        )
        merged["instrument_type"] = existing["instrument_type"] or record["instrument_type"]
        by_symbol[key] = merged
        enriched += 1

    # Turnover last, so it lands on curated and added rows alike.
    with_turnover = 0
    for record in by_symbol.values():
        per_exchange = turnover.get(record["exchange"].upper(), {})
        value = _float(per_exchange.get(record["symbol"].upper(), 0.0))
        record["turnover"] = str(int(value))
        if value > 0:
            with_turnover += 1

    listed = {_text(row.get("yahoo_symbol")).upper() for row in listings}
    stale: list[tuple[str, str]] = []
    if not keep_stale and turnover:
        # Only meaningful when turnover was actually loaded. Without it every
        # row reads as "no turnover" and this rule would delete the universe.
        for key in list(by_symbol):
            record = by_symbol[key]
            if key in listed or _float(record["turnover"]) > 0:
                continue
            if key not in curated_keys:
                continue
            stale.append(
                (
                    record["yahoo_symbol"],
                    f"absent from the {record['exchange'] or 'exchange'} listing and "
                    "no turnover on the bhavcopy date",
                )
            )
            del by_symbol[key]

    rows = sorted(
        by_symbol.values(),
        key=lambda r: (r["exchange"].upper(), r["symbol"].upper()),
    )
    return BuildReport(
        rows=tuple(rows),
        curated_kept=len(curated_keys) - len(stale),
        curated_enriched=enriched,
        added=added,
        stale=tuple(stale),
        skipped=tuple(skipped),
        with_turnover=with_turnover,
        with_isin=sum(1 for r in rows if r["isin"]),
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_universe(path: Path, rows: Sequence[Mapping[str, str]], backup: bool = True) -> str:
    """Write the master CSV atomically. Returns the backup path, or "".

    Two safety properties, both of which have to hold on a machine with no
    version control:

    * **A backup first.** The file being replaced may carry hand-written
       aliases that exist nowhere else, and this repository is not a git
       checkout, so an overwrite would otherwise be unrecoverable.
    * **Atomic replace.** Written to a temporary file in the same directory
       and moved into place, so an interrupted write cannot leave a
       half-truncated universe that the app would happily load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    backup_path = ""
    if backup and path.is_file():
        target = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, target)
        backup_path = str(target)

    # mkstemp in the TARGET directory, so os.replace is a same-filesystem
    # rename and therefore atomic. A temp file elsewhere would make it a copy,
    # which can be interrupted half-written.
    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _text(row.get(column)) for column in CSV_COLUMNS})
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return backup_path


def build_meta(
    sources: Sequence[Mapping[str, object]],
    report: BuildReport,
    existing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Provenance for ``universe_meta.json``.

    Exists so the UI can state how stale the list is. A universe presented
    without its date implies it is current, which for an end-of-day file is
    never quite true.
    """
    meta: dict[str, object] = dict(existing or {})
    meta["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["rows"] = report.total
    meta["with_isin"] = report.with_isin
    meta["with_turnover"] = report.with_turnover
    meta["sources"] = [
        {
            "source": _text(item.get("source")),
            "url": _text(item.get("url")),
            "rows": _int(item.get("rows")),
            "as_of": _text(item.get("as_of")),
        }
        for item in sources
    ]
    dates = [_text(item.get("as_of")) for item in sources if _text(item.get("as_of"))]
    meta["as_of"] = min(dates) if dates else ""
    return meta


# ---------------------------------------------------------------------------
# entry point (the only part that touches the network)
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.universe_build",
        description=(
            "Rebuild the Indian stock-master file from the official NSE and "
            "BSE listings, ranked by bhavcopy turnover. Run it when you want "
            "the list refreshed; the application itself never does."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Where the stock-master CSVs live.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change and write nothing.",
    )
    parser.add_argument(
        "--on",
        default="",
        metavar="YYYY-MM-DD",
        help="Bhavcopy date to start walking back from. Defaults to today.",
    )
    parser.add_argument(
        "--no-bhavcopy",
        action="store_true",
        help=(
            "Skip the bhavcopies. The listings alone still build a complete "
            "universe, but with no turnover the search ranking falls back to "
            "alphabetical within a tier -- which is measurably worse."
        ),
    )
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help=(
            "Keep curated rows the exchange no longer lists and that show no "
            "turnover. They will be searchable but may not price."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep a .bak of the file being replaced. Rarely a good idea.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log each download.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch, merge and write. Returns an exit code and never raises.

    Exit codes: ``0`` written (or a clean dry run), ``1`` nothing was written
    because a source failed, ``2`` could not start.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):  # exotic stream
                reconfigure(encoding="utf-8", errors="replace")

    # Deferred: this module's parse/merge API is importable and testable with
    # no network stack loaded at all.
    from src.data import exchange as ex

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"No data directory at {data_dir}.", file=sys.stderr)
        return 2

    on: date | None = None
    if args.on:
        try:
            on = date.fromisoformat(args.on)
        except ValueError:
            print(f"{args.on!r} is not an ISO date (try 2026-09-02).", file=sys.stderr)
            return 2

    listings: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []
    notices: list[str] = []
    sources: list[dict[str, object]] = []

    for label, fetch, parse in (
        ("NSE", ex.fetch_nse_listing, parse_nse_listing),
        ("BSE", ex.fetch_bse_listing, parse_bse_listing),
    ):
        result = fetch()
        print(f"  {result.summary}")
        if not result.ok:
            # A failed listing means an incomplete universe, and writing one
            # would silently remove instruments the user can currently find.
            print(
                f"The {label} listing could not be fetched, so nothing was written "
                "and the existing universe is untouched.",
                file=sys.stderr,
            )
            return 1
        records, dropped = parse(result.rows)
        listings.extend(records)
        skipped.extend(dropped)
        sources.append(
            {"source": result.source, "url": result.url, "rows": len(result.rows), "as_of": ""}
        )

    turnover: dict[str, dict[str, float]] = {}
    if args.no_bhavcopy:
        notices.append(
            "Built without bhavcopy turnover (--no-bhavcopy), so search ranks "
            "alphabetically within a match tier."
        )
    else:
        for label in ("NSE", "BSE"):
            attempts: list[str] = []
            result = ex.latest_bhavcopy(label, on=on, attempts=attempts)
            print(f"  {result.summary}")
            if args.verbose:
                for line in attempts:
                    print(f"     {line}")
            if not result.ok:
                # Not fatal: the universe is still complete, only its ranking
                # is weaker. Said out loud rather than left to be discovered.
                notices.append(f"{label} turnover unavailable - {result.notice}")
                continue
            turnover[label] = parse_turnover(result.rows)
            sources.append(
                {
                    "source": result.source,
                    "url": result.url,
                    "rows": len(result.rows),
                    "as_of": result.as_of.isoformat() if result.as_of else "",
                }
            )

    target = data_dir / INDIA_FILE
    seed_path = data_dir / CURATED_FILE
    if not seed_path.is_file():
        # First run after the split: adopt the existing output as the seed, so
        # a hand-edited file is not lost by the upgrade that introduced the
        # separation.
        seed = load_curated(target)
        if seed:
            notices.append(
                f"{CURATED_FILE} did not exist, so {INDIA_FILE} was used as the "
                "curated seed for this run. It has now been written out; edit "
                "that file for hand-maintained aliases from here on."
            )
            write_universe(seed_path, seed, backup=False)
    else:
        seed = load_curated(seed_path)

    report = merge(seed, listings, turnover, skipped, notices, keep_stale=args.keep_stale)
    print()
    print(report.describe())

    if args.dry_run:
        print(f"\nDry run: {target} was not modified.")
        return 0

    backup = write_universe(target, report.rows, backup=not args.no_backup)
    meta_path = data_dir / META_FILE
    existing_meta: dict[str, object] = {}
    if meta_path.is_file():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing_meta = {}
    meta_path.write_text(
        json.dumps(build_meta(sources, report, existing_meta), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {target} ({report.total:,} rows) and {meta_path}.")
    if backup:
        print(f"Previous file kept at {backup}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
