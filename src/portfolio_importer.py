"""Broker-agnostic CSV/Excel import for holdings and trade ledgers.

The problem
-----------
No two brokers name their columns the same way. Zerodha exports
``Symbol, Quantity Available, Average Price``; Groww writes
``Stock Name, Quantity, Average buy price``; Upstox uses
``Company Name, Instrument, Qty, Avg. Cost``; a spreadsheet a human made says
``Ticker, Shares, Cost``. The previous import path called ``pd.read_csv`` and
handed the frame straight to validation, so anything but the exact canonical
header silently produced "Missing column(s): ticker, quantity, buy_price" and
the user had to rename columns by hand.

What this module does
---------------------
1. **Detects** which source column means what, through an alias table plus a
   token-overlap fallback for headers nobody anticipated.
2. **Normalises** everything onto the canonical schema the rest of the
   application already speaks — ``ticker, quantity, buy_price, purchase_date``
   for holdings and ``ticker, trade_date, action, quantity, price, fees,
   currency, note`` for ledgers.
3. **Resolves** tickers through :mod:`src.stock_universe`, so a broker's bare
   ``RELIANCE`` becomes ``RELIANCE.NS``.
4. **Reports** everything it did and everything it could not do, per row, as an
   :class:`ImportPreview` the UI renders BEFORE anything is committed.

Non-negotiable: **no row is ever silently discarded.** Every rejected row comes
back in :attr:`ImportPreview.rejected` carrying the reason it was rejected, and
every coercion that changed a value is recorded as a :class:`RowIssue`. An
importer that quietly drops two of your twelve holdings is worse than one that
refuses the file, because the resulting portfolio still looks plausible.

Pure module: pandas only, no Streamlit, no network. The universe lookups it
performs read bundled local CSVs.
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from src import isin as isin_mod
from src import stock_universe as universe
from src.config import DEFAULT_BASE_CURRENCY, PREFERRED_EXCHANGE

logger = logging.getLogger(__name__)

__all__ = [
    "ImportKind",
    "HOLDING_COLUMNS",
    "HOLDING_REQUIRED",
    "TRANSACTION_COLUMNS",
    "TRANSACTION_REQUIRED",
    "COLUMN_ALIASES",
    "ColumnMapping",
    "RowIssue",
    "ImportPreview",
    "read_table",
    "MAX_PREAMBLE_LINES",
    "detect_columns",
    "build_preview",
    "NOT_PRESENT",
    "apply_overrides",
    "remap",
    "redetect",
    "import_holdings",
    "import_transactions",
]

ImportKind = Literal["holdings", "transactions"]
Severity = Literal["error", "warning", "info"]

# ---------------------------------------------------------------------------
# canonical schemas
# ---------------------------------------------------------------------------
#: Canonical holdings schema. ``purchase_date`` is normalised to the name the
#: rest of the app already uses (``buy_date``) on the way out of
#: :meth:`ImportPreview.to_holdings`, so both vocabularies work.
HOLDING_COLUMNS: tuple[str, ...] = (
    "ticker",
    "quantity",
    "buy_price",
    "purchase_date",
    "isin",
)
HOLDING_REQUIRED: tuple[str, ...] = ("ticker", "quantity", "buy_price")

TRANSACTION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "action",
    "quantity",
    "price",
    "fees",
    "currency",
    "note",
    "isin",
)
TRANSACTION_REQUIRED: tuple[str, ...] = ("ticker", "trade_date", "action", "quantity", "price")

# ---------------------------------------------------------------------------
# alias table
# ---------------------------------------------------------------------------
#: Canonical field -> source header aliases, in PRIORITY order. Aliases are
#: compared after normalisation (lower-case, punctuation to underscore), so
#: "Avg. Price", "avg price" and "AVG_PRICE" all reduce to ``avg_price``.
#:
#: Order matters where headers overlap. A holdings export can legitimately
#: contain both ``average_price`` (what you paid) and ``price`` (what it is
#: worth now); listing the explicit cost aliases first means ``buy_price``
#: claims ``average_price`` before the ambiguous bare ``price`` is considered.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ticker": (
        "ticker",
        "symbol",
        "tradingsymbol",
        "trading_symbol",
        "stock",
        "security",
        "instrument",
        "scrip",
        "scrip_code",
        "scrip_name",
        "stock_symbol",
        "stock_name",
        "company",
        "company_name",
        "name",
        "yahoo_symbol",
        "nse_symbol",
        "isin",
    ),
    "quantity": (
        "quantity",
        "qty",
        "shares",
        "units",
        "no_of_shares",
        "number_of_shares",
        "quantity_available",
        "available_quantity",
        "holding_quantity",
        "net_quantity",
        "closing_quantity",
        "share_count",
    ),
    "buy_price": (
        "buy_price",
        "buy_average",
        "average_buy_price",
        "avg_buy_price",
        "average_price",
        "avg_price",
        "average_cost",
        "avg_cost",
        "cost_price",
        "purchase_price",
        "buy_rate",
        "buy_avg",
        "avg_trading_price",
        "average_trading_price",
        "cost_per_share",
        "cost_basis_per_share",
        "price_per_unit",
        "rate",
        "cost",
        "price",
    ),
    "purchase_date": (
        "purchase_date",
        "buy_date",
        "first_bought",
        "acquisition_date",
        "date_of_purchase",
        "date",
        "transaction_date",
        "trade_date",
    ),
    "trade_date": (
        "trade_date",
        "transaction_date",
        "date",
        "order_date",
        "execution_date",
        "trade_dt",
        "settlement_date",
        "value_date",
        "purchase_date",
    ),
    "action": (
        "action",
        "trade_type",
        "transaction_type",
        "type",
        "side",
        "buy_sell",
        "buy_or_sell",
        "order_type",
        "txn_type",
    ),
    "price": (
        "price",
        "trade_price",
        "execution_price",
        "rate",
        "unit_price",
        "price_per_share",
        "average_price",
        "avg_price",
        "buy_price",
    ),
    "fees": (
        "fees",
        "fee",
        "brokerage",
        "charges",
        "total_charges",
        "commission",
        "transaction_charges",
        "taxes_and_charges",
        "stt",
        "cost_of_trade",
    ),
    "isin": (
        "isin",
        "isin_code",
        "isin_number",
        "isin_no",
        "security_isin",
        "instrument_isin",
        "scrip_isin",
    ),
    "currency": ("currency", "ccy", "trade_currency", "quote_currency"),
    "note": ("note", "notes", "remark", "remarks", "comment", "comments", "description"),
}

#: Headers that must NEVER be taken as a canonical field, whatever they look
#: like. A holdings export's "Previous Closing Price" and "LTP" are current
#: market data, not what the user paid, and mapping either onto ``buy_price``
#: would silently report a 0% return on the whole portfolio.
BLOCKED_HEADERS: frozenset[str] = frozenset(
    {
        "ltp",
        "last_price",
        "last_traded_price",
        "market_price",
        "current_price",
        "closing_price",
        "close_price",
        "previous_closing_price",
        "prev_close",
        "current_value",
        "market_value",
        "pnl",
        "p_l",
        "profit_loss",
        "unrealized_p_l",
        "unrealised_p_l",
        "unrealized_pnl",
        "day_change",
        "net_chg",
        "net_change",
        "day_p_l",
    }
)

#: Recognised trade verbs -> the canonical action. Everything else is reported.
ACTION_WORDS: dict[str, str] = {
    "BUY": "BUY",
    "B": "BUY",
    "BOUGHT": "BUY",
    "PURCHASE": "BUY",
    "PURCHASED": "BUY",
    "CREDIT": "BUY",
    "DEBIT": "BUY",
    "SELL": "SELL",
    "S": "SELL",
    "SOLD": "SELL",
    "SALE": "SELL",
    "SELL_ORDER": "SELL",
}

_HEADER_PUNCT = re.compile(r"[^a-z0-9]+")
#: Everything that can decorate a number in a broker export: thousands
#: separators, currency symbols and codes, stray unicode spaces.
_NUMBER_NOISE = re.compile(r"[,\s  ₹$€£¥]|(?:^|(?<=\d))(?:INR|USD|EUR|GBP|JPY)\b", re.I)
#: Headers that mean "this column holds ISINs". Used as a SEMANTIC signal,
#: separate from the shape test: a column called "ISIN" is treated as ISINs
#: even for a row whose check digit is wrong (so the typo is reported as a
#: typo), while a column called "Symbol" is only read as an ISIN when the
#: value validates -- because "NOTANISIN123" happens to have ISIN shape, and
#: mis-reading a real ticker as an identifier would be worse than either.
_ISIN_HEADERS: frozenset[str] = frozenset(
    {
        "isin",
        "isin_code",
        "isin_number",
        "isin_no",
        "security_isin",
        "instrument_isin",
        "scrip_isin",
    }
)

#: Date formats tried before falling back to pandas' own parser. Day-first is
#: listed first because Indian broker exports are overwhelmingly DD-MM-YYYY,
#: and 03/04/2024 is genuinely ambiguous — getting it backwards silently moves
#: a trade by a month and corrupts every holding-period and XIRR figure.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
)


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------
def _norm_header(text: object) -> str:
    """``"Average Buy Price "`` -> ``"average_buy_price"``."""
    return _HEADER_PUNCT.sub("_", str(text or "").strip().lower()).strip("_")


def _header_tokens(header: str) -> frozenset[str]:
    return frozenset(t for t in header.split("_") if t)


def _clean_number(value: object) -> float | None:
    """Parse a broker-formatted number. ``None`` when it is not one.

    Handles ``"1,234.56"``, ``"₹ 1,234.56"``, ``"1 234,56"``-style spacing,
    and accounting negatives ``"(120.50)"``. Returns ``None`` — never 0.0 —
    for blanks and junk, because a missing quantity and a zero quantity mean
    completely different things and only one of them is a valid holding.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None

    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NAT", "N/A", "NA", "-", "--", ""}:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = _NUMBER_NOISE.sub("", text)
    text = text.rstrip("%")
    if not text or text in {"-", "+", "."}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return -out if negative else out


def _clean_date(value: object) -> pd.Timestamp | None:
    """Parse a date, trying day-first formats before pandas' inference.

    Returns ``None`` when unparseable rather than ``NaT``, so callers get a
    plain truthiness check and cannot accidentally propagate ``NaT`` into a
    date arithmetic that would come back as a silent NaN holding period.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.normalize()

    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NAT", "N/A", "NA", "-"}:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(pd.to_datetime(text, format=fmt)).normalize()
        except (ValueError, TypeError):
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    except Exception:  # noqa: BLE001 - exotic input types
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _clean_text(value: object) -> str:
    """Trimmed string; "" for blanks and every NaN sentinel."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in {"NAN", "NONE", "NAT", "N/A"} else text


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMapping:
    """Which source column was read as which canonical field.

    ``mapping`` is canonical -> source header (the original spelling, so the
    UI can show the user their own column names back). The two annotation
    tuples record *how* each field got there, which is what lets the UI be
    honest about its own confidence:

    * ``fuzzy`` — matched by token overlap rather than a known alias. A guess.
    * ``overridden`` — set by hand by the user through the import preview. Not
      a guess at all, and therefore never also listed in ``fuzzy``.
    """

    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    fuzzy: tuple[str, ...] = ()
    overridden: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing

    def source_of(self, canonical: str) -> str | None:
        return self.mapping.get(canonical)

    def confidence_of(self, canonical: str) -> str:
        """How this field was mapped, as a word for the preview table."""
        if canonical in self.overridden:
            return "you chose this"
        if canonical in self.fuzzy:
            return "needs review"
        return "matched"

    def blocked(self) -> tuple[str, ...]:
        """Overridden fields pointed at a market-price / P&L column.

        Automatic detection can never claim a :data:`BLOCKED_HEADERS` column,
        but an explicit human override can — someone may genuinely have a file
        whose "LTP" column really does hold their cost. That decision is
        theirs to make; this exists so the UI can warn instead of either
        silently obeying or silently refusing.
        """
        return tuple(
            canonical
            for canonical in self.overridden
            if _norm_header(self.mapping.get(canonical, "")) in BLOCKED_HEADERS
        )

    def as_frame(self) -> pd.DataFrame:
        """Two-column table for display: source header -> canonical field."""
        rows = [
            {
                "Your column": source,
                "Read as": canonical,
                "Confidence": self.confidence_of(canonical),
            }
            for canonical, source in self.mapping.items()
        ]
        rows.extend(
            {"Your column": source, "Read as": "— ignored —", "Confidence": ""}
            for source in self.unmapped
        )
        return pd.DataFrame(rows, columns=["Your column", "Read as", "Confidence"])


@dataclass(frozen=True)
class RowIssue:
    """One thing that happened to one row, with enough detail to act on it.

    ``row`` is the 1-based line number in the user's file counting the header,
    which is what their spreadsheet shows — reporting a 0-based frame index
    would make them count rows by hand.
    """

    row: int
    column: str
    value: str
    reason: str
    severity: Severity = "error"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"Row {self.row} · {self.column}: {self.reason}"


@dataclass(frozen=True)
class ImportPreview:
    """Everything one import attempt produced, before anything is committed.

    ``frame`` holds only the rows that survived; ``rejected`` holds the rest
    WITH a ``_reason`` column. The two together always account for every row in
    the source file — :meth:`accounted_for` asserts exactly that, and it is
    the invariant that makes "nothing was silently dropped" checkable rather
    than merely claimed.

    ``source_frame`` is the raw parsed file, carried along so :func:`remap` can
    re-derive the whole preview under a corrected mapping without re-reading
    (and re-uploading) anything. It is present even on the fatal
    missing-column path — which is exactly the path where the user most needs
    to point a field at a different column.
    """

    kind: ImportKind
    mapping: ColumnMapping
    frame: pd.DataFrame
    rejected: pd.DataFrame
    issues: tuple[RowIssue, ...] = ()
    resolved: dict[str, str] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_rows: int = 0
    source_columns: tuple[str, ...] = ()
    fatal: str = ""
    source_frame: pd.DataFrame | None = None
    #: ISIN -> Yahoo symbol for rows whose identifier was an ISIN.
    isin_resolved: dict[str, str] = field(default_factory=dict)
    #: Valid ISINs that no table could turn into a symbol. These rows are
    #: rejected, because an unresolvable identifier cannot be priced -- but
    #: they are listed separately from ordinary bad rows since the fix is
    #: different: look the ISIN up, rather than correct a number.
    isin_unresolved: tuple[str, ...] = ()
    #: How many ISIN -> symbol pairings this file taught the local table.
    isin_learned: int = 0

    # -- status ----------------------------------------------------------
    @property
    def ok(self) -> bool:
        """True when there is at least one usable row and no fatal error."""
        return not self.fatal and not self.frame.empty

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_rejected(self) -> int:
        return int(len(self.rejected))

    @property
    def n_tickers(self) -> int:
        if self.frame.empty or "ticker" not in self.frame.columns:
            return 0
        return int(self.frame["ticker"].nunique())

    @property
    def accounted_for(self) -> bool:
        """Every source row is either imported or explicitly rejected.

        The invariant that makes "nothing is silently discarded" testable
        rather than merely asserted in a docstring. A whole-FILE refusal (a
        required column missing, an unreadable upload) counts as accounted
        for: ``fatal`` explains every row at once, and no row was processed
        individually.
        """
        if self.fatal and self.n_rows == 0 and self.n_rejected == 0:
            return True
        return self.n_rows + self.n_rejected == self.source_rows

    @property
    def needs_review(self) -> tuple[str, ...]:
        """Symbols the local universe could not confirm or could not disambiguate.

        These are NOT errors. A symbol absent from the bundled files may price
        perfectly well; only the data layer can settle that. The UI shows them
        as "needs review" so the user can eyeball a typo before analysing.
        """
        return tuple(dict.fromkeys([*self.unresolved, *self.ambiguous]))

    def errors(self) -> tuple[RowIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    def warnings(self) -> tuple[RowIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    # -- reporting -------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """Checklist rendered above the preview table.

        Uses ✓ / ⚠ / ✗ rather than prose so a 40-row broker file's outcome is
        readable in one glance.
        """
        # Override notes belong on the fatal path too: a user who has just
        # re-pointed a column needs to see that their choice was taken even
        # when the file is still not importable for some other reason.
        override_lines: list[str] = []
        if self.mapping.overridden:
            override_lines.append(
                f"✓ column(s) {', '.join(self.mapping.overridden)} mapped by you, not by detection"
            )
        for canonical in self.mapping.blocked():
            override_lines.append(
                f"⚠ you mapped {canonical} to '{self.mapping.mapping[canonical]}', which "
                "normally holds CURRENT market data rather than what you paid. If that is "
                "wrong, every return figure will be near zero."
            )

        # An unresolvable ISIN explains WHY rows failed, so it belongs on the
        # fatal path too -- that is precisely the case where "No row could be
        # imported" on its own tells the user nothing actionable. Held in its
        # own list so that on the SUCCESS path it can sit with the other ✗
        # lines instead of interrupting the ✓ ones.
        isin_problems: list[str] = []
        if self.isin_unresolved:
            count = len(self.isin_unresolved)
            isin_problems.append(
                f"✗ {count} ISIN{'s' if count != 1 else ''} could not be mapped to a "
                "symbol — the rows are listed below"
            )

        if self.fatal:
            return [*override_lines, *isin_problems, f"✗ {self.fatal}"]

        noun = "holding" if self.kind == "holdings" else "transaction"
        lines = [f"✓ {self.n_rows} {noun}{'s' if self.n_rows != 1 else ''} recognised"]
        lines.extend(override_lines)

        confirmed = len([t for t in self.resolved.values() if t])
        if confirmed:
            lines.append(f"✓ {confirmed} ticker{'s' if confirmed != 1 else ''} matched locally")
        if self.isin_resolved:
            count = len(self.isin_resolved)
            lines.append(
                f"✓ {count} ISIN{'s' if count != 1 else ''} resolved to a tradeable symbol"
            )
        if self.isin_learned:
            lines.append(
                f"✓ {self.isin_learned} ISIN → symbol pairing(s) learned from this file "
                "for future imports"
            )

        if self.ambiguous:
            lines.append(
                f"⚠ {len(self.ambiguous)} ticker(s) list on more than one exchange — "
                f"defaulted to {PREFERRED_EXCHANGE}, confirm below"
            )
        if self.unresolved:
            lines.append(
                f"⚠ {len(self.unresolved)} ticker(s) not in the bundled stock-master "
                "files — they may still price; confirm the spelling"
            )
        lines.extend(isin_problems)
        if self.n_rejected:
            lines.append(f"✗ {self.n_rejected} row(s) cannot be imported — listed below")
        if self.mapping.fuzzy:
            lines.append(
                f"⚠ column(s) {', '.join(self.mapping.fuzzy)} matched by similarity, "
                "not by a known name — check the mapping"
            )
        if self.mapping.missing:
            lines.append(f"✗ required column(s) not found: {', '.join(self.mapping.missing)}")
        return lines

    # -- handoff ---------------------------------------------------------
    def to_holdings(self) -> pd.DataFrame:
        """Holdings frame in the shape the session grid and store expect.

        Renames ``purchase_date`` to ``buy_date``: that is the optional column
        :func:`src.analytics.positions.clean_holdings` already understands, and
        inventing a second name for the same thing would mean the date silently
        stopped being carried through.
        """
        if self.kind != "holdings" or self.frame.empty:
            return pd.DataFrame(columns=["ticker", "quantity", "buy_price"])
        out = self.frame.copy()
        if "purchase_date" in out.columns:
            dates = pd.to_datetime(out["purchase_date"], errors="coerce")
            out = out.drop(columns=["purchase_date"])
            if dates.notna().any():
                out["buy_date"] = dates.dt.strftime("%Y-%m-%d").where(dates.notna(), "")
        out = _drop_empty_isin(out)
        return out.reset_index(drop=True)

    def to_transactions(self) -> pd.DataFrame:
        """Ledger frame in the shape :func:`src.analytics.lots.parse_transactions` reads."""
        if self.kind != "transactions" or self.frame.empty:
            return pd.DataFrame(columns=list(TRANSACTION_COLUMNS))
        out = self.frame.copy()
        if "trade_date" in out.columns:
            dates = pd.to_datetime(out["trade_date"], errors="coerce")
            out["trade_date"] = dates.dt.strftime("%Y-%m-%d").where(dates.notna(), "")
        out = _drop_empty_isin(out)
        return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
#: Lines of preamble tolerated above the real header. Broker exports often
#: open with an account-details block ("Account Statement", "Client: XX1234").
MAX_PREAMBLE_LINES: int = 6

#: CSV parse attempts, in order. ``sep=None`` with the python engine sniffs the
#: delimiter, which is what makes semicolon-separated European exports work.
_READ_ATTEMPTS: tuple[dict, ...] = (
    {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
    {"encoding": "latin-1", "sep": None, "engine": "python"},
)


def _source_bytes(source: object) -> tuple[bytes | None, str]:
    """Read an upload/path/bytes into raw bytes once. ``(bytes, error)``.

    Reading once matters: an uploaded file object is a stream, and the header
    detection below needs to re-parse the same content several times.
    """
    if isinstance(source, bytes):
        return source, ""
    if hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        return (data if isinstance(data, bytes) else str(data).encode("utf-8")), ""
    # A filesystem path, as a str or any os.PathLike (pathlib.Path included).
    # Accepting only str here was a real gap: every caller inside the repo --
    # tests, scripts, the sample loader -- naturally hands over a Path.
    if isinstance(source, (str, os.PathLike)):
        try:
            return Path(source).read_bytes(), ""
        except OSError as exc:
            return None, str(exc)
    return None, "unsupported source type"


def read_table(source: object, filename: str = "", skiprows: int = 0) -> tuple[pd.DataFrame, str]:
    """Read a CSV (or Excel, if the optional engine is installed) into a frame.

    Returns ``(frame, error)``. ``error`` is a human-readable sentence and the
    frame is empty whenever it is non-empty; this never raises, because a bad
    upload is a normal user event and must not reach the error boundary.

    Handles what real exports throw at it: a UTF-8 BOM, Latin-1 encoding from
    Excel on Windows, and sniffed delimiters (comma, semicolon, tab).
    ``skiprows`` drops leading preamble lines -- callers that know the expected
    schema use :func:`_read_with_header_detection` to find the right value
    rather than guessing here.
    """
    name = str(filename or getattr(source, "name", "") or "uploaded file")

    if name.lower().endswith((".xlsx", ".xls")):
        try:
            frame = pd.read_excel(source, skiprows=skiprows or 0)
        except ImportError:
            return (
                pd.DataFrame(),
                f"{name} is an Excel workbook and no Excel reader is installed. "
                "Save it as CSV, or install openpyxl.",
            )
        except Exception as exc:  # noqa: BLE001 - any parse failure
            return pd.DataFrame(), f"Could not read {name}: {exc}"
        return frame, ""

    raw, read_error = _source_bytes(source)
    if raw is None:
        return pd.DataFrame(), f"Could not read {name}: {read_error}"
    if not raw.strip():
        return pd.DataFrame(), f"{name} is empty."

    last_error = ""
    for options in _READ_ATTEMPTS:
        try:
            frame = pd.read_csv(io.BytesIO(raw), skiprows=skiprows or 0, **options)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        return frame, ""

    return pd.DataFrame(), f"Could not read {name} as CSV ({last_error or 'unknown format'})."


def _read_with_header_detection(
    source: object, filename: str, kind: ImportKind
) -> tuple[pd.DataFrame, str]:
    """Read the file, finding the real header row if it is not the first line.

    Detection is by OUTCOME, not by shape. An earlier version retried only
    when the first parse produced a single column, which missed the common
    case entirely: the delimiter sniffer splits a prose preamble line like
    ``"Account Statement"`` on whitespace into two columns, so the shape test
    never fired and the whole file was refused for a missing ticker column.

    So instead: parse, ask :func:`detect_columns` whether the required fields
    are all there, and if not, try skipping one more line. The first offset
    whose mapping is complete wins; if none is, the offset-0 result is returned
    so the error the user sees describes their actual header rather than some
    arbitrary row four lines down.
    """
    # Resolve the display/extension name BEFORE the source is consumed into
    # bytes: the retries below hand read_table a bare BytesIO, which carries no
    # .name, so an Excel path with no explicit filename would otherwise lose its
    # extension and be parsed as CSV.
    name = str(filename or getattr(source, "name", "") or "")
    if not name and isinstance(source, (str, os.PathLike)):
        name = Path(source).name

    if name.lower().endswith((".xlsx", ".xls")):
        # Excel needs the original source (pandas reads the workbook itself),
        # and a spreadsheet's header row is reliably the first one.
        return read_table(source, name)

    raw, read_error = _source_bytes(source)
    if raw is None:
        return pd.DataFrame(), f"Could not read {name or 'the file'}: {read_error}"

    first: tuple[pd.DataFrame, str] | None = None
    for skip in range(MAX_PREAMBLE_LINES):
        frame, error = read_table(io.BytesIO(raw), name, skiprows=skip)
        if first is None:
            first = (frame, error)
        if error or frame.empty:
            break
        if detect_columns(frame, kind).ok:
            return frame, ""
    return first if first is not None else (pd.DataFrame(), "")


# ---------------------------------------------------------------------------
# column detection
# ---------------------------------------------------------------------------
def _wanted_fields(kind: ImportKind) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if kind == "transactions":
        return TRANSACTION_COLUMNS, TRANSACTION_REQUIRED
    return HOLDING_COLUMNS, HOLDING_REQUIRED


def detect_columns(frame: pd.DataFrame | None, kind: ImportKind = "holdings") -> ColumnMapping:
    """Work out which source column is which canonical field.

    Two passes, and the order between them is the whole point:

    1. **Alias pass** — every canonical field, in schema order, claims the
       best-priority source header it recognises. Fields are processed in
       schema order and a claimed header is removed from the pool, so
       ``quantity`` cannot steal the header ``buy_price`` wanted.
    2. **Token pass** — canonical fields still unmatched try token overlap
       against the leftovers (``Purchase Rate/Share`` -> ``buy_price``). These
       are flagged in :attr:`ColumnMapping.fuzzy` because a similarity match is
       a guess, and a guess about which column is your cost basis is exactly
       the kind of thing a user must see before importing.

    Headers in :data:`BLOCKED_HEADERS` are never claimed by either pass.
    """
    wanted, required = _wanted_fields(kind)
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return ColumnMapping(missing=required)

    # normalised header -> original header, first spelling wins on duplicates
    pool: dict[str, str] = {}
    for column in frame.columns:
        key = _norm_header(column)
        if key and key not in pool:
            pool[key] = str(column)

    mapping: dict[str, str] = {}
    fuzzy: list[str] = []
    available = dict(pool)

    # -- pass 1: known aliases -------------------------------------------
    for canonical in wanted:
        for alias in COLUMN_ALIASES.get(canonical, ()):
            if alias in BLOCKED_HEADERS:
                continue
            if alias in available:
                mapping[canonical] = available.pop(alias)
                break

    # -- pass 2: token overlap for what is still missing ------------------
    for canonical in wanted:
        if canonical in mapping:
            continue
        target_tokens = _header_tokens(canonical)
        alias_tokens = [_header_tokens(a) for a in COLUMN_ALIASES.get(canonical, ())]
        best_key, best_overlap = "", 0.0
        for key in available:
            if key in BLOCKED_HEADERS:
                continue
            tokens = _header_tokens(key)
            if not tokens:
                continue
            scores = [
                len(tokens & candidate) / len(tokens | candidate)
                for candidate in (target_tokens, *alias_tokens)
                if candidate
            ]
            overlap = max(scores, default=0.0)
            if overlap > best_overlap:
                best_key, best_overlap = key, overlap
        # 0.5 Jaccard means at least half the tokens agree; below that the
        # "match" is coincidence and a wrong guess is worse than no guess.
        if best_key and best_overlap >= 0.5:
            mapping[canonical] = available.pop(best_key)
            fuzzy.append(canonical)

    return ColumnMapping(
        mapping=mapping,
        unmapped=tuple(available.values()),
        missing=tuple(c for c in required if c not in mapping),
        fuzzy=tuple(fuzzy),
    )


# ---------------------------------------------------------------------------
# ticker resolution
# ---------------------------------------------------------------------------
def _infer_preferred_exchange(raw_tickers: list[str]) -> str | None:
    """Guess the file's home market from the symbols that are unambiguous.

    A Zerodha export is entirely NSE symbols, so its bare ``INFY`` row means
    ``INFY.NS`` — not the NYSE ADR that owns that exact Yahoo symbol. Rather
    than hard-code that, this counts the rows that resolve to exactly ONE
    venue and takes the majority. Returns ``None`` when there is no clear
    majority, in which case ambiguous symbols are reported for review instead
    of being decided by a coin flip.
    """
    votes: dict[str, int] = {}
    for raw in raw_tickers:
        if raw.upper().endswith((".NS", ".BO")):
            votes["NSE"] = votes.get("NSE", 0) + 1
            continue
        candidates = universe.candidates_for(raw)
        if len(candidates) == 1:
            exchange = candidates[0].exchange.upper()
            if exchange:
                votes[exchange] = votes.get(exchange, 0) + 1
    if not votes:
        return None
    top, count = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
    return top if count * 2 > sum(votes.values()) else None


# ---------------------------------------------------------------------------
# preview construction
# ---------------------------------------------------------------------------
def _drop_empty_isin(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``isin`` column when no row carries one.

    Kept when populated, because an ISIN is genuinely useful provenance to
    carry into a saved portfolio. Dropped when empty, so a file that never
    mentioned ISINs does not gain a blank column in the holdings grid.
    """
    if "isin" not in frame.columns:
        return frame
    values = frame["isin"].astype("string").fillna("")
    if not (values.str.strip() != "").any():
        return frame.drop(columns=["isin"])
    return frame


def _empty_frame(kind: ImportKind) -> pd.DataFrame:
    columns = TRANSACTION_COLUMNS if kind == "transactions" else HOLDING_COLUMNS
    return pd.DataFrame(columns=list(columns))


def build_preview(
    frame: pd.DataFrame | None,
    kind: ImportKind = "holdings",
    resolve_tickers: bool = True,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    mapping: ColumnMapping | None = None,
    learn_isins: bool = True,
) -> ImportPreview:
    """Normalise a raw frame onto the canonical schema and report everything.

    ``mapping`` may be supplied to override detection — that is how the UI
    offers "read this column as buy_price instead" without re-detecting.

    Validation applied per row, each producing a :class:`RowIssue` and, when
    fatal to that row, a line in ``rejected``:

    * ticker must be present after cleaning
    * an ISIN in the ticker column is rejected explicitly (it identifies the
      security but is not a Yahoo symbol, and there is no offline ISIN table
      here to map it — saying so beats importing an unpriceable row)
    * quantity and price must parse and be strictly positive
    * ``action`` must be a recognised trade verb (ledgers only)
    * an unparseable date is a WARNING for holdings (the date is optional) and
      an ERROR for a ledger (FIFO ordering is meaningless without it)
    """
    columns = tuple(str(c) for c in (frame.columns if frame is not None else ()))
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return ImportPreview(
            kind=kind,
            mapping=ColumnMapping(missing=_wanted_fields(kind)[1]),
            frame=_empty_frame(kind),
            rejected=pd.DataFrame(),
            source_columns=columns,
            fatal="The file contains no data rows.",
        )

    detected = mapping or detect_columns(frame, kind)
    if not detected.ok:
        return ImportPreview(
            kind=kind,
            mapping=detected,
            frame=_empty_frame(kind),
            rejected=pd.DataFrame(),
            source_rows=int(len(frame)),
            source_columns=columns,
            source_frame=frame,
            fatal=(
                "Could not find a column for: "
                + ", ".join(detected.missing)
                + ". Pick the right column below, or rename it in your file."
            ),
        )

    ticker_source = detected.mapping["ticker"]
    raw_tickers = [_clean_text(v).upper() for v in frame[ticker_source].tolist()]
    preferred = (
        _infer_preferred_exchange(raw_tickers) or PREFERRED_EXCHANGE if resolve_tickers else None
    )

    # Is the ticker column itself an ISIN column? Two different questions get
    # asked below: "was this column *labelled* ISIN" (semantic) and "does this
    # value validate as one" (arithmetic). See _ISIN_HEADERS.
    isin_source = detected.mapping.get("isin")
    ticker_is_isin_column = _norm_header(ticker_source) in _ISIN_HEADERS

    # A file carrying an ISIN column AND a symbol column has already told us
    # every pairing it contains. Recording them costs nothing and is the best
    # source of ISIN mappings there is -- the user's own broker data -- so a
    # later ISIN-only export from the same broker resolves.
    isin_learned = 0
    if learn_isins and isin_source and not ticker_is_isin_column:
        discovered = isin_mod.pairs_from_frame(frame, isin_source, ticker_source)
        resolved_pairs = {
            code: universe.normalise_symbol(symbol, prefer_exchange=preferred)
            for code, symbol in discovered.items()
        }
        if resolved_pairs:
            isin_learned, _problems = isin_mod.learn(resolved_pairs)

    kept: list[dict] = []
    dropped: list[dict] = []
    issues: list[RowIssue] = []
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    ambiguous: dict[str, tuple[str, ...]] = {}
    isin_resolved: dict[str, str] = {}
    isin_unresolved: list[str] = []

    for offset, record in enumerate(frame.to_dict("records")):
        line = offset + 2  # +1 for 0-based, +1 for the header row
        raw_ticker = _clean_text(record.get(ticker_source)).upper()

        # Loop variables are bound as defaults, not captured: a closure that
        # reads `line` from the enclosing scope would see the LAST row's value
        # if it were ever called after the loop moved on, which is the classic
        # late-binding bug and would attach every rejection reason to the
        # wrong row number.
        def reject(
            reason: str,
            column: str = "ticker",
            value: str = "",
            *,
            _line: int = line,
            _ticker: str = raw_ticker,
            _record: dict = record,
        ) -> None:
            issues.append(RowIssue(_line, column, value or _ticker, reason, "error"))
            dropped.append({**_record, "_row": _line, "_reason": reason})

        # A blank symbol is still recoverable when the row carries an ISIN.
        if not raw_ticker and isin_source:
            raw_ticker = isin_mod.normalise(record.get(isin_source))
        if not raw_ticker:
            reject("no ticker/symbol in this row")
            continue

        # -- ISIN -> symbol ----------------------------------------------
        row_isin = ""
        if isin_source:
            row_isin = isin_mod.normalise(record.get(isin_source))
        looks_like_isin = isin_mod.is_well_formed(raw_ticker)
        if looks_like_isin and (ticker_is_isin_column or isin_mod.is_valid(raw_ticker)):
            code = isin_mod.normalise(raw_ticker)
            row_isin = row_isin or code
            if not isin_mod.is_valid(code):
                expected = isin_mod.check_digit(code[:11]) or "?"
                reject(
                    f"{code} is not a valid ISIN: the check digit should be "
                    f"{expected}, not {code[11]}. That is almost always a typo or a "
                    "truncated copy-paste.",
                    "isin",
                )
                continue
            match = isin_mod.resolve(code)
            if match is None:
                isin_unresolved.append(code)
                issued = isin_mod.country_name(code)
                reject(
                    f"{code} is a valid ISIN"
                    + (f" issued in {issued}" if issued else "")
                    + " but is not in the local ISIN table, so it cannot be turned "
                    "into a tradeable symbol offline. Include a symbol column in the "
                    "export, or look the ISIN up online from the import preview.",
                    "isin",
                )
                continue
            raw_ticker = match.yahoo_symbol
            isin_resolved[code] = match.yahoo_symbol

        # -- symbol ------------------------------------------------------
        # Symbol findings are held row-locally and committed only if the row
        # survives. A row rejected for a missing quantity must not also be
        # reported as "ticker needs review": the user would go hunting for a
        # symbol problem that does not exist.
        symbol = raw_ticker
        row_resolved: tuple[str, str] | None = None
        row_unresolved = ""
        row_ambiguous: tuple[str, tuple[str, ...]] | None = None

        if resolve_tickers:
            candidates = universe.candidates_for(raw_ticker)
            if candidates:
                match = universe.resolve(raw_ticker, prefer_exchange=preferred)
                if match is not None:
                    symbol = match.yahoo_symbol
                    row_resolved = (raw_ticker, symbol)
                    if len({c.exchange for c in candidates}) > 1:
                        venues = tuple(c.yahoo_symbol for c in candidates)
                        row_ambiguous = (raw_ticker, venues)
                        issues.append(
                            RowIssue(
                                line,
                                "ticker",
                                raw_ticker,
                                f"lists on several exchanges ({', '.join(venues)}); "
                                f"read as {symbol}",
                                "warning",
                            )
                        )
            else:
                row_unresolved = raw_ticker

        # -- numbers -----------------------------------------------------
        row: dict[str, object] = {"ticker": symbol}
        numeric_ok = True
        price_field = "price" if kind == "transactions" else "buy_price"
        for canonical in ("quantity", price_field):
            source = detected.mapping.get(canonical)
            value = _clean_number(record.get(source)) if source else None
            if value is None:
                reject(
                    f"{canonical.replace('_', ' ')} is missing or not a number "
                    f"({_clean_text(record.get(source))!r})",
                    canonical,
                    raw_ticker,
                )
                numeric_ok = False
                break
            if value <= 0:
                reject(
                    f"{canonical.replace('_', ' ')} must be greater than zero (got {value:g})",
                    canonical,
                    raw_ticker,
                )
                numeric_ok = False
                break
            row[canonical] = float(value)
        if not numeric_ok:
            continue

        # -- action (ledgers only) ---------------------------------------
        if kind == "transactions":
            raw_action = _clean_text(record.get(detected.mapping.get("action"))).upper()
            key = _norm_header(raw_action).upper()
            action = ACTION_WORDS.get(raw_action) or ACTION_WORDS.get(key)
            if action is None:
                reject(
                    f"unrecognised trade type {raw_action!r} — expected BUY or SELL",
                    "action",
                    raw_ticker,
                )
                continue
            row["action"] = action

        # -- dates -------------------------------------------------------
        date_field = "trade_date" if kind == "transactions" else "purchase_date"
        date_source = detected.mapping.get(date_field)
        parsed_date = _clean_date(record.get(date_source)) if date_source else None
        if date_source and parsed_date is None:
            raw_value = _clean_text(record.get(date_source))
            if kind == "transactions":
                reject(
                    f"trade date {raw_value!r} could not be parsed — FIFO lot matching "
                    "needs a real date",
                    "trade_date",
                    raw_ticker,
                )
                continue
            if raw_value:
                issues.append(
                    RowIssue(
                        line,
                        "purchase_date",
                        raw_value,
                        "date could not be parsed and was left blank (holdings do not require it)",
                        "warning",
                    )
                )
        row[date_field] = parsed_date

        # -- optional ledger extras --------------------------------------
        if kind == "transactions":
            fees = _clean_number(record.get(detected.mapping.get("fees")))
            if fees is not None and fees < 0:
                issues.append(
                    RowIssue(
                        line,
                        "fees",
                        f"{fees:g}",
                        "negative fees read as a positive cost (fees are always a cost)",
                        "warning",
                    )
                )
                fees = abs(fees)
            row["fees"] = float(fees or 0.0)
            # A currency is only ever asserted when the FILE speaks to it.
            # With no currency column at all this stays blank, and
            # src.analytics.lots.parse_transactions falls back to the
            # Transaction default -- because stamping the reporting currency
            # on a US broker's dollar trades would label them "INR", and a
            # wrong label, unlike a missing one, is not visibly wrong.
            # When the file DOES have the column, a blank cell in it means
            # "the usual currency", so base_currency fills that gap.
            currency_source = detected.mapping.get("currency")
            row["currency"] = (
                (_clean_text(record.get(currency_source)).upper() or base_currency)
                if currency_source
                else ""
            )
            row["note"] = _clean_text(record.get(detected.mapping.get("note")))

        row["isin"] = row_isin
        row["_row"] = line
        kept.append(row)

        # The row survived — now its symbol findings are worth reporting.
        if row_resolved is not None:
            resolved[row_resolved[0]] = row_resolved[1]
        if row_ambiguous is not None:
            ambiguous[row_ambiguous[0]] = row_ambiguous[1]
        if row_unresolved and row_unresolved not in unresolved:
            unresolved.append(row_unresolved)

    schema, _ = _wanted_fields(kind)
    clean = pd.DataFrame(kept)[[*schema]] if kept else _empty_frame(kind)
    for numeric in ("quantity", "buy_price", "price", "fees"):
        if numeric in clean.columns:
            clean[numeric] = pd.to_numeric(clean[numeric], errors="coerce").astype("float64")
    # Dates must be a real datetime64 column, not an object column mixing
    # Timestamps with None: the latter makes .dt unavailable and silently
    # turns every downstream date comparison into an object comparison.
    for date_column in ("purchase_date", "trade_date"):
        if date_column in clean.columns:
            clean[date_column] = pd.to_datetime(clean[date_column], errors="coerce")

    rejected = pd.DataFrame(dropped) if dropped else pd.DataFrame()

    return ImportPreview(
        kind=kind,
        mapping=detected,
        frame=clean.reset_index(drop=True),
        rejected=rejected,
        issues=tuple(issues),
        resolved=resolved,
        unresolved=tuple(unresolved),
        ambiguous=ambiguous,
        source_rows=int(len(frame)),
        source_columns=columns,
        source_frame=frame,
        isin_resolved=isin_resolved,
        isin_unresolved=tuple(dict.fromkeys(isin_unresolved)),
        isin_learned=isin_learned,
        fatal="" if kept else "No row in the file could be imported — see the reasons below.",
    )


# ---------------------------------------------------------------------------
# manual column overrides
# ---------------------------------------------------------------------------
#: Sentinel meaning "this field is not present in the file". Distinct from
#: ``None`` so a caller can tell "leave this field alone" (absent from the
#: overrides dict) from "unmap it" (present, with this value or ``None``).
NOT_PRESENT: str = "__not_present__"


def _unmapped_columns(source_columns: Sequence[str], claimed: dict[str, str]) -> tuple[str, ...]:
    """Source columns no canonical field is reading, in file order."""
    taken = set(claimed.values())
    return tuple(str(c) for c in source_columns if str(c) not in taken)


def apply_overrides(
    mapping: ColumnMapping,
    overrides: dict[str, str | None],
    source_columns: Sequence[str],
    kind: ImportKind = "holdings",
) -> ColumnMapping:
    """Return ``mapping`` with the named canonical fields re-pointed.

    ``overrides`` is canonical field -> source column. A value of ``None``,
    ``""`` or :data:`NOT_PRESENT` unmaps the field instead. Fields absent from
    the dict are left exactly as they were, so a caller only has to send what
    actually changed.

    Three rules, each there for a reason:

    * **One column, one field.** Pointing ``buy_price`` at a column another
      field currently holds takes it away from that field, rather than letting
      two canonical names read the same data. The displaced field then shows up
      in ``missing`` if it was required, which is visible rather than subtle.
    * **A named column that is not in the file unmaps the field.** It does not
      raise and it does not keep the old value: you asked for a column that
      does not exist, so there is no column for that field. That surfaces
      loudly through ``missing`` instead of quietly ignoring your instruction.
    * **An overridden field leaves ``fuzzy``.** A similarity guess the user has
      since confirmed by hand is not a guess any more, and continuing to flag
      it "needs review" would train them to ignore the flag.

    :data:`BLOCKED_HEADERS` is deliberately NOT enforced here. It exists to stop
    *automatic* detection reading a market price as a cost basis; a human who
    explicitly picks that column may know something the alias table does not.
    :meth:`ColumnMapping.blocked` reports the choice so the UI can warn, which
    respects the decision without hiding the risk.
    """
    wanted, required = _wanted_fields(kind)
    known = {str(c) for c in source_columns}
    claimed = dict(mapping.mapping)

    # Only instructions that were actually HONOURED land in ``overridden``.
    # The distinction that matters: deliberately setting a field to "not
    # present" IS the user's decision and worth recording, whereas naming a
    # column the file does not have is an instruction that could not be
    # carried out -- and labelling that "you chose this" would credit the user
    # with a mapping they do not have.
    honoured: list[str] = []

    for canonical, source in overrides.items():
        if canonical not in wanted:
            logger.warning(
                "Ignoring override for %r: not a %s field (expected one of %s).",
                canonical,
                kind,
                ", ".join(wanted),
            )
            continue

        chosen = "" if source is None else str(source).strip()
        if not chosen or chosen == NOT_PRESENT:
            claimed.pop(canonical, None)
            honoured.append(canonical)
            continue
        if chosen not in known:
            logger.warning(
                "Override for %r names column %r, which the file does not have; "
                "treating the field as absent.",
                canonical,
                chosen,
            )
            claimed.pop(canonical, None)
            continue

        for other, existing in list(claimed.items()):
            if other != canonical and existing == chosen:
                claimed.pop(other)
        claimed[canonical] = chosen
        honoured.append(canonical)

    # A field displaced by a reassignment loses its override record along with
    # its column: it is now unmapped through no choice of the user's.
    overridden = tuple(
        f for f in dict.fromkeys([*mapping.overridden, *honoured]) if f in claimed or f in honoured
    )
    return ColumnMapping(
        mapping=claimed,
        unmapped=_unmapped_columns(source_columns, claimed),
        missing=tuple(c for c in required if c not in claimed),
        fuzzy=tuple(f for f in mapping.fuzzy if f not in set(overridden)),
        overridden=overridden,
    )


def remap(
    preview: ImportPreview,
    overrides: dict[str, str | None],
    resolve_tickers: bool = True,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    learn_isins: bool = True,
) -> ImportPreview:
    """Rebuild ``preview`` with some canonical fields pointed at other columns.

    This is what makes a failed import recoverable in place. Detection covers
    the brokers we know about and a token-similarity fallback for the rest, but
    a header nobody anticipated ("Cost of acquisition per unit") still ends as
    ``✗ Could not find a column for: buy_price`` — and until now that was a
    dead end that sent the user back to their spreadsheet to rename a column.

    Re-derives the whole preview from :attr:`ImportPreview.source_frame`, so
    every per-row validation, ticker resolution and rejection reason is
    recomputed under the corrected mapping. Nothing is re-read from disk and
    nothing is re-uploaded.

    Returns the preview UNCHANGED when there is nothing to re-derive (no source
    frame, or an empty ``overrides``), so a no-op apply is genuinely a no-op.
    """
    if not overrides:
        return preview
    frame = preview.source_frame
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        logger.warning("Cannot remap a preview that carries no source frame.")
        return preview

    mapping = apply_overrides(preview.mapping, overrides, preview.source_columns, preview.kind)
    return build_preview(frame, preview.kind, resolve_tickers, base_currency, mapping, learn_isins)


def redetect(
    preview: ImportPreview,
    resolve_tickers: bool = True,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    learn_isins: bool = True,
) -> ImportPreview:
    """Rebuild ``preview`` from automatic detection, discarding every override.

    The escape hatch from a manual mapping the user has talked themselves into.
    """
    frame = preview.source_frame
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return preview
    return build_preview(frame, preview.kind, resolve_tickers, base_currency, None, learn_isins)


# ---------------------------------------------------------------------------
# top-level entry points
# ---------------------------------------------------------------------------
def import_holdings(
    source: object,
    filename: str = "",
    resolve_tickers: bool = True,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    learn_isins: bool = True,
) -> ImportPreview:
    """Read a holdings file and return its preview. Never raises."""
    frame, error = _read_with_header_detection(source, filename, "holdings")
    if error:
        return ImportPreview(
            kind="holdings",
            mapping=ColumnMapping(missing=HOLDING_REQUIRED),
            frame=_empty_frame("holdings"),
            rejected=pd.DataFrame(),
            fatal=error,
        )
    return build_preview(frame, "holdings", resolve_tickers, base_currency, None, learn_isins)


def import_transactions(
    source: object,
    filename: str = "",
    resolve_tickers: bool = True,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    learn_isins: bool = True,
) -> ImportPreview:
    """Read a trade-ledger file and return its preview. Never raises."""
    frame, error = _read_with_header_detection(source, filename, "transactions")
    if error:
        return ImportPreview(
            kind="transactions",
            mapping=ColumnMapping(missing=TRANSACTION_REQUIRED),
            frame=_empty_frame("transactions"),
            rejected=pd.DataFrame(),
            fatal=error,
        )
    return build_preview(frame, "transactions", resolve_tickers, base_currency, None, learn_isins)
