"""Local persistence for saved portfolios (stdlib ``sqlite3`` only).

Why SQLite and not a pile of CSVs: a portfolio is a small relational object
(one portfolio, many holdings, many transactions) and the interesting
operation - "delete this portfolio and everything under it" - is exactly what
a foreign key with ``ON DELETE CASCADE`` is for. One file, no server, no
dependency, and the whole thing is still `sqlite3 quantfolio.db` inspectable.

Schema changes
--------------
``CREATE TABLE IF NOT EXISTS`` creates a table but never alters one, so a
column added to an existing installation would silently not exist. Every
additive column therefore also appears in :data:`_MIGRATIONS`, which
``_ensure_schema`` applies with ``ALTER TABLE ... ADD COLUMN`` after checking
``PRAGMA table_info``. That is idempotent, works on a database created by any
earlier version, and needs no version counter -- the table's own columns are
the version.

Conventions
-----------
* Timestamps (``created_at`` / ``updated_at``) are ISO-8601 UTC strings.
* ``trade_date`` is stored as an ISO calendar date ("2024-03-17") and read
  back as a string; ``pd.to_datetime`` in the analytics layer parses it.
* Quantities and prices are stored as REAL in the instrument's own currency;
  the store performs NO valuation, NO FX and NO validation of trading logic -
  that is the analytics layer's job. It only refuses rows it cannot represent
  (blank ticker, unparseable number, unparseable date) and says nothing else.
* Concurrency: Streamlit re-runs a script on whichever thread it likes, so the
  connection is opened with ``check_same_thread=False`` and every statement is
  serialised through a re-entrant lock. Writes are wrapped in a transaction,
  so a failed save leaves the previous version of the portfolio intact.
* Every statement is parameterised. No user value is ever interpolated into
  SQL text. The only interpolation anywhere in this module builds the
  schedule column list out of the module constant ``SCHEDULE_COLUMNS``,
  precomputed once below into ``_SCHEDULE_SQL_COLUMNS``; every value still
  travels as a bound parameter.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DB_FILENAME, DEFAULT_BASE_CURRENCY

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DB_PATH",
    "HOLDINGS_COLUMNS",
    "TRANSACTIONS_COLUMNS",
    "WATCHLIST_COLUMNS",
    "SCHEDULE_COLUMNS",
    "ALERT_SETTINGS_COLUMNS",
    "PortfolioStore",
]

#: ``<project root>/data/quantfolio.db`` - this file is ``<root>/src/data/store.py``.
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[2] / "data" / DB_FILENAME

#: Column contract of the frames returned by :meth:`PortfolioStore.load_portfolio`.
HOLDINGS_COLUMNS: tuple[str, ...] = ("ticker", "quantity", "buy_price")
#: Column contract of :meth:`PortfolioStore.load_watchlist`. Mirrors
#: :data:`src.watchlist.WATCHLIST_COLUMNS`; a test asserts they do not drift.
#: The last five carry the price-alert rule and its state and were added
#: later, hence the migration below.
WATCHLIST_COLUMNS: tuple[str, ...] = (
    "ticker",
    "note",
    "target_price",
    "added_at",
    "direction",
    "alert_enabled",
    "alert_state",
    "last_fired",
    "reference_price",
)

#: Column contract of :meth:`PortfolioStore.load_alert_settings`. One row,
#: holding who to notify and how hard a rule is to re-trigger -- and, as with
#: `schedules`, no credential of any kind.
ALERT_SETTINGS_COLUMNS: tuple[str, ...] = (
    "enabled",
    "recipients",
    "rearm_pct",
    "cooldown_hours",
    "use_session_range",
    "timezone",
)
#: Column contract of :meth:`PortfolioStore.load_schedules`. Mirrors
#: :data:`src.reporting.schedule.SCHEDULE_FIELDS`; a test asserts the two do
#: not drift apart.
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "name",
    "portfolio",
    "recipients",
    "frequency",
    "hour",
    "minute",
    "weekday",
    "day_of_month",
    "timezone",
    "enabled",
    "last_sent",
    "period",
    "benchmark",
    "base_currency",
    "attach_excel",
)

#: Precomputed from the constants above so no SQL string is ever assembled
#: from anything a user supplied. Values are always bound parameters.
_SCHEDULE_SQL_COLUMNS = ", ".join(SCHEDULE_COLUMNS)
_SCHEDULE_SQL_PLACEHOLDERS = ", ".join("?" for _ in SCHEDULE_COLUMNS)
_WATCHLIST_SQL_COLUMNS = ", ".join(WATCHLIST_COLUMNS)
_WATCHLIST_SQL_PLACEHOLDERS = ", ".join("?" for _ in WATCHLIST_COLUMNS)
_ALERT_SQL_COLUMNS = ", ".join(ALERT_SETTINGS_COLUMNS)
_ALERT_SQL_PLACEHOLDERS = ", ".join("?" for _ in ALERT_SETTINGS_COLUMNS)
TRANSACTIONS_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "action",
    "quantity",
    "price",
    "fees",
    "currency",
    "note",
)

#: How long SQLite waits for another writer before giving up (milliseconds).
#: Only relevant when a second process has the file open; the in-process lock
#: already serialises this instance's own statements.
_BUSY_TIMEOUT_MS = 5000

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS portfolios (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL UNIQUE,
        base_currency TEXT    NOT NULL,
        created_at    TEXT    NOT NULL,
        updated_at    TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS holdings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL
                     REFERENCES portfolios(id) ON DELETE CASCADE,
        ticker       TEXT    NOT NULL,
        quantity     REAL    NOT NULL,
        buy_price    REAL    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL
                     REFERENCES portfolios(id) ON DELETE CASCADE,
        ticker       TEXT    NOT NULL,
        trade_date   TEXT    NOT NULL,
        action       TEXT    NOT NULL,
        quantity     REAL    NOT NULL,
        price        REAL    NOT NULL,
        fees         REAL    NOT NULL DEFAULT 0,
        currency     TEXT    NOT NULL,
        note         TEXT    NOT NULL DEFAULT ''
    )
    """,
    # The watchlist is deliberately NOT scoped to a portfolio. A watched name
    # has no quantity and no cost basis -- it is not a holding of anything --
    # and in practice people keep one list of "things I am considering" across
    # every portfolio they model. Hence a flat table with a UNIQUE ticker
    # rather than a foreign key onto portfolios.
    """
    CREATE TABLE IF NOT EXISTS watchlist (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT    NOT NULL UNIQUE,
        note            TEXT    NOT NULL DEFAULT '',
        target_price    REAL,
        added_at        TEXT    NOT NULL DEFAULT '',
        direction       TEXT    NOT NULL DEFAULT '',
        alert_enabled   INTEGER NOT NULL DEFAULT 0,
        alert_state     TEXT    NOT NULL DEFAULT 'armed',
        last_fired      TEXT    NOT NULL DEFAULT '',
        reference_price REAL
    )
    """,
    # Alert delivery settings: exactly one row, id pinned to 1 by a CHECK so
    # the table cannot accumulate duplicates that would silently disagree
    # about who gets notified. Like `schedules`, it holds no SMTP credential:
    # the mail server is read from the environment at send time.
    """
    CREATE TABLE IF NOT EXISTS alert_settings (
        id                INTEGER PRIMARY KEY CHECK (id = 1),
        enabled           INTEGER NOT NULL DEFAULT 0,
        recipients        TEXT    NOT NULL DEFAULT '',
        rearm_pct         REAL    NOT NULL DEFAULT 1.0,
        cooldown_hours    REAL    NOT NULL DEFAULT 12.0,
        use_session_range INTEGER NOT NULL DEFAULT 1,
        timezone          TEXT    NOT NULL DEFAULT 'Asia/Kolkata'
    )
    """,
    # Recurring tearsheet deliveries. Like the watchlist, NOT a child of
    # portfolios: it stores the portfolio by NAME so that deleting and
    # recreating a portfolio does not silently orphan a live schedule, and so
    # the CLI runner can resolve it without an id it has no way to know.
    #
    # What this table deliberately does NOT hold: any SMTP credential. The
    # host, username and password are read from the environment or
    # st.secrets at send time. A password in data/quantfolio.db would be a
    # plaintext secret inside the project directory, which is exactly the
    # thing .gitignore cannot save you from once it has been copied.
    """
    CREATE TABLE IF NOT EXISTS schedules (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL UNIQUE,
        portfolio     TEXT    NOT NULL DEFAULT '',
        recipients    TEXT    NOT NULL DEFAULT '',
        frequency     TEXT    NOT NULL DEFAULT 'weekly',
        hour          INTEGER NOT NULL DEFAULT 8,
        minute        INTEGER NOT NULL DEFAULT 0,
        weekday       INTEGER NOT NULL DEFAULT 0,
        day_of_month  INTEGER NOT NULL DEFAULT 1,
        timezone      TEXT    NOT NULL DEFAULT 'Asia/Kolkata',
        enabled       INTEGER NOT NULL DEFAULT 1,
        last_sent     TEXT    NOT NULL DEFAULT '',
        period        TEXT    NOT NULL DEFAULT '1y',
        benchmark     TEXT    NOT NULL DEFAULT '',
        base_currency TEXT    NOT NULL DEFAULT 'INR',
        attach_excel  INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON holdings(portfolio_id)",
    "CREATE INDEX IF NOT EXISTS idx_txn_portfolio ON transactions(portfolio_id)",
)

#: ``table -> ((column, DDL), ...)`` applied to databases created before a
#: column existed. ``CREATE TABLE IF NOT EXISTS`` leaves an existing table
#: exactly as it was, so without this a watchlist saved by an earlier version
#: would keep four columns and every alert write would fail with "no such
#: column" -- on the one database that already has the user's data in it.
#:
#: Additive only, and deliberately so: no column is ever dropped, renamed or
#: retyped here. Those need a table rebuild, which risks the data this file
#: exists to protect; an unused column costs nothing.
_MIGRATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "watchlist": (
        ("direction", "TEXT NOT NULL DEFAULT ''"),
        ("alert_enabled", "INTEGER NOT NULL DEFAULT 0"),
        ("alert_state", "TEXT NOT NULL DEFAULT 'armed'"),
        ("last_fired", "TEXT NOT NULL DEFAULT ''"),
        ("reference_price", "REAL"),
    ),
}


# ---------------------------------------------------------------------------
# coercion helpers (a store must never crash on a messy frame)
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC string, seconds resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _snake_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` with lower_snake_case column labels."""
    out = frame.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def _as_ticker(value: object) -> str:
    """Upper-cased symbol; "" for blanks and the usual NaN sentinels."""
    text = str(value).strip().upper() if value is not None else ""
    return "" if text in {"", "NAN", "NONE", "NAT", "N/A"} else text


def _as_float(value: object, default: float | None = None) -> float | None:
    """Finite float or ``default`` (NaN, infinity and junk are not stored)."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_flag(value: object, default: bool = False) -> bool:
    """A boolean for a SQLite 0/1 column, NaN-safely.

    ``bool(float("nan"))`` is **True**, so a DataFrame column that exists but
    holds no value -- which is what ``pd.DataFrame(rows, columns=...)``
    produces for a key the rows never supplied -- would be written as 1. For
    ``alert_enabled`` that means switching price alerts on for every row of a
    watchlist that had none.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f", "", "nan", "none", "null"}:
        return False
    return default


def _as_int(value: object, default: int) -> int:
    """An integer, or ``default``. Accepts "8", 8.0 and numpy scalars alike.

    A stored hour that has somehow become unreadable must not stop the whole
    schedule table from loading, so this never raises.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return int(number)


def _as_text(value: object, default: str = "") -> str:
    """Trimmed string; ``default`` for missing values and NaN sentinels."""
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    text = str(value).strip()
    return default if text.upper() in {"", "NAN", "NONE", "NAT"} else text


def _as_iso_date(value: object) -> str:
    """ISO calendar date ("YYYY-MM-DD"); "" when unparseable."""
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001 - exotic types
        return ""
    if stamp is None or pd.isna(stamp):
        return ""
    try:
        return pd.Timestamp(stamp).date().isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _empty_holdings() -> pd.DataFrame:
    """Typed empty holdings frame with the documented column contract."""
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"),
            "buy_price": pd.Series(dtype="float64"),
        }
    )


def _empty_watchlist() -> pd.DataFrame:
    """Typed empty watchlist frame with the documented column contract."""
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "note": pd.Series(dtype="object"),
            "target_price": pd.Series(dtype="float64"),
            "added_at": pd.Series(dtype="object"),
            "direction": pd.Series(dtype="object"),
            "alert_enabled": pd.Series(dtype="bool"),
            "alert_state": pd.Series(dtype="object"),
            "last_fired": pd.Series(dtype="object"),
            "reference_price": pd.Series(dtype="float64"),
        }
    )


def _empty_transactions() -> pd.DataFrame:
    """Typed empty transactions frame with the documented column contract."""
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "trade_date": pd.Series(dtype="object"),
            "action": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"),
            "price": pd.Series(dtype="float64"),
            "fees": pd.Series(dtype="float64"),
            "currency": pd.Series(dtype="object"),
            "note": pd.Series(dtype="object"),
        }
    )


def _holding_rows(raw: pd.DataFrame | None) -> list[tuple[str, float, float]]:
    """Coerce a holdings frame to storable ``(ticker, quantity, buy_price)``.

    Rows with a blank ticker or a non-finite quantity/buy price are dropped -
    they cannot be represented and would resurface as garbage on load. Extra
    columns (``buy_date``, ``sector``, ``target_weight_pct``, ...) are not part
    of the schema and are ignored. Negative quantities are preserved: a short
    position is a legitimate thing to store, and judging it is not the store's
    job.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return []
    frame = _snake_columns(raw)
    if not set(HOLDINGS_COLUMNS).issubset(frame.columns):
        logger.warning(
            "Holdings frame is missing %s; nothing stored.",
            sorted(set(HOLDINGS_COLUMNS) - set(frame.columns)),
        )
        return []

    rows: list[tuple[str, float, float]] = []
    for record in frame.to_dict("records"):
        ticker = _as_ticker(record.get("ticker"))
        quantity = _as_float(record.get("quantity"))
        buy_price = _as_float(record.get("buy_price"))
        if not ticker or quantity is None or buy_price is None:
            continue
        rows.append((ticker, float(quantity), float(buy_price)))
    return rows


def _transaction_rows(
    raw: pd.DataFrame | None, base_currency: str
) -> list[tuple[str, str, str, float, float, float, str, str]]:
    """Coerce a transactions frame to storable rows.

    Required: ``ticker``, ``trade_date``, ``action``, ``quantity``, ``price``.
    Optional: ``fees`` (0.0), ``currency`` (the portfolio's base) and ``note``.
    ``action`` is stored upper-cased VERBATIM - mapping "B"/"PURCHASE" onto
    BUY and rejecting unknown verbs belongs to the analytics layer, which owns
    the trade vocabulary; the store must round-trip what it was handed.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return []
    frame = _snake_columns(raw)
    required = ("ticker", "trade_date", "action", "quantity", "price")
    if not set(required).issubset(frame.columns):
        logger.warning(
            "Transactions frame is missing %s; nothing stored.",
            sorted(set(required) - set(frame.columns)),
        )
        return []

    rows: list[tuple[str, str, str, float, float, float, str, str]] = []
    for record in frame.to_dict("records"):
        ticker = _as_ticker(record.get("ticker"))
        trade_date = _as_iso_date(record.get("trade_date"))
        action = _as_text(record.get("action")).upper()
        quantity = _as_float(record.get("quantity"))
        price = _as_float(record.get("price"))
        if not ticker or not trade_date or not action:
            continue
        if quantity is None or price is None:
            continue
        rows.append(
            (
                ticker,
                trade_date,
                action,
                float(quantity),
                float(price),
                float(_as_float(record.get("fees"), 0.0) or 0.0),
                _as_text(record.get("currency"), base_currency).upper(),
                _as_text(record.get("note"), ""),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class PortfolioStore:
    """Thread-safe SQLite store for named portfolios.

    Usage::

        with PortfolioStore() as store:
            pid = store.save_portfolio("Core", holdings, "INR", transactions)
            holdings, transactions = store.load_portfolio("Core")

    The schema is created on construction if absent, so there is no separate
    migration step. Deleting a portfolio cascades to its holdings and
    transactions (``PRAGMA foreign_keys=ON`` is set on every connection -
    SQLite disables FK enforcement by default).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open (and if needed create) the database at ``db_path``.

        ``None`` uses :data:`DEFAULT_DB_PATH`; the literal ``":memory:"`` opens
        a private in-memory database, which is what the tests use.
        """
        raw_path = DEFAULT_DB_PATH if db_path is None else db_path
        self._memory = str(raw_path) == ":memory:"
        self.db_path: Path | str = str(raw_path) if self._memory else Path(raw_path)
        self._lock = threading.RLock()

        if not self._memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=_BUSY_TIMEOUT_MS / 1000
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_MS)}")
        if not self._memory:
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error as exc:  # pragma: no cover - filesystem dependent
                logger.warning("WAL journal unavailable at %s: %s", self.db_path, exc)
        self._ensure_schema()

    # -- lifecycle --------------------------------------------------------
    def _connection(self) -> sqlite3.Connection:
        """The live connection, or raise if the store has been closed."""
        if self._conn is None:
            raise RuntimeError("PortfolioStore is closed.")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create the tables and indices, then apply additive migrations.

        Idempotent, and safe on a database created by any earlier version:
        creation is ``IF NOT EXISTS`` and each migration is applied only when
        ``PRAGMA table_info`` shows the column missing. There is no version
        counter to keep in sync -- the columns present *are* the version.
        """
        with self._lock:
            conn = self._connection()
            with conn:
                for statement in _SCHEMA:
                    conn.execute(statement)
                self._apply_migrations(conn)

    @staticmethod
    def _apply_migrations(conn: sqlite3.Connection) -> list[str]:
        """Add any missing additive column. Returns what was added.

        A failure is logged and skipped rather than raised: an un-migrated
        column costs one unavailable feature, whereas a store that refuses to
        open costs the user every portfolio they have saved.
        """
        applied: list[str] = []
        for table, columns in _MIGRATIONS.items():
            try:
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            except sqlite3.Error as exc:  # pragma: no cover - table absent entirely
                logger.warning("Could not inspect %s: %s", table, exc)
                continue
            if not existing:
                continue  # the CREATE above will have made it current
            for column, ddl in columns:
                if column in existing:
                    continue
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                except sqlite3.Error as exc:
                    logger.warning("Could not add %s.%s: %s", table, column, exc)
                    continue
                applied.append(f"{table}.{column}")
        if applied:
            logger.info("Schema migrated: added %s", ", ".join(applied))
        return applied

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def __enter__(self) -> PortfolioStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- lookup -----------------------------------------------------------
    def _resolve_id(self, name_or_id: int | str) -> int | None:
        """Resolve a portfolio reference to its id, or ``None`` if unknown.

        An ``int`` is treated as an id. A string is matched against the NAME
        first and only then, if it is all digits, against the id - so a
        portfolio literally called "12" can still be addressed by name.
        """
        conn = self._connection()
        if isinstance(name_or_id, bool):
            return None
        if isinstance(name_or_id, int):
            row = conn.execute(
                "SELECT id FROM portfolios WHERE id = ?", (int(name_or_id),)
            ).fetchone()
            return int(row["id"]) if row is not None else None

        name = _as_text(name_or_id)
        if not name:
            return None
        row = conn.execute("SELECT id FROM portfolios WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return int(row["id"])
        if name.isdigit():
            row = conn.execute("SELECT id FROM portfolios WHERE id = ?", (int(name),)).fetchone()
            return int(row["id"]) if row is not None else None
        return None

    def list_portfolios(self) -> list[dict[str, Any]]:
        """Every saved portfolio, most recently updated first.

        Each dict carries ``id, name, base_currency, n_holdings,
        n_transactions, created_at, updated_at``. Returns ``[]`` for an empty
        database.
        """
        with self._lock:
            rows = (
                self._connection()
                .execute(
                    """
                    SELECT p.id, p.name, p.base_currency, p.created_at, p.updated_at,
                           (SELECT COUNT(*) FROM holdings h
                             WHERE h.portfolio_id = p.id) AS n_holdings,
                           (SELECT COUNT(*) FROM transactions t
                             WHERE t.portfolio_id = p.id) AS n_transactions
                      FROM portfolios p
                     ORDER BY p.updated_at DESC, p.name ASC
                    """
                )
                .fetchall()
            )
        return [
            {
                "id": int(row["id"]),
                "name": str(row["name"]),
                "base_currency": str(row["base_currency"]),
                "n_holdings": int(row["n_holdings"]),
                "n_transactions": int(row["n_transactions"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    # -- write ------------------------------------------------------------
    def save_portfolio(
        self,
        name: str,
        holdings: pd.DataFrame,
        base_currency: str = DEFAULT_BASE_CURRENCY,
        transactions: pd.DataFrame | None = None,
    ) -> int:
        """Insert or update the portfolio called ``name``; returns its id.

        Upsert semantics, all inside ONE transaction so an interrupted save
        can never leave half a portfolio behind:

        * the portfolio row is created, or its ``base_currency`` and
          ``updated_at`` are refreshed;
        * holdings are ALWAYS replaced wholesale by ``holdings``;
        * transactions are replaced only when a frame is passed. ``None``
          means "leave the stored ledger alone", so saving an edited holdings
          table from the UI cannot silently destroy the trade history.

        Rows that cannot be represented (blank ticker, non-finite number,
        unparseable date) are skipped rather than raising. ``ValueError`` is
        raised only for a blank portfolio name, which is a caller bug: the
        name is the primary key users address the portfolio by.
        """
        clean_name = _as_text(name)
        if not clean_name:
            raise ValueError("Portfolio name must not be blank.")
        currency = _as_text(base_currency, DEFAULT_BASE_CURRENCY).upper()

        holding_rows = _holding_rows(holdings)
        txn_rows = None if transactions is None else _transaction_rows(transactions, currency)
        now = _utc_now_iso()

        with self._lock:
            conn = self._connection()
            with conn:
                row = conn.execute(
                    "SELECT id FROM portfolios WHERE name = ?", (clean_name,)
                ).fetchone()
                if row is None:
                    cursor = conn.execute(
                        """
                        INSERT INTO portfolios
                               (name, base_currency, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (clean_name, currency, now, now),
                    )
                    portfolio_id = int(cursor.lastrowid or 0)
                else:
                    portfolio_id = int(row["id"])
                    conn.execute(
                        """
                        UPDATE portfolios
                           SET base_currency = ?, updated_at = ?
                         WHERE id = ?
                        """,
                        (currency, now, portfolio_id),
                    )

                conn.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
                if holding_rows:
                    conn.executemany(
                        """
                        INSERT INTO holdings (portfolio_id, ticker, quantity, buy_price)
                        VALUES (?, ?, ?, ?)
                        """,
                        [(portfolio_id, *values) for values in holding_rows],
                    )

                if txn_rows is not None:
                    conn.execute(
                        "DELETE FROM transactions WHERE portfolio_id = ?",
                        (portfolio_id,),
                    )
                    if txn_rows:
                        conn.executemany(
                            """
                            INSERT INTO transactions
                                   (portfolio_id, ticker, trade_date, action,
                                    quantity, price, fees, currency, note)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [(portfolio_id, *values) for values in txn_rows],
                        )
        return portfolio_id

    def rename_portfolio(self, name_or_id: int | str, new_name: str) -> bool:
        """Rename a portfolio; ``True`` when a row actually changed.

        Returns ``False`` (never raises) when the portfolio does not exist,
        the new name is blank, or another portfolio already owns that name -
        ``name`` is UNIQUE and the store will not silently merge two of them.
        """
        clean_name = _as_text(new_name)
        now = _utc_now_iso()
        with self._lock:
            conn = self._connection()
            portfolio_id = self._resolve_id(name_or_id)
            if portfolio_id is None or not clean_name:
                return False
            try:
                with conn:
                    cursor = conn.execute(
                        "UPDATE portfolios SET name = ?, updated_at = ? WHERE id = ?",
                        (clean_name, now, portfolio_id),
                    )
            except sqlite3.IntegrityError as exc:
                logger.warning("Rename to %r rejected: %s", clean_name, exc)
                return False
            return cursor.rowcount > 0

    def delete_portfolio(self, name_or_id: int | str) -> bool:
        """Delete a portfolio and (by FK cascade) its holdings and trades.

        ``True`` when a row was removed, ``False`` when the portfolio was not
        found. Never raises.
        """
        with self._lock:
            conn = self._connection()
            portfolio_id = self._resolve_id(name_or_id)
            if portfolio_id is None:
                return False
            with conn:
                cursor = conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
            return cursor.rowcount > 0

    # -- read -------------------------------------------------------------
    def load_portfolio(self, name_or_id: int | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(holdings, transactions)`` for one portfolio.

        ``holdings`` has columns ``ticker, quantity, buy_price``;
        ``transactions`` has ``ticker, trade_date, action, quantity, price,
        fees, currency, note`` with ``trade_date`` as an ISO date string,
        ordered by trade date then insertion order.

        An unknown portfolio yields two EMPTY, correctly-typed frames rather
        than an exception - a missing save is a normal UI state, not an error.
        """
        with self._lock:
            conn = self._connection()
            portfolio_id = self._resolve_id(name_or_id)
            if portfolio_id is None:
                return _empty_holdings(), _empty_transactions()
            holding_rows = conn.execute(
                """
                SELECT ticker, quantity, buy_price
                  FROM holdings
                 WHERE portfolio_id = ?
                 ORDER BY id
                """,
                (portfolio_id,),
            ).fetchall()
            txn_rows = conn.execute(
                """
                SELECT ticker, trade_date, action, quantity, price,
                       fees, currency, note
                  FROM transactions
                 WHERE portfolio_id = ?
                 ORDER BY trade_date, id
                """,
                (portfolio_id,),
            ).fetchall()

        holdings = _empty_holdings()
        if holding_rows:
            holdings = pd.DataFrame(
                [tuple(row) for row in holding_rows], columns=list(HOLDINGS_COLUMNS)
            )
            holdings["quantity"] = holdings["quantity"].astype("float64")
            holdings["buy_price"] = holdings["buy_price"].astype("float64")

        transactions = _empty_transactions()
        if txn_rows:
            transactions = pd.DataFrame(
                [tuple(row) for row in txn_rows], columns=list(TRANSACTIONS_COLUMNS)
            )
            for column in ("quantity", "price", "fees"):
                transactions[column] = transactions[column].astype("float64")

        return holdings, transactions

    # -- watchlist --------------------------------------------------------
    def load_watchlist(self) -> pd.DataFrame:
        """The saved watchlist in the order it was stored.

        Columns ``ticker, note, target_price, added_at``; ``target_price`` is
        ``NaN`` where the user set none, which is what
        :func:`src.watchlist.from_frame` reads back as ``None``. An empty
        database yields an EMPTY, correctly-typed frame, never an exception.

        Ordered by ``id`` (insertion order), NOT by recency: save is a
        wholesale replace, so ordering the read differently from the write
        would flip the list on every save/load cycle. Presentation order is
        the UI's business.
        """
        try:
            with self._lock:
                rows = (
                    self._connection()
                    .execute(f"SELECT {_WATCHLIST_SQL_COLUMNS} FROM watchlist ORDER BY id")
                    .fetchall()
                )
        except sqlite3.Error as exc:
            # Only reachable if the alert migration could not be applied. An
            # empty watchlist is a survivable state; a page that will not load
            # is not.
            logger.warning("Watchlist could not be read from %s: %s", self.db_path, exc)
            return _empty_watchlist()
        if not rows:
            return _empty_watchlist()
        frame = pd.DataFrame([tuple(row) for row in rows], columns=list(WATCHLIST_COLUMNS))
        for numeric in ("target_price", "reference_price"):
            frame[numeric] = pd.to_numeric(frame[numeric], errors="coerce").astype("float64")
        for text in ("note", "added_at", "direction", "alert_state", "last_fired"):
            frame[text] = frame[text].fillna("").astype("object")
        # SQLite has no boolean type; normalise here so no caller has to
        # remember that 0/1 came back instead of False/True.
        frame["alert_enabled"] = (
            pd.to_numeric(frame["alert_enabled"], errors="coerce").fillna(0).astype(bool)
        )
        return frame

    def save_watchlist(self, watchlist: pd.DataFrame | None) -> int:
        """Replace the entire watchlist; returns the number of rows stored.

        Wholesale replacement inside one transaction, deliberately: the
        watchlist is small, the UI always holds the complete list in session
        state, and a diffing upsert would be more code and more ways to end up
        with a half-applied edit. ``None`` stores nothing and clears the table,
        matching "the user emptied their watchlist".

        Rows with a blank ticker are skipped (empty grid rows). A duplicate
        ticker keeps its FIRST occurrence -- ``ticker`` is UNIQUE and silently
        merging two notes onto one symbol would lose one of them.

        The five alert columns are written alongside, so a rule and the target
        it watches can never be saved out of step. Missing alert keys fall
        back to "no alert", which is what a frame from an older export, or
        from a ``st.data_editor`` grid that shows only the visible columns,
        legitimately means.
        """
        rows: list[tuple[object, ...]] = []
        if watchlist is not None and isinstance(watchlist, pd.DataFrame) and not watchlist.empty:
            frame = _snake_columns(watchlist)
            if "ticker" not in frame.columns:
                logger.warning("Watchlist frame has no ticker column; nothing stored.")
            else:
                seen: set[str] = set()
                for record in frame.to_dict("records"):
                    ticker = _as_ticker(record.get("ticker"))
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    target = _as_float(record.get("target_price"))
                    reference = _as_float(record.get("reference_price"))
                    state = _as_text(record.get("alert_state"), "armed").lower()
                    rows.append(
                        (
                            ticker,
                            _as_text(record.get("note"), ""),
                            float(target) if target is not None and target > 0 else None,
                            _as_iso_date(record.get("added_at")),
                            _as_text(record.get("direction"), "").lower(),
                            1 if _as_flag(record.get("alert_enabled")) else 0,
                            state if state in {"armed", "triggered"} else "armed",
                            _as_text(record.get("last_fired"), ""),
                            float(reference) if reference is not None and reference > 0 else None,
                        )
                    )

        with self._lock:
            conn = self._connection()
            with conn:
                conn.execute("DELETE FROM watchlist")
                if rows:
                    conn.executemany(
                        "INSERT INTO watchlist "
                        f"({_WATCHLIST_SQL_COLUMNS}) VALUES ({_WATCHLIST_SQL_PLACEHOLDERS})",
                        rows,
                    )
        return len(rows)

    def update_alert_state(
        self,
        ticker: str,
        state: str,
        last_fired: object = None,
        direction: str | None = None,
    ) -> bool:
        """Write one watchlist row's alert state. True if a row moved.

        A single targeted UPDATE touching only the alert columns, and the ONLY
        write the alert runner performs. The same reasoning as
        :meth:`mark_schedule_sent`: :meth:`save_watchlist` is a wholesale
        replace, so a cron job using it would silently undo a note or a target
        the user edited in the browser while quotes were being fetched.

        It is also the commit point of the fire-once guarantee in
        :func:`src.alerts.evaluate`. Written immediately after a successful
        send, so a crash between sending and recording costs one duplicate
        alert at worst -- never a missed one.
        """
        key = _as_ticker(ticker)
        if not key:
            return False
        clean_state = _as_text(state, "armed").lower()
        if clean_state not in {"armed", "triggered"}:
            clean_state = "armed"
        stamp = (
            last_fired.isoformat(timespec="seconds")
            if isinstance(last_fired, datetime)
            else _as_text(last_fired, "")
        )

        with self._lock:
            conn = self._connection()
            with conn:
                if direction is None:
                    cursor = conn.execute(
                        """
                        UPDATE watchlist
                           SET alert_state = ?, last_fired = ?
                         WHERE ticker = ?
                        """,
                        (clean_state, stamp, key),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE watchlist
                           SET alert_state = ?, last_fired = ?, direction = ?
                         WHERE ticker = ?
                        """,
                        (clean_state, stamp, _as_text(direction, "").lower(), key),
                    )
        return bool(cursor.rowcount)

    # -- alert settings ----------------------------------------------------
    def load_alert_settings(self) -> dict[str, object]:
        """The single alert-settings row, as a plain record.

        Returns the defaults -- alerting OFF, nobody notified -- when nothing
        has been saved. Alerts must never be on by default: an e-mail nobody
        asked for is worse than no e-mail.
        """
        defaults: dict[str, object] = {
            "enabled": False,
            "recipients": "",
            "rearm_pct": 1.0,
            "cooldown_hours": 12.0,
            "use_session_range": True,
            "timezone": "Asia/Kolkata",
        }
        try:
            with self._lock:
                row = (
                    self._connection()
                    .execute(f"SELECT {_ALERT_SQL_COLUMNS} FROM alert_settings WHERE id = 1")
                    .fetchone()
                )
        except sqlite3.Error as exc:
            logger.warning("Alert settings could not be read from %s: %s", self.db_path, exc)
            return defaults
        if row is None:
            return defaults

        record = dict(zip(ALERT_SETTINGS_COLUMNS, tuple(row), strict=True))
        record["enabled"] = bool(record.get("enabled"))
        record["use_session_range"] = bool(record.get("use_session_range"))
        record["rearm_pct"] = _as_float(record.get("rearm_pct"), 1.0)
        record["cooldown_hours"] = _as_float(record.get("cooldown_hours"), 12.0)
        record["recipients"] = _as_text(record.get("recipients"), "")
        record["timezone"] = _as_text(record.get("timezone"), "Asia/Kolkata")
        return record

    def save_alert_settings(self, record: Mapping[str, object] | None) -> bool:
        """Replace the alert settings. True when a row was written.

        ``INSERT OR REPLACE`` against the pinned ``id = 1``, so the table
        holds exactly one row however often this is called -- two rows
        disagreeing about who gets notified would be a genuinely confusing
        bug to diagnose from the outside.
        """
        if record is None or not isinstance(record, Mapping):
            return False
        rearm = _as_float(record.get("rearm_pct"), 1.0)
        cooldown = _as_float(record.get("cooldown_hours"), 12.0)
        values = (
            1 if _as_flag(record.get("enabled")) else 0,
            _as_text(record.get("recipients"), ""),
            float(rearm if rearm is not None else 1.0),
            float(cooldown if cooldown is not None else 12.0),
            1 if _as_flag(record.get("use_session_range"), True) else 0,
            _as_text(record.get("timezone"), "Asia/Kolkata"),
        )
        with self._lock:
            conn = self._connection()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO alert_settings "
                    f"(id, {_ALERT_SQL_COLUMNS}) VALUES (1, {_ALERT_SQL_PLACEHOLDERS})",
                    values,
                )
        return True

    # -- scheduled reports -------------------------------------------------
    def load_schedules(self) -> list[dict[str, object]]:
        """Every stored schedule, as plain records in insertion order.

        Records rather than a DataFrame or a dataclass: a schedule is a small
        heterogeneous thing, and returning dicts keeps this module free of any
        import from the reporting layer. ``src.reporting.schedule.from_record``
        turns one of these into a ``Schedule``.

        A missing table or an unreadable database yields an EMPTY list, never
        an exception -- the runner must be able to conclude "nothing is
        scheduled" without crashing a cron job.
        """
        try:
            with self._lock:
                rows = (
                    self._connection()
                    .execute(f"SELECT {_SCHEDULE_SQL_COLUMNS} FROM schedules ORDER BY id")
                    .fetchall()
                )
        except sqlite3.Error as exc:
            logger.warning("Schedules could not be read from %s: %s", self.db_path, exc)
            return []

        records: list[dict[str, object]] = []
        for row in rows:
            record = dict(zip(SCHEDULE_COLUMNS, tuple(row), strict=True))
            # SQLite has no boolean type; normalise here so callers never have
            # to remember that 0/1 came back instead of False/True.
            record["enabled"] = bool(record.get("enabled"))
            record["attach_excel"] = bool(record.get("attach_excel"))
            records.append(record)
        return records

    def save_schedules(self, records: Sequence[Mapping[str, object]] | None) -> int:
        """Replace every schedule; returns the number stored.

        Wholesale replacement in one transaction, matching
        :meth:`save_watchlist`: the list is small and the UI always holds all
        of it. Rows with a blank name are skipped, and a duplicate name keeps
        its FIRST occurrence, because ``name`` is UNIQUE and merging two
        schedules onto one name would lose one of them.

        Note the asymmetry with :meth:`mark_schedule_sent`: the UI replaces the
        whole list, the runner touches one timestamp. A runner that replaced
        the list would clobber an edit made while it was working.
        """
        rows: list[tuple[object, ...]] = []
        seen: set[str] = set()
        for record in records or ():
            if not isinstance(record, Mapping):
                continue
            name = _as_text(record.get("name"), "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            rows.append(
                (
                    name,
                    _as_text(record.get("portfolio"), ""),
                    _as_text(record.get("recipients"), ""),
                    _as_text(record.get("frequency"), "weekly"),
                    _as_int(record.get("hour"), 8),
                    _as_int(record.get("minute"), 0),
                    _as_int(record.get("weekday"), 0),
                    _as_int(record.get("day_of_month"), 1),
                    _as_text(record.get("timezone"), "Asia/Kolkata"),
                    1 if _as_flag(record.get("enabled"), True) else 0,
                    _as_text(record.get("last_sent"), ""),
                    _as_text(record.get("period"), "1y"),
                    _as_text(record.get("benchmark"), ""),
                    _as_text(record.get("base_currency"), "INR"),
                    1 if _as_flag(record.get("attach_excel")) else 0,
                )
            )

        with self._lock:
            conn = self._connection()
            with conn:
                conn.execute("DELETE FROM schedules")
                if rows:
                    conn.executemany(
                        "INSERT INTO schedules "
                        f"({_SCHEDULE_SQL_COLUMNS}) VALUES ({_SCHEDULE_SQL_PLACEHOLDERS})",
                        rows,
                    )
        return len(rows)

    def mark_schedule_sent(self, name: str, when: object) -> bool:
        """Record that ``name`` was delivered at ``when``. True if a row moved.

        A single targeted UPDATE, and the ONLY write the scheduled runner
        performs. Two reasons it is not a read-modify-replace:

        * it cannot clobber a schedule the user edited while the report was
          being built and sent, and
        * it is the commit point of the idempotence guarantee in
          :func:`src.reporting.schedule.is_due`. Written immediately after a
          successful send, so a crash between sending and recording costs one
          duplicate e-mail at worst -- never a lost report.
        """
        stamp = (
            when.isoformat(timespec="seconds") if isinstance(when, datetime) else str(when or "")
        )
        target = str(name or "").strip()
        if not target:
            return False
        with self._lock:
            conn = self._connection()
            with conn:
                cursor = conn.execute(
                    "UPDATE schedules SET last_sent = ? WHERE name = ?",
                    (stamp, target),
                )
        return bool(cursor.rowcount)
