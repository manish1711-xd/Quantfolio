"""SQLite persistence for saved equity research history and tear-sheets.

Thread-safe SQLite storage for research reports.
Permits saving research records, browsing history, and reloading or deleting past runs.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import DB_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_DB_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / DB_FILENAME
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_research_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL,
    company_name     TEXT    NOT NULL DEFAULT '',
    price            REAL,
    intrinsic_value  REAL,
    recommendation   TEXT    NOT NULL DEFAULT 'Hold',
    summary          TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_ticker ON equity_research_history(ticker);
"""

_lock = threading.RLock()


def init_history_db(db_path: str | Path | None = None) -> None:
    """Initialize SQLite database table for equity research history."""
    target_path = Path(db_path) if db_path else DEFAULT_HISTORY_DB_PATH
    if str(target_path) != ":memory:":
        target_path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        conn = sqlite3.connect(str(target_path), check_same_thread=False)
        try:
            with conn:
                conn.executescript(_SCHEMA)
        finally:
            conn.close()


def save_research_record(
    ticker: str,
    company_name: str,
    price: Optional[float],
    intrinsic_value: Optional[float],
    recommendation: str,
    summary: str,
    db_path: str | Path | None = None,
) -> int:
    """Persist an equity research record. Returns the new record id."""
    target_path = Path(db_path) if db_path else DEFAULT_HISTORY_DB_PATH
    init_history_db(target_path)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clean_ticker = str(ticker).strip().upper()
    clean_name = str(company_name or clean_ticker).strip()

    with _lock:
        conn = sqlite3.connect(str(target_path), check_same_thread=False)
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO equity_research_history
                        (ticker, company_name, price, intrinsic_value, recommendation, summary, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_ticker,
                        clean_name,
                        float(price) if price is not None else None,
                        float(intrinsic_value) if intrinsic_value is not None else None,
                        str(recommendation or "Hold"),
                        str(summary or ""),
                        now,
                    ),
                )
                return int(cur.lastrowid or 0)
        finally:
            conn.close()


def load_research_history(
    ticker: Optional[str] = None,
    limit: int = 50,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch research history records, ordered newest-first."""
    target_path = Path(db_path) if db_path else DEFAULT_HISTORY_DB_PATH
    init_history_db(target_path)

    with _lock:
        conn = sqlite3.connect(str(target_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            if ticker:
                rows = conn.execute(
                    """
                    SELECT id, ticker, company_name, price, intrinsic_value, recommendation, summary, created_at
                      FROM equity_research_history
                     WHERE ticker = ?
                     ORDER BY id DESC
                     LIMIT ?
                    """,
                    (str(ticker).strip().upper(), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, ticker, company_name, price, intrinsic_value, recommendation, summary, created_at
                      FROM equity_research_history
                     ORDER BY id DESC
                     LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()

            return [
                {
                    "id": int(r["id"]),
                    "ticker": str(r["ticker"]),
                    "company_name": str(r["company_name"]),
                    "price": float(r["price"]) if r["price"] is not None else None,
                    "intrinsic_value": float(r["intrinsic_value"]) if r["intrinsic_value"] is not None else None,
                    "recommendation": str(r["recommendation"]),
                    "summary": str(r["summary"]),
                    "created_at": str(r["created_at"]),
                }
                for r in rows
            ]
        finally:
            conn.close()


def delete_research_record(
    record_id: int,
    db_path: str | Path | None = None,
) -> bool:
    """Delete a research history entry by ID."""
    target_path = Path(db_path) if db_path else DEFAULT_HISTORY_DB_PATH
    init_history_db(target_path)

    with _lock:
        conn = sqlite3.connect(str(target_path), check_same_thread=False)
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM equity_research_history WHERE id = ?",
                    (int(record_id),),
                )
                return cur.rowcount > 0
        finally:
            conn.close()
