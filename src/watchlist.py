"""Watchlist: instruments tracked without holding them.

A watchlist is the other half of a portfolio tool. The portfolio answers "what
do I own and what is it doing"; the watchlist answers "what am I considering,
and has it reached the level I said I would act at". Keeping them separate
matters analytically: a watched name has no quantity and no cost basis, so it
must never enter position P&L, weights, risk statistics or attribution — which
is exactly why this is its own module and its own table rather than a flag on
a holding.

Pure module: pandas only, no Streamlit and no network. Quotes and
profiles are passed IN, so :func:`watch_table` is testable against hand-built
fixtures like every other calculation in this codebase.

Two deliberate restraints:

* **A target price is a reminder, not a forecast.** ``to_target_pct`` is the
  arithmetic distance from the current price to a number the user typed. The
  module never suggests a target, never ranks watchlist entries by
  attractiveness, and never implies a target will be reached.
* **A target can carry a direction, and alerting is opt-in.** ``direction``
  records whether the target is a level to rise to or to fall to, and
  ``reference_price`` records the price when it was set so that direction can
  be inferred if the user did not state one. Both exist because
  :func:`triggered` could not distinguish those cases; :mod:`src.alerts` owns
  the state machine that acts on them. Setting a target never switches
  alerting on by itself — an e-mail nobody asked for is worse than no
  e-mail.
* **Missing data stays missing.** No 52-week range from the provider means
  ``range_position_pct`` is NaN, not 50. A fabricated midpoint would read as
  "fairly valued" on a chart.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

import pandas as pd

from src.models import Quote, SecurityProfile

__all__ = [
    "MAX_ITEMS",
    "MAX_NOTE_LENGTH",
    "WATCHLIST_COLUMNS",
    "TABLE_COLUMNS",
    "WatchItem",
    "normalise_ticker",
    "add",
    "remove",
    "update_item",
    "contains",
    "from_frame",
    "to_frame",
    "watch_table",
    "summarise",
    "triggered",
    "with_alert",
]

#: Soft cap on watchlist size. Each entry costs one quote and one profile
#: request per refresh; a 500-name watchlist would make the page unusably slow
#: and hammer a free endpoint. The UI refuses politely past this.
MAX_ITEMS: int = 100

#: Notes are a memo to self, not a document.
MAX_NOTE_LENGTH: int = 200

#: Storage/round-trip column contract. The last five columns carry the alert
#: rule and its state; they were added after the first four, so
#: :class:`src.data.store.PortfolioStore` migrates an existing table rather
#: than assuming a fresh one.
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

#: Exact column order of :func:`watch_table`.
TABLE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "name",
    "sector",
    "currency",
    "price",
    "day_change_pct",
    "week52_low",
    "week52_high",
    "range_position_pct",
    "target_price",
    "to_target_pct",
    "direction",
    "alert_enabled",
    "alert_state",
    "in_portfolio",
    "note",
    "added_at",
    "as_of",
)

_MISSING_TEXT: frozenset[str] = frozenset({"", "NAN", "NONE", "NAT", "N/A", "NULL"})


def _nan() -> float:
    return float("nan")


def _as_float(value: object) -> float | None:
    """Finite float or ``None``. Never returns 0.0 for a missing value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in _MISSING_TEXT else text


def _as_bool(value: object, default: bool = False) -> bool:
    """A boolean, NaN-safely. ``float("nan")`` reads as ``default``, not True.

    The trap this exists to close: ``bool(float("nan"))`` is **True**, so a
    frame that names ``alert_enabled`` but supplies no value for it -- a CSV
    round-trip, or a database migrated from before alerts existed -- would
    switch alerting ON for every row and start e-mailing about targets set
    long ago. Found by a test asserting that a frame without the alert columns
    loads as "no alert".
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


def _as_datetime(value: object) -> datetime | None:
    """A tz-naive datetime, or ``None``. Never raises on a stored string.

    Naive throughout, matching :mod:`src.reporting.schedule`: every
    comparison is between two times in the same zone, and SQLite has no aware
    type. An unreadable value reads as "never fired", which risks one extra
    alert rather than silently suppressing every future one.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    text = _as_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001 - exotic input
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).date()


def normalise_ticker(text: object) -> str:
    """Upper-case, strip, and resolve through the local stock universe.

    ``"reliance"`` becomes ``"RELIANCE.NS"`` so a watchlist entry added by
    company name is stored as the symbol the data layer can actually price.
    Unknown text is returned upper-cased and otherwise unchanged, for the same
    reason :func:`src.stock_universe.normalise_symbol` does: absence from the
    curated file is not proof a symbol does not exist.
    """
    from src import stock_universe
    from src.config import PREFERRED_EXCHANGE

    raw = _as_text(text).upper()
    if not raw:
        return ""
    return stock_universe.normalise_symbol(raw, prefer_exchange=PREFERRED_EXCHANGE)


# ---------------------------------------------------------------------------
# the item
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WatchItem:
    """One watched instrument.

    ``target_price`` is ``None`` when the user set none — distinct from 0.0,
    which would render as a 100% fall to target. It is in the instrument's own
    quote currency, matching the convention in
    :mod:`src.analytics.positions` where prices stay native.
    """

    ticker: str
    note: str = ""
    target_price: float | None = None
    added_at: date | None = None
    #: ``"above"``, ``"below"`` or ``""``. Which side of the price the target
    #: was on -- the fact :func:`triggered` never had. Interpreted by
    #: :mod:`src.alerts`; ``""`` means "infer it from ``reference_price``".
    direction: str = ""
    #: Opt-in. A target is a reminder; an e-mail is an interruption, so one
    #: never implies the other.
    alert_enabled: bool = False
    #: ``"armed"`` (waiting for the level) or ``"triggered"`` (reached, and
    #: silent until it re-arms). Advanced only by :func:`src.alerts.evaluate`.
    alert_state: str = "armed"
    last_fired: datetime | None = None
    #: The price when the target was set, so a direction can be inferred
    #: later. Recording it is the whole reason an alert can tell an upside
    #: level from an entry level.
    reference_price: float | None = None

    def __post_init__(self) -> None:
        # Normalise on construction so no other code has to remember to.
        object.__setattr__(self, "ticker", _as_text(self.ticker).upper())
        object.__setattr__(self, "note", _as_text(self.note)[:MAX_NOTE_LENGTH])
        target = _as_float(self.target_price)
        object.__setattr__(self, "target_price", target if target and target > 0 else None)

        # Kept as free-form text rather than validated against a vocabulary
        # here: src.alerts.normalise_direction owns that, and importing it
        # would make this module depend on the layer that consumes it.
        object.__setattr__(self, "direction", _as_text(self.direction).lower())
        object.__setattr__(self, "alert_enabled", _as_bool(self.alert_enabled))
        state = _as_text(self.alert_state).lower()
        object.__setattr__(
            self, "alert_state", state if state in {"armed", "triggered"} else "armed"
        )
        object.__setattr__(self, "last_fired", _as_datetime(self.last_fired))
        reference = _as_float(self.reference_price)
        object.__setattr__(
            self, "reference_price", reference if reference and reference > 0 else None
        )

    @property
    def has_target(self) -> bool:
        return self.target_price is not None

    @property
    def alerting(self) -> bool:
        """Whether this rule can actually produce an e-mail.

        A direction is required: without one it is unknown whether the target
        is a level to rise to or to fall to, and a rule that cannot be
        evaluated must not read as active. ``direction`` may be blank when
        ``reference_price`` allows it to be inferred.
        """
        return bool(
            self.alert_enabled
            and self.has_target
            and (self.direction or self.reference_price is not None)
        )

    def distance_to_target(self, price: float | None) -> float:
        """Percentage move from ``price`` to the target. NaN when undefined.

        Positive means the target is ABOVE the current price. Sign convention
        is stated here because a distance-to-target with an ambiguous sign is
        worse than no number at all.
        """
        current = _as_float(price)
        if self.target_price is None or current is None or current <= 0:
            return _nan()
        return float((self.target_price / current - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# collection operations (all return NEW tuples; nothing mutates in place)
# ---------------------------------------------------------------------------
def contains(items: Sequence[WatchItem], ticker: object) -> bool:
    key = _as_text(ticker).upper()
    return any(item.ticker == key for item in items)


def add(
    items: Sequence[WatchItem],
    ticker: object,
    note: str = "",
    target_price: float | None = None,
    added_at: date | None = None,
    resolve: bool = True,
    direction: str = "",
    alert_enabled: bool = False,
    reference_price: float | None = None,
) -> tuple[tuple[WatchItem, ...], str]:
    """Add one instrument. Returns ``(items, message)``.

    ``message`` is "" on success and a human-readable reason otherwise —
    blank ticker, already present, or the list is full. Returning the reason
    rather than raising keeps the caller free of try/except around a normal
    UI action, and guarantees the user is told *why* nothing happened.
    """
    symbol = normalise_ticker(ticker) if resolve else _as_text(ticker).upper()
    if not symbol:
        return tuple(items), "Enter a ticker or company name."
    if contains(items, symbol):
        return tuple(items), f"{symbol} is already on the watchlist."
    if len(items) >= MAX_ITEMS:
        return (
            tuple(items),
            f"The watchlist holds at most {MAX_ITEMS} instruments — each one costs a "
            "price request on every refresh. Remove something first.",
        )
    item = WatchItem(
        ticker=symbol,
        note=note,
        target_price=target_price,
        added_at=added_at or datetime.now(timezone.utc).date(),
        direction=direction,
        alert_enabled=alert_enabled,
        reference_price=reference_price,
    )
    return (*tuple(items), item), ""


def remove(items: Sequence[WatchItem], ticker: object) -> tuple[WatchItem, ...]:
    """Every item except ``ticker``. A no-op when it is not present."""
    key = _as_text(ticker).upper()
    return tuple(item for item in items if item.ticker != key)


def update_item(
    items: Sequence[WatchItem],
    ticker: object,
    note: str | None = None,
    target_price: float | None = None,
    clear_target: bool = False,
    direction: str | None = None,
    alert_enabled: bool | None = None,
    reference_price: float | None = None,
) -> tuple[WatchItem, ...]:
    """Edit one item's note, target and/or alert rule, preserving order.

    ``target_price=None`` means "leave the target alone"; pass
    ``clear_target=True`` to actually remove one. Without that distinction
    there is no way to express "delete my target" through a single optional
    argument. Every alert argument follows the same convention: ``None``
    leaves it untouched.

    Clearing the target also clears the alert, because a rule with no level
    cannot fire and leaving ``alert_enabled`` set would show an active alert
    that never triggers. Changing the target or the direction resets the state
    to ``armed`` -- a new level has not been reached yet, whatever the old one
    had done -- and that reset is what stops an edit from inheriting a stale
    ``triggered`` and going permanently silent.
    """
    key = _as_text(ticker).upper()
    out: list[WatchItem] = []
    for item in items:
        if item.ticker != key:
            out.append(item)
            continue
        changes: dict[str, object] = {}
        if note is not None:
            changes["note"] = note
        if clear_target:
            changes["target_price"] = None
            changes["alert_enabled"] = False
            changes["direction"] = ""
            changes["alert_state"] = "armed"
            changes["last_fired"] = None
        elif target_price is not None:
            changes["target_price"] = target_price
            if _as_float(target_price) != _as_float(item.target_price):
                changes["alert_state"] = "armed"
        if direction is not None and not clear_target:
            changes["direction"] = direction
            if _as_text(direction).lower() != item.direction:
                changes["alert_state"] = "armed"
        if alert_enabled is not None and not clear_target:
            changes["alert_enabled"] = alert_enabled
        if reference_price is not None:
            changes["reference_price"] = reference_price
        out.append(replace(item, **changes) if changes else item)  # type: ignore[arg-type]
    return tuple(out)


def with_alert(items: Sequence[WatchItem], updated: Sequence[WatchItem]) -> tuple[WatchItem, ...]:
    """``items`` with alert state taken from ``updated``, matched on ticker.

    The seam between the state machine and the rest of the app.
    :func:`src.alerts.evaluate` returns whole items with their state advanced;
    this merges only the alert fields back, so an evaluation running in a cron
    job cannot overwrite a note or a target the user edited in the browser
    meanwhile. Same reasoning as the targeted ``UPDATE`` in
    :meth:`src.data.store.PortfolioStore.mark_schedule_sent`.
    """
    by_ticker = {item.ticker: item for item in updated}
    out: list[WatchItem] = []
    for item in items:
        fresh = by_ticker.get(item.ticker)
        if fresh is None:
            out.append(item)
            continue
        out.append(
            replace(
                item,
                direction=fresh.direction,
                alert_state=fresh.alert_state,
                last_fired=fresh.last_fired,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# frame round-trip
# ---------------------------------------------------------------------------
def from_frame(raw: pd.DataFrame | None) -> tuple[WatchItem, ...]:
    """Parse a stored/edited frame into items, skipping unusable rows.

    Column names are matched case- and space-insensitively. Rows with a blank
    ticker are dropped silently: those are empty grid rows from
    ``st.data_editor``, not data the user lost.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return ()

    frame = raw.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "ticker" not in frame.columns:
        return ()

    seen: set[str] = set()
    items: list[WatchItem] = []
    for record in frame.to_dict("records"):
        symbol = _as_text(record.get("ticker")).upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        items.append(
            WatchItem(
                ticker=symbol,
                note=_as_text(record.get("note")),
                target_price=_as_float(record.get("target_price")),
                added_at=_as_date(record.get("added_at")),
                # Absent alert columns fall back to the dataclass defaults, so
                # a watchlist saved before alerts existed loads unchanged and
                # simply has no rules on it.
                direction=_as_text(record.get("direction")),
                alert_enabled=_as_bool(record.get("alert_enabled")),
                alert_state=_as_text(record.get("alert_state")),
                last_fired=_as_datetime(record.get("last_fired")),
                reference_price=_as_float(record.get("reference_price")),
            )
        )
    return tuple(items)


def to_frame(items: Iterable[WatchItem]) -> pd.DataFrame:
    """Items as a storable frame with the :data:`WATCHLIST_COLUMNS` contract."""
    rows = [
        {
            "ticker": item.ticker,
            "note": item.note,
            "target_price": item.target_price,
            "added_at": item.added_at.isoformat() if item.added_at else "",
            "direction": item.direction,
            "alert_enabled": item.alert_enabled,
            "alert_state": item.alert_state,
            "last_fired": item.last_fired.isoformat(timespec="seconds") if item.last_fired else "",
            "reference_price": item.reference_price,
        }
        for item in items
    ]
    if not rows:
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
    frame = pd.DataFrame(rows, columns=list(WATCHLIST_COLUMNS))
    for numeric in ("target_price", "reference_price"):
        frame[numeric] = pd.to_numeric(frame[numeric], errors="coerce").astype("float64")
    frame["alert_enabled"] = frame["alert_enabled"].astype(bool)
    return frame


# ---------------------------------------------------------------------------
# the display table
# ---------------------------------------------------------------------------
def _range_position(price: float | None, low: float | None, high: float | None) -> float:
    """Where ``price`` sits in the 52-week range, 0 = low and 100 = high.

    NaN when either bound is missing or the range has no width. A degenerate
    range is genuinely uninformative, and reporting 50 for it would invent a
    "mid-range" reading out of nothing.
    """
    p, lo, hi = _as_float(price), _as_float(low), _as_float(high)
    if p is None or lo is None or hi is None or hi <= lo:
        return _nan()
    return float((p - lo) / (hi - lo) * 100.0)


def watch_table(
    items: Sequence[WatchItem],
    quotes: dict[str, Quote] | None = None,
    profiles: dict[str, SecurityProfile] | None = None,
    held_tickers: Iterable[str] = (),
) -> pd.DataFrame:
    """Build the watchlist display table.

    Every market field comes from the ``quotes`` / ``profiles`` passed in and
    is NaN when the provider had nothing — a watched instrument whose quote
    failed still appears as a row, because silently dropping it would make the
    user think they had removed it.

    ``in_portfolio`` flags names that are also holdings, which is the single
    most useful cross-reference on this page: it stops a user from "watching"
    something they already own and forgot about.

    Returns a frame with exactly :data:`TABLE_COLUMNS`, empty (but correctly
    typed and ordered) for an empty watchlist.
    """
    quotes = quotes or {}
    profiles = profiles or {}
    held = {str(t).strip().upper() for t in held_tickers}

    if not items:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in TABLE_COLUMNS})

    rows: list[dict[str, object]] = []
    for item in items:
        quote = quotes.get(item.ticker)
        profile = profiles.get(item.ticker)

        price = _as_float(getattr(quote, "price", None))
        low = _as_float(getattr(profile, "fifty_two_week_low", None))
        high = _as_float(getattr(profile, "fifty_two_week_high", None))

        rows.append(
            {
                "ticker": item.ticker,
                "name": (profile.display_name if profile is not None else item.ticker),
                "sector": (profile.sector_label if profile is not None else "Unclassified"),
                "currency": _as_text(getattr(quote, "currency", None))
                or _as_text(getattr(profile, "currency", None)),
                "price": price if price is not None else _nan(),
                "day_change_pct": (float(quote.day_change_pct) if quote is not None else _nan()),
                "week52_low": low if low is not None else _nan(),
                "week52_high": high if high is not None else _nan(),
                "range_position_pct": _range_position(price, low, high),
                "target_price": (item.target_price if item.target_price is not None else _nan()),
                "to_target_pct": item.distance_to_target(price),
                "direction": item.direction,
                "alert_enabled": bool(item.alerting),
                "alert_state": item.alert_state,
                "in_portfolio": item.ticker in held,
                "note": item.note,
                "added_at": item.added_at.isoformat() if item.added_at else "",
                "as_of": getattr(quote, "as_of", None),
            }
        )

    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    for numeric in (
        "price",
        "day_change_pct",
        "week52_low",
        "week52_high",
        "range_position_pct",
        "target_price",
        "to_target_pct",
    ):
        table[numeric] = pd.to_numeric(table[numeric], errors="coerce").astype("float64")
    table["in_portfolio"] = table["in_portfolio"].astype(bool)
    table["alert_enabled"] = table["alert_enabled"].astype(bool)
    return table


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def triggered(table: pd.DataFrame | None, tolerance_pct: float = 2.0) -> pd.DataFrame:
    """Rows whose current price is within ``tolerance_pct`` of the user's target.

    Scope is stated precisely because it is narrower than "the target was hit".
    Only the CURRENT price is stored, never the price at the moment the target
    was set, so this cannot know whether a target was an upside level the price
    rose to or an entry level it fell to — and it does not pretend to. It
    reports proximity: ``|to_target_pct| <= tolerance_pct``. A name that blew
    through its target last week and kept going will not appear, which is why
    the UI labels this "near your level", not "triggered".

    Emphatically not a recommendation. The target is a number the user typed
    and this is arithmetic against it.
    """
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    if "to_target_pct" not in table.columns:
        return pd.DataFrame(columns=list(table.columns))
    distance = pd.to_numeric(table["to_target_pct"], errors="coerce")
    hit = distance.abs() <= max(float(tolerance_pct), 0.0)
    return table.loc[hit.fillna(False)].copy()


def summarise(table: pd.DataFrame | None) -> dict[str, object]:
    """Headline counts and averages for the watchlist KPI row.

    Undefined aggregates are ``float("nan")``, never 0.0 — an average day
    change over zero priced rows does not exist, and the UI renders NaN as an
    em dash rather than as a flat market.
    """
    empty: dict[str, object] = {
        "n_items": 0,
        "n_priced": 0,
        "n_with_target": 0,
        "n_in_portfolio": 0,
        "avg_day_change_pct": _nan(),
        "best_ticker": None,
        "best_day_change_pct": _nan(),
        "worst_ticker": None,
        "worst_day_change_pct": _nan(),
    }
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return empty

    out = dict(empty)
    out["n_items"] = int(len(table))
    prices = pd.to_numeric(table.get("price"), errors="coerce")
    out["n_priced"] = int(prices.notna().sum())
    out["n_with_target"] = int(
        pd.to_numeric(table.get("target_price"), errors="coerce").notna().sum()
    )
    if "in_portfolio" in table.columns:
        out["n_in_portfolio"] = int(table["in_portfolio"].astype(bool).sum())

    changes = pd.to_numeric(table.get("day_change_pct"), errors="coerce").dropna()
    if not changes.empty:
        out["avg_day_change_pct"] = float(changes.mean())
        best = changes.idxmax()
        worst = changes.idxmin()
        out["best_ticker"] = str(table.at[best, "ticker"])
        out["best_day_change_pct"] = float(changes.loc[best])
        out["worst_ticker"] = str(table.at[worst, "ticker"])
        out["worst_day_change_pct"] = float(changes.loc[worst])
    return out
