"""Schedule model and due-time arithmetic for recurring tearsheets.

Why this is a separate, pure module
-----------------------------------
Streamlit is a request-response web application. It has no scheduler, no
background worker and no guarantee that any process is running when nobody has
the browser open. A "daily 8am email" implemented as a thread inside the app
would:

* not run at all when the app is idle or restarted,
* run two, three or ten times when several browser sessions are open,
* and die silently, which is the worst failure mode a scheduled job can have.

So this application does NOT pretend to schedule. It stores the schedule, and
a real scheduler -- cron, systemd timers, Windows Task Scheduler, a CI cron --
invokes ``python -m src.reporting.cli`` on a cadence. That command asks this
module what is due and sends it. The app is the configuration surface; the OS
is the clock.

The one property that actually matters
--------------------------------------
**Idempotence.** A cron entry running every fifteen minutes must send an 08:00
report exactly once, not four times an hour. :func:`is_due` therefore compares
``last_sent`` against the most recent scheduled OCCURRENCE rather than against
the wall clock:

    due  <=>  there is an occurrence at or before now
              AND last_sent is before that occurrence

That formulation also handles a late or missed run correctly: if the machine
was asleep at 08:00 and cron first fires at 11:20, the 08:00 slot is still
unsent, so it goes out once and then stops. No missed report, no duplicate.

Pure module: stdlib only, no I/O and no Streamlit, so every path through the
calendar arithmetic is testable against fixed clocks.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, tzinfo

logger = logging.getLogger(__name__)

__all__ = [
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "FREQUENCIES",
    "WEEKDAY_NAMES",
    "DEFAULT_TIMEZONE",
    "DEFAULT_GRACE_MINUTES",
    "Schedule",
    "resolve_timezone",
    "now_in",
    "previous_occurrence",
    "next_occurrence",
    "is_due",
    "due_schedules",
    "starting_next_slot",
    "describe",
    "valid_email",
    "clean_recipients",
    "SCHEDULE_FIELDS",
    "to_record",
    "from_record",
]

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"

FREQUENCIES: tuple[str, ...] = (DAILY, WEEKLY, MONTHLY)

#: Monday is 0, matching ``datetime.weekday()``.
WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

#: India-first, matching the rest of the application's defaults.
DEFAULT_TIMEZONE = "Asia/Kolkata"

#: How late a run may be and still count as "on time" for reporting purposes.
#: It does NOT gate sending -- a missed slot is always sent once, however late
#: -- it only distinguishes "cron is running normally" from "this was missed".
DEFAULT_GRACE_MINUTES: int = 120

#: Deliberately conservative: a practical subset of RFC 5322 rather than the
#: full grammar. Rejecting a valid-but-exotic address is a minor annoyance;
#: accepting a malformed one means a silent bounce the user never learns about.
_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def valid_email(address: object) -> bool:
    """True for a plausibly deliverable address."""
    text = str(address or "").strip()
    return bool(text) and len(text) <= 254 and bool(_EMAIL.match(text))


def clean_recipients(addresses: object) -> tuple[tuple[str, ...], list[str]]:
    """``(accepted, rejected_messages)`` from a list or comma-separated string.

    Rejections are REPORTED, never dropped: a typo in one address of five
    would otherwise mean one person silently stops receiving the report, which
    is exactly the kind of failure nobody notices for months.
    """
    if addresses is None:
        return (), []
    if isinstance(addresses, str):
        candidates = re.split(r"[,;\s]+", addresses)
    elif isinstance(addresses, Iterable):
        candidates = [str(a) for a in addresses]
    else:
        return (), [f"Recipients could not be read from {type(addresses).__name__}."]

    accepted: list[str] = []
    rejected: list[str] = []
    for raw in candidates:
        address = str(raw).strip().strip("<>")
        if not address:
            continue
        if not valid_email(address):
            rejected.append(f"{address!r} is not a valid email address — skipped.")
            continue
        if address.lower() in {a.lower() for a in accepted}:
            continue
        accepted.append(address)
    return tuple(accepted), rejected


def resolve_timezone(name: object) -> tuple[tzinfo | None, str]:
    """The tzinfo for an IANA name. ``(tzinfo, warning)``.

    ``(None, warning)`` means fall back to the machine's local time. That
    happens when the platform has no timezone database -- Windows ships none,
    so ``zoneinfo`` needs the ``tzdata`` package there. Degrading to local time
    with a stated warning is better than refusing to schedule at all, but the
    warning matters: an "08:00 daily" report firing at 02:30 is useless, and
    the user needs to know which clock is being used.
    """
    label = str(name or "").strip() or DEFAULT_TIMEZONE
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(label), ""
    except Exception as exc:  # noqa: BLE001 - missing tzdata, bad name, old Python
        logger.warning("Timezone %r unavailable (%s); using local time.", label, exc)
        return None, (
            f"Timezone {label!r} is unavailable on this machine ({exc}). Times are "
            "being interpreted in the SERVER's local timezone instead. Install the "
            "`tzdata` package to fix this."
        )


def now_in(timezone_name: object = DEFAULT_TIMEZONE) -> datetime:
    """Current wall-clock time in ``timezone_name``, as a NAIVE datetime.

    Naive on purpose. Every comparison in this module is between two times in
    the same zone, and a naive local time is far easier to reason about (and
    to store in SQLite) than an aware one. The zone is applied here, once, at
    the boundary.
    """
    tz, _warning = resolve_timezone(timezone_name)
    if tz is None:
        return datetime.now()
    return datetime.now(tz).replace(tzinfo=None)


def _clamp_day(year: int, month: int, day: int) -> int:
    """Largest valid day-of-month at or below ``day``.

    A "31st of each month" schedule must fire on 28 or 29 February, not skip
    the month and not raise. Clamping is the only behaviour that keeps a
    monthly report monthly.
    """
    return min(max(int(day), 1), calendar.monthrange(year, month)[1])


def _at(day: datetime, hour: int, minute: int) -> datetime:
    return datetime.combine(day.date(), time(hour=hour, minute=minute))


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Schedule:
    """One recurring tearsheet delivery.

    ``portfolio`` names a portfolio saved in the SQLite store, because a
    scheduled run has no browser session to read holdings from — it has to be
    able to reconstruct the analysis from persisted data alone. A schedule
    pointing at a portfolio that no longer exists is reported by the runner
    rather than silently skipped.
    """

    name: str
    portfolio: str
    recipients: tuple[str, ...] = ()
    frequency: str = WEEKLY
    hour: int = 8
    minute: int = 0
    #: Monday is 0. Used by WEEKLY only.
    weekday: int = 0
    #: Used by MONTHLY only; clamped to the month's length.
    day_of_month: int = 1
    timezone: str = DEFAULT_TIMEZONE
    enabled: bool = True
    last_sent: datetime | None = None
    #: Analysis settings the runner reproduces.
    period: str = "1y"
    benchmark: str = "NIFTY 50"
    base_currency: str = "INR"
    attach_excel: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "portfolio", str(self.portfolio or "").strip())
        frequency = str(self.frequency or "").strip().lower()
        object.__setattr__(self, "frequency", frequency if frequency in FREQUENCIES else WEEKLY)
        object.__setattr__(self, "hour", min(max(int(self.hour or 0), 0), 23))
        object.__setattr__(self, "minute", min(max(int(self.minute or 0), 0), 59))
        object.__setattr__(self, "weekday", min(max(int(self.weekday or 0), 0), 6))
        object.__setattr__(self, "day_of_month", min(max(int(self.day_of_month or 1), 1), 31))
        accepted, _rejected = clean_recipients(self.recipients)
        object.__setattr__(self, "recipients", accepted)

    @property
    def ok(self) -> bool:
        """Usable: named, pointing at a portfolio, with somewhere to send it."""
        return bool(self.name and self.portfolio and self.recipients)

    @property
    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.name:
            out.append("The schedule has no name.")
        if not self.portfolio:
            out.append("No saved portfolio is selected, so there is nothing to report on.")
        if not self.recipients:
            out.append("No valid recipient address.")
        return tuple(out)

    def mark_sent(self, when: datetime | None) -> Schedule:
        """A copy with ``last_sent`` set to ``when``. Never mutates.

        ``None`` is accepted and means "never sent", which is what resets a
        schedule so its next occurrence is delivered again.
        """
        return replace(self, last_sent=when)


# ---------------------------------------------------------------------------
# occurrences
# ---------------------------------------------------------------------------
def previous_occurrence(schedule: Schedule, now: datetime) -> datetime | None:
    """The most recent scheduled moment at or before ``now``.

    This, not the wall clock, is what ``last_sent`` is compared against —
    which is what makes sending idempotent under a cron that fires far more
    often than the schedule does.
    """
    if schedule.frequency == DAILY:
        today = _at(now, schedule.hour, schedule.minute)
        return today if today <= now else today - timedelta(days=1)

    if schedule.frequency == WEEKLY:
        candidate = _at(now, schedule.hour, schedule.minute)
        delta = (now.weekday() - schedule.weekday) % 7
        candidate -= timedelta(days=delta)
        if candidate > now:
            candidate -= timedelta(days=7)
        return candidate

    if schedule.frequency == MONTHLY:
        day = _clamp_day(now.year, now.month, schedule.day_of_month)
        candidate = datetime(now.year, now.month, day, schedule.hour, schedule.minute)
        if candidate <= now:
            return candidate
        # Step into the previous month and clamp again: 31 January back from
        # 15 February is 31 January, not an invalid 31st of a 28-day month.
        year = now.year - 1 if now.month == 1 else now.year
        month = 12 if now.month == 1 else now.month - 1
        day = _clamp_day(year, month, schedule.day_of_month)
        return datetime(year, month, day, schedule.hour, schedule.minute)

    return None  # pragma: no cover - __post_init__ constrains the frequency


def next_occurrence(schedule: Schedule, now: datetime) -> datetime | None:
    """The next scheduled moment strictly after ``now``. For display only."""
    if schedule.frequency == DAILY:
        candidate = _at(now, schedule.hour, schedule.minute)
        return candidate if candidate > now else candidate + timedelta(days=1)

    if schedule.frequency == WEEKLY:
        candidate = _at(now, schedule.hour, schedule.minute)
        delta = (schedule.weekday - now.weekday()) % 7
        candidate += timedelta(days=delta)
        if candidate <= now:
            candidate += timedelta(days=7)
        return candidate

    if schedule.frequency == MONTHLY:
        day = _clamp_day(now.year, now.month, schedule.day_of_month)
        candidate = datetime(now.year, now.month, day, schedule.hour, schedule.minute)
        if candidate > now:
            return candidate
        year = now.year + 1 if now.month == 12 else now.year
        month = 1 if now.month == 12 else now.month + 1
        day = _clamp_day(year, month, schedule.day_of_month)
        return datetime(year, month, day, schedule.hour, schedule.minute)

    return None  # pragma: no cover


def is_due(schedule: Schedule, now: datetime | None = None) -> bool:
    """Whether ``schedule`` should be sent at ``now``.

    The idempotence guarantee:

        due  <=>  enabled AND usable
                  AND an occurrence exists at or before now
                  AND last_sent is before that occurrence

    Comparing against the OCCURRENCE rather than the clock is what lets a
    fifteen-minute cron send an 08:00 report once. It also means a missed slot
    is sent once when the machine comes back, however late — better late than
    never, but never twice.
    """
    if not schedule.enabled or not schedule.ok:
        return False
    moment = now if now is not None else now_in(schedule.timezone)
    occurrence = previous_occurrence(schedule, moment)
    if occurrence is None:
        return False
    return schedule.last_sent is None or schedule.last_sent < occurrence


def starting_next_slot(schedule: Schedule, now: datetime | None = None) -> Schedule:
    """A copy whose first delivery is its NEXT slot, not an immediate one.

    Without this, a schedule created at 14:00 saying "daily at 08:00" is
    instantly due -- 08:00 has already passed today and nothing has been sent
    for it -- so the next cron tick would mail a report seven minutes after
    the user pressed Save. Technically correct, and surprising enough that it
    reads as a bug.

    So a newly created schedule is stamped as though its most recent past
    occurrence had already been delivered. Note this is applied only when a
    schedule is CREATED: it must never be applied on load, or every restart
    would swallow whatever was outstanding.
    """
    moment = now if now is not None else now_in(schedule.timezone)
    return schedule.mark_sent(previous_occurrence(schedule, moment))


def due_schedules(schedules: Sequence[Schedule], now: datetime | None = None) -> list[Schedule]:
    """Every schedule ready to send, earliest occurrence first.

    Ordered so that if the runner is interrupted part-way through a backlog,
    the oldest outstanding report has already gone out.
    """
    moment = now
    ready = [s for s in schedules if is_due(s, moment or now_in(s.timezone))]
    ready.sort(
        key=lambda s: (
            previous_occurrence(s, moment or now_in(s.timezone)) or datetime.min,
            s.name,
        )
    )
    return ready


def describe(schedule: Schedule, now: datetime | None = None) -> str:
    """A one-line human description, including when it next fires."""
    moment = now if now is not None else now_in(schedule.timezone)
    clock = f"{schedule.hour:02d}:{schedule.minute:02d}"

    if schedule.frequency == DAILY:
        cadence = f"every day at {clock}"
    elif schedule.frequency == WEEKLY:
        cadence = f"every {WEEKDAY_NAMES[schedule.weekday]} at {clock}"
    else:
        cadence = f"on day {schedule.day_of_month} of each month at {clock}"

    upcoming = next_occurrence(schedule, moment)
    when = f", next {upcoming:%d %b %Y %H:%M}" if upcoming else ""
    state = "" if schedule.enabled else " (paused)"
    return f"{cadence} {schedule.timezone}{when}{state}"


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
#: The persisted field order. Kept here rather than in the store because a
#: dict <-> dataclass conversion is pure: the store deals in plain records and
#: never imports this module, so the data layer stays free of any dependency
#: on the reporting layer.
SCHEDULE_FIELDS: tuple[str, ...] = (
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


def to_record(schedule: Schedule) -> dict[str, object]:
    """A flat, SQLite-friendly record. Recipients join with ", ".

    Note what is NOT here: no password, no SMTP host, no credential of any
    kind. A schedule says *when* and *to whom*; how to reach the mail server is
    read from the environment at send time and never persisted.
    """
    return {
        "name": schedule.name,
        "portfolio": schedule.portfolio,
        "recipients": ", ".join(schedule.recipients),
        "frequency": schedule.frequency,
        "hour": int(schedule.hour),
        "minute": int(schedule.minute),
        "weekday": int(schedule.weekday),
        "day_of_month": int(schedule.day_of_month),
        "timezone": schedule.timezone,
        "enabled": bool(schedule.enabled),
        "last_sent": schedule.last_sent.isoformat(timespec="seconds")
        if schedule.last_sent is not None
        else "",
        "period": schedule.period,
        "benchmark": schedule.benchmark,
        "base_currency": schedule.base_currency,
        "attach_excel": bool(schedule.attach_excel),
    }


def _parse_datetime(value: object) -> datetime | None:
    """A stored timestamp, or ``None``. Never raises on a corrupt value.

    An unreadable ``last_sent`` reads as "never sent", which risks ONE
    duplicate e-mail. The alternative default -- treating it as "just sent" --
    would silently suppress the report indefinitely, and a scheduled report
    that stops arriving is the failure nobody notices.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Unreadable last_sent %r; treating the slot as unsent.", text)
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _parse_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _parse_int(value: object, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def from_record(record: object) -> Schedule:
    """Rebuild a :class:`Schedule` from a stored record.

    Tolerant by design: a missing or unparseable field falls back to the
    dataclass default rather than raising, because one corrupt row must not
    stop every other schedule from running. ``__post_init__`` then clamps
    whatever survives into range.
    """
    data: dict[str, object] = dict(record) if isinstance(record, dict) else {}
    return Schedule(
        name=str(data.get("name") or ""),
        portfolio=str(data.get("portfolio") or ""),
        recipients=clean_recipients(data.get("recipients"))[0],
        frequency=str(data.get("frequency") or WEEKLY),
        hour=_parse_int(data.get("hour"), 8),
        minute=_parse_int(data.get("minute"), 0),
        weekday=_parse_int(data.get("weekday"), 0),
        day_of_month=_parse_int(data.get("day_of_month"), 1),
        timezone=str(data.get("timezone") or DEFAULT_TIMEZONE),
        enabled=_parse_bool(data.get("enabled"), True),
        last_sent=_parse_datetime(data.get("last_sent")),
        period=str(data.get("period") or "1y"),
        benchmark=str(data.get("benchmark") or "NIFTY 50"),
        base_currency=str(data.get("base_currency") or "INR"),
        attach_excel=_parse_bool(data.get("attach_excel"), False),
    )
