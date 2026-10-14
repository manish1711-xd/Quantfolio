"""Price alerts on watchlist targets: the state machine, and what it refuses to claim.

An alert is not the same question as :func:`src.watchlist.triggered`. That
function reports *proximity* — ``|to_target_pct| <= tolerance`` — and its
docstring says exactly why it can do no better: the watchlist stored only a
target, never which side of the price that target was on, so it could not tell
an upside level the price rose to from an entry level it fell to. This module
supplies the missing half. Every alert carries a **direction**, so "notify me
when RELIANCE rises to 1,350" and "notify me when it falls to 1,200" are
different rules with different answers.

The two properties that make an alert an alert
----------------------------------------------
**It fires on a transition, not on a condition.** A rule that e-mailed whenever
``price >= level`` would send every fifteen minutes for as long as the price
stayed there, and a notification that arrives ninety times is not a
notification. So each rule is a two-state machine — ``armed`` then
``triggered`` — and only the ``armed -> triggered`` edge sends anything.

**It re-arms on retreat, not on a timer.** Coming back to ``armed`` the instant
the price dips a paisa below the level would produce a burst of alerts from one
price wobbling on the level. Re-arming therefore requires the price to retreat
past the level by ``rearm_pct`` — plain hysteresis — *and* a minimum interval
between two fires for the same instrument, which catches a genuinely volatile
price crossing even a wide band repeatedly in one session. The band stops
flapping; the cooldown bounds the worst case.

What an alert here cannot tell you
----------------------------------
Stated in the module that computes it, not left to the UI:

* **The feed is delayed.** NSE/BSE quotes run ~15 minutes behind, so an alert
  means "your level had been reached as of this timestamp", never "is being
  reached now". Every event carries the quote's own ``as_of``.
* **Between two checks, only the session range is evidence.** The runner sees
  samples, not a continuous price. A spike that crossed the level and came
  back between two polls is invisible in the last price — which is why
  ``day_high`` / ``day_low`` are consulted as well, and why every event says
  which of the two saw the crossing. A crossing that happened entirely inside
  an *earlier*, already-closed session is simply not recoverable from a quote.
* **It is arithmetic against a number the user typed.** No target is ever
  suggested, no rule is ranked as more attractive than another, and reaching a
  level is not a reason to trade.

Pure module: pandas and stdlib only, no Streamlit and no network. Quotes are
passed in, so every branch of the state machine is testable against hand-built
fixtures at a fixed clock.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.models import Quote
from src.watchlist import WatchItem

__all__ = [
    "ABOVE",
    "BELOW",
    "DIRECTIONS",
    "DIRECTION_LABELS",
    "ARMED",
    "TRIGGERED",
    "STATES",
    "EVIDENCE_LAST",
    "EVIDENCE_HIGH",
    "EVIDENCE_LOW",
    "EVENT_COLUMNS",
    "DEFAULT_REARM_PCT",
    "DEFAULT_COOLDOWN_HOURS",
    "AlertSettings",
    "AlertEvent",
    "AlertOutcome",
    "normalise_direction",
    "infer_direction",
    "arm",
    "crossing",
    "rearmed",
    "evaluate",
    "events_frame",
    "describe_rule",
    "alert_state_table",
]

#: The price rose to (or through) the level.
ABOVE = "above"
#: The price fell to (or through) the level.
BELOW = "below"

DIRECTIONS: tuple[str, ...] = (ABOVE, BELOW)

#: Phrased as the user would say it, because "above/below" alone is ambiguous
#: about whether it describes the level or the move.
DIRECTION_LABELS: dict[str, str] = {
    ABOVE: "rises to or above",
    BELOW: "falls to or below",
}

#: Waiting for the level to be reached.
ARMED = "armed"
#: The level has been reached; nothing more will be sent until it re-arms.
TRIGGERED = "triggered"

STATES: tuple[str, ...] = (ARMED, TRIGGERED)

#: Which observation saw the crossing. Reported on every event because "the
#: last price is through your level" and "the level was touched at some point
#: in this session" are different facts, and a user acting on the second one
#: should know the price may have already come back.
EVIDENCE_LAST = "last price"
EVIDENCE_HIGH = "session high"
EVIDENCE_LOW = "session low"

#: Column contract of :func:`events_frame`.
EVENT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "direction",
    "level",
    "price",
    "currency",
    "through_by_pct",
    "evidence",
    "as_of",
    "is_intraday",
    "note",
)

#: How far the price must retreat back past the level before the rule re-arms,
#: as a percentage of the level. One percent is wide enough to absorb the
#: tick-by-tick noise of a price sitting on a round number and narrow enough
#: that a genuine reversal still re-arms the same day.
DEFAULT_REARM_PCT: float = 1.0

#: Minimum hours between two fires for the same instrument. A second guard
#: behind the re-arm band, for a price volatile enough to clear even that.
DEFAULT_COOLDOWN_HOURS: float = 12.0

_NAN = float("nan")


def _nan() -> float:
    return _NAN


def _as_float(value: object) -> float | None:
    """Finite float, or ``None``. Never raises, never returns NaN."""
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
    text = str(value).strip()
    return "" if text.upper() in {"NAN", "NONE", "NAT", "N/A", "NULL"} else text


def _clean_recipients(addresses: object) -> tuple[tuple[str, ...], list[str]]:
    """Validated e-mail addresses, delegated to :mod:`src.reporting.schedule`.

    Imported inside the function on purpose. ``src.alerts`` is a domain module
    like :mod:`src.watchlist`, and ``src.reporting`` is an output layer, so
    importing it at module scope would make every module that touches a
    watchlist depend on the reporting stack. Address validation lives there
    because that is where it was first needed, and one indirection is cheaper
    than either duplicating the rule or inverting the layering.
    """
    from src.reporting.schedule import clean_recipients

    return clean_recipients(addresses)


def _naive(moment: object) -> datetime | None:
    """A tz-naive datetime, or ``None``. Comparisons must not mix awareness."""
    if not isinstance(moment, datetime):
        return None
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------
def normalise_direction(value: object) -> str:
    """``ABOVE``, ``BELOW``, or ``""`` for anything unrecognised.

    Accepts the words a user or an older stored row might carry — "up",
    "rise", "over" all mean :data:`ABOVE` — because a direction that fails to
    parse silently disarms an alert the user believes is running.
    """
    text = _as_text(value).lower()
    if text in {ABOVE, "up", "rise", "rises", "over", "gte", ">=", ">"}:
        return ABOVE
    if text in {BELOW, "down", "fall", "falls", "under", "lte", "<=", "<"}:
        return BELOW
    return ""


def infer_direction(level: object, reference_price: object) -> str:
    """The direction a target implies, given the price when it was set.

    A target above the price at the time is an upside level to rise to; below
    it, an entry level to fall to. This is the inference the watchlist could
    not make before, because it never recorded the reference price.

    Returns ``""`` when there is nothing to infer from — no reference price,
    or a target sitting exactly on it. Guessing in that case would attach a
    direction the user never chose to a rule that then quietly never fires.
    """
    target = _as_float(level)
    reference = _as_float(reference_price)
    if target is None or reference is None or target <= 0 or reference <= 0:
        return ""
    if target > reference:
        return ABOVE
    if target < reference:
        return BELOW
    return ""


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlertSettings:
    """How alerts are delivered, and how hard they are to re-trigger.

    Deliberately holds no SMTP credential: like
    :class:`src.reporting.schedule.Schedule`, this says who to notify, and the
    mail server is read from the environment at send time.
    """

    enabled: bool = False
    #: Always a validated tuple after construction. ``__post_init__`` also
    #: accepts a comma-separated string defensively, but :meth:`parse` is the
    #: declared way in for stored or user-typed data -- keeping the field's
    #: own type honest is what lets every reader downstream rely on a tuple.
    recipients: tuple[str, ...] = ()
    rearm_pct: float = DEFAULT_REARM_PCT
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS
    #: Consult ``day_high`` / ``day_low`` as well as the last price, so a
    #: crossing between two checks is not missed. Every event says which
    #: observation saw it.
    use_session_range: bool = True
    timezone: str = "Asia/Kolkata"

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        rearm = _as_float(self.rearm_pct)
        object.__setattr__(
            self,
            "rearm_pct",
            min(max(rearm if rearm is not None else DEFAULT_REARM_PCT, 0.0), 50.0),
        )
        cooldown = _as_float(self.cooldown_hours)
        object.__setattr__(
            self,
            "cooldown_hours",
            min(max(cooldown if cooldown is not None else DEFAULT_COOLDOWN_HOURS, 0.0), 24.0 * 30),
        )
        object.__setattr__(self, "use_session_range", bool(self.use_session_range))
        object.__setattr__(self, "timezone", _as_text(self.timezone) or "Asia/Kolkata")
        object.__setattr__(self, "recipients", _clean_recipients(self.recipients)[0])

    @classmethod
    def parse(
        cls,
        enabled: object = False,
        recipients: object = (),
        rearm_pct: object = DEFAULT_REARM_PCT,
        cooldown_hours: object = DEFAULT_COOLDOWN_HOURS,
        use_session_range: object = True,
        timezone: object = "Asia/Kolkata",
    ) -> AlertSettings:
        """Build from untyped values -- a stored record, or a form.

        The declared entry point for anything that came out of SQLite or off a
        widget, where the recipients arrive as a comma-separated string and
        the numbers may be text. Coercing here is what lets the dataclass
        itself promise a validated tuple and real floats.
        """

        def number(value: object, default: float) -> float:
            coerced = _as_float(value)
            return coerced if coerced is not None else default

        return cls(
            enabled=bool(enabled),
            recipients=_clean_recipients(recipients)[0],
            rearm_pct=number(rearm_pct, DEFAULT_REARM_PCT),
            cooldown_hours=number(cooldown_hours, DEFAULT_COOLDOWN_HOURS),
            use_session_range=bool(use_session_range),
            timezone=_as_text(timezone) or "Asia/Kolkata",
        )

    @property
    def ok(self) -> bool:
        """Deliverable: switched on, with somewhere to send."""
        return bool(self.enabled and self.recipients)

    @property
    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.enabled:
            out.append("Alert e-mails are switched off.")
        if not self.recipients:
            out.append("No valid recipient address for alerts.")
        return tuple(out)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlertEvent:
    """One ``armed -> triggered`` transition, ready to be reported."""

    ticker: str
    direction: str
    level: float
    price: float
    currency: str = ""
    #: How far past the level the observation is, signed in the direction of
    #: the rule. Always >= 0 for a genuine crossing.
    through_by_pct: float = _NAN
    evidence: str = EVIDENCE_LAST
    as_of: datetime | None = None
    is_intraday: bool = False
    note: str = ""

    @property
    def headline(self) -> str:
        """One line, naming the evidence rather than implying a live cross."""
        moved = DIRECTION_LABELS.get(self.direction, self.direction)
        amount = (
            f" ({self.through_by_pct:+.2f}% past it)"
            if self.through_by_pct == self.through_by_pct
            else ""
        )
        currency = f"{self.currency} " if self.currency else ""
        return (
            f"{self.ticker} {moved} {currency}{self.level:,.2f} — "
            f"{self.evidence} {currency}{self.price:,.2f}{amount}"
        )


@dataclass(frozen=True)
class AlertOutcome:
    """The result of one evaluation pass.

    ``items`` is the full watchlist with state advanced, which the caller
    persists. ``events`` is what to notify. ``skipped`` names every rule that
    could NOT be evaluated and why — a rule that silently fails to evaluate is
    indistinguishable from one that is not triggered, and that is precisely
    the failure an alert must never have.
    """

    events: tuple[AlertEvent, ...] = ()
    items: tuple[WatchItem, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    rearmed: tuple[str, ...] = ()
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        """Whether any rule's state moved, i.e. whether ``items`` needs saving."""
        return bool(self.events or self.rearmed)

    @property
    def n_evaluated(self) -> int:
        return sum(1 for item in self.items if item.alert_enabled and item.has_target) - len(
            self.skipped
        )


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------
def arm(item: WatchItem, price: object, direction: object = None) -> WatchItem:
    """Prepare ``item`` for alerting from the price it is being set at.

    Two jobs, both there to stop a surprise e-mail:

    * It records ``reference_price``, which is what lets
      :func:`infer_direction` know later whether the target was an upside
      level or an entry level.
    * If the price is **already** past the level, the rule starts
      ``triggered`` rather than ``armed``, so it does not immediately announce
      a crossing that happened before the alert existed. The UI can then say
      "already through your level; this will fire when it re-crosses", which
      is the truth. This mirrors
      :func:`src.reporting.schedule.starting_next_slot` — a rule created now
      should describe the future, not the past.
    """
    current = _as_float(price)
    chosen = normalise_direction(direction) if direction is not None else ""
    if not chosen:
        chosen = normalise_direction(item.direction) or infer_direction(item.target_price, current)

    state = ARMED
    if item.target_price is not None and current is not None and chosen:
        hit, _evidence = crossing(chosen, item.target_price, current)
        if hit:
            state = TRIGGERED

    return replace(
        item,
        direction=chosen,
        reference_price=current,
        alert_state=state,
        last_fired=None,
    )


def crossing(
    direction: object,
    level: object,
    price: object,
    day_high: object = None,
    day_low: object = None,
    use_session_range: bool = True,
) -> tuple[bool, str]:
    """Has the level been reached? ``(hit, evidence)``.

    The last price is checked first, because "it is there now" is the
    strongest statement available. Only if that misses is the session range
    consulted, and then the evidence is labelled honestly as the session high
    or low: the level was touched, and the price may since have come back.

    ``>=`` and ``<=`` rather than strict inequalities — a price that lands
    exactly on the level has reached it, and a user who typed a round number
    is describing the most likely print of the day.
    """
    rule = normalise_direction(direction)
    target = _as_float(level)
    last = _as_float(price)
    if not rule or target is None or target <= 0:
        return False, ""

    if last is not None:
        if rule == ABOVE and last >= target:
            return True, EVIDENCE_LAST
        if rule == BELOW and last <= target:
            return True, EVIDENCE_LAST

    if not use_session_range:
        return False, ""

    if rule == ABOVE:
        high = _as_float(day_high)
        if high is not None and high >= target:
            return True, EVIDENCE_HIGH
    else:
        low = _as_float(day_low)
        if low is not None and low <= target:
            return True, EVIDENCE_LOW
    return False, ""


def rearmed(
    direction: object, level: object, price: object, rearm_pct: float = DEFAULT_REARM_PCT
) -> bool:
    """Whether a triggered rule should go back to waiting.

    Requires the price to have retreated past the level by ``rearm_pct`` of
    the level, not merely to have crossed back. Without that band, one price
    oscillating around a round number re-arms and re-fires repeatedly, and the
    user learns to ignore the alerts — which is a worse outcome than not
    having them.

    Only the last price is used here, never the session range: re-arming on a
    session low would undo a trigger the same session produced.
    """
    rule = normalise_direction(direction)
    target = _as_float(level)
    last = _as_float(price)
    if not rule or target is None or target <= 0 or last is None:
        return False

    band = target * (max(float(rearm_pct), 0.0) / 100.0)
    if rule == ABOVE:
        return last < target - band
    return last > target + band


def _quote_fields(
    quote: object,
) -> tuple[float | None, float | None, float | None, str, datetime | None, bool]:
    price = _as_float(getattr(quote, "price", None))
    high = _as_float(getattr(quote, "day_high", None))
    low = _as_float(getattr(quote, "day_low", None))
    currency = _as_text(getattr(quote, "currency", None))
    as_of = _naive(getattr(quote, "as_of", None))
    intraday = bool(getattr(quote, "is_intraday", False))
    return price, high, low, currency, as_of, intraday


def evaluate(
    items: Sequence[WatchItem],
    quotes: Mapping[str, Quote] | None = None,
    settings: AlertSettings | None = None,
    now: datetime | None = None,
) -> AlertOutcome:
    """Advance every alert rule one step against the latest quotes.

    Pure: takes items and quotes, returns new items and the events to send.
    Nothing is e-mailed, nothing is persisted and nothing is mutated here, so
    the whole state machine is testable at a fixed clock.

    Rules with no target, or with alerting switched off, are passed through
    untouched. Rules that cannot be evaluated — no quote, no direction — are
    reported in ``skipped`` with a reason, never treated as "not triggered".
    """
    config = settings if settings is not None else AlertSettings()
    moment = _naive(now) or datetime.now(timezone.utc).replace(tzinfo=None)
    quote_map = {str(k).strip().upper(): v for k, v in (quotes or {}).items()}

    events: list[AlertEvent] = []
    updated: list[WatchItem] = []
    skipped: list[tuple[str, str]] = []
    woke: list[str] = []
    notices: list[str] = []

    for item in items:
        if not item.alert_enabled or not item.has_target:
            updated.append(item)
            continue

        quote = quote_map.get(item.ticker)
        if quote is None:
            skipped.append(
                (item.ticker, "no quote was available, so the alert could not be checked")
            )
            updated.append(item)
            continue

        price, high, low, currency, as_of, intraday = _quote_fields(quote)
        if price is None:
            skipped.append((item.ticker, "the quote carried no usable price"))
            updated.append(item)
            continue

        rule = normalise_direction(item.direction) or infer_direction(
            item.target_price, item.reference_price
        )
        if not rule:
            skipped.append(
                (
                    item.ticker,
                    "no direction is set, so it is unclear whether the target is a "
                    "level to rise to or to fall to — set one to activate the alert",
                )
            )
            updated.append(item)
            continue

        # has_target above already established this is a positive float;
        # mypy cannot carry that through the property.
        level = float(item.target_price)  # type: ignore[arg-type]
        state = _as_text(item.alert_state).lower() or ARMED

        if state == TRIGGERED:
            if rearmed(rule, level, price, config.rearm_pct):
                updated.append(replace(item, alert_state=ARMED))
                woke.append(item.ticker)
            else:
                updated.append(item)
            continue

        hit, evidence = crossing(
            rule, level, price, high, low, use_session_range=config.use_session_range
        )
        if not hit:
            updated.append(item)
            continue

        # Cooldown is checked only on the firing edge, so a suppressed alert
        # still advances to `triggered`: the level HAS been reached, and
        # leaving it armed would fire the moment the cooldown lapsed, long
        # after the crossing was news.
        last_fired = _naive(item.last_fired)
        if (
            last_fired is not None
            and config.cooldown_hours > 0
            and moment - last_fired < timedelta(hours=config.cooldown_hours)
        ):
            notices.append(
                f"{item.ticker} reached your level again within "
                f"{config.cooldown_hours:g}h of the last alert, so no second e-mail "
                "was sent."
            )
            updated.append(replace(item, alert_state=TRIGGERED))
            continue

        observed = (
            price if evidence == EVIDENCE_LAST else (high if evidence == EVIDENCE_HIGH else low)
        )
        observed_price = float(observed) if observed is not None else price
        through = (
            (observed_price / level - 1.0) * 100.0
            if rule == ABOVE
            else (1.0 - observed_price / level) * 100.0
        )

        events.append(
            AlertEvent(
                ticker=item.ticker,
                direction=rule,
                level=level,
                price=observed_price,
                currency=currency,
                through_by_pct=float(through),
                evidence=evidence,
                as_of=as_of,
                is_intraday=intraday,
                note=item.note,
            )
        )
        updated.append(replace(item, alert_state=TRIGGERED, last_fired=moment, direction=rule))

    return AlertOutcome(
        events=tuple(events),
        items=tuple(updated),
        skipped=tuple(skipped),
        rearmed=tuple(woke),
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# presentation helpers (still pure)
# ---------------------------------------------------------------------------
def events_frame(events: Iterable[AlertEvent]) -> pd.DataFrame:
    """Events as a frame with the :data:`EVENT_COLUMNS` contract."""
    rows = [
        {
            "ticker": event.ticker,
            "direction": event.direction,
            "level": event.level,
            "price": event.price,
            "currency": event.currency,
            "through_by_pct": event.through_by_pct,
            "evidence": event.evidence,
            "as_of": event.as_of,
            "is_intraday": event.is_intraday,
            "note": event.note,
        }
        for event in events
    ]
    if not rows:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in EVENT_COLUMNS})
    frame = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    for numeric in ("level", "price", "through_by_pct"):
        frame[numeric] = pd.to_numeric(frame[numeric], errors="coerce").astype("float64")
    return frame


def describe_rule(item: WatchItem) -> str:
    """One human line for a rule, or why it will not fire."""
    if not item.has_target:
        return "No target set."
    level = float(item.target_price)  # type: ignore[arg-type]
    rule = normalise_direction(item.direction) or infer_direction(
        item.target_price, item.reference_price
    )
    if not item.alert_enabled:
        return f"Target {level:,.2f} — no e-mail (alerting off for this instrument)."
    if not rule:
        return f"Target {level:,.2f} — inactive: no direction set."
    phrase = DIRECTION_LABELS[rule]
    if _as_text(item.alert_state).lower() == TRIGGERED:
        return (
            f"Already through {level:,.2f} — will e-mail again once the price "
            f"retreats and {phrase} {level:,.2f} afresh."
        )
    return f"Waiting: e-mail when the price {phrase} {level:,.2f}."


def alert_state_table(items: Sequence[WatchItem]) -> pd.DataFrame:
    """One row per watchlist entry describing its alert rule, for display."""
    rows = [
        {
            "ticker": item.ticker,
            "alert": bool(item.alert_enabled and item.has_target),
            "direction": normalise_direction(item.direction)
            or infer_direction(item.target_price, item.reference_price),
            "level": item.target_price if item.target_price is not None else _nan(),
            "state": _as_text(item.alert_state).lower() or ARMED,
            "last_fired": item.last_fired,
            "rule": describe_rule(item),
        }
        for item in items
    ]
    columns = ["ticker", "alert", "direction", "level", "state", "last_fired", "rule"]
    if not rows:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    frame = pd.DataFrame(rows, columns=columns)
    frame["level"] = pd.to_numeric(frame["level"], errors="coerce").astype("float64")
    frame["alert"] = frame["alert"].astype(bool)
    return frame
