"""Drawdown analytics: the depth, duration and recovery of every decline.

Maximum drawdown is a single number and it hides almost everything an
investor cares about. This module decomposes a value series into the full
sequence of peak -> trough -> recovery *episodes*, so the UI can answer the
questions a headline "-32%" cannot: how long did it take to get there, how
long did it take to come back, how much of the period was spent underwater,
and is the portfolio still in a hole right now.

Conventions
-----------
Drawdown        d_t = V_t / running_max(V) - 1   (<= 0, a NEGATIVE decimal)
                imported from ``src.analytics.risk`` and re-exported here so
                the two modules cannot drift apart.
Episode         opens on the first observation strictly BELOW the running
                peak, tracks the lowest value reached (the trough), and closes
                only when the series makes a NEW HIGH (>= the prior peak).
                An episode still below its peak at the end of the series is
                reported with ``recovery_date=None`` and
                ``recovery_days=None`` -- it is censored, not recovered, and
                pretending otherwise flatters the track record.
Day counts      CALENDAR days taken from the index (a 21-trading-day drawdown
                spans ~30 calendar days). Reported as ``int``.
Ordering        episodes come back DEEPEST FIRST (``depth`` ascending, and
                ``depth`` is negative).
Degenerate in   every public function returns ``float("nan")``, ``None`` or an
                empty Series/DataFrame/list. Nothing here raises.

A date-like index is required for episode extraction, because a calendar-day
count is meaningless over a positional index; ``drawdown_episodes`` returns an
empty list rather than inventing epoch dates from integers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.risk import drawdown_series
from src.models import DrawdownEpisode

__all__ = [
    "DEFAULT_TOP_N",
    "DEFAULT_MIN_DEPTH",
    "EPISODE_COLUMNS",
    "drawdown_series",
    "underwater_days",
    "drawdown_episodes",
    "episodes_dataframe",
    "time_underwater_pct",
    "current_drawdown",
    "average_recovery_days",
]

# ---------------------------------------------------------------------------
# presentation defaults
# ---------------------------------------------------------------------------
#: How many episodes the UI shows by default. This is a *display* choice, not
#: a statistical threshold, which is why it lives here rather than in
#: ``src.config`` -- and why it is a parameter on every call site.
DEFAULT_TOP_N: int = 5

#: Declines shallower than 1% are noise in a daily series (a single quiet
#: session) and would swamp the table; also a display filter, also overridable.
DEFAULT_MIN_DEPTH: float = 0.01

#: Column order of :func:`episodes_dataframe`, stable for the UI and Excel.
EPISODE_COLUMNS: tuple[str, ...] = (
    "peak_date",
    "trough_date",
    "recovery_date",
    "depth",
    "depth_pct",
    "drawdown_days",
    "recovery_days",
    "total_days",
    "recovered",
)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> float:
    """Best-effort float coercion; NaN for anything unparseable or infinite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _clean_values(values: Any) -> pd.Series:
    """Copy-coerce a value series to float, drop NaN/inf, sort a date index.

    Never mutates the caller's object (``copy=True``). Mirrors the helper in
    ``src.analytics.risk`` so both modules digest the same inputs the same way.
    """
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    try:
        series = pd.Series(values, copy=True)
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        series = series.astype("float64")
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    index = series.index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def _index_dates(index: pd.Index) -> list[date] | None:
    """Calendar dates behind a series index, or None when there are none.

    A positional/integer index is rejected outright: coercing 0, 1, 2 into
    1970-01-01 nanosecond stamps would produce day counts that look precise
    and are entirely fictional.
    """
    if index is None or len(index) == 0:
        return None
    if isinstance(index, pd.DatetimeIndex):
        stamps: pd.Index = index
    elif pd.api.types.is_numeric_dtype(index) or pd.api.types.is_bool_dtype(index):
        return None
    else:
        try:
            stamps = pd.to_datetime(pd.Index(index), errors="coerce")
        except (TypeError, ValueError):
            return None
    try:
        if pd.isna(stamps).any():
            return None
        return [pd.Timestamp(stamp).date() for stamp in stamps]
    except (TypeError, ValueError, AttributeError):
        return None


def _peak_positions(values: pd.Series) -> np.ndarray:
    """For every observation, the position of the running peak in force.

    Position ``i`` is its own peak whenever ``V_i`` equals the running maximum
    (i.e. the series is at a new -- or equal -- high); otherwise it inherits
    the position of the most recent high. The first observation is always a
    peak by definition.
    """
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    at_high = values.to_numpy(dtype=float) >= values.cummax().to_numpy(dtype=float)
    at_high[0] = True
    positions = np.arange(n, dtype=float)
    marked = pd.Series(np.where(at_high, positions, np.nan))
    return marked.ffill().to_numpy(dtype=float).astype(np.int64)


def _episode(
    dates: Sequence[date],
    peak_pos: int,
    trough_pos: int,
    peak_value: float,
    trough_value: float,
    recovery_pos: int | None,
) -> DrawdownEpisode | None:
    """Build one :class:`DrawdownEpisode`; None when the depth is undefined.

    Depth is ``trough / peak - 1``; a non-positive or non-finite peak makes
    that ratio meaningless, so the episode is dropped rather than reported as
    a fabricated number.
    """
    if not math.isfinite(peak_value) or peak_value <= 0.0:
        return None
    depth = _to_float(trough_value / peak_value - 1.0)
    if not math.isfinite(depth):
        return None
    recovery_date = dates[recovery_pos] if recovery_pos is not None else None
    recovery_days = (
        int((dates[recovery_pos] - dates[trough_pos]).days) if recovery_pos is not None else None
    )
    return DrawdownEpisode(
        peak_date=dates[peak_pos],
        trough_date=dates[trough_pos],
        recovery_date=recovery_date,
        depth=depth,
        drawdown_days=int((dates[trough_pos] - dates[peak_pos]).days),
        recovery_days=recovery_days,
    )


# ---------------------------------------------------------------------------
# underwater profile
# ---------------------------------------------------------------------------
def underwater_days(values: pd.Series) -> pd.Series:
    """Consecutive days spent below the running peak, aligned to the index.

        u_t = (t - t_peak) in calendar days,  0 whenever V_t is a new high

    where ``t_peak`` is the date of the most recent running maximum. The
    counter resets to 0 the moment the series makes a new high, so plotting it
    gives the classic "time underwater" saw-tooth beneath the equity curve.

    Calendar days are used when the index is date-like; over a positional
    index the count degrades to a number of OBSERVATIONS (stated here because
    the two are not interchangeable). Returns an EMPTY Series for empty input;
    never raises.
    """
    v = _clean_values(values)
    if v.empty:
        return pd.Series(dtype="int64", name="underwater_days")

    peaks = _peak_positions(v)
    dates = _index_dates(v.index)
    if dates is None:
        counts = np.arange(len(v), dtype=np.int64) - peaks
    else:
        ordinals = np.array([d.toordinal() for d in dates], dtype=np.int64)
        counts = ordinals - ordinals[peaks]
    return pd.Series(counts.astype("int64"), index=v.index, name="underwater_days")


def time_underwater_pct(values: pd.Series) -> float:
    """Share of observations spent below the running peak, in PERCENT [0, 100].

        TUW = 100 * count(d_t < 0) / count(d_t defined)

    Measured in OBSERVATIONS rather than calendar days, so it is directly
    comparable across series sampled on the same trading calendar. The first
    observation is a peak by construction, so a series can never score a full
    100. Observations whose running peak is non-positive (drawdown undefined)
    are excluded from both numerator and denominator.

    NaN for an empty series; 0.0 for a monotonically rising one.
    """
    dd = drawdown_series(_clean_values(values)).dropna()
    if dd.empty:
        return float("nan")
    share = float((dd.to_numpy(dtype=float) < 0.0).sum()) / float(len(dd)) * 100.0
    return share if math.isfinite(share) else float("nan")


def current_drawdown(values: pd.Series) -> dict:
    """Where the portfolio stands right now relative to its all-time high.

    Returns ``{depth, peak_date, peak_value, days_underwater, underwater}``:

    * ``depth``           -- ``V_last / running_max - 1``, a NEGATIVE decimal
                             (0.0 when sitting at a new high), NaN when the
                             running peak is non-positive.
    * ``peak_date``       -- date of the high currently in force (None over a
                             positional index).
    * ``peak_value``      -- that high's value.
    * ``days_underwater`` -- calendar days since that high (0 at a new high;
                             observations over a positional index).
    * ``underwater``      -- ``depth < 0``.

    For empty input every field is NaN/None and ``underwater`` is False.
    Never raises.
    """
    out: dict = {
        "depth": float("nan"),
        "peak_date": None,
        "peak_value": float("nan"),
        "days_underwater": None,
        "underwater": False,
    }
    v = _clean_values(values)
    if v.empty:
        return out

    peaks = _peak_positions(v)
    peak_pos = int(peaks[-1])
    peak_value = _to_float(v.iloc[peak_pos])
    last_value = _to_float(v.iloc[-1])

    depth = float("nan")
    if math.isfinite(peak_value) and peak_value > 0.0 and math.isfinite(last_value):
        depth = _to_float(last_value / peak_value - 1.0)

    dates = _index_dates(v.index)
    if dates is None:
        days = int(len(v) - 1 - peak_pos)
        peak_date: date | None = None
    else:
        days = int((dates[-1] - dates[peak_pos]).days)
        peak_date = dates[peak_pos]

    out["depth"] = depth
    out["peak_date"] = peak_date
    out["peak_value"] = peak_value
    out["days_underwater"] = days
    out["underwater"] = bool(math.isfinite(depth) and depth < 0.0)
    return out


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
def drawdown_episodes(
    values: pd.Series,
    top_n: int = DEFAULT_TOP_N,
    min_depth: float = DEFAULT_MIN_DEPTH,
) -> list[DrawdownEpisode]:
    """Every peak -> trough -> recovery cycle, deepest first.

    The state machine, stated explicitly because this is the fiddly part:

    1. The running peak starts at the first observation.
    2. An episode OPENS on the first value strictly below that peak; the
       lowest value seen while it is open is its trough (ties keep the
       EARLIEST trough date).
    3. The episode CLOSES only when the series reaches a new high
       (``V_t >= peak``); that date is the recovery date, and the peak is
       re-based there.
    4. An episode still open at the end of the series is emitted with
       ``recovery_date=None`` and ``recovery_days=None``.

    ``depth = trough / peak - 1`` (negative decimal). Episodes shallower than
    ``abs(min_depth)`` are discarded, then the survivors are sorted by depth
    ASCENDING -- most negative, i.e. deepest, first -- and the first ``top_n``
    are returned. A non-positive or None ``top_n`` returns all of them.

    ``drawdown_days`` and ``recovery_days`` are CALENDAR days read off the
    index, so a date-like index is required: over a positional index this
    returns an EMPTY list rather than fabricating dates. Empty input, a single
    observation, an all-NaN series and a monotonically rising series all
    return ``[]``. Never raises.
    """
    v = _clean_values(values)
    if len(v) < 2:
        return []
    dates = _index_dates(v.index)
    if dates is None:
        return []

    floor = abs(_to_float(min_depth))
    if not math.isfinite(floor):
        floor = 0.0

    prices = v.to_numpy(dtype=float)
    episodes: list[DrawdownEpisode] = []

    peak_pos, peak_value = 0, float(prices[0])
    trough_pos, trough_value = 0, float(prices[0])
    open_episode = False

    for i in range(1, len(prices)):
        price = float(prices[i])
        if price >= peak_value:
            if open_episode:
                built = _episode(dates, peak_pos, trough_pos, peak_value, trough_value, i)
                if built is not None and built.depth <= -floor:
                    episodes.append(built)
                open_episode = False
            peak_pos, peak_value = i, price
            continue
        if not open_episode:
            open_episode = True
            trough_pos, trough_value = i, price
        elif price < trough_value:
            trough_pos, trough_value = i, price

    if open_episode:
        built = _episode(dates, peak_pos, trough_pos, peak_value, trough_value, None)
        if built is not None and built.depth <= -floor:
            episodes.append(built)

    episodes.sort(key=lambda ep: (ep.depth, ep.peak_date))
    try:
        limit = int(top_n) if top_n is not None else 0
    except (TypeError, ValueError):
        limit = 0
    return episodes[:limit] if limit > 0 else episodes


def episodes_dataframe(episodes: Sequence[DrawdownEpisode] | None) -> pd.DataFrame:
    """Tabulate episodes for the UI: one row each, in the order supplied.

    Columns are :data:`EPISODE_COLUMNS`. ``depth`` stays a negative decimal and
    ``depth_pct`` is the same number in percent, so charts and tables can pick
    the unit they need without re-deriving it. ``recovery_days`` and
    ``total_days`` are ``pd.NA`` for an episode that has not recovered -- NOT
    zero, which would read as "recovered instantly".

    Returns an EMPTY frame carrying the full column set when there is nothing
    to show, so callers can index columns unconditionally. Never raises.
    """
    rows: list[dict] = []
    for episode in episodes or []:
        depth = _to_float(getattr(episode, "depth", float("nan")))
        recovery_days = getattr(episode, "recovery_days", None)
        total_days = getattr(episode, "total_days", None)
        rows.append(
            {
                "peak_date": getattr(episode, "peak_date", None),
                "trough_date": getattr(episode, "trough_date", None),
                "recovery_date": getattr(episode, "recovery_date", None),
                "depth": depth,
                "depth_pct": depth * 100.0,
                "drawdown_days": getattr(episode, "drawdown_days", None),
                "recovery_days": pd.NA if recovery_days is None else recovery_days,
                "total_days": pd.NA if total_days is None else total_days,
                "recovered": bool(getattr(episode, "recovery_date", None) is not None),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(EPISODE_COLUMNS))
    return pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))


def average_recovery_days(episodes: Sequence[DrawdownEpisode] | None) -> float:
    """Mean trough -> recovery time, in calendar days, over RECOVERED episodes.

    Episodes still underwater are excluded rather than counted as zero or
    censored at "today": including them would bias the average downwards
    exactly when the portfolio is in its worst hole.

    NaN when nothing is supplied and when no episode has recovered.
    """
    spans = [
        float(getattr(episode, "recovery_days", None))
        for episode in (episodes or [])
        if getattr(episode, "recovery_days", None) is not None
    ]
    finite = [span for span in spans if math.isfinite(span)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))
