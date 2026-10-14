"""Display formatting: currency, percentages, compact magnitudes, dates.

Pure functions with no Streamlit import, so they are unit-testable and reusable
by the Excel/HTML report writers.

Two conventions worth stating explicitly, because they are the ones people get
wrong in Indian-market tools:

* **Indian digit grouping.** Above one thousand, Indian notation groups the
  last three digits and then pairs: ``12,34,567.89``, not ``1,234,567.89``.
* **Missing is missing.** Every formatter renders ``None``/``NaN``/``inf`` as an
  em dash. A portfolio tool that prints ``0.00`` for an undefined Sharpe ratio is
  lying to its user, and that is the single fastest way to lose their trust.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from src.config import CURRENCY_SYMBOLS, LAKH_CRORE_CURRENCIES

DASH = "—"  # em dash, used for "not available"
UP = "▲"  # black up-pointing triangle
DOWN = "▼"  # black down-pointing triangle
FLAT = "–"  # en dash


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _is_missing(x: object) -> bool:
    """True for None, NaN, +/-inf and anything that is not numeric at all."""
    if x is None:
        return True
    try:
        value = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return not math.isfinite(value)


def _trim_trailing_zeros(text: str) -> str:
    """Drop a trailing ``.50`` -> ``.5`` and ``.00`` -> ``''`` — but only ever
    after a decimal point.

    A bare ``.rstrip("0")`` is a silent 10x bug: with ``decimals=0`` the string
    ``"10"`` has no decimal point, so stripping zeros turns ten crore into one.
    """
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def safe(x: object, default: str = DASH) -> str:
    """Stringify anything, collapsing missing values to a dash."""
    if _is_missing(x):
        return default if not isinstance(x, str) else x
    return str(x)


def fmt_num(x: object, decimals: int = 2, dash: str = DASH) -> str:
    """Thousands-separated fixed-point number."""
    if _is_missing(x):
        return dash
    return f"{float(x):,.{decimals}f}"  # type: ignore[arg-type]


def fmt_pct(x: object, decimals: int = 2, signed: bool = False) -> str:
    """Percentage from a value that is ALREADY in percent units (12.5 -> '12.50%')."""
    if _is_missing(x):
        return DASH
    sign = "+" if signed and float(x) > 0 else ""  # type: ignore[arg-type]
    return f"{sign}{float(x):,.{decimals}f}%"  # type: ignore[arg-type]


def fmt_ratio_pct(x: object, decimals: int = 2, signed: bool = False) -> str:
    """Percentage from a DECIMAL ratio (0.125 -> '12.50%')."""
    if _is_missing(x):
        return DASH
    return fmt_pct(float(x) * 100.0, decimals, signed)  # type: ignore[arg-type]


def fmt_signed(x: object, decimals: int = 2) -> str:
    """Number with an explicit + or - sign."""
    if _is_missing(x):
        return DASH
    value = float(x)  # type: ignore[arg-type]
    return f"{value:+,.{decimals}f}"


def fmt_multiple(x: object, decimals: int = 2) -> str:
    """Ratio rendered as a multiple: 1.35 -> '1.35x'."""
    if _is_missing(x):
        return DASH
    return f"{float(x):,.{decimals}f}x"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Indian digit grouping
# ---------------------------------------------------------------------------
def indian_group(x: object, decimals: int = 2) -> str:
    """Indian 2-2-3 digit grouping: 1234567.89 -> '12,34,567.89'.

    The last three integer digits form one group; everything to the left is
    grouped in pairs. Negative numbers keep the sign outside the grouping.
    """
    if _is_missing(x):
        return DASH
    value = float(x)  # type: ignore[arg-type]
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")

    if len(whole) <= 3:
        grouped = whole
    else:
        head, tail = whole[:-3], whole[-3:]
        pairs = []
        while len(head) > 2:
            pairs.insert(0, head[-2:])
            head = head[:-2]
        if head:
            pairs.insert(0, head)
        grouped = ",".join(pairs) + "," + tail

    return f"{sign}{grouped}" + (f".{frac}" if decimals > 0 else "")


# ---------------------------------------------------------------------------
# money
# ---------------------------------------------------------------------------
def currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get((currency or "").upper(), "")


def fmt_money(
    x: object,
    currency: str = "INR",
    decimals: int = 0,
    compact: bool = False,
    signed: bool = False,
) -> str:
    """Currency-prefixed amount, Indian-grouped for INR.

    ``compact=True`` switches to lakh/crore (INR) or K/M/B (everything else),
    which is what makes large portfolio values readable on a KPI card.
    """
    if _is_missing(x):
        return DASH
    if compact:
        return fmt_compact(x, currency, signed=signed)

    value = float(x)  # type: ignore[arg-type]
    symbol = currency_symbol(currency)
    body = (
        indian_group(abs(value), decimals)
        if (currency or "").upper() in LAKH_CRORE_CURRENCIES
        else f"{abs(value):,.{decimals}f}"
    )
    if value < 0:
        prefix = "-"
    elif signed:
        prefix = "+"
    else:
        prefix = ""
    return f"{prefix}{symbol}{body}"


#: (threshold, divisor, suffix) ladders, largest first.
_INDIAN_SCALE = ((1e7, 1e7, " Cr"), (1e5, 1e5, " L"), (1e3, 1e3, " K"))
_WESTERN_SCALE = ((1e12, 1e12, "T"), (1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e3, 1e3, "K"))


def fmt_compact(x: object, currency: str = "INR", decimals: int = 2, signed: bool = False) -> str:
    """Human-scale magnitude: '1.24 Cr' for INR, '$1.24M' otherwise."""
    if _is_missing(x):
        return DASH
    value = float(x)  # type: ignore[arg-type]
    symbol = currency_symbol(currency)
    magnitude = abs(value)

    ladder = _INDIAN_SCALE if (currency or "").upper() in LAKH_CRORE_CURRENCIES else _WESTERN_SCALE
    scaled, suffix = magnitude, ""
    for threshold, divisor, unit in ladder:
        if magnitude >= threshold:
            scaled, suffix = magnitude / divisor, unit
            break
    body = _trim_trailing_zeros(f"{scaled:,.{decimals}f}") if suffix else f"{magnitude:,.0f}"

    if value < 0:
        prefix = "-"
    elif signed:
        prefix = "+"
    else:
        prefix = ""
    return f"{prefix}{symbol}{body}{suffix}"


def fmt_shares(x: object) -> str:
    """Share counts: integral when whole, up to 4 decimals for fractional units."""
    if _is_missing(x):
        return DASH
    value = float(x)  # type: ignore[arg-type]
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,d}"
    return _trim_trailing_zeros(f"{value:,.4f}")


# ---------------------------------------------------------------------------
# dates & durations
# ---------------------------------------------------------------------------
def fmt_date(d: object, fmt: str = "%d %b %Y") -> str:
    """Format a date, rendering anything missing or unparseable as a dash.

    The ``pd.NaT`` check has to come FIRST: NaT is a subclass of ``datetime``,
    so an ``isinstance(d, datetime)`` branch happily accepts it and then
    ``NaT.strftime`` raises ``ValueError("NaTType does not support strftime")``.
    A missing date must render as a dash, never crash the page drawing it.
    """
    if d is None:
        return DASH

    import pandas as pd

    try:
        if pd.isna(d):
            return DASH
    except (TypeError, ValueError):
        pass  # not a scalar pandas understands; fall through

    if isinstance(d, (datetime, date)):
        return d.strftime(fmt)
    try:
        parsed = pd.to_datetime(d, errors="coerce")
        return DASH if parsed is None or pd.isna(parsed) else parsed.strftime(fmt)
    except Exception:  # noqa: BLE001 - formatting must never raise
        return safe(d)


def fmt_datetime(d: object, fmt: str = "%d %b %Y, %H:%M") -> str:
    return fmt_date(d, fmt)


def humanize_days(n: object) -> str:
    """Duration in calendar days rendered as '2y 3m', '5m 12d' or '18 days'."""
    if _is_missing(n):
        return DASH
    days = int(round(float(n)))  # type: ignore[arg-type]
    if days < 0:
        return DASH
    if days < 31:
        return f"{days} day{'s' if days != 1 else ''}"
    years, remainder = divmod(days, 365)
    months = remainder // 30
    if years and months:
        return f"{years}y {months}m"
    if years:
        return f"{years}y"
    left = remainder - months * 30
    return f"{months}m {left}d" if left else f"{months}m"


# ---------------------------------------------------------------------------
# tone helpers (used by the KPI cards and tables)
# ---------------------------------------------------------------------------
def delta_tone(x: object, invert: bool = False) -> str:
    """Map a signed number to a semantic tone name.

    ``invert=True`` is for metrics where lower is better (volatility, drawdown,
    tracking error) so the colour still means "good" / "bad" rather than
    "positive" / "negative".
    """
    if _is_missing(x):
        return "muted"
    value = float(x)  # type: ignore[arg-type]
    if value == 0:
        return "muted"
    good = (value > 0) != invert
    return "gain" if good else "loss"


def arrow(x: object) -> str:
    if _is_missing(x):
        return ""
    value = float(x)  # type: ignore[arg-type]
    if value > 0:
        return UP
    if value < 0:
        return DOWN
    return FLAT


def fmt_delta(x: object, decimals: int = 2, suffix: str = "%") -> str:
    """Arrow + signed magnitude, e.g. '▲ 2.41%'."""
    if _is_missing(x):
        return DASH
    return f"{arrow(x)} {abs(float(x)):,.{decimals}f}{suffix}"  # type: ignore[arg-type]


def band_tone(band: str) -> str:
    """Risk band name -> tone."""
    return {"Low": "gain", "Moderate": "warn", "High": "loss"}.get(band, "muted")
