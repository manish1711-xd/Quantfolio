"""Technical indicators and rule-based technical signals for one instrument.

Scope: everything in this module operates on a SINGLE security's price history
(a close series, or an OHLCV frame). Nothing here touches the network and
nothing mutates its input -- every function copies before it computes, and
every degenerate input (empty, one row, all-NaN, zero variance) returns NaN /
an empty object rather than raising.

Conventions -- the maths behind the charts:

Simple moving average    SMA_t(n)  = mean(P_(t-n+1) .. P_t)
Exponential MA           EMA_t(s)  = a*P_t + (1-a)*EMA_(t-1),   a = 2/(s+1)
Wilder smoothing         X_t       = a*x_t + (1-a)*X_(t-1),     a = 1/period
Relative strength index  RSI_t     = 100 - 100/(1 + avg_gain/avg_loss)
                                     avg_gain/avg_loss WILDER-smoothed
MACD                     macd      = EMA(fast) - EMA(slow)
                         signal    = EMA(macd, signal_span)
                         histogram = macd - signal
Bollinger bands          middle    = SMA(period)
                         upper     = middle + n_std * sd(period)
                         lower     = middle - n_std * sd(period)
                         percent_b = (P - lower) / (upper - lower)
                         bandwidth = (upper - lower) / middle * 100
True range               TR_t      = max(H-L, |H - C_(t-1)|, |L - C_(t-1)|)
Average true range       ATR       = Wilder smoothing of TR over `period`

RSI and ATR use *Wilder's* exponential smoothing -- pandas
``ewm(alpha=1/period, adjust=False)`` -- not a simple rolling mean. That is the
definition from Wilder's 1978 "New Concepts in Technical Trading Systems" and
the one every charting package implements; a rolling-mean RSI produces visibly
different numbers and is the classic tell of a copied formula.

Bollinger's own construction uses the POPULATION standard deviation (ddof=0)
over the same window as the middle band, so that is what is used here (the
rest of the package uses ddof=1 for risk statistics, where the sample
convention is the correct one).

Every period, band width and overbought/oversold level comes from
``config.TECHNICALS`` so the assumptions stay auditable in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TECHNICALS, TRADING_DAYS, TechnicalConfig
from src.models import Insight

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Every Insight this module emits carries this category.
CATEGORY: str = "technical"

#: Sessions within which a moving-average cross is still treated as news
#: rather than as an established posture. ``config.TechnicalConfig`` has no
#: field for this, so it is declared (and documented) here.
CROSS_LOOKBACK_SESSIONS: int = 5

#: The "52-week" in ``distance_from_52w`` -- a definition, not a tunable.
WEEKS_PER_YEAR: int = 52

#: Column layout produced by :func:`indicator_frame` for a default config.
_MACD_COLUMNS: tuple[str, ...] = ("macd", "signal", "histogram")
_BOLLINGER_COLUMNS: tuple[str, ...] = (
    "middle",
    "upper",
    "lower",
    "bandwidth",
    "percent_b",
)

#: Order used when sorting emitted insights (most actionable first).
_LEVEL_RANK: dict[str, int] = {"critical": 0, "warning": 1, "positive": 2, "info": 3}


# ---------------------------------------------------------------------------
# coercion helpers (private)
# ---------------------------------------------------------------------------
def _coerce_series(data: Any, name: str | None = None) -> pd.Series:
    """Return a float copy of ``data`` with +/-inf mapped to NaN.

    Accepts a Series, a one-column DataFrame or any sequence. Never mutates
    the caller's object and never raises: unparseable input becomes an empty
    float Series.
    """
    if data is None:
        return pd.Series(dtype=float, name=name)
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0] if data.shape[1] else pd.Series(dtype=float)
    if isinstance(data, pd.Series):
        out = pd.to_numeric(data, errors="coerce")
        index = data.index
        label = name if name is not None else data.name
    else:
        try:
            out = pd.Series(list(data))
        except TypeError:
            return pd.Series(dtype=float, name=name)
        out = pd.to_numeric(out, errors="coerce")
        index = out.index
        label = name
    values = np.asarray(out, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    series = pd.Series(values, index=index, name=label, dtype=float)

    # Every indicator here is POSITIONAL: rolling windows, ewm smoothing and
    # diff() all walk the series in row order and assume it runs oldest ->
    # newest. A newest-first table (which is how most brokers and several data
    # APIs export) would otherwise produce a confidently wrong RSI with no
    # warning at all. Sorting a dated index costs nothing and removes the trap;
    # a non-dated index is left exactly as given, since its order is meaningful
    # only to the caller.
    if isinstance(series.index, pd.DatetimeIndex) and not series.index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def _nan_like(series: pd.Series, name: str | None = None) -> pd.Series:
    """All-NaN float Series aligned to ``series`` (empty stays empty)."""
    return pd.Series(np.nan, index=series.index, name=name, dtype=float)


def _empty_frame(columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Zero-row float frame carrying the documented column layout."""
    return pd.DataFrame({c: pd.Series(dtype=float) for c in columns})


def _nan_frame(index: pd.Index, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """All-NaN float frame with ``columns``, aligned to ``index``."""
    return pd.DataFrame(
        {c: pd.Series(np.nan, index=index, dtype=float) for c in columns},
        index=index,
    )


def _as_int(value: Any) -> int | None:
    """Best-effort positive-int coercion; ``None`` when it is not usable."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 1 else None


def _as_float(value: Any) -> float:
    """Best-effort float coercion; NaN when it is not usable."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _column(frame: Any, name: str) -> pd.Series | None:
    """Case-insensitive column lookup that tolerates MultiIndex columns."""
    if not isinstance(frame, pd.DataFrame) or frame.columns.size == 0:
        return None
    target = name.strip().lower()
    for col in frame.columns:
        label = col[0] if isinstance(col, tuple) else col
        if str(label).strip().lower() == target:
            picked = frame[col]
            return _coerce_series(picked, name)
    return None


def _close_of(data: Any) -> pd.Series:
    """Extract the close series from a Series or an OHLCV-style frame."""
    if isinstance(data, pd.DataFrame):
        for candidate in ("close", "adj close", "adj_close", "price"):
            col = _column(data, candidate)
            if col is not None:
                return col
        return _coerce_series(data, "close")
    return _coerce_series(data, "close")


def _last_valid(series: pd.Series) -> tuple[Any, float]:
    """(index_label, value) of the last finite observation, or (None, NaN)."""
    clean = series.dropna()
    if clean.empty:
        return None, float("nan")
    return clean.index[-1], float(clean.iloc[-1])


def _label(index_value: Any) -> str:
    """Human-readable index label for evidence strings."""
    if isinstance(index_value, pd.Timestamp):
        return index_value.strftime("%d %b %Y")
    return str(index_value)


def _fmt(value: float, decimals: int = 2) -> str:
    """Fixed-decimal number for evidence strings; 'n/a' when undefined."""
    return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average: mean of the last ``window`` closes.

    ``SMA_t = (P_(t-window+1) + ... + P_t) / window``. Requires a full window
    (``min_periods=window``), so the first ``window - 1`` values are NaN by
    construction. Returns an all-NaN series aligned to the input when
    ``window`` is not a usable positive integer, and an empty series for empty
    input.
    """
    values = _coerce_series(series)
    win = _as_int(window)
    if win is None:
        return _nan_like(values, "SMA")
    name = f"SMA{win}"
    if values.empty:
        return _nan_like(values, name)
    return values.rolling(window=win, min_periods=win).mean().rename(name)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the standard span parameterisation.

    ``EMA_t = a*P_t + (1-a)*EMA_(t-1)`` with ``a = 2 / (span + 1)``
    (``adjust=False``, i.e. the recursive form used by charting packages).
    ``min_periods=span`` keeps the leading values NaN so a half-formed average
    is never plotted. NaN inputs are skipped, not treated as zeros.
    """
    values = _coerce_series(series)
    length = _as_int(span)
    if length is None:
        return _nan_like(values, "EMA")
    name = f"EMA{length}"
    if values.empty:
        return _nan_like(values, name)
    return values.ewm(span=length, adjust=False, min_periods=length).mean().rename(name)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential smoothing: ``alpha = 1/period``, ``adjust=False``.

    ``X_t = x_t/period + X_(t-1)*(period-1)/period``. This is the smoother
    behind RSI and ATR; it is deliberately slower than an EMA of the same
    span (an EMA with the same alpha would have ``span = 2*period - 1``).
    """
    values = _coerce_series(series)
    length = _as_int(period)
    if length is None or values.empty:
        return _nan_like(values, values.name)
    return values.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = TECHNICALS.rsi_period) -> pd.Series:
    """Wilder's Relative Strength Index, bounded to [0, 100].

    ``RSI = 100 - 100 / (1 + RS)`` where ``RS = avg_gain / avg_loss`` and both
    averages use WILDER'S smoothing (``ewm(alpha=1/period, adjust=False)``),
    NOT a simple rolling mean -- that distinction is the whole definition.

    Conventions at the boundaries:
      * no losses in the window (a monotonically rising series) -> RS is
        infinite -> RSI = 100;
      * no gains -> RSI = 0;
      * a perfectly flat series (no gains and no losses) is neither strong nor
        weak, so RSI is reported as 50 rather than 0/0.

    Returns an empty series for empty input and an all-NaN series when
    ``period`` is unusable or there are fewer than ``period + 1`` closes.
    """
    values = _coerce_series(series)
    length = _as_int(period)
    if length is None:
        return _nan_like(values, "RSI")
    if values.empty:
        return _nan_like(values, "RSI")

    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)

    # avg_loss == 0 with positive gains -> RS = inf -> RSI = 100 already.
    # Both zero -> 0/0 -> NaN, which is the flat-market case: report 50.
    flat = (avg_gain <= 0.0) & (avg_loss <= 0.0)
    out = out.where(~flat, 50.0)
    out = out.clip(lower=0.0, upper=100.0)
    return out.rename("RSI")


def macd(
    series: pd.Series,
    fast: int = TECHNICALS.macd_fast,
    slow: int = TECHNICALS.macd_slow,
    signal: int = TECHNICALS.macd_signal,
) -> pd.DataFrame:
    """Moving Average Convergence/Divergence.

    ``macd = EMA(fast) - EMA(slow)``; ``signal = EMA(macd, signal)``;
    ``histogram = macd - signal`` (exactly -- the histogram is never
    recomputed from anything else, so the identity always holds).

    Columns: ``macd``, ``signal``, ``histogram``. Leading values are NaN until
    both EMAs are fully formed. Degenerate spans return an all-NaN frame
    aligned to the input; empty input returns an empty frame with the same
    columns.
    """
    values = _coerce_series(series)
    fast_span, slow_span, signal_span = _as_int(fast), _as_int(slow), _as_int(signal)
    if values.empty:
        return _empty_frame(_MACD_COLUMNS)
    if fast_span is None or slow_span is None or signal_span is None:
        return _nan_frame(values.index, _MACD_COLUMNS)

    line = ema(values, fast_span) - ema(values, slow_span)
    signal_line = ema(line, signal_span)
    histogram = line - signal_line
    return pd.DataFrame(
        {
            "macd": line.astype(float),
            "signal": signal_line.astype(float),
            "histogram": histogram.astype(float),
        },
        index=values.index,
    )


# ---------------------------------------------------------------------------
# volatility bands
# ---------------------------------------------------------------------------
def bollinger_bands(
    series: pd.Series,
    period: int = TECHNICALS.bollinger_period,
    n_std: float = TECHNICALS.bollinger_std,
) -> pd.DataFrame:
    """Bollinger bands with %B and bandwidth.

    ``middle = SMA(period)``; ``upper/lower = middle +/- n_std * sd`` where
    ``sd`` is the POPULATION standard deviation (ddof=0) over the same window,
    which is Bollinger's own construction.

    ``percent_b = (P - lower) / (upper - lower)`` -- 0.5 exactly on the middle
    band, 1.0 on the upper band, 0.0 on the lower band, outside [0, 1] when
    price closes beyond a band. NaN when the band has zero width (a perfectly
    flat window), because the position is genuinely undefined there.

    ``bandwidth = (upper - lower) / middle * 100`` -- a percent-of-price
    measure of how wide the bands are (the "squeeze" indicator). NaN when the
    middle band is not strictly positive.

    Columns: ``middle``, ``upper``, ``lower``, ``bandwidth``, ``percent_b``.
    """
    values = _coerce_series(series)
    length = _as_int(period)
    width = _as_float(n_std)
    if values.empty:
        return _empty_frame(_BOLLINGER_COLUMNS)
    if length is None or not np.isfinite(width):
        return _nan_frame(values.index, _BOLLINGER_COLUMNS)

    rolling = values.rolling(window=length, min_periods=length)
    middle = rolling.mean()
    sd = rolling.std(ddof=0)
    upper = middle + width * sd
    lower = middle - width * sd

    span = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_b = (values - lower) / span.where(span > 0)
        bandwidth = span / middle.where(middle > 0) * 100.0

    return pd.DataFrame(
        {
            "middle": middle.astype(float),
            "upper": upper.astype(float),
            "lower": lower.astype(float),
            "bandwidth": bandwidth.astype(float),
            "percent_b": percent_b.astype(float),
        },
        index=values.index,
    )


def true_range(ohlc: pd.DataFrame) -> pd.Series:
    """Wilder's true range, ``TR_t = max(H-L, |H-C_(t-1)|, |L-C_(t-1)|)``.

    On the first bar there is no previous close, so TR falls back to the
    high-low range. TR is non-negative by construction whenever High >= Low.
    Returns an empty series when High/Low/Close cannot be located.
    """
    high = _column(ohlc, "high")
    low = _column(ohlc, "low")
    close = _column(ohlc, "close")
    if high is None or low is None or close is None:
        index = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.Index([])
        return pd.Series(np.nan, index=index, dtype=float, name="true_range")

    index = high.index
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    if h.size == 0 or h.size != lo.size or h.size != c.size:
        return pd.Series(np.nan, index=index, dtype=float, name="true_range")

    prev_close = np.empty_like(c)
    prev_close[0] = np.nan
    prev_close[1:] = c[:-1]

    # NaN-safe row-wise max without numpy's "All-NaN slice" warning.
    stacked = np.vstack([np.abs(h - lo), np.abs(h - prev_close), np.abs(lo - prev_close)])
    filled = np.where(np.isnan(stacked), -np.inf, stacked)
    tr = filled.max(axis=0)
    tr = np.where(np.isneginf(tr), np.nan, tr)
    return pd.Series(tr, index=index, dtype=float, name="true_range")


def atr(ohlc: pd.DataFrame, period: int = TECHNICALS.atr_period) -> pd.Series:
    """Average True Range: Wilder-smoothed true range (non-negative).

    ``ATR_t = ATR_(t-1) + (TR_t - ATR_(t-1)) / period`` -- i.e.
    ``ewm(alpha=1/period, adjust=False)`` over the true range, not a simple
    rolling mean. Reported in price units; divide by price for a percentage
    volatility read. NaN until ``period`` true ranges exist.
    """
    length = _as_int(period)
    tr = true_range(ohlc)
    if tr.empty or length is None:
        return _nan_like(tr, "ATR")
    return wilder_smooth(tr, length).rename("ATR")


# ---------------------------------------------------------------------------
# range positioning
# ---------------------------------------------------------------------------
def distance_from_52w(series: pd.Series) -> dict:
    """Where the last close sits inside its trailing 52-week range.

    The window is the trailing 52 calendar weeks when the index is a
    DatetimeIndex, else the last ``config.TRADING_DAYS`` observations.

    Returns ``{last, high, low, pct_from_high, pct_from_low, position_pct,
    n_obs}`` where ``pct_from_high = (last/high - 1) * 100`` (<= 0),
    ``pct_from_low = (last/low - 1) * 100`` (>= 0) and ``position_pct =
    (last - low) / (high - low) * 100`` (0 = at the low, 100 = at the high).
    ``position_pct`` is NaN for a flat range, where position is undefined.
    """
    nan = float("nan")
    blank = {
        "last": nan,
        "high": nan,
        "low": nan,
        "pct_from_high": nan,
        "pct_from_low": nan,
        "position_pct": nan,
        "n_obs": 0,
    }
    values = _coerce_series(series).dropna()
    if values.empty:
        return blank

    window = values
    if isinstance(values.index, pd.DatetimeIndex):
        # DateOffset rather than Timedelta: pandas 2.3 on numpy 2.5 deprecates
        # the bare-integer Timedelta construction.
        cutoff = values.index[-1] - pd.DateOffset(weeks=WEEKS_PER_YEAR)
        window = values[values.index >= cutoff]
    else:
        window = values.tail(TRADING_DAYS)
    if window.empty:
        window = values.tail(TRADING_DAYS)

    last = float(window.iloc[-1])
    high = float(window.max())
    low = float(window.min())
    span = high - low
    return {
        "last": last,
        "high": high,
        "low": low,
        "pct_from_high": (last / high - 1.0) * 100.0 if high > 0 else nan,
        "pct_from_low": (last / low - 1.0) * 100.0 if low > 0 else nan,
        "position_pct": (last - low) / span * 100.0 if span > 0 else nan,
        "n_obs": int(window.size),
    }


# ---------------------------------------------------------------------------
# assembled indicator frame
# ---------------------------------------------------------------------------
def indicator_columns(cfg: TechnicalConfig = TECHNICALS) -> list[str]:
    """Column layout :func:`indicator_frame` produces for ``cfg``.

    Stable and config-driven so the chart layer can address columns by name
    (``SMA20``, ``BB_UPPER``, ...) without recomputing anything.
    """
    columns = ["close"]
    columns += [f"SMA{w}" for w in getattr(cfg, "sma_windows", ()) if _as_int(w)]
    columns += [f"EMA{s}" for s in getattr(cfg, "ema_windows", ()) if _as_int(s)]
    columns += ["RSI", "MACD", "MACD_SIGNAL", "MACD_HIST"]
    columns += [
        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "BB_BANDWIDTH",
        "BB_PERCENT_B",
    ]
    columns += ["ATR"]
    return columns


def indicator_frame(ohlcv: pd.DataFrame, cfg: TechnicalConfig = TECHNICALS) -> pd.DataFrame:
    """Every indicator for one instrument, aligned to the input index.

    Accepts an OHLCV frame (column names matched case-insensitively) or a bare
    close series. Columns are those of :func:`indicator_columns`: ``close``,
    one ``SMA<w>`` per ``cfg.sma_windows``, one ``EMA<s>`` per
    ``cfg.ema_windows``, ``RSI``, ``MACD``/``MACD_SIGNAL``/``MACD_HIST``,
    ``BB_*`` and ``ATR``.

    The column set never changes with the data: when High/Low are absent
    (a close-only frame) ``ATR`` is present but all-NaN, so the UI can bind to
    a fixed layout. Empty input returns an empty frame with those columns.
    """
    columns = indicator_columns(cfg)
    close = _close_of(ohlcv)
    if close.empty:
        return _empty_frame(columns)

    out = pd.DataFrame(index=close.index)
    out["close"] = close.astype(float)

    for window in getattr(cfg, "sma_windows", ()):
        win = _as_int(window)
        if win is not None:
            out[f"SMA{win}"] = sma(close, win)
    for span in getattr(cfg, "ema_windows", ()):
        length = _as_int(span)
        if length is not None:
            out[f"EMA{length}"] = ema(close, length)

    out["RSI"] = rsi(close, getattr(cfg, "rsi_period", TECHNICALS.rsi_period))

    macd_frame = macd(
        close,
        getattr(cfg, "macd_fast", TECHNICALS.macd_fast),
        getattr(cfg, "macd_slow", TECHNICALS.macd_slow),
        getattr(cfg, "macd_signal", TECHNICALS.macd_signal),
    )
    out["MACD"] = macd_frame["macd"]
    out["MACD_SIGNAL"] = macd_frame["signal"]
    out["MACD_HIST"] = macd_frame["histogram"]

    bands = bollinger_bands(
        close,
        getattr(cfg, "bollinger_period", TECHNICALS.bollinger_period),
        getattr(cfg, "bollinger_std", TECHNICALS.bollinger_std),
    )
    out["BB_MIDDLE"] = bands["middle"]
    out["BB_UPPER"] = bands["upper"]
    out["BB_LOWER"] = bands["lower"]
    out["BB_BANDWIDTH"] = bands["bandwidth"]
    out["BB_PERCENT_B"] = bands["percent_b"]

    atr_series = atr(ohlcv, getattr(cfg, "atr_period", TECHNICALS.atr_period))
    if isinstance(atr_series, pd.Series) and not atr_series.empty:
        out["ATR"] = atr_series.reindex(close.index).astype(float)
    else:
        out["ATR"] = np.nan

    return out.reindex(columns=columns)


# ---------------------------------------------------------------------------
# signal rules
# ---------------------------------------------------------------------------
def _trend_regime_insight(close: pd.Series, cfg: TechnicalConfig) -> Insight | None:
    """Price versus the longest configured SMA (the 200-day regime filter)."""
    windows = sorted({w for w in map(_as_int, getattr(cfg, "sma_windows", ())) if w})
    if not windows:
        return None
    window = windows[-1]
    _, average = _last_valid(sma(close, window))
    _, last = _last_valid(close)
    if not np.isfinite(average) or not np.isfinite(last) or average <= 0:
        return None

    gap = (last / average - 1.0) * 100.0
    above = gap >= 0.0
    text = (
        f"Price is {abs(gap):.1f}% {'above' if above else 'below'} its "
        f"{window}-day moving average - the long-term trend filter is "
        f"{'constructive' if above else 'negative'}."
    )
    evidence = (
        f"close={_fmt(last)}, SMA{window}={_fmt(average)}, "
        f"gap={gap:+.2f}% vs the 0.00% regime threshold "
        f"(price above/below the {window}-day average)"
    )
    return Insight(
        level="positive" if above else "warning",
        text=text,
        category=CATEGORY,
        evidence=evidence,
    )


def _cross_insight(close: pd.Series, cfg: TechnicalConfig) -> Insight | None:
    """Golden / death cross of the two longest configured SMAs (50 vs 200)."""
    windows = sorted({w for w in map(_as_int, getattr(cfg, "sma_windows", ())) if w})
    if len(windows) < 2:
        return None
    fast_window, slow_window = windows[-2], windows[-1]
    spread = (sma(close, fast_window) - sma(close, slow_window)).dropna()
    if spread.size < 2:
        return None

    lookback = int(min(CROSS_LOOKBACK_SESSIONS + 1, spread.size))
    recent = spread.iloc[-lookback:]
    signs = np.sign(recent.to_numpy(dtype=float))
    event: tuple[str, Any] | None = None
    for i in range(1, signs.size):
        if signs[i - 1] <= 0.0 and signs[i] > 0.0:
            event = ("golden", recent.index[i])
        elif signs[i - 1] >= 0.0 and signs[i] < 0.0:
            event = ("death", recent.index[i])

    fast_value = float(sma(close, fast_window).dropna().iloc[-1])
    slow_value = float(sma(close, slow_window).dropna().iloc[-1])
    spread_now = float(spread.iloc[-1])

    if event is not None:
        kind, when = event
        golden = kind == "golden"
        text = (
            f"{'Golden' if golden else 'Death'} cross: the {fast_window}-day "
            f"average crossed {'above' if golden else 'below'} the "
            f"{slow_window}-day average on {_label(when)}."
        )
        evidence = (
            f"SMA{fast_window}={_fmt(fast_value)}, SMA{slow_window}="
            f"{_fmt(slow_value)}, spread={spread_now:+.2f} "
            f"(sign change within the last {CROSS_LOOKBACK_SESSIONS} sessions)"
        )
        return Insight(
            level="positive" if golden else "warning",
            text=text,
            category=CATEGORY,
            evidence=evidence,
        )

    above = spread_now >= 0.0
    text = (
        f"No recent {fast_window}/{slow_window}-day crossover; the "
        f"{fast_window}-day average is holding "
        f"{'above' if above else 'below'} the {slow_window}-day average."
    )
    evidence = (
        f"SMA{fast_window}={_fmt(fast_value)}, SMA{slow_window}="
        f"{_fmt(slow_value)}, spread={spread_now:+.2f}, no sign change in the "
        f"last {CROSS_LOOKBACK_SESSIONS} sessions"
    )
    return Insight(level="info", text=text, category=CATEGORY, evidence=evidence)


def _rsi_insight(close: pd.Series, cfg: TechnicalConfig) -> Insight | None:
    """RSI overbought / oversold against the configured levels."""
    period = getattr(cfg, "rsi_period", TECHNICALS.rsi_period)
    overbought = _as_float(getattr(cfg, "rsi_overbought", TECHNICALS.rsi_overbought))
    oversold = _as_float(getattr(cfg, "rsi_oversold", TECHNICALS.rsi_oversold))
    _, value = _last_valid(rsi(close, period))
    if not np.isfinite(value) or not np.isfinite(overbought) or not np.isfinite(oversold):
        return None

    if value >= overbought:
        return Insight(
            level="warning",
            text=(
                f"{period}-day RSI at {value:.0f} is in overbought territory - "
                "momentum is stretched and mean reversion is common from here."
            ),
            category=CATEGORY,
            evidence=f"RSI({period})={_fmt(value, 1)} >= overbought {_fmt(overbought, 1)}",
        )
    if value <= oversold:
        return Insight(
            level="warning",
            text=(
                f"{period}-day RSI at {value:.0f} is in oversold territory - "
                "the security is under sustained selling pressure."
            ),
            category=CATEGORY,
            evidence=f"RSI({period})={_fmt(value, 1)} <= oversold {_fmt(oversold, 1)}",
        )
    return Insight(
        level="info",
        text=f"{period}-day RSI at {value:.0f} is neutral - momentum is not stretched.",
        category=CATEGORY,
        evidence=(
            f"RSI({period})={_fmt(value, 1)}, inside the "
            f"{_fmt(oversold, 1)}-{_fmt(overbought, 1)} neutral band"
        ),
    )


def _bollinger_insight(close: pd.Series, cfg: TechnicalConfig) -> Insight | None:
    """Close outside the Bollinger bands, else where %B sits inside them."""
    period = getattr(cfg, "bollinger_period", TECHNICALS.bollinger_period)
    n_std = _as_float(getattr(cfg, "bollinger_std", TECHNICALS.bollinger_std))
    bands = bollinger_bands(close, period, n_std)
    if bands.empty or bands.shape[0] != close.size:
        return None
    # Carry the close positionally: the bands frame shares the close index,
    # which may contain duplicate labels for hand-built frames.
    bands = bands.assign(price=close.to_numpy(dtype=float))
    usable = bands.dropna(subset=["upper", "lower", "price"])
    if usable.empty:
        return None

    upper = float(usable["upper"].iloc[-1])
    lower = float(usable["lower"].iloc[-1])
    price = float(usable["price"].iloc[-1])
    percent_b = _as_float(usable["percent_b"].iloc[-1])

    if price > upper:
        return Insight(
            level="warning",
            text=(
                f"Price closed above the upper Bollinger band "
                f"({period}-day, {_fmt(n_std, 1)} sd) - an extended move that "
                "often precedes consolidation."
            ),
            category=CATEGORY,
            evidence=(
                f"close={_fmt(price)} > upper band {_fmt(upper)}, "
                f"%B={_fmt(percent_b, 2)} vs the 1.00 band threshold"
            ),
        )
    if price < lower:
        return Insight(
            level="warning",
            text=(
                f"Price closed below the lower Bollinger band "
                f"({period}-day, {_fmt(n_std, 1)} sd) - a statistically "
                "unusual decline versus its own recent range."
            ),
            category=CATEGORY,
            evidence=(
                f"close={_fmt(price)} < lower band {_fmt(lower)}, "
                f"%B={_fmt(percent_b, 2)} vs the 0.00 band threshold"
            ),
        )
    if not np.isfinite(percent_b):
        # Zero-width band: the window has no dispersion at all, so %B is
        # genuinely undefined rather than zero.
        return Insight(
            level="info",
            text=(
                f"The {period}-day Bollinger bands have collapsed to zero "
                "width - the price has not moved over the window, so its "
                "position inside the bands is undefined."
            ),
            category=CATEGORY,
            evidence=(
                f"close={_fmt(price)}, upper=lower={_fmt(upper)}, "
                f"band width 0.00, %B undefined (0/0)"
            ),
        )
    return Insight(
        level="info",
        text=(
            f"Price is inside its {period}-day Bollinger bands at "
            f"{_fmt(percent_b * 100.0, 0)}% of the band width "
            "(50% is the middle band)."
        ),
        category=CATEGORY,
        evidence=(
            f"close={_fmt(price)}, band {_fmt(lower)} to {_fmt(upper)}, "
            f"%B={_fmt(percent_b, 2)} inside the 0.00-1.00 range"
        ),
    )


def _macd_insight(close: pd.Series, cfg: TechnicalConfig) -> Insight | None:
    """MACD histogram sign flip (momentum turning), else its current posture."""
    fast = getattr(cfg, "macd_fast", TECHNICALS.macd_fast)
    slow = getattr(cfg, "macd_slow", TECHNICALS.macd_slow)
    signal_span = getattr(cfg, "macd_signal", TECHNICALS.macd_signal)
    histogram = macd(close, fast, slow, signal_span)["histogram"].dropna()
    if histogram.size < 2:
        return None

    last = float(histogram.iloc[-1])
    prev = float(histogram.iloc[-2])
    when = _label(histogram.index[-1])
    stem = f"MACD({fast},{slow},{signal_span})"

    if prev <= 0.0 < last:
        return Insight(
            level="positive",
            text=(
                f"MACD histogram turned positive on {when} - short-term "
                "momentum has crossed above its signal line."
            ),
            category=CATEGORY,
            evidence=(
                f"{stem} histogram {prev:+.4f} -> {last:+.4f}, "
                "sign flip across the 0.0000 threshold"
            ),
        )
    if prev >= 0.0 > last:
        return Insight(
            level="warning",
            text=(
                f"MACD histogram turned negative on {when} - short-term "
                "momentum has crossed below its signal line."
            ),
            category=CATEGORY,
            evidence=(
                f"{stem} histogram {prev:+.4f} -> {last:+.4f}, "
                "sign flip across the 0.0000 threshold"
            ),
        )
    positive = last >= 0.0
    return Insight(
        level="info",
        text=(
            f"MACD momentum is {'positive' if positive else 'negative'} and "
            "has not flipped sign in the latest session."
        ),
        category=CATEGORY,
        evidence=(
            f"{stem} histogram {prev:+.4f} -> {last:+.4f}, "
            "no sign change across the 0.0000 threshold"
        ),
    )


def technical_signals(ohlcv: pd.DataFrame, cfg: TechnicalConfig = TECHNICALS) -> list[Insight]:
    """Rule-based technical read of one instrument as ``Insight`` objects.

    Rules covered, each firing off a threshold in ``config.TECHNICALS``:
      * golden / death cross of the two longest SMAs (50 vs 200 by default);
      * RSI overbought / oversold versus ``rsi_overbought`` / ``rsi_oversold``;
      * a close outside the Bollinger bands;
      * a MACD histogram sign flip (momentum crossing its signal line);
      * price versus the longest SMA (the 200-day regime filter).

    Every Insight carries ``category="technical"`` and an ``evidence`` string
    containing the observed number AND the threshold that fired it. Rules with
    insufficient history are silently skipped, so the list is empty rather
    than wrong when the window is short. The list is ordered
    critical -> warning -> positive -> info, matching the rest of the app.
    """
    close = _close_of(ohlcv)
    if close.dropna().size < 2:
        return []

    candidates = (
        _trend_regime_insight(close, cfg),
        _cross_insight(close, cfg),
        _rsi_insight(close, cfg),
        _bollinger_insight(close, cfg),
        _macd_insight(close, cfg),
    )
    found = [item for item in candidates if item is not None]
    return sorted(found, key=lambda ins: _LEVEL_RANK.get(ins.level, len(_LEVEL_RANK)))
