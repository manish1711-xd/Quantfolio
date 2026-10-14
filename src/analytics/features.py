"""Feature construction: turning a price history into a supervised matrix.

This module is the input half of :mod:`src.analytics.forecast`. It answers one
question -- *what did this instrument look like at the close of day t?* -- as a
row of roughly fifty numbers, and pairs it with *what the price did over the
next h sessions* as the label.

Three properties are enforced by construction, because every one of them is a
way a stock-prediction model silently lies about its own accuracy:

**1. No lookahead, anywhere.**
    The feature at row ``t`` is a function of observations up to and including
    ``t``. Nothing is centred, nothing is back-filled, nothing is standardised
    against statistics of the whole sample (scaling happens inside each
    training fold, in :mod:`src.analytics.forecast`, never here). The target at
    row ``t`` is ``log(C_{t+h} / C_t)`` and is the only forward-looking column
    in the frame -- which is why it is returned separately rather than mixed
    into the feature matrix where a careless ``X = frame.drop(...)`` could
    leave it behind.

**2. Everything is scale-free.**
    Not one feature carries the price level. They are log returns, ratios of a
    price to its own moving average, z-scores, volatility ratios, and positions
    inside a trailing range. A model handed the raw close learns "the price
    went up historically, so it goes up", which fits a trend perfectly in
    sample and transfers to nothing.

**3. Missing is missing.**
    A window that has not filled yet produces NaN, and the rows carrying NaN
    are dropped by the caller rather than imputed. Forward-filling an
    unavailable 200-day moving average would invent a trend signal out of the
    first observation.

Feature families
================

============= ===============================================================
family        what it measures
============= ===============================================================
returns       individual lagged daily log returns -- short-horizon
              autocorrelation, the only genuinely documented weak-form effect
momentum      cumulative log return over 10 to 252 sessions, plus the classic
              12-1 month momentum that skips the short-term reversal window
volatility    realised volatility over three windows, the short/long
              volatility ratio (a regime signal), downside volatility, and
              the 63-day skew and excess kurtosis of returns
trend         distance from four moving averages and the spread between them
              -- the golden/death cross geometry as continuous numbers
oscillator    Wilder RSI, MACD histogram, Bollinger %B and bandwidth, ATR --
              all normalised by price where they carry a unit
range         position inside the trailing 252-session range, distance from
              its high and low, the intraday bar range, where the close sat
              inside its own bar, and the overnight gap
volume        volume z-score against its own 21-session norm, and the
              short/long volume ratio
calendar      day of week and month encoded as sine/cosine pairs, plus a
              turn-of-month flag
============= ===============================================================

``volume`` and the bar-shape half of ``range`` need High/Low/Volume columns.
When the caller supplies a bare close series -- which is exactly what the
portfolio value series is -- those families are skipped and the matrix comes
back narrower rather than fabricated. :func:`available_families` reports which
ones a given input can support, so the UI can say what it is working with.

No I/O, no Streamlit, no global state; every function copies its input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURES, TECHNICALS, TRADING_DAYS, FeatureConfig, TechnicalConfig

__all__ = [
    "FEATURE_FAMILIES",
    "OHLCV_DEPENDENT",
    "TARGET_NAME",
    "available_families",
    "build_features",
    "close_series",
    "describe_feature",
    "feature_catalogue",
    "forward_log_return",
    "latest_row",
    "supervised_matrix",
]

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Name of the label column produced by :func:`forward_log_return`.
TARGET_NAME: str = "forward_log_return"

#: Family -> one-line description, in the order the UI should present them.
FEATURE_FAMILIES: dict[str, str] = {
    "returns": "Lagged daily log returns — short-horizon autocorrelation.",
    "momentum": "Cumulative return over 10–252 sessions, including 12-1 momentum.",
    "volatility": "Realised volatility, the short/long vol ratio, skew and kurtosis.",
    "trend": "Distance from the 10/20/50/200 moving averages and their spreads.",
    "oscillator": "Wilder RSI, MACD histogram, Bollinger %B and bandwidth, ATR.",
    "range": "Position in the trailing 252-session range and the shape of the bar.",
    "volume": "Volume against its own norm, and the short/long volume ratio.",
    "calendar": "Day of week and month as cyclical encodings, turn of month.",
}

#: Families that cannot be built from a close series alone.
OHLCV_DEPENDENT: frozenset[str] = frozenset({"volume"})

#: Columns of a full OHLCV frame, in the order the data layer supplies them.
_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

#: Sessions at the start and end of a month treated as "turn of month".
_TURN_OF_MONTH_DAYS: int = 3

#: Momentum skip window for the 12-1 feature, in sessions (~one month).
_MOMENTUM_SKIP: int = 21

#: Guard against dividing by a standard deviation that is numerically zero.
_EPSILON: float = 1e-12

#: Feature name -> plain-English reading, used by the UI's importance table.
#: Generated names (``ret_lag_3``) are described by :func:`describe_feature`
#: from their family prefix; this table covers the ones with no pattern.
_DESCRIPTIONS: dict[str, str] = {
    "mom_12_1": "12-month return excluding the last month — the classic momentum signal.",
    "vol_ratio": "Short-window volatility divided by long-window volatility: a regime gauge.",
    "downside_vol_21": "Volatility of negative days only, over 21 sessions.",
    "skew_63": "Skewness of daily returns over 63 sessions.",
    "kurtosis_63": "Excess kurtosis of daily returns over 63 sessions.",
    "sma_20_50_spread": "20-day average above (+) or below (−) the 50-day, as a ratio.",
    "sma_50_200_spread": "50-day average against the 200-day — golden/death cross geometry.",
    "rsi_14": "Wilder RSI, re-centred so 0 is the neutral 50 reading.",
    "macd_hist": "MACD histogram divided by price, so it is comparable across names.",
    "bb_percent_b": "Where the close sits between the Bollinger bands (0 = lower, 1 = upper).",
    "bb_bandwidth": "Band width as a fraction of the middle band — a volatility proxy.",
    "atr_ratio": "Average true range divided by the close.",
    "range_position_252": "Position inside the trailing 252-session high/low range.",
    "high_252_gap": "Distance below the trailing 252-session high.",
    "low_252_gap": "Distance above the trailing 252-session low.",
    "bar_range": "High minus low, divided by the close.",
    "close_location": "Where the close sat inside its own bar (0 = low, 1 = high).",
    "overnight_gap": "Open against the previous close.",
    "volume_z_21": "Volume in standard deviations from its own 21-session mean.",
    "volume_ratio": "Short-window average volume divided by the long-window average.",
    "dow_sin": "Day of week, sine component of a cyclical encoding.",
    "dow_cos": "Day of week, cosine component of a cyclical encoding.",
    "month_sin": "Month of year, sine component of a cyclical encoding.",
    "month_cos": "Month of year, cosine component of a cyclical encoding.",
    "turn_of_month": "1 within three sessions of a month boundary, else 0.",
}


# ---------------------------------------------------------------------------
# input hygiene
# ---------------------------------------------------------------------------
def _to_frame(data: Any) -> pd.DataFrame:
    """Normalise any accepted input into an ascending OHLCV-shaped frame.

    Accepts a full OHLCV frame, a one-column frame, a Series, or any sequence.
    The result always carries a ``Close`` column; ``Open``/``High``/``Low``/
    ``Volume`` appear only when the caller actually supplied them. Duplicated
    index labels keep their last observation, +/-inf becomes NaN, and rows
    with no close are dropped -- a bar without a close is unusable for every
    feature downstream.

    Returns an empty frame rather than raising for anything unparseable.
    """
    if data is None:
        return pd.DataFrame(columns=["Close"], dtype="float64")

    try:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            if "Close" not in frame.columns:
                if frame.shape[1] != 1:
                    return pd.DataFrame(columns=["Close"], dtype="float64")
                frame = frame.rename(columns={frame.columns[0]: "Close"})
            keep = [c for c in _OHLCV_COLUMNS if c in frame.columns]
            frame = frame.loc[:, keep]
        elif isinstance(data, pd.Series):
            frame = data.to_frame(name="Close")
        else:
            values = pd.Series(np.asarray(data, dtype="float64").ravel())
            frame = values.to_frame(name="Close")
    except (TypeError, ValueError):
        return pd.DataFrame(columns=["Close"], dtype="float64")

    frame = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame[frame["Close"].notna() & (frame["Close"] > 0.0)]
    return frame


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    """True when every named column exists and carries at least one number."""
    return all(c in frame.columns and frame[c].notna().any() for c in columns)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf at a zero divisor."""
    denom = denominator.where(denominator.abs() > _EPSILON)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential smoothing -- ``alpha = 1 / period``, not a mean.

    Duplicated from :mod:`src.analytics.technicals` rather than imported so
    this module stays a leaf of the dependency graph and can be reasoned about
    (and tested) on its own. The convention is identical -- including
    ``min_periods``, so the first ``period`` rows are NaN rather than a
    "14-period RSI" seeded from three observations -- and a regression test
    pins the two implementations against each other.
    """
    span = max(int(period), 1)
    return series.ewm(alpha=1.0 / span, adjust=False, min_periods=span).mean()


# ---------------------------------------------------------------------------
# family builders (each returns a dict of name -> Series)
# ---------------------------------------------------------------------------
def _returns_family(log_returns: pd.Series, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Individual lagged daily log returns.

    ``ret_lag_1`` is *yesterday's* return as seen from today's close, so the
    model never sees a return it could not have observed. Lag k is the return
    of session ``t - k + 1``, i.e. ``ret_lag_1`` is today's own completed move.
    """
    out: dict[str, pd.Series] = {}
    for lag in cfg.return_lags:
        step = max(int(lag), 1)
        out[f"ret_lag_{step}"] = log_returns.shift(step - 1)
    return out


def _momentum_family(log_returns: pd.Series, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Cumulative log return over each window, plus 12-1 month momentum."""
    out: dict[str, pd.Series] = {}
    for window in cfg.momentum_windows:
        span = max(int(window), 2)
        out[f"mom_{span}"] = log_returns.rolling(span, min_periods=span).sum()

    year = max(int(TRADING_DAYS), _MOMENTUM_SKIP + 2)
    total = log_returns.rolling(year, min_periods=year).sum()
    recent = log_returns.rolling(_MOMENTUM_SKIP, min_periods=_MOMENTUM_SKIP).sum()
    out["mom_12_1"] = total - recent
    return out


def _volatility_family(log_returns: pd.Series, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Realised volatility, the regime ratio, downside vol and higher moments."""
    out: dict[str, pd.Series] = {}
    for window in cfg.volatility_windows:
        span = max(int(window), 2)
        out[f"vol_{span}"] = log_returns.rolling(span, min_periods=span).std(ddof=1)

    short, long = (max(int(w), 2) for w in cfg.vol_regime_windows)
    short_vol = log_returns.rolling(short, min_periods=short).std(ddof=1)
    long_vol = log_returns.rolling(long, min_periods=long).std(ddof=1)
    out["vol_ratio"] = _safe_divide(short_vol, long_vol)

    downside = log_returns.where(log_returns < 0.0, 0.0)
    out["downside_vol_21"] = downside.rolling(21, min_periods=21).std(ddof=1)

    moment = max(int(cfg.moment_window), 4)
    out[f"skew_{moment}"] = log_returns.rolling(moment, min_periods=moment).skew()
    out[f"kurtosis_{moment}"] = log_returns.rolling(moment, min_periods=moment).kurt()
    return out


def _trend_family(close: pd.Series, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Distance from each moving average, and the spreads between them."""
    out: dict[str, pd.Series] = {}
    averages: dict[int, pd.Series] = {}
    for window in cfg.ma_windows:
        span = max(int(window), 2)
        average = close.rolling(span, min_periods=span).mean()
        averages[span] = average
        out[f"dist_sma_{span}"] = _safe_divide(close, average) - 1.0

    if 20 in averages and 50 in averages:
        out["sma_20_50_spread"] = _safe_divide(averages[20], averages[50]) - 1.0
    if 50 in averages and 200 in averages:
        out["sma_50_200_spread"] = _safe_divide(averages[50], averages[200]) - 1.0
    return out


def _oscillator_family(
    frame: pd.DataFrame, close: pd.Series, cfg: TechnicalConfig
) -> dict[str, pd.Series]:
    """RSI, MACD histogram, Bollinger %B and bandwidth, and the ATR ratio.

    Each one is normalised so it is comparable across instruments: the RSI is
    re-centred on its neutral 50, the MACD histogram and the ATR are divided by
    the close, and %B is already a 0-1 position by construction.
    """
    out: dict[str, pd.Series] = {}

    # Boundary conventions are copied from ``technicals.rsi`` exactly: a stretch
    # with no losing day gives infinite relative strength and a genuine RSI of
    # 100, while a perfectly flat stretch (no gains AND no losses) is neither
    # strong nor weak and reports 50 rather than 0/0. The two implementations
    # are pinned against each other by a regression test.
    delta = close.diff()
    gains = _wilder(delta.clip(lower=0.0), cfg.rsi_period)
    losses = _wilder((-delta).clip(lower=0.0), cfg.rsi_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = gains / losses
        rsi = 100.0 - 100.0 / (1.0 + strength)
    flat = (gains <= 0.0) & (losses <= 0.0)
    rsi = rsi.where(~flat, 50.0).clip(lower=0.0, upper=100.0)
    # Re-centred on the neutral 50 and scaled to roughly [-1, 1], so the
    # feature is comparable with every other scale-free column.
    out[f"rsi_{cfg.rsi_period}"] = (rsi - 50.0) / 50.0

    fast = close.ewm(span=max(cfg.macd_fast, 1), adjust=False).mean()
    slow = close.ewm(span=max(cfg.macd_slow, 1), adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=max(cfg.macd_signal, 1), adjust=False).mean()
    out["macd_hist"] = _safe_divide(macd_line - signal, close)

    period = max(int(cfg.bollinger_period), 2)
    middle = close.rolling(period, min_periods=period).mean()
    # Bollinger's own construction uses the POPULATION standard deviation.
    deviation = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + cfg.bollinger_std * deviation
    lower = middle - cfg.bollinger_std * deviation
    out["bb_percent_b"] = _safe_divide(close - lower, upper - lower)
    out["bb_bandwidth"] = _safe_divide(upper - lower, middle)

    if _has(frame, "High", "Low"):
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                frame["High"] - frame["Low"],
                (frame["High"] - previous_close).abs(),
                (frame["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_ratio"] = _safe_divide(_wilder(true_range, cfg.atr_period), close)
    return out


def _range_family(
    frame: pd.DataFrame, close: pd.Series, cfg: FeatureConfig
) -> dict[str, pd.Series]:
    """Where the price sits in its trailing range, and the shape of the bar."""
    out: dict[str, pd.Series] = {}
    window = max(int(cfg.range_window), 2)
    high = close.rolling(window, min_periods=window).max()
    low = close.rolling(window, min_periods=window).min()
    out[f"range_position_{window}"] = _safe_divide(close - low, high - low)
    out[f"high_{window}_gap"] = _safe_divide(close, high) - 1.0
    out[f"low_{window}_gap"] = _safe_divide(close, low) - 1.0

    if _has(frame, "High", "Low"):
        span = frame["High"] - frame["Low"]
        out["bar_range"] = _safe_divide(span, close)
        out["close_location"] = _safe_divide(close - frame["Low"], span)
    if _has(frame, "Open"):
        out["overnight_gap"] = _safe_divide(frame["Open"], close.shift(1)) - 1.0
    return out


def _volume_family(frame: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Volume against its own norm. Skipped entirely when volume is absent."""
    if not _has(frame, "Volume"):
        return {}
    volume = frame["Volume"].astype("float64")
    short, long = (max(int(w), 2) for w in cfg.volume_windows)

    mean_long = volume.rolling(long, min_periods=long).mean()
    std_long = volume.rolling(long, min_periods=long).std(ddof=1)
    mean_short = volume.rolling(short, min_periods=short).mean()
    return {
        f"volume_z_{long}": _safe_divide(volume - mean_long, std_long),
        "volume_ratio": _safe_divide(mean_short, mean_long),
    }


def _calendar_family(index: pd.Index) -> dict[str, pd.Series]:
    """Cyclical day-of-week / month encodings and a turn-of-month flag.

    Sine/cosine pairs rather than an integer: a linear model handed
    ``day_of_week = 4`` learns that Friday is four times Tuesday, which is not
    a statement anyone intended to make. The pair encodes the wrap-around so
    Friday and Monday sit next to each other, as they do in the calendar.

    Returns an empty dict for a non-datetime index -- a positional index
    carries no calendar information to encode.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return {}
    weekday = pd.Series(index.dayofweek.to_numpy(dtype="float64"), index=index)
    month = pd.Series(index.month.to_numpy(dtype="float64"), index=index)
    day = pd.Series(index.day.to_numpy(dtype="float64"), index=index)
    days_in_month = pd.Series(index.days_in_month.to_numpy(dtype="float64"), index=index)

    near_start = day <= _TURN_OF_MONTH_DAYS
    near_end = (days_in_month - day) < _TURN_OF_MONTH_DAYS
    return {
        "dow_sin": np.sin(2.0 * np.pi * weekday / 5.0),
        "dow_cos": np.cos(2.0 * np.pi * weekday / 5.0),
        "month_sin": np.sin(2.0 * np.pi * month / 12.0),
        "month_cos": np.cos(2.0 * np.pi * month / 12.0),
        "turn_of_month": (near_start | near_end).astype("float64"),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def available_families(data: Any) -> tuple[str, ...]:
    """Families this input can support, in :data:`FEATURE_FAMILIES` order.

    A bare close series drops ``volume``; a frame with no datetime index drops
    ``calendar``. Reported rather than silently applied, so the UI can say
    which signals a portfolio value series is and is not being shown.
    """
    frame = _to_frame(data)
    if frame.empty:
        return ()
    families = [
        name
        for name in FEATURE_FAMILIES
        if not (name in OHLCV_DEPENDENT and not _has(frame, "Volume"))
    ]
    if not isinstance(frame.index, pd.DatetimeIndex):
        families = [name for name in families if name != "calendar"]
    return tuple(families)


def build_features(
    data: Any,
    cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
) -> pd.DataFrame:
    """Full feature matrix for one instrument, one row per session.

    Parameters
    ----------
    data:
        OHLCV frame, one-column frame, or a close Series. Families that need
        columns the input does not carry are skipped, never fabricated.
    cfg:
        Window configuration; defaults to ``config.FEATURES``.
    technical_cfg:
        Indicator periods for the oscillator family; defaults to
        ``config.TECHNICALS`` so the RSI here and the RSI on the Security page
        are the same number by construction.

    Returns a float frame on the input's own index, with columns ordered by
    family. Early rows carry NaN where a window has not filled --
    :func:`supervised_matrix` drops them. An unusable input returns an empty
    frame rather than raising, and the caller's object is never modified.
    """
    frame = _to_frame(data)
    if frame.empty or len(frame) < 2:
        return pd.DataFrame(index=frame.index)

    settings = cfg if isinstance(cfg, FeatureConfig) else FEATURES
    indicators = technical_cfg if isinstance(technical_cfg, TechnicalConfig) else TECHNICALS

    close = frame["Close"]
    log_returns = np.log(close).diff()

    columns: dict[str, pd.Series] = {}
    columns.update(_returns_family(log_returns, settings))
    columns.update(_momentum_family(log_returns, settings))
    columns.update(_volatility_family(log_returns, settings))
    columns.update(_trend_family(close, settings))
    columns.update(_oscillator_family(frame, close, indicators))
    columns.update(_range_family(frame, close, settings))
    columns.update(_volume_family(frame, settings))
    columns.update(_calendar_family(frame.index))

    features = pd.DataFrame(columns, index=frame.index).astype("float64")
    features = features.replace([np.inf, -np.inf], np.nan)
    # A column that never produced a single number -- a 252-session window on
    # 200 rows -- is dropped entirely rather than kept as an all-NaN column,
    # which would make EVERY row incomplete and leave the caller with nothing.
    #
    # The consequence is deliberate and worth stating: the width of this matrix
    # depends on how much history it was given, so a model fitted on five years
    # has seen features a model fitted on six months never had. The forecaster
    # reports the column count it actually used for exactly that reason, and
    # refuses outright below ``ForecastConfig.min_train_rows``.
    return features.dropna(axis=1, how="all")


def forward_log_return(close: Any, horizon: int) -> pd.Series:
    """The label: ``log(C_{t+h} / C_t)``, NaN for the last ``h`` rows.

    Log rather than simple return for two reasons that both matter here: it is
    additive across time, so a 21-day label is exactly the sum of its 21 daily
    parts; and it is far closer to symmetric, which every squared-error loss in
    the model layer implicitly assumes.

    The trailing NaNs are the point of the function, not an artefact -- they
    mark the rows that can be *predicted from* but not *trained on*.
    """
    frame = _to_frame(close)
    if frame.empty:
        return pd.Series(dtype="float64", name=TARGET_NAME)
    steps = max(int(horizon), 1)
    logs = np.log(frame["Close"])
    target = logs.shift(-steps) - logs
    target.name = TARGET_NAME
    return target.replace([np.inf, -np.inf], np.nan)


def supervised_matrix(
    data: Any,
    horizon: int,
    cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned ``(X, y)`` for one horizon, ready for the model layer.

    ``X`` keeps every row whose features are complete -- **including the final
    rows whose label is still unknown**, because those are exactly the rows a
    live forecast is made from. ``y`` is aligned to ``X`` and carries NaN for
    the last ``horizon`` rows.

    The split between "can be trained on" and "can be predicted from" is
    therefore ``y.notna()``, and it is left to the caller so the distinction
    stays visible at the call site instead of being buried here.

    Returns ``(empty frame, empty Series)`` when the input cannot support a
    single complete row.
    """
    features = build_features(data, cfg, technical_cfg)
    if features.empty or features.shape[1] == 0:
        return pd.DataFrame(), pd.Series(dtype="float64", name=TARGET_NAME)

    target = forward_log_return(data, horizon)
    complete = features.dropna(how="any")
    if complete.empty:
        return pd.DataFrame(), pd.Series(dtype="float64", name=TARGET_NAME)

    aligned = target.reindex(complete.index)
    aligned.name = TARGET_NAME
    return complete, aligned


def close_series(data: Any) -> pd.Series:
    """The cleaned, ascending close series any accepted input reduces to.

    The public spelling of the normalisation every other function here starts
    with, so callers that need the last price (or the index the features sit
    on) get exactly the series the features were built from rather than
    re-deriving it and risking a different answer at the boundary.
    """
    frame = _to_frame(data)
    if frame.empty:
        return pd.Series(dtype="float64", name="Close")
    return frame["Close"]


def latest_row(features: pd.DataFrame) -> pd.DataFrame:
    """The single most recent feature row, as a one-row frame.

    A frame rather than a Series on purpose: every estimator in the model layer
    expects a 2-D design matrix, and ``.iloc[-1]`` silently produces a 1-D
    object that numpy will happily reinterpret as one sample per *feature*.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        return pd.DataFrame()
    return features.iloc[[-1]]


def family_of(name: str) -> str:
    """The family a generated feature name belongs to."""
    label = str(name)
    if label.startswith("ret_lag_"):
        return "returns"
    if label.startswith("mom_"):
        return "momentum"
    if label.startswith(("vol_", "downside_vol", "skew_", "kurtosis_")):
        return "volatility"
    if label.startswith(("dist_sma_", "sma_")):
        return "trend"
    if label.startswith(("rsi_", "macd_", "bb_", "atr_")):
        return "oscillator"
    if label.startswith(("range_position", "high_", "low_", "bar_range", "close_location")):
        return "range"
    if label == "overnight_gap":
        return "range"
    if label.startswith("volume"):
        return "volume"
    if label.startswith(("dow_", "month_")) or label == "turn_of_month":
        return "calendar"
    return "other"


def describe_feature(name: str) -> str:
    """Plain-English reading of one feature, for the importance table."""
    label = str(name)
    if label in _DESCRIPTIONS:
        return _DESCRIPTIONS[label]
    if label.startswith("ret_lag_"):
        back = int(label.rsplit("_", 1)[-1]) - 1
        if back <= 0:
            return "Today's own completed daily log return."
        return f"The daily log return {back} session{'s' if back > 1 else ''} before today."
    if label.startswith("mom_"):
        return f"Cumulative log return over the last {label.rsplit('_', 1)[-1]} sessions."
    if label.startswith("vol_") and label != "vol_ratio":
        return f"Realised volatility of daily returns over {label.rsplit('_', 1)[-1]} sessions."
    if label.startswith("dist_sma_"):
        window = label.rsplit("_", 1)[-1]
        return f"Close above (+) or below (−) its {window}-session moving average, as a ratio."
    if label.startswith("range_position_"):
        return f"Position inside the trailing {label.rsplit('_', 1)[-1]}-session range."
    if label.startswith("volume_z_"):
        return f"Volume in standard deviations from its {label.rsplit('_', 1)[-1]}-session mean."
    return "Derived price feature."


def feature_catalogue(
    data: Any = None,
    cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
) -> pd.DataFrame:
    """Every feature that would be built, with its family and description.

    Pass the instrument's own data to see exactly what it supports; pass
    nothing for the full catalogue the configuration defines. Columns:
    ``feature, family, description``.
    """
    if data is None:
        index = pd.bdate_range("2000-01-03", periods=max(TRADING_DAYS * 2, 600))
        rng = np.random.default_rng(0)
        walk = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(index)))
        data = pd.DataFrame(
            {
                "Open": walk,
                "High": walk * 1.01,
                "Low": walk * 0.99,
                "Close": walk,
                "Volume": np.full(len(index), 1.0e6),
            },
            index=index,
        )

    features = build_features(data, cfg, technical_cfg)
    if features.empty:
        return pd.DataFrame(columns=["feature", "family", "description"])

    order = {name: position for position, name in enumerate(FEATURE_FAMILIES)}
    rows = [
        {
            "feature": name,
            "family": family_of(name),
            "description": describe_feature(name),
        }
        for name in features.columns
    ]
    catalogue = pd.DataFrame(rows)
    catalogue["_order"] = catalogue["family"].map(lambda f: order.get(f, len(order)))
    return (
        catalogue.sort_values(["_order", "feature"]).drop(columns="_order").reset_index(drop=True)
    )
