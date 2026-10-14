"""Technical-analysis charts for a single security: candles, panels, range bar.

The centrepiece is :func:`price_with_panels` - the three-row layout every
trading terminal converges on, because it is the layout that answers the three
questions a chartist asks in order: where is price relative to its trend and
its volatility envelope (row 1), is the move stretched (row 2), and is the
momentum behind it building or fading (row 3). Sharing one x axis is what
makes the three readable as a single argument rather than three pictures.

Input contract: an OHLCV frame (columns matched case-insensitively, so
yfinance's ``Open/High/Low/Close/Volume`` and a lower-cased frame both work)
plus the indicator frame from ``src.analytics.technicals.indicator_frame``,
whose columns are ``close``, ``SMA20/50/200``, ``EMA12/26``, ``RSI``,
``MACD``/``MACD_SIGNAL``/``MACD_HIST``, ``BB_MIDDLE/UPPER/LOWER/BANDWIDTH/
PERCENT_B`` and ``ATR``. Indicators are never recomputed here - the chart
layer draws, the analytics layer decides.

Colour conventions: gain/loss red-green is spent only where it carries
meaning (candle bodies, MACD histogram sign, volume up/down days). Moving
averages are drawn from the colourblind-safe qualitative ramp, deliberately
skipping its red and green entries so an overlay is never mistaken for a
directional signal. Every period, band width and RSI level comes from
``config.TECHNICALS``, so the guide lines cannot drift from the maths.

No Streamlit import; every function returns a ``go.Figure`` and never raises.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import theme
from src.config import TECHNICALS, TechnicalConfig

__all__ = [
    "price_chart",
    "rsi_chart",
    "macd_chart",
    "volume_chart",
    "price_with_panels",
    "range_52w_bar",
]

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Prefix of the last-resort placeholder message. Any figure carrying it is a
#: bug, not a data condition - the tests assert it never appears.
_FALLBACK_PREFIX = "Chart unavailable: "

#: Overlays offered by :func:`price_chart`, in a fixed order so a given moving
#: average keeps its colour across every figure in the app.
_OVERLAY_ORDER: tuple[str, ...] = ("SMA20", "SMA50", "SMA200", "EMA12", "EMA26")

#: Indices into ``theme.CATEGORICAL`` for the overlays: blue, cyan, purple,
#: teal, olive. The ramp's red and green entries are skipped on purpose - on a
#: candle chart those hues already mean "up day" and "down day".
_OVERLAY_COLOR_INDEX: tuple[int, ...] = (0, 4, 5, 7, 9)

#: Default figure heights (px). The three-panel chart is deliberately taller
#: than the house 280-460 range: at 460 px the RSI and MACD panels collapse to
#: ~90 px each and stop being legible.
_PRICE_HEIGHT = 440
_RSI_HEIGHT = 280
_MACD_HEIGHT = 300
_VOLUME_HEIGHT = 280
_PANELS_HEIGHT = 680
_RANGE_HEIGHT = 280

#: Row proportions of :func:`price_with_panels` - price gets ~60%.
_PANEL_ROW_HEIGHTS: tuple[float, float, float] = (0.60, 0.19, 0.21)

_DEFAULT_SHOW: tuple[str, ...] = ("SMA20", "SMA50", "SMA200", "BB")

_OHLC_KEYS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _never_raises(func: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Guarantee the chart contract: a Figure comes back, always.

    Each function validates its own inputs and returns a *specific*
    :func:`theme.empty_figure` for every anticipated degenerate case; this
    decorator only catches the unanticipated.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
            return theme.empty_figure(
                f"{_FALLBACK_PREFIX}{type(exc).__name__} while drawing this figure."
            )

    return wrapper


def _rgba(color: str, alpha: float) -> str:
    """Re-express a theme colour at a given opacity (hex or rgb/rgba input)."""
    text = str(color).strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) >= 6:
            red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
    if text.startswith("rgb"):
        inner = text[text.find("(") + 1 : text.rfind(")")]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 3:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha:.3f})"
    return text


def _is_finite(value: Any) -> bool:
    """True only for a real, finite number (None/NaN/str -> False)."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _footnote(fig: go.Figure, text: str) -> None:
    """``theme.caption`` nudged clear of the x-axis title.

    The theme pins a caption 0.16 of the plot height below the axis, which is
    exactly where an axis title also sits. Every figure here titles its x
    axis, so the note is shifted down and the bottom margin opened to match -
    a caption that lands on top of the axis title is worse than no caption.
    """
    theme.caption(fig, text)
    notes = fig.layout.annotations or ()
    if notes:
        notes[-1].update(yshift=-26)
    fig.update_layout(margin={"b": 104})


def _price_axis_title(currency_symbol: str) -> str:
    """Axis title carrying its unit, per the house rule."""
    symbol = (currency_symbol or "").strip()
    return f"Price ({symbol})" if symbol else "Price"


def _ohlc_frame(ohlcv: Any) -> pd.DataFrame:
    """Normalise any price input to lower-cased numeric OHLCV columns.

    Accepts an OHLCV DataFrame (case-insensitive, MultiIndex columns
    flattened to their first level), or a bare close Series. Never mutates the
    caller's object; returns an empty frame when nothing usable is present.
    """
    if ohlcv is None:
        return pd.DataFrame()

    if isinstance(ohlcv, pd.Series):
        series = pd.to_numeric(ohlcv.copy(), errors="coerce")
        return pd.DataFrame({"close": series})

    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        return pd.DataFrame()

    frame = ohlcv.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) for col in frame.columns]

    out = pd.DataFrame(index=frame.index)
    seen: set[str] = set()
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _OHLC_KEYS and key not in seen:
            values = frame[column]
            if isinstance(values, pd.DataFrame):  # duplicate labels
                values = values.iloc[:, 0]
            out[key] = pd.to_numeric(values, errors="coerce")
            seen.add(key)

    if "close" not in out.columns and frame.shape[1] == 1:
        out["close"] = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    if out.empty or "close" not in out.columns:
        return pd.DataFrame()

    # Drop holiday gap rows only when there is a real close to keep: a frame
    # whose close is entirely missing may still carry plottable volume.
    if out["close"].notna().any():
        out = out.dropna(subset=["close"])
    # An unsortable mixed index keeps its insertion order rather than raising.
    with contextlib.suppress(TypeError):  # pragma: no cover - unsortable index
        out = out.sort_index()
    return out


def _column_labels(frame: Any) -> set[str]:
    """Lower-cased column labels of a frame (MultiIndex flattened)."""
    if not isinstance(frame, pd.DataFrame):
        return set()
    if isinstance(frame.columns, pd.MultiIndex):
        return {str(col[0]).strip().lower() for col in frame.columns}
    return {str(col).strip().lower() for col in frame.columns}


def _price_failure_reason(ohlcv: Any) -> str:
    """Name the ACTUAL reason a price input yielded nothing plottable.

    "No data" is never a useful message on a dashboard; the reader needs to
    know whether the security has no history, the wrong columns, or a window
    in which every close is missing.
    """
    if ohlcv is None:
        return "No price history is available for this security."
    if isinstance(ohlcv, (pd.Series, pd.DataFrame)):
        if len(ohlcv) == 0:
            return "No price history is available for this security."
        if (
            isinstance(ohlcv, pd.DataFrame)
            and "close" not in _column_labels(ohlcv)
            and ohlcv.shape[1] != 1
        ):
            return "No Close column found in this price history."
        return "Every close in this price history is missing."
    return "The price history is not a table this chart can read."


def _has_any_column(frame: Any, names: set[str]) -> bool:
    """True when the frame carries at least one of ``names`` (any casing)."""
    return bool(_column_labels(frame) & names)


def _indicator_series(indicators: Any, name: str) -> pd.Series | None:
    """Look a column up case-insensitively and return it numeric, or None."""
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        return None
    wanted = name.strip().lower()
    for column in indicators.columns:
        if str(column).strip().lower() == wanted:
            values = indicators[column]
            if isinstance(values, pd.DataFrame):  # duplicate labels
                values = values.iloc[:, 0]
            series = pd.to_numeric(values.copy(), errors="coerce")
            return series if series.notna().any() else None
    return None


def _window_label(index: pd.Index) -> str:
    """``"03 Jan 2024 to 28 Jun 2024"`` for a date index, else a bar count."""
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
            return f"{index[0]:%d %b %Y} to {index[-1]:%d %b %Y}"
    except (TypeError, ValueError):  # pragma: no cover - exotic index
        pass
    return f"{len(index):,} observations"


def _overlay_color(name: str) -> str:
    """Stable qualitative colour for a named moving average."""
    key = name.strip().upper()
    if key in _OVERLAY_ORDER:
        position = _OVERLAY_ORDER.index(key)
    else:
        position = len(_OVERLAY_ORDER)
    index = _OVERLAY_COLOR_INDEX[position % len(_OVERLAY_COLOR_INDEX)]
    return theme.CATEGORICAL[index % len(theme.CATEGORICAL)]


def _axis_suffix(row: int | None) -> str:
    """``None|1 -> "y"``, ``2 -> "y2"`` - for annotations pinned to a panel."""
    if row is None or row <= 1:
        return "y"
    return f"y{int(row)}"


def _money_axis_row(fig: go.Figure, currency_symbol: str, row: int) -> None:
    """Currency formatting for ONE subplot row.

    ``theme.money_axis`` has no row/col hook (it formats every y axis at
    once), which would put a currency prefix on the RSI and MACD panels.
    """
    fig.update_yaxes(
        tickprefix=currency_symbol or "",
        tickformat="~s",
        separatethousands=True,
        row=row,
        col=1,
    )


def _hline_row(
    fig: go.Figure,
    y: float,
    label: str,
    color: str,
    row: int | None = None,
    dash: str = "dot",
) -> None:
    """Annotated guide line, optionally scoped to one subplot row.

    Mirrors ``theme.add_reference_line`` for the subplot case, which the theme
    helper does not cover.
    """
    kwargs: dict[str, Any] = {
        "y": y,
        "line_width": 1.0,
        "line_dash": dash,
        "line_color": color,
    }
    if label:
        kwargs |= {
            "annotation_text": label,
            "annotation_position": "top left",
            "annotation_font_size": 10,
            "annotation_font_color": color,
        }
    if row is not None:
        kwargs |= {"row": row, "col": 1}
    fig.add_hline(**kwargs)


def _panel_note(fig: go.Figure, text: str, row: int) -> None:
    """Muted centred note inside one panel when its indicator is missing."""
    fig.add_annotation(
        text=text,
        xref="paper",
        yref=f"{_axis_suffix(row)} domain",
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"size": 11, "color": theme.palette()["subtext"]},
    )


def _normalise_show(show: Iterable[str] | None) -> set[str]:
    """Upper-cased overlay request set; ``BOLLINGER`` is an alias for ``BB``."""
    if show is None:
        return set()
    wanted: set[str] = set()
    for item in show:
        key = str(item).strip().upper()
        if key in {"BOLLINGER", "BBANDS", "BOLLINGER BANDS"}:
            key = "BB"
        wanted.add(key)
    return wanted


# ---------------------------------------------------------------------------
# shared trace builders
# ---------------------------------------------------------------------------
def _add_price_traces(
    fig: go.Figure,
    frame: pd.DataFrame,
    indicators: pd.DataFrame | None,
    show: set[str],
    currency_symbol: str,
    cfg: TechnicalConfig,
    row: int | None = None,
) -> None:
    """Bollinger envelope, moving averages and candles, in paint order.

    The envelope is added first so its fill sits under the averages and the
    candles: ``fill="tonexty"`` fills to the immediately preceding trace, so
    the lower band must be added directly before the upper one.
    """
    palette = theme.palette()
    symbol = currency_symbol or ""

    def add(trace: go.Scatter | go.Candlestick) -> None:
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=1)

    upper = _indicator_series(indicators, "BB_UPPER")
    lower = _indicator_series(indicators, "BB_LOWER")
    if "BB" in show and upper is not None and lower is not None:
        band_name = f"Bollinger ({cfg.bollinger_period}, {cfg.bollinger_std:g})"
        add(
            go.Scatter(
                x=lower.index,
                y=lower.to_numpy(dtype=float),
                mode="lines",
                name=band_name,
                legendgroup="bb",
                showlegend=False,
                line={"width": 0.9, "color": _rgba(palette["accent"], 0.45)},
                hovertemplate=f"Lower band  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )
        add(
            go.Scatter(
                x=upper.index,
                y=upper.to_numpy(dtype=float),
                mode="lines",
                name=band_name,
                legendgroup="bb",
                showlegend=True,
                fill="tonexty",
                fillcolor=_rgba(palette["accent"], 0.10),
                line={"width": 0.9, "color": _rgba(palette["accent"], 0.45)},
                hovertemplate=f"Upper band  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )

    for name in _OVERLAY_ORDER:
        if name not in show:
            continue
        series = _indicator_series(indicators, name)
        if series is None:
            continue
        label = f"{name[:3]} {name[3:]}" if name[3:].isdigit() else name
        add(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(dtype=float),
                mode="lines",
                name=label,
                line={"width": 1.5, "color": _overlay_color(name)},
                hovertemplate=f"{label}  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )

    has_ohlc = all(key in frame.columns for key in ("open", "high", "low"))
    if has_ohlc and frame[["open", "high", "low"]].notna().any().all():
        add(
            go.Candlestick(
                x=frame.index,
                open=frame["open"].to_numpy(dtype=float),
                high=frame["high"].to_numpy(dtype=float),
                low=frame["low"].to_numpy(dtype=float),
                close=frame["close"].to_numpy(dtype=float),
                name="Price",
                increasing={
                    "line": {"color": palette["gain"], "width": 1.0},
                    "fillcolor": palette["gain"],
                },
                decreasing={
                    "line": {"color": palette["loss"], "width": 1.0},
                    "fillcolor": palette["loss"],
                },
                whiskerwidth=0.0,
                hovertemplate=(
                    f"O {symbol}%{{open:,.2f}}   H {symbol}%{{high:,.2f}}<br>"
                    f"L {symbol}%{{low:,.2f}}   C {symbol}%{{close:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )
    else:
        add(
            go.Scatter(
                x=frame.index,
                y=frame["close"].to_numpy(dtype=float),
                mode="lines",
                name="Close",
                line={"width": 1.8, "color": palette["accent"]},
                hovertemplate=f"Close  {symbol}%{{y:,.2f}}<extra></extra>",
            )
        )


def _add_rsi_traces(
    fig: go.Figure,
    series: pd.Series,
    cfg: TechnicalConfig,
    row: int | None = None,
    shade: bool = True,
    legend: bool = False,
) -> None:
    """RSI line plus the overbought / midline / oversold guides."""
    palette = theme.palette()
    trace = go.Scatter(
        x=series.index,
        y=series.to_numpy(dtype=float),
        mode="lines",
        name=f"RSI ({cfg.rsi_period})",
        showlegend=legend,
        line={"width": 1.6, "color": palette["accent_2"]},
        hovertemplate=f"RSI ({cfg.rsi_period})  %{{y:.1f}}<extra></extra>",
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)

    if shade:
        rect: dict[str, Any] = {
            "fillcolor": _rgba(palette["warning"], 0.08),
            "line_width": 0,
            "layer": "below",
        }
        if row is not None:
            rect |= {"row": row, "col": 1}
        fig.add_hrect(y0=cfg.rsi_overbought, y1=100, **rect)
        fig.add_hrect(y0=0, y1=cfg.rsi_oversold, **rect)

    _hline_row(
        fig, cfg.rsi_overbought, f"{cfg.rsi_overbought:g} overbought", palette["warning"], row=row
    )
    _hline_row(fig, cfg.rsi_oversold, f"{cfg.rsi_oversold:g} oversold", palette["warning"], row=row)
    _hline_row(fig, 50.0, "", palette["muted"], row=row, dash="dot")

    if row is None:
        fig.update_yaxes(range=[0, 100], dtick=25)
    else:
        fig.update_yaxes(range=[0, 100], dtick=25, row=row, col=1)


def _macd_columns(
    frame: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    """Resolve MACD/signal/histogram from either naming convention.

    ``analytics.technicals.macd`` returns ``macd/signal/histogram``;
    ``indicator_frame`` returns ``MACD/MACD_SIGNAL/MACD_HIST``. Both are
    accepted so the chart works with either producer.
    """
    line = _indicator_series(frame, "macd")
    signal = _indicator_series(frame, "signal")
    if signal is None:
        signal = _indicator_series(frame, "macd_signal")
    hist = _indicator_series(frame, "histogram")
    if hist is None:
        hist = _indicator_series(frame, "macd_hist")
    return line, signal, hist


def _add_macd_traces(
    fig: go.Figure,
    line: pd.Series | None,
    signal: pd.Series | None,
    hist: pd.Series | None,
    cfg: TechnicalConfig,
    row: int | None = None,
    legend: bool = True,
) -> None:
    """Signed histogram bars under the MACD and signal lines."""
    palette = theme.palette()

    def add(trace: go.Scatter | go.Bar) -> None:
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=1)

    if hist is not None:
        values = hist.to_numpy(dtype=float)
        add(
            go.Bar(
                x=hist.index,
                y=values,
                name="Histogram",
                showlegend=False,
                marker={"color": theme.sign_colors(values), "line": {"width": 0}},
                opacity=0.55,
                hovertemplate="Histogram  %{y:,.3f}<extra></extra>",
            )
        )
    if line is not None:
        add(
            go.Scatter(
                x=line.index,
                y=line.to_numpy(dtype=float),
                mode="lines",
                name=f"MACD ({cfg.macd_fast},{cfg.macd_slow})",
                showlegend=legend,
                line={"width": 1.6, "color": theme.CATEGORICAL[0]},
                hovertemplate="MACD  %{y:,.3f}<extra></extra>",
            )
        )
    if signal is not None:
        add(
            go.Scatter(
                x=signal.index,
                y=signal.to_numpy(dtype=float),
                mode="lines",
                name=f"Signal ({cfg.macd_signal})",
                showlegend=legend,
                line={"width": 1.4, "color": palette["warning"], "dash": "dot"},
                hovertemplate="Signal  %{y:,.3f}<extra></extra>",
            )
        )


# ---------------------------------------------------------------------------
# price
# ---------------------------------------------------------------------------
@_never_raises
def price_chart(
    ohlcv: pd.DataFrame | pd.Series,
    indicators: pd.DataFrame | None = None,
    show: Sequence[str] | None = _DEFAULT_SHOW,
    currency_symbol: str = "",
    ticker: str = "",
    cfg: TechnicalConfig = TECHNICALS,
    height: int | None = None,
) -> go.Figure:
    """Daily candles with the requested moving averages and Bollinger envelope.

    Candle bodies use the gain/loss pair (close above or below the open); the
    envelope is a translucent accent band; each moving average keeps a fixed
    qualitative colour so it is recognisable across figures. Falls back to a
    close line when Open/High/Low are absent (a close-only history, which is
    what ``fetch_price_history`` returns).

    Parameters
    ----------
    ohlcv:
        OHLCV frame (case-insensitive columns) or a bare close Series.
    indicators:
        Frame from ``analytics.technicals.indicator_frame``; ``None`` draws
        price alone.
    show:
        Overlay names to draw - any of ``SMA20/SMA50/SMA200/EMA12/EMA26`` plus
        ``BB`` for the Bollinger envelope. Overlays that are requested but
        absent from ``indicators`` are silently skipped.
    currency_symbol, ticker:
        Labelling only.
    cfg:
        Supplies the Bollinger period and width printed in the legend, so the
        label can never disagree with the maths that produced the band.
    height:
        Caller override; defaults to 440 px.

    Assumes one row per trading session, already adjusted for splits and
    dividends (``auto_adjust=True`` upstream); gaps are shown as gaps, never
    interpolated.
    """
    frame = _ohlc_frame(ohlcv)
    if frame.empty:
        return theme.empty_figure(_price_failure_reason(ohlcv))
    closes = int(frame["close"].notna().sum())
    if closes == 0:
        return theme.empty_figure("Every close in this price history is missing.")
    if closes < 2:
        return theme.empty_figure(
            "Only one price observation - a price chart needs a history to draw."
        )

    fig = go.Figure()
    _add_price_traces(fig, frame, indicators, _normalise_show(show), currency_symbol, cfg, row=None)
    if not fig.data:
        return theme.empty_figure("Nothing plottable in this price history.")

    name = (ticker or "").strip()
    title = f"{name} price" if name else "Price"
    theme.apply(
        fig,
        title,
        subtitle=f"Daily candles - {_window_label(frame.index)}",
        height=height or _PRICE_HEIGHT,
        unified_hover=True,
    )
    fig.update_xaxes(title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(title_text=_price_axis_title(currency_symbol))
    theme.money_axis(fig, currency_symbol, axis="y")
    return fig


# ---------------------------------------------------------------------------
# oscillator panels
# ---------------------------------------------------------------------------
@_never_raises
def rsi_chart(
    rsi: pd.Series,
    cfg: TechnicalConfig = TECHNICALS,
    ticker: str = "",
    height: int | None = None,
) -> go.Figure:
    """Relative Strength Index with its overbought / oversold guides.

    The 0-100 axis is fixed so the eye can compare two securities without
    re-reading the scale, the conventional 70/30 levels are drawn from
    ``config.TECHNICALS`` and their zones lightly shaded, and 50 is marked
    because a persistent RSI above or below it is the trend read.

    Parameters
    ----------
    rsi:
        RSI series from ``analytics.technicals.rsi`` (Wilder-smoothed), with
        leading NaNs for the warm-up period.
    cfg:
        Source of the period and the overbought/oversold levels.
    ticker, height:
        Labelling and a caller height override (default 280 px).

    Assumes Wilder's smoothing upstream; a rolling-mean RSI plots on the same
    axis but crosses these thresholds at different times.
    """
    if not isinstance(rsi, pd.Series) or rsi.empty:
        return theme.empty_figure("No RSI series was supplied for this security.")
    series = pd.to_numeric(rsi.copy(), errors="coerce").dropna()
    if series.empty:
        return theme.empty_figure(
            f"RSI is still warming up - it needs more than {cfg.rsi_period} sessions."
        )
    if len(series) < 2:
        return theme.empty_figure("Only one RSI observation - nothing to plot through time.")

    fig = go.Figure()
    _add_rsi_traces(fig, series, cfg, row=None, shade=True, legend=False)

    name = (ticker or "").strip()
    theme.apply(
        fig,
        f"{name} RSI" if name else f"RSI ({cfg.rsi_period})",
        subtitle=(f"Wilder-smoothed, {cfg.rsi_period}-session - {_window_label(series.index)}"),
        height=height or _RSI_HEIGHT,
        legend=False,
        unified_hover=True,
    )
    fig.update_xaxes(title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(title_text="RSI (0-100)")
    _footnote(
        fig,
        f"Above {cfg.rsi_overbought:g} is conventionally overbought and below "
        f"{cfg.rsi_oversold:g} oversold, but in a strong trend RSI can stay pinned "
        "for weeks - it is a stretch gauge, not a timing signal.",
    )
    return fig


@_never_raises
def macd_chart(
    macd: pd.DataFrame,
    cfg: TechnicalConfig = TECHNICALS,
    ticker: str = "",
    height: int | None = None,
) -> go.Figure:
    """MACD line, signal line and the signed convergence histogram.

    The histogram is the signed quantity here, so it takes the gain/loss pair
    and an emphasised zero line: a bar crossing zero is the momentum flip the
    indicator exists to show.

    Parameters
    ----------
    macd:
        Frame with either ``macd/signal/histogram`` (the
        ``analytics.technicals.macd`` output) or ``MACD/MACD_SIGNAL/MACD_HIST``
        (the ``indicator_frame`` layout).
    cfg:
        Supplies the spans quoted in the legend and subtitle.
    ticker, height:
        Labelling and a caller height override (default 300 px).

    Assumes both spans were computed on the same close series; the values are
    in price units, so the scale is not comparable across securities.
    """
    if not isinstance(macd, pd.DataFrame) or macd.empty:
        return theme.empty_figure("No MACD frame was supplied for this security.")

    line, signal, hist = _macd_columns(macd)
    if line is None and signal is None and hist is None:
        known = {"macd", "signal", "histogram", "macd_signal", "macd_hist"}
        if _has_any_column(macd, known):
            return theme.empty_figure(
                f"MACD has no finite values yet - it needs more than {cfg.macd_slow} sessions."
            )
        return theme.empty_figure(
            "No macd / signal / histogram columns found in the supplied frame."
        )
    lengths = [len(s.dropna()) for s in (line, signal, hist) if s is not None]
    if not lengths or max(lengths) < 2:
        return theme.empty_figure(
            f"MACD is still warming up - it needs more than {cfg.macd_slow} sessions."
        )

    fig = go.Figure()
    _add_macd_traces(fig, line, signal, hist, cfg, row=None, legend=True)
    theme.add_zero_line(fig, axis="y")

    name = (ticker or "").strip()
    index = (line if line is not None else (signal if signal is not None else hist)).index
    theme.apply(
        fig,
        f"{name} MACD" if name else "MACD",
        subtitle=(
            f"EMA {cfg.macd_fast} minus EMA {cfg.macd_slow}, signal EMA "
            f"{cfg.macd_signal} - {_window_label(index)}"
        ),
        height=height or _MACD_HEIGHT,
        unified_hover=True,
        barmode="relative",
    )
    fig.update_xaxes(title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(title_text="MACD (price units)")
    return fig


@_never_raises
def volume_chart(
    ohlcv: pd.DataFrame,
    currency_symbol: str = "",  # noqa: ARG001 - kept so every builder here shares one signature
    ticker: str = "",
    cfg: TechnicalConfig = TECHNICALS,
    height: int | None = None,
) -> go.Figure:
    """Traded volume, coloured by the session's direction.

    Bars take the gain colour when the close is at or above the prior close
    and the loss colour otherwise - the standard reading, which turns raw
    volume into "who was in control". A moving average of volume is overlaid
    so a spike is judged against the security's own norm rather than an
    absolute number.

    Parameters
    ----------
    ohlcv:
        Frame containing at least ``Volume`` (and ideally ``Close``, without
        which the bars fall back to a single neutral colour).
    currency_symbol:
        Unused for the axis (volume is share count) but kept in the signature
        so every chart in this module takes the same labelling arguments.
    ticker, cfg, height:
        Labelling, the averaging window (``cfg.sma_windows[0]``) and a caller
        height override (default 280 px).

    Assumes one bar per session; volume is share count, not turnover, so it is
    not comparable across securities with very different prices.
    """
    frame = _ohlc_frame(ohlcv)
    if frame.empty:
        return theme.empty_figure(_price_failure_reason(ohlcv))
    if "volume" not in frame.columns:
        return theme.empty_figure("This price history carries no Volume column.")
    volume = frame["volume"].dropna()
    if volume.empty:
        return theme.empty_figure("Volume is missing for every session in this window.")
    if len(volume) < 2:
        return theme.empty_figure("Only one volume observation - nothing to plot through time.")

    palette = theme.palette()
    if "close" in frame.columns and frame["close"].notna().any():
        # Session direction is a signed quantity, so it takes the gain/loss
        # pair - but the first bar (and any gap) has no prior close to compare
        # against and must not be coloured as though it did.
        direction = frame["close"].reindex(volume.index).diff().to_numpy(dtype=float)
        colors = [
            palette["muted"] if not np.isfinite(delta) else color
            for delta, color in zip(direction, theme.sign_colors(direction))
        ]
        direction_note = "Bars coloured by session direction"
    else:
        colors = [palette["muted"]] * len(volume)
        direction_note = "No close data - session direction unavailable"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=volume.index,
            y=volume.to_numpy(dtype=float),
            name="Volume",
            showlegend=False,
            marker={"color": colors, "line": {"width": 0}},
            opacity=0.65,
            hovertemplate="Volume  %{y:,.0f} shares<extra></extra>",
        )
    )

    window = int(cfg.sma_windows[0]) if getattr(cfg, "sma_windows", ()) else 20
    if len(volume) > window:
        average = volume.rolling(window, min_periods=window).mean()
        fig.add_trace(
            go.Scatter(
                x=average.index,
                y=average.to_numpy(dtype=float),
                mode="lines",
                name=f"{window}-session average",
                line={"width": 1.6, "color": palette["accent"]},
                hovertemplate=(f"{window}-session average  %{{y:,.0f}} shares<extra></extra>"),
            )
        )

    name = (ticker or "").strip()
    theme.apply(
        fig,
        f"{name} volume" if name else "Traded volume",
        subtitle=f"{direction_note} - {_window_label(volume.index)}",
        height=height or _VOLUME_HEIGHT,
        unified_hover=True,
    )
    fig.update_xaxes(title_text="Date", rangeslider_visible=False)
    fig.update_yaxes(title_text="Volume (shares)", tickformat="~s", separatethousands=True)
    return fig


# ---------------------------------------------------------------------------
# the three-panel terminal view
# ---------------------------------------------------------------------------
@_never_raises
def price_with_panels(
    ohlcv: pd.DataFrame | pd.Series,
    indicators: pd.DataFrame,
    currency_symbol: str = "",
    ticker: str = "",
    show: Sequence[str] | None = _DEFAULT_SHOW,
    cfg: TechnicalConfig = TECHNICALS,
    height: int | None = None,
) -> go.Figure:
    """The full technical view: price, RSI and MACD on one shared time axis.

    Row 1 (60% of the height) carries the candles, the Bollinger envelope and
    the moving averages; row 2 the RSI with its 70/30 guides; row 3 the MACD
    lines over the signed histogram. The x axis is shared and the range slider
    is off, so panning row 1 pans all three and no vertical space is spent on
    a control the panels replace.

    Parameters
    ----------
    ohlcv:
        OHLCV frame (case-insensitive columns) or a bare close Series.
    indicators:
        Frame from ``analytics.technicals.indicator_frame`` supplying
        ``BB_UPPER``/``BB_LOWER``, the ``SMA``/``EMA`` overlays, ``RSI`` and
        ``MACD``/``MACD_SIGNAL``/``MACD_HIST``. A panel whose indicator is
        absent keeps its slot and says so rather than silently vanishing -
        the layout must not shift under the reader between securities.
    currency_symbol, ticker:
        Labelling.
    show:
        Row-1 overlays, as in :func:`price_chart`.
    cfg:
        Periods and levels quoted on the guides and in the legend.
    height:
        Caller override. Defaults to 680 px: three stacked panels need the
        room, and squeezing them into the usual single-chart height makes the
        oscillators unreadable.

    Assumes all three panels were computed from the same close series, so a
    reading in one row is directly comparable with the same date in another.
    """
    frame = _ohlc_frame(ohlcv)
    if frame.empty:
        return theme.empty_figure(_price_failure_reason(ohlcv))
    closes = int(frame["close"].notna().sum())
    if closes == 0:
        return theme.empty_figure("Every close in this price history is missing.")
    if closes < 2:
        return theme.empty_figure("Only one price observation - the panels need a history to draw.")
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        return theme.empty_figure("No indicator frame supplied - the RSI and MACD panels need one.")

    palette = theme.palette()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.032,
        row_heights=list(_PANEL_ROW_HEIGHTS),
    )

    _add_price_traces(fig, frame, indicators, _normalise_show(show), currency_symbol, cfg, row=1)

    rsi_series = _indicator_series(indicators, "RSI")
    if rsi_series is not None and rsi_series.dropna().size >= 2:
        _add_rsi_traces(fig, rsi_series.dropna(), cfg, row=2, shade=True, legend=False)
    else:
        _panel_note(fig, f"RSI needs more than {cfg.rsi_period} sessions", row=2)
        fig.update_yaxes(range=[0, 100], dtick=25, row=2, col=1)

    line, signal, hist = _macd_columns(indicators)
    if any(s is not None and s.dropna().size >= 2 for s in (line, signal, hist)):
        _add_macd_traces(fig, line, signal, hist, cfg, row=3, legend=True)
        fig.update_yaxes(
            zeroline=True, zerolinecolor=palette["axis"], zerolinewidth=1, row=3, col=1
        )
    else:
        _panel_note(fig, f"MACD needs more than {cfg.macd_slow} sessions", row=3)

    name = (ticker or "").strip()
    theme.apply(
        fig,
        f"{name} technical view" if name else "Technical view",
        subtitle=(
            f"Daily candles, Bollinger ({cfg.bollinger_period}, {cfg.bollinger_std:g}), "
            f"RSI ({cfg.rsi_period}), MACD ({cfg.macd_fast}, {cfg.macd_slow}, "
            f"{cfg.macd_signal}) - {_window_label(frame.index)}"
        ),
        height=height or _PANELS_HEIGHT,
        unified_hover=True,
        barmode="relative",
    )
    # One shared axis, one set of date labels, and no range slider stealing
    # height from the panels that replaced it.
    fig.update_xaxes(rangeslider_visible=False, showticklabels=False)
    fig.update_xaxes(title_text="Date", showticklabels=True, row=3, col=1)
    fig.update_yaxes(title_text=_price_axis_title(currency_symbol), row=1, col=1)
    fig.update_yaxes(title_text=f"RSI ({cfg.rsi_period})", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    _money_axis_row(fig, currency_symbol, row=1)
    return fig


# ---------------------------------------------------------------------------
# 52-week range
# ---------------------------------------------------------------------------
@_never_raises
def range_52w_bar(
    price: float | Mapping[str, Any],
    low: float | None = None,
    high: float | None = None,
    ticker: str = "",
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Bullet bar placing the last price inside its 52-week range.

    A muted track spans low to high, a coloured segment fills it from the low
    up to the last price, and a tick marks the price itself; the low, high and
    last values are annotated on the chart so it reads without a tooltip. The
    fill takes the gain colour in the upper half of the range and the loss
    colour in the lower half - position within the range is the whole point of
    the chart, so it earns the semantic pair.

    Parameters
    ----------
    price:
        Last price. As a convenience the whole
        ``analytics.technicals.distance_from_52w`` dict may be passed here
        instead, in which case its ``last``/``low``/``high`` keys are used.
    low, high:
        52-week extremes, from the same source as ``price``.
    ticker, currency_symbol, height:
        Labelling and a caller height override (default 280 px).

    Assumes all three numbers come from ONE window of daily closes - mixing a
    provider's 52-week high with a locally computed close would place the
    marker against a range it was never measured in. A price outside the range
    (a fresh high or low) is drawn honestly outside the track rather than
    clamped.
    """
    if isinstance(price, Mapping):
        source: Mapping[str, Any] = price
        last_value = source.get("last")
        low = source.get("low") if low is None else low
        high = source.get("high") if high is None else high
    else:
        last_value = price

    if not (_is_finite(last_value) and _is_finite(low) and _is_finite(high)):
        return theme.empty_figure(
            "The 52-week high, low or last price is not available for this security."
        )

    last_f, low_f, high_f = float(last_value), float(low), float(high)
    if high_f < low_f:
        return theme.empty_figure("The 52-week high is below the low - the range is inconsistent.")
    span = high_f - low_f
    if span <= 0:
        return theme.empty_figure(
            "The 52-week high equals the low - there is no range to position within."
        )

    palette = theme.palette()
    symbol = currency_symbol or ""
    position = (last_f - low_f) / span * 100.0
    from_high = (last_f / high_f - 1.0) * 100.0 if high_f > 0 else float("nan")
    from_low = (last_f / low_f - 1.0) * 100.0 if low_f > 0 else float("nan")
    fill_color = palette["gain"] if position >= 50.0 else palette["loss"]
    fill_end = min(max(last_f, low_f), high_f)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[span],
            base=[low_f],
            y=[0.0],
            orientation="h",
            width=0.5,
            name="52-week range",
            showlegend=False,
            marker={"color": _rgba(palette["muted"], 0.30), "line": {"width": 0}},
            customdata=[[low_f, high_f, span]],
            hovertemplate=(
                f"52-week range<br>low {symbol}%{{customdata[0]:,.2f}} to high "
                f"{symbol}%{{customdata[1]:,.2f}}<br>"
                f"width {symbol}%{{customdata[2]:,.2f}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[max(fill_end - low_f, 0.0)],
            base=[low_f],
            y=[0.0],
            orientation="h",
            width=0.28,
            name="Position in range",
            showlegend=False,
            marker={"color": _rgba(fill_color, 0.85), "line": {"width": 0}},
            customdata=[[position]],
            hovertemplate=("%{customdata[0]:.0f}% of the way from the 52-week low<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_f],
            y=[0.0],
            mode="markers",
            name="Last price",
            showlegend=False,
            marker={
                "symbol": "line-ns",
                "size": 26,
                "line": {"width": 2.6, "color": palette["text"]},
                "color": palette["text"],
            },
            customdata=[[position, from_high, from_low]],
            hovertemplate=(
                f"Last {symbol}%{{x:,.2f}}<br>"
                "%{customdata[0]:.0f}% of the range<br>"
                "%{customdata[1]:+.1f}% from the high, "
                "%{customdata[2]:+.1f}% from the low"
                "<extra></extra>"
            ),
        )
    )

    theme.annotate(
        fig,
        f"52w low  {symbol}{low_f:,.2f}",
        x=low_f,
        y=-0.52,
        xanchor="left",
        yanchor="top",
    )
    theme.annotate(
        fig,
        f"52w high  {symbol}{high_f:,.2f}",
        x=high_f,
        y=-0.52,
        xanchor="right",
        yanchor="top",
    )
    if position < 15.0:
        last_anchor = "left"
    elif position > 85.0:
        last_anchor = "right"
    else:
        last_anchor = "center"
    theme.annotate(
        fig,
        f"last  {symbol}{last_f:,.2f}",
        x=last_f,
        y=0.42,
        xanchor=last_anchor,
        yanchor="bottom",
        color=palette["text"],
        size=12.5,
    )

    pad = span * 0.09
    left = min(low_f, last_f) - pad
    right = max(high_f, last_f) + pad

    name = (ticker or "").strip()
    theme.apply(
        fig,
        f"{name} 52-week range" if name else "52-week range",
        subtitle=(
            f"Last price sits {position:.0f}% of the way from the low to the high - "
            f"{from_high:+.1f}% from the high, {from_low:+.1f}% above the low"
        ),
        height=height or _RANGE_HEIGHT,
        legend=False,
        barmode="overlay",
    )
    fig.update_xaxes(title_text=_price_axis_title(currency_symbol), range=[left, right])
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], fixedrange=True)
    theme.money_axis(fig, currency_symbol, axis="x")
    _footnote(
        fig,
        "Range measured on daily closes over the trailing 52 weeks; intraday "
        "extremes printed by a data provider will differ slightly.",
    )
    return fig
