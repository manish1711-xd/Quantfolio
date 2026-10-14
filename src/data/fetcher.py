"""Market-data access layer (Yahoo Finance via ``yfinance``).

Design notes
------------
* Every network call in the whole project lives in this module. The rest of
  the codebase is pure maths, which keeps it unit-testable offline: tests
  monkeypatch ``yfinance.Ticker`` and never touch the wire.
* Yahoo Finance does NOT guarantee real-time quotes. NSE/BSE prices are
  typically delayed (~15 minutes) and outside market hours you simply get the
  latest close. Every value returned here therefore carries its own timestamp
  and a freshness label so the UI can state *exactly* how fresh the data is,
  and never over-claims "live".
* One bad ticker must never break the whole request: the batch helpers return
  ``(results, errors)`` pairs instead of raising, so the app can keep working
  with whatever data it *did* get.
* No caching lives here. The TTLs in ``src.config`` (``QUOTE_TTL_SECONDS`` and
  friends) are applied by the UI layer with ``st.cache_data``; keeping this
  module cache-free makes it a clean, deterministic, monkeypatchable boundary.

Conventions
-----------
* ``auto_adjust=True`` everywhere -> the ``Close`` column is already adjusted
  for splits and dividends (the modern yfinance replacement for "Adj Close").
* Every history index is converted to tz-naive, time-stripped calendar dates
  (see ``_naive_dates``) so NSE (Asia/Kolkata) and US (America/New_York)
  series can be aligned in one frame.
* FX: ``fetch_fx_rate(a, b)`` is the price of ONE unit of ``a`` expressed in
  ``b``; multiplying a value in ``a`` by that rate converts it to ``b``.
* Failures are reported, never invented. Scalars degrade to ``float("nan")``,
  frames/series to empty objects, and profiles to a bare ticker-only record.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.config import (
    DEFAULT_BENCHMARK,
    DEFAULT_BENCHMARKS,
    DEFAULT_PERIOD,
    USD_QUOTED_PAIRS,
    VALID_PERIODS,
)
from src.models import Quote, SecurityProfile

logger = logging.getLogger(__name__)

__all__ = [
    "DataFetchError",
    "DEFAULT_BENCHMARK_SYMBOL",
    "OHLCV_COLUMNS",
    "fetch_quote",
    "fetch_quotes",
    "fetch_price_history",
    "fetch_ohlcv",
    "fetch_benchmark_history",
    "fetch_profile",
    "fetch_profiles",
    "fetch_statements",
    "fetch_dividends",
    "fetch_fx_rate",
    "fetch_fx_rates",
    "fx_pair_symbol",
    "search_symbols",
    "lookup_isin",
    "lookup_isins",
    "OPTION_COLUMNS",
    "option_chain_available",
    "fetch_option_expiries",
    "fetch_option_chain",
    "validate_ticker",
]

#: Symbol of the default benchmark, resolved through the config mapping so the
#: default here can never drift from the one the UI offers.
DEFAULT_BENCHMARK_SYMBOL: str = DEFAULT_BENCHMARKS[DEFAULT_BENCHMARK]

#: Column order returned by :func:`fetch_ohlcv`.
OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

#: Short window used to recover a last close when the quote endpoint is cold.
_QUOTE_FALLBACK_PERIOD = "5d"

_SUFFIX_HINT = (
    "check the symbol (Indian stocks need the .NS or .BO suffix, "
    "e.g. RELIANCE.NS) and your internet connection"
)


class DataFetchError(Exception):
    """Raised when market data for a ticker cannot be retrieved."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clean_ticker(ticker: object) -> str:
    """Normalise a user-typed symbol: strip whitespace, upper-case."""
    return str(ticker).strip().upper()


def _clean_ccy(code: object) -> str:
    """Normalise an ISO-4217 currency code; returns "" for junk input."""
    text = str(code).strip().upper()
    return text if text and text not in {"NAN", "NONE", "N/A"} else ""


def _dedupe(tickers: Iterable[str]) -> list[str]:
    """Upper-case, strip, drop blanks, keep first-seen order."""
    return [t for t in dict.fromkeys(_clean_ticker(t) for t in tickers) if t]


def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Strip timezone and time-of-day so mixed exchanges align on calendar dates.

    NSE history comes back tz-aware in Asia/Kolkata, US tickers in
    America/New_York. Concatenating those directly produces a broken object
    index, so we normalise everything to naive calendar dates first.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _finite_float(value: object) -> float | None:
    """Coerce to ``float``; ``None`` for missing, NaN, infinite or unparseable."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _text(value: object) -> str | None:
    """Coerce to a trimmed string; ``None`` for missing/blank/NaN sentinels."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
    except TypeError:  # pragma: no cover - defensive
        return None
    out = str(value).strip()
    return out if out and out.upper() not in {"NAN", "NONE", "N/A"} else None


def _validate_period(period: str) -> str:
    """Check a yfinance period string against ``config.VALID_PERIODS``.

    An unsupported window is a programming error (the UI only ever offers the
    configured list), not degenerate market data, so this raises rather than
    silently substituting a different window and mislabelling the result.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got {period!r}")
    return period


def _fi_get(fast_info: object, key: str) -> object:
    """Read one field from a yfinance ``fast_info`` (attribute or mapping)."""
    try:
        value = getattr(fast_info, key, None)
    except Exception:  # noqa: BLE001 - lazy properties can throw on network errors
        value = None
    if value is None:
        try:
            value = fast_info[key]  # type: ignore[index]
        except Exception:  # noqa: BLE001 - not a mapping, or key absent
            value = None
    return value


def _fast_info_snapshot(tk: object) -> dict[str, object]:
    """Best-effort read of the near-live quote endpoint. Never raises."""
    snap: dict[str, object] = {
        "price": None,
        "currency": None,
        "previous_close": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
    }
    try:
        fast_info = tk.fast_info
    except Exception:  # noqa: BLE001 - any failure just means "use the fallback"
        return snap
    if fast_info is None:
        return snap

    snap["price"] = _finite_float(_fi_get(fast_info, "last_price"))
    snap["currency"] = _text(_fi_get(fast_info, "currency"))
    snap["previous_close"] = _finite_float(_fi_get(fast_info, "previous_close"))
    snap["day_high"] = _finite_float(_fi_get(fast_info, "day_high"))
    snap["day_low"] = _finite_float(_fi_get(fast_info, "day_low"))
    volume = _fi_get(fast_info, "last_volume")
    if volume is None:
        volume = _fi_get(fast_info, "regular_market_volume")
    snap["volume"] = _finite_float(volume)
    return snap


def _history(ticker: str, period: str) -> pd.DataFrame:
    """Raw daily history for one symbol. Raises on provider failure."""
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


def _empty_ohlcv() -> pd.DataFrame:
    """Correctly-typed, correctly-named empty OHLCV frame."""
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in OHLCV_COLUMNS})
    frame.index = pd.DatetimeIndex([], name="Date")
    return frame


def _empty_dividends() -> pd.Series:
    """Correctly-typed empty dividend series."""
    return pd.Series(
        dtype="float64",
        index=pd.DatetimeIndex([], name="Date"),
        name="dividend",
    )


# ---------------------------------------------------------------------------
# latest quote
# ---------------------------------------------------------------------------
def fetch_quote(ticker: str) -> Quote:
    """Latest available price for one ticker.

    Strategy: try the fast quote endpoint first (freshest, and cheap), fall
    back to the last daily close from a 5-day history request. Day high/low,
    volume and the previous close are filled in when the provider supplies
    them; ``Quote.day_change_pct`` is ``(price / previous_close - 1) * 100``
    and is NaN when the previous close is unknown.

    Raises ``DataFetchError`` with a human-readable reason if neither the
    quote endpoint nor the history fallback yields a usable price.
    """
    ticker = _clean_ticker(ticker)
    if not ticker:
        raise DataFetchError("Empty ticker symbol.")

    now = datetime.now(timezone.utc)
    try:
        tk = yf.Ticker(ticker)
    except Exception as exc:
        raise DataFetchError(f"{ticker}: request failed ({exc}).") from exc

    # --- fast path: near-live quote --------------------------------------
    snap = _fast_info_snapshot(tk)
    price = snap["price"]
    currency = snap["currency"]
    if isinstance(price, float) and price > 0:
        return Quote(
            ticker=ticker,
            price=float(price),
            currency=str(currency) if currency else "N/A",
            as_of=now,
            fetched_at=now,
            is_intraday=True,
            previous_close=snap["previous_close"],  # type: ignore[arg-type]
            day_high=snap["day_high"],  # type: ignore[arg-type]
            day_low=snap["day_low"],  # type: ignore[arg-type]
            volume=snap["volume"],  # type: ignore[arg-type]
        )

    # --- fallback: latest daily close ------------------------------------
    try:
        hist = _history(ticker, _QUOTE_FALLBACK_PERIOD)
    except Exception as exc:
        raise DataFetchError(f"{ticker}: request failed ({exc}).") from exc

    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        raise DataFetchError(f"{ticker}: no price data returned - {_SUFFIX_HINT}.")

    valid = hist[hist["Close"].notna()]
    if valid.empty:
        raise DataFetchError(f"{ticker}: price history contained no valid closes.")

    close = valid["Close"]
    last = valid.iloc[-1]
    previous_close = _finite_float(close.iloc[-2]) if len(close) >= 2 else snap["previous_close"]
    return Quote(
        ticker=ticker,
        price=float(close.iloc[-1]),
        currency=str(currency) if currency else "N/A",
        as_of=pd.Timestamp(close.index[-1]).to_pydatetime(),
        fetched_at=now,
        is_intraday=False,
        previous_close=previous_close,  # type: ignore[arg-type]
        day_high=_finite_float(last.get("High")),
        day_low=_finite_float(last.get("Low")),
        volume=_finite_float(last.get("Volume")),
    )


def fetch_quotes(tickers: Iterable[str]) -> tuple[dict[str, Quote], dict[str, str]]:
    """Latest quotes for many tickers.

    Returns ``(quotes, errors)`` - a failed ticker lands in ``errors`` with a
    readable message and never aborts the others.
    """
    quotes: dict[str, Quote] = {}
    errors: dict[str, str] = {}
    for t in _dedupe(tickers):
        try:
            quotes[t] = fetch_quote(t)
        except DataFetchError as exc:
            errors[t] = str(exc)
        except Exception as exc:  # noqa: BLE001 - unexpected library errors
            errors[t] = f"Unexpected error: {exc}"
    return quotes, errors


# ---------------------------------------------------------------------------
# historical prices
# ---------------------------------------------------------------------------
def fetch_price_history(
    tickers: Iterable[str], period: str = DEFAULT_PERIOD
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Adjusted daily close history for many tickers.

    * ``auto_adjust=True`` -> the ``Close`` column is already adjusted for
      splits and dividends (the modern yfinance replacement for 'Adj Close').
    * Each ticker is fetched independently so one failure cannot poison the
      batch (and so we can report exactly which symbol failed and why).
    * Different exchanges trade on different calendars, so after aligning on
      calendar dates we forward-fill holiday gaps and then trim to the window
      where *every* ticker has data (required for fair correlations/betas).

    Returns ``(prices, errors)`` where ``prices`` has one column per ticker,
    a tz-naive ascending DatetimeIndex and no NaNs. An all-failure batch
    returns an empty frame, never an exception.
    """
    _validate_period(period)

    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    for t in _dedupe(tickers):
        try:
            hist = _history(t, period)
            close = hist.get("Close", pd.Series(dtype=float)).dropna()
            if close.empty:
                errors[t] = f"no historical prices returned - {_SUFFIX_HINT}"
                continue
            close = close.astype(float)
            close.index = _naive_dates(close.index)
            close = close[~close.index.duplicated(keep="last")]
            series[t] = close.rename(t)
        except Exception as exc:  # noqa: BLE001
            errors[t] = f"history download failed ({exc})"

    if not series:
        return pd.DataFrame(), errors

    prices = pd.concat(series.values(), axis=1).sort_index()

    # Trim to the window where EVERY ticker genuinely has data, at BOTH ends,
    # before forward-filling. Doing it the other way round (ffill then dropna)
    # only trims the leading edge: a ticker that stops trading early -- a
    # delisting, a suspension, a symbol change -- gets carried forward at a
    # stale price for the rest of the window, which reads as a flat, zero-
    # volatility asset and quietly biases every correlation, beta and
    # volatility estimate downward.
    last_valid = min(
        (s.last_valid_index() for s in series.values() if s.last_valid_index() is not None),
        default=None,
    )
    first_valid = max(
        (s.first_valid_index() for s in series.values() if s.first_valid_index() is not None),
        default=None,
    )
    if first_valid is not None and last_valid is not None and first_valid <= last_valid:
        prices = prices.loc[first_valid:last_valid]

    # Only NOW fill holiday gaps: inside the common window a missing bar really
    # is one exchange being shut, not an absent listing.
    prices = prices.ffill().dropna()
    prices.index.name = "Date"
    return prices, errors


def fetch_ohlcv(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Split/dividend-adjusted OHLCV bars for one ticker.

    Columns ``Open, High, Low, Close, Volume`` (those the provider supplies)
    on a tz-naive, ascending, duplicate-free calendar-date index. Rows with no
    close are dropped, since a bar without a close is unusable for every
    indicator downstream.

    Returns an EMPTY frame (never raises) on any provider or symbol failure;
    a ``ValueError`` is raised only for an unsupported ``period`` string.
    """
    _validate_period(period)
    symbol = _clean_ticker(ticker)
    if not symbol:
        return _empty_ohlcv()

    try:
        hist = _history(symbol, period)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OHLCV download failed for %s: %s", symbol, exc)
        return _empty_ohlcv()

    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return _empty_ohlcv()

    frame = hist.copy()  # never mutate the provider's frame
    try:
        frame.index = _naive_dates(frame.index)
    except Exception as exc:  # noqa: BLE001 - non-datetime index
        logger.warning("OHLCV index for %s is not datetime-like: %s", symbol, exc)
        return _empty_ohlcv()

    columns = [c for c in OHLCV_COLUMNS if c in frame.columns]
    frame = frame.loc[:, columns].astype("float64")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame[frame["Close"].notna()]
    frame.index.name = "Date"
    return frame


def fetch_benchmark_history(
    symbol: str = DEFAULT_BENCHMARK_SYMBOL, period: str = DEFAULT_PERIOD
) -> pd.Series:
    """Adjusted close series for a benchmark index (default: NIFTY 50).

    Raises ``DataFetchError`` if the benchmark cannot be fetched - the caller
    decides whether that is fatal (it isn't: the app just hides the benchmark
    content and keeps everything else working).
    """
    prices, errors = fetch_price_history([symbol], period)
    if prices.empty:
        reason = errors.get(_clean_ticker(symbol), "no data returned")
        raise DataFetchError(f"Benchmark {symbol}: {reason}")
    return prices.iloc[:, 0]


# ---------------------------------------------------------------------------
# reference data (profiles)
# ---------------------------------------------------------------------------
#: SecurityProfile text field -> candidate yfinance ``.info`` keys, in order.
_PROFILE_TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "name": ("longName", "displayName", "shortName"),
    "short_name": ("shortName", "symbol"),
    "sector": ("sector", "sectorDisp"),
    "industry": ("industry", "industryDisp"),
    "country": ("country",),
    "exchange": ("fullExchangeName", "exchange"),
    "currency": ("currency", "financialCurrency"),
    "quote_type": ("quoteType", "typeDisp"),
    "website": ("website",),
    "summary": ("longBusinessSummary", "description"),
}

#: SecurityProfile numeric field -> candidate yfinance ``.info`` keys.
_PROFILE_NUMERIC_KEYS: dict[str, tuple[str, ...]] = {
    "market_cap": ("marketCap", "totalAssets"),
    "trailing_pe": ("trailingPE",),
    "forward_pe": ("forwardPE",),
    "price_to_book": ("priceToBook",),
    "return_on_equity": ("returnOnEquity",),
    "debt_to_equity": ("debtToEquity",),
    "profit_margin": ("profitMargins",),
    "revenue_growth": ("revenueGrowth",),
    "dividend_yield": ("dividendYield", "yield", "trailingAnnualDividendYield"),
    "beta": ("beta", "beta3Year"),
    "fifty_two_week_high": ("fiftyTwoWeekHigh",),
    "fifty_two_week_low": ("fiftyTwoWeekLow",),
}

#: Above this, a "yield" field is read as a percentage rather than a decimal.
#: yfinance switched dividendYield from 0.012 to 1.2 for the same 1.2% yield,
#: so we normalise on magnitude: no listed equity yields more than 100%.
_YIELD_PERCENT_CUTOFF = 1.0


def _raw_info(ticker: str) -> dict[str, object]:
    """Fetch ``.info`` defensively; an empty dict on any failure."""
    try:
        tk = yf.Ticker(ticker)
    except Exception:  # noqa: BLE001
        return {}
    for accessor in ("info", "get_info"):
        try:
            attr = getattr(tk, accessor, None)
            info = attr() if callable(attr) else attr
        except Exception:  # noqa: BLE001 - yfinance raises on delisted/unknown
            continue
        if isinstance(info, dict) and info:
            return info
    return {}


def _first_text(info: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _text(info.get(key))
        if value is not None:
            return value
    return None


def _first_number(info: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _finite_float(info.get(key))
        if value is not None:
            return value
    return None


def fetch_profile(ticker: str) -> SecurityProfile:
    """Reference / fundamental data for one instrument.

    Maps yfinance ``.info`` onto :class:`~src.models.SecurityProfile`
    defensively: every field is looked up through a candidate key list,
    type-coerced, and mapped to ``None`` when missing, NaN or infinite. Yahoo
    coverage is patchy (ETFs report no sector, Indian small caps no ROE), so a
    missing field is honestly absent rather than zero.

    ``dividend_yield`` is normalised to a DECIMAL (0.012 = 1.2%): yfinance
    reports it as a percentage in recent versions, so any value above
    ``_YIELD_PERCENT_CUTOFF`` is divided by 100.

    NEVER raises: a provider failure yields a profile with only the ticker set.
    """
    symbol = _clean_ticker(ticker)
    if not symbol:
        return SecurityProfile(ticker="")

    info = _raw_info(symbol)
    if not info:
        return SecurityProfile(ticker=symbol)

    fields: dict[str, object] = {"ticker": symbol}
    for field_name, keys in _PROFILE_TEXT_KEYS.items():
        fields[field_name] = _first_text(info, keys)
    for field_name, keys in _PROFILE_NUMERIC_KEYS.items():
        fields[field_name] = _first_number(info, keys)

    currency = fields.get("currency")
    if isinstance(currency, str):
        fields["currency"] = currency.upper()
    quote_type = fields.get("quote_type")
    if isinstance(quote_type, str):
        fields["quote_type"] = quote_type.upper()

    dividend_yield = fields.get("dividend_yield")
    if isinstance(dividend_yield, float) and dividend_yield > _YIELD_PERCENT_CUTOFF:
        fields["dividend_yield"] = dividend_yield / 100.0

    try:
        return SecurityProfile(**fields)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - defensive against schema drift
        logger.warning("Profile mapping failed for %s: %s", symbol, exc)
        return SecurityProfile(ticker=symbol)


def fetch_profiles(tickers: Iterable[str]) -> dict[str, SecurityProfile]:
    """Profiles for many tickers, keyed by cleaned symbol.

    There is no error channel: :func:`fetch_profile` already degrades to a
    bare profile, and reference data is never load-bearing for a calculation.
    """
    return {t: fetch_profile(t) for t in _dedupe(tickers)}


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------
def fetch_dividends(ticker: str) -> pd.Series:
    """Cash dividends per share, indexed by tz-naive ex-date.

    Amounts are in the instrument's native currency and are NOT adjusted (the
    price history from :func:`fetch_price_history` already is, so never add
    these back to an adjusted return series - that double counts).

    Returns an EMPTY float series on any failure; never raises.
    """
    symbol = _clean_ticker(ticker)
    if not symbol:
        return _empty_dividends()

    try:
        raw = yf.Ticker(symbol).dividends
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dividend download failed for %s: %s", symbol, exc)
        return _empty_dividends()

    if raw is None or len(raw) == 0:
        return _empty_dividends()

    try:
        series = pd.Series(raw).astype("float64")  # astype copies: no mutation
        series.index = _naive_dates(series.index)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dividend series for %s is malformed: %s", symbol, exc)
        return _empty_dividends()

    series = series.dropna()
    series = series[series > 0]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "dividend"
    series.index.name = "Date"
    return series


# ---------------------------------------------------------------------------
# foreign exchange
# ---------------------------------------------------------------------------
def fx_pair_symbol(from_ccy: str, to_ccy: str) -> str:
    """Yahoo FX symbol for the ``from_ccy`` -> ``to_ccy`` conversion.

    Convention: ``"AAABBB=X"`` is the price of ONE unit of AAA in BBB, so
    ``fx_pair_symbol("USD", "INR") == "USDINR=X"``.

    Yahoo lists the majors the way the market quotes them - EURUSD=X, not
    USDEUR=X - so when one leg is USD and the other is in
    ``config.USD_QUOTED_PAIRS`` the symbol comes back in that convention
    regardless of the direction asked for. The caller inverts the rate when
    the symbol's base (its first three characters) is not ``from_ccy``;
    :func:`fetch_fx_rate` does exactly that.

    Returns "" when either code is missing or both are the same currency.
    """
    src = _clean_ccy(from_ccy)
    dst = _clean_ccy(to_ccy)
    if not src or not dst or src == dst:
        return ""
    if dst == "USD" and src in USD_QUOTED_PAIRS:
        return f"{src}USD=X"
    if src == "USD" and dst in USD_QUOTED_PAIRS:
        return f"{dst}USD=X"
    return f"{src}{dst}=X"


def fetch_fx_rate(from_ccy: str, to_ccy: str) -> float:
    """Price of ONE unit of ``from_ccy`` expressed in ``to_ccy``.

    ``value_in_to = value_in_from * fetch_fx_rate(from, to)``.

    Returns exactly ``1.0`` when the two codes are the same currency and
    ``float("nan")`` when either code is blank or the pair cannot be priced.
    Market-convention pairs (EURUSD=X and friends) are inverted here so the
    returned number always answers the direction that was asked.
    """
    src = _clean_ccy(from_ccy)
    dst = _clean_ccy(to_ccy)
    if not src or not dst:
        return float("nan")
    if src == dst:
        return 1.0

    symbol = fx_pair_symbol(src, dst)
    if not symbol:
        return float("nan")

    try:
        quote = fetch_quote(symbol)
    except DataFetchError as exc:
        logger.warning("FX rate %s->%s unavailable: %s", src, dst, exc)
        return float("nan")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FX rate %s->%s failed: %s", src, dst, exc)
        return float("nan")

    rate = _finite_float(quote.price)
    if rate is None or rate <= 0:
        return float("nan")
    inverted = symbol[:3] != src
    return 1.0 / rate if inverted else rate


def fetch_fx_rates(currencies: Iterable[str], base: str) -> tuple[dict[str, float], dict[str, str]]:
    """Conversion rates from each currency INTO ``base``.

    Returns ``({ccy: rate_to_base}, errors)``. The base currency is always
    present with a rate of exactly ``1.0``. A currency that cannot be priced
    is OMITTED from the rates dict (rather than carrying a NaN that would
    silently poison every downstream valuation) and explained in ``errors``,
    so the UI can warn that those positions are shown unconverted.
    """
    dst = _clean_ccy(base)
    if not dst:
        return {}, {"base": "Base currency is missing; no FX conversion possible."}

    rates: dict[str, float] = {dst: 1.0}
    errors: dict[str, str] = {}
    wanted = [c for c in dict.fromkeys(_clean_ccy(c) for c in currencies) if c]

    for ccy in wanted:
        if ccy == dst:
            continue
        rate = fetch_fx_rate(ccy, dst)
        if math.isfinite(rate) and rate > 0:
            rates[ccy] = rate
        else:
            errors[ccy] = (
                f"no FX rate available for {ccy} -> {dst}; {ccy} positions are left unconverted"
            )
    return rates, errors


# ---------------------------------------------------------------------------
# symbol search (online fallback for the local stock universe)
# ---------------------------------------------------------------------------
#: Yahoo quote-type values worth offering in a portfolio tool. Options,
#: futures and currencies are filtered out: none of them can be a holding in
#: this application's position model.
_SEARCHABLE_QUOTE_TYPES: frozenset[str] = frozenset({"EQUITY", "ETF", "MUTUALFUND", "INDEX"})


def search_symbols(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Ask the provider to resolve a free-text company name to symbols.

    This is the ONLINE half of stock search. :mod:`src.stock_universe` handles
    the offline half against bundled CSVs and is always tried first; this
    function exists only for the long tail the curated files do not cover
    (small caps, non-US/India listings, recent renames and new IPOs).

    Returns a list of plain dicts -- ``symbol``, ``name``, ``exchange``,
    ``quote_type``, ``source`` -- deliberately NOT
    :class:`~src.stock_universe.StockRecord`, because a provider hit carries
    no verified country or currency and must not be mistaken for a curated
    row. ``source`` is always ``"yahoo-search"`` so the UI can label where a
    result came from.

    NEVER raises and never returns a partial-but-plausible symbol: an empty
    list means "ask again or type the symbol yourself", which is honest, where
    a fabricated suffix would not be.
    """
    text = str(query or "").strip()
    if not text or limit <= 0:
        return []

    search_cls = getattr(yf, "Search", None)
    if search_cls is None:  # pragma: no cover - depends on yfinance version
        logger.info("Installed yfinance exposes no Search API; online lookup unavailable.")
        return []

    try:
        # raise_errors=False: a search miss or a rate limit must degrade to an
        # empty result list, not take down the page the user is typing into.
        response = search_cls(
            text,
            max_results=int(limit),
            news_count=0,
            lists_count=0,
            enable_fuzzy_query=True,
            raise_errors=False,
        )
        quotes = getattr(response, "quotes", None) or []
    except Exception as exc:  # noqa: BLE001 - any provider/transport failure
        logger.warning("Symbol search for %r failed: %s", text, exc)
        return []

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        symbol = _clean_ticker(quote.get("symbol"))
        if not symbol or symbol in seen:
            continue
        quote_type = (_text(quote.get("quoteType")) or "").upper()
        if quote_type and quote_type not in _SEARCHABLE_QUOTE_TYPES:
            continue
        seen.add(symbol)
        results.append(
            {
                "symbol": symbol,
                "name": (
                    _text(quote.get("longname"))
                    or _text(quote.get("shortname"))
                    or _text(quote.get("name"))
                    or symbol
                ),
                "exchange": _text(quote.get("exchDisp")) or _text(quote.get("exchange")) or "",
                "quote_type": quote_type or "EQUITY",
                "source": "yahoo-search",
            }
        )
        if len(results) >= limit:
            break
    return results


def lookup_isin(isin: str) -> dict[str, str] | None:
    """Resolve one ISIN to a tradeable symbol through the provider's search.

    The online half of ISIN resolution. :mod:`src.isin` answers offline from
    its seed and learned tables; this covers the long tail those cannot.

    Two guards, because a wrong ISIN mapping puts somebody else's company in
    the user's portfolio while everything on screen still looks right:

    * The ISIN is validated (shape and ISO 6166 check digit) BEFORE any
      request. Asking the provider about a typo can only produce a plausible
      wrong answer, so the typo is refused here instead.
    * The result's country must not contradict the ISIN's own prefix. An
      ``INE...`` identifier is an Indian security, so a hit that is not an
      NSE/BSE listing is rejected rather than returned. This is exactly the
      mis-pairing a check digit cannot catch.

    Returns the same dict shape as :func:`search_symbols` (with ``source``
    set to ``"yahoo-isin"``), or ``None`` when nothing trustworthy came back.
    NEVER raises, and never invents a symbol.
    """
    from src import isin as isin_mod

    code = isin_mod.normalise(isin)
    if not isin_mod.is_valid(code):
        logger.info("Refusing to look up %r: not a valid ISIN.", isin)
        return None

    results = search_symbols(code, limit=5)
    if not results:
        return None

    issued = isin_mod.country_name(code)
    for hit in results:
        symbol = hit.get("symbol", "")
        if issued == "India" and not symbol.endswith((".NS", ".BO")):
            logger.info(
                "Skipping %s for %s: the ISIN is Indian but the symbol is not an NSE/BSE listing.",
                symbol,
                code,
            )
            continue
        if issued == "United States" and "." in symbol:
            # A US ISIN naming a foreign listing is the same mis-pairing in
            # the other direction; US symbols carry no exchange suffix.
            continue
        return {**hit, "isin": code, "source": "yahoo-isin"}
    return None


def lookup_isins(isins: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve many ISINs. Returns ``({isin: symbol}, {isin: reason})``.

    One failure never aborts the others, matching every other batch helper in
    this module: a ten-ISIN file with one unlisted security should resolve the
    other nine and say what happened to the tenth.
    """
    found: dict[str, str] = {}
    errors: dict[str, str] = {}
    for raw in dict.fromkeys(str(i).strip().upper() for i in isins):
        if not raw:
            continue
        try:
            hit = lookup_isin(raw)
        except Exception as exc:  # noqa: BLE001 - defensive; lookup_isin swallows its own
            errors[raw] = f"lookup failed ({exc})"
            continue
        if hit is None:
            errors[raw] = "the provider returned no matching tradeable instrument"
        else:
            found[raw] = hit["symbol"]
    return found, errors


# ---------------------------------------------------------------------------
# option chains
# ---------------------------------------------------------------------------
#: Columns returned by :func:`fetch_option_chain`, in order.
OPTION_COLUMNS: tuple[str, ...] = (
    "contract",
    "option_type",
    "strike",
    "last_price",
    "bid",
    "ask",
    "implied_vol",
    "volume",
    "open_interest",
    "in_the_money",
)

#: Suffixes for which Yahoo Finance publishes no option chain at all. This is
#: not a guess: ``AAPL`` returns 21 expiries while ``RELIANCE.NS`` and
#: ``^NSEI`` return zero. For an India-first application that is the single
#: biggest data gap in the product, so it is named explicitly and reported as
#: "not available from this provider" rather than as "no options exist".
NO_CHAIN_SUFFIXES: tuple[str, ...] = (".NS", ".BO")


def option_chain_available(ticker: str) -> tuple[bool, str]:
    """Whether this provider can serve an option chain for ``ticker``.

    ``(available, reason)``. Checked BEFORE any request, because the honest
    answer for an NSE symbol is a specific explanation rather than an empty
    frame the user would read as "this stock has no options" — NIFTY and
    Reliance options are among the most heavily traded contracts in the world.
    """
    symbol = _clean_ticker(ticker)
    if not symbol:
        return False, "No ticker supplied."
    if symbol.startswith("^") and ("NSE" in symbol or "BSE" in symbol):
        return False, (
            f"Yahoo Finance publishes no option chain for {symbol}. NIFTY and SENSEX "
            "options are among the most heavily traded contracts anywhere, but they "
            "are not in this feed — enter the contract manually from your broker's "
            "screen, or connect an NSE or broker feed."
        )
    if symbol.endswith(NO_CHAIN_SUFFIXES):
        return False, (
            f"Yahoo Finance publishes no option chain for NSE/BSE listings, so none is "
            f"available for {symbol}. Indian equity options are liquid and real — the "
            "gap is in this data source, not in the market. Enter the strike, premium "
            "and expiry manually from your broker and every calculation below still "
            "works."
        )
    return True, ""


def fetch_option_expiries(ticker: str) -> tuple[tuple[str, ...], str]:
    """Available expiry dates for ``ticker``. ``(expiries, reason)``.

    An empty tuple always comes with a reason, so the caller can distinguish
    "this provider does not carry it" from "the request failed" from "this
    underlying genuinely has no listed options".
    """
    symbol = _clean_ticker(ticker)
    available, reason = option_chain_available(symbol)
    if not available:
        return (), reason

    try:
        expiries = yf.Ticker(symbol).options
    except Exception as exc:  # noqa: BLE001 - any provider/transport failure
        logger.warning("Option expiries for %s failed: %s", symbol, exc)
        return (), f"{symbol}: option expiries could not be fetched ({exc})."

    cleaned = tuple(str(e) for e in (expiries or ()) if str(e).strip())
    if not cleaned:
        return (), (
            f"{symbol}: the provider returned no expiry dates. Either this underlying "
            "has no listed options, or the feed is temporarily missing them."
        )
    return cleaned, ""


def fetch_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, str]:
    """Calls and puts for one expiry, in one normalised frame.

    ``(chain, reason)`` with the :data:`OPTION_COLUMNS` contract and an
    ``option_type`` column of ``"call"`` / ``"put"`` — one frame rather than
    two, because every consumer wants to filter by strike across both sides.

    Prices come through UNTOUCHED: no midpoint substitution, no filling a
    missing bid with the last trade. An option that has not traded today has a
    stale ``last_price`` and that is information the caller needs, not noise to
    smooth over. Rows with no usable strike are dropped.
    """
    symbol = _clean_ticker(ticker)
    available, reason = option_chain_available(symbol)
    if not available:
        return pd.DataFrame(columns=list(OPTION_COLUMNS)), reason
    if not str(expiry or "").strip():
        return pd.DataFrame(columns=list(OPTION_COLUMNS)), "No expiry date supplied."

    try:
        chain = yf.Ticker(symbol).option_chain(str(expiry))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Option chain for %s %s failed: %s", symbol, expiry, exc)
        return (
            pd.DataFrame(columns=list(OPTION_COLUMNS)),
            f"{symbol} {expiry}: chain could not be fetched ({exc}).",
        )

    frames: list[pd.DataFrame] = []
    for kind, attribute in (("call", "calls"), ("put", "puts")):
        side = getattr(chain, attribute, None)
        if side is None or not isinstance(side, pd.DataFrame) or side.empty:
            continue
        block = pd.DataFrame(
            {
                "contract": side.get("contractSymbol", pd.Series(dtype="object")).astype(str),
                "option_type": kind,
                "strike": pd.to_numeric(side.get("strike"), errors="coerce"),
                "last_price": pd.to_numeric(side.get("lastPrice"), errors="coerce"),
                "bid": pd.to_numeric(side.get("bid"), errors="coerce"),
                "ask": pd.to_numeric(side.get("ask"), errors="coerce"),
                "implied_vol": pd.to_numeric(side.get("impliedVolatility"), errors="coerce"),
                "volume": pd.to_numeric(side.get("volume"), errors="coerce"),
                "open_interest": pd.to_numeric(side.get("openInterest"), errors="coerce"),
                "in_the_money": side.get("inTheMoney", pd.Series(dtype="object")),
            }
        )
        frames.append(block)

    if not frames:
        return (
            pd.DataFrame(columns=list(OPTION_COLUMNS)),
            f"{symbol} {expiry}: the provider returned an empty chain.",
        )

    out = pd.concat(frames, ignore_index=True)
    out = out[out["strike"].notna() & (out["strike"] > 0)]
    if out.empty:
        return (
            pd.DataFrame(columns=list(OPTION_COLUMNS)),
            f"{symbol} {expiry}: no contract in the chain had a usable strike.",
        )
    out = out.sort_values(["option_type", "strike"]).reset_index(drop=True)
    return out[list(OPTION_COLUMNS)], ""


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def fetch_statements(ticker: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Annual balance sheet, income statement and cash flow for one ticker.

    ``({"balance_sheet": df, "income": df, "cash_flow": df}, notice)``. Frames
    are ``line label x period`` exactly as the provider returns them -- no
    renaming and no filling happens here, because normalising is
    :mod:`src.analytics.fundamentals`' job and this module stays a thin,
    monkeypatchable boundary.

    Never raises. A statement the provider does not carry comes back as an
    EMPTY frame and is named in ``notice``, so the caller can tell "this
    company files no cash-flow statement here" from "the request failed".

    Slow: three separate provider calls, each fetching a multi-year table.
    Cache it (:mod:`src.ui.data_access` applies a 24-hour TTL) and, for a
    whole-universe sweep, rate-limit it.
    """
    symbol = _clean_ticker(ticker)
    if not symbol:
        return _empty_statements(), "No ticker was given."

    try:
        handle = yf.Ticker(symbol)
    except Exception as exc:  # noqa: BLE001 - provider construction can fail
        logger.warning("Statement handle for %s failed: %s", symbol, exc)
        return _empty_statements(), f"{symbol}: the provider could not be reached ({exc})"

    out: dict[str, pd.DataFrame] = {}
    absent: list[str] = []
    for key, attribute in (
        ("balance_sheet", "balance_sheet"),
        ("income", "financials"),
        ("cash_flow", "cashflow"),
    ):
        try:
            frame = getattr(handle, attribute)
        except Exception as exc:  # noqa: BLE001 - one statement must not sink the rest
            logger.warning("%s %s failed: %s", symbol, attribute, exc)
            frame = None
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            out[key] = frame
        else:
            out[key] = pd.DataFrame()
            absent.append(key.replace("_", " "))

    if len(absent) == 3:
        return out, f"{symbol}: the provider published no financial statements."
    notice = f"{symbol}: no {', '.join(absent)} published." if absent else ""
    return out, notice


def _empty_statements() -> dict[str, pd.DataFrame]:
    return {"balance_sheet": pd.DataFrame(), "income": pd.DataFrame(), "cash_flow": pd.DataFrame()}


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Cheap "does this symbol price?" check for the ticker-entry UI.

    Returns ``(ok, message)``. ``ok`` is True when a positive price could be
    obtained; the message is either a one-line confirmation with the price and
    currency, or the provider's own reason for failing. Never raises.
    """
    symbol = _clean_ticker(ticker)
    if not symbol:
        return False, "Enter a ticker symbol."

    try:
        quote = fetch_quote(symbol)
    except DataFetchError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error ({exc})."

    price = _finite_float(quote.price)
    if price is None or price <= 0:
        return False, f"{symbol}: provider returned no usable price."
    return True, f"{symbol}: {price:,.2f} {quote.currency} - {quote.freshness_label}"
