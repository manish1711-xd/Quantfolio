"""Cached bridge between Streamlit and the market-data layer.

This is the ONLY module in the UI that talks to :mod:`src.data.fetcher`, and
every call goes through ``st.cache_data`` with a TTL chosen for how fast that
kind of data actually moves:

======================  =========  ==============================================
data                    TTL        why
======================  =========  ==============================================
quotes                  5 min      prices move; Yahoo is delayed ~15 min anyway
daily history           30 min     one more bar appears per day at most
FX rates                30 min     same cadence as history for our purposes
profiles / fundamentals 24 h       sector and market cap do not change intraday
======================  =========  ==============================================

Arguments are tuples rather than lists because Streamlit hashes them to build
the cache key, and a list is unhashable. That is also why ``period`` is passed
explicitly instead of being read from a config object inside the function: the
cache key must contain everything that changes the result.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    HISTORY_TTL_SECONDS,
    PROFILE_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    SEARCH_TTL_SECONDS,
)
from src.data import fetcher as dfetch
from src.models import Quote, SecurityProfile


# ---------------------------------------------------------------------------
# quotes
# ---------------------------------------------------------------------------
@st.cache_data(ttl=QUOTE_TTL_SECONDS, show_spinner=False)
def load_quotes(tickers: tuple[str, ...]) -> tuple[dict[str, Quote], dict[str, str]]:
    """Latest available prices; failures are returned, never raised."""
    return dfetch.fetch_quotes(list(tickers))


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_history(tickers: tuple[str, ...], period: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Adjusted daily closes aligned to the window where every ticker has data."""
    return dfetch.fetch_price_history(list(tickers), period)


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_benchmark(symbol: str, period: str) -> pd.Series:
    """Benchmark index level series. Raises ``DataFetchError`` if unavailable."""
    return dfetch.fetch_benchmark_history(symbol, period)


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """Full OHLCV bars for the single-security research page."""
    return dfetch.fetch_ohlcv(ticker, period)


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_long_history(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Maximum available history, used only by the stress-test page.

    Kept separate from :func:`load_history` on purpose: replaying the 2008
    crisis needs decades of data, but loading decades for every page would be
    slow and would silently change the risk window the user selected.
    """
    return dfetch.fetch_price_history(list(tickers), "max")


# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=PROFILE_TTL_SECONDS, show_spinner=False)
def load_profiles(tickers: tuple[str, ...]) -> dict[str, SecurityProfile]:
    """Sector, industry and fundamentals. Missing coverage yields a bare profile."""
    return dfetch.fetch_profiles(list(tickers))


@st.cache_data(ttl=PROFILE_TTL_SECONDS, show_spinner=False)
def load_statements(ticker: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Annual statements for one ticker, cached for a day.

    A 24-hour TTL because annual accounts change a few times a year at most,
    and three provider calls per ticker is far too slow to repeat on a rerun.
    """
    return dfetch.fetch_statements(ticker)


@st.cache_data(ttl=PROFILE_TTL_SECONDS, show_spinner=False)
def load_dividends(ticker: str) -> pd.Series:
    return dfetch.fetch_dividends(ticker)


# ---------------------------------------------------------------------------
# symbol search
# ---------------------------------------------------------------------------
@st.cache_data(ttl=SEARCH_TTL_SECONDS, show_spinner=False)
def search_online(query: str, limit: int) -> list[dict[str, str]]:
    """Provider symbol search, cached for an hour.

    Only ever called on an explicit button press: the local universe in
    :mod:`src.stock_universe` answers every keystroke offline, and firing a
    network request per character typed would be both slow and abusive of a
    free endpoint. A one-hour TTL is generous because the answer to "what is
    Reliance's ticker" does not change.
    """
    return dfetch.search_symbols(query, limit)


@st.cache_data(ttl=SEARCH_TTL_SECONDS, show_spinner=False)
def lookup_isins_online(isins: tuple[str, ...]) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve ISINs to symbols through the provider, cached for an hour.

    Only ever called on an explicit button press. An ISIN -> symbol pairing is
    effectively immutable, so this is cached as aggressively as symbol search.
    """
    return dfetch.lookup_isins(list(isins))


# ---------------------------------------------------------------------------
# option chains
# ---------------------------------------------------------------------------
@st.cache_data(ttl=QUOTE_TTL_SECONDS, show_spinner=False)
def load_option_expiries(ticker: str) -> tuple[tuple[str, ...], str]:
    """Expiry dates for one underlying, with the reason when there are none."""
    return dfetch.fetch_option_expiries(ticker)


@st.cache_data(ttl=QUOTE_TTL_SECONDS, show_spinner=False)
def load_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, str]:
    """Calls and puts for one expiry.

    On the QUOTE ttl, not the history one: option premia move with the
    underlying and with implied volatility, so a 30-minute-old chain would
    misprice every overlay figure derived from it.
    """
    return dfetch.fetch_option_chain(ticker, expiry)


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------
@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_fx_rates(
    currencies: tuple[str, ...], base: str
) -> tuple[dict[str, float], dict[str, str]]:
    """Spot rates converting each currency into ``base`` (base itself maps to 1.0)."""
    return dfetch.fetch_fx_rates(list(currencies), base)


# ---------------------------------------------------------------------------
# cache control
# ---------------------------------------------------------------------------
def clear_market_cache() -> None:
    """Drop every cached market response so the next run refetches."""
    for fn in (
        load_quotes,
        load_history,
        load_benchmark,
        load_ohlcv,
        load_long_history,
        load_profiles,
        load_dividends,
        load_fx_rates,
        search_online,
        lookup_isins_online,
        load_option_expiries,
        load_option_chain,
    ):
        try:
            fn.clear()
        except Exception:  # noqa: BLE001 - clearing is best-effort
            continue
