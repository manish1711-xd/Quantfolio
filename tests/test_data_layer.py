"""The data layer — ``src.data.fetcher`` and ``src.data.store``, entirely offline.

This is the only part of QuantFolio that talks to the outside world, so it is
the only part whose tests could plausibly need the wire. They do not. A fake
``yfinance.Ticker`` (``FakeMarket`` below) is monkeypatched in, which buys three
things a live test cannot: the failure modes that matter most (a raising
provider, a cold quote endpoint, a delisted symbol, a text price) are reachable
on demand; the timezone alignment across two exchanges is exercised on a
calendar we control; and the whole module runs in well under a second, in CI, on
a Sunday, without an API key.

Verified here
-------------
**fetcher — quotes.** ``fetch_quote`` prefers the near-live ``fast_info``
endpoint and stamps ``is_intraday=True``; when that endpoint is missing, cold or
reports a non-positive price it falls back to the last daily close and stamps
``is_intraday=False``. That flag is the whole basis of the UI's freshness label,
so both directions are pinned. A provider exception becomes a ``DataFetchError``
naming both the symbol and the underlying reason; ``fetch_quotes`` converts that
into an entry in ``errors`` so one bad ticker cannot cost you the other nine.

**fetcher — history.** The alignment contract: an Asia/Kolkata series and an
America/New_York series are normalised onto naive calendar dates, holiday gaps
are forward-filled, and the frame is trimmed to the window every ticker priced.
Left tz-aware those two indices share not one timestamp, so the "4 rows, not 8"
assertion is the real proof the normalisation happened. ``auto_adjust=True`` is
asserted on the call itself — a silent flip to ``False`` would replace every
split-adjusted close in the product with a raw one and nothing else would fail.

**fetcher — reference data and FX.** ``fetch_profile`` never raises: an
exploding ``.info`` yields a bare ticker-only profile rather than an exception
five layers below the UI. ``fx_pair_symbol`` follows the market's quoting
convention (``EURUSD=X``, never ``USDEUR=X``) for ``config.USD_QUOTED_PAIRS``,
and ``fetch_fx_rate`` inverts the rate when it does, so the number always
answers the direction that was asked. ``fetch_fx_rates`` always carries the base
currency at exactly 1.0 and OMITS what it could not price instead of shipping a
NaN into a valuation.

**store.** Round trip, upsert-by-name (replace, never duplicate), cascade
delete, rename, ``list_portfolios`` counts, the empty-frames answer for an
unknown name, schema creation on a fresh file, and two stores on one file. The
sharpest distinction pinned here is ``transactions=None`` (leave the ledger
alone) versus ``transactions=pd.DataFrame()`` (wipe it): saving an edited
holdings grid from the UI must not silently destroy the trade history.

One test is marked ``xfail`` — see ``test_the_trailing_edge_is_trimmed_...``,
which asserts the common-window contract the docstring promises on the edge
where the implementation forward-fills a stale price past it instead.

Deterministic, offline, no RNG, no sleeps. The single ``@pytest.mark.network``
test at the end is a live-contract smoke check and is deselected by CI.
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from src.analytics.lots import parse_transactions
from src.config import USD_QUOTED_PAIRS, VALID_PERIODS
from src.data import fetcher
from src.data.fetcher import (
    DEFAULT_BENCHMARK_SYMBOL,
    OHLCV_COLUMNS,
    DataFetchError,
    fetch_benchmark_history,
    fetch_dividends,
    fetch_fx_rate,
    fetch_fx_rates,
    fetch_ohlcv,
    fetch_price_history,
    fetch_profile,
    fetch_profiles,
    fetch_quote,
    fetch_quotes,
    fx_pair_symbol,
    search_symbols,
    validate_ticker,
)
from src.data.store import (
    ALERT_SETTINGS_COLUMNS,
    HOLDINGS_COLUMNS,
    SCHEDULE_COLUMNS,
    TRANSACTIONS_COLUMNS,
    WATCHLIST_COLUMNS,
    PortfolioStore,
)
from src.models import Transaction

#: Session open times, used to build tz-aware indices the way yfinance returns
#: them: local wall-clock time carrying the exchange's own timezone.
NSE_SESSION = "09:15"
NYSE_SESSION = "09:30"


# ---------------------------------------------------------------------------
# a fake yfinance — the only "market data source" this module ever touches
# ---------------------------------------------------------------------------
class ProviderDown(RuntimeError):
    """Stand-in for the assorted exceptions yfinance raises on a bad request.

    yfinance is not disciplined about its exception types (JSONDecodeError,
    KeyError, requests exceptions and its own classes all escape), which is
    exactly why the fetcher catches broadly. Using a bespoke type here proves
    the fetcher is not quietly relying on a particular one.
    """


@dataclass
class FakeSymbol:
    """What the fake provider knows about one symbol.

    Any field may be an ``Exception`` instance, in which case touching that
    part of the API raises it — that is how the failure paths are reached.
    """

    fast_info: Any = None
    history: Any = None
    info: Any = None
    get_info: Any = None
    dividends: Any = None


class FakeTicker:
    """Drop-in for ``yfinance.Ticker`` covering the surface the fetcher uses."""

    def __init__(self, market: FakeMarket, symbol: str) -> None:
        self._market = market
        self._symbol = symbol
        market.constructed.append(symbol)

    def _entry(self) -> FakeSymbol:
        try:
            return self._market.symbols[self._symbol]
        except KeyError:
            raise ProviderDown(f"{self._symbol}: no data found, symbol may be delisted") from None

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Exception):
            raise value
        return value

    @property
    def fast_info(self) -> Any:
        return self._unwrap(self._entry().fast_info)

    @property
    def info(self) -> Any:
        return self._unwrap(self._entry().info)

    @property
    def dividends(self) -> Any:
        return self._unwrap(self._entry().dividends)

    def __getattr__(self, name: str) -> Any:
        # ``get_info`` exists only when the fixture supplied one, so that the
        # fetcher's "try .info, then .get_info()" ladder can be tested both ways.
        if name == "get_info":
            entry = self._entry()
            if entry.get_info is None:
                raise AttributeError(name)
            return lambda: self._unwrap(entry.get_info)
        raise AttributeError(name)

    def history(self, period: str = "1mo", auto_adjust: bool = False) -> pd.DataFrame:
        self._market.history_calls.append((self._symbol, period, auto_adjust))
        frame = self._unwrap(self._entry().history)
        # Deliberately NOT copied: the fetcher promises never to mutate the
        # provider's frame, and returning the original is what lets us check it.
        return pd.DataFrame() if frame is None else frame


class FakeMarket:
    """Registry of fake symbols plus a log of every call the fetcher made."""

    def __init__(self) -> None:
        self.symbols: dict[str, FakeSymbol] = {}
        self.history_calls: list[tuple[str, str, bool]] = []
        self.constructed: list[str] = []

    def add(self, symbol: str, **fields: Any) -> FakeMarket:
        self.symbols[symbol] = FakeSymbol(**fields)
        return self

    def ticker(self, symbol: str) -> FakeTicker:
        return FakeTicker(self, str(symbol))


class AttrFastInfo:
    """``fast_info`` exposed as attributes (modern yfinance)."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class ExplodingFastInfo:
    """``fast_info`` whose lazy properties raise — a cold or throttled endpoint."""

    def __getattr__(self, name: str) -> Any:
        raise ProviderDown(f"quote endpoint unavailable ({name})")


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> FakeMarket:
    """Replace ``yfinance.Ticker`` for the duration of one test."""
    fake = FakeMarket()
    monkeypatch.setattr(fetcher.yf, "Ticker", fake.ticker)
    return fake


def exchange_index(dates: list[str], tz: str, session: str) -> pd.DatetimeIndex:
    """tz-aware session timestamps, the shape yfinance hands back."""
    return pd.DatetimeIndex([f"{day} {session}" for day in dates]).tz_localize(tz)


def close_frame(dates: list[str], closes: list[float], tz: str, session: str) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=exchange_index(dates, tz, session))


# ---------------------------------------------------------------------------
# fetch_quote — the fast (near-live) path
# ---------------------------------------------------------------------------
def test_fetch_quote_prefers_the_fast_quote_and_marks_it_intraday(market: FakeMarket) -> None:
    """The freshest endpoint wins, and the price carries ``is_intraday=True``.

    Numbers chosen so the derived fields are checkable by eye:
      day_change     = 2460 - 2400              = 60
      day_change_pct = (2460 / 2400 - 1) * 100  = 2.5%
    """
    market.add(
        "RELIANCE.NS",
        fast_info=AttrFastInfo(
            last_price=2460.0,
            currency="INR",
            previous_close=2400.0,
            day_high=2470.0,
            day_low=2395.0,
            last_volume=1_250_000.0,
        ),
    )

    quote = fetch_quote("  reliance.ns ")  # stripped and upper-cased on the way in

    assert quote.ticker == "RELIANCE.NS"
    assert quote.price == pytest.approx(2460.0, abs=1e-12)
    assert quote.is_intraday is True
    assert quote.currency == "INR"
    assert quote.previous_close == pytest.approx(2400.0, abs=1e-12)
    assert quote.day_change == pytest.approx(60.0, abs=1e-12)
    assert quote.day_change_pct == pytest.approx(2.5, abs=1e-12)
    assert quote.day_high == pytest.approx(2470.0, abs=1e-12)
    assert quote.day_low == pytest.approx(2395.0, abs=1e-12)
    assert quote.volume == pytest.approx(1_250_000.0, abs=1e-12)
    assert quote.freshness_label == "Latest traded price (may be delayed ~15 min)"


def test_the_fast_path_costs_no_history_request(market: FakeMarket) -> None:
    """Regression guard: the point of ``fast_info`` is that it is cheap.

    Falling through to history on every quote would multiply the request count
    by one per holding and get the app rate-limited.
    """
    market.add("AAA", fast_info=AttrFastInfo(last_price=10.0, currency="INR"))

    fetch_quote("AAA")

    assert market.history_calls == []


def test_a_mapping_style_fast_info_is_read_too(market: FakeMarket) -> None:
    """yfinance's ``FastInfo`` supports BOTH attribute and ``[]`` access.

    Which one it prefers has changed between releases, so the fetcher tries
    both; a dict here exercises the ``__getitem__`` branch.
    """
    market.add("AAPL", fast_info={"last_price": 195.5, "currency": "USD", "previous_close": 190.0})

    quote = fetch_quote("AAPL")

    assert quote.price == pytest.approx(195.5, abs=1e-12)
    assert quote.currency == "USD"
    assert quote.is_intraday is True
    # keys the mapping simply does not carry degrade to None, never to 0.0
    assert quote.day_high is None
    assert quote.volume is None


def test_the_provider_currency_string_is_carried_through_verbatim(market: FakeMarket) -> None:
    """The fetcher does not upper-case a quote currency (``fetch_profile`` does).

    Pinned rather than "fixed" here because the consumer already compensates:
    ``positions._fx_rate_for`` retries the lookup upper-cased, so a provider
    reporting "inr" still finds the {"INR": ...} rate. If that retry is ever
    removed, this test is the breadcrumb explaining why it existed.
    """
    market.add("AAA", fast_info=AttrFastInfo(last_price=10.0, currency="inr"))

    assert fetch_quote("AAA").currency == "inr"


# ---------------------------------------------------------------------------
# fetch_quote — the last-daily-close fallback
# ---------------------------------------------------------------------------
def test_a_missing_fast_info_falls_back_to_the_last_daily_close(market: FakeMarket) -> None:
    """No quote endpoint at all -> last close, and ``is_intraday`` must flip.

    Closes 101, 103, 107 on 12/13/14 March: price is the LAST (107) and
    previous_close is the one before it (103), so
    day_change_pct = (107 / 103 - 1) * 100 = 3.883495145631068...
    """
    index = exchange_index(["2024-03-12", "2024-03-13", "2024-03-14"], "Asia/Kolkata", NSE_SESSION)
    market.add(
        "TCS.NS",
        history=pd.DataFrame(
            {
                "Close": [101.0, 103.0, 107.0],
                "High": [102.0, 104.0, 108.0],
                "Low": [100.0, 102.0, 106.0],
                "Volume": [10.0, 20.0, 30.0],
            },
            index=index,
        ),
    )

    quote = fetch_quote("TCS.NS")

    assert quote.price == pytest.approx(107.0, abs=1e-12)
    assert quote.is_intraday is False
    assert quote.freshness_label == "Latest available closing price"
    assert quote.previous_close == pytest.approx(103.0, abs=1e-12)
    assert quote.day_change_pct == pytest.approx((107.0 / 103.0 - 1.0) * 100.0, rel=1e-12)
    assert quote.day_change_pct == pytest.approx(3.8834951456310, abs=1e-9)
    assert quote.day_high == pytest.approx(108.0, abs=1e-12)
    assert quote.day_low == pytest.approx(106.0, abs=1e-12)
    assert quote.volume == pytest.approx(30.0, abs=1e-12)
    # ``as_of`` is the session the close belongs to, NOT the time we asked.
    assert pd.Timestamp(quote.as_of).date() == pd.Timestamp("2024-03-14").date()
    assert quote.fetched_at >= quote.as_of.replace(tzinfo=quote.fetched_at.tzinfo)


@pytest.mark.parametrize(
    ("bad_price", "why"),
    [
        (0.0, "a zero price is not a price"),
        (-1.0, "a negative price is provider junk"),
        (float("nan"), "NaN means the endpoint had nothing"),
        (None, "the field was absent entirely"),
        ("n/a", "some versions return a string sentinel"),
    ],
)
def test_a_non_positive_fast_price_is_rejected_in_favour_of_the_close(
    market: FakeMarket, bad_price: object, why: str
) -> None:
    """Every non-positive/unusable fast price must route to the history fallback."""
    market.add(
        "AAA",
        fast_info=AttrFastInfo(last_price=bad_price, currency="INR"),
        history=close_frame(["2024-03-14"], [88.0], "Asia/Kolkata", NSE_SESSION),
    )

    quote = fetch_quote("AAA")

    assert quote.price == pytest.approx(88.0, abs=1e-12), why
    assert quote.is_intraday is False


def test_an_exploding_fast_info_falls_back_instead_of_raising(market: FakeMarket) -> None:
    """A throttled quote endpoint is a normal Tuesday, not an error."""
    market.add(
        "AAA",
        fast_info=ExplodingFastInfo(),
        history=close_frame(["2024-03-14"], [55.0], "Asia/Kolkata", NSE_SESSION),
    )

    quote = fetch_quote("AAA")

    assert quote.price == pytest.approx(55.0, abs=1e-12)
    assert quote.is_intraday is False


def test_the_fallback_asks_for_five_days_of_split_adjusted_history(market: FakeMarket) -> None:
    """Regression guard on the request itself.

    ``auto_adjust=True`` is what makes ``Close`` the split/dividend-adjusted
    series the whole analytics package assumes. Flipping it to False would
    silently swap in raw prices and every return series would be wrong on
    exactly the days that matter most, with nothing else failing.
    """
    market.add("AAA", history=close_frame(["2024-03-14"], [55.0], "Asia/Kolkata", NSE_SESSION))

    fetch_quote("AAA")

    assert market.history_calls == [("AAA", "5d", True)]


def test_the_fallback_keeps_a_currency_the_quote_endpoint_did_supply(market: FakeMarket) -> None:
    """Price unusable but currency known: keep the currency, take the close."""
    market.add(
        "AAA",
        fast_info=AttrFastInfo(last_price=0.0, currency="USD"),
        history=close_frame(["2024-03-14"], [12.0], "Asia/Kolkata", NSE_SESSION),
    )

    quote = fetch_quote("AAA")

    assert quote.currency == "USD"
    assert quote.is_intraday is False


def test_an_unknown_currency_is_reported_as_not_available_never_guessed(
    market: FakeMarket,
) -> None:
    """ "N/A" is honest; defaulting to INR would mis-convert every US holding."""
    market.add("AAA", history=close_frame(["2024-03-14"], [12.0], "Asia/Kolkata", NSE_SESSION))

    assert fetch_quote("AAA").currency == "N/A"


def test_a_single_row_of_history_leaves_the_previous_close_unknown(market: FakeMarket) -> None:
    """One close cannot produce a day change; NaN says so, 0.0 would lie."""
    market.add("AAA", history=close_frame(["2024-03-14"], [12.0], "Asia/Kolkata", NSE_SESSION))

    quote = fetch_quote("AAA")

    assert quote.previous_close is None
    assert math.isnan(quote.day_change_pct)


def test_rows_without_a_close_are_skipped_when_picking_the_last_price(
    market: FakeMarket,
) -> None:
    """A NaN-closed final bar (a halted session) must not become the price."""
    market.add(
        "AAA",
        history=pd.DataFrame(
            {"Close": [10.0, 11.0, np.nan]},
            index=exchange_index(
                ["2024-03-12", "2024-03-13", "2024-03-14"], "Asia/Kolkata", NSE_SESSION
            ),
        ),
    )

    quote = fetch_quote("AAA")

    assert quote.price == pytest.approx(11.0, abs=1e-12)
    assert quote.previous_close == pytest.approx(10.0, abs=1e-12)


# ---------------------------------------------------------------------------
# fetch_quote — failures surface as a readable DataFetchError
# ---------------------------------------------------------------------------
def test_a_raising_provider_surfaces_as_a_datafetcherror_naming_symbol_and_reason(
    market: FakeMarket,
) -> None:
    market.add("BOOM.NS", history=ProviderDown("connection reset by peer"))

    with pytest.raises(DataFetchError) as excinfo:
        fetch_quote("BOOM.NS")

    message = str(excinfo.value)
    assert "BOOM.NS" in message, "the user must be told WHICH symbol failed"
    assert "connection reset by peer" in message, "and WHY it failed"


def test_an_unknown_symbol_surfaces_as_a_datafetcherror(market: FakeMarket) -> None:
    with pytest.raises(DataFetchError, match="GARBAGE"):
        fetch_quote("garbage")


def test_an_empty_history_frame_is_reported_with_the_suffix_hint(market: FakeMarket) -> None:
    """The commonest real cause is a missing .NS/.BO suffix, so say so."""
    market.add("RELIANCE", history=pd.DataFrame())

    with pytest.raises(DataFetchError) as excinfo:
        fetch_quote("RELIANCE")

    message = str(excinfo.value)
    assert "no price data returned" in message
    assert ".NS" in message and ".BO" in message


def test_history_without_a_close_column_is_reported_not_key_errored(
    market: FakeMarket,
) -> None:
    market.add(
        "AAA",
        history=pd.DataFrame(
            {"Open": [1.0]},
            index=exchange_index(["2024-03-14"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    with pytest.raises(DataFetchError, match="no price data returned"):
        fetch_quote("AAA")


def test_an_all_nan_close_column_gets_its_own_message(market: FakeMarket) -> None:
    """Distinct from "no data": the provider answered, the answer was empty."""
    market.add(
        "AAA",
        history=pd.DataFrame(
            {"Close": [np.nan, np.nan]},
            index=exchange_index(["2024-03-13", "2024-03-14"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    with pytest.raises(DataFetchError, match="contained no valid closes"):
        fetch_quote("AAA")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_ticker_is_rejected_before_any_request(market: FakeMarket, blank: str) -> None:
    with pytest.raises(DataFetchError, match="Empty ticker symbol"):
        fetch_quote(blank)

    assert market.constructed == [], "a blank symbol must not cost a request"


# ---------------------------------------------------------------------------
# fetch_quotes — one bad ticker must never poison the batch
# ---------------------------------------------------------------------------
def test_fetch_quotes_isolates_a_failure_and_still_returns_the_others(
    market: FakeMarket,
) -> None:
    market.add("AAA", fast_info=AttrFastInfo(last_price=10.0, currency="INR"))
    market.add("BBB", history=ProviderDown("HTTP 429 too many requests"))
    market.add("CCC", fast_info=AttrFastInfo(last_price=30.0, currency="INR"))

    quotes, errors = fetch_quotes(["AAA", "BBB", "CCC"])

    assert set(quotes) == {"AAA", "CCC"}
    assert set(errors) == {"BBB"}
    assert "429" in errors["BBB"], "the provider's own reason is passed through"
    assert quotes["AAA"].price == pytest.approx(10.0, abs=1e-12)
    assert quotes["CCC"].price == pytest.approx(30.0, abs=1e-12)


def test_every_requested_ticker_is_accounted_for_exactly_once(market: FakeMarket) -> None:
    """Identity: quotes and errors partition the request. Nothing vanishes."""
    market.add("AAA", fast_info=AttrFastInfo(last_price=10.0, currency="INR"))
    market.add("BBB", history=ProviderDown("boom"))
    requested = {"AAA", "BBB", "CCC"}  # CCC is not in the registry at all

    quotes, errors = fetch_quotes(requested)

    assert set(quotes) | set(errors) == requested
    assert set(quotes).isdisjoint(errors)


def test_an_unexpected_library_error_is_caught_too(market: FakeMarket) -> None:
    """A text price escapes ``fetch_quote`` as a ValueError, not a DataFetchError.

    The isolation guarantee has to be total, otherwise a single malformed row
    from the provider takes down the whole portfolio load.
    """
    market.add(
        "WEIRD",
        history=pd.DataFrame(
            {"Close": ["not-a-number"]},
            index=exchange_index(["2024-03-14"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    quotes, errors = fetch_quotes(["WEIRD"])

    assert quotes == {}
    assert errors["WEIRD"].startswith("Unexpected error:")


def test_the_request_list_is_deduplicated_and_normalised(market: FakeMarket) -> None:
    """ "aaa", " AAA " and "AAA" are one symbol and must cost one request."""
    market.add("AAA", fast_info=AttrFastInfo(last_price=10.0, currency="INR"))

    quotes, errors = fetch_quotes([" aaa ", "AAA", "", "   ", "aaa"])

    assert set(quotes) == {"AAA"}
    assert errors == {}
    assert market.constructed == ["AAA"]


def test_fetch_quotes_of_nothing_is_two_empty_dicts(market: FakeMarket) -> None:
    assert fetch_quotes([]) == ({}, {})


# ---------------------------------------------------------------------------
# fetch_price_history — cross-exchange alignment, the module's hardest job
# ---------------------------------------------------------------------------
@pytest.fixture
def two_exchanges(market: FakeMarket) -> FakeMarket:
    """One NSE ticker and one NYSE ticker on deliberately mismatched calendars.

    INFY.NS  (Asia/Kolkata, 09:15 local)  02, 03, ____, 05, 08 Jan   <- 04 is a
                                                                        local
                                                                        holiday
    MSFT     (America/New_York, 09:30)    __, 03, 04,   05, 08 Jan   <- starts a
                                                                        day late
    """
    market.add(
        "INFY.NS",
        history=close_frame(
            ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"],
            [1500.0, 1510.0, 1530.0, 1540.0],
            "Asia/Kolkata",
            NSE_SESSION,
        ),
    )
    market.add(
        "MSFT",
        history=close_frame(
            ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            [370.0, 372.0, 374.0, 376.0],
            "America/New_York",
            NYSE_SESSION,
        ),
    )
    return market


def test_two_exchanges_align_onto_one_row_per_calendar_date(two_exchanges: FakeMarket) -> None:
    """The whole alignment contract in one hand-built frame.

    Left tz-aware the two indices share NO timestamp: 09:15+05:30 is 03:45Z and
    09:30-05:00 is 14:30Z. A naive concat would therefore produce eight rows and
    a frame where no correlation, beta or portfolio value is computable. Four
    rows is the proof that ``_naive_dates`` normalised both to calendar dates.

    Row by row:
      02 Jan  dropped   — MSFT had not started, so it is outside the common window
      03 Jan  1510 / 370
      04 Jan  1510 / 372 — NSE holiday, forward-filled from 03 Jan
      05 Jan  1530 / 374
      08 Jan  1540 / 376
    """
    prices, errors = fetch_price_history(["INFY.NS", "MSFT"], "1y")

    expected = pd.DataFrame(
        {
            "INFY.NS": [1510.0, 1510.0, 1530.0, 1540.0],
            "MSFT": [370.0, 372.0, 374.0, 376.0],
        },
        index=pd.DatetimeIndex(
            ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"], name="Date"
        ),
    )

    assert errors == {}
    assert len(prices) == 4, "eight rows would mean the timezones were never stripped"
    # check_freq=False: pandas infers a BusinessDay freq for these four dates,
    # which is an accident of the sample calendar and not part of the contract.
    pd.testing.assert_frame_equal(prices, expected, check_freq=False)


def test_the_aligned_index_is_naive_midnight_and_ascending(two_exchanges: FakeMarket) -> None:
    """Regression guard on the index convention every downstream module assumes."""
    prices, _errors = fetch_price_history(["INFY.NS", "MSFT"], "1y")

    assert prices.index.tz is None
    assert (prices.index == prices.index.normalize()).all(), "time-of-day must be stripped"
    assert prices.index.is_monotonic_increasing
    assert prices.index.name == "Date"
    assert not prices.index.has_duplicates


def test_a_local_holiday_is_forward_filled_not_read_as_a_missing_day(
    two_exchanges: FakeMarket,
) -> None:
    """4 Jan: the NSE was shut, so INFY holds its 3 Jan close of 1510.

    Dropping the row instead would throw away a live US session; leaving it NaN
    would propagate into every return series downstream.
    """
    prices, _errors = fetch_price_history(["INFY.NS", "MSFT"], "1y")

    assert prices.loc["2024-01-04", "INFY.NS"] == pytest.approx(1510.0, abs=1e-12)
    assert prices.loc["2024-01-04", "MSFT"] == pytest.approx(372.0, abs=1e-12)
    assert not prices.isna().to_numpy().any(), "the contract is a frame with no NaNs"


def test_the_leading_edge_is_trimmed_to_the_first_date_every_ticker_priced(
    two_exchanges: FakeMarket,
) -> None:
    """2 Jan is dropped: MSFT has no price, and there is nothing to fill from.

    Keeping it (at a back-filled or NaN price) would bias every correlation and
    beta computed over the window.
    """
    prices, _errors = fetch_price_history(["INFY.NS", "MSFT"], "1y")

    assert prices.index[0] == pd.Timestamp("2024-01-03")
    assert pd.Timestamp("2024-01-02") not in prices.index


# Regression guard for a fixed defect:
# fetch_price_history forward-fills BEFORE dropna, so a ticker that stops trading early is
# carried forward at a stale price and the frame extends past the window where every
# ticker actually has data — the leading edge is trimmed but the trailing edge is not
def test_the_trailing_edge_is_trimmed_to_the_last_date_every_ticker_priced(
    market: FakeMarket,
) -> None:
    """The other half of the documented "common window" contract.

    STOPPED.NS last prints on 08 Jan; RUNNING keeps going to 10 Jan. The
    docstring promises a trim "to the window where *every* ticker has data", so
    the frame should end on 08 Jan. It does not: ffill carries 1540.0 forward,
    manufacturing two zero-return days that flatter volatility and correlation.
    """
    market.add(
        "STOPPED.NS",
        history=close_frame(
            ["2024-01-05", "2024-01-08"],
            [1530.0, 1540.0],
            "Asia/Kolkata",
            NSE_SESSION,
        ),
    )
    market.add(
        "RUNNING",
        history=close_frame(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
            [370.0, 372.0, 374.0, 376.0],
            "America/New_York",
            NYSE_SESSION,
        ),
    )

    prices, _errors = fetch_price_history(["STOPPED.NS", "RUNNING"], "1y")

    assert prices.index[-1] == pd.Timestamp("2024-01-08")


def test_a_provider_frame_returned_out_of_order_comes_back_ascending(
    market: FakeMarket,
) -> None:
    """Unsorted input is a real yfinance failure mode on multi-request batches."""
    market.add(
        "AAA",
        history=close_frame(
            ["2024-01-08", "2024-01-03", "2024-01-05"],
            [104.0, 101.0, 103.0],
            "Asia/Kolkata",
            NSE_SESSION,
        ),
    )

    prices, _errors = fetch_price_history(["AAA"], "1y")

    assert list(prices.index) == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]
    assert list(prices["AAA"]) == pytest.approx([101.0, 103.0, 104.0], abs=1e-12)


def test_a_duplicated_calendar_date_keeps_the_last_observation(market: FakeMarket) -> None:
    """Two bars stamped the same day (a re-request straddling a session) collapse.

    Keeping the LAST is the right convention: it is the more recent print.
    """
    market.add(
        "AAA",
        history=close_frame(
            ["2024-01-03", "2024-01-03", "2024-01-05"],
            [101.0, 102.0, 103.0],
            "Asia/Kolkata",
            NSE_SESSION,
        ),
    )

    prices, _errors = fetch_price_history(["AAA"], "1y")

    assert len(prices) == 2
    assert prices.loc["2024-01-03", "AAA"] == pytest.approx(102.0, abs=1e-12)


def test_the_provider_frame_is_never_mutated(market: FakeMarket) -> None:
    """The fetcher works on copies; yfinance caches its frames between calls."""
    raw = close_frame(["2024-01-03", "2024-01-05"], [101.0, 103.0], "Asia/Kolkata", NSE_SESSION)
    pristine = raw.copy(deep=True)
    market.add("AAA", history=raw)

    fetch_price_history(["AAA"], "1y")
    fetch_ohlcv("AAA", "1y")

    pd.testing.assert_frame_equal(raw, pristine)


# ---------------------------------------------------------------------------
# fetch_price_history — periods and degradation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_period", ["7d", "1Y", "ytd", "", "1 year"])
def test_an_invalid_period_raises_valueerror_before_any_request(
    market: FakeMarket, bad_period: str
) -> None:
    """An unsupported window is a caller bug, not degenerate market data.

    Silently substituting "1y" would mislabel the result, and every annualised
    statistic computed from it would be quoted over the wrong horizon.
    """
    with pytest.raises(ValueError, match="period must be one of"):
        fetch_price_history(["AAA"], bad_period)

    assert market.constructed == []


@pytest.mark.parametrize("period", VALID_PERIODS)
def test_every_configured_period_is_accepted_and_passed_through(
    market: FakeMarket, period: str
) -> None:
    """The UI only ever offers ``config.VALID_PERIODS``; all of them must work."""
    market.add("AAA", history=close_frame(["2024-01-03"], [101.0], "Asia/Kolkata", NSE_SESSION))

    fetch_price_history(["AAA"], period)

    assert market.history_calls == [("AAA", period, True)]


def test_a_ticker_with_no_prices_is_reported_while_the_others_survive(
    market: FakeMarket,
) -> None:
    market.add("GOOD", history=close_frame(["2024-01-03"], [101.0], "Asia/Kolkata", NSE_SESSION))
    market.add(
        "BLANK",
        history=pd.DataFrame(
            {"Close": [np.nan]},
            index=exchange_index(["2024-01-03"], "Asia/Kolkata", NSE_SESSION),
        ),
    )
    market.add("DEAD", history=ProviderDown("delisted"))

    prices, errors = fetch_price_history(["GOOD", "BLANK", "DEAD"], "1y")

    assert list(prices.columns) == ["GOOD"]
    assert set(errors) == {"BLANK", "DEAD"}
    assert "no historical prices" in errors["BLANK"]
    assert "delisted" in errors["DEAD"]


def test_an_all_failure_batch_returns_an_empty_frame_not_an_exception(
    market: FakeMarket,
) -> None:
    """The app must keep working with whatever it did get — here, nothing."""
    market.add("AAA", history=ProviderDown("boom"))

    prices, errors = fetch_price_history(["AAA", "BBB"], "1y")

    assert prices.empty
    assert set(errors) == {"AAA", "BBB"}


def test_an_empty_ticker_list_returns_an_empty_frame_and_no_errors(
    market: FakeMarket,
) -> None:
    prices, errors = fetch_price_history([], "1y")

    assert prices.empty
    assert errors == {}


def test_the_default_period_is_one_year(market: FakeMarket) -> None:
    """Regression guard: the default window sizes every headline statistic."""
    market.add("AAA", history=close_frame(["2024-01-03"], [101.0], "Asia/Kolkata", NSE_SESSION))

    fetch_price_history(["AAA"])

    assert market.history_calls == [("AAA", "1y", True)]


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------
def test_fetch_ohlcv_returns_the_documented_columns_as_floats(market: FakeMarket) -> None:
    market.add(
        "AAA",
        history=pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [12.0, 13.0],
                "Low": [9.0, 10.5],
                "Close": [11.0, 12.5],
                "Volume": [1000, 2000],  # integers on the wire, floats on the way out
                "Dividends": [0.0, 0.0],  # provider extras are dropped
            },
            index=exchange_index(["2024-01-03", "2024-01-05"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    frame = fetch_ohlcv("aaa", "1y")

    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert (frame.dtypes == "float64").all()
    assert frame.index.tz is None
    assert frame.index.name == "Date"
    assert frame["Volume"].iloc[-1] == pytest.approx(2000.0, abs=1e-12)


def test_fetch_ohlcv_drops_bars_with_no_close(market: FakeMarket) -> None:
    """A bar without a close is unusable for every indicator downstream."""
    market.add(
        "AAA",
        history=pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [12.0, 13.0],
                "Low": [9.0, 10.5],
                "Close": [11.0, np.nan],
                "Volume": [1000.0, 2000.0],
            },
            index=exchange_index(["2024-01-03", "2024-01-05"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    frame = fetch_ohlcv("AAA", "1y")

    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2024-01-03")


@pytest.mark.parametrize(
    "symbol",
    ["MISSING", "  "],
)
def test_fetch_ohlcv_degrades_to_a_typed_empty_frame_and_never_raises(
    market: FakeMarket, symbol: str
) -> None:
    """Charts must render an empty state, not a traceback."""
    frame = fetch_ohlcv(symbol, "1y")

    assert frame.empty
    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert (frame.dtypes == "float64").all()


def test_fetch_ohlcv_rejects_an_invalid_period(market: FakeMarket) -> None:
    """The ONLY thing that raises here — everything else degrades to empty."""
    with pytest.raises(ValueError, match="period must be one of"):
        fetch_ohlcv("AAA", "7d")


def test_fetch_ohlcv_survives_a_non_datetime_index(market: FakeMarket) -> None:
    market.add(
        "AAA",
        history=pd.DataFrame({"Close": [1.0, 2.0]}, index=["not", "dates"]),
    )

    assert fetch_ohlcv("AAA", "1y").empty


def test_the_ohlcv_column_contract_is_the_conventional_bar_order() -> None:
    assert OHLCV_COLUMNS == ("Open", "High", "Low", "Close", "Volume")


# ---------------------------------------------------------------------------
# fetch_benchmark_history
# ---------------------------------------------------------------------------
def test_the_benchmark_series_is_the_single_price_column(market: FakeMarket) -> None:
    market.add(
        "^NSEI",
        history=close_frame(
            ["2024-01-03", "2024-01-05"], [21500.0, 21700.0], "Asia/Kolkata", NSE_SESSION
        ),
    )

    series = fetch_benchmark_history("^NSEI", "1y")

    assert isinstance(series, pd.Series)
    assert list(series) == pytest.approx([21500.0, 21700.0], abs=1e-12)
    assert series.index.tz is None


def test_a_missing_benchmark_raises_with_the_underlying_reason(market: FakeMarket) -> None:
    """Not fatal to the app — the caller hides the benchmark panel — but the
    reason has to survive to the message so the user can act on it."""
    market.add("^NSEI", history=ProviderDown("index feed unavailable"))

    with pytest.raises(DataFetchError) as excinfo:
        fetch_benchmark_history("^NSEI", "1y")

    assert "^NSEI" in str(excinfo.value)
    assert "index feed unavailable" in str(excinfo.value)


def test_the_default_benchmark_symbol_is_resolved_through_the_config_mapping() -> None:
    """Regression guard: the default must never drift from what the UI offers."""
    assert DEFAULT_BENCHMARK_SYMBOL == "^NSEI"


# ---------------------------------------------------------------------------
# fetch_profile — reference data that must NEVER raise
# ---------------------------------------------------------------------------
def test_a_full_info_payload_maps_onto_the_profile_fields(market: FakeMarket) -> None:
    market.add(
        "RELIANCE.NS",
        info={
            "longName": "Reliance Industries Limited",
            "shortName": "RELIANCE",
            "sector": "Energy",
            "industry": "Oil & Gas Refining",
            "country": "India",
            "fullExchangeName": "NSE",
            "currency": "inr",
            "quoteType": "equity",
            "marketCap": 1.85e13,
            "trailingPE": 24.5,
            "beta": 1.12,
        },
    )

    profile = fetch_profile("reliance.ns")

    assert profile.ticker == "RELIANCE.NS"
    assert profile.name == "Reliance Industries Limited"
    assert profile.short_name == "RELIANCE"
    assert profile.sector == "Energy"
    assert profile.currency == "INR", "currency is normalised to upper case here"
    assert profile.quote_type == "EQUITY"
    assert profile.market_cap == pytest.approx(1.85e13, rel=1e-12)
    assert profile.trailing_pe == pytest.approx(24.5, abs=1e-12)
    assert profile.beta == pytest.approx(1.12, abs=1e-12)


def test_fetch_profile_never_raises_when_info_explodes(market: FakeMarket) -> None:
    """The contract that lets the UI call this for every holding unguarded."""
    market.add("BOOM.NS", info=ProviderDown("Expecting value: line 1 column 1 (char 0)"))

    profile = fetch_profile("boom.ns")

    assert profile.ticker == "BOOM.NS"
    assert profile.name is None
    assert profile.sector is None
    assert profile.market_cap is None
    assert profile.display_name == "BOOM.NS", "falls back to the symbol itself"


@pytest.mark.parametrize("payload", [{}, None, "not-a-dict", []])
def test_a_useless_info_payload_yields_a_bare_profile(market: FakeMarket, payload: object) -> None:
    market.add("AAA", info=payload)

    profile = fetch_profile("AAA")

    assert profile.ticker == "AAA"
    assert profile.name is None


def test_the_get_info_callable_is_tried_when_the_info_property_fails(
    market: FakeMarket,
) -> None:
    """yfinance has shipped both spellings; the fetcher must survive either."""
    market.add(
        "AAA",
        info=ProviderDown("info property removed in this version"),
        get_info={"longName": "Alpha Industries", "sector": "Energy"},
    )

    profile = fetch_profile("AAA")

    assert profile.name == "Alpha Industries"
    assert profile.sector == "Energy"


def test_a_percentage_dividend_yield_is_normalised_to_a_decimal(market: FakeMarket) -> None:
    """yfinance changed 0.012 to 1.2 for the same 1.2% yield. 1.2 / 100 = 0.012.

    Without this the UI would print "120.0%" for a perfectly ordinary stock.
    """
    market.add("AAA", info={"dividendYield": 1.2})

    assert fetch_profile("AAA").dividend_yield == pytest.approx(0.012, rel=1e-12)


def test_a_decimal_dividend_yield_is_left_alone(market: FakeMarket) -> None:
    """0.012 is already 1.2% and must not be divided again into 0.00012."""
    market.add("AAA", info={"dividendYield": 0.012})

    assert fetch_profile("AAA").dividend_yield == pytest.approx(0.012, rel=1e-12)


def test_the_yield_cutoff_is_strict_so_exactly_one_survives(market: FakeMarket) -> None:
    """The cutoff is ``> 1.0``: 1.0 is read as a 100% yield and left as is.

    An edge case, but pinning it documents that the rule is a magnitude
    heuristic with a known boundary rather than an exact test.
    """
    market.add("AAA", info={"dividendYield": 1.0})

    assert fetch_profile("AAA").dividend_yield == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("junk", [float("nan"), float("inf"), None, "n/a", "unknown"])
def test_unusable_numeric_fields_become_none_never_zero(market: FakeMarket, junk: object) -> None:
    """A missing P/E is absent, not zero — zero would sort to the top of a screen."""
    market.add("AAA", info={"trailingPE": junk})

    assert fetch_profile("AAA").trailing_pe is None


def test_a_missing_sector_falls_back_to_the_quote_type_label(market: FakeMarket) -> None:
    """ETFs report no sector; "Unclassified" for a fund would be misleading."""
    market.add("NIFTYBEES.NS", info={"longName": "Nippon Nifty BeES", "quoteType": "ETF"})

    profile = fetch_profile("NIFTYBEES.NS")

    assert profile.sector is None
    assert profile.sector_label == "Fund / ETF"


def test_a_blank_ticker_yields_an_empty_profile_without_a_request(
    market: FakeMarket,
) -> None:
    profile = fetch_profile("   ")

    assert profile.ticker == ""
    assert market.constructed == []


def test_fetch_profiles_is_keyed_by_the_cleaned_symbol(market: FakeMarket) -> None:
    market.add("AAA", info={"longName": "Alpha"})
    market.add("BBB", info=ProviderDown("boom"))

    profiles = fetch_profiles([" aaa ", "BBB", "aaa"])

    assert set(profiles) == {"AAA", "BBB"}
    assert profiles["AAA"].name == "Alpha"
    assert profiles["BBB"].name is None, "a failure is a bare profile, not a missing key"


# ---------------------------------------------------------------------------
# fetch_dividends
# ---------------------------------------------------------------------------
def test_dividends_are_returned_on_a_naive_ascending_index(market: FakeMarket) -> None:
    market.add(
        "AAA",
        dividends=pd.Series(
            [9.0, 8.0],
            index=exchange_index(["2024-05-10", "2023-11-10"], "Asia/Kolkata", NSE_SESSION),
        ),
    )

    series = fetch_dividends("aaa")

    assert list(series) == pytest.approx([8.0, 9.0], abs=1e-12)
    assert list(series.index) == [pd.Timestamp("2023-11-10"), pd.Timestamp("2024-05-10")]
    assert series.name == "dividend"


def test_non_positive_dividend_rows_are_dropped(market: FakeMarket) -> None:
    """A zero-value "dividend" row is a provider artefact, not an event."""
    market.add(
        "AAA",
        dividends=pd.Series(
            [0.0, 5.0, np.nan],
            index=exchange_index(
                ["2024-01-03", "2024-02-03", "2024-03-03"], "Asia/Kolkata", NSE_SESSION
            ),
        ),
    )

    series = fetch_dividends("AAA")

    assert list(series) == pytest.approx([5.0], abs=1e-12)


@pytest.mark.parametrize("symbol", ["MISSING", "  "])
def test_fetch_dividends_degrades_to_an_empty_series(market: FakeMarket, symbol: str) -> None:
    series = fetch_dividends(symbol)

    assert series.empty
    assert series.dtype == "float64"


# ---------------------------------------------------------------------------
# FX — symbol convention, direction and the base-currency identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("src", "dst", "symbol", "why"),
    [
        ("USD", "INR", "USDINR=X", "the direct pair, quoted as asked"),
        ("INR", "USD", "INRUSD=X", "INR is not a USD-quoted major, so no flip"),
        ("EUR", "USD", "EURUSD=X", "the market convention, as asked"),
        ("USD", "EUR", "EURUSD=X", "flipped: Yahoo lists EURUSD=X, never USDEUR=X"),
        ("GBP", "USD", "GBPUSD=X", "cable, as asked"),
        ("USD", "GBP", "GBPUSD=X", "flipped for the same reason"),
        ("EUR", "INR", "EURINR=X", "neither leg is USD, so the flip does not apply"),
        ("usd", " inr ", "USDINR=X", "codes are cleaned before the lookup"),
    ],
)
def test_fx_pair_symbol_follows_the_market_quoting_convention(
    src: str, dst: str, symbol: str, why: str
) -> None:
    assert fx_pair_symbol(src, dst) == symbol, why


def test_every_usd_quoted_major_flips_in_both_directions() -> None:
    """Identity over ``config.USD_QUOTED_PAIRS``: both directions name one symbol.

    Adding a currency to that frozenset must not need a change here — which is
    the point of driving the test from the config rather than a hard-coded list.
    """
    for ccy in USD_QUOTED_PAIRS:
        assert fx_pair_symbol(ccy, "USD") == f"{ccy}USD=X"
        assert fx_pair_symbol("USD", ccy) == f"{ccy}USD=X"


@pytest.mark.parametrize(
    ("src", "dst"),
    [("INR", "INR"), ("inr", "INR"), ("", "INR"), ("INR", ""), ("nan", "INR")],
)
def test_fx_pair_symbol_is_blank_when_there_is_nothing_to_convert(src: str, dst: str) -> None:
    assert fx_pair_symbol(src, dst) == ""


def test_converting_a_currency_to_itself_is_exactly_one(market: FakeMarket) -> None:
    """Exactly 1.0, with no request: any other answer would drift the totals."""
    assert fetch_fx_rate("INR", "inr") == 1.0
    assert market.constructed == []


def test_a_direct_pair_is_used_as_quoted(market: FakeMarket) -> None:
    """USDINR=X at 83.25 means one USD buys 83.25 INR, so the rate IS 83.25."""
    market.add("USDINR=X", fast_info=AttrFastInfo(last_price=83.25, currency="INR"))

    assert fetch_fx_rate("USD", "INR") == pytest.approx(83.25, abs=1e-12)


def test_a_market_convention_pair_is_inverted_to_answer_the_question_asked(
    market: FakeMarket,
) -> None:
    """EURUSD=X at 1.25 is USD per EUR, so USD -> EUR is 1 / 1.25 = 0.8.

    Returning 1.25 for USD -> EUR would overstate a euro-denominated holding by
    56% (1.25 / 0.8), which is the sort of error a tearsheet never announces.
    """
    market.add("EURUSD=X", fast_info=AttrFastInfo(last_price=1.25, currency="USD"))

    assert fetch_fx_rate("USD", "EUR") == pytest.approx(0.8, abs=1e-12)
    assert fetch_fx_rate("EUR", "USD") == pytest.approx(1.25, abs=1e-12)


def test_the_two_directions_of_one_pair_are_reciprocal(market: FakeMarket) -> None:
    """Identity: rate(a->b) * rate(b->a) == 1 for any pair that prices at all."""
    market.add("EURUSD=X", fast_info=AttrFastInfo(last_price=1.0825, currency="USD"))

    there = fetch_fx_rate("EUR", "USD")
    back = fetch_fx_rate("USD", "EUR")

    assert there * back == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_an_unusable_fx_price_becomes_nan_never_a_silent_one(
    market: FakeMarket, bad_price: float
) -> None:
    """NaN forces the caller to decide; 1.0 would silently value USD as INR."""
    market.add(
        "USDINR=X",
        fast_info=AttrFastInfo(last_price=bad_price, currency="INR"),
        history=pd.DataFrame(),
    )

    assert math.isnan(fetch_fx_rate("USD", "INR"))


def test_an_unpriceable_pair_is_nan_rather_than_an_exception(market: FakeMarket) -> None:
    assert math.isnan(fetch_fx_rate("USD", "INR"))
    assert math.isnan(fetch_fx_rate("", "INR"))
    assert math.isnan(fetch_fx_rate("USD", ""))


def test_fetch_fx_rates_always_carries_the_base_at_exactly_one(market: FakeMarket) -> None:
    """The base-currency identity. Every valuation multiplies by this rate, so
    a base of 0.9999 would quietly shrink the whole portfolio."""
    market.add("USDINR=X", fast_info=AttrFastInfo(last_price=83.25, currency="INR"))

    rates, errors = fetch_fx_rates(["USD", "INR"], "INR")

    assert rates["INR"] == 1.0
    assert rates["USD"] == pytest.approx(83.25, abs=1e-12)
    assert errors == {}


def test_the_base_is_present_even_when_it_was_not_requested(market: FakeMarket) -> None:
    rates, _errors = fetch_fx_rates([], "inr")

    assert rates == {"INR": 1.0}


def test_a_currency_that_cannot_be_priced_is_omitted_and_explained(
    market: FakeMarket,
) -> None:
    """Omission, not NaN: a NaN rate would poison every total it touches, while
    a missing key lets ``positions`` fall back and flag the row as unconverted."""
    market.add("USDINR=X", fast_info=AttrFastInfo(last_price=83.25, currency="INR"))

    rates, errors = fetch_fx_rates(["USD", "JPY"], "INR")

    assert "JPY" not in rates
    assert "JPY" in errors
    assert "unconverted" in errors["JPY"], "the UI has to be able to warn about this"
    assert set(rates) == {"INR", "USD"}


def test_every_requested_currency_is_either_priced_or_explained(
    market: FakeMarket,
) -> None:
    """Identity: rates and errors together account for the whole request."""
    market.add("USDINR=X", fast_info=AttrFastInfo(last_price=83.25, currency="INR"))
    requested = ["USD", "JPY", "EUR", "INR"]

    rates, errors = fetch_fx_rates(requested, "INR")

    assert set(rates) | set(errors) == set(requested) | {"INR"}
    assert set(rates).isdisjoint(errors)


def test_a_blank_base_currency_is_refused_with_a_message(market: FakeMarket) -> None:
    rates, errors = fetch_fx_rates(["USD"], "   ")

    assert rates == {}
    assert "base" in errors


def test_duplicate_currencies_are_requested_once(market: FakeMarket) -> None:
    market.add("USDINR=X", fast_info=AttrFastInfo(last_price=83.25, currency="INR"))

    fetch_fx_rates([" usd ", "USD", "usd"], "INR")

    assert market.constructed == ["USDINR=X"]


# ---------------------------------------------------------------------------
# validate_ticker
# ---------------------------------------------------------------------------
def test_a_priceable_symbol_validates_with_its_price_and_freshness(
    market: FakeMarket,
) -> None:
    market.add("RELIANCE.NS", fast_info=AttrFastInfo(last_price=2460.0, currency="INR"))

    ok, message = validate_ticker("reliance.ns")

    assert ok is True
    assert "RELIANCE.NS" in message
    assert "2,460.00" in message, "the confirmation shows the price it actually found"
    assert "INR" in message


def test_an_unknown_symbol_fails_validation_with_the_provider_reason(
    market: FakeMarket,
) -> None:
    ok, message = validate_ticker("NOSUCH")

    assert ok is False
    assert "NOSUCH" in message


def test_a_blank_symbol_asks_the_user_for_one(market: FakeMarket) -> None:
    assert validate_ticker("  ") == (False, "Enter a ticker symbol.")


# ---------------------------------------------------------------------------
# store — fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A database inside a directory that does NOT exist yet.

    The store is expected to create its parent directory, which is what happens
    on a first run against ``<project>/data/quantfolio.db``.
    """
    return tmp_path / "nested" / "quantfolio-test.db"


@pytest.fixture
def store(db_path: Path):
    with PortfolioStore(db_path) as open_store:
        yield open_store


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic, strictly increasing ISO-8601 UTC clock.

    ``updated_at`` has one-second resolution, so two saves in the same real
    second would tie and make the ordering assertions flaky. This makes every
    write land on its own timestamp.
    """
    ticks = iter(f"2026-03-{day:02d}T09:00:00+00:00" for day in range(1, 29))
    monkeypatch.setattr("src.data.store._utc_now_iso", lambda: next(ticks))


def raw_count(path: Path, table: str) -> int:
    """Row count read through a SEPARATE connection — proves it hit the file."""
    with closing(sqlite3.connect(str(path))) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def table_names(path: Path) -> set[str]:
    with closing(sqlite3.connect(str(path))) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row[0]) for row in rows}


# ---------------------------------------------------------------------------
# store — schema creation
# ---------------------------------------------------------------------------
def test_the_schema_is_created_on_a_fresh_file(db_path: Path) -> None:
    """No migration step: constructing the store is the migration."""
    assert not db_path.exists()

    with PortfolioStore(db_path):
        pass

    assert db_path.exists()
    assert {"portfolios", "holdings", "transactions"}.issubset(table_names(db_path))


def test_a_fresh_database_lists_no_portfolios(store: PortfolioStore) -> None:
    assert store.list_portfolios() == []


def test_reopening_an_existing_file_does_not_recreate_the_schema(
    db_path: Path, holdings_frame: pd.DataFrame
) -> None:
    """``CREATE TABLE IF NOT EXISTS`` — the second open must not wipe the data."""
    with PortfolioStore(db_path) as first:
        first.save_portfolio("Core", holdings_frame)

    with PortfolioStore(db_path) as second:
        holdings, _txns = second.load_portfolio("Core")

    assert len(holdings) == 3


# ---------------------------------------------------------------------------
# store — round trip
# ---------------------------------------------------------------------------
def test_holdings_round_trip_unchanged(store: PortfolioStore, holdings_frame: pd.DataFrame) -> None:
    store.save_portfolio("Core Equity", holdings_frame, "INR")

    holdings, _txns = store.load_portfolio("Core Equity")

    pd.testing.assert_frame_equal(holdings, holdings_frame)


def test_transactions_round_trip_into_the_same_ledger_the_analytics_layer_parses(
    store: PortfolioStore,
    holdings_frame: pd.DataFrame,
    transactions_frame: pd.DataFrame,
    transactions: list[Transaction],
) -> None:
    """End-to-end identity across two modules, which is the point of the store.

    Saving the shared raw frame and loading it back must yield something
    ``lots.parse_transactions`` turns into exactly the shared ``transactions``
    fixture — the same ledger the FIFO tests in tests/test_lots.py rely on.
    Any lossy coercion in the store (a date reformatted, a fee dropped, an
    action mangled) breaks this and nothing else would notice.
    """
    store.save_portfolio("Core Equity", holdings_frame, "INR", transactions_frame)

    _holdings, stored = store.load_portfolio("Core Equity")
    parsed, problems = parse_transactions(stored)

    assert problems == []
    assert parsed == transactions


def test_the_store_records_a_broker_action_verbatim(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """ "B" is stored as "B", upper-cased but NOT translated to "BUY".

    Mapping broker vocabulary belongs to ``lots.parse_transactions``, which owns
    the alias table. A store that "helpfully" normalised would make the two
    layers disagree about what the user actually uploaded.
    """
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    _holdings, stored = store.load_portfolio("Core")

    # source actions were BUY / buy / B / SELL
    assert list(stored["action"]) == ["BUY", "BUY", "B", "SELL"]


def test_trade_dates_come_back_as_iso_strings_in_chronological_order(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    _holdings, stored = store.load_portfolio("Core")

    assert list(stored["trade_date"]) == [
        "2023-01-10",
        "2023-06-15",
        "2023-07-01",
        "2024-08-20",
    ]
    assert stored["trade_date"].map(type).eq(str).all()


def test_the_loaded_frames_match_the_declared_column_contract(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    holdings, txns = store.load_portfolio("Core")

    assert list(holdings.columns) == list(HOLDINGS_COLUMNS)
    assert list(txns.columns) == list(TRANSACTIONS_COLUMNS)
    assert holdings["quantity"].dtype == "float64"
    assert txns["fees"].dtype == "float64"


def test_optional_transaction_fields_default_to_the_portfolio_base_and_a_blank_note(
    store: PortfolioStore,
) -> None:
    """The frame carried no currency and no note; the store fills them in."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "trade_date": ["2024-01-10"],
            "action": ["BUY"],
            "quantity": [10.0],
            "price": [100.0],
        }
    )

    store.save_portfolio("Core", pd.DataFrame(), "usd", raw)
    _holdings, stored = store.load_portfolio("Core")

    assert stored.loc[0, "currency"] == "USD"
    assert stored.loc[0, "note"] == ""
    assert stored.loc[0, "fees"] == pytest.approx(0.0, abs=1e-12)


def test_column_labels_are_snake_cased_on_the_way_in(store: PortfolioStore) -> None:
    """A CSV exported with "Buy Price" must not be silently discarded."""
    raw = pd.DataFrame({"Ticker": [" aaa "], "Quantity": [10], "Buy Price": [100.0]})

    store.save_portfolio("Core", raw)
    holdings, _txns = store.load_portfolio("Core")

    assert list(holdings["ticker"]) == ["AAA"]
    assert holdings.loc[0, "buy_price"] == pytest.approx(100.0, abs=1e-12)


def test_a_short_position_is_stored_as_given(store: PortfolioStore) -> None:
    """Negative quantities are legitimate; judging them is not the store's job."""
    raw = pd.DataFrame({"ticker": ["AAA"], "quantity": [-5.0], "buy_price": [100.0]})

    store.save_portfolio("Core", raw)
    holdings, _txns = store.load_portfolio("Core")

    assert holdings.loc[0, "quantity"] == pytest.approx(-5.0, abs=1e-12)


@pytest.mark.parametrize(
    ("bad_row", "why"),
    [
        ({"ticker": [""], "quantity": [1.0], "buy_price": [1.0]}, "blank ticker"),
        ({"ticker": ["AAA"], "quantity": [np.nan], "buy_price": [1.0]}, "NaN quantity"),
        ({"ticker": ["AAA"], "quantity": [np.inf], "buy_price": [1.0]}, "infinite quantity"),
        ({"ticker": ["AAA"], "quantity": [1.0], "buy_price": ["junk"]}, "unparseable price"),
    ],
)
def test_a_row_that_cannot_be_represented_is_skipped_not_stored_as_garbage(
    store: PortfolioStore, bad_row: dict[str, list[Any]], why: str
) -> None:
    store.save_portfolio("Core", pd.DataFrame(bad_row))

    holdings, _txns = store.load_portfolio("Core")

    assert holdings.empty, why


def test_a_holdings_frame_missing_a_required_column_stores_nothing(
    store: PortfolioStore,
) -> None:
    """No partial write: an unusable frame leaves the portfolio empty, and the
    portfolio row itself is still created so the user's name is not lost."""
    store.save_portfolio("Core", pd.DataFrame({"ticker": ["AAA"], "quantity": [1.0]}))

    holdings, _txns = store.load_portfolio("Core")

    assert holdings.empty
    assert [p["name"] for p in store.list_portfolios()] == ["Core"]


def test_extra_columns_are_ignored_rather_than_rejected(store: PortfolioStore) -> None:
    """The editor grid carries sector/target columns the schema knows nothing of."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "quantity": [10.0],
            "buy_price": [100.0],
            "sector": ["Energy"],
            "target_weight_pct": [25.0],
        }
    )

    store.save_portfolio("Core", raw)
    holdings, _txns = store.load_portfolio("Core")

    assert list(holdings.columns) == list(HOLDINGS_COLUMNS)
    assert len(holdings) == 1


# ---------------------------------------------------------------------------
# store — upsert by name replaces, never duplicates
# ---------------------------------------------------------------------------
def test_saving_the_same_name_twice_updates_one_row_instead_of_adding_another(
    store: PortfolioStore, db_path: Path, holdings_frame: pd.DataFrame
) -> None:
    first_id = store.save_portfolio("Core", holdings_frame)
    second_id = store.save_portfolio(
        "Core", pd.DataFrame({"ticker": ["ZZZ"], "quantity": [1.0], "buy_price": [2.0]})
    )

    assert second_id == first_id, "the name is the key users address the portfolio by"
    assert raw_count(db_path, "portfolios") == 1
    assert len(store.list_portfolios()) == 1


def test_holdings_are_replaced_wholesale_not_merged(
    store: PortfolioStore, db_path: Path, holdings_frame: pd.DataFrame
) -> None:
    """Three holdings in, one holding in: the answer is one, not four.

    Merging would resurrect a position the user deleted in the editor.
    """
    store.save_portfolio("Core", holdings_frame)
    store.save_portfolio(
        "Core", pd.DataFrame({"ticker": ["ZZZ"], "quantity": [1.0], "buy_price": [2.0]})
    )

    holdings, _txns = store.load_portfolio("Core")

    assert list(holdings["ticker"]) == ["ZZZ"]
    assert raw_count(db_path, "holdings") == 1


def test_a_resave_refreshes_the_base_currency(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    store.save_portfolio("Core", holdings_frame, "INR")
    store.save_portfolio("Core", holdings_frame, "usd")

    assert store.list_portfolios()[0]["base_currency"] == "USD"


def test_created_at_survives_an_upsert_while_updated_at_moves(
    store: PortfolioStore, holdings_frame: pd.DataFrame, frozen_clock: None
) -> None:
    """A portfolio has one birthday. Only ``updated_at`` tracks edits."""
    store.save_portfolio("Core", holdings_frame)  # clock tick 1 -> 2026-03-01
    store.save_portfolio("Core", holdings_frame)  # clock tick 2 -> 2026-03-02

    row = store.list_portfolios()[0]

    assert row["created_at"] == "2026-03-01T09:00:00+00:00"
    assert row["updated_at"] == "2026-03-02T09:00:00+00:00"


def test_timestamps_are_timezone_aware_iso_8601(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """Naive local timestamps would sort wrong across a DST change."""
    store.save_portfolio("Core", holdings_frame)

    row = store.list_portfolios()[0]

    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None
    assert datetime.fromisoformat(row["updated_at"]).tzinfo is not None


# ---------------------------------------------------------------------------
# store — transactions: None preserves the ledger, a frame replaces it
# ---------------------------------------------------------------------------
def test_saving_holdings_alone_leaves_the_stored_ledger_untouched(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """The distinction that protects a user's trade history.

    Editing the holdings grid in the UI re-saves with ``transactions=None``. If
    that wiped the ledger, every realised-P&L and XIRR number would vanish on a
    keystroke, with no error and no undo.
    """
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    store.save_portfolio("Core", holdings_frame, "INR")  # transactions omitted

    _holdings, stored = store.load_portfolio("Core")
    assert len(stored) == 4


def test_saving_an_empty_transactions_frame_deliberately_clears_the_ledger(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """The other side of the same coin: an EMPTY frame is an instruction.

    ``None`` means "leave it alone"; ``pd.DataFrame()`` means "there are no
    trades". Collapsing the two would make "delete all my transactions"
    impossible to express.
    """
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    store.save_portfolio("Core", holdings_frame, "INR", pd.DataFrame())

    _holdings, stored = store.load_portfolio("Core")
    assert stored.empty


# ---------------------------------------------------------------------------
# store — listing
# ---------------------------------------------------------------------------
def test_list_portfolios_reports_the_holding_and_transaction_counts(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """conftest fixtures: 3 holdings (AAA/BBB/CCC), 4 transactions."""
    store.save_portfolio("Core Equity", holdings_frame, "INR", transactions_frame)

    rows = store.list_portfolios()

    assert len(rows) == 1
    assert rows[0]["name"] == "Core Equity"
    assert rows[0]["n_holdings"] == 3
    assert rows[0]["n_transactions"] == 4
    assert rows[0]["base_currency"] == "INR"
    assert isinstance(rows[0]["id"], int)


def test_the_holding_count_follows_a_resave_down_as_well_as_up(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    store.save_portfolio("Core", holdings_frame)
    assert store.list_portfolios()[0]["n_holdings"] == 3

    store.save_portfolio(
        "Core", pd.DataFrame({"ticker": ["ZZZ"], "quantity": [1.0], "buy_price": [2.0]})
    )

    assert store.list_portfolios()[0]["n_holdings"] == 1


def test_the_most_recently_updated_portfolio_is_listed_first(
    store: PortfolioStore, holdings_frame: pd.DataFrame, frozen_clock: None
) -> None:
    """The UI's picker shows what you were last working on at the top."""
    store.save_portfolio("Alpha", holdings_frame)  # 2026-03-01
    store.save_portfolio("Bravo", holdings_frame)  # 2026-03-02
    store.save_portfolio("Alpha", holdings_frame)  # 2026-03-03, back to the top

    assert [row["name"] for row in store.list_portfolios()] == ["Alpha", "Bravo"]


# ---------------------------------------------------------------------------
# store — rename
# ---------------------------------------------------------------------------
def test_rename_moves_the_portfolio_and_keeps_its_contents(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    portfolio_id = store.save_portfolio("Old Name", holdings_frame, "INR", transactions_frame)

    assert store.rename_portfolio("Old Name", "New Name") is True

    holdings, txns = store.load_portfolio("New Name")
    assert len(holdings) == 3
    assert len(txns) == 4
    assert store.load_portfolio("Old Name")[0].empty
    assert store.list_portfolios()[0]["id"] == portfolio_id, "the id is stable across a rename"


def test_renaming_an_unknown_portfolio_returns_false(store: PortfolioStore) -> None:
    assert store.rename_portfolio("nope", "whatever") is False


@pytest.mark.parametrize("blank", ["", "   ", "nan"])
def test_renaming_to_a_blank_name_is_refused(
    store: PortfolioStore, holdings_frame: pd.DataFrame, blank: str
) -> None:
    store.save_portfolio("Core", holdings_frame)

    assert store.rename_portfolio("Core", blank) is False
    assert [p["name"] for p in store.list_portfolios()] == ["Core"]


def test_renaming_onto_an_existing_name_is_refused_not_merged(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """``name`` is UNIQUE. Two portfolios must never be silently combined."""
    store.save_portfolio("Alpha", holdings_frame)
    store.save_portfolio("Bravo", holdings_frame)

    assert store.rename_portfolio("Bravo", "Alpha") is False
    assert {p["name"] for p in store.list_portfolios()} == {"Alpha", "Bravo"}


# ---------------------------------------------------------------------------
# store — delete cascades
# ---------------------------------------------------------------------------
def test_deleting_a_portfolio_removes_its_holdings_and_transactions(
    store: PortfolioStore,
    db_path: Path,
    holdings_frame: pd.DataFrame,
    transactions_frame: pd.DataFrame,
) -> None:
    """``ON DELETE CASCADE`` is only enforced with ``PRAGMA foreign_keys=ON``,
    which SQLite disables by default — so this is a real regression guard, not a
    test of SQLite. Orphaned child rows would be counted by nothing and deleted
    by nothing, and would resurface under a recycled portfolio id.
    """
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)
    assert raw_count(db_path, "holdings") == 3
    assert raw_count(db_path, "transactions") == 4

    assert store.delete_portfolio("Core") is True

    assert raw_count(db_path, "portfolios") == 0
    assert raw_count(db_path, "holdings") == 0
    assert raw_count(db_path, "transactions") == 0


def test_deleting_one_portfolio_leaves_the_others_intact(
    store: PortfolioStore, db_path: Path, holdings_frame: pd.DataFrame
) -> None:
    store.save_portfolio("Keep", holdings_frame)
    store.save_portfolio("Drop", holdings_frame)

    store.delete_portfolio("Drop")

    assert [p["name"] for p in store.list_portfolios()] == ["Keep"]
    assert raw_count(db_path, "holdings") == 3


def test_deleting_something_that_is_not_there_is_false_not_an_exception(
    store: PortfolioStore,
) -> None:
    assert store.delete_portfolio("nope") is False
    assert store.delete_portfolio(999) is False


def test_delete_is_idempotent(store: PortfolioStore, holdings_frame: pd.DataFrame) -> None:
    store.save_portfolio("Core", holdings_frame)

    assert store.delete_portfolio("Core") is True
    assert store.delete_portfolio("Core") is False


# ---------------------------------------------------------------------------
# store — addressing a portfolio by name or by id
# ---------------------------------------------------------------------------
def test_a_portfolio_can_be_addressed_by_its_integer_id(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    portfolio_id = store.save_portfolio("Core", holdings_frame)

    holdings, _txns = store.load_portfolio(portfolio_id)

    assert len(holdings) == 3


def test_a_portfolio_literally_called_twelve_is_found_by_name_first(
    store: PortfolioStore,
) -> None:
    """Name beats id for a string argument, so a numeric NAME stays reachable.

    Portfolio id 1 holds AAA; the portfolio *named* "12" holds ZZZ. Looking up
    "12" must return ZZZ even though 12 also parses as an id.
    """
    store.save_portfolio(
        "First", pd.DataFrame({"ticker": ["AAA"], "quantity": [1.0], "buy_price": [1.0]})
    )
    store.save_portfolio(
        "12", pd.DataFrame({"ticker": ["ZZZ"], "quantity": [1.0], "buy_price": [1.0]})
    )

    holdings, _txns = store.load_portfolio("12")

    assert list(holdings["ticker"]) == ["ZZZ"]


def test_a_digit_string_falls_back_to_an_id_lookup(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """No portfolio is NAMED "1", so "1" is tried as an id."""
    portfolio_id = store.save_portfolio("Core", holdings_frame)

    holdings, _txns = store.load_portfolio(str(portfolio_id))

    assert len(holdings) == 3


def test_a_boolean_is_not_treated_as_an_id(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """``True == 1`` in Python; resolving it as id 1 would load a portfolio the
    caller never asked for."""
    store.save_portfolio("Core", holdings_frame)

    holdings, _txns = store.load_portfolio(True)

    assert holdings.empty


# ---------------------------------------------------------------------------
# store — the missing-portfolio contract
# ---------------------------------------------------------------------------
def test_loading_an_unknown_name_yields_two_typed_empty_frames(
    store: PortfolioStore,
) -> None:
    """A missing save is a normal UI state (a first run), not an error."""
    holdings, txns = store.load_portfolio("never saved")

    assert holdings.empty and txns.empty
    assert list(holdings.columns) == list(HOLDINGS_COLUMNS)
    assert list(txns.columns) == list(TRANSACTIONS_COLUMNS)
    assert holdings["quantity"].dtype == "float64"
    assert txns["price"].dtype == "float64"


def test_the_empty_frames_are_shaped_like_the_populated_ones(
    store: PortfolioStore, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """Identity: the empty answer and the real answer have the same schema, so
    downstream code needs no special case for "nothing saved yet"."""
    store.save_portfolio("Core", holdings_frame, "INR", transactions_frame)
    full_holdings, full_txns = store.load_portfolio("Core")
    empty_holdings, empty_txns = store.load_portfolio("nope")

    assert list(empty_holdings.columns) == list(full_holdings.columns)
    assert list(empty_txns.columns) == list(full_txns.columns)
    assert empty_holdings.dtypes.to_dict() == full_holdings.dtypes.to_dict()


@pytest.mark.parametrize("blank", ["", "   ", "nan"])
def test_a_blank_portfolio_name_cannot_be_saved(
    store: PortfolioStore, holdings_frame: pd.DataFrame, blank: str
) -> None:
    """The one caller bug the store refuses to paper over: without a name there
    is no way for the user to ever address the portfolio again."""
    with pytest.raises(ValueError, match="must not be blank"):
        store.save_portfolio(blank, holdings_frame)


def test_a_portfolio_name_is_trimmed_before_it_becomes_the_key(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """ "Core" and " Core " are the same portfolio, not two."""
    first = store.save_portfolio("  Core  ", holdings_frame)
    second = store.save_portfolio("Core", holdings_frame)

    assert first == second
    assert [p["name"] for p in store.list_portfolios()] == ["Core"]


# ---------------------------------------------------------------------------
# store — two stores on one file
# ---------------------------------------------------------------------------
def test_two_stores_on_the_same_file_see_each_others_committed_writes(
    db_path: Path, holdings_frame: pd.DataFrame
) -> None:
    """Streamlit re-runs the script on whichever thread it likes and may hold
    more than one store open at once. Each write is committed, so neither
    connection can be looking at a stale or half-written portfolio."""
    with PortfolioStore(db_path) as first, PortfolioStore(db_path) as second:
        first.save_portfolio("From First", holdings_frame)
        assert [p["name"] for p in second.list_portfolios()] == ["From First"]

        second.save_portfolio("From Second", holdings_frame)
        assert {p["name"] for p in first.list_portfolios()} == {"From First", "From Second"}

        second.delete_portfolio("From First")
        assert first.load_portfolio("From First")[0].empty
        assert len(first.list_portfolios()) == 1


def test_a_write_through_one_store_survives_closing_both(
    db_path: Path, holdings_frame: pd.DataFrame, transactions_frame: pd.DataFrame
) -> None:
    """Durability: nothing lives only in a connection's memory."""
    with PortfolioStore(db_path) as writer:
        writer.save_portfolio("Core", holdings_frame, "INR", transactions_frame)

    with PortfolioStore(db_path) as reader:
        holdings, txns = reader.load_portfolio("Core")

    assert len(holdings) == 3
    assert len(txns) == 4


def test_an_in_memory_store_never_touches_the_filesystem(tmp_path: Path) -> None:
    """The ":memory:" escape hatch must not create a file called ":memory:"."""
    with PortfolioStore(":memory:") as memory_store:
        memory_store.save_portfolio(
            "Core", pd.DataFrame({"ticker": ["AAA"], "quantity": [1.0], "buy_price": [2.0]})
        )
        assert len(memory_store.list_portfolios()) == 1

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# store — lifecycle
# ---------------------------------------------------------------------------
def test_close_is_safe_to_call_more_than_once(db_path: Path) -> None:
    open_store = PortfolioStore(db_path)

    open_store.close()
    open_store.close()  # must not raise on the second call


def test_using_a_closed_store_says_so_plainly(db_path: Path) -> None:
    open_store = PortfolioStore(db_path)
    open_store.close()

    with pytest.raises(RuntimeError, match="closed"):
        open_store.list_portfolios()


def test_the_context_manager_closes_the_connection(db_path: Path) -> None:
    with PortfolioStore(db_path) as open_store:
        pass

    with pytest.raises(RuntimeError, match="closed"):
        open_store.list_portfolios()


def test_the_declared_column_contracts_are_self_consistent() -> None:
    """Regression guard: the constants other modules import must not drift."""
    assert HOLDINGS_COLUMNS == ("ticker", "quantity", "buy_price")
    # The first five transaction columns are exactly lots.REQUIRED_COLUMNS, in
    # order, which is what lets a stored ledger be parsed without a rename step.
    assert TRANSACTIONS_COLUMNS[:5] == ("ticker", "trade_date", "action", "quantity", "price")
    assert {"ticker", "quantity"}.issubset(TRANSACTIONS_COLUMNS)


# ---------------------------------------------------------------------------
# fetcher — symbol search (the online half of stock search)
# ---------------------------------------------------------------------------
class FakeSearch:
    """Stand-in for ``yfinance.Search``.

    Records the kwargs it was constructed with, so the tests can assert the
    fetcher asked for what it claims to ask for -- ``raise_errors=False`` in
    particular, since a rate-limited search must degrade to an empty list
    rather than take down the page the user is typing into.
    """

    calls: ClassVar[list[dict]] = []
    payload: ClassVar[list] = []
    explode: ClassVar[bool] = False

    def __init__(self, query, **kwargs):
        type(self).calls.append({"query": query, **kwargs})
        if type(self).explode:
            raise RuntimeError("provider refused the search")
        self.quotes = list(type(self).payload)


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch):
    """Install ``FakeSearch`` as ``yfinance.Search`` and reset its state."""
    import yfinance

    FakeSearch.calls = []
    FakeSearch.payload = []
    FakeSearch.explode = False
    monkeypatch.setattr(yfinance, "Search", FakeSearch)
    return FakeSearch


def test_search_symbols_maps_provider_hits_onto_plain_dicts(fake_search) -> None:
    fake_search.payload = [
        {
            "symbol": "RELIANCE.NS",
            "longname": "Reliance Industries Limited",
            "exchDisp": "NSE",
            "quoteType": "EQUITY",
        }
    ]

    results = search_symbols("reliance")

    assert results == [
        {
            "symbol": "RELIANCE.NS",
            "name": "Reliance Industries Limited",
            "exchange": "NSE",
            "quote_type": "EQUITY",
            "source": "yahoo-search",
        }
    ]


def test_a_provider_hit_is_always_labelled_with_its_source(fake_search) -> None:
    """The UI must be able to tell a provider guess from a curated row."""
    fake_search.payload = [{"symbol": "X", "shortname": "X Corp"}]

    assert search_symbols("x")[0]["source"] == "yahoo-search"


def test_the_shortname_is_used_when_there_is_no_longname(fake_search) -> None:
    fake_search.payload = [{"symbol": "X", "shortname": "X Corp"}]

    assert search_symbols("x")[0]["name"] == "X Corp"


def test_a_nameless_hit_falls_back_to_its_symbol_never_to_blank(fake_search) -> None:
    fake_search.payload = [{"symbol": "X"}]

    assert search_symbols("x")[0]["name"] == "X"


def test_options_and_futures_are_filtered_out(fake_search) -> None:
    """None of them can be a holding in this application's position model."""
    fake_search.payload = [
        {"symbol": "AAPL", "shortname": "Apple", "quoteType": "EQUITY"},
        {"symbol": "AAPL240119C", "shortname": "Apple call", "quoteType": "OPTION"},
        {"symbol": "ES=F", "shortname": "E-mini", "quoteType": "FUTURE"},
        {"symbol": "USDINR=X", "shortname": "USD/INR", "quoteType": "CURRENCY"},
    ]

    assert [r["symbol"] for r in search_symbols("apple")] == ["AAPL"]


@pytest.mark.parametrize("kind", ["EQUITY", "ETF", "MUTUALFUND", "INDEX"])
def test_every_holdable_instrument_type_survives_the_filter(fake_search, kind: str) -> None:
    fake_search.payload = [{"symbol": "X", "shortname": "X", "quoteType": kind}]

    assert len(search_symbols("x")) == 1


def test_a_hit_with_no_quote_type_is_kept_as_an_equity(fake_search) -> None:
    """Absent metadata is not a reason to hide a symbol the user asked for."""
    fake_search.payload = [{"symbol": "X", "shortname": "X Corp"}]

    assert search_symbols("x")[0]["quote_type"] == "EQUITY"


def test_duplicate_symbols_are_returned_once(fake_search) -> None:
    fake_search.payload = [
        {"symbol": "AAPL", "shortname": "Apple", "quoteType": "EQUITY"},
        {"symbol": "aapl", "shortname": "Apple again", "quoteType": "EQUITY"},
    ]

    assert [r["symbol"] for r in search_symbols("apple")] == ["AAPL"]


def test_the_search_limit_is_respected(fake_search) -> None:
    fake_search.payload = [
        {"symbol": f"T{i}", "shortname": f"T{i}", "quoteType": "EQUITY"} for i in range(20)
    ]

    assert len(search_symbols("t", limit=3)) == 3


def test_the_search_is_asked_not_to_raise(fake_search) -> None:
    """A throttled provider must yield an empty list, not an exception."""
    search_symbols("reliance")

    assert fake_search.calls[0]["raise_errors"] is False


def test_no_news_or_lists_are_requested(fake_search) -> None:
    """Asking for news on a ticker-lookup keystroke is wasted bandwidth."""
    search_symbols("reliance")

    assert fake_search.calls[0]["news_count"] == 0
    assert fake_search.calls[0]["lists_count"] == 0


def test_an_exploding_search_provider_yields_an_empty_list(fake_search) -> None:
    fake_search.explode = True

    assert search_symbols("reliance") == []


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_query_never_reaches_the_provider(fake_search, blank) -> None:
    assert search_symbols(blank) == []
    assert fake_search.calls == []


def test_a_non_positive_search_limit_never_reaches_the_provider(fake_search) -> None:
    assert search_symbols("reliance", limit=0) == []
    assert fake_search.calls == []


def test_a_malformed_provider_row_is_skipped_not_fatal(fake_search) -> None:
    fake_search.payload = [
        "not a dict",
        {"symbol": "", "shortname": "blank symbol"},
        {"symbol": "AAPL", "shortname": "Apple", "quoteType": "EQUITY"},
    ]

    assert [r["symbol"] for r in search_symbols("apple")] == ["AAPL"]


def test_a_yfinance_without_a_search_api_degrades_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older yfinance builds expose no Search class; that is not a crash."""
    import yfinance

    monkeypatch.delattr(yfinance, "Search", raising=False)

    assert search_symbols("reliance") == []


# ---------------------------------------------------------------------------
# store — watchlist
# ---------------------------------------------------------------------------
def watch_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(WATCHLIST_COLUMNS))


def test_a_fresh_store_has_an_empty_but_typed_watchlist(store: PortfolioStore) -> None:
    frame = store.load_watchlist()

    assert frame.empty
    assert tuple(frame.columns) == WATCHLIST_COLUMNS
    assert frame["target_price"].dtype == "float64"


def test_the_watchlist_round_trips(store: PortfolioStore) -> None:
    stored = store.save_watchlist(
        watch_frame(
            [
                {
                    "ticker": "RELIANCE.NS",
                    "note": "core candidate",
                    "target_price": 1500.0,
                    "added_at": "2025-01-10",
                },
                {"ticker": "TCS.NS", "note": "", "target_price": None, "added_at": "2025-02-20"},
            ]
        )
    )

    assert stored == 2
    loaded = store.load_watchlist()
    assert list(loaded["ticker"]) == ["RELIANCE.NS", "TCS.NS"]
    assert float(loaded["target_price"].iloc[0]) == 1500.0
    assert loaded["note"].iloc[0] == "core candidate"
    assert loaded["added_at"].iloc[0] == "2025-01-10"


def test_the_watchlist_round_trip_preserves_insertion_order(store: PortfolioStore) -> None:
    """Save is a wholesale replace, so a differently-ordered read would flip
    the list on every save/load cycle."""
    tickers = ["C.NS", "A.NS", "B.NS"]
    store.save_watchlist(watch_frame([{"ticker": t} for t in tickers]))

    assert list(store.load_watchlist()["ticker"]) == tickers


def test_an_absent_target_reads_back_as_nan_not_zero(store: PortfolioStore) -> None:
    """0.0 would render as a 100% fall to zero on the watchlist table."""
    store.save_watchlist(watch_frame([{"ticker": "TCS.NS", "target_price": None}]))

    assert math.isnan(float(store.load_watchlist()["target_price"].iloc[0]))


def test_a_zero_or_negative_target_is_stored_as_no_target(store: PortfolioStore) -> None:
    store.save_watchlist(
        watch_frame(
            [{"ticker": "A.NS", "target_price": 0.0}, {"ticker": "B.NS", "target_price": -5.0}]
        )
    )

    assert store.load_watchlist()["target_price"].isna().all()


def test_saving_replaces_the_whole_watchlist(store: PortfolioStore) -> None:
    store.save_watchlist(watch_frame([{"ticker": "A.NS"}, {"ticker": "B.NS"}]))
    store.save_watchlist(watch_frame([{"ticker": "C.NS"}]))

    assert list(store.load_watchlist()["ticker"]) == ["C.NS"]


def test_saving_an_empty_watchlist_frame_clears_it(store: PortfolioStore) -> None:
    store.save_watchlist(watch_frame([{"ticker": "A.NS"}]))

    assert store.save_watchlist(pd.DataFrame()) == 0
    assert store.load_watchlist().empty


def test_saving_a_none_watchlist_clears_it(store: PortfolioStore) -> None:
    """'The user emptied their watchlist' must be storable."""
    store.save_watchlist(watch_frame([{"ticker": "A.NS"}]))

    assert store.save_watchlist(None) == 0
    assert store.load_watchlist().empty


def test_a_blank_watchlist_ticker_row_is_skipped(store: PortfolioStore) -> None:
    stored = store.save_watchlist(
        watch_frame([{"ticker": "A.NS"}, {"ticker": ""}, {"ticker": None}, {"ticker": "nan"}])
    )

    assert stored == 1
    assert list(store.load_watchlist()["ticker"]) == ["A.NS"]


def test_a_duplicated_watchlist_ticker_keeps_its_first_occurrence(store: PortfolioStore) -> None:
    """``ticker`` is UNIQUE; silently merging two notes would lose one."""
    stored = store.save_watchlist(
        watch_frame([{"ticker": "A.NS", "note": "first"}, {"ticker": "a.ns", "note": "second"}])
    )

    assert stored == 1
    assert store.load_watchlist()["note"].iloc[0] == "first"


def test_a_watchlist_frame_without_a_ticker_column_stores_nothing(
    store: PortfolioStore,
) -> None:
    assert store.save_watchlist(pd.DataFrame({"note": ["orphan"]})) == 0


def test_watchlist_column_names_are_matched_case_insensitively(store: PortfolioStore) -> None:
    stored = store.save_watchlist(
        pd.DataFrame({"Ticker": ["A.NS"], "Target Price": [100.0], "Note": ["x"]})
    )

    assert stored == 1
    assert float(store.load_watchlist()["target_price"].iloc[0]) == 100.0


def test_an_unparseable_added_at_is_stored_blank_not_as_junk_text(
    store: PortfolioStore,
) -> None:
    store.save_watchlist(watch_frame([{"ticker": "A.NS", "added_at": "whenever"}]))

    assert store.load_watchlist()["added_at"].iloc[0] == ""


def test_the_watchlist_survives_reopening_the_database(db_path: Path) -> None:
    with PortfolioStore(db_path) as first:
        first.save_watchlist(watch_frame([{"ticker": "A.NS", "target_price": 42.0}]))

    with PortfolioStore(db_path) as second:
        assert float(second.load_watchlist()["target_price"].iloc[0]) == 42.0


def test_the_watchlist_is_independent_of_portfolios(store: PortfolioStore, holdings_frame) -> None:
    """A watched name is not a holding of anything, so deleting a portfolio
    must not cascade into it."""
    store.save_portfolio("Core", holdings_frame)
    store.save_watchlist(watch_frame([{"ticker": "A.NS"}]))

    store.delete_portfolio("Core")

    assert list(store.load_watchlist()["ticker"]) == ["A.NS"]


def test_the_watchlist_column_contract_matches_the_analytics_module() -> None:
    """Regression guard: the store and src.watchlist must agree on the schema."""
    from src.watchlist import WATCHLIST_COLUMNS as ANALYTICS_COLUMNS

    assert WATCHLIST_COLUMNS == ANALYTICS_COLUMNS


# ---------------------------------------------------------------------------
# store - the alert columns, and migrating a database that predates them
# ---------------------------------------------------------------------------
#: The watchlist table exactly as versions before price alerts created it.
_LEGACY_WATCHLIST = """
    CREATE TABLE watchlist (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT    NOT NULL UNIQUE,
        note         TEXT    NOT NULL DEFAULT '',
        target_price REAL,
        added_at     TEXT    NOT NULL DEFAULT ''
    )
"""


def _legacy_database(path: Path, rows: int = 1) -> None:
    """Build a store the way an earlier version would have left it."""
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(_LEGACY_WATCHLIST)
        for i in range(rows):
            conn.execute(
                "INSERT INTO watchlist (ticker, note, target_price, added_at) VALUES (?,?,?,?)",
                (f"OLD{i}.NS", f"note {i}", 100.0 + i, "2026-01-15"),
            )
    conn.close()


def test_a_database_predating_alerts_is_migrated_in_place(tmp_path: Path) -> None:
    """The one that matters: CREATE TABLE IF NOT EXISTS leaves an existing
    table untouched, so without the migration every alert write would fail
    with "no such column" on the one database holding the user's data."""
    path = tmp_path / "legacy.db"
    _legacy_database(path)

    with PortfolioStore(path) as store:
        columns = {row[1] for row in store._connection().execute("PRAGMA table_info(watchlist)")}

    assert {"direction", "alert_enabled", "alert_state", "last_fired", "reference_price"} <= columns


def test_migrating_preserves_every_existing_row(tmp_path: Path) -> None:
    """An additive ALTER TABLE, never a rebuild: the data this file exists to
    protect must survive the upgrade untouched."""
    path = tmp_path / "legacy.db"
    _legacy_database(path, rows=3)

    with PortfolioStore(path) as store:
        frame = store.load_watchlist()

    assert list(frame["ticker"]) == ["OLD0.NS", "OLD1.NS", "OLD2.NS"]
    assert list(frame["note"]) == ["note 0", "note 1", "note 2"]
    assert list(frame["target_price"]) == [100.0, 101.0, 102.0]


def test_a_migrated_row_has_no_alert_switched_on(tmp_path: Path) -> None:
    """An upgrade must not start e-mailing about targets set years ago."""
    path = tmp_path / "legacy.db"
    _legacy_database(path)

    with PortfolioStore(path) as store:
        record = store.load_watchlist().to_dict("records")[0]

    assert record["alert_enabled"] is False
    assert record["direction"] == ""
    assert record["alert_state"] == "armed"


def test_migrating_is_idempotent(tmp_path: Path) -> None:
    """Opening the store repeatedly must not attempt the ALTER again."""
    path = tmp_path / "legacy.db"
    _legacy_database(path)

    for _ in range(3):
        with PortfolioStore(path) as store:
            assert not store.load_watchlist().empty


def test_a_fresh_database_needs_no_migration(tmp_path: Path) -> None:
    """The columns are in _SCHEMA too, so a new store is already current and
    the migration finds nothing to do."""
    with PortfolioStore(tmp_path / "fresh.db") as store:
        applied = store._apply_migrations(store._connection())

    assert applied == []


def test_the_alert_columns_round_trip(store: PortfolioStore) -> None:
    store.save_watchlist(
        watch_frame(
            [
                {
                    "ticker": "AAA.NS",
                    "note": "keep",
                    "target_price": 110.0,
                    "direction": "above",
                    "alert_enabled": True,
                    "alert_state": "triggered",
                    "last_fired": "2026-09-03T10:30:00",
                    "reference_price": 100.0,
                }
            ]
        )
    )
    record = store.load_watchlist().to_dict("records")[0]

    assert record["direction"] == "above"
    assert record["alert_enabled"] is True
    assert record["alert_state"] == "triggered"
    assert record["last_fired"] == "2026-09-03T10:30:00"
    assert record["reference_price"] == 100.0


def test_a_frame_without_alert_columns_saves_as_no_alert(store: PortfolioStore) -> None:
    """What a CSV export or an older grid legitimately means."""
    store.save_watchlist(watch_frame([{"ticker": "AAA.NS", "target_price": 110.0}]))
    record = store.load_watchlist().to_dict("records")[0]

    assert record["alert_enabled"] is False
    assert record["alert_state"] == "armed"


def test_an_absent_alert_column_never_switches_alerting_on(
    store: PortfolioStore,
) -> None:
    """Regression guard. ``pd.DataFrame(rows, columns=...)`` fills a key the
    rows never supplied with NaN, and ``bool(float("nan"))`` is **True** -- so
    this path once stored ``alert_enabled = 1`` for every row of a watchlist
    that had no alerts at all. On an upgraded database that would have started
    e-mailing about targets set long ago.
    """
    frame = pd.DataFrame(
        [{"ticker": "AAA.NS", "target_price": 110.0}], columns=list(WATCHLIST_COLUMNS)
    )
    assert frame["alert_enabled"].isna().all(), "the fixture must exercise the NaN path"

    store.save_watchlist(frame)

    assert not store.load_watchlist()["alert_enabled"].any()


def test_an_absent_schedule_flag_does_not_become_true(store: PortfolioStore) -> None:
    """The same NaN trap on the schedules table."""
    record = schedule_record()
    record["attach_excel"] = float("nan")
    store.save_schedules([record])

    assert store.load_schedules()[0]["attach_excel"] is False


def test_an_unrecognised_alert_state_is_stored_as_armed(store: PortfolioStore) -> None:
    """A corrupt state must not leave a rule permanently silent."""
    store.save_watchlist(
        watch_frame([{"ticker": "AAA.NS", "target_price": 110.0, "alert_state": "exploded"}])
    )

    assert store.load_watchlist()["alert_state"].iloc[0] == "armed"


def test_updating_the_alert_state_touches_only_that_row(store: PortfolioStore) -> None:
    """A targeted UPDATE, and the only write the alert runner performs: a
    wholesale replace would undo a note the user edited in the browser while
    quotes were being fetched."""
    store.save_watchlist(
        watch_frame(
            [
                {"ticker": "AAA.NS", "note": "first", "target_price": 110.0},
                {"ticker": "BBB.NS", "note": "second", "target_price": 90.0},
            ]
        )
    )

    assert store.update_alert_state("AAA.NS", "triggered", datetime(2026, 9, 3, 10, 30))
    rows = {r["ticker"]: r for r in store.load_watchlist().to_dict("records")}

    assert rows["AAA.NS"]["alert_state"] == "triggered"
    assert rows["AAA.NS"]["last_fired"] == "2026-09-03T10:30:00"
    assert rows["AAA.NS"]["note"] == "first"
    assert rows["BBB.NS"]["alert_state"] == "armed"
    assert rows["BBB.NS"]["last_fired"] == ""


def test_updating_the_alert_state_can_also_persist_the_direction(
    store: PortfolioStore,
) -> None:
    """An inferred direction is written back so it stops being re-inferred
    from a reference price that no longer reflects the user's intent."""
    store.save_watchlist(watch_frame([{"ticker": "AAA.NS", "target_price": 110.0}]))

    store.update_alert_state("AAA.NS", "triggered", datetime(2026, 9, 3, 10, 30), "above")

    assert store.load_watchlist()["direction"].iloc[0] == "above"


def test_updating_an_unknown_ticker_reports_that_nothing_moved(
    store: PortfolioStore,
) -> None:
    assert store.update_alert_state("NOPE.NS", "triggered") is False


def test_updating_with_a_blank_ticker_is_refused(store: PortfolioStore) -> None:
    store.save_watchlist(watch_frame([{"ticker": "AAA.NS", "target_price": 110.0}]))

    assert store.update_alert_state("", "triggered") is False


def test_an_unrecognised_state_is_written_as_armed(store: PortfolioStore) -> None:
    store.save_watchlist(watch_frame([{"ticker": "AAA.NS", "target_price": 110.0}]))
    store.update_alert_state("AAA.NS", "nonsense")

    assert store.load_watchlist()["alert_state"].iloc[0] == "armed"


def test_the_watchlist_column_contract_still_matches_the_analytics_module() -> None:
    """Widened on both sides at once, or the round trip loses the alert."""
    from src.watchlist import WATCHLIST_COLUMNS as ANALYTICS_COLUMNS

    assert WATCHLIST_COLUMNS == ANALYTICS_COLUMNS


# ---------------------------------------------------------------------------
# store - alert settings
# ---------------------------------------------------------------------------
def test_alert_settings_default_to_off(store: PortfolioStore) -> None:
    """An e-mail nobody asked for is worse than no e-mail, so a fresh store
    must never report alerting as on."""
    record = store.load_alert_settings()

    assert record["enabled"] is False
    assert record["recipients"] == ""


def test_alert_settings_round_trip(store: PortfolioStore) -> None:
    store.save_alert_settings(
        {
            "enabled": True,
            "recipients": "a@x.com, b@x.com",
            "rearm_pct": 2.5,
            "cooldown_hours": 6.0,
            "use_session_range": False,
            "timezone": "America/New_York",
        }
    )
    record = store.load_alert_settings()

    assert record["enabled"] is True
    assert record["recipients"] == "a@x.com, b@x.com"
    assert record["rearm_pct"] == 2.5
    assert record["cooldown_hours"] == 6.0
    assert record["use_session_range"] is False
    assert record["timezone"] == "America/New_York"


def test_saving_alert_settings_repeatedly_keeps_exactly_one_row(
    store: PortfolioStore,
) -> None:
    """Two rows disagreeing about who gets notified would be a genuinely
    confusing bug to diagnose from the outside, so id is pinned to 1."""
    for i in range(5):
        store.save_alert_settings({"enabled": True, "recipients": f"user{i}@x.com"})

    count = store._connection().execute("SELECT COUNT(*) FROM alert_settings").fetchone()[0]

    assert count == 1
    assert store.load_alert_settings()["recipients"] == "user4@x.com"


def test_saving_none_alert_settings_changes_nothing(store: PortfolioStore) -> None:
    store.save_alert_settings({"enabled": True, "recipients": "a@x.com"})

    assert store.save_alert_settings(None) is False
    assert store.load_alert_settings()["enabled"] is True


def test_alert_settings_survive_reopening_the_database(db_path: Path) -> None:
    with PortfolioStore(db_path) as first:
        first.save_alert_settings({"enabled": True, "recipients": "a@x.com"})

    with PortfolioStore(db_path) as second:
        assert second.load_alert_settings()["recipients"] == "a@x.com"


def test_alert_settings_hold_no_credential() -> None:
    """As with `schedules`: who to notify, never how to reach the mail
    server. A password in data/quantfolio.db would be a plaintext secret
    inside the project directory."""
    columns = {c.lower() for c in ALERT_SETTINGS_COLUMNS}

    assert not columns & {"password", "smtp_password", "host", "smtp_host", "username"}


def test_unparseable_alert_settings_fall_back_rather_than_raising(
    store: PortfolioStore,
) -> None:
    store.save_alert_settings({"enabled": True, "rearm_pct": "wide", "cooldown_hours": None})
    record = store.load_alert_settings()

    assert record["rearm_pct"] == 1.0
    assert record["cooldown_hours"] == 12.0


# ---------------------------------------------------------------------------
# store — scheduled reports
# ---------------------------------------------------------------------------
def schedule_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "name": "Weekly review",
        "portfolio": "Core",
        "recipients": "owner@example.com",
        "frequency": "weekly",
        "hour": 8,
        "minute": 0,
        "weekday": 0,
        "day_of_month": 1,
        "timezone": "Asia/Kolkata",
        "enabled": True,
        "last_sent": "",
        "period": "1y",
        "benchmark": "NIFTY 50",
        "base_currency": "INR",
        "attach_excel": False,
    }
    record.update(overrides)
    return record


def test_a_fresh_store_has_no_schedules(store: PortfolioStore) -> None:
    assert store.load_schedules() == []


def test_a_schedule_round_trips_field_for_field(store: PortfolioStore) -> None:
    original = schedule_record(
        frequency="monthly", day_of_month=28, hour=19, minute=45, attach_excel=True
    )
    store.save_schedules([original])

    assert store.load_schedules() == [original]


def test_sqlite_integers_read_back_as_booleans(store: PortfolioStore) -> None:
    """SQLite has no boolean type, so the store normalises 0/1 here rather
    than leaving every caller to remember."""
    store.save_schedules([schedule_record(enabled=False, attach_excel=True)])
    record = store.load_schedules()[0]

    assert record["enabled"] is False
    assert record["attach_excel"] is True


def test_saving_replaces_the_whole_schedule_list(store: PortfolioStore) -> None:
    store.save_schedules([schedule_record(name="A"), schedule_record(name="B")])
    store.save_schedules([schedule_record(name="C")])

    assert [r["name"] for r in store.load_schedules()] == ["C"]


def test_saving_none_clears_every_schedule(store: PortfolioStore) -> None:
    store.save_schedules([schedule_record()])

    assert store.save_schedules(None) == 0
    assert store.load_schedules() == []


def test_a_nameless_schedule_row_is_skipped(store: PortfolioStore) -> None:
    stored = store.save_schedules([schedule_record(name=""), schedule_record(name="Real")])

    assert stored == 1
    assert [r["name"] for r in store.load_schedules()] == ["Real"]


def test_a_duplicate_name_keeps_its_first_occurrence(store: PortfolioStore) -> None:
    """``name`` is UNIQUE, and silently merging two schedules onto one name
    would lose one of them."""
    store.save_schedules([schedule_record(hour=6), schedule_record(hour=18)])
    records = store.load_schedules()

    assert len(records) == 1
    assert records[0]["hour"] == 6


def test_duplicate_names_are_matched_case_insensitively(store: PortfolioStore) -> None:
    store.save_schedules([schedule_record(name="Weekly"), schedule_record(name="WEEKLY")])

    assert len(store.load_schedules()) == 1


def test_an_unreadable_hour_falls_back_rather_than_raising(store: PortfolioStore) -> None:
    store.save_schedules([schedule_record(hour="half past eight")])

    assert store.load_schedules()[0]["hour"] == 8


def test_a_non_mapping_row_is_ignored(store: PortfolioStore) -> None:
    assert store.save_schedules(["not a record", schedule_record()]) == 1


def test_schedules_keep_their_insertion_order(store: PortfolioStore) -> None:
    """Save is a wholesale replace, so ordering the read differently from the
    write would flip the list on every save/load cycle."""
    names = ["Third", "First", "Second"]
    store.save_schedules([schedule_record(name=n) for n in names])

    assert [r["name"] for r in store.load_schedules()] == names


def test_schedules_survive_reopening_the_database(db_path: Path) -> None:
    with PortfolioStore(db_path) as first:
        first.save_schedules([schedule_record()])

    with PortfolioStore(db_path) as second:
        assert [r["name"] for r in second.load_schedules()] == ["Weekly review"]


def test_marking_a_schedule_sent_touches_only_that_row(store: PortfolioStore) -> None:
    """The only write the scheduled runner performs, and deliberately narrow:
    it cannot clobber a schedule the user edited while a report was building."""
    store.save_schedules([schedule_record(name="A"), schedule_record(name="B")])

    assert store.mark_schedule_sent("A", datetime(2026, 9, 1, 8, 0))
    by_name = {r["name"]: r for r in store.load_schedules()}

    assert by_name["A"]["last_sent"] == "2026-09-01T08:00:00"
    assert by_name["B"]["last_sent"] == ""


def test_marking_an_unknown_schedule_reports_that_nothing_moved(
    store: PortfolioStore,
) -> None:
    assert store.mark_schedule_sent("nope", datetime(2026, 9, 1, 8, 0)) is False


def test_marking_with_a_blank_name_is_refused(store: PortfolioStore) -> None:
    store.save_schedules([schedule_record()])

    assert store.mark_schedule_sent("", datetime(2026, 9, 1, 8, 0)) is False


def test_a_schedule_is_independent_of_portfolios(
    store: PortfolioStore, holdings_frame: pd.DataFrame
) -> None:
    """Stored by NAME, not by foreign key: deleting and recreating a portfolio
    must not silently orphan a live schedule."""
    store.save_portfolio("Core", holdings_frame)
    store.save_schedules([schedule_record()])

    store.delete_portfolio("Core")

    assert [r["portfolio"] for r in store.load_schedules()] == ["Core"]


def test_no_credential_column_exists_on_the_schedules_table(
    store: PortfolioStore,
) -> None:
    """The security property: an SMTP password in data/quantfolio.db would be
    a plaintext secret sitting inside the project directory."""
    columns = {c.lower() for c in SCHEDULE_COLUMNS}

    assert not columns & {"password", "smtp_password", "host", "smtp_host", "username"}


def test_the_schedule_column_contract_matches_the_reporting_module() -> None:
    """Regression guard: the store and src.reporting.schedule must agree."""
    from src.reporting.schedule import SCHEDULE_FIELDS

    assert SCHEDULE_COLUMNS == SCHEDULE_FIELDS


# ---------------------------------------------------------------------------
# fetcher — ISIN lookup
# ---------------------------------------------------------------------------
def test_a_valid_isin_resolves_through_the_provider(fake_search) -> None:
    fake_search.payload = [
        {"symbol": "RELIANCE.NS", "longname": "Reliance Industries", "quoteType": "EQUITY"}
    ]

    hit = fetcher.lookup_isin("INE002A01018")

    assert hit is not None
    assert hit["symbol"] == "RELIANCE.NS"
    assert hit["isin"] == "INE002A01018"
    assert hit["source"] == "yahoo-isin"


def test_a_malformed_isin_never_reaches_the_provider(fake_search) -> None:
    """Asking about a typo can only produce a plausible wrong answer."""
    assert fetcher.lookup_isin("INE002A01019") is None
    assert fetcher.lookup_isin("RELIANCE") is None
    assert fake_search.calls == [], "no request should have been made"


def test_an_indian_isin_refuses_a_non_indian_listing(fake_search) -> None:
    """The mis-pairing a check digit cannot catch."""
    fake_search.payload = [{"symbol": "RIGD.L", "longname": "Reliance GDR", "quoteType": "EQUITY"}]

    assert fetcher.lookup_isin("INE002A01018") is None


def test_an_indian_isin_skips_past_a_foreign_hit_to_the_nse_one(fake_search) -> None:
    fake_search.payload = [
        {"symbol": "RIGD.L", "longname": "Reliance GDR", "quoteType": "EQUITY"},
        {"symbol": "RELIANCE.NS", "longname": "Reliance Industries", "quoteType": "EQUITY"},
    ]

    assert fetcher.lookup_isin("INE002A01018")["symbol"] == "RELIANCE.NS"


def test_a_us_isin_refuses_a_suffixed_foreign_listing(fake_search) -> None:
    fake_search.payload = [{"symbol": "AAPL.MX", "longname": "Apple Mexico", "quoteType": "EQUITY"}]

    assert fetcher.lookup_isin("US0378331005") is None


def test_a_us_isin_accepts_an_unsuffixed_symbol(fake_search) -> None:
    fake_search.payload = [{"symbol": "AAPL", "longname": "Apple", "quoteType": "EQUITY"}]

    assert fetcher.lookup_isin("US0378331005")["symbol"] == "AAPL"


def test_a_provider_with_nothing_to_say_yields_none(fake_search) -> None:
    fake_search.payload = []

    assert fetcher.lookup_isin("INE002A01018") is None


def test_an_exploding_provider_yields_none_rather_than_raising(fake_search) -> None:
    fake_search.explode = True

    assert fetcher.lookup_isin("INE002A01018") is None


def test_lookup_isins_isolates_a_failure_and_still_returns_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_one(code):
        if code == "INE002A01018":
            return {"symbol": "RELIANCE.NS", "isin": code, "source": "yahoo-isin"}
        return None

    monkeypatch.setattr(fetcher, "lookup_isin", fake_one)

    found, errors = fetcher.lookup_isins(["INE002A01018", "GB0002634946"])

    assert found == {"INE002A01018": "RELIANCE.NS"}
    assert "GB0002634946" in errors


def test_lookup_isins_deduplicates_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_one(code):
        seen.append(code)
        return

    monkeypatch.setattr(fetcher, "lookup_isin", fake_one)
    fetcher.lookup_isins(["ine002a01018", "INE002A01018", "", "  "])

    assert seen == ["INE002A01018"]


def test_lookup_isins_of_nothing_is_two_empty_dicts() -> None:
    assert fetcher.lookup_isins([]) == ({}, {})


# ---------------------------------------------------------------------------
# fetcher - financial statements
# ---------------------------------------------------------------------------
class FakeStatementTicker:
    """Minimal ``yfinance.Ticker`` exposing only the statement surface."""

    frames: ClassVar[dict[str, pd.DataFrame]] = {}
    explode: ClassVar[set[str]] = set()

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def _get(self, name: str) -> pd.DataFrame:
        if name in type(self).explode:
            raise RuntimeError(f"provider refused {name}")
        return type(self).frames.get(name, pd.DataFrame())

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self._get("balance_sheet")

    @property
    def financials(self) -> pd.DataFrame:
        return self._get("income")

    @property
    def cashflow(self) -> pd.DataFrame:
        return self._get("cash_flow")


def _statement_frame(label: str) -> pd.DataFrame:
    periods = pd.to_datetime(["2025-03-31", "2024-03-31"])
    return pd.DataFrame({p: [100.0, 200.0] for p in periods}, index=[label, f"{label} Other"])


@pytest.fixture
def fake_statements(monkeypatch: pytest.MonkeyPatch):
    FakeStatementTicker.explode = set()
    FakeStatementTicker.frames = {
        "balance_sheet": _statement_frame("Total Assets"),
        "income": _statement_frame("Total Revenue"),
        "cash_flow": _statement_frame("Operating Cash Flow"),
    }
    monkeypatch.setattr(fetcher.yf, "Ticker", FakeStatementTicker)
    return FakeStatementTicker


def test_all_three_statements_come_back_keyed(fake_statements) -> None:
    frames, notice = fetcher.fetch_statements("RELIANCE.NS")

    assert notice == ""
    assert set(frames) == {"balance_sheet", "income", "cash_flow"}
    assert frames["balance_sheet"].shape == (2, 2)


def test_the_frames_are_passed_through_untouched(fake_statements) -> None:
    """No renaming and no filling here: normalising is the analytics layer's
    job, and this module stays a thin monkeypatchable boundary."""
    frames, _notice = fetcher.fetch_statements("RELIANCE.NS")

    assert "Total Assets" in frames["balance_sheet"].index
    assert frames["balance_sheet"].iat[0, 0] == 100.0


def test_a_blank_ticker_is_refused(fake_statements) -> None:
    frames, notice = fetcher.fetch_statements("   ")

    assert all(f.empty for f in frames.values())
    assert "No ticker" in notice


def test_a_missing_statement_is_named_not_hidden(fake_statements) -> None:
    """ "This company files no cash-flow statement here" and "the request
    failed" are different facts, and a caller has to be able to tell them
    apart."""
    fake_statements.frames = {"balance_sheet": _statement_frame("Total Assets")}

    frames, notice = fetcher.fetch_statements("RELIANCE.NS")

    assert not frames["balance_sheet"].empty
    assert frames["cash_flow"].empty
    assert "cash flow" in notice
    assert "income" in notice


def test_no_statements_at_all_is_reported(fake_statements) -> None:
    fake_statements.frames = {}

    frames, notice = fetcher.fetch_statements("ETF.NS")

    assert all(f.empty for f in frames.values())
    assert "published no financial statements" in notice


def test_one_exploding_statement_does_not_sink_the_others(fake_statements) -> None:
    """Three separate provider calls; one failing must not lose the two that
    worked."""
    fake_statements.explode = {"cash_flow"}

    frames, notice = fetcher.fetch_statements("RELIANCE.NS")

    assert not frames["balance_sheet"].empty
    assert not frames["income"].empty
    assert frames["cash_flow"].empty
    assert "cash flow" in notice


def test_every_statement_exploding_is_reported_not_raised(fake_statements) -> None:
    fake_statements.explode = {"balance_sheet", "income", "cash_flow"}

    frames, notice = fetcher.fetch_statements("RELIANCE.NS")

    assert all(f.empty for f in frames.values())
    assert "published no financial statements" in notice


def test_an_empty_frame_counts_as_absent(fake_statements) -> None:
    fake_statements.frames = {
        "balance_sheet": _statement_frame("Total Assets"),
        "income": pd.DataFrame(),
        "cash_flow": _statement_frame("Operating Cash Flow"),
    }

    frames, notice = fetcher.fetch_statements("RELIANCE.NS")

    assert frames["income"].empty
    assert "income" in notice


def test_the_statements_feed_the_fundamental_scores(fake_statements) -> None:
    """The integration: fetched frames become Statements and score."""
    from src.analytics import fundamentals as fu

    frames, _notice = fetcher.fetch_statements("RELIANCE.NS")
    statements = fu.read_statements(
        frames["balance_sheet"], frames["income"], frames["cash_flow"], ticker="RELIANCE.NS"
    )

    assert statements.ok
    assert statements.n_periods == 2
    assert statements.latest.get("total_assets") == 100.0


# ---------------------------------------------------------------------------
# fetcher — option chains
# ---------------------------------------------------------------------------
class FakeChain:
    """Stand-in for the object ``yfinance.Ticker.option_chain`` returns."""

    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


class FakeOptionTicker:
    """Minimal ``yfinance.Ticker`` exposing only the option surface."""

    expiries: ClassVar[tuple[str, ...]] = ("2026-01-16", "2026-02-20")
    chain: ClassVar[FakeChain | None] = None
    explode: ClassVar[bool] = False

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def options(self):
        if type(self).explode:
            raise RuntimeError("provider refused the expiry list")
        return type(self).expiries

    def option_chain(self, date=None, tz=None):
        if type(self).explode:
            raise RuntimeError("provider refused the chain")
        return type(self).chain


def _side(strikes: list[float], itm: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": [f"X{int(s)}" for s in strikes],
            "strike": strikes,
            "lastPrice": [round(2.0 + i * 0.5, 2) for i in range(len(strikes))],
            "bid": [round(1.9 + i * 0.5, 2) for i in range(len(strikes))],
            "ask": [round(2.1 + i * 0.5, 2) for i in range(len(strikes))],
            "impliedVolatility": [0.24 + i * 0.01 for i in range(len(strikes))],
            "volume": [100 + i for i in range(len(strikes))],
            "openInterest": [1000 + i for i in range(len(strikes))],
            "inTheMoney": itm,
        }
    )


@pytest.fixture
def fake_options(monkeypatch: pytest.MonkeyPatch):
    """Install ``FakeOptionTicker`` and give it a two-strike chain."""
    FakeOptionTicker.expiries = ("2026-01-16", "2026-02-20")
    FakeOptionTicker.explode = False
    FakeOptionTicker.chain = FakeChain(
        calls=_side([95.0, 105.0], [True, False]),
        puts=_side([95.0, 105.0], [False, True]),
    )
    monkeypatch.setattr(fetcher.yf, "Ticker", FakeOptionTicker)
    return FakeOptionTicker


# -- availability, checked BEFORE any request -------------------------------
def test_a_us_symbol_is_reported_as_having_chains() -> None:
    available, reason = fetcher.option_chain_available("AAPL")

    assert available
    assert reason == ""


@pytest.mark.parametrize("symbol", ["RELIANCE.NS", "TATASTEEL.BO", "reliance.ns"])
def test_an_nse_or_bse_symbol_is_reported_as_having_no_chains(symbol: str) -> None:
    """The message must name the DATA SOURCE as the gap. Indian equity options
    are extremely liquid, so an empty table would read as "this stock has no
    options", which is flatly untrue."""
    available, reason = fetcher.option_chain_available(symbol)

    assert not available
    assert "no option chain" in reason
    assert "manually" in reason


def test_an_indian_index_gets_its_own_explanation() -> None:
    available, reason = fetcher.option_chain_available("^NSEI")

    assert not available
    assert "NIFTY" in reason


def test_a_us_index_is_reported_as_available() -> None:
    assert fetcher.option_chain_available("^GSPC")[0]


def test_a_blank_symbol_is_refused() -> None:
    available, reason = fetcher.option_chain_available("")

    assert not available
    assert reason


def test_an_unavailable_symbol_never_reaches_the_provider(fake_options) -> None:
    """Checked before the request, so an NSE symbol costs no network call."""
    expiries, reason = fetcher.fetch_option_expiries("RELIANCE.NS")

    assert expiries == ()
    assert "no option chain" in reason


# -- expiries ---------------------------------------------------------------
def test_expiries_are_returned_for_a_us_symbol(fake_options) -> None:
    expiries, reason = fetcher.fetch_option_expiries("AAPL")

    assert expiries == ("2026-01-16", "2026-02-20")
    assert reason == ""


def test_an_empty_expiry_list_comes_with_a_reason(fake_options) -> None:
    """An empty tuple ALWAYS carries an explanation, so the caller can tell
    "provider does not carry it" from "request failed" from "no listed
    options"."""
    fake_options.expiries = ()

    expiries, reason = fetcher.fetch_option_expiries("AAPL")

    assert expiries == ()
    assert "no expiry dates" in reason


def test_an_exploding_provider_yields_no_expiries_and_a_reason(fake_options) -> None:
    fake_options.explode = True

    expiries, reason = fetcher.fetch_option_expiries("AAPL")

    assert expiries == ()
    assert "could not be fetched" in reason


# -- the chain --------------------------------------------------------------
def test_calls_and_puts_come_back_in_one_frame(fake_options) -> None:
    """One frame, not two: every consumer wants to filter by strike across
    both sides."""
    chain, reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert reason == ""
    assert tuple(chain.columns) == fetcher.OPTION_COLUMNS
    assert set(chain["option_type"]) == {"call", "put"}
    assert len(chain) == 4


def test_the_chain_is_sorted_by_type_then_strike(fake_options) -> None:
    chain, _reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    calls = chain[chain["option_type"] == "call"]
    assert list(calls["strike"]) == sorted(calls["strike"])


def test_chain_prices_are_passed_through_untouched(fake_options) -> None:
    """No midpoint substitution and no filling a missing bid with a stale
    trade. An option that has not traded today has a stale last price, and
    that is information the caller needs."""
    chain, _reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")
    row = chain[(chain["option_type"] == "call") & (chain["strike"] == 95.0)].iloc[0]

    assert row["last_price"] == pytest.approx(2.0)
    assert row["bid"] == pytest.approx(1.9)
    assert row["ask"] == pytest.approx(2.1)


def test_rows_without_a_usable_strike_are_dropped(fake_options) -> None:
    fake_options.chain = FakeChain(
        calls=pd.DataFrame(
            {
                "contractSymbol": ["A", "B", "C"],
                "strike": [100.0, None, -5.0],
                "lastPrice": [1.0, 2.0, 3.0],
            }
        ),
        puts=pd.DataFrame(),
    )

    chain, _reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert len(chain) == 1
    assert chain["strike"].iloc[0] == 100.0


def test_a_chain_with_no_usable_strike_is_reported(fake_options) -> None:
    fake_options.chain = FakeChain(
        calls=pd.DataFrame({"contractSymbol": ["A"], "strike": [None], "lastPrice": [1.0]}),
        puts=pd.DataFrame(),
    )

    chain, reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert chain.empty
    assert "usable strike" in reason


def test_an_empty_chain_is_reported(fake_options) -> None:
    fake_options.chain = FakeChain(calls=pd.DataFrame(), puts=pd.DataFrame())

    chain, reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert chain.empty
    assert "empty chain" in reason


def test_a_one_sided_chain_still_works(fake_options) -> None:
    """Some expiries genuinely quote only one side."""
    fake_options.chain = FakeChain(calls=_side([100.0], [False]), puts=pd.DataFrame())

    chain, reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert reason == ""
    assert set(chain["option_type"]) == {"call"}


def test_an_exploding_chain_request_is_reported_not_raised(fake_options) -> None:
    fake_options.explode = True

    chain, reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")

    assert chain.empty
    assert "could not be fetched" in reason


def test_a_missing_expiry_argument_is_refused(fake_options) -> None:
    chain, reason = fetcher.fetch_option_chain("AAPL", "")

    assert chain.empty
    assert "No expiry" in reason


def test_an_nse_chain_request_is_refused_with_the_data_source_reason(
    fake_options,
) -> None:
    chain, reason = fetcher.fetch_option_chain("RELIANCE.NS", "2026-01-16")

    assert chain.empty
    assert tuple(chain.columns) == fetcher.OPTION_COLUMNS
    assert "no option chain" in reason


def test_the_chain_feeds_the_overlay_analytics(fake_options) -> None:
    """The integration: a fetched row becomes an OptionQuote and prices."""
    from src.analytics import options as op

    chain, _reason = fetcher.fetch_option_chain("AAPL", "2026-01-16")
    row = chain[(chain["option_type"] == "call") & (chain["strike"] == 105.0)].iloc[0]
    quote = op.OptionQuote(
        option_type="call",
        strike=float(row["strike"]),
        premium=float(row["last_price"]),
        expiry=pd.Timestamp("2026-01-16").date(),
        implied_vol=float(row["implied_vol"]),
        contract_size=100,
    )

    overlay = op.covered_call(100.0, 500, quote, as_of=date(2025, 12, 17))

    assert overlay.ok
    assert overlay.contracts == 5


# ---------------------------------------------------------------------------
# live provider contract — the ONLY test here that needs the wire
# ---------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.slow
def test_the_live_provider_still_exposes_the_surface_the_fetcher_depends_on() -> None:
    """Deselect with ``-m "not network"``; CI does.

    Every other test in this module runs against ``FakeMarket``, which is only
    as good as its resemblance to yfinance. This is the one check that can
    notice the real library dropping ``fast_info``, renaming ``last_price`` or
    changing what ``history(auto_adjust=True)`` returns. It SKIPS rather than
    fails when the provider is unreachable, so it can never be flaky — an
    unreachable Yahoo is not a defect in this repository.
    """
    yfinance = pytest.importorskip("yfinance")

    ticker = yfinance.Ticker(DEFAULT_BENCHMARK_SYMBOL)
    try:
        frame = ticker.history(period="5d", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 - any provider failure is a skip
        pytest.skip(f"provider unreachable: {exc}")

    if frame is None or frame.empty:
        pytest.skip("provider returned no rows (holiday, throttling or outage)")

    assert "Close" in frame.columns, "the fetcher reads the adjusted Close column"
    assert isinstance(frame.index, pd.DatetimeIndex), "_naive_dates assumes a datetime index"
    assert hasattr(ticker, "fast_info"), "the fast-quote path depends on this attribute"
