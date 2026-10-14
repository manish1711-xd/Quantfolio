"""Shared, fully deterministic test fixtures.

Every fixture here is synthetic and seeded. The entire suite runs offline: no
network, no API key, no market hours. That is a deliberate design constraint —
analytics you cannot test without the internet are analytics you cannot trust in
CI, and a portfolio tool whose test suite depends on live prices is a tool whose
tests fail for reasons that have nothing to do with the code.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import isin
from src.models import Quote, SecurityProfile, Transaction

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEED = 20240101
START = "2023-01-02"


# ---------------------------------------------------------------------------
# price data
# ---------------------------------------------------------------------------
def make_prices(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    days: int = 500,
    seed: int = SEED,
    start: str = START,
    drift: float = 0.0004,
    vol: float = 0.012,
    common_factor: float = 0.6,
) -> pd.DataFrame:
    """Correlated geometric random walks on a business-day index.

    ``common_factor`` mixes a shared market factor into each asset, so the
    generated data has realistic positive cross-correlation. Uncorrelated toy
    data hides exactly the bugs a correlation/beta/optimiser test should catch.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=days)
    market = rng.normal(drift, vol, days)

    frame = {}
    for i, ticker in enumerate(tickers):
        idio = rng.normal(drift * (1 + 0.2 * i), vol * (1 + 0.15 * i), days)
        returns = common_factor * market + (1 - common_factor) * idio
        frame[ticker] = 100.0 * (1 + i * 0.5) * np.cumprod(1.0 + returns)
    return pd.DataFrame(frame, index=index)


def make_ohlcv(ticker: str = "AAA", days: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Open/High/Low/Close/Volume frame consistent with a single random walk."""
    close = make_prices((ticker,), days=days, seed=seed).iloc[:, 0]
    rng = np.random.default_rng(seed + 1)
    spread = close * rng.uniform(0.004, 0.018, len(close))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + spread / 2,
            "Low": np.minimum(open_, close) - spread / 2,
            "Close": close,
            "Volume": rng.integers(1e5, 5e6, len(close)).astype(float),
        },
        index=close.index,
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def long_prices() -> pd.DataFrame:
    """Ten years of history — long enough to cover the stress windows."""
    return make_prices(days=2600, start="2015-01-01")


@pytest.fixture
def benchmark_prices() -> pd.Series:
    return make_prices(("BMK",), seed=SEED + 99).iloc[:, 0]


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


@pytest.fixture
def portfolio_values(prices: pd.DataFrame) -> pd.Series:
    values = prices.mul(pd.Series({"AAA": 10.0, "BBB": 5.0, "CCC": 8.0}), axis=1).sum(axis=1)
    values.name = "portfolio_value"
    return values


# ---------------------------------------------------------------------------
# quotes, profiles, holdings
# ---------------------------------------------------------------------------
def make_quote(
    ticker: str,
    price: float,
    currency: str = "INR",
    previous_close: float | None = None,
) -> Quote:
    now = datetime(2025, 9, 1, 10, 30, tzinfo=timezone.utc)
    return Quote(
        ticker=ticker,
        price=price,
        currency=currency,
        as_of=now,
        fetched_at=now,
        is_intraday=False,
        previous_close=previous_close,
    )


@pytest.fixture
def quotes(prices: pd.DataFrame) -> dict[str, Quote]:
    return {ticker: make_quote(ticker, float(prices[ticker].iloc[-1])) for ticker in prices.columns}


@pytest.fixture
def profiles() -> dict[str, SecurityProfile]:
    return {
        "AAA": SecurityProfile(
            "AAA",
            name="Alpha Industries",
            sector="Energy",
            industry="Refining",
            currency="INR",
            quote_type="EQUITY",
        ),
        "BBB": SecurityProfile(
            "BBB",
            name="Beta Software",
            sector="Technology",
            industry="IT Services",
            currency="INR",
            quote_type="EQUITY",
        ),
        "CCC": SecurityProfile(
            "CCC", name="Gamma Index Fund", sector=None, currency="INR", quote_type="ETF"
        ),
    }


@pytest.fixture
def holdings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "quantity": [10.0, 5.0, 8.0],
            "buy_price": [100.0, 140.0, 190.0],
        }
    )


# ---------------------------------------------------------------------------
# transactions
# ---------------------------------------------------------------------------
@pytest.fixture
def transactions() -> list[Transaction]:
    """A hand-checked ledger: two buys and a partial sell on AAA, one buy on BBB.

    FIFO expectations for AAA — BUY 10@100, BUY 10@120, SELL 15@150 — are
    exercised directly in tests/test_lots.py, so keep these numbers stable.
    """
    return [
        Transaction("AAA", date(2023, 1, 10), "BUY", 10, 100.0, fees=10.0),
        Transaction("AAA", date(2023, 6, 15), "BUY", 10, 120.0, fees=12.0),
        Transaction("BBB", date(2023, 7, 1), "BUY", 5, 140.0, fees=7.0),
        Transaction("AAA", date(2024, 8, 20), "SELL", 15, 150.0, fees=15.0),
    ]


@pytest.fixture
def transactions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "trade_date": ["2023-01-10", "2023-06-15", "2023-07-01", "2024-08-20"],
            "action": ["BUY", "buy", "B", "SELL"],
            "quantity": [10, 10, 5, 15],
            "price": [100.0, 120.0, 140.0, 150.0],
            "fees": [10.0, 12.0, 7.0, 15.0],
        }
    )


# ---------------------------------------------------------------------------
# degenerate inputs — every analytics function must survive these
# ---------------------------------------------------------------------------
@pytest.fixture
def empty_series() -> pd.Series:
    return pd.Series(dtype="float64")


@pytest.fixture
def single_value_series() -> pd.Series:
    return pd.Series([100.0], index=pd.bdate_range(START, periods=1))


@pytest.fixture
def flat_series() -> pd.Series:
    """Zero-variance series — the classic divide-by-zero trap."""
    return pd.Series([100.0] * 60, index=pd.bdate_range(START, periods=60))


@pytest.fixture
def nan_series() -> pd.Series:
    return pd.Series([np.nan] * 40, index=pd.bdate_range(START, periods=40))


# ---------------------------------------------------------------------------
# ISIN cache isolation
# ---------------------------------------------------------------------------
#: The repository's real data directory. Read from, never written to by tests.
REPO_DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session", autouse=True)
def _isolated_portfolio_db(tmp_path_factory):
    """Point ``PortfolioStore()`` at a throwaway file for the whole session.

    Several UI modules -- the sidebar, the Watchlist page, the scheduled-email
    panel -- construct ``PortfolioStore()`` with no argument, which resolves
    ``store.DEFAULT_DB_PATH``. Under ``AppTest`` that means the headless page
    tests would read, create tables in, and journal against the developer's
    real ``data/quantfolio.db``. Redirecting the default keeps the suite off
    the user's own saved portfolios entirely.

    Session-scoped and autouse because those modules cache their store in
    ``st.cache_resource``: a connection opened before the redirect would
    outlive it. ``DEFAULT_DB_PATH`` is read inside ``__init__`` rather than
    bound as a default argument, so reassigning the module attribute is
    enough.
    """
    from src.data import store as store_module

    original = store_module.DEFAULT_DB_PATH
    store_module.DEFAULT_DB_PATH = tmp_path_factory.mktemp("store") / "quantfolio.db"
    try:
        yield store_module.DEFAULT_DB_PATH
    finally:
        store_module.DEFAULT_DB_PATH = original


@pytest.fixture(scope="session", autouse=True)
def _isolated_isin_cache(tmp_path_factory):
    """Redirect the ISIN map's data directory at a throwaway copy.

    Importing a broker file that carries both an ISIN and a symbol column
    *learns* the pairing and writes ``data/isin_learned.csv``. That is correct
    product behaviour and completely unacceptable inside a test run: the suite
    would mutate a tracked directory, and one test's learned mappings would
    leak into another's expectations.

    Session-scoped so the seed file is copied once rather than 3,000 times.
    ``load_map`` is ``lru_cache``d on its (defaulted) argument, so the cache is
    cleared on both sides of the swap.
    """
    cache = tmp_path_factory.mktemp("isin_cache")
    seed = REPO_DATA / isin.SEED_FILE
    if seed.is_file():
        shutil.copy(seed, cache / isin.SEED_FILE)

    original = isin.DATA_DIR
    isin.DATA_DIR = cache
    isin.refresh()
    try:
        yield cache
    finally:
        isin.DATA_DIR = original
        isin.refresh()


@pytest.fixture
def isin_cache(_isolated_isin_cache):
    """The isolated ISIN directory, with any learned mappings wiped.

    Use in any test that exercises learning, so it starts from the seed alone
    regardless of what ran before it.
    """
    learned = _isolated_isin_cache / isin.LEARNED_FILE
    learned.unlink(missing_ok=True)
    isin.refresh()
    yield _isolated_isin_cache
    learned.unlink(missing_ok=True)
    isin.refresh()
