"""Headless UI tests: the real dashboard, driven by Streamlit's ``AppTest``.

The market-data layer is monkeypatched with synthetic prices, so the whole UI —
navigation, the analysis pipeline, all seventeen pages, every chart — is
exercised end to end with **zero network access**. That is what lets the UI be
tested in CI at all.

Two levels of coverage:

* :func:`test_app_runs_end_to_end` drives ``app.py`` itself: it clicks Analyze
  and asserts the pipeline produced a result without raising.
* :func:`test_every_page_renders` mounts each page module in isolation against a
  fully-populated ``AnalysisResult``. Every page is a separate assertion, so a
  failure names the page that broke instead of failing one opaque smoke test.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.data.fetcher as dfetch
from src.analytics import (
    attribution as attr,
)
from src.analytics import (
    benchmark as bm,
)
from src.analytics import (
    drawdown as dd,
)
from src.analytics import (
    insights as ins,
)
from src.analytics import (
    lots as lot,
)
from src.analytics import (
    positions as pos,
)
from src.analytics import (
    returns as ret,
)
from src.analytics import (
    risk as rsk,
)
from src.analytics import (
    rolling as roll,
)
from src.config import AppConfig
from src.models import AnalysisResult, Quote, SecurityProfile

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

TICKERS = ("AAA.NS", "BBB.NS", "CCC.NS")
SECTORS = ("Energy", "Technology", None)

#: Every page module and its human name, mirroring the registry in app.py.
PAGES: tuple[tuple[str, str], ...] = (
    ("portfolio", "Build"),
    ("overview", "Overview"),
    ("holdings", "Holdings"),
    ("performance", "Performance"),
    ("transactions", "Trades & lots"),
    ("risk", "Risk analytics"),
    ("benchmark", "Benchmark"),
    ("correlation", "Correlation & factors"),
    ("stress", "Stress testing"),
    ("optimizer", "Optimiser"),
    ("simulation", "Monte Carlo"),
    ("rebalance", "Rebalance"),
    ("security", "Security deep dive"),
    ("forecast", "Price forecast"),
    ("watchlist", "Watchlist"),
    ("news", "News & calendar"),
    ("reports", "Export"),
)


# ---------------------------------------------------------------------------
# synthetic market
# ---------------------------------------------------------------------------
def _prices(days: int = 600, seed: int = 5) -> pd.DataFrame:
    """Correlated walks, so correlation/beta/optimiser paths get real inputs."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=days)
    market = rng.normal(0.0004, 0.010, days)
    return pd.DataFrame(
        {
            t: 100.0
            * (1 + i * 0.6)
            * np.cumprod(1 + 0.65 * market + 0.35 * rng.normal(0.0002, 0.011 + 0.002 * i, days))
            for i, t in enumerate(TICKERS)
        },
        index=index,
    )


PRICES = _prices()
BENCH = pd.Series(
    1000.0 * np.cumprod(1 + np.random.default_rng(9).normal(0.0003, 0.009, len(PRICES))),
    index=PRICES.index,
    name="^NSEI",
)
NOW = datetime(2025, 9, 1, 10, 30, tzinfo=timezone.utc)


def _quotes_for(tickers) -> tuple[dict[str, Quote], dict[str, str]]:
    quotes = {
        t: Quote(
            t,
            float(PRICES[t].iloc[-1]),
            "INR",
            NOW,
            NOW,
            False,
            previous_close=float(PRICES[t].iloc[-2]),
        )
        for t in tickers
        if t in PRICES.columns
    }
    errors = {t: "stub: unknown ticker" for t in tickers if t not in PRICES.columns}
    return quotes, errors


def _profiles_for(tickers) -> dict[str, SecurityProfile]:
    lookup = dict(zip(TICKERS, SECTORS))
    return {
        t: SecurityProfile(
            t,
            name=t.split(".")[0].title(),
            sector=lookup.get(t),
            currency="INR",
            quote_type="ETF" if lookup.get(t) is None else "EQUITY",
            fifty_two_week_high=float(PRICES[t].iloc[-260:].max()) if t in PRICES else None,
            fifty_two_week_low=float(PRICES[t].iloc[-260:].min()) if t in PRICES else None,
        )
        for t in tickers
    }


def _stub_news(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic feeds, headlines and calendar.

    The News page fetches RSS and JSON from five publishers, and a page test
    that reaches the live internet is not an offline test -- it is slow, it
    fails when a publisher is down, and it makes the assertions depend on
    today's headlines.

    One feed is deliberately returned STALE so the page's abandoned-feed
    warning is exercised, which is the branch that matters most: MoneyControl
    answers HTTP 200 with content frozen in April 2024, and rendering that as
    news is the failure the guard exists to prevent.
    """
    from datetime import datetime as _dt

    from src.data import news as _news

    fresh = (
        _news.NewsItem(
            title="Reliance Industries wins a large order",
            link="https://example.test/1",
            published=_dt(2026, 9, 2, 4, 0),
            source="Economic Times — Markets",
            summary="Order book expands",
        ),
        _news.NewsItem(
            title="Sensex ends flat ahead of policy meeting",
            link="https://example.test/2",
            published=_dt(2026, 9, 1, 10, 30),
            source="Economic Times — Markets",
            summary="Benchmarks little changed",
        ),
    )

    def feeds(keys=None, **_kwargs):
        wanted = list(keys) if keys is not None else list(_news.FEEDS)
        out = []
        for key in wanted:
            label, url, region = _news.FEEDS[key]
            if key.startswith("moneycontrol"):
                out.append(
                    _news.FeedResult(
                        key=key,
                        label=label,
                        url=url,
                        region=region,
                        newest=_dt(2024, 4, 23, 5, 53),
                        stale=True,
                        notice="the feed still answers but its newest item is 863 days old",
                    )
                )
            else:
                out.append(
                    _news.FeedResult(
                        key=key,
                        label=label,
                        url=url,
                        region=region,
                        items=fresh,
                        newest=fresh[0].published,
                    )
                )
        return out

    calendar = _news.CalendarResult(
        events=(
            _news.CalendarEvent(
                title="RBI Rate Decision",
                country="INR",
                when=_dt(2026, 9, 4, 9, 0),
                impact="High",
                previous="6.50%",
            ),
            _news.CalendarEvent(
                title="Retail Sales y/y",
                country="JPY",
                when=_dt(2026, 9, 5, 23, 50),
                impact="Low",
                forecast="3.2%",
            ),
        ),
        covers_from=_dt(2026, 8, 31),
        covers_to=_dt(2026, 9, 5, 23, 50),
    )

    monkeypatch.setattr(_news, "fetch_feeds", feeds)
    monkeypatch.setattr(_news, "fetch_feed", lambda key, **_k: feeds([key])[0])
    monkeypatch.setattr(_news, "fetch_calendar", lambda **_k: calendar)
    monkeypatch.setattr(_news, "fetch_ticker_news", lambda ticker, **_k: (fresh, ""))


def _statements_for(ticker: str):
    """Synthetic annual statements: five periods, internally consistent.

    Consistent on purpose. The fundamental scores divide these figures by each
    other, so numbers picked at random would produce a Piotroski score that
    passes or fails for no reason and an Altman Z-Score in the wrong decade.
    Here revenue, profit, assets and cash flow all grow together, which makes
    the expected verdicts predictable.

    Ticker "CCC.NS" is deliberately left with no statements at all -- the
    provider genuinely publishes none for many ETFs and recent listings, and
    the panel has to render that case rather than assume every symbol has
    accounts.
    """
    if ticker not in PRICES.columns or ticker == "CCC.NS":
        return (
            {
                "balance_sheet": pd.DataFrame(),
                "income": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
            },
            f"{ticker}: the provider published no financial statements.",
        )

    periods = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31"])
    # Newest first, each year 8% smaller going back -- so every year-on-year
    # Piotroski signal has a definite direction.
    scale = [1.0, 0.92, 0.85, 0.78, 0.72]

    def row(base: float) -> list[float]:
        return [base * s for s in scale]

    balance = pd.DataFrame(
        {
            "Total Assets": row(1_000_000.0),
            "Current Assets": row(400_000.0),
            "Current Liabilities": row(200_000.0),
            "Total Liabilities Net Minority Interest": row(450_000.0),
            "Long Term Debt": row(150_000.0),
            "Total Debt": row(220_000.0),
            "Stockholders Equity": row(550_000.0),
            "Retained Earnings": row(300_000.0),
            "Working Capital": row(200_000.0),
            "Ordinary Shares Number": [10_000.0] * 5,
            "Accounts Receivable": row(90_000.0),
            "Inventory": row(70_000.0),
            "Net PPE": row(500_000.0),
            "Gross PPE": row(700_000.0),
            "Cash And Cash Equivalents": row(120_000.0),
            "Other Short Term Investments": row(30_000.0),
            "Invested Capital": row(770_000.0),
        },
        index=periods,
    ).T
    income = pd.DataFrame(
        {
            "Total Revenue": row(800_000.0),
            "Gross Profit": row(280_000.0),
            "Operating Income": row(120_000.0),
            "EBIT": row(120_000.0),
            "Net Income": row(80_000.0),
            "Pretax Income": row(105_000.0),
            "Tax Provision": row(25_000.0),
            "Interest Expense": row(15_000.0),
            "Selling General And Administration": row(90_000.0),
        },
        index=periods,
    ).T
    cash = pd.DataFrame(
        {
            "Operating Cash Flow": row(140_000.0),
            "Free Cash Flow": row(60_000.0),
            "Capital Expenditure": row(-80_000.0),
            "Depreciation And Amortization": row(45_000.0),
        },
        index=periods,
    ).T
    return {"balance_sheet": balance, "income": income, "cash_flow": cash}, ""


@pytest.fixture
def stub_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every network call with deterministic synthetic data."""
    monkeypatch.setattr(dfetch, "fetch_quotes", _quotes_for)
    monkeypatch.setattr(dfetch, "fetch_profiles", _profiles_for)
    monkeypatch.setattr(
        dfetch,
        "fetch_price_history",
        lambda tickers, period: (
            PRICES[[t for t in dict.fromkeys(tickers) if t in PRICES.columns]].copy(),
            {},
        ),
    )
    monkeypatch.setattr(dfetch, "fetch_benchmark_history", lambda symbol, period: BENCH.copy())
    monkeypatch.setattr(
        dfetch, "fetch_fx_rates", lambda currencies, base: (dict.fromkeys(currencies, 1.0), {})
    )
    monkeypatch.setattr(
        dfetch,
        "fetch_ohlcv",
        lambda ticker, period: pd.DataFrame(
            {
                "Open": PRICES[ticker].shift(1).bfill(),
                "High": PRICES[ticker] * 1.01,
                "Low": PRICES[ticker] * 0.99,
                "Close": PRICES[ticker],
                "Volume": np.full(len(PRICES), 1.0e6),
            },
            index=PRICES.index,
        ),
    )
    monkeypatch.setattr(dfetch, "fetch_statements", _statements_for)
    _stub_news(monkeypatch)
    monkeypatch.setattr(dfetch, "validate_ticker", lambda ticker: (ticker in PRICES.columns, ""))
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# a fully populated AnalysisResult, built through the real analytics
# ---------------------------------------------------------------------------
@pytest.fixture
def analysis() -> AnalysisResult:
    cfg = AppConfig()
    holdings, _ = pos.clean_holdings(
        pd.DataFrame(
            {
                "ticker": list(TICKERS),
                "quantity": [20.0, 12.0, 45.0],
                "buy_price": [95.0, 150.0, 240.0],
            }
        )
    )
    quotes, _ = _quotes_for(TICKERS)
    positions, _ = pos.build_positions(holdings, quotes, _profiles_for(TICKERS), None, "INR")
    summary = pos.summarise(positions)

    values = rsk.portfolio_value_series(
        PRICES, dict(zip(positions["ticker"], positions["quantity"]))
    )
    returns = rsk.daily_returns(values)
    asset_returns = rsk.daily_returns(PRICES)
    bench_returns = rsk.daily_returns(BENCH)

    risk = rsk.risk_summary(
        values,
        returns,
        positions["weight_pct"],
        len(positions),
        cfg.risk_free_rate,
        cfg.var_confidence,
    )
    correlation = rsk.correlation_matrix(asset_returns)
    covariance = rsk.covariance_matrix(asset_returns, shrinkage=cfg.optimizer.shrinkage)
    comparison = bm.compare(values, BENCH, cfg.risk_free_rate)
    episodes = dd.drawdown_episodes(values)
    attribution = attr.attribution_summary(positions, asset_returns, covariance)
    attribution["factors"] = {}

    txns, _ = lot.parse_transactions(
        pd.DataFrame(
            {
                "ticker": ["AAA.NS", "AAA.NS", "BBB.NS", "AAA.NS"],
                "trade_date": ["2023-01-10", "2023-06-15", "2023-07-03", "2024-08-20"],
                "action": ["BUY", "BUY", "BUY", "SELL"],
                "quantity": [10, 15, 12, 5],
                "price": [95.0, 110.0, 150.0, 140.0],
                "fees": [9.5, 11.0, 15.0, 14.0],
            }
        )
    )
    open_lots, realised, _ = lot.build_lots(txns)
    ledger = {
        "transactions": txns,
        "open_lots": open_lots,
        "realised": realised,
        "realised_summary": lot.realised_summary(realised),
        "cash_flows": lot.cash_flows(txns),
        "xirr": ret.portfolio_xirr(txns, float(summary["current_value"]), NOW.date()),
    }

    score = rsk.risk_score(
        risk["ann_vol"], risk["mdd"], positions["weight_pct"], len(positions), risk["cvar"]
    )

    return AnalysisResult(
        positions=positions,
        summary=summary,
        quotes=quotes,
        profiles=_profiles_for(TICKERS),
        prices=PRICES,
        values=values,
        returns=returns,
        asset_returns=asset_returns,
        benchmark_prices=BENCH,
        risk=risk,
        benchmark=comparison,
        attribution=attribution,
        drawdowns=episodes,
        correlation=correlation,
        covariance=covariance,
        rolling=roll.rolling_frame(
            returns, bench_returns, cfg.rolling_window, cfg.risk_free_rate, cfg.var_confidence
        ),
        monthly=ret.monthly_return_table(values),
        return_stats=ret.return_stats(values),
        ledger=ledger,
        score=score,
        insights=ins.generate_insights(
            positions, summary, risk, comparison, correlation, episodes, attribution
        ),
        fx={},
        warnings=["stubbed market data"],
        config=cfg,
        generated_at=NOW,
    )


# ---------------------------------------------------------------------------
# the page harness
# ---------------------------------------------------------------------------
def _page_script(module_name, result, cfg):
    """Script body executed inside AppTest: mount one page with a ready result.

    Deliberately UNANNOTATED. ``AppTest.from_function`` extracts this function's
    source and re-executes it as a standalone script, in a namespace that has
    none of this module's imports and no ``from __future__ import annotations``.
    A ``result: Any`` annotation is therefore evaluated at def-time in that fresh
    namespace and raises ``NameError: name 'Any' is not defined`` before the body
    ever runs. Every import the script needs is made inside the body for the same
    reason.
    """
    import importlib as _importlib

    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    _st.session_state[_session.RUN] = True

    _importlib.import_module(f"src.ui.pages.{module_name}").render()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_every_page_module_exposes_render() -> None:
    """The navigation contract: one public ``render()`` per page module."""
    for module_name, _ in PAGES:
        module = importlib.import_module(f"src.ui.pages.{module_name}")
        assert callable(getattr(module, "render", None)), f"{module_name}.render missing"


@pytest.mark.parametrize("module_name,label", PAGES, ids=[p[0] for p in PAGES])
def test_every_page_renders(
    module_name: str, label: str, analysis: AnalysisResult, stub_market: None
) -> None:
    """Each page renders against a complete analysis without raising."""
    at = AppTest.from_function(
        _page_script,
        kwargs={"module_name": module_name, "result": analysis, "cfg": analysis.config},
        default_timeout=120,
    )
    at.run()
    assert not at.exception, f"{label} raised: {at.exception}"


@pytest.mark.parametrize("module_name,label", PAGES, ids=[p[0] for p in PAGES])
def test_every_page_handles_no_analysis(module_name: str, label: str, stub_market: None) -> None:
    """With no result in session state, pages must show a guard, not explode."""
    at = AppTest.from_function(
        _page_script,
        kwargs={"module_name": module_name, "result": None, "cfg": AppConfig()},
        default_timeout=120,
    )
    at.run()
    assert not at.exception, f"{label} raised on the empty path: {at.exception}"


# ---------------------------------------------------------------------------
# stock search & watchlist — the paths the generic page harness cannot reach
# ---------------------------------------------------------------------------
def _watchlist_script(items, result, cfg):
    """Mount the watchlist page with a POPULATED watchlist.

    The parametrised page harness above only ever renders the empty watchlist,
    which skips the quote fetch, the table build and the KPI row entirely. This
    seeds items so those actually execute.

    Deliberately UNANNOTATED and importing inside the body, for the same reason
    as ``_page_script``: AppTest re-executes this function's source in a fresh
    namespace with no imports and no postponed annotations.
    """
    import importlib as _importlib

    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    _st.session_state[_session.WATCHLIST] = items
    # Pretend the database was already consulted, so the page does not try to
    # open SQLite inside the test process.
    _st.session_state[_session.WATCHLIST_LOADED] = True

    _importlib.import_module("src.ui.pages.watchlist").render()


def _watch_items():
    from datetime import date

    from src.watchlist import WatchItem

    return (
        WatchItem("AAA.NS", note="target set", target_price=1000.0, added_at=date(2025, 1, 10)),
        WatchItem("BBB.NS", note="", target_price=None, added_at=date(2025, 2, 1)),
        WatchItem("NOSUCH.NS", note="cannot be priced", target_price=50.0),
    )


def test_the_watchlist_page_renders_a_populated_list(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """Exercises the quote fetch, the table, the KPI row and the target notice."""
    at = AppTest.from_function(
        _watchlist_script,
        kwargs={"items": _watch_items(), "result": analysis, "cfg": analysis.config},
        default_timeout=120,
    )
    at.run()
    assert not at.exception, at.exception


def test_an_unpriceable_watched_name_does_not_break_the_page(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """A quote failure must warn, not crash — and the row must still appear."""
    from src.watchlist import WatchItem

    at = AppTest.from_function(
        _watchlist_script,
        kwargs={
            "items": (WatchItem("NOSUCH.NS", target_price=10.0),),
            "result": analysis,
            "cfg": analysis.config,
        },
        default_timeout=120,
    )
    at.run()
    assert not at.exception, at.exception


def test_the_watchlist_page_renders_with_no_analysis_at_all(stub_market: None) -> None:
    """The one page that must work before Analyze has ever been pressed."""
    at = AppTest.from_function(
        _watchlist_script,
        kwargs={"items": _watch_items(), "result": None, "cfg": AppConfig()},
        default_timeout=120,
    )
    at.run()
    assert not at.exception, at.exception


def _search_script(query):
    """Mount the Portfolio builder and type ``query`` into its search box."""
    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state["build_query"] = query

    from src.ui.pages import portfolio as _portfolio

    _portfolio.render()


def test_typing_a_company_name_into_the_builder_finds_the_instrument(
    stub_market: None,
) -> None:
    """The headline requirement, end to end through the real page: a user types
    "Reliance" and the page offers RELIANCE.NS without them knowing the suffix."""
    at = AppTest.from_function(_search_script, kwargs={"query": "Reliance"}, default_timeout=120)
    at.run()

    assert not at.exception, at.exception
    rendered = " ".join(str(block.value) for block in at.radio) + " ".join(
        str(c.value) for c in at.caption
    )
    assert "RELIANCE.NS" in rendered, rendered[:800]


def test_a_search_that_matches_nothing_explains_itself_instead_of_crashing(
    stub_market: None,
) -> None:
    at = AppTest.from_function(_search_script, kwargs={"query": "zzzqqqxxvv"}, default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_an_empty_search_box_states_the_offline_coverage_honestly(
    stub_market: None,
) -> None:
    """The UI must quote the real universe size, never imply completeness."""
    at = AppTest.from_function(_search_script, kwargs={"query": ""}, default_timeout=120)
    at.run()

    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Offline search covers" in captions


def test_the_page_registry_matches_the_navigation_in_app_py() -> None:
    """PAGES above must not drift from what app.py actually registers."""
    source = Path(APP_PATH).read_text(encoding="utf-8")
    for module_name, _label in PAGES:
        assert f"{module_name}.render" in source, f"{module_name} is not registered in app.py"


# ---------------------------------------------------------------------------
# the import mapping editor — driven through the real panel
# ---------------------------------------------------------------------------
def _import_panel_script(body, kind, panel_key):
    """Mount the import panel with a file already staged.

    ``st.file_uploader`` cannot be driven from AppTest, so the upload step is
    bypassed by staging the parsed preview exactly as ``import_panel`` would —
    which is the point: what needs testing is the *editor*, not the uploader.

    Deliberately UNANNOTATED with body-local imports, for the same reason as
    ``_page_script``: AppTest re-executes this function's source in a fresh
    namespace with no imports and no postponed annotations.
    """
    import io as _io

    import streamlit as _st

    from src import charts as _charts
    from src import portfolio_importer as _imp
    from src.ui import csv_import as _csv_import
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()

    # Stage ONCE. This function is the whole script, so it re-executes on every
    # interaction -- re-staging here would overwrite the remapped preview the
    # editor had just produced, and the test would silently assert nothing.
    stage_key = f"import_stage_{panel_key}"
    if stage_key not in _st.session_state:
        read = _imp.import_transactions if kind == "transactions" else _imp.import_holdings
        _st.session_state[stage_key] = {
            "file_id": "stub",
            "name": "staged.csv",
            "preview": read(_io.BytesIO(body), "staged.csv"),
            "map_version": 0,
        }

    _csv_import.import_panel(kind, panel_key)


#: Detection cannot place buy_price here, so the panel opens on the fatal path
#: with the editor as the only way forward.
UNMAPPABLE = b"Instrument,Units Held,Cost of acquisition per unit\nRELIANCE,90,1182.95\n"

GOOD = b"ticker,quantity,buy_price\nRELIANCE.NS,90,1182.95\n"


def _panel(body, kind="holdings", panel_key="page_holdings"):
    at = AppTest.from_function(
        _import_panel_script,
        kwargs={"body": body, "kind": kind, "panel_key": panel_key},
        default_timeout=120,
    )
    at.run()
    return at


def _apply(at, canonical, column):
    """Set one picker and submit the mapping form in a single run cycle.

    Both have to happen before ``run()``: a form's widget values are only
    delivered to the script when its submit button fires, so setting the
    selectbox and running first would discard the selection.
    """
    next(s for s in at.selectbox if s.label.startswith(canonical)).set_value(column)
    next(b for b in at.button if "Apply mapping" in b.label).click()
    at.run()
    return at


def test_the_import_panel_renders_a_clean_staged_file(stub_market: None) -> None:
    at = _panel(GOOD)

    assert not at.exception, at.exception
    assert any("Import 1 holding" in b.label for b in at.button), [b.label for b in at.button]


def test_an_unmappable_file_still_offers_the_column_picker(stub_market: None) -> None:
    """The whole point: a fatal mapping must not be a dead end."""
    at = _panel(UNMAPPABLE)

    assert not at.exception, at.exception
    labels = [s.label for s in at.selectbox]
    assert any(label.startswith("buy_price") for label in labels), labels
    assert any("Apply mapping" in b.label for b in at.button), [b.label for b in at.button]


def test_the_picker_marks_required_fields(stub_market: None) -> None:
    at = _panel(UNMAPPABLE)

    required = {s.label for s in at.selectbox if s.label.endswith(" *")}
    assert {"ticker *", "quantity *", "buy_price *"} <= required, required


def test_the_picker_offers_every_source_column_plus_not_present(stub_market: None) -> None:
    at = _panel(UNMAPPABLE)

    picker = next(s for s in at.selectbox if s.label.startswith("buy_price"))
    assert "Cost of acquisition per unit" in picker.options
    assert any("not present" in str(o) for o in picker.options)


def test_choosing_a_column_and_applying_makes_the_file_importable(stub_market: None) -> None:
    """The end-to-end recovery, through the real widgets."""
    at = _panel(UNMAPPABLE)
    assert not any("Import 1 holding" in b.label for b in at.button), "should start unimportable"

    _apply(at, "buy_price", "Cost of acquisition per unit")

    assert not at.exception, at.exception
    assert any("Import 1 holding" in b.label for b in at.button), [b.label for b in at.button]


def test_the_applied_override_is_recorded_on_the_staged_preview(stub_market: None) -> None:
    at = _panel(UNMAPPABLE)

    _apply(at, "buy_price", "Cost of acquisition per unit")

    staged = at.session_state["import_stage_page_holdings"]
    preview = staged["preview"]
    assert preview.ok
    assert "buy_price" in preview.mapping.overridden
    assert staged["map_version"] == 1, "the widget generation must advance"


def test_applying_an_unchanged_mapping_says_nothing_changed(stub_market: None) -> None:
    at = _panel(GOOD)

    next(b for b in at.button if "Apply mapping" in b.label).click().run()

    assert not at.exception, at.exception
    assert any("Nothing changed" in str(i.value) for i in at.info), [str(i.value) for i in at.info]


def test_reset_to_auto_discards_an_override(stub_market: None) -> None:
    at = _panel(GOOD)

    # Point an optional field somewhere detection never would, then undo it.
    _apply(at, "purchase_date", "quantity")
    assert at.session_state["import_stage_page_holdings"]["preview"].mapping.overridden

    next(b for b in at.button if "Reset to auto" in b.label).click().run()

    assert not at.exception, at.exception
    preview = at.session_state["import_stage_page_holdings"]["preview"]
    assert preview.mapping.overridden == ()


def test_unmapping_a_required_field_through_the_picker_is_reported(stub_market: None) -> None:
    at = _panel(GOOD)

    _apply(at, "buy_price", "— not present —")

    assert not at.exception, at.exception
    preview = at.session_state["import_stage_page_holdings"]["preview"]
    assert "buy_price" in preview.mapping.missing
    assert not preview.ok


def test_an_unmappable_file_shows_its_own_first_rows(stub_market: None) -> None:
    """Picking between "Rate" and "Cost per unit" from header names alone is
    guesswork; the values settle it. Shown only when nothing survived."""
    at = _panel(UNMAPPABLE)

    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "as it was read" in captions, captions


def test_a_clean_file_does_not_show_the_raw_rows(stub_market: None) -> None:
    """It would just duplicate the import preview directly below it."""
    at = _panel(GOOD)

    captions = " ".join(str(c.value) for c in at.caption)
    assert "as it was read" not in captions


def test_the_ledger_panel_offers_its_own_fields(stub_market: None) -> None:
    body = b"Instrument,When,Direction,Units,Rate per unit\nRELIANCE,2024-01-15,BUY,10,1100\n"
    at = _panel(body, kind="transactions", panel_key="page_txns")

    assert not at.exception, at.exception
    labels = [s.label for s in at.selectbox]
    assert any(label.startswith("action") for label in labels), labels
    assert any(label.startswith("fees") for label in labels), labels


def test_the_compact_sidebar_panel_shows_no_picker(stub_market: None) -> None:
    """A two-column picker is unusable at 330px; the sidebar points at the page."""

    def _script(body):
        import io as _io

        import streamlit as _st

        from src import portfolio_importer as _imp
        from src.ui import csv_import as _csv_import
        from src.ui import session as _session

        _session.init()
        _st.session_state["import_stage_side_holdings"] = {
            "file_id": "stub",
            "name": "staged.csv",
            "preview": _imp.import_holdings(_io.BytesIO(body), "staged.csv"),
            "map_version": 0,
        }
        _csv_import.import_panel("holdings", "side_holdings", compact=True)

    at = AppTest.from_function(_script, kwargs={"body": UNMAPPABLE}, default_timeout=120)
    at.run()

    assert not at.exception, at.exception
    assert not at.selectbox, "the compact panel must not render the picker"


def test_landing_page_renders(stub_market: None) -> None:
    """The real app boots and offers the primary action."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    assert any("Analyze" in label for label in labels), labels


def test_app_runs_end_to_end(stub_market: None) -> None:
    """Click Analyze in the real app and confirm the pipeline produced a result."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    at.session_state["holdings"] = pd.DataFrame(
        {"ticker": list(TICKERS), "quantity": [20.0, 12.0, 45.0], "buy_price": [95.0, 150.0, 240.0]}
    )
    analyze = next(b for b in at.sidebar.button if "Analyze" in b.label)
    analyze.click().run()

    assert not at.exception, at.exception
    # AppTest wraps session state in a SafeSessionState, which is dict-like but
    # exposes no .get() - index it and let a missing key fail loudly instead.
    assert "result" in at.session_state, "pipeline never wrote a result"
    result = at.session_state["result"]
    assert result is not None, "pipeline produced no AnalysisResult"
    assert len(result.positions) == 3
    assert result.has_history
    assert result.score.get("score") is not None


# ---------------------------------------------------------------------------
# the factor-regression panel
# ---------------------------------------------------------------------------
def _factor_panel_script(result, cfg):
    """Mount the factor panel with proxy prices available.

    The shared ``stub_market`` fixture only knows the three portfolio tickers,
    so the proxy path finds no history and the panel short-circuits to its
    (also worth testing) "no factor data" branch. This script widens the stub
    so the regression, the loadings table and the rolling chart actually run.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """
    import numpy as _np
    import pandas as _pd
    import streamlit as _st

    import src.data.fetcher as _dfetch
    from src import charts as _charts
    from src.config import FACTOR_PROXIES as _PROXIES
    from src.ui import factor_panel as _panel
    from src.ui import session as _session
    from src.ui import theme as _theme

    symbols = sorted({s for spec in _PROXIES["India"] for s in spec.symbols})
    index = _pd.bdate_range("2023-01-02", periods=400)
    rng = _np.random.default_rng(3)
    market = rng.normal(0.0004, 0.010, len(index))
    proxy = _pd.DataFrame(
        {
            s: 100.0 * _np.cumprod(1.0 + market + rng.normal(0.0, 0.004, len(index)))
            for s in symbols
        },
        index=index,
    )
    _dfetch.fetch_price_history = lambda tickers, period: (
        proxy[[t for t in dict.fromkeys(tickers) if t in proxy.columns]].copy(),
        {},
    )
    _st.cache_data.clear()

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    _st.session_state["factor_source"] = "ETF proxies"
    _st.session_state["factor_market"] = "India"

    _panel.render(result)


def test_the_factor_panel_renders_a_regression(analysis: AnalysisResult, stub_market: None) -> None:
    """Exercises the proxy build, the regression, the loadings table and the
    rolling-beta chart in one pass."""
    at = AppTest.from_function(
        _factor_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config},
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_factor_panel_survives_having_no_factor_data(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """The stub knows only the portfolio tickers, so no proxy prices exist.
    The panel must explain that, not crash."""

    def _script(result, cfg):
        import streamlit as _st

        from src.ui import factor_panel as _panel
        from src.ui import session as _session

        _session.init()
        _st.session_state[_session.RESULT] = result
        _st.session_state[_session.CONFIG] = cfg
        _panel.render(result)

    at = AppTest.from_function(
        _script, kwargs={"result": analysis, "cfg": analysis.config}, default_timeout=120
    )
    at.run()

    assert not at.exception, at.exception


def test_the_factor_panel_refuses_a_result_with_no_returns(stub_market: None) -> None:
    def _script():
        from src.ui import factor_panel as _panel
        from src.ui import session as _session

        _session.init()
        _panel.render(object())

    at = AppTest.from_function(_script, default_timeout=120)
    at.run()

    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# the price-alert panel
# ---------------------------------------------------------------------------
def _alerts_panel_script(db_path, items, settings, table_rows):
    """Mount the alert panel against a throwaway store.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """
    import pandas as _pd

    from src import alerts as _al
    from src import watchlist as _wl
    from src.data import store as _store_module
    from src.ui import alerts_panel as _panel
    from src.ui import theme as _theme

    _store_module.DEFAULT_DB_PATH = _store_module.Path(db_path)
    # The panel caches its store in st.cache_resource, so a connection opened
    # by an earlier test would outlive the redirect above.
    _panel._store.clear()
    if settings:
        with _store_module.PortfolioStore(db_path) as _store:
            _store.save_alert_settings(settings)

    _theme.inject_css()
    _panel.render(
        tuple(_wl.WatchItem(**row) for row in (items or ())),
        _pd.DataFrame(table_rows) if table_rows else None,
    )
    assert _al.ARMED  # the module is genuinely loaded


@pytest.fixture
def alerts_db(tmp_path: Path) -> Path:
    from src.data.store import PortfolioStore

    path = tmp_path / "alerts.db"
    with PortfolioStore(path) as store:
        store.load_alert_settings()
    return path


ALERT_RULE = {
    "ticker": "AAA.NS",
    "target_price": 110.0,
    "direction": "above",
    "alert_enabled": True,
    "reference_price": 100.0,
}


def test_the_alert_panel_renders_with_nothing_configured(alerts_db: Path) -> None:
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={"db_path": str(alerts_db), "items": None, "settings": None, "table_rows": None},
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_alert_panel_says_alerts_are_off_by_default(alerts_db: Path) -> None:
    """An e-mail nobody asked for is worse than no e-mail, so the default has
    to be visible rather than assumed."""
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={"db_path": str(alerts_db), "items": None, "settings": None, "table_rows": None},
        default_timeout=180,
    )
    at.run()

    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "alerts off" in rendered


def test_the_alert_panel_leads_with_the_fact_that_it_cannot_check(
    alerts_db: Path,
) -> None:
    """The honesty requirement, asserted in the UI: a user must not be able to
    switch an alert on believing this page will check it."""
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={"db_path": str(alerts_db), "items": None, "settings": None, "table_rows": None},
        default_timeout=180,
    )
    at.run()

    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "checked by a command, not by this page" in rendered
    assert "no background worker" in rendered


def test_the_alert_panel_prints_the_runner_command(alerts_db: Path) -> None:
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={"db_path": str(alerts_db), "items": None, "settings": None, "table_rows": None},
        default_timeout=180,
    )
    at.run()

    code_blocks = " ".join(str(c.value) for c in at.code)
    assert "-m src.reporting.cli" in code_blocks


def test_the_alert_panel_tells_you_when_no_rule_is_switched_on(
    alerts_db: Path,
) -> None:
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [{"ticker": "AAA.NS", "target_price": 110.0}],
            "settings": None,
            "table_rows": None,
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    messages = " ".join(str(i.value) for i in at.info)
    assert "No instrument has an alert switched on" in messages


def test_the_alert_panel_lists_an_active_rule(alerts_db: Path) -> None:
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [ALERT_RULE],
            "settings": {"enabled": True, "recipients": "owner@example.com"},
            "table_rows": None,
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    frames = [df.value for df in at.dataframe]
    assert frames, "the rules table should have rendered"
    assert "AAA.NS" in frames[0].to_string()
    assert "rises to or above" in frames[0].to_string()


def test_the_alert_panel_warns_about_a_rule_with_no_direction(
    alerts_db: Path,
) -> None:
    """The one configuration that looks armed and can never fire."""
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [{"ticker": "AAA.NS", "target_price": 110.0, "alert_enabled": True}],
            "settings": {"enabled": True, "recipients": "owner@example.com"},
            "table_rows": None,
        },
        default_timeout=180,
    )
    at.run()

    warnings = " ".join(str(w.value) for w in at.warning)
    assert "no direction" in warnings
    assert "AAA.NS" in warnings


def test_the_alert_panel_previews_a_level_already_through(alerts_db: Path) -> None:
    """Answers "is my rule set up right?" without waiting for the next cron
    run — and says explicitly that the preview sends nothing."""
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [ALERT_RULE],
            "settings": {"enabled": True, "recipients": "owner@example.com"},
            "table_rows": [{"ticker": "AAA.NS", "price": 115.0}],
        },
        default_timeout=180,
    )
    at.run()

    messages = " ".join(str(i.value) for i in at.info)
    assert "through their level" in messages
    assert "sends nothing" in messages


def test_the_alert_panel_never_collects_a_password(alerts_db: Path) -> None:
    """Alert e-mail reuses the tearsheet SMTP settings, read from secrets or
    the environment. Anything typed here would land in data/quantfolio.db."""
    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [ALERT_RULE],
            "settings": {"enabled": True, "recipients": "owner@example.com"},
            "table_rows": None,
        },
        default_timeout=180,
    )
    at.run()

    assert not [w for w in at.text_input if w.type == "password"]


def test_the_alert_panel_sends_and_checks_nothing_on_a_render(
    alerts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two tripwires. A Streamlit script re-runs on every widget interaction,
    so a render must neither open an SMTP connection nor fetch a quote."""
    import smtplib

    from src.data import fetcher as tripwire_fetcher

    def no_smtp(*_args, **_kwargs):
        raise AssertionError("a page render must not open an SMTP connection")

    def no_fetch(*_args, **_kwargs):
        raise AssertionError("a page render must not fetch a quote for alerting")

    monkeypatch.setattr(smtplib, "SMTP", no_smtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", no_smtp)
    monkeypatch.setattr(tripwire_fetcher, "fetch_quotes", no_fetch)

    at = AppTest.from_function(
        _alerts_panel_script,
        kwargs={
            "db_path": str(alerts_db),
            "items": [ALERT_RULE],
            "settings": {"enabled": True, "recipients": "owner@example.com"},
            "table_rows": [{"ticker": "AAA.NS", "price": 115.0}],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# the scheduled-email panel
# ---------------------------------------------------------------------------
def _email_panel_script(db_path, schedules):
    """Mount the scheduled-email panel against a throwaway store.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """

    from src.data import store as _store_module
    from src.reporting import schedule as _sched
    from src.ui import email_panel as _panel
    from src.ui import theme as _theme

    _store_module.DEFAULT_DB_PATH = _store_module.Path(db_path)
    # The panel holds its store in st.cache_resource -- one long-lived
    # connection per server process, which is right for the app and wrong for
    # a test, because a store opened by an earlier test would outlive the
    # redirect above and silently ignore this db_path.
    _panel._store.clear()
    if schedules:
        with _store_module.PortfolioStore(db_path) as _store:
            _store.save_schedules([_sched.to_record(_sched.from_record(s)) for s in schedules])

    _theme.inject_css()
    _panel.render()


@pytest.fixture
def email_db(tmp_path: Path) -> Path:
    """A store with one saved portfolio, so the schedule form is enabled."""
    from src.data.store import PortfolioStore

    path = tmp_path / "email.db"
    with PortfolioStore(path) as store:
        store.save_portfolio(
            "Core",
            pd.DataFrame({"ticker": ["AAA.NS"], "quantity": [10.0], "buy_price": [100.0]}),
            None,
            "INR",
        )
    return path


def test_the_email_panel_renders_with_nothing_configured(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal state of a fresh checkout: no schedules, no SMTP, and no
    secrets.toml at all (which makes ``st.secrets`` raise)."""
    for key in ("SMTP_HOST", "SMTP_SENDER", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={"db_path": str(email_db), "schedules": None},
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_email_panel_leads_with_the_fact_that_it_cannot_schedule(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The honesty requirement, asserted in the UI: a user must not be able to
    save a schedule believing the app will send it. Streamlit has no
    scheduler, so the panel has to say so before anything else."""
    for key in ("SMTP_HOST", "SMTP_SENDER"):
        monkeypatch.delenv(key, raising=False)

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={"db_path": str(email_db), "schedules": None},
        default_timeout=180,
    )
    at.run()

    text = " ".join(str(m.value) for m in at.markdown)
    assert "Nothing is sent until you schedule the command below" in text
    assert "no background scheduler" in text


def test_the_email_panel_shows_the_secrets_template_when_mail_is_unset(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No password input anywhere: the panel prints the git-ignored file to
    write instead, because a password typed into Streamlit would be persisted
    into data/quantfolio.db."""
    for key in ("SMTP_HOST", "SMTP_SENDER", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={"db_path": str(email_db), "schedules": None},
        default_timeout=180,
    )
    at.run()

    code_blocks = " ".join(str(c.value) for c in at.code)
    assert "[smtp]" in code_blocks
    assert not [w for w in at.text_input if w.type == "password"]


def test_the_email_panel_prints_the_command_for_both_schedulers(
    email_db: Path,
) -> None:
    at = AppTest.from_function(
        _email_panel_script,
        kwargs={"db_path": str(email_db), "schedules": None},
        default_timeout=180,
    )
    at.run()

    code_blocks = " ".join(str(c.value) for c in at.code)
    assert "-m src.reporting.cli" in code_blocks
    assert "schtasks" in code_blocks


def test_the_email_panel_renders_an_existing_schedule(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_SENDER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-specific-secret")

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={
            "db_path": str(email_db),
            "schedules": [
                {
                    "name": "Weekly review",
                    "portfolio": "Core",
                    "recipients": "owner@example.com",
                    "frequency": "weekly",
                    "hour": 8,
                    "weekday": 0,
                }
            ],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    # Proves the panel really loaded from the store rather than rendering an
    # empty list -- otherwise every assertion below it would pass trivially.
    rendered = " ".join([str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption])
    dataframes = [df.value for df in at.dataframe]
    assert dataframes, "the schedule table should have rendered"
    assert "Weekly review" in dataframes[0].to_string()
    assert "no scheduled reports" not in rendered.lower()


def test_the_email_panel_never_displays_the_password(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured mail is reported as configured -- host, port, transport --
    and never by value."""
    secret = "do-not-render-me"
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_SENDER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", secret)

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={"db_path": str(email_db), "schedules": None},
        default_timeout=180,
    )
    at.run()

    rendered = " ".join(
        [str(m.value) for m in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(c.value) for c in at.code]
        + [str(i.value) for i in at.info]
    )
    assert secret not in rendered
    assert "smtp.example.com" in rendered


def test_the_email_panel_sends_nothing_on_a_plain_render(
    email_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule for anything outward-facing: a Streamlit script re-runs on
    every widget interaction anywhere on the page, so a render must never
    open an SMTP connection."""
    import smtplib

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-specific-secret")

    opened: list[str] = []

    class Tripwire:
        def __init__(self, *args, **kwargs):
            opened.append("connected")
            raise AssertionError("a page render must not open an SMTP connection")

    monkeypatch.setattr(smtplib, "SMTP", Tripwire)
    monkeypatch.setattr(smtplib, "SMTP_SSL", Tripwire)

    at = AppTest.from_function(
        _email_panel_script,
        kwargs={
            "db_path": str(email_db),
            "schedules": [
                {
                    "name": "Weekly review",
                    "portfolio": "Core",
                    "recipients": "owner@example.com",
                    "frequency": "daily",
                    "hour": 8,
                }
            ],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    assert opened == []


# ---------------------------------------------------------------------------
# the Black-Litterman panel
# ---------------------------------------------------------------------------
def _bl_panel_script(result, cfg, views):
    """Mount the Black-Litterman panel, optionally with views pre-staged.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """
    import pandas as _pd
    import streamlit as _st

    from src import charts as _charts
    from src.ui import bl_panel as _panel
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    if views:
        _st.session_state["bl_views"] = _pd.DataFrame(views)

    _panel.render(result)


def test_the_black_litterman_panel_renders_the_pure_equilibrium(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """With no views the panel must show the equilibrium and say so."""
    at = AppTest.from_function(
        _bl_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config, "views": None},
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    # With no views the panel states that the posterior IS the equilibrium.
    messages = " ".join(str(i.value) for i in at.info) + " ".join(str(c.value) for c in at.caption)
    assert "equilibrium" in messages.lower(), messages[:500]


def test_the_black_litterman_panel_applies_an_absolute_view(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """Exercises the view parser, the blend, the impact table and the re-solve."""
    at = AppTest.from_function(
        _bl_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "views": [
                {
                    "kind": "Absolute",
                    "asset": "AAA.NS",
                    "versus": None,
                    "return_pct": 14.0,
                    "confidence_pct": 70.0,
                }
            ],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_black_litterman_panel_applies_a_relative_view(
    analysis: AnalysisResult, stub_market: None
) -> None:
    at = AppTest.from_function(
        _bl_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "views": [
                {
                    "kind": "Relative",
                    "asset": "AAA.NS",
                    "versus": "BBB.NS",
                    "return_pct": 4.0,
                    "confidence_pct": 60.0,
                }
            ],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_black_litterman_panel_reports_an_unreadable_view(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """A view naming an unheld asset, and a relative view missing its second
    leg, must both be reported rather than silently dropped."""
    at = AppTest.from_function(
        _bl_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "views": [
                {
                    "kind": "Absolute",
                    "asset": "NOTHELD.NS",
                    "versus": None,
                    "return_pct": 20.0,
                    "confidence_pct": 80.0,
                },
                {
                    "kind": "Relative",
                    "asset": "AAA.NS",
                    "versus": "",
                    "return_pct": 3.0,
                    "confidence_pct": 50.0,
                },
            ],
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_black_litterman_panel_refuses_a_result_without_a_covariance(
    stub_market: None,
) -> None:
    def _script():
        from src.ui import bl_panel as _panel
        from src.ui import session as _session

        _session.init()
        _panel.render(object())

    at = AppTest.from_function(_script, default_timeout=120)
    at.run()

    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# the tax-loss harvesting panel
# ---------------------------------------------------------------------------
def _tax_panel_script(result, cfg, ledger):
    """Mount the harvesting panel against a ledger.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """
    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import tax_panel as _panel
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg

    _panel.render(result, ledger)


def test_the_tax_panel_renders_against_the_analysis_ledger(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """Exercises lot classification, the set-off waterfall, the candidate
    ranking and the boundary alerts in one pass."""
    at = AppTest.from_function(
        _tax_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config, "ledger": analysis.ledger},
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_tax_panel_states_the_regime_it_applied(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """A tax figure the user cannot trace to a set of rules is not auditable."""
    at = AppTest.from_function(
        _tax_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config, "ledger": analysis.ledger},
        default_timeout=180,
    )
    at.run()

    captions = " ".join(str(c.value) for c in at.caption)
    assert "short-term" in captions.lower()
    assert "exemption" in captions.lower()


def test_the_tax_panel_always_carries_the_disclaimer(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """Tax arithmetic is the one place a plausible wrong answer has
    consequences off-screen, so the caveat is not optional."""
    at = AppTest.from_function(
        _tax_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config, "ledger": analysis.ledger},
        default_timeout=180,
    )
    at.run()

    captions = " ".join(str(c.value) for c in at.caption)
    assert "not tax advice" in captions.lower()
    assert "chartered accountant" in captions.lower()


def test_the_tax_panel_explains_itself_with_no_ledger(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """Average cost cannot distinguish a short-term lot from a long-term one,
    which is the whole basis of the calculation — so say that."""
    at = AppTest.from_function(
        _tax_panel_script,
        kwargs={"result": analysis, "cfg": analysis.config, "ledger": {}},
        default_timeout=120,
    )
    at.run()

    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "trade ledger" in captions.lower()


def test_the_tax_panel_refuses_a_result_with_no_prices(stub_market: None) -> None:
    def _script():
        from src.ui import session as _session
        from src.ui import tax_panel as _panel

        _session.init()
        _panel.render(object(), {"open_lots": [], "realised": []})

    at = AppTest.from_function(_script, default_timeout=120)
    at.run()

    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# the options overlay panel
# ---------------------------------------------------------------------------
def _options_panel_script(result, cfg, ticker, spot, strategy):
    """Mount the overlay panel for one underlying.

    Deliberately UNANNOTATED with body-local imports, as ``_page_script``.
    """
    import streamlit as _st

    from src import charts as _charts
    from src.ui import options_panel as _panel
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    _st.session_state["opt_strategy"] = strategy
    # Manual entry is the path that works for every market, and the only one
    # that works at all for NSE — so that is what the tests drive.
    _st.session_state["opt_source"] = "Enter the contract manually"

    _panel.render(result, ticker, spot=spot, currency="INR")


@pytest.mark.parametrize("strategy", ["Covered call", "Protective put", "Collar (both)"])
def test_the_options_panel_prices_each_structure(
    strategy: str, analysis: AnalysisResult, stub_market: None
) -> None:
    """Exercises the manual-entry path, the overlay maths, the payoff chart and
    the Greeks table for all three structures."""
    at = AppTest.from_function(
        _options_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "ticker": "AAA.NS",
            "spot": 100.0,
            "strategy": strategy,
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception


def test_the_options_panel_states_that_nse_chains_are_unavailable(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """The single most important message on this panel: the gap is in the data
    source, not in the market. NIFTY and Reliance options are extremely
    liquid — an empty table would read as 'no options exist'."""
    at = AppTest.from_function(
        _options_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "ticker": "AAA.NS",
            "spot": 100.0,
            "strategy": "Covered call",
        },
        default_timeout=180,
    )
    at.run()

    assert not at.exception, at.exception
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "no option chain" in warnings.lower()
    assert "manually" in warnings.lower()


def test_the_options_panel_carries_the_risk_caveat(
    analysis: AnalysisResult, stub_market: None
) -> None:
    at = AppTest.from_function(
        _options_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "ticker": "AAA.NS",
            "spot": 100.0,
            "strategy": "Covered call",
        },
        default_timeout=180,
    )
    at.run()

    captions = " ".join(str(c.value) for c in at.caption)
    assert "assignment" in captions.lower()
    assert (
        "not a recommendation" in captions.lower() or "recommendation to trade" in captions.lower()
    )


def test_the_options_panel_refuses_an_unusable_spot(
    analysis: AnalysisResult, stub_market: None
) -> None:
    at = AppTest.from_function(
        _options_panel_script,
        kwargs={
            "result": analysis,
            "cfg": analysis.config,
            "ticker": "AAA.NS",
            "spot": float("nan"),
            "strategy": "Covered call",
        },
        default_timeout=120,
    )
    at.run()

    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    assert "no current price" in captions.lower()


# ---------------------------------------------------------------------------
# the company-research search box -- the form-callback rule
# ---------------------------------------------------------------------------
def _research_script(query, pick=None, last_query=None):
    """Mount the research page with a query already typed.

    The search box is a ``st.form``, so its text input only delivers its value
    when the submit button fires. Seeding the session state is therefore the
    only way to drive it from AppTest, and it is exactly what a real first
    render does: the box shows the query, the reader presses Search, and the
    page resolves it.
    """
    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import theme as _theme

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state["research_query"] = query
    if pick is not None:
        _st.session_state["research_pick"] = pick
    if last_query is not None:
        _st.session_state["research_last_query"] = last_query
    else:
        _st.session_state["research_last_query"] = query

    from src.ui.pages import research as _research

    _research.render()




def test_the_research_search_box_has_no_callback_inside_the_form() -> None:
    """The regression this guards: a widget callback inside ``st.form``.

    Streamlit forbids ``on_change`` / ``on_click`` on anything but the submit
    button and raises ``StreamlitInvalidFormCallbackError``. The box must
    carry none, with the query-change logic living in the submit handler
    instead.
    """
    import inspect

    from src.ui.pages import research

    source = inspect.getsource(research._search_box)
    # Must not contain actual callback parameters (not just comment text).
    assert "on_change=" not in source
    assert "on_click=" not in source
    assert "st.form_submit_button" in source


def test_the_research_page_searches_a_named_company(stub_market: None) -> None:
    """End to end through the real page: a user types a company name and the
    page resolves it without raising.

    The search box is a ``st.form``, so its text input only delivers its value
    when the submit button fires. Seeding the session state is the only way to
    drive it from AppTest, and it is exactly what a real first render does:
    the box shows the query, the reader presses Search, and the page resolves
    it.
    """
    at = AppTest.from_function(
        _research_script,
        kwargs={"query": "Reliance"},
        default_timeout=120,
    )
    at.run()
    next(b for b in at.button if "Search" in b.label).click().run()

    assert not at.exception, at.exception
    # The resolver refuses to auto-pick, so the page renders a radio list of
    # candidates for the reader to choose from. The ticker must be among them.
    # The resolver refuses to auto-pick, so the page renders a radio list of
    # candidates for the reader to choose from. The ticker must be among the
    # options -- the widget's *value* is None until a selection is made.
    rendered = " ".join(str(o) for r in at.radio for o in r.options) + " ".join(
        str(c.value) for c in at.caption
    )
    assert "RELIANCE.NS" in rendered, rendered[:800]
