"""The forecast page, driven headlessly against a synthetic market.

``test_app_smoke`` already asserts every page renders without raising. That is
not enough here: the forecast page has a large, load-bearing *failure* path —
a history too short to validate renders an explanation and returns early — and
a page test that only checks "did not raise" passes just as happily when the
success path never executes.

So these tests read what was actually rendered. They exist because the most
likely regression in this page is not a crash; it is the page quietly showing
"no forecast was produced" forever after someone changes a window constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.data.fetcher as dfetch
from src.analytics import positions as pos
from src.analytics import risk as rsk
from src.config import FORECAST, AppConfig
from src.models import AnalysisResult, Quote, SecurityProfile

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

TICKER = "AAA.NS"

#: Long enough for the 252-session lookbacks AND a five-fold purged
#: walk-forward at the default 21-session horizon. The page's whole success
#: path depends on clearing that bar, so the fixture clears it deliberately.
DAYS = 1400


def _prices() -> pd.Series:
    rng = np.random.default_rng(17)
    index = pd.bdate_range("2018-01-02", periods=DAYS)
    return pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.011, DAYS)), index=index, name=TICKER
    )


PRICES = _prices()
NOW = pd.Timestamp("2025-09-01 10:30", tz="UTC").to_pydatetime()


def _ohlcv(ticker: str, period: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": PRICES.shift(1).bfill(),
            "High": PRICES * 1.01,
            "Low": PRICES * 0.99,
            "Close": PRICES,
            "Volume": np.full(len(PRICES), 1.0e6),
        },
        index=PRICES.index,
    )


@pytest.fixture
def stub_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """One synthetic instrument, no network, deterministic."""
    quote = Quote(TICKER, float(PRICES.iloc[-1]), "INR", NOW, NOW, False)
    monkeypatch.setattr(dfetch, "fetch_ohlcv", _ohlcv)
    monkeypatch.setattr(dfetch, "fetch_quotes", lambda tickers: ({TICKER: quote}, {}))
    monkeypatch.setattr(
        dfetch,
        "fetch_profiles",
        lambda tickers: {TICKER: SecurityProfile(TICKER, name="Alpha", currency="INR")},
    )
    st.cache_data.clear()


@pytest.fixture
def analysis() -> AnalysisResult:
    cfg = AppConfig()
    holdings, _ = pos.clean_holdings(
        pd.DataFrame({"ticker": [TICKER], "quantity": [10.0], "buy_price": [95.0]})
    )
    quote = Quote(TICKER, float(PRICES.iloc[-1]), "INR", NOW, NOW, False)
    profiles = {TICKER: SecurityProfile(TICKER, name="Alpha", currency="INR")}
    positions, _ = pos.build_positions(holdings, quote and {TICKER: quote}, profiles, None, "INR")

    frame = PRICES.to_frame()
    values = rsk.portfolio_value_series(frame, {TICKER: 10.0})
    returns = rsk.daily_returns(values)
    return AnalysisResult(
        positions=positions,
        summary=pos.summarise(positions),
        quotes={TICKER: quote},
        profiles=profiles,
        prices=frame,
        values=values,
        returns=returns,
        asset_returns=rsk.daily_returns(frame),
        config=cfg,
        generated_at=NOW,
    )


def _script(result, cfg):
    """Mount the forecast page with a ready analysis. Deliberately unannotated.

    ``AppTest.from_function`` re-executes this function's source in a fresh
    namespace with none of this module's imports and no ``from __future__``,
    so an annotation would raise ``NameError`` before the body ever runs.
    """
    import streamlit as _st

    from src import charts as _charts
    from src.ui import session as _session
    from src.ui import theme as _theme
    from src.ui.pages import forecast as _page

    _theme.inject_css()
    _charts.set_mode("dark")
    _session.init()
    _st.session_state[_session.RESULT] = result
    _st.session_state[_session.CONFIG] = cfg
    _st.session_state[_session.RUN] = True
    _page.render()


def _run(result, cfg) -> AppTest:
    at = AppTest.from_function(_script, kwargs={"result": result, "cfg": cfg}, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at: AppTest) -> str:
    """Everything the page wrote, as one lowercase blob."""
    parts = [element.value for element in at.markdown]
    parts += [element.value for element in at.caption]
    return " ".join(str(p) for p in parts).lower()


# ---------------------------------------------------------------------------
# the success path
# ---------------------------------------------------------------------------
def test_the_page_reaches_a_verdict_rather_than_an_empty_state(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """The failure path must not be what renders on a perfectly good history."""
    body = _text(_run(analysis, analysis.config))

    assert "no forecast was produced" not in body
    assert "qf-verdict" in body, "the headline conclusion must be rendered first"
    assert "random walk" in body, "the benchmark must be named wherever skill is quoted"


def test_the_page_states_the_disclaimer_before_any_number(
    analysis: AnalysisResult, stub_market: None
) -> None:
    body = _text(_run(analysis, analysis.config))
    assert "conditional statistical estimate" in body
    assert "qf-hero" in body


def test_the_forecast_interval_and_the_probability_are_both_shown(
    analysis: AnalysisResult, stub_market: None
) -> None:
    body = _text(_run(analysis, analysis.config))
    assert "point forecast" in body
    assert "interval" in body
    assert "probability the price is higher" in body
    assert "qf-meter" in body


def test_the_validation_evidence_is_rendered_not_just_the_forecast(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """A page that showed only the cone would always look confident."""
    body = _text(_run(analysis, analysis.config))
    assert "directional accuracy" in body
    assert "out-of-sample r" in body
    assert "diebold" in body
    assert "measured skill" in body


def test_the_page_explains_why_it_uses_a_different_window_from_the_sidebar(
    analysis: AnalysisResult, stub_market: None
) -> None:
    at = _run(analysis, analysis.config)
    assert any("window differs" in str(e.label).lower() for e in at.expander)


def test_every_chart_on_the_page_is_a_real_figure(
    analysis: AnalysisResult, stub_market: None
) -> None:
    """No figure may fall through to the chart layer's last-resort placeholder.

    Read off ``proto.spec`` — the serialised figure — rather than ``.value``,
    which for a chart element resolves through widget state that a
    non-selectable chart never registers.
    """
    at_charts = _run(analysis, analysis.config).get("plotly_chart")
    assert len(at_charts) >= 2, "the cone and the diagnostics must both render"
    for element in at_charts:
        spec = str(element.proto.spec)
        assert "Chart unavailable" not in spec, "a builder hit its last-resort guard"
        assert '"data":[' in spec


# ---------------------------------------------------------------------------
# the failure path
# ---------------------------------------------------------------------------
def test_a_history_too_short_to_validate_explains_itself(
    analysis: AnalysisResult, stub_market: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the contract: refuse, and say what was missing."""
    short = _ohlcv(TICKER, "10y").iloc[:300]
    monkeypatch.setattr(dfetch, "fetch_ohlcv", lambda ticker, period: short)
    st.cache_data.clear()

    body = _text(_run(analysis, analysis.config))
    assert "no forecast was produced" in body
    assert str(FORECAST.min_train_rows) in body, "the refusal must name the threshold it missed"
    assert "nothing was estimated" in body, "the refusal must say nothing was substituted"


def test_a_provider_returning_nothing_is_a_page_state_not_a_crash(
    analysis: AnalysisResult, stub_market: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dfetch, "fetch_ohlcv", lambda ticker, period: pd.DataFrame())
    st.cache_data.clear()

    body = _text(_run(analysis, analysis.config))
    assert "no price history" in body


def test_the_page_guards_when_there_is_no_analysis_at_all(stub_market: None) -> None:
    at = _run(None, AppConfig())
    assert "no analysis yet" in _text(at)
