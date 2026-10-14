"""Contract and arithmetic tests for ``src.charts`` -- all 44 public figure builders.

A chart module is where a correct number goes to die. The analytics can be
perfect and the report still lie, because a builder scaled a decimal drawdown
twice, rebased one series and not the other, or handed Plotly a ``numpy.float64``
that blows up only at serialisation time -- in production, after the analysis
succeeded. So this module tests three separate things.

**1. The universal contract, applied to every name in** ``charts.__all__``.
Parametrised over all 44 builders, each is asserted to

* return a ``plotly.graph_objects.Figure``;
* survive ``fig.to_json()`` -- the single cheapest catch for a numpy dtype,
  ``Timestamp`` or ``NaT`` that Plotly's JSON encoder cannot write, which is a
  failure that never shows up while a figure is merely constructed;
* draw at least one trace **or** carry a layout annotation, which is the
  documented "cannot honestly plot this" form produced by
  ``theme.empty_figure``;
* end every ``hovertemplate`` it sets with ``<extra></extra>``, the convention
  that suppresses Plotly's second, unstyled trace-name box. A trace with no
  hovertemplate is skipped, not failed.

A second parametrised pass feeds every builder **empty, single-row and all-NaN**
versions of its real arguments and asserts the same contract holds: a placeholder
that states the reason, never an exception. Degrading a chart into a sentence is
a feature; a traceback on a page of a live dashboard is not.

Inputs are built from the **real analytics** on the shared seeded fixtures, so
every column name, unit and dtype is the one the app actually passes. A chart
tested against a hand-made frame with invented column names tests nothing.

**2. Hand-computed golden values and unit conventions.** Reversing one of these
silently corrupts an entire report, so each gets its own test with the
arithmetic in a comment: the equal-weight reference line is ``100 / n``; a
decimal drawdown is scaled by 100 and a percent one is not; ``growth_line``
keeps money units alone but rebases BOTH series to 100 the moment a benchmark
arrives; the correlation scale is pinned to [-1, +1] so a colour means the same
number in every screenshot; the risk gauge clamps to [0, 100].

**3. Identities that cannot rot.** Donut slices sum to portfolio value;
waterfall steps sum to ``total P&L / total invested``; capital weight and risk
contribution each sum to 100%; the scree cumulative ends at 100%; a folded
weights donut still sums to 100%; the terminal-value histogram is a share of
paths and sums to 100%.

Deterministic and offline: one seeded price panel, a 200-path Monte Carlo, no
network, no sleeps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import src.charts as charts

# ``tests/`` is on sys.path under pytest's default (prepend) import mode, so the
# shared builders behind the conftest fixtures can be reused directly. They are
# needed at module scope: constructing the analytics inputs for 44 charts once
# per session is the difference between a fast suite and a slow one, and a
# session-scoped fixture cannot depend on function-scoped ones.
from conftest import make_ohlcv, make_prices, make_quote
from src.analytics import attribution as attr
from src.analytics import benchmark as bm
from src.analytics import drawdown as dd
from src.analytics import factors as fac
from src.analytics import forecast as fcst
from src.analytics import fundamentals as fun
from src.analytics import montecarlo as mc
from src.analytics import optimize as opt
from src.analytics import positions as pos
from src.analytics import rebalance as reb
from src.analytics import returns as ret
from src.analytics import risk as rsk
from src.analytics import rolling as roll
from src.analytics import stress as strs
from src.analytics import technicals as tech
from src.charts import theme
from src.models import SecurityProfile

# ---------------------------------------------------------------------------
# the portfolio every chart is drawn from
# ---------------------------------------------------------------------------
TICKERS: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE")
SECTORS: tuple[str | None, ...] = ("Energy", "Technology", "Energy", "Financials", None)
QUANTITIES: tuple[float, ...] = (10.0, 5.0, 8.0, 3.0, 12.0)
BUY_PRICES: tuple[float, ...] = (100.0, 140.0, 190.0, 250.0, 300.0)
RISK_FREE = 0.02
ROLLING_WINDOW = 63

#: Exports of :mod:`src.charts` that are not figure builders.
NON_BUILDERS = frozenset({"theme", "empty_figure", "set_mode"})

#: Every public figure builder, in ``__all__`` order.
CHART_NAMES: tuple[str, ...] = tuple(n for n in charts.__all__ if n not in NON_BUILDERS)

#: The three shapes of bad data every builder promises to degrade on.
DEGENERATE_MODES: tuple[str, ...] = ("empty", "single-row", "all-NaN")


def _positions_frame() -> pd.DataFrame:
    """A real positions table, through ``clean_holdings`` -> ``build_positions``."""
    prices = _PRICES
    quotes = {t: make_quote(t, float(prices[t].iloc[-1])) for t in TICKERS}
    profiles = {
        t: SecurityProfile(
            t,
            name=f"{t} Industries",
            sector=s,
            currency="INR",
            quote_type="ETF" if s is None else "EQUITY",
        )
        for t, s in zip(TICKERS, SECTORS)
    }
    holdings, problems = pos.clean_holdings(
        pd.DataFrame(
            {
                "ticker": list(TICKERS),
                "quantity": list(QUANTITIES),
                "buy_price": list(BUY_PRICES),
            }
        )
    )
    assert problems == [], problems
    positions, _ = pos.build_positions(holdings, quotes, profiles, None, "INR")
    return positions


# Five correlated walks: three assets collapse the efficient frontier to a
# single feasible point, which would leave the optimiser charts untested.
_PRICES = make_prices(TICKERS, days=1400, start="2019-01-01")
_LONG_PRICES = make_prices(TICKERS, days=2600, start="2015-01-01")
_BENCH = make_prices(("BMK",), days=1400, start="2019-01-01").iloc[:, 0]


@pytest.fixture(scope="session")
def book() -> dict[str, Any]:
    """Every analytics artefact the 52 builders consume, computed once."""
    prices = _PRICES
    asset_returns = prices.pct_change().iloc[1:]
    values = rsk.portfolio_value_series(prices, dict(zip(TICKERS, QUANTITIES)))
    returns = rsk.daily_returns(values)
    bench_returns = rsk.daily_returns(_BENCH)
    positions = _positions_frame()
    weights = positions.set_index("ticker")["weight_pct"] / 100.0
    cov = rsk.covariance_matrix(asset_returns, shrinkage=0.1)
    corr = rsk.correlation_matrix(asset_returns)
    mu = opt.expected_returns(asset_returns)
    ocov = opt.covariance(asset_returns)
    portfolios = {
        opt.LABEL_MAX_SHARPE: opt.max_sharpe(mu, ocov, RISK_FREE),
        opt.LABEL_MIN_VARIANCE: opt.min_variance(mu, ocov, RISK_FREE),
        opt.LABEL_EQUAL_WEIGHT: opt.equal_weight(mu, ocov, RISK_FREE),
    }
    simulation = mc.run_simulation(
        values, returns, method="bootstrap", n_paths=200, horizon_days=60, seed=42
    )
    ohlcv = make_ohlcv("AAA", days=400)
    # A hand-built balance sheet whose identity is deliberately IMPERFECT:
    # assets exceed liabilities plus equity by a stable margin, which is the
    # minority-interest case the chart exists to draw.
    _periods = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31"])
    balance_frame = pd.DataFrame(
        {
            "Total Assets": [1000.0, 900.0, 820.0],
            "Current Assets": [400.0, 360.0, 330.0],
            "Current Liabilities": [200.0, 180.0, 170.0],
            "Total Liabilities Net Minority Interest": [520.0, 470.0, 430.0],
            "Long Term Debt": [300.0, 280.0, 250.0],
            "Total Debt": [420.0, 390.0, 350.0],
            "Stockholders Equity": [430.0, 385.0, 350.0],
            "Retained Earnings": [250.0, 220.0, 195.0],
            "Net PPE": [500.0, 460.0, 420.0],
            "Cash And Cash Equivalents": [120.0, 100.0, 90.0],
            "Ordinary Shares Number": [10.0, 10.0, 10.0],
        },
        index=_periods,
    ).T
    balance_statements = fun.read_statements(
        balance_frame, pd.DataFrame(), pd.DataFrame(), "AAA", "INR"
    )
    # The forecaster needs a 252-session lookback satisfied AND enough rows
    # left over for a purged walk-forward, so it gets its own longer series.
    forecast_bars = make_ohlcv("AAA", days=1400, seed=11)
    forecast_run = fcst.run_forecast(forecast_bars, 21, "ridge", with_importance=True)
    forecast_nodes = fcst.forecast_path(forecast_bars, (5, 21, 63), "ridge")
    forecast_board = fcst.model_leaderboard(
        forecast_bars, 21, models=("random_walk", "drift", "ridge")
    )
    pca = fac.pca_decomposition(asset_returns)
    cluster = fac.cluster_assets(corr)
    asset_stats = pd.DataFrame(
        {
            "ann_return": asset_returns.apply(ret.annualized_return),
            "ann_vol": asset_returns.apply(rsk.annualized_volatility),
        }
    )
    return {
        "prices": prices,
        "asset_returns": asset_returns,
        "values": values,
        "returns": returns,
        "benchmark": _BENCH,
        "positions": positions,
        "allocation": pos.allocation_by(positions, "sector"),
        "drift": reb.drift_table(positions, reb.equal_weight_targets(positions["ticker"]), 5.0),
        "growth": bm.compare(values, _BENCH, RISK_FREE)["growth"],
        "relative_strength": bm.relative_strength(values, _BENCH),
        "contribution": attr.return_contribution(positions),
        "monthly": ret.monthly_return_table(values),
        "drawdown": rsk.drawdown_series(values),
        "episodes": dd.drawdown_episodes(values),
        "var": rsk.value_at_risk(returns),
        "cvar": rsk.conditional_var(returns),
        "risk_contribution": attr.risk_contribution(weights, cov),
        "score": rsk.risk_score(
            rsk.annualized_volatility(returns),
            rsk.max_drawdown(values),
            positions["weight_pct"],
            len(positions),
            rsk.conditional_var(returns),
        ),
        "asset_stats": asset_stats,
        "rolling": roll.rolling_frame(returns, bench_returns, ROLLING_WINDOW, RISK_FREE, 0.95),
        "stress": strs.historical_stress(
            _LONG_PRICES, weights, portfolio_value=float(positions["current_value"].sum())
        ),
        "correlation": corr,
        "clustered_order": fac.clustered_order(corr),
        "linkage": cluster["linkage"],
        "cluster_order": cluster["order"],
        "pca": pca,
        "rolling_correlation": fac.rolling_average_correlation(asset_returns, ROLLING_WINDOW),
        "frontier": opt.efficient_frontier(mu, ocov, RISK_FREE, n_points=25),
        "random": opt.random_portfolios(mu, ocov, RISK_FREE, n=400, seed=7),
        "portfolios": portfolios,
        "current": opt.equal_weight(mu, ocov, RISK_FREE),
        "simulation": simulation,
        "ohlcv": ohlcv,
        "indicators": tech.indicator_frame(ohlcv),
        "rsi": tech.rsi(ohlcv["Close"]),
        "macd": tech.macd(ohlcv["Close"]),
        "range_52w": tech.distance_from_52w(ohlcv["Close"]),
        "balance_statements": balance_statements,
        "balance_check": fun.balance_sheet_check(balance_statements)["rows"],
        "forecast_bars": forecast_bars,
        "forecast_run": forecast_run,
        "forecast_nodes": forecast_nodes,
        "forecast_board": forecast_board,
    }


@pytest.fixture(scope="session")
def chart_cases(book: dict[str, Any]) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    """``{builder name: (positional args, keyword args)}`` for all 52 charts."""
    sim = book["simulation"]
    pca = book["pca"]
    return {
        # allocation
        "allocation_donut": ((book["positions"],), {}),
        "allocation_sunburst": ((book["positions"],), {}),
        "allocation_treemap": ((book["positions"],), {}),
        "sector_bar": ((book["allocation"],), {}),
        "weight_bar": ((book["positions"],), {}),
        "weight_vs_target_bar": ((book["drift"],), {}),
        # performance
        "benchmark_comparison": ((book["growth"],), {}),
        "contribution_waterfall": ((book["contribution"],), {}),
        "cumulative_return_chart": ((book["values"], book["benchmark"]), {}),
        "growth_line": ((book["values"],), {}),
        "monthly_heatmap": ((book["monthly"],), {}),
        "pnl_bar": ((book["positions"],), {}),
        "relative_strength_chart": ((book["relative_strength"],), {}),
        "return_bar": ((book["positions"],), {}),
        "value_bar": ((book["positions"],), {}),
        # risk
        "drawdown_chart": ((book["drawdown"],), {}),
        "drawdown_episodes_chart": ((book["values"], book["episodes"]), {}),
        "return_distribution": ((book["returns"], book["var"], book["cvar"]), {}),
        "risk_contribution_chart": ((book["risk_contribution"],), {}),
        "risk_gauge": ((book["score"]["score"], str(book["score"].get("band", ""))), {}),
        "risk_return_scatter": ((book["positions"], book["asset_stats"]), {}),
        "rolling_chart": ((book["rolling"], "volatility"), {}),
        "rolling_multi_chart": ((book["rolling"],), {}),
        "stress_bar": ((book["stress"],), {}),
        "underwater_chart": ((book["values"],), {}),
        "var_backtest_chart": ((book["returns"], book["var"]), {}),
        # correlation & factors
        "correlation_heatmap": ((book["correlation"], book["clustered_order"]), {}),
        "dendrogram": ((book["linkage"], book["cluster_order"]), {}),
        "pca_loadings_heatmap": ((pca["loadings"],), {}),
        "pca_scree": ((pca["explained_variance_ratio"], pca["cumulative"]), {}),
        "rolling_correlation_chart": ((book["rolling_correlation"],), {}),
        # statements
        "balance_sheet_chart": ((book["balance_check"],), {}),
        # forecast
        "forecast_cone_chart": (
            (
                book["forecast_bars"]["Close"],
                book["forecast_nodes"]["path"],
                book["forecast_nodes"]["nodes"],
                "₹",
            ),
            {},
        ),
        "backtest_scatter": ((book["forecast_run"]["predictions"],), {}),
        "prediction_timeline_chart": ((book["forecast_run"]["predictions"],), {}),
        "model_leaderboard_chart": ((book["forecast_board"],), {}),
        "feature_importance_chart": ((book["forecast_run"]["importance"],), {}),
        "residual_distribution_chart": ((book["forecast_run"]["residuals"],), {}),
        "horizon_skill_chart": ((book["forecast_nodes"]["nodes"],), {}),
        # optimiser
        "efficient_frontier_chart": (
            (book["frontier"], book["random"], book["portfolios"], book["current"]),
            {},
        ),
        "frontier_weights_area": ((book["frontier"],), {}),
        "weights_comparison": ((book["portfolios"],), {}),
        "weights_donut": ((book["portfolios"][opt.LABEL_MAX_SHARPE].weights,), {}),
        # simulation
        "fan_chart": ((sim["percentile_paths"], sim["start_value"]), {}),
        "sample_paths_chart": ((sim["paths"],), {}),
        "terminal_distribution": ((sim["terminal"], sim["start_value"]), {}),
        # technicals
        "macd_chart": ((book["macd"],), {}),
        "price_chart": ((book["ohlcv"], book["indicators"]), {}),
        "price_with_panels": ((book["ohlcv"], book["indicators"]), {}),
        "range_52w_bar": ((book["range_52w"],), {}),
        "rsi_chart": ((book["rsi"],), {}),
        "volume_chart": ((book["ohlcv"],), {}),
    }


@pytest.fixture(scope="session")
def figures(chart_cases: dict[str, Any]) -> dict[str, go.Figure]:
    """Each builder's healthy figure, built once and shared by the contract tests."""
    out: dict[str, go.Figure] = {}
    for name, (args, kwargs) in chart_cases.items():
        out[name] = getattr(charts, name)(*args, **kwargs)
    return out


@pytest.fixture(autouse=True)
def _restore_chart_mode():
    """No test may leak a palette change into the next one."""
    before = theme.mode()
    yield
    theme.set_mode(before)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _annotations(fig: go.Figure) -> tuple[Any, ...]:
    return tuple(fig.layout.annotations or ())


def _hovertemplates(trace: Any) -> list[str]:
    """Every hovertemplate string on a trace (Plotly allows one per point)."""
    raw = getattr(trace, "hovertemplate", None)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x) for x in raw if x is not None]
    return [str(raw)]


def _degrade(value: Any, mode: str) -> Any:
    """Return ``value`` emptied, cut to one row, or blanked to NaN.

    Frames, series and arrays degrade structurally; mappings lose their entries;
    a bare float becomes NaN, because a missing VaR threshold or risk score is
    exactly how these builders are called on a portfolio with no history.
    """
    if isinstance(value, pd.DataFrame):
        if mode == "empty":
            return value.iloc[0:0].copy()
        if mode == "single-row":
            return value.iloc[:1].copy()
        return value.copy().assign(**{str(c): np.nan for c in value.columns})
    if isinstance(value, pd.Series):
        if mode == "empty":
            return value.iloc[0:0].copy()
        if mode == "single-row":
            return value.iloc[:1].copy()
        return pd.Series(np.nan, index=value.index, name=value.name)
    if isinstance(value, np.ndarray):
        if mode == "empty":
            return np.empty((0, *value.shape[1:]), dtype=float)
        if mode == "single-row":
            return value[:1]
        return np.full(value.shape, np.nan, dtype=float)
    if isinstance(value, dict):
        if mode == "empty":
            return {}
        if mode == "single-row":
            first = next(iter(value))
            return {first: value[first]}
        return dict.fromkeys(value, np.nan)
    if isinstance(value, (list, tuple)):
        return type(value)() if mode != "single-row" else type(value)(list(value)[:1])
    if isinstance(value, float):
        return float("nan")
    return value


# ---------------------------------------------------------------------------
# the public surface itself
# ---------------------------------------------------------------------------
def test_the_package_exports_fifty_two_figure_builders_and_three_helpers() -> None:
    """``__all__`` is the page layer's whole vocabulary; drift in it is an API break."""
    assert len(charts.__all__) == len(set(charts.__all__)), "duplicate name in __all__"
    assert NON_BUILDERS <= set(charts.__all__)
    assert len(CHART_NAMES) == 52, sorted(CHART_NAMES)
    for name in charts.__all__:
        assert hasattr(charts, name), f"{name} is exported but not importable"
    for name in CHART_NAMES:
        assert callable(getattr(charts, name)), f"{name} is not callable"


def test_every_exported_builder_has_a_test_case(chart_cases: dict[str, Any]) -> None:
    """A new chart cannot be added to ``__all__`` without being exercised here."""
    assert set(chart_cases) == set(CHART_NAMES), {
        "untested": sorted(set(CHART_NAMES) - set(chart_cases)),
        "stale": sorted(set(chart_cases) - set(CHART_NAMES)),
    }


# ---------------------------------------------------------------------------
# the universal contract, on real analytics output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_chart_returns_a_plotly_figure(name: str, figures: dict[str, go.Figure]) -> None:
    """The page layer hands the result straight to ``st.plotly_chart``."""
    assert isinstance(figures[name], go.Figure)


@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_chart_survives_json_serialisation(name: str, figures: dict[str, go.Figure]) -> None:
    """``to_json`` is the cheapest catch for a dtype Plotly's encoder cannot write.

    ``numpy.float64``, ``numpy.bool_``, ``Timestamp`` and ``NaT`` all build a
    Figure happily and only fail when it is serialised -- which, in production,
    is after the analysis has succeeded and the page is being drawn.
    """
    payload = figures[name].to_json()
    assert isinstance(payload, str) and payload.startswith("{")


@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_chart_draws_a_trace_or_states_why_it_cannot(
    name: str, figures: dict[str, go.Figure]
) -> None:
    """The documented degradation is a placeholder with a message, never a blank."""
    fig = figures[name]
    assert len(fig.data) > 0 or len(_annotations(fig)) > 0


@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_hovertemplate_suppresses_the_secondary_trace_box(
    name: str, figures: dict[str, go.Figure]
) -> None:
    """A hovertemplate must end in ``<extra></extra>``.

    Without it Plotly draws a second, unstyled box carrying the trace name
    beside every tooltip -- which on a themed chart reads as a rendering bug.
    Traces that set no hovertemplate keep Plotly's default and are skipped.
    """
    for trace in figures[name].data:
        for template in _hovertemplates(trace):
            assert template.endswith("<extra></extra>"), (
                f"{name}: {trace.type} hovertemplate ends {template[-40:]!r}"
            )


# ---------------------------------------------------------------------------
# degenerate inputs: empty, one row, all-NaN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", DEGENERATE_MODES)
@pytest.mark.parametrize("name", CHART_NAMES)
def test_every_chart_degrades_instead_of_raising(
    name: str, mode: str, chart_cases: dict[str, Any]
) -> None:
    """Empty / one-row / all-NaN input must still produce an explaining Figure.

    This is the contract that keeps a dashboard alive on the day a data feed
    returns nothing: a chart that cannot honestly be drawn says so in an
    annotation. It must not raise, and it must not hand back ``None``.
    """
    args, kwargs = chart_cases[name]
    fig = getattr(charts, name)(*(_degrade(a, mode) for a in args), **kwargs)

    assert isinstance(fig, go.Figure), f"{name} returned {type(fig).__name__} on {mode}"
    assert fig.to_json().startswith("{")
    assert len(fig.data) > 0 or len(_annotations(fig)) > 0, (
        f"{name} drew nothing and explained nothing on {mode} input"
    )


# ---------------------------------------------------------------------------
# unit conventions -- reversing one silently corrupts a whole report
# ---------------------------------------------------------------------------
def test_a_decimal_drawdown_series_is_rescaled_into_percent() -> None:
    """-0.25 is a quarter of the portfolio, so it must plot at -25, not -0.25."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.drawdown_chart(pd.Series([0.0, -0.25, -0.10], index=idx))

    assert list(fig.data[0].y) == pytest.approx([0.0, -25.0, -10.0])


def test_a_drawdown_series_already_in_percent_is_left_alone() -> None:
    """The documented heuristic: a decimal drawdown can never be worse than -1.0,
    so a minimum below -1.0 means the caller already scaled it. -25 stays -25.
    """
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.drawdown_chart(pd.Series([0.0, -25.0, -10.0], index=idx))

    assert list(fig.data[0].y) == pytest.approx([0.0, -25.0, -10.0])


def test_a_total_wipeout_still_reads_as_a_decimal_drawdown() -> None:
    """-1.0 is the deepest a decimal drawdown can be, so it scales to -100%.

    The boundary is strict (``min < -1.0``), which is what makes -1.0 fall on
    the decimal side. Getting this off by one would print a -1% loss for a
    portfolio that went to zero.
    """
    idx = pd.bdate_range("2023-01-02", periods=2)
    fig = charts.drawdown_chart(pd.Series([0.0, -1.0], index=idx))

    assert list(fig.data[0].y) == pytest.approx([0.0, -100.0])


def test_growth_line_keeps_money_units_when_no_benchmark_is_supplied() -> None:
    """Alone, the chart is a mark-to-market value line: 121 must stay 121."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.growth_line(pd.Series([100.0, 110.0, 121.0], index=idx))

    assert list(fig.data[0].y) == pytest.approx([100.0, 110.0, 121.0])


def test_growth_line_rebases_both_series_to_one_hundred_once_a_benchmark_arrives() -> None:
    """Comparing rupees with index points is meaningless, so BOTH are rebased.

    portfolio 100 -> 110 -> 121  becomes  100, 110, 121  (110/100, 121/100)
    benchmark  50 ->  55 ->  52  becomes  100, 110, 104  (55/50 = 1.10, 52/50 = 1.04)
    """
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.growth_line(
        pd.Series([100.0, 110.0, 121.0], index=idx),
        benchmark=pd.Series([50.0, 55.0, 52.0], index=idx),
        benchmark_name="NIFTY",
    )

    curves = {str(t.name): np.asarray(t.y, dtype=float) for t in fig.data if t.y is not None}
    assert curves["Portfolio"] == pytest.approx([100.0, 110.0, 121.0])
    assert curves["NIFTY"] == pytest.approx([100.0, 110.0, 104.0])


def test_benchmark_comparison_rebases_a_growth_frame_that_was_not_normalised() -> None:
    """Portfolio 200 -> 220 -> 242 rebases to 100, 110, 121 (220/200, 242/200)."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    growth = pd.DataFrame(
        {"Portfolio": [200.0, 220.0, 242.0], "Benchmark": [50.0, 55.0, 52.0]}, index=idx
    )
    fig = charts.benchmark_comparison(growth)

    curves = {str(t.name): np.asarray(t.y, dtype=float) for t in fig.data if t.name}
    assert curves["Portfolio"] == pytest.approx([100.0, 110.0, 121.0])
    assert curves["Benchmark"] == pytest.approx([100.0, 110.0, 104.0])


def test_cumulative_return_starts_every_curve_at_exactly_zero_percent() -> None:
    """(V_t / V_0 - 1) * 100:  100 -> 0%, 110 -> 10%, 121 -> 21%.

    Both curves starting at zero is what makes the vertical gap on any date the
    cumulative excess return, so the zero start is load-bearing, not cosmetic.
    """
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.cumulative_return_chart(pd.Series([100.0, 110.0, 121.0], index=idx))

    y = np.asarray(fig.data[0].y, dtype=float)
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx([0.0, 10.0, 21.0])


def test_relative_strength_is_rebased_to_one_hundred_at_the_first_date() -> None:
    """A raw ratio renders identically to an already-rebased one.

    2.0 -> 2.2 -> 2.1 becomes 100, 110, 105 (2.2/2.0 = 1.10, 2.1/2.0 = 1.05).
    """
    idx = pd.bdate_range("2023-01-02", periods=3)
    fig = charts.relative_strength_chart(pd.Series([2.0, 2.2, 2.1], index=idx))

    line = next(t for t in fig.data if t.name and "elative" in str(t.name))
    assert np.asarray(line.y, dtype=float) == pytest.approx([100.0, 110.0, 105.0])


def test_the_correlation_colour_scale_is_pinned_to_minus_one_and_plus_one(
    figures: dict[str, go.Figure],
) -> None:
    """A pinned scale is why the same colour means the same number in every
    screenshot; letting Plotly auto-range would make two reports incomparable.
    """
    heatmap = figures["correlation_heatmap"].data[0]

    assert heatmap.zmin == pytest.approx(-1.0)
    assert heatmap.zmax == pytest.approx(1.0)


def test_the_correlation_diagonal_is_exactly_one(figures: dict[str, go.Figure]) -> None:
    """rho(x, x) = 1 for every asset -- an identity, so it cannot rot."""
    heatmap = figures["correlation_heatmap"].data[0]
    z = np.asarray(heatmap.z, dtype=float)

    assert np.diag(z) == pytest.approx(np.ones(z.shape[0]))
    assert list(heatmap.x) == list(heatmap.y), "a correlation heatmap must be square"


def test_the_monthly_heatmap_keeps_the_annual_column_on_its_own_panel(
    figures: dict[str, go.Figure],
) -> None:
    """Compounded annual returns span a far wider range than single months;
    forcing both onto one colour scale would wash the monthly grid out.
    """
    panels = [t for t in figures["monthly_heatmap"].data if t.type == "heatmap"]

    assert len(panels) == 2
    assert len(panels[0].x) == 12, "the month panel carries Jan..Dec"
    assert tuple(panels[1].x) == ("Year",)


def test_the_treemap_colour_scale_is_fixed_and_does_not_follow_the_data(
    book: dict[str, Any],
) -> None:
    """A tile's colour must mean the same thing on every portfolio.

    The bound used to be the 90th percentile of the observed absolute returns,
    which kept the ramp saturated on any book — and made the same deep blue
    worth +8% on a quiet portfolio and +140% on a volatile one. Two portfolios
    could not be compared, and neither could the same portfolio a week apart.

    So the scale is pinned to ±``RETURN_COLOR_LIMIT_PCT`` and this test feeds it
    two portfolios whose returns differ by two orders of magnitude to prove the
    bound does not move.
    """
    from src.charts.allocation import RETURN_COLOR_LIMIT_PCT

    calm = book["positions"].copy()
    calm["pnl_pct"] = [0.4, -0.3, 0.2, -0.1, 0.5]

    wild = book["positions"].copy()
    wild["pnl_pct"] = [410.0, -62.0, 180.0, -44.0, 730.0]

    for name, frame in (("calm", calm), ("wild", wild)):
        marker = charts.allocation_treemap(frame).data[0].marker
        assert float(marker.cmin) == pytest.approx(-RETURN_COLOR_LIMIT_PCT), name
        assert float(marker.cmax) == pytest.approx(RETURN_COLOR_LIMIT_PCT), name
        assert float(marker.cmid) == pytest.approx(0.0), name


def test_the_treemap_colourbar_ticks_land_on_round_numbers(
    figures: dict[str, go.Figure],
) -> None:
    """A fixed bound means the ticks can be fixed too: -30 / -15 / 0 / +15 / +30."""
    from src.charts.allocation import RETURN_COLOR_LIMIT_PCT

    bar = figures["allocation_treemap"].data[0].marker.colorbar
    assert float(bar.dtick) == pytest.approx(RETURN_COLOR_LIMIT_PCT / 2.0)
    assert bar.ticksuffix == "%"


def test_the_treemap_states_its_colour_bound_in_the_figure_itself(
    figures: dict[str, go.Figure],
) -> None:
    """A saturating scale is only honest if the figure says where it saturates."""
    fig = figures["allocation_treemap"]
    printed = str(fig.layout.title.text) + " ".join(
        str(a.text) for a in (fig.layout.annotations or ())
    )
    assert "30%" in printed
    assert "saturat" in printed.lower()


# ---------------------------------------------------------------------------
# hand-computed golden values
# ---------------------------------------------------------------------------
def test_the_equal_weight_reference_line_sits_at_one_hundred_over_n(
    book: dict[str, Any],
) -> None:
    """Five holdings -> 100 / 5 = 20.0% is the weight each would carry equally weighted."""
    fig = charts.weight_bar(book["positions"])

    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert lines, "the equal-weight reference line is missing"
    assert float(lines[0].x0) == pytest.approx(100.0 / len(TICKERS))
    assert float(lines[0].x0) == pytest.approx(float(lines[0].x1)), "must be vertical"


def test_the_risk_gauge_clamps_a_score_onto_the_zero_to_hundred_scale() -> None:
    """The rubric is defined on [0, 100]; a needle off the dial is a lie."""
    assert charts.risk_gauge(31.7).data[0].value == pytest.approx(31.7)
    assert charts.risk_gauge(130.0).data[0].value == pytest.approx(100.0)
    assert charts.risk_gauge(-5.0).data[0].value == pytest.approx(0.0)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_a_missing_risk_score_draws_a_placeholder_not_a_zero_needle(missing: Any) -> None:
    """ "Insufficient data" and "the safest possible portfolio" are not the same
    reading, so an undefined score must never render as a needle at zero.
    """
    fig = charts.risk_gauge(missing)

    assert len(fig.data) == 0
    assert len(_annotations(fig)) == 1


def test_the_fifty_two_week_bullet_places_the_price_by_arithmetic() -> None:
    """low 100, high 200, last 150.

    track  : base 100, width 100  (the whole 52-week range)
    fill   : base 100, width  50  (150 - 100, i.e. halfway up the range)
    """
    fig = charts.range_52w_bar(150.0, low=100.0, high=200.0, ticker="AAA")

    bars = [t for t in fig.data if t.type == "bar"]
    assert float(bars[0].base[0]) == pytest.approx(100.0)
    assert float(bars[0].x[0]) == pytest.approx(100.0)
    assert float(bars[1].base[0]) == pytest.approx(100.0)
    assert float(bars[1].x[0]) == pytest.approx(50.0)


def test_the_fifty_two_week_bullet_refuses_an_inconsistent_range() -> None:
    """A high below the low is bad reference data, not a chart to draw."""
    fig = charts.range_52w_bar(150.0, low=200.0, high=100.0)

    assert len(fig.data) == 0
    assert "below the low" in _annotations(fig)[0].text


def test_the_fifty_two_week_bullet_refuses_a_zero_width_range() -> None:
    """high == low leaves nothing to position the price within."""
    fig = charts.range_52w_bar(150.0, low=150.0, high=150.0)

    assert len(fig.data) == 0
    assert "no range" in _annotations(fig)[0].text


def test_the_weights_donut_folds_the_tail_and_still_sums_to_one_hundred() -> None:
    """A: 50%, B: 30%, E: 18%, and C (1.2%) + D (0.8%) folded into Other = 2.0%.

    50 + 30 + 18 + 2 = 100, so folding loses no allocation -- it only moves it.
    """
    fig = charts.weights_donut(
        {"A": 0.50, "B": 0.30, "C": 0.012, "D": 0.008, "E": 0.18}, min_slice_pct=1.5
    )
    slices = fig.data[0]

    assert list(slices.labels) == ["A", "B", "E", "Other (2 names)"]
    assert np.asarray(slices.values, dtype=float) == pytest.approx([50.0, 30.0, 18.0, 2.0])
    assert float(np.sum(slices.values)) == pytest.approx(100.0)


def test_the_weights_donut_reads_fractions_and_percentages_identically() -> None:
    """{A: 0.6, B: 0.4} and {A: 60, B: 40} are the same portfolio."""
    as_fractions = charts.weights_donut({"A": 0.6, "B": 0.4})
    as_percent = charts.weights_donut({"A": 60.0, "B": 40.0})

    assert np.asarray(as_fractions.data[0].values, dtype=float) == pytest.approx([60.0, 40.0])
    assert np.asarray(as_percent.data[0].values, dtype=float) == pytest.approx([60.0, 40.0])


def test_the_scree_plot_reads_fractions_and_percentages_identically() -> None:
    """explained_variance_ratio arrives as fractions summing to 1 or as percents."""
    from_fractions = charts.pca_scree(np.array([0.7, 0.2, 0.1]))
    from_percent = charts.pca_scree(np.array([70.0, 20.0, 10.0]))

    bars_a = next(t for t in from_fractions.data if t.type == "bar")
    bars_b = next(t for t in from_percent.data if t.type == "bar")
    assert np.asarray(bars_a.y, dtype=float) == pytest.approx([70.0, 20.0, 10.0])
    assert np.asarray(bars_b.y, dtype=float) == pytest.approx([70.0, 20.0, 10.0])


def test_the_donut_is_drawn_as_a_donut(figures: dict[str, go.Figure]) -> None:
    """A regression guard on the shape itself: a zero hole is a pie chart."""
    assert 0.0 < float(figures["allocation_donut"].data[0].hole) < 1.0


# ---------------------------------------------------------------------------
# identities -- true for any portfolio, so they cannot rot
# ---------------------------------------------------------------------------
def test_the_allocation_donut_slices_sum_to_the_portfolio_market_value(
    book: dict[str, Any], figures: dict[str, go.Figure]
) -> None:
    """Every rupee of market value lands in exactly one wedge."""
    slices = figures["allocation_donut"].data[0]
    total = float(book["positions"]["current_value"].sum())

    assert float(np.sum(slices.values)) == pytest.approx(total, rel=1e-12)
    assert len(slices.labels) == len(TICKERS), "one wedge per holding, none dropped"


def test_the_sunburst_root_ring_equals_the_whole_portfolio(
    book: dict[str, Any], figures: dict[str, go.Figure]
) -> None:
    """``branchvalues='total'`` means a parent's value IS the sum of its children,
    so the sector ring must reconstruct the portfolio exactly.
    """
    node = figures["allocation_sunburst"].data[0]
    total = float(book["positions"]["current_value"].sum())
    parents = list(node.parents)
    values = np.asarray(node.values, dtype=float)

    top = values[[i for i, parent in enumerate(parents) if not parent]]
    assert float(top.sum()) == pytest.approx(total, rel=1e-12)
    assert node.branchvalues == "total"


def test_the_waterfall_steps_sum_to_the_portfolio_return_on_invested_capital(
    book: dict[str, Any], figures: dict[str, go.Figure]
) -> None:
    """Percentage returns per holding do not add; contributions do.

    sum(contribution_pct) must equal total P&L / total invested x 100 -- which
    is the whole reason a waterfall is an honest way to draw them.
    """
    bar = figures["contribution_waterfall"].data[0]
    steps = np.asarray([y for y, m in zip(bar.y, bar.measure) if m == "relative"], dtype=float)
    positions = book["positions"]
    expected = float(positions["pnl"].sum()) / float(positions["investment"].sum()) * 100.0

    assert float(steps.sum()) == pytest.approx(expected, rel=1e-9)
    assert float(steps.sum()) == pytest.approx(
        float(book["contribution"]["contribution_pct"].sum()), rel=1e-12
    )
    assert list(bar.measure)[-1] == "total", "the closing bar is the portfolio return"


def test_capital_weight_and_risk_contribution_each_sum_to_one_hundred_percent(
    figures: dict[str, go.Figure],
) -> None:
    """The chart exists to show the two diverging, so both must be complete splits."""
    series = {
        str(t.name): np.asarray(t.x, dtype=float)
        for t in figures["risk_contribution_chart"].data
        if t.x is not None
    }
    assert len(series) == 2, sorted(series)
    for label, values in series.items():
        assert float(values.sum()) == pytest.approx(100.0, abs=1e-6), label


def test_the_scree_cumulative_line_ends_at_one_hundred_percent() -> None:
    """Explained-variance ratios sum to 1 by construction: 70 + 20 + 10 = 100.

    Cumulative: 70, 70 + 20 = 90, 90 + 10 = 100.
    """
    fig = charts.pca_scree(np.array([0.7, 0.2, 0.1]))
    line = next(t for t in fig.data if t.type == "scatter")

    assert np.asarray(line.y, dtype=float) == pytest.approx([70.0, 90.0, 100.0])


def test_explained_variance_ratios_from_the_real_decomposition_sum_to_one(
    book: dict[str, Any],
) -> None:
    """The identity the scree plot is drawn from, checked at the source."""
    ratios = np.asarray(book["pca"]["explained_variance_ratio"], dtype=float)

    assert float(ratios.sum()) == pytest.approx(1.0, abs=1e-9)


def test_a_derived_cumulative_line_closes_on_one_hundred_percent(
    book: dict[str, Any],
) -> None:
    """With ``cumulative`` omitted the line is derived from the (already scaled)
    bars, which isolates the supplied-cumulative path tested below.
    """
    fig = charts.pca_scree(book["pca"]["explained_variance_ratio"])
    line = next(t for t in fig.data if t.type == "scatter")

    assert float(np.asarray(line.y, dtype=float)[-1]) == pytest.approx(100.0, abs=1e-6)


def test_a_supplied_cumulative_vector_of_fractions_is_scaled_to_percent(
    figures: dict[str, go.Figure],
) -> None:
    """Regression guard on a fixed unit bug in the scree plot.

    ``factors.pca_decomposition()['cumulative']`` is a running total of
    fractions -- ``[0.66, 0.76, 0.86, 0.93, 1.00]`` for this five-asset panel --
    and ``src/ui/pages/correlation.py`` hands it straight to ``pca_scree``.
    Detecting "already percent" from the vector's SUM classifies it wrongly
    (that sum is 4.22), which pins the curve to the floor of an axis fixed to
    [0, 105] and silently suppresses the "N components are needed to reach 80%
    of variance" caption. Detecting from the MAXIMUM is unambiguous for both
    shapes: a fraction vector never exceeds 1, a percent vector's leading
    component always does.
    """
    line = next(t for t in figures["pca_scree"].data if t.type == "scatter")
    y = np.asarray(line.y, dtype=float)

    assert float(y[-1]) == pytest.approx(100.0, abs=1e-6)
    assert float(np.nansum(np.asarray(figures["pca_scree"].data[0].y, dtype=float))) == (
        pytest.approx(100.0, abs=1e-6)
    ), "the bars are in the same percent units as the line"
    assert np.all(np.diff(y) >= -1e-9), "a cumulative line never decreases"


def test_a_cumulative_vector_already_in_percent_is_not_scaled_twice() -> None:
    """[70, 90, 100] is already the answer; multiplying it again would give 10000."""
    fig = charts.pca_scree(np.array([0.7, 0.2, 0.1]), np.array([70.0, 90.0, 100.0]))
    line = next(t for t in fig.data if t.type == "scatter")

    assert np.asarray(line.y, dtype=float) == pytest.approx([70.0, 90.0, 100.0])


def test_the_terminal_distribution_is_a_share_of_paths_summing_to_one_hundred() -> None:
    """Plotting counts would make two runs with different path counts
    incomparable; plotting shares keeps the y axis meaningful across runs.
    """
    terminal = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0])
    fig = charts.terminal_distribution(terminal, 100.0)

    assert float(np.sum(np.asarray(fig.data[0].y, dtype=float))) == pytest.approx(100.0)


def test_the_fan_chart_anchors_a_reference_line_at_the_starting_value() -> None:
    """The whole chart reads as "relative to today", so today must be marked."""
    paths = pd.DataFrame(
        {"p5": [100.0, 95.0, 90.0], "p50": [100.0, 101.0, 102.0], "p95": [100.0, 108.0, 116.0]}
    )
    fig = charts.fan_chart(paths, 100.0)

    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert lines, "no start-value reference line"
    assert float(lines[0].y0) == pytest.approx(100.0)
    median = next(t for t in fig.data if t.name and "edian" in str(t.name))
    assert np.asarray(median.y, dtype=float) == pytest.approx([100.0, 101.0, 102.0])


def test_the_frontier_weights_area_traces_one_band_per_asset_it_kept(
    book: dict[str, Any], figures: dict[str, go.Figure]
) -> None:
    """Every band is stacked, so the count of bands is the count of names shown."""
    bands = [t for t in figures["frontier_weights_area"].data if t.stackgroup]

    assert 0 < len(bands) <= len(TICKERS) + 1  # +1 for a possible pooled "Other"
    assert len(figures["frontier_weights_area"].data) == len(bands)


# ---------------------------------------------------------------------------
# order- and label-insensitivity
# ---------------------------------------------------------------------------
def test_an_unsorted_value_series_plots_identically_to_a_sorted_one() -> None:
    """Charts sort defensively: a CSV in the wrong order must not draw a zigzag."""
    idx = pd.bdate_range("2023-01-02", periods=5)
    values = pd.Series([100.0, 110.0, 105.0, 120.0, 118.0], index=idx)
    shuffled = values.iloc[[3, 0, 4, 1, 2]]

    ordered = charts.growth_line(values)
    scrambled = charts.growth_line(shuffled)

    assert list(scrambled.data[0].y) == pytest.approx(list(ordered.data[0].y))
    assert list(scrambled.data[0].x) == list(ordered.data[0].x)


@pytest.mark.parametrize("builder", ["growth_line", "underwater_chart", "cumulative_return_chart"])
def test_duplicate_index_labels_do_not_raise(builder: str) -> None:
    """A duplicated trade date in an uploaded file must not take a page down."""
    idx = pd.bdate_range("2023-01-02", periods=2)
    duplicated = pd.Series([100.0, 110.0, 105.0], index=[idx[0], idx[0], idx[1]])

    fig = getattr(charts, builder)(duplicated)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 or len(_annotations(fig)) > 0


def test_an_unknown_rolling_column_names_what_is_available(book: dict[str, Any]) -> None:
    """The failure message has to be actionable, not just "no data"."""
    fig = charts.rolling_chart(book["rolling"], "sharpé")

    assert len(fig.data) == 0
    message = _annotations(fig)[0].text
    assert "sharp" in message and "available" in message


# ---------------------------------------------------------------------------
# theme: palette, mode switching, semantic colour discipline
# ---------------------------------------------------------------------------
def test_switching_the_mode_swaps_the_gain_colour_and_switching_back_restores_it() -> None:
    """The UI calls ``set_mode`` once per render; a stuck palette means a dark
    chart on a light page (or the reverse) for the rest of the session.
    """
    theme.set_mode("dark")
    dark_gain = theme.palette()["gain"]

    theme.set_mode("light")
    light_gain = theme.palette()["gain"]

    theme.set_mode("dark")
    restored = theme.palette()["gain"]

    assert dark_gain == theme.DARK_PALETTE["gain"]
    assert light_gain == theme.LIGHT_PALETTE["gain"]
    assert dark_gain != light_gain
    assert restored == dark_gain


def test_an_unrecognised_mode_is_ignored_rather_than_breaking_every_figure() -> None:
    """Only "dark" and "light" exist; anything else leaves the palette untouched."""
    theme.set_mode("light")
    theme.set_mode("solarized")

    assert theme.mode() == "light"
    assert theme.palette() is not theme.DARK_PALETTE
    assert theme.palette()["gain"] == theme.LIGHT_PALETTE["gain"]


def test_the_mode_in_force_when_a_figure_is_built_is_the_one_it_wears() -> None:
    """Not just the palette dict -- the colour must reach the trace."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    drawdown = pd.Series([0.0, -0.10, -0.20], index=idx)

    theme.set_mode("dark")
    dark_line = charts.drawdown_chart(drawdown).data[0].line.color
    theme.set_mode("light")
    light_line = charts.drawdown_chart(drawdown).data[0].line.color

    assert dark_line == theme.DARK_PALETTE["loss"]
    assert light_line == theme.LIGHT_PALETTE["loss"]


def test_both_palettes_define_exactly_the_same_semantic_keys() -> None:
    """A key present in one palette and absent from the other is a KeyError
    waiting for whichever mode the user happens to be in.
    """
    assert set(theme.DARK_PALETTE) == set(theme.LIGHT_PALETTE)
    assert {"gain", "loss", "accent", "benchmark", "muted", "text", "grid"} <= set(
        theme.DARK_PALETTE
    )


def test_the_categorical_ramp_never_reuses_the_gain_or_loss_hue() -> None:
    """Red and green are reserved for money made and lost.

    If a qualitative series could be drawn in the gain green, a reader would
    have to check the legend to know whether a colour meant "this holding" or
    "up" -- which is exactly the ambiguity the design system exists to remove.
    """
    ramp = set(theme.CATEGORICAL)

    assert len(theme.CATEGORICAL) == len(ramp), "duplicate colour in the ramp"
    for palette in (theme.DARK_PALETTE, theme.LIGHT_PALETTE):
        assert palette["gain"] not in ramp
        assert palette["loss"] not in ramp


def test_sign_colors_maps_the_sign_of_a_number_onto_the_semantic_pair() -> None:
    """Zero is not a loss: the boundary is ``>= 0`` -> gain."""
    p = theme.palette()

    assert theme.sign_colors([1.5, -1.5, 0.0]) == [p["gain"], p["loss"], p["gain"]]
    assert theme.sign_colors(["not a number"]) == [p["muted"]]


def test_volume_bars_are_not_coloured_for_a_session_with_no_prior_close(
    book: dict[str, Any],
) -> None:
    """The first bar has nothing to compare against.

    ``sign_colors`` maps a NaN difference to the loss colour (NaN >= 0 is
    False), so ``volume_chart`` overrides it to the neutral muted colour --
    otherwise every security would open its volume history with a red bar it
    had no evidence for.
    """
    fig = charts.volume_chart(book["ohlcv"])
    bars = next(t for t in fig.data if t.type == "bar")
    colors = list(bars.marker.color)

    assert colors[0] == theme.palette()["muted"]
    assert len(colors) == len(bars.y)
    assert set(colors[1:]) <= {theme.palette()["gain"], theme.palette()["loss"]}


def test_the_empty_figure_carries_its_reason_and_draws_no_trace() -> None:
    """The placeholder is the documented degradation, so its shape is a contract."""
    fig = theme.empty_figure("Not enough overlapping history.")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert [a.text for a in _annotations(fig)] == ["Not enough overlapping history."]
    assert fig.layout.showlegend is False


def test_the_chart_package_re_exports_the_theme_helpers_the_pages_use() -> None:
    """``charts.set_mode`` and ``charts.empty_figure`` are the page-layer aliases."""
    assert charts.set_mode is theme.set_mode
    assert charts.empty_figure is theme.empty_figure
    assert charts.theme is theme
