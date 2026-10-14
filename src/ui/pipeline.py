"""Analysis orchestration: inputs and market data in, ``AnalysisResult`` out.

This module is the seam between the UI and the analytics package. It is the
only place that knows the *order* things must happen in, which keeps seventeen
page modules free of any computation at all — they read fields off the result
object and render them.

It runs once per input change. ``session.run_key()`` fingerprints the holdings
and every setting that could alter a number, so switching pages (which reruns
the whole Streamlit script) reuses the cached result instead of refetching the
market.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.analytics import attribution as attr
from src.analytics import benchmark as bm
from src.analytics import drawdown as dd
from src.analytics import factors as fct
from src.analytics import insights as ins
from src.analytics import lots as lot
from src.analytics import positions as pos
from src.analytics import returns as ret
from src.analytics import risk as rsk
from src.analytics import rolling as roll
from src.config import MIN_OBSERVATIONS, AppConfig
from src.data.fetcher import DataFetchError
from src.models import AnalysisResult
from src.ui import data_access, session


class NoValidHoldings(Exception):
    """Raised when nothing analysable survives validation."""


class NoMarketData(Exception):
    """Raised when not a single holding could be priced."""


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def _resolve_fx(
    positions_currencies: list[str], cfg: AppConfig, warnings: list[str]
) -> dict[str, float]:
    """Spot rates converting each holding currency into the reporting currency."""
    foreign = {c for c in positions_currencies if c and c != cfg.base_currency}
    if not cfg.convert_fx or not foreign:
        return {}
    rates, errors = data_access.load_fx_rates(
        tuple(sorted(foreign | {cfg.base_currency})), cfg.base_currency
    )
    for ccy, message in errors.items():
        warnings.append(
            f"FX rate {ccy}->{cfg.base_currency} unavailable ({message}); "
            f"{ccy} holdings are counted at face value."
        )
    return rates


def _price_history(tickers: tuple[str, ...], cfg: AppConfig, warnings: list[str]) -> pd.DataFrame:
    prices, errors = data_access.load_history(tickers, cfg.period)
    for ticker, message in errors.items():
        warnings.append(
            f"No price history for {ticker}: {message} — excluded from risk statistics."
        )
    return prices


def _benchmark_block(
    values: pd.Series, cfg: AppConfig, warnings: list[str]
) -> tuple[dict, pd.Series | None]:
    if len(values) < 2:
        return {"ok": False, "reason": "not enough portfolio history"}, None
    try:
        prices = data_access.load_benchmark(cfg.benchmark_symbol, cfg.period)
    except DataFetchError as exc:
        warnings.append(f"Benchmark unavailable: {exc}")
        return {"ok": False, "reason": str(exc)}, None
    return bm.compare(values, prices, cfg.risk_free_rate), prices


def _ledger_block(warnings: list[str], raw: pd.DataFrame | None = None) -> dict:
    """FIFO lots, realised P&L and money-weighted return from the trade ledger.

    ``raw`` defaults to the session's ledger; the scheduled runner passes a
    frame loaded straight from SQLite, because a cron job has no session.
    """
    if raw is None:
        raw = session.get_transactions()
    if raw is None or raw.empty:
        return {}
    txns, problems = lot.parse_transactions(raw)
    warnings.extend(problems)
    if not txns:
        return {}
    open_lots, realised, lot_problems = lot.build_lots(txns)
    warnings.extend(lot_problems)
    return {
        "transactions": txns,
        "open_lots": open_lots,
        "realised": realised,
        "realised_summary": lot.realised_summary(realised),
        "cash_flows": lot.cash_flows(txns),
    }


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run(
    cfg: AppConfig,
    holdings: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> AnalysisResult:
    """Execute the full pipeline and return a populated :class:`AnalysisResult`.

    ``holdings`` and ``transactions`` default to whatever is in Streamlit's
    session state, which is what every page does. Passing them explicitly runs
    the identical pipeline with NO session at all -- the path
    :mod:`src.reporting.cli` uses to build a scheduled tearsheet from a cron
    job, where there is no browser and ``st.session_state`` is empty.

    Nothing else differs between the two: same fetches, same statistics, same
    warnings. A scheduled report that used a second, simplified code path
    would eventually disagree with the dashboard, and the whole point of the
    e-mail is that it says what the app says.
    """
    warnings: list[str] = []

    # -- 1. validate user input ------------------------------------------
    raw_holdings = session.get_holdings() if holdings is None else holdings
    holdings, problems = pos.clean_holdings(raw_holdings)
    warnings.extend(problems)
    if holdings.empty:
        raise NoValidHoldings("Add at least one holding with a positive quantity and buy price.")

    tickers = tuple(holdings["ticker"])

    # -- 2. market data ---------------------------------------------------
    quotes, quote_errors = data_access.load_quotes(tickers)
    for ticker, message in quote_errors.items():
        warnings.append(f"Skipping {ticker}: {message}")

    profiles = data_access.load_profiles(tickers)

    quote_currencies = [q.currency for q in quotes.values() if q.currency]
    fx_rates = _resolve_fx(quote_currencies, cfg, warnings)

    # -- 3. positions -----------------------------------------------------
    positions, skipped = pos.build_positions(
        holdings, quotes, profiles, fx_rates or None, cfg.base_currency
    )
    if positions.empty:
        raise NoMarketData(
            "No market data could be fetched for any holding. Check the symbols "
            "(Indian stocks need a .NS or .BO suffix) and your connection."
        )
    for ticker in skipped:
        if ticker not in quote_errors:
            warnings.append(f"Skipping {ticker}: no usable quote.")

    summary = pos.summarise(positions)
    if not cfg.convert_fx and len(summary.get("currencies", [])) > 1:
        warnings.append(
            f"Holdings span {', '.join(summary['currencies'])} and FX conversion is "
            f"off — aggregate totals mix currencies and are indicative only."
        )

    # -- 4. history and portfolio value series ----------------------------
    priced = tuple(positions["ticker"])
    prices = _price_history(priced, cfg, warnings)

    if prices.empty:
        values = pd.Series(dtype="float64")
        asset_returns = pd.DataFrame()
    else:
        values = rsk.portfolio_value_series(
            prices, dict(zip(positions["ticker"], positions["quantity"]))
        )
        asset_returns = rsk.daily_returns(prices)

    port_returns = rsk.daily_returns(values) if len(values) > 1 else pd.Series(dtype="float64")

    if 1 < len(values) < MIN_OBSERVATIONS:
        warnings.append(
            f"Only {len(values)} overlapping trading days of history — risk statistics "
            f"are indicative at best ({MIN_OBSERVATIONS}+ recommended). Try a longer window."
        )

    # -- 5. risk ----------------------------------------------------------
    risk = rsk.risk_summary(
        values,
        port_returns,
        positions["weight_pct"],
        int(summary.get("n_holdings", 0)),
        cfg.risk_free_rate,
        cfg.var_confidence,
    )
    score = rsk.risk_score(
        risk.get("ann_vol", float("nan")),
        risk.get("mdd", float("nan")),
        positions["weight_pct"],
        int(summary.get("n_holdings", 0)),
        risk.get("cvar", float("nan")),
    )
    episodes = dd.drawdown_episodes(values) if len(values) > 1 else []

    # -- 6. benchmark -----------------------------------------------------
    bench, bench_prices = _benchmark_block(values, cfg, warnings)

    # -- 7. cross-sectional structure -------------------------------------
    correlation = None
    factor_block: dict = {}
    if isinstance(asset_returns, pd.DataFrame) and asset_returns.shape[1] > 1:
        correlation = rsk.correlation_matrix(asset_returns)
        factor_block = {
            "pca": fct.pca_decomposition(asset_returns),
            "clusters": fct.cluster_assets(correlation),
            "average_correlation": fct.average_correlation(correlation),
        }

    covariance = (
        rsk.covariance_matrix(asset_returns, shrinkage=cfg.optimizer.shrinkage)
        if isinstance(asset_returns, pd.DataFrame) and asset_returns.shape[1] > 1
        else None
    )
    attribution = attr.attribution_summary(positions, asset_returns, covariance)
    attribution["factors"] = factor_block

    # -- 8. rolling -------------------------------------------------------
    bench_returns = (
        rsk.daily_returns(bench_prices)
        if bench_prices is not None and len(bench_prices) > 1
        else pd.Series(dtype="float64")
    )
    rolling_frame = roll.rolling_frame(
        port_returns,
        bench_returns,
        cfg.rolling_window,
        cfg.risk_free_rate,
        cfg.var_confidence,
    )

    # -- 9. ledger --------------------------------------------------------
    ledger = _ledger_block(warnings, transactions)
    if ledger:
        flows = list(ledger["cash_flows"])
        as_of = datetime.now(timezone.utc).date()
        ledger["xirr"] = ret.portfolio_xirr(
            ledger["transactions"], float(summary.get("current_value", 0.0)), as_of
        )
        ledger["cash_flows"] = flows

    # -- 10. narrative ----------------------------------------------------
    notes = ins.generate_insights(
        positions, summary, risk, bench, correlation, episodes, attribution
    )

    return AnalysisResult(
        positions=positions,
        summary=summary,
        quotes=quotes,
        profiles=profiles,
        prices=prices,
        values=values,
        returns=port_returns,
        asset_returns=asset_returns,
        benchmark_prices=bench_prices,
        risk=risk,
        benchmark=bench,
        attribution=attribution,
        drawdowns=episodes,
        correlation=correlation,
        covariance=covariance,
        rolling=rolling_frame,
        monthly=ret.monthly_return_table(values) if len(values) > 21 else pd.DataFrame(),
        return_stats=ret.return_stats(values) if len(values) > 1 else {},
        ledger=ledger,
        score=score,
        insights=notes,
        fx=fx_rates,
        warnings=warnings,
        config=cfg,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# session-aware wrapper
# ---------------------------------------------------------------------------
def ensure_result(cfg: AppConfig) -> None:
    """Run the pipeline only when the inputs actually changed.

    Streamlit reruns the script on every interaction and on every page switch.
    Without this guard, clicking through the navigation would refetch the market
    a dozen times.
    """
    if not st.session_state.get(session.RUN):
        return

    key = session.run_key(cfg, session.get_holdings())
    if session.has_result() and st.session_state.get(session.LAST_RUN_KEY) == key:
        return

    try:
        with st.spinner("Fetching prices and computing analytics…"):
            session.set_result(run(cfg))
        st.session_state[session.LAST_RUN_KEY] = key
    except (NoValidHoldings, NoMarketData) as exc:
        session.set_result(None)
        st.session_state[session.RUN] = False
        st.error(str(exc))
