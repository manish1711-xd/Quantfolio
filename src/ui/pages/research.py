"""Company research -- a full report on a company you do not necessarily own.

Every other page answers a question about *the portfolio*. This one starts from
nothing but a name and answers the question the user actually asked: should I
own this company at all?

The page is deliberately a **reader**, not an engine. Every number comes from a
module that can be tested without Streamlit -- :mod:`src.analytics.company` for
the statements, :mod:`src.analytics.dcf` for the valuation,
:mod:`src.analytics.peers` for the comparison, :mod:`src.analytics.scorecard`
for the verdict -- and every fetch goes through :mod:`src.ui.data_access`. What
lives here is ordering, wording and state.

Two design rules run through the whole file, and they are the reason it is
shaped the way it is:

**Never auto-select a company.** The provider resolves ``"tcs"`` to Tecsys Inc
of Canada as readily as to Tata Consultancy Services. Resolution therefore ends
in a radio button the reader has to press, and the header of the report states
the ticker, exchange, currency and data-as-of date so a wrong pick is visible
rather than merely plausible.

**Never invent a number.** A missing input renders the house em-dash. A section
that cannot be computed says why in words instead of showing a blank chart or a
zero. The ML section in particular reports its honest out-of-sample performance,
including -- usually -- that it is indistinguishable from noise.

Unlike the pages that reason about holdings, this page does **not** call
:func:`src.ui.guards.require_result`: researching a company is independent of
whether the portfolio has been analysed, and requiring a run first would put a
gate in front of the one page that needs no portfolio data.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import company as co
from src.analytics import dcf, ml_signal
from src.analytics import fundamentals as fu
from src.analytics import peers as pr
from src.analytics import scorecard as sc
from src.analytics import sentiment as sen
from src.config import DATA_SOURCE, DISCLAIMER
from src.data.research import Candidate
from src.ui import components as ui
from src.ui import data_access, fundamentals_panel, session
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    currency_symbol,
    fmt_compact,
    fmt_date,
    fmt_money,
    fmt_multiple,
    fmt_num,
    fmt_ratio_pct,
)

#: Session keys. The chosen ticker outlives the search box so that changing a
#: DCF slider does not silently re-run the resolution and lose the selection.
_QUERY_KEY = "research_query"
_PICK_KEY = "research_pick"
#: The query as it was when the search was last submitted. A form's widgets
#: only deliver their values on submit, so the eager "clear the pick when the
#: query changes" behaviour has to live here, at the only moment the script
#: can compare the new text against the old one.
_LAST_QUERY_KEY = "research_last_query"

#: Price window fetched for the risk and ML sections. Five years gives a
#: walk-forward forecaster enough folds to report a fold-to-fold spread rather
#: than a single number, and is where the volatility percentile starts to mean
#: something.
_WINDOW = "5y"

#: Headlines scored. The measurement below de-clusters to one observation per
#: day, so a larger pull buys days rather than sample size -- but only up to
#: what the feed actually holds, which is a few months at most.
_NEWS_LIMIT = 80

#: Maximum peers compared. Each one costs a provider ``.info`` call, and beyond
#: about this many the marginal company adds a name to a median without
#: changing it.
_MAX_PEERS = 10

#: Placeholder long-run risk-free rates by country, used only to seed the WACC
#: field. Deliberately visible and editable rather than fetched: a rate pulled
#: silently from an API is a stale number presented as a current one, and the
#: reader is going to disagree with the discount rate anyway.
_RISK_FREE_SEED: dict[str, float] = {
    "India": 0.070,
    "United States": 0.045,
    "United Kingdom": 0.045,
    "Japan": 0.015,
}

#: Equity risk premium used in the same seeding calculation. A convention, not
#: a measurement, and stated as such where it is shown.
_EQUITY_RISK_PREMIUM = 0.055

#: The band label -> the tone the verdict banner is drawn in.
_VERDICT_TONE: dict[str, str] = {
    "Strong buy": "gain",
    "Buy": "gain",
    "Hold": "warn",
    "Reduce": "loss",
    "Avoid": "loss",
}


# ---------------------------------------------------------------------------
# display helpers (no analytics happen in this module)
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Coerce to a finite float, or NaN. The same contract as the analytics layer."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _text(value: object) -> str:
    """Stringify a possibly-missing reference field, collapsing blanks to a dash."""
    if value is None:
        return DASH
    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return DASH
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.upper() not in {"NAN", "NONE", "NAT"} else DASH


def _pct(value: object) -> str:
    """A decimal rate as a percentage, or an em-dash."""
    return fmt_ratio_pct(value)


def _money(value: object, currency: str) -> str:
    return fmt_money(value, currency) if _num(value) == _num(value) else DASH


def _compact(value: object, currency: str) -> str:
    return fmt_compact(value, currency) if _num(value) == _num(value) else DASH


def _tone_for(value: float, invert: bool = False) -> str:
    """Sign of a finite number -> a semantic tone name, muted when missing."""
    if math.isnan(value):
        return "muted"
    if value == 0.0:
        return "neutral"
    positive = value > 0
    return "gain" if positive is not invert else "loss"


def _search_box() -> str:
    """The one input the whole page hangs off. Returns the current query text."""
    with st.form("research_search", border=False):
        cols = st.columns([5, 1], vertical_alignment="bottom")
        with cols[0]:
            st.text_input(
                "Company name or ticker",
                key=_QUERY_KEY,
                placeholder="Polycab India, RELIANCE.NS, Tata Consultancy...",
                help=(
                    "Any listed company the bundled universe or the provider knows. "
                    "Private companies file nothing public, so nothing can be analysed "
                    "for them -- the page says so rather than guessing."
                ),
            )
        with cols[1]:
            submitted = st.form_submit_button("Search", width="stretch")
    query = str(st.session_state.get(_QUERY_KEY, "") or "").strip()
    if submitted:
        # The button exists to give the Enter key and the mouse the same
        # meaning; the query itself already lives in session state.
        #
        # Drop the previous pick when the query actually changed. A form's
        # widgets only deliver their values on submit, so the eager version
        # of this could not live inside the form anyway -- the same
        # guarantee, reached at the only moment the script can see the new
        # text. Without it the radio would keep pointing at an index that
        # belongs to a different query's option list, which is how a reader
        # ends up reading a report about a company they did not ask for.
        previous = st.session_state.get(_LAST_QUERY_KEY)
        st.session_state[_LAST_QUERY_KEY] = query
        if previous != query:
            st.session_state.pop(_PICK_KEY, None)
    return query


def _pick_candidate(resolution: Any) -> Candidate | None:
    """Render the candidate list and return **only** what the reader chose.

    Returns ``None`` until an explicit selection is made, even when the
    resolution found exactly one company. That is not a usability oversight: a
    single candidate is not the same as a correct one, and an auto-pick would
    make the ambiguity the resolver just refused to paper over invisible again.
    """
    if not resolution.ok:
        if resolution.notices:
            ui.empty_state(
                "No company matched",
                " ".join(resolution.notices),
                icon="🔍",
                hint="Try a fuller name, or the exchange-qualified ticker (e.g. INFY.NS).",
            )
        return None

    if len(resolution.candidates) == 1:
        st.caption("One match. Confirm it below — nothing is selected until you do.")
    else:
        st.caption(
            f"{len(resolution.candidates)} matches. Short names are genuinely ambiguous, "
            "so pick the company you meant — the report is built for that listing only."
        )

    options = [c.label for c in resolution.candidates]
    chosen = st.radio(
        "Which company?",
        options,
        key=_PICK_KEY,
        index=None,
        label_visibility="collapsed",
    )
    if chosen is None:
        return None
    for candidate in resolution.candidates:
        if candidate.label == chosen:
            return candidate
    return None


def _search_online() -> bool:
    """Whether the provider may be asked. Offline-only skips every network call."""
    return bool(
        st.toggle(
            "Search the provider too",
            value=True,
            help=(
                "Off searches only the bundled universe, which is instant and works "
                "offline. On adds the provider's own lookup, which finds listings the "
                "bundle misses but is slower and can mis-resolve short names."
            ),
        )
    )


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------
def _load_company(symbol: str) -> dict[str, Any]:
    """Everything one report needs, in one place, each piece failing soft.

    Returns a dict rather than a dataclass on purpose: half of these can be
    missing independently, and a report that disappears because one provider
    call failed is worse than one that shows its statements with the
    market-derived rows blank.
    """
    profiles = data_access.load_profiles((symbol,))
    quote, quote_failures = data_access.load_quotes((symbol,))
    statements, statement_notice = data_access.load_statements(symbol)
    bars = data_access.load_ohlcv(symbol, _WINDOW)

    profile = profiles.get(symbol)
    price = _num(getattr(quote.get(symbol), "price", None))
    if math.isnan(price) and isinstance(bars, pd.DataFrame) and not bars.empty:
        # The daily bar is a day old at worst and the same series every other
        # calculation on the page uses, so it is a better fallback than a blank.
        price = _num(bars["Close"].dropna().iloc[-1]) if "Close" in bars else float("nan")

    currency = (
        str(getattr(profile, "currency", "") or "")
        or str(getattr(quote.get(symbol), "currency", "") or "")
        or ""
    )
    market_cap = _num(getattr(profile, "market_cap", None))

    read = fu.read_statements(
        balance_sheet=statements.get("balance_sheet"),
        income=statements.get("income"),
        cash_flow=statements.get("cash_flow"),
        ticker=symbol,
        currency=currency,
    )
    snapshot = co.snapshot(
        read,
        market_cap=None if math.isnan(market_cap) else market_cap,
        price=None if math.isnan(price) else price,
        country=str(getattr(profile, "country", "") or ""),
    )

    return {
        "symbol": symbol,
        "profile": profile,
        "quote": quote.get(symbol),
        "quote_failures": quote_failures,
        "statements": read,
        "statement_notice": statement_notice,
        "bars": bars,
        "price": price,
        "currency": currency,
        "market_cap": market_cap,
        "snapshot": snapshot,
    }


def _price_series(bars: object) -> tuple[tuple[date, float], ...]:
    """Chronological ``(day, close)`` pairs from the OHLCV frame, or ``()``.

    Kept as plain tuples rather than a Series because the sentiment measurement
    aligns headlines to the *next* close, and a (day, close) sequence makes that
    lookup explicit instead of hiding it behind an index.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty or "Close" not in bars.columns:
        return ()
    closes = pd.to_numeric(bars["Close"], errors="coerce")
    out: list[tuple[date, float]] = []
    for stamp, value in closes.items():
        value = _num(value)
        if math.isnan(value):
            continue
        day = stamp.date() if hasattr(stamp, "date") else None
        if day is not None:
            out.append((day, value))
    return tuple(out)


# ---------------------------------------------------------------------------
# peers
# ---------------------------------------------------------------------------
def _profile_metrics(profile: Any) -> dict[str, float]:
    """One peer's comparable metrics, read from the provider's own profile.

    Both sides of every comparison are read through this function, including
    the subject, and that is the point: a peer table that took the subject from
    its statements and the peers from their profiles would be comparing a
    fiscal-year figure against a trailing-twelve-month one and calling the
    difference a premium. The cost is that these rows disagree slightly with
    the statement-derived ratios elsewhere on the page, which the section says
    out loud.

    **``debt_to_equity`` is divided by 100 here.** The provider returns it as a
    percentage (``45.2`` means 0.452x) while every other ratio on the page is a
    decimal, and a comparison that mixed the two would report a company as forty
    times more levered than it is without raising anything.
    """
    if profile is None:
        return {}
    metrics = {
        "pe": _num(getattr(profile, "trailing_pe", None)),
        "forward_pe": _num(getattr(profile, "forward_pe", None)),
        "pb": _num(getattr(profile, "price_to_book", None)),
        "roe": _num(getattr(profile, "return_on_equity", None)),
        "net_margin": _num(getattr(profile, "profit_margin", None)),
        "revenue_growth": _num(getattr(profile, "revenue_growth", None)),
        "dividend_yield": _num(getattr(profile, "dividend_yield", None)),
    }
    leverage = _num(getattr(profile, "debt_to_equity", None))
    if not math.isnan(leverage):
        metrics["debt_to_equity"] = leverage / 100.0
    return {key: value for key, value in metrics.items() if not math.isnan(value)}


def _peer_pool(symbol: str, sector: str, country: str, limit: int) -> tuple[str, ...]:
    """Candidate peer symbols: same country, same sector, most prominent first.

    The bundled universe supplies the *candidate* list because it is offline and
    instant, and the provider supplies the labels the clustering fallback groups
    on. Note the taxonomies differ -- the CSV says ``"Capital Goods"`` where the
    provider says ``"Industrials"`` -- so this narrows the pool by the CSV's
    own sector and does not attempt to translate between the two.
    """
    from src import stock_universe as su

    records = su.load_universe()
    pool = [
        r
        for r in records
        if r.yahoo_symbol != symbol
        and not r.is_fund
        and r.instrument_type == "Equity"
        and (not country or r.country == country)
        and (not sector or r.sector == sector)
    ]
    pool.sort(key=lambda r: r.prominence, reverse=True)
    return tuple(r.yahoo_symbol for r in pool[:limit])


def _peer_comparison(snapshot: co.CompanySnapshot, profile: Any, count: int) -> pr.PeerComparison:
    """Build the comparison, or a comparison with no peers and a reason."""
    subject = pr.PeerCompany(
        symbol=snapshot.ticker,
        name=str(getattr(profile, "display_name", "") or snapshot.ticker),
        sector=str(getattr(profile, "sector", "") or ""),
        industry=str(getattr(profile, "industry", "") or ""),
        currency=snapshot.currency,
        metrics=_profile_metrics(profile),
    )

    pool = _peer_pool(
        snapshot.ticker,
        str(getattr(profile, "sector", "") or ""),
        str(getattr(profile, "country", "") or ""),
        count,
    )
    if not pool:
        return pr.relative_valuation(subject, ())

    profiles = data_access.load_profiles(pool)
    companies = [
        pr.PeerCompany(
            symbol=symbol,
            name=str(getattr(profiles.get(symbol), "display_name", "") or symbol),
            sector=str(getattr(profiles.get(symbol), "sector", "") or ""),
            industry=str(getattr(profiles.get(symbol), "industry", "") or ""),
            currency=str(getattr(profiles.get(symbol), "currency", "") or ""),
            metrics=_profile_metrics(profiles.get(symbol)),
        )
        for symbol in pool
    ]
    usable = [c for c in companies if c.metrics]
    return pr.relative_valuation(subject, usable)


def _cluster(comparison: pr.PeerComparison) -> pr.PeerCluster | None:
    """Group the peer set, reporting the method that actually produced it."""
    if len(comparison.peers) < 3:
        return None
    return pr.cluster_peers(comparison.subject, comparison.peers)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _header(data: dict[str, Any], candidate: Candidate) -> None:
    """The identity strip: what was picked, and what it is priced in."""
    profile = data["profile"]
    name = str(getattr(profile, "display_name", "") or candidate.name or data["symbol"])
    exchange = _text(getattr(profile, "exchange", None) or candidate.exchange)
    country = _text(getattr(profile, "country", None) or candidate.country)
    quote = data["quote"]
    as_of = getattr(quote, "as_of", None)
    provenance = fmt_date(as_of) if as_of is not None else "date unavailable"

    st.markdown(f"### {name}")
    ui.badges(
        [
            (data["symbol"], "accent"),
            (exchange, "neutral"),
            (country, "neutral"),
            (data["currency"] or DASH, "neutral"),
            (f"prices as of {provenance}", "neutral"),
        ]
    )
    summary = _text(getattr(profile, "summary", None))
    if summary != DASH:
        with st.expander("What the company does", expanded=False):
            st.markdown(summary)


def _snapshot_section(data: dict[str, Any]) -> None:
    """Levels and returns at a glance. The first thing a reader scans."""
    company: co.CompanySnapshot = data["snapshot"]
    currency = data["currency"]

    if not company.ok:
        ui.empty_state(
            "No annual statements returned",
            "The provider returned no usable income statement for this listing, so "
            "there is nothing to compute a snapshot from.",
            icon="📄",
            hint="Some exchange suffixes and funds carry no statements at all.",
        )
        return

    cards = [
        {"label": "Price", "value": _money(data["price"], currency)},
        {"label": "Market cap", "value": _compact(data["market_cap"], currency)},
        {"label": "Revenue (latest year)", "value": _compact(company.revenue, currency)},
        {"label": "Net income", "value": _compact(company.net_income, currency)},
    ]
    ui.kpi_row(cards)

    stats = [
        ("Revenue growth, 1y", _pct(company.revenue_cagr)),
        ("Revenue growth, 3y CAGR", _pct(company.revenue_cagr_3y)),
        ("Net margin", _pct(company.net_margin)),
        ("Return on equity", _pct(company.roe)),
        ("Return on invested capital", _pct(company.roic)),
        ("Free cash flow margin", _pct(company.fcf_margin)),
        ("Free cash flow yield", _pct(company.fcf_yield)),
        ("Debt / equity", fmt_multiple(company.debt_to_equity)),
        ("Net debt", _compact(company.net_debt, currency)),
        ("Interest cover", fmt_multiple(company.interest_cover)),
        ("Current ratio", fmt_multiple(company.current_ratio)),
        ("Cash conversion", _pct(company.cash_conversion)),
    ]
    ui.stat_grid(stats, columns=4)

    st.caption(
        f"Figures cover {company.span or 'the periods the provider returned'} "
        f"({company.n_periods} annual period(s)). Rates are computed from the "
        "reported statements, not taken from the provider's summary fields."
    )
    if company.notices:
        ui.warning_list(company.notices, "What the statements did not carry")


def _business_section(data: dict[str, Any]) -> None:
    """Sector, industry and scale, as the provider labels them."""
    profile = data["profile"]
    if profile is None:
        st.caption("No reference profile was returned for this listing.")
        return

    rows = [
        ("Sector", _text(getattr(profile, "sector", None))),
        ("Industry", _text(getattr(profile, "industry", None))),
        ("Country", _text(getattr(profile, "country", None))),
        ("Exchange", _text(getattr(profile, "exchange", None))),
        ("Instrument", _text(getattr(profile, "quote_type", None))),
        ("Market cap", _compact(getattr(profile, "market_cap", None), data["currency"])),
        ("Website", _text(getattr(profile, "website", None))),
    ]
    ui.stat_grid(rows, columns=3)
    st.caption(
        "The sector and industry labels are the provider's, and they do not match the "
        "coarser taxonomy in the bundled universe — a peer set selected on one is not "
        "the same set as one selected on the other."
    )


def _financials_section(data: dict[str, Any]) -> None:
    """The statements themselves: levels, then shape as common-size lines."""
    company: co.CompanySnapshot = data["snapshot"]
    statements: fu.Statements = data["statements"]
    currency = data["currency"]

    if not company.ok:
        st.caption("No statements to show.")
        return

    # `retained_series` is oldest-first and indexed by period end, while
    # `statements.periods` runs newest-first. Reading the labels off the series
    # index rather than off `periods` is what keeps the bars aligned with their
    # own years instead of labelling every bar with the wrong one.
    revenue_series = co.retained_series(statements, "revenue")
    profit_series = co.retained_series(statements, "net_income").reindex(revenue_series.index)
    if not revenue_series.empty:
        periods = [str(stamp)[:4] for stamp in revenue_series.index]
        trend = ch.financial_trend_chart(
            periods,
            list(revenue_series),
            list(profit_series),
            currency_symbol=currency_symbol(currency),
        )
        ui.chart(trend)
        st.caption(
            "Bars rather than lines: a company does not grow smoothly between one "
            "annual report and the next, and a line would imply it does."
        )

    size = co.common_size(statements, "income", max_periods=5)
    if not size.empty:
        st.markdown("**Income statement, as a share of revenue**")
        shape = ch.common_size_chart(size)
        ui.chart(shape)
        st.caption(
            "Every line is a share of its own year's revenue, so the *shape* of the "
            "business is comparable across years and across companies of any size."
        )

    growth = co.growth_table(statements, max_periods=5)
    if not growth.empty:
        st.markdown("**Growth by line**")
        ui.dataframe(growth, height=260)


def _quality_section(data: dict[str, Any]) -> None:
    """Ratios, then the four forensic scores and what each one refuses to say."""
    company: co.CompanySnapshot = data["snapshot"]
    if not company.ok:
        return

    st.markdown("**Ratios**")
    rows = co.ratio_rows(company)
    ui.stat_grid([(label, value) for label, value in rows], columns=3)

    quality = company.quality
    st.markdown("**Forensic scores**")
    fundamentals_panel.score_cards(quality)

    notes = []
    for score in quality.scores:
        if score.ok:
            notes.append(f"**{score.headline}** — {score.coverage}")
        else:
            notes.append(
                f"**{score.name}: not computed** — {score.coverage}. " + " ".join(score.notices)
            )
    st.markdown("  \n".join(notes))

    ui.methodology(
        "How these scores are read",
        "Each score answers a narrow, checkable question and none of them is a "
        "verdict.\n\n"
        "- **Piotroski F-Score** counts nine accounting tests of profitability, "
        "leverage and efficiency. It was built to separate winners from losers "
        "*within* a cheap subset of the market, and is weak outside it.\n"
        "- **Altman Z** is a bankruptcy model, and the version used here depends on "
        "whether the company is financial and which market it trades in.\n"
        "- **Beneish M-Score** flags the *statistical signature* of earnings "
        "manipulation. A high score is a prompt to read the filings, never a finding "
        "of fraud — a fast-growing company with heavy receivables scores like one.\n"
        "- **Accruals ratio** measures how much of reported profit is not cash. "
        "Persistently high accruals predict disappointing earnings on average, and the "
        "size of that effect is small.\n\n"
        "A score that could not be computed is reported as missing with its reason, "
        "never as a zero — a company with insufficient history is not a bad company.",
    )


def _valuation_section(data: dict[str, Any]) -> dict[str, float]:
    """The DCF, its sensitivity, the reverse DCF, and the peer multiples.

    Returns the raw values the scorecard needs, because the valuation is where
    those inputs are decided and passing them back is cheaper than recomputing
    a DCF inside the scorecard section.
    """
    company: co.CompanySnapshot = data["snapshot"]
    statements: fu.Statements = data["statements"]
    currency = data["currency"]
    profile = data["profile"]
    out = {"dcf_upside": float("nan"), "peer_value": float("nan")}

    if not company.ok:
        st.caption("No statements to value.")
        return out

    financials = dcf.CompanyFinancials(
        base_revenue=company.revenue,
        net_debt=company.net_debt,
        shares=company.shares,
        current_price=data["price"],
        base_ebitda=company.ebitda,
    )
    if not financials.ok:
        st.warning(
            "A discounted cash flow needs "
            + ", ".join(financials.missing)
            + ", and the statements did not carry them all. Nothing is valued rather "
            "than valuing a company with substituted zeros."
        )
        return out

    history_growth = co.revenue_growth_rates(statements)
    seed = dcf.default_assumptions(
        historical_growth=history_growth,
        historical_margin=(company.operating_margin,),
        years=5,
    )

    st.markdown("**Discounted cash flow**")
    st.caption(
        "Free cash flow to the firm, discounted at the cost of capital, with a "
        "perpetuity after an explicit forecast. Change any assumption and the "
        "answer moves — which is the honest behaviour, not a defect."
    )

    controls = st.columns(4)
    horizon = controls[0].slider("Forecast years", 3, 10, 5, key="dcf_years")
    start_growth = controls[1].slider(
        "Year-1 revenue growth",
        -20.0,
        60.0,
        round(float(seed.revenue_growth[0]) * 100.0, 1),
        step=0.5,
        format="%.1f%%",
        key="dcf_growth",
    )
    margin = controls[2].slider(
        "Operating margin",
        -10.0,
        60.0,
        round(float(company.operating_margin) * 100.0, 1)
        if not math.isnan(company.operating_margin)
        else 15.0,
        step=0.5,
        format="%.1f%%",
        key="dcf_margin",
    )
    terminal = controls[3].slider(
        "Terminal growth",
        -2.0,
        6.0,
        round(dcf.DEFAULT_TERMINAL_GROWTH * 100.0, 1),
        step=0.25,
        format="%.2f%%",
        key="dcf_terminal",
    )

    wacc_seed = _seed_wacc(profile)
    wacc_cols = st.columns([3, 5])
    wacc_pct = wacc_cols[0].slider(
        "Discount rate (WACC)",
        4.0,
        25.0,
        round(wacc_seed * 100.0, 1),
        step=0.25,
        format="%.2f%%",
        key="dcf_wacc",
        help=(
            "Cash flows are discounted at the weighted average cost of capital. "
            "Seeded from the provider's beta and a placeholder risk-free rate you "
            "should replace with the actual yield on the relevant government bond."
        ),
    )
    with wacc_cols[1]:
        st.caption(
            f"Seeded at {wacc_seed:.1%} from a placeholder risk-free rate plus "
            f"{_EQUITY_RISK_PREMIUM:.1%} equity risk premium times the provider's "
            f"beta ({_text(getattr(profile, 'beta', None))}). This is a starting "
            "point for your own view, not an estimate this tool stands behind."
        )

    assumptions = _assumptions(
        horizon=horizon,
        start_growth=start_growth / 100.0,
        margin=margin / 100.0,
        wacc=wacc_pct / 100.0,
        terminal_growth=terminal / 100.0,
    )
    result = dcf.run_dcf(assumptions, financials, effective_tax_rate=company.effective_tax_rate)

    if not result.ok:
        st.warning("The valuation did not resolve: " + "; ".join(result.notices))
        return out

    upside = (
        (result.value_per_share - data["price"]) / data["price"]
        if _num(data["price"]) > 0
        else float("nan")
    )
    out["dcf_upside"] = upside

    ui.kpi_row(
        [
            {"label": "Value per share", "value": _money(result.value_per_share, currency)},
            {"label": "Market price", "value": _money(data["price"], currency)},
            {
                "label": "Upside to fair value",
                "value": _pct(upside),
                "tone": _tone_for(upside),
            },
            {"label": "Enterprise value", "value": _compact(result.enterprise_value, currency)},
        ]
    )
    st.caption(
        f"{result.pv_terminal / result.enterprise_value:.0%} of that enterprise value "
        "sits in the perpetuity beyond the forecast, which is the part most sensitive "
        "to terminal growth and the part no model can evidence."
        if _num(result.pv_terminal) and _num(result.enterprise_value)
        else ""
    )
    if result.notices:
        ui.warning_list(result.notices, "Valuation notes")

    rows, cols = _sensitivity_ranges(wacc_pct / 100.0, terminal / 100.0)
    grid = dcf.sensitivity_grid(
        assumptions, financials, rows, cols, row_field="wacc_value", col_field="terminal_growth"
    )
    if grid.ok:
        st.markdown("**How much the answer depends on the two assumptions you cannot check**")
        heatmap = ch.sensitivity_heatmap(
            grid.rows,
            grid.cols,
            grid.values,
            row_title="WACC",
            col_title="terminal growth",
            highlight=(grid.base_row, grid.base_col) if grid.base_row >= 0 else None,
        )
        ui.chart(heatmap)
        if not math.isnan(grid.spread):
            st.caption(
                f"Across this grid the value per share ranges over "
                f"{grid.spread:.0%} of its lowest reading. The boxed cell is the "
                "assumption pair the headline number came from."
            )

    reverse = dcf.reverse_dcf(assumptions, financials, field_name="revenue_cagr")
    if reverse.ok:
        st.markdown("**What the market price implies**")
        st.markdown(
            f"At today's price of {_money(reverse.current_price, currency)}, this model "
            f"needs revenue to compound at **{_pct(reverse.implied_growth)} a year** for "
            "the whole forecast to be worth the market's own valuation. That is the "
            'falsifiable version of the question: not "is it cheap?" but "do I '
            'believe that growth rate?"'
        )
        if not math.isnan(company.revenue_cagr_3y):
            beat = reverse.implied_growth <= company.revenue_cagr_3y
            st.caption(
                f"The company actually compounded revenue at {_pct(company.revenue_cagr_3y)} "
                f"over the last three years, so the price implies a {'' if beat else 'faster'} "
                f"rate than it has been achieving."
                if beat
                else f"The company compounded revenue at only {_pct(company.revenue_cagr_3y)} "
                "over the last three years — the price is asking it to accelerate."
            )
        if reverse.notices:
            ui.warning_list(reverse.notices, "Reverse DCF notes")
    elif reverse.notices:
        ui.warning_list(reverse.notices, "Reverse DCF notes")

    return out


def _seed_wacc(profile: Any) -> float:
    """CAPM with a *placeholder* risk-free rate, purely to preselect a slider."""
    beta = _num(getattr(profile, "beta", None))
    country = str(getattr(profile, "country", "") or "")
    risk_free = _RISK_FREE_SEED.get(country)
    if risk_free is None or math.isnan(beta):
        return 0.12
    cost_of_equity = risk_free + beta * _EQUITY_RISK_PREMIUM
    return min(max(cost_of_equity, 0.05), 0.25)


def _assumptions(
    *,
    horizon: int,
    start_growth: float,
    margin: float,
    wacc: float,
    terminal_growth: float,
) -> dcf.DcfAssumptions:
    """Build the forecast from the reader's knobs, fading growth to the terminal rate.

    The fade rule matches :func:`src.analytics.dcf.default_assumptions`, which
    seeded the sliders — restated here rather than called, because that function
    clamps its input to the range of the *history* it was handed and would
    silently overrule a reader who deliberately chose a rate outside it.
    """
    n = max(int(horizon), 1)
    path = tuple(
        start_growth + (terminal_growth - start_growth) * (i / max(n - 1, 1)) for i in range(n)
    )
    return dcf.DcfAssumptions(
        revenue_growth=path,
        ebit_margin=(margin,) * n,
        wacc_value=wacc,
        terminal_growth=terminal_growth,
    )


def _sensitivity_ranges(wacc: float, terminal: float) -> tuple[list[float], list[float]]:
    """A five-by-five grid centred on the reader's own assumptions.

    Centred rather than wide: a grid spanning 6% to 20% WACC produces a wall of
    numbers whose corners describe companies nobody is valuing, and the reader
    cannot see the *gradient* that is the only thing the chart is for.
    """
    rows = [max(wacc + step, 0.01) for step in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    cols = [terminal + step for step in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    return rows, cols


def _peers_section(data: dict[str, Any]) -> None:
    """Relative valuation against companies the reader agrees are comparable."""
    company: co.CompanySnapshot = data["snapshot"]
    profile = data["profile"]
    if not company.ok or profile is None:
        st.caption("No company profile, so no peer set could be assembled.")
        return

    count = st.slider(
        "Peers compared",
        3,
        _MAX_PEERS,
        6,
        key="peer_count",
        help="Each peer costs a provider lookup. Beyond about six the median stops moving.",
    )
    with st.spinner("Gathering peer multiples..."):
        comparison = _peer_comparison(company, profile, count)

    if not comparison.peers:
        ui.empty_state(
            "No comparable companies found",
            "The bundled universe holds no other listed company in this sector and "
            "country, so there is nothing to compare against.",
            icon="👥",
            hint="Sector labels differ between the bundled universe and the provider.",
        )
        return

    rows = [
        {
            "label": c.spec.label,
            "relative": c.relative,
            "favourable": c.favourable,
            "subject_display": pr.format_metric(c.subject, c.spec.unit),
            "median_display": pr.format_metric(c.peer_median, c.spec.unit),
            "n_peers": c.n_peers,
        }
        for c in comparison.comparisons
    ]
    figure = ch.peer_median_chart(rows)
    ui.chart(figure)
    st.caption(
        "Bars are the gap to the peer median, coloured by whether that gap is "
        "*favourable on the metric's own terms* — above the median on return on "
        "equity is green, above it on P/E is red. The two are not the same "
        "direction, and a chart coloured by sign alone would say they were."
    )

    headers = ("Metric", "This company", "Peer median", "Gap", "Reads as")
    table = [headers]
    for c in comparison.comparisons:
        table.append(
            (
                c.spec.label,
                pr.format_metric(c.subject, c.spec.unit),
                pr.format_metric(c.peer_median, c.spec.unit),
                "—" if math.isnan(c.relative) else f"{c.relative:+.0%}",
                c.verdict,
            )
        )
    ui.dataframe(pd.DataFrame(table[1:], columns=list(table[0])), height=300)

    cluster = _cluster(comparison)
    if cluster is not None and cluster.ok:
        st.caption(cluster.caption)
        for notice in cluster.notices:
            st.caption(f"⚠ {notice}")
    if comparison.notices:
        ui.warning_list(comparison.notices, "Peer comparison notes")
    st.caption(
        "Both this company and its peers are read from the same provider fields here, so "
        "the comparison is like-for-like. That is also why these rows do not exactly "
        "match the statement-derived ratios above: the provider computes its own on a "
        "trailing basis, and mixing the two would turn a definitional difference into a "
        "premium."
    )


def _risk_section(data: dict[str, Any], bars: object) -> dict[str, float]:
    """Realised volatility and drawdown, both measured and both described."""
    out = {"volatility": float("nan"), "max_drawdown": float("nan")}
    series = _price_series(bars)
    if len(series) < 30:
        st.caption(
            "Fewer than thirty price observations were returned, which is too few "
            "for a risk statistic to mean anything."
        )
        return out

    prices = [value for _, value in series]
    regime = ml_signal.volatility_regime(prices)
    st.markdown(f"**{regime.reading}**")

    figure = ch.volatility_regime_chart(regime.series, current=regime.current, median=regime.median)
    ui.chart(figure)

    closes = pd.Series(prices)
    drawdown = closes / closes.cummax() - 1.0
    out["volatility"] = regime.current
    out["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else float("nan")

    ui.stat_grid(
        [
            ("Realised volatility", _pct(regime.current)),
            ("Median volatility", _pct(regime.median)),
            ("Worst drawdown in window", _pct(out["max_drawdown"])),
            ("Sessions observed", f"{regime.n_obs:,}"),
        ],
        columns=4,
    )
    st.caption(
        "Volatility is an exponentially-weighted standard deviation of daily log "
        "returns, annualised. It describes the last month against this company's own "
        "history — it is not a forecast, though volatility clusters, so a calm month "
        "does make a calm next month somewhat more likely."
    )
    for notice in regime.notices:
        st.caption(notice)
    return out


def _news_section(data: dict[str, Any], bars: object) -> dict[str, float]:
    """Headlines, their tone, and whether that tone has predicted anything."""
    symbol = data["symbol"]
    profile = data["profile"]
    name = str(getattr(profile, "display_name", "") or "")
    out = {"sentiment": float("nan")}

    with st.spinner("Reading recent coverage..."):
        items, notice = data_access.load_company_news(symbol, name, limit=_NEWS_LIMIT)

    if not items:
        ui.empty_state(
            "No recent coverage found",
            notice or "The news feed returned nothing for this company.",
            icon="📰",
            hint="The feed is a Google News search; thinly covered companies return little.",
        )
        return out

    observations = sen.score_headlines(items)
    st.markdown(f"**{sen.tone_reading(observations)}**")

    directional = [o for o in observations if o.directional]
    if directional:
        mean = sum(o.score for o in directional) / len(directional)
        out["sentiment"] = mean
        ui.stat_grid(
            [
                ("Headlines read", f"{len(observations):,}"),
                ("Of which the lexicon scored", f"{len(directional):,}"),
                ("Mean tone of scored items", f"{mean:+.2f}"),
                ("Days covered", f"{len({o.day for o in observations if o.day}):,}"),
            ],
            columns=4,
        )

    st.markdown("**Does the tone of coverage predict anything?**")
    turning = sen.sentiment_turning(observations, _price_series(bars), horizon_days=20)
    st.markdown(turning.verdict)
    if turning.usable:
        figure = ch.sentiment_scatter(
            turning.clusters, rank_ic=turning.rank_ic, horizon=turning.horizon_days
        )
        ui.chart(figure)
    for line in turning.notices:
        st.caption(line)
    ui.methodology(
        "How the tone score is built, and why the measurement is usually nothing",
        "**The lexicon is ours, not a pretrained model.** No NLP library is installed "
        "and a finance-specific bag of words with negation and intensifier windows is "
        "both more transparent and more accurate on headlines than a general sentiment "
        'model scoring "profit warning" as neutral.\n\n'
        "**Headlines are collapsed to one score per day before the correlation is "
        "computed.** Twelve headlines on one day are one observation about that day's "
        "coverage, not twelve independent ones; counting them separately is the "
        "standard way a sentiment backtest manufactures significance.\n\n"
        "**Only the return *after* each headline day is used.** Otherwise a positive "
        "reading can come from coverage that merely reported a move that had already "
        "happened.\n\n"
        "**Read the rank correlation as a hypothesis, not a finding.** A retail-sized "
        "sample of a few dozen days will produce a number in either direction, and the "
        "page says when that number is indistinguishable from no relationship, which "
        "is the usual answer.",
    )

    with st.expander(f"Headlines ({len(observations)})", expanded=False):
        for observation in observations[:40]:
            published = fmt_date(observation.published) if observation.published else DASH
            terms = ", ".join(observation.matched) if observation.matched else "no signal term"
            st.markdown(
                f"`{observation.score:+.2f}` **{observation.title}**  \n"
                f"<span style='opacity:.7'>{published} · {observation.label} · matched: "
                f"{terms}</span>",
                unsafe_allow_html=True,
            )
    return out


def _ml_section(data: dict[str, Any], bars: object) -> None:
    """The modelling. Ordered safest claim first, and labelled as such."""
    symbol = data["symbol"]

    st.markdown("**Forward-return model — the least safe claim here**")
    st.caption(
        "A walk-forward model on price- and volume-derived features, validated on "
        "folds it was never trained on. The output is the measured out-of-sample "
        "rank correlation, never a predicted price: a forecast presented without its "
        "own track record is a number with no evidence attached."
    )
    with st.spinner("Fitting and validating..."):
        signal = ml_signal.forward_return_signal(
            bars, horizon=20, model="ridge", with_importance=True
        )

    if not signal.measured:
        st.markdown(signal.verdict)
        for notice in signal.notices:
            st.caption(notice)
    else:
        tone = "warn" if signal.indistinguishable_from_noise else "gain"
        ui.kpi_row(
            [
                {
                    "label": "Out-of-sample rank IC",
                    "value": f"{signal.rank_ic:+.3f}",
                    "tone": tone,
                },
                {"label": "Fold spread", "value": fmt_num(signal.ic_spread, 3)},
                {"label": "Folds", "value": f"{signal.n_folds}"},
                {"label": "Observations", "value": f"{signal.n_obs:,}"},
            ],
            columns=4,
        )
        st.markdown(signal.verdict)
        if signal.importance:
            importance = pd.DataFrame(
                [{"feature": name, "importance": value} for name, value in signal.importance]
            )
            figure = ch.feature_importance_chart(importance)
            ui.chart(figure)
            st.caption(
                "Permutation importance on the final fold: how much the score "
                "degrades when each feature is shuffled. It says which inputs the "
                "model leaned on, not which ones carry a real relationship."
            )
        for notice in signal.notices:
            st.caption(notice)
    st.info(signal.honesty_note, icon="⚖️")

    st.divider()
    st.markdown("**Forensic outliers against the peer set**")
    st.caption(
        "An isolation forest across the peer companies' accounts. It reports *which* "
        'inputs are unusual, because "this company is unusual" is unactionable while '
        '"its debt-to-equity is 4.2x against a peer median of 0.8x" can be opened in '
        "the annual report and checked."
    )

    profile = data["profile"]
    company: co.CompanySnapshot = data["snapshot"]
    if profile is None or not company.ok or not symbol:
        st.caption("A peer set is needed for this and none could be assembled.")
        return

    pool = _peer_pool(
        symbol,
        str(getattr(profile, "sector", "") or ""),
        str(getattr(profile, "country", "") or ""),
        _MAX_PEERS,
    )
    if not pool:
        st.caption("No peers in the same sector and country to compare against.")
        return

    with st.spinner("Scoring outliers..."):
        profiles = data_access.load_profiles(pool)
        subject = co.peer_metrics(company)
        peer_metrics = [_profile_metrics(profiles.get(s)) for s in pool]
        score = ml_signal.forensic_outliers(subject, [m for m in peer_metrics if m], symbol=symbol)
    st.markdown(score.reading)
    for notice in score.notices:
        st.caption(notice)


def _verdict_section(
    data: dict[str, Any], valuation: dict[str, float], risk: dict[str, float]
) -> None:
    """The transparent scorecard. Every input, weight and pull, visible."""
    company: co.CompanySnapshot = data["snapshot"]
    if not company.ok:
        st.caption("The verdict needs statements, and none were returned.")
        return

    preset = st.radio(
        "Weighting",
        list(sc.WEIGHT_PRESETS),
        index=0,
        horizontal=True,
        key="scorecard_preset",
        help=(
            "The verdict is a weighted composite, so the weights *are* the opinion. "
            "Changing them changes the answer, and that is the point: an opaque score "
            "would be unfalsifiable."
        ),
    )

    inputs = co.scorecard_inputs(
        company,
        dcf_upside=valuation.get("dcf_upside", float("nan")),
        peer_value=valuation.get("peer_value", float("nan")),
        sentiment=risk.get("sentiment", float("nan")),
        volatility=risk.get("volatility", float("nan")),
        max_drawdown=risk.get("max_drawdown", float("nan")),
    )
    card = sc.build_scorecard(inputs, sc.WEIGHT_PRESETS[preset])

    if not card.ok:
        ui.empty_state(
            "Not enough inputs to issue a view",
            f"Only {card.coverage:.0%} of the requested weight could be scored, against "
            f"{sc.MIN_COVERAGE:.0%} needed before a band is issued. A verdict built on a "
            "handful of inputs would be more confident than the evidence.",
            icon="⚖️",
        )
        for notice in card.notices:
            st.caption(notice)
        return

    ui.verdict(
        card.verdict,
        f"Composite {card.composite:.3f} on a 0–1 scale, from {len(card.scored)} of "
        f"{len(card.factors)} inputs ({card.coverage:.0%} of the requested weight). "
        + (
            "The inputs disagree with each other, which is itself the finding."
            if card.contested
            else "The inputs broadly agree."
        ),
        tone=_VERDICT_TONE.get(card.verdict, "neutral"),
        evidence="Educational analysis, not investment advice.",
    )

    rows = [
        {
            "label": f.spec.label,
            "influence": f.influence,
            "score": f.score,
            "display": f.spec.display(f.raw),
            "effective_weight": f.effective_weight,
        }
        for f in card.scored
    ]
    figure = ch.scorecard_breakdown_chart(rows, composite=card.composite)
    ui.chart(figure)
    st.caption(
        "Bars are how hard each input *pulled* the composite away from neutral, "
        "not how many points it contributed. A maximally adverse input contributes "
        "nothing while being the reason the verdict reads as it does, and a chart "
        "of contributions would draw that objection as an empty bar."
    )

    with st.expander("Every input, its value, its weight and how it scored"):
        table = [("Input", "Group", "Value", "Score", "Weight", "Pulls")]
        for factor in card.ranked:
            table.append(
                (
                    factor.spec.label,
                    factor.spec.group,
                    factor.spec.display(factor.raw),
                    fmt_num(factor.score, 2),
                    f"{factor.effective_weight:.0%}",
                    factor.reading,
                )
            )
        ui.dataframe(pd.DataFrame(table[1:], columns=list(table[0])), height=420)

    if card.excluded:
        st.caption(
            "Excluded for want of data: "
            + ", ".join(f.spec.label for f in card.excluded)
            + ". Those inputs carry no weight rather than scoring zero."
        )
    for notice in card.notices:
        st.caption(notice)

    ui.methodology(
        "How the composite is built",
        "Each input is mapped onto a 0–1 scale by piecewise-linear interpolation "
        "between stated anchors, then weighted and summed. The anchors are judgements "
        "written out in full in `src/analytics/scorecard.py`, so a reader can disagree "
        "with the *scale* rather than only with the answer.\n\n"
        "Weights are renormalised over the inputs that could actually be scored, so a "
        "missing figure lowers *coverage* rather than dragging the composite toward "
        "zero. Below "
        f"{sc.MIN_COVERAGE:.0%} coverage no band is issued at all.\n\n"
        "The bands are composite thresholds — Avoid below 0.25, Reduce to 0.40, Hold to "
        "0.60, Buy to 0.75, Strong buy above — and they are a reporting convention, not "
        "an expected return.",
    )


def _risk_factors(data: dict[str, Any], risk: dict[str, float]) -> None:
    """The objections, written from the numbers rather than invented.

    Every line is a restatement of something already computed above. Nothing
    here is a scenario the data does not support, because a risk section that
    invents a plausible-sounding worry is worse than an empty one: it teaches
    the reader to trust a section that has no measurement behind it.
    """
    company: co.CompanySnapshot = data["snapshot"]
    factors: list[tuple[str, str]] = []

    def add(severity: str, text: str) -> None:
        factors.append((severity, text))

    if _num(company.debt_to_equity) > 1.5:
        add(
            "high",
            f"Leverage: debt is {company.debt_to_equity:.1f}x equity. Debt amplifies "
            "both directions, and it is the one input management cannot argue with.",
        )
    if _num(company.interest_cover) < 3.0 and not math.isnan(company.interest_cover):
        add(
            "high",
            f"Interest cover of {company.interest_cover:.1f}x leaves little room for a "
            "weak year before operating profit stops covering the interest bill.",
        )
    if _num(company.fcf_margin) < 0:
        add(
            "high", "Free cash flow is negative — the business is consuming cash, not producing it."
        )
    if _num(company.cash_conversion) < 0.7 and not math.isnan(company.cash_conversion):
        add(
            "medium",
            f"Cash conversion of {company.cash_conversion:.0%}: reported profit is not "
            "arriving as cash, which is where accrual problems show up first.",
        )
    quality = company.quality
    if quality.beneish.ok and _num(quality.beneish.value) > -1.78:
        add(
            "medium",
            f"Beneish M-Score of {quality.beneish.value:.2f} sits above the −1.78 "
            "threshold that flags a manipulation *signature*. That is a prompt to read "
            "the accounting notes, not a finding of misconduct.",
        )
    if quality.altman.ok and _num(quality.altman.value) < 1.81:
        add("high", f"Altman Z of {quality.altman.value:.2f} is inside the distress zone.")
    if quality.piotroski.ok and _num(quality.piotroski.value) <= 3:
        add(
            "medium",
            f"Piotroski F-Score of {quality.piotroski.value:.0f}/9: the profitability and "
            "efficiency tests are mostly failing.",
        )
    if _num(risk.get("max_drawdown")) < -0.40:
        add(
            "medium",
            f"The price fell {abs(risk['max_drawdown']):.0%} peak-to-trough within the "
            "window examined — this is a name that can lose nearly half its value "
            "without anything breaking.",
        )
    if _num(company.revenue_cagr_3y) < 0 and not math.isnan(company.revenue_cagr_3y):
        add(
            "medium",
            f"Revenue has shrunk at {company.revenue_cagr_3y:.1%} a year over three years.",
        )
    if math.isnan(company.net_margin) or math.isnan(company.roe):
        add(
            "unknown",
            "Profitability could not be established from the statements returned, so "
            "the quality side of this report is incomplete rather than poor.",
        )
    prompt = ", ".join(company.missing)
    if prompt:
        add("unknown", f"A valuation could not be completed: {prompt} were not reported.")

    if not factors:
        st.caption(
            "None of the mechanical checks on this page fired: leverage, coverage, cash "
            "conversion and the forensic scores all sit in ordinary ranges. That is a "
            "statement about the *accounts*, and says nothing about competition, "
            "regulation, management or valuation risk, none of which this tool measures."
        )
        return

    st.markdown(
        "\n\n".join(
            f"{ui.badge(severity.upper(), 'loss' if severity == 'high' else 'warn')} {text}"
            if severity != "unknown"
            else f"{ui.badge('UNKNOWN', 'muted')} {text}"
            for severity, text in factors
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "These are mechanical restatements of the measurements above. The risks that "
        "actually decide whether a company is a good investment — competitive position, "
        "management, regulation, the industry's economics — are not measured anywhere "
        "in this tool, and their absence here is not evidence that they are absent."
    )


def _method_section(data: dict[str, Any], candidate: Candidate) -> None:
    """Provenance, method, and the limits, all in one place a reader can check."""
    company: co.CompanySnapshot = data["snapshot"]
    metadata = company.as_metadata()

    ui.methodology(
        "Where every number came from",
        f"**Prices, profiles and statements:** {DATA_SOURCE}. Delayed or "
        "latest-available, and not guaranteed real-time.\n\n"
        f"**Identity:** resolved from the query *{candidate.name or candidate.symbol}* to "
        f"`{metadata.get('ticker', '—')}` — matched on {candidate.matched_on or 'the provider lookup'} "
        f"from the {candidate.source} source. The report is for that listing only; the "
        "same company listed elsewhere is a different instrument in a different "
        "currency.\n\n"
        f"**Statements cover:** {metadata.get('span') or 'the periods returned'} "
        f"({metadata.get('periods', 0)} annual period(s)). Annual data is restated by "
        "the provider, so the figures here are as filed *now*, not as first reported — "
        "which is why the ML section below uses price features rather than statement "
        "features for anything it calls out-of-sample.\n\n"
        f"**News:** a Google News search scoped to this company, scored by the "
        f"in-house finance lexicon in `src/analytics/news.py`.\n\n"
        f"**Peers:** other companies the bundled universe files under the same sector "
        f"and country, ranked by traded value.\n\n"
        "**Not fetched from anywhere:** the risk-free rate behind the discount-rate "
        "seed and the equity risk premium. Both are stated above as placeholders you "
        "are expected to replace.",
        expanded=True,
    )

    st.warning(
        f"**{DISCLAIMER}** This report is generated from third-party data that this "
        "tool does not verify, on models whose assumptions are visible above precisely "
        "because a reader should be able to disagree with them. It is a starting point "
        "for research, not a substitute for it.",
        icon="⚠️",
    )

    cols = st.columns([1, 1])
    if cols[0].button("Set as the app's active ticker", width="stretch"):
        session.select_ticker(data["symbol"])
        st.success(f"{data['symbol']} is now the active ticker on the other pages.")
    cols[1].caption(
        "Every figure on this page is recomputed from the analysis modules rather than "
        "stored, so re-running the page after a price move gives the current answer "
        "rather than a cached one."
    )


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def _report(candidate: Candidate) -> None:
    """Everything below the search box, for one confirmed company."""
    with st.spinner(f"Building the report for {candidate.name or candidate.symbol}..."):
        data = _load_company(candidate.symbol)

    profile = data["profile"]
    if getattr(profile, "is_fund", False):
        ui.empty_state(
            "This is a fund, not an operating company",
            f"The provider reports {candidate.symbol} as a "
            f"{_text(getattr(profile, 'quote_type', None))}, so there are no income "
            "statement and balance sheet accounts to analyse. Everything on this page "
            "is built from company statements.",
            icon="📦",
            hint="Try a listed operating company instead.",
        )
        return

    if not data["snapshot"].ok and not data["statements"].ok:
        ui.empty_state(
            "Nothing to analyse for this listing",
            data["statement_notice"]
            or "The provider returned no usable statements and no market data for this symbol.",
            icon="🚫",
            hint=(
                "Private companies file nothing public, so this page cannot report on "
                "them. Some thinly traded listings and funds also carry no statements."
            ),
        )
        return

    _header(data, candidate)
    _snapshot_section(data)

    st.divider()
    st.markdown("## Business and industry")
    _business_section(data)

    st.divider()
    st.markdown("## Financial statements")
    _financials_section(data)

    st.divider()
    st.markdown("## Ratios and quality")
    _quality_section(data)

    st.divider()
    st.markdown("## Valuation")
    valuation = _valuation_section(data)

    st.divider()
    st.markdown("## Peer comparison")
    _peers_section(data)

    st.divider()
    st.markdown("## Risk")
    bars = data["bars"]
    risk = _risk_section(data, bars)

    st.divider()
    st.markdown("## News and sentiment")
    risk.update(_news_section(data, bars))

    st.divider()
    st.markdown("## Machine learning")
    _ml_section(data, bars)

    st.divider()
    st.markdown("## The verdict")
    _verdict_section(data, valuation, risk)

    st.divider()
    st.markdown("## Risk factors")
    _risk_factors(data, risk)

    st.divider()
    st.markdown("## Method and sources")
    _method_section(data, candidate)

    if data["quote_failures"]:
        ui.warning_list(list(data["quote_failures"].values()), "Quote failures")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the company research page. Registered in ``app.py``."""
    ui_theme.page_header(
        "Company research",
        "Type a company name and read a full report on it",
        icon="🔎",
    )
    st.caption(
        "A report on a company you do not have to own. Financial, valuation, peer, "
        "risk, news and modelling sections, and a verdict whose every input and weight "
        "is on the page — because an opinion you cannot interrogate is just a rumour "
        "with a chart."
    )

    online = _search_online()
    query = _search_box()
    if not query:
        ui.empty_state(
            "Search for a company",
            "Enter a name or ticker above. Any listed company works — the report is "
            "not restricted to what the portfolio holds.",
            icon="🔎",
            hint="Private and unlisted companies file nothing public, so there is no data to report on.",
        )
        return

    with st.spinner("Resolving the name..."):
        resolution = data_access.resolve_company(query, online=online)

    candidate = _pick_candidate(resolution)
    if candidate is None:
        for notice in resolution.notices:
            st.caption(notice)
        return

    _report(candidate)
