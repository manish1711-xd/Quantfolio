"""One company's statements, assembled into the shapes the report needs.

:mod:`src.analytics.fundamentals` already computes the ratios, the Piotroski
and Altman scores and the Beneish index. This module is the *adapter* between
that engine and the three consumers downstream -- the DCF, the peer table and
the scorecard -- so that each of them receives correctly-named, correctly-
scaled inputs instead of each reaching into the raw statement frames itself.

It exists mainly to hold one class of bug in one place. **The engine and the
scorecard disagree about units in at least one place, and the disagreement is
silent.** ``accruals_ratio`` is returned in *percent* (``-5.56`` means -5.6%)
while every other rate in this package is a decimal (``0.095`` means 9.5%),
and a scorecard fed ``-5.56`` where it expects ``-0.0556`` does not error -- it
clamps to the top of the scale and reports the best possible score. So the unit
conversion happens exactly once, here, is written down per field, and is
asserted in the tests.

The other job is refusing to invent. Every field is NaN when the statements do
not carry it, and the derived growth rates need two real periods before they
exist at all: a CAGR computed from a single year is not a slow estimate, it is a
fabrication, and the reader would have no way to tell.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.analytics import fundamentals as fu

__all__ = [
    "CompanySnapshot",
    "snapshot",
    "revenue_growth_rates",
    "common_size",
    "growth_table",
    "peer_metrics",
    "scorecard_inputs",
    "ratio_rows",
    "retained_series",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Coerce to a finite float, or NaN."""
    if value is None:
        return _NAN
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return number if math.isfinite(number) else _NAN


def _div(numerator: object, denominator: object) -> float:
    """NaN-safe division. A zero or missing denominator is missing, not infinite."""
    top, bottom = _num(numerator), _num(denominator)
    if math.isnan(top) or math.isnan(bottom) or bottom == 0:
        return _NAN
    out = top / bottom
    return out if math.isfinite(out) else _NAN


def _cagr(series: Sequence[float], years: int) -> float:
    """Compound growth from the first to the last of ``series``.

    Refuses when either end is non-positive. A series that starts at -5 and ends
    at +10 has no compound growth rate, and the ``((end/start) ** (1/n)) - 1``
    formula would either raise or return a complex number; reporting NaN and
    letting the page print an em-dash is the only honest answer.
    """
    if years < 1 or len(series) < 2:
        return _NAN
    start, end = _num(series[0]), _num(series[-1])
    if math.isnan(start) or math.isnan(end) or start <= 0 or end <= 0:
        return _NAN
    span = min(years, len(series) - 1)
    return (end / start) ** (1.0 / span) - 1.0


def _series_of(statements: fu.Statements, field_name: str) -> list[float]:
    """One field across every period, oldest first, as plain floats."""
    frame = statements.series(field_name)
    if not isinstance(frame, pd.Series):
        return []
    return [_num(v) for v in frame.tolist()]


def _latest(statements: fu.Statements, field_name: str) -> float:
    values = _series_of(statements, field_name)
    return values[-1] if values else _NAN


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CompanySnapshot:
    """Everything about one company that the report needs, from its statements.

    Deliberately flat and pre-computed: the page, the PDF and the tests all read
    the same fields, so a definition of "net debt" cannot drift between three
    renderers. Values are NaN when the statements did not carry them.
    """

    ticker: str = ""
    currency: str = ""
    span: str = ""
    n_periods: int = 0

    # levels, latest period
    revenue: float = _NAN
    gross_profit: float = _NAN
    ebit: float = _NAN
    net_income: float = _NAN
    operating_cash_flow: float = _NAN
    free_cash_flow: float = _NAN
    capex: float = _NAN
    depreciation: float = _NAN

    # balance sheet
    total_assets: float = _NAN
    total_debt: float = _NAN
    cash: float = _NAN
    equity: float = _NAN
    shares: float = _NAN
    net_debt: float = _NAN

    # returns and margins, latest period
    roe: float = _NAN
    roic: float = _NAN
    roa: float = _NAN
    gross_margin: float = _NAN
    operating_margin: float = _NAN
    net_margin: float = _NAN
    ebitda_margin: float = _NAN
    fcf_margin: float = _NAN
    debt_to_equity: float = _NAN
    interest_cover: float = _NAN
    current_ratio: float = _NAN
    asset_turnover: float = _NAN
    cash_conversion: float = _NAN
    capex_to_revenue: float = _NAN
    effective_tax_rate: float = _NAN

    # growth, per year
    revenue_cagr: float = _NAN
    revenue_cagr_3y: float = _NAN
    earnings_cagr: float = _NAN

    # market-derived, supplied by the caller rather than the statements
    market_cap: float = _NAN
    price: float = _NAN
    fcf_yield: float = _NAN
    earnings_yield: float = _NAN
    book_value_per_share: float = _NAN
    ebitda: float = _NAN

    # scores
    quality: fu.QualityReport = field(default_factory=fu.QualityReport)
    revenue_series: tuple[float, ...] = ()
    net_income_series: tuple[float, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when there is enough here to value the business."""
        return not math.isnan(self.revenue) and self.revenue > 0

    @property
    def missing(self) -> tuple[str, ...]:
        """Human names of the inputs a valuation needs and does not have."""
        gaps = []
        if math.isnan(self.revenue) or self.revenue <= 0:
            gaps.append("revenue")
        if math.isnan(self.net_debt):
            gaps.append("net debt")
        if math.isnan(self.shares) or self.shares <= 0:
            gaps.append("share count")
        return tuple(gaps)

    def as_metadata(self) -> dict[str, object]:
        """The identifying header, so every renderer states the same facts."""
        return {
            "ticker": self.ticker,
            "currency": self.currency,
            "span": self.span,
            "periods": self.n_periods,
        }


def snapshot(
    statements: fu.Statements,
    market_cap: float | None = None,
    price: float | None = None,
    country: str = "",
) -> CompanySnapshot:
    """Assemble a snapshot from annual statements.

    ``market_cap`` and ``price`` come from the quote and profile, not from the
    statements; they are optional because the page can still show the financial
    half of the report when the market data call fails, and a report that
    vanishes entirely because a price was missing is worse than one that shows
    its statements with the market-derived rows blank.
    """
    if not isinstance(statements, fu.Statements) or not statements.ok:
        return CompanySnapshot(
            ticker=getattr(statements, "ticker", ""),
            notices=("No annual statements were available for this company.",),
        )

    notices: list[str] = list(statements.notices)
    ratio_set = fu.ratios(statements)
    ratios: Mapping[str, float] = ratio_set.values
    quality = fu.quality_summary(statements, market_cap=market_cap, country=country)

    revenue_series = tuple(_series_of(statements, "revenue"))
    income_series = tuple(_series_of(statements, "net_income"))
    n = statements.n_periods

    revenue = _latest(statements, "revenue")
    ebit = _latest(statements, "ebit")
    ebitda = _latest(statements, "ebitda")
    depreciation = _latest(statements, "depreciation")
    if math.isnan(ebitda) and not math.isnan(ebit) and not math.isnan(depreciation):
        # Fall back to EBIT + D&A only when the provider omitted the line
        # outright, and only when both parts are present.
        ebitda = ebit + depreciation

    total_debt = _latest(statements, "total_debt")
    cash = _latest(statements, "cash")
    net_debt = total_debt - cash if not (math.isnan(total_debt) or math.isnan(cash)) else _NAN

    shares = _latest(statements, "shares")
    equity = _latest(statements, "equity")

    ebitda_margin = _div(ebitda, revenue)
    fcf = _latest(statements, "free_cash_flow")
    market = _num(market_cap)

    booked = market if not math.isnan(market) else _NAN
    fcf_yield = _div(fcf, booked)
    earnings_yield = _div(_latest(statements, "net_income"), booked)
    book_per_share = _div(equity, shares)

    two_year_growth = _cagr(revenue_series, 1)
    three_year_growth = _cagr(revenue_series[-4:], 3) if len(revenue_series) >= 4 else _NAN

    if n < 2:
        notices.append(
            "Only one annual period is available, so no growth rate can be "
            "computed — a single year has no trend in it."
        )

    return CompanySnapshot(
        ticker=statements.ticker,
        currency=statements.currency,
        span=statements.span,
        n_periods=n,
        revenue=revenue,
        gross_profit=_latest(statements, "gross_profit"),
        ebit=ebit,
        net_income=_latest(statements, "net_income"),
        operating_cash_flow=_latest(statements, "operating_cash_flow"),
        free_cash_flow=fcf,
        capex=_latest(statements, "capex"),
        depreciation=depreciation,
        total_assets=_latest(statements, "total_assets"),
        total_debt=total_debt,
        cash=cash,
        equity=equity,
        shares=shares,
        net_debt=net_debt,
        roe=_num(ratios.get("roe")),
        roic=_num(ratios.get("roic")),
        roa=_num(ratios.get("roa")),
        gross_margin=_num(ratios.get("gross_margin")),
        operating_margin=_num(ratios.get("operating_margin")),
        net_margin=_num(ratios.get("net_margin")),
        ebitda_margin=ebitda_margin,
        fcf_margin=_num(ratios.get("fcf_margin")),
        debt_to_equity=_num(ratios.get("debt_to_equity")),
        interest_cover=_num(ratios.get("interest_cover")),
        current_ratio=_num(ratios.get("current_ratio")),
        asset_turnover=_num(ratios.get("asset_turnover")),
        cash_conversion=_num(ratios.get("cash_conversion")),
        capex_to_revenue=_num(ratios.get("capex_to_revenue")),
        effective_tax_rate=_num(ratios.get("effective_tax_rate")),
        revenue_cagr=two_year_growth if not math.isnan(two_year_growth) else _NAN,
        revenue_cagr_3y=three_year_growth,
        earnings_cagr=_cagr(income_series, 1),
        market_cap=booked,
        price=_num(price),
        fcf_yield=fcf_yield,
        earnings_yield=earnings_yield,
        book_value_per_share=book_per_share,
        ebitda=ebitda,
        quality=quality,
        revenue_series=revenue_series,
        net_income_series=income_series,
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# growth and common-size tables
# ---------------------------------------------------------------------------
def revenue_growth_rates(
    statements: fu.Statements, field_name: str = "revenue"
) -> tuple[float, ...]:
    """Year-over-year growth rates, oldest first. Empty when there is no trend.

    Delegates to :func:`src.analytics.dcf.historical_rates` so the definition of
    "historical growth" is one function rather than two that agree today.
    """
    from src.analytics import dcf

    return dcf.historical_rates(statements, field_name)


#: Lines shown in the common-size income statement, in reading order.
_COMMON_SIZE_INCOME: tuple[tuple[str, str], ...] = (
    ("revenue", "Revenue"),
    ("gross_profit", "Gross profit"),
    ("operating_income", "Operating income"),
    ("ebit", "EBIT"),
    ("net_income", "Net income"),
    ("operating_cash_flow", "Operating cash flow"),
    ("capex", "Capital expenditure"),
    ("free_cash_flow", "Free cash flow"),
)

#: Balance-sheet lines, expressed against total assets.
_COMMON_SIZE_BALANCE: tuple[tuple[str, str], ...] = (
    ("total_assets", "Total assets"),
    ("current_assets", "Current assets"),
    ("receivables", "Receivables"),
    ("inventory", "Inventory"),
    ("net_ppe", "Net fixed assets"),
    ("current_liabilities", "Current liabilities"),
    ("total_debt", "Total debt"),
    ("cash", "Cash and equivalents"),
    ("equity", "Shareholders' equity"),
)


def common_size(
    statements: fu.Statements,
    which: str = "income",
    max_periods: int = 5,
) -> pd.DataFrame:
    """Statements restated as percentages, so companies of any size compare.

    Income lines are a share of revenue and balance-sheet lines a share of total
    assets -- the standard bases, and the reason a small company's 12% margin
    can be read directly against a large one's. A line whose basis is missing
    stays NaN rather than becoming 0%, which would read as "this company has no
    fixed assets" instead of "this line was not reported".
    """
    layout = _COMMON_SIZE_INCOME if which == "income" else _COMMON_SIZE_BALANCE
    basis_field = "revenue" if which == "income" else "total_assets"

    if not isinstance(statements, fu.Statements) or not statements.ok:
        return pd.DataFrame()

    basis = statements.series(basis_field)
    if not isinstance(basis, pd.Series) or basis.empty:
        return pd.DataFrame()

    # `series` returns oldest first; the report reads newest first.
    columns = list(reversed(basis.index.tolist()))[:max_periods]
    data: dict[object, list[float]] = {}
    for column in columns:
        values = []
        denominator = _num(basis.get(column))
        for field_name, _ in layout:
            line = statements.series(field_name)
            raw = _num(line.get(column)) if isinstance(line, pd.Series) else _NAN
            values.append(_div(raw, denominator))
        data[column] = values

    return pd.DataFrame(data, index=[label for _, label in layout])


def growth_table(statements: fu.Statements, max_periods: int = 5) -> pd.DataFrame:
    """Levels and year-on-year growth for the lines a reader tracks, newest first."""
    if not isinstance(statements, fu.Statements) or not statements.ok:
        return pd.DataFrame()

    rows: dict[str, list[float]] = {}
    for field_name, label in (
        ("revenue", "Revenue"),
        ("ebit", "EBIT"),
        ("net_income", "Net income"),
        ("operating_cash_flow", "Operating cash flow"),
        ("free_cash_flow", "Free cash flow"),
    ):
        series = statements.series(field_name)
        if not isinstance(series, pd.Series) or series.empty:
            continue
        ordered = list(reversed(series.tolist()))[:max_periods]
        growth = [_NAN] * len(ordered)
        for i in range(len(ordered) - 1):
            prior, current = _num(ordered[i + 1]), _num(ordered[i])
            growth[i] = _div(current - prior, abs(prior) if prior else _NAN)
        rows[f"{label} (level)"] = [_num(v) for v in ordered]
        rows[f"{label} (growth)"] = growth

    if not rows:
        return pd.DataFrame()

    index = list(reversed(statements.series("revenue").index.tolist()))[:max_periods]
    return pd.DataFrame(rows, index=index).T


def retained_series(statements: fu.Statements, field_name: str) -> pd.Series:
    """A statement field as a plain, oldest-first Series, for a chart builder.

    Returns an empty Series rather than raising when the statements are absent:
    a chart with no data should render nothing, not take the page down.
    """
    if not isinstance(statements, fu.Statements) or not statements.ok:
        return pd.Series(dtype=float)
    series = statements.series(field_name)
    return series if isinstance(series, pd.Series) else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# adapters for the downstream consumers
# ---------------------------------------------------------------------------
def peer_metrics(company: CompanySnapshot) -> dict[str, float]:
    """This company's figures keyed for :data:`src.analytics.peers.METRICS`.

    Note the two renames -- the fundamentals engine says ``interest_cover`` and
    ``operating_margin`` where the peer table says ``interest_coverage`` and
    ``ebitda_margin``-style names -- and that ``ev_ebitda`` and ``peg`` are
    absent on purpose: enterprise value needs a market cap *and* a net debt the
    caller may not have, and PEG needs a growth estimate rather than a
    historical rate. A missing metric is left out rather than approximated.
    """
    return {
        "pe": _div(company.market_cap, company.net_income),
        "forward_pe": _NAN,  # needs analyst estimates, which the statements do not hold
        "ev_ebitda": _div(
            (company.market_cap + company.net_debt)
            if not (math.isnan(company.market_cap) or math.isnan(company.net_debt))
            else _NAN,
            company.ebitda,
        ),
        "pb": _div(company.market_cap, company.equity),
        "dividend_yield": _NAN,  # from the dividend history, not the statements
        "roe": company.roe,
        "roce": company.roic,
        "net_margin": company.net_margin,
        "ebitda_margin": company.ebitda_margin,
        "revenue_growth": company.revenue_cagr,
        "earnings_growth": company.earnings_cagr,
        "debt_to_equity": company.debt_to_equity,
        "net_debt_to_ebitda": _div(company.net_debt, company.ebitda),
        "current_ratio": company.current_ratio,
        "interest_coverage": company.interest_cover,
    }


def scorecard_inputs(
    company: CompanySnapshot,
    *,
    dcf_upside: float = _NAN,
    peer_value: float = _NAN,
    sentiment: float = _NAN,
    volatility: float = _NAN,
    max_drawdown: float = _NAN,
) -> dict[str, float]:
    """Everything :func:`src.analytics.scorecard.build_scorecard` needs.

    The five keyword inputs are the ones the statements cannot supply: they come
    from the DCF, the peer table, the news feed and the price history. Each
    defaults to NaN, which the scorecard treats as *excluded* rather than as a
    zero, so a missing market-data call makes the verdict less confident instead
    of silently worse.
    """
    accruals = company.quality.accruals
    return {
        "dcf_upside": _num(dcf_upside),
        "peer_value": _num(peer_value),
        "fcf_yield": company.fcf_yield,
        # `.ok` rather than `.available` throughout: `ScoreResult.available` is an
        # integer *count* of the inputs that ran, not a flag, so `if score.available`
        # is true whenever one signal ran -- including the case where the value is
        # still None and belongs in the composite as missing.
        "piotroski": company.quality.piotroski.value if company.quality.piotroski.ok else _NAN,
        "altman_z": company.quality.altman.value if company.quality.altman.ok else _NAN,
        "beneish_m": company.quality.beneish.value if company.quality.beneish.ok else _NAN,
        # THE UNIT CONVERSION THIS MODULE EXISTS FOR. `accruals_ratio` reports
        # percent, every other rate here is a decimal, and the scorecard's
        # anchors are decimals. Feeding -5.56 to a scale that ends at 0.25 does
        # not raise -- it clamps to the best possible score.
        "accruals": accruals.value / 100.0 if accruals.ok else _NAN,
        "revenue_growth": company.revenue_cagr,
        "roe": company.roe,
        "net_margin": company.net_margin,
        "debt_to_equity": company.debt_to_equity,
        "interest_coverage": company.interest_cover,
        "volatility": _num(volatility),
        "max_drawdown": _num(max_drawdown),
        "sentiment": _num(sentiment),
    }


def ratio_rows(company: CompanySnapshot) -> tuple[tuple[str, str], ...]:
    """The ratio table as ``(label, formatted)`` pairs, missing shown as em-dash.

    Formatted here rather than in the page so the on-screen table and the PDF
    cannot disagree, and so an unavailable ratio is the same ``—`` in both
    instead of a zero in one of them.
    """

    def pct(value: float) -> str:
        return "—" if math.isnan(value) else f"{value:.1%}"

    def times(value: float) -> str:
        return "—" if math.isnan(value) else f"{value:.2f}x"

    def level(value: float) -> str:
        return "—" if math.isnan(value) else f"{value:,.2f}"

    return (
        ("Return on equity", pct(company.roe)),
        ("Return on invested capital", pct(company.roic)),
        ("Return on assets", pct(company.roa)),
        ("Gross margin", pct(company.gross_margin)),
        ("Operating margin", pct(company.operating_margin)),
        ("EBITDA margin", pct(company.ebitda_margin)),
        ("Net margin", pct(company.net_margin)),
        ("Free cash flow margin", pct(company.fcf_margin)),
        ("Cash conversion (OCF / net income)", times(company.cash_conversion)),
        ("Asset turnover", times(company.asset_turnover)),
        ("Debt / equity", times(company.debt_to_equity)),
        ("Interest cover", times(company.interest_cover)),
        ("Current ratio", times(company.current_ratio)),
        ("Capital expenditure / revenue", pct(company.capex_to_revenue)),
        ("Effective tax rate", pct(company.effective_tax_rate)),
        ("Free cash flow yield", pct(company.fcf_yield)),
        ("Earnings yield", pct(company.earnings_yield)),
        ("Book value per share", level(company.book_value_per_share)),
        ("Revenue CAGR", pct(company.revenue_cagr)),
        ("Revenue CAGR (3y)", pct(company.revenue_cagr_3y)),
        ("Earnings CAGR", pct(company.earnings_cagr)),
    )
