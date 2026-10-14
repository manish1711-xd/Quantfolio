"""Domain models shared across every layer.

These dataclasses are the contract between the data layer, the analytics
layer and the UI. Keeping them here (rather than letting each module invent
its own dict shape) is what allows the analytics package to be tested with
hand-built fixtures and no network.

All models are immutable where practical -- an analytics pipeline that cannot
mutate its inputs is dramatically easier to reason about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np

from src.config import DATA_SOURCE

#: Name tokens that identify a fund when the provider does not classify one.
#: Word-bounded so a company whose name merely contains "Fundamental" is never
#: reclassified, and applied only when no sector was reported at all.
_FUND_NAME_PATTERN = re.compile(r"\b(ETF|BEES|INDEX\s+FUND|MUTUAL\s+FUND|FUND\s+OF\s+FUNDS)\b")

InsightLevel = Literal["positive", "info", "warning", "critical"]
TradeAction = Literal["BUY", "SELL"]


# ---------------------------------------------------------------------------
# market data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Quote:
    """A single price observation with full provenance.

    The positional order of the first six fields is frozen for backwards
    compatibility with existing fixtures: ``Quote(ticker, price, currency,
    as_of, fetched_at, is_intraday)``. Newer fields are keyword-optional.

    Attributes
    ----------
    price:          last available price (intraday quote or last daily close)
    as_of:          the timestamp the price refers to
    fetched_at:     when *we* asked the provider (always UTC)
    is_intraday:    True  -> from the live-ish quote endpoint (may still be
                    delayed ~15 min for NSE/BSE)
                    False -> latest available daily close
    previous_close: prior session close, when the provider supplies it
    """

    ticker: str
    price: float
    currency: str
    as_of: datetime
    fetched_at: datetime
    is_intraday: bool
    source: str = DATA_SOURCE
    previous_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None

    @property
    def freshness_label(self) -> str:
        return (
            "Latest traded price (may be delayed ~15 min)"
            if self.is_intraday
            else "Latest available closing price"
        )

    @property
    def day_change(self) -> float:
        """Absolute move versus the previous close (NaN when unknown)."""
        if self.previous_close is None or not np.isfinite(self.previous_close):
            return float("nan")
        return float(self.price - self.previous_close)

    @property
    def day_change_pct(self) -> float:
        """Percentage move versus the previous close (NaN when unknown)."""
        if (
            self.previous_close is None
            or not np.isfinite(self.previous_close)
            or self.previous_close <= 0
        ):
            return float("nan")
        return float((self.price / self.previous_close - 1.0) * 100.0)


@dataclass(frozen=True)
class SecurityProfile:
    """Slow-moving reference and fundamental data for one instrument.

    Every field is optional: Yahoo Finance coverage is patchy, especially for
    Indian small caps and ETFs, and the UI must degrade gracefully rather than
    pretend a missing P/E is zero.
    """

    ticker: str
    name: str | None = None
    short_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    quote_type: str | None = None  # EQUITY / ETF / INDEX / MUTUALFUND
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    dividend_yield: float | None = None  # decimal (0.012 = 1.2%)
    beta: float | None = None  # provider-supplied, 5y monthly
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    website: str | None = None
    summary: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.short_name or self.ticker

    @property
    def is_fund(self) -> bool:
        """True for an ETF or mutual fund, including ones Yahoo mislabels.

        ``quote_type`` alone is not enough. Yahoo Finance reports
        ``quoteType="EQUITY"`` for essentially every Indian ETF -- NIFTYBEES.NS
        and GOLDBEES.NS both come back as equities with no sector and no
        ``fundFamily`` -- so the instrument name is the only remaining signal.
        The name check runs only as a fallback, and only for instruments that
        reported no sector at all, so it cannot reclassify a real company.
        """
        if (self.quote_type or "").upper() in {"ETF", "MUTUALFUND"}:
            return True
        name = (self.name or self.short_name or "").upper()
        return bool(_FUND_NAME_PATTERN.search(name))

    @property
    def sector_label(self) -> str:
        """Sector with a sane fallback for ETFs and indices that report none."""
        if self.sector:
            return self.sector
        if self.is_fund:
            return "Fund / ETF"
        if (self.quote_type or "").upper() == "INDEX":
            return "Index"
        return "Unclassified"


# ---------------------------------------------------------------------------
# holdings & transactions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transaction:
    """One executed trade.

    ``fees`` is always a positive cost. A BUY increases the position and its
    cost basis by ``quantity * price + fees``; a SELL decreases the position
    and realises P&L net of ``fees``.
    """

    ticker: str
    trade_date: date
    action: TradeAction
    quantity: float
    price: float
    fees: float = 0.0
    currency: str = "INR"
    note: str = ""

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.action == "BUY" else -self.quantity

    @property
    def gross_value(self) -> float:
        return float(self.quantity * self.price)

    @property
    def cash_flow(self) -> float:
        """Cash leaving (-) or entering (+) the investor's pocket."""
        if self.action == "BUY":
            return -(self.gross_value + self.fees)
        return self.gross_value - self.fees


@dataclass(frozen=True)
class Lot:
    """An open tax lot produced by FIFO matching."""

    ticker: str
    open_date: date
    quantity: float
    cost_per_share: float

    @property
    def cost_basis(self) -> float:
        return float(self.quantity * self.cost_per_share)


@dataclass(frozen=True)
class RealisedTrade:
    """A closed (matched) buy/sell pair with its realised P&L."""

    ticker: str
    open_date: date
    close_date: date
    quantity: float
    cost_per_share: float
    sale_per_share: float
    fees: float = 0.0

    @property
    def proceeds(self) -> float:
        return float(self.quantity * self.sale_per_share - self.fees)

    @property
    def cost_basis(self) -> float:
        return float(self.quantity * self.cost_per_share)

    @property
    def realised_pnl(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def realised_pct(self) -> float:
        basis = self.cost_basis
        return float(self.realised_pnl / basis * 100.0) if basis > 0 else float("nan")

    @property
    def holding_days(self) -> int:
        return int((self.close_date - self.open_date).days)

    @property
    def is_long_term(self) -> bool:
        """India treats listed-equity gains as long-term beyond 12 months."""
        return self.holding_days > 365


# ---------------------------------------------------------------------------
# analytics outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Insight:
    """One rule-derived, plain-English observation.

    ``evidence`` records the numbers and threshold that fired the rule, so the
    UI can show *why* a sentence appeared. Nothing here is predicted or
    generated -- every Insight traces back to a constant in ``src.config``.
    """

    level: InsightLevel
    text: str
    category: str = "general"
    evidence: str = ""

    # Keeps ``Insight("info", "text")`` working like the v1 two-field version.
    def __post_init__(self) -> None:  # pragma: no cover - trivial
        pass


@dataclass(frozen=True)
class DrawdownEpisode:
    """A single peak -> trough -> recovery cycle."""

    peak_date: date
    trough_date: date
    recovery_date: date | None
    depth: float  # negative decimal, e.g. -0.23
    drawdown_days: int  # peak -> trough
    recovery_days: int | None  # trough -> recovery (None if still underwater)

    @property
    def total_days(self) -> int | None:
        if self.recovery_days is None:
            return None
        return self.drawdown_days + self.recovery_days

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None


@dataclass(frozen=True)
class OptimizedPortfolio:
    """Result of one portfolio-optimisation objective."""

    label: str
    weights: dict[str, float]  # ticker -> weight fraction, sums to 1
    expected_return: float  # annualised decimal
    volatility: float  # annualised decimal
    sharpe: float
    converged: bool = True
    note: str = ""


@dataclass
class AnalysisResult:
    """Everything one 'Analyze' run produced, handed to the UI as a unit.

    A plain mutable container on purpose: the pipeline fills it in stages and
    every page reads from it, which keeps Streamlit page code free of any
    computation.
    """

    positions: object  # pd.DataFrame
    summary: dict = field(default_factory=dict)
    quotes: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    prices: object | None = None  # pd.DataFrame of adjusted closes
    values: object | None = None  # pd.Series portfolio value
    returns: object | None = None  # pd.Series portfolio daily returns
    asset_returns: object | None = None  # pd.DataFrame per-asset returns
    benchmark_prices: object | None = None  # pd.Series
    risk: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)
    drawdowns: list = field(default_factory=list)
    correlation: object | None = None  # pd.DataFrame
    covariance: object | None = None  # pd.DataFrame, annualised & shrunk
    rolling: object | None = None  # pd.DataFrame of rolling statistics
    monthly: object | None = None  # pd.DataFrame, years x months (%)
    return_stats: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)  # FIFO lots, realised P&L, XIRR
    score: dict = field(default_factory=dict)
    insights: list = field(default_factory=list)
    fx: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    config: object | None = None  # AppConfig used for this run
    generated_at: datetime | None = None

    @property
    def n_days(self) -> int:
        try:
            return int(len(self.values))  # type: ignore[arg-type]
        except TypeError:
            return 0

    @property
    def has_history(self) -> bool:
        return self.n_days > 1
