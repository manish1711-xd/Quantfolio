"""Domain dataclasses for Equity Research Report Generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquityResearchInputs:
    """User-supplied inputs for report generation."""
    ticker: str
    period: str = "1y"
    benchmark: str = "^NSEI"
    wacc_override: Optional[float] = None
    growth_override: Optional[float] = None
    terminal_growth_override: Optional[float] = None


@dataclass
class FundamentalData:
    """Calculated and extracted fundamental metrics."""
    gross_margin: float = np.nan
    operating_margin: float = np.nan
    net_margin: float = np.nan
    eps: float = np.nan
    roe: float = np.nan
    roce: float = np.nan
    debt_to_equity: float = np.nan
    debt_to_asset: float = np.nan
    revenue_latest: float = np.nan
    profit_latest: float = np.nan
    free_cash_flow: float = np.nan
    shares_outstanding: float = np.nan
    currency: str = "INR"
    statements_available: dict[str, bool] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class TechnicalData:
    """Calculated technical indicators and statistical metrics."""
    sma: dict[str, float] = field(default_factory=dict)
    ema: dict[str, float] = field(default_factory=dict)
    rsi: float = np.nan
    macd: dict[str, float] = field(default_factory=dict)
    volatility: float = np.nan
    beta: float = np.nan
    sharpe_ratio: float = np.nan
    max_drawdown: float = np.nan
    current_price: float = np.nan
    period_high: float = np.nan
    period_low: float = np.nan


@dataclass
class DCFInputs:
    """Parameters for Discounted Cash Flow valuation."""
    fcffree: float
    wacc: float
    growth_rate: float
    terminal_growth_rate: float = 0.03
    shares_outstanding: Optional[float] = None
    net_debt: float = 0.0
    projection_years: int = 5


@dataclass
class ValuationResult:
    """Outputs from DCF and relative valuation."""
    intrinsic_value: Optional[float]
    fair_value_low: Optional[float]
    fair_value_high: Optional[float]
    recommendation: str  # "Buy", "Hold", "Sell", or "Insufficient data"
    dcf_interpolated: dict[str, float] = field(default_factory=dict)
    wacc_used: float = np.nan
    terminal_value: float = np.nan
    enterprise_value: float = np.nan
    equity_value: float = np.nan
    sensitivity_table: Optional[pd.DataFrame] = None
    status_notes: list[str] = field(default_factory=list)


@dataclass
class PeerComparisonResult:
    """Multiples and comparative metrics against sector peers."""
    target_ticker: str
    peers: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_multiples: dict[str, Optional[float]] = field(default_factory=dict)
    sector_averages: dict[str, float] = field(default_factory=dict)
    percentile_rank: dict[str, float] = field(default_factory=dict)
    sufficient_data: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class AIReportSections:
    """Markdown/text content for each section of the research report."""
    executive_summary: str
    company_overview: str
    financial_analysis: str
    valuation: str
    risks: str
    conclusion: str
    sources: str
    assumptions: str


@dataclass
class EquityResearchReport:
    """Full comprehensive equity research report container."""
    ticker: str
    as_of_date: str
    sections: AIReportSections
    quote: dict[str, Any] = field(default_factory=dict)
    fundamental: Optional[FundamentalData] = None
    technical: Optional[TechnicalData] = None
    valuation: Optional[ValuationResult] = None
    peer_comparison: Optional[PeerComparisonResult] = None
