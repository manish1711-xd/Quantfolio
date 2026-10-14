"""Formatted Excel export: the artefact a user emails to somebody else.

An analytics run is a lot of numbers. This module turns one
:class:`~src.models.AnalysisResult` into a workbook that a colleague can open
without asking a single question: a title block that says what the file is,
when it was produced and what it is not (advice), one sheet per analytical
theme, frozen headers, autofilters, per-column number formats, and colour
scales that make the shape of the P&L visible before the reader parses a
digit.

Three conventions, stated once so no sheet has to repeat them:

* **Money is base currency, prices are native.** ``investment``,
  ``current_value`` and ``pnl`` are converted at the ``fx_rate`` on the row;
  ``buy_price`` and ``current_price`` stay in the instrument's own currency and
  therefore carry no currency symbol. The Holdings sheet says so in a footnote.
* **Missing is blank.** A NaN, None, NaT or infinity is written as an empty
  cell, never as a zero. A workbook that silently fabricates zeros is worse
  than one with gaps.
* **Colours come from the chart design system.** ``src.charts.theme`` owns the
  gain/loss/accent language; the fills here are light tints of exactly those
  colours, so the workbook and the dashboard read as one product.

Every section is optional: a result with no benchmark, no correlation matrix
and no transaction ledger simply yields a shorter workbook, never an error.
No Streamlit import - this runs in a script, a cron job or a test just as well
as in the app.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_range

from src import config as cfg
from src.charts import theme

# ---------------------------------------------------------------------------
# public contract
# ---------------------------------------------------------------------------
#: Sheet order of a fully populated workbook. Sheets whose section of the
#: analysis is missing are omitted, so the real workbook is a subsequence.
SHEETS: tuple[str, ...] = (
    "Summary",
    "Holdings",
    "Risk",
    "Benchmark",
    "Contribution",
    "Correlation",
    "Drawdowns",
    "Transactions",
)

#: Body font. Deliberately NOT the app's Inter webfont: a workbook is opened on
#: someone else's machine, and Calibri is the one face Excel always has.
_BODY_FONT = "Calibri"

_MAX_ROWS = 50_000  # hard cap per table; Excel tops out at 1,048,576
_MAX_TEXT = 32_000  # Excel's cell limit is 32,767 characters


# ---------------------------------------------------------------------------
# column semantics
# ---------------------------------------------------------------------------
# Format "kinds". A kind is a display contract, not a dtype: it says what the
# number MEANS (base-currency money, percent already in percent units, a
# decimal fraction, a pure ratio) and therefore how it must be rendered.
_MONEY = "money"
_PRICE = "price"
_PCT_UNIT = "pct_unit"  # 12.34 means 12.34%
_PCT_DEC = "pct_dec"  # 0.1234 means 12.34%
_RATIO = "ratio"
_INT = "int"
_QTY = "qty"
_FX = "fx"
_DATE = "date"
_TEXT = "text"
_WRAP = "wrap"

_MONEY_COLUMNS: frozenset[str] = frozenset(
    {
        "investment",
        "current_value",
        "market_value",
        "pnl",
        "day_change",
        "cost_basis",
        "proceeds",
        "realised_pnl",
        "trade_value",
        "target_value",
        "value_gap",
        "value_change",
        "projected_value",
        "peak_value",
        "cash_flow",
        "contribution",
        "total_investment",
        "total_pnl",
        "buy_value",
        "sell_value",
        "net_cash",
        "total_realised",
        "total_cost_basis",
        "avg_win",
        "avg_loss",
        "short_term_pnl",
        "long_term_pnl",
        "component_var",
        "var_value",
    }
)

_PRICE_COLUMNS: frozenset[str] = frozenset(
    {
        "buy_price",
        "current_price",
        "price",
        "cost_per_share",
        "sale_per_share",
        "fees",
        "open",
        "high",
        "low",
        "close",
    }
)

_PCT_DEC_COLUMNS: frozenset[str] = frozenset(
    {
        "depth",
        "coverage",
        "weight",
        "mdd",
        "max_drawdown",
        "sigma",
    }
)

_INT_COLUMNS: frozenset[str] = frozenset(
    {
        "n",
        "n_holdings",
        "n_days",
        "n_obs",
        "n_trades",
        "days",
        "drawdown_days",
        "recovery_days",
        "total_days",
        "holding_days",
        "volume",
        "breaches",
        "up_months",
        "down_months",
        "in_band_count",
        "n_assets",
    }
)

_DATE_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "buy_date",
        "trade_date",
        "open_date",
        "close_date",
        "peak_date",
        "trough_date",
        "recovery_date",
        "start",
        "end",
        "start_date",
        "end_date",
        "as_of",
    }
)

_QTY_COLUMNS: frozenset[str] = frozenset({"quantity", "shares"})
_FX_COLUMNS: frozenset[str] = frozenset({"fx_rate", "rate"})
_RATIO_COLUMNS: frozenset[str] = frozenset(
    {
        "beta",
        "mctr",
        "cctr",
        "ratio",
        "sharpe",
        "sortino",
        "hhi",
        "gini",
    }
)
_WRAP_COLUMNS: frozenset[str] = frozenset(
    {
        "note",
        "evidence",
        "observation",
        "summary",
        "reason",
        "interpretation",
    }
)

#: Percent-unit columns whose name does not end in ``_pct``.
_PCT_UNIT_COLUMNS: frozenset[str] = frozenset(
    {
        "win_rate",
        "ulcer",
        "turnover",
        "drift",
        "weight_percent",
    }
)

#: Signed columns that get a red-neutral-green three-colour scale. These are
#: the columns a reader scans for shape rather than for a specific value.
_SCALE_COLUMNS: frozenset[str] = frozenset(
    {
        "pnl",
        "pnl_pct",
        "day_change",
        "day_change_pct",
        "contribution_pct",
        "contribution_share_pct",
        "realised_pnl",
        "realised_pct",
        "drift_pct",
        "value_change",
        "value_gap",
        "depth",
        "portfolio_return_pct",
        "implied_portfolio_pct",
        "worst_asset_pct",
        "shock_pct",
        "return_pct",
        "excess_return_pct",
        "cash_flow",
        "total_return_pct",
    }
)

#: Magnitude columns that get an in-cell data bar.
_BAR_COLUMNS: frozenset[str] = frozenset(
    {
        "weight_pct",
        "risk_contribution_pct",
        "current_value",
        "cost_basis",
    }
)

#: Columns whose sum is meaningful, so the total row can carry a SUBTOTAL.
_ADDITIVE_COLUMNS: frozenset[str] = frozenset(
    {
        "investment",
        "current_value",
        "pnl",
        "weight_pct",
        "day_change",
        "cost_basis",
        "proceeds",
        "realised_pnl",
        "trade_value",
        "target_value",
        "contribution_pct",
        "risk_contribution_pct",
        "target_weight_pct",
        "value_change",
        "n",
    }
)

#: Human column headings. Anything absent falls back to a de-underscored,
#: sentence-cased version of the field name.
_COLUMN_LABELS: dict[str, str] = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "industry": "Industry",
    "currency": "Ccy",
    "quantity": "Quantity",
    "shares": "Shares",
    "buy_price": "Avg buy price",
    "current_price": "Current price",
    "fx_rate": "FX rate",
    "day_change_pct": "Day %",
    "investment": "Invested",
    "current_value": "Market value",
    "pnl": "P&L",
    "pnl_pct": "Return %",
    "weight_pct": "Weight %",
    "target_weight_pct": "Target %",
    "buy_date": "Buy date",
    "contribution_pct": "Contribution %",
    "contribution_share_pct": "Share of P&L %",
    "mctr": "MCTR",
    "cctr": "CCTR",
    "risk_contribution_pct": "Risk contribution %",
    "ratio": "Risk / weight",
    "peak_date": "Peak",
    "trough_date": "Trough",
    "recovery_date": "Recovered",
    "depth": "Depth",
    "drawdown_days": "Fall (days)",
    "recovery_days": "Recovery (days)",
    "total_days": "Total (days)",
    "status": "Status",
    "open_date": "Opened",
    "close_date": "Closed",
    "cost_per_share": "Cost / share",
    "sale_per_share": "Sale / share",
    "fees": "Fees",
    "cost_basis": "Cost basis",
    "proceeds": "Proceeds",
    "realised_pnl": "Realised P&L",
    "realised_pct": "Realised %",
    "holding_days": "Held (days)",
    "term": "Term",
    "trade_date": "Date",
    "action": "Action",
    "price": "Price",
    "cash_flow": "Cash flow",
    "gross_value": "Gross value",
    "note": "Note",
    "n": "Count",
    "level": "Level",
    "category": "Category",
    "observation": "Observation",
    "evidence": "Evidence",
    "metric": "Metric",
    "value": "Value",
    "drift_pct": "Drift %",
    "in_band": "In band",
    "long_term": "Long term",
    "n_trades": "Closed trades",
    "win_rate": "Win rate",
    "avg_win": "Average win",
    "avg_loss": "Average loss",
    "profit_factor": "Profit factor",
    "total_realised": "Total realised",
    "total_cost_basis": "Total cost basis",
    "best": "Best trade",
    "worst": "Worst trade",
    "short_term_pnl": "Short-term P&L",
    "long_term_pnl": "Long-term P&L",
    "avg_holding_days": "Average holding (days)",
    "xirr": "XIRR",
    "problem": "Problem",
    "coverage": "Coverage",
}


# ---------------------------------------------------------------------------
# metric catalogues (label, kind and - crucially - what the number means)
# ---------------------------------------------------------------------------
_RISK_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ann_return",
        "Annualised return",
        _PCT_DEC,
        "Geometric CAGR of the portfolio value series over the analysis window.",
    ),
    (
        "ann_vol",
        "Annualised volatility",
        _PCT_DEC,
        "Standard deviation of daily returns scaled by sqrt(252).",
    ),
    (
        "downside_dev",
        "Downside deviation",
        _PCT_DEC,
        "Volatility of returns below the minimum acceptable return only.",
    ),
    ("sharpe", "Sharpe ratio", _RATIO, "(Annualised return - risk-free) / annualised volatility."),
    (
        "sortino",
        "Sortino ratio",
        _RATIO,
        "Sharpe with downside deviation in the denominator; upside volatility is not penalised.",
    ),
    (
        "calmar",
        "Calmar ratio",
        _RATIO,
        "Annualised return divided by the absolute maximum drawdown.",
    ),
    (
        "martin",
        "Martin ratio",
        _RATIO,
        "Excess return per unit of Ulcer index, so both depth and duration of "
        "drawdowns are charged for.",
    ),
    (
        "omega",
        "Omega ratio",
        _RATIO,
        "Probability-weighted gains over losses relative to the minimum acceptable return.",
    ),
    ("mdd", "Maximum drawdown", _PCT_DEC, "Worst peak-to-trough decline in the value series."),
    (
        "ulcer",
        "Ulcer index",
        _PCT_UNIT,
        "Root mean square of percentage drawdowns; long shallow pain scores worse "
        "than one sharp dip.",
    ),
    (
        "var",
        "Value at Risk (historical)",
        _PCT_DEC,
        "Empirical one-day loss quantile at the stated confidence. Negative by convention.",
    ),
    (
        "cvar",
        "Conditional VaR (expected shortfall)",
        _PCT_DEC,
        "Average one-day loss on the days worse than the historical VaR.",
    ),
    (
        "var_gaussian",
        "VaR (Gaussian)",
        _PCT_DEC,
        "mu + z * sigma. Assumes normal returns, which understates fat-tailed markets.",
    ),
    (
        "var_cornish_fisher",
        "VaR (Cornish-Fisher)",
        _PCT_DEC,
        "Gaussian VaR corrected for the sample's skew and excess kurtosis.",
    ),
    (
        "skew",
        "Skewness",
        _RATIO,
        "Asymmetry of daily returns. Negative means the left tail is longer.",
    ),
    (
        "kurtosis",
        "Excess kurtosis",
        _RATIO,
        "Fat-tailedness relative to a normal distribution (0 = normal).",
    ),
    (
        "tail_ratio",
        "Tail ratio",
        _RATIO,
        "95th percentile gain over the absolute 5th percentile loss.",
    ),
    ("hit_ratio", "Hit ratio", _PCT_DEC, "Share of days that closed positive."),
    (
        "gain_to_pain",
        "Gain to pain ratio",
        _RATIO,
        "Sum of returns over the absolute sum of negative returns.",
    ),
    ("best_day", "Best day", _PCT_DEC, "Largest single-day gain in the window."),
    ("worst_day", "Worst day", _PCT_DEC, "Largest single-day loss in the window."),
    ("n_obs", "Observations", _INT, "Overlapping daily observations behind every statistic above."),
    (
        "confidence",
        "VaR confidence",
        _PCT_DEC,
        "Confidence level used for the VaR and expected-shortfall family.",
    ),
)

_BENCHMARK_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "portfolio_return",
        "Portfolio return (annualised)",
        _PCT_DEC,
        "Geometric annualised return over the overlapping window.",
    ),
    (
        "benchmark_return",
        "Benchmark return (annualised)",
        _PCT_DEC,
        "Same window, same convention, so the comparison is like for like.",
    ),
    ("excess_return", "Excess return", _PCT_DEC, "Portfolio minus benchmark annualised return."),
    (
        "alpha",
        "Jensen's alpha",
        _PCT_DEC,
        "R_p - [R_f + beta * (R_b - R_f)]: return not explained by market exposure.",
    ),
    ("beta", "Beta", _RATIO, "Sensitivity of daily portfolio returns to the benchmark."),
    ("correlation", "Correlation", _RATIO, "Correlation of daily portfolio and benchmark returns."),
    ("r_squared", "R-squared", _RATIO, "Share of portfolio variance explained by the benchmark."),
    (
        "portfolio_vol",
        "Portfolio volatility",
        _PCT_DEC,
        "Annualised standard deviation of daily returns.",
    ),
    (
        "benchmark_vol",
        "Benchmark volatility",
        _PCT_DEC,
        "Annualised standard deviation of benchmark daily returns.",
    ),
    (
        "tracking_error",
        "Tracking error",
        _PCT_DEC,
        "Annualised standard deviation of active (portfolio - benchmark) returns.",
    ),
    ("information_ratio", "Information ratio", _RATIO, "Active return per unit of tracking error."),
    (
        "treynor",
        "Treynor ratio",
        _RATIO,
        "Excess return per unit of beta, i.e. per unit of systematic risk.",
    ),
    (
        "m2",
        "M-squared (Modigliani)",
        _PCT_DEC,
        "Risk-free + portfolio Sharpe * benchmark volatility: what the portfolio "
        "would have returned at the benchmark's risk.",
    ),
    (
        "up_capture",
        "Up capture",
        _RATIO,
        "Mean portfolio return in months the benchmark rose, over the benchmark's own mean.",
    ),
    (
        "down_capture",
        "Down capture",
        _RATIO,
        "The same in months the benchmark fell. Below 1 is defensive.",
    ),
    (
        "capture_ratio",
        "Capture ratio",
        _RATIO,
        "Up capture divided by down capture; above 1 is asymmetry in your favour.",
    ),
    ("up_months", "Up months", _INT, "Months in which the benchmark rose."),
    ("down_months", "Down months", _INT, "Months in which the benchmark fell."),
    (
        "best_relative_month",
        "Best relative month",
        _PCT_DEC,
        "Largest monthly outperformance of the benchmark.",
    ),
    (
        "worst_relative_month",
        "Worst relative month",
        _PCT_DEC,
        "Largest monthly underperformance of the benchmark.",
    ),
    (
        "portfolio_mdd",
        "Portfolio max drawdown",
        _PCT_DEC,
        "Worst peak-to-trough decline of the portfolio in the window.",
    ),
    (
        "benchmark_mdd",
        "Benchmark max drawdown",
        _PCT_DEC,
        "Worst peak-to-trough decline of the benchmark in the window.",
    ),
    (
        "n_days",
        "Overlapping days",
        _INT,
        "Days on which both series have data. Everything above uses only these.",
    ),
)

_CONCENTRATION_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("n_holdings", "Holdings", _INT, "Number of distinct positions."),
    (
        "hhi",
        "Herfindahl index",
        _RATIO,
        "Sum of squared weights. 1.0 is a single name, 1/N is equal weight.",
    ),
    (
        "effective_holdings",
        "Effective holdings",
        _RATIO,
        "1 / HHI: how many equally weighted names this book really behaves like.",
    ),
    ("top1_pct", "Top 1 weight", _PCT_UNIT, "Weight of the largest position."),
    ("top3_pct", "Top 3 weight", _PCT_UNIT, "Combined weight of the top three."),
    ("top5_pct", "Top 5 weight", _PCT_UNIT, "Combined weight of the top five."),
    (
        "gini",
        "Gini coefficient",
        _RATIO,
        "Inequality of the weight distribution: 0 is perfectly even, 1 is one name.",
    ),
)

_LEDGER_SECTIONS: tuple[tuple[str, str], ...] = (
    ("transactions", "Transactions"),
    ("lots", "Open tax lots (FIFO)"),
    ("realised", "Realised trades (FIFO matched)"),
    ("realised_trades", "Realised trades (FIFO matched)"),
    ("holdings", "Holdings implied by the ledger"),
    ("cash_flows", "External cash flows"),
    ("summary", "Realised P&L summary"),
    ("realised_summary", "Realised P&L summary"),
    ("problems", "Ledger problems"),
)

#: Dataclass properties worth materialising as columns when a ledger hands us
#: model objects rather than a frame.
_PROPERTY_EXTRAS: dict[str, tuple[str, ...]] = {
    "Transaction": ("gross_value", "cash_flow"),
    "Lot": ("cost_basis",),
    "RealisedTrade": (
        "cost_basis",
        "proceeds",
        "realised_pnl",
        "realised_pct",
        "holding_days",
        "is_long_term",
    ),
}


# ---------------------------------------------------------------------------
# colour helpers - every colour is derived from the chart design system
# ---------------------------------------------------------------------------
def _palette() -> dict[str, str]:
    """The light palette: a workbook is read on white paper and white sheets."""
    return theme.LIGHT_PALETTE


def _tint(color: str, amount: float) -> str:
    """Blend a hex colour toward white (``amount`` 0 keeps it, 1 is white)."""
    raw = str(color).lstrip("#")
    if len(raw) != 6:
        return "#FFFFFF"
    try:
        channels = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return "#FFFFFF"
    weight = min(max(float(amount), 0.0), 1.0)
    mixed = [int(round(c + (255 - c) * weight)) for c in channels]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _shade(color: str, amount: float) -> str:
    """Blend a hex colour toward black (``amount`` 0 keeps it, 1 is black)."""
    raw = str(color).lstrip("#")
    if len(raw) != 6:
        return "#000000"
    try:
        channels = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return "#000000"
    weight = min(max(float(amount), 0.0), 1.0)
    mixed = [int(round(c * (1.0 - weight))) for c in channels]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


# ---------------------------------------------------------------------------
# number formats
# ---------------------------------------------------------------------------
def _money_format(symbol: str) -> str:
    """Excel number format for base-currency money, negatives in red."""
    core = "#,##0.00"
    if symbol:
        quoted = '"' + str(symbol).replace('"', "") + '"'
        return f"{quoted}{core};[Red]-{quoted}{core}"
    return f"{core};[Red]-{core}"


_NUMBER_FORMATS: dict[str, str] = {
    _PRICE: "#,##0.00;[Red]-#,##0.00",
    _PCT_UNIT: '#,##0.00"%";[Red]-#,##0.00"%"',
    _PCT_DEC: "0.00%;[Red]-0.00%",
    _RATIO: "0.00;[Red]-0.00",
    _INT: "#,##0;[Red]-#,##0",
    _QTY: "#,##0.####;[Red]-#,##0.####",
    _FX: "0.0000",
    _DATE: "dd mmm yyyy",
    _TEXT: "",
    _WRAP: "",
}


class _Formats:
    """Every cell format the workbook uses, built once per workbook.

    Excel formats are workbook-scoped objects, so they are created here rather
    than at each write site; that also keeps the visual language in one place.
    """

    def __init__(self, book: xlsxwriter.Workbook, currency_symbol: str = "") -> None:
        """Build every format for ``book``, money prefixed with the symbol."""
        self.book = book
        self.symbol = currency_symbol
        pal = _palette()
        self.accent = pal["accent"]
        self.gain = pal["gain"]
        self.loss = pal["loss"]
        self.warning = pal["warning"]
        self.text = pal["text"]
        self.subtext = pal["subtext"]
        hairline = _tint(pal["muted"], 0.72)
        self.hairline = hairline
        self.scale_low = _tint(pal["loss"], 0.62)
        self.scale_mid = _tint(pal["muted"], 0.93)
        self.scale_high = _tint(pal["gain"], 0.62)
        self.bar = _tint(pal["accent"], 0.42)
        self.corr_low = _tint(str(theme.DIVERGING[0][1]), 0.55)
        self.corr_high = _tint(str(theme.DIVERGING[-1][1]), 0.55)

        base = {"font_name": _BODY_FONT, "font_size": 10, "valign": "vcenter"}
        cell = {**base, "bottom": 1, "bottom_color": hairline}

        self.title = book.add_format(
            {
                "font_name": _BODY_FONT,
                "font_size": 18,
                "bold": True,
                "font_color": pal["accent"],
                "valign": "vcenter",
            }
        )
        self.tagline = book.add_format(
            {
                **base,
                "font_size": 11,
                "italic": True,
                "font_color": pal["subtext"],
            }
        )
        self.meta = book.add_format(
            {
                **base,
                "font_size": 9.5,
                "font_color": pal["subtext"],
            }
        )
        self.meta_strong = book.add_format(
            {
                **base,
                "font_size": 9.5,
                "bold": True,
                "font_color": pal["text"],
            }
        )
        self.disclaimer = book.add_format(
            {
                **base,
                "font_size": 9,
                "italic": True,
                "font_color": pal["subtext"],
                "text_wrap": True,
                "valign": "top",
                "top": 1,
                "top_color": hairline,
            }
        )
        self.sheet_title = book.add_format(
            {
                "font_name": _BODY_FONT,
                "font_size": 13,
                "bold": True,
                "font_color": pal["accent"],
                "valign": "vcenter",
            }
        )
        self.sheet_subtitle = book.add_format(
            {
                **base,
                "font_size": 9.5,
                "font_color": pal["subtext"],
                "text_wrap": False,
            }
        )
        self.section = book.add_format(
            {
                "font_name": _BODY_FONT,
                "font_size": 11,
                "bold": True,
                "font_color": pal["text"],
                "valign": "vcenter",
                "bottom": 2,
                "bottom_color": _tint(pal["accent"], 0.55),
            }
        )
        self.header = book.add_format(
            {
                "font_name": _BODY_FONT,
                "font_size": 10,
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": _shade(pal["accent"], 0.18),
                "align": "left",
                "valign": "vcenter",
                "text_wrap": True,
                "bottom": 1,
                "bottom_color": _shade(pal["accent"], 0.35),
            }
        )
        self.header_right = book.add_format(
            {
                "font_name": _BODY_FONT,
                "font_size": 10,
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": _shade(pal["accent"], 0.18),
                "align": "right",
                "valign": "vcenter",
                "text_wrap": True,
                "bottom": 1,
                "bottom_color": _shade(pal["accent"], 0.35),
            }
        )
        self.label = book.add_format(
            {
                **cell,
                "font_color": pal["subtext"],
            }
        )
        self.label_strong = book.add_format(
            {
                **cell,
                "bold": True,
                "font_color": pal["text"],
            }
        )
        self.note = book.add_format(
            {
                **cell,
                "font_size": 9.5,
                "font_color": pal["subtext"],
                "text_wrap": True,
                "valign": "top",
            }
        )
        self.footnote = book.add_format(
            {
                **base,
                "font_size": 9,
                "italic": True,
                "font_color": pal["subtext"],
                "text_wrap": True,
                "valign": "top",
            }
        )

        money = _money_format(currency_symbol)
        self._by_kind: dict[str, Any] = {
            _MONEY: book.add_format({**cell, "num_format": money}),
        }
        for kind, num_format in _NUMBER_FORMATS.items():
            spec = dict(cell)
            if num_format:
                spec["num_format"] = num_format
            if kind == _WRAP:
                spec |= {"text_wrap": True, "valign": "top", "font_size": 9.5}
            if kind == _DATE:
                spec["align"] = "left"
            self._by_kind[kind] = book.add_format(spec)

        # Bold twins of the numeric kinds for total rows.
        top_rule = {
            "top": 2,
            "top_color": _tint(pal["accent"], 0.4),
            "bottom": 1,
            "bottom_color": hairline,
        }
        self._totals: dict[str, Any] = {}
        for kind in (_MONEY, *_NUMBER_FORMATS):
            spec = {**base, **top_rule, "bold": True, "font_color": pal["text"]}
            num_format = money if kind == _MONEY else _NUMBER_FORMATS.get(kind, "")
            if num_format:
                spec["num_format"] = num_format
            self._totals[kind] = book.add_format(spec)

        # Insight severity keeps the app's semantics: red for critical, amber
        # for warning, green for positive, accent for neutral information.
        self.level_formats = {
            level: book.add_format({**cell, "bold": True, "font_color": colour})
            for level, colour in (
                ("critical", pal["loss"]),
                ("warning", pal["warning"]),
                ("positive", pal["gain"]),
                ("info", pal["accent"]),
            )
        }

    def cell(self, kind: str) -> Any:
        """Body-cell format for a display kind."""
        return self._by_kind.get(kind, self._by_kind[_TEXT])

    def total(self, kind: str) -> Any:
        """Total-row format for a display kind."""
        return self._totals.get(kind, self._totals[_TEXT])


# ---------------------------------------------------------------------------
# value coercion
# ---------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    """True for None, NaN, NaT, pd.NA and non-finite floats."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    if isinstance(value, (str, bytes, bool, np.bool_, int, np.integer)):
        return False
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series, pd.DataFrame)):
        return False
    try:
        flag = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(flag) if isinstance(flag, (bool, np.bool_)) else False


def _safe_int(value: Any, default: int = 0) -> int:
    """Integer view of a possibly-NaN, possibly-None, possibly-text value."""
    if _is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _strip_tz(value: datetime) -> datetime:
    """Excel has no timezone concept; store the UTC wall clock instead."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _write_cell(worksheet: Any, row: int, col: int, value: Any, fmt: Any) -> None:
    """Write one value with the right xlsxwriter primitive for its type.

    Missing values become genuinely empty (but still formatted) cells, which is
    what makes "no data" visually different from "zero" in the finished file.
    """
    if _is_missing(value):
        worksheet.write_blank(row, col, None, fmt)
        return
    if isinstance(value, (bool, np.bool_)):
        worksheet.write_string(row, col, "Yes" if bool(value) else "No", fmt)
        return
    if isinstance(value, (int, np.integer)):
        worksheet.write_number(row, col, int(value), fmt)
        return
    if isinstance(value, (float, np.floating)):
        worksheet.write_number(row, col, float(value), fmt)
        return
    if isinstance(value, pd.Timestamp):
        worksheet.write_datetime(row, col, _strip_tz(value.to_pydatetime()), fmt)
        return
    if isinstance(value, datetime):
        worksheet.write_datetime(row, col, _strip_tz(value), fmt)
        return
    if isinstance(value, date):
        worksheet.write_datetime(row, col, value, fmt)
        return
    if isinstance(value, str):
        worksheet.write_string(row, col, value[:_MAX_TEXT], fmt)
        return
    worksheet.write_string(row, col, str(value)[:_MAX_TEXT], fmt)


def _render_length(value: Any, kind: str) -> int:
    """Approximate rendered width of a value, for column sizing."""
    if _is_missing(value):
        return 0
    if kind == _DATE:
        return 12
    if isinstance(value, (bool, np.bool_)):
        return 3
    if isinstance(value, (int, np.integer, float, np.floating)):
        number = float(value)
        if kind == _PCT_DEC:
            return len(f"{number * 100:,.2f}") + 1
        if kind == _PCT_UNIT:
            return len(f"{number:,.2f}") + 1
        if kind == _INT:
            return len(f"{number:,.0f}")
        if kind == _MONEY:
            return len(f"{number:,.2f}") + 3  # allow for the currency symbol
        return len(f"{number:,.4f}")
    return len(str(value))


def _set_width(worksheet: Any, col: int, width: float) -> None:
    """Set a column width, never narrowing what an earlier section needed.

    Sheets stack several tables of different shapes in the same columns, and a
    plain ``set_column`` from the last table would squash the first one. Widths
    are therefore tracked per worksheet and only ever grow.
    """
    tracked = getattr(worksheet, "_qf_widths", None)
    if tracked is None:
        tracked = {}
        worksheet._qf_widths = tracked
    if width > tracked.get(col, 0.0):
        tracked[col] = float(width)
        worksheet.set_column(col, col, float(width))


# ---------------------------------------------------------------------------
# frame plumbing
# ---------------------------------------------------------------------------
def _pretty(name: Any) -> str:
    """Human column heading for a snake_case field name."""
    key = str(name)
    if key in _COLUMN_LABELS:
        return _COLUMN_LABELS[key]
    text = key.replace("_pct", " %").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else key


def _kind_for_value(name: Any, value: Any) -> str:
    """Display kind for a single scalar, using its type as a tie-breaker."""
    kind = _kind_for(name)
    if kind != _TEXT or isinstance(value, (bool, np.bool_)):
        return kind
    if isinstance(value, (int, np.integer)):
        return _INT
    if isinstance(value, (float, np.floating)):
        return _RATIO
    return kind


def _kind_for(name: Any, series: pd.Series | None = None) -> str:
    """Decide how a column should be displayed from its name and dtype."""
    key = str(name).lower()
    if series is not None and pd.api.types.is_datetime64_any_dtype(series):
        return _DATE
    if key in _DATE_COLUMNS or key.endswith("_date"):
        return _DATE
    if series is not None and pd.api.types.is_bool_dtype(series):
        return _TEXT
    if key in _WRAP_COLUMNS:
        return _WRAP
    if key in _MONEY_COLUMNS:
        return _MONEY
    if key in _PRICE_COLUMNS:
        return _PRICE
    if key in _FX_COLUMNS:
        return _FX
    if key in _QTY_COLUMNS:
        return _QTY
    if key in _INT_COLUMNS:
        return _INT
    if key.endswith("_pct") or key in _PCT_UNIT_COLUMNS:
        return _PCT_UNIT
    if key in _PCT_DEC_COLUMNS:
        return _PCT_DEC
    if key in _RATIO_COLUMNS:
        return _RATIO
    if series is not None and pd.api.types.is_numeric_dtype(series):
        return _RATIO
    return _TEXT


def _column_width(header: str, values: Sequence[Any], kind: str) -> float:
    """Width sized to the widest rendered cell, clamped to sane bounds."""
    widest = len(header)
    for value in values[:400]:
        widest = max(widest, _render_length(value, kind))
    if kind == _WRAP:
        return 46.0
    return float(min(max(widest + 3, 9), 40))


def _records_to_frame(records: Sequence[Any]) -> pd.DataFrame | None:
    """Best-effort DataFrame from dataclasses, dicts or tuples."""
    if not records:
        return None
    first = records[0]
    if is_dataclass(first) and not isinstance(first, type):
        extras = _PROPERTY_EXTRAS.get(type(first).__name__, ())
        rows: list[dict[str, Any]] = []
        for item in records:
            try:
                row = dict(asdict(item))
            except (TypeError, ValueError):
                row = {
                    field: getattr(item, field, None)
                    for field in getattr(item, "__dataclass_fields__", {})
                }
            for prop in extras:
                try:
                    row[prop] = getattr(item, prop)
                except Exception:  # noqa: BLE001 - defensive  # pragma: no cover
                    row[prop] = None
            rows.append(row)
        frame = pd.DataFrame(rows)
        if "is_long_term" in frame.columns:
            # Match the analytics layer's realised-trade schema, which reports
            # the tax term as text rather than a boolean.
            frame["term"] = np.where(
                frame["is_long_term"].fillna(False).astype(bool), "Long", "Short"
            )
            frame = frame.drop(columns="is_long_term")
        return frame
    if isinstance(first, Mapping):
        return pd.DataFrame([dict(item) for item in records])
    if isinstance(first, (tuple, list)):
        width = len(first)
        if width == 2:
            return pd.DataFrame(list(records), columns=["date", "cash_flow"])
        return pd.DataFrame(list(records), columns=[f"col_{i + 1}" for i in range(width)])
    return pd.DataFrame({"value": list(records)})


def _as_frame(obj: Any) -> pd.DataFrame | None:
    """Coerce whatever the pipeline stored into a table, or return None.

    Never mutates the caller's object: DataFrames are copied before use.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return None if obj.empty else obj.copy()
    if isinstance(obj, pd.Series):
        if obj.empty:
            return None
        frame = obj.rename(obj.name or "value").reset_index()
        frame.columns = [str(c) for c in frame.columns]
        return frame
    if isinstance(obj, (list, tuple)):
        return _records_to_frame(list(obj))
    return None


def _as_mapping(obj: Any) -> dict[str, Any]:
    """A plain dict view of a mapping-ish object; empty dict otherwise."""
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _field(result: Any, name: str, default: Any = None) -> Any:
    """Read a field from an AnalysisResult or from a plain dict alike."""
    if result is None:
        return default
    if isinstance(result, Mapping):
        value = result.get(name, default)
    else:
        value = getattr(result, name, default)
    return default if value is None else value


def _reorder(frame: pd.DataFrame, preferred: Sequence[str]) -> pd.DataFrame:
    """Put known columns in the documented order, keep extras on the right."""
    known = [c for c in preferred if c in frame.columns]
    extras = [c for c in frame.columns if c not in known]
    return frame[known + extras]


# ---------------------------------------------------------------------------
# table writer
# ---------------------------------------------------------------------------
def _write_table(
    worksheet: Any,
    fmts: _Formats,
    frame: pd.DataFrame,
    *,
    start_row: int = 0,
    start_col: int = 0,
    freeze_cols: int = 0,
    freeze: bool = True,
    autofilter: bool = True,
    total_row: bool = False,
    pretty_headers: bool = True,
    scale_columns: bool = True,
    set_widths: bool = True,
) -> int:
    """Write a DataFrame as a formatted, filterable table.

    Adds the frozen header, the autofilter, per-column number formats, a
    three-colour scale on signed P&L/return columns, data bars on weight-like
    columns and an optional SUBTOTAL row that follows the filter. Returns the
    first free row beneath the table.
    """
    data = frame.copy()
    if len(data) > _MAX_ROWS:
        data = data.head(_MAX_ROWS)
    if len(data.columns) == 0:
        return start_row
    columns = [str(c) for c in data.columns]
    kinds = [_kind_for(name, data.iloc[:, i]) for i, name in enumerate(columns)]
    # Column-wise, so each column keeps its own dtype (a row-wise iteration
    # collapses a mixed frame to object and turns integers into floats).
    values = [data.iloc[:, i].tolist() for i in range(len(columns))]
    n_rows = len(data)
    header_row = start_row
    first_data = header_row + 1
    last_data = header_row + max(n_rows, 1)

    for offset, (name, kind) in enumerate(zip(columns, kinds)):
        col = start_col + offset
        heading = _pretty(name) if pretty_headers else name
        numeric = kind in {_MONEY, _PRICE, _PCT_UNIT, _PCT_DEC, _RATIO, _INT, _QTY, _FX}
        worksheet.write_string(
            header_row,
            col,
            heading[:_MAX_TEXT],
            fmts.header_right if numeric else fmts.header,
        )
        if set_widths:
            _set_width(worksheet, col, _column_width(str(heading), values[offset], kind))
        cell_format = fmts.cell(kind)
        for r, value in enumerate(values[offset]):
            _write_cell(worksheet, first_data + r, col, value, cell_format)
    worksheet.set_row(header_row, 26)

    if n_rows and autofilter:
        worksheet.autofilter(header_row, start_col, last_data, start_col + len(columns) - 1)
    if freeze:
        worksheet.freeze_panes(first_data, start_col + freeze_cols)

    if n_rows and scale_columns:
        _apply_column_rules(worksheet, fmts, columns, first_data, last_data, start_col)

    next_row = last_data + 1
    if n_rows and total_row:
        _write_total_row(
            worksheet,
            fmts,
            data,
            columns,
            kinds,
            row=next_row,
            start_col=start_col,
            first_data=first_data,
            last_data=last_data,
        )
        next_row += 1
    return next_row + 1


def _apply_column_rules(
    worksheet: Any,
    fmts: _Formats,
    columns: Sequence[str],
    first_data: int,
    last_data: int,
    start_col: int,
) -> None:
    """Colour scales on signed columns, data bars on magnitude columns."""
    for offset, name in enumerate(columns):
        col = start_col + offset
        key = str(name).lower()
        if key in _SCALE_COLUMNS:
            worksheet.conditional_format(
                first_data,
                col,
                last_data,
                col,
                {
                    "type": "3_color_scale",
                    "min_type": "min",
                    "mid_type": "num",
                    "mid_value": 0,
                    "max_type": "max",
                    "min_color": fmts.scale_low,
                    "mid_color": fmts.scale_mid,
                    "max_color": fmts.scale_high,
                },
            )
        elif key in _BAR_COLUMNS:
            worksheet.conditional_format(
                first_data,
                col,
                last_data,
                col,
                {
                    "type": "data_bar",
                    "bar_color": fmts.bar,
                    "bar_solid": True,
                    "bar_border_color": fmts.bar,
                },
            )


def _write_total_row(
    worksheet: Any,
    fmts: _Formats,
    data: pd.DataFrame,
    columns: Sequence[str],
    kinds: Sequence[str],
    *,
    row: int,
    start_col: int,
    first_data: int,
    last_data: int,
) -> None:
    """A SUBTOTAL row, so the totals follow whatever the user filters to.

    Only genuinely additive columns are summed. A ratio such as ``pnl_pct`` is
    left blank rather than given a meaningless column sum.
    """
    worksheet.write_string(row, start_col, "Total", fmts.total(_TEXT))
    for offset, (name, kind) in enumerate(zip(columns, kinds)):
        if offset == 0:
            continue
        col = start_col + offset
        if str(name).lower() not in _ADDITIVE_COLUMNS:
            worksheet.write_blank(row, col, None, fmts.total(kind))
            continue
        series = pd.to_numeric(data.iloc[:, offset], errors="coerce")
        cached = float(series.sum(skipna=True)) if series.notna().any() else None
        cell_range = xl_range(first_data, col, last_data, col)
        if cached is None or not math.isfinite(cached):
            worksheet.write_blank(row, col, None, fmts.total(kind))
        else:
            worksheet.write_formula(
                row, col, f"=SUBTOTAL(109,{cell_range})", fmts.total(kind), cached
            )


def _write_metric_rows(
    worksheet: Any,
    fmts: _Formats,
    rows: Sequence[tuple[str, Any, str, str]],
    *,
    start_row: int,
    start_col: int = 0,
    with_notes: bool = True,
) -> int:
    """Write ``(label, value, kind, note)`` lines as a metric table body."""
    row = start_row
    for label, value, kind, note in rows:
        worksheet.write_string(row, start_col, str(label)[:_MAX_TEXT], fmts.label_strong)
        _write_cell(worksheet, row, start_col + 1, value, fmts.cell(kind))
        if with_notes:
            worksheet.write_string(
                row, start_col + 2, str(note)[:_MAX_TEXT] if note else "", fmts.note
            )
        row += 1
    return row


def _write_metric_sheet(
    worksheet: Any,
    fmts: _Formats,
    rows: Sequence[tuple[str, Any, str, str]],
    *,
    header_row: int,
) -> int:
    """Metric / Value / What it measures table with a frozen, filterable header."""
    worksheet.write_string(header_row, 0, "Metric", fmts.header)
    worksheet.write_string(header_row, 1, "Value", fmts.header_right)
    worksheet.write_string(header_row, 2, "What it measures", fmts.header)
    worksheet.set_row(header_row, 26)
    worksheet.freeze_panes(header_row + 1, 0)
    last = _write_metric_rows(worksheet, fmts, rows, start_row=header_row + 1)
    if rows:
        worksheet.autofilter(header_row, 0, last - 1, 2)
    return last + 1


def _metric_rows_from(
    source: Mapping[str, Any],
    catalogue: Sequence[tuple[str, str, str, str]],
) -> list[tuple[str, Any, str, str]]:
    """Pick the catalogued metrics that this run actually produced."""
    rows: list[tuple[str, Any, str, str]] = []
    for key, label, kind, note in catalogue:
        if key not in source:
            continue
        value = source.get(key)
        if _is_missing(value) and not isinstance(value, str):
            # Still show the row: an em-dash-equivalent blank is informative,
            # it says "we tried and it is undefined", not "we forgot".
            rows.append((label, None, kind, note))
            continue
        rows.append((label, value, kind, note))
    return rows


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------
def _stamp(value: Any) -> str:
    """ASCII timestamp such as ``01 Sep 2026 14:05 UTC``."""
    if _is_missing(value):
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        suffix = " UTC" if value.tzinfo is not None else ""
        return _strip_tz(value).strftime("%d %b %Y %H:%M") + suffix
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    return str(value)


def _page_setup(worksheet: Any, landscape: bool, header_row: int | None = None) -> None:
    """Print setup shared by every sheet: no gridlines, fitted width, footer."""
    worksheet.hide_gridlines(2)
    if landscape:
        worksheet.set_landscape()
    else:
        worksheet.set_portrait()
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(0.4, 0.4, 0.6, 0.6)
    worksheet.set_footer(
        f"&L&8{_escape_header(cfg.APP_NAME)} {cfg.APP_VERSION}"
        f"&C&8{_escape_header(cfg.DISCLAIMER)[:180]}&R&8Page &P of &N"
    )
    if header_row is not None:
        worksheet.repeat_rows(header_row)


def _escape_header(text: str) -> str:
    """Excel header/footer syntax treats ``&`` as an escape character."""
    return str(text).replace("&", "&&")


def _sheet_intro(
    worksheet: Any,
    fmts: _Formats,
    title: str,
    subtitle: str = "",
) -> int:
    """Compact per-sheet header line. Returns the first content row."""
    worksheet.set_row(0, 22)
    worksheet.write_string(0, 0, f"{cfg.APP_NAME} {cfg.APP_VERSION}  |  {title}", fmts.sheet_title)
    if subtitle:
        worksheet.write_string(1, 0, subtitle[:_MAX_TEXT], fmts.sheet_subtitle)
        return 3
    return 2


def _footnote(worksheet: Any, fmts: _Formats, row: int, text: str, width_cols: int = 6) -> int:
    """Methodology footnote under a table."""
    worksheet.merge_range(row, 0, row, max(width_cols - 1, 1), text[:_MAX_TEXT], fmts.footnote)
    worksheet.set_row(row, 26)
    return row + 2


# ---------------------------------------------------------------------------
# sheet builders
# ---------------------------------------------------------------------------
def _sheet_summary(
    worksheet: Any,
    fmts: _Formats,
    result: Any,
    config: Any,
    base_currency: str,
    notes: Sequence[str],
) -> None:
    """Title block, portfolio KPIs, run parameters and the insight log."""
    book_summary = _as_mapping(_field(result, "summary"))
    risk = _as_mapping(_field(result, "risk"))
    score = _as_mapping(_field(result, "score"))
    stats = _as_mapping(_field(result, "return_stats"))
    insights = _field(result, "insights", []) or []
    warnings = _field(result, "warnings", []) or []
    generated = _field(result, "generated_at")

    _set_width(worksheet, 0, 34)
    _set_width(worksheet, 1, 20)
    _set_width(worksheet, 2, 68)
    _set_width(worksheet, 3, 52)
    _page_setup(worksheet, landscape=False)

    # ---- title block ----
    worksheet.set_row(0, 30)
    worksheet.merge_range(0, 0, 0, 3, f"{cfg.APP_NAME} {cfg.APP_VERSION}", fmts.title)
    worksheet.merge_range(1, 0, 1, 3, cfg.APP_TAGLINE, fmts.tagline)
    stamp = _stamp(generated) or _stamp(datetime.now(timezone.utc))
    worksheet.merge_range(
        2,
        0,
        2,
        3,
        f"Portfolio analysed {stamp}  |  Base currency {base_currency}"
        f"  |  Source: {cfg.DATA_SOURCE}",
        fmts.meta,
    )
    worksheet.set_row(3, 30)
    worksheet.merge_range(3, 0, 3, 3, cfg.DISCLAIMER, fmts.disclaimer)
    row = 5

    def section(title: str, current: int) -> int:
        """Underlined section heading spanning the sheet's four columns."""
        worksheet.write_string(current, 0, title, fmts.section)
        for col in (1, 2, 3):
            worksheet.write_blank(current, col, None, fmts.section)
        return current + 1

    # ---- portfolio snapshot ----
    if book_summary.get("n_holdings"):
        row = section("Portfolio snapshot", row)
        best = book_summary.get("best")
        worst = book_summary.get("worst")
        largest = book_summary.get("largest_holding")
        snapshot: list[tuple[str, Any, str, str]] = [
            ("Holdings", book_summary.get("n_holdings"), _INT, ""),
            ("Invested", book_summary.get("total_investment"), _MONEY, ""),
            ("Market value", book_summary.get("current_value"), _MONEY, ""),
            ("Unrealised P&L", book_summary.get("total_pnl"), _MONEY, ""),
            ("Total return", book_summary.get("total_return_pct"), _PCT_UNIT, ""),
            ("Day change", book_summary.get("day_change"), _MONEY, ""),
            ("Day change %", book_summary.get("day_change_pct"), _PCT_UNIT, ""),
            (
                "Gainers / losers",
                f"{_safe_int(book_summary.get('n_gainers'))} up / "
                f"{_safe_int(book_summary.get('n_losers'))} down",
                _TEXT,
                "",
            ),
            ("Best performer", _pair_text(best), _TEXT, ""),
            ("Worst performer", _pair_text(worst), _TEXT, ""),
            ("Largest holding", _pair_text(largest, "%", signed=False), _TEXT, ""),
            (
                "Currencies held",
                ", ".join(str(c) for c in (book_summary.get("currencies") or [])) or None,
                _TEXT,
                "",
            ),
        ]
        row = _write_metric_rows(worksheet, fmts, snapshot, start_row=row, with_notes=False)
        row += 1

        row = section("Concentration", row)
        concentration: list[tuple[str, Any, str, str]] = [
            ("Herfindahl index (HHI)", book_summary.get("hhi"), _RATIO, ""),
            ("Effective holdings (1 / HHI)", book_summary.get("effective_holdings"), _RATIO, ""),
            ("Top 5 weight", book_summary.get("top5_weight_pct"), _PCT_UNIT, ""),
        ]
        sectors = book_summary.get("sectors") or {}
        for name, weight in list(sectors.items())[:6]:
            concentration.append((f"Sector: {name}", weight, _PCT_UNIT, ""))
        row = _write_metric_rows(worksheet, fmts, concentration, start_row=row, with_notes=False)
        row += 1

    # ---- headline risk ----
    headline_keys = {"ann_return", "ann_vol", "sharpe", "sortino", "mdd", "var", "cvar"}
    headline = _metric_rows_from(
        risk, tuple(spec for spec in _RISK_ROWS if spec[0] in headline_keys)
    )
    if headline or score:
        row = section("Headline risk", row)
        if headline:
            row = _write_metric_rows(worksheet, fmts, headline, start_row=row, with_notes=False)
        if score:
            band = score.get("band")
            row = _write_metric_rows(
                worksheet,
                fmts,
                [
                    ("Risk score (0-100)", score.get("score"), _RATIO, ""),
                    ("Risk band", band if isinstance(band, str) else None, _TEXT, ""),
                ],
                start_row=row,
                with_notes=False,
            )
        row += 1

    # ---- performance stats ----
    if stats:
        row = section("Performance", row)
        perf: list[tuple[str, Any, str, str]] = [
            ("Cumulative return", stats.get("cumulative_pct"), _PCT_UNIT, ""),
            ("Annualised return", stats.get("annualized_pct"), _PCT_UNIT, ""),
            ("Best month", stats.get("best_month_pct"), _PCT_UNIT, ""),
            ("Worst month", stats.get("worst_month_pct"), _PCT_UNIT, ""),
            (
                "Positive months",
                _ratio_text(stats.get("positive_months"), stats.get("total_months")),
                _TEXT,
                "",
            ),
            ("Best day", stats.get("best_day_pct"), _PCT_UNIT, ""),
            ("Worst day", stats.get("worst_day_pct"), _PCT_UNIT, ""),
            ("Longest winning streak", stats.get("longest_win_streak"), _INT, ""),
            ("Longest losing streak", stats.get("longest_loss_streak"), _INT, ""),
        ]
        row = _write_metric_rows(worksheet, fmts, perf, start_row=row, with_notes=False)
        row += 1

    # ---- run parameters ----
    row = section("Run parameters", row)
    period = _field(config, "period")
    params: list[tuple[str, Any, str, str]] = [
        ("History window", cfg.PERIOD_LABELS.get(str(period), period), _TEXT, ""),
        ("Benchmark", _field(config, "benchmark_name"), _TEXT, ""),
        ("Base currency", base_currency, _TEXT, ""),
        ("Risk-free rate (annual)", _field(config, "risk_free_rate"), _PCT_DEC, ""),
        ("VaR confidence", _field(config, "var_confidence"), _PCT_DEC, ""),
        ("FX conversion", _field(config, "convert_fx"), _TEXT, ""),
        ("Rolling window (days)", _field(config, "rolling_window"), _INT, ""),
        ("Trading days per year", cfg.TRADING_DAYS, _INT, ""),
        ("Generated", stamp, _TEXT, ""),
    ]
    row = _write_metric_rows(worksheet, fmts, params, start_row=row, with_notes=False)
    row += 1

    # ---- insights ----
    insight_frame = _insights_frame(insights)
    if insight_frame is not None:
        row = section("Insights", row)
        first_data = row + 1
        row = _write_table(
            worksheet,
            fmts,
            insight_frame,
            start_row=row,
            freeze=False,
            autofilter=True,
            scale_columns=False,
        )
        last_data = first_data + len(insight_frame) - 1
        for level, fmt in fmts.level_formats.items():
            worksheet.conditional_format(
                first_data,
                0,
                last_data,
                0,
                {
                    "type": "text",
                    "criteria": "containing",
                    "value": level.title(),
                    "format": fmt,
                },
            )

    # ---- warnings and export notes ----
    messages = [str(w) for w in warnings if str(w).strip()]
    messages += [str(n) for n in notes if str(n).strip()]
    if messages:
        row = section("Warnings and export notes", row)
        for message in messages[:80]:
            worksheet.merge_range(row, 0, row, 3, message[:_MAX_TEXT], fmts.note)
            row += 1

    worksheet.set_tab_color(_palette()["accent"])


def _pair_text(pair: Any, suffix: str = "%", signed: bool = True) -> str | None:
    """Render a ``(ticker, value)`` tuple as ``TICKER +12.34%``.

    ``signed`` is False for magnitudes such as a weight, where a leading plus
    would read as a gain rather than a share of the book.
    """
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        return None
    ticker, value = pair
    if _is_missing(value):
        return str(ticker)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(ticker)
    return f"{ticker}  {number:+,.2f}{suffix}" if signed else f"{ticker}  {number:,.2f}{suffix}"


def _ratio_text(numerator: Any, denominator: Any) -> str | None:
    """``23 of 36`` style text, or None when either side is missing."""
    if _is_missing(numerator) or _is_missing(denominator):
        return None
    try:
        return f"{int(numerator)} of {int(denominator)}"
    except (TypeError, ValueError):
        return None


def _insights_frame(insights: Any) -> pd.DataFrame | None:
    """Insight objects (or dicts) as a Level/Category/Observation/Evidence table."""
    if not isinstance(insights, (list, tuple)) or not insights:
        return None
    rows: list[dict[str, Any]] = []
    order = {"critical": 0, "warning": 1, "positive": 2, "info": 3}
    for item in insights:
        if isinstance(item, Mapping):
            level = item.get("level", "info")
            text = item.get("text", "")
            category = item.get("category", "general")
            evidence = item.get("evidence", "")
        else:
            level = getattr(item, "level", "info")
            text = getattr(item, "text", "")
            category = getattr(item, "category", "general")
            evidence = getattr(item, "evidence", "")
        rows.append(
            {
                "level": str(level).title(),
                "category": str(category).title(),
                "observation": str(text),
                "evidence": str(evidence),
                "_order": order.get(str(level).lower(), 4),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("_order", kind="stable")
    return frame.drop(columns="_order").reset_index(drop=True)


def _sheet_holdings(
    worksheet: Any,
    fmts: _Formats,
    positions: pd.DataFrame,
    base_currency: str,
    stamp: str,
) -> None:
    """Position-level table: quantities, prices, FX, value, P&L and weight."""
    order = (
        "ticker",
        "name",
        "sector",
        "industry",
        "quantity",
        "buy_price",
        "current_price",
        "currency",
        "fx_rate",
        "day_change_pct",
        "investment",
        "current_value",
        "pnl",
        "pnl_pct",
        "weight_pct",
        "buy_date",
        "target_weight_pct",
    )
    frame = _reorder(positions, order)
    subtitle = (
        f"Marked at {stamp}. Money columns are in {base_currency}; prices are "
        f"in each instrument's own currency (see the Ccy column)."
    )
    row = _sheet_intro(worksheet, fmts, "Holdings", subtitle)
    _page_setup(worksheet, landscape=True, header_row=row)
    row = _write_table(worksheet, fmts, frame, start_row=row, freeze_cols=1, total_row=True)
    _footnote(
        worksheet,
        fmts,
        row,
        "Invested, market value and P&L are converted to the base currency at "
        "the FX rate on each row; return % is FX-neutral because both legs of "
        "the ratio use the same rate. The total row is a SUBTOTAL, so it "
        "recalculates when you filter the table.",
        width_cols=min(len(frame.columns), 8),
    )
    worksheet.set_tab_color(_palette()["accent_2"])


def _sheet_risk(
    worksheet: Any,
    fmts: _Formats,
    risk: Mapping[str, Any],
    score: Mapping[str, Any],
    n_days: int,
) -> None:
    """Every headline risk statistic with its unit and its meaning."""
    confidence = risk.get("confidence", cfg.DEFAULT_VAR_CONFIDENCE)
    try:
        conf_text = f"{float(confidence) * 100:.0f}%"
    except (TypeError, ValueError):
        conf_text = f"{cfg.DEFAULT_VAR_CONFIDENCE * 100:.0f}%"
    observations = _safe_int(risk.get("n_obs"), _safe_int(n_days))
    subtitle = (
        (f"Daily observations: {observations}. " if observations else "")
        + f"VaR and expected shortfall are one-day losses at {conf_text} "
        f"confidence; returns are annualised with {cfg.TRADING_DAYS} trading "
        f"days."
    )
    row = _sheet_intro(worksheet, fmts, "Risk", subtitle)
    _set_width(worksheet, 0, 34)
    _set_width(worksheet, 1, 16)
    _set_width(worksheet, 2, 82)
    _page_setup(worksheet, landscape=False, header_row=row)

    row = _write_metric_sheet(worksheet, fmts, _metric_rows_from(risk, _RISK_ROWS), header_row=row)

    if score:
        worksheet.write_string(row, 0, "Risk score", fmts.section)
        worksheet.write_blank(row, 1, None, fmts.section)
        worksheet.write_blank(row, 2, None, fmts.section)
        row += 1
        components = _as_mapping(score.get("components"))
        rows: list[tuple[str, Any, str, str]] = [
            (
                "Score (0-100)",
                score.get("score"),
                _RATIO,
                "Rubric-weighted composite; higher means more risk taken.",
            ),
            (
                "Band",
                score.get("band"),
                _TEXT,
                f"Bands from config.RISK_RUBRIC: below "
                f"{cfg.RISK_RUBRIC.low_band:.0f} Low, up to "
                f"{cfg.RISK_RUBRIC.high_band:.0f} Moderate, above it High.",
            ),
            (
                "Points available",
                score.get("max_points"),
                _RATIO,
                "Components with missing inputs are dropped from BOTH the score "
                "and this denominator, so a gap never flatters the portfolio.",
            ),
        ]
        for name, points in components.items():
            rows.append(
                (
                    f"Component: {str(name).replace('_', ' ')}",
                    points,
                    _RATIO,
                    "Raw points earned, before rescaling.",
                )
            )
        row = _write_metric_rows(worksheet, fmts, rows, start_row=row)
        row += 1

    _footnote(
        worksheet,
        fmts,
        row,
        "Undefined statistics are left blank rather than reported as zero. "
        "Ratios use the risk-free rate from the run parameters on the Summary "
        "sheet.",
        width_cols=3,
    )
    worksheet.set_tab_color(_palette()["warning"])


def _sheet_benchmark(
    worksheet: Any,
    fmts: _Formats,
    benchmark: Mapping[str, Any],
    benchmark_name: str,
) -> None:
    """Relative performance: excess return, alpha/beta, capture and tracking."""
    n_days = benchmark.get("n_days")
    subtitle = (
        f"Portfolio versus {benchmark_name} on the overlapping window"
        + (f" ({_safe_int(n_days)} days)" if not _is_missing(n_days) else "")
        + ". Both sides use identical dates and the same annualisation."
    )
    row = _sheet_intro(worksheet, fmts, "Benchmark", subtitle)
    _set_width(worksheet, 0, 34)
    _set_width(worksheet, 1, 16)
    _set_width(worksheet, 2, 82)
    _page_setup(worksheet, landscape=False, header_row=row)

    if benchmark.get("ok") is False:
        reason = str(benchmark.get("reason") or "The benchmark comparison could not be computed.")
        worksheet.merge_range(row, 0, row, 2, reason, fmts.note)
        worksheet.set_tab_color(_palette()["benchmark"])
        return

    row = _write_metric_sheet(
        worksheet,
        fmts,
        _metric_rows_from(benchmark, _BENCHMARK_ROWS),
        header_row=row,
    )

    methodology = benchmark.get("alpha_methodology")
    if isinstance(methodology, str) and methodology.strip():
        row = _footnote(worksheet, fmts, row, methodology, width_cols=3)
    worksheet.set_tab_color(_palette()["benchmark"])


def _sheet_contribution(
    worksheet: Any,
    fmts: _Formats,
    attribution: Mapping[str, Any],
    base_currency: str,
) -> None:
    """Who moved the portfolio, and who carries the risk that moved it."""
    subtitle = (
        f"Return contributions sum to the portfolio return; risk "
        f"contributions sum to 100% of portfolio volatility. Money in "
        f"{base_currency}."
    )
    row = _sheet_intro(worksheet, fmts, "Contribution", subtitle)
    _page_setup(worksheet, landscape=True)

    concentration = _as_mapping(attribution.get("concentration"))
    if concentration:
        worksheet.write_string(row, 0, "Concentration", fmts.section)
        for col in (1, 2):
            worksheet.write_blank(row, col, None, fmts.section)
        row += 1
        _set_width(worksheet, 0, 30)
        _set_width(worksheet, 1, 16)
        row = _write_metric_rows(
            worksheet,
            fmts,
            _metric_rows_from(concentration, _CONCENTRATION_ROWS),
            start_row=row,
            with_notes=False,
        )
        row += 1

    blocks: tuple[tuple[str, str, Sequence[str]], ...] = (
        (
            "return_contribution",
            "Return contribution",
            (
                "ticker",
                "weight_pct",
                "pnl",
                "pnl_pct",
                "contribution_pct",
                "contribution_share_pct",
            ),
        ),
        (
            "risk_contribution",
            "Risk contribution",
            ("ticker", "weight_pct", "mctr", "cctr", "risk_contribution_pct"),
        ),
        (
            "risk_vs_weight",
            "Risk carried versus weight held",
            ("ticker", "weight_pct", "risk_contribution_pct", "ratio"),
        ),
        ("top_contributors", "Top contributors", ()),
        ("top_detractors", "Top detractors", ()),
    )
    wrote_any = False
    for key, heading, order in blocks:
        frame = _as_frame(attribution.get(key))
        if frame is None:
            continue
        if order:
            frame = _reorder(frame, order)
        worksheet.write_string(row, 0, heading, fmts.section)
        for col in range(1, max(len(frame.columns), 2)):
            worksheet.write_blank(row, col, None, fmts.section)
        row = _write_table(
            worksheet,
            fmts,
            frame,
            start_row=row + 1,
            freeze=not wrote_any,
            freeze_cols=1,
            total_row=key in {"return_contribution", "risk_contribution"},
        )
        wrote_any = True

    if not wrote_any and not concentration:
        worksheet.write_string(
            row,
            0,
            "No attribution was computed for this run - it needs both a "
            "positions table and per-asset returns.",
            fmts.note,
        )
    if wrote_any:
        _footnote(
            worksheet,
            fmts,
            row,
            "Contribution % is each position's P&L over total invested capital, "
            "so the column sums to the portfolio's total return. Risk "
            "contribution splits portfolio volatility by marginal contribution "
            "(w_i * (Sigma w)_i / sigma_p) and assumes the covariance matrix of "
            "the analysis window persists. Concentration statistics above use "
            "portfolio weights, where HHI is the sum of squared weights and "
            "effective holdings is its reciprocal.",
            width_cols=6,
        )
    worksheet.set_tab_color(theme.CATEGORICAL[0])


def _sheet_correlation(
    worksheet: Any,
    fmts: _Formats,
    corr: pd.DataFrame,
    n_days: int,
) -> None:
    """Pairwise correlation of daily returns, with a diverging colour scale."""
    matrix = corr.copy()
    matrix.index = [str(i) for i in matrix.index]
    matrix.columns = [str(c) for c in matrix.columns]
    subtitle = (
        "Pearson correlation of daily returns"
        + (f" over {_safe_int(n_days)} overlapping days" if _safe_int(n_days) else "")
        + ". Colour matches the dashboard heatmap: blue is co-movement, red is "
        "the diversifying end."
    )
    row = _sheet_intro(worksheet, fmts, "Correlation", subtitle)
    _page_setup(worksheet, landscape=True, header_row=row)

    flat = matrix.rename_axis("Ticker").reset_index()
    header_row = row
    first_data = header_row + 1
    last_data = header_row + len(flat)
    row = _write_table(
        worksheet,
        fmts,
        flat,
        start_row=header_row,
        freeze_cols=1,
        autofilter=False,
        pretty_headers=False,
        scale_columns=False,
    )
    if len(matrix.columns):
        worksheet.conditional_format(
            first_data,
            1,
            last_data,
            len(matrix.columns),
            {
                "type": "3_color_scale",
                "min_type": "num",
                "min_value": -1,
                "min_color": fmts.corr_low,
                "mid_type": "num",
                "mid_value": 0,
                "mid_color": fmts.scale_mid,
                "max_type": "num",
                "max_value": 1,
                "max_color": fmts.corr_high,
            },
        )
    _footnote(
        worksheet,
        fmts,
        row,
        "Correlation is measured on simple daily returns over the analysis "
        "window only; it is not stable through regime changes and typically "
        "rises in a crisis, exactly when diversification is needed most.",
        width_cols=min(len(matrix.columns) + 1, 8),
    )
    worksheet.set_tab_color(theme.CATEGORICAL[4])


def _sheet_drawdowns(worksheet: Any, fmts: _Formats, frame: pd.DataFrame) -> None:
    """Peak to trough to recovery episodes, deepest first."""
    subtitle = (
        "Each episode runs from an all-time high to its trough and ends when a "
        "new high is reached. Day counts are calendar days; a blank recovery "
        "date means the portfolio was still underwater at the end of the window."
    )
    row = _sheet_intro(worksheet, fmts, "Drawdowns", subtitle)
    _page_setup(worksheet, landscape=True, header_row=row)
    row = _write_table(worksheet, fmts, frame, start_row=row, freeze_cols=0)
    _footnote(
        worksheet,
        fmts,
        row,
        "Depth is measured on the portfolio value series, which reflects the "
        "current holdings held constant across the window rather than the "
        "trades actually made.",
        width_cols=min(len(frame.columns), 8),
    )
    worksheet.set_tab_color(_palette()["loss"])


def _episodes_frame(episodes: Any) -> pd.DataFrame | None:
    """DrawdownEpisode objects (or a ready frame) as a table."""
    frame = _as_frame(episodes)
    if frame is None:
        return None
    if "depth" in frame.columns:
        order = (
            "peak_date",
            "trough_date",
            "recovery_date",
            "depth",
            "drawdown_days",
            "recovery_days",
            "total_days",
            "status",
        )
        if "total_days" not in frame.columns and {"drawdown_days", "recovery_days"} <= set(
            frame.columns
        ):
            frame["total_days"] = pd.to_numeric(
                frame["drawdown_days"], errors="coerce"
            ) + pd.to_numeric(frame["recovery_days"], errors="coerce")
        if "status" not in frame.columns and "recovery_date" in frame.columns:
            frame["status"] = np.where(frame["recovery_date"].isna(), "Underwater", "Recovered")
        frame = _reorder(frame, order)
    return frame


def _sheet_transactions(
    worksheet: Any,
    fmts: _Formats,
    ledger: Mapping[str, Any],
    base_currency: str,
) -> None:
    """The FIFO ledger: trades, open lots, realised P&L and cash flows."""
    subtitle = (
        f"FIFO tax-lot accounting. Buy fees are capitalised into the cost "
        f"basis and sell fees are apportioned pro rata across the matched "
        f"slices. Money in {base_currency}."
    )
    row = _sheet_intro(worksheet, fmts, "Transactions", subtitle)
    _page_setup(worksheet, landscape=True)

    written: set[int] = set()
    wrote_any = False
    for key, heading in _LEDGER_SECTIONS:
        value = ledger.get(key)
        if value is None or id(value) in written:
            continue
        frame = _as_frame(value)
        mapping = _as_mapping(value)
        if frame is None and not mapping:
            continue
        if frame is not None and key == "problems" and list(frame.columns) == ["value"]:
            frame = frame.rename(columns={"value": "problem"})
        written.add(id(value))
        worksheet.write_string(row, 0, heading, fmts.section)
        span = len(frame.columns) if frame is not None else 3
        for col in range(1, max(span, 2)):
            worksheet.write_blank(row, col, None, fmts.section)
        row += 1
        if frame is not None:
            row = _write_table(
                worksheet,
                fmts,
                frame,
                start_row=row,
                freeze=not wrote_any,
                freeze_cols=1,
                total_row=key in {"realised", "realised_trades", "lots"},
            )
        else:
            _set_width(worksheet, 0, 30)
            _set_width(worksheet, 1, 18)
            rows = [
                (_pretty(name), val, _kind_for_value(name, val), "")
                for name, val in mapping.items()
                if not isinstance(val, (pd.DataFrame, pd.Series, np.ndarray, list, dict))
            ]
            row = _write_metric_rows(worksheet, fmts, rows, start_row=row, with_notes=False) + 1
        wrote_any = True

    xirr = ledger.get("xirr")
    if not _is_missing(xirr):
        wrote_any = True
        worksheet.write_string(row, 0, "Money-weighted return", fmts.section)
        worksheet.write_blank(row, 1, None, fmts.section)
        row = _write_metric_rows(
            worksheet,
            fmts,
            [
                (
                    "XIRR (annualised)",
                    xirr,
                    _PCT_DEC,
                    "Internal rate of return on the dated cash flows, ACT/365.",
                )
            ],
            start_row=row + 1,
        )
        row += 1

    if not wrote_any:
        worksheet.write_string(
            row,
            0,
            "No transaction ledger was supplied, so cost basis and realised "
            "P&L could not be reconstructed.",
            fmts.note,
        )
    else:
        _footnote(
            worksheet,
            fmts,
            row,
            "Realised P&L is net of fees on both legs. Holding periods are "
            "calendar days; the Term column applies the Indian listed-equity "
            "convention of long term beyond 365 days, which may not match your "
            "jurisdiction.",
            width_cols=8,
        )
    worksheet.set_tab_color(theme.CATEGORICAL[7])


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _attribution_has_content(attribution: Mapping[str, Any]) -> bool:
    """True when attribution holds at least one table or concentration stat."""
    if _as_mapping(attribution.get("concentration")):
        return True
    return any(
        _as_frame(attribution.get(key)) is not None
        for key in (
            "return_contribution",
            "risk_contribution",
            "risk_vs_weight",
            "top_contributors",
            "top_detractors",
        )
    )


def _ledger_has_content(ledger: Mapping[str, Any]) -> bool:
    """True when the FIFO ledger holds at least one table, stat or XIRR."""
    if not _is_missing(ledger.get("xirr")):
        return True
    for key, _heading in _LEDGER_SECTIONS:
        value = ledger.get(key)
        if _as_frame(value) is not None or _as_mapping(value):
            return True
    return False


def _correlation_is_usable(corr: pd.DataFrame | None) -> bool:
    """True when the matrix has at least one finite off-diagonal correlation.

    An all-NaN matrix (too little overlapping history) would render as a grid of
    empty cells, which says less than not shipping the sheet at all.
    """
    if corr is None or len(corr.columns) < 2 or len(corr.index) < 2:
        return False
    try:
        values = corr.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    except (TypeError, ValueError):
        return False
    if values.shape[0] != values.shape[1]:
        return bool(np.isfinite(values).any())
    np.fill_diagonal(values, np.nan)
    return bool(np.isfinite(values).any())


def _base_currency_of(result: Any, config: Any) -> str:
    """Base currency from the run config, the summary, or the default."""
    for candidate in (
        _field(config, "base_currency"),
        _as_mapping(_field(result, "summary")).get("base_currency"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().upper()
    return cfg.DEFAULT_BASE_CURRENCY


def _currency_symbol_of(base_currency: str, config: Any) -> str:
    """Currency symbol for the base currency (empty string when unknown)."""
    symbol = _field(config, "currency_symbol")
    if isinstance(symbol, str) and symbol:
        return symbol
    return cfg.CURRENCY_SYMBOLS.get(base_currency, "")


def _set_properties(book: xlsxwriter.Workbook, stamp: str, base: str) -> None:
    """Document properties, so the file identifies itself in Explorer too."""
    book.set_properties(
        {
            "title": f"{cfg.APP_NAME} portfolio analysis",
            "subject": cfg.APP_TAGLINE,
            "author": f"{cfg.APP_NAME} {cfg.APP_VERSION}",
            "company": cfg.APP_NAME,
            "category": "Portfolio analytics",
            "keywords": f"portfolio, risk, performance, {base}",
            "comments": f"Generated {stamp}. {cfg.DISCLAIMER}",
        }
    )


def build_workbook(result: Any, config: Any = None) -> bytes:
    """Render one analysis run as a formatted multi-sheet .xlsx and return bytes.

    Parameters
    ----------
    result:
        An :class:`~src.models.AnalysisResult` (or any object/mapping exposing
        the same field names). Every field may be ``None`` or empty: sheets
        whose section is missing are simply not created, and the workbook
        always contains at least the ``Summary`` sheet with its title block and
        disclaimer.
    config:
        The :class:`~src.config.AppConfig` used for the run, or ``None``. It
        supplies the base currency, benchmark name, risk-free rate and window
        that are printed as the run parameters, so the reader can see the
        assumptions the numbers were produced under.

    Returns
    -------
    bytes
        The complete ``.xlsx`` file, built in memory - suitable for
        ``st.download_button`` or an email attachment.

    Notes
    -----
    Sheets follow :data:`SHEETS` order. Money columns are in the base currency,
    per-share prices stay in each instrument's native currency, and any
    undefined statistic is written as a blank cell rather than a zero. A failure
    while writing one sheet is caught and reported in the Summary sheet's
    "Warnings and export notes" section rather than losing the whole workbook.
    """
    base_currency = _base_currency_of(result, config)
    symbol = _currency_symbol_of(base_currency, config)
    generated = _field(result, "generated_at")
    stamp = _stamp(generated) or _stamp(datetime.now(timezone.utc))

    buffer = io.BytesIO()
    book = xlsxwriter.Workbook(
        buffer, {"in_memory": True, "remove_timezone": True, "default_date_format": "dd mmm yyyy"}
    )
    try:
        _set_properties(book, stamp, base_currency)
        fmts = _Formats(book, symbol)
        notes: list[str] = []

        positions = _as_frame(_field(result, "positions"))
        risk = _as_mapping(_field(result, "risk"))
        score = _as_mapping(_field(result, "score"))
        benchmark = _as_mapping(_field(result, "benchmark"))
        attribution = _as_mapping(_field(result, "attribution"))
        correlation = _as_frame(_field(result, "correlation"))
        episodes = _episodes_frame(_field(result, "drawdowns"))
        ledger = _as_mapping(_field(result, "ledger"))
        n_days = _safe_int(_field(result, "n_days"))
        benchmark_name = str(_field(config, "benchmark_name") or cfg.DEFAULT_BENCHMARK)

        plan: list[tuple[str, Callable[[Any], None]]] = []
        if positions is not None:
            plan.append(
                ("Holdings", lambda ws: _sheet_holdings(ws, fmts, positions, base_currency, stamp))
            )
        if risk or score:
            plan.append(("Risk", lambda ws: _sheet_risk(ws, fmts, risk, score, n_days)))
        if benchmark:
            plan.append(
                ("Benchmark", lambda ws: _sheet_benchmark(ws, fmts, benchmark, benchmark_name))
            )
        if _attribution_has_content(attribution):
            plan.append(
                (
                    "Contribution",
                    lambda ws: _sheet_contribution(ws, fmts, attribution, base_currency),
                )
            )
        if _correlation_is_usable(correlation):
            plan.append(
                ("Correlation", lambda ws: _sheet_correlation(ws, fmts, correlation, n_days))
            )
        if episodes is not None:
            plan.append(("Drawdowns", lambda ws: _sheet_drawdowns(ws, fmts, episodes)))
        if _ledger_has_content(ledger):
            plan.append(
                ("Transactions", lambda ws: _sheet_transactions(ws, fmts, ledger, base_currency))
            )

        # Sheets are created in SHEETS order first so the tab strip reads
        # correctly, then populated - Summary last, because it reports on any
        # section that failed to write.
        summary_sheet = book.add_worksheet("Summary")
        built = [(name, book.add_worksheet(name), builder) for name, builder in plan]
        for name, worksheet, builder in built:
            try:
                builder(worksheet)
            except Exception as exc:  # noqa: BLE001 - defensive  # pragma: no cover
                notes.append(f"{name}: sheet could not be completed ({type(exc).__name__}: {exc}).")
        try:
            _sheet_summary(summary_sheet, fmts, result, config, base_currency, notes)
        except Exception as exc:  # noqa: BLE001 - defensive  # pragma: no cover
            summary_sheet.write_string(0, 0, f"{cfg.APP_NAME} {cfg.APP_VERSION}")
            summary_sheet.write_string(
                1, 0, f"Summary could not be completed ({type(exc).__name__}: {exc})."
            )
            summary_sheet.write_string(2, 0, cfg.DISCLAIMER)
        summary_sheet.activate()
    finally:
        book.close()
    return buffer.getvalue()


def positions_csv(positions: Any) -> bytes:
    """Serialise the positions table as UTF-8 CSV bytes for a raw-data download.

    Columns are emitted in the documented ``build_positions`` order with any
    extra columns kept on the right. Missing values are written as empty
    fields, never as zeros, and floats use fixed notation (no scientific
    exponents) so the file loads cleanly into Excel or a database. A ``None``
    or empty table yields a header-only CSV rather than an empty file, so a
    download always produces something openable.

    The output is encoded ``utf-8-sig``: the byte-order mark is what makes
    Excel on Windows read non-ASCII tickers and names correctly.
    """
    order = (
        "ticker",
        "name",
        "sector",
        "industry",
        "quantity",
        "buy_price",
        "current_price",
        "currency",
        "fx_rate",
        "day_change_pct",
        "investment",
        "current_value",
        "pnl",
        "pnl_pct",
        "weight_pct",
        "buy_date",
        "target_weight_pct",
    )
    frame = _as_frame(positions)
    if frame is None:
        frame = pd.DataFrame(columns=list(order))
    else:
        frame = _reorder(frame, order)
    text = frame.to_csv(index=False, float_format="%.6f")
    return text.encode("utf-8-sig")
