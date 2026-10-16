"""Balance-sheet, income-statement and cash-flow analysis — scores, not forecasts.

What this module is, and deliberately is not
--------------------------------------------
It computes **published, documented scores** from financial statements:
Piotroski's F-Score, Altman's Z-Score, Beneish's M-Score, Sloan's accruals
ratio, return on invested capital and leverage trends. Every one of them is a
*measurement of what the accounts say*, with a citation and a threshold from
the original paper.

None of them is a return forecast, and this module contains no model that
predicts a price. That separation is the point. A fundamental score answers
"how strong do these accounts look, on a published rule" — which is
answerable. "What will this stock return over ten years" is not answerable
from five annual statements, and :func:`horizon_availability` says so per
horizon, with the arithmetic, rather than emitting a number.

The rule every score here obeys
-------------------------------
**A score computed from incomplete inputs is reported as incomplete, never as
a low score.** Piotroski's F-Score is nine binary signals; if only six can be
computed, the answer is "4 of 6 signals passed, 3 unavailable", never "4 out
of 9". Treating a missing signal as a failed one is the single easiest way to
make a healthy company look distressed, and it is what most implementations
of this score do wrong.

So every result carries ``available`` alongside its value, and the UI shows
both. A score with too few inputs to mean anything returns ``None`` for its
verdict rather than guessing.

Data limitations that shape the results
---------------------------------------
* **Five annual periods, at most.** That is what the provider gives. Piotroski
  and Beneish both need two consecutive years, so the oldest period yields no
  score, and a newly listed company may yield none at all.
* **The statements are RESTATED, not point-in-time.** A figure labelled 2022
  is today's view of 2022, including any restatement published since. That is
  fine for "how do these accounts look now" and fatal for backtesting, which
  is why :func:`horizon_availability` reports what could be trained and
  what could not.
* **Interest expense is frequently absent** for a debt-free company — and
  absent is not zero. An interest-cover ratio is reported as unavailable
  rather than as infinity.
* **Capital expenditure arrives negative** (a cash outflow) and is used with
  its sign made explicit at each site, because a sign error here silently
  inverts free cash flow.

Pure module: pandas and stdlib only, no Streamlit and no network. Statements
are passed IN as frames, so every score is testable against hand-built
fixtures with known answers.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

__all__ = [
    "LINE_ITEMS",
    "REQUIRED_FOR_PIOTROSKI",
    "PIOTROSKI_SIGNALS",
    "BENEISH_VARIABLES",
    "ALTMAN_EM_CONSTANT",
    "EMERGING_MARKETS",
    "RETURN_HORIZONS",
    "MIN_PERIODS_FOR_TREND",
    "FiscalPeriod",
    "Statements",
    "Signal",
    "ScoreResult",
    "RatioSet",
    "line_item",
    "read_statements",
    "piotroski",
    "altman_z",
    "beneish_m",
    "accruals_ratio",
    "ratios",
    "QualityReport",
    "quality_summary",
    "HorizonCheck",
    "horizon_availability",
    "BALANCE_SHEET_LAYOUT",
    "BALANCE_SHEET_TOTALS",
    "BALANCE_SHEET_COUNT_FIELDS",
    "BALANCE_SHEET_TOLERANCE",
    "balance_sheet_statement",
    "balance_sheet_check",
]

#: Canonical field -> the provider's row labels, best first.
#:
#: **Measured, not guessed.** Every label here was confirmed present on both an
#: NSE and a US statement before being relied on; a mistyped label does not
#: raise, it silently yields NaN and quietly zeroes a score. Several fields
#: have genuine alternates (``Receivables`` appears on US filings,
#: ``Accounts Receivable`` on Indian ones), which is why this is a list per
#: field rather than a single string.
LINE_ITEMS: dict[str, tuple[str, ...]] = {
    # -- balance sheet ---------------------------------------------------
    "total_assets": ("Total Assets",),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "total_liabilities": ("Total Liabilities Net Minority Interest", "Total Liabilities"),
    "long_term_debt": ("Long Term Debt",),
    "total_debt": ("Total Debt",),
    "equity": ("Stockholders Equity", "Common Stock Equity"),
    "retained_earnings": ("Retained Earnings",),
    "working_capital": ("Working Capital",),
    "shares": ("Ordinary Shares Number", "Share Issued"),
    "receivables": ("Accounts Receivable", "Receivables"),
    "inventory": ("Inventory",),
    "net_ppe": ("Net PPE",),
    "gross_ppe": ("Gross PPE",),
    "cash": ("Cash And Cash Equivalents",),
    "short_term_investments": ("Other Short Term Investments",),
    "invested_capital": ("Invested Capital",),
    # -- income statement ------------------------------------------------
    "revenue": ("Total Revenue",),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income",),
    "ebit": ("EBIT", "Operating Income"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "pretax_income": ("Pretax Income",),
    "tax": ("Tax Provision",),
    "interest_expense": ("Interest Expense",),
    "sga": ("Selling General And Administration",),
    # -- cash flow -------------------------------------------------------
    "operating_cash_flow": ("Operating Cash Flow",),
    "free_cash_flow": ("Free Cash Flow",),
    "capex": ("Capital Expenditure",),
    "depreciation": (
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
        "Reconciled Depreciation",
    ),
}

#: Fields without which no Piotroski signal can be computed at all.
REQUIRED_FOR_PIOTROSKI: tuple[str, ...] = ("total_assets", "net_income")

#: Piotroski's nine signals, in the order of the 2000 paper, each with the
#: dimension it belongs to. Names are the paper's.
PIOTROSKI_SIGNALS: tuple[tuple[str, str, str], ...] = (
    ("roa_positive", "Profitability", "Return on assets is positive"),
    ("cfo_positive", "Profitability", "Operating cash flow is positive"),
    ("roa_improving", "Profitability", "Return on assets improved year on year"),
    ("accruals", "Profitability", "Operating cash flow exceeds net income"),
    ("leverage_falling", "Leverage", "Long-term debt to assets fell"),
    ("liquidity_rising", "Leverage", "Current ratio improved"),
    ("no_dilution", "Leverage", "No new shares issued"),
    ("margin_rising", "Efficiency", "Gross margin improved"),
    ("turnover_rising", "Efficiency", "Asset turnover improved"),
)

#: Beneish's eight variables, in the order of the 1999 paper.
BENEISH_VARIABLES: tuple[str, ...] = ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA")

#: Added to Altman's Z''-Score for emerging markets, per Altman (2005). It
#: shifts the scale so the thresholds published for US non-manufacturers apply
#: to an emerging-market issuer.
ALTMAN_EM_CONSTANT = 3.25

#: Countries whose issuers get Altman's emerging-market adjustment. Only the
#: markets this application actually covers; anything not listed is treated as
#: a developed market, which is the conservative direction because it does NOT
#: add 3.25 to the score.
EMERGING_MARKETS: frozenset[str] = frozenset({"india", "in"})

#: Horizons the UI offers, in months. The long ones are listed deliberately:
#: :func:`horizon_availability` reports each as unavailable *with the
#: arithmetic*, which is more useful than omitting them and leaving the reader
#: to wonder whether it was an oversight.
RETURN_HORIZONS: tuple[tuple[str, int], ...] = (
    ("1 month", 1),
    ("3 months", 3),
    ("1 year", 12),
    ("3 years", 36),
    ("5 years", 60),
    ("7 years", 84),
    ("10 years", 120),
    ("11 years", 132),
)

#: Fewest annual periods before a trend statement means anything.
MIN_PERIODS_FOR_TREND = 3

_NAN = float("nan")


def _nan() -> float:
    return _NAN


def _num(value: object) -> float:
    """Finite float, or NaN. The only way a missing input enters a formula."""
    if value is None or isinstance(value, bool):
        return _NAN
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _ok(*values: float) -> bool:
    """True when every value is a real number. Guards every formula."""
    return all(not math.isnan(v) for v in values)


def _div(numerator: float, denominator: float) -> float:
    """Ratio, or NaN. Never raises, and never returns infinity.

    A zero denominator means the ratio is undefined, not enormous. Returning
    ``inf`` would propagate into a score and render as a spuriously perfect
    result -- the interest-cover of a company with no reported interest being
    the case that matters.
    """
    if not _ok(numerator, denominator) or denominator == 0.0:
        return _NAN
    out = numerator / denominator
    return out if math.isfinite(out) else _NAN


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------
def _label_key(label: object) -> str:
    """A row label reduced to lowercase alphanumerics, for tolerant matching.

    The provider has renamed its statement row labels at least once:
    yfinance 0.2.x returned ``"Total Revenue"`` and 1.x returns
    ``"TotalRevenue"``. Both spell the same line, and a strict string compare
    silently yields NaN for every multi-word item on the newer format --
    revenue, operating income, tax and interest expense all became NaN with no
    error anywhere. Folding case and punctuation away makes :data:`LINE_ITEMS`
    survive this rename and the next one, instead of needing a second spelling
    enumerated for every field.
    """
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def line_item(frame: object, field_name: str, column: object) -> float:
    """One canonical field from one period of a statement frame, or NaN.

    Looks up every alternate label in :data:`LINE_ITEMS` and takes the first
    that is present *and* numeric, so a row that exists but is blank for this
    period falls through to the alternate rather than poisoning the result.

    An exact match is tried first, then a punctuation- and case-insensitive
    pass, which is what keeps this working across provider label renames. The
    tolerant pass is only reached on a miss, so the common path stays a plain
    dict lookup.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _NAN
    if column not in frame.columns:
        return _NAN

    candidates = LINE_ITEMS.get(field_name, (field_name,))
    for label in candidates:
        if label in frame.index:
            value = _num(frame.at[label, column])
            if not math.isnan(value):
                return value

    wanted = {_label_key(label) for label in candidates}
    for row in frame.index:
        if _label_key(row) in wanted:
            value = _num(frame.at[row, column])
            if not math.isnan(value):
                return value
    return _NAN


@dataclass(frozen=True)
class FiscalPeriod:
    """One fiscal year's figures, keyed by canonical field name.

    Frozen and flat on purpose: a period is the unit every score consumes, and
    a mapping of known field names is far easier to build a fixture for than
    three provider-shaped frames.
    """

    period_end: date | None
    values: Mapping[str, float] = field(default_factory=dict)

    def get(self, field_name: str) -> float:
        return _num(self.values.get(field_name))

    @property
    def label(self) -> str:
        return self.period_end.isoformat() if self.period_end else "unknown"

    @property
    def n_available(self) -> int:
        return sum(1 for v in self.values.values() if not math.isnan(_num(v)))


@dataclass(frozen=True)
class Statements:
    """A company's annual statements, newest first.

    Newest first because that is the order the provider returns and the order
    a reader wants; every score that needs a prior year takes ``periods[i+1]``
    and says so.
    """

    ticker: str = ""
    currency: str = ""
    periods: tuple[FiscalPeriod, ...] = ()
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return bool(self.periods)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def latest(self) -> FiscalPeriod | None:
        return self.periods[0] if self.periods else None

    @property
    def span(self) -> str:
        if not self.periods:
            return "no periods"
        newest, oldest = self.periods[0].label, self.periods[-1].label
        return newest if newest == oldest else f"{oldest} to {newest}"

    def series(self, field_name: str) -> pd.Series:
        """One field across every period, oldest first, indexed by period end.

        Oldest first here, unlike ``periods``: a series is for plotting and
        for ``pct_change``, both of which want time running forwards.
        """
        ordered = list(reversed(self.periods))
        index = [p.period_end for p in ordered]
        return pd.Series([p.get(field_name) for p in ordered], index=index, dtype="float64")


def read_statements(
    balance_sheet: object = None,
    income: object = None,
    cash_flow: object = None,
    ticker: str = "",
    currency: str = "",
) -> Statements:
    """Normalise three provider frames into :class:`Statements`.

    The provider's frames are ``line label x period`` with inconsistent
    labels between markets; this collapses them onto :data:`LINE_ITEMS` so
    every score downstream reads one vocabulary. Nothing is computed here and
    nothing is filled in -- a field the provider omitted stays NaN, and the
    scores decide what that means.

    Periods are the union of the three frames' columns, so a company whose
    cash-flow statement runs one year shorter than its balance sheet still
    yields every period either one can support.
    """
    frames = {
        "balance_sheet": balance_sheet,
        "income": income,
        "cash_flow": cash_flow,
    }
    notices: list[str] = []

    columns: list[object] = []
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            notices.append(f"No {name.replace('_', ' ')} was available.")
            continue
        for column in frame.columns:
            if column not in columns:
                columns.append(column)

    if not columns:
        return Statements(
            ticker=str(ticker or ""),
            currency=str(currency or ""),
            notices=tuple(notices) or ("No financial statements were available.",),
        )

    # Newest first. Sorted descending on the raw column values, which are
    # timestamps; a column that will not compare falls to the end rather than
    # raising.
    def sort_key(column: object) -> tuple[int, str]:
        stamp = pd.Timestamp(column) if not isinstance(column, pd.Timestamp) else column
        try:
            return (0, stamp.isoformat())
        except (AttributeError, ValueError):  # pragma: no cover - exotic column
            return (1, str(column))

    with contextlib.suppress(TypeError):  # unorderable columns
        columns.sort(key=sort_key, reverse=True)

    periods: list[FiscalPeriod] = []
    for column in columns:
        values: dict[str, float] = {}
        for field_name in LINE_ITEMS:
            for frame in frames.values():
                value = line_item(frame, field_name, column)
                if not math.isnan(value):
                    values[field_name] = value
                    break
            else:
                values[field_name] = _NAN

        try:
            ends = pd.Timestamp(column).date()
        except (TypeError, ValueError):  # pragma: no cover - exotic column
            ends = None
        periods.append(FiscalPeriod(period_end=ends, values=values))

    return Statements(
        ticker=str(ticker or ""),
        currency=str(currency or ""),
        periods=tuple(periods),
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    """One binary test with its inputs, or ``None`` when it could not run."""

    name: str
    group: str
    description: str
    passed: bool | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.passed is not None


@dataclass(frozen=True)
class ScoreResult:
    """A score, what it was computed from, and what it could not see.

    ``value`` is ``None`` when too little was available for the number to
    mean anything. The alternative -- returning a low score built from
    missing inputs -- is the mistake this whole module is arranged to avoid.
    """

    name: str
    value: float | None = None
    scale: str = ""
    verdict: str = ""
    available: int = 0
    total: int = 0
    signals: tuple[Signal, ...] = ()
    inputs: Mapping[str, float] = field(default_factory=dict)
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def coverage(self) -> str:
        if not self.total:
            return "no inputs"
        if self.available == self.total:
            return f"all {self.total} inputs available"
        return f"{self.available} of {self.total} inputs available"

    @property
    def headline(self) -> str:
        if self.value is None:
            return f"{self.name}: not available — {self.coverage}"
        shown = f"{self.value:.0f}" if float(self.value).is_integer() else f"{self.value:.2f}"
        scale = f" / {self.scale}" if self.scale else ""
        verdict = f" — {self.verdict}" if self.verdict else ""
        return f"{self.name} {shown}{scale}{verdict}"


# ---------------------------------------------------------------------------
# Piotroski F-Score (Piotroski 2000)
# ---------------------------------------------------------------------------
def piotroski(statements: Statements, index: int = 0) -> ScoreResult:
    """Piotroski's nine-signal F-Score for the period at ``index``.

    Piotroski, J. (2000), *Value Investing: The Use of Historical Financial
    Statement Information to Separate Winners from Losers*. Nine binary tests
    across profitability, leverage/liquidity and operating efficiency; a score
    of 8-9 is strong, 0-2 weak.

    **The score is out of however many signals could be computed**, and each
    one reports whether it ran. Scoring a missing signal as a failure is the
    common implementation bug: a debt-free company that reports no long-term
    debt figure would lose the leverage point and look worse than a leveraged
    one. Here it loses nothing and the coverage says so.

    Needs the prior year for five of the nine signals, so a company with one
    period of statements yields four at most.
    """
    if not statements.ok:
        return ScoreResult(
            name="Piotroski F-Score",
            scale="9",
            total=9,
            notices=("No financial statements were available.",),
        )
    if index >= statements.n_periods:
        return ScoreResult(
            name="Piotroski F-Score",
            scale="9",
            total=9,
            notices=(f"Period {index} is outside the {statements.n_periods} available.",),
        )

    now = statements.periods[index]
    prior = statements.periods[index + 1] if index + 1 < statements.n_periods else None
    notices: list[str] = []
    if prior is None:
        notices.append("No prior year, so the five year-on-year signals could not be computed.")

    def signal(name: str, passed: bool | None, detail: str = "") -> Signal:
        key, group, description = next(s for s in PIOTROSKI_SIGNALS if s[0] == name)
        return Signal(name=key, group=group, description=description, passed=passed, detail=detail)

    assets_now = now.get("total_assets")
    assets_prior = prior.get("total_assets") if prior else _NAN
    # Piotroski scales by BEGINNING-of-year assets, i.e. the prior balance
    # sheet. Using this year's is a common shortcut that shifts every ROA.
    roa_now = _div(now.get("net_income"), assets_prior if prior else _NAN)
    roa_prior = (
        _div(prior.get("net_income"), statements.periods[index + 2].get("total_assets"))
        if prior is not None and index + 2 < statements.n_periods
        else _NAN
    )
    cfo_now = now.get("operating_cash_flow")

    signals: list[Signal] = [
        signal(
            "roa_positive",
            None if math.isnan(roa_now) else roa_now > 0,
            "" if math.isnan(roa_now) else f"ROA {roa_now * 100:.1f}%",
        ),
        signal(
            "cfo_positive",
            None if math.isnan(cfo_now) else cfo_now > 0,
            "" if math.isnan(cfo_now) else f"CFO {cfo_now:,.0f}",
        ),
        signal(
            "roa_improving",
            None if not _ok(roa_now, roa_prior) else roa_now > roa_prior,
            ""
            if not _ok(roa_now, roa_prior)
            else f"{roa_prior * 100:.1f}% to {roa_now * 100:.1f}%",
        ),
        signal(
            "accruals",
            None
            if not _ok(cfo_now, now.get("net_income"), assets_prior)
            else _div(cfo_now, assets_prior) > roa_now,
            "cash earnings versus reported earnings",
        ),
    ]

    # -- leverage and liquidity ------------------------------------------
    ltd_now = _div(now.get("long_term_debt"), assets_now)
    ltd_prior = _div(prior.get("long_term_debt"), assets_prior) if prior else _NAN
    current_now = _div(now.get("current_assets"), now.get("current_liabilities"))
    current_prior = (
        _div(prior.get("current_assets"), prior.get("current_liabilities")) if prior else _NAN
    )
    shares_now = now.get("shares")
    shares_prior = prior.get("shares") if prior else _NAN

    signals += [
        signal(
            "leverage_falling",
            None if not _ok(ltd_now, ltd_prior) else ltd_now <= ltd_prior,
            "" if not _ok(ltd_now, ltd_prior) else f"{ltd_prior:.3f} to {ltd_now:.3f}",
        ),
        signal(
            "liquidity_rising",
            None if not _ok(current_now, current_prior) else current_now > current_prior,
            ""
            if not _ok(current_now, current_prior)
            else f"{current_prior:.2f} to {current_now:.2f}",
        ),
        signal(
            "no_dilution",
            None if not _ok(shares_now, shares_prior) else shares_now <= shares_prior,
            ""
            if not _ok(shares_now, shares_prior)
            else f"{shares_prior:,.0f} to {shares_now:,.0f} shares",
        ),
    ]

    # -- operating efficiency --------------------------------------------
    margin_now = _div(now.get("gross_profit"), now.get("revenue"))
    margin_prior = _div(prior.get("gross_profit"), prior.get("revenue")) if prior else _NAN
    turnover_now = _div(now.get("revenue"), assets_prior if prior else _NAN)
    turnover_prior = (
        _div(prior.get("revenue"), statements.periods[index + 2].get("total_assets"))
        if prior is not None and index + 2 < statements.n_periods
        else _NAN
    )

    signals += [
        signal(
            "margin_rising",
            None if not _ok(margin_now, margin_prior) else margin_now > margin_prior,
            ""
            if not _ok(margin_now, margin_prior)
            else f"{margin_prior * 100:.1f}% to {margin_now * 100:.1f}%",
        ),
        signal(
            "turnover_rising",
            None if not _ok(turnover_now, turnover_prior) else turnover_now > turnover_prior,
            ""
            if not _ok(turnover_now, turnover_prior)
            else f"{turnover_prior:.2f} to {turnover_now:.2f}",
        ),
    ]

    available = [s for s in signals if s.available]
    passed = sum(1 for s in available if s.passed)

    if not available:
        return ScoreResult(
            name="Piotroski F-Score",
            scale="9",
            total=9,
            signals=tuple(signals),
            notices=(*notices, "None of the nine signals could be computed."),
        )

    verdict = ""
    if len(available) == 9:
        verdict = "strong" if passed >= 8 else ("weak" if passed <= 2 else "middling")
    else:
        notices.append(
            f"Scored out of {len(available)}, not 9: a signal that could not be "
            "computed is not counted as a failure."
        )

    return ScoreResult(
        name="Piotroski F-Score",
        value=float(passed),
        scale=str(len(available)),
        verdict=verdict,
        available=len(available),
        total=9,
        signals=tuple(signals),
        inputs={"roa": roa_now, "cfo": cfo_now, "current_ratio": current_now},
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# Altman Z-Score (Altman 1968; Z''-EM per Altman 2005)
# ---------------------------------------------------------------------------
def altman_z(
    statements: Statements,
    market_cap: float | None = None,
    index: int = 0,
    country: str = "",
    emerging_market: bool | None = None,
) -> ScoreResult:
    """Altman's Z''-Score for distress risk.

    Two variants exist and they are not interchangeable:

    * The original 1968 **Z** is calibrated on listed *manufacturers* and
      includes an asset-turnover term (X5) that makes a software company look
      distressed and a retailer look safe purely on its business model.
    * **Z''** drops X5 and re-weights the rest for non-manufacturers; adding
      :data:`ALTMAN_EM_CONSTANT` (Altman 2005) shifts it onto the published
      thresholds for an emerging-market issuer.

    ``Z''`` is the default here because this application is India-first and
    covers every sector, so the variant that is *not* sector-dependent is the
    defensible one. Thresholds: above 5.85 safe, 4.15-5.85 grey, below 4.15
    distress. The variant actually used is named in ``verdict``, because a
    Z-Score quoted without its variant is unreadable.

    The variant is chosen from ``country`` rather than defaulted, because the
    EM adjustment adds 3.25 and applying it to a US issuer inflates the score
    against thresholds never calibrated for it. ``emerging_market`` overrides
    the lookup when the caller knows better; with neither, a developed-market
    variant is assumed, which is the conservative direction.

    ``market_cap`` is required for X4 (market value of equity to total
    liabilities). Without it the score is reported as unavailable rather than
    substituting book equity, which would measure something different and
    quietly relabel it.
    """
    if not statements.ok or index >= statements.n_periods:
        return ScoreResult(
            name="Altman Z-Score", total=4, notices=("No financial statements were available.",)
        )

    now = statements.periods[index]
    assets = now.get("total_assets")
    liabilities = now.get("total_liabilities")

    working_capital = now.get("working_capital")
    if math.isnan(working_capital):
        working_capital = now.get("current_assets") - now.get("current_liabilities")
        working_capital = _num(working_capital)

    x1 = _div(working_capital, assets)
    x2 = _div(now.get("retained_earnings"), assets)
    x3 = _div(now.get("ebit"), assets)
    x4 = _div(_num(market_cap), liabilities)

    terms = {
        "X1 working capital/assets": x1,
        "X2 retained earnings/assets": x2,
        "X3 EBIT/assets": x3,
        "X4 market equity/liabilities": x4,
    }
    available = sum(1 for v in terms.values() if not math.isnan(v))
    notices: list[str] = []
    if math.isnan(x4):
        notices.append(
            "X4 needs a market capitalisation and total liabilities; book equity "
            "is NOT substituted, because that measures something else."
        )

    if available < 4:
        return ScoreResult(
            name="Altman Z-Score",
            available=available,
            total=4,
            inputs=terms,
            notices=(
                *notices,
                "Every term is required: a Z-Score missing a term is not a lower "
                "Z-Score, it is no Z-Score.",
            ),
        )

    if emerging_market is None:
        emerging_market = str(country or "").strip().lower() in EMERGING_MARKETS

    if emerging_market:
        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4 + ALTMAN_EM_CONSTANT
        variant = "Z''-EM (non-manufacturing, emerging market)"
        safe, distress = 5.85, 4.15
    else:
        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        variant = "Z'' (non-manufacturing)"
        safe, distress = 2.60, 1.10

    zone = "safe zone" if z > safe else ("distress zone" if z < distress else "grey zone")

    return ScoreResult(
        name="Altman Z-Score",
        value=float(z),
        verdict=f"{zone} — {variant}",
        available=4,
        total=4,
        inputs=terms,
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# Beneish M-Score (Beneish 1999)
# ---------------------------------------------------------------------------
def beneish_m(statements: Statements, index: int = 0) -> ScoreResult:
    """Beneish's eight-variable M-Score for earnings-manipulation risk.

    Beneish, M. (1999), *The Detection of Earnings Manipulation*. Above
    **-1.78** the model classifies a firm as a likely manipulator. That is a
    statement about the *probability the accounts have been managed*, not an
    accusation and not a prediction of returns — the paper's own false-positive
    rate is substantial, and a high score on a company with a large genuine
    acquisition is expected rather than suspicious.

    Needs two consecutive years for every variable, so it is unavailable for
    the oldest period and for a company with one year of statements. All eight
    variables are required: the published coefficients are only meaningful
    together, so a partial M-Score is reported as unavailable rather than
    summed over the terms that happened to be computable.
    """
    if not statements.ok or index + 1 >= statements.n_periods:
        return ScoreResult(
            name="Beneish M-Score",
            total=8,
            notices=("Two consecutive fiscal years are required.",),
        )

    now = statements.periods[index]
    prior = statements.periods[index + 1]

    def both(field_name: str) -> tuple[float, float]:
        return now.get(field_name), prior.get(field_name)

    revenue_now, revenue_prior = both("revenue")
    assets_now, assets_prior = both("total_assets")
    receivables_now, receivables_prior = both("receivables")
    ppe_now, ppe_prior = both("net_ppe")
    securities_now, securities_prior = both("short_term_investments")
    depreciation_now, depreciation_prior = both("depreciation")
    sga_now, sga_prior = both("sga")
    debt_now, debt_prior = both("total_debt")
    gross_now, gross_prior = both("gross_profit")

    # Days sales in receivables index.
    dsri = _div(_div(receivables_now, revenue_now), _div(receivables_prior, revenue_prior))
    # Gross margin index -- PRIOR over CURRENT, so a deteriorating margin
    # raises the score. The order is a classic place to get this backwards.
    gmi = _div(_div(gross_prior, revenue_prior), _div(gross_now, revenue_now))
    # Asset quality index: non-current, non-PPE assets as a share of total.
    aqi_now = _div(assets_now - now.get("current_assets") - ppe_now - securities_now, assets_now)
    aqi_prior = _div(
        assets_prior - prior.get("current_assets") - ppe_prior - securities_prior, assets_prior
    )
    aqi = _div(aqi_now, aqi_prior)
    # Sales growth index.
    sgi = _div(revenue_now, revenue_prior)
    # Depreciation index: PRIOR rate over CURRENT, so a slowing rate (which
    # flatters earnings) raises the score.
    depi = _div(
        _div(depreciation_prior, depreciation_prior + ppe_prior),
        _div(depreciation_now, depreciation_now + ppe_now),
    )
    # SG&A index.
    sgai = _div(_div(sga_now, revenue_now), _div(sga_prior, revenue_prior))
    # Leverage index.
    lvgi = _div(
        _div(debt_now + now.get("current_liabilities"), assets_now),
        _div(debt_prior + prior.get("current_liabilities"), assets_prior),
    )
    # Total accruals to total assets. Beneish uses income before
    # extraordinaries less cash from operations, scaled by assets.
    tata = _div(now.get("net_income") - now.get("operating_cash_flow"), assets_now)

    terms = dict(zip(BENEISH_VARIABLES, (dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata), strict=True))
    available = sum(1 for v in terms.values() if not math.isnan(v))

    if available < len(BENEISH_VARIABLES):
        absent = [k for k, v in terms.items() if math.isnan(v)]
        return ScoreResult(
            name="Beneish M-Score",
            available=available,
            total=len(BENEISH_VARIABLES),
            inputs=terms,
            notices=(
                f"Missing {', '.join(absent)}. The published coefficients are only "
                "meaningful together, so no partial score is reported.",
            ),
        )

    m = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    flagged = m > -1.78

    return ScoreResult(
        name="Beneish M-Score",
        value=float(m),
        verdict=(
            "above the -1.78 threshold — accounts warrant a closer read"
            if flagged
            else "below the -1.78 threshold"
        ),
        available=available,
        total=len(BENEISH_VARIABLES),
        inputs=terms,
        notices=(
            "A screen for aggressive accounting, not an accusation: the model's "
            "false-positive rate is substantial, and a large genuine acquisition "
            "raises it too.",
        ),
    )


# ---------------------------------------------------------------------------
# accruals and ratios
# ---------------------------------------------------------------------------
def accruals_ratio(statements: Statements, index: int = 0) -> ScoreResult:
    """Sloan's accruals ratio: how much of earnings is not cash.

    Sloan, R. (1996). ``(net income - operating cash flow) / total assets``.
    High positive accruals mean reported profit is running ahead of the cash
    collected, which historically preceded weaker subsequent earnings. Stated
    as a measurement, not a forecast.
    """
    if not statements.ok or index >= statements.n_periods:
        return ScoreResult(name="Accruals ratio", total=3, notices=("No statements available.",))

    now = statements.periods[index]
    income = now.get("net_income")
    cfo = now.get("operating_cash_flow")
    assets = now.get("total_assets")
    ratio = _div(income - cfo, assets) if _ok(income, cfo, assets) else _NAN

    if math.isnan(ratio):
        return ScoreResult(
            name="Accruals ratio",
            available=sum(1 for v in (income, cfo, assets) if not math.isnan(v)),
            total=3,
            notices=("Needs net income, operating cash flow and total assets.",),
        )

    return ScoreResult(
        name="Accruals ratio",
        value=float(ratio * 100.0),
        scale="%",
        verdict=(
            "earnings running ahead of cash"
            if ratio > 0.10
            else ("cash ahead of earnings" if ratio < -0.10 else "earnings broadly cash-backed")
        ),
        available=3,
        total=3,
        inputs={"net_income": income, "operating_cash_flow": cfo, "total_assets": assets},
    )


@dataclass(frozen=True)
class RatioSet:
    """Plain ratios for one period. NaN wherever an input was missing."""

    period: str = ""
    values: Mapping[str, float] = field(default_factory=dict)
    notices: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str) -> float:
        return _num(self.values.get(name))

    @property
    def available(self) -> int:
        """How many ratios actually have a number behind them."""
        return sum(1 for value in self.values.values() if not math.isnan(_num(value)))


def ratios(statements: Statements, index: int = 0) -> RatioSet:
    """Profitability, leverage and efficiency ratios for one period.

    Every ratio goes through :func:`_div`, so a zero or missing denominator
    yields NaN rather than infinity. Interest cover is the case that matters:
    a company reporting no interest expense has an *undefined* cover, not an
    infinite one, and rendering it as infinite would flatter a company whose
    filing is simply incomplete.
    """
    if not statements.ok or index >= statements.n_periods:
        return RatioSet(notices=("No statements available.",))

    now = statements.periods[index]
    prior = statements.periods[index + 1] if index + 1 < statements.n_periods else None
    average_assets = (
        (now.get("total_assets") + prior.get("total_assets")) / 2.0
        if prior is not None and _ok(now.get("total_assets"), prior.get("total_assets"))
        else now.get("total_assets")
    )
    average_equity = (
        (now.get("equity") + prior.get("equity")) / 2.0
        if prior is not None and _ok(now.get("equity"), prior.get("equity"))
        else now.get("equity")
    )

    capex = now.get("capex")
    # Capex arrives NEGATIVE (a cash outflow). Made explicit here because a
    # sign error silently inverts free cash flow.
    capex_outflow = abs(capex) if not math.isnan(capex) else _NAN

    ebit = now.get("ebit")
    tax_rate = _div(now.get("tax"), now.get("pretax_income"))
    nopat = ebit * (1.0 - tax_rate) if _ok(ebit, tax_rate) else _NAN

    values = {
        "gross_margin": _div(now.get("gross_profit"), now.get("revenue")),
        "operating_margin": _div(now.get("operating_income"), now.get("revenue")),
        "net_margin": _div(now.get("net_income"), now.get("revenue")),
        "roa": _div(now.get("net_income"), average_assets),
        "roe": _div(now.get("net_income"), average_equity),
        "roic": _div(_num(nopat), now.get("invested_capital")),
        "asset_turnover": _div(now.get("revenue"), average_assets),
        "current_ratio": _div(now.get("current_assets"), now.get("current_liabilities")),
        "debt_to_equity": _div(now.get("total_debt"), now.get("equity")),
        "debt_to_assets": _div(now.get("total_debt"), now.get("total_assets")),
        "interest_cover": _div(ebit, now.get("interest_expense")),
        "cash_conversion": _div(now.get("operating_cash_flow"), now.get("net_income")),
        "capex_to_revenue": _div(_num(capex_outflow), now.get("revenue")),
        "fcf_margin": _div(now.get("free_cash_flow"), now.get("revenue")),
        "effective_tax_rate": tax_rate,
    }

    notices: list[str] = []
    if math.isnan(values["interest_cover"]):
        notices.append(
            "Interest cover is unavailable: the filing reports no interest expense, "
            "which is not the same as infinite cover."
        )
    if prior is None:
        notices.append("ROA, ROE and asset turnover use closing rather than average balances.")

    return RatioSet(period=now.label, values=values, notices=tuple(notices))


@dataclass(frozen=True)
class HorizonCheck:
    """Whether a return model could be *trained* for one horizon, and why not.

    Typed rather than a dict, for the same reason as
    :class:`QualityReport`: the UI joins ``horizon`` strings into a sentence,
    and a ``dict[str, object]`` makes that a type error the checker is right
    to raise.
    """

    horizon: str
    months: int
    pairs: int
    available: bool
    reason: str


def horizon_availability(
    statements: Statements, horizons: Sequence[tuple[str, int]] = RETURN_HORIZONS
) -> list[HorizonCheck]:
    """Per horizon: could a return model be *trained* on the data available?

    This is the honest answer to "predict the return over 1 month, 1 year,
    3 years, 5, 7, 10, 11 years and so on". A horizon is not a free parameter.
    Training a model that maps fundamentals at time T to the return from T to
    T+N needs pairs of exactly that shape, and each pair consumes N of history:

        usable pairs  =  (annual periods available) - (horizon in years)

    With five annual statements that yields four one-year pairs, two
    three-year pairs, and **zero** from five years out. A number produced for a
    ten-year horizon would not be a weak prediction; it would be a prediction
    fitted to no observations at all.

    Two caveats apply even where the arithmetic works, and travel with the
    result:

    * the statements are restated, so a "2022" input silently contains
      information published in 2024 -- look-ahead bias,
    * delisted companies are absent from the provider entirely, so the panel
      contains only survivors.

    Returns one record per horizon with ``available``, ``pairs`` and a
    ``reason``, so the UI can list every horizon the user asked for and say
    plainly which ones it will not answer.
    """
    periods = statements.n_periods
    out: list[HorizonCheck] = []

    for label, months in horizons:
        # Ceiling, because a horizon that straddles a fiscal year still needs
        # that whole year of forward history to label an observation.
        needed = max(1, math.ceil(months / 12.0))
        pairs = max(periods - needed, 0) if periods else 0

        if not periods:
            reason = "No financial statements are available for this instrument."
        elif pairs <= 0:
            reason = (
                f"Needs {needed} year(s) of history beyond each observation and only "
                f"{periods} annual period(s) exist, which leaves no training pairs at "
                "all. A number here would be fitted to zero observations."
            )
        elif pairs < 3:
            reason = (
                f"Only {pairs} training pair(s) from {periods} annual period(s) - too "
                "few for a fit that means anything out of sample."
            )
        else:
            reason = (
                f"{pairs} training pair(s) per instrument from {periods} annual "
                "period(s). Usable cross-sectionally, but the statements are restated "
                "(look-ahead bias) and delisted companies are absent (survivorship "
                "bias), so treat any fit as explanatory rather than predictive."
            )

        out.append(
            HorizonCheck(
                horizon=label,
                months=months,
                pairs=pairs,
                available=pairs >= 3,
                reason=reason,
            )
        )
    return out


@dataclass(frozen=True)
class QualityReport:
    """Every score for one period, as a typed object rather than a dict.

    A ``dict[str, object]`` was tried first and cost every caller its types:
    the UI had to re-assert that ``summary["piotroski"]`` is a
    :class:`ScoreResult` before touching it, which the type checker rightly
    refused to take on trust.
    """

    ticker: str = ""
    currency: str = ""
    span: str = ""
    n_periods: int = 0
    piotroski: ScoreResult = field(default_factory=lambda: ScoreResult(name="Piotroski F-Score"))
    altman: ScoreResult = field(default_factory=lambda: ScoreResult(name="Altman Z-Score"))
    beneish: ScoreResult = field(default_factory=lambda: ScoreResult(name="Beneish M-Score"))
    accruals: ScoreResult = field(default_factory=lambda: ScoreResult(name="Accruals ratio"))
    ratios: RatioSet = field(default_factory=RatioSet)
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scores(self) -> tuple[ScoreResult, ...]:
        return (self.piotroski, self.altman, self.beneish, self.accruals)

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Names of the scores that could not be computed, for the UI to list."""
        return tuple(score.name for score in self.scores if not score.ok)


def quality_summary(
    statements: Statements, market_cap: float | None = None, country: str = ""
) -> QualityReport:
    """Every score for the latest period, plus what could not be computed.

    One call for the UI, so a page does not have to know which scores need a
    prior year and which need a market capitalisation.
    """
    f_score = piotroski(statements)
    z_score = altman_z(statements, market_cap, country=country)
    m_score = beneish_m(statements)
    accruals = accruals_ratio(statements)
    ratio_set = ratios(statements)

    return QualityReport(
        ticker=statements.ticker,
        currency=statements.currency,
        span=statements.span,
        n_periods=statements.n_periods,
        piotroski=f_score,
        altman=z_score,
        beneish=m_score,
        accruals=accruals,
        ratios=ratio_set,
        notices=tuple(statements.notices),
    )


# ---------------------------------------------------------------------------
# the balance sheet, as a statement
# ---------------------------------------------------------------------------
#: The balance sheet in reading order: ``(section, label, field, level)``.
#:
#: ``level`` is the indent depth, and it encodes subordination rather than
#: decoration -- a level-1 row is a component of the level-0 subtotal or total
#: it sits against. ``field`` names the canonical key in :data:`LINE_ITEMS`.
#:
#: Ordered the way a balance sheet is read: liquid assets first, then the
#: current subtotal, then the long-lived assets, then the total; liabilities
#: and equity after. The provider supplies no finer breakdown than this, and
#: inventing intermediate rows the filing does not carry would be presenting a
#: guess as a statement.
BALANCE_SHEET_LAYOUT: tuple[tuple[str, str, str, int], ...] = (
    ("Assets", "Cash and cash equivalents", "cash", 1),
    ("Assets", "Short-term investments", "short_term_investments", 1),
    ("Assets", "Accounts receivable", "receivables", 1),
    ("Assets", "Inventory", "inventory", 1),
    ("Assets", "Total current assets", "current_assets", 0),
    ("Assets", "Net property, plant & equipment", "net_ppe", 1),
    ("Assets", "Gross property, plant & equipment", "gross_ppe", 1),
    ("Assets", "TOTAL ASSETS", "total_assets", 0),
    ("Liabilities", "Total current liabilities", "current_liabilities", 0),
    ("Liabilities", "Long-term debt", "long_term_debt", 1),
    ("Liabilities", "Total debt", "total_debt", 1),
    ("Liabilities", "TOTAL LIABILITIES", "total_liabilities", 0),
    ("Equity", "Retained earnings", "retained_earnings", 1),
    ("Equity", "TOTAL EQUITY", "equity", 0),
    ("Memo", "Working capital", "working_capital", 1),
    ("Memo", "Invested capital", "invested_capital", 1),
    ("Memo", "Shares outstanding", "shares", 1),
)

#: Balance-sheet fields that are a COUNT, not an amount of money. Formatting
#: these with a currency symbol produces "₹1,353 Cr shares", which is a
#: category error a reader has to mentally undo on every glance.
BALANCE_SHEET_COUNT_FIELDS: frozenset[str] = frozenset({"shares"})

#: Rows the layout marks as a subtotal or total, for the UI to emphasise.
BALANCE_SHEET_TOTALS: frozenset[str] = frozenset(
    label for _section, label, _field, level in BALANCE_SHEET_LAYOUT if level == 0
)

#: A balance sheet balances exactly in the filing. It frequently does NOT
#: balance in the provider's extract, because "total liabilities net minority
#: interest" and "stockholders equity" both exclude the minority interest that
#: sits between them, so it falls out of the middle of the identity. A residual
#: inside this fraction of total assets is that known gap; beyond it, something
#: is genuinely wrong with the extract and the scores built on it.
BALANCE_SHEET_TOLERANCE: float = 0.01


def balance_sheet_statement(statements: Statements, max_periods: int = 5) -> pd.DataFrame:
    """The balance sheet laid out as a statement, newest period first.

    Columns::

        section, line, field, level, is_total, source, <period label> x n

    One column per fiscal period, labelled by period end and ordered newest
    first, because that is the direction a reader compares from. Values are
    raw floats in the company's reporting currency: formatting belongs to the
    caller, which knows the locale and the compaction the rest of its page uses.

    A line the provider did not supply is **NaN in every period, and the row is
    kept**. Dropping it would silently shorten the statement and leave the
    reader unable to tell "this company reports no inventory" from "we could
    not find the inventory row". The UI renders NaN as an em dash, and the
    row's presence is the disclosure.

    ``source`` is ``"reported"`` for a figure read from the filing and
    ``"derived"`` for one computed here, so the two can never be confused.
    Only ``working_capital`` is ever derived, only when the provider omitted it
    while supplying both components, and the flag says so on that row.

    Returns an empty frame with the metadata columns when there are no periods.
    """
    columns = ["section", "line", "field", "level", "is_total", "source"]
    if not isinstance(statements, Statements) or not statements.periods:
        return pd.DataFrame(columns=columns)

    periods = list(statements.periods[: max(int(max_periods), 1)])
    labels = [p.label for p in periods]

    rows: list[dict[str, Any]] = []
    for section, label, field_name, level in BALANCE_SHEET_LAYOUT:
        values: list[float] = []
        derived = False
        for period in periods:
            value = period.get(field_name)
            if field_name == "working_capital" and math.isnan(value):
                current = period.get("current_assets")
                liabilities = period.get("current_liabilities")
                if not math.isnan(current) and not math.isnan(liabilities):
                    value = current - liabilities
                    derived = True
            values.append(value)

        row: dict[str, Any] = {
            "section": section,
            "line": label,
            "field": field_name,
            "level": int(level),
            "is_total": label in BALANCE_SHEET_TOTALS,
            "source": "derived" if derived else "reported",
        }
        row.update(dict(zip(labels, values)))
        rows.append(row)

    return pd.DataFrame(rows, columns=columns + labels)


def balance_sheet_check(statements: Statements, max_periods: int = 5) -> dict[str, Any]:
    """Does the extract satisfy ``assets = liabilities + equity``, per period?

    The identity is the one property a balance sheet cannot fail, which makes
    it the cheapest possible test of whether the provider's extract is
    trustworthy enough to read. It is checked and REPORTED rather than quietly
    assumed, because a statement that does not balance is a statement whose
    quality scores should not be believed either.

    Returns ``{ok, diagnosis, reason, rows, n_checked, n_balanced,
    worst_residual_pct}``, where ``diagnosis`` is one of ``"balanced"``,
    ``"minority_interest"`` (a stable positive gap, which is structural and
    expected in a holding company) or ``"inconsistent"`` (a gap that changes
    sign or comes and goes, which nothing structural explains).
    ``rows`` is a frame of ``period, total_assets, total_liabilities, equity,
    residual, residual_pct, balanced`` where::

        residual     = total_assets - (total_liabilities + equity)
        residual_pct = residual / total_assets * 100

    A period missing any of the three figures is skipped rather than scored,
    and ``n_checked`` reports how many could be tested at all.

    A small positive residual is usually **minority interest**, excluded from
    both of the provider's figures; that is why the tolerance is
    :data:`BALANCE_SHEET_TOLERANCE` of total assets rather than zero.
    """
    blank: dict[str, Any] = {
        "ok": False,
        "diagnosis": "unavailable",
        "reason": "No statements were available to check.",
        "rows": pd.DataFrame(),
        "n_checked": 0,
        "n_balanced": 0,
        "worst_residual_pct": _NAN,
    }
    if not isinstance(statements, Statements) or not statements.periods:
        return blank

    records: list[dict[str, Any]] = []
    for period in statements.periods[: max(int(max_periods), 1)]:
        assets = period.get("total_assets")
        liabilities = period.get("total_liabilities")
        equity = period.get("equity")
        if math.isnan(assets) or math.isnan(liabilities) or math.isnan(equity):
            continue
        residual = assets - (liabilities + equity)
        records.append(
            {
                "period": period.label,
                "total_assets": assets,
                "total_liabilities": liabilities,
                "equity": equity,
                "residual": residual,
                "residual_pct": (residual / assets * 100.0) if assets else _NAN,
                "balanced": bool(assets and abs(residual) <= abs(assets) * BALANCE_SHEET_TOLERANCE),
            }
        )

    if not records:
        blank["reason"] = (
            "No period reported total assets, total liabilities and equity together, "
            "so the accounting identity could not be tested."
        )
        return blank

    frame = pd.DataFrame(records)
    n_balanced = int(frame["balanced"].sum())
    worst = float(frame["residual_pct"].abs().max())

    # A residual that is POSITIVE in every period is the signature of minority
    # interest: it sits between the provider's two figures and is excluded from
    # both, so assets exceed their sum by a stable margin. A residual that
    # changes sign, or appears in some periods and not others, is not explained
    # by anything structural and is the case worth worrying about. Reporting
    # both as "does not balance" would cry wolf on every holding company.
    positive = frame["residual"] > 0
    consistent = bool(positive.all()) and float(frame["residual_pct"].min()) > 0.0

    if n_balanced == len(frame):
        diagnosis = "balanced"
        reason = (
            f"Assets equal liabilities plus equity in all {len(frame)} periods "
            f"(largest gap {worst:.2f}% of total assets)."
        )
    elif consistent:
        diagnosis = "minority_interest"
        reason = (
            f"Assets exceed liabilities plus equity by {frame['residual_pct'].min():.1f}"
            f"-{worst:.1f}% of total assets in every period. A gap that is positive and "
            f"stable like this is minority interest: the provider's 'total liabilities net "
            f"minority interest' and 'stockholders equity' both exclude it, so it falls out "
            f"of the middle of the identity. The statement is consistent; the equity line "
            f"simply is not the whole of the funding side."
        )
    else:
        diagnosis = "inconsistent"
        reason = (
            f"{len(frame) - n_balanced} of {len(frame)} periods do not balance, and the "
            f"gap is not a stable positive margin (largest {worst:.2f}% of total assets). "
            f"That is not explained by minority interest, so treat this extract — and any "
            f"score computed from it — with suspicion."
        )

    return {
        "ok": n_balanced == len(frame),
        "diagnosis": diagnosis,
        "reason": reason,
        "rows": frame,
        "n_checked": int(len(frame)),
        "n_balanced": n_balanced,
        "worst_residual_pct": worst,
    }
