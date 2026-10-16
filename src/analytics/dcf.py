"""Discounted cash flow valuation — an assumption document, not an answer.

What a DCF is, stated honestly
------------------------------
A DCF does not *find* a company's value. It takes a set of assumptions about
the future and returns the price those assumptions imply. Change the terminal
growth rate by one percentage point and the "intrinsic value" moves by tens of
percent. So this module is built to make the assumptions visible and the
sensitivity legible, because the number on its own is the least informative
part of the calculation.

That is also why :func:`reverse_dcf` exists and why it is arguably the most
useful function here. Instead of asserting a growth rate and reading off a
price, it asks the only question with a defensible answer: **what growth does
the market price already imply?** That is falsifiable. "This stock is worth
1,240" is not.

The two things every DCF gets wrong
-----------------------------------
1. **Terminal value dominates.** In a typical five-year forecast, 60-80% of
   the enterprise value is terminal value — a single Gordon-growth line. This
   module *reports that share* rather than hiding it, and warns when it is
   extreme, because a valuation that is mostly terminal value is a statement
   about the terminal assumption, not about the business.
2. **The exit multiple and the perpetuity growth rate disagree.** Applying an
   exit EBITDA multiple and a Gordon growth rate to the same company should
   give roughly the same terminal value. When they do not, one of the
   assumptions is wrong. Both are computed and both are shown.

Conventions
-----------
* **Free cash flow to the firm** — ``EBIT x (1 - tax) + D&A - capex - change in
  working capital``. Firm-level, so it is discounted at WACC and yields
  enterprise value before debt is removed.
* **Sign of capex.** Capital expenditure arrives from the provider as a negative
  cash outflow. Here it is handled as a positive *magnitude* and subtracted, so
  a sign error cannot silently invert free cash flow. ``capex_pct_revenue`` is
  therefore a positive fraction.
* **Mid-year convention.** Cash arrives through the year, not in a lump on the
  last day, so year *t* is discounted at ``t - 0.5``. This is on by default and
  is worth roughly half a year of discounting.
* **Undefined is NaN, never zero.** A missing input propagates as ``NaN`` and
  the caller renders an em dash. A DCF computed on a zero where a real figure
  belonged is worse than no DCF.

Pure module: stdlib, pandas and scipy only. No Streamlit, no network. Every
figure is passed in, so the whole calculation is testable against hand-built
fixtures with analytically known answers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

from scipy.optimize import brentq

__all__ = [
    "DEFAULT_ERP",
    "DEFAULT_RISK_FREE",
    "DEFAULT_TAX_RATE",
    "DEFAULT_TERMINAL_GROWTH",
    "MAX_TERMINAL_VALUE_SHARE",
    "MIN_WACC_SPREAD",
    "CompanyFinancials",
    "WaccInputs",
    "WaccResult",
    "DcfAssumptions",
    "DcfResult",
    "SensitivityGrid",
    "ReverseDcfResult",
    "cost_of_equity",
    "after_tax_cost_of_debt",
    "wacc",
    "forecast_fcff",
    "run_dcf",
    "sensitivity_grid",
    "reverse_dcf",
    "default_assumptions",
]

#: Indian 10-year G-Sec yield, the standard rupee risk-free proxy. A default
#: for the UI only -- the page seeds this from a live input where it can, and
#: says so when it cannot.
DEFAULT_RISK_FREE: float = 0.070

#: Equity risk premium for an Indian large cap. Damodaran's India figure has
#: run in the 6.5-8% range in recent years; 7% is a defensible mid-point and
#: the page exposes it as an editable assumption like every other input here.
DEFAULT_ERP: float = 0.070

#: Indian corporate tax rate under the concessional regime, inclusive of
#: surcharge and cess. Used only when the statements do not yield an effective
#: rate of their own.
DEFAULT_TAX_RATE: float = 0.2517

#: Terminal growth. Capped below long-run nominal GDP growth on purpose: no
#: company grows faster than the economy forever. Tied to Indian nominal GDP
#: rather than a global figure, since a rupee cash flow is being discounted.
DEFAULT_TERMINAL_GROWTH: float = 0.045

#: WACC must exceed terminal growth or the Gordon formula divides by a
#: negative and returns a negative enterprise value. Anything closer than this
#: is treated as an unusable combination rather than quietly computing a
#: meaningless number.
MIN_WACC_SPREAD: float = 0.005

#: Above this share, the valuation is mostly a statement about the terminal
#: assumption. Reported as a warning, never suppressed.
MAX_TERMINAL_VALUE_SHARE: float = 0.85

_NAN = float("nan")


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Finite float, or NaN. The only way an external input enters a formula."""
    if value is None or isinstance(value, bool):
        return _NAN
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _ok(*values: float) -> bool:
    """True when every value is a real number."""
    return all(not math.isnan(v) for v in values)


def _div(numerator: float, denominator: float) -> float:
    """Ratio, or NaN. Never raises and never returns infinity."""
    if not _ok(numerator, denominator) or denominator == 0.0:
        return _NAN
    out = numerator / denominator
    return out if math.isfinite(out) else _NAN


# ---------------------------------------------------------------------------
# cost of capital
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaccInputs:
    """Everything the cost of capital is built from, so each piece is visible.

    Weights are *target* weights, not today's book weights. Using the current
    book debt/equity to weight the cost of capital is one of the commonest
    errors in a student DCF: a company that happens to be 80% equity funded
    today is not thereby an 80%-equity company forever, and book equity has
    little to do with market equity.
    """

    risk_free: float = DEFAULT_RISK_FREE
    equity_risk_premium: float = DEFAULT_ERP
    beta: float = 1.0
    pretax_cost_of_debt: float = _NAN
    tax_rate: float = DEFAULT_TAX_RATE
    equity_weight: float = _NAN
    debt_weight: float = _NAN
    size_premium: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "risk_free",
            "equity_risk_premium",
            "beta",
            "pretax_cost_of_debt",
            "tax_rate",
            "equity_weight",
            "debt_weight",
            "size_premium",
        ):
            try:
                float(getattr(self, name))
            except (TypeError, ValueError):
                object.__setattr__(self, name, _NAN)


def cost_of_equity(
    risk_free: float,
    beta: float,
    equity_risk_premium: float,
    size_premium: float = 0.0,
) -> float:
    """CAPM cost of equity: ``rf + beta x ERP + size premium``.

    The size premium defaults to zero and is exposed because CAPM with a
    single market factor is known to understate the required return on small
    caps -- a real issue for an Indian mid-cap, and one the reader should be
    able to price rather than have silently applied or silently omitted.
    """
    rf, b, erp, sp = _num(risk_free), _num(beta), _num(equity_risk_premium), _num(size_premium)
    if not _ok(rf, b, erp, sp):
        return _NAN
    return rf + b * erp + sp


def after_tax_cost_of_debt(pretax_cost: float, tax_rate: float) -> float:
    """``kd x (1 - t)``. Interest is deductible, so debt is cheap after tax."""
    kd, t = _num(pretax_cost), _num(tax_rate)
    if not _ok(kd, t):
        return _NAN
    return kd * (1.0 - t)


@dataclass(frozen=True)
class WaccResult:
    """The cost of capital and the build-up behind it."""

    wacc: float = _NAN
    cost_of_equity: float = _NAN
    cost_of_debt: float = _NAN
    equity_weight: float = _NAN
    debt_weight: float = _NAN
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not math.isnan(self.wacc)

    @property
    def build_up(self) -> str:
        """One-line arithmetic a reader can check by hand."""
        if not self.ok:
            return "cost of capital could not be computed"
        return (
            f"{self.equity_weight:.0%} x {self.cost_of_equity:.2%} + "
            f"{self.debt_weight:.0%} x {self.cost_of_debt:.2%} = {self.wacc:.2%}"
        )


def wacc(inputs: WaccInputs) -> WaccResult:
    """Weighted average cost of capital, with the build-up returned.

    Weights are normalised to sum to one when both are given, so a caller that
    supplies 60/40 or 0.6/0.4 gets the same answer. If only one is supplied the
    other is inferred, because debt + equity = capital by construction.
    """
    notices: list[str] = []
    ke = cost_of_equity(
        inputs.risk_free, inputs.beta, inputs.equity_risk_premium, inputs.size_premium
    )
    kd = after_tax_cost_of_debt(inputs.pretax_cost_of_debt, inputs.tax_rate)

    we, wd = _num(inputs.equity_weight), _num(inputs.debt_weight)
    if _ok(we, wd):
        total = we + wd
        if total <= 0:
            return WaccResult(notices=("Capital weights sum to zero.",))
        we, wd = we / total, wd / total
    elif _ok(we):
        wd = 1.0 - we
    elif _ok(wd):
        we = 1.0 - wd
    else:
        return WaccResult(notices=("Neither capital weight was supplied.",))

    if math.isnan(kd):
        notices.append(
            "Cost of debt is unavailable, so the cost of capital is computed on "
            "equity alone and understates the true figure."
        )
        if we >= 1.0:
            return WaccResult(
                wacc=ke,
                cost_of_equity=ke,
                equity_weight=1.0,
                debt_weight=0.0,
                notices=tuple(notices),
            )
        kd = 0.0
    if math.isnan(ke):
        return WaccResult(notices=(*notices, "Cost of equity could not be computed."))

    out = we * ke + wd * kd
    if not math.isfinite(out):
        return WaccResult(notices=(*notices, "The weighted cost of capital was not finite."))
    return WaccResult(
        wacc=out,
        cost_of_equity=ke,
        cost_of_debt=kd,
        equity_weight=we,
        debt_weight=wd,
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# assumptions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompanyFinancials:
    """The bridge from a forecast to a per-share value.

    ``net_debt`` is total debt less cash, and it is what converts enterprise
    value into equity value. ``shares`` is the diluted count where the provider
    gives one, since options and convertibles are claims on the same equity.
    """

    base_revenue: float = _NAN
    net_debt: float = _NAN
    shares: float = _NAN
    current_price: float = _NAN
    base_ebitda: float = _NAN

    @property
    def ok(self) -> bool:
        return _ok(self.base_revenue, self.net_debt, self.shares) and self.shares > 0

    @property
    def missing(self) -> tuple[str, ...]:
        """Which inputs are absent, so the page can say what is blocking it."""
        out = []
        if math.isnan(_num(self.base_revenue)):
            out.append("revenue")
        if math.isnan(_num(self.net_debt)):
            out.append("net debt")
        if math.isnan(_num(self.shares)):
            out.append("share count")
        return tuple(out)


@dataclass(frozen=True)
class DcfAssumptions:
    """The forecast, stated as assumptions rather than buried in code.

    ``revenue_growth`` and ``ebit_margin`` are per-year sequences so a fading
    growth path is expressible; :meth:`constant` builds the flat case. Lengths
    must agree, and the horizon is their length.
    """

    revenue_growth: tuple[float, ...] = (0.10, 0.10, 0.10, 0.10, 0.10)
    ebit_margin: tuple[float, ...] = (0.15, 0.15, 0.15, 0.15, 0.15)
    tax_rate: float = DEFAULT_TAX_RATE
    da_pct_revenue: float = 0.04
    capex_pct_revenue: float = 0.05
    nwc_pct_incremental_revenue: float = 0.10
    wacc_value: float = 0.12
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH
    exit_ebitda_multiple: float = _NAN
    mid_year: bool = True

    @classmethod
    def constant(
        cls,
        years: int = 5,
        revenue_growth: float = 0.10,
        ebit_margin: float = 0.15,
        **kwargs: object,
    ) -> DcfAssumptions:
        """A flat forecast, which is what a first pass should always be."""
        n = max(int(years), 1)
        return cls(
            revenue_growth=(revenue_growth,) * n,
            ebit_margin=(ebit_margin,) * n,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def horizon(self) -> int:
        return len(self.revenue_growth)

    @property
    def problems(self) -> tuple[str, ...]:
        """Reason the assumption set cannot be valued, if any."""
        out: list[str] = []
        if self.horizon == 0:
            out.append("The forecast has no years.")
        if len(self.ebit_margin) != self.horizon:
            out.append("Revenue growth and margin paths are different lengths.")
        if any(math.isnan(g) for g in self.revenue_growth):
            out.append("A revenue growth rate is missing.")
        if any(math.isnan(m) for m in self.ebit_margin):
            out.append("An EBIT margin is missing.")
        if math.isnan(_num(self.wacc_value)):
            out.append("The discount rate is missing.")
        elif _num(self.wacc_value) <= _num(self.terminal_growth) + MIN_WACC_SPREAD:
            out.append("The discount rate must exceed terminal growth.")
        if math.isnan(_num(self.terminal_growth)):
            out.append("The terminal growth rate is missing.")
        return tuple(out)

    def usable(self) -> bool:
        return not self.problems


# ---------------------------------------------------------------------------
# the forecast
# ---------------------------------------------------------------------------
def forecast_fcff(
    assumptions: DcfAssumptions, base_revenue: float
) -> tuple[dict[str, list[float]], float, float]:
    """Project free cash flow to the firm, year by year.

    Returns ``(table, revenue_final, fcff_final)`` where ``table`` holds one
    list per line so the page can render the whole forecast — the working is
    the deliverable, not just the answer.

    Working capital is charged on *incremental* revenue only. A growing company
    ties up more cash in receivables and inventory, and that is a real cost of
    growth; charging it on the whole revenue base every year would double-count
    it and understate the valuation.
    """
    revenue = _num(base_revenue)
    if math.isnan(revenue):
        return {}, _NAN, _NAN

    columns = ("revenue", "ebit", "nopat", "depreciation", "capex", "nwc_change", "fcff")
    table: dict[str, list[float]] = {c: [] for c in columns}

    prior = revenue
    for index in range(assumptions.horizon):
        growth = _num(assumptions.revenue_growth[index])
        margin = _num(assumptions.ebit_margin[index])
        year_revenue = prior * (1.0 + growth) if _ok(growth) else _NAN

        ebit = year_revenue * margin if _ok(year_revenue, margin) else _NAN
        tax = _num(assumptions.tax_rate)
        nopat = ebit * (1.0 - tax) if _ok(ebit, tax) else _NAN

        da = year_revenue * _num(assumptions.da_pct_revenue) if _ok(year_revenue) else _NAN
        capex = year_revenue * _num(assumptions.capex_pct_revenue) if _ok(year_revenue) else _NAN

        # Charged on the increment, and on the first year's increment over the
        # base year's revenue -- which is not zero, hence the explicit delta.
        delta = year_revenue - prior if _ok(year_revenue, prior) else _NAN
        nwc = delta * _num(assumptions.nwc_pct_incremental_revenue) if _ok(delta) else _NAN

        if _ok(nopat, da, capex, nwc):
            fcff = nopat + da - capex - nwc
        else:
            fcff = _NAN

        table["revenue"].append(year_revenue)
        table["ebit"].append(ebit)
        table["nopat"].append(nopat)
        table["depreciation"].append(da)
        table["capex"].append(capex)
        table["nwc_change"].append(nwc)
        table["fcff"].append(fcff)
        prior = year_revenue

    final_revenue = table["revenue"][-1] if table["revenue"] else _NAN
    final_fcff = table["fcff"][-1] if table["fcff"] else _NAN
    return table, final_revenue, final_fcff


# ---------------------------------------------------------------------------
# the valuation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DcfResult:
    """Enterprise value, equity value, per-share value, and the caveats."""

    value_per_share: float = _NAN
    equity_value: float = _NAN
    enterprise_value: float = _NAN
    pv_explicit: float = _NAN
    pv_terminal: float = _NAN
    terminal_value: float = _NAN
    terminal_value_exit: float = _NAN
    fcff_final: float = _NAN
    final_revenue: float = _NAN
    revenue_cagr: float = _NAN
    net_debt: float = _NAN
    shares: float = _NAN
    current_price: float = _NAN
    wacc: float = _NAN
    terminal_growth: float = _NAN
    forecast: dict[str, list[float]] = field(default_factory=dict)
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not math.isnan(self.value_per_share) and self.value_per_share > 0

    @property
    def upside(self) -> float:
        """``value / price - 1``. NaN when either side is unknown."""
        return (
            _div(self.value_per_share, self.current_price) - 1.0
            if _ok(self.value_per_share, self.current_price) and self.current_price > 0
            else _NAN
        )

    @property
    def terminal_share(self) -> float:
        """Fraction of enterprise value that is terminal value.

        Reported because it is usually the largest single number in a DCF and
        almost never shown.
        """
        return _div(self.pv_terminal, self.enterprise_value)

    @property
    def terminal_disagreement(self) -> float:
        """Percent gap between the Gordon and exit-multiple terminal values.

        Zero means the two terminal methods agree, which is the sign of a
        coherent assumption set. A large gap means at least one of the exit
        multiple or the terminal growth rate is wrong.
        """
        if not _ok(self.terminal_value, self.terminal_value_exit):
            return _NAN
        return _div(self.terminal_value - self.terminal_value_exit, self.terminal_value_exit)


def run_dcf(
    assumptions: DcfAssumptions,
    financials: CompanyFinancials,
    effective_tax_rate: float = _NAN,
) -> DcfResult:
    """Value the company under ``assumptions``.

    ``value_per_share`` is NaN whenever anything essential is missing — a
    partial DCF is not a valuation, and returning a number computed from
    substituted zeros is exactly how an unsound model produces a confident
    answer.

    ``effective_tax_rate`` is the rate actually borne in the accounts (tax
    divided by pretax profit). When supplied it overrides the statutory
    default in ``assumptions``, because a company paying 12% is worth more
    than the same company paying 25%, and the statutory rate is a fallback for
    when the statements cannot say.
    """
    head: list[str] = []
    observed_tax = _num(effective_tax_rate)
    if _ok(observed_tax) and observed_tax != _num(assumptions.tax_rate):
        assumptions = _replace(assumptions, tax_rate=observed_tax)
        head.append(
            f"Discounted at the {observed_tax:.1%} effective tax rate from the accounts "
            f"rather than the {DEFAULT_TAX_RATE:.1%} statutory default."
        )

    problems = assumptions.problems
    if problems:
        return DcfResult(notices=tuple(head) + problems)
    if not financials.ok:
        return DcfResult(
            notices=(
                *head,
                "Cannot value this company: missing " + ", ".join(financials.missing) + ".",
            )
        )

    table, final_revenue, final_fcff = forecast_fcff(assumptions, financials.base_revenue)
    if not table:
        return DcfResult(notices=("No forecast could be built from the supplied revenue.",))

    rate = _num(assumptions.wacc_value)
    growth = _num(assumptions.terminal_growth)
    notices: list[str] = list(head)

    if rate <= growth + MIN_WACC_SPREAD:
        return DcfResult(
            notices=(
                f"A {rate:.1%} discount rate against {growth:.1%} terminal growth is not "
                "a usable combination — the perpetuity formula divides by a negative.",
            )
        )

    # -- explicit period --------------------------------------------------
    pv_explicit = 0.0
    for index, cash in enumerate(table["fcff"], start=1):
        cash = _num(cash)
        if math.isnan(cash):
            return DcfResult(notices=("A forecast year produced no cash flow.",))
        exponent = index - 0.5 if assumptions.mid_year else float(index)
        pv_explicit += cash / (1.0 + rate) ** exponent

    # -- terminal value ---------------------------------------------------
    years = assumptions.horizon
    if math.isnan(final_fcff):
        return DcfResult(notices=("The final forecast year produced no cash flow.",))
    terminal_value = final_fcff * (1.0 + growth) / (rate - growth)

    # Discounted from the end of the final year. The mid-year convention
    # applies to within-year cash flows, not to a perpetuity that starts the
    # day year N ends, so this uses the full-year exponent.
    pv_terminal = terminal_value / (1.0 + rate) ** years

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - _num(financials.net_debt)
    per_share = _div(equity_value, _num(financials.shares))

    # -- the exit-multiple cross-check ------------------------------------
    terminal_exit = _NAN
    multiple = _num(assumptions.exit_ebitda_multiple)
    if _ok(multiple, financials.base_ebitda):
        final_ebit = table["ebit"][-1] if table["ebit"] else _NAN
        final_da = table["depreciation"][-1] if table["depreciation"] else _NAN
        if _ok(final_ebit, final_da):
            terminal_exit = (final_ebit + final_da) * multiple

    # -- how fast did we assume it grows? ---------------------------------
    cagr = _div(final_revenue, _num(financials.base_revenue))
    if not math.isnan(cagr) and cagr > 0 and years > 0:
        cagr = cagr ** (1.0 / years) - 1.0
    else:
        cagr = _NAN

    share_of_value = _div(pv_terminal, enterprise_value)
    if not math.isnan(share_of_value) and share_of_value > MAX_TERMINAL_VALUE_SHARE:
        notices.append(
            f"{share_of_value:.0%} of the valuation is terminal value. At this level the "
            "number is mostly a statement about the terminal assumption, not about the "
            "business — read the sensitivity grid, not the headline."
        )
    if equity_value <= 0:
        notices.append(
            "Net debt exceeds enterprise value, so equity is worthless under these "
            "assumptions. Either the debt figure or the forecast is wrong."
        )

    return DcfResult(
        value_per_share=per_share,
        equity_value=equity_value,
        enterprise_value=enterprise_value,
        pv_explicit=pv_explicit,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        terminal_value_exit=terminal_exit,
        fcff_final=final_fcff,
        final_revenue=final_revenue,
        revenue_cagr=cagr,
        net_debt=_num(financials.net_debt),
        shares=_num(financials.shares),
        current_price=_num(financials.current_price),
        wacc=rate,
        terminal_growth=growth,
        forecast=table,
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensitivityGrid:
    """A two-way table of values per share, with the base cell located.

    ``base_row`` / ``base_col`` let the UI highlight the assumption pair the
    headline number came from, so the reader can see how far the answer moves
    for a one-notch change in either direction.
    """

    row_label: str = ""
    col_label: str = ""
    rows: tuple[float, ...] = ()
    cols: tuple[float, ...] = ()
    values: tuple[tuple[float, ...], ...] = ()
    base_row: int = -1
    base_col: int = -1
    current_price: float = _NAN

    @property
    def ok(self) -> bool:
        return bool(self.values)

    @property
    def spread(self) -> float:
        """``max / min - 1`` across the grid: how elastic the answer is."""
        flat = [v for row in self.values for v in row if not math.isnan(v)]
        if len(flat) < 2:
            return _NAN
        low, high = min(flat), max(flat)
        return _div(high - low, low) if low > 0 else _NAN

    @property
    def base_value(self) -> float:
        if self.base_row < 0 or self.base_col < 0:
            return _NAN
        try:
            return self.values[self.base_row][self.base_col]
        except (IndexError, TypeError):
            return _NAN


def _replace(assumption_set: DcfAssumptions, **changes: object) -> DcfAssumptions:
    """A copy of ``assumption_set`` with fields replaced, without dataclasses.replace.

    ``dataclasses.replace`` re-runs ``__post_init__`` and would re-validate the
    tuple lengths mid-grid; a direct field copy is both faster and quieter.
    """
    data = {f.name: getattr(assumption_set, f.name) for f in fields(assumption_set)}
    data.update(changes)
    return DcfAssumptions(**data)  # type: ignore[arg-type]


def sensitivity_grid(
    assumptions: DcfAssumptions,
    financials: CompanyFinancials,
    row_values: Sequence[float],
    col_values: Sequence[float],
    row_field: str = "wacc_value",
    col_field: str = "terminal_growth",
) -> SensitivityGrid:
    """Value the company across a grid of two assumptions.

    Defaults to discount rate against terminal growth, which is the pair a DCF
    is most sensitive to and the pair a reader should look at first. Any two
    scalar fields of :class:`DcfAssumptions` work.
    """
    rows = tuple(_num(v) for v in row_values)
    cols = tuple(_num(v) for v in col_values)
    if not rows or not cols:
        return SensitivityGrid(row_label=row_field, col_label=col_field)

    grid: list[tuple[float, ...]] = []
    for row in rows:
        line: list[float] = []
        for col in cols:
            trial = _replace(assumptions, **{row_field: row, col_field: col})
            line.append(run_dcf(trial, financials).value_per_share)
        grid.append(tuple(line))

    base_row = base_col = -1
    for index, row in enumerate(rows):
        if abs(row - _num(getattr(assumptions, row_field, _NAN))) < 1e-12 and base_row < 0:
            base_row = index
    for index, col in enumerate(cols):
        if abs(col - _num(getattr(assumptions, col_field, _NAN))) < 1e-12 and base_col < 0:
            base_col = index

    return SensitivityGrid(
        row_label=row_field,
        col_label=col_field,
        rows=rows,
        cols=cols,
        values=tuple(grid),
        base_row=base_row,
        base_col=base_col,
        current_price=_num(financials.current_price),
    )


# ---------------------------------------------------------------------------
# reverse DCF
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReverseDcfResult:
    """The growth the current price implies — the falsifiable question."""

    implied_growth: float = _NAN
    implied_margin: float = _NAN
    solved_for: str = ""
    current_price: float = _NAN
    lo: float = _NAN
    hi: float = _NAN
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not math.isnan(self.implied_growth) or not math.isnan(self.implied_margin)


def reverse_dcf(
    assumptions: DcfAssumptions,
    financials: CompanyFinancials,
    field_name: str = "revenue_cagr",
    lo: float = -0.20,
    hi: float = 0.60,
) -> ReverseDcfResult:
    """Solve for the single assumption that reproduces today's price.

    ``field_name`` selects which assumption is solved for:

    * ``"revenue_cagr"`` — flattens the growth path to a constant and solves
      for it. The answer is directly comparable to a growth rate a reader has
      an opinion about.
    * ``"wacc_value"`` — solves for the discount rate the price implies.

    Bracket-and-bisect via ``brentq``, which needs the value to change sign
    across ``[lo, hi]``. When it does not — a price the model cannot reach even
    at 60% growth — the result is NaN with the bracket reported, rather than a
    boundary value presented as an answer.
    """
    price = _num(financials.current_price)
    if math.isnan(price) or price <= 0:
        return ReverseDcfResult(
            solved_for=field_name, notices=("A current price is needed to invert the model.",)
        )
    if not financials.ok:
        return ReverseDcfResult(
            solved_for=field_name,
            notices=("Cannot invert: missing " + ", ".join(financials.missing) + ".",),
        )

    years = assumptions.horizon

    def gap(value: float) -> float:
        if field_name == "revenue_cagr":
            trial = _replace(assumptions, revenue_growth=(value,) * years)
        elif field_name == "wacc_value":
            trial = _replace(assumptions, wacc_value=value)
        else:
            trial = _replace(assumptions, **{field_name: value})
        result = run_dcf(trial, financials)
        if not result.ok:
            # Discontinuous regions (rate below growth) are pushed to a large
            # positive gap so brentq stays on the continuous branch instead of
            # bisecting across a division by zero.
            return 1e9
        return result.value_per_share - price

    try:
        solved = brentq(gap, _num(lo), _num(hi), maxiter=200, xtol=1e-8)
    except (ValueError, RuntimeError) as exc:
        return ReverseDcfResult(
            solved_for=field_name,
            current_price=price,
            lo=_num(lo),
            hi=_num(hi),
            notices=(
                f"No growth rate between {_num(lo):.0%} and {_num(hi):.0%} reproduces a "
                f"price of {price:,.2f} ({type(exc).__name__}). The price sits outside "
                "the range this model can produce.",
            ),
        )

    if field_name == "revenue_cagr":
        return ReverseDcfResult(
            implied_growth=solved,
            solved_for=field_name,
            current_price=price,
            lo=_num(lo),
            hi=_num(hi),
        )
    if field_name == "wacc_value":
        return ReverseDcfResult(
            implied_growth=solved,
            solved_for="discount rate",
            current_price=price,
            lo=_num(lo),
            hi=_num(hi),
        )
    return ReverseDcfResult(
        implied_margin=solved,
        solved_for=field_name,
        current_price=price,
        lo=_num(lo),
        hi=_num(hi),
    )


# ---------------------------------------------------------------------------
# seeding assumptions from history
# ---------------------------------------------------------------------------
def default_assumptions(
    historical_growth: Sequence[float] = (),
    historical_margin: Sequence[float] = (),
    years: int = 5,
) -> DcfAssumptions:
    """Seed a defensible first-pass forecast from the company's own history.

    Growth is the *median of the historical rates, faded towards the long-run
    rate*, never the latest year extrapolated. A company that just grew 40%
    does not grow 40% for five more years, and a model that asserts so is not a
    forecast but a wish. The fade is linear from the historical figure down to
    ``DEFAULT_TERMINAL_GROWTH`` by the final forecast year, which is a
    transparent rule the reader can argue with.

    Takes only *rates*, never the revenue level: revenue itself belongs to the
    company being valued (:class:`CompanyFinancials`), and a signature that
    accepted it here would imply the forecast was scaled to it when nothing in
    this function touches it.
    """
    n = max(int(years), 1)
    growth = [g for g in (_num(v) for v in historical_growth) if not math.isnan(g)]
    margin = [m for m in (_num(v) for v in historical_margin) if not math.isnan(m)]

    if growth:
        ordered = sorted(growth)
        mid = len(ordered) // 2
        start_growth = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        # Never seed a forecast above what was actually achieved recently, and
        # never below a floor that would imply the business shrinks forever.
        start_growth = max(min(start_growth, max(growth)), -0.10)
    else:
        start_growth = 0.08

    end_growth = DEFAULT_TERMINAL_GROWTH
    path = tuple(start_growth + (end_growth - start_growth) * (i / max(n - 1, 1)) for i in range(n))

    if margin:
        mid_margin = margin[-1]
    else:
        mid_margin = 0.12

    return DcfAssumptions(
        revenue_growth=path,
        ebit_margin=(mid_margin,) * n,
    )


def historical_rates(statements: object, field_name: str = "revenue") -> tuple[float, ...]:
    """Year-over-year growth rates of one statement field, newest last.

    Kept here rather than in the page so the *definition* of "historical
    growth" is one function, testable, and the same everywhere it is quoted.
    """
    series = getattr(statements, "series", None)
    if not callable(series):
        return ()
    values = series(field_name)
    if values is None or len(values) < 2:
        return ()
    shifts = values.shift(1)
    rates = (values - shifts) / shifts.abs()
    return tuple(_num(v) for v in rates.dropna())
