"""Indian capital-gains arithmetic and tax-loss harvesting for listed equity.

What this does
--------------
Takes the FIFO tax lots :mod:`src.analytics.lots` already produces, classifies
every open lot as short- or long-term against the 12-month boundary, and works
out what realising each losing lot would actually save in tax **given the gains
already realised this financial year**.

That last clause is the whole point, and it is what most harvesting tools get
wrong. A loss is not worth ``loss x tax rate``. It is worth whatever the
set-off rules let it absorb:

* short-term capital LOSS can be set off against short-term AND long-term gains
* long-term capital LOSS can be set off against long-term gains ONLY
* long-term gains carry an annual exemption, so offsetting them below that
  threshold saves exactly nothing
* anything unabsorbed carries forward up to eight assessment years, and only if
  the return is filed on time

So harvesting a Rs 50,000 long-term loss against Rs 20,000 of long-term gains
in a year where the exemption has not been used saves **zero rupees today**.
Reporting it as "saves Rs 6,250" would be a fabrication, and this module
reports the carry-forward instead.

The counter-intuitive result this makes visible
-----------------------------------------------
For a LOSS, short-term is worth more than long-term: it offsets a wider set of
gains at a higher rate. So a losing lot approaching its first anniversary is
*more* valuable harvested before the boundary than after -- the opposite of the
usual "hold for long-term treatment" instinct, which is correct for gains and
backwards for losses. :func:`boundary_alerts` surfaces both cases.

On wash sales
-------------
India has **no general wash-sale rule** for listed equity. Selling to realise a
loss and repurchasing immediately is not disallowed the way the US 30-day rule
disallows it. That is a genuine structural advantage and this module says so --
but it is narrower than "anything goes": Section 94(7) still disallows losses
arising from dividend stripping, and general anti-avoidance provisions exist.
:func:`harvest_plan` attaches that caveat rather than implying a blank cheque.

Grandfathering (Section 112A)
-----------------------------
For equity acquired on or before 31 January 2018 the cost of acquisition is the
higher of actual cost and the lower of (31-Jan-2018 fair market value, sale
consideration). The statutory FMV is the **highest quoted price on that date**,
which this application does not hold -- an adjusted closing price is neither
the high nor unadjusted. So a pre-cutoff lot is FLAGGED as needing that figure
and its gain is computed on actual cost with the limitation stated, rather than
silently using a number that is not the statutory one.

Not tax advice
--------------
A simplified model of Sections 111A, 112A and 70-74, adequate for ranking
harvesting candidates and no substitute for a chartered accountant. Surcharge,
cess, residency, set-off against other heads and every non-equity asset class
are out of scope. See :data:`src.config.TAX_DISCLAIMER`.

Pure module: pandas/numpy only. No I/O, no network, no Streamlit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from src.config import (
    FINANCIAL_YEAR_START_MONTH,
    GRANDFATHER_ACQUISITION_CUTOFF,
    INDIAN_TAX_REGIMES,
    LOSS_CARRY_FORWARD_YEARS,
    TAX_BOUNDARY_WARNING_DAYS,
    IndianEquityTaxRegime,
)
from src.models import Lot, RealisedTrade

logger = logging.getLogger(__name__)

__all__ = [
    "SHORT_TERM",
    "LONG_TERM",
    "TaxLot",
    "GainsPosition",
    "SetOffResult",
    "HarvestCandidate",
    "HarvestPlan",
    "financial_year",
    "financial_year_label",
    "regime_for",
    "holding_days",
    "is_long_term",
    "days_to_long_term",
    "classify_lots",
    "realised_gains",
    "tax_on",
    "set_off",
    "harvest_candidates",
    "harvest_plan",
    "boundary_alerts",
]

SHORT_TERM = "short"
LONG_TERM = "long"

_NAN = float("nan")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001 - exotic input
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).date()


# ---------------------------------------------------------------------------
# calendar and regime
# ---------------------------------------------------------------------------
def financial_year(as_of: object = None) -> tuple[date, date]:
    """The Indian financial year containing ``as_of``: ``(1 April, 31 March)``.

    The Indian FY runs April to March, so a January sale belongs to the year
    that STARTED the previous April. Using the calendar year instead would put
    Q4 gains in the wrong year and misstate every exemption calculation.
    """
    today = _as_date(as_of) or date.today()
    year = today.year if today.month >= FINANCIAL_YEAR_START_MONTH else today.year - 1
    start = date(year, FINANCIAL_YEAR_START_MONTH, 1)
    end = date(year + 1, FINANCIAL_YEAR_START_MONTH, 1) - timedelta(days=1)
    return start, end


def financial_year_label(as_of: object = None) -> str:
    """``"FY2024-25"`` for the financial year containing ``as_of``."""
    start, end = financial_year(as_of)
    return f"FY{start.year}-{str(end.year)[-2:]}"


def regime_for(as_of: object = None) -> IndianEquityTaxRegime:
    """The tax regime in force on ``as_of``.

    India changed the rates mid-year on 23 July 2024, so this is a lookup
    rather than a constant: a sale on 22 July 2024 is taxed at 15%/10% and one
    two days later at 20%/12.5%.
    """
    when = _as_date(as_of) or date.today()
    chosen = INDIAN_TAX_REGIMES[0]
    for regime in INDIAN_TAX_REGIMES:
        if when >= regime.effective_from:
            chosen = regime
    return chosen


def holding_days(open_date: object, as_of: object = None) -> int:
    """Calendar days held. ``-1`` when the open date is unusable."""
    opened = _as_date(open_date)
    today = _as_date(as_of) or date.today()
    if opened is None:
        return -1
    return int((today - opened).days)


def is_long_term(days: int, regime: IndianEquityTaxRegime | None = None) -> bool:
    """True when a holding is long-term: held MORE than 12 months.

    Strictly greater than 365 days, matching ``RealisedTrade.is_long_term``.
    Exactly 365 days is still short-term, and a tool that rounded that the
    other way would misclassify every position sold on its anniversary.
    """
    rules = regime or regime_for()
    return days > rules.long_term_days


def days_to_long_term(days: int, regime: IndianEquityTaxRegime | None = None) -> int:
    """Days remaining until long-term treatment. ``0`` once already there."""
    rules = regime or regime_for()
    return max(0, rules.long_term_days + 1 - days)


# ---------------------------------------------------------------------------
# lots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaxLot:
    """One open FIFO lot, with its unrealised gain and tax character."""

    ticker: str
    open_date: date
    quantity: float
    cost_per_share: float
    price: float
    days_held: int
    character: str
    grandfathered: bool = False

    @property
    def cost_basis(self) -> float:
        return float(self.quantity * self.cost_per_share)

    @property
    def market_value(self) -> float:
        return float(self.quantity * self.price)

    @property
    def unrealised(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealised_pct(self) -> float:
        basis = self.cost_basis
        return float(self.unrealised / basis * 100.0) if basis > 0 else _NAN

    @property
    def is_loss(self) -> bool:
        return self.unrealised < 0.0

    @property
    def is_long(self) -> bool:
        return self.character == LONG_TERM

    def days_to_boundary(self, regime: IndianEquityTaxRegime | None = None) -> int:
        return days_to_long_term(self.days_held, regime)


def classify_lots(
    open_lots: Sequence[Lot] | None,
    prices: Mapping[str, float] | None,
    as_of: object = None,
    regime: IndianEquityTaxRegime | None = None,
) -> tuple[list[TaxLot], list[str]]:
    """Classify open lots by holding period and mark their unrealised gain.

    ``(lots, warnings)``. A lot whose ticker has no price is SKIPPED with a
    warning rather than valued at cost: pretending an unpriced holding is flat
    would put a zero-gain lot into the harvest ranking as if it were analysed.

    Per LOT, not per position, deliberately. FIFO lots of the same ticker have
    different open dates, so one tranche can be short-term and another long-term
    -- and harvesting is a lot-level decision. Aggregating first would destroy
    exactly the information the feature needs.
    """
    rules = regime or regime_for(as_of)
    today = _as_date(as_of) or date.today()
    prices = prices or {}

    out: list[TaxLot] = []
    warnings: list[str] = []
    unpriced: list[str] = []

    for lot in open_lots or ():
        ticker = str(getattr(lot, "ticker", "") or "").strip()
        opened = _as_date(getattr(lot, "open_date", None))
        quantity = _to_float(getattr(lot, "quantity", _NAN))
        cost = _to_float(getattr(lot, "cost_per_share", _NAN))
        if not ticker or opened is None or not math.isfinite(quantity) or quantity <= 0:
            warnings.append(f"Skipped an unusable lot for {ticker or 'an unnamed holding'}.")
            continue
        if not math.isfinite(cost) or cost < 0:
            warnings.append(f"{ticker}: lot opened {opened} has no usable cost basis.")
            continue

        price = _to_float(prices.get(ticker, _NAN))
        if not math.isfinite(price) or price <= 0:
            if ticker not in unpriced:
                unpriced.append(ticker)
            continue

        days = int((today - opened).days)
        out.append(
            TaxLot(
                ticker=ticker,
                open_date=opened,
                quantity=quantity,
                cost_per_share=cost,
                price=price,
                days_held=days,
                character=LONG_TERM if is_long_term(days, rules) else SHORT_TERM,
                grandfathered=opened <= GRANDFATHER_ACQUISITION_CUTOFF,
            )
        )

    if unpriced:
        warnings.append(
            f"No current price for {', '.join(unpriced)} — those lots are excluded "
            "from the harvest analysis rather than valued at cost."
        )

    grandfathered = sorted({lot.ticker for lot in out if lot.grandfathered})
    if grandfathered:
        warnings.append(
            f"{', '.join(grandfathered)} include lots acquired on or before "
            f"{GRANDFATHER_ACQUISITION_CUTOFF:%d %b %Y}, so Section 112A grandfathering "
            "applies: the cost of acquisition is the higher of actual cost and the "
            "lower of (31-Jan-2018 fair market value, sale price). The statutory FMV is "
            "the HIGHEST QUOTED PRICE that day, which this application does not hold — "
            "the gains shown for those lots use actual cost and will therefore be "
            "OVERSTATED. Supply the FMV to your accountant for the real figure."
        )
    return out, warnings


# ---------------------------------------------------------------------------
# realised gains this year
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GainsPosition:
    """Gains and losses already realised in one financial year, by character."""

    stcg: float = 0.0
    stcl: float = 0.0
    ltcg: float = 0.0
    ltcl: float = 0.0
    n_trades: int = 0
    year_label: str = ""

    @property
    def net_stcg(self) -> float:
        return self.stcg - self.stcl

    @property
    def net_ltcg(self) -> float:
        return self.ltcg - self.ltcl

    @property
    def has_gains(self) -> bool:
        return self.stcg > 0.0 or self.ltcg > 0.0


def realised_gains(
    trades: Iterable[RealisedTrade] | None,
    as_of: object = None,
    regime: IndianEquityTaxRegime | None = None,
) -> GainsPosition:
    """Split the financial year's realised trades into the four buckets.

    Only trades CLOSED inside the financial year containing ``as_of`` count:
    capital gains are assessed year by year, and last year's realised profit
    cannot absorb this year's harvested loss.
    """
    start, end = financial_year(as_of)
    rules = regime or regime_for(as_of)

    stcg = stcl = ltcg = ltcl = 0.0
    count = 0
    for trade in trades or ():
        closed = _as_date(getattr(trade, "close_date", None))
        opened = _as_date(getattr(trade, "open_date", None))
        pnl = _to_float(getattr(trade, "realised_pnl", _NAN))
        if closed is None or opened is None or not math.isfinite(pnl):
            continue
        if not (start <= closed <= end):
            continue
        count += 1
        long_term = is_long_term(int((closed - opened).days), rules)
        if pnl >= 0:
            if long_term:
                ltcg += pnl
            else:
                stcg += pnl
        elif long_term:
            ltcl += abs(pnl)
        else:
            stcl += abs(pnl)

    return GainsPosition(
        stcg=stcg,
        stcl=stcl,
        ltcg=ltcg,
        ltcl=ltcl,
        n_trades=count,
        year_label=financial_year_label(as_of),
    )


# ---------------------------------------------------------------------------
# the set-off waterfall
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SetOffResult:
    """The outcome of applying losses to gains under Sections 70/71/74."""

    taxable_stcg: float = 0.0
    taxable_ltcg: float = 0.0
    exemption_used: float = 0.0
    stcl_used_against_stcg: float = 0.0
    stcl_used_against_ltcg: float = 0.0
    ltcl_used_against_ltcg: float = 0.0
    stcl_carried_forward: float = 0.0
    ltcl_carried_forward: float = 0.0
    #: Loss consumed against long-term gains that the annual exemption would
    #: have sheltered anyway -- absorbed by law, worth nothing in tax. See
    #: :func:`set_off` for why this cannot be avoided.
    wasted_on_exempt_gains: float = 0.0
    tax: float = 0.0
    regime: str = ""

    @property
    def total_carried_forward(self) -> float:
        return self.stcl_carried_forward + self.ltcl_carried_forward

    @property
    def losses_used(self) -> float:
        return (
            self.stcl_used_against_stcg + self.stcl_used_against_ltcg + self.ltcl_used_against_ltcg
        )

    @property
    def has_waste(self) -> bool:
        return self.wasted_on_exempt_gains > 0.0


def tax_on(
    stcg: float, ltcg: float, regime: IndianEquityTaxRegime | None = None
) -> tuple[float, float]:
    """Tax on already-net gains. ``(tax, exemption_used)``.

    The long-term exemption applies to the NET long-term gain after set-off,
    which is why it cannot be applied earlier in the waterfall.
    """
    rules = regime or regime_for()
    short = max(_to_float(stcg) or 0.0, 0.0)
    long_gain = max(_to_float(ltcg) or 0.0, 0.0)
    exemption = min(long_gain, rules.ltcg_exemption)
    taxable_long = max(0.0, long_gain - exemption)
    return (
        short * rules.stcg_rate + taxable_long * rules.ltcg_rate,
        exemption,
    )


def set_off(
    position: GainsPosition,
    regime: IndianEquityTaxRegime | None = None,
) -> SetOffResult:
    """Apply losses to gains under Sections 70/71/74, then compute the tax.

    Two things are true and they pull in different directions, which is what
    makes this worth writing out:

    **Quantum is MANDATORY.** Sections 70 and 71 say a capital loss *shall* be
    set off against available capital gains of the permitted character in the
    same year. A taxpayer cannot elect to carry a loss forward while holding
    gains it could absorb. This has a painful consequence the tool must
    surface rather than optimise away: a long-term loss set off against a
    long-term gain that the annual exemption already shelters is **consumed
    for no tax benefit at all**. An earlier draft of this function "optimised"
    around that by only offsetting gains above the exemption — which would
    have produced a lower tax figure than the law permits, and is the single
    most dangerous kind of bug this module could contain.

    **Ordering is ELECTIVE.** Where a loss may be set off against more than
    one kind of gain, the statute prescribes no sequence, so the beneficial
    order is available:

    1. **STCL against STCG** — the highest rate, so the most tax saved per
       rupee of loss.
    2. **LTCL against LTCG** — LTCL can go nowhere else, so it takes first
       claim on the long-term gains before STCL competes for them. Reversing
       this would strand a usable LTCL while spending an STCL that could have
       absorbed short-term gains.
    3. **Remaining STCL against LTCG** — in full, because it is mandatory.

    ``wasted_on_exempt_gains`` reports the loss destroyed in step 2 or 3
    against gains the exemption would have covered anyway. That number is the
    reason to think about *when* to harvest, and no amount of arithmetic here
    can make it go away.
    """
    rules = regime or regime_for()

    stcg = max(position.stcg, 0.0)
    ltcg = max(position.ltcg, 0.0)
    stcl = max(position.stcl, 0.0)
    ltcl = max(position.ltcl, 0.0)
    gross_ltcg = ltcg

    # 1. STCL -> STCG (elective order: highest rate first)
    used_st_st = min(stcl, stcg)
    stcg -= used_st_st
    stcl -= used_st_st

    # 2. LTCL -> LTCG (mandatory, and LTCL has nowhere else to go)
    used_lt_lt = min(ltcl, ltcg)
    ltcg -= used_lt_lt
    ltcl -= used_lt_lt

    # 3. Remaining STCL -> LTCG, IN FULL. Mandatory under Section 70(2); the
    #    exemption does not create an opt-out.
    used_st_lt = min(stcl, ltcg)
    ltcg -= used_st_lt
    stcl -= used_st_lt

    # How much of the loss applied to long-term gains actually bought
    # anything. Loss that pushed the gain below the exemption threshold did
    # not: that rupee was already tax-free.
    applied_to_ltcg = used_lt_lt + used_st_lt
    headroom = max(0.0, gross_ltcg - rules.ltcg_exemption)
    useful = min(applied_to_ltcg, headroom)
    wasted = max(0.0, applied_to_ltcg - useful)

    tax, exemption_used = tax_on(stcg, ltcg, rules)

    return SetOffResult(
        taxable_stcg=stcg,
        taxable_ltcg=max(0.0, ltcg - exemption_used),
        exemption_used=exemption_used,
        stcl_used_against_stcg=used_st_st,
        stcl_used_against_ltcg=used_st_lt,
        ltcl_used_against_ltcg=used_lt_lt,
        stcl_carried_forward=stcl,
        ltcl_carried_forward=ltcl,
        wasted_on_exempt_gains=wasted,
        tax=tax,
        regime=rules.label,
    )


# ---------------------------------------------------------------------------
# harvesting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HarvestCandidate:
    """One losing lot, with what realising it would actually save.

    ``tax_saved_now`` is the marginal reduction in this year's tax bill,
    computed by re-running the whole set-off waterfall with this loss added --
    not by multiplying the loss by a rate. Those differ whenever the exemption
    or the set-off hierarchy binds, which is most of the time.
    """

    lot: TaxLot
    loss: float
    character: str
    tax_saved_now: float
    carried_forward: float
    days_to_boundary: int

    @property
    def ticker(self) -> str:
        return self.lot.ticker

    @property
    def saves_nothing_now(self) -> bool:
        return self.tax_saved_now <= 0.0

    @property
    def effective_rate(self) -> float:
        """Tax saved per rupee of loss realised. 0 when nothing is absorbed."""
        return self.tax_saved_now / self.loss if self.loss > 0 else 0.0


def harvest_candidates(
    lots: Sequence[TaxLot],
    position: GainsPosition,
    regime: IndianEquityTaxRegime | None = None,
) -> list[HarvestCandidate]:
    """Rank losing lots by the tax each would actually save, best first.

    Each candidate is priced MARGINALLY against the current position: the
    waterfall is re-run with that one loss added and the difference taken. That
    is the only correct way to price it, because the value of a loss depends
    entirely on what gains are available to absorb it -- the same Rs 50,000
    long-term loss is worth Rs 6,250 in a year with large long-term gains and
    exactly nothing in a year with none.

    Ties break on the larger loss then the ticker, so the ordering is stable
    and the same portfolio always renders the same list.
    """
    rules = regime or regime_for()
    baseline = set_off(position, rules).tax

    candidates: list[HarvestCandidate] = []
    for lot in lots:
        if not lot.is_loss:
            continue
        loss = abs(lot.unrealised)
        if loss <= 0:
            continue

        if lot.character == LONG_TERM:
            hypothetical = GainsPosition(
                stcg=position.stcg,
                stcl=position.stcl,
                ltcg=position.ltcg,
                ltcl=position.ltcl + loss,
            )
        else:
            hypothetical = GainsPosition(
                stcg=position.stcg,
                stcl=position.stcl + loss,
                ltcg=position.ltcg,
                ltcl=position.ltcl,
            )

        after = set_off(hypothetical, rules)
        saved = max(0.0, baseline - after.tax)
        absorbed = saved / (rules.ltcg_rate if lot.character == LONG_TERM else rules.stcg_rate)
        candidates.append(
            HarvestCandidate(
                lot=lot,
                loss=loss,
                character=lot.character,
                tax_saved_now=saved,
                carried_forward=max(0.0, loss - absorbed),
                days_to_boundary=lot.days_to_boundary(rules),
            )
        )

    candidates.sort(key=lambda c: (-c.tax_saved_now, -c.loss, c.ticker))
    return candidates


@dataclass(frozen=True)
class HarvestPlan:
    """Everything the harvesting analysis produced."""

    as_of: date
    regime: IndianEquityTaxRegime
    year_label: str = ""
    position: GainsPosition = field(default_factory=GainsPosition)
    before: SetOffResult = field(default_factory=SetOffResult)
    after: SetOffResult = field(default_factory=SetOffResult)
    candidates: tuple[HarvestCandidate, ...] = ()
    selected: tuple[HarvestCandidate, ...] = ()
    lots: tuple[TaxLot, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.lots)

    @property
    def total_loss_available(self) -> float:
        return sum(c.loss for c in self.candidates)

    @property
    def total_loss_selected(self) -> float:
        return sum(c.loss for c in self.selected)

    @property
    def tax_saved(self) -> float:
        """Reduction in this year's bill from the selected harvest."""
        return max(0.0, self.before.tax - self.after.tax)

    @property
    def carried_forward_created(self) -> float:
        return max(0.0, self.after.total_carried_forward - self.before.total_carried_forward)

    def candidates_frame(self) -> pd.DataFrame:
        """One row per harvestable lot, ranked."""
        if not self.candidates:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "open_date",
                    "quantity",
                    "days_held",
                    "character",
                    "cost_basis",
                    "market_value",
                    "loss",
                    "tax_saved_now",
                    "carried_forward",
                    "effective_rate_pct",
                    "days_to_long_term",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "ticker": c.ticker,
                    "open_date": c.lot.open_date,
                    "quantity": c.lot.quantity,
                    "days_held": c.lot.days_held,
                    "character": "Long-term" if c.character == LONG_TERM else "Short-term",
                    "cost_basis": c.lot.cost_basis,
                    "market_value": c.lot.market_value,
                    "loss": c.loss,
                    "tax_saved_now": c.tax_saved_now,
                    "carried_forward": c.carried_forward,
                    "effective_rate_pct": c.effective_rate * 100.0,
                    "days_to_long_term": c.days_to_boundary,
                }
                for c in self.candidates
            ]
        )

    def lots_frame(self) -> pd.DataFrame:
        """Every classified lot, gains and losses alike."""
        if not self.lots:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ticker": lot.ticker,
                    "open_date": lot.open_date,
                    "quantity": lot.quantity,
                    "days_held": lot.days_held,
                    "character": "Long-term" if lot.is_long else "Short-term",
                    "cost_basis": lot.cost_basis,
                    "market_value": lot.market_value,
                    "unrealised": lot.unrealised,
                    "unrealised_pct": lot.unrealised_pct,
                    "days_to_long_term": lot.days_to_boundary(self.regime),
                    "grandfathered": lot.grandfathered,
                }
                for lot in self.lots
            ]
        )


def harvest_plan(
    open_lots: Sequence[Lot] | None,
    prices: Mapping[str, float] | None,
    realised: Iterable[RealisedTrade] | None = None,
    as_of: object = None,
    select: Sequence[str] | None = None,
    max_candidates: int | None = None,
) -> HarvestPlan:
    """The full harvesting analysis. Never raises.

    ``select`` names the lots to harvest, as ``"TICKER@YYYY-MM-DD"`` keys; when
    ``None`` every candidate that saves tax today is selected, which is the
    sensible default (harvesting a loss that saves nothing this year is a real
    choice, but it should be a deliberate one).

    ``max_candidates`` truncates the ranked list for display only; the set-off
    arithmetic always uses the full selection.
    """
    today = _as_date(as_of) or date.today()
    rules = regime_for(today)

    lots, warnings = classify_lots(open_lots, prices, today, rules)
    position = realised_gains(realised, today, rules)
    before = set_off(position, rules)
    candidates = harvest_candidates(lots, position, rules)

    if select is None:
        chosen = tuple(c for c in candidates if c.tax_saved_now > 0.0)
    else:
        wanted = {str(k) for k in select}
        chosen = tuple(
            c for c in candidates if f"{c.ticker}@{c.lot.open_date.isoformat()}" in wanted
        )

    harvested_stcl = sum(c.loss for c in chosen if c.character == SHORT_TERM)
    harvested_ltcl = sum(c.loss for c in chosen if c.character == LONG_TERM)
    after = set_off(
        GainsPosition(
            stcg=position.stcg,
            stcl=position.stcl + harvested_stcl,
            ltcg=position.ltcg,
            ltcl=position.ltcl + harvested_ltcl,
            n_trades=position.n_trades,
            year_label=position.year_label,
        ),
        rules,
    )

    notes: list[str] = []
    if not position.has_gains and candidates:
        notes.append(
            f"No gains have been realised in {position.year_label}, so harvesting a "
            "loss saves nothing THIS year. It still creates a carry-forward usable for "
            f"up to {LOSS_CARRY_FORWARD_YEARS} assessment years — but only if the "
            "return is filed by the due date under Section 139(1)."
        )
    if position.ltcg > 0 and position.ltcg <= rules.ltcg_exemption:
        notes.append(
            f"Long-term gains of Rs {position.ltcg:,.0f} are already inside the "
            f"Rs {rules.ltcg_exemption:,.0f} annual exemption. Set-off is MANDATORY "
            "under Section 70, so a long-term loss harvested now would be consumed "
            "against that already-tax-free gain and buy nothing — it cannot be held "
            "back for a future year instead. Short-term losses can still offset "
            "short-term gains at the higher rate."
        )
    if after.has_waste:
        notes.append(
            f"Rs {after.wasted_on_exempt_gains:,.0f} of loss is absorbed against "
            "long-term gains the exemption already shelters, so it produces no tax "
            "saving. Set-off is compulsory, not elective — the only way to avoid this "
            "is to harvest in a different year, not to choose differently now."
        )
    notes.append(
        "India has no general wash-sale rule for listed equity, so repurchasing a "
        "harvested holding is not disallowed the way the US 30-day rule disallows it. "
        "That is narrower than 'anything goes': Section 94(7) still disallows losses "
        "arising from dividend stripping, and general anti-avoidance provisions apply."
    )

    return HarvestPlan(
        as_of=today,
        regime=rules,
        year_label=position.year_label,
        position=position,
        before=before,
        after=after,
        candidates=tuple(candidates[:max_candidates] if max_candidates else candidates),
        selected=chosen,
        lots=tuple(lots),
        warnings=tuple(dict.fromkeys(warnings)),
        notes=tuple(dict.fromkeys(notes)),
    )


# ---------------------------------------------------------------------------
# the 12-month boundary
# ---------------------------------------------------------------------------
def boundary_alerts(
    lots: Sequence[TaxLot],
    regime: IndianEquityTaxRegime | None = None,
    window_days: int = TAX_BOUNDARY_WARNING_DAYS,
) -> pd.DataFrame:
    """Lots whose tax character is about to change, with what to do about it.

    The two cases point in OPPOSITE directions, which is the insight:

    * A **gain** approaching the boundary should usually WAIT — crossing it
      moves the gain from the short-term rate to the lower long-term rate.
    * A **loss** approaching the boundary is usually better harvested BEFORE —
      a short-term loss offsets both short- and long-term gains at the higher
      rate, while a long-term loss can only offset long-term gains. Holding a
      loss for "long-term treatment" makes it less useful, which is the
      opposite of the instinct that applies to gains.

    Columns: ``ticker, open_date, days_held, days_to_long_term, unrealised,
    character, action, reason``. Empty frame when nothing is near the boundary.
    """
    rules = regime or regime_for()
    columns = [
        "ticker",
        "open_date",
        "days_held",
        "days_to_long_term",
        "unrealised",
        "character",
        "action",
        "reason",
    ]

    rows: list[dict[str, object]] = []
    for lot in lots or ():
        remaining = lot.days_to_boundary(rules)
        if lot.is_long or remaining <= 0 or remaining > max(int(window_days), 0):
            continue
        if lot.is_loss:
            action = "Consider harvesting BEFORE the boundary"
            reason = (
                f"A short-term loss offsets both short- and long-term gains, and at "
                f"{rules.stcg_pct:.0f}%. In {remaining} day(s) it becomes long-term, "
                f"which can only offset long-term gains at {rules.ltcg_pct:.1f}% — so "
                "waiting makes this loss less useful, not more."
            )
        else:
            action = "Consider waiting"
            reason = (
                f"Selling now taxes this gain at {rules.stcg_pct:.0f}%. Holding "
                f"{remaining} more day(s) moves it to {rules.ltcg_pct:.1f}% with the "
                f"annual Rs {rules.ltcg_exemption:,.0f} exemption available."
            )
        rows.append(
            {
                "ticker": lot.ticker,
                "open_date": lot.open_date,
                "days_held": lot.days_held,
                "days_to_long_term": remaining,
                "unrealised": lot.unrealised,
                "character": "Short-term",
                "action": action,
                "reason": reason,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("days_to_long_term")
