"""Tax-loss harvesting UI: what a loss is actually worth, and when to take it.

Mounted by the Trades & lots page, which already owns the FIFO ledger this
needs.

The design constraint that shapes everything here: the honest answer is often
"this saves you nothing". A harvesting screen that always shows an encouraging
number would be worse than useless, because a user acting on a fabricated
saving pays real brokerage and STT to achieve nothing. So the panel leads with
the tax actually saved, states plainly when that is zero, and separates the
carry-forward from the cash benefit.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

from src.analytics import tax as tx
from src.config import (
    LOSS_CARRY_FORWARD_YEARS,
    TAX_BOUNDARY_WARNING_DAYS,
    TAX_DISCLAIMER,
)
from src.ui import components as ui
from src.ui import formatters as fmt

__all__ = ["render"]

#: Session key for the user's lot selection.
_SELECTION = "tax_harvest_selection"


def _lot_key(ticker: str, opened: date) -> str:
    return f"{ticker}@{opened.isoformat()}"


def _prices_from(result: object) -> dict[str, float]:
    """Current native prices per ticker, from the positions table.

    Native, not converted: capital gains are computed in the currency the
    instrument trades in, and these rules are Indian-equity rules anyway.
    """
    positions = getattr(result, "positions", None)
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return {}
    if "ticker" not in positions.columns or "current_price" not in positions.columns:
        return {}
    out: dict[str, float] = {}
    for ticker, price in zip(positions["ticker"], positions["current_price"]):
        try:
            value = float(price)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            out[str(ticker)] = value
    return out


def _regime_banner(plan: tx.HarvestPlan) -> None:
    regime = plan.regime
    st.caption(
        f"**{plan.year_label}** · rules in force: {regime.label} — short-term "
        f"{regime.stcg_pct:.0f}%, long-term {regime.ltcg_pct:.1f}% with a "
        f"₹{regime.ltcg_exemption:,.0f} annual exemption · long-term beyond "
        f"{regime.long_term_days} days · {regime.statute}"
    )


def _kpis(plan: tx.HarvestPlan) -> None:
    position = plan.position
    ui.kpi_row(
        [
            {
                "label": "Realised gains this year",
                "value": fmt.fmt_money(position.stcg + position.ltcg, "INR"),
                "tone": "neutral",
                "footnote": f"ST {fmt.fmt_money(position.stcg, 'INR')} · "
                f"LT {fmt.fmt_money(position.ltcg, 'INR')}",
                "help_text": "Only trades closed inside this financial year. Last "
                "year's profit cannot absorb this year's harvested loss.",
            },
            {
                "label": "Tax before harvesting",
                "value": fmt.fmt_money(plan.before.tax, "INR"),
                "tone": "loss" if plan.before.tax > 0 else "neutral",
                "footnote": f"exemption used {fmt.fmt_money(plan.before.exemption_used, 'INR')}",
            },
            {
                "label": "Tax saved by harvesting",
                "value": fmt.fmt_money(plan.tax_saved, "INR"),
                "tone": "gain" if plan.tax_saved > 0 else "warn",
                "footnote": f"{len(plan.selected)} lot(s) selected",
                "help_text": "The actual reduction in this year's bill, not the loss "
                "multiplied by a rate. Those differ whenever the exemption or the "
                "set-off hierarchy binds.",
            },
            {
                "label": "Carried forward",
                "value": fmt.fmt_money(plan.after.total_carried_forward, "INR"),
                "tone": "accent" if plan.after.total_carried_forward > 0 else "neutral",
                "footnote": f"usable for {LOSS_CARRY_FORWARD_YEARS} assessment years",
                "help_text": "Unabsorbed loss. Only available if the return is filed "
                "by the Section 139(1) due date.",
            },
        ],
        columns=4,
    )


def _headline(plan: tx.HarvestPlan) -> None:
    """The honest verdict, stated before any table."""
    if not plan.candidates:
        st.success(
            "No holding is currently at a loss, so there is nothing to harvest. "
            "That is a good problem to have."
        )
        return

    available = fmt.fmt_money(plan.total_loss_available, "INR")
    if plan.tax_saved > 0:
        st.success(
            f"Harvesting the selected lots would reduce this year's capital-gains tax "
            f"by **{fmt.fmt_money(plan.tax_saved, 'INR')}**, from "
            f"{fmt.fmt_money(plan.before.tax, 'INR')} to "
            f"{fmt.fmt_money(plan.after.tax, 'INR')}. "
            f"{available} of unrealised loss is available in total."
        )
    else:
        st.warning(
            f"{available} of unrealised loss is available, but harvesting it would save "
            "**nothing in tax this year** — there are not enough realised gains of the "
            "right character to absorb it. The loss would instead carry forward for up "
            f"to {LOSS_CARRY_FORWARD_YEARS} assessment years. Weigh that against the "
            "brokerage, STT and spread you would pay today to realise it."
        )

    if plan.after.has_waste:
        st.error(
            f"**{fmt.fmt_money(plan.after.wasted_on_exempt_gains, 'INR')} of loss would "
            "be destroyed for no benefit.** It gets absorbed against long-term gains "
            "the annual exemption already shelters. Set-off is compulsory under "
            "Section 70 — you cannot elect to keep the loss for a better year — so the "
            "only way to avoid this is to harvest in a different financial year."
        )


def _selector(plan: tx.HarvestPlan) -> list[str]:
    """Which lots to harvest. Defaults to those that actually save tax."""
    if not plan.candidates:
        return []

    options = [_lot_key(c.ticker, c.lot.open_date) for c in plan.candidates]
    labels = {
        _lot_key(c.ticker, c.lot.open_date): (
            f"{c.ticker} · {c.lot.open_date:%d %b %Y} · "
            f"{'LT' if c.character == tx.LONG_TERM else 'ST'} · "
            f"loss {fmt.fmt_money(c.loss, 'INR')} · "
            f"saves {fmt.fmt_money(c.tax_saved_now, 'INR')}"
        )
        for c in plan.candidates
    }
    default = [_lot_key(c.ticker, c.lot.open_date) for c in plan.candidates if c.tax_saved_now > 0]

    chosen = st.multiselect(
        "Lots to harvest",
        options,
        default=st.session_state.get(_SELECTION, default),
        key=_SELECTION,
        format_func=lambda key: labels.get(key, key),
        help="Pre-selected: every lot that reduces this year's tax. Lots that save "
        "nothing are left out by default — harvesting them is a legitimate choice, "
        "but it should be a deliberate one.",
    )
    return list(chosen)


def _candidates_table(plan: tx.HarvestPlan) -> None:
    frame = plan.candidates_frame()
    if frame.empty:
        return
    display = frame.rename(
        columns={
            "ticker": "Ticker",
            "open_date": "Bought",
            "quantity": "Qty",
            "days_held": "Days held",
            "character": "Character",
            "cost_basis": "Cost",
            "market_value": "Value",
            "loss": "Loss",
            "tax_saved_now": "Tax saved now",
            "carried_forward": "Carried forward",
            "effective_rate_pct": "Effective rate",
            "days_to_long_term": "Days to LT",
        }
    )
    ui.dataframe(
        display,
        column_config={
            "Qty": st.column_config.NumberColumn(format="%.4f"),
            "Cost": st.column_config.NumberColumn(format="%.2f"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Loss": st.column_config.NumberColumn(format="%.2f"),
            "Tax saved now": st.column_config.NumberColumn(
                format="%.2f",
                help="Marginal reduction in this year's bill from realising this one "
                "lot, given the gains already realised. Zero means the loss cannot be "
                "absorbed this year.",
            ),
            "Carried forward": st.column_config.NumberColumn(format="%.2f"),
            "Effective rate": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Tax saved per rupee of loss. Below the headline rate whenever "
                "the exemption or the set-off hierarchy binds.",
            ),
            "Days to LT": st.column_config.NumberColumn(
                format="%d", help="0 once the lot is already long-term."
            ),
        },
    )


def _boundary(plan: tx.HarvestPlan) -> None:
    alerts = tx.boundary_alerts(plan.lots, plan.regime, TAX_BOUNDARY_WARNING_DAYS)
    if alerts.empty:
        return

    st.markdown("**Lots whose tax character is about to change**")
    st.caption(
        "The two cases point in **opposite** directions, which is the part that trips "
        "people up. For a *gain*, crossing the 12-month boundary lowers the rate — so "
        "wait. For a *loss*, crossing it makes the loss **less** useful, because a "
        "short-term loss offsets both kinds of gain at the higher rate while a "
        "long-term loss offsets only long-term gains. Holding a loss for “long-term "
        "treatment” is the wrong instinct."
    )
    display = alerts.rename(
        columns={
            "ticker": "Ticker",
            "open_date": "Bought",
            "days_held": "Days held",
            "days_to_long_term": "Days to LT",
            "unrealised": "Unrealised",
            "character": "Now",
            "action": "Consider",
            "reason": "Why",
        }
    )
    ui.dataframe(
        display,
        column_config={
            "Unrealised": st.column_config.NumberColumn(format="%.2f"),
            "Days to LT": st.column_config.NumberColumn(format="%d"),
        },
    )


def _set_off_detail(plan: tx.HarvestPlan) -> None:
    with st.expander("The set-off, line by line", expanded=False):
        st.caption(
            "How each loss was applied. **Quantum is mandatory** under Sections 70/71 — "
            "a loss must be set off against available gains of the permitted character "
            "in the same year, and cannot be held back. **Ordering is elective**, so "
            "the beneficial sequence is used: short-term losses go to short-term gains "
            "first (higher rate), long-term losses take first claim on long-term gains "
            "(they can go nowhere else), then any remaining short-term loss follows."
        )
        rows = [
            ("Short-term loss → short-term gain", plan.after.stcl_used_against_stcg),
            ("Long-term loss → long-term gain", plan.after.ltcl_used_against_ltcg),
            ("Short-term loss → long-term gain", plan.after.stcl_used_against_ltcg),
            ("Long-term exemption used", plan.after.exemption_used),
            ("Taxable short-term gain", plan.after.taxable_stcg),
            ("Taxable long-term gain", plan.after.taxable_ltcg),
            ("Short-term loss carried forward", plan.after.stcl_carried_forward),
            ("Long-term loss carried forward", plan.after.ltcl_carried_forward),
            ("Loss wasted on already-exempt gains", plan.after.wasted_on_exempt_gains),
            ("Tax payable", plan.after.tax),
        ]
        ui.dataframe(
            pd.DataFrame(rows, columns=["Step", "Amount (₹)"]),
            column_config={"Amount (₹)": st.column_config.NumberColumn(format="%.2f")},
        )


def _methodology(plan: tx.HarvestPlan) -> None:
    regime = plan.regime
    ui.methodology(
        "The rules this uses, and what it leaves out",
        f"""
**Rates applied — {regime.label}** ({regime.statute})

| | Rate | Notes |
|---|---|---|
| Short-term (≤ {regime.long_term_days} days) | {regime.stcg_pct:.0f}% | Section 111A, STT-paid listed equity |
| Long-term (> {regime.long_term_days} days) | {regime.ltcg_pct:.1f}% | Section 112A |
| Long-term exemption | ₹{regime.ltcg_exemption:,.0f} | per financial year, on net long-term gain |

India changed these mid-year on **23 July 2024**, so a sale on 22 July and one on
24 July of that year are taxed differently. The rules are looked up by date rather
than hard-coded.

**Set-off (Sections 70, 71, 74)**

- Short-term loss → short-term **and** long-term gains
- Long-term loss → long-term gains **only**
- Unabsorbed loss carries forward **{LOSS_CARRY_FORWARD_YEARS} assessment years**, and only
  if the return is filed by the Section 139(1) due date
- Capital losses cannot be set off against salary or business income

**Wash sales.** India has no general wash-sale rule for listed equity, so repurchasing
a harvested holding is not disallowed the way the US 30-day rule disallows it. That is
narrower than "anything goes": Section 94(7) still disallows losses arising from
dividend stripping, and general anti-avoidance provisions apply.

**Grandfathering (Section 112A).** For equity bought on or before 31 January 2018 the
cost of acquisition is the higher of actual cost and the lower of (31-Jan-2018 fair
market value, sale price). The statutory FMV is the **highest quoted price** that day,
which this application does not hold — an adjusted close is neither. Pre-cutoff lots
are therefore flagged, and their gains shown here on **actual cost, which overstates
them**. Take the FMV to your accountant.

**Out of scope entirely**

Surcharge and cess · resident vs non-resident treatment · set-off against other income
heads · unlisted shares, debt, property, gold and every other asset class · STT itself
(not deductible under 111A/112A) · advance-tax timing · the effect of the sale on your
portfolio's risk, which is the Optimiser and Risk pages' business, not this one's.

**{TAX_DISCLAIMER}**
""",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(result: object, ledger: dict | None = None) -> None:
    """Mount the panel against a completed analysis and its FIFO ledger."""
    # `ledger or ...` would be wrong: an empty dict is falsy, so a caller that
    # deliberately passed "no ledger" would have the result's own ledger
    # substituted behind its back. Only a MISSING argument falls back.
    if ledger is None:
        ledger = getattr(result, "ledger", None) or {}
    open_lots = ledger.get("open_lots") or []
    realised = ledger.get("realised") or []

    if not open_lots:
        st.caption(
            "Harvesting analysis needs FIFO tax lots, which come from the trade "
            "ledger. Add transactions on the Portfolio page — average cost alone "
            "cannot tell a short-term lot from a long-term one, and that distinction "
            "is the whole basis of the calculation."
        )
        return

    prices = _prices_from(result)
    if not prices:
        st.caption("No current prices are available for this run.")
        return

    # A first pass establishes the candidate list the selector needs.
    preview = tx.harvest_plan(open_lots, prices, realised)
    _regime_banner(preview)

    selection = _selector(preview)
    plan = tx.harvest_plan(open_lots, prices, realised, select=selection)

    _headline(plan)
    _kpis(plan)
    ui.warning_list(list(plan.warnings), "Data limitations affecting these figures")

    if plan.candidates:
        st.markdown("**Harvestable lots, ranked by tax actually saved**")
        _candidates_table(plan)

    _boundary(plan)
    _set_off_detail(plan)

    for note in plan.notes:
        st.caption(f"ℹ {note}")

    _methodology(plan)
    st.caption(f"⚠️ {TAX_DISCLAIMER}")
