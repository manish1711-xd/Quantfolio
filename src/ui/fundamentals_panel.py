"""Fundamental quality scores for one holding, and the horizons we won't forecast.

Mounted by the Security deep dive page.

Two jobs, and the second one is the reason this panel reads the way it does.

**Show what the accounts say.** Piotroski's F-Score, Altman's Z-Score,
Beneish's M-Score, Sloan accruals and the ratio set — each with its citation,
its threshold and, crucially, **how many of its inputs were actually
available**. A score built from six of nine signals is shown as six of nine,
never as a low nine, because treating a missing signal as a failed one makes a
healthy company look distressed.

**Say plainly which return horizons cannot be answered.** The request was for
predicted returns at 1 month, 1 year, 3, 5, 7, 10 and 11 years. With five
annual statements the arithmetic allows a fit at the short end and *nothing at
all* from five years out — a ten-year model would be fitted to zero
observations. So every horizon asked for is listed, with the number of
training pairs behind it and the reason, rather than quietly dropping the ones
that are impossible or printing a number that came from nowhere.

That table is the feature, not an apology for a missing one. A portfolio tool
that prints a confident ten-year forecast built from no data is worse than one
that explains why it won't.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import fundamentals as fu
from src.ui import components as ui
from src.ui import data_access
from src.ui import formatters as fmt
from src.ui import theme as ui_theme

__all__ = ["render", "score_cards", "horizon_table"]

#: Ratio key -> (label, how to format, one-line meaning).
_RATIO_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("gross_margin", "Gross margin", "pct", "Gross profit over revenue"),
    ("operating_margin", "Operating margin", "pct", "Operating income over revenue"),
    ("net_margin", "Net margin", "pct", "Net income over revenue"),
    ("roa", "Return on assets", "pct", "Net income over average total assets"),
    ("roe", "Return on equity", "pct", "Net income over average shareholders' equity"),
    (
        "roic",
        "Return on invested capital",
        "pct",
        "After-tax operating profit over invested capital",
    ),
    ("asset_turnover", "Asset turnover", "num", "Revenue over average total assets"),
    ("current_ratio", "Current ratio", "num", "Current assets over current liabilities"),
    ("debt_to_equity", "Debt to equity", "num", "Total debt over shareholders' equity"),
    ("debt_to_assets", "Debt to assets", "num", "Total debt over total assets"),
    ("interest_cover", "Interest cover", "num", "EBIT over interest expense"),
    ("cash_conversion", "Cash conversion", "num", "Operating cash flow over net income"),
    ("capex_to_revenue", "Capex to revenue", "pct", "Capital spend as a share of revenue"),
    ("fcf_margin", "Free cash flow margin", "pct", "Free cash flow over revenue"),
    ("effective_tax_rate", "Effective tax rate", "pct", "Tax over pre-tax income"),
)


def _fmt(value: float, kind: str) -> str:
    """A ratio, or an em dash. Never a zero standing in for a missing input."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fmt.DASH
    return fmt.fmt_pct(value * 100.0, 1) if kind == "pct" else fmt.fmt_num(value, 2)


def _tone_for(result: fu.ScoreResult) -> str:
    """A colour for a score, and grey whenever it could not be computed."""
    if not result.ok:
        return "neutral"
    verdict = result.verdict.lower()
    if "strong" in verdict or "safe" in verdict:
        return "gain"
    if "weak" in verdict or "distress" in verdict or "above the -1.78" in verdict:
        return "loss"
    return "neutral"


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def score_cards(report: fu.QualityReport) -> None:
    """The four headline scores, each showing its own input coverage."""
    cards = []
    for result in report.scores:
        if result.ok:
            value = (
                f"{result.value:.0f} / {result.scale}"
                if result.name.startswith("Piotroski")
                else f"{result.value:.2f}{'%' if result.scale == '%' else ''}"
            )
        else:
            value = fmt.DASH
        cards.append(
            {
                "label": result.name,
                "value": value,
                "tone": _tone_for(result),
                "footnote": result.verdict or result.coverage,
            }
        )
    ui.kpi_row(cards, columns=4)


def _piotroski_detail(result: fu.ScoreResult) -> None:
    if not result.signals:
        return
    rows = [
        {
            "Signal": signal.description,
            "Group": signal.group,
            "Result": "pass" if signal.passed else ("fail" if signal.passed is False else "n/a"),
            "Detail": signal.detail or fmt.DASH,
        }
        for signal in result.signals
    ]
    ui.dataframe(pd.DataFrame(rows))
    unavailable = [s for s in result.signals if not s.available]
    if unavailable:
        st.caption(
            f"⚠ {len(unavailable)} of 9 signals could not be computed, so the score is "
            f"out of {result.scale}. A signal without data is **not** counted as a "
            "failure — doing that is what makes a healthy company look distressed."
        )


# ---------------------------------------------------------------------------
# horizons
# ---------------------------------------------------------------------------
def horizon_table(statements: fu.Statements) -> None:
    """Every requested return horizon, and whether it can be answered at all."""
    rows = fu.horizon_availability(statements)
    frame = pd.DataFrame(
        [
            {
                "Horizon": row.horizon,
                "Training pairs": row.pairs,
                "Can be modelled": "yes" if row.available else "no",
                "Why": row.reason,
            }
            for row in rows
        ]
    )
    ui.dataframe(frame)

    modellable = [row.horizon for row in rows if row.available]
    blocked = [row.horizon for row in rows if not row.available]

    if modellable:
        st.caption(
            "Only **" + ", ".join(modellable) + "** have enough history for a fit to "
            "be attempted at all — and even there the statements are restated and "
            "delisted companies are missing, so a fit is explanatory, not predictive."
        )
    if blocked:
        st.warning(
            "**No forecast is offered for " + ", ".join(blocked) + ".** "
            "A model for a horizon of N years needs pairs of (accounts at year T, "
            "return from T to T+N), and each pair consumes N years of history. Five "
            "annual statements leave none at five years and beyond. A number here "
            "would be fitted to zero observations — so the app declines rather than "
            "inventing one."
        )
        st.caption(
            "What would unlock it: a point-in-time fundamentals database with 15+ "
            "years of history and delisted companies retained (Compustat, "
            "CMIE Prowess, Refinitiv). That is a paid data problem, not a modelling "
            "one — no amount of model choice creates observations that do not exist."
        )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(
    ticker: str,
    market_cap: float | None = None,
    country: str = "",
    price: float | None = None,
) -> None:
    """Draw the panel for one instrument. Fetches (cached) statements itself.

    ``market_cap`` is the provider's figure where it has one. When it does
    not, and only then, it is derived from the shares on the balance sheet
    times ``price`` -- both native to the instrument, so Altman's X4 stays
    dimensionally consistent with the liabilities it divides into. If neither
    is available the Z-Score reports itself unavailable rather than falling
    back to book equity.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return

    with st.spinner(f"Fetching {symbol} financial statements…"):
        frames, notice = data_access.load_statements(symbol)

    statements = fu.read_statements(
        frames.get("balance_sheet"),
        frames.get("income"),
        frames.get("cash_flow"),
        ticker=symbol,
    )

    if not statements.ok:
        ui.empty_state(
            "No financial statements for this instrument",
            notice
            or "The data provider publishes no balance sheet, income statement or "
            "cash-flow statement for this symbol. Common for ETFs, indices and "
            "recently listed companies.",
            icon="📄",
        )
        return

    if notice:
        st.caption(f"⚠ {notice}")

    cap = market_cap
    derived = False
    if cap is None or (isinstance(cap, float) and math.isnan(cap)):
        shares = statements.latest.get("shares") if statements.latest else float("nan")
        spot = float(price) if price is not None else float("nan")
        if shares == shares and shares > 0 and spot == spot and spot > 0:
            cap = shares * spot
            derived = True

    report = fu.quality_summary(statements, cap, country=country)
    st.caption(
        f"{statements.n_periods} annual period(s), {statements.span}. "
        "Figures are the provider's **restated** statements — today's view of each "
        "year, including later restatements."
    )

    if derived:
        st.caption(
            "The provider published no market capitalisation, so the Z-Score's "
            "market-equity term uses shares on the balance sheet times the last "
            "price."
        )

    score_cards(report)

    ui_theme.spacer(0.4)
    balance_sheet_section(statements)

    if report.unavailable:
        st.info(
            "Not computable for this company: **"
            + "**, **".join(report.unavailable)
            + "**. Each says why in its own section below — a score is reported as "
            "unavailable rather than as a low number."
        )

    ui_theme.spacer(0.3)
    with st.expander("Piotroski F-Score — the nine signals", expanded=False):
        st.caption(
            "Piotroski (2000). Nine binary tests across profitability, "
            "leverage/liquidity and efficiency. 8–9 strong, 0–2 weak. Five of the "
            "nine compare against the prior year, so a company with one year of "
            "statements can score at most four."
        )
        _piotroski_detail(report.piotroski)

    with st.expander("Altman Z-Score and Beneish M-Score", expanded=False):
        for result in (report.altman, report.beneish):
            st.markdown(f"**{result.headline}**")
            st.caption(result.coverage)
            terms = {name: _fmt(value, "num") for name, value in result.inputs.items()}
            if terms:
                ui.dataframe(pd.DataFrame([terms]))
            for line in result.notices:
                st.caption(f"⚠ {line}")
            ui_theme.spacer(0.2)

    with st.expander("Ratios", expanded=False):
        ratio_set = report.ratios
        ui.dataframe(
            pd.DataFrame(
                [
                    {
                        "Ratio": label,
                        "Value": _fmt(ratio_set.get(key), kind),
                        "Definition": meaning,
                    }
                    for key, label, kind, meaning in _RATIO_ROWS
                ]
            )
        )
        for line in ratio_set.notices:
            st.caption(f"⚠ {line}")

    ui_theme.spacer(0.4)
    ui_theme.section("RETURN HORIZONS", "\U0001f52e")
    st.caption("What can and cannot be modelled from the statements available, and why.")
    horizon_table(statements)

    ui.methodology(
        "Why these are scores and not forecasts",
        """
**Every number above measures what the accounts say.** Each score is a
published rule with a citation and a threshold from its original paper —
Piotroski (2000), Altman (1968, and 2005 for the emerging-market variant),
Beneish (1999), Sloan (1996). None of them is a prediction, and none of them
becomes one by being displayed next to a price.

**A score is reported as incomplete, never as low.** The F-Score is out of
however many of its nine signals could be computed; the Z-Score and M-Score
are withheld entirely if any term is missing, because their published
coefficients are only meaningful together. Interest cover on a company that
reports no interest expense is *unavailable*, not infinite.

**The Altman variant is chosen from the issuer's market.** Z″ drops the
asset-turnover term that makes a software company look distressed and a
retailer look safe on business model alone; the +3.25 emerging-market
adjustment is applied to Indian issuers and withheld from developed-market
ones, because applying it everywhere inflates the score against thresholds it
was never calibrated for. The variant used is printed with the score.

**Restated, not point-in-time.** A figure labelled 2022 is today's view of
2022. That is the right basis for "how do these accounts look now" and the
wrong basis for backtesting — which is exactly why the horizon table above
refuses the long horizons instead of fitting them.

**Not investment advice.** A high F-Score is a description of past accounts,
not a reason to buy, and a Beneish score above the threshold is a prompt to
read the filings, not an accusation.
""",
    )


# ---------------------------------------------------------------------------
# the balance sheet itself
# ---------------------------------------------------------------------------
#: Columns of the statement frame that are metadata rather than a period.
_STATEMENT_META: tuple[str, ...] = ("section", "line", "field", "level", "is_total", "source")

#: Verdict tone and headline per :func:`fundamentals.balance_sheet_check`
#: diagnosis. A stable positive gap is NOT an error, so it does not get the
#: caution tone that an erratic one does.
_CHECK_TONE: dict[str, tuple[str, str]] = {
    "balanced": ("pass", "The extract balances"),
    "minority_interest": ("neutral", "The gap is minority interest"),
    "inconsistent": ("caution", "This extract does not balance"),
    "unavailable": ("caution", "The identity could not be tested"),
}


def _statement_table(statements: Any, currency: str) -> pd.DataFrame:
    """Format the statement frame for display: indented lines, dashes for gaps.

    Compact magnitudes rather than full figures -- an Indian balance sheet runs
    to twelve digits and a column of those is unreadable at a glance, so this
    renders lakh/crore for INR and K/M/B otherwise.

    Iterates with ``iterrows`` rather than ``itertuples`` deliberately: the
    period columns are dates like ``2026-03-31``, which are not valid Python
    identifiers, and ``itertuples`` silently renames them to ``_6``, ``_7`` and
    so on.
    """
    frame = fu.balance_sheet_statement(statements)
    if frame.empty:
        return frame

    periods = [c for c in frame.columns if c not in _STATEMENT_META]
    columns: dict[str, list[str]] = {"Line item": []}
    for column in periods:
        columns[column] = []

    for _, row in frame.iterrows():
        indent = "   " * int(row["level"])
        marker = "  · derived" if row["source"] == "derived" else ""
        columns["Line item"].append(f"{indent}{row['line']}{marker}")
        counted = row["field"] in fu.BALANCE_SHEET_COUNT_FIELDS
        for column in periods:
            value = row[column]
            columns[column].append(
                _count(value, currency) if counted else fmt.fmt_compact(value, currency)
            )

    return pd.DataFrame(columns)


def _count(value: float, currency: str) -> str:
    """A share count: the reader's own magnitude ladder, with no currency symbol.

    A share count is not money, and "₹1,353 Cr" for a share count is a category
    error the reader has to undo on every glance. But an Indian reader still
    expects crore rather than billions, and that ladder is chosen by currency
    inside ``fmt_compact`` -- so the symbol is stripped afterwards rather than
    the currency being dropped, which would silently switch the scale to K/M/B
    as well.
    """
    symbol = fmt.currency_symbol(currency)
    text = fmt.fmt_compact(value, currency)
    return text[len(symbol) :] if symbol and text.startswith(symbol) else text


def balance_sheet_section(statements: Any) -> None:
    """The balance sheet as a statement, plus whether the extract balances.

    The statement is what was asked for, but the identity check is what makes
    it worth reading: assets must equal liabilities plus equity, and an extract
    that fails that is an extract whose quality scores should not be trusted
    either. So the verdict goes above the table rather than under it.
    """
    ui_theme.section("BALANCE SHEET", "\U0001f4d0")

    currency = str(getattr(statements, "currency", "") or "INR")
    check = fu.balance_sheet_check(statements)
    tone, headline = _CHECK_TONE.get(
        str(check.get("diagnosis", "unavailable")), _CHECK_TONE["unavailable"]
    )
    ui.verdict(
        headline,
        str(check.get("reason", "")),
        tone=tone,
        evidence=(
            f"assets − (liabilities + equity): {check.get('n_balanced', 0)} of "
            f"{check.get('n_checked', 0)} periods within "
            f"{fu.BALANCE_SHEET_TOLERANCE:.0%} of total assets"
        ),
    )

    table = _statement_table(statements, currency)
    if table.empty:
        ui.empty_state(
            "No balance sheet to show",
            "The provider returned no period carrying balance-sheet lines.",
            icon="\U0001f4c4",
        )
        return

    ui.dataframe(table, height=min(80 + 34 * len(table), 680))
    st.caption(
        f"Newest period first, in {currency}. A line the provider did not supply shows as "
        f"— and **the row is kept**, so you can tell 'this company reports no "
        f"inventory' from 'we could not find the inventory row'. Figures are compacted. "
        f"The only computed line is working capital, marked *derived* on the periods "
        f"where the provider omitted it and it was reconstructed from current assets "
        f"minus current liabilities."
    )

    rows = check.get("rows")
    if isinstance(rows, pd.DataFrame) and not rows.empty:
        ui_theme.spacer(0.3)
        ui.chart(ch.balance_sheet_chart(rows, fmt.currency_symbol(currency)))

    with st.expander("Why a balance sheet might not balance"):
        st.markdown(
            f"""
A balance sheet balances **exactly** in the filing. The provider's extract often
does not, and the reason is almost always **minority interest**.

The two figures this app reads are *Total Liabilities Net Minority Interest* and
*Stockholders Equity*. Both exclude the stake outside shareholders hold in
consolidated subsidiaries — so that stake falls out of the middle of the
identity, and assets come out higher than the two of them together.

That is why the check classifies rather than simply passing or failing:

| Diagnosis | What it means |
|---|---|
| **Balanced** | Assets equal liabilities plus equity to within {fu.BALANCE_SHEET_TOLERANCE:.0%} of assets, in every period |
| **Minority interest** | A gap that is positive in *every* period and stable in size. Structural, expected in a holding company, and not a data problem |
| **Inconsistent** | A gap that changes sign, or appears in some periods and not others. Nothing structural explains that, so treat this extract — and every score built on it — with suspicion |

Measured on this app's own universe: **RELIANCE.NS** shows a steady +7.0% to
+8.5% gap across four years, which is Jio Platforms and Reliance Retail
Ventures carrying large non-controlling interests. **TCS.NS** balances to within
0.7% and **AAPL** to 0.0%.

**Restated, not point-in-time.** Every figure above is today's view of that year,
including any restatement published since — correct for reading the accounts,
wrong for backtesting.
"""
        )
