"""Stress testing - what this book has lost, and what it could lose.

Three different machines answer three different questions, and the page's job
is to keep them apart:

* **Crisis replay.** Today's weights dropped into a documented crisis window,
  compounded through each holding's own price history. Nothing is modelled,
  but a portfolio built in 2024 has no 2008 in it - so every row carries a
  ``coverage`` fraction and the reader is never handed a partial replay
  without it.
* **Benchmark shock ladder.** First-order CAPM: an index move times beta. One
  line of algebra and a long list of reasons to distrust it.
* **Worst realised windows.** The worst holding periods this exact value
  series actually lived through. Measurement, not modelling - and bounded by
  the window, because the worst thing that happened is not the worst thing
  that can happen.

This is the one page permitted to fetch its own data. The analysis window is
typically a year; replaying the global financial crisis needs decades, and
loading decades on every page would be slow and would silently change the risk
window the user chose. So the long panel sits behind a button, is cached in
session state, and its provenance is stated wherever a number depends on it.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import stress as strs
from src.config import FACTOR_SHOCKS, HISTORICAL_STRESS, PERIOD_LABELS
from src.ui import components as ui
from src.ui import (
    data_access,  # the one page allowed its own fetch - see docstring
    guards,
)
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    delta_tone,
    fmt_date,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
)

#: Session key holding the long-history panel, so a rerun does not refetch.
#: The cached entry records the ticker tuple it was fetched for, so editing the
#: portfolio invalidates it instead of silently replaying the wrong book.
_HISTORY_KEY = "stress_long_history"

#: Holding periods offered to the worst-window scan, in trading sessions.
_WINDOW_CHOICES: tuple[int, ...] = (5, 10, 21, 63)
_WINDOW_LABELS: dict[int, str] = {
    5: "5 sessions (one week)",
    10: "10 sessions (two weeks)",
    21: "21 sessions (one month)",
    63: "63 sessions (one quarter)",
}
_DEFAULT_WINDOW: int = 21

#: How many non-overlapping bad stretches to list.
_TOP_N: int = 5

#: Benchmark shocks promoted from the ladder onto KPI cards, in percent.
_HEADLINE_SHOCKS: tuple[float, ...] = (-10.0, -20.0)

#: Coverage at or above this counts as a full replay (float tolerance only).
_FULL_COVERAGE: float = 0.999


# ---------------------------------------------------------------------------
# small display helpers (no analytics happen in this module)
# ---------------------------------------------------------------------------
def _finite(value: Any) -> float | None:
    """Float when the value is a real, finite number; ``None`` otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    """Stringify a possibly-missing label, collapsing blanks to an em dash."""
    if value is None:
        return DASH
    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return DASH
    except (TypeError, ValueError):
        pass
    label = str(value).strip()
    return label if label and label.upper() not in {"NAN", "NONE", "NAT"} else DASH


def _date_text(stamp: Any) -> str:
    """``fmt_date`` that survives ``NaT`` - which raises on ``strftime``."""
    if stamp is None:
        return DASH
    try:
        if bool(pd.isna(stamp)):
            return DASH
    except (TypeError, ValueError):
        pass
    return fmt_date(stamp)


def _weights(positions: pd.DataFrame) -> dict[str, float]:
    """Ticker -> weight FRACTION of current market value."""
    weights = pd.to_numeric(positions.get("weight_pct"), errors="coerce")
    return {
        str(ticker): float(weight) / 100.0
        for ticker, weight in zip(positions["ticker"], weights)
        if pd.notna(weight)
    }


def _panel_reach(positions: pd.DataFrame, prices: pd.DataFrame) -> tuple[float, int, int]:
    """How much of today's book is actually in the fetched panel.

    Returns ``(weight percent present, names present, names held)``. This is a
    provenance check, not an analytic: a symbol whose download failed is absent
    from the panel entirely, and the replay's own ``coverage`` column is then
    measured against the survivors rather than against the whole portfolio.
    """
    held = len(positions)
    columns = {str(c) for c in getattr(prices, "columns", [])}
    present = positions["ticker"].astype(str).isin(columns)
    weight = pd.to_numeric(positions.loc[present, "weight_pct"], errors="coerce").sum()
    return float(weight), int(present.sum()), int(held)


def _basis_label(coverage: float) -> str:
    """One-word verdict on how much of the panel a scenario row rests on."""
    if not math.isfinite(coverage) or coverage <= 0.0:
        return "No overlap"
    return "Full panel" if coverage >= _FULL_COVERAGE else "Partial sleeve"


def _shock_row(frame: pd.DataFrame, shock_pct: float) -> pd.Series | None:
    """The ladder row for a given benchmark move, matched on the shock itself."""
    if frame is None or frame.empty or "shock_pct" not in frame.columns:
        return None
    shocks = pd.to_numeric(frame["shock_pct"], errors="coerce")
    match = frame.loc[(shocks - shock_pct).abs() < 1e-9]
    return None if match.empty else match.iloc[0]


# ---------------------------------------------------------------------------
# 1. the long-history panel
# ---------------------------------------------------------------------------
def _cached_panel(tickers: tuple[str, ...]) -> tuple[Any, dict[str, str]] | None:
    """The panel already fetched for exactly these symbols, or ``None``."""
    cached = st.session_state.get(_HISTORY_KEY)
    if not isinstance(cached, dict):
        return None
    if tuple(cached.get("tickers") or ()) != tickers:
        return None
    return cached.get("prices"), dict(cached.get("errors") or {})


def _panel_gate(result: Any, cfg: Any, tickers: tuple[str, ...]) -> tuple[Any, dict[str, str]]:
    """Render the loader, or hand back the panel already in session state."""
    cached = _cached_panel(tickers)
    if cached is not None:
        return cached

    window = PERIOD_LABELS.get(getattr(cfg, "period", ""), str(getattr(cfg, "period", "")))
    ui.empty_state(
        "Scenario replay needs its own, much longer fetch",
        f"The analysis window is {window} ({result.n_days} sessions). It does not "
        f"contain 2008, and scoring a crisis against history the portfolio never saw "
        f"would be a fabricated number. Pulling the maximum available daily history "
        f"for these {len(tickers)} symbols is a separate, slower download, so it sits "
        f"on a button instead of on every page load.",
        icon="\U000023f3",
        hint="It is fetched once and kept for this session. The shock ladder and the "
        "worst realised windows below need no extra data and are already computed.",
    )
    if st.button(
        "Load full history for scenario replay",
        type="primary",
        key="stress_load_history",
        width="stretch",
    ):
        with ui.loading(f"Downloading maximum available daily history for {len(tickers)} symbols"):
            prices, errors = data_access.load_long_history(tickers)
        st.session_state[_HISTORY_KEY] = {
            "tickers": tickers,
            "prices": prices,
            "errors": errors,
        }
        st.rerun()
    return None, {}


def _refetch_button(key: str, label: str = "Refetch panel") -> None:
    """Drop the cached panel and the cached download, then reload the page."""
    if st.button(label, key=key, width="stretch"):
        st.session_state.pop(_HISTORY_KEY, None)
        data_access.load_long_history.clear()
        st.rerun()


def _panel_provenance(
    positions: pd.DataFrame, prices: pd.DataFrame, errors: dict[str, str]
) -> tuple[float, int, int]:
    """Panel span, reach into the book, refetch control and download errors."""
    weight_pct, present, held = _panel_reach(positions, prices)
    index = getattr(prices, "index", None)
    span = (
        f"{len(prices):,} sessions from {_date_text(index[0])} to {_date_text(index[-1])}"
        if index is not None and len(index)
        else "no usable sessions"
    )

    left, right = st.columns([4, 1], gap="small")
    left.caption(
        f"**Replay panel:** {span}. {present} of {held} holdings are in it, carrying "
        f"{fmt_pct(weight_pct)} of today's portfolio weight."
    )
    with right:
        _refetch_button("stress_refetch_panel")

    ui.warning_list(
        [f"**{ticker}** - {message}" for ticker, message in sorted(errors.items())],
        "Symbols missing from the replay panel",
    )
    if present < held:
        st.warning(
            f"{held - present} of {held} holdings returned no long history, so they are "
            f"not in the replay at all. Every coverage figure below is measured against "
            f"the {fmt_pct(weight_pct)} of weight that is in the panel, not against the "
            f"whole book - the missing names are unknown in every scenario, not unharmed."
        )
    return weight_pct, present, held


# ---------------------------------------------------------------------------
# 2. historical crisis replay
# ---------------------------------------------------------------------------
def _replay_kpis(
    frame: pd.DataFrame, result: Any, cfg: Any, weight_pct: float, present: int, held: int
) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    impacts = pd.to_numeric(frame["portfolio_return_pct"], errors="coerce")
    coverage = pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.0)
    evaluated = int(impacts.notna().sum())
    mdd = _finite((result.risk or {}).get("mdd"))

    worst_card: dict[str, Any] = {
        "label": "Worst scenario replayed",
        "value": DASH,
        "tone": "muted",
        "footnote": "No scenario window overlaps the panel",
        "help_text": "Covered-sleeve return of the harshest window that could be replayed.",
    }
    if evaluated:
        position = impacts.idxmin()
        row = frame.loc[position]
        cover = float(coverage.loc[position])
        money = _finite(row.get("value_change"))
        worst_card = {
            "label": "Worst scenario replayed",
            "value": fmt_pct(row["portfolio_return_pct"]),
            "tone": delta_tone(row["portfolio_return_pct"]),
            "delta": f"{fmt_ratio_pct(cover, 0)} of panel weight covered",
            "delta_tone": "muted" if cover >= _FULL_COVERAGE else "warn",
            "footnote": (
                f"{_text(row.get('scenario'))}, "
                + (
                    f"{fmt_money(money, ccy, 0, signed=True)} on today's book"
                    if money is not None
                    else "no portfolio value to price it in"
                )
            ),
            "help_text": "The deepest replayed loss, and how much of the panel it rests on.",
        }

    ui.kpi_row(
        [
            worst_card,
            {
                "label": "Scenarios evaluated",
                "value": f"{evaluated} of {len(frame)}",
                "tone": "neutral" if evaluated == len(frame) else "warn",
                "footnote": (
                    f"{len(frame) - evaluated} predate the panel and report no number"
                    if evaluated < len(frame)
                    else "Every window overlaps the panel"
                ),
                "help_text": "A window with zero coverage is kept and left blank, never "
                "scored as a flat 0%.",
            },
            {
                "label": "Panel reach",
                "value": fmt_pct(weight_pct),
                "tone": "neutral" if present == held else "warn",
                "footnote": f"{present} of {held} holdings have long history",
                "help_text": "Share of today's portfolio weight present in the fetched panel.",
            },
            {
                "label": "Realised max drawdown",
                "value": fmt_ratio_pct(mdd),
                "tone": "neutral",
                "footnote": f"Actually lived through in the {result.n_days}-session window",
                "help_text": "The reality check: what this book has already done to itself.",
            },
        ],
        columns=4,
    )

    if not evaluated:
        st.caption(
            f"**Not one of the {len(frame)} documented windows overlaps this panel.** Every "
            f"crisis predates the price history available for these holdings, so no "
            f"replay is possible and none is invented. That is itself the finding: this "
            f"book has never been through one of these, and nothing on this page can say "
            f"how it would behave in the next. The realised-window scan further down is "
            f"the only empirical answer this data supports."
        )
        return

    if mdd is None:
        return

    worst = float(impacts.min())
    thin = int((coverage < _FULL_COVERAGE).sum())
    if worst <= mdd * 100.0:
        reading = (
            "The replay is therefore telling you something the recent window does not - "
            "these holdings have a worse regime in their own history than anything they "
            "have shown you lately."
        )
    elif thin:
        reading = (
            f"Before reading that as resilience: {thin} of {len(frame)} windows are only "
            f"partly covered or could not be evaluated at all. The crises that would hurt "
            f"most are usually the ones your holdings are too young to have been in, and "
            f"a stress table full of gaps flatters the book."
        )
    elif present < held:
        reading = (
            f"Every window in the panel was fully covered - but the panel is only "
            f"{fmt_pct(weight_pct)} of the book. The {held - present} holdings missing "
            f"from it are absent from every scenario above, so the table is milder than "
            f"the portfolio it claims to describe."
        )
    else:
        reading = (
            "Every window was fully covered and the panel holds the whole book, so this "
            "is a finding rather than a data gap: what has hurt this portfolio hardest "
            "is inside the recent window, not in the historical scenarios. Old crises "
            "hit the sectors of their own day, and this is not that portfolio."
        )
    st.caption(
        f"Read that last pair together. The harshest scenario this panel can replay is "
        f"{fmt_pct(worst)} against a {fmt_ratio_pct(mdd)} drawdown the portfolio has "
        f"actually lived through. {reading}"
    )


def _replay_table(frame: pd.DataFrame, cfg: Any) -> None:
    symbol = getattr(cfg, "currency_symbol", "")
    coverage = pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.0)

    display = pd.DataFrame(
        {
            "Scenario": frame["scenario"].astype(str),
            "Coverage": coverage * 100.0,
            "Basis": [_basis_label(float(c)) for c in coverage],
            "From": frame["start"],
            "To": frame["end"],
            "Sessions": pd.to_numeric(frame["days"], errors="coerce"),
            "Portfolio impact": pd.to_numeric(frame["portfolio_return_pct"], errors="coerce"),
            "On today's book": pd.to_numeric(frame["value_change"], errors="coerce"),
            "Worst holding": [_text(v) for v in frame["worst_asset"]],
            "Its return": pd.to_numeric(frame["worst_asset_pct"], errors="coerce"),
        }
    ).sort_values("Portfolio impact", na_position="last")

    ui.dataframe(
        display,
        height=min(38 * (len(display) + 1) + 10, 420),
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium", pinned=True),
            "Coverage": st.column_config.ProgressColumn(
                "Coverage",
                format="%.0f%%",
                min_value=0.0,
                max_value=100.0,
                help="Share of panel weight that genuinely had prices inside this "
                "window. The impact beside it describes that sleeve alone.",
            ),
            "Basis": st.column_config.TextColumn(
                "Basis",
                width="small",
                help="Full panel, a partial sleeve, or no overlap at all.",
            ),
            "From": st.column_config.DateColumn(
                "From",
                format="DD MMM YYYY",
                width="small",
                help="The first date actually replayed, not the scenario's nominal start.",
            ),
            "To": st.column_config.DateColumn("To", format="DD MMM YYYY", width="small"),
            "Sessions": st.column_config.NumberColumn(
                "Sessions",
                format="%d",
                width="small",
                help="Trading observations used in the replay.",
            ),
            "Portfolio impact": st.column_config.NumberColumn(
                "Portfolio impact",
                format="%.2f%%",
                help="Weight-renormalised return of the covered names over the window.",
            ),
            "On today's book": st.column_config.NumberColumn(
                "On today's book",
                format=f"{symbol}%.0f",
                help="That percentage applied to the current market value.",
            ),
            "Worst holding": st.column_config.TextColumn("Worst holding", width="small"),
            "Its return": st.column_config.NumberColumn("Its return", format="%.2f%%"),
        },
    )
    st.caption(
        "Sorted worst first; blank cells are windows with nothing to measure, never "
        "zeros. **Coverage is the column to read first.** Coverage 62% means only 62% "
        "of the panel's weight existed during that window, so the impact beside it is "
        "the return of that 62% sleeve renormalised to sum to one - the other 38% is "
        "unknown, not unharmed. Coverage 0% means the scenario cannot be evaluated at "
        "all, and the row is kept blank rather than dropped, because a silently missing "
        "row reads as a crisis that did no damage."
    )


def _replay_notes(frame: pd.DataFrame) -> None:
    """Every scenario's context and its coverage caveat, verbatim from analytics."""
    lines: list[str] = []
    for _, row in frame.iterrows():
        window = f"{_date_text(row.get('start'))} to {_date_text(row.get('end'))}"
        note = _text(row.get("note"))
        lines.append(
            f"- **{_text(row.get('scenario'))}** ({window}) - "
            + (note if note != DASH else "no context recorded for this window.")
        )
    with st.expander(f"Scenario definitions and coverage notes ({len(frame)})"):
        st.markdown("\n".join(lines))


def _replay(result: Any, cfg: Any) -> None:
    ui_theme.section("Crisis replay", "\U0001f4c9")

    positions = result.positions
    tickers = tuple(str(t) for t in positions["ticker"])
    prices, errors = _panel_gate(result, cfg, tickers)
    if prices is None:
        return

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        ui.empty_state(
            "The long-history fetch came back empty",
            "No symbol in this portfolio returned usable historical prices, so there is "
            "nothing to replay. Nothing is estimated in its place.",
            icon="\U0001f4ed",
            hint="Check the ticker symbols on the Portfolio page - Indian equities need "
            "the .NS or .BO suffix - then refetch.",
        )
        ui.warning_list(
            [f"**{ticker}** - {message}" for ticker, message in sorted(errors.items())],
            "Download errors",
        )
        _refetch_button("stress_refetch_empty", "Clear and try the fetch again")
        return

    weight_pct, present, held = _panel_provenance(positions, prices, errors)

    value = _finite((result.summary or {}).get("current_value"))
    frame = strs.historical_stress(
        prices,
        _weights(positions),
        HISTORICAL_STRESS,
        portfolio_value=value if value is not None else float("nan"),
    )

    _replay_kpis(frame, result, cfg, weight_pct, present, held)
    ui.chart(ch.stress_bar(frame, currency_symbol=getattr(cfg, "currency_symbol", "")))
    st.caption(
        "Each bar is today's weights compounded through that window's actual prices. "
        "Scenarios that could only be partly covered are labelled with their coverage "
        "on the axis; scenarios with no overlap are absent from the chart entirely and "
        "appear blank in the table below rather than as a harmless zero."
    )

    _replay_table(frame, cfg)
    _replay_notes(frame)

    ui.methodology(
        "Crisis replay - the arithmetic, and why coverage is the whole story",
        f"""
For each window, every holding's return is its own first-to-last valid close
inside the dates that actually overlap, `r_i = P_last / P_first - 1`, and the
portfolio figure is the weight-renormalised return of the names that had data:

```
coverage = sum(w_i)  over covered names i          (fraction of panel weight)
impact   = sum(w_i * r_i) / coverage               (the covered sleeve's return)
```

Dividing by coverage is the point of the whole page. The naive alternative -
treating a name with no history as flat - quietly converts *"no idea what this
did in the GFC"* into *"this did nothing in the GFC"*, and every fabricated 0%
makes the portfolio look more robust than it is.

**A structural limit of this particular panel.** The loader aligns all symbols
onto the dates they *all* traded and trims to that common window, which is what
makes correlations and betas fair elsewhere in the app. The side effect here is
that coverage tends to come out all-or-nothing: a window starting before your
youngest holding listed is reported as no overlap for the whole panel, rather
than as a partial sleeve. So the scenarios that do evaluate are replayed on the
full panel, and the ones that do not are genuinely unanswerable with this data.
This panel reaches {fmt_pct(weight_pct)} of today's weight
({present} of {held} holdings).

**What the replay still assumes.** Today's weights, held constant, dropped into
an old regime. It ignores what you actually owned then, any rebalancing you
would have done, dividends beyond those already in the adjusted closes, and
liquidity - in a real crisis the exit price is not the close. It also assumes
the crisis is a fair template for the next one, which is the assumption every
historical stress test makes and none can defend.

**Currency.** Impacts are computed on each instrument's own price series and
reported in {getattr(cfg, "base_currency", "the base currency")} at today's FX
rate. A crisis that moved the exchange rate as well is therefore understated
for foreign holdings.
        """,
    )


# ---------------------------------------------------------------------------
# 3. beta-implied benchmark shocks
# ---------------------------------------------------------------------------
def _beta_kpis(frame: pd.DataFrame, cfg: Any, beta: float, value: float | None) -> None:
    ccy = getattr(cfg, "base_currency", "INR")
    name = getattr(cfg, "benchmark_name", "benchmark")

    cards: list[dict[str, Any]] = [
        {
            "label": "Current value",
            "value": fmt_money(value, ccy, 0) if value is not None else DASH,
            "tone": "accent",
            "footnote": "The book every projection below is applied to",
        },
        {
            "label": f"Beta vs {name}",
            "value": fmt_num(beta, 2),
            "tone": "warn" if beta >= 1.0 else "neutral",
            "footnote": f"A 1% index move implies {fmt_pct(beta, 2, signed=True)} here",
            "help_text": "cov(portfolio, benchmark) / var(benchmark) over the analysis window.",
        },
    ]

    for shock in _HEADLINE_SHOCKS:
        row = _shock_row(frame, shock)
        implied = None if row is None else _finite(row.get("implied_portfolio_pct"))
        projected = None if row is None else _finite(row.get("projected_value"))
        change = None if row is None else _finite(row.get("value_change"))
        cards.append(
            {
                "label": f"{name} {fmt_pct(shock, 0, signed=True)}",
                "value": fmt_money(projected, ccy, 0) if projected is not None else DASH,
                "tone": delta_tone(implied),
                "delta": fmt_pct(implied, 2, signed=True) if implied is not None else DASH,
                "delta_tone": delta_tone(implied),
                "footnote": (
                    f"{fmt_money(change, ccy, 0, signed=True)} from here"
                    if change is not None
                    else "No portfolio value to project"
                ),
                "help_text": "Projected portfolio value = value x (1 + beta x shock).",
            }
        )

    ui.kpi_row(cards, columns=4)


def _beta_table(frame: pd.DataFrame, cfg: Any) -> None:
    symbol = getattr(cfg, "currency_symbol", "")
    display = pd.DataFrame(
        {
            "Scenario": frame["scenario"].astype(str),
            "Benchmark move": pd.to_numeric(frame["shock_pct"], errors="coerce"),
            "Implied portfolio move": pd.to_numeric(
                frame["implied_portfolio_pct"], errors="coerce"
            ),
            "Change in value": pd.to_numeric(frame["value_change"], errors="coerce"),
            "Projected value": pd.to_numeric(frame["projected_value"], errors="coerce"),
        }
    )
    ui.dataframe(
        display,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="small"),
            "Benchmark move": st.column_config.NumberColumn(
                "Benchmark move", format="%.1f%%", help="The assumed index move."
            ),
            "Implied portfolio move": st.column_config.NumberColumn(
                "Implied portfolio move", format="%.2f%%", help="beta x shock."
            ),
            "Change in value": st.column_config.NumberColumn(
                "Change in value", format=f"{symbol}%.0f"
            ),
            "Projected value": st.column_config.NumberColumn(
                "Projected value", format=f"{symbol}%.0f"
            ),
        },
    )


def _beta_shocks(result: Any, cfg: Any) -> None:
    ui_theme.section("Benchmark shock ladder", "\U000026a1")

    bench = result.benchmark or {}
    beta = _finite(bench.get("beta"))
    value = _finite((result.summary or {}).get("current_value"))
    ccy = getattr(cfg, "base_currency", "INR")
    name = getattr(cfg, "benchmark_name", "the benchmark")

    if beta is None:
        reason = bench.get("reason") or "the benchmark series could not be aligned"
        ui.empty_state(
            "No beta, so nothing to shock",
            f"This view maps a {name} move onto the portfolio by multiplying it by "
            f"beta, and beta could not be estimated on this run: {reason}. Assuming "
            f"1.0 would quietly turn 'unknown' into 'moves exactly with the index', so "
            f"nothing is projected instead.",
            icon="\U0001f9ee",
            hint="Choose a benchmark that overlaps your holdings' history in the "
            "sidebar, then re-run. The crisis replay above needs no benchmark.",
        )
        return

    frame = strs.beta_shock(value if value is not None else float("nan"), beta, FACTOR_SHOCKS)
    _beta_kpis(frame, cfg, beta, value)

    r_squared = _finite(bench.get("r_squared"))
    n_days = int(bench.get("n_days", 0) or 0)
    st.caption(
        f"Beta {fmt_num(beta, 2)} was measured on {n_days} sessions where the portfolio "
        f"and {name} both traded"
        + (
            f", and the index explains {fmt_ratio_pct(r_squared)} of this book's daily "
            f"variance. The remaining {fmt_ratio_pct(1.0 - r_squared)} is idiosyncratic "
            f"- stock-specific movement this ladder cannot see at all."
            if r_squared is not None
            else ". Read it next to R-squared on the Benchmark page: a low R-squared "
            "means most of the movement is idiosyncratic and beta describes the "
            "book poorly."
        )
    )

    _beta_table(frame, cfg)

    st.caption(
        "One line of algebra: `implied move = beta x benchmark move`, and the projected "
        "value is today's book multiplied by one plus that. Everything else about the "
        "portfolio - which names, which sectors, how crowded the positioning - is "
        "assumed away."
    )
    _beta_convergence_caption(frame, beta, value, ccy, name)

    ui.methodology(
        "Beta shock - the whole model, and four reasons to distrust it",
        f"""
```
implied portfolio return = beta x benchmark shock
change in value          = beta x benchmark shock x portfolio value
projected value          = portfolio value + change in value
```

That is the entire model. Its limitations matter more than its arithmetic:

* **First order only - idiosyncratic risk is assumed away.** Beta describes the
  part of this portfolio that moves *with* {name}. A single-name blow-up,
  a fraud, a failed drug trial or a regulatory hit has nothing to do with the
  index and does not appear anywhere in this table. On a concentrated book that
  is the larger risk, and this ladder is silent about it.
* **Beta is not stable in a crisis, and that biases the table one way.**
  Cross-sectional correlations converge toward one when everything is sold at
  once, so a low-beta book routinely trades like a 1.0-beta book exactly when it
  matters. For a portfolio with beta below 1 these downside rows are an
  **optimistic bound: they understate the loss.**
* **Instantaneous and linear.** No path, no convexity, no gap risk, no
  rebalancing, no time for anything to be re-estimated. A 20% fall arriving over
  a year and a 20% fall arriving over a fortnight are the same row here, and are
  not the same event for anyone holding it.
* **One benchmark, one window.** Beta is {fmt_num(beta, 2)} against {name}
  over {n_days} overlapping sessions. A book whose real factor exposure differs
  from that index - small caps, one sector, another currency - is not described
  by it, however precise the number looks.

**Sign convention.** A positive shock is an index rally; the ladder keeps one
upside row so the linearity of the model is visible rather than implied. Money
columns are in {ccy} at today's value of
{fmt_money(value, ccy, 0) if value is not None else "the portfolio"}.
        """,
    )


def _beta_convergence_caption(
    frame: pd.DataFrame, beta: float, value: float | None, ccy: str, name: str
) -> None:
    """Price the 'beta converges to 1' caveat in money rather than in adjectives.

    Uses the harshest shock actually on the ladder, so the sentence cannot drift
    out of step with ``config.FACTOR_SHOCKS``.
    """
    if value is None or abs(beta - 1.0) < 0.02:
        return
    shocks = pd.to_numeric(frame.get("shock_pct"), errors="coerce").dropna()
    if shocks.empty or float(shocks.min()) >= 0.0:
        return

    shock = float(shocks.min()) / 100.0
    fall = fmt_pct(abs(shock) * 100.0, 0)
    at_beta = value * (1.0 + beta * shock)
    at_one = value * (1.0 + shock)
    gap = abs(at_one - at_beta)
    if beta < 1.0:
        st.caption(
            f"What that second caveat costs, in money: on a {name} fall of {fall} the "
            f"ladder projects {fmt_money(at_beta, ccy, 0)}. If the sell-off dragged this "
            f"book's beta to 1.0 - the usual behaviour when correlations converge - the "
            f"same fall lands at {fmt_money(at_one, ccy, 0)}, a further "
            f"{fmt_money(gap, ccy, 0)} of loss the table does not show."
        )
    else:
        st.caption(
            f"Working the caveat the other way: with beta above 1, a {fall} {name} fall "
            f"projects {fmt_money(at_beta, ccy, 0)} against {fmt_money(at_one, ccy, 0)} "
            f"for the index itself. Convergence toward 1.0 in a crisis would make this "
            f"row {fmt_money(gap, ccy, 0)} too pessimistic - but only if the book's extra "
            f"sensitivity really is beta rather than concentration, which beta cannot "
            f"tell you."
        )


# ---------------------------------------------------------------------------
# 4. worst realised windows
# ---------------------------------------------------------------------------
def _worst_windows(result: Any, cfg: Any) -> None:
    ui_theme.section("Worst stretches actually lived", "\U000023f1")

    if not guards.require_history(result, "The worst-window scan"):
        return

    usable = tuple(w for w in _WINDOW_CHOICES if result.n_days > w) or _WINDOW_CHOICES
    default = _DEFAULT_WINDOW if _DEFAULT_WINDOW in usable else usable[0]
    window = int(
        st.select_slider(
            "Holding period",
            options=usable,
            value=default,
            format_func=lambda w: _WINDOW_LABELS.get(w, f"{w} sessions"),
            key="stress_worst_window",
            help="How long you would have been holding. Longer periods find deeper "
            "losses because there is more time for them to compound.",
        )
    )

    frame = strs.worst_rolling_windows(result.values, window, _TOP_N)
    ccy = getattr(cfg, "base_currency", "INR")
    symbol = getattr(cfg, "currency_symbol", "")
    value = _finite((result.summary or {}).get("current_value"))
    scanned = max(result.n_days - window, 0)

    if frame.empty:
        ui.empty_state(
            "No complete window of that length",
            f"A {window}-session holding period needs {window + 1} observations and this "
            f"analysis window has {result.n_days}. Nothing is extrapolated to fill it.",
            icon="\U0001f4c9",
            hint="Pick a shorter holding period, or widen the history window in the sidebar.",
        )
        return

    worst = frame.iloc[0]
    worst_pct = _finite(worst.get("return_pct"))
    on_today = None if (worst_pct is None or value is None) else value * worst_pct / 100.0

    ui.kpi_row(
        [
            {
                "label": f"Worst {window}-session stretch",
                "value": fmt_pct(worst_pct),
                "tone": delta_tone(worst_pct),
                "delta": f"{_date_text(worst.get('start'))} to {_date_text(worst.get('end'))}",
                "delta_tone": "muted",
                "footnote": (
                    f"{fmt_money(worst.get('start_value'), ccy, 0)} down to "
                    f"{fmt_money(worst.get('end_value'), ccy, 0)} at the prices of the day"
                ),
                "help_text": "The deepest realised loss over a holding period of this length.",
            },
            {
                "label": "Same move on today's book",
                "value": fmt_money(on_today, ccy, 0, signed=True) if on_today is not None else DASH,
                "tone": delta_tone(on_today),
                "footnote": (
                    f"{fmt_pct(worst_pct)} applied to {fmt_money(value, ccy, 0)}"
                    if value is not None
                    else "No current value to apply it to"
                ),
                "help_text": "A restatement of the same percentage at today's scale, "
                "not a second measurement.",
            },
            {
                "label": "Windows scanned",
                "value": f"{scanned:,}",
                "tone": "neutral",
                "footnote": (
                    f"Every {window}-session period inside {result.n_days} sessions, overlapping"
                ),
                "help_text": "The listed windows are the worst non-overlapping ones "
                "drawn from this many candidates.",
            },
        ],
        columns=3,
    )

    display = pd.DataFrame(
        {
            "Rank": pd.to_numeric(frame["rank"], errors="coerce"),
            "From": frame["start"],
            "To": frame["end"],
            "Sessions": pd.to_numeric(frame["days"], errors="coerce"),
            "Calendar days": pd.to_numeric(frame["calendar_days"], errors="coerce"),
            "Return": pd.to_numeric(frame["return_pct"], errors="coerce"),
            "Value at start": pd.to_numeric(frame["start_value"], errors="coerce"),
            "Value at end": pd.to_numeric(frame["end_value"], errors="coerce"),
        }
    )
    ui.dataframe(
        display,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "From": st.column_config.DateColumn("From", format="DD MMM YYYY", width="small"),
            "To": st.column_config.DateColumn("To", format="DD MMM YYYY", width="small"),
            "Sessions": st.column_config.NumberColumn("Sessions", format="%d", width="small"),
            "Calendar days": st.column_config.NumberColumn(
                "Calendar days",
                format="%.0f",
                width="small",
                help="Wall-clock length, including weekends and holidays.",
            ),
            "Return": st.column_config.NumberColumn("Return", format="%.2f%%"),
            "Value at start": st.column_config.NumberColumn(
                "Value at start", format=f"{symbol}%.0f"
            ),
            "Value at end": st.column_config.NumberColumn("Value at end", format=f"{symbol}%.0f"),
        },
    )

    st.caption(
        f"These actually happened to this exact portfolio inside the "
        f"{PERIOD_LABELS.get(getattr(cfg, 'period', ''), str(getattr(cfg, 'period', '')))} "
        f"analysis window - no model, no distribution, no assumption beyond the one this "
        f"whole application makes, that today's share counts are held constant back "
        f"through the window. The {len(frame)} listed are non-overlapping, so the same "
        f"bad fortnight can never appear five times with its start date nudged by a day."
    )

    ui.methodology(
        "Worst realised windows - selection rule and what it does not tell you",
        f"""
Every trailing return `V_t / V_(t-{window}) - 1` is evaluated across the
{scanned:,} available {window}-session periods, then windows are taken greedily
worst-first, skipping any that overlaps one already taken. Overlap is measured in
*return* space: two windows may share a single boundary price - the close of one
is the open of the next - but never a day's move, so nothing is double-counted.

Greedy selection is not guaranteed to find the globally worst non-overlapping
*set*; it is the standard, deterministic and explainable choice, and it buys the
property that matters here - five genuinely distinct episodes rather than one
episode reported five times.

**What this is.** Measurement. There is no model, no fitted distribution and no
parameter to argue about, which makes it the most defensible number on the page.

**What it is not.** A bound. The worst thing that happened inside
{result.n_days} sessions is not the worst thing that can happen, and a
{window}-session scan says nothing about a loss that takes two years to unfold.
The window also has to be long enough to contain a real drawdown: on a quiet
year this table lists the worst of a benign sample and reads reassuringly.

**Value columns** are the portfolio's own value at those dates - today's share
counts at that day's prices - so they are on the scale of the *analysis window*,
not of today's book. The "same move on today's book" card restates the
percentage at the current market value, which is the number worth carrying.
        """,
    )


# ---------------------------------------------------------------------------
# 5. choosing between the three
# ---------------------------------------------------------------------------
def _toolkit(result: Any, cfg: Any) -> None:
    ui_theme.section("Which tool answers which question", "\U0001f9ed")

    name = getattr(cfg, "benchmark_name", "the benchmark")
    ui.methodology(
        "Three stress methods, three different questions",
        f"""
| Question you are asking | Tool | What it assumes | Where it breaks |
| --- | --- | --- | --- |
| "What did a real crisis do to this exact mix?" | **Crisis replay** | These weights, that market. Nothing modelled. | Coverage. Names that did not exist then are unknown, and the last crisis is not a template for the next one. |
| "What if {name} drops 20% from here?" | **Benchmark shock ladder** | One linear factor, a stable beta, no stock-specific risk. | Beta drifts toward 1.0 in a sell-off, so low-beta books are flattered. Idiosyncratic blow-ups are invisible. |
| "What is the worst this book has actually done?" | **Worst realised windows** | Only that today's share counts held constant. | It is a floor made of facts, not a ceiling: bounded by the {result.n_days} sessions analysed. |

**How to use them together.** Start with the realised windows - they are the
only numbers here with no model in them, and they set the scale of what this
portfolio does on a bad month. Use the crisis replay for narrative and for the
coverage question: if most scenarios come back uncovered, you are holding a book
too young to have been tested, which is itself the finding. Use the shock ladder
last, for a fast what-if on a portfolio you can already characterise with a
single beta - and read its downside rows as optimistic.

**What none of them is.** A probability. Nothing on this page says how *likely*
any of it is; a stress test is a conditional statement - *if* this happens, *then*
roughly this. The probabilistic view lives elsewhere: one-day VaR and expected
shortfall on **Risk analytics**, and the forward-looking distribution of outcomes
on **Monte Carlo**. Nor is any of it advice, a forecast, or a bound on losses.

**Everything here is gross.** No brokerage, taxes, slippage or liquidity. In the
windows replayed above, the exit price for a mid or small cap was frequently not
the close - which is the one way real crisis losses reliably exceed the modelled
ones.
        """,
        expanded=True,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Stress testing",
        "What this book lost in past crises, what a benchmark shock does to it, "
        "and the worst it has actually done",
        icon="\U0001f32a",
    )

    result = guards.require_result()
    if result is None:
        return

    cfg = result.config
    positions = result.positions
    if positions is None or len(positions) == 0 or "ticker" not in positions:
        ui.empty_state(
            "No priced positions to stress",
            "The analysis ran but no holding produced a usable quote, so there are no "
            "weights to replay through a crisis.",
            icon="\U0001f4ed",
            hint="Check the ticker symbols on the Portfolio page, then re-run.",
        )
        ui.warning_list(result.warnings, "Data notes from this run")
        return

    summary = result.summary or {}
    bench = result.benchmark or {}
    ui.badges(
        [
            (f"{len(positions)} holdings at today's weights", "accent"),
            (
                f"{len(HISTORICAL_STRESS)} crisis windows, {len(FACTOR_SHOCKS)} benchmark shocks",
                "neutral",
            ),
            (
                f"Beta {fmt_num(bench.get('beta'), 2)} vs "
                f"{getattr(cfg, 'benchmark_name', 'benchmark')}"
                if _finite(bench.get("beta")) is not None
                else "Beta unavailable",
                "neutral" if _finite(bench.get("beta")) is not None else "warn",
            ),
            (
                f"Book {fmt_money(summary.get('current_value'), getattr(cfg, 'base_currency', 'INR'), 0, compact=True)}",
                "muted",
            ),
        ]
    )
    st.caption(
        "Three independent views, deliberately not blended into one score: a replay of "
        "documented crises against today's weights, a first-order benchmark shock, and "
        "the worst stretches this portfolio has genuinely lived through. Where a number "
        "rests on partial data, the page says so beside it rather than in a footnote."
    )

    _replay(result, cfg)
    ui_theme.spacer(0.5)
    _beta_shocks(result, cfg)
    ui_theme.spacer(0.5)
    _worst_windows(result, cfg)
    ui_theme.spacer(0.5)
    _toolkit(result, cfg)
