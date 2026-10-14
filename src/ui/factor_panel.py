"""Factor-regression UI: pick a factor source, regress, show the loadings.

Lives outside ``pages/`` because it is a self-contained panel the Correlation
& factors page mounts, and keeping it separate keeps that page readable.

Two factor sources, and the difference between them is stated everywhere it
matters rather than once in small print:

* **ETF proxies** — built from tradeable prices already reachable through the
  data layer. Convenient, no upload, and *not* the Fama-French factors.
* **A factor file** — Kenneth French's data library, the IIM Ahmedabad Indian
  dataset, or the user's own. The rigorous option.

The panel never presents a proxy loading as an academic factor loading, and
never shows a beta without its standard error.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.analytics import factor_model as fm
from src.config import (
    FACTOR_PROXIES,
    FACTOR_PROXY_GAPS,
    FACTOR_ROLLING_WINDOW,
    TRADING_DAYS,
)
from src.ui import components as ui
from src.ui import data_access
from src.ui import formatters as fmt

__all__ = ["render"]

#: Session keys for the staged factor file (parsed once per upload).
_FILE_KEY = "factor_file"

#: Markets offered, keyed to :data:`src.config.FACTOR_PROXIES`.
_MARKETS = tuple(FACTOR_PROXIES)


def _proxy_caption(market: str) -> None:
    """State plainly what the proxies are and are not."""
    st.caption(
        "**These are proxies, not the Fama-French factors.** The academic factors are "
        "built from the whole cross-section of stocks sorted on size and book-to-market "
        "and are not investable; an ETF spread is investable and measurably different. "
        "The gap includes fund fees, the index provider's own screening rules, and each "
        "ETF's inception date — which also caps how far back the regression can run."
    )
    gap = FACTOR_PROXY_GAPS.get(market)
    if gap:
        st.info(gap)


def _proxy_table(specs) -> None:
    ui.dataframe(
        pd.DataFrame(
            [
                {
                    "Factor": spec.name,
                    "Meaning": spec.label,
                    "Built as": spec.formula,
                    "Caveat": spec.caveat or "—",
                }
                for spec in specs
            ]
        )
    )


def _load_proxies(market: str, period: str, risk_free_rate: float):
    """Fetch proxy prices and build the factor returns. ``(factors, warnings)``."""
    specs = FACTOR_PROXIES.get(market, ())
    symbols = tuple(dict.fromkeys(s for spec in specs for s in spec.symbols))
    if not symbols:
        return pd.DataFrame(), [f"No factor proxies are defined for {market}."], specs

    prices, errors = data_access.load_history(symbols, period)
    warnings = [f"{ticker}: {message}" for ticker, message in errors.items()]
    factors, build_warnings = fm.build_proxy_factors(prices, specs, risk_free_rate=risk_free_rate)
    return factors, [*warnings, *build_warnings], specs


def _factor_source(cfg) -> tuple[pd.DataFrame, list[str], str, tuple]:
    """The source picker. ``(factors, warnings, label, specs)``."""
    source = st.radio(
        "Factor source",
        ["ETF proxies", "Factor file (French / IIM Ahmedabad)"],
        horizontal=True,
        key="factor_source",
        help="Proxies need no upload but are not the academic factors. A published "
        "factor file is the rigorous option.",
    )

    if source.startswith("ETF"):
        market = st.selectbox(
            "Proxy market",
            _MARKETS,
            index=_MARKETS.index("India") if "India" in _MARKETS else 0,
            key="factor_market",
            help="Pick the market your holdings actually trade in — regressing Indian "
            "holdings on US factors mostly measures the exchange-rate and time-zone "
            "mismatch.",
        )
        _proxy_caption(market)
        with st.spinner("Fetching factor-proxy prices…"):
            factors, warnings, specs = _load_proxies(market, cfg.period, cfg.risk_free_rate)
        if specs:
            with st.expander("How each proxy is built", expanded=False):
                _proxy_table(specs)
        return factors, warnings, f"{market} ETF proxies", specs

    st.caption(
        "Upload a factor file: Kenneth French's data library (US) or the IIM Ahmedabad "
        "dataset (India). The first column must be the date (ISO, `YYYYMMDD` or "
        "`YYYYMM`); every other numeric column becomes a factor. Published files are "
        "usually quoted in **percent** and are converted automatically — the decision "
        "made is always reported."
    )
    uploaded = st.file_uploader("Factor file", type=["csv", "txt"], key="factor_upload")
    scale = st.radio(
        "Values are in",
        ["auto", "percent", "decimal"],
        horizontal=True,
        key="factor_scale",
        help="Getting this wrong scales every loading by 100. 'auto' inspects the "
        "magnitude and states what it chose.",
    )

    if uploaded is None:
        return pd.DataFrame(), ["No factor file uploaded yet."], "factor file", ()

    staged = st.session_state.get(_FILE_KEY)
    if staged is None or staged.get("id") != (uploaded.file_id, scale):
        with st.spinner(f"Reading {uploaded.name}…"):
            frame, warnings = fm.parse_factor_csv(uploaded, scale=scale)
        st.session_state[_FILE_KEY] = {
            "id": (uploaded.file_id, scale),
            "frame": frame,
            "warnings": warnings,
        }
        staged = st.session_state[_FILE_KEY]

    frame: pd.DataFrame = staged["frame"]
    warnings = list(staged["warnings"])
    if frame.empty:
        return frame, warnings, uploaded.name, ()

    # RF is a rate series, not a factor to load on.
    candidates = [c for c in frame.columns if c.upper() != "RF"]
    chosen = st.multiselect(
        "Factors to include",
        candidates,
        default=candidates[:5],
        key="factor_columns",
        help="Each extra factor costs a degree of freedom and adds collinearity. "
        "Three to five is the usual range.",
    )
    if not chosen:
        return pd.DataFrame(), [*warnings, "Select at least one factor."], uploaded.name, ()
    return frame[chosen], warnings, uploaded.name, ()


def _kpis(result: fm.RegressionResult) -> None:
    alpha_tone = (
        ("gain" if result.alpha_annual > 0 else "loss") if result.alpha_significant() else "neutral"
    )
    ui.kpi_row(
        [
            {
                "label": "Annualised alpha",
                "value": fmt.fmt_ratio_pct(result.alpha_annual, signed=True),
                "tone": alpha_tone,
                "delta": f"t = {fmt.fmt_num(result.alpha_t_stat)}",
                "delta_tone": alpha_tone,
                "footnote": "significant at 5%"
                if result.alpha_significant()
                else "not distinguishable from zero",
                "help_text": "The part of the excess return these factors do not "
                "explain. Read the t-statistic, not the level: an alpha with |t| below "
                "about 2 is indistinguishable from luck.",
            },
            {
                "label": "R²",
                "value": fmt.fmt_pct(result.explained_pct),
                "tone": "neutral",
                "footnote": f"adjusted {fmt.fmt_pct(result.adj_r_squared * 100.0)}",
                "help_text": "Share of the portfolio's return variation the factors account for.",
            },
            {
                "label": "Idiosyncratic vol",
                "value": fmt.fmt_ratio_pct(result.residual_vol_annual),
                "tone": "neutral",
                "footnote": "annualised residual",
                "help_text": "Annualised volatility of what the factors leave "
                "unexplained — stock-specific risk the model does not price.",
            },
            {
                "label": "Observations",
                "value": f"{result.n_observations:,}",
                "tone": "gain" if result.reliable else "warn",
                "footnote": f"{result.n_factors} factors · {result.dof} d.o.f.",
            },
        ],
        columns=4,
    )


def _loadings_table(result: fm.RegressionResult) -> None:
    frame = result.summary_frame()
    ui.dataframe(
        frame,
        column_config={
            "Estimate": st.column_config.NumberColumn(format="%.4f"),
            "Std error": st.column_config.NumberColumn(format="%.4f"),
            "t-stat": st.column_config.NumberColumn(format="%.2f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "VIF": st.column_config.NumberColumn(
                format="%.1f",
                help="Variance inflation factor. Above 5 the factors overlap enough "
                "that their individual loadings are unstable, even when the overall "
                "fit is fine.",
            ),
        },
    )
    st.caption(
        f"Standard errors: **{result.se_method}**. "
        "A loading is worth reading only alongside its standard error — |t| below "
        "about 2 means the exposure is not distinguishable from zero at the usual 5% "
        "level, whatever the point estimate looks like."
    )


def _significant_prose(result: fm.RegressionResult) -> None:
    """One sentence naming what the regression actually found."""
    real = [load for load in result.loadings if load.significant()]
    if not real:
        st.info(
            "No factor loading is distinguishable from zero at the 5% level over this "
            "window. That is a finding, not a failure: it usually means the window is "
            "short, or the portfolio's moves are not well described by these factors."
        )
        return
    parts = [
        f"**{load.name}** {load.beta:+.2f} (t = {load.t_stat:+.1f})"
        for load in sorted(real, key=lambda item: abs(item.t_stat), reverse=True)
    ]
    st.markdown("Statistically detectable exposures: " + ", ".join(parts) + ".")


def _contributions(result: fm.RegressionResult, factors: pd.DataFrame) -> None:
    frame = fm.factor_contributions(result, factors)
    if frame.empty:
        return
    with st.expander("Return decomposition implied by the model", expanded=False):
        st.caption(
            "``beta × mean factor return``, annualised, plus alpha. This is the "
            "MODEL's attribution — what the fitted relationship implies — not a "
            "realised P&L attribution from your actual positions. The Holdings page "
            "does the latter, and the two answer different questions."
        )
        display = frame.rename(
            columns={
                "factor": "Factor",
                "beta": "Beta",
                "mean_annual": "Factor return p.a.",
                "contribution_annual": "Contribution p.a.",
                "share_pct": "Share of gross %",
            }
        )
        ui.dataframe(
            display,
            column_config={
                "Beta": st.column_config.NumberColumn(format="%.3f"),
                "Factor return p.a.": st.column_config.NumberColumn(format="%.2f%%"),
                "Contribution p.a.": st.column_config.NumberColumn(format="%.2f%%"),
                "Share of gross %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )


def _rolling(returns, factors: pd.DataFrame, risk_free_rate: float) -> None:
    """Loading drift. Often the most useful thing on the page."""
    window = st.select_slider(
        "Rolling window (trading days)",
        options=[63, 126, 252],
        value=FACTOR_ROLLING_WINDOW,
        key="factor_rolling_window",
        help="A single full-sample loading averages away the fact that exposures "
        "drift. This re-estimates them over a moving window.",
    )
    frame = fm.rolling_betas(returns, factors, window=int(window), risk_free_rate=risk_free_rate)
    if frame.empty:
        st.caption(
            f"Not enough overlapping history for a {window}-day rolling window. "
            "Widen the history window in the sidebar or pick a shorter rolling window."
        )
        return

    betas = frame.drop(columns=["alpha_annual", "r_squared"], errors="ignore")
    st.line_chart(betas, height=300)
    st.caption(
        "Point estimates only. Standard errors are deliberately omitted here: "
        "overlapping windows make a strip of t-statistics look far more informative "
        "than it is, and the honest use of this chart is the SHAPE of the exposure "
        "over time."
    )

    drift = {
        column: float(betas[column].iloc[-1] - betas[column].iloc[0])
        for column in betas.columns
        if math.isfinite(_safe(betas[column].iloc[-1]))
        and math.isfinite(_safe(betas[column].iloc[0]))
    }
    if drift:
        biggest = max(drift.items(), key=lambda item: abs(item[1]))
        if abs(biggest[1]) >= 0.20:
            st.markdown(
                f"Largest drift: **{biggest[0]}** moved {biggest[1]:+.2f} from the "
                "first window to the last — the full-sample loading above is an "
                "average across that change, not a description of the portfolio today."
            )


def _safe(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _methodology(result: fm.RegressionResult) -> None:
    ui.methodology(
        "How this regression is estimated",
        f"""
**The model**

```
r_p − r_f  =  α  +  Σ βᵢ · Fᵢ  +  ε
```

estimated by ordinary least squares on **{result.n_observations:,}** overlapping
observations with **{result.dof}** degrees of freedom.

**Standard errors — {result.se_method}.** Return residuals are heteroskedastic, and
both factors and residuals are often serially correlated. Classical OLS standard
errors assume neither, so they come out too small and t-statistics too large. On
persistent data the difference is not marginal: a t-statistic of 16 becomes 8 once
the Newey-West correction is applied. The truncation lag follows the usual
`floor(4·(n/100)^(2/9))` rule.

**Alpha annualisation.** α is the per-period intercept; the headline figure is
`α × {result.periods_per_year}`. Linear scaling is the factor-literature convention and keeps
the t-statistic meaningful, since annualising the estimate and its standard error
by the same constant leaves the ratio unchanged. Note this differs slightly from
the Jensen alpha on the **Benchmark** page, which is built from compounded
annualised returns — both are defensible estimators of related quantities.

**Excess returns.** The portfolio return has the risk-free rate subtracted before
regressing. Skipping that step would load the risk-free rate into the intercept
and quietly inflate alpha.

**What would make this wrong**

- A window too short for the number of factors — {result.n_factors} factors need
  many more than {result.n_factors + 2} observations before the loadings mean anything.
- Highly collinear factors: R² stays high while the individual loadings become
  unstable. The VIF column is there to make that visible.
- Regressing holdings on the factors of a market they do not trade in, which
  mostly measures the calendar and currency mismatch.
- Reading a loading without its t-statistic.
""",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(result: object) -> None:
    """Mount the panel against a completed :class:`AnalysisResult`."""
    returns = getattr(result, "returns", None)
    cfg = getattr(result, "config", None)
    if returns is None or cfg is None or len(returns) < 2:
        st.caption(
            "A factor regression needs a portfolio return series; this run produced "
            "too little history."
        )
        return

    factors, warnings, label, _specs = _factor_source(cfg)
    ui.warning_list(warnings, "Factor data notes")
    if factors is None or factors.empty:
        return

    regression = fm.factor_regression(
        returns,
        factors,
        risk_free_rate=cfg.risk_free_rate,
        robust=True,
        periods_per_year=TRADING_DAYS,
    )

    if not regression.ok:
        ui.empty_state(
            "Regression not estimated",
            "; ".join(regression.warnings)
            or "The portfolio and factor series could not be regressed.",
            icon="📐",
            hint="Try a longer history window, or fewer factors.",
        )
        return

    st.caption(
        f"Factors: **{label}** · {fmt.fmt_date(regression.start)} → {fmt.fmt_date(regression.end)}"
    )
    _kpis(regression)
    ui.warning_list(list(regression.warnings), "Statistical caveats")

    ui_spacer = st.container()
    with ui_spacer:
        _significant_prose(regression)
        _loadings_table(regression)

    _contributions(regression, factors)

    st.markdown("**How the exposures have drifted**")
    _rolling(returns, factors, cfg.risk_free_rate)

    _methodology(regression)
