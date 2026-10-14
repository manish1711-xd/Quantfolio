"""Black-Litterman UI: state views, see the blend, re-optimise on the posterior.

Mounted by the Optimiser page rather than living in it, because the page is
already long and this is a self-contained flow: pick a prior, state views,
compare.

Deliberately ADDITIVE. The page's own historical-mean estimates and every
solver it already runs are untouched; this panel re-solves the same public
objectives on the Black-Litterman posterior and shows the two answers side by
side. That comparison is the point — "what changed because of my views" is far
more informative than a single set of weights with no baseline.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.analytics import black_litterman as bl
from src.analytics import optimize as opt
from src.analytics import risk as rsk
from src.config import TRADING_DAYS
from src.ui import components as ui
from src.ui import formatters as fmt

__all__ = ["render"]

#: Editor column contract for the views grid.
_VIEW_COLUMNS = ("kind", "asset", "versus", "return_pct", "confidence_pct")

_ABSOLUTE = "Absolute"
_RELATIVE = "Relative"

#: Prior choices, each a different and clearly-labelled assumption.
_PRIOR_CAP = "Market-cap weights"
_PRIOR_CURRENT = "My current weights"
_PRIOR_EQUAL = "Equal weights"


def _blank_views() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": pd.Series(dtype="object"),
            "asset": pd.Series(dtype="object"),
            "versus": pd.Series(dtype="object"),
            "return_pct": pd.Series(dtype="float64"),
            "confidence_pct": pd.Series(dtype="float64"),
        }
    )


def _prior_weights(
    choice: str, positions: pd.DataFrame, assets: list[str], profiles
) -> tuple[pd.Series, list[str], str]:
    """``(weights, warnings, description)`` for the chosen prior."""
    if choice == _PRIOR_EQUAL:
        return (
            pd.Series(1.0 / len(assets), index=assets),
            [],
            "an equally-weighted portfolio of your holdings",
        )

    if choice == _PRIOR_CURRENT:
        if "weight_pct" not in positions.columns:
            return pd.Series(dtype="float64"), ["Current weights are unavailable."], ""
        weights = (
            pd.Series(
                positions["weight_pct"].to_numpy() / 100.0,
                index=[str(t) for t in positions["ticker"]],
            )
            .reindex(assets)
            .fillna(0.0)
        )
        return (
            weights,
            [
                "Using your own allocation as the equilibrium assumes you are already "
                "holding the optimal portfolio, so the equilibrium returns are simply "
                "the ones that would justify what you already own. Views then read as "
                "departures from your own status quo, which is a coherent question but "
                "not the one Black-Litterman was written to answer."
            ],
            "your current allocation treated as the equilibrium",
        )

    weights, warnings = bl.market_cap_weights(positions, profiles)
    if weights.empty:
        # Falling back rather than dead-ending. Yahoo reports no market cap for
        # most ETFs and many smaller Indian listings, so an ETF-heavy book hits
        # this routinely -- and "Black-Litterman is unavailable" would be a poor
        # answer when a perfectly usable prior is one line away. The substitution
        # is a different assumption, so it is stated, not slipped in.
        fallback, fallback_warnings, label = _prior_weights(
            _PRIOR_CURRENT, positions, assets, profiles
        )
        if fallback.empty:
            return fallback, [*warnings, *fallback_warnings], ""
        return (
            fallback,
            [
                *warnings,
                "**Fell back to your current weights** as the equilibrium prior, since "
                "no market-cap data was available. Pick a prior explicitly above if "
                "that is not what you want.",
                *fallback_warnings,
            ],
            label,
        )
    return (
        weights.reindex(assets).fillna(0.0),
        warnings,
        "a cap-weighted portfolio of your holdings",
    )


def _calibrate_delta(result, weights: pd.Series, cov: pd.DataFrame) -> tuple[float, list[str]]:
    """Back delta out of the prior portfolio's own premium, if possible."""
    returns = getattr(result, "returns", None)
    cfg = getattr(result, "config", None)
    rate = getattr(cfg, "risk_free_rate", 0.0) or 0.0

    try:
        vector = weights.reindex(cov.columns).fillna(0.0).to_numpy(dtype="float64")
        variance = float(vector @ cov.to_numpy(dtype="float64") @ vector)
    except Exception:  # noqa: BLE001 - defensive on a malformed matrix
        return float("nan"), ["Prior portfolio variance could not be computed."]

    premium = float("nan")
    if returns is not None and len(returns) > 1:
        annual = rsk.annualized_return(returns) if hasattr(rsk, "annualized_return") else None
        if annual is None or not math.isfinite(float(annual)):
            annual = float(pd.Series(returns).mean()) * TRADING_DAYS
        premium = float(annual) - float(rate)

    delta = bl.implied_risk_aversion(premium, variance)
    if math.isfinite(delta) and delta > 0:
        return delta, [
            f"Risk aversion δ = {delta:.2f}, calibrated as (portfolio excess return "
            f"{premium:.2%}) ÷ (prior variance {variance:.4f}). It scales the whole "
            "equilibrium, so the relative tilts matter more than the levels."
        ]
    return float("nan"), []


def _parse_views(frame: pd.DataFrame | None, assets: list[str]) -> tuple[list[bl.View], list[str]]:
    """Turn the editor grid into views. Bad rows are reported, never dropped silently."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return [], []

    views: list[bl.View] = []
    problems: list[str] = []
    for offset, record in enumerate(frame.to_dict("records"), start=1):
        asset = str(record.get("asset") or "").strip()
        if not asset:
            continue  # a blank grid row, not an error

        kind = str(record.get("kind") or _ABSOLUTE).strip() or _ABSOLUTE
        try:
            expected = float(record.get("return_pct")) / 100.0
        except (TypeError, ValueError):
            problems.append(f"View {offset} ({asset}): expected return is not a number.")
            continue
        if not math.isfinite(expected):
            problems.append(f"View {offset} ({asset}): expected return is missing.")
            continue

        try:
            confidence = float(record.get("confidence_pct")) / 100.0
        except (TypeError, ValueError):
            confidence = bl.DEFAULT_CONFIDENCE
        if not math.isfinite(confidence):
            confidence = bl.DEFAULT_CONFIDENCE

        if asset not in assets:
            problems.append(
                f"View {offset}: {asset} is not among the optimisable holdings "
                f"({', '.join(assets)}) — only names with usable price history qualify."
            )
            continue

        if kind.lower().startswith("rel"):
            versus = str(record.get("versus") or "").strip()
            if not versus:
                problems.append(
                    f"View {offset} ({asset}): a relative view needs a second asset "
                    "in the 'versus' column."
                )
                continue
            if versus not in assets:
                problems.append(f"View {offset}: {versus} is not among the holdings.")
                continue
            if versus == asset:
                problems.append(f"View {offset}: {asset} cannot outperform itself.")
                continue
            views.append(bl.relative_view(asset, versus, expected, confidence))
        else:
            views.append(bl.absolute_view(asset, expected, confidence))
    return views, problems


def _views_editor(assets: list[str]) -> pd.DataFrame:
    st.caption(
        "One row per view. **Absolute** = “this asset will return X”. "
        "**Relative** = “this asset beats *versus* by X” — the more defensible kind, "
        "since it needs no opinion about the level of the market. Returns are "
        "annualised and in **excess of the risk-free rate**, matching the equilibrium."
    )
    current = st.session_state.get("bl_views")
    if current is None or not isinstance(current, pd.DataFrame):
        current = _blank_views()

    edited = st.data_editor(
        current,
        num_rows="dynamic",
        width="stretch",
        key="bl_views_editor",
        column_config={
            "kind": st.column_config.SelectboxColumn(
                "Type", options=[_ABSOLUTE, _RELATIVE], default=_ABSOLUTE, width="small"
            ),
            "asset": st.column_config.SelectboxColumn("Asset", options=assets, width="medium"),
            "versus": st.column_config.SelectboxColumn(
                "versus", options=assets, width="medium", help="Relative views only."
            ),
            "return_pct": st.column_config.NumberColumn(
                "Excess return % p.a.", format="%.2f", width="small"
            ),
            "confidence_pct": st.column_config.NumberColumn(
                "Confidence %",
                min_value=0.0,
                max_value=100.0,
                default=50.0,
                format="%.0f",
                width="small",
                help="50% reproduces He-Litterman's view uncertainty. 100% enforces "
                "the view almost exactly; 0% ignores it.",
            ),
        },
    )
    st.session_state["bl_views"] = edited
    return edited


def _blend_chart(result: bl.BlackLittermanResult) -> None:
    frame = result.summary_frame()
    if frame.empty:
        return
    chart = frame.set_index("asset")[["equilibrium", "posterior"]] * 100.0
    st.bar_chart(chart, height=280)
    st.caption(
        "Annualised excess returns. Where the two bars match, nobody had a view and "
        "the equilibrium stands — which is exactly the behaviour that makes this safe "
        "to leave switched on."
    )


def _blend_table(result: bl.BlackLittermanResult) -> None:
    frame = result.summary_frame()
    display = frame.rename(
        columns={
            "asset": "Asset",
            "market_weight": "Prior weight",
            "equilibrium": "Equilibrium",
            "posterior": "Posterior",
            "tilt": "Tilt",
        }
    )
    for column in ("Prior weight", "Equilibrium", "Posterior", "Tilt"):
        display[column] = display[column] * 100.0
    ui.dataframe(
        display,
        column_config={
            "Prior weight": st.column_config.NumberColumn(format="%.2f%%"),
            "Equilibrium": st.column_config.NumberColumn(format="%+.2f%%"),
            "Posterior": st.column_config.NumberColumn(format="%+.2f%%"),
            "Tilt": st.column_config.NumberColumn(
                format="%+.2f%%",
                help="Posterior minus equilibrium: what your views actually changed. "
                "Non-zero for assets you never mentioned is correct — the tilt "
                "propagates through the covariance.",
            ),
        },
    )

    spillover = [
        row["asset"]
        for _, row in frame.iterrows()
        if abs(float(row["tilt"])) > 1e-6
        and row["asset"] not in {a for view in result.views for a in view.assets}
    ]
    if spillover:
        st.markdown(
            f"**{', '.join(spillover)}** carries no view of its own but still moved: "
            "Black-Litterman propagates a view through the covariance, so a correlated "
            "asset shifts with it. That is the feature, not a bug — it is what "
            "separates this from simply overwriting an expected return."
        )


def _view_impacts(result: bl.BlackLittermanResult) -> None:
    frame = result.view_impacts()
    if frame.empty:
        return
    display = frame.rename(
        columns={
            "view": "View",
            "equilibrium": "Equilibrium implies",
            "your view": "You say",
            "difference": "Surprise",
            "confidence": "Confidence",
        }
    )
    for column in ("Equilibrium implies", "You say", "Surprise", "Confidence"):
        display[column] = display[column] * 100.0
    ui.dataframe(
        display,
        column_config={
            "Equilibrium implies": st.column_config.NumberColumn(format="%+.2f%%"),
            "You say": st.column_config.NumberColumn(format="%+.2f%%"),
            "Surprise": st.column_config.NumberColumn(
                format="%+.2f%%",
                help="The gap the model actually acts on. A view that merely restates "
                "the equilibrium changes nothing, however confident it is.",
            ),
            "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )


def _reoptimise(
    result: bl.BlackLittermanResult,
    historical_mu: pd.Series | None,
    risk_free_rate: float,
    max_weight: float,
) -> None:
    """Solve the same objectives on the posterior, next to the historical answer."""
    posterior_mu = result.posterior_mu()
    cov = result.posterior_covariance

    rows: list[dict[str, object]] = []
    for label, solver in (
        ("Max Sharpe", opt.max_sharpe),
        ("Min variance", opt.min_variance),
    ):
        blended = solver(posterior_mu, cov, risk_free_rate, max_weight=max_weight)
        if not blended.weights:
            continue
        baseline = None
        if historical_mu is not None and not historical_mu.empty:
            aligned = historical_mu.reindex(cov.columns)
            if aligned.notna().all():
                baseline = solver(
                    aligned, result.prior_covariance, risk_free_rate, max_weight=max_weight
                )
        for asset, weight in blended.weights.items():
            rows.append(
                {
                    "Objective": label,
                    "Asset": asset,
                    "Black-Litterman": weight * 100.0,
                    "Historical mean": (
                        baseline.weights.get(asset, 0.0) * 100.0 if baseline else float("nan")
                    ),
                }
            )

    if not rows:
        st.caption("Neither objective converged on the posterior inputs.")
        return

    frame = pd.DataFrame(rows)
    frame["Change"] = frame["Black-Litterman"] - frame["Historical mean"]
    ui.dataframe(
        frame,
        column_config={
            "Black-Litterman": st.column_config.NumberColumn(format="%.2f%%"),
            "Historical mean": st.column_config.NumberColumn(format="%.2f%%"),
            "Change": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    st.caption(
        "Both columns use the same solver, the same bounds and the same risk-free "
        "rate; only the expected-return vector differs. The historical column is the "
        "one the rest of this page uses — and the one whose input is a five-year "
        "sample mean, which is the estimator Black-Litterman exists to replace."
    )


def _methodology(result: bl.BlackLittermanResult, prior_label: str) -> None:
    ui.methodology(
        "How the blend is computed",
        f"""
**1 · Reverse optimisation.** The equilibrium excess returns are backed out of the
prior weights:

```
Π = δ · Σ · w_prior
```

with δ = **{result.risk_aversion:.2f}** and the prior being *{prior_label}*. This step is
exactly invertible — pushing Π back through an unconstrained optimiser at the same δ
returns the prior weights to machine precision, which is what makes them an
*equilibrium* rather than just a rescaling.

**2 · Views.** Each view is a row of `P` with a target in `Q`. Absolute views have a
single 1; relative views are `+1, −1` and sum to zero. `Ω` holds their uncertainty,
derived from your confidence as `((1−c)/c) · P τΣ P'` — so c = 50% reproduces
He-Litterman's specification, c → 100% enforces the view, c → 0% ignores it.

**3 · Posterior.**

```
E[R] = Π + τΣP' (P τΣ P' + Ω)⁻¹ (Q − P Π)
Σ_post = Σ + τΣ − τΣP'(P τΣ P' + Ω)⁻¹P τΣ
```

τ = **{result.tau}**. Computed in this (Woodbury) form rather than the textbook
`[(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[…]` because it inverts one {result.n_views}×{result.n_views} matrix instead of
inverting both Ω and τΣ — and a confident view drives Ω towards singular. The two
forms are algebraically identical and the test suite asserts they agree.

**What this is not**

- **Not the market portfolio.** True Black-Litterman uses the global cap-weighted
  market. This uses *{prior_label}*, so Π is the equilibrium of your own universe —
  a narrower claim, and the closest defensible one available here.
- **Not a forecast.** Every number below is a restatement of *your* views under a
  coherent set of assumptions. Nothing here generates or scores a view.
- **τ has no agreed value.** It sets how far views move the answer. 0.05 is common;
  He-Litterman use 0.025. It is exposed above rather than buried.
""",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(
    result: object, historical_mu: pd.Series | None = None, max_weight: float = 0.35
) -> None:
    """Mount the panel against a completed :class:`AnalysisResult`."""
    cov = getattr(result, "covariance", None)
    positions = getattr(result, "positions", None)
    cfg = getattr(result, "config", None)

    if not isinstance(cov, pd.DataFrame) or cov.empty or cfg is None:
        st.caption(
            "Black-Litterman needs a covariance matrix, which requires at least two "
            "holdings with overlapping price history."
        )
        return
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        st.caption("No positions available for this run.")
        return

    assets = [str(c) for c in cov.columns]
    rate = float(getattr(cfg, "risk_free_rate", 0.0) or 0.0)

    prior_choice = st.radio(
        "Equilibrium prior",
        [_PRIOR_CAP, _PRIOR_CURRENT, _PRIOR_EQUAL],
        horizontal=True,
        key="bl_prior",
        help="Black-Litterman reverse-optimises a portfolio to get equilibrium "
        "returns. Which portfolio you choose IS an assumption, so it is stated "
        "rather than defaulted silently.",
    )
    weights, prior_warnings, prior_label = _prior_weights(
        prior_choice, positions, assets, getattr(result, "profiles", None)
    )
    ui.warning_list(prior_warnings, "About this prior")
    if weights.empty:
        return

    delta, delta_notes = _calibrate_delta(result, weights, cov)
    tau = st.select_slider(
        "τ (prior uncertainty)",
        options=[0.01, 0.025, 0.05, 0.10, 0.25],
        value=bl.DEFAULT_TAU,
        key="bl_tau",
        help="How firmly the equilibrium is held. Larger τ lets your views move the "
        "answer further. There is no agreed value — He-Litterman use 0.025.",
    )

    edited = _views_editor(assets)
    views, view_problems = _parse_views(edited, assets)
    ui.warning_list(view_problems, "Views that could not be read")

    blend = bl.black_litterman(
        cov,
        weights,
        views,
        risk_aversion=delta,
        tau=float(tau),
        risk_free_rate=rate,
        prior_source=prior_label,
    )
    if not blend.ok:
        ui.empty_state(
            "Blend not computed",
            "; ".join(blend.warnings) or "The equilibrium could not be derived.",
            icon="⚖️",
        )
        return

    for note in delta_notes:
        st.caption(note)
    ui.warning_list(list(blend.warnings), "Model notes")

    ui.kpi_row(
        [
            {
                "label": "Views applied",
                "value": str(blend.n_views),
                "tone": "accent" if blend.n_views else "neutral",
                "footnote": "0 = pure equilibrium",
            },
            {
                "label": "Risk aversion δ",
                "value": fmt.fmt_num(blend.risk_aversion),
                "tone": "neutral",
                "footnote": "calibrated" if delta_notes else "assumed default",
            },
            {
                "label": "Largest tilt",
                "value": fmt.fmt_ratio_pct(
                    blend.tilt().abs().max() if blend.n_views else 0.0, signed=False
                ),
                "tone": "neutral",
                "footnote": "posterior vs equilibrium",
            },
            {
                "label": "τ",
                "value": f"{blend.tau:g}",
                "tone": "neutral",
                "footnote": "prior uncertainty",
            },
        ],
        columns=4,
    )

    if blend.n_views == 0:
        st.info(
            "No views applied, so the posterior **is** the equilibrium — exactly, not "
            "approximately. Add a view above to see it move. Leaving it empty is a "
            "legitimate use: the equilibrium returns are already a far better "
            "optimiser input than a five-year sample mean."
        )

    _blend_chart(blend)
    _blend_table(blend)

    if blend.n_views:
        st.markdown("**What each view added**")
        _view_impacts(blend)

    st.markdown("**Optimised weights: Black-Litterman vs historical means**")
    _reoptimise(blend, historical_mu, rate, max_weight)

    _methodology(blend, prior_label or prior_choice.lower())
