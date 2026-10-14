"""Optimiser - the efficient frontier and what each objective would hold.

Three questions, in order:

* **Where does this book sit?** Scored on the same expected returns and the same
  covariance matrix as every model portfolio, so the comparison is like for
  like rather than one estimate measured against another.
* **What would each objective hold instead?** Max Sharpe, minimum variance,
  risk parity, maximum diversification and equal weight, all solved long-only
  under one per-name cap.
* **How much of that difference should be believed?** Very little of the
  return-forecast half. The page says so at length rather than burying it.

This is the one page in the application that computes rather than reads. The
pipeline stores a covariance matrix but no frontier, and the frontier has to be
re-solved whenever the weight cap, the shrinkage intensity or the
expected-return convention changes - so the solve lives here, is driven by three
explicit controls, and is cached in session state against those controls so an
unrelated rerun does not pay for it twice.

Units, since this page mixes both conventions: everything the optimiser returns
(``expected_return``, ``volatility``, weights) is a **decimal**;
``positions.weight_pct`` and the targets handed to the Rebalance page are
**percent**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import optimize as opt
from src.analytics import rebalance as rb
from src.analytics import risk as rsk
from src.config import OPTIMIZER, TRADING_DAYS
from src.models import OptimizedPortfolio
from src.ui import bl_panel, guards, session
from src.ui import components as ui
from src.ui import theme as ui_theme
from src.ui.formatters import delta_tone, fmt_money, fmt_num, fmt_pct, fmt_ratio_pct

#: Session key holding the last solve plus the controls that produced it.
_SOLUTION_KEY = "optimizer_solution"

#: Expected-return conventions offered, and how they are labelled.
_METHOD_LABELS: dict[str, str] = {
    "historical": "Historical (geometric)",
    "mean": "Arithmetic mean",
}

#: The frontier point used for the like-for-like comparison must sit within
#: this much annualised return of the current portfolio, or there is no
#: comparable point and the page says so instead of inventing one.
_RETURN_MATCH_TOL = 0.005

#: Weight changes smaller than this (percentage points) are solver dust, not
#: trades.
_DUST_PP = 0.10

#: Strategy pre-selected in the picker. Minimum variance needs no return
#: forecast, which makes it the honest default on a page this fragile.
_DEFAULT_STRATEGY = opt.LABEL_MIN_VARIANCE


# ---------------------------------------------------------------------------
# small local helpers
# ---------------------------------------------------------------------------
def _f(value: Any) -> float:
    """Best-effort float; NaN for anything that will not convert."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(value: Any) -> bool:
    return math.isfinite(_f(value))


def _complete_cases(asset_returns: Any) -> pd.DataFrame:
    """Numeric daily returns restricted to dates on which every asset traded.

    Mirrors what :func:`src.analytics.optimize.optimize_all` does internally, so
    the expected-return vector and the covariance matrix are always estimated
    on identical rows. Pairwise-complete estimation buys a few observations and
    can return a non-PSD matrix, which lets an optimiser manufacture a
    portfolio with negative variance.
    """
    try:
        frame = pd.DataFrame(asset_returns).copy()
    except (TypeError, ValueError):
        return pd.DataFrame(dtype=float)
    if frame.empty or frame.shape[1] == 0:
        return pd.DataFrame(dtype=float)
    try:
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame = frame.apply(pd.to_numeric, errors="coerce").astype("float64")
    except (TypeError, ValueError):
        return pd.DataFrame(dtype=float)
    frame = frame.where(np.isfinite(frame))
    return frame.dropna(axis=0, how="any")


def _weights_of(portfolio: Any) -> dict[str, float]:
    """``{ticker: fraction}`` for an ``OptimizedPortfolio``, defensively."""
    weights = getattr(portfolio, "weights", None) or {}
    out: dict[str, float] = {}
    for key, value in dict(weights).items():
        number = _f(value)
        if math.isfinite(number):
            out[str(key)] = number
    return out


def _largest_weight(portfolio: Any) -> float:
    weights = _weights_of(portfolio)
    return max(weights.values()) if weights else float("nan")


def _names_held(portfolio: Any, floor: float = 0.001) -> int:
    """Positions above 0.1% - anything smaller is not a holding, it is dust."""
    return sum(1 for w in _weights_of(portfolio).values() if w >= floor)


def _effective_bets(portfolio: Any) -> float:
    weights = _weights_of(portfolio)
    if not weights:
        return float("nan")
    return _f(rsk.effective_number_of_bets(pd.Series(weights, dtype=float)))


def _truncate(text: str, limit: int = 90) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "..."


# ---------------------------------------------------------------------------
# 1. controls
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Controls:
    """The three knobs that change the answer, as decimal fractions."""

    max_weight: float
    shrinkage: float
    method: str

    @property
    def uses_pipeline_estimates(self) -> bool:
        """True when ``optimize_all`` would estimate mu and Sigma this way."""
        return (
            abs(self.shrinkage - float(OPTIMIZER.shrinkage)) < 1e-9 and self.method == "historical"
        )


def _controls(cfg: Any) -> _Controls:
    defaults = getattr(cfg, "optimizer", OPTIMIZER)
    cap_default = int(round(_f(getattr(defaults, "max_weight", OPTIMIZER.max_weight)) * 100))
    shrink_default = int(round(_f(getattr(defaults, "shrinkage", OPTIMIZER.shrinkage)) * 100))

    left, middle, right = st.columns([1.1, 1.1, 1.0], gap="medium")

    cap = left.slider(
        "Max weight per name",
        min_value=10,
        max_value=100,
        value=int(min(max(cap_default, 10), 100)),
        step=5,
        format="%d%%",
        key="optimizer_max_weight",
        help="Long-only upper bound applied to every name in every objective "
        "and at every point on the frontier. It is the single most "
        "effective defence against an optimiser that wants to put the "
        "whole book into last year's winner.",
    )
    shrinkage = middle.slider(
        "Covariance shrinkage",
        min_value=0,
        max_value=50,
        value=int(min(max(shrink_default, 0), 50)),
        step=5,
        format="%d%%",
        key="optimizer_shrinkage",
        help="Intensity of the pull from the sample covariance toward a scaled "
        "identity matrix (same average variance, zero correlation). Higher "
        "means more stable weights and less faith in the sample "
        "correlations.",
    )
    method = right.selectbox(
        "Expected returns",
        options=("historical", "mean"),
        index=0,
        format_func=lambda key: _METHOD_LABELS.get(key, key),
        key="optimizer_mu_method",
        help="Historical compounds the realised path and annualises it "
        "geometrically. Arithmetic mean is mean(daily) x 252; it sits "
        "roughly sigma^2 / 2 above the geometric figure and is the "
        "internally consistent input if a mean-variance objective is read "
        "literally.",
    )

    return _Controls(
        max_weight=float(cap) / 100.0,
        shrinkage=float(shrinkage) / 100.0,
        method=str(method),
    )


# ---------------------------------------------------------------------------
# 2. the solve (cached against the controls)
# ---------------------------------------------------------------------------
def _current_weights(positions: Any) -> dict[str, float]:
    """``{ticker: fraction}`` from the position weights, which are in percent."""
    if positions is None or len(positions) == 0:
        return {}
    columns = getattr(positions, "columns", [])
    if "ticker" not in columns or "weight_pct" not in columns:
        return {}
    weights: dict[str, float] = {}
    for ticker, weight_pct in zip(positions["ticker"], positions["weight_pct"]):
        value = _f(weight_pct)
        if math.isfinite(value):
            weights[str(ticker)] = value / 100.0
    return weights


def _fingerprint(
    asset_returns: Any, rf: float, controls: _Controls, current: dict[str, float]
) -> str:
    """Everything that would change the solution, as one comparable string."""
    columns = ",".join(str(c) for c in getattr(asset_returns, "columns", []))
    index = getattr(asset_returns, "index", None)
    span = ""
    if index is not None and len(index):
        span = f"{index[0]}~{index[-1]}~{len(index)}"
    weights = ";".join(f"{k}:{v:.8f}" for k, v in sorted(current.items()))
    return "|".join(
        [
            columns,
            span,
            weights,
            f"{rf:.8f}",
            f"{controls.max_weight:.4f}",
            f"{controls.shrinkage:.4f}",
            controls.method,
        ]
    )


def _score_current(
    current_weights: dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    warnings: list[str],
) -> OptimizedPortfolio | None:
    """Score the book the investor actually holds on the page's mu / Sigma."""
    if not current_weights:
        return None
    labels = [str(c) for c in cov.columns]
    known = set(labels)

    missing = sorted(t for t in current_weights if t not in known)
    if missing:
        warnings.append(
            "Current portfolio holds "
            + ", ".join(missing)
            + " with no usable price history - excluded from the comparison."
        )

    kept = {t: _f(w) for t, w in current_weights.items() if t in known and _finite(w)}
    total = float(sum(kept.values()))
    if not math.isfinite(total) or total <= 0.0:
        warnings.append("Current portfolio weights do not sum to anything positive.")
        return None

    expected, volatility, sharpe = opt.portfolio_performance(kept, mu, cov, rf)
    return OptimizedPortfolio(
        label=opt.LABEL_CURRENT,
        weights={label: kept.get(label, 0.0) / total for label in labels},
        expected_return=expected,
        volatility=volatility,
        sharpe=sharpe,
        converged=True,
        note="Weights normalised to 100% over the names with price history.",
    )


def _solve_with_controls(
    clean: pd.DataFrame, rf: float, current_weights: dict[str, float], controls: _Controls
) -> dict[str, Any]:
    """``optimize_all``'s contract, re-estimated under the page's own controls.

    ``optimize_all`` fixes the expected-return convention at *historical* and
    the shrinkage intensity at ``config.OPTIMIZER.shrinkage``. When either
    control is moved off those values the suite has to be re-solved on the new
    mu / Sigma; every objective, the frontier and the random cloud are still the
    library's own public solvers, so the two paths differ only in their inputs.
    """
    warnings: list[str] = [
        f"Estimates on this run use the page controls - "
        f"{_METHOD_LABELS.get(controls.method, controls.method).lower()} expected "
        f"returns and {fmt_ratio_pct(controls.shrinkage, 0)} covariance shrinkage - "
        f"not the pipeline defaults (historical, "
        f"{fmt_ratio_pct(OPTIMIZER.shrinkage, 0)}). Every number below moves with them."
    ]
    n_obs, n_assets = int(clean.shape[0]), int(clean.shape[1])

    empty: dict[str, Any] = {
        "mu": pd.Series(dtype=float),
        "cov": pd.DataFrame(dtype=float),
        "frontier": pd.DataFrame(dtype=float),
        "random": pd.DataFrame(columns=list(opt.FRONTIER_COLUMNS), dtype=float),
        "portfolios": {},
        "current": None,
        "n_assets": n_assets,
        "warnings": warnings,
    }
    if n_assets == 0 or n_obs < 2:
        warnings.append(
            "Not enough overlapping return history to estimate a covariance "
            "matrix - optimisation skipped."
        )
        return empty

    if n_obs < opt.MIN_OBSERVATIONS:
        warnings.append(
            f"Only {n_obs} overlapping observations (below the "
            f"{opt.MIN_OBSERVATIONS} minimum): every estimate here is noisy."
        )
    if n_obs < opt.COVARIANCE_OBSERVATION_RATIO * n_assets:
        warnings.append(
            f"{n_obs} observations for {n_assets} assets is below the "
            f"{opt.COVARIANCE_OBSERVATION_RATIO:.0f}x rule of thumb - the sample "
            "covariance is unstable and the optimiser will fit its own noise."
        )

    mu = opt.expected_returns(clean, method=controls.method)
    cov = opt.covariance(clean, shrinkage=controls.shrinkage)
    if cov.empty:
        warnings.append("Covariance matrix could not be estimated.")
        return empty
    mu = mu.reindex(cov.columns)

    cap = controls.max_weight
    portfolios: dict[str, OptimizedPortfolio] = {
        opt.LABEL_MAX_SHARPE: opt.max_sharpe(mu, cov, rf, max_weight=cap),
        opt.LABEL_MIN_VARIANCE: opt.min_variance(mu, cov, rf, max_weight=cap),
        opt.LABEL_RISK_PARITY: opt.risk_parity(cov, rf, mu=mu, max_weight=cap),
        opt.LABEL_MAX_DIVERSIFICATION: opt.max_diversification(
            cov, mu=mu, risk_free_rate=rf, max_weight=cap
        ),
        opt.LABEL_EQUAL_WEIGHT: opt.equal_weight(mu, cov, rf),
    }

    seen: set[str] = set()
    for label, portfolio in portfolios.items():
        if portfolio.note and portfolio.note not in seen:
            seen.add(portfolio.note)
            warnings.append(portfolio.note)
        if not portfolio.converged and portfolio.weights:
            warnings.append(
                f"{label}: the solver did not fully converge; the best feasible "
                "weights found are shown."
            )

    return {
        "mu": mu,
        "cov": cov,
        "frontier": opt.efficient_frontier(mu, cov, rf, max_weight=cap),
        "random": opt.random_portfolios(mu, cov, rf),
        "portfolios": portfolios,
        "current": _score_current(current_weights, mu, cov, rf, warnings),
        "n_assets": n_assets,
        "warnings": warnings,
    }


def _solve(
    result: Any, cfg: Any, controls: _Controls, current_weights: dict[str, float]
) -> tuple[dict[str, Any], int]:
    """The solution for these controls, from session state where possible."""
    rf = _f(getattr(cfg, "risk_free_rate", 0.0))
    rf = rf if math.isfinite(rf) else 0.0
    key = _fingerprint(result.asset_returns, rf, controls, current_weights)

    cached = st.session_state.get(_SOLUTION_KEY)
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached["solution"], int(cached.get("n_obs", 0))

    clean = _complete_cases(result.asset_returns)
    n_obs = int(clean.shape[0])
    with ui.loading(
        "Solving five objectives, sweeping the efficient frontier and sampling the feasible set..."
    ):
        if controls.uses_pipeline_estimates:
            solution = opt.optimize_all(
                result.asset_returns,
                rf,
                current_weights=current_weights,
                max_weight=controls.max_weight,
            )
        else:
            solution = _solve_with_controls(clean, rf, current_weights, controls)

    st.session_state[_SOLUTION_KEY] = {"key": key, "solution": solution, "n_obs": n_obs}
    return solution, n_obs


# ---------------------------------------------------------------------------
# 3. solver notes
# ---------------------------------------------------------------------------
def _solver_notes(res: dict[str, Any]) -> None:
    portfolios: dict[str, Any] = res.get("portfolios") or {}
    chips: list[tuple[str, str]] = []
    for label, portfolio in portfolios.items():
        if not getattr(portfolio, "converged", True):
            chips.append((f"{label}: solver did not converge", "loss"))
        note = str(getattr(portfolio, "note", "") or "").strip()
        if note:
            chips.append((f"{label}: {_truncate(note)}", "warn"))
    if portfolios and not chips:
        chips.append((f"All {len(portfolios)} objectives converged", "gain"))
    if chips:
        ui.badges(chips)

    ui.warning_list(res.get("warnings") or [], "Optimiser notes")


# ---------------------------------------------------------------------------
# 4. headline KPIs
# ---------------------------------------------------------------------------
def _headline(res: dict[str, Any], result: Any, cfg: Any, n_obs: int) -> None:
    portfolios: dict[str, Any] = res.get("portfolios") or {}
    current = res.get("current")
    tangency = portfolios.get(opt.LABEL_MAX_SHARPE)
    minvar = portfolios.get(opt.LABEL_MIN_VARIANCE)

    held = len(result.positions) if result.positions is not None else 0
    n_assets = int(res.get("n_assets", 0) or 0)

    current_sharpe = _f(getattr(current, "sharpe", float("nan")))
    tangency_sharpe = _f(getattr(tangency, "sharpe", float("nan")))
    current_vol = _f(getattr(current, "volatility", float("nan")))
    minvar_vol = _f(getattr(minvar, "volatility", float("nan")))

    sharpe_gap = tangency_sharpe - current_sharpe
    vol_gap = minvar_vol - current_vol

    cards: list[dict[str, Any]] = [
        {
            "label": "Universe optimised",
            "value": f"{n_assets}",
            "tone": "accent",
            "footnote": (f"of {held} priced holdings, over {n_obs:,} complete-case sessions"),
            "help_text": (
                "Names with usable return history on the dates every holding "
                "traded. A name with no overlap cannot enter a covariance matrix "
                "and is excluded rather than assumed flat."
            ),
        },
        {
            "label": "Current Sharpe (model)",
            "value": fmt_num(current_sharpe, 2),
            "tone": delta_tone(current_sharpe),
            "footnote": (
                f"Realised Sharpe on the value series is {fmt_num(result.risk.get('sharpe'), 2)}"
            ),
            "help_text": (
                "Your live weights scored on this page's estimated expected "
                "returns and shrunk covariance - not the same estimator as the "
                "realised Sharpe on the Overview page."
            ),
        },
        {
            "label": "Tangency Sharpe",
            "value": fmt_num(tangency_sharpe, 2),
            "tone": "neutral",
            "delta": (f"{fmt_num(sharpe_gap, 2)} above current" if _finite(sharpe_gap) else None),
            "delta_tone": delta_tone(sharpe_gap),
            "footnote": "Best in-sample reward per unit of total volatility",
            "help_text": (
                "The Max Sharpe portfolio under the current weight cap. It is "
                "the single most estimation-sensitive number on this page."
            ),
        },
        {
            "label": "Min-variance volatility",
            "value": fmt_ratio_pct(minvar_vol),
            "tone": "neutral",
            "delta": (
                f"{fmt_ratio_pct(abs(vol_gap))} {'below' if vol_gap < 0 else 'above'} current"
                if _finite(vol_gap)
                else None
            ),
            "delta_tone": delta_tone(vol_gap, invert=True),
            "footnote": (f"Current book runs at {fmt_ratio_pct(current_vol)} annualised"),
            "help_text": (
                "Lowest achievable annualised volatility for a long-only book "
                "under the cap. Needs no return forecast at all, which is why "
                "it is the most trustworthy number here."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    if _finite(current_sharpe) and _finite(result.risk.get("sharpe")):
        st.caption(
            f"Two Sharpe ratios, deliberately: **{fmt_num(current_sharpe, 2)}** is "
            f"today's weights scored on the optimiser's estimates, and "
            f"**{fmt_num(result.risk.get('sharpe'), 2)}** is what the portfolio "
            f"value series actually delivered. They differ because the first uses "
            f"a shrunk covariance and a per-asset expected-return vector, while "
            f"the second measures the realised path - and because the realised "
            f"path carries whatever the weights were, not today's. Neither is a "
            f"forecast; both describe these {n_obs:,} sessions."
        )


# ---------------------------------------------------------------------------
# 5. the frontier
# ---------------------------------------------------------------------------
def _frontier_points(frontier: Any) -> pd.DataFrame | None:
    """The frontier reduced to two clean numeric columns, or ``None``."""
    if frontier is None or not isinstance(frontier, pd.DataFrame) or frontier.empty:
        return None
    if "return" not in frontier.columns or "volatility" not in frontier.columns:
        return None
    points = pd.DataFrame(
        {
            "return": pd.to_numeric(frontier["return"], errors="coerce"),
            "volatility": pd.to_numeric(frontier["volatility"], errors="coerce"),
        }
    ).dropna()
    return points if not points.empty else None


def _closest(points: pd.DataFrame, target: float, column: str) -> pd.Series:
    """Frontier row whose ``column`` sits nearest ``target``."""
    return points.loc[(points[column] - target).abs().idxmin()]


def _money_note(saving: float, value: float, cfg: Any) -> str:
    """The volatility difference translated into money, or nothing."""
    if not (_finite(saving) and _finite(value)) or saving <= 0:
        return ""
    return (
        f" On today's {fmt_money(value, cfg.base_currency, 0)} book that is about "
        f"{fmt_money(saving * value, cfg.base_currency, 0)} less "
        f"one-standard-deviation annual movement."
    )


def _efficiency_gap(res: dict[str, Any], cfg: Any, result: Any, cap: float) -> None:
    """The page's central sentence: what the frontier says is being wasted.

    Three genuinely different situations, and they must not be described in the
    same words. The current weights are **not** bound by the per-name cap the
    frontier respects, so the book can sit below the frontier's left-hand anchor
    (the ordinary, dominated case), inside its span (the like-for-like case), or
    beyond its right-hand end (a constraint mismatch, not out-performance).
    """
    current = res.get("current")
    if current is None:
        st.caption(
            "No current portfolio could be scored against the frontier, so there "
            "is no efficiency gap to quote - see the optimiser notes above."
        )
        return

    cur_ret = _f(current.expected_return)
    cur_vol = _f(current.volatility)
    points = _frontier_points(res.get("frontier"))
    if points is None or not (_finite(cur_ret) and _finite(cur_vol)):
        st.caption(
            "The frontier came back empty for this universe, so the current "
            "portfolio cannot be placed against it."
        )
        return

    lo, hi = float(points["return"].min()), float(points["return"].max())
    value = _f(result.summary.get("current_value"))

    # -- below the frontier's left-hand anchor: the ordinary dominated case ---
    if cur_ret < lo - _RETURN_MATCH_TOL:
        anchor = points.loc[points["return"].idxmin()]
        anchor_ret, anchor_vol = float(anchor["return"]), float(anchor["volatility"])
        saving = cur_vol - anchor_vol
        uplift = anchor_ret - cur_ret
        if saving > 0:
            relative = saving / cur_vol if cur_vol > 0 else float("nan")
            st.caption(
                f"**Dominated, in sample.** The book is priced at "
                f"{fmt_ratio_pct(cur_ret)} expected return on "
                f"{fmt_ratio_pct(cur_vol)} annualised volatility, which sits below "
                f"the left-hand end of the frontier altogether. The "
                f"minimum-variance portfolio - the frontier's leftmost point, and "
                f"the one that needs no return forecast at all - earned "
                f"{fmt_ratio_pct(anchor_ret)} on {fmt_ratio_pct(anchor_vol)}: "
                f"{fmt_ratio_pct(uplift)} more return for {fmt_ratio_pct(saving)} "
                f"less volatility, {fmt_ratio_pct(relative, 0)} of the risk removed."
                f"{_money_note(saving, value, cfg)} There is no like-for-like point "
                f"at the book's own expected return because the frontier does not "
                f"go that low - every efficient portfolio beat it on both axes over "
                f"this window. Strict dominance is the ordinary result when weights "
                f"were set by conviction or cost basis rather than by the "
                f"covariance structure, and it remains a hindsight statement about "
                f"one sample."
            )
        else:
            st.caption(
                f"The book earns {fmt_ratio_pct(cur_ret)} on "
                f"{fmt_ratio_pct(cur_vol)} volatility, below the frontier's "
                f"left-hand end ({fmt_ratio_pct(anchor_ret)} at "
                f"{fmt_ratio_pct(anchor_vol)}) on return but under it on risk too. "
                f"Running less volatility than any capped long-only portfolio the "
                f"optimiser can build means the current weights themselves breach "
                f"the {fmt_ratio_pct(cap, 0)} per-name cap - largest position "
                f"{fmt_ratio_pct(_largest_weight(current), 0)} - so the two are not "
                f"being solved over the same opportunity set."
            )
        return

    # -- beyond the frontier's right-hand end: a constraint mismatch ----------
    if cur_ret > hi + _RETURN_MATCH_TOL:
        largest = _largest_weight(current)
        st.caption(
            f"No like-for-like frontier point exists: the book's estimated "
            f"expected return of {fmt_ratio_pct(cur_ret)} is higher than anything "
            f"the {fmt_ratio_pct(cap, 0)} per-name cap allows, and the frontier "
            f"tops out at {fmt_ratio_pct(hi)}. That is a constraint mismatch, not "
            f"out-performance - the current weights are not cap-compliant (largest "
            f"position {fmt_ratio_pct(largest, 0)}), so they reach into a region "
            f"the constrained frontier is forbidden to visit. Raise the cap to at "
            f"least {fmt_ratio_pct(largest, 0)} to compare them honestly."
        )
        return

    # -- inside the frontier's span: the like-for-like comparison -------------
    match = _closest(points, cur_ret, "return")
    match_ret, match_vol = float(match["return"]), float(match["volatility"])
    saving = cur_vol - match_vol
    if saving <= 0:
        st.caption(
            f"At {fmt_ratio_pct(cur_ret)} expected return the book already runs at "
            f"{fmt_ratio_pct(cur_vol)} volatility against the frontier's "
            f"{fmt_ratio_pct(match_vol)} - it sits on, or inside, the constrained "
            f"frontier. Read that as a constraint artefact rather than a victory: "
            f"the current weights are not bound by the {fmt_ratio_pct(cap, 0)} cap "
            f"every frontier point respects, so they are being measured against a "
            f"strictly smaller opportunity set."
        )
        return

    relative = saving / cur_vol if cur_vol > 0 else float("nan")
    st.caption(
        f"**The gap, stated plainly.** The book is priced at "
        f"{fmt_ratio_pct(cur_ret)} expected return and {fmt_ratio_pct(cur_vol)} "
        f"annualised volatility. The frontier point at the same expected return "
        f"({fmt_ratio_pct(match_ret)}) carries {fmt_ratio_pct(match_vol)} - so "
        f"{fmt_ratio_pct(saving)} of volatility, {fmt_ratio_pct(relative, 0)} of "
        f"the risk being run, bought nothing over this window."
        f"{_money_note(saving, value, cfg)} That is the size of the *in-sample* "
        f"inefficiency, measured with hindsight on one sample. It is not a saving "
        f"you can bank."
    )

    uplift_point = _closest(points, cur_vol, "volatility")
    extra = float(uplift_point["return"]) - cur_ret
    if extra > 0:
        st.caption(
            f"Read the other way round - holding risk constant instead of return - "
            f"the frontier point at {fmt_ratio_pct(float(uplift_point['volatility']))} "
            f"volatility earned {fmt_ratio_pct(float(uplift_point['return']))}, "
            f"{fmt_ratio_pct(extra)} more than the book. Same sample, same "
            f"hindsight, same caveat."
        )


def _frontier(res: dict[str, Any], cfg: Any, controls: _Controls, result: Any) -> None:
    ui.chart(
        ch.efficient_frontier_chart(
            res.get("frontier"),
            res.get("random"),
            res.get("portfolios"),
            res.get("current"),
            _f(getattr(cfg, "risk_free_rate", 0.0)),
            height=560,
        )
    )

    n_assets = int(res.get("n_assets", 0) or 0)
    min_names = math.ceil(1.0 / controls.max_weight) if controls.max_weight > 0 else 0
    cap_pct = fmt_ratio_pct(controls.max_weight, 0)
    if min_names <= 1:
        cap_note = (
            f"A {cap_pct} cap does not constrain the mix at all - one name could "
            f"legally hold the entire book."
        )
    elif min_names >= n_assets:
        cap_note = (
            f"A {cap_pct} cap forces every one of the {n_assets} names into every "
            f"solution, so the optimiser is choosing between a very small set of "
            f"feasible mixes."
        )
    else:
        cap_note = (
            f"A {cap_pct} cap forces at least {min_names} of the {n_assets} names "
            f"into every solution."
        )

    st.caption(
        f"The faint cloud is {opt.DEFAULT_RANDOM_PORTFOLIOS:,} random long-only "
        f"portfolios drawn uniformly over the simplex - the feasible set, coloured "
        f"by Sharpe. The bold line is the efficient frontier: for each level of "
        f"expected return, the lowest volatility a long-only book capped at "
        f"{cap_pct} per name can achieve. The straight line from "
        f"{fmt_ratio_pct(getattr(cfg, 'risk_free_rate', 0.0))} through the tangency "
        f"portfolio is the capital market line - the risk/return menu you would "
        f"face if you could hold cash alongside the tangency mix. The amber star is "
        f"what you own today. {cap_note}"
    )

    _efficiency_gap(res, cfg, result, controls.max_weight)

    ui.methodology(
        "How mu, Sigma, the frontier and the cloud are built",
        f"""
**One estimate, shared by everything on the page.** The expected-return vector
`mu` and the covariance matrix `Sigma` are estimated once, on the
**complete cases** - the dates on which every asset in the universe traded - and
every portfolio, every frontier point and every random draw is then scored on
those same inputs. Estimating them per-objective would make the chart
internally inconsistent.

**Expected returns** ({_METHOD_LABELS.get(controls.method, controls.method)}):
`historical` compounds the realised daily path and annualises geometrically,
`mu_i = prod(1 + r_it) ** ({TRADING_DAYS} / n) - 1`; `mean` is the arithmetic
convention `mean(r_i) x {TRADING_DAYS}`, which sits roughly `sigma^2 / 2`
above it. An asset whose cumulative growth is non-positive has no real geometric
annualisation and returns a dash rather than a fabricated number.

**Covariance**: `S = {TRADING_DAYS} x cov(r)` on complete cases, then shrunk
toward a scaled identity with the same average variance and zero correlation,
`Sigma = (1 - a) S + a (trace(S) / n) I`, with `a =
{fmt_ratio_pct(controls.shrinkage, 0)}` from the slider above. Annualising by
{TRADING_DAYS} assumes daily returns are serially uncorrelated.

**The frontier** is a sequence of `min w' Sigma w` solves subject to
`w . mu >= target`, `sum(w) = 1`, `0 <= w_i <=
{fmt_ratio_pct(controls.max_weight, 0)}`, warm-started from the previous point.
The sweep runs from the minimum-variance portfolio to the highest return the cap
allows, **not** to `max(mu)` - a capped long-only book generally cannot reach
the best single asset's return, and plotting infeasible points would be a lie.

**The cloud** is Dirichlet(alpha = 1) sampling of the long-only simplex. It shows
the shape of the feasible set; it is not a probability distribution over anything
and its density near the frontier is an artefact of the sampler, not evidence.

**The capital market line** assumes you can borrow and lend freely at
{fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))}. Most investors cannot
borrow anywhere near the risk-free rate, so treat the segment to the right of
the tangency point as geometry rather than an available portfolio.
        """,
    )


# ---------------------------------------------------------------------------
# 6. strategy comparison table
# ---------------------------------------------------------------------------
def _comparison_frame(res: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    entries: list[tuple[str, Any]] = []

    current = res.get("current")
    if current is not None:
        entries.append(("Current book", current))
    portfolios: dict[str, Any] = res.get("portfolios") or {}
    for label in opt.PORTFOLIO_LABELS:
        if label in portfolios:
            entries.append((label, portfolios[label]))
    for label, portfolio in portfolios.items():
        if label not in opt.PORTFOLIO_LABELS:
            entries.append((label, portfolio))

    for label, portfolio in entries:
        converged = bool(getattr(portfolio, "converged", True))
        rows.append(
            {
                "Portfolio": label,
                "Expected return": fmt_ratio_pct(getattr(portfolio, "expected_return", None)),
                "Volatility": fmt_ratio_pct(getattr(portfolio, "volatility", None)),
                "Sharpe": fmt_num(getattr(portfolio, "sharpe", None), 2),
                "Effective bets": fmt_num(_effective_bets(portfolio), 1),
                "Largest weight": fmt_ratio_pct(_largest_weight(portfolio), 1),
                "Names held": f"{_names_held(portfolio)}",
                "Solver": (
                    "scored, not solved"
                    if label == "Current book"
                    else ("converged" if converged else "did not converge")
                ),
            }
        )
    return pd.DataFrame(rows)


def _comparison(res: dict[str, Any], cfg: Any, controls: _Controls) -> None:
    frame = _comparison_frame(res)
    if frame.empty:
        ui.empty_state(
            "No portfolios were solved",
            "Every objective came back empty for this universe, so there is nothing to compare.",
            icon="\U0001f9ee",
        )
        return

    ui.dataframe(frame, height=min(38 * (len(frame) + 1) + 10, 420))

    portfolios: dict[str, Any] = res.get("portfolios") or {}
    current = res.get("current")
    equal = portfolios.get(opt.LABEL_EQUAL_WEIGHT)
    tangency = portfolios.get(opt.LABEL_MAX_SHARPE)

    bits: list[str] = []
    if current is not None and equal is not None:
        beat = _f(equal.sharpe) - _f(current.sharpe)
        if _finite(beat):
            verdict = (
                f"beats the current book by {fmt_num(abs(beat), 2)} Sharpe"
                if beat > 0
                else f"trails the current book by {fmt_num(abs(beat), 2)} Sharpe"
            )
            bits.append(
                f"Equal weight - no estimation, no solver, no view - {verdict}. "
                f"It is the benchmark every optimiser on this page has to beat, and "
                f"out of sample it very often wins"
            )
    largest = _largest_weight(tangency) if tangency is not None else float("nan")
    if _finite(largest):
        universe = int(res.get("n_assets", 0) or 0)
        if controls.max_weight >= 0.999:
            bits.append(
                f"With the cap effectively switched off, Max Sharpe puts "
                f"{fmt_ratio_pct(largest, 0)} into a single name across "
                f"{_names_held(tangency)} of {universe} - which is exactly what an "
                f"unconstrained mean-variance objective does once it believes its "
                f"own return estimates"
            )
        elif largest >= controls.max_weight - 1e-6:
            bits.append(
                f"Max Sharpe pins its largest position exactly on the "
                f"{fmt_ratio_pct(controls.max_weight, 0)} cap - the tell that the "
                f"objective wanted more of that name and only the constraint "
                f"stopped it, which is precisely what the cap is there for"
            )
        else:
            bits.append(
                f"Max Sharpe holds {_names_held(tangency)} of {universe} names, its "
                f"largest at {fmt_ratio_pct(largest, 0)}, without hitting the "
                f"{fmt_ratio_pct(controls.max_weight, 0)} cap"
            )
    if bits:
        st.caption(". ".join(bits) + ".")

    ui.methodology(
        "What each objective actually maximises",
        f"""
All five are solved long-only (`w_i >= 0`), fully invested (`sum(w) = 1`) and
capped per name, on the shared mu / Sigma described above. Read every figure as
a **decimal annualised** quantity.

* **Max Sharpe** maximises `(w . mu - rf) / sqrt(w' Sigma w)` with
  `rf = {fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))}`. It is the tangency
  portfolio and it depends on `mu` more than anything else here, which makes it
  the least stable output on the page.
* **Min Variance** minimises `w' Sigma w`. **It never looks at `mu` at all** - no
  return forecast enters it, which is exactly why its weights move far less when
  the sample changes.
* **Risk Parity** equalises each name's contribution to portfolio volatility,
  `w_i (Sigma w)_i` equal across i. Also forecast-free. It systematically
  overweights low-volatility names, which is a real exposure, not a free lunch.
* **Max Diversification** maximises `(w . sigma) / sqrt(w' Sigma w)`, the ratio of
  weighted-average asset volatility to portfolio volatility. Forecast-free again,
  and it targets correlation structure rather than variance.
* **Equal Weight** is `1/N`. It has no parameters to estimate and therefore no
  estimation error - the reason it is such a stubborn benchmark.

**Effective bets** is `1 / sum(w_i^2)`: how many equally-weighted positions would
carry this much concentration. **Names held** counts weights of at least 0.1%;
anything below that is solver dust, not a position. *Scored, not solved* on the
current book means its weights were taken as given and priced on the same
mu / Sigma - no optimisation was applied to it.
        """,
    )


# ---------------------------------------------------------------------------
# 7. target allocation and the hand-off to Rebalance
# ---------------------------------------------------------------------------
def _target_frame(
    target_weights: dict[str, float], current_weights: dict[str, float]
) -> pd.DataFrame:
    tickers = sorted(set(target_weights) | set(current_weights))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        target_pct = target_weights.get(ticker, 0.0) * 100.0
        current_pct = current_weights.get(ticker, 0.0) * 100.0
        change = target_pct - current_pct
        if abs(change) < _DUST_PP:
            action = "Hold"
        else:
            action = "Buy" if change > 0 else "Sell"
        rows.append(
            {
                "Ticker": ticker,
                "Target %": target_pct,
                "Current %": current_pct,
                "Change (pp)": change,
                "Action": action,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("Change (pp)", key=lambda s: s.abs(), ascending=False)


def _turnover_pct(frame: pd.DataFrame) -> float:
    """One-way turnover: half the sum of absolute weight changes, in percent."""
    if frame.empty or "Change (pp)" not in frame.columns:
        return float("nan")
    return float(frame["Change (pp)"].abs().sum()) / 2.0


def _strategy_picker(res: dict[str, Any], current_weights: dict[str, float], cfg: Any) -> None:
    portfolios: dict[str, Any] = res.get("portfolios") or {}
    if not portfolios:
        return

    labels = [label for label in opt.PORTFOLIO_LABELS if label in portfolios]
    labels += [label for label in portfolios if label not in labels]
    default_index = labels.index(_DEFAULT_STRATEGY) if _DEFAULT_STRATEGY in labels else 0

    chosen = st.selectbox(
        "Target strategy",
        options=labels,
        index=default_index,
        key="optimizer_strategy",
        help="Minimum variance is pre-selected on purpose: it is the only "
        "objective on this list that needs no return forecast and it "
        "therefore survives a change of sample better than the rest.",
    )
    portfolio = portfolios[chosen]
    target_weights = _weights_of(portfolio)

    left, right = st.columns([1.0, 1.3], gap="large")

    with left:
        ui.chart(ch.weights_donut(target_weights, title=str(chosen), height=380))

    with right:
        frame = _target_frame(target_weights, current_weights)
        ui.dataframe(
            frame,
            height=min(38 * (len(frame) + 1) + 10, 380),
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Target %": st.column_config.NumberColumn(
                    "Target",
                    format="%.2f%%",
                    help="What this objective would hold, as a share of the book.",
                ),
                "Current %": st.column_config.NumberColumn(
                    "Current",
                    format="%.2f%%",
                    help="Today's market-value weight from the positions table.",
                ),
                "Change (pp)": st.column_config.NumberColumn(
                    "Change",
                    format="%+.2f",
                    help="Target minus current, in percentage points of the book.",
                ),
                "Action": st.column_config.TextColumn("Action", width="small"),
            },
        )

        turnover = _turnover_pct(frame)
        buys = int((frame["Action"] == "Buy").sum()) if not frame.empty else 0
        sells = int((frame["Action"] == "Sell").sum()) if not frame.empty else 0
        st.caption(
            f"Reaching **{chosen}** means turning over "
            f"{fmt_pct(turnover)} of the book one way - {buys} name(s) bought, "
            f"{sells} sold, everything within {fmt_num(_DUST_PP, 2)} percentage "
            f"points left alone as rounding rather than a trade. One-way turnover "
            f"is half the sum of "
            f"the absolute weight changes; costs, taxes and market impact are not "
            f"modelled anywhere on this page, and in a taxable account they "
            f"routinely exceed the in-sample gain shown above."
        )

        if st.button(
            f"Send {chosen} to Rebalance",
            type="primary",
            key="optimizer_send_targets",
            width="stretch",
        ):
            targets = rb.targets_from_optimizer(target_weights)
            if not targets:
                st.warning("This portfolio has no positive weights to send - nothing was written.")
            else:
                st.session_state[session.TARGET_WEIGHTS] = targets
                st.success(
                    f"{chosen} stored as the target allocation: {len(targets)} "
                    f"name(s), summing to {fmt_pct(sum(targets.values()), 1)}. "
                    f"Open **Strategy > Rebalance** in the sidebar to see drift "
                    f"bands and the executable order list."
                )

    ui.methodology(
        "What 'target weights' commit you to",
        """
Weights leave this page as **percentages** summing to exactly 100. Dust below
0.0001% of the book and any negative weight are dropped first and the remainder
renormalised, so a target list never contains a position too small to trade.

**Change** is in *percentage points of the portfolio*, not a percentage change in
the position: moving a name from 4% to 6% is `+2.00` here and a 50% increase in
the holding. **One-way turnover** is `sum(|change|) / 2` = the fraction of the
book that changes hands once, which is the number brokerage and impact costs
scale with.

**Not modelled:** brokerage, STT, stamp duty, exchange fees, bid/ask spread,
market impact, and - the big one in a taxable Indian account - capital gains
crystallised on every sale. A rebalance that looks worth a percentage point a
year *in sample* can easily be net negative after those. The Rebalance page
applies a drift band precisely so that small, statistically meaningless
deviations do not generate trades.

**Cash is not an asset here.** Every objective is fully invested by construction
(`sum(w) = 1`), so "hold more cash" is never among the answers, however sensible
it might be.
        """,
    )


# ---------------------------------------------------------------------------
# 8. weight structure
# ---------------------------------------------------------------------------
def _weight_views(res: dict[str, Any], current_weights: dict[str, float]) -> None:
    portfolios: dict[str, Any] = res.get("portfolios") or {}
    if not portfolios:
        return

    by_label: dict[str, Any] = {}
    current = res.get("current")
    if current is not None:
        by_label["Current book"] = _weights_of(current)
    elif current_weights:
        by_label["Current book"] = dict(current_weights)
    for label in opt.PORTFOLIO_LABELS:
        if label in portfolios:
            by_label[label] = _weights_of(portfolios[label])

    across_tab, along_tab = st.tabs(["Across strategies", "Along the frontier"])

    with across_tab:
        ui.chart(ch.weights_comparison(by_label))
        st.caption(
            "One bar group per holding, one bar per objective. Names where every "
            "bar is roughly the same height are agreed on by objectives that use "
            "completely different information - that agreement is the closest "
            "thing to a robust signal this page produces. A name that is large in "
            "exactly one strategy is that objective's artefact, not a conclusion."
        )

    with along_tab:
        ui.chart(ch.frontier_weights_area(res.get("frontier")))
        st.caption(
            "Left to right is a walk from the minimum-variance portfolio to the "
            "highest return the cap allows, and each vertical slice is one fully "
            "invested long-only book. A smooth reallocation means the frontier is "
            "well conditioned; a sharp step means the solver is flipping between "
            "near-identical solutions, which is what estimation error looks like "
            "when you can see it."
        )


# ---------------------------------------------------------------------------
# 9. the candid methodology
# ---------------------------------------------------------------------------
def _estimate_error_table(res: dict[str, Any], n_obs: int, limit: int = 12) -> str:
    """Markdown table of mu against the standard error of its own estimate."""
    mu = res.get("mu")
    cov = res.get("cov")
    if mu is None or cov is None or len(getattr(mu, "index", [])) == 0:
        return ""
    if not isinstance(cov, pd.DataFrame) or cov.empty:
        return ""

    years = n_obs / float(TRADING_DAYS)
    if years <= 0:
        return ""

    lines = [
        "| Asset | Expected return | Annualised vol | Std error of that estimate | t |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tickers = [str(t) for t in cov.columns][:limit]
    for ticker in tickers:
        expected = _f(mu.get(ticker)) if hasattr(mu, "get") else float("nan")
        variance = _f(cov.loc[ticker, ticker]) if ticker in cov.columns else float("nan")
        sigma = math.sqrt(variance) if math.isfinite(variance) and variance >= 0 else float("nan")
        stderr = sigma / math.sqrt(years) if math.isfinite(sigma) else float("nan")
        t_stat = (
            expected / stderr
            if math.isfinite(expected) and math.isfinite(stderr) and stderr > 0
            else float("nan")
        )
        lines.append(
            f"| {ticker} | {fmt_ratio_pct(expected)} | {fmt_ratio_pct(sigma)} "
            f"| {fmt_ratio_pct(stderr)} | {fmt_num(t_stat, 2)} |"
        )
    remaining = len(cov.columns) - len(tickers)
    if remaining > 0:
        lines.append(f"| *and {remaining} more* | | | | |")
    return "\n".join(lines)


def _candid_methodology(res: dict[str, Any], controls: _Controls, n_obs: int, cfg: Any) -> None:
    table = _estimate_error_table(res, n_obs)
    years = n_obs / float(TRADING_DAYS) if n_obs else float("nan")
    cap_pct = fmt_ratio_pct(controls.max_weight, 0)

    evidence = (
        f"""
Here is the arithmetic, on your own data. The standard error of an annualised
mean return estimated from {n_obs:,} daily observations ({fmt_num(years, 1)}
years) is roughly `sigma / sqrt(T_years)`:

{table}

Look at the `t` column before you believe anything on this page. A t below
about 2 means the expected return is statistically indistinguishable from zero -
and the optimiser has just built a portfolio by ranking those numbers against
each other.
"""
        if table
        else ""
    )

    ui.methodology(
        "Read this before you act on any of it",
        f"""
**Mean-variance optimisation is an error maximiser.** That is not a criticism
from outside the field; it is the standard result (Michaud, 1989). The optimiser
takes `mu` and `Sigma` as if they were known, and then does exactly what it was
asked to do: it pours capital into whichever asset had the best estimated return
per unit of estimated risk. Since the estimates are noisy, the assets with the
most *flattering estimation error* are the ones that get overweighted. The
objective systematically buys the noise.

**Expected returns are the weak link, by an enormous margin.** Covariances are
estimable from daily data; mean returns are not. The standard error of a mean
return falls with `sqrt(T)`, so halving it takes four times the history - and a
20% volatility asset needs on the order of a century of data before its
annualised mean is pinned to within a percentage point. Nothing about using a
longer window fixes this within any horizon that is also *relevant*.
{evidence}
**Small input changes swing the weights.** Two assets with similar risk and a
high correlation are nearly interchangeable to the objective, so a hair's
difference in their estimated returns can flip the allocation from one to the
other entirely. That is why the frontier weight bands sometimes step rather than
glide, and why re-running this page next month with one more month of data can
produce a visibly different portfolio.

**What this page does about it, and what it cannot do.**

* *Shrinkage* ({fmt_ratio_pct(controls.shrinkage, 0)} here) pulls the sample
  covariance toward a scaled identity - same average variance, zero correlation.
  It deliberately biases the estimate in exchange for a large reduction in
  variance, which is a trade worth making when the sample matrix is close to
  singular. With {int(res.get("n_assets", 0) or 0)} assets and {n_obs:,}
  observations you want *more* shrinkage, not less, when that ratio is tight.
  Shrinkage cannot manufacture data.
* *The {cap_pct} per-name cap* is a blunt instrument and that is the point. It
  is a prior - "no single name deserves more than {cap_pct} of the book whatever
  the sample says" - imposed because the objective has no such prior of its own.
  Remove it and watch the tangency portfolio collapse into two or three names.
* Neither addresses the fundamental problem, which is that `mu` is nearly
  unknowable.

**So trust the forecast-free outputs more.** **Min Variance**, **Risk Parity**
and **Max Diversification** never read `mu` at all. They use only the covariance
structure - the part of the problem that daily data can actually speak to - and
their weights are correspondingly far more stable across samples. **Equal
Weight** estimates nothing whatsoever and has beaten optimised portfolios out of
sample often enough to be a serious benchmark rather than a straw man. **Max
Sharpe** is the most eye-catching point on the chart and the least reliable.

**Everything here is in-sample.** Every expected return, volatility and Sharpe on
this page is computed on the same {n_obs:,} sessions the weights were chosen
from. A portfolio optimised on a sample will always look good on that sample;
that is a tautology, not evidence. There is no walk-forward test, no
out-of-sample period and no transaction costs behind any number shown.

**This is not a recommendation.** It is a set of constrained optimisations over
one historical sample, presented so you can see what the mathematics implies and
where it is fragile. Position sizing that ignores your liquidity needs, tax
position, time horizon and conviction is arithmetic, not advice - and the
risk-free rate of {fmt_ratio_pct(getattr(cfg, "risk_free_rate", 0.0))} used
throughout is the one you typed into the sidebar, not a live yield.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Optimiser",
        "Where this book sits against the efficient frontier, what each objective "
        "would hold instead, and how much of that to believe",
        icon="\U0001f9ec",
    )

    result = guards.require_result()
    if result is None:
        return
    if not guards.require_multiple_assets(result, what="Portfolio optimisation", minimum=2):
        return
    if not guards.require_observations(result, what="Portfolio optimisation"):
        return

    cfg = result.config

    ui_theme.section("CONSTRAINTS AND ESTIMATORS", "\U0001f39b")
    controls = _controls(cfg)

    current_weights = _current_weights(result.positions)
    res, n_obs = _solve(result, cfg, controls, current_weights)

    _solver_notes(res)

    if not (res.get("portfolios") or {}):
        ui.empty_state(
            "The optimiser produced no solutions",
            "No objective could be solved on this universe - see the optimiser "
            "notes above for the reason.",
            icon="\U0001f9ee",
            hint="A longer history window, or dropping a holding with almost no "
            "overlapping price history, usually fixes it.",
        )
        return

    _headline(res, result, cfg, n_obs)

    ui_theme.spacer(0.5)
    ui_theme.section("THE EFFICIENT FRONTIER", "\U0001f4c8")
    _frontier(res, cfg, controls, result)

    ui_theme.spacer(0.4)
    ui_theme.section("OBJECTIVE BY OBJECTIVE", "\U00002696")
    _comparison(res, cfg, controls)

    ui_theme.spacer(0.4)
    ui_theme.section("TARGET ALLOCATION", "\U0001f3af")
    _strategy_picker(res, current_weights, cfg)

    ui_theme.spacer(0.4)
    ui_theme.section("WHAT EACH OBJECTIVE HOLDS", "\U0001f9e9")
    _weight_views(res, current_weights)

    ui_theme.spacer(0.4)
    ui_theme.section("BLACK-LITTERMAN", "\U00002696")
    st.caption(
        "Everything above is optimised on historical mean returns — the estimator "
        "the health warning below is about. Black-Litterman replaces it: start from "
        "the returns the market must already expect for current prices to be an "
        "equilibrium, then move them only as far as your own stated views justify."
    )
    bl_panel.render(result, historical_mu=res.get("mu"), max_weight=controls.max_weight)

    ui_theme.spacer(0.4)
    ui_theme.section("THE HONEST HEALTH WARNING", "\U0001f6a8")
    _candid_methodology(res, controls, n_obs, cfg)
