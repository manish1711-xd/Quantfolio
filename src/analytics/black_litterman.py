"""Black-Litterman: blend market-implied returns with explicit investor views.

The problem it solves
---------------------
Mean-variance optimisation is exquisitely sensitive to its expected-return
input, and historical means are a terrible estimator of it. Feed
:mod:`src.analytics.optimize` five years of realised averages and it will
happily concentrate the whole portfolio in whatever happened to run hardest --
an answer that says more about the sample than about the future. The optimiser
page in this application says so out loud, which is honest but not a fix.

Black-Litterman is the fix. It starts from the returns the *market* must
already expect for current prices to be an equilibrium, and moves them only as
far as the investor's stated views justify, in proportion to how confident
those views are. No view on an asset means its expected return stays at
equilibrium -- so the optimiser is no longer being handed noise it cannot
distinguish from signal.

The mathematics
---------------
**1. Reverse optimisation** turns weights into the returns that would make them
optimal::

    Pi = delta * Sigma * w_mkt

with ``delta`` the market's risk aversion, calibrated as
``delta = (E[r_mkt] - r_f) / sigma_mkt^2``. This step is exactly invertible:
feeding ``Pi`` into an unconstrained mean-variance optimiser at the same
``delta`` returns ``w_mkt`` to floating-point precision, which is the defining
property of equilibrium returns and is asserted in the test suite.

**2. Views** are stated as ``P E[R] = Q``, where each row of ``P`` picks the
assets a view is about. An absolute view ("A returns 12%") is a row with a
single 1. A relative view ("A beats B by 3%") is a row of ``+1, -1`` summing
to zero. ``Omega`` holds the uncertainty of each view.

**3. The posterior** blends them::

    E[R] = Pi + tau*Sigma*P' (P tau*Sigma P' + Omega)^-1 (Q - P Pi)

This is the Woodbury form, used here in preference to the textbook
``[(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]``
because it inverts only one k-by-k matrix (k = number of views, usually two or
three) instead of inverting both ``Omega`` and ``tau*Sigma``. ``Omega`` is
routinely near-singular -- a confident view drives its diagonal towards zero --
so the textbook form is a numerical trap. Both forms are algebraically
identical and the test suite asserts they agree, which is what justifies using
the faster, safer one.

**4. Posterior covariance** (He and Litterman)::

    M = tau*Sigma - tau*Sigma*P' (P tau*Sigma P' + Omega)^-1 P tau*Sigma
    Sigma_post = Sigma + M

``M`` is the covariance of the estimation error in the posterior mean; adding
it widens the covariance to reflect that the mean itself is uncertain.

Honest limits, stated up front
------------------------------
* **"The market" here is not the market.** True Black-Litterman uses the global
  cap-weighted market portfolio. A portfolio tool has the user's own holdings,
  so :func:`market_cap_weights` cap-weights *those*. The resulting ``Pi`` is
  "the equilibrium returns implied by a cap-weighted portfolio of your
  universe", which is a different and much narrower claim. It is the closest
  defensible thing available and this module never calls it anything else.
* **Views are the user's, and so are the consequences.** Nothing here
  generates, suggests or scores a view. The model's output is a restatement of
  the user's own opinions under a coherent set of assumptions.
* **tau has no agreed value.** 0.05 is a common choice and it matters: it sets
  how much the views move the answer. It is a named, configurable constant
  rather than a hidden literal.

Pure module: numpy/pandas only. No I/O, no network, no Streamlit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TAU",
    "DEFAULT_RISK_AVERSION",
    "DEFAULT_CONFIDENCE",
    "MIN_VARIANCE_FLOOR",
    "View",
    "BlackLittermanResult",
    "absolute_view",
    "relative_view",
    "implied_risk_aversion",
    "market_implied_returns",
    "reverse_optimise_weights",
    "market_cap_weights",
    "pick_matrix",
    "omega_he_litterman",
    "omega_from_confidence",
    "black_litterman",
    "posterior_master_formula",
]

#: Scales the uncertainty of the equilibrium prior. There is no agreed value:
#: the original paper is vague, He-Litterman use 0.025, Idzorek suggests
#: 0.01-0.05, and 1/n is also common. It controls how far the views can move
#: the answer, so it is exposed rather than buried.
DEFAULT_TAU: float = 0.05

#: Fallback market risk aversion when it cannot be calibrated from data.
#: ``delta`` around 2.5 corresponds to roughly a 6% equity premium at 15%
#: volatility, which is the textbook illustration.
DEFAULT_RISK_AVERSION: float = 2.5

#: Default view confidence (Idzorek scale). 0.5 reproduces He-Litterman's
#: ``Omega = diag(P tau Sigma P')`` exactly, which is the natural neutral point.
DEFAULT_CONFIDENCE: float = 0.5

#: Variances below this are treated as zero. Prevents a degenerate view (one
#: whose portfolio has no variance) from producing an infinite precision that
#: would silently dominate the blend.
MIN_VARIANCE_FLOOR: float = 1e-14

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


def _clean_covariance(cov: pd.DataFrame | None) -> tuple[list[str], np.ndarray]:
    """Symmetric, finite covariance matrix. ``([], empty)`` when unusable.

    Symmetrised as ``(S + S') / 2`` rather than trusted: a shrunk sample
    covariance can come back very slightly asymmetric from floating-point
    accumulation, and every inverse downstream assumes symmetry.
    """
    if cov is None or not isinstance(cov, pd.DataFrame) or cov.empty:
        return [], np.empty((0, 0))
    if cov.shape[0] != cov.shape[1]:
        return [], np.empty((0, 0))

    labels = [str(c) for c in cov.columns]
    if list(cov.index.astype(str)) != labels:
        return [], np.empty((0, 0))

    matrix = cov.to_numpy(dtype="float64", copy=True)
    if not np.all(np.isfinite(matrix)):
        return [], np.empty((0, 0))
    return labels, (matrix + matrix.T) / 2.0


def _align_weights(weights: object, labels: Sequence[str]) -> np.ndarray:
    """Weights in ``labels`` order, missing names as 0.0, normalised to sum 1."""
    if isinstance(weights, (pd.Series, Mapping)):
        mapping = {str(k): _to_float(v) for k, v in weights.items()}
    else:
        try:
            values = list(weights)  # type: ignore[arg-type]
        except TypeError:
            return np.array([])
        if len(values) != len(labels):
            return np.array([])
        mapping = {str(k): _to_float(v) for k, v in zip(labels, values)}

    vector = np.array([mapping.get(label, 0.0) for label in labels], dtype="float64")
    vector = np.where(np.isfinite(vector), vector, 0.0)
    total = float(vector.sum())
    if total <= 0.0:
        return np.array([])
    return vector / total


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ x = rhs``, falling back to a pseudo-inverse.

    ``np.linalg.solve`` is preferred (faster and better conditioned than
    forming an explicit inverse) but raises on a singular matrix, which a
    perfectly-confident pair of contradictory views can produce. The
    pseudo-inverse gives the least-squares answer there instead of an
    exception five layers below the UI.
    """
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        logger.warning("Black-Litterman system was singular; using a pseudo-inverse.")
        return np.linalg.pinv(matrix) @ rhs


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class View:
    """One investor view: ``sum(weights_i * E[R_i]) = expected_return``.

    ``weights`` summing to 0 makes this a RELATIVE view ("A beats B by 3%");
    summing to 1 makes it ABSOLUTE ("A returns 12%"). Both are legitimate and
    the distinction matters for interpretation, so it is a property rather
    than something the caller has to track.

    ``confidence`` is on Idzorek's 0-1 scale. 0.5 is the neutral point that
    reproduces He-Litterman's ``Omega``; 1.0 asks for the view to hold almost
    exactly; 0.0 means the view is ignored entirely.

    ``expected_return`` is an ANNUALISED EXCESS return, matching the
    equilibrium prior. Handing it a total return would tilt every posterior by
    the risk-free rate.
    """

    assets: tuple[str, ...]
    weights: tuple[float, ...]
    expected_return: float
    confidence: float = DEFAULT_CONFIDENCE
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", tuple(str(a).strip() for a in self.assets))
        object.__setattr__(self, "weights", tuple(_to_float(w) for w in self.weights))
        confidence = _to_float(self.confidence)
        if not math.isfinite(confidence):
            confidence = DEFAULT_CONFIDENCE
        object.__setattr__(self, "confidence", float(min(max(confidence, 0.0), 1.0)))

    @property
    def ok(self) -> bool:
        """Usable: same-length non-empty legs, finite numbers, some exposure."""
        if not self.assets or len(self.assets) != len(self.weights):
            return False
        if not all(math.isfinite(w) for w in self.weights):
            return False
        if not math.isfinite(self.expected_return):
            return False
        return any(abs(w) > 0.0 for w in self.weights)

    @property
    def is_relative(self) -> bool:
        return abs(sum(self.weights)) < 1e-9

    @property
    def description(self) -> str:
        if self.label:
            return self.label
        if len(self.assets) == 1:
            return f"{self.assets[0]} returns {self.expected_return:.2%}"
        longs = [a for a, w in zip(self.assets, self.weights) if w > 0]
        shorts = [a for a, w in zip(self.assets, self.weights) if w < 0]
        if self.is_relative and longs and shorts:
            return f"{'+'.join(longs)} beats {'+'.join(shorts)} by {self.expected_return:.2%}"
        return f"{len(self.assets)}-asset view: {self.expected_return:.2%}"


def absolute_view(
    asset: str,
    expected_return: float,
    confidence: float = DEFAULT_CONFIDENCE,
    label: str = "",
) -> View:
    """ "``asset`` will return ``expected_return``" (annualised excess)."""
    return View((asset,), (1.0,), expected_return, confidence, label)


def relative_view(
    outperformer: str,
    underperformer: str,
    outperformance: float,
    confidence: float = DEFAULT_CONFIDENCE,
    label: str = "",
) -> View:
    """ "``outperformer`` beats ``underperformer`` by ``outperformance``".

    The classic Black-Litterman view, and the more defensible kind: a relative
    call needs no opinion about the level of the market, only about the spread.
    """
    return View((outperformer, underperformer), (1.0, -1.0), outperformance, confidence, label)


def pick_matrix(
    views: Sequence[View], assets: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[View], list[str]]:
    """Build ``(P, Q, kept_views, warnings)`` for the given asset order.

    A view naming an asset outside the universe is DROPPED with a warning
    rather than silently ignored: a three-view blend that quietly became a
    two-view blend is a different model, and the user's missing opinion is the
    thing they would most want to know about.
    """
    labels = [str(a) for a in assets]
    position = {label: i for i, label in enumerate(labels)}

    rows: list[np.ndarray] = []
    targets: list[float] = []
    kept: list[View] = []
    warnings: list[str] = []

    for index, view in enumerate(views or (), start=1):
        name = view.label or f"view {index}"
        if not view.ok:
            warnings.append(f"{name}: incomplete or non-numeric, so it was dropped.")
            continue
        if view.confidence <= 0.0:
            warnings.append(f"{name}: confidence is zero, so it was dropped.")
            continue
        missing = [a for a in view.assets if a not in position]
        if missing:
            warnings.append(
                f"{name}: {', '.join(missing)} is not in the analysed universe, so the "
                "view was dropped. Only holdings with usable price history can carry a "
                "view."
            )
            continue

        row = np.zeros(len(labels), dtype="float64")
        for asset, weight in zip(view.assets, view.weights):
            row[position[asset]] += float(weight)
        if not np.any(np.abs(row) > 0.0):
            warnings.append(f"{name}: its legs cancel out exactly, so it says nothing.")
            continue
        rows.append(row)
        targets.append(float(view.expected_return))
        kept.append(view)

    if not rows:
        return np.empty((0, len(labels))), np.empty(0), [], warnings
    return np.vstack(rows), np.array(targets, dtype="float64"), kept, warnings


# ---------------------------------------------------------------------------
# equilibrium
# ---------------------------------------------------------------------------
def implied_risk_aversion(market_excess_return: float, market_variance: float) -> float:
    """``delta = (E[r_mkt] - r_f) / sigma_mkt^2``. NaN when uncalibratable.

    The market's aversion to risk, backed out of the premium it demands. NaN
    rather than a default is deliberate at this level: the caller decides
    whether to fall back to :data:`DEFAULT_RISK_AVERSION` and say so, which is
    better than this function quietly inventing an equilibrium.
    """
    premium = _to_float(market_excess_return)
    variance = _to_float(market_variance)
    if not math.isfinite(premium) or not math.isfinite(variance):
        return _NAN
    if variance <= MIN_VARIANCE_FLOOR:
        return _NAN
    return float(premium / variance)


def market_implied_returns(
    cov: pd.DataFrame | None,
    market_weights: object,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> pd.Series:
    """Reverse optimisation: ``Pi = delta * Sigma * w_mkt``.

    The annualised EXCESS returns that make ``market_weights`` mean-variance
    optimal. Everything Black-Litterman does starts here, and the step is
    exactly invertible -- see :func:`reverse_optimise_weights`.

    Empty series when the covariance or the weights are unusable.
    """
    labels, matrix = _clean_covariance(cov)
    if not labels:
        return pd.Series(dtype="float64")
    weights = _align_weights(market_weights, labels)
    delta = _to_float(risk_aversion)
    if weights.size == 0 or not math.isfinite(delta):
        return pd.Series(dtype="float64")
    return pd.Series(delta * (matrix @ weights), index=labels, name="equilibrium")


def reverse_optimise_weights(
    cov: pd.DataFrame | None,
    expected_excess_returns: object,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> pd.Series:
    """Unconstrained mean-variance weights: ``w = (delta * Sigma)^-1 mu``.

    The inverse of :func:`market_implied_returns`, and the identity that
    proves the equilibrium step is right: round-tripping market weights
    through ``Pi`` and back must return them unchanged. Unconstrained on
    purpose -- no long-only bound, no budget constraint -- because that is what
    makes the identity exact. For an investable portfolio use
    :mod:`src.analytics.optimize`, which applies real constraints.
    """
    labels, matrix = _clean_covariance(cov)
    if not labels:
        return pd.Series(dtype="float64")
    delta = _to_float(risk_aversion)
    if not math.isfinite(delta) or delta == 0.0:
        return pd.Series(dtype="float64")

    if isinstance(expected_excess_returns, (pd.Series, Mapping)):
        mu = np.array([_to_float(expected_excess_returns.get(label, _NAN)) for label in labels])
    else:
        mu = np.asarray(expected_excess_returns, dtype="float64")
    if mu.size != len(labels) or not np.all(np.isfinite(mu)):
        return pd.Series(dtype="float64")

    weights = _solve(delta * matrix, mu)
    return pd.Series(weights, index=labels, name="weights")


def market_cap_weights(
    positions: pd.DataFrame | None,
    profiles: Mapping[str, object] | None = None,
) -> tuple[pd.Series, list[str]]:
    """Cap weights for the holdings, from provider market caps.

    ``(weights, warnings)``. Reads ``market_cap`` from the positions frame if
    present, otherwise from the supplied profiles.

    **This is not the market portfolio.** Black-Litterman's equilibrium is the
    global cap-weighted market; this cap-weights the user's own holdings. The
    resulting prior is "the returns implied by a cap-weighted portfolio of your
    universe" -- narrower, and the closest defensible thing a portfolio tool
    has. Coverage is reported because Yahoo's market-cap field is patchy for
    Indian small caps and absent for most ETFs, and a cap weighting computed
    from half the book is not a cap weighting.
    """
    if positions is None or not isinstance(positions, pd.DataFrame) or positions.empty:
        return pd.Series(dtype="float64"), ["No positions supplied."]
    if "ticker" not in positions.columns:
        return pd.Series(dtype="float64"), ["Positions frame has no ticker column."]

    tickers = [str(t) for t in positions["ticker"]]
    caps: list[float] = []
    missing: list[str] = []

    frame_caps = positions["market_cap"] if "market_cap" in positions.columns else None
    for index, ticker in enumerate(tickers):
        value = _NAN
        if frame_caps is not None:
            value = _to_float(frame_caps.iloc[index])
        if not math.isfinite(value) and profiles:
            value = _to_float(getattr(profiles.get(ticker), "market_cap", _NAN))
        if math.isfinite(value) and value > 0.0:
            caps.append(value)
        else:
            caps.append(0.0)
            missing.append(ticker)

    total = float(sum(caps))
    warnings: list[str] = []
    if total <= 0.0:
        return pd.Series(dtype="float64"), [
            "No market-cap data was available for any holding, so a cap-weighted "
            "equilibrium cannot be built. Use your current weights or equal weights "
            "as the prior instead."
        ]
    if missing:
        warnings.append(
            f"No market cap for {', '.join(missing)} — excluded from the cap weighting, "
            f"which therefore covers {len(tickers) - len(missing)} of {len(tickers)} "
            "holdings. ETFs and smaller listings often report none."
        )
    return pd.Series(np.array(caps) / total, index=tickers, name="market_weight"), warnings


# ---------------------------------------------------------------------------
# view uncertainty
# ---------------------------------------------------------------------------
def omega_he_litterman(pick: np.ndarray, cov: np.ndarray, tau: float = DEFAULT_TAU) -> np.ndarray:
    """``Omega = diag(P tau Sigma P')`` -- He and Litterman's specification.

    Each view's uncertainty is proportional to the variance of the portfolio
    that view is about, so a view on a volatile spread is automatically
    treated as less precise than a view on a stable one. Diagonal because
    views are assumed independent; the off-diagonal terms of
    ``P tau Sigma P'`` are discarded deliberately, not overlooked.
    """
    if pick.size == 0:
        return np.empty((0, 0))
    scaled = float(tau) * cov
    variances = np.diag(pick @ scaled @ pick.T).astype("float64")
    variances = np.where(variances > MIN_VARIANCE_FLOOR, variances, MIN_VARIANCE_FLOOR)
    return np.diag(variances)


def omega_from_confidence(
    pick: np.ndarray,
    cov: np.ndarray,
    confidences: Sequence[float],
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Idzorek-style ``Omega`` from a 0-1 confidence per view.

        omega_k = ((1 - c_k) / c_k) * (P_k tau Sigma P_k')

    The limits are what make this usable in an interface: ``c = 0.5`` returns
    He-Litterman's specification exactly, ``c -> 1`` drives the variance to
    zero so the view is enforced almost exactly, and ``c -> 0`` drives it to
    infinity so the view is ignored. Monotone in between.

    This is the widely-used simplification of Idzorek's method, not his full
    iterative calibration (which solves for the ``omega`` that makes the
    posterior tilt exactly ``c`` times the fully-confident tilt). The
    simplification is monotone and has the correct endpoints, which is what an
    interactive confidence slider actually needs.
    """
    if pick.size == 0:
        return np.empty((0, 0))
    base = np.diag(omega_he_litterman(pick, cov, tau)).astype("float64")
    out = np.empty_like(base)
    for index, variance in enumerate(base):
        confidence = _to_float(confidences[index]) if index < len(confidences) else _NAN
        if not math.isfinite(confidence):
            confidence = DEFAULT_CONFIDENCE
        confidence = min(max(confidence, 0.0), 1.0)
        if confidence >= 1.0:
            out[index] = MIN_VARIANCE_FLOOR
        elif confidence <= 0.0:
            out[index] = np.inf
        else:
            out[index] = (1.0 - confidence) / confidence * variance
    # An infinite variance is a view with zero weight; a very large finite
    # number keeps every downstream inverse well defined and is numerically
    # indistinguishable from ignoring the view.
    finite_cap = max(float(np.max(base)) * 1e12, 1e6)
    return np.diag(np.where(np.isfinite(out), out, finite_cap))


# ---------------------------------------------------------------------------
# the posterior
# ---------------------------------------------------------------------------
def posterior_master_formula(
    prior: np.ndarray,
    cov: np.ndarray,
    pick: np.ndarray,
    view_returns: np.ndarray,
    omega: np.ndarray,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """The textbook Black-Litterman form, kept for verification only.

        E[R] = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
               [(tau Sigma)^-1 Pi + P' Omega^-1 Q]

    :func:`black_litterman` uses the Woodbury form instead, which inverts one
    k-by-k matrix rather than inverting both ``Omega`` and ``tau Sigma``. That
    matters because a confident view drives ``Omega`` towards singularity.
    Exposed so the test suite can assert the two agree -- an algebraic identity
    holding numerically is strong evidence neither implementation has drifted.
    """
    scaled = float(tau) * cov
    if pick.size == 0:
        return prior.copy()
    scaled_inv = np.linalg.pinv(scaled)
    omega_inv = np.linalg.pinv(omega)
    left = scaled_inv + pick.T @ omega_inv @ pick
    right = scaled_inv @ prior + pick.T @ omega_inv @ view_returns
    return _solve(left, right)


@dataclass(frozen=True)
class BlackLittermanResult:
    """Prior, posterior and every input that produced them.

    Every intermediate is kept because Black-Litterman's value is entirely in
    being auditable: a posterior return the user cannot trace back to a view
    and a confidence is just another opaque forecast, which is the thing this
    model exists to replace.
    """

    assets: tuple[str, ...] = ()
    prior_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    posterior_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    posterior_covariance: pd.DataFrame = field(default_factory=pd.DataFrame)
    prior_covariance: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    implied_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    views: tuple[View, ...] = ()
    pick: pd.DataFrame = field(default_factory=pd.DataFrame)
    omega: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_aversion: float = _NAN
    tau: float = DEFAULT_TAU
    risk_free_rate: float = 0.0
    prior_source: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.posterior_returns.empty

    @property
    def n_views(self) -> int:
        return len(self.views)

    def tilt(self) -> pd.Series:
        """Posterior minus prior: what the views actually changed."""
        if not self.ok:
            return pd.Series(dtype="float64")
        return (self.posterior_returns - self.prior_returns).rename("tilt")

    def posterior_mu(self, risk_free_rate: float | None = None) -> pd.Series:
        """Posterior TOTAL expected returns, ready for the optimiser.

        Black-Litterman works in excess returns throughout;
        :mod:`src.analytics.optimize` expects total annualised returns. The
        risk-free rate is added back here rather than left to the caller,
        because forgetting it understates every expected return by the rate
        and quietly rotates the whole efficient frontier.
        """
        if not self.ok:
            return pd.Series(dtype="float64")
        rate = self.risk_free_rate if risk_free_rate is None else _to_float(risk_free_rate)
        if not math.isfinite(rate):
            rate = 0.0
        return (self.posterior_returns + rate).rename("expected_return")

    def prior_mu(self, risk_free_rate: float | None = None) -> pd.Series:
        """Equilibrium TOTAL expected returns, for a side-by-side comparison."""
        if self.prior_returns.empty:
            return pd.Series(dtype="float64")
        rate = self.risk_free_rate if risk_free_rate is None else _to_float(risk_free_rate)
        if not math.isfinite(rate):
            rate = 0.0
        return (self.prior_returns + rate).rename("expected_return")

    def view_impacts(self) -> pd.DataFrame:
        """Per view: what equilibrium implied, what the user said, the gap.

        The single most informative table in the model. A view that merely
        restates the equilibrium changes nothing however confident it is, and
        this is where that becomes obvious.
        """
        if self.pick.empty or not self.ok:
            return pd.DataFrame(
                columns=["view", "equilibrium", "your view", "difference", "confidence"]
            )
        prior = self.prior_returns.reindex(self.pick.columns).to_numpy(dtype="float64")
        pick = self.pick.to_numpy(dtype="float64")
        implied = pick @ prior
        rows = [
            {
                "view": view.description,
                "equilibrium": float(implied[index]),
                "your view": float(view.expected_return),
                "difference": float(view.expected_return - implied[index]),
                "confidence": float(view.confidence),
            }
            for index, view in enumerate(self.views)
        ]
        return pd.DataFrame(rows)

    def summary_frame(self) -> pd.DataFrame:
        """Per asset: equilibrium, posterior, tilt and the market weight."""
        if not self.ok:
            return pd.DataFrame(
                columns=["asset", "market_weight", "equilibrium", "posterior", "tilt"]
            )
        return pd.DataFrame(
            {
                "asset": list(self.assets),
                "market_weight": self.market_weights.reindex(list(self.assets)).to_numpy(),
                "equilibrium": self.prior_returns.reindex(list(self.assets)).to_numpy(),
                "posterior": self.posterior_returns.reindex(list(self.assets)).to_numpy(),
                "tilt": self.tilt().reindex(list(self.assets)).to_numpy(),
            }
        )


def black_litterman(
    cov: pd.DataFrame | None,
    market_weights: object,
    views: Sequence[View] = (),
    risk_aversion: float | None = None,
    tau: float = DEFAULT_TAU,
    risk_free_rate: float = 0.0,
    omega: np.ndarray | None = None,
    prior_source: str = "",
) -> BlackLittermanResult:
    """Blend the equilibrium prior with ``views``. Never raises.

    ``risk_aversion=None`` falls back to :data:`DEFAULT_RISK_AVERSION` with a
    warning, because an uncalibrated ``delta`` scales the whole prior and the
    user should know it was assumed rather than measured.

    With no usable views the posterior is EXACTLY the prior -- asserted in the
    test suite, and the property that makes the model safe to leave switched on:
    an asset nobody has an opinion about keeps its equilibrium return instead of
    inheriting a historical average's noise.
    """
    warnings: list[str] = []

    labels, matrix = _clean_covariance(cov)
    if not labels:
        return BlackLittermanResult(
            warnings=(
                "The covariance matrix is missing or malformed, so no equilibrium can "
                "be computed. Black-Litterman needs at least two holdings with "
                "overlapping price history.",
            ),
            tau=tau,
            risk_free_rate=risk_free_rate,
        )

    weights = _align_weights(market_weights, labels)
    if weights.size == 0:
        return BlackLittermanResult(
            warnings=(
                "The prior weights could not be used (all zero, wrong length, or "
                "non-numeric), so there is no portfolio to reverse-optimise.",
            ),
            tau=tau,
            risk_free_rate=risk_free_rate,
        )

    delta = _to_float(risk_aversion) if risk_aversion is not None else _NAN
    if not math.isfinite(delta) or delta <= 0.0:
        delta = DEFAULT_RISK_AVERSION
        warnings.append(
            f"Market risk aversion could not be calibrated from the data, so the "
            f"textbook default delta = {DEFAULT_RISK_AVERSION} was assumed. It scales "
            "the whole equilibrium prior, so treat the levels as illustrative and the "
            "relative tilts as the output that matters."
        )

    tau_value = _to_float(tau)
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        tau_value = DEFAULT_TAU
        warnings.append(f"tau was unusable; the default {DEFAULT_TAU} was used.")

    prior_vector = delta * (matrix @ weights)
    prior = pd.Series(prior_vector, index=labels, name="equilibrium")

    pick, view_returns, kept, view_warnings = pick_matrix(views, labels)
    warnings.extend(view_warnings)

    scaled = tau_value * matrix

    if pick.size == 0:
        # No views: the posterior IS the prior, and the covariance is untouched.
        posterior_vector = prior_vector.copy()
        posterior_cov = matrix.copy()
        omega_frame = pd.DataFrame()
        pick_frame = pd.DataFrame(columns=labels)
        if views:
            warnings.append("No view survived validation, so the result is the pure equilibrium.")
    else:
        if omega is None:
            omega_matrix = omega_from_confidence(
                pick, matrix, [view.confidence for view in kept], tau_value
            )
        else:
            omega_matrix = np.asarray(omega, dtype="float64")
            if omega_matrix.shape != (pick.shape[0], pick.shape[0]):
                omega_matrix = omega_from_confidence(
                    pick, matrix, [view.confidence for view in kept], tau_value
                )
                warnings.append(
                    "The supplied Omega had the wrong shape and was replaced by the "
                    "confidence-derived one."
                )

        middle = pick @ scaled @ pick.T + omega_matrix
        # Woodbury form: one k-by-k inverse, and Omega is never inverted.
        gain = scaled @ pick.T @ np.linalg.pinv(middle)
        surprise = view_returns - pick @ prior_vector
        posterior_vector = prior_vector + gain @ surprise

        estimation_error = scaled - gain @ pick @ scaled
        posterior_cov = matrix + estimation_error
        posterior_cov = (posterior_cov + posterior_cov.T) / 2.0

        omega_frame = pd.DataFrame(
            omega_matrix,
            index=[view.description for view in kept],
            columns=[view.description for view in kept],
        )
        pick_frame = pd.DataFrame(pick, index=[view.description for view in kept], columns=labels)

    posterior = pd.Series(posterior_vector, index=labels, name="posterior")
    implied = reverse_optimise_weights(
        pd.DataFrame(posterior_cov, index=labels, columns=labels), posterior, delta
    )

    return BlackLittermanResult(
        assets=tuple(labels),
        prior_returns=prior,
        posterior_returns=posterior,
        posterior_covariance=pd.DataFrame(posterior_cov, index=labels, columns=labels),
        prior_covariance=pd.DataFrame(matrix, index=labels, columns=labels),
        market_weights=pd.Series(weights, index=labels, name="market_weight"),
        implied_weights=implied,
        views=tuple(kept),
        pick=pick_frame,
        omega=omega_frame,
        risk_aversion=delta,
        tau=tau_value,
        risk_free_rate=_to_float(risk_free_rate)
        if math.isfinite(_to_float(risk_free_rate))
        else 0.0,
        prior_source=prior_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )
