"""Attribution: who earned the money, and who is actually carrying the risk.

A weight column answers "how much do I own?". It does not answer either of
the two questions that matter when a portfolio misbehaves:

1. RETURN ATTRIBUTION -- which names produced the portfolio's return? A 3%
   position that doubled contributed more than a 30% position that drifted
   1%. Contribution is measured against the portfolio's own cost base, so the
   per-name numbers ADD UP to the headline return rather than merely ranking
   alongside it.

2. RISK ATTRIBUTION -- which names produced the portfolio's volatility?
   Volatility is not additive across holdings (correlation cancels part of
   it), but it *is* exactly decomposable, because portfolio volatility is a
   homogeneous function of degree one in the weights. Euler's theorem then
   gives an exact, non-arbitrary split.

Formulas (stated once here, restated on each function)
------------------------------------------------------
Return contribution     c_i     = pnl_i / total_investment * 100        [%]
                        sum_i c_i = total_pnl / total_investment * 100
                                  = the portfolio's total return %      (identity)
Contribution share      s_i     = pnl_i / total_pnl * 100               [%]

Portfolio volatility    sigma_p = sqrt(w' S w)
Marginal contribution   MCTR_i  = d(sigma_p) / d(w_i) = (S w)_i / sigma_p
Component contribution  CCTR_i  = w_i * MCTR_i
                        sum_i CCTR_i = (w' S w) / sigma_p = sigma_p     (identity)
Risk contribution %     rc_i    = CCTR_i / sigma_p * 100
                        sum_i rc_i = 100                                (identity)
Component VaR           CVaR_i  = CCTR_i * z * sqrt(h),  z = Phi^-1(1 - c)
                        sum_i CVaR_i = sigma_p * z * sqrt(h) = parametric VaR

Sanity anchor: for n equal-weighted assets with identical variances and zero
correlation, S = s^2 I and w_i = 1/n, so MCTR_i = s / sqrt(n),
CCTR_i = s / (n * sqrt(n)) and every rc_i is exactly 1/n of the risk. Nothing
in this module needs a special case to produce that -- it falls out of the
algebra, and the test suite asserts it.

Conventions and guarantees
--------------------------
* UNITS. ``cov`` is used exactly as supplied. Pass an ANNUALISED covariance
  matrix (``risk.covariance_matrix(..., annualize=True)``) and ``sigma_p``,
  ``mctr`` and ``cctr`` are annualised decimals; pass a daily one and they are
  daily. ``risk_contribution_pct`` is scale-free either way, because both the
  numerator and the denominator carry the same units.
* WEIGHTS may be supplied in percent or as fractions, as a dict, Series or
  sequence. They are aligned to ``cov.columns`` by label (names missing from
  the weights count as zero) or positionally when the lengths match, and are
  always normalised by their own sum before use.
* Nothing here mutates its input; every frame is copied or rebuilt.
* Degenerate input (empty, one row, all-NaN, zero variance, single asset)
  returns an empty frame / NaN, never an exception and never a fabricated 0.
* Constants come from ``src.config``; the only local constants are the
  presentation-level ones ``src.config`` does not carry, declared below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analytics.risk import (
    covariance_matrix,
    effective_number_of_bets,
    herfindahl_index,
)
from src.config import DEFAULT_VAR_CONFIDENCE

__all__ = [
    "COMPONENT_VAR_COLUMNS",
    "CONCENTRATION_KEYS",
    "RETURN_CONTRIBUTION_COLUMNS",
    "RISK_CONTRIBUTION_COLUMNS",
    "RISK_VS_WEIGHT_COLUMNS",
    "TOP_N",
    "attribution_summary",
    "component_var",
    "concentration_metrics",
    "gini_coefficient",
    "marginal_contribution_to_risk",
    "return_contribution",
    "risk_contribution",
]

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
#: Exact output schema of :func:`return_contribution`, in order.
RETURN_CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "weight_pct",
    "pnl",
    "pnl_pct",
    "contribution_pct",
    "contribution_share_pct",
)

#: Exact output schema of :func:`risk_contribution`, in order.
RISK_CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "weight_pct",
    "mctr",
    "cctr",
    "risk_contribution_pct",
)

#: Exact output schema of :func:`component_var`, in order.
COMPONENT_VAR_COLUMNS: tuple[str, ...] = (
    "ticker",
    "weight_pct",
    "mctr",
    "cctr",
    "component_var",
    "component_var_pct",
)

#: Exact output schema of the ``risk_vs_weight`` frame, in order.
RISK_VS_WEIGHT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "weight_pct",
    "risk_contribution_pct",
    "ratio",
)

#: Keys always present in the :func:`concentration_metrics` dict.
CONCENTRATION_KEYS: tuple[str, ...] = (
    "hhi",
    "effective_holdings",
    "top1_pct",
    "top3_pct",
    "top5_pct",
    "gini",
    "n_holdings",
)

#: How many names the summary's leader-board frames carry. A presentation
#: count, not a statistical threshold, which is why it lives here rather than
#: in ``src.config``.
TOP_N: int = 5

#: Columns whose text form must survive intact into the output frames.
_TEXT_COLUMNS: frozenset[str] = frozenset({"ticker"})


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _nan() -> float:
    """The single 'undefined' value this module returns."""
    return float("nan")


def _to_float(value: Any) -> float:
    """Best-effort float coercion; NaN for anything unparseable or infinite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _nan()
    return out if math.isfinite(out) else _nan()


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """A zero-row frame carrying the documented schema and dtypes."""
    data = {
        name: pd.Series(dtype="object" if name in _TEXT_COLUMNS else "float64") for name in columns
    }
    return pd.DataFrame(data)


def _is_frame(obj: Any) -> bool:
    """True for a non-empty DataFrame (the only shape the readers accept)."""
    return isinstance(obj, pd.DataFrame) and not obj.empty


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Copy one column as floats, or an all-NaN column when it is absent.

    Non-numeric entries become NaN rather than raising, which is what lets a
    hand-typed holdings table flow through the attribution without stopping
    the pipeline.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    series = pd.to_numeric(frame[name], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).astype("float64")


def _weight_series(weights: Any, index: pd.Index | None = None) -> pd.Series:
    """Coerce weights (dict / Series / sequence) to a float Series.

    Mirrors the coercion in ``src.analytics.risk`` so both modules digest the
    same inputs identically. With ``index`` supplied the weights are aligned by
    label when the labels overlap (missing names become 0.0), otherwise
    positionally when the lengths match. Unusable input yields an EMPTY
    Series, never an exception.
    """
    if weights is None:
        return pd.Series(dtype=float)
    try:
        if isinstance(weights, Mapping):
            series = pd.Series(dict(weights))
        elif isinstance(weights, pd.Series):
            series = pd.Series(weights, copy=True)
        elif isinstance(weights, pd.DataFrame):
            return pd.Series(dtype=float)
        else:
            series = pd.Series(list(weights))
    except (TypeError, ValueError):
        return pd.Series(dtype=float)

    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.index.has_duplicates:
        # A ticker listed twice is one exposure: add the slices rather than
        # letting the label alignment below fail on the duplicate.
        series = series.groupby(level=0).sum()

    if index is not None:
        if len(set(series.index) & set(index)) > 0:
            series = series.reindex(index)
        elif len(series) == len(index):
            series = pd.Series(series.to_numpy(dtype=float), index=index)
        else:
            return pd.Series(dtype=float)
        # Fill AFTER both branches, not just the label one. A NaN weight means
        # "no known exposure" and must read as zero however it was supplied;
        # left in place on the positional path it makes w' Sigma w non-finite
        # and NaNs out the entire decomposition, so the same portfolio spelled
        # two ways would return two different answers.
        return series.fillna(0.0).astype("float64")
    return series.dropna().astype("float64")


def _normalise_weights(weights: pd.Series) -> pd.Series:
    """Scale weights so they sum to 1 (accepts percent or fraction input).

    Returns an EMPTY Series when the input is empty or sums to zero, which is
    what makes every weight-driven statistic degrade to NaN instead of
    dividing by zero.
    """
    if weights.empty:
        return pd.Series(dtype=float)
    total = _to_float(weights.sum())
    if not math.isfinite(total) or total == 0.0:
        return pd.Series(dtype=float)
    return (weights / total).astype("float64")


def _valid_cov(cov: Any) -> tuple[np.ndarray | None, pd.Index]:
    """Validate a covariance matrix and return ``(matrix, labels)``.

    Accepts a labelled DataFrame (the documented input) or a bare square
    ndarray, which is labelled positionally. ``(None, empty index)`` is
    returned -- rather than an exception -- for anything unusable: empty,
    non-square, or carrying a non-finite entry. A NaN anywhere in a covariance
    matrix poisons every quadratic form built from it, so it is rejected as a
    whole instead of being silently patched with zeros.
    """
    blank = pd.Index([])
    if cov is None:
        return None, blank
    if isinstance(cov, np.ndarray):
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
            return None, blank
        frame = pd.DataFrame(cov, index=range(cov.shape[0]), columns=range(cov.shape[1]))
    elif isinstance(cov, pd.DataFrame):
        frame = cov
    else:
        return None, blank

    if frame.empty or frame.shape[0] != frame.shape[1]:
        return None, blank
    try:
        matrix = frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None, blank
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return None, blank
    return matrix, pd.Index(frame.columns)


def _z_quantile(confidence: float) -> float:
    """Standard-normal quantile at the LOSS tail: z = Phi^-1(1 - confidence).

    Negative for any sensible confidence level (z = -1.645 at 95%), which is
    what makes the component VaRs come out as negative decimal losses in line
    with ``risk.value_at_risk``.
    """
    level = _to_float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return _nan()
    return _to_float(norm.ppf(1.0 - level))


def _risk_decomposition(
    weights: Any, cov: Any
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray, float]:
    """Shared Euler decomposition behind every risk-attribution function.

    Returns ``(labels, w, mctr, cctr, sigma_p)``. When the covariance matrix
    or the weights are unusable the labels are empty; when they are usable but
    the portfolio variance is not strictly positive (a zero-variance book) the
    weights survive and ``mctr``/``cctr``/``sigma_p`` are NaN, so the caller
    can still show WHAT is held while honestly reporting that its risk split
    is undefined.
    """
    matrix, labels = _valid_cov(cov)
    if matrix is None:
        return pd.Index([]), np.empty(0), np.empty(0), np.empty(0), _nan()

    w = _normalise_weights(_weight_series(weights, labels))
    if w.empty or len(w) != matrix.shape[0]:
        return pd.Index([]), np.empty(0), np.empty(0), np.empty(0), _nan()

    vector = w.to_numpy(dtype=float)
    marginal_variance = matrix @ vector  # (S w)_i
    portfolio_var = _to_float(vector @ marginal_variance)
    if not math.isfinite(portfolio_var) or portfolio_var <= 0.0:
        blank = np.full(vector.size, np.nan)
        return labels, vector, blank, blank.copy(), _nan()

    sigma_p = math.sqrt(portfolio_var)
    mctr = marginal_variance / sigma_p
    cctr = vector * mctr
    return labels, vector, mctr, cctr, float(sigma_p)


def _rank_series(frame: pd.DataFrame) -> pd.Series:
    """The column the leader-boards rank on: contribution, else raw P&L.

    ``contribution_pct`` needs a cost base; a positions frame without an
    ``investment`` column cannot supply one, and ranking by absolute P&L is
    then the honest fallback rather than an empty table.
    """
    if "contribution_pct" in frame.columns and frame["contribution_pct"].notna().any():
        return frame["contribution_pct"]
    return frame.get("pnl", pd.Series(np.nan, index=frame.index, dtype="float64"))


# ---------------------------------------------------------------------------
# return attribution
# ---------------------------------------------------------------------------
def return_contribution(positions: pd.DataFrame | None) -> pd.DataFrame:
    """Per-holding share of the portfolio's return, measured against cost.

        contribution_pct_i       = pnl_i / total_investment * 100
        contribution_share_pct_i = pnl_i / total_pnl * 100

    The first is the one that matters: it is ADDITIVE, so

        sum_i contribution_pct_i = total_pnl / total_investment * 100

    which is exactly the portfolio's total return percentage. That identity is
    what turns a ranking into an attribution -- every name's number can be read
    as "this holding put X percentage points on (or took X off) the headline
    return". ``contribution_share_pct`` splits the P&L itself into 100% and is
    NaN when the portfolio's net P&L is zero (an undefined split, not a zero
    one); it can exceed 100 or go negative when winners and losers offset.

    Both money columns are whatever base currency ``positions`` carries;
    ``weight_pct`` and ``pnl_pct`` are passed through unchanged from
    :func:`src.analytics.positions.build_positions`.

    Rows are sorted by ``contribution_pct`` descending (NaN last). The frame's
    ``.attrs`` carry ``total_investment``, ``total_pnl`` and
    ``total_return_pct`` so the caller can display the identity it sums to.

    Returns an EMPTY frame with the documented columns when ``positions`` is
    missing, empty, or carries no ``ticker``/``pnl`` pair. Never mutates the
    input and never raises.
    """
    columns = list(RETURN_CONTRIBUTION_COLUMNS)
    if not _is_frame(positions):
        return _empty_frame(columns)
    if "ticker" not in positions.columns or "pnl" not in positions.columns:
        return _empty_frame(columns)

    source = positions.copy()
    pnl = _numeric_column(source, "pnl")
    investment = _numeric_column(source, "investment")

    total_investment = _to_float(investment.sum(skipna=True))
    total_pnl = _to_float(pnl.sum(skipna=True))

    if math.isfinite(total_investment) and total_investment > 0.0:
        contribution = pnl / total_investment * 100.0
        total_return_pct = total_pnl / total_investment * 100.0
    else:
        contribution = pd.Series(np.nan, index=source.index, dtype="float64")
        total_return_pct = _nan()

    if math.isfinite(total_pnl) and total_pnl != 0.0:
        share = pnl / total_pnl * 100.0
    else:
        share = pd.Series(np.nan, index=source.index, dtype="float64")

    frame = pd.DataFrame(
        {
            "ticker": source["ticker"].astype(str).to_numpy(),
            "weight_pct": _numeric_column(source, "weight_pct").to_numpy(),
            "pnl": pnl.to_numpy(),
            "pnl_pct": _numeric_column(source, "pnl_pct").to_numpy(),
            "contribution_pct": contribution.to_numpy(),
            "contribution_share_pct": share.to_numpy(),
        },
        columns=columns,
    )
    frame = frame.sort_values("contribution_pct", ascending=False, na_position="last").reset_index(
        drop=True
    )
    frame.attrs["total_investment"] = total_investment
    frame.attrs["total_pnl"] = total_pnl
    frame.attrs["total_return_pct"] = total_return_pct
    return frame


# ---------------------------------------------------------------------------
# risk attribution (Euler decomposition of portfolio volatility)
# ---------------------------------------------------------------------------
def marginal_contribution_to_risk(weights: Any, cov: pd.DataFrame) -> pd.Series:
    """Sensitivity of portfolio volatility to each weight.

        MCTR_i = d(sigma_p) / d(w_i) = (S w)_i / sigma_p,
        sigma_p = sqrt(w' S w)

    Read as "add one more unit of weight to this name (funded pro rata from
    the rest) and portfolio volatility moves by MCTR_i". It is a derivative,
    not a share, so it does NOT sum to anything meaningful --
    :func:`risk_contribution` supplies the additive version.

    Units follow ``cov``: an annualised covariance matrix gives an annualised
    MCTR. Weights may be percent or fractions and are normalised internally;
    they are aligned to ``cov.columns``.

    Returns an EMPTY Series when ``cov`` or the weights are unusable, and an
    all-NaN Series indexed by ``cov.columns`` when the portfolio variance is
    zero (an undefined derivative, not a zero one). Never raises.
    """
    labels, _w, mctr, _cctr, _sigma = _risk_decomposition(weights, cov)
    if len(labels) == 0:
        return pd.Series(dtype=float)
    return pd.Series(mctr, index=labels, dtype="float64", name="mctr")


def risk_contribution(weights: Any, cov: pd.DataFrame) -> pd.DataFrame:
    """Exact additive split of portfolio volatility across holdings.

        sigma_p = sqrt(w' S w)
        MCTR_i  = (S w)_i / sigma_p
        CCTR_i  = w_i * MCTR_i
        rc_i    = CCTR_i / sigma_p * 100                         [%]

    Because sigma_p is homogeneous of degree one in w, Euler's theorem makes
    this decomposition EXACT rather than approximate:

        sum_i CCTR_i = w' S w / sigma_p = sigma_p      -> the identity tested
        sum_i rc_i   = 100

    So a 5% position in a volatile, highly correlated name can legitimately
    carry 15% of the portfolio's risk -- that gap is the whole point of the
    table, and :func:`attribution_summary` surfaces it as ``risk_vs_weight``.
    For n equal-weighted, equal-variance, uncorrelated assets every ``rc_i`` is
    exactly ``100 / n``.

    ``weight_pct`` is reported in PERCENT after normalisation (it sums to 100),
    while ``mctr`` and ``cctr`` carry ``cov``'s own units -- annualised
    decimals when ``cov`` is annualised. Rows are sorted by
    ``risk_contribution_pct`` descending; ``.attrs["portfolio_volatility"]``
    holds ``sigma_p`` and ``.attrs["portfolio_variance"]`` holds ``sigma_p^2``.

    Returns an EMPTY frame for an unusable covariance matrix or unusable
    weights, and a frame whose risk columns are NaN (weights intact) when the
    portfolio variance is zero. Never mutates its input and never raises.
    """
    columns = list(RISK_CONTRIBUTION_COLUMNS)
    labels, w, mctr, cctr, sigma_p = _risk_decomposition(weights, cov)
    if len(labels) == 0:
        return _empty_frame(columns)

    if math.isfinite(sigma_p) and sigma_p > 0.0:
        contribution = cctr / sigma_p * 100.0
    else:
        contribution = np.full(w.size, np.nan)

    frame = pd.DataFrame(
        {
            "ticker": [str(name) for name in labels],
            "weight_pct": w * 100.0,
            "mctr": mctr,
            "cctr": cctr,
            "risk_contribution_pct": contribution,
        },
        columns=columns,
    )
    frame = frame.sort_values(
        "risk_contribution_pct", ascending=False, na_position="last"
    ).reset_index(drop=True)
    frame.attrs["portfolio_volatility"] = sigma_p
    frame.attrs["portfolio_variance"] = sigma_p**2 if math.isfinite(sigma_p) else _nan()
    return frame


def component_var(
    weights: Any,
    cov: pd.DataFrame,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Split the portfolio's parametric VaR across its holdings.

        z          = Phi^-1(1 - confidence)                     (negative)
        VaR_p      = z * sigma_p * sqrt(horizon)
        CVaR_i     = CCTR_i * z * sqrt(horizon)
        sum_i CVaR_i = VaR_p                                    (identity)

    Same SIGN CONVENTION as ``risk.value_at_risk``: a NEGATIVE decimal loss,
    so -0.021 reads "2.1% of portfolio value". ``component_var_pct`` is each
    name's share of that loss and sums to 100; it is numerically identical to
    ``risk_contribution_pct`` because the VaR scaling ``z * sqrt(horizon)`` is
    common to every name -- which is precisely the limitation of a Gaussian
    decomposition: it re-labels volatility shares as tail shares and cannot
    see a name whose risk is skew or jump risk rather than variance.

    UNITS -- the one thing to get right. ``sqrt(horizon_days)`` scales
    ``cov``'s own period, so ``cov`` must be expressed per unit of
    ``horizon_days``: pass a DAILY covariance matrix
    (``risk.covariance_matrix(..., annualize=False)``) for a horizon in
    trading days. Passing an annualised matrix with ``horizon_days=1`` reports
    a one-YEAR VaR; ``.attrs["convention"]`` records this so a report can
    print it.

    Returns an EMPTY frame for an unusable covariance matrix, unusable
    weights, a confidence outside (0, 1) or a non-positive horizon, and a
    weights-only frame with NaN risk columns when the portfolio variance is
    zero. Never raises.
    """
    columns = list(COMPONENT_VAR_COLUMNS)
    z = _z_quantile(confidence)
    horizon = _to_float(horizon_days)
    if not math.isfinite(z) or not math.isfinite(horizon) or horizon <= 0.0:
        return _empty_frame(columns)

    labels, w, mctr, cctr, sigma_p = _risk_decomposition(weights, cov)
    if len(labels) == 0:
        return _empty_frame(columns)

    scale = z * math.sqrt(horizon)
    component = cctr * scale
    if math.isfinite(sigma_p) and sigma_p > 0.0:
        portfolio_var_value = sigma_p * scale
        share = cctr / sigma_p * 100.0
    else:
        portfolio_var_value = _nan()
        share = np.full(w.size, np.nan)

    frame = pd.DataFrame(
        {
            "ticker": [str(name) for name in labels],
            "weight_pct": w * 100.0,
            "mctr": mctr,
            "cctr": cctr,
            "component_var": component,
            "component_var_pct": share,
        },
        columns=columns,
    )
    frame = frame.sort_values("component_var_pct", ascending=False, na_position="last").reset_index(
        drop=True
    )
    frame.attrs["portfolio_var"] = portfolio_var_value
    frame.attrs["portfolio_volatility"] = sigma_p
    frame.attrs["confidence"] = _to_float(confidence)
    frame.attrs["horizon_days"] = horizon
    frame.attrs["z"] = z
    frame.attrs["convention"] = (
        "Negative decimal loss over horizon_days periods of the covariance matrix's own frequency."
    )
    return frame


# ---------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------
def gini_coefficient(weights: Any) -> float:
    """Inequality of the weight distribution: 0 = equal, -> 1 = one name.

        G = sum_i sum_j |w_i - w_j| / (2 * n^2 * mean(w))

    evaluated in O(n log n) through the equivalent sorted form

        G = 2 * sum_i( i * w_(i) ) / (n * sum(w)) - (n + 1) / n,   i = 1..n

    with ``w_(i)`` the weights sorted ascending. This is the POPULATION Gini
    (no n/(n-1) small-sample correction), so its supremum for n names is
    (n - 1) / n rather than exactly 1: a book where one position approaches
    100% of the value scores 0.9 across ten names and 0.99 across a hundred.
    Equal weights score exactly 0, and a single holding scores 0 as well --
    a one-element distribution is trivially equal, which is why the Gini is
    reported ALONGSIDE the Herfindahl index (which correctly reads 1.0 there)
    rather than instead of it.

    Weights may be percentages or fractions; only their relative sizes matter.
    NaN for empty input, a non-positive total, or any negative weight (short
    positions make the Lorenz construction meaningless).
    """
    w = _weight_series(weights)
    if w.empty:
        return _nan()
    values = w.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0.0):
        return _nan()

    total = _to_float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        return _nan()

    n = values.size
    ordered = np.sort(values)
    ranks = np.arange(1, n + 1, dtype=float)
    result = 2.0 * float(np.sum(ranks * ordered)) / (n * total) - (n + 1.0) / n
    if not math.isfinite(result):
        return _nan()
    return float(min(max(result, 0.0), 1.0))


def concentration_metrics(positions: pd.DataFrame | None) -> dict:
    """How lopsided the book is, from four complementary angles.

    Reads ``weight_pct`` from a positions frame, drops non-finite and negative
    entries, and renormalises what is left so a filtered subset is still
    self-consistent. Returns::

        hhi                 sum(w_i^2), fractions      1/n (equal) .. 1 (one name)
        effective_holdings  1 / hhi                    "how many equal bets?"
        top1_pct            largest weight             [%]
        top3_pct            three largest weights      [%]
        top5_pct            five largest weights       [%]
        gini                see :func:`gini_coefficient`  0 (equal) .. ->1
        n_holdings          usable weights, as an int

    HHI and the effective count come from ``src.analytics.risk`` so the two
    modules cannot drift apart. HHI answers "how concentrated?" and is
    dominated by the largest position; the Gini answers "how UNEVENLY is the
    money spread?" across the whole distribution, which is a different
    question -- a book of ten equal names and a book of one name both score a
    Gini of 0, and only the HHI (0.1 versus 1.0) separates them. The pair is
    therefore read together. ``topK_pct`` returns the running total when fewer
    than K names are held.

    Every value is NaN (``n_holdings`` 0) when ``positions`` is missing, empty,
    carries no ``weight_pct`` column, or has no usable weights. Never mutates
    the input and never raises.
    """
    blank: dict[str, Any] = {key: _nan() for key in CONCENTRATION_KEYS}
    blank["n_holdings"] = 0

    if not _is_frame(positions) or "weight_pct" not in positions.columns:
        return blank

    raw = _numeric_column(positions, "weight_pct").dropna()
    raw = raw[raw >= 0.0]
    if raw.empty:
        return blank

    n = int(len(raw))
    fractions = _normalise_weights(raw)
    if fractions.empty:
        result = dict(blank)
        result["n_holdings"] = n
        return result

    ordered = np.sort(fractions.to_numpy(dtype=float))[::-1]

    def _top(k: int) -> float:
        return float(np.sum(ordered[: min(k, ordered.size)]) * 100.0)

    return {
        "hhi": herfindahl_index(fractions * 100.0),
        "effective_holdings": effective_number_of_bets(fractions),
        "top1_pct": _top(1),
        "top3_pct": _top(3),
        "top5_pct": _top(5),
        "gini": gini_coefficient(fractions),
        "n_holdings": n,
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _risk_vs_weight(risk_frame: pd.DataFrame) -> pd.DataFrame:
    """Names whose risk share outruns their capital share.

        ratio_i = risk_contribution_pct_i / weight_pct_i

    Above 1.0 the position is punching above its weight in RISK terms (higher
    volatility, or a correlation profile that fails to diversify); below 1.0 it
    is a genuine diversifier. Sorted by ``ratio`` descending, NaN last.
    """
    columns = list(RISK_VS_WEIGHT_COLUMNS)
    if not _is_frame(risk_frame):
        return _empty_frame(columns)

    weight = _numeric_column(risk_frame, "weight_pct")
    contribution = _numeric_column(risk_frame, "risk_contribution_pct")
    ratio = contribution / weight.where(weight > 0.0)

    frame = pd.DataFrame(
        {
            "ticker": risk_frame["ticker"].astype(str).to_numpy(),
            "weight_pct": weight.to_numpy(),
            "risk_contribution_pct": contribution.to_numpy(),
            "ratio": ratio.to_numpy(),
        },
        columns=columns,
    )
    return frame.sort_values("ratio", ascending=False, na_position="last").reset_index(drop=True)


def attribution_summary(
    positions: pd.DataFrame | None,
    asset_returns: pd.DataFrame | None = None,
    cov: pd.DataFrame | None = None,
) -> dict:
    """Everything the attribution page needs, computed once.

    Returns::

        return_contribution   DataFrame  see :func:`return_contribution`
        risk_contribution     DataFrame  see :func:`risk_contribution`
        concentration         dict       see :func:`concentration_metrics`
        top_contributors      DataFrame  up to TOP_N names that ADDED return
        top_detractors        DataFrame  up to TOP_N names that COST return
        risk_vs_weight        DataFrame  ticker, weight_pct,
                                         risk_contribution_pct, ratio

    When ``cov`` is not supplied it is estimated from ``asset_returns`` as the
    ANNUALISED sample covariance with NO shrinkage
    (``risk.covariance_matrix(asset_returns, shrinkage=0.0, annualize=True)``):
    attribution describes the risk the portfolio actually ran, so the estimate
    is left unbiased here, whereas the optimiser -- which inverts the matrix
    and is therefore hostage to its noisiest eigenvector -- shrinks it. Pass
    ``cov`` explicitly to override that choice.

    Weights come from the positions' own ``weight_pct`` column, keyed by
    ticker, so the risk split is measured on the portfolio as it stands today.
    The leader-boards rank on ``contribution_pct`` and fall back to raw ``pnl``
    when no cost base is available.

    Every section degrades independently: an unusable covariance matrix still
    yields the return attribution and the concentration metrics. Nothing here
    mutates its inputs and nothing raises.
    """
    returns_frame = return_contribution(positions)
    concentration = concentration_metrics(positions)

    matrix = cov
    if matrix is None and asset_returns is not None:
        matrix = covariance_matrix(asset_returns, shrinkage=0.0, annualize=True)

    weights: Any = None
    if _is_frame(positions) and {"ticker", "weight_pct"} <= set(positions.columns):
        weights = pd.Series(
            _numeric_column(positions, "weight_pct").to_numpy(),
            index=positions["ticker"].astype(str).to_numpy(),
            dtype="float64",
        ).dropna()

    risk_frame = risk_contribution(weights, matrix)

    rank = _rank_series(returns_frame)
    if returns_frame.empty or rank.notna().sum() == 0:
        winners = _empty_frame(RETURN_CONTRIBUTION_COLUMNS)
        losers = _empty_frame(RETURN_CONTRIBUTION_COLUMNS)
    else:
        ordered = returns_frame.assign(_rank=rank.to_numpy())
        winners = (
            ordered[ordered["_rank"] > 0.0]
            .sort_values("_rank", ascending=False)
            .head(TOP_N)
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
        losers = (
            ordered[ordered["_rank"] < 0.0]
            .sort_values("_rank", ascending=True)
            .head(TOP_N)
            .drop(columns="_rank")
            .reset_index(drop=True)
        )

    return {
        "return_contribution": returns_frame,
        "risk_contribution": risk_frame,
        "concentration": concentration,
        "top_contributors": winners,
        "top_detractors": losers,
        "risk_vs_weight": _risk_vs_weight(risk_frame),
    }
