"""The ML section: volatility regime, forensic outliers, and an honest backtest.

Three things live here, and they are deliberately three different *kinds* of
claim, ordered from safest to least safe.

**The volatility regime** is a description of the recent past. Realised
volatility is measured with an exponentially-weighted standard deviation, placed
against its own history as a percentile, and labelled. Nothing is predicted. The
"regime" is a statement about the last few weeks relative to the last few years,
and it is checkable by anyone with the price series.

**Forensic outlier scores** are a statement about how unusual *this company's
accounts* are against its peer set. An isolation forest on margins, growth,
returns and leverage finds companies that sit far from the crowd. That is a
prompt to read the filings, never an accusation -- a genuinely unique business
model scores as an outlier too, and the module says so in as many words.

**The supervised forward-return model** is the only part that makes a claim
about the future, and it is the part most likely to mislead. It is built on
:mod:`src.analytics.features` and :mod:`src.analytics.forecast`, which already
construct features without look-ahead by design and evaluate them with purged
walk-forward validation and rank IC. This module adds no new modelling; it adds
the *reporting discipline* -- out-of-sample rank IC with its fold-by-fold
spread, and an explicit statement when the result is indistinguishable from
noise, which is the normal outcome.

The single most important number in this file is
:attr:`ReturnSignal.indistinguishable_from_noise`. A page that showed a
"predicted return" without it would be lying by omission, because for this kind
of model on this much data the honest answer is almost always "no measurable
edge".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "EWMA_SPAN",
    "REGIME_BANDS",
    "VolatilityRegime",
    "AnomalyScore",
    "ReturnSignal",
    "MIN_ROWS_FOR_FOREST",
    "volatility_regime",
    "forensic_outliers",
    "forward_return_signal",
    "NO_EDGE_IC",
]

#: Span of the exponentially-weighted volatility estimate, in trading sessions.
#: Twenty sessions is about a month: long enough that one gap does not dominate,
#: short enough that it responds to a genuine change in conditions.
EWMA_SPAN = 20

#: Percentile band -> regime label, ascending. Read as "this month's volatility
#: sits in the Nth percentile of this company's own history".
REGIME_BANDS: tuple[tuple[float, str], ...] = (
    (0.20, "unusually calm"),
    (0.45, "calm"),
    (0.70, "normal"),
    (0.90, "elevated"),
    (1.01, "stressed"),
)

#: An isolation forest on fewer rows than this is fitting noise. The value is
#: deliberately conservative because the forest's contamination parameter
#: implies it will always find *some* outliers, however few real ones exist.
MIN_ROWS_FOR_FOREST = 12

#: An out-of-sample rank IC below this is not a signal. Matches the threshold in
#: :mod:`src.analytics.sentiment` on purpose: the two sections should not
#: disagree about what counts as "no measurable relationship".
NO_EDGE_IC = 0.03


def _num(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


# ---------------------------------------------------------------------------
# volatility regime -- a description of the recent past
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VolatilityRegime:
    """Where recent volatility sits inside the company's own history."""

    current: float = float("nan")
    median: float = float("nan")
    percentile: float = float("nan")
    regime: str = "unknown"
    n_obs: int = 0
    series: tuple[float, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not math.isnan(self.current) and self.n_obs > 0

    @property
    def reading(self) -> str:
        """A sentence stating the regime without implying a forecast."""
        if not self.ok:
            return "Volatility could not be estimated from the history available."
        if math.isnan(self.percentile):
            return (
                f"Realised volatility is {self.current:.1%} annualised. There is not "
                f"enough history to say where that sits in this company's own range."
            )
        multiple = (
            f"{self.current / self.median:.2f}x" if self.median not in (0.0, math.nan) else "—"
        )
        return (
            f"Realised volatility is {self.current:.1%} annualised, in the "
            f"{self.percentile:.0%} percentile of its own {self.n_obs} sessions and "
            f"{multiple} the median. Conditions are {self.regime}. This is a "
            "description of the last month, not a forecast of the next one — "
            "volatility clusters, so a calm month makes a calm next month somewhat "
            "more likely, and nothing more than that."
        )


def volatility_regime(prices: Sequence[float]) -> VolatilityRegime:
    """Exponentially-weighted realised volatility, and where it currently sits.

    Takes prices only, not dates: the estimate is a function of the return
    series, and accepting a date index the function never reads would suggest a
    time-awareness it does not have.

    Deliberately EWMA rather than GARCH. A GARCH fit would add two parameters
    and a numerical optimiser to estimate the same thing, and on a few thousand
    daily observations the two agree on the level closely enough that the extra
    machinery buys nothing a reader can see. The choice is stated rather than
    hidden, and :data:`EWMA_SPAN` is the single knob.
    """
    values = np.asarray([_num(v) for v in prices], dtype="float64")
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < EWMA_SPAN + 2:
        return VolatilityRegime(
            n_obs=int(values.size),
            notices=(
                f"Only {values.size} usable price(s); at least {EWMA_SPAN + 2} are "
                "needed before a volatility estimate responds to anything but noise.",
            ),
        )

    returns = np.diff(np.log(values))
    series = pd.Series(returns, dtype="float64").ewm(span=EWMA_SPAN, adjust=False).std()
    annualised = series * math.sqrt(252.0)
    annualised = annualised.dropna()
    if annualised.empty:
        return VolatilityRegime(
            n_obs=int(values.size), notices=("No volatility could be estimated.",)
        )

    current = float(annualised.iloc[-1])
    history = annualised.to_numpy(dtype="float64")
    median = float(np.median(history))
    percentile = float(np.mean(history <= current))

    regime = "unknown"
    for ceiling, label in REGIME_BANDS:
        if percentile < ceiling:
            regime = label
            break

    return VolatilityRegime(
        current=current,
        median=median,
        percentile=percentile,
        regime=regime,
        n_obs=int(len(history)),
        series=tuple(float(v) for v in history[-(len(history)) :]),
    )


# ---------------------------------------------------------------------------
# forensic outliers -- a statement about the accounts
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """How far one company's accounts sit from its peer set."""

    symbol: str = ""
    score: float = float("nan")
    rank: int = 0
    n_peers: int = 0
    unusual: tuple[tuple[str, float], ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.n_peers > 0 and not math.isnan(self.score)

    @property
    def reading(self) -> str:
        if not self.ok:
            return "Not enough comparable accounts to place this company against its peers."
        if not self.unusual:
            return (
                f"Sits inside the peer distribution on every input ({self.n_peers} "
                "companies compared). Nothing here stands out."
            )
        listed = ", ".join(f"{name} ({value:,.2f})" for name, value in self.unusual)
        return (
            f"Unusual against its {self.n_peers} peers on: {listed}. An outlier is "
            "a reason to read the filings, not evidence of anything — a genuinely "
            "different business model sits outside the crowd by definition."
        )


def forensic_outliers(
    subject: Mapping[str, float],
    peers: Sequence[Mapping[str, float]],
    symbol: str = "",
    contamination: float = 0.1,
) -> AnomalyScore:
    """Isolation-forest outlier score for ``subject`` against ``peers``.

    Reports *which* features are unusual, not just that the company is. A bare
    outlier flag on a financial statement is unactionable -- "this company is
    unusual" tells a reader nothing they can check, while "its debt-to-equity is
    4.2 against a peer median of 0.8" is a fact they can open the annual report
    and verify.

    Missing features are dropped from the comparison rather than imputed: a
    company that reports no inventory cannot be compared on inventory, and
    filling in the peer median would place it exactly where it does not belong.
    """
    features = [k for k in subject if not math.isnan(_num(subject.get(k)))]
    usable = [
        key
        for key in features
        if sum(1 for p in peers if not math.isnan(_num(p.get(key)))) >= max(3, len(peers) // 2)
    ]
    if not usable:
        return AnomalyScore(
            symbol=symbol,
            notices=("No metric carried a value for both this company and its peers.",),
        )

    rows = np.asarray([[_num(p.get(k)) for k in usable] for p in peers], dtype="float64")
    keep = np.isfinite(rows).all(axis=1)
    rows = rows[keep]
    if rows.shape[0] < MIN_ROWS_FOR_FOREST:
        return AnomalyScore(
            symbol=symbol,
            n_peers=int(rows.shape[0]),
            notices=(
                f"Only {rows.shape[0]} peer(s) report every comparison input; an "
                f"isolation forest on fewer than {MIN_ROWS_FOR_FOREST} rows fits noise.",
            ),
        )

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return AnomalyScore(
            symbol=symbol,
            notices=("Outlier detection needs scikit-learn, which is not installed.",),
        )

    scaler = StandardScaler().fit(rows)
    forest = IsolationForest(random_state=0, contamination=contamination, n_estimators=200)
    forest.fit(scaler.transform(rows))
    vector = np.asarray([[_num(subject.get(k)) for k in usable]], dtype="float64")
    if not np.isfinite(vector).all():
        return AnomalyScore(
            symbol=symbol,
            n_peers=int(rows.shape[0]),
            notices=("This company is missing at least one of the comparison inputs.",),
        )

    # score_samples: lower is more anomalous. Negated so a larger number means
    # "more unusual", which is the direction a reader expects from a score.
    raw = -float(forest.score_samples(scaler.transform(vector))[0])
    population = -forest.score_samples(scaler.transform(rows))
    rank = int(np.sum(population <= raw))
    percentile = rank / max(len(population), 1)

    scaled = scaler.transform(vector)[0]
    deviations = sorted(
        ((k, float(v)) for k, v in zip(usable, scaled, strict=True)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    # Two standard deviations from the peer mean, in the peer's own units after
    # standardisation. Anything closer is ordinary variation.
    unusual = tuple(
        (name, subject.get(name, float("nan"))) for name, z in deviations if abs(z) >= 2.0
    )

    notices: list[str] = []
    if math.isnan(percentile):
        notices.append("The peer population was too small for a percentile.")

    return AnomalyScore(
        symbol=symbol,
        score=raw,
        rank=rank,
        n_peers=int(rows.shape[0]),
        unusual=unusual,
        notices=tuple(notices),
    )


# ---------------------------------------------------------------------------
# the supervised signal -- the least safe claim, reported most carefully
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReturnSignal:
    """Out-of-sample performance of a forward-return model on one instrument.

    Carries no "predicted price". The honest output of this exercise is the
    *measured* out-of-sample skill, and a predicted price presented without it
    would be a number with no evidence attached.
    """

    model: str = "ridge"
    horizon: int = 20
    rank_ic: float = float("nan")
    ic_by_fold: tuple[float, ...] = ()
    skill: float = float("nan")
    directional: float = float("nan")
    n_obs: int = 0
    n_folds: int = 0
    importance: tuple[tuple[str, float], ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        return not math.isnan(self.rank_ic) and self.n_obs > 0

    @property
    def ic_spread(self) -> float:
        """Standard deviation of the per-fold ICs -- the honest error bar.

        More informative than the pooled IC, because a model with a decent mean
        IC and a fold-to-fold spread wider than the mean is a model that is
        right on average for reasons that will not repeat.
        """
        folds = [v for v in self.ic_by_fold if not math.isnan(v)]
        if len(folds) < 2:
            return float("nan")
        return float(np.std(folds, ddof=1))

    @property
    def indistinguishable_from_noise(self) -> bool:
        """The headline judgement: is this anything at all?"""
        if not self.measured:
            return True
        if abs(self.rank_ic) <= NO_EDGE_IC:
            return True
        spread = self.ic_spread
        # The fold-to-fold variation being as large as the effect itself means
        # the mean is not distinguishable from zero in any useful sense.
        return not math.isnan(spread) and spread >= abs(self.rank_ic)

    @property
    def verdict(self) -> str:
        if not self.measured:
            return (
                "No out-of-sample measurement was possible — there is not enough "
                "history to train on and still test on data the model never saw."
            )
        if self.indistinguishable_from_noise:
            return (
                f"Out-of-sample rank IC {self.rank_ic:+.3f} across {self.n_folds} fold(s) "
                f"— indistinguishable from no relationship. This is the normal result "
                f"and it is not a defect in the model."
            )
        return (
            f"Out-of-sample rank IC {self.rank_ic:+.3f} across {self.n_folds} fold(s), "
            f"fold spread {self.ic_spread:.3f}. Positive, but this is a weekly-scale "
            "statistical tendency on one company's history — not a price target."
        )

    @property
    def honesty_note(self) -> str:
        """The caveat that must be rendered beside any number from here."""
        return (
            "**This is not a price prediction and it is not a recommendation.** Rank IC "
            "measures whether the model's *ordering* of days matched the ordering of "
            "what the price did next, out of sample. A value near zero — which is what "
            "this normally is — means the model found nothing, and reporting that is "
            "the point. The features are price- and volume-derived and are "
            "reconstructable point-in-time; any figure taken from the statements is "
            "restated rather than point-in-time and is exposed to look-ahead bias."
        )


def forward_return_signal(
    data: object,
    horizon: int = 20,
    model: str = "ridge",
    with_importance: bool = False,
) -> ReturnSignal:
    """Run the walk-forward forecaster and reduce it to its honest measurements.

    A thin adapter over :func:`src.analytics.forecast.run_forecast`, which
    already does the purged walk-forward validation and the rank-IC scoring.
    Nothing is re-implemented here; what this function adds is the *reading* --
    the fold spread and the noise judgement that decide whether the number is
    worth a reader's attention.
    """
    try:
        from src.analytics import forecast as fc
    except ImportError as exc:  # pragma: no cover - import-time only
        return ReturnSignal(
            model=model, horizon=horizon, notices=(f"The forecast engine is missing ({exc}).",)
        )

    try:
        run = fc.run_forecast(data, horizon=horizon, model=model, with_importance=with_importance)
    except Exception as exc:  # noqa: BLE001 - a failed fit is a page state, not a crash
        return ReturnSignal(
            model=model,
            horizon=horizon,
            notices=(f"The walk-forward fit failed: {type(exc).__name__}: {exc}",),
        )

    if not run.get("ok"):
        return ReturnSignal(
            model=str(run.get("model", model)),
            horizon=int(run.get("horizon", horizon) or horizon),
            notices=(str(run.get("reason", "The forecast engine returned no result.")),),
        )

    metrics = run.get("metrics") or {}
    per_fold = run.get("per_fold")
    fold_ics: list[float] = []
    if isinstance(per_fold, pd.DataFrame) and "ic" in per_fold.columns:
        fold_ics = [float(v) for v in pd.to_numeric(per_fold["ic"], errors="coerce").dropna()]
    elif isinstance(per_fold, Sequence):
        fold_ics = [
            _num(getattr(f, "ic", None)) if not isinstance(f, dict) else _num(f.get("ic"))
            for f in per_fold
        ]
        fold_ics = [v for v in fold_ics if not math.isnan(v)]

    importance: tuple[tuple[str, float], ...] = ()
    raw_importance = run.get("importance")
    if isinstance(raw_importance, pd.DataFrame) and not raw_importance.empty:
        column = (
            "importance" if "importance" in raw_importance.columns else raw_importance.columns[-1]
        )
        name_column = (
            "feature" if "feature" in raw_importance.columns else raw_importance.columns[0]
        )
        importance = tuple(
            (str(name), float(value))
            for name, value in zip(
                raw_importance[name_column], raw_importance[column], strict=False
            )
            if math.isfinite(_num(value))
        )

    return ReturnSignal(
        model=str(run.get("model", model)),
        horizon=int(run.get("horizon", horizon) or horizon),
        rank_ic=_num(metrics.get("ic")),
        ic_by_fold=tuple(fold_ics),
        skill=_num(metrics.get("skill")),
        directional=_num(metrics.get("directional")),
        n_obs=int(run.get("n_obs", 0) or 0),
        n_folds=len(fold_ics),
        importance=importance,
        notices=tuple(run.get("notices") or ()),
    )
