"""Supervised forward-return models: the machine-learning forecaster.

What this module does, precisely: it fits a regression from the features in
:mod:`src.analytics.features` -- all of them known at the close of session
``t`` -- onto ``log(C_{t+h} / C_t)``, the log return over the next ``h``
sessions. It then measures that regression **out of sample, against a random
walk**, and reports the answer with the prediction interval its own errors
imply.

It is not a crystal ball, and the architecture is built to stop it pretending
to be one.

Why a *forecast* is offered here when the fundamentals panel refuses one
=======================================================================

The Security page states that no return forecast can be built from annual
accounts, and that remains true: five annual periods give four one-year
training pairs and zero five-year ones, so the model would be fitted to no
observations at all. That is a *data volume* problem, and it does not exist
here. A ten-year daily history is ~2,500 rows, which is enough to fit a
regularised model, hold out 30% of it in sequential blocks, and get an error
estimate with a standard error worth quoting. Whether the resulting model has
any *skill* is a separate question -- and the whole point of this module is
that the question gets answered rather than assumed.

The four guards that make the numbers honest
============================================

**Purged walk-forward validation.**
    The data is split into sequential expanding-window folds: train on
    everything up to a point, test on the block that follows, move forward,
    repeat. No shuffling, ever -- a shuffled split lets the model train on
    Thursday to predict Wednesday, which is how a published stock predictor
    reports 95% accuracy. Between each training block and the test block that
    follows it, ``h - 1 + purge_extra`` rows are **purged**: a label at
    horizon ``h`` spans ``h`` future sessions, so the last few training labels
    physically overlap the test window.

**A random walk is the benchmark, not zero.**
    Every skill number is quoted against "the price ``h`` days from now is the
    price today" -- a forecast of exactly zero log return. That is the
    benchmark a price forecaster has to beat, and most do not. ``skill = 1 -
    MAE_model / MAE_naive`` is therefore *negative* for a model that is worse
    than doing nothing, and the UI says so rather than reporting the raw error
    where any number looks small.

**Conformal prediction intervals.**
    The band around a forecast is the empirical quantile range of the model's
    own out-of-sample residuals, not a Gaussian assumption and not the
    training fit. A bad model produces a wide band automatically. Signed
    quantiles rather than a symmetric ``+/- q`` so a skewed error distribution
    stays skewed.

**Diebold-Mariano, with HAC errors.**
    "Better than the random walk" is a claim that needs a test. The
    Diebold-Mariano statistic on the squared-error loss differential supplies
    one, with a Newey-West variance at ``h - 1`` lags (overlapping horizons
    make the loss differential serially correlated by construction) and the
    Harvey-Leybourne-Newbold small-sample correction.

The model menu
==============

============== ============= ==================================================
key            family        what it assumes
============== ============= ==================================================
random_walk    baseline      nothing. Predicts zero forward return; the
                             reference every skill number is quoted against
drift          baseline      the sample mean forward return persists
ridge          linear        a linear relationship, with L2 shrinkage chosen
                             by an inner time-series CV
elastic_net    linear        linear and SPARSE -- L1 drives most coefficients
                             to exactly zero, which is its own feature-selection
random_forest  tree ensemble non-linear thresholds and interactions, averaged
                             over bootstrapped shallow trees
gradient_boost tree ensemble sequential residual fitting; the most flexible
                             model here and the easiest to overfit
ensemble       blend         inverse-error weighted blend of ridge, forest and
                             boosting, weights learned on an inner holdout
============== ============= ==================================================

The two baselines are not filler. On daily equity data they routinely win, and
a menu whose weakest entry is a tuned gradient booster would hide that.

Dependencies and degradation
============================

``scikit-learn`` is required for everything but the two baselines. It is a
declared dependency, but if it is missing the module still imports, the
baselines still run, and every other model reports ``ok=False`` with the
reason -- rather than taking the application down at import time.

No I/O, no Streamlit, no global state. Deterministic for a given seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import features as feat
from src.config import (
    FEATURES,
    FORECAST,
    FORECAST_DIRECTION_EDGE,
    FORECAST_MIN_SKILL,
    TECHNICALS,
    TRADING_DAYS,
    FeatureConfig,
    ForecastConfig,
    TechnicalConfig,
)
from src.models import Insight

__all__ = [
    "CATEGORY",
    "DEFAULT_LEADERBOARD",
    "MODELS",
    "SKLEARN_AVAILABLE",
    "Fold",
    "ModelSpec",
    "available_models",
    "best_model",
    "conformal_bounds",
    "diebold_mariano",
    "forecast_insights",
    "forecast_path",
    "make_folds",
    "model_leaderboard",
    "portfolio_forecast",
    "resolve_model",
    "run_forecast",
    "score_predictions",
    "walk_forward",
]

# ---------------------------------------------------------------------------
# optional dependency
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by whichever branch the environment has
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    _SKLEARN_REASON = ""
except ImportError as exc:  # pragma: no cover - only on a broken install
    SKLEARN_AVAILABLE = False
    _SKLEARN_REASON = (
        f"scikit-learn is not installed ({exc}). Only the random-walk and drift "
        "baselines are available; install it with `pip install -r requirements.txt`."
    )

# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------
#: Category carried by every :class:`~src.models.Insight` this module emits.
CATEGORY: str = "forecast"

#: Models raced against each other by default on the leaderboard.
DEFAULT_LEADERBOARD: tuple[str, ...] = (
    "random_walk",
    "drift",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boost",
    "ensemble",
)

#: Members of the blended model, in a fixed order so its weights are readable.
_ENSEMBLE_MEMBERS: tuple[str, ...] = ("ridge", "random_forest", "gradient_boost")

#: Fraction of each training block the ensemble holds out to learn its weights.
_ENSEMBLE_HOLDOUT: float = 0.25

#: Minimum rows in an ensemble weighting holdout before it is attempted at all.
_MIN_ENSEMBLE_HOLDOUT_ROWS: int = 30

#: Guard against a zero denominator in an inverse-error weight.
_EPSILON: float = 1e-12

#: Inner time-series CV folds used by RidgeCV / ElasticNetCV to pick a penalty.
_INNER_CV_SPLITS: int = 3

#: Minimum out-of-sample predictions before a metric is reported at all.
#: Below this, directional accuracy is a coin flip measured four times.
MIN_TEST_ROWS: int = 30


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float = float("nan")) -> float:
    """Best-effort float; ``default`` for None / non-finite / garbage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_int(value: Any, default: int) -> int:
    """Best-effort int; ``default`` for None / NaN / garbage."""
    number = _as_float(value)
    return int(number) if math.isfinite(number) else default


def _finite(values: Any) -> np.ndarray:
    """A 1-D float array with every non-finite entry removed."""
    array = np.asarray(values, dtype="float64").ravel()
    return array[np.isfinite(array)]


def _settings(cfg: Any) -> ForecastConfig:
    return cfg if isinstance(cfg, ForecastConfig) else FORECAST


# ---------------------------------------------------------------------------
# baseline estimators (no third-party dependency)
# ---------------------------------------------------------------------------
class RandomWalk:
    """Predicts a forward log return of exactly zero.

    The efficient-market null, and the reference every skill number in this
    module is quoted against. It has no parameters, cannot overfit, and on
    daily equity data it is a genuinely hard opponent -- which is the fact a
    forecasting page exists to convey.
    """

    def fit(self, x: Any, y: Any) -> RandomWalk:  # noqa: ARG002 - sklearn API
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.zeros(len(x), dtype="float64")


class DriftBaseline:
    """Predicts the mean forward return observed in the training block.

    The random walk *with drift*. It carries exactly one estimated number, and
    that number's standard error is usually larger than the number itself --
    which is why a model that beats it is worth more attention than a model
    that merely beats zero.
    """

    def __init__(self) -> None:
        self.drift_: float = 0.0

    def fit(self, x: Any, y: Any) -> DriftBaseline:  # noqa: ARG002 - sklearn API
        sample = _finite(y)
        self.drift_ = float(np.mean(sample)) if sample.size else 0.0
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.full(len(x), self.drift_, dtype="float64")


class InverseErrorEnsemble:
    """Weighted blend whose weights are learned on an inner, later holdout.

    ``fit`` splits the training block chronologically, fits every member on the
    earlier part, scores them on the later part, sets ``w_i ∝ 1 / MAE_i``, then
    **refits every member on the whole training block**. The weighting never
    sees a row outside the training block, so the blend is as leak-free as its
    members and can be dropped straight into the same walk-forward.

    Falls back to equal weights when the inner holdout is too short to
    distinguish the members -- an honest tie rather than a weighting fitted to
    fifteen observations.
    """

    def __init__(self, members: Sequence[tuple[str, Any]], holdout: float = _ENSEMBLE_HOLDOUT):
        self.members = list(members)
        self.holdout = float(holdout)
        self.weights_: dict[str, float] = {}

    def fit(self, x: Any, y: Any) -> InverseErrorEnsemble:
        frame = pd.DataFrame(x)
        target = pd.Series(np.asarray(y, dtype="float64").ravel())
        n_rows = len(frame)
        cut = int(round(n_rows * (1.0 - self.holdout)))
        usable = n_rows - cut >= _MIN_ENSEMBLE_HOLDOUT_ROWS and cut >= _MIN_ENSEMBLE_HOLDOUT_ROWS

        errors: dict[str, float] = {}
        if usable:
            for name, model in self.members:
                try:
                    model.fit(frame.iloc[:cut], target.iloc[:cut])
                    predicted = np.asarray(model.predict(frame.iloc[cut:]), dtype="float64")
                    errors[name] = float(np.mean(np.abs(target.iloc[cut:].to_numpy() - predicted)))
                except Exception:  # noqa: BLE001 - a failed member drops out of the blend
                    errors[name] = float("inf")

        raw = {
            name: (
                1.0 / (errors[name] + _EPSILON)
                if math.isfinite(errors.get(name, float("inf")))
                else 0.0
            )
            for name, _ in self.members
        }
        total = sum(raw.values())
        if total <= 0.0:
            share = 1.0 / max(len(self.members), 1)
            raw = {name: share for name, _ in self.members}
            total = 1.0
        self.weights_ = {name: value / total for name, value in raw.items()}

        for _, model in self.members:
            model.fit(frame, target)
        return self

    def predict(self, x: Any) -> np.ndarray:
        frame = pd.DataFrame(x)
        blended = np.zeros(len(frame), dtype="float64")
        for name, model in self.members:
            weight = self.weights_.get(name, 0.0)
            if weight <= 0.0:
                continue
            blended += weight * np.asarray(model.predict(frame), dtype="float64")
        return blended


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """One entry on the model menu, with its honest assumption and weakness.

    ``assumption`` and ``weakness`` are rendered verbatim in the UI. A model
    whose limitation cannot be stated in a sentence has no business being on a
    page that draws a forward cone.
    """

    key: str
    label: str
    family: str  # baseline | linear | tree | blend
    description: str
    assumption: str
    weakness: str
    needs_sklearn: bool
    #: Factory taking the config and returning a fresh, unfitted estimator.
    #: A factory rather than an instance so every fold gets a clean model:
    #: re-fitting one shared object across folds is how state leaks between
    #: them.
    build: Callable[[ForecastConfig], Any] = field(repr=False)

    @property
    def is_baseline(self) -> bool:
        return self.family == "baseline"


def _inner_cv() -> Any:
    """Chronological inner CV used to select a regularisation strength."""
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(n_splits=_INNER_CV_SPLITS)


def _build_ridge(cfg: ForecastConfig) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RidgeCV(alphas=np.asarray(cfg.ridge_alphas, dtype="float64"), cv=_inner_cv()),
            ),
        ]
    )


def _build_elastic_net(cfg: ForecastConfig) -> Any:
    """Elastic net with BOTH sides standardised.

    The target is a log return with a standard deviation around 0.02, and
    coordinate descent's convergence tolerance is scaled by the target's own
    variance. Left unscaled it hits ``max_iter`` and emits a convergence
    warning on almost every fold, so the target goes through its own scaler
    and the prediction is inverse-transformed back into return units.
    """
    inner = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                # An explicit alpha array rather than the automatic path: the
                # keyword that sizes that path was renamed between scikit-learn
                # 1.6 and 1.9, and the path's lower end reaches alphas where
                # coordinate descent on collinear price features never
                # converges. An array is accepted by every version.
                ElasticNetCV(
                    l1_ratio=list(cfg.elastic_l1_ratios),
                    alphas=np.asarray(cfg.elastic_alphas, dtype="float64"),
                    cv=_inner_cv(),
                    max_iter=5000,
                    tol=1e-4,
                    selection="random",
                    random_state=cfg.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def _build_random_forest(cfg: ForecastConfig) -> Any:
    # ``n_jobs=-1`` halves the wall-clock cost of every walk-forward. The FOREST
    # is identical at any thread count -- each tree's randomness is drawn from
    # the seeded parent generator before any of them is dispatched -- but the
    # prediction is accumulated across threads, so the floating-point summation
    # ORDER varies and two runs agree to about 1e-15 relative rather than
    # bit-exactly. That is far below the precision of anything displayed (a
    # price to two decimals, a skill score to four), and it is the one place in
    # this application where a repeated run is not bit-identical. Set
    # ``n_jobs=1`` here if exact byte-level reproducibility ever matters more
    # than the speed.
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        bootstrap=True,
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _build_gradient_boost(cfg: ForecastConfig) -> Any:
    """Histogram gradient boosting, with early stopping deliberately OFF.

    sklearn's built-in early stopping carves its validation slice out with a
    SHUFFLED split, which on a time series means stopping on rows that sit
    inside the training window's own future. The iteration count is fixed and
    the regularisation is carried by a shallow depth, a large leaf minimum and
    an explicit L2 penalty instead -- slower to tune, honest by construction.
    """
    return HistGradientBoostingRegressor(
        max_iter=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=min(cfg.max_depth, 3),
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=cfg.random_state,
    )


def _build_ensemble(cfg: ForecastConfig) -> Any:
    return InverseErrorEnsemble(
        [(key, MODELS[key].build(cfg)) for key in _ENSEMBLE_MEMBERS],
    )


MODELS: dict[str, ModelSpec] = {
    "random_walk": ModelSpec(
        key="random_walk",
        label="Random walk",
        family="baseline",
        description="Predicts a forward return of exactly zero.",
        assumption="Prices are a martingale: today's price is the best estimate of every "
        "future price, so no feature carries usable information.",
        weakness="It cannot be right about anything specific. It is here because it is the "
        "opponent every other model has to beat, and frequently does not.",
        needs_sklearn=False,
        build=lambda _cfg: RandomWalk(),
    ),
    "drift": ModelSpec(
        key="drift",
        label="Historical drift",
        family="baseline",
        description="Predicts the average forward return seen in training.",
        assumption="The instrument's mean return is stable, and the future looks like the "
        "average of the past.",
        weakness="The standard error of a mean return estimated from a few years of daily "
        "data is usually larger than the estimate, so this is a very noisy constant.",
        needs_sklearn=False,
        build=lambda _cfg: DriftBaseline(),
    ),
    "ridge": ModelSpec(
        key="ridge",
        label="Ridge regression",
        family="linear",
        description="Linear model with L2 shrinkage, penalty chosen by inner time-series CV.",
        assumption="The forward return is a linear function of the features, with correlated "
        "features sharing their coefficient rather than fighting over it.",
        weakness="Cannot represent an interaction or a threshold. Every coefficient is a "
        "single number applied in every market regime.",
        needs_sklearn=True,
        build=_build_ridge,
    ),
    "elastic_net": ModelSpec(
        key="elastic_net",
        label="Elastic net",
        family="linear",
        description="Linear model mixing L1 and L2 penalties; performs its own selection.",
        assumption="Most of the features carry no signal, and the ones that do act linearly. "
        "L1 sets the rest to exactly zero.",
        weakness="With correlated features the survivor of a correlated group is close to "
        "arbitrary, so the sparsity pattern is less stable than it looks.",
        needs_sklearn=True,
        build=_build_elastic_net,
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random forest",
        family="tree",
        description="Bootstrapped shallow decision trees, averaged.",
        assumption="The relationship is non-linear and involves thresholds and interactions "
        "that a linear model cannot see.",
        weakness="It can only interpolate within the range of the training data — it cannot "
        "extrapolate a trend, and its averaging pulls every prediction toward the sample mean.",
        needs_sklearn=True,
        build=_build_random_forest,
    ),
    "gradient_boost": ModelSpec(
        key="gradient_boost",
        label="Gradient boosting",
        family="tree",
        description="Trees fitted sequentially to the residuals of the trees before them.",
        assumption="A weak learner applied repeatedly to what is left over can approximate "
        "almost any function of the features.",
        weakness="The most flexible model here, so on a near-zero signal-to-noise series it is "
        "the one most likely to fit noise. Watch the gap between its training fit and its "
        "walk-forward skill.",
        needs_sklearn=True,
        build=_build_gradient_boost,
    ),
    "ensemble": ModelSpec(
        key="ensemble",
        label="Blended ensemble",
        family="blend",
        description="Ridge, forest and boosting, inverse-error weighted on an inner holdout.",
        assumption="The members make partly independent errors, so averaging cancels some of "
        "the variance without adding bias.",
        weakness="If every member is wrong in the same direction — which is what a regime "
        "change produces — blending them changes nothing but the confidence they project.",
        needs_sklearn=True,
        build=_build_ensemble,
    ),
}

#: Spellings a UI or a saved config might supply, mapped to a registry key.
_MODEL_ALIASES: dict[str, str] = {
    "naive": "random_walk",
    "random_walk": "random_walk",
    "randomwalk": "random_walk",
    "rw": "random_walk",
    "zero": "random_walk",
    "drift": "drift",
    "mean": "drift",
    "ridge": "ridge",
    "l2": "ridge",
    "linear": "ridge",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "enet": "elastic_net",
    "lasso": "elastic_net",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "forest": "random_forest",
    "gradient_boost": "gradient_boost",
    "gradient_boosting": "gradient_boost",
    "gbm": "gradient_boost",
    "boosting": "gradient_boost",
    "ensemble": "ensemble",
    "blend": "ensemble",
}


def resolve_model(model: str | None) -> str:
    """Map a user or UI spelling onto a registry key.

    Unknown spellings resolve to ``"ridge"`` -- the simplest model that can
    actually use a feature, and the one least likely to embarrass itself on a
    short history.
    """
    if model is None:
        return "ridge"
    key = str(model).strip().lower().replace(" ", "_").replace("-", "_")
    return _MODEL_ALIASES.get(key, "ridge")


def available_models() -> tuple[str, ...]:
    """Registry keys this installation can actually fit, in menu order."""
    return tuple(key for key, spec in MODELS.items() if SKLEARN_AVAILABLE or not spec.needs_sklearn)


# ---------------------------------------------------------------------------
# walk-forward split construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One expanding-window train/test block, as positional row indices.

    ``train_end`` is exclusive and already excludes the purged rows, so
    ``train_end <= test_start`` always holds with a gap of exactly
    ``purge`` rows between them.
    """

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge: int

    @property
    def n_train(self) -> int:
        return max(self.train_end - self.train_start, 0)

    @property
    def n_test(self) -> int:
        return max(self.test_end - self.test_start, 0)


def make_folds(n_rows: int, horizon: int, cfg: ForecastConfig = FORECAST) -> tuple[Fold, ...]:
    """Sequential expanding-window folds with a purge gap, newest last.

    The last ``test_fraction`` of the rows is divided into ``n_splits``
    contiguous blocks. Fold ``k`` trains on everything from row 0 up to
    ``purge`` rows before its test block, where ``purge = h - 1 + purge_extra``.

    That gap is not decoration. The label at row ``t`` is the return from ``t``
    to ``t + h``; without the purge, the final ``h - 1`` training labels are
    partly composed of the very returns the test block is about to be scored
    on. For a 63-day horizon that is three months of the answer sitting inside
    the training set.

    Returns an empty tuple when the data cannot support even one fold with a
    training block of at least ``min_initial_train`` rows.
    """
    settings = _settings(cfg)
    rows = max(_as_int(n_rows, 0), 0)
    steps = max(_as_int(horizon, 1), 1)
    purge = max(steps - 1 + max(settings.purge_extra, 0), 0)
    n_splits = max(_as_int(settings.n_splits, 1), 1)

    test_total = int(round(rows * min(max(settings.test_fraction, 0.05), 0.9)))
    if test_total < n_splits:
        return ()
    block = test_total // n_splits
    if block < 1:
        return ()

    first_test = rows - test_total
    folds: list[Fold] = []
    for k in range(n_splits):
        test_start = first_test + k * block
        # The last block absorbs the remainder so no observation is discarded.
        test_end = rows if k == n_splits - 1 else test_start + block
        train_end = test_start - purge
        if train_end < settings.min_initial_train or test_end - test_start < 1:
            continue
        folds.append(
            Fold(
                index=len(folds),
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge=purge,
            )
        )
    return tuple(folds)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def diebold_mariano(
    model_errors: Any, benchmark_errors: Any, horizon: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss, HAC-corrected.

    Tests ``H0: the two forecasts have equal expected squared error`` against a
    two-sided alternative. The loss differential is
    ``d_t = e_bench^2 - e_model^2``, so a POSITIVE statistic means the model
    beat the benchmark.

    Overlapping horizons make ``d_t`` serially correlated by construction, so
    its variance is estimated with a Newey-West (Bartlett) kernel at ``h - 1``
    lags -- the same HAC convention the factor-regression module uses, and for
    the same reason: the uncorrected variance is too small, which turns noise
    into significance.

    The Harvey-Leybourne-Newbold small-sample correction is applied and the
    statistic referred to a ``t(n - 1)`` distribution rather than a Normal,
    because a walk-forward on daily data produces tens of test points, not
    thousands.

    Returns ``(nan, nan)`` when the differential is degenerate (identical
    forecasts, or fewer than eight usable points).
    """
    model = np.asarray(model_errors, dtype="float64").ravel()
    bench = np.asarray(benchmark_errors, dtype="float64").ravel()
    if model.size != bench.size or model.size < 8:
        return float("nan"), float("nan")

    differential = bench**2 - model**2
    differential = differential[np.isfinite(differential)]
    n = differential.size
    if n < 8:
        return float("nan"), float("nan")

    mean = float(np.mean(differential))
    centred = differential - mean
    lags = max(min(int(horizon) - 1, n - 2), 0)

    variance = float(np.dot(centred, centred) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    if not math.isfinite(variance) or variance <= 0.0:
        return float("nan"), float("nan")

    statistic = mean / math.sqrt(variance / n)

    # Harvey-Leybourne-Newbold: the HAC variance is biased downward in small
    # samples, and the bias grows with the horizon.
    correction = (n + 1.0 - 2.0 * (lags + 1) + (lags + 1) * lags / n) / n
    if correction > 0.0:
        statistic *= math.sqrt(correction)

    try:
        from scipy import stats

        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    except Exception:  # noqa: BLE001 - scipy is a hard dependency; be safe anyway
        p_value = float("nan")
    return float(statistic), p_value


def score_predictions(y_true: Any, y_pred: Any, horizon: int = 1) -> dict[str, float]:
    """Out-of-sample scorecard for one set of predictions.

    Every key is a DECIMAL log return or a dimensionless ratio.

    ============== ============================================================
    key            meaning
    ============== ============================================================
    mae, rmse      average absolute / root-mean-squared error, in log-return
                   units over the whole horizon
    mae_naive      the same error for a forecast of exactly zero -- the random
                   walk, and the denominator of every skill number below
    mase           ``mae / mae_naive``. Below 1.0 beats the random walk
    skill          ``1 - mase``. Positive beats the random walk; the sign is
                   what matters and the magnitude is usually tiny
    r2_oos         ``1 - SSE / SS_naive``, the Campbell-Thompson out-of-sample
                   R-squared. NEGATIVE is the normal result for a price
                   forecaster, and it means the model is worse than zero
    directional    share of predictions whose SIGN was right, over the rows
                   where the model actually called a direction AND the realised
                   move was not exactly flat. A constant-zero forecast calls no
                   direction, so the random walk reports NaN rather than 0%
    direction_calls how many rows that share was measured on
    up_rate        share of realised moves that were positive -- the base rate
                   directional accuracy must be compared against, not 50%
    ic             Spearman rank correlation between prediction and outcome:
                   does the model at least ORDER the outcomes correctly?
    dm_stat, dm_p  Diebold-Mariano statistic and p-value against the random
                   walk. Positive statistic, small p, means a real difference
    n_test         out-of-sample observations behind all of the above
    ============== ============================================================

    Returns NaNs rather than raising for empty or misaligned input.
    """
    truth = np.asarray(y_true, dtype="float64").ravel()
    predicted = np.asarray(y_pred, dtype="float64").ravel()
    blank: dict[str, float] = {
        "mae": float("nan"),
        "rmse": float("nan"),
        "mae_naive": float("nan"),
        "mase": float("nan"),
        "skill": float("nan"),
        "r2_oos": float("nan"),
        "directional": float("nan"),
        "direction_calls": 0.0,
        "up_rate": float("nan"),
        "ic": float("nan"),
        "dm_stat": float("nan"),
        "dm_p": float("nan"),
        "n_test": 0.0,
    }
    if truth.size == 0 or truth.size != predicted.size:
        return blank

    usable = np.isfinite(truth) & np.isfinite(predicted)
    truth, predicted = truth[usable], predicted[usable]
    if truth.size == 0:
        return blank

    errors = truth - predicted
    naive_errors = truth  # a zero forecast errs by exactly the realised return

    mae = float(np.mean(np.abs(errors)))
    mae_naive = float(np.mean(np.abs(naive_errors)))
    sse = float(np.dot(errors, errors))
    sse_naive = float(np.dot(naive_errors, naive_errors))

    out = dict(blank)
    out["n_test"] = float(truth.size)
    out["mae"] = mae
    out["rmse"] = float(math.sqrt(sse / truth.size))
    out["mae_naive"] = mae_naive
    if mae_naive > _EPSILON:
        out["mase"] = mae / mae_naive
        out["skill"] = 1.0 - mae / mae_naive
    if sse_naive > _EPSILON:
        out["r2_oos"] = 1.0 - sse / sse_naive

    moved = np.abs(truth) > _EPSILON
    if moved.any():
        out["up_rate"] = float(np.mean(truth[moved] > 0.0))
        # A forecast of exactly zero calls no direction at all. Scoring it as
        # wrong every time would report the random walk at 0% accuracy, which
        # reads as "always wrong" rather than "never answered the question".
        called = moved & (np.abs(predicted) > _EPSILON)
        if called.any():
            out["directional"] = float(
                np.mean(np.sign(predicted[called]) == np.sign(truth[called]))
            )
            out["direction_calls"] = float(np.count_nonzero(called))

    if truth.size >= 8 and np.std(predicted) > _EPSILON:
        try:
            from scipy import stats

            correlation = stats.spearmanr(predicted, truth).statistic
            out["ic"] = _as_float(correlation)
        except Exception:  # noqa: BLE001 - a missing correlation is not a failure
            out["ic"] = float("nan")

    statistic, p_value = diebold_mariano(errors, naive_errors, horizon)
    out["dm_stat"] = statistic
    out["dm_p"] = p_value
    return out


# ---------------------------------------------------------------------------
# conformal intervals
# ---------------------------------------------------------------------------
def conformal_bounds(
    residuals: Any, alpha: float = FORECAST.conformal_alpha
) -> tuple[float, float]:
    """Signed residual quantiles defining a ``1 - alpha`` prediction interval.

    Split-conformal, in its simplest honest form: the interval around a new
    prediction is ``[pred + q_lo, pred + q_hi]`` where the two quantiles come
    from the model's own OUT-OF-SAMPLE residuals ``y_true - y_pred``.

    Signed rather than the usual symmetric ``+/- |r|`` quantile, because
    equity returns are left-skewed and so are the errors of a model fitted to
    them. A symmetric band would overstate the upside and understate the
    downside by the same amount, which is the wrong error to make here.

    Marginal coverage, not conditional: the band covers ``1 - alpha`` of
    outcomes *on average over the backtest*, and says nothing about whether
    today in particular is a calm or a violent day. Returns ``(nan, nan)``
    when there are too few residuals to quantify anything.
    """
    sample = _finite(residuals)
    level = _as_float(alpha, FORECAST.conformal_alpha)
    if sample.size < 8 or not (0.0 < level < 1.0):
        return float("nan"), float("nan")
    low = float(np.quantile(sample, level / 2.0))
    high = float(np.quantile(sample, 1.0 - level / 2.0))
    return low, high


def _direction_probability(point: float, residuals: Any) -> float:
    """P(price is higher in ``h`` sessions), from the empirical error law.

    The predictive distribution of the forward log return is taken to be
    ``point + residual`` where ``residual`` is drawn from the out-of-sample
    residual sample. The probability of a rise is then simply the share of
    that distribution above zero -- no Normal assumption, and automatically
    close to the base rate whenever the model has no skill.
    """
    sample = _finite(residuals)
    centre = _as_float(point)
    if sample.size < 8 or not math.isfinite(centre):
        return float("nan")
    return float(np.mean(centre + sample > 0.0))


# ---------------------------------------------------------------------------
# the walk-forward itself
# ---------------------------------------------------------------------------
def walk_forward(
    x: pd.DataFrame,
    y: pd.Series,
    model: str = "ridge",
    horizon: int = 1,
    cfg: ForecastConfig = FORECAST,
) -> dict[str, Any]:
    """Fit and score one model across every purged expanding-window fold.

    ``x`` and ``y`` must already be restricted to LABELLED rows -- see
    :func:`src.analytics.features.supervised_matrix`, whose ``y.notna()`` mask
    is the boundary between "can be trained on" and "can only be predicted
    from".

    Returns ``{ok, reason, model, horizon, folds, n_train_last, predictions,
    residuals, metrics, per_fold}``. ``predictions`` is a frame indexed like
    the test rows with columns ``y_true, y_pred, fold``; ``per_fold`` carries
    the same scorecard computed inside each block, which is what exposes a
    model that worked until 2022 and stopped.

    Never raises: a model that blows up inside a fold is reported as a failed
    fold, and a run with no usable folds returns ``ok=False`` with the reason.
    """
    settings = _settings(cfg)
    key = resolve_model(model)
    spec = MODELS[key]
    steps = max(_as_int(horizon, 1), 1)

    failure: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "model": key,
        "label": spec.label,
        "horizon": steps,
        "folds": 0,
        "n_train_last": 0,
        "predictions": pd.DataFrame(columns=["y_true", "y_pred", "fold"]),
        "residuals": np.empty(0, dtype="float64"),
        "metrics": score_predictions([], [], steps),
        "per_fold": pd.DataFrame(),
    }

    if spec.needs_sklearn and not SKLEARN_AVAILABLE:
        failure["reason"] = _SKLEARN_REASON
        return failure
    if not isinstance(x, pd.DataFrame) or x.empty or not isinstance(y, pd.Series) or y.empty:
        failure["reason"] = "No supervised rows were produced from this price history."
        return failure
    if len(x) != len(y):
        failure["reason"] = f"Feature matrix ({len(x)}) and labels ({len(y)}) disagree in length."
        return failure
    if len(x) < settings.min_train_rows:
        failure["reason"] = (
            f"Only {len(x)} labelled rows; {settings.min_train_rows} are required before a "
            f"walk-forward can hold out enough data to measure anything. Widen the history "
            f"window, or shorten the horizon — a {steps}-session horizon alone costs "
            f"{steps} rows off the end."
        )
        return failure

    folds = make_folds(len(x), steps, settings)
    if not folds:
        failure["reason"] = (
            f"{len(x)} labelled rows cannot be split into {settings.n_splits} test blocks with "
            f"a {steps - 1 + settings.purge_extra}-row purge and a "
            f"{settings.min_initial_train}-row minimum training block."
        )
        return failure

    pieces: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    last_train = 0

    for fold in folds:
        x_train = x.iloc[fold.train_start : fold.train_end]
        y_train = y.iloc[fold.train_start : fold.train_end]
        x_test = x.iloc[fold.test_start : fold.test_end]
        y_test = y.iloc[fold.test_start : fold.test_end]
        try:
            estimator = spec.build(settings)
            estimator.fit(x_train, y_train)
            predicted = np.asarray(estimator.predict(x_test), dtype="float64")
        except Exception as exc:  # noqa: BLE001 - one bad fold must not kill the run
            failures.append(f"fold {fold.index + 1}: {type(exc).__name__}: {exc}")
            continue

        last_train = fold.n_train
        block = pd.DataFrame(
            {"y_true": y_test.to_numpy(dtype="float64"), "y_pred": predicted, "fold": fold.index},
            index=y_test.index,
        )
        pieces.append(block)
        fold_scores = score_predictions(block["y_true"], block["y_pred"], steps)
        fold_rows.append(
            {
                "fold": fold.index + 1,
                "train_rows": fold.n_train,
                "test_rows": fold.n_test,
                "train_end": x.index[fold.train_end - 1],
                "test_start": x.index[fold.test_start],
                "test_end": x.index[fold.test_end - 1],
                **{k: fold_scores[k] for k in ("mae", "skill", "directional", "r2_oos")},
            }
        )

    if not pieces:
        failure["reason"] = "Every walk-forward fold failed to fit. " + "; ".join(failures)
        return failure

    predictions = pd.concat(pieces).sort_index()
    residuals = (predictions["y_true"] - predictions["y_pred"]).to_numpy(dtype="float64")
    metrics = score_predictions(predictions["y_true"], predictions["y_pred"], steps)

    return {
        "ok": True,
        "reason": "; ".join(failures),
        "model": key,
        "label": spec.label,
        "horizon": steps,
        "folds": len(pieces),
        "n_train_last": last_train,
        "predictions": predictions,
        "residuals": residuals[np.isfinite(residuals)],
        "metrics": metrics,
        "per_fold": pd.DataFrame(fold_rows),
    }


# ---------------------------------------------------------------------------
# feature importance
# ---------------------------------------------------------------------------
def _permutation_importance(
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cfg: ForecastConfig,
) -> pd.DataFrame:
    """Permutation importance measured on held-out rows, never on the fit.

    Each feature's values are shuffled and the increase in error recorded. A
    feature whose shuffling does not hurt carried nothing the model used,
    whatever its coefficient says. Measured out-of-sample deliberately: an
    in-sample importance ranks the features the model overfitted on.

    Returns ``feature, family, importance, std, description`` sorted by
    importance, or an empty frame when it cannot be computed.
    """
    columns = ["feature", "family", "importance", "std", "description"]
    if not SKLEARN_AVAILABLE or x.empty or len(x) < MIN_TEST_ROWS:
        return pd.DataFrame(columns=columns)
    try:
        report = permutation_importance(
            estimator,
            x,
            y,
            n_repeats=max(_as_int(cfg.importance_repeats, 5), 1),
            random_state=cfg.random_state,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
    except Exception:  # noqa: BLE001 - importance is a nicety, not the result
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        {
            "feature": list(x.columns),
            "family": [feat.family_of(c) for c in x.columns],
            "importance": np.asarray(report.importances_mean, dtype="float64"),
            "std": np.asarray(report.importances_std, dtype="float64"),
        }
    )
    frame["description"] = frame["feature"].map(feat.describe_feature)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the end-to-end forecast
# ---------------------------------------------------------------------------
def _horizon_note(key: str, horizon: int, metrics: dict[str, float], n_obs: int) -> str:
    """Plain-English verdict on what the walk-forward actually found."""
    spec = MODELS[key]
    skill = _as_float(metrics.get("skill"))
    direction = _as_float(metrics.get("directional"))
    p_value = _as_float(metrics.get("dm_p"))
    n_test = _as_int(metrics.get("n_test"), 0)

    if not math.isfinite(skill):
        return (
            f"{spec.label} produced no measurable skill estimate over {horizon} sessions; "
            f"treat the forecast as illustrative only."
        )

    if skill > FORECAST_MIN_SKILL:
        head = (
            f"Over {n_test} out-of-sample observations the {spec.label.lower()} was "
            f"{skill * 100.0:.1f}% more accurate than assuming the price does not move."
        )
    else:
        head = (
            f"Over {n_test} out-of-sample observations the {spec.label.lower()} was "
            f"{abs(skill) * 100.0:.1f}% LESS accurate than assuming the price does not move. "
            f"On this instrument, at this horizon, the model has no demonstrated edge."
        )

    if math.isfinite(p_value):
        verdict = (
            f"The Diebold-Mariano test puts the chance of seeing a difference this large by "
            f"luck alone at {p_value * 100.0:.0f}%."
        )
        if p_value > 0.10:
            verdict += " That is not evidence of skill."
    else:
        verdict = "There were too few test points to test the difference for significance."

    if math.isfinite(direction):
        base = _as_float(metrics.get("up_rate"), 0.5)
        aim = (
            f"It called the direction correctly {direction * 100.0:.0f}% of the time, against "
            f"a {base * 100.0:.0f}% base rate of up moves"
        )
        aim += (
            "."
            if direction < FORECAST_DIRECTION_EDGE
            else " — an edge worth a second look, and "
            "worth re-checking on a different window before acting on."
        )
    else:
        aim = "Directional accuracy could not be measured."

    shared = (
        f"Fitted on {n_obs} sessions of this instrument's own price history and nothing else: "
        f"no fundamentals, no news, no other instrument, no knowledge of the future. "
        f"{spec.weakness}"
    )
    return " ".join((head, verdict, aim, shared))


def run_forecast(
    data: Any,
    horizon: int = FORECAST.default_horizon,
    model: str = "ridge",
    cfg: ForecastConfig = FORECAST,
    feature_cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
    with_importance: bool = True,
) -> dict[str, Any]:
    """Build features, validate a model, and forecast the next ``horizon`` sessions.

    Parameters
    ----------
    data:
        OHLCV frame or close Series for ONE instrument.
    horizon:
        Trading sessions ahead. The label is ``log(C_{t+h} / C_t)``.
    model:
        A key from :data:`MODELS` or any alias in ``_MODEL_ALIASES``.
    with_importance:
        Compute permutation importance on the final fold. Costs a few seconds;
        turn it off when racing many models on a leaderboard.

    Returns a dict with ``ok`` and, on success:

    ``last_price, last_date, predicted_price, predicted_return, lower_price,
    upper_price, lower_return, upper_return, confidence, prob_up`` --
    the forecast and its conformal interval;

    ``metrics, per_fold, predictions, residuals, folds, n_train, n_features,
    features, importance, note, model, label, horizon`` --
    everything needed to judge whether the forecast above is worth reading.

    ``predicted_price`` is the point forecast of the MEDIAN outcome, not an
    expectation: the model predicts a log return and ``S_0 * exp(r)`` is the
    median of a distribution whose mean sits above it. That is the honest
    reading of the number and it is stated in the note.

    Never raises. Insufficient data, a missing dependency or a failed fit all
    return ``{"ok": False, "reason": ...}``.
    """
    settings = _settings(cfg)
    key = resolve_model(model)
    spec = MODELS[key]
    steps = max(_as_int(horizon, settings.default_horizon), 1)

    failure: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "model": key,
        "label": spec.label,
        "horizon": steps,
        "n_obs": 0,
        "n_train": 0,
        "n_features": 0,
    }

    if spec.needs_sklearn and not SKLEARN_AVAILABLE:
        failure["reason"] = _SKLEARN_REASON
        return failure

    x_all, y_all = feat.supervised_matrix(data, steps, feature_cfg, technical_cfg)
    if x_all.empty:
        failure["reason"] = (
            "No complete feature rows could be built from this price history. The longest "
            f"lookback window is {max(feature_cfg.range_window, TRADING_DAYS)} sessions, so a "
            "shorter history produces no usable row at all."
        )
        return failure

    labelled = y_all.notna()
    x_train, y_train = x_all.loc[labelled], y_all.loc[labelled]
    failure["n_obs"] = int(len(x_all))
    failure["n_features"] = int(x_all.shape[1])

    backtest = walk_forward(x_train, y_train, key, steps, settings)
    if not backtest["ok"]:
        failure["reason"] = backtest["reason"]
        return failure

    live = feat.latest_row(x_all)
    if live.empty:
        failure["reason"] = "The most recent session has no complete feature row."
        return failure

    try:
        estimator = spec.build(settings)
        estimator.fit(x_train, y_train)
        point = float(np.asarray(estimator.predict(live), dtype="float64")[0])
    except Exception as exc:  # noqa: BLE001 - a failed final fit is a page state
        failure["reason"] = f"The final fit on all {len(x_train)} rows failed: {exc}"
        return failure

    residuals = backtest["residuals"]
    low_q, high_q = conformal_bounds(residuals, settings.conformal_alpha)
    probability_up = _direction_probability(point, residuals)
    # The median residual is the model's measured out-of-sample BIAS. It is
    # what makes the conformal band sit off-centre around the point forecast,
    # and reporting it is better than quietly re-centring the band on zero.
    bias = float(np.median(residuals)) if residuals.size else float("nan")

    close = feat.close_series(data)
    last_price = float(close.iloc[-1])
    last_date = close.index[-1]

    importance = pd.DataFrame()
    if with_importance:
        final_fold = make_folds(len(x_train), steps, settings)
        if final_fold:
            held = final_fold[-1]
            importance = _permutation_importance(
                estimator,
                x_train.iloc[held.test_start : held.test_end],
                y_train.iloc[held.test_start : held.test_end],
                settings,
            )

    weights = getattr(estimator, "weights_", None)

    result: dict[str, Any] = {
        "ok": True,
        "reason": backtest["reason"],
        "model": key,
        "label": spec.label,
        "family": spec.family,
        "horizon": steps,
        "horizon_years": float(steps / TRADING_DAYS),
        "last_price": last_price,
        "last_date": last_date,
        "predicted_return": point,
        "predicted_price": last_price * math.exp(point),
        "lower_return": low_q + point if math.isfinite(low_q) else float("nan"),
        "upper_return": high_q + point if math.isfinite(high_q) else float("nan"),
        "lower_price": last_price * math.exp(point + low_q)
        if math.isfinite(low_q)
        else float("nan"),
        "upper_price": last_price * math.exp(point + high_q)
        if math.isfinite(high_q)
        else float("nan"),
        "confidence": float(settings.confidence),
        "prob_up": probability_up,
        "bias": bias,
        "metrics": backtest["metrics"],
        "per_fold": backtest["per_fold"],
        "predictions": backtest["predictions"],
        "residuals": residuals,
        "folds": backtest["folds"],
        "n_obs": int(len(x_all)),
        "n_train": int(len(x_train)),
        "n_features": int(x_all.shape[1]),
        "features": tuple(x_all.columns),
        "families": feat.available_families(data),
        "importance": importance,
        "ensemble_weights": dict(weights) if isinstance(weights, dict) else {},
        "purge": max(steps - 1 + settings.purge_extra, 0),
    }
    result["note"] = _horizon_note(key, steps, backtest["metrics"], int(len(x_all)))
    return result


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------
def model_leaderboard(
    data: Any,
    horizon: int = FORECAST.default_horizon,
    models: Sequence[str] = DEFAULT_LEADERBOARD,
    cfg: ForecastConfig = FORECAST,
    feature_cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
) -> pd.DataFrame:
    """Race every model over the same folds and rank them by out-of-sample skill.

    One supervised matrix, one set of folds, every model scored on identical
    test rows -- which is the only way a comparison means anything. The two
    baselines are included and are not decoration: when ``random_walk`` wins,
    that IS the finding, and a leaderboard that omitted it would be hiding it.

    Columns: ``model, label, family, skill, mase, mae, rmse, r2_oos,
    directional, ic, dm_stat, dm_p, folds, n_test, beats_naive, ok, reason``,
    sorted by ``skill`` descending. Returns an empty frame when the data
    cannot support a single fold.
    """
    settings = _settings(cfg)
    steps = max(_as_int(horizon, settings.default_horizon), 1)
    columns = [
        "model",
        "label",
        "family",
        "skill",
        "mase",
        "mae",
        "rmse",
        "r2_oos",
        "directional",
        "ic",
        "dm_stat",
        "dm_p",
        "folds",
        "n_test",
        "beats_naive",
        "ok",
        "reason",
    ]

    x_all, y_all = feat.supervised_matrix(data, steps, feature_cfg, technical_cfg)
    if x_all.empty:
        return pd.DataFrame(columns=columns)
    labelled = y_all.notna()
    x_train, y_train = x_all.loc[labelled], y_all.loc[labelled]

    rows: list[dict[str, Any]] = []
    for name in models:
        key = resolve_model(name)
        spec = MODELS[key]
        run = walk_forward(x_train, y_train, key, steps, settings)
        metrics = run["metrics"]
        rows.append(
            {
                "model": key,
                "label": spec.label,
                "family": spec.family,
                "skill": metrics["skill"],
                "mase": metrics["mase"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_oos": metrics["r2_oos"],
                "directional": metrics["directional"],
                "ic": metrics["ic"],
                "dm_stat": metrics["dm_stat"],
                "dm_p": metrics["dm_p"],
                "folds": run["folds"],
                "n_test": metrics["n_test"],
                "beats_naive": bool(
                    math.isfinite(metrics["skill"]) and metrics["skill"] > FORECAST_MIN_SKILL
                ),
                "ok": run["ok"],
                "reason": run["reason"],
            }
        )

    board = pd.DataFrame(rows, columns=columns)
    if board.empty:
        return board
    return board.sort_values("skill", ascending=False, na_position="last").reset_index(drop=True)


def best_model(board: pd.DataFrame, fallback: str = "ridge") -> str:
    """Winning NON-baseline key from a leaderboard, or ``fallback``.

    The baselines are excluded from "best" on purpose: they are the yardstick,
    and selecting the random walk as the model would leave nothing to plot.
    The UI is expected to say loudly when the yardstick won anyway.
    """
    if not isinstance(board, pd.DataFrame) or board.empty or "model" not in board.columns:
        return resolve_model(fallback)
    usable = board[board["ok"] & ~board["family"].eq("baseline")]
    usable = usable[usable["skill"].notna()]
    if usable.empty:
        return resolve_model(fallback)
    return str(usable.iloc[0]["model"])


# ---------------------------------------------------------------------------
# multi-horizon path
# ---------------------------------------------------------------------------
def forecast_path(
    data: Any,
    horizons: Sequence[int] = FORECAST.horizons,
    model: str = "ridge",
    cfg: ForecastConfig = FORECAST,
    feature_cfg: FeatureConfig = FEATURES,
    technical_cfg: TechnicalConfig = TECHNICALS,
) -> dict[str, Any]:
    """A forecast cone built from one DIRECT model per horizon.

    Direct rather than recursive, deliberately. A recursive forecast feeds its
    own one-step prediction back in as an input and iterates, which compounds
    the model's error and -- worse -- feeds it features (volatility, RSI,
    volume) that it has to invent. Fitting a separate model to each horizon's
    own label costs more compute and answers the question that was actually
    asked.

    The consequence is stated rather than hidden: the nodes are independent
    fits, so the cone is **not** guaranteed to widen monotonically, and a kink
    between two horizons is a real disagreement between two models rather than
    a plotting artefact.

    Returns ``{ok, reason, model, label, last_price, last_date, nodes, path,
    horizons}``. ``nodes`` is one row per horizon with the forecast, interval
    and skill; ``path`` interpolates between them in log space on a forward
    business-day index for drawing, with an ``is_node`` flag marking the rows
    that were actually fitted.
    """
    settings = _settings(cfg)
    key = resolve_model(model)
    spec = MODELS[key]
    wanted = sorted({max(_as_int(h, 0), 0) for h in horizons} - {0})

    out: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "model": key,
        "label": spec.label,
        "last_price": float("nan"),
        "last_date": None,
        "nodes": pd.DataFrame(),
        "path": pd.DataFrame(),
        "horizons": tuple(wanted),
    }
    if not wanted:
        out["reason"] = "No positive horizon was requested."
        return out

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    last_price, last_date = float("nan"), None

    for steps in wanted:
        run = run_forecast(
            data, steps, key, settings, feature_cfg, technical_cfg, with_importance=False
        )
        if not run["ok"]:
            reasons.append(f"{steps} sessions: {run['reason']}")
            continue
        last_price, last_date = run["last_price"], run["last_date"]
        rows.append(
            {
                "horizon": steps,
                "predicted_price": run["predicted_price"],
                "lower_price": run["lower_price"],
                "upper_price": run["upper_price"],
                "predicted_return": run["predicted_return"],
                "lower_return": run["lower_return"],
                "upper_return": run["upper_return"],
                "prob_up": run["prob_up"],
                "skill": run["metrics"]["skill"],
                "directional": run["metrics"]["directional"],
                "dm_p": run["metrics"]["dm_p"],
                "n_test": run["metrics"]["n_test"],
            }
        )

    if not rows:
        out["reason"] = "; ".join(reasons) or "No horizon could be fitted."
        return out

    nodes = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    out.update(
        {
            "ok": True,
            "reason": "; ".join(reasons),
            "last_price": last_price,
            "last_date": last_date,
            "nodes": nodes,
            "path": _interpolate_path(nodes, last_price, last_date),
        }
    )
    return out


def _interpolate_path(nodes: pd.DataFrame, last_price: float, last_date: Any) -> pd.DataFrame:
    """Daily cone for drawing: exact at the nodes, log-linear between them.

    Interpolation happens in LOG-return space, so a value halfway between a
    +2% and a +6% node is +4% compounded rather than the arithmetic midpoint
    of two prices. Every interpolated row is flagged ``is_node = False`` so a
    chart or a table can show which numbers were fitted and which were drawn.
    """
    if nodes.empty or not math.isfinite(_as_float(last_price)) or last_price <= 0.0:
        return pd.DataFrame()

    horizons = [0, *nodes["horizon"].astype(int).tolist()]
    base = math.log(float(last_price))
    logs = {
        "point": [base, *np.log(nodes["predicted_price"].to_numpy(dtype="float64")).tolist()],
        "lower": [base, *np.log(nodes["lower_price"].to_numpy(dtype="float64")).tolist()],
        "upper": [base, *np.log(nodes["upper_price"].to_numpy(dtype="float64")).tolist()],
    }

    steps = np.arange(0, horizons[-1] + 1, dtype="float64")
    grid = {
        name: np.exp(np.interp(steps, np.asarray(horizons, dtype="float64"), values))
        for name, values in logs.items()
    }

    if isinstance(last_date, (pd.Timestamp,)) or last_date is not None:
        try:
            index = pd.bdate_range(pd.Timestamp(last_date), periods=len(steps))
        except (TypeError, ValueError):
            index = pd.Index(steps.astype(int), name="sessions_ahead")
    else:
        index = pd.Index(steps.astype(int), name="sessions_ahead")

    path = pd.DataFrame(grid, index=index)
    path["sessions_ahead"] = steps.astype(int)
    path["is_node"] = path["sessions_ahead"].isin(horizons)
    return path


# ---------------------------------------------------------------------------
# portfolio aggregation
# ---------------------------------------------------------------------------
def portfolio_forecast(
    forecasts: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Aggregate per-holding forecasts into one portfolio expectation.

    The expected return aggregates **exactly**: a portfolio's simple return is
    the weighted mean of its holdings' simple returns, so each holding's log
    forecast is converted to simple, weighted, and converted back. (Weighting
    the log returns directly would be wrong by Jensen's inequality, and it is a
    common enough error to be worth naming.)

    The INTERVAL does not aggregate exactly, and the honest answer is a range
    rather than a number, so two bounds are returned:

    ``band_comonotonic``
        every holding at its own lower (or upper) bound simultaneously. Assumes
        perfect correlation: the widest defensible band, and the right one to
        look at for a crash.
    ``band_independent``
        per-holding half-widths added in quadrature. Assumes zero correlation,
        which no equity portfolio has: the narrowest defensible band, shown so
        the gap between the two is visible.

    The truth is between them and depends on a correlation the forecaster never
    estimated. Only holdings present in BOTH inputs contribute, and the
    ``coverage`` key reports the weight actually forecast -- an unforecastable
    holding is disclosed, never treated as flat.
    """
    usable = {
        ticker: run
        for ticker, run in (forecasts or {}).items()
        if isinstance(run, dict) and run.get("ok") and ticker in (weights or {})
    }
    blank: dict[str, Any] = {
        "ok": False,
        "reason": "No holding produced a usable forecast.",
        "expected_return": float("nan"),
        "coverage": 0.0,
        "n_holdings": 0,
        "contributions": pd.DataFrame(),
    }
    if not usable:
        return blank

    total_weight = float(sum(_as_float(weights.get(t), 0.0) for t in weights))
    if total_weight <= 0.0:
        blank["reason"] = "Portfolio weights sum to zero."
        return blank

    rows: list[dict[str, Any]] = []
    covered = 0.0
    for ticker, run in usable.items():
        weight = _as_float(weights.get(ticker), 0.0)
        if weight <= 0.0:
            continue
        share = weight / total_weight
        covered += share
        rows.append(
            {
                "ticker": ticker,
                "weight_pct": share * 100.0,
                "predicted_return": math.expm1(_as_float(run["predicted_return"], 0.0)),
                "lower_return": math.expm1(_as_float(run["lower_return"], 0.0)),
                "upper_return": math.expm1(_as_float(run["upper_return"], 0.0)),
                "contribution": share * math.expm1(_as_float(run["predicted_return"], 0.0)),
                "prob_up": _as_float(run.get("prob_up")),
                "skill": _as_float((run.get("metrics") or {}).get("skill")),
                "model": run.get("label", ""),
            }
        )

    if not rows or covered <= 0.0:
        return blank

    frame = pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)
    # Renormalise across the covered weight so the figure answers "what do the
    # holdings I COULD forecast imply", not "what if the rest are flat".
    scale = 1.0 / covered
    expected = float((frame["contribution"] * scale).sum())

    half_low = (frame["weight_pct"] / 100.0 * scale) * (
        frame["predicted_return"] - frame["lower_return"]
    )
    half_high = (frame["weight_pct"] / 100.0 * scale) * (
        frame["upper_return"] - frame["predicted_return"]
    )
    comonotonic = (
        expected - float(half_low.sum()),
        expected + float(half_high.sum()),
    )
    independent = (
        expected - float(np.sqrt(np.square(half_low).sum())),
        expected + float(np.sqrt(np.square(half_high).sum())),
    )

    return {
        "ok": True,
        "reason": "",
        "expected_return": expected,
        "band_comonotonic": comonotonic,
        "band_independent": independent,
        "coverage": float(covered),
        "n_holdings": int(len(frame)),
        "contributions": frame,
    }


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------
def forecast_insights(run: dict[str, Any]) -> list[Insight]:
    """Rule-derived observations about a completed forecast.

    Every sentence traces back to a named constant in ``src.config`` --
    :data:`~src.config.FORECAST_MIN_SKILL` and
    :data:`~src.config.FORECAST_DIRECTION_EDGE` -- and each carries the
    evidence that fired it. Nothing here is generated text about the price.
    """
    if not isinstance(run, dict) or not run.get("ok"):
        return []

    metrics = run.get("metrics") or {}
    skill = _as_float(metrics.get("skill"))
    direction = _as_float(metrics.get("directional"))
    p_value = _as_float(metrics.get("dm_p"))
    r2 = _as_float(metrics.get("r2_oos"))
    n_test = _as_int(metrics.get("n_test"), 0)
    horizon = _as_int(run.get("horizon"), 0)
    notes: list[Insight] = []

    if math.isfinite(skill) and skill <= FORECAST_MIN_SKILL:
        notes.append(
            Insight(
                level="critical",
                text=(
                    f"This model does not beat a random walk at {horizon} sessions. Assuming "
                    f"the price stays where it is would have been more accurate over the "
                    f"backtest, so the forecast below should not inform a trade."
                ),
                category=CATEGORY,
                evidence=f"skill {skill:+.3f} (threshold {FORECAST_MIN_SKILL:+.2f}), "
                f"out-of-sample R² {r2:+.3f}, n={n_test}",
            )
        )
    elif math.isfinite(skill):
        notes.append(
            Insight(
                level="positive",
                text=(
                    f"The model was {skill * 100.0:.1f}% more accurate than a random walk over "
                    f"{n_test} out-of-sample observations at {horizon} sessions."
                ),
                category=CATEGORY,
                evidence=f"skill {skill:+.3f}, MASE {_as_float(metrics.get('mase')):.3f}, "
                f"out-of-sample R² {r2:+.3f}",
            )
        )

    if math.isfinite(p_value) and math.isfinite(skill) and skill > FORECAST_MIN_SKILL:
        if p_value > 0.10:
            notes.append(
                Insight(
                    level="warning",
                    text=(
                        "That improvement is not statistically distinguishable from luck. A "
                        "Diebold-Mariano test cannot reject the hypothesis that this model and "
                        "a random walk forecast equally well."
                    ),
                    category=CATEGORY,
                    evidence=f"DM statistic {_as_float(metrics.get('dm_stat')):+.2f}, "
                    f"p = {p_value:.3f} (two-sided, HAC at {max(horizon - 1, 0)} lags)",
                )
            )
        else:
            notes.append(
                Insight(
                    level="positive",
                    text=(
                        "The improvement over a random walk is statistically significant on "
                        "this sample — which is uncommon, and worth re-testing on a different "
                        "window before you rely on it."
                    ),
                    category=CATEGORY,
                    evidence=f"DM statistic {_as_float(metrics.get('dm_stat')):+.2f}, "
                    f"p = {p_value:.3f}",
                )
            )

    if math.isfinite(direction):
        base = _as_float(metrics.get("up_rate"), 0.5)
        if direction >= FORECAST_DIRECTION_EDGE:
            notes.append(
                Insight(
                    level="info",
                    text=(
                        f"Directional accuracy was {direction * 100.0:.0f}%, against a base rate "
                        f"of {base * 100.0:.0f}% up moves. Direction can be right while the "
                        f"magnitude is wrong, and only magnitude pays."
                    ),
                    category=CATEGORY,
                    evidence=f"hit rate {direction:.3f} vs up-rate {base:.3f} "
                    f"(threshold {FORECAST_DIRECTION_EDGE:.2f})",
                )
            )
        else:
            notes.append(
                Insight(
                    level="warning",
                    text=(
                        f"Directional accuracy was {direction * 100.0:.0f}% — at or below the "
                        f"{FORECAST_DIRECTION_EDGE * 100.0:.0f}% level this app treats as "
                        f"distinguishable from a coin flip."
                    ),
                    category=CATEGORY,
                    evidence=f"hit rate {direction:.3f}, base rate {base:.3f}",
                )
            )

    if n_test and n_test < MIN_TEST_ROWS:
        notes.append(
            Insight(
                level="warning",
                text=(
                    f"Only {n_test} out-of-sample observations back every number on this page. "
                    f"That is too few to distinguish skill from noise at any horizon."
                ),
                category=CATEGORY,
                evidence=f"n_test {n_test} < {MIN_TEST_ROWS} required",
            )
        )

    width = _as_float(run.get("upper_return")) - _as_float(run.get("lower_return"))
    if math.isfinite(width):
        notes.append(
            Insight(
                level="info",
                text=(
                    f"The {_as_float(run.get('confidence'), 0.8) * 100.0:.0f}% prediction "
                    f"interval spans {width * 100.0:.1f} percentage points of return. That "
                    f"width — not the centre — is the result this page produces."
                ),
                category=CATEGORY,
                evidence=f"conformal band from {run.get('folds', 0)} walk-forward folds, "
                f"{len(run.get('residuals', ()))} residuals",
            )
        )
    return notes
