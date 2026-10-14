"""The forecaster: does it learn, does it admit when it cannot, and is it honest?

The suite is built around one pair of tests that matter more than all the rest:

* :func:`test_the_pipeline_finds_a_signal_that_is_genuinely_there` feeds the
  models a series with a real, documented mean-reverting signal and asserts
  they find it. Without this, every "no skill" result elsewhere could be a
  broken pipeline rather than an efficient market.
* :func:`test_the_pipeline_reports_no_edge_on_a_pure_random_walk` feeds them a
  series with nothing in it and asserts they do NOT report an edge. Without
  this, the machinery could be leaking and nobody would notice, because a
  leaking forecaster looks excellent.

Together they pin the property the whole feature exists to provide: the number
on the screen tracks whether there is anything to find.

Everything is seeded and offline. No network, no market hours, no model
persisted between tests.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv, make_prices
from src.analytics import forecast as fc
from src.config import FORECAST, ForecastConfig

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: Enough rows for the 252-session lookbacks plus a five-fold walk-forward.
DAYS = 1600

#: Models that need scikit-learn are skipped wholesale if it is absent, rather
#: than failing a suite for a dependency the module is explicit about.
requires_sklearn = pytest.mark.skipif(
    not fc.SKLEARN_AVAILABLE, reason="scikit-learn is not installed"
)


# ---------------------------------------------------------------------------
# fixtures: one series with a signal, one without
# ---------------------------------------------------------------------------
def _series_from_returns(returns: np.ndarray, start: str = "2016-01-03") -> pd.DataFrame:
    """OHLCV bars whose closes compound the supplied log returns exactly."""
    index = pd.bdate_range(start, periods=len(returns))
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.006,
            "Low": close * 0.994,
            "Close": close,
            "Volume": np.full(len(returns), 1.0e6),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def predictable() -> pd.DataFrame:
    """AR(1) log returns with rho = -0.35: strong, real, one-day mean reversion.

    Deliberately far stronger than any real market. The point is not realism —
    it is that a pipeline which cannot recover a signal this obvious is broken,
    and every "no skill" verdict it produces elsewhere would be meaningless.
    """
    rng = np.random.default_rng(42)
    shocks = rng.normal(0.0, 0.012, DAYS)
    returns = np.zeros(DAYS)
    for step in range(1, DAYS):
        returns[step] = -0.35 * returns[step - 1] + shocks[step]
    return _series_from_returns(returns)


@pytest.fixture(scope="module")
def unpredictable() -> pd.DataFrame:
    """I.i.d. log returns: a pure random walk with drift, and nothing to find."""
    rng = np.random.default_rng(7)
    return _series_from_returns(rng.normal(0.0004, 0.012, DAYS))


@pytest.fixture(scope="module")
def run(predictable: pd.DataFrame) -> dict:
    return fc.run_forecast(predictable, 1, "ridge")


# ---------------------------------------------------------------------------
# THE two tests
# ---------------------------------------------------------------------------
@requires_sklearn
@pytest.mark.parametrize("model", ["ridge", "elastic_net", "random_forest"])
def test_the_pipeline_finds_a_signal_that_is_genuinely_there(
    predictable: pd.DataFrame, model: str
) -> None:
    """A real AR(1) signal must show up as positive out-of-sample skill.

    This is the test that makes every negative result elsewhere trustworthy:
    it proves the walk-forward, the purge and the scoring do not destroy a
    signal that exists, so when they report none, there is none.
    """
    result = fc.run_forecast(predictable, 1, model, with_importance=False)
    assert result["ok"], result["reason"]

    metrics = result["metrics"]
    assert metrics["skill"] > 0.02, f"{model} failed to find an obvious signal"
    assert metrics["r2_oos"] > 0.02
    assert metrics["directional"] > 0.55
    assert metrics["dm_p"] < 0.05, "a signal this strong must be statistically detectable"
    assert metrics["dm_stat"] > 0, "a positive statistic means the model beat the random walk"


def test_the_pipeline_reports_no_edge_on_a_pure_random_walk(
    unpredictable: pd.DataFrame,
) -> None:
    """With nothing to find, the forecaster must not report finding something.

    A leaking pipeline reports skill here. The bound is deliberately loose —
    a fitted model can beat a random walk by luck on one sample — but a leak
    produces skill far above it.
    """
    result = fc.run_forecast(unpredictable, 21, "ridge", with_importance=False)
    assert result["ok"], result["reason"]
    assert result["metrics"]["skill"] < 0.05, "suspiciously good on data with no signal"
    assert result["metrics"]["r2_oos"] < 0.05


# ---------------------------------------------------------------------------
# fold construction: the purge is the whole point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 5, 21, 63])
def test_every_fold_purges_exactly_the_overlapping_labels(horizon: int) -> None:
    """The gap between a training block and its test block is ``h - 1 + extra``.

    A label at horizon ``h`` spans ``h`` future sessions, so without the gap the
    last ``h - 1`` training labels are built from the very returns the test
    block is about to be scored on. For a 63-session horizon that is three
    months of the answer sitting inside the training set.
    """
    folds = fc.make_folds(2000, horizon, FORECAST)
    assert folds, "2,000 rows must support a walk-forward at every offered horizon"

    expected_purge = horizon - 1 + FORECAST.purge_extra
    for fold in folds:
        assert fold.purge == expected_purge
        assert fold.test_start - fold.train_end == expected_purge
        assert fold.train_end <= fold.test_start
        assert fold.n_train >= FORECAST.min_initial_train
        assert fold.n_test >= 1


def test_folds_are_sequential_contiguous_and_expanding() -> None:
    """No shuffling, no overlap, no gap in the tested rows, training only grows."""
    folds = fc.make_folds(2000, 5, FORECAST)
    assert len(folds) == FORECAST.n_splits

    for earlier, later in pairwise(folds):
        assert later.test_start == earlier.test_end, "test blocks must tile without a gap"
        assert later.train_end > earlier.train_end, "training must expand, not slide"
        assert later.train_start == earlier.train_start == 0

    assert folds[-1].test_end == 2000, "the last block must absorb the remainder"
    tested = folds[-1].test_end - folds[0].test_start
    assert tested == pytest.approx(2000 * FORECAST.test_fraction, rel=0.02)


def test_a_test_row_is_never_inside_any_training_block() -> None:
    """Stated as a set identity, so it cannot be satisfied by an off-by-one."""
    for fold in fc.make_folds(1500, 21, FORECAST):
        train = set(range(fold.train_start, fold.train_end))
        test = set(range(fold.test_start, fold.test_end))
        assert not (train & test)
        assert max(train) < min(test)


def test_a_history_too_short_to_validate_produces_no_folds_at_all() -> None:
    assert fc.make_folds(50, 21, FORECAST) == ()
    assert fc.make_folds(0, 1, FORECAST) == ()
    # 63-session horizon on 260 rows: the purge alone eats the training block.
    assert fc.make_folds(260, 63, FORECAST) == ()


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def test_the_random_walk_predicts_exactly_zero() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    predicted = fc.RandomWalk().fit(x, pd.Series([0.1, -0.2, 0.3])).predict(x)
    assert np.array_equal(predicted, np.zeros(3))


def test_the_drift_baseline_predicts_the_training_mean() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.1, 0.2, 0.3, 0.4])
    model = fc.DriftBaseline().fit(x, y)
    assert model.drift_ == pytest.approx(0.25)
    assert np.allclose(model.predict(x), 0.25)


def test_the_random_walk_scores_exactly_zero_skill(unpredictable: pd.DataFrame) -> None:
    """An identity: the benchmark cannot beat or lose to itself.

    ``skill = 1 - MAE_model / MAE_random-walk`` is 0 for the random walk by
    construction, and ``r2_oos`` is 0 for the same reason. If either drifts,
    the denominator has stopped being what the docstring says it is.
    """
    result = fc.run_forecast(unpredictable, 5, "random_walk", with_importance=False)
    assert result["ok"], result["reason"]
    assert result["metrics"]["skill"] == pytest.approx(0.0, abs=1e-12)
    assert result["metrics"]["mase"] == pytest.approx(1.0, abs=1e-12)
    assert result["metrics"]["r2_oos"] == pytest.approx(0.0, abs=1e-12)
    assert result["predicted_return"] == 0.0
    assert result["predicted_price"] == pytest.approx(result["last_price"])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_a_perfect_forecast_scores_perfectly() -> None:
    truth = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03, 0.01, 0.02])
    scores = fc.score_predictions(truth, truth, horizon=1)
    assert scores["mae"] == pytest.approx(0.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["skill"] == pytest.approx(1.0)
    assert scores["mase"] == pytest.approx(0.0)
    assert scores["r2_oos"] == pytest.approx(1.0)
    assert scores["directional"] == pytest.approx(1.0)
    assert scores["n_test"] == 10


def test_a_constant_zero_forecast_calls_no_direction_rather_than_calling_them_all_wrong() -> None:
    """``sign(0)`` matches nothing, so scoring it as 0% would read as "always wrong".

    The random walk does not answer the direction question at all, and the
    scorecard must say so with a NaN rather than a damning zero.
    """
    truth = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03, 0.01, 0.02])
    scores = fc.score_predictions(truth, np.zeros_like(truth), horizon=1)
    assert np.isnan(scores["directional"])
    assert scores["direction_calls"] == 0
    assert scores["up_rate"] == pytest.approx(0.6)
    assert scores["skill"] == pytest.approx(0.0)


def test_an_inverted_forecast_scores_negative_skill_and_zero_direction() -> None:
    truth = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03])
    scores = fc.score_predictions(truth, -truth, horizon=1)
    assert scores["skill"] < 0.0
    assert scores["mase"] == pytest.approx(2.0)
    assert scores["r2_oos"] == pytest.approx(-3.0)
    assert scores["directional"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "truth,predicted",
    [([], []), ([0.01], [0.01, 0.02]), ([np.nan] * 5, [np.nan] * 5)],
    ids=["empty", "length-mismatch", "all-nan"],
)
def test_unusable_predictions_score_as_missing_rather_than_raising(truth, predicted) -> None:
    scores = fc.score_predictions(truth, predicted, horizon=1)
    assert np.isnan(scores["mae"])
    assert np.isnan(scores["skill"])
    assert scores["n_test"] == 0


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------
def test_identical_forecasts_are_not_distinguishable() -> None:
    errors = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03, 0.01, 0.02])
    statistic, p_value = fc.diebold_mariano(errors, errors, horizon=1)
    assert np.isnan(statistic) and np.isnan(p_value)


def test_a_clearly_better_forecast_yields_a_positive_significant_statistic() -> None:
    rng = np.random.default_rng(11)
    benchmark = rng.normal(0.0, 0.05, 300)
    model = benchmark * 0.25  # four times smaller errors, every single time
    statistic, p_value = fc.diebold_mariano(model, benchmark, horizon=1)
    assert statistic > 0.0
    assert p_value < 0.01


def test_the_statistic_is_antisymmetric_in_its_two_arguments() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 0.02, 200), rng.normal(0, 0.03, 200)
    forward, _ = fc.diebold_mariano(a, b, horizon=5)
    reverse, _ = fc.diebold_mariano(b, a, horizon=5)
    assert forward == pytest.approx(-reverse, rel=1e-12)


def test_too_few_points_yield_no_statistic() -> None:
    assert fc.diebold_mariano([0.1, 0.2], [0.3, 0.4], 1) == (
        pytest.approx(float("nan"), nan_ok=True),
        pytest.approx(float("nan"), nan_ok=True),
    )


# ---------------------------------------------------------------------------
# conformal intervals
# ---------------------------------------------------------------------------
def test_the_conformal_band_covers_its_nominal_share_of_the_residuals() -> None:
    """The defining property: an 80% band contains 80% of the sample it came from."""
    rng = np.random.default_rng(5)
    residuals = rng.normal(0.0, 0.03, 5000)
    low, high = fc.conformal_bounds(residuals, 0.20)
    covered = np.mean((residuals >= low) & (residuals <= high))
    assert covered == pytest.approx(0.80, abs=0.01)


def test_the_band_is_signed_so_a_skewed_error_law_stays_skewed() -> None:
    """A symmetric ``+/- q`` band would understate the downside by construction."""
    rng = np.random.default_rng(9)
    # Left-skewed, like the errors of a model fitted to equity returns.
    residuals = -np.abs(rng.gumbel(0.0, 0.02, 4000)) + 0.01
    low, high = fc.conformal_bounds(residuals, 0.20)
    assert abs(low) > abs(high), "the lower tail is longer and the band must show it"


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.40])
def test_a_tighter_miscoverage_gives_a_wider_band(alpha: float) -> None:
    rng = np.random.default_rng(13)
    residuals = rng.normal(0.0, 0.02, 3000)
    low, high = fc.conformal_bounds(residuals, alpha)
    wider_low, wider_high = fc.conformal_bounds(residuals, alpha / 2.0)
    assert wider_high - wider_low > high - low


def test_too_few_residuals_produce_no_band_rather_than_a_narrow_one() -> None:
    assert all(np.isnan(v) for v in fc.conformal_bounds([0.1, 0.2, 0.3], 0.2))
    assert all(np.isnan(v) for v in fc.conformal_bounds([], 0.2))
    assert all(np.isnan(v) for v in fc.conformal_bounds(np.zeros(50), 1.5))


# ---------------------------------------------------------------------------
# the end-to-end run
# ---------------------------------------------------------------------------
@requires_sklearn
def test_a_run_reports_its_forecast_its_interval_and_its_evidence(run: dict) -> None:
    assert run["ok"], run["reason"]
    assert run["lower_price"] < run["predicted_price"] < run["upper_price"]
    assert 0.0 <= run["prob_up"] <= 1.0
    assert run["n_train"] > FORECAST.min_train_rows
    assert run["folds"] == FORECAST.n_splits
    assert run["purge"] == 0 + FORECAST.purge_extra  # horizon 1
    assert len(run["features"]) == run["n_features"] > 30
    assert len(run["residuals"]) == run["metrics"]["n_test"]
    assert not run["per_fold"].empty
    assert not run["importance"].empty
    assert run["note"]


@requires_sklearn
def test_the_price_forecast_is_the_log_return_compounded_onto_todays_close(run: dict) -> None:
    """``S_hat = S_0 * exp(r_hat)`` — the identity the KPI card claims."""
    assert run["predicted_price"] == pytest.approx(
        run["last_price"] * np.exp(run["predicted_return"]), rel=1e-12
    )
    assert run["lower_price"] == pytest.approx(
        run["last_price"] * np.exp(run["lower_return"]), rel=1e-12
    )
    assert run["upper_price"] == pytest.approx(
        run["last_price"] * np.exp(run["upper_return"]), rel=1e-12
    )


@requires_sklearn
@pytest.mark.parametrize("model", ["ridge", "random_forest", "gradient_boost", "ensemble"])
def test_a_run_is_reproducible_for_a_given_seed(predictable: pd.DataFrame, model: str) -> None:
    """Same data, same settings, same number — including the sampling models.

    A forecast nobody can reproduce is a forecast nobody can audit, and the
    tree ensembles are where it is easiest to get this wrong.

    The tolerance is not slack: the random forest predicts by accumulating
    across threads, so the floating-point summation ORDER varies between runs
    and two forecasts agree to ~1e-15 relative rather than bit-exactly. That is
    ten orders of magnitude below anything this application prints, and the
    trade is documented at the estimator that makes it. Every non-threaded
    model here is exact, and this test would catch a genuine seeding bug — an
    unseeded forest disagrees in the third decimal, not the fifteenth.
    """
    first = fc.run_forecast(predictable, 5, model, with_importance=False)
    second = fc.run_forecast(predictable, 5, model, with_importance=False)
    assert first["predicted_return"] == pytest.approx(second["predicted_return"], rel=1e-12)
    assert first["metrics"]["skill"] == pytest.approx(second["metrics"]["skill"], rel=1e-12)
    assert first["lower_price"] == pytest.approx(second["lower_price"], rel=1e-12)


@requires_sklearn
def test_the_interval_widens_with_the_horizon(predictable: pd.DataFrame) -> None:
    """Uncertainty compounds. A one-day band must be tighter than a quarter's."""
    short = fc.run_forecast(predictable, 1, "ridge", with_importance=False)
    long = fc.run_forecast(predictable, 63, "ridge", with_importance=False)
    short_width = short["upper_return"] - short["lower_return"]
    long_width = long["upper_return"] - long["lower_return"]
    assert long_width > short_width


def test_a_close_only_series_still_produces_a_forecast() -> None:
    """The portfolio value series has no bar and no volume. It must still work."""
    values = make_prices(("AAA",), days=DAYS, start="2016-01-04").iloc[:, 0]
    result = fc.run_forecast(values, 21, "random_walk", with_importance=False)
    assert result["ok"], result["reason"]
    assert "volume" not in result["families"]
    assert result["n_features"] > 20


@pytest.mark.parametrize(
    "bad",
    [None, pd.DataFrame(), pd.Series(dtype="float64")],
    ids=["none", "empty-frame", "empty-series"],
)
def test_unusable_input_is_refused_with_a_reason(bad: object) -> None:
    result = fc.run_forecast(bad, 21, "ridge")
    assert result["ok"] is False
    assert result["reason"]
    assert result["n_train"] == 0


def test_a_short_history_is_refused_and_the_reason_names_the_shortfall() -> None:
    result = fc.run_forecast(make_ohlcv("AAA", days=400), 21, "ridge")
    assert result["ok"] is False
    assert str(FORECAST.min_train_rows) in result["reason"]


def test_the_callers_frame_is_never_modified(predictable: pd.DataFrame) -> None:
    before = predictable.copy()
    fc.run_forecast(predictable, 5, "random_walk", with_importance=False)
    fc.forecast_path(predictable, (1, 5), "random_walk")
    pd.testing.assert_frame_equal(predictable, before)


# ---------------------------------------------------------------------------
# the multi-horizon path
# ---------------------------------------------------------------------------
@requires_sklearn
def test_the_path_starts_at_todays_close_and_marks_every_fitted_horizon(
    predictable: pd.DataFrame,
) -> None:
    horizons = (1, 5, 21)
    result = fc.forecast_path(predictable, horizons, "ridge")
    assert result["ok"], result["reason"]

    nodes, path = result["nodes"], result["path"]
    assert list(nodes["horizon"]) == sorted(horizons)
    assert len(path) == max(horizons) + 1

    # Day 0 is today, exactly, on all three lines: the cone opens from a point.
    assert path["point"].iloc[0] == pytest.approx(result["last_price"], rel=1e-12)
    assert path["lower"].iloc[0] == pytest.approx(result["last_price"], rel=1e-12)
    assert path["upper"].iloc[0] == pytest.approx(result["last_price"], rel=1e-12)

    marked = path[path["is_node"]]
    assert set(marked["sessions_ahead"]) == {0, *horizons}
    for row in nodes.itertuples():
        drawn = path.loc[path["sessions_ahead"] == row.horizon, "point"].iloc[0]
        assert drawn == pytest.approx(row.predicted_price, rel=1e-9), (
            "a marked node must be the fitted value, not the interpolation"
        )


@requires_sklearn
def test_interpolation_between_nodes_happens_in_log_space(
    predictable: pd.DataFrame,
) -> None:
    """Halfway between +2% and +6% is +4% COMPOUNDED, not the mean of two prices."""
    result = fc.forecast_path(predictable, (10, 20), "ridge")
    assert result["ok"]
    path = result["path"]
    start = np.log(float(path.loc[path["sessions_ahead"] == 10, "point"].iloc[0]))
    end = np.log(float(path.loc[path["sessions_ahead"] == 20, "point"].iloc[0]))
    middle = float(path.loc[path["sessions_ahead"] == 15, "point"].iloc[0])
    assert np.log(middle) == pytest.approx((start + end) / 2.0, rel=1e-9)


def test_a_path_with_no_fittable_horizon_is_refused_with_a_reason() -> None:
    result = fc.forecast_path(make_ohlcv("AAA", days=300), (21, 63), "ridge")
    assert result["ok"] is False
    assert result["reason"]
    assert result["nodes"].empty


def test_a_path_with_no_positive_horizon_is_refused(predictable: pd.DataFrame) -> None:
    assert fc.forecast_path(predictable, (), "ridge")["ok"] is False
    assert fc.forecast_path(predictable, (0, -5), "ridge")["ok"] is False


# ---------------------------------------------------------------------------
# the leaderboard
# ---------------------------------------------------------------------------
@requires_sklearn
def test_the_leaderboard_scores_every_model_on_identical_held_out_rows(
    predictable: pd.DataFrame,
) -> None:
    models = ("random_walk", "drift", "ridge", "random_forest")
    board = fc.model_leaderboard(predictable, 5, models=models)

    assert set(board["model"]) == set(models)
    assert board["n_test"].nunique() == 1, "a comparison on different rows is not a comparison"
    assert board["folds"].nunique() == 1
    assert board["skill"].is_monotonic_decreasing, "the board must be sorted by skill"
    assert board.loc[board["model"] == "random_walk", "skill"].iloc[0] == pytest.approx(0.0)


@requires_sklearn
def test_the_baselines_are_on_the_board_but_never_selected_as_best(
    predictable: pd.DataFrame,
) -> None:
    """They are the yardstick. Choosing one as "the model" would leave nothing to plot.

    The UI is responsible for saying loudly when the yardstick won anyway, and
    the ``beats_naive`` column is what it reads.
    """
    board = fc.model_leaderboard(predictable, 5, models=("random_walk", "drift", "ridge"))
    assert "random_walk" in set(board["model"])
    assert fc.best_model(board) == "ridge"
    assert set(board.columns) >= {"beats_naive", "dm_p", "directional", "reason"}


def test_best_model_falls_back_rather_than_raising_on_an_empty_board() -> None:
    assert fc.best_model(pd.DataFrame()) == "ridge"
    assert fc.best_model(pd.DataFrame(), fallback="random_forest") == "random_forest"


def test_a_history_too_short_to_score_reports_every_model_as_unfitted() -> None:
    """Not an empty board: a board of named failures, each with its reason.

    An empty table on screen reads as a rendering bug. A table saying "ridge:
    only 53 labelled rows, 250 required" is the same fact, told usefully — and
    it is what the page renders into its "could not be fitted" list.
    """
    board = fc.model_leaderboard(make_ohlcv("AAA", days=200), 21)
    assert not board.empty
    assert set(board["model"]) == set(fc.DEFAULT_LEADERBOARD)
    assert not board["ok"].any()
    assert board["reason"].map(bool).all()
    assert board["skill"].isna().all(), "a model that never fitted has no skill to report"
    assert not board["beats_naive"].any()


def test_a_history_with_no_usable_feature_row_returns_an_empty_board() -> None:
    """Nothing to score at all — not even a named failure per model."""
    board = fc.model_leaderboard(pd.DataFrame(), 21)
    assert board.empty
    assert list(board.columns), "the column shape survives even with no rows"


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_every_model_states_what_it_assumes_and_where_it_breaks() -> None:
    """A model presented without both is a model being sold rather than explained."""
    for key, spec in fc.MODELS.items():
        assert spec.key == key
        assert spec.label and spec.description
        assert len(spec.assumption) > 40, key
        assert len(spec.weakness) > 40, key
        assert spec.family in {"baseline", "linear", "tree", "blend"}


def test_the_default_leaderboard_only_names_registered_models() -> None:
    assert set(fc.DEFAULT_LEADERBOARD) <= set(fc.MODELS)
    assert "random_walk" in fc.DEFAULT_LEADERBOARD, "the benchmark must always be raced"


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Random Forest", "random_forest"),
        ("random-forest", "random_forest"),
        ("RF", "random_forest"),
        ("gbm", "gradient_boost"),
        ("naive", "random_walk"),
        ("historical nonsense", "ridge"),
        (None, "ridge"),
    ],
)
def test_a_model_spelling_resolves_to_a_registry_key(spelling, expected: str) -> None:
    assert fc.resolve_model(spelling) == expected


def test_the_baselines_stay_available_without_scikit_learn() -> None:
    available = fc.available_models()
    assert "random_walk" in available
    assert "drift" in available


@requires_sklearn
def test_the_ensemble_weights_are_a_normalised_distribution(
    predictable: pd.DataFrame,
) -> None:
    result = fc.run_forecast(predictable, 5, "ensemble", with_importance=False)
    assert result["ok"], result["reason"]
    weights = result["ensemble_weights"]
    assert set(weights) == set(fc._ENSEMBLE_MEMBERS)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(0.0 <= w <= 1.0 for w in weights.values())


# ---------------------------------------------------------------------------
# portfolio aggregation
# ---------------------------------------------------------------------------
def _fake_run(point: float, low: float, high: float, skill: float = 0.01) -> dict:
    """A minimal successful run dict, for testing aggregation arithmetic alone."""
    return {
        "ok": True,
        "predicted_return": np.log1p(point),
        "lower_return": np.log1p(low),
        "upper_return": np.log1p(high),
        "prob_up": 0.55,
        "metrics": {"skill": skill},
        "label": "Test model",
    }


def test_the_weighted_expected_return_aggregates_exactly() -> None:
    """A portfolio's simple return IS the weighted mean of its holdings'.

    Weighting the LOG returns directly would be wrong by Jensen's inequality,
    and it is a common enough error to be worth pinning.
    """
    forecasts = {
        "AAA": _fake_run(0.10, -0.05, 0.25),
        "BBB": _fake_run(-0.04, -0.20, 0.12),
    }
    aggregate = fc.portfolio_forecast(forecasts, {"AAA": 60.0, "BBB": 40.0})
    assert aggregate["ok"]
    assert aggregate["expected_return"] == pytest.approx(0.6 * 0.10 + 0.4 * -0.04, rel=1e-9)
    assert aggregate["coverage"] == pytest.approx(1.0)
    assert aggregate["n_holdings"] == 2


def test_the_comonotonic_band_always_contains_the_independent_one() -> None:
    """Perfect correlation is the widest defensible band; zero is the narrowest."""
    forecasts = {t: _fake_run(0.05, -0.10, 0.20) for t in ("AAA", "BBB", "CCC")}
    aggregate = fc.portfolio_forecast(forecasts, dict.fromkeys(forecasts, 100.0 / 3))
    low_co, high_co = aggregate["band_comonotonic"]
    low_in, high_in = aggregate["band_independent"]
    assert low_co < low_in < aggregate["expected_return"] < high_in < high_co


def test_an_unforecastable_holding_is_disclosed_not_treated_as_flat() -> None:
    """Dropping a holding from the denominator would flatter the coverage figure."""
    forecasts = {
        "AAA": _fake_run(0.10, -0.05, 0.25),
        "BBB": {"ok": False, "reason": "too little history"},
    }
    aggregate = fc.portfolio_forecast(forecasts, {"AAA": 70.0, "BBB": 30.0})
    assert aggregate["ok"]
    assert aggregate["coverage"] == pytest.approx(0.70)
    assert aggregate["n_holdings"] == 1
    # Renormalised across covered weight: the answer is "what the holdings I
    # could forecast imply", not "what if the other 30% is flat".
    assert aggregate["expected_return"] == pytest.approx(0.10, rel=1e-9)


@pytest.mark.parametrize(
    "forecasts,weights",
    [({}, {"AAA": 100.0}), ({"AAA": _fake_run(0.1, 0.0, 0.2)}, {}), ({}, {})],
    ids=["no-forecasts", "no-weights", "neither"],
)
def test_aggregation_refuses_rather_than_inventing_a_portfolio_number(
    forecasts: dict, weights: dict
) -> None:
    aggregate = fc.portfolio_forecast(forecasts, weights)
    assert aggregate["ok"] is False
    assert aggregate["reason"]
    assert np.isnan(aggregate["expected_return"])


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------
def test_a_model_that_loses_to_a_random_walk_gets_a_critical_insight(
    unpredictable: pd.DataFrame,
) -> None:
    """The finding a forecasting page most needs to state, stated at the top level."""
    result = fc.run_forecast(unpredictable, 21, "random_walk", with_importance=False)
    # The random walk scores exactly zero skill, which is at the threshold.
    assert result["metrics"]["skill"] <= 0.0
    notes = fc.forecast_insights(result)
    assert any(note.level == "critical" for note in notes)
    assert all(note.category == fc.CATEGORY for note in notes)
    assert all(note.evidence for note in notes), "every insight must cite its numbers"


@requires_sklearn
def test_insights_cite_the_threshold_that_fired_them(run: dict) -> None:
    notes = fc.forecast_insights(run)
    assert notes
    for note in notes:
        assert note.level in {"positive", "info", "warning", "critical"}
        assert note.text and note.evidence


def test_a_failed_run_generates_no_narrative_at_all() -> None:
    assert fc.forecast_insights({"ok": False, "reason": "nope"}) == []
    assert fc.forecast_insights({}) == []
    assert fc.forecast_insights(None) == []


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_the_config_is_internally_consistent() -> None:
    assert 0.0 < FORECAST.conformal_alpha < 1.0
    assert FORECAST.confidence == pytest.approx(1.0 - FORECAST.conformal_alpha)
    assert FORECAST.default_horizon in FORECAST.horizons
    assert all(h > 0 for h in FORECAST.horizons)
    assert FORECAST.min_initial_train <= FORECAST.min_train_rows
    assert 0.0 < FORECAST.test_fraction < 1.0
    assert FORECAST.n_splits >= 2


def test_a_custom_config_changes_the_fold_geometry() -> None:
    """The knobs are real: nothing below reads a module-level default instead."""
    tight = ForecastConfig(n_splits=3, test_fraction=0.15, purge_extra=5)
    folds = fc.make_folds(2000, 10, tight)
    assert len(folds) == 3
    assert all(fold.purge == 10 - 1 + 5 for fold in folds)
    assert folds[-1].test_end - folds[0].test_start == pytest.approx(300, abs=2)
