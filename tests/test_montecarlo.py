"""Forward simulation — ``src.analytics.montecarlo``.

A Monte Carlo fan chart is the most persuasive picture a portfolio tool draws
and the easiest one to draw wrongly: nothing in the output tells a reader that
the drift was applied without its Ito correction, that the seed was ignored, or
that the "simulation" of nine days of history is pure noise dressed as a
distribution. So this module is tested along four axes.

1. **The output contract, exactly.** Every generator returns
   ``(n_paths, horizon_days + 1)`` with column 0 equal to ``start_value`` on
   every path. Column ``t`` is the value after ``t`` simulated days, which is
   why the ``+ 1`` is not an off-by-one: day 0 is today, not a forecast.
2. **Hand-computed goldens.** A bootstrap of a CONSTANT return series is a
   certainty, not a distribution: resampling 1% forever gives
   ``1000 * 1.01 ** 5 = 1051.0100501`` on every path, and a GBM at zero
   volatility is ``100 * exp(3 * 0.001) = 100.30045045``. Both are worked out
   in comments beside the assertion.
3. **Identities that cannot rot.** Determinism (same seed -> identical array,
   different seed -> different array); the Ito adjustment (the mean simulated
   LOG step is ``mu - sigma^2/2``, not ``mu``, and the gap ``sigma^2/2`` is
   what separates a correct GBM from a plausible-looking wrong one); the fan
   chart's percentile columns are ordered ``p5 <= p25 <= p50 <= p75 <= p95``
   at EVERY horizon step; ``prob_loss`` is a probability; terminal CVaR is never
   better than terminal VaR; a bootstrap can never invent a day worse than the
   worst day it was shown.
4. **Documented degradation.** Too little history, a non-positive starting
   value, a degenerate shape or an unusable ``df`` return
   ``{"ok": False, "reason": ...}`` or an EMPTY ``(0, horizon + 1)`` array —
   never a plausible-looking fiction. That is the difference between a tool
   that says "I cannot answer this" and one that quietly makes something up.

Deterministic and offline: every RNG here is seeded, and the largest simulation
in the file is 800 paths over 252 days.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.montecarlo import (
    DEFAULT_STUDENT_T_DF,
    GAIN_THRESHOLD,
    METHODS,
    MIN_STUDENT_T_DF,
    TERMINAL_HIGH_PCT,
    TERMINAL_LOW_PCT,
    resolve_method,
    run_simulation,
    simulate_bootstrap,
    simulate_gbm,
    simulate_student_t,
    summarise_paths,
)
from src.config import MIN_OBSERVATIONS, SIMULATION, TRADING_DAYS

# ---------------------------------------------------------------------------
# a single, shared parameter set for the parametric generators
#
# sigma = 3% a day is deliberately large: the Ito drag over a year is
# sigma^2 * T / 2 = 0.0009 * 252 / 2 = 0.1134 in log space, comfortably bigger
# than the sampling noise of a median taken over 800 paths (~0.021), so a GBM
# that forgot the -sigma^2/2 term cannot hide inside the tolerance.
# ---------------------------------------------------------------------------
START = 1_000_000.0
MU = 0.0005
SIGMA = 0.03
HORIZON = 252
PATHS = 800
SEED = 20240101

#: (mu - sigma^2/2) — the LOG drift, written out rather than rounded.
LOG_DRIFT = MU - 0.5 * SIGMA * SIGMA  # 0.0005 - 0.00045 = 0.00005
#: sigma^2/2 — the volatility drag per day; the whole Ito adjustment.
DRAG_PER_DAY = 0.5 * SIGMA * SIGMA  # 0.00045


# ---------------------------------------------------------------------------
# local, deterministic samples (the shared fixtures cover the rest)
# ---------------------------------------------------------------------------
def constant_returns(rate: float = 0.01, n: int = 40) -> pd.Series:
    """A return series with no dispersion at all.

    Resampling it is a certainty rather than a simulation, which is exactly
    what makes the bootstrap's terminal value computable by hand.
    """
    return pd.Series([rate] * n)


@pytest.fixture
def gbm_paths() -> np.ndarray:
    """One shared 800 x 253 GBM matrix; several tests read moments off it."""
    return simulate_gbm(START, MU, SIGMA, HORIZON, PATHS, seed=SEED)


@pytest.fixture
def port_returns(portfolio_values: pd.Series) -> pd.Series:
    """Daily simple returns of the shared value series (499 observations).

    Computed positionally rather than with ``pct_change`` so no pandas
    fill-method deprecation can leak into this module.
    """
    values = portfolio_values.to_numpy(dtype=float)
    return pd.Series(values[1:] / values[:-1] - 1.0, index=portfolio_values.index[1:])


@pytest.fixture
def four_path_matrix() -> np.ndarray:
    """Start 100, terminals 88 / 100 / 112 / 132 — every statistic by hand."""
    return np.array(
        [
            [100.0, 88.0],
            [100.0, 100.0],
            [100.0, 112.0],
            [100.0, 132.0],
        ]
    )


def _generators() -> list[tuple[str, object]]:
    """``(label, callable(horizon, n_paths, seed) -> paths)`` for all three."""
    return [
        ("gbm", lambda h, n, s: simulate_gbm(START, MU, SIGMA, h, n, seed=s)),
        (
            "bootstrap",
            lambda h, n, s: simulate_bootstrap(
                pd.Series(np.linspace(-0.02, 0.02, 60)), START, h, n, seed=s
            ),
        ),
        (
            "student_t",
            lambda h, n, s: simulate_student_t(START, MU, SIGMA, 5.0, h, n, seed=s),
        ),
    ]


GENERATORS = _generators()
GENERATOR_IDS = [label for label, _ in GENERATORS]


# ---- output shape contract -----------------------------------------------
@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_every_generator_returns_n_paths_by_horizon_plus_one(label, generate):
    # The "+ 1" is day 0 — today, which is observed, not forecast. A generator
    # that returned (n_paths, horizon) would silently shift every fan chart one
    # day to the left and drop the anchor the chart is drawn against.
    paths = generate(63, 300, SEED)
    assert paths.shape == (300, 64)
    assert paths.dtype == np.float64


@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_column_zero_is_the_start_value_on_every_single_path(label, generate):
    paths = generate(30, 400, SEED)
    assert paths[:, 0].tolist() == [START] * 400


@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_paths_are_finite_and_never_negative(label, generate):
    # Log-space compounding cannot go negative and the simple-return path is
    # floored at zero; a negative portfolio value would be a modelling error.
    paths = generate(63, 300, SEED)
    assert np.isfinite(paths).all()
    assert (paths >= 0.0).all()


# ---- determinism ----------------------------------------------------------
@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_the_same_seed_reproduces_the_identical_array(label, generate):
    # Bit-for-bit, not "statistically similar": a report regenerated tomorrow
    # from the same inputs must print the same numbers as the one filed today.
    assert np.array_equal(generate(40, 300, 7), generate(40, 300, 7))


@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_a_different_seed_produces_a_different_array(label, generate):
    # The complementary half: a "seeded" generator that ignores its seed would
    # pass the reproducibility test above and be badly broken.
    assert not np.array_equal(generate(40, 300, 7), generate(40, 300, 8))


def test_run_simulation_is_reproducible_for_a_fixed_seed(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    first = run_simulation(
        portfolio_values,
        port_returns,
        method="bootstrap",
        horizon_days=63,
        n_paths=300,
        seed=11,
    )
    second = run_simulation(
        portfolio_values,
        port_returns,
        method="bootstrap",
        horizon_days=63,
        n_paths=300,
        seed=11,
    )
    assert np.array_equal(first["paths"], second["paths"])
    assert first["median_terminal"] == pytest.approx(second["median_terminal"], rel=1e-15)


# ---- GBM: the Ito drift adjustment ---------------------------------------
def test_gbm_log_steps_carry_the_ito_adjusted_drift(gbm_paths: np.ndarray):
    """The single most important assertion about the GBM generator.

    Each step is ``exp((mu - sigma^2/2) + sigma * z)``, so the mean of the
    realised LOG steps estimates ``mu - sigma^2/2``:

        0.0005 - 0.5 * 0.03**2 = 0.0005 - 0.00045 = 0.00005

    A generator that applied the arithmetic drift inside ``exp()`` without the
    correction would centre these steps on ``mu = 0.0005`` instead — 0.00045
    higher. The standard error of this mean over 800 * 252 = 201,600 draws is
    ``sigma / sqrt(201600) = 6.7e-5``, so the tolerance below (2e-4, ~3 s.e.)
    admits the correct value and rejects the uncorrected one by a wide margin.
    """
    log_steps = np.diff(np.log(gbm_paths), axis=1)
    assert log_steps.shape == (PATHS, HORIZON)
    assert float(log_steps.mean()) == pytest.approx(LOG_DRIFT, abs=2.0e-4)
    # ... and the uncorrected drift is outside that band, which is the point.
    assert abs(float(log_steps.mean()) - MU) > 2.0e-4


def test_gbm_log_step_dispersion_is_the_sigma_that_was_passed(gbm_paths: np.ndarray):
    # sigma has to keep meaning "daily volatility"; if the generator rescaled
    # it, every VaR and every fan width downstream would be wrong by that factor.
    log_steps = np.diff(np.log(gbm_paths), axis=1)
    assert float(log_steps.std(ddof=1)) == pytest.approx(SIGMA, rel=0.01)


def test_gbm_median_terminal_follows_the_drag_adjusted_drift(gbm_paths: np.ndarray):
    # median[S_T] = S_0 * exp((mu - sigma^2/2) * T)
    #             = 1e6 * exp(0.00005 * 252) = 1e6 * exp(0.0126)
    # Asserted in log space: |log(median/S_0) - 0.0126| < 0.06. The no-Ito
    # alternative sits at mu * T = 0.126, i.e. 0.1134 higher — well outside.
    median = float(np.median(gbm_paths[:, -1]))
    assert math.log(median / START) == pytest.approx(LOG_DRIFT * HORIZON, abs=0.06)
    assert DRAG_PER_DAY * HORIZON == pytest.approx(0.1134, abs=1e-12)


def test_gbm_mean_terminal_follows_the_arithmetic_drift(gbm_paths: np.ndarray):
    # E[S_T] = S_0 * exp(mu * T) = 1e6 * exp(0.126) = 1,134,282.17
    # The relative standard error of a lognormal mean over 800 paths is
    # sqrt(exp(sigma^2 T) - 1)/sqrt(n) = 1.8%, hence the 8% band.
    expected = START * math.exp(MU * HORIZON)
    assert expected == pytest.approx(1_134_282.168283025, rel=1e-12)
    assert float(np.mean(gbm_paths[:, -1])) == pytest.approx(expected, rel=0.08)


def test_the_gbm_median_sits_below_the_mean_by_the_volatility_drag(
    gbm_paths: np.ndarray,
):
    # Identity of any multiplicative process: compounding is right-skewed, so
    # the typical outcome is worse than the average outcome. This is the reason
    # a fan chart's centre line must be the median, not the mean.
    terminal = gbm_paths[:, -1]
    assert float(np.median(terminal)) < float(np.mean(terminal))
    assert float(np.median(terminal)) < START * math.exp(MU * HORIZON)


def test_gbm_with_zero_volatility_is_deterministic_compounding_of_the_drift():
    # sigma = 0 removes both the shock and the drag, leaving S_t = S_0*exp(mu*t):
    #   day 1: 100 * exp(0.001) = 100.10005002
    #   day 3: 100 * exp(0.003) = 100.30045045
    # Every path is identical because there is nothing left to randomise.
    paths = simulate_gbm(100.0, 0.001, 0.0, horizon_days=3, n_paths=5, seed=1)
    assert paths.shape == (5, 4)
    assert paths[:, -1] == pytest.approx([100.0 * math.exp(0.003)] * 5, rel=1e-12)
    assert float(paths[0, -1]) == pytest.approx(100.30045045, abs=1e-8)
    assert float(paths[0, 1]) == pytest.approx(100.0 * math.exp(0.001), rel=1e-12)


@pytest.mark.parametrize(
    ("start", "mu", "sigma", "horizon", "n_paths"),
    [
        (0.0, 0.0, 0.01, 10, 5),  # non-positive start value
        (-100.0, 0.0, 0.01, 10, 5),  # negative start value
        (float("nan"), 0.0, 0.01, 10, 5),  # unusable start value
        (100.0, 0.0, -0.01, 10, 5),  # negative volatility
        (100.0, 0.0, float("nan"), 10, 5),  # unusable volatility
        (100.0, 0.0, 0.01, 10, 0),  # no paths requested
    ],
)
def test_gbm_returns_an_empty_matrix_rather_than_raising(start, mu, sigma, horizon, n_paths):
    # Documented degradation: zero rows, but still the promised column count,
    # so a caller can inspect ``.shape[0] == 0`` instead of catching.
    paths = simulate_gbm(start, mu, sigma, horizon, n_paths)
    assert paths.shape == (0, horizon + 1)


# ---- Student-t ------------------------------------------------------------
def test_student_t_innovations_are_standardised_to_the_requested_sigma():
    # A raw t(5) has variance df/(df-2) = 5/3 = 1.667, so an unstandardised
    # implementation would inflate the daily volatility by sqrt(5/3) = 1.29 —
    # a 29% error in every simulated risk number, invisible on the chart.
    paths = simulate_student_t(START, MU, SIGMA, 5.0, HORIZON, PATHS, seed=SEED)
    log_steps = np.diff(np.log(paths), axis=1)
    assert float(log_steps.std(ddof=1)) == pytest.approx(SIGMA, rel=0.02)
    assert float(log_steps.mean()) == pytest.approx(LOG_DRIFT, abs=3.0e-4)


def test_student_t_has_fatter_tails_than_gbm_at_the_same_two_moments(
    gbm_paths: np.ndarray,
):
    # Same mu, same sigma, materially different tail: that is the entire reason
    # the method exists. Excess kurtosis of a t(5) innovation is 6/(5-4) = 6.
    t_paths = simulate_student_t(START, MU, SIGMA, 5.0, HORIZON, PATHS, seed=SEED)
    t_steps = pd.Series(np.diff(np.log(t_paths), axis=1).ravel())
    g_steps = pd.Series(np.diff(np.log(gbm_paths), axis=1).ravel())
    assert float(g_steps.kurt()) == pytest.approx(0.0, abs=0.3)  # Normal
    assert float(t_steps.kurt()) > 2.0  # fat
    assert float(t_steps.min()) < float(g_steps.min())  # worse worst day


@pytest.mark.parametrize("df", [2.0, 2.5, 1.0, 0.0, -5.0, float("nan")])
def test_student_t_refuses_a_df_with_no_usable_variance(df):
    # Var[t(df)] = df/(df-2) is infinite at df <= 2 and explodes as df -> 2, so
    # "sigma" would stop meaning volatility. MIN_STUDENT_T_DF = 2.5 is the floor
    # and it is exclusive.
    assert MIN_STUDENT_T_DF == 2.5
    assert simulate_student_t(100.0, 0.0, 0.01, df, 10, 5).shape == (0, 11)


def test_student_t_default_df_is_the_documented_equity_ballpark():
    assert DEFAULT_STUDENT_T_DF == 5.0


# ---- bootstrap ------------------------------------------------------------
def test_bootstrap_of_a_constant_return_series_is_a_hand_computable_certainty():
    # Every observation is +1%, so every draw is +1% and every path compounds
    # identically:  1000 * 1.01**5 = 1051.0100501000002
    # There is no distribution here at all, and the generator must not invent one.
    paths = simulate_bootstrap(constant_returns(0.01, 40), 1000.0, 5, 500, seed=3)
    assert paths.shape == (500, 6)
    assert np.unique(paths[:, -1]).size == 1
    assert float(paths[0, -1]) == pytest.approx(1000.0 * 1.01**5, rel=1e-12)
    assert float(paths[0, -1]) == pytest.approx(1051.0100501, abs=1e-7)
    # Intermediate columns compound too:  1000 * 1.01**2 = 1020.10
    assert float(paths[0, 2]) == pytest.approx(1020.10, abs=1e-10)


def test_bootstrap_draws_only_returns_it_was_actually_shown():
    # Resampling WITH replacement from {-2%, 0%, +3%} can produce any sequence
    # of those three and nothing else — a fourth growth factor would mean the
    # generator was smoothing or interpolating the sample.
    sample = pd.Series([-0.02, 0.0, 0.03] * 12)
    paths = simulate_bootstrap(sample, 100.0, 40, 300, seed=5)
    growth = np.round(paths[:, 1:] / paths[:, :-1], 12)
    assert set(np.unique(growth)) == {0.98, 1.0, 1.03}


def test_bootstrap_cannot_invent_a_day_worse_than_the_worst_day_observed(
    port_returns: pd.Series,
):
    # The stated limitation of a historical bootstrap, asserted as a property:
    # the empirical sample is a hard floor on the simulated single-day tail.
    worst = float(port_returns.min())
    paths = simulate_bootstrap(port_returns, START, 63, 500, seed=9)
    daily = paths[:, 1:] / paths[:, :-1] - 1.0
    assert float(daily.min()) >= worst - 1e-12
    assert float(daily.max()) <= float(port_returns.max()) + 1e-12


def test_bootstrap_reproduces_the_samples_own_mean_return(port_returns: pd.Series):
    # 500 * 63 = 31,500 i.i.d. draws from the sample, so the mean of the drawn
    # returns converges on the sample mean. This is what "the empirical
    # distribution survives exactly" means, stated as a testable identity.
    paths = simulate_bootstrap(port_returns, START, 63, 500, seed=9)
    drawn = paths[:, 1:] / paths[:, :-1] - 1.0
    tolerance = 4.0 * float(port_returns.std(ddof=1)) / math.sqrt(drawn.size)
    assert float(drawn.mean()) == pytest.approx(float(port_returns.mean()), abs=tolerance)


def test_bootstrap_absorbs_a_bankrupt_path_at_zero():
    # A -150% day is not physical for a long-only book: growth is floored at
    # zero and a path that reaches zero stays there rather than turning
    # negative and "recovering".
    paths = simulate_bootstrap(pd.Series([-1.5] * 10), 100.0, 3, 4, seed=1)
    assert paths[:, 0].tolist() == [100.0] * 4
    assert paths[:, 1:].tolist() == [[0.0, 0.0, 0.0]] * 4


def test_block_bootstrap_is_clamped_to_the_length_of_the_sample():
    # Asking for 100-day blocks from 30 observations must not index past the
    # end of the sample; the block is clamped and the shape is still honoured.
    sample = pd.Series(np.linspace(-0.01, 0.01, 30))
    paths = simulate_bootstrap(sample, 100.0, 10, 40, seed=3, block_size=100)
    assert paths.shape == (40, 11)
    assert np.isfinite(paths).all()


def test_block_bootstrap_differs_from_the_iid_bootstrap_at_the_same_seed(
    port_returns: pd.Series,
):
    # If block_size were ignored, the two would be identical and the "block"
    # method would be a lie told with a different label.
    iid = simulate_bootstrap(port_returns, START, 40, 200, seed=4, block_size=1)
    block = simulate_bootstrap(port_returns, START, 40, 200, seed=4, block_size=10)
    assert not np.array_equal(iid, block)
    assert block.shape == iid.shape


@pytest.mark.parametrize(
    "returns",
    [None, pd.Series(dtype=float), pd.Series([np.nan] * 10), pd.Series(["a", "b"])],
    ids=["none", "empty", "all-nan", "non-numeric"],
)
def test_bootstrap_of_unusable_history_is_an_empty_matrix(returns):
    assert simulate_bootstrap(returns, 100.0, 10, 5).shape == (0, 11)


# ---- summarise_paths: hand-computed goldens ------------------------------
def test_summarise_paths_is_hand_computable_on_a_four_path_matrix(
    four_path_matrix: np.ndarray,
):
    # Terminals 88, 100, 112, 132 against a start of 100:
    #   mean    = (88 + 100 + 112 + 132) / 4 = 432 / 4 = 108
    #   median  = (100 + 112) / 2            = 106
    #   worst / best                          = 88 / 132
    #   expected_return_pct = 108/100 - 1     = 8.0 %
    summary = summarise_paths(four_path_matrix, 100.0)
    assert summary["mean_terminal"] == pytest.approx(108.0, abs=1e-12)
    assert summary["median_terminal"] == pytest.approx(106.0, abs=1e-12)
    assert summary["worst_terminal"] == pytest.approx(88.0, abs=1e-12)
    assert summary["best_terminal"] == pytest.approx(132.0, abs=1e-12)
    assert summary["expected_return_pct"] == pytest.approx(8.0, abs=1e-9)
    assert summary["n_paths"] == 4
    assert summary["horizon_days"] == 1


def test_prob_loss_and_prob_gain_count_the_right_paths(four_path_matrix: np.ndarray):
    # prob_loss counts terminals STRICTLY below the start: only 88, so 1/4.
    # A path that finished exactly flat did not lose money.
    # prob_gain_10pct counts terminals at or above 100 * 1.10 = 110: 112 and
    # 132, so 2/4.
    summary = summarise_paths(four_path_matrix, 100.0)
    assert GAIN_THRESHOLD == 0.10
    assert summary["prob_loss"] == pytest.approx(0.25, abs=1e-12)
    assert summary["prob_gain_10pct"] == pytest.approx(0.50, abs=1e-12)


def test_terminal_var_and_cvar_are_hand_computable(four_path_matrix: np.ndarray):
    # Sorted terminals [88, 100, 112, 132]; numpy's linear 5th percentile lands
    # at index 0.05 * 3 = 0.15, i.e. 88 + 0.15 * (100 - 88) = 89.8.
    #   var_terminal  = 89.8/100 - 1 = -0.102
    #   the breaching tail is {88} alone, so
    #   cvar_terminal = 88/100 - 1   = -0.12
    # Both are HORIZON losses as negative decimals, measured against the start.
    summary = summarise_paths(four_path_matrix, 100.0, confidence=0.95)
    assert summary["confidence"] == pytest.approx(0.95, abs=1e-12)
    assert summary["var_terminal"] == pytest.approx(-0.102, abs=1e-12)
    assert summary["cvar_terminal"] == pytest.approx(-0.12, abs=1e-12)
    assert summary["cvar_terminal"] <= summary["var_terminal"] < 0.0


def test_terminal_percentile_keys_are_hand_computable(four_path_matrix: np.ndarray):
    # p5  -> index 0.05 * 3 = 0.15 -> 88 + 0.15 * 12 = 89.8
    # p95 -> index 0.95 * 3 = 2.85 -> 112 + 0.85 * 20 = 129.0
    summary = summarise_paths(four_path_matrix, 100.0)
    assert (TERMINAL_LOW_PCT, TERMINAL_HIGH_PCT) == (5.0, 95.0)
    assert summary["p5_terminal"] == pytest.approx(89.8, abs=1e-12)
    assert summary["p95_terminal"] == pytest.approx(129.0, abs=1e-12)
    assert summary["p5_terminal"] <= summary["median_terminal"] <= summary["p95_terminal"]


def test_the_fan_chart_is_hand_computable_on_the_four_path_matrix(
    four_path_matrix: np.ndarray,
):
    # Day 1 cross-section of [88, 100, 112, 132], numpy linear interpolation:
    #   p25 -> index 0.75  -> 88  + 0.75 * 12 =  97.0
    #   p50 -> index 1.50  -> 100 + 0.50 * 12 = 106.0
    #   p75 -> index 2.25  -> 112 + 0.25 * 20 = 117.0
    fan = summarise_paths(four_path_matrix, 100.0)["percentile_paths"]
    assert list(fan.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert fan.index.name == "day"
    assert fan.loc[1].to_numpy() == pytest.approx([89.8, 97.0, 106.0, 117.0, 129.0], abs=1e-12)


# ---- summarise_paths: identities -----------------------------------------
def test_the_fan_chart_starts_every_percentile_at_the_start_value(
    gbm_paths: np.ndarray,
):
    # Day 0 has no dispersion — every path is standing on today's value — so
    # all five percentile lines must meet at exactly the start value.
    fan = summarise_paths(gbm_paths, START)["percentile_paths"]
    assert len(fan) == HORIZON + 1
    assert fan.loc[0].to_numpy() == pytest.approx([START] * 5, rel=1e-12)


@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_percentile_columns_are_ordered_at_every_horizon_step(label, generate):
    # p5 <= p25 <= p50 <= p75 <= p95 on EVERY row, not just the last one. A
    # crossing anywhere would mean the fan chart's bands were computed against
    # the wrong axis, and the shaded regions would be nonsense.
    fan = summarise_paths(generate(126, 500, SEED), START)["percentile_paths"]
    grid = fan.to_numpy()
    assert grid.shape == (127, 5)
    assert np.all(np.diff(grid, axis=1) >= -1e-9)


@pytest.mark.parametrize(("label", "generate"), GENERATORS, ids=GENERATOR_IDS)
def test_prob_loss_is_a_probability(label, generate):
    summary = summarise_paths(generate(126, 500, SEED), START)
    assert 0.0 <= summary["prob_loss"] <= 1.0
    assert 0.0 <= summary["prob_gain_10pct"] <= 1.0


def test_expected_return_pct_restates_the_mean_terminal(gbm_paths: np.ndarray):
    # Identity: expected_return_pct == (mean_terminal / start - 1) * 100.
    # The *100 is the whole risk of this key: reporting 0.08 where the UI
    # prints "%" understates the forecast a hundredfold.
    summary = summarise_paths(gbm_paths, START)
    assert summary["expected_return_pct"] == pytest.approx(
        (summary["mean_terminal"] / START - 1.0) * 100.0, rel=1e-12
    )


def test_summarise_paths_does_not_mutate_the_matrix_it_was_given(
    gbm_paths: np.ndarray,
):
    before = gbm_paths.copy()
    summarise_paths(gbm_paths, START)
    assert np.array_equal(gbm_paths, before)


def test_custom_percentiles_are_labelled_from_their_own_values(gbm_paths: np.ndarray):
    fan = summarise_paths(gbm_paths, START, percentiles=(10, 50, 90))["percentile_paths"]
    assert list(fan.columns) == ["p10", "p50", "p90"]


def test_percentiles_outside_the_open_unit_range_fall_back_to_the_default(
    gbm_paths: np.ndarray,
):
    # 0 and 100 are not percentiles of a fan chart (they are the min and max of
    # a finite sample and carry no confidence at all), so they are discarded;
    # with nothing usable left the configured default is restored.
    fan = summarise_paths(gbm_paths, START, percentiles=(0, 100, -5, 250))["percentile_paths"]
    assert list(fan.columns) == [f"p{p}" for p in SIMULATION.percentiles]


@pytest.mark.parametrize(
    "matrix",
    [np.empty((0, 6)), np.zeros((5, 1)), np.zeros(5), "not a matrix"],
    ids=["no-paths", "single-column", "one-dimensional", "not-numeric"],
)
def test_summarise_paths_of_an_unusable_matrix_is_nan_throughout(matrix):
    summary = summarise_paths(matrix, 100.0)
    assert summary["n_paths"] == 0
    assert summary["percentile_paths"].empty
    assert summary["terminal"].size == 0
    for key in ("mean_terminal", "median_terminal", "prob_loss", "var_terminal"):
        assert math.isnan(summary[key])


def test_summarise_paths_without_a_usable_start_value_still_reports_the_levels():
    # The terminal LEVELS are knowable without a base; the RATIOS are not, and
    # the function reports exactly that split rather than dividing by zero.
    summary = summarise_paths(np.array([[0.0, 90.0], [0.0, 110.0]]), 0.0)
    assert summary["mean_terminal"] == pytest.approx(100.0, abs=1e-12)
    assert math.isnan(summary["prob_loss"])
    assert math.isnan(summary["expected_return_pct"])


# ---- method resolution ----------------------------------------------------
@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("gbm", "gbm"),
        ("GBM", "gbm"),
        ("lognormal", "gbm"),
        ("parametric", "gbm"),
        ("bootstrap", "bootstrap"),
        ("historical", "bootstrap"),
        ("empirical", "bootstrap"),
        ("block", "block_bootstrap"),
        ("Moving Block", "block_bootstrap"),
        ("moving-block-bootstrap", "block_bootstrap"),
        ("student_t", "student_t"),
        ("Student-T", "student_t"),
        ("t", "student_t"),
        ("fat_tail", "student_t"),
    ],
)
def test_resolve_method_maps_ui_spellings_onto_canonical_names(spelling, expected):
    assert resolve_method(spelling) in METHODS
    assert resolve_method(spelling) == expected


@pytest.mark.parametrize("spelling", [None, "", "nonsense", "monte carlo", 42])
def test_an_unknown_method_falls_back_to_the_assumption_free_bootstrap(spelling):
    # Falling back to GBM would silently impose a Normal tail on a user who
    # asked for something else; the bootstrap assumes least.
    assert resolve_method(spelling) == "bootstrap"


# ---- run_simulation: end to end ------------------------------------------
def test_run_simulation_starts_every_path_at_the_last_portfolio_value(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    result = run_simulation(
        portfolio_values,
        port_returns,
        method="bootstrap",
        horizon_days=63,
        n_paths=400,
        seed=11,
    )
    assert result["ok"] is True
    assert result["start_value"] == pytest.approx(float(portfolio_values.iloc[-1]), rel=1e-12)
    assert result["paths"].shape == (400, 64)
    assert result["paths"][:, 0].tolist() == [result["start_value"]] * 400


def test_run_simulation_reports_the_shape_and_horizon_it_actually_ran(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    result = run_simulation(portfolio_values, port_returns, horizon_days=63, n_paths=400, seed=11)
    # 63 trading days is a quarter: 63 / 252 = 0.25 years exactly.
    assert result["horizon_days"] == 63
    assert result["horizon_years"] == pytest.approx(63.0 / TRADING_DAYS, rel=1e-12)
    assert result["horizon_years"] == pytest.approx(0.25, abs=1e-12)
    assert result["n_paths"] == 400
    assert result["n_obs"] == len(port_returns)


def test_run_simulation_estimates_mu_and_sigma_from_the_sample(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    # Regression guard on the two parameters every parametric method rests on:
    # the arithmetic mean and the BESSEL-CORRECTED sample std of daily returns.
    result = run_simulation(
        portfolio_values,
        port_returns,
        method="gbm",
        horizon_days=63,
        n_paths=300,
        seed=11,
    )
    assert result["mu_daily"] == pytest.approx(float(port_returns.mean()), rel=1e-12)
    assert result["sigma_daily"] == pytest.approx(float(port_returns.std(ddof=1)), rel=1e-12)


def test_run_simulation_labels_the_bootstrap_by_the_block_size_it_used(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    # The block size, not the requested label, is what actually distinguishes
    # the two bootstraps, so it has the last word on how the run is reported.
    plain = run_simulation(
        portfolio_values,
        port_returns,
        method="bootstrap",
        horizon_days=40,
        n_paths=300,
        seed=2,
    )
    blocked = run_simulation(
        portfolio_values,
        port_returns,
        method="bootstrap",
        block_size=10,
        horizon_days=40,
        n_paths=300,
        seed=2,
    )
    demoted = run_simulation(
        portfolio_values,
        port_returns,
        method="block_bootstrap",
        block_size=1,
        horizon_days=40,
        n_paths=300,
        seed=2,
    )
    assert (plain["method"], plain["block_size"]) == ("bootstrap", 1)
    assert (blocked["method"], blocked["block_size"]) == ("block_bootstrap", 10)
    assert (demoted["method"], demoted["block_size"]) == ("bootstrap", 1)


def test_run_simulation_derives_returns_from_the_value_series_when_given_none(
    portfolio_values: pd.Series,
):
    # n prices imply n-1 returns; the simulation page can therefore be driven
    # from a value series alone.
    result = run_simulation(
        portfolio_values, None, method="gbm", horizon_days=21, n_paths=300, seed=1
    )
    assert result["ok"] is True
    assert result["n_obs"] == len(portfolio_values) - 1


def test_run_simulation_accepts_a_bare_starting_number(port_returns: pd.Series):
    result = run_simulation(
        250_000.0, port_returns, method="gbm", horizon_days=21, n_paths=300, seed=1
    )
    assert result["ok"] is True
    assert result["start_value"] == pytest.approx(250_000.0, rel=1e-12)


def test_run_simulation_of_a_flat_history_warns_that_nothing_is_being_explored(
    flat_series: pd.Series,
):
    # A zero-variance sample produces one repeated path. The run is not an
    # error, but presenting it as a distribution would be, so the note says so
    # in the first word.
    result = run_simulation(flat_series, None, method="gbm", horizon_days=10, n_paths=50, seed=1)
    assert result["ok"] is True
    assert result["sigma_daily"] == pytest.approx(0.0, abs=1e-15)
    assert result["note"].startswith("WARNING")
    assert np.unique(result["terminal"]).size == 1
    assert result["prob_loss"] == pytest.approx(0.0, abs=1e-12)


# ---- run_simulation: honest refusal --------------------------------------
def test_run_simulation_refuses_a_history_shorter_than_the_minimum():
    # MIN_OBSERVATIONS = 30. Below it a "distribution" is noise with a fan
    # chart drawn around it, so the answer is ok=False and a reason a user can
    # act on — not a plausible-looking picture.
    assert MIN_OBSERVATIONS == 30
    short = pd.Series(np.linspace(100.0, 110.0, 20), index=pd.bdate_range("2023-01-02", periods=20))
    result = run_simulation(short, None, method="gbm")
    assert result["ok"] is False
    assert result["n_obs"] == 19
    assert "19 usable daily returns" in result["reason"]
    assert str(MIN_OBSERVATIONS) in result["reason"]
    assert "paths" not in result


def test_run_simulation_refuses_a_single_observation(single_value_series: pd.Series):
    result = run_simulation(single_value_series, None)
    assert result["ok"] is False
    assert result["n_obs"] == 0


def test_run_simulation_refuses_an_all_nan_history(nan_series: pd.Series):
    result = run_simulation(nan_series, None)
    assert result["ok"] is False
    assert math.isnan(result["start_value"])
    assert "starting portfolio value" in result["reason"]


@pytest.mark.parametrize("values", [None, 0.0, -1000.0, float("nan")])
def test_run_simulation_refuses_a_non_positive_starting_value(values, port_returns: pd.Series):
    result = run_simulation(values, port_returns)
    assert result["ok"] is False
    assert "starting portfolio value" in result["reason"]


@pytest.mark.parametrize(("horizon", "paths_n"), [(0, 300), (-5, 300), (63, 0), (63, -1)])
def test_run_simulation_refuses_a_degenerate_simulation_size(
    horizon, paths_n, portfolio_values: pd.Series, port_returns: pd.Series
):
    result = run_simulation(portfolio_values, port_returns, horizon_days=horizon, n_paths=paths_n)
    assert result["ok"] is False
    assert "Degenerate simulation size" in result["reason"]


def test_run_simulation_refuses_an_unusable_student_t_df(
    portfolio_values: pd.Series, port_returns: pd.Series
):
    result = run_simulation(
        portfolio_values,
        port_returns,
        method="student_t",
        df=2.0,
        horizon_days=21,
        n_paths=300,
        seed=1,
    )
    assert result["ok"] is False
    assert "degrees of freedom" in result["reason"]


def test_a_failed_run_still_carries_the_context_a_ui_needs(portfolio_values: pd.Series):
    # The failure dict is not bare: it names the method, the shape asked for
    # and the seed, so the page can render the request that could not be met.
    result = run_simulation(portfolio_values, None, method="t", horizon_days=0)
    assert result["ok"] is False
    assert result["method"] == "student_t"
    assert result["horizon_days"] == 0
    assert result["seed"] == SIMULATION.seed


@pytest.mark.parametrize("method", METHODS)
def test_every_method_produces_a_complete_summary(
    method, portfolio_values: pd.Series, port_returns: pd.Series
):
    result = run_simulation(
        portfolio_values,
        port_returns,
        method=method,
        horizon_days=63,
        n_paths=400,
        seed=6,
    )
    assert result["ok"] is True
    assert result["paths"].shape == (400, 64)
    assert 0.0 <= result["prob_loss"] <= 1.0
    assert result["worst_terminal"] <= result["median_terminal"] <= result["best_terminal"]
    assert result["cvar_terminal"] <= result["var_terminal"]
    fan = result["percentile_paths"]
    assert np.all(np.diff(fan.to_numpy(), axis=1) >= -1e-9)
    assert isinstance(result["note"], str) and len(result["note"]) > 100
