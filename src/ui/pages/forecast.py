"""Price forecast -- a machine-learning model, and the evidence about whether it works.

This page exists to answer a question users ask constantly and most tools
answer dishonestly: *what will this be worth in a month?* The answer here is a
number, an interval, and -- given equal billing with both -- the measured
out-of-sample record of the model that produced them.

The layout encodes that priority. The verdict comes before the chart. The
leaderboard includes the two baselines that the fitted models frequently lose
to. The cone's width is the model's own held-out residual spread, so a useless
model draws a useless-looking cone without anyone having to read a table.

Everything computational lives in :mod:`src.analytics.forecast` and
:mod:`src.analytics.features`; every figure in :mod:`src.charts.forecast`.
Like the Security page, this one is permitted to fetch its own OHLCV, because
a forecaster needs far more history than the risk window the sidebar selected
-- and it says so on screen rather than silently using a different window from
the rest of the application.

Runs are cached in session state against the exact control tuple that produced
them, so switching a widget that does not affect the model never silently
refits it under the same numbers.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import features as feat
from src.analytics import forecast as fc
from src.config import (
    FORECAST,
    FORECAST_DIRECTION_EDGE,
    FORECAST_DISCLAIMER,
    FORECAST_MIN_SKILL,
    PERIOD_LABELS,
    TRADING_DAYS,
)
from src.ui import components as ui
from src.ui import data_access, guards, session
from src.ui import theme as ui_theme
from src.ui.formatters import (
    currency_symbol,
    delta_tone,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
)

# ---------------------------------------------------------------------------
# control vocabulary
# ---------------------------------------------------------------------------
#: Training windows offered, with the yfinance period string behind each.
#: Deliberately longer than the sidebar's risk window: a walk-forward needs
#: enough rows to hold out 30% of them and still leave a trainable block, and
#: the longest feature lookback alone eats a year off the front.
_TRAINING_WINDOWS: tuple[tuple[str, str], ...] = (
    ("5 years", "5y"),
    ("10 years", "10y"),
    ("Maximum available", "max"),
)
_DEFAULT_WINDOW = "10y"

#: yfinance period string -> the label the user picked it by.
_WINDOW_NAMES: dict[str, str] = {period: label for label, period in _TRAINING_WINDOWS}

#: Horizon label per forecast length, in the order the slider presents them.
_HORIZON_LABELS: dict[int, str] = {
    1: "1 session (next close)",
    5: "1 week (5 sessions)",
    10: "2 weeks (10 sessions)",
    21: "1 month (21 sessions)",
    63: "1 quarter (63 sessions)",
}
_HORIZON_WORDS: dict[int, str] = {
    1: "tomorrow",
    5: "a week",
    10: "a fortnight",
    21: "a month",
    63: "a quarter",
}

#: Sentinel offered alongside the real model keys.
_AUTO = "__auto__"

#: Session keys holding the memoised runs. Each maps a control signature to a
#: result, so an unrelated rerun reuses the fit instead of redrawing a
#: different random forest under the same headline number.
_RUN_CACHE = "forecast_run_cache"
_PATH_CACHE = "forecast_path_cache"
_BOARD_CACHE = "forecast_board_cache"
_BOTTOM_UP_CACHE = "forecast_bottom_up_cache"

#: Memo entries kept per cache. Each carries a path matrix and a residual
#: vector, so an unbounded cache would grow with every widget interaction.
_MAX_MEMO_ENTRIES: int = 6

#: Holdings forecast individually before the bottom-up view refuses to run.
#: Each one is a full walk-forward; twenty of them is a minute of compute.
_MAX_BOTTOM_UP: int = 12

#: Rows shown in the per-fold stability table before it is truncated.
_MAX_FOLD_ROWS: int = 12


class _Controls(NamedTuple):
    """Everything that determines a run -- and therefore the cache key."""

    ticker: str
    window: str
    horizon: int
    model: str


# ---------------------------------------------------------------------------
# small display helpers (no analytics happen in this module)
# ---------------------------------------------------------------------------
def _num(value: Any) -> float:
    """``value`` as a finite float, or NaN when it is not a real number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _ok(value: Any) -> bool:
    return math.isfinite(_num(value))


def _int(value: Any, default: int) -> int:
    number = _num(value)
    return int(number) if math.isfinite(number) else default


def _horizon_words(days: int) -> str:
    return _HORIZON_WORDS.get(days, f"{days} trading sessions")


def _memo(cache_key: str, signature: Any, build: Any) -> Any:
    """Return ``build()`` for this signature, from session state when unchanged.

    A plain dict in session state rather than ``st.cache_data``: the cached
    values carry numpy arrays and fitted-model metadata that Streamlit's hasher
    would have to serialise on every rerun, and the whole point is to avoid
    touching them at all when nothing changed.
    """
    store = st.session_state.get(cache_key)
    if not isinstance(store, dict):
        store = {}
    key = repr(signature)
    if key in store:
        return store[key]
    value = build()
    # Bounded, insertion-ordered: the oldest entry leaves when a new one
    # arrives. Python dicts preserve insertion order, so this is a FIFO.
    while len(store) >= _MAX_MEMO_ENTRIES:
        store.pop(next(iter(store)))
    store[key] = value
    st.session_state[cache_key] = store
    return value


def _data_signature(ohlcv: pd.DataFrame) -> tuple[str, str, str]:
    """Fingerprint of the price history a run depends on.

    Strings deliberately: a NaN close must compare equal to itself or the memo
    would miss on every rerun.
    """
    try:
        return (str(len(ohlcv)), str(ohlcv.index[-1]), f"{float(ohlcv['Close'].iloc[-1]):.6f}")
    except Exception:  # noqa: BLE001 - a cache key must never raise
        return ("0", "none", "nan")


def _currency_for(result: Any, ticker: str) -> str:
    """The currency this instrument's PRICES are quoted in.

    Not the reporting currency: the forecast is a statement about the traded
    price, so converting it into the portfolio's base currency would fold an FX
    forecast into an equity one without saying so.
    """
    profile = (result.profiles or {}).get(ticker) if result is not None else None
    currency = getattr(profile, "currency", None)
    if currency:
        return str(currency)
    quote = (result.quotes or {}).get(ticker) if result is not None else None
    if quote is not None and getattr(quote, "currency", None):
        return str(quote.currency)
    return getattr(result.config, "base_currency", "INR") if result is not None else "INR"


def _skill_tone(skill: Any) -> str:
    value = _num(skill)
    if not math.isfinite(value):
        return "muted"
    return "gain" if value > FORECAST_MIN_SKILL else "loss"


# ---------------------------------------------------------------------------
# 1. controls
# ---------------------------------------------------------------------------
def _ticker_options(result: Any) -> list[str]:
    """Priced holdings, most valuable first, so the default is worth looking at."""
    positions = getattr(result, "positions", None)
    if positions is None or len(positions) == 0 or "ticker" not in positions.columns:
        return []
    frame = positions
    if "current_value" in frame.columns:
        frame = frame.sort_values("current_value", ascending=False)
    return [str(t) for t in frame["ticker"]]


def _controls(result: Any) -> _Controls | None:
    """Render the four settings that define a run and return them as a tuple."""
    ui_theme.section("WHAT TO FORECAST", "🎛")

    options = _ticker_options(result)
    col_ticker, col_window, col_horizon, col_model = st.columns([1.5, 1.1, 1.5, 1.5], gap="medium")

    with col_ticker:
        if options:
            default = session.selected_ticker(options[0])
            index = options.index(default) if default in options else 0
            ticker = st.selectbox(
                "Instrument",
                options=options,
                index=index,
                key="forecast_ticker",
                help="Any holding in the analysed portfolio. Research a symbol you do not "
                "own on the Security deep dive page first.",
            )
            session.select_ticker(ticker)
        else:
            ticker = ""
            st.selectbox("Instrument", options=["—"], disabled=True, key="forecast_ticker_empty")

    with col_window:
        window_label = st.selectbox(
            "Training history",
            options=[label for label, _ in _TRAINING_WINDOWS],
            index=1,
            key="forecast_window",
            help="Independent of the sidebar's risk window on purpose. Risk statistics "
            "describe the period you chose; a model needs every row it can get, and "
            "the longest feature lookback alone costs a year off the front.",
        )
        window = dict(_TRAINING_WINDOWS).get(window_label, _DEFAULT_WINDOW)

    with col_horizon:
        horizon = int(
            st.select_slider(
                "Forecast horizon",
                options=list(FORECAST.horizons),
                value=FORECAST.default_horizon,
                format_func=lambda d: _HORIZON_LABELS.get(d, f"{d} sessions"),
                key="forecast_horizon",
                help="Trading sessions ahead. Each horizon is a separately fitted and "
                "separately validated model — nothing is extrapolated from a shorter one.",
            )
        )

    with col_model:
        keys = list(fc.available_models())
        labels = {key: fc.MODELS[key].label for key in keys}
        labels[_AUTO] = "Auto — best on the leaderboard"
        choices = [_AUTO, *keys]
        model = st.selectbox(
            "Model",
            options=choices,
            index=choices.index("ridge") if "ridge" in choices else 0,
            format_func=lambda key: labels.get(key, key),
            key="forecast_model",
            help="Auto races every model over identical held-out blocks and picks the "
            "winner, which costs noticeably more compute. The two baselines are on the "
            "menu because they are frequently the honest answer.",
        )

    if not ticker:
        return None
    return _Controls(ticker, window, horizon, model)


# ---------------------------------------------------------------------------
# 2. running the model
# ---------------------------------------------------------------------------
def _load_history(ticker: str, window: str) -> pd.DataFrame:
    """OHLCV bars for the training window, or an empty frame."""
    with ui.loading(f"Loading {ticker} daily bars…"):
        try:
            return data_access.load_ohlcv(ticker, window)
        except Exception as exc:  # noqa: BLE001 - a data failure is a page state
            st.caption(f"The provider request failed: {exc}")
            return pd.DataFrame()


def _resolve_model(ohlcv: pd.DataFrame, controls: _Controls) -> tuple[str, pd.DataFrame]:
    """The model key to fit, plus the leaderboard when Auto selected it."""
    if controls.model != _AUTO:
        return controls.model, pd.DataFrame()
    board = _leaderboard(ohlcv, controls)
    return fc.best_model(board), board


def _leaderboard(ohlcv: pd.DataFrame, controls: _Controls) -> pd.DataFrame:
    signature = (controls.ticker, controls.window, controls.horizon, _data_signature(ohlcv))
    return _memo(
        _BOARD_CACHE,
        signature,
        lambda: _run_leaderboard(ohlcv, controls.horizon),
    )


def _run_leaderboard(ohlcv: pd.DataFrame, horizon: int) -> pd.DataFrame:
    with ui.loading(f"Racing {len(fc.DEFAULT_LEADERBOARD)} models over identical held-out blocks…"):
        return fc.model_leaderboard(ohlcv, horizon)


def _forecast(ohlcv: pd.DataFrame, controls: _Controls, model: str) -> dict[str, Any]:
    signature = (controls.ticker, controls.window, controls.horizon, model, _data_signature(ohlcv))
    return _memo(
        _RUN_CACHE,
        signature,
        lambda: _run_forecast(ohlcv, controls.horizon, model),
    )


def _run_forecast(ohlcv: pd.DataFrame, horizon: int, model: str) -> dict[str, Any]:
    with ui.loading(f"Fitting {fc.MODELS[model].label.lower()} and validating it out of sample…"):
        return fc.run_forecast(ohlcv, horizon, model)


def _path(ohlcv: pd.DataFrame, controls: _Controls, model: str) -> dict[str, Any]:
    signature = (controls.ticker, controls.window, model, _data_signature(ohlcv))
    return _memo(
        _PATH_CACHE,
        signature,
        lambda: _run_path(ohlcv, model),
    )


def _run_path(ohlcv: pd.DataFrame, model: str) -> dict[str, Any]:
    with ui.loading(f"Fitting one model per horizon ({len(FORECAST.horizons)} of them)…"):
        return fc.forecast_path(ohlcv, FORECAST.horizons, model)


# ---------------------------------------------------------------------------
# 3. the verdict
# ---------------------------------------------------------------------------
def _verdict(run: dict[str, Any]) -> None:
    """The page's headline conclusion, stated before any chart is drawn."""
    metrics = run.get("metrics") or {}
    skill = _num(metrics.get("skill"))
    p_value = _num(metrics.get("dm_p"))
    direction = _num(metrics.get("directional"))
    n_test = _int(metrics.get("n_test"), 0)
    horizon = _int(run.get("horizon"), 0)
    label = str(run.get("label", "This model"))

    evidence = (
        f"skill {skill:+.4f} · MASE {_num(metrics.get('mase')):.4f} · "
        f"out-of-sample R² {_num(metrics.get('r2_oos')):+.4f} · "
        f"Diebold-Mariano p {p_value:.3f} · {n_test} held-out observations · "
        f"{_int(run.get('folds'), 0)} purged folds"
    )

    if not math.isfinite(skill):
        ui.verdict(
            "No skill estimate could be produced",
            f"{label} fitted, but the walk-forward did not return a usable error estimate. "
            f"Treat the forecast below as illustration only.",
            tone="caution",
            evidence=evidence,
        )
        return

    if skill <= FORECAST_MIN_SKILL:
        ui.verdict(
            f"{label} does not beat a random walk at {horizon} sessions",
            f"Over {n_test} observations it never saw during training, this model was "
            f"{abs(skill) * 100.0:.1f}% LESS accurate than simply assuming the price does not "
            f"move. The forecast below is what the model says; the evidence says not to act "
            f"on it. This is the ordinary result for daily equity prices, and a page that "
            f"hid it would be the problem.",
            tone="fail",
            evidence=evidence,
        )
        return

    significant = math.isfinite(p_value) and p_value <= 0.10
    tone = "pass" if significant else "caution"
    tail = (
        "A Diebold-Mariano test says a difference this large is unlikely to be luck "
        f"(p = {p_value:.3f}) — which is uncommon enough to be worth re-testing on a "
        "different window before you rely on it."
        if significant
        else "But the difference is not statistically distinguishable from luck, so treat "
        "it as an encouraging sample rather than a demonstrated edge."
    )
    ui.verdict(
        f"{label} beat a random walk by {skill * 100.0:.1f}% at {horizon} sessions",
        f"Measured on {n_test} observations it never saw during training, with "
        f"{direction * 100.0:.0f}% directional accuracy. {tail}"
        if math.isfinite(direction)
        else f"Measured on {n_test} observations it never saw during training. {tail}",
        tone=tone,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# 4. the forecast itself
# ---------------------------------------------------------------------------
def _forecast_kpis(run: dict[str, Any], currency: str) -> None:
    ui_theme.section("THE FORECAST", "🎯")

    last = _num(run.get("last_price"))
    point = _num(run.get("predicted_price"))
    lower = _num(run.get("lower_price"))
    upper = _num(run.get("upper_price"))
    ret = (
        math.expm1(_num(run.get("predicted_return")))
        if _ok(run.get("predicted_return"))
        else float("nan")
    )
    confidence = _num(run.get("confidence"))
    horizon = _int(run.get("horizon"), 0)
    words = _horizon_words(horizon)
    width = (upper - lower) if (_ok(lower) and _ok(upper)) else float("nan")

    cards: list[dict[str, Any]] = [
        {
            "label": f"Point forecast in {words}",
            "value": fmt_money(point, currency, 2),
            "tone": delta_tone(ret),
            "delta": f"{fmt_ratio_pct(ret, signed=True)} from {fmt_money(last, currency, 2)}",
            "delta_tone": delta_tone(ret),
            "footnote": f"{horizon} trading sessions ahead",
            "help_text": (
                "The model predicts a LOG return, and this is today's price times its "
                "exponential — which is the MEDIAN of the implied distribution, not its mean. "
                "The mean sits above it, because compounding is multiplicative."
            ),
        },
        {
            "label": f"{confidence * 100:.0f}% interval — low",
            "value": fmt_money(lower, currency, 2),
            "tone": "neutral",
            "delta": fmt_ratio_pct(
                (lower / last - 1.0) if (_ok(lower) and _ok(last) and last > 0) else float("nan"),
                signed=True,
            ),
            "delta_tone": "muted",
            "footnote": f"{(1 - confidence) / 2 * 100:.0f}% of outcomes fell below this",
            "help_text": (
                "Not a worst case. It is the lower conformal quantile of the model's own "
                "out-of-sample errors, carried onto the point forecast."
            ),
        },
        {
            "label": f"{confidence * 100:.0f}% interval — high",
            "value": fmt_money(upper, currency, 2),
            "tone": "neutral",
            "delta": fmt_ratio_pct(
                (upper / last - 1.0) if (_ok(upper) and _ok(last) and last > 0) else float("nan"),
                signed=True,
            ),
            "delta_tone": "muted",
            "footnote": f"Band spans {fmt_money(width, currency, 2)}",
            "help_text": (
                "The WIDTH of this band is the result this page produces. A model with no "
                "skill produces a very wide one, which is the honest output."
            ),
        },
        {
            "label": "Measured skill",
            "value": fmt_pct(_num((run.get("metrics") or {}).get("skill")) * 100.0, 2, signed=True),
            "tone": _skill_tone((run.get("metrics") or {}).get("skill")),
            "delta": "vs a random walk, out of sample",
            "delta_tone": "muted",
            "footnote": f"{_int((run.get('metrics') or {}).get('n_test'), 0)} held-out observations",
            "help_text": (
                "1 − (model error / random-walk error). Zero means the model is exactly as "
                "accurate as predicting no change; negative means it is worse."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    probability = _num(run.get("prob_up"))
    bias = _num(run.get("bias"))
    col_meter, col_note = st.columns([1.0, 2.0], gap="medium")
    with col_meter:
        ui.meter(
            "Probability the price is higher",
            probability,
            reference=0.5,
            tone="gain" if probability > 0.5 else "loss",
            footnote="Marker is a coin flip. Read the distance from it, not the number.",
        )
    with col_note:
        st.caption(
            "That probability is the share of the model's own out-of-sample error "
            "distribution that lands above today's price once it is shifted onto the point "
            "forecast — no Normal assumption anywhere. "
            + (
                f"The model's median held-out error was {fmt_ratio_pct(math.expm1(bias), signed=True)}, "
                f"which is its measured bias and the reason the interval is not centred on the "
                f"point forecast."
                if _ok(bias)
                else ""
            )
        )


def _cone(history: pd.DataFrame, path: dict[str, Any], run: dict[str, Any], currency: str) -> None:
    ui_theme.section("THE CONE", "🌪")

    if not path.get("ok"):
        ui.empty_state(
            "No multi-horizon cone",
            path.get("reason") or "No horizon could be fitted from this history.",
            icon="🚧",
            hint="The single-horizon forecast above is unaffected — the cone needs a model "
            "per horizon, and the longest one needs the most rows.",
        )
        return

    ui.chart(
        ch.forecast_cone_chart(
            history,
            path.get("path"),
            path.get("nodes"),
            currency_symbol(currency),
            _num(run.get("confidence")),
            height=500,
        )
    )

    nodes = path.get("nodes")
    if isinstance(nodes, pd.DataFrame) and not nodes.empty:
        wins = int(nodes["skill"].gt(FORECAST_MIN_SKILL).sum())
        st.caption(
            f"Every marker is its own fitted, purged, separately validated model — "
            f"{len(nodes)} of them — so the cone is not one model extrapolated forward and a "
            f"kink between two markers is a genuine disagreement between two fits. "
            f"{wins} of {len(nodes)} horizons beat a random walk out of sample. Widen the "
            f"band by looking at it, not by reading the centre: the centre is a drift "
            f"estimate, and drift is the least reliable quantity in the model."
        )


# ---------------------------------------------------------------------------
# 5. validation
# ---------------------------------------------------------------------------
def _validation(run: dict[str, Any]) -> None:
    ui_theme.section("IS IT ANY GOOD?", "🔍")

    metrics = run.get("metrics") or {}
    direction = _num(metrics.get("directional"))
    base_rate = _num(metrics.get("up_rate"))
    r2 = _num(metrics.get("r2_oos"))
    p_value = _num(metrics.get("dm_p"))
    ic = _num(metrics.get("ic"))

    cards: list[dict[str, Any]] = [
        {
            "label": "Directional accuracy",
            "value": fmt_ratio_pct(direction, 1),
            "tone": "gain" if direction >= FORECAST_DIRECTION_EDGE else "warn",
            "delta": f"base rate {fmt_ratio_pct(base_rate, 1)} up",
            "delta_tone": "muted",
            "footnote": f"On {_int(metrics.get('direction_calls'), 0)} rows where it called one",
            "help_text": (
                "Compare against the base rate, not 50%. An instrument that rose 58% of the "
                "time makes a 55% hit rate WORSE than always saying up."
            ),
        },
        {
            "label": "Out-of-sample R²",
            "value": fmt_num(r2, 4),
            "tone": "gain" if r2 > 0 else "loss",
            "delta": "vs predicting zero",
            "delta_tone": "muted",
            "footnote": "Campbell–Thompson convention",
            "help_text": (
                "1 − SSE/SS of a zero forecast. Negative is the normal result and means the "
                "model explains less than nothing. Positive values in the low single-digit "
                "percentages are considered strong in the published literature."
            ),
        },
        {
            "label": "Rank correlation (IC)",
            "value": fmt_num(ic, 4),
            "tone": delta_tone(ic),
            "delta": "Spearman, prediction vs outcome",
            "delta_tone": "muted",
            "footnote": "Does it at least ORDER outcomes correctly?",
            "help_text": (
                "A model can rank outcomes usefully while getting every magnitude wrong. "
                "That is still tradeable in a cross-sectional strategy, and not on one name."
            ),
        },
        {
            "label": "Diebold–Mariano p",
            "value": fmt_num(p_value, 3),
            "tone": "gain" if p_value <= 0.10 else "warn",
            "delta": f"statistic {fmt_num(metrics.get('dm_stat'), 2)}",
            "delta_tone": "muted",
            "footnote": f"HAC at {max(_int(run.get('horizon'), 1) - 1, 0)} lags",
            "help_text": (
                "Tests equal squared-error loss against the random walk, two-sided. A "
                "positive statistic favours the model; a small p says the difference is "
                "unlikely to be chance."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    tab_time, tab_scatter, tab_error, tab_folds = st.tabs(
        ["Through time", "Predicted vs realised", "Error distribution", "Fold by fold"]
    )
    with tab_time:
        ui.chart(ch.prediction_timeline_chart(run.get("predictions")))
    with tab_scatter:
        ui.chart(ch.backtest_scatter(run.get("predictions")))
    with tab_error:
        ui.chart(
            ch.residual_distribution_chart(run.get("residuals"), 1.0 - _num(run.get("confidence")))
        )
    with tab_folds:
        _fold_table(run)


def _fold_table(run: dict[str, Any]) -> None:
    """Per-block stability -- the table that exposes a model that stopped working."""
    folds = run.get("per_fold")
    if not isinstance(folds, pd.DataFrame) or folds.empty:
        st.caption("No per-fold detail was recorded for this run.")
        return

    frame = folds.head(_MAX_FOLD_ROWS).copy()
    display = pd.DataFrame(
        {
            "Fold": frame["fold"].astype(int),
            "Trained on": frame["train_rows"].map(lambda n: f"{int(n):,} rows"),
            "Train ends": frame["train_end"].map(lambda d: pd.Timestamp(d).strftime("%d %b %Y")),
            "Tested": frame.apply(
                lambda r: (
                    f"{pd.Timestamp(r['test_start']):%d %b %Y} → "
                    f"{pd.Timestamp(r['test_end']):%d %b %Y}"
                ),
                axis=1,
            ),
            "Rows": frame["test_rows"].astype(int),
            "Skill": frame["skill"].map(lambda v: fmt_pct(_num(v) * 100.0, 2, signed=True)),
            "Direction": frame["directional"].map(lambda v: fmt_ratio_pct(v, 0)),
            "R² (OOS)": frame["r2_oos"].map(lambda v: fmt_num(v, 3)),
        }
    )
    ui.dataframe(display)

    positive = int(frame["skill"].gt(FORECAST_MIN_SKILL).sum())
    purge = _int(run.get("purge"), 0)
    st.caption(
        f"{positive} of {len(frame)} blocks beat a random walk. An average skill number "
        f"cannot show a model that worked until a regime change and then stopped; this table "
        f"can. Note the gap between each 'train ends' date and the test window that follows "
        f"it: that is the **{purge}-row purge**, dropped because a "
        f"{_int(run.get('horizon'), 0)}-session label spans "
        f"{_int(run.get('horizon'), 0)} future sessions and the last few training labels "
        f"would otherwise overlap the very returns being scored."
    )


# ---------------------------------------------------------------------------
# 6. leaderboard
# ---------------------------------------------------------------------------
def _leaderboard_section(ohlcv: pd.DataFrame, controls: _Controls, board: pd.DataFrame) -> None:
    ui_theme.section("MODEL LEADERBOARD", "🏁")

    if board.empty:
        st.caption(
            "Every model is fitted on the same rows, validated on the same held-out blocks, "
            "and scored against the same random walk — which is the only way the comparison "
            "means anything. It refits seven models, so it runs on request."
        )
        if not st.button("Race every model", type="primary", key="forecast_race"):
            return
        board = _leaderboard(ohlcv, controls)

    if board.empty:
        ui.empty_state(
            "No model could be scored",
            "This history cannot support a single purged walk-forward fold at this horizon.",
            icon="🏳",
        )
        return

    ui.chart(ch.model_leaderboard_chart(board))

    display = pd.DataFrame(
        {
            "Model": board["label"],
            "Family": board["family"].str.replace("_", " ").str.title(),
            "Skill": board["skill"].map(lambda v: fmt_pct(_num(v) * 100.0, 2, signed=True)),
            "MASE": board["mase"].map(lambda v: fmt_num(v, 4)),
            "Direction": board["directional"].map(lambda v: fmt_ratio_pct(v, 1)),
            "R² (OOS)": board["r2_oos"].map(lambda v: fmt_num(v, 4)),
            "IC": board["ic"].map(lambda v: fmt_num(v, 3)),
            "DM p": board["dm_p"].map(lambda v: fmt_num(v, 3)),
            "Beats random walk": board["beats_naive"].map(lambda flag: "Yes" if flag else "No"),
        }
    )
    ui.dataframe(display)

    winners = board[board["beats_naive"] & ~board["family"].eq("baseline")]
    n_ok = int(board["ok"].sum())
    if winners.empty:
        st.caption(
            "None of the fitted models beat a random walk on this instrument at this "
            "horizon. That is the finding, and it is the most common one: daily equity "
            "prices are close to a martingale, which is why the baselines are on the "
            "leaderboard rather than hidden behind it."
        )
    else:
        st.caption(
            f"{len(winners)} of {n_ok} fitted models beat a random walk out of sample. "
            f"Check the Diebold–Mariano column before treating that as an edge: on "
            f"{_int(board['n_test'].max(), 0)} test points, a skill of a fraction of a "
            f"percent is well inside what luck produces."
        )

    failed = board[~board["ok"]]
    if not failed.empty:
        ui.warning_list(
            [f"{row.label}: {row.reason}" for row in failed.itertuples()],
            title="Models that could not be fitted",
        )


# ---------------------------------------------------------------------------
# 7. what the model used
# ---------------------------------------------------------------------------
def _importance(run: dict[str, Any]) -> None:
    ui_theme.section("WHAT THE MODEL USED", "🧬")

    importance = run.get("importance")
    col_chart, col_detail = st.columns([1.6, 1.0], gap="medium")

    with col_chart:
        ui.chart(ch.feature_importance_chart(importance))
    with col_detail:
        spec = fc.MODELS.get(str(run.get("model")), None)
        if spec is not None:
            ui.model_card(
                spec.label,
                spec.description,
                rows=(("Assumes", spec.assumption), ("Weakness", spec.weakness)),
                badge_text=spec.family.replace("_", " ").title(),
            )
        weights = run.get("ensemble_weights") or {}
        if weights:
            st.caption(
                "Blend weights learned on an inner holdout inside the training block: "
                + ", ".join(
                    f"{fc.MODELS[k].label} {v * 100:.0f}%"
                    for k, v in weights.items()
                    if k in fc.MODELS
                )
            )

    if isinstance(importance, pd.DataFrame) and not importance.empty:
        top = importance.head(3)
        names = ", ".join(f"`{row.feature}`" for row in top.itertuples())
        st.caption(
            f"Importance is measured by shuffling one feature's values on held-out rows and "
            f"recording how much worse the model gets — so a feature scores zero when the "
            f"model was not using it, whatever its coefficient says. Here the three that cost "
            f"the most when removed were {names}. On a genuinely unforecastable series every "
            f"bar sits near zero, and that is information rather than a broken chart."
        )

    with st.expander(f"The {_int(run.get('n_features'), 0)} features it was offered"):
        _feature_catalogue(run)


def _feature_catalogue(run: dict[str, Any]) -> None:
    """Every feature the model saw, grouped by family, with its reading."""
    names = list(run.get("features") or ())
    if not names:
        st.caption("No feature catalogue was recorded for this run.")
        return
    catalogue = pd.DataFrame(
        {
            "Feature": names,
            "Family": [feat.family_of(n) for n in names],
            "What it measures": [feat.describe_feature(n) for n in names],
        }
    )
    ui.dataframe(catalogue.sort_values(["Family", "Feature"]), height=340)

    families = run.get("families") or ()
    missing = [f for f in feat.FEATURE_FAMILIES if f not in families]
    st.caption(
        "Every one is computed from observations up to and including the session it sits on, "
        "and every one is scale-free — a log return, a ratio, a z-score or a position inside a "
        "range. Nothing carries the price level, because a model handed the raw close learns "
        '"it went up before, so it goes up".'
        + (
            f" Families skipped for this instrument: {', '.join(missing)} — the provider "
            f"supplied no column to build them from."
            if missing
            else ""
        )
    )


# ---------------------------------------------------------------------------
# 8. failure
# ---------------------------------------------------------------------------
def _failure(run: dict[str, Any], controls: _Controls, rows: int) -> None:
    """Say plainly why no model was fitted, rather than showing empty charts."""
    ui.empty_state(
        "No forecast was produced",
        run.get("reason") or "The forecaster declined to fit a model to this history.",
        icon="🛑",
        hint=f"A purged walk-forward needs at least {FORECAST.min_train_rows} labelled rows "
        f"after the feature lookbacks are satisfied. This window supplied {rows} bars.",
    )
    st.caption(
        f"Attempted: {controls.ticker}, "
        f"{_HORIZON_LABELS.get(controls.horizon, f'{controls.horizon} sessions')} horizon, "
        f"{_WINDOW_NAMES.get(controls.window, controls.window)} of "
        f"history. Nothing was estimated, padded or substituted to get past the shortfall."
    )


# ---------------------------------------------------------------------------
# 9. portfolio-level forecast
# ---------------------------------------------------------------------------
def _portfolio_tab(result: Any, controls: _Controls, model: str) -> None:
    horizon = controls.horizon
    values = getattr(result, "values", None)
    if values is None or len(values) < FORECAST.min_train_rows:
        ui.empty_state(
            "Not enough portfolio history",
            f"A portfolio-level model needs at least {FORECAST.min_train_rows} sessions of "
            f"value history after the feature lookbacks are satisfied; this analysis has "
            f"{0 if values is None else len(values)}.",
            icon="🕰",
            hint="Widen the history window in the sidebar — the portfolio value series is "
            "built from the overlapping history of every holding, so one recent listing "
            "shortens it for the whole book.",
        )
        return

    ui_theme.section("TOP-DOWN: THE PORTFOLIO AS ONE INSTRUMENT", "📈")
    st.caption(
        "The same pipeline, applied to your portfolio's own value series. It sees no "
        "High/Low/Volume — a portfolio has no bar — so the volume family and the bar-shape "
        "features are skipped rather than fabricated, and the model is told which ones it is "
        "missing."
    )

    signature = (
        "portfolio",
        horizon,
        model,
        str(getattr(result, "generated_at", "")),
        str(len(values)),
    )
    run = _memo(_RUN_CACHE, signature, lambda: _run_forecast(values, horizon, model))

    if not run.get("ok"):
        ui.empty_state(
            "No portfolio forecast",
            run.get("reason") or "The forecaster declined to fit the portfolio value series.",
            icon="🛑",
        )
        return

    _verdict(run)
    currency = getattr(result.config, "base_currency", "INR")
    _forecast_kpis(run, currency)
    ui.insight_list(fc.forecast_insights(run))

    ui_theme.spacer(0.4)
    _validation(run)

    ui_theme.spacer(0.4)
    _bottom_up(result, controls, model)


def _bottom_up(result: Any, controls: _Controls, model: str) -> None:
    horizon = controls.horizon
    """Per-holding forecasts, weighted into one portfolio expectation."""
    ui_theme.section("BOTTOM-UP: ONE MODEL PER HOLDING", "🧩")

    positions = getattr(result, "positions", None)
    if positions is None or len(positions) == 0:
        st.caption("No priced holdings to forecast.")
        return
    if len(positions) > _MAX_BOTTOM_UP:
        st.caption(
            f"This portfolio holds {len(positions)} names. Each one needs its own purged "
            f"walk-forward, so the bottom-up view is capped at {_MAX_BOTTOM_UP} holdings to "
            f"keep the page usable. Use the Security tab for the rest."
        )
        return

    st.caption(
        f"Fits and validates a separate model for each of the {len(positions)} holdings, then "
        f"weights the results. Expected return aggregates exactly; the interval does not, "
        f"which is why two bounds are reported rather than one."
    )
    if not st.button("Forecast every holding", type="primary", key="forecast_bottom_up"):
        return

    period = controls.window
    tickers = tuple(str(t) for t in positions["ticker"])
    signature = (tickers, horizon, model, period, str(getattr(result, "generated_at", "")))

    def build() -> dict[str, Any]:
        runs: dict[str, dict[str, Any]] = {}
        progress = st.progress(0.0, text="Fitting per-holding models…")
        for position, ticker in enumerate(tickers, start=1):
            progress.progress(
                position / len(tickers), text=f"Fitting {ticker} ({position}/{len(tickers)})…"
            )
            try:
                bars = data_access.load_ohlcv(ticker, period)
            except Exception:  # noqa: BLE001 - one unavailable symbol is a data note
                continue
            runs[ticker] = fc.run_forecast(bars, horizon, model, with_importance=False)
        progress.empty()
        return runs

    runs = _memo(_BOTTOM_UP_CACHE, signature, build)
    weights = {
        str(row.ticker): float(getattr(row, "weight_pct", 0.0)) for row in positions.itertuples()
    }
    aggregate = fc.portfolio_forecast(runs, weights)

    if not aggregate.get("ok"):
        ui.empty_state(
            "No holding produced a usable forecast",
            aggregate.get("reason") or "Every per-holding model declined to fit.",
            icon="🛑",
        )
        return

    expected = _num(aggregate.get("expected_return"))
    low_co, high_co = aggregate.get("band_comonotonic", (float("nan"), float("nan")))
    low_in, high_in = aggregate.get("band_independent", (float("nan"), float("nan")))
    coverage = _num(aggregate.get("coverage"))

    ui.kpi_row(
        [
            {
                "label": "Weighted expected return",
                "value": fmt_ratio_pct(expected, 2, signed=True),
                "tone": delta_tone(expected),
                "delta": f"over {_horizon_words(horizon)}",
                "delta_tone": "muted",
                "footnote": f"{aggregate.get('n_holdings', 0)} holdings forecast",
                "help_text": "Simple returns weighted by position weight. Log returns would "
                "be wrong to weight directly — Jensen's inequality.",
            },
            {
                "label": "Band if everything moves together",
                "value": f"{fmt_ratio_pct(low_co, 1, signed=True)} … {fmt_ratio_pct(high_co, 1, signed=True)}",
                "tone": "neutral",
                "delta": "perfect correlation",
                "delta_tone": "muted",
                "footnote": "The widest defensible band — the crash case",
            },
            {
                "label": "Band if nothing does",
                "value": f"{fmt_ratio_pct(low_in, 1, signed=True)} … {fmt_ratio_pct(high_in, 1, signed=True)}",
                "tone": "neutral",
                "delta": "zero correlation",
                "delta_tone": "muted",
                "footnote": "The narrowest — and no equity book has it",
            },
            {
                "label": "Weight forecast",
                "value": fmt_ratio_pct(coverage, 1),
                "tone": "gain" if coverage > 0.9 else "warn",
                "delta": f"{fmt_ratio_pct(1.0 - coverage, 1)} unforecastable",
                "delta_tone": "muted",
                "footnote": "Uncovered weight is disclosed, never treated as flat",
            },
        ],
        columns=4,
    )

    contributions = aggregate.get("contributions")
    if isinstance(contributions, pd.DataFrame) and not contributions.empty:
        display = pd.DataFrame(
            {
                "Ticker": contributions["ticker"],
                "Weight": contributions["weight_pct"].map(lambda v: fmt_pct(v, 1)),
                "Expected": contributions["predicted_return"].map(
                    lambda v: fmt_ratio_pct(v, 2, signed=True)
                ),
                "Interval": contributions.apply(
                    lambda r: (
                        f"{fmt_ratio_pct(r['lower_return'], 1, signed=True)} … "
                        f"{fmt_ratio_pct(r['upper_return'], 1, signed=True)}"
                    ),
                    axis=1,
                ),
                "Contribution": contributions["contribution"].map(
                    lambda v: fmt_ratio_pct(v, 2, signed=True)
                ),
                "P(up)": contributions["prob_up"].map(lambda v: fmt_ratio_pct(v, 0)),
                "Skill": contributions["skill"].map(
                    lambda v: fmt_pct(_num(v) * 100.0, 2, signed=True)
                ),
            }
        )
        ui.dataframe(display)

        skilled = int(contributions["skill"].gt(FORECAST_MIN_SKILL).sum())
        st.caption(
            f"{skilled} of {len(contributions)} holdings had a model that beat a random walk "
            f"out of sample. A weighted average of mostly-unskilled forecasts is still an "
            f"unskilled forecast — the Skill column is there so that cannot be read past. "
            f"The two bands differ because the forecaster never estimated a correlation "
            f"between its own errors across holdings; the truth is between them."
        )


# ---------------------------------------------------------------------------
# 10. method
# ---------------------------------------------------------------------------
def _method_tab() -> None:
    ui_theme.section("THE MODELS ON OFFER", "🧠")
    specs = list(fc.MODELS.values())
    for start in range(0, len(specs), 2):
        columns = st.columns(2, gap="medium")
        for column, spec in zip(columns, specs[start : start + 2]):
            with column:
                ui.model_card(
                    spec.label,
                    spec.description,
                    rows=(("Assumes", spec.assumption), ("Weakness", spec.weakness)),
                    badge_text=spec.family.replace("_", " ").title(),
                    badge_tone="neutral" if spec.family == "baseline" else "accent",
                )
        ui_theme.spacer(0.35)

    ui_theme.section("HOW A FORECAST IS BUILT", "⚙")
    st.markdown(
        f"""
**1 · Features.** {len(feat.FEATURE_FAMILIES)} families of scale-free signals are computed
from the instrument's own daily bars — lagged returns, momentum over five windows,
realised volatility and its short/long regime ratio, distance from four moving averages,
Wilder RSI, MACD, Bollinger %B, ATR, position inside the trailing 252-session range,
volume z-scores, and cyclical calendar encodings. Every one uses observations up to and
including the session it sits on. Nothing carries the price level.

**2 · Label.** The target is `log(C[t+h] / C[t])` — the log return over the next *h*
sessions. Log rather than simple because it is additive across time and far closer to
symmetric, which every squared-error loss implicitly assumes.

**3 · Purged walk-forward.** The data is split into {FORECAST.n_splits} sequential
expanding-window blocks covering the last {FORECAST.test_fraction:.0%} of the rows. Train,
test on what follows, move forward. **No shuffling, ever** — a shuffled split lets a model
train on Thursday to predict Wednesday, and it is how published stock predictors report
95% accuracy. Between each training block and its test block, `h − 1 + {FORECAST.purge_extra}`
rows are **purged**, because an *h*-session label physically overlaps the *h* sessions
that follow it.

**4 · Scoring against a random walk.** Every skill number is
`1 − MAE_model / MAE_random-walk`, where the random walk predicts exactly zero. Negative
skill means the model was worse than assuming the price does not move — the ordinary
result, reported as such.

**5 · Significance.** A Diebold–Mariano test on the squared-error loss differential, with
a Newey–West variance at `h − 1` lags (overlapping horizons make the differential serially
correlated by construction) and the Harvey–Leybourne–Newbold small-sample correction.

**6 · Conformal interval.** The band is the empirical quantile range of the model's own
out-of-sample residuals at {FORECAST.conformal_alpha / 2:.0%} and
{1 - FORECAST.conformal_alpha / 2:.0%} — not a Gaussian assumption, not the training fit.
Signed rather than symmetric, because equity return errors are skewed. A bad model
produces a wide band automatically.

**7 · Direction probability.** The share of the empirical predictive distribution
(`point forecast + out-of-sample residual`) that lands above zero. No Normal assumption,
and it collapses toward the base rate exactly when the model has nothing to say.
"""
    )

    ui_theme.section("WHAT THIS CANNOT DO", "🚧")
    st.markdown(
        """
- **It cannot see anything but price.** No earnings, no news, no order flow, no
  macro, no other instrument. A model built on one series' own history is a
  weak-form efficiency test, and weak-form efficiency mostly holds.
- **It cannot survive a regime change**, and it has no way to know one has started. The
  fold-by-fold table is the only warning available, and it is retrospective.
- **Skill measured on one instrument over one window is a weak claim.** Test the same
  model on ten names before you believe any of them. The leaderboard here makes the
  comparison easy and the result is usually humbling.
- **It is not backtested as a strategy.** Forecast accuracy is not profit: this page
  models no transaction cost, no spread, no slippage, no market impact, no position
  sizing, no borrow cost and no tax. A model with real statistical skill can still lose
  money after all six.
- **The interval is marginal, not conditional.** It covers its nominal share of outcomes
  on average across the backtest. It does not widen because today is a volatile day.
- **The provider's history is split- and dividend-adjusted as of today**, so the features
  on an old row are computed from numbers that were not knowable on that date. The effect
  is small for a liquid equity and real for one with a corporate-action history.
- **No deep learning, deliberately.** An LSTM or a transformer has millions of parameters
  and a daily bar history has a few thousand rows with a signal-to-noise ratio near zero.
  That combination produces a model that memorises its training set and a dependency that
  is larger than the rest of this application put together. The constraint is data, not
  ambition.
"""
    )

    ui_theme.section("THE FEATURE CATALOGUE", "🧬")
    catalogue = feat.feature_catalogue()
    if not catalogue.empty:
        ui.dataframe(
            catalogue.rename(
                columns={
                    "feature": "Feature",
                    "family": "Family",
                    "description": "What it measures",
                }
            ),
            height=400,
        )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Price forecast",
        "A model, its forecast, and the out-of-sample evidence about whether it works",
        icon="🔮",
    )

    result = guards.require_result()
    if result is None:
        return

    ui.hero(
        "Machine learning · walk-forward validated",
        "A forecast is only worth what its backtest says it is worth",
        FORECAST_DISCLAIMER,
    )

    if not fc.SKLEARN_AVAILABLE:
        st.warning(
            "scikit-learn is not installed, so only the random-walk and drift baselines are "
            "available. Install the project requirements to enable the fitted models."
        )

    controls = _controls(result)
    if controls is None:
        ui.empty_state(
            "No instrument to forecast",
            "The analysed portfolio has no priced holding to build a model from.",
            icon="🔎",
            hint="Add a holding on the Portfolio › Build page and run the analysis again.",
        )
        return

    tab_security, tab_portfolio, tab_method = st.tabs(["Instrument", "Portfolio", "How this works"])

    with tab_security:
        ohlcv = _load_history(controls.ticker, controls.window)
        if ohlcv is None or len(ohlcv) == 0 or "Close" not in ohlcv.columns:
            ui.empty_state(
                f"No price history for {controls.ticker}",
                "The provider returned no usable daily bars over the training window.",
                icon="📭",
                hint="Check the suffix (.NS / .BO for India, none for US listings), or pick a "
                "shorter training window.",
            )
            return

        model, board = _resolve_model(ohlcv, controls)
        run = _forecast(ohlcv, controls, model)
        currency = _currency_for(result, controls.ticker)

        ui.badges(
            [
                (controls.ticker, "accent"),
                (fc.MODELS[model].label, "accent"),
                (_HORIZON_LABELS.get(controls.horizon, f"{controls.horizon} sessions"), "neutral"),
                (f"{len(ohlcv):,} daily bars", "neutral"),
                (f"{_int(run.get('n_train'), 0):,} labelled rows", "muted"),
                (f"{_int(run.get('folds'), 0)} purged folds", "muted"),
                ("Statistical estimate, not a prediction", "warn"),
            ]
        )

        if not run.get("ok"):
            _failure(run, controls, len(ohlcv))
            return

        _verdict(run)
        _forecast_kpis(run, currency)

        ui_theme.spacer(0.4)
        _cone(ohlcv, _path(ohlcv, controls, model), run, currency)

        ui_theme.spacer(0.4)
        ui_theme.section("WHAT THE EVIDENCE SAYS", "💬")
        ui.insight_list(fc.forecast_insights(run))
        st.caption(run.get("note", ""))

        ui_theme.spacer(0.4)
        _validation(run)

        ui_theme.spacer(0.4)
        _leaderboard_section(ohlcv, controls, board)

        ui_theme.spacer(0.4)
        _importance(run)

        ui_theme.spacer(0.4)
        ui.methodology(
            "Why this window differs from the sidebar's",
            f"Risk statistics elsewhere in the app use the "
            f"**{PERIOD_LABELS.get(getattr(result.config, 'period', ''), 'selected')}** window "
            f"you chose, because those numbers are meant to describe that period. A model is "
            f"different: it needs every row it can get. The longest feature lookback is "
            f"{TRADING_DAYS} sessions, the walk-forward holds out "
            f"{FORECAST.test_fraction:.0%} of what remains, and each fold needs at least "
            f"{FORECAST.min_initial_train} rows to train on — so a one-year window cannot "
            f"produce a single valid fold. This page therefore fetches its own history and "
            f"says so, rather than quietly analysing a different period from the rest of the "
            f"application.",
        )

    with tab_portfolio:
        model_key = controls.model if controls.model != _AUTO else "ridge"
        _portfolio_tab(result, controls, model_key)

    with tab_method:
        _method_tab()
