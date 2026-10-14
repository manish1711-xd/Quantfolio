"""Monte Carlo -- the distribution of outcomes this book could print next.

The page answers one question and refuses to answer a different one that looks
like it: *given the return distribution this portfolio actually printed, what
range of values does propagating it forward produce?* That is not a forecast.
Nothing here knows anything about the future; every path is a rearrangement or a
parameterisation of the past, and the only honest output is the **width** of the
resulting distribution, not its centre.

Everything else on the page is read off the completed
:class:`~src.models.AnalysisResult`. The one exception -- and it is a deliberate
one -- is :func:`src.analytics.montecarlo.run_simulation`, which is invoked here
because the simulation is driven by controls that live on this page (model,
horizon, path count, seed, block length) and cannot be precomputed by the
pipeline. The result is cached in session state against the exact control tuple
that produced it, so a rerun triggered by an unrelated widget never silently
redraws a different set of paths under the same numbers.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import pandas as pd
import streamlit as st

from src import charts as ch
from src.analytics import montecarlo as mc
from src.config import MIN_OBSERVATIONS, SIMULATION, TRADING_DAYS
from src.ui import components as ui
from src.ui import guards
from src.ui import theme as ui_theme
from src.ui.formatters import (
    DASH,
    delta_tone,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
)

# ---------------------------------------------------------------------------
# control vocabulary
# ---------------------------------------------------------------------------
#: Selectbox label -> canonical method name accepted by ``mc.run_simulation``.
#: The bootstrap leads because it assumes least about the return distribution.
_METHODS: tuple[tuple[str, str], ...] = (
    ("Block bootstrap", "block_bootstrap"),
    ("IID bootstrap", "bootstrap"),
    ("GBM (lognormal)", "gbm"),
    ("Student-t (fat tails)", "student_t"),
)

#: Canonical method name -> the label used on chips and in the parameter grid.
_METHOD_NAMES: dict[str, str] = {
    "block_bootstrap": "Block bootstrap",
    "bootstrap": "IID bootstrap",
    "gbm": "GBM (lognormal)",
    "student_t": "Student-t",
}

#: The same four, phrased to sit mid-sentence.
_METHOD_PROSE: dict[str, str] = {
    "block_bootstrap": "a block bootstrap of its own returns",
    "bootstrap": "an IID bootstrap of its own returns",
    "gbm": "geometric Brownian motion",
    "student_t": "Student-t innovations",
}

#: Horizons offered, in trading sessions, with their calendar translation.
_HORIZONS: tuple[int, ...] = (63, 126, 252, 504, 1260)
_HORIZON_LABELS: dict[int, str] = {
    63: "3 months (63 sessions)",
    126: "6 months (126 sessions)",
    252: "1 year (252 sessions)",
    504: "2 years (504 sessions)",
    1260: "5 years (1,260 sessions)",
}
_HORIZON_WORDS: dict[int, str] = {
    63: "three months",
    126: "six months",
    252: "one year",
    504: "two years",
    1260: "five years",
}

#: Path counts offered. 500 is a sketch, 5,000 stabilises the tail percentiles.
_PATH_CHOICES: tuple[int, ...] = (500, 2000, 5000)

#: Upper bound on the block length, and the sample fraction it may not exceed.
#: Blocks longer than a quarter of the sample stop resampling and start
#: replaying the window itself.
_MAX_BLOCK: int = 63
_BLOCK_SAMPLE_DIVISOR: int = 4

#: Trajectories drawn on the spaghetti chart. Beyond a few hundred it is mud.
_PATHS_DRAWN: int = 120

#: Session key holding the last simulation, fingerprinted by its inputs.
_CACHE_KEY = "simulation_last_run"

#: Probability-of-loss colouring only. These are display thresholds, not
#: statistical ones, and nothing downstream depends on them.
_LOSS_ALARM: float = 0.50
_LOSS_WATCH: float = 0.30

#: Upper bound of the seed input: the numpy generator accepts far more, but a
#: 32-bit integer is what a person can copy into a report and retype correctly.
_MAX_SEED: int = 2**31 - 1


class _Controls(NamedTuple):
    """Everything that determines a run -- and therefore the cache key."""

    method: str
    label: str
    horizon_days: int
    n_paths: int
    seed: int
    block_size: int


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
    """True when ``value`` is a real, finite number worth printing."""
    return math.isfinite(_num(value))


def _int(value: Any, default: int) -> int:
    """Best-effort int, falling back to ``default`` for None/NaN/garbage."""
    number = _num(value)
    return int(number) if math.isfinite(number) else default


def _plain(text: Any) -> str:
    """Neutralise markdown emphasis so a model note renders exactly as written.

    ``montecarlo`` writes its note in plain English and uses ``*`` for
    multiplication (``start * exp(mu * T)``). Two of those in one sentence is an
    italic run in markdown, which silently eats the operators.
    """
    return str(text).replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")


def _total_return(value: Any, start: Any) -> float:
    """Total return over the horizon as a DECIMAL: ``V_T / V_0 - 1``."""
    end, base = _num(value), _num(start)
    if not (math.isfinite(end) and math.isfinite(base)) or base <= 0.0:
        return float("nan")
    return end / base - 1.0


def _annualised(value: Any, start: Any, years: Any) -> float:
    """Constant compound rate connecting the two ends, as a DECIMAL."""
    end, base, span = _num(value), _num(start), _num(years)
    if not (math.isfinite(end) and math.isfinite(base) and math.isfinite(span)):
        return float("nan")
    if base <= 0.0 or end <= 0.0 or span <= 0.0:
        return float("nan")
    return (end / base) ** (1.0 / span) - 1.0


def _level_of(column: Any) -> float:
    """``'p95' -> 95.0``; NaN for any label that is not a percentile."""
    text = str(column).strip().lower()
    return _num(text[1:] if text.startswith("p") else text)


def _reading(level: float) -> str:
    """Plain-English gloss on one percentile of the terminal distribution."""
    if not math.isfinite(level):
        return DASH
    if level < 50.0:
        return f"{level:g}% of paths finished at or below this"
    if level > 50.0:
        return f"{100.0 - level:g}% of paths finished at or above this"
    return "Half the paths finished above this, half below"


def _loss_tone(prob: Any) -> str:
    """Colour only: amber past a third of futures losing, red past half."""
    value = _num(prob)
    if not math.isfinite(value):
        return "muted"
    if value >= _LOSS_ALARM:
        return "loss"
    return "warn" if value >= _LOSS_WATCH else "neutral"


def _horizon_words(days: int) -> str:
    return _HORIZON_WORDS.get(days, f"{days} trading days")


def _observations(result: Any) -> int:
    """Daily return observations the pipeline measured this window on."""
    reported = _num((result.risk or {}).get("n_obs"))
    if math.isfinite(reported) and reported > 0:
        return int(reported)
    return max(int(result.n_days) - 1, 0)


# ---------------------------------------------------------------------------
# 1. controls
# ---------------------------------------------------------------------------
def _controls(result: Any, cfg: Any) -> _Controls:
    """Render the five settings that define a run and return them as a tuple."""
    ui_theme.section("PROJECTION SETTINGS", "🎛")

    defaults = getattr(cfg, "simulation", None) or SIMULATION
    n_obs = _observations(result)

    col_method, col_horizon, col_paths, col_seed = st.columns([1.5, 1.5, 1.0, 0.9], gap="medium")

    with col_method:
        label = st.selectbox(
            "Return model",
            options=[name for name, _ in _METHODS],
            index=0,
            key="sim_method",
            help="Block bootstrap resamples contiguous runs of real days, so some "
            "volatility clustering survives. IID bootstrap resamples single "
            "days: real fat tails, no clustering. GBM assumes normal "
            "log-returns. Student-t keeps the normal's shape but fattens "
            "both tails.",
        )
        method = dict(_METHODS)[label]

    with col_horizon:
        horizon = int(
            st.select_slider(
                "Horizon",
                options=_HORIZONS,
                value=(defaults.horizon_days if defaults.horizon_days in _HORIZONS else 252),
                format_func=lambda d: _HORIZON_LABELS.get(d, f"{d} sessions"),
                key="sim_horizon",
                help="Trading sessions projected forward. The further out, the "
                "more the answer depends on the drift estimate, which is the "
                "least reliable input in the model.",
            )
        )

    with col_paths:
        n_paths = int(
            st.select_slider(
                "Paths",
                options=_PATH_CHOICES,
                value=(defaults.n_paths if defaults.n_paths in _PATH_CHOICES else 2000),
                format_func=lambda n: f"{n:,}",
                key="sim_paths",
                help="How many futures to draw. The median stabilises quickly; the "
                "5th and 95th percentiles need thousands before they stop "
                "moving between seeds.",
            )
        )

    with col_seed:
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=_MAX_SEED,
                value=_int(defaults.seed, 42),
                step=1,
                key="sim_seed",
                help="Fixes the pseudo-random draw. Same settings, same seed, same "
                "paths -- which is what makes any number below auditable.",
            )
        )

    block = 1
    if method == "block_bootstrap":
        ceiling = max(2, min(_MAX_BLOCK, n_obs // _BLOCK_SAMPLE_DIVISOR))
        block_col, note_col = st.columns([1.0, 3.5], gap="medium")
        with block_col:
            block = int(
                st.number_input(
                    "Block length (sessions)",
                    min_value=2,
                    max_value=ceiling,
                    value=min(max(_int(defaults.block_size, 10), 2), ceiling),
                    step=1,
                    key="sim_block",
                    help="Length of the contiguous run resampled as one unit.",
                )
            )
        with note_col:
            st.caption(
                f"Dependence up to **{block} sessions** -- volatility clustering, "
                f"short-run momentum and reversal -- survives inside a block and is "
                f"broken at every join. Longer blocks preserve more structure but "
                f"draw from fewer distinct starting points; the ceiling here is "
                f"{ceiling} sessions, a quarter of the {n_obs} observations "
                f"available, beyond which the run stops resampling history and "
                f"starts replaying it."
            )

    return _Controls(method, label, horizon, n_paths, seed, block)


# ---------------------------------------------------------------------------
# 2. running (and caching) the simulation
# ---------------------------------------------------------------------------
def _data_signature(result: Any) -> tuple[str, str, str]:
    """Fingerprint of the analysis inputs a simulation depends on.

    Formatted as strings deliberately: a NaN starting value must compare equal
    to itself, or the cache would miss on every rerun.
    """
    try:
        last = float(result.values.iloc[-1])
    except Exception:  # noqa: BLE001 - a cache key must never raise
        last = float("nan")
    return (str(result.generated_at), str(result.n_days), f"{last:.6f}")


def _simulate(result: Any, controls: _Controls) -> dict[str, Any]:
    """The run for exactly these controls, from session state when unchanged.

    The cached entry carries the full ``(n_paths, horizon + 1)`` path matrix,
    which is what the spaghetti chart draws; at the largest settings that is a
    few tens of megabytes held for the session, and it is the price of not
    re-simulating on every unrelated widget interaction.
    """
    signature = (controls, _data_signature(result))
    cached = st.session_state.get(_CACHE_KEY)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return dict(cached.get("sim") or {})

    with ui.loading(
        f"Drawing {controls.n_paths:,} paths over {controls.horizon_days} trading days"
    ):
        sim = mc.run_simulation(
            result.values,
            result.returns,
            method=controls.method,
            horizon_days=controls.horizon_days,
            n_paths=controls.n_paths,
            seed=controls.seed,
            block_size=controls.block_size,
        )
    st.session_state[_CACHE_KEY] = {"signature": signature, "sim": sim}
    return sim


def _failure(sim: dict[str, Any], controls: _Controls, result: Any) -> None:
    """Say plainly why no paths were drawn, rather than showing empty charts."""
    ui.empty_state(
        "No simulation was run",
        sim.get("reason") or "The simulator declined to produce paths from these inputs.",
        icon="🛑",
        hint=f"A forward simulation needs at least {MIN_OBSERVATIONS} daily return "
        f"observations; this window supplied {int(sim.get('n_obs', 0) or 0)}. "
        f"Widen the history window in the sidebar.",
    )
    st.caption(
        f"Attempted: {controls.label}, {_HORIZON_LABELS.get(controls.horizon_days, '')} "
        f"horizon, {controls.n_paths:,} paths, seed {controls.seed}, on "
        f"{result.n_days} sessions of history. Nothing was estimated or "
        f"substituted to get past the shortfall."
    )


def _run_chips(sim: dict[str, Any], result: Any) -> None:
    """Provenance strip: the exact settings behind every number below."""
    method = str(sim.get("method", ""))
    horizon = _int(sim.get("horizon_days"), 0)
    chips: list[tuple[str, str]] = [
        (_METHOD_NAMES.get(method, method or DASH), "accent"),
        (f"Horizon {_HORIZON_LABELS.get(horizon, f'{horizon} sessions')}", "neutral"),
        (f"{_int(sim.get('n_paths'), 0):,} paths", "neutral"),
        (f"Seed {_int(sim.get('seed'), 0)}", "muted"),
    ]
    if method == "block_bootstrap":
        chips.append((f"Block {_int(sim.get('block_size'), 1)} sessions", "neutral"))
    if method == "student_t":
        chips.append((f"df {fmt_num(sim.get('df'), 1)}", "neutral"))
    chips.append((f"Estimated on {_int(sim.get('n_obs'), 0)} daily returns", "muted"))
    chips.append((f"{result.n_days} sessions of history", "muted"))
    chips.append(("Simulated outcomes, not a forecast", "warn"))
    ui.badges(chips)


# ---------------------------------------------------------------------------
# 3. the fan
# ---------------------------------------------------------------------------
def _fan(sim: dict[str, Any], cfg: Any) -> None:
    ui_theme.section("THE CONE OF OUTCOMES", "🌪")

    ui.chart(
        ch.fan_chart(
            sim.get("percentile_paths"),
            _num(sim.get("start_value")),
            getattr(cfg, "currency_symbol", ""),
            height=520,
        )
    )

    ccy = getattr(cfg, "base_currency", "INR")
    start = _num(sim.get("start_value"))
    p5 = _num(sim.get("p5_terminal"))
    p95 = _num(sim.get("p95_terminal"))
    horizon = _int(sim.get("horizon_days"), 0)
    width = p95 - p5 if (_ok(p5) and _ok(p95)) else float("nan")

    st.caption(
        f"Day 0 is today's {fmt_money(start, ccy, 0)}; the cone opens from there "
        f"because uncertainty compounds with the square root of time. Each band is "
        f"a percentile of the {_int(sim.get('n_paths'), 0):,} simulated values **on "
        f"that day**, not a trajectory -- no single simulated portfolio traced the "
        f"P5 line, and the path that is worst at day 60 is usually not the one that "
        f"is worst at day {horizon}. By the horizon the P5-P95 band spans "
        f"{fmt_money(p5, ccy, 0)} to {fmt_money(p95, ccy, 0)}, a spread of "
        f"{fmt_money(width, ccy, 0)} or {fmt_ratio_pct(_total_return(width + start, start))} "
        f"of today's value. That width is the output worth reading. The centre is "
        f"an estimate of drift, and drift is the least reliable number in the model."
    )


# ---------------------------------------------------------------------------
# 4. headline outcomes
# ---------------------------------------------------------------------------
def _outcome_kpis(sim: dict[str, Any], cfg: Any) -> None:
    ui_theme.section("WHERE IT COULD LAND", "🎯")

    ccy = getattr(cfg, "base_currency", "INR")
    start = _num(sim.get("start_value"))
    median = _num(sim.get("median_terminal"))
    mean = _num(sim.get("mean_terminal"))
    p5 = _num(sim.get("p5_terminal"))
    p95 = _num(sim.get("p95_terminal"))
    prob_loss = _num(sim.get("prob_loss"))
    prob_gain = _num(sim.get("prob_gain_10pct"))
    n_paths = _int(sim.get("n_paths"), 0)
    horizon = _int(sim.get("horizon_days"), 0)
    words = _horizon_words(horizon)

    median_return = _total_return(median, start)
    p5_return = _total_return(p5, start)
    p95_return = _total_return(p95, start)

    cards: list[dict[str, Any]] = [
        {
            "label": "Median outcome",
            "value": fmt_money(median, ccy, 0),
            "tone": delta_tone(median_return),
            "delta": f"{fmt_ratio_pct(median_return, signed=True)} over {words}",
            "delta_tone": delta_tone(median_return),
            "footnote": f"Half of {n_paths:,} paths finished above this, half below",
            "help_text": (
                "The 50th percentile of terminal values. The mean of the same paths "
                f"is {fmt_money(mean, ccy, 0)} -- higher, because compounding is "
                "multiplicative and the upside is unbounded while the downside "
                "stops at zero."
            ),
        },
        {
            "label": "5th percentile",
            "value": fmt_money(p5, ccy, 0),
            "tone": delta_tone(p5_return),
            "delta": fmt_ratio_pct(p5_return, signed=True),
            "delta_tone": delta_tone(p5_return),
            "footnote": "One path in twenty finished at or below this",
            "help_text": (
                "A bad outcome, not the worst one. Five percent of simulated "
                "futures were worse than this, and the model's worst case is "
                f"{fmt_money(sim.get('worst_terminal'), ccy, 0)}."
            ),
        },
        {
            "label": "95th percentile",
            "value": fmt_money(p95, ccy, 0),
            "tone": delta_tone(p95_return),
            "delta": fmt_ratio_pct(p95_return, signed=True),
            "delta_tone": delta_tone(p95_return),
            "footnote": "One path in twenty finished at or above this",
            "help_text": (
                "The upper edge of the band, not a target. It is as much a "
                "statement about the sample's volatility as about its return."
            ),
        },
        {
            "label": "Probability of loss",
            "value": fmt_ratio_pct(prob_loss, 1),
            "tone": _loss_tone(prob_loss),
            "delta": (
                f"{fmt_ratio_pct(prob_gain, 1)} finished "
                f"{fmt_ratio_pct(mc.GAIN_THRESHOLD, 0)} or more up"
            ),
            "delta_tone": "muted",
            "footnote": f"Share of paths ending below {fmt_money(start, ccy, 0)}",
            "help_text": (
                "Nominal, before inflation, costs, taxes and any contributions. "
                "It is the share of simulated futures that finished below today's "
                "value, not a probability of losing money in any real-world sense."
            ),
        },
    ]
    ui.kpi_row(cards, columns=4)

    var_terminal = _num(sim.get("var_terminal"))
    cvar_terminal = _num(sim.get("cvar_terminal"))
    confidence = _num(sim.get("confidence"))
    if _ok(median_return) and _ok(prob_loss):
        st.caption(
            f"The two halves of that row are not in conflict. The median path gains "
            f"{fmt_ratio_pct(median_return, signed=True)} over {words} while "
            f"{fmt_ratio_pct(prob_loss, 1)} of futures still end below today's "
            f"{fmt_money(start, ccy, 0)}: a right-skewed terminal distribution puts "
            f"the mean ({fmt_money(mean, ccy, 0)}) above the median "
            f"({fmt_money(median, ccy, 0)}) and leaves a large mass sitting under "
            f"the starting line. At "
            f"{fmt_pct(confidence * 100.0, 0) if _ok(confidence) else DASH} "
            f"confidence the terminal VaR is {fmt_ratio_pct(var_terminal)} -- in the "
            f"worst {fmt_ratio_pct(1.0 - confidence, 0) if _ok(confidence) else DASH} "
            f"of simulated futures the book is at least that far down at the "
            f"horizon, and the average of that tail (expected shortfall) is "
            f"{fmt_ratio_pct(cvar_terminal)}."
        )

    ui.methodology(
        "What these four numbers are, and what they are not",
        f"""
**Terminal values are end-of-horizon only.** Every card describes where a path
*finished* after {horizon} sessions. None of them says anything about the route:
a path can be {fmt_ratio_pct(0.30, 0)} under water in month three and still
finish at the median. Path-dependent risk -- the drawdown you would actually have to sit
through -- is not on this page; it is on **Risk analytics** and **Stress
testing**, measured on history rather than simulated.

**Probability of loss** = `count(terminal < start) / {n_paths}`, in nominal base
currency, with no contributions, no withdrawals, no rebalancing, no brokerage,
no taxes and no inflation. Adjusting for inflation alone would move it several
points at any realistic horizon.

**Terminal VaR and expected shortfall** are quoted over the **whole
{words} horizon**, not per day, and against the starting value. They are
percentiles of a *simulated* distribution, so they inherit every assumption the
generator makes -- they are not the historical VaR on the Risk page and should
never be compared with it directly.

**Percentiles carry sampling error too.** With {n_paths:,} paths the median is
stable but P5 and P95 still move between seeds. If a conclusion flips when the
seed changes, the path count was too low for that conclusion.
        """,
    )


# ---------------------------------------------------------------------------
# 5. distribution and individual paths
# ---------------------------------------------------------------------------
def _shape(sim: dict[str, Any], cfg: Any) -> None:
    ui_theme.section("THE SHAPE OF THE OUTCOME SET", "📐")

    symbol = getattr(cfg, "currency_symbol", "")
    ccy = getattr(cfg, "base_currency", "INR")
    start = _num(sim.get("start_value"))
    n_paths = _int(sim.get("n_paths"), 0)
    drawn = min(_PATHS_DRAWN, n_paths) if n_paths else 0

    left, right = st.columns(2, gap="large")

    with left:
        ui.chart(ch.terminal_distribution(sim.get("terminal"), start, symbol, height=430))
        st.caption(
            f"The same {n_paths:,} outcomes as the fan's final column, binned. The "
            f"right skew is structural, not a property of this portfolio: a loss is "
            f"bounded at -100% while the gain is unbounded, so multiplicative "
            f"compounding always drags the long tail to the right. That is why the "
            f"mean ({fmt_money(sim.get('mean_terminal'), ccy, 0)}) sits above the "
            f"median ({fmt_money(sim.get('median_terminal'), ccy, 0)}) and why the "
            f"median is the number to quote."
        )

    with right:
        ui.chart(
            ch.sample_paths_chart(
                sim.get("paths"),
                n_show=_PATHS_DRAWN,
                start_value=start,
                currency_symbol=symbol,
                height=430,
            )
        )
        st.caption(
            f"{drawn:,} of {n_paths:,} paths, picked at even intervals rather than "
            f"randomly, so the picture is reproducible and is not a selection of the "
            f"extremes. This is the antidote to the fan chart: a real future is one "
            f"jagged line that spends long stretches far from the median, not a "
            f"smooth band. The solid line is the median across every path, and no "
            f"single path follows it."
        )


# ---------------------------------------------------------------------------
# 6. the percentile ladder
# ---------------------------------------------------------------------------
def _ladder_frame(sim: dict[str, Any], cfg: Any) -> pd.DataFrame:
    """Terminal percentiles converted into returns over the horizon."""
    frame = sim.get("percentile_paths")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    ccy = getattr(cfg, "base_currency", "INR")
    final = frame.iloc[-1]
    start = _num(sim.get("start_value"))
    years = _num(sim.get("horizon_years"))

    rows: list[dict[str, str]] = []
    for column in frame.columns:
        level = _level_of(column)
        if not math.isfinite(level):
            continue
        value = _num(final.get(column))
        rows.append(
            {
                "Percentile": f"P{level:g}",
                "Terminal value": fmt_money(value, ccy, 0),
                "Change vs today": fmt_money(value - start, ccy, 0, signed=True),
                "Total return": fmt_ratio_pct(_total_return(value, start), signed=True),
                "Annualised": fmt_ratio_pct(_annualised(value, start, years), signed=True),
                "What it means": _reading(level),
            }
        )
    return pd.DataFrame(rows)


def _ladder(sim: dict[str, Any], cfg: Any) -> None:
    ui_theme.section("PERCENTILE LADDER", "🪜")

    frame = _ladder_frame(sim, cfg)
    if frame.empty:
        ui.empty_state(
            "No percentile ladder to show",
            "The run produced no percentile paths, so there are no terminal "
            "quantiles to convert into returns.",
            icon="🪜",
        )
        return

    ui.dataframe(frame)

    ccy = getattr(cfg, "base_currency", "INR")
    start = _num(sim.get("start_value"))
    years = _num(sim.get("horizon_years"))
    horizon = _int(sim.get("horizon_days"), 0)

    st.caption(
        f"Every row is the same {_int(sim.get('n_paths'), 0):,} paths cut at a "
        f"different quantile of where they finished, then expressed against today's "
        f"{fmt_money(start, ccy, 0)}. The most extreme paths in the run finished at "
        f"{fmt_money(sim.get('worst_terminal'), ccy, 0)} and "
        f"{fmt_money(sim.get('best_terminal'), ccy, 0)} -- the edges of this sample "
        f"of futures, not bounds on the real one."
    )

    sub_year = _ok(years) and years < 1.0
    annual_verdict = "an extrapolation" if sub_year else "a compounding restatement"
    annual_note = (
        "Annualising a sub-year outcome asks what would happen if the rest of the "
        "year repeated it, which the simulation never claimed. The total-return "
        "column is the honest one at this horizon."
        if sub_year
        else "It is the constant compound rate that connects the two ends, so it "
        "reads on the same scale as the annualised return on the Overview page "
        "-- but that one is a measurement of the past and this one is a "
        "property of a simulated distribution."
    )

    ui.methodology(
        "Turning simulated values into returns",
        f"""
```
total return = terminal / start - 1                       (start = {fmt_money(start, ccy, 0)})
annualised   = (terminal / start) ** (1 / years) - 1      (years = {fmt_num(years, 2)})
years        = {horizon} sessions / {TRADING_DAYS} trading days
```

**The annualised column is a convention, and at this horizon it is
{annual_verdict}.** {annual_note}

**A percentile is not a path.** The P95 terminal value is the 95th percentile of
where paths *ended*. It is not the endpoint of the path that was 95th percentile
throughout: the ordering of paths reshuffles at every step, which is exactly why
the fan chart's bands must not be read as scenarios.

**Percentiles are estimates.** They are computed from a finite sample of
{_int(sim.get("n_paths"), 0):,} draws, so the outer rungs of this ladder carry
sampling error of their own. Re-running with a different seed will move P5 and
P95 visibly and the median barely at all.
        """,
    )


# ---------------------------------------------------------------------------
# 7. model parameters
# ---------------------------------------------------------------------------
def _parameters(sim: dict[str, Any], cfg: Any) -> None:
    ui_theme.section("MODEL PARAMETERS", "🔬")

    ccy = getattr(cfg, "base_currency", "INR")
    method = str(sim.get("method", ""))
    mu_daily = _num(sim.get("mu_daily"))
    sigma_daily = _num(sim.get("sigma_daily"))
    confidence = _num(sim.get("confidence"))

    ann_mu = mu_daily * TRADING_DAYS if _ok(mu_daily) else float("nan")
    ann_sigma = sigma_daily * math.sqrt(TRADING_DAYS) if _ok(sigma_daily) else float("nan")

    stats: list[tuple[str, str]] = [
        ("Generator", _METHOD_NAMES.get(method, method or DASH)),
        (
            "Horizon",
            f"{_int(sim.get('horizon_days'), 0)} sessions "
            f"({fmt_num(sim.get('horizon_years'), 2)}y)",
        ),
        ("Paths", f"{_int(sim.get('n_paths'), 0):,}"),
        ("Seed", f"{_int(sim.get('seed'), 0)}"),
        ("Sample size", f"{_int(sim.get('n_obs'), 0)} daily returns"),
        ("Drift, annualised", fmt_ratio_pct(ann_mu, signed=True)),
        ("Volatility, annualised", fmt_ratio_pct(ann_sigma)),
        (
            "Block length",
            f"{_int(sim.get('block_size'), 1)} sessions" if method == "block_bootstrap" else DASH,
        ),
        (
            "Student-t df",
            fmt_num(sim.get("df"), 1) if method == "student_t" else DASH,
        ),
        ("Starting value", fmt_money(sim.get("start_value"), ccy, 0)),
        ("Mean terminal value", fmt_money(sim.get("mean_terminal"), ccy, 0)),
        ("Expected return", fmt_pct(sim.get("expected_return_pct"), signed=True)),
        (
            f"Terminal VaR {fmt_pct(confidence * 100.0, 0) if _ok(confidence) else ''}".strip(),
            fmt_ratio_pct(sim.get("var_terminal")),
        ),
        ("Terminal expected shortfall", fmt_ratio_pct(sim.get("cvar_terminal"))),
        ("Worst simulated path", fmt_money(sim.get("worst_terminal"), ccy, 0)),
        ("Best simulated path", fmt_money(sim.get("best_terminal"), ccy, 0)),
    ]
    ui.stat_grid(stats, columns=4)

    st.caption(
        f"Drift and volatility are the daily sample moments of the "
        f"{_int(sim.get('n_obs'), 0)} returns in this window -- mean "
        f"{fmt_ratio_pct(mu_daily, 3)} and standard deviation "
        f"{fmt_ratio_pct(sigma_daily, 3)} per day -- shown here annualised by "
        f"{TRADING_DAYS} and sqrt({TRADING_DAYS}). Only GBM and Student-t consume "
        f"them; both bootstraps resample the observed returns directly and never "
        f"see a fitted parameter."
    )


# ---------------------------------------------------------------------------
# 8. the honest part
# ---------------------------------------------------------------------------
def _caveats(sim: dict[str, Any], result: Any) -> None:
    ui_theme.section("WHAT THIS IS ASSUMING", "⚠")

    note = sim.get("note")
    if note:
        st.markdown(f"**The generator's own account of itself.** {_plain(note)}")

    n_obs = _int(sim.get("n_obs"), 0)
    horizon = _int(sim.get("horizon_days"), 0)
    block = _int(sim.get("block_size"), 1)
    method = str(sim.get("method", ""))

    # The comparison below describes all four generators, so the block-length
    # sentences have to stay grammatical on a run that used no blocks at all.
    in_blocks = method == "block_bootstrap" and block > 1
    block_phrase = (
        f"{block} sessions per block on this run"
        if in_blocks
        else "the block length is the control above; this run drew single days"
    )
    block_state = (
        f"`b` = {block} sessions on this run"
        if in_blocks
        else "`b` is the block length set above, unused by this run"
    )
    block_edge = (
        f"the first and last {block - 1} observations"
        if in_blocks
        else "the first and last `b - 1` observations"
    )

    st.markdown(
        f"""
**Every method on this page assumes the future is drawn from the same
distribution as the past.** That is an assumption, not a finding, and it is the
one that breaks first. What each choice buys and what it costs:

* **IID bootstrap** resamples the {n_obs} observed days one at a time with
  replacement, so the real skew and fat tails of this portfolio survive exactly.
  It cannot invent a crash bigger than any single day in the sample -- the worst
  day this book actually had is a hard floor on the simulated tail -- and drawing
  days independently destroys the ordering, so sustained drawdowns are
  understated.
* **Block bootstrap** resamples contiguous runs of days rather than single days
  ({block_phrase}), so some volatility clustering and short-run momentum survive
  inside each block. The dependence is still severed at every join, and the same
  hard floor on the tail applies.
* **GBM** assumes log-returns are independent and normally distributed. Markets
  are neither. A normal tail systematically understates the probability of
  extreme losses, which is the specific failure that matters here: the model is
  most wrong exactly where you most want it to be right.
* **Student-t** replaces the normal shock with a fat-tailed one, which produces
  the extreme days GBM cannot. But it is symmetric and equity returns are not, so
  it fattens the upside as much as the crash, and it is still independent draws:
  fat tails with no clustering.

**None of the four models a regime change.** No rate cycle, no war, no
devaluation, no liquidity event, no change in what these companies do. There is
no rebalancing, no contribution or withdrawal, no dividend, no brokerage, no
tax and no inflation. The portfolio is held frozen at today's weights for
{horizon} sessions.
        """
    )

    st.caption(
        f"Read this as one sentence: **this is {n_obs} days of history propagated "
        f"forward under {_METHOD_PROSE.get(method, 'one stated model')}, and "
        f"nothing else.** It describes the sample, not the future. If the last "
        f"{result.n_days} sessions were an unusually calm or unusually violent "
        f"stretch, every number on this page inherits that and none of them warns "
        f"you about it. It is not a forecast, not a projection of your wealth, and "
        f"not advice."
    )

    ui.methodology(
        "How each path is actually generated, and why the seed is on screen",
        f"""
**GBM -- and the drift correction that is not optional.**
`S_(t+1) = S_t * exp((mu - sigma^2 / 2) + sigma * z)` with `z ~ N(0, 1)`, where
`mu` and `sigma` are the daily mean and sample standard deviation of the
{n_obs} observed returns. The `- sigma^2 / 2` term is the Ito convexity
correction. Without it, simulating `exp(mu + sigma * z)` would give an average
step of `exp(mu + sigma^2 / 2)`, so the entire fan would drift upward for no
economic reason at all -- a pure artefact of `exp()` being convex. With it,
`E[S_T] = S_0 * exp(mu * T)` while `median[S_T] = S_0 * exp((mu - sigma^2 / 2) * T)`,
and the gap between those two lines is the volatility drag, about
{fmt_ratio_pct(_num(sim.get("sigma_daily")) ** 2 / 2.0 * TRADING_DAYS)} a year at
this portfolio's volatility.

**Bootstrap.** Draw {horizon} day-indices uniformly with replacement from the
{n_obs} observed daily returns and compound them:
`S_t = S_(t-1) * (1 + r_drawn)`. Nothing is fitted and nothing is assumed about
the shape of the distribution -- the empirical distribution *is* the model, so
skew and excess kurtosis come through unaltered. The moving-block version draws
contiguous runs of `b` days instead of single days and lays them end to end
({block_state}). Blocks are non-circular, which slightly under-weights
{block_edge} of the window.

**Student-t.** GBM's normal shock replaced by a t-distribution standardised to
unit variance, `z = t_df / sqrt(df / (df - 2))`, so `sigma` still means what it
says while the tails get heavier. Degrees of freedom are estimated from the
sample excess kurtosis by `df = 4 + 6 / k` and clamped to
[{fmt_num(mc.MIN_STUDENT_T_DF, 1)}, {fmt_num(mc.MAX_STUDENT_T_DF, 0)}]; below
{fmt_num(mc.MIN_STUDENT_T_DF, 1)} the variance is infinite and "volatility"
stops meaning anything.

**Why the seed is exposed.** A simulation whose output cannot be reproduced is an
anecdote. Fixing the pseudo-random seed makes this page deterministic: the same
five settings redraw the same {_int(sim.get("n_paths"), 0):,} paths on any machine,
on any day, so a number quoted from here can be *checked* rather than trusted.
That also makes the seed a diagnostic -- move it from {_int(sim.get("seed"), 0)}
to {_int(sim.get("seed"), 0) + 1} and re-read the ladder. Anything that moves
materially was never a result; it was Monte Carlo noise, and the honest response
is more paths, not a better story.
        """,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    ui_theme.page_header(
        "Monte Carlo",
        "Propagating this window's return distribution forward -- and stating "
        "exactly what that assumes",
        icon="🎲",
    )

    result = guards.require_result()
    if result is None:
        return

    if not guards.require_observations(
        result,
        minimum=MIN_OBSERVATIONS + 1,
        what="Forward simulation",
    ):
        return

    cfg = result.config

    controls = _controls(result, cfg)
    sim = _simulate(result, controls)

    if not sim.get("ok"):
        _failure(sim, controls, result)
        return

    _run_chips(sim, result)
    _fan(sim, cfg)

    ui_theme.spacer(0.4)
    _outcome_kpis(sim, cfg)

    ui_theme.spacer(0.4)
    _shape(sim, cfg)

    ui_theme.spacer(0.4)
    _ladder(sim, cfg)

    ui_theme.spacer(0.4)
    _parameters(sim, cfg)

    ui_theme.spacer(0.4)
    _caveats(sim, result)
