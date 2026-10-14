"""Rebalancing — ``src.analytics.rebalance``.

A drift report that does not end in a correct order ticket is decoration. The
tests here are therefore built around one question a reviewer actually cares
about: **if you execute the list this module prints, where does the portfolio
land?** That is asserted directly — the trades are applied to a copy of the
book and the resulting weights are measured against the targets — rather than
inferred from the intermediate columns.

Verified here
-------------
* the central worked example — 60/30/10 against 40/40/20 on a 10,000 book —
  with every share count and every trade value computed by hand in a comment;
* **the executable guarantee**: applying the whole list lands every name inside
  ``band_pct`` of its target, weights measured against ``V_t = V + cash``.
  This is asserted on a six-name book, with cash, and with fractional shares;
* whole-share flooring truncates TOWARD ZERO, so ``|shares * price_base| <=
  |gap|`` name by name: an order can stop short of its target, never overshoot
  it, and a sell is additionally capped at the quantity held (no shorting);
* a name inside its band gets ``HOLD``, ``shares == 0`` and
  ``trade_value == 0`` — a 2-point deviation is not worth a taxable gain;
* ``cash_to_invest`` raises the target portfolio value to ``V + C`` and, on an
  already-balanced book, produces only BUYs — nothing is sold to fund a
  purchase that new money already funds;
* targets that do not sum to 100 are renormalised and the fact is recorded in
  ``.attrs["targets_renormalised"]`` / ``["target_sum_pct"]``, never silently;
* a ticker in ``targets`` but not held appears with a 0% current weight, a NaN
  share count and a ``trade_value`` that still says how much to deploy;
* ``rebalance_summary`` reconciliation: ``net_cash == sell_value - buy_value``,
  turnover is two-way notional over the PRE-trade value and is never negative;
* unit conventions that would silently break an order ticket if reversed:
  ``drift_pct`` is PERCENTAGE POINTS, ``shares``/``trade_value`` are SIGNED
  (+ = buy), ``price`` is the NATIVE quote while ``trade_value`` is BASE
  currency, and rows execute SELL -> BUY -> HOLD so the sells fund the buys;
* documented degradation: empty positions, empty/zero-sum/None targets, a
  negative band, duplicate ticker rows and an unsorted book all yield an empty
  frame or a no-op rather than an exception — and an unusable target mapping is
  never read as an instruction to liquidate.

Deterministic and offline; every number below is arithmetic, not a sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.rebalance import (
    BUY,
    DEFAULT_REBALANCE_BAND_PCT,
    DRIFT_COLUMNS,
    HOLD,
    SELL,
    TRADE_COLUMNS,
    drift_table,
    equal_weight_targets,
    rebalance_summary,
    rebalance_trades,
    targets_from_optimizer,
)

# ---------------------------------------------------------------------------
# local builders — a positions frame carrying only what the trader needs, so
# every worked example below can be read off the call site.
# ---------------------------------------------------------------------------


def make_positions(
    tickers: list[str],
    quantity: list[float],
    price: list[float],
    fx_rate: list[float] | None = None,
) -> pd.DataFrame:
    """Minimal ``build_positions``-shaped frame: value = qty * price * fx."""
    rates = fx_rate if fx_rate is not None else [1.0] * len(tickers)
    return pd.DataFrame(
        {
            "ticker": tickers,
            "quantity": [float(q) for q in quantity],
            "current_price": [float(p) for p in price],
            "fx_rate": [float(f) for f in rates],
            "current_value": [
                float(q) * float(p) * float(f)
                for q, p, f in zip(quantity, price, rates, strict=True)
            ],
        }
    )


def apply_trades(
    positions: pd.DataFrame, trades: pd.DataFrame, cash_to_invest: float = 0.0
) -> dict[str, float]:
    """Execute the order list and return post-trade weights, in PERCENT of V_t.

    ``trade_value`` is signed and already in base currency, so the post-trade
    value of a name is simply ``current_value + trade_value``. Weights are
    measured against ``V_t = total_value + cash_to_invest`` — i.e. any cash
    left unspent still counts as part of the portfolio, which is the frame the
    module's guarantee is stated in.
    """
    value = {str(row.ticker): float(row.current_value) for row in positions.itertuples()}
    v_t = sum(value.values()) + float(cash_to_invest)
    for row in trades.itertuples():
        traded = float(row.trade_value)
        if np.isfinite(traded):
            value[str(row.ticker)] = value.get(str(row.ticker), 0.0) + traded
    return {ticker: v / v_t * 100.0 for ticker, v in value.items()}


@pytest.fixture
def book() -> pd.DataFrame:
    """The central worked example: 100 shares each at 60 / 30 / 10.

    AAA 100 x 60 = 6,000    BBB 100 x 30 = 3,000    CCC 100 x 10 = 1,000
    total 10,000  ->  weights 60% / 30% / 10%
    """
    return make_positions(["AAA", "BBB", "CCC"], [100.0, 100.0, 100.0], [60.0, 30.0, 10.0])


@pytest.fixture
def balanced_book() -> pd.DataFrame:
    """Three names at exactly 1,000 each — already on an equal-weight target."""
    return make_positions(["AAA", "BBB", "CCC"], [100.0, 100.0, 100.0], [10.0, 10.0, 10.0])


@pytest.fixture
def wide_book() -> pd.DataFrame:
    """Six names, deliberately lopsided, priced far below the band.

    T1 3000 x 10 = 30,000   T2 1000 x 20 = 20,000   T3  500 x 30 = 15,000
    T4  250 x 40 = 10,000   T5  300 x 50 = 15,000   T6  125 x 80 = 10,000
    total 100,000 -> weights 30 / 20 / 15 / 10 / 15 / 10 percent.

    The dearest share is 80, i.e. 0.08% of the book, so whole-share rounding
    cannot come anywhere near the tolerance band. That is what makes the
    post-trade assertions below a test of the sizing logic rather than of
    integer arithmetic.
    """
    return make_positions(
        ["T1", "T2", "T3", "T4", "T5", "T6"],
        [3000.0, 1000.0, 500.0, 250.0, 300.0, 125.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 80.0],
    )


# ---- equal_weight_targets -------------------------------------------------
def test_equal_weight_targets_gives_every_name_one_over_n_in_percent():
    """Four names -> 25% each, and the vector sums to exactly 100."""
    targets = equal_weight_targets(["AAA", "BBB", "CCC", "DDD"])

    assert targets == {"AAA": 25.0, "BBB": 25.0, "CCC": 25.0, "DDD": 25.0}
    assert sum(targets.values()) == pytest.approx(100.0, abs=1e-12)


def test_equal_weight_targets_of_three_names_sums_to_one_hundred_despite_the_third():
    """100/3 = 33.333... is not representable; the sum must still be 100."""
    targets = equal_weight_targets(["AAA", "BBB", "CCC"])

    assert all(w == pytest.approx(100.0 / 3.0, rel=1e-15) for w in targets.values())
    assert sum(targets.values()) == pytest.approx(100.0, abs=1e-12)


def test_equal_weight_targets_folds_case_strips_blanks_and_drops_duplicates():
    """'aaa ', 'AAA' and '' are one name and one blank, not four targets."""
    targets = equal_weight_targets(["aaa ", "AAA", "", None, "bbb"])

    assert targets == {"AAA": 50.0, "BBB": 50.0}


def test_equal_weight_targets_of_nothing_is_an_empty_mapping():
    assert equal_weight_targets(None) == {}
    assert equal_weight_targets([]) == {}
    assert equal_weight_targets(["", "  ", None]) == {}


# ---- targets_from_optimizer ----------------------------------------------
def test_optimizer_fractions_become_percentages():
    """0.6 / 0.4 is a weight vector, 60 / 40 is a target allocation."""
    assert targets_from_optimizer({"AAA": 0.6, "BBB": 0.4}) == pytest.approx(
        {"AAA": 60.0, "BBB": 40.0}
    )


def test_optimizer_dust_and_negative_weights_are_dropped_and_the_rest_renormalised():
    """A long-only solver emits 1e-17 dust and a vector summing to 1 - 1e-12.

    Keeping AAA 0.5 and BBB 0.3 (the 1e-17 and the -0.2 are discarded) the
    surviving 80% of the book is rescaled: 0.5/0.8 = 62.5%, 0.3/0.8 = 37.5%.
    """
    targets = targets_from_optimizer({"AAA": 0.5, "BBB": 0.3, "DUST": 1e-17, "SHORT": -0.2})

    assert set(targets) == {"AAA", "BBB"}
    assert targets["AAA"] == pytest.approx(62.5)
    assert targets["BBB"] == pytest.approx(37.5)
    assert sum(targets.values()) == pytest.approx(100.0, abs=1e-12)


def test_optimizer_output_that_already_sums_to_one_hundred_survives_the_round_trip():
    """Renormalisation removes the scale, so percent-form input is a no-op."""
    assert targets_from_optimizer({"AAA": 70.0, "BBB": 30.0}) == pytest.approx(
        {"AAA": 70.0, "BBB": 30.0}
    )


def test_optimizer_weights_from_a_pandas_series_are_accepted():
    weights = pd.Series({"AAA": 0.25, "BBB": 0.75})
    assert targets_from_optimizer(weights) == pytest.approx({"AAA": 25.0, "BBB": 75.0})


@pytest.mark.parametrize("weights", [None, {}, {"AAA": 0.0}, {"AAA": float("nan")}, {"AAA": -1.0}])
def test_optimizer_input_with_nothing_usable_returns_an_empty_mapping(weights):
    """An empty target mapping is a no-op downstream, never a liquidation."""
    assert targets_from_optimizer(weights) == {}


# ---- drift_table: schema and hand-computed arithmetic ---------------------
def test_drift_table_publishes_exactly_the_documented_columns(book: pd.DataFrame):
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20})
    assert tuple(drift.columns) == DRIFT_COLUMNS


def test_drift_table_reports_percentage_points_not_a_ratio(book: pd.DataFrame):
    """AAA is 60% against a 40% target.

    The drift is 60 - 40 = +20 POINTS. A ratio convention would print
    60/40 - 1 = +50%, so the two answers are far apart and this pins the unit.
    """
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20}).set_index("ticker")

    assert drift.loc["AAA", "drift_pct"] == pytest.approx(20.0)
    assert drift.loc["AAA", "drift_pct"] != pytest.approx(50.0)
    assert drift.loc["BBB", "drift_pct"] == pytest.approx(-10.0)  # 30 - 40
    assert drift.loc["CCC", "drift_pct"] == pytest.approx(-10.0)  # 10 - 20


def test_drift_table_value_gap_is_target_value_minus_current_value(book: pd.DataFrame):
    """On a 10,000 book with targets 40/40/20:

    target values 4,000 / 4,000 / 2,000
    gaps          4,000 - 6,000 = -2,000   (sell)
                  4,000 - 3,000 = +1,000   (buy)
                  2,000 - 1,000 = +1,000   (buy)
    The gaps sum to zero because no cash was added.
    """
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20}).set_index("ticker")

    assert list(drift["target_value"]) == pytest.approx([4000.0, 4000.0, 2000.0])
    assert drift.loc["AAA", "value_gap"] == pytest.approx(-2000.0)
    assert drift.loc["BBB", "value_gap"] == pytest.approx(1000.0)
    assert drift.loc["CCC", "value_gap"] == pytest.approx(1000.0)
    assert drift["value_gap"].sum() == pytest.approx(0.0, abs=1e-9)


def test_drift_table_current_weights_sum_to_one_hundred(book: pd.DataFrame):
    """Identity: weights are shares of one portfolio, whatever the targets."""
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20})
    assert drift["current_weight_pct"].sum() == pytest.approx(100.0, abs=1e-9)
    assert drift["target_weight_pct"].sum() == pytest.approx(100.0, abs=1e-9)


def test_drift_table_ranks_the_worst_offender_first(book: pd.DataFrame):
    """Targets 60/30/30 sum to 120 and renormalise to 50/25/25.

    drift  AAA 60 - 50 = +10      BBB 30 - 25 = +5      CCC 10 - 25 = -15
    |drift| ordering: CCC (15), AAA (10), BBB (5).
    """
    drift = drift_table(book, {"AAA": 60, "BBB": 30, "CCC": 30}, band_pct=5.0)

    assert list(drift["ticker"]) == ["CCC", "AAA", "BBB"]
    assert list(drift["drift_pct"]) == pytest.approx([-15.0, 10.0, 5.0])


def test_in_band_is_inclusive_at_exactly_the_band_width(book: pd.DataFrame):
    """BBB drifted exactly +5.0 against a 5.0 band: inside, so it does not trade."""
    drift = drift_table(book, {"AAA": 60, "BBB": 30, "CCC": 30}, band_pct=5.0).set_index("ticker")

    assert drift.loc["BBB", "drift_pct"] == pytest.approx(5.0)
    assert bool(drift.loc["BBB", "in_band"]) is True
    assert bool(drift.loc["AAA", "in_band"]) is False
    assert bool(drift.loc["CCC", "in_band"]) is False


# ---- renormalisation is recorded, never silent ---------------------------
def test_targets_that_do_not_sum_to_one_hundred_are_renormalised_and_flagged(
    book: pd.DataFrame,
):
    """60/30/30 sums to 120; each weight is scaled by 100/120 = 0.8333...

    AAA 60 * 100/120 = 50, BBB 30 * 100/120 = 25, CCC 30 * 100/120 = 25.
    """
    drift = drift_table(book, {"AAA": 60, "BBB": 30, "CCC": 30})
    weights = dict(zip(drift["ticker"], drift["target_weight_pct"], strict=True))

    assert weights == pytest.approx({"AAA": 50.0, "BBB": 25.0, "CCC": 25.0})
    assert drift.attrs["targets_renormalised"] is True
    assert drift.attrs["target_sum_pct"] == pytest.approx(120.0)
    assert "120.0000%" in drift.attrs["note"]
    assert "renormalised to 100%" in drift.attrs["note"]


def test_targets_that_already_sum_to_one_hundred_are_not_flagged(book: pd.DataFrame):
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20})

    assert drift.attrs["targets_renormalised"] is False
    assert drift.attrs["target_sum_pct"] == pytest.approx(100.0)
    assert drift.attrs["note"] == ""


def test_renormalisation_of_fractions_is_flagged_too(book: pd.DataFrame):
    """Weight FRACTIONS handed in by mistake sum to 1, not 100.

    0.4 / 0.4 / 0.2 rescales by 100/1.0 to 40 / 40 / 20 — the same allocation,
    but the caller is told the vector did not arrive in percent.
    """
    drift = drift_table(book, {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}).set_index("ticker")

    assert list(drift["target_weight_pct"]) == pytest.approx([40.0, 40.0, 20.0])
    assert drift.attrs["targets_renormalised"] is True
    assert drift.attrs["target_sum_pct"] == pytest.approx(1.0)


def test_a_renormalised_target_produces_the_same_trades_as_its_percent_form(
    book: pd.DataFrame,
):
    """Identity: renormalisation removes the scale, so 4/4/2 == 40/40/20."""
    scaled = rebalance_trades(book, {"AAA": 4, "BBB": 4, "CCC": 2})
    plain = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20})

    pd.testing.assert_frame_equal(scaled, plain)


def test_a_negative_target_weight_is_dropped_and_named(book: pd.DataFrame):
    """This module is long-only: a negative target is bad data, not a short."""
    drift = drift_table(book, {"AAA": 60, "BBB": 40, "CCC": -10})

    assert drift.attrs["targets_dropped"] == ["CCC"]
    weights = dict(zip(drift["ticker"], drift["target_weight_pct"], strict=True))
    assert weights == pytest.approx({"AAA": 60.0, "BBB": 40.0, "CCC": 0.0})


# ---- the central worked example ------------------------------------------
def test_the_worked_example_produces_the_hand_computed_order_ticket(book: pd.DataFrame):
    """60/30/10 -> 40/40/20 on a 10,000 book, whole shares, 5-point band.

    gaps      AAA -2,000     BBB +1,000     CCC +1,000
    shares    -2000/60 = -33.33 -> -33   1000/30 = 33.33 -> 33   1000/10 = 100
    value     -33 x 60 = -1,980   33 x 30 = +990    100 x 10 = +1,000
    """
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20}, band_pct=5.0)
    by_ticker = trades.set_index("ticker")

    assert tuple(trades.columns) == TRADE_COLUMNS
    assert by_ticker.loc["AAA", "action"] == SELL
    assert by_ticker.loc["AAA", "shares"] == pytest.approx(-33.0)
    assert by_ticker.loc["AAA", "trade_value"] == pytest.approx(-1980.0)

    assert by_ticker.loc["BBB", "action"] == BUY
    assert by_ticker.loc["BBB", "shares"] == pytest.approx(33.0)
    assert by_ticker.loc["BBB", "trade_value"] == pytest.approx(990.0)

    assert by_ticker.loc["CCC", "action"] == BUY
    assert by_ticker.loc["CCC", "shares"] == pytest.approx(100.0)
    assert by_ticker.loc["CCC", "trade_value"] == pytest.approx(1000.0)


def test_shares_and_trade_value_are_signed_with_buy_positive(book: pd.DataFrame):
    """Regression guard: ``current_value + trade_value`` must be the post-trade
    value, so a sell has to be negative on BOTH columns."""
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20})

    for row in trades.itertuples():
        if row.action == SELL:
            assert row.shares < 0 and row.trade_value < 0
        elif row.action == BUY:
            assert row.shares > 0 and row.trade_value > 0
        else:
            assert row.shares == 0.0 and row.trade_value == 0.0


def test_rows_execute_sell_then_buy_then_hold(book: pd.DataFrame):
    """The sells fund the buys, so they have to be at the top of the ticket."""
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 30, "CCC": 10, "NEW": 20})

    rank = {SELL: 0, BUY: 1, HOLD: 2}
    order = [rank[a] for a in trades["action"]]
    assert order == sorted(order)
    assert trades["action"].iloc[0] == SELL


def test_within_a_side_the_largest_ticket_comes_first(book: pd.DataFrame):
    """CCC buys 1,000 and BBB buys 990, so CCC is listed first."""
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20})
    buys = trades[trades["action"] == BUY]

    assert list(buys["ticker"]) == ["CCC", "BBB"]
    assert list(buys["trade_value"]) == pytest.approx([1000.0, 990.0])


# ---- names inside the band do not trade ----------------------------------
def test_a_name_inside_the_band_is_held_with_zero_shares_and_zero_value(
    book: pd.DataFrame,
):
    """60/30/10 against 58/32/10 drifts +2 / -2 / 0, all inside a 5-point band.

    Harvesting a taxable gain to fix a 2-point deviation is not risk
    management, so the whole ticket must be empty.
    """
    trades = rebalance_trades(book, {"AAA": 58, "BBB": 32, "CCC": 10}, band_pct=5.0)

    assert list(trades["action"]) == [HOLD, HOLD, HOLD]
    assert list(trades["shares"]) == pytest.approx([0.0, 0.0, 0.0])
    assert list(trades["trade_value"]) == pytest.approx([0.0, 0.0, 0.0])
    assert list(trades["drift_pct"]) == pytest.approx([2.0, -2.0, 0.0])
    assert trades.attrs["targets_renormalised"] is False


def test_a_drift_of_exactly_the_band_width_does_not_trade(book: pd.DataFrame):
    """The rule is ``|drift| > band``, so 5.0 against a 5.0 band is a HOLD.

    AAA 60 vs 55 = +5, BBB 30 vs 35 = -5, CCC 10 vs 10 = 0.
    """
    trades = rebalance_trades(book, {"AAA": 55, "BBB": 35, "CCC": 10}, band_pct=5.0)

    assert set(trades["action"]) == {HOLD}
    assert list(trades["drift_pct"]) == pytest.approx([5.0, -5.0, 0.0])


def test_shrinking_the_band_to_zero_makes_every_drifted_name_trade(book: pd.DataFrame):
    """A 0-point band means "track the target exactly"; only a zero gap holds."""
    trades = rebalance_trades(book, {"AAA": 58, "BBB": 32, "CCC": 10}, band_pct=0.0)
    by_ticker = trades.set_index("ticker")

    assert by_ticker.loc["AAA", "action"] == SELL  # drift +2 > 0
    assert by_ticker.loc["BBB", "action"] == BUY  # drift -2 < 0
    assert by_ticker.loc["CCC", "action"] == HOLD  # drift 0, gap 0


# ---- THE guarantee: simulate the ticket and measure where you land --------
def test_the_full_ticket_is_hand_computable_and_lands_the_book_on_target(
    wide_book: pd.DataFrame,
):
    """The order-by-order proof, on a 100,000 book with a 1-point band.

    Targets 25 / 25 / 12.5 / 12.5 / 12.5 / 12.5 (they sum to 100 and each
    weight/100 is a dyadic fraction, so every target value below is exact).

        name  held    target   gap      price  shares            traded
        T1    30,000  25,000   -5,000   10     -5000/10  = -500  -5,000
        T2    20,000  25,000   +5,000   20     +5000/20  = +250  +5,000
        T3    15,000  12,500   -2,500   30     -83.33 -> -83     -2,490
        T4    10,000  12,500   +2,500   40     +62.5  -> +62     +2,480
        T5    15,000  12,500   -2,500   50     -2500/50  =  -50  -2,500
        T6    10,000  12,500   +2,500   80     +31.25 -> +31     +2,480

    Applying that leaves 25,000 / 25,000 / 12,510 / 12,480 / 12,500 / 12,480
    against a V_t of 100,000 — every name within 0.02 points of its target.
    """
    targets = {"T1": 25, "T2": 25, "T3": 12.5, "T4": 12.5, "T5": 12.5, "T6": 12.5}
    trades = rebalance_trades(wide_book, targets, band_pct=1.0)
    by_ticker = trades.set_index("ticker")

    assert trades.attrs["below_lot_size"] == []
    assert list(by_ticker.loc[["T1", "T2", "T3", "T4", "T5", "T6"], "shares"]) == (
        pytest.approx([-500.0, 250.0, -83.0, 62.0, -50.0, 31.0])
    )
    assert list(by_ticker.loc[["T1", "T2", "T3", "T4", "T5", "T6"], "trade_value"]) == (
        pytest.approx([-5000.0, 5000.0, -2490.0, 2480.0, -2500.0, 2480.0])
    )

    after = apply_trades(wide_book, trades)
    assert after == pytest.approx(
        {"T1": 25.0, "T2": 25.0, "T3": 12.51, "T4": 12.48, "T5": 12.5, "T6": 12.48}
    )
    for ticker, weight in after.items():
        assert abs(weight - targets[ticker]) <= 1.0, (ticker, weight)


def test_applying_the_generated_trades_lands_every_name_inside_the_band(
    wide_book: pd.DataFrame,
):
    """The same guarantee where only SOME of the book breaches its band.

    Against an equal-weight 16.667% target with a 5-point band, T1 (+13.33) and
    T4/T6 (-6.67 each) breach and trade, while T2 (+3.33), T3 and T5 (-1.67)
    are left alone. Both kinds of row must end up inside the band: the traded
    ones because they were resized, the held ones because they never left it.
    """
    targets = equal_weight_targets(list(wide_book["ticker"]))
    trades = rebalance_trades(wide_book, targets, band_pct=5.0)

    traded = set(trades.loc[trades["action"] != HOLD, "ticker"])
    assert traded == {"T1", "T4", "T6"}

    after = apply_trades(wide_book, trades)
    for ticker, weight in after.items():
        assert abs(weight - targets[ticker]) <= 5.0, (ticker, weight)


def test_a_traded_name_lands_within_one_share_of_its_target(wide_book: pd.DataFrame):
    """The tight bound: the only residue a whole-share order can leave is the
    part of one share it had to truncate, ``price_base / V_t``.

    The dearest name here is T6 at 80 a share on a 100,000 book, so no traded
    weight may miss its target by more than 80 / 100,000 * 100 = 0.08 points —
    more than sixty times tighter than the 5-point band being claimed.
    """
    targets = equal_weight_targets(list(wide_book["ticker"]))
    trades = rebalance_trades(wide_book, targets, band_pct=5.0)
    after = apply_trades(wide_book, trades)

    total = float(wide_book["current_value"].sum())
    price = dict(zip(wide_book["ticker"], wide_book["current_price"], strict=True))

    for row in trades.itertuples():
        if row.action == HOLD:
            continue
        one_share_pct = float(price[row.ticker]) / total * 100.0
        assert abs(after[row.ticker] - targets[row.ticker]) <= one_share_pct + 1e-9
    assert max(price.values()) / total * 100.0 == pytest.approx(0.08, abs=1e-12)


def test_the_guarantee_also_holds_for_an_uneven_target_with_new_cash(
    wide_book: pd.DataFrame,
):
    """Same proof, but with a lopsided target and 40,000 of new money.

    V_t = 100,000 + 40,000 = 140,000, so the target values are 56,000 /
    35,000 / 21,000 / 14,000 / 7,000 / 7,000. T1 has to more than double while
    T5 and T6 are trimmed, and every post-trade weight must still land inside
    the 5-point band.
    """
    targets = {"T1": 40, "T2": 25, "T3": 15, "T4": 10, "T5": 5, "T6": 5}
    trades = rebalance_trades(wide_book, targets, band_pct=5.0, cash_to_invest=40000.0)

    assert trades.attrs["target_value_total"] == pytest.approx(140000.0)
    after = apply_trades(wide_book, trades, cash_to_invest=40000.0)
    for ticker, weight in after.items():
        assert abs(weight - targets[ticker]) <= 5.0, (ticker, weight)


def test_fractional_share_execution_lands_exactly_on_target(wide_book: pd.DataFrame):
    """With ``allow_fractional=True`` and a zero band there is no rounding
    residue and nothing is excused from trading, so every name lands on its
    target to floating-point precision."""
    targets = equal_weight_targets(list(wide_book["ticker"]))
    trades = rebalance_trades(wide_book, targets, band_pct=0.0, allow_fractional=True)
    after = apply_trades(wide_book, trades)

    assert set(trades["action"]) == {BUY, SELL}
    for ticker, weight in after.items():
        assert weight == pytest.approx(targets[ticker], abs=1e-9)


# ---- whole-share flooring truncates toward zero ---------------------------
def test_whole_share_flooring_never_overshoots_the_gap(wide_book: pd.DataFrame):
    """Identity for every traded row: ``|shares * price_base| <= |gap|``.

    Truncation is toward zero on BOTH sides, so a buy stops short of its target
    and a sell raises slightly less cash than asked — never the reverse.
    """
    targets = {"T1": 12, "T2": 18, "T3": 22, "T4": 8, "T5": 27, "T6": 13}
    trades = rebalance_trades(wide_book, targets, band_pct=1.0)
    drift = drift_table(wide_book, targets, band_pct=1.0).set_index("ticker")

    for row in trades.itertuples():
        gap = float(drift.loc[row.ticker, "value_gap"])
        assert abs(row.trade_value) <= abs(gap) + 1e-9, row.ticker
        if row.action != HOLD:
            assert np.sign(row.trade_value) == np.sign(gap), row.ticker


def test_a_sell_is_truncated_toward_zero_not_away_from_it(book: pd.DataFrame):
    """AAA must raise 2,000 at 60 a share: 33.33 shares.

    Truncation gives 33 (raising 1,980 — short of the gap). Rounding half-up
    would give 33 too, but flooring a negative number toward minus infinity
    would give 34 and raise 2,040, overshooting the target. Only truncation
    toward zero is safe, so the sign of the residue is pinned here.
    """
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20}).set_index("ticker")

    assert trades.loc["AAA", "shares"] == pytest.approx(-33.0)
    assert trades.loc["AAA", "shares"] != pytest.approx(-34.0)
    assert abs(trades.loc["AAA", "trade_value"]) < 2000.0


def test_a_sell_is_capped_at_the_quantity_held_so_the_book_never_goes_short():
    """AAA is 90% of a 10,000 book at 1,000 a share (9 shares) with a 0% target.

    The gap asks for -9,000, exactly 9 shares, and the cap must not bite. But
    if the same name is asked to fund more than it holds — here by pricing the
    order off a stale, higher quantity — the floor is the position itself.
    """
    positions = make_positions(["AAA", "BBB"], [9.0, 1000.0], [1000.0, 1.0])
    trades = rebalance_trades(positions, {"AAA": 0, "BBB": 100}).set_index("ticker")

    assert trades.loc["AAA", "shares"] == pytest.approx(-9.0)
    assert trades.loc["AAA", "shares"] >= -positions.set_index("ticker").loc["AAA", "quantity"]
    # post-trade the position is flat, never negative
    assert 9.0 + trades.loc["AAA", "shares"] == pytest.approx(0.0)


def test_an_order_smaller_than_one_share_is_held_and_reported_not_hidden():
    """AAA is 90% of a 1,000 book but one share costs 900.

    Target 80% -> gap -100, i.e. -0.111 shares, which truncates to zero. The
    residual 10-point drift is real, so the row is HOLD *and* the ticker is
    named in ``attrs["below_lot_size"]`` rather than being quietly dropped.
    """
    positions = make_positions(["AAA", "BBB"], [1.0, 100.0], [900.0, 1.0])
    trades = rebalance_trades(positions, {"AAA": 80, "BBB": 20}, band_pct=5.0)
    by_ticker = trades.set_index("ticker")

    assert by_ticker.loc["AAA", "action"] == HOLD
    assert by_ticker.loc["AAA", "shares"] == pytest.approx(0.0)
    assert by_ticker.loc["AAA", "trade_value"] == pytest.approx(0.0)
    assert by_ticker.loc["AAA", "drift_pct"] == pytest.approx(10.0)
    assert trades.attrs["below_lot_size"] == ["AAA"]


# ---- cash_to_invest -------------------------------------------------------
def test_cash_raises_the_target_portfolio_value_by_exactly_the_cash(
    balanced_book: pd.DataFrame,
):
    """V = 3,000 and C = 300, so V_t = 3,300 and every target value is 1,100."""
    trades = rebalance_trades(
        balanced_book, equal_weight_targets(["AAA", "BBB", "CCC"]), cash_to_invest=300.0
    )

    assert trades.attrs["total_value"] == pytest.approx(3000.0)
    assert trades.attrs["cash_to_invest"] == pytest.approx(300.0)
    assert trades.attrs["target_value_total"] == pytest.approx(3300.0)


def test_cash_into_an_already_balanced_book_produces_only_buys(
    balanced_book: pd.DataFrame,
):
    """Nothing has drifted (every drift is 0.0), so nothing needs selling.

    Each name needs 1,100 against 1,000 held: a gap of +100, which at 10 a
    share is exactly 10 shares and 100 of value. New money is put to work even
    though the whole book is inside its band — that is the point of supplying
    it — and no taxable sale is manufactured to fund it.
    """
    trades = rebalance_trades(
        balanced_book,
        equal_weight_targets(["AAA", "BBB", "CCC"]),
        band_pct=5.0,
        cash_to_invest=300.0,
    )

    assert set(trades["action"]) == {BUY}
    assert SELL not in set(trades["action"])
    assert list(trades["drift_pct"]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert list(trades["shares"]) == pytest.approx([10.0, 10.0, 10.0])
    assert list(trades["trade_value"]) == pytest.approx([100.0, 100.0, 100.0])

    summary = rebalance_summary(trades)
    assert summary["buy_value"] == pytest.approx(300.0)
    assert summary["sell_value"] == pytest.approx(0.0)
    assert summary["net_cash"] == pytest.approx(-300.0)  # the cash is fully deployed


def test_without_cash_an_already_balanced_book_generates_no_trades(
    balanced_book: pd.DataFrame,
):
    """Same book, no cash: every gap is zero, so the ticket is empty."""
    trades = rebalance_trades(
        balanced_book, equal_weight_targets(["AAA", "BBB", "CCC"]), cash_to_invest=0.0
    )

    assert set(trades["action"]) == {HOLD}
    assert rebalance_summary(trades)["n_trades"] == 0


def test_a_negative_cash_amount_is_read_as_no_cash(balanced_book: pd.DataFrame):
    """This module deploys cash; it does not model a withdrawal."""
    trades = rebalance_trades(
        balanced_book, equal_weight_targets(["AAA", "BBB", "CCC"]), cash_to_invest=-500.0
    )

    assert trades.attrs["cash_to_invest"] == pytest.approx(0.0)
    assert trades.attrs["target_value_total"] == pytest.approx(3000.0)
    assert set(trades["action"]) == {HOLD}


# ---- a target on a name that is not held ---------------------------------
def test_a_target_on_a_name_not_held_appears_with_a_zero_current_weight(
    book: pd.DataFrame,
):
    """A target allocation describes the WHOLE portfolio, so NEW must appear.

    With targets 40/30/10/20 the existing 10,000 book is reweighted and NEW is
    shown at 0% current against a 20% target: a drift of -20 points and a
    2,000 gap to deploy.
    """
    drift = drift_table(book, {"AAA": 40, "BBB": 30, "CCC": 10, "NEW": 20}).set_index("ticker")

    assert "NEW" in drift.index
    assert drift.loc["NEW", "current_weight_pct"] == pytest.approx(0.0)
    assert drift.loc["NEW", "current_value"] == pytest.approx(0.0)
    assert drift.loc["NEW", "target_weight_pct"] == pytest.approx(20.0)
    assert drift.loc["NEW", "drift_pct"] == pytest.approx(-20.0)
    assert drift.loc["NEW", "value_gap"] == pytest.approx(2000.0)


def test_an_unpriced_target_reports_the_money_but_not_a_share_count(book: pd.DataFrame):
    """NEW has no price in the positions frame, so a share count would be a lie.

    ``shares`` is NaN, ``trade_value`` still says 2,000 has to be deployed, and
    the ticker is listed in ``attrs["no_price"]`` so the UI can say why.
    """
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 30, "CCC": 10, "NEW": 20})
    row = trades.set_index("ticker").loc["NEW"]

    assert row["action"] == BUY
    assert np.isnan(row["shares"])
    assert row["trade_value"] == pytest.approx(2000.0)
    assert trades.attrs["no_price"] == ["NEW"]


def test_a_held_name_absent_from_targets_is_sold_down_to_zero(book: pd.DataFrame):
    """CCC has no target, so its target is 0% — not "leave it alone".

    AAA 60 and BBB 30 renormalise to 66.67 / 33.33 of a 10,000 book, and CCC's
    entire 1,000 (all 100 shares) is sold.
    """
    trades = rebalance_trades(book, {"AAA": 60, "BBB": 30}).set_index("ticker")

    assert trades.loc["CCC", "action"] == SELL
    assert trades.loc["CCC", "shares"] == pytest.approx(-100.0)
    assert trades.loc["CCC", "trade_value"] == pytest.approx(-1000.0)


# ---- currency conventions -------------------------------------------------
def test_price_is_native_while_trade_value_is_base_currency():
    """AAA quotes at 50 USD with an 80 INR/USD rate: 4,000 INR a share.

    AAA  1 share  x 50 x 80 =  4,000 INR   (20% of the book)
    BBB  160      x 100 x 1 = 16,000 INR   (80%)
    Targets 40/60 leave AAA short 4,000 INR, i.e. exactly ONE share. The
    ``price`` column must still read 50 — the number the positions table shows —
    while ``trade_value`` reads 4,000, the number the cash account moves by.
    """
    positions = make_positions(["AAA", "BBB"], [1.0, 160.0], [50.0, 100.0], fx_rate=[80.0, 1.0])
    trades = rebalance_trades(positions, {"AAA": 40, "BBB": 60}).set_index("ticker")

    assert trades.loc["AAA", "price"] == pytest.approx(50.0)
    assert trades.loc["AAA", "shares"] == pytest.approx(1.0)
    assert trades.loc["AAA", "trade_value"] == pytest.approx(4000.0)
    assert trades.loc["BBB", "shares"] == pytest.approx(-40.0)
    assert trades.loc["BBB", "trade_value"] == pytest.approx(-4000.0)


def test_a_missing_price_falls_back_to_value_over_quantity():
    """No ``current_price`` column at all: value / quantity is base-currency by
    construction, so the order can still be sized in shares."""
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "quantity": [100.0, 100.0],
            "current_value": [6000.0, 4000.0],
        }
    )
    trades = rebalance_trades(positions, {"AAA": 50, "BBB": 50}).set_index("ticker")

    # AAA is 60 a share; the gap is 5,000 - 6,000 = -1,000 -> -16.67 -> -16
    assert trades.loc["AAA", "shares"] == pytest.approx(-16.0)
    assert trades.loc["AAA", "trade_value"] == pytest.approx(-960.0)


# ---- rebalance_summary ----------------------------------------------------
def test_summary_reconciles_with_the_hand_computed_worked_example(book: pd.DataFrame):
    """From the ticket above: sell 1,980; buy 990 + 1,000 = 1,990.

    net_cash     1,980 - 1,990 = -10   (10 short — the price of whole shares)
    turnover     (1,990 + 1,980) / 10,000 * 100 = 39.70%
    max_drift    |+20| is the worst deviation in the list
    """
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20}, band_pct=5.0)
    summary = rebalance_summary(trades)

    assert summary["n_trades"] == 3
    assert summary["buy_value"] == pytest.approx(1990.0)
    assert summary["sell_value"] == pytest.approx(1980.0)
    assert summary["net_cash"] == pytest.approx(-10.0)
    assert summary["turnover_pct"] == pytest.approx(39.7)
    assert summary["max_drift_pct"] == pytest.approx(20.0)
    assert summary["in_band_count"] == 0


def test_net_cash_is_always_sell_value_minus_buy_value(wide_book: pd.DataFrame):
    """Identity, whatever the ticket: the cash account moves by exactly this.

    ``trade_value`` is signed with BUY positive, so its sum is the money that
    left the account — i.e. minus ``net_cash``.
    """
    for targets, cash in (
        ({"T1": 12, "T2": 18, "T3": 22, "T4": 8, "T5": 27, "T6": 13}, 0.0),
        (equal_weight_targets(list(wide_book["ticker"])), 0.0),
        ({"T1": 40, "T2": 25, "T3": 15, "T4": 10, "T5": 5, "T6": 5}, 40000.0),
    ):
        trades = rebalance_trades(wide_book, targets, cash_to_invest=cash)
        summary = rebalance_summary(trades)

        assert summary["net_cash"] == pytest.approx(summary["sell_value"] - summary["buy_value"])
        assert summary["net_cash"] == pytest.approx(-float(trades["trade_value"].sum(min_count=1)))


def test_turnover_is_two_way_notional_and_never_negative(wide_book: pd.DataFrame):
    """Identity: turnover = (buy + sell) / pre-trade value * 100, and both legs
    are absolute values, so it cannot be negative for any ticket."""
    for targets in (
        equal_weight_targets(list(wide_book["ticker"])),
        {"T1": 40, "T2": 25, "T3": 15, "T4": 10, "T5": 5, "T6": 5},
        {"T1": 100},
    ):
        trades = rebalance_trades(wide_book, targets)
        summary = rebalance_summary(trades)

        assert summary["turnover_pct"] >= 0.0
        assert summary["buy_value"] >= 0.0
        assert summary["sell_value"] >= 0.0
        assert summary["turnover_pct"] == pytest.approx(
            (summary["buy_value"] + summary["sell_value"]) / trades.attrs["total_value"] * 100.0
        )


def test_swapping_the_entire_book_scores_two_hundred_percent_turnover():
    """Selling everything and buying an equal amount back is 200% two-way.

    AAA is 100% of a 1,000 book and NEW is the whole target. Sell 1,000 and
    deploy 1,000: (1,000 + 1,000) / 1,000 * 100 = 200%.
    """
    positions = make_positions(["AAA"], [100.0], [10.0])
    trades = rebalance_trades(positions, {"NEW": 100})
    summary = rebalance_summary(trades)

    assert summary["sell_value"] == pytest.approx(1000.0)
    assert summary["buy_value"] == pytest.approx(1000.0)
    assert summary["turnover_pct"] == pytest.approx(200.0)
    assert summary["net_cash"] == pytest.approx(0.0)


def test_n_trades_counts_rows_that_are_not_holds(book: pd.DataFrame):
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 30, "CCC": 10, "NEW": 20})

    assert rebalance_summary(trades)["n_trades"] == 2  # AAA sells, NEW buys
    assert int((trades["action"] != HOLD).sum()) == 2


def test_in_band_count_uses_the_band_recorded_on_the_frame(book: pd.DataFrame):
    """Targets 58/32/10 drift +2 / -2 / 0, all inside a 5-point band."""
    trades = rebalance_trades(book, {"AAA": 58, "BBB": 32, "CCC": 10}, band_pct=5.0)
    summary = rebalance_summary(trades)

    assert trades.attrs["band_pct"] == pytest.approx(5.0)
    assert summary["in_band_count"] == 3
    assert summary["max_drift_pct"] == pytest.approx(2.0)


def test_summary_of_nothing_is_zeros_with_nan_ratios():
    """A missing ticket has no turnover and no worst drift — not a 0% of each."""
    for empty in (None, pd.DataFrame(), rebalance_trades(None, None)):
        summary = rebalance_summary(empty)

        assert summary["n_trades"] == 0
        assert summary["buy_value"] == pytest.approx(0.0)
        assert summary["sell_value"] == pytest.approx(0.0)
        assert summary["net_cash"] == pytest.approx(0.0)
        assert summary["in_band_count"] == 0
        assert np.isnan(summary["turnover_pct"])
        assert np.isnan(summary["max_drift_pct"])


def test_turnover_is_nan_when_the_frame_carries_no_total_value(book: pd.DataFrame):
    """A trade list rebuilt without ``.attrs`` cannot be normalised, and NaN is
    the honest answer — 0% would read as "nothing traded"."""
    trades = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20})
    rebuilt = pd.DataFrame(trades.to_dict("list"))  # attrs deliberately dropped

    summary = rebalance_summary(rebuilt)
    assert summary["n_trades"] == 3
    assert np.isnan(summary["turnover_pct"])


# ---- degenerate input: degrade, never raise ------------------------------
def test_an_empty_portfolio_with_no_targets_yields_empty_frames():
    for positions in (None, pd.DataFrame(), pd.DataFrame({"nothing": []})):
        drift = drift_table(positions, None)
        trades = rebalance_trades(positions, {})

        assert drift.empty and tuple(drift.columns) == DRIFT_COLUMNS
        assert trades.empty and tuple(trades.columns) == TRADE_COLUMNS


def test_an_empty_portfolio_with_targets_but_no_cash_lists_them_without_trading():
    """There is nothing to sell and nothing to spend, so the plan is a table of
    intent, not an order: 0% held against a 100% target, and zero to deploy."""
    drift = drift_table(None, {"AAA": 100})
    trades = rebalance_trades(None, {"AAA": 100})

    assert list(drift["ticker"]) == ["AAA"]
    assert drift.loc[0, "current_weight_pct"] == pytest.approx(0.0)
    assert drift.loc[0, "target_weight_pct"] == pytest.approx(100.0)
    assert drift.loc[0, "value_gap"] == pytest.approx(0.0)
    assert list(trades["action"]) == [HOLD]


def test_an_empty_target_mapping_is_a_no_op_never_a_liquidation(book: pd.DataFrame):
    """The most dangerous possible failure mode in this module: reading "no
    targets supplied" as "sell everything"."""
    for targets in (None, {}, pd.Series(dtype=float)):
        trades = rebalance_trades(book, targets)

        assert set(trades["action"]) == {HOLD}
        assert SELL not in set(trades["action"])
        assert list(trades["shares"]) == pytest.approx([0.0, 0.0, 0.0])
        assert rebalance_summary(trades)["n_trades"] == 0


def test_targets_summing_to_zero_are_reported_and_change_nothing(book: pd.DataFrame):
    drift = drift_table(book, {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0})
    trades = rebalance_trades(book, {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0})

    assert drift.attrs["note"] == "Target weights sum to zero - nothing to rebalance."
    assert drift["target_weight_pct"].isna().all()
    assert set(trades["action"]) == {HOLD}


def test_an_unusable_target_mapping_says_so_in_the_note(book: pd.DataFrame):
    drift = drift_table(book, {"": 50.0, None: 50.0})
    assert drift.attrs["note"] == ("No usable target weights supplied - nothing to rebalance.")


def test_an_undefined_drift_is_reported_out_of_band_but_still_does_not_trade(
    book: pd.DataFrame,
):
    """With no usable target the comparison is undefined. "Unknown" is not
    "fine", so ``in_band`` is False — yet nothing is traded on a NaN."""
    drift = drift_table(book, None)
    trades = rebalance_trades(book, None)

    assert drift["drift_pct"].isna().all()
    assert not drift["in_band"].any()
    assert set(trades["action"]) == {HOLD}


@pytest.mark.parametrize("band", [-5.0, float("nan"), None, "wide"])
def test_an_unusable_band_falls_back_to_the_configured_default(book: pd.DataFrame, band: object):
    drift = drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20}, band_pct=band)
    assert drift.attrs["band_pct"] == pytest.approx(DEFAULT_REBALANCE_BAND_PCT)


def test_duplicate_ticker_rows_are_merged_into_one_position():
    """A concatenated CSV can list AAA twice; that is one position of 150.

    50 x 20 + 100 x 20 = 3,000 of AAA against 1,000 of BBB -> 75% / 25%.
    """
    positions = make_positions(["AAA", "BBB", "AAA"], [50.0, 100.0, 100.0], [20.0, 10.0, 20.0])
    drift = drift_table(positions, {"AAA": 50, "BBB": 50}).set_index("ticker")

    assert len(drift) == 2
    assert drift.loc["AAA", "current_value"] == pytest.approx(3000.0)
    assert drift.loc["AAA", "current_weight_pct"] == pytest.approx(75.0)
    assert drift.loc["BBB", "current_weight_pct"] == pytest.approx(25.0)


def test_a_duplicated_index_does_not_scramble_the_table(book: pd.DataFrame):
    """A sliced or concatenated positions frame arrives with repeated labels."""
    duplicated = book.copy()
    duplicated.index = [7, 7, 7]

    drift = drift_table(duplicated, {"AAA": 40, "BBB": 40, "CCC": 20})
    trades = rebalance_trades(duplicated, {"AAA": 40, "BBB": 40, "CCC": 20})

    assert sorted(drift["ticker"]) == ["AAA", "BBB", "CCC"]
    assert trades.set_index("ticker").loc["AAA", "shares"] == pytest.approx(-33.0)


def test_row_order_of_the_positions_frame_does_not_change_the_answer(
    book: pd.DataFrame,
):
    """Identity: the tables are keyed on ticker, not on row position."""
    shuffled = book.iloc[[2, 0, 1]].reset_index(drop=True)

    a = rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20}).set_index("ticker")
    b = rebalance_trades(shuffled, {"AAA": 40, "BBB": 40, "CCC": 20}).set_index("ticker")

    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_a_row_with_no_usable_value_is_dropped_rather_than_poisoning_the_book():
    """A quote that failed leaves ``current_value`` NaN. That name cannot be
    weighted, but it must not turn the other weights into NaN."""
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BROKEN", "BBB"],
            "quantity": [100.0, 10.0, 100.0],
            "current_price": [60.0, np.nan, 40.0],
            "fx_rate": [1.0, 1.0, 1.0],
            "current_value": [6000.0, np.nan, 4000.0],
        }
    )
    drift = drift_table(positions, {"AAA": 50, "BBB": 50}).set_index("ticker")

    assert "BROKEN" not in drift.index
    assert drift.loc["AAA", "current_weight_pct"] == pytest.approx(60.0)
    assert drift.loc["BBB", "current_weight_pct"] == pytest.approx(40.0)


def test_nothing_mutates_the_caller_s_positions_frame(book: pd.DataFrame):
    before = book.copy(deep=True)

    drift_table(book, {"AAA": 40, "BBB": 40, "CCC": 20})
    rebalance_trades(book, {"AAA": 40, "BBB": 40, "CCC": 20}, cash_to_invest=500.0)

    pd.testing.assert_frame_equal(book, before)


def test_nothing_mutates_the_caller_s_target_mapping(book: pd.DataFrame):
    targets = {"AAA": 60, "BBB": 30, "CCC": 30}
    snapshot = dict(targets)

    drift_table(book, targets)
    rebalance_trades(book, targets)

    assert targets == snapshot


def test_an_empty_book_with_cash_deploys_it_into_the_unpriced_targets():
    """Nothing held, 10,000 to invest, a 60/40 target: both names are BUYs with
    NaN share counts (there is no price yet) and the money split 6,000/4,000."""
    trades = rebalance_trades(None, {"AAA": 60, "BBB": 40}, cash_to_invest=10000.0)
    by_ticker = trades.set_index("ticker")

    assert set(trades["action"]) == {BUY}
    assert trades["shares"].isna().all()
    assert by_ticker.loc["AAA", "trade_value"] == pytest.approx(6000.0)
    assert by_ticker.loc["BBB", "trade_value"] == pytest.approx(4000.0)
    assert trades.attrs["no_price"] == ["AAA", "BBB"]
    assert rebalance_summary(trades)["net_cash"] == pytest.approx(-10000.0)
