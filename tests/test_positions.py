"""Position-level P&L, FX conversion and portfolio aggregation.

``src/analytics/positions.py`` is the first number a user ever sees: it turns a
messy holdings grid into "you invested X, you now have Y, that is Z%". Every
downstream page (risk, attribution, optimiser, the PDF tearsheet) reads the
positions table this module produces, so a sign error or a currency mix-up here
is silently inherited by the whole product.

What this file verifies
-----------------------
* **Hand-computed golden values.** The arithmetic is written out in comments
  next to each assertion -- 10 shares bought at 100, now worth 120, is 1000
  invested / 1200 held / +200 / +20%. A test that only checks ``f(x) == f(x)``
  would not have caught the v1 bug where P&L% was computed against market value
  instead of cost.
* **Mathematical identities** that cannot rot as the code evolves: weights sum
  to 100; ``total_pnl == current_value - total_investment``; every
  ``allocation_by`` key sums to 100; ``hhi * effective_holdings == 1``;
  effective holdings of an equal-weight book of N names is exactly N.
* **The FX convention**, which is the subtlest thing in the module: prices stay
  in the instrument's NATIVE currency, money columns are scaled by ``fx_rate``,
  and because both legs of the P&L are converted at the same rate ``pnl_pct``
  must be bit-for-bit identical with and without FX.
* **Documented degradation** rather than exceptions: empty input, ``None``,
  a missing quote, a NaN quantity, a zero cost basis, all-invalid rows.
* **Regression guards** on the conventions the UI depends on: the exact
  ``POSITION_COLUMNS`` schema and order, the exact wording of the bad-row
  problem message, one message per ticker, blank tickers dropped in silence,
  and duplicate lots merged at a quantity-weighted average price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_quote
from src.analytics.positions import (
    ALLOCATION_KEYS,
    POSITION_COLUMNS,
    allocation_by,
    build_positions,
    clean_holdings,
    effective_holdings,
    summarise,
)
from src.models import SecurityProfile

# ---------------------------------------------------------------------------
# local hand-built fixtures
#
# The shared conftest fixtures are used wherever an IDENTITY is being checked.
# Where a GOLDEN VALUE is being checked the frame is built inline instead, so
# the expected number can be derived in three lines of arithmetic in a comment.
# ---------------------------------------------------------------------------

#: A three-name book whose every headline statistic is computable by hand.
#:
#:   AAA  10 @ 100 cost, now 120  ->  inv 1000, val 1200, pnl  +200, +20%
#:   BBB   8 @ 125 cost, now 100  ->  inv 1000, val  800, pnl  -200, -20%
#:   CCC   5 @  50 cost, now  80  ->  inv  250, val  400, pnl  +150, +60%
#:   ------------------------------------------------------------------
#:   totals                           inv 2250, val 2400, pnl  +150
#:
#: Previous closes are deliberately different from the buy prices so that a
#: day-change bug cannot hide behind an accidentally equal P&L number.
_GOLDEN_TICKERS = ("AAA", "BBB", "CCC")
_GOLDEN_QUANTITY = (10.0, 8.0, 5.0)
_GOLDEN_BUY = (100.0, 125.0, 50.0)
_GOLDEN_PRICE = (120.0, 100.0, 80.0)
_GOLDEN_PREV = (100.0, 80.0, 100.0)


@pytest.fixture
def golden_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(_GOLDEN_TICKERS),
            "quantity": list(_GOLDEN_QUANTITY),
            "buy_price": list(_GOLDEN_BUY),
        }
    )


@pytest.fixture
def golden_quotes() -> dict:
    return {
        ticker: make_quote(ticker, price, previous_close=prev)
        for ticker, price, prev in zip(_GOLDEN_TICKERS, _GOLDEN_PRICE, _GOLDEN_PREV, strict=True)
    }


@pytest.fixture
def golden_positions(golden_holdings, golden_quotes, profiles) -> pd.DataFrame:
    """The golden book enriched with the shared ``profiles`` fixture."""
    positions, skipped = build_positions(golden_holdings, golden_quotes, profiles)
    assert skipped == []
    return positions


def _equal_weight_positions(n: int) -> pd.DataFrame:
    """``n`` names, identical quantity and price -> exactly 100/n percent each."""
    tickers = [f"T{i}" for i in range(n)]
    holdings = pd.DataFrame({"ticker": tickers, "quantity": [2.0] * n, "buy_price": [50.0] * n})
    quotes = {t: make_quote(t, 100.0) for t in tickers}
    positions, _ = build_positions(holdings, quotes)
    return positions


# ---- clean_holdings: normalisation of raw user input -----------------------
def test_clean_holdings_accepts_flexible_csv_column_names():
    """ "Ticker" / " Quantity" / "Buy Price" must normalise to the canonical schema."""
    raw = pd.DataFrame({"Ticker": ["aaa"], " Quantity": [10], "Buy Price": [100.0]})

    clean, problems = clean_holdings(raw)

    assert problems == []
    assert list(clean.columns) == ["ticker", "quantity", "buy_price"]
    assert clean.loc[0, "ticker"] == "AAA"  # also upper-cased
    assert clean.loc[0, "quantity"] == pytest.approx(10.0)
    assert clean.loc[0, "buy_price"] == pytest.approx(100.0)


def test_clean_holdings_strips_and_upper_cases_tickers():
    raw = pd.DataFrame({"ticker": ["  reliance.ns  "], "quantity": [1], "buy_price": [1.0]})

    clean, problems = clean_holdings(raw)

    assert problems == []
    assert clean["ticker"].tolist() == ["RELIANCE.NS"]


def test_clean_holdings_reports_exactly_one_problem_per_invalid_ticker():
    """Two bad BBB rows must produce ONE message, not two -- the grid would spam."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "BBB", "CCC"],
            "quantity": [10, -5, 3, 2],  # BBB row 1 has a negative quantity
            "buy_price": [100, 20, -1, "oops"],  # BBB row 2 negative, CCC unparseable
        }
    )

    clean, problems = clean_holdings(raw)

    assert clean["ticker"].tolist() == ["AAA"]
    assert problems == [
        "BBB: quantity and buy price must be positive numbers - row skipped.",
        "CCC: quantity and buy price must be positive numbers - row skipped.",
    ]


def test_clean_holdings_drops_blank_ticker_rows_silently():
    """Empty editor rows are not user errors -- they must NOT raise a message."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "", None, "   ", "nan"],
            "quantity": [10, 1, 1, 1, 1],
            "buy_price": [100.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    clean, problems = clean_holdings(raw)

    assert clean["ticker"].tolist() == ["AAA"]
    assert problems == []  # four rows vanished without a word


def test_clean_holdings_merges_duplicate_tickers_at_a_quantity_weighted_average():
    """10 @ 100 plus 30 @ 200 -> 40 shares at (1000 + 6000) / 40 = 175."""
    raw = pd.DataFrame(
        {"ticker": ["AAA", "AAA"], "quantity": [10, 30], "buy_price": [100.0, 200.0]}
    )

    clean, problems = clean_holdings(raw)

    assert problems == []
    assert len(clean) == 1
    assert clean.loc[0, "ticker"] == "AAA"
    assert clean.loc[0, "quantity"] == pytest.approx(40.0)
    assert clean.loc[0, "buy_price"] == pytest.approx(175.0)


def test_clean_holdings_merge_conserves_total_cost():
    """Identity: the merged lot must cost exactly what the separate lots cost."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "quantity": [3.0, 7.0, 11.0],
            "buy_price": [101.5, 99.25, 143.75],
        }
    )
    expected_cost = 3.0 * 101.5 + 7.0 * 99.25 + 11.0 * 143.75  # = 2580.5

    clean, _ = clean_holdings(raw)

    merged_cost = float(clean.loc[0, "quantity"] * clean.loc[0, "buy_price"])
    assert merged_cost == pytest.approx(expected_cost, rel=1e-12)


def test_clean_holdings_keeps_optional_columns_and_takes_the_earliest_buy_date():
    """A merged name keeps the EARLIEST lot date and sums its target weight."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "quantity": [10, 10],
            "buy_price": [100.0, 100.0],
            "buy_date": ["2023-06-15", "2023-01-10"],
            "sector": ["", "Energy"],
            "target_weight_pct": [20.0, 30.0],
        }
    )

    clean, problems = clean_holdings(raw)

    assert problems == []
    assert clean.loc[0, "buy_date"] == pd.Timestamp("2023-01-10")
    assert clean.loc[0, "sector"] == "Energy"  # first NON-BLANK label wins
    assert clean.loc[0, "target_weight_pct"] == pytest.approx(50.0)


def test_clean_holdings_returns_a_deterministic_ticker_order_for_unsorted_input():
    raw = pd.DataFrame(
        {"ticker": ["ZZZ", "AAA", "MMM"], "quantity": [1, 2, 3], "buy_price": [1, 2, 3]}
    )

    clean, _ = clean_holdings(raw)

    assert clean["ticker"].tolist() == ["AAA", "MMM", "ZZZ"]


def test_clean_holdings_keeps_a_single_valid_row():
    raw = pd.DataFrame({"ticker": ["AAA"], "quantity": [1.5], "buy_price": [12.5]})

    clean, problems = clean_holdings(raw)

    assert problems == []
    assert len(clean) == 1
    assert clean.loc[0, "quantity"] == pytest.approx(1.5)


# ---- clean_holdings: documented degradation, never an exception ------------
def test_clean_holdings_flags_none_input_rather_than_crashing():
    clean, problems = clean_holdings(None)

    assert clean.empty
    assert list(clean.columns) == ["ticker", "quantity", "buy_price"]
    assert problems == ["Portfolio is empty - add at least one holding."]


def test_clean_holdings_flags_an_empty_frame_rather_than_crashing():
    clean, problems = clean_holdings(pd.DataFrame())

    assert clean.empty
    assert problems == ["Portfolio is empty - add at least one holding."]


def test_clean_holdings_reports_missing_required_columns():
    raw = pd.DataFrame({"ticker": ["AAA"], "quantity": [1]})  # no buy_price

    clean, problems = clean_holdings(raw)

    assert clean.empty
    assert problems == ["Missing column(s): buy_price."]


def test_clean_holdings_reports_when_every_row_is_invalid():
    raw = pd.DataFrame({"ticker": ["AAA"], "quantity": [0], "buy_price": [100.0]})

    clean, problems = clean_holdings(raw)

    assert clean.empty
    assert problems[-1] == "No valid holdings after validation."


# ---- build_positions: hand-computed position mathematics -------------------
def test_position_maths_for_ten_shares_bought_at_100_and_now_worth_120():
    # 10 shares at 100 -> 1000 invested; now 120 -> 1200 value; P/L 200 = +20%
    # a single holding is the whole book, so weight_pct is exactly 100
    holdings = pd.DataFrame({"ticker": ["AAA"], "quantity": [10.0], "buy_price": [100.0]})

    positions, skipped = build_positions(holdings, {"AAA": make_quote("AAA", 120.0)})

    assert skipped == []
    row = positions.loc[0]
    assert row["investment"] == pytest.approx(1000.0)
    assert row["current_value"] == pytest.approx(1200.0)
    assert row["pnl"] == pytest.approx(200.0)
    assert row["pnl_pct"] == pytest.approx(20.0)
    assert row["weight_pct"] == pytest.approx(100.0)


def test_a_loss_making_position_reports_a_negative_pnl_and_pct():
    # 8 shares at 125 -> 1000 invested; now 100 -> 800 value; P/L -200 = -20%
    holdings = pd.DataFrame({"ticker": ["BBB"], "quantity": [8.0], "buy_price": [125.0]})

    positions, _ = build_positions(holdings, {"BBB": make_quote("BBB", 100.0)})

    assert positions.loc[0, "pnl"] == pytest.approx(-200.0)
    assert positions.loc[0, "pnl_pct"] == pytest.approx(-20.0)


def test_weights_sum_to_100_and_match_the_value_shares():
    # AAA: 10 @ 120 = 1200      BBB: 8 @ 100 = 800      total = 2000
    # 1200 / 2000 = 60%         800 / 2000 = 40%
    holdings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "quantity": [10.0, 8.0], "buy_price": [100.0, 100.0]}
    )
    quotes = {"AAA": make_quote("AAA", 120.0), "BBB": make_quote("BBB", 100.0)}

    positions, _ = build_positions(holdings, quotes)

    weights = dict(zip(positions["ticker"], positions["weight_pct"], strict=True))
    assert weights["AAA"] == pytest.approx(60.0)
    assert weights["BBB"] == pytest.approx(40.0)
    assert positions["weight_pct"].sum() == pytest.approx(100.0, abs=1e-9)


def test_weights_sum_to_100_for_the_shared_fixture_portfolio(holdings_frame, quotes, profiles):
    """Identity check on realistic random-walk prices, not hand-picked ones."""
    positions, skipped = build_positions(holdings_frame, quotes, profiles)

    assert skipped == []
    assert positions["weight_pct"].sum() == pytest.approx(100.0, abs=1e-9)


def test_pnl_identity_holds_row_by_row(golden_positions):
    """pnl == current_value - investment, for every single row."""
    residual = (
        golden_positions["current_value"] - golden_positions["investment"] - golden_positions["pnl"]
    )
    assert residual.abs().max() == pytest.approx(0.0, abs=1e-9)


def test_positions_are_sorted_by_current_value_descending(golden_positions):
    # values are 1200 (AAA), 800 (BBB), 400 (CCC)
    assert golden_positions["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert golden_positions["current_value"].is_monotonic_decreasing


def test_position_columns_follow_the_documented_schema(golden_positions):
    """Regression guard: the UI, Excel export and HTML report index by name."""
    assert list(golden_positions.columns) == list(POSITION_COLUMNS)
    assert POSITION_COLUMNS[:4] == ("ticker", "name", "sector", "industry")
    assert POSITION_COLUMNS[-1] == "weight_pct"


def test_optional_holdings_columns_are_appended_after_the_documented_schema():
    holdings = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "quantity": [10.0],
            "buy_price": [100.0],
            "buy_date": pd.to_datetime(["2023-01-10"]),
            "target_weight_pct": [50.0],
        }
    )

    positions, _ = build_positions(holdings, {"AAA": make_quote("AAA", 120.0)})

    assert list(positions.columns) == [
        *POSITION_COLUMNS,
        "buy_date",
        "target_weight_pct",
    ]


# ---- build_positions: quotes, skipping and reference data ------------------
def test_a_missing_quote_skips_the_position_and_is_returned_in_skipped():
    holdings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "quantity": [10.0, 5.0], "buy_price": [100.0, 10.0]}
    )

    positions, skipped = build_positions(holdings, {"AAA": make_quote("AAA", 120.0)})

    assert skipped == ["BBB"]
    assert positions["ticker"].tolist() == ["AAA"]
    # the surviving name absorbs the whole book
    assert positions.loc[0, "weight_pct"] == pytest.approx(100.0)


def test_a_non_positive_quote_price_skips_the_position():
    holdings = pd.DataFrame({"ticker": ["AAA"], "quantity": [10.0], "buy_price": [100.0]})

    positions, skipped = build_positions(holdings, {"AAA": make_quote("AAA", 0.0)})

    assert skipped == ["AAA"]
    assert positions.empty


def test_a_nan_quantity_skips_the_position_rather_than_poisoning_the_totals():
    holdings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "quantity": [np.nan, 5.0], "buy_price": [100.0, 10.0]}
    )
    quotes = {"AAA": make_quote("AAA", 120.0), "BBB": make_quote("BBB", 20.0)}

    positions, skipped = build_positions(holdings, quotes)

    assert skipped == ["AAA"]
    assert positions["ticker"].tolist() == ["BBB"]
    assert np.isfinite(positions["current_value"]).all()


def test_every_quote_missing_yields_an_empty_frame_with_the_full_schema():
    holdings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "quantity": [10.0, 5.0], "buy_price": [100.0, 10.0]}
    )

    positions, skipped = build_positions(holdings, {})

    assert skipped == ["AAA", "BBB"]
    assert len(positions) == 0
    assert list(positions.columns) == list(POSITION_COLUMNS)


def test_none_holdings_yields_an_empty_frame_rather_than_raising():
    positions, skipped = build_positions(None, {"AAA": make_quote("AAA", 1.0)})

    assert len(positions) == 0
    assert skipped == []
    assert positions.attrs["base_currency"] == "INR"


def test_profiles_supply_name_sector_and_industry(golden_positions):
    aaa = golden_positions.set_index("ticker").loc["AAA"]

    assert aaa["name"] == "Alpha Industries"
    assert aaa["sector"] == "Energy"
    assert aaa["industry"] == "Refining"


def test_an_etf_without_a_sector_falls_back_to_fund_slash_etf(golden_positions):
    """CCC is quote_type='ETF' with sector=None -> SecurityProfile.sector_label."""
    ccc = golden_positions.set_index("ticker").loc["CCC"]

    assert ccc["name"] == "Gamma Index Fund"
    assert ccc["sector"] == "Fund / ETF"
    assert ccc["industry"] == "Unclassified"  # ETFs have no industry either


def test_a_holdings_sector_column_fills_in_when_the_profile_has_none():
    holdings = pd.DataFrame(
        {
            "ticker": ["XXX"],
            "quantity": [1.0],
            "buy_price": [1.0],
            "sector": ["Utilities"],
        }
    )
    profile = {"XXX": SecurityProfile("XXX", name="Ex Co", sector=None)}

    positions, _ = build_positions(holdings, {"XXX": make_quote("XXX", 2.0)}, profile)

    assert positions.loc[0, "sector"] == "Utilities"


def test_a_profile_sector_beats_the_holdings_sector_column():
    holdings = pd.DataFrame(
        {"ticker": ["XXX"], "quantity": [1.0], "buy_price": [1.0], "sector": ["Guess"]}
    )
    profile = {"XXX": SecurityProfile("XXX", sector="Financials")}

    positions, _ = build_positions(holdings, {"XXX": make_quote("XXX", 2.0)}, profile)

    assert positions.loc[0, "sector"] == "Financials"


def test_a_missing_profile_degrades_to_the_ticker_and_unclassified():
    holdings = pd.DataFrame({"ticker": ["ZZZ"], "quantity": [1.0], "buy_price": [1.0]})

    positions, _ = build_positions(holdings, {"ZZZ": make_quote("ZZZ", 2.0)}, {})

    assert positions.loc[0, "name"] == "ZZZ"
    assert positions.loc[0, "sector"] == "Unclassified"
    assert positions.loc[0, "industry"] == "Unclassified"


# ---- FX conversion: native prices, converted money -------------------------
def _usd_book() -> tuple[pd.DataFrame, dict]:
    """One US name: 10 shares bought at $100, now $120."""
    holdings = pd.DataFrame({"ticker": ["USX"], "quantity": [10.0], "buy_price": [100.0]})
    quotes = {"USX": make_quote("USX", 120.0, currency="USD")}
    return holdings, quotes


def test_fx_leaves_the_native_prices_unchanged():
    """A dollar stock in a rupee book is still shown at its dollar price."""
    holdings, quotes = _usd_book()

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0})

    assert positions.loc[0, "currency"] == "USD"
    assert positions.loc[0, "buy_price"] == pytest.approx(100.0)
    assert positions.loc[0, "current_price"] == pytest.approx(120.0)
    assert positions.loc[0, "fx_rate"] == pytest.approx(80.0)


def test_fx_scales_every_money_column_by_the_rate():
    # 10 x 100 x 80 = 80,000 invested;  10 x 120 x 80 = 96,000 held
    # P/L = 96,000 - 80,000 = 16,000  (= 200 USD of profit x 80)
    holdings, quotes = _usd_book()

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0})

    assert positions.loc[0, "investment"] == pytest.approx(80_000.0)
    assert positions.loc[0, "current_value"] == pytest.approx(96_000.0)
    assert positions.loc[0, "pnl"] == pytest.approx(16_000.0)


def test_pnl_pct_is_identical_with_and_without_fx():
    """Both legs convert at the same rate, so the percentage is FX-neutral."""
    holdings, quotes = _usd_book()

    unconverted, _ = build_positions(holdings, quotes)
    converted, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0})

    # 120 / 100 - 1 = +20%, whatever the rupee does
    assert unconverted.loc[0, "pnl_pct"] == pytest.approx(20.0)
    assert converted.loc[0, "pnl_pct"] == pytest.approx(
        float(unconverted.loc[0, "pnl_pct"]), rel=1e-12
    )


def test_fx_scaling_is_exactly_proportional_across_the_whole_book(golden_holdings):
    """Every money column of a foreign book scales by exactly the same factor."""
    quotes = {
        t: make_quote(t, p, currency="USD")
        for t, p in zip(_GOLDEN_TICKERS, _GOLDEN_PRICE, strict=True)
    }
    rate = 83.25

    base, _ = build_positions(golden_holdings, quotes, fx_rates={"USD": 1.0})
    scaled, _ = build_positions(golden_holdings, quotes, fx_rates={"USD": rate})

    for column in ("investment", "current_value", "pnl"):
        assert scaled[column].to_numpy() == pytest.approx(base[column].to_numpy() * rate, rel=1e-12)
    # weights are ratios of converted values, so FX cancels out entirely
    assert scaled["weight_pct"].to_numpy() == pytest.approx(
        base["weight_pct"].to_numpy(), rel=1e-12
    )


def test_fx_rate_lookup_is_case_insensitive():
    """A quote reporting 'usd' must still match an fx_rates key of 'USD'."""
    holdings = pd.DataFrame({"ticker": ["USX"], "quantity": [2.0], "buy_price": [50.0]})
    quotes = {"USX": make_quote("USX", 50.0, currency="usd")}

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0})

    assert positions.loc[0, "fx_rate"] == pytest.approx(80.0)
    # 2 x 50 x 80 = 8,000
    assert positions.loc[0, "current_value"] == pytest.approx(8_000.0)


@pytest.mark.parametrize("bad_rate", [0.0, -3.0, float("nan"), float("inf")])
def test_an_unusable_fx_rate_degrades_to_one_and_is_reported(bad_rate):
    """Never poison the totals with NaN -- fall back to 1.0 and say so out loud."""
    holdings, quotes = _usd_book()

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": bad_rate})

    assert positions.loc[0, "fx_rate"] == pytest.approx(1.0)
    assert positions.loc[0, "current_value"] == pytest.approx(1200.0)  # 10 x 120 x 1
    assert positions.attrs["fx_unconverted"] == ["USD"]


def test_a_currency_absent_from_fx_rates_is_reported_as_unconverted():
    holdings = pd.DataFrame({"ticker": ["EUX"], "quantity": [1.0], "buy_price": [1.0]})
    quotes = {"EUX": make_quote("EUX", 2.0, currency="EUR")}

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0})

    assert positions.loc[0, "fx_rate"] == pytest.approx(1.0)
    assert positions.attrs["fx_unconverted"] == ["EUR"]


def test_the_base_currency_itself_is_never_flagged_as_unconverted():
    """INR in an INR book needs no rate; flagging it would be noise, not news."""
    holdings = pd.DataFrame({"ticker": ["INX"], "quantity": [1.0], "buy_price": [1.0]})
    quotes = {"INX": make_quote("INX", 2.0, currency="INR")}

    positions, _ = build_positions(holdings, quotes, fx_rates={"USD": 80.0}, base_currency="INR")

    assert positions.loc[0, "fx_rate"] == pytest.approx(1.0)
    assert positions.attrs["fx_unconverted"] == []


def test_the_positions_frame_records_its_base_currency():
    holdings, quotes = _usd_book()

    positions, _ = build_positions(holdings, quotes, base_currency="usd")

    assert positions.attrs["base_currency"] == "usd"


def test_no_fx_rates_reproduces_the_legacy_single_currency_behaviour(
    golden_holdings, golden_quotes
):
    """fx_rates=None must be identical to every rate being exactly 1.0."""
    without, _ = build_positions(golden_holdings, golden_quotes, fx_rates=None)
    with_ones, _ = build_positions(golden_holdings, golden_quotes, fx_rates={"INR": 1.0})

    assert (without["fx_rate"] == 1.0).all()
    assert without["current_value"].to_numpy() == pytest.approx(
        with_ones["current_value"].to_numpy(), rel=1e-12
    )


# ---- summarise: hand-computed totals --------------------------------------
def test_summary_totals_are_hand_computable(golden_positions):
    # investment  1000 + 1000 + 250 = 2250
    # value       1200 +  800 + 400 = 2400
    # P/L                             = 150
    # return      150 / 2250 x 100    = 6.666...%
    summary = summarise(golden_positions)

    assert summary["n_holdings"] == 3
    assert summary["total_investment"] == pytest.approx(2250.0)
    assert summary["current_value"] == pytest.approx(2400.0)
    assert summary["total_pnl"] == pytest.approx(150.0)
    assert summary["total_return_pct"] == pytest.approx(150.0 / 2250.0 * 100.0)
    assert summary["currencies"] == ["INR"]


def test_summary_pnl_identity_holds(holdings_frame, quotes, profiles):
    """total_pnl == current_value - total_investment, on fixture data."""
    positions, _ = build_positions(holdings_frame, quotes, profiles)

    summary = summarise(positions)

    assert summary["total_pnl"] == pytest.approx(
        summary["current_value"] - summary["total_investment"], rel=1e-12
    )
    assert summary["current_value"] == pytest.approx(
        float(positions["current_value"].sum()), rel=1e-12
    )


def test_summary_names_the_best_and_worst_performer_by_pnl_pct(golden_positions):
    # AAA +20%, BBB -20%, CCC +60%  ->  best CCC, worst BBB
    summary = summarise(golden_positions)

    assert summary["best"][0] == "CCC"
    assert summary["best"][1] == pytest.approx(60.0)
    assert summary["worst"][0] == "BBB"
    assert summary["worst"][1] == pytest.approx(-20.0)


def test_summary_names_the_largest_holding_by_weight(golden_positions):
    # AAA is 1200 / 2400 = 50% of the book
    summary = summarise(golden_positions)

    assert summary["largest_holding"][0] == "AAA"
    assert summary["largest_holding"][1] == pytest.approx(50.0)


def test_summary_counts_gainers_and_losers(golden_positions):
    # AAA +200 and CCC +150 are gainers; BBB -200 is the only loser
    summary = summarise(golden_positions)

    assert summary["n_gainers"] == 2
    assert summary["n_losers"] == 1
    assert summary["n_gainers"] + summary["n_losers"] <= summary["n_holdings"]


# ---- summarise: concentration --------------------------------------------
def test_summary_hhi_is_hand_computable(golden_positions):
    # weights 50 / 33.333 / 16.667 -> (1/2)^2 + (1/3)^2 + (1/6)^2
    #                               = 0.25 + 0.111111 + 0.027778 = 0.3888889
    expected = 0.5**2 + (1 / 3) ** 2 + (1 / 6) ** 2

    summary = summarise(golden_positions)

    assert summary["hhi"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 < summary["hhi"] <= 1.0


def test_summary_hhi_and_effective_holdings_are_reciprocal(golden_positions):
    """Identity: effective_holdings == 1 / HHI, always."""
    summary = summarise(golden_positions)

    assert summary["hhi"] * summary["effective_holdings"] == pytest.approx(1.0, rel=1e-12)
    # 1 / 0.3888889 = 2.571428...
    assert summary["effective_holdings"] == pytest.approx(18.0 / 7.0, rel=1e-12)


def test_summary_top5_weight_pct_is_100_for_a_three_name_book(golden_positions):
    """With fewer than five names, "top 5" is the whole portfolio."""
    summary = summarise(golden_positions)

    assert summary["top5_weight_pct"] == pytest.approx(100.0, abs=1e-9)


def test_summary_top5_weight_pct_takes_only_the_five_largest():
    # eight equal names at 12.5% each -> top five = 5 x 12.5 = 62.5%
    positions = _equal_weight_positions(8)

    summary = summarise(positions)

    assert summary["top5_weight_pct"] == pytest.approx(62.5, abs=1e-9)


def test_summary_sectors_sum_to_100_and_are_ranked_by_weight(golden_positions):
    # Energy 50 (AAA), Technology 33.333 (BBB), Fund / ETF 16.667 (CCC)
    summary = summarise(golden_positions)

    assert list(summary["sectors"]) == ["Energy", "Technology", "Fund / ETF"]
    assert summary["sectors"]["Energy"] == pytest.approx(50.0)
    assert summary["sectors"]["Technology"] == pytest.approx(100.0 / 3.0)
    assert summary["sectors"]["Fund / ETF"] == pytest.approx(100.0 / 6.0)
    assert sum(summary["sectors"].values()) == pytest.approx(100.0, abs=1e-9)


def test_summary_records_the_base_currency(golden_positions):
    assert summarise(golden_positions)["base_currency"] == "INR"


# ---- summarise: the value-weighted day change -----------------------------
def test_summary_day_change_is_value_weighted(golden_positions):
    # AAA  1200 now, +20% today -> yesterday 1200 / 1.20 = 1000, change +200
    # BBB   800 now, +25% today -> yesterday  800 / 1.25 =  640, change +160
    # CCC   400 now, -20% today -> yesterday  400 / 0.80 =  500, change -100
    # ---------------------------------------------------------------------
    #                                        base 2140,  change +260
    # day_change_pct = 260 / 2140 x 100 = 12.14953...%
    summary = summarise(golden_positions)

    assert summary["day_change"] == pytest.approx(260.0, rel=1e-12)
    assert summary["day_change_pct"] == pytest.approx(260.0 / 2140.0 * 100.0, rel=1e-12)


def test_summary_excludes_positions_without_a_previous_close_from_the_day_change():
    """A name with no previous close is EXCLUDED, never silently treated as flat."""
    # AAA 10 @ 110 = 1100 now, +10% today -> yesterday 1000, change +100
    # BBB 10 @ 200 = 2000 now, no previous close -> not in the calculation at all
    # day_change_pct = 100 / 1000 x 100 = 10%, NOT 100 / 3000 = 3.33%
    holdings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "quantity": [10.0, 10.0], "buy_price": [100.0, 100.0]}
    )
    quotes = {
        "AAA": make_quote("AAA", 110.0, previous_close=100.0),
        "BBB": make_quote("BBB", 200.0),
    }
    positions, _ = build_positions(holdings, quotes)

    summary = summarise(positions)

    assert summary["day_change"] == pytest.approx(100.0, rel=1e-12)
    assert summary["day_change_pct"] == pytest.approx(10.0, rel=1e-12)


def test_summary_day_change_is_nan_when_no_position_reports_a_previous_close(
    holdings_frame, quotes, profiles
):
    """The shared quotes fixture carries no previous close -- report NaN, not 0."""
    positions, _ = build_positions(holdings_frame, quotes, profiles)

    summary = summarise(positions)

    assert np.isnan(summary["day_change"])
    assert np.isnan(summary["day_change_pct"])


# ---- summarise: degradation ----------------------------------------------
def test_summary_of_none_is_just_n_holdings_zero():
    assert summarise(None) == {"n_holdings": 0}


def test_summary_of_an_empty_frame_is_just_n_holdings_zero():
    empty, _ = build_positions(pd.DataFrame(), {})

    assert summarise(empty) == {"n_holdings": 0}


def test_summary_total_return_pct_is_nan_when_nothing_was_invested():
    """A zero cost basis is a divide-by-zero: report NaN, never a fake infinity."""
    holdings = pd.DataFrame({"ticker": ["GIFT"], "quantity": [10.0], "buy_price": [0.0]})

    positions, _ = build_positions(holdings, {"GIFT": make_quote("GIFT", 50.0)})
    summary = summarise(positions)

    assert positions.loc[0, "investment"] == pytest.approx(0.0)
    assert positions.loc[0, "pnl"] == pytest.approx(500.0)  # 10 x 50 - 0
    assert np.isnan(positions.loc[0, "pnl_pct"])
    assert np.isnan(summary["total_return_pct"])
    # a row with no pnl_pct cannot win or lose the best/worst race
    assert "best" not in summary


# ---- allocation_by --------------------------------------------------------
@pytest.mark.parametrize("key", ALLOCATION_KEYS)
def test_allocation_weights_sum_to_100_for_every_key(golden_positions, key):
    """Identity: each bucket's share of value must partition the whole book."""
    allocation = allocation_by(golden_positions, key)

    assert list(allocation.columns) == [key, "current_value", "weight_pct", "n"]
    assert allocation["weight_pct"].sum() == pytest.approx(100.0, abs=1e-9)
    assert allocation["n"].sum() == len(golden_positions)
    assert allocation["current_value"].sum() == pytest.approx(2400.0)


def test_allocation_by_sector_is_hand_computable(golden_positions):
    # Energy 1200 (50%), Technology 800 (33.33%), Fund / ETF 400 (16.67%) of 2400
    allocation = allocation_by(golden_positions, "sector").set_index("sector")

    assert allocation.loc["Energy", "current_value"] == pytest.approx(1200.0)
    assert allocation.loc["Energy", "weight_pct"] == pytest.approx(50.0)
    assert allocation.loc["Technology", "weight_pct"] == pytest.approx(100.0 / 3.0)
    assert allocation.loc["Fund / ETF", "weight_pct"] == pytest.approx(100.0 / 6.0)


def test_allocation_by_ticker_reproduces_the_position_weights(golden_positions):
    allocation = allocation_by(golden_positions, "ticker").set_index("ticker")
    expected = golden_positions.set_index("ticker")["weight_pct"]

    for ticker, weight in expected.items():
        assert allocation.loc[ticker, "weight_pct"] == pytest.approx(float(weight), rel=1e-12)


def test_allocation_is_sorted_by_value_and_counts_rows_per_bucket():
    # two Energy names (600 + 400 = 1000) against one Tech name (1500)
    positions = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "sector": ["Energy", "Energy", "Technology"],
            "current_value": [600.0, 400.0, 1500.0],
        }
    )

    allocation = allocation_by(positions, "sector")

    assert allocation["sector"].tolist() == ["Technology", "Energy"]
    assert allocation["n"].tolist() == [1, 2]
    # 1500 / 2500 = 60%, 1000 / 2500 = 40%
    assert allocation["weight_pct"].tolist() == pytest.approx([60.0, 40.0])


def test_allocation_buckets_missing_labels_as_unclassified():
    positions = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "sector": [None, "   "],
            "current_value": [300.0, 700.0],
        }
    )

    allocation = allocation_by(positions, "sector")

    assert allocation["sector"].tolist() == ["Unclassified"]
    assert allocation.loc[0, "current_value"] == pytest.approx(1000.0)
    assert allocation.loc[0, "weight_pct"] == pytest.approx(100.0)


def test_allocation_with_an_unknown_key_returns_an_empty_frame(golden_positions):
    allocation = allocation_by(golden_positions, "moon_phase")

    assert allocation.empty
    assert list(allocation.columns) == ["key", "current_value", "weight_pct", "n"]


def test_allocation_of_an_empty_portfolio_returns_an_empty_frame():
    assert allocation_by(None, "sector").empty
    assert allocation_by(pd.DataFrame(), "sector").empty


def test_allocation_returns_empty_when_the_grouping_column_is_absent():
    positions = pd.DataFrame({"ticker": ["A"], "current_value": [1.0]})

    assert allocation_by(positions, "sector").empty


# ---- effective_holdings ---------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10])
def test_effective_holdings_of_an_equal_weight_portfolio_is_exactly_n(n):
    """1 / sum((1/n)^2) = 1 / (n x 1/n^2) = n. The defining property."""
    positions = _equal_weight_positions(n)

    assert positions["weight_pct"].to_numpy() == pytest.approx([100.0 / n] * n, rel=1e-12)
    assert effective_holdings(positions) == pytest.approx(float(n), rel=1e-12)


def test_effective_holdings_of_a_single_holding_is_one():
    holdings = pd.DataFrame({"ticker": ["ONE"], "quantity": [3.0], "buy_price": [10.0]})

    positions, _ = build_positions(holdings, {"ONE": make_quote("ONE", 20.0)})

    assert positions.loc[0, "weight_pct"] == pytest.approx(100.0)
    assert effective_holdings(positions) == pytest.approx(1.0, rel=1e-12)


def test_effective_holdings_is_hand_computable_for_the_golden_book(golden_positions):
    # 1 / (0.25 + 1/9 + 1/36) = 1 / (7/18) = 18/7 = 2.571428...
    assert effective_holdings(golden_positions) == pytest.approx(18.0 / 7.0, rel=1e-12)


def test_effective_holdings_collapses_towards_one_for_a_concentrated_book():
    """Ten tickers, but 90% in one name: the book behaves like ~1.2 holdings."""
    # w = (0.90, then nine names at 0.10/9 = 0.011111)
    # HHI = 0.81 + 9 x 0.011111^2 = 0.81 + 0.0011111 = 0.8111111
    # 1 / 0.8111111 = 1.23288
    weights = [90.0] + [10.0 / 9.0] * 9
    positions = pd.DataFrame({"ticker": [f"T{i}" for i in range(10)], "weight_pct": weights})

    result = effective_holdings(positions)

    assert result == pytest.approx(1.0 / (0.81 + 9 * (0.1 / 9) ** 2), rel=1e-9)
    assert 1.0 < result < 2.0


def test_effective_holdings_renormalises_a_subset_of_the_book():
    """Half of an equal-weight four-name book is still a coherent two-name book."""
    positions = _equal_weight_positions(4)

    subset = positions.head(2)  # two names at 25% each -> renormalised to 50/50

    assert effective_holdings(subset) == pytest.approx(2.0, rel=1e-12)


def test_effective_holdings_is_nan_for_empty_or_weightless_input():
    assert np.isnan(effective_holdings(None))
    assert np.isnan(effective_holdings(pd.DataFrame()))
    # a frame with the column present but entirely NaN is just as unusable
    assert np.isnan(effective_holdings(pd.DataFrame({"weight_pct": [np.nan, np.nan]})))
    # ... and so is a frame that never had the column
    assert np.isnan(effective_holdings(pd.DataFrame({"ticker": ["A"]})))


# ---- regression guards ----------------------------------------------------
def test_summarise_degrades_gracefully_on_a_duplicate_index():
    """summarise() must not assume a unique index.

    It once used ``positions.loc[series.idxmax()]``; idxmax returns an index
    *label*, and on a non-unique index -- which any caller who concatenates or
    slices positions without reset_index produces -- that .loc yields a
    DataFrame, so ``float(best["pnl_pct"])`` raised TypeError. Positional
    lookup fixed it; this test keeps it fixed.
    """
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "sector": ["Energy", "Technology"],
            "currency": ["INR", "INR"],
            "quantity": [10.0, 10.0],
            "investment": [1000.0, 1000.0],
            "current_value": [1200.0, 800.0],
            "pnl": [200.0, -200.0],
            "pnl_pct": [20.0, -20.0],
            "weight_pct": [60.0, 40.0],
            "day_change_pct": [1.0, -1.0],
        },
        index=[0, 0],  # e.g. two slices concatenated without reset_index()
    )

    summary = summarise(positions)

    assert summary["best"] == ("AAA", pytest.approx(20.0))
    assert summary["worst"] == ("BBB", pytest.approx(-20.0))
    assert summary["largest_holding"] == ("AAA", pytest.approx(60.0))
