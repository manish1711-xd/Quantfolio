"""Tests for the fundamental scores.

Pinned against **hand-computed values**, not against whatever the code
currently returns. Two constructions make that possible without a spreadsheet:

* a *flat* company, whose figures are identical in both years, so every
  Beneish index collapses to exactly 1.0 and the M-Score reduces to
  ``-4.84 + (sum of coefficients) + 4.679 x TATA`` — which pins all eight
  coefficients at once,
* a *growing* company where every year-on-year Piotroski signal has a
  definite direction, so 9/9 is expected rather than incidental.

The property most of this file is really about: **a score computed from
incomplete inputs is reported as incomplete, never as a low score.** Treating
a missing signal as a failed one is the standard bug in implementations of
these scores, and it makes a healthy company look distressed.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from src.analytics import fundamentals as fu

PERIODS = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31"])

#: Every figure identical across periods. Every Beneish index is then exactly
#: 1.0 and every Piotroski year-on-year comparison is a tie.
FLAT = {
    "Total Assets": 1_000_000.0,
    "Current Assets": 400_000.0,
    "Current Liabilities": 200_000.0,
    "Total Liabilities Net Minority Interest": 450_000.0,
    "Long Term Debt": 150_000.0,
    "Total Debt": 220_000.0,
    "Stockholders Equity": 550_000.0,
    "Retained Earnings": 300_000.0,
    "Working Capital": 200_000.0,
    "Ordinary Shares Number": 10_000.0,
    "Accounts Receivable": 90_000.0,
    "Inventory": 70_000.0,
    "Net PPE": 500_000.0,
    "Gross PPE": 700_000.0,
    "Cash And Cash Equivalents": 120_000.0,
    "Other Short Term Investments": 30_000.0,
    "Invested Capital": 770_000.0,
    "Total Revenue": 800_000.0,
    "Gross Profit": 280_000.0,
    "Operating Income": 120_000.0,
    "EBIT": 120_000.0,
    "Net Income": 80_000.0,
    "Pretax Income": 105_000.0,
    "Tax Provision": 25_000.0,
    "Interest Expense": 15_000.0,
    "Selling General And Administration": 90_000.0,
    "Operating Cash Flow": 140_000.0,
    "Free Cash Flow": 60_000.0,
    "Capital Expenditure": -80_000.0,
    "Depreciation And Amortization": 45_000.0,
}

BALANCE_ROWS = (
    "Total Assets",
    "Current Assets",
    "Current Liabilities",
    "Total Liabilities Net Minority Interest",
    "Long Term Debt",
    "Total Debt",
    "Stockholders Equity",
    "Retained Earnings",
    "Working Capital",
    "Ordinary Shares Number",
    "Accounts Receivable",
    "Inventory",
    "Net PPE",
    "Gross PPE",
    "Cash And Cash Equivalents",
    "Other Short Term Investments",
    "Invested Capital",
)
INCOME_ROWS = (
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "EBIT",
    "Net Income",
    "Pretax Income",
    "Tax Provision",
    "Interest Expense",
    "Selling General And Administration",
)
CASH_ROWS = (
    "Operating Cash Flow",
    "Free Cash Flow",
    "Capital Expenditure",
    "Depreciation And Amortization",
)


def frames(
    scales: tuple[float, ...] = (1.0, 1.0, 1.0), overrides: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three provider-shaped frames. ``scales`` multiplies each period.

    ``scales[0]`` is the newest period, matching the provider's column order.
    An override applies to the newest period only, which is how a single
    missing or altered line item is tested.
    """
    base = dict(FLAT)
    base.update(overrides or {})

    def build(rows: tuple[str, ...]) -> pd.DataFrame:
        data = {}
        for i, period in enumerate(PERIODS[: len(scales)]):
            factor = scales[i]
            data[period] = [
                (base[r] * factor if r in (overrides or {}) and i == 0 else FLAT[r] * factor)
                if base.get(r) is not None
                else float("nan")
                for r in rows
            ]
        frame = pd.DataFrame(data, index=list(rows))
        for r in overrides or {}:
            if r in frame.index:
                frame.loc[r, PERIODS[0]] = overrides[r]
        return frame

    return build(BALANCE_ROWS), build(INCOME_ROWS), build(CASH_ROWS)


def flat_statements(overrides: dict | None = None) -> fu.Statements:
    bs, inc, cf = frames((1.0, 1.0, 1.0), overrides)
    return fu.read_statements(bs, inc, cf, ticker="FLAT.NS")


#: A company improving on all nine Piotroski dimensions, newest period first.
#:
#: Built explicitly rather than by scaling FLAT, because scaling every line by
#: one factor leaves every *ratio* unchanged -- margin, turnover and the
#: current ratio all stay flat, so five of the nine signals have no direction
#: and "improving on every dimension" is not what the fixture expresses. Each
#: series below is chosen so its signal has a definite, hand-checkable sign.
IMPROVING: dict[str, tuple[float, float, float]] = {
    # Assets grow slowly; earnings and cash grow faster -> ROA rises.
    "Total Assets": (1_000_000.0, 950_000.0, 900_000.0),
    "Net Income": (100_000.0, 80_000.0, 60_000.0),
    "Operating Cash Flow": (150_000.0, 120_000.0, 100_000.0),
    # Long-term debt falls outright -> leverage/assets falls.
    "Long Term Debt": (100_000.0, 120_000.0, 140_000.0),
    # Current assets rise against flat current liabilities -> current ratio up.
    "Current Assets": (500_000.0, 400_000.0, 350_000.0),
    "Current Liabilities": (200_000.0, 200_000.0, 200_000.0),
    # No issuance.
    "Ordinary Shares Number": (10_000.0, 10_000.0, 10_000.0),
    # Gross profit grows faster than revenue -> margin up. Revenue grows
    # faster than assets -> turnover up.
    "Total Revenue": (1_000_000.0, 900_000.0, 900_000.0),
    "Gross Profit": (400_000.0, 300_000.0, 280_000.0),
}


def growing_statements() -> fu.Statements:
    """Newest period best on every dimension, so every signal has direction."""
    bs, inc, cf = frames((1.0, 1.0, 1.0))
    for frame in (bs, inc, cf):
        for label, series in IMPROVING.items():
            if label in frame.index:
                for i, value in enumerate(series):
                    frame.loc[label, PERIODS[i]] = value
    return fu.read_statements(bs, inc, cf, ticker="GROW.NS")


# ---------------------------------------------------------------------------
# reading statements
# ---------------------------------------------------------------------------
def test_the_vocabulary_maps_provider_labels_to_canonical_fields() -> None:
    st = flat_statements()

    assert st.latest.get("total_assets") == 1_000_000.0
    assert st.latest.get("operating_cash_flow") == 140_000.0
    assert st.latest.get("revenue") == 800_000.0


def test_periods_come_back_newest_first() -> None:
    """Matching the provider, and every score that needs a prior year takes
    ``periods[i + 1]`` on that basis."""
    st = flat_statements()

    assert st.n_periods == 3
    assert st.periods[0].period_end == date(2025, 3, 31)
    assert st.periods[-1].period_end == date(2023, 3, 31)


def test_five_periods_load_when_the_provider_supplies_them() -> None:
    bs, inc, cf = frames((1.0, 0.95, 0.9, 0.85, 0.8))

    assert fu.read_statements(bs, inc, cf).n_periods == 5


def test_an_alternate_label_is_used_when_the_first_is_absent() -> None:
    """US filings say "Receivables", Indian ones "Accounts Receivable"."""
    bs, inc, cf = frames()
    bs = bs.rename(index={"Accounts Receivable": "Receivables"})

    st = fu.read_statements(bs, inc, cf)

    assert st.latest.get("receivables") == 90_000.0


def test_a_row_present_but_blank_falls_through_to_the_alternate() -> None:
    bs, inc, cf = frames()
    bs.loc["Receivables"] = float("nan")

    st = fu.read_statements(bs, inc, cf)

    assert st.latest.get("receivables") == 90_000.0


def test_a_missing_field_stays_nan_rather_than_zero() -> None:
    """The whole module's premise: absent is not zero."""
    bs, inc, cf = frames()
    bs = bs.drop(index=["Retained Earnings"])

    st = fu.read_statements(bs, inc, cf)

    assert math.isnan(st.latest.get("retained_earnings"))


def test_no_statements_at_all_is_reported_not_raised() -> None:
    st = fu.read_statements(None, None, None, ticker="NOTHING")

    assert not st.ok
    assert st.n_periods == 0
    assert st.notices


def test_a_partial_statement_set_still_yields_periods() -> None:
    """A company whose cash-flow statement is missing still has a balance
    sheet worth scoring."""
    bs, inc, _cf = frames()

    st = fu.read_statements(bs, inc, None)

    assert st.n_periods == 3
    assert any("cash flow" in n for n in st.notices)
    assert math.isnan(st.latest.get("operating_cash_flow"))


def test_a_series_runs_oldest_first_for_plotting() -> None:
    series = flat_statements().series("revenue")

    assert len(series) == 3
    assert list(series.index) == [date(2023, 3, 31), date(2024, 3, 31), date(2025, 3, 31)]


def test_the_span_names_both_ends() -> None:
    assert flat_statements().span == "2023-03-31 to 2025-03-31"


# ---------------------------------------------------------------------------
# Piotroski
# ---------------------------------------------------------------------------
def test_a_company_improving_on_every_dimension_scores_nine() -> None:
    result = fu.piotroski(growing_statements())

    assert result.ok
    assert result.value == 9.0
    assert result.available == 9
    assert result.verdict == "strong"


def test_a_flat_company_fails_the_improvement_signals() -> None:
    """Nothing changed, so "improved" is false everywhere it is tested. The
    level signals still pass."""
    result = fu.piotroski(flat_statements())

    assert result.ok
    assert result.available == 9
    passed = {s.name for s in result.signals if s.passed}
    assert "roa_positive" in passed
    assert "cfo_positive" in passed
    assert "roa_improving" not in passed
    assert "margin_rising" not in passed


def test_return_on_assets_is_scaled_by_beginning_assets() -> None:
    """Piotroski scales by the PRIOR balance sheet. Using this year's is a
    common shortcut that shifts every ROA and flips the improvement signal."""
    st = growing_statements()
    result = fu.piotroski(st)
    roa_signal = next(s for s in result.signals if s.name == "roa_positive")

    # 100,000 net income over the PRIOR year's 950,000 assets = 10.5%.
    # Scaling by THIS year's 1,000,000 would give 10.0% -- the shortcut this
    # assertion exists to catch.
    assert "10.5%" in roa_signal.detail


def test_a_single_period_yields_only_the_level_signals() -> None:
    """Five of the nine need a prior year, so a newly listed company can score
    at most four — and the score says it is out of four."""
    bs, inc, cf = frames((1.0,))
    result = fu.piotroski(fu.read_statements(bs, inc, cf))

    assert result.available <= 4
    assert result.scale == str(result.available)
    assert any("No prior year" in n for n in result.notices)


def test_a_missing_signal_is_not_counted_as_a_failure() -> None:
    """The bug this module exists to avoid. A company that reports no
    long-term debt loses the leverage signal, and must not lose a point."""
    bs, inc, cf = frames()
    bs = bs.drop(index=["Long Term Debt"])
    result = fu.piotroski(fu.read_statements(bs, inc, cf))

    leverage = next(s for s in result.signals if s.name == "leverage_falling")
    assert leverage.passed is None
    assert result.available == 8
    assert result.scale == "8"
    assert any("not counted as a failure" in n for n in result.notices)


def test_a_partial_score_carries_no_verdict() -> None:
    """ "strong" and "weak" are thresholds on the full nine; quoting them on a
    partial score would be meaningless."""
    bs, inc, cf = frames()
    bs = bs.drop(index=["Long Term Debt"])

    assert fu.piotroski(fu.read_statements(bs, inc, cf)).verdict == ""


def test_every_signal_is_reported_even_when_it_could_not_run() -> None:
    """So the UI can show nine rows and mark the gaps, rather than silently
    showing eight."""
    bs, inc, cf = frames()
    bs = bs.drop(index=["Long Term Debt"])
    result = fu.piotroski(fu.read_statements(bs, inc, cf))

    assert len(result.signals) == 9
    assert sum(1 for s in result.signals if not s.available) == 1


def test_piotroski_with_no_statements_is_unavailable() -> None:
    result = fu.piotroski(fu.read_statements(None, None, None))

    assert not result.ok
    assert result.value is None


def test_a_period_index_past_the_end_is_refused() -> None:
    result = fu.piotroski(flat_statements(), index=99)

    assert not result.ok
    assert "outside" in result.notices[0]


# ---------------------------------------------------------------------------
# Altman
# ---------------------------------------------------------------------------
def test_the_emerging_market_z_score_matches_the_hand_computation() -> None:
    """X1 0.2, X2 0.3, X3 0.12, X4 500,000/450,000.

    6.56(0.2) + 3.26(0.3) + 6.72(0.12) + 1.05(1.11111) + 3.25 = 7.513067
    """
    result = fu.altman_z(flat_statements(), market_cap=500_000.0, country="India")

    assert result.ok
    assert result.value == pytest.approx(7.513067, abs=1e-5)
    assert "EM" in result.verdict


def test_the_developed_market_variant_omits_the_constant() -> None:
    """Applying the +3.25 emerging-market adjustment to a US issuer inflates
    the score against thresholds it was never calibrated for."""
    result = fu.altman_z(flat_statements(), market_cap=500_000.0, country="United States")

    assert result.value == pytest.approx(7.513067 - fu.ALTMAN_EM_CONSTANT, abs=1e-5)
    assert "EM" not in result.verdict


def test_the_variant_used_is_named_in_the_verdict() -> None:
    """A Z-Score quoted without its variant is unreadable."""
    for country in ("India", "United States"):
        assert (
            "non-manufacturing"
            in fu.altman_z(flat_statements(), 500_000.0, country=country).verdict
        )


def test_an_explicit_variant_overrides_the_country() -> None:
    result = fu.altman_z(
        flat_statements(), 500_000.0, country="United States", emerging_market=True
    )

    assert "EM" in result.verdict


def test_an_unknown_country_is_treated_as_developed() -> None:
    """The conservative direction: it does NOT add 3.25."""
    result = fu.altman_z(flat_statements(), 500_000.0, country="")

    assert result.value == pytest.approx(7.513067 - fu.ALTMAN_EM_CONSTANT, abs=1e-5)


def test_without_a_market_capitalisation_the_score_is_withheld() -> None:
    """Book equity is NOT substituted: it measures something else, and
    relabelling it would be quietly wrong."""
    result = fu.altman_z(flat_statements(), market_cap=None)

    assert not result.ok
    assert result.available == 3
    assert any("book equity is not substituted" in n.lower() for n in result.notices)


def test_a_missing_term_withholds_the_whole_score() -> None:
    """A Z-Score missing a term is not a lower Z-Score."""
    bs, inc, cf = frames()
    bs = bs.drop(index=["Retained Earnings"])

    result = fu.altman_z(fu.read_statements(bs, inc, cf), market_cap=500_000.0)

    assert not result.ok
    assert result.available == 3


def test_working_capital_is_derived_when_not_reported() -> None:
    bs, inc, cf = frames()
    bs = bs.drop(index=["Working Capital"])

    result = fu.altman_z(fu.read_statements(bs, inc, cf), 500_000.0, country="India")

    # 400,000 current assets - 200,000 current liabilities = the reported figure.
    assert result.value == pytest.approx(7.513067, abs=1e-5)


def test_a_healthy_company_lands_in_the_safe_zone() -> None:
    assert "safe zone" in fu.altman_z(flat_statements(), 500_000.0, country="India").verdict


def test_a_weak_company_lands_in_the_distress_zone() -> None:
    """A tiny market value alone is not enough: the flat company's retained
    earnings and EBIT keep it safe. Distress needs the operating terms weak
    too, which is what makes the thresholds meaningful rather than a proxy
    for market cap."""
    bs, inc, cf = frames(
        overrides={
            "Working Capital": -300_000.0,
            "Retained Earnings": -400_000.0,
            "EBIT": -150_000.0,
            "Operating Income": -150_000.0,
        }
    )

    result = fu.altman_z(fu.read_statements(bs, inc, cf), 10_000.0, country="India")

    assert "distress zone" in result.verdict


# ---------------------------------------------------------------------------
# Beneish
# ---------------------------------------------------------------------------
def test_the_m_score_matches_the_hand_computation_on_a_flat_company() -> None:
    """With both years identical every index is exactly 1.0, so

        M = -4.84 + (0.920+0.528+0.404+0.892+0.115-0.172-0.327) + 4.679(TATA)
          = -4.84 + 2.360 + 4.679(-0.06) = -2.760740

    which pins all eight published coefficients in one assertion.
    TATA = (80,000 - 140,000) / 1,000,000 = -0.06.
    """
    result = fu.beneish_m(flat_statements())

    assert result.ok
    assert result.value == pytest.approx(-2.760740, abs=1e-6)


def test_every_index_is_one_on_a_flat_company() -> None:
    result = fu.beneish_m(flat_statements())

    for name in ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI"):
        assert result.inputs[name] == pytest.approx(1.0), name


def test_the_gross_margin_index_rises_when_the_margin_deteriorates() -> None:
    """GMI is PRIOR over CURRENT. Getting the order backwards inverts the
    signal, which is a classic implementation error."""
    bs, inc, cf = frames(overrides={"Gross Profit": 140_000.0})

    result = fu.beneish_m(fu.read_statements(bs, inc, cf))

    assert result.inputs["GMI"] > 1.0


def test_the_depreciation_index_rises_when_the_rate_slows() -> None:
    """A slowing depreciation rate flatters earnings, so DEPI should rise."""
    bs, inc, cf = frames(overrides={"Depreciation And Amortization": 20_000.0})

    result = fu.beneish_m(fu.read_statements(bs, inc, cf))

    assert result.inputs["DEPI"] > 1.0


def test_the_threshold_is_the_published_minus_one_point_seven_eight() -> None:
    bs, inc, cf = frames(overrides={"Net Income": 400_000.0})

    result = fu.beneish_m(fu.read_statements(bs, inc, cf))

    assert result.value > -1.78
    assert "above the -1.78 threshold" in result.verdict


def test_one_year_of_statements_yields_no_m_score() -> None:
    bs, inc, cf = frames((1.0,))

    result = fu.beneish_m(fu.read_statements(bs, inc, cf))

    assert not result.ok
    assert "two consecutive" in result.notices[0].lower()


def test_a_partial_m_score_is_never_reported() -> None:
    """The published coefficients are only meaningful together."""
    bs, inc, cf = frames()
    inc = inc.drop(index=["Selling General And Administration"])

    result = fu.beneish_m(fu.read_statements(bs, inc, cf))

    assert not result.ok
    assert result.available == 7
    assert "SGAI" in result.notices[0]


def test_the_m_score_says_it_is_a_screen_not_an_accusation() -> None:
    result = fu.beneish_m(flat_statements())

    joined = " ".join(result.notices).lower()
    assert "not an accusation" in joined
    assert "false-positive" in joined


# ---------------------------------------------------------------------------
# accruals
# ---------------------------------------------------------------------------
def test_the_accruals_ratio_matches_the_hand_computation() -> None:
    """(80,000 - 140,000) / 1,000,000 = -6.0%."""
    result = fu.accruals_ratio(flat_statements())

    assert result.ok
    assert result.value == pytest.approx(-6.0, abs=1e-9)


def test_a_modest_accruals_gap_reads_as_cash_backed() -> None:
    """-6% is inside the +/-10% band: earnings and cash broadly agree, and
    calling that "cash ahead" would over-read a small difference."""
    assert "broadly cash-backed" in fu.accruals_ratio(flat_statements()).verdict


def test_cash_well_ahead_of_earnings_is_described_as_such() -> None:
    bs, inc, cf = frames(overrides={"Operating Cash Flow": 300_000.0})

    result = fu.accruals_ratio(fu.read_statements(bs, inc, cf))

    assert result.value == pytest.approx(-22.0, abs=1e-9)
    assert "cash ahead of earnings" in result.verdict


def test_earnings_running_ahead_of_cash_is_flagged() -> None:
    bs, inc, cf = frames(overrides={"Net Income": 300_000.0})

    result = fu.accruals_ratio(fu.read_statements(bs, inc, cf))

    assert "earnings running ahead of cash" in result.verdict


def test_accruals_without_cash_flow_are_unavailable() -> None:
    bs, inc, _cf = frames()

    result = fu.accruals_ratio(fu.read_statements(bs, inc, None))

    assert not result.ok


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------
def test_the_ratios_match_hand_computations() -> None:
    r = fu.ratios(flat_statements())

    assert r.get("gross_margin") == pytest.approx(280_000 / 800_000)
    assert r.get("current_ratio") == pytest.approx(2.0)
    assert r.get("debt_to_equity") == pytest.approx(220_000 / 550_000)
    assert r.get("interest_cover") == pytest.approx(120_000 / 15_000)
    assert r.get("cash_conversion") == pytest.approx(140_000 / 80_000)
    assert r.get("effective_tax_rate") == pytest.approx(25_000 / 105_000)


def test_capex_is_used_as_an_outflow_magnitude() -> None:
    """Capex arrives negative. A sign error here silently inverts the ratio."""
    r = fu.ratios(flat_statements())

    assert r.get("capex_to_revenue") == pytest.approx(80_000 / 800_000)
    assert r.get("capex_to_revenue") > 0


def test_interest_cover_is_unavailable_not_infinite() -> None:
    """A company reporting no interest expense has an UNDEFINED cover. Showing
    infinity would flatter a company whose filing is merely incomplete."""
    bs, inc, cf = frames()
    inc = inc.drop(index=["Interest Expense"])

    r = fu.ratios(fu.read_statements(bs, inc, cf))

    assert math.isnan(r.get("interest_cover"))
    assert any("not the same as infinite" in n for n in r.notices)


def test_a_zero_denominator_yields_nan_not_infinity() -> None:
    bs, inc, cf = frames(overrides={"Total Revenue": 0.0})

    r = fu.ratios(fu.read_statements(bs, inc, cf))

    assert math.isnan(r.get("gross_margin"))


def test_returns_use_average_balances_when_a_prior_year_exists() -> None:
    """Flat statements make the average equal the closing balance, so this
    pins the formula rather than the data."""
    r = fu.ratios(flat_statements())

    assert r.get("roa") == pytest.approx(80_000 / 1_000_000)


def test_a_single_period_says_it_used_closing_balances() -> None:
    bs, inc, cf = frames((1.0,))

    r = fu.ratios(fu.read_statements(bs, inc, cf))

    assert any("closing rather than average" in n for n in r.notices)


def test_the_ratio_set_counts_what_it_could_compute() -> None:
    assert fu.ratios(flat_statements()).available >= 14


# ---------------------------------------------------------------------------
# horizon availability -- the honest answer about forecasting
# ---------------------------------------------------------------------------
def test_five_annual_periods_support_the_short_horizons_only() -> None:
    """The arithmetic behind the whole feature: five periods give four
    one-year pairs, two three-year pairs, and nothing at all from five out."""
    bs, inc, cf = frames((1.0, 0.95, 0.9, 0.85, 0.8))
    rows = {r.horizon: r for r in fu.horizon_availability(fu.read_statements(bs, inc, cf))}

    assert rows["1 year"].pairs == 4
    assert rows["3 years"].pairs == 2
    assert rows["5 years"].pairs == 0
    assert rows["10 years"].pairs == 0


def test_long_horizons_have_no_training_pairs_at_all() -> None:
    """Not "a weak model" — a model fitted to zero observations."""
    st = flat_statements()

    for label in ("5 years", "7 years", "10 years", "11 years"):
        row = next(r for r in fu.horizon_availability(st) if r.horizon == label)
        assert row.pairs == 0
        assert row.available is False
        assert "zero observations" in row.reason


def test_every_requested_horizon_is_listed_rather_than_omitted() -> None:
    """Listing them with a reason is more useful than dropping them and
    leaving the reader wondering whether it was an oversight."""
    labels = [r.horizon for r in fu.horizon_availability(flat_statements())]

    assert labels == [label for label, _months in fu.RETURN_HORIZONS]
    assert "11 years" in labels


def test_a_usable_horizon_still_reports_its_biases() -> None:
    """Even where the arithmetic works, the restatement and survivorship
    problems travel with the answer."""
    bs, inc, cf = frames((1.0, 0.95, 0.9, 0.85, 0.8))
    rows = fu.horizon_availability(fu.read_statements(bs, inc, cf))
    one_year = next(r for r in rows if r.horizon == "1 year")

    assert one_year.available is True
    assert "restated" in one_year.reason
    assert "survivorship" in one_year.reason


def test_no_statements_means_no_horizon_is_available() -> None:
    rows = fu.horizon_availability(fu.read_statements(None, None, None))

    assert all(not r.available for r in rows)
    assert all("No financial statements" in r.reason for r in rows)


def test_a_sub_annual_horizon_still_needs_a_forward_year() -> None:
    """A one-month horizon cannot be labelled without a subsequent
    observation, so it is bounded by periods - 1, not by zero."""
    bs, inc, cf = frames((1.0,))
    rows = {r.horizon: r for r in fu.horizon_availability(fu.read_statements(bs, inc, cf))}

    assert rows["1 month"].pairs == 0


# ---------------------------------------------------------------------------
# the summary the UI consumes
# ---------------------------------------------------------------------------
def test_the_summary_carries_every_score() -> None:
    summary = fu.quality_summary(flat_statements(), 500_000.0, country="India")

    for score in summary.scores:
        assert score.name
    assert summary.ratios.available > 0
    assert summary.n_periods == 3


def test_the_summary_names_what_could_not_be_computed() -> None:
    summary = fu.quality_summary(flat_statements(), market_cap=None)

    assert "Altman Z-Score" in summary.unavailable


def test_the_summary_passes_the_country_to_altman() -> None:
    india = fu.quality_summary(flat_statements(), 500_000.0, country="India")
    us = fu.quality_summary(flat_statements(), 500_000.0, country="United States")

    assert india.altman.value > us.altman.value


def test_a_score_result_headline_reads_as_a_sentence() -> None:
    result = fu.piotroski(growing_statements())

    assert "Piotroski F-Score 9 / 9" in result.headline
    assert "strong" in result.headline


def test_an_unavailable_score_headline_says_why() -> None:
    result = fu.altman_z(flat_statements(), market_cap=None)

    assert "not available" in result.headline
    assert "of 4 inputs" in result.headline
