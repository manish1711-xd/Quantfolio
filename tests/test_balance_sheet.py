"""The balance sheet as a statement, and the one identity it cannot fail.

Two properties carry this module.

:func:`test_a_line_the_provider_omitted_keeps_its_row` is the disclosure rule.
A statement that silently drops the rows it could not find leaves the reader
unable to tell "this company reports no inventory" from "we could not find the
inventory row", and those are completely different facts.

:func:`test_a_stable_positive_gap_is_diagnosed_as_minority_interest` is the
reason the identity check classifies instead of just passing or failing. Every
Indian holding company shows a multi-percent residual because the provider
excludes minority interest from *both* of its figures; reporting that as "does
not balance" would cry wolf on a large share of the NSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import fundamentals as fun

#: Three fiscal years, newest first as the provider returns them.
PERIODS = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31"])


def _frame(**rows: list[float]) -> pd.DataFrame:
    """A provider-shaped balance sheet: line labels down, periods across.

    The period count follows the longest row supplied, so a test that needs
    only one year says so with a one-element list rather than padding.
    """
    width = max((len(v) for v in rows.values()), default=0)
    return pd.DataFrame(rows, index=PERIODS[:width]).T


def _statements(frame: pd.DataFrame) -> fun.Statements:
    return fun.read_statements(frame, pd.DataFrame(), pd.DataFrame(), "TEST.NS", "INR")


@pytest.fixture
def balanced() -> fun.Statements:
    """A company that balances exactly: assets == liabilities + equity."""
    return _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0, 800.0],
                "Current Assets": [400.0, 360.0, 320.0],
                "Current Liabilities": [200.0, 180.0, 160.0],
                "Total Liabilities Net Minority Interest": [550.0, 500.0, 450.0],
                "Long Term Debt": [300.0, 280.0, 250.0],
                "Total Debt": [420.0, 390.0, 350.0],
                "Stockholders Equity": [450.0, 400.0, 350.0],
                "Retained Earnings": [250.0, 220.0, 190.0],
                "Net PPE": [500.0, 450.0, 400.0],
                "Cash And Cash Equivalents": [120.0, 100.0, 90.0],
                "Ordinary Shares Number": [10.0, 10.0, 10.0],
            }
        )
    )


@pytest.fixture
def holding_company() -> fun.Statements:
    """Assets exceed liabilities + equity by a stable ~8% — minority interest."""
    return _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0, 800.0],
                "Total Liabilities Net Minority Interest": [520.0, 468.0, 416.0],
                "Stockholders Equity": [400.0, 360.0, 320.0],
                "Current Assets": [400.0, 360.0, 320.0],
                "Current Liabilities": [200.0, 180.0, 160.0],
            }
        )
    )


# ---------------------------------------------------------------------------
# the statement
# ---------------------------------------------------------------------------
def test_the_statement_reads_newest_first_with_one_column_per_period(
    balanced: fun.Statements,
) -> None:
    frame = fun.balance_sheet_statement(balanced)
    periods = [c for c in frame.columns if c not in fun.BALANCE_SHEET_LAYOUT[0]]
    period_columns = [c for c in frame.columns if str(c).startswith("20")]

    assert period_columns == ["2026-03-31", "2025-03-31", "2024-03-31"]
    assert len(frame) == len(fun.BALANCE_SHEET_LAYOUT)
    assert set(frame["section"]) == {"Assets", "Liabilities", "Equity", "Memo"}
    assert periods  # the metadata columns survived


def test_every_figure_is_the_one_the_provider_reported(balanced: fun.Statements) -> None:
    """No rescaling, no sign flips, no rounding on the way through."""
    frame = fun.balance_sheet_statement(balanced).set_index("line")

    assert frame.loc["TOTAL ASSETS", "2026-03-31"] == pytest.approx(1000.0)
    assert frame.loc["TOTAL LIABILITIES", "2026-03-31"] == pytest.approx(550.0)
    assert frame.loc["TOTAL EQUITY", "2026-03-31"] == pytest.approx(450.0)
    assert frame.loc["Cash and cash equivalents", "2025-03-31"] == pytest.approx(100.0)
    assert frame.loc["Net property, plant & equipment", "2024-03-31"] == pytest.approx(400.0)


def test_a_line_the_provider_omitted_keeps_its_row(balanced: fun.Statements) -> None:
    """Missing is NaN in every period — and the row stays, as the disclosure.

    Dropping it would shorten the statement silently and destroy the reader's
    ability to distinguish "not reported" from "not found".
    """
    frame = fun.balance_sheet_statement(balanced).set_index("line")

    assert "Inventory" in frame.index, "an unreported line must still appear"
    inventory = frame.loc["Inventory", ["2026-03-31", "2025-03-31", "2024-03-31"]]
    assert inventory.isna().all()


def test_totals_are_flagged_so_the_ui_can_emphasise_them(balanced: fun.Statements) -> None:
    frame = fun.balance_sheet_statement(balanced)
    totals = set(frame.loc[frame["is_total"], "line"])

    assert "TOTAL ASSETS" in totals
    assert "TOTAL LIABILITIES" in totals
    assert "TOTAL EQUITY" in totals
    assert "Cash and cash equivalents" not in totals, "a component is not a total"
    assert totals == set(fun.BALANCE_SHEET_TOTALS)


def test_indent_level_encodes_subordination_not_decoration(
    balanced: fun.Statements,
) -> None:
    """A level-1 row is a component; a level-0 row is the subtotal or total."""
    frame = fun.balance_sheet_statement(balanced).set_index("line")

    assert frame.loc["Cash and cash equivalents", "level"] == 1
    assert frame.loc["Total current assets", "level"] == 0
    assert frame.loc["TOTAL ASSETS", "level"] == 0
    assert frame.loc["Long-term debt", "level"] == 1


def test_working_capital_is_derived_only_when_absent_and_says_so() -> None:
    """The one computed line, flagged so it cannot be mistaken for reported."""
    supplied = _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0, 800.0],
                "Current Assets": [400.0, 360.0, 320.0],
                "Current Liabilities": [150.0, 140.0, 130.0],
                "Working Capital": [111.0, 111.0, 111.0],
            }
        )
    )
    frame = fun.balance_sheet_statement(supplied).set_index("line")
    assert frame.loc["Working capital", "2026-03-31"] == pytest.approx(111.0)
    assert frame.loc["Working capital", "source"] == "reported"

    omitted = _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0, 800.0],
                "Current Assets": [400.0, 360.0, 320.0],
                "Current Liabilities": [150.0, 140.0, 130.0],
            }
        )
    )
    derived = fun.balance_sheet_statement(omitted).set_index("line")
    assert derived.loc["Working capital", "2026-03-31"] == pytest.approx(250.0)
    assert derived.loc["Working capital", "source"] == "derived"


def test_the_period_count_is_capped_without_reordering(balanced: fun.Statements) -> None:
    frame = fun.balance_sheet_statement(balanced, max_periods=2)
    period_columns = [c for c in frame.columns if str(c).startswith("20")]
    assert period_columns == ["2026-03-31", "2025-03-31"], "newest two, still newest first"


@pytest.mark.parametrize("bad", [None, "not statements", fun.Statements()])
def test_no_statements_yields_an_empty_frame_rather_than_raising(bad: object) -> None:
    frame = fun.balance_sheet_statement(bad)  # type: ignore[arg-type]
    assert frame.empty
    assert list(frame.columns) == [
        "section",
        "line",
        "field",
        "level",
        "is_total",
        "source",
    ]


# ---------------------------------------------------------------------------
# the accounting identity
# ---------------------------------------------------------------------------
def test_a_company_that_balances_is_reported_as_balanced(balanced: fun.Statements) -> None:
    check = fun.balance_sheet_check(balanced)

    assert check["ok"] is True
    assert check["diagnosis"] == "balanced"
    assert check["n_balanced"] == check["n_checked"] == 3
    assert check["worst_residual_pct"] == pytest.approx(0.0, abs=1e-9)
    assert (check["rows"]["residual"].abs() < 1e-9).all()


def test_a_stable_positive_gap_is_diagnosed_as_minority_interest(
    holding_company: fun.Statements,
) -> None:
    """The case that would otherwise cry wolf on every Indian holding company.

    A residual that is positive in EVERY period and stable in size is minority
    interest falling out of the middle of the identity — structural, expected,
    and not a data fault. It must not be reported the same way as a gap that
    changes sign.
    """
    check = fun.balance_sheet_check(holding_company)

    assert check["ok"] is False, "it genuinely does not balance"
    assert check["diagnosis"] == "minority_interest"
    assert check["n_balanced"] == 0
    assert (check["rows"]["residual"] > 0).all()
    assert "minority interest" in check["reason"].lower()
    assert "suspicion" not in check["reason"].lower(), "this is not the alarming case"


def test_a_gap_that_changes_sign_is_diagnosed_as_inconsistent() -> None:
    """Nothing structural makes assets fall BELOW liabilities plus equity."""
    erratic = _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0, 800.0],
                "Total Liabilities Net Minority Interest": [520.0, 500.0, 400.0],
                "Stockholders Equity": [400.0, 460.0, 320.0],
            }
        )
    )
    check = fun.balance_sheet_check(erratic)

    assert check["diagnosis"] == "inconsistent"
    assert "suspicion" in check["reason"].lower()
    assert not (check["rows"]["residual"] > 0).all()


def test_the_residual_is_exactly_assets_minus_liabilities_plus_equity(
    holding_company: fun.Statements,
) -> None:
    rows = fun.balance_sheet_check(holding_company)["rows"]
    expected = rows["total_assets"] - (rows["total_liabilities"] + rows["equity"])

    pd.testing.assert_series_equal(rows["residual"], expected, check_names=False)
    pd.testing.assert_series_equal(
        rows["residual_pct"],
        rows["residual"] / rows["total_assets"] * 100.0,
        check_names=False,
    )


def test_the_tolerance_is_a_share_of_assets_not_an_absolute_amount() -> None:
    """A 1% gap is immaterial on a large book and fatal on a tiny one."""
    tolerance = fun.BALANCE_SHEET_TOLERANCE
    inside = _statements(
        _frame(
            **{
                "Total Assets": [1000.0],
                "Total Liabilities Net Minority Interest": [600.0],
                "Stockholders Equity": [400.0 - 1000.0 * tolerance * 0.5],
            }
        )
    )
    outside = _statements(
        _frame(
            **{
                "Total Assets": [1000.0],
                "Total Liabilities Net Minority Interest": [600.0],
                "Stockholders Equity": [400.0 - 1000.0 * tolerance * 2.0],
            }
        )
    )
    assert fun.balance_sheet_check(inside)["ok"] is True
    assert fun.balance_sheet_check(outside)["ok"] is False


def test_a_period_missing_any_of_the_three_totals_is_skipped_not_scored() -> None:
    """Two of three figures cannot test an identity with three terms."""
    partial = _statements(
        _frame(
            **{
                "Total Assets": [1000.0, 900.0],
                "Total Liabilities Net Minority Interest": [550.0, np.nan],
                "Stockholders Equity": [450.0, 400.0],
            }
        )
    )
    check = fun.balance_sheet_check(partial)

    assert check["n_checked"] == 1, "only the complete period is testable"
    assert list(check["rows"]["period"]) == ["2026-03-31"]


@pytest.mark.parametrize("bad", [None, "not statements", fun.Statements()])
def test_no_statements_yields_a_reason_rather_than_raising(bad: object) -> None:
    check = fun.balance_sheet_check(bad)  # type: ignore[arg-type]

    assert check["ok"] is False
    assert check["diagnosis"] == "unavailable"
    assert check["reason"]
    assert check["n_checked"] == 0


def test_a_company_reporting_none_of_the_three_totals_says_it_could_not_test() -> None:
    nothing = _statements(_frame(**{"Net PPE": [500.0, 450.0]}))
    check = fun.balance_sheet_check(nothing)

    assert check["diagnosis"] == "unavailable"
    assert "could not be tested" in check["reason"]


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def test_the_chart_draws_the_identity_as_two_bars(holding_company: fun.Statements) -> None:
    import src.charts as charts

    rows = fun.balance_sheet_check(holding_company)["rows"]
    fig = charts.balance_sheet_chart(rows, "₹")
    names = [t.name for t in fig.data]

    assert "Total assets" in names
    assert "Liabilities" in names
    assert "Equity" in names
    assert "Minority interest / unexplained" in names, "a positive gap must be drawn"
    assert "Chart unavailable" not in str(fig.to_dict())


def test_the_chart_omits_the_gap_segment_when_there_is_none(
    balanced: fun.Statements,
) -> None:
    """A zero-height hatched bar on a balanced company would invent a concern."""
    import src.charts as charts

    rows = fun.balance_sheet_check(balanced)["rows"]
    names = [t.name for t in charts.balance_sheet_chart(rows).data]

    assert "Minority interest / unexplained" not in names


def test_the_chart_runs_time_forwards_even_though_the_table_does_not(
    holding_company: fun.Statements,
) -> None:
    """A table is compared newest-first; a trend is read oldest-first."""
    import src.charts as charts

    rows = fun.balance_sheet_check(holding_company)["rows"]
    assert rows["period"].iloc[0] == "2026-03-31", "the frame is newest first"

    drawn = list(charts.balance_sheet_chart(rows).data[0].x)
    assert drawn == ["2024-03-31", "2025-03-31", "2026-03-31"]
