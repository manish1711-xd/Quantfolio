"""Smart CSV/broker import — ``src.portfolio_importer``, entirely offline.

The contract this module has to honour is unusually strict, because its output
becomes someone's portfolio and a plausible-looking wrong answer is worse than
a refusal.

Verified here
-------------
**Nothing is silently discarded.** The central invariant, asserted directly:
``n_rows + n_rejected == source_rows`` for every file shape the suite throws at
it — good rows, bad numbers, bad dates, bad actions, ISINs, blanks, all-bad
files. Every rejected row carries a reason. If a future change starts dropping
a row quietly, ``accounted_for`` goes false and these tests fail.

**Column detection.** Real export headers from Zerodha, Groww, Upstox and a
hand-made spreadsheet all map onto the canonical schema. Aliases beat token
similarity; explicit cost aliases beat the ambiguous bare ``price``; and the
blocklist means "Previous Closing Price", "LTP" and "Unrealized P&L" can NEVER
be read as a cost basis — mapping current market data onto ``buy_price`` would
report a 0% return on the whole portfolio and nothing would look wrong.

**Number and date coercion.** Thousands separators, currency symbols and
accounting negatives parse. A missing number is ``None``, never 0.0, because a
zero quantity and an absent quantity are different facts and only one of them
is a valid holding. Dates are day-first: ``03/04/2024`` is genuinely ambiguous
and reading it month-first would move an Indian trade by a month, silently
corrupting every holding period and the XIRR.

**Ticker resolution respects the file's home market.** A file of NSE symbols
resolves its bare ``INFY`` to ``INFY.NS``, not to the NYSE ADR that owns that
exact Yahoo symbol; a file of US symbols does the opposite. That inference is
majority-vote over the unambiguous rows, and reports "needs review" rather than
guessing when there is no majority.

**Rejected rows do not pollute the symbol report.** A row dropped for a missing
quantity must not also appear under "ticker needs review" — the user would go
hunting for a symbol problem that does not exist.

**Handoff shapes are what the rest of the app already reads.**
``to_holdings()`` output survives ``positions.clean_holdings`` and
``to_transactions()`` output survives ``lots.parse_transactions`` with no
losses, which is the real proof the schemas line up.

Deterministic, offline, no RNG, no network.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src import portfolio_importer as imp
from src.analytics import lots, positions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def csv_bytes(text: str) -> io.BytesIO:
    """A CSV body as an uploadable byte stream."""
    return io.BytesIO(text.strip().encode("utf-8"))


def holdings_of(text: str, **kwargs) -> imp.ImportPreview:
    return imp.import_holdings(csv_bytes(text), "holdings.csv", **kwargs)


def ledger_of(text: str, **kwargs) -> imp.ImportPreview:
    return imp.import_transactions(csv_bytes(text), "ledger.csv", **kwargs)


ZERODHA = """
Symbol,Quantity Available,Quantity Discrepant,Average Price,Previous Closing Price,Unrealized P&L
RELIANCE,90,0,"1,182.95","1,410.20",20452.50
INFY,85,0,1429.80,1550.00,9639.00
TCS,38,0,2815.97,3100.00,10832.14
"""

GROWW = """
Stock Name,ISIN,Quantity,Average buy price
Reliance Industries Ltd,INE002A01018,90,1182.95
Hindustan Unilever Ltd,INE030A01027,40,2450.10
"""

UPSTOX = """
Company Name,Instrument,Qty,Avg. Cost
Tata Motors Ltd,TATAMOTORS,120,650.00
Titan Company Ltd,TITAN,25,3100.50
"""

CANONICAL = """
ticker,quantity,buy_price
RELIANCE.NS,90,1182.95
AAPL,6,182.50
"""


# ---------------------------------------------------------------------------
# the central invariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        ZERODHA,
        GROWW,
        UPSTOX,
        CANONICAL,
        # one bad number
        "ticker,quantity,buy_price\nRELIANCE,90,1182.95\nBAD,,500",
        # one zero quantity
        "ticker,quantity,buy_price\nRELIANCE,90,1182.95\nBAD,0,500",
        # one negative price
        "ticker,quantity,buy_price\nRELIANCE,90,1182.95\nBAD,10,-5",
        # a blank ticker
        "ticker,quantity,buy_price\n,90,1182.95\nTCS,38,2815.97",
        # an ISIN
        "ticker,quantity,buy_price\nINE002A01018,90,1182.95",
        # every row bad
        "ticker,quantity,buy_price\nA,,\nB,,",
        # text where a number belongs
        "ticker,quantity,buy_price\nRELIANCE,ninety,1182.95",
    ],
)
def test_every_source_row_is_either_imported_or_explicitly_rejected(body: str) -> None:
    """The promise. No row may vanish without an explanation."""
    preview = holdings_of(body)
    assert preview.accounted_for, (
        f"{preview.source_rows} source rows became {preview.n_rows} imported "
        f"+ {preview.n_rejected} rejected"
    )


def test_every_rejected_row_carries_a_human_readable_reason() -> None:
    preview = holdings_of("ticker,quantity,buy_price\nRELIANCE,90,1182.95\nBAD,,500\nWORSE,10,0\n")
    assert preview.n_rejected == 2
    assert "_reason" in preview.rejected.columns
    for reason in preview.rejected["_reason"]:
        assert isinstance(reason, str) and len(reason) > 10, reason


def test_every_rejected_row_names_its_line_number_in_the_users_file() -> None:
    """1-based counting the header, which is what a spreadsheet shows."""
    preview = holdings_of("ticker,quantity,buy_price\nRELIANCE,90,1182.95\nBAD,,500\n")
    assert list(preview.rejected["_row"]) == [3]


def test_a_rejected_row_keeps_its_original_columns_so_it_can_be_fixed() -> None:
    preview = holdings_of("Symbol,Qty,Average Price\nRELIANCE,90,1182.95\nBAD,,500\n")
    assert {"Symbol", "Qty", "Average Price"} <= set(preview.rejected.columns)


# ---------------------------------------------------------------------------
# column detection — real broker headers
# ---------------------------------------------------------------------------
def test_a_zerodha_export_maps_without_renaming_anything() -> None:
    preview = holdings_of(ZERODHA)
    assert preview.mapping.mapping["ticker"] == "Symbol"
    assert preview.mapping.mapping["quantity"] == "Quantity Available"
    assert preview.mapping.mapping["buy_price"] == "Average Price"
    assert preview.n_rows == 3


def test_a_groww_export_maps_the_stock_name_column() -> None:
    preview = holdings_of(GROWW)
    assert preview.mapping.mapping["ticker"] == "Stock Name"
    assert preview.mapping.mapping["buy_price"] == "Average buy price"
    assert preview.n_rows == 2


def test_an_upstox_style_abbreviated_header_maps() -> None:
    """ "Qty" and "Avg. Cost" — punctuation is normalised away."""
    preview = holdings_of(UPSTOX)
    assert preview.mapping.mapping["quantity"] == "Qty"
    assert preview.mapping.mapping["buy_price"] == "Avg. Cost"
    assert preview.n_rows == 2


@pytest.mark.parametrize(
    "header",
    [
        "ticker",
        "symbol",
        "tradingsymbol",
        "Trading Symbol",
        "stock",
        "security",
        "instrument",
        "Scrip",
    ],
)
def test_every_documented_ticker_alias_is_recognised(header: str) -> None:
    preview = holdings_of(f"{header},quantity,buy_price\nRELIANCE,10,100\n")
    assert preview.mapping.mapping["ticker"] == header
    assert preview.n_rows == 1


@pytest.mark.parametrize("header", ["quantity", "qty", "QTY", "shares", "units", "No of Shares"])
def test_every_documented_quantity_alias_is_recognised(header: str) -> None:
    preview = holdings_of(f"ticker,{header},buy_price\nRELIANCE,10,100\n")
    assert preview.mapping.mapping["quantity"] == header
    assert float(preview.frame["quantity"].iloc[0]) == 10.0


@pytest.mark.parametrize(
    "header",
    [
        "buy_price",
        "average_price",
        "avg_price",
        "Average Buy Price",
        "purchase_price",
        "Avg. Cost",
        "Cost Price",
    ],
)
def test_every_documented_buy_price_alias_is_recognised(header: str) -> None:
    preview = holdings_of(f"ticker,quantity,{header}\nRELIANCE,10,100\n")
    assert preview.mapping.mapping["buy_price"] == header
    assert float(preview.frame["buy_price"].iloc[0]) == 100.0


@pytest.mark.parametrize(
    "header", ["date", "purchase_date", "transaction_date", "trade_date", "Buy Date"]
)
def test_every_documented_date_alias_is_recognised(header: str) -> None:
    preview = holdings_of(f"ticker,quantity,buy_price,{header}\nRELIANCE,10,100,2024-01-15\n")
    assert preview.mapping.mapping["purchase_date"] == header
    assert str(preview.frame["purchase_date"].iloc[0])[:10] == "2024-01-15"


def test_headers_are_matched_case_and_punctuation_insensitively() -> None:
    preview = holdings_of("TICKER ,  Qty. , AVG_PRICE \nRELIANCE,10,100\n")
    assert set(preview.mapping.mapping) >= {"ticker", "quantity", "buy_price"}
    assert preview.n_rows == 1


def test_an_unmapped_column_is_reported_as_ignored_not_dropped_silently() -> None:
    preview = holdings_of("ticker,quantity,buy_price,MyOwnNotes\nRELIANCE,10,100,hello\n")
    assert "MyOwnNotes" in preview.mapping.unmapped


def test_the_mapping_frame_shows_the_users_own_column_names_back() -> None:
    frame = holdings_of(ZERODHA).mapping.as_frame()
    assert "Quantity Available" in set(frame["Your column"])
    assert "quantity" in set(frame["Read as"])


# ---------------------------------------------------------------------------
# the blocklist — the sharpest mis-mapping risk
# ---------------------------------------------------------------------------
def test_previous_closing_price_is_never_read_as_a_cost_basis() -> None:
    """Mapping it onto buy_price would report a ~0% return on the portfolio."""
    preview = holdings_of(ZERODHA)
    assert preview.mapping.mapping["buy_price"] == "Average Price"
    assert float(preview.frame["buy_price"].iloc[0]) == 1182.95, "not the 1410.20 close"


@pytest.mark.parametrize(
    "blocked", ["LTP", "Last Price", "Current Price", "Market Price", "Closing Price", "P&L"]
)
def test_a_blocked_header_is_never_claimed_by_any_field(blocked: str) -> None:
    preview = holdings_of(f"ticker,quantity,{blocked}\nRELIANCE,10,100\n")
    assert "buy_price" in preview.mapping.missing
    assert blocked in preview.mapping.unmapped
    assert preview.fatal, "a file with no cost basis must be refused, not guessed at"


def test_an_explicit_cost_alias_beats_the_ambiguous_bare_price_column() -> None:
    """An export can carry both what you paid and what it is worth now."""
    preview = holdings_of("ticker,quantity,average_price,price\nRELIANCE,10,1100,1400\n")
    assert preview.mapping.mapping["buy_price"] == "average_price"
    assert float(preview.frame["buy_price"].iloc[0]) == 1100.0


def test_a_bare_price_column_is_used_when_it_is_the_only_candidate() -> None:
    preview = holdings_of("ticker,quantity,price\nRELIANCE,10,1100\n")
    assert preview.mapping.mapping["buy_price"] == "price"
    assert preview.n_rows == 1


def test_one_column_is_never_claimed_by_two_fields() -> None:
    preview = holdings_of(ZERODHA)
    claimed = list(preview.mapping.mapping.values())
    assert len(claimed) == len(set(claimed))


# ---------------------------------------------------------------------------
# similarity fallback
# ---------------------------------------------------------------------------
def test_an_unanticipated_header_can_still_match_by_token_overlap() -> None:
    preview = holdings_of("ticker,share_quantity,buy_price\nRELIANCE,10,100\n")
    assert preview.mapping.mapping["quantity"] == "share_quantity"


def test_a_similarity_match_is_flagged_for_review_not_presented_as_certain() -> None:
    """A guess about which column is your cost basis must be visible."""
    preview = holdings_of("ticker,share_quantity,buy_price\nRELIANCE,10,100\n")
    assert "quantity" in preview.mapping.fuzzy
    assert any("similarity" in line for line in preview.summary_lines())


def test_a_coincidental_header_is_not_matched() -> None:
    """Below half token agreement, a wrong guess is worse than no guess."""
    preview = holdings_of("ticker,quantity,favourite_colour\nRELIANCE,10,blue\n")
    assert "buy_price" in preview.mapping.missing


# ---------------------------------------------------------------------------
# number coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1234.56", 1234.56),
        ('"1,234.56"', 1234.56),
        ("₹1234.56", 1234.56),
        ('"₹ 1,234.56"', 1234.56),
        ("$1234.56", 1234.56),
        ("(120.50)", -120.50),
        ("1234", 1234.0),
    ],
)
def test_broker_formatted_numbers_parse(raw: str, expected: float) -> None:
    preview = holdings_of(f"ticker,quantity,buy_price\nRELIANCE,10,{raw}\n")
    if expected > 0:
        assert float(preview.frame["buy_price"].iloc[0]) == pytest.approx(expected)
    else:
        assert preview.n_rejected == 1, "a negative cost basis is not importable"


@pytest.mark.parametrize("blank", ["", " ", "-", "N/A", "NA", "nan", "None"])
def test_a_missing_number_is_missing_not_zero(blank: str) -> None:
    """A zero quantity and an absent quantity are different facts."""
    assert imp._clean_number(blank) is None


def test_a_zero_quantity_is_rejected_with_a_reason_naming_zero() -> None:
    preview = holdings_of("ticker,quantity,buy_price\nRELIANCE,0,1100\n")
    assert preview.n_rejected == 1
    assert "greater than zero" in preview.rejected["_reason"].iloc[0]


def test_a_boolean_is_not_silently_read_as_one_or_zero() -> None:
    assert imp._clean_number(True) is None
    assert imp._clean_number(False) is None


def test_an_infinite_value_is_rejected() -> None:
    assert imp._clean_number(float("inf")) is None
    assert imp._clean_number(float("nan")) is None


# ---------------------------------------------------------------------------
# date coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-01-15", "2024-01-15"),
        ("15-01-2024", "2024-01-15"),
        ("15/01/2024", "2024-01-15"),
        ("15-Jan-2024", "2024-01-15"),
        ("15 Jan 2024", "2024-01-15"),
        ("2024/01/15", "2024-01-15"),
        ("15.01.2024", "2024-01-15"),
    ],
)
def test_common_date_formats_parse(raw: str, expected: str) -> None:
    preview = holdings_of(f"ticker,quantity,buy_price,date\nRELIANCE,10,100,{raw}\n")
    assert str(preview.frame["purchase_date"].iloc[0])[:10] == expected


def test_an_ambiguous_date_is_read_day_first() -> None:
    """03/04/2024 is 3 April in an Indian export. Month-first would move it a month."""
    preview = holdings_of("ticker,quantity,buy_price,date\nRELIANCE,10,100,03/04/2024\n")
    assert str(preview.frame["purchase_date"].iloc[0])[:10] == "2024-04-03"


def test_an_unparseable_holdings_date_is_a_warning_not_a_rejection() -> None:
    """The date is optional for holdings, so losing the row would be wrong."""
    preview = holdings_of("ticker,quantity,buy_price,date\nRELIANCE,10,100,whenever\n")
    assert preview.n_rows == 1
    assert preview.n_rejected == 0
    assert any(i.column == "purchase_date" for i in preview.warnings())


def test_an_unparseable_trade_date_rejects_the_row() -> None:
    """FIFO lot matching is meaningless without a real date."""
    preview = ledger_of("ticker,date,action,quantity,price\nRELIANCE,whenever,BUY,10,100\n")
    assert preview.n_rows == 0
    assert "FIFO" in preview.rejected["_reason"].iloc[0]


def test_the_date_column_is_a_real_datetime_dtype() -> None:
    """An object column mixing Timestamps and None loses the .dt accessor."""
    preview = holdings_of(
        "ticker,quantity,buy_price,date\nRELIANCE,10,100,2024-01-15\nTCS,5,200,\n"
    )
    assert pd.api.types.is_datetime64_any_dtype(preview.frame["purchase_date"])


# ---------------------------------------------------------------------------
# ticker resolution
# ---------------------------------------------------------------------------
def test_bare_indian_symbols_gain_their_nse_suffix() -> None:
    preview = holdings_of(ZERODHA)
    assert list(preview.frame["ticker"]) == ["RELIANCE.NS", "INFY.NS", "TCS.NS"]


def test_a_company_name_resolves_to_a_symbol() -> None:
    preview = holdings_of(GROWW)
    assert list(preview.frame["ticker"]) == ["RELIANCE.NS", "HINDUNILVR.NS"]


def test_a_us_symbol_is_left_alone() -> None:
    preview = holdings_of("ticker,quantity,buy_price\nAAPL,6,182.50\nMSFT,10,410\n")
    assert list(preview.frame["ticker"]) == ["AAPL", "MSFT"]


def test_an_nse_majority_file_resolves_its_ambiguous_symbol_to_the_indian_line() -> None:
    """The whole point of the market inference: INFY here means INFY.NS."""
    preview = holdings_of(ZERODHA)
    assert preview.resolved["INFY"] == "INFY.NS"


def test_a_us_majority_file_leaves_its_ambiguous_symbol_on_the_us_venue() -> None:
    preview = holdings_of(
        "ticker,quantity,buy_price\nAAPL,6,182.50\nMSFT,10,410\nNVDA,5,120\nINFY,20,18\n"
    )
    assert preview.frame["ticker"].iloc[-1] == "INFY", "a US file means the ADR"


def test_an_ambiguous_symbol_is_reported_for_review_with_its_alternatives() -> None:
    preview = holdings_of(ZERODHA)
    assert "INFY" in preview.ambiguous
    # Every candidate is offered, and the NSE line and the US ADR are both
    # among them -- that pair is what makes the row genuinely ambiguous.
    # The exact SET is not pinned: since the universe is generated from the
    # exchange listings it also contains Infosys's BSE line, and a third
    # listing appearing is the universe getting more complete, not a defect.
    alternatives = set(preview.ambiguous["INFY"])
    assert {"INFY", "INFY.NS"} <= alternatives
    assert len(alternatives) >= 2
    assert "INFY" in preview.needs_review


def test_an_unknown_symbol_is_imported_and_flagged_never_dropped() -> None:
    """Absence from the curated file is not proof a symbol cannot be priced."""
    preview = holdings_of("ticker,quantity,buy_price\nSOMESMALLCAP,10,45\n")
    assert preview.n_rows == 1
    assert preview.frame["ticker"].iloc[0] == "SOMESMALLCAP"
    assert "SOMESMALLCAP" in preview.unresolved
    assert "SOMESMALLCAP" in preview.needs_review


def test_a_rejected_row_does_not_appear_in_the_symbol_review_list() -> None:
    """The regression: a row dropped for a bad quantity is not a symbol problem."""
    preview = holdings_of("ticker,quantity,buy_price\nRELIANCE,90,1100\nBADROW,,0\n")
    assert preview.n_rejected == 1
    assert "BADROW" not in preview.needs_review
    assert "BADROW" not in preview.unresolved


def test_resolution_can_be_switched_off_entirely() -> None:
    preview = holdings_of(ZERODHA, resolve_tickers=False)
    assert list(preview.frame["ticker"]) == ["RELIANCE", "INFY", "TCS"]
    assert preview.resolved == {}


def test_a_known_isin_resolves_to_its_symbol() -> None:
    """This file used to be a dead end. It now imports."""
    preview = holdings_of("ISIN,Units,Purchase Price\nINE002A01018,90,1182.95\n")

    assert preview.n_rows == 1
    assert preview.frame["ticker"].iloc[0] == "RELIANCE.NS"
    assert preview.isin_resolved == {"INE002A01018": "RELIANCE.NS"}


def test_an_unknown_isin_is_refused_with_an_actionable_reason() -> None:
    """Resolution is table-driven, never guessed: an ISIN the table does not
    hold is reported, because a fabricated symbol would be far worse."""
    preview = holdings_of("ISIN,Units,Purchase Price\nGB0002634946,90,100\n")

    assert preview.n_rows == 0
    assert preview.isin_unresolved == ("GB0002634946",)
    reason = preview.rejected["_reason"].iloc[0]
    assert "not in the local ISIN table" in reason


def test_a_ticker_that_merely_looks_isin_shaped_is_not_refused() -> None:
    """The pattern is 2 letters + 9 alphanumerics + a check digit, nothing looser."""
    preview = holdings_of("ticker,quantity,buy_price\nBAJAJ-AUTO,5,9000\n")
    assert preview.n_rows == 1


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------
LEDGER = """
tradingsymbol,transaction_date,trade_type,qty,price,brokerage
RELIANCE,14-03-2023,buy,50,"1,120.00",140.00
INFY,21-04-2023,B,60,1360.00,204.00
INFY,01-07-2024,sell,4,1806.00,27.10
"""


def test_a_messy_indian_ledger_imports_cleanly() -> None:
    preview = ledger_of(LEDGER)
    assert preview.n_rows == 3
    assert list(preview.frame["action"]) == ["BUY", "BUY", "SELL"]


@pytest.mark.parametrize(
    "verb,expected",
    [
        ("BUY", "BUY"),
        ("buy", "BUY"),
        ("B", "BUY"),
        ("Bought", "BUY"),
        ("PURCHASE", "BUY"),
        ("SELL", "SELL"),
        ("sell", "SELL"),
        ("S", "SELL"),
        ("Sold", "SELL"),
        ("SALE", "SELL"),
    ],
)
def test_every_documented_trade_verb_maps(verb: str, expected: str) -> None:
    preview = ledger_of(f"ticker,date,action,quantity,price\nRELIANCE,2024-01-15,{verb},10,100\n")
    assert preview.frame["action"].iloc[0] == expected


def test_an_unrecognised_trade_verb_rejects_the_row_naming_the_verb() -> None:
    preview = ledger_of("ticker,date,action,quantity,price\nRELIANCE,2024-01-15,HOLD,10,100\n")
    assert preview.n_rows == 0
    assert "HOLD" in preview.rejected["_reason"].iloc[0]


def test_negative_fees_are_read_as_a_positive_cost_and_reported() -> None:
    """A fee is a cost, never a credit — and a flipped sign must be visible."""
    preview = ledger_of(
        "ticker,date,action,quantity,price,fees\nRELIANCE,2024-01-15,BUY,10,100,-25\n"
    )
    assert float(preview.frame["fees"].iloc[0]) == 25.0
    assert any(i.column == "fees" for i in preview.warnings())


def test_a_missing_fees_column_defaults_to_zero() -> None:
    preview = ledger_of("ticker,date,action,quantity,price\nRELIANCE,2024-01-15,BUY,10,100\n")
    assert float(preview.frame["fees"].iloc[0]) == 0.0


def test_no_currency_column_leaves_the_currency_blank_rather_than_guessing() -> None:
    """Stamping "INR" on a US broker's dollar trades would be a wrong label."""
    preview = ledger_of(
        "ticker,date,action,quantity,price\nAAPL,2024-01-15,BUY,10,172.50\n",
        base_currency="INR",
    )
    assert preview.frame["currency"].iloc[0] == ""


def test_a_currency_column_is_honoured_and_a_blank_cell_falls_back_to_the_base() -> None:
    preview = ledger_of(
        "ticker,date,action,quantity,price,currency\n"
        "AAPL,2024-01-15,BUY,10,172.50,USD\n"
        "TCS,2024-01-16,BUY,5,3000,\n",
        base_currency="INR",
    )
    assert list(preview.frame["currency"]) == ["USD", "INR"]


def test_a_ledger_needs_a_date_column_to_import_at_all() -> None:
    preview = ledger_of("ticker,action,quantity,price\nRELIANCE,BUY,10,100\n")
    assert "trade_date" in preview.mapping.missing
    assert preview.fatal


def test_a_ledger_needs_an_action_column_to_import_at_all() -> None:
    preview = ledger_of("ticker,date,quantity,price\nRELIANCE,2024-01-15,10,100\n")
    assert "action" in preview.mapping.missing
    assert preview.fatal


# ---------------------------------------------------------------------------
# whole-file failures
# ---------------------------------------------------------------------------
def test_a_missing_required_column_names_exactly_which_one() -> None:
    preview = holdings_of("Company,Sector,Weight\nReliance,Energy,12\n")
    assert set(preview.mapping.missing) == {"quantity", "buy_price"}
    assert "quantity" in preview.fatal and "buy_price" in preview.fatal
    assert not preview.ok


def test_a_whole_file_refusal_still_counts_as_accounted_for() -> None:
    """fatal explains every row at once; no row was processed individually."""
    preview = holdings_of("Company,Sector\nReliance,Energy\n")
    assert preview.accounted_for


def test_an_unreadable_upload_is_reported_not_raised() -> None:
    preview = imp.import_holdings(io.BytesIO(b"\x00\x01\x02\xff\xfe"), "junk.csv")
    assert not preview.ok
    assert preview.fatal


def test_an_empty_file_is_reported_not_raised() -> None:
    preview = imp.import_holdings(io.BytesIO(b""), "empty.csv")
    assert not preview.ok
    assert preview.fatal


def test_a_header_only_file_is_reported_as_having_no_data_rows() -> None:
    preview = holdings_of("ticker,quantity,buy_price")
    assert not preview.ok
    assert "no data" in preview.fatal.lower()


def test_a_file_where_every_row_fails_says_so_rather_than_reporting_success() -> None:
    preview = holdings_of("ticker,quantity,buy_price\nA,,\nB,,\n")
    assert not preview.ok
    assert preview.n_rejected == 2
    assert preview.fatal


def test_build_preview_of_none_is_a_fatal_preview_not_an_exception() -> None:
    preview = imp.build_preview(None, "holdings")
    assert not preview.ok
    assert preview.fatal


def test_build_preview_of_an_empty_frame_is_a_fatal_preview() -> None:
    preview = imp.build_preview(pd.DataFrame(), "holdings")
    assert not preview.ok


def test_detect_columns_of_none_reports_every_required_field_missing() -> None:
    assert set(imp.detect_columns(None, "holdings").missing) == set(imp.HOLDING_REQUIRED)


# ---------------------------------------------------------------------------
# reading quirks of real files
# ---------------------------------------------------------------------------
def test_a_semicolon_delimited_file_is_read() -> None:
    frame, error = imp.read_table(csv_bytes("ticker;quantity;buy_price\nRELIANCE;10;100"))
    assert not error
    assert list(frame.columns) == ["ticker", "quantity", "buy_price"]


def test_a_utf8_bom_does_not_mangle_the_first_column_name() -> None:
    body = ("ticker,quantity,buy_price\nRELIANCE,10,100\n").encode("utf-8-sig")
    preview = imp.import_holdings(io.BytesIO(body), "bom.csv")
    assert preview.n_rows == 1


def test_a_latin1_encoded_file_is_read_rather_than_refused() -> None:
    body = "ticker,quantity,buy_price,note\nRELIANCE,10,100,caf\xe9\n".encode("latin-1")
    preview = imp.import_holdings(io.BytesIO(body), "latin.csv")
    assert preview.n_rows == 1


def test_a_preamble_above_the_real_header_is_skipped() -> None:
    """Broker exports often start with an account-details block."""
    body = "Account Statement\nClient: XX1234\nticker,quantity,buy_price\nRELIANCE,10,100\n"
    preview = imp.import_holdings(csv_bytes(body), "preamble.csv")
    assert preview.n_rows == 1
    assert preview.frame["ticker"].iloc[0] == "RELIANCE.NS"


def test_a_pathlib_path_source_is_read(tmp_path: Path) -> None:
    """Every in-repo caller hands over a Path, not a stream."""
    target = tmp_path / "holdings.csv"
    target.write_text("ticker,quantity,buy_price\nRELIANCE,10,100\n", encoding="utf-8")

    preview = imp.import_holdings(target)

    assert preview.n_rows == 1
    assert preview.frame["ticker"].iloc[0] == "RELIANCE.NS"


def test_a_str_path_source_is_read(tmp_path: Path) -> None:
    target = tmp_path / "holdings.csv"
    target.write_text("ticker,quantity,buy_price\nTCS,5,2800\n", encoding="utf-8")

    assert imp.import_holdings(str(target)).n_rows == 1


def test_a_missing_path_is_reported_not_raised(tmp_path: Path) -> None:
    preview = imp.import_holdings(tmp_path / "nope.csv")

    assert not preview.ok
    assert preview.fatal


def test_an_unsupported_source_type_is_reported_not_raised() -> None:
    preview = imp.import_holdings(12345)

    assert not preview.ok
    assert preview.fatal


def test_an_excel_extension_is_honoured_even_from_a_bare_path(tmp_path: Path) -> None:
    """A .xlsx Path must not be silently parsed as CSV."""
    target = tmp_path / "book.xlsx"
    target.write_bytes(b"not really a workbook")

    preview = imp.import_holdings(target)

    assert not preview.ok, "a corrupt workbook must be refused, not read as text"


def test_an_excel_file_without_a_reader_is_reported_not_raised() -> None:
    frame, error = imp.read_table(io.BytesIO(b"not really xlsx"), "book.xlsx")
    assert frame.empty
    assert error


def test_duplicate_source_headers_do_not_produce_a_dataframe_column() -> None:
    """Two columns normalising to one name must not make df["ticker"] a frame."""
    preview = holdings_of("Ticker,ticker,quantity,buy_price\nRELIANCE,TCS,10,100\n")
    assert preview.n_rows == 1


# ---------------------------------------------------------------------------
# handoff to the rest of the application
# ---------------------------------------------------------------------------
def test_to_holdings_output_survives_clean_holdings_without_losses() -> None:
    """The real proof the schemas line up."""
    preview = holdings_of(ZERODHA)
    clean, problems = positions.clean_holdings(preview.to_holdings())
    assert len(clean) == preview.n_rows
    assert problems == []


def test_to_holdings_renames_the_date_to_the_column_clean_holdings_reads() -> None:
    preview = holdings_of("ticker,quantity,buy_price,date\nRELIANCE,10,100,2024-01-15\n")
    frame = preview.to_holdings()
    assert "buy_date" in frame.columns
    assert "purchase_date" not in frame.columns
    clean, _ = positions.clean_holdings(frame)
    assert "buy_date" in clean.columns


def test_to_holdings_omits_the_date_column_entirely_when_no_row_had_one() -> None:
    """An all-blank date column would only add noise to the grid."""
    frame = holdings_of(ZERODHA).to_holdings()
    assert "buy_date" not in frame.columns


def test_to_transactions_output_survives_parse_transactions_without_losses() -> None:
    preview = ledger_of(LEDGER)
    parsed, problems = lots.parse_transactions(preview.to_transactions())
    assert len(parsed) == preview.n_rows
    assert not [p for p in problems if "skipped" in p]


def test_a_full_ledger_round_trip_builds_fifo_lots() -> None:
    """End to end: broker CSV -> importer -> lots engine -> realised P&L."""
    preview = ledger_of(LEDGER)
    parsed, _ = lots.parse_transactions(preview.to_transactions())
    open_lots, realised, _ = lots.build_lots(parsed)
    assert open_lots, "two buys and a partial sell must leave open lots"
    assert len(realised) == 1
    assert realised[0].ticker == "INFY.NS"


def test_to_holdings_of_a_ledger_preview_is_empty_rather_than_wrong() -> None:
    assert holdings_of(ZERODHA).to_transactions().empty
    assert ledger_of(LEDGER).to_holdings().empty


def test_the_preview_frame_has_exactly_the_canonical_schema() -> None:
    assert tuple(holdings_of(ZERODHA).frame.columns) == imp.HOLDING_COLUMNS
    assert tuple(ledger_of(LEDGER).frame.columns) == imp.TRANSACTION_COLUMNS


# ---------------------------------------------------------------------------
# reporting surface
# ---------------------------------------------------------------------------
def test_the_summary_reports_the_recognised_count_the_ui_shows() -> None:
    lines = holdings_of(ZERODHA).summary_lines()
    assert any("3 holdings recognised" in line for line in lines)


def test_the_summary_flags_rejections_with_a_cross() -> None:
    lines = holdings_of("ticker,quantity,buy_price\nRELIANCE,10,100\nBAD,,0\n").summary_lines()
    assert any(line.startswith("✗") and "cannot be imported" in line for line in lines)


def test_the_summary_of_a_clean_file_carries_no_crosses() -> None:
    assert all(not line.startswith("✗") for line in holdings_of(CANONICAL).summary_lines())


def test_issues_are_partitioned_into_errors_and_warnings() -> None:
    preview = holdings_of("ticker,quantity,buy_price,date\nRELIANCE,10,100,junk\nBAD,,0,\n")
    assert len(preview.errors()) + len(preview.warnings()) == len(preview.issues)
    assert preview.errors() and preview.warnings()


def test_n_tickers_counts_distinct_symbols_not_rows() -> None:
    preview = holdings_of(
        "ticker,quantity,buy_price\nRELIANCE,10,100\nRELIANCE,5,110\nTCS,3,2800\n"
    )
    assert preview.n_rows == 3
    assert preview.n_tickers == 2


def test_a_row_issue_renders_with_its_line_number() -> None:
    issue = imp.RowIssue(7, "quantity", "", "is missing", "error")
    assert "Row 7" in str(issue) and "quantity" in str(issue)


# ---------------------------------------------------------------------------
# the files this repository actually ships
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_the_shipped_broker_export_sample_imports() -> None:
    """The demo file must actually demonstrate the feature.

    It is a deliberately awkward Zerodha-shaped file: thousands separators, an
    ISIN column, a "Previous Closing Price" column that must never be read as a
    cost basis, a symbol that lists on two exchanges, and one unusable row.
    """
    preview = imp.import_holdings(DATA_DIR / "sample_broker_export.csv")

    assert preview.ok
    assert preview.accounted_for
    assert preview.n_rows == 9
    assert preview.n_rejected == 1
    # Bare broker symbols gained their suffix; the US name did not.
    assert "RELIANCE.NS" in set(preview.frame["ticker"])
    assert "AAPL" in set(preview.frame["ticker"])
    # The cost basis is what was paid, never the previous close.
    reliance = preview.frame.loc[preview.frame["ticker"] == "RELIANCE.NS"]
    assert float(reliance["buy_price"].iloc[0]) == 1182.95


def test_the_shipped_broker_export_survives_clean_holdings() -> None:
    preview = imp.import_holdings(DATA_DIR / "sample_broker_export.csv")
    clean, problems = positions.clean_holdings(preview.to_holdings())

    assert len(clean) == preview.n_rows
    assert problems == []


def test_the_shipped_holdings_sample_imports() -> None:
    """The canonical-schema sample must keep working through the new path."""
    preview = imp.import_holdings(DATA_DIR / "sample_portfolio.csv")

    assert preview.ok
    assert preview.accounted_for
    assert preview.n_rejected == 0


def test_the_shipped_ledger_sample_imports_and_builds_lots() -> None:
    preview = imp.import_transactions(DATA_DIR / "sample_transactions.csv")

    assert preview.ok
    assert preview.accounted_for
    assert preview.n_rejected == 0

    parsed, _ = lots.parse_transactions(preview.to_transactions())
    assert len(parsed) == preview.n_rows
    open_lots, _realised, _ = lots.build_lots(parsed)
    assert open_lots


def test_the_shipped_samples_all_carry_symbols_the_universe_knows() -> None:
    """A demo file full of unresolvable symbols would teach the wrong lesson."""
    from src import stock_universe

    for name in ("sample_portfolio.csv", "sample_broker_export.csv"):
        preview = imp.import_holdings(DATA_DIR / name)
        unknown = [t for t in preview.frame["ticker"] if stock_universe.resolve(t) is None]
        assert not unknown, f"{name} carries symbols absent from the master files: {unknown}"


# ---------------------------------------------------------------------------
# manual column overrides
# ---------------------------------------------------------------------------
#: A header nobody could reasonably have an alias for. This is the dead end the
#: override feature exists to remove: detection cannot place `buy_price`, so
#: without a manual picker the user has to go rename a column in Excel.
UNANTICIPATED = """
Instrument,Units Held,Cost of acquisition per unit
RELIANCE,90,1182.95
TCS,38,2815.97
"""


def test_a_header_no_alias_covers_is_a_fatal_preview_before_any_override() -> None:
    preview = holdings_of(UNANTICIPATED)

    assert not preview.ok
    assert "buy_price" in preview.mapping.missing
    assert "Cost of acquisition per unit" in preview.mapping.unmapped


def test_an_override_recovers_that_file_in_place() -> None:
    """The headline behaviour: a dead end becomes a two-field correction."""
    preview = holdings_of(UNANTICIPATED)

    fixed = imp.remap(preview, {"buy_price": "Cost of acquisition per unit"})

    assert fixed.ok
    assert fixed.accounted_for
    assert fixed.n_rows == 2
    assert float(fixed.frame["buy_price"].iloc[0]) == 1182.95
    assert list(fixed.frame["ticker"]) == ["RELIANCE.NS", "TCS.NS"]


def test_an_override_reruns_every_per_row_validation() -> None:
    """remap re-derives the whole preview, it does not patch the old one."""
    preview = holdings_of("Instrument,Units Held,Cost per unit\nRELIANCE,90,1182.95\nBAD,10,\n")

    fixed = imp.remap(preview, {"buy_price": "Cost per unit"})

    assert fixed.n_rows == 1
    assert fixed.n_rejected == 1
    assert fixed.accounted_for, "the invariant must survive a remap"
    assert "buy price" in fixed.rejected["_reason"].iloc[0]


def test_an_override_reruns_ticker_resolution() -> None:
    preview = holdings_of(UNANTICIPATED)

    fixed = imp.remap(preview, {"buy_price": "Cost of acquisition per unit"})

    assert fixed.resolved["RELIANCE"] == "RELIANCE.NS"


def test_an_overridden_field_is_recorded_as_the_users_choice() -> None:
    fixed = imp.remap(holdings_of(UNANTICIPATED), {"buy_price": "Cost of acquisition per unit"})

    assert fixed.mapping.overridden == ("buy_price",)
    assert fixed.mapping.confidence_of("buy_price") == "you chose this"
    assert "you chose this" in set(fixed.mapping.as_frame()["Confidence"])


def test_an_overridden_field_stops_being_flagged_as_a_similarity_guess() -> None:
    """Continuing to say "needs review" after the user reviewed it trains them
    to ignore the flag."""
    preview = holdings_of("ticker,share_quantity,buy_price\nRELIANCE,10,100\n")
    assert "quantity" in preview.mapping.fuzzy

    fixed = imp.remap(preview, {"quantity": "share_quantity"})

    assert "quantity" not in fixed.mapping.fuzzy
    assert "quantity" in fixed.mapping.overridden


def test_the_summary_reports_that_a_column_was_mapped_by_hand() -> None:
    fixed = imp.remap(holdings_of(UNANTICIPATED), {"buy_price": "Cost of acquisition per unit"})

    assert any("mapped by you" in line for line in fixed.summary_lines())


def test_override_notes_appear_even_on_a_still_fatal_preview() -> None:
    """Someone who just re-pointed a column must see their choice was taken,
    even when the file remains unimportable for another reason."""
    preview = holdings_of(UNANTICIPATED)

    still_broken = imp.remap(preview, {"quantity": None})

    assert not still_broken.ok
    assert any("mapped by you" in line for line in still_broken.summary_lines())


# -- one column, one field --------------------------------------------------
def test_reassigning_a_column_takes_it_away_from_the_field_that_had_it() -> None:
    preview = holdings_of(ZERODHA)
    assert preview.mapping.source_of("buy_price") == "Average Price"

    stolen = imp.remap(preview, {"quantity": "Average Price"})

    assert stolen.mapping.source_of("quantity") == "Average Price"
    assert stolen.mapping.source_of("buy_price") is None
    assert "buy_price" in stolen.mapping.missing, "the displaced field must be visible"


def test_a_displaced_required_field_makes_the_preview_fatal_not_silently_wrong() -> None:
    stolen = imp.remap(holdings_of(ZERODHA), {"quantity": "Average Price"})

    assert not stolen.ok
    assert stolen.fatal


def test_no_two_canonical_fields_ever_read_the_same_column() -> None:
    stolen = imp.remap(holdings_of(ZERODHA), {"quantity": "Average Price"})

    sources = list(stolen.mapping.mapping.values())
    assert len(sources) == len(set(sources))


def test_the_unmapped_list_is_recomputed_after_an_override() -> None:
    fixed = imp.remap(holdings_of(UNANTICIPATED), {"buy_price": "Cost of acquisition per unit"})

    assert "Cost of acquisition per unit" not in fixed.mapping.unmapped


# -- unmapping and bad input ------------------------------------------------
@pytest.mark.parametrize("blank", [None, "", imp.NOT_PRESENT])
def test_an_override_of_nothing_unmaps_the_field(blank) -> None:
    unmapped = imp.remap(holdings_of(ZERODHA), {"buy_price": blank})

    assert unmapped.mapping.source_of("buy_price") is None
    assert "buy_price" in unmapped.mapping.missing


def test_unmapping_an_optional_field_is_not_fatal() -> None:
    preview = holdings_of("ticker,quantity,buy_price,date\nRELIANCE,10,100,2024-01-15\n")
    assert preview.mapping.source_of("purchase_date") == "date"

    without = imp.remap(preview, {"purchase_date": None})

    assert without.ok
    assert without.n_rows == 1
    assert "buy_date" not in without.to_holdings().columns


def test_naming_a_column_the_file_does_not_have_unmaps_rather_than_keeping_the_old_one() -> None:
    """You asked for a column that does not exist, so there is no column for
    that field — which surfaces loudly instead of ignoring your instruction."""
    broken = imp.remap(holdings_of(ZERODHA), {"buy_price": "No Such Column"})

    assert broken.mapping.source_of("buy_price") is None
    assert "buy_price" in broken.mapping.missing


def test_an_override_for_an_unknown_canonical_field_is_ignored() -> None:
    preview = holdings_of(ZERODHA)

    unchanged = imp.remap(preview, {"not_a_real_field": "Symbol"})

    assert unchanged.mapping.mapping == preview.mapping.mapping
    assert unchanged.mapping.overridden == ()


def test_an_override_for_a_field_of_the_other_kind_is_ignored() -> None:
    """`action` is a ledger field; a holdings preview must not grow one."""
    preview = holdings_of(ZERODHA)

    unchanged = imp.remap(preview, {"action": "Symbol"})

    assert "action" not in unchanged.mapping.mapping


def test_an_empty_override_dict_returns_the_very_same_preview() -> None:
    preview = holdings_of(ZERODHA)

    assert imp.remap(preview, {}) is preview


def test_remapping_a_preview_with_no_source_frame_returns_it_unchanged() -> None:
    """An unreadable upload has nothing to re-derive from."""
    preview = imp.import_holdings(io.BytesIO(b""), "empty.csv")
    assert preview.source_frame is None

    assert imp.remap(preview, {"buy_price": "whatever"}) is preview


# -- the blocklist, under an explicit human override ------------------------
def test_an_explicit_override_may_point_at_a_blocked_column() -> None:
    """The blocklist stops AUTOMATIC detection reading a market price as a cost
    basis. A human who deliberately picks that column may know something the
    alias table does not, so the choice is theirs."""
    forced = imp.remap(holdings_of(ZERODHA), {"buy_price": "Previous Closing Price"})

    assert forced.ok
    assert float(forced.frame["buy_price"].iloc[0]) == 1410.20


def test_a_blocked_override_is_reported_so_the_choice_is_not_hidden() -> None:
    forced = imp.remap(holdings_of(ZERODHA), {"buy_price": "Previous Closing Price"})

    assert forced.mapping.blocked() == ("buy_price",)
    assert any("CURRENT market data" in line for line in forced.summary_lines())


def test_an_unforced_mapping_reports_no_blocked_columns() -> None:
    assert holdings_of(ZERODHA).mapping.blocked() == ()


# -- ledgers ---------------------------------------------------------------
#: A ledger whose date, action and price headers are all unrecognisable.
OPAQUE_LEDGER = """
Instrument,When,Direction,Units,Rate per unit
RELIANCE,2023-01-10,BUY,10,1000
RELIANCE,2024-08-20,SELL,4,1400
"""


def test_an_opaque_ledger_names_every_field_it_could_not_place() -> None:
    preview = ledger_of(OPAQUE_LEDGER)

    assert set(preview.mapping.missing) == {"trade_date", "action", "price"}
    assert not preview.ok


def test_several_ledger_fields_can_be_overridden_in_one_go() -> None:
    fixed = imp.remap(
        ledger_of(OPAQUE_LEDGER),
        {"trade_date": "When", "action": "Direction", "price": "Rate per unit"},
    )

    assert fixed.ok
    assert fixed.n_rows == 2
    assert list(fixed.frame["action"]) == ["BUY", "SELL"]
    assert float(fixed.frame["price"].iloc[0]) == 1000.0
    assert str(fixed.frame["trade_date"].iloc[0])[:10] == "2023-01-10"


def test_an_overridden_ledger_still_parses_through_the_lots_engine() -> None:
    """End to end: an unmappable broker file becomes realised P&L."""
    fixed = imp.remap(
        ledger_of(OPAQUE_LEDGER),
        {"trade_date": "When", "action": "Direction", "price": "Rate per unit"},
    )

    parsed, _ = lots.parse_transactions(fixed.to_transactions())
    assert len(parsed) == 2
    _open_lots, realised, _ = lots.build_lots(parsed)
    assert len(realised) == 1
    assert realised[0].ticker == "RELIANCE.NS"


# -- redetect --------------------------------------------------------------
def test_redetect_discards_every_override() -> None:
    forced = imp.remap(holdings_of(ZERODHA), {"buy_price": "Previous Closing Price"})

    back = imp.redetect(forced)

    assert back.mapping.source_of("buy_price") == "Average Price"
    assert back.mapping.overridden == ()
    assert back.mapping.blocked() == ()


def test_redetect_of_a_preview_with_no_source_frame_returns_it_unchanged() -> None:
    preview = imp.import_holdings(io.BytesIO(b""), "empty.csv")

    assert imp.redetect(preview) is preview


# -- overrides compose ------------------------------------------------------
def test_successive_overrides_accumulate_rather_than_replacing_each_other() -> None:
    """The UI applies one form submission at a time; the record must build up."""
    first = imp.remap(holdings_of(UNANTICIPATED), {"buy_price": "Cost of acquisition per unit"})
    second = imp.remap(first, {"quantity": "Units Held"})

    assert set(second.mapping.overridden) == {"buy_price", "quantity"}
    assert second.ok


def test_an_override_dropped_for_naming_a_missing_column_is_not_claimed_as_chosen() -> None:
    """`overridden` must only ever list fields that actually ended up mapped —
    otherwise the table would show "you chose this" against a blank column."""
    broken = imp.remap(holdings_of(ZERODHA), {"buy_price": "No Such Column"})

    assert "buy_price" not in broken.mapping.overridden


# -- apply_overrides in isolation ------------------------------------------
def test_apply_overrides_is_usable_without_a_preview() -> None:
    mapping = imp.ColumnMapping(mapping={"ticker": "A", "quantity": "B"})

    out = imp.apply_overrides(mapping, {"buy_price": "C"}, ("A", "B", "C"), "holdings")

    assert out.mapping == {"ticker": "A", "quantity": "B", "buy_price": "C"}
    assert out.ok
    assert out.unmapped == ()


def test_apply_overrides_reports_unmapped_columns_in_file_order() -> None:
    mapping = imp.ColumnMapping(mapping={"ticker": "A"})

    out = imp.apply_overrides(mapping, {"quantity": "C"}, ("A", "B", "C", "D"), "holdings")

    assert out.unmapped == ("B", "D")


# ---------------------------------------------------------------------------
# ISIN -> ticker resolution
# ---------------------------------------------------------------------------
ISIN_ONLY = """
ISIN,Units,Purchase Price
INE002A01018,90,1182.95
INE040A01034,170,646.64
US0378331005,6,182.50
"""

BOTH_COLUMNS = """
Tradingsymbol,ISIN,Quantity,Average Price
TATAELXSI,INE670A01012,10,7000
PERSISTENT,INE262H01021,20,5500
"""


def test_an_isin_only_file_imports(isin_cache) -> None:
    """The dead end this feature removes."""
    preview = holdings_of(ISIN_ONLY)

    assert preview.ok
    assert preview.accounted_for
    assert list(preview.frame["ticker"]) == ["RELIANCE.NS", "HDFCBANK.NS", "AAPL"]


def test_the_resolved_isins_are_reported(isin_cache) -> None:
    preview = holdings_of(ISIN_ONLY)

    assert preview.isin_resolved == {
        "INE002A01018": "RELIANCE.NS",
        "INE040A01034": "HDFCBANK.NS",
        "US0378331005": "AAPL",
    }
    assert any("ISINs resolved" in line for line in preview.summary_lines())


def test_the_isin_is_carried_onto_the_imported_rows(isin_cache) -> None:
    """Useful provenance to keep alongside the symbol."""
    preview = holdings_of(ISIN_ONLY)

    assert list(preview.frame["isin"]) == [
        "INE002A01018",
        "INE040A01034",
        "US0378331005",
    ]


def test_a_mistyped_isin_is_rejected_with_the_expected_check_digit(isin_cache) -> None:
    """Arithmetic, not opinion: the row names the digit it should have had."""
    preview = holdings_of("ISIN,Units,Purchase Price\nINE002A01019,90,1182.95\n")

    assert preview.n_rows == 0
    reason = preview.rejected["_reason"].iloc[0]
    assert "check digit should be 8" in reason
    assert "typo" in reason


def test_a_mistyped_isin_is_not_reported_as_merely_unmappable(isin_cache) -> None:
    """A typo and an unlisted security need different fixes, so they are
    reported differently."""
    preview = holdings_of("ISIN,Units,Purchase Price\nINE002A01019,90,1182.95\n")

    assert preview.isin_unresolved == (), "a typo is not an unresolved ISIN"


def test_an_unresolvable_isin_is_listed_separately_from_ordinary_bad_rows(
    isin_cache,
) -> None:
    preview = holdings_of("ISIN,Units,Purchase Price\nGB0002634946,90,100\n")

    assert preview.isin_unresolved == ("GB0002634946",)
    assert preview.n_rejected == 1
    assert any("could not be mapped" in line for line in preview.summary_lines())


def test_an_unresolvable_isins_reason_names_its_country(isin_cache) -> None:
    preview = holdings_of("ISIN,Units,Purchase Price\nGB0002634946,90,100\n")

    assert "United Kingdom" in preview.rejected["_reason"].iloc[0]


def test_a_partly_resolvable_isin_file_imports_what_it_can(isin_cache) -> None:
    preview = holdings_of(
        "ISIN,Units,Purchase Price\nINE002A01018,90,1182.95\nGB0002634946,10,100\n"
    )

    assert preview.n_rows == 1
    assert preview.n_rejected == 1
    assert preview.accounted_for


# -- learning ---------------------------------------------------------------
def test_a_file_with_both_columns_teaches_the_isin_table(isin_cache) -> None:
    """The best source of mappings there is: the user's own broker data."""
    preview = holdings_of(BOTH_COLUMNS)

    assert preview.isin_learned == 2
    assert any("learned from this file" in line for line in preview.summary_lines())


def test_a_later_isin_only_export_resolves_what_was_learned(isin_cache) -> None:
    """The whole point of learning, end to end."""
    before = holdings_of("ISIN,Units,Purchase Price\nINE670A01012,10,7000\n")
    assert not before.ok, "should not resolve before the pairing is known"

    holdings_of(BOTH_COLUMNS)

    after = holdings_of("ISIN,Units,Purchase Price\nINE670A01012,10,7000\n")
    assert after.ok
    assert after.frame["ticker"].iloc[0] == "TATAELXSI.NS"


def test_learned_symbols_are_stored_in_their_resolved_yahoo_form(isin_cache) -> None:
    """A broker's bare "TATAELXSI" is stored as TATAELXSI.NS, so the learned
    mapping is directly usable rather than needing resolving again."""
    holdings_of(BOTH_COLUMNS)

    from src import isin as isin_mod

    assert isin_mod.resolve("INE670A01012").yahoo_symbol == "TATAELXSI.NS"


def test_learning_can_be_switched_off(isin_cache) -> None:
    preview = holdings_of(BOTH_COLUMNS, learn_isins=False)

    assert preview.isin_learned == 0
    assert not (isin_cache / "isin_learned.csv").exists()


def test_an_isin_only_file_teaches_nothing(isin_cache) -> None:
    """There is no symbol in it to pair the ISIN with."""
    preview = holdings_of(ISIN_ONLY)

    assert preview.isin_learned == 0


def test_a_ledger_with_both_columns_also_teaches(isin_cache) -> None:
    preview = ledger_of(
        "Tradingsymbol,ISIN,Date,Action,Qty,Price\nTATAELXSI,INE670A01012,2024-01-15,BUY,10,7000\n"
    )

    assert preview.ok
    assert preview.isin_learned == 1


# -- precedence -------------------------------------------------------------
def test_a_symbol_column_wins_over_an_isin_column(isin_cache) -> None:
    """ISIN resolution is a fallback, never a substitute for a real symbol."""
    preview = holdings_of(BOTH_COLUMNS)

    assert preview.mapping.source_of("ticker") == "Tradingsymbol"
    assert preview.mapping.source_of("isin") == "ISIN"
    assert preview.isin_resolved == {}, "no resolution was needed"


def test_the_isin_column_is_no_longer_reported_as_ignored(isin_cache) -> None:
    """It used to be dropped as an unmapped column; now it is read."""
    preview = holdings_of(BOTH_COLUMNS)

    assert "ISIN" not in preview.mapping.unmapped


def test_a_blank_symbol_falls_back_to_the_rows_isin(isin_cache) -> None:
    """Some exports leave the symbol empty for a few rows."""
    preview = holdings_of(
        "Tradingsymbol,ISIN,Quantity,Average Price\n"
        "RELIANCE,INE002A01018,90,1182.95\n"
        ",INE040A01034,170,646.64\n"
    )

    assert preview.n_rows == 2
    assert list(preview.frame["ticker"]) == ["RELIANCE.NS", "HDFCBANK.NS"]


def test_a_real_ticker_that_merely_has_isin_shape_is_not_read_as_one(
    isin_cache,
) -> None:
    """ "NOTANISIN123" is ISIN-shaped. In a column labelled Symbol it must be
    treated as a symbol, because mis-reading a ticker as an identifier would
    be worse than either failure."""
    preview = holdings_of("Symbol,Quantity,Average Price\nNOTANISIN123,10,100\n")

    assert preview.n_rows == 1
    assert preview.frame["ticker"].iloc[0] == "NOTANISIN123"
    assert preview.isin_unresolved == ()


def test_an_isin_in_a_symbol_column_is_still_resolved_when_it_validates(
    isin_cache,
) -> None:
    """A valid check digit is strong enough evidence on its own."""
    preview = holdings_of("Symbol,Quantity,Average Price\nINE002A01018,90,1182.95\n")

    assert preview.frame["ticker"].iloc[0] == "RELIANCE.NS"


# -- the isin column in the handoff ----------------------------------------
def test_the_isin_column_is_dropped_when_no_row_carries_one(isin_cache) -> None:
    """A file that never mentioned ISINs must not gain a blank column."""
    assert "isin" not in holdings_of(ZERODHA).to_holdings().columns


def test_the_isin_column_survives_into_the_holdings_handoff(isin_cache) -> None:
    assert "isin" in holdings_of(ISIN_ONLY).to_holdings().columns


def test_an_isin_carrying_handoff_still_survives_clean_holdings(isin_cache) -> None:
    """clean_holdings ignores columns outside its schema; this proves it."""
    preview = holdings_of(ISIN_ONLY)
    clean, problems = positions.clean_holdings(preview.to_holdings())

    assert len(clean) == preview.n_rows
    assert problems == []


def test_an_isin_carrying_ledger_still_parses_through_the_lots_engine(
    isin_cache,
) -> None:
    preview = ledger_of(
        "ISIN,Date,Action,Qty,Price\n"
        "INE002A01018,2023-01-10,BUY,10,1000\n"
        "INE002A01018,2024-08-20,SELL,4,1400\n"
    )
    assert preview.ok

    parsed, _ = lots.parse_transactions(preview.to_transactions())
    assert len(parsed) == 2
    _open_lots, realised, _ = lots.build_lots(parsed)
    assert len(realised) == 1
    assert realised[0].ticker == "RELIANCE.NS"


# -- the shipped sample ----------------------------------------------------
def test_the_shipped_broker_sample_teaches_its_isin_pairings(isin_cache) -> None:
    """It carries both columns, so importing it grows the local table."""
    preview = imp.import_holdings(DATA_DIR / "sample_broker_export.csv")

    assert preview.isin_learned > 0


def test_every_isin_in_the_shipped_sample_is_valid(isin_cache) -> None:
    """A demo file with a bad check digit would teach the wrong lesson."""
    from src import isin as isin_mod

    frame, error = imp.read_table(DATA_DIR / "sample_broker_export.csv")
    assert not error

    bad = [
        code for code in frame["ISIN"].dropna() if str(code).strip() and not isin_mod.is_valid(code)
    ]
    assert not bad, f"invalid ISINs in the shipped sample: {bad}"


def test_the_shipped_isin_only_sample_demonstrates_all_three_outcomes(isin_cache) -> None:
    """The demo file is built to show resolution, an unlisted ISIN and a typo,
    because a sample that only shows the happy path teaches nothing about the
    failure modes that matter."""
    preview = imp.import_holdings(DATA_DIR / "sample_isin_only.csv")

    assert preview.accounted_for
    assert preview.n_rows == 6, "six seeded ISINs resolve"
    assert preview.n_rejected == 2, "one unlisted ISIN and one typo"
    assert preview.isin_unresolved == ("GB0002634946",)

    reasons = " ".join(preview.rejected["_reason"])
    assert "not in the local ISIN table" in reasons
    assert "check digit should be 8" in reasons


def test_every_isin_in_the_isin_only_sample_is_deliberate(isin_cache) -> None:
    """Exactly one row is meant to be an invalid ISIN. If a second one stops
    validating, the file has drifted rather than the code."""
    from src import isin as isin_mod

    frame, error = imp.read_table(DATA_DIR / "sample_isin_only.csv")
    assert not error

    invalid = [c for c in frame["ISIN"] if not isin_mod.is_valid(c)]
    assert invalid == ["INE002A01019"], invalid
