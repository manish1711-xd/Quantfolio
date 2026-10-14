"""Tests for building the stock-master file from the exchange files.

Offline throughout: every parse and merge function takes rows and returns
rows, so the whole build is exercised against fixtures shaped like the real
downloads. :mod:`src.data.exchange` (the network half) is tested separately.

The properties that matter here are all about **not losing or inventing
instruments**:

* a curated row, its hand-written aliases and its previous name all survive a
  refresh,
* an instrument two independent sources agree has stopped trading is dropped
  and named, not silently kept,
* nothing is written unless the whole build succeeded, and the file being
  replaced is backed up first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src import universe_build as ub

# Rows shaped exactly like the real files, including NSE's leading-space
# header (already stripped by exchange._csv_rows) and BSE's JSON keys.
NSE_ROWS = [
    {
        "SYMBOL": "RELIANCE",
        "NAME OF COMPANY": "Reliance Industries Limited",
        "SERIES": "EQ",
        "ISIN NUMBER": "INE002A01018",
    },
    {
        "SYMBOL": "TCS",
        "NAME OF COMPANY": "Tata Consultancy Services Limited",
        "SERIES": "EQ",
        "ISIN NUMBER": "INE467B01029",
    },
    {
        "SYMBOL": "SOMESME",
        "NAME OF COMPANY": "Some SME Limited",
        "SERIES": "SM",
        "ISIN NUMBER": "INE000000001",
    },
]

BSE_ROWS = [
    {
        "SCRIP_CD": "500325",
        "scrip_id": "RELIANCE",
        "Scrip_Name": "RELIANCE",
        "Issuer_Name": "Reliance Industries Ltd",
        "ISIN_NUMBER": "INE002A01018",
        "INDUSTRY": "Refineries",
        "Status": "Active",
        "GROUP": "A",
    },
    {
        "SCRIP_CD": "500002",
        "scrip_id": "ABB",
        "Scrip_Name": "ABB",
        "Issuer_Name": "ABB India Limited",
        "ISIN_NUMBER": "INE117A01022",
        "INDUSTRY": "Heavy Electrical Equipment",
        "Status": "Active",
        "GROUP": "A",
    },
]

BHAVCOPY_ROWS = [
    {
        "TckrSymb": "RELIANCE",
        "SctySrs": "EQ",
        "TtlTrfVal": "17668058961.00",
        "ISIN": "INE002A01018",
    },
    {"TckrSymb": "TCS", "SctySrs": "EQ", "TtlTrfVal": "5923000000.00", "ISIN": "INE467B01029"},
    {"TckrSymb": "SGBJUN28", "SctySrs": "GB", "TtlTrfVal": "1532731.15", "ISIN": "IN0020200104"},
]

CURATED = [
    {
        "symbol": "RELIANCE",
        "yahoo_symbol": "RELIANCE.NS",
        "name": "Reliance Industries Ltd",
        "exchange": "NSE",
        "country": "India",
        "currency": "INR",
        "instrument_type": "EQUITY",
        "sector": "Energy",
        "aliases": "RIL|Reliance",
        "isin": "",
        "turnover": "",
    },
    {
        "symbol": "NIFTYBEES",
        "yahoo_symbol": "NIFTYBEES.NS",
        "name": "Nippon India ETF Nifty 50 BeES",
        "exchange": "NSE",
        "country": "India",
        "currency": "INR",
        "instrument_type": "ETF",
        "sector": "Index fund",
        "aliases": "Nifty BeES",
        "isin": "",
        "turnover": "",
    },
]


def nse_records() -> list[dict[str, str]]:
    return ub.parse_nse_listing(NSE_ROWS)[0]


def bse_records() -> list[dict[str, str]]:
    return ub.parse_bse_listing(BSE_ROWS)[0]


#: Turnover for every curated row, so a test about ONE stale symbol is not
#: quietly also dropping the ETF for want of a turnover figure.
ALIVE = {"NSE": {"RELIANCE": 1.0e9, "NIFTYBEES": 2.1e9}}


def by_symbol(rows) -> dict[str, dict[str, str]]:
    return {row["yahoo_symbol"]: row for row in rows}


# ---------------------------------------------------------------------------
# parsing the NSE listing
# ---------------------------------------------------------------------------
def test_the_nse_listing_becomes_ns_suffixed_records() -> None:
    """The suffix is the whole point: Yahoo cannot price a bare "RELIANCE"."""
    records = by_symbol(nse_records())

    assert records["RELIANCE.NS"]["symbol"] == "RELIANCE"
    assert records["RELIANCE.NS"]["exchange"] == "NSE"
    assert records["RELIANCE.NS"]["currency"] == "INR"
    assert records["RELIANCE.NS"]["country"] == "India"


def test_the_nse_listing_supplies_the_name_and_isin() -> None:
    record = by_symbol(nse_records())["RELIANCE.NS"]

    assert record["name"] == "Reliance Industries Limited"
    assert record["isin"] == "INE002A01018"


def test_a_non_equity_nse_series_is_skipped_with_a_reason() -> None:
    """SME and government-security series are not reliably priced, and an
    unpriceable search result is worse than a missing one."""
    _records, skipped = ub.parse_nse_listing(NSE_ROWS)

    assert ("SOMESME", "NSE series SM is not a priced equity series") in skipped


def test_only_the_priced_equity_series_survive() -> None:
    assert {r["symbol"] for r in nse_records()} == {"RELIANCE", "TCS"}


def test_a_listing_row_with_no_symbol_is_skipped() -> None:
    _records, skipped = ub.parse_nse_listing([{"SYMBOL": "", "NAME OF COMPANY": "X"}])

    assert skipped == [("", "no symbol")]


def test_the_legal_form_is_stripped_into_an_alias() -> None:
    """So "Reliance Industries" also finds it, not only the full legal name."""
    record = by_symbol(nse_records())["RELIANCE.NS"]

    assert "Reliance Industries" in record["aliases"]


def test_a_missing_series_is_kept_rather_than_guessed() -> None:
    records, skipped = ub.parse_nse_listing(
        [{"SYMBOL": "AAA", "NAME OF COMPANY": "AAA Ltd", "ISIN NUMBER": "INE000000001"}]
    )

    assert skipped == []
    assert records[0]["yahoo_symbol"] == "AAA.NS"


# ---------------------------------------------------------------------------
# parsing the BSE listing
# ---------------------------------------------------------------------------
def test_the_bse_listing_uses_the_trading_symbol_not_the_scrip_code() -> None:
    """Measured, and load-bearing: Yahoo prices ``RELIANCE.BO`` and returns
    nothing at all for ``500325.BO``. Getting this wrong would have produced
    thousands of rows whose prices never load."""
    records = by_symbol(bse_records())

    assert "RELIANCE.BO" in records
    assert "500325.BO" not in records
    assert records["RELIANCE.BO"]["symbol"] == "RELIANCE"


def test_a_bse_row_without_a_trading_symbol_is_skipped_with_its_code() -> None:
    _records, skipped = ub.parse_bse_listing(
        [{"SCRIP_CD": "999999", "scrip_id": "", "Issuer_Name": "Mystery Ltd"}]
    )

    assert len(skipped) == 1
    assert skipped[0][0] == "999999"
    assert "Yahoo cannot address it" in skipped[0][1]


def test_an_inactive_bse_scrip_is_skipped() -> None:
    _records, skipped = ub.parse_bse_listing(
        [{"scrip_id": "DEAD", "Status": "Suspended", "Issuer_Name": "Dead Ltd"}]
    )

    assert skipped == [("DEAD", "BSE status is suspended, not active")]


def test_the_bse_industry_becomes_the_sector() -> None:
    """BSE publishes one and NSE does not, so this is the only sector data the
    exchange files supply."""
    assert by_symbol(bse_records())["ABB.BO"]["sector"] == "Heavy Electrical Equipment"


def test_the_bse_issuer_name_is_preferred_over_the_short_scrip_name() -> None:
    record = by_symbol(bse_records())["ABB.BO"]

    assert record["name"] == "ABB India Limited"
    assert "ABB" in record["aliases"]


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------
def test_turnover_is_read_per_symbol() -> None:
    turnover = ub.parse_turnover(BHAVCOPY_ROWS)

    assert turnover["RELIANCE"] == pytest.approx(17_668_058_961.0)
    assert turnover["TCS"] == pytest.approx(5_923_000_000.0)


def test_a_non_equity_series_contributes_no_turnover() -> None:
    """A sovereign gold bond's turnover says nothing about how prominent a
    company is, and counting it would let it outrank a listed business."""
    assert "SGBJUN28" not in ub.parse_turnover(BHAVCOPY_ROWS)


def test_a_repeated_symbol_keeps_the_larger_turnover() -> None:
    """A ranking signal, not an accounting figure: the largest observation is
    the safest reading of "how much interest", and summing sessions would
    double-count."""
    turnover = ub.parse_turnover(
        [
            {"TckrSymb": "AAA", "SctySrs": "EQ", "TtlTrfVal": "100"},
            {"TckrSymb": "AAA", "SctySrs": "EQ", "TtlTrfVal": "900"},
            {"TckrSymb": "AAA", "SctySrs": "EQ", "TtlTrfVal": "500"},
        ]
    )

    assert turnover["AAA"] == pytest.approx(900.0)


@pytest.mark.parametrize("value", ["", "-", "nan", "abc", "-5", None])
def test_an_unreadable_turnover_is_dropped_not_guessed(value: object) -> None:
    turnover = ub.parse_turnover([{"TckrSymb": "AAA", "SctySrs": "EQ", "TtlTrfVal": value}])

    assert turnover.get("AAA", 0.0) == 0.0


def test_commas_in_a_turnover_are_tolerated() -> None:
    turnover = ub.parse_turnover([{"TckrSymb": "AAA", "SctySrs": "EQ", "TtlTrfVal": "1,234.50"}])

    assert turnover["AAA"] == pytest.approx(1234.5)


# ---------------------------------------------------------------------------
# the merge: never lose a curated row
# ---------------------------------------------------------------------------
def test_a_curated_alias_survives_a_refresh() -> None:
    """ "RIL" exists in no exchange file. Losing it would be a regression
    disguised as an upgrade."""
    report = ub.merge(CURATED, nse_records() + bse_records())
    record = by_symbol(report.rows)["RELIANCE.NS"]

    assert "RIL" in record["aliases"].split("|")
    assert "Reliance" in record["aliases"].split("|")


def test_the_exchange_spelling_wins_for_the_name() -> None:
    report = ub.merge(CURATED, nse_records())

    assert by_symbol(report.rows)["RELIANCE.NS"]["name"] == "Reliance Industries Limited"


def test_the_superseded_name_is_kept_as_an_alias() -> None:
    """The bug this guards. Replacing "Reliance Industries Ltd" with the
    exchange's "Reliance Industries Limited" deleted a lookup key that worked,
    and because BSE's own spelling happened to BE the old one, the importer
    started rewriting NSE holdings to .BO symbols."""
    report = ub.merge(CURATED, nse_records())
    aliases = by_symbol(report.rows)["RELIANCE.NS"]["aliases"].split("|")

    assert "Reliance Industries Ltd" in aliases


def test_the_curated_sector_is_kept_where_the_exchange_has_none() -> None:
    """NSE's listing carries no sector at all, so the curated value is the
    only one there is."""
    report = ub.merge(CURATED, nse_records())

    assert by_symbol(report.rows)["RELIANCE.NS"]["sector"] == "Energy"


def test_the_curated_instrument_type_is_kept() -> None:
    """An ETF must not be relabelled EQUITY by a listing that only knows
    about equities."""
    report = ub.merge(CURATED, nse_records(), {"NSE": {"NIFTYBEES": 2.1e9}})

    assert by_symbol(report.rows)["NIFTYBEES.NS"]["instrument_type"] == "ETF"


def test_an_isin_is_filled_in_from_the_exchange() -> None:
    report = ub.merge(CURATED, nse_records())

    assert by_symbol(report.rows)["RELIANCE.NS"]["isin"] == "INE002A01018"


def test_listing_rows_with_no_curated_counterpart_are_added() -> None:
    report = ub.merge(CURATED, nse_records() + bse_records())

    assert "TCS.NS" in by_symbol(report.rows)
    assert "ABB.BO" in by_symbol(report.rows)
    assert report.added == 3  # TCS.NS, RELIANCE.BO, ABB.BO


def test_the_output_is_ordered_deterministically() -> None:
    """The file is committed, so a stable order keeps a refresh's diff
    readable."""
    first = ub.merge(CURATED, nse_records() + bse_records()).rows
    second = ub.merge(CURATED, list(reversed(bse_records() + nse_records()))).rows

    assert [r["yahoo_symbol"] for r in first] == [r["yahoo_symbol"] for r in second]


def test_turnover_is_attached_per_exchange() -> None:
    """So the NSE and BSE lines of one company are ranked on their own
    volumes rather than sharing one."""
    report = ub.merge(
        CURATED,
        nse_records() + bse_records(),
        {"NSE": {"RELIANCE": 1.76e10}, "BSE": {"RELIANCE": 8.9e8}},
    )
    records = by_symbol(report.rows)

    assert float(records["RELIANCE.NS"]["turnover"]) == pytest.approx(1.76e10)
    assert float(records["RELIANCE.BO"]["turnover"]) == pytest.approx(8.9e8)


def test_an_instrument_that_did_not_trade_gets_zero_turnover() -> None:
    report = ub.merge(CURATED, nse_records(), ALIVE)

    assert by_symbol(report.rows)["TCS.NS"]["turnover"] == "0"


# ---------------------------------------------------------------------------
# the merge: dropping what has stopped trading
# ---------------------------------------------------------------------------
def test_a_curated_row_absent_from_the_listing_with_no_turnover_is_dropped() -> None:
    """Two independent sources have to agree. This caught TATAMOTORS.NS after
    the demerger into TMCV/TMPV -- Yahoo also refuses to price it, and
    offering a searchable symbol that cannot be priced is the same harm as
    inventing one."""
    curated = [*CURATED, {**CURATED[0], "symbol": "TATAMOTORS", "yahoo_symbol": "TATAMOTORS.NS"}]

    report = ub.merge(curated, nse_records(), ALIVE)

    assert "TATAMOTORS.NS" not in by_symbol(report.rows)
    assert [symbol for symbol, _why in report.stale] == ["TATAMOTORS.NS"]


def test_a_dropped_row_is_named_with_its_reason() -> None:
    curated = [*CURATED, {**CURATED[0], "symbol": "GONE", "yahoo_symbol": "GONE.NS"}]

    report = ub.merge(curated, nse_records(), ALIVE)

    assert "absent from the NSE listing" in report.stale[0][1]
    assert "GONE.NS" in report.describe()


def test_a_curated_etf_with_turnover_is_kept() -> None:
    """The same rule keeps the eleven bundled NSE ETFs: absent from an
    *equity* listing by definition, but really trading."""
    report = ub.merge(CURATED, nse_records(), {"NSE": {"NIFTYBEES": 2.1e9}})

    assert "NIFTYBEES.NS" in by_symbol(report.rows)
    assert report.stale == ()


def test_a_row_present_in_the_listing_is_never_dropped_for_lack_of_turnover() -> None:
    """A currently listed company that simply did not trade that day is still
    listed, and the listing is the authority on that."""
    report = ub.merge(CURATED, nse_records(), ALIVE)

    assert "TCS.NS" in by_symbol(report.rows)


def test_nothing_is_dropped_when_no_turnover_was_loaded_at_all() -> None:
    """The guard that stops --no-bhavcopy from deleting the universe: without
    turnover every row reads as "did not trade"."""
    curated = [*CURATED, {**CURATED[0], "symbol": "GONE", "yahoo_symbol": "GONE.NS"}]

    report = ub.merge(curated, nse_records(), {})

    assert report.stale == ()
    assert "GONE.NS" in by_symbol(report.rows)


def test_keep_stale_overrides_the_rule() -> None:
    curated = [*CURATED, {**CURATED[0], "symbol": "GONE", "yahoo_symbol": "GONE.NS"}]

    report = ub.merge(curated, nse_records(), {"NSE": {"RELIANCE": 1.0}}, keep_stale=True)

    assert "GONE.NS" in by_symbol(report.rows)
    assert report.stale == ()


def test_an_added_listing_row_is_never_treated_as_stale() -> None:
    """Only curated rows are candidates: a listing row is in the listing by
    definition."""
    report = ub.merge([], nse_records(), {"NSE": {"RELIANCE": 1.0e9}})

    assert report.stale == ()
    assert len(report.rows) == 2


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def test_the_report_counts_what_it_did() -> None:
    report = ub.merge(CURATED, nse_records() + bse_records(), ALIVE)

    assert report.total == len(report.rows)
    assert report.curated_kept == 2
    assert report.curated_enriched == 1
    assert report.with_isin == 4  # RELIANCE.NS/.BO, TCS.NS, ABB.BO
    assert report.with_turnover == 2


def test_the_report_groups_skip_reasons() -> None:
    _records, skipped = ub.parse_nse_listing(NSE_ROWS)
    report = ub.merge(CURATED, nse_records(), skipped=skipped)

    assert "1 x NSE series SM is not a priced equity series" in report.describe()


def test_a_notice_is_carried_into_the_report() -> None:
    report = ub.merge(CURATED, nse_records(), notices=["turnover unavailable"])

    assert "turnover unavailable" in report.describe()


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------
def test_a_file_written_before_isin_existed_still_loads(tmp_path: Path) -> None:
    """Backward compatibility of the column contract: readers match by name."""
    path = tmp_path / "old.csv"
    path.write_text(
        "symbol,yahoo_symbol,name,exchange,country,currency,instrument_type,sector,aliases\n"
        "RELIANCE,RELIANCE.NS,Reliance Industries Ltd,NSE,India,INR,EQUITY,Energy,RIL\n",
        encoding="utf-8",
    )

    rows = ub.load_curated(path)

    assert len(rows) == 1
    assert rows[0]["aliases"] == "RIL"
    assert rows[0]["isin"] == ""
    assert rows[0]["turnover"] == ""


def test_a_missing_curated_file_is_not_an_error(tmp_path: Path) -> None:
    assert ub.load_curated(tmp_path / "nope.csv") == []


def test_writing_round_trips_every_column(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    report = ub.merge(CURATED, nse_records() + bse_records(), ALIVE)

    ub.write_universe(path, report.rows)
    written = ub.load_curated(path)

    assert len(written) == report.total
    assert {r["yahoo_symbol"] for r in written} == {r["yahoo_symbol"] for r in report.rows}


def test_the_header_is_the_documented_contract(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    ub.write_universe(path, nse_records())

    with path.open(encoding="utf-8", newline="") as handle:
        assert tuple(next(csv.reader(handle))) == ub.CSV_COLUMNS


def test_writing_backs_up_the_previous_file(tmp_path: Path) -> None:
    """This repository is not under version control, so an overwrite would
    otherwise be unrecoverable."""
    path = tmp_path / "out.csv"
    ub.write_universe(path, CURATED, backup=False)

    backup = ub.write_universe(path, nse_records())

    assert backup
    assert Path(backup).is_file()
    assert {r["yahoo_symbol"] for r in ub.load_curated(Path(backup))} == {
        "RELIANCE.NS",
        "NIFTYBEES.NS",
    }


def test_the_first_write_has_nothing_to_back_up(tmp_path: Path) -> None:
    assert ub.write_universe(tmp_path / "fresh.csv", nse_records()) == ""


def test_backup_can_be_declined(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    ub.write_universe(path, CURATED, backup=False)

    assert ub.write_universe(path, nse_records(), backup=False) == ""
    assert not (tmp_path / "out.csv.bak").exists()


def test_a_failed_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """Atomic replace: an interrupted write must not leave a half-truncated
    universe that the app would happily load."""

    class Exploding(dict):
        def get(self, key: object, default: object = None) -> object:
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        ub.write_universe(tmp_path / "out.csv", [Exploding()])

    assert list(tmp_path.glob("*.tmp")) == []
    assert not (tmp_path / "out.csv").exists()


def test_an_interrupted_write_does_not_damage_the_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    ub.write_universe(path, nse_records(), backup=False)
    before = path.read_text(encoding="utf-8")

    class Exploding(dict):
        def get(self, key: object, default: object = None) -> object:
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        ub.write_universe(path, [Exploding()], backup=False)

    assert path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_the_meta_records_the_bhavcopy_date_actually_used() -> None:
    """A universe presented without its date implies it is current, and for an
    end-of-day file that is never quite true."""
    report = ub.merge(CURATED, nse_records())
    meta = ub.build_meta(
        [
            {"source": "NSE equity listing", "url": "u1", "rows": 2567, "as_of": ""},
            {"source": "NSE bhavcopy", "url": "u2", "rows": 3633, "as_of": "2026-09-02"},
        ],
        report,
    )

    assert meta["as_of"] == "2026-09-02"
    assert meta["rows"] == report.total
    assert len(meta["sources"]) == 2


def test_the_meta_is_json_serialisable() -> None:
    meta = ub.build_meta([], ub.merge(CURATED, nse_records()))

    assert json.loads(json.dumps(meta))["rows"] > 0


def test_the_meta_has_no_date_when_no_source_had_one() -> None:
    """Better an empty string the UI can render as "unknown" than today's date
    standing in for a file that may be weeks old."""
    meta = ub.build_meta(
        [{"source": "NSE equity listing", "url": "u", "rows": 10, "as_of": ""}],
        ub.merge(CURATED, nse_records()),
    )

    assert meta["as_of"] == ""


def test_the_meta_takes_the_oldest_source_date() -> None:
    """The universe is only as fresh as its stalest input."""
    meta = ub.build_meta(
        [
            {"source": "a", "url": "", "rows": 1, "as_of": "2026-09-02"},
            {"source": "b", "url": "", "rows": 1, "as_of": "2026-08-28"},
        ],
        ub.merge(CURATED, nse_records()),
    )

    assert meta["as_of"] == "2026-08-28"


# ---------------------------------------------------------------------------
# the CLI, without a network
# ---------------------------------------------------------------------------
def test_a_missing_data_directory_cannot_start(tmp_path: Path) -> None:
    assert ub.main(["--data-dir", str(tmp_path / "nope")]) == 2


def test_an_unparseable_date_is_refused(tmp_path: Path) -> None:
    assert ub.main(["--data-dir", str(tmp_path), "--on", "yesterday"]) == 2


def test_a_failed_listing_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh that half-worked is worse than one that did not run: search
    would quietly stop finding things."""
    from src.data import exchange as ex

    (tmp_path / ub.INDIA_FILE).write_text(
        "symbol,yahoo_symbol,name\nRELIANCE,RELIANCE.NS,Reliance\n", encoding="utf-8"
    )
    before = (tmp_path / ub.INDIA_FILE).read_text(encoding="utf-8")
    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(notice="the CDN refused")
    )

    code = ub.main(["--data-dir", str(tmp_path)])

    assert code == 1
    assert (tmp_path / ub.INDIA_FILE).read_text(encoding="utf-8") == before


def test_a_successful_run_writes_the_file_and_the_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data import exchange as ex

    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )

    import datetime as dt

    monkeypatch.setattr(
        ex,
        "latest_bhavcopy",
        lambda label, **_k: ex.SourceFile(
            rows=tuple(BHAVCOPY_ROWS), as_of=dt.date(2026, 9, 2), source=f"{label} bhavcopy"
        ),
    )

    code = ub.main(["--data-dir", str(tmp_path)])

    assert code == 0
    rows = ub.load_curated(tmp_path / ub.INDIA_FILE)
    assert {r["yahoo_symbol"] for r in rows} >= {"RELIANCE.NS", "TCS.NS", "RELIANCE.BO", "ABB.BO"}
    meta = json.loads((tmp_path / ub.META_FILE).read_text(encoding="utf-8"))
    assert meta["as_of"] == "2026-09-02"


def test_a_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import exchange as ex

    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )
    monkeypatch.setattr(ex, "latest_bhavcopy", lambda label, **_k: ex.SourceFile(notice="closed"))

    code = ub.main(["--data-dir", str(tmp_path), "--dry-run"])

    assert code == 0
    assert not (tmp_path / ub.INDIA_FILE).exists()
    assert not (tmp_path / ub.META_FILE).exists()


def test_an_unavailable_bhavcopy_is_a_notice_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The universe is still complete without it; only the ranking is weaker,
    and that is said out loud rather than left to be discovered."""
    from src.data import exchange as ex

    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )
    monkeypatch.setattr(
        ex, "latest_bhavcopy", lambda label, **_k: ex.SourceFile(notice="market closed all week")
    )

    code = ub.main(["--data-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "turnover unavailable" in out
    assert (tmp_path / ub.INDIA_FILE).is_file()


def test_no_bhavcopy_skips_the_download_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from src.data import exchange as ex

    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )

    def explode(*_a: object, **_k: object) -> None:
        raise AssertionError("--no-bhavcopy must not download a bhavcopy")

    monkeypatch.setattr(ex, "latest_bhavcopy", explode)

    code = ub.main(["--data-dir", str(tmp_path), "--no-bhavcopy"])

    assert code == 0
    assert "without bhavcopy turnover" in capsys.readouterr().out


def test_the_curated_seed_is_never_written_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build reads the seed and writes a different file. They were briefly
    the same file, which meant running the build twice replaced the pristine
    seed's backup with a generated one and lost the hand-written aliases."""
    from src.data import exchange as ex

    ub.write_universe(tmp_path / ub.CURATED_FILE, CURATED, backup=False)
    seed_before = (tmp_path / ub.CURATED_FILE).read_text(encoding="utf-8")

    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )
    monkeypatch.setattr(ex, "latest_bhavcopy", lambda label, **_k: ex.SourceFile(notice="closed"))

    for _ in range(3):
        assert ub.main(["--data-dir", str(tmp_path)]) == 0

    assert (tmp_path / ub.CURATED_FILE).read_text(encoding="utf-8") == seed_before


def test_running_twice_produces_the_same_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotence: the build is derived entirely from the seed and the
    downloads, so nothing accumulates run over run."""
    from src.data import exchange as ex

    ub.write_universe(tmp_path / ub.CURATED_FILE, CURATED, backup=False)
    monkeypatch.setattr(
        ex, "fetch_nse_listing", lambda **_k: ex.SourceFile(rows=tuple(NSE_ROWS), source="NSE")
    )
    monkeypatch.setattr(
        ex, "fetch_bse_listing", lambda **_k: ex.SourceFile(rows=tuple(BSE_ROWS), source="BSE")
    )
    monkeypatch.setattr(ex, "latest_bhavcopy", lambda label, **_k: ex.SourceFile(notice="closed"))

    ub.main(["--data-dir", str(tmp_path)])
    first = (tmp_path / ub.INDIA_FILE).read_text(encoding="utf-8")
    ub.main(["--data-dir", str(tmp_path)])

    assert (tmp_path / ub.INDIA_FILE).read_text(encoding="utf-8") == first
