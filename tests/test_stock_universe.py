"""The local stock-master universe — ``src.stock_universe``, entirely offline.

This module is the thing standing between "type RELIANCE.NS" and "type
Reliance", so what it must get right is narrow and sharp:

Verified here
-------------
**Loading.** The bundled CSVs parse, every row carries a Yahoo symbol, a
missing file is skipped rather than fatal, a malformed file is skipped rather
than fatal, and de-duplication by Yahoo symbol means listing a symbol twice
(or in two files) cannot produce two search results. The cache is a real cache
— a second call does not re-read the disk — and :func:`refresh` clears it.

**Resolution is conservative.** ``resolve`` matches only on an exact Yahoo
symbol, an exact native symbol, or an exact name/alias. It NEVER fuzzy-matches,
because its output is used to rewrite what the user typed: silently turning
"RELAINCE" into RELIANCE.NS and adding it to a portfolio is the failure mode
this pins shut. ``normalise_symbol`` passes unknown text through unchanged for
the same reason — absence from a curated file is not proof a symbol is invalid,
and inventing a ``.NS`` suffix would break a working US ticker.

**Ambiguity is surfaced, not decided.** ``INFY`` is both the NYSE ADR (exact
Yahoo symbol) and the root of NSE's ``INFY.NS``. ``candidates_for`` returns
both; ``prefer_exchange`` lets the importer pick the NSE line for an Indian
broker file without that preference ever being able to invent a match that was
not already there.

**Search ranks identifiers above names.** A ticker query can never be
outranked by a name match, results are unique by Yahoo symbol, ordering is
deterministic (so the same query always renders the same list), and the fuzzy
tier is suppressed entirely when a confident hit exists — padding "reliance"
with a 37-point guess at "Elevance Health" would teach users to distrust the
top row, which is the one that was right.

**Typos reach the right company.** The regression that motivated the fuzzy
pool design: comparing a typo against a full name ("relaince" vs "reliance
industries ltd") scores below any usable cutoff and lets a short unrelated
name win, so the pool holds short keys instead. Six real typos are pinned.

Deterministic, offline, no RNG, no network. Every test that needs a custom
universe writes CSVs into ``tmp_path`` and passes ``data_dir``, so none of them
can be broken by editing the bundled files' contents.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from src import stock_universe as su

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
HEADER = "symbol,yahoo_symbol,name,exchange,country,currency,instrument_type,sector,aliases"


def write_universe(directory: Path, filename: str, *rows: str) -> Path:
    """Write one universe CSV into ``directory`` and return its path."""
    path = directory / filename
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear the module cache around every test.

    ``load_universe`` is ``lru_cache``d on the data directory, and the tests
    deliberately point it at different directories. Without this, a test that
    ran earlier would serve its universe to a later one.
    """
    su.refresh()
    yield
    su.refresh()


@pytest.fixture
def tiny(tmp_path: Path) -> str:
    """A three-row universe with a deliberate cross-exchange collision."""
    write_universe(
        tmp_path,
        "indian_stocks.csv",
        "RELIANCE,RELIANCE.NS,Reliance Industries Ltd,NSE,India,INR,EQUITY,Energy,RIL|Reliance",
        "INFY,INFY.NS,Infosys Ltd,NSE,India,INR,EQUITY,Information Technology,Infosys",
    )
    write_universe(
        tmp_path,
        "us_stocks.csv",
        "INFY,INFY,Infosys Limited ADR,NYSE,India,USD,EQUITY,Information Technology,Infosys ADR",
        "AAPL,AAPL,Apple Inc,NASDAQ,United States,USD,EQUITY,Information Technology,Apple",
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# the bundled files
# ---------------------------------------------------------------------------
def test_the_bundled_universe_loads_and_is_not_empty() -> None:
    universe = su.load_universe()
    assert len(universe) > 100, "the shipped master files should carry a useful universe"


def test_every_bundled_row_carries_a_yahoo_symbol_and_a_name() -> None:
    for record in su.load_universe():
        assert record.yahoo_symbol, f"{record} has no Yahoo symbol"
        assert record.name, f"{record.yahoo_symbol} has no name"


def test_bundled_yahoo_symbols_are_unique() -> None:
    """Duplicates would surface as duplicate search results."""
    symbols = [r.yahoo_symbol for r in su.load_universe()]
    assert len(symbols) == len(set(symbols))


def test_bundled_indian_equities_all_carry_an_exchange_suffix() -> None:
    """A bare NSE symbol in the master file would fail to price."""
    for record in su.load_universe():
        if record.exchange == "NSE":
            assert record.yahoo_symbol.endswith(".NS"), record.yahoo_symbol


def test_bundled_indices_are_typed_as_indices_not_equities() -> None:
    """An index priced as a holding would be nonsense; the type gates that."""
    for record in su.load_universe():
        if record.yahoo_symbol.startswith("^"):
            assert record.instrument_type == "INDEX", record.yahoo_symbol
            assert record.is_index


def test_the_prompts_worked_example_resolves() -> None:
    """The exact case from the requirement: "Reliance" -> Reliance Industries.

    Asserted on the IDENTITY, not on the exact name string. The universe is
    now generated from the exchange's own listing, so the spelling is the
    exchange's to change -- it moved from the curated "Reliance Industries
    Ltd" to NSE's "Reliance Industries Limited" the first time the listing was
    pulled. Pinning the string would make this test fail on a rename that is
    not a defect; pinning the symbol, exchange and country is what the
    requirement actually asked for.
    """
    hits = su.search("Reliance", limit=5)
    assert hits, "searching Reliance must return something"
    top = hits[0].record
    assert "reliance industries" in top.name.lower()
    assert top.yahoo_symbol == "RELIANCE.NS"
    assert top.exchange == "NSE"
    assert top.country == "India"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("TCS", "TCS.NS"),
        ("INFY", "INFY"),  # exact Yahoo symbol wins: the NYSE ADR owns it
        ("HDFCBANK", "HDFCBANK.NS"),
        ("ICICIBANK", "ICICIBANK.NS"),
        ("AAPL", "AAPL"),
    ],
)
def test_the_prompts_named_tickers_all_resolve(query: str, expected: str) -> None:
    assert su.normalise_symbol(query) == expected


# ---------------------------------------------------------------------------
# loading edge cases
# ---------------------------------------------------------------------------
def test_a_missing_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The app must still start if someone deletes a master file."""
    write_universe(tmp_path, "us_stocks.csv", "AAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,")
    universe = su.load_universe(str(tmp_path))
    assert [r.yahoo_symbol for r in universe] == ["AAPL"]


def test_an_empty_directory_yields_an_empty_universe_not_an_exception(tmp_path: Path) -> None:
    assert su.load_universe(str(tmp_path)) == ()
    assert su.search("anything", data_dir=str(tmp_path)) == []
    assert su.resolve("anything", data_dir=str(tmp_path)) is None


def test_a_file_without_the_required_columns_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "indian_stocks.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    write_universe(tmp_path, "us_stocks.csv", "AAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,")
    assert [r.yahoo_symbol for r in su.load_universe(str(tmp_path))] == ["AAPL"]


def test_a_row_without_a_yahoo_symbol_is_dropped(tmp_path: Path) -> None:
    """That field is the one thing the rest of the app cannot work without."""
    write_universe(
        tmp_path,
        "us_stocks.csv",
        "AAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,",
        "GHOST,,Ghost Corp,NASDAQ,US,USD,EQUITY,Tech,",
    )
    assert [r.yahoo_symbol for r in su.load_universe(str(tmp_path))] == ["AAPL"]


def test_a_utf8_bom_does_not_break_the_header(tmp_path: Path) -> None:
    """Excel writes a BOM; without utf-8-sig the first column name is mangled."""
    path = tmp_path / "us_stocks.csv"
    path.write_text(
        HEADER + "\nAAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,\n",
        encoding="utf-8-sig",
    )
    assert [r.yahoo_symbol for r in su.load_universe(str(tmp_path))] == ["AAPL"]


def test_a_symbol_listed_in_two_files_is_loaded_once_first_file_winning(tmp_path: Path) -> None:
    write_universe(tmp_path, "indian_stocks.csv", "X,X.NS,From India,NSE,India,INR,EQUITY,Energy,")
    write_universe(tmp_path, "us_stocks.csv", "X,X.NS,From US,NYSE,US,USD,EQUITY,Energy,")
    universe = su.load_universe(str(tmp_path))
    assert len(universe) == 1
    assert universe[0].name == "From India", "indian_stocks.csv is listed first"


def test_the_universe_is_cached_and_refresh_clears_it(tmp_path: Path) -> None:
    write_universe(tmp_path, "us_stocks.csv", "AAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,")
    assert len(su.load_universe(str(tmp_path))) == 1

    write_universe(
        tmp_path,
        "us_stocks.csv",
        "AAPL,AAPL,Apple Inc,NASDAQ,US,USD,EQUITY,Tech,",
        "MSFT,MSFT,Microsoft Corporation,NASDAQ,US,USD,EQUITY,Tech,",
    )
    assert len(su.load_universe(str(tmp_path))) == 1, "the cache should still be serving"

    su.refresh()
    assert len(su.load_universe(str(tmp_path))) == 2, "refresh must re-read the disk"


def test_aliases_are_split_on_the_pipe(tiny: str) -> None:
    record = su.by_yahoo_symbol("RELIANCE.NS", tiny)
    assert record is not None
    assert record.aliases == ("RIL", "Reliance")


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_an_exact_yahoo_symbol_resolves(tiny: str) -> None:
    assert su.resolve("RELIANCE.NS", tiny).yahoo_symbol == "RELIANCE.NS"


def test_a_bare_native_symbol_resolves_to_the_suffixed_one(tiny: str) -> None:
    assert su.resolve("RELIANCE", tiny).yahoo_symbol == "RELIANCE.NS"


def test_resolution_is_case_and_whitespace_insensitive(tiny: str) -> None:
    for text in ("reliance", "  Reliance  ", "ReLiAnCe", "'RELIANCE'"):
        assert su.normalise_symbol(text, tiny) == "RELIANCE.NS", text


def test_a_full_company_name_resolves(tiny: str) -> None:
    assert su.normalise_symbol("Reliance Industries Ltd", tiny) == "RELIANCE.NS"


def test_an_alias_resolves(tiny: str) -> None:
    assert su.normalise_symbol("RIL", tiny) == "RELIANCE.NS"


def test_resolve_never_fuzzy_matches(tiny: str) -> None:
    """The safety property. A typo must not silently become a real holding."""
    assert su.resolve("RELAINCE", tiny) is None
    assert su.resolve("relianc", tiny) is None
    assert su.resolve("Reliance Industrys", tiny) is None


def test_an_unknown_symbol_passes_through_upper_cased_and_unchanged(tiny: str) -> None:
    """Absence from a curated file is not proof a symbol does not exist."""
    assert su.normalise_symbol("wxyz.ns", tiny) == "WXYZ.NS"
    assert su.normalise_symbol("someSmallCap", tiny) == "SOMESMALLCAP"


def test_blank_input_resolves_to_nothing(tiny: str) -> None:
    for blank in ("", "   ", None):
        assert su.resolve(blank, tiny) is None  # type: ignore[arg-type]
        assert su.normalise_symbol(blank, tiny) == ""  # type: ignore[arg-type]


def test_by_yahoo_symbol_is_exact_only(tiny: str) -> None:
    assert su.by_yahoo_symbol("AAPL", tiny) is not None
    assert su.by_yahoo_symbol("AAP", tiny) is None
    assert su.by_yahoo_symbol("RELIANCE", tiny) is None, "the native symbol is not the Yahoo one"


# ---------------------------------------------------------------------------
# ambiguity
# ---------------------------------------------------------------------------
def test_candidates_for_returns_every_venue_of_an_ambiguous_symbol(tiny: str) -> None:
    symbols = [r.yahoo_symbol for r in su.candidates_for("INFY", tiny)]
    assert set(symbols) == {"INFY", "INFY.NS"}


def test_an_unambiguous_symbol_yields_exactly_one_candidate(tiny: str) -> None:
    assert len(su.candidates_for("AAPL", tiny)) == 1


def test_the_exact_yahoo_symbol_wins_by_default(tiny: str) -> None:
    """Someone typing a literal Yahoo symbol gets the instrument that owns it."""
    assert su.resolve("INFY", tiny).yahoo_symbol == "INFY"


def test_prefer_exchange_picks_the_indian_line_of_an_ambiguous_symbol(tiny: str) -> None:
    """The importer's case: a Zerodha file's bare INFY means INFY.NS."""
    assert su.resolve("INFY", tiny, prefer_exchange="NSE").yahoo_symbol == "INFY.NS"
    assert su.normalise_symbol("INFY", tiny, prefer_exchange="NSE") == "INFY.NS"


def test_prefer_exchange_cannot_invent_a_match_that_was_not_there(tiny: str) -> None:
    """The preference reorders candidates; it never conjures one."""
    assert su.resolve("AAPL", tiny, prefer_exchange="NSE").yahoo_symbol == "AAPL"
    assert su.resolve("NOSUCHTHING", tiny, prefer_exchange="NSE") is None


def test_prefer_exchange_is_ignored_when_that_venue_has_no_candidate(tiny: str) -> None:
    assert su.resolve("INFY", tiny, prefer_exchange="LSE").yahoo_symbol == "INFY"


# ---------------------------------------------------------------------------
# search ranking
# ---------------------------------------------------------------------------
def test_search_of_a_blank_query_is_empty(tiny: str) -> None:
    for blank in ("", "   "):
        assert su.search(blank, data_dir=tiny) == []


def test_search_results_are_unique_by_yahoo_symbol() -> None:
    """The stated requirement: no duplicate results."""
    for query in ("reliance", "bank", "nifty", "tata", "etf", "a"):
        symbols = [h.record.yahoo_symbol for h in su.search(query, limit=0)]
        assert len(symbols) == len(set(symbols)), query


def test_search_ordering_is_deterministic() -> None:
    """The same query must always render the same list, in the same order."""
    for query in ("tata", "bank", "gold"):
        first = [h.record.yahoo_symbol for h in su.search(query, limit=0)]
        second = [h.record.yahoo_symbol for h in su.search(query, limit=0)]
        assert first == second, query


def test_search_respects_the_limit() -> None:
    assert len(su.search("bank", limit=3)) <= 3


def test_a_limit_of_zero_or_less_returns_everything() -> None:
    capped = su.search("bank", limit=2)
    everything = su.search("bank", limit=0)
    assert len(everything) >= len(capped)


def test_an_exact_ticker_can_never_be_outranked_by_a_name_match() -> None:
    """Someone who types a ticker means it."""
    hits = su.search("ITC", limit=6)
    assert hits[0].record.yahoo_symbol == "ITC.NS"
    assert hits[0].matched_on == "ticker"


def test_a_shorter_symbol_wins_a_prefix_tie() -> None:
    """ "TCS" should surface TCS.NS ahead of any longer symbol starting TCS."""
    hits = su.search("TCS", limit=5)
    assert hits[0].record.yahoo_symbol == "TCS.NS"


def test_an_alias_match_reports_itself_as_an_alias_match() -> None:
    """`matched_on` is shown in the UI, so it has to be truthful."""
    hits = su.search("HUL", limit=3)
    assert hits[0].record.yahoo_symbol == "HINDUNILVR.NS"
    assert hits[0].matched_on == "alias"


def test_a_multi_word_query_matches_on_all_its_tokens() -> None:
    hits = su.search("bank nifty", limit=3)
    assert hits[0].record.yahoo_symbol == "^NSEBANK"


def test_a_confident_hit_suppresses_the_fuzzy_tier() -> None:
    """The regression: "reliance" must not be padded with an Elevance guess."""
    hits = su.search("reliance", limit=8)
    assert all(h.matched_on != "fuzzy match" for h in hits), [
        (h.record.yahoo_symbol, h.matched_on) for h in hits
    ]


@pytest.mark.parametrize(
    "typo,expected",
    [
        ("relaince", "RELIANCE.NS"),
        ("infosis", "INFY.NS"),
        ("hdfc bnak", "HDFCBANK.NS"),
        # Not pinned to a symbol: see the note in the test body.
        ("tata mtors", "tata motors"),
        ("asain paints", "ASIANPAINT.NS"),
        ("marutti", "MARUTI.NS"),
    ],
)
def test_a_typo_still_surfaces_the_intended_company_as_the_top_hit(
    typo: str, expected: str
) -> None:
    """Comparing a typo against a FULL name scores too low to be usable, which
    is why the fuzzy pool holds short keys instead. This is that regression."""
    hits = su.search(typo, limit=5)
    assert hits, f"no hit at all for {typo!r}"
    top = hits[0].record
    if expected.endswith((".NS", ".BO")) or expected.isupper():
        assert top.yahoo_symbol == expected, [h.record.yahoo_symbol for h in hits]
    else:
        # A company whose SYMBOL the exchange has retired is matched on its
        # name instead. "TATAMOTORS.NS" was pinned here until the demerger
        # into TMCV/TMPV retired it -- Yahoo now refuses to price it, and the
        # generated universe correctly drops it. Asserting the company rather
        # than the ticker is what makes this test survive the next rename.
        assert expected in top.name.lower(), [(h.record.yahoo_symbol, h.record.name) for h in hits]


def test_a_fuzzy_hit_is_labelled_as_fuzzy() -> None:
    """A suggestion must not present itself as a confirmed match."""
    hits = su.search("relaince", limit=3)
    assert hits[0].matched_on == "fuzzy match"


def test_pure_nonsense_returns_nothing() -> None:
    assert su.search("zzzqqqxxvv", limit=5) == []


def test_search_scores_never_exceed_the_exact_match_ceiling() -> None:
    for hit in su.search("RELIANCE.NS", limit=5):
        assert 0.0 < hit.score <= 100.0


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def test_an_exchange_filter_excludes_every_other_venue() -> None:
    hits = su.search("infy", limit=0, exchanges=("NSE",))
    assert hits
    assert all(h.record.exchange == "NSE" for h in hits)


def test_an_instrument_type_filter_excludes_every_other_type() -> None:
    hits = su.search("nifty", limit=0, instrument_types=("ETF",))
    assert hits
    assert all(h.record.instrument_type == "ETF" for h in hits)


def test_filters_are_case_insensitive() -> None:
    assert su.search("infy", limit=0, exchanges=("nse",)) == su.search(
        "infy", limit=0, exchanges=("NSE",)
    )


def test_a_filter_that_matches_nothing_returns_nothing_rather_than_falling_back() -> None:
    assert su.search("reliance", limit=5, exchanges=("NASDAQ",)) == []


# ---------------------------------------------------------------------------
# suffix hints
# ---------------------------------------------------------------------------
def test_a_bare_unknown_symbol_gets_a_suffix_hint(tiny: str) -> None:
    hint = su.suffix_hint("SOMECO")
    assert hint is not None
    assert ".NS" in hint and ".BO" in hint


def test_a_symbol_that_already_carries_a_suffix_gets_no_hint() -> None:
    for symbol in ("RELIANCE.NS", "TATASTEEL.BO", "VOD.L", "SHOP.TO"):
        assert su.suffix_hint(symbol) is None, symbol


def test_a_known_symbol_gets_no_hint() -> None:
    """AAPL needs no suffix and saying otherwise would be wrong advice."""
    assert su.suffix_hint("AAPL") is None
    assert su.suffix_hint("RELIANCE") is None


def test_an_index_symbol_gets_no_hint() -> None:
    assert su.suffix_hint("^NSEI") is None


def test_blank_input_gets_no_hint() -> None:
    assert su.suffix_hint("") is None


# ---------------------------------------------------------------------------
# records & facets
# ---------------------------------------------------------------------------
def test_the_label_reads_as_name_symbol_exchange(tiny: str) -> None:
    assert su.by_yahoo_symbol("RELIANCE.NS", tiny).label == (
        "Reliance Industries Ltd · RELIANCE.NS · NSE"
    )


def test_an_etf_is_flagged_as_a_fund_and_an_index_is_not() -> None:
    assert su.by_yahoo_symbol("NIFTYBEES.NS").is_fund
    assert not su.by_yahoo_symbol("RELIANCE.NS").is_fund
    assert su.by_yahoo_symbol("^NSEI").is_index
    assert not su.by_yahoo_symbol("^NSEI").is_fund


def test_exchanges_are_returned_in_priority_order() -> None:
    """NSE first: this is an India-first application and the order is visible."""
    codes = su.exchanges()
    assert codes[0] == "NSE"
    ranked = [c for c in su.EXCHANGE_PRIORITY if c in codes]
    assert list(codes[: len(ranked)]) == ranked


def test_countries_and_instrument_types_are_sorted_and_deduplicated() -> None:
    for facet in (su.countries(), su.instrument_types()):
        assert list(facet) == sorted(set(facet))


def test_the_summary_total_equals_the_universe_size() -> None:
    summary = su.universe_summary()
    assert summary["total"] == len(su.load_universe())


def test_the_summary_exchange_counts_sum_to_the_total() -> None:
    """The UI quotes these numbers as its honest coverage statement."""
    summary = su.universe_summary()
    per_exchange = sum(v for k, v in summary.items() if k != "total")
    assert per_exchange == summary["total"]


# ---------------------------------------------------------------------------
# prominence: ranking a full-market universe
# ---------------------------------------------------------------------------
PROMINENCE_CSV = (
    "symbol,yahoo_symbol,name,exchange,country,currency,instrument_type,sector,"
    "aliases,isin,turnover\n"
    # Two "HDFC" prefixes where the BUSIER one has the LONGER symbol: the case
    # the old 0.5-per-character prefix penalty got wrong.
    "HDFCBANK,HDFCBANK.NS,HDFC Bank Limited,NSE,India,INR,EQUITY,Banking,,"
    "INE040A01034,25947000000\n"
    "HDFCAMC,HDFCAMC.NS,HDFC Asset Management Company Limited,NSE,India,INR,"
    "EQUITY,Finance,,INE127D01025,2406000000\n"
    # Two lines of ONE company: the dual-listing case.
    "RELIANCE,RELIANCE.NS,Reliance Industries Limited,NSE,India,INR,EQUITY,"
    "Energy,RIL,INE002A01018,17668000000\n"
    "RELIANCE,RELIANCE.BO,Reliance Industries Ltd,BSE,India,INR,EQUITY,"
    "Refineries,,INE002A01018,891000000\n"
    # A near-namesake that traded almost nothing: the fuzzy-tier case.
    "RCOM,RCOM.NS,Reliance Communications Limited,NSE,India,INR,EQUITY,Telecom,,"
    "INE330H01018,3000000\n"
    # Never traded, so no prominence at all.
    "QUIET,QUIET.NS,Reliance Quiet Holdings Limited,NSE,India,INR,EQUITY,Misc,,"
    "INE000000001,0\n"
)


@pytest.fixture
def prominence_dir(tmp_path: Path) -> str:
    (tmp_path / "indian_stocks.csv").write_text(PROMINENCE_CSV, encoding="utf-8")
    su.refresh()
    yield str(tmp_path)
    su.refresh()


def top(query: str, data_dir: str, limit: int = 6) -> list[str]:
    return [h.record.yahoo_symbol for h in su.search(query, limit=limit, data_dir=data_dir)]


def test_turnover_is_read_from_the_file(prominence_dir: str) -> None:
    record = su.by_yahoo_symbol("HDFCBANK.NS", prominence_dir)

    assert record.turnover == pytest.approx(25_947_000_000.0)
    assert record.isin == "INE040A01034"


@pytest.mark.parametrize("value", ["", "-", "abc", "nan", "-100"])
def test_an_unreadable_turnover_reads_as_zero(value: str, tmp_path: Path) -> None:
    """It feeds a ranking comparison, and a NaN there would make the sort
    order depend on the input sequence."""
    (tmp_path / "indian_stocks.csv").write_text(
        "symbol,yahoo_symbol,name,turnover\nAAA,AAA.NS,AAA Ltd," + value + "\n",
        encoding="utf-8",
    )
    su.refresh()

    assert su.by_yahoo_symbol("AAA.NS", str(tmp_path)).turnover == 0.0
    su.refresh()


def test_prominence_spans_the_full_scale(prominence_dir: str) -> None:
    """Normalised over the range the data actually spans, so the busiest
    instrument scores 1.0 and the quietest traded one scores 0.0."""
    records = {r.yahoo_symbol: r for r in su.load_universe(prominence_dir)}

    # HDFC Bank is the busiest row in the fixture, so it defines the top.
    assert records["HDFCBANK.NS"].prominence == pytest.approx(1.0)
    # RCOM traded the least, so it sits on the floor -- but ABOVE zero, which
    # is reserved for instruments that did not trade at all.
    assert records["RCOM.NS"].prominence == pytest.approx(su._PROMINENCE_FLOOR)
    assert su._PROMINENCE_FLOOR < records["HDFCAMC.NS"].prominence < 1.0


def test_an_untraded_instrument_has_no_prominence(prominence_dir: str) -> None:
    assert su.by_yahoo_symbol("QUIET.NS", prominence_dir).prominence == 0.0


def test_prominence_discriminates_between_the_largest_names(prominence_dir: str) -> None:
    """The property a percentile could not deliver. Ranked within an exchange,
    HDFC Bank and HDFC AMC came out 0.016 apart despite a 10x difference in
    traded value -- not enough to outweigh one symbol character."""
    records = {r.yahoo_symbol: r for r in su.load_universe(prominence_dir)}
    gap = records["HDFCBANK.NS"].prominence - records["HDFCAMC.NS"].prominence

    assert gap > 0.1


def test_a_universe_with_no_turnover_leaves_every_row_at_zero(tmp_path: Path) -> None:
    """So an older file, or one built with --no-bhavcopy, ranks exactly as it
    did before turnover existed."""
    (tmp_path / "indian_stocks.csv").write_text(
        "symbol,yahoo_symbol,name\nAAA,AAA.NS,AAA Ltd\nBBB,BBB.NS,BBB Ltd\n",
        encoding="utf-8",
    )
    su.refresh()

    assert all(r.prominence == 0.0 for r in su.load_universe(str(tmp_path)))
    su.refresh()


def test_one_turnover_value_shared_by_all_does_not_divide_by_zero(tmp_path: Path) -> None:
    (tmp_path / "indian_stocks.csv").write_text(
        "symbol,yahoo_symbol,name,turnover\nAAA,AAA.NS,AAA Ltd,500\nBBB,BBB.NS,BBB Ltd,500\n",
        encoding="utf-8",
    )
    su.refresh()

    assert all(r.prominence == 1.0 for r in su.load_universe(str(tmp_path)))
    su.refresh()


# -- what the ranking actually fixes ---------------------------------------
def test_a_prefix_query_prefers_the_busier_company(prominence_dir: str) -> None:
    """ "hdfc" returned HDFC AMC (7 characters, ₹241 cr) ahead of HDFC Bank
    (8 characters, ₹2,595 cr) until turnover outweighed symbol length."""
    assert top("hdfc", prominence_dir)[0] == "HDFCBANK.NS"


def test_a_typo_prefers_the_busier_namesake(prominence_dir: str) -> None:
    """Every record sharing a fuzzy key scores identically on the difflib
    ratio alone, so the winner was whichever symbol sorted first -- which is
    how "relaince" returned Reliance Communications."""
    assert top("relaince", prominence_dir)[0] == "RELIANCE.NS"


def test_a_name_substring_prefers_the_busier_company(prominence_dir: str) -> None:
    hits = top("reliance", prominence_dir)

    assert hits[0] == "RELIANCE.NS"
    assert hits.index("RCOM.NS") > hits.index("RELIANCE.NS")


def test_an_untraded_namesake_ranks_below_one_that_barely_traded(
    prominence_dir: str,
) -> None:
    """ "Did not trade" must rank below "traded a little". Both scored 0.0
    until a floor was reserved for the untraded case, which made them
    indistinguishable and left the alphabet to decide."""
    hits = top("reliance", prominence_dir)

    assert hits.index("RCOM.NS") < hits.index("QUIET.NS")
    assert hits[-1] == "QUIET.NS"


def test_the_nse_line_of_a_dual_listing_wins(prominence_dir: str) -> None:
    """India-first is a stated product decision, and the NSE line is also the
    busier one -- both rules agree here, and the test pins the outcome."""
    hits = top("RELIANCE", prominence_dir)

    assert hits[0] == "RELIANCE.NS"
    assert hits[1] == "RELIANCE.BO"


def test_an_exact_ticker_is_never_displaced_by_a_bigger_company(
    prominence_dir: str,
) -> None:
    """Typing a ticker means that ticker. Exact tiers carry no prominence
    bonus for exactly this reason."""
    assert top("RCOM", prominence_dir)[0] == "RCOM.NS"
    assert top("HDFCAMC", prominence_dir)[0] == "HDFCAMC.NS"


def test_an_exact_match_scores_its_tier_exactly(prominence_dir: str) -> None:
    hits = su.search("RCOM.NS", limit=1, data_dir=prominence_dir)

    assert hits[0].score == pytest.approx(su._SCORE_EXACT_YAHOO)


# -- the invariant that keeps the tiers ordered ----------------------------
def test_the_prominence_bonus_cannot_promote_a_weaker_match() -> None:
    """The load-bearing inequality: the weakest possible member of a tier
    still beats the strongest possible member of the tier below, whatever
    either one traded. Without it a substring match on a large company would
    outrank a name match on a small one."""
    assert su._PROMINENCE_BONUS <= su._MIN_TIER_GAP - su._MAX_PREFIX_PENALTY


def test_the_declared_minimum_tier_gap_is_true_of_the_tiers() -> None:
    """_MIN_TIER_GAP is asserted against the constants rather than trusted, so
    lowering a tier cannot silently invalidate the inequality above."""
    tiers = sorted(
        [
            su._SCORE_EXACT_YAHOO,
            su._SCORE_EXACT_SYMBOL,
            su._SCORE_EXACT_NAME,
            su._SCORE_SYMBOL_PREFIX,
            su._SCORE_NAME_PREFIX,
            su._SCORE_ALL_TOKENS,
            su._SCORE_NAME_SUBSTRING,
            su._SCORE_SYMBOL_SUBSTRING,
        ]
    )
    gaps = [b - a for a, b in itertools.pairwise(tiers) if b - a > 0]

    assert min(gaps) >= su._MIN_TIER_GAP


def test_an_exact_alias_is_worth_as_much_as_an_exact_name() -> None:
    """An alias IS a name the instrument is known by. Scoring it lower meant
    that BSE's own spelling of Reliance -- which happens to be the curated
    "Reliance Industries Ltd" -- matched on `name` while the NSE row matched
    the same query on `alias`, and the less liquid line won."""
    assert su._SCORE_EXACT_ALIAS == su._SCORE_EXACT_NAME


def test_an_alias_query_resolves_to_the_primary_listing(prominence_dir: str) -> None:
    assert top("RIL", prominence_dir)[0] == "RELIANCE.NS"


# -- provenance -------------------------------------------------------------
def test_the_as_of_date_is_read_from_the_meta(tmp_path: Path) -> None:
    (tmp_path / "indian_stocks.csv").write_text(
        "symbol,yahoo_symbol,name\nAAA,AAA.NS,AAA Ltd\n", encoding="utf-8"
    )
    (tmp_path / "universe_meta.json").write_text(
        '{"as_of": "2026-09-02", "rows": 1}', encoding="utf-8"
    )
    su.refresh()

    assert su.as_of(str(tmp_path)) == "2026-09-02"
    assert su.universe_meta(str(tmp_path))["rows"] == 1
    su.refresh()


def test_a_universe_with_no_meta_reports_no_date(tmp_path: Path) -> None:
    """ "" means the file was not generated -- a hand-maintained or older
    universe. Callers say "date unknown" rather than guessing today."""
    (tmp_path / "indian_stocks.csv").write_text(
        "symbol,yahoo_symbol,name\nAAA,AAA.NS,AAA Ltd\n", encoding="utf-8"
    )
    su.refresh()

    assert su.as_of(str(tmp_path)) == ""
    assert su.universe_meta(str(tmp_path)) == {}
    su.refresh()


def test_unreadable_meta_is_not_an_error(tmp_path: Path) -> None:
    (tmp_path / "universe_meta.json").write_text("{not json", encoding="utf-8")
    su.refresh()

    assert su.universe_meta(str(tmp_path)) == {}
    su.refresh()


# -- the shipped file -------------------------------------------------------
def test_the_shipped_universe_carries_turnover_and_isins() -> None:
    """A smoke check on the generated file itself: if a refresh ever wrote a
    universe with no turnover, search would silently fall back to
    alphabetical ranking and nothing else would fail."""
    records = su.load_universe()
    indian = [r for r in records if r.country.lower() == "india"]

    assert len(indian) > 5000, "the generated Indian universe looks truncated"
    assert sum(1 for r in indian if r.turnover > 0) > 4000
    assert sum(1 for r in indian if r.isin) > 5000


def test_the_shipped_universe_has_both_indian_exchanges() -> None:
    summary = su.universe_summary()

    assert summary.get("NSE", 0) > 2000
    assert summary.get("BSE", 0) > 4000


def test_every_shipped_nse_or_bse_symbol_has_an_exchange_suffix() -> None:
    """A bare symbol cannot be priced, so one in the file is a build bug.

    Scoped by EXCHANGE, not by country: Indian ADRs (``INFY``, ``HDB``) are
    country India on a US exchange, and ``^NSEI`` is an index. All three are
    correct without a ``.NS``/``.BO`` suffix.
    """
    wrong = [
        r.yahoo_symbol
        for r in su.load_universe()
        if r.exchange.upper() in {"NSE", "BSE"} and not r.yahoo_symbol.endswith((".NS", ".BO"))
    ]

    assert not wrong, wrong[:10]
