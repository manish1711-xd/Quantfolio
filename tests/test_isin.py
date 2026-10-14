"""ISIN validation and resolution — ``src.isin``, entirely offline.

A wrong ISIN mapping puts somebody else's company in the portfolio while every
number on screen still looks plausible, so this module's tests are about
refusing to be wrong rather than about breadth of coverage.

Verified here
-------------
**The check-digit algorithm is right.** ISO 6166 is a mod-10 (Luhn) checksum
over the letter-expanded first eleven characters, and it is easy to implement
subtly wrong — off-by-one in which digits get doubled produces an
implementation that validates roughly half of all real ISINs. It is therefore
pinned against nine independently-known identifiers, including the
``AU0000XVGZA3`` example from the standard itself, and against the digit
recomputation for each.

**A typo cannot reach the portfolio.** Single-character corruption of a valid
ISIN is rejected, as are truncation, wrong length and non-ISIN text. This is
what makes a mistyped identifier a reported error rather than a silent
mis-holding.

**Every shipped mapping is machine-verified.** The seed file's ISINs all pass
the check digit, every symbol exists in the bundled stock universe, and every
ISIN's country prefix agrees with that instrument's country. A mistyped seed
row is therefore a test failure, not a wrong holding — which is the only reason
shipping a seed table at all is defensible.

**Nothing is ever guessed.** There is no fuzzy matching and no inference from
the ISIN body. Resolution is table lookup or nothing, and "nothing" is
reported.

**Learning is validated and reversible.** Pairings learned from a user's own
file are validated before storage, rejections are reported rather than dropped,
the file round-trips, and learned entries override seeded ones because the
user's own broker data outranks this repository's best effort.

Deterministic, offline, no RNG, no network. The ``isin_cache`` fixture from
``conftest`` isolates every write, so no test can touch ``data/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src import isin
from src import stock_universe as su

#: The real shipped file, deliberately read directly rather than through the
#: isolated copy: these tests exist to police what the repository ships.
SEED_PATH = Path(__file__).resolve().parent.parent / "data" / isin.SEED_FILE

#: Independently-known ISINs, used as algorithm test vectors. ``AU0000XVGZA3``
#: is the worked example in ISO 6166 itself.
KNOWN: tuple[tuple[str, str], ...] = (
    ("US0378331005", "Apple"),
    ("US5949181045", "Microsoft"),
    ("US0231351067", "Amazon"),
    ("US88160R1014", "Tesla"),
    ("GB0002634946", "BAE Systems"),
    ("IE00B4L5Y983", "iShares Core MSCI World"),
    ("NL0000009165", "Heineken"),
    ("AU0000XVGZA3", "ISO 6166 worked example"),
    ("INE002A01018", "Reliance Industries"),
)


# ---------------------------------------------------------------------------
# the check-digit algorithm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code,name", KNOWN, ids=[n for _, n in KNOWN])
def test_a_known_isin_validates(code: str, name: str) -> None:
    assert isin.is_valid(code), f"{name} ({code}) should be valid"


@pytest.mark.parametrize("code,name", KNOWN, ids=[n for _, n in KNOWN])
def test_the_check_digit_is_recomputed_exactly(code: str, name: str) -> None:
    """The strong form: not just "valid" but "we derive the same digit"."""
    assert isin.check_digit(code[:11]) == code[11]


def test_the_documented_worked_example_holds() -> None:
    """The docstring claims Apple's body sums to 45 and yields 5. It must."""
    assert isin.check_digit("US037833100") == "5"


@pytest.mark.parametrize(
    "code",
    [
        "US0378331006",  # Apple, last digit +1
        "US0378331004",  # Apple, last digit -1
        "INE002A01019",  # Reliance, last digit +1
        "US0231351060",
        "GB0002634940",
    ],
)
def test_a_single_digit_corruption_is_rejected(code: str) -> None:
    """The whole point of the checksum."""
    assert not isin.is_valid(code)


def test_a_transposition_is_usually_caught() -> None:
    """Luhn catches adjacent transpositions except the 0/9 pair."""
    assert not isin.is_valid("US0378313005")


@pytest.mark.parametrize(
    "junk",
    [
        "",
        "   ",
        None,
        "US037833100",  # 11 chars: truncated
        "US03783310055",  # 13 chars
        "0S0378331005",  # digit where the country prefix belongs
        "USA378331005",  # three-letter prefix
        "US037833100X",  # letter where the check digit belongs
        "RELIANCE.NS",
        "NOTANISIN",
    ],
)
def test_non_isin_input_is_rejected(junk: object) -> None:
    assert not isin.is_valid(junk)


def test_isin_shaped_nonsense_is_shaped_but_not_valid() -> None:
    """ "NOTANISIN123" has the ISO 6166 SHAPE. Only the checksum separates it
    from a real identifier, which is why the importer needs both tests."""
    assert isin.is_well_formed("NOTANISIN123")
    assert not isin.is_valid("NOTANISIN123")


def test_the_check_digit_of_a_wrong_length_body_is_empty_not_fabricated() -> None:
    """A caller must never compare against a made-up digit."""
    assert isin.check_digit("US03783310") == ""
    assert isin.check_digit("US037833100X5") == ""
    assert isin.check_digit("") == ""


def test_every_check_digit_is_a_single_decimal_digit() -> None:
    for code, _name in KNOWN:
        digit = isin.check_digit(code[:11])
        assert len(digit) == 1 and digit.isdigit()


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "INE002A01018",
        "ine002a01018",
        "  INE002A01018  ",
        "INE-002-A01018",
        "IN E002A010 18",
        "INE002A01018\n",
        "INE_002A01018",
    ],
)
def test_the_punctuation_people_paste_with_an_isin_is_stripped(raw: str) -> None:
    assert isin.normalise(raw) == "INE002A01018"
    assert isin.is_valid(raw)


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_normalising_nothing_yields_an_empty_string(blank: object) -> None:
    assert isin.normalise(blank) == ""


# ---------------------------------------------------------------------------
# country
# ---------------------------------------------------------------------------
def test_the_country_prefix_is_read() -> None:
    assert isin.country_code("INE002A01018") == "IN"
    assert isin.country_name("INE002A01018") == "India"
    assert isin.country_name("US0378331005") == "United States"


def test_an_unknown_prefix_has_no_opinion_rather_than_a_guess() -> None:
    """XS is Euroclear/Clearstream and names no single country."""
    assert isin.country_name("XS1234567890") is None


def test_a_malformed_isin_has_no_country() -> None:
    assert isin.country_code("nonsense") == ""
    assert isin.country_name("nonsense") is None


# ---------------------------------------------------------------------------
# what the repository ships
# ---------------------------------------------------------------------------
def _seed_rows() -> list[dict[str, str]]:
    with SEED_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_seed_file_exists_and_is_not_empty() -> None:
    assert SEED_PATH.is_file()
    assert _seed_rows()


def test_every_shipped_isin_passes_its_own_check_digit() -> None:
    """The guard that makes shipping a seed table defensible at all: a mistyped
    row is a build failure rather than a wrong holding."""
    bad = [
        f"{row['isin']} (expected check digit "
        f"{isin.check_digit(row['isin'][:11]) or '?'}, got {row['isin'][-1]})"
        for row in _seed_rows()
        if not isin.is_valid(row["isin"])
    ]
    assert not bad, "invalid ISINs in the shipped seed file: " + "; ".join(bad)


def test_every_shipped_symbol_exists_in_the_stock_universe() -> None:
    """A mapping onto a symbol the app cannot price is useless."""
    missing = [
        f"{row['isin']} -> {row['yahoo_symbol']}"
        for row in _seed_rows()
        if su.by_yahoo_symbol(row["yahoo_symbol"]) is None
    ]
    assert not missing, "seed rows naming unknown symbols: " + "; ".join(missing)


def test_every_shipped_isin_agrees_with_its_symbols_country() -> None:
    """Catches the class of mis-pairing a check digit cannot: an ``INE...``
    identifier resolving to a NASDAQ listing."""
    clashes = []
    for row in _seed_rows():
        issued = isin.country_name(row["isin"])
        record = su.by_yahoo_symbol(row["yahoo_symbol"])
        if issued and record and record.country and issued != record.country:
            clashes.append(
                f"{row['isin']} says {issued}, {row['yahoo_symbol']} is {record.country}"
            )
    assert not clashes, "country mismatches: " + "; ".join(clashes)


def test_shipped_isins_are_unique() -> None:
    codes = [row["isin"] for row in _seed_rows()]
    assert len(codes) == len(set(codes))


def test_shipped_symbols_are_unique() -> None:
    """Two ISINs mapping to one symbol would be a copy-paste slip."""
    symbols = [row["yahoo_symbol"] for row in _seed_rows()]
    assert len(symbols) == len(set(symbols))


def test_every_shipped_row_is_marked_as_seeded() -> None:
    assert {row["source"] for row in _seed_rows()} == {"seed"}


def test_the_seed_file_has_the_documented_columns() -> None:
    with SEED_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == isin.MAP_COLUMNS


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_a_seeded_isin_resolves(isin_cache: Path) -> None:
    record = isin.resolve("INE002A01018")

    assert record is not None
    assert record.yahoo_symbol == "RELIANCE.NS"
    assert record.source == "seed"
    assert record.country == "India"


def test_resolution_normalises_its_input(isin_cache: Path) -> None:
    assert isin.resolve(" ine-002-a01018 ").yahoo_symbol == "RELIANCE.NS"


def test_an_invalid_isin_never_resolves_even_if_the_table_held_it(
    isin_cache: Path,
) -> None:
    """Validation comes first, deliberately: a corrupt table row must not be
    honoured just because it is present."""
    (isin_cache / isin.LEARNED_FILE).write_text(
        "isin,yahoo_symbol,name,source\nINE002A01019,WRONG.NS,Wrong,learned\n",
        encoding="utf-8",
    )
    isin.refresh()

    assert isin.resolve("INE002A01019") is None


def test_a_corrupt_row_is_dropped_at_load_rather_than_served(isin_cache: Path) -> None:
    (isin_cache / isin.LEARNED_FILE).write_text(
        "isin,yahoo_symbol,name,source\n"
        "INE002A01019,WRONG.NS,Wrong,learned\n"
        "INE009A01021,INFY.NS,Infosys,learned\n",
        encoding="utf-8",
    )
    isin.refresh()

    loaded = isin.load_map()
    assert "INE002A01019" not in loaded
    assert loaded["INE009A01021"].yahoo_symbol == "INFY.NS"


def test_a_valid_but_unknown_isin_resolves_to_nothing(isin_cache: Path) -> None:
    """Table lookup or nothing. There is no inference from the ISIN body."""
    assert isin.is_valid("GB0002634946")
    assert isin.resolve("GB0002634946") is None


@pytest.mark.parametrize("blank", [None, "", "   ", "not an isin"])
def test_resolving_nothing_yields_none(blank: object, isin_cache: Path) -> None:
    assert isin.resolve(blank) is None


def test_a_country_cross_check_refuses_a_contradictory_mapping(
    isin_cache: Path,
) -> None:
    """The guard for the failure a checksum cannot see."""
    assert isin.resolve("INE002A01018", expected_country="India") is not None
    assert isin.resolve("INE002A01018", expected_country="United States") is None


def test_the_cross_check_is_skipped_when_the_prefix_is_unknown(isin_cache: Path) -> None:
    """It must never reject on ignorance."""
    (isin_cache / isin.LEARNED_FILE).write_text(
        "isin,yahoo_symbol,name,source\nXS0000000009,SOMETHING,Thing,learned\n",
        encoding="utf-8",
    )
    isin.refresh()
    if not isin.is_valid("XS0000000009"):
        pytest.skip("hand-built XS ISIN does not validate; the point stands elsewhere")

    assert isin.resolve("XS0000000009", expected_country="Anywhere") is not None


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------
def test_a_learned_pair_becomes_resolvable(isin_cache: Path) -> None:
    stored, rejected = isin.learn({"INE670A01012": "TATAELXSI.NS"})

    assert stored == 1
    assert rejected == []
    assert isin.resolve("INE670A01012").yahoo_symbol == "TATAELXSI.NS"


def test_a_learned_pair_is_marked_as_learned_not_seeded(isin_cache: Path) -> None:
    """Provenance is load-bearing: the UI states where a mapping came from."""
    isin.learn({"INE670A01012": "TATAELXSI.NS"})

    assert isin.resolve("INE670A01012").source == "learned"


def test_learning_persists_to_a_file(isin_cache: Path) -> None:
    isin.learn({"INE670A01012": "TATAELXSI.NS"})

    path = isin_cache / isin.LEARNED_FILE
    assert path.is_file()
    frame = pd.read_csv(path)
    assert tuple(frame.columns) == isin.MAP_COLUMNS
    assert frame["isin"].iloc[0] == "INE670A01012"


def test_learning_merges_rather_than_replacing(isin_cache: Path) -> None:
    isin.learn({"INE670A01012": "TATAELXSI.NS"})
    isin.learn({"INE262H01021": "PERSISTENT.NS"})

    assert isin.resolve("INE670A01012") is not None
    assert isin.resolve("INE262H01021") is not None


def test_relearning_an_isin_updates_its_symbol(isin_cache: Path) -> None:
    isin.learn({"INE670A01012": "WRONG.NS"})
    isin.learn({"INE670A01012": "TATAELXSI.NS"})

    assert isin.resolve("INE670A01012").yahoo_symbol == "TATAELXSI.NS"


def test_a_learned_pair_overrides_a_seeded_one(isin_cache: Path) -> None:
    """The user's own broker export outranks this repository's best effort."""
    assert isin.resolve("INE002A01018").yahoo_symbol == "RELIANCE.NS"

    isin.learn({"INE002A01018": "RELIANCE.BO"})

    record = isin.resolve("INE002A01018")
    assert record.yahoo_symbol == "RELIANCE.BO"
    assert record.source == "learned"


def test_an_invalid_isin_is_refused_with_the_expected_digit_named(
    isin_cache: Path,
) -> None:
    """Reported, never dropped: silently discarding half a broker file's
    mappings would make the next ISIN-only import fail for no visible reason."""
    stored, rejected = isin.learn({"INE002A01019": "RELIANCE.NS"})

    assert stored == 0
    assert len(rejected) == 1
    assert "check digit" in rejected[0]
    assert "expected 8" in rejected[0]


def test_a_non_isin_key_is_refused_and_explained(isin_cache: Path) -> None:
    stored, rejected = isin.learn({"RELIANCE": "RELIANCE.NS"})

    assert stored == 0
    assert "shape of an ISIN" in rejected[0]


def test_a_pair_with_no_symbol_is_refused(isin_cache: Path) -> None:
    stored, rejected = isin.learn({"INE002A01018": ""})

    assert stored == 0
    assert "no symbol" in rejected[0]


def test_valid_pairs_are_stored_even_when_others_in_the_batch_fail(
    isin_cache: Path,
) -> None:
    stored, rejected = isin.learn(
        {"INE670A01012": "TATAELXSI.NS", "INE002A01019": "BAD.NS", "JUNK": "X"}
    )

    assert stored == 1
    assert len(rejected) == 2
    assert isin.resolve("INE670A01012") is not None


def test_learning_accepts_an_iterable_of_pairs_as_well_as_a_mapping(
    isin_cache: Path,
) -> None:
    stored, _ = isin.learn([("INE670A01012", "TATAELXSI.NS")])

    assert stored == 1


def test_a_dry_run_validates_without_writing(isin_cache: Path) -> None:
    stored, rejected = isin.learn({"INE670A01012": "TATAELXSI.NS"}, persist=False)

    assert stored == 1
    assert rejected == []
    assert not (isin_cache / isin.LEARNED_FILE).exists()


def test_learning_nothing_writes_nothing(isin_cache: Path) -> None:
    stored, rejected = isin.learn({})

    assert stored == 0
    assert rejected == []
    assert not (isin_cache / isin.LEARNED_FILE).exists()


def test_a_learned_name_is_stored_when_supplied(isin_cache: Path) -> None:
    isin.learn({"INE670A01012": "TATAELXSI.NS"}, names={"INE670A01012": "Tata Elxsi Ltd"})

    assert isin.resolve("INE670A01012").name == "Tata Elxsi Ltd"


# ---------------------------------------------------------------------------
# extracting pairs from a frame
# ---------------------------------------------------------------------------
def test_pairs_are_extracted_from_a_file_carrying_both_columns() -> None:
    """The best source of mappings there is, and it costs nothing."""
    frame = pd.DataFrame(
        {
            "ISIN": ["INE002A01018", "INE009A01021"],
            "Symbol": ["RELIANCE", "INFY"],
        }
    )

    assert isin.pairs_from_frame(frame, "ISIN", "Symbol") == {
        "INE002A01018": "RELIANCE",
        "INE009A01021": "INFY",
    }


def test_invalid_isins_are_not_extracted() -> None:
    frame = pd.DataFrame({"ISIN": ["INE002A01019", "INE009A01021"], "Symbol": ["BAD", "INFY"]})

    assert isin.pairs_from_frame(frame, "ISIN", "Symbol") == {"INE009A01021": "INFY"}


def test_rows_with_a_blank_symbol_are_not_extracted() -> None:
    frame = pd.DataFrame({"ISIN": ["INE002A01018"], "Symbol": [""]})

    assert isin.pairs_from_frame(frame, "ISIN", "Symbol") == {}


def test_the_first_spelling_of_a_repeated_isin_wins() -> None:
    frame = pd.DataFrame({"ISIN": ["INE002A01018", "INE002A01018"], "Symbol": ["FIRST", "SECOND"]})

    assert isin.pairs_from_frame(frame, "ISIN", "Symbol") == {"INE002A01018": "FIRST"}


@pytest.mark.parametrize(
    "frame",
    [None, pd.DataFrame(), pd.DataFrame({"ISIN": ["INE002A01018"]})],
)
def test_extracting_from_an_unusable_frame_yields_nothing(frame: object) -> None:
    assert isin.pairs_from_frame(frame, "ISIN", "Symbol") == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# loading & reporting
# ---------------------------------------------------------------------------
def test_a_missing_learned_file_is_not_an_error(isin_cache: Path) -> None:
    (isin_cache / isin.LEARNED_FILE).unlink(missing_ok=True)
    isin.refresh()

    assert isin.load_map()


def test_an_unreadable_map_file_degrades_to_the_other_source(isin_cache: Path) -> None:
    (isin_cache / isin.LEARNED_FILE).write_bytes(b"\x00\x01\x02 not,a,csv\x00")
    isin.refresh()

    assert isin.resolve("INE002A01018") is not None, "the seed must still load"


def test_a_map_file_without_the_required_columns_is_skipped(isin_cache: Path) -> None:
    (isin_cache / isin.LEARNED_FILE).write_text("foo,bar\n1,2\n", encoding="utf-8")
    isin.refresh()

    assert isin.resolve("INE002A01018") is not None


def test_the_map_is_cached_and_refresh_clears_it(isin_cache: Path) -> None:
    before = len(isin.load_map())
    (isin_cache / isin.LEARNED_FILE).write_text(
        "isin,yahoo_symbol,name,source\nINE670A01012,TATAELXSI.NS,,learned\n",
        encoding="utf-8",
    )
    assert len(isin.load_map()) == before, "the cache should still be serving"

    isin.refresh()
    assert len(isin.load_map()) == before + 1


def test_the_summary_counts_by_source_and_sums_to_the_total(isin_cache: Path) -> None:
    isin.learn({"INE670A01012": "TATAELXSI.NS"})

    summary = isin.map_summary()
    assert summary["total"] == len(isin.load_map())
    assert sum(v for k, v in summary.items() if k != "total") == summary["total"]
    assert summary["learned"] == 1


def test_the_record_label_names_the_instrument_and_its_symbol() -> None:
    record = isin.IsinRecord("INE002A01018", "RELIANCE.NS", "Reliance Industries Ltd")

    assert record.label == "Reliance Industries Ltd (RELIANCE.NS)"


def test_a_nameless_record_falls_back_to_its_symbol() -> None:
    assert isin.IsinRecord("INE002A01018", "RELIANCE.NS").label == "RELIANCE.NS (RELIANCE.NS)"


# ---------------------------------------------------------------------------
# regression: NaN is truthy
# ---------------------------------------------------------------------------
def test_a_blank_symbol_cell_is_never_learned_as_the_symbol_nan() -> None:
    """Regression. ``str(value or "")`` looks like a safe blank guard and is
    not: ``float("nan")`` is TRUTHY, so a missing cell survived the ``or``,
    stringified to "nan" and was stored as the literal symbol "NAN" — a
    permanent wrong mapping learned from an empty cell."""
    frame = pd.DataFrame(
        {"ISIN": ["INE002A01018", "INE009A01021"], "Symbol": [float("nan"), "INFY"]}
    )

    pairs = isin.pairs_from_frame(frame, "ISIN", "Symbol")

    assert "NAN" not in pairs.values()
    assert pairs == {"INE009A01021": "INFY"}


@pytest.mark.parametrize("blank", [float("nan"), None, "", "  ", "nan", "None", "NaT", "NULL"])
def test_no_blank_sentinel_is_ever_accepted_as_a_symbol(blank: object, isin_cache: Path) -> None:
    stored, rejected = isin.learn({"INE002A01018": blank})

    assert stored == 0
    assert rejected and "no symbol" in rejected[0]


def test_a_csv_row_with_a_missing_symbol_is_dropped_at_load(isin_cache: Path) -> None:
    (isin_cache / isin.LEARNED_FILE).write_text(
        "isin,yahoo_symbol,name,source\nINE002A01018,,Blank,learned\n",
        encoding="utf-8",
    )
    isin.refresh()

    record = isin.resolve("INE002A01018")
    assert record is not None, "the seed row must still be there"
    assert record.source == "seed", "the blank learned row must not have overridden it"
