"""Relative valuation, checked against hand-computed medians and directions.

Testing philosophy for this module
----------------------------------
A snapshot of "what the code currently returns" is worthless here for the same
reason it is worthless in :mod:`tests.test_dcf`: it locks in whatever the module
happened to produce, bugs included, and a change that silently breaks the
arithmetic still passes.

So the fixtures below are deliberately tiny and their aggregates are the kind a
reader can check on the back of an envelope. Peers on ``[10, 12, 400]`` have a
median of exactly 12 -- and a mean of ~140.7 -- which is why the loss-making
peer with the enormous P/E cannot move it. A subject at 15 against that median
is a +25% premium, full stop. Every such number is asserted literally.

The other half of the file is about *direction*, which is the property the whole
module turns on: a high ROE is favourable and a high debt-to-equity is not, so
being "above the median" is not a verdict by itself. Several tests exist purely
to pin that high ROE and high leverage move in opposite directions, and that a
row sitting exactly on the median is neither.


Two ``xfail`` tests at the foot of the file record genuine defects discovered
while writing this file. They are marked rather than fixed, per the brief, and
they fail today.
"""

from __future__ import annotations

import math

import pytest

from src.analytics.peers import (
    DEFAULT_CLUSTER_FEATURES,
    METRICS,
    MIN_PEERS_FOR_MEDIAN,
    PeerCompany,
    cluster_peers,
    peer_table,
    rank_in_peers,
    relative_valuation,
)

EM_DASH = "—"


# ---------------------------------------------------------------------------
# fixtures -- small, explicit, hand-checkable
# ---------------------------------------------------------------------------
def _peer(symbol: str, **metrics: float) -> PeerCompany:
    """A company carrying exactly the metrics named, so each fixture reads clean."""
    return PeerCompany(symbol=symbol, metrics=dict(metrics))


def _clustered(symbol: str, level: float = 0.0) -> PeerCompany:
    """A company whose five clustering features all step together with ``level``.

    Complete rows only, so it can take part in clustering rather than being
    dropped as an imputation candidate.
    """
    return PeerCompany(
        symbol=symbol,
        metrics={
            "net_margin": 0.05 + 0.01 * level,
            "ebitda_margin": 0.10 + 0.01 * level,
            "revenue_growth": 0.02 + 0.01 * level,
            "roe": 0.10 + 0.01 * level,
            "debt_to_equity": 0.20 + 0.02 * level,
        },
    )


# ---------------------------------------------------------------------------
# the metric registry
# ---------------------------------------------------------------------------
def test_every_metric_spec_key_matches_its_registry_key() -> None:
    """Two sources of a key that can drift apart must not be allowed to."""
    for key, spec in METRICS.items():
        assert spec.key == key


def test_the_valuation_multiples_are_lower_is_better() -> None:
    for key in (
        "pe",
        "forward_pe",
        "ev_ebitda",
        "pb",
        "peg",
        "debt_to_equity",
        "net_debt_to_ebitda",
    ):
        assert METRICS[key].higher_is_better is False, key


def test_the_return_and_quality_metrics_are_higher_is_better() -> None:
    for key in (
        "roe",
        "roce",
        "net_margin",
        "ebitda_margin",
        "revenue_growth",
        "earnings_growth",
        "dividend_yield",
        "current_ratio",
        "interest_coverage",
    ):
        assert METRICS[key].higher_is_better is True, key


def test_direction_word_reads_straight_off_the_spec() -> None:
    assert METRICS["pe"].direction_word == "lower is better"
    assert METRICS["roe"].direction_word == "higher is better"


def test_every_cluster_feature_is_a_registered_metric() -> None:
    """A feature that is not a metric would be silently dropped by ``get``."""
    for feature in DEFAULT_CLUSTER_FEATURES:
        assert feature in METRICS


def test_the_thin_peer_threshold_is_four() -> None:
    """The fixtures below assume it; pin it so a change is deliberate."""
    assert MIN_PEERS_FOR_MEDIAN == 4


# ---------------------------------------------------------------------------
# the median, and why not the mean
# ---------------------------------------------------------------------------
def test_a_loss_making_peer_with_a_huge_pe_does_not_move_the_median() -> None:
    """The whole reason the module aggregates with a median.

    Peers on [10, 12, 400] have a median of 12, while one loss-maker whose
    reported P/E runs into the hundreds drags the mean to ~140.7. A subject at
    15 is a 25% premium to the median and a rounding error against the mean.
    """
    subject = _peer("SUB", pe=15.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=400.0)]

    row = relative_valuation(subject, peers, metrics=["pe"]).read("pe")

    assert row is not None
    assert row.peer_median == pytest.approx(12.0)
    assert row.peer_min == pytest.approx(10.0)
    assert row.peer_max == pytest.approx(400.0)
    assert row.n_peers == 3
    assert row.relative == pytest.approx(15.0 / 12.0 - 1.0)  # +25%
    # The mean that the median exists to resist, computed here by hand.
    assert (10.0 + 12.0 + 400.0) / 3 == pytest.approx(140.6667, abs=1e-3)


def test_an_even_peer_count_takes_the_midpoint_of_the_two_central_values() -> None:
    """[10, 12, 14, 16] -> (12 + 14) / 2 = 13, not 12 or 14."""
    subject = _peer("SUB", pe=13.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0), _peer("P4", pe=16.0)]

    row = relative_valuation(subject, peers, metrics=["pe"]).read("pe")

    assert row is not None
    assert row.peer_median == pytest.approx(13.0)
    assert row.relative == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# relative: subject / median - 1, and when it refuses
# ---------------------------------------------------------------------------
def test_relative_is_the_gap_over_the_median_exactly() -> None:
    subject = _peer("SUB", pe=15.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0)]

    row = relative_valuation(subject, peers, metrics=["pe"]).read("pe")

    assert row is not None
    assert row.peer_median == pytest.approx(12.0)
    assert row.relative == pytest.approx(15.0 / 12.0 - 1.0)
    assert row.relative == pytest.approx(0.25)


def test_relative_is_undefined_when_the_peer_median_is_zero() -> None:
    """Dividing by a zero median would produce an infinite "premium"."""
    subject = _peer("SUB", revenue_growth=0.10)
    peers = [
        _peer("P1", revenue_growth=0.0),
        _peer("P2", revenue_growth=0.0),
        _peer("P3", revenue_growth=0.0),
    ]

    row = relative_valuation(subject, peers, metrics=["revenue_growth"]).read("revenue_growth")

    assert row is not None
    assert row.available is True
    assert row.peer_median == pytest.approx(0.0)
    assert math.isnan(row.relative)


def test_relative_is_undefined_when_the_peer_median_is_negative() -> None:
    """+10% against a peer median of -5% is not "300% better" -- it is nothing."""
    subject = _peer("SUB", revenue_growth=0.10)
    peers = [
        _peer("P1", revenue_growth=-0.05),
        _peer("P2", revenue_growth=-0.05),
        _peer("P3", revenue_growth=-0.05),
    ]

    row = relative_valuation(subject, peers, metrics=["revenue_growth"]).read("revenue_growth")

    assert row is not None
    assert row.peer_median == pytest.approx(-0.05)
    assert math.isnan(row.relative)


# ---------------------------------------------------------------------------
# direction: favourable / unfavourable depends on the metric, not the arithmetic
# ---------------------------------------------------------------------------
def test_a_high_roe_is_favourable() -> None:
    subject = _peer("SUB", roe=0.25)
    peers = [_peer("P1", roe=0.18), _peer("P2", roe=0.20), _peer("P3", roe=0.22)]

    row = relative_valuation(subject, peers, metrics=["roe"]).read("roe")

    assert row is not None
    assert row.peer_median == pytest.approx(0.20)
    assert row.above_median is True
    assert row.favourable is True
    assert row.richer_than_peers is False


def test_a_high_debt_to_equity_is_not_favourable() -> None:
    subject = _peer("SUB", debt_to_equity=0.60)
    peers = [
        _peer("P1", debt_to_equity=0.30),
        _peer("P2", debt_to_equity=0.40),
        _peer("P3", debt_to_equity=0.50),
    ]

    row = relative_valuation(subject, peers, metrics=["debt_to_equity"]).read("debt_to_equity")

    assert row is not None
    assert row.peer_median == pytest.approx(0.40)
    assert row.above_median is True
    assert row.favourable is False


def test_less_debt_than_the_peer_median_is_favourable() -> None:
    """The mirror image: below the median on leverage is the good side."""
    subject = _peer("SUB", debt_to_equity=0.30)
    peers = [
        _peer("P1", debt_to_equity=0.40),
        _peer("P2", debt_to_equity=0.50),
        _peer("P3", debt_to_equity=0.60),
    ]

    row = relative_valuation(subject, peers, metrics=["debt_to_equity"]).read("debt_to_equity")

    assert row is not None
    assert row.above_median is False
    assert row.favourable is True


def test_a_row_exactly_on_the_median_is_neither() -> None:
    """Not favourable, not rich, not above: a tie is not a win or a loss."""
    subject = _peer("SUB", roe=0.20)
    peers = [_peer("P1", roe=0.18), _peer("P2", roe=0.20), _peer("P3", roe=0.22)]

    row = relative_valuation(subject, peers, metrics=["roe"]).read("roe")

    assert row is not None
    assert row.available is True
    assert row.above_median is False
    assert row.favourable is False
    assert row.richer_than_peers is False
    assert row.verdict == "in line with peers"


# ---------------------------------------------------------------------------
# verdict wording, which differs by metric kind
# ---------------------------------------------------------------------------
def test_a_multiple_above_the_median_reads_as_expensive() -> None:
    subject = _peer("SUB", pe=15.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0)]

    row = relative_valuation(subject, peers, metrics=["pe"]).read("pe")

    assert row is not None
    assert row.verdict == "above the peer median — expensive"


def test_a_multiple_below_the_median_reads_as_cheap() -> None:
    subject = _peer("SUB", pe=8.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0)]

    row = relative_valuation(subject, peers, metrics=["pe"]).read("pe")

    assert row is not None
    assert row.verdict == "below the peer median — cheap"


def test_a_return_above_the_median_reads_as_good() -> None:
    subject = _peer("SUB", roe=0.25)
    peers = [_peer("P1", roe=0.18), _peer("P2", roe=0.20), _peer("P3", roe=0.22)]

    row = relative_valuation(subject, peers, metrics=["roe"]).read("roe")

    assert row is not None
    assert row.verdict == "above the peer median — good"


def test_a_return_below_the_median_reads_as_unfavourable() -> None:
    subject = _peer("SUB", roe=0.10)
    peers = [_peer("P1", roe=0.18), _peer("P2", roe=0.20), _peer("P3", roe=0.22)]

    row = relative_valuation(subject, peers, metrics=["roe"]).read("roe")

    assert row is not None
    assert row.verdict == "below the peer median — unfavourable"


# ---------------------------------------------------------------------------
# thin peer sets
# ---------------------------------------------------------------------------
def test_fewer_than_the_minimum_peers_is_flagged_thin_and_noticed() -> None:
    subject = _peer("SUB", pe=15.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0)]

    result = relative_valuation(subject, peers, metrics=["pe"])
    row = result.read("pe")

    assert row is not None
    assert row.n_peers == MIN_PEERS_FOR_MEDIAN - 1
    assert row.thin is True
    assert row in result.thin_rows
    assert any("fewer than 4 peers" in n for n in result.notices)


def test_exactly_the_minimum_number_of_peers_is_not_thin() -> None:
    subject = _peer("SUB", pe=15.0)
    peers = [_peer("P1", pe=8.0), _peer("P2", pe=10.0), _peer("P3", pe=12.0), _peer("P4", pe=14.0)]

    result = relative_valuation(subject, peers, metrics=["pe"])
    row = result.read("pe")

    assert row is not None
    assert row.n_peers == MIN_PEERS_FOR_MEDIAN
    assert row.thin is False
    assert result.thin_rows == ()
    assert not any("fewer than" in n for n in result.notices)


def test_a_peer_missing_the_metric_is_excluded_from_that_row_only() -> None:
    """The count is per row precisely because coverage varies per metric."""
    subject = _peer("SUB", pe=15.0, roe=0.30)
    peers = [
        _peer("P1", pe=10.0, roe=0.10),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=14.0),  # reports no ROE
        _peer("P4", pe=16.0, roe=0.25),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe"])

    assert result.read("pe") is not None
    assert result.read("pe").n_peers == 4
    assert result.read("roe") is not None
    assert result.read("roe").n_peers == 3


# ---------------------------------------------------------------------------
# the subject is not its own peer
# ---------------------------------------------------------------------------
def test_the_subject_is_excluded_from_its_own_peer_set() -> None:
    """Including the subject would drag the median towards itself.

    The subject carries a P/E of 1000; if it were allowed into the peer set the
    median of [10, 12, 400, 1000] would still be 12 here, so make the effect
    unmistakable by checking the count and the reported peer tuple as well.
    """
    subject = _peer("SUB", pe=1000.0)
    without = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=400.0)]

    a = relative_valuation(subject, without, metrics=["pe"]).read("pe")
    b = relative_valuation(subject, [subject, *without], metrics=["pe"]).read("pe")

    assert a is not None and b is not None
    assert a.peer_median == pytest.approx(12.0)
    assert b.peer_median == pytest.approx(a.peer_median)
    assert b.n_peers == 3


# ---------------------------------------------------------------------------
# refusing to compare
# ---------------------------------------------------------------------------
def test_a_metric_the_subject_does_not_report_is_not_comparable() -> None:
    subject = _peer("SUB", pe=15.0)  # reports no ROE
    peers = [
        _peer("P1", pe=10.0, roe=0.18),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=14.0, roe=0.22),
    ]

    row = relative_valuation(subject, peers, metrics=["pe", "roe"]).read("roe")

    assert row is not None
    assert math.isnan(row.subject)
    assert row.peer_median == pytest.approx(0.20)  # peers still reported it
    assert row.available is False
    assert row.favourable is False
    assert row.verdict == "not comparable"


def test_no_peers_leaves_every_row_uncompared_and_says_so() -> None:
    subject = _peer("SUB", pe=15.0)

    result = relative_valuation(subject, [], metrics=["pe"])
    row = result.read("pe")

    assert row is not None
    assert row.n_peers == 0
    assert row.available is False
    assert row.verdict == "not comparable"
    assert any("No peers were supplied" in n for n in result.notices)


def test_an_unknown_metric_is_ignored_with_a_notice() -> None:
    result = relative_valuation(
        _peer("SUB", pe=15.0), [_peer("P1", pe=10.0)], metrics=["pe", "nonsense"]
    )

    assert result.read("pe") is not None
    assert result.read("nonsense") is None
    assert any("Unknown metric" in n for n in result.notices)


# ---------------------------------------------------------------------------
# the rendered table
# ---------------------------------------------------------------------------
def test_peer_table_has_a_header_and_one_row_per_metric() -> None:
    subject = _peer("SUB", pe=15.0, roe=0.30)
    peers = [
        _peer("P1", pe=10.0, roe=0.18),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=14.0, roe=0.22),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe", "debt_to_equity"])
    table = peer_table(result)

    assert table[0] == (
        "Metric",
        "Company",
        "Peer median",
        "Peer low",
        "Peer high",
        "Peers",
        "Reads as",
    )
    assert len(table) == 1 + len(result.comparisons) == 4


def test_peer_table_renders_an_em_dash_for_a_missing_value_never_a_zero() -> None:
    subject = _peer("SUB", pe=15.0)  # no ROE
    peers = [
        _peer("P1", pe=10.0, roe=0.18),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=14.0, roe=0.22),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe"])
    table = peer_table(result)
    roe_row = next(row for row in table if row[0] == "ROE")

    assert roe_row[1] == EM_DASH
    assert roe_row[1] != "0"
    assert roe_row[-1] == "not comparable"


def test_peer_table_formats_a_multiple_and_a_rate_the_way_a_reader_expects() -> None:
    subject = _peer("SUB", pe=15.0, roe=0.30)
    peers = [
        _peer("P1", pe=10.0, roe=0.18),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=14.0, roe=0.22),
    ]

    table = peer_table(relative_valuation(subject, peers, metrics=["pe", "roe"]))
    pe_row = next(row for row in table if row[0] == "P/E")
    roe_row = next(row for row in table if row[0] == "ROE")

    assert pe_row[1] == "15.0x"
    assert pe_row[2] == "12.0x"
    assert roe_row[1] == "30.0%"
    assert roe_row[2] == "20.0%"


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def test_rank_one_is_the_cheapest_on_a_lower_is_better_multiple() -> None:
    subject = _peer("SUB", pe=8.0)
    peers = [_peer("P1", pe=10.0), _peer("P2", pe=12.0), _peer("P3", pe=14.0)]

    assert rank_in_peers(subject, peers, "pe") == (1, 4)


def test_rank_one_is_the_most_profitable_on_a_higher_is_better_return() -> None:
    subject = _peer("SUB", roe=0.30)
    peers = [_peer("P1", roe=0.10), _peer("P2", roe=0.20), _peer("P3", roe=0.25)]

    assert rank_in_peers(subject, peers, "roe") == (1, 4)


def test_rank_counts_only_the_peers_strictly_ahead_of_the_subject() -> None:
    """Two peers beat an ROE of 0.15, so the subject is third of four."""
    subject = _peer("SUB", roe=0.15)
    peers = [_peer("P1", roe=0.10), _peer("P2", roe=0.20), _peer("P3", roe=0.25)]

    assert rank_in_peers(subject, peers, "roe") == (3, 4)


def test_rank_excludes_the_subject_from_its_own_peer_set() -> None:
    subject = _peer("SUB", pe=8.0)
    peers = [subject, _peer("P1", pe=10.0)]

    assert rank_in_peers(subject, peers, "pe") == (1, 2)


def test_a_missing_value_ranks_nowhere_rather_than_last() -> None:
    """ "Last" would be a false statement about a company that reported no line."""
    subject = _peer("SUB", pe=8.0)  # reports no ROE
    peers = [_peer("P1", roe=0.10), _peer("P2", roe=0.20)]

    assert rank_in_peers(subject, peers, "roe") == (0, 0)


def test_an_unregistered_metric_ranks_nowhere() -> None:
    subject = _peer("SUB", pe=8.0)

    assert rank_in_peers(subject, [_peer("P1", pe=10.0)], "nonsense") == (0, 0)


# ---------------------------------------------------------------------------
# peer discovery
# ---------------------------------------------------------------------------
def test_peers_come_from_the_industry_label_when_it_is_broad_enough() -> None:
    subject = PeerCompany("SUB", industry="Banks", sector="Financials")
    peers = [PeerCompany(s, industry="Banks", sector="Financials") for s in ("A", "B", "C", "D")]

    cluster = cluster_peers(subject, peers)

    assert cluster.method == "industry"
    assert cluster.industry_label == "Banks"
    assert set(cluster.symbols) == {"A", "B", "C", "D"}
    assert "SUB" not in cluster.symbols


def test_a_too_thin_industry_label_falls_back_to_the_coarser_sector() -> None:
    subject = PeerCompany("SUB", industry="Banks", sector="Financials")
    peers = [
        PeerCompany("A", industry="Banks", sector="Financials"),
        PeerCompany("B", industry="Banks", sector="Financials"),
        PeerCompany("C", industry="Insurance", sector="Financials"),
        PeerCompany("D", industry="Insurance", sector="Financials"),
        PeerCompany("E", industry="Insurance", sector="Financials"),
    ]

    cluster = cluster_peers(subject, peers)

    # Only two peers share "Banks"; five share the "Financials" sector.
    assert cluster.method == "sector"
    assert set(cluster.symbols) == {"A", "B", "C", "D", "E"}
    assert any("too few" in n for n in cluster.notices)


def test_a_subject_missing_a_cluster_feature_is_reported_not_raised() -> None:
    """The regression this guards: a StopIteration when the subject fell out of
    the complete-rows matrix and ``next(...)`` could not find it.

    The subject carries an industry whose label is too thin and reports no ROE,
    so the label branches fall through and the clustering guard is what stops
    it. It must return, not raise.
    """
    subject = PeerCompany(
        "SUB",
        industry="Banks",
        metrics={
            "net_margin": 0.10,
            "ebitda_margin": 0.15,
            "revenue_growth": 0.05,
            "debt_to_equity": 0.50,
        },  # deliberately no roe
    )
    peers = [
        PeerCompany("A", industry="Banks"),
        *[_clustered(f"C{i}", float(i)) for i in range(1, 7)],
    ]

    cluster = cluster_peers(subject, peers)

    assert cluster.method == "universe"
    assert cluster.members == ()
    assert any("roe" in n for n in cluster.notices)


def test_too_few_complete_rows_falls_back_to_the_universe() -> None:
    subject = _clustered("SUB", 0.0)
    peers = [_clustered("C1", 1.0), _clustered("C2", 2.0)]  # 3 rows, below 2 * minimum

    cluster = cluster_peers(subject, peers)

    assert cluster.method == "universe"
    assert any("too few" in n for n in cluster.notices)


def test_clustered_peers_never_contain_the_subject() -> None:
    subject = _clustered("SUB", 0.0)
    peers = [_clustered(f"C{i}", float(i)) for i in range(1, 13)]

    cluster = cluster_peers(subject, peers)

    assert cluster.method == "cluster"
    assert cluster.members
    assert cluster.ok is True
    assert "SUB" not in cluster.symbols


def test_an_empty_universe_yields_no_members() -> None:
    cluster = cluster_peers(PeerCompany("SUB"), [])

    assert cluster.method == "universe"
    assert cluster.members == ()


# ---------------------------------------------------------------------------
# balance, in words
# ---------------------------------------------------------------------------
def test_balance_reports_how_many_metrics_beat_the_median_and_how_many_lag() -> None:
    subject = _peer("SUB", pe=15.0, roe=0.30, debt_to_equity=0.30)
    peers = [
        _peer("P1", pe=10.0, roe=0.18, debt_to_equity=0.40),
        _peer("P2", pe=12.0, roe=0.20, debt_to_equity=0.50),
        _peer("P3", pe=14.0, roe=0.22, debt_to_equity=0.60),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe", "debt_to_equity"])

    # P/E 15 > 12: expensive (bad). ROE 0.30 > 0.20: good. D/E 0.30 < 0.50: good.
    assert len(result.favourable_rows) == 2
    assert len(result.unfavourable_rows) == 1
    assert result.balance == "Better on 2 of 3 comparable metrics, worse on 1."


def test_balance_reads_all_when_the_subject_beats_every_metric() -> None:
    subject = _peer("SUB", pe=8.0, roe=0.30, debt_to_equity=0.20)
    peers = [
        _peer("P1", pe=10.0, roe=0.18, debt_to_equity=0.40),
        _peer("P2", pe=12.0, roe=0.20, debt_to_equity=0.50),
        _peer("P3", pe=14.0, roe=0.22, debt_to_equity=0.60),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe", "debt_to_equity"])

    assert result.balance == "Better than the peer median on all 3 comparable metric(s)."


def test_balance_reads_worse_when_the_subject_lags_every_metric() -> None:
    subject = _peer("SUB", pe=20.0, roe=0.10, debt_to_equity=0.90)
    peers = [
        _peer("P1", pe=10.0, roe=0.18, debt_to_equity=0.40),
        _peer("P2", pe=12.0, roe=0.20, debt_to_equity=0.50),
        _peer("P3", pe=14.0, roe=0.22, debt_to_equity=0.60),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe", "debt_to_equity"])

    assert result.balance == "Worse than the peer median on all 3 comparable metric(s)."


# ---------------------------------------------------------------------------
# known defects (xfail), recorded rather than fixed
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="PeerComparison.unfavourable_rows is 'available and not favourable', so a row "
    "sitting exactly on the median (which MetricComparison.favourable documents as "
    "'neither') is counted as unfavourable. On a set of on-median rows balance() therefore "
    "prints 'Worse than the peer median on all N comparable metric(s).' instead of the "
    "'Sits on the peer median for every comparable metric.' its own dead branch anticipates.",
)
def test_a_subject_on_every_median_is_described_as_sitting_there() -> None:
    subject = _peer("SUB", pe=12.0, roe=0.20)
    peers = [
        _peer("P1", pe=12.0, roe=0.20),
        _peer("P2", pe=12.0, roe=0.20),
        _peer("P3", pe=12.0, roe=0.20),
    ]

    result = relative_valuation(subject, peers, metrics=["pe", "roe"])

    assert len(result.available) == 2
    assert result.favourable_rows == ()
    # Observed: 'Worse than the peer median on all 2 comparable metric(s).'
    assert result.balance == "Sits on the peer median for every comparable metric."


@pytest.mark.xfail(
    reason="The sector fallback labels the group with the subject's *industry* "
    "(industry_label=subject.industry or subject.sector) even though the members share the "
    "subject's *sector*. caption then reads \"Peers share the 'Banks' sector label.\" when "
    "the sector is actually 'Financials'.",
)
def test_the_sector_fallback_names_the_sector_it_actually_used() -> None:
    subject = PeerCompany("SUB", industry="Banks", sector="Financials")
    peers = [
        PeerCompany("A", industry="Banks", sector="Financials"),
        PeerCompany("B", industry="Banks", sector="Financials"),
        PeerCompany("C", industry="Insurance", sector="Financials"),
        PeerCompany("D", industry="Insurance", sector="Financials"),
        PeerCompany("E", industry="Insurance", sector="Financials"),
    ]

    cluster = cluster_peers(subject, peers)

    assert cluster.method == "sector"
    # Observed caption: "Peers share the 'Banks' sector label."
    assert "Financials" in cluster.caption
