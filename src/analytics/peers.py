"""Relative valuation: how one company is priced against the companies it resembles.

Absolute valuation (:mod:`src.analytics.dcf`) answers "what is this worth?".
Relative valuation answers the different and usually easier question "what is
everyone else paying for this kind of business?", and the two disagreeing is
itself information worth showing side by side.

Three decisions shape this module, and each exists because the obvious
alternative produces a number that looks precise and means nothing.

**The median, not the mean.** A peer set screened on P/E will contain a company
that lost money last year, whose P/E is either negative or in the hundreds
depending on which flavour the vendor reports. A mean is destroyed by one such
row; a median is not. Every aggregate here is a median.

**Direction is a property of the metric, not of the arithmetic.** Trading at
twice the peer median P/E is a premium; earning twice the peer median ROE is
not a premium, it is a better business. So premium and discount are only
meaningful relative to a metric's own :class:`MetricSpec`, and the module never
reports "above median" as though that were a verdict on its own.

**A comparison from three peers is not a comparison.** Every metric carries the
number of peers it was actually computed on, and a set below
:data:`MIN_PEERS_FOR_MEDIAN` is flagged rather than silently reported as though
the median were robust. Peers missing the metric are excluded from that metric
only, so the count varies per row -- which is why it is per row.

Peer *selection* has the same problem in reverse: sector labels are often too
coarse (a two-line "Technology" bucket spanning a foundry and a software firm),
and factor clustering fixes that but produces groups with no name. Both are
offered; :func:`cluster_peers` prefers the label when it is usable and falls
back to clustering, reporting which it used.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MetricSpec",
    "METRICS",
    "DEFAULT_CLUSTER_FEATURES",
    "MIN_PEERS_FOR_MEDIAN",
    "PeerCompany",
    "MetricComparison",
    "PeerComparison",
    "PeerCluster",
    "relative_valuation",
    "cluster_peers",
    "format_metric",
    "peer_table",
    "rank_in_peers",
]

#: A peer set thinner than this cannot support a median worth trusting, and is
#: reported with a notice instead of being presented as a robust benchmark.
MIN_PEERS_FOR_MEDIAN = 4

_NAN = float("nan")


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Coerce to a finite float, or NaN. Never raises, never invents a zero."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return number if math.isfinite(number) else _NAN


def _div(numerator: float, denominator: float) -> float:
    """NaN-safe division: a zero or missing denominator yields NaN, not infinity."""
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return _NAN
    out = numerator / denominator
    return out if math.isfinite(out) else _NAN


# ---------------------------------------------------------------------------
# the metric registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetricSpec:
    """What a relative-valuation row means, beyond its arithmetic.

    ``higher_is_better`` is the field the whole module turns on. Without it a
    renderer can only say "above the peer median", which is good news for ROE,
    bad news for debt-to-equity, and for P/E is bad news *only if* the earnings
    are comparable -- a distinction the number cannot carry by itself and the
    caption has to.
    """

    key: str
    label: str
    higher_is_better: bool
    unit: str = ""  # "x" for multiples, "%" for rates, "" for ratios
    note: str = ""

    @property
    def direction_word(self) -> str:
        """``"higher is better"`` / ``"lower is better"``, for captions."""
        return "higher is better" if self.higher_is_better else "lower is better"


#: The comparison set. Ordered as a report would read it: what you pay, what
#: you get, and what you are carrying.
METRICS: dict[str, MetricSpec] = {
    "pe": MetricSpec(
        "pe",
        "P/E",
        higher_is_better=False,
        unit="x",
        note="Price over trailing earnings. Lower is cheaper, "
        "unless the earnings are not comparable.",
    ),
    "forward_pe": MetricSpec(
        "forward_pe",
        "Forward P/E",
        higher_is_better=False,
        unit="x",
        note="On next-year estimates, which are analyst opinions.",
    ),
    "ev_ebitda": MetricSpec(
        "ev_ebitda",
        "EV/EBITDA",
        higher_is_better=False,
        unit="x",
        note="Capital-structure neutral, so the fairest cross-company "
        "multiple when debt ratios differ.",
    ),
    "pb": MetricSpec(
        "pb",
        "P/B",
        higher_is_better=False,
        unit="x",
        note="Price over book equity. Meaningful for banks and asset-heavy "
        "firms, close to meaningless for asset-light ones.",
    ),
    "peg": MetricSpec(
        "peg",
        "PEG",
        higher_is_better=False,
        unit="x",
        note="P/E over growth. A rule of thumb, not a valuation.",
    ),
    "dividend_yield": MetricSpec(
        "dividend_yield", "Dividend yield", higher_is_better=True, unit="%"
    ),
    "roe": MetricSpec(
        "roe",
        "ROE",
        higher_is_better=True,
        unit="%",
        note="Return on equity. Rises with leverage, so read beside debt.",
    ),
    "roce": MetricSpec(
        "roce",
        "ROCE",
        higher_is_better=True,
        unit="%",
        note="Return on capital employed -- the leverage-free cousin of ROE.",
    ),
    "net_margin": MetricSpec("net_margin", "Net margin", higher_is_better=True, unit="%"),
    "ebitda_margin": MetricSpec("ebitda_margin", "EBITDA margin", higher_is_better=True, unit="%"),
    "revenue_growth": MetricSpec(
        "revenue_growth", "Revenue growth", higher_is_better=True, unit="%"
    ),
    "earnings_growth": MetricSpec(
        "earnings_growth", "Earnings growth", higher_is_better=True, unit="%"
    ),
    "debt_to_equity": MetricSpec(
        "debt_to_equity", "Debt / equity", higher_is_better=False, unit="x"
    ),
    "net_debt_to_ebitda": MetricSpec(
        "net_debt_to_ebitda", "Net debt / EBITDA", higher_is_better=False, unit="x"
    ),
    "current_ratio": MetricSpec("current_ratio", "Current ratio", higher_is_better=True, unit="x"),
    "interest_coverage": MetricSpec(
        "interest_coverage", "Interest coverage", higher_is_better=True, unit="x"
    ),
}

#: Features used when peers are discovered by clustering rather than by a
#: sector label. Chosen to describe the *business* rather than the price: two
#: firms with similar margins, growth and leverage belong together even if one
#: is currently expensive, and including a valuation multiple here would sort
#: peers by the very thing being compared.
DEFAULT_CLUSTER_FEATURES: tuple[str, ...] = (
    "net_margin",
    "ebitda_margin",
    "revenue_growth",
    "roe",
    "debt_to_equity",
)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PeerCompany:
    """One company's comparable metrics, already extracted from its statements.

    Extraction lives in the UI/data layer; this module only compares. Metrics
    are keyed by :data:`METRICS` keys, and anything absent or non-finite is
    treated as missing for every metric independently.
    """

    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    def get(self, metric: str) -> float:
        return _num(self.metrics.get(metric))

    @property
    def label(self) -> str:
        return self.name or self.symbol


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One row: the subject's metric, its peer distribution, and the gap."""

    spec: MetricSpec
    subject: float = _NAN
    peer_median: float = _NAN
    peer_min: float = _NAN
    peer_max: float = _NAN
    n_peers: int = 0

    @property
    def available(self) -> bool:
        """True when both sides exist, so a premium or discount means something."""
        return not (math.isnan(self.subject) or math.isnan(self.peer_median))

    @property
    def relative(self) -> float:
        """``subject / median - 1``. Positive is above the peer median.

        Undefined when the median is zero or negative, which happens for
        metrics that can legitimately be negative -- growth, margins, earnings.
        A company growing at +10% against a peer median of -5% is not "300%
        better", so this returns NaN and the caller reports the two levels.
        """
        if not self.available or self.peer_median <= 0:
            return _NAN
        return _div(self.subject - self.peer_median, self.peer_median)

    @property
    def thin(self) -> bool:
        """The peer set behind this row is too small to lean on."""
        return self.n_peers < MIN_PEERS_FOR_MEDIAN

    @property
    def above_median(self) -> bool:
        return self.available and self.subject > self.peer_median

    @property
    def favourable(self) -> bool:
        """True when the subject beats the median *on this metric's own terms*.

        A high ROE is favourable and a high debt-to-equity is not, so this
        cannot be a comparison against the median alone -- it has to consult
        :attr:`MetricSpec.higher_is_better`. A metric sitting exactly on the
        median is neither.
        """
        if not self.available or self.subject == self.peer_median:
            return False
        return self.above_median is self.spec.higher_is_better

    @property
    def richer_than_peers(self) -> bool:
        """Pays more than the median on a multiple -- expensive, not better.

        Kept separate from :attr:`favourable` because a caption about a
        *valuation* multiple wants the word "expensive", which would be the
        wrong word for leverage.
        """
        return self.above_median and not self.spec.higher_is_better

    @property
    def verdict(self) -> str:
        """A phrase a caption can print. Never a buy/sell call on its own."""
        if not self.available:
            return "not comparable"
        if self.subject == self.peer_median:
            return "in line with peers"
        comparison = "above" if self.above_median else "below"
        if self.spec.higher_is_better:
            flavour = "good" if self.favourable else "unfavourable"
        else:
            flavour = "expensive" if self.above_median else "cheap"
        return f"{comparison} the peer median — {flavour}"


@dataclass(frozen=True, slots=True)
class PeerComparison:
    """The whole table plus what it took to build it."""

    subject: PeerCompany | None = None
    comparisons: tuple[MetricComparison, ...] = ()
    peers: tuple[PeerCompany, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.subject is not None and any(c.available for c in self.comparisons)

    @property
    def n_peers(self) -> int:
        return len(self.peers)

    @property
    def available(self) -> tuple[MetricComparison, ...]:
        return tuple(c for c in self.comparisons if c.available)

    @property
    def thin_rows(self) -> tuple[MetricComparison, ...]:
        return tuple(c for c in self.comparisons if c.available and c.thin)

    @property
    def favourable_rows(self) -> tuple[MetricComparison, ...]:
        """Rows where the subject beats the peer median on that metric's own terms."""
        return tuple(c for c in self.comparisons if c.favourable)

    @property
    def unfavourable_rows(self) -> tuple[MetricComparison, ...]:
        """Rows where the subject is meaningfully worse than the peer median.

        Excludes rows where the subject sits on the peer median: that is
        reported by the balance string, not by counting the row as favourable
        or unfavourable.
        """
        return tuple(
            c
            for c in self.comparisons
            if c.available
            and not c.favourable
            and c.subject != c.peer_median  # exclude on-median rows
        )

    def read(self, metric: str) -> MetricComparison | None:
        """One row by metric key, for a caller that wants a single number."""
        for comparison in self.comparisons:
            if comparison.spec.key == metric:
                return comparison
        return None

    @property
    def balance(self) -> str:
        """How lopsided the picture is, in words rather than a score.

        Deliberately not collapsed into a single number: a company cheap on
        four multiples and expensive on one is a different situation from one
        cheap on every multiple, and averaging them hides which. Each metric is
        judged by its own direction, so "favourable" already means cheaper on a
        multiple and more profitable on a return.
        """
        good, bad = len(self.favourable_rows), len(self.unfavourable_rows)
        total = good + bad
        if not total:
            return "Sits on the peer median for every comparable metric."
        if good and not bad:
            return f"Better than the peer median on all {total} comparable metric(s)."
        if bad and not good:
            return f"Worse than the peer median on all {total} comparable metric(s)."
        return f"Better on {good} of {total} comparable metrics, worse on {bad}."


@dataclass(frozen=True, slots=True)
class PeerCluster:
    """A peer set and an honest account of how it was chosen."""

    subject_symbol: str = ""
    members: tuple[PeerCompany, ...] = ()
    method: str = ""  # "industry" | "sector" | "cluster" | "universe"
    k: int = 0
    silhouette: float = _NAN
    notices: tuple[str, ...] = ()
    industry_label: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.members)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(m.symbol for m in self.members)

    @property
    def caption(self) -> str:
        """One line stating how the peers were found -- never left implicit."""
        if self.method == "industry":
            return f"Peers share the '{self.industry_label}' industry label."
        if self.method == "sector":
            return f"Peers share the '{self.industry_label}' sector label."
        if self.method == "cluster":
            quality = (
                ""
                if math.isnan(self.silhouette)
                else f" Separation quality {self.silhouette:.2f} (1.0 is cleanly separated)."
            )
            return (
                f"Industry labels were too thin or too coarse, so the universe was "
                f"clustered into {self.k} groups on margin, growth, return and leverage; "
                f"{len(self.members)} companies fell in the subject's group.{quality}"
            )
        return "No peer group could be formed; only the subject's own metrics are shown."


# ---------------------------------------------------------------------------
# relative valuation
# ---------------------------------------------------------------------------
def _median(values: Sequence[float]) -> float:
    """Median of the finite values, or NaN when there are none."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return _NAN
    return float(np.median(np.asarray(finite, dtype=float)))


def relative_valuation(
    subject: PeerCompany,
    peers: Iterable[PeerCompany],
    metrics: Sequence[str] | None = None,
) -> PeerComparison:
    """Compare one company against its peers, metric by metric.

    ``metrics`` defaults to every registered metric. The subject is excluded
    from its own peer set if it appears there -- a company compared against a
    group containing itself drags the median towards itself, which flatters
    exactly the names that most need the comparison.
    """
    wanted = list(metrics) if metrics is not None else list(METRICS)
    peer_list = [p for p in peers if p.symbol != subject.symbol]
    notices: list[str] = []

    if not wanted:
        return PeerComparison(subject=subject, notices=("No metrics were requested.",))

    unknown = [m for m in wanted if m not in METRICS]
    if unknown:
        notices.append("Unknown metric(s) ignored: " + ", ".join(sorted(unknown)) + ".")
    keys = [m for m in wanted if m in METRICS]

    comparisons = []
    for key in keys:
        spec = METRICS[key]
        values = [p.get(key) for p in peer_list]
        finite = [v for v in values if not math.isnan(v)]
        comparisons.append(
            MetricComparison(
                spec=spec,
                subject=subject.get(key),
                peer_median=_median(finite),
                peer_min=min(finite) if finite else _NAN,
                peer_max=max(finite) if finite else _NAN,
                n_peers=len(finite),
            )
        )

    if not peer_list:
        notices.append(
            "No peers were supplied, so every row shows the subject's own figure and no comparison."
        )
    else:
        missing_everywhere = [c.spec.label for c in comparisons if not c.n_peers]
        if missing_everywhere:
            notices.append(
                "No peer reported "
                + ", ".join(missing_everywhere)
                + ", so those rows are not compared."
            )
        thin = [c for c in comparisons if c.available and c.thin]
        if thin:
            notices.append(
                f"{len(thin)} of {sum(1 for c in comparisons if c.available)} compared "
                f"metric(s) rest on fewer than {MIN_PEERS_FOR_MEDIAN} peers, so their "
                "medians are indicative at best."
            )

    return PeerComparison(
        subject=subject,
        comparisons=tuple(comparisons),
        peers=tuple(peer_list),
        notices=tuple(notices),
    )


def peer_table(comparison: PeerComparison) -> tuple[tuple[str, ...], ...]:
    """The comparison as rows of strings, ready for a table renderer.

    Kept here rather than in the presentation layer so the page and the PDF
    cannot format the same numbers two different ways, and so a missing value
    is the same em-dash in both.
    """
    header = ("Metric", "Company", "Peer median", "Peer low", "Peer high", "Peers", "Reads as")
    rows: list[tuple[str, ...]] = [header]
    for c in comparison.comparisons:
        rows.append(
            (
                c.spec.label,
                format_metric(c.subject, c.spec.unit),
                format_metric(c.peer_median, c.spec.unit),
                format_metric(c.peer_min, c.spec.unit),
                format_metric(c.peer_max, c.spec.unit),
                str(c.n_peers) if c.n_peers else "—",
                c.verdict,
            )
        )
    return tuple(rows)


def format_metric(value: float, unit: str) -> str:
    """One metric value as a reader sees it. Missing is an em-dash, never a zero.

    Public because the peer bar chart labels its bars with the same two numbers
    ``peer_table`` puts in its cells, and a chart whose tooltip says ``14.2``
    about the value a table calls ``14.2%`` is a bug the reader has to catch.
    The unit lives on :class:`MetricSpec`, so the format follows the metric
    rather than the call site.
    """
    if math.isnan(value):
        return "—"
    if unit == "%":
        return f"{value:.1%}"
    if unit == "x":
        return f"{value:.1f}x"
    return f"{value:,.2f}"


def rank_in_peers(
    subject: PeerCompany,
    peers: Iterable[PeerCompany],
    metric: str,
) -> tuple[int, int]:
    """``(rank, of)`` for one metric, ordered best-first by the metric's own direction.

    Rank 1 is the most attractive company on that metric -- the cheapest on
    P/E, the most profitable on ROE. A company whose value is missing ranks
    nowhere and returns ``(0, 0)``, because "last" would be a false statement
    about a company that simply did not report the line.
    """
    if metric not in METRICS:
        return (0, 0)
    spec = METRICS[metric]
    mine = subject.get(metric)
    if math.isnan(mine):
        return (0, 0)

    others = [p.get(metric) for p in peers if p.symbol != subject.symbol]
    finite = [v for v in others if not math.isnan(v)]
    if not finite:
        return (0, 0)

    # Count how many peers are strictly more attractive than the subject.
    better = sum(1 for v in finite if (v > mine if spec.higher_is_better else v < mine))
    return (better + 1, len(finite) + 1)


# ---------------------------------------------------------------------------
# peer discovery
# ---------------------------------------------------------------------------
def _cluster(features: np.ndarray, k: int) -> np.ndarray | None:
    """Agglomerative clustering on standardised features, or None if unavailable.

    Imported lazily: the analytics layer must stay importable when sklearn is
    absent, and peer discovery then degrades to sector labels rather than
    taking the whole page down.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover - exercised only without sklearn
        return None

    scaled = StandardScaler().fit_transform(features)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(scaled)


def _silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Separation quality of a labelling, or NaN when it cannot be measured."""
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:  # pragma: no cover
        return _NAN
    distinct = set(labels.tolist())
    if len(distinct) < 2 or len(distinct) >= len(labels):
        return _NAN
    try:
        return float(silhouette_score(features, labels))
    except ValueError:  # pragma: no cover - degenerate groupings
        return _NAN


def cluster_peers(
    subject: PeerCompany,
    universe: Iterable[PeerCompany],
    features: Sequence[str] = DEFAULT_CLUSTER_FEATURES,
    max_peers: int = 12,
    k: int | None = None,
) -> PeerCluster:
    """Find the companies most comparable to ``subject``.

    Prefers the exchange's own industry label, because it is a name a reader
    can check. Falls back to clustering when that label yields too few
    companies to compare against, and only then -- a cluster is a defensible
    peer set but an unnameable one, so the caption has to say so.
    """
    others = [p for p in universe if p.symbol != subject.symbol]
    if not others:
        return PeerCluster(subject_symbol=subject.symbol, method="universe")

    if subject.industry:
        same = [p for p in others if p.industry == subject.industry]
        if len(same) >= MIN_PEERS_FOR_MEDIAN:
            return PeerCluster(
                subject_symbol=subject.symbol,
                members=tuple(same[:max_peers]),
                method="industry",
                notices=("Peers were taken from the provider's industry label.",)
                if len(same) > max_peers
                else (),
                industry_label=subject.industry,
            )

    if subject.sector:
        same = [p for p in others if p.sector == subject.sector]
        if len(same) >= MIN_PEERS_FOR_MEDIAN:
            return PeerCluster(
                subject_symbol=subject.symbol,
                members=tuple(same[:max_peers]),
                method="sector",
                notices=(
                    f"The industry label gave too few peers, so the coarser "
                    f"'{subject.sector}' sector was used. "
                    "Those companies are not necessarily comparable.",
                )
                if subject.industry
                else (),
                industry_label=subject.sector,
            )

    # Clustering needs complete rows: a company missing a feature cannot be
    # placed in a space defined by that feature, and imputing it would place it
    # somewhere arbitrary while claiming otherwise. That applies to the subject
    # too, and it is checked first, because a subject with no row in the matrix
    # has no cluster to be a member of -- the grouping simply cannot be made.
    blind = [f for f in features if math.isnan(subject.get(f))]
    if blind:
        return PeerCluster(
            subject_symbol=subject.symbol,
            method="universe",
            notices=(
                "Companies cannot be grouped into peer clusters because the subject "
                "does not report " + ", ".join(blind) + ". Peer comparison is unavailable.",
            ),
        )

    usable = [p for p in [subject, *others] if all(not math.isnan(p.get(f)) for f in features)]
    if len(usable) < 2 * MIN_PEERS_FOR_MEDIAN:
        return PeerCluster(
            subject_symbol=subject.symbol,
            method="universe",
            notices=(
                f"Only {len(usable)} companies reported every clustering input "
                f"({'/'.join(features)}), too few to group. No peer comparison is shown.",
            ),
        )

    matrix = np.asarray([[c.get(f) for f in features] for c in usable], dtype=float)
    n = len(usable)
    groups = k or max(2, min(8, n // 5))
    groups = min(groups, n - 1)
    if groups < 2:
        return PeerCluster(subject_symbol=subject.symbol, method="universe")

    labels = _cluster(matrix, groups)
    if labels is None:
        return PeerCluster(
            subject_symbol=subject.symbol,
            method="universe",
            notices=(
                "Peer clustering needs scikit-learn, which is not installed. "
                "Only the subject's own metrics are shown.",
            ),
        )

    # Find the subject's own row to learn which cluster it landed in. It is
    # guaranteed present: `usable` was built from a list starting with it.
    index_of_subject = next(i for i, c in enumerate(usable) if c.symbol == subject.symbol)
    mine = labels[index_of_subject]
    members = [
        (i, c) for i, c in enumerate(usable) if labels[i] == mine and c.symbol != subject.symbol
    ]

    # Rank within the cluster by similarity, so a capped peer list keeps the
    # nearest neighbours rather than whichever happened to be first. Ranked by
    # position rather than by `list.index`, which would collapse two companies
    # that happen to carry identical metrics onto one another.
    if len(members) > max_peers:
        centre = matrix[index_of_subject]
        members.sort(key=lambda pair: float(np.linalg.norm(matrix[pair[0]] - centre)))
        members = members[:max_peers]

    return PeerCluster(
        subject_symbol=subject.symbol,
        members=tuple(c for _, c in members),
        method="cluster",
        k=int(len(set(labels.tolist()))),
        silhouette=_silhouette(matrix, labels),
        notices=(
            "Clusters are formed from financial characteristics only; they carry no "
            "industry name, so membership should be eyeballed before it is trusted.",
        ),
        industry_label=subject.industry or subject.sector,
    )
