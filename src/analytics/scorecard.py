"""The verdict: a weighted scorecard where every input and weight is visible.

The user of this tool asked for an opinion -- should I own this company? -- and
the temptation is to answer with one number. This module exists to make that
answer **checkable** instead.

Three rules, each of which is the opposite of what a scoring widget usually
does:

**1. Nothing is scored on partial inputs.** The standard implementation gives a
missing metric a zero: no P/E, no point. That is not caution, it is a silent
lie, because a company that does not report a line is now rated as though it
had reported the worst possible value. Here an unavailable factor is *excluded*
and the remaining weights are renormalised, so the composite answers the
question it can actually answer. If too little of the intended weight survives,
the scorecard refuses to give a band at all rather than averaging a hunch.

**2. The composite never hides a disagreement.** Buying a great business at an
awful price and buying a mediocre business at a wonderful price can produce the
same average and are not the same decision. So the dispersion across factors is
computed and a wide spread is reported as *contested*, with the two camps named,
rather than smoothed away.

**3. No forecast enters the weights.** The walk-forward return model elsewhere in
this package reports an out-of-sample rank IC that is, honestly, close to zero
and frequently indistinguishable from noise. Feeding that number into a verdict
would dress noise as evidence and make the verdict look better supported than it
is. The model is displayed beside the scorecard as a separate finding, at weight
zero, and it stays there unless it ever earns its way in.

Bands are deliberately coarse. Five bands on a composite that rests on a DCF
whose terminal value is 70% of the answer do not support precision, and a
"62/100, buy" reads as far more authoritative than the inputs justify.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

__all__ = [
    "FactorSpec",
    "FACTORS",
    "DEFAULT_WEIGHTS",
    "WEIGHT_PRESETS",
    "MIN_COVERAGE",
    "BANDS",
    "Factor",
    "Scorecard",
    "build_scorecard",
    "score_from_anchors",
]

_NAN = float("nan")

#: Below this share of the intended weight, no verdict band is issued. Weight is
#: the honest proxy for evidence: a composite built from 40% of the intended
#: inputs is a different object from one built from 95%, and giving both a label
#: would make the label meaningless.
MIN_COVERAGE = 0.60

#: Upper bound of each band, ascending. Chosen coarse on purpose -- see the
#: module docstring on why five decimal-free bands is already generous.
BANDS: tuple[tuple[float, str], ...] = (
    (0.25, "Avoid"),
    (0.40, "Reduce"),
    (0.60, "Hold"),
    (0.75, "Buy"),
    (1.01, "Strong buy"),
)

#: Factors whose scores disagree by more than this are called out. Set well
#: above the sampling noise of an anchor-interpolated score and well below the
#: spread a genuinely contradictory case produces.
CONTESTED_SPREAD = 0.30


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Coerce to a finite float, or NaN. None and non-numbers are missing."""
    if value is None:
        return _NAN
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _NAN
    return number if math.isfinite(number) else _NAN


def score_from_anchors(value: object, anchors: Sequence[tuple[float, float]]) -> float:
    """Map a raw metric onto 0..1 by piecewise-linear interpolation between anchors.

    ``anchors`` are ``(raw_value, score)`` pairs sorted by raw value ascending.
    Below the first anchor the score clamps to its score, above the last to
    that one -- an extrapolated score would let a single outlier company push
    past the ends of the scale, which is exactly where a scoring widget becomes
    nonsense.

    Scores need not ascend with values: leverage anchors descend, so higher debt
    scores lower without any special case here.

    A figure that is not a number returns NaN, never 0.0. That distinction is
    load-bearing: NaN means "excluded from the composite", while 0.0 means "as
    bad as this scale goes", and conflating them is the bug this module's first
    rule exists to prevent.
    """
    number = _num(value)
    if math.isnan(number) or not anchors:
        return _NAN

    if number <= anchors[0][0]:
        return float(anchors[0][1])
    if number >= anchors[-1][0]:
        return float(anchors[-1][1])

    for (low_x, low_y), (high_x, high_y) in pairwise(anchors):
        if low_x <= number <= high_x:
            if high_x == low_x:  # pragma: no cover - a degenerate anchor pair
                return float(high_y)
            share = (number - low_x) / (high_x - low_x)
            return float(low_y + share * (high_y - low_y))
    return _NAN  # pragma: no cover - anchors are checked to span the value


# ---------------------------------------------------------------------------
# the factor registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FactorSpec:
    """One input to the verdict, and the scale it is judged on."""

    key: str
    label: str
    group: str
    anchors: tuple[tuple[float, float], ...]
    unit: str = ""
    note: str = ""

    def score(self, raw: object) -> float:
        return score_from_anchors(raw, self.anchors)

    def display(self, raw: object) -> str:
        """The raw value as a reader should see it. Missing is an em-dash."""
        number = _num(raw)
        if math.isnan(number):
            return "—"
        if self.unit == "%":
            return f"{number:.1%}"
        if self.unit == "x":
            return f"{number:.1f}x"
        if self.unit == "/9":
            return f"{number:.0f} / 9"
        return f"{number:,.2f}"


#: The scorecard's inputs. Each anchor set is a judgement, and each is written
#: out in full so a reader can disagree with the *scale* rather than only with
#: the answer -- which is the point of showing the weights at all.
FACTORS: dict[str, FactorSpec] = {
    # -- valuation ---------------------------------------------------------
    "dcf_upside": FactorSpec(
        "dcf_upside",
        "DCF upside to fair value",
        "Valuation",
        unit="%",
        anchors=((-0.50, 0.00), (-0.20, 0.25), (0.00, 0.50), (0.25, 0.75), (0.60, 1.00)),
        note="Price against the discounted cash flow value. The heaviest input, "
        "and the one most sensitive to a terminal-growth assumption.",
    ),
    "peer_value": FactorSpec(
        "peer_value",
        "Cheapness against peers",
        "Valuation",
        unit="%",
        anchors=((0.00, 0.00), (0.25, 0.25), (0.50, 0.50), (0.75, 0.75), (1.00, 1.00)),
        note="Share of the comparable lower-is-better metrics on which the company "
        "beats its peer median.",
    ),
    "fcf_yield": FactorSpec(
        "fcf_yield",
        "Free cash flow yield",
        "Valuation",
        unit="%",
        anchors=((0.00, 0.10), (0.03, 0.45), (0.06, 0.75), (0.10, 1.00)),
        note="Free cash flow over market capitalisation -- what the business "
        "returns in cash for the price paid.",
    ),
    # -- quality -----------------------------------------------------------
    "piotroski": FactorSpec(
        "piotroski",
        "Piotroski F-Score",
        "Quality",
        unit="/9",
        anchors=((0.0, 0.00), (3.0, 0.25), (5.0, 0.50), (7.0, 0.80), (9.0, 1.00)),
        note="Nine accounting signals of improving fundamentals.",
    ),
    "altman_z": FactorSpec(
        "altman_z",
        "Altman Z-Score",
        "Quality",
        anchors=((1.10, 0.00), (1.81, 0.30), (2.60, 0.60), (3.00, 0.80), (4.50, 1.00)),
        note="Distance from financial distress. The 1.81 and 2.99 marks are "
        "Altman's own distress and safe zones; the middle anchors only "
        "interpolate between them.",
    ),
    "beneish_m": FactorSpec(
        "beneish_m",
        "Beneish M-Score",
        "Quality",
        anchors=((-3.00, 1.00), (-2.22, 0.70), (-1.78, 0.35), (-1.00, 0.10), (0.00, 0.00)),
        note="Likelihood of earnings manipulation. ABOVE -1.78 is Beneish's own "
        "flag, so the scale runs the other way.",
    ),
    "accruals": FactorSpec(
        "accruals",
        "Accruals ratio",
        "Quality",
        anchors=((-0.10, 1.00), (0.00, 0.70), (0.05, 0.45), (0.10, 0.15), (0.25, 0.00)),
        note="Profit not backed by cash. High accruals precede disappointment.",
    ),
    # -- growth and profitability -----------------------------------------
    "revenue_growth": FactorSpec(
        "revenue_growth",
        "Revenue growth",
        "Growth",
        unit="%",
        anchors=((-0.05, 0.00), (0.00, 0.35), (0.08, 0.60), (0.15, 0.85), (0.30, 1.00)),
    ),
    "roe": FactorSpec(
        "roe",
        "Return on equity",
        "Growth",
        unit="%",
        anchors=((0.00, 0.00), (0.08, 0.30), (0.15, 0.60), (0.22, 0.85), (0.35, 1.00)),
        note="Read beside leverage: debt raises ROE without improving the business.",
    ),
    "net_margin": FactorSpec(
        "net_margin",
        "Net margin",
        "Growth",
        unit="%",
        anchors=((-0.05, 0.00), (0.03, 0.35), (0.08, 0.60), (0.15, 0.85), (0.25, 1.00)),
    ),
    # -- financial strength ------------------------------------------------
    "debt_to_equity": FactorSpec(
        "debt_to_equity",
        "Debt / equity",
        "Financial strength",
        unit="x",
        anchors=((0.00, 1.00), (0.30, 0.85), (0.80, 0.60), (1.50, 0.30), (2.50, 0.00)),
    ),
    "interest_coverage": FactorSpec(
        "interest_coverage",
        "Interest coverage",
        "Financial strength",
        unit="x",
        anchors=((1.0, 0.00), (2.0, 0.25), (4.0, 0.55), (8.0, 0.85), (15.0, 1.00)),
        note="Times interest is earned. Below 1.0 the operations do not pay the interest.",
    ),
    # -- risk ---------------------------------------------------------------
    "volatility": FactorSpec(
        "volatility",
        "Annualised volatility",
        "Risk",
        unit="%",
        anchors=((0.12, 1.00), (0.20, 0.80), (0.30, 0.55), (0.45, 0.25), (0.70, 0.00)),
        note="A property of the price, not of the business -- a quiet price is not "
        "the same as a safe company.",
    ),
    "max_drawdown": FactorSpec(
        "max_drawdown",
        "Maximum drawdown",
        "Risk",
        unit="%",
        anchors=((-0.65, 0.00), (-0.45, 0.25), (-0.30, 0.50), (-0.15, 0.80), (0.00, 1.00)),
        note="The worst peak-to-trough fall already experienced. Negative.",
    ),
    # -- sentiment ----------------------------------------------------------
    "sentiment": FactorSpec(
        "sentiment",
        "News tone",
        "Sentiment",
        anchors=((-1.00, 0.00), (-0.40, 0.20), (0.00, 0.50), (0.40, 0.80), (1.00, 1.00)),
        note="A lexicon score over recent headlines. Short-lived by nature, which "
        "is why it carries a small weight.",
    ),
}

#: The default weighting. Valuation and quality dominate; news tone is a small
#: slice because it decays within days and a research note outlives it.
DEFAULT_WEIGHTS: dict[str, float] = {
    "dcf_upside": 0.20,
    "peer_value": 0.10,
    "fcf_yield": 0.06,
    "piotroski": 0.08,
    "altman_z": 0.07,
    "beneish_m": 0.05,
    "accruals": 0.03,
    "revenue_growth": 0.10,
    "roe": 0.08,
    "net_margin": 0.04,
    "debt_to_equity": 0.06,
    "interest_coverage": 0.04,
    "volatility": 0.04,
    "max_drawdown": 0.04,
    "sentiment": 0.01,
}

#: Alternative weightings, so a reader can watch the verdict move. Each is a
#: defensible stance rather than a slider preset, and each is named for the
#: stance it takes.
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "Balanced (default)": dict(DEFAULT_WEIGHTS),
    "Value": {
        **dict.fromkeys(DEFAULT_WEIGHTS, 0.0),
        "dcf_upside": 0.35,
        "peer_value": 0.20,
        "fcf_yield": 0.15,
        "revenue_growth": 0.10,
        "sentiment": 0.0,
        "volatility": 0.10,
        "max_drawdown": 0.10,
    },
    "Quality": {
        **dict.fromkeys(DEFAULT_WEIGHTS, 0.0),
        "piotroski": 0.20,
        "altman_z": 0.15,
        "beneish_m": 0.12,
        "accruals": 0.08,
        "roe": 0.15,
        "net_margin": 0.10,
        "revenue_growth": 0.10,
        "dcf_upside": 0.10,
    },
    "Defensive": {
        **dict.fromkeys(DEFAULT_WEIGHTS, 0.0),
        "debt_to_equity": 0.18,
        "interest_coverage": 0.14,
        "altman_z": 0.16,
        "volatility": 0.16,
        "max_drawdown": 0.16,
        "fcf_yield": 0.12,
        "net_margin": 0.08,
    },
}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Factor:
    """One scored line of the scorecard."""

    spec: FactorSpec
    raw: object = None
    score: float = _NAN
    weight: float = 0.0
    effective_weight: float = 0.0  # after renormalising over available factors

    @property
    def available(self) -> bool:
        return not math.isnan(self.score)

    @property
    def contribution(self) -> float:
        """Points this factor adds to the composite, out of 1.

        NaN when the factor is unavailable, so a caller summing contributions
        cannot accidentally count an excluded input as zero.
        """
        if not self.available:
            return _NAN
        return self.score * self.effective_weight

    @property
    def influence(self) -> float:
        """How far this factor pulled the composite from neutral, in either direction.

        Distinct from :attr:`contribution` on purpose, and the distinction is
        not cosmetic. A maximally adverse input contributes *zero* points but is
        frequently the reason the verdict reads as it does; ranking drivers by
        contribution alone would build a narrative that mentions only the
        supportive factors and quietly omits the objection.
        """
        if not self.available:
            return _NAN
        return self.effective_weight * abs(self.score - 0.5)

    @property
    def reading(self) -> str:
        """How this factor is pulling, in words, with its own value shown.

        Strength is measured as distance from the neutral 0.5 in *both*
        directions, so a 0.02 reads as strongly adverse rather than weakly --
        the failure mode of a one-sided adverb ladder, which silences exactly
        the factors a reader most needs to see.
        """
        if not self.available:
            return "Not available."
        shown = self.spec.display(self.raw)
        if self.score > 0.55:
            strength = (
                "strongly"
                if self.score >= 0.75
                else ("moderately" if self.score >= 0.65 else "mildly")
            )
            return f"{strength} supportive at {shown}."
        if self.score < 0.45:
            strength = (
                "strongly"
                if self.score <= 0.25
                else ("moderately" if self.score <= 0.35 else "mildly")
            )
            return f"{strength} adverse at {shown}."
        return f"Neutral at {shown}."


@dataclass(frozen=True, slots=True)
class Scorecard:
    """The composite, its band, and everything needed to disagree with it."""

    factors: tuple[Factor, ...] = ()
    composite: float = _NAN
    coverage: float = 0.0
    verdict: str = "Insufficient inputs"
    notices: tuple[str, ...] = ()
    weights_used: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a band was actually issued."""
        return not math.isnan(self.composite) and self.coverage >= MIN_COVERAGE

    @property
    def scored(self) -> tuple[Factor, ...]:
        return tuple(f for f in self.factors if f.available)

    @property
    def excluded(self) -> tuple[Factor, ...]:
        return tuple(f for f in self.factors if not f.available)

    @property
    def by_group(self) -> dict[str, tuple[Factor, ...]]:
        """Factors grouped for display, in the registry's order."""
        groups: dict[str, list[Factor]] = {}
        for factor in self.factors:
            groups.setdefault(factor.spec.group, []).append(factor)
        return {name: tuple(items) for name, items in groups.items()}

    @property
    def ranked(self) -> tuple[Factor, ...]:
        """Available factors by how hard they pulled the answer, largest first.

        Ranked by :attr:`Factor.influence` rather than by contribution: a
        maximally adverse input that carries 20% of the weight moves the verdict
        far more than a supportive one carrying 1%, and it is the adverse input
        a reader most needs named.
        """
        return tuple(sorted(self.scored, key=lambda f: f.influence, reverse=True))

    @property
    def spread(self) -> float:
        """Weighted dispersion of the factor scores, 0 when they agree."""
        scored = self.scored
        if len(scored) < 2:
            return 0.0
        total = sum(f.effective_weight for f in scored)
        if total <= 0:
            return 0.0
        variance = sum(f.effective_weight * (f.score - self.composite) ** 2 for f in scored) / total
        return math.sqrt(max(variance, 0.0))

    @property
    def contested(self) -> bool:
        """True when the factors disagree too much for one label to be honest."""
        return self.spread > CONTESTED_SPREAD

    @property
    def strongest(self) -> Factor | None:
        scored = self.scored
        return max(scored, key=lambda f: f.score) if scored else None

    @property
    def weakest(self) -> Factor | None:
        scored = self.scored
        return min(scored, key=lambda f: f.score) if scored else None

    @property
    def explanation(self) -> str:
        """A paragraph a reader could argue with, naming the drivers by influence."""
        if not self.scored:
            return "Nothing could be scored, so no view is offered."

        top = self.ranked[:3]
        drivers = ", ".join(f"{f.spec.label} ({f.spec.display(f.raw)})" for f in top)
        sentence = f"Most influential inputs: {drivers}."

        if not self.ok:
            return (
                f"Only {self.coverage:.0%} of the intended weighting could be computed, "
                f"below the {MIN_COVERAGE:.0%} needed for a band, so no verdict is given. "
                f"{sentence}"
            )

        caution = ""
        if self.contested:
            strongest, weakest = self.strongest, self.weakest
            if strongest is not None and weakest is not None:
                caution = (
                    f" The inputs disagree: {strongest.spec.label} is "
                    f"{strongest.spec.display(strongest.raw)} while "
                    f"{weakest.spec.label} is {weakest.spec.display(weakest.raw)}. "
                    f"Read the {self.verdict.lower()} label with that spread in mind."
                )
        return f"{sentence}{caution}"


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _band(composite: float) -> str:
    for ceiling, name in BANDS:
        if composite < ceiling:
            return name
    return BANDS[-1][1]  # pragma: no cover - the last ceiling exceeds 1.0


def build_scorecard(
    inputs: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
) -> Scorecard:
    """Turn raw metrics into a weighted composite and a band.

    ``inputs`` maps :data:`FACTORS` keys to raw values -- a rate as a decimal
    (``0.18`` for 18%), a score as its own number, ``None`` or NaN for anything
    the statements did not yield. Unknown keys are ignored and reported.

    Weights are renormalised over the factors that could be scored, so the
    composite is always on a 0..1 scale regardless of how many inputs survived.
    ``coverage`` records how much of the *requested* weight survived, and a
    scorecard below :data:`MIN_COVERAGE` returns no band.
    """
    requested = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    notices: list[str] = []

    unknown = sorted(set(inputs) - set(FACTORS))
    if unknown:
        notices.append("Ignored unrecognised input(s): " + ", ".join(unknown) + ".")
    unweighted = sorted(set(FACTORS) - set(requested))
    if unweighted:
        notices.append(
            "No weight was given for " + ", ".join(unweighted) + ", so those inputs "
            "do not affect the verdict."
        )
    negative = sorted(k for k, w in requested.items() if _num(w) < 0)
    if negative:
        notices.append(
            "Negative weights are not meaningful here and were clamped to zero: "
            + ", ".join(negative)
            + "."
        )

    factors: list[Factor] = []
    for key, spec in FACTORS.items():
        weight = max(_num(requested.get(key, 0.0)), 0.0)
        if math.isnan(weight):
            weight = 0.0
        raw = inputs.get(key)
        factors.append(Factor(spec=spec, raw=raw, score=spec.score(raw), weight=weight))

    available_weight = sum(f.weight for f in factors if f.available)
    requested_weight = sum(f.weight for f in factors)
    coverage = (available_weight / requested_weight) if requested_weight > 0 else 0.0

    if requested_weight <= 0:
        return Scorecard(
            factors=tuple(factors),
            coverage=0.0,
            verdict="Insufficient inputs",
            notices=(
                *notices,
                "No factor carries any weight, so there is nothing to combine.",
            ),
            weights_used=requested,
        )

    if available_weight <= 0:
        return Scorecard(
            factors=tuple(factors),
            coverage=0.0,
            verdict="Insufficient inputs",
            notices=(*notices, "No input could be scored, so no verdict is offered."),
            weights_used=requested,
        )

    # Renormalise so the composite is on a 0..1 scale whatever survived.
    factors = [
        Factor(
            spec=f.spec,
            raw=f.raw,
            score=f.score,
            weight=f.weight,
            effective_weight=(f.weight / available_weight) if f.available else 0.0,
        )
        for f in factors
    ]
    composite = sum(f.score * f.effective_weight for f in factors if f.available)

    card = Scorecard(
        factors=tuple(factors),
        composite=composite,
        coverage=coverage,
        verdict=_band(composite) if coverage >= MIN_COVERAGE else "Insufficient inputs",
        notices=tuple(notices),
        weights_used=requested,
    )

    extra: list[str] = []
    if coverage < MIN_COVERAGE:
        extra.append(
            f"Only {coverage:.0%} of the intended weighting could be computed and "
            f"{MIN_COVERAGE:.0%} is the floor for a verdict, so the inputs are shown "
            "without a band."
        )
    if card.contested:
        extra.append(
            f"The factors disagree (dispersion {card.spread:.2f}), which one blended "
            "label cannot represent."
        )
    return Scorecard(
        factors=card.factors,
        composite=card.composite,
        coverage=card.coverage,
        verdict=card.verdict,
        notices=card.notices + tuple(extra),
        weights_used=card.weights_used,
    )
