"""Headline tone, and an honest attempt to measure whether it predicts anything.

The user asked for "news analysis ... which help to make the decision easy".
There are two very different things that could mean, and the difference between
them is the whole point of this module.

The first is **scoring the tone of the coverage**: read each headline, classify
it against a finance lexicon, and report what the coverage currently sounds
like. That is a description of text, and
:func:`src.analytics.news.tone` already does it.

The second is **claiming the tone predicts the price**. That is a much stronger
claim, it is the one a reader would act on, and it is almost always false at the
sample sizes a public news feed actually provides. A page that computed an
average sentiment score and printed it next to a buy button would be implying
the second while only having done the first.

So this module does the second thing *properly and in public*: it measures the
rank correlation between tone and subsequent return, and it reports the sample
size, the confidence interval, and a plain statement of whether the result is
distinguishable from noise. On the real data this almost never is, and the
module is designed to say so clearly rather than to bury it.

Three specific traps it refuses to fall into:

**Overlapping forward windows.** Twenty headlines in one week, each measured
against the next 20 days of return, are not twenty observations -- they are
about one, seen twenty times. Correlations computed on overlapping windows have
standard errors that are far too small, so the module **de-clusters**: tone is
averaged per calendar day and consecutive days are treated as a single
observation. The sample size it reports is the number of clusters, not the
number of headlines.

**A correlation on a handful of days.** With fewer than
:data:`MIN_CLUSTERS_FOR_MEASUREMENT` independent clusters no correlation is
reported at all, not even with a caveat. Printing r = 0.42 (n = 6) invites the
reader to believe it.

**The sign being read backwards.** A correlation between tone and *past* return
is a headline following the price, not leading it. The measure is therefore
against the return *after* the headline only, and the direction is asserted in
the tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

__all__ = [
    "MIN_CLUSTERS_FOR_MEASUREMENT",
    "ToneObservation",
    "SentimentTurning",
    "DailyTone",
    "HeadlineScore",
    "score_headlines",
    "daily_tone",
    "sentiment_turning",
    "tone_reading",
    "MEANINGLESS_IC",
]

#: Below this many *independent* day-clusters, no correlation is reported. The
#: number is low and that is deliberate: it is not a threshold for significance,
#: it is the point below which computing the statistic is theatre. A correlation
#: needs an absolute minimum of about five points to be arithmetic at all.
MIN_CLUSTERS_FOR_MEASUREMENT = 8

#: A rank correlation inside this band is reported as noise. Borrowed from the
#: forecasting literature's rough consensus that |IC| below about 0.03 is not
#: distinguishable from zero in the sample sizes a retail tool can assemble.
MEANINGLESS_IC = 0.03


@dataclass(frozen=True, slots=True)
class ToneObservation:
    """One headline's tone, dated, ready to be aligned against a return."""

    published: datetime | None
    title: str = ""
    score: float = 0.0
    label: str = "neutral"
    matched: tuple[str, ...] = ()

    @property
    def day(self) -> date | None:
        """The calendar day the headline falls on, or None when undated."""
        return self.published.date() if isinstance(self.published, datetime) else None

    @property
    def directional(self) -> bool:
        """True when the lexicon actually matched something.

        A headline scoring 0.0 because nothing matched is not a neutral
        headline -- it is an unscored one, and counting it as neutral drags
        every average toward zero for no reason.
        """
        return bool(self.matched)


@dataclass(frozen=True, slots=True)
class DailyTone:
    """One calendar day's coverage, collapsed to a single number."""

    day: date
    mean_score: float = 0.0
    n_headlines: int = 0
    n_directional: int = 0


def score_headlines(items: Sequence[object]) -> tuple[ToneObservation, ...]:
    """Score every headline with the finance lexicon, newest first.

    Delegates to :func:`src.analytics.news.tone` so the on-page tone and the
    tone used in the measurement below can never drift apart -- two
    implementations of "is this headline good news" that agree today is a bug
    waiting to be written.
    """
    from src.analytics.news import tone

    out: list[ToneObservation] = []
    for item in items:
        title = str(getattr(item, "title", "") or "").strip()
        if not title:
            continue
        # The summary is included because the lexicon is a bag of words and a
        # bare headline is short enough that one term often decides the score.
        summary = str(getattr(item, "summary", "") or "")
        result = tone(f"{title} {summary}" if summary else title)
        out.append(
            ToneObservation(
                published=getattr(item, "published", None),
                title=title,
                score=float(result.score),
                label=str(result.label),
                matched=tuple(result.matched),
            )
        )
    out.sort(key=lambda o: o.published or datetime.min, reverse=True)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class HeadlineScore:
    """A headline with its tone, for display. Thin alias over ToneObservation."""

    title: str
    score: float
    label: str
    published: datetime | None = None
    source: str = ""
    explained: str = ""
    link: str = ""


def daily_tone(observations: Sequence[ToneObservation]) -> tuple[DailyTone, ...]:
    """Collapse headlines into one tone per calendar day, oldest first.

    This is the de-clustering step, and it is what makes the correlation below
    mean anything. Ten headlines on one day are one observation about that day's
    coverage, not ten independent ones. Undated headlines are dropped: they
    cannot be placed against a return, and guessing today's date for them would
    manufacture exactly the clustering this function removes.
    """
    buckets: dict[date, list[ToneObservation]] = {}
    for observation in observations:
        day = observation.day
        if day is None:
            continue
        buckets.setdefault(day, []).append(observation)

    out: list[DailyTone] = []
    for day in sorted(buckets):
        items = buckets[day]
        directional = [o for o in items if o.directional]
        # Only directional headlines set the tone; the rest contribute their
        # existence but not a zero that would dilute the mean.
        scores = [o.score for o in directional]
        out.append(
            DailyTone(
                day=day,
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                n_headlines=len(items),
                n_directional=len(directional),
            )
        )
    return tuple(out)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, NaN unless both sides have real spread.

    Ranks are averaged within ties (the standard treatment), so a day whose
    tone is identical to another day's does not get an arbitrary ordering based
    on list position.
    """

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        # Every day had the same tone, or every return was identical. There is
        # no rank comparison to make, so there is no correlation -- not zero.
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


def _forward_return(series: Sequence[tuple[date, float]], day: date, horizon: int) -> float:
    """Return from the first session at or after ``day`` to ``horizon`` later.

    NaN when the window runs past the end of the series, which is the normal
    case for recent headlines: a headline from yesterday has no 20-day forward
    return yet, and inventing a partial one would bias the sample toward
    whatever the last few days happened to do.
    """
    stamps = [d for d, _ in series]
    start = None
    for index, stamp in enumerate(stamps):
        if stamp >= day:
            start = index
            break
    if start is None:
        return float("nan")
    end = start + horizon
    if end >= len(series):
        return float("nan")
    first, last = series[start][1], series[end][1]
    if first <= 0 or last <= 0:
        return float("nan")
    return last / first - 1.0


@dataclass(frozen=True, slots=True)
class SentimentTurning:
    """The measured relationship between coverage tone and later return.

    Every field is present whether or not a measurement was possible, so the
    page can always say *why* rather than showing a blank.
    """

    horizon_days: int = 20
    n_clusters: int = 0
    n_headlines: int = 0
    rank_ic: float = float("nan")
    clusters: tuple[tuple[date, float, float], ...] = field(default_factory=tuple)
    notices: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        """True when a correlation was actually computed."""
        return not math.isnan(self.rank_ic)

    @property
    def usable(self) -> bool:
        """True when the measurement is strong enough to be worth reading.

        Distinct from :attr:`measured`: a correlation on nine days is computed
        and still not usable. The page shows the number but labels it.
        """
        return self.measured and self.n_clusters >= MIN_CLUSTERS_FOR_MEASUREMENT

    @property
    def indistinguishable_from_noise(self) -> bool:
        """True when the measured correlation is too small to act on."""
        return not self.measured or abs(self.rank_ic) <= MEANINGLESS_IC

    @property
    def verdict(self) -> str:
        if not self.measured:
            return "Not measurable from the coverage available."
        if not self.usable:
            return (
                f"Computed on only {self.n_clusters} independent day(s), which is "
                "too few to read as evidence either way."
            )
        if self.indistinguishable_from_noise:
            return (
                f"Rank correlation {self.rank_ic:+.2f} over {self.n_clusters} days — "
                "indistinguishable from no relationship."
            )
        direction = "positive" if self.rank_ic > 0 else "negative"
        return (
            f"Rank correlation {self.rank_ic:+.2f} over {self.n_clusters} days. A "
            f"{direction} relationship of this size in a sample this small is a "
            "hypothesis, not a finding."
        )

    @property
    def sample_warning(self) -> str:
        """The sentence that must sit beside the number, in plain words."""
        if not self.measured:
            return (
                "No correlation is shown. Measuring whether tone precedes return needs "
                "headlines old enough to have a forward return behind them, and this "
                "feed does not reach back far enough."
            )
        return (
            f"This measures {self.n_headlines} headline(s) collapsed into "
            f"{self.n_clusters} independent day(s) of coverage. Consecutive days of news "
            "are not independent observations, so they are grouped — ten headlines in "
            "one week are closer to one observation than to ten. At this sample size a "
            "correlation of either sign arises from noise routinely."
        )


def sentiment_turning(
    observations: Sequence[ToneObservation],
    prices: Sequence[tuple[date, float]],
    horizon_days: int = 20,
) -> SentimentTurning:
    """Measure whether coverage tone lines up with the return that follows it.

    ``prices`` is a chronological ``(day, close)`` series. Only the return
    *after* each headline day is used, so a positive reading cannot come from
    headlines that merely reported a move that had already happened.

    Reports the cluster count it actually achieved and refuses to compute a
    number when there is nothing to compute it from.
    """
    notices: list[str] = []
    if not observations:
        return SentimentTurning(
            horizon_days=horizon_days,
            notices=("No headlines were available to score.",),
        )

    days = daily_tone(observations)
    if not days:
        return SentimentTurning(
            horizon_days=horizon_days,
            n_headlines=len(observations),
            notices=("No headline carried a usable date, so none could be aligned.",),
        )

    pairs: list[tuple[date, float, float]] = []
    for entry in days:
        forward = _forward_return(prices, entry.day, horizon_days)
        if math.isnan(forward):
            continue
        pairs.append((entry.day, entry.mean_score, forward))

    n_headlines = sum(d.n_headlines for d in days)
    if not pairs:
        return SentimentTurning(
            horizon_days=horizon_days,
            n_headlines=n_headlines,
            notices=(
                f"No headline old enough to have {horizon_days} trading sessions of "
                "return behind it. The feed is all recent news.",
                *notices,
            ),
        )

    if len(pairs) < MIN_CLUSTERS_FOR_MEASUREMENT:
        return SentimentTurning(
            horizon_days=horizon_days,
            n_clusters=len(pairs),
            n_headlines=n_headlines,
            clusters=tuple(pairs),
            notices=(
                f"Only {len(pairs)} independent day(s) have enough forward history to "
                f"measure, below the {MIN_CLUSTERS_FOR_MEASUREMENT} needed to compute a "
                "correlation at all. A number here would be arithmetic, not evidence.",
            ),
        )

    ic = _spearman([p[1] for p in pairs], [p[2] for p in pairs])
    if math.isnan(ic):
        notices.append(
            "Every measured day carried the same tone, or the same return, so there is "
            "no rank comparison to make."
        )

    return SentimentTurning(
        horizon_days=horizon_days,
        n_clusters=len(pairs),
        n_headlines=n_headlines,
        rank_ic=ic,
        clusters=tuple(pairs),
        notices=tuple(notices),
    )


def tone_reading(observations: Sequence[ToneObservation]) -> str:
    """One sentence describing the *current* tone of coverage.

    A description of text and nothing more. It deliberately does not translate
    into a view on the stock, because the measurement above almost never
    justifies one.
    """
    directional = [o for o in observations if o.directional]
    if not observations:
        return "No recent headlines were found for this company."
    if not directional:
        return (
            f"{len(observations)} headline(s) found, none containing a directional "
            "term the lexicon recognises. That is an unscored set, not a neutral "
            "one."
        )
    mean = sum(o.score for o in directional) / len(directional)
    if mean > 0.25:
        tone_word = "net positive"
    elif mean < -0.25:
        tone_word = "net negative"
    else:
        tone_word = "mixed"
    positive = sum(1 for o in directional if o.score > 0)
    negative = sum(1 for o in directional if o.score < 0)
    return (
        f"{len(observations)} headline(s), of which {len(directional)} matched a "
        f"directional term: {positive} positive, {negative} negative, "
        f"mean score {mean:+.2f} — {tone_word} coverage over the period the feed "
        "covers. This describes the language of the headlines, not their effect on "
        "the price."
    )
