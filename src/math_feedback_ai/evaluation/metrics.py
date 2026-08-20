"""Small deterministic metrics; no model judge is required.

Every rate includes its numerator and denominator so that tiny development
sets cannot accidentally look more authoritative than they are.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class MetricResult:
    """A scalar metric and the raw counts from which it was computed."""

    value: float
    numerator: float
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("metric denominator must be positive")
        if not math.isfinite(self.value) or not math.isfinite(self.numerator):
            raise ValueError("metric values must be finite")


@dataclass(frozen=True, slots=True)
class SystemObservation:
    """Per-example provider usage collected by the orchestration layer."""

    model_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("model_calls", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class SystemSummary:
    """Aggregate usage; optional fields remain absent when providers omit them."""

    examples: int
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    mean_latency_ms: float | None
    p95_latency_ms: float | None


def _paired[LabelT: Hashable](gold: Sequence[LabelT], predicted: Sequence[LabelT]) -> None:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted sequences must have the same length")
    if not gold:
        raise ValueError("at least one example is required")


def accuracy[LabelT: Hashable](gold: Sequence[LabelT], predicted: Sequence[LabelT]) -> MetricResult:
    """Return ordinary exact-match accuracy."""

    _paired(gold, predicted)
    correct = sum(expected == actual for expected, actual in zip(gold, predicted, strict=True))
    return MetricResult(value=correct / len(gold), numerator=correct, denominator=len(gold))


def macro_f1[LabelT: Hashable](
    gold: Sequence[LabelT],
    predicted: Sequence[LabelT],
    *,
    labels: Iterable[LabelT] | None = None,
) -> MetricResult:
    """Return unweighted F1 across explicit or observed labels.

    An explicit label with no gold or predicted examples receives F1 zero. This
    makes a declared taxonomy visible even on a small benchmark.
    """

    _paired(gold, predicted)
    evaluated_labels = tuple(dict.fromkeys(labels or (*gold, *predicted)))
    if not evaluated_labels:
        raise ValueError("at least one label is required")

    scores: list[float] = []
    for label in evaluated_labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    total = sum(scores)
    return MetricResult(
        value=total / len(scores),
        numerator=total,
        denominator=len(scores),
    )


def false_rejection_rate[LabelT: Hashable](
    gold: Sequence[LabelT],
    predicted: Sequence[LabelT],
    *,
    correct_labels: frozenset[LabelT],
    rejection_labels: frozenset[LabelT],
) -> MetricResult:
    """Measure correct attempts that were positively labelled as erroneous.

    Abstention/indeterminate predictions are intentionally not counted as false
    rejection; they should be reported separately by status metrics.
    """

    _paired(gold, predicted)
    eligible = [index for index, label in enumerate(gold) if label in correct_labels]
    if not eligible:
        raise ValueError("false rejection rate needs at least one correct example")
    rejected = sum(predicted[index] in rejection_labels for index in eligible)
    return MetricResult(
        value=rejected / len(eligible), numerator=rejected, denominator=len(eligible)
    )


def localization_accuracy(
    gold_positions: Sequence[int | None],
    predicted_positions: Sequence[int | None],
    *,
    tolerance: int = 0,
) -> MetricResult:
    """Score localizable errors within an ordinal tolerance.

    Correct/incomplete cases with no gold error are excluded. Their null
    semantics belong in schema and diagnosis-status tests, not this metric.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    _paired(gold_positions, predicted_positions)
    eligible = [index for index, value in enumerate(gold_positions) if value is not None]
    if not eligible:
        raise ValueError("localization accuracy needs at least one localized gold issue")
    matched = 0
    for index in eligible:
        expected = gold_positions[index]
        actual = predicted_positions[index]
        if expected is not None and actual is not None and abs(expected - actual) <= tolerance:
            matched += 1
    return MetricResult(value=matched / len(eligible), numerator=matched, denominator=len(eligible))


def rate(flags: Sequence[bool]) -> MetricResult:
    """Return the fraction of examples for which a binary event occurred."""

    if not flags:
        raise ValueError("at least one example is required")
    occurred = sum(flags)
    return MetricResult(value=occurred / len(flags), numerator=occurred, denominator=len(flags))


def summarize_system_metrics(observations: Sequence[SystemObservation]) -> SystemSummary:
    """Aggregate calls, optional tokens, and nearest-rank p95 latency."""

    if not observations:
        raise ValueError("at least one system observation is required")

    input_tokens = [item.input_tokens for item in observations if item.input_tokens is not None]
    output_tokens = [item.output_tokens for item in observations if item.output_tokens is not None]
    latencies = sorted(item.latency_ms for item in observations if item.latency_ms is not None)
    p95: float | None = None
    mean: float | None = None
    if latencies:
        p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
        mean = fmean(latencies)

    return SystemSummary(
        examples=len(observations),
        model_calls=sum(item.model_calls for item in observations),
        input_tokens=sum(input_tokens) if input_tokens else None,
        output_tokens=sum(output_tokens) if output_tokens else None,
        mean_latency_ms=mean,
        p95_latency_ms=p95,
    )


def label_counts[LabelT: Hashable](labels: Sequence[LabelT]) -> dict[LabelT, int]:
    """Expose deterministic label counts for benchmark audit reports."""

    return dict(Counter(labels))
