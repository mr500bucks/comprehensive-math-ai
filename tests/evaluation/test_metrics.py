from __future__ import annotations

import pytest

from math_feedback_ai.evaluation.metrics import (
    SystemObservation,
    accuracy,
    false_rejection_rate,
    localization_accuracy,
    macro_f1,
    rate,
    summarize_system_metrics,
)


def test_accuracy_reports_counts() -> None:
    result = accuracy(["correct", "wrong", "wrong"], ["correct", "correct", "wrong"])
    assert result.value == pytest.approx(2 / 3)
    assert result.numerator == 2
    assert result.denominator == 3


def test_macro_f1_includes_declared_missing_label() -> None:
    result = macro_f1(
        ["correct", "wrong"],
        ["correct", "wrong"],
        labels=["correct", "wrong", "indeterminate"],
    )
    assert result.value == pytest.approx(2 / 3)
    assert result.denominator == 3


def test_false_rejection_excludes_abstention() -> None:
    result = false_rejection_rate(
        ["fully_correct", "correct_but_inefficient", "incorrect"],
        ["incorrect", "indeterminate", "incorrect"],
        correct_labels=frozenset({"fully_correct", "correct_but_inefficient"}),
        rejection_labels=frozenset({"incorrect"}),
    )
    assert result.value == 0.5
    assert result.denominator == 2


def test_localization_exact_and_adjacent() -> None:
    gold = [1, 3, None]
    predicted = [2, 3, None]
    assert localization_accuracy(gold, predicted).value == 0.5
    assert localization_accuracy(gold, predicted, tolerance=1).value == 1.0


def test_rate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        rate([])


def test_system_metrics_keep_missing_usage_visible() -> None:
    summary = summarize_system_metrics(
        [
            SystemObservation(model_calls=2, input_tokens=100, output_tokens=20, latency_ms=10),
            SystemObservation(model_calls=1, latency_ms=30),
        ]
    )
    assert summary.examples == 2
    assert summary.model_calls == 3
    assert summary.input_tokens == 100
    assert summary.output_tokens == 20
    assert summary.mean_latency_ms == 20
    assert summary.p95_latency_ms == 30


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [([], []), (["a"], []), ([], ["a"])],
)
def test_pair_metrics_reject_empty_or_mismatched_inputs(
    gold: list[str], predicted: list[str]
) -> None:
    with pytest.raises(ValueError):
        accuracy(gold, predicted)
