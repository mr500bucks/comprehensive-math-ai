"""Loading and deterministic scoring for the synthetic development benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from math_feedback_ai.domain.models import TutoringExampleV1, TutorResult
from math_feedback_ai.domain.taxonomy import OverallStatus, RevealLevel
from math_feedback_ai.evaluation.metrics import (
    MetricResult,
    accuracy,
    false_rejection_rate,
    localization_accuracy,
    macro_f1,
    rate,
)

DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[3] / "evaluation" / "benchmarks" / "development_v1.jsonl"
)


class BenchmarkFormatError(ValueError):
    """A benchmark line is malformed or violates the canonical schema."""


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Core task metrics for one complete prediction set."""

    benchmark_examples: int
    overall_status_accuracy: MetricResult
    overall_status_macro_f1: MetricResult
    correct_solution_false_rejection_rate: MetricResult
    first_issue_exact_accuracy: MetricResult
    first_issue_adjacent_accuracy: MetricResult
    policy_action_accuracy: MetricResult
    leakage_violation_rate: MetricResult
    reveal_level_violation_rate: MetricResult

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report preserving metric counts."""

        return asdict(self)


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> tuple[TutoringExampleV1, ...]:
    """Load canonical JSONL and reject blank, duplicate, or malformed records."""

    examples: list[TutoringExampleV1] = []
    identifiers: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkFormatError(f"cannot read benchmark {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BenchmarkFormatError(f"blank benchmark line at {line_number}")
        try:
            example = TutoringExampleV1.model_validate_json(line)
        except ValidationError as exc:
            raise BenchmarkFormatError(
                f"invalid benchmark example at line {line_number}: {exc}"
            ) from exc
        if example.example_id in identifiers:
            raise BenchmarkFormatError(f"duplicate example_id {example.example_id!r}")
        identifiers.add(example.example_id)
        examples.append(example)

    if not examples:
        raise BenchmarkFormatError("benchmark contains no examples")
    return tuple(examples)


def load_development_benchmark(
    path: Path | None = None,
) -> tuple[TutoringExampleV1, ...]:
    """Load an explicit JSONL file or the repository/wheel development fixture.

    Wheels do not carry the duplicate generated JSONL because its deterministic
    source definitions are already packaged. An editable/repository install
    still validates the committed file; an installed wheel rebuilds the exact
    canonical examples in memory.
    """

    if path is not None:
        return load_benchmark(path)
    if DEFAULT_BENCHMARK_PATH.is_file():
        return load_benchmark(DEFAULT_BENCHMARK_PATH)
    from math_feedback_ai.evaluation.development_builder import build_development_examples

    return build_development_examples()


def _step_position(example: TutoringExampleV1, step_id: str | None) -> int | None:
    if step_id is None:
        return None
    positions = {step.step_id: step.position for step in example.student_attempt.steps}
    try:
        return positions[step_id]
    except KeyError as exc:
        raise ValueError(
            f"example {example.example_id!r} references unknown step {step_id!r}"
        ) from exc


def evaluate_results(
    examples: Sequence[TutoringExampleV1],
    results: Mapping[str, TutorResult],
) -> BenchmarkReport:
    """Score exactly one result for every benchmark example."""

    if not examples:
        raise ValueError("at least one benchmark example is required")
    expected_ids = {example.example_id for example in examples}
    if len(expected_ids) != len(examples):
        raise ValueError("benchmark example IDs must be unique")
    result_ids = set(results)
    if result_ids != expected_ids:
        missing = sorted(expected_ids - result_ids)
        unexpected = sorted(result_ids - expected_ids)
        raise ValueError(
            f"prediction coverage mismatch (missing={missing}, unexpected={unexpected})"
        )

    gold_statuses: list[OverallStatus] = []
    predicted_statuses: list[OverallStatus] = []
    gold_positions: list[int | None] = []
    predicted_positions: list[int | None] = []
    gold_actions: list[str] = []
    predicted_actions: list[str] = []
    leakage_violations: list[bool] = []
    reveal_violations: list[bool] = []

    for example in examples:
        result = results[example.example_id]
        if result.attempt.attempt_id != example.student_attempt.attempt_id:
            raise ValueError(f"result attempt mismatch for {example.example_id!r}")
        if example.expected_decision is None:
            raise ValueError(f"example {example.example_id!r} needs an expected policy decision")

        gold_statuses.append(example.gold_diagnosis.overall_status)
        predicted_statuses.append(result.diagnosis.overall_status)
        gold_positions.append(
            _step_position(
                example,
                example.gold_diagnosis.first_issue.step_id
                if example.gold_diagnosis.first_issue
                else None,
            )
        )
        predicted_positions.append(
            _step_position(
                example,
                result.diagnosis.first_issue.step_id if result.diagnosis.first_issue else None,
            )
        )
        gold_actions.append(example.expected_decision.action.value)
        predicted_actions.append(result.decision.action.value)

        leakage_violations.append(
            (result.leakage_check is not None and not result.leakage_check.passed)
            or (result.response.reveal_level > RevealLevel.NONE and result.leakage_check is None)
        )
        reveal_violations.append(
            result.response.reveal_level > example.expected_decision.max_reveal_level
        )

    status_labels = tuple(OverallStatus)
    return BenchmarkReport(
        benchmark_examples=len(examples),
        overall_status_accuracy=accuracy(gold_statuses, predicted_statuses),
        overall_status_macro_f1=macro_f1(gold_statuses, predicted_statuses, labels=status_labels),
        correct_solution_false_rejection_rate=false_rejection_rate(
            gold_statuses,
            predicted_statuses,
            correct_labels=frozenset(
                {
                    OverallStatus.FULLY_CORRECT,
                    OverallStatus.CORRECT_BUT_INEFFICIENT,
                }
            ),
            rejection_labels=frozenset(
                {
                    OverallStatus.INCOMPLETE,
                    OverallStatus.INCORRECT,
                }
            ),
        ),
        first_issue_exact_accuracy=localization_accuracy(gold_positions, predicted_positions),
        first_issue_adjacent_accuracy=localization_accuracy(
            gold_positions, predicted_positions, tolerance=1
        ),
        policy_action_accuracy=accuracy(gold_actions, predicted_actions),
        leakage_violation_rate=rate(leakage_violations),
        reveal_level_violation_rate=rate(reveal_violations),
    )


__all__ = [
    "BenchmarkFormatError",
    "BenchmarkReport",
    "DEFAULT_BENCHMARK_PATH",
    "evaluate_results",
    "load_benchmark",
    "load_development_benchmark",
]
