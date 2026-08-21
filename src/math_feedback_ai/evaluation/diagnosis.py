"""Diagnosis-only experiment runner with transparent per-case predictions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING, Any

from math_feedback_ai.diagnosis.service import DiagnosisAttemptTrace, DiagnosisService
from math_feedback_ai.domain.models import (
    DiagnosisV1,
    Problem,
    StudentAttempt,
    TutorDecision,
    TutoringExampleV1,
)
from math_feedback_ai.domain.taxonomy import IssueCode, OverallStatus

if TYPE_CHECKING:
    from math_feedback_ai.evaluation.pilot import DiagnosisPilotCaseV1


class ReferenceMode(StrEnum):
    """Whether optional non-exhaustive references reach diagnosis."""

    WITH_REFERENCES = "with_references"
    WITHOUT_REFERENCES = "without_references"


@dataclass(frozen=True, slots=True)
class DiagnosisEvaluationCase:
    """Provider-independent case used by diagnosis experiments."""

    case_id: str
    problem: Problem
    student_attempt: StudentAttempt
    expected_diagnosis: DiagnosisV1
    expected_decision: TutorDecision | None
    category: str
    human_review_status: str
    mathematical_domains: tuple[str, ...] = ()
    characteristics: tuple[str, ...] = ()
    ambiguity_notes: str | None = None

    @property
    def is_valid_alternative(self) -> bool:
        return self.category == "alternative_valid" or "alternative_valid" in self.characteristics


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Metric value with an explicit denominator; zero means not applicable."""

    value: float | None
    numerator: float
    denominator: int


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    below_half: int


@dataclass(frozen=True, slots=True)
class ProviderUsageSummary:
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    token_observed_calls: int
    latency_observed_calls: int
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    models: tuple[str, ...]
    attempt_outcomes: dict[str, int]


@dataclass(frozen=True, slots=True)
class DiagnosisCasePrediction:
    case_id: str
    category: str
    human_review_status: str
    problem: Problem
    student_attempt: StudentAttempt
    mathematical_domains: tuple[str, ...]
    characteristics: tuple[str, ...]
    ambiguity_notes: str | None
    expected: DiagnosisV1
    predicted: DiagnosisV1
    attempts: tuple[DiagnosisAttemptTrace, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "human_review_status": self.human_review_status,
            "problem": self.problem.model_dump(mode="json"),
            "student_attempt": self.student_attempt.model_dump(mode="json"),
            "mathematical_domains": list(self.mathematical_domains),
            "characteristics": list(self.characteristics),
            "ambiguity_notes": self.ambiguity_notes,
            "expected": self.expected.model_dump(mode="json"),
            "predicted": self.predicted.model_dump(mode="json"),
            "attempts": [asdict(item) for item in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class DiagnosisExperimentReport:
    """Aggregate diagnosis metrics plus every underlying prediction."""

    benchmark_name: str
    annotation_status: str
    reference_mode: ReferenceMode
    examples: int
    metrics: dict[str, MetricSnapshot]
    confidence: ConfidenceSummary
    provider_usage: ProviderUsageSummary
    predictions: tuple[DiagnosisCasePrediction, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "annotation_status": self.annotation_status,
            "reference_mode": self.reference_mode.value,
            "examples": self.examples,
            "metrics": {name: asdict(metric) for name, metric in self.metrics.items()},
            "confidence": asdict(self.confidence),
            "provider_usage": asdict(self.provider_usage),
            "predictions": [item.as_dict() for item in self.predictions],
        }


@dataclass(frozen=True, slots=True)
class DiagnosisArtifactPaths:
    """Files produced for one reproducible diagnosis experiment condition."""

    summary: Path
    predictions: Path


def development_evaluation_cases(
    examples: Sequence[TutoringExampleV1],
) -> tuple[DiagnosisEvaluationCase, ...]:
    """Adapt canonical development fixtures to diagnosis-only cases."""

    cases: list[DiagnosisEvaluationCase] = []
    for example in examples:
        category = _category_from_notes(example.provenance.notes)
        cases.append(
            DiagnosisEvaluationCase(
                case_id=example.example_id,
                problem=example.problem,
                student_attempt=example.student_attempt,
                expected_diagnosis=example.gold_diagnosis,
                expected_decision=example.expected_decision,
                category=category,
                human_review_status="not_human_validated",
            )
        )
    return tuple(cases)


def pilot_evaluation_cases(
    cases: Sequence[DiagnosisPilotCaseV1],
) -> tuple[DiagnosisEvaluationCase, ...]:
    """Adapt provisional pilot cases without upgrading their review status."""

    return tuple(
        DiagnosisEvaluationCase(
            case_id=case.case_id,
            problem=case.problem,
            student_attempt=case.student_attempt,
            expected_diagnosis=case.proposed_diagnosis,
            expected_decision=case.proposed_decision,
            category=case.category,
            human_review_status=case.human_review_status.value,
            mathematical_domains=case.mathematical_domains,
            characteristics=case.characteristics,
            ambiguity_notes=case.ambiguity_notes,
        )
        for case in cases
    )


def run_diagnosis_experiment(
    *,
    benchmark_name: str,
    annotation_status: str,
    cases: Sequence[DiagnosisEvaluationCase],
    service: DiagnosisService,
    reference_mode: ReferenceMode,
) -> DiagnosisExperimentReport:
    """Run exactly one traced diagnosis for every case and score it."""

    if not cases:
        raise ValueError("at least one diagnosis evaluation case is required")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("diagnosis evaluation case IDs must be unique")

    predictions: list[DiagnosisCasePrediction] = []
    for case in cases:
        references = (
            case.problem.reference_solutions
            if reference_mode is ReferenceMode.WITH_REFERENCES
            else ()
        )
        run = service.diagnose_with_trace(
            case.problem,
            case.student_attempt,
            reference_solutions=references,
        )
        predictions.append(
            DiagnosisCasePrediction(
                case_id=case.case_id,
                category=case.category,
                human_review_status=case.human_review_status,
                problem=case.problem,
                student_attempt=case.student_attempt,
                mathematical_domains=case.mathematical_domains,
                characteristics=case.characteristics,
                ambiguity_notes=case.ambiguity_notes,
                expected=case.expected_diagnosis,
                predicted=run.diagnosis,
                attempts=run.attempts,
            )
        )

    return _build_report(
        benchmark_name=benchmark_name,
        annotation_status=annotation_status,
        reference_mode=reference_mode,
        cases=cases,
        predictions=predictions,
    )


def write_diagnosis_report(report: DiagnosisExperimentReport, path: Path) -> None:
    """Persist a complete report without dropping failed per-case outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


_UNSAFE_ARTIFACT_CHARACTER = re.compile(r"[^A-Za-z0-9._-]+")


def write_diagnosis_artifacts(
    report: DiagnosisExperimentReport,
    output_directory: Path,
    *,
    stem: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> DiagnosisArtifactPaths:
    """Write a compact summary and lossless one-prediction-per-line JSONL.

    The summary intentionally omits predictions and names the companion JSONL.
    Output is deterministic for a fixed report and does not include wall-clock
    timestamps, host paths, or secrets.
    """

    raw_stem = stem or f"{report.benchmark_name}.{report.reference_mode.value}"
    safe_stem = _UNSAFE_ARTIFACT_CHARACTER.sub("-", raw_stem).strip(".-")
    if not safe_stem:
        raise ValueError("diagnosis artifact stem must contain a safe filename character")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = DiagnosisArtifactPaths(
        summary=output_directory / f"{safe_stem}.summary.json",
        predictions=output_directory / f"{safe_stem}.predictions.jsonl",
    )
    existing = tuple(path for path in (paths.summary, paths.predictions) if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"diagnosis artifacts already exist: {names}")
    report_payload = report.as_dict()
    report_payload.pop("predictions")
    report_payload["predictions_file"] = paths.predictions.name
    if run_metadata is not None:
        report_payload["run_metadata"] = dict(run_metadata)
    if report.annotation_status == "pending":
        report_payload["research_qualification"] = (
            "These results use provisional Codex-generated annotations and are not "
            "validated research results."
        )
    paths.summary.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths.predictions.write_text(
        "".join(
            f"{json.dumps(prediction.as_dict(), ensure_ascii=False, sort_keys=True)}\n"
            for prediction in report.predictions
        ),
        encoding="utf-8",
        newline="\n",
    )
    return paths


def diagnosis_case_set_sha256(cases: Sequence[DiagnosisEvaluationCase]) -> str:
    """Fingerprint all scored inputs and proposed labels in source order."""

    payload = [
        {
            "case_id": case.case_id,
            "problem": case.problem.model_dump(mode="json"),
            "student_attempt": case.student_attempt.model_dump(mode="json"),
            "expected_diagnosis": case.expected_diagnosis.model_dump(mode="json"),
            "expected_decision": (
                case.expected_decision.model_dump(mode="json")
                if case.expected_decision is not None
                else None
            ),
            "category": case.category,
            "human_review_status": case.human_review_status,
            "mathematical_domains": case.mathematical_domains,
            "characteristics": case.characteristics,
            "ambiguity_notes": case.ambiguity_notes,
        }
        for case in cases
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _build_report(
    *,
    benchmark_name: str,
    annotation_status: str,
    reference_mode: ReferenceMode,
    cases: Sequence[DiagnosisEvaluationCase],
    predictions: Sequence[DiagnosisCasePrediction],
) -> DiagnosisExperimentReport:
    gold_statuses = [item.expected.overall_status for item in predictions]
    predicted_statuses = [item.predicted.overall_status for item in predictions]
    correct_statuses = {
        OverallStatus.FULLY_CORRECT,
        OverallStatus.CORRECT_BUT_INEFFICIENT,
    }
    rejection_statuses = {OverallStatus.INCOMPLETE, OverallStatus.INCORRECT}

    gold_positions = [
        _first_issue_position(case, item.expected)
        for case, item in zip(cases, predictions, strict=True)
    ]
    predicted_positions = [
        _first_issue_position(case, item.predicted)
        for case, item in zip(cases, predictions, strict=True)
    ]
    localization_indices = [
        index for index, position in enumerate(gold_positions) if position is not None
    ]
    issue_indices = [
        index for index, item in enumerate(predictions) if item.expected.first_issue is not None
    ]
    expected_issues: list[IssueCode] = []
    predicted_issues: list[IssueCode | None] = []
    for index in issue_indices:
        expected_issue = predictions[index].expected.first_issue
        if expected_issue is None:  # pragma: no cover - guarded by issue_indices
            raise AssertionError("issue index must identify an expected first issue")
        predicted_issue = predictions[index].predicted.first_issue
        expected_issues.append(expected_issue.code)
        predicted_issues.append(predicted_issue.code if predicted_issue is not None else None)
    incorrect_indices = [
        index for index, status in enumerate(gold_statuses) if status is OverallStatus.INCORRECT
    ]
    correct_indices = [
        index for index, status in enumerate(gold_statuses) if status in correct_statuses
    ]
    alternative_indices = [index for index, case in enumerate(cases) if case.is_valid_alternative]
    incomplete_indices = [
        index for index, status in enumerate(gold_statuses) if status is OverallStatus.INCOMPLETE
    ]

    metrics = {
        "overall_status_accuracy": _accuracy(gold_statuses, predicted_statuses),
        "overall_status_macro_f1": _macro_f1(
            gold_statuses, predicted_statuses, tuple(OverallStatus)
        ),
        "incorrect_as_correct_rate": _indexed_rate(
            incorrect_indices,
            lambda index: predicted_statuses[index] in correct_statuses,
        ),
        "correct_as_incorrect_rate": _indexed_rate(
            correct_indices,
            lambda index: predicted_statuses[index] is OverallStatus.INCORRECT,
        ),
        "valid_alternative_false_rejection_rate": _indexed_rate(
            alternative_indices,
            lambda index: predicted_statuses[index] in rejection_statuses,
        ),
        "first_issue_exact_accuracy": _indexed_rate(
            localization_indices,
            lambda index: predicted_positions[index] == gold_positions[index],
        ),
        "first_issue_adjacent_accuracy": _indexed_rate(
            localization_indices,
            lambda index: _positions_within_one(gold_positions[index], predicted_positions[index]),
        ),
        "issue_category_accuracy": _accuracy(expected_issues, predicted_issues),
        "issue_category_macro_f1": _macro_f1(
            expected_issues,
            predicted_issues,
            (*tuple(IssueCode), None),
        ),
        "completion_gap_accuracy": _indexed_rate(
            incomplete_indices,
            lambda index: (
                predictions[index].predicted.overall_status is OverallStatus.INCOMPLETE
                and predictions[index].predicted.completion_gap is not None
            ),
        ),
        "abstention_rate": _rate(
            [status is OverallStatus.INDETERMINATE for status in predicted_statuses]
        ),
        "structured_output_failure_rate": _rate(
            [
                any(
                    trace.outcome in {"invalid_structured_output", "output_error"}
                    for trace in item.attempts
                )
                for item in predictions
            ]
        ),
        "unrecovered_structured_output_failure_rate": _rate(
            [
                item.predicted.overall_status is OverallStatus.INDETERMINATE
                and any(
                    trace.outcome in {"invalid_structured_output", "output_error"}
                    for trace in item.attempts
                )
                for item in predictions
            ]
        ),
        "generation_length_limit_rate": _rate(
            [
                any(trace.finish_reason == "length" for trace in item.attempts)
                for item in predictions
            ]
        ),
    }
    confidences = [item.predicted.confidence for item in predictions]
    return DiagnosisExperimentReport(
        benchmark_name=benchmark_name,
        annotation_status=annotation_status,
        reference_mode=reference_mode,
        examples=len(cases),
        metrics=metrics,
        confidence=ConfidenceSummary(
            count=len(confidences),
            minimum=min(confidences),
            maximum=max(confidences),
            mean=fmean(confidences),
            median=median(confidences),
            below_half=sum(value < 0.5 for value in confidences),
        ),
        provider_usage=_summarize_usage(predictions),
        predictions=tuple(predictions),
    )


def _first_issue_position(case: DiagnosisEvaluationCase, diagnosis: DiagnosisV1) -> int | None:
    if diagnosis.first_issue is None:
        return None
    positions = {step.step_id: step.position for step in case.student_attempt.steps}
    return positions.get(diagnosis.first_issue.step_id)


def _category_from_notes(notes: str | None) -> str:
    if notes is None:
        return "unspecified"
    for item in notes.split(";"):
        key, separator, value = item.strip().partition("=")
        if separator and key == "category" and value:
            return value
    return "unspecified"


def _positions_within_one(expected: int | None, predicted: int | None) -> bool:
    return expected is not None and predicted is not None and abs(predicted - expected) <= 1


def _accuracy(expected: Sequence[object], predicted: Sequence[object]) -> MetricSnapshot:
    if len(expected) != len(predicted):
        raise ValueError("metric inputs must have the same length")
    matches = sum(left == right for left, right in zip(expected, predicted, strict=True))
    return _snapshot(matches, len(expected))


def _macro_f1(
    expected: Sequence[object],
    predicted: Sequence[object],
    labels: Sequence[object],
) -> MetricSnapshot:
    if len(expected) != len(predicted):
        raise ValueError("metric inputs must have the same length")
    if not expected:
        return _snapshot(0, 0)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            gold == label and actual == label
            for gold, actual in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            gold != label and actual == label
            for gold, actual in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            gold == label and actual != label
            for gold, actual in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return MetricSnapshot(value=fmean(scores), numerator=sum(scores), denominator=len(scores))


def _indexed_rate(indices: Sequence[int], predicate: Callable[[int], bool]) -> MetricSnapshot:
    return _snapshot(sum(bool(predicate(index)) for index in indices), len(indices))


def _rate(flags: Sequence[bool]) -> MetricSnapshot:
    return _snapshot(sum(flags), len(flags))


def _snapshot(numerator: float, denominator: int) -> MetricSnapshot:
    return MetricSnapshot(
        value=None if denominator == 0 else numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _summarize_usage(
    predictions: Sequence[DiagnosisCasePrediction],
) -> ProviderUsageSummary:
    attempts = [trace for prediction in predictions for trace in prediction.attempts]
    input_values = [
        trace.usage.input_tokens
        for trace in attempts
        if trace.usage is not None and trace.usage.input_tokens is not None
    ]
    output_values = [
        trace.usage.output_tokens
        for trace in attempts
        if trace.usage is not None and trace.usage.output_tokens is not None
    ]
    token_observed_calls = sum(
        trace.usage is not None
        and trace.usage.input_tokens is not None
        and trace.usage.output_tokens is not None
        for trace in attempts
    )
    latencies = sorted(trace.latency_ms for trace in attempts if trace.latency_ms is not None)
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
    models = tuple(sorted({trace.model_name for trace in attempts if trace.model_name is not None}))
    return ProviderUsageSummary(
        model_calls=len(attempts),
        input_tokens=sum(input_values) if input_values else None,
        output_tokens=sum(output_values) if output_values else None,
        token_observed_calls=token_observed_calls,
        latency_observed_calls=len(latencies),
        mean_latency_ms=fmean(latencies) if latencies else None,
        p95_latency_ms=p95,
        models=models,
        attempt_outcomes=dict(sorted(Counter(item.outcome for item in attempts).items())),
    )


__all__ = [
    "DiagnosisArtifactPaths",
    "ConfidenceSummary",
    "DiagnosisCasePrediction",
    "DiagnosisEvaluationCase",
    "DiagnosisExperimentReport",
    "MetricSnapshot",
    "ProviderUsageSummary",
    "ReferenceMode",
    "development_evaluation_cases",
    "diagnosis_case_set_sha256",
    "pilot_evaluation_cases",
    "run_diagnosis_experiment",
    "write_diagnosis_report",
    "write_diagnosis_artifacts",
]
