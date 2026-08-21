"""Deterministic ablation comparison and cautious diagnosis failure analysis."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from math_feedback_ai.diagnosis.service import DiagnosisAttemptTrace
from math_feedback_ai.domain.models import DiagnosisV1
from math_feedback_ai.domain.taxonomy import OverallStatus
from math_feedback_ai.evaluation.diagnosis import (
    DiagnosisEvaluationCase,
    DiagnosisExperimentReport,
    MetricSnapshot,
    ReferenceMode,
)


@dataclass(frozen=True, slots=True)
class CaseAblationChange:
    case_id: str
    category: str
    without_status: OverallStatus
    with_status: OverallStatus
    without_confidence: float
    with_confidence: float
    status_changed: bool
    reference_helped: bool
    possible_reference_anchoring: bool


@dataclass(frozen=True, slots=True)
class ReferenceAblationReport:
    benchmark_name: str
    examples: int
    metric_deltas_with_minus_without: dict[str, float | None]
    confidence_mean_delta: float
    input_token_delta: int | None
    output_token_delta: int | None
    mean_latency_ms_delta: float | None
    reference_helped_case_ids: tuple[str, ...]
    possible_reference_anchoring_case_ids: tuple[str, ...]
    changes: tuple[CaseAblationChange, ...]
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureFinding:
    case_id: str
    categories: tuple[str, ...]
    expected: DiagnosisV1
    predicted: DiagnosisV1
    attempts: tuple[DiagnosisAttemptTrace, ...]
    likely_cause: str
    recommended_fix: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "categories": list(self.categories),
            "expected": self.expected.model_dump(mode="json"),
            "predicted": self.predicted.model_dump(mode="json"),
            "attempts": [asdict(item) for item in self.attempts],
            "likely_cause": self.likely_cause,
            "recommended_fix": self.recommended_fix,
        }


@dataclass(frozen=True, slots=True)
class FailureAnalysisReport:
    benchmark_name: str
    annotation_status: str
    evaluated_cases: int
    failed_cases: int
    category_counts: dict[str, int]
    representative_case_ids: dict[str, tuple[str, ...]]
    findings: tuple[FailureFinding, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "annotation_status": self.annotation_status,
            "evaluated_cases": self.evaluated_cases,
            "failed_cases": self.failed_cases,
            "category_counts": self.category_counts,
            "representative_case_ids": self.representative_case_ids,
            "findings": [item.as_dict() for item in self.findings],
        }


def compare_reference_ablation(
    *,
    cases: Sequence[DiagnosisEvaluationCase],
    with_references: DiagnosisExperimentReport,
    without_references: DiagnosisExperimentReport,
) -> ReferenceAblationReport:
    """Compare matched WITH/WITHOUT runs and flag possible reference anchoring."""

    if with_references.reference_mode is not ReferenceMode.WITH_REFERENCES:
        raise ValueError("with_references report has the wrong reference mode")
    if without_references.reference_mode is not ReferenceMode.WITHOUT_REFERENCES:
        raise ValueError("without_references report has the wrong reference mode")
    if with_references.benchmark_name != without_references.benchmark_name:
        raise ValueError("ablation reports must use the same benchmark")

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("ablation cases must have unique IDs")
    with_by_id = {item.case_id: item for item in with_references.predictions}
    without_by_id = {item.case_id: item for item in without_references.predictions}
    if set(with_by_id) != set(case_by_id) or set(without_by_id) != set(case_by_id):
        raise ValueError("ablation prediction coverage does not match the cases")

    changes: list[CaseAblationChange] = []
    for case in cases:
        with_prediction = with_by_id[case.case_id].predicted
        without_prediction = without_by_id[case.case_id].predicted
        expected_status = case.expected_diagnosis.overall_status
        without_correct = without_prediction.overall_status is expected_status
        with_correct = with_prediction.overall_status is expected_status
        anchoring = (
            case.is_valid_alternative
            and without_prediction.overall_status
            in {OverallStatus.FULLY_CORRECT, OverallStatus.CORRECT_BUT_INEFFICIENT}
            and with_prediction.overall_status
            in {OverallStatus.INCOMPLETE, OverallStatus.INCORRECT}
        )
        changes.append(
            CaseAblationChange(
                case_id=case.case_id,
                category=case.category,
                without_status=without_prediction.overall_status,
                with_status=with_prediction.overall_status,
                without_confidence=without_prediction.confidence,
                with_confidence=with_prediction.confidence,
                status_changed=(
                    without_prediction.overall_status is not with_prediction.overall_status
                ),
                reference_helped=not without_correct and with_correct,
                possible_reference_anchoring=anchoring,
            )
        )

    metric_names = sorted(set(with_references.metrics) | set(without_references.metrics))
    metric_deltas = {
        name: _metric_delta(with_references.metrics.get(name), without_references.metrics.get(name))
        for name in metric_names
    }
    anchoring_ids = tuple(item.case_id for item in changes if item.possible_reference_anchoring)
    helped_ids = tuple(item.case_id for item in changes if item.reference_helped)
    return ReferenceAblationReport(
        benchmark_name=with_references.benchmark_name,
        examples=len(cases),
        metric_deltas_with_minus_without=metric_deltas,
        confidence_mean_delta=(
            with_references.confidence.mean - without_references.confidence.mean
        ),
        input_token_delta=_optional_delta(
            with_references.provider_usage.input_tokens,
            without_references.provider_usage.input_tokens,
        ),
        output_token_delta=_optional_delta(
            with_references.provider_usage.output_tokens,
            without_references.provider_usage.output_tokens,
        ),
        mean_latency_ms_delta=_optional_delta(
            with_references.provider_usage.mean_latency_ms,
            without_references.provider_usage.mean_latency_ms,
        ),
        reference_helped_case_ids=helped_ids,
        possible_reference_anchoring_case_ids=anchoring_ids,
        changes=tuple(changes),
        recommendation=_ablation_recommendation(metric_deltas, anchoring_ids),
    )


def analyze_diagnosis_failures(
    *,
    cases: Sequence[DiagnosisEvaluationCase],
    report: DiagnosisExperimentReport,
) -> FailureAnalysisReport:
    """Classify observable failures without pretending to know hidden causes."""

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("failure-analysis cases must have unique IDs")
    if {item.case_id for item in report.predictions} != set(case_by_id):
        raise ValueError("failure-analysis prediction coverage does not match the cases")

    findings: list[FailureFinding] = []
    for prediction in report.predictions:
        case = case_by_id[prediction.case_id]
        categories = _failure_categories(case, prediction.expected, prediction.predicted)
        if not categories:
            continue
        if any(trace.outcome == "invalid_structured_output" for trace in prediction.attempts):
            categories.insert(0, "malformed_structured_output")
        categories = list(dict.fromkeys(categories))
        cause, fix = _cause_and_fix(categories)
        findings.append(
            FailureFinding(
                case_id=case.case_id,
                categories=tuple(categories),
                expected=prediction.expected,
                predicted=prediction.predicted,
                attempts=prediction.attempts,
                likely_cause=cause,
                recommended_fix=fix,
            )
        )

    counts = Counter(category for finding in findings for category in finding.categories)
    representatives = {
        category: tuple(finding.case_id for finding in findings if category in finding.categories)[
            :5
        ]
        for category in sorted(counts)
    }
    return FailureAnalysisReport(
        benchmark_name=report.benchmark_name,
        annotation_status=report.annotation_status,
        evaluated_cases=len(cases),
        failed_cases=len(findings),
        category_counts=dict(sorted(counts.items())),
        representative_case_ids=representatives,
        findings=tuple(findings),
    )


def write_analysis_report(
    report: ReferenceAblationReport | FailureAnalysisReport, path: Path
) -> None:
    """Write one deterministic JSON analysis artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _failure_categories(
    case: DiagnosisEvaluationCase,
    expected: DiagnosisV1,
    predicted: DiagnosisV1,
) -> list[str]:
    categories: list[str] = []
    expected_status = expected.overall_status
    predicted_status = predicted.overall_status
    if expected_status is predicted_status:
        pass
    elif predicted_status is OverallStatus.INDETERMINATE:
        categories.append("excessive_abstention")
    elif case.is_valid_alternative and predicted_status in {
        OverallStatus.INCOMPLETE,
        OverallStatus.INCORRECT,
    }:
        categories.append("valid_alternative_rejected")
    elif (
        expected_status is OverallStatus.INCOMPLETE and predicted_status is OverallStatus.INCORRECT
    ):
        categories.append("incomplete_treated_as_incorrect")
    elif (
        expected_status is OverallStatus.INCORRECT and predicted_status is OverallStatus.INCOMPLETE
    ):
        categories.append("incorrect_treated_as_incomplete")
    elif expected_status is OverallStatus.INCORRECT and predicted_status in {
        OverallStatus.FULLY_CORRECT,
        OverallStatus.CORRECT_BUT_INEFFICIENT,
    }:
        categories.append("mathematical_reasoning_failure")
    else:
        categories.append("status_mismatch")

    expected_position = _position(case, expected)
    predicted_position = _position(case, predicted)
    if expected_position is not None and predicted_position != expected_position:
        if predicted_position is not None and abs(predicted_position - expected_position) == 1:
            categories.append("first_error_off_by_one")
        else:
            categories.append("first_error_localization_failure")
        if case.student_attempt.segmentation_ambiguous:
            categories.append("possible_parser_segmentation_problem")

    if (
        expected.first_issue is not None
        and predicted.first_issue is not None
        and expected_position == predicted_position
        and expected.first_issue.code is not predicted.first_issue.code
    ):
        categories.append("correct_localization_wrong_taxonomy")

    if categories and predicted.confidence >= 0.8:
        categories.append("overconfidence")
    return categories


def _position(case: DiagnosisEvaluationCase, diagnosis: DiagnosisV1) -> int | None:
    if diagnosis.first_issue is None:
        return None
    positions = {step.step_id: step.position for step in case.student_attempt.steps}
    return positions.get(diagnosis.first_issue.step_id)


def _cause_and_fix(categories: Sequence[str]) -> tuple[str, str]:
    if "malformed_structured_output" in categories:
        return (
            "The provider returned content that failed schema or cross-object validation.",
            "Inspect the recorded attempt outcome and strengthen provider schema enforcement only if the pattern repeats.",
        )
    if "possible_parser_segmentation_problem" in categories:
        return (
            "Conservative segmentation may have shifted the localization target; model and parser contributions require manual inspection.",
            "Review the raw spans and add a parser regression only if the segmentation itself is demonstrably wrong.",
        )
    if "valid_alternative_rejected" in categories:
        return (
            "The model or prompt may have anchored on a reference method; an ablation is needed to distinguish the cause.",
            "Compare the matched no-reference prediction before changing prompts or policy.",
        )
    if "correct_localization_wrong_taxonomy" in categories:
        return (
            "The mathematical location was preserved but the compact issue taxonomy was applied differently.",
            "Review annotation guidance and schema fit before adding taxonomy-specific prompt wording.",
        )
    if "excessive_abstention" in categories:
        return (
            "The validated output did not support a non-indeterminate diagnosis; this may reflect provider uncertainty, validation, or prompt ambiguity.",
            "Inspect per-attempt outcomes and confidence before changing the confidence floor.",
        )
    return (
        "A structurally valid diagnosis disagreed with the proposed annotation; automated evidence cannot separate mathematical-model and prompt errors.",
        "Manually inspect the case and matched ablation before making a targeted change.",
    )


def _metric_delta(
    with_metric: MetricSnapshot | None, without_metric: MetricSnapshot | None
) -> float | None:
    if with_metric is None or without_metric is None:
        return None
    if with_metric.value is None or without_metric.value is None:
        return None
    return with_metric.value - without_metric.value


def _optional_delta(with_value: int | float | None, without_value: int | float | None) -> Any:
    if with_value is None or without_value is None:
        return None
    return with_value - without_value


def _ablation_recommendation(
    metric_deltas: Mapping[str, float | None],
    anchoring_ids: Sequence[str],
) -> str:
    if anchoring_ids:
        return "independent_validity_then_reference_review"
    status_delta = metric_deltas.get("overall_status_accuracy")
    localization_delta = metric_deltas.get("first_issue_exact_accuracy")
    if status_delta is not None and status_delta > 0 and (localization_delta or 0) >= 0:
        return "with_references"
    if status_delta is not None and status_delta < 0:
        return "without_references"
    return "insufficient_evidence"


__all__ = [
    "CaseAblationChange",
    "FailureAnalysisReport",
    "FailureFinding",
    "ReferenceAblationReport",
    "analyze_diagnosis_failures",
    "compare_reference_ablation",
    "write_analysis_report",
]
