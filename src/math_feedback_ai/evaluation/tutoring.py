"""End-to-end tutoring evaluation with provider-neutral call observations.

The runner in this module exercises the ordinary parser, diagnosis service,
deterministic policy, hint generator, and final safety gate.  It records only
provider metadata at the model boundary: prompts and generated private content
are deliberately absent from call observations.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from math_feedback_ai.diagnosis.service import (
    DiagnosisRun,
    DiagnosisService,
    DiagnosisServiceConfig,
)
from math_feedback_ai.domain.models import (
    DiagnosisV1,
    HintPlan,
    Problem,
    ReferenceSolution,
    StudentAttempt,
    TutorDecision,
    TutoringExampleV1,
    TutorResult,
)
from math_feedback_ai.domain.taxonomy import RevealLevel, TutorAction
from math_feedback_ai.hints.generator import HintGenerationResult, HintGenerator
from math_feedback_ai.hints.leakage import HintSafetyContext
from math_feedback_ai.model.client import (
    GenerationResult,
    ModelClient,
    ModelClientError,
    StructuredContent,
    StructuredGenerationRequest,
    TextGenerationRequest,
    TokenUsage,
)
from math_feedback_ai.orchestration.tutor import TutorContext, TutorOrchestrator, TutorSafetyError

type ModelCallOperation = Literal["structured", "text"]
type ModelCallOutcome = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class TutoringEvaluationCase:
    """One independently labelled input for a complete tutoring run."""

    case_id: str
    problem: Problem
    student_solution: str
    expected_decision: TutorDecision
    attempt_id: str | None = None
    context: TutorContext = field(default_factory=TutorContext)
    category: str = "unspecified"
    human_review_status: str = "not_recorded"

    def __post_init__(self) -> None:
        for name in ("case_id", "student_solution", "category", "human_review_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must contain non-whitespace text")
        if self.attempt_id is not None and not self.attempt_id.strip():
            raise ValueError("attempt_id must contain non-whitespace text when provided")


@dataclass(frozen=True, slots=True)
class ModelCallObservation:
    """Non-content metadata for one provider-neutral model call."""

    call_number: int
    operation: ModelCallOperation
    stage: str
    outcome: ModelCallOutcome
    usage: TokenUsage | None = None
    model_name: str | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.call_number <= 0:
            raise ValueError("call_number must be positive")
        if not self.stage.strip():
            raise ValueError("stage must contain non-whitespace text")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative when provided")
        if self.outcome == "success" and self.error_type is not None:
            raise ValueError("successful calls cannot carry an error_type")
        if self.outcome == "error" and not self.error_type:
            raise ValueError("failed calls require an error_type")

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_number": self.call_number,
            "operation": self.operation,
            "stage": self.stage,
            "outcome": self.outcome,
            "usage": asdict(self.usage) if self.usage is not None else None,
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class TutoringUsageSummary:
    """Aggregate provider observations for the complete tutoring pipeline."""

    model_calls: int
    diagnosis_calls: int
    hint_calls: int
    successful_calls: int
    failed_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    token_observed_calls: int
    input_token_observed_calls: int
    output_token_observed_calls: int
    total_token_observed_calls: int
    latency_observed_calls: int
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    models: tuple[str, ...]
    outcomes_by_stage: dict[str, int]


@dataclass(frozen=True, slots=True)
class TutoringMetricSnapshot:
    """Metric value with an explicit, possibly inapplicable denominator."""

    value: float | None
    numerator: float
    denominator: int


@dataclass(frozen=True, slots=True)
class HintGenerationObservation:
    """Content-free audit signals retained from one hint-generator invocation."""

    model_call_count: int
    successful_generation_count: int
    regenerated: bool
    safe_fallback_used: bool
    model_output_published: bool
    error_types: tuple[str, ...]
    rejected_violation_codes: tuple[str, ...]

    @property
    def failed(self) -> bool:
        """Whether calls occurred but no model-generated hint reached the student."""

        return (
            self.model_call_count > 0
            and self.safe_fallback_used
            and not self.model_output_published
        )


@dataclass(frozen=True, slots=True)
class TutoringCasePrediction:
    """One complete pipeline result and the model calls that produced it."""

    case_id: str
    category: str
    expected_decision: TutorDecision
    result: TutorResult | None
    model_calls: tuple[ModelCallObservation, ...]
    diagnosis_attempt_outcomes: tuple[str, ...]
    hint_generation: HintGenerationObservation | None = None
    pipeline_error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must contain non-whitespace text")
        if (self.result is None) == (self.pipeline_error_type is None):
            raise ValueError("exactly one of result or pipeline_error_type must be present")

    @property
    def action_agrees(self) -> bool:
        return (
            self.result is not None and self.result.decision.action is self.expected_decision.action
        )

    @property
    def reveal_compliant(self) -> bool:
        return (
            self.result is not None
            and self.result.decision.max_reveal_level <= self.expected_decision.max_reveal_level
            and self.result.response.reveal_level <= self.expected_decision.max_reveal_level
        )

    @property
    def leakage_violation(self) -> bool:
        return (
            self.result is None
            or self.result.leakage_check is None
            or not self.result.leakage_check.passed
        )

    @property
    def safe_fallback_used(self) -> bool:
        return self.result is not None and self.result.response.safe_fallback_used

    @property
    def hint_generation_failed(self) -> bool:
        return self.hint_generation is not None and self.hint_generation.failed

    @property
    def is_hint_case(self) -> bool:
        expected_hint = self.expected_decision.action not in {
            TutorAction.WAIT,
            TutorAction.ASK_STUDENT,
        }
        actual_hint = self.result is not None and self.result.decision.action not in {
            TutorAction.WAIT,
            TutorAction.ASK_STUDENT,
        }
        return expected_hint or actual_hint

    def as_dict(self) -> dict[str, Any]:
        actual: dict[str, Any] | None = None
        if self.result is not None:
            leakage = self.result.leakage_check
            actual = {
                "attempt_id": self.result.attempt.attempt_id,
                "diagnosis_status": self.result.diagnosis.overall_status.value,
                "first_issue_step_id": (
                    self.result.diagnosis.first_issue.step_id
                    if self.result.diagnosis.first_issue is not None
                    else None
                ),
                "first_issue_code": (
                    self.result.diagnosis.first_issue.code.value
                    if self.result.diagnosis.first_issue is not None
                    else None
                ),
                "decision": self.result.decision.model_dump(mode="json"),
                "response": self.result.response.model_dump(mode="json"),
                "leakage_check": leakage.model_dump(mode="json") if leakage is not None else None,
            }
        return {
            "case_id": self.case_id,
            "category": self.category,
            "expected_decision": self.expected_decision.model_dump(mode="json"),
            "actual": actual,
            "pipeline_error_type": self.pipeline_error_type,
            "signals": {
                "action_agrees": self.action_agrees,
                "reveal_compliant": self.reveal_compliant,
                "leakage_violation": self.leakage_violation,
                "safe_fallback_used": self.safe_fallback_used,
                "hint_generation_failed": self.hint_generation_failed,
            },
            "diagnosis_attempt_outcomes": list(self.diagnosis_attempt_outcomes),
            "hint_generation": (
                asdict(self.hint_generation) if self.hint_generation is not None else None
            ),
            "model_calls": [item.as_dict() for item in self.model_calls],
        }


@dataclass(frozen=True, slots=True)
class HintReviewRecord:
    """Compact public-output record intended for qualitative hint review."""

    case_id: str
    category: str
    problem_statement: str
    student_solution: str
    expected_action: TutorAction
    actual_action: TutorAction | None
    expected_max_reveal_level: RevealLevel
    actual_reveal_level: RevealLevel | None
    target_step_id: str | None
    tutor_message: str | None
    leakage_passed: bool | None
    leakage_violation_codes: tuple[str, ...]
    safe_fallback_used: bool
    hint_generation_failed: bool
    pipeline_error_type: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "problem_statement": self.problem_statement,
            "student_solution": self.student_solution,
            "expected_action": self.expected_action.value,
            "actual_action": self.actual_action.value if self.actual_action is not None else None,
            "expected_max_reveal_level": int(self.expected_max_reveal_level),
            "actual_reveal_level": (
                int(self.actual_reveal_level) if self.actual_reveal_level is not None else None
            ),
            "target_step_id": self.target_step_id,
            "tutor_message": self.tutor_message,
            "leakage_passed": self.leakage_passed,
            "leakage_violation_codes": list(self.leakage_violation_codes),
            "safe_fallback_used": self.safe_fallback_used,
            "hint_generation_failed": self.hint_generation_failed,
            "pipeline_error_type": self.pipeline_error_type,
        }


@dataclass(frozen=True, slots=True)
class TutoringExperimentReport:
    """Aggregate complete-pipeline metrics with auditable per-case records."""

    benchmark_name: str
    annotation_status: str
    examples: int
    hint_cases: int
    metrics: dict[str, TutoringMetricSnapshot]
    provider_usage: TutoringUsageSummary
    predictions: tuple[TutoringCasePrediction, ...]
    hint_review: tuple[HintReviewRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "annotation_status": self.annotation_status,
            "examples": self.examples,
            "hint_cases": self.hint_cases,
            "metrics": {name: asdict(metric) for name, metric in self.metrics.items()},
            "provider_usage": asdict(self.provider_usage),
            "predictions": [item.as_dict() for item in self.predictions],
            "hint_review": [item.as_dict() for item in self.hint_review],
        }


@dataclass(frozen=True, slots=True)
class TutoringArtifactPaths:
    """Deterministic files produced for one complete-pipeline experiment."""

    summary: Path
    predictions: Path
    hint_review: Path


def development_tutoring_cases(
    examples: Sequence[TutoringExampleV1],
) -> tuple[TutoringEvaluationCase, ...]:
    """Adapt canonical development fixtures to first-turn tutoring cases."""

    cases: list[TutoringEvaluationCase] = []
    for example in examples:
        if example.expected_decision is None:
            raise ValueError(f"example {example.example_id!r} needs an expected decision")
        cases.append(
            TutoringEvaluationCase(
                case_id=example.example_id,
                problem=example.problem,
                student_solution=example.student_attempt.raw_text,
                expected_decision=example.expected_decision,
                attempt_id=example.student_attempt.attempt_id,
                category=_category_from_notes(example.provenance.notes),
                human_review_status="not_human_validated",
            )
        )
    return tuple(cases)


def run_tutoring_experiment(
    *,
    benchmark_name: str,
    annotation_status: str,
    cases: Sequence[TutoringEvaluationCase],
    client: ModelClient,
    diagnosis_config: DiagnosisServiceConfig | None = None,
    hint_timeout_seconds: float = 30.0,
    use_references: bool = True,
) -> TutoringExperimentReport:
    """Run every case through the standard pipeline and retain provider traces."""

    if not benchmark_name.strip():
        raise ValueError("benchmark_name must contain non-whitespace text")
    if not annotation_status.strip():
        raise ValueError("annotation_status must contain non-whitespace text")
    if not cases:
        raise ValueError("at least one tutoring evaluation case is required")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("tutoring evaluation case IDs must be unique")

    observed_client = _ObservedModelClient(client)
    observed_diagnoser = _ObservedDiagnoser(DiagnosisService(observed_client, diagnosis_config))
    observed_hint_producer = _ObservedHintProducer(
        HintGenerator(observed_client, timeout_seconds=hint_timeout_seconds)
    )
    orchestrator = TutorOrchestrator(
        diagnosis_service=observed_diagnoser,
        hint_generator=observed_hint_producer,
    )
    predictions: list[TutoringCasePrediction] = []

    for case in cases:
        call_start = len(observed_client.calls)
        diagnosis_start = len(observed_diagnoser.runs)
        hint_start = len(observed_hint_producer.results)
        try:
            result = orchestrator.tutor(
                case.problem,
                case.student_solution,
                context=case.context,
                attempt_id=case.attempt_id,
                reference_solutions=(case.problem.reference_solutions if use_references else ()),
            )
        except TutorSafetyError as exc:
            diagnosis_run = _single_new_item(
                observed_diagnoser.runs,
                diagnosis_start,
                "diagnosis run",
            )
            hint_result = _optional_single_new_item(
                observed_hint_producer.results,
                hint_start,
                "hint-generation result",
            )
            prediction = TutoringCasePrediction(
                case_id=case.case_id,
                category=case.category,
                expected_decision=case.expected_decision,
                result=None,
                model_calls=tuple(observed_client.calls[call_start:]),
                diagnosis_attempt_outcomes=tuple(
                    attempt.outcome for attempt in diagnosis_run.attempts
                ),
                hint_generation=_hint_generation_observation(hint_result, None),
                pipeline_error_type=type(exc).__name__,
            )
        else:
            diagnosis_run = _single_new_item(
                observed_diagnoser.runs,
                diagnosis_start,
                "diagnosis run",
            )
            hint_result = _optional_single_new_item(
                observed_hint_producer.results,
                hint_start,
                "hint-generation result",
            )
            prediction = TutoringCasePrediction(
                case_id=case.case_id,
                category=case.category,
                expected_decision=case.expected_decision,
                result=result,
                model_calls=tuple(observed_client.calls[call_start:]),
                diagnosis_attempt_outcomes=tuple(
                    attempt.outcome for attempt in diagnosis_run.attempts
                ),
                hint_generation=_hint_generation_observation(hint_result, result),
            )
        predictions.append(prediction)

    return _build_tutoring_report(
        benchmark_name=benchmark_name,
        annotation_status=annotation_status,
        cases=cases,
        predictions=predictions,
    )


_UNSAFE_ARTIFACT_CHARACTER = re.compile(r"[^A-Za-z0-9._-]+")


def write_tutoring_artifacts(
    report: TutoringExperimentReport,
    output_directory: Path,
    *,
    stem: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> TutoringArtifactPaths:
    """Write a compact summary, content-minimized predictions, and hint-review JSONL."""

    raw_stem = stem or report.benchmark_name
    safe_stem = _UNSAFE_ARTIFACT_CHARACTER.sub("-", raw_stem).strip(".-")
    if not safe_stem:
        raise ValueError("tutoring artifact stem must contain a safe filename character")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = TutoringArtifactPaths(
        summary=output_directory / f"{safe_stem}.summary.json",
        predictions=output_directory / f"{safe_stem}.predictions.jsonl",
        hint_review=output_directory / f"{safe_stem}.hint-review.jsonl",
    )
    existing = tuple(
        path for path in (paths.summary, paths.predictions, paths.hint_review) if path.exists()
    )
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite tutoring artifacts: {joined}")

    summary = report.as_dict()
    summary.pop("predictions")
    summary.pop("hint_review")
    summary["predictions_file"] = paths.predictions.name
    summary["hint_review_file"] = paths.hint_review.name
    if run_metadata is not None:
        summary["run_metadata"] = dict(run_metadata)
    if report.annotation_status != "human_validated":
        summary["research_qualification"] = (
            "These tutoring results do not use fully human-validated labels and are not "
            "standalone evidence of tutoring efficacy."
        )

    paths.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths.predictions.write_text(
        "".join(
            f"{json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True)}\n"
            for item in report.predictions
        ),
        encoding="utf-8",
        newline="\n",
    )
    paths.hint_review.write_text(
        "".join(
            f"{json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True)}\n"
            for item in report.hint_review
        ),
        encoding="utf-8",
        newline="\n",
    )
    return paths


@dataclass(slots=True)
class _ObservedModelClient:
    inner: ModelClient
    calls: list[ModelCallObservation] = field(default_factory=list)

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        try:
            result = self.inner.generate_text(request)
        except ModelClientError as exc:
            self._record_error("text", request.metadata, exc)
            raise
        self._record_result("text", request.metadata, result)
        return result

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> GenerationResult[StructuredContent]:
        try:
            result = self.inner.generate_structured(request)
        except ModelClientError as exc:
            self._record_error("structured", request.metadata, exc)
            raise
        self._record_result("structured", request.metadata, result)
        return result

    def _record_result(
        self,
        operation: ModelCallOperation,
        metadata: Mapping[str, str],
        result: GenerationResult[object],
    ) -> None:
        self.calls.append(
            ModelCallObservation(
                call_number=len(self.calls) + 1,
                operation=operation,
                stage=_stage(metadata, operation),
                outcome="success",
                usage=result.usage,
                model_name=result.model_name,
                finish_reason=result.finish_reason,
                latency_ms=result.latency_ms,
            )
        )

    def _record_error(
        self,
        operation: ModelCallOperation,
        metadata: Mapping[str, str],
        error: ModelClientError,
    ) -> None:
        self.calls.append(
            ModelCallObservation(
                call_number=len(self.calls) + 1,
                operation=operation,
                stage=_stage(metadata, operation),
                outcome="error",
                error_type=type(error).__name__,
            )
        )


@dataclass(slots=True)
class _ObservedDiagnoser:
    inner: DiagnosisService
    runs: list[DiagnosisRun] = field(default_factory=list)

    def diagnose(
        self,
        problem: Problem,
        attempt: StudentAttempt,
        reference_solutions: Sequence[ReferenceSolution] | None = None,
    ) -> DiagnosisV1:
        run = self.inner.diagnose_with_trace(
            problem,
            attempt,
            reference_solutions=reference_solutions,
        )
        self.runs.append(run)
        return run.diagnosis


@dataclass(slots=True)
class _ObservedHintProducer:
    inner: HintGenerator
    results: list[HintGenerationResult] = field(default_factory=list)

    def generate(
        self,
        plan: HintPlan,
        *,
        safety_context: HintSafetyContext | None = None,
    ) -> HintGenerationResult:
        result = self.inner.generate(plan, safety_context=safety_context)
        self.results.append(result)
        return result


def _build_tutoring_report(
    *,
    benchmark_name: str,
    annotation_status: str,
    cases: Sequence[TutoringEvaluationCase],
    predictions: Sequence[TutoringCasePrediction],
) -> TutoringExperimentReport:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    prediction_ids = [item.case_id for item in predictions]
    if prediction_ids != [case.case_id for case in cases]:
        raise ValueError("predictions must remain in tutoring case order")

    all_indices = tuple(range(len(predictions)))
    completed_indices = tuple(
        index for index, item in enumerate(predictions) if item.result is not None
    )
    hint_call_indices = tuple(
        index
        for index, item in enumerate(predictions)
        if item.hint_generation is not None and item.hint_generation.model_call_count > 0
    )
    metrics = {
        "tutor_action_agreement": _indexed_snapshot(
            predictions,
            all_indices,
            lambda item: item.action_agrees,
        ),
        "reveal_compliance": _indexed_snapshot(
            predictions,
            completed_indices,
            lambda item: item.reveal_compliant,
        ),
        "leakage_violation_rate": _indexed_snapshot(
            predictions,
            completed_indices,
            lambda item: item.leakage_violation,
        ),
        "safe_fallback_rate": _indexed_snapshot(
            predictions,
            completed_indices,
            lambda item: item.safe_fallback_used,
        ),
        "hint_generation_failure_rate": _indexed_snapshot(
            predictions,
            hint_call_indices,
            lambda item: item.hint_generation_failed,
        ),
        "pipeline_failure_rate": _indexed_snapshot(
            predictions,
            all_indices,
            lambda item: item.result is None,
        ),
    }
    review = tuple(
        _hint_review_record(case, prediction)
        for case, prediction in zip(cases, predictions, strict=True)
        if prediction.is_hint_case
    )
    calls = [call for item in predictions for call in item.model_calls]
    return TutoringExperimentReport(
        benchmark_name=benchmark_name,
        annotation_status=annotation_status,
        examples=len(cases),
        hint_cases=sum(item.is_hint_case for item in predictions),
        metrics=metrics,
        provider_usage=_summarize_usage(calls),
        predictions=tuple(predictions),
        hint_review=review,
    )


def _hint_review_record(
    case: TutoringEvaluationCase,
    prediction: TutoringCasePrediction,
) -> HintReviewRecord:
    result = prediction.result
    leakage = result.leakage_check if result is not None else None
    return HintReviewRecord(
        case_id=case.case_id,
        category=case.category,
        problem_statement=case.problem.statement,
        student_solution=case.student_solution,
        expected_action=case.expected_decision.action,
        actual_action=result.response.action if result is not None else None,
        expected_max_reveal_level=case.expected_decision.max_reveal_level,
        actual_reveal_level=result.response.reveal_level if result is not None else None,
        target_step_id=result.response.target_step_id if result is not None else None,
        tutor_message=result.response.message if result is not None else None,
        leakage_passed=leakage.passed if leakage is not None else None,
        leakage_violation_codes=(
            tuple(item.code for item in leakage.violations) if leakage is not None else ()
        ),
        safe_fallback_used=prediction.safe_fallback_used,
        hint_generation_failed=prediction.hint_generation_failed,
        pipeline_error_type=prediction.pipeline_error_type,
    )


def _summarize_usage(calls: Sequence[ModelCallObservation]) -> TutoringUsageSummary:
    input_tokens = [
        item.usage.input_tokens
        for item in calls
        if item.usage is not None and item.usage.input_tokens is not None
    ]
    output_tokens = [
        item.usage.output_tokens
        for item in calls
        if item.usage is not None and item.usage.output_tokens is not None
    ]
    total_tokens = [
        item.usage.total_tokens
        for item in calls
        if item.usage is not None and item.usage.total_tokens is not None
    ]
    token_observed_calls = sum(
        item.usage is not None
        and any(
            value is not None
            for value in (
                item.usage.input_tokens,
                item.usage.output_tokens,
                item.usage.total_tokens,
            )
        )
        for item in calls
    )
    latencies = sorted(item.latency_ms for item in calls if item.latency_ms is not None)
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
    return TutoringUsageSummary(
        model_calls=len(calls),
        diagnosis_calls=sum(item.stage == "diagnosis" for item in calls),
        hint_calls=sum(item.stage == "hint_generator" for item in calls),
        successful_calls=sum(item.outcome == "success" for item in calls),
        failed_calls=sum(item.outcome == "error" for item in calls),
        input_tokens=sum(input_tokens) if input_tokens else None,
        output_tokens=sum(output_tokens) if output_tokens else None,
        total_tokens=sum(total_tokens) if total_tokens else None,
        token_observed_calls=token_observed_calls,
        input_token_observed_calls=len(input_tokens),
        output_token_observed_calls=len(output_tokens),
        total_token_observed_calls=len(total_tokens),
        latency_observed_calls=len(latencies),
        mean_latency_ms=fmean(latencies) if latencies else None,
        p95_latency_ms=p95,
        models=tuple(sorted({item.model_name for item in calls if item.model_name is not None})),
        outcomes_by_stage=dict(
            sorted(Counter(f"{item.stage}:{item.outcome}" for item in calls).items())
        ),
    )


def _stage(metadata: Mapping[str, str], operation: ModelCallOperation) -> str:
    return metadata.get("stage") or metadata.get("component") or f"{operation}_generation"


def _category_from_notes(notes: str | None) -> str:
    if notes is None:
        return "unspecified"
    for item in notes.split(";"):
        key, separator, value = item.strip().partition("=")
        if separator and key == "category" and value:
            return value
    return "unspecified"


def _hint_generation_observation(
    generation: HintGenerationResult | None,
    public_result: TutorResult | None,
) -> HintGenerationObservation | None:
    if generation is None:
        return None
    model_output_published = (
        public_result is not None
        and not public_result.response.safe_fallback_used
        and any(
            item.content.strip() == public_result.response.message
            for item in generation.generations
        )
    )
    violation_codes = tuple(
        dict.fromkeys(
            violation.code for check in generation.rejected_checks for violation in check.violations
        )
    )
    return HintGenerationObservation(
        model_call_count=generation.model_call_count,
        successful_generation_count=len(generation.generations),
        regenerated=generation.regenerated,
        safe_fallback_used=(
            generation.safe_fallback_used
            or (public_result is not None and public_result.response.safe_fallback_used)
        ),
        model_output_published=model_output_published,
        error_types=generation.errors,
        rejected_violation_codes=violation_codes,
    )


def _indexed_snapshot(
    predictions: Sequence[TutoringCasePrediction],
    indices: Sequence[int],
    predicate: Callable[[TutoringCasePrediction], bool],
) -> TutoringMetricSnapshot:
    numerator = sum(bool(predicate(predictions[index])) for index in indices)
    return _snapshot(numerator, len(indices))


def _snapshot(numerator: float, denominator: int) -> TutoringMetricSnapshot:
    return TutoringMetricSnapshot(
        value=numerator / denominator if denominator else None,
        numerator=numerator,
        denominator=denominator,
    )


def _single_new_item[T](items: Sequence[T], start: int, label: str) -> T:
    new_items = items[start:]
    if len(new_items) != 1:
        raise RuntimeError(f"expected exactly one {label}, observed {len(new_items)}")
    return new_items[0]


def _optional_single_new_item[T](
    items: Sequence[T],
    start: int,
    label: str,
) -> T | None:
    new_items = items[start:]
    if len(new_items) > 1:
        raise RuntimeError(f"expected at most one {label}, observed {len(new_items)}")
    return new_items[0] if new_items else None


__all__ = [
    "HintGenerationObservation",
    "HintReviewRecord",
    "ModelCallObservation",
    "TutoringArtifactPaths",
    "TutoringCasePrediction",
    "TutoringEvaluationCase",
    "TutoringExperimentReport",
    "TutoringMetricSnapshot",
    "TutoringUsageSummary",
    "development_tutoring_cases",
    "run_tutoring_experiment",
    "write_tutoring_artifacts",
]
