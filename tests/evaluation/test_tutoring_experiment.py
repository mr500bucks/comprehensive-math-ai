from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from math_feedback_ai.diagnosis.service import DiagnosisServiceConfig
from math_feedback_ai.domain import (
    DiagnosisV1,
    Issue,
    IssueCode,
    OverallStatus,
    Problem,
    ReferenceSolution,
    RevealLevel,
    StepAssessment,
    StepStatus,
    StudentAttempt,
    TutorAction,
    TutorDecision,
)
from math_feedback_ai.evaluation.tutoring import (
    TutoringEvaluationCase,
    TutoringExperimentReport,
    run_tutoring_experiment,
    write_tutoring_artifacts,
)
from math_feedback_ai.model.client import (
    GenerationResult,
    ModelTransportError,
    StructuredContent,
    TokenUsage,
)
from math_feedback_ai.model.fake import FakeModelClient
from math_feedback_ai.orchestration.tutor import TutorContext
from math_feedback_ai.parsing.steps import parse_student_attempt

_SECRET_REFERENCE = (
    "PRIVATE_REFERENCE_DO_NOT_PUBLISH: subtract four, divide by two, and report x=7."
)
_REJECTED_CANDIDATE = "The final answer is x=7."


def _problem() -> Problem:
    return Problem(
        problem_id="evaluation.pipeline",
        statement="Solve the equation and justify each step.",
        reference_solutions=(
            ReferenceSolution(reference_id="private-reference", text=_SECRET_REFERENCE),
        ),
    )


def _attempt(raw_text: str, attempt_id: str) -> StudentAttempt:
    return parse_student_attempt(
        "evaluation.pipeline",
        raw_text,
        attempt_id=attempt_id,
    )


def _correct(attempt: StudentAttempt) -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.FULLY_CORRECT,
        step_assessments=tuple(
            StepAssessment(
                step_id=step.step_id,
                status=StepStatus.VALID,
                confidence=0.99,
            )
            for step in attempt.steps
        ),
        reusable_prefix_end_step_id=attempt.steps[-1].step_id,
        earlier_reasoning_usable=True,
        confidence=0.99,
    )


def _incorrect(attempt: StudentAttempt) -> DiagnosisV1:
    target = attempt.steps[1]
    return DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.INCORRECT,
        step_assessments=(
            StepAssessment(
                step_id=attempt.steps[0].step_id,
                status=StepStatus.VALID,
                confidence=0.98,
            ),
            StepAssessment(
                step_id=target.step_id,
                status=StepStatus.INVALID,
                issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
                explanation="The arithmetic transition is invalid.",
                confidence=0.98,
            ),
        ),
        first_issue=Issue(
            step_id=target.step_id,
            code=IssueCode.COMPUTATION_ARITHMETIC,
            explanation="The arithmetic transition is invalid.",
            concept_tags=("maintaining equality",),
            confidence=0.98,
        ),
        reusable_prefix_end_step_id=attempt.steps[0].step_id,
        earlier_reasoning_usable=True,
        confidence=0.98,
    )


def _manual_decision(
    action: TutorAction,
    level: RevealLevel,
    target_step_id: str | None = None,
) -> TutorDecision:
    return TutorDecision(
        action=action,
        max_reveal_level=level,
        target_step_id=target_step_id,
        rationale_code="manual_evaluation_label",
    )


def _diagnosis_generation(
    diagnosis: DiagnosisV1,
    *,
    latency_ms: float,
) -> GenerationResult[StructuredContent]:
    return GenerationResult(
        content=diagnosis.model_dump(mode="json"),
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        model_name="diagnosis-model",
        finish_reason="stop",
        latency_ms=latency_ms,
    )


def _hint_generation(
    text: str,
    *,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> GenerationResult[str]:
    return GenerationResult(
        content=text,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        model_name="hint-model",
        finish_reason="stop",
        latency_ms=latency_ms,
    )


def _four_case_fixture() -> tuple[
    tuple[TutoringEvaluationCase, ...],
    tuple[GenerationResult[StructuredContent], ...],
]:
    problem = _problem()
    specs = (
        ("wait", "1. 2 + 2 = 4\n2. The equality is verified."),
        ("rewrite", "1. 2x + 4 = 18\n2. 2x = 22"),
        ("fallback", "1. 2x + 4 = 18\n2. 2x = 21"),
        ("provider-error", "1. 2x + 4 = 18\n2. 2x = 20"),
    )
    attempts = tuple(_attempt(text, f"attempt-{case_id}") for case_id, text in specs)
    diagnoses = (
        _correct(attempts[0]),
        _incorrect(attempts[1]),
        _incorrect(attempts[2]),
        _incorrect(attempts[3]),
    )
    cases = (
        TutoringEvaluationCase(
            case_id="wait",
            category="correct",
            problem=problem,
            student_solution=specs[0][1],
            attempt_id=attempts[0].attempt_id,
            expected_decision=_manual_decision(TutorAction.WAIT, RevealLevel.NONE),
        ),
        *(
            TutoringEvaluationCase(
                case_id=case_id,
                category="incorrect",
                problem=problem,
                student_solution=raw_text,
                attempt_id=attempt.attempt_id,
                expected_decision=_manual_decision(
                    TutorAction.LIGHT_HINT,
                    RevealLevel.LIGHT_DIRECTION,
                    diagnosis.first_issue.step_id if diagnosis.first_issue else None,
                ),
                context=TutorContext(final_answers=("x=7",)),
            )
            for (case_id, raw_text), attempt, diagnosis in zip(
                specs[1:], attempts[1:], diagnoses[1:], strict=True
            )
        ),
    )
    structured = tuple(
        _diagnosis_generation(diagnosis, latency_ms=latency)
        for diagnosis, latency in zip(diagnoses, (10.0, 20.0, 30.0, 40.0), strict=True)
    )
    return cases, structured


def _run_four_cases() -> tuple[TutoringExperimentReport, FakeModelClient]:
    cases, structured = _four_case_fixture()
    safe_rewrite = "What relationship should remain unchanged as you check this transition?"
    client = FakeModelClient(
        structured_responses=structured,
        text_responses=(
            _hint_generation(
                _REJECTED_CANDIDATE,
                input_tokens=3,
                output_tokens=1,
                latency_ms=5.0,
            ),
            _hint_generation(
                safe_rewrite,
                input_tokens=4,
                output_tokens=2,
                latency_ms=6.0,
            ),
            _hint_generation(
                _REJECTED_CANDIDATE,
                input_tokens=3,
                output_tokens=1,
                latency_ms=7.0,
            ),
            _hint_generation(
                "Therefore x = 7.",
                input_tokens=3,
                output_tokens=1,
                latency_ms=8.0,
            ),
            ModelTransportError("private provider detail"),
        ),
    )
    report = run_tutoring_experiment(
        benchmark_name="complete-pipeline",
        annotation_status="not_human_validated",
        cases=cases,
        client=client,
    )
    return report, client


def test_complete_pipeline_metrics_and_provider_observations() -> None:
    report, client = _run_four_cases()

    assert report.examples == 4
    assert report.hint_cases == 3
    assert report.metrics["tutor_action_agreement"].value == 1.0
    assert report.metrics["reveal_compliance"].value == 1.0
    assert report.metrics["leakage_violation_rate"].value == 0.0
    assert report.metrics["safe_fallback_rate"].numerator == 2
    assert report.metrics["safe_fallback_rate"].denominator == 4
    assert report.metrics["hint_generation_failure_rate"].numerator == 2
    assert report.metrics["hint_generation_failure_rate"].denominator == 3
    assert report.metrics["pipeline_failure_rate"].value == 0.0

    usage = report.provider_usage
    assert usage.model_calls == 9
    assert usage.diagnosis_calls == 4
    assert usage.hint_calls == 5
    assert usage.successful_calls == 8
    assert usage.failed_calls == 1
    assert usage.input_tokens == 53
    assert usage.output_tokens == 25
    assert usage.total_tokens == 78
    assert usage.token_observed_calls == 8
    assert usage.input_token_observed_calls == 8
    assert usage.output_token_observed_calls == 8
    assert usage.total_token_observed_calls == 8
    assert usage.latency_observed_calls == 8
    assert usage.mean_latency_ms == pytest.approx(15.75)
    assert usage.p95_latency_ms == 40.0
    assert usage.models == ("diagnosis-model", "hint-model")
    assert usage.outcomes_by_stage == {
        "diagnosis:success": 4,
        "hint_generator:error": 1,
        "hint_generator:success": 4,
    }
    assert client.structured_call_count == 4
    assert client.text_call_count == 5

    wait, rewrite, fallback, provider_error = report.predictions
    assert wait.hint_generation is None
    assert wait.diagnosis_attempt_outcomes == ("success",)
    assert rewrite.hint_generation is not None
    assert rewrite.hint_generation.regenerated
    assert rewrite.hint_generation.model_output_published
    assert not rewrite.hint_generation.failed
    assert rewrite.hint_generation.rejected_violation_codes
    assert fallback.hint_generation is not None and fallback.hint_generation.failed
    assert fallback.hint_generation.successful_generation_count == 2
    assert provider_error.hint_generation is not None
    assert provider_error.hint_generation.failed
    assert provider_error.hint_generation.error_types == ("ModelTransportError",)
    assert provider_error.model_calls[-1].outcome == "error"
    assert provider_error.model_calls[-1].error_type == "ModelTransportError"

    serialized = json.dumps(report.as_dict(), sort_keys=True)
    assert _SECRET_REFERENCE not in serialized
    assert _REJECTED_CANDIDATE not in serialized
    assert "private provider detail" not in serialized


def test_hint_failure_metric_is_not_applicable_when_no_hint_call_occurs() -> None:
    cases, structured = _four_case_fixture()
    client = FakeModelClient(structured_responses=(structured[0],))

    report = run_tutoring_experiment(
        benchmark_name="wait-only",
        annotation_status="human_validated",
        cases=(cases[0],),
        client=client,
        diagnosis_config=DiagnosisServiceConfig(
            max_output_tokens=321,
            max_attempts=1,
        ),
    )

    metric = report.metrics["hint_generation_failure_rate"]
    assert metric.value is None
    assert metric.numerator == 0
    assert metric.denominator == 0
    assert report.provider_usage.hint_calls == 0
    assert client.structured_requests[0].max_output_tokens == 321


def test_action_and_reveal_metrics_use_independent_expected_labels() -> None:
    cases, structured = _four_case_fixture()
    base = cases[1]
    escalated = TutoringEvaluationCase(
        case_id="manual-disagreement",
        category=base.category,
        problem=base.problem,
        student_solution=base.student_solution,
        attempt_id=base.attempt_id,
        expected_decision=base.expected_decision,
        context=TutorContext(
            prior_attempt_count=2,
            prior_action=TutorAction.TARGETED_HINT,
            current_reveal_level=RevealLevel.LOCATION,
            final_answers=("x=7",),
        ),
    )
    client = FakeModelClient(
        structured_responses=(structured[1],),
        text_responses=("Name the rule that keeps both sides balanced and check its conditions.",),
    )

    report = run_tutoring_experiment(
        benchmark_name="manual-label-disagreement",
        annotation_status="human_validated",
        cases=(escalated,),
        client=client,
    )

    prediction = report.predictions[0]
    assert prediction.result is not None
    assert prediction.result.decision.action is TutorAction.STRONG_HINT
    assert prediction.result.response.reveal_level is RevealLevel.CONCEPT
    assert report.metrics["tutor_action_agreement"].value == 0.0
    assert report.metrics["reveal_compliance"].value == 0.0


def test_artifacts_are_compact_private_safe_and_refuse_overwrite() -> None:
    report, _ = _run_four_cases()
    output_directory = Path("build")
    stem = f"test-tutoring-artifacts-{os.getpid()}"
    paths = write_tutoring_artifacts(
        report,
        output_directory,
        stem=stem,
        run_metadata={"provider": "scripted"},
    )
    try:
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        prediction_lines = paths.predictions.read_text(encoding="utf-8").splitlines()
        review_lines = paths.hint_review.read_text(encoding="utf-8").splitlines()
        all_artifacts = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (paths.summary, paths.predictions, paths.hint_review)
        )
        assert paths.summary.name == f"{stem}.summary.json"
        assert summary["predictions_file"] == paths.predictions.name
        assert summary["hint_review_file"] == paths.hint_review.name
        assert summary["run_metadata"] == {"provider": "scripted"}
        assert "research_qualification" in summary
        assert len(prediction_lines) == 4
        assert len(review_lines) == 3
        assert _SECRET_REFERENCE not in all_artifacts
        assert _REJECTED_CANDIDATE not in all_artifacts
        assert "SAFETY_REWRITE_REQUIRED" not in all_artifacts

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_tutoring_artifacts(report, output_directory, stem=stem)
        assert (
            write_tutoring_artifacts(
                report,
                output_directory,
                stem=stem,
                overwrite=True,
            )
            == paths
        )
    finally:
        for path in (paths.summary, paths.predictions, paths.hint_review):
            path.unlink(missing_ok=True)


def test_runner_rejects_duplicate_case_ids_before_calling_model() -> None:
    cases, _ = _four_case_fixture()
    duplicate = cases[0].__class__(
        case_id=cases[0].case_id,
        problem=cases[1].problem,
        student_solution=cases[1].student_solution,
        expected_decision=cases[1].expected_decision,
        attempt_id=cases[1].attempt_id,
    )
    client = FakeModelClient()

    with pytest.raises(ValueError, match="case IDs must be unique"):
        run_tutoring_experiment(
            benchmark_name="duplicates",
            annotation_status="human_validated",
            cases=(cases[0], duplicate),
            client=client,
        )

    assert client.structured_call_count == 0
    assert client.text_call_count == 0
