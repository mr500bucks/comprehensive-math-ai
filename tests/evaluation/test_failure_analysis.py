from __future__ import annotations

from math_feedback_ai.diagnosis.service import DiagnosisService, DiagnosisServiceConfig
from math_feedback_ai.domain.models import DiagnosisV1, Issue, StepAssessment
from math_feedback_ai.domain.taxonomy import (
    IssueCode,
    OverallStatus,
    ReferenceRelation,
    StepStatus,
)
from math_feedback_ai.evaluation.benchmark import load_development_benchmark
from math_feedback_ai.evaluation.diagnosis import (
    ReferenceMode,
    development_evaluation_cases,
    run_diagnosis_experiment,
)
from math_feedback_ai.evaluation.failure_analysis import (
    analyze_diagnosis_failures,
    compare_reference_ablation,
)
from math_feedback_ai.model.client import ModelOutputError
from math_feedback_ai.model.fake import FakeModelClient


def _incorrect_alternative_diagnosis(case_id: str) -> DiagnosisV1:
    case = next(
        case
        for case in development_evaluation_cases(load_development_benchmark())
        if case.case_id == case_id
    )
    first_step = case.student_attempt.steps[0]
    assessments = [
        StepAssessment(
            step_id=first_step.step_id,
            status=StepStatus.INVALID,
            issue_codes=(IssueCode.REASONING_INVALID_INFERENCE,),
            explanation="The alternative route was incorrectly rejected for this test.",
            confidence=0.9,
        )
    ]
    assessments.extend(
        StepAssessment(
            step_id=step.step_id,
            status=StepStatus.AMBIGUOUS,
            issue_codes=(IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,),
            explanation="This depends on the prior step.",
            confidence=0.7,
        )
        for step in case.student_attempt.steps[1:]
    )
    return DiagnosisV1(
        attempt_id=case.student_attempt.attempt_id,
        overall_status=OverallStatus.INCORRECT,
        step_assessments=tuple(assessments),
        first_issue=Issue(
            step_id=first_step.step_id,
            code=IssueCode.REASONING_INVALID_INFERENCE,
            explanation="The alternative route was incorrectly rejected for this test.",
            confidence=0.9,
        ),
        earlier_reasoning_usable=False,
        reference_relation=ReferenceRelation.SAME_METHOD,
        confidence=0.9,
    )


def test_failure_analysis_keeps_every_abstention_case_actionable() -> None:
    examples = load_development_benchmark()
    cases = development_evaluation_cases(examples)
    client = FakeModelClient(
        structured_responses=[ModelOutputError("provider unavailable") for _ in cases]
    )
    experiment = run_diagnosis_experiment(
        benchmark_name="development_v1",
        annotation_status="not_human_validated",
        cases=cases,
        service=DiagnosisService(client, DiagnosisServiceConfig(max_attempts=1)),
        reference_mode=ReferenceMode.WITHOUT_REFERENCES,
    )

    analysis = analyze_diagnosis_failures(cases=cases, report=experiment)

    assert analysis.evaluated_cases == 40
    assert analysis.failed_cases == 37
    assert analysis.category_counts["excessive_abstention"] == 35
    assert analysis.category_counts["first_error_localization_failure"] == 23
    assert len(analysis.findings) == 37
    assert analysis.annotation_status == "not_human_validated"


def test_reference_ablation_flags_alternative_proof_anchoring() -> None:
    alternative_case = development_evaluation_cases(load_development_benchmark())[6]
    expected_without_reference = alternative_case.expected_diagnosis.model_copy(
        update={"reference_relation": ReferenceRelation.NOT_USED}
    )
    without_client = FakeModelClient(
        structured_responses=[expected_without_reference.model_dump(mode="json")]
    )
    with_client = FakeModelClient(
        structured_responses=[
            _incorrect_alternative_diagnosis(alternative_case.case_id).model_dump(mode="json")
        ]
    )
    config = DiagnosisServiceConfig(max_attempts=1, minimum_confidence=0.0)
    without_report = run_diagnosis_experiment(
        benchmark_name="alternative-proof-ablation",
        annotation_status="not_human_validated",
        cases=(alternative_case,),
        service=DiagnosisService(without_client, config),
        reference_mode=ReferenceMode.WITHOUT_REFERENCES,
    )
    with_report = run_diagnosis_experiment(
        benchmark_name="alternative-proof-ablation",
        annotation_status="not_human_validated",
        cases=(alternative_case,),
        service=DiagnosisService(with_client, config),
        reference_mode=ReferenceMode.WITH_REFERENCES,
    )

    comparison = compare_reference_ablation(
        cases=(alternative_case,),
        with_references=with_report,
        without_references=without_report,
    )

    assert comparison.possible_reference_anchoring_case_ids == (alternative_case.case_id,)
    assert comparison.metric_deltas_with_minus_without["overall_status_accuracy"] == -1.0
    assert comparison.recommendation == "independent_validity_then_reference_review"
