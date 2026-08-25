from __future__ import annotations

from collections import Counter

from math_feedback_ai.domain import IssueCode, OverallStatus, StepStatus, TutorAction
from math_feedback_ai.evaluation.pilot import DiagnosisPilotReviewedCaseV1
from math_feedback_ai.evaluation.reviewed_diagnosis_pilot import (
    DEFAULT_RECONCILIATION_PATH,
    DEFAULT_REVIEWED_PILOT_PATH,
    build_reviewed_diagnosis_pilot_cases,
    committed_reviewed_diagnosis_pilot_is_current,
    load_reviewed_diagnosis_pilot,
    reviewed_diagnosis_pilot_sha256,
    reviewed_reconciliation_sha256,
    validate_built_reviewed_diagnosis_pilot,
)

EXPECTED_REVIEWED_SHA256 = "1839a5c343031d14d9b620e6206768ee302fe2ca4ab8ffb13a6862b079eb1816"
EXPECTED_RECONCILIATION_SHA256 = "b6cf001b7d41715e088c53058d77c810faabfe774eb2e057d12bda16c330c38a"


def _by_id() -> dict[str, DiagnosisPilotReviewedCaseV1]:
    return {case.case_id: case for case in build_reviewed_diagnosis_pilot_cases()}


def test_reviewed_pilot_is_deterministic_current_and_human_reviewed() -> None:
    cases = build_reviewed_diagnosis_pilot_cases()

    assert len(cases) == 50
    assert cases == build_reviewed_diagnosis_pilot_cases()
    assert load_reviewed_diagnosis_pilot() == cases
    assert all(case.human_review_status == "reviewed" for case in cases)
    assert all(case.reviewed_diagnosis.schema_version == "1.1" for case in cases)
    assert validate_built_reviewed_diagnosis_pilot().passed
    assert committed_reviewed_diagnosis_pilot_is_current()
    assert reviewed_diagnosis_pilot_sha256() == EXPECTED_REVIEWED_SHA256
    assert reviewed_reconciliation_sha256() == EXPECTED_RECONCILIATION_SHA256
    assert DEFAULT_REVIEWED_PILOT_PATH.is_file()
    assert DEFAULT_RECONCILIATION_PATH.is_file()


def test_exactly_eight_cases_record_modified_dispositions() -> None:
    modified = {
        case.case_id
        for case in build_reviewed_diagnosis_pilot_cases()
        if case.review_disposition == "modified"
    }

    assert modified == {
        "pilot.14.expand-fourth-power",
        "pilot.15.gcd-by-divisor-lists",
        "pilot.16.three-coins-enumeration",
        "pilot.17.derivative-from-definition",
        "pilot.22.transpose-sign",
        "pilot.36.pythagorean-on-any-triangle",
        "pilot.40.zero-derivative-at-point",
        "pilot.44.gcd-lucky-claim",
    }


def test_reviewed_status_action_and_category_distributions_are_reconciled() -> None:
    cases = build_reviewed_diagnosis_pilot_cases()

    assert Counter(case.reviewed_diagnosis.overall_status for case in cases) == {
        OverallStatus.FULLY_CORRECT: 16,
        OverallStatus.CORRECT_BUT_INEFFICIENT: 1,
        OverallStatus.INCOMPLETE: 4,
        OverallStatus.INCORRECT: 27,
        OverallStatus.INDETERMINATE: 2,
    }
    assert Counter(case.reviewed_decision.action for case in cases) == {
        TutorAction.WAIT: 17,
        TutorAction.ASK_STUDENT: 4,
        TutorAction.VERIFY_STEP: 2,
        TutorAction.LIGHT_HINT: 27,
    }
    assert Counter(case.category for case in cases)["correct_but_inefficient"] == 1


def test_ordinary_valid_methods_are_fully_correct_without_issues() -> None:
    cases = _by_id()
    for case_id in (
        "pilot.14.expand-fourth-power",
        "pilot.16.three-coins-enumeration",
        "pilot.17.derivative-from-definition",
    ):
        case = cases[case_id]
        diagnosis = case.reviewed_diagnosis
        assert diagnosis.overall_status is OverallStatus.FULLY_CORRECT
        assert diagnosis.first_issue is None
        assert all(item.status is StepStatus.VALID for item in diagnosis.step_assessments)
        assert diagnosis.reusable_prefix_end_step_id == diagnosis.step_assessments[-1].step_id
        assert case.reviewed_decision.action is TutorAction.WAIT


def test_divisor_listing_is_valid_but_inefficient_not_a_math_error() -> None:
    case = _by_id()["pilot.15.gcd-by-divisor-lists"]
    diagnosis = case.reviewed_diagnosis

    assert diagnosis.overall_status is OverallStatus.CORRECT_BUT_INEFFICIENT
    assert diagnosis.first_issue is None
    assert [item.status for item in diagnosis.step_assessments] == [
        StepStatus.VALID_BUT_INEFFICIENT,
        StepStatus.VALID_BUT_INEFFICIENT,
        StepStatus.VALID,
    ]
    assert all(not item.issue_codes for item in diagnosis.step_assessments)
    assert all(item.efficiency_note for item in diagnosis.step_assessments[:2])
    assert diagnosis.reusable_prefix_end_step_id == "step_8ec7d4c717cfcd6c"


def test_reviewed_root_issue_corrections_match_the_adjudication() -> None:
    cases = _by_id()

    transpose = cases["pilot.22.transpose-sign"].reviewed_diagnosis
    assert transpose.first_issue is not None
    assert transpose.first_issue.code is IssueCode.COMPUTATION_ARITHMETIC

    triangle = cases["pilot.36.pythagorean-on-any-triangle"].reviewed_diagnosis
    assert triangle.first_issue is not None
    assert triangle.first_issue.step_id == "step_f16d27cdb2f0894a"
    assert triangle.first_issue.code is IssueCode.CONDITION_MISSING
    assert triangle.step_assessments[0].status is StepStatus.UNSUPPORTED
    assert triangle.step_assessments[1].issue_codes == (IssueCode.THEOREM_MISUSED,)
    assert triangle.step_assessments[1].depends_on_step_ids == ("step_f16d27cdb2f0894a",)

    derivative = cases["pilot.40.zero-derivative-at-point"].reviewed_diagnosis
    assert derivative.first_issue is not None
    assert derivative.first_issue.step_id == "step_74d25c1c53494810"
    assert derivative.step_assessments[1].status is StepStatus.DEPENDENT_ON_PREVIOUS_ERROR
    assert derivative.reusable_prefix_end_step_id is None

    lucky = cases["pilot.44.gcd-lucky-claim"].reviewed_diagnosis
    assert lucky.first_issue is not None
    assert "35=5×7" in lucky.first_issue.explanation
    assert "64=2^6" in lucky.first_issue.explanation


def test_dependent_steps_form_explicit_backward_causal_links() -> None:
    cases = build_reviewed_diagnosis_pilot_cases()
    dependent = [
        (case, index, assessment)
        for case in cases
        for index, assessment in enumerate(case.reviewed_diagnosis.step_assessments)
        if assessment.status is StepStatus.DEPENDENT_ON_PREVIOUS_ERROR
    ]

    assert len(dependent) == 19
    for case, index, assessment in dependent:
        prior = {item.step_id: item for item in case.reviewed_diagnosis.step_assessments[:index]}
        assert assessment.depends_on_step_ids
        assert all(step_id in prior for step_id in assessment.depends_on_step_ids)
        assert all(
            prior[step_id].status
            in {
                StepStatus.INVALID,
                StepStatus.UNSUPPORTED,
                StepStatus.DEPENDENT_ON_PREVIOUS_ERROR,
            }
            for step_id in assessment.depends_on_step_ids
        )


def test_missing_context_cases_remain_indeterminate_after_review() -> None:
    cases = _by_id()
    for case_id in (
        "pilot.49.diagram-dependent-angle",
        "pilot.50.custom-operation-notation",
    ):
        case = cases[case_id]
        assert case.reviewed_diagnosis.overall_status is OverallStatus.INDETERMINATE
        assert case.ambiguity_notes
