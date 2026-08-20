from __future__ import annotations

import pytest

from math_feedback_ai.domain.models import CompletionGap, DiagnosisV1, Issue, StepAssessment
from math_feedback_ai.domain.taxonomy import (
    IssueCode,
    OverallStatus,
    RevealLevel,
    StepStatus,
    TutorAction,
    TutorMode,
)
from math_feedback_ai.policy.intervention import InterventionPolicy


def _valid_assessment() -> StepAssessment:
    return StepAssessment(step_id="s1", status=StepStatus.VALID, confidence=0.98)


def _correct(
    status: OverallStatus = OverallStatus.FULLY_CORRECT,
    *,
    confidence: float = 0.98,
) -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id="a1",
        overall_status=status,
        step_assessments=(_valid_assessment(),),
        earlier_reasoning_usable=True,
        confidence=confidence,
    )


def _incorrect(
    *,
    code: IssueCode = IssueCode.COMPUTATION_ARITHMETIC,
    confidence: float = 0.95,
) -> DiagnosisV1:
    assessment = StepAssessment(
        step_id="s1",
        status=StepStatus.INVALID,
        issue_codes=(code,),
        explanation="The operation does not preserve equality.",
        confidence=confidence,
    )
    return DiagnosisV1(
        attempt_id="a1",
        overall_status=OverallStatus.INCORRECT,
        step_assessments=(assessment,),
        first_issue=Issue(
            step_id="s1",
            code=code,
            explanation="The operation does not preserve equality.",
            confidence=confidence,
        ),
        earlier_reasoning_usable=False,
        confidence=confidence,
    )


def _incomplete(*, confidence: float = 0.92) -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id="a1",
        overall_status=OverallStatus.INCOMPLETE,
        step_assessments=(_valid_assessment(),),
        completion_gap=CompletionGap(
            after_step_id="s1",
            description="The valid prefix stops before isolating the variable.",
            confidence=confidence,
        ),
        reusable_prefix_end_step_id="s1",
        earlier_reasoning_usable=True,
        confidence=confidence,
    )


def _incomplete_without_localizable_gap(*, confidence: float = 0.92) -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id="a1",
        overall_status=OverallStatus.INCOMPLETE,
        step_assessments=(_valid_assessment(),),
        completion_gap=CompletionGap(
            after_step_id=None,
            description="More work is needed, but no exact transition can be targeted.",
            confidence=confidence,
        ),
        earlier_reasoning_usable=True,
        confidence=confidence,
    )


@pytest.mark.parametrize(
    ("status", "rationale"),
    [
        (OverallStatus.FULLY_CORRECT, "solution_fully_correct"),
        (
            OverallStatus.CORRECT_BUT_INEFFICIENT,
            "valid_solution_no_unsolicited_optimization",
        ),
    ],
)
def test_valid_work_does_not_receive_an_invented_hint(
    status: OverallStatus,
    rationale: str,
) -> None:
    decision = InterventionPolicy().decide(_correct(status))

    assert decision.action is TutorAction.WAIT
    assert decision.max_reveal_level is RevealLevel.NONE
    assert decision.target_step_id is None
    assert decision.rationale_code == rationale


def test_indeterminate_without_target_asks_student_for_information() -> None:
    diagnosis = DiagnosisV1(
        attempt_id="a1",
        overall_status=OverallStatus.INDETERMINATE,
        step_assessments=(
            StepAssessment(
                step_id="s1",
                status=StepStatus.AMBIGUOUS,
                issue_codes=(IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,),
                explanation="The notation is ambiguous.",
                confidence=0.2,
            ),
        ),
        earlier_reasoning_usable=False,
        confidence=0.2,
    )

    decision = InterventionPolicy().decide(diagnosis)

    assert decision.action is TutorAction.ASK_STUDENT
    assert decision.max_reveal_level is RevealLevel.NONE
    assert decision.rationale_code == "diagnosis_needs_more_information"


def test_low_confidence_localized_issue_requests_verification() -> None:
    decision = InterventionPolicy().decide(_incorrect(confidence=0.4))

    assert decision.action is TutorAction.VERIFY_STEP
    assert decision.max_reveal_level is RevealLevel.NONE
    assert decision.target_step_id == "s1"
    assert decision.rationale_code == "diagnosis_requires_verification"


def test_productive_incomplete_first_attempt_asks_student_to_continue() -> None:
    decision = InterventionPolicy().decide(_incomplete())

    assert decision.action is TutorAction.ASK_STUDENT
    assert decision.max_reveal_level is RevealLevel.NONE
    assert decision.target_step_id == "s1"
    assert decision.rationale_code == "productive_incomplete_prompt_for_next_step"


def test_incomplete_reasoning_escalates_more_slowly_than_an_error() -> None:
    decision = InterventionPolicy().decide(_incomplete(), prior_attempt_count=1)

    assert decision.action is TutorAction.LIGHT_HINT
    assert decision.max_reveal_level is RevealLevel.LIGHT_DIRECTION
    assert decision.target_step_id == "s1"


def test_incomplete_gap_without_step_never_selects_action_requiring_target() -> None:
    decision = InterventionPolicy().decide(
        _incomplete_without_localizable_gap(),
        prior_attempt_count=4,
        current_reveal_level=RevealLevel.CONCEPT,
        mode=TutorMode.GUIDED,
    )

    assert decision.action is TutorAction.EXPLAIN_CONCEPT
    assert decision.max_reveal_level is RevealLevel.CONCEPT
    assert decision.target_step_id is None


def test_first_localized_issue_gets_smallest_hint() -> None:
    decision = InterventionPolicy().decide(_incorrect())

    assert decision.action is TutorAction.LIGHT_HINT
    assert decision.max_reveal_level is RevealLevel.LIGHT_DIRECTION
    assert decision.target_step_id == "s1"
    assert decision.rationale_code == "localized_issue_first_hint"


@pytest.mark.parametrize(
    ("current", "expected_action", "expected_level"),
    [
        (RevealLevel.LIGHT_DIRECTION, TutorAction.TARGETED_HINT, RevealLevel.LOCATION),
        (RevealLevel.LOCATION, TutorAction.STRONG_HINT, RevealLevel.CONCEPT),
    ],
)
def test_prior_reveal_explicitly_escalates_one_level(
    current: RevealLevel,
    expected_action: TutorAction,
    expected_level: RevealLevel,
) -> None:
    decision = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=int(current),
        current_reveal_level=current,
    )

    assert decision.action is expected_action
    assert decision.max_reveal_level is expected_level


def test_conceptual_issue_uses_explain_concept_at_level_three() -> None:
    decision = InterventionPolicy().decide(
        _incorrect(code=IssueCode.THEOREM_MISUSED),
        prior_attempt_count=2,
        current_reveal_level=RevealLevel.LOCATION,
    )

    assert decision.action is TutorAction.EXPLAIN_CONCEPT
    assert decision.max_reveal_level is RevealLevel.CONCEPT


def test_hint_only_ceiling_cannot_be_overridden_by_student_request() -> None:
    decision = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=20,
        current_reveal_level=RevealLevel.SUBSTANTIAL_SCAFFOLD,
        mode=TutorMode.HINT_ONLY,
        full_solution_authorized=True,
        requested_reveal_level=RevealLevel.FULL_SOLUTION,
    )

    assert decision.max_reveal_level is RevealLevel.CONCEPT
    assert decision.action is TutorAction.STRONG_HINT
    assert not decision.full_solution_authorized


def test_student_can_request_less_revealing_response() -> None:
    decision = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=5,
        current_reveal_level=RevealLevel.CONCEPT,
        requested_reveal_level=RevealLevel.NONE,
    )

    assert decision.action is TutorAction.ASK_STUDENT
    assert decision.max_reveal_level is RevealLevel.NONE
    assert decision.rationale_code == "localized_issue_student_requested_no_hint"


def test_guided_mode_stops_at_partial_solution() -> None:
    decision = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=20,
        current_reveal_level=RevealLevel.CONCEPT,
        mode=TutorMode.GUIDED,
        full_solution_authorized=True,
    )

    assert decision.max_reveal_level is RevealLevel.SUBSTANTIAL_SCAFFOLD
    assert decision.action is TutorAction.SHOW_PARTIAL_SOLUTION
    assert not decision.full_solution_authorized


def test_full_solution_mode_still_requires_explicit_authorization() -> None:
    unauthorized = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=20,
        current_reveal_level=RevealLevel.SUBSTANTIAL_SCAFFOLD,
        mode=TutorMode.FULL_SOLUTION_ALLOWED,
        full_solution_authorized=False,
    )
    authorized = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=20,
        current_reveal_level=RevealLevel.SUBSTANTIAL_SCAFFOLD,
        mode=TutorMode.FULL_SOLUTION_ALLOWED,
        full_solution_authorized=True,
    )

    assert unauthorized.action is TutorAction.SHOW_PARTIAL_SOLUTION
    assert unauthorized.max_reveal_level is RevealLevel.SUBSTANTIAL_SCAFFOLD
    assert not unauthorized.full_solution_authorized
    assert authorized.action is TutorAction.SHOW_FULL_SOLUTION
    assert authorized.max_reveal_level is RevealLevel.FULL_SOLUTION
    assert authorized.full_solution_authorized


def test_prior_action_is_respected_when_reveal_counter_is_stale() -> None:
    decision = InterventionPolicy().decide(
        _incorrect(),
        prior_attempt_count=1,
        prior_action=TutorAction.TARGETED_HINT,
        current_reveal_level=RevealLevel.NONE,
    )

    assert decision.max_reveal_level is RevealLevel.CONCEPT
    assert decision.action is TutorAction.STRONG_HINT


def test_invalid_policy_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="prior_attempt_count"):
        InterventionPolicy().decide(_incorrect(), prior_attempt_count=-1)
    with pytest.raises(ValueError, match="confidence_threshold"):
        InterventionPolicy(confidence_threshold=1.1)
