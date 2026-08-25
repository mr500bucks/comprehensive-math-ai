from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from math_feedback_ai.domain import (
    CompletionGap,
    DiagnosisV1,
    ExampleProvenance,
    HintPlan,
    Issue,
    IssueCode,
    LeakageCheckResult,
    LeakageSeverity,
    LeakageViolation,
    OverallStatus,
    Problem,
    ReferenceSolution,
    RevealLevel,
    SolutionStep,
    StepAssessment,
    StepParseResult,
    StepParseStrategy,
    StepStatus,
    StudentAttempt,
    TutorAction,
    TutorDecision,
    TutoringExampleV1,
    TutorResponse,
    TutorResult,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def make_problem() -> Problem:
    return Problem(
        problem_id="problem.linear-1",
        statement="Solve x + 1 = 2.",
        reference_solutions=(
            ReferenceSolution(
                reference_id="reference.subtract",
                text="Subtract 1 from both sides to get x = 1.",
                method_label="inverse operation",
            ),
        ),
    )


def make_attempt() -> StudentAttempt:
    raw = "x = 1\nx + 1 = 2"
    return StudentAttempt(
        attempt_id="attempt.linear-1",
        problem_id="problem.linear-1",
        raw_text=raw,
        steps=(
            SolutionStep(
                step_id="step.solve",
                position=0,
                text="x = 1",
                start_offset=0,
                end_offset=5,
            ),
            SolutionStep(
                step_id="step.check",
                position=1,
                text="x + 1 = 2",
                start_offset=6,
                end_offset=len(raw),
            ),
        ),
    )


def valid_assessments() -> tuple[StepAssessment, ...]:
    return tuple(
        StepAssessment(
            step_id=step_id,
            status=StepStatus.VALID,
            confidence=0.98,
        )
        for step_id in ("step.solve", "step.check")
    )


def fully_correct_diagnosis() -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id="attempt.linear-1",
        overall_status=OverallStatus.FULLY_CORRECT,
        step_assessments=valid_assessments(),
        reusable_prefix_end_step_id="step.check",
        earlier_reasoning_usable=True,
        confidence=0.98,
        confidence_reasons=("Each equality is valid.",),
    )


def incorrect_diagnosis() -> DiagnosisV1:
    assessments = (
        StepAssessment(
            step_id="step.solve",
            status=StepStatus.INVALID,
            issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
            explanation="The subtraction is incorrect.",
            confidence=0.95,
        ),
        StepAssessment(
            step_id="step.check",
            status=StepStatus.AMBIGUOUS,
            issue_codes=(IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,),
            explanation="This depends on the previous invalid value.",
            confidence=0.7,
        ),
    )
    return DiagnosisV1(
        attempt_id="attempt.linear-1",
        overall_status=OverallStatus.INCORRECT,
        step_assessments=assessments,
        first_issue=Issue(
            step_id="step.solve",
            code=IssueCode.COMPUTATION_ARITHMETIC,
            explanation="The subtraction is incorrect.",
            evidence="Substitution does not satisfy the original equation.",
            concept_tags=("inverse operations",),
            confidence=0.95,
        ),
        earlier_reasoning_usable=False,
        confidence=0.95,
    )


def test_problem_preserves_plural_non_exhaustive_references() -> None:
    problem = make_problem()

    assert isinstance(problem.reference_solutions, tuple)
    assert problem.reference_solutions[0].reference_id == "reference.subtract"


@pytest.mark.parametrize("statement", ["", " ", "\n\t"])
def test_problem_rejects_blank_statement(statement: str) -> None:
    with pytest.raises(ValidationError, match="problem statement|at least 1 character"):
        Problem(problem_id="problem.blank", statement=statement)


def test_problem_rejects_duplicate_reference_ids() -> None:
    duplicate = ReferenceSolution(reference_id="same", text="One solution")
    with pytest.raises(ValidationError, match="reference solution IDs"):
        Problem(
            problem_id="problem.duplicate",
            statement="A problem",
            reference_solutions=(duplicate, duplicate),
        )


def test_student_attempt_requires_exact_source_spans() -> None:
    with pytest.raises(ValidationError, match="exact raw-text slice"):
        StudentAttempt(
            attempt_id="attempt.bad-span",
            problem_id="problem.linear-1",
            raw_text="x = 1",
            steps=(
                SolutionStep(
                    step_id="step.bad",
                    position=0,
                    text="x=1",
                    start_offset=0,
                    end_offset=5,
                ),
            ),
        )


def test_student_attempt_never_allows_non_whitespace_text_outside_steps() -> None:
    with pytest.raises(ValidationError, match="outside the recorded step spans"):
        StudentAttempt(
            attempt_id="attempt.discarded",
            problem_id="problem.linear-1",
            raw_text="unrecorded x = 1",
            steps=(
                SolutionStep(
                    step_id="step.only",
                    position=0,
                    text="x = 1",
                    start_offset=11,
                    end_offset=16,
                ),
            ),
        )


def test_student_attempt_rejects_duplicate_ids_and_noncontiguous_positions() -> None:
    raw = "a\nb"
    duplicate_id_steps = (
        SolutionStep(step_id="step.same", position=0, text="a", start_offset=0, end_offset=1),
        SolutionStep(step_id="step.same", position=1, text="b", start_offset=2, end_offset=3),
    )
    with pytest.raises(ValidationError, match="IDs must be unique"):
        StudentAttempt(
            attempt_id="attempt.duplicate",
            problem_id="problem.test",
            raw_text=raw,
            steps=duplicate_id_steps,
        )

    noncontiguous_steps = (
        SolutionStep(step_id="step.a", position=0, text="a", start_offset=0, end_offset=1),
        SolutionStep(step_id="step.b", position=2, text="b", start_offset=2, end_offset=3),
    )
    with pytest.raises(ValidationError, match="contiguous and zero-based"):
        StudentAttempt(
            attempt_id="attempt.positions",
            problem_id="problem.test",
            raw_text=raw,
            steps=noncontiguous_steps,
        )


def test_ambiguous_parse_requires_a_reason_and_marks_steps() -> None:
    step = SolutionStep(
        step_id="step.ambiguous",
        position=0,
        text="Maybe this is one or two steps.",
        start_offset=0,
        end_offset=31,
        segmentation_ambiguous=True,
    )
    with pytest.raises(ValidationError, match="ambiguous=True"):
        StepParseResult(
            raw_text=step.text,
            steps=(step,),
            strategy=StepParseStrategy.SINGLE_CHUNK,
        )
    with pytest.raises(ValidationError, match="explain why"):
        StepParseResult(
            raw_text=step.text,
            steps=(step,),
            strategy=StepParseStrategy.SINGLE_CHUNK,
            ambiguous=True,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("inf"), float("nan")])
def test_confidence_is_bounded_and_finite(confidence: float) -> None:
    with pytest.raises(ValidationError):
        StepAssessment(
            step_id="step.one",
            status=StepStatus.VALID,
            confidence=confidence,
        )


def test_invalid_and_unsupported_assessments_require_evidence_structure() -> None:
    with pytest.raises(ValidationError, match="issue code"):
        StepAssessment(
            step_id="step.one",
            status=StepStatus.INVALID,
            explanation="It is invalid.",
            confidence=0.8,
        )
    with pytest.raises(ValidationError, match="explanation"):
        StepAssessment(
            step_id="step.one",
            status=StepStatus.UNSUPPORTED,
            issue_codes=(IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,),
            confidence=0.8,
        )


def test_valid_step_cannot_carry_a_mathematical_error_code() -> None:
    with pytest.raises(ValidationError, match="valid steps"):
        StepAssessment(
            step_id="step.one",
            status=StepStatus.VALID,
            issue_codes=(IssueCode.COMPUTATION_ALGEBRAIC,),
            confidence=0.8,
        )


def test_fully_correct_diagnosis_has_no_issue_or_completion_gap() -> None:
    diagnosis = fully_correct_diagnosis()

    assert diagnosis.first_issue is None
    assert diagnosis.completion_gap is None
    assert diagnosis.earlier_reasoning_usable

    with pytest.raises(ValidationError, match="fully correct"):
        DiagnosisV1(
            attempt_id=diagnosis.attempt_id,
            overall_status=OverallStatus.FULLY_CORRECT,
            step_assessments=diagnosis.step_assessments,
            completion_gap=CompletionGap(
                after_step_id="step.check",
                description="Continue.",
                confidence=0.5,
            ),
            earlier_reasoning_usable=True,
            confidence=0.9,
        )


def test_correct_but_inefficient_diagnosis_cannot_hide_invalid_reasoning() -> None:
    invalid = StepAssessment(
        step_id="step.solve",
        status=StepStatus.INVALID,
        issue_codes=(IssueCode.COMPUTATION_ALGEBRAIC,),
        explanation="The transformation is invalid.",
        confidence=0.9,
    )

    with pytest.raises(ValidationError, match="must remain mathematically valid"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.CORRECT_BUT_INEFFICIENT,
            step_assessments=(invalid,),
            earlier_reasoning_usable=False,
            confidence=0.9,
        )

    with pytest.raises(ValidationError, match="must be marked usable"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.CORRECT_BUT_INEFFICIENT,
            step_assessments=(
                StepAssessment(
                    step_id="step.solve",
                    status=StepStatus.VALID,
                    issue_codes=(IssueCode.RELEVANCE_IRRELEVANT,),
                    explanation="This valid route is unnecessarily long.",
                    confidence=0.9,
                ),
            ),
            earlier_reasoning_usable=False,
            confidence=0.9,
        )


def test_schema_1_1_represents_valid_inefficiency_without_a_fake_issue() -> None:
    assessment = StepAssessment(
        step_id="step.list",
        status=StepStatus.VALID_BUT_INEFFICIENT,
        efficiency_note="Correct, but Euclid's algorithm is substantially more direct.",
        confidence=0.95,
    )
    diagnosis = DiagnosisV1(
        schema_version="1.1",
        attempt_id="attempt.gcd",
        overall_status=OverallStatus.CORRECT_BUT_INEFFICIENT,
        step_assessments=(assessment,),
        reusable_prefix_end_step_id="step.list",
        earlier_reasoning_usable=True,
        confidence=0.95,
    )

    assert diagnosis.first_issue is None
    assert not assessment.issue_codes
    assert "efficiency_note" in assessment.model_dump(mode="json")

    with pytest.raises(ValidationError, match="efficiency note"):
        StepAssessment(
            step_id="step.list",
            status=StepStatus.VALID_BUT_INEFFICIENT,
            confidence=0.95,
        )


def test_dependent_step_names_an_earlier_root_error_and_cannot_be_first_issue() -> None:
    root = StepAssessment(
        step_id="step.discriminant",
        status=StepStatus.INVALID,
        issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
        explanation="The discriminant arithmetic is wrong.",
        confidence=0.95,
    )
    dependent = StepAssessment(
        step_id="step.formula",
        status=StepStatus.DEPENDENT_ON_PREVIOUS_ERROR,
        explanation="The substitution is coherent only with the wrong discriminant.",
        depends_on_step_ids=("step.discriminant",),
        confidence=0.95,
    )
    diagnosis = DiagnosisV1(
        schema_version="1.1",
        attempt_id="attempt.quadratic",
        overall_status=OverallStatus.INCORRECT,
        step_assessments=(root, dependent),
        first_issue=Issue(
            step_id="step.discriminant",
            code=IssueCode.COMPUTATION_ARITHMETIC,
            explanation="The discriminant arithmetic is wrong.",
            confidence=0.95,
        ),
        earlier_reasoning_usable=False,
        confidence=0.95,
    )

    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step.discriminant"

    with pytest.raises(ValidationError, match="earlier assessed"):
        DiagnosisV1(
            schema_version="1.1",
            attempt_id="attempt.quadratic",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=(dependent, root),
            first_issue=Issue(
                step_id="step.discriminant",
                code=IssueCode.COMPUTATION_ARITHMETIC,
                explanation="The discriminant arithmetic is wrong.",
                confidence=0.95,
            ),
            earlier_reasoning_usable=False,
            confidence=0.95,
        )


def test_incorrect_diagnosis_requires_a_localized_issue() -> None:
    with pytest.raises(ValidationError, match="localize a first issue"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=valid_assessments(),
            earlier_reasoning_usable=True,
            confidence=0.2,
        )


def test_first_issue_must_match_its_assessment_code_and_step() -> None:
    issue = Issue(
        step_id="step.solve",
        code=IssueCode.COMPUTATION_ARITHMETIC,
        explanation="Arithmetic issue.",
        confidence=0.9,
    )
    with pytest.raises(ValidationError, match="must appear"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=(
                StepAssessment(
                    step_id="step.solve",
                    status=StepStatus.INVALID,
                    issue_codes=(IssueCode.COMPUTATION_ALGEBRAIC,),
                    explanation="Algebra issue.",
                    confidence=0.9,
                ),
            ),
            first_issue=issue,
            earlier_reasoning_usable=False,
            confidence=0.9,
        )

    with pytest.raises(ValidationError, match="unassessed step IDs"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=(
                StepAssessment(
                    step_id="step.other",
                    status=StepStatus.INVALID,
                    issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
                    explanation="Arithmetic issue.",
                    confidence=0.9,
                ),
            ),
            first_issue=issue,
            earlier_reasoning_usable=False,
            confidence=0.9,
        )


def test_first_issue_must_be_the_earliest_questionable_assessment() -> None:
    assessments = (
        StepAssessment(
            step_id="step.solve",
            status=StepStatus.UNSUPPORTED,
            issue_codes=(IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,),
            explanation="The value is asserted without support.",
            confidence=0.8,
        ),
        StepAssessment(
            step_id="step.check",
            status=StepStatus.INVALID,
            issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
            explanation="The arithmetic is wrong.",
            confidence=0.9,
        ),
    )
    with pytest.raises(ValidationError, match="earliest questionable"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=assessments,
            first_issue=Issue(
                step_id="step.check",
                code=IssueCode.COMPUTATION_ARITHMETIC,
                explanation="The arithmetic is wrong.",
                confidence=0.9,
            ),
            earlier_reasoning_usable=False,
            confidence=0.9,
        )


def test_irrelevant_valid_preamble_does_not_preempt_first_mathematical_error() -> None:
    diagnosis = DiagnosisV1(
        attempt_id="attempt.linear-1",
        overall_status=OverallStatus.INCORRECT,
        step_assessments=(
            StepAssessment(
                step_id="step.preamble",
                status=StepStatus.VALID,
                issue_codes=(IssueCode.RELEVANCE_IRRELEVANT,),
                explanation="This valid observation is irrelevant.",
                confidence=0.9,
            ),
            StepAssessment(
                step_id="step.error",
                status=StepStatus.INVALID,
                issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
                explanation="The arithmetic is wrong.",
                confidence=0.9,
            ),
        ),
        first_issue=Issue(
            step_id="step.error",
            code=IssueCode.COMPUTATION_ARITHMETIC,
            explanation="The arithmetic is wrong.",
            confidence=0.9,
        ),
        earlier_reasoning_usable=False,
        confidence=0.9,
    )

    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step.error"

    with pytest.raises(ValidationError, match="mathematical issue"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCORRECT,
            step_assessments=diagnosis.step_assessments,
            first_issue=Issue(
                step_id="step.preamble",
                code=IssueCode.RELEVANCE_IRRELEVANT,
                explanation="This valid observation is irrelevant.",
                confidence=0.9,
            ),
            earlier_reasoning_usable=False,
            confidence=0.9,
        )


def test_reusable_prefix_requires_reasoning_to_be_marked_usable() -> None:
    with pytest.raises(ValidationError, match="reusable prefix"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INDETERMINATE,
            step_assessments=valid_assessments(),
            reusable_prefix_end_step_id="step.solve",
            earlier_reasoning_usable=False,
            confidence=0.3,
        )


def test_incomplete_is_a_valid_prefix_with_a_separate_gap() -> None:
    diagnosis = DiagnosisV1(
        attempt_id="attempt.linear-1",
        overall_status=OverallStatus.INCOMPLETE,
        step_assessments=valid_assessments(),
        completion_gap=CompletionGap(
            after_step_id="step.check",
            description="The result still needs to be stated.",
            confidence=0.9,
        ),
        reusable_prefix_end_step_id="step.check",
        earlier_reasoning_usable=True,
        confidence=0.9,
    )

    assert diagnosis.first_issue is None
    assert diagnosis.completion_gap is not None


def test_incomplete_rejects_missing_gap_or_invalid_prefix() -> None:
    with pytest.raises(ValidationError, match="requires a completion gap"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCOMPLETE,
            step_assessments=valid_assessments(),
            earlier_reasoning_usable=True,
            confidence=0.9,
        )

    bad_assessment = StepAssessment(
        step_id="step.solve",
        status=StepStatus.INVALID,
        issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
        explanation="Bad arithmetic.",
        confidence=0.9,
    )
    with pytest.raises(ValidationError, match="valid reasoning prefix"):
        DiagnosisV1(
            attempt_id="attempt.linear-1",
            overall_status=OverallStatus.INCOMPLETE,
            step_assessments=(bad_assessment,),
            completion_gap=CompletionGap(
                after_step_id="step.solve",
                description="Stopped.",
                confidence=0.8,
            ),
            earlier_reasoning_usable=True,
            confidence=0.8,
        )


def test_safe_indeterminate_factory_assesses_every_step() -> None:
    attempt = make_attempt()

    diagnosis = DiagnosisV1.indeterminate(attempt, "Model output was invalid.")

    assert diagnosis.overall_status is OverallStatus.INDETERMINATE
    assert diagnosis.confidence == 0.0
    assert diagnosis.first_issue is None
    assert {item.step_id for item in diagnosis.step_assessments} == {
        step.step_id for step in attempt.steps
    }
    assert all(item.status is StepStatus.AMBIGUOUS for item in diagnosis.step_assessments)


@pytest.mark.parametrize(
    ("action", "level", "target", "authorized"),
    [
        (TutorAction.WAIT, RevealLevel.LIGHT_DIRECTION, None, False),
        (TutorAction.LIGHT_HINT, RevealLevel.LOCATION, None, False),
        (TutorAction.TARGETED_HINT, RevealLevel.LOCATION, None, False),
        (TutorAction.SHOW_FULL_SOLUTION, RevealLevel.FULL_SOLUTION, None, False),
    ],
)
def test_tutor_decision_rejects_action_reveal_or_authorization_mismatch(
    action: TutorAction,
    level: RevealLevel,
    target: str | None,
    authorized: bool,
) -> None:
    with pytest.raises(ValidationError):
        TutorDecision(
            action=action,
            max_reveal_level=level,
            target_step_id=target,
            rationale_code="test.invalid",
            full_solution_authorized=authorized,
        )


def test_hint_plan_minimizes_reference_context_below_level_four() -> None:
    decision = TutorDecision(
        action=TutorAction.LIGHT_HINT,
        max_reveal_level=RevealLevel.LIGHT_DIRECTION,
        target_step_id="step.solve",
        rationale_code="localized.first_hint",
    )
    with pytest.raises(ValidationError, match="reference text is forbidden"):
        HintPlan(
            problem_id="problem.linear-1",
            problem_statement="Solve x + 1 = 2.",
            attempt_id="attempt.linear-1",
            context_steps=make_attempt().steps,
            decision=decision,
            diagnostic_summary="Check the inverse operation.",
            reference_excerpt="Subtract 1 from both sides.",
        )


def test_hint_plan_rejects_content_that_is_both_allowed_and_forbidden() -> None:
    decision = TutorDecision(
        action=TutorAction.LIGHT_HINT,
        max_reveal_level=RevealLevel.LIGHT_DIRECTION,
        target_step_id="step.solve",
        rationale_code="localized.first_hint",
    )
    with pytest.raises(ValidationError, match="both allowed and forbidden"):
        HintPlan(
            problem_id="problem.linear-1",
            problem_statement="Solve x + 1 = 2.",
            attempt_id="attempt.linear-1",
            context_steps=make_attempt().steps,
            decision=decision,
            diagnostic_summary="Check the inverse operation.",
            allowed_content=("Mention inverse operations.",),
            forbidden_content=("Mention inverse operations.",),
        )


def test_leakage_result_passes_with_warning_but_not_hard_violation() -> None:
    warning = LeakageViolation(
        code="excessive_detail",
        message="The response is longer than the soft limit.",
        severity=LeakageSeverity.WARNING,
    )
    result = LeakageCheckResult(passed=True, violations=(warning,))
    assert result.severity is LeakageSeverity.WARNING

    with pytest.raises(ValidationError, match="no HARD"):
        LeakageCheckResult(
            passed=True,
            violations=(
                LeakageViolation(
                    code="final_answer_leakage",
                    message="Final answer leaked.",
                    severity=LeakageSeverity.HARD,
                ),
            ),
        )


def test_tutor_result_rejects_dangling_or_incomplete_diagnosis_references() -> None:
    attempt = make_attempt()
    decision = TutorDecision(
        action=TutorAction.WAIT,
        max_reveal_level=RevealLevel.NONE,
        rationale_code="solution.fully_correct",
    )
    response = TutorResponse(
        action=TutorAction.WAIT,
        reveal_level=RevealLevel.NONE,
        message="Your reasoning is complete.",
    )
    result = TutorResult(
        problem=make_problem(),
        attempt=attempt,
        diagnosis=fully_correct_diagnosis(),
        decision=decision,
        response=response,
    )
    assert result.response.message == "Your reasoning is complete."

    partial_diagnosis = DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.FULLY_CORRECT,
        step_assessments=(valid_assessments()[0],),
        reusable_prefix_end_step_id="step.solve",
        earlier_reasoning_usable=True,
        confidence=0.9,
    )
    with pytest.raises(ValidationError, match="assess every and only"):
        TutorResult(
            problem=make_problem(),
            attempt=attempt,
            diagnosis=partial_diagnosis,
            decision=decision,
            response=response,
        )

    reversed_diagnosis = DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.FULLY_CORRECT,
        step_assessments=tuple(reversed(valid_assessments())),
        reusable_prefix_end_step_id="step.check",
        earlier_reasoning_usable=True,
        confidence=0.9,
    )
    with pytest.raises(ValidationError, match="student step order"):
        TutorResult(
            problem=make_problem(),
            attempt=attempt,
            diagnosis=reversed_diagnosis,
            decision=decision,
            response=response,
        )


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    problem = make_problem()
    with pytest.raises(ValidationError, match="frozen"):
        problem.statement = "Changed"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs"):
        Problem(
            problem_id="problem.extra",
            statement="A statement",
            hidden_answer="42",  # type: ignore[call-arg]
        )


def test_canonical_example_fixture_matches_typed_contract() -> None:
    fixture_path = REPOSITORY_ROOT / "data" / "examples" / "tutoring_example.v1.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    example = TutoringExampleV1.model_validate(payload)

    assert example.schema_version == "1.0"
    assert example.provenance.synthetic is True
    assert example.problem.reference_solutions


def test_committed_json_schema_matches_the_typed_contract() -> None:
    schema_path = REPOSITORY_ROOT / TutoringExampleV1.SCHEMA_PATH
    committed_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert committed_schema == TutoringExampleV1.canonical_json_schema()


def test_canonical_example_cross_validates_problem_attempt_and_step_refs() -> None:
    attempt = make_attempt()
    with pytest.raises(ValidationError, match="problem_id must match"):
        TutoringExampleV1(
            example_id="example.bad-problem",
            provenance=ExampleProvenance(source="unit test", synthetic=True),
            problem=Problem(problem_id="other", statement="Other"),
            student_attempt=attempt,
            gold_diagnosis=fully_correct_diagnosis(),
        )
