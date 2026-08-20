from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import assert_type

import pytest

from math_feedback_ai.domain import (
    CompletionGap,
    DiagnosisV1,
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
    StepAssessment,
    StepStatus,
    StudentAttempt,
    TutorAction,
    TutorMode,
)
from math_feedback_ai.hints.generator import HintGenerationResult, HintGenerator
from math_feedback_ai.hints.leakage import HintSafetyContext
from math_feedback_ai.model.client import GenerationResult, ModelClient, TokenUsage
from math_feedback_ai.model.fake import FakeModelClient
from math_feedback_ai.orchestration.tutor import (
    TutorContext,
    TutorOrchestrator,
    TutorSafetyError,
)
from math_feedback_ai.parsing.steps import parse_student_attempt

DiagnosisBuilder = Callable[[StudentAttempt], DiagnosisV1]


@dataclass(slots=True)
class RecordingDiagnoser:
    builder: DiagnosisBuilder
    calls: int = 0
    references_seen: list[tuple[ReferenceSolution, ...]] = field(default_factory=list)

    def diagnose(
        self,
        problem: Problem,
        attempt: StudentAttempt,
        reference_solutions: Sequence[ReferenceSolution] | None = None,
    ) -> DiagnosisV1:
        assert problem.problem_id == attempt.problem_id
        self.calls += 1
        self.references_seen.append(tuple(reference_solutions or ()))
        return self.builder(attempt)


@dataclass(slots=True)
class NeverHintProducer:
    calls: int = 0

    def generate(
        self,
        plan: HintPlan,
        *,
        safety_context: HintSafetyContext | None = None,
    ) -> HintGenerationResult:
        self.calls += 1
        raise AssertionError(f"hint generation was not expected for {plan.decision.action}")


@dataclass(slots=True)
class RecordingHintProducer:
    inner: HintGenerator
    calls: int = 0
    last_result: HintGenerationResult | None = None

    def generate(
        self,
        plan: HintPlan,
        *,
        safety_context: HintSafetyContext | None = None,
    ) -> HintGenerationResult:
        self.calls += 1
        result = self.inner.generate(plan, safety_context=safety_context)
        self.last_result = result
        return result


@dataclass(frozen=True, slots=True)
class StaticHintProducer:
    text: str
    claimed_safe: bool = True

    def generate(
        self,
        plan: HintPlan,
        *,
        safety_context: HintSafetyContext | None = None,
    ) -> HintGenerationResult:
        del plan, safety_context
        check = (
            LeakageCheckResult(passed=True)
            if self.claimed_safe
            else _hard_failure("producer_rejected")
        )
        return HintGenerationResult(text=self.text, safety_check=check, model_call_count=0)


class AlwaysRejectChecker:
    def check(
        self,
        candidate: str,
        context: HintSafetyContext,
    ) -> LeakageCheckResult:
        del candidate, context
        return _hard_failure("forced_rejection")


def _hard_failure(code: str) -> LeakageCheckResult:
    return LeakageCheckResult(
        passed=False,
        violations=(
            LeakageViolation(
                code=code,
                message="The test safety gate rejected this response.",
                severity=LeakageSeverity.HARD,
            ),
        ),
    )


def _problem(*, references: tuple[ReferenceSolution, ...] = ()) -> Problem:
    return Problem(
        problem_id="p1",
        statement="Solve the equation while showing your reasoning.",
        reference_solutions=references,
    )


def _correct_diagnosis(attempt: StudentAttempt) -> DiagnosisV1:
    return DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.FULLY_CORRECT,
        step_assessments=tuple(
            StepAssessment(step_id=step.step_id, status=StepStatus.VALID, confidence=0.99)
            for step in attempt.steps
        ),
        earlier_reasoning_usable=True,
        confidence=0.99,
    )


def _incomplete_diagnosis(attempt: StudentAttempt) -> DiagnosisV1:
    last_step_id = attempt.steps[-1].step_id
    return DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.INCOMPLETE,
        step_assessments=tuple(
            StepAssessment(step_id=step.step_id, status=StepStatus.VALID, confidence=0.95)
            for step in attempt.steps
        ),
        completion_gap=CompletionGap(
            after_step_id=last_step_id,
            description="The valid prefix stops before the requested result is established.",
            confidence=0.95,
        ),
        reusable_prefix_end_step_id=last_step_id,
        earlier_reasoning_usable=True,
        confidence=0.95,
    )


def _incorrect_diagnosis(
    attempt: StudentAttempt,
    *,
    issue_position: int = 1,
) -> DiagnosisV1:
    target_position = min(issue_position, len(attempt.steps) - 1)
    target = attempt.steps[target_position]
    assessments: list[StepAssessment] = []
    for step in attempt.steps:
        if step.position < target_position:
            assessments.append(
                StepAssessment(step_id=step.step_id, status=StepStatus.VALID, confidence=0.98)
            )
        else:
            assessments.append(
                StepAssessment(
                    step_id=step.step_id,
                    status=StepStatus.INVALID,
                    issue_codes=(IssueCode.COMPUTATION_ARITHMETIC,),
                    explanation="The arithmetic transition is not valid.",
                    confidence=0.96,
                )
            )
    reusable_prefix = attempt.steps[target_position - 1].step_id if target_position > 0 else None
    return DiagnosisV1(
        attempt_id=attempt.attempt_id,
        overall_status=OverallStatus.INCORRECT,
        step_assessments=tuple(assessments),
        first_issue=Issue(
            step_id=target.step_id,
            code=IssueCode.COMPUTATION_ARITHMETIC,
            explanation="The first arithmetic transition is not valid.",
            concept_tags=("maintaining equality",),
            confidence=0.96,
        ),
        reusable_prefix_end_step_id=reusable_prefix,
        earlier_reasoning_usable=target_position > 0,
        confidence=0.96,
    )


def test_fully_correct_attempt_takes_direct_wait_path() -> None:
    diagnoser = RecordingDiagnoser(_correct_diagnosis)
    hints = NeverHintProducer()
    orchestrator = TutorOrchestrator(diagnosis_service=diagnoser, hint_generator=hints)

    result = orchestrator.tutor(_problem(), "2 + 2 = 4", attempt_id="a-wait")

    assert result.decision.action is TutorAction.WAIT
    assert result.hint_plan is None
    assert result.response.action is TutorAction.WAIT
    assert result.response.reveal_level is RevealLevel.NONE
    assert not result.response.requires_student_response
    assert not result.response.safe_fallback_used
    assert result.leakage_check is not None and result.leakage_check.passed
    assert hints.calls == 0
    assert diagnoser.calls == 1


def test_productive_incomplete_attempt_takes_direct_ask_path() -> None:
    diagnoser = RecordingDiagnoser(_incomplete_diagnosis)
    hints = NeverHintProducer()
    orchestrator = TutorOrchestrator(diagnosis_service=diagnoser, hint_generator=hints)

    result = orchestrator.tutor(_problem(), "2x + 4 = 18", attempt_id="a-ask")

    assert result.decision.action is TutorAction.ASK_STUDENT
    assert result.hint_plan is None
    assert result.response.message == "What would you try next, and why?"
    assert result.response.requires_student_response
    assert hints.calls == 0


def test_normal_hint_path_keeps_private_context_out_of_prompt_and_public_response() -> None:
    secret_reference = "SECRET_REFERENCE: subtract four, divide by two, and report x=7."
    reference = ReferenceSolution(reference_id="r1", text=secret_reference)
    diagnoser = RecordingDiagnoser(_incorrect_diagnosis)
    model = FakeModelClient(
        text_responses=["What relationship should stay unchanged as you move into the second line?"]
    )
    hints = RecordingHintProducer(HintGenerator(model))
    orchestrator = TutorOrchestrator(diagnosis_service=diagnoser, hint_generator=hints)
    downstream = "Therefore I divide both sides by two and conclude x = 11"
    raw_work = f"2x + 4 = 18\n2x = 22\n{downstream}"

    result = orchestrator.tutor(
        _problem(references=(reference,)),
        raw_work,
        context=TutorContext(final_answers=("x=7",)),
        attempt_id="a-normal",
    )

    assert result.decision.action is TutorAction.LIGHT_HINT
    assert result.hint_plan is not None
    assert result.hint_plan.reference_excerpt is None
    assert result.hint_plan.forbidden_content == ()
    assert len(result.hint_plan.context_steps) == 2
    assert hints.calls == 1
    assert model.text_call_count == 1
    prompt = model.text_requests[0].prompt
    assert secret_reference not in prompt
    assert downstream not in prompt
    assert "x=7" not in prompt
    public_payload = json.dumps(result.response.model_dump(mode="json"))
    assert secret_reference not in public_payload
    assert downstream not in public_payload
    assert "x=7" not in public_payload
    assert diagnoser.references_seen == [(reference,)]
    assert result.leakage_check is not None and result.leakage_check.passed


def test_downstream_leak_triggers_one_rewrite_and_preserves_usage_trace() -> None:
    diagnoser = RecordingDiagnoser(_incorrect_diagnosis)
    downstream = "Therefore I divide both sides by two and conclude x = 11"
    first = GenerationResult(
        content=downstream,
        usage=TokenUsage(input_tokens=20, output_tokens=12, total_tokens=32),
        latency_ms=4.0,
    )
    second = GenerationResult(
        content="Which operation in the second line deserves a closer check?",
        usage=TokenUsage(input_tokens=24, output_tokens=9, total_tokens=33),
        latency_ms=3.0,
    )
    model = FakeModelClient(text_responses=[first, second])
    hints = RecordingHintProducer(HintGenerator(model))
    orchestrator = TutorOrchestrator(diagnosis_service=diagnoser, hint_generator=hints)

    result = orchestrator.tutor(
        _problem(),
        f"2x + 4 = 18\n2x = 22\n{downstream}",
        attempt_id="a-rewrite",
    )

    assert model.text_call_count == 2
    assert hints.last_result is not None
    assert hints.last_result.regenerated
    assert hints.last_result.model_call_count == 2
    assert sum(item.usage.input_tokens or 0 for item in hints.last_result.generations) == 44
    assert sum(item.usage.output_tokens or 0 for item in hints.last_result.generations) == 21
    assert sum(item.latency_ms or 0 for item in hints.last_result.generations) == 7.0
    assert result.response.message == second.content
    assert result.leakage_check is not None and result.leakage_check.passed


def test_two_leaking_candidates_return_deterministic_safe_fallback() -> None:
    diagnoser = RecordingDiagnoser(_incorrect_diagnosis)
    model = FakeModelClient(
        text_responses=["The final answer is x=7.", "Therefore x=7 is the answer."]
    )
    orchestrator = TutorOrchestrator(
        diagnosis_service=diagnoser,
        hint_generator=HintGenerator(model),
    )

    result = orchestrator.tutor(
        _problem(),
        "2x + 4 = 18\n2x = 22",
        context=TutorContext(final_answers=("x=7",)),
        attempt_id="a-fallback",
    )

    assert model.text_call_count == 2
    assert result.response.safe_fallback_used
    assert "x=7" not in result.response.message
    assert result.leakage_check is not None and result.leakage_check.passed


def test_orchestration_boundary_rechecks_forged_safe_generation() -> None:
    orchestrator = TutorOrchestrator(
        diagnosis_service=RecordingDiagnoser(_incorrect_diagnosis),
        hint_generator=StaticHintProducer("The final answer is x=7.", claimed_safe=True),
    )

    result = orchestrator.tutor(
        _problem(),
        "2x + 4 = 18\n2x = 22",
        context=TutorContext(final_answers=("x=7",)),
        attempt_id="a-forged",
    )

    assert result.response.safe_fallback_used
    assert "x=7" not in result.response.message
    assert result.leakage_check is not None and result.leakage_check.passed


def test_pipeline_fails_closed_when_candidate_and_fallback_are_rejected() -> None:
    orchestrator = TutorOrchestrator(
        diagnosis_service=RecordingDiagnoser(_incorrect_diagnosis),
        hint_generator=StaticHintProducer("A harmless-looking hint."),
        leakage_checker=AlwaysRejectChecker(),
    )

    with pytest.raises(TutorSafetyError, match="no response passed"):
        orchestrator.tutor(
            _problem(),
            "2x + 4 = 18\n2x = 22",
            attempt_id="a-fail-closed",
        )


def test_direct_response_also_fails_closed_when_gate_rejects_it() -> None:
    orchestrator = TutorOrchestrator(
        diagnosis_service=RecordingDiagnoser(_correct_diagnosis),
        hint_generator=NeverHintProducer(),
        leakage_checker=AlwaysRejectChecker(),
    )

    with pytest.raises(TutorSafetyError, match="non-hint response"):
        orchestrator.tutor(_problem(), "2 + 2 = 4", attempt_id="a-direct-reject")


def test_explicit_reference_override_is_used_without_entering_low_reveal_prompt() -> None:
    stored = ReferenceSolution(reference_id="stored", text="STORED_PRIVATE_REFERENCE")
    override = ReferenceSolution(reference_id="override", text="OVERRIDE_PRIVATE_REFERENCE")
    diagnoser = RecordingDiagnoser(_incorrect_diagnosis)
    model = FakeModelClient(text_responses=["Which transition should you inspect first?"])
    orchestrator = TutorOrchestrator(
        diagnosis_service=diagnoser,
        hint_generator=HintGenerator(model),
    )

    orchestrator.tutor(
        _problem(references=(stored,)),
        "2x + 4 = 18\n2x = 22",
        reference_solutions=(override,),
        attempt_id="a-reference-override",
    )

    assert diagnoser.references_seen == [(override,)]
    prompt = model.text_requests[0].prompt
    assert stored.text not in prompt
    assert override.text not in prompt


def test_from_model_client_is_typed_and_runs_both_model_stages() -> None:
    raw_work = "2x + 4 = 18\n2x = 22"
    attempt = parse_student_attempt("p1", raw_work, attempt_id="a-model-client")
    diagnosis_payload = _incorrect_diagnosis(attempt).model_dump(mode="json")
    hint_generation = GenerationResult(
        content="Which arithmetic transition should you check first?",
        usage=TokenUsage(input_tokens=30, output_tokens=8, total_tokens=38),
        latency_ms=5.0,
    )
    fake = FakeModelClient(
        structured_responses=[diagnosis_payload],
        text_responses=[hint_generation],
    )
    client: ModelClient = fake

    orchestrator = TutorOrchestrator.from_model_client(client)
    assert_type(orchestrator, TutorOrchestrator)
    result = orchestrator.tutor(
        _problem(),
        raw_work,
        attempt_id="a-model-client",
    )

    assert result.decision.action is TutorAction.LIGHT_HINT
    assert fake.structured_call_count == 1
    assert fake.text_call_count == 1
    assert fake.structured_requests[0].metadata["stage"] == "diagnosis"
    assert fake.text_requests[0].metadata["component"] == "hint_generator"
    assert hint_generation.usage.total_tokens == 38


def test_prior_context_drives_attempt_number_and_escalation() -> None:
    orchestrator = TutorOrchestrator(
        diagnosis_service=RecordingDiagnoser(_incorrect_diagnosis),
        hint_generator=StaticHintProducer("Which condition must hold before that operation?"),
    )

    result = orchestrator.tutor(
        _problem(),
        "2x + 4 = 18\n2x = 22",
        context=TutorContext(
            prior_attempt_count=2,
            prior_action=TutorAction.TARGETED_HINT,
            current_reveal_level=RevealLevel.LOCATION,
            mode=TutorMode.HINT_ONLY,
        ),
        attempt_id="a-escalated",
    )

    assert result.attempt.attempt_number == 3
    assert result.decision.max_reveal_level is RevealLevel.CONCEPT
    assert result.decision.action is TutorAction.STRONG_HINT


def test_tutor_context_rejects_invalid_session_values() -> None:
    with pytest.raises(ValueError, match="prior_attempt_count"):
        TutorContext(prior_attempt_count=-1)
    with pytest.raises(ValueError, match="blank"):
        TutorContext(final_answers=(" ",))
