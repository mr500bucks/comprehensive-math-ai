"""Thin orchestration of the independently testable tutoring stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from math_feedback_ai.diagnosis.service import DiagnosisService
from math_feedback_ai.domain.models import (
    DiagnosisV1,
    HintPlan,
    LeakageCheckResult,
    Problem,
    ReferenceSolution,
    StudentAttempt,
    TutorDecision,
    TutorResponse,
    TutorResult,
)
from math_feedback_ai.domain.taxonomy import RevealLevel, TutorAction, TutorMode
from math_feedback_ai.hints.generator import HintGenerationResult, HintGenerator
from math_feedback_ai.hints.leakage import HintSafetyContext, LeakageChecker
from math_feedback_ai.model.client import ModelClient
from math_feedback_ai.parsing.steps import parse_student_attempt
from math_feedback_ai.policy.intervention import InterventionPolicy


class TutorSafetyError(RuntimeError):
    """No response could satisfy the policy and leakage gates."""


class Diagnoser(Protocol):
    """Replaceable structured mathematical diagnosis stage."""

    def diagnose(
        self,
        problem: Problem,
        attempt: StudentAttempt,
        reference_solutions: Sequence[ReferenceSolution] | None = None,
    ) -> DiagnosisV1: ...


class TutoringPolicy(Protocol):
    """Replaceable policy contract; the initial implementation is deterministic."""

    def decide(
        self,
        diagnosis: DiagnosisV1,
        *,
        prior_attempt_count: int = 0,
        prior_action: TutorAction | None = None,
        current_reveal_level: RevealLevel = RevealLevel.NONE,
        mode: TutorMode = TutorMode.HINT_ONLY,
        full_solution_authorized: bool = False,
        requested_reveal_level: RevealLevel | None = None,
    ) -> TutorDecision: ...


class HintProducer(Protocol):
    """Replaceable reveal-constrained generation stage."""

    def generate(
        self,
        plan: HintPlan,
        *,
        safety_context: HintSafetyContext | None = None,
    ) -> HintGenerationResult: ...


class SafetyChecker(Protocol):
    """Replaceable final policy/leakage validation stage."""

    def check(
        self,
        candidate: str,
        context: HintSafetyContext,
    ) -> LeakageCheckResult: ...


class AttemptParser(Protocol):
    """Source-preserving parser callable used at the pipeline boundary."""

    def __call__(
        self,
        problem_id: str,
        raw_text: str,
        *,
        attempt_id: str | None = None,
        attempt_number: int = 1,
    ) -> StudentAttempt: ...


@dataclass(frozen=True, slots=True)
class TutorContext:
    """Trusted session/policy facts for the current response.

    ``requested_reveal_level`` is only a student preference. The deterministic
    policy may honor it as a lower ceiling but never as permission to exceed
    ``mode`` or ``full_solution_authorized``.
    """

    prior_attempt_count: int = 0
    prior_action: TutorAction | None = None
    current_reveal_level: RevealLevel = RevealLevel.NONE
    mode: TutorMode = TutorMode.HINT_ONLY
    full_solution_authorized: bool = False
    requested_reveal_level: RevealLevel | None = None
    final_answers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prior_attempt_count < 0:
            raise ValueError("prior_attempt_count must be non-negative")
        if any(not answer.strip() for answer in self.final_answers):
            raise ValueError("final_answers cannot contain blank values")


@dataclass(slots=True)
class TutorOrchestrator:
    """Coordinate parsing, diagnosis, policy, generation, and safety checks."""

    diagnosis_service: Diagnoser
    hint_generator: HintProducer
    policy: TutoringPolicy = field(default_factory=InterventionPolicy)
    leakage_checker: SafetyChecker = field(default_factory=LeakageChecker)
    attempt_parser: AttemptParser = parse_student_attempt

    @classmethod
    def from_model_client(cls, client: ModelClient) -> TutorOrchestrator:
        """Build the standard vertical slice around one provider-neutral client."""

        return cls(
            diagnosis_service=DiagnosisService(client),
            hint_generator=HintGenerator(client),
        )

    def tutor(
        self,
        problem: Problem,
        student_solution: str,
        *,
        context: TutorContext | None = None,
        reference_solutions: Sequence[ReferenceSolution] | None = None,
        attempt_id: str | None = None,
    ) -> TutorResult:
        """Run one student attempt through the complete tutoring pipeline."""

        tutor_context = context or TutorContext()
        attempt = self.attempt_parser(
            problem.problem_id,
            student_solution,
            attempt_id=attempt_id,
            attempt_number=tutor_context.prior_attempt_count + 1,
        )
        references = tuple(
            problem.reference_solutions if reference_solutions is None else reference_solutions
        )
        diagnosis = self.diagnosis_service.diagnose(
            problem,
            attempt,
            reference_solutions=references,
        )
        decision = self.policy.decide(
            diagnosis,
            prior_attempt_count=tutor_context.prior_attempt_count,
            prior_action=tutor_context.prior_action,
            current_reveal_level=tutor_context.current_reveal_level,
            mode=tutor_context.mode,
            full_solution_authorized=tutor_context.full_solution_authorized,
            requested_reveal_level=tutor_context.requested_reveal_level,
        )

        if decision.action in {TutorAction.WAIT, TutorAction.ASK_STUDENT}:
            return self._non_hint_result(problem, attempt, diagnosis, decision)

        plan = self._build_hint_plan(
            problem=problem,
            attempt=attempt,
            diagnosis=diagnosis,
            decision=decision,
            references=references,
        )
        safety_context = self._safety_context(
            attempt=attempt,
            decision=decision,
            references=references,
            final_answers=tutor_context.final_answers,
            forbidden_content=plan.forbidden_content,
        )
        generation = self.hint_generator.generate(plan, safety_context=safety_context)
        if generation.safety_check.passed:
            # Recheck at the orchestration boundary so a custom HintProducer
            # cannot bypass the configured final gate with a forged result.
            final_check = self.leakage_checker.check(generation.text, safety_context)
            generation = replace(generation, safety_check=final_check)
        generation = self._ensure_safe_fallback(generation, safety_context)
        response = TutorResponse(
            action=decision.action,
            reveal_level=decision.max_reveal_level,
            message=generation.text,
            target_step_id=decision.target_step_id,
            requires_student_response=decision.action is not TutorAction.SHOW_FULL_SOLUTION,
            safe_fallback_used=generation.safe_fallback_used,
        )
        return TutorResult(
            problem=problem,
            attempt=attempt,
            diagnosis=diagnosis,
            decision=decision,
            hint_plan=plan,
            response=response,
            leakage_check=generation.safety_check,
        )

    def _non_hint_result(
        self,
        problem: Problem,
        attempt: StudentAttempt,
        diagnosis: DiagnosisV1,
        decision: TutorDecision,
    ) -> TutorResult:
        messages = {
            TutorAction.WAIT: (
                "Your submitted reasoning is mathematically valid; no hint is needed right now."
            ),
            TutorAction.ASK_STUDENT: "What would you try next, and why?",
        }
        message = messages[decision.action]
        check = self.leakage_checker.check(
            message,
            HintSafetyContext(
                max_reveal_level=decision.max_reveal_level,
                action=decision.action,
            ),
        )
        if not check.passed:
            raise TutorSafetyError("deterministic non-hint response failed its safety check")
        response = TutorResponse(
            action=decision.action,
            reveal_level=decision.max_reveal_level,
            message=message,
            target_step_id=decision.target_step_id,
            requires_student_response=decision.action is TutorAction.ASK_STUDENT,
        )
        return TutorResult(
            problem=problem,
            attempt=attempt,
            diagnosis=diagnosis,
            decision=decision,
            response=response,
            leakage_check=check,
        )

    @staticmethod
    def _build_hint_plan(
        *,
        problem: Problem,
        attempt: StudentAttempt,
        diagnosis: DiagnosisV1,
        decision: TutorDecision,
        references: Sequence[ReferenceSolution],
    ) -> HintPlan:
        target_position = next(
            (step.position for step in attempt.steps if step.step_id == decision.target_step_id),
            len(attempt.steps) - 1,
        )
        context_steps = tuple(step for step in attempt.steps if step.position <= target_position)
        if diagnosis.first_issue is not None:
            summary = diagnosis.first_issue.explanation
            target_issue = diagnosis.first_issue
        elif diagnosis.completion_gap is not None:
            summary = diagnosis.completion_gap.description
            target_issue = None
        else:
            summary = "The submitted step needs verification before stronger guidance."
            target_issue = None

        allowed_content: tuple[str, ...] = ()
        if decision.max_reveal_level >= RevealLevel.CONCEPT and target_issue is not None:
            allowed_content = target_issue.concept_tags
        reference_excerpt = (
            references[0].text
            if references and decision.max_reveal_level >= RevealLevel.FULL_SOLUTION
            else None
        )
        return HintPlan(
            problem_id=problem.problem_id,
            problem_statement=problem.statement,
            attempt_id=attempt.attempt_id,
            context_steps=context_steps,
            decision=decision,
            diagnostic_summary=summary,
            target_issue=target_issue,
            allowed_content=allowed_content,
            reference_excerpt=reference_excerpt,
        )

    @staticmethod
    def _safety_context(
        *,
        attempt: StudentAttempt,
        decision: TutorDecision,
        references: Sequence[ReferenceSolution],
        final_answers: tuple[str, ...],
        forbidden_content: tuple[str, ...],
    ) -> HintSafetyContext:
        target_position = next(
            (step.position for step in attempt.steps if step.step_id == decision.target_step_id),
            len(attempt.steps) - 1,
        )
        downstream = tuple(step.text for step in attempt.steps if step.position > target_position)
        return HintSafetyContext(
            max_reveal_level=decision.max_reveal_level,
            action=decision.action,
            final_answers=final_answers,
            reference_continuations=tuple(reference.text for reference in references),
            downstream_steps=downstream,
            forbidden_phrases=forbidden_content,
        )

    def _ensure_safe_fallback(
        self,
        generation: HintGenerationResult,
        context: HintSafetyContext,
    ) -> HintGenerationResult:
        if generation.safety_check.passed:
            return generation
        fallback = "What justifies the transition at the highlighted step?"
        check = self.leakage_checker.check(fallback, context)
        if not check.passed:
            raise TutorSafetyError("no response passed the leakage and policy checks")
        return HintGenerationResult(
            text=fallback,
            safety_check=check,
            model_call_count=generation.model_call_count,
            generations=generation.generations,
            rejected_checks=(*generation.rejected_checks, generation.safety_check),
            safe_fallback_used=True,
            regenerated=generation.regenerated,
            errors=generation.errors,
        )


__all__ = ["TutorContext", "TutorOrchestrator", "TutorSafetyError"]
