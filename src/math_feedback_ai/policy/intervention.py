"""Small deterministic policy for choosing the next tutoring intervention."""

from __future__ import annotations

from dataclasses import dataclass

from math_feedback_ai.domain import (
    DiagnosisV1,
    IssueCode,
    OverallStatus,
    RevealLevel,
    TutorAction,
    TutorDecision,
    TutorMode,
)

_ACTION_LEVEL: dict[TutorAction, RevealLevel] = {
    TutorAction.WAIT: RevealLevel.NONE,
    TutorAction.ASK_STUDENT: RevealLevel.NONE,
    TutorAction.VERIFY_STEP: RevealLevel.NONE,
    TutorAction.LIGHT_HINT: RevealLevel.LIGHT_DIRECTION,
    TutorAction.TARGETED_HINT: RevealLevel.LOCATION,
    TutorAction.STRONG_HINT: RevealLevel.CONCEPT,
    TutorAction.EXPLAIN_CONCEPT: RevealLevel.CONCEPT,
    TutorAction.SHOW_PARTIAL_SOLUTION: RevealLevel.SUBSTANTIAL_SCAFFOLD,
    TutorAction.SHOW_FULL_SOLUTION: RevealLevel.FULL_SOLUTION,
}


@dataclass(frozen=True, slots=True)
class InterventionPolicy:
    """Select a bounded action using auditable rules rather than another model.

    ``prior_attempt_count`` and ``current_reveal_level`` are trusted session
    facts, while ``requested_reveal_level`` is only a student preference. It is
    intentionally unable to raise the system mode's ceiling.
    """

    confidence_threshold: float = 0.65

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

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
    ) -> TutorDecision:
        """Choose the smallest justified next action.

        The method never mutates the diagnosis and never infers correctness from
        a student's request. Stronger levels require prior attempts, a prior
        revealed level, or both.
        """

        if prior_attempt_count < 0:
            raise ValueError("prior_attempt_count must be non-negative")
        current = self._coerce_level(current_reveal_level, "current_reveal_level")
        try:
            mode = TutorMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported tutoring mode: {mode!r}") from exc
        if prior_action is not None:
            try:
                prior_action = TutorAction(prior_action)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported prior action: {prior_action!r}") from exc
        requested_ceiling = (
            self._coerce_level(requested_reveal_level, "requested_reveal_level")
            if requested_reveal_level is not None
            else RevealLevel.FULL_SOLUTION
        )

        target_step_id = self._target_step_id(diagnosis)
        status = diagnosis.overall_status
        issue = diagnosis.first_issue

        if (
            status is OverallStatus.INDETERMINATE
            or diagnosis.confidence < self.confidence_threshold
            or (issue is not None and issue.code is IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE)
        ):
            if target_step_id:
                return self._decision(
                    TutorAction.VERIFY_STEP,
                    RevealLevel.NONE,
                    target_step_id,
                    "diagnosis_requires_verification",
                )
            return self._decision(
                TutorAction.ASK_STUDENT,
                RevealLevel.NONE,
                None,
                "diagnosis_needs_more_information",
            )

        if status is OverallStatus.FULLY_CORRECT:
            return self._decision(
                TutorAction.WAIT,
                RevealLevel.NONE,
                None,
                "solution_fully_correct",
            )

        if status is OverallStatus.CORRECT_BUT_INEFFICIENT:
            return self._decision(
                TutorAction.WAIT,
                RevealLevel.NONE,
                None,
                "valid_solution_no_unsolicited_optimization",
            )

        ceiling = self._mode_ceiling(mode, full_solution_authorized)
        ceiling = RevealLevel(min(int(ceiling), int(requested_ceiling)))
        effective_current = max(current, self._prior_action_level(prior_action))

        if status is OverallStatus.INCOMPLETE and issue is None:
            if prior_attempt_count == 0 and effective_current == RevealLevel.NONE:
                return self._decision(
                    TutorAction.ASK_STUDENT,
                    RevealLevel.NONE,
                    target_step_id,
                    "productive_incomplete_prompt_for_next_step",
                )
            # Incompleteness is escalated one tier more slowly than a known
            # mathematical issue: ask first, then offer a directional hint.
            eligible = max(int(effective_current) + 1, min(prior_attempt_count, 5))
            level = RevealLevel(min(eligible, int(ceiling)))
            level = self._level_without_target(level, target_step_id)
            action = self._action_for_level(level, issue_code=None)
            return self._decision(
                action,
                level,
                target_step_id,
                "incomplete_reasoning_escalated",
                full_solution_authorized=(
                    action is TutorAction.SHOW_FULL_SOLUTION and full_solution_authorized
                ),
            )

        if status is OverallStatus.INCORRECT or issue is not None:
            if issue is None:
                return self._decision(
                    TutorAction.VERIFY_STEP,
                    RevealLevel.NONE,
                    target_step_id,
                    "incorrect_but_issue_not_localized",
                )
            eligible = max(int(effective_current) + 1, min(prior_attempt_count + 1, 5))
            level = RevealLevel(min(eligible, int(ceiling)))
            level = self._level_without_target(level, target_step_id)
            action = self._action_for_level(level, issue_code=issue.code)
            if level is RevealLevel.NONE:
                rationale = "localized_issue_student_requested_no_hint"
            elif level is RevealLevel.LIGHT_DIRECTION:
                rationale = "localized_issue_first_hint"
            else:
                rationale = "localized_issue_escalated"
            return self._decision(
                action,
                level,
                target_step_id,
                rationale,
                full_solution_authorized=(
                    action is TutorAction.SHOW_FULL_SOLUTION and full_solution_authorized
                ),
            )

        # DiagnosisV1 currently makes this branch unreachable, but failing
        # closed protects future status additions until policy rules are added.
        return self._decision(
            TutorAction.ASK_STUDENT,
            RevealLevel.NONE,
            target_step_id,
            "status_not_supported_by_policy",
        )

    @staticmethod
    def _mode_ceiling(mode: TutorMode, full_solution_authorized: bool) -> RevealLevel:
        if mode is TutorMode.HINT_ONLY:
            return RevealLevel.CONCEPT
        if mode is TutorMode.GUIDED:
            return RevealLevel.SUBSTANTIAL_SCAFFOLD
        if mode is TutorMode.FULL_SOLUTION_ALLOWED and full_solution_authorized:
            return RevealLevel.FULL_SOLUTION
        return RevealLevel.SUBSTANTIAL_SCAFFOLD

    @staticmethod
    def _prior_action_level(action: TutorAction | None) -> RevealLevel:
        if action is None:
            return RevealLevel.NONE
        return _ACTION_LEVEL[action]

    @staticmethod
    def _action_for_level(
        level: RevealLevel,
        *,
        issue_code: IssueCode | None,
    ) -> TutorAction:
        if level is RevealLevel.NONE:
            return TutorAction.ASK_STUDENT
        if level is RevealLevel.LIGHT_DIRECTION:
            return TutorAction.LIGHT_HINT
        if level is RevealLevel.LOCATION:
            return TutorAction.TARGETED_HINT
        if level is RevealLevel.CONCEPT:
            conceptual_codes = {
                IssueCode.CONDITION_MISSING,
                IssueCode.THEOREM_MISUSED,
                IssueCode.CONCEPT_MISUNDERSTOOD,
            }
            return (
                TutorAction.EXPLAIN_CONCEPT
                if issue_code is None or issue_code in conceptual_codes
                else TutorAction.STRONG_HINT
            )
        if level is RevealLevel.SUBSTANTIAL_SCAFFOLD:
            return TutorAction.SHOW_PARTIAL_SOLUTION
        return TutorAction.SHOW_FULL_SOLUTION

    @staticmethod
    def _level_without_target(
        level: RevealLevel,
        target_step_id: str | None,
    ) -> RevealLevel:
        if target_step_id is not None:
            return level
        if level is RevealLevel.LOCATION:
            return RevealLevel.LIGHT_DIRECTION
        if level is RevealLevel.SUBSTANTIAL_SCAFFOLD:
            return RevealLevel.CONCEPT
        return level

    @staticmethod
    def _target_step_id(diagnosis: DiagnosisV1) -> str | None:
        if diagnosis.first_issue is not None:
            return diagnosis.first_issue.step_id
        if diagnosis.completion_gap is not None:
            return diagnosis.completion_gap.after_step_id
        return None

    @staticmethod
    def _coerce_level(value: RevealLevel, field_name: str) -> RevealLevel:
        try:
            return RevealLevel(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a reveal level from 0 to 5") from exc

    @staticmethod
    def _decision(
        action: TutorAction,
        level: RevealLevel,
        target_step_id: str | None,
        rationale_code: str,
        *,
        full_solution_authorized: bool = False,
    ) -> TutorDecision:
        return TutorDecision(
            action=action,
            max_reveal_level=level,
            target_step_id=target_step_id,
            rationale_code=rationale_code,
            full_solution_authorized=full_solution_authorized,
        )


__all__ = ["InterventionPolicy"]
