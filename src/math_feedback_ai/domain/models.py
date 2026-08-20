"""Canonical, provider-neutral domain models for Math Feedback AI.

These models are deliberately stricter than ordinary transport objects.  They
make contradictory diagnoses, dangling step references, and accidental public
disclosure of reference solutions difficult to represent.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .taxonomy import (
    ISSUE_TAXONOMY_VERSION,
    IssueCode,
    LeakageSeverity,
    OverallStatus,
    ReferenceRelation,
    RevealLevel,
    StepParseStrategy,
    StepStatus,
    TutorAction,
)

StableId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class CanonicalModel(BaseModel):
    """Base configuration shared by serialized research contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )

    @staticmethod
    def _require_nonblank(value: str, field_name: str) -> str:
        if not value.strip():
            raise ValueError(f"{field_name} must contain non-whitespace text")
        return value


class ReferenceSolution(CanonicalModel):
    """One optional example solution; references are never exhaustive templates."""

    reference_id: StableId
    text: Annotated[str, Field(min_length=1)]
    method_label: Annotated[str, Field(min_length=1, max_length=120)] | None = None

    @field_validator("text")
    @classmethod
    def _text_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "reference solution")

    @field_validator("method_label")
    @classmethod
    def _label_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "method_label")


class Problem(CanonicalModel):
    """A math problem and zero or more explicitly non-exhaustive references."""

    problem_id: StableId
    statement: Annotated[str, Field(min_length=1)]
    reference_solutions: tuple[ReferenceSolution, ...] = ()

    @field_validator("statement")
    @classmethod
    def _statement_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "problem statement")

    @model_validator(mode="after")
    def _reference_ids_are_unique(self) -> Self:
        ids = [reference.reference_id for reference in self.reference_solutions]
        if len(ids) != len(set(ids)):
            raise ValueError("reference solution IDs must be unique")
        return self


class SolutionStep(CanonicalModel):
    """A source-preserving slice of the raw student solution.

    start_offset is inclusive and end_offset is exclusive. text is required to
    equal that exact slice when embedded in a StudentAttempt.
    """

    step_id: StableId
    position: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1)]
    start_offset: Annotated[int, Field(ge=0)]
    end_offset: Annotated[int, Field(gt=0)]
    segmentation_ambiguous: bool = False

    @field_validator("text")
    @classmethod
    def _step_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "solution step")

    @model_validator(mode="after")
    def _span_is_well_formed(self) -> Self:
        if self.end_offset <= self.start_offset:
            raise ValueError("step end_offset must be greater than start_offset")
        return self


def _validate_steps_against_raw(
    raw_text: str,
    steps: tuple[SolutionStep, ...],
    *,
    require_contiguous_positions: bool,
) -> None:
    if not steps:
        raise ValueError("at least one solution step is required")

    ids = [step.step_id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("solution step IDs must be unique")

    positions = [step.position for step in steps]
    if require_contiguous_positions and positions != list(range(len(steps))):
        raise ValueError("solution step positions must be contiguous and zero-based")
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("solution steps must be in strictly increasing position order")

    previous_end = 0
    for step in steps:
        if step.end_offset > len(raw_text):
            raise ValueError(f"step {step.step_id!r} ends outside the raw solution")
        if step.start_offset < previous_end:
            raise ValueError("solution step spans must not overlap")
        if raw_text[step.start_offset : step.end_offset] != step.text:
            raise ValueError(f"step {step.step_id!r} text must equal its exact raw-text slice")
        if raw_text[previous_end : step.start_offset].strip():
            raise ValueError("non-whitespace text exists outside the recorded step spans")
        previous_end = step.end_offset

    if raw_text[previous_end:].strip():
        raise ValueError("non-whitespace text exists after the final recorded step")


class StepParseResult(CanonicalModel):
    """Source-preserving result returned by the deterministic step parser."""

    raw_text: Annotated[str, Field(min_length=1)]
    steps: Annotated[tuple[SolutionStep, ...], Field(min_length=1)]
    strategy: StepParseStrategy
    ambiguous: bool = False
    ambiguity_reasons: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    @field_validator("raw_text")
    @classmethod
    def _raw_text_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "raw solution")

    @model_validator(mode="after")
    def _steps_preserve_source(self) -> Self:
        _validate_steps_against_raw(self.raw_text, self.steps, require_contiguous_positions=True)
        any_step_ambiguous = any(step.segmentation_ambiguous for step in self.steps)
        if any_step_ambiguous and not self.ambiguous:
            raise ValueError("ambiguous steps require ambiguous=True")
        if self.ambiguous and not self.ambiguity_reasons:
            raise ValueError("an ambiguous parse must explain why it is ambiguous")
        if not self.ambiguous and self.ambiguity_reasons:
            raise ValueError("unambiguous parses cannot contain ambiguity reasons")
        return self


class StudentAttempt(CanonicalModel):
    """A complete, source-preserving student attempt."""

    attempt_id: StableId
    problem_id: StableId
    raw_text: Annotated[str, Field(min_length=1)]
    steps: Annotated[tuple[SolutionStep, ...], Field(min_length=1)]
    attempt_number: Annotated[int, Field(ge=1)] = 1
    segmentation_ambiguous: bool = False
    parse_strategy: StepParseStrategy = StepParseStrategy.SINGLE_CHUNK
    parse_notes: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    @field_validator("raw_text")
    @classmethod
    def _attempt_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "student attempt")

    @model_validator(mode="after")
    def _steps_preserve_attempt(self) -> Self:
        _validate_steps_against_raw(self.raw_text, self.steps, require_contiguous_positions=True)
        any_step_ambiguous = any(step.segmentation_ambiguous for step in self.steps)
        if any_step_ambiguous and not self.segmentation_ambiguous:
            raise ValueError("ambiguous steps require segmentation_ambiguous=True")
        if self.segmentation_ambiguous and not self.parse_notes:
            raise ValueError("ambiguous segmentation must include a parse note")
        if not self.segmentation_ambiguous and self.parse_notes:
            raise ValueError("unambiguous segmentation cannot include parse notes")
        return self


class StepAssessment(CanonicalModel):
    """A structured mathematical assessment of one student-authored step."""

    step_id: StableId
    status: StepStatus
    issue_codes: tuple[IssueCode, ...] = ()
    explanation: Annotated[str, Field(min_length=1)] | None = None
    evidence: Annotated[str, Field(min_length=1)] | None = None
    confidence: Confidence

    @field_validator("explanation", "evidence")
    @classmethod
    def _optional_text_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "assessment text")

    @model_validator(mode="after")
    def _assessment_is_coherent(self) -> Self:
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("assessment issue_codes must be unique")
        if self.status in {StepStatus.INVALID, StepStatus.UNSUPPORTED}:
            if not self.issue_codes:
                raise ValueError("invalid or unsupported steps require an issue code")
            if self.explanation is None:
                raise ValueError("invalid or unsupported steps require an explanation")
        if self.status is StepStatus.AMBIGUOUS and self.explanation is None:
            raise ValueError("ambiguous steps require an explanation")
        if self.status is StepStatus.VALID:
            disallowed = set(self.issue_codes) - {IssueCode.RELEVANCE_IRRELEVANT}
            if disallowed:
                raise ValueError("valid steps may only carry relevance.irrelevant")
        return self


class Issue(CanonicalModel):
    """The earliest meaningful localized issue in an attempt."""

    taxonomy_version: Literal["1.0"] = ISSUE_TAXONOMY_VERSION
    step_id: StableId
    code: IssueCode
    explanation: Annotated[str, Field(min_length=1)]
    evidence: Annotated[str, Field(min_length=1)] | None = None
    concept_tags: tuple[Annotated[str, Field(min_length=1, max_length=80)], ...] = ()
    confidence: Confidence

    @field_validator("explanation", "evidence")
    @classmethod
    def _issue_text_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "issue text")

    @model_validator(mode="after")
    def _concept_tags_are_unique(self) -> Self:
        if len(self.concept_tags) != len(set(self.concept_tags)):
            raise ValueError("concept_tags must be unique")
        return self


class CompletionGap(CanonicalModel):
    """Where a mathematically usable prefix stopped without becoming incorrect."""

    after_step_id: StableId | None
    description: Annotated[str, Field(min_length=1)]
    concept_tags: tuple[Annotated[str, Field(min_length=1, max_length=80)], ...] = ()
    confidence: Confidence

    @field_validator("description")
    @classmethod
    def _description_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "completion gap description")

    @model_validator(mode="after")
    def _gap_tags_are_unique(self) -> Self:
        if len(self.concept_tags) != len(set(self.concept_tags)):
            raise ValueError("completion-gap concept_tags must be unique")
        return self


class DiagnosisV1(CanonicalModel):
    """Version 1 structured mathematical diagnosis, independent of tutor policy."""

    schema_version: Literal["1.0"] = "1.0"
    attempt_id: StableId
    overall_status: OverallStatus
    step_assessments: Annotated[tuple[StepAssessment, ...], Field(min_length=1)]
    first_issue: Issue | None = None
    completion_gap: CompletionGap | None = None
    reusable_prefix_end_step_id: StableId | None = None
    earlier_reasoning_usable: bool
    reference_relation: ReferenceRelation = ReferenceRelation.NOT_USED
    confidence: Confidence
    confidence_reasons: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    @model_validator(mode="after")
    def _diagnosis_is_coherent(self) -> Self:
        assessment_ids = [item.step_id for item in self.step_assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("each step may be assessed only once")
        known_ids = set(assessment_ids)

        referenced_ids = {
            step_id
            for step_id in (
                self.first_issue.step_id if self.first_issue else None,
                self.completion_gap.after_step_id if self.completion_gap else None,
                self.reusable_prefix_end_step_id,
            )
            if step_id is not None
        }
        dangling = referenced_ids - known_ids
        if dangling:
            raise ValueError(f"diagnosis references unassessed step IDs: {sorted(dangling)}")

        if self.first_issue is not None:
            assessment = next(
                item for item in self.step_assessments if item.step_id == self.first_issue.step_id
            )
            if self.first_issue.code not in assessment.issue_codes:
                raise ValueError("first_issue code must appear on its step assessment")
            if (
                assessment.status is StepStatus.VALID
                and self.first_issue.code is not IssueCode.RELEVANCE_IRRELEVANT
            ):
                raise ValueError("a mathematical first issue cannot point to a valid step")
            if (
                self.overall_status is OverallStatus.INCORRECT
                and assessment.status is StepStatus.VALID
            ):
                raise ValueError("an incorrect diagnosis must localize a mathematical issue")
            questionable = next(
                (
                    item
                    for item in self.step_assessments
                    if item.status is not StepStatus.VALID
                    or (
                        self.overall_status is OverallStatus.CORRECT_BUT_INEFFICIENT
                        and item.issue_codes
                    )
                ),
                None,
            )
            if questionable is not None and questionable.step_id != self.first_issue.step_id:
                raise ValueError("first_issue must be the earliest questionable assessment")

        if self.reusable_prefix_end_step_id is not None and not self.earlier_reasoning_usable:
            raise ValueError("a reusable prefix requires earlier_reasoning_usable=True")

        if self.overall_status is OverallStatus.FULLY_CORRECT:
            if self.first_issue is not None or self.completion_gap is not None:
                raise ValueError("a fully correct diagnosis cannot contain an issue or gap")
            if any(
                assessment.status is not StepStatus.VALID or assessment.issue_codes
                for assessment in self.step_assessments
            ):
                raise ValueError("all steps in a fully correct diagnosis must be valid")
            if not self.earlier_reasoning_usable:
                raise ValueError("fully correct reasoning must be marked usable")

        if self.overall_status is OverallStatus.INCORRECT and self.first_issue is None:
            raise ValueError("an incorrect diagnosis must localize a first issue")

        if self.overall_status is OverallStatus.INCOMPLETE:
            if self.completion_gap is None:
                raise ValueError("an incomplete diagnosis requires a completion gap")
            if self.first_issue is not None:
                raise ValueError("a merely incomplete diagnosis cannot contain a first issue")
            if any(
                assessment.status is not StepStatus.VALID for assessment in self.step_assessments
            ):
                raise ValueError(
                    "a merely incomplete diagnosis must contain a valid reasoning prefix"
                )
            if not self.earlier_reasoning_usable:
                raise ValueError("an incomplete valid prefix must be marked usable")

        if self.overall_status is OverallStatus.CORRECT_BUT_INEFFICIENT:
            if self.completion_gap is not None:
                raise ValueError("correct-but-inefficient work cannot contain a completion gap")
            if any(
                assessment.status is not StepStatus.VALID for assessment in self.step_assessments
            ):
                raise ValueError("correct-but-inefficient work must remain mathematically valid")
            if (
                self.first_issue is not None
                and self.first_issue.code is not IssueCode.RELEVANCE_IRRELEVANT
            ):
                raise ValueError(
                    "correct-but-inefficient work may only localize relevance.irrelevant"
                )
            if not self.earlier_reasoning_usable:
                raise ValueError("correct-but-inefficient reasoning must be marked usable")

        return self

    @classmethod
    def indeterminate(cls, attempt: StudentAttempt, reason: str) -> DiagnosisV1:
        """Build a safe diagnosis when reliable structured analysis is unavailable."""

        cls._require_nonblank(reason, "indeterminate reason")
        assessments = tuple(
            StepAssessment(
                step_id=step.step_id,
                status=StepStatus.AMBIGUOUS,
                issue_codes=(IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,),
                explanation=reason,
                confidence=0.0,
            )
            for step in attempt.steps
        )
        return cls(
            attempt_id=attempt.attempt_id,
            overall_status=OverallStatus.INDETERMINATE,
            step_assessments=assessments,
            earlier_reasoning_usable=False,
            confidence=0.0,
            confidence_reasons=(reason,),
        )


_ACTION_REVEAL_BOUNDS: dict[TutorAction, tuple[int, int]] = {
    TutorAction.WAIT: (0, 0),
    TutorAction.ASK_STUDENT: (0, 0),
    TutorAction.VERIFY_STEP: (0, 2),
    TutorAction.LIGHT_HINT: (1, 1),
    TutorAction.TARGETED_HINT: (2, 3),
    TutorAction.STRONG_HINT: (3, 4),
    TutorAction.EXPLAIN_CONCEPT: (3, 4),
    TutorAction.SHOW_PARTIAL_SOLUTION: (4, 4),
    TutorAction.SHOW_FULL_SOLUTION: (5, 5),
}


def _validate_action_reveal(
    action: TutorAction,
    reveal_level: RevealLevel,
    *,
    full_solution_authorized: bool,
) -> None:
    minimum, maximum = _ACTION_REVEAL_BOUNDS[action]
    if not minimum <= int(reveal_level) <= maximum:
        raise ValueError(f"{action.value} requires reveal level between {minimum} and {maximum}")
    if action is TutorAction.SHOW_FULL_SOLUTION and not full_solution_authorized:
        raise ValueError("full solutions require explicit authorization")
    if action is not TutorAction.SHOW_FULL_SOLUTION and full_solution_authorized:
        raise ValueError("full-solution authorization is only valid for that action")


class TutorDecision(CanonicalModel):
    """Machine-readable output of the deterministic tutoring policy."""

    action: TutorAction
    max_reveal_level: RevealLevel
    target_step_id: StableId | None = None
    rationale_code: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=120,
            pattern=r"^[a-z0-9][a-z0-9._-]*$",
        ),
    ]
    rationale: Annotated[str, Field(min_length=1)] | None = None
    full_solution_authorized: bool = False

    @field_validator("rationale")
    @classmethod
    def _rationale_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "rationale")

    @model_validator(mode="after")
    def _action_matches_reveal_level(self) -> Self:
        _validate_action_reveal(
            self.action,
            self.max_reveal_level,
            full_solution_authorized=self.full_solution_authorized,
        )
        if (
            self.action
            in {
                TutorAction.VERIFY_STEP,
                TutorAction.TARGETED_HINT,
                TutorAction.STRONG_HINT,
                TutorAction.SHOW_PARTIAL_SOLUTION,
            }
            and self.target_step_id is None
        ):
            raise ValueError(f"{self.action.value} requires a target_step_id")
        return self


class HintPlan(CanonicalModel):
    """Minimal, reveal-bounded context passed to hint generation.

    Reference text is omitted by default and forbidden below level 4.  The plan
    contains decisions already made by diagnosis and policy; a generator must
    not reinterpret them.
    """

    problem_id: StableId
    problem_statement: Annotated[str, Field(min_length=1)]
    attempt_id: StableId
    context_steps: Annotated[tuple[SolutionStep, ...], Field(min_length=1)]
    decision: TutorDecision
    diagnostic_summary: Annotated[str, Field(min_length=1)]
    target_issue: Issue | None = None
    allowed_content: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    forbidden_content: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    reference_excerpt: Annotated[str, Field(min_length=1)] | None = None

    @field_validator("problem_statement", "diagnostic_summary", "reference_excerpt")
    @classmethod
    def _plan_text_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "hint plan text")

    @model_validator(mode="after")
    def _plan_is_minimal_and_coherent(self) -> Self:
        if self.decision.action in {TutorAction.WAIT, TutorAction.ASK_STUDENT}:
            raise ValueError("WAIT and ASK_STUDENT do not require a HintPlan")
        ids = [step.step_id for step in self.context_steps]
        if len(ids) != len(set(ids)):
            raise ValueError("HintPlan context step IDs must be unique")
        positions = [step.position for step in self.context_steps]
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ValueError("HintPlan context steps must remain in source order")
        if self.decision.target_step_id is not None and self.decision.target_step_id not in ids:
            raise ValueError("the decision target must be present in context_steps")
        if self.target_issue is not None:
            if self.target_issue.step_id not in ids:
                raise ValueError("the target issue must reference a context step")
            if (
                self.decision.target_step_id is not None
                and self.target_issue.step_id != self.decision.target_step_id
            ):
                raise ValueError("target_issue and decision must target the same step")
        if (
            self.reference_excerpt is not None
            and self.decision.max_reveal_level < RevealLevel.SUBSTANTIAL_SCAFFOLD
        ):
            raise ValueError("reference text is forbidden below reveal level 4")
        if len(self.allowed_content) != len(set(self.allowed_content)):
            raise ValueError("allowed_content entries must be unique")
        if len(self.forbidden_content) != len(set(self.forbidden_content)):
            raise ValueError("forbidden_content entries must be unique")
        overlap = set(self.allowed_content) & set(self.forbidden_content)
        if overlap:
            raise ValueError("content cannot be both allowed and forbidden")
        return self


class TutorResponse(CanonicalModel):
    """Small public response that excludes private diagnosis/reference content."""

    action: TutorAction
    reveal_level: RevealLevel
    message: Annotated[str, Field(min_length=1)]
    target_step_id: StableId | None = None
    requires_student_response: bool = False
    safe_fallback_used: bool = False

    @field_validator("message")
    @classmethod
    def _message_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "tutor response")

    @model_validator(mode="after")
    def _response_action_matches_level(self) -> Self:
        # Public responses do not carry authorization.  That is checked on the
        # corresponding TutorDecision/TutorResult.
        minimum, maximum = _ACTION_REVEAL_BOUNDS[self.action]
        if not minimum <= int(self.reveal_level) <= maximum:
            raise ValueError(
                f"{self.action.value} requires reveal level between {minimum} and {maximum}"
            )
        if (
            self.action
            in {
                TutorAction.VERIFY_STEP,
                TutorAction.TARGETED_HINT,
                TutorAction.STRONG_HINT,
                TutorAction.SHOW_PARTIAL_SOLUTION,
            }
            and self.target_step_id is None
        ):
            raise ValueError(f"{self.action.value} requires a target_step_id")
        return self


class LeakageViolation(CanonicalModel):
    """One interpretable finding from a leakage or policy check."""

    code: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=120,
            pattern=r"^[a-z0-9][a-z0-9._-]*$",
        ),
    ]
    message: Annotated[str, Field(min_length=1)]
    severity: LeakageSeverity

    @field_validator("message")
    @classmethod
    def _violation_message_is_nonblank(cls, value: str) -> str:
        return cls._require_nonblank(value, "leakage violation message")


_SEVERITY_RANK = {
    LeakageSeverity.INFO: 0,
    LeakageSeverity.WARNING: 1,
    LeakageSeverity.HARD: 2,
}


class LeakageCheckResult(CanonicalModel):
    """Structured result from interpretable, non-formal leakage checks."""

    passed: bool
    violations: tuple[LeakageViolation, ...] = ()
    severity: LeakageSeverity | None = None

    @model_validator(mode="after")
    def _result_matches_violations(self) -> Self:
        has_hard_violation = any(
            violation.severity is LeakageSeverity.HARD for violation in self.violations
        )
        if self.passed == has_hard_violation:
            raise ValueError("passed must be true exactly when no HARD violation exists")
        expected = (
            max(
                (violation.severity for violation in self.violations),
                key=_SEVERITY_RANK.__getitem__,
            )
            if self.violations
            else None
        )
        if self.severity is not None and self.severity is not expected:
            raise ValueError("severity must equal the most severe violation")
        if self.severity is None and expected is not None:
            object.__setattr__(self, "severity", expected)
        return self


def _validate_attempt_diagnosis(attempt: StudentAttempt, diagnosis: DiagnosisV1) -> None:
    if diagnosis.attempt_id != attempt.attempt_id:
        raise ValueError("diagnosis attempt_id must match the student attempt")
    attempt_ids = {step.step_id for step in attempt.steps}
    assessed_ids = {item.step_id for item in diagnosis.step_assessments}
    if assessed_ids != attempt_ids:
        missing = sorted(attempt_ids - assessed_ids)
        unknown = sorted(assessed_ids - attempt_ids)
        raise ValueError(
            "diagnosis must assess every and only student step "
            f"(missing={missing}, unknown={unknown})"
        )
    if [item.step_id for item in diagnosis.step_assessments] != [
        step.step_id for step in attempt.steps
    ]:
        raise ValueError("diagnosis assessments must remain in student step order")


def _validate_step_references(step_ids: Iterable[str], *references: str | None) -> None:
    known = set(step_ids)
    dangling = {
        reference for reference in references if reference is not None and reference not in known
    }
    if dangling:
        raise ValueError(f"dangling student step references: {sorted(dangling)}")


class TutorResult(CanonicalModel):
    """Research/debug result containing private pipeline artifacts and response."""

    problem: Problem
    attempt: StudentAttempt
    diagnosis: DiagnosisV1
    decision: TutorDecision
    hint_plan: HintPlan | None = None
    response: TutorResponse
    leakage_check: LeakageCheckResult | None = None

    @model_validator(mode="after")
    def _pipeline_artifacts_agree(self) -> Self:
        if self.attempt.problem_id != self.problem.problem_id:
            raise ValueError("attempt problem_id must match the problem")
        _validate_attempt_diagnosis(self.attempt, self.diagnosis)
        step_ids = [step.step_id for step in self.attempt.steps]
        _validate_step_references(
            step_ids,
            self.decision.target_step_id,
            self.response.target_step_id,
        )
        if self.response.action is not self.decision.action:
            raise ValueError("public response action must match the policy decision")
        if self.response.reveal_level > self.decision.max_reveal_level:
            raise ValueError("public response exceeds the authorized reveal level")
        if self.response.target_step_id != self.decision.target_step_id:
            raise ValueError("public response target must match the policy decision")
        if self.hint_plan is not None:
            if self.hint_plan.problem_id != self.problem.problem_id:
                raise ValueError("HintPlan problem_id must match the problem")
            if self.hint_plan.problem_statement != self.problem.statement:
                raise ValueError("HintPlan statement must match the problem")
            if self.hint_plan.attempt_id != self.attempt.attempt_id:
                raise ValueError("HintPlan attempt_id must match the attempt")
            if self.hint_plan.decision != self.decision:
                raise ValueError("HintPlan must contain the same policy decision")
            attempt_steps = {step.step_id: step for step in self.attempt.steps}
            for context_step in self.hint_plan.context_steps:
                if attempt_steps.get(context_step.step_id) != context_step:
                    raise ValueError("HintPlan context steps must be exact slices of the attempt")
        return self


class ExampleProvenance(CanonicalModel):
    """Minimal provenance needed to distinguish fixtures from research data."""

    source: Annotated[str, Field(min_length=1, max_length=240)]
    synthetic: bool
    license: Annotated[str, Field(min_length=1, max_length=120)] | None = None
    notes: Annotated[str, Field(min_length=1)] | None = None

    @field_validator("source", "license", "notes")
    @classmethod
    def _provenance_text_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_nonblank(value, "provenance text")


class TutoringExampleV1(CanonicalModel):
    """Canonical versioned storage contract for one labelled tutoring example."""

    schema_version: Literal["1.0"] = "1.0"
    example_id: StableId
    provenance: ExampleProvenance
    problem: Problem
    student_attempt: StudentAttempt
    gold_diagnosis: DiagnosisV1
    expected_decision: TutorDecision | None = None
    ideal_responses: tuple[TutorResponse, ...] = ()

    SCHEMA_PATH: ClassVar[str] = "schemas/tutoring_example.v1.schema.json"

    @model_validator(mode="after")
    def _example_references_are_sound(self) -> Self:
        if self.student_attempt.problem_id != self.problem.problem_id:
            raise ValueError("student_attempt problem_id must match the problem")
        _validate_attempt_diagnosis(self.student_attempt, self.gold_diagnosis)
        step_ids = [step.step_id for step in self.student_attempt.steps]
        if self.expected_decision is not None:
            _validate_step_references(step_ids, self.expected_decision.target_step_id)
        for response in self.ideal_responses:
            _validate_step_references(step_ids, response.target_step_id)
            if (
                self.expected_decision is not None
                and response.action is not self.expected_decision.action
            ):
                raise ValueError("ideal response actions must match the expected policy decision")
            if (
                self.expected_decision is not None
                and response.reveal_level > self.expected_decision.max_reveal_level
            ):
                raise ValueError("ideal responses cannot exceed the expected reveal ceiling")
            if (
                self.expected_decision is not None
                and response.target_step_id != self.expected_decision.target_step_id
            ):
                raise ValueError("ideal response targets must match the expected policy decision")
        return self

    @classmethod
    def canonical_json_schema(cls) -> dict[str, Any]:
        """Return the authoritative JSON Schema for the v1 example contract."""

        return cls.model_json_schema(
            ref_template="#/$defs/{model}",
            mode="validation",
        )
