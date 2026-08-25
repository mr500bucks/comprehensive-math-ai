"""Contracts and structural validation for provisional diagnosis pilot data.

Structural validation cannot establish mathematical correctness.  Every label in
this module remains a proposal until a qualified human reviewer records a
disposition.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from math_feedback_ai.domain import (
    DiagnosisV1,
    Problem,
    StudentAttempt,
    TutorDecision,
    TutoringExampleV1,
)

PilotId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HumanReviewStatus(StrEnum):
    """Lifecycle state of a provisional annotation."""

    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    REVIEWED = "reviewed"


class PilotProvenance(BaseModel):
    """Origin and research-use warning for a pilot case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: NonBlank
    authoring_method: Literal["codex_proposed"] = "codex_proposed"
    synthetic: Literal[True] = True
    license: NonBlank = "CC0-1.0"
    research_use: Literal["provisional_only"] = "provisional_only"
    notes: NonBlank


class DiagnosisPilotCaseV1(BaseModel):
    """One proposed pilot annotation, distinct from canonical gold examples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["diagnosis_pilot.v1"] = "diagnosis_pilot.v1"
    case_id: PilotId
    category: PilotId
    mathematical_domains: Annotated[tuple[PilotId, ...], Field(min_length=1)]
    characteristics: Annotated[tuple[PilotId, ...], Field(min_length=1)]
    provenance: PilotProvenance
    human_review_status: HumanReviewStatus
    problem: Problem
    student_attempt: StudentAttempt
    proposed_diagnosis: DiagnosisV1
    proposed_decision: TutorDecision
    annotation_explanation: NonBlank
    ambiguity_notes: NonBlank | None = None

    @field_validator("mathematical_domains", "characteristics")
    @classmethod
    def _metadata_values_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("pilot metadata values must be unique")
        return value

    @model_validator(mode="after")
    def _references_are_consistent(self) -> Self:
        if self.human_review_status is not HumanReviewStatus.PENDING:
            raise ValueError("v1 Codex-proposed pilot cases must remain pending human review")
        if self.student_attempt.problem_id != self.problem.problem_id:
            raise ValueError("student attempt must reference the pilot problem")
        if self.proposed_diagnosis.attempt_id != self.student_attempt.attempt_id:
            raise ValueError("proposed diagnosis must reference the pilot attempt")
        step_ids = tuple(step.step_id for step in self.student_attempt.steps)
        assessment_ids = tuple(
            assessment.step_id for assessment in self.proposed_diagnosis.step_assessments
        )
        if assessment_ids != step_ids:
            raise ValueError("proposed assessments must cover every parsed step in source order")
        known_ids = set(step_ids)
        target = self.proposed_decision.target_step_id
        if target is not None and target not in known_ids:
            raise ValueError("proposed decision targets an unknown step")
        if self.ambiguity_notes is not None and "ambiguous" not in self.characteristics:
            raise ValueError("ambiguity notes require the ambiguous characteristic")
        if "ambiguous" in self.characteristics and self.ambiguity_notes is None:
            raise ValueError("ambiguous cases require adjudication notes")
        return self


class ReviewedPilotProvenance(BaseModel):
    """Traceable review provenance without claiming ecological validity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: NonBlank
    original_provisional_version: Literal["diagnosis_pilot_v1"] = "diagnosis_pilot_v1"
    original_provisional_sha256: Literal[
        "42ff58f9f5694c36385aa395ed44fd2ee711c421c83cff73c8327c6b3d6c77d5"
    ] = "42ff58f9f5694c36385aa395ed44fd2ee711c421c83cff73c8327c6b3d6c77d5"
    review_basis: Literal["independent_human_mathematical_adjudication"] = (
        "independent_human_mathematical_adjudication"
    )
    synthetic: Literal[True] = True
    license: NonBlank = "CC0-1.0"
    research_use: Literal["human_reviewed_synthetic_benchmark"] = (
        "human_reviewed_synthetic_benchmark"
    )
    notes: NonBlank


class DiagnosisPilotReviewedCaseV1(BaseModel):
    """One reconciled human-reviewed annotation derived from the provisional pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["diagnosis_pilot.reviewed.v1"] = "diagnosis_pilot.reviewed.v1"
    benchmark_version: Literal["diagnosis_pilot_v1_reviewed"] = "diagnosis_pilot_v1_reviewed"
    case_id: PilotId
    category: PilotId
    mathematical_domains: Annotated[tuple[PilotId, ...], Field(min_length=1)]
    characteristics: Annotated[tuple[PilotId, ...], Field(min_length=1)]
    provenance: ReviewedPilotProvenance
    human_review_status: Literal["reviewed"] = "reviewed"
    review_disposition: Literal["approved", "modified"]
    problem: Problem
    student_attempt: StudentAttempt
    reviewed_diagnosis: DiagnosisV1
    reviewed_decision: TutorDecision
    annotation_explanation: NonBlank
    ambiguity_notes: NonBlank | None = None

    @field_validator("mathematical_domains", "characteristics")
    @classmethod
    def _metadata_values_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("reviewed pilot metadata values must be unique")
        return value

    @model_validator(mode="after")
    def _references_are_consistent(self) -> Self:
        if self.student_attempt.problem_id != self.problem.problem_id:
            raise ValueError("student attempt must reference the reviewed problem")
        if self.reviewed_diagnosis.schema_version != "1.1":
            raise ValueError("reviewed diagnosis must use corrected schema 1.1")
        if self.reviewed_diagnosis.attempt_id != self.student_attempt.attempt_id:
            raise ValueError("reviewed diagnosis must reference the reviewed attempt")
        step_ids = tuple(step.step_id for step in self.student_attempt.steps)
        assessment_ids = tuple(
            assessment.step_id for assessment in self.reviewed_diagnosis.step_assessments
        )
        if assessment_ids != step_ids:
            raise ValueError("reviewed assessments must cover every parsed step in source order")
        target = self.reviewed_decision.target_step_id
        if target is not None and target not in set(step_ids):
            raise ValueError("reviewed decision targets an unknown step")
        if self.ambiguity_notes is not None and "ambiguous" not in self.characteristics:
            raise ValueError("ambiguity notes require the ambiguous characteristic")
        if "ambiguous" in self.characteristics and self.ambiguity_notes is None:
            raise ValueError("ambiguous cases require adjudication notes")
        return self


type DiagnosisPilotCase = DiagnosisPilotCaseV1 | DiagnosisPilotReviewedCaseV1


class PilotValidationIssue(BaseModel):
    """One deterministic structural finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: PilotId
    case_ids: Annotated[tuple[PilotId, ...], Field(min_length=1)]
    message: NonBlank


class PilotValidationReport(BaseModel):
    """Results of non-mathematical benchmark validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: Annotated[int, Field(ge=0)]
    category_counts: dict[str, int]
    domain_counts: dict[str, int]
    issues: tuple[PilotValidationIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues


_SPACE_OR_PUNCTUATION = re.compile(r"[^a-z0-9]+")


def _normalized(text: str) -> str:
    return " ".join(_SPACE_OR_PUNCTUATION.sub(" ", text.casefold()).split())


def _near_duplicate(left: str, right: str, *, threshold: float) -> bool:
    normalized_left = _normalized(left)
    normalized_right = _normalized(right)
    if normalized_left == normalized_right:
        return True
    if min(len(normalized_left), len(normalized_right)) < 40:
        return False
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= threshold


def _reference_leaks(case: DiagnosisPilotCase) -> bool:
    student = _normalized(case.student_attempt.raw_text)
    student_tokens = student.split()
    for reference in case.problem.reference_solutions:
        reference_text = _normalized(reference.text)
        reference_tokens = reference_text.split()
        if len(reference_tokens) >= 8 and reference_text in student:
            return True
        # A long verbatim run is unlikely to be incidental, while short shared
        # equations and theorem names are expected in mathematical solutions.
        window = 12
        if len(reference_tokens) >= window and len(student_tokens) >= window:
            student_ngrams = {
                tuple(student_tokens[index : index + window])
                for index in range(len(student_tokens) - window + 1)
            }
            if any(
                tuple(reference_tokens[index : index + window]) in student_ngrams
                for index in range(len(reference_tokens) - window + 1)
            ):
                return True
    return False


def validate_pilot_cases(
    cases: Sequence[DiagnosisPilotCase],
    *,
    comparison_examples: Iterable[TutoringExampleV1] = (),
) -> PilotValidationReport:
    """Validate structure, duplication, and leakage without judging mathematics."""

    issues: list[PilotValidationIssue] = []
    ids = [case.case_id for case in cases]
    for case_id, count in Counter(ids).items():
        if count > 1:
            issues.append(
                PilotValidationIssue(
                    code="duplicate.case_id",
                    case_ids=(case_id,),
                    message="Case ID appears more than once.",
                )
            )

    for left_index, left in enumerate(cases):
        if _reference_leaks(left):
            issues.append(
                PilotValidationIssue(
                    code="leakage.reference_into_student",
                    case_ids=(left.case_id,),
                    message="Student text contains a long verbatim fragment of a reference.",
                )
            )
        for right in cases[left_index + 1 :]:
            if _near_duplicate(left.problem.statement, right.problem.statement, threshold=0.92):
                issues.append(
                    PilotValidationIssue(
                        code="duplicate.problem",
                        case_ids=(left.case_id, right.case_id),
                        message="Pilot problem statements are exact or near duplicates.",
                    )
                )
            if _near_duplicate(
                left.student_attempt.raw_text,
                right.student_attempt.raw_text,
                threshold=0.94,
            ):
                issues.append(
                    PilotValidationIssue(
                        code="duplicate.student_solution",
                        case_ids=(left.case_id, right.case_id),
                        message="Pilot student solutions are exact or near duplicates.",
                    )
                )

    for case in cases:
        for comparison in comparison_examples:
            if _near_duplicate(
                case.problem.statement, comparison.problem.statement, threshold=0.92
            ):
                issues.append(
                    PilotValidationIssue(
                        code="duplicate.problem_across_splits",
                        case_ids=(case.case_id, comparison.example_id),
                        message="Pilot problem duplicates a comparison-split problem.",
                    )
                )

    return PilotValidationReport(
        case_count=len(cases),
        category_counts=dict(sorted(Counter(case.category for case in cases).items())),
        domain_counts=dict(
            sorted(
                Counter(domain for case in cases for domain in case.mathematical_domains).items()
            )
        ),
        issues=tuple(issues),
    )


__all__ = [
    "DiagnosisPilotCase",
    "DiagnosisPilotCaseV1",
    "DiagnosisPilotReviewedCaseV1",
    "HumanReviewStatus",
    "PilotProvenance",
    "PilotValidationIssue",
    "PilotValidationReport",
    "ReviewedPilotProvenance",
    "validate_pilot_cases",
]
