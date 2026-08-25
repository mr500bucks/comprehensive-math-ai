"""Small, versioned taxonomies used by the tutoring pipeline.

The values in this module are serialized contracts.  Additions are backwards
compatible, but renaming or repurposing a value requires a schema version bump.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, Literal

ISSUE_TAXONOMY_VERSION: Final[Literal["1.0"]] = "1.0"


class OverallStatus(StrEnum):
    """Mathematical status of the student's attempt as a whole."""

    FULLY_CORRECT = "fully_correct"
    CORRECT_BUT_INEFFICIENT = "correct_but_inefficient"
    INCOMPLETE = "incomplete"
    INCORRECT = "incorrect"
    INDETERMINATE = "indeterminate"


class StepStatus(StrEnum):
    """Status of an individual student-authored step."""

    VALID = "valid"
    VALID_BUT_INEFFICIENT = "valid_but_inefficient"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    DEPENDENT_ON_PREVIOUS_ERROR = "dependent_on_previous_error"


class IssueCode(StrEnum):
    """Compact hierarchical issue taxonomy, version 1.0."""

    REASONING_LOGICAL_GAP = "reasoning.logical_gap"
    REASONING_INVALID_INFERENCE = "reasoning.invalid_inference"
    JUSTIFICATION_UNJUSTIFIED_CLAIM = "justification.unjustified_claim"
    CONDITION_MISSING = "condition.missing"
    COMPUTATION_ALGEBRAIC = "computation.algebraic"
    COMPUTATION_ARITHMETIC = "computation.arithmetic"
    THEOREM_MISUSED = "theorem.misused"
    CONCEPT_MISUNDERSTOOD = "concept.misunderstood"
    RELEVANCE_IRRELEVANT = "relevance.irrelevant"
    COMPLETION_INCOMPLETE = "completion.incomplete"
    UNKNOWN_INSUFFICIENT_CONFIDENCE = "unknown.insufficient_confidence"


class RevealLevel(IntEnum):
    """Maximum amount of mathematical information a response may reveal."""

    NONE = 0
    LIGHT_DIRECTION = 1
    LOCATION = 2
    CONCEPT = 3
    SUBSTANTIAL_SCAFFOLD = 4
    FULL_SOLUTION = 5


class TutorAction(StrEnum):
    """Transparent action selected by the deterministic tutoring policy."""

    WAIT = "WAIT"
    ASK_STUDENT = "ASK_STUDENT"
    VERIFY_STEP = "VERIFY_STEP"
    LIGHT_HINT = "LIGHT_HINT"
    TARGETED_HINT = "TARGETED_HINT"
    STRONG_HINT = "STRONG_HINT"
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    SHOW_PARTIAL_SOLUTION = "SHOW_PARTIAL_SOLUTION"
    SHOW_FULL_SOLUTION = "SHOW_FULL_SOLUTION"


class TutorMode(StrEnum):
    """System-level tutoring mode; a student request cannot raise its ceiling."""

    HINT_ONLY = "hint_only"
    GUIDED = "guided"
    FULL_SOLUTION_ALLOWED = "full_solution_allowed"


class ReferenceRelation(StrEnum):
    """How the attempt relates to optional, non-exhaustive references."""

    NOT_USED = "not_used"
    SAME_METHOD = "same_method"
    ALTERNATIVE_PLAUSIBLE = "alternative_plausible"
    ALTERNATIVE_VERIFIED = "alternative_verified"


class StepParseStrategy(StrEnum):
    """Deterministic strategy used to segment a raw student attempt."""

    EXPLICIT_NUMBERING = "explicit_numbering"
    PARAGRAPHS = "paragraphs"
    ALGEBRA_LINES = "algebra_lines"
    SINGLE_CHUNK = "single_chunk"


class LeakageSeverity(StrEnum):
    """Severity assigned to an interpretable leakage or policy violation."""

    INFO = "info"
    WARNING = "warning"
    HARD = "hard"
