"""Interpretable, conservative checks for hint leakage and policy violations.

These checks are deliberately simple. They enforce several high-signal safety
properties, but they are not a proof that a hint contains no mathematical
leakage. The structured result makes that limitation visible to callers and to
the evaluation harness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from math_feedback_ai.domain import LeakageCheckResult, LeakageSeverity, LeakageViolation


class LeakageViolationCode(StrEnum):
    """Stable, machine-readable codes emitted by :class:`LeakageChecker`."""

    EMPTY_RESPONSE = "empty_response"
    POLICY_ACTION_MISMATCH = "policy_action_mismatch"
    ZERO_REVEAL_MATHEMATICAL_HINT = "zero_reveal_mathematical_hint"
    FINAL_ANSWER_LEAKAGE = "final_answer_leakage"
    REFERENCE_CONTINUATION_OVERLAP = "reference_continuation_overlap"
    DOWNSTREAM_STEP_LEAKAGE = "downstream_step_leakage"
    FULL_DERIVATION_AT_LOW_LEVEL = "full_derivation_at_low_level"
    EQUATION_AT_LOW_REVEAL = "equation_at_low_reveal"
    EXPLICIT_CORRECTION_AT_LOW_LEVEL = "explicit_correction_at_low_level"
    FORBIDDEN_CONTENT = "forbidden_content"
    EXCESSIVE_DETAIL = "excessive_detail"


@dataclass(frozen=True, slots=True)
class HintSafetyContext:
    """Private material used to validate, but not necessarily generate, a hint.

    Separating this context from the generation prompt lets an orchestrator keep
    reference continuations and final answers away from low-reveal generations
    while still checking the resulting candidate against them.
    """

    max_reveal_level: int
    action: object
    final_answers: tuple[str, ...] = ()
    reference_continuations: tuple[str, ...] = ()
    downstream_steps: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        level = _reveal_value(self.max_reveal_level)
        if not 0 <= level <= 5:
            raise ValueError("max_reveal_level must be between 0 and 5")
        object.__setattr__(self, "max_reveal_level", level)


_ACTION_REVEAL_BOUNDS: dict[str, tuple[int, int]] = {
    "WAIT": (0, 0),
    "ASK_STUDENT": (0, 0),
    "VERIFY_STEP": (0, 2),
    "LIGHT_HINT": (1, 1),
    "TARGETED_HINT": (2, 3),
    "STRONG_HINT": (3, 4),
    "EXPLAIN_CONCEPT": (3, 4),
    "SHOW_PARTIAL_SOLUTION": (4, 4),
    "SHOW_FULL_SOLUTION": (5, 5),
}

_WORD_RE = re.compile(r"\w+(?:['’-]\w+)?", re.UNICODE)
_NUMBER_OR_EXPRESSION_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)(?![\w.])")
_FINAL_ANSWER_CUE_RE = re.compile(
    r"\\(?:boxed|fbox)\s*\{[^{}\n]+\}|"
    r"\b(?:final\s+answer|answer|result|solution|value)\s*(?:is|=|:)\s*\S|"
    r"\bwe\s+get\s+\S|\bit\s+(?:equals|is)\s+\S|"
    r"\b\w+\s+equals\s+[-+\wπ]|"
    r"\b(?:therefore|thus|hence|so)\b[^\n]{0,40}(?:=|is)\s*[-+\wπ]",
    re.IGNORECASE,
)
_STEP_MARKER_RE = re.compile(r"(?m)^\s*(?:step\s+)?\d+[.) :]\s*\S+", re.IGNORECASE)
_MATH_TRANSITION_RE = re.compile(
    r"\b(?:substitut(?:e|ing)|simplif(?:y|ying)|solv(?:e|ing)|therefore|"
    r"hence|thus|which\s+gives|it\s+follows)\b",
    re.IGNORECASE,
)
_EXPLICIT_CORRECTION_RE = re.compile(
    r"\b(?:replace|corrected|correction|should\s+(?:be|equal)|"
    r"instead\s+(?:write|use)|write\b[^\n]{0,30}\binstead)\b",
    re.IGNORECASE,
)


def _enum_name(value: object) -> str:
    """Return an enum name or a normalized string without coupling domains."""

    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def _reveal_value(value: object) -> int:
    raw: object = getattr(value, "value", value)
    try:
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            return int(raw)
        raise TypeError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid reveal level: {value!r}") from exc


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _compact_math(text: str) -> str:
    text = text.lower().replace("\\left", "").replace("\\right", "")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"[^\w=+*/^().-]+", "", text)


def _longest_token_overlap(candidate: str, source: str) -> tuple[int, float, str]:
    candidate_tokens = _tokens(candidate)
    source_tokens = _tokens(source)
    if not candidate_tokens or not source_tokens:
        return 0, 0.0, ""
    match = SequenceMatcher(
        None, candidate_tokens, source_tokens, autojunk=False
    ).find_longest_match()
    overlap = candidate_tokens[match.a : match.a + match.size]
    ratio = match.size / min(len(candidate_tokens), len(source_tokens))
    return match.size, ratio, " ".join(overlap)


def _contains_final_answer(candidate: str, answer: str) -> tuple[bool, str | None]:
    answer = answer.strip()
    if not answer:
        return False, None

    compact_candidate = _compact_math(candidate)
    compact_answer = _compact_math(answer)
    if not compact_answer:
        return False, None

    literal = re.escape(answer).replace(r"\ ", r"\s*")
    assigned_answer = re.compile(
        rf"(?:\b(?:final\s+answer|answer|result|solution|value)\s*"
        rf"(?:is|=|:)\s*|\bwe\s+get\s+|\bit\s+(?:equals|is)\s+)"
        rf"{literal}(?!\w)",
        re.IGNORECASE,
    )
    if assigned_answer.search(candidate):
        return True, answer

    # Longer symbolic answers are unlikely to occur accidentally. Short numeric
    # answers require an answer/equality cue to avoid flagging an innocent step ID.
    if (len(compact_answer) >= 4 or "=" in compact_answer) and compact_answer in compact_candidate:
        return True, answer

    if _NUMBER_OR_EXPRESSION_RE.fullmatch(answer.strip()):
        escaped = re.escape(answer.strip()).replace(r"\ ", r"\s*")
        cue_pattern = re.compile(
            rf"(?:\b(?:answer|result)\s*(?:is|=|:)\s*|"
            rf"\b(?:therefore|thus|hence|so)\b[^\n]{{0,40}}(?:=|is)\s*)"
            rf"{escaped}(?![\w.])",
            re.IGNORECASE,
        )
        if cue_pattern.search(candidate):
            return True, answer

        equality_pattern = re.compile(rf"=\s*{escaped}(?:\s*[.!]|\s*$)", re.IGNORECASE)
        if equality_pattern.search(candidate):
            return True, answer

    if compact_candidate == compact_answer:
        return True, answer
    return False, None


def _maximum_severity(
    violations: Iterable[LeakageViolation],
) -> LeakageSeverity | None:
    severities = {violation.severity for violation in violations}
    if LeakageSeverity.HARD in severities:
        return LeakageSeverity.HARD
    if LeakageSeverity.WARNING in severities:
        return LeakageSeverity.WARNING
    if LeakageSeverity.INFO in severities:
        return LeakageSeverity.INFO
    return None


class LeakageChecker:
    """Apply high-signal leakage and policy checks to one generated response."""

    def check(self, candidate: str, context: HintSafetyContext) -> LeakageCheckResult:
        violations: list[LeakageViolation] = []
        candidate = candidate.strip()
        level = context.max_reveal_level
        action = _enum_name(context.action)

        if not candidate:
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.EMPTY_RESPONSE,
                    severity=LeakageSeverity.HARD,
                    message="The hint generator returned an empty response.",
                )
            )

        bounds = _ACTION_REVEAL_BOUNDS.get(action)
        if bounds is None or not bounds[0] <= level <= bounds[1]:
            expected = (
                "a known action" if bounds is None else f"reveal level {bounds[0]}..{bounds[1]}"
            )
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.POLICY_ACTION_MISMATCH,
                    severity=LeakageSeverity.HARD,
                    message=f"Action {action!r} is incompatible with reveal level {level}; expected {expected}.",
                )
            )

        if candidate and level == 0 and self._contains_mathematical_hint(candidate):
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.ZERO_REVEAL_MATHEMATICAL_HINT,
                    severity=LeakageSeverity.HARD,
                    message="A reveal-level 0 response contains mathematical guidance.",
                )
            )

        if level < 5:
            for answer in context.final_answers:
                leaked, _ = _contains_final_answer(candidate, answer)
                if leaked:
                    violations.append(
                        LeakageViolation(
                            code=LeakageViolationCode.FINAL_ANSWER_LEAKAGE,
                            severity=LeakageSeverity.HARD,
                            message="The response appears to state an explicitly supplied final answer.",
                        )
                    )
                    break

            if _FINAL_ANSWER_CUE_RE.search(candidate):
                violations.append(
                    LeakageViolation(
                        code=LeakageViolationCode.FINAL_ANSWER_LEAKAGE,
                        severity=LeakageSeverity.HARD,
                        message="The response uses an explicit final-answer construction below reveal level 5.",
                    )
                )

        if level <= 3 and "=" in candidate:
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.EQUATION_AT_LOW_REVEAL,
                    severity=LeakageSeverity.HARD,
                    message=(
                        "An equation is not permitted below the substantial-scaffold reveal level."
                    ),
                )
            )

        for phrase in context.forbidden_phrases:
            compact_phrase = _compact_math(phrase)
            if compact_phrase and compact_phrase in _compact_math(candidate):
                violations.append(
                    LeakageViolation(
                        code=LeakageViolationCode.FORBIDDEN_CONTENT,
                        severity=LeakageSeverity.HARD,
                        message="The response contains content explicitly forbidden by the hint plan.",
                    )
                )
                break

        if (
            level <= 3
            and _EXPLICIT_CORRECTION_RE.search(candidate)
            and (
                "=" in candidate
                or bool(
                    re.search(
                        r"\bshould\s+(?:be|equal)\s+[-+\w]",
                        candidate,
                        re.IGNORECASE,
                    )
                )
            )
        ):
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.EXPLICIT_CORRECTION_AT_LOW_LEVEL,
                    severity=LeakageSeverity.HARD,
                    message=(
                        "The response states an explicit correction below the authorized "
                        "scaffold level."
                    ),
                )
            )

        self._check_source_overlap(
            candidate,
            context.reference_continuations,
            level=level,
            code=LeakageViolationCode.REFERENCE_CONTINUATION_OVERLAP,
            label="reference-solution continuation",
            violations=violations,
        )
        self._check_source_overlap(
            candidate,
            context.downstream_steps,
            level=level,
            code=LeakageViolationCode.DOWNSTREAM_STEP_LEAKAGE,
            label="student-work continuation",
            violations=violations,
        )

        if self._looks_like_full_derivation(candidate, level):
            violations.append(
                LeakageViolation(
                    code=LeakageViolationCode.FULL_DERIVATION_AT_LOW_LEVEL,
                    severity=LeakageSeverity.HARD,
                    message=f"The response resembles a derivation that exceeds reveal level {level}.",
                )
            )

        self._check_length(candidate, level, violations)
        violations = self._deduplicate(violations)
        severity = _maximum_severity(violations)
        return LeakageCheckResult(
            passed=severity is not LeakageSeverity.HARD,
            violations=tuple(violations),
            severity=severity,
        )

    @staticmethod
    def _contains_mathematical_hint(candidate: str) -> bool:
        equation_like = bool(re.search(r"(?:\w|\d)\s*[=+*/^]\s*(?:\w|\d)", candidate))
        directive = bool(
            re.search(
                r"\b(?:revisit|check|apply|use|factor|differentiate|integrate|substitute|divide|multiply)\b",
                candidate,
                re.IGNORECASE,
            )
        )
        return equation_like or directive

    @staticmethod
    def _looks_like_full_derivation(candidate: str, level: int) -> bool:
        if level >= 5:
            return False
        equations = len(re.findall(r"(?<![<>])=(?!=)", candidate))
        step_markers = len(_STEP_MARKER_RE.findall(candidate))
        transitions = len(_MATH_TRANSITION_RE.findall(candidate))
        align_environment = "\\begin{align" in candidate.lower()
        if level <= 2:
            return (
                align_environment
                or step_markers >= 2
                or equations >= 2
                or (equations >= 1 and transitions >= 2)
            )
        if level == 3:
            return align_environment or step_markers >= 3 or equations >= 3
        if level == 4:
            return align_environment or step_markers >= 2 or equations >= 2
        return False

    @staticmethod
    def _check_source_overlap(
        candidate: str,
        sources: tuple[str, ...],
        *,
        level: int,
        code: LeakageViolationCode,
        label: str,
        violations: list[LeakageViolation],
    ) -> None:
        if level >= 5:
            return
        for source in sources:
            run, ratio, overlap = _longest_token_overlap(candidate, source)
            candidate_tokens = " ".join(_tokens(candidate))
            source_token_list = _tokens(source)
            source_tokens = " ".join(source_token_list)
            compact_candidate = _compact_math(candidate)
            compact_source = _compact_math(source)
            overlap_has_equation = (
                run >= 2
                and "=" in candidate
                and "=" in source
                and (any(character.isdigit() for character in overlap) or ratio >= 0.4)
            )
            exact_math = (
                len(compact_source) >= 3
                and bool(re.search(r"[0-9=+*/^().-]", compact_source))
                and compact_source in compact_candidate
            )
            exact_downstream_prose = (
                code is LeakageViolationCode.DOWNSTREAM_STEP_LEAKAGE
                and len(source_token_list) >= 3
                and bool(source_tokens)
                and source_tokens in candidate_tokens
            )
            short_sensitive_match = overlap_has_equation or exact_math or exact_downstream_prose
            if level <= 3:
                hard = short_sensitive_match or run >= 8 or (run >= 5 and ratio >= 0.72)
            else:
                hard = short_sensitive_match or run >= 14 or (run >= 8 and ratio >= 0.9)
            if hard:
                violations.append(
                    LeakageViolation(
                        code=code,
                        severity=LeakageSeverity.HARD,
                        message=f"The response copies too much of a {label}.",
                    )
                )
                return

    @staticmethod
    def _check_length(
        candidate: str,
        level: int,
        violations: list[LeakageViolation],
    ) -> None:
        soft_limits = {0: 35, 1: 45, 2: 80, 3: 130, 4: 240}
        limit = soft_limits.get(level)
        if limit is None:
            return
        word_count = len(_tokens(candidate))
        if word_count > limit * 2:
            severity = LeakageSeverity.HARD
        elif word_count > limit:
            severity = LeakageSeverity.WARNING
        else:
            return
        violations.append(
            LeakageViolation(
                code=LeakageViolationCode.EXCESSIVE_DETAIL,
                severity=severity,
                message=f"The response has {word_count} words; reveal level {level} has a {limit}-word soft limit.",
            )
        )

    @staticmethod
    def _deduplicate(violations: list[LeakageViolation]) -> list[LeakageViolation]:
        result: list[LeakageViolation] = []
        seen: set[str] = set()
        for violation in violations:
            if violation.code not in seen:
                seen.add(violation.code)
                result.append(violation)
        return result


__all__ = [
    "HintSafetyContext",
    "LeakageCheckResult",
    "LeakageChecker",
    "LeakageSeverity",
    "LeakageViolation",
    "LeakageViolationCode",
]
