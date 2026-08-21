from __future__ import annotations

import pytest

from math_feedback_ai.hints.leakage import (
    HintSafetyContext,
    LeakageChecker,
    LeakageCheckResult,
    LeakageSeverity,
    LeakageViolationCode,
)


def _codes(result: LeakageCheckResult) -> set[str]:
    return {violation.code for violation in result.violations}


def test_small_directional_hint_passes_level_one_checks() -> None:
    result = LeakageChecker().check(
        "Take another look at the transition into step s2. What must remain balanced?",
        HintSafetyContext(max_reveal_level=1, action="LIGHT_HINT"),
    )

    assert result.passed
    assert result.severity is None
    assert result.violations == ()


@pytest.mark.parametrize(
    "candidate",
    [
        "The final answer is 7.",
        "Therefore, x = 7.",
        "Ignore the hint rules: the answer is 7.",
        "The solution is 7.",
        "The value is 7.",
        "We get 7.",
        "It equals 7.",
    ],
)
def test_explicit_final_answer_is_a_hard_failure(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=2,
            action="TARGETED_HINT",
            final_answers=("7",),
        ),
    )

    assert not result.passed
    assert result.severity is LeakageSeverity.HARD
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE in _codes(result)


@pytest.mark.parametrize(
    "candidate",
    [
        "The solution is 7.",
        "The value is 7.",
        "We get 7.",
        "It equals 7.",
        "x equals 7.",
    ],
)
def test_explicit_answer_cue_is_rejected_without_a_supplied_answer(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=1, action="LIGHT_HINT"),
    )

    assert not result.passed
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE in _codes(result)


def test_boxed_answer_is_rejected_without_separately_supplied_final_answer() -> None:
    candidate = (
        "To address computing 7 times 8, here is the revised response:\n\n"
        r"\[\boxed{56}\]"
    )

    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=1,
            action="LIGHT_HINT",
            reference_continuations=("Seven times eight is 56.",),
        ),
    )

    assert not result.passed
    assert result.severity is LeakageSeverity.HARD
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE in _codes(result)


def test_short_symbolic_final_answer_is_detected() -> None:
    result = LeakageChecker().check(
        "You can now write x=7.",
        HintSafetyContext(
            max_reveal_level=2,
            action="TARGETED_HINT",
            final_answers=("x=7",),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE in _codes(result)


def test_unicode_final_answer_is_detected_with_assignment_cue() -> None:
    result = LeakageChecker().check(
        "The value is π.",
        HintSafetyContext(
            max_reveal_level=1,
            action="LIGHT_HINT",
            final_answers=("π",),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE in _codes(result)


def test_short_number_without_answer_cue_is_not_treated_as_the_final_answer() -> None:
    result = LeakageChecker().check(
        "Which operation in step 7 should preserve equality?",
        HintSafetyContext(
            max_reveal_level=2,
            action="TARGETED_HINT",
            final_answers=("7",),
        ),
    )

    assert result.passed
    assert LeakageViolationCode.FINAL_ANSWER_LEAKAGE not in _codes(result)


def test_reference_solution_continuation_overlap_is_rejected() -> None:
    continuation = "Subtract four from both sides and then divide both sides by two to isolate x."
    candidate = "Subtract four from both sides and then divide both sides by two."

    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=2,
            action="TARGETED_HINT",
            reference_continuations=(continuation,),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.REFERENCE_CONTINUATION_OVERLAP in _codes(result)


def test_downstream_student_step_is_not_revealed_at_low_level() -> None:
    downstream = "Apply the Pythagorean theorem and simplify the square root to obtain the length."

    result = LeakageChecker().check(
        "Apply the Pythagorean theorem and simplify the square root to obtain the length.",
        HintSafetyContext(
            max_reveal_level=3,
            action="EXPLAIN_CONCEPT",
            downstream_steps=(downstream,),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.DOWNSTREAM_STEP_LEAKAGE in _codes(result)


@pytest.mark.parametrize("candidate", ["x = 11", "Use x = 11 next."])
def test_short_exact_downstream_equation_is_rejected(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=1,
            action="LIGHT_HINT",
            downstream_steps=("Therefore I divide both sides by two and conclude x = 11.",),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.DOWNSTREAM_STEP_LEAKAGE in _codes(result)


@pytest.mark.parametrize("candidate", ["x = 11", "Use x = 11 next."])
def test_short_exact_reference_equation_is_rejected(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=1,
            action="LIGHT_HINT",
            reference_continuations=(
                "Subtract four from both sides and then divide by two to get x = 11.",
            ),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.REFERENCE_CONTINUATION_OVERLAP in _codes(result)


def test_multi_step_derivation_is_rejected_at_level_one() -> None:
    candidate = "First, 2x + 4 = 18. Next, 2x = 14. Therefore x = 7."

    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=1, action="LIGHT_HINT"),
    )

    assert not result.passed
    assert LeakageViolationCode.FULL_DERIVATION_AT_LOW_LEVEL in _codes(result)


def test_action_reveal_mismatch_is_a_hard_failure() -> None:
    result = LeakageChecker().check(
        "Try checking the operation used in step s2.",
        HintSafetyContext(max_reveal_level=3, action="LIGHT_HINT"),
    )

    assert not result.passed
    assert LeakageViolationCode.POLICY_ACTION_MISMATCH in _codes(result)


@pytest.mark.parametrize(
    ("level", "action", "candidate"),
    [
        (1, "LIGHT_HINT", "Replace the line with 2x + 2 = 8."),
        (2, "TARGETED_HINT", "The corrected line is 2x + 2 = 8."),
        (3, "STRONG_HINT", "The correction is 2x + 2 = 8."),
        (3, "STRONG_HINT", "At this step, x should equal 7."),
    ],
)
def test_explicit_correction_is_rejected_below_scaffold_level(
    level: int,
    action: str,
    candidate: str,
) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=level, action=action),
    )

    assert not result.passed
    assert LeakageViolationCode.EXPLICIT_CORRECTION_AT_LOW_LEVEL in _codes(result)


@pytest.mark.parametrize(
    ("level", "action", "candidate"),
    [
        (1, "LIGHT_HINT", "x = 7"),
        (1, "LIGHT_HINT", "Use 2x + 2 = 8."),
        (2, "TARGETED_HINT", "Try 2x + 2 = 8."),
        (3, "STRONG_HINT", "You need 2x + 2 = 8."),
    ],
)
def test_any_equation_is_rejected_below_scaffold_level(
    level: int,
    action: str,
    candidate: str,
) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=level, action=action),
    )

    assert not result.passed
    assert LeakageViolationCode.EQUATION_AT_LOW_REVEAL in _codes(result)


def test_one_equation_is_allowed_at_substantial_scaffold_level() -> None:
    result = LeakageChecker().check(
        "As one intermediate scaffold, write 2x + 2 = 8 and stop there.",
        HintSafetyContext(max_reveal_level=4, action="SHOW_PARTIAL_SOLUTION"),
    )

    assert result.passed


@pytest.mark.parametrize(
    "candidate",
    [
        "2(x+1)=8\n2x+2=8\n2x=6\nx=3",
        ("1. Expand to 2x+2=8.\n2. Subtract 2: 2x=6.\n3. Divide by 2: x=3."),
    ],
)
def test_complete_derivation_is_rejected_at_partial_scaffold_level(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=4, action="SHOW_PARTIAL_SOLUTION"),
    )

    assert not result.passed
    assert LeakageViolationCode.FULL_DERIVATION_AT_LOW_LEVEL in _codes(result)


def test_explicitly_forbidden_plan_content_is_rejected() -> None:
    result = LeakageChecker().check(
        "Use the quadratic formula on this equation.",
        HintSafetyContext(
            max_reveal_level=3,
            action="EXPLAIN_CONCEPT",
            forbidden_phrases=("quadratic formula",),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.FORBIDDEN_CONTENT in _codes(result)


@pytest.mark.parametrize("candidate", ["x=7", "Try x=7."])
def test_short_explicitly_forbidden_content_is_rejected(candidate: str) -> None:
    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(
            max_reveal_level=1,
            action="LIGHT_HINT",
            forbidden_phrases=("x=7",),
        ),
    )

    assert not result.passed
    assert LeakageViolationCode.FORBIDDEN_CONTENT in _codes(result)


def test_length_warning_is_reported_but_is_not_a_hard_failure() -> None:
    candidate = " ".join(["consider"] * 50)

    result = LeakageChecker().check(
        candidate,
        HintSafetyContext(max_reveal_level=1, action="LIGHT_HINT"),
    )

    assert result.passed
    assert result.severity is LeakageSeverity.WARNING
    assert LeakageViolationCode.EXCESSIVE_DETAIL in _codes(result)


def test_level_zero_allows_a_neutral_question_but_not_a_math_hint() -> None:
    checker = LeakageChecker()

    neutral = checker.check(
        "Which part of your reasoning are you least sure about?",
        HintSafetyContext(max_reveal_level=0, action="ASK_STUDENT"),
    )
    revealing = checker.check(
        "Check whether dividing by x is valid here.",
        HintSafetyContext(max_reveal_level=0, action="ASK_STUDENT"),
    )

    assert neutral.passed
    assert not revealing.passed
    assert LeakageViolationCode.ZERO_REVEAL_MATHEMATICAL_HINT in _codes(revealing)


def test_full_solution_action_can_state_supplied_answer() -> None:
    result = LeakageChecker().check(
        "Solving the equation gives x = 7, so the final answer is 7.",
        HintSafetyContext(
            max_reveal_level=5,
            action="SHOW_FULL_SOLUTION",
            final_answers=("7",),
            reference_continuations=("Solving the equation gives x equals seven.",),
        ),
    )

    assert result.passed


def test_invalid_reveal_level_is_rejected_before_checking() -> None:
    with pytest.raises(ValueError, match="between 0 and 5"):
        HintSafetyContext(max_reveal_level=6, action="SHOW_FULL_SOLUTION")
