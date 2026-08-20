from __future__ import annotations

import pytest

from math_feedback_ai.domain import StepParseStrategy
from math_feedback_ai.parsing import parse_solution_steps, parse_student_attempt


def assert_source_is_preserved(raw: str) -> None:
    result = parse_solution_steps(raw)
    cursor = 0
    for step in result.steps:
        assert step.text == raw[step.start_offset : step.end_offset]
        assert not raw[cursor : step.start_offset].strip()
        cursor = step.end_offset
    assert not raw[cursor:].strip()


def test_explicitly_numbered_solution_is_split_in_order() -> None:
    raw = "1. Subtract 1 from both sides.\n2. Therefore x = 1.\n3. Check: 1 + 1 = 2."

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.EXPLICIT_NUMBERING
    assert result.ambiguous is False
    assert [step.position for step in result.steps] == [0, 1, 2]
    assert [step.text for step in result.steps] == [
        "1. Subtract 1 from both sides.",
        "2. Therefore x = 1.",
        "3. Check: 1 + 1 = 2.",
    ]
    assert_source_is_preserved(raw)


def test_numbered_step_keeps_unlabelled_continuation_lines_together() -> None:
    raw = (
        "Step 1: Let n = 2k.\n"
        "This uses the definition of even.\n"
        "Step 2: Then n^2 = 4k^2.\n"
        "So n^2 is even."
    )

    result = parse_solution_steps(raw)

    assert len(result.steps) == 2
    assert "definition of even" in result.steps[0].text
    assert "So n^2 is even" in result.steps[1].text
    assert_source_is_preserved(raw)


def test_parenthesized_and_letter_markers_are_supported_but_mixing_is_flagged() -> None:
    raw = "(1) Establish the base case.\n(b) Apply the induction hypothesis."

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.EXPLICIT_NUMBERING
    assert result.ambiguous
    assert "mixed explicit step-marker styles" in result.ambiguity_reasons
    assert len(result.steps) == 2


def test_nonconsecutive_numbering_is_preserved_and_marked_ambiguous() -> None:
    raw = "1. First claim.\n3. A later claim."

    result = parse_solution_steps(raw)

    assert len(result.steps) == 2
    assert result.ambiguous
    assert "non-consecutive numeric step markers" in result.ambiguity_reasons


def test_preamble_before_numbered_steps_is_not_discarded() -> None:
    raw = "My proof is as follows.\n1. First claim.\n2. Second claim."

    result = parse_solution_steps(raw)

    assert len(result.steps) == 3
    assert result.steps[0].text == "My proof is as follows."
    assert result.ambiguous
    assert any("before explicit" in reason for reason in result.ambiguity_reasons)
    assert_source_is_preserved(raw)


def test_prose_proof_paragraphs_are_steps_but_hard_wrapped_prose_is_not() -> None:
    paragraphs = (
        "Let n be even, so n = 2k for an integer k.\n\n"
        "Then n^2 = 4k^2 = 2(2k^2), hence n^2 is even."
    )
    paragraph_result = parse_solution_steps(paragraphs)
    assert paragraph_result.strategy is StepParseStrategy.PARAGRAPHS
    assert len(paragraph_result.steps) == 2
    assert not paragraph_result.ambiguous

    hard_wrapped = (
        "Let n be even and write n = 2k.\n"
        "Substituting this expression shows the square has a factor of two."
    )
    wrapped_result = parse_solution_steps(hard_wrapped)
    assert wrapped_result.strategy is StepParseStrategy.SINGLE_CHUNK
    assert len(wrapped_result.steps) == 1
    assert wrapped_result.ambiguous
    assert "\n" in wrapped_result.steps[0].text


def test_algebra_derivation_uses_one_source_span_per_nonblank_line() -> None:
    raw = "2x + 4 = 10\n2x = 6\nx = 3"

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.ALGEBRA_LINES
    assert [step.text for step in result.steps] == ["2x + 4 = 10", "2x = 6", "x = 3"]
    assert not result.ambiguous
    assert_source_is_preserved(raw)


def test_multiline_displayed_equation_can_start_with_expression_only_line() -> None:
    raw = "f(x)\n= (x + 1)^2\n= x^2 + 2x + 1"

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.ALGEBRA_LINES
    assert len(result.steps) == 3
    assert result.steps[0].text == "f(x)"


def test_blank_lines_and_crlf_offsets_are_preserved() -> None:
    raw = "  First paragraph.\r\n\r\n\tSecond paragraph.  \r\n"

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.PARAGRAPHS
    assert [step.text for step in result.steps] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert result.steps[0].start_offset == 2
    assert_source_is_preserved(raw)


@pytest.mark.parametrize("raw", ["", " ", "\n\t\r\n"])
def test_blank_input_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        parse_solution_steps(raw)


@pytest.mark.parametrize("raw", [None, 42, ["x = 1"]])
def test_non_string_input_is_rejected(raw: object) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        parse_solution_steps(raw)  # type: ignore[arg-type]


def test_bullets_are_not_overinterpreted_as_mathematical_steps() -> None:
    raw = "- claim one\n- claim two"

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.SINGLE_CHUNK
    assert len(result.steps) == 1
    assert result.ambiguous
    assert "bullet markers" in result.ambiguity_reasons[0]
    assert result.steps[0].text == raw


def test_mixed_prose_and_equations_stay_in_a_larger_ambiguous_chunk() -> None:
    raw = "We now simplify the expression.\nx + 1 = 4\nx = 3"

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.SINGLE_CHUNK
    assert len(result.steps) == 1
    assert result.ambiguous
    assert any("mixed prose" in reason for reason in result.ambiguity_reasons)
    assert result.steps[0].text == raw


def test_repeated_identical_steps_receive_unique_deterministic_ids() -> None:
    raw = "1. Check this.\n2. Check this."

    first = parse_solution_steps(raw)
    second = parse_solution_steps(raw)

    first_ids = [step.step_id for step in first.steps]
    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [step.step_id for step in second.steps]


def test_content_ids_do_not_depend_on_fragile_numeric_positions() -> None:
    original = parse_solution_steps("1. x = 1\n2. x + 1 = 2")
    expanded = parse_solution_steps("1. State the equation.\n2. x = 1\n3. x + 1 = 2")

    assert original.steps[0].step_id == expanded.steps[1].step_id
    assert original.steps[1].step_id == expanded.steps[2].step_id


def test_parse_student_attempt_preserves_raw_text_and_has_stable_attempt_id() -> None:
    raw = "1. x + 1 = 2\n2. x = 1"

    first = parse_student_attempt("problem.linear", raw)
    second = parse_student_attempt("problem.linear", raw)

    assert first.raw_text == raw
    assert first.attempt_id == second.attempt_id
    assert first.problem_id == "problem.linear"
    assert first.steps == second.steps
    assert first.parse_strategy is StepParseStrategy.EXPLICIT_NUMBERING


def test_custom_attempt_id_and_number_are_retained() -> None:
    attempt = parse_student_attempt(
        "problem.linear",
        "x = 1",
        attempt_id="attempt.user-supplied",
        attempt_number=3,
    )

    assert attempt.attempt_id == "attempt.user-supplied"
    assert attempt.attempt_number == 3


def test_a_single_unambiguous_line_remains_one_chunk() -> None:
    raw = "By the Pythagorean theorem, c^2 = a^2 + b^2."

    result = parse_solution_steps(raw)

    assert result.strategy is StepParseStrategy.SINGLE_CHUNK
    assert result.ambiguous is False
    assert result.steps[0].text == raw
