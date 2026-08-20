"""Conservative deterministic parsing of raw student solutions into steps.

The parser deliberately prefers a larger source-preserving chunk over invented
mathematical structure.  It only splits on strong surface signals: repeated
explicit numbering, blank-line paragraph boundaries, or consistently
equation-like lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from math_feedback_ai.domain.models import (
    SolutionStep,
    StepParseResult,
    StudentAttempt,
)
from math_feedback_ai.domain.taxonomy import StepParseStrategy

_NUMBERED_LINE = re.compile(
    r"^[ \t]*(?:"
    r"(?:[Ss]tep[ \t]+)?(?P<number>\d+)[.):]"
    r"|\((?P<parenthesized>\d+)\)"
    r"|\((?P<parenthesized_letter>[A-Za-z])\)"
    r"|(?P<letter>[A-Za-z])[.)]"
    r")[ \t]+"
)
_BULLET_LINE = re.compile(r"^[ \t]*[-*\u2022][ \t]+")
_PARAGRAPH_BREAK = re.compile(
    r"(?:\r\n|\r|\n)[ \t]*(?:\r\n|\r|\n)"
    r"(?:[ \t]*(?:\r\n|\r|\n))*"
)
_MATH_RELATION = re.compile(
    r"(?:=|<=|>=|!=|<|>|\u2264|\u2265|\u2260|\u2248|\u2261|"
    r"\\(?:leq|geq|neq|approx|equiv)\b|(?:=>|->|\u21d2|\u2192))"
)


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    text: str


def _lines_with_offsets(raw_text: str) -> tuple[_Line, ...]:
    lines: list[_Line] = []
    cursor = 0
    for text in raw_text.splitlines(keepends=True):
        end = cursor + len(text)
        lines.append(_Line(cursor, end, text.rstrip("\r\n")))
        cursor = end
    if cursor < len(raw_text):
        lines.append(_Line(cursor, len(raw_text), raw_text[cursor:]))
    elif not lines:
        lines.append(_Line(0, len(raw_text), raw_text))
    return tuple(lines)


def _trim_span(raw_text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _stable_step_ids(texts: tuple[str, ...]) -> tuple[str, ...]:
    """Return content-derived IDs, disambiguating exact duplicate steps."""

    seen: dict[str, int] = {}
    result: list[str] = []
    for text in texts:
        normalized = " ".join(text.split())
        normalized = _NUMBERED_LINE.sub("", normalized, count=1)
        digest = sha256(normalized.encode("utf-8")).hexdigest()[:16]
        base = f"step_{digest}"
        occurrence = seen.get(base, 0) + 1
        seen[base] = occurrence
        result.append(base if occurrence == 1 else f"{base}_{occurrence}")
    return tuple(result)


def _numbering_ambiguities(
    matches: tuple[re.Match[str], ...],
) -> tuple[str, ...]:
    kinds: list[str] = []
    numeric_values: list[int] = []
    for match in matches:
        if match.group("number") is not None:
            kinds.append("number")
            numeric_values.append(int(match.group("number")))
        elif match.group("parenthesized") is not None:
            kinds.append("parenthesized")
            numeric_values.append(int(match.group("parenthesized")))
        else:
            kinds.append("letter")

    reasons: list[str] = []
    if len(set(kinds)) > 1:
        reasons.append("mixed explicit step-marker styles")
    if numeric_values and len(numeric_values) == len(matches):
        expected = list(range(numeric_values[0], numeric_values[0] + len(matches)))
        if numeric_values != expected:
            reasons.append("non-consecutive numeric step markers")
    return tuple(reasons)


def _looks_algebraic(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _BULLET_LINE.match(stripped):
        return False
    if _MATH_RELATION.search(stripped):
        return True
    if re.fullmatch(r"[A-Za-z0-9_{}()[\].,^+\-*/\\ ]+", stripped):
        # Function/expression-only lines are common at the beginning of a
        # displayed derivation. Plain multi-word prose remains a larger chunk.
        has_math_punctuation = bool(re.search(r"[0-9_{}()[\].,^+\-*/\\]", stripped))
        is_plain_words = bool(re.fullmatch(r"[A-Za-z]+(?:[ \t]+[A-Za-z]+)+[.!?]?", stripped))
        if has_math_punctuation and not is_plain_words:
            return True
    # A continuation line in a displayed derivation may begin with an operator.
    return bool(re.match(r"^(?:[+\-*/]|\\(?:cdot|times)\b)\s*\S", stripped))


def _make_result(
    raw_text: str,
    spans: tuple[tuple[int, int], ...],
    strategy: StepParseStrategy,
    reasons: tuple[str, ...] = (),
) -> StepParseResult:
    texts = tuple(raw_text[start:end] for start, end in spans)
    step_ids = _stable_step_ids(texts)
    ambiguous = bool(reasons)
    steps = tuple(
        SolutionStep(
            step_id=step_id,
            position=position,
            text=text,
            start_offset=start,
            end_offset=end,
            segmentation_ambiguous=ambiguous,
        )
        for position, (step_id, text, (start, end)) in enumerate(
            zip(step_ids, texts, spans, strict=True)
        )
    )
    return StepParseResult(
        raw_text=raw_text,
        steps=steps,
        strategy=strategy,
        ambiguous=ambiguous,
        ambiguity_reasons=reasons,
    )


def _numbered_spans(
    raw_text: str, lines: tuple[_Line, ...]
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]] | None:
    marked: list[tuple[_Line, re.Match[str]]] = []
    for line in lines:
        match = _NUMBERED_LINE.match(line.text)
        if match is not None:
            marked.append((line, match))
    if len(marked) < 2:
        return None

    reasons = list(_numbering_ambiguities(tuple(match for _, match in marked)))
    spans: list[tuple[int, int]] = []
    first_start = marked[0][0].start
    preamble = _trim_span(raw_text, 0, first_start)
    if preamble is not None:
        spans.append(preamble)
        reasons.append("unlabelled text appears before explicit step markers")

    for index, (line, _) in enumerate(marked):
        end = marked[index + 1][0].start if index + 1 < len(marked) else len(raw_text)
        span = _trim_span(raw_text, line.start, end)
        if span is not None:
            spans.append(span)
    return tuple(spans), tuple(dict.fromkeys(reasons))


def _paragraph_spans(raw_text: str) -> tuple[tuple[int, int], ...] | None:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in _PARAGRAPH_BREAK.finditer(raw_text):
        span = _trim_span(raw_text, start, separator.start())
        if span is not None:
            spans.append(span)
        start = separator.end()
    final = _trim_span(raw_text, start, len(raw_text))
    if final is not None:
        spans.append(final)
    return tuple(spans) if len(spans) >= 2 else None


def _nonblank_line_spans(raw_text: str, lines: tuple[_Line, ...]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for line in lines:
        span = _trim_span(raw_text, line.start, line.end)
        if span is not None:
            result.append(span)
    return tuple(result)


def parse_solution_steps(raw_text: str) -> StepParseResult:
    """Parse raw student text without losing or fabricating source structure.

    ``TypeError`` is raised for non-string input and ``ValueError`` for empty or
    whitespace-only input.  Ambiguous surface structure is retained as a larger
    chunk and explicitly described in the result.
    """

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    if not raw_text.strip():
        raise ValueError("raw_text must contain non-whitespace text")

    lines = _lines_with_offsets(raw_text)

    numbered = _numbered_spans(raw_text, lines)
    if numbered is not None:
        spans, reasons = numbered
        return _make_result(
            raw_text,
            spans,
            StepParseStrategy.EXPLICIT_NUMBERING,
            reasons,
        )

    paragraph_spans = _paragraph_spans(raw_text)
    if paragraph_spans is not None:
        bullet_count = sum(bool(_BULLET_LINE.match(line.text)) for line in lines)
        reasons = (
            ("bullet-like lines inside paragraphs may represent finer steps",)
            if bullet_count >= 2
            else ()
        )
        return _make_result(
            raw_text,
            paragraph_spans,
            StepParseStrategy.PARAGRAPHS,
            reasons,
        )

    line_spans = _nonblank_line_spans(raw_text, lines)
    line_texts = tuple(raw_text[start:end] for start, end in line_spans)
    if len(line_spans) >= 2 and all(_looks_algebraic(text) for text in line_texts):
        return _make_result(
            raw_text,
            line_spans,
            StepParseStrategy.ALGEBRA_LINES,
        )

    full_span = _trim_span(raw_text, 0, len(raw_text))
    assert full_span is not None  # guarded by the non-blank check above
    single_chunk_reasons: list[str] = []
    bullet_count = sum(bool(_BULLET_LINE.match(line.text)) for line in lines)
    algebraic_count = sum(_looks_algebraic(text) for text in line_texts)
    if bullet_count >= 2:
        single_chunk_reasons.append(
            "bullet markers are not treated as reliable mathematical step boundaries"
        )
    if len(line_spans) >= 2 and 0 < algebraic_count < len(line_spans):
        single_chunk_reasons.append(
            "mixed prose and equation-like lines could be hard wrapping or separate steps"
        )
    if len(line_spans) >= 2 and not single_chunk_reasons:
        single_chunk_reasons.append(
            "line breaks alone are not reliable boundaries for prose reasoning"
        )
    return _make_result(
        raw_text,
        (full_span,),
        StepParseStrategy.SINGLE_CHUNK,
        tuple(single_chunk_reasons),
    )


def parse_student_attempt(
    problem_id: str,
    raw_text: str,
    *,
    attempt_id: str | None = None,
    attempt_number: int = 1,
) -> StudentAttempt:
    """Parse text and package it as a stable, validated ``StudentAttempt``."""

    result = parse_solution_steps(raw_text)
    if attempt_id is None:
        digest = sha256(f"{problem_id}\0{raw_text}".encode()).hexdigest()[:16]
        attempt_id = f"attempt_{digest}"
    return StudentAttempt(
        attempt_id=attempt_id,
        problem_id=problem_id,
        raw_text=result.raw_text,
        steps=result.steps,
        attempt_number=attempt_number,
        segmentation_ambiguous=result.ambiguous,
        parse_strategy=result.strategy,
        parse_notes=result.ambiguity_reasons,
    )


__all__ = ["parse_solution_steps", "parse_student_attempt"]
