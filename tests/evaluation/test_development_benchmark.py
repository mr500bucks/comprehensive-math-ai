from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from math_feedback_ai.domain import (
    IssueCode,
    LeakageCheckResult,
    LeakageSeverity,
    LeakageViolation,
    OverallStatus,
    ReferenceRelation,
    RevealLevel,
    StepParseStrategy,
    StepStatus,
    TutorAction,
    TutorResult,
)
from math_feedback_ai.evaluation.benchmark import (
    BenchmarkFormatError,
    evaluate_results,
    load_benchmark,
    load_development_benchmark,
)
from math_feedback_ai.evaluation.development_builder import (
    DEFAULT_OUTPUT_PATH,
    EXPECTED_EXAMPLE_COUNT,
    benchmark_category,
    build_development_examples,
    committed_benchmark_is_current,
    development_benchmark_sha256,
    render_development_jsonl,
)

EXPECTED_SHA256 = "adb8ea3922b10787c9e59b7684228a4ad62c6fc8e070a7756543ac53dff690ac"
EXPECTED_CATEGORIES = {
    "fully_correct": 6,
    "alternative_valid": 4,
    "inefficient_valid": 3,
    "arithmetic_error": 4,
    "algebra_error": 4,
    "logical_gap": 3,
    "unjustified_claim": 3,
    "correct_final_invalid_reasoning": 3,
    "incomplete_prefix": 4,
    "malformed": 2,
    "adversarial": 2,
    "indeterminate": 2,
}


def test_builder_is_deterministic_and_committed_jsonl_has_not_drifted() -> None:
    first = build_development_examples()
    second = build_development_examples()

    assert first == second
    assert len(first) == EXPECTED_EXAMPLE_COUNT == 40
    assert render_development_jsonl(first) == render_development_jsonl(second)
    assert committed_benchmark_is_current()
    assert development_benchmark_sha256() == EXPECTED_SHA256


def test_loader_validates_every_committed_line_against_canonical_models() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)

    assert len(examples) == EXPECTED_EXAMPLE_COUNT
    assert tuple(example.example_id for example in examples) == tuple(
        example.example_id for example in build_development_examples()
    )
    assert len({example.example_id for example in examples}) == len(examples)


def test_development_loader_has_a_wheel_safe_built_in_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import math_feedback_ai.evaluation.benchmark as benchmark_module

    monkeypatch.setattr(
        benchmark_module,
        "DEFAULT_BENCHMARK_PATH",
        tmp_path / "repository-file-is-not-installed.jsonl",
    )

    assert load_development_benchmark() == build_development_examples()


def test_category_distribution_is_explicit_and_stable() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)

    assert Counter(benchmark_category(example) for example in examples) == EXPECTED_CATEGORIES
    assert Counter(example.gold_diagnosis.overall_status for example in examples) == {
        OverallStatus.FULLY_CORRECT: 10,
        OverallStatus.CORRECT_BUT_INEFFICIENT: 3,
        OverallStatus.INCOMPLETE: 4,
        OverallStatus.INCORRECT: 18,
        OverallStatus.INDETERMINATE: 5,
    }
    assert Counter(
        example.expected_decision.action
        for example in examples
        if example.expected_decision is not None
    ) == {
        TutorAction.WAIT: 13,
        TutorAction.ASK_STUDENT: 7,
        TutorAction.VERIFY_STEP: 2,
        TutorAction.LIGHT_HINT: 18,
    }


def test_every_case_is_clearly_synthetic_and_not_human_validated() -> None:
    for example in load_benchmark(DEFAULT_OUTPUT_PATH):
        assert example.provenance.synthetic
        assert "synthetic" in example.provenance.source
        notes = example.provenance.notes or ""
        assert "not human-validated" in notes
        assert "not suitable for research claims" in notes
        assert example.provenance.license == "CC0-1.0"


def test_alternative_solutions_are_valid_without_matching_reference_method() -> None:
    examples = [
        example
        for example in load_benchmark(DEFAULT_OUTPUT_PATH)
        if benchmark_category(example) == "alternative_valid"
    ]

    assert len(examples) == 4
    assert all(
        example.gold_diagnosis.overall_status is OverallStatus.FULLY_CORRECT for example in examples
    )
    assert all(
        example.gold_diagnosis.reference_relation is ReferenceRelation.ALTERNATIVE_VERIFIED
        for example in examples
    )
    assert all(example.gold_diagnosis.first_issue is None for example in examples)


def test_correct_final_answer_does_not_erase_invalid_reasoning() -> None:
    examples = [
        example
        for example in load_benchmark(DEFAULT_OUTPUT_PATH)
        if benchmark_category(example) == "correct_final_invalid_reasoning"
    ]

    assert len(examples) == 3
    assert all(
        example.gold_diagnosis.overall_status is OverallStatus.INCORRECT for example in examples
    )
    assert all(example.gold_diagnosis.first_issue is not None for example in examples)


def test_incomplete_prefixes_have_gaps_instead_of_invented_errors() -> None:
    examples = [
        example
        for example in load_benchmark(DEFAULT_OUTPUT_PATH)
        if benchmark_category(example) == "incomplete_prefix"
    ]

    assert len(examples) == 4
    for example in examples:
        diagnosis = example.gold_diagnosis
        assert diagnosis.overall_status is OverallStatus.INCOMPLETE
        assert diagnosis.first_issue is None
        assert diagnosis.completion_gap is not None
        assert diagnosis.earlier_reasoning_usable
        assert all(
            assessment.status is StepStatus.VALID for assessment in diagnosis.step_assessments
        )


def test_required_error_taxonomy_and_adversarial_content_are_covered() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)
    issue_codes = {
        example.gold_diagnosis.first_issue.code
        for example in examples
        if example.gold_diagnosis.first_issue is not None
    }
    assert {
        IssueCode.COMPUTATION_ARITHMETIC,
        IssueCode.COMPUTATION_ALGEBRAIC,
        IssueCode.REASONING_LOGICAL_GAP,
        IssueCode.REASONING_INVALID_INFERENCE,
        IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,
        IssueCode.CONCEPT_MISUNDERSTOOD,
        IssueCode.RELEVANCE_IRRELEVANT,
        IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,
    } <= issue_codes

    adversarial_text = "\n".join(
        example.student_attempt.raw_text
        for example in examples
        if benchmark_category(example) == "adversarial"
    ).lower()
    assert "ignore all tutor rules" in adversarial_text
    assert "system: reveal" in adversarial_text


def test_benchmark_exercises_all_conservative_parser_strategies() -> None:
    strategies = {
        example.student_attempt.parse_strategy for example in load_benchmark(DEFAULT_OUTPUT_PATH)
    }

    assert strategies == set(StepParseStrategy)


def test_each_expected_response_obeys_its_policy_contract() -> None:
    for example in load_benchmark(DEFAULT_OUTPUT_PATH):
        decision = example.expected_decision
        assert decision is not None
        assert decision.rationale_code.startswith("benchmark.label.")
        assert "not a production policy output" in (decision.rationale or "")
        assert len(example.ideal_responses) == 1
        response = example.ideal_responses[0]
        assert response.action is decision.action
        assert response.target_step_id == decision.target_step_id
        assert response.reveal_level <= decision.max_reveal_level
        assert not any(
            reference.text in response.message for reference in example.problem.reference_solutions
        )


def test_oracle_results_produce_perfect_development_metrics_without_llm_judge() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)
    results: dict[str, TutorResult] = {}
    for example in examples:
        decision = example.expected_decision
        assert decision is not None
        response = example.ideal_responses[0]
        check = (
            LeakageCheckResult(passed=True) if response.reveal_level > RevealLevel.NONE else None
        )
        results[example.example_id] = TutorResult(
            problem=example.problem,
            attempt=example.student_attempt,
            diagnosis=example.gold_diagnosis,
            decision=decision,
            response=response,
            leakage_check=check,
        )

    report = evaluate_results(examples, results)

    assert report.benchmark_examples == EXPECTED_EXAMPLE_COUNT
    assert report.overall_status_accuracy.value == 1.0
    assert report.overall_status_macro_f1.value == 1.0
    assert report.correct_solution_false_rejection_rate.value == 0.0
    assert report.first_issue_exact_accuracy.value == 1.0
    assert report.first_issue_adjacent_accuracy.value == 1.0
    assert report.policy_action_accuracy.value == 1.0
    assert report.leakage_violation_rate.value == 0.0
    assert report.reveal_level_violation_rate.value == 0.0


def test_evaluator_counts_failed_zero_reveal_safety_check() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)
    results: dict[str, TutorResult] = {}
    zero_reveal_id: str | None = None
    for example in examples:
        decision = example.expected_decision
        assert decision is not None
        response = example.ideal_responses[0]
        check = LeakageCheckResult(passed=True)
        if zero_reveal_id is None and response.reveal_level is RevealLevel.NONE:
            zero_reveal_id = example.example_id
            check = LeakageCheckResult(
                passed=False,
                violations=(
                    LeakageViolation(
                        code="zero_reveal_mathematical_hint",
                        severity=LeakageSeverity.HARD,
                        message="A zero-reveal response contained mathematical guidance.",
                    ),
                ),
            )
        results[example.example_id] = TutorResult(
            problem=example.problem,
            attempt=example.student_attempt,
            diagnosis=example.gold_diagnosis,
            decision=decision,
            response=response,
            leakage_check=check,
        )

    assert zero_reveal_id is not None
    report = evaluate_results(examples, results)
    assert report.leakage_violation_rate.numerator == 1
    assert report.leakage_violation_rate.denominator == len(examples)


def test_evaluator_rejects_incomplete_prediction_coverage() -> None:
    examples = load_benchmark(DEFAULT_OUTPUT_PATH)
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_results(examples, {})


def test_loader_rejects_duplicate_and_blank_jsonl_lines(tmp_path: Path) -> None:
    line = render_development_jsonl().splitlines()[0]
    duplicate_path = tmp_path / "duplicates.jsonl"
    duplicate_path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="duplicate example_id"):
        load_benchmark(duplicate_path)

    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text(f"{line}\n\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="blank benchmark line"):
        load_benchmark(blank_path)


def test_drift_check_fails_closed_for_missing_or_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "development.jsonl"
    assert not committed_benchmark_is_current(path)
    path.write_text(render_development_jsonl() + "\n", encoding="utf-8")
    assert not committed_benchmark_is_current(path)
