from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from math_feedback_ai.domain import OverallStatus, ReferenceRelation, TutorAction
from math_feedback_ai.evaluation.benchmark import BenchmarkFormatError
from math_feedback_ai.evaluation.development_builder import build_development_examples
from math_feedback_ai.evaluation.diagnosis_pilot_builder import (
    DEFAULT_PILOT_PATH,
    DEFAULT_REVIEW_PATH,
    EXPECTED_PILOT_CASE_COUNT,
    build_diagnosis_pilot_cases,
    committed_diagnosis_pilot_is_current,
    diagnosis_pilot_review_sha256,
    diagnosis_pilot_sha256,
    load_diagnosis_pilot,
    render_diagnosis_pilot_jsonl,
    render_diagnosis_pilot_review,
    validate_built_diagnosis_pilot,
    write_diagnosis_pilot,
)
from math_feedback_ai.evaluation.pilot import (
    DiagnosisPilotCaseV1,
    HumanReviewStatus,
    validate_pilot_cases,
)

EXPECTED_JSONL_SHA256 = "42ff58f9f5694c36385aa395ed44fd2ee711c421c83cff73c8327c6b3d6c77d5"
EXPECTED_REVIEW_SHA256 = "2be6e2f4d421551e53e984e06a5527bcefedea75b8432f4106116b3e3c0f11b7"
EXPECTED_CATEGORIES = {
    "algebraic_error": 3,
    "ambiguous_difficult": 2,
    "arithmetic_slip": 3,
    "concept_misunderstood": 3,
    "correct_but_inefficient": 4,
    "correct_final_invalid_reasoning": 3,
    "fully_correct_conventional": 7,
    "fully_correct_unconventional": 6,
    "incomplete_valid": 4,
    "invalid_cancellation_division": 3,
    "logical_gap": 3,
    "missing_condition": 3,
    "theorem_misuse": 3,
    "unsupported_claim": 3,
}
EXPECTED_DOMAINS = {
    "algebra": 14,
    "calculus": 3,
    "combinatorics": 6,
    "functions": 8,
    "geometry": 8,
    "inequalities": 3,
    "number_theory": 9,
    "probability": 3,
}


def test_pilot_builder_is_deterministic_and_committed_artifacts_are_current() -> None:
    first = build_diagnosis_pilot_cases()
    second = build_diagnosis_pilot_cases()

    assert first == second
    assert len(first) == EXPECTED_PILOT_CASE_COUNT == 50
    assert render_diagnosis_pilot_jsonl(first) == render_diagnosis_pilot_jsonl(second)
    assert render_diagnosis_pilot_review(first) == render_diagnosis_pilot_review(second)
    assert committed_diagnosis_pilot_is_current()
    assert diagnosis_pilot_sha256() == EXPECTED_JSONL_SHA256
    assert diagnosis_pilot_review_sha256() == EXPECTED_REVIEW_SHA256


def test_loader_returns_typed_pending_cases() -> None:
    loaded = load_diagnosis_pilot()

    assert loaded == build_diagnosis_pilot_cases()
    assert all(isinstance(case, DiagnosisPilotCaseV1) for case in loaded)
    assert all(case.human_review_status is HumanReviewStatus.PENDING for case in loaded)
    assert all(case.provenance.authoring_method == "codex_proposed" for case in loaded)
    assert all(case.provenance.research_use == "provisional_only" for case in loaded)
    assert all("not mathematical ground truth" in case.provenance.notes for case in loaded)


def test_loader_has_a_wheel_safe_built_in_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import math_feedback_ai.evaluation.diagnosis_pilot_builder as pilot_builder

    monkeypatch.setattr(pilot_builder, "DEFAULT_PILOT_PATH", tmp_path / "not-installed.jsonl")

    assert load_diagnosis_pilot() == build_diagnosis_pilot_cases()


def test_distribution_is_explicit_stable_and_mathematically_diverse() -> None:
    report = validate_built_diagnosis_pilot()

    assert report.passed
    assert report.category_counts == EXPECTED_CATEGORIES
    assert report.domain_counts == EXPECTED_DOMAINS
    assert Counter(
        case.proposed_diagnosis.overall_status for case in build_diagnosis_pilot_cases()
    ) == {
        OverallStatus.FULLY_CORRECT: 13,
        OverallStatus.CORRECT_BUT_INEFFICIENT: 4,
        OverallStatus.INCOMPLETE: 4,
        OverallStatus.INCORRECT: 27,
        OverallStatus.INDETERMINATE: 2,
    }


def test_proposed_actions_and_reveal_ceilings_are_conservative() -> None:
    cases = build_diagnosis_pilot_cases()

    assert Counter(case.proposed_decision.action for case in cases) == {
        TutorAction.WAIT: 17,
        TutorAction.ASK_STUDENT: 4,
        TutorAction.VERIFY_STEP: 2,
        TutorAction.LIGHT_HINT: 27,
    }
    assert all(int(case.proposed_decision.max_reveal_level) <= 1 for case in cases)


def test_alternative_cases_are_labelled_independently_of_reference_method() -> None:
    alternatives = [
        case
        for case in build_diagnosis_pilot_cases()
        if case.category == "fully_correct_unconventional"
    ]

    assert len(alternatives) == 6
    assert all("alternative_valid" in case.characteristics for case in alternatives)
    assert all(
        case.proposed_diagnosis.reference_relation is ReferenceRelation.ALTERNATIVE_VERIFIED
        for case in alternatives
    )


def test_ambiguous_cases_are_explicit_and_require_human_adjudication() -> None:
    ambiguous = [
        case for case in build_diagnosis_pilot_cases() if "ambiguous" in case.characteristics
    ]

    assert [case.case_id for case in ambiguous] == [
        "pilot.49.diagram-dependent-angle",
        "pilot.50.custom-operation-notation",
    ]
    assert all(case.ambiguity_notes for case in ambiguous)
    assert all(not case.problem.reference_solutions for case in ambiguous)
    assert all(
        case.proposed_diagnosis.overall_status is OverallStatus.INDETERMINATE for case in ambiguous
    )


def test_review_artifact_is_complete_and_review_friendly() -> None:
    review = DEFAULT_REVIEW_PATH.read_text(encoding="utf-8")

    assert review.count("\n## pilot.") == EXPECTED_PILOT_CASE_COUNT
    assert review.count("**Disposition:** `PENDING`") == EXPECTED_PILOT_CASE_COUNT
    assert review.count("**Reviewer notes:**") == EXPECTED_PILOT_CASE_COUNT
    for label in (
        "**Problem:**",
        "**Student solution:**",
        "**Proposed status:**",
        "**Proposed first issue:**",
        "**Proposed issue category:**",
        "**Proposed reusable prefix end:**",
        "**Proposed tutor action / maximum reveal:**",
        "**Short rationale:**",
    ):
        assert review.count(label) == EXPECTED_PILOT_CASE_COUNT
    assert "not mathematical ground truth" in review


def test_validator_checks_pilot_and_development_split_for_duplicates() -> None:
    report = validate_pilot_cases(
        build_diagnosis_pilot_cases(),
        comparison_examples=build_development_examples(),
    )

    assert report.passed


def test_validator_detects_duplicate_ids_problems_and_solutions() -> None:
    first = build_diagnosis_pilot_cases()[0]
    duplicate = DiagnosisPilotCaseV1.model_validate(
        {**first.model_dump(mode="python"), "case_id": "pilot.duplicate"}
    )

    report = validate_pilot_cases((first, first, duplicate))
    codes = {issue.code for issue in report.issues}

    assert "duplicate.case_id" in codes
    assert "duplicate.problem" in codes
    assert "duplicate.student_solution" in codes


def test_validator_detects_reference_text_copied_into_student_solution() -> None:
    case = build_diagnosis_pilot_cases()[0]
    reference = case.problem.reference_solutions[0]
    copied_reference = reference.model_copy(update={"text": case.student_attempt.raw_text})
    leaking_problem = case.problem.model_copy(update={"reference_solutions": (copied_reference,)})
    leaking_case = DiagnosisPilotCaseV1.model_validate(
        {**case.model_dump(mode="python"), "problem": leaking_problem}
    )

    report = validate_pilot_cases((leaking_case,))

    assert [issue.code for issue in report.issues] == ["leakage.reference_into_student"]


def test_model_rejects_non_pending_proposals_and_dangling_targets() -> None:
    case = build_diagnosis_pilot_cases()[0]
    payload = case.model_dump(mode="python")
    payload["human_review_status"] = "approved"
    with pytest.raises(ValidationError, match="must remain pending"):
        DiagnosisPilotCaseV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["proposed_decision"] = {
        **case.proposed_decision.model_dump(mode="python"),
        "action": TutorAction.LIGHT_HINT,
        "max_reveal_level": 1,
        "target_step_id": "step_does_not_exist",
    }
    with pytest.raises(ValidationError, match="unknown step"):
        DiagnosisPilotCaseV1.model_validate(payload)


def test_loader_rejects_blank_duplicate_and_malformed_lines(tmp_path: Path) -> None:
    line = render_diagnosis_pilot_jsonl().splitlines()[0]
    blank = tmp_path / "blank.jsonl"
    blank.write_text(f"{line}\n\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="blank pilot line"):
        load_diagnosis_pilot(blank)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="duplicate.case_id"):
        load_diagnosis_pilot(duplicate)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"case_id": "missing-fields"}\n', encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="invalid pilot case"):
        load_diagnosis_pilot(malformed)


def test_writer_and_drift_check_fail_closed(tmp_path: Path) -> None:
    jsonl = tmp_path / "pilot.jsonl"
    review = tmp_path / "review.md"

    assert not committed_diagnosis_pilot_is_current(jsonl, review)
    assert write_diagnosis_pilot(jsonl, review) == (jsonl, review)
    assert committed_diagnosis_pilot_is_current(jsonl, review)

    review.write_text(review.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
    assert not committed_diagnosis_pilot_is_current(jsonl, review)


def test_committed_jsonl_and_review_paths_exist() -> None:
    assert DEFAULT_PILOT_PATH.is_file()
    assert DEFAULT_REVIEW_PATH.is_file()
