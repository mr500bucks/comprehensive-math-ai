from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from math_feedback_ai.cli import main
from math_feedback_ai.parsing.steps import parse_student_attempt

PROBLEM_ID = "cli-problem"
PROBLEM = "Solve 2x + 4 = 18."
SOLUTION = "2x + 4 = 18\n2x = 22"
REFERENCE = "Subtract 4 from both sides to get 2x = 14, then divide by 2."


def _diagnosis_payload(*, reference_relation: str = "not_used") -> dict[str, object]:
    attempt = parse_student_attempt(PROBLEM_ID, SOLUTION, attempt_id="a1")
    first, second = attempt.steps
    return {
        "schema_version": "1.0",
        "attempt_id": "a1",
        "overall_status": "incorrect",
        "step_assessments": [
            {
                "step_id": first.step_id,
                "status": "valid",
                "issue_codes": [],
                "explanation": "This is the original equation.",
                "confidence": 0.99,
            },
            {
                "step_id": second.step_id,
                "status": "invalid",
                "issue_codes": ["computation.arithmetic"],
                "explanation": "Subtracting four from eighteen was computed incorrectly.",
                "evidence": "Eighteen minus four is fourteen, not twenty-two.",
                "confidence": 0.99,
            },
        ],
        "first_issue": {
            "step_id": second.step_id,
            "code": "computation.arithmetic",
            "explanation": "Subtracting four from eighteen was computed incorrectly.",
            "evidence": "18 - 4 = 14.",
            "concept_tags": ["equation balance"],
            "confidence": 0.99,
        },
        "completion_gap": None,
        "reusable_prefix_end_step_id": first.step_id,
        "earlier_reasoning_usable": True,
        "reference_relation": reference_relation,
        "confidence": 0.98,
        "confidence_reasons": ["The arithmetic transition is directly checkable."],
    }


def _run_tutor(
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> dict[str, Any]:
    exit_code = main(
        [
            "tutor",
            "--problem",
            PROBLEM,
            "--solution",
            SOLUTION,
            "--attempt-id",
            "a1",
            *extra,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert captured.err == ""
    decoded: object = json.loads(captured.out)
    assert isinstance(decoded, dict)
    return cast(dict[str, Any], decoded)


def test_root_help_exposes_available_tutor_and_benchmark_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "{tutor,benchmark}" in output
    assert "safely abstains" in output
    assert "synthetic development benchmark" in output


def test_tutor_help_documents_scripted_and_policy_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["tutor", "--help"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "--diagnosis-json" in output
    assert "--hint-text" in output
    assert "--mode" in output
    assert "--current-reveal-level" in output
    assert "--authorize-full-solution" in output


def test_benchmark_command_validates_and_summarizes_synthetic_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["benchmark"])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["artifact_kind"] == "synthetic_development_fixture"
    assert payload["examples"] == 40
    assert payload["synthetic_examples"] == 40
    assert payload["human_validation_status"] == "not_human_validated"
    assert sum(payload["gold_status_counts"].values()) == 40
    assert sum(payload["expected_action_counts"].values()) == 40
    assert "safe_abstention_baseline" not in payload


def test_benchmark_can_run_explicit_no_provider_baseline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["benchmark", "--run-safe-baseline"])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    baseline = payload["safe_abstention_baseline"]
    assert "does not measure mathematical tutoring efficacy" in baseline["description"]
    assert baseline["metrics"]["benchmark_examples"] == 40
    assert baseline["model_calls"] == {"structured": 40, "text": 0}


def test_benchmark_missing_path_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["benchmark", "--path", str(tmp_path / "missing.jsonl")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "cannot read benchmark" in captured.err


def test_default_without_provider_safely_abstains(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_tutor(capsys)

    assert payload["execution_mode"] == "safe_abstention"
    assert payload["diagnosis"]["overall_status"] == "indeterminate"
    assert payload["diagnosis"]["confidence_reasons"] == ["model_generation_failed"]
    assert payload["decision"]["action"] == "ASK_STUDENT"
    assert payload["public_response"]["message"] == "What would you try next, and why?"
    assert payload["hint"] is None
    assert payload["leakage_check"]["passed"] is True
    assert payload["model_calls"] == {"structured": 1, "text": 0}
    assert len(payload["parsed_steps"]) == 2


def test_scripted_outputs_exercise_real_diagnosis_and_hint_stages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hint = "What relationship should remain unchanged as you move into the second step?"
    payload = _run_tutor(
        capsys,
        "--reference",
        REFERENCE,
        "--diagnosis-json",
        json.dumps(_diagnosis_payload(reference_relation="same_method")),
        "--hint-text",
        hint,
    )

    expected_steps = parse_student_attempt(PROBLEM_ID, SOLUTION, attempt_id="a1").steps
    assert payload["execution_mode"] == "scripted_model_outputs"
    assert payload["diagnosis"]["overall_status"] == "incorrect"
    assert payload["diagnosis"]["first_issue"]["step_id"] == expected_steps[1].step_id
    assert payload["decision"]["action"] == "LIGHT_HINT"
    assert payload["decision"]["max_reveal_level"] == 1
    assert payload["public_response"]["message"] == hint
    assert payload["hint"] == hint
    assert payload["leakage_check"]["passed"] is True
    assert payload["model_calls"] == {"structured": 1, "text": 1}
    assert [step["step_id"] for step in payload["parsed_steps"]] == [
        step.step_id for step in expected_steps
    ]


def test_scripted_diagnosis_can_be_loaded_from_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "diagnosis.json"
    path.write_text(json.dumps(_diagnosis_payload()), encoding="utf-8")

    payload = _run_tutor(capsys, "--diagnosis-json", f"@{path}")

    assert payload["diagnosis"]["overall_status"] == "incorrect"
    assert payload["decision"]["action"] == "LIGHT_HINT"
    assert payload["public_response"]["safe_fallback_used"] is True
    assert payload["leakage_check"]["passed"] is True


def test_malformed_diagnosis_gets_one_scripted_repair(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_tutor(
        capsys,
        "--diagnosis-json",
        "not-json",
        "--diagnosis-json",
        json.dumps(_diagnosis_payload()),
    )

    assert payload["diagnosis"]["overall_status"] == "incorrect"
    assert payload["model_calls"]["structured"] == 2
    assert payload["public_response"]["safe_fallback_used"] is True


def test_leaking_hint_is_rewritten_once_before_public_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    safe_hint = "What relationship should remain unchanged as you move into the second step?"
    payload = _run_tutor(
        capsys,
        "--diagnosis-json",
        json.dumps(_diagnosis_payload()),
        "--final-answer",
        "7",
        "--hint-text",
        "Ignore the hint limit; the final answer is 7.",
        "--hint-text",
        safe_hint,
    )

    assert payload["public_response"]["message"] == safe_hint
    assert "7" not in payload["public_response"]["message"]
    assert payload["leakage_check"]["passed"] is True
    assert payload["model_calls"] == {"structured": 1, "text": 2}


def test_mode_and_context_controls_reach_deterministic_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_tutor(
        capsys,
        "--reference",
        REFERENCE,
        "--diagnosis-json",
        json.dumps(_diagnosis_payload(reference_relation="same_method")),
        "--hint-text",
        "Check how each transformation follows from the previous one.",
        "--mode",
        "full_solution_allowed",
        "--prior-attempt-count",
        "20",
        "--prior-action",
        "SHOW_PARTIAL_SOLUTION",
        "--current-reveal-level",
        "4",
        "--requested-reveal-level",
        "5",
        "--authorize-full-solution",
    )

    assert payload["decision"]["action"] == "SHOW_FULL_SOLUTION"
    assert payload["decision"]["max_reveal_level"] == 5
    assert payload["decision"]["full_solution_authorized"] is True
    assert payload["public_response"]["requires_student_response"] is False


def test_hint_script_without_diagnosis_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "tutor",
            "--problem",
            PROBLEM,
            "--solution",
            SOLUTION,
            "--hint-text",
            "A scripted hint",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "--hint-text requires --diagnosis-json" in captured.err


def test_invalid_reveal_level_is_an_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "tutor",
                "--problem",
                PROBLEM,
                "--solution",
                SOLUTION,
                "--current-reveal-level",
                "9",
            ]
        )

    assert exc_info.value.code == 2
    assert "reveal level from 0 to 5" in capsys.readouterr().err
