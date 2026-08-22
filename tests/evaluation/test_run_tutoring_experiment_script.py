from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_tutoring_experiment import main


def test_none_provider_runs_complete_development_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = f"test-tutoring-runner-{os.getpid()}"
    output_directory = Path("build")
    expected_paths = (
        output_directory / f"{stem}.summary.json",
        output_directory / f"{stem}.predictions.jsonl",
        output_directory / f"{stem}.hint-review.jsonl",
    )
    try:
        assert (
            main(
                [
                    "--provider",
                    "none",
                    "--diagnosis-attempts",
                    "1",
                    "--diagnosis-max-output-tokens",
                    "777",
                    "--output-dir",
                    str(output_directory),
                    "--artifact-stem",
                    stem,
                ]
            )
            == 0
        )
        manifest = json.loads(capsys.readouterr().out)
        summary = json.loads(expected_paths[0].read_text(encoding="utf-8"))
        predictions = expected_paths[1].read_text(encoding="utf-8").splitlines()

        assert manifest["benchmark"] == "development_v1"
        assert manifest["examples"] == 40
        assert manifest["model_calls"] == 40
        assert Path(manifest["summary"]) == expected_paths[0].resolve()
        assert Path(manifest["predictions"]) == expected_paths[1].resolve()
        assert Path(manifest["hint_review"]) == expected_paths[2].resolve()
        assert summary["examples"] == 40
        assert summary["provider_usage"]["diagnosis_calls"] == 40
        assert summary["provider_usage"]["hint_calls"] == 0
        assert summary["metrics"]["hint_generation_failure_rate"] == {
            "denominator": 0,
            "numerator": 0,
            "value": None,
        }
        assert summary["run_metadata"]["diagnosis_service"]["max_output_tokens"] == 777
        assert summary["run_metadata"]["provider"] == {
            "behavior": "safe_abstention_control",
            "provider": "none",
        }
        assert len(predictions) == 40
        assert all(
            json.loads(line)["diagnosis_attempt_outcomes"] == ["output_error"]
            for line in predictions
        )
    finally:
        for path in expected_paths:
            path.unlink(missing_ok=True)


def test_llama_provider_requires_explicit_model_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--provider", "llama-cpp"]) == 2
    assert "--model-path is required" in capsys.readouterr().err


def test_oracle_diagnosis_mode_is_explicitly_qualified(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--provider",
                "none",
                "--oracle-diagnosis",
                "--diagnosis-attempts",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(manifest["summary"]).read_text(encoding="utf-8"))

    assert summary["run_metadata"]["oracle_diagnosis"] is True
    assert summary["run_metadata"]["provider"]["diagnosis_source"] == ("development_gold_fixture")
    assert (
        "not end-to-end model quality"
        in summary["run_metadata"]["provider"]["qualification"].lower()
    )
    assert summary["provider_usage"]["hint_calls"] > 0


def test_none_provider_rejects_local_model_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--provider", "none", "--model-path", "unused.gguf"]) == 2
    assert "--model-path requires --provider llama-cpp" in capsys.readouterr().err
