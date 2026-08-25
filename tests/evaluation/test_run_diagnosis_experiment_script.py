from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_diagnosis_experiment import main


def test_safe_development_ablation_writes_two_complete_conditions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--benchmark",
                "development",
                "--provider",
                "none",
                "--reference-mode",
                "both",
                "--diagnosis-attempts",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    manifest = json.loads(captured.out)

    assert manifest["benchmark"] == "development_v1"
    assert len(manifest["outputs"]) == 2
    for output in manifest["outputs"]:
        summary_path = Path(output["summary"])
        predictions_path = Path(output["predictions"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = predictions_path.read_text(encoding="utf-8").splitlines()
        assert summary["examples"] == 40
        assert summary["provider_usage"]["model_calls"] == 40
        assert summary["run_metadata"]["provider"]["provider"] == "none"
        assert len(summary["run_metadata"]["case_set_sha256"]) == 64
        assert len(predictions) == 40
        failure_analysis = json.loads(Path(output["failure_analysis"]).read_text(encoding="utf-8"))
        assert failure_analysis["evaluated_cases"] == 40
    ablation = json.loads(Path(manifest["reference_ablation"]).read_text(encoding="utf-8"))
    assert ablation["examples"] == 40


def test_safe_pilot_run_preserves_pending_research_qualification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--benchmark",
                "pilot",
                "--provider",
                "none",
                "--reference-mode",
                "without",
                "--diagnosis-attempts",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    manifest = json.loads(captured.out)
    output = manifest["outputs"][0]
    summary = json.loads(Path(output["summary"]).read_text(encoding="utf-8"))
    predictions = Path(output["predictions"]).read_text(encoding="utf-8").splitlines()

    assert manifest["annotation_status"] == "pending"
    assert "not validated research results" in manifest["research_qualification"]
    assert "not validated research results" in summary["research_qualification"]
    assert summary["examples"] == 50
    assert len(predictions) == 50


def test_safe_reviewed_pilot_run_records_comparison_dimensions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--benchmark",
                "pilot-reviewed",
                "--provider",
                "none",
                "--reference-mode",
                "without",
                "--diagnosis-attempts",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(manifest["outputs"][0]["summary"]).read_text(encoding="utf-8"))
    metadata = summary["run_metadata"]

    assert manifest["benchmark"] == "diagnosis_pilot_v1_reviewed"
    assert manifest["annotation_status"] == "reviewed"
    assert "research_qualification" not in manifest
    assert metadata["benchmark_version"] == "diagnosis_pilot_v1_reviewed"
    assert metadata["diagnosis_schema_version"] == "1.1"
    assert metadata["prompt_version"] == "diagnosis_v2"
    assert metadata["reference_mode"] == "without_references"


def test_provider_specific_options_fail_before_any_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--benchmark",
                "development",
                "--provider",
                "none",
                "--model",
                "unexpected-model",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "provider-specific model options" in captured.err
    assert not tuple(tmp_path.iterdir())
