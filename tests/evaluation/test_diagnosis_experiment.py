from __future__ import annotations

import json
from pathlib import Path

import pytest

from math_feedback_ai.diagnosis.service import DiagnosisService, DiagnosisServiceConfig
from math_feedback_ai.domain.taxonomy import OverallStatus
from math_feedback_ai.evaluation.benchmark import load_development_benchmark
from math_feedback_ai.evaluation.diagnosis import (
    ReferenceMode,
    development_evaluation_cases,
    diagnosis_case_set_sha256,
    reviewed_pilot_evaluation_cases,
    run_diagnosis_experiment,
    write_diagnosis_artifacts,
    write_diagnosis_report,
)
from math_feedback_ai.evaluation.reviewed_diagnosis_pilot import (
    build_reviewed_diagnosis_pilot_cases,
)
from math_feedback_ai.model.client import ModelOutputError
from math_feedback_ai.model.fake import FakeModelClient


def test_gold_replay_produces_perfect_diagnosis_report() -> None:
    examples = load_development_benchmark()
    cases = development_evaluation_cases(examples)
    client = FakeModelClient(
        structured_responses=[
            example.gold_diagnosis.model_dump(mode="json") for example in examples
        ]
    )

    report = run_diagnosis_experiment(
        benchmark_name="development_v1",
        annotation_status="not_human_validated",
        cases=cases,
        service=DiagnosisService(
            client,
            DiagnosisServiceConfig(max_attempts=1, minimum_confidence=0.0),
        ),
        reference_mode=ReferenceMode.WITH_REFERENCES,
    )

    assert report.examples == 40
    assert report.metrics["overall_status_accuracy"].value == 1.0
    assert report.metrics["first_issue_exact_accuracy"].value == 1.0
    assert report.metrics["issue_category_accuracy"].value == 1.0
    assert report.metrics["valid_alternative_false_rejection_rate"].value == 0.0
    assert report.metrics["abstention_rate"].value == pytest.approx(0.125)
    assert report.provider_usage.model_calls == 40
    assert report.provider_usage.attempt_outcomes == {"success": 40}
    assert len(report.predictions) == 40
    assert cases[6].category == "alternative_valid"
    assert cases[6].is_valid_alternative


def test_safe_abstention_report_exposes_failures_without_hiding_cases() -> None:
    examples = load_development_benchmark()
    cases = development_evaluation_cases(examples)
    client = FakeModelClient(
        structured_responses=[ModelOutputError("provider unavailable") for _ in cases]
    )

    report = run_diagnosis_experiment(
        benchmark_name="development_v1",
        annotation_status="not_human_validated",
        cases=cases,
        service=DiagnosisService(client, DiagnosisServiceConfig(max_attempts=1)),
        reference_mode=ReferenceMode.WITHOUT_REFERENCES,
    )

    assert report.metrics["overall_status_accuracy"].value == pytest.approx(0.125)
    assert report.metrics["abstention_rate"].value == 1.0
    assert report.metrics["incorrect_as_correct_rate"].value == 0.0
    assert report.metrics["valid_alternative_false_rejection_rate"].value == 0.0
    assert report.provider_usage.attempt_outcomes == {"output_error": 40}
    assert all(
        prediction.predicted.overall_status is OverallStatus.INDETERMINATE
        for prediction in report.predictions
    )


def test_reviewed_gold_replay_scores_inefficiency_and_dependency_semantics() -> None:
    reviewed = build_reviewed_diagnosis_pilot_cases()
    cases = reviewed_pilot_evaluation_cases(reviewed)
    report = run_diagnosis_experiment(
        benchmark_name="diagnosis_pilot_v1_reviewed",
        annotation_status="reviewed",
        cases=cases,
        service=DiagnosisService(
            FakeModelClient(
                structured_responses=[
                    case.reviewed_diagnosis.model_dump(mode="json") for case in reviewed
                ]
            ),
            DiagnosisServiceConfig(max_attempts=1, minimum_confidence=0.0),
        ),
        reference_mode=ReferenceMode.WITH_REFERENCES,
    )

    assert report.metrics["valid_but_inefficient_step_accuracy"].value == 1.0
    assert report.metrics["valid_inefficiency_as_error_rate"].value == 0.0
    assert report.metrics["dependent_step_accuracy"].value == 1.0
    assert report.metrics["dependent_as_independent_error_rate"].value == 0.0


def test_report_writer_preserves_every_prediction(tmp_path: Path) -> None:
    example = load_development_benchmark()[0]
    cases = development_evaluation_cases((example,))
    client = FakeModelClient(structured_responses=[example.gold_diagnosis.model_dump(mode="json")])
    report = run_diagnosis_experiment(
        benchmark_name="one-case",
        annotation_status="not_human_validated",
        cases=cases,
        service=DiagnosisService(client, DiagnosisServiceConfig(max_attempts=1)),
        reference_mode=ReferenceMode.WITH_REFERENCES,
    )
    output_path = tmp_path / "nested" / "report.json"

    write_diagnosis_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "one-case"
    assert payload["predictions"][0]["case_id"] == example.example_id
    assert payload["predictions"][0]["expected"]["overall_status"] == "fully_correct"
    assert payload["predictions"][0]["predicted"]["overall_status"] == "fully_correct"


def test_split_artifact_writer_separates_summary_and_jsonl(tmp_path: Path) -> None:
    example = load_development_benchmark()[0]
    cases = development_evaluation_cases((example,))
    report = run_diagnosis_experiment(
        benchmark_name="one case/unsafe name",
        annotation_status="pending",
        cases=cases,
        service=DiagnosisService(
            FakeModelClient(structured_responses=[example.gold_diagnosis.model_dump(mode="json")]),
            DiagnosisServiceConfig(max_attempts=1),
        ),
        reference_mode=ReferenceMode.WITH_REFERENCES,
    )

    paths = write_diagnosis_artifacts(
        report,
        tmp_path / "nested",
        run_metadata={"provider": {"name": "fake", "secret": "not-included"}},
    )

    assert paths.summary.name == "one-case-unsafe-name.with_references.summary.json"
    assert paths.predictions.name == "one-case-unsafe-name.with_references.predictions.jsonl"
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line) for line in paths.predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert "predictions" not in summary
    assert summary["predictions_file"] == paths.predictions.name
    assert "provisional Codex-generated annotations" in summary["research_qualification"]
    assert summary["run_metadata"]["provider"]["name"] == "fake"
    assert [item["case_id"] for item in predictions] == [example.example_id]
    assert predictions[0]["problem"]["problem_id"] == example.problem.problem_id
    assert predictions[0]["student_attempt"]["raw_text"] == example.student_attempt.raw_text

    with pytest.raises(FileExistsError, match="already exist"):
        write_diagnosis_artifacts(report, tmp_path / "nested")

    replaced = write_diagnosis_artifacts(report, tmp_path / "nested", overwrite=True)
    assert replaced == paths


def test_case_set_fingerprint_is_order_and_content_sensitive() -> None:
    examples = load_development_benchmark()[:2]
    cases = development_evaluation_cases(examples)

    assert diagnosis_case_set_sha256(cases) == diagnosis_case_set_sha256(cases)
    assert diagnosis_case_set_sha256(cases) != diagnosis_case_set_sha256(tuple(reversed(cases)))
    assert len(diagnosis_case_set_sha256(cases)) == 64


def test_duplicate_case_ids_are_rejected_before_model_calls() -> None:
    example = load_development_benchmark()[0]
    case = development_evaluation_cases((example,))[0]
    client = FakeModelClient()

    with pytest.raises(ValueError, match="IDs must be unique"):
        run_diagnosis_experiment(
            benchmark_name="duplicates",
            annotation_status="not_human_validated",
            cases=(case, case),
            service=DiagnosisService(client),
            reference_mode=ReferenceMode.WITH_REFERENCES,
        )

    assert client.structured_call_count == 0
