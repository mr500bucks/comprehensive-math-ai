from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy

import pytest

from math_feedback_ai.diagnosis.service import DiagnosisService, DiagnosisServiceConfig
from math_feedback_ai.domain.models import (
    Problem,
    ReferenceSolution,
    SolutionStep,
    StudentAttempt,
)
from math_feedback_ai.domain.taxonomy import OverallStatus
from math_feedback_ai.model.client import ModelTimeoutError, ModelTransportError
from math_feedback_ai.model.fake import FakeModelClient


def _problem() -> Problem:
    return Problem(
        problem_id="p1",
        statement="Assume x = 1. Evaluate x + (2 + 2).",
        reference_solutions=(
            ReferenceSolution(
                reference_id="ref-1",
                text="Substitute x = 1 and compute 2 + 2 = 4, obtaining 5.",
                method_label="direct substitution",
            ),
        ),
    )


def _attempt(*, problem_id: str = "p1") -> StudentAttempt:
    raw_text = "x = 1\n2 + 2 = 5"
    return StudentAttempt(
        attempt_id="a1",
        problem_id=problem_id,
        raw_text=raw_text,
        steps=(
            SolutionStep(
                step_id="step-1",
                position=0,
                text="x = 1",
                start_offset=0,
                end_offset=5,
            ),
            SolutionStep(
                step_id="step-2",
                position=1,
                text="2 + 2 = 5",
                start_offset=6,
                end_offset=len(raw_text),
            ),
        ),
    )


def _incorrect_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "attempt_id": "a1",
        "overall_status": "incorrect",
        "step_assessments": [
            {
                "step_id": "step-1",
                "status": "valid",
                "issue_codes": [],
                "explanation": "This restates the given value.",
                "evidence": "The problem assumes x = 1.",
                "confidence": 0.99,
            },
            {
                "step_id": "step-2",
                "status": "invalid",
                "issue_codes": ["computation.arithmetic"],
                "explanation": "The arithmetic equality is false.",
                "evidence": "Two plus two is four, not five.",
                "confidence": 0.99,
            },
        ],
        "first_issue": {
            "step_id": "step-2",
            "code": "computation.arithmetic",
            "explanation": "The arithmetic equality is false.",
            "evidence": "2 + 2 = 4.",
            "concept_tags": ["arithmetic"],
            "confidence": 0.99,
        },
        "completion_gap": None,
        "reusable_prefix_end_step_id": "step-1",
        "earlier_reasoning_usable": True,
        "reference_relation": "same_method",
        "confidence": 0.95,
        "confidence_reasons": ["The contradiction is directly checkable."],
    }


def test_diagnose_validates_mapping_and_records_versioned_request() -> None:
    client = FakeModelClient(structured_responses=[_incorrect_payload()])
    service = DiagnosisService(client)

    diagnosis = service.diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step-2"
    assert client.structured_call_count == 1
    request = client.structured_requests[0]
    assert request.schema_name == "diagnosis_v1"
    assert request.metadata["prompt_version"] == "diagnosis_v1"
    assert "explicitly non-exhaustive" in request.prompt
    assert "Difference from a reference solution is never" in request.prompt
    assert "ref-1" in request.prompt


def test_diagnose_accepts_json_text() -> None:
    client = FakeModelClient(structured_responses=[json.dumps(_incorrect_payload())])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INCORRECT


def test_malformed_output_gets_one_bounded_repair_attempt() -> None:
    client = FakeModelClient(structured_responses=["not json", _incorrect_payload()])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert client.structured_call_count == 2
    assert "REPAIR_ATTEMPT" not in client.structured_requests[0].prompt
    assert "REPAIR_ATTEMPT" in client.structured_requests[1].prompt
    assert client.structured_requests[1].metadata["generation_attempt"] == "2"


def test_unknown_enum_after_repair_returns_indeterminate() -> None:
    unknown_status = _incorrect_payload()
    unknown_status["overall_status"] = "mostly_fine"
    client = FakeModelClient(structured_responses=[unknown_status, deepcopy(unknown_status)])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INDETERMINATE
    assert diagnosis.confidence == 0
    assert client.structured_call_count == 2


def test_timeout_returns_indeterminate_without_unbounded_retry() -> None:
    client = FakeModelClient(structured_responses=[ModelTimeoutError("provider timed out")])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INDETERMINATE
    assert client.structured_call_count == 1


def test_transient_transport_failure_can_use_single_retry() -> None:
    client = FakeModelClient(
        structured_responses=[ModelTransportError("temporary"), _incorrect_payload()]
    )

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert client.structured_call_count == 2


def test_low_confidence_diagnosis_becomes_indeterminate_without_retry() -> None:
    low_confidence = _incorrect_payload()
    low_confidence["confidence"] = 0.2
    client = FakeModelClient(structured_responses=[low_confidence])
    service = DiagnosisService(
        client,
        DiagnosisServiceConfig(minimum_confidence=0.6),
    )

    diagnosis = service.diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INDETERMINATE
    assert client.structured_call_count == 1


def test_unknown_step_reference_is_rejected_then_repaired() -> None:
    unknown_step = _incorrect_payload()
    first_issue = unknown_step["first_issue"]
    assert isinstance(first_issue, dict)
    first_issue["step_id"] = "invented-step"
    assessments = unknown_step["step_assessments"]
    assert isinstance(assessments, list)
    assert isinstance(assessments[1], dict)
    assessments[1]["step_id"] = "invented-step"
    client = FakeModelClient(structured_responses=[unknown_step, _incorrect_payload()])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step-2"
    assert client.structured_call_count == 2


def test_missing_step_assessment_is_rejected_then_repaired() -> None:
    missing_step = _incorrect_payload()
    assessments = missing_step["step_assessments"]
    assert isinstance(assessments, list)
    missing_step["step_assessments"] = [assessments[1]]
    missing_step["reusable_prefix_end_step_id"] = None
    client = FakeModelClient(structured_responses=[missing_step, _incorrect_payload()])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert len(diagnosis.step_assessments) == 2
    assert client.structured_call_count == 2


def test_reordered_step_assessments_are_rejected_then_repaired() -> None:
    problem = _problem()
    attempt = _attempt()
    valid = _incorrect_payload()
    reordered = dict(valid)
    assessments = valid["step_assessments"]
    assert isinstance(assessments, list)
    reordered["step_assessments"] = list(reversed(assessments))
    client = FakeModelClient(structured_responses=[reordered, valid])

    diagnosis = DiagnosisService(client).diagnose(problem, attempt)

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert tuple(item.step_id for item in diagnosis.step_assessments) == tuple(
        step.step_id for step in attempt.steps
    )
    assert client.structured_call_count == 2


def test_incorrect_status_cannot_be_based_only_on_an_irrelevant_valid_step() -> None:
    inconsistent = _incorrect_payload()
    assessments = inconsistent["step_assessments"]
    assert isinstance(assessments, list)
    assessments[0] = {
        "step_id": "step-1",
        "status": "valid",
        "issue_codes": ["relevance.irrelevant"],
        "explanation": "This step is unnecessary.",
        "confidence": 0.9,
    }
    assessments[1] = {
        "step_id": "step-2",
        "status": "valid",
        "issue_codes": [],
        "explanation": "This is accepted for the structural test.",
        "confidence": 0.9,
    }
    inconsistent["first_issue"] = {
        "step_id": "step-1",
        "code": "relevance.irrelevant",
        "explanation": "This step is unnecessary.",
        "confidence": 0.9,
    }
    client = FakeModelClient(structured_responses=[inconsistent, _incorrect_payload()])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step-2"
    assert client.structured_call_count == 2


def test_irrelevant_valid_preamble_does_not_block_later_first_error() -> None:
    payload = _incorrect_payload()
    assessments = payload["step_assessments"]
    assert isinstance(assessments, list)
    first = assessments[0]
    assert isinstance(first, dict)
    first["issue_codes"] = ["relevance.irrelevant"]
    first["explanation"] = "This valid restatement is unnecessary."
    client = FakeModelClient(structured_responses=[payload])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt())

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert diagnosis.first_issue is not None
    assert diagnosis.first_issue.step_id == "step-2"
    assert client.structured_call_count == 1


def test_explicit_empty_references_override_problem_references() -> None:
    no_reference_payload = _incorrect_payload()
    no_reference_payload["reference_relation"] = "not_used"
    client = FakeModelClient(structured_responses=[no_reference_payload])

    diagnosis = DiagnosisService(client).diagnose(_problem(), _attempt(), reference_solutions=[])

    assert diagnosis.overall_status is OverallStatus.INCORRECT
    assert '"reference_solutions": []' in client.structured_requests[0].prompt
    assert "ref-1" not in client.structured_requests[0].prompt


def test_problem_attempt_mismatch_is_an_input_error() -> None:
    client = FakeModelClient(structured_responses=[_incorrect_payload()])

    with pytest.raises(ValueError, match="same problem_id"):
        DiagnosisService(client).diagnose(_problem(), _attempt(problem_id="other"))

    assert client.structured_call_count == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DiagnosisServiceConfig(timeout_seconds=0),
        lambda: DiagnosisServiceConfig(max_output_tokens=0),
        lambda: DiagnosisServiceConfig(minimum_confidence=1.1),
        lambda: DiagnosisServiceConfig(max_attempts=3),
    ],
)
def test_invalid_service_configuration_is_rejected(
    factory: Callable[[], DiagnosisServiceConfig],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_default_service_configuration_is_valid() -> None:
    config = DiagnosisServiceConfig()

    assert config.max_attempts == 2
