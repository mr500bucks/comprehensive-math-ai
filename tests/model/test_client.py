from __future__ import annotations

from collections.abc import Callable

import pytest

from math_feedback_ai.model.client import (
    GenerationResult,
    ModelTimeoutError,
    StructuredGenerationRequest,
    TextGenerationRequest,
    TokenUsage,
)
from math_feedback_ai.model.fake import FakeModelClient, FakeModelExhaustedError


def test_fake_model_returns_scripted_text_and_records_usage() -> None:
    expected = GenerationResult(
        content="inspect the transition",
        usage=TokenUsage(input_tokens=12, output_tokens=4, total_tokens=16),
        model_name="fake-v1",
        latency_ms=2.5,
    )
    client = FakeModelClient(text_responses=[expected])
    request = TextGenerationRequest(prompt="Give a hint", metadata={"stage": "hint"})

    actual = client.generate_text(request)

    assert actual == expected
    assert client.text_requests == [request]
    assert client.text_call_count == 1


def test_fake_model_returns_mapping_for_structured_generation() -> None:
    client = FakeModelClient(structured_responses=[{"status": "fully_correct"}])
    request = StructuredGenerationRequest(
        prompt="Diagnose",
        response_schema={"type": "object"},
        schema_name="diagnosis_v1",
    )

    result = client.generate_structured(request)

    assert result.content == {"status": "fully_correct"}
    assert result.model_name == "deterministic-fake"
    assert client.structured_requests == [request]


def test_fake_model_raises_scripted_timeout() -> None:
    timeout = ModelTimeoutError("timed out")
    client = FakeModelClient(structured_responses=[timeout])
    request = StructuredGenerationRequest(
        prompt="Diagnose",
        response_schema={"type": "object"},
        schema_name="diagnosis_v1",
    )

    with pytest.raises(ModelTimeoutError, match="timed out"):
        client.generate_structured(request)


def test_fake_model_fails_clearly_when_script_is_exhausted() -> None:
    client = FakeModelClient()

    with pytest.raises(FakeModelExhaustedError, match="no scripted text response"):
        client.generate_text(TextGenerationRequest(prompt="hello"))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TextGenerationRequest(prompt="  "), "prompt must not be blank"),
        (
            lambda: TextGenerationRequest(prompt="ok", timeout_seconds=0),
            "timeout_seconds must be positive",
        ),
        (
            lambda: TextGenerationRequest(prompt="ok", max_output_tokens=0),
            "max_output_tokens must be positive",
        ),
        (
            lambda: StructuredGenerationRequest(
                prompt="ok", response_schema={}, schema_name="diagnosis"
            ),
            "response_schema must not be empty",
        ),
    ],
)
def test_generation_requests_reject_invalid_values(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        TokenUsage(input_tokens=-1)
