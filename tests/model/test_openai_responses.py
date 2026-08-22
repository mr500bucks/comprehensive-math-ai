from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from math_feedback_ai.domain.models import DiagnosisV1
from math_feedback_ai.model.client import (
    ModelOutputError,
    ModelTimeoutError,
    ModelTransportError,
    StructuredGenerationRequest,
    TextGenerationRequest,
)
from math_feedback_ai.model.openai_responses import (
    OpenAIConfigurationError,
    OpenAIResponsesClient,
    OpenAIResponsesConfig,
)


@dataclass(slots=True)
class StubTransport:
    responses: list[Mapping[str, Any] | Exception]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(text: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "model": "gpt-test-2026-01-01",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 31,
            "output_tokens": 7,
            "total_tokens": 38,
            "input_tokens_details": {"cached_tokens": 5},
        },
    }


def _config(**overrides: object) -> OpenAIResponsesConfig:
    values: dict[str, object] = {
        "model": "gpt-test-2026-01-01",
        "api_key": "test-secret-not-real",
        "max_retries": 0,
    }
    values.update(overrides)
    return OpenAIResponsesConfig(**values)  # type: ignore[arg-type]


def test_text_generation_uses_responses_api_and_reports_usage() -> None:
    transport = StubTransport([_response("Inspect the transition into your second step.")])
    client = OpenAIResponsesClient(_config(), transport=transport)

    result = client.generate_text(
        TextGenerationRequest(
            prompt="private prompt",
            system_prompt="system policy",
            timeout_seconds=4.5,
            max_output_tokens=123,
            temperature=0.0,
            metadata={"stage": "hint"},
        )
    )

    assert result.content == "Inspect the transition into your second step."
    assert result.model_name == "gpt-test-2026-01-01"
    assert result.finish_reason == "completed"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.usage.input_tokens == 31
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 38
    assert result.usage.cached_input_tokens == 5
    assert client.text_call_count == 1
    assert client.structured_call_count == 0

    sent = transport.requests[0]
    assert sent["url"] == "https://api.openai.com/v1/responses"
    assert sent["timeout_seconds"] == 4.5
    assert sent["headers"]["Authorization"] == "Bearer test-secret-not-real"
    assert sent["payload"] == {
        "model": "gpt-test-2026-01-01",
        "input": [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "private prompt"},
        ],
        "max_output_tokens": 123,
        "temperature": 0.0,
        "store": False,
        "metadata": {"stage": "hint"},
    }


def test_structured_generation_requests_json_schema_and_decodes_object() -> None:
    transport = StubTransport([_response(json.dumps({"status": "incorrect"}))])
    client = OpenAIResponsesClient(_config(strict_schema=True), transport=transport)

    result = client.generate_structured(
        StructuredGenerationRequest(
            prompt="diagnose",
            response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            schema_name="diagnosis_v1",
            max_output_tokens=512,
        )
    )

    assert result.content == {"status": "incorrect"}
    assert client.structured_call_count == 1
    response_format = transport.requests[0]["payload"]["text"]["format"]
    assert response_format == {
        "type": "json_schema",
        "name": "diagnosis_v1",
        "schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "additionalProperties": False,
            "required": ["status"],
        },
        "strict": True,
    }


def test_strict_schema_conversion_makes_pydantic_objects_provider_compatible() -> None:
    transport = StubTransport([_response("{}")])
    client = OpenAIResponsesClient(_config(strict_schema=True), transport=transport)

    client.generate_structured(
        StructuredGenerationRequest(
            prompt="diagnose",
            response_schema=DiagnosisV1.model_json_schema(),
            schema_name="diagnosis_v1",
        )
    )

    schema = transport.requests[0]["payload"]["text"]["format"]["schema"]
    stack: list[object] = [schema]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            assert "default" not in item
            assert "title" not in item
            properties = item.get("properties")
            if isinstance(properties, dict):
                assert item["additionalProperties"] is False
                assert item["required"] == list(properties)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"status": "incomplete", "output": []}, "did not complete"),
        ({"status": "completed", "output": []}, "no output text"),
        (
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}],
            },
            "refused",
        ),
    ],
)
def test_invalid_or_refused_provider_output_is_not_exposed(
    response: Mapping[str, Any],
    message: str,
) -> None:
    client = OpenAIResponsesClient(_config(), transport=StubTransport([response]))

    with pytest.raises(ModelOutputError, match=message):
        client.generate_text(TextGenerationRequest(prompt="private"))


def test_malformed_structured_text_is_reported() -> None:
    client = OpenAIResponsesClient(_config(), transport=StubTransport([_response("not-json")]))

    with pytest.raises(ModelOutputError, match="malformed structured output"):
        client.generate_structured(
            StructuredGenerationRequest(
                prompt="diagnose",
                response_schema={"type": "object"},
                schema_name="diagnosis_v1",
            )
        )


def test_transport_failure_has_one_bounded_retry() -> None:
    transport = StubTransport([ModelTransportError("temporary"), _response("ok")])
    client = OpenAIResponsesClient(_config(max_retries=1), transport=transport)

    result = client.generate_text(TextGenerationRequest(prompt="hello"))

    assert result.content == "ok"
    assert len(transport.requests) == 2
    assert client.text_call_count == 1
    assert client.request_attempt_count == 2


def test_timeout_is_preserved_after_bounded_retry() -> None:
    transport = StubTransport([ModelTimeoutError("first"), ModelTimeoutError("second")])
    client = OpenAIResponsesClient(_config(max_retries=1), transport=transport)

    with pytest.raises(ModelTimeoutError, match="second"):
        client.generate_text(TextGenerationRequest(prompt="hello"))

    assert len(transport.requests) == 2


def test_configuration_requires_explicit_model_key_and_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(OpenAIConfigurationError, match="model"):
        OpenAIResponsesConfig.from_env()
    with pytest.raises(OpenAIConfigurationError, match="OPENAI_API_KEY"):
        OpenAIResponsesConfig(model="gpt-test", api_key="")
    with pytest.raises(OpenAIConfigurationError, match="HTTPS"):
        OpenAIResponsesConfig(model="gpt-test", api_key="secret", base_url="http://example.com")


def test_configuration_repr_redacts_key() -> None:
    config = _config()

    assert "test-secret-not-real" not in repr(config)
    assert "<redacted>" in repr(config)
    serialized = json.dumps(config.public_metadata())
    assert "test-secret-not-real" not in serialized
    assert '"provider": "openai_responses"' in serialized
    assert '"model": "gpt-test-2026-01-01"' in serialized
