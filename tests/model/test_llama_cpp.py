from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from math_feedback_ai.model.client import (
    ModelOutputError,
    ModelTimeoutError,
    StructuredGenerationRequest,
    TextGenerationRequest,
)
from math_feedback_ai.model.llama_cpp import (
    DEFAULT_QWEN_FILENAME,
    DEFAULT_QWEN_REPOSITORY,
    DEFAULT_QWEN_REVISION,
    DEFAULT_QWEN_SHA256,
    DEFAULT_QWEN_SIZE_BYTES,
    LlamaCppConfig,
    LlamaCppConfigurationError,
    LlamaCppModelClient,
)


@dataclass(slots=True)
class StubBackend:
    responses: list[Mapping[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create_chat_completion(self, **kwargs: Any) -> Mapping[str, Any]:
        self.requests.append(kwargs)
        return self.responses.pop(0)


@dataclass(slots=True)
class MutableClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _response(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def _config(path: Path, *, expected_sha256: str | None = None) -> LlamaCppConfig:
    content = path.read_bytes()
    return LlamaCppConfig(
        model_path=path,
        model_repository="test/repository",
        model_revision="abc123",
        model_filename=path.name,
        expected_sha256=expected_sha256 or hashlib.sha256(content).hexdigest(),
        expected_size_bytes=len(content),
        n_ctx=2_048,
        n_threads=2,
        n_threads_batch=4,
        seed=42,
    )


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny-test.gguf"
    path.write_bytes(b"test-only-not-a-real-model")
    return path


def test_client_is_lazy_and_passes_safe_cpu_configuration(tmp_path: Path) -> None:
    path = _model_file(tmp_path)
    backend = StubBackend([_response("Inspect your second step.")])
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> StubBackend:
        factory_calls.append(kwargs)
        return backend

    client = LlamaCppModelClient(_config(path), backend_factory=factory)
    assert client.loaded is False
    assert factory_calls == []

    result = client.generate_text(
        TextGenerationRequest(
            prompt="private prompt",
            system_prompt="system policy",
            max_output_tokens=77,
            temperature=0.0,
        )
    )

    assert client.loaded is True
    assert client.timeout_stopping_supported is True
    assert result.content == "Inspect your second step."
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 25
    assert result.model_name == "test/repository@abc123:tiny-test.gguf"
    assert result.finish_reason == "stop"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert factory_calls == [
        {
            "model_path": str(path.resolve()),
            "n_ctx": 2_048,
            "n_threads": 2,
            "n_threads_batch": 4,
            "seed": 42,
            "chat_format": "chatml",
            "n_gpu_layers": 0,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": False,
        }
    ]
    sent = backend.requests[0]
    assert sent["messages"] == [
        {"role": "system", "content": "system policy"},
        {"role": "user", "content": "private prompt"},
    ]
    assert sent["max_tokens"] == 77
    assert sent["temperature"] == 0.0
    assert sent["top_p"] == 1.0
    assert sent["seed"] == 42
    assert sent["stream"] is False
    assert "response_format" not in sent


def test_structured_generation_uses_json_schema_grammar(tmp_path: Path) -> None:
    path = _model_file(tmp_path)
    backend = StubBackend([_response('{"status":"incorrect"}')])
    client = LlamaCppModelClient(_config(path), backend_factory=lambda **_kwargs: backend)

    result = client.generate_structured(
        StructuredGenerationRequest(
            prompt="diagnose",
            schema_name="diagnosis_v1",
            response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            max_output_tokens=512,
        )
    )

    assert result.content == {"status": "incorrect"}
    assert backend.requests[0]["response_format"] == {
        "type": "json_object",
        "schema": {"type": "object", "properties": {"status": {"type": "string"}}},
    }
    assert client.structured_call_count == 1
    assert client.text_call_count == 0


def test_artifact_hash_is_checked_before_backend_load(tmp_path: Path) -> None:
    path = _model_file(tmp_path)
    factory_called = False

    def factory(**_kwargs: Any) -> StubBackend:
        nonlocal factory_called
        factory_called = True
        return StubBackend([])

    client = LlamaCppModelClient(
        _config(path, expected_sha256="0" * 64),
        backend_factory=factory,
    )

    with pytest.raises(LlamaCppConfigurationError, match="SHA-256 mismatch"):
        client.generate_text(TextGenerationRequest(prompt="hello"))

    assert factory_called is False
    assert client.loaded is False


def test_config_rejects_non_gguf_and_public_metadata_hides_directories(tmp_path: Path) -> None:
    wrong_suffix = tmp_path / "model.bin"
    wrong_suffix.write_bytes(b"not a gguf")

    with pytest.raises(LlamaCppConfigurationError, match=".gguf"):
        _config(wrong_suffix)

    path = _model_file(tmp_path)
    metadata = _config(path).public_metadata()
    assert metadata["model_filename"] == path.name
    assert str(tmp_path) not in str(metadata)
    assert metadata["n_gpu_layers"] == 0


def test_default_artifact_metadata_is_exact_and_pinned(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_QWEN_FILENAME
    path.write_bytes(b"metadata-only test fixture")
    config = LlamaCppConfig(
        model_path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_size_bytes=path.stat().st_size,
    )

    assert config.model_repository == DEFAULT_QWEN_REPOSITORY
    assert config.model_revision == DEFAULT_QWEN_REVISION
    assert DEFAULT_QWEN_SHA256 == "9614a50f03c897028920ca0dc4365da570bf587f9ee7768261216fe370b37e8e"
    assert DEFAULT_QWEN_SIZE_BYTES == 986_048_832


def test_timeout_criterion_discards_partial_model_output(tmp_path: Path) -> None:
    path = _model_file(tmp_path)
    clock = MutableClock()

    class TimeoutBackend:
        def create_chat_completion(self, **kwargs: Any) -> Mapping[str, Any]:
            clock.value = 2.0
            criteria = kwargs["stopping_criteria"]
            assert isinstance(criteria, list)
            assert criteria[0](None, None) is True
            return _response("partial output must not escape")

    client = LlamaCppModelClient(
        _config(path),
        backend_factory=lambda **_kwargs: TimeoutBackend(),
        monotonic=clock,
    )

    with pytest.raises(ModelTimeoutError, match="timed out"):
        client.generate_text(TextGenerationRequest(prompt="hello", timeout_seconds=1.0))


def test_chat_api_without_stopping_criteria_remains_compatible(tmp_path: Path) -> None:
    path = _model_file(tmp_path)

    class LegacyBackend:
        def create_chat_completion(
            self,
            *,
            messages: object,
            max_tokens: int,
            temperature: float,
            top_p: float,
            seed: int,
            stream: bool,
        ) -> Mapping[str, Any]:
            assert messages
            assert max_tokens == 4
            assert temperature == 0.0
            assert top_p == 1.0
            assert seed == 42
            assert stream is False
            return _response("compatible")

    client = LlamaCppModelClient(
        _config(path),
        backend_factory=lambda **_kwargs: cast(Any, LegacyBackend()),
    )

    result = client.generate_text(TextGenerationRequest(prompt="hello", max_output_tokens=4))

    assert result.content == "compatible"
    assert client.timeout_stopping_supported is False


@pytest.mark.parametrize("content", ["", "[]", "not-json"])
def test_invalid_structured_output_is_rejected(tmp_path: Path, content: str) -> None:
    path = _model_file(tmp_path)
    backend = StubBackend([_response(content)])
    client = LlamaCppModelClient(_config(path), backend_factory=lambda **_kwargs: backend)
    request = StructuredGenerationRequest(
        prompt="diagnose",
        schema_name="diagnosis_v1",
        response_schema={"type": "object"},
    )

    with pytest.raises(ModelOutputError):
        client.generate_structured(request)
