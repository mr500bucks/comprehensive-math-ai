"""Lazy local GGUF adapter backed by optional ``llama-cpp-python``.

The module itself has no dependency on llama-cpp-python.  Native bindings and
model weights are loaded only on the first generation request, after the local
artifact has passed exact size and SHA-256 verification.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from math_feedback_ai.model.client import (
    GenerationResult,
    ModelOutputError,
    ModelTimeoutError,
    ModelTransportError,
    StructuredContent,
    StructuredGenerationRequest,
    TextGenerationRequest,
    TokenUsage,
)

DEFAULT_QWEN_REPOSITORY = "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF"
DEFAULT_QWEN_REVISION = "951ed2aea09c43e331c612e74d83e4a23ca98e3b"
DEFAULT_QWEN_FILENAME = "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
DEFAULT_QWEN_SHA256 = "9614a50f03c897028920ca0dc4365da570bf587f9ee7768261216fe370b37e8e"
DEFAULT_QWEN_SIZE_BYTES = 986_048_832


class LlamaCppConfigurationError(ValueError):
    """Local runtime or model configuration is invalid."""


class LlamaCppBackend(Protocol):
    """Subset of the llama-cpp high-level API used by this adapter."""

    def create_chat_completion(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one non-streaming chat completion."""


type BackendFactory = Callable[..., LlamaCppBackend]
type StoppingListFactory = Callable[[list[object]], object]


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Reproducible CPU configuration for one verified local GGUF artifact."""

    model_path: Path
    model_repository: str = DEFAULT_QWEN_REPOSITORY
    model_revision: str = DEFAULT_QWEN_REVISION
    model_filename: str = DEFAULT_QWEN_FILENAME
    expected_sha256: str = DEFAULT_QWEN_SHA256
    expected_size_bytes: int = DEFAULT_QWEN_SIZE_BYTES
    n_ctx: int = 4_096
    n_threads: int = 6
    n_threads_batch: int = 12
    seed: int = 1_337
    chat_format: str = "chatml"

    def __post_init__(self) -> None:
        unresolved = self.model_path.absolute()
        if str(unresolved).startswith("\\\\"):
            raise LlamaCppConfigurationError("model_path must not use a network share")
        if any(
            component.is_symlink() or os.path.isjunction(component)
            for component in (unresolved, *unresolved.parents)
        ):
            raise LlamaCppConfigurationError("model_path must not traverse links or junctions")
        try:
            resolved = self.model_path.resolve(strict=True)
        except OSError as exc:
            raise LlamaCppConfigurationError("local GGUF model file does not exist") from exc
        if not resolved.is_file():
            raise LlamaCppConfigurationError("model_path must be a regular non-symlink file")
        if resolved.suffix.lower() != ".gguf":
            raise LlamaCppConfigurationError("model_path must point to a .gguf file")
        if not self.model_repository.strip() or not self.model_revision.strip():
            raise LlamaCppConfigurationError("model repository and revision must not be blank")
        if not self.model_filename.strip() or Path(self.model_filename).name != self.model_filename:
            raise LlamaCppConfigurationError("model_filename must be one local file name")
        if resolved.name != self.model_filename.strip():
            raise LlamaCppConfigurationError("local GGUF file name does not match model_filename")
        digest = self.expected_sha256.lower().strip()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise LlamaCppConfigurationError("expected_sha256 must be a 64-character hex digest")
        if self.expected_size_bytes <= 0:
            raise LlamaCppConfigurationError("expected_size_bytes must be positive")
        if not 512 <= self.n_ctx <= 131_072:
            raise LlamaCppConfigurationError("n_ctx must lie between 512 and 131072")
        if self.n_threads <= 0 or self.n_threads_batch <= 0:
            raise LlamaCppConfigurationError("thread counts must be positive")
        if self.seed < 0:
            raise LlamaCppConfigurationError("seed must be non-negative")
        if not self.chat_format.strip():
            raise LlamaCppConfigurationError("chat_format must not be blank")
        object.__setattr__(self, "model_path", resolved)
        object.__setattr__(self, "model_repository", self.model_repository.strip())
        object.__setattr__(self, "model_revision", self.model_revision.strip())
        object.__setattr__(self, "model_filename", self.model_filename.strip())
        object.__setattr__(self, "expected_sha256", digest)
        object.__setattr__(self, "chat_format", self.chat_format.strip())

    def verify_artifact(self) -> None:
        """Fail closed unless the local bytes match the pinned artifact."""

        try:
            actual_size = self.model_path.stat().st_size
        except OSError as exc:
            raise LlamaCppConfigurationError("cannot inspect local GGUF model file") from exc
        if actual_size != self.expected_size_bytes:
            raise LlamaCppConfigurationError(
                f"GGUF size mismatch: expected {self.expected_size_bytes}, got {actual_size}"
            )
        digest = hashlib.sha256()
        try:
            with self.model_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1_048_576), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise LlamaCppConfigurationError("cannot hash local GGUF model file") from exc
        actual_digest = digest.hexdigest()
        if actual_digest != self.expected_sha256:
            raise LlamaCppConfigurationError(
                f"GGUF SHA-256 mismatch: expected {self.expected_sha256}, got {actual_digest}"
            )

    def public_metadata(self) -> dict[str, object]:
        """Return portable experiment metadata without exposing local directories."""

        return {
            "provider": "llama_cpp",
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "model_filename": self.model_filename,
            "model_sha256": self.expected_sha256,
            "model_size_bytes": self.expected_size_bytes,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads_batch,
            "seed": self.seed,
            "chat_format": self.chat_format,
            "n_gpu_layers": 0,
        }


@dataclass(slots=True)
class _DeadlineCriterion:
    deadline: float
    monotonic: Callable[[], float]
    triggered: bool = False

    def __call__(self, _input_ids: object, _scores: object, **_kwargs: object) -> bool:
        self.triggered = self.monotonic() >= self.deadline
        return self.triggered


class LlamaCppModelClient:
    """Synchronous, serialized, lazily loaded local ``ModelClient``.

    The timeout is best effort: token generation is stopped at the next
    stopping-criteria check, while native model loading or one long prompt
    evaluation cannot be interrupted safely by Python.
    """

    def __init__(
        self,
        config: LlamaCppConfig,
        *,
        backend_factory: BackendFactory | None = None,
        stopping_list_factory: StoppingListFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._stopping_list_factory = stopping_list_factory
        self._monotonic = monotonic
        self._backend: LlamaCppBackend | None = None
        self._artifact_verified = False
        self._supports_stopping_criteria: bool | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self.structured_call_count = 0
        self.text_call_count = 0

    @property
    def loaded(self) -> bool:
        """Whether native bindings and model weights have been initialized."""

        return self._backend is not None

    @property
    def timeout_stopping_supported(self) -> bool | None:
        """Whether the loaded chat API accepts token-level stopping criteria."""

        return self._supports_stopping_criteria

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        self.text_call_count += 1
        result = self._generate(request)
        return GenerationResult(
            content=result.content,
            usage=result.usage,
            model_name=result.model_name,
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
        )

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> GenerationResult[StructuredContent]:
        self.structured_call_count += 1
        result = self._generate(
            request,
            response_format={"type": "json_object", "schema": dict(request.response_schema)},
        )
        try:
            decoded: object = json.loads(result.content)
        except json.JSONDecodeError:
            raise ModelOutputError("llama-cpp returned malformed structured output") from None
        if not isinstance(decoded, dict):
            raise ModelOutputError("llama-cpp structured output was not a JSON object")
        return GenerationResult(
            content=cast(dict[str, Any], decoded),
            usage=result.usage,
            model_name=result.model_name,
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
        )

    def _generate(
        self,
        request: TextGenerationRequest | StructuredGenerationRequest,
        *,
        response_format: Mapping[str, Any] | None = None,
    ) -> GenerationResult[str]:
        started = self._monotonic()
        if not self._generation_lock.acquire(timeout=request.timeout_seconds):
            raise ModelTimeoutError("llama-cpp request timed out waiting for the local model")
        try:
            backend = self._ensure_backend()
            deadline = started + request.timeout_seconds
            if self._monotonic() >= deadline:
                raise ModelTimeoutError("llama-cpp request timed out while loading the local model")
            criterion = _DeadlineCriterion(deadline=deadline, monotonic=self._monotonic)
            messages: list[dict[str, str]] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "top_p": 1.0,
                "seed": self.config.seed,
                "stream": False,
            }
            if self._supports_stopping_criteria:
                kwargs["stopping_criteria"] = self._make_stopping_list([criterion])
            if response_format is not None:
                kwargs["response_format"] = dict(response_format)
            try:
                raw = backend.create_chat_completion(**kwargs)
            except Exception as exc:
                if criterion.triggered or self._monotonic() >= deadline:
                    raise ModelTimeoutError("llama-cpp generation timed out") from None
                if isinstance(exc, MemoryError):
                    raise ModelTransportError(
                        "llama-cpp exhausted memory during generation"
                    ) from None
                raise ModelTransportError("llama-cpp generation failed") from None
            if criterion.triggered or self._monotonic() >= deadline:
                raise ModelTimeoutError("llama-cpp generation timed out")
            return self._parse_result(raw, latency_ms=(self._monotonic() - started) * 1_000)
        finally:
            self._generation_lock.release()

    def _ensure_backend(self) -> LlamaCppBackend:
        if self._backend is not None:
            return self._backend
        with self._load_lock:
            if self._backend is not None:
                return self._backend
            if not self._artifact_verified:
                self.config.verify_artifact()
                self._artifact_verified = True
            factory = self._backend_factory
            if factory is None:
                try:
                    module = importlib.import_module("llama_cpp")
                    factory = cast(BackendFactory, module.Llama)
                    if self._stopping_list_factory is None:
                        self._stopping_list_factory = cast(
                            StoppingListFactory,
                            module.StoppingCriteriaList,
                        )
                except (ImportError, AttributeError):
                    raise ModelTransportError(
                        "llama-cpp-python is not installed with the required high-level API"
                    ) from None
            try:
                self._backend = factory(
                    model_path=str(self.config.model_path),
                    n_ctx=self.config.n_ctx,
                    n_threads=self.config.n_threads,
                    n_threads_batch=self.config.n_threads_batch,
                    seed=self.config.seed,
                    chat_format=self.config.chat_format,
                    n_gpu_layers=0,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
            except Exception as exc:
                if isinstance(exc, MemoryError):
                    raise ModelTransportError(
                        "llama-cpp exhausted memory while loading the model"
                    ) from None
                raise ModelTransportError("llama-cpp could not load the verified model") from None
            self._supports_stopping_criteria = _accepts_keyword(
                self._backend.create_chat_completion,
                "stopping_criteria",
            )
            return self._backend

    def _make_stopping_list(self, criteria: list[object]) -> object:
        if self._stopping_list_factory is not None:
            return self._stopping_list_factory(criteria)
        # Injected test backends accept the list directly. A real backend sets
        # the native StoppingCriteriaList factory during lazy import.
        return criteria

    def _parse_result(
        self,
        raw: Mapping[str, Any],
        *,
        latency_ms: float,
    ) -> GenerationResult[str]:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelOutputError("llama-cpp response omitted its first choice")
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelOutputError("llama-cpp response contained no output text")
        finish_reason = choice.get("finish_reason")
        usage = _extract_usage(raw.get("usage"))
        return GenerationResult(
            content=content,
            usage=usage,
            model_name=(
                f"{self.config.model_repository}@{self.config.model_revision}:"
                f"{self.config.model_filename}"
            ),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            latency_ms=latency_ms,
        )


def _extract_usage(raw_usage: object) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_optional_nonnegative_int(raw_usage.get("prompt_tokens")),
        output_tokens=_optional_nonnegative_int(raw_usage.get("completion_tokens")),
        total_tokens=_optional_nonnegative_int(raw_usage.get("total_tokens")),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _accepts_keyword(callable_object: Callable[..., object], keyword: str) -> bool:
    """Conservatively detect version-dependent llama-cpp keyword support."""

    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
