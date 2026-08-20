"""Small provider-neutral interfaces for language-model generation.

The domain and tutoring layers depend on these contracts rather than on a
specific hosted API or local-model runtime.  Provider adapters are expected to
translate their native errors into the error hierarchy defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

type StructuredContent = Mapping[str, Any] | str
"""A structured response, either decoded JSON-like data or its JSON text."""


class ModelClientError(RuntimeError):
    """Base class for failures while invoking a model provider."""


class ModelTimeoutError(ModelClientError):
    """The provider did not finish within the request timeout."""


class ModelTransportError(ModelClientError):
    """The provider could not be reached or returned a transport-level error."""


class ModelOutputError(ModelClientError):
    """The provider returned output that could not be exposed to the caller."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported usage; fields are ``None`` when unavailable."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided")


@dataclass(frozen=True, slots=True)
class TextGenerationRequest:
    """A provider-independent text generation request."""

    prompt: str
    system_prompt: str | None = None
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1_024
    temperature: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_common_request_fields(
            prompt=self.prompt,
            timeout_seconds=self.timeout_seconds,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


@dataclass(frozen=True, slots=True)
class StructuredGenerationRequest:
    """A request for JSON-compatible structured content.

    ``response_schema`` is descriptive.  Callers must still validate the
    returned content because not every provider can enforce JSON Schema.
    """

    prompt: str
    response_schema: Mapping[str, Any]
    schema_name: str
    system_prompt: str | None = None
    timeout_seconds: float = 30.0
    max_output_tokens: int = 2_048
    temperature: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_common_request_fields(
            prompt=self.prompt,
            timeout_seconds=self.timeout_seconds,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        if not self.schema_name.strip():
            raise ValueError("schema_name must not be blank")
        if not self.response_schema:
            raise ValueError("response_schema must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationResult[ContentT]:
    """Generated content plus optional provider metadata."""

    content: ContentT
    usage: TokenUsage = field(default_factory=TokenUsage)
    model_name: str | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative when provided")


@runtime_checkable
class ModelClient(Protocol):
    """Minimal synchronous interface implemented by every model provider."""

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        """Generate ordinary text or raise a :class:`ModelClientError`."""

    def generate_structured(
        self, request: StructuredGenerationRequest
    ) -> GenerationResult[StructuredContent]:
        """Generate structured content or raise a :class:`ModelClientError`."""


def _validate_common_request_fields(
    *,
    prompt: str,
    timeout_seconds: float,
    max_output_tokens: int,
    temperature: float,
) -> None:
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
