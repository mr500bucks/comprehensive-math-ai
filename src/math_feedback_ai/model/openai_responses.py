"""OpenAI Responses API adapter for the provider-neutral model boundary.

The adapter deliberately uses only Python's standard library.  This keeps the
core package small and makes provider activation an explicit runtime choice.
No network access occurs during construction or import.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
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

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_MAX_RESPONSE_BYTES = 16 * 1_024 * 1_024


class OpenAIConfigurationError(ValueError):
    """OpenAI runtime configuration is absent or unsafe."""


class _NonRetryableOpenAITransportError(ModelTransportError):
    """A provider/credential rejection that another identical call cannot fix."""


class JsonTransport(Protocol):
    """Small injectable HTTP seam used to keep provider tests offline."""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """POST JSON and return a decoded JSON object."""


@dataclass(frozen=True, slots=True)
class OpenAIResponsesConfig:
    """Configuration for an OpenAI Responses API client.

    ``api_key`` is intentionally excluded from repr so diagnostics and test
    failures cannot accidentally reveal it.  The model must be selected
    explicitly rather than silently changing as provider defaults evolve.
    """

    model: str
    api_key: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    organization: str | None = None
    project: str | None = None
    max_retries: int = 1
    strict_schema: bool = True

    def __post_init__(self) -> None:
        model = self.model.strip()
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        organization = self.organization.strip() if self.organization else None
        project = self.project.strip() if self.project else None
        if not model:
            raise OpenAIConfigurationError("OpenAI model must not be blank")
        if not api_key:
            raise OpenAIConfigurationError("OPENAI_API_KEY is required")
        if "\r" in api_key or "\n" in api_key:
            raise OpenAIConfigurationError("OPENAI_API_KEY contains an invalid line break")
        for field_name, value in (("organization", organization), ("project", project)):
            if value is not None and ("\r" in value or "\n" in value):
                raise OpenAIConfigurationError(
                    f"OpenAI {field_name} contains an invalid line break"
                )
        if isinstance(self.max_retries, bool) or self.max_retries not in (0, 1, 2):
            raise OpenAIConfigurationError("max_retries must be 0, 1, or 2")
        if not isinstance(self.strict_schema, bool):
            raise OpenAIConfigurationError("strict_schema must be boolean")
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise OpenAIConfigurationError(
                "OpenAI base URL must be an HTTPS origin/path without credentials, query, or fragment"
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "organization", organization)
        object.__setattr__(self, "project", project)

    def __repr__(self) -> str:
        return (
            "OpenAIResponsesConfig("
            f"model={self.model!r}, api_key='<redacted>', base_url={self.base_url!r}, "
            f"organization={'<configured>' if self.organization else None!r}, "
            f"project={'<configured>' if self.project else None!r}, "
            f"max_retries={self.max_retries!r}, strict_schema={self.strict_schema!r})"
        )

    def public_metadata(self) -> dict[str, object]:
        """Return reproducibility metadata that is safe to serialize."""

        return {
            "provider": "openai_responses",
            "model": self.model,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "strict_schema": self.strict_schema,
            "store": False,
            "organization_configured": self.organization is not None,
            "project_configured": self.project is not None,
        }

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 1,
    ) -> OpenAIResponsesConfig:
        """Read normal OpenAI environment configuration without logging it."""

        selected_model = model or os.environ.get("OPENAI_MODEL", "")
        return cls(
            model=selected_model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            organization=(
                os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION") or None
            ),
            project=os.environ.get("OPENAI_PROJECT_ID") or None,
            max_retries=max_retries,
        )


class UrllibJsonTransport:
    """Blocking HTTPS JSON transport with explicit timeout/error translation."""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 504):
                raise ModelTimeoutError(f"OpenAI request timed out with HTTP {exc.code}") from None
            if exc.code not in (409, 429) and exc.code < 500:
                raise _NonRetryableOpenAITransportError(
                    f"OpenAI request was rejected with HTTP {exc.code}"
                ) from None
            raise ModelTransportError(f"OpenAI request failed with HTTP {exc.code}") from None
        except TimeoutError:
            raise ModelTimeoutError("OpenAI request timed out") from None
        except (urllib.error.URLError, OSError):
            raise ModelTransportError("OpenAI request could not reach the provider") from None

        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise ModelOutputError("OpenAI response exceeded the configured safety limit")

        try:
            decoded: object = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelOutputError("OpenAI returned a non-JSON response") from None
        if not isinstance(decoded, dict):
            raise ModelOutputError("OpenAI returned a JSON response that was not an object")
        return cast(dict[str, Any], decoded)


class OpenAIResponsesClient:
    """Synchronous `ModelClient` implementation for ``POST /v1/responses``."""

    def __init__(
        self,
        config: OpenAIResponsesConfig,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibJsonTransport()
        self.structured_call_count = 0
        self.text_call_count = 0
        self.request_attempt_count = 0

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        self.text_call_count += 1
        payload = self._common_payload(request)
        response, latency_ms = self._invoke(payload, request.timeout_seconds)
        return self._result(response, latency_ms=latency_ms)

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> GenerationResult[StructuredContent]:
        self.structured_call_count += 1
        payload = self._common_payload(request)
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.schema_name,
                "schema": _openai_response_schema(
                    request.response_schema,
                    strict=self.config.strict_schema,
                ),
                "strict": self.config.strict_schema,
            }
        }
        response, latency_ms = self._invoke(payload, request.timeout_seconds)
        text_result = self._result(response, latency_ms=latency_ms)
        try:
            decoded: object = json.loads(text_result.content)
        except json.JSONDecodeError:
            raise ModelOutputError("OpenAI returned malformed structured output") from None
        if not isinstance(decoded, dict):
            raise ModelOutputError("OpenAI structured output was not a JSON object")
        return GenerationResult(
            content=cast(dict[str, Any], decoded),
            usage=text_result.usage,
            model_name=text_result.model_name,
            finish_reason=text_result.finish_reason,
            latency_ms=text_result.latency_ms,
        )

    def _common_payload(
        self,
        request: TextGenerationRequest | StructuredGenerationRequest,
    ) -> dict[str, Any]:
        input_messages: list[dict[str, str]] = []
        if request.system_prompt:
            input_messages.append({"role": "system", "content": request.system_prompt})
        input_messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_messages,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "store": False,
        }
        if request.metadata:
            payload["metadata"] = dict(request.metadata)
        return payload

    def _invoke(
        self,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> tuple[Mapping[str, Any], float]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "math-feedback-ai/0.1.0",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project

        started = time.perf_counter()
        for attempt_index in range(self.config.max_retries + 1):
            try:
                self.request_attempt_count += 1
                response = self._transport.post_json(
                    url=f"{self.config.base_url.rstrip('/')}/responses",
                    headers=headers,
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                )
            except _NonRetryableOpenAITransportError:
                raise
            except (ModelTimeoutError, ModelTransportError):
                if attempt_index == self.config.max_retries:
                    raise
            else:
                return response, (time.perf_counter() - started) * 1_000
        raise AssertionError("bounded provider retry loop did not return or raise")

    def _result(
        self,
        response: Mapping[str, Any],
        *,
        latency_ms: float,
    ) -> GenerationResult[str]:
        status = response.get("status")
        if status != "completed":
            raise ModelOutputError(f"OpenAI response did not complete (status={status!r})")
        content = _extract_output_text(response)
        usage = _extract_usage(response.get("usage"))
        model = response.get("model")
        return GenerationResult(
            content=content,
            usage=usage,
            model_name=model if isinstance(model, str) else self.config.model,
            finish_reason=cast(str, status),
            latency_ms=latency_ms,
        )


def _extract_output_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise ModelOutputError("OpenAI response omitted its output array")
    fragments: list[str] = []
    refused = False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        blocks = item.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "refusal":
                refused = True
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                fragments.append(cast(str, block["text"]))
    text = "".join(fragments)
    if refused:
        raise ModelOutputError("OpenAI refused the generation request")
    if not text.strip():
        raise ModelOutputError("OpenAI response contained no output text")
    return text


def _extract_usage(raw_usage: object) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        return TokenUsage()
    details = raw_usage.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return TokenUsage(
        input_tokens=_optional_nonnegative_int(raw_usage.get("input_tokens")),
        output_tokens=_optional_nonnegative_int(raw_usage.get("output_tokens")),
        total_tokens=_optional_nonnegative_int(raw_usage.get("total_tokens")),
        cached_input_tokens=_optional_nonnegative_int(cached),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _openai_response_schema(
    schema: Mapping[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    """Convert ordinary application JSON Schema into the strict API subset.

    Pydantic emits defaults and marks fields with Python defaults as optional.
    OpenAI strict structured output instead requires every object property in
    ``required``; nullable fields express optional values.  Application-side
    validation still enforces the original, richer schema after generation.
    """

    if not strict:
        return dict(schema)

    def transform(value: object) -> object:
        if isinstance(value, Mapping):
            transformed = {
                str(key): transform(item)
                for key, item in value.items()
                if key not in ("default", "title")
            }
            properties = transformed.get("properties")
            if isinstance(properties, dict):
                transformed["additionalProperties"] = False
                transformed["required"] = list(properties)
            return transformed
        if isinstance(value, list):
            return [transform(item) for item in value]
        return value

    converted = transform(schema)
    if not isinstance(converted, dict):  # pragma: no cover - Mapping input guarantees this
        raise ModelOutputError("structured-output schema conversion failed")
    return cast(dict[str, Any], converted)
