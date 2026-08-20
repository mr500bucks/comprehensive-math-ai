"""Deterministic model client for unit and integration tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from math_feedback_ai.model.client import (
    GenerationResult,
    ModelClientError,
    StructuredContent,
    StructuredGenerationRequest,
    TextGenerationRequest,
)


class FakeModelExhaustedError(ModelClientError):
    """Raised when a test invokes an unscripted fake-model operation."""


TextScriptItem = str | GenerationResult[str] | Exception
StructuredScriptItem = StructuredContent | GenerationResult[StructuredContent] | Exception


@dataclass(slots=True)
class FakeModelClient:
    """Return scripted responses in FIFO order and record every request.

    Raw scripted content is automatically wrapped in :class:`GenerationResult`.
    Scripted exceptions are raised unchanged, which makes timeout and transport
    failure paths deterministic to test.
    """

    text_responses: Iterable[TextScriptItem] = ()
    structured_responses: Iterable[StructuredScriptItem] = ()
    text_requests: list[TextGenerationRequest] = field(default_factory=list, init=False)
    structured_requests: list[StructuredGenerationRequest] = field(default_factory=list, init=False)
    _text_queue: deque[TextScriptItem] = field(init=False, repr=False)
    _structured_queue: deque[StructuredScriptItem] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._text_queue = deque(self.text_responses)
        self._structured_queue = deque(self.structured_responses)

    @property
    def text_call_count(self) -> int:
        return len(self.text_requests)

    @property
    def structured_call_count(self) -> int:
        return len(self.structured_requests)

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        self.text_requests.append(request)
        if not self._text_queue:
            raise FakeModelExhaustedError("no scripted text response remains")
        item = self._text_queue.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, GenerationResult):
            return item
        return GenerationResult(content=item, model_name="deterministic-fake")

    def generate_structured(
        self, request: StructuredGenerationRequest
    ) -> GenerationResult[StructuredContent]:
        self.structured_requests.append(request)
        if not self._structured_queue:
            raise FakeModelExhaustedError("no scripted structured response remains")
        item = self._structured_queue.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, GenerationResult):
            return item
        return GenerationResult(content=item, model_name="deterministic-fake")
