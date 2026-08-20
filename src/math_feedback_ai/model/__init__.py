"""Provider-neutral model access contracts."""

from math_feedback_ai.model.client import (
    GenerationResult,
    ModelClient,
    ModelClientError,
    ModelOutputError,
    ModelTimeoutError,
    ModelTransportError,
    StructuredContent,
    StructuredGenerationRequest,
    TextGenerationRequest,
    TokenUsage,
)

__all__ = [
    "GenerationResult",
    "ModelClient",
    "ModelClientError",
    "ModelOutputError",
    "ModelTimeoutError",
    "ModelTransportError",
    "StructuredContent",
    "StructuredGenerationRequest",
    "TextGenerationRequest",
    "TokenUsage",
]
