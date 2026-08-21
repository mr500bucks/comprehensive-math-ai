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
from math_feedback_ai.model.llama_cpp import (
    LlamaCppConfig,
    LlamaCppConfigurationError,
    LlamaCppModelClient,
)
from math_feedback_ai.model.openai_responses import (
    OpenAIConfigurationError,
    OpenAIResponsesClient,
    OpenAIResponsesConfig,
)

__all__ = [
    "GenerationResult",
    "LlamaCppConfig",
    "LlamaCppConfigurationError",
    "LlamaCppModelClient",
    "ModelClient",
    "ModelClientError",
    "ModelOutputError",
    "ModelTimeoutError",
    "ModelTransportError",
    "OpenAIConfigurationError",
    "OpenAIResponsesClient",
    "OpenAIResponsesConfig",
    "StructuredContent",
    "StructuredGenerationRequest",
    "TextGenerationRequest",
    "TokenUsage",
]
