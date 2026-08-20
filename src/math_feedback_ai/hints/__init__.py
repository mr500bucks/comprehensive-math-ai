"""Reveal-constrained hint generation and safety checks."""

from .generator import HINT_PROMPT_VERSION, HintGenerationResult, HintGenerator
from .leakage import (
    HintSafetyContext,
    LeakageChecker,
    LeakageCheckResult,
    LeakageSeverity,
    LeakageViolation,
    LeakageViolationCode,
)

__all__ = [
    "HINT_PROMPT_VERSION",
    "HintGenerationResult",
    "HintGenerator",
    "HintSafetyContext",
    "LeakageCheckResult",
    "LeakageChecker",
    "LeakageSeverity",
    "LeakageViolation",
    "LeakageViolationCode",
]
