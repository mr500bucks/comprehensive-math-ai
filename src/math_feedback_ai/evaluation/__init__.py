"""Deterministic evaluation utilities for tutoring research."""

from math_feedback_ai.evaluation.diagnosis_pilot_builder import load_diagnosis_pilot
from math_feedback_ai.evaluation.metrics import (
    MetricResult,
    SystemObservation,
    SystemSummary,
    accuracy,
    false_rejection_rate,
    localization_accuracy,
    macro_f1,
    rate,
    summarize_system_metrics,
)
from math_feedback_ai.evaluation.pilot import (
    DiagnosisPilotCaseV1,
    HumanReviewStatus,
    validate_pilot_cases,
)
from math_feedback_ai.evaluation.tutoring import (
    HintGenerationObservation,
    HintReviewRecord,
    ModelCallObservation,
    TutoringArtifactPaths,
    TutoringCasePrediction,
    TutoringEvaluationCase,
    TutoringExperimentReport,
    TutoringMetricSnapshot,
    TutoringUsageSummary,
    development_tutoring_cases,
    run_tutoring_experiment,
    write_tutoring_artifacts,
)

__all__ = [
    "DiagnosisPilotCaseV1",
    "HumanReviewStatus",
    "HintGenerationObservation",
    "HintReviewRecord",
    "MetricResult",
    "ModelCallObservation",
    "SystemObservation",
    "SystemSummary",
    "TutoringArtifactPaths",
    "TutoringCasePrediction",
    "TutoringEvaluationCase",
    "TutoringExperimentReport",
    "TutoringMetricSnapshot",
    "TutoringUsageSummary",
    "accuracy",
    "development_tutoring_cases",
    "false_rejection_rate",
    "localization_accuracy",
    "load_diagnosis_pilot",
    "macro_f1",
    "rate",
    "run_tutoring_experiment",
    "summarize_system_metrics",
    "validate_pilot_cases",
    "write_tutoring_artifacts",
]
