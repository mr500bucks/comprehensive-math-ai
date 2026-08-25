"""Run the provider-neutral complete tutoring pipeline on development fixtures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from math_feedback_ai.diagnosis.prompt import DIAGNOSIS_PROMPT_VERSION
from math_feedback_ai.diagnosis.service import DiagnosisServiceConfig
from math_feedback_ai.domain.models import DiagnosisV1
from math_feedback_ai.evaluation.benchmark import load_development_benchmark
from math_feedback_ai.evaluation.reviewed_diagnosis_pilot import (
    REVIEWED_PILOT_VERSION,
    load_reviewed_diagnosis_pilot,
)
from math_feedback_ai.evaluation.tutoring import (
    development_tutoring_cases,
    reviewed_pilot_tutoring_cases,
    run_tutoring_experiment,
    write_tutoring_artifacts,
)
from math_feedback_ai.hints.generator import HINT_PROMPT_VERSION
from math_feedback_ai.model.client import (
    GenerationResult,
    ModelClient,
    ModelOutputError,
    StructuredContent,
    StructuredGenerationRequest,
    TextGenerationRequest,
)
from math_feedback_ai.model.fake import FakeModelClient
from math_feedback_ai.model.llama_cpp import (
    DEFAULT_QWEN_FILENAME,
    DEFAULT_QWEN_REPOSITORY,
    DEFAULT_QWEN_REVISION,
    DEFAULT_QWEN_SHA256,
    DEFAULT_QWEN_SIZE_BYTES,
    LlamaCppConfig,
    LlamaCppModelClient,
)

DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/results/tutoring")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=("development", "pilot-reviewed"),
        default="development",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        help="explicit JSONL override for the selected benchmark",
    )
    parser.add_argument(
        "--provider",
        choices=("none", "llama-cpp"),
        default="none",
        help="none runs a deterministic safe-abstention control",
    )
    parser.add_argument("--model-path", type=Path, help="verified local GGUF model path")
    parser.add_argument("--model-repository", default=DEFAULT_QWEN_REPOSITORY)
    parser.add_argument("--model-revision", default=DEFAULT_QWEN_REVISION)
    parser.add_argument("--model-filename", default=DEFAULT_QWEN_FILENAME)
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--model-sha256", default=DEFAULT_QWEN_SHA256)
    parser.add_argument("--model-size-bytes", type=_positive_int, default=DEFAULT_QWEN_SIZE_BYTES)
    parser.add_argument("--llama-context-window", type=_positive_int, default=4_096)
    parser.add_argument("--llama-threads", type=_positive_int, default=6)
    parser.add_argument("--llama-batch-threads", type=_positive_int, default=12)
    parser.add_argument("--llama-seed", type=_nonnegative_int, default=1_337)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--diagnosis-max-output-tokens", type=_positive_int, default=2_048)
    parser.add_argument("--minimum-confidence", type=_confidence, default=0.5)
    parser.add_argument("--diagnosis-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--generic-response-schema",
        action="store_true",
        help="disable input-specific diagnosis schema constraints for a baseline",
    )
    parser.add_argument(
        "--oracle-diagnosis",
        action="store_true",
        help=(
            "use fixture diagnoses and exercise policy/hint/safety only; "
            "this is not an end-to-end model-quality result"
        ),
    )
    parser.add_argument(
        "--without-references",
        action="store_true",
        help="omit non-exhaustive references from diagnosis and hint safety context",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--artifact-stem",
        help="safe filename stem override for parallel or repeated experiment runs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing artifacts for this benchmark",
    )
    return parser


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _confidence(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must lie between 0 and 1")
    return parsed


def _validate_provider_options(namespace: argparse.Namespace) -> None:
    provider = cast(str, namespace.provider)
    model_path = cast(Path | None, namespace.model_path)
    if provider == "none" and model_path is not None:
        raise ValueError("--model-path requires --provider llama-cpp")
    if provider == "llama-cpp" and model_path is None:
        raise ValueError("--model-path is required for --provider llama-cpp")


def _make_client(
    namespace: argparse.Namespace,
    *,
    maximum_diagnosis_calls: int,
) -> tuple[ModelClient, dict[str, object]]:
    if cast(str, namespace.provider) == "none":
        return (
            FakeModelClient(
                structured_responses=(
                    ModelOutputError("no live model provider is configured")
                    for _ in range(maximum_diagnosis_calls)
                )
            ),
            {"provider": "none", "behavior": "safe_abstention_control"},
        )

    model_path = cast(Path, namespace.model_path)
    config = LlamaCppConfig(
        model_path=model_path,
        model_repository=cast(str, namespace.model_repository),
        model_revision=cast(str, namespace.model_revision),
        model_filename=cast(str, namespace.model_filename),
        expected_sha256=cast(str, namespace.model_sha256),
        expected_size_bytes=cast(int, namespace.model_size_bytes),
        n_ctx=cast(int, namespace.llama_context_window),
        n_threads=cast(int, namespace.llama_threads),
        n_threads_batch=cast(int, namespace.llama_batch_threads),
        seed=cast(int, namespace.llama_seed),
        quantization=cast(str, namespace.quantization),
    )
    metadata = config.public_metadata()
    try:
        metadata["runtime_version"] = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        metadata["runtime_version"] = "not_installed"
    return LlamaCppModelClient(config), metadata


@dataclass(slots=True)
class _OracleDiagnosisClient:
    """Route structured calls to fixture diagnoses and text calls to a real client."""

    inner: ModelClient
    diagnoses: Iterator[DiagnosisV1]

    def generate_structured(
        self,
        _request: StructuredGenerationRequest,
    ) -> GenerationResult[StructuredContent]:
        try:
            diagnosis = next(self.diagnoses)
        except StopIteration:
            raise ModelOutputError("oracle diagnosis fixture was exhausted") from None
        return GenerationResult(
            content=diagnosis.model_dump(mode="json"),
            model_name="development_gold_diagnosis_oracle",
            finish_reason="fixture",
            latency_ms=0.0,
        )

    def generate_text(self, request: TextGenerationRequest) -> GenerationResult[str]:
        return self.inner.generate_text(request)


def _run(namespace: argparse.Namespace) -> int:
    _validate_provider_options(namespace)
    benchmark_path = cast(Path | None, namespace.benchmark_path)
    benchmark = cast(str, namespace.benchmark)
    if benchmark == "pilot-reviewed":
        reviewed = load_reviewed_diagnosis_pilot(benchmark_path)
        cases = reviewed_pilot_tutoring_cases(reviewed)
        benchmark_name = REVIEWED_PILOT_VERSION
        annotation_status = "reviewed"
        schema_version = "1.1"
        oracle_diagnoses = (case.reviewed_diagnosis for case in reviewed)
    else:
        examples = load_development_benchmark(benchmark_path)
        cases = development_tutoring_cases(examples)
        benchmark_name = "development_v1"
        annotation_status = "not_human_validated"
        schema_version = "1.0"
        oracle_diagnoses = (example.gold_diagnosis for example in examples)
    attempts = cast(int, namespace.diagnosis_attempts)
    provider_client, provider_metadata = _make_client(
        namespace,
        maximum_diagnosis_calls=len(cases) * attempts,
    )
    oracle_diagnosis = cast(bool, namespace.oracle_diagnosis)
    client: ModelClient = provider_client
    if oracle_diagnosis:
        client = _OracleDiagnosisClient(
            inner=provider_client,
            diagnoses=iter(oracle_diagnoses),
        )
        provider_metadata["diagnosis_source"] = (
            "development_gold_fixture"
            if benchmark == "development"
            else f"{benchmark_name}_gold_fixture"
        )
        provider_metadata["qualification"] = (
            "Oracle-diagnosis hint/safety diagnostic; not end-to-end model quality."
        )
    service_config = DiagnosisServiceConfig(
        timeout_seconds=cast(float, namespace.timeout_seconds),
        max_output_tokens=cast(int, namespace.diagnosis_max_output_tokens),
        minimum_confidence=cast(float, namespace.minimum_confidence),
        max_attempts=attempts,
        specialize_response_schema=not cast(bool, namespace.generic_response_schema),
    )
    report = run_tutoring_experiment(
        benchmark_name=benchmark_name,
        annotation_status=annotation_status,
        cases=cases,
        client=client,
        diagnosis_config=service_config,
        hint_timeout_seconds=cast(float, namespace.timeout_seconds),
        use_references=not cast(bool, namespace.without_references),
    )
    if isinstance(provider_client, LlamaCppModelClient):
        provider_metadata["timeout_token_stopping_supported"] = (
            provider_client.timeout_stopping_supported
        )
    metadata: dict[str, object] = {
        "benchmark_version": benchmark_name,
        "diagnosis_schema_version": schema_version,
        "benchmark_source": (
            str(benchmark_path.resolve()) if benchmark_path is not None else "committed_or_built_in"
        ),
        "prompt_versions": {
            "diagnosis": DIAGNOSIS_PROMPT_VERSION,
            "hint": HINT_PROMPT_VERSION,
        },
        "diagnosis_service": {
            "timeout_seconds": service_config.timeout_seconds,
            "max_output_tokens": service_config.max_output_tokens,
            "minimum_confidence": service_config.minimum_confidence,
            "max_attempts": service_config.max_attempts,
            "specialize_response_schema": service_config.specialize_response_schema,
        },
        "hint_timeout_seconds": cast(float, namespace.timeout_seconds),
        "oracle_diagnosis": oracle_diagnosis,
        "reference_mode": (
            "without_references" if cast(bool, namespace.without_references) else "with_references"
        ),
        "provider": provider_metadata,
    }
    paths = write_tutoring_artifacts(
        report,
        cast(Path, namespace.output_dir),
        stem=cast(str | None, namespace.artifact_stem),
        run_metadata=metadata,
        overwrite=cast(bool, namespace.overwrite),
    )
    payload = {
        "benchmark": report.benchmark_name,
        "annotation_status": report.annotation_status,
        "summary": str(paths.summary.resolve()),
        "predictions": str(paths.predictions.resolve()),
        "hint_review": str(paths.hint_review.resolve()),
        "examples": report.examples,
        "tutor_action_agreement": report.metrics["tutor_action_agreement"].value,
        "reveal_compliance": report.metrics["reveal_compliance"].value,
        "leakage_violation_rate": report.metrics["leakage_violation_rate"].value,
        "model_calls": report.provider_usage.model_calls,
        "research_qualification": (
            "Synthetic development labels are not evidence of tutoring efficacy."
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        return _run(namespace)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
