"""Run reproducible diagnosis-only development or provisional-pilot experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from math_feedback_ai.diagnosis.prompt import DIAGNOSIS_PROMPT_VERSION
from math_feedback_ai.diagnosis.service import DiagnosisService, DiagnosisServiceConfig
from math_feedback_ai.evaluation.benchmark import load_development_benchmark
from math_feedback_ai.evaluation.diagnosis import (
    DiagnosisEvaluationCase,
    DiagnosisExperimentReport,
    ReferenceMode,
    development_evaluation_cases,
    diagnosis_case_set_sha256,
    pilot_evaluation_cases,
    reviewed_pilot_evaluation_cases,
    run_diagnosis_experiment,
    write_diagnosis_artifacts,
)
from math_feedback_ai.evaluation.diagnosis_pilot_builder import load_diagnosis_pilot
from math_feedback_ai.evaluation.failure_analysis import (
    analyze_diagnosis_failures,
    compare_reference_ablation,
    write_analysis_report,
)
from math_feedback_ai.evaluation.reviewed_diagnosis_pilot import (
    REVIEWED_PILOT_VERSION,
    load_reviewed_diagnosis_pilot,
)
from math_feedback_ai.model.client import ModelClient, ModelOutputError
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
from math_feedback_ai.model.openai_responses import (
    OpenAIResponsesClient,
    OpenAIResponsesConfig,
)

DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/results/diagnosis")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=("development", "pilot", "pilot-reviewed"),
        required=True,
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        help="explicit JSONL override; omit for the committed or built-in benchmark",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("with", "without", "both"),
        default="both",
        help="run diagnosis with references, without references, or both conditions",
    )
    parser.add_argument(
        "--provider",
        choices=("none", "openai", "llama-cpp"),
        default="none",
        help="none runs the safe-abstention control without network or model loading",
    )
    parser.add_argument("--model", help="OpenAI model ID or OPENAI_MODEL override")
    parser.add_argument("--openai-base-url", help="HTTPS OpenAI Responses API base URL")
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
    parser.add_argument(
        "--provider-max-retries",
        type=_provider_retry_count,
        default=1,
        metavar="0..2",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=2_048)
    parser.add_argument("--minimum-confidence", type=_confidence, default=0.5)
    parser.add_argument("--diagnosis-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--generic-response-schema",
        action="store_true",
        help="disable input-specific ID/field constraints for a controlled baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing summary/prediction pair for the same condition",
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


def _provider_retry_count(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > 2:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
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


def _reference_modes(value: str) -> tuple[ReferenceMode, ...]:
    if value == "with":
        return (ReferenceMode.WITH_REFERENCES,)
    if value == "without":
        return (ReferenceMode.WITHOUT_REFERENCES,)
    return (ReferenceMode.WITH_REFERENCES, ReferenceMode.WITHOUT_REFERENCES)


def _load_cases(
    benchmark: str,
    path: Path | None,
) -> tuple[str, str, str, tuple[DiagnosisEvaluationCase, ...]]:
    if benchmark == "development":
        examples = load_development_benchmark(path)
        return (
            "development_v1",
            "not_human_validated",
            "1.0",
            development_evaluation_cases(examples),
        )
    if benchmark == "pilot-reviewed":
        reviewed_cases = load_reviewed_diagnosis_pilot(path)
        return (
            REVIEWED_PILOT_VERSION,
            "reviewed",
            "1.1",
            reviewed_pilot_evaluation_cases(reviewed_cases),
        )
    provisional_cases = load_diagnosis_pilot(path)
    return (
        "diagnosis_pilot_v1",
        "pending",
        "1.0",
        pilot_evaluation_cases(provisional_cases),
    )


def _validate_provider_options(namespace: argparse.Namespace) -> None:
    provider = cast(str, namespace.provider)
    has_openai_options = namespace.model is not None or namespace.openai_base_url is not None
    has_llama_path = namespace.model_path is not None
    if provider == "none" and (has_openai_options or has_llama_path):
        raise ValueError("provider-specific model options require a live --provider")
    if provider == "openai" and has_llama_path:
        raise ValueError("--model-path requires --provider llama-cpp")
    if provider == "llama-cpp" and has_openai_options:
        raise ValueError("--model and --openai-base-url require --provider openai")


def _make_client(
    namespace: argparse.Namespace,
    *,
    maximum_calls: int,
) -> tuple[ModelClient, dict[str, object]]:
    provider = cast(str, namespace.provider)
    if provider == "openai":
        openai_config = OpenAIResponsesConfig.from_env(
            model=cast(str | None, namespace.model),
            base_url=cast(str | None, namespace.openai_base_url),
            max_retries=cast(int, namespace.provider_max_retries),
        )
        return OpenAIResponsesClient(openai_config), openai_config.public_metadata()
    if provider == "llama-cpp":
        model_path = cast(Path | None, namespace.model_path)
        if model_path is None:
            raise ValueError("--model-path is required for --provider llama-cpp")
        llama_config = LlamaCppConfig(
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
        metadata = llama_config.public_metadata()
        try:
            metadata["runtime_version"] = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            metadata["runtime_version"] = "not_installed"
        return LlamaCppModelClient(llama_config), metadata
    return (
        FakeModelClient(
            structured_responses=(
                ModelOutputError("no live model provider is configured")
                for _ in range(maximum_calls)
            )
        ),
        {"provider": "none", "behavior": "safe_abstention_control"},
    )


def _run(namespace: argparse.Namespace) -> int:
    _validate_provider_options(namespace)
    benchmark_path = cast(Path | None, namespace.benchmark_path)
    benchmark_name, annotation_status, schema_version, cases = _load_cases(
        cast(str, namespace.benchmark),
        benchmark_path,
    )
    modes = _reference_modes(cast(str, namespace.reference_mode))
    attempts = cast(int, namespace.diagnosis_attempts)
    client, provider_metadata = _make_client(
        namespace,
        maximum_calls=len(cases) * len(modes) * attempts,
    )
    service_config = DiagnosisServiceConfig(
        timeout_seconds=cast(float, namespace.timeout_seconds),
        max_output_tokens=cast(int, namespace.max_output_tokens),
        minimum_confidence=cast(float, namespace.minimum_confidence),
        max_attempts=attempts,
        specialize_response_schema=not cast(bool, namespace.generic_response_schema),
    )
    service = DiagnosisService(client, service_config)
    output_directory = cast(Path, namespace.output_dir)
    outputs: list[dict[str, object]] = []
    reports: dict[ReferenceMode, DiagnosisExperimentReport] = {}
    for mode in modes:
        report = run_diagnosis_experiment(
            benchmark_name=benchmark_name,
            annotation_status=annotation_status,
            cases=cases,
            service=service,
            reference_mode=mode,
        )
        if isinstance(client, LlamaCppModelClient):
            provider_metadata["timeout_token_stopping_supported"] = (
                client.timeout_stopping_supported
            )
        metadata: dict[str, object] = {
            "benchmark": benchmark_name,
            "benchmark_version": benchmark_name,
            "diagnosis_schema_version": schema_version,
            "case_set_sha256": diagnosis_case_set_sha256(cases),
            "benchmark_source": (
                str(benchmark_path.resolve())
                if benchmark_path is not None
                else "committed_or_built_in"
            ),
            "prompt_version": DIAGNOSIS_PROMPT_VERSION,
            "reference_mode": mode.value,
            "diagnosis_service": {
                "timeout_seconds": service_config.timeout_seconds,
                "max_output_tokens": service_config.max_output_tokens,
                "minimum_confidence": service_config.minimum_confidence,
                "max_attempts": service_config.max_attempts,
                "temperature": 0.0,
                "specialize_response_schema": service_config.specialize_response_schema,
            },
            "provider": provider_metadata,
        }
        paths = write_diagnosis_artifacts(
            report,
            output_directory,
            run_metadata=metadata,
            overwrite=cast(bool, namespace.overwrite),
        )
        reports[mode] = report
        failure_path = output_directory / f"{benchmark_name}.{mode.value}.failures.json"
        if failure_path.exists() and not cast(bool, namespace.overwrite):
            raise FileExistsError(f"analysis artifact already exists: {failure_path}")
        write_analysis_report(
            analyze_diagnosis_failures(cases=cases, report=report),
            failure_path,
        )
        outputs.append(
            {
                "reference_mode": mode.value,
                "summary": str(paths.summary.resolve()),
                "predictions": str(paths.predictions.resolve()),
                "failure_analysis": str(failure_path.resolve()),
                "examples": report.examples,
                "overall_status_accuracy": report.metrics["overall_status_accuracy"].value,
                "abstention_rate": report.metrics["abstention_rate"].value,
            }
        )
    payload: dict[str, object] = {
        "benchmark": benchmark_name,
        "annotation_status": annotation_status,
        "outputs": outputs,
    }
    if set(reports) == {
        ReferenceMode.WITH_REFERENCES,
        ReferenceMode.WITHOUT_REFERENCES,
    }:
        ablation = compare_reference_ablation(
            cases=cases,
            with_references=reports[ReferenceMode.WITH_REFERENCES],
            without_references=reports[ReferenceMode.WITHOUT_REFERENCES],
        )
        ablation_path = output_directory / f"{benchmark_name}.reference_ablation.json"
        if ablation_path.exists() and not cast(bool, namespace.overwrite):
            raise FileExistsError(f"analysis artifact already exists: {ablation_path}")
        write_analysis_report(ablation, ablation_path)
        payload["reference_ablation"] = str(ablation_path.resolve())
    if annotation_status == "pending":
        payload["research_qualification"] = (
            "These results use provisional Codex-generated annotations and are not "
            "validated research results."
        )
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
