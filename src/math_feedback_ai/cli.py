"""Research CLI for exercising the complete tutoring vertical slice."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from math_feedback_ai.diagnosis.service import DiagnosisService, DiagnosisServiceConfig
from math_feedback_ai.domain.models import Problem, ReferenceSolution, TutorResult
from math_feedback_ai.domain.taxonomy import RevealLevel, TutorAction, TutorMode
from math_feedback_ai.evaluation.benchmark import (
    DEFAULT_BENCHMARK_PATH,
    evaluate_results,
    load_development_benchmark,
)
from math_feedback_ai.hints.generator import HintGenerator
from math_feedback_ai.model.client import ModelOutputError
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
from math_feedback_ai.orchestration.tutor import TutorContext, TutorOrchestrator

type CliModelClient = FakeModelClient | OpenAIResponsesClient | LlamaCppModelClient


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without performing any model or file access."""

    parser = argparse.ArgumentParser(
        prog="math-feedback-ai",
        description=(
            "Run the research tutoring pipeline. It safely abstains by default; OpenAI or "
            "a verified local llama-cpp model can be selected explicitly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    tutor = commands.add_parser(
        "tutor",
        help="parse and diagnose one student solution",
        description=(
            "Parse a student attempt, validate structured diagnosis, apply deterministic "
            "policy, generate a reveal-bounded hint, and run leakage checks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tutor.add_argument("--problem", required=True, help="math problem statement")
    tutor.add_argument("--solution", required=True, help="student solution text")
    tutor.add_argument(
        "--provider",
        choices=("none", "openai", "llama-cpp"),
        default="none",
        help="model provider; none preserves provider-free safe abstention",
    )
    tutor.add_argument(
        "--model",
        help="provider model ID; for OpenAI, required here or through OPENAI_MODEL",
    )
    tutor.add_argument(
        "--openai-base-url",
        help="advanced: HTTPS OpenAI API base URL; defaults to OPENAI_BASE_URL or official API",
    )
    tutor.add_argument(
        "--model-path",
        type=Path,
        help="local GGUF path; required for --provider llama-cpp",
    )
    tutor.add_argument(
        "--model-sha256",
        default=DEFAULT_QWEN_SHA256,
        help="expected lowercase SHA-256 for the local GGUF artifact",
    )
    tutor.add_argument(
        "--model-size-bytes",
        type=_positive_int,
        default=DEFAULT_QWEN_SIZE_BYTES,
        help="expected exact local GGUF size",
    )
    tutor.add_argument(
        "--llama-context-window",
        type=_positive_int,
        default=4_096,
        help="local llama-cpp token context size",
    )
    tutor.add_argument(
        "--llama-threads",
        type=_positive_int,
        default=6,
        help="CPU threads used for local token generation",
    )
    tutor.add_argument(
        "--llama-batch-threads",
        type=_positive_int,
        default=12,
        help="CPU threads used for local prompt/batch evaluation",
    )
    tutor.add_argument(
        "--llama-seed",
        type=_nonnegative_int,
        default=1_337,
        help="deterministic local generation seed",
    )
    tutor.add_argument(
        "--provider-max-retries",
        type=_provider_retry_count,
        default=1,
        metavar="0..2",
        help="bounded retries for transient provider transport/timeout failures",
    )
    tutor.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="timeout applied to each diagnosis or hint generation request",
    )
    tutor.add_argument("--problem-id", default="cli-problem", help="stable problem ID")
    tutor.add_argument("--attempt-id", help="stable attempt ID; derived when omitted")
    tutor.add_argument(
        "--reference",
        action="append",
        default=[],
        help="optional non-exhaustive reference solution; may be repeated",
    )
    tutor.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in TutorMode),
        default=TutorMode.HINT_ONLY.value,
        help="trusted tutoring mode",
    )
    tutor.add_argument(
        "--prior-attempt-count",
        type=_nonnegative_int,
        default=0,
        help="number of earlier attempts in the trusted session context",
    )
    tutor.add_argument(
        "--prior-action",
        choices=tuple(action.value for action in TutorAction),
        help="last tutoring action in the trusted session context",
    )
    tutor.add_argument(
        "--current-reveal-level",
        type=_reveal_level,
        default=RevealLevel.NONE,
        metavar="0..5",
        help="highest reveal level already used",
    )
    tutor.add_argument(
        "--requested-reveal-level",
        type=_reveal_level,
        metavar="0..5",
        help="student preference; can lower but never raise the policy ceiling",
    )
    tutor.add_argument(
        "--authorize-full-solution",
        action="store_true",
        help="trusted authorization; effective only in full_solution_allowed mode",
    )
    tutor.add_argument(
        "--final-answer",
        action="append",
        default=[],
        help="known final answer to protect from leakage; may be repeated",
    )
    tutor.add_argument(
        "--diagnosis-json",
        action="append",
        default=[],
        metavar="JSON_OR_@FILE",
        help=(
            "scripted structured model output; prefix a path with @ to read it. "
            "May be repeated once to exercise bounded repair."
        ),
    )
    tutor.add_argument(
        "--hint-text",
        action="append",
        default=[],
        metavar="TEXT_OR_@FILE",
        help=(
            "scripted hint-model output; prefix a path with @ to read it. "
            "May be repeated once to exercise safety regeneration."
        ),
    )
    benchmark = commands.add_parser(
        "benchmark",
        help="inspect the committed synthetic development benchmark",
        description=(
            "Validate and summarize the synthetic development fixture. Optionally run a "
            "provider-free safe-abstention baseline; this is not an efficacy result."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    benchmark.add_argument(
        "--path",
        type=Path,
        help="benchmark JSONL path; omit for the repository or built-in development fixture",
    )
    benchmark.add_argument(
        "--run-safe-baseline",
        action="store_true",
        help="score the deterministic no-provider abstention behavior",
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


def _reveal_level(value: str) -> RevealLevel:
    try:
        return RevealLevel(int(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a reveal level from 0 to 5") from exc


def _provider_retry_count(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > 2:
        raise argparse.ArgumentTypeError("must be a provider retry count from 0 to 2")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _read_scripted_text(value: str) -> str:
    if value.startswith("@"):
        raw_path = value[1:]
        if not raw_path:
            raise ValueError("@ must be followed by a scripted-output file path")
        text = Path(raw_path).read_text(encoding="utf-8")
    else:
        text = value
    if not text.strip():
        raise ValueError("scripted model output must not be blank")
    return text


def _run_tutor(namespace: argparse.Namespace) -> int:
    diagnosis_values = cast(list[str], namespace.diagnosis_json)
    hint_values = cast(list[str], namespace.hint_text)
    if len(diagnosis_values) > 2:
        raise ValueError("--diagnosis-json may be supplied at most twice")
    if len(hint_values) > 2:
        raise ValueError("--hint-text may be supplied at most twice")
    if hint_values and not diagnosis_values:
        raise ValueError("--hint-text requires --diagnosis-json")

    provider = cast(str, namespace.provider)
    if provider != "none" and (diagnosis_values or hint_values):
        raise ValueError("scripted model outputs cannot be combined with a live provider")
    if provider == "none" and (
        namespace.model is not None
        or namespace.openai_base_url is not None
        or namespace.model_path is not None
    ):
        raise ValueError("provider-specific model options require a live --provider")
    if provider == "openai" and namespace.model_path is not None:
        raise ValueError("--model-path requires --provider llama-cpp")
    if provider == "llama-cpp" and (
        namespace.model is not None or namespace.openai_base_url is not None
    ):
        raise ValueError("--model and --openai-base-url require --provider openai")

    diagnosis_scripts = tuple(_read_scripted_text(value) for value in diagnosis_values)
    hint_scripts = tuple(_read_scripted_text(value) for value in hint_values)
    if provider == "openai":
        client: CliModelClient = OpenAIResponsesClient(
            OpenAIResponsesConfig.from_env(
                model=cast(str | None, namespace.model),
                base_url=cast(str | None, namespace.openai_base_url),
                max_retries=cast(int, namespace.provider_max_retries),
            )
        )
        execution_mode = "live_openai"
        diagnosis_attempts = 2
    elif provider == "llama-cpp":
        model_path = cast(Path | None, namespace.model_path)
        if model_path is None:
            raise ValueError("--model-path is required for --provider llama-cpp")
        client = LlamaCppModelClient(
            LlamaCppConfig(
                model_path=model_path,
                model_repository=DEFAULT_QWEN_REPOSITORY,
                model_revision=DEFAULT_QWEN_REVISION,
                model_filename=DEFAULT_QWEN_FILENAME,
                expected_sha256=cast(str, namespace.model_sha256),
                expected_size_bytes=cast(int, namespace.model_size_bytes),
                n_ctx=cast(int, namespace.llama_context_window),
                n_threads=cast(int, namespace.llama_threads),
                n_threads_batch=cast(int, namespace.llama_batch_threads),
                seed=cast(int, namespace.llama_seed),
            )
        )
        execution_mode = "local_llama_cpp"
        diagnosis_attempts = 2
    else:
        structured_responses = (
            diagnosis_scripts
            if diagnosis_scripts
            else (ModelOutputError("no live model provider is configured"),)
        )
        client = FakeModelClient(
            structured_responses=structured_responses,
            text_responses=hint_scripts,
        )
        execution_mode = "scripted_model_outputs" if diagnosis_scripts else "safe_abstention"
        diagnosis_attempts = 2 if len(diagnosis_scripts) == 2 else 1

    timeout_seconds = cast(float, namespace.timeout_seconds)
    diagnosis_config = DiagnosisServiceConfig(
        max_attempts=diagnosis_attempts,
        timeout_seconds=timeout_seconds,
    )
    orchestrator = TutorOrchestrator(
        diagnosis_service=DiagnosisService(client, diagnosis_config),
        hint_generator=HintGenerator(client, timeout_seconds=timeout_seconds),
    )

    reference_texts = cast(list[str], namespace.reference)
    references = tuple(
        ReferenceSolution(reference_id=f"reference-{index}", text=text)
        for index, text in enumerate(reference_texts, start=1)
    )
    problem = Problem(
        problem_id=cast(str, namespace.problem_id),
        statement=cast(str, namespace.problem),
        reference_solutions=references,
    )
    prior_action_value = cast(str | None, namespace.prior_action)
    context = TutorContext(
        prior_attempt_count=cast(int, namespace.prior_attempt_count),
        prior_action=TutorAction(prior_action_value) if prior_action_value else None,
        current_reveal_level=cast(RevealLevel, namespace.current_reveal_level),
        mode=TutorMode(cast(str, namespace.mode)),
        full_solution_authorized=cast(bool, namespace.authorize_full_solution),
        requested_reveal_level=cast(RevealLevel | None, namespace.requested_reveal_level),
        final_answers=tuple(cast(list[str], namespace.final_answer)),
    )
    result = orchestrator.tutor(
        problem,
        cast(str, namespace.solution),
        context=context,
        attempt_id=cast(str | None, namespace.attempt_id),
    )
    payload = _result_payload(
        result,
        client=client,
        execution_mode=execution_mode,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _result_payload(
    result: TutorResult,
    *,
    client: CliModelClient,
    execution_mode: str,
) -> dict[str, object]:
    leakage = (
        result.leakage_check.model_dump(mode="json") if result.leakage_check is not None else None
    )
    hint = result.response.message if result.response.reveal_level > RevealLevel.NONE else None
    return {
        "execution_mode": execution_mode,
        "parse": {
            "attempt_id": result.attempt.attempt_id,
            "strategy": result.attempt.parse_strategy.value,
            "segmentation_ambiguous": result.attempt.segmentation_ambiguous,
            "notes": list(result.attempt.parse_notes),
        },
        "parsed_steps": [step.model_dump(mode="json") for step in result.attempt.steps],
        "diagnosis": result.diagnosis.model_dump(mode="json"),
        "decision": result.decision.model_dump(mode="json"),
        "public_response": result.response.model_dump(mode="json"),
        "hint": hint,
        "leakage_check": leakage,
        "model_calls": {
            "structured": client.structured_call_count,
            "text": client.text_call_count,
        },
    }


def _run_benchmark(namespace: argparse.Namespace) -> int:
    path = cast(Path | None, namespace.path)
    examples = load_development_benchmark(path)
    status_counts = Counter(example.gold_diagnosis.overall_status.value for example in examples)
    action_counts = Counter(
        example.expected_decision.action.value
        for example in examples
        if example.expected_decision is not None
    )
    synthetic_examples = sum(example.provenance.synthetic for example in examples)
    all_synthetic = synthetic_examples == len(examples)
    payload: dict[str, object] = {
        "artifact_kind": (
            "synthetic_development_fixture" if all_synthetic else "benchmark_fixture"
        ),
        "path": (
            str(path.resolve())
            if path is not None
            else (
                str(DEFAULT_BENCHMARK_PATH.resolve())
                if DEFAULT_BENCHMARK_PATH.is_file()
                else "built-in:development_v1"
            )
        ),
        "examples": len(examples),
        "synthetic_examples": synthetic_examples,
        "human_validation_status": (
            "not_human_validated" if all_synthetic else "not_recorded_by_schema"
        ),
        "gold_status_counts": dict(sorted(status_counts.items())),
        "expected_action_counts": dict(sorted(action_counts.items())),
    }

    if cast(bool, namespace.run_safe_baseline):
        client = FakeModelClient(
            structured_responses=tuple(
                ModelOutputError("no live model provider is configured") for _ in examples
            )
        )
        orchestrator = TutorOrchestrator(
            diagnosis_service=DiagnosisService(
                client,
                DiagnosisServiceConfig(max_attempts=1),
            ),
            hint_generator=HintGenerator(client),
        )
        results = {
            example.example_id: orchestrator.tutor(
                example.problem,
                example.student_attempt.raw_text,
                attempt_id=example.student_attempt.attempt_id,
            )
            for example in examples
        }
        report = evaluate_results(examples, results)
        payload["safe_abstention_baseline"] = {
            "description": (
                "No-provider fallback only; this does not measure mathematical tutoring efficacy."
            ),
            "metrics": report.as_dict(),
            "model_calls": {
                "structured": client.structured_call_count,
                "text": client.text_call_count,
            },
        }

    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a conventional process exit status."""

    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        if namespace.command == "tutor":
            return _run_tutor(namespace)
        if namespace.command == "benchmark":
            return _run_benchmark(namespace)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {namespace.command}")


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point
    raise SystemExit(main())
