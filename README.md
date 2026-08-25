# Math Feedback AI

Math Feedback AI is a small research implementation for testing one tutoring
pipeline:

```text
problem + student work
  -> source-preserving step parser
  -> structured mathematical diagnosis
  -> deterministic intervention policy
  -> reveal-constrained hint generation
  -> leakage checks
  -> public tutor response
```

This repository does **not** contain a generally reliable autonomous math
tutor. Mathematical diagnosis and hint wording still require a model provider,
and the quality of those judgments must be measured rather than assumed.
Isolated OpenAI Responses and local llama-cpp adapters are available, but the
command-line interface defaults to no provider and fails closed to an
`indeterminate` diagnosis unless a researcher explicitly selects an adapter or
supplies scripted output.

## What is implemented

- Strict, versioned Pydantic contracts for problems, attempts, diagnoses,
  issues, policy decisions, hint plans, leakage results, and responses.
- A compact versioned issue taxonomy and explicit reveal levels 0 through 5.
- A conservative deterministic parser that preserves the original text,
  character spans, order, stable content-derived step IDs, and segmentation
  ambiguity.
- A provider-neutral synchronous model interface, isolated OpenAI Responses
  and local llama-cpp adapters, and a deterministic fake for tests. Provider
  output is always validated at the domain boundary.
- A diagnosis-only prompt and service with one bounded repair attempt and an
  `indeterminate` fallback for timeouts, invalid output, or low confidence.
- A transparent tutoring policy. Correct work receives no invented hint;
  uncertain work is verified or clarified; stronger disclosures require
  trusted escalation; hint-only mode cannot be overridden by student text.
- Single-candidate hint generation with reveal-minimized context, interpretable
  leakage checks, at most one constrained regeneration, and a deterministic
  safe fallback.
- A thin orchestrator that keeps the full private `TutorResult` separate from
  the smaller student-facing `TutorResponse`.
- Deterministic evaluation metrics and a clearly labelled synthetic development
  benchmark. It is a regression fixture, not evidence of tutoring efficacy.
- Unit, integration, regression, and adversarial tests; strict type checking,
  linting, formatting checks, and CI dependency auditing.

## Setup

The package metadata permits Python 3.12 through 3.14; Python 3.12 is the
tested CI baseline.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
```

On POSIX systems, use `.venv/bin/python` instead. The lock file pins the
complete development and runtime environment. The looser ranges in
`pyproject.toml` describe package compatibility, not a reproducible test
environment.

Run the quality gates with:

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```

## Research CLI

The `math-feedback-ai tutor` command accepts a problem, student work, and an
optional non-exhaustive reference solution. Run its help for the exact research
options:

```powershell
math-feedback-ai tutor --help
```

Without an explicitly selected provider or scripted output, the command
deliberately abstains from judging the mathematics and asks the student for
more information. Supplying captured diagnosis JSON and hint text is intended
for reproducible experiments; it is not a production trust boundary. The
command prints parsed steps, the validated diagnosis, policy decision, public
response, leakage result, and logical model-call counts as structured JSON.

To use OpenAI, select an available model explicitly and provide the API key
through the normal environment rather than a command-line argument:

```powershell
$env:OPENAI_API_KEY = "<secret>"
$env:OPENAI_MODEL = "<supported-model-id>"
math-feedback-ai tutor --provider openai --problem "..." --solution "..."
```

The adapter uses `POST /v1/responses`, converts the application schema to the
strict Structured Outputs subset for diagnosis, uses temperature zero, sends
`store: false`, records reported token usage and elapsed latency, and
translates timeout, transport, refusal, and malformed-output failures into the
provider-neutral error types. Transient request retries are bounded and
configurable with `--provider-max-retries`. The adapter has no additional
Python dependency. `OPENAI_BASE_URL` or
`--openai-base-url` is an advanced override and must name a trusted HTTPS
endpoint because the configured API key is sent to it.

For a small CPU-only local baseline, install the optional native runtime and
place this exact, manually obtained artifact under an ignored local directory:

- repository: `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF`
- revision: `951ed2aea09c43e331c612e74d83e4a23ca98e3b`
- file: `Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf`
- size: `986048832` bytes
- SHA-256: `9614a50f03c897028920ca0dc4365da570bf587f9ee7768261216fe370b37e8e`

```powershell
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install -r requirements-local-cpu.lock `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv\Scripts\python -m pip install -e . --no-deps
math-feedback-ai tutor --provider llama-cpp `
  --model-path .models\Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf `
  --timeout-seconds 300 --problem "..." --solution "..."
```

The local adapter imports the native runtime and memory-maps weights only on
the first request, after verifying the exact byte count and SHA-256. It uses a
fixed seed, CPU-only loading, configurable context/thread counts, llama.cpp's
JSON-Schema grammar for diagnosis, and best-effort per-token timeout stopping.
Token-level stopping is used only when the installed chat API exposes that
capability; llama-cpp-python 0.3.35 does not, so that version can enforce the
deadline only before and after its native chat call. Native model loading and a
single native generation operation cannot be safely interrupted mid-call, so
the timeout is not a hard process deadline. The repository never downloads a
model automatically, and CI uses injected test backends rather than native
weights.

## Contracts and benchmark

The canonical example contract is generated from `TutoringExampleV1` into
[`schemas/tutoring_example.v1.schema.json`](schemas/tutoring_example.v1.schema.json).
A minimal valid record is in
[`data/examples/tutoring_example.v1.json`](data/examples/tutoring_example.v1.json).
Diagnosis schema `1.1` adds two causal step states while continuing to load
historical `1.0` fixtures: `valid_but_inefficient` uses an `efficiency_note`
without creating `first_issue`, and `dependent_on_previous_error` records its
causal predecessors in `depends_on_step_ids`.

The development benchmark under `evaluation/benchmarks/` contains only
synthetic, explicitly marked fixtures. Use it to detect code regressions and
inspect metric behavior—not to estimate accuracy on real students.

Run a diagnosis-only reference ablation with a selected provider using the
standalone experiment command. It writes a compact summary JSON and a
self-contained, one-case-per-line prediction JSONL for each condition under the
ignored `evaluation/results/` directory:

```powershell
.venv\Scripts\python scripts\run_diagnosis_experiment.py `
  --benchmark development --provider openai --reference-mode both
```

Use `--benchmark pilot` for the immutable 50-case provisional pilot; its labels
retain the pending-review qualification. Use `--benchmark pilot-reviewed` for
the separately versioned human-reviewed synthetic benchmark. Human review
improves label reliability, not ecological validity. Drift-check it with
`python scripts/build_reviewed_diagnosis_pilot.py --check`.

Run the experiment script with `--help` for local llama-cpp settings,
timeouts, confidence thresholds, explicit output paths, and overwrite control.
The actual 2026-08-21 local baseline, ablation, failure analysis, tutoring run,
and resulting research decision are summarized in
[`docs/RESEARCH_CYCLE_20260821.md`](docs/RESEARCH_CYCLE_20260821.md).

Reference solutions are plural and non-exhaustive throughout the contracts and
prompts. A method difference is never sufficient evidence of an error. The
model remains responsible for the underlying mathematical judgment; the
current code validates consistency, but it is not a proof checker.

## Historical experiments

The dated folders under `training/` are preserved records from July 2026, not
active environments. They contain configuration summaries, dependency lists,
aggregate losses, and narrative summaries. The claimed dataset and splits,
formatting/training code, prompt used for training, checkpoints/adapters, raw
logs, and generated evaluations were not found. Both runs are therefore marked
**legacy / partially unreproducible** in
[`training/legacy_artifact_status.json`](training/legacy_artifact_status.json).
No historical measurement was edited or backfilled.

## Security and data handling

- Student, problem, and reference text is untrusted input and is delimited or
  JSON-quoted in prompts.
- API keys are read only from the process environment and are never accepted as
  command-line flags, serialized into results, or committed by the project.
- The OpenAI adapter performs no network request during import or construction;
  network access begins only after explicit `--provider openai` selection.
- The local adapter executes no remote model code and refuses unverified,
  renamed, or symlinked GGUF files before native loading.
- Raw/private data and local model artifacts are ignored by Git. Commit only
  redacted fixtures and manifests.
- Leakage checks are useful guardrails, not formal guarantees. A production
  system would need provider-specific controls, privacy review, monitoring, and
  human evaluation before handling student data.

## Deliberately deferred

The next phases are not currently implemented: grounded or symbolic
verification, human-validated benchmark data, learner profiles, curriculum
alignment, longitudinal outcomes, multiple hint candidates/ranking,
fine-tuning, a web UI, authentication, and production storage/observability.
See [`docs/roadmap.md`](docs/roadmap.md) for the scoped next steps.
