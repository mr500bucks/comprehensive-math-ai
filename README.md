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
and no live provider adapter is included. The command-line interface therefore
fails closed to an `indeterminate` diagnosis unless a researcher supplies
scripted model output. This makes the default executable and safe for local
inspection without pretending that deterministic code can judge arbitrary
mathematics.

## What is implemented

- Strict, versioned Pydantic contracts for problems, attempts, diagnoses,
  issues, policy decisions, hint plans, leakage results, and responses.
- A compact versioned issue taxonomy and explicit reveal levels 0 through 5.
- A conservative deterministic parser that preserves the original text,
  character spans, order, stable content-derived step IDs, and segmentation
  ambiguity.
- A provider-neutral synchronous model interface plus a deterministic fake for
  tests. Provider output is always validated at the domain boundary.
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

Without scripted model output, the command deliberately abstains from judging
the mathematics and asks the student for more information. Supplying captured
diagnosis JSON and hint text is intended for provider-adapter development and
reproducible experiments; it is not a production trust boundary. The command
prints parsed steps, the validated diagnosis, policy decision, public response,
and leakage result as structured JSON.

## Contracts and benchmark

The canonical example contract is generated from `TutoringExampleV1` into
[`schemas/tutoring_example.v1.schema.json`](schemas/tutoring_example.v1.schema.json).
A minimal valid record is in
[`data/examples/tutoring_example.v1.json`](data/examples/tutoring_example.v1.json).
The development benchmark under `evaluation/benchmarks/` contains only
synthetic, explicitly marked fixtures. Use it to detect code regressions and
inspect metric behavior—not to estimate accuracy on real students.

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
- No API keys, provider credentials, remote-code loading, or model-weight
  deserialization are part of the active implementation.
- Raw/private data and local model artifacts are ignored by Git. Commit only
  redacted fixtures and manifests.
- Leakage checks are useful guardrails, not formal guarantees. A production
  system would need provider-specific controls, privacy review, monitoring, and
  human evaluation before handling student data.

## Deliberately deferred

The next phases are not currently implemented: a real provider adapter,
grounded or symbolic verification, human-validated benchmark data, learner
profiles, curriculum alignment, longitudinal outcomes, multiple hint
candidates/ranking, fine-tuning, a web UI, authentication, and production
storage/observability. See [`docs/roadmap.md`](docs/roadmap.md) for the scoped
next steps.
