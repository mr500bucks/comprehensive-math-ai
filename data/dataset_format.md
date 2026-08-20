# Canonical tutoring example format

The active storage contract is `TutoringExampleV1`, schema version `1.0`,
defined in `src/math_feedback_ai/domain/models.py`. Its generated JSON Schema is
committed at `schemas/tutoring_example.v1.schema.json`; an exact-schema drift
test keeps code and schema synchronized.

Each record contains:

- `example_id` and provenance, including a required `synthetic` flag;
- a problem with zero or more non-exhaustive reference solutions;
- the raw student attempt plus source-preserving parsed steps;
- a structured gold diagnosis with step assessments, an optional earliest
  issue, or a separate completion gap;
- an optional expected deterministic tutoring decision;
- zero or more ideal public responses.

Validation enforces cross-object step references, confidence bounds, diagnosis
status invariants, response reveal ceilings, and agreement between expected
decisions and ideal responses. Unknown fields are rejected. The source model,
not a hand-maintained example in this document, is authoritative.

Version `1.0` intentionally does not model learner profiles, multi-turn
outcomes, curriculum standards, or private provider traces. Add those through a
new version or a backward-compatible optional extension only after a concrete
evaluation need exists.
