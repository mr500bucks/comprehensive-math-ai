# Diagnosis Pilot Annotation Guidelines

The diagnosis pilot contains synthetic cases and **Codex-proposed annotations**.
Every record begins with `human_review_status: pending`. Structural validation
does not establish mathematical truth. A qualified reviewer must approve,
modify, reject, or mark each case ambiguous before its labels can support a
research claim.

## Review workflow

1. Open `evaluation/review/diagnosis_pilot_v1_review.md` rather than raw JSONL.
2. Check the problem and student solution independently before consulting the
   proposed label or optional reference. References are non-exhaustive examples,
   never answer templates.
3. Validate status, earliest issue or completion gap, issue category, reusable
   prefix, tutor action, and reveal ceiling.
4. Replace `PENDING` with exactly one disposition: `APPROVE`, `MODIFY`, `REJECT`,
   or `AMBIGUOUS`. State every required correction briefly in reviewer notes.
5. Reconcile reviewed decisions into a separately versioned artifact. Do not
   silently edit the generated pilot JSONL or call pending labels ground truth.

The reconciled artifact is `diagnosis_pilot_v1_reviewed.jsonl`. It records
completed human mathematical review while retaining synthetic provenance.

## Valid inefficiency and downstream dependency

A valid but inefficient method is not a mathematical error. Use
`valid_but_inefficient`, include an `efficiency_note`, leave `first_issue` null,
and allow the full valid solution to remain reusable.

When a later step is locally coherent only because it uses an earlier wrong
result, use `dependent_on_previous_error` and name the causal predecessor in
`depends_on_step_ids`. Reserve `invalid` or `unsupported` for an independent
local defect. A later independent defect can also record that it depends on an
earlier root error.

## Core definitions

### First meaningful error

The earliest student-authored step whose mathematical operation, inference, or
claim prevents the reasoning from being accepted as written. Localize the cause,
not merely a later consequence. Ignore harmless style differences and an
irrelevant preamble unless that material changes the argument.

### Invalid versus unsupported

- **Invalid**: the written operation or inference is mathematically false under
  the stated conditions. Examples include illegal cancellation, a false algebraic
  identity, or applying a theorem where a known hypothesis fails.
- **Unsupported**: the claim might be true, but the written work does not justify
  it. Examples include inferring parallel lines from appearance or skipping the
  induction step. Do not classify a claim as invalid merely because its proof is
  absent.

### Incorrect versus incomplete

- **Incorrect**: at least one meaningful written step is invalid or unsupported.
- **Incomplete**: all written mathematical progress is usable, but a required
  conclusion, case, base condition, or final computation is missing. Do not invent
  an error just because the solution stops early.

If an attempt contains both a valid prefix and an actual error, label it
incorrect and localize the error. If omitted conditions make a purported final
answer unreliable, use incorrect rather than incomplete.

### Correct but inefficient

Every mathematical step and conclusion is correct, but the method includes
materially unnecessary work for the task and context. Inefficiency is not a
mathematical error. Do not use this label for a merely unfamiliar or elegant
alternative.

### Correct alternative proofs

Judge validity independently of the supplied reference. A correct proof may use
different lemmas, representations, order, or strategy. Reference mismatch alone
must never cause rejection. Mark `reference_relation=alternative_verified` only
after verifying the student's argument on its own terms.

### Ambiguous cases

Use `AMBIGUOUS` when missing diagrams, undefined notation, conflicting reasonable
interpretations, or genuinely disputed mathematical conventions prevent a stable
label. Explain what additional context or adjudication would resolve it. Do not
use ambiguity merely to avoid a difficult but well-posed judgment.

### Completion gap

For a valid incomplete attempt, identify what kind of step is still needed and
the last written step after which it belongs. A completion gap is not a fabricated
student step and must not be used alongside a first mathematical error.

### Reusable prefix

The longest initial sequence of student steps that can safely be retained when
continuing or repairing the solution. It ends immediately before the first
meaningful error. For a wholly correct or valid incomplete attempt, it normally
ends at the last written step. Leave it empty if the first step is already
unusable.

## Tutor-action labels

- `WAIT`: the response should not interrupt correct work or manufacture an issue.
- `ASK_STUDENT`: request the student's next step or missing context without
  supplying new mathematics.
- `VERIFY_STEP`: ask for clarification of an ambiguous step or notation.
- `LIGHT_HINT`: indicate that the localized step or transition needs checking,
  without disclosing the correction.
- `TARGETED_HINT`: identify the operation, condition, or concept to inspect.
- `STRONG_HINT` / `EXPLAIN_CONCEPT`: supply more direct conceptual scaffolding
  while withholding a full worked solution.
- `SHOW_PARTIAL_SOLUTION`: continue from the reusable prefix with limited worked
  mathematics.
- `SHOW_FULL_SOLUTION`: use only with explicit authorization.

The proposed pilot uses conservative first-turn ceilings: level 0 for waiting,
questions, and ambiguous verification; level 1 for a light hint on a localized
error. A reviewer may recommend a different action, but must never raise reveal
solely because a reference solution is available.

## Taxonomy boundary checks

- An arithmetic slip evaluates a valid numerical setup incorrectly.
- An algebraic error misapplies a symbolic rule or sign transformation.
- A logical gap omits an inference required for the conclusion.
- An unjustified claim has insufficient written evidence.
- A missing condition omits a domain restriction, sign case, hypothesis, or
  candidate check that changes validity.
- Theorem misuse applies a named result outside its hypotheses.
- Concept misunderstanding reflects an incorrect definition or relationship.

When two categories seem plausible, choose the code describing the earliest
cause and note the alternative in reviewer comments.

## Research-use rule

Pending proposals may be used only for developmental debugging, with this exact
qualification in any report:

> These results use provisional Codex-generated annotations and are not validated research results.
