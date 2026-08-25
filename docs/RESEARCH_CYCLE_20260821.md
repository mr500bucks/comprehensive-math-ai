# Local-model research cycle — 2026-08-21

## Scope and qualification

This cycle connected and actually ran a zero-API-cost local model, built a
provisional 50-case diagnosis pilot, ran the 40-case synthetic development
benchmark, performed a reference ablation, exercised the complete tutoring
pipeline, and converted a real leakage failure into a regression.

Development labels are synthetic and not evidence of efficacy. All pilot labels
remain Codex-proposed with `human_review_status: pending`; they are not
mathematical ground truth.

## Local provider

- Model family: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- GGUF repository: `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF`
- Revision: `951ed2aea09c43e331c612e74d83e4a23ca98e3b`
- Artifact: `Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf`
- Size: `986048832` bytes
- SHA-256: `9614a50f03c897028920ca0dc4365da570bf587f9ee7768261216fe370b37e8e`
- Runtime: `llama-cpp-python==0.3.35`, CPU only, context 4096, seed 1337,
  6 generation threads, 12 batch threads
- Observed process working set: approximately 1.83 GiB

The host has an Intel i5-10400, 11.75 GiB RAM, Intel UHD 630 integrated
graphics, and no NVIDIA/CUDA runtime. The 7B Q4_K_M weight file alone is about
4.68 GB, before runtime buffers, so it was not downloaded.

## Diagnosis results

The generic-schema control without references produced no usable diagnosis:
40/40 abstentions, 12.5% status accuracy (five gold indeterminate cases), 4.44%
macro-F1, and 0/23 exact or adjacent localization. All 40 calls had a structured
output failure: 25 failed domain validation and 15 returned no exposable output.

Two evidence-driven interoperability changes were then enabled:

1. constrain generated attempt/step identifiers and assessment count to values
   deterministic parsing already knows;
2. normalize only fields named exactly `confidence` from explicit percentages
   in `(1, 100]` to the `[0, 1]` contract, recording every normalization path.

| Metric | With references | Without references |
|---|---:|---:|
| Status accuracy | 10.0% | 12.5% |
| Macro-F1 | 6.67% | 8.56% |
| Incorrect accepted as correct/inefficient | 44.44% | 50.0% |
| Abstention / structured failure | 47.5% | 45.0% |
| Exact / adjacent first-issue localization | 0% / 0% | 0% / 0% |
| Issue-category accuracy | 0% | 0% |
| Completion-gap accuracy | 0% | 0% |
| Mean observed latency | 21.91 s | 20.16 s |
| P95 observed latency | 39.08 s | 37.95 s |

Every usable diagnosis collapsed to `correct_but_inefficient`; no usable output
localized an issue. The matched ablation found no strictly detected anchoring or
reference-helped case, and recommends diagnosis without references because it
was marginally better and faster. Observed token totals are lower bounds because
the native client cannot report tokens for calls that return no exposable
content.

## Complete tutoring pipeline

Using the selected no-reference diagnosis strategy on all 40 development cases:

- tutor-action agreement: 12/40 (30%);
- reveal compliance: 40/40;
- public-boundary leakage violations: 0/40;
- pipeline failures: 0/40;
- actual actions: 22 `WAIT`, 18 `ASK_STUDENT`;
- generated hint calls: 0.

Because weak diagnosis never requested a hint, a separate and explicitly
qualified oracle-diagnosis diagnostic exercised real local hint generation. It
made 36 text calls for 18 expected hint cases (one generation plus one bounded
rewrite each). A first run exposed one false negative: a level-1 response boxed
the final answer `56`. The checker now treats LaTeX `\boxed{...}` and
`\fbox{...}` as hard final-answer leakage below level 5. On the controlled
rerun, all 18 model hint cases failed closed to deterministic safe fallback;
public leakage and reveal violations remained 0/40.

## Consistency and failure analysis

A three-turn real-model probe produced one unjustified status flip and no
justified revision. The deterministic contract fixture still covers the desired
0/1 unjustified-flip and 1/1 justified-revision behavior, showing that trusted
session/policy logic is stable while model diagnosis is not.

Dominant automated failure categories were excessive abstention, first-error
localization failure, malformed structured output, mathematical reasoning
failure, overconfidence, and status mismatch. Parser segmentation was not a
material failure pattern, so no parser changes were made. Diagnosis prompt v1
was retained because the measured failures did not support a narrowly targeted
prompt edit.

## Decision

Do not fine-tune. First complete human mathematical review of the pilot, then
compare a stronger base model through the same provider-neutral evaluation
interface. The 1.5B Q4 model is useful for infrastructure and fail-closed safety
testing but is not a viable mathematical diagnosis or hint-generation baseline.

## Follow-up reconciliation — 2026-08-25

The independent mathematical review is now reconciled into
`evaluation/benchmarks/diagnosis_pilot_v1_reviewed.jsonl`; the provisional file
and historical 1.5B results remain unchanged. Diagnosis schema `1.1` separates
valid inefficiency from mathematical error and records downstream dependencies.
Prompt `diagnosis_v2` exists for future schema-compatible comparisons; no
further 1.5B prompt experiment was run.

The next capacity test is the reviewed diagnosis-only reference ablation with a
stronger base model. Full tutoring is gated on usable diagnosis, and
fine-tuning remains out of scope until a capable base model exposes stable,
task-level failure classes.
