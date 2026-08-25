# Roadmap

The repository now has a tested research vertical slice: validation, parsing,
structured diagnosis, deterministic tutoring policy, reveal-bounded hinting,
leakage checks, orchestration, a CLI, and deterministic evaluation.

Isolated OpenAI Responses and lazy local llama-cpp adapters now translate
provider failures, structured output, token usage, and latency into the neutral
model contract. Both remain opt-in. OpenAI was not exercised because the user
declined API-key setup. The pinned 1.5B Qwen Q4_K_M GGUF was verified and run
locally through llama-cpp; model weights are still never downloaded or loaded
by CI.

The local synthetic-development baseline showed that the 1.5B model is an
infrastructure baseline, not a viable diagnosis model: it mode-collapsed,
abstained frequently, and did not localize first issues. The highest-value next
phase is therefore human review of the prepared 50-case pilot followed by a
controlled comparison with a stronger base model. That work should:

1. collect blinded expert labels for correct alternative solutions, earliest
   issues, and appropriate tutor actions;
2. run the same provider-neutral evaluation interface on stronger external
   compute without arbitrary remote code or unsafe unpinned weight loading;
3. report uncertainty, false rejection of correct work, error localization,
   leakage, calls, tokens, and latency;
4. calibrate thresholds only after mathematical capability improves.

Later candidates, in order of evidence needed, are grounded mathematical
verification, multi-turn outcome capture, minimal learner-state tracking, and
human evaluation of hint usefulness. Fine-tuning is not justified by the
current evidence because the benchmark is pending review and the tested base
model is too weak to separate prompt, schema, and mathematical-capability
effects. Multiple candidates/ranking, curriculum alignment, a web interface,
authentication, and production analytics remain deferred until core diagnosis
is reliable.

## Reviewed-benchmark milestone (2026-08-25)

The 50-case review is reconciled in a separately versioned artifact; the
provisional pilot remains byte-for-byte unchanged. The contract now represents
valid inefficiency without a fake error and captures downstream dependencies
for future counterfactual repair analysis.

Next, run the reviewed diagnosis-only reference ablation with the pinned
Qwen3-4B-Instruct-2507 Q4_K_M candidate. Run full tutoring only if diagnosis
produces enough usable outputs and meaningful localization. Do not fine-tune
until the gate in `docs/STRONGER_MODEL_BENCHMARK.md` is met.
