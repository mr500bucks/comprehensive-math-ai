# Stronger-model benchmark plan

## Ranked options

1. **Current host: `Qwen/Qwen3-4B-Instruct-2507`, Q4_K_M.** The official model
   is Apache-2.0, 4B parameters, non-thinking, and reports stronger mathematics
   and instruction-following than earlier small Qwen models. The pinned GGUF is
   2,497,280,736 bytes, so a CPU-only 8K-context run is realistic within the
   recorded 11.75 GiB RAM, though materially slower than the 1.5B baseline.
   Sources: [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
   [GGUF repository](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF).
2. **Higher-resource open model: `Qwen/Qwen3.5-27B`, Q4_K_M.** This is a
   quality-oriented option for a future machine with at least roughly 32 GiB
   system RAM or about 24 GiB VRAM. It is not suitable for the current host.
   Sources: [model card](https://huggingface.co/Qwen/Qwen3.5-27B),
   [GGUF repository](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF).
3. **Later paid API ceiling: `gpt-5.6-terra`.** Use only if paid API access is
   explicitly allowed later. It provides a provider-independent structured-
   output comparator; this plan does not create an API key. Source:
   [official documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

`Qwen2.5-Math-7B-Instruct` is not the first recommendation: its Q4_K_M weights
fit less comfortably and solver specialization does not guarantee the
structured classification, localization, and tutoring judgment measured here.
A llama.cpp grammar can constrain syntax, not semantic correctness.

## Pinned next diagnosis experiment

Do not run this command until the candidate file is present at the shown path
and its existing adapter verification passes. The 2026-08-25 preflight saw only
1.11 GiB RAM free despite 11.75 GiB installed, so close applications or reboot
and confirm at least 6 GiB available before loading the model. It runs the diagnosis-only
50-case comparison both without and with references:

```powershell
python scripts/run_diagnosis_experiment.py `
  --benchmark pilot-reviewed `
  --provider llama-cpp `
  --reference-mode both `
  --model-path .models\qwen3-4b-instruct-2507-q4-k-m\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
  --model-repository bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF `
  --model-revision ae44f08e1392f39c0e474af10c3ff8355c8b6688 `
  --model-filename Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
  --model-sha256 2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e `
  --model-size-bytes 2497280736 `
  --quantization Q4_K_M `
  --llama-context-window 8192 `
  --llama-threads 6 `
  --llama-batch-threads 12 `
  --llama-seed 1337 `
  --diagnosis-attempts 2 `
  --output-dir evaluation\results\diagnosis\qwen3-4b-instruct-2507-q4km-reviewed
```

Every result records provider, model repository/revision/file hash,
quantization, context/thread/seed settings, benchmark version, diagnosis schema
version, prompt version, and reference condition. Preserve the 1.5B results as
the historical infrastructure baseline.

Proceed to full tutoring only if the diagnosis run has enough non-abstaining,
schema-valid outputs to make localization, issue-category, completion-gap, and
dependency metrics meaningful. Do not spend hint calls after another
fundamentally unusable diagnosis result. Once that gate passes,
`scripts/run_tutoring_experiment.py --benchmark pilot-reviewed` accepts the same
pinned llama.cpp model metadata options.

## Fine-tuning gate

Fine-tuning is not justified until all of the following are true:

- the reviewed fixed benchmark and drift hashes are in place;
- at least one sufficiently capable stronger base model has been evaluated;
- dominant failure classes are stable across enough cases to act on;
- prompt/schema failures have been separated from model-capacity failures;
- task metrics identify a training objective beyond language-model loss;
- a separate train/validation design prevents tuning directly to these 50 cases.
