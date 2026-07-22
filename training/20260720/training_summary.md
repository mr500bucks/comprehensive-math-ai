# Stage 1 Training Summary

**Date:** 2026-07-20

## Overview

This experiment represents the first successful fine-tuning stage of the Adaptive Math Tutor project.

The objective was to adapt Qwen2.5-Math-7B-Instruct into a tutoring-focused model capable of:

- Identifying the first incorrect step in a student's solution
- Explaining mathematical mistakes
- Providing progressive hints
- Avoiding premature answer disclosure
- Producing pedagogically useful feedback

---

## Base Model

Qwen/Qwen2.5-Math-7B-Instruct

---

## Training Method

- QLoRA (4-bit)
- PEFT LoRA fine-tuning
- Supervised Fine-Tuning (SFT)

---

## Dataset

Training:
- train.jsonl

Validation:
- validation.jsonl

---

## Results

Training completed successfully.

Runtime:
Approximately 2 hours.

Training epochs:
3

Validation loss:

- Step 100: 0.129901
- Step 200: 0.077361

The validation loss consistently decreased throughout training with no evidence of divergence.

---

## Next Steps

- Load the trained LoRA adapter
- Build an inference notebook
- Evaluate on the validation set
- Benchmark against held-out datasets
- Perform qualitative failure analysis
