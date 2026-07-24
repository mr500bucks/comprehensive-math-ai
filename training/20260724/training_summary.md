# Stage 2 Training Summary

Date: 2026-07-24

## Overview

This experiment represents the second fine-tuning stage of the Adaptive Math Tutor project.

The objective was to improve:
- first incorrect step detection
- progressive hint generation
- minimal answer leakage
- conversational tutoring behavior

## Changes From Previous Experiment

- Simplified SFT response format
- Removed evaluator metadata from training text
- Added clearer correct/incorrect behavior handling

## Base Model

Qwen/Qwen2.5-Math-7B-Instruct

## Training Method

QLoRA (4-bit)
PEFT LoRA fine-tuning
Supervised Fine-Tuning (SFT)

## Results

- Validation loss:
  - step 100:
  - step 200:

- Notes:
  Model learned structured tutoring responses. Further evaluation focuses on edge cases and correctness verification.
