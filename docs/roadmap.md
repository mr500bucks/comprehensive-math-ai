# Roadmap

The repository now has a tested research vertical slice: validation, parsing,
structured diagnosis, deterministic tutoring policy, reveal-bounded hinting,
leakage checks, orchestration, a CLI, and deterministic evaluation.

The highest-value next phase is a real provider adapter plus a small,
human-reviewed diagnostic benchmark. That work should:

1. translate provider timeouts, transport failures, structured output, token
   usage, and latency into the existing neutral client contract;
2. run without arbitrary remote code or unsafe weight deserialization;
3. collect blinded expert labels for correct alternative solutions, earliest
   issues, and appropriate tutor actions;
4. report uncertainty, false rejection of correct work, error localization,
   leakage, calls, tokens, and latency;
5. calibrate thresholds before adding training or product surfaces.

Later candidates, in order of evidence needed, are grounded mathematical
verification, multi-turn outcome capture, minimal learner-state tracking, and
human evaluation of hint usefulness. Multiple candidates/ranking, fine-tuning,
curriculum alignment, a web interface, authentication, and production analytics
remain deferred until the core diagnosis is shown to be reliable.
