"""Generate, validate, or drift-check the provisional diagnosis pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from math_feedback_ai.evaluation.diagnosis_pilot_builder import (
    DEFAULT_PILOT_PATH,
    DEFAULT_REVIEW_PATH,
    committed_diagnosis_pilot_is_current,
    diagnosis_pilot_review_sha256,
    diagnosis_pilot_sha256,
    validate_built_diagnosis_pilot,
    write_diagnosis_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PILOT_PATH)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if validation fails or either generated artifact has drifted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_built_diagnosis_pilot()
    if not report.passed:
        for issue in report.issues:
            print(f"{issue.code}: {', '.join(issue.case_ids)} — {issue.message}")
        return 1
    if args.check:
        if not committed_diagnosis_pilot_is_current(args.output, args.review_output):
            print("diagnosis pilot drift detected")
            return 1
        print(f"diagnosis pilot is current: {args.output}")
    else:
        paths = write_diagnosis_pilot(args.output, args.review_output)
        print(f"wrote provisional diagnosis pilot: {paths[0]}")
        print(f"wrote human review worksheet: {paths[1]}")
    print(f"cases: {report.case_count}")
    print(f"jsonl sha256: {diagnosis_pilot_sha256()}")
    print(f"review sha256: {diagnosis_pilot_review_sha256()}")
    print("human review status: pending (not mathematical ground truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
