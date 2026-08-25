"""Generate, validate, or drift-check the reviewed diagnosis pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from math_feedback_ai.evaluation.reviewed_diagnosis_pilot import (
    DEFAULT_RECONCILIATION_PATH,
    DEFAULT_REVIEWED_PILOT_PATH,
    committed_reviewed_diagnosis_pilot_is_current,
    reviewed_diagnosis_pilot_sha256,
    reviewed_reconciliation_sha256,
    validate_built_reviewed_diagnosis_pilot,
    write_reviewed_diagnosis_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REVIEWED_PILOT_PATH)
    parser.add_argument("--reconciliation-output", type=Path, default=DEFAULT_RECONCILIATION_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if validation fails or generated artifacts drifted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_built_reviewed_diagnosis_pilot()
    if not report.passed:
        for issue in report.issues:
            print(f"{issue.code}: {', '.join(issue.case_ids)} — {issue.message}")
        return 1
    if args.check:
        if not committed_reviewed_diagnosis_pilot_is_current(
            args.output, args.reconciliation_output
        ):
            print("reviewed diagnosis pilot drift detected")
            return 1
        print(f"reviewed diagnosis pilot is current: {args.output}")
    else:
        paths = write_reviewed_diagnosis_pilot(args.output, args.reconciliation_output)
        print(f"wrote reviewed diagnosis pilot: {paths[0]}")
        print(f"wrote reconciliation: {paths[1]}")
    print(f"cases: {report.case_count}")
    print(f"jsonl sha256: {reviewed_diagnosis_pilot_sha256()}")
    print(f"reconciliation sha256: {reviewed_reconciliation_sha256()}")
    print("human review status: reviewed; synthetic provenance retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
