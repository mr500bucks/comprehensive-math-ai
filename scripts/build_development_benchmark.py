"""Generate or verify the synthetic development benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from math_feedback_ai.evaluation.development_builder import (
    DEFAULT_OUTPUT_PATH,
    committed_benchmark_is_current,
    development_benchmark_sha256,
    write_development_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL destination (defaults to the committed development benchmark)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero instead of writing when the destination has drifted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        if committed_benchmark_is_current(args.output):
            print(f"benchmark is current: {args.output}")
            print(f"sha256: {development_benchmark_sha256()}")
            return 0
        print(f"benchmark drift detected: {args.output}")
        return 1

    destination = write_development_benchmark(args.output)
    print(f"wrote synthetic development benchmark: {destination}")
    print(f"sha256: {development_benchmark_sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
