"""Command-line entry point: cm4ai-evaluate."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .io import MODES, STRICT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cm4ai-evaluate",
        description="Score predicted protein images against ground truth images.",
    )
    parser.add_argument(
        "manifest",
        help="CSV or JSON manifest with sample_id, prediction, ground_truth and optional mask",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=STRICT,
        help="strict: challenge image specification (default); lenient: accept and record deviations",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="results",
        help="writes <prefix>.jsonl (one record per image) and <prefix>.csv (default: results)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("cm4ai-evaluate: scoring is not implemented yet.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
