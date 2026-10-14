"""Command-line entry point: cm4ai-evaluate."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from . import __version__, scoring
from .io import MODES, STRICT
from .manifest import ManifestError, read_manifest
from .outputs import write_csv, write_jsonl
from .runner import ERROR, score_row

INSTALL_HINT = 'pip install "cm4ai-evaluate[{extra}]" (or pip install -e ".[{extra}]" in the repository folder)'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cm4ai-evaluate",
        description="Score predicted protein images against ground truth images.",
    )
    parser.add_argument(
        "manifest",
        help="CSV or JSON manifest: prediction, ground_truth, optional prediction_id and mask",
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
    parser.add_argument(
        "--inception",
        action="store_true",
        help="also compute Inception L2 (needs the [inception] extra; downloads about 100 MB "
        "of model weights on first use)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="device for Inception (default: cuda, then mps, then cpu, whichever is available)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _say(message: str = "") -> None:
    print(message, file=sys.stderr)


def _skipped_metrics(include_inception: bool) -> dict[str, list[str]]:
    """Metrics left out because their extra is not installed, by extra."""
    skipped: dict[str, list[str]] = {}
    for metric in scoring.METRICS:
        if metric.requires == scoring.INCEPTION and not include_inception:
            continue
        if not scoring.extra_installed(metric.requires):
            skipped.setdefault(metric.requires, []).extend(metric.outputs)
    return skipped


def _summary(records: list[dict]) -> None:
    errors = [r for r in records if r["status"] == ERROR]
    scored = len(records) - len(errors)
    _say(f"Scored {scored} of {len(records)} rows.")
    if errors:
        _say(f"{len(errors)} row(s) not scored:")
        for r in errors[:10]:
            _say(f"  {r['prediction_id']}: {r['error']['message']}")
        if len(errors) > 10:
            _say(f"  ... and {len(errors) - 10} more (see the error column)")
    non_compliant = sum(1 for r in records if r["status"] != ERROR and not r["spec_compliant"])
    if non_compliant:
        _say(f"{non_compliant} scored row(s) are not spec compliant.")
    warnings = Counter(
        f"{d['role']} {d['code']}: {d['message']}" for r in records for d in r["deviations"] if d["warning"]
    )
    for text, count in warnings.most_common():
        _say(f"  warning ({count} row(s)) {text}")
    skips = Counter((r["mask_skip_reason"] or "").split(":")[0] for r in records if r["mask_skip_reason"])
    for code, count in skips.most_common():
        _say(f"  mask skipped ({count} row(s)): {code}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = read_manifest(args.manifest)
    except ManifestError as error:
        _say(f"cm4ai-evaluate: {error}")
        return 2
    if manifest.ignored_columns:
        _say(f"Ignoring manifest column(s): {', '.join(manifest.ignored_columns)}")

    if args.inception and not scoring.extra_installed(scoring.INCEPTION):
        _say("cm4ai-evaluate: --inception needs torch and torchvision.")
        _say("  Install with: " + INSTALL_HINT.format(extra="inception"))
        return 2
    for extra, names in _skipped_metrics(args.inception).items():
        _say(f"Skipping {', '.join(names)}: the [{extra}] packages are not installed.")
        _say("  Install with: " + INSTALL_HINT.format(extra=extra))
    metrics = scoring.available_metrics(include_inception=args.inception)

    extractor = None
    if args.inception:
        from .extended import InceptionFeatureExtractor

        _say("Loading Inception (downloads model weights on first use)...")
        extractor = InceptionFeatureExtractor(device=args.device)
        _say(f"  on device {extractor.device}")

    rows = manifest.rows
    _say(f"Scoring {len(rows)} row(s) in {args.mode} mode.")
    step = max(1, len(rows) // 10)
    records = []
    for i, row in enumerate(rows, start=1):
        records.append(score_row(row, args.mode, metrics, extractor))
        if i % step == 0 or i == len(rows):
            _say(f"  {i}/{len(rows)}")

    prefix = Path(args.output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = prefix.with_name(prefix.name + ".jsonl")
    csv_path = prefix.with_name(prefix.name + ".csv")
    write_jsonl(records, jsonl_path)
    write_csv(records, scoring.output_names(metrics), csv_path)

    _summary(records)
    _say(f"Wrote {jsonl_path} and {csv_path}")
    return 1 if any(r["status"] == ERROR for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
