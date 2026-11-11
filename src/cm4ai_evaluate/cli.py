"""Command-line entry point: cm4ai-evaluate."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from . import __version__, panel, scoring
from .io import MODES, STRICT
from .manifest import ManifestError, read_manifest
from .outputs import cell_summary, group_summary, write_csv, write_jsonl, write_table
from .runner import ERROR, score_row
from .selection import SelectionError, resolve

INSTALL_HINT = 'pip install "cm4ai-evaluate[{extra}]" (or pip install -e ".[{extra}]" in the repository folder)'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cm4ai-evaluate",
        description="Score predicted protein images against ground truth images.",
    )
    parser.add_argument(
        "manifest",
        help="CSV or JSON manifest: prediction, ground_truth; optional prediction_id, mask, "
        "cell_id, group_id, nucleus, microtubules, er",
    )
    parser.add_argument(
        "--metrics",
        help="comma-separated tiers (basic, extended, subcell, inception, all) and metric names "
        "(default: basic,extended)",
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
        help="writes <prefix>.jsonl and <prefix>.csv (one record per prediction), <prefix>_cells.csv, "
        "and <prefix>_groups.csv when the manifest has group_id (default: results)",
    )
    parser.add_argument(
        "--inception",
        action="store_true",
        help="also compute Inception L2 (same as adding inception to --metrics; downloads about "
        "100 MB of model weights on first use)",
    )
    parser.add_argument(
        "--subcell-weights",
        help="SubCell weights file (default: downloaded once, 349 MB, to the user cache folder)",
    )
    parser.add_argument(
        "--location-head",
        help="location head (.npz: W, b, classes) for subcell_location_divergence; without it that "
        "metric is skipped",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="device for SubCell and Inception (default: cuda, then mps, then cpu, whichever is available)",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        help="in <prefix>_groups.csv, flag groups with fewer complete cells than this",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _say(message: str = "") -> None:
    print(message, file=sys.stderr)


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


def _panel_requirements(chosen, manifest, location_head) -> tuple:
    """Drop panel metrics whose inputs are absent, saying why."""
    kept = []
    for metric in chosen:
        if panel.NEEDS_GROUP in metric.needs and not manifest.has_groups:
            _say(f"Skipping {metric.name}: needs a group_id column (the other cells of each protein).")
            continue
        if panel.NEEDS_REFERENCES in metric.needs and not manifest.has_references:
            _say(f"Skipping {metric.name}: needs the nucleus, microtubules and er columns.")
            continue
        if panel.NEEDS_HEAD in metric.needs and not location_head:
            _say(f"Skipping {metric.name}: needs --location-head (not public at present).")
            continue
        kept.append(metric)
    return tuple(kept)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        manifest = read_manifest(args.manifest)
    except ManifestError as error:
        _say(f"cm4ai-evaluate: {error}")
        return 2
    if manifest.ignored_columns:
        _say(f"Ignoring manifest column(s): {', '.join(manifest.ignored_columns)}")

    try:
        selection = resolve(args.metrics, include_inception=args.inception)
    except SelectionError as error:
        _say(f"cm4ai-evaluate: {error}")
        return 2
    for extra, names in selection.missing.items():
        if selection.explicit or extra == scoring.INCEPTION:
            _say(f"cm4ai-evaluate: {', '.join(names)} need the [{extra}] packages.")
            _say("  Install with: " + INSTALL_HINT.format(extra=extra))
            return 2
        _say(f"Skipping {', '.join(names)}: the [{extra}] packages are not installed.")
        _say("  Install with: " + INSTALL_HINT.format(extra=extra))
    metrics = selection.pair
    chosen_panel = _panel_requirements(selection.panel, manifest, args.location_head)

    extractor = None
    if any(m.requires == scoring.INCEPTION for m in metrics):
        from .extended import InceptionFeatureExtractor

        _say("Loading Inception (downloads model weights on first use)...")
        extractor = InceptionFeatureExtractor(device=args.device)
        _say(f"  on device {extractor.device}")

    subcell_extractor = head = None
    if any(m.requires == scoring.SUBCELL for m in chosen_panel):
        from . import subcell

        try:
            if panel.LOCATION in chosen_panel:
                head = subcell.load_location_head(args.location_head)
            _say("Loading SubCell ...")
            subcell_extractor = subcell.SubCellExtractor(args.subcell_weights, device=args.device, say=_say)
        except (OSError, ValueError, KeyError) as error:
            _say(f"cm4ai-evaluate: {error}")
            return 2
        _say(f"  on device {subcell_extractor.device}")

    rows = manifest.rows
    _say(f"Scoring {len(rows)} row(s) in {args.mode} mode.")
    step = max(1, len(rows) // 10)
    records = []
    for i, row in enumerate(rows, start=1):
        records.append(score_row(row, args.mode, metrics, extractor))
        if i % step == 0 or i == len(rows):
            _say(f"  {i}/{len(rows)}")

    cells, cell_values = panel.score_panel(
        rows, records, args.mode, chosen_panel, subcell_extractor, head, say=_say
    )

    prefix = Path(args.output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = prefix.with_name(prefix.name + ".jsonl")
    csv_path = prefix.with_name(prefix.name + ".csv")
    cells_path = prefix.with_name(prefix.name + "_cells.csv")
    groups_path = prefix.with_name(prefix.name + "_groups.csv")
    pair_names = scoring.output_names(metrics)
    image_names = panel.image_level(chosen_panel)
    cell_names = panel.cell_level(chosen_panel)
    write_jsonl(records, jsonl_path)
    write_csv(records, pair_names, csv_path, masked_only=image_names)
    cell_rows = cell_summary(cells, cell_values, pair_names + image_names, cell_names)
    write_table(cell_rows, cells_path)
    written = [jsonl_path, csv_path, cells_path]
    if manifest.has_groups:
        groups = group_summary(
            cell_rows, pair_names + image_names + cell_names, args.min_cells, tuple(image_names + cell_names)
        )
        write_table(groups, groups_path)
        written.append(groups_path)

    _summary(records)
    for metric in chosen_panel:
        _say(f"  {metric.name}: category {metric.category}, one value per {metric.level}, lower is better")
    _say("Wrote " + ", ".join(str(p) for p in written))
    return 1 if any(r["status"] == ERROR for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
