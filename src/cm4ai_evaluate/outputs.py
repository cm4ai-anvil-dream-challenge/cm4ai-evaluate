"""Write results: JSON Lines for downstream aggregation, CSV for people.

JSONL: one record per manifest row, shaped like the challenge grading rows
(an identifier plus a metrics object), with masked results, deviations,
notes and any error alongside.

CSV: one row per manifest row. Every metric appears twice, prefixed
unmasked_ and masked_; then the mask pixel count and skip reason,
deviations, notes, the error, and the input paths.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def csv_columns(metric_names: list[str]) -> list[str]:
    return (
        ["prediction_id", "status", "mode", "spec_compliant"]
        + [f"unmasked_{n}" for n in metric_names]
        + [f"masked_{n}" for n in metric_names]
        + ["mask_pixel_count", "mask_skip_reason", "deviations", "notes", "error"]
        + ["prediction", "ground_truth", "mask"]
    )


def write_csv(records: list[dict], metric_names: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_columns(metric_names))
        for r in records:
            unmasked = r["metrics"] or {}
            masked = r["masked_metrics"] or {}
            deviations = "; ".join(f"{d['role']} {d['code']}" for d in r["deviations"])
            notes = "; ".join(f"{k}: {v}" for k, v in r["notes"].items())
            error = f"{r['error']['code']}: {r['error']['message']}" if r["error"] else None
            writer.writerow(
                [r["prediction_id"], r["status"], r["mode"], _cell(r["spec_compliant"])]
                + [_cell(unmasked.get(n)) for n in metric_names]
                + [_cell(masked.get(n)) for n in metric_names]
                + [_cell(r["mask_pixel_count"]), _cell(r["mask_skip_reason"]), deviations, notes]
                + [_cell(error), r["prediction"], r["ground_truth"], _cell(r["mask"])]
            )
