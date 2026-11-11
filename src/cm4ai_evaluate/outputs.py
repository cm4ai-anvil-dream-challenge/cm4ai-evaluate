"""Write results: JSON Lines for downstream aggregation, CSV for people.

JSONL: one record per manifest row, shaped like the challenge grading rows
(an identifier plus a metrics object), with masked results, deviations,
notes and any error alongside.

CSV: one row per manifest row. Every pair metric appears twice, prefixed
unmasked_ and masked_; panel metrics, computed inside the cell mask only,
appear once, prefixed masked_. Then the mask pixel count and skip reason,
deviations, notes, the error, and the input paths.

Summaries (CSV): <prefix>_cells.csv, one row per cell, with each masked
metric averaged over the cell's predictions and the per-cell metrics; and,
when the manifest has group_id, <prefix>_groups.csv, one row per group with
each cell value averaged over the group's complete cells. Following the
challenge grading, a cell's average is empty when any of its predictions
lacks that value.
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


def csv_columns(metric_names: list[str], masked_only: list[str] = ()) -> list[str]:
    return (
        ["prediction_id", "status", "mode", "spec_compliant"]
        + [f"unmasked_{n}" for n in metric_names]
        + [f"masked_{n}" for n in list(metric_names) + list(masked_only)]
        + ["mask_pixel_count", "mask_skip_reason", "deviations", "notes", "error"]
        + ["prediction", "ground_truth", "mask"]
    )


def write_csv(records: list[dict], metric_names: list[str], path: Path, masked_only: list[str] = ()) -> None:
    masked_names = list(metric_names) + list(masked_only)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_columns(metric_names, masked_only))
        for r in records:
            unmasked = r["metrics"] or {}
            masked = r["masked_metrics"] or {}
            deviations = "; ".join(f"{d['role']} {d['code']}" for d in r["deviations"])
            notes = "; ".join(f"{k}: {v}" for k, v in r["notes"].items())
            error = f"{r['error']['code']}: {r['error']['message']}" if r["error"] else None
            writer.writerow(
                [r["prediction_id"], r["status"], r["mode"], _cell(r["spec_compliant"])]
                + [_cell(unmasked.get(n)) for n in metric_names]
                + [_cell(masked.get(n)) for n in masked_names]
                + [_cell(r["mask_pixel_count"]), _cell(r["mask_skip_reason"]), deviations, notes]
                + [_cell(error), r["prediction"], r["ground_truth"], _cell(r["mask"])]
            )


def _mean(values: list) -> float | None:
    """Mean, or None if there are no values or any is missing."""
    if not values or any(v is None for v in values):
        return None
    return float(sum(values) / len(values))


def cell_summary(cells: dict, cell_values: dict, masked_names: list[str], cell_names: list[str]) -> list[dict]:
    rows = []
    for cell_id, cell in cells.items():
        records = [record for _, record in cell.rows]
        scored = [r for r in records if r["status"] == "scored"]
        row = {
            "cell_id": cell_id,
            "group_id": cell.group_id,
            "n_predictions": len(records),
            "n_scored": len(scored),
            "complete": cell.complete,
        }
        for name in masked_names:
            row[name] = _mean([(r["masked_metrics"] or {}).get(name) for r in records]) if cell.complete else None
        for name in cell_names:
            value = cell_values.get(cell_id, {}).get(name)
            row[name] = value if value is not None and value == value else None
        rows.append(row)
    return rows


def group_summary(
    cell_rows: list[dict], value_names: list[str], min_cells: int | None, counted: tuple = ()
) -> list[dict]:
    """Mean over complete cells; for names in counted, also how many cells had a value."""
    groups: dict[str, list[dict]] = {}
    for row in cell_rows:
        if row["group_id"]:
            groups.setdefault(row["group_id"], []).append(row)
    out = []
    for group_id, rows in groups.items():
        complete = [r for r in rows if r["complete"]]
        row = {"group_id": group_id, "n_cells": len(rows), "n_cells_complete": len(complete)}
        if min_cells is not None:
            row["enough_cells"] = len(complete) >= min_cells
        for name in value_names:
            values = [r[name] for r in complete if r[name] is not None]
            row[name] = float(sum(values) / len(values)) if values else None
            if name in counted:
                row[f"{name}_n_cells"] = len(values)
        out.append(row)
    return out


def write_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = list(rows[0])
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in columns])
