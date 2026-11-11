"""Manifest input: the only way to pair images.

CSV or JSON, chosen by file extension. Required columns: prediction and
ground_truth. Optional columns: prediction_id, mask, cell_id, group_id, and
the reference channels nucleus, microtubules and er. Relative paths
resolve against the manifest's folder; a leading ~ means the home folder.
Other columns are ignored (reported once), so an existing manifest with
extra columns can be reused.

prediction_id identifies one prediction (one row), matching the
blast-proticelli manifest and the participant contract. It is any non-empty
string and must be unique within the manifest; several predictions can come
from the same sample, or the same images can be scored more than once. When
the column is absent, each row's 1-based row number is used instead.

cell_id groups the predictions of one cell (the challenge asks for several
per cell); when absent, each row is its own cell. Rows of one cell must share
ground_truth, mask, group_id and reference channels.
group_id groups cells for per-protein summaries and for metrics that compare
a cell with the other cells of its protein (usually the antibody); when
absent there are no groups. The reference channels are needed only by the
SubCell location metric.

JSON form: a list of objects, one per row, with the same keys as the CSV
columns.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("prediction", "ground_truth")
OPTIONAL_COLUMNS = ("prediction_id", "mask", "cell_id", "group_id")
REFERENCE_COLUMNS = ("nucleus", "microtubules", "er")


class ManifestError(ValueError):
    """The manifest cannot be used; nothing is scored."""


@dataclass(frozen=True)
class ManifestRow:
    prediction_id: str
    prediction: Path
    ground_truth: Path
    mask: Path | None
    row_number: int  # 1-based, counting data rows only
    cell_id: str = ""  # empty only if built by hand; read_manifest always fills it
    group_id: str | None = None
    references: dict[str, Path] | None = None  # nucleus, microtubules, er


@dataclass(frozen=True)
class Manifest:
    path: Path
    rows: tuple[ManifestRow, ...]
    ignored_columns: tuple[str, ...]
    has_groups: bool = False
    has_references: bool = False


def _resolve(value: str, base: Path) -> Path:
    # Not Path.resolve(): that would follow symlinks, which the image checks
    # need to see.
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else base / path


def _records(path: Path) -> tuple[list[dict], list[str]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = [c.strip() for c in (reader.fieldnames or [])]
            records = [
                {(k or "").strip(): (v or "") for k, v in row.items() if k is not None}
                for row in reader
            ]
        return records, columns
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ManifestError(f"{path.name} is not valid JSON: {error}") from error
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ManifestError(f"{path.name} must be a JSON list of objects, one per row")
        columns: list[str] = []
        for item in data:
            columns += [k for k in item if k not in columns]
        records = [{k: "" if v is None else str(v) for k, v in item.items()} for item in data]
        return records, columns
    raise ManifestError(f"manifest must be .csv or .json, got {path.name}")


def read_manifest(path: str | Path) -> Manifest:
    """Read and check a manifest. Raises ManifestError with the row at fault."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    records, columns = _records(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ManifestError(
            f"{path.name} is missing required column(s): {', '.join(missing)} "
            f"(found: {', '.join(columns) or 'none'})"
        )
    if not records:
        raise ManifestError(f"{path.name} has no rows")
    has_id = "prediction_id" in columns
    has_cell = "cell_id" in columns
    has_group = "group_id" in columns
    present = [c for c in REFERENCE_COLUMNS if c in columns]
    if present and len(present) < len(REFERENCE_COLUMNS):
        missing_refs = [c for c in REFERENCE_COLUMNS if c not in columns]
        raise ManifestError(
            f"{path.name} has reference column(s) {', '.join(present)} but not "
            f"{', '.join(missing_refs)}: give all three or none"
        )
    has_references = len(present) == len(REFERENCE_COLUMNS)
    base = path.parent
    rows, seen = [], {}
    for number, record in enumerate(records, start=1):
        for column in REQUIRED_COLUMNS:
            if not record.get(column, "").strip():
                raise ManifestError(f"row {number}: {column} is empty")
        prediction_id = record.get("prediction_id", "").strip() if has_id else str(number)
        if not prediction_id:
            raise ManifestError(f"row {number}: prediction_id is empty")
        if prediction_id in seen:
            raise ManifestError(
                f"row {number}: prediction_id {prediction_id!r} repeats row {seen[prediction_id]}"
            )
        seen[prediction_id] = number
        mask = record.get("mask", "").strip()
        cell_id = record.get("cell_id", "").strip() if has_cell else ""
        if not cell_id:
            cell_id = prediction_id
        group_id = record.get("group_id", "").strip() or None if has_group else None
        references = None
        if has_references:
            values = {c: record.get(c, "").strip() for c in REFERENCE_COLUMNS}
            if all(values.values()):
                references = {c: _resolve(v, base) for c, v in values.items()}
        rows.append(
            ManifestRow(
                prediction_id=prediction_id,
                prediction=_resolve(record["prediction"], base),
                ground_truth=_resolve(record["ground_truth"], base),
                mask=_resolve(mask, base) if mask else None,
                row_number=number,
                cell_id=cell_id,
                group_id=group_id,
                references=references,
            )
        )
    first: dict[str, ManifestRow] = {}
    for row in rows if has_cell else ():
        other = first.setdefault(row.cell_id, row)
        for attribute in ("ground_truth", "mask", "group_id", "references"):
            if getattr(row, attribute) != getattr(other, attribute):
                raise ManifestError(
                    f"row {row.row_number}: cell {row.cell_id!r} has a different {attribute} "
                    f"from row {other.row_number}; the predictions of one cell share them"
                )
    known = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS) | set(REFERENCE_COLUMNS)
    ignored = tuple(c for c in columns if c not in known)
    return Manifest(
        path=path,
        rows=tuple(rows),
        ignored_columns=ignored,
        has_groups=has_group and any(r.group_id for r in rows),
        has_references=has_references,
    )
