"""Score every row of a manifest, one record per row.

A row whose prediction or ground truth fails a check is recorded with
status "error" and the run continues; a mask that fails a check only skips
the masked metrics for that row.
"""

from __future__ import annotations

import math
from typing import Iterator

from . import io, scoring
from .extended import MetricResult
from .manifest import ManifestRow

SCORED = "scored"
ERROR = "error"


def _value(result: MetricResult) -> float | None:
    """JSON has no infinity, so an infinite value is written as null (its note explains)."""
    if result.value is None or not math.isfinite(result.value):
        return None
    return result.value


def _deviations(role: str, deviations) -> list[dict]:
    return [
        {"role": role, "code": d.code, "message": d.message, "warning": d.warn}
        for d in deviations
    ]


def score_row(
    row: ManifestRow,
    mode: str,
    metrics: tuple[scoring.Metric, ...],
    extractor=None,
) -> dict:
    """One output record for one manifest row."""
    record = {
        "prediction_id": row.prediction_id,
        "status": SCORED,
        "mode": mode,
        "spec_compliant": True,
        "metrics": None,
        "masked_metrics": None,
        "masked_methods": None,
        "mask_pixel_count": None,
        "mask_skip_reason": None,
        "notes": {},
        "deviations": [],
        "error": None,
        "prediction": str(row.prediction),
        "ground_truth": str(row.ground_truth),
        "mask": str(row.mask) if row.mask else None,
    }
    try:
        prediction = io.read_image(row.prediction, io.PREDICTION, mode)
        ground_truth = io.read_image(row.ground_truth, io.GROUND_TRUTH, mode)
        io.check_pair(prediction, ground_truth)
    except io.ImageCheckError as error:
        record.update(
            status=ERROR,
            spec_compliant=False,
            error={"role": error.role, "code": error.code, "message": str(error)},
        )
        return record
    record["deviations"] = _deviations(io.PREDICTION, prediction.deviations) + _deviations(
        io.GROUND_TRUTH, ground_truth.deviations
    )

    mask = None
    if row.mask is not None:
        mask_result = io.read_mask(row.mask, prediction.pixels.shape, mode)
        record["deviations"] += _deviations(io.MASK, mask_result.deviations)
        if mask_result.mask is None:
            record["mask_skip_reason"] = mask_result.skip_reason
        else:
            mask = mask_result.mask
            record["mask_pixel_count"] = int(mask.sum())

    results = scoring.score_pair(prediction, ground_truth, mask, metrics, extractor)
    record["metrics"] = {name: _value(r) for name, r in results["unmasked"].items()}
    notes = {f"unmasked {n}": r.note for n, r in results["unmasked"].items() if r.note}
    if results["masked"] is not None:
        record["masked_metrics"] = {name: _value(r) for name, r in results["masked"].items()}
        record["masked_methods"] = scoring.mask_methods(metrics)
        notes.update({f"masked {n}": r.note for n, r in results["masked"].items() if r.note})
    record["notes"] = notes
    record["spec_compliant"] = not record["deviations"] and record["mask_skip_reason"] is None
    return record


def score_manifest(rows, mode, metrics, extractor=None) -> Iterator[dict]:
    for row in rows:
        yield score_row(row, mode, metrics, extractor)



