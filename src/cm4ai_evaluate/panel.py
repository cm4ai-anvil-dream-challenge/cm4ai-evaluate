"""The panel metrics: measures from the evaluation team's preliminary panel.

Unlike the pair metrics in scoring.py, these need more than one image pair:
brightness spread compares a prediction with the other cells of its protein,
the SubCell metrics need the model and (for location) the reference channels,
and the energy score compares the several predictions of one cell. So they
run in a second pass over the whole manifest, after the pair metrics.

Each value is raw and oriented lower is better. The category is the panel's;
no weights or anchors are applied (see docs/metrics.md).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import io
from . import panel_math as pm
from .scoring import CORE, SUBCELL

IMAGE = "image"  # one value per prediction
CELL = "cell"  # one value per cell (all its predictions)

NEEDS_GROUP = "group_id"
NEEDS_REFERENCES = "reference channels"
NEEDS_HEAD = "location head"

CELL_MASK = "cell_mask"  # masked method label: computed inside the cell mask only


@dataclass(frozen=True)
class PanelMetric:
    name: str
    level: str
    category: str
    requires: str  # extra
    needs: frozenset
    panel_name: str


BRIGHTNESS = PanelMetric(
    "brightness_spread_w1", IMAGE, "Brightness spread", CORE, frozenset({NEEDS_GROUP}), "hist_w1_si"
)
PATTERN = PanelMetric(
    "subcell_pattern_distance", IMAGE, "Pattern similarity", SUBCELL, frozenset(), "emb_dist_subcell"
)
LOCATION = PanelMetric(
    "subcell_location_divergence",
    IMAGE,
    "Location",
    SUBCELL,
    frozenset({NEEDS_REFERENCES, NEEDS_HEAD}),
    "loc2_kolfhead_jsd (negated)",
)
ENERGY = PanelMetric(
    "subcell_energy_score", CELL, "Variability/accuracy", SUBCELL, frozenset(), "q3m_es_fair_ludvig_subcell"
)

PANEL_METRICS: tuple[PanelMetric, ...] = (BRIGHTNESS, PATTERN, LOCATION, ENERGY)


def image_level(metrics) -> list[str]:
    return [m.name for m in metrics if m.level == IMAGE]


def cell_level(metrics) -> list[str]:
    return [m.name for m in metrics if m.level == CELL]


@dataclass
class Cell:
    cell_id: str
    group_id: str | None
    rows: list  # (ManifestRow, record)
    problem: str | None = None

    @property
    def complete(self) -> bool:
        """Every prediction listed for this cell scored, regardless of count."""
        return all(record["status"] == "scored" for _, record in self.rows)


@dataclass
class CellImages:
    truth: np.ndarray
    mask: np.ndarray
    references: np.ndarray | None = None


def build_cells(rows, records) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    for row, record in zip(rows, records):
        cell = cells.setdefault(row.cell_id, Cell(row.cell_id, row.group_id, []))
        cell.rows.append((row, record))
    return cells


def _unit(image: io.LoadedImage) -> np.ndarray:
    return pm.unit_float32(image.pixels, image.max_value)


def _load_cell(cell: Cell, mode: str, want_references: bool) -> CellImages | None:
    """Truth, mask and (if wanted) reference channels, from the cell's first row."""
    row = cell.rows[0][0]
    if row.mask is None:
        cell.problem = "no mask"
        return
    try:
        truth = io.read_image(row.ground_truth, io.GROUND_TRUTH, mode)
    except io.ImageCheckError as error:
        cell.problem = f"ground truth unreadable: {error}"
        return
    mask = io.read_mask(row.mask, truth.pixels.shape, mode)
    if mask.mask is None:
        cell.problem = f"mask skipped: {mask.skip_reason}"
        return
    images = CellImages(_unit(truth), mask.mask)
    if want_references and row.references:
        try:
            channels = [io.read_image(row.references[c], io.GROUND_TRUTH, mode) for c in ("nucleus", "microtubules", "er")]
            if any(c.pixels.shape != truth.pixels.shape for c in channels):
                raise io.ImageCheckError("reference", "shape", "reference channel size differs from the ground truth")
            images.references = np.stack([_unit(c) for c in channels])
        except io.ImageCheckError:
            images.references = None
    return images


def _set(record: dict, name: str, value: float, note: str | None = None) -> None:
    if record["masked_metrics"] is None:
        return
    record["masked_metrics"][name] = None if not np.isfinite(value) else float(value)
    record["masked_methods"][name] = CELL_MASK
    if note:
        record["notes"][f"masked {name}"] = note


def _usable(record: dict) -> bool:
    return record["status"] == "scored" and record["masked_metrics"] is not None


def _prediction(row, mode) -> np.ndarray:
    return _unit(io.read_image(row.prediction, io.PREDICTION, mode))


def score_brightness(cells: dict[str, Cell], mode: str) -> None:
    """Brightness spread: each prediction against the mean histogram of the other cells of its protein.

    Other cells are those with a different cell_id and a different ground truth file, so
    the reference never contains the cell's own image."""
    truth_cdf = {}
    for key, cell in cells.items():
        images = _load_cell(cell, mode, False)
        truth_cdf[key] = pm.brightness_cdf(images.truth, images.mask) if images is not None else None
        del images
    by_group = defaultdict(list)
    for k, c in cells.items():
        if c.group_id:
            by_group[c.group_id].append(k)
    for k, cell in cells.items():
        images = _load_cell(cell, mode, False)
        if images is None:
            continue
        if not cell.group_id:
            reference, why = None, "no group_id for this row"
        else:
            own = cell.rows[0][0].ground_truth
            others = [
                truth_cdf[o] for o in by_group[cell.group_id] if o != k and cells[o].rows[0][0].ground_truth != own
            ]
            reference = pm.brightness_reference(others)
            why = None if reference is not None else "no other cell of this group has usable ground truth"
        for row, record in cell.rows:
            if not _usable(record):
                continue
            if reference is None:
                _set(record, BRIGHTNESS.name, float("nan"), why)
                continue
            value = pm.brightness_spread_w1(_prediction(row, mode), images.mask, reference)
            note = None if np.isfinite(value) else "prediction has no in-mask signal (99th percentile at or below 1/1020)"
            _set(record, BRIGHTNESS.name, value, note)
        del images


def _cell_features(images, rows, mode, extractor, arms):
    batch = extractor.batch
    chunks = {arm: [] for arm in arms}
    for start in range(0, len(rows) + 1, batch):
        stop = min(start + batch, len(rows) + 1)
        protein = np.stack([
            images.truth if i == 0 else _prediction(rows[i - 1][0], mode)
            for i in range(start, stop)
        ])
        masks = np.repeat(images.mask[None], len(protein), axis=0)
        refs = np.repeat(images.references[None], len(protein), axis=0) if "full" in arms else None
        features = extractor.features(protein, masks, refs, arms=arms)
        for arm in arms:
            chunks[arm].append(features[arm])
    return {arm: np.concatenate(values) for arm, values in chunks.items()}


def score_subcell(cells, mode, chosen, extractor, head, say=print) -> dict[str, float]:
    """Pattern distance and location per prediction; energy score per cell. Returns {cell_id: energy}."""
    want_location = LOCATION in chosen
    want_energy = ENERGY in chosen
    arms = ("protein", "full") if want_location else ("protein",)
    energy: dict[str, float] = {}
    todo = list(cells.values())
    say(f"SubCell features for {len(todo)} cell(s) and their predictions on {extractor.device} ...")
    for n, cell in enumerate(todo, start=1):
        images = _load_cell(cell, mode, want_location)
        if images is None:
            continue
        rows = [(row, record) for row, record in cell.rows if _usable(record)]
        if not rows:
            del images
            continue
        use_full = want_location and images.references is not None
        cell_arms = arms if use_full else ("protein",)
        try:
            features = _cell_features(images, rows, mode, extractor, cell_arms)
        except io.ImageCheckError as error:
            note = str(error)
            for _, record in rows:
                for metric in chosen:
                    if metric.requires == SUBCELL:
                        if metric.level == IMAGE:
                            _set(record, metric.name, float("nan"), note)
                        else:
                            record["notes"][f"cell {metric.name}"] = note
            if want_energy:
                energy[cell.cell_id] = float("nan")
            say(f"Skipping SubCell for cell {cell.cell_id}: {note}")
            del images
            continue
        del images
        truth_protein = features["protein"][0]
        for i, (row, record) in enumerate(rows, start=1):
            if PATTERN in chosen:
                _set(record, PATTERN.name, pm.pattern_distance(features["protein"][i], truth_protein))
            if want_location:
                if use_full:
                    value = pm.location_divergence(features["full"][i], features["full"][0], head["W"], head["b"])
                    _set(record, LOCATION.name, value)
                else:
                    _set(record, LOCATION.name, float("nan"), "reference channels missing or unreadable")
        if want_energy:
            if cell.complete:
                energy[cell.cell_id] = pm.energy_score(features["protein"][1:], truth_protein)
            else:
                energy[cell.cell_id] = float("nan")
        if n % max(1, len(todo) // 10) == 0 or n == len(todo):
            say(f"  {n}/{len(todo)} cells")
    return energy


def score_panel(rows, records, mode, chosen, extractor=None, head=None, say=print):
    """Add the chosen panel metrics to the records in place; return the cells and per-cell values."""
    cells = build_cells(rows, records)
    if not chosen:
        return cells, {}
    if BRIGHTNESS in chosen:
        score_brightness(cells, mode)
    cell_values: dict[str, dict[str, float]] = defaultdict(dict)
    if any(m.requires == SUBCELL for m in chosen):
        energy = score_subcell(cells, mode, chosen, extractor, head, say=say)
        for k, v in energy.items():
            cell_values[k][ENERGY.name] = v
    return cells, cell_values
