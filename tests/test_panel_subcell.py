"""SubCell bookkeeping with a stand-in feature extractor (no torch needed).

The real model's features were checked against dream_eval outside the test
suite: identical on 32 sample images (docs/metrics.md, Provenance).
"""

import gc
import weakref

import numpy as np
import pytest
from PIL import Image

from cm4ai_evaluate import panel
from cm4ai_evaluate import panel_math as pm
from cm4ai_evaluate.manifest import read_manifest
from cm4ai_evaluate.runner import score_row


class FakeExtractor:
    """Features = a fixed projection of the masked image, so values are deterministic."""

    device = "fake"
    batch = 8

    def __init__(self):
        self.calls = []

    def features(self, protein, mask, references=None, arms=("protein",)):
        self.calls.append((len(protein), tuple(arms), references is not None))
        flat = (np.asarray(protein) * mask).reshape(len(protein), -1)
        base = np.stack([flat[:, i::1536].sum(1) for i in range(1536)], 1).astype(np.float32) + 1.0
        out = {"protein": base}
        if "full" in arms:
            out["full"] = base + np.asarray(references).reshape(len(protein), -1)[:, :1536]
        return out


def png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (512, 512), dtype=np.uint8)).save(path)


@pytest.fixture
def manifest(tmp_path):
    mask = np.zeros((512, 512), np.uint8)
    mask[100:400, 100:400] = 255
    Image.fromarray(mask).save(tmp_path / "m.png")
    for name, seed in [("g0", 0), ("g1", 1), ("p0", 10), ("p1", 11), ("p2", 12), ("n", 20), ("t", 21), ("e", 22)]:
        png(tmp_path / f"{name}.png", seed)
    lines = ["prediction_id,prediction,ground_truth,mask,cell_id,group_id,nucleus,microtubules,er"]
    lines += [f"c0-v{v},p{v}.png,g0.png,m.png,c0,A,n.png,t.png,e.png" for v in range(2)]
    lines += ["c1-v0,p2.png,g1.png,m.png,c1,A,,,", "c1-v1,missing.png,g1.png,m.png,c1,A,,,"]
    (tmp_path / "m.csv").write_text("\n".join(lines) + "\n")
    return read_manifest(tmp_path / "m.csv")


def test_subcell_values_and_energy(manifest):
    rows = manifest.rows
    records = [score_row(r, "strict", ()) for r in rows]
    fake = FakeExtractor()
    head = {"W": np.random.default_rng(0).normal(size=(4, 1536)), "b": np.zeros(4)}
    chosen = (panel.PATTERN, panel.LOCATION, panel.ENERGY)
    cells, values = panel.score_panel(rows, records, "strict", chosen, fake, head, say=lambda *_: None)
    by_id = {r["prediction_id"]: r for r in records}

    # one extractor call per cell: the truth plus its scored predictions
    assert fake.calls == [(3, ("protein", "full"), True), (2, ("protein",), False)]
    assert by_id["c0-v0"]["masked_metrics"]["subcell_pattern_distance"] > 0
    assert by_id["c0-v0"]["masked_methods"]["subcell_pattern_distance"] == panel.CELL_MASK
    assert by_id["c0-v1"]["masked_metrics"]["subcell_location_divergence"] >= 0
    # c1 has no reference channels: location empty with a note
    assert by_id["c1-v0"]["masked_metrics"]["subcell_location_divergence"] is None
    assert "reference channels" in by_id["c1-v0"]["notes"]["masked subcell_location_divergence"]

    # energy score: complete cell c0 gets a value equal to the math on its two predictions
    truth = pm.unit_float32(np.asarray(Image.open(rows[0].ground_truth)))
    mask = np.asarray(Image.open(rows[0].mask)) > 0
    feats = fake.features(
        np.stack([truth] + [pm.unit_float32(np.asarray(Image.open(r.prediction))) for r in rows[:2]]),
        np.repeat(mask[None], 3, 0),
    )["protein"]
    assert values["c0"]["subcell_energy_score"] == pytest.approx(pm.energy_score(feats[1:], feats[0]))
    # c1 lost a prediction (file missing): incomplete, so no energy score
    assert not cells["c1"].complete
    assert np.isnan(values["c1"]["subcell_energy_score"])


def test_cell_images_do_not_accumulate_across_cells(manifest, monkeypatch):
    load = panel._load_cell
    arrays = []
    max_live_bytes = 0

    def observed_load(cell, mode, want_references):
        nonlocal max_live_bytes
        gc.collect()
        live_bytes = sum(ref().nbytes for ref in arrays if ref() is not None)
        max_live_bytes = max(max_live_bytes, live_bytes)
        images = load(cell, mode, want_references)
        owner = images if images is not None else cell
        arrays.extend(
            weakref.ref(value)
            for value in (getattr(owner, name, None) for name in ("truth", "mask", "references"))
            if value is not None
        )
        gc.collect()
        live_bytes = sum(ref().nbytes for ref in arrays if ref() is not None)
        max_live_bytes = max(max_live_bytes, live_bytes)
        return images

    monkeypatch.setattr(panel, "_load_cell", observed_load)
    rows = manifest.rows
    records = [score_row(row, "strict", ()) for row in rows]
    panel.score_panel(rows, records, "strict", (panel.PATTERN,), FakeExtractor(), say=lambda *_: None)

    gc.collect()
    live_bytes = sum(ref().nbytes for ref in arrays if ref() is not None)
    one_cell_bytes = 512 * 512 * (np.dtype(np.float32).itemsize + np.dtype(bool).itemsize)
    assert max_live_bytes <= one_cell_bytes
    assert live_bytes == 0


def test_energy_score_uses_each_cells_manifest_prediction_count(tmp_path):
    mask = np.full((512, 512), 255, dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "mask.png")
    lines = ["prediction_id,prediction,ground_truth,mask,cell_id"]
    expected = {}
    for count in (1, 2, 5):
        cell_id = f"k{count}"
        truth_name = f"{cell_id}-truth.png"
        png(tmp_path / truth_name, count)
        prediction_names = []
        for index in range(count):
            prediction_name = f"{cell_id}-p{index}.png"
            png(tmp_path / prediction_name, 10 + count + index)
            prediction_names.append(prediction_name)
            lines.append(f"{cell_id}-v{index},{prediction_name},{truth_name},mask.png,{cell_id}")
        expected[cell_id] = (truth_name, prediction_names)
    manifest_path = tmp_path / "variable-k.csv"
    manifest_path.write_text("\n".join(lines) + "\n")
    manifest = read_manifest(manifest_path)
    records = [score_row(row, "strict", ()) for row in manifest.rows]
    fake = FakeExtractor()

    cells, values = panel.score_panel(
        manifest.rows, records, "strict", (panel.ENERGY,), fake, say=lambda *_: None
    )

    for cell_id, (truth_name, prediction_names) in expected.items():
        assert cells[cell_id].complete
        truth = pm.unit_float32(np.asarray(Image.open(tmp_path / truth_name)))
        predictions = [pm.unit_float32(np.asarray(Image.open(tmp_path / name))) for name in prediction_names]
        features = fake.features(
            np.stack([truth, *predictions]),
            np.repeat((mask > 0)[None], len(predictions) + 1, axis=0),
        )["protein"]
        assert values[cell_id][panel.ENERGY.name] == pytest.approx(
            pm.energy_score(features[1:], features[0])
        )
