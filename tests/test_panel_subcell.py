"""SubCell bookkeeping with a stand-in feature extractor (no torch needed).

The real model's features were checked against dream_eval outside the test
suite: identical on 32 sample images (docs/metrics.md, Provenance).
"""

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
    feats = fake.features(
        np.stack([cells["c0"].truth] + [pm.unit_float32(np.asarray(Image.open(r.prediction))) for r in rows[:2]]),
        np.repeat(cells["c0"].mask[None], 3, 0),
    )["protein"]
    assert values["c0"]["subcell_energy_score"] == pytest.approx(pm.energy_score(feats[1:], feats[0]))
    # c1 lost a prediction (file missing): incomplete, so no energy score
    assert not cells["c1"].complete
    assert np.isnan(values["c1"]["subcell_energy_score"])
