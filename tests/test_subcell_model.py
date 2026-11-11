"""The real SubCell model. Runs only with the [subcell] extra and a weights file:
CM4AI_SUBCELL_WEIGHTS=/path/to/all_channels_MAE-CellS-ProtS-Pool.pth pytest tests/test_subcell_model.py
"""

import os

import numpy as np
import pytest

pytest.importorskip("torch")
WEIGHTS = os.environ.get("CM4AI_SUBCELL_WEIGHTS")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="set CM4AI_SUBCELL_WEIGHTS to run")


def test_features_shape_and_empty_mask():
    from cm4ai_evaluate.subcell import SubCellExtractor

    rng = np.random.default_rng(0)
    protein = rng.random((2, 512, 512), dtype=np.float32)
    references = rng.random((2, 3, 512, 512), dtype=np.float32)
    mask = np.zeros((2, 512, 512), bool)
    mask[0, 100:400, 100:400] = True  # second cell: empty mask
    out = SubCellExtractor(WEIGHTS, device="cpu").features(protein, mask, references, arms=("protein", "full"))
    assert out["protein"].shape == out["full"].shape == (2, 1536)
    assert np.isfinite(out["protein"][0]).all() and np.isnan(out["protein"][1]).all()
