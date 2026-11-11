"""SubCell input shape checks that need no pretrained weights."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
subcell = pytest.importorskip("cm4ai_evaluate.subcell")


class UnexpectedModel:
    patch = 16

    def __call__(self, image):
        raise AssertionError("incompatible feature inputs must fail before model inference")


def extractor(model):
    result = object.__new__(subcell.SubCellExtractor)
    result.device = torch.device("cpu")
    result.batch = 1
    result.model = model
    return result


@pytest.mark.parametrize(
    "protein,mask,references,arms",
    [
        (np.zeros((1, 8, 16)), np.ones((1, 8, 16)), None, ("protein",)),
        (np.zeros((1, 16, 16)), np.ones((1, 16, 15)), None, ("protein",)),
        (np.zeros((1, 16, 16)), np.ones((1, 16, 16)), np.ones((1, 3, 15, 16)), ("full",)),
    ],
    ids=["smaller-than-patch", "mask-size-mismatch", "reference-size-mismatch"],
)
def test_incompatible_features_raise_value_error(protein, mask, references, arms):
    with pytest.raises(ValueError):
        extractor(UnexpectedModel()).features(protein, mask, references, arms=arms)


@pytest.mark.parametrize("shape", [(16, 16), (17, 31), (32, 48)])
def test_features_accept_patch_size_and_nonmultiples(shape):
    model = subcell.SubCellViT(d=24, depth=1, heads=3, patch=16, img=32, pool_int=16).eval()
    values = np.random.default_rng(0).random((1, *shape), dtype=np.float32)
    features = extractor(model).features(values, np.ones_like(values, dtype=bool))
    assert features["protein"].shape == (1, 48)
    assert np.isfinite(features["protein"]).all()
