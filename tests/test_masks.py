"""Masked metrics: each method uses exactly the pixels it should."""

import math

import numpy as np
import pytest

from cm4ai_evaluate import extended as ex
from cm4ai_evaluate import io, masks, scoring
from cm4ai_evaluate.statistics import STATISTIC_NAMES

pytest.importorskip("skimage")
pytest.importorskip("sklearn")

SHAPE = (128, 128)


def textured(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    base = np.exp(-((x - 64) ** 2 + (y - 60) ** 2) / 900.0) + 0.3 * rng.random(SHAPE)
    return np.round(np.clip(base / base.max(), 0, 1) * 255).astype(np.uint8)


def loaded(pixels, role=io.PREDICTION):
    return io.LoadedImage(path=None, role=role, pixels=pixels)


def cell_mask():
    y, x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return (x - 64) ** 2 / 40**2 + (y - 60) ** 2 / 30**2 <= 1  # an ellipse


def values(results, key="masked"):
    return {k: v.value for k, v in results[key].items()}


@pytest.fixture(scope="module")
def pair():
    return loaded(textured(1)), loaded(textured(2), io.GROUND_TRUTH)


ALL = scoring.available_metrics(include_inception=False)


# Bounding box


def test_bounding_box():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3:8] = True
    mask[6, 4] = True
    assert masks.bounding_box(mask) == (slice(2, 7), slice(3, 8))


def test_bounding_box_empty():
    with pytest.raises(ValueError):
        masks.bounding_box(np.zeros((4, 4), bool))


# Haralick, exact GLCM


def test_full_mask_haralick_equals_unmasked():
    image = textured(3) / 255.0
    full = np.ones(SHAPE, bool)
    assert masks.masked_haralick_features(image, full) == pytest.approx(
        ex.haralick_features(image), rel=1e-12
    )


def test_masked_haralick_matches_brute_force():
    # Count neighbour pairs by hand on a small image and compare contrast
    rng = np.random.default_rng(5)
    q = rng.integers(0, 256, (12, 12))
    image = q / 255.0
    mask = np.zeros((12, 12), bool)
    mask[2:9, 3:11] = True
    mask[5, 5] = False
    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]  # 0, 45, 90, 135 degrees in skimage
    contrasts = []
    for dy, dx in offsets:
        diffs = []
        for y in range(12):
            for x in range(12):
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < 12 and 0 <= x2 < 12 and mask[y, x] and mask[y2, x2]:
                    diffs.append((q[y, x] - q[y2, x2]) ** 2)
        contrasts.append(np.mean(diffs))  # symmetric pairs give the same mean
    assert masks.masked_haralick_features(image, mask)["contrast"] == pytest.approx(
        np.mean(contrasts)
    )


def test_masked_haralick_ignores_outside_pixels():
    image = textured(6) / 255.0
    mask = cell_mask()
    changed = image.copy()
    changed[~mask] = np.random.default_rng(7).random((~mask).sum())
    assert masks.masked_haralick_features(changed, mask) == pytest.approx(
        masks.masked_haralick_features(image, mask)
    )


def test_masked_haralick_no_pairs():
    mask = np.zeros(SHAPE, bool)
    mask[10, 10] = True
    mask[20, 20] = True
    result = masks.masked_haralick_similarity(textured(1) / 255, textured(2) / 255, mask)
    assert result.value is None
    assert "pairs" in result.note or "constant" in result.note


# Scoring a pair


def test_no_mask_gives_no_masked_results(pair):
    results = scoring.score_pair(*pair, metrics=ALL)
    assert results["masked"] is None
    assert set(results["unmasked"]) == set(scoring.output_names(ALL))


def test_full_mask_equals_unmasked(pair):
    results = scoring.score_pair(*pair, mask=np.ones(SHAPE, bool), metrics=ALL)
    masked, unmasked = values(results), values(results, "unmasked")
    for name in unmasked:
        assert masked[name] == pytest.approx(unmasked[name], rel=1e-9), name


def test_pixel_and_glcm_metrics_ignore_everything_outside_mask(pair):
    p, t = pair
    mask = cell_mask()
    noisy = p.pixels.copy()
    noisy[~mask] = 255 - noisy[~mask]
    before = values(scoring.score_pair(p, t, mask=mask, metrics=ALL))
    after = values(scoring.score_pair(loaded(noisy), t, mask=mask, metrics=ALL))
    methods = scoring.mask_methods(ALL)
    for name, method in methods.items():
        if method in (masks.PIXELS, masks.GLCM):
            assert after[name] == pytest.approx(before[name], rel=1e-12), name


def test_bounding_box_metrics_ignore_only_outside_the_box(pair):
    p, t = pair
    mask = cell_mask()
    rows, cols = masks.bounding_box(mask)
    outside_box = np.ones(SHAPE, bool)
    outside_box[rows, cols] = False
    noisy = p.pixels.copy()
    noisy[outside_box] = 255 - noisy[outside_box]
    before = values(scoring.score_pair(p, t, mask=mask, metrics=ALL))
    after = values(scoring.score_pair(loaded(noisy), t, mask=mask, metrics=ALL))
    for name, method in scoring.mask_methods(ALL).items():
        if method == masks.BOUNDING_BOX:
            assert after[name] == pytest.approx(before[name], rel=1e-12), name


def test_masked_statistics_count_only_inside(pair):
    p, t = pair
    mask = cell_mask()
    masked = values(scoring.score_pair(p, t, mask=mask, metrics=ALL))
    diff = np.abs(p.pixels.astype(float) - t.pixels.astype(float))[mask]
    assert masked["mean_absolute_error"] == pytest.approx(diff.mean())
    assert masked["maximum_absolute_error"] == diff.max()


def test_constant_inside_mask_gives_none(pair):
    _, t = pair
    mask = cell_mask()
    flat = np.where(mask, 100, textured(9)).astype(np.uint8)  # constant inside the cell only
    results = scoring.score_pair(loaded(flat), t, mask=mask, metrics=ALL)
    assert results["unmasked"]["pearson_r"].value is not None
    assert results["masked"]["pearson_r"].value is None
    assert results["masked"]["haralick_similarity"].value is None


def test_tiny_mask_ms_ssim_none_with_note(pair):
    mask = np.zeros(SHAPE, bool)
    mask[50:55, 50:58] = True
    results = scoring.score_pair(*pair, mask=mask, metrics=ALL)
    ms = results["masked"]["ms_ssim"]
    assert ms.value is None and "smaller than" in ms.note
    assert math.isfinite(results["masked"]["mean_absolute_error"].value)


def test_small_mask_uses_fewer_ms_ssim_scales(pair):
    # A 40 x 40 box allows fewer than five scales: the early-stop branch
    mask = np.zeros(SHAPE, bool)
    mask[40:80, 40:80] = True
    value = scoring.score_pair(*pair, mask=mask, metrics=ALL)["masked"]["ms_ssim"].value
    assert 0 < value <= 1


# Registry


def test_registry_names_unique_and_complete():
    names = scoring.output_names()
    assert len(names) == len(set(names))
    assert set(STATISTIC_NAMES) <= set(names)
    assert set(scoring.mask_methods().values()) <= set(masks.MASK_METHODS)


def test_subset_of_metrics(pair):
    only = tuple(m for m in scoring.METRICS if m.outputs == ("pearson_r",))
    results = scoring.score_pair(*pair, mask=cell_mask(), metrics=only)
    assert list(results["unmasked"]) == ["pearson_r"] == list(results["masked"])


def test_inception_needs_extractor(pair):
    with pytest.raises(ValueError):
        scoring.score_pair(*pair, metrics=scoring.METRICS)


def test_inception_masked_on_bounding_box(pair):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    extractor = ex.InceptionFeatureExtractor()
    results = scoring.score_pair(*pair, mask=cell_mask(), extractor=extractor)
    assert results["masked"]["inception_l2"].value > 0
    assert results["unmasked"]["inception_l2"].value > 0
