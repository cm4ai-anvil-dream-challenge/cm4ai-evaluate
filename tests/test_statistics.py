"""The nine statistics, checked against values worked out by hand."""

import math

import numpy as np
import pytest

from cm4ai_evaluate.statistics import STATISTIC_NAMES, compute_statistics


def test_returns_all_nine_and_pixel_count():
    result = compute_statistics(np.ones((2, 2)), np.ones((2, 2)))
    assert set(result) == set(STATISTIC_NAMES) | {"pixel_count"}


def test_hand_worked_example():
    # p = [0, 10, 20, 30], t = [0, 0, 20, 40]; differences 0, 10, 0, -10
    p = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    t = np.array([[0, 0], [20, 40]], dtype=np.uint8)
    r = compute_statistics(p, t)
    assert r["absolute_error_sum"] == 20
    assert r["squared_error_sum"] == 200
    assert r["dot_product_sum"] == 0 + 0 + 400 + 1200
    assert r["prediction_square_sum"] == 0 + 100 + 400 + 900
    assert r["ground_truth_square_sum"] == 0 + 0 + 400 + 1600
    assert r["maximum_absolute_error"] == 10
    assert r["mean_absolute_error"] == 5
    assert r["root_mean_squared_error"] == pytest.approx(math.sqrt(50))
    assert r["cosine_similarity"] == pytest.approx(1600 / math.sqrt(1400 * 2000))
    assert r["pixel_count"] == 4


def test_uint8_does_not_overflow():
    # 255 - 0 must be 255, and 255 * 255 must not wrap around
    p = np.full((512, 512), 255, dtype=np.uint8)
    t = np.zeros((512, 512), dtype=np.uint8)
    r = compute_statistics(p, t)
    assert r["maximum_absolute_error"] == 255
    assert r["prediction_square_sum"] == 255**2 * 512 * 512
    assert r["mean_absolute_error"] == 255


def test_identical_images():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (512, 512)).astype(np.float64)
    r = compute_statistics(image, image)
    assert r["absolute_error_sum"] == 0
    assert r["maximum_absolute_error"] == 0
    assert r["root_mean_squared_error"] == 0
    assert r["cosine_similarity"] == pytest.approx(1.0)


def test_cosine_ignores_brightness_scale():
    rng = np.random.default_rng(1)
    t = rng.integers(1, 100, (64, 64)).astype(np.float64)
    assert compute_statistics(2 * t, t)["cosine_similarity"] == pytest.approx(1.0)


def test_cosine_is_not_pearson():
    # A constant offset lowers cosine below 1; Pearson would stay at 1.
    t = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert compute_statistics(t + 100, t)["cosine_similarity"] < 0.95


@pytest.mark.parametrize("which", ["prediction", "ground_truth", "both"])
def test_all_black_image_scored_with_null_cosine(which):
    black = np.zeros((8, 8))
    other = np.full((8, 8), 7.0)
    p = black if which in ("prediction", "both") else other
    t = black if which in ("ground_truth", "both") else other
    r = compute_statistics(p, t)
    assert r["cosine_similarity"] is None
    assert math.isfinite(r["mean_absolute_error"])


def test_mask_uses_only_inside_pixels():
    p = np.array([[0, 100], [50, 50]], dtype=np.float64)
    t = np.array([[0, 0], [50, 60]], dtype=np.float64)
    mask = np.array([[True, False], [True, True]])
    r = compute_statistics(p, t, mask)
    # Inside pixels: (0, 0), (50, 50), (50, 60); the 100 vs 0 pixel is ignored
    assert r["pixel_count"] == 3
    assert r["absolute_error_sum"] == 10
    assert r["maximum_absolute_error"] == 10
    assert r["mean_absolute_error"] == pytest.approx(10 / 3)


def test_full_mask_equals_no_mask():
    rng = np.random.default_rng(2)
    p, t = rng.integers(0, 256, (2, 32, 32)).astype(np.float64)
    assert compute_statistics(p, t, np.ones((32, 32), bool)) == compute_statistics(p, t)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        compute_statistics(np.zeros((4, 4)), np.zeros((4, 5)))


def test_mask_shape_mismatch():
    with pytest.raises(ValueError):
        compute_statistics(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((2, 2), bool))


def test_empty_mask():
    with pytest.raises(ValueError):
        compute_statistics(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4), bool))
