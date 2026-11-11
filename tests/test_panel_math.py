"""Panel math: properties of each measure. Agreement with dream_eval itself was
checked outside the test suite (docs/metrics.md, Provenance), since its code
is not a dependency."""

import math

import numpy as np
import pytest

from cm4ai_evaluate import panel_math as pm


def blob(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    image = np.exp(-((x - 32) ** 2 + (y - 30) ** 2) / 200.0) + 0.1 * rng.random((64, 64))
    return (image / image.max() * scale).astype(np.float32)


MASK = np.zeros((64, 64), bool)
MASK[10:54, 8:56] = True


def test_unit_float32_matches_uint8_over_255():
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(pm.unit_float32(pixels), pixels.astype(np.float32) / 255.0)


def test_brightness_identical_histogram_is_zero():
    image = blob(1)
    reference = pm.brightness_reference([pm.brightness_cdf(image, MASK)])
    assert pm.brightness_spread_w1(image, MASK, reference) == 0.0


def test_brightness_ignores_overall_scale():
    # Each image is divided by its own in-mask 99th percentile, so a uniformly
    # dimmer copy has the same histogram shape (this measures spread, not level).
    reference = pm.brightness_reference([pm.brightness_cdf(blob(2), MASK)])
    a = pm.brightness_spread_w1(blob(1), MASK, reference)
    b = pm.brightness_spread_w1(blob(1, scale=0.5), MASK, reference)
    assert a == pytest.approx(b, abs=2e-3)


def test_brightness_no_signal_is_nan():
    reference = pm.brightness_reference([pm.brightness_cdf(blob(2), MASK)])
    dark = np.zeros((64, 64), np.float32)
    assert math.isnan(pm.brightness_spread_w1(dark, MASK, reference))
    assert pm.brightness_cdf(dark, MASK) is None


def test_brightness_without_reference_is_nan():
    assert math.isnan(pm.brightness_spread_w1(blob(1), MASK, None))
    assert pm.brightness_reference([None, None]) is None


def test_brightness_bounded():
    reference = pm.brightness_reference([pm.brightness_cdf(blob(s), MASK) for s in (2, 3)])
    value = pm.brightness_spread_w1(blob(1), MASK, reference)
    assert 0 <= value <= 1


def test_pattern_distance():
    f = np.random.default_rng(0).random(1536)
    assert pm.pattern_distance(f, f) == pytest.approx(0, abs=1e-12)
    assert pm.pattern_distance(f, -f) == pytest.approx(2)


def test_jensen_shannon_bounds():
    assert pm.jensen_shannon(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0)
    assert pm.jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(math.log(2), rel=1e-9)


def test_location_divergence_same_features_zero():
    rng = np.random.default_rng(1)
    f, w, b = rng.random(1536), rng.normal(size=(5, 1536)), rng.normal(size=5)
    assert pm.location_divergence(f, f, w, b) == pytest.approx(0, abs=1e-12)
    assert math.isnan(pm.location_divergence(np.full(1536, np.nan), f, w, b))


def test_energy_score():
    rng = np.random.default_rng(2)
    y = rng.random(8)
    x = rng.random((4, 8))
    acc = np.linalg.norm(x - y, axis=1).mean()
    spread = sum(np.linalg.norm(x[i] - x[j]) for i in range(4) for j in range(4) if i != j) / (2 * 4 * 3)
    assert pm.energy_score(x, y) == pytest.approx(acc - spread)
    assert pm.energy_score(x[:1], y) == pytest.approx(np.linalg.norm(x[0] - y))  # K = 1: accuracy only
    assert math.isnan(pm.energy_score(np.full((2, 8), np.nan), y))


def test_energy_score_rewards_spread_matching_uncertainty():
    # Identical samples get no credit for spread; samples spread around the
    # truth score better than the same number of samples all at one wrong point.
    y = np.zeros(2)
    collapsed = np.array([[1.0, 0.0]] * 4)
    spread = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert pm.energy_score(spread, y) < pm.energy_score(collapsed, y)
