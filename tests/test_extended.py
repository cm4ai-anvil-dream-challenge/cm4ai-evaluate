"""Notebook metrics: known values, constant images and identical images."""

import math

import numpy as np
import pytest

from cm4ai_evaluate import extended as ex

pytest.importorskip("skimage")
pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def cell_like():
    """A smooth blob image on [0, 1] with texture, plus a noisy copy."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:128, 0:128]
    blob = np.exp(-((x - 64) ** 2 + (y - 60) ** 2) / 800.0)
    texture = 0.2 * (np.sin(x / 3.0) * np.cos(y / 5.0) + 1) / 2
    image = np.clip(0.7 * blob + texture, 0, 1)
    image = np.round(image * 255) / 255
    noisy = np.clip(image + rng.normal(0, 0.1, image.shape), 0, 1)
    return image, noisy


CONSTANT = np.full((128, 128), 0.5)


# Identical images give the best possible value


def test_identical_images(cell_like):
    image, _ = cell_like
    assert ex.pearson_r(image, image).value == pytest.approx(1.0)
    assert ex.spearman_r(image, image).value == pytest.approx(1.0)
    psnr = ex.psnr_db(image, image)
    assert psnr.value == math.inf and "identical" in psnr.note
    assert ex.ms_ssim(image, image).value == pytest.approx(1.0)
    assert ex.haralick_similarity(image, image).value == pytest.approx(1.0)
    assert ex.gabor_similarity(image, image).value == pytest.approx(1.0)


def test_noise_lowers_every_similarity(cell_like):
    image, noisy = cell_like
    assert ex.pearson_r(noisy, image).value < 0.99
    assert ex.ms_ssim(noisy, image).value < 0.99
    assert ex.haralick_similarity(noisy, image).value < 0.99
    assert ex.gabor_similarity(noisy, image).value < 0.99
    assert ex.mutual_information_nats(noisy, image).value < ex.mutual_information_nats(
        image, image
    ).value


# Known values


def test_psnr_known_value():
    # Every pixel off by 0.1 on [0, 1]: MSE 0.01, so PSNR = 20 dB
    t = np.full((8, 8), 0.5)
    assert ex.psnr_db(t + 0.1, t).value == pytest.approx(20.0)


def test_psnr_data_range_matters():
    # The earlier evaluate.py bug: 0 to 255 values with data_range=1.0
    t = np.full((8, 8), 0.5)
    on_unit = ex.psnr_db(t + 0.1, t).value
    wrong = ex.psnr_db((t + 0.1) * 255, t * 255, data_range=1.0).value
    right = ex.psnr_db((t + 0.1) * 255, t * 255, data_range=255.0).value
    assert right == pytest.approx(on_unit)
    assert wrong == pytest.approx(on_unit - 20 * math.log10(255))


def test_spearman_is_rank_based():
    t = np.linspace(0.01, 1, 64).reshape(8, 8)
    assert ex.spearman_r(t**3, t).value == pytest.approx(1.0)
    assert ex.pearson_r(t**3, t).value < 0.95


def test_symmetric(cell_like):
    image, noisy = cell_like
    for metric in (ex.pearson_r, ex.spearman_r, ex.psnr_db, ex.ms_ssim,
                   ex.mutual_information_nats, ex.haralick_similarity, ex.gabor_similarity):
        assert metric(noisy, image).value == pytest.approx(metric(image, noisy).value)


# Constant images are scored, not skipped


@pytest.mark.parametrize(
    "metric",
    [ex.pearson_r, ex.spearman_r, ex.haralick_similarity, ex.gabor_similarity],
)
@pytest.mark.parametrize("which", ["prediction", "ground_truth"])
def test_constant_gives_none_with_note(cell_like, metric, which):
    image, _ = cell_like
    p, t = (CONSTANT, image) if which == "prediction" else (image, CONSTANT)
    result = metric(p, t)
    assert result.value is None
    assert which.replace("_", " ") in result.note


def test_constant_both_note(cell_like):
    assert "both" in ex.pearson_r(CONSTANT, CONSTANT).note


def test_constant_finite_metrics(cell_like):
    image, _ = cell_like
    assert math.isfinite(ex.psnr_db(CONSTANT, image).value)
    assert 0 <= ex.ms_ssim(CONSTANT, image).value <= 1
    mi = ex.mutual_information_nats(CONSTANT, image)
    assert mi.value == 0.0 and mi.note


def test_all_black_prediction_is_scored(cell_like):
    image, _ = cell_like
    black = np.zeros_like(image)
    assert ex.pearson_r(black, image).value is None
    assert math.isfinite(ex.psnr_db(black, image).value)


def test_ms_ssim_on_full_size_image_uses_five_scales():
    # 512 / 2**4 = 32 >= 22, so no early stop; just check it runs and is in range
    rng = np.random.default_rng(3)
    a = rng.random((512, 512))
    value = ex.ms_ssim(a, np.clip(a + 0.05, 0, 1)).value
    assert 0 < value <= 1


# Inception: needs torch and a one-time weights download


def test_inception_identical_and_different(cell_like):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image, noisy = cell_like
    extractor = ex.InceptionFeatureExtractor()
    assert ex.inception_l2(image, image, extractor).value == pytest.approx(0.0, abs=1e-3)
    assert ex.inception_l2(noisy, image, extractor).value > 1.0


# Speed-ups match the notebook's direct methods


def test_gabor_fft_matches_skimage_direct(cell_like):
    from skimage import filters

    image, _ = cell_like
    rng = np.random.default_rng(4)
    for img in (image, rng.random((512, 512)), rng.random((37, 61))):
        fast = ex.gabor_magnitudes(img)
        k = 0
        for frequency in ex.GABOR_FREQUENCIES:
            for angle in ex.GABOR_ANGLES_DEGREES:
                real, imag = filters.gabor(img, frequency=frequency, theta=np.deg2rad(angle),
                                           sigma_x=2, sigma_y=2)
                np.testing.assert_allclose(fast[k], np.sqrt(real**2 + imag**2), rtol=0, atol=1e-10)
                k += 1


def test_inception_batch_matches_single(cell_like):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image, noisy = cell_like
    extractor = ex.InceptionFeatureExtractor()
    batched = extractor.extract_many([image, noisy, image[::-1]], batch_size=2)
    for i, single in enumerate([image, noisy, image[::-1]]):
        np.testing.assert_allclose(batched[i], extractor.extract(single), rtol=1e-4, atol=1e-4)
