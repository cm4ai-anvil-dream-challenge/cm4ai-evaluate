"""Metrics ported from the ProtiCelli quality evaluation notebook.

Source: CellProfiling/ProtiCelli, analysis/quality_evaluation/
evaluate_predictions.ipynb at commit 1e876ef (the notebook is unchanged
since it was added in 7ab763c). Each function below names the notebook
function it ports.

All functions take prediction and ground truth on the [0, 1] scale
(LoadedImage.as_unit()), which is what the notebook's data_range=1.0
assumes. The earlier evaluate.py passed 0 to 255 values with
data_range=1.0, which made PSNR and MS-SSIM wrong.

Changes from the notebook, all deliberate:

- Argument order is (prediction, ground_truth) throughout, matching
  statistics.py. Every metric here is symmetric, so values are unaffected.
- The notebook skips constant images (every pixel the same value). Here they
  are scored; metrics undefined for them return None and a note, via
  MetricResult.
- PSNR of identical images is infinite; it is returned as math.inf with a
  note, and the output layer decides how to write it.
- The device for Inception adds Apple MPS to the notebook's CUDA or CPU.
- Gabor responses are computed with Fourier transforms rather than direct
  convolution (faster; same values to floating point rounding).
- Inception runs images through the network in batches.
- Mutual information omits the notebook's per-image rescaling of values
  above 1, which cannot occur on [0, 1] inputs.

Pearson, Spearman and PSNR need only the core
dependencies. MS-SSIM, mutual information, Haralick and Gabor use the
[extended] extra (scikit-image, scikit-learn); Inception uses [inception]
(torch, torchvision).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats




@dataclass(frozen=True)
class MetricResult:
    """A metric value, or None with a note saying why it is undefined."""

    value: float | None
    note: str | None = None


def _constant(prediction: np.ndarray, ground_truth: np.ndarray) -> str | None:
    """Which input is constant, as words for a note, or None if neither is."""
    p_const = bool(np.all(prediction == prediction.flat[0]))
    t_const = bool(np.all(ground_truth == ground_truth.flat[0]))
    if p_const and t_const:
        return "both images are"
    if p_const:
        return "the prediction is"
    if t_const:
        return "the ground truth is"
    return None


def _undefined_if_constant(prediction, ground_truth) -> MetricResult | None:
    which = _constant(prediction, ground_truth)
    if which is None:
        return None
    return MetricResult(None, f"undefined: {which} constant (every pixel the same value)")


def _size_text(image: np.ndarray) -> str:
    return " x ".join(str(n) for n in np.shape(image))


def _as_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


# Pixel correlations (notebook: pixel_correlations)


def pearson_r(prediction: np.ndarray, ground_truth: np.ndarray) -> MetricResult:
    p, t = _as_float(prediction).ravel(), _as_float(ground_truth).ravel()
    undefined = _undefined_if_constant(p, t)
    if undefined:
        return undefined
    return MetricResult(float(stats.pearsonr(t, p)[0]))


def spearman_r(prediction: np.ndarray, ground_truth: np.ndarray) -> MetricResult:
    p, t = _as_float(prediction).ravel(), _as_float(ground_truth).ravel()
    undefined = _undefined_if_constant(p, t)
    if undefined:
        return undefined
    return MetricResult(float(stats.spearmanr(t, p).statistic))


# PSNR and MS-SSIM (notebook: compute_psnr, compute_ms_ssim)


def psnr_db(prediction: np.ndarray, ground_truth: np.ndarray, data_range: float = 1.0) -> MetricResult:
    p, t = _as_float(prediction), _as_float(ground_truth)
    mse = float(np.mean((t - p) ** 2))
    if mse == 0.0:
        return MetricResult(math.inf, "infinite: images are identical")
    return MetricResult(float(20 * np.log10(data_range) - 10 * np.log10(mse)))


MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ms_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    win_size: int = 11,
    data_range: float = 1.0,
) -> MetricResult:
    """Multi-scale SSIM, ported line for line from compute_ms_ssim.

    At each of up to five scales, SSIM is computed and the image pair is
    halved by 2 x 2 averaging. Contrast-structure terms from the coarser
    scales are combined with the final scale's SSIM using the standard
    weights. Defined for constant images (SSIM's stabilising constants keep
    it finite), so no None case.
    """
    from skimage.metrics import structural_similarity

    if min(np.shape(ground_truth)) < win_size:
        return MetricResult(
            None, f"undefined: image region {_size_text(ground_truth)} is smaller than the "
            f"{win_size} x {win_size} SSIM window"
        )
    weights = np.array(MS_SSIM_WEIGHTS)
    img1, img2 = _as_float(ground_truth), _as_float(prediction)
    mcs, mssim = [], []
    for scale in range(len(weights)):
        sim, cs_map = structural_similarity(
            img1, img2, win_size=win_size, data_range=data_range, full=True
        )
        mssim.append(sim)
        mcs.append(np.mean(cs_map))
        if scale < len(weights) - 1:
            if min(img1.shape) >= win_size * 2:
                h, w = img1.shape
                img1 = img1[: h // 2 * 2, : w // 2 * 2]
                img2 = img2[: h // 2 * 2, : w // 2 * 2]
                img1 = (img1[::2, ::2] + img1[1::2, ::2] + img1[::2, 1::2] + img1[1::2, 1::2]) / 4
                img2 = (img2[::2, ::2] + img2[1::2, ::2] + img2[::2, 1::2] + img2[1::2, 1::2]) / 4
            else:
                weights = weights[: scale + 1] / weights[: scale + 1].sum()
                break
    eps = 1e-8
    mcs = np.clip(mcs, eps, 1.0)
    mssim = np.clip(mssim, eps, 1.0)
    if len(mcs) > 1:
        value = np.prod(mcs[:-1] ** weights[:-1]) * mssim[-1] ** weights[-1]
    else:
        value = mssim[0]
    return MetricResult(float(np.clip(value, 0.0, 1.0)))


# Mutual information (notebook: compute_mutual_info)


def mutual_information_nats(
    prediction: np.ndarray, ground_truth: np.ndarray, bins: int = 256
) -> MetricResult:
    """Mutual information between binned intensities, in nats.

    A constant image shares no information with anything, so the notebook's
    value of 0.0 is kept rather than None.
    """
    from sklearn.metrics import mutual_info_score

    r, p = _as_float(ground_truth).ravel(), _as_float(prediction).ravel()
    if np.all(r == r[0]) or np.all(p == p[0]):
        return MetricResult(0.0, "zero: a constant image carries no information")
    # The notebook also rescaled each image by its own maximum when values
    # exceeded 1. Inputs here are always on [0, 1], so that step is omitted.
    edges = np.linspace(0, 1, bins)
    r_bin = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    p_bin = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    with warnings.catch_warnings():
        # scikit-learn warns when there are many distinct bins relative to the
        # number of pixels (small masks); the labels here are bins, not classes.
        warnings.simplefilter("ignore", UserWarning)
        value = mutual_info_score(r_bin, p_bin)
    return MetricResult(float(value))


# Texture (notebook: haralick_similarity, gabor_similarity)

HARALICK_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
HARALICK_ANGLES_DEGREES = (0, 45, 90, 135)


def haralick_features(image: np.ndarray, levels: int = 256) -> dict[str, float]:
    """Six GLCM properties, averaged over four directions at distance 1."""
    from skimage.feature import graycomatrix, graycoprops

    quantized = (np.clip(_as_float(image), 0, 1) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[np.deg2rad(a) for a in HARALICK_ANGLES_DEGREES],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return {name: float(np.mean(graycoprops(glcm, name))) for name in HARALICK_PROPERTIES}


def _feature_similarity(f1: dict[str, float], f2: dict[str, float]) -> float:
    """Mean over features of 1 - |a - b| / max(|a|, |b|)."""
    sims = []
    for key in f1:
        max_v = max(abs(f1[key]), abs(f2[key]), 1e-10)
        sims.append(1.0 - abs(f1[key] - f2[key]) / max_v)
    return float(np.mean(sims))


def haralick_similarity(prediction: np.ndarray, ground_truth: np.ndarray) -> MetricResult:
    undefined = _undefined_if_constant(_as_float(prediction), _as_float(ground_truth))
    if undefined:
        return undefined
    return MetricResult(
        _feature_similarity(haralick_features(ground_truth), haralick_features(prediction))
    )


GABOR_FREQUENCIES = (0.1, 0.3, 0.5)
GABOR_ANGLES_DEGREES = (0, 45, 90, 135)


GABOR_SIGMA = 2


@lru_cache(maxsize=None)
def _gabor_kernels() -> tuple[np.ndarray, ...]:
    """The 12 complex kernels skimage.filters.gabor builds for these settings."""
    from skimage.filters import gabor_kernel

    return tuple(
        gabor_kernel(
            frequency,
            theta=np.deg2rad(angle),
            sigma_x=GABOR_SIGMA,
            sigma_y=GABOR_SIGMA,
            dtype=np.complex128,
        )
        for frequency in GABOR_FREQUENCIES
        for angle in GABOR_ANGLES_DEGREES
    )


@lru_cache(maxsize=8)
def _gabor_kernel_spectra(padded_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Fourier transforms of the kernels, centred at the origin, for one image size."""
    from scipy import fft

    spectra = []
    for kernel in _gabor_kernels():
        ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
        placed = np.zeros(padded_shape, dtype=np.complex128)
        placed[: kernel.shape[0], : kernel.shape[1]] = kernel
        placed = np.roll(placed, (-ry, -rx), axis=(0, 1))
        spectra.append(fft.fft2(placed))
    return tuple(spectra)


def gabor_magnitudes(image: np.ndarray) -> list[np.ndarray]:
    """Gabor filter response magnitudes at 3 frequencies x 4 orientations.

    Equivalent to the notebook's calls to skimage.filters.gabor (direct
    convolution with mode "reflect"), computed with Fourier transforms for
    speed: the image is padded by symmetric reflection, which is the same
    boundary rule, transformed once, and multiplied by each kernel's
    transform. Results agree with the direct method to floating point
    rounding (see tests/test_extended.py).
    """
    from scipy import fft

    values = _as_float(image)
    pad = max(max(k.shape) for k in _gabor_kernels()) // 2
    padded = np.pad(values, pad, mode="symmetric")
    spectrum = fft.fft2(padded)
    height, width = values.shape
    magnitudes = []
    for kernel_spectrum in _gabor_kernel_spectra(padded.shape):
        response = fft.ifft2(spectrum * kernel_spectrum)[pad : pad + height, pad : pad + width]
        magnitudes.append(np.abs(response))
    return magnitudes


def gabor_similarity(prediction: np.ndarray, ground_truth: np.ndarray) -> MetricResult:
    """Mean Pearson correlation of the 12 Gabor magnitude maps."""
    undefined = _undefined_if_constant(_as_float(prediction), _as_float(ground_truth))
    if undefined:
        return undefined
    correlations = []
    for m1, m2 in zip(gabor_magnitudes(ground_truth), gabor_magnitudes(prediction)):
        f1, f2 = m1.ravel(), m2.ravel()
        f1 = (f1 - f1.mean()) / (f1.std() + 1e-10)
        f2 = (f2 - f2.mean()) / (f2.std() + 1e-10)
        c = np.corrcoef(f1, f2)[0, 1]
        correlations.append(c if not np.isnan(c) else 0.0)
    return MetricResult(float(np.mean(correlations)))


# Inception (notebook: InceptionFeatureExtractor, per-pair l2_fid_proxy)


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InceptionFeatureExtractor:
    """ImageNet InceptionV3 with the classifier removed: 2,048 features per image.

    Loading downloads the torchvision weights on first use. Create one
    extractor and reuse it for every image.
    """

    def __init__(self, device: str | None = None):
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = device or default_device()
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = nn.Identity()
        self.model = model.to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _prepare(self, image: np.ndarray):
        as_uint8 = (np.clip(_as_float(image), 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([as_uint8] * 3, axis=-1)  # grey replicated to three channels
        return self.preprocess(rgb)

    def extract_many(self, images: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """Features for several [0, 1] images, run through the network in batches.

        Returns an array of shape (number of images, 2048). In eval mode the
        network treats each image independently, so batching changes speed,
        not results (beyond GPU floating point rounding).
        """
        import torch

        features = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = torch.stack([self._prepare(im) for im in images[start : start + batch_size]])
                features.append(self.model(batch.to(self.device)).cpu().numpy())
        return np.concatenate(features, axis=0)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Features for one [0, 1] image, as in the notebook."""
        return self.extract_many([image])[0]


def inception_l2(
    prediction: np.ndarray, ground_truth: np.ndarray, extractor: InceptionFeatureExtractor
) -> MetricResult:
    """L2 distance between Inception features (the notebook's per-pair FID proxy).

    Lower is better. Not FID itself, which compares distributions across a
    whole set of images.
    """
    truth_features, prediction_features = extractor.extract_many([ground_truth, prediction])
    return MetricResult(float(np.linalg.norm(truth_features - prediction_features)))
