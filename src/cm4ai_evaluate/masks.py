"""Masked versions of each metric group.

A mask is a boolean array, True inside the cell. Three methods, chosen per
metric in scoring.py:

- PIXELS: only pixels inside the mask (the nine statistics, Pearson,
  Spearman, PSNR, mutual information). K becomes the mask's pixel count.
- GLCM: Haralick texture from co-occurrence pairs where both pixels are
  inside the mask. Exact: outside pixels become grey level 0, inside pixels
  keep their value shifted up by one (levels 1 to 256), and every pair
  involving level 0 is dropped. The shift leaves all six properties
  unchanged, because each depends only on level differences or on
  centred values, so a full mask gives exactly the unmasked result.
- BOUNDING_BOX: both images cropped to the smallest rectangle containing
  the mask (MS-SSIM, Gabor, Inception). These need a rectangular image, so
  background inside the rectangle is included.

Reading and checking the mask file (and skipping a mask that does not fit)
happens in io.read_mask.
"""

from __future__ import annotations

import numpy as np

from .extended import (
    HARALICK_ANGLES_DEGREES,
    HARALICK_PROPERTIES,
    MetricResult,
    _feature_similarity,
    _undefined_if_constant,
)

PIXELS = "pixels"
GLCM = "glcm"
BOUNDING_BOX = "bounding_box"
MASK_METHODS = (PIXELS, GLCM, BOUNDING_BOX)


def bounding_box(mask: np.ndarray) -> tuple[slice, slice]:
    """Row and column slices of the smallest rectangle containing the mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("mask has no foreground")
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows, cols = bounding_box(mask)
    return image[rows, cols]


def masked_haralick_features(
    image: np.ndarray, mask: np.ndarray, levels: int = 256
) -> dict[str, float] | None:
    """The six Haralick properties from pairs with both pixels inside the mask.

    image is on [0, 1] and quantized as in extended.haralick_features.
    Returns None if no neighbouring pair lies inside the mask.
    """
    from skimage.feature import graycomatrix, graycoprops

    quantized = (np.clip(np.asarray(image, dtype=np.float64), 0, 1) * (levels - 1)).astype(
        np.uint16
    )
    shifted = np.where(mask, quantized + 1, 0).astype(np.uint16)
    counts = graycomatrix(
        shifted,
        distances=[1],
        angles=[np.deg2rad(a) for a in HARALICK_ANGLES_DEGREES],
        levels=levels + 1,
        symmetric=True,
        normed=False,
    ).astype(np.float64)
    inside = counts[1:, 1:, :, :]  # drop every pair involving an outside pixel
    totals = inside.sum(axis=(0, 1), keepdims=True)
    if np.any(totals == 0):
        return None
    normed = inside / totals
    return {name: float(np.mean(graycoprops(normed, name))) for name in HARALICK_PROPERTIES}


def masked_haralick_similarity(
    prediction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray
) -> MetricResult:
    undefined = _undefined_if_constant(prediction[mask], ground_truth[mask])
    if undefined:
        return undefined
    truth = masked_haralick_features(ground_truth, mask)
    predicted = masked_haralick_features(prediction, mask)
    if truth is None or predicted is None:
        return MetricResult(None, "undefined: no neighbouring pixel pairs inside the mask")
    return MetricResult(_feature_similarity(truth, predicted))
