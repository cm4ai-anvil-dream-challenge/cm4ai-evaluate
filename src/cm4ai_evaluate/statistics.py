"""The nine per-image statistics of the current challenge grading.

Reimplemented (not imported) from the approach in cm4ai-anvil-dream-tools:

- organizer/grading/comparison.py at commit 3a6ec06
  (compare_prediction_images: the six sums and the maximum per image pair)
- participant/benchmarking/metrics.py at participant commit f4c5661
  (ComparisonMetrics.from_sums: MAE, RMSE and cosine from the sums)

Computed on raw 0 to 255 values converted to float64, with K pixels per
image: K = 262,144 for a whole 512 x 512 image, or the number of pixels
inside the mask when a mask is given.
"""

from __future__ import annotations

import math

import numpy as np

# Field names follow the tools repo so outputs stay comparable, except that
# the real image is called ground_truth here (target in the tools repo).
STATISTIC_NAMES = (
    "absolute_error_sum",
    "squared_error_sum",
    "dot_product_sum",
    "prediction_square_sum",
    "ground_truth_square_sum",  # target_square_sum in the tools repo
    "maximum_absolute_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "cosine_similarity",
)


def compute_statistics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """The nine statistics for one image pair, plus the pixel count K.

    prediction and ground_truth are arrays of the same shape on the 0 to 255
    scale (LoadedImage.as_0_255()). mask, if given, is a boolean array of
    the same shape; only pixels where it is True are used.

    cosine_similarity is None when either image is all zero (inside the
    mask), because it is undefined there. It is not mean centred, so it is
    not Pearson's r.
    """
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(ground_truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and ground truth {t.shape} differ in shape")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != p.shape:
            raise ValueError(f"mask {mask.shape} differs from images {p.shape}")
        p = p[mask]
        t = t[mask]
    pixel_count = int(p.size)
    if pixel_count < 1:
        raise ValueError("no pixels to compare")

    # The six primitive values, as in comparison.py.
    difference = p - t
    absolute = np.abs(difference)
    absolute_error_sum = float(absolute.sum(dtype=np.float64))
    squared_error_sum = float(np.square(difference).sum(dtype=np.float64))
    dot_product_sum = float(np.multiply(p, t).sum(dtype=np.float64))
    prediction_square_sum = float(np.square(p).sum(dtype=np.float64))
    ground_truth_square_sum = float(np.square(t).sum(dtype=np.float64))
    maximum_absolute_error = float(absolute.max(initial=0.0))

    # Derived values, as in ComparisonMetrics.from_sums.
    denominator = math.sqrt(prediction_square_sum) * math.sqrt(ground_truth_square_sum)
    if denominator == 0.0:
        cosine = None
    else:
        cosine = max(-1.0, min(1.0, dot_product_sum / denominator))

    return {
        "absolute_error_sum": absolute_error_sum,
        "squared_error_sum": squared_error_sum,
        "dot_product_sum": dot_product_sum,
        "prediction_square_sum": prediction_square_sum,
        "ground_truth_square_sum": ground_truth_square_sum,
        "maximum_absolute_error": maximum_absolute_error,
        "mean_absolute_error": absolute_error_sum / pixel_count,
        "root_mean_squared_error": math.sqrt(squared_error_sum / pixel_count),
        "cosine_similarity": cosine,
        "pixel_count": pixel_count,
    }
