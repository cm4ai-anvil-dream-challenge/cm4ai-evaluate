"""The nine per-image statistics of the current challenge grading.

Reimplemented (not imported) from the approach in cm4ai-anvil-dream-tools:
organizer/grading/comparison.py (sums per image pair) and
participant/benchmarking/metrics.py (ComparisonMetrics.from_sums).
Exact source commits are cited beside each function as it is written.

Computed on raw 0 to 255 values converted to float64, with K pixels per
image (K = 262,144 unmasked; the mask's pixel count when masked).
"""

from __future__ import annotations

# Field names follow the tools repo so outputs stay comparable.
STATISTIC_NAMES = (
    "absolute_error_sum",
    "squared_error_sum",
    "dot_product_sum",
    "prediction_square_sum",
    "target_square_sum",  # open: rename to ground_truth_square_sum?
    "maximum_absolute_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "cosine_similarity",
)
