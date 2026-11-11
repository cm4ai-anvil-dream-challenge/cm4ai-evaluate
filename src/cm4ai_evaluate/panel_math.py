"""Math of the evaluation team's preliminary metric panel (C3_hist), numpy only.

Reimplemented, not imported, from the evaluation team's spec module
c3metrics.py (c3metrics-1.0-20261020), as copied into dream_eval 0.3.0:
cm4ai-anvil-dream-tools, branch refactor/shared-organizer-manifest-subsets,
commit 581edc9, file dream_eval-0.3.0/dream_eval/metrics_math.py. The
operations, constants and float types follow that file so the values agree;
tests/test_panel_math.py checks them.

Names here are descriptive, not the panel's (docs/metrics.md maps them), and
every value is oriented lower is better: the panel's location value is the
negative of location_divergence.

Nothing here weights or anchors the values: the challenge combines metrics
into one score with constants that are not public and not final.
"""

from __future__ import annotations

import numpy as np

HIST_BINS = 1020
SI_PERCENTILE = 99.0
SI_EPS = 1.0 / 255 / 4  # in-mask 99th percentile at or below this: no usable signal


def unit_float32(pixels: np.ndarray, max_value: float = 255.0) -> np.ndarray:
    """Pixels on the [0, 1] scale as float32, as the panel reads them (uint8 / 255)."""
    return np.asarray(pixels).astype(np.float32) / np.float32(max_value)


# Brightness spread (panel: hist_w1_si, category Brightness spread)


def _region_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(image, np.float32).ravel()[np.flatnonzero(np.asarray(mask, bool).ravel())]


def _normalise(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Divide by the in-mask 99th percentile and clip to [0, 1]; None if there is no signal."""
    values = _region_values(image, mask)
    if values.size == 0:
        return None
    scale = float(np.percentile(values, SI_PERCENTILE))
    if not np.isfinite(scale) or scale <= SI_EPS:
        return None
    return np.clip(np.asarray(image, np.float32) / np.float32(scale), 0.0, 1.0).astype(np.float32)


def _cdf(values: np.ndarray) -> np.ndarray:
    index = np.minimum((np.asarray(values, np.float64) * HIST_BINS).astype(np.int64), HIST_BINS - 1)
    counts = np.bincount(index, minlength=HIST_BINS).astype(np.float64)
    return np.cumsum(counts) / max(counts.sum(), 1)


def brightness_cdf(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Cumulative histogram of in-mask intensities after normalising, or None if no signal."""
    normalised = _normalise(image, mask)
    return None if normalised is None else _cdf(_region_values(normalised, mask))


def brightness_reference(cdfs: list[np.ndarray | None]) -> np.ndarray | None:
    """Mean of the given cells' histograms (the panel uses the protein's other cells)."""
    usable = [c for c in cdfs if c is not None]
    return np.mean(usable, axis=0) if usable else None


def brightness_spread_w1(image: np.ndarray, mask: np.ndarray, reference: np.ndarray | None) -> float:
    """Wasserstein-1 distance between the image's histogram and the reference; nan if undefined."""
    cdf = brightness_cdf(image, mask)
    if cdf is None or reference is None:
        return float("nan")
    return float(np.abs(cdf - reference).sum() / HIST_BINS)


# SubCell feature comparisons


def pattern_distance(prediction_features: np.ndarray, truth_features: np.ndarray) -> float:
    """Cosine distance of protein-only SubCell features (panel: emb_dist_subcell, Pattern similarity)."""
    f = np.asarray(prediction_features, np.float64)
    y = np.asarray(truth_features, np.float64)
    return float(1 - (f @ y) / (np.linalg.norm(f) * np.linalg.norm(y)))


def location_probabilities(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Independent per-class probabilities from a linear head on L2-normalised features."""
    x = np.atleast_2d(np.asarray(features, np.float64))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return 1 / (1 + np.exp(-(x @ np.asarray(weights, np.float64).T + np.asarray(bias, np.float64))))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (natural log) after normalising each to sum 1; 0 to ln 2."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float((a * np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def location_divergence(
    prediction_features: np.ndarray, truth_features: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> float:
    """Divergence of location-head probabilities, 4-channel features (panel: minus loc2_kolfhead_jsd)."""
    p = location_probabilities(prediction_features, weights, bias)[0]
    q = location_probabilities(truth_features, weights, bias)[0]
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return float("nan")
    return jensen_shannon(p, q)


def energy_score(prediction_features: np.ndarray, truth_features: np.ndarray) -> float:
    """Fair energy score of a cell's K predictions against the truth (panel: Variability/accuracy).

    Mean distance of each prediction to the truth, minus half the mean distance
    between pairs of predictions. With K = 1 it is the distance to the truth.
    """
    x = np.atleast_2d(np.asarray(prediction_features, np.float64))
    y = np.asarray(truth_features, np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    accuracy = np.linalg.norm(x - y, axis=1).mean()
    k = len(x)
    if k == 1:
        return float(accuracy)
    pairwise = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return float(accuracy - (pairwise.sum() - np.trace(pairwise)) / (2 * k * (k - 1)))
