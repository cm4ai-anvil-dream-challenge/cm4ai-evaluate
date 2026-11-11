"""The metric registry, and scoring one image pair with and without a mask.

Every metric the tool reports has one entry in METRICS: its output names,
the function that computes it, the optional extra it needs, the scale its
inputs use, and its masked method. Adding, removing or renaming a metric
means changing its entry here; nothing else in the tool lists metrics.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import extended as ex
from . import masks
from .extended import MetricResult
from .io import LoadedImage
from .statistics import STATISTIC_NAMES, compute_statistics

CORE = "core"
EXTENDED = "extended"
INCEPTION = "inception"
SUBCELL = "subcell"

UNIT = "unit"  # [0, 1]
RAW = "0_255"  # 0 to 255, as in the challenge grading


@dataclass(frozen=True)
class Metric:
    """One registry entry. compute(prediction, ground_truth) returns {output name: MetricResult}."""

    outputs: tuple[str, ...]
    compute: Callable[..., dict[str, MetricResult]]
    requires: str
    scale: str
    mask_method: str


def _single(name: str, function: Callable[..., MetricResult]):
    return lambda p, t, **_: {name: function(p, t)}


def _statistics(p, t, **_):
    values = compute_statistics(p, t)
    return {name: MetricResult(values[name]) for name in STATISTIC_NAMES}


def _haralick(p, t, mask=None, **_):
    if mask is None:
        return {"haralick_similarity": ex.haralick_similarity(p, t)}
    return {"haralick_similarity": masks.masked_haralick_similarity(p, t, mask)}


def _inception(p, t, extractor=None, **_):
    return {"inception_l2": ex.inception_l2(p, t, extractor)}


METRICS: tuple[Metric, ...] = (
    Metric(STATISTIC_NAMES, _statistics, CORE, RAW, masks.PIXELS),
    Metric(("pearson_r",), _single("pearson_r", ex.pearson_r), CORE, UNIT, masks.PIXELS),
    Metric(("spearman_r",), _single("spearman_r", ex.spearman_r), CORE, UNIT, masks.PIXELS),
    Metric(("psnr_db",), _single("psnr_db", ex.psnr_db), CORE, UNIT, masks.PIXELS),
    Metric(("ms_ssim",), _single("ms_ssim", ex.ms_ssim), EXTENDED, UNIT, masks.BOUNDING_BOX),
    Metric(
        ("mutual_information_nats",),
        _single("mutual_information_nats", ex.mutual_information_nats),
        EXTENDED,
        UNIT,
        masks.PIXELS,
    ),
    Metric(("haralick_similarity",), _haralick, EXTENDED, UNIT, masks.GLCM),
    Metric(
        ("gabor_similarity",),
        _single("gabor_similarity", ex.gabor_similarity),
        EXTENDED,
        UNIT,
        masks.BOUNDING_BOX,
    ),
    Metric(("inception_l2",), _inception, INCEPTION, UNIT, masks.BOUNDING_BOX),
)

_EXTRA_MODULES = {
    CORE: (),
    EXTENDED: ("skimage", "sklearn"),
    INCEPTION: ("torch", "torchvision"),
    SUBCELL: ("torch",),
}


def output_names(metrics: tuple[Metric, ...] = METRICS) -> list[str]:
    return [name for metric in metrics for name in metric.outputs]


def extra_installed(extra: str) -> bool:
    return all(importlib.util.find_spec(module) for module in _EXTRA_MODULES[extra])


def available_metrics(include_inception: bool = True) -> tuple[Metric, ...]:
    """Registry entries whose extra is installed (Inception only if requested)."""
    chosen = []
    for metric in METRICS:
        if metric.requires == INCEPTION and not include_inception:
            continue
        if extra_installed(metric.requires):
            chosen.append(metric)
    return tuple(chosen)


def _inputs(image: LoadedImage, scale: str) -> np.ndarray:
    return image.as_0_255() if scale == RAW else image.as_unit()


def _score(metric, p, t, mask, extractor) -> dict[str, MetricResult]:
    if mask is None:
        return metric.compute(p, t, extractor=extractor)
    if metric.mask_method == masks.PIXELS:
        return metric.compute(p[mask], t[mask], extractor=extractor)
    if metric.mask_method == masks.BOUNDING_BOX:
        return metric.compute(masks.crop(p, mask), masks.crop(t, mask), extractor=extractor)
    return metric.compute(p, t, mask=mask, extractor=extractor)  # GLCM handles the mask itself


def score_pair(
    prediction: LoadedImage,
    ground_truth: LoadedImage,
    mask: np.ndarray | None = None,
    metrics: tuple[Metric, ...] | None = None,
    extractor=None,
) -> dict[str, dict[str, MetricResult] | None]:
    """Every metric unmasked, and masked when a usable mask is given.

    Returns {"unmasked": {...}, "masked": {...} or None}. extractor is an
    extended.InceptionFeatureExtractor, required if Inception is among the
    metrics; reuse one across pairs.
    """
    if metrics is None:
        metrics = available_metrics(include_inception=extractor is not None)
    if any(m.requires == INCEPTION for m in metrics) and extractor is None:
        raise ValueError("Inception is selected but no extractor was given")
    results: dict[str, dict[str, MetricResult] | None] = {"unmasked": {}, "masked": None}
    if mask is not None:
        results["masked"] = {}
    for metric in metrics:
        p, t = _inputs(prediction, metric.scale), _inputs(ground_truth, metric.scale)
        results["unmasked"].update(_score(metric, p, t, None, extractor))
        if mask is not None:
            results["masked"].update(_score(metric, p, t, mask, extractor))
    return results


def mask_methods(metrics: tuple[Metric, ...] = METRICS) -> dict[str, str]:
    """Output name to masked method, for labelling the output."""
    return {name: metric.mask_method for metric in metrics for name in metric.outputs}
