"""Metrics ported from the ProtiCelli quality evaluation notebook.

Source: CellProfiling/ProtiCelli, analysis/quality_evaluation/
evaluate_predictions.ipynb (commit 1e876ef). Pearson, Spearman, PSNR,
MS-SSIM, mutual information, Haralick, Gabor and Inception L2.

These work on images converted to [0, 1], and PSNR and MS-SSIM are called
with data_range=1.0 to match (the fix for the mismatch in the earlier
evaluate.py). Metrics undefined for constant images return None with a note.
Heavy dependencies are optional extras: [extended] and [inception].
"""

from __future__ import annotations
