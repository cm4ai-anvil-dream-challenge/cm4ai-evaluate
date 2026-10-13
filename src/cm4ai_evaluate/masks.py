"""Optional binary masks and the masked versions of each metric group.

- Nine statistics, Pearson, Spearman, PSNR, mutual information: only
  pixels inside the mask.
- Haralick: exact GLCM with outside-mask pixels as level 0 and cell pixels
  1 to 255, then all pairs involving level 0 dropped.
- MS-SSIM, Gabor, Inception: both images cropped to the mask's bounding box.

A mask that does not fit (wrong dimensions, not binary, no foreground)
never raises: masked results are skipped and the reason is recorded.
"""

from __future__ import annotations
