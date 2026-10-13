"""Image readers and input checks.

Two modes:

- strict (default): the challenge's current image specification. Prediction
  and ground truth are PNG, 512 x 512, single channel (PIL mode "L"), uint8,
  finite, and not symlinks; a mask has the same shape and is binary. Strict
  is a subset of the checks in cm4ai-anvil-dream-tools organizer/images.py.
- lenient: accepts TIFF and JPG, RGB or RGBA with identical colour planes,
  other sizes, and 16-bit or float data (scaled to [0, 1]). Each relaxation
  is recorded as a deviation and the result carries spec_compliant: false.

Prediction and ground truth are treated identically in both modes.
"""

from __future__ import annotations

STRICT = "strict"
LENIENT = "lenient"
MODES = (STRICT, LENIENT)

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
