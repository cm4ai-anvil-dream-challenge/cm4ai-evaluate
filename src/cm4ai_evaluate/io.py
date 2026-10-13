"""Image readers and input checks.

Two modes:

- strict (default): the challenge's current image specification. Prediction
  and ground truth are PNG, 512 x 512, single channel (PIL mode "L"), uint8,
  finite, and not symlinks; a mask has the same form with values {0, 255}.
  Strict is a subset of the checks in cm4ai-anvil-dream-tools
  organizer/images.py (commit 07f4dec, read_image and read_prediction),
  which also accepts TIFF ground truth.
- lenient: accepts TIFF and JPG, RGB or RGBA with identical colour planes,
  other sizes, and 16-bit or float data (scaled to [0, 1]). Each relaxation
  is recorded as a deviation and the result carries spec_compliant: false.
  Not implemented yet.

Prediction and ground truth are treated identically in both modes. A
prediction or ground truth that fails a check raises ImageCheckError. A mask
that fails a check never raises: read_mask returns the reason instead, and
masked results are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

STRICT = "strict"
LENIENT = "lenient"
MODES = (STRICT, LENIENT)

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

PREDICTION = "prediction"
GROUND_TRUTH = "ground_truth"
MASK = "mask"

# Values a strict mask may contain: background 0, foreground 255, matching
# the challenge's sample masks.
STRICT_MASK_VALUES = (0, 255)


class ImageCheckError(ValueError):
    """An input image fails a check for the selected mode.

    code is a short, stable label for the failed check (for CSV output and
    tests); the message is for people.
    """

    def __init__(self, role: str, code: str, message: str):
        super().__init__(f"{role}: {message}")
        self.role = role
        self.code = code


@dataclass
class LoadedImage:
    """A checked image as a 2D array, plus how it was read.

    pixels keeps the file's own values and dtype (uint8 in strict mode).
    max_value is the value that maps to 1.0 on the [0, 1] scale.
    """

    path: Path
    role: str
    pixels: np.ndarray
    max_value: float = 255.0
    deviations: list[str] = field(default_factory=list)

    @property
    def spec_compliant(self) -> bool:
        return not self.deviations

    def as_0_255(self) -> np.ndarray:
        """float64 values on the 0 to 255 scale (the nine statistics)."""
        values = self.pixels.astype(np.float64)
        if self.max_value != 255.0:
            values = values * (255.0 / self.max_value)
        return values

    def as_unit(self) -> np.ndarray:
        """float64 values on the [0, 1] scale (the notebook metrics)."""
        return self.pixels.astype(np.float64) / self.max_value


@dataclass
class MaskResult:
    """A boolean mask (True = inside the cell), or the reason it was skipped."""

    path: Path
    mask: np.ndarray | None
    skip_reason: str | None = None


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == LENIENT:
        raise NotImplementedError("lenient mode is not implemented yet")


def _read_strict_png(path: str | Path, role: str) -> tuple[Path, np.ndarray]:
    """Shared strict checks for any role. Raises ImageCheckError."""
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ImageCheckError(role, "symlink", f"must not be a symlink: {source}")
    if not source.exists():
        raise ImageCheckError(role, "missing", f"file not found: {source}")
    if not source.is_file():
        raise ImageCheckError(role, "not_a_file", f"not a regular file: {source}")
    if source.suffix.casefold() != ".png":
        raise ImageCheckError(role, "not_png", f"must be a .png file: {source.name}")
    try:
        with Image.open(source) as image:
            if image.format != "PNG":
                raise ImageCheckError(
                    role, "not_png", f"file content is {image.format}, not PNG: {source.name}"
                )
            if image.mode != "L":
                raise ImageCheckError(
                    role,
                    "not_grayscale_uint8",
                    f"PNG must be single-channel 8-bit grayscale (PIL mode L), "
                    f"got mode {image.mode}: {source.name}",
                )
            pixels = np.asarray(image)
    except ImageCheckError:
        raise
    except Exception as error:  # PIL raises several types for corrupt files
        raise ImageCheckError(role, "unreadable", f"could not read {source.name}: {error}")
    if pixels.shape != IMAGE_SHAPE:
        raise ImageCheckError(
            role,
            "wrong_size",
            f"must be {IMAGE_HEIGHT} x {IMAGE_WIDTH}, got "
            f"{' x '.join(str(n) for n in pixels.shape)}: {source.name}",
        )
    if pixels.dtype != np.uint8:
        raise ImageCheckError(role, "not_uint8", f"must be uint8, got {pixels.dtype}")
    if not np.isfinite(pixels).all():
        raise ImageCheckError(role, "non_finite", f"contains NaN or infinity: {source.name}")
    return source, pixels


def read_image(path: str | Path, role: str, mode: str = STRICT) -> LoadedImage:
    """Read and check a prediction or ground truth image."""
    if role not in (PREDICTION, GROUND_TRUTH):
        raise ValueError(f"role must be {PREDICTION!r} or {GROUND_TRUTH!r}, got {role!r}")
    _check_mode(mode)
    source, pixels = _read_strict_png(path, role)
    return LoadedImage(path=source, role=role, pixels=pixels)


def check_pair(prediction: LoadedImage, ground_truth: LoadedImage) -> None:
    """Prediction and ground truth must have identical pixel dimensions (both modes)."""
    if prediction.pixels.shape != ground_truth.pixels.shape:
        raise ImageCheckError(
            PREDICTION,
            "size_mismatch",
            f"dimensions {prediction.pixels.shape} differ from ground truth "
            f"{ground_truth.pixels.shape}",
        )


def read_mask(path: str | Path, shape: tuple[int, int], mode: str = STRICT) -> MaskResult:
    """Read and check a mask against the image shape. Never raises for a bad mask."""
    _check_mode(mode)
    source = Path(path).expanduser()
    try:
        source, pixels = _read_strict_png(path, MASK)
    except ImageCheckError as error:
        return MaskResult(path=source, mask=None, skip_reason=f"{error.code}: {error}")
    if pixels.shape != tuple(shape):
        return MaskResult(
            path=source,
            mask=None,
            skip_reason=f"size_mismatch: mask {pixels.shape} differs from images {tuple(shape)}",
        )
    values = np.unique(pixels)
    if not np.isin(values, STRICT_MASK_VALUES).all():
        shown = ", ".join(str(v) for v in values[:6]) + (", ..." if values.size > 6 else "")
        return MaskResult(
            path=source,
            mask=None,
            skip_reason=f"not_binary: mask values must be 0 and 255, found {shown}",
        )
    mask = pixels == 255
    if not mask.any():
        return MaskResult(path=source, mask=None, skip_reason="no_foreground: mask is all 0")
    return MaskResult(path=source, mask=mask)
