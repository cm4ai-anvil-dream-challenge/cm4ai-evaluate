"""Image readers and input checks.

Two modes:

- strict (default): the challenge's current image specification. Prediction
  and ground truth are PNG, 512 x 512, single channel (PIL mode "L"), uint8,
  finite, and not symlinks; a mask has the same form with values {0, 255}.
  Strict is a subset of the checks in cm4ai-anvil-dream-tools
  organizer/images.py (commit 07f4dec, read_image and read_prediction),
  which also accepts TIFF ground truth.
- lenient: accepts what strict rejects where the meaning is unambiguous,
  and records each relaxation as a Deviation. A result with any deviation
  has spec_compliant False. Deviations marked warn=True are also worth
  showing to the person running the tool.

Prediction and ground truth are treated identically in both modes. A
prediction or ground truth that fails a check raises ImageCheckError. A mask
that fails a check never raises: read_mask returns the reason instead, and
masked results are skipped.

Lenient rules (both images):

- symlinks allowed (deviation)
- TIFF allowed (deviation, no warning); JPEG allowed with a lossy-format
  warning; other formats rejected
- singleton axes, e.g. 1 x 512 x 512, squeezed (deviation)
- RGB or RGBA whose colour planes are identical collapsed to one channel
  (warning); alpha, if present, must be fully opaque. Grayscale plus alpha
  (PIL mode LA) is treated the same way. Palette PNGs are expanded through
  their colour table first. Anything else multi-channel is rejected.
- sizes other than 512 x 512 allowed (warning); prediction and ground truth
  must still match each other
- unsigned integers divided by their type's maximum; 16-bit images whose
  values never exceed 4,095 get an extra warning (likely 12-bit data, will
  appear dim); floats accepted only within [0, 1]; signed integers,
  booleans and other types rejected
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

TIFF_SUFFIXES = (".tif", ".tiff")
TWELVE_BIT_MAX = 4095


class ImageCheckError(ValueError):
    """An input image fails a check for the selected mode.

    code is a short, stable label for the failed check (for CSV output and
    tests); the message is for people.
    """

    def __init__(self, role: str, code: str, message: str):
        super().__init__(f"{role}: {message}")
        self.role = role
        self.code = code


@dataclass(frozen=True)
class Deviation:
    """One way a lenient input departs from the challenge specification."""

    code: str
    message: str
    warn: bool = False

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


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
    deviations: list[Deviation] = field(default_factory=list)

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
    deviations: list[Deviation] = field(default_factory=list)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _check_path(path: str | Path, role: str, mode: str) -> tuple[Path, list[Deviation]]:
    """Symlink, existence and regular-file checks shared by both modes."""
    source = Path(path).expanduser()
    deviations = []
    if source.is_symlink():
        if mode == STRICT:
            raise ImageCheckError(role, "symlink", f"must not be a symlink: {source}")
        deviations.append(Deviation("symlink", f"{source.name} is a symlink"))
    if not source.exists():
        raise ImageCheckError(role, "missing", f"file not found: {source}")
    if not source.is_file():
        raise ImageCheckError(role, "not_a_file", f"not a regular file: {source}")
    return source, deviations


# Strict


def _read_strict_png(path: str | Path, role: str) -> tuple[Path, np.ndarray]:
    """Strict checks for any role. Raises ImageCheckError."""
    source, _ = _check_path(path, role, STRICT)
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
        raise ImageCheckError(role, "wrong_size", f"must be 512 x 512, got {_size(pixels)}")
    if pixels.dtype != np.uint8:
        raise ImageCheckError(role, "not_uint8", f"must be uint8, got {pixels.dtype}")
    if not np.isfinite(pixels).all():
        raise ImageCheckError(role, "non_finite", f"contains NaN or infinity: {source.name}")
    return source, pixels


def _size(pixels: np.ndarray) -> str:
    return " x ".join(str(n) for n in pixels.shape)


# Lenient


def _decode_lenient(source: Path, role: str) -> tuple[np.ndarray, list[Deviation]]:
    """Decode any accepted format to an array; record format deviations."""
    deviations = []
    is_tiff_name = source.suffix.casefold() in TIFF_SUFFIXES
    try:
        if is_tiff_name:
            import tifffile

            pixels = np.asarray(tifffile.imread(source))
            deviations.append(Deviation("tiff", "TIFF rather than PNG"))
            return pixels, deviations
        with Image.open(source) as image:
            if image.format == "TIFF":
                image.close()
                import tifffile

                pixels = np.asarray(tifffile.imread(source))
                deviations.append(Deviation("tiff", "TIFF rather than PNG"))
                return pixels, deviations
            if image.format == "JPEG":
                deviations.append(
                    Deviation("jpeg", "JPEG is a lossy format; pixel values are approximate", True)
                )
            elif image.format != "PNG":
                raise ImageCheckError(
                    role,
                    "unsupported_format",
                    f"format {image.format} is not accepted (PNG, TIFF or JPEG): {source.name}",
                )
            if image.mode == "P":
                image = image.convert("RGBA")
                deviations.append(Deviation("palette", "palette PNG expanded through its colour table"))
            elif image.mode == "1":
                raise ImageCheckError(
                    role, "one_bit", f"1-bit image has no intensity range (is this a mask?): {source.name}"
                )
            pixels = np.asarray(image)
    except ImageCheckError:
        raise
    except Exception as error:
        raise ImageCheckError(role, "unreadable", f"could not read {source.name}: {error}")
    return pixels, deviations


def _to_single_channel(pixels: np.ndarray, role: str, name: str) -> tuple[np.ndarray, list[Deviation]]:
    """Squeeze singleton axes; collapse grey RGB, RGBA or LA; else reject."""
    deviations = []
    if pixels.ndim > 2 and 1 in pixels.shape:
        squeezed = np.squeeze(pixels)
        if squeezed.ndim >= 2:
            deviations.append(
                Deviation("singleton_axes", f"shape {_size(pixels)} squeezed to {_size(squeezed)}")
            )
            pixels = squeezed
    if pixels.ndim == 2:
        return pixels, deviations
    if pixels.ndim == 3 and pixels.shape[-1] in (2, 3, 4):
        channels = pixels.shape[-1]
        colour = pixels[..., :1] if channels == 2 else pixels[..., :3]
        if channels in (2, 4):
            alpha = pixels[..., -1]
            opaque = np.iinfo(pixels.dtype).max if pixels.dtype.kind == "u" else 1.0
            if not np.all(alpha == opaque):
                raise ImageCheckError(
                    role, "transparent", f"alpha channel is not fully opaque: {name}"
                )
        if not np.all(colour == colour[..., :1]):
            raise ImageCheckError(
                role,
                "not_single_channel",
                f"{channels} channels with differing colour planes; cannot reduce to one: {name}",
            )
        label = {2: "grayscale plus alpha", 3: "RGB", 4: "RGBA"}[channels]
        deviations.append(
            Deviation("collapsed_channels", f"{label} with identical colour planes reduced to one", True)
        )
        return colour[..., 0], deviations
    raise ImageCheckError(
        role, "not_single_channel", f"shape {_size(pixels)} is not a single-channel image: {name}"
    )


def _scale(pixels: np.ndarray, role: str, name: str) -> tuple[float, list[Deviation]]:
    """max_value for an accepted dtype, with deviations; rejects the rest."""
    kind = pixels.dtype.kind
    if kind == "u":
        max_value = float(np.iinfo(pixels.dtype).max)
        if pixels.dtype == np.uint8:
            return max_value, []
        deviations = [
            Deviation(
                "bit_depth",
                f"{pixels.dtype} scaled to [0, 1] by dividing by {int(max_value)}",
                True,
            )
        ]
        if pixels.dtype == np.uint16 and pixels.max() <= TWELVE_BIT_MAX:
            deviations.append(
                Deviation(
                    "fits_12_bit",
                    "values fit in 12 bits; scaled as 16-bit, so it will appear dim",
                    True,
                )
            )
        return max_value, deviations
    if kind == "f":
        if not np.isfinite(pixels).all():
            raise ImageCheckError(role, "non_finite", f"contains NaN or infinity: {name}")
        low, high = float(pixels.min()), float(pixels.max())
        if low < 0.0 or high > 1.0:
            raise ImageCheckError(
                role,
                "float_out_of_range",
                f"float values must lie within [0, 1], found {low:g} to {high:g}: {name}",
            )
        return 1.0, [Deviation("bit_depth", f"{pixels.dtype} values on [0, 1]", True)]
    raise ImageCheckError(
        role, "unsupported_dtype", f"{pixels.dtype} is not accepted (unsigned integer or float): {name}"
    )


def _read_lenient(path: str | Path, role: str) -> tuple[Path, np.ndarray, list[Deviation]]:
    """Path, format and channel handling for lenient mode (no scaling yet)."""
    source, deviations = _check_path(path, role, LENIENT)
    pixels, more = _decode_lenient(source, role)
    deviations += more
    pixels, more = _to_single_channel(pixels, role, source.name)
    deviations += more
    if pixels.shape != IMAGE_SHAPE:
        deviations.append(Deviation("size", f"{_size(pixels)} rather than 512 x 512", True))
    return source, pixels, deviations


# Public readers


def read_image(path: str | Path, role: str, mode: str = STRICT) -> LoadedImage:
    """Read and check a prediction or ground truth image."""
    if role not in (PREDICTION, GROUND_TRUTH):
        raise ValueError(f"role must be {PREDICTION!r} or {GROUND_TRUTH!r}, got {role!r}")
    _check_mode(mode)
    if mode == STRICT:
        source, pixels = _read_strict_png(path, role)
        return LoadedImage(path=source, role=role, pixels=pixels)
    source, pixels, deviations = _read_lenient(path, role)
    max_value, more = _scale(pixels, role, source.name)
    return LoadedImage(
        path=source, role=role, pixels=pixels, max_value=max_value, deviations=deviations + more
    )


def check_pair(prediction: LoadedImage, ground_truth: LoadedImage) -> None:
    """Prediction and ground truth must have identical pixel dimensions (both modes)."""
    if prediction.pixels.shape != ground_truth.pixels.shape:
        raise ImageCheckError(
            PREDICTION,
            "size_mismatch",
            f"dimensions {prediction.pixels.shape} differ from ground truth "
            f"{ground_truth.pixels.shape}",
        )


def _lenient_mask_values(pixels: np.ndarray) -> np.ndarray | None:
    """Foreground for an accepted binary encoding, or None if not binary.

    Accepted: booleans, or at most two values where one is 0 and the other
    is 1, 255 or the integer type's maximum.
    """
    if pixels.dtype == bool:
        return pixels
    values = np.unique(pixels)
    if values.size > 2:
        return None
    if values.size == 2 and values[0] != 0:
        return None
    top = values[-1]
    if top == 0:
        return np.zeros(pixels.shape, dtype=bool)
    allowed = {1, 255}
    if pixels.dtype.kind == "u":
        allowed.add(int(np.iinfo(pixels.dtype).max))
    if top not in allowed:
        return None
    return pixels == top


def _is_one_bit_png(source: Path) -> bool:
    try:
        with Image.open(source) as image:
            return image.format == "PNG" and image.mode == "1"
    except Exception:
        return False


def read_mask(path: str | Path, shape: tuple[int, int], mode: str = STRICT) -> MaskResult:
    """Read and check a mask against the image shape. Never raises for a bad mask.

    Strict: the same file checks as the images, values exactly {0, 255}.
    Lenient: the same relaxations as the images, plus 1-bit images and
    values {0, 1} or {0, type maximum}.
    """
    _check_mode(mode)
    source = Path(path).expanduser()
    deviations: list[Deviation] = []
    try:
        if mode == STRICT:
            source, pixels = _read_strict_png(path, MASK)
        else:
            source, deviations = _check_path(path, MASK, LENIENT)
            if _is_one_bit_png(source):
                with Image.open(source) as image:
                    pixels = np.asarray(image)
                deviations.append(Deviation("one_bit", "1-bit mask"))
            else:
                source, pixels, deviations = _read_lenient(path, MASK)
    except ImageCheckError as error:
        return MaskResult(path=source, mask=None, skip_reason=f"{error.code}: {error}")
    except Exception as error:
        return MaskResult(path=source, mask=None, skip_reason=f"unreadable: {error}")
    if pixels.shape != tuple(shape):
        return MaskResult(
            path=source,
            mask=None,
            skip_reason=f"size_mismatch: mask {pixels.shape} differs from images {tuple(shape)}",
            deviations=deviations,
        )
    if mode == STRICT:
        values = np.unique(pixels)
        mask = pixels == 255 if np.isin(values, STRICT_MASK_VALUES).all() else None
        expected = "0 and 255"
    else:
        mask = _lenient_mask_values(pixels)
        expected = "two values: 0 and one of 1, 255 or the type maximum"
        if (
            mask is not None
            and pixels.dtype != bool
            and not np.isin(np.unique(pixels), STRICT_MASK_VALUES).all()
        ):
            deviations.append(Deviation("mask_values", "mask values other than {0, 255}"))
    if mask is None:
        values = np.unique(pixels)
        shown = ", ".join(str(v) for v in values[:6]) + (", ..." if values.size > 6 else "")
        return MaskResult(
            path=source,
            mask=None,
            skip_reason=f"not_binary: mask values must be {expected}, found {shown}",
            deviations=deviations,
        )
    if not mask.any():
        return MaskResult(
            path=source, mask=None, skip_reason="no_foreground: mask is all 0", deviations=deviations
        )
    return MaskResult(path=source, mask=mask, deviations=deviations)
