"""Strict reader: the challenge image specification and organizer/images.py rules."""

import numpy as np
import pytest
import tifffile
from PIL import Image

from cm4ai_evaluate import io

SHAPE = io.IMAGE_SHAPE


def gradient(shape=SHAPE):
    return (np.arange(shape[0] * shape[1]) % 256).astype(np.uint8).reshape(shape)


def save_png(path, array, mode=None):
    Image.fromarray(array, mode=mode).save(path, format="PNG")
    return path


def mask_array(values=(0, 255)):
    mask = np.full(SHAPE, values[0], dtype=np.uint8)
    mask[100:300, 150:400] = values[1]
    return mask


# Accepted inputs


@pytest.mark.parametrize("role", [io.PREDICTION, io.GROUND_TRUTH])
def test_valid_png_is_read_unchanged(tmp_path, role):
    array = gradient()
    loaded = io.read_image(save_png(tmp_path / "a.png", array), role)
    assert loaded.pixels.dtype == np.uint8
    np.testing.assert_array_equal(loaded.pixels, array)
    assert loaded.spec_compliant
    assert loaded.role == role


def test_uppercase_suffix_is_accepted(tmp_path):
    io.read_image(save_png(tmp_path / "a.PNG", gradient()), io.PREDICTION)


def test_scales_are_exact(tmp_path):
    array = gradient()
    loaded = io.read_image(save_png(tmp_path / "a.png", array), io.PREDICTION)
    np.testing.assert_array_equal(loaded.as_0_255(), array.astype(np.float64))
    np.testing.assert_allclose(loaded.as_unit(), array / 255.0)
    assert loaded.as_unit().max() <= 1.0


def test_constant_image_is_accepted(tmp_path):
    io.read_image(save_png(tmp_path / "black.png", np.zeros(SHAPE, np.uint8)), io.PREDICTION)


# Rejected inputs: each check has its own code


def expect(code, path, role=io.PREDICTION):
    with pytest.raises(io.ImageCheckError) as info:
        io.read_image(path, role)
    assert info.value.code == code
    assert info.value.role == role
    return info.value


def test_missing(tmp_path):
    expect("missing", tmp_path / "nope.png")


def test_directory(tmp_path):
    (tmp_path / "dir.png").mkdir()
    expect("not_a_file", tmp_path / "dir.png")


def test_symlink(tmp_path):
    real = save_png(tmp_path / "real.png", gradient())
    (tmp_path / "link.png").symlink_to(real)
    expect("symlink", tmp_path / "link.png")


@pytest.mark.parametrize("role", [io.PREDICTION, io.GROUND_TRUTH])
def test_tiff_rejected_for_both_roles(tmp_path, role):
    tifffile.imwrite(tmp_path / "a.tiff", gradient())
    expect("not_png", tmp_path / "a.tiff", role)


def test_jpeg_named_png(tmp_path):
    Image.fromarray(gradient()).save(tmp_path / "fake.png", format="JPEG")
    expect("not_png", tmp_path / "fake.png")


@pytest.mark.parametrize(
    "make",
    [
        lambda: Image.fromarray(np.stack([gradient()] * 3, axis=-1)),  # RGB
        lambda: Image.fromarray(np.stack([gradient()] * 4, axis=-1)),  # RGBA
        lambda: Image.fromarray(np.stack([gradient()] * 2, axis=-1), mode="LA"),
        lambda: Image.fromarray(gradient()).convert("P"),  # palette
        lambda: Image.fromarray(gradient() > 127),  # 1-bit
        lambda: Image.fromarray(gradient().astype(np.uint16) * 257),  # 16-bit
    ],
    ids=["RGB", "RGBA", "LA", "palette", "1-bit", "16-bit"],
)
def test_non_L_modes(tmp_path, make):
    make().save(tmp_path / "a.png", format="PNG")
    error = expect("not_grayscale_uint8", tmp_path / "a.png")
    assert "mode" in str(error)


@pytest.mark.parametrize("shape", [(256, 256), (512, 511), (1024, 1024)])
def test_wrong_size(tmp_path, shape):
    expect("wrong_size", save_png(tmp_path / "a.png", gradient(shape)))


def test_truncated_png(tmp_path):
    good = save_png(tmp_path / "good.png", gradient()).read_bytes()
    (tmp_path / "bad.png").write_bytes(good[: len(good) // 2])
    expect("unreadable", tmp_path / "bad.png")


def test_unknown_role(tmp_path):
    with pytest.raises(ValueError):
        io.read_image(save_png(tmp_path / "a.png", gradient()), "target")


def test_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        io.read_image(save_png(tmp_path / "a.png", gradient()), io.PREDICTION, mode="loose")


# Pair check


def test_pair_same_shape(tmp_path):
    a = io.read_image(save_png(tmp_path / "a.png", gradient()), io.PREDICTION)
    b = io.read_image(save_png(tmp_path / "b.png", gradient()), io.GROUND_TRUTH)
    io.check_pair(a, b)


def test_pair_mismatch():
    a = io.LoadedImage(path=None, role=io.PREDICTION, pixels=np.zeros((4, 4), np.uint8))
    b = io.LoadedImage(path=None, role=io.GROUND_TRUTH, pixels=np.zeros((4, 5), np.uint8))
    with pytest.raises(io.ImageCheckError) as info:
        io.check_pair(a, b)
    assert info.value.code == "size_mismatch"


# Masks: never raise, skip with a reason


def test_mask_valid(tmp_path):
    result = io.read_mask(save_png(tmp_path / "m.png", mask_array()), SHAPE)
    assert result.skip_reason is None
    assert result.mask.dtype == bool
    assert result.mask.sum() == 200 * 250


def test_mask_all_foreground_is_valid(tmp_path):
    result = io.read_mask(save_png(tmp_path / "m.png", np.full(SHAPE, 255, np.uint8)), SHAPE)
    assert result.mask.all()


@pytest.mark.parametrize(
    "array, code",
    [
        (mask_array((0, 1)), "not_binary"),
        (gradient(), "not_binary"),
        (np.zeros(SHAPE, np.uint8), "no_foreground"),
        (mask_array()[:256, :256], "wrong_size"),
    ],
    ids=["0-1", "grey", "empty", "small"],
)
def test_mask_skipped(tmp_path, array, code):
    result = io.read_mask(save_png(tmp_path / "m.png", array), SHAPE)
    assert result.mask is None
    assert result.skip_reason.startswith(code)


def test_mask_missing_is_skipped_not_raised(tmp_path):
    result = io.read_mask(tmp_path / "nope.png", SHAPE)
    assert result.mask is None
    assert result.skip_reason.startswith("missing")


def test_mask_rgb_skipped(tmp_path):
    Image.fromarray(np.stack([mask_array()] * 3, axis=-1)).save(tmp_path / "m.png")
    assert io.read_mask(tmp_path / "m.png", SHAPE).skip_reason.startswith("not_grayscale_uint8")


def test_mask_shape_differs_from_images(tmp_path):
    result = io.read_mask(save_png(tmp_path / "m.png", mask_array()), (256, 256))
    assert result.skip_reason.startswith("size_mismatch")
