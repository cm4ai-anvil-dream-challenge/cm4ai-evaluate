"""Lenient reader: each relaxation in the handoff's strict vs lenient table."""

import numpy as np
import pytest
import tifffile
from PIL import Image

from cm4ai_evaluate import io

SHAPE = io.IMAGE_SHAPE
LENIENT = io.LENIENT


def gradient(shape=SHAPE, dtype=np.uint8):
    return (np.arange(shape[0] * shape[1]) % 256).astype(dtype).reshape(shape)


def save_png(path, array, mode=None):
    Image.fromarray(array, mode=mode).save(path, format="PNG")
    return path


def read(path, role=io.PREDICTION):
    return io.read_image(path, role, mode=LENIENT)


def codes(loaded):
    return [d.code for d in loaded.deviations]


def warned(loaded):
    return [d.code for d in loaded.deviations if d.warn]


def expect_error(code, path):
    with pytest.raises(io.ImageCheckError) as info:
        read(path)
    assert info.value.code == code


# A spec-compliant image reads the same in both modes


@pytest.mark.parametrize("role", [io.PREDICTION, io.GROUND_TRUTH])
def test_compliant_image_has_no_deviations(tmp_path, role):
    path = save_png(tmp_path / "a.png", gradient())
    loaded = read(path, role)
    assert loaded.spec_compliant and loaded.deviations == []
    np.testing.assert_array_equal(loaded.pixels, io.read_image(path, role).pixels)


# Errors in both modes


def test_missing(tmp_path):
    expect_error("missing", tmp_path / "nope.png")


def test_truncated(tmp_path):
    good = save_png(tmp_path / "good.png", gradient()).read_bytes()
    (tmp_path / "bad.png").write_bytes(good[: len(good) // 2])
    expect_error("unreadable", tmp_path / "bad.png")


def test_pair_size_mismatch_still_errors(tmp_path):
    a = read(save_png(tmp_path / "a.png", gradient((256, 256))))
    b = read(save_png(tmp_path / "b.png", gradient()), io.GROUND_TRUTH)
    with pytest.raises(io.ImageCheckError) as info:
        io.check_pair(a, b)
    assert info.value.code == "size_mismatch"


# Symlinks


def test_symlink_allowed_as_deviation(tmp_path):
    real = save_png(tmp_path / "real.png", gradient())
    (tmp_path / "link.png").symlink_to(real)
    loaded = read(tmp_path / "link.png")
    assert codes(loaded) == ["symlink"]
    assert not loaded.spec_compliant


# Formats


def test_tiff_allowed_without_warning(tmp_path):
    tifffile.imwrite(tmp_path / "a.tiff", gradient())
    loaded = read(tmp_path / "a.tiff")
    assert codes(loaded) == ["tiff"] and warned(loaded) == []
    np.testing.assert_array_equal(loaded.pixels, gradient())


def test_tiff_content_with_png_name(tmp_path):
    tifffile.imwrite(tmp_path / "a.png", gradient())
    assert "tiff" in codes(read(tmp_path / "a.png"))


def test_jpeg_allowed_with_warning(tmp_path):
    Image.fromarray(gradient()).save(tmp_path / "a.jpg", format="JPEG")
    assert warned(read(tmp_path / "a.jpg")) == ["jpeg"]


def test_grey_rgb_jpeg_collapses(tmp_path):
    Image.fromarray(np.stack([gradient()] * 3, -1)).save(tmp_path / "a.jpg", quality=90)
    assert codes(read(tmp_path / "a.jpg")) == ["jpeg", "collapsed_channels"]


def test_other_format_rejected(tmp_path):
    Image.fromarray(gradient()).save(tmp_path / "a.bmp", format="BMP")
    expect_error("unsupported_format", tmp_path / "a.bmp")


# Channels


@pytest.mark.parametrize(
    "array, mode, label",
    [
        (np.stack([gradient()] * 3, -1), None, "RGB"),
        (np.dstack([gradient()] * 3 + [np.full(SHAPE, 255, np.uint8)]), None, "RGBA"),
        (np.dstack([gradient(), np.full(SHAPE, 255, np.uint8)]), "LA", "grayscale plus alpha"),
    ],
    ids=["RGB", "RGBA", "LA"],
)
def test_identical_planes_collapse_with_warning(tmp_path, array, mode, label):
    loaded = read(save_png(tmp_path / "a.png", array, mode))
    assert warned(loaded) == ["collapsed_channels"]
    assert label in str(loaded.deviations[0])
    np.testing.assert_array_equal(loaded.pixels, gradient())


def test_differing_planes_rejected(tmp_path):
    rgb = np.stack([gradient(), np.fliplr(gradient()), gradient()], -1)
    expect_error("not_single_channel", save_png(tmp_path / "a.png", rgb))


def test_transparent_alpha_rejected(tmp_path):
    rgba = np.dstack([gradient()] * 3 + [np.full(SHAPE, 128, np.uint8)])
    expect_error("transparent", save_png(tmp_path / "a.png", rgba))


def test_palette_expanded(tmp_path):
    Image.fromarray(gradient()).convert("P").save(tmp_path / "a.png")
    loaded = read(tmp_path / "a.png")
    assert codes(loaded) == ["palette", "collapsed_channels"]
    np.testing.assert_array_equal(loaded.pixels, gradient())


def test_one_bit_image_rejected(tmp_path):
    Image.fromarray(gradient() > 127).save(tmp_path / "a.png")
    expect_error("one_bit", tmp_path / "a.png")


def test_singleton_axes_squeezed(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", gradient()[None, :, :, None])
    loaded = read(tmp_path / "a.tif")
    assert "singleton_axes" in codes(loaded)
    assert loaded.pixels.shape == SHAPE


def test_channel_first_stack_rejected(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", np.stack([gradient()] * 3), photometric="minisblack")
    expect_error("not_single_channel", tmp_path / "a.tif")


# Size


def test_other_size_allowed_with_warning(tmp_path):
    loaded = read(save_png(tmp_path / "a.png", gradient((256, 300))))
    assert warned(loaded) == ["size"]
    assert loaded.pixels.shape == (256, 300)


# Bit depth and scaling


def test_sixteen_bit_scaled(tmp_path):
    values = gradient(dtype=np.uint16) * 257  # 0 to 65535
    save_png(tmp_path / "a.png", values)
    loaded = read(tmp_path / "a.png")
    assert warned(loaded) == ["bit_depth"]
    assert loaded.max_value == 65535
    np.testing.assert_allclose(loaded.as_unit(), gradient() / 255.0)
    np.testing.assert_allclose(loaded.as_0_255(), gradient().astype(float))


def test_twelve_bit_in_sixteen_warns_dim(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", (gradient(dtype=np.uint16) * 16))  # max 4080
    loaded = read(tmp_path / "a.tif")
    assert warned(loaded) == ["bit_depth", "fits_12_bit"]
    assert loaded.as_unit().max() < 0.07


def test_float_in_unit_range_accepted(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", (gradient() / 255.0).astype(np.float32))
    loaded = read(tmp_path / "a.tif")
    assert loaded.max_value == 1.0
    np.testing.assert_allclose(loaded.as_0_255(), gradient().astype(float), atol=1e-4)


def test_float_out_of_range_rejected(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", gradient().astype(np.float32))  # 0 to 255
    expect_error("float_out_of_range", tmp_path / "a.tif")


def test_float_nan_rejected(tmp_path):
    values = (gradient() / 255.0).astype(np.float32)
    values[0, 0] = np.nan
    tifffile.imwrite(tmp_path / "a.tif", values)
    expect_error("non_finite", tmp_path / "a.tif")


@pytest.mark.parametrize("dtype", [np.int16, bool])
def test_signed_and_boolean_rejected(tmp_path, dtype):
    tifffile.imwrite(tmp_path / "a.tif", gradient().astype(dtype))
    expect_error("unsupported_dtype", tmp_path / "a.tif")


# Masks in lenient mode


def mask_array(values=(0, 255), dtype=np.uint8):
    mask = np.full(SHAPE, values[0], dtype=dtype)
    mask[100:300, 150:400] = values[1]
    return mask


def read_mask(path):
    return io.read_mask(path, SHAPE, mode=LENIENT)


def test_mask_spec_values_no_deviation(tmp_path):
    result = read_mask(save_png(tmp_path / "m.png", mask_array()))
    assert result.mask.sum() == 200 * 250 and result.deviations == []


def test_mask_zero_one_accepted(tmp_path):
    result = read_mask(save_png(tmp_path / "m.png", mask_array((0, 1))))
    assert result.mask.sum() == 200 * 250
    assert [d.code for d in result.deviations] == ["mask_values"]


def test_mask_one_bit_png(tmp_path):
    Image.fromarray(mask_array() > 0).save(tmp_path / "m.png")
    result = read_mask(tmp_path / "m.png")
    assert result.mask.sum() == 200 * 250
    assert [d.code for d in result.deviations] == ["one_bit"]


def test_mask_sixteen_bit_max(tmp_path):
    tifffile.imwrite(tmp_path / "m.tif", mask_array((0, 65535), np.uint16))
    assert read_mask(tmp_path / "m.tif").mask.sum() == 200 * 250


def test_mask_boolean_tiff(tmp_path):
    tifffile.imwrite(tmp_path / "m.tif", mask_array((0, 1)).astype(bool))
    assert read_mask(tmp_path / "m.tif").mask.sum() == 200 * 250


@pytest.mark.parametrize(
    "array, code",
    [
        (mask_array((0, 7)), "not_binary"),
        (mask_array((1, 255)), "not_binary"),
        (gradient(), "not_binary"),
        (np.zeros(SHAPE, np.uint8), "no_foreground"),
    ],
    ids=["0-7", "1-255", "grey", "empty"],
)
def test_mask_skipped(tmp_path, array, code):
    result = read_mask(save_png(tmp_path / "m.png", array))
    assert result.mask is None and result.skip_reason.startswith(code)


def test_mask_other_size_skipped_not_raised(tmp_path):
    result = read_mask(save_png(tmp_path / "m.png", mask_array()[:256, :256]))
    assert result.skip_reason.startswith("size_mismatch")


def test_mask_symlink_recorded(tmp_path):
    real = save_png(tmp_path / "real.png", mask_array())
    (tmp_path / "link.png").symlink_to(real)
    result = read_mask(tmp_path / "link.png")
    assert result.mask is not None and [d.code for d in result.deviations] == ["symlink"]


def test_mask_missing_skipped(tmp_path):
    assert read_mask(tmp_path / "nope.png").skip_reason.startswith("missing")
