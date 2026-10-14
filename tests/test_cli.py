"""End to end: manifest in, JSONL and CSV out."""

import csv
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from cm4ai_evaluate import cli

pytest.importorskip("skimage")


def image(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:512, 0:512]
    base = np.exp(-((x - 256) ** 2 + (y - 240) ** 2) / 20000.0) + 0.2 * rng.random((512, 512))
    return np.round(base / base.max() * 255).astype(np.uint8)


@pytest.fixture
def folder(tmp_path):
    Image.fromarray(image(1)).save(tmp_path / "p1.png")
    Image.fromarray(image(2)).save(tmp_path / "g1.png")
    mask = np.zeros((512, 512), np.uint8)
    mask[150:350, 120:400] = 255
    Image.fromarray(mask).save(tmp_path / "m1.png")
    Image.fromarray(mask // 255).save(tmp_path / "m01.png")  # values 0 and 1
    tifffile.imwrite(tmp_path / "p2.tif", image(3))
    return tmp_path


def run(folder, rows, *extra):
    lines = ["prediction_id,prediction,ground_truth,mask"] + [",".join(r) for r in rows]
    (folder / "m.csv").write_text("\n".join(lines) + "\n")
    code = cli.main([str(folder / "m.csv"), "-o", str(folder / "out" / "run"), *extra])
    records = [json.loads(line) for line in (folder / "out" / "run.jsonl").read_text().splitlines()]
    with (folder / "out" / "run.csv").open() as handle:
        table = list(csv.DictReader(handle))
    return code, records, table


def test_strict_run(folder):
    code, records, table = run(
        folder,
        [
            ("with_mask", "p1.png", "g1.png", "m1.png"),
            ("no_mask", "p1.png", "g1.png", ""),
            ("identical", "g1.png", "g1.png", "m1.png"),
            ("missing", "nope.png", "g1.png", ""),
            ("tiff", "p2.tif", "g1.png", ""),
            ("mask_01", "p1.png", "g1.png", "m01.png"),
        ],
    )
    assert code == 1  # some rows could not be scored
    by_id = {r["prediction_id"]: r for r in records}
    assert [r["prediction_id"] for r in records] == [row["prediction_id"] for row in table]

    r = by_id["with_mask"]
    assert r["status"] == "scored" and r["spec_compliant"]
    assert r["mask_pixel_count"] == 200 * 280
    assert set(r["metrics"]) == set(r["masked_metrics"])
    assert r["masked_methods"]["haralick_similarity"] == "glcm"
    assert r["metrics"]["mean_absolute_error"] != r["masked_metrics"]["mean_absolute_error"]

    assert by_id["no_mask"]["masked_metrics"] is None
    assert by_id["identical"]["metrics"]["psnr_db"] is None
    assert "infinite" in by_id["identical"]["notes"]["unmasked psnr_db"]
    assert by_id["missing"]["status"] == "error" and by_id["missing"]["error"]["code"] == "missing"
    assert by_id["tiff"]["error"]["code"] == "not_png"
    assert by_id["mask_01"]["mask_skip_reason"].startswith("not_binary")
    assert by_id["mask_01"]["masked_metrics"] is None
    assert not by_id["mask_01"]["spec_compliant"]

    row = {r["prediction_id"]: r for r in table}["with_mask"]
    assert float(row["unmasked_pearson_r"]) == pytest.approx(by_id["with_mask"]["metrics"]["pearson_r"])
    assert row["masked_pearson_r"] != ""
    assert {r["prediction_id"]: r for r in table}["no_mask"]["masked_pearson_r"] == ""


def test_lenient_run(folder):
    code, records, table = run(
        folder,
        [("tiff", "p2.tif", "g1.png", "m01.png")],
        "--mode", "lenient",
    )
    assert code == 0
    r = records[0]
    assert r["status"] == "scored" and not r["spec_compliant"]
    assert {(d["role"], d["code"]) for d in r["deviations"]} == {
        ("prediction", "tiff"), ("mask", "mask_values")
    }
    assert r["masked_metrics"] is not None
    assert table[0]["deviations"] == "prediction tiff; mask mask_values"


def test_jsonl_is_strict_json(folder):
    _, records, _ = run(folder, [("identical", "g1.png", "g1.png", "")])
    text = (folder / "out" / "run.jsonl").read_text()
    assert "Infinity" not in text and "NaN" not in text


def test_manifest_error_exit_code(folder, capsys):
    (folder / "bad.csv").write_text("prediction\na.png\n")
    assert cli.main([str(folder / "bad.csv")]) == 2
    assert "ground_truth" in capsys.readouterr().err


def test_default_mode_is_strict():
    args = cli.build_parser().parse_args(["m.csv"])
    assert args.mode == "strict" and args.output_prefix == "results" and not args.inception
