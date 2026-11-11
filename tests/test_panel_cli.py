"""End to end with the panel metrics that need no extra packages."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

from cm4ai_evaluate import cli
from cm4ai_evaluate.manifest import ManifestError, read_manifest


def image(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:512, 0:512]
    base = np.exp(-((x - 256) ** 2 + (y - 240) ** 2) / (10000.0 + 2000 * seed)) + 0.2 * rng.random((512, 512))
    return np.round(base / base.max() * 255).astype(np.uint8)


@pytest.fixture
def folder(tmp_path):
    mask = np.zeros((512, 512), np.uint8)
    mask[150:350, 120:400] = 255
    Image.fromarray(mask).save(tmp_path / "m.png")
    for i in range(6):
        Image.fromarray(image(i)).save(tmp_path / f"g{i}.png")
        Image.fromarray(image(i + 10)).save(tmp_path / f"p{i}.png")
    return tmp_path


def write(folder, header, rows):
    (folder / "m.csv").write_text("\n".join([header] + [",".join(r) for r in rows]) + "\n")
    return folder / "m.csv"


def table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_groups_cells_and_brightness(folder, capsys):
    # Two groups of three cells; cell c0 has two predictions (K = 2).
    rows = [("c0-v0", "p0.png", "g0.png", "m.png", "c0", "A"), ("c0-v1", "p1.png", "g0.png", "m.png", "c0", "A")]
    rows += [(f"c{i}-v0", f"p{i}.png", f"g{i}.png", "m.png", f"c{i}", "A" if i < 3 else "B") for i in range(1, 6)]
    manifest = write(folder, "prediction_id,prediction,ground_truth,mask,cell_id,group_id", rows)
    code = cli.main([str(manifest), "-o", str(folder / "out" / "run"), "--metrics", "basic", "--min-cells", "3"])
    assert code == 0
    images = table(folder / "out" / "run.csv")
    assert "masked_brightness_spread_w1" in images[0]
    assert "unmasked_brightness_spread_w1" not in images[0]
    assert all(0 < float(r["masked_brightness_spread_w1"]) < 1 for r in images)

    cells = {r["cell_id"]: r for r in table(folder / "out" / "run_cells.csv")}
    assert len(cells) == 6 and cells["c0"]["n_predictions"] == "2"
    v = [float(r["masked_brightness_spread_w1"]) for r in images if r["prediction_id"].startswith("c0-")]
    assert float(cells["c0"]["brightness_spread_w1"]) == pytest.approx(sum(v) / 2)

    groups = {r["group_id"]: r for r in table(folder / "out" / "run_groups.csv")}
    assert groups["A"]["n_cells"] == "3" and groups["A"]["enough_cells"] == "true"
    assert groups["B"]["n_cells"] == "3"
    err = capsys.readouterr().err
    assert "brightness_spread_w1: category Brightness spread, one value per image" in err


def test_lone_cell_in_group_has_no_reference(folder):
    rows = [("a", "p0.png", "g0.png", "m.png", "A"), ("b", "p1.png", "g1.png", "m.png", "B")]
    write(folder, "prediction_id,prediction,ground_truth,mask,group_id", rows)
    cli.main([str(folder / "m.csv"), "-o", str(folder / "out" / "run"), "--metrics", "basic"])
    images = table(folder / "out" / "run.csv")
    assert images[0]["masked_brightness_spread_w1"] == ""
    assert "no other cell of this group" in images[0]["notes"]


def test_without_group_brightness_is_skipped(folder, capsys):
    write(folder, "prediction_id,prediction,ground_truth,mask", [("a", "p0.png", "g0.png", "m.png")])
    assert cli.main([str(folder / "m.csv"), "-o", str(folder / "out" / "run"), "--metrics", "basic"]) == 0
    assert "Skipping brightness_spread_w1: needs a group_id column" in capsys.readouterr().err
    assert "masked_brightness_spread_w1" not in table(folder / "out" / "run.csv")[0]
    assert not (folder / "out" / "run_groups.csv").exists()
    assert (folder / "out" / "run_cells.csv").exists()


def test_each_row_is_a_cell_by_default(folder):
    rows = [("a", "p0.png", "g0.png", "m.png"), ("b", "p1.png", "g0.png", "")]
    m = read_manifest(write(folder, "prediction_id,prediction,ground_truth,mask", rows))
    assert [r.cell_id for r in m.rows] == ["a", "b"]


def test_reference_leaves_out_same_ground_truth(folder):
    # Without cell_id, two predictions of g0 are two "cells"; neither may use g0 as reference.
    rows = [("a", "p0.png", "g0.png", "m.png", "A"), ("b", "p1.png", "g0.png", "m.png", "A")]
    write(folder, "prediction_id,prediction,ground_truth,mask,group_id", rows)
    cli.main([str(folder / "m.csv"), "-o", str(folder / "out" / "run"), "--metrics", "basic"])
    assert all(r["masked_brightness_spread_w1"] == "" for r in table(folder / "out" / "run.csv"))


def test_cell_rows_must_agree(folder):
    rows = [("a", "p0.png", "g0.png", "m.png", "c0"), ("b", "p1.png", "g1.png", "m.png", "c0")]
    with pytest.raises(ManifestError, match="cell 'c0' has a different ground_truth"):
        read_manifest(write(folder, "prediction_id,prediction,ground_truth,mask,cell_id", rows))


def test_reference_columns_all_or_none(folder):
    with pytest.raises(ManifestError, match="give all three or none"):
        read_manifest(write(folder, "prediction,ground_truth,nucleus", [("p0.png", "g0.png", "g1.png")]))


def test_unknown_metric_exits_2(folder, capsys):
    write(folder, "prediction,ground_truth", [("p0.png", "g0.png")])
    assert cli.main([str(folder / "m.csv"), "-o", str(folder / "out" / "run"), "--metrics", "nope"]) == 2


def test_lenient_tiny_image_keeps_pair_scores_and_reports_subcell_skip(folder, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from cm4ai_evaluate import subcell

    class TinyExtractor(subcell.SubCellExtractor):
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.batch = 1
            self.model = subcell.SubCellViT(d=24, depth=1, heads=3, patch=16, img=32, pool_int=16).eval()

    monkeypatch.setattr(subcell, "SubCellExtractor", TinyExtractor)
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(folder / "tiny-p.png")
    Image.fromarray(np.flipud(np.arange(64, dtype=np.uint8).reshape(8, 8))).save(folder / "tiny-g.png")
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(folder / "tiny-m.png")
    manifest = write(
        folder,
        "prediction_id,prediction,ground_truth,mask",
        [("tiny", "tiny-p.png", "tiny-g.png", "tiny-m.png")],
    )

    assert cli.main(
        [str(manifest), "-o", str(folder / "out" / "tiny"), "--mode", "lenient", "--metrics", "basic,subcell"]
    ) == 0
    record = json.loads((folder / "out" / "tiny.jsonl").read_text())
    assert record["status"] == "scored"
    assert record["metrics"]["pearson_r"] is not None
    assert record["masked_metrics"]["subcell_pattern_distance"] is None
    assert "16" in record["notes"]["masked subcell_pattern_distance"]
    assert "16" in record["notes"]["cell subcell_energy_score"]
    assert table(folder / "out" / "tiny_cells.csv")[0]["subcell_energy_score"] == ""
    err = capsys.readouterr().err
    assert "SubCell" in err and "8 x 8" in err
