"""Manifest reading: columns, identifiers, paths and errors."""

import json
from pathlib import Path

import pytest

from cm4ai_evaluate.manifest import ManifestError, read_manifest


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_basic(tmp_path):
    m = read_manifest(write(tmp_path / "m.csv", "prediction_id,prediction,ground_truth,mask\n"
                                                "a,p/1.png,g/1.png,k/1.png\nb,p/2.png,g/2.png,\n"))
    assert [r.prediction_id for r in m.rows] == ["a", "b"]
    assert m.rows[0].prediction == tmp_path / "p/1.png"
    assert m.rows[0].mask == tmp_path / "k/1.png"
    assert m.rows[1].mask is None  # empty cell means no mask


def test_json_basic(tmp_path):
    rows = [{"prediction_id": 7, "prediction": "a.png", "ground_truth": "b.png", "mask": None}]
    m = read_manifest(write(tmp_path / "m.json", json.dumps(rows)))
    assert m.rows[0].prediction_id == "7" and m.rows[0].mask is None


def test_relative_absolute_and_home_paths(tmp_path):
    m = read_manifest(write(tmp_path / "m.csv", f"prediction,ground_truth\n~/x.png,{tmp_path}/y.png\n"))
    assert m.rows[0].prediction == Path.home() / "x.png"
    assert m.rows[0].ground_truth == tmp_path / "y.png"


def test_symlinks_not_resolved(tmp_path):
    (tmp_path / "real.png").write_bytes(b"")
    (tmp_path / "link.png").symlink_to(tmp_path / "real.png")
    m = read_manifest(write(tmp_path / "m.csv", "prediction,ground_truth\nlink.png,real.png\n"))
    assert m.rows[0].prediction.name == "link.png"


def test_row_numbers_when_no_prediction_id(tmp_path):
    m = read_manifest(write(tmp_path / "m.csv", "prediction,ground_truth\na,b\nc,d\n"))
    assert [r.prediction_id for r in m.rows] == ["1", "2"]


def test_extra_columns_ignored_and_reported(tmp_path):
    m = read_manifest(write(tmp_path / "m.csv", "prediction,ground_truth,source_db,nucleus_image\na,b,x,y\n"))
    assert m.ignored_columns == ("source_db", "nucleus_image")


def test_excel_byte_order_mark(tmp_path):
    path = tmp_path / "m.csv"
    path.write_bytes("﻿prediction,ground_truth\na,b\n".encode("utf-8"))
    assert read_manifest(path).rows[0].prediction.name == "a"


@pytest.mark.parametrize(
    "name, text, message",
    [
        ("m.csv", "prediction,truth\na,b\n", "missing required column(s): ground_truth"),
        ("m.csv", "prediction,ground_truth\n", "no rows"),
        ("m.csv", "prediction,ground_truth\na,\n", "row 1: ground_truth is empty"),
        ("m.csv", "prediction_id,prediction,ground_truth\nx,a,b\n,c,d\n", "row 2: prediction_id is empty"),
        ("m.csv", "prediction_id,prediction,ground_truth\nx,a,b\nx,c,d\n", "repeats row 1"),
        ("m.json", '{"prediction": "a"}', "list of objects"),
        ("m.json", "[{", "not valid JSON"),
        ("m.txt", "prediction,ground_truth\na,b\n", ".csv or .json"),
    ],
    ids=["missing-column", "no-rows", "empty-cell", "empty-id", "duplicate-id",
         "json-object", "bad-json", "extension"],
)
def test_errors(tmp_path, name, text, message):
    with pytest.raises(ManifestError) as info:
        read_manifest(write(tmp_path / name, text))
    assert message in str(info.value)


def test_missing_manifest(tmp_path):
    with pytest.raises(ManifestError):
        read_manifest(tmp_path / "nope.csv")


TEMPLATES = Path(__file__).resolve().parents[1] / "examples"


def test_templates_parse_and_agree():
    csv_rows = read_manifest(TEMPLATES / "manifest_template.csv").rows
    json_rows = read_manifest(TEMPLATES / "manifest_template.json").rows
    assert len(csv_rows) == 3
    for a, b in zip(csv_rows, json_rows):
        assert (a.prediction_id, a.prediction, a.ground_truth, a.mask) == (
            b.prediction_id, b.prediction, b.ground_truth, b.mask
        )
    assert csv_rows[2].mask is None
