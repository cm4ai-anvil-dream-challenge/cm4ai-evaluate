import pytest

from cm4ai_evaluate import cli, io, manifest, statistics


def test_modes():
    assert io.MODES == ("strict", "lenient")


def test_nine_statistics():
    assert len(statistics.STATISTIC_NAMES) == 9
    assert "ground_truth_square_sum" in statistics.STATISTIC_NAMES


def test_manifest_columns():
    assert manifest.REQUIRED_COLUMNS == ("prediction", "ground_truth")
    assert manifest.OPTIONAL_COLUMNS == ("prediction_id", "mask")


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "--mode" in capsys.readouterr().out
