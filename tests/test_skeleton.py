import pytest

from cm4ai_evaluate import cli, io, manifest, statistics


def test_modes():
    assert io.MODES == ("strict", "lenient")


def test_nine_statistics():
    assert len(statistics.STATISTIC_NAMES) == 9


def test_manifest_columns():
    assert manifest.REQUIRED_COLUMNS == ("sample_id", "prediction", "ground_truth")


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "--mode" in capsys.readouterr().out


def test_cli_default_mode_is_strict():
    args = cli.build_parser().parse_args(["m.csv"])
    assert args.mode == "strict"
    assert args.output_prefix == "results"
