import pytest

from cm4ai_evaluate import panel, scoring
from cm4ai_evaluate.selection import SelectionError, resolve


def names(selection):
    return scoring.output_names(selection.pair) + [m.name for m in selection.panel]


def test_default_is_basic_and_extended():
    s = resolve(None)
    assert not s.explicit
    assert "pearson_r" in names(s) and "brightness_spread_w1" in names(s)
    assert "inception_l2" not in names(s) + s.missing.get("inception", [])
    assert not any(m.requires == scoring.SUBCELL for m in s.panel)


def test_basic_only():
    s = resolve("basic")
    assert s.explicit
    assert all(m.requires == scoring.CORE for m in s.pair)
    assert [m.name for m in s.panel] == ["brightness_spread_w1"]


def test_single_names_and_statistics_come_together():
    s = resolve("cosine_similarity,brightness_spread_w1")
    assert "mean_absolute_error" in names(s)  # the nine statistics share one entry
    assert "pearson_r" not in names(s)


def test_subcell_tier_lists_three():
    s = resolve("subcell")
    chosen = [m.name for m in s.panel] + s.missing.get("subcell", [])
    assert chosen == [m.name for m in panel.PANEL_METRICS if m.requires == scoring.SUBCELL]


def test_inception_flag_adds_inception():
    s = resolve("basic", include_inception=True)
    assert "inception_l2" in names(s) + s.missing.get("inception", [])


def test_unknown_name():
    with pytest.raises(SelectionError, match="unknown metric or tier 'nope'"):
        resolve("basic,nope")


def test_every_panel_metric_has_level_and_category():
    for m in panel.PANEL_METRICS:
        assert m.level in (panel.IMAGE, panel.CELL)
        assert m.category and m.panel_name
