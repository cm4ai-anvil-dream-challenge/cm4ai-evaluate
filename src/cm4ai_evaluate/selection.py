"""Choosing which metrics to run: --metrics takes tiers and metric names.

Tiers group metrics by cost and what they need installed:
  basic      no extra packages; seconds per image (nine statistics, Pearson,
             Spearman, PSNR, brightness spread)
  extended   the [extended] extra (MS-SSIM, mutual information, Haralick, Gabor)
  subcell    the [subcell] extra plus a one-time 349 MB weights download; a
             GPU is much faster (pattern distance, location, energy score)
  inception  the [inception] extra (Inception L2)
  all        every tier
A metric name selects the registry entry that computes it (selecting one of
the nine statistics computes all nine, as they come from the same sums).
Default: basic and extended, whichever are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import panel, scoring

TIERS = {
    "basic": scoring.CORE,
    "extended": scoring.EXTENDED,
    "subcell": scoring.SUBCELL,
    "inception": scoring.INCEPTION,
}
DEFAULT = ("basic", "extended")


class SelectionError(ValueError):
    pass


@dataclass
class Selection:
    pair: tuple = ()
    panel: tuple = ()
    explicit: bool = False
    missing: dict = field(default_factory=dict)  # extra -> metric names left out


def known_names() -> list[str]:
    return list(TIERS) + ["all"] + scoring.output_names() + [m.name for m in panel.PANEL_METRICS]


def resolve(spec: str | None, include_inception: bool = False) -> Selection:
    tokens = [t.strip() for t in spec.split(",") if t.strip()] if spec else list(DEFAULT)
    if include_inception:
        tokens.append("inception")
    pair, chosen_panel = [], []
    for token in tokens:
        if token == "all":
            extras = set(TIERS.values())
        elif token in TIERS:
            extras = {TIERS[token]}
        else:
            extras = set()
        for metric in scoring.METRICS:
            if metric.requires in extras or token in metric.outputs:
                if metric not in pair:
                    pair.append(metric)
        for metric in panel.PANEL_METRICS:
            if metric.requires in extras or token == metric.name:
                if metric not in chosen_panel:
                    chosen_panel.append(metric)
        if not extras and not any(token in m.outputs for m in scoring.METRICS) and not any(
            token == m.name for m in panel.PANEL_METRICS
        ):
            raise SelectionError(f"unknown metric or tier {token!r}; choose from: {', '.join(known_names())}")
    selection = Selection(explicit=bool(spec))
    for metric in pair:
        if scoring.extra_installed(metric.requires):
            selection.pair += (metric,)
        else:
            selection.missing.setdefault(metric.requires, []).extend(metric.outputs)
    for metric in chosen_panel:
        if scoring.extra_installed(metric.requires):
            selection.panel += (metric,)
        else:
            selection.missing.setdefault(metric.requires, []).append(metric.name)
    # keep registry order, whatever order the tokens came in
    selection.pair = tuple(m for m in scoring.METRICS if m in selection.pair)
    selection.panel = tuple(m for m in panel.PANEL_METRICS if m in selection.panel)
    return selection
