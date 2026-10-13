"""Manifest input: the only way to pair images.

CSV or JSON, chosen by file extension. Columns: sample_id, prediction,
ground_truth, and optional mask. Relative paths resolve against the
manifest's folder.
"""

from __future__ import annotations

REQUIRED_COLUMNS = ("sample_id", "prediction", "ground_truth")
OPTIONAL_COLUMNS = ("mask",)
