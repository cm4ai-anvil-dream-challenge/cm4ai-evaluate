"""Manifest input: the only way to pair images.

CSV or JSON, chosen by file extension. Required columns: prediction and
ground_truth. Optional columns: prediction_id and mask. Relative paths resolve
against the manifest's folder.

prediction_id identifies one prediction (one row), matching the
blast-proticelli manifest and the participant contract. It is any non-empty
string and must be unique within the manifest; several predictions can come
from the same sample, or the same images can be scored more than once. When
the column is absent, each row's 1-based row number is used instead.
"""

from __future__ import annotations

REQUIRED_COLUMNS = ("prediction", "ground_truth")
OPTIONAL_COLUMNS = ("prediction_id", "mask")
