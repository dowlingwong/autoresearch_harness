"""Metric parser for the lr_synthetic node.

Reads ``val_score: 0.XXXXXX`` from the run log.
Returns a float in [0, 1] (validation AUC).
"""
from __future__ import annotations

import re
from pathlib import Path


_VAL_SCORE_RE = re.compile(
    r"^val_score:\s*([\d.]+(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def parse_val_score(log_path: str | Path) -> float | None:
    """Return the val_score from *log_path*, or None if not found."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    match = _VAL_SCORE_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
