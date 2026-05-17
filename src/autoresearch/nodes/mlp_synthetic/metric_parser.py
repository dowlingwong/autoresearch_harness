from __future__ import annotations

import re


_VAL_SCORE_RE = re.compile(
    r"^val_score:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def parse_val_score(stdout: str) -> dict[str, float]:
    """Parse the validation score emitted by nodes/mlp_synthetic/train.py."""
    match = _VAL_SCORE_RE.search(stdout)
    if not match:
        return {}
    return {"val_score": float(match.group(1))}
