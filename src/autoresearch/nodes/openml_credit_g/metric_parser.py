from __future__ import annotations

import re


_VAL_AUC_RE = re.compile(
    r"^(?:VAL_AUC=|val_auc:\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def parse_val_auc(stdout: str) -> dict[str, float]:
    match = _VAL_AUC_RE.search(stdout)
    if not match:
        return {}
    return {"val_auc": float(match.group(1))}


def parse_val_score(stdout: str) -> dict[str, float]:
    parsed = parse_val_auc(stdout)
    return {"val_score": parsed["val_auc"]} if parsed else {}
