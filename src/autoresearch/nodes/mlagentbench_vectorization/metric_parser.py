from __future__ import annotations

import re


_SPEED_RE = re.compile(
    r"^(?:SPEED_SCORE=|speed_score:\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def parse_speed_score(stdout: str) -> dict[str, float]:
    match = _SPEED_RE.search(stdout)
    if not match:
        return {}
    return {"speed_score": float(match.group(1))}
