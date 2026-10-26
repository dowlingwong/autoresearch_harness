"""Parse the final Track G metric without accepting a stale earlier result."""
from __future__ import annotations

import math
from pathlib import Path


def parse_val_bpb(log_path: str | Path) -> float | None:
    lines = [line.partition(":")[2].strip() for line in
             Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("val_bpb:")]
    if not lines:
        return None
    try:
        value = float(lines[-1])
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None
