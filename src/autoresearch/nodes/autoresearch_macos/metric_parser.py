"""Metric parser for the autoresearch-macos node (Karpathy nanochat GPT training).

The training script prints a final summary block after a ``---`` separator:

    ---
    val_bpb:          1.234567
    training_seconds: 300.1
    ...

``val_bpb`` (validation bits per byte) is the primary metric. Lower is better.
The parser reads the LAST ``val_bpb:`` occurrence so it picks up the final
evaluation, not any intermediate logging.

The training loop also emits ``\\r``-terminated progress lines; these do not
interfere with the regex because ``re.MULTILINE`` anchors ``^`` to newlines,
not carriage returns.
"""
from __future__ import annotations

import re
from pathlib import Path

_VAL_BPB_RE = re.compile(
    r"^val_bpb:\s*([\d.]+(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def parse_val_bpb(log_path: str | Path) -> float | None:
    """Return the final val_bpb from *log_path*, or None if not found.

    Scans the full log and returns the last match so that, even if an
    intermediate checkpoint prints ``val_bpb``, the final evaluation wins.
    A value is rejected if it is not finite or is outside the physically
    meaningful range (0, 20).
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    matches = _VAL_BPB_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if not (0.0 < value < 20.0):
        return None
    return value
