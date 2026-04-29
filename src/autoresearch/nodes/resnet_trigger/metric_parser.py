from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class MetricParseError(ValueError):
    """Raised when a run log does not contain a valid ResNet-trigger metric."""


@dataclass(frozen=True)
class ParsedResNetMetrics:
    metric_name: str
    metric_value: float
    metric_direction: str
    source_log: str
    raw_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_val_auc(log_path: str | Path) -> ParsedResNetMetrics:
    """Parse the Stage 2 scientific metric from a ResNet-trigger run log.

    The Stage 1 harness uses lower-is-better `val_bpb = 1 - best_val_auc`.
    The Stage 2 paper-facing node contract reports higher-is-better `val_auc`.
    """

    path = Path(log_path)
    if not path.exists():
        raise MetricParseError(f"run log not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    raw = _extract_numeric_metrics(text)

    if "val_bpb" in raw:
        val_auc = 1.0 - raw["val_bpb"]
    elif "val_auc" in raw:
        val_auc = raw["val_auc"]
    else:
        raise MetricParseError("run log did not contain val_bpb or val_auc")

    if not math.isfinite(val_auc):
        raise MetricParseError("parsed val_auc is not finite")

    raw["val_auc"] = val_auc
    return ParsedResNetMetrics(
        metric_name="val_auc",
        metric_value=val_auc,
        metric_direction="maximize",
        source_log=str(path),
        raw_metrics=raw,
    )


def parse_val_auc_dict(log_path: str | Path) -> dict[str, Any]:
    return parse_val_auc(log_path).to_dict()


def _extract_numeric_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$", text, flags=re.MULTILINE):
        key, value = match.groups()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics

