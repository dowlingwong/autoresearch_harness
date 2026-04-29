from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from autoresearch.memory.schemas import TrialRecord
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.spec import NodeSpec

STAGE2_MEMORY_MODES = (
    MemoryMode.NONE,
    MemoryMode.APPEND_ONLY_SUMMARY,
    MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE,
)


@dataclass(frozen=True)
class MemoryAblationRow:
    node_id: str
    memory_mode: str
    budget: int
    planned_trials: int
    raw_memory_chars: int
    manager_context_chars: int
    compression_ratio: float
    repeated_bad_count: int
    repeated_bad_rate: float
    repeated_invalid_count: int
    repeated_degraded_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_memory_ablation_plan(node_spec: NodeSpec, records: list[TrialRecord], budget: int) -> list[MemoryAblationRow]:
    rows: list[MemoryAblationRow] = []
    for mode in STAGE2_MEMORY_MODES:
        context = build_memory_context(records, mode, node_spec=node_spec, budget_index=1)
        stats = context.repeated_bad_stats
        rows.append(
            MemoryAblationRow(
                node_id=node_spec.name,
                memory_mode=mode.value,
                budget=budget,
                planned_trials=budget,
                raw_memory_chars=context.raw_memory_chars,
                manager_context_chars=context.compressed_chars,
                compression_ratio=context.compression_ratio,
                repeated_bad_count=stats.repeated_bad_count,
                repeated_bad_rate=stats.repeated_bad_rate,
                repeated_invalid_count=stats.repeated_invalid_count,
                repeated_degraded_count=stats.repeated_degraded_count,
            )
        )
    return rows


def export_memory_ablation_summary(rows: list[MemoryAblationRow], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "node_id",
        "memory_mode",
        "budget",
        "planned_trials",
        "raw_memory_chars",
        "manager_context_chars",
        "compression_ratio",
        "repeated_bad_count",
        "repeated_bad_rate",
        "repeated_invalid_count",
        "repeated_degraded_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return path

