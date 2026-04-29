from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from autoresearch.memory.schemas import TrialRecord
from autoresearch.memory.similarity import RepeatedBadStats, compute_repeated_bad_stats
from autoresearch.nodes.spec import NodeSpec


class MemoryMode(StrEnum):
    NONE = "none"
    APPEND_ONLY_SUMMARY = "append_only_summary"
    APPEND_ONLY_SUMMARY_WITH_RATIONALE = "append_only_summary_with_rationale"


@dataclass(frozen=True)
class MemoryContext:
    mode: MemoryMode
    context_text: str
    raw_memory_chars: int
    compressed_chars: int
    compression_ratio: float
    repeated_bad_stats: RepeatedBadStats

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["repeated_bad_stats"] = self.repeated_bad_stats.to_dict()
        return payload


def build_memory_context(records: list[TrialRecord], mode: str | MemoryMode, node_spec: NodeSpec, budget_index: int) -> MemoryContext:
    memory_mode = MemoryMode(mode)
    raw_text = "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records)
    repeated = compute_repeated_bad_stats(records)

    if memory_mode == MemoryMode.NONE:
        context = "\n".join(
            [
                f"node={node_spec.name}",
                f"budget_index={budget_index}",
                f"metric={node_spec.metric_name} direction={node_spec.metric_direction}",
                f"editable_paths={','.join(node_spec.editable_paths)}",
            ]
        )
    elif memory_mode == MemoryMode.APPEND_ONLY_SUMMARY:
        context = "\n".join(_summary_line(record, include_rationale=False) for record in records)
    else:
        lines = [_summary_line(record, include_rationale=True) for record in records]
        if repeated.repeated_bad_count:
            lines.append(f"repeated_bad_warnings={','.join(repeated.flagged_trial_ids)}")
        lines.append(_best_strategy_line(records, node_spec.metric_name))
        context = "\n".join(line for line in lines if line)

    compressed = len(context)
    raw = len(raw_text)
    return MemoryContext(
        mode=memory_mode,
        context_text=context,
        raw_memory_chars=raw,
        compressed_chars=compressed,
        compression_ratio=(compressed / raw) if raw else 0.0,
        repeated_bad_stats=repeated,
    )


def _summary_line(record: TrialRecord, include_rationale: bool) -> str:
    metric = record.parsed_metrics.get("val_auc")
    metric_text = f"{metric:.6f}" if metric is not None else "missing"
    line = (
        f"{record.trial_id}: decision={record.decision.value}; "
        f"summary={record.proposal_summary}; metric={metric_text}; "
        f"delta={record.delta_vs_best}"
    )
    if include_rationale:
        failure = record.failure_category.value if record.failure_category else "none"
        line += f"; rationale={record.decision_rationale}; failure={failure}"
    return line


def _best_strategy_line(records: list[TrialRecord], metric_name: str) -> str:
    best = None
    for record in records:
        if metric_name not in record.parsed_metrics:
            continue
        if best is None or record.parsed_metrics[metric_name] > best.parsed_metrics[metric_name]:
            best = record
    if best is None:
        return "best_strategy=none"
    return f"best_strategy={best.proposal_summary}; metric={best.parsed_metrics[metric_name]:.6f}"

