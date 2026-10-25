from __future__ import annotations

import json
from dataclasses import asdict, dataclass
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from strenum import StrEnum  # type: ignore[no-redef]

from autoresearch.memory.schemas import TrialRecord
from autoresearch.memory.similarity import RepeatedBadStats, compute_repeated_bad_stats
from autoresearch.nodes.spec import NodeSpec


class MemoryMode(StrEnum):
    NONE = "none"
    APPEND_ONLY_SUMMARY = "append_only_summary"
    APPEND_ONLY_SUMMARY_WITH_RATIONALE = "append_only_summary_with_rationale"


# R1-SKIP-DISPLAY (PLAN §10, decided 25 Oct 2026): how a proposal skipped by
# the control plane at E1 is shown in the manager's memory.  ``None`` (the
# default) is the historical display (the generic record line); campaigns that
# declare no display had no control-plane skips.  ``one_line/v1`` shows one
# line with the fixed text below and no metric, delta, rationale or failure.
# The display changes the proposal law, so it is recorded in every manifest.
SKIP_DISPLAY_ONE_LINE = "one_line/v1"
SKIP_DISPLAYS: tuple[str, ...] = (SKIP_DISPLAY_ONE_LINE,)
SKIP_DISPLAY_TEXT = "skipped by the control plane: predicted no gain"


def check_skip_display(skip_display: str | None) -> str | None:
    if skip_display is not None and skip_display not in SKIP_DISPLAYS:
        raise ValueError(f"unknown skip_display {skip_display!r}; declared: {SKIP_DISPLAYS}")
    return skip_display


def is_control_plane_skip(record: TrialRecord) -> bool:
    """A trial the controller skipped at E1 (not a D25 inactive-key skip)."""
    control = (record.extra or {}).get("control") or {}
    return (
        record.decision.value == "skipped"
        and control.get("epoch") == "E1"
        and control.get("action") == "skip"
        and control.get("reason") != "inactive_key"
    )


def _skip_display_line(record: TrialRecord) -> str:
    """``one_line/v1``: the trial id, the proposal summary and the fixed text; no metric."""
    return f"{record.trial_id}: summary={record.proposal_summary}; {SKIP_DISPLAY_TEXT}"


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


def build_memory_context(
    records: list[TrialRecord],
    mode: str | MemoryMode,
    node_spec: NodeSpec,
    budget_index: int,
    rationale_max_tokens: int | None = None,
    skip_display: str | None = None,
) -> MemoryContext:
    """Build a ``MemoryContext`` for injection into the manager.

    Args:
        records: All prior ``TrialRecord`` objects for this campaign.
        mode: Memory mode — ``none``, ``append_only_summary``, or
            ``append_only_summary_with_rationale``.
        node_spec: The current node specification.
        budget_index: Current trial index (1-based).
        rationale_max_tokens: When set and mode is
            ``append_only_summary_with_rationale``, truncate each trial's
            rationale to this many whitespace-delimited tokens before
            injection.  ``None`` (default) means no truncation.
            Use this parameter to test the verbosity hypothesis: does shorter
            rationale improve or match summary-only performance?
        skip_display: ``None`` (default, historical) or ``"one_line/v1"``
            (R1-SKIP-DISPLAY): a control-plane E1 skip is shown as one line,
            ``"<trial_id>: summary=<summary>; skipped by the control plane:
            predicted no gain"``, with no metric.  Other records are unchanged.
    """
    memory_mode = MemoryMode(mode)
    check_skip_display(skip_display)

    def line_of(record: TrialRecord, **kwargs) -> str:
        if skip_display == SKIP_DISPLAY_ONE_LINE and is_control_plane_skip(record):
            return _skip_display_line(record)
        return _summary_line(record, **kwargs)
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
        context = "\n".join(
            line_of(record, include_rationale=False, metric_name=node_spec.metric_name)
            for record in records
        )
    else:
        lines = [
            line_of(
                record,
                include_rationale=True,
                rationale_max_tokens=rationale_max_tokens,
                metric_name=node_spec.metric_name,
            )
            for record in records
        ]
        if repeated.repeated_bad_count:
            lines.append(f"repeated_bad_warnings={','.join(repeated.flagged_trial_ids)}")
        lines.append(_best_strategy_line(records, node_spec.metric_name, node_spec.metric_direction))
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


def _truncate_tokens(text: str, max_tokens: int | None) -> str:
    """Truncate *text* to at most *max_tokens* whitespace-delimited tokens.

    Returns *text* unchanged when *max_tokens* is ``None`` or the text is
    already within the limit.
    """
    if max_tokens is None or not text:
        return text
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + " …"


def _summary_line(
    record: TrialRecord,
    include_rationale: bool,
    rationale_max_tokens: int | None = None,
    metric_name: str | None = None,
) -> str:
    # The line shows the node's declared metric (D24).  When a metric name is
    # given it is used strictly: a missing value shows "missing" rather than
    # leaking a different parsed metric (e.g. train_loss).  ``in`` is used
    # instead of ``or`` so that a metric of exactly 0.0 is shown, not skipped.
    # The old val_auc/first-parsed fallbacks remain only for callers that pass
    # no metric name.
    if metric_name is not None:
        metric = record.parsed_metrics.get(metric_name)
    elif "val_auc" in record.parsed_metrics:
        metric = record.parsed_metrics["val_auc"]
    else:
        metric = next(iter(record.parsed_metrics.values()), None)
    metric_text = f"{metric:.6f}" if metric is not None else "missing"
    line = (
        f"{record.trial_id}: decision={record.decision.value}; "
        f"summary={record.proposal_summary}; metric={metric_text}; "
        f"delta={record.delta_vs_best}"
    )
    if include_rationale:
        failure = record.failure_category.value if record.failure_category else "none"
        rationale = _truncate_tokens(record.decision_rationale or "", rationale_max_tokens)
        line += f"; rationale={rationale}; failure={failure}"
    return line


def _best_strategy_line(
    records: list[TrialRecord],
    metric_name: str,
    metric_direction: str = "maximize",
) -> str:
    best = None
    for record in records:
        if metric_name not in record.parsed_metrics:
            continue
        if best is None:
            best = record
        elif metric_direction == "minimize":
            if record.parsed_metrics[metric_name] < best.parsed_metrics[metric_name]:
                best = record
        elif record.parsed_metrics[metric_name] > best.parsed_metrics[metric_name]:
            best = record
    if best is None:
        return "best_strategy=none"
    return f"best_strategy={best.proposal_summary}; metric={best.parsed_metrics[metric_name]:.6f}"

