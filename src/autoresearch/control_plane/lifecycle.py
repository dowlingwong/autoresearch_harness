from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.control_plane.state_machine import TrialState, transition
from autoresearch.memory.provenance import build_trial_provenance, stable_id
from autoresearch.memory.schemas import (
    ExecutionStatus,
    FailureCategory,
    TrialDecision,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.nodes.spec import NodeSpec


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class TrialLifecycle:
    trial_id: str
    state: TrialState = TrialState.INITIALIZED
    history: list[TrialState] = field(default_factory=lambda: [TrialState.INITIALIZED])

    def advance(self, target: TrialState) -> TrialState:
        self.state = transition(self.state, target)
        self.history.append(self.state)
        return self.state


def build_trial_record_from_legacy_result(
    *,
    legacy_result: dict[str, Any],
    node_spec: NodeSpec,
    campaign_id: str,
    budget_index: int,
    manager_mode: str,
    worker_mode: str,
    memory_mode: str,
    timestamp_start: str | None = None,
    timestamp_end: str | None = None,
    wall_clock_seconds: float = 0.0,
) -> TrialRecord:
    """Convert one existing Stage 1 result payload into a Stage 2 trial record."""

    run_payload = dict(legacy_result.get("run", legacy_result))
    decision_payload = dict(legacy_result.get("decision", {}))
    packet = dict(run_payload.get("state", {}).get("pending_experiment", {}).get("packet", {}))
    if not packet:
        packet = dict(run_payload.get("packet", {}))

    trial_id = str(
        run_payload.get("trial_id")
        or stable_id("trial", campaign_id, budget_index, run_payload.get("commit", ""), packet.get("description", ""))
    )
    worker = dict(run_payload.get("worker", {}))
    worker_result = dict(worker.get("last_result", {})) if isinstance(worker.get("last_result"), dict) else {}
    changed_files = tuple(str(path) for path in worker_result.get("changed_files", packet.get("targeted_files", ["train.py"])))

    scope = validate_edit_scope(changed_files, node_spec)
    experiment = dict(run_payload.get("experiment") or decision_payload.get("experiment") or {})
    success = bool(experiment.get("success", False))
    val_bpb = experiment.get("val_bpb")
    parsed_metrics: dict[str, float] = {}
    if val_bpb is not None:
        parsed_metrics["val_auc"] = 1.0 - float(val_bpb)

    recommended_status = str(run_payload.get("recommended_status", "discard"))
    decision_text = str(decision_payload.get("decision", recommended_status))

    validity_status = ValidityStatus.VALID
    failure_category: FailureCategory | None = None
    if not scope.valid:
        validity_status = ValidityStatus.INVALID
        failure_category = FailureCategory.INVALID_EDIT_SCOPE
    elif not success:
        validity_status = ValidityStatus.INVALID
        failure_category = FailureCategory.RUNTIME_ERROR
    elif not parsed_metrics:
        validity_status = ValidityStatus.INVALID
        failure_category = FailureCategory.METRIC_MISSING

    if validity_status == ValidityStatus.INVALID:
        decision = TrialDecision.FAILED_INVALID
    elif decision_text in {"keep", "kept"}:
        decision = TrialDecision.KEPT
    else:
        decision = TrialDecision.DISCARDED

    current_best = _extract_best_before(run_payload)
    delta = None
    if current_best is not None and parsed_metrics:
        delta = parsed_metrics["val_auc"] - current_best

    return TrialRecord(
        trial_id=trial_id,
        campaign_id=campaign_id,
        node_id=node_spec.name,
        budget_index=budget_index,
        timestamp_start=timestamp_start or str(run_payload.get("created_at") or iso_now()),
        timestamp_end=timestamp_end or iso_now(),
        manager_mode=manager_mode,
        worker_mode=worker_mode,
        memory_mode=memory_mode,
        proposal_summary=str(packet.get("description") or run_payload.get("description") or "legacy-stage1-trial"),
        proposal_rationale=str(packet.get("objective") or ""),
        targeted_files=changed_files or ("train.py",),
        patch_ref=str(run_payload.get("patch_ref") or ""),
        git_commit_before=str(run_payload.get("base_commit") or ""),
        git_commit_after=str(run_payload.get("commit") or ""),
        execution_status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
        validity_status=validity_status,
        failure_category=failure_category,
        raw_log_ref=str(experiment.get("log_path") or run_payload.get("log_path") or ""),
        parsed_metrics=parsed_metrics,
        current_best_before=current_best,
        delta_vs_best=delta,
        decision=decision,
        decision_rationale=str(decision_payload.get("rationale") or decision_payload.get("decision") or recommended_status),
        wall_clock_seconds=wall_clock_seconds,
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
        extra={"legacy_result": run_payload},
    )


def build_trial_records_from_legacy_loop_result(
    *,
    legacy_loop_result: dict[str, Any],
    node_spec: NodeSpec,
    campaign_id: str,
    manager_mode: str,
    worker_mode: str,
    memory_mode: str,
    starting_budget_index: int = 1,
) -> list[TrialRecord]:
    history = legacy_loop_result.get("history", [])
    if not isinstance(history, list):
        history = []
    records: list[TrialRecord] = []
    for offset, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        records.append(
            build_trial_record_from_legacy_result(
                legacy_result=item,
                node_spec=node_spec,
                campaign_id=campaign_id,
                budget_index=starting_budget_index + offset,
                manager_mode=manager_mode,
                worker_mode=worker_mode,
                memory_mode=memory_mode,
            )
        )
    return records


def _extract_best_before(run_payload: dict[str, Any]) -> float | None:
    state = dict(run_payload.get("state", {}))
    best_bpb = state.get("best_bpb")
    if best_bpb is None:
        return None
    return 1.0 - float(best_bpb)
