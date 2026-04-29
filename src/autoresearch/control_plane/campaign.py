from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.control_plane.budget import BudgetState
from autoresearch.control_plane.decision import decide_trial
from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.baseline_manager import BaselineManager
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.provenance import build_trial_provenance
from autoresearch.memory.schemas import ExecutionStatus, FailureCategory, TrialDecision, TrialRecord, ValidityStatus
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import DryRunWorker, Worker, WorkerResult


class PendingTrialError(RuntimeError):
    """Raised when a campaign already has an active pending trial."""


@dataclass(frozen=True)
class CampaignRunResult:
    campaign_id: str
    records_path: str
    records_written: int
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _pending_guard_path(records_path: Path) -> Path:
    return records_path.parent / (records_path.stem + "_pending.json")


def _acquire_pending(records_path: Path, *, campaign_id: str, trial_id: str, budget_index: int) -> Path:
    guard = _pending_guard_path(records_path)
    if guard.exists():
        existing = json.loads(guard.read_text())
        raise PendingTrialError(
            f"Campaign already has a pending trial: {existing.get('trial_id')}. "
            "Use scripts/recover_pending.py to inspect, fail, or clear it: " + str(guard)
        )
    guard.write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "trial_id": trial_id,
                "budget_index": budget_index,
                "records_path": str(records_path),
                "started": _iso_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return guard


def _release_pending(guard: Path) -> None:
    if guard.exists():
        guard.unlink()


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def list_pending_guards(search_root: str | Path) -> list[Path]:
    """Return pending-trial guard files below a ledger directory or repo root."""
    root = Path(search_root)
    if root.is_file():
        return [root] if root.name.endswith("_pending.json") else []
    return sorted(root.rglob("*_pending.json")) if root.exists() else []


def inspect_pending_guard(guard_path: str | Path) -> dict[str, Any]:
    """Load one pending guard as a JSON object."""
    path = Path(guard_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PendingTrialError(f"pending guard is not a JSON object: {path}")
    payload["guard_path"] = str(path)
    return payload


def clear_pending_guard(guard_path: str | Path) -> Path:
    """Remove a pending guard without appending a failure record."""
    path = Path(guard_path)
    if path.exists():
        path.unlink()
    return path


def mark_pending_failed(
    *,
    guard_path: str | Path,
    node_spec: NodeSpec,
    manager_mode: str = "unknown_manager",
    worker_mode: str = "unknown_worker",
    memory_mode: str = "unknown",
    failure_message: str = "Recovered stale pending trial.",
) -> TrialRecord:
    """Append a failed-invalid TrialRecord for a stale pending guard, then clear it."""
    guard = inspect_pending_guard(guard_path)
    records_ref = str(guard.get("records_path") or "").strip()
    if not records_ref:
        raise PendingTrialError(f"pending guard does not include records_path: {guard_path}")
    records_path = Path(records_ref)

    budget_index = int(guard.get("budget_index", 1))
    campaign_id = str(guard.get("campaign_id") or str(guard.get("trial_id", "")).rsplit("-trial-", 1)[0])
    trial_id = str(guard.get("trial_id") or f"{campaign_id}-trial-{budget_index:03d}")
    now = _iso_now()
    record = TrialRecord(
        trial_id=trial_id,
        campaign_id=campaign_id,
        node_id=node_spec.name,
        budget_index=budget_index,
        timestamp_start=str(guard.get("started") or now),
        timestamp_end=now,
        manager_mode=manager_mode,
        worker_mode=worker_mode,
        memory_mode=memory_mode,
        proposal_summary="pending-trial-recovered-as-failed",
        proposal_rationale="",
        targeted_files=node_spec.editable_paths,
        patch_ref="",
        git_commit_before="",
        git_commit_after="",
        execution_status=ExecutionStatus.FAILED,
        validity_status=ValidityStatus.INVALID,
        failure_category=FailureCategory.RUNTIME_ERROR,
        raw_log_ref="",
        parsed_metrics={},
        current_best_before=None,
        delta_vs_best=None,
        decision=TrialDecision.FAILED_INVALID,
        decision_rationale=f"Pending guard recovered as failed: {failure_message}",
        wall_clock_seconds=0.0,
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
        extra={"pending_guard": guard, "recovery_failure_message": failure_message},
    )
    TrialAppendStore(records_path).append(record)
    clear_pending_guard(guard_path)
    return record


def run_real_campaign(
    *,
    node_spec: NodeSpec,
    campaign_id: str,
    budget: int,
    manager_mode: str,
    memory_mode: str,
    records_path: str | Path,
    worker: Worker,
    manager_llm: object | None = None,
) -> CampaignRunResult:
    """Run a fixed-budget campaign with a real worker.

    The control plane owns every state transition:
    - builds memory context from prior records before each trial
    - acquires a pending-trial guard before calling the worker
    - releases the guard on completion or failure
    - decides keep/discard based on parsed metric, never trusting the worker
    - appends one authoritative TrialRecord per trial
    """
    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    manager = _manager(manager_mode, llm=manager_llm)
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget)
    current_best = _current_best(existing_records, node_spec.metric_name)

    records: list[TrialRecord] = []
    while not budget_state.exhausted:
        budget_index = budget_state.next_budget_index
        trial_id = f"{campaign_id}-trial-{budget_index:03d}"

        memory_context = build_memory_context(
            existing_records + records, MemoryMode(memory_mode), node_spec, budget_index
        )
        status = ManagerStatus(
            campaign_id=campaign_id,
            budget_index=budget_index,
            current_best_metric=current_best,
            metric_name=node_spec.metric_name,
            metric_direction=node_spec.metric_direction,
        )
        proposal = manager.propose_next_trial(status, memory_context, node_spec)

        guard = _acquire_pending(
            records_path,
            campaign_id=campaign_id,
            trial_id=trial_id,
            budget_index=budget_index,
        )
        ts_start = _iso_now()
        try:
            worker_result = worker.run_trial(proposal, node_spec, budget_index)
        except Exception as exc:
            worker_result = WorkerResult(
                worker_mode=getattr(worker, "mode", "unknown_worker"),
                changed_files=(),
                success=False,
                parsed_metrics={},
                raw_log_ref="",
                patch_ref="",
                git_commit_before="",
                git_commit_after="",
                failure_message=str(exc),
            )
        finally:
            ts_end = _iso_now()
            _release_pending(guard)

        record = _record_from_worker_result(
            campaign_id=campaign_id,
            budget_index=budget_index,
            node_spec=node_spec,
            manager_mode=proposal.manager_mode,
            memory_mode=memory_mode,
            proposal_summary=proposal.proposal_summary,
            proposal_rationale=proposal.proposal_rationale,
            proposal_extra=proposal.extra,
            worker_result=worker_result,
            current_best=current_best,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
        )
        records.append(record)
        store.append_many([record])

        if record.decision == TrialDecision.KEPT and node_spec.metric_name in record.parsed_metrics:
            current_best = record.parsed_metrics[node_spec.metric_name]
        budget_state = budget_state.consume_one()

    return CampaignRunResult(
        campaign_id=campaign_id,
        records_path=str(records_path),
        records_written=len(records),
        dry_run=False,
    )


def run_dry_campaign(
    *,
    node_spec: NodeSpec,
    campaign_id: str,
    budget: int,
    manager_mode: str,
    memory_mode: str,
    records_path: str | Path,
    manager_llm: object | None = None,
) -> CampaignRunResult:
    manager = _manager(manager_mode, llm=manager_llm)
    worker = DryRunWorker()
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget)
    current_best = _current_best(existing_records, node_spec.metric_name)

    records: list[TrialRecord] = []
    while not budget_state.exhausted:
        budget_index = budget_state.next_budget_index
        memory_context = build_memory_context(existing_records + records, MemoryMode(memory_mode), node_spec, budget_index)
        status = ManagerStatus(
            campaign_id=campaign_id,
            budget_index=budget_index,
            current_best_metric=current_best,
            metric_name=node_spec.metric_name,
            metric_direction=node_spec.metric_direction,
        )
        proposal = manager.propose_next_trial(status, memory_context, node_spec)
        worker_result = worker.run_trial(proposal, node_spec, budget_index)
        record = _record_from_worker_result(
            campaign_id=campaign_id,
            budget_index=budget_index,
            node_spec=node_spec,
            manager_mode=proposal.manager_mode,
            memory_mode=memory_mode,
            proposal_summary=proposal.proposal_summary,
            proposal_rationale=proposal.proposal_rationale,
            proposal_extra=proposal.extra,
            worker_result=worker_result,
            current_best=current_best,
        )
        records.append(record)
        if record.decision == TrialDecision.KEPT and node_spec.metric_name in record.parsed_metrics:
            current_best = record.parsed_metrics[node_spec.metric_name]
        budget_state = budget_state.consume_one()

    store.append_many(records)
    return CampaignRunResult(campaign_id=campaign_id, records_path=str(records_path), records_written=len(records), dry_run=True)


def _record_from_worker_result(
    *,
    campaign_id: str,
    budget_index: int,
    node_spec: NodeSpec,
    manager_mode: str,
    memory_mode: str,
    proposal_summary: str,
    proposal_rationale: str,
    proposal_extra: dict[str, Any] | None = None,
    worker_result: WorkerResult,
    current_best: float | None,
    timestamp_start: str | None = None,
    timestamp_end: str | None = None,
) -> TrialRecord:
    scope = validate_edit_scope(worker_result.changed_files, node_spec)
    metric = worker_result.parsed_metrics.get(node_spec.metric_name)
    validity = ValidityStatus.VALID if worker_result.success and scope.valid and metric is not None else ValidityStatus.INVALID
    if not scope.valid:
        failure = FailureCategory.INVALID_EDIT_SCOPE
    elif not worker_result.success:
        failure = FailureCategory.RUNTIME_ERROR
    elif metric is None:
        failure = FailureCategory.METRIC_MISSING
    else:
        failure = None
    decision = decide_trial(
        validity_status=validity,
        candidate_metric=metric,
        current_best_metric=current_best,
        metric_direction=node_spec.metric_direction,
        failure_category=failure,
    )
    trial_id = f"{campaign_id}-trial-{budget_index:03d}"
    now = _iso_now()
    return TrialRecord(
        trial_id=trial_id,
        campaign_id=campaign_id,
        node_id=node_spec.name,
        budget_index=budget_index,
        timestamp_start=timestamp_start or now,
        timestamp_end=timestamp_end or now,
        manager_mode=manager_mode,
        worker_mode=worker_result.worker_mode,
        memory_mode=memory_mode,
        proposal_summary=proposal_summary,
        proposal_rationale=proposal_rationale,
        targeted_files=worker_result.changed_files or node_spec.editable_paths,
        patch_ref=worker_result.patch_ref,
        git_commit_before=worker_result.git_commit_before,
        git_commit_after=worker_result.git_commit_after,
        execution_status=ExecutionStatus.SUCCESS if worker_result.success else ExecutionStatus.FAILED,
        validity_status=validity,
        failure_category=decision.failure_category,
        raw_log_ref=worker_result.raw_log_ref,
        parsed_metrics=worker_result.parsed_metrics,
        current_best_before=current_best,
        delta_vs_best=decision.delta_vs_best,
        decision=decision.decision,
        decision_rationale=decision.rationale,
        wall_clock_seconds=1.0,
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
        extra={
            "manager": proposal_extra or {},
            "worker": worker_result.extra,
            "worker_failure_message": worker_result.failure_message,
            "scope_validation": scope.__dict__,
        },
    )


def _manager(manager_mode: str, llm: object | None = None):
    if manager_mode == "baseline_manager":
        return BaselineManager()
    if manager_mode == "prompt_manager":
        return PromptManager()
    if manager_mode == "langgraph_manager":
        from autoresearch.manager.langgraph_manager import LangGraphManager
        return LangGraphManager(llm=llm)
    raise ValueError(f"unknown manager mode: {manager_mode}")


def _current_best(records: list[TrialRecord], metric_name: str) -> float | None:
    values = [record.parsed_metrics[metric_name] for record in records if metric_name in record.parsed_metrics]
    return max(values) if values else None
