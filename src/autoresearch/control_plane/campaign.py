from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path_str: str) -> str | None:
    """Return the hex SHA-256 digest of the file at *path_str*, or None if unavailable.

    Returns None for blank paths and for paths that do not exist on disk
    (e.g. dry-run synthetic refs).
    """
    import hashlib

    if not path_str or not path_str.strip():
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _patch_is_empty(patch_ref: str) -> bool:
    """Return True if *patch_ref* refers to a patch file that contains no diff content.

    A patch is considered empty (no-op) when:
    - the patch_ref string is blank (worker returned no path), or
    - the file exists on disk and contains only whitespace.

    Returns False for paths that do not exist on disk (e.g. dry-run synthetic refs),
    so this check never triggers spuriously in dry-run campaigns.
    """
    if not patch_ref or not patch_ref.strip():
        return True
    path = Path(patch_ref)
    if not path.exists():
        return False  # synthetic / dry-run path — not a real empty patch
    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        return True
    for line in content.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            return False
    return True

from autoresearch.control_plane.budget import BudgetState
from autoresearch.control_plane.decision import decide_trial
from autoresearch.control_plane.events import emit
from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.manager.baseline_manager import BaselineManager
from autoresearch.manager.hyperparam_edits import (
    build_effective_config,
    constants_after_edit,
    parse_train_constants,
    values_equal,
)
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.provenance import build_trial_provenance
from autoresearch.memory.research_context import load_node_research_context_ref
from autoresearch.memory.schemas import ExecutionStatus, FailureCategory, TrialDecision, TrialRecord, ValidityStatus
from autoresearch.memory.similarity import compute_repeated_bad_stats
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


@dataclass(frozen=True)
class _EditableStateSnapshot:
    root: Path
    contents: dict[str, bytes | None]


def _pending_guard_path(records_path: Path) -> Path:
    stem = records_path.stem
    campaign_id = stem[:-len("_trials")] if stem.endswith("_trials") else stem
    return records_path.parent / f"{campaign_id}_pending.json"


def _acquire_pending(records_path: Path, *, campaign_id: str, trial_id: str, budget_index: int) -> Path:
    guard = _pending_guard_path(records_path)
    if guard.exists():
        raw = guard.read_text(encoding="utf-8").strip()
        if not raw:
            # Empty guard left by a previous release that could only truncate
            # (e.g. read-only filesystem mount). Treat as absent.
            pass
        else:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError:
                existing = {"trial_id": "unknown (corrupt guard)"}
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
    if not guard.exists():
        return
    try:
        guard.unlink()
    except (PermissionError, OSError):
        # Fallback for read-only or restricted filesystem mounts: truncate to
        # empty so _acquire_pending treats it as absent on the next campaign.
        try:
            guard.write_text("", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass  # best-effort; stale guards can be resolved with recover_pending.py


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _worker_node_root(worker: Worker) -> Path | None:
    for attr in ("node_root", "_node_root"):
        value = getattr(worker, attr, None)
        if value:
            return Path(value).resolve()
    return None


def _manager_status(
    *,
    campaign_id: str,
    budget_index: int,
    current_best: float | None,
    node_spec: NodeSpec,
    worker: Worker | None = None,
) -> ManagerStatus:
    constants: dict[str, str] = {}
    effective_config: dict[str, str] = {}
    root = _worker_node_root(worker) if worker is not None else None
    if root is not None:
        for rel_path in node_spec.editable_paths:
            constants.update(parse_train_constants(_editable_path(root, rel_path)))
        editable_symbols = tuple(getattr(node_spec, "editable_symbols", ()) or ())
        if editable_symbols:
            allowed = set(editable_symbols)
            constants = {key: value for key, value in constants.items() if key in allowed}
        effective_config = build_effective_config(constants)
    return ManagerStatus(
        campaign_id=campaign_id,
        budget_index=budget_index,
        current_best_metric=current_best,
        metric_name=node_spec.metric_name,
        metric_direction=node_spec.metric_direction,
        current_constants=constants,
        effective_config=effective_config,
    )


def _editable_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else root / path


def _snapshot_editable_state(worker: Worker, node_spec: NodeSpec) -> _EditableStateSnapshot | None:
    root = _worker_node_root(worker)
    if root is None:
        return None
    contents: dict[str, bytes | None] = {}
    for path_str in node_spec.editable_paths:
        path = _editable_path(root, path_str)
        contents[path_str] = path.read_bytes() if path.exists() and path.is_file() else None
    return _EditableStateSnapshot(root=root, contents=contents)


def _restore_editable_state(snapshot: _EditableStateSnapshot | None) -> None:
    if snapshot is None:
        return
    for path_str, content in snapshot.contents.items():
        path = _editable_path(snapshot.root, path_str)
        if content is None:
            if path.exists() and path.is_file():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _precondition_failure_result(
    proposal: ManagerProposal,
    node_spec: NodeSpec,
    worker: Worker,
) -> WorkerResult | None:
    """Reject structured no-op/impossible edits before calling the worker."""
    edit = proposal.extra.get("structured_edit")
    if not isinstance(edit, dict):
        if proposal.extra.get("proposal_precondition_failed"):
            return _structured_failure_result(
                worker,
                category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
                message="proposal selector could not find a non-no-op structured edit",
                proposal=proposal,
            )
        return None
    if edit.get("type") not in {"python_constant", "config_value"}:
        return None
    path = str(edit.get("path") or "")
    symbol = str(edit.get("symbol") or "")
    old_value = str(edit.get("old") or "")
    new_value = str(edit.get("new") or "")
    if path not in node_spec.editable_paths:
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"structured edit path is not editable: {path}",
            proposal=proposal,
        )
    if not symbol or not old_value or not new_value:
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message="structured edit is missing symbol, old, or new value",
            proposal=proposal,
        )
    if values_equal(old_value, new_value):
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"structured edit is a no-op: {symbol} already equals {new_value}",
            proposal=proposal,
        )
    editable_symbols = tuple(getattr(node_spec, "editable_symbols", ()) or ())
    if editable_symbols and symbol not in set(editable_symbols):
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"structured edit symbol is not editable: {symbol}",
            proposal=proposal,
        )
    root = _worker_node_root(worker)
    if root is None:
        return None
    edit_path = root / path
    constants = parse_train_constants(edit_path)
    current = constants.get(symbol)
    if current is None:
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"symbol is not present in {path}: {symbol}",
            proposal=proposal,
        )
    if not values_equal(current, old_value):
        return _structured_failure_result(
            worker,
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"structured edit precondition failed for {symbol}: expected {old_value}, found {current}",
            proposal=proposal,
            extra={"actual_value": current},
        )
    effective_key = str(edit.get("effective_key") or "")
    if effective_key:
        before = build_effective_config(constants).get(effective_key, "")
        after = build_effective_config(
            constants_after_edit(constants, symbol=symbol, new_value=new_value)
        ).get(effective_key, "")
        if values_equal(before, after):
            return _structured_failure_result(
                worker,
                category=FailureCategory.EFFECTIVE_CONFIG_UNCHANGED,
                message=f"structured edit does not change effective training config: {effective_key}={before}",
                proposal=proposal,
                extra={"effective_key": effective_key, "effective_before": before, "effective_after": after},
            )
    return None


def _structured_failure_result(
    worker: Worker,
    *,
    category: FailureCategory,
    message: str,
    proposal: ManagerProposal,
    extra: dict[str, object] | None = None,
) -> WorkerResult:
    payload = {
        "failure_category": category.value,
        "structured_edit": proposal.extra.get("structured_edit", {}),
        "worker_skipped": True,
    }
    if extra:
        payload.update(extra)
    return WorkerResult(
        worker_mode=f"{getattr(worker, 'mode', 'unknown_worker')}_precheck",
        changed_files=(),
        success=False,
        parsed_metrics={},
        raw_log_ref="",
        patch_ref="",
        git_commit_before="",
        git_commit_after="",
        failure_message=message,
        extra=payload,
    )


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
    manager_temperature: float | None = None,
    proposal_backend: object | None = None,
    event_store: object | None = None,
    rationale_max_tokens: int | None = None,
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

    manager = proposal_backend or _manager(manager_mode, llm=manager_llm, temperature=manager_temperature)
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget, consumed_trials=len(existing_records))
    current_best = _current_best(existing_records, node_spec.metric_name)

    records: list[TrialRecord] = []
    campaign_started = time.perf_counter()
    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=None,
        event_type="campaign_started",
        payload={
            "budget": budget,
            "manager_mode": manager_mode,
            "memory_mode": memory_mode,
            "node_id": node_spec.name,
        },
    )
    while not budget_state.exhausted:
        budget_index = budget_state.next_budget_index
        trial_id = f"{campaign_id}-trial-{budget_index:03d}"
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_started",
            payload={"trial_id": trial_id, "budget_index": budget_index},
        )

        memory_context = build_memory_context(
            existing_records + records, MemoryMode(memory_mode), node_spec, budget_index,
            rationale_max_tokens=rationale_max_tokens,
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="memory_context_built",
            payload={
                "mode": memory_context.mode.value,
                "compressed_chars": memory_context.compressed_chars,
                "repeated_bad_count": memory_context.repeated_bad_stats.repeated_bad_count,
            },
        )
        status = _manager_status(
            campaign_id=campaign_id,
            budget_index=budget_index,
            current_best=current_best,
            node_spec=node_spec,
            worker=worker,
        )
        proposal = _proposal_with_campaign_metadata(
            _proposal_with_worker_memory(
                manager.propose_next_trial(status, memory_context, node_spec),
                memory_context=memory_context,
            ),
            node_spec=node_spec,
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="proposal_created",
            payload={"summary": proposal.proposal_summary, "manager_mode": proposal.manager_mode},
        )

        guard = _acquire_pending(
            records_path,
            campaign_id=campaign_id,
            trial_id=trial_id,
            budget_index=budget_index,
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="pending_guard_acquired",
            payload={"guard_path": str(guard)},
        )
        ts_start = _iso_now()
        worker_started = time.perf_counter()
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="worker_started",
            payload={"worker_mode": getattr(worker, "mode", "unknown_worker")},
        )
        editable_snapshot = _snapshot_editable_state(worker, node_spec)
        precondition_result = _precondition_failure_result(proposal, node_spec, worker)
        if precondition_result is not None:
            worker_result = precondition_result
        else:
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
                    extra={"failure_category": FailureCategory.EDIT_FAILED.value},
                )
        ts_end = _iso_now()
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="worker_finished",
            payload={
                "success": worker_result.success,
                "elapsed_seconds": max(time.perf_counter() - worker_started, 0.0),
            },
        )

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
        scope_payload = dict(record.extra.get("scope_validation", {}))
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="scope_validated",
            payload={
                "valid": bool(scope_payload.get("valid")),
                "changed_files": list(scope_payload.get("changed_paths") or worker_result.changed_files),
            },
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="metric_parsed",
            payload={
                "metric_name": node_spec.metric_name,
                "value": record.parsed_metrics.get(node_spec.metric_name),
            },
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="decision_made",
            payload={
                "decision": record.decision.value,
                "delta_vs_best": record.delta_vs_best,
                "failure_category": (
                    record.failure_category.value if record.failure_category else None
                ),
            },
        )
        # Compute cumulative repeated-bad count for this trial (including itself)
        # and stamp it onto the immutable record before persisting.
        rbs = compute_repeated_bad_stats(existing_records + records + [record])
        record = replace(record, repeated_bad_count=rbs.repeated_bad_count)
        records.append(record)
        store.append_many([record])
        if record.decision != TrialDecision.KEPT:
            _restore_editable_state(editable_snapshot)
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_record_appended",
            payload={"trial_id": trial_id, "repeated_bad_count": rbs.repeated_bad_count},
        )
        _release_pending(guard)
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="pending_guard_released",
            payload={},
        )

        if record.decision == TrialDecision.KEPT and node_spec.metric_name in record.parsed_metrics:
            current_best = record.parsed_metrics[node_spec.metric_name]
        budget_state = budget_state.consume_one()

    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=None,
        event_type="campaign_completed",
        payload={
            "records_written": len(records),
            "elapsed_seconds": max(time.perf_counter() - campaign_started, 0.0),
        },
    )
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
    manager_temperature: float | None = None,
    dry_run_profile: str = "monotonic",
    proposal_backend: object | None = None,
    event_store: object | None = None,
) -> CampaignRunResult:
    manager = proposal_backend or _manager(manager_mode, llm=manager_llm, temperature=manager_temperature)
    worker = DryRunWorker(profile=dry_run_profile)
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget, consumed_trials=len(existing_records))
    current_best = _current_best(existing_records, node_spec.metric_name)

    records: list[TrialRecord] = []
    campaign_started = time.perf_counter()
    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=None,
        event_type="campaign_started",
        payload={
            "budget": budget,
            "manager_mode": manager_mode,
            "memory_mode": memory_mode,
            "node_id": node_spec.name,
        },
    )
    while not budget_state.exhausted:
        budget_index = budget_state.next_budget_index
        trial_id = f"{campaign_id}-trial-{budget_index:03d}"
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_started",
            payload={"trial_id": trial_id, "budget_index": budget_index},
        )
        memory_context = build_memory_context(existing_records + records, MemoryMode(memory_mode), node_spec, budget_index)
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="memory_context_built",
            payload={
                "mode": memory_context.mode.value,
                "compressed_chars": memory_context.compressed_chars,
                "repeated_bad_count": memory_context.repeated_bad_stats.repeated_bad_count,
            },
        )
        status = _manager_status(
            campaign_id=campaign_id,
            budget_index=budget_index,
            current_best=current_best,
            node_spec=node_spec,
            worker=None,
        )
        proposal = _proposal_with_campaign_metadata(
            _proposal_with_worker_memory(
                manager.propose_next_trial(status, memory_context, node_spec),
                memory_context=memory_context,
            ),
            node_spec=node_spec,
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="proposal_created",
            payload={"summary": proposal.proposal_summary, "manager_mode": proposal.manager_mode},
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="pending_guard_acquired",
            payload={"guard_path": None, "synthetic": True},
        )
        worker_started = time.perf_counter()
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="worker_started",
            payload={"worker_mode": worker.mode},
        )
        worker_result = worker.run_trial(proposal, node_spec, budget_index)
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="worker_finished",
            payload={
                "success": worker_result.success,
                "elapsed_seconds": max(time.perf_counter() - worker_started, 0.0),
            },
        )
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
        scope_payload = dict(record.extra.get("scope_validation", {}))
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="scope_validated",
            payload={
                "valid": bool(scope_payload.get("valid")),
                "changed_files": list(scope_payload.get("changed_paths") or worker_result.changed_files),
            },
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="metric_parsed",
            payload={
                "metric_name": node_spec.metric_name,
                "value": record.parsed_metrics.get(node_spec.metric_name),
            },
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="decision_made",
            payload={
                "decision": record.decision.value,
                "delta_vs_best": record.delta_vs_best,
                "failure_category": (
                    record.failure_category.value if record.failure_category else None
                ),
            },
        )
        # Compute cumulative repeated-bad count for this trial (including itself).
        rbs = compute_repeated_bad_stats(existing_records + records + [record])
        record = replace(record, repeated_bad_count=rbs.repeated_bad_count)
        records.append(record)
        store.append_many([record])
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_record_appended",
            payload={"trial_id": trial_id, "repeated_bad_count": rbs.repeated_bad_count},
        )
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="pending_guard_released",
            payload={},
        )
        if record.decision == TrialDecision.KEPT and node_spec.metric_name in record.parsed_metrics:
            current_best = record.parsed_metrics[node_spec.metric_name]
        budget_state = budget_state.consume_one()

    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=None,
        event_type="campaign_completed",
        payload={
            "records_written": len(records),
            "elapsed_seconds": max(time.perf_counter() - campaign_started, 0.0),
        },
    )
    return CampaignRunResult(campaign_id=campaign_id, records_path=str(records_path), records_written=len(records), dry_run=True)


def _elapsed_seconds(start: str | None, end: str | None) -> float:
    """Return wall-clock seconds between two ISO-8601 UTC timestamps.

    Falls back to 0.0 if either timestamp is missing or unparseable (e.g.
    dry-run paths that do not supply real timestamps).
    """
    if not start or not end:
        return 0.0
    try:
        t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max((t1 - t0).total_seconds(), 0.0)
    except (ValueError, TypeError):
        return 0.0


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
    worker_failure_category = str(worker_result.extra.get("failure_category", "")).strip()
    worker_failure = None
    if worker_failure_category:
        try:
            worker_failure = FailureCategory(worker_failure_category)
        except ValueError:
            worker_failure = None

    # Detect no-op patches: worker succeeded but produced no real diff.
    # This must be checked before the validity decision so it is recorded as
    # failed_invalid / no_op_patch rather than silently treated as a valid trial.
    no_op = (
        worker_failure == FailureCategory.NO_OP_PATCH
        or (worker_result.success and _patch_is_empty(worker_result.patch_ref))
    )
    effective_config_unchanged = (
        worker_failure == FailureCategory.EFFECTIVE_CONFIG_UNCHANGED
        or worker_result.extra.get("effective_config_changed") is False
    )

    validity = (
        ValidityStatus.VALID
        if (
            worker_result.success
            and scope.valid
            and metric is not None
            and not no_op
            and not effective_config_unchanged
            and worker_failure is None
        )
        else ValidityStatus.INVALID
    )
    if no_op:
        failure = FailureCategory.NO_OP_PATCH
    elif effective_config_unchanged:
        failure = FailureCategory.EFFECTIVE_CONFIG_UNCHANGED
    elif not scope.valid:
        failure = FailureCategory.INVALID_EDIT_SCOPE
    elif worker_failure is not None:
        failure = worker_failure
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
        wall_clock_seconds=_elapsed_seconds(timestamp_start, timestamp_end),
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
        extra={
            "manager": proposal_extra or {},
            "worker": worker_result.extra,
            "worker_failure_message": worker_result.failure_message,
            "scope_validation": scope.__dict__,
        },
        # Reproducibility hashes: patch_hash is computed here from the patch file;
        # the others are best captured by the worker before edits are applied and
        # forwarded via worker_result.extra so they can be picked up below.
        patch_hash=_sha256_file(worker_result.patch_ref),
        node_state_hash=worker_result.extra.get("node_state_hash") or None,
        fast_config_hash=worker_result.extra.get("fast_config_hash") or None,
        training_seed=worker_result.extra.get("training_seed") or None,
    )


def _manager(manager_mode: str, llm: object | None = None, temperature: float | None = None):
    if manager_mode == "baseline_manager":
        return BaselineManager()
    if manager_mode == "prompt_manager":
        return PromptManager()
    if manager_mode == "langgraph_manager":
        from autoresearch.manager.langgraph_manager import LangGraphManager
        kwargs: dict = {"llm": llm}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return LangGraphManager(**kwargs)
    raise ValueError(f"unknown manager mode: {manager_mode}")


def _proposal_with_worker_memory(proposal: ManagerProposal, *, memory_context) -> ManagerProposal:
    """Attach read-only prior-trial memory for workers that can use it."""
    extra = dict(proposal.extra)
    extra.setdefault("worker_memory_mode", memory_context.mode.value)
    extra.setdefault("worker_memory_context_text", memory_context.context_text)
    extra.setdefault("worker_repeated_bad_stats", memory_context.repeated_bad_stats.to_dict())
    return replace(proposal, extra=extra)


def _proposal_with_campaign_metadata(proposal: ManagerProposal, *, node_spec: NodeSpec) -> ManagerProposal:
    """Attach optional reproducibility metadata that is independent of the worker."""
    extra = dict(proposal.extra)
    ref = load_node_research_context_ref(node_spec=node_spec, repo_root=_repo_root())
    if ref is not None:
        extra.setdefault("research_context", ref.to_dict())
    return replace(proposal, extra=extra)


def _current_best(records: list[TrialRecord], metric_name: str) -> float | None:
    values = [record.parsed_metrics[metric_name] for record in records if metric_name in record.parsed_metrics]
    return max(values) if values else None
