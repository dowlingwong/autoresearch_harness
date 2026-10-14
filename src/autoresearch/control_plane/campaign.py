from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


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

from autoresearch.control_plane.budget import BudgetState, ComputeBudget
from autoresearch.control_plane.controller import (
    ContinueController,
    ControlDecision,
    Controller,
    build_control_view,
    compute_mask,
    proposal_digest,
)
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
from autoresearch.nodes.active_keys import inactive_key_in_proposal, resolve_active_keys
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import DryRunWorker, Worker, WorkerResult

# D25: how many times the control plane re-asks the manager for a proposal
# after an inactive key was named, before the trial is skipped.  Each revision
# is an LLM call charged to the LLM-call cap; no worker run is ever spent.
MAX_INACTIVE_KEY_REVISIONS = 2


class SeedInjectionUnsupportedError(RuntimeError):
    """Raised when a seeded (verify) run is requested from a worker without seed support."""


def _supports_seed_injection(worker: object) -> bool:
    """True when the worker can run a trial at an injected seed (E2 verify)."""
    return hasattr(worker, "run_trial_at_seed")


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
        # D25: a node that declares active keys shows the manager only those
        # keys (prompt and structured-edit resolution).  Undeclared nodes are
        # unchanged.
        active = resolve_active_keys(node_spec, constants)
        if active is not None:
            active_set = set(active)
            constants = {key: value for key, value in constants.items() if key in active_set}
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


def _budget_view(budget_state: BudgetState, compute_budget: ComputeBudget) -> dict[str, object]:
    return {
        "trials_remaining": max(budget_state.total_trials - budget_state.consumed_trials, 0),
        "trials_total": budget_state.total_trials,
        **compute_budget.remaining(),
    }


def _controller_hook(
    *,
    controller: Controller,
    log_default_decisions: bool,
    event_store: object | None,
    campaign_id: str,
    trial_id: str,
    epoch: str,
    budget_index: int,
    ledger_prefix: list[TrialRecord],
    current_best: float | None,
    incumbent_stack_depth: int,
    proposal: ManagerProposal | None,
    provisional: TrialRecord | None,
    budget_state: BudgetState,
    compute_budget: ComputeBudget,
    node_spec: NodeSpec,
    **feasibility: object,
) -> tuple[ControlDecision, dict[str, bool]]:
    """Build the view, compute the mask, ask the controller, and log the call.

    A non-default controller (or ``log_default_decisions``) emits one
    ``control_decision`` event.  The default ``ContinueController`` emits
    nothing, which keeps the pre-hook record and event streams identical.
    """
    view = build_control_view(
        epoch=epoch,  # type: ignore[arg-type]
        budget_index=budget_index,
        ledger_prefix=ledger_prefix,
        current_best=current_best,
        incumbent_stack_depth=incumbent_stack_depth,
        proposal=proposal,
        provisional=provisional,
        budget=_budget_view(budget_state, compute_budget),
        node=node_spec,
    )
    mask = compute_mask(view, **feasibility)  # type: ignore[arg-type]
    decision = controller.decide(view, mask)
    if (not isinstance(controller, ContinueController)) or log_default_decisions:
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="control_decision",
            payload={
                "epoch": epoch,
                "action": decision.action,
                "scores": dict(decision.scores),
                "mask": dict(mask),
                "controller_version": decision.controller_version,
                "input_hash": decision.input_hash,
                "llm_calls": decision.llm_calls,
            },
        )
    return decision, mask


def _editable_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else root / path


def _emit_llm_call(event_store: object | None, *, campaign_id: str, trial_id: str, proposal: ManagerProposal) -> None:
    """Emit an ``llm_call`` event when a manager reports per-call telemetry.

    Managers that make no LLM call (baseline, prompt) carry no ``llm_call``
    key, so no event is emitted and the default event stream is unchanged.
    """
    meta = (proposal.extra or {}).get("llm_call")
    if not meta:
        return
    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=trial_id,
        event_type="llm_call",
        payload=dict(meta),
    )


@dataclass
class _IncumbentEntry:
    """One kept incumbent, stored so E0 ``demote`` can revert it."""

    snapshot: _EditableStateSnapshot | None
    current_best_before: float | None
    metric: float | None
    verified: bool


@dataclass
class SnapshotContext:
    """Live campaign state handed to an optional ``snapshot_hook`` at an epoch.

    The hook is trusted engineering code (L7).  ``incumbent_snapshot`` is the
    pre-trial editable state; the hook reads the current workspace separately
    when it needs the candidate bytes (E2).
    """

    epoch: str
    budget_index: int
    campaign_id: str
    trial_id: str
    node_spec: NodeSpec
    worker: Worker
    proposal: ManagerProposal | None
    provisional: TrialRecord | None
    worker_result: WorkerResult | None
    incumbent_snapshot: _EditableStateSnapshot | None
    ledger_prefix: list[TrialRecord]
    current_best: float | None
    incumbent_stack: list[_IncumbentEntry]
    budget_state: BudgetState
    compute_budget: ComputeBudget
    controller: Controller
    manager: object
    memory_mode: str
    manager_mode: str
    t_min: int
    verify_runs: int
    verify_split_policy: str


def _invoke_snapshot_hook(snapshot_hook: object | None, **kwargs: object) -> None:
    if snapshot_hook is not None:
        snapshot_hook(SnapshotContext(**kwargs))  # type: ignore[misc]


# Actions run_real_campaign implements at each epoch (BUILD_RUNBOOK L6).
_EPOCH_ACTIONS: dict[str, tuple[str, ...]] = {
    "E0": ("continue", "stop", "demote"),
    "E1": ("continue", "skip", "revise"),
    "E2": ("continue", "verify"),
}


def _resolve_action(
    decision: ControlDecision,
    mask: dict[str, bool],
    *,
    epoch: str,
    event_store: object | None,
    campaign_id: str,
    trial_id: str,
) -> str:
    """Return the action to execute, falling back to ``continue`` if it is not allowed.

    A controller choice that the mask rules out, or that is not implemented at
    *epoch*, never crashes the campaign (E1/E2 run while the pending guard is
    held).  It is logged as a ``control_decision_rejected`` event
    ``{epoch, action, mask}`` and the trial proceeds as ``continue``.
    """
    action = decision.action
    if action in _EPOCH_ACTIONS.get(epoch, ("continue",)) and mask.get(action, False):
        return action
    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=trial_id,
        event_type="control_decision_rejected",
        payload={"epoch": epoch, "action": action, "mask": dict(mask)},
    )
    return "continue"


def _accepts_kwarg(manager: object, name: str) -> bool:
    """True when ``manager.propose_next_trial`` takes the keyword *name*."""
    import inspect

    method = getattr(manager, "propose_next_trial", None)
    if method is None:
        return False
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _accepts_revision_feedback(manager: object) -> bool:
    """True when ``manager.propose_next_trial`` takes ``revision_feedback``."""
    return _accepts_kwarg(manager, "revision_feedback")


def _revision_feedback(memory_context: object) -> str:
    stats = getattr(memory_context, "repeated_bad_stats", None)
    if stats is not None and getattr(stats, "repeated_bad_count", 0):
        return "repeated direction: avoid the parameter/direction already tried"
    return "step too large: propose a smaller bounded step"


def _editable_state_hash(worker: Worker, node_spec: NodeSpec) -> str | None:
    """SHA-256 over the editable files, used by verify swap and demote audits."""
    import hashlib

    root = _worker_node_root(worker)
    if root is None:
        return None
    digest = hashlib.sha256()
    for path_str in node_spec.editable_paths:
        digest.update(path_str.encode("utf-8"))
        path = _editable_path(root, path_str)
        digest.update(path.read_bytes() if path.exists() and path.is_file() else b"<missing>")
    return digest.hexdigest()


def _effective_config_at(worker: Worker, node_spec: NodeSpec) -> dict[str, str] | None:
    """Effective config of the current workspace (D27), or None without a node root."""
    root = _worker_node_root(worker)
    if root is None:
        return None
    constants: dict[str, str] = {}
    for rel_path in node_spec.editable_paths:
        constants.update(parse_train_constants(_editable_path(root, rel_path)))
    return build_effective_config(constants)


def _llm_cap_reached(compute_budget: ComputeBudget) -> bool:
    return (
        compute_budget.max_llm_calls is not None
        and compute_budget.llm_calls >= compute_budget.max_llm_calls
    )


def _skipped_record(
    *,
    campaign_id: str,
    budget_index: int,
    node_spec: NodeSpec,
    proposal: ManagerProposal,
    memory_mode: str,
    current_best: float | None,
    timestamp_start: str,
    timestamp_end: str,
    control: dict[str, Any],
    rationale: str | None = None,
) -> TrialRecord:
    trial_id = f"{campaign_id}-trial-{budget_index:03d}"
    return TrialRecord(
        trial_id=trial_id,
        campaign_id=campaign_id,
        node_id=node_spec.name,
        budget_index=budget_index,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        manager_mode=proposal.manager_mode,
        worker_mode="control_plane",
        memory_mode=memory_mode,
        proposal_summary=proposal.proposal_summary,
        proposal_rationale=proposal.proposal_rationale,
        targeted_files=node_spec.editable_paths,
        patch_ref="",
        git_commit_before="",
        git_commit_after="",
        execution_status=ExecutionStatus.FAILED,
        validity_status=ValidityStatus.VALID,
        failure_category=None,
        raw_log_ref="",
        parsed_metrics={},
        current_best_before=current_best,
        delta_vs_best=None,
        decision=TrialDecision.SKIPPED,
        decision_rationale=rationale or "Skipped by controller at E1: proposal not executed.",
        wall_clock_seconds=_elapsed_seconds(timestamp_start, timestamp_end),
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
        extra={"control": control},
    )


def _noedit_proposal(proposal: ManagerProposal) -> ManagerProposal:
    """A copy that runs the current workspace without applying an edit."""
    extra = {key: value for key, value in proposal.extra.items() if key != "structured_edit"}
    extra.pop("deterministic_patch", None)
    return replace(
        proposal,
        objective="Run the restored configuration unchanged; make no edit.",
        extra=extra,
    )


def _run_worker_at_seed(
    worker: Worker,
    proposal: ManagerProposal,
    node_spec: NodeSpec,
    budget_index: int,
    seed: int,
    split_seed: int | None = None,
) -> WorkerResult:
    """Run one verify re-run at training seed *seed*.

    *split_seed* ``None`` leaves the worker's data split where it is (the
    ``"fixed"`` verify split policy); an int moves the split as well.

    Never falls back to an unseeded ``run_trial``: paired re-runs without seed
    control would be meaningless, so a worker without ``run_trial_at_seed``
    raises :class:`SeedInjectionUnsupportedError`.  The E2 mask already rules
    verify out for such workers.
    """
    if not _supports_seed_injection(worker):
        raise SeedInjectionUnsupportedError(
            f"worker {getattr(worker, 'mode', type(worker).__name__)!r} does not support "
            "seeded runs (run_trial_at_seed); verify cannot run"
        )
    return worker.run_trial_at_seed(
        proposal, node_spec, budget_index, train_seed=seed, split_seed=split_seed
    )


def _run_verify(
    *,
    record: TrialRecord,
    proposal: ManagerProposal,
    node_spec: NodeSpec,
    worker: Worker,
    incumbent_snapshot: _EditableStateSnapshot | None,
    base_budget_index: int,
    verify_runs: int,
    compute_budget: ComputeBudget,
    verify_split_policy: Literal["fixed", "vary"] = "fixed",
) -> tuple[TrialRecord, dict[str, Any]]:
    """Paired fresh-seed verification with a frozen response rule (E2).

    ``verify_split_policy="fixed"`` (default) varies only the training seed and
    passes ``split_seed=None``, so every re-run sees the campaign's data split.
    ``"vary"`` sets the split seed equal to the training seed for each pair.
    The policy is recorded as ``extra.control.verify.split_policy``.
    """
    if verify_split_policy not in ("fixed", "vary"):
        raise ValueError(f"unknown verify_split_policy: {verify_split_policy!r}")
    metric_name = node_spec.metric_name
    direction = node_spec.metric_direction
    candidate_snapshot = _snapshot_editable_state(worker, node_spec)
    candidate_hash_before = _editable_state_hash(worker, node_spec)
    seeds = [1000 + base_budget_index * 10 + index for index in range(verify_runs)]
    noedit = _noedit_proposal(proposal)

    candidate_metrics: list[float | None] = []
    for index, seed in enumerate(seeds):
        run_t0 = time.perf_counter()
        result = _run_worker_at_seed(
            worker, noedit, node_spec, 900000 + base_budget_index * 10 + index, seed,
            split_seed=seed if verify_split_policy == "vary" else None,
        )
        compute_budget = compute_budget.consume_worker_run(time.perf_counter() - run_t0)
        candidate_metrics.append(result.parsed_metrics.get(metric_name) if result.success else None)

    _restore_editable_state(incumbent_snapshot)
    incumbent_hash = _editable_state_hash(worker, node_spec)
    incumbent_metrics: list[float | None] = []
    for index, seed in enumerate(seeds):
        run_t0 = time.perf_counter()
        result = _run_worker_at_seed(
            worker, noedit, node_spec, 950000 + base_budget_index * 10 + index, seed,
            split_seed=seed if verify_split_policy == "vary" else None,
        )
        compute_budget = compute_budget.consume_worker_run(time.perf_counter() - run_t0)
        incumbent_metrics.append(result.parsed_metrics.get(metric_name) if result.success else None)

    _restore_editable_state(candidate_snapshot)
    candidate_hash_after = _editable_state_hash(worker, node_spec)

    pairs = [
        (candidate, incumbent)
        for candidate, incumbent in zip(candidate_metrics, incumbent_metrics)
        if candidate is not None and incumbent is not None
    ]
    deltas = [candidate - incumbent for candidate, incumbent in pairs]
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    improved = mean_delta < 0 if direction == "minimize" else mean_delta > 0
    response = "keep" if (pairs and improved) else "reject"

    original_decision = record.decision
    decision = record.decision
    failure = record.failure_category
    rationale = record.decision_rationale
    if response == "keep" and original_decision != TrialDecision.KEPT:
        decision = TrialDecision.KEPT
        failure = None
        rationale = (
            f"Verify rescued a discard: mean paired delta {mean_delta:.6f} "
            f"improves the {direction} objective."
        )
    elif response == "reject" and original_decision == TrialDecision.KEPT:
        decision = TrialDecision.DISCARDED
        failure = FailureCategory.DEGRADED_METRIC
        rationale = (
            f"Verify vetoed a keep: mean paired delta {mean_delta:.6f} "
            f"does not improve the {direction} objective."
        )
    verify_meta = {
        "r": verify_runs,
        "seeds": seeds,
        "split_policy": verify_split_policy,
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "mean_delta": mean_delta,
        "response": response,
        "original_decision": original_decision.value,
        "candidate_state_hash_before": candidate_hash_before,
        "candidate_state_hash_after": candidate_hash_after,
        "incumbent_state_hash": incumbent_hash,
    }
    updated = replace(
        record, decision=decision, failure_category=failure, decision_rationale=rationale
    )
    return updated, {"verify": verify_meta, "compute_budget": compute_budget}


def _perform_demote(
    *,
    incumbent_stack: list[_IncumbentEntry],
    worker: Worker,
    node_spec: NodeSpec,
    current_best: float | None,
    event_store: object | None,
    campaign_id: str,
    trial_id: str,
) -> float | None:
    """Pop the top incumbent if it is an unverified keep; return the restored current_best.

    Demote never reaches below the top of the stack, so a verified keep is
    never reverted (directly or by deleting the stack above an older entry).
    The E0 mask enforces the same rule.
    """
    if not incumbent_stack:
        raise ValueError("demote requested with an empty incumbent stack")
    if incumbent_stack[-1].verified:
        raise ValueError("demote requested but the top incumbent is a verified keep")
    hash_before = _editable_state_hash(worker, node_spec)
    entry = incumbent_stack.pop()
    _restore_editable_state(entry.snapshot)
    hash_after = _editable_state_hash(worker, node_spec)
    emit(
        event_store,
        campaign_id=campaign_id,
        trial_id=trial_id,
        event_type="incumbent_demoted",
        payload={"hash_before": hash_before, "hash_after": hash_after},
    )
    return entry.current_best_before


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
    ungoverned: bool = False,
    controller: Controller | None = None,
    log_default_decisions: bool = False,
    compute_budget: ComputeBudget | None = None,
    t_min: int = 3,
    verify_runs: int = 2,
    verify_split_policy: Literal["fixed", "vary"] = "fixed",
    snapshot_hook: object | None = None,
    initial_current_best: float | None = None,
    initial_incumbent_stack: list[_IncumbentEntry] | None = None,
) -> CampaignRunResult:
    """Run a fixed-budget campaign with a real worker.

    The control plane owns every state transition:
    - builds memory context from prior records before each trial
    - acquires a pending-trial guard before calling the worker
    - releases the guard on completion or failure
    - decides keep/discard based on parsed metric, never trusting the worker
    - appends one authoritative TrialRecord per trial

    If *ungoverned* is True the three governance mechanisms that would
    otherwise protect the campaign are disabled, to produce the
    "ungoverned counterfactual" used in the paper's Level 2 analysis:

      1. Pending-trial guard — NOT written/deleted.  A mid-trial crash
         leaves no sentinel file; the next campaign start has no way to
         detect it.
      2. Ledger append on failure — failed_invalid trials are NOT written
         to the append-only JSONL ledger.  From the ledger's perspective
         those trials never happened.
      3. Scope enforcement — still performed and recorded in the
         observation log, but does NOT block the trial from being appended
         if it somehow succeeds (scope violations can execute).

    A lightweight *_ungoverned_obs.jsonl* observation log is written
    alongside the ledger so that we can compare the two and show exactly
    what the governed ledger would have captured versus what the ungoverned
    regime left behind.

    **Controller (R1, BUILD_RUNBOOK L5/L6).**  *controller* is asked at E0
    (``continue``/``stop``/``demote``), E1 (``continue``/``skip``/``revise``)
    and E2 (``continue``/``verify``).  The default ``ContinueController``
    reproduces the pre-hook records exactly.

    - A choice the feasibility mask rules out, or one not implemented at that
      epoch, is logged as ``control_decision_rejected`` ``{epoch, action,
      mask}`` and the trial proceeds as ``continue``; it never raises.
    - ``revise`` re-calls the manager with ``revision_feedback=<text>``; it is
      feasible only for non-deterministic managers that accept that keyword.
    - ``verify`` runs *verify_runs* paired seeded re-runs; it is feasible only
      when the worker has ``run_trial_at_seed``.  *verify_split_policy*
      ``"fixed"`` (default) varies only the training seed; ``"vary"`` moves the
      split seed with it.  The policy is recorded in ``extra.control.verify``.
    - ``demote`` pops the top of the incumbent stack and is feasible only when
      that top entry is an unverified keep; verified keeps are never reverted.

    *compute_budget* counts every worker run (including verify re-runs),
    worker seconds, and LLM calls; proposals from deterministic managers
    (``deterministic = True``) are not counted as LLM calls.
    """
    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    # Ungoverned observation log — written only when ungoverned=True.
    # Each line: lightweight JSON with what an ungoverned regime actually saw.
    obs_path: Path | None = None
    if ungoverned:
        stem = records_path.stem  # e.g. "some_campaign_trials"
        cid_stem = stem[:-len("_trials")] if stem.endswith("_trials") else stem
        obs_path = records_path.parent / f"{cid_stem}_ungoverned_obs.jsonl"

    manager = proposal_backend or _manager(manager_mode, llm=manager_llm, temperature=manager_temperature)
    controller = controller or ContinueController()
    compute_budget = compute_budget or ComputeBudget()
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget, consumed_trials=len(existing_records))
    current_best = (
        initial_current_best
        if initial_current_best is not None
        else _current_best(existing_records, node_spec.metric_name, node_spec.metric_direction)
    )
    incumbent_stack: list[_IncumbentEntry] = list(initial_incumbent_stack or [])
    stop_requested = False
    # D27: record-only novelty flags.  The incumbent's effective config starts
    # as the workspace configuration and follows keeps and demotions; the seen
    # list holds the effective config of every evaluated trial.
    incumbent_effective: dict[str, str] | None = _effective_config_at(worker, node_spec)
    seen_effective: list[dict[str, str]] = []

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
    while not budget_state.exhausted and not compute_budget.exhausted:
        budget_index = budget_state.next_budget_index
        trial_id = f"{campaign_id}-trial-{budget_index:03d}"
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_started",
            payload={"trial_id": trial_id, "budget_index": budget_index},
        )

        e0_decision, _e0_mask = _controller_hook(
            controller=controller,
            log_default_decisions=log_default_decisions,
            event_store=event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            epoch="E0",
            budget_index=budget_index,
            ledger_prefix=existing_records + records,
            current_best=current_best,
            incumbent_stack_depth=len(incumbent_stack),
            proposal=None,
            provisional=None,
            budget_state=budget_state,
            compute_budget=compute_budget,
            node_spec=node_spec,
            top_of_stack_unverified=bool(incumbent_stack) and not incumbent_stack[-1].verified,
            t_min=t_min,
        )
        if e0_decision.llm_calls:
            compute_budget = compute_budget.consume_llm_call(e0_decision.llm_calls)
        _invoke_snapshot_hook(
            snapshot_hook,
            epoch="E0", budget_index=budget_index, campaign_id=campaign_id, trial_id=trial_id,
            node_spec=node_spec, worker=worker, proposal=None, provisional=None, worker_result=None,
            incumbent_snapshot=None, ledger_prefix=existing_records + records, current_best=current_best,
            incumbent_stack=incumbent_stack, budget_state=budget_state, compute_budget=compute_budget,
            controller=controller, manager=manager, memory_mode=memory_mode, manager_mode=manager_mode,
            t_min=t_min, verify_runs=verify_runs, verify_split_policy=verify_split_policy,
        )
        e0_action = _resolve_action(
            e0_decision, _e0_mask, epoch="E0", event_store=event_store,
            campaign_id=campaign_id, trial_id=trial_id,
        )
        if e0_action == "stop":
            emit(
                event_store,
                campaign_id=campaign_id,
                trial_id=trial_id,
                event_type="campaign_stopped_by_controller",
                payload={"remaining": _budget_view(budget_state, compute_budget)},
            )
            stop_requested = True
            break
        if e0_action == "demote":
            current_best = _perform_demote(
                incumbent_stack=incumbent_stack,
                worker=worker,
                node_spec=node_spec,
                current_best=current_best,
                event_store=event_store,
                campaign_id=campaign_id,
                trial_id=trial_id,
            )
            restored = _effective_config_at(worker, node_spec)
            if restored is not None:
                incumbent_effective = restored

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
        if not getattr(manager, "deterministic", False):
            # Only LLM-backed managers make a call; deterministic managers
            # (baseline, prompt) consume no LLM budget.
            compute_budget = compute_budget.consume_llm_call()
        _emit_llm_call(event_store, campaign_id=campaign_id, trial_id=trial_id, proposal=proposal)

        # D25: a proposal that names an inactive key is rejected before any
        # worker run.  The manager may revise (each revision is an LLM call
        # charged to the cap); when the revisions are exhausted the trial is
        # skipped with reason ``inactive_key`` and no worker run.
        inactive_rejected: dict[str, Any] | None = None
        if resolve_active_keys(node_spec, status.current_constants) is not None:
            rejected_key = inactive_key_in_proposal(proposal, node_spec, status.current_constants)
            revisions_used = 0
            while (
                rejected_key is not None
                and revisions_used < MAX_INACTIVE_KEY_REVISIONS
                and _accepts_revision_feedback(manager)
                and not getattr(manager, "deterministic", False)
                and not _llm_cap_reached(compute_budget)
            ):
                emit(
                    event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    event_type="proposal_rejected_inactive_key",
                    payload={
                        "key": rejected_key,
                        "revision": revisions_used,
                        "active_keys": list(resolve_active_keys(node_spec, status.current_constants) or ()),
                    },
                )
                proposal = manager.propose_next_trial(
                    status,
                    memory_context,
                    node_spec,
                    revision_feedback=(
                        f"inactive key: {rejected_key} has no effect for the current "
                        "configuration; choose one of the active editable keys"
                    ),
                    revision_attempt=revisions_used + 1,
                )
                compute_budget = compute_budget.consume_llm_call()
                proposal = _proposal_with_campaign_metadata(
                    _proposal_with_worker_memory(proposal, memory_context=memory_context),
                    node_spec=node_spec,
                )
                _emit_llm_call(event_store, campaign_id=campaign_id, trial_id=trial_id, proposal=proposal)
                revisions_used += 1
                rejected_key = inactive_key_in_proposal(proposal, node_spec, status.current_constants)
            if rejected_key is not None:
                inactive_rejected = {"key": rejected_key, "revisions": revisions_used}
                emit(
                    event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    event_type="proposal_rejected_inactive_key",
                    payload={
                        "key": rejected_key,
                        "revision": revisions_used,
                        "active_keys": list(resolve_active_keys(node_spec, status.current_constants) or ()),
                        "skipped": True,
                    },
                )

        if ungoverned:
            # UNGOVERNED: pending guard is not written.
            # A crash here leaves no sentinel; it is silently unrecoverable.
            guard = None  # type: ignore[assignment]
            emit(
                event_store,
                campaign_id=campaign_id,
                trial_id=trial_id,
                event_type="pending_guard_acquired",
                payload={"guard_path": None, "ungoverned": True},
            )
        else:
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
        editable_snapshot = _snapshot_editable_state(worker, node_spec)
        precondition_result = _precondition_failure_result(proposal, node_spec, worker)
        control_meta: dict[str, Any] = {}
        skip_trial = False
        worker_result: WorkerResult | None = None
        if precondition_result is not None:
            emit(
                event_store, campaign_id=campaign_id, trial_id=trial_id,
                event_type="worker_started",
                payload={"worker_mode": getattr(worker, "mode", "unknown_worker")},
            )
            worker_result = precondition_result
            emit(
                event_store, campaign_id=campaign_id, trial_id=trial_id,
                event_type="worker_finished",
                payload={"success": worker_result.success,
                         "elapsed_seconds": max(time.perf_counter() - worker_started, 0.0)},
            )
        else:
            _invoke_snapshot_hook(
                snapshot_hook,
                epoch="E1", budget_index=budget_index, campaign_id=campaign_id, trial_id=trial_id,
                node_spec=node_spec, worker=worker, proposal=proposal, provisional=None,
                worker_result=None, incumbent_snapshot=editable_snapshot,
                ledger_prefix=existing_records + records, current_best=current_best,
                incumbent_stack=incumbent_stack, budget_state=budget_state,
                compute_budget=compute_budget, controller=controller, manager=manager,
                memory_mode=memory_mode, manager_mode=manager_mode, t_min=t_min,
                verify_runs=verify_runs, verify_split_policy=verify_split_policy,
            )
            if inactive_rejected is not None:
                # D25: revisions exhausted on an inactive-key proposal; skip
                # without asking the controller and without any worker run.
                skip_trial = True
                control_meta = {"epoch": "E1", "action": "skip", "reason": "inactive_key", **inactive_rejected}
            else:
                e1_decision, e1_mask = _controller_hook(
                    controller=controller,
                    log_default_decisions=log_default_decisions,
                    event_store=event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    epoch="E1",
                    budget_index=budget_index,
                    ledger_prefix=existing_records + records,
                    current_best=current_best,
                    incumbent_stack_depth=len(incumbent_stack),
                    proposal=proposal,
                    provisional=None,
                    budget_state=budget_state,
                    compute_budget=compute_budget,
                    node_spec=node_spec,
                    proposal_passed_precondition=True,
                    manager_deterministic=bool(getattr(manager, "deterministic", False)),
                    revision_feedback_supported=_accepts_revision_feedback(manager),
                    revisions_this_trial=0,
                    llm_cap_reached=_llm_cap_reached(compute_budget),
                )
                if e1_decision.llm_calls:
                    compute_budget = compute_budget.consume_llm_call(e1_decision.llm_calls)
                e1_action = _resolve_action(
                    e1_decision, e1_mask, epoch="E1", event_store=event_store,
                    campaign_id=campaign_id, trial_id=trial_id,
                )
                if e1_action == "skip":
                    skip_trial = True
                    control_meta = {"epoch": "E1", "action": "skip"}
                else:
                    if e1_action == "revise":
                        feedback = _revision_feedback(memory_context)
                        revise_kwargs: dict[str, Any] = {"revision_feedback": feedback}
                        if _accepts_kwarg(manager, "revision_attempt"):
                            revise_kwargs["revision_attempt"] = 1
                        revised = manager.propose_next_trial(
                            status, memory_context, node_spec, **revise_kwargs
                        )
                        compute_budget = compute_budget.consume_llm_call()
                        revised = _proposal_with_campaign_metadata(
                            _proposal_with_worker_memory(revised, memory_context=memory_context),
                            node_spec=node_spec,
                        )
                        control_meta = {
                            "epoch": "E1",
                            "action": "revise",
                            "revision": {
                                "feedback": feedback,
                                "original": proposal_digest(proposal),
                                "revised": proposal_digest(revised),
                            },
                        }
                        emit(
                            event_store, campaign_id=campaign_id, trial_id=trial_id,
                            event_type="proposal_revised",
                            payload={"summary": revised.proposal_summary, "feedback": feedback},
                        )
                        _emit_llm_call(event_store, campaign_id=campaign_id, trial_id=trial_id, proposal=revised)
                        proposal = revised
                    emit(
                        event_store, campaign_id=campaign_id, trial_id=trial_id,
                        event_type="worker_started",
                        payload={"worker_mode": getattr(worker, "mode", "unknown_worker")},
                    )
                    recheck = _precondition_failure_result(proposal, node_spec, worker)
                    if recheck is not None:
                        worker_result = recheck
                    else:
                        worker_run_t0 = time.perf_counter()
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
                        compute_budget = compute_budget.consume_worker_run(time.perf_counter() - worker_run_t0)
                    emit(
                        event_store, campaign_id=campaign_id, trial_id=trial_id,
                        event_type="worker_finished",
                        payload={"success": worker_result.success,
                                 "elapsed_seconds": max(time.perf_counter() - worker_started, 0.0)},
                    )
        ts_end = _iso_now()

        candidate_effective: dict[str, str] | None = None
        if skip_trial:
            record = _skipped_record(
                campaign_id=campaign_id,
                budget_index=budget_index,
                node_spec=node_spec,
                proposal=proposal,
                memory_mode=memory_mode,
                current_best=current_best,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                control=control_meta,
                rationale=(
                    f"Skipped at E1: proposal named an inactive key "
                    f"({control_meta.get('key')}); revisions exhausted."
                    if control_meta.get("reason") == "inactive_key"
                    else None
                ),
            )
            rbs = compute_repeated_bad_stats(existing_records + records + [record])
            record = replace(record, repeated_bad_count=rbs.repeated_bad_count)
            records.append(record)
        else:
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
            # D27: record-only novelty flags on the evaluated trial.  Emitted
            # as its own event so TrialRecord bytes stay unchanged.
            candidate_effective = _effective_config_at(worker, node_spec)
            if candidate_effective is not None:
                candidate_metric = record.parsed_metrics.get(node_spec.metric_name)
                emit(
                    event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    event_type="edit_novelty",
                    payload={
                        "config_equals_incumbent": (
                            incumbent_effective is not None
                            and candidate_effective == incumbent_effective
                        ),
                        "config_seen_before": any(
                            candidate_effective == seen for seen in seen_effective
                        ),
                        "metric_identical_to_incumbent": (
                            candidate_metric is not None
                            and current_best is not None
                            and float(candidate_metric) == float(current_best)
                        ),
                    },
                )
                seen_effective.append(candidate_effective)
            # Compute cumulative repeated-bad count for this trial (including itself)
            # and stamp it onto the immutable record before persisting.
            rbs = compute_repeated_bad_stats(existing_records + records + [record])
            record = replace(record, repeated_bad_count=rbs.repeated_bad_count)
            _invoke_snapshot_hook(
                snapshot_hook,
                epoch="E2", budget_index=budget_index, campaign_id=campaign_id, trial_id=trial_id,
                node_spec=node_spec, worker=worker, proposal=proposal, provisional=record,
                worker_result=worker_result, incumbent_snapshot=editable_snapshot,
                ledger_prefix=existing_records + records, current_best=current_best,
                incumbent_stack=incumbent_stack, budget_state=budget_state,
                compute_budget=compute_budget, controller=controller, manager=manager,
                memory_mode=memory_mode, manager_mode=manager_mode, t_min=t_min,
                verify_runs=verify_runs, verify_split_policy=verify_split_policy,
            )
            e2_decision, e2_mask = _controller_hook(
                controller=controller,
                log_default_decisions=log_default_decisions,
                event_store=event_store,
                campaign_id=campaign_id,
                trial_id=trial_id,
                epoch="E2",
                budget_index=budget_index,
                ledger_prefix=existing_records + records,
                current_best=current_best,
                incumbent_stack_depth=len(incumbent_stack),
                proposal=proposal,
                provisional=record,
                budget_state=budget_state,
                compute_budget=compute_budget,
                node_spec=node_spec,
                seed_injection_supported=_supports_seed_injection(worker),
                verify_runs=verify_runs,
                worker_runs_remaining=compute_budget.remaining()["worker_runs_remaining"],
            )
            if e2_decision.llm_calls:
                compute_budget = compute_budget.consume_llm_call(e2_decision.llm_calls)
            e2_action = _resolve_action(
                e2_decision, e2_mask, epoch="E2", event_store=event_store,
                campaign_id=campaign_id, trial_id=trial_id,
            )
            if e2_action == "verify":
                record, verify_result = _run_verify(
                    record=record,
                    proposal=proposal,
                    node_spec=node_spec,
                    worker=worker,
                    incumbent_snapshot=editable_snapshot,
                    base_budget_index=budget_index,
                    verify_runs=verify_runs,
                    compute_budget=compute_budget,
                    verify_split_policy=verify_split_policy,
                )
                compute_budget = verify_result["compute_budget"]
                control_meta = {**control_meta, "verify": verify_result["verify"]}
            if control_meta:
                record = replace(record, extra={**record.extra, "control": control_meta})
            records.append(record)

        if ungoverned:
            # UNGOVERNED: write an observation entry regardless of outcome,
            # but only append to the governed ledger if the trial would have
            # been kept or discarded (i.e. valid).  Failed-invalid trials are
            # silently dropped — exactly what a system without a governed
            # ledger would do.
            _write_ungoverned_obs(obs_path, record, trial_id, budget_index)
            if record.decision != TrialDecision.FAILED_INVALID:
                store.append_many([record])
                emit(
                    event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    event_type="trial_record_appended",
                    payload={"trial_id": trial_id, "repeated_bad_count": rbs.repeated_bad_count,
                             "ungoverned": True},
                )
            else:
                emit(
                    event_store,
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    event_type="trial_record_dropped_ungoverned",
                    payload={"trial_id": trial_id,
                             "failure_category": record.failure_category.value if record.failure_category else None},
                )
            # No pending guard to release.
        else:
            store.append_many([record])
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

        if record.decision != TrialDecision.KEPT:
            _restore_editable_state(editable_snapshot)

        if record.decision == TrialDecision.KEPT and node_spec.metric_name in record.parsed_metrics:
            current_best = record.parsed_metrics[node_spec.metric_name]
        if record.decision == TrialDecision.KEPT and candidate_effective is not None:
            incumbent_effective = candidate_effective
        if record.decision == TrialDecision.KEPT:
            incumbent_stack.append(
                _IncumbentEntry(
                    snapshot=editable_snapshot,
                    current_best_before=record.current_best_before,
                    metric=record.parsed_metrics.get(node_spec.metric_name),
                    verified=bool(control_meta.get("verify", {}).get("response") == "keep"),
                )
            )
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
    # In ungoverned mode only valid (kept/discarded) trials reach the ledger;
    # failed_invalid trials are silently dropped.  Report what's actually there.
    if ungoverned:
        ledger_written = sum(
            1 for r in records if r.decision != TrialDecision.FAILED_INVALID
        )
    else:
        ledger_written = len(records)

    return CampaignRunResult(
        campaign_id=campaign_id,
        records_path=str(records_path),
        records_written=ledger_written,
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
    controller: Controller | None = None,
    log_default_decisions: bool = False,
    compute_budget: ComputeBudget | None = None,
) -> CampaignRunResult:
    """Run a fixed-budget campaign with the synthetic ``DryRunWorker``.

    The controller hooks run at E0/E1/E2 as in :func:`run_real_campaign`, but
    the dry campaign supports only the ``continue`` action.  Any other action
    raises ``NotImplementedError`` by design: skip, revise, verify, demote and
    stop are implemented in :func:`run_real_campaign` only.  LLM calls are
    counted only for non-deterministic managers.
    """
    manager = proposal_backend or _manager(manager_mode, llm=manager_llm, temperature=manager_temperature)
    worker = DryRunWorker(profile=dry_run_profile)
    controller = controller or ContinueController()
    compute_budget = compute_budget or ComputeBudget()
    store = TrialAppendStore(records_path)
    existing_records = store.read_all()
    budget_state = BudgetState(total_trials=budget, consumed_trials=len(existing_records))
    current_best = _current_best(existing_records, node_spec.metric_name, node_spec.metric_direction)

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
    while not budget_state.exhausted and not compute_budget.exhausted:
        budget_index = budget_state.next_budget_index
        trial_id = f"{campaign_id}-trial-{budget_index:03d}"
        emit(
            event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type="trial_started",
            payload={"trial_id": trial_id, "budget_index": budget_index},
        )
        e0_decision, _e0_mask = _controller_hook(
            controller=controller,
            log_default_decisions=log_default_decisions,
            event_store=event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            epoch="E0",
            budget_index=budget_index,
            ledger_prefix=existing_records + records,
            current_best=current_best,
            incumbent_stack_depth=0,
            proposal=None,
            provisional=None,
            budget_state=budget_state,
            compute_budget=compute_budget,
            node_spec=node_spec,
            top_of_stack_unverified=False,
        )
        if e0_decision.llm_calls:
            compute_budget = compute_budget.consume_llm_call(e0_decision.llm_calls)
        if e0_decision.action != "continue":
            raise NotImplementedError(
                f"E0 action {e0_decision.action!r}: run_dry_campaign supports only "
                "'continue' (by design); use run_real_campaign"
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
        if not getattr(manager, "deterministic", False):
            # Only LLM-backed managers make a call; deterministic managers
            # (baseline, prompt) consume no LLM budget.
            compute_budget = compute_budget.consume_llm_call()
        _emit_llm_call(event_store, campaign_id=campaign_id, trial_id=trial_id, proposal=proposal)
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
        e1_decision, _e1_mask = _controller_hook(
            controller=controller,
            log_default_decisions=log_default_decisions,
            event_store=event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            epoch="E1",
            budget_index=budget_index,
            ledger_prefix=existing_records + records,
            current_best=current_best,
            incumbent_stack_depth=0,
            proposal=proposal,
            provisional=None,
            budget_state=budget_state,
            compute_budget=compute_budget,
            node_spec=node_spec,
            proposal_passed_precondition=True,
            manager_deterministic=bool(getattr(manager, "deterministic", False)),
            revisions_this_trial=0,
            llm_cap_reached=(
                compute_budget.max_llm_calls is not None
                and compute_budget.llm_calls >= compute_budget.max_llm_calls
            ),
        )
        if e1_decision.llm_calls:
            compute_budget = compute_budget.consume_llm_call(e1_decision.llm_calls)
        if e1_decision.action != "continue":
            raise NotImplementedError(
                f"E1 action {e1_decision.action!r}: run_dry_campaign supports only "
                "'continue' (by design); use run_real_campaign"
            )
        worker_run_t0 = time.perf_counter()
        worker_result = worker.run_trial(proposal, node_spec, budget_index)
        compute_budget = compute_budget.consume_worker_run(time.perf_counter() - worker_run_t0)
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
        e2_decision, _e2_mask = _controller_hook(
            controller=controller,
            log_default_decisions=log_default_decisions,
            event_store=event_store,
            campaign_id=campaign_id,
            trial_id=trial_id,
            epoch="E2",
            budget_index=budget_index,
            ledger_prefix=existing_records + records,
            current_best=current_best,
            incumbent_stack_depth=0,
            proposal=proposal,
            provisional=record,
            budget_state=budget_state,
            compute_budget=compute_budget,
            node_spec=node_spec,
            seed_injection_supported=_supports_seed_injection(worker),
            worker_runs_remaining=compute_budget.remaining()["worker_runs_remaining"],
        )
        if e2_decision.llm_calls:
            compute_budget = compute_budget.consume_llm_call(e2_decision.llm_calls)
        if e2_decision.action != "continue":
            raise NotImplementedError(
                f"E2 action {e2_decision.action!r}: run_dry_campaign supports only "
                "'continue' (by design); use run_real_campaign"
            )
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


def _write_ungoverned_obs(obs_path: "Path | None", record: "TrialRecord", trial_id: str, budget_index: int) -> None:
    """Append a lightweight observation entry to the ungoverned observation log.

    This captures what actually happened on disk during an ungoverned run,
    for comparison with the governed ledger in the paper's Level 2 analysis.
    The observation log is NOT the governed ledger — it contains only the
    raw execution outcome, with no provenance IDs, no taxonomy, and no
    decision authority.
    """
    if obs_path is None:
        return
    obs = {
        "trial_id": trial_id,
        "budget_index": budget_index,
        "timestamp_start": record.timestamp_start,
        "timestamp_end": record.timestamp_end,
        "wall_clock_seconds": record.wall_clock_seconds,
        "worker_success": record.execution_status.value == "success",
        "failure_message": (record.extra or {}).get("worker_failure_message"),
        # What governance would have classified this as — but is NOT recorded
        # in the ungoverned ledger:
        "governance_would_have_classified": {
            "decision": record.decision.value,
            "failure_category": record.failure_category.value if record.failure_category else None,
            "validity_status": record.validity_status.value,
        },
        "ungoverned_ledger_entry": record.decision.value != "failed_invalid",
        "note": (
            "This trial was dropped silently (not recorded in ledger)"
            if record.decision.value == "failed_invalid"
            else "This trial was recorded in the ungoverned ledger"
        ),
    }
    with open(obs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")


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


def _current_best(
    records: list[TrialRecord],
    metric_name: str,
    metric_direction: str = "maximize",
) -> float | None:
    """Best metric over *kept* records, respecting the node's direction.

    Only records with ``decision == KEPT`` count: invalid records may still
    carry a parsed metric (for audit) but never define the incumbent.  On
    minimise nodes the best value is the minimum.
    """
    values = [
        record.parsed_metrics[metric_name]
        for record in records
        if record.decision == TrialDecision.KEPT and metric_name in record.parsed_metrics
    ]
    if not values:
        return None
    return max(values) if metric_direction == "maximize" else min(values)
