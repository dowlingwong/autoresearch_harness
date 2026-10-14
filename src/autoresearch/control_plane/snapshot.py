"""Campaign snapshots, restore and branching (BUILD_RUNBOOK L7).

An optional ``snapshot_hook`` on ``run_real_campaign`` is invoked at E0/E1/E2
with the live campaign state.  ``SnapshotRecorder`` implements that hook and
writes a :class:`CampaignSnapshot` to disk: the editable bytes, the declared
non-editable node-state paths, the ledger prefix, the explicit ``current_best``
and incumbent stack, the budget and compute-budget states, the seed settings,
the manager config and the controller identity, plus the epoch-local proposal
and provisional record.  Every layer gets a SHA-256 hash in a manifest.

``restore_snapshot`` materialises a snapshot into a fresh, isolated node
directory (``copytree``) with its own ledger and event paths.  ``resume_campaign``
completes the captured trial with a forced first action and then hands off to
``run_real_campaign`` for a horizon of further trials.  Siblings never share
writable paths.

Everything here is ENGINEERING_ONLY.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoresearch.control_plane.budget import BudgetState, ComputeBudget
from autoresearch.control_plane import campaign as _campaign
from autoresearch.control_plane.campaign import SnapshotContext, _IncumbentEntry
from autoresearch.control_plane.controller import (
    ContinueController,
    ControlDecision,
    ControlView,
    Controller,
    build_control_view,
    compute_mask,
)
from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import TrialDecision, TrialRecord
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult
from autoresearch.worker.local_worker import LocalWorker

DEFAULT_BRANCH_STATE_PATHS: tuple[str, ...] = (
    "artifacts",
    "results.tsv",
    "experiment_memory.jsonl",
    ".autoresearch_artifacts",
    "run.log",
)

MANIFEST_SCHEMA = "r1.campaign_snapshot/1"


@dataclass
class StackEntrySnapshot:
    files: dict[str, bytes | None]
    current_best_before: float | None
    metric: float | None
    verified: bool


@dataclass
class CampaignSnapshot:
    node_name: str
    campaign_id: str
    epoch: str
    budget_index: int
    node_source_root: str
    editable_paths: tuple[str, ...]
    branch_state_paths: tuple[str, ...]
    editable_files: dict[str, bytes | None]
    incumbent_files: dict[str, bytes | None]
    node_state_blobs: dict[str, bytes]
    ledger_prefix: tuple[TrialRecord, ...]
    current_best: float | None
    incumbent_stack: tuple[StackEntrySnapshot, ...]
    budget_total: int
    budget_consumed: int
    compute_budget_state: dict[str, Any]
    train_seed: Any
    split_seed: Any
    manager: dict[str, Any]
    manager_mode: str
    memory_mode: str
    controller_version: str
    t_min: int
    verify_runs: int
    verify_split_policy: str
    proposal: ManagerProposal | None
    provisional: TrialRecord | None
    worker_result: WorkerResult | None
    worker_mode: str
    # D27 F3 fix: hashes of every config evaluated so far (full effective
    # config and its active-key projection).  ``None`` means the snapshot was
    # taken before this field existed: a resume records "unknown before
    # resume" and never treats it as an empty set.
    seen_effective_hashes: tuple[str, ...] | None = None
    seen_active_hashes: tuple[str, ...] | None = None
    layer_hashes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _hash_file_map(mapping: dict[str, bytes | None]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(mapping):
        data = mapping[rel]
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"<absent>" if data is None else hashlib.sha256(data).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_records(records: tuple[TrialRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = json.dumps(record.to_dict(), sort_keys=True, default=str)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _branch_state_paths(node_spec: NodeSpec) -> tuple[str, ...]:
    declared = tuple(getattr(node_spec, "branch_state_paths", ()) or ())
    return declared or DEFAULT_BRANCH_STATE_PATHS


def _collect_node_state(node_root: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for rel in paths:
        target = node_root / rel
        if target.is_file():
            blobs[rel] = target.read_bytes()
        elif target.is_dir():
            for file in sorted(target.rglob("*")):
                if file.is_file():
                    blobs[str(file.relative_to(node_root))] = file.read_bytes()
    return blobs


# ---------------------------------------------------------------------------
# Serialisation of small objects
# ---------------------------------------------------------------------------

def _proposal_to_dict(proposal: ManagerProposal | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "manager_mode": proposal.manager_mode,
        "proposal_summary": proposal.proposal_summary,
        "proposal_rationale": proposal.proposal_rationale,
        "target_files": list(proposal.target_files),
        "objective": proposal.objective,
        "extra": json.loads(json.dumps(proposal.extra, default=str)),
    }


def _proposal_from_dict(payload: dict[str, Any] | None) -> ManagerProposal | None:
    if payload is None:
        return None
    return ManagerProposal(
        manager_mode=str(payload["manager_mode"]),
        proposal_summary=str(payload["proposal_summary"]),
        proposal_rationale=str(payload.get("proposal_rationale", "")),
        target_files=tuple(str(p) for p in payload.get("target_files", ())),
        objective=str(payload.get("objective", "")),
        extra=dict(payload.get("extra", {})),
    )


def _worker_result_to_dict(result: WorkerResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = result.to_dict()
    payload["extra"] = json.loads(json.dumps(result.extra, default=str))
    return payload


def _worker_result_from_dict(payload: dict[str, Any] | None) -> WorkerResult | None:
    if payload is None:
        return None
    return WorkerResult(
        worker_mode=str(payload["worker_mode"]),
        changed_files=tuple(str(f) for f in payload.get("changed_files", ())),
        success=bool(payload["success"]),
        parsed_metrics={str(k): float(v) for k, v in (payload.get("parsed_metrics") or {}).items()},
        raw_log_ref=str(payload.get("raw_log_ref", "")),
        patch_ref=str(payload.get("patch_ref", "")),
        git_commit_before=str(payload.get("git_commit_before", "")),
        git_commit_after=str(payload.get("git_commit_after", "")),
        failure_message=payload.get("failure_message"),
        extra=dict(payload.get("extra", {})),
    )


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_snapshot(
    context: SnapshotContext,
    *,
    model_digest: str | None = None,
) -> CampaignSnapshot:
    node_root = _campaign._worker_node_root(context.worker)
    if node_root is None:
        raise ValueError("cannot snapshot a worker with no node_root")
    editable_paths = tuple(context.node_spec.editable_paths)
    branch_paths = _branch_state_paths(context.node_spec)

    editable_files = _campaign._snapshot_editable_state(context.worker, context.node_spec)
    editable_map = dict(editable_files.contents) if editable_files is not None else {}
    incumbent_map = (
        dict(context.incumbent_snapshot.contents) if context.incumbent_snapshot is not None else dict(editable_map)
    )
    node_state = _collect_node_state(node_root, branch_paths)

    stack_entries: list[StackEntrySnapshot] = []
    for entry in context.incumbent_stack:
        files = dict(entry.snapshot.contents) if entry.snapshot is not None else {}
        stack_entries.append(
            StackEntrySnapshot(
                files=files,
                current_best_before=entry.current_best_before,
                metric=entry.metric,
                verified=bool(entry.verified),
            )
        )

    manager = context.manager
    manager_cfg = {
        "mode": getattr(manager, "mode", context.manager_mode),
        "deterministic": bool(getattr(manager, "deterministic", False)),
        "model": getattr(manager, "_model", None),
        "temperature": getattr(manager, "_temperature", None),
        "model_digest": model_digest,
    }

    snapshot = CampaignSnapshot(
        node_name=context.node_spec.name,
        campaign_id=context.campaign_id,
        epoch=context.epoch,
        budget_index=context.budget_index,
        node_source_root=str(node_root),
        editable_paths=editable_paths,
        branch_state_paths=branch_paths,
        editable_files=editable_map,
        incumbent_files=incumbent_map,
        node_state_blobs=node_state,
        ledger_prefix=tuple(context.ledger_prefix),
        current_best=context.current_best,
        incumbent_stack=tuple(stack_entries),
        budget_total=context.budget_state.total_trials,
        budget_consumed=context.budget_state.consumed_trials,
        compute_budget_state=context.compute_budget.to_dict(),
        train_seed=getattr(context.worker, "_train_seed", None),
        split_seed=getattr(context.worker, "_split_seed", None),
        manager=manager_cfg,
        manager_mode=context.manager_mode,
        memory_mode=context.memory_mode,
        controller_version=getattr(context.controller, "version", type(context.controller).__name__),
        t_min=context.t_min,
        verify_runs=context.verify_runs,
        verify_split_policy=context.verify_split_policy,
        proposal=context.proposal,
        provisional=context.provisional,
        worker_result=context.worker_result,
        worker_mode=getattr(context.worker, "mode", type(context.worker).__name__),
        seen_effective_hashes=tuple(context.seen_effective_hashes),
        seen_active_hashes=tuple(context.seen_active_hashes),
    )
    snapshot.layer_hashes = _layer_hashes(snapshot)
    return snapshot


def _layer_hashes(snapshot: CampaignSnapshot) -> dict[str, str]:
    layers = {
        "editable": _hash_file_map(snapshot.editable_files),
        "incumbent": _hash_file_map(snapshot.incumbent_files),
        "node_state": _hash_file_map(snapshot.node_state_blobs),
        "ledger_prefix": _hash_records(snapshot.ledger_prefix),
    }
    for index, entry in enumerate(snapshot.incumbent_stack):
        layers[f"stack[{index}]"] = _hash_file_map(entry.files)
    return layers


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------

def _write_bytes_tree(root: Path, mapping: dict[str, bytes | None]) -> None:
    for rel, data in mapping.items():
        if data is None:
            continue
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _read_bytes_tree(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    if not root.exists():
        return result
    for file in sorted(root.rglob("*")):
        if file.is_file():
            result[str(file.relative_to(root))] = file.read_bytes()
    return result


def write_snapshot(snapshot: CampaignSnapshot, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_bytes_tree(out / "editable", snapshot.editable_files)
    _write_bytes_tree(out / "incumbent", snapshot.incumbent_files)
    _write_bytes_tree(out / "node_state", snapshot.node_state_blobs)
    for index, entry in enumerate(snapshot.incumbent_stack):
        _write_bytes_tree(out / "stack" / str(index), entry.files)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "label": "ENGINEERING_ONLY",
        "node_name": snapshot.node_name,
        "campaign_id": snapshot.campaign_id,
        "epoch": snapshot.epoch,
        "budget_index": snapshot.budget_index,
        "node_source_root": snapshot.node_source_root,
        "editable_paths": list(snapshot.editable_paths),
        "branch_state_paths": list(snapshot.branch_state_paths),
        "current_best": snapshot.current_best,
        "budget_total": snapshot.budget_total,
        "budget_consumed": snapshot.budget_consumed,
        "compute_budget_state": snapshot.compute_budget_state,
        "train_seed": snapshot.train_seed,
        "split_seed": snapshot.split_seed,
        "manager": snapshot.manager,
        "manager_mode": snapshot.manager_mode,
        "memory_mode": snapshot.memory_mode,
        "controller_version": snapshot.controller_version,
        "t_min": snapshot.t_min,
        "verify_runs": snapshot.verify_runs,
        "verify_split_policy": snapshot.verify_split_policy,
        "worker_mode": snapshot.worker_mode,
        "seen_effective_hashes": (
            None if snapshot.seen_effective_hashes is None else list(snapshot.seen_effective_hashes)
        ),
        "seen_active_hashes": (
            None if snapshot.seen_active_hashes is None else list(snapshot.seen_active_hashes)
        ),
        "layer_hashes": snapshot.layer_hashes,
        "ledger_prefix": [record.to_dict() for record in snapshot.ledger_prefix],
        "incumbent_stack": [
            {
                "current_best_before": entry.current_best_before,
                "metric": entry.metric,
                "verified": entry.verified,
                "layer_hash": _hash_file_map(entry.files),
            }
            for entry in snapshot.incumbent_stack
        ],
        "proposal": _proposal_to_dict(snapshot.proposal),
        "provisional": snapshot.provisional.to_dict() if snapshot.provisional else None,
        "worker_result": _worker_result_to_dict(snapshot.worker_result),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def read_snapshot(out_dir: str | Path) -> CampaignSnapshot:
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    editable = _read_bytes_tree(out / "editable")
    incumbent = _read_bytes_tree(out / "incumbent")
    node_state = _read_bytes_tree(out / "node_state")

    stack: list[StackEntrySnapshot] = []
    for entry_meta in manifest.get("incumbent_stack", []):
        index = len(stack)
        stack.append(
            StackEntrySnapshot(
                files=_read_bytes_tree(out / "stack" / str(index)),
                current_best_before=entry_meta.get("current_best_before"),
                metric=entry_meta.get("metric"),
                verified=bool(entry_meta.get("verified", False)),
            )
        )

    ledger_prefix = tuple(TrialRecord.from_mapping(item) for item in manifest.get("ledger_prefix", []))
    provisional = (
        TrialRecord.from_mapping(manifest["provisional"]) if manifest.get("provisional") else None
    )
    # D27 F3 fix: a manifest without the sets is an old-format snapshot; its
    # resume must report "unknown before resume", never a silent empty set.
    seen_effective = manifest.get("seen_effective_hashes")
    seen_active = manifest.get("seen_active_hashes")
    return CampaignSnapshot(
        node_name=str(manifest["node_name"]),
        campaign_id=str(manifest["campaign_id"]),
        epoch=str(manifest["epoch"]),
        budget_index=int(manifest["budget_index"]),
        node_source_root=str(manifest["node_source_root"]),
        editable_paths=tuple(str(p) for p in manifest.get("editable_paths", ())),
        branch_state_paths=tuple(str(p) for p in manifest.get("branch_state_paths", ())),
        editable_files=editable,
        incumbent_files=incumbent,
        node_state_blobs=node_state,
        ledger_prefix=ledger_prefix,
        current_best=manifest.get("current_best"),
        incumbent_stack=tuple(stack),
        budget_total=int(manifest["budget_total"]),
        budget_consumed=int(manifest["budget_consumed"]),
        compute_budget_state=dict(manifest.get("compute_budget_state", {})),
        train_seed=manifest.get("train_seed"),
        split_seed=manifest.get("split_seed"),
        manager=dict(manifest.get("manager", {})),
        manager_mode=str(manifest.get("manager_mode", "baseline_manager")),
        memory_mode=str(manifest.get("memory_mode", "none")),
        controller_version=str(manifest.get("controller_version", "unknown")),
        t_min=int(manifest.get("t_min", 3)),
        verify_runs=int(manifest.get("verify_runs", 2)),
        verify_split_policy=str(manifest.get("verify_split_policy", "fixed")),
        proposal=_proposal_from_dict(manifest.get("proposal")),
        provisional=provisional,
        worker_result=_worker_result_from_dict(manifest.get("worker_result")),
        worker_mode=str(manifest.get("worker_mode", "local_worker")),
        seen_effective_hashes=None if seen_effective is None else tuple(str(h) for h in seen_effective),
        seen_active_hashes=None if seen_active is None else tuple(str(h) for h in seen_active),
        layer_hashes=dict(manifest.get("layer_hashes", {})),
    )


def snapshot_state_hashes(snapshot: CampaignSnapshot) -> dict[str, str]:
    """Recompute the per-layer hashes from the snapshot's in-memory bytes."""
    return _layer_hashes(snapshot)


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

@dataclass
class RestoredCampaign:
    into_dir: Path
    node_spec: NodeSpec
    records_path: Path
    events_path: Path
    worker: Any
    manager: Any
    current_best: float | None
    incumbent_stack: list[_IncumbentEntry]
    budget_total: int
    budget_consumed: int
    compute_budget: ComputeBudget
    snapshot: CampaignSnapshot


def _build_manager(snapshot: CampaignSnapshot, manager_override: Any | None) -> Any:
    if manager_override is not None:
        return manager_override
    mode = snapshot.manager_mode
    if mode in ("baseline_manager", "prompt_manager"):
        return _campaign._manager(mode)
    if mode == "langgraph_manager":
        from autoresearch.manager.langgraph_manager import LangGraphManager

        cfg = snapshot.manager or {}
        model = cfg.get("model") or "qwen2.5-coder:7b"
        return LangGraphManager(model=model, temperature=float(cfg.get("temperature") or 0.2))
    raise ValueError(f"cannot rebuild manager for mode {mode!r}; pass manager=")


def _restore_incumbent_stack(snapshot: CampaignSnapshot, into_dir: Path) -> list[_IncumbentEntry]:
    stack: list[_IncumbentEntry] = []
    for entry in snapshot.incumbent_stack:
        snapshot_obj = _campaign._EditableStateSnapshot(root=into_dir, contents=dict(entry.files))
        stack.append(
            _IncumbentEntry(
                snapshot=snapshot_obj,
                current_best_before=entry.current_best_before,
                metric=entry.metric,
                verified=entry.verified,
            )
        )
    return stack


def restore_snapshot(
    snapshot: CampaignSnapshot,
    into_dir: str | Path,
    *,
    worker: Any | None = None,
    manager: Any | None = None,
    events_path: str | Path | None = None,
) -> RestoredCampaign:
    into = Path(into_dir)
    source = Path(snapshot.node_source_root)
    ignore = shutil.ignore_patterns("__pycache__", ".git", ".autoresearch_artifacts", *snapshot.branch_state_paths)
    shutil.copytree(source, into, dirs_exist_ok=True, ignore=ignore)

    # Editable files: the snapshot's current workspace state (E1 incumbent / E2 candidate).
    for rel, data in snapshot.editable_files.items():
        target = into / rel
        if data is None:
            if target.exists():
                target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Non-editable node-state blobs.
    _write_bytes_tree(into, snapshot.node_state_blobs)

    node_spec = load_registered_node(snapshot.node_name)
    records_path = into / "branch_trials.jsonl"
    if events_path is None:
        events_path = into / "branch_events.jsonl"

    if worker is None:
        if snapshot.worker_mode != "local_worker":
            raise ValueError(f"cannot rebuild worker for mode {snapshot.worker_mode!r}; pass worker=")
        worker = LocalWorker(
            node_root=into,
            artifact_dir=into / ".autoresearch_artifacts",
            train_seed=snapshot.train_seed,
            split_seed=snapshot.split_seed,
        )
    restored_manager = _build_manager(snapshot, manager)

    # Write the ledger prefix so the resumed campaign resumes from it.
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.ledger_prefix:
        store = TrialAppendStore(records_path)
        store.append_many(list(snapshot.ledger_prefix))

    compute_budget = ComputeBudget(
        max_worker_runs=snapshot.compute_budget_state.get("max_worker_runs"),
        max_worker_seconds=snapshot.compute_budget_state.get("max_worker_seconds"),
        max_llm_calls=snapshot.compute_budget_state.get("max_llm_calls"),
        worker_runs=int(snapshot.compute_budget_state.get("worker_runs", 0)),
        worker_seconds=float(snapshot.compute_budget_state.get("worker_seconds", 0.0)),
        llm_calls=int(snapshot.compute_budget_state.get("llm_calls", 0)),
    )
    return RestoredCampaign(
        into_dir=into,
        node_spec=node_spec,
        records_path=records_path,
        events_path=Path(events_path),
        worker=worker,
        manager=restored_manager,
        current_best=snapshot.current_best,
        incumbent_stack=_restore_incumbent_stack(snapshot, into),
        budget_total=snapshot.budget_total,
        budget_consumed=snapshot.budget_consumed,
        compute_budget=compute_budget,
        snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Snapshot recorder (the hook)
# ---------------------------------------------------------------------------

class SnapshotRecorder:
    """A ``snapshot_hook`` that writes the Nth snapshot at selected epochs."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        epochs: tuple[str, ...] = ("E1", "E2"),
        take_index: int = 1,
        model_digest: str | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.epochs = tuple(epochs)
        self.take_index = take_index
        self.model_digest = model_digest
        self.snapshots: list[CampaignSnapshot] = []
        self.manifest_path: Path | None = None
        self._seen = 0

    def __call__(self, context: SnapshotContext) -> None:
        if context.epoch not in self.epochs:
            return
        self._seen += 1
        if self._seen != self.take_index:
            return
        snapshot = capture_snapshot(context, model_digest=self.model_digest)
        target = self.out_dir
        target.mkdir(parents=True, exist_ok=True)
        self.manifest_path = write_snapshot(snapshot, target)
        self.snapshots.append(snapshot)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class FirstActionThenPolicy:
    """Controller: force one action, then delegate to *policy*."""

    def __init__(self, *, epoch: str, budget_index: int, action: str, policy: Controller | None = None) -> None:
        self.epoch = epoch
        self.budget_index = budget_index
        self.action = action
        self.policy = policy or ContinueController()
        self.version = f"first-action/{action}"
        self._used = False

    def decide(self, view: ControlView, mask: dict[str, bool]) -> ControlDecision:
        if not self._used and view.epoch == self.epoch and view.budget_index == self.budget_index:
            self._used = True
            return ControlDecision(
                action=self.action,
                controller_version=self.version,
                input_hash=view.input_hash(),
            )
        return self.policy.decide(view, mask)


def _resolve_first_action(
    *,
    epoch: str,
    budget_index: int,
    action: str,
    snapshot: CampaignSnapshot,
    node_spec: NodeSpec,
    ledger_prefix: list[TrialRecord],
    current_best: float | None,
    incumbent_stack_depth: int,
    proposal: ManagerProposal | None,
    provisional: TrialRecord | None,
    budget_state: BudgetState,
    compute_budget: ComputeBudget,
    event_store: object | None,
    campaign_id: str,
    trial_id: str,
    manager: object,
    worker: object,
) -> str:
    view = build_control_view(
        epoch=epoch,  # type: ignore[arg-type]
        budget_index=budget_index,
        ledger_prefix=ledger_prefix,
        current_best=current_best,
        incumbent_stack_depth=incumbent_stack_depth,
        proposal=proposal,
        provisional=provisional,
        budget={"trials_remaining": budget_state.total_trials - budget_state.consumed_trials},
        node=node_spec,
    )
    mask = compute_mask(
        view,
        proposal_passed_precondition=True,
        manager_deterministic=bool(getattr(manager, "deterministic", False)),
        revision_feedback_supported=_campaign._accepts_revision_feedback(manager),
        seed_injection_supported=_campaign._supports_seed_injection(worker),
        t_min=snapshot.t_min,
        verify_runs=snapshot.verify_runs,
        worker_runs_remaining=compute_budget.remaining().get("worker_runs_remaining"),
        llm_cap_reached=bool(compute_budget.max_llm_calls is not None
                             and compute_budget.llm_calls >= compute_budget.max_llm_calls),
        top_of_stack_unverified=bool(incumbent_stack_depth and snapshot.incumbent_stack
                                     and not snapshot.incumbent_stack[-1].verified),
    )
    return _campaign._resolve_action(
        ControlDecision(action=action, controller_version="forced", input_hash=view.input_hash()),
        mask,
        epoch=epoch,
        event_store=event_store,
        campaign_id=campaign_id,
        trial_id=trial_id,
    )


def _complete_captured_trial(
    restored: RestoredCampaign,
    *,
    first_action: str,
    event_store: object | None = None,
) -> TrialRecord:
    """Execute the captured epoch's forced first action and append one record.

    At E1 the captured proposal is replayed through the worker (the manager is
    *not* called again for it).  At E2 the worker already ran, so the captured
    provisional record and worker result are committed directly.
    """
    snapshot = restored.snapshot
    node_spec = restored.node_spec
    worker = restored.worker
    manager = restored.manager
    campaign_id = snapshot.campaign_id
    budget_index = snapshot.budget_index
    trial_id = f"{campaign_id}-trial-{budget_index:03d}"
    budget_state = BudgetState(total_trials=restored.budget_total, consumed_trials=restored.budget_consumed)

    memory_context = build_memory_context(
        list(snapshot.ledger_prefix), MemoryMode(snapshot.memory_mode), node_spec, budget_index
    )
    status = _campaign._manager_status(
        campaign_id=campaign_id, budget_index=budget_index,
        current_best=restored.current_best, node_spec=node_spec, worker=worker,
    )
    proposal = snapshot.proposal
    effective = _resolve_first_action(
        epoch=snapshot.epoch,
        budget_index=budget_index,
        action=first_action,
        snapshot=snapshot,
        node_spec=node_spec,
        ledger_prefix=list(snapshot.ledger_prefix),
        current_best=restored.current_best,
        incumbent_stack_depth=len(restored.incumbent_stack),
        proposal=proposal,
        provisional=snapshot.provisional,
        budget_state=budget_state,
        compute_budget=restored.compute_budget,
        event_store=event_store,
        campaign_id=campaign_id,
        trial_id=trial_id,
        manager=manager,
        worker=worker,
    )

    incumbent_snapshot = _campaign._EditableStateSnapshot(
        root=restored.into_dir, contents=dict(snapshot.incumbent_files)
    )
    control_meta: dict[str, Any] = {"epoch": snapshot.epoch, "action": effective, "resumed": True}
    ts_start = _campaign._iso_now()

    if effective == "skip":
        record = _campaign._skipped_record(
            campaign_id=campaign_id, budget_index=budget_index, node_spec=node_spec,
            proposal=proposal, memory_mode=snapshot.memory_mode, current_best=restored.current_best,
            timestamp_start=ts_start, timestamp_end=_campaign._iso_now(), control=control_meta,
        )
    elif snapshot.epoch == "E2":
        record = snapshot.provisional
        if record is None:
            raise ValueError("E2 snapshot has no provisional record")
        if effective == "verify":
            record, verify_info = _campaign._run_verify(
                record=record, proposal=proposal, node_spec=node_spec, worker=worker,
                incumbent_snapshot=incumbent_snapshot, base_budget_index=budget_index,
                verify_runs=snapshot.verify_runs, compute_budget=restored.compute_budget,
                verify_split_policy=snapshot.verify_split_policy,
            )
            # Keep the verify evidence and charge its re-runs, exactly as the live E2 path does.
            control_meta["verify"] = verify_info["verify"]
            restored.compute_budget = verify_info["compute_budget"]
    else:
        if effective == "revise":
            feedback = _campaign._revision_feedback(memory_context)
            original = proposal
            proposal = manager.propose_next_trial(
                status, memory_context, node_spec, revision_feedback=feedback
            )
            restored.compute_budget = restored.compute_budget.consume_llm_call()
            proposal = _campaign._proposal_with_campaign_metadata(
                _campaign._proposal_with_worker_memory(proposal, memory_context=memory_context),
                node_spec=node_spec,
            )
            control_meta["revision"] = {
                "feedback": feedback,
                "original": _campaign.proposal_digest(original),
                "revised": _campaign.proposal_digest(proposal),
            }
        if proposal is None:
            raise ValueError("snapshot has no proposal to replay at this epoch")
        precondition = _campaign._precondition_failure_result(proposal, node_spec, worker)
        if precondition is not None:
            worker_result = precondition
        else:
            run_t0 = _campaign.time.perf_counter()
            try:
                worker_result = worker.run_trial(proposal, node_spec, budget_index)
            except Exception as exc:  # mirror the live loop: a worker exception is a failed edit
                worker_result = WorkerResult(
                    worker_mode=getattr(worker, "mode", "unknown_worker"), changed_files=(), success=False,
                    parsed_metrics={}, raw_log_ref="", patch_ref="", git_commit_before="",
                    git_commit_after="", failure_message=str(exc),
                    extra={"failure_category": _campaign.FailureCategory.EDIT_FAILED.value},
                )
            restored.compute_budget = restored.compute_budget.consume_worker_run(
                _campaign.time.perf_counter() - run_t0
            )
        record = _campaign._record_from_worker_result(
            campaign_id=campaign_id, budget_index=budget_index, node_spec=node_spec,
            manager_mode=proposal.manager_mode, memory_mode=snapshot.memory_mode,
            proposal_summary=proposal.proposal_summary, proposal_rationale=proposal.proposal_rationale,
            proposal_extra=proposal.extra, worker_result=worker_result,
            current_best=restored.current_best, timestamp_start=ts_start,
            timestamp_end=_campaign._iso_now(),
        )

    record = _campaign.replace(record, extra={**record.extra, "control": control_meta})
    TrialAppendStore(restored.records_path).append_many([record])
    # Commit semantics of the live loop (campaign.py): a non-keep restores the
    # incumbent's editable files; a keep pushes the incumbent onto the stack.
    if record.decision != TrialDecision.KEPT:
        _campaign._restore_editable_state(incumbent_snapshot)
    else:
        restored.incumbent_stack.append(
            _campaign._IncumbentEntry(
                snapshot=incumbent_snapshot,
                current_best_before=record.current_best_before,
                metric=record.parsed_metrics.get(node_spec.metric_name),
                verified=bool(control_meta.get("verify", {}).get("response") == "keep"),
            )
        )
    if event_store is not None:
        _campaign.emit(
            event_store, campaign_id=campaign_id, trial_id=trial_id,
            event_type="branch_trial_committed", payload={"action": effective, "epoch": snapshot.epoch},
        )
    return record


@dataclass
class ResumeResult:
    restored: RestoredCampaign
    first_record: TrialRecord | None
    continuation: Any


def resume_campaign(
    snapshot: CampaignSnapshot,
    into_dir: str | Path,
    *,
    first_action: str,
    policy: Controller | None = None,
    horizon: int = 0,
    worker: Any | None = None,
    manager: Any | None = None,
) -> ResumeResult:
    """Restore *snapshot* into a fresh dir and continue it.

    The captured epoch runs *first_action* (rejected to ``continue`` if the
    mask forbids it); then *policy* runs for *horizon* further trials.  At E1
    or E2 the captured proposal is used and the manager is not called again for
    it.
    """
    restored = restore_snapshot(snapshot, into_dir, worker=worker, manager=manager)
    event_store = CampaignEventStore(restored.events_path)

    # D27 F3 fix: seed the novelty seen-sets from the snapshot.  A snapshot
    # taken before the sets existed (old format, both ``None``) resumes with
    # "unknown before resume", recorded on the first ``edit_novelty`` event of
    # the resumed segment; it is never treated as an empty set.
    seen_effective = snapshot.seen_effective_hashes
    seen_active = snapshot.seen_active_hashes
    resume_seeds: dict[str, Any] = {
        "initial_seen_effective_hashes": seen_effective or (),
        "initial_seen_active_hashes": seen_active or (),
        "seen_before_resume_unknown": seen_effective is None or seen_active is None,
    }

    if snapshot.epoch == "E0":
        controller = FirstActionThenPolicy(
            epoch="E0", budget_index=snapshot.budget_index, action=first_action, policy=policy
        )
        continuation = _campaign.run_real_campaign(
            node_spec=restored.node_spec,
            campaign_id=snapshot.campaign_id,
            budget=restored.budget_consumed + horizon + 1,
            manager_mode=snapshot.manager_mode,
            memory_mode=snapshot.memory_mode,
            records_path=restored.records_path,
            worker=restored.worker,
            proposal_backend=restored.manager,
            event_store=event_store,
            controller=controller,
            compute_budget=restored.compute_budget,
            verify_runs=snapshot.verify_runs,
            verify_split_policy=snapshot.verify_split_policy,
            initial_current_best=restored.current_best,
            initial_incumbent_stack=restored.incumbent_stack,
            **resume_seeds,
        )
        return ResumeResult(restored=restored, first_record=None, continuation=continuation)

    first_record = _complete_captured_trial(restored, first_action=first_action, event_store=event_store)
    continuation = None
    if horizon > 0:
        continuation = _campaign.run_real_campaign(
            node_spec=restored.node_spec,
            campaign_id=snapshot.campaign_id,
            budget=restored.budget_consumed + 1 + horizon,
            manager_mode=snapshot.manager_mode,
            memory_mode=snapshot.memory_mode,
            records_path=restored.records_path,
            worker=restored.worker,
            proposal_backend=restored.manager,
            event_store=event_store,
            controller=policy or ContinueController(),
            compute_budget=restored.compute_budget,
            verify_runs=snapshot.verify_runs,
            verify_split_policy=snapshot.verify_split_policy,
            initial_current_best=(
                first_record.parsed_metrics.get(restored.node_spec.metric_name)
                if first_record.decision == TrialDecision.KEPT
                else restored.current_best
            ),
            initial_incumbent_stack=restored.incumbent_stack,
            **resume_seeds,
        )
    return ResumeResult(restored=restored, first_record=first_record, continuation=continuation)


AUDIT_SCHEMA = "r1.restore_audit/1"


def run_restore_audit(
    snapshot: CampaignSnapshot,
    out_dir: str | Path,
    *,
    original_metric: float | None,
    worker: Any | None = None,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Replay the original action from a restored snapshot and audit equality.

    Writes ``restore_audit.json`` with the per-layer hashes of the snapshot and
    of a fresh restore, and the exact reproduction result.  Equality is never
    loosened to a tolerance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    check = restore_snapshot(snapshot, out / "restore_check", worker=worker, manager=manager)
    restored_editable = {rel: (check.into_dir / rel).read_bytes() for rel in snapshot.editable_paths}
    restored_node_state = _collect_node_state(check.into_dir, snapshot.branch_state_paths)
    restored_prefix = tuple(TrialAppendStore(check.records_path).read_all())
    restored_hashes = {
        "editable": _hash_file_map(restored_editable),
        "node_state": _hash_file_map(restored_node_state),
        "ledger_prefix": _hash_records(restored_prefix),
    }

    replay = resume_campaign(snapshot, out / "replay", first_action="continue", worker=worker, manager=manager)
    replayed_metric = (
        replay.first_record.parsed_metrics.get(check.node_spec.metric_name)
        if replay.first_record is not None
        else None
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "label": "ENGINEERING_ONLY",
        "node": snapshot.node_name,
        "epoch": snapshot.epoch,
        "budget_index": snapshot.budget_index,
        "snapshot_layer_hashes": snapshot.layer_hashes,
        "restored_layer_hashes": restored_hashes,
        "editable_hash_match": restored_hashes["editable"] == snapshot.layer_hashes.get("editable"),
        "node_state_hash_match": restored_hashes["node_state"] == snapshot.layer_hashes.get("node_state"),
        "ledger_prefix_hash_match": restored_hashes["ledger_prefix"] == snapshot.layer_hashes.get("ledger_prefix"),
        "original_metric": original_metric,
        "replayed_metric": replayed_metric,
        "reproduced": replayed_metric == original_metric,
    }
    (out / "restore_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit

