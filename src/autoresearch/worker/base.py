from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from typing import Protocol

from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.spec import NodeSpec


@dataclass(frozen=True)
class WorkerResult:
    worker_mode: str
    changed_files: tuple[str, ...]
    success: bool
    parsed_metrics: dict[str, float]
    raw_log_ref: str
    patch_ref: str
    git_commit_before: str
    git_commit_after: str
    failure_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["changed_files"] = list(self.changed_files)
        return payload


class Worker(Protocol):
    mode: str

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
        ...


class DryRunWorker:
    mode = "dry_run_worker"

    def __init__(self, profile: str = "monotonic") -> None:
        if profile not in {
            "monotonic",
            "mixed_lifecycle",
            "ablation_none",
            "ablation_append_only_summary",
            "ablation_append_only_summary_with_rationale",
        }:
            raise ValueError(f"unknown dry-run worker profile: {profile}")
        self.profile = profile

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
        if self.profile == "mixed_lifecycle":
            return self._profiled_result(
                node_spec,
                budget_index,
                metrics={1: 0.781, 2: 0.782, 3: 0.7815, 5: 0.783},
                no_op_trials={4},
            )
        if self.profile == "ablation_none":
            return self._profiled_result(
                node_spec,
                budget_index,
                metrics={1: 0.781, 2: 0.780, 5: 0.7805},
                no_op_trials={3, 4},
            )
        if self.profile == "ablation_append_only_summary":
            return self._profiled_result(
                node_spec,
                budget_index,
                metrics={1: 0.781, 2: 0.780, 3: 0.782, 4: 0.7815, 5: 0.783},
                no_op_trials=set(),
            )
        if self.profile == "ablation_append_only_summary_with_rationale":
            return self._profiled_result(
                node_spec,
                budget_index,
                metrics={1: 0.781, 2: 0.780, 3: 0.782, 4: 0.783, 5: 0.784},
                no_op_trials=set(),
            )

        metric = 0.78 + (budget_index * 0.001)
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=node_spec.editable_paths,
            success=True,
            parsed_metrics={node_spec.metric_name: metric},
            raw_log_ref=f"experiments/artifacts/dry_run_trial_{budget_index:03d}/run.log",
            patch_ref=f"experiments/artifacts/dry_run_trial_{budget_index:03d}/patch.diff",
            git_commit_before=f"dry-before-{budget_index:03d}",
            git_commit_after=f"dry-after-{budget_index:03d}",
        )

    def _profiled_result(
        self,
        node_spec: NodeSpec,
        budget_index: int,
        *,
        metrics: dict[int, float],
        no_op_trials: set[int],
    ) -> WorkerResult:
        if budget_index in no_op_trials:
            return WorkerResult(
                worker_mode=self.mode,
                changed_files=node_spec.editable_paths,
                success=True,
                parsed_metrics={},
                raw_log_ref=f"experiments/artifacts/dry_run_trial_{budget_index:03d}/run.log",
                patch_ref="",
                git_commit_before=f"dry-before-{budget_index:03d}",
                git_commit_after=f"dry-after-{budget_index:03d}",
                failure_message="no_op_patch",
                extra={"failure_category": "no_op_patch", "training_skipped": True},
            )

        metric = metrics.get(budget_index, max(metrics.values()) + (budget_index * 0.0001))
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=node_spec.editable_paths,
            success=True,
            parsed_metrics={node_spec.metric_name: metric},
            raw_log_ref=f"experiments/artifacts/dry_run_trial_{budget_index:03d}/run.log",
            patch_ref=f"experiments/artifacts/dry_run_trial_{budget_index:03d}/patch.diff",
            git_commit_before=f"dry-before-{budget_index:03d}",
            git_commit_after=f"dry-after-{budget_index:03d}",
            extra={"dry_run_profile": self.profile},
        )
