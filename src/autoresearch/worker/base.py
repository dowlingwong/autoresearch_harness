from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
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

