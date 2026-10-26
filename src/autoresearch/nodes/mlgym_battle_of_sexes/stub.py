"""Fake-only worker for the local smoke; never invokes an MLGym environment."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.mlgym_battle_of_sexes.metric_parser import adapt_evaluation
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult


class FakeMLGymWorker:
    """Exercise the adapter with injected fabricated task.evaluate() results."""

    mode = "mlgym_fake_only"

    def __init__(self, *, artifact_dir: Path, evaluate: Callable[[int], tuple[dict, str]]):
        self.artifact_dir = Path(artifact_dir)
        self.evaluate = evaluate

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
        if node_spec.name != "mlgym_battle_of_sexes" or not proposal.target_files:
            raise ValueError("fake MLGym worker requires a task proposal with target files")
        validate_edit_scope(proposal.target_files, node_spec).require_valid()
        metrics = adapt_evaluation(self.evaluate(budget_index))
        trial_dir = self.artifact_dir / f"trial-{budget_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        # Illustrative only: no source file is edited and no task code is executed.
        patch = trial_dir / "synthetic.patch"
        patch.write_text("--- a/strategy.py\n+++ b/strategy.py\n-# before\n+# fake candidate\n")
        log = trial_dir / "synthetic.txt"
        log.write_text(f"FAKE ONLY; adapted metrics: {metrics!r}\n")
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=proposal.target_files,
            success=bool(metrics),
            parsed_metrics=metrics,
            raw_log_ref=str(log),
            patch_ref=str(patch),
            git_commit_before="fake-before",
            git_commit_after="fake-after",
            failure_message=None if metrics else "fake evaluator returned an invalid metric",
            extra={"synthetic_only": True},
        )
