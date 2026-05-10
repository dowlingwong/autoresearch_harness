from __future__ import annotations

import subprocess
from pathlib import Path

from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult


class NoOpPatchWorker:
    """Worker used to exercise the no-op patch guard.

    It reports a successful worker turn but emits an empty patch and no metrics.
    The control plane should mark the trial failed_invalid / no_op_patch without
    running training.
    """

    mode = "stress_no_op_patch_worker"

    def __init__(self, *, node_root: str | Path, artifacts_dir: str | Path) -> None:
        self.node_root = Path(node_root).resolve()
        self.artifacts_dir = Path(artifacts_dir).resolve()

    def run_trial(
        self,
        proposal: ManagerProposal,
        node_spec: NodeSpec,
        budget_index: int,
    ) -> WorkerResult:
        artifact_dir = self.artifacts_dir / f"trial-{budget_index:03d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        patch_ref = artifact_dir / "patch.diff"
        raw_log_ref = artifact_dir / "run.log"
        patch_ref.write_text("", encoding="utf-8")
        raw_log_ref.write_text(
            "Synthetic no-op stress trial: empty patch emitted; training skipped.\n",
            encoding="utf-8",
        )
        commit = _short_head(self.node_root)
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=node_spec.editable_paths,
            success=True,
            parsed_metrics={},
            raw_log_ref=str(raw_log_ref),
            patch_ref=str(patch_ref),
            git_commit_before=commit,
            git_commit_after=commit,
            failure_message="no_op_patch",
            extra={"failure_category": "no_op_patch", "training_skipped": True},
        )


def _short_head(cwd: Path) -> str:
    run = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return run.stdout.strip() if run.returncode == 0 and run.stdout.strip() else "unknown"
