from __future__ import annotations

import subprocess
from pathlib import Path

from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult


class ScopeViolationWorker:
    """Worker used for KDD stress tests.

    It emits a concrete patch against a frozen path without applying it. The
    control plane should reject the trial before accepting any state change.
    """

    mode = "stress_scope_violation_worker"

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
        forbidden_path = node_spec.frozen_paths[0] if node_spec.frozen_paths else "prepare.py"
        patch_ref.write_text(_scope_violation_patch(forbidden_path), encoding="utf-8")
        raw_log_ref.write_text(
            "Synthetic stress trial: generated forbidden-path patch and skipped training.\n",
            encoding="utf-8",
        )
        commit = _short_head(self.node_root)
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=(forbidden_path,),
            success=True,
            parsed_metrics={node_spec.metric_name: 0.0},
            raw_log_ref=str(raw_log_ref),
            patch_ref=str(patch_ref),
            git_commit_before=commit,
            git_commit_after=commit,
            extra={"stress_mode": "scope_violation", "training_skipped": True},
        )


def _scope_violation_patch(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 0000000..1111111 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,2 @@\n"
        " # frozen file\n"
        "+# synthetic forbidden edit for governance stress test\n"
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
