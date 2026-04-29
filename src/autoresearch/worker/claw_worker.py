from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from autoresearch.legacy.claw_code import ClawCodeAutoresearchAdapter
from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult


def _packet_from_proposal(
    proposal: ManagerProposal,
    node_spec: NodeSpec,
    budget_index: int,
    packet_defaults: dict | None = None,
) -> dict:
    """Build an AutoresearchExperimentPacket dict driven by the Stage 2 ManagerProposal.

    objective and description always come from the proposal.
    train_command, timeout, log_path, results_tsv, and syntax_check_command
    fall back to node_spec values, then to explicit packet_defaults overrides.
    """
    defaults = packet_defaults or {}
    timeout_seconds = int(
        defaults.get("timeout_seconds", node_spec.default_budget.max_wall_clock_hours * 3600)
    )
    return {
        "objective": proposal.objective,
        "description": f"{proposal.proposal_summary} [trial-{budget_index:03d}]",
        "train_command": defaults.get("train_command", node_spec.run_command),
        "timeout_seconds": timeout_seconds,
        "log_path": defaults.get("log_path", "run.log"),
        "results_tsv": defaults.get("results_tsv", "results.tsv"),
        "syntax_check_command": defaults.get(
            "syntax_check_command", "python3 -m py_compile train.py"
        ),
    }


class ClawWorker:
    """Stage 2 worker that drives the legacy claw-code loop with a proposal-generated packet.

    Each call to run_trial():
      1. Generates a packet JSON from the ManagerProposal and NodeSpec.
      2. Writes the packet to experiments/artifacts/{trial_id}/generated_packet.json.
      3. Calls the legacy autoresearch loop with the generated packet.
      4. Returns a WorkerResult that the Stage 2 control plane can validate and decide on.
    """

    mode = "claw_style_worker"

    def __init__(
        self,
        repo_root: str | Path,
        node_root: str | Path,
        artifacts_dir: str | Path | None = None,
        packet_defaults: dict | None = None,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.node_root = Path(node_root).resolve()
        self.artifacts_dir = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else self.repo_root / "experiments" / "artifacts"
        )
        self.packet_defaults = packet_defaults or {}
        self.model = model
        self.host = host
        self.adapter = ClawCodeAutoresearchAdapter(self.repo_root)

    @classmethod
    def from_packet_defaults_file(
        cls,
        repo_root: str | Path,
        node_root: str | Path,
        packet_defaults_path: str | Path,
        artifacts_dir: str | Path | None = None,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
    ) -> "ClawWorker":
        """Load packet defaults (timeout, log_path, etc.) from a JSON file.

        The file's objective and description fields are ignored — those always
        come from the ManagerProposal.
        """
        defaults = json.loads(Path(packet_defaults_path).read_text())
        defaults.pop("objective", None)
        defaults.pop("description", None)
        return cls(
            repo_root=repo_root,
            node_root=node_root,
            artifacts_dir=artifacts_dir,
            packet_defaults=defaults,
            model=model,
            host=host,
        )

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
        trial_id = f"trial-{budget_index:03d}"
        artifact_dir = self.artifacts_dir / trial_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Generate packet driven by the Stage 2 manager proposal
        packet = _packet_from_proposal(proposal, node_spec, budget_index, self.packet_defaults)
        generated_packet_path = artifact_dir / "generated_packet.json"
        generated_packet_path.write_text(json.dumps(packet, indent=2))

        result = self.adapter.loop(
            node_root=self.node_root,
            packet_path=generated_packet_path,
            model=self.model,
            host=self.host,
            iterations=1,
            retry_limit=1,
        )

        return _extract_worker_result(
            result,
            node_spec,
            packet_ref=str(generated_packet_path),
            artifact_dir=artifact_dir,
            node_root=self.node_root,
        )

    def run_trial_with_packet_path(
        self,
        proposal: ManagerProposal,
        node_spec: NodeSpec,
        budget_index: int,
        packet_path: str | Path,
    ) -> WorkerResult:
        """Low-level escape hatch: call the loop with an explicit packet path.

        The proposal is recorded in the TrialRecord but does not override the packet.
        Only use this for debugging or legacy compatibility — prefer run_trial().
        """
        result = self.adapter.loop(
            node_root=self.node_root,
            packet_path=packet_path,
            model=self.model,
            host=self.host,
            iterations=1,
            retry_limit=1,
        )
        return _extract_worker_result(
            result,
            node_spec,
            packet_ref=str(packet_path),
            artifact_dir=self.artifacts_dir / f"trial-{budget_index:03d}",
            node_root=self.node_root,
        )


def _extract_worker_result(
    loop_result: dict,
    node_spec: NodeSpec,
    packet_ref: str,
    artifact_dir: str | Path | None = None,
    node_root: str | Path | None = None,
) -> WorkerResult:
    """Parse a raw loop_autoresearch() return dict into a typed WorkerResult."""
    artifact_path = Path(artifact_dir).resolve() if artifact_dir is not None else None
    if artifact_path is not None:
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "legacy_loop_result.json").write_text(
            json.dumps(loop_result, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    history = loop_result.get("history", [])
    item = history[0] if isinstance(history, list) and history else {}
    run_payload = item.get("run", item) if isinstance(item, dict) else {}
    worker = run_payload.get("worker", {}) if isinstance(run_payload, dict) else {}
    last_result = worker.get("last_result", {}) if isinstance(worker, dict) else {}
    experiment = run_payload.get("experiment", {}) if isinstance(run_payload, dict) else {}
    if not isinstance(experiment, dict):
        experiment = {}

    val_bpb = experiment.get("val_bpb")
    metrics: dict[str, float] = {}
    if val_bpb is not None:
        metrics[node_spec.metric_name] = 1.0 - float(val_bpb)

    changed_files: tuple[str, ...]
    if isinstance(last_result, dict) and last_result.get("changed_files"):
        changed_files = tuple(str(f) for f in last_result["changed_files"])
    else:
        changed_files = ()

    raw_log_ref = str(experiment.get("log_path", "")) if isinstance(experiment, dict) else ""
    raw_log_artifact = _capture_raw_log(raw_log_ref, artifact_path, node_root)
    parsed_metrics_ref = _write_parsed_metrics(metrics, artifact_path)
    patch_diff_ref = _write_patch_diff(run_payload, artifact_path, node_root)
    extra = {
        "generated_packet_ref": packet_ref,
        "patch_diff_ref": patch_diff_ref,
        "raw_log_ref": raw_log_artifact or raw_log_ref,
        "parsed_metrics_ref": parsed_metrics_ref,
        "legacy_loop_result_ref": str(artifact_path / "legacy_loop_result.json") if artifact_path else "",
        "legacy_recommended_status": str(run_payload.get("recommended_status", "")) if isinstance(run_payload, dict) else "",
        "legacy_worker_stop_reason": str(last_result.get("stop_reason", "")) if isinstance(last_result, dict) else "",
    }

    return WorkerResult(
        worker_mode=ClawWorker.mode,
        changed_files=changed_files,
        success=bool(experiment.get("success", False)),
        parsed_metrics=metrics,
        raw_log_ref=raw_log_artifact or raw_log_ref,
        patch_ref=patch_diff_ref or packet_ref,
        git_commit_before=str(run_payload.get("base_commit", "")) if isinstance(run_payload, dict) else "",
        git_commit_after=str(run_payload.get("commit", "")) if isinstance(run_payload, dict) else "",
        extra=extra,
    )


def _capture_raw_log(raw_log_ref: str, artifact_dir: Path | None, node_root: str | Path | None) -> str:
    if artifact_dir is None or not raw_log_ref:
        return ""
    source = Path(raw_log_ref)
    if not source.is_absolute() and node_root is not None:
        source = Path(node_root) / source
    if not source.exists():
        return ""
    target = artifact_dir / "run.log"
    shutil.copyfile(source, target)
    return str(target)


def _write_parsed_metrics(metrics: dict[str, float], artifact_dir: Path | None) -> str:
    if artifact_dir is None:
        return ""
    target = artifact_dir / "parsed_metrics.json"
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


def _write_patch_diff(run_payload: dict, artifact_dir: Path | None, node_root: str | Path | None) -> str:
    if artifact_dir is None or node_root is None:
        return ""
    before = str(run_payload.get("base_commit", "")).strip()
    after = str(run_payload.get("commit", "")).strip()
    commands: list[list[str]] = []
    if before and after and "unknown" not in {before, after} and not after.endswith("-dirty"):
        commands.append(["git", "diff", f"{before}..{after}", "--"])
    if before and after.endswith("-dirty") and "unknown" not in before:
        commands.append(["git", "diff", before, "--"])
    commands.append(["git", "diff", "--"])

    for command in commands:
        run = subprocess.run(
            command,
            cwd=Path(node_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode == 0 and run.stdout.strip():
            target = artifact_dir / "patch.diff"
            target.write_text(run.stdout, encoding="utf-8")
            return str(target)
    return ""
