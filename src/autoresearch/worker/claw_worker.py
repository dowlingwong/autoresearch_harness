from __future__ import annotations

import json
import hashlib
import difflib
import re
import shutil
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any

from autoresearch.legacy.claw_code import ClawCodeAutoresearchAdapter
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.hyperparam_edits import (
    build_effective_config,
    constants_after_edit,
    parse_train_constants,
    values_equal,
)
from autoresearch.memory.schemas import FailureCategory
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
    objective = _objective_with_worker_memory(proposal)
    return {
        "objective": objective,
        "description": f"{proposal.proposal_summary} [trial-{budget_index:03d}]",
        "train_command": defaults.get("train_command", node_spec.run_command),
        "timeout_seconds": timeout_seconds,
        "log_path": defaults.get("log_path", "run.log"),
        "results_tsv": defaults.get("results_tsv", "results.tsv"),
        "syntax_check_command": defaults.get(
            "syntax_check_command", "python3 -m py_compile train.py"
        ),
        "stage2_memory_mode": proposal.extra.get("worker_memory_mode", ""),
        "stage2_memory_context": proposal.extra.get("worker_memory_context_text", ""),
        "stage2_repeated_bad_stats": proposal.extra.get("worker_repeated_bad_stats", {}),
    }


def _objective_with_worker_memory(proposal: ManagerProposal) -> str:
    context = str(proposal.extra.get("worker_memory_context_text", "")).strip()
    if not context:
        return proposal.objective
    repeated = proposal.extra.get("worker_repeated_bad_stats", {})
    repeated_ids = []
    if isinstance(repeated, dict):
        repeated_ids = list(repeated.get("flagged_trial_ids", []) or [])
    warning = ""
    if repeated_ids:
        warning = (
            "\nRepeated-bad warning: avoid making changes similar to these prior "
            f"bad trials: {', '.join(str(item) for item in repeated_ids[:12])}."
        )
    return "\n\n".join(
        [
            proposal.objective,
            (
                "Prior Stage 2 trial memory follows. Use it to avoid repeating "
                "discarded, invalid, no-op, or failed changes. If a prior idea did "
                "not work, choose a materially different bounded train.py change."
                f"{warning}\n{context}"
            ),
        ]
    )


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
        allow_any_branch: bool = False,
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
        self.allow_any_branch = allow_any_branch
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
        allow_any_branch: bool = False,
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
            allow_any_branch=allow_any_branch,
        )

    def run_trial(self, proposal: ManagerProposal, node_spec: NodeSpec, budget_index: int) -> WorkerResult:
        trial_id = f"trial-{budget_index:03d}"
        artifact_dir = self.artifacts_dir / trial_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pre_trial_diff = _git_diff_text(self.node_root)
        node_state_hash = _hash_editable_state(self.node_root, node_spec.editable_paths)
        training_seed = _extract_training_seed(self.node_root / "train.py")

        # Generate packet driven by the Stage 2 manager proposal
        packet = _packet_from_proposal(proposal, node_spec, budget_index, self.packet_defaults)
        fast_config_hash = _hash_training_config(packet)
        generated_packet_path = artifact_dir / "generated_packet.json"
        generated_packet_path.write_text(json.dumps(packet, indent=2))

        if proposal.extra.get("deterministic_patch") and isinstance(proposal.extra.get("structured_edit"), dict):
            return _run_deterministic_constant_trial(
                proposal=proposal,
                node_spec=node_spec,
                packet_ref=str(generated_packet_path),
                artifact_dir=artifact_dir,
                node_root=self.node_root,
                command=str(packet["train_command"]),
                log_path=str(packet["log_path"]),
                syntax_check_command=str(packet["syntax_check_command"]),
                pre_trial_diff=pre_trial_diff,
                reproducibility={
                    "node_state_hash": node_state_hash,
                    "fast_config_hash": fast_config_hash,
                    "training_seed": training_seed,
                },
            )

        result = self.adapter.loop(
            node_root=self.node_root,
            packet_path=generated_packet_path,
            model=self.model,
            host=self.host,
            iterations=1,
            retry_limit=1,
            allow_any_branch=self.allow_any_branch,
        )

        return _extract_worker_result(
            result,
            node_spec,
            packet_ref=str(generated_packet_path),
            artifact_dir=artifact_dir,
            node_root=self.node_root,
            fallback_command=str(packet["train_command"]),
            fallback_log_path=str(packet["log_path"]),
            pre_trial_diff=pre_trial_diff,
            reproducibility={
                "node_state_hash": node_state_hash,
                "fast_config_hash": fast_config_hash,
                "training_seed": training_seed,
            },
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
            allow_any_branch=self.allow_any_branch,
        )
        return _extract_worker_result(
            result,
            node_spec,
            packet_ref=str(packet_path),
            artifact_dir=self.artifacts_dir / f"trial-{budget_index:03d}",
            node_root=self.node_root,
            fallback_command=node_spec.run_command,
            fallback_log_path="run.log",
        )


def _extract_worker_result(
    loop_result: dict,
    node_spec: NodeSpec,
    packet_ref: str,
    artifact_dir: str | Path | None = None,
    node_root: str | Path | None = None,
    fallback_command: str | None = None,
    fallback_log_path: str | None = None,
    pre_trial_diff: str = "",
    reproducibility: dict[str, str | None] | None = None,
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
    no_op_patch = bool(run_payload.get("no_op_patch") or run_payload.get("error") == "no_op_patch")
    # Pre-captured diff: written by the legacy loop BEFORE discard restores train.py.
    candidate_patch_diff = (
        str(item.get("candidate_patch_diff", "")).strip() if isinstance(item, dict) else ""
    )

    val_bpb = experiment.get("val_bpb")
    metrics: dict[str, float] = {}
    if val_bpb is not None:
        metrics[node_spec.metric_name] = 1.0 - float(val_bpb)

    changed_files: tuple[str, ...]
    if isinstance(last_result, dict) and last_result.get("changed_files"):
        changed_files = tuple(_normalize_changed_file(str(f), node_root) for f in last_result["changed_files"])
    else:
        changed_files = _detect_changed_files(node_root, pre_trial_diff=pre_trial_diff)

    fallback_run: dict[str, Any] = {}
    if not experiment.get("success") and changed_files and fallback_command and node_root is not None:
        fallback_run = _run_fallback_experiment(
            node_spec=node_spec,
            node_root=Path(node_root),
            command=fallback_command,
            log_path=fallback_log_path or "run.log",
        )
        if fallback_run.get("success"):
            metrics[node_spec.metric_name] = float(fallback_run["metric"])
            experiment = {
                "success": True,
                "log_path": str(fallback_run["log_path"]),
                "fallback_stage2_run": True,
            }

    raw_log_ref = str(experiment.get("log_path", "") or fallback_log_path or "") if isinstance(experiment, dict) else ""
    raw_log_artifact = _capture_raw_log(raw_log_ref, artifact_path, node_root)
    parsed_metrics_ref = _write_parsed_metrics(metrics, artifact_path)
    patch_diff_ref = _write_patch_diff(
        run_payload, artifact_path, node_root, changed_files, candidate_patch_diff
    )
    # A real captured diff overrides a legacy no_op_patch claim.  The legacy
    # side marks no_op when train_hash_before == train_hash_after, but the
    # timing of that check can be wrong when AUTORESEARCH_NO_LEGACY_COMMITS=1;
    # the patch file is ground truth.
    if patch_diff_ref and no_op_patch:
        no_op_patch = False
    extra = {
        "generated_packet_ref": packet_ref,
        "patch_diff_ref": patch_diff_ref,
        "raw_log_ref": raw_log_artifact or raw_log_ref,
        "parsed_metrics_ref": parsed_metrics_ref,
        "legacy_loop_result_ref": str(artifact_path / "legacy_loop_result.json") if artifact_path else "",
        "legacy_recommended_status": str(run_payload.get("recommended_status", "")) if isinstance(run_payload, dict) else "",
        "legacy_worker_stop_reason": str(last_result.get("stop_reason", "")) if isinstance(last_result, dict) else "",
        "stage2_fallback_run": fallback_run,
    }
    if no_op_patch:
        extra["failure_category"] = "no_op_patch"
        extra["training_skipped"] = True
    elif (
        not experiment.get("success")
        and not changed_files
        and str(run_payload.get("error", "")).strip() == "worker did not modify train.py"
    ):
        extra["failure_category"] = FailureCategory.EDIT_FAILED.value
    extra.update({k: v for k, v in (reproducibility or {}).items() if v is not None})

    return WorkerResult(
        worker_mode=ClawWorker.mode,
        changed_files=changed_files,
        success=True if no_op_patch else bool(experiment.get("success", False)),
        parsed_metrics=metrics,
        raw_log_ref=raw_log_artifact or raw_log_ref,
        # Use the diff file path (or empty string) — never fall back to the
        # generated_packet.json path, which is not a diff and would fool
        # _patch_is_empty() into classifying every undiffable trial as no_op_patch.
        patch_ref=patch_diff_ref,
        git_commit_before=str(run_payload.get("base_commit", "")) if isinstance(run_payload, dict) else "",
        git_commit_after=str(run_payload.get("commit", "")) if isinstance(run_payload, dict) else "",
        failure_message=(
            str(run_payload.get("error", ""))
            if no_op_patch or extra.get("failure_category") == FailureCategory.EDIT_FAILED.value
            else None
        ),
        extra=extra,
    )


def _run_deterministic_constant_trial(
    *,
    proposal: ManagerProposal,
    node_spec: NodeSpec,
    packet_ref: str,
    artifact_dir: Path,
    node_root: Path,
    command: str,
    log_path: str,
    syntax_check_command: str,
    pre_trial_diff: str,
    reproducibility: dict[str, str | None] | None = None,
) -> WorkerResult:
    edit = dict(proposal.extra.get("structured_edit") or {})
    path = str(edit.get("path") or "train.py")
    symbol = str(edit.get("symbol") or "")
    old_value = str(edit.get("old") or "")
    new_value = str(edit.get("new") or "")
    train_path = node_root / path
    constants_before = parse_train_constants(train_path)
    effective_before = build_effective_config(constants_before)
    failure_extra = {
        "generated_packet_ref": packet_ref,
        "structured_edit": edit,
        "deterministic_patch": True,
        "effective_config_before": effective_before,
    }

    if not symbol or not old_value or not new_value:
        return _failed_deterministic_result(
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message="structured edit is missing symbol, old, or new value",
            artifact_dir=artifact_dir,
            extra=failure_extra,
            reproducibility=reproducibility,
        )
    current = constants_before.get(symbol)
    if current is None or not values_equal(current, old_value):
        return _failed_deterministic_result(
            category=FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            message=f"structured edit precondition failed for {symbol}: expected {old_value}, found {current}",
            artifact_dir=artifact_dir,
            extra={**failure_extra, "actual_value": current},
            reproducibility=reproducibility,
        )
    constants_after = constants_after_edit(constants_before, symbol=symbol, new_value=new_value)
    effective_after = build_effective_config(constants_after)
    effective_key = str(edit.get("effective_key") or "")
    if effective_key and values_equal(effective_before.get(effective_key, ""), effective_after.get(effective_key, "")):
        return _failed_deterministic_result(
            category=FailureCategory.EFFECTIVE_CONFIG_UNCHANGED,
            message=f"structured edit does not change effective training config: {effective_key}",
            artifact_dir=artifact_dir,
            extra={
                **failure_extra,
                "effective_config_after": effective_after,
                "effective_key": effective_key,
                "effective_config_changed": False,
            },
            reproducibility=reproducibility,
        )

    before_text = train_path.read_text(encoding="utf-8")
    _replace_python_constant(train_path, symbol=symbol, old_value=old_value, new_value=new_value)
    after_text = train_path.read_text(encoding="utf-8")
    candidate_patch_diff = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    changed_files = _detect_changed_files(node_root, pre_trial_diff=pre_trial_diff) or (path,)
    patch_ref = _write_patch_diff(
        {},
        artifact_dir,
        node_root,
        changed_files,
        candidate_patch_diff=candidate_patch_diff,
    )
    syntax = subprocess.run(
        ["/bin/zsh", "-lc", syntax_check_command],
        cwd=node_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if syntax.returncode != 0:
        syntax_log = artifact_dir / "syntax.log"
        syntax_log.write_text((syntax.stdout or "") + (syntax.stderr or ""), encoding="utf-8")
        return _failed_deterministic_result(
            category=FailureCategory.SYNTAX_ERROR,
            message=f"syntax check failed: {syntax_check_command}",
            artifact_dir=artifact_dir,
            extra={
                **failure_extra,
                "effective_config_after": effective_after,
                "patch_diff_ref": patch_ref,
                "raw_log_ref": str(syntax_log),
                "effective_config_changed": True,
            },
            changed_files=changed_files,
            raw_log_ref=str(syntax_log),
            patch_ref=patch_ref,
            reproducibility=reproducibility,
        )

    fallback_run = _run_fallback_experiment(
        node_spec=node_spec,
        node_root=node_root,
        command=command,
        log_path=log_path,
    )
    raw_log_artifact = _capture_raw_log(str(fallback_run.get("log_path") or log_path), artifact_dir, node_root)
    metrics: dict[str, float] = {}
    if fallback_run.get("success"):
        metrics[node_spec.metric_name] = float(fallback_run["metric"])
    parsed_metrics_ref = _write_parsed_metrics(metrics, artifact_dir)
    before_commit = _git_rev_parse(node_root)
    extra = {
        **failure_extra,
        "effective_config_after": effective_after,
        "effective_config_changed": True,
        "patch_diff_ref": patch_ref,
        "raw_log_ref": raw_log_artifact,
        "parsed_metrics_ref": parsed_metrics_ref,
        "legacy_loop_result_ref": "",
        "stage2_fallback_run": fallback_run,
    }
    if not fallback_run.get("success"):
        if fallback_run.get("returncode", 0) != 0:
            extra["failure_category"] = FailureCategory.RUNTIME_ERROR.value
        else:
            extra["failure_category"] = FailureCategory.METRIC_MISSING.value
    extra.update({k: v for k, v in (reproducibility or {}).items() if v is not None})
    return WorkerResult(
        worker_mode=f"{ClawWorker.mode}_deterministic_patch",
        changed_files=changed_files,
        success=bool(fallback_run.get("success")),
        parsed_metrics=metrics,
        raw_log_ref=raw_log_artifact or str(fallback_run.get("log_path") or ""),
        patch_ref=patch_ref,
        git_commit_before=before_commit,
        git_commit_after=f"{before_commit}-dirty" if patch_ref else before_commit,
        failure_message=None if fallback_run.get("success") else str(fallback_run.get("parse_error") or fallback_run.get("stderr_tail") or "training failed"),
        extra=extra,
    )


def _replace_python_constant(
    train_path: Path,
    *,
    symbol: str,
    old_value: str,
    new_value: str,
) -> None:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(symbol)}\s*=\s*)(?P<value>[^#\n]+?)(?P<suffix>\s*(?:#.*)?)$")
    lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    output: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        match = pattern.match(body)
        if match and values_equal(match.group("value").strip(), old_value):
            output.append(f"{match.group('prefix')}{new_value}{match.group('suffix')}{newline}")
            changed = True
        else:
            output.append(line)
    if not changed:
        raise ValueError(f"could not replace {symbol}={old_value} in {train_path}")
    train_path.write_text("".join(output), encoding="utf-8")


def _failed_deterministic_result(
    *,
    category: FailureCategory,
    message: str,
    artifact_dir: Path,
    extra: dict[str, Any],
    changed_files: tuple[str, ...] = (),
    raw_log_ref: str = "",
    patch_ref: str = "",
    reproducibility: dict[str, str | None] | None = None,
) -> WorkerResult:
    parsed_metrics_ref = _write_parsed_metrics({}, artifact_dir)
    payload = {
        **extra,
        "failure_category": category.value,
        "parsed_metrics_ref": parsed_metrics_ref,
        "training_skipped": category
        in {
            FailureCategory.PROPOSAL_PRECONDITION_FAILED,
            FailureCategory.EFFECTIVE_CONFIG_UNCHANGED,
        },
    }
    payload.update({k: v for k, v in (reproducibility or {}).items() if v is not None})
    return WorkerResult(
        worker_mode=f"{ClawWorker.mode}_deterministic_patch",
        changed_files=changed_files,
        success=False,
        parsed_metrics={},
        raw_log_ref=raw_log_ref,
        patch_ref=patch_ref,
        git_commit_before="",
        git_commit_after="",
        failure_message=message,
        extra=payload,
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


def _write_patch_diff(
    run_payload: dict,
    artifact_dir: Path | None,
    node_root: str | Path | None,
    changed_files: tuple[str, ...] = (),
    candidate_patch_diff: str = "",
) -> str:
    """Compute and write patch.diff, trying git commands then the pre-captured fallback.

    ``candidate_patch_diff`` is the raw diff text captured by the legacy loop
    BEFORE ``discard_autoresearch_candidate`` restores train.py.  It is the
    most reliable source when AUTORESEARCH_NO_LEGACY_COMMITS=1 (Stage 2
    default), because the working-tree changes are reverted before this function
    runs.  Git-range commands are attempted first for the real-commit path.
    """
    if artifact_dir is None or node_root is None:
        return ""
    before = str(run_payload.get("base_commit", "")).strip()
    after = str(run_payload.get("commit", "")).strip()
    commands: list[list[str]] = []
    pathspec = ["--", *(changed_files or ["."])]
    if before and after and "unknown" not in {before, after} and not after.endswith("-dirty"):
        commands.append(["git", "diff", f"{before}..{after}", *pathspec])
    if before and after.endswith("-dirty") and "unknown" not in before:
        commands.append(["git", "diff", before, *pathspec])
    commands.append(["git", "diff", *pathspec])

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

    # Final fallback: pre-captured diff from the legacy loop (written before
    # discard_autoresearch_candidate restored train.py).  This is the primary
    # path for Stage 2 campaigns using AUTORESEARCH_NO_LEGACY_COMMITS=1.
    if candidate_patch_diff.strip():
        target = artifact_dir / "patch.diff"
        target.write_text(candidate_patch_diff, encoding="utf-8")
        return str(target)

    return ""


def _detect_changed_files(node_root: str | Path | None, pre_trial_diff: str = "") -> tuple[str, ...]:
    if node_root is None:
        return ()
    if pre_trial_diff == _git_diff_text(node_root):
        return ()
    run = subprocess.run(
        ["git", "diff", "--name-only", "--", "."],
        cwd=Path(node_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return ()
    return tuple(
        path
        for path in (_normalize_changed_file(line.strip(), node_root) for line in run.stdout.splitlines())
        if path
    )


def _git_diff_text(node_root: str | Path | None) -> str:
    if node_root is None:
        return ""
    run = subprocess.run(
        ["git", "diff", "--", "."],
        cwd=Path(node_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return run.stdout if run.returncode == 0 else ""


def _git_rev_parse(node_root: str | Path) -> str:
    run = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(node_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return run.stdout.strip() if run.returncode == 0 else ""


def _hash_editable_state(node_root: str | Path, editable_paths: tuple[str, ...]) -> str:
    root = Path(node_root)
    digest = hashlib.sha256()
    for rel_path in sorted(editable_paths):
        path = root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.exists() else b"")
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_training_config(packet: dict[str, Any]) -> str:
    relevant = {
        "train_command": packet.get("train_command", ""),
        "timeout_seconds": packet.get("timeout_seconds", ""),
        "log_path": packet.get("log_path", ""),
        "results_tsv": packet.get("results_tsv", ""),
        "syntax_check_command": packet.get("syntax_check_command", ""),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_training_seed(train_path: Path) -> str | None:
    if not train_path.exists():
        return None
    text = train_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^\s*SEED\s*=\s*([0-9]+)\s*$", text)
    return match.group(1) if match else None


def _normalize_changed_file(path: str, node_root: str | Path | None) -> str:
    if node_root is None:
        return path
    root_run = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(node_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if root_run.returncode != 0:
        return path
    repo_root = Path(root_run.stdout.strip()).resolve()
    absolute = (repo_root / path).resolve()
    try:
        return str(absolute.relative_to(Path(node_root).resolve()))
    except ValueError:
        return path


def _run_fallback_experiment(
    *,
    node_spec: NodeSpec,
    node_root: Path,
    command: str,
    log_path: str,
) -> dict[str, Any]:
    run = subprocess.run(
        ["/bin/zsh", "-lc", command],
        cwd=node_root,
        capture_output=True,
        text=True,
        timeout=int(node_spec.default_budget.max_wall_clock_hours * 3600)
        if node_spec.default_budget.max_wall_clock_hours
        else None,
        check=False,
    )
    target_log = Path(log_path)
    if not target_log.is_absolute():
        target_log = node_root / target_log
    log_text = "".join(
        part
        for part in (
            run.stdout,
            "\n" if run.stdout and run.stderr else "",
            run.stderr,
        )
        if part
    )
    target_log.parent.mkdir(parents=True, exist_ok=True)
    if log_text or not target_log.exists():
        target_log.write_text(log_text, encoding="utf-8")
    else:
        log_text = target_log.read_text(encoding="utf-8", errors="replace")
    if run.returncode != 0:
        return {
            "success": False,
            "returncode": run.returncode,
            "log_path": str(target_log),
            "stderr_tail": run.stderr[-1000:],
        }
    try:
        parser = _resolve_metric_parser(node_spec.metric_parser)
        parsed = _parse_fallback_metric(parser, target_log, log_text)
        metric = _extract_fallback_metric(parsed, node_spec.metric_name)
        return {
            "success": True,
            "returncode": run.returncode,
            "log_path": str(target_log),
            "metric": float(metric),
        }
    except Exception as exc:
        return {
            "success": False,
            "returncode": run.returncode,
            "log_path": str(target_log),
            "parse_error": str(exc),
        }


def _resolve_metric_parser(spec: str):
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"invalid metric parser spec: {spec}")
    return getattr(import_module(module_name), attr)


def _parse_fallback_metric(parser, log_path: Path, log_text: str):
    """Support both path-based and stdout-text metric parsers."""

    try:
        parsed = parser(log_path)
        if _extract_fallback_metric(parsed, "") is not None:
            return parsed
    except (TypeError, OSError, ValueError):
        pass
    return parser(log_text)


def _extract_fallback_metric(parsed, metric_name: str) -> float | None:
    metric = getattr(parsed, "metric_value", None)
    if metric is not None:
        return float(metric)
    if isinstance(parsed, dict):
        for key in (metric_name, "metric_value", "val_auc", "val_score"):
            if key and key in parsed:
                return float(parsed[key])
    return None
