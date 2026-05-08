"""Local worker backend for the autoresearch Stage 2 framework.

``LocalWorker`` applies bounded hyperparameter changes directly to Python
source files without requiring the claw-code harness or an external LLM.  It
parses the ``ManagerProposal.objective`` for a recognised change pattern,
applies the edit in-place, runs the node's ``run_command``, captures the log,
and returns a ``WorkerResult``.

Recognised objective pattern
-----------------------------
The worker looks for a line of the form::

    Change <param> from <old_value> to <new_value>

where ``<param>`` is a Python identifier that appears as a bare assignment
(``PARAM = ...``) in one of the ``target_files``.  The match is
case-insensitive.  If no recognised pattern is found the trial is executed
without any code modification (useful for measuring the clean baseline).

Limitations
-----------
- Only top-level bare assignments are rewritten (``PARAM = value``).
- Numeric, string, and boolean values are supported.
- Does not support multi-file edits in a single trial.
- Does not invoke a language model; the proposal must supply the concrete
  old/new values explicitly.

This worker is designed for:
1. Deterministic smoke testing of the control plane without Ollama.
2. Running fast hyperparameter sweeps where the manager issues precise
   numerical changes.
3. Serving as the reference implementation for future worker backends.
"""
from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoresearch.common.logging import get_logger
from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult

_log = get_logger(__name__)

# Regex: "Change <param> from <old> to <new>"
_CHANGE_RE = re.compile(
    r"change\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+(\S+)\s+to\s+(\S+)",
    re.IGNORECASE,
)


@dataclass
class _AppliedEdit:
    file_path: Path
    param: str
    old_value: str
    new_value: str
    original_source: str
    patched_source: str

    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.original_source.splitlines(keepends=True),
                self.patched_source.splitlines(keepends=True),
                fromfile=f"a/{self.file_path.name}",
                tofile=f"b/{self.file_path.name}",
            )
        )


class LocalWorker:
    """Direct-edit worker backend that requires no external LLM or harness.

    Args:
        node_root: Absolute path to the experiment node directory (where
            the editable source files live and the run command is executed).
        artifact_dir: Directory where trial artefacts (log, diff, metrics)
            are written.  Created on first use.
        timeout_seconds: Wall-clock timeout for the run command.
    """

    mode = "local_worker"

    def __init__(
        self,
        node_root: str | Path,
        artifact_dir: Optional[str | Path] = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self._node_root = Path(node_root).resolve()
        self._artifact_dir: Optional[Path] = Path(artifact_dir).resolve() if artifact_dir else None
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public Worker interface
    # ------------------------------------------------------------------

    def run_trial(
        self,
        proposal: ManagerProposal,
        node_spec: NodeSpec,
        budget_index: int,
    ) -> WorkerResult:
        """Apply the proposal's change, run the experiment, return results.

        Steps:
        1. Parse the objective for a Change directive.
        2. Locate the target file in ``node_root``.
        3. Apply the substitution and record the diff.
        4. Execute ``node_spec.run_command`` in ``node_root``.
        5. Parse ``val_bpb`` / ``val_auc`` from stdout+stderr.
        6. Restore the original file on failure (best-effort).
        7. Return a ``WorkerResult``.
        """
        trial_label = f"trial-{budget_index:03d}"
        artifact_dir = self._resolve_artifact_dir(trial_label)

        edit: Optional[_AppliedEdit] = None
        changed_files: tuple[str, ...] = ()
        patch_ref = ""
        git_before = f"local-before-{budget_index:03d}"
        git_after = f"local-before-{budget_index:03d}"  # updated after edit

        # --- parse and apply edit ---
        edit_error: Optional[str] = None
        parsed_change = _parse_change_directive(proposal.objective)
        if parsed_change is None:
            _log.info("%s: no Change directive found; running without edit", trial_label)
        else:
            param, old_val, new_val = parsed_change
            target_file = self._locate_target_file(param, proposal, node_spec)
            if target_file is None:
                edit_error = f"parameter '{param}' not found in any target file"
                _log.warning("%s: %s", trial_label, edit_error)
            else:
                edit_result = _apply_edit(target_file, param, old_val, new_val)
                if isinstance(edit_result, str):
                    edit_error = edit_result
                    _log.warning("%s: edit failed: %s", trial_label, edit_error)
                else:
                    edit = edit_result
                    changed_files = (str(target_file.relative_to(self._node_root)),)
                    git_after = f"local-after-{budget_index:03d}"

        if edit_error:
            return WorkerResult(
                worker_mode=self.mode,
                changed_files=(),
                success=False,
                parsed_metrics={},
                raw_log_ref="",
                patch_ref="",
                git_commit_before=git_before,
                git_commit_after=git_after,
                failure_message=edit_error,
            )

        # --- write patch diff ---
        if edit is not None:
            diff_text = edit.diff()
            patch_path = artifact_dir / "patch.diff"
            patch_path.write_text(diff_text, encoding="utf-8")
            patch_ref = str(patch_path)
            _log.info("%s: applied edit %s: %s → %s", trial_label, param, old_val, new_val)

        # --- run experiment ---
        t0 = time.monotonic()
        log_path, run_success, failure_msg, parsed_metrics = self._run_command(
            node_spec, artifact_dir, trial_label
        )
        elapsed = time.monotonic() - t0
        _log.info(
            "%s: run_command finished in %.1fs; success=%s metrics=%s",
            trial_label,
            elapsed,
            run_success,
            parsed_metrics,
        )

        # --- restore on failure (best-effort) ---
        if not run_success and edit is not None:
            _restore_file(edit)

        return WorkerResult(
            worker_mode=self.mode,
            changed_files=changed_files,
            success=run_success,
            parsed_metrics=parsed_metrics,
            raw_log_ref=str(log_path) if log_path else "",
            patch_ref=patch_ref,
            git_commit_before=git_before,
            git_commit_after=git_after,
            failure_message=failure_msg if not run_success else None,
            extra={"node_root": str(self._node_root)},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_artifact_dir(self, trial_label: str) -> Path:
        if self._artifact_dir is not None:
            d = self._artifact_dir / trial_label
        else:
            d = self._node_root / ".autoresearch_artifacts" / trial_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _locate_target_file(
        self,
        param: str,
        proposal: ManagerProposal,
        node_spec: NodeSpec,
    ) -> Optional[Path]:
        """Find which editable file contains the top-level assignment for *param*."""
        candidates = list(proposal.target_files) or list(node_spec.editable_paths)
        for rel_path in candidates:
            abs_path = self._node_root / rel_path
            if not abs_path.exists():
                continue
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            if _assignment_pattern(param).search(source):
                return abs_path
        return None

    def _run_command(
        self,
        node_spec: NodeSpec,
        artifact_dir: Path,
        trial_label: str,
    ) -> tuple[Optional[Path], bool, Optional[str], dict[str, float]]:
        """Run the node's run_command and capture output.

        Returns:
            (log_path, success, failure_message, parsed_metrics)
        """
        log_path = artifact_dir / "run.log"
        cmd = node_spec.run_command
        _log.info("%s: running: %s", trial_label, cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self._node_root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            combined = result.stdout + "\n" + result.stderr
            log_path.write_text(combined, encoding="utf-8")
            success = result.returncode == 0
            failure_msg = f"exit code {result.returncode}" if not success else None
            metrics = _parse_metrics_from_log(combined)
            return log_path, success, failure_msg, metrics
        except subprocess.TimeoutExpired as exc:
            msg = f"run_command timed out after {self._timeout}s"
            log_path.write_text(str(exc), encoding="utf-8")
            return log_path, False, msg, {}
        except Exception as exc:  # noqa: BLE001
            msg = f"run_command raised {type(exc).__name__}: {exc}"
            log_path.write_text(msg, encoding="utf-8")
            return log_path, False, msg, {}


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, unit-testable)
# ---------------------------------------------------------------------------

def _parse_change_directive(objective: str) -> Optional[tuple[str, str, str]]:
    """Extract (param, old_value, new_value) from an objective string.

    Returns ``None`` if no recognised directive is found.
    """
    match = _CHANGE_RE.search(objective)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _assignment_pattern(param: str) -> re.Pattern[str]:
    """Return a regex matching a bare top-level assignment for *param*."""
    return re.compile(
        r"^(" + re.escape(param) + r"\s*=\s*)(.+?)(\s*)$",
        re.MULTILINE,
    )


def _apply_edit(
    file_path: Path,
    param: str,
    old_value: str,
    new_value: str,
) -> _AppliedEdit | str:
    """Rewrite *param = old_value* → *param = new_value* in *file_path*.

    Returns an ``_AppliedEdit`` on success, or a string error message on
    failure.
    """
    source = file_path.read_text(encoding="utf-8")
    pattern = _assignment_pattern(param)
    match = pattern.search(source)
    if match is None:
        return f"assignment '{param} = ...' not found in {file_path.name}"

    # Verify the existing value loosely (strip whitespace/quotes).
    current_raw = match.group(2).strip()
    if not _values_loosely_equal(current_raw, old_value):
        return (
            f"'{param}' currently has value {current_raw!r}, "
            f"expected {old_value!r}; skipping edit to avoid clobbering"
        )

    patched = pattern.sub(r"\g<1>" + new_value + r"\g<3>", source)
    file_path.write_text(patched, encoding="utf-8")
    return _AppliedEdit(
        file_path=file_path,
        param=param,
        old_value=old_value,
        new_value=new_value,
        original_source=source,
        patched_source=patched,
    )


def _restore_file(edit: _AppliedEdit) -> None:
    """Restore the original source after a failed run (best-effort)."""
    try:
        edit.file_path.write_text(edit.original_source, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not restore %s: %s", edit.file_path, exc)


def _values_loosely_equal(a: str, b: str) -> bool:
    """Compare two value strings loosely (strip surrounding quotes/spaces)."""
    def norm(s: str) -> str:
        return s.strip().strip("\"'")
    return norm(a) == norm(b)


def _parse_metrics_from_log(text: str) -> dict[str, float]:
    """Extract key: value metric lines from combined run output.

    Handles the ResNet-trigger log format (``val_bpb: 0.22``) and converts
    ``val_bpb`` → ``val_auc = 1 - val_bpb`` automatically.
    """
    metrics: dict[str, float] = {}
    for match in re.finditer(
        r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$",
        text,
        flags=re.MULTILINE,
    ):
        key, value = match.groups()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    if "val_bpb" in metrics and "val_auc" not in metrics:
        metrics["val_auc"] = 1.0 - metrics["val_bpb"]
    return metrics
