"""Per-run working-copy runner with resource limits (M10 step 0; TRACK_G_RUNBOOK §2).

For each run the runner:

1. copies the node (the incumbent) into ``<runs_root>/<run_id>/work`` -- it
   refuses to reuse an existing run directory, so nothing is overwritten;
2. optionally applies a ``manager_proposal/v2-diff`` diff there through
   :func:`autoresearch.trackg.diff_support.apply_diff` (scope, inert and
   duplicate checks); a refused, inert or duplicate diff is not run;
3. runs the node's run command with the fixed budget (``{budget}`` in the
   command and ``AR_BUDGET_KIND``/``AR_BUDGET_VALUE`` in the environment) and:

   - **GPU by UUID** (``CUDA_VISIBLE_DEVICES=<GPU-…>``; indices are refused;
     no UUID hides every GPU),
   - a **CPU cap** (``sched_setaffinity`` to a declared CPU list) and a
     **memory cap** (``RLIMIT_DATA`` or ``RLIMIT_AS``; or ``external`` when a
     scheduler cgroup such as Slurm's ``--mem`` enforces it, which is recorded),
   - a **timeout of factor × measured run time** (factor 3 by default; a
     provisional timeout is allowed only for the first, measuring run and is
     flagged in the record),
   - the node's fixed thread and data-loader worker counts,
   - the deterministic settings of :mod:`autoresearch.trackg.determinism`;

4. classifies the run with the base parser (:mod:`autoresearch.trackg.outcome`)
   and writes ``run_record.json`` beside the copy.

The timeout is an infrastructure guard, not the budget: the budget is the
fixed step or token count of the node spec (R1-BX-D5).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from autoresearch.trackg import determinism
from autoresearch.trackg.diff_support import DiffApplication, EvaluatedStates, apply_diff, code_state_hashes
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import INFRASTRUCTURE_FAILURE, ProcessExit, RunOutcome, parse_run_log
from autoresearch.trackg.selection import parse_sealed_test_metric

RUN_RECORD_SCHEMA = "trackg_run/v1"
_GPU_UUID_RE = re.compile(r"^(GPU|MIG)-[0-9A-Za-z][0-9A-Za-z/\-]{7,}$")
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


class RunnerError(ValueError):
    """Raised for an invalid runner configuration (before anything runs)."""


@dataclass(frozen=True)
class ResourceLimits:
    """Per-run limits.  CPU and memory caps are required: give ``cpu_list`` and either
    ``memory_bytes`` (enforced by an rlimit) or ``memory_method="external"`` with
    ``external_limit`` naming what enforces it (for example ``"slurm --mem=64G"``).
    A run without a cap needs an explicit ``uncapped_reason``, which is recorded."""

    gpu_uuids: tuple[str, ...] = ()
    cpu_list: tuple[int, ...] | None = None
    memory_bytes: int | None = None
    memory_method: Literal["rlimit_data", "rlimit_as", "external"] = "rlimit_data"
    measured_run_seconds: float | None = None
    timeout_factor: float = 3.0
    provisional_timeout_seconds: float | None = None
    kill_grace_seconds: float = 10.0
    external_limit: str | None = None
    uncapped_reason: str | None = None

    def __post_init__(self) -> None:
        for uuid in self.gpu_uuids:
            if not _GPU_UUID_RE.match(str(uuid)):
                raise RunnerError(f"GPU must be given by UUID (GPU-… or MIG-…), got {uuid!r}")
        if self.cpu_list is not None and (not self.cpu_list or any(c < 0 for c in self.cpu_list)):
            raise RunnerError("cpu_list must be a non-empty list of CPU ids")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise RunnerError("memory_bytes must be positive")
        if self.memory_method not in {"rlimit_data", "rlimit_as", "external"}:
            raise RunnerError("memory_method must be rlimit_data, rlimit_as or external")
        if self.timeout_factor <= 0:
            raise RunnerError("timeout_factor must be positive")
        if self.measured_run_seconds is None and self.provisional_timeout_seconds is None:
            raise RunnerError("give measured_run_seconds (timeout = factor × it) or, for the "
                              "first measuring run only, provisional_timeout_seconds")
        if self.measured_run_seconds is not None and self.measured_run_seconds <= 0:
            raise RunnerError("measured_run_seconds must be positive")
        if self.memory_method == "external" and not self.external_limit:
            raise RunnerError("memory_method 'external' needs external_limit (what enforces it)")
        if self.memory_method != "external" and self.external_limit:
            raise RunnerError("external_limit is only for memory_method 'external'")
        missing = self.missing_caps()
        if missing and not self.uncapped_reason:
            raise RunnerError(f"no {' and no '.join(missing)} cap: give one, or an explicit uncapped_reason")

    def missing_caps(self) -> list[str]:
        missing = []
        if self.cpu_list is None:
            missing.append("CPU")
        if self.memory_bytes is None and self.memory_method != "external":
            missing.append("memory")
        return missing

    def timeout(self) -> tuple[float, str]:
        if self.measured_run_seconds is not None:
            return self.timeout_factor * self.measured_run_seconds, (
                f"{self.timeout_factor:g} x measured {self.measured_run_seconds:g} s")
        return float(self.provisional_timeout_seconds), "provisional (measuring run)"  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_uuids"] = list(self.gpu_uuids)
        payload["cpu_list"] = list(self.cpu_list) if self.cpu_list is not None else None
        payload["uncapped"] = self.missing_caps()
        return payload


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    work_dir: Path
    outcome: RunOutcome | None
    record: dict[str, Any] = field(default_factory=dict)
    code_dir: Path | None = None


class WorkingCopyRunner:
    def __init__(self, spec: TrackGNodeSpec, node_root: str | Path, runs_root: str | Path) -> None:
        self.spec = spec
        self.node_root = Path(node_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        if not self.node_root.is_dir():
            raise RunnerError(f"node root {self.node_root} is not a directory")
        if self.runs_root == self.node_root or self.node_root in self.runs_root.parents:
            raise RunnerError("runs_root must be outside the node root")
        self.base_spec = spec.to_base_node_spec()

    # ------------------------------------------------------------------ copy
    def prepare(self, run_id: str) -> tuple[Path, Path]:
        if not _RUN_ID_RE.match(run_id):
            raise RunnerError(f"invalid run id {run_id!r}")
        run_dir = self.runs_root / run_id
        if run_dir.exists():
            raise RunnerError(f"run directory {run_dir} already exists; run ids are never reused")
        work = run_dir / "work"
        run_dir.mkdir(parents=True)
        shutil.copytree(self.node_root, work, ignore=_COPY_IGNORE, symlinks=True)
        return run_dir, work

    # ------------------------------------------------------------------ run
    def run(
        self,
        run_id: str,
        limits: ResourceLimits,
        *,
        train_seed: int | None = None,
        split_seed: int | None = None,
        diff_text: str | None = None,
        evaluated: EvaluatedStates | None = None,
        register_evaluated: bool = True,
        keep_code_snapshot: bool = True,
        interpreter: str = "python3",
    ) -> RunResult:
        """Copy, optionally apply a diff, run under limits, classify and record.

        With *keep_code_snapshot* the copy is snapshotted to ``<run_dir>/code``
        after the diff is applied and before the run, so the code without any
        run outputs can be promoted as the next incumbent.
        """
        if limits.cpu_list is not None:
            available = os.sched_getaffinity(0)
            outside = sorted(set(limits.cpu_list) - available)
            if outside:
                raise RunnerError(f"cpu_list {outside} not in this process's CPU set {sorted(available)}")
        run_dir, work = self.prepare(run_id)
        record: dict[str, Any] = {
            "schema": RUN_RECORD_SCHEMA,
            "run_id": run_id,
            "node": self.spec.name,
            "node_spec_sha256": self.spec.raw_sha256,
            "run_budget": {"kind": self.spec.run_budget.kind, "value": self.spec.run_budget.value},
            "workers": asdict(self.spec.workers),
            "seeds": {"train": train_seed, "split": split_seed},
            "limits": limits.to_dict(),
            "diff": None,
        }
        normalised, raw = code_state_hashes(work, self.base_spec)
        record["incumbent_code_state_sha256"], record["incumbent_code_state_raw_sha256"] = normalised, raw
        application: DiffApplication | None = None
        if diff_text is not None:
            # ``interpreter``: the run's Python, for the new-file shadowing check (kit review round 3 note)
            application = apply_diff(work, diff_text, self.base_spec, evaluated=evaluated,
                                     allow_new_files=self.spec.allow_new_files, interpreter=interpreter)
            record["diff"] = {"status": application.status, **application.provenance}
            if not application.applied:
                record["executed"] = False
                record["outcome"] = None
                self._write_record(run_dir, record)
                return RunResult(run_id, run_dir, work, None, record)
        state, state_raw = code_state_hashes(work, self.base_spec)
        record["code_state_sha256"], record["code_state_raw_sha256"] = state, state_raw
        code_dir = run_dir / "code"
        if keep_code_snapshot:
            shutil.copytree(work, code_dir, symlinks=True, ignore=_COPY_IGNORE)
            record["code_snapshot"] = "code"

        process, env_applied, det_record, elapsed, kill_info = self._execute(
            run_dir, work, limits, train_seed, split_seed)
        log_path = run_dir / "run.log"
        outcome = parse_run_log(log_path, process, self.spec)
        record.update({
            "executed": True,
            "command": self.spec.budgeted_run_command(),
            "environment_applied": env_applied,
            "determinism": det_record,
            "process": {"returncode": process.returncode, "timed_out": process.timed_out,
                        "launch_error": process.launch_error, **kill_info},
            "elapsed_seconds_info_only": round(elapsed, 3),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None,
            "outcome": outcome.to_dict(),
        })
        sealed = parse_sealed_test_metric(
            log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else "", self.spec)
        record.update(sealed)
        # An infrastructure failure is not an evaluated state: it is rerun, not deduplicated.
        if evaluated is not None and register_evaluated and outcome.kind != INFRASTRUCTURE_FAILURE:
            evaluated.register(
                run_id, code_state_sha256=state,
                normalised_diff_sha256=application.provenance["normalised_diff_sha256"] if application else None,
                incumbent_code_state_sha256=record["incumbent_code_state_sha256"])
        self._write_record(run_dir, record)
        return RunResult(run_id, run_dir, work, outcome, record, code_dir if keep_code_snapshot else None)

    # ------------------------------------------------------------------ internals
    def build_environment(self, run_dir: Path, limits: ResourceLimits, train_seed: int | None,
                          split_seed: int | None) -> tuple[dict[str, str], dict[str, str]]:
        shim_dir = run_dir / "_ar_shim"
        determinism.write_shim(shim_dir)
        applied: dict[str, str] = {}
        applied.update(determinism.environment_settings(
            self.spec.determinism, self.spec.workers.threads, self.spec.workers.dataloader_workers))
        applied["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        applied["CUDA_VISIBLE_DEVICES"] = ",".join(limits.gpu_uuids)
        applied["AR_BUDGET_KIND"] = self.spec.run_budget.kind
        applied["AR_BUDGET_VALUE"] = str(self.spec.run_budget.value)
        applied[determinism.DETERMINISM_RECORD_ENV] = str(run_dir / "determinism_shim.json")
        train_env, split_env = self.spec.seed_env
        if train_seed is not None:
            applied[train_env] = str(int(train_seed))
        if split_seed is not None:
            applied[split_env] = str(int(split_seed))
        env = dict(os.environ)
        env.update(applied)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(shim_dir) + (os.pathsep + existing if existing else "")
        applied["PYTHONPATH(prefix)"] = str(shim_dir)
        return env, applied

    def _execute(self, run_dir: Path, work: Path, limits: ResourceLimits, train_seed: int | None,
                 split_seed: int | None) -> tuple[ProcessExit, dict[str, str], dict[str, Any], float, dict]:
        env, applied = self.build_environment(run_dir, limits, train_seed, split_seed)
        timeout, basis = limits.timeout()
        applied["timeout_seconds"] = f"{timeout:g} ({basis})"
        log_path = run_dir / "run.log"
        start = time.monotonic()
        with log_path.open("wb") as log:
            try:
                proc = subprocess.Popen(
                    self.spec.budgeted_run_command(), shell=True, cwd=work, env=env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, preexec_fn=_limits_preexec(limits),
                )
            except (OSError, subprocess.SubprocessError) as error:
                # includes a failure inside preexec_fn (for example a CPU id the host lacks)
                return (ProcessExit(None, launch_error=repr(error)), applied,
                        determinism.read_record(run_dir / "determinism_shim.json", env), 0.0,
                        {"killed_with": None, "leftover_processes_killed": False})
            timed_out = False
            try:
                returncode = proc.wait(timeout=timeout)
                kill_info = {"killed_with": None,
                             "leftover_processes_killed": _kill_leftovers(proc.pid)}
            except subprocess.TimeoutExpired:
                timed_out = True
                killed_with = _kill_group(proc, limits.kill_grace_seconds)
                returncode = proc.returncode
                kill_info = {"killed_with": killed_with, "leftover_processes_killed": killed_with == "SIGKILL"}
        elapsed = time.monotonic() - start
        det_record = determinism.read_record(run_dir / "determinism_shim.json", env)
        return ProcessExit(returncode, timed_out=timed_out), applied, det_record, elapsed, kill_info

    @staticmethod
    def _write_record(run_dir: Path, record: dict[str, Any]) -> None:
        target = run_dir / "run_record.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")


def _limits_preexec(limits: ResourceLimits):
    cpu_list = limits.cpu_list
    memory = limits.memory_bytes
    method = limits.memory_method

    def apply() -> None:  # runs in the child between fork and exec
        if cpu_list is not None:
            os.sched_setaffinity(0, set(cpu_list))
        if memory is not None and method != "external":
            which = resource.RLIMIT_AS if method == "rlimit_as" else resource.RLIMIT_DATA
            resource.setrlimit(which, (memory, memory))

    return apply


def _run_members(leader: int) -> list[int] | None:
    """Non-zombie processes in the run's process group **or session** (both have the id of the
    session leader, because the shell starts with ``start_new_session``).  A child that left
    the group with ``setpgid`` but stayed in the session is included; one that called
    ``setsid`` is not.  ``None`` if ``/proc`` is unavailable."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    members = []
    me = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        if len(fields) > 3 and fields[0] != "Z" and str(leader) in (fields[2], fields[3]):
            members.append(int(entry.name))
    return members


def _group_alive(pgid: int) -> bool:
    """Whether any non-zombie process is in the run's process group or session."""
    members = _run_members(pgid)
    if members is not None:
        return bool(members)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sigkill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in _run_members(pgid) or []:  # session members outside the group
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _kill_group(proc: subprocess.Popen, grace: float) -> str:
    """SIGTERM the run's process group; after *grace* seconds SIGKILL it if any member is
    still alive, whatever the shell's state (a child may ignore SIGTERM)."""
    pgid = proc.pid  # start_new_session: the shell leads its own session and group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    while time.monotonic() < deadline and _group_alive(pgid):
        time.sleep(0.05)
    killed_with = "SIGTERM"
    if _group_alive(pgid):
        _sigkill_group(pgid)
        killed_with = "SIGKILL"
    proc.wait()
    return killed_with


def _kill_leftovers(pgid: int) -> bool:
    """After a normal exit, SIGKILL any process left in the run's group (orphaned workers)."""
    if _group_alive(pgid):
        _sigkill_group(pgid)
        return True
    return False
