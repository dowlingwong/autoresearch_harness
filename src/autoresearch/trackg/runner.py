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

The timeout is a guard, not the budget: the budget is the fixed step or token
count of the node spec (R1-BX-D5).  R1-TRACKG-PORT-DECISIONS (4): a run that
exceeds a **measured** timeout (census baseline × factor 3) is a proposal
failure (``divergent``, reason ``timeout``) and is not rerun; only the
provisional timeout of the measuring run gives ``infrastructure_failure``.

With ``limits.cgroup_parent`` the run's processes join a fresh cgroup v2
directory, and a timeout or the end of the run kills the whole cgroup
(R1-HARNESS-ISOLATION).  A node with an ``isolation`` block runs isolated:
see :mod:`autoresearch.trackg.isolation` (frozen parent started without a
shell, worker sandbox, result channel, append-only log, frozen files pinned).
An isolated run refuses each missing host condition (no cgroup, root, no
worker sandbox, ``ptrace_scope`` 0) unless ``limits.isolation_waivers`` names
it with a reason; the record lists the waivers and says the run is then not
fully isolated.  Every run's working copy must match the frozen-file pin taken
from the node kit (and a campaign's expected incumbent code state) before it
runs: a mismatch raises :class:`IntegrityError` (STOP).
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

from autoresearch.trackg import determinism, isolation
from autoresearch.trackg.diff_support import DiffApplication, EvaluatedStates, apply_diff, code_state_hashes
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import INFRASTRUCTURE_FAILURE, ProcessExit, RunOutcome, parse_run_log, parse_run_text
from autoresearch.trackg.selection import parse_sealed_test_metric

RUN_RECORD_SCHEMA = "trackg_run/v1"
_GPU_UUID_RE = re.compile(r"^(GPU|MIG)-[0-9A-Za-z][0-9A-Za-z/\-]{7,}$")
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


WAIVERS = ("no_cgroup", "root", "no_worker_sandbox", "ptrace_scope", "timeout_factor")


class RunnerError(ValueError):
    """Raised for an invalid runner configuration (before anything runs)."""


class IntegrityError(RunnerError):
    """The working copy's frozen files differ from the pin taken from the node kit, or the incumbent's code
    state differs from the one the campaign expects: something outside the declared diffs changed them (STOP)."""


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
    cgroup_parent: str | None = None
    # waiver name -> reason; names: no_cgroup, root, no_worker_sandbox, ptrace_scope, timeout_factor
    isolation_waivers: dict[str, str] = field(default_factory=dict)

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
        unknown = set(self.isolation_waivers) - set(WAIVERS)
        if unknown or not all(isinstance(v, str) and v.strip() for v in self.isolation_waivers.values()):
            raise RunnerError(f"isolation_waivers: names must be among {WAIVERS} with a non-empty reason; got "
                              f"{dict(self.isolation_waivers)}")
        if self.cgroup_parent is not None and "no_cgroup" in self.isolation_waivers:
            raise RunnerError("give cgroup_parent or the no_cgroup waiver, not both")
        if self.measured_run_seconds is not None and self.timeout_factor != 3.0 and \
                "timeout_factor" not in self.isolation_waivers:
            raise RunnerError("R1-TRACKG-PORT-DECISIONS (4): the timeout is 3 x the measured run time; another "
                              "factor needs the timeout_factor waiver")
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

    @property
    def timeout_is_proposal_failure(self) -> bool:
        """R1-TRACKG-PORT-DECISIONS (4): only a measured timeout makes a timeout the proposal's failure."""
        return self.measured_run_seconds is not None

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
    def __init__(self, spec: TrackGNodeSpec, node_root: str | Path, runs_root: str | Path, *,
                 frozen_pin: dict[str, str] | None = None, protect_paths: tuple[str | Path, ...] = ()) -> None:
        self.spec = spec
        self.node_root = Path(node_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        if not self.node_root.is_dir():
            raise RunnerError(f"node root {self.node_root} is not a directory")
        if self.runs_root == self.node_root or self.node_root in self.runs_root.parents:
            raise RunnerError("runs_root must be outside the node root")
        self.base_spec = spec.to_base_node_spec()
        # The frozen files as the node kit has them now (a campaign passes the pin it took from the kit once).
        self.frozen_pin = dict(frozen_pin) if frozen_pin is not None else \
            isolation.frozen_file_hashes(self.node_root, spec.frozen_paths)
        self.protect_paths = tuple(Path(p).resolve() for p in protect_paths)

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
        expected_incumbent_state: str | None = None,
    ) -> RunResult:
        """Copy, optionally apply a diff, run under limits, classify and record.

        With *keep_code_snapshot* the copy is snapshotted to ``<run_dir>/code``
        after the diff is applied and before the run, so the code without any
        run outputs can be promoted as the next incumbent.
        """
        isolated = self.spec.isolation is not None
        waivers = limits.isolation_waivers
        if isolated:
            if limits.cgroup_parent is None and "no_cgroup" not in waivers:
                raise RunnerError("an isolated run needs limits.cgroup_parent (or the no_cgroup waiver)")
            if os.geteuid() == 0 and "root" not in waivers:
                raise RunnerError("an isolated run as root is not isolated (non-dumpable processes need a "
                                  "non-root user); the root waiver runs it anyway (tests)")
            scope = isolation.ptrace_scope()
            if (scope is None or scope < 1) and "ptrace_scope" not in waivers:
                raise RunnerError(f"Yama ptrace_scope is {scope}: an unrelated process of the same user could attach "
                                  "to the parent during its start-up; set it to 1 or give the ptrace_scope waiver")
        if limits.cpu_list is not None:
            available = os.sched_getaffinity(0)
            outside = sorted(set(limits.cpu_list) - available)
            if outside:
                raise RunnerError(f"cpu_list {outside} not in this process's CPU set {sorted(available)}")
        run_dir, work = self.prepare(run_id)
        copied = isolation.frozen_file_hashes(work, self.spec.frozen_paths)
        if copied != self.frozen_pin:
            changed = sorted(k for k in set(copied) | set(self.frozen_pin) if copied.get(k) != self.frozen_pin.get(k))
            raise IntegrityError(f"STOP: the frozen files of {self.node_root} differ from the pin taken from the node "
                                 f"kit: {changed[:10]}")
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
            # the census pins this hash for T1 (a committed manifest), so a kit changed before setup is caught too
            "frozen_pin_sha256": hashlib.sha256(json.dumps(self.frozen_pin, sort_keys=True).encode()).hexdigest(),
        }
        normalised, raw = code_state_hashes(work, self.base_spec)
        record["incumbent_code_state_sha256"], record["incumbent_code_state_raw_sha256"] = normalised, raw
        if expected_incumbent_state is not None and raw != expected_incumbent_state:
            raise IntegrityError(f"STOP: the incumbent {self.node_root} has code state {raw}, the campaign expects "
                                 f"{expected_incumbent_state}")
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

        log_path = run_dir / "run.log"
        result_text = None
        override = None
        if isolated:
            process, env_applied, det_record, elapsed, kill_info, iso_record, result_text, override = \
                self._execute_isolated(run_dir, work, limits, train_seed, split_seed, interpreter)
            record["isolation"] = iso_record
            command: Any = iso_record["argv"]
        else:
            process, env_applied, det_record, elapsed, kill_info = self._execute(
                run_dir, work, limits, train_seed, split_seed)
            command = self.spec.budgeted_run_command()
        if isolated:
            verified_text = record["isolation"].pop("_verified_log")
            outcome = override or parse_run_text(verified_text, process, self.spec,
                                                 timeout_is_proposal_failure=limits.timeout_is_proposal_failure,
                                                 result_text=result_text)
        else:
            outcome = parse_run_log(log_path, process, self.spec,
                                    timeout_is_proposal_failure=limits.timeout_is_proposal_failure)
        record.update({
            "executed": True,
            "command": command,
            "environment_applied": env_applied,
            "determinism": det_record,
            "process": {"returncode": process.returncode, "timed_out": process.timed_out,
                        "launch_error": process.launch_error, **kill_info},
            "elapsed_seconds_info_only": round(elapsed, 3),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None,
            "outcome": outcome.to_dict(),
        })
        sealed = parse_sealed_test_metric(
            result_text if isolated else
            (log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""), self.spec)
        record.update(sealed)
        # An infrastructure failure is not an evaluated state: it is rerun, not deduplicated.
        if evaluated is not None and register_evaluated and outcome.kind != INFRASTRUCTURE_FAILURE:
            evaluated.register(
                run_id, code_state_sha256=state,
                normalised_diff_sha256=application.provenance["normalised_diff_sha256"] if application else None,
                incumbent_code_state_sha256=record["incumbent_code_state_sha256"])
        self._write_record(run_dir, record)
        iso_rec = record.get("isolation") or {}
        if iso_rec.get("parent_traced_by") or iso_rec.get("parent_dumpable_at_accept"):
            raise IntegrityError(f"STOP: the run's parent was traced (TracerPid {iso_rec.get('parent_traced_by')}) or "
                                 f"dumpable at the accept ({iso_rec.get('parent_dumpable_at_accept')}): a process "
                                 f"outside the run interfered; see {run_dir / 'run_record.json'}")
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
        cgroup = self._make_cgroup(limits, run_dir.name)
        preexec = _limits_preexec(limits, cgroup)
        start = time.monotonic()
        with log_path.open("wb") as log:
            try:
                proc = subprocess.Popen(
                    self.spec.budgeted_run_command(), shell=True, cwd=work, env=env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, preexec_fn=preexec,
                )
            except (OSError, subprocess.SubprocessError) as error:
                # includes a failure inside preexec_fn (for example a CPU id the host lacks)
                cg = _close_cgroup(cgroup)
                return (ProcessExit(None, launch_error=repr(error)), applied,
                        determinism.read_record(run_dir / "determinism_shim.json", env), 0.0,
                        {"killed_with": None, "leftover_processes_killed": False, "cgroup": cg})
            timed_out = False
            try:
                returncode = proc.wait(timeout=timeout)
                cg_kill = cgroup.kill() if cgroup else None
                kill_info = {"killed_with": None,
                             "leftover_processes_killed": _kill_leftovers(proc.pid)}
            except subprocess.TimeoutExpired:
                timed_out = True
                cg_kill = cgroup.kill(limits.kill_grace_seconds) if cgroup else None
                killed_with = "cgroup.kill" if cgroup else _kill_group(proc, limits.kill_grace_seconds)
                if cgroup:
                    _kill_group(proc, limits.kill_grace_seconds)  # anything outside the cgroup in the session
                proc.wait()
                returncode = proc.returncode
                kill_info = {"killed_with": killed_with,
                             "leftover_processes_killed": killed_with in ("SIGKILL", "cgroup.kill")}
            kill_info["cgroup"] = _close_cgroup(cgroup, cg_kill)
        elapsed = time.monotonic() - start
        if not timed_out and elapsed > timeout:   # e.g. the runner was stopped while the run went on
            timed_out = True
            kill_info["timed_out_by_elapsed"] = round(elapsed, 3)
        det_record = determinism.read_record(run_dir / "determinism_shim.json", env)
        return ProcessExit(returncode, timed_out=timed_out), applied, det_record, elapsed, kill_info

    @staticmethod
    def _make_cgroup(limits: ResourceLimits, run_id: str) -> "isolation.RunCgroup | None":
        if limits.cgroup_parent is None:
            return None
        try:
            return isolation.RunCgroup.create(limits.cgroup_parent, run_id)
        except isolation.IsolationError as error:
            raise RunnerError(str(error)) from error

    def _sandbox_config(self, run_dir: Path, work: Path, interpreter: str) -> dict:
        """The worker's file system (see :mod:`autoresearch.trackg.isolation`): an allow-list.  Read-only: the system
        directories, the run interpreter's installation and ``sys.path``, the data, the determinism shim; ``/sys``
        without its submounts (no cgroup file system); writable: the working copy, with its frozen paths read-only;
        ``/dev`` as is; a private ``/tmp``.  Nothing else exists for the worker (no home directory, no runs tree, no
        ``/run``)."""
        ro: list[Path] = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc")]
        interp = _interpreter_paths(interpreter)
        # The sandbox recreates a symlink it is handed but does not follow it, so the interpreter's whole symlink
        # chain must be in the allow-list, not only its realpath: a uv venv's ``bin/python`` is a symlink to uv's
        # **unversioned** alias directory, whose target (the versioned directory) is what the realpath binds.
        # Without the alias the worker's PID 1 cannot exec ``sys.executable`` inside the sandbox (ENOENT).
        for candidate in (interp[1] if len(interp) > 1 else None, Path(interpreter)):
            if candidate is not None and os.path.isabs(str(candidate)):
                interp += [link for link in _symlink_chain(candidate) if link not in interp]
        home = Path(os.environ.get("HOME", "/nonexistent")).resolve()
        broad = [p for p in interp if p.resolve() in (Path("/"), home, self.runs_root) or
                 any(p.resolve() in q.parents for q in (home, self.runs_root))]
        if broad:   # review round 3: a PYTHONPATH entry such as the home directory would expose that whole tree
            raise RunnerError(f"the run interpreter's sys.path holds {broad}, which would expose the home directory "
                              "or the runs tree to the worker; remove them (PYTHONPATH)")
        ro += interp
        data = work / "data"
        if data.is_symlink():
            ro.append(data.resolve())
        ro.append((run_dir / "_ar_shim").resolve())
        ro = sorted({p for p in ro if os.path.lexists(p)}, key=lambda p: (len(p.parts), str(p)))
        ro_in_rw = sorted({str((work / entry.rstrip("/").removesuffix("/*")).resolve() if not
                               (work / entry.rstrip("/")).is_symlink() else work / entry.rstrip("/"))
                           for entry in self.spec.frozen_paths if os.path.lexists(work / entry.rstrip("/"))})
        root = run_dir / "_root"
        root.mkdir()
        # caches and a home directory inside the working copy (no real home directory exists for the worker)
        scratch = {"HOME": work / ".ar_home", "XDG_CACHE_HOME": work / ".ar_cache",
                   "TRITON_CACHE_DIR": work / ".ar_cache" / "triton",
                   "TORCHINDUCTOR_CACHE_DIR": work / ".ar_cache" / "inductor",
                   "MPLCONFIGDIR": work / ".ar_cache" / "matplotlib"}
        for path in scratch.values():
            path.mkdir(parents=True, exist_ok=True)
        env = {k: str(v) for k, v in scratch.items()}
        env[determinism.DETERMINISM_RECORD_ENV] = str(work / ".ar_determinism_shim.json")
        return {"root": str(root), "ro": [str(p) for p in ro], "ro_flat": ["/sys"], "rw": [str(work.resolve())],
                "ro_in_rw": ro_in_rw, "dev": "/dev", "env": env,
                "shared_dev_shm": bool(self.spec.isolation and self.spec.isolation.shared_dev_shm)}

    def _execute_isolated(self, run_dir: Path, work: Path, limits: ResourceLimits, train_seed: int | None,
                          split_seed: int | None, interpreter: str):
        """The isolated run (R1-HARNESS-ISOLATION); see :mod:`autoresearch.trackg.isolation`."""
        iso_spec = self.spec.isolation
        assert iso_spec is not None
        waivers = dict(limits.isolation_waivers)
        env, applied = self.build_environment(run_dir, limits, train_seed, split_seed)
        for name, value in iso_spec.env:
            env[name] = value
            applied[name] = value
        timeout, basis = limits.timeout()
        applied["timeout_seconds"] = f"{timeout:g} ({basis})"
        argv = iso_spec.argv(interpreter, self.spec.run_budget.value)
        euid = os.geteuid()
        isolation.set_non_dumpable()
        sandbox = None
        if "no_worker_sandbox" not in waivers:
            sandbox = self._sandbox_config(run_dir, work, interpreter)
            ok, why = isolation.worker_sandbox_available(interpreter, sandbox, run_dir / "_sandbox_probe")
            if not ok:
                raise RunnerError(f"this host refuses the worker sandbox ({why}); give the no_worker_sandbox waiver "
                                  "to run without it (then the run is not isolated)")
            env[isolation.SANDBOX_ENV] = json.dumps(sandbox)
            applied[determinism.DETERMINISM_RECORD_ENV] = sandbox["env"][determinism.DETERMINISM_RECORD_ENV]
        frozen_before = isolation.frozen_file_hashes(work, self.spec.frozen_paths)
        cgroup = self._make_cgroup(limits, run_dir.name)
        preexec_limits = _limits_preexec(limits, cgroup)
        log_path = run_dir / "run.log"
        start = time.monotonic()
        exit_ = isolation.execute(argv, cwd=work, env=env, log_path=log_path, timeout=timeout,
                                  cgroup=cgroup if cgroup is not None else _NoCgroup(),
                                  preexec=preexec_limits, drain_grace=limits.kill_grace_seconds,
                                  sweep=_kill_leftovers)
        elapsed = time.monotonic() - start
        cg_record = _close_cgroup(cgroup, exit_.kill if cgroup else None)
        verified = isolation.verify_log(log_path, exit_.log_sha256_written, exit_.log_bytes_written)
        log_intact = verified is not None
        frozen_after = isolation.frozen_file_hashes(work, self.spec.frozen_paths)
        changed = sorted(k for k in set(frozen_before) | set(frozen_after)
                         if frozen_before.get(k) != frozen_after.get(k))
        result_bytes = exit_.result_text.encode("utf-8")
        result_sha = hashlib.sha256(result_bytes).hexdigest()
        fd = os.open(run_dir / "result_channel.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(result_bytes)
        if log_intact:
            isolation.append_runner_line(log_path, f"AR_RUNNER result_channel_sha256={result_sha} "
                                                   f"bytes={len(result_bytes)}")
        process = ProcessExit(exit_.returncode, timed_out=exit_.timed_out, launch_error=exit_.launch_error)
        kill = exit_.kill or {}
        override = None
        if exit_.launch_error is None and not exit_.timed_out:   # the timeout rule decides first, as for any run
            if not log_intact:
                override = RunOutcome.divergent("log_tampered")
            elif changed:
                override = RunOutcome.divergent("frozen_file_changed", files=changed[:20])
            elif kill.get("result_channel_error"):
                override = RunOutcome.divergent("result_channel_error", error=kill["result_channel_error"])
            elif exit_.result_overflow:
                override = RunOutcome.divergent("result_channel_overflow",
                                                limit_bytes=isolation.RESULT_CHANNEL_MAX_BYTES)
        # A node option that reopens a channel the sandbox otherwise closes is an exception: listed next to the
        # waivers, and the run is not counted as effective (review round 4, N2).
        exceptions = ["shared_dev_shm"] if iso_spec.shared_dev_shm else []
        effective = (not waivers and not exceptions and euid != 0 and cgroup is not None
                     and sandbox is not None)
        iso_record = {
            "argv": argv,
            "effective": effective,
            "waivers": waivers,
            "exceptions": exceptions,
            "euid": euid,
            "ptrace_scope": isolation.ptrace_scope(),
            "runner_dumpable": isolation.is_dumpable(),
            "worker_sandbox": sandbox,
            "cgroup": cg_record,
            "nsdelegate": isolation.cgroup2_nsdelegate(cgroup.path.parent) if cgroup is not None else None,
            "log_intact": log_intact,
            "log_bytes_from_run": exit_.log_bytes_written,
            "result_channel_sha256": result_sha,
            "result_channel_overflow": exit_.result_overflow,
            "frozen_files_checked": len(frozen_before),
            "parent_traced_by": kill.get("parent_traced_by"),
            "parent_dumpable_at_accept": kill.get("parent_dumpable_at_accept"),
            "frozen_files_changed": changed,
            "limits": isolation.limits_text(),
            "_verified_log": (verified or b"").decode("utf-8", errors="replace"),
        }
        record_path = Path(sandbox["env"][determinism.DETERMINISM_RECORD_ENV]) if sandbox else \
            run_dir / "determinism_shim.json"
        det_record = determinism.read_record(record_path, env)
        kill_info = {"killed_with": ("cgroup.kill" if cgroup else "session_sweep") if exit_.timed_out else None,
                     "leftover_processes_killed": bool(kill.get("processes_before") or
                                                       kill.get("session_sweep_killed") or
                                                       kill.get("orphans_killed")),
                     "timed_out_by_elapsed": kill.get("timed_out_by_elapsed")}
        return process, applied, det_record, elapsed, kill_info, iso_record, exit_.result_text, override

    @staticmethod
    def _write_record(run_dir: Path, record: dict[str, Any]) -> None:
        target = run_dir / "run_record.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")


class _NoCgroup:
    """Stand-in when an isolated run has no cgroup (``no_cgroup`` waiver): only the session sweep kills."""

    def kill(self, grace: float = 10.0) -> dict[str, Any]:
        return {"method": "none", "processes_before": 0, "empty_after": True}

    def populated(self) -> bool:
        return False


_PREFIX_CACHE: dict[str, list[Path]] = {}


def _symlink_chain(path: str | Path) -> list[Path]:
    """Every symlink on the way from *path* to its target, outermost first.

    The worker sandbox rebuilds its allow-list with ``ar_isolated.bind()``, which recreates a symlink it is
    given (``os.readlink``) and does not follow it; every link on the way to the real file therefore has to be
    bound too.  A uv virtualenv's ``bin/python`` is a symlink to uv's unversioned alias directory
    (``…/cpython-3.12-linux-x86_64-gnu``), itself a symlink to the versioned directory whose ``bin/python3.12``
    is the real executable: only the realpath used to be bound, so the worker's PID 1 could not exec
    ``sys.executable`` (ENOENT)."""
    chain: list[Path] = []
    if not os.path.isabs(str(path)):
        return chain
    current = Path(str(path))
    for _ in range(40):     # a symlink loop is refused by the kernel anyway; do not spin here
        parts = current.parts
        link = next((Path(*parts[:index]) for index in range(1, len(parts) + 1)
                     if os.path.islink(Path(*parts[:index]))), None)
        if link is None:
            return chain
        chain.append(link)
        target = os.readlink(link)
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(str(link)), target)
        current = Path(os.path.normpath(os.path.join(target, *parts[len(link.parts):])))
    return chain


def _interpreter_paths(interpreter: str) -> list[Path]:
    """The run interpreter's executable, ``sys.prefix``, ``sys.base_prefix`` and ``sys.path`` entries (its
    installation, read-only for the worker)."""
    if interpreter not in _PREFIX_CACHE:
        code = ("import json, sys, os; print(json.dumps([os.path.realpath(sys.executable), sys.executable, sys.prefix, "
                "sys.base_prefix] + [p for p in sys.path if p]))")
        out = subprocess.run([interpreter, "-s", "-c", code], capture_output=True, text=True, timeout=60, cwd="/")
        if out.returncode != 0:
            raise RunnerError(f"cannot query {interpreter}: {out.stderr[-300:]}")
        _PREFIX_CACHE[interpreter] = [Path(p) for p in json.loads(out.stdout.strip().splitlines()[-1])]
    return list(_PREFIX_CACHE[interpreter])


def _close_cgroup(cgroup: "isolation.RunCgroup | None", kill: dict | None = None) -> dict[str, Any] | None:
    if cgroup is None:
        return None
    if cgroup.populated():
        kill = {**(kill or {}), "final": cgroup.kill()}
    return {"path": str(cgroup.path), "migration_possible": cgroup.migration_possible, "kill": kill,
            "removed": cgroup.remove()}


def _limits_preexec(limits: ResourceLimits, cgroup: "isolation.RunCgroup | None" = None):
    cpu_list = limits.cpu_list
    memory = limits.memory_bytes
    method = limits.memory_method
    join = isolation.join_cgroup_preexec(cgroup) if cgroup is not None else None

    def apply() -> None:  # runs in the child between fork and exec
        if cpu_list is not None:
            os.sched_setaffinity(0, set(cpu_list))
        if memory is not None and method != "external":
            which = resource.RLIMIT_AS if method == "rlimit_as" else resource.RLIMIT_DATA
            resource.setrlimit(which, (memory, memory))
        if join is not None:
            join()

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
