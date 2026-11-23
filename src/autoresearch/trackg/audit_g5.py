"""G.5 T1 label engines: one campaign wave, or one audit chunk, of one ported node (the G.5 harness unit).

The paper repository's G.5 job (``sweep2/t1_tasl/g5_job.py``) writes the input file, starts this module with
``python -m autoresearch.trackg.audit_g5 <subcommand> ...`` inside its delegated scope on TASL-2, and reads back
only the named output members.  This module never reads a label, an evaluation result or a sealed file; it runs
code states and writes what it ran.  AI-assisted.

The contract (``g5_job.engine_contract()``), item by item:

* ``ENGINE_CONTRACT = "g5_tasl_engine_contract/1"``, and both input schemas, are in this file (``check_engines``
  greps for the three strings at the G.5 commit).
* ``campaign --campaign <trackg_g5_campaign/v1> --harness-root <h> --out <dir> [--gpu-uuid <uuid>]`` runs the
  wave's campaign instances in order, each with :func:`~autoresearch.trackg.campaign_v2_diff.run_v2_diff_campaign`
  (``manager_proposal/v2-diff``) as G.4 does, and writes ``<out>/instance_<k>_<mode>/campaign_ledger.jsonl`` (the
  G.4 ledger schema ``trackg_v2_diff_campaign_ledger/v1``; every executed run record carries ``executed`` and
  ``elapsed_seconds_info_only``, every manager call its ``request`` with ``cache_prompt: false``) and
  ``<out>/campaign_record.json`` (``stop``).  Every run record in a ledger that has a code state carries
  ``code_state_sha256`` at its top level, a not-run duplicate or inert diff included (:class:`G5CampaignRunner`
  lifts it from the diff provenance; the paper's assembly reads it there).  Each instance has training seed = split
  seed = manager seed = 5100 + k, inside 5101-5112, and k must be one of the wave's instances (wave 1: 1-4; wave 2:
  5-12).
* **Deadline** (review M1).  The job kills the engine at ``deadline_epoch``, and an isolated run it supervised would
  outlive it.  So no run starts (campaign or audit) unless 3 x census + 60 s fit before the deadline, and no manager
  call starts unless 120 s more fit as well (:class:`CampaignGuard`); the engine then stops with ``stop_kind:
  "deadline"``.
* A reply whose completion reached the cap (``usage.completion_tokens >= max_tokens``, the counting rule of the
  job's diagnostics) is an ``invalid_proposal`` (R1-EXAM-B1-TRUNCATION-INVALID): :class:`TruncationInvalidManager`
  applies it to G.5 campaigns only, so ``manager_v2_diff.py`` and G.4's pilot are unchanged.
* ``audit --chunk <trackg_g5_audit_chunk/v1> --harness-root <h> --out <dir> [--gpu-uuid <uuid>]`` re-materialises
  every unit (the baseline plus the unit's ordered diffs, applied with the runner's own diff code), checks that its
  normalised code-state hash equals the chunk's ``code_state_sha256`` before anything runs, and then runs each
  planned (unit, seed) through :class:`~autoresearch.trackg.runner.WorkingCopyRunner` with training seed = split
  seed = the audit seed (``split_policy: "vary"``) and the unisolated-run gate
  (:func:`autoresearch.trackg.noise_floor.check_effective`).  It writes ``<out>/rows.jsonl``
  (``trackg_g5_audit_row/v1``, one row per planned run) and ``<out>/chunk_record.json`` (``stop``).  The audit
  seeds are the chunk's K block: K = 10 is 7001-7010, K = 6 is 7001-7003 and 7006-7008.
* Row kinds: ``ok`` (an executed run with a finite ``outcome.value``, the validation metric) and ``divergent`` (an
  executed run without one) carry the run's isolation block.  An infrastructure failure is rerun up to 2 times; if
  every attempt is an infrastructure failure the run is ONE row of kind ``infrastructure_failure`` (``executed`` may
  be false and the isolation block absent).
* **Edit-caused failures are divergent, never infrastructure** (3b-1 review, ruling 1).  The runner's classifier
  (:mod:`autoresearch.trackg.outcome`) already makes a non-finite loss or metric, a missing metric, a measured
  timeout and a non-zero exit (the edited code crashed in the worker, so the frozen parent failed) divergent.  Of
  its infrastructure outcomes, :func:`attribute` gives three to the code state: a kill while the job scope's memory
  limit was hit (``oom`` and ``oom_kill`` both rose in its ``memory.events``), a fault signal of the frozen parent
  that recurs on the rerun of the same code state and seed, and a default infrastructure log pattern that appears
  only after the run's first ``AR_DEVICE`` line.  A kill without that evidence, a first fault signal, a launch error
  and other log patterns stay infrastructure failures and are rerun.  The audit and (through
  :class:`G5CampaignRunner`) the campaign apply the same rules.
* **Records** carry ``stop`` (text) and ``stop_kind``, one of ``refused`` (input, pins, node), ``deadline``,
  ``integrity``, ``unisolated``, ``transport``, ``baseline_failed``, ``runner`` (a host refusal outside a run
  attempt) and ``error`` (anything else, caught so that a record is always written).
* Members: the job reads only the ledgers, ``out/chunk.json`` (its own file) and ``out/audit/rows.jsonl``.  The run
  directories under ``<out>`` (run records, logs, result channels; campaign working copies and code snapshots with
  any binaries a run wrote) stay in ``out.tar`` unread.  As in G.1b and G.4, a run record on disk keeps the sealed
  test block the runner parsed; no row and no ledger line carries it.
* The engines' stop text names a cause and an index, never a metric value or a per-seed outcome (round-2 review
  n1).  The job puts a fixed reason in its clear files anyway.

The guards of the existing callers are unchanged: ``pilot_g4`` keeps its four pilot nodes, seeds 4001-4099 and
ENGINEERING_ONLY; ``noise_floor`` keeps its declared-edit plan and refuses 7001-7010 and 9001-9999.  This module
has its own entry points and seed rules, and only reuses their pieces (the record gate, the node-pin check, the
CPU-only node set, the manager settings and the timed transport).  Nothing in ``noise_floor``'s import closure
imports this module (P5).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from autoresearch.trackg import gpu_identity
from autoresearch.trackg.campaign_v2_diff import V2DiffCampaignError, run_v2_diff_campaign
from autoresearch.trackg.diff_support import apply_diff, code_state_hashes
from autoresearch.trackg.manager_v2_diff import Chat, ManagerTransportStop, V2DiffManager, openai_compatible_chat
from autoresearch.trackg.node_template import TrackGNodeSpec, load_trackg_node_spec
from autoresearch.trackg.noise_floor import CPU_ONLY_NODES, NoiseFloorStop, check_effective, check_node_pins
from autoresearch.trackg.outcome import (DEFAULT_INFRASTRUCTURE_LOG_PATTERNS, DIVERGENT, INFRASTRUCTURE_FAILURE, OK,
                                         RunOutcome)
from autoresearch.trackg.pilot_g4 import MAX_TOKENS, MEMORY_MODES, timed_chat
from autoresearch.trackg.runner import _COPY_IGNORE as RUNNER_COPY_IGNORE
from autoresearch.trackg.runner import IntegrityError, ResourceLimits, RunnerError, RunResult, WorkingCopyRunner

ENGINE_CONTRACT = "g5_tasl_engine_contract/1"
CAMPAIGN_SCHEMA = "trackg_g5_campaign/v1"
CHUNK_SCHEMA = "trackg_g5_audit_chunk/v1"
ROW_SCHEMA = "trackg_g5_audit_row/v1"
CAMPAIGN_RECORD_SCHEMA = "trackg_g5_campaign_record/v1"
CHUNK_RECORD_SCHEMA = "trackg_g5_audit_chunk_record/v1"
LABEL = "EVIDENCE"
LEDGER_NAME = "campaign_ledger.jsonl"

INSTANCE_SEED_BASE = 5100                          # instance k: training = split = manager seed 5100 + k (§5)
INSTANCE_SEEDS = range(5101, 5113)                 # G.5's block; disjoint from G.4 4001-4099 and audit 7001-7010
WAVE_INSTANCES = {1: range(1, 5), 2: range(5, 13)}
AUDIT_SEED_BLOCKS = {10: (range(7001, 7006), range(7006, 7011)),       # (label, evaluation) per K
                     6: (range(7001, 7004), range(7006, 7009))}
INFRASTRUCTURE_RERUNS = 2
MAX_MEASURED_RUN_SECONDS = 300.0                   # as noise_floor and pilot_g4: census run time in (0, 300]
CRASH_SIGNALS = frozenset({signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGILL, signal.SIGABRT,
                           signal.SIGTRAP, signal.SIGSYS})
KILL_EXIT_SIGNALS = {137: signal.SIGKILL, 143: signal.SIGTERM}         # shell-reported, as outcome.py's defaults
RUN_DEADLINE_MARGIN_SECONDS = 60.0                 # a run starts only if 3 x census + 60 s fit before the deadline
LLM_CALL_ALLOWANCE_SECONDS = 120.0                 # a manager call starts only if this plus one run fit as well
AR_DEVICE_LINE = re.compile(r"^AR_DEVICE ", re.MULTILINE)   # printed by every GPU node before editable code runs
STOP_KINDS = ("refused", "deadline", "integrity", "unisolated", "transport", "baseline_failed", "runner", "error")
CAMPAIGN_KEYS = frozenset({
    "schema", "label", "run_id", "node", "family", "wave", "spec_path", "node_dir", "node_spec_sha256",
    "committed_pin_sha256", "data_manifest_sha256", "proposal_schema", "extra_payload", "budget", "w_semantics",
    "instances", "max_tokens", "temperature", "measured_run_seconds", "timeout_factor", "base_url", "model",
    "cgroup_parent", "interpreter", "isolation_waivers", "deadline_epoch", "memory_limit"})
INSTANCE_KEYS = frozenset({"instance", "train_seed", "split_seed", "manager_seed", "memory_mode"})
CHUNK_KEYS = frozenset({
    "schema", "label", "run_id", "node", "family", "wave", "chunk", "K", "spec_path", "node_dir",
    "node_spec_sha256", "committed_pin_sha256", "data_manifest_sha256", "measured_run_seconds", "timeout_factor",
    "split_policy", "units", "runs", "interpreter", "cgroup_parent", "isolation_waivers", "record_gate",
    "deadline_epoch", "memory_limit"})
RUN_KEYS = frozenset({"unit", "seed", "train_seed", "split_seed"})
ISOLATION_KEYS = ("effective", "waivers", "exceptions", "log_intact", "parent_traced_by", "parent_dumpable_at_accept")
GATE_TEXT = "noise_floor.check_effective: effective:true, waivers {}, exceptions [], log intact, parent not traced"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_NODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class G5EngineStop(RuntimeError):
    """A pre-declared STOP of a G.5 engine: the wave or chunk ends; what it wrote stays on disk.  ``kind`` is one of
    :data:`STOP_KINDS` and goes into the record's ``stop_kind``."""

    def __init__(self, message: str, kind: str = "refused") -> None:
        super().__init__(message)
        if kind not in STOP_KINDS:
            raise ValueError(f"unknown stop kind {kind!r}")
        self.kind = kind


def _need(condition: bool, message: str, kind: str = "refused") -> None:
    if not condition:
        raise G5EngineStop(message, kind)


_LEDGER_STOP_KINDS = {"stop_unisolated": "unisolated", "baseline_failed": "baseline_failed",
                      "stop_integrity": "integrity", "stop_transport": "transport"}


def _stop_kind(exc: BaseException, ledger: Path | None = None) -> str:
    """The structured class of an engine stop (the record's ``stop_kind``), so the job need not parse the text."""
    if isinstance(exc, G5EngineStop):
        return exc.kind
    if isinstance(exc, IntegrityError):
        return "integrity"
    if isinstance(exc, ManagerTransportStop):
        return "transport"
    if isinstance(exc, V2DiffCampaignError) and ledger is not None and ledger.is_file():
        lines = [x for x in ledger.read_text().splitlines() if x.strip()]
        if lines:
            return _LEDGER_STOP_KINDS.get(json.loads(lines[-1]).get("decision"), "error")
    if isinstance(exc, RunnerError):
        return "runner"
    if isinstance(exc, gpu_identity.GpuIdentityError):
        return "refused"
    return "error"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stop_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:600]


# =============================================================================================== shared checks
def _check_common(data: dict[str, Any]) -> None:
    """Fields both input files share (the job writes them from the same node entry)."""
    _need(data.get("label") == LABEL, f"STOP: a G.5 engine file is {LABEL}")
    _need(isinstance(data.get("node"), str) and bool(_NODE.fullmatch(data["node"])),
          f"STOP: invalid node id {data.get('node')!r}")
    for key in ("run_id", "family", "spec_path", "node_dir", "cgroup_parent", "interpreter", "memory_limit"):
        _need(isinstance(data.get(key), str) and bool(data[key].strip()), f"STOP: {key} must be a non-empty string")
    for key in ("spec_path", "node_dir"):
        _need(not data[key].startswith("/") and ".." not in Path(data[key]).parts,
              f"STOP: {key} must be a relative path inside the harness checkout")
    _need(type(data.get("wave")) is int and data["wave"] in WAVE_INSTANCES, "STOP: wave must be 1 or 2")
    for key in ("node_spec_sha256", "committed_pin_sha256"):
        _need(isinstance(data.get(key), str) and bool(_SHA.fullmatch(data[key])), f"STOP: {key} must be a sha256")
    dm = data.get("data_manifest_sha256")
    _need(dm is None or (isinstance(dm, str) and bool(_SHA.fullmatch(dm))),
          "STOP: data_manifest_sha256 must be a sha256, or null for a node without data")
    seconds = data.get("measured_run_seconds")
    _need(_is_number(seconds) and 0 < seconds <= MAX_MEASURED_RUN_SECONDS,
          f"STOP: measured_run_seconds (the census timeout basis) must be in (0, {MAX_MEASURED_RUN_SECONDS:g}]")
    _need(data.get("timeout_factor") == 3.0, "STOP: R1-TRACKG-PORT-DECISIONS (4): the timeout is 3 x the census time")
    _need(data.get("isolation_waivers") == {}, "STOP: a G.5 engine file carries no isolation waiver")
    _need(_is_number(data.get("deadline_epoch")), "STOP: deadline_epoch must be a number")


def load_node(data: dict[str, Any], harness_root: Path) -> tuple[TrackGNodeSpec, Path, dict[str, str]]:
    """The node from the harness checkout, checked against the file's pins (the census's committed-file pin and the
    data manifest, as noise_floor checks them); returns the spec, the node kit and its full frozen pin."""
    root = Path(harness_root).resolve()
    spec_path = (root / data["spec_path"]).resolve()
    node_dir = (root / data["node_dir"]).resolve()
    _need(root in node_dir.parents and root in spec_path.parents,
          "STOP: spec_path and node_dir must lie inside the harness checkout")
    spec = load_trackg_node_spec(spec_path)
    _need(spec.name == data["node"], f"STOP: {data['spec_path']} is node {spec.name}, the file names {data['node']}")
    _need(spec.raw_sha256 == data["node_spec_sha256"], "STOP: the node spec sha256 differs from the file's pin")
    _need(spec.isolation is not None, f"STOP: {data['node']} has no isolation block: unisolated nodes are refused")
    try:
        pin = check_node_pins(spec, node_dir, data["committed_pin_sha256"], data["data_manifest_sha256"])
    except NoiseFloorStop as exc:
        raise G5EngineStop(str(exc), "refused") from exc
    return spec, node_dir, pin


def build_limits(data: dict[str, Any], gpu_uuids: tuple[str, ...], cpu_list: tuple[int, ...]) -> ResourceLimits:
    """The run limits: census timeout basis x 3, the job's delegated cgroup and external memory cap, no waiver."""
    return ResourceLimits(gpu_uuids=gpu_uuids, cpu_list=cpu_list, memory_bytes=None, memory_method="external",
                          external_limit=data["memory_limit"], measured_run_seconds=float(data["measured_run_seconds"]),
                          timeout_factor=3.0, cgroup_parent=data["cgroup_parent"], isolation_waivers={})


def resolve_gpus(node: str, gpu_uuid: str | None, env: Mapping[str, str],
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> tuple[str, ...]:
    """A direct host run on TASL-2 (R1-T1-HOST-TASL): a GPU node names its leased GPU with ``--gpu-uuid``; a CPU-only
    node (noise_floor's set) runs without any GPU and with ``CUDA_VISIBLE_DEVICES`` empty."""
    if gpu_uuid is None:
        _need(node in CPU_ONLY_NODES,
              f"STOP: {node} needs --gpu-uuid (only {sorted(CPU_ONLY_NODES)} run without a GPU)")
        _need(not any(env.get(name) for name in gpu_identity.SOURCES)
              and env.get("CUDA_VISIBLE_DEVICES", "") == "",
              "STOP: a CPU-only run needs no scheduler GPU variable and CUDA_VISIBLE_DEVICES empty")
        return ()
    _need(node not in CPU_ONLY_NODES, f"STOP: {node} is a CPU-only node; it runs without --gpu-uuid")
    gpus, _ = gpu_identity.resolve_explicit_on_host(gpu_uuid, env, runner=runner)
    _need(len(gpus) == 1, f"STOP: one GPU per G.5 engine run, got {gpus}")
    return gpus


# =============================================================================================== campaign
def validate_campaign(data: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed validation of a ``trackg_g5_campaign/v1`` file (``g5_job.campaign_file`` writes it)."""
    _need(isinstance(data, dict) and data.get("schema") == CAMPAIGN_SCHEMA,
          f"STOP: campaign file must be {CAMPAIGN_SCHEMA}")
    _need(set(data) == CAMPAIGN_KEYS, f"STOP: campaign file keys differ from {CAMPAIGN_SCHEMA}: "
                                      f"{sorted(set(data) ^ CAMPAIGN_KEYS)}")
    _check_common(data)
    _need(data["proposal_schema"] == "manager_proposal/v2-diff", "STOP: G.5 campaigns use manager_proposal/v2-diff")
    _need(data["extra_payload"] == {"cache_prompt": False},
          "STOP: every request carries cache_prompt: false and nothing else extra")
    _need(data["w_semantics"] == "manager_calls", "STOP: W counts manager calls (G5-W A); G5-W B is not built")
    _need(type(data["budget"]) is int and 1 <= data["budget"] <= 50, "STOP: budget (W) must be an integer in [1, 50]")
    _need(data["max_tokens"] == MAX_TOKENS, f"STOP: max_tokens must be the completion cap {MAX_TOKENS}")
    t = data["temperature"]
    _need(_is_number(t) and 0 <= t <= 2, "STOP: temperature must be a number in [0, 2]")
    for key in ("base_url", "model"):
        _need(isinstance(data[key], str) and bool(data[key].strip()), f"STOP: {key} must be a non-empty string")
    instances = data["instances"]
    _need(isinstance(instances, list) and bool(instances), "STOP: instances must be a non-empty list")
    wave = WAVE_INSTANCES[data["wave"]]
    seen: set[tuple[int, str]] = set()
    by_mode: dict[str, set[int]] = {}
    for inst in instances:
        _need(isinstance(inst, dict) and set(inst) == INSTANCE_KEYS,
              f"STOP: an instance has exactly {sorted(INSTANCE_KEYS)}")
        k, mode = inst["instance"], inst["memory_mode"]
        _need(type(k) is int and k in wave, f"STOP 10: instance {k!r} is not an instance of wave {data['wave']}")
        _need(mode in MEMORY_MODES, f"STOP: memory_mode must be one of {sorted(MEMORY_MODES)}")
        for key in ("train_seed", "split_seed", "manager_seed"):
            seed = inst[key]
            _need(type(seed) is int and seed in INSTANCE_SEEDS,
                  f"STOP 10: {key} {seed!r} is outside the G.5 instance seeds 5101-5112")
            _need(seed == INSTANCE_SEED_BASE + k,
                  f"STOP 10: instance {k} has {key} {seed}, not {INSTANCE_SEED_BASE + k}")
        _need((k, mode) not in seen, f"STOP: instance {k} appears twice in memory mode {mode}")
        seen.add((k, mode))
        by_mode.setdefault(mode, set()).add(k)
    for mode, ks in by_mode.items():
        _need(ks == set(wave), f"STOP: memory mode {mode} does not hold exactly the wave's instances {list(wave)}")
    return data


def instance_dir_name(inst: dict[str, Any]) -> str:
    """``instance_<k>_<mode>``: the member name the job's ledger pattern reads."""
    return f"instance_{inst['instance']}_{inst['memory_mode']}"


@dataclass
class CampaignGuard:
    """What a G.5 campaign checks before every run and every manager call (review M1): the job kills the engine at
    ``deadline_epoch``, and an isolated run it was supervising would outlive it, so nothing starts that could end
    past the deadline.  It also carries the ``memory.events`` source and the crash-signal counts for
    :func:`attribute` (the same rules as the audit)."""

    deadline_epoch: float
    run_seconds: float                       # 3 x census + RUN_DEADLINE_MARGIN_SECONDS
    clock: Callable[[], float]
    memory_cgroup: Path | None = None
    memory_events: Callable[[Path | None], dict[str, int] | None] | None = None
    crashes: dict[tuple[str, int | None], int] = field(default_factory=dict)

    def check_run(self, run_id: str) -> None:
        _need(self.clock() + self.run_seconds <= self.deadline_epoch,
              f"STOP: deadline: campaign run {run_id} could not finish before the wave's deadline", "deadline")

    def check_call(self, step: int) -> None:
        _need(self.clock() + LLM_CALL_ALLOWANCE_SECONDS + self.run_seconds <= self.deadline_epoch,
              f"STOP: deadline: the manager call of step {step} and its run could not finish before the wave's "
              f"deadline", "deadline")

    def events(self) -> dict[str, int] | None:
        return self.memory_events(self.memory_cgroup) if self.memory_events else None


class G5CampaignRunner(WorkingCopyRunner):
    """The campaign's runner, with three additions for G.5 (``run_record.json`` on disk stays the runner's own):

    * before every run, ``guard.check_run`` (review M1): no run starts unless 3 x census + 60 s fit before the
      wave's deadline (:class:`G5EngineStop`, kind ``deadline``; ``run_v2_diff_campaign`` lets it propagate);
    * a diff that was not run (duplicate, inert or refused) gets its code state from the diff provenance at the top
      level of the run record (``code_state_sha256``, ``code_state_raw_sha256``, ``code_state_source:
      "diff_provenance"``); a refused diff whose provenance has no code state gets none;
    * an executed run's infrastructure outcome goes through :func:`attribute`, as in the audit (an OOM kill at the
      scope's memory limit, a fault signal that recurs on the rerun, a default infrastructure log pattern only after
      ``AR_DEVICE`` are divergent); the record then carries ``outcome_attribution`` and the new outcome, and a state
      made divergent is registered as evaluated, as the runner registers any non-infrastructure run."""

    def __init__(self, *args, guard: CampaignGuard | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.guard = guard

    def run(self, run_id, limits, **kw):  # noqa: ANN001 - the parent's signature
        guard = self.guard
        if guard is not None:
            guard.check_run(run_id)
        before = guard.events() if guard is not None else None
        result = super().run(run_id, limits, **kw)
        after = guard.events() if guard is not None else None
        rec = result.record
        diff = rec.get("diff") or {}
        if not rec.get("executed") and "code_state_sha256" not in rec and diff.get("code_state_sha256"):
            rec["code_state_sha256"] = diff["code_state_sha256"]
            rec["code_state_raw_sha256"] = diff.get("code_state_raw_sha256")
            rec["code_state_source"] = "diff_provenance"
        if guard is not None and rec.get("executed") and result.outcome is not None:
            key = (rec.get("code_state_sha256"), kw.get("train_seed"))
            prior = guard.crashes.get(key, 0)
            outcome, why = attribute(result.outcome, before, after, log_text=_run_log(result),
                                     prior_crash=prior > 0)
            if why is not None:
                rec["outcome_attribution"] = why
                if why.get("crash_signal"):
                    guard.crashes[key] = prior + 1
            if outcome is not result.outcome:
                result.outcome = outcome
                rec["outcome"] = outcome.to_dict()
                evaluated = kw.get("evaluated")
                if evaluated is not None and kw.get("register_evaluated", True):
                    evaluated.register(run_id, code_state_sha256=rec["code_state_sha256"],
                                       normalised_diff_sha256=diff.get("normalised_diff_sha256"),
                                       incumbent_code_state_sha256=rec.get("incumbent_code_state_sha256"))
        return result


def _run_log(result: RunResult) -> str | None:
    path = result.run_dir / "run.log"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


@dataclass
class TruncationInvalidManager(V2DiffManager):
    """The v2-diff manager with R1-EXAM-B1-TRUNCATION-INVALID: a reply whose completion reached the cap is an
    invalid proposal (charged, nothing runs), whether or not it parses.  The rule is the job's counting rule
    (``usage.completion_tokens >= max_tokens``), so its diagnostics never see a reply at the cap that ran.  With a
    ``guard``, no call starts unless the call allowance plus one run fit before the wave's deadline (review M1)."""

    guard: CampaignGuard | None = None

    def propose(self, spec, incumbent_dir, history, step):  # noqa: ANN001 - the parent's signature
        if self.guard is not None:
            self.guard.check_call(step)
        proposal, call = super().propose(spec, incumbent_dir, history, step)
        usage = call.get("usage") or {}
        try:
            truncated = float(usage.get("completion_tokens") or 0) >= self.max_tokens
        except (TypeError, ValueError):
            truncated = False
        call["truncated"] = bool(truncated)
        if truncated and proposal is not None:
            call["parse_error"] = (f"truncated: the completion reached the cap of {self.max_tokens} tokens "
                                   "(R1-EXAM-B1-TRUNCATION-INVALID); the reply is not used")
            return None, call
        return proposal, call


def run_campaign(data: dict[str, Any], harness_root: Path, out_dir: Path, *, gpu_uuids: tuple[str, ...],
                 cpu_list: tuple[int, ...], chat: Chat | None = None, campaign=run_v2_diff_campaign,
                 gate=check_effective, clock: Callable[[], float] = time.time,
                 memory_events: Callable[[Path | None], dict[str, int] | None] | None = None) -> dict[str, Any]:
    """One campaign wave of one node: the instances in order, each a G.4 v2-diff campaign with the record gate on
    every executed run, the deadline checked before every run and manager call (:class:`CampaignGuard`) and
    :func:`attribute` applied to every executed run (:class:`G5CampaignRunner`).  The first instance that STOPs ends
    the wave: its ledger holds the STOP line (a gate refusal, a failed baseline, an integrity or transport stop), or,
    for a deadline stop, simply ends; ``campaign_record.json`` names it in ``stop`` and ``stop_kind``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {"schema": CAMPAIGN_RECORD_SCHEMA, "label": LABEL, "engine_contract": ENGINE_CONTRACT,
                              "node": data["node"], "run_id": data["run_id"], "wave": data["wave"], "instances": [],
                              "stop": None, "stop_kind": None, "started_epoch": clock(), "gate": GATE_TEXT,
                              "truncation_invalid": f"completion_tokens >= {data['max_tokens']}"}
    ledger: Path | None = None
    try:
        spec, node_dir, _pin = load_node(data, harness_root)
        limits = build_limits(data, gpu_uuids, cpu_list)
        record["limits"] = limits.to_dict()
        mem_cgroup = memory_limit_cgroup(data["cgroup_parent"])
        record["memory_events_cgroup"] = str(mem_cgroup) if mem_cgroup is not None else None
        guard = CampaignGuard(deadline_epoch=float(data["deadline_epoch"]),
                              run_seconds=3.0 * float(data["measured_run_seconds"]) + RUN_DEADLINE_MARGIN_SECONDS,
                              clock=clock, memory_cgroup=mem_cgroup,
                              memory_events=memory_events or read_memory_events)
        record["deadline_rule"] = (f"a run starts only with 3 x census + {RUN_DEADLINE_MARGIN_SECONDS:g} s left, a "
                                   f"manager call only with {LLM_CALL_ALLOWANCE_SECONDS:g} s more")
        transport = timed_chat(chat or openai_compatible_chat(data["base_url"]))
        for inst in data["instances"]:
            _need(clock() < float(data["deadline_epoch"]),
                  f"STOP: deadline: reached before instance {inst['instance']} ({inst['memory_mode']})", "deadline")
            manager = TruncationInvalidManager(chat=transport, model=data["model"], seed=inst["manager_seed"],
                                               max_tokens=int(data["max_tokens"]),
                                               temperature=float(data["temperature"]),
                                               include_history=MEMORY_MODES[inst["memory_mode"]],
                                               extra_payload={"cache_prompt": False}, guard=guard)
            entry = {"instance": inst["instance"], "memory_mode": inst["memory_mode"], "dir": instance_dir_name(inst)}
            record["instances"].append(entry)
            ledger = out_dir / instance_dir_name(inst) / LEDGER_NAME
            started = clock()
            try:
                campaign(spec, node_dir, out_dir / instance_dir_name(inst), manager, limits, data["budget"],
                         enable_v2_diff=True, train_seed=inst["train_seed"], split_seed=inst["split_seed"],
                         interpreter=data["interpreter"], record_gate=gate,
                         runner_factory=functools.partial(G5CampaignRunner, guard=guard))
            finally:   # a STOPped instance still counts its time
                entry["wall_seconds"] = round(clock() - started, 1)
    except Exception as exc:  # noqa: BLE001 - every stop is recorded, with its structured kind (review m4)
        record["stop"], record["stop_kind"] = _stop_text(exc), _stop_kind(exc, ledger)
    record["finished_epoch"] = clock()
    (out_dir / "campaign_record.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


# =============================================================================================== audit: input
def audit_seeds(k: int) -> tuple[list[int], list[int]]:
    """(label seeds, evaluation seeds) of the K block (P4: K is 10 or 6)."""
    _need(type(k) is int and k in AUDIT_SEED_BLOCKS, "STOP: K is 10 or 6")
    label, evaluation = AUDIT_SEED_BLOCKS[k]
    return list(label), list(evaluation)


def validate_chunk(data: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed validation of a ``trackg_g5_audit_chunk/v1`` file (``g5_job.audit_chunk_file`` writes it)."""
    _need(isinstance(data, dict) and data.get("schema") == CHUNK_SCHEMA, f"STOP: chunk file must be {CHUNK_SCHEMA}")
    _need(set(data) == CHUNK_KEYS,
          f"STOP: chunk file keys differ from {CHUNK_SCHEMA}: {sorted(set(data) ^ CHUNK_KEYS)}")
    _check_common(data)
    _need(type(data["chunk"]) is int and data["chunk"] >= 0, "STOP: chunk must be a non-negative integer")
    label, evaluation = audit_seeds(data["K"])
    allowed = set(label) | set(evaluation)
    _need(data["split_policy"] == "vary",
          "STOP: G.5 audits vary the split seed with the audit seed (split_policy vary)")
    _need(data["record_gate"] == "noise_floor.check_effective", "STOP: the record gate is noise_floor.check_effective")
    units = data["units"]
    _need(isinstance(units, list), "STOP: units must be a list")
    names: list[str] = []
    for u in units:
        _need(isinstance(u, dict) and set(u) == {"unit", "code_state_sha256", "diffs"},
              "STOP: a unit has exactly unit, code_state_sha256, diffs")
        _need(isinstance(u["unit"], str) and bool(u["unit"]), "STOP: a unit id must be a non-empty string")
        _need(isinstance(u["code_state_sha256"], str) and bool(_SHA.fullmatch(u["code_state_sha256"])),
              f"STOP: unit {len(names)}: code_state_sha256 must be a sha256")
        _need(isinstance(u["diffs"], list) and all(isinstance(d, str) and d for d in u["diffs"]),
              f"STOP: unit {len(names)}: diffs must be a list of non-empty diff texts")
        names.append(u["unit"])
    _need(len(set(names)) == len(names), "STOP: a unit appears twice")
    runs = data["runs"]
    _need(isinstance(runs, list), "STOP: runs must be a list")
    seen: set[tuple[str, int]] = set()
    for i, r in enumerate(runs):
        _need(isinstance(r, dict) and set(r) == RUN_KEYS, f"STOP: run {i} has exactly {sorted(RUN_KEYS)}")
        _need(r["unit"] in names, f"STOP: run {i} names a unit that is not in the chunk")
        seed = r["seed"]
        _need(type(seed) is int and seed in allowed, f"STOP 10: run {i}: seed {seed!r} is not an audit seed of "
                                                     f"K = {data['K']}")
        _need(r["train_seed"] == seed and r["split_seed"] == seed,
              f"STOP: run {i}: training seed = split seed = the audit seed")
        _need((r["unit"], seed) not in seen, f"STOP: run {i} repeats a planned (unit, seed)")
        seen.add((r["unit"], seed))
    _need({r["unit"] for r in runs} == set(names), "STOP: every unit of the chunk must have a planned run")
    return data


# =============================================================================================== audit: states
def prepare_states(spec: TrackGNodeSpec, node_dir: Path, data: dict[str, Any], root: Path
                   ) -> dict[str, tuple[Path, str | None]]:
    """Re-materialise every unit before anything runs: the incumbent is the node kit plus the unit's diffs but the
    last (each applied with the runner's diff code, as the campaign applied it), and the last diff must apply to it
    and give the chunk's normalised code-state hash.  Returns, per unit, the directory the runner copies and the
    diff it applies (None for the baseline)."""
    base_spec = spec.to_base_node_spec()
    interpreter = data["interpreter"]
    kit_state = code_state_hashes(node_dir, base_spec)[0]
    built: dict[tuple[str, ...], Path] = {(): node_dir}
    out: dict[str, tuple[Path, str | None]] = {}

    def incumbent(prefix: tuple[str, ...], index: int) -> Path:
        for n in range(1, len(prefix) + 1):
            key = prefix[:n]
            if key in built:
                continue
            target = root / f"state_{len(built):04d}"
            shutil.copytree(built[prefix[:n - 1]], target, ignore=RUNNER_COPY_IGNORE, symlinks=True)
            app = apply_diff(target, prefix[n - 1], base_spec, evaluated=None, allow_new_files=spec.allow_new_files,
                             interpreter=interpreter)
            _need(app.applied, f"STOP: integrity: unit {index}: diff {n} of its chain does not apply "
                               f"({app.status}) when re-materialised", "integrity")
            built[key] = target
        return built[prefix]

    for index, unit in enumerate(data["units"]):
        diffs, expected = unit["diffs"], unit["code_state_sha256"]
        if not diffs:
            _need(kit_state == expected, f"STOP: integrity: unit {index}: the node kit's code state differs from the "
                                         f"chunk's baseline hash", "integrity")
            out[unit["unit"]] = (node_dir, None)
            continue
        inc = incumbent(tuple(diffs[:-1]), index)
        app = apply_diff(inc, diffs[-1], base_spec, evaluated=None, write=False, allow_new_files=spec.allow_new_files,
                         interpreter=interpreter)
        _need(app.applied and app.provenance.get("code_state_sha256") == expected,
              f"STOP: integrity: unit {index}: the re-materialised code state differs from the chunk's hash "
              f"({app.status})", "integrity")
        out[unit["unit"]] = (inc, diffs[-1])
    return out


# =============================================================================================== audit: attribution
def memory_limit_cgroup(cgroup_parent: str) -> Path | None:
    """The cgroup whose memory limit governs the runs: the nearest of ``cgroup_parent`` and its ancestors with a
    ``memory.max`` other than ``max`` and a ``memory.events`` file (the job's ``MemoryMax`` scope); None if none."""
    path = Path(cgroup_parent)
    for cand in (path, *path.parents):
        if cand == Path(cand.anchor):
            break
        try:
            if (cand / "memory.events").is_file() and (cand / "memory.max").read_text().strip() != "max":
                return cand
        except OSError:
            continue
    return None


def read_memory_events(cgroup: Path | None) -> dict[str, int] | None:
    if cgroup is None:
        return None
    try:
        text = (Path(cgroup) / "memory.events").read_text()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if value.strip().isdigit():
            out[key] = int(value)
    return out


def attribute(outcome: RunOutcome, before: dict[str, int] | None, after: dict[str, int] | None, *,
              log_text: str | None = None, prior_crash: bool = False) -> tuple[RunOutcome, dict[str, Any] | None]:
    """Ruling 1: an infrastructure outcome that the code state caused becomes divergent.

    The runner's classifier already makes divergent: a non-finite loss or metric, no metric, the budget not
    reached, a measured timeout, and a non-zero exit (the edited code runs only in the sandboxed worker; its crash, a
    signal death included, makes the frozen parent exit non-zero).  Of its infrastructure outcomes, three cases are
    re-attributed to the code state:

    * ``oom_kill_memory_limit`` -- a ``killed`` run while ``oom`` and ``oom_kill`` both rose in the ``memory.events``
      of the cgroup holding the job's memory limit: the scope's own limit was reached during the run.  The scope
      holds the node wrapper, the G.5 job, this engine and the run; the run is the only one whose memory varies with
      the code state, so its memory reached the limit;
    * ``crash_signal_repeated`` -- a synchronous fault signal (:data:`CRASH_SIGNALS`) of the run's frozen parent on
      this attempt **and** on an earlier attempt of the same code state and seed (``prior_crash``).  A first fault
      stays infrastructure and is rerun, since a parent fault can also be a host fault (review m2); the record's
      attribution says ``crash_signal: true`` so the caller can count it;
    * ``infrastructure_pattern_after_ar_device`` -- a match of a **default** infrastructure log pattern
      (``outcome.DEFAULT_INFRASTRUCTURE_LOG_PATTERNS``) whose every occurrence lies after the run's first
      ``AR_DEVICE`` line, that is in the output of the edited code's phase (review m3).  Node-specific patterns are
      anchored before ``AR_DEVICE`` by design and stay infrastructure; so does a default pattern in a node that never
      prints ``AR_DEVICE`` (``mlgym_houseprice``).

    Everything else keeps its class and is rerun: a kill without evidence that the run caused it (a host-wide OOM
    raises ``oom_kill`` without ``oom``; an outside SIGTERM), a launch error, no exit status.  ``ok`` and
    ``divergent`` pass unchanged."""
    if outcome.kind != INFRASTRUCTURE_FAILURE:
        return outcome, None
    evidence: dict[str, Any] = {"original_kind": outcome.kind, "original_reason": outcome.reason}
    if before is None or after is None:
        evidence["memory_events"] = "unavailable"
        oom = oom_kill = 0
    else:
        oom = after.get("oom", 0) - before.get("oom", 0)
        oom_kill = after.get("oom_kill", 0) - before.get("oom_kill", 0)
        evidence["memory_events"] = {"oom": oom, "oom_kill": oom_kill}
    if outcome.reason == "infrastructure_log_pattern":
        pattern = outcome.detail.get("pattern")
        device = AR_DEVICE_LINE.search(log_text or "")
        hits = [m.start() for m in re.finditer(pattern, log_text or "", re.MULTILINE)] if pattern else []
        evidence["pattern"] = pattern
        if pattern in DEFAULT_INFRASTRUCTURE_LOG_PATTERNS and device is not None and hits \
                and min(hits) > device.start():
            return RunOutcome.divergent("infrastructure_pattern_after_ar_device"), dict(
                evidence, rule="edit-caused: a default infrastructure pattern only in the edited code's output")
        return outcome, dict(evidence, rule="infrastructure (a log pattern before AR_DEVICE, node-specific, or no "
                                            "AR_DEVICE line)")
    if outcome.reason != "killed":
        return outcome, dict(evidence, rule="infrastructure (not a kill)")
    sig = outcome.detail.get("signal") or KILL_EXIT_SIGNALS.get(outcome.detail.get("returncode"))
    evidence["signal"] = int(sig) if sig is not None else None
    if oom > 0 and oom_kill > 0:
        return RunOutcome.divergent("oom_kill_memory_limit"), dict(evidence, rule="edit-caused: the run reached the "
                                                                                  "job scope's memory limit")
    if sig is not None and int(sig) in CRASH_SIGNALS:
        evidence["crash_signal"] = True
        if prior_crash:
            return RunOutcome.divergent("crash_signal_repeated"), dict(
                evidence, rule="edit-caused: a fault signal on the rerun of the same code state and seed")
        return outcome, dict(evidence, rule="infrastructure on the first fault signal (rerun; divergent if it recurs)")
    return outcome, dict(evidence, rule="infrastructure (a kill without evidence that the run caused it)")


# =============================================================================================== audit: run
def _isolation_block(record: dict[str, Any]) -> dict[str, Any]:
    iso = record.get("isolation") or {}
    return {key: iso.get(key) for key in ISOLATION_KEYS}


def _record_sha(result: RunResult) -> str | None:
    path = result.run_dir / "run_record.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_audit(data: dict[str, Any], harness_root: Path, out_dir: Path, *, gpu_uuids: tuple[str, ...],
              cpu_list: tuple[int, ...], runner_factory: Callable[..., WorkingCopyRunner] = WorkingCopyRunner,
              gate: Callable[[dict[str, Any]], None] = check_effective, clock: Callable[[], float] = time.time,
              memory_events: Callable[[Path | None], dict[str, int] | None] = read_memory_events
              ) -> dict[str, Any]:
    """One audit chunk: re-materialise the units, then run every planned (unit, seed) in order and write one row each.

    STOPs (the chunk ends; rows written so far stay): a unit that does not re-materialise to its hash; a run that
    does not execute the planned code state; an integrity error from the runner; a run record that fails the gate
    (its row is written with its isolation block, without a value, so the job sees STOP 2); a run that cannot finish
    before the chunk deadline (3 x the census time + 60 s left)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows_path = out_dir / "rows.jsonl"
    rows_path.touch(exist_ok=False)
    states_root = out_dir / "states"
    runs = data["runs"]
    record: dict[str, Any] = {"schema": CHUNK_RECORD_SCHEMA, "label": LABEL, "engine_contract": ENGINE_CONTRACT,
                              "node": data["node"], "run_id": data["run_id"], "wave": data["wave"],
                              "chunk": data["chunk"], "K": data["K"], "expected_runs": len(runs), "written_rows": 0,
                              "stop": None, "stop_kind": None, "started_epoch": clock(), "gate": GATE_TEXT,
                              "infrastructure_reruns": INFRASTRUCTURE_RERUNS}
    try:
        spec, node_dir, pin = load_node(data, harness_root)
        limits = build_limits(data, gpu_uuids, cpu_list)
        record["limits"] = limits.to_dict()
        mem_cgroup = memory_limit_cgroup(data["cgroup_parent"])
        record["memory_events_cgroup"] = str(mem_cgroup) if mem_cgroup is not None else None
        states_root.mkdir()
        states = prepare_states(spec, node_dir, data, states_root)
        record["units_rematerialised"] = len(states)
        expected = {u["unit"]: u["code_state_sha256"] for u in data["units"]}
        index_of = {u["unit"]: i for i, u in enumerate(data["units"])}
        runners: dict[Path, WorkingCopyRunner] = {}
        budget = 3.0 * float(data["measured_run_seconds"]) + 60.0
        with rows_path.open("a", encoding="utf-8") as rows:
            for n, run in enumerate(runs):
                unit, seed = run["unit"], run["seed"]
                root, diff = states[unit]
                runner = runners.get(root) or runners.setdefault(
                    root, runner_factory(spec, root, out_dir / "runs", frozen_pin=pin, protect_paths=(node_dir,)))
                attempts: list[dict[str, Any]] = []
                row: dict[str, Any] | None = None
                for attempt in range(1 + INFRASTRUCTURE_RERUNS):
                    _need(clock() + budget <= float(data["deadline_epoch"]),
                          f"STOP: deadline: run {n + 1} of {len(runs)} could not finish before the chunk deadline",
                          "deadline")
                    run_id = f"u{index_of[unit]:03d}_s{seed}_a{attempt}"
                    before = memory_events(mem_cgroup)
                    try:
                        result = runner.run(run_id, limits, train_seed=seed, split_seed=seed, diff_text=diff,
                                            evaluated=None, register_evaluated=False, keep_code_snapshot=False,
                                            interpreter=data["interpreter"])
                    except IntegrityError:
                        raise
                    except RunnerError as exc:      # refused by the host before the run: infrastructure, rerun
                        attempts.append({"run_id": run_id, "executed": False, "kind": INFRASTRUCTURE_FAILURE,
                                         "reason": "runner_error", "error": str(exc)[:300]})
                        continue
                    after = memory_events(mem_cgroup)
                    rec = result.record
                    _need(rec.get("executed") is True, f"STOP: integrity: run {n + 1} was not executed (diff "
                                                       f"{(rec.get('diff') or {}).get('status')})", "integrity")
                    _need(rec.get("code_state_sha256") == expected[unit],
                          f"STOP: integrity: run {n + 1} ran another code state than the planned unit's", "integrity")
                    prior_crash = any((a.get("attribution") or {}).get("crash_signal") for a in attempts)
                    outcome, attribution = attribute(result.outcome, before, after, log_text=_run_log(result),
                                                     prior_crash=prior_crash)
                    shutil.rmtree(result.work_dir, ignore_errors=True)   # the node copy; record and log stay
                    base = {"run_id": result.run_id, "executed": True,
                            "code_state_sha256": rec.get("code_state_sha256"),
                            "record_sha256": _record_sha(result), "frozen_pin_sha256": rec.get("frozen_pin_sha256"),
                            "elapsed_seconds_info_only": rec.get("elapsed_seconds_info_only"),
                            "isolation": _isolation_block(rec), "attribution": attribution}
                    try:
                        gate(rec)
                    except Exception as exc:  # noqa: BLE001 - the gate's refusal STOPs the chunk
                        if outcome.kind in (OK, DIVERGENT):
                            row = _row(data, run, base, {"kind": outcome.kind, "reason": outcome.reason or None},
                                       attempts, gate_refusal=str(exc)[:300])
                            rows.write(json.dumps(row, sort_keys=True) + "\n")
                            rows.flush()
                            record["written_rows"] += 1
                        raise G5EngineStop(f"STOP: stop_unisolated: run {n + 1} failed the record gate",
                                           "unisolated") from exc
                    if outcome.kind == INFRASTRUCTURE_FAILURE:
                        attempts.append({**base, "kind": outcome.kind, "reason": outcome.reason})
                        continue
                    out_kind = {"kind": outcome.kind, "reason": outcome.reason or None}
                    if outcome.kind == OK:
                        _need(_is_number(outcome.value) and math.isfinite(outcome.value),
                              f"STOP: run {n + 1}: an ok outcome without a finite value", "error")
                        out_kind["value"] = float(outcome.value)
                    row = _row(data, run, base, out_kind, attempts)
                    break
                if row is None:                              # every attempt was an infrastructure failure
                    last = dict(attempts[-1])
                    final = {"kind": INFRASTRUCTURE_FAILURE, "reason": last.pop("reason")}
                    last.pop("kind")
                    last.setdefault("code_state_sha256", None)
                    row = _row(data, run, last, final, attempts[:-1])
                rows.write(json.dumps(row, sort_keys=True) + "\n")
                rows.flush()
                record["written_rows"] += 1
    except Exception as exc:  # noqa: BLE001 - every stop is recorded, with its structured kind (review m4)
        record["stop"], record["stop_kind"] = _stop_text(exc), _stop_kind(exc)
    finally:
        shutil.rmtree(states_root, ignore_errors=True)       # re-derivable from the chunk file; hashes checked
    record["finished_epoch"] = clock()
    (out_dir / "chunk_record.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


def _row(data: dict[str, Any], run: dict[str, Any], base: dict[str, Any], outcome: dict[str, Any],
         earlier: list[dict[str, Any]], gate_refusal: str | None = None) -> dict[str, Any]:
    row = {"schema": ROW_SCHEMA, "node": data["node"], "wave": data["wave"], "chunk": data["chunk"],
           "unit": run["unit"], "seed": run["seed"], "train_seed": run["train_seed"], "split_seed": run["split_seed"],
           **{k: v for k, v in base.items() if not (k == "isolation" and not base.get("executed"))},
           "outcome": outcome, "attempts": len(earlier) + 1, "infrastructure_reruns": len(earlier),
           "earlier_attempts": [{k: a.get(k) for k in ("run_id", "executed", "kind", "reason")} for a in earlier]}
    if gate_refusal is not None:
        row["gate_refusal"] = gate_refusal
    return row


# =============================================================================================== CLI
def _cpu_list() -> tuple[int, ...]:
    return tuple(sorted(os.sched_getaffinity(0)))


def _refusal_record(out: Path, name: str, schema: str, message: str) -> None:
    """Best effort: a refused engine call still leaves its record (``stop``) where the job looks for it."""
    try:
        out.mkdir(parents=True, exist_ok=False)
        (out / name).write_text(json.dumps({"schema": schema, "label": LABEL, "engine_contract": ENGINE_CONTRACT,
                                            "stop": message[:600], "stop_kind": "refused",
                                            "refused_before_running": True}, indent=1) + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G.5 T1 label engines: one campaign wave or one audit chunk (EVIDENCE).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, flag in (("campaign", "--campaign"), ("audit", "--chunk")):
        p = sub.add_parser(name)
        p.add_argument(flag, required=True, type=Path, dest="infile")
        p.add_argument("--harness-root", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--gpu-uuid", default=None, help="the full UUID of the leased worker GPU (GPU nodes)")
    a = ap.parse_args(argv)
    campaign = a.cmd == "campaign"
    record_name, record_schema = (("campaign_record.json", CAMPAIGN_RECORD_SCHEMA) if campaign
                                  else ("chunk_record.json", CHUNK_RECORD_SCHEMA))
    try:
        raw = json.loads(a.infile.read_text())
        data = validate_campaign(raw) if campaign else validate_chunk(raw)
        gpus = resolve_gpus(data["node"], a.gpu_uuid, os.environ)
    except Exception as exc:  # noqa: BLE001 - a refused call still leaves its record
        _refusal_record(a.out, record_name, record_schema, str(exc))
        print(str(exc), file=sys.stderr)
        return 2
    if campaign:
        rec = run_campaign(data, a.harness_root, a.out, gpu_uuids=gpus, cpu_list=_cpu_list())
        print(json.dumps({"node": data["node"], "wave": data["wave"], "instances_started": len(rec["instances"]),
                          "stopped": rec["stop"] is not None}))
    else:
        rec = run_audit(data, a.harness_root, a.out, gpu_uuids=gpus, cpu_list=_cpu_list())
        print(json.dumps({"node": data["node"], "wave": data["wave"], "chunk": data["chunk"],
                          "expected_runs": rec["expected_runs"], "written_rows": rec["written_rows"],
                          "stopped": rec["stop"] is not None}))
    return 0 if rec["stop"] is None else 3


if __name__ == "__main__":
    sys.exit(main())
