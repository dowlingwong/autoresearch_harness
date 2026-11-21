"""G.1b noise-floor chunk engine for Track G T1 nodes (R1-T1-HOST-H200, R1-TRACKG-G3-EARLY, D21).

One *chunk* is a list of runs of one ported node: the baseline ``c0`` and the node's declared representative
edits ``c1 … cN`` (``representative_edits/manifest.json``), each at a seed pair taken from the D21 protocol
(``default`` / ``default_repeat`` at the node's own default seeds, and ``train`` / ``split`` / ``joint`` at
30 seeds each).  The paper repository's G.1b job (``sweep2/g1b/``) writes the chunk file; this module only
runs it, through :class:`~autoresearch.trackg.runner.WorkingCopyRunner`, and writes one E1-schema row per run.

**The unisolated-run gate.**  Every executed run's record must say ``isolation.effective: true`` with
``waivers == {}`` and ``exceptions == []``; the limits this module builds never carry a waiver.  A record that
fails the gate STOPs the chunk at once: the row is not written, the record stays on disk as evidence, and the
chunk record names the run.  The gate is not configurable from the command line.

Pins (kit review of G.1b, B1).  The chunk's ``committed_pin_sha256`` is the census's committed-file pin: the
node's frozen files without the ``data`` link (``frozen_file_hashes`` records that link by its target path, which
differs between hosts, so it is left out here).  The data bytes are bound by ``data_manifest_sha256``: the sha256
of the node's ``data/data_manifest.json`` must equal the census value, and every run's frozen parent checks the
data files against that manifest.  A node without data (Breakout) has ``data_manifest_sha256: null`` and must not
have a ``data`` link.

Other STOPs: a node whose spec, frozen pin or declared-edit manifest differs from the chunk's pins; an edit file
whose sha256 differs from the manifest; a run plan that is not the canonical expansion of the chunk's
``(conditions, seed range)``; an integrity error from the runner; an infrastructure failure (it is a property of
the host, not of the node, so the chunk stops rather than leave a gap); a run that cannot finish before the
chunk deadline.  A divergent run (for example a timeout, or a non-finite loss under an edit) is a row with
``success: false``, as in E1.

Nothing here reads labels or sealed test metrics: rows carry the validation metric only; the sealed test value
stays in the run record and is not copied.  ENGINEERING_ONLY; AI-assisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autoresearch.trackg import gpu_identity, isolation
from autoresearch.trackg.node_template import TrackGNodeSpec, load_trackg_node_spec
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK
from autoresearch.trackg.runner import IntegrityError, ResourceLimits, RunnerError, RunResult, WorkingCopyRunner

CHUNK_SCHEMA = "trackg_noise_chunk/v1"
ROW_SCHEMA = "r1.e1_noise_floor/2"
CHUNK_RECORD_SCHEMA = "trackg_noise_chunk_record/v1"
CONDITIONS = ("default", "default_repeat", "train", "split", "joint")
SEED_CONDITIONS = ("train", "split", "joint")
SEEDS_PER_CONDITION = 30
SEED_BASES = {"train": 1001, "split": 2001, "joint": 3001}
CPU_ONLY_NODES = frozenset({"mlgym_houseprice"})  # the only port node that runs without a GPU (U4 census)
RESERVED_SEEDS = (range(7001, 7011), range(9001, 10000))  # B.2 audit seeds and exam seeds: never used here
_SHA = re.compile(r"^[0-9a-f]{64}$")
_NODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class NoiseFloorStop(RuntimeError):
    """A pre-declared STOP: the chunk ends, its evidence stays on disk."""


class UnisolatedRunError(NoiseFloorStop):
    """A run record that is not effective:true with empty waivers and exceptions."""


# ----------------------------------------------------------------------------------------------- the gate
def check_effective(record: dict[str, Any]) -> None:
    """The unisolated-run gate (R1-HARNESS-ISOLATION; G.1b): refuse any record that is not effectively isolated."""
    iso = record.get("isolation")
    if not isinstance(iso, dict):
        raise UnisolatedRunError(f"run {record.get('run_id')}: the record has no isolation block (unisolated run)")
    problems = []
    if iso.get("effective") is not True:
        problems.append(f"effective={iso.get('effective')!r}")
    if iso.get("waivers") != {}:
        problems.append(f"waivers={iso.get('waivers')!r}")
    if iso.get("exceptions") != []:
        problems.append(f"exceptions={iso.get('exceptions')!r}")
    if iso.get("log_intact") is not True:
        problems.append(f"log_intact={iso.get('log_intact')!r}")
    if iso.get("parent_traced_by") or iso.get("parent_dumpable_at_accept"):
        problems.append("parent traced or dumpable at the accept")
    if problems:
        raise UnisolatedRunError(f"run {record.get('run_id')}: not effectively isolated ({', '.join(problems)})")


# ----------------------------------------------------------------------------------------------- plan
def expected_runs(n_edits: int, conditions: list[str], seed_range: tuple[int, int] | None) -> list[dict[str, Any]]:
    """The canonical run list of a chunk: every configuration, in order, for each condition (and seed)."""
    configs = ["c0"] + [f"c{k}" for k in range(1, n_edits + 1)]
    runs: list[dict[str, Any]] = []
    for condition in conditions:
        if condition in ("default", "default_repeat"):
            runs += [{"config": c, "condition": condition, "train_seed": None, "split_seed": None} for c in configs]
            continue
        lo, hi = seed_range  # type: ignore[misc]
        for seed in range(lo, hi + 1):
            train = seed if condition in ("train", "joint") else None
            split = seed if condition in ("split", "joint") else None
            runs += [{"config": c, "condition": condition, "train_seed": train, "split_seed": split} for c in configs]
    return runs


def _seed_range_ok(condition: str, seed_range: tuple[int, int]) -> bool:
    base = SEED_BASES[condition]
    lo, hi = seed_range
    return base <= lo <= hi <= base + SEEDS_PER_CONDITION - 1


@dataclass(frozen=True)
class Chunk:
    data: dict[str, Any]

    @property
    def node(self) -> str:
        return self.data["node"]


def validate_chunk(data: dict[str, Any]) -> Chunk:
    """Fail-closed validation of a chunk file (the paper-side job writes it; this side re-checks it)."""
    if data.get("schema") != CHUNK_SCHEMA:
        raise NoiseFloorStop(f"STOP: chunk schema must be {CHUNK_SCHEMA}")
    if data.get("label") != "ENGINEERING_ONLY":
        raise NoiseFloorStop("STOP: a G.1b chunk is ENGINEERING_ONLY")
    node = data.get("node")
    if not isinstance(node, str) or not _NODE.fullmatch(node):
        raise NoiseFloorStop(f"STOP: invalid node id {node!r}")
    for key in ("node_spec_sha256", "committed_pin_sha256", "edits_manifest_sha256"):
        if not isinstance(data.get(key), str) or not _SHA.fullmatch(data[key]):
            raise NoiseFloorStop(f"STOP: chunk.{key} must be a sha256")
    dm = data.get("data_manifest_sha256", "missing")
    if dm is not None and (not isinstance(dm, str) or not _SHA.fullmatch(dm)):
        raise NoiseFloorStop("STOP: chunk.data_manifest_sha256 must be a sha256, or null for a node without data")
    n_edits = data.get("n_edits")
    if type(n_edits) is not int or not 3 <= n_edits <= 5:
        raise NoiseFloorStop("STOP: chunk.n_edits must be the node's declared edit count (3-5)")
    conditions = data.get("conditions")
    if not isinstance(conditions, list) or not conditions or any(c not in CONDITIONS for c in conditions) \
            or len(set(conditions)) != len(conditions):
        raise NoiseFloorStop(f"STOP: chunk.conditions must be distinct members of {CONDITIONS}")
    seeded = [c for c in conditions if c in SEED_CONDITIONS]
    if seeded and len(conditions) != 1:
        raise NoiseFloorStop("STOP: a seeded chunk holds exactly one condition (train, split or joint)")
    seed_range = data.get("seed_range")
    if seeded:
        if not (isinstance(seed_range, list) and len(seed_range) == 2 and all(type(s) is int for s in seed_range)
                and _seed_range_ok(seeded[0], tuple(seed_range))):
            raise NoiseFloorStop(f"STOP: seed_range {seed_range!r} is not inside the {seeded[0]} block "
                                 f"{SEED_BASES[seeded[0]]}…{SEED_BASES[seeded[0]] + SEEDS_PER_CONDITION - 1}")
        for reserved in RESERVED_SEEDS:
            if any(s in reserved for s in range(seed_range[0], seed_range[1] + 1)):
                raise NoiseFloorStop("STOP: the chunk touches reserved audit or exam seeds")
    elif seed_range is not None:
        raise NoiseFloorStop("STOP: a default chunk has no seed_range")
    runs = data.get("runs")
    want = expected_runs(n_edits, conditions, tuple(seed_range) if seeded else None)
    if runs != want:
        raise NoiseFloorStop("STOP: chunk.runs is not the canonical expansion of (n_edits, conditions, seed_range)")
    seconds = data.get("measured_run_seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not 0 < seconds <= 300:
        raise NoiseFloorStop("STOP: measured_run_seconds (census per-run time) must be in (0, 300]")
    if data.get("timeout_factor", 3.0) != 3.0:
        raise NoiseFloorStop("STOP: R1-TRACKG-PORT-DECISIONS (4): the node timeout is 3 x the census run time")
    deadline = data.get("deadline_epoch")
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        raise NoiseFloorStop("STOP: chunk.deadline_epoch must be a number (job start + walltime - reserve)")
    for key in ("cgroup_parent", "interpreter", "memory_limit"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise NoiseFloorStop(f"STOP: chunk.{key} must be a non-empty string")
    if data.get("device", "gpu") not in ("gpu", "cpu"):
        raise NoiseFloorStop("STOP: chunk.device must be gpu or cpu (a CPU-only node such as mlgym_houseprice)")
    if data.get("device", "gpu") == "cpu" and node not in CPU_ONLY_NODES:
        raise NoiseFloorStop(f"STOP: {node} is not a CPU-only node; only {sorted(CPU_ONLY_NODES)} run without a GPU")
    if not isinstance(data.get("stop_on_timeout", False), bool):
        raise NoiseFloorStop("STOP: chunk.stop_on_timeout must be true or false")
    if data.get("isolation_waivers", {}) != {}:
        raise NoiseFloorStop("STOP: a G.1b chunk carries no isolation waiver")
    return Chunk(data)


# ----------------------------------------------------------------------------------------------- node checks
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_pin_sha256(pin: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()


def committed_view(pin: dict[str, str]) -> dict[str, str]:
    """The pin without the host-specific ``data`` link entry (the committed files only)."""
    return {k: v for k, v in pin.items() if not (k == "data" and v.startswith("link:"))}


def check_node_pins(spec: TrackGNodeSpec, node_dir: Path, committed_pin: str, data_manifest: str | None) -> dict:
    """The committed-file pin (without the host-specific data link) and the data manifest against the census;
    returns the full pin (with the link) for the runner."""
    pin = isolation.frozen_file_hashes(node_dir, spec.frozen_paths)
    committed = frozen_pin_sha256(committed_view(pin))
    if committed != committed_pin:
        raise NoiseFloorStop(f"STOP: the committed-file pin {committed} differs from the census pin {committed_pin}")
    link = node_dir / "data"
    if data_manifest is None:
        if link.exists() or link.is_symlink():
            raise NoiseFloorStop("STOP: the census declares no data for this node, but it has a data entry")
        return pin
    manifest_file = link / "data_manifest.json"
    if not manifest_file.is_file():
        raise NoiseFloorStop(f"STOP: {manifest_file} missing: run the node's setup_command first")
    got = sha256_file(manifest_file)
    if got != data_manifest:
        raise NoiseFloorStop(f"STOP: data manifest sha256 {got} differs from the census {data_manifest} "
                             "(the data bytes differ from the census data)")
    return pin


def load_node(chunk: Chunk, harness_root: Path) -> tuple[TrackGNodeSpec, Path, dict[str, str], list[str]]:
    """Load the node from the harness checkout and check it against the chunk's pins; return the edit texts."""
    data = chunk.data
    spec_path = harness_root / data["spec_path"]
    node_dir = (harness_root / data["node_dir"]).resolve()
    if harness_root.resolve() not in node_dir.parents or harness_root.resolve() not in spec_path.resolve().parents:
        raise NoiseFloorStop("STOP: spec_path and node_dir must lie inside the harness checkout")
    spec = load_trackg_node_spec(spec_path)
    if spec.name != chunk.node:
        raise NoiseFloorStop(f"STOP: {spec_path} is node {spec.name}, the chunk names {chunk.node}")
    if spec.raw_sha256 != data["node_spec_sha256"]:
        raise NoiseFloorStop(f"STOP: node spec sha256 {spec.raw_sha256} differs from the pin {data['node_spec_sha256']}")
    if spec.isolation is None:
        raise NoiseFloorStop(f"STOP: {chunk.node} has no isolation block: unisolated nodes are refused")
    pin = check_node_pins(spec, node_dir, data["committed_pin_sha256"], data["data_manifest_sha256"])
    manifest_path = node_dir / "representative_edits" / "manifest.json"
    if sha256_file(manifest_path) != data["edits_manifest_sha256"]:
        raise NoiseFloorStop("STOP: representative_edits/manifest.json differs from the pin")
    manifest = json.loads(manifest_path.read_text())
    edits = manifest.get("edits") or []
    if manifest.get("node") != chunk.node or manifest.get("node_spec_sha256") != spec.raw_sha256 \
            or len(edits) != data["n_edits"]:
        raise NoiseFloorStop("STOP: the declared-edit manifest does not match the node or the chunk's edit count")
    texts = []
    for index, entry in enumerate(edits, start=1):
        if entry.get("id") != f"E{index}":
            raise NoiseFloorStop(f"STOP: declared edits must be E1…E{len(edits)} in order")
        path = node_dir / "representative_edits" / entry["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise NoiseFloorStop(f"STOP: {entry['file']} sha256 differs from the manifest")
        texts.append(raw.decode("utf-8"))
    return spec, node_dir, pin, texts


def build_limits(chunk: Chunk, gpu_uuids: tuple[str, ...], cpu_list: tuple[int, ...]) -> ResourceLimits:
    """The run limits: census time x 3, the job's delegated cgroup, no waiver of any kind."""
    data = chunk.data
    return ResourceLimits(gpu_uuids=gpu_uuids, cpu_list=cpu_list, memory_bytes=None, memory_method="external",
                          external_limit=data["memory_limit"], measured_run_seconds=float(data["measured_run_seconds"]),
                          timeout_factor=3.0, cgroup_parent=data["cgroup_parent"], isolation_waivers={})


def run_id_of(run: dict[str, Any]) -> str:
    t = "d" if run["train_seed"] is None else run["train_seed"]
    s = "d" if run["split_seed"] is None else run["split_seed"]
    return f"{run['config']}_{run['condition']}_t{t}_s{s}"


def row_of(chunk: Chunk, run: dict[str, Any], result: RunResult, record_sha: str) -> dict[str, Any]:
    record = result.record
    outcome = record.get("outcome") or {}
    kind = outcome.get("kind")
    iso = record.get("isolation") or {}
    return {
        "schema": ROW_SCHEMA,
        "node": chunk.node,
        "config": run["config"],
        "condition": run["condition"],
        "train_seed": run["train_seed"],
        "split_seed": run["split_seed"],
        "metric": outcome.get("value") if kind == OK else None,
        "success": kind == OK,
        "failure": None if kind == OK else f"{kind}:{outcome.get('reason')}",
        "edit_applied": (record.get("diff") or {}).get("status") if run["config"] != "c0" else None,
        "wall_seconds": record.get("elapsed_seconds_info_only"),
        "run_id": result.run_id,
        "record_sha256": record_sha,
        "code_state_sha256": record.get("code_state_sha256"),
        "frozen_pin_sha256": record.get("frozen_pin_sha256"),
        "isolation": {key: iso.get(key) for key in ("effective", "waivers", "exceptions", "log_intact",
                                                     "parent_traced_by", "parent_dumpable_at_accept")},
    }


Runner = Callable[..., RunResult]


def run_chunk(chunk: Chunk, harness_root: Path, out_dir: Path, *, gpu_uuids: tuple[str, ...],
              cpu_list: tuple[int, ...], runner_factory: Callable[..., WorkingCopyRunner] = WorkingCopyRunner,
              gate: Callable[[dict[str, Any]], None] = check_effective, clock: Callable[[], float] = time.time
              ) -> dict[str, Any]:
    """Run every run of the chunk in order; write ``runs.jsonl``, ``edits.json``, the run records and
    ``chunk_record.json``.  Returns the chunk record (``stop`` is set when the chunk STOPped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    runs_root = out_dir / "runs"
    record: dict[str, Any] = {"schema": CHUNK_RECORD_SCHEMA, "label": "ENGINEERING_ONLY", "node": chunk.node,
                              "chunk_id": chunk.data.get("chunk_id"), "expected_runs": len(chunk.data["runs"]),
                              "written_rows": 0, "successful_rows": 0, "stop": None,
                              "started_epoch": clock(), "gate": "effective:true, waivers {}, exceptions []"}
    rows_path = out_dir / "runs.jsonl"
    try:
        spec, node_dir, pin, edit_texts = load_node(chunk, harness_root)
        limits = build_limits(chunk, gpu_uuids, cpu_list)
        record["limits"] = limits.to_dict()
        manifest = json.loads((node_dir / "representative_edits" / "manifest.json").read_text())
        (out_dir / "edits.json").write_text(json.dumps({chunk.node: [
            {"edit_id": e["id"], "symbol": e["id"], "old": "baseline", "new": e["file"], "patch_sha256": e["sha256"],
             "code_state_sha256": e["code_state_sha256"]} for e in manifest["edits"]]}, indent=1) + "\n")
        runner = runner_factory(spec, node_dir, runs_root, frozen_pin=pin)
        budget = 3.0 * float(chunk.data["measured_run_seconds"]) + 60.0
        with rows_path.open("x") as rows:
            for run in chunk.data["runs"]:
                if clock() + budget > float(chunk.data["deadline_epoch"]):
                    raise NoiseFloorStop(f"STOP: deadline: {run_id_of(run)} could not finish before the chunk deadline")
                diff = None if run["config"] == "c0" else edit_texts[int(run["config"][1:]) - 1]
                result = runner.run(run_id_of(run), limits, train_seed=run["train_seed"],
                                    split_seed=run["split_seed"], diff_text=diff,
                                    interpreter=chunk.data["interpreter"], keep_code_snapshot=False)
                if not result.record.get("executed"):
                    raise NoiseFloorStop(f"STOP: {result.run_id} was not executed (diff "
                                         f"{(result.record.get('diff') or {}).get('status')}): a declared edit must apply")
                gate(result.record)
                kind = (result.record.get("outcome") or {}).get("kind")
                if kind == INFRASTRUCTURE_FAILURE:
                    raise NoiseFloorStop(f"STOP: {result.run_id} infrastructure failure "
                                         f"({result.record['outcome'].get('reason')}): host problem, chunk stops")
                if kind not in (OK, DIVERGENT):
                    raise NoiseFloorStop(f"STOP: {result.run_id} has outcome kind {kind!r}")
                record_path = result.run_dir / "run_record.json"
                row = row_of(chunk, run, result, sha256_file(record_path))
                rows.write(json.dumps(row, sort_keys=True) + "\n")
                rows.flush()
                record["written_rows"] += 1
                record["successful_rows"] += int(row["success"])
                shutil.rmtree(result.work_dir, ignore_errors=True)  # the node copy; the record and log stay
                if chunk.data.get("stop_on_timeout") and str(row.get("failure") or "").startswith("divergent:timeout"):
                    # direct host runs on another GPU model (R1-T1-HOST-TASL): the census timing is an estimate, so
                    # the first run that reaches the declared timeout stops the chunk for the author
                    record["stop_kind"] = "timeout"
                    raise NoiseFloorStop(f"STOP: {result.run_id} reached the declared timeout "
                                         f"(3 x {chunk.data['measured_run_seconds']} s) and stop_on_timeout is set")
    except (NoiseFloorStop, IntegrityError, RunnerError, gpu_identity.GpuIdentityError) as exc:
        record["stop"] = f"{type(exc).__name__}: {exc}"
    record["finished_epoch"] = clock()
    (out_dir / "chunk_record.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


# ----------------------------------------------------------------------------------------------- CLI
def _cpu_list() -> tuple[int, ...]:
    return tuple(sorted(os.sched_getaffinity(0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one G.1b noise-floor chunk (ENGINEERING_ONLY).")
    ap.add_argument("--chunk", required=True, type=Path)
    ap.add_argument("--harness-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpu-uuid", default=None,
                    help="direct host run (R1-T1-HOST-TASL): the full UUID of the leased GPU; without it the GPU is "
                         "resolved from the HTCondor assignment")
    a = ap.parse_args(argv)
    try:
        chunk = validate_chunk(json.loads(a.chunk.read_text()))
        if chunk.data.get("device", "gpu") == "cpu":
            if a.gpu_uuid is not None or any(os.environ.get(n) for n in gpu_identity.SOURCES):
                raise NoiseFloorStop("STOP: a CPU-only chunk runs without any GPU")
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
                raise NoiseFloorStop("STOP: a CPU-only chunk needs CUDA_VISIBLE_DEVICES empty")
            gpus = ()
        else:
            if a.gpu_uuid is not None:
                gpus, note = gpu_identity.resolve_explicit_on_host(a.gpu_uuid, os.environ)
            else:
                gpus, note = gpu_identity.resolve_from_job(os.environ)
            if len(gpus) != 1:
                raise NoiseFloorStop(f"STOP: one GPU per G.1b job, got {gpus}")
    except (NoiseFloorStop, gpu_identity.GpuIdentityError, OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    rec = run_chunk(chunk, a.harness_root, a.out, gpu_uuids=gpus, cpu_list=_cpu_list())
    print(json.dumps({k: rec.get(k) for k in ("node", "chunk_id", "expected_runs", "written_rows", "stop")}))
    return 0 if rec.get("stop") is None and rec["written_rows"] == rec["expected_runs"] else 3


if __name__ == "__main__":
    sys.exit(main())
