"""Shared checks for Track G port kits (U3): static checks and a synthetic-data smoke.

Used by tests/trackg/test_port_<suite>.py.  Two tiers:

* static checks (always run; no torch): the node spec loads; the frozen and
  editable files exist; the shared frozen helpers are byte-identical across
  the port's nodes; the frozen wrapper, not the editable code, prints the
  metric lines; the committed representative-edits manifest is complete
  (3-5 edits, every E*.diff listed with its sha256, the spec hash and the
  baseline code-state hash are current, nothing measured) and re-declaring the
  edits reproduces it (byte for byte under the declaring Python 3.12; under
  another minor version without the AST-based normalised code-state hashes); a diff that touches any frozen file is refused
  (``scope_violation``) and a diff that only touches an editable file applies.
* synthetic smoke (skipped without the node's runtime packages): random data
  of the real shapes (no task data), a reduced step budget, and the stub
  manager of ``run_smoke_campaign`` with a frozen-path diff, a deliberately
  broken diff (non-finite loss) and the first declared edit; plus every
  declared edit run once on the baseline, and a second baseline run with the
  same seeds, which must give the same metric (on the test host's CPU; the
  GPU determinism check is G.1b's).  It checks the outcome classes and that
  each run record is complete.  It measures nothing about the task.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

import yaml

from autoresearch.trackg.declared_edits import declare_edits
from autoresearch.trackg.diff_support import apply_diff
from autoresearch.trackg.node_template import TrackGNodeSpec, load_trackg_node_spec
from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
from autoresearch.trackg.smoke import run_smoke_campaign

ROOT = Path(__file__).resolve().parents[2]
SHARED_FROZEN = ("ar_data.py", "ar_fixed_loop.py")
METRIC_PREFIXES = ("AR_METRIC", "AR_TEST")
DATA_SUFFIXES = {".npy", ".npz", ".parquet", ".csv", ".pkl", ".pt", ".pth", ".safetensors", ".bin", ".arrow",
                 ".tar", ".gz", ".zip"}
# The manifests were declared with Python 3.12 (scripts/trackg_declare_edits.py); see check_manifest.
DECLARED_WITH_PYTHON = (3, 12)
RECORD_KEYS = ("schema", "run_id", "node", "node_spec_sha256", "run_budget", "workers", "seeds", "limits",
               "diff", "incumbent_code_state_sha256", "code_state_sha256", "executed", "command",
               "environment_applied", "determinism", "process", "log_sha256", "outcome")


def spec_path(node: str) -> Path:
    return ROOT / "configs" / "trackg" / "nodes" / f"{node}.yaml"


def node_dir(node: str) -> Path:
    return ROOT / "nodes" / node


def load(node: str) -> TrackGNodeSpec:
    return load_trackg_node_spec(spec_path(node))


def edit_files(node: str) -> list[Path]:
    return sorted((node_dir(node) / "representative_edits").glob("E*.diff"))


def frozen_files(node: str) -> list[Path]:
    """Every committed file under the node's frozen paths (the data link is made by setup)."""
    spec, base = load(node), node_dir(node)
    out = []
    for entry in spec.frozen_paths:
        path = base / entry.rstrip("/")
        if path.is_symlink():
            continue  # the data link made by setup: its target is outside the node
        if entry.endswith("/"):
            out += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
        else:
            out.append(path)
    return out


def committed_files(base: Path) -> list[Path]:
    """Files of the node directory, without caches and without anything behind the setup's data link."""
    out = []
    for path in base.rglob("*"):
        rel = path.relative_to(base)
        if rel.parts and rel.parts[0] == "data" and (base / "data").is_symlink():
            continue
        if path.is_file() and "__pycache__" not in rel.parts:
            out.append(path)
    return sorted(out)


def touch_diff(rel: str, text: str) -> str:
    """A one-line diff that changes the first line of *rel* (content *text*)."""
    first = text.splitlines()[0]
    return f"--- a/{rel}\n+++ b/{rel}\n@@ -1,1 +1,1 @@\n-{first}\n+{first} # edited\n"


# --------------------------------------------------------------------------- static checks

def check_static(node: str, port_nodes: list[str]) -> None:
    spec = load(node)
    base = node_dir(node)
    assert spec.name == node
    assert spec.run_budget.kind in ("steps", "tokens") and spec.run_budget.value >= 1
    assert spec.metric.metric_regex.startswith("^AR_METRIC ")
    assert spec.selection.selection_split == "validation"
    if spec.selection.test_scored:
        assert spec.metric.test_metric_regex and spec.metric.test_metric_regex.startswith("^AR_TEST ")
    assert spec.seed_env == ("AR_TRAIN_SEED", "AR_SPLIT_SEED")
    assert spec.allow_new_files is False
    for rel in spec.editable_paths:
        assert (base / rel).is_file(), f"editable {rel} missing"
    for rel in spec.frozen_paths:
        if rel.rstrip("/") == "data":
            continue
        assert (base / rel.rstrip("/")).exists(), f"frozen {rel} missing"
    assert "run_fixed_budget.py" in spec.frozen_paths
    if (base / "prepare_data.py").exists():
        assert "data/" in spec.frozen_paths and "prepare_data.py" in spec.frozen_paths
    assert "representative_edits/" in spec.frozen_paths
    # Checklist F: the licence/provenance record is frozen, and no task data are committed.
    assert "LICENCES.md" in spec.frozen_paths and (base / "LICENCES.md").is_file()
    data_like = [str(p.relative_to(base)) for p in committed_files(base)
                 if p.suffix in DATA_SUFFIXES or p.stat().st_size > 1_000_000]
    assert not data_like, f"files that look like task data are committed: {data_like}"
    data = base / "data"
    assert not data.exists() or data.is_symlink(), "data/ may only be the link made by setup, never committed"
    # No committed file is outside both scopes, so nothing is silently editable or unlisted.
    listed = {p.resolve() for p in frozen_files(node)} | {(base / r).resolve() for r in spec.editable_paths}
    stray = [str(p.relative_to(base)) for p in committed_files(base) if p.resolve() not in listed]
    assert not stray, f"files in neither editable nor frozen paths: {stray}"
    # Shared frozen helpers are identical in every node of the port that has them.
    for name in SHARED_FROZEN:
        if (base / name).exists():
            copies = {n: (node_dir(n) / name).read_bytes() for n in port_nodes if (node_dir(n) / name).exists()}
            assert len(set(copies.values())) == 1, f"{name} differs between nodes {sorted(copies)}"
            assert name in spec.frozen_paths
    # The frozen wrapper prints the metric, through the saved output descriptor (emit), and ends with
    # os._exit after it (finish); editable code never mentions the result lines.
    wrapper = (base / "run_fixed_budget.py").read_text()
    assert re.search(r'emit\(f"AR_METRIC ' + re.escape(spec.metric_name) + "=", wrapper)
    if spec.selection.test_scored:
        assert re.search(r'emit\(f"AR_TEST ' + re.escape(spec.selection.test_metric_name) + "=", wrapper)
    assert not re.search(r'print\(f?"AR_(METRIC|TEST|STEPS|TOKENS) ', wrapper), "result lines must go through emit"
    assert re.search(r'\nif __name__ == "__main__":\n    main\(\)\n    (ar_data\.)?finish\(\)', wrapper), \
        "the wrapper must end the process with finish() right after main()"
    # The output is sealed (descriptors 1 and 2 to /dev/null) right before the first AR_METRIC line, and from there
    # to the end of main() only result lines follow: no editable code runs after it.
    head = wrapper[:wrapper.rindex("\n", 0, wrapper.index('emit(f"AR_METRIC'))].rstrip().splitlines()
    assert head and head[-1].strip().startswith(("ar_data.seal()", "seal()")), "seal() must precede the AR_METRIC line"
    tail = wrapper[wrapper.index('emit(f"AR_METRIC'):wrapper.index('\nif __name__ == "__main__":')]
    for line in tail.splitlines()[1:]:
        text = line.strip()
        assert not text or text.startswith(("ar_data.emit(", "emit(", "ar_data.finish()", "finish()", "#")), \
            f"code after the first AR_METRIC line: {text!r}"
    for rel in spec.editable_paths:
        text = (base / rel).read_text()
        for prefix in METRIC_PREFIXES + ("AR_STEPS", "AR_TOKENS"):
            assert prefix not in text, f"editable {rel} mentions {prefix}"
        assert not re.search(r"cudnn\.benchmark\s*=\s*True", text)
        assert not re.search(r"use_deterministic_algorithms\(\s*False", text)
    for path in frozen_files(node):
        if path.suffix == ".py":
            text = path.read_text()
            assert not re.search(r"cudnn\.benchmark\s*=\s*True", text), path
            compile(text, str(path), "exec")
    for rel in spec.editable_paths:
        if rel.endswith(".py"):
            compile((base / rel).read_text(), rel, "exec")


def check_manifest(node: str) -> dict:
    spec = load(node)
    base = node_dir(node)
    manifest_path = base / "representative_edits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    edits = edit_files(node)
    assert 3 <= len(edits) <= 5
    assert manifest["schema"] == "trackg_representative_edits/v1"
    assert manifest["node"] == node and manifest["measured"] is False
    assert manifest["node_spec_sha256"] == spec.raw_sha256, "spec changed after the edits were declared"
    assert manifest["run_budget"] == {"kind": spec.run_budget.kind, "value": spec.run_budget.value}
    assert [e["file"] for e in manifest["edits"]] == [p.name for p in edits]
    assert [e["id"] for e in manifest["edits"]] == [f"E{i}" for i in range(1, len(edits) + 1)]
    for entry, path in zip(manifest["edits"], edits):
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert set(entry["files_touched"]) <= set(spec.editable_paths)
    again = declare_edits(spec, base, edits)
    if sys.version_info[:2] == DECLARED_WITH_PYTHON:
        assert again == manifest, "re-declaring the edits does not reproduce the committed manifest"
    else:
        # The harness's normalised code-state hash is built from the interpreter's AST and differs
        # between Python minor versions (3.12 vs 3.13 for these nodes); everything else must match.
        assert strip_normalised_state(again) == strip_normalised_state(manifest), (
            "re-declaring the edits does not reproduce the committed manifest (normalised code-state "
            f"hashes not compared: declared with Python {DECLARED_WITH_PYTHON}, running {sys.version_info[:2]})")
    return manifest


def strip_normalised_state(manifest: dict) -> dict:
    out = copy.deepcopy(manifest)
    out.pop("baseline_code_state_sha256")
    for entry in out["edits"]:
        entry.pop("code_state_sha256")
        entry.pop("incumbent_code_state_sha256")
    return out


def composable_pairs(node: str) -> dict[str, str]:
    """For every ordered pair (Ei, Ej) of declared edits: whether Ej still applies after Ei
    (``applied``/``inert``/``refused``/``duplicate``), as the stub manager would see it."""
    import tempfile

    spec = load(node)
    node_spec = spec.to_base_node_spec()
    edits = edit_files(node)
    out = {}
    for first in edits:
        for second in edits:
            if first == second:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "w"
                shutil.copytree(node_dir(node), work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "data"))
                one = apply_diff(work, first.read_text(), node_spec, allow_new_files=False)
                assert one.status == "applied"
                two = apply_diff(work, second.read_text(), node_spec, write=False, allow_new_files=False)
                out[f"{first.stem}+{second.stem}"] = two.status
    return out


def check_composable(node: str) -> dict[str, str]:
    """At least one pair of declared edits composes (for the S1.1 interaction measurement)."""
    pairs = composable_pairs(node)
    assert any(v == "applied" for v in pairs.values()), f"no pair of declared edits composes: {pairs}"
    return pairs


def check_scope(node: str) -> None:
    spec = load(node)
    base = node_dir(node)
    node_spec = spec.to_base_node_spec()
    for path in frozen_files(node):
        rel = str(path.relative_to(base))
        text = path.read_text(encoding="utf-8")
        result = apply_diff(base, touch_diff(rel, text), node_spec, write=False, allow_new_files=False)
        assert result.status == "refused", f"diff to frozen {rel} was {result.status}"
        assert result.provenance["refusal"] == "scope_violation", (rel, result.provenance)
    # A new file is refused too (allow_new_files is false).
    new = "--- /dev/null\n+++ b/helper_extra.py\n@@ -0,0 +1,1 @@\n+X = 1\n"
    assert apply_diff(base, new, node_spec, write=False, allow_new_files=False).status == "refused"
    rel = spec.editable_paths[0]
    ok = apply_diff(base, touch_diff(rel, (base / rel).read_text()), node_spec, write=False,
                    allow_new_files=False)
    # In scope: applied, or inert when the touched first line is a comment (config.yaml).
    assert ok.status in ("applied", "inert") and ok.provenance["scope"]["valid"], ok.provenance


# --------------------------------------------------------------------------- synthetic smoke

def have(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def load_node_module(node: str, name: str):
    spec = importlib.util.spec_from_file_location(f"_port_{node}_{name}", node_dir(node) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_copy(node: str, tmp: Path, make_data: Callable[[Path], dict] | None, data_files: list[str]) -> Path:
    """A copy of the node with synthetic data linked as ``data`` (the repository node is not touched).
    Without *make_data* (a node without task data) the copy has no data link."""
    work = tmp / "node" / node
    # The repository node may already carry the setup's data link (after deployment): never follow it.
    shutil.copytree(node_dir(node), work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "data"))
    if make_data is None:
        return work
    data = tmp / "data" / node
    data.mkdir(parents=True)
    source = make_data(data)
    ar_data = load_node_module(node, "ar_data")
    ar_data.write_manifest(data, node, data_files, {"synthetic": True, **source})
    ar_data.link_data(work, data)
    return work


def reduced_spec(node: str, budget: int) -> TrackGNodeSpec:
    mapping = copy.deepcopy(yaml.safe_load(spec_path(node).read_text()))
    mapping["run_budget"]["value"] = budget
    return TrackGNodeSpec.from_mapping(mapping)


def limits(seconds: float = 600.0, gpu_uuids: tuple[str, ...] = (), cpu_list: tuple[int, ...] | None = None,
           external_limit: str = "test host (no cap)") -> ResourceLimits:
    """Runner limits for the smoke.  On a GPU host pass the card's UUID, the CPU list and what enforces
    the memory cap; without a UUID the runner hides every GPU and the run is a CPU run."""
    cpus = cpu_list or tuple(sorted(os.sched_getaffinity(0)))[:4]
    return ResourceLimits(gpu_uuids=tuple(gpu_uuids), cpu_list=cpus, memory_bytes=None,
                          memory_method="external", external_limit=external_limit,
                          provisional_timeout_seconds=seconds, kill_grace_seconds=2.0)


def broken_diff(node: str, anchor: str, replacement: str, rel: str | None = None) -> str:
    """A diff of an editable file that should make the run divergent (default: the first editable file)."""
    rel = rel or load(node).editable_paths[0]
    assert rel in load(node).editable_paths
    text = (node_dir(node) / rel).read_text()
    assert text.count(anchor) == 1, anchor
    import difflib
    new = text.replace(anchor, replacement)
    return "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))


def run_smoke(node: str, tmp: Path, work: Path, budget: int | None, broken: str, interpreter: str = sys.executable,
              run_limits: ResourceLimits | None = None):
    """Stub-manager smoke on synthetic data; returns (smoke records, per-edit records).  The last
    per-edit record is a second baseline run with the same seeds (repeatability check)."""
    spec = load(node) if budget is None else reduced_spec(node, budget)
    lim = run_limits or limits()
    rel_frozen = "run_fixed_budget.py"
    frozen = touch_diff(rel_frozen, (work / rel_frozen).read_text())
    edits = [p.read_text() for p in edit_files(node)]
    records = run_smoke_campaign(spec, work, tmp / "runs_smoke", [frozen, broken, edits[0]], lim,
                                 train_seed=8001, split_seed=8001)
    per_edit = []
    for index, diff in enumerate(edits, start=1):
        result = WorkingCopyRunner(spec, work, tmp / "runs_edits").run(
            f"E{index}", lim, train_seed=8001, split_seed=8001, diff_text=diff, interpreter=interpreter)
        per_edit.append(result.record)
    repeat = WorkingCopyRunner(spec, work, tmp / "runs_edits").run(
        "B2", lim, train_seed=8001, split_seed=8001, interpreter=interpreter)
    per_edit.append(repeat.record)
    check_baseline_log(spec, (tmp / "runs_smoke" / "s000_baseline" / "run.log").read_text(errors="replace"))
    return records, per_edit


def check_baseline_log(spec: TrackGNodeSpec, log: str) -> None:
    """The node's patterns really see the baseline's lines: loss lines (all finite), the metric
    line, the step counter at the budget, the sealed test line, and no stray carriage returns."""
    losses = re.findall(spec.metric.loss_regex, log, re.MULTILINE)
    assert losses, "loss_regex matched no line of the baseline log"
    assert all(float(v) == float(v) for v in losses)
    assert len(re.findall(spec.metric.metric_regex, log, re.MULTILINE)) == 1
    if spec.metric.steps_regex:
        steps = [int(v) for v in re.findall(spec.metric.steps_regex, log, re.MULTILINE)]
        assert steps and max(steps) >= spec.run_budget.value
    if spec.selection.test_scored:
        assert len(re.findall(spec.metric.test_metric_regex, log, re.MULTILINE)) == 1
    assert "\r" not in log.replace("\r\n", "\n"), "carriage returns in the log can hide lines from ^-anchored patterns"


def check_smoke(records: list[dict], per_edit: list[dict], expect_divergent: set[str] = frozenset(),
                torch_node: bool = True) -> None:
    baseline, frozen, broken, first = records
    assert baseline["decision"] == "baseline" and baseline["record"]["outcome"]["kind"] == "ok", baseline
    assert frozen["decision"] == "not_run_refused", frozen
    assert frozen["record"]["diff"]["refusal"] == "scope_violation"
    assert broken["decision"] == "discarded_divergent", broken
    assert broken["record"]["outcome"]["reason"] in ("non_finite_loss", "non_finite_metric"), broken
    assert first["record"]["executed"] is True
    for entry in (baseline, broken, first):
        record = entry["record"]
        missing = [k for k in RECORD_KEYS if k not in record]
        assert not missing, f"run record lacks {missing}"
        if torch_node and record.get("executed", True):
            assert record["determinism"]["not_forced"] == ["dataloader_workers"], record["determinism"]
        assert record["process"]["killed_with"] is None
        assert "sealed" not in record  # the manager view never shows the test metric
    *per_edit, repeat = per_edit
    if torch_node:
        assert repeat["determinism"]["not_forced"] == ["dataloader_workers"], repeat["determinism"]
    assert repeat["outcome"]["kind"] == "ok" and repeat["outcome"]["value"] == baseline["record"]["outcome"]["value"], (
        "a second baseline run with the same seeds gave a different metric", repeat["outcome"], baseline["record"]["outcome"])
    for index, record in enumerate(per_edit, start=1):
        assert record["diff"]["status"] == "applied", record["diff"]
        kind = record["outcome"]["kind"]
        if f"E{index}" in expect_divergent:
            assert kind in ("ok", "divergent"), record["outcome"]
        else:
            assert kind == "ok", (f"E{index}", record["outcome"])
        assert record["determinism"]["shim_loaded"] is True
        if torch_node:
            assert record["determinism"]["not_forced"] == ["dataloader_workers"], record["determinism"]
