"""G.1b noise-floor chunk engine: plan validation, node pins, the unisolated-run gate (CPU only, fake runner).

A real isolated run needs a non-root uid, Yama ptrace_scope >= 1 and a delegated cgroup (no waiver is allowed
in G.1b), which this test container does not have; the end-to-end check is the cluster's ENGINEERING job.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from autoresearch.trackg import noise_floor as nf
from autoresearch.trackg.runner import RunResult

HARNESS = Path(__file__).resolve().parents[2]
NODE = "mlab_cifar10"
SPEC = "configs/trackg/nodes/mlab_cifar10.yaml"
NODE_DIR = "nodes/mlab_cifar10"


def _pins():
    from autoresearch.trackg import isolation
    from autoresearch.trackg.node_template import load_trackg_node_spec
    spec = load_trackg_node_spec(HARNESS / SPEC)
    pin = isolation.frozen_file_hashes(HARNESS / NODE_DIR, spec.frozen_paths)
    manifest = HARNESS / NODE_DIR / "representative_edits" / "manifest.json"
    return spec.raw_sha256, nf.frozen_pin_sha256(pin), nf.sha256_file(manifest), len(json.loads(manifest.read_text())["edits"])


def chunk(conditions=("train",), seed_range=(1001, 1003), **over) -> dict:
    spec_sha, pin_sha, man_sha, n = _pins()
    seeded = conditions[0] in nf.SEED_CONDITIONS
    data = {
        "schema": nf.CHUNK_SCHEMA, "label": "ENGINEERING_ONLY", "chunk_id": "mlab_cifar10-train-1001-1003",
        "node": NODE, "spec_path": SPEC, "node_dir": NODE_DIR,
        "node_spec_sha256": spec_sha, "committed_pin_sha256": pin_sha, "data_manifest_sha256": None,
        "edits_manifest_sha256": man_sha, "n_edits": n,
        "conditions": list(conditions), "seed_range": list(seed_range) if seeded else None,
        "runs": nf.expected_runs(n, list(conditions), tuple(seed_range) if seeded else None),
        "measured_run_seconds": 12.297, "timeout_factor": 3.0, "deadline_epoch": 10 ** 12,
        "cgroup_parent": "/sys/fs/cgroup/job", "interpreter": "/env/bin/python", "memory_limit": "HTCondor request_memory",
        "isolation_waivers": {},
    }
    data.update(over)
    return data


def effective_record(run_id="r", **iso_over) -> dict:
    iso = {"effective": True, "waivers": {}, "exceptions": [], "log_intact": True, "parent_traced_by": None,
           "parent_dumpable_at_accept": False}
    iso.update(iso_over)
    return {"run_id": run_id, "executed": True, "isolation": iso, "outcome": {"kind": "ok", "value": 0.5, "reason": ""},
            "diff": None, "elapsed_seconds_info_only": 1.0, "code_state_sha256": "a" * 64, "frozen_pin_sha256": "b" * 64}


# ---------------------------------------------------------------------------------------------- plan
def test_canonical_plan_counts():
    n = _pins()[3]
    assert n == 4  # the kits declare four edits per node (G1B author decision on the edit count)
    per_node = (len(nf.expected_runs(n, ["default", "default_repeat"], None))
                + sum(len(nf.expected_runs(n, [c], (nf.SEED_BASES[c], nf.SEED_BASES[c] + 29))) for c in nf.SEED_CONDITIONS))
    assert per_node == (n + 1) * (30 * 3 + 2) == 460
    assert nf.expected_runs(5, ["default", "default_repeat"], None).__len__() + 3 * 6 * 30 == 552


def test_valid_chunk_and_default_chunk():
    nf.validate_chunk(chunk())
    nf.validate_chunk(chunk(conditions=("default", "default_repeat"), seed_range=None))
    runs = nf.expected_runs(4, ["joint"], (3001, 3001))
    assert runs[0] == {"config": "c0", "condition": "joint", "train_seed": 3001, "split_seed": 3001}
    assert nf.expected_runs(4, ["split"], (2005, 2005))[1] == {"config": "c1", "condition": "split",
                                                                "train_seed": None, "split_seed": 2005}


@pytest.mark.parametrize("over, needle", [
    ({"isolation_waivers": {"root": "x"}}, "no isolation waiver"),
    ({"timeout_factor": 2.0}, "3 x the census"),
    ({"measured_run_seconds": 0}, "measured_run_seconds"),
    ({"label": "EVIDENCE"}, "ENGINEERING_ONLY"),
    ({"n_edits": 6}, "n_edits"),
    ({"seed_range": [1001, 1031]}, "not inside"),
    ({"seed_range": [2001, 2003]}, "not inside"),
    ({"committed_pin_sha256": "0" * 63}, "committed_pin_sha256"),
    ({"data_manifest_sha256": "x"}, "data_manifest_sha256"),
])
def test_chunk_refusals(over, needle):
    data = chunk()
    data.update(over)
    with pytest.raises(nf.NoiseFloorStop, match=needle):
        nf.validate_chunk(data)


def test_runs_must_be_the_canonical_expansion():
    data = chunk()
    data["runs"] = data["runs"][:-1]
    with pytest.raises(nf.NoiseFloorStop, match="canonical expansion"):
        nf.validate_chunk(data)
    data = chunk()
    data["runs"][0]["train_seed"] = 7001
    with pytest.raises(nf.NoiseFloorStop, match="canonical expansion"):
        nf.validate_chunk(data)
    with pytest.raises(nf.NoiseFloorStop, match="exactly one condition"):
        nf.validate_chunk(chunk(conditions=("train", "split")))


# ---------------------------------------------------------------------------------------------- gate
def test_gate_accepts_an_effective_record():
    nf.check_effective(effective_record())


@pytest.mark.parametrize("over", [
    {"effective": False}, {"waivers": {"root": "test"}}, {"exceptions": ["shared_dev_shm"]}, {"log_intact": False},
    {"parent_traced_by": 123}, {"parent_dumpable_at_accept": True}, {"effective": None}, {"waivers": None},
])
def test_gate_refuses(over):
    with pytest.raises(nf.UnisolatedRunError):
        nf.check_effective(effective_record(**over))


def test_gate_refuses_a_record_without_isolation():
    rec = effective_record()
    del rec["isolation"]
    with pytest.raises(nf.UnisolatedRunError, match="unisolated"):
        nf.check_effective(rec)


# ---------------------------------------------------------------------------------------------- node pins
def test_load_node_checks_the_pins():
    spec, node_dir, pin, texts = nf.load_node(nf.validate_chunk(chunk()), HARNESS)
    assert spec.name == NODE and len(texts) == 4 and all(t.startswith(("---", "diff")) or "@@" in t for t in texts)
    for key in ("node_spec_sha256", "committed_pin_sha256", "edits_manifest_sha256"):
        data = chunk(**{key: "f" * 64})
        with pytest.raises(nf.NoiseFloorStop, match="differs"):
            nf.load_node(nf.validate_chunk(data), HARNESS)


def _copy_harness(tmp: Path) -> Path:
    """A minimal harness root holding the node and its spec (so a test can add a data link or change files)."""
    import shutil
    root = tmp / "h"
    (root / "configs/trackg/nodes").mkdir(parents=True)
    shutil.copy2(HARNESS / SPEC, root / SPEC)
    shutil.copytree(HARNESS / NODE_DIR, root / NODE_DIR, symlinks=True,
                    ignore=shutil.ignore_patterns("data", "__pycache__"))
    return root


def test_data_link_is_left_out_of_the_pin_and_the_data_manifest_is_checked(tmp_path):
    """Kit review of G.1b, B1: the data link's host path must not change the pin; the data bytes are bound by the
    census data manifest sha256."""
    root = _copy_harness(tmp_path)
    data = tmp_path / "somewhere_on_h200" / "mlab_cifar10"
    data.mkdir(parents=True)
    (data / "data_manifest.json").write_text('{"files": {}}\n')
    (root / NODE_DIR / "data").symlink_to(data)
    good = nf.sha256_file(data / "data_manifest.json")
    spec, node_dir, pin, _ = nf.load_node(nf.validate_chunk(chunk(data_manifest_sha256=good)), root)
    assert pin["data"] == "link:" + str(data)  # the runner still pins the full tree, link included
    with pytest.raises(nf.NoiseFloorStop, match="differ from the census data"):
        nf.load_node(nf.validate_chunk(chunk(data_manifest_sha256="e" * 64)), root)
    with pytest.raises(nf.NoiseFloorStop, match="declares no data"):
        nf.load_node(nf.validate_chunk(chunk(data_manifest_sha256=None)), root)
    (root / NODE_DIR / "data").unlink()
    with pytest.raises(nf.NoiseFloorStop, match="missing"):
        nf.load_node(nf.validate_chunk(chunk(data_manifest_sha256=good)), root)


def test_edit_file_sha_and_unisolated_spec_are_refused(tmp_path):
    root = _copy_harness(tmp_path)
    edit = root / NODE_DIR / "representative_edits" / "E2.diff"
    edit.write_text(edit.read_text() + "\n")
    with pytest.raises(nf.NoiseFloorStop, match="committed-file pin"):
        nf.load_node(nf.validate_chunk(chunk()), root)
    # even with a pin taken from the changed tree, the edit file must match the manifest's sha256
    from autoresearch.trackg import isolation
    from autoresearch.trackg.node_template import load_trackg_node_spec as _load
    changed = nf.frozen_pin_sha256(isolation.frozen_file_hashes(root / NODE_DIR, _load(root / SPEC).frozen_paths))
    with pytest.raises(nf.NoiseFloorStop, match="E2.diff sha256 differs"):
        nf.load_node(nf.validate_chunk(chunk(committed_pin_sha256=changed)), root)
    import yaml
    root2 = _copy_harness(tmp_path / "b")
    mapping = yaml.safe_load((root2 / SPEC).read_text())
    del mapping["isolation"]
    (root2 / SPEC).write_text(yaml.safe_dump(mapping))
    from autoresearch.trackg.node_template import load_trackg_node_spec
    with pytest.raises(nf.NoiseFloorStop, match="no isolation block"):
        nf.load_node(nf.validate_chunk(chunk(node_spec_sha256=load_trackg_node_spec(root2 / SPEC).raw_sha256)), root2)


# ---------------------------------------------------------------------------------------------- run loop
class FakeRunner:
    """Stands in for WorkingCopyRunner: writes a record per run and returns what the test scripted."""

    script: dict[str, dict] = {}
    calls: list[dict] = []

    def __init__(self, spec, node_dir, runs_root, *, frozen_pin=None):
        self.runs_root = Path(runs_root)
        assert frozen_pin

    def run(self, run_id, limits, *, train_seed=None, split_seed=None, diff_text=None, interpreter="python3",
            keep_code_snapshot=True, **_):
        assert limits.isolation_waivers == {} and limits.timeout()[0] == pytest.approx(3 * 12.297)
        assert keep_code_snapshot is False
        FakeRunner.calls.append({"run_id": run_id, "train": train_seed, "split": split_seed, "diff": diff_text})
        run_dir = self.runs_root / run_id
        (run_dir / "work").mkdir(parents=True)
        record = copy.deepcopy(FakeRunner.script.get(run_id, effective_record(run_id)))
        record["run_id"] = run_id
        if diff_text is not None:
            record["diff"] = {"status": "applied"}
        (run_dir / "run_record.json").write_text(json.dumps(record))
        return RunResult(run_id, run_dir, run_dir / "work", None, record)


@pytest.fixture
def fake():
    FakeRunner.script, FakeRunner.calls = {}, []
    return FakeRunner


def _rows(out: Path):
    return [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]


def test_run_chunk_writes_one_row_per_run(tmp_path, fake):
    c = nf.validate_chunk(chunk())
    rec = nf.run_chunk(c, HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,),
                       runner_factory=fake)
    assert rec["stop"] is None and rec["written_rows"] == rec["expected_runs"] == 15
    rows = _rows(tmp_path / "out")
    assert [(r["config"], r["train_seed"], r["split_seed"]) for r in rows[:5]] == [
        ("c0", 1001, None), ("c1", 1001, None), ("c2", 1001, None), ("c3", 1001, None), ("c4", 1001, None)]
    assert all(r["success"] and r["metric"] == 0.5 and r["isolation"]["effective"] is True for r in rows)
    assert fake.calls[0]["diff"] is None and fake.calls[1]["diff"] is not None
    assert json.loads((tmp_path / "out" / "edits.json").read_text())[NODE][0]["edit_id"] == "E1"
    assert not any((tmp_path / "out" / "runs").glob("*/work"))  # node copies removed, records kept
    assert all("sealed" not in r and "test" not in json.dumps(r) for r in rows)


def test_unisolated_record_stops_the_chunk(tmp_path, fake):
    fake.script["c2_train_t1001_sd"] = effective_record(waivers={"root": "x"})
    rec = nf.run_chunk(nf.validate_chunk(chunk()), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",),
                       cpu_list=(0,), runner_factory=fake)
    assert rec["stop"].startswith("UnisolatedRunError") and rec["written_rows"] == 2
    assert len(_rows(tmp_path / "out")) == 2
    assert (tmp_path / "out" / "runs" / "c2_train_t1001_sd" / "run_record.json").is_file()  # evidence kept


def test_infrastructure_failure_stops_divergent_is_a_failed_row(tmp_path, fake):
    div = effective_record()
    div["outcome"] = {"kind": "divergent", "value": None, "reason": "timeout"}
    fake.script["c1_train_t1001_sd"] = div
    infra = effective_record()
    infra["outcome"] = {"kind": "infrastructure_failure", "value": None, "reason": "AR_INFRA"}
    fake.script["c0_train_t1002_sd"] = infra
    rec = nf.run_chunk(nf.validate_chunk(chunk()), HARNESS, tmp_path / "out", gpu_uuids=("GPU-12345678-aaaa",),
                       cpu_list=(0,), runner_factory=fake)
    rows = _rows(tmp_path / "out")
    assert rows[1]["success"] is False and rows[1]["failure"] == "divergent:timeout" and rows[1]["metric"] is None
    assert rec["written_rows"] == 5 and "infrastructure failure" in rec["stop"]


def test_stop_on_timeout_stops_right_after_the_first_timeout_row(tmp_path, fake):
    div = effective_record()
    div["outcome"] = {"kind": "divergent", "value": None, "reason": "timeout"}
    fake.script["c1_train_t1001_sd"] = div
    rec = nf.run_chunk(nf.validate_chunk(chunk(stop_on_timeout=True)), HARNESS, tmp_path / "out",
                       gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,), runner_factory=fake)
    assert rec["written_rows"] == 2 and rec["stop_kind"] == "timeout" and "declared timeout" in rec["stop"]
    assert _rows(tmp_path / "out")[1]["failure"] == "divergent:timeout"      # the evidence row is kept
    assert len(fake.calls) == 2
    with pytest.raises(nf.NoiseFloorStop, match="stop_on_timeout"):
        nf.validate_chunk(chunk(stop_on_timeout="yes"))
    with pytest.raises(nf.NoiseFloorStop, match="not a CPU-only node"):
        nf.validate_chunk(chunk(device="cpu"))


def test_unapplied_edit_and_deadline_stop(tmp_path, fake):
    refused = effective_record()
    refused["executed"] = False
    fake.script["c1_train_t1001_sd"] = refused
    rec = nf.run_chunk(nf.validate_chunk(chunk()), HARNESS, tmp_path / "a", gpu_uuids=("GPU-12345678-aaaa",),
                       cpu_list=(0,), runner_factory=fake)
    assert "was not executed" in rec["stop"]
    fake.script = {}
    rec = nf.run_chunk(nf.validate_chunk(chunk(deadline_epoch=100.0)), HARNESS, tmp_path / "b",
                       gpu_uuids=("GPU-12345678-aaaa",), cpu_list=(0,), runner_factory=fake, clock=lambda: 50.0)
    assert "deadline" in rec["stop"] and rec["written_rows"] == 0


def test_cli_refuses_without_an_htcondor_gpu(tmp_path, monkeypatch):
    for name in ("NVIDIA_VISIBLE_DEVICES", "_CONDOR_AssignedGPUs"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(chunk()))
    assert nf.main(["--chunk", str(path), "--harness-root", str(HARNESS), "--out", str(tmp_path / "o")]) == 2
    bad = chunk(isolation_waivers={"root": "x"})
    path.write_text(json.dumps(bad))
    assert nf.main(["--chunk", str(path), "--harness-root", str(HARNESS), "--out", str(tmp_path / "o")]) == 2


SMI = ("GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-4d623e76-7587-ee15-a633-46f60d220d94)\n")
UUID = "GPU-4d623e76-7587-ee15-a633-46f60d220d94"


def test_explicit_gpu_for_a_direct_host_run():
    from autoresearch.trackg import gpu_identity as gi
    assert gi.resolve_explicit(UUID, {}, SMI)[0] == (UUID,)
    for env, match in (({"NVIDIA_VISIBLE_DEVICES": "GPU-4d623e76"}, "scheduler variables"),
                       ({"CUDA_VISIBLE_DEVICES": "0"}, "must be empty")):
        with pytest.raises(gi.GpuIdentityError, match=match):
            gi.resolve_explicit(UUID, env, SMI)
    with pytest.raises(gi.GpuIdentityError, match="full GPU UUID"):
        gi.resolve_explicit("GPU-4d623e76", {}, SMI)
    with pytest.raises(gi.GpuIdentityError, match="exactly once"):
        gi.resolve_explicit("GPU-00000000-0000-0000-0000-000000000000", {}, SMI)


def test_cli_explicit_gpu_and_cpu_only_chunks(tmp_path, monkeypatch):
    for name in ("NVIDIA_VISIBLE_DEVICES", "_CONDOR_AssignedGPUs", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    def fake_run_chunk(chunk_, root, out, *, gpu_uuids, cpu_list, **kw):
        seen["gpus"] = gpu_uuids
        return {"node": chunk_.node, "chunk_id": None, "expected_runs": 1, "written_rows": 1, "stop": None}
    monkeypatch.setattr(nf, "run_chunk", fake_run_chunk)
    monkeypatch.setattr(nf.gpu_identity, "resolve_explicit_on_host",
                        lambda uuid, env: nf.gpu_identity.resolve_explicit(uuid, env, SMI))
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(chunk()))
    args = ["--chunk", str(path), "--harness-root", str(HARNESS), "--out", str(tmp_path / "o")]
    assert nf.main(args + ["--gpu-uuid", UUID]) == 0 and seen["gpus"] == (UUID,)
    path.write_text(json.dumps(chunk(device="cpu")))
    assert nf.main(args) == 2                                      # mlab_cifar10 is not a CPU-only node
    monkeypatch.setattr(nf, "CPU_ONLY_NODES", frozenset({NODE}))   # as for mlgym_houseprice
    assert nf.main(args) == 0 and seen["gpus"] == ()
    assert nf.main(args + ["--gpu-uuid", UUID]) == 2              # a CPU-only chunk takes no GPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert nf.main(args) == 2
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    path.write_text(json.dumps(chunk(device="tpu")))
    assert nf.main(args) == 2

