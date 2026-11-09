"""M10 step 0: the per-run working-copy runner on fake runs (real subprocesses, CPU only)."""
from __future__ import annotations

import json
import os
import time

import pytest

from autoresearch.trackg.diff_support import EvaluatedStates
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK
from autoresearch.trackg.runner import ResourceLimits, RunnerError, WorkingCopyRunner

from .fixtures import make_node, spec_mapping

LIMITS = ResourceLimits(measured_run_seconds=10.0)


def runner_for(tmp_path, mode="ok", **spec_overrides):
    node = make_node(tmp_path, mode)
    spec = TrackGNodeSpec.from_mapping(spec_mapping(**spec_overrides))
    return WorkingCopyRunner(spec, node, tmp_path / "runs"), node


def test_ok_run_records_outcome_limits_and_determinism(tmp_path):
    runner, node = runner_for(tmp_path)
    before = (node / "train.py").read_bytes()
    result = runner.run("r1", LIMITS, train_seed=8001, split_seed=42)
    assert result.outcome.kind == OK and result.outcome.value == pytest.approx(0.6)
    record = json.loads((result.run_dir / "run_record.json").read_text())
    assert record["schema"] == "trackg_run/v1" and record["outcome"]["kind"] == "ok"
    assert record["run_budget"] == {"kind": "steps", "value": 5}
    assert record["environment_applied"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert record["environment_applied"]["CUDA_VISIBLE_DEVICES"] == ""
    assert record["environment_applied"]["OMP_NUM_THREADS"] == "1"
    assert record["environment_applied"]["AR_NUM_WORKERS"] == "2"
    assert record["environment_applied"]["AR_TRAIN_SEED"] == "8001"
    assert record["environment_applied"]["timeout_seconds"].startswith("30 (3 x measured 10 s)")
    det = record["determinism"]
    assert det["shim_loaded"] is True
    statuses = {s["setting"]: s for s in det["settings"]}
    assert set(statuses) == {"cudnn.benchmark=False", "torch.use_deterministic_algorithms(True)",
                             "torch.set_num_threads"}
    try:
        import torch  # noqa: F401
    except ImportError:
        # torch absent: recorded as not forced, and the run still succeeds
        assert all(s["status"] == "not_forced" and s["reason"].startswith("torch_absent") for s in statuses.values())
        assert det["not_forced"] == sorted(statuses)
    # the test metric is sealed, not the selection value
    assert record["sealed"] == {"test_accuracy": 0.4242}
    # the node itself is untouched; the copy is per run
    assert (node / "train.py").read_bytes() == before
    assert result.work_dir != node and (result.work_dir / "train.py").exists()


def test_nan_loss_run_is_divergent(tmp_path):
    runner, _ = runner_for(tmp_path, mode="nan")
    result = runner.run("r1", LIMITS)
    assert (result.outcome.kind, result.outcome.reason) == (DIVERGENT, "non_finite_loss")


def test_no_metric_and_short_runs_are_divergent(tmp_path):
    runner, _ = runner_for(tmp_path, mode="nometric")
    assert runner.run("a", LIMITS).outcome.reason == "no_metric_at_budget"
    runner2, _ = runner_for(tmp_path / "s", mode="short")
    assert runner2.run("b", LIMITS).outcome.reason == "budget_not_reached"
    runner3, _ = runner_for(tmp_path / "c", mode="crash")
    assert runner3.run("c", LIMITS).outcome.reason == "nonzero_exit"


def test_timeout_is_infrastructure_failure_and_the_process_group_is_killed(tmp_path):
    runner, _ = runner_for(tmp_path, mode="sleep")
    limits = ResourceLimits(measured_run_seconds=0.2, timeout_factor=3.0, kill_grace_seconds=2.0)
    start = time.monotonic()
    result = runner.run("r1", limits)
    assert time.monotonic() - start < 20
    assert (result.outcome.kind, result.outcome.reason) == (INFRASTRUCTURE_FAILURE, "timeout")
    assert result.record["process"]["timed_out"] is True


def test_cpu_and_memory_caps_are_applied(tmp_path):
    runner, _ = runner_for(tmp_path, mode="affinity")
    cpu = sorted(os.sched_getaffinity(0))[0]
    limits = ResourceLimits(measured_run_seconds=10, cpu_list=(cpu,), memory_bytes=2 * 1024 ** 3)
    result = runner.run("r1", limits)
    log = (result.run_dir / "run.log").read_text()
    assert f"affinity: [{cpu}]" in log
    assert f"rlimit_data: {2 * 1024 ** 3}" in log
    assert result.outcome.kind == OK


def test_gpu_uuid_is_exported_and_indices_are_refused(tmp_path):
    runner, _ = runner_for(tmp_path, mode="env")
    uuid = "GPU-12345678-90ab-cdef-1234-567890abcdef"
    result = runner.run("r1", ResourceLimits(gpu_uuids=(uuid,), measured_run_seconds=10))
    assert f"env CUDA_VISIBLE_DEVICES='{uuid}'" in (result.run_dir / "run.log").read_text()
    for bad in ("0", "1", "cuda:0", "GPU-1"):
        with pytest.raises(RunnerError):
            ResourceLimits(gpu_uuids=(bad,), measured_run_seconds=10)


def test_timeout_basis_is_required_and_run_ids_are_never_reused(tmp_path):
    with pytest.raises(RunnerError):
        ResourceLimits()
    provisional = ResourceLimits(provisional_timeout_seconds=30)
    assert provisional.timeout() == (30.0, "provisional (measuring run)")
    runner, _ = runner_for(tmp_path)
    runner.run("r1", LIMITS)
    with pytest.raises(RunnerError):
        runner.run("r1", LIMITS)
    with pytest.raises(RunnerError):
        runner.run("../escape", LIMITS)


def test_frozen_path_edit_is_refused_and_not_run(tmp_path):
    runner, node = runner_for(tmp_path)
    diff = ("--- a/evaluate.py\n+++ b/evaluate.py\n@@ -1,2 +1,2 @@\n"
            " def evaluate(preds, labels):\n"
            "-    return sum(p == l for p, l in zip(preds, labels)) / len(labels)\n"
            "+    return 1.0\n")
    result = runner.run("r1", LIMITS, diff_text=diff)
    assert result.outcome is None and result.record["executed"] is False
    assert result.record["diff"]["status"] == "refused"
    assert result.record["diff"]["refusal"] == "scope_violation"
    assert "evaluate.py is outside editable_paths" in result.record["diff"]["refusal_detail"]
    assert (result.work_dir / "evaluate.py").read_text() == (node / "evaluate.py").read_text()
    assert not (result.run_dir / "run.log").exists()


def test_diff_run_records_provenance_and_registers_the_evaluated_state(tmp_path):
    runner, node = runner_for(tmp_path)
    evaluated = EvaluatedStates()
    base = runner.run("base", LIMITS, evaluated=evaluated)
    diff = "--- a/train.py\n+++ b/train.py\n@@ -7,1 +7,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
    first = runner.run("t1", LIMITS, diff_text=diff, evaluated=evaluated)
    assert first.outcome.kind == OK and first.outcome.value == pytest.approx(0.7)
    prov = first.record["diff"]
    assert prov["schema"] == "manager_proposal/v2-diff" and prov["files_touched"] == ["train.py"]
    assert prov["code_state_sha256"] == first.record["code_state_sha256"]
    assert prov["incumbent_code_state_sha256"] == base.record["code_state_sha256"]
    again = runner.run("t2", LIMITS, diff_text=diff, evaluated=evaluated)
    assert again.record["diff"]["status"] == "duplicate" and again.record["diff"]["duplicate_of"] == "t1"
    assert again.outcome is None
    assert "LEARNING_RATE = 0.1" in (node / "train.py").read_text()


FAKE_TORCH = '''
__version__ = "0.0-fake"
class _Cudnn:
    benchmark = True
    deterministic = False
class backends:
    cudnn = _Cudnn
_calls = []
def use_deterministic_algorithms(mode, warn_only=False):
    import os
    if os.environ.get("FAKE_TORCH_REFUSE") == "1":
        raise RuntimeError("refused by fake torch")
    _calls.append((mode, warn_only))
def set_num_threads(n):
    _calls.append(("threads", n))
'''


def test_determinism_settings_are_forced_when_torch_is_present_and_failures_are_recorded(tmp_path, monkeypatch):
    fake = tmp_path / "fake_torch"
    (fake / "torch").mkdir(parents=True)
    (fake / "torch" / "__init__.py").write_text(FAKE_TORCH)
    monkeypatch.setenv("PYTHONPATH", str(fake))
    runner, _ = runner_for(tmp_path / "a",
                           determinism={"deterministic_warn_only": True})
    det = runner.run("r1", LIMITS).record["determinism"]
    statuses = {s["setting"]: (s["status"], s["reason"]) for s in det["settings"]}
    assert det["shim_loaded"] and det["torch_version"] == "0.0-fake" and det["warn_only"] is True
    assert statuses["cudnn.benchmark=False"] == ("forced", "")
    assert statuses["torch.use_deterministic_algorithms(True)"] == ("forced", "warn_only=True")
    assert statuses["torch.set_num_threads"] == ("forced", "1")
    assert det["not_forced"] == []
    monkeypatch.setenv("FAKE_TORCH_REFUSE", "1")
    runner2, _ = runner_for(tmp_path / "b")
    det2 = runner2.run("r2", LIMITS).record["determinism"]
    assert det2["not_forced"] == ["torch.use_deterministic_algorithms(True)"]


def test_non_python_command_records_shim_not_loaded(tmp_path):
    metric = {"metric_regex": r"^val_accuracy:\s*(\S+)", "test_metric_regex": r"^test_accuracy:\s*(\S+)"}
    runner, _ = runner_for(tmp_path, run_command="echo val_accuracy: 0.5", metric=metric)
    record = runner.run("r1", LIMITS).record
    assert record["outcome"]["kind"] == "ok"
    assert record["determinism"]["shim_loaded"] is False
    assert all(s["reason"] == "shim_not_loaded" for s in record["determinism"]["settings"])


def test_infrastructure_failure_is_not_registered_as_an_evaluated_state(tmp_path):
    runner, _ = runner_for(tmp_path, mode="sleep")
    evaluated = EvaluatedStates()
    result = runner.run("r1", ResourceLimits(measured_run_seconds=0.1, kill_grace_seconds=1.0), evaluated=evaluated)
    assert result.outcome.kind == INFRASTRUCTURE_FAILURE
    assert evaluated.by_state == {}
