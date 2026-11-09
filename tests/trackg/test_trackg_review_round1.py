"""Regression tests for the round-1 review of the Track G port kit (findings 1-15)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from autoresearch.trackg import runner as runner_module
from autoresearch.trackg.diff_support import EvaluatedStates, apply_diff, code_state_hashes
from autoresearch.trackg.node_template import TrackGNodeSpec, TrackGNodeSpecError, WallClockBudgetError
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK, ProcessExit, classify_run
from autoresearch.trackg.runner import ResourceLimits, RunnerError, WorkingCopyRunner
from autoresearch.trackg.selection import SelectionPolicyError
from autoresearch.trackg.smoke import run_smoke_campaign

from .fixtures import make_node, spec_mapping

LIMITS = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=10.0)


def base_spec(**overrides):
    return TrackGNodeSpec.from_mapping(spec_mapping(**overrides)).to_base_node_spec()


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return state[state.rfind(")") + 2:].split()[0] != "Z"


# ---------------------------------------------------------------- 1. duplicate hash and context

TWO = "def a():\n    x = 1\n    return x\n\n\ndef b():\n    x = 1\n    return x\n"


def _two_location_node(tmp_path):
    node = make_node(tmp_path)
    (node / "model" / "two.py").write_text(TWO)
    return node


def test_same_change_at_two_places_is_not_a_duplicate(tmp_path):
    node, base = _two_location_node(tmp_path), base_spec()
    in_a = ("--- a/model/two.py\n+++ b/model/two.py\n@@ -1,3 +1,3 @@\n def a():\n-    x = 1\n+    x = 2\n"
            "     return x\n")
    in_b = ("--- a/model/two.py\n+++ b/model/two.py\n@@ -6,3 +6,3 @@\n def b():\n-    x = 1\n+    x = 2\n"
            "     return x\n")
    evaluated = EvaluatedStates()
    first = apply_diff(node, in_a, base, evaluated=evaluated, write=False)
    evaluated.register("A", code_state_sha256=first.provenance["code_state_sha256"],
                       normalised_diff_sha256=first.provenance["normalised_diff_sha256"],
                       incumbent_code_state_sha256=first.provenance["incumbent_code_state_sha256"])
    second = apply_diff(node, in_b, base, evaluated=evaluated, write=False)
    assert second.status == "applied" and second.provenance["duplicate"] is False
    assert second.provenance["normalised_diff_sha256"] != first.provenance["normalised_diff_sha256"]


def test_split_and_combined_hunks_do_not_collide(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    (node / "model" / "f.py").write_text("a = 1\nc = 3\n")
    split = "--- a/model/f.py\n+++ b/model/f.py\n@@ -1,1 +1,1 @@\n-a = 1\n+b = 2\n@@ -2,1 +2,1 @@\n-c = 3\n+d = 4\n"
    combined = "--- a/model/f.py\n+++ b/model/f.py\n@@ -1,2 +1,2 @@\n-a = 1\n-c = 3\n+b = 2\n+d = 4\n"
    h1 = apply_diff(node, split, base, write=False).provenance["normalised_diff_sha256"]
    h2 = apply_diff(node, combined, base, write=False).provenance["normalised_diff_sha256"]
    assert h1 != h2


def test_same_diff_against_another_incumbent_is_reported_not_deduplicated(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    lr = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
    width = "--- a/model/net.py\n+++ b/model/net.py\n@@ -1,1 +1,1 @@\n-WIDTH = 64\n+WIDTH = 128\n"
    evaluated = EvaluatedStates()
    first = apply_diff(node, lr, base, evaluated=evaluated, write=False)
    evaluated.register("t1", code_state_sha256=first.provenance["code_state_sha256"],
                       normalised_diff_sha256=first.provenance["normalised_diff_sha256"],
                       incumbent_code_state_sha256=first.provenance["incumbent_code_state_sha256"])
    assert apply_diff(node, width, base).status == "applied"  # a keep changes the incumbent
    again = apply_diff(node, lr, base, evaluated=evaluated, write=False)
    assert again.status == "applied" and again.provenance["duplicate"] is False
    assert again.provenance["seen_diff_other_incumbent"] == "t1"


# ---------------------------------------------------------------- 2. timeout kills the whole group

SURVIVOR = """import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open("child.pid", "w").write(str(os.getpid()))
time.sleep(60)
"""


def test_timeout_sigkills_a_sigterm_ignoring_child_under_a_compound_command(tmp_path):
    node = make_node(tmp_path)
    (node / "job.py").write_text(SURVIVOR)
    spec = TrackGNodeSpec.from_mapping(spec_mapping(run_command="cd . && python3 job.py"))
    runner = WorkingCopyRunner(spec, node, tmp_path / "runs")
    limits = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=0.5, kill_grace_seconds=1.0)
    result = runner.run("r1", limits)
    pid = int((result.work_dir / "child.pid").read_text())
    time.sleep(0.2)
    assert result.outcome.kind == INFRASTRUCTURE_FAILURE and result.outcome.reason == "timeout"
    assert result.record["process"]["killed_with"] == "SIGKILL"
    assert not _alive(pid)


def test_leftover_processes_after_a_normal_exit_are_killed_and_recorded(tmp_path):
    node = make_node(tmp_path)
    spec = TrackGNodeSpec.from_mapping(spec_mapping(
        run_command="(sleep 60 & echo $! > sleeper.pid) && python3 train.py"))
    result = WorkingCopyRunner(spec, node, tmp_path / "runs").run("r1", LIMITS)
    pid = int((result.work_dir / "sleeper.pid").read_text())
    time.sleep(0.2)
    assert result.outcome.kind == OK
    assert result.record["process"]["leftover_processes_killed"] is True
    assert not _alive(pid)


# ---------------------------------------------------------------- 3. binary files in the code state


def test_binary_file_in_an_editable_directory_is_hashed_by_bytes(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    (node / "model" / "vocab.bin").write_bytes(bytes([0xff, 0xfe, 0x00, 0x80]))
    h0 = code_state_hashes(node, base)
    lr = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
    assert apply_diff(node, lr, base).status == "applied"
    (node / "model" / "vocab.bin").write_bytes(bytes([0xff, 0xfe, 0x00, 0x81]))
    assert code_state_hashes(node, base)[0] not in {h0[0]}


# ---------------------------------------------------------------- 4. the test metric cannot be selected


def test_loose_metric_regex_matching_the_test_line_is_divergent_at_run_time():
    log = "step 1 loss: 0.5\nval_accuracy: 0.7\ntest_accuracy: 0.42\n"
    outcome = classify_run(log, ProcessExit(0), metric_regex=r"accuracy:\s*(\S+)",
                           loss_regex=r"^step\s+\d+\s+loss:\s*(\S+)", test_metric_regex=r"^test_accuracy:\s*(\S+)")
    assert (outcome.kind, outcome.reason) == (DIVERGENT, "metric_regex_matches_test_line")


def test_loose_metric_regex_is_refused_at_load_time():
    metric = {"metric_regex": r"accuracy:\s*(\S+)", "loss_regex": r"^step\s+\d+\s+loss:\s*(\S+)",
              "test_metric_regex": r"^test_accuracy:\s*(\S+)"}
    with pytest.raises(SelectionPolicyError):
        TrackGNodeSpec.from_mapping(spec_mapping(metric=metric))


# ---------------------------------------------------------------- 6. wall-clock refusal


@pytest.mark.parametrize("overrides", [
    {"external_benchmark": {"time_limit": 3600}},
    {"external_benchmark": {"timeout_s": 10}},
    {"notes": {"runtime_minutes": 5}},
    {"notes": {"max_time": 5}},
    {"external_benchmark": {"hours": 1}},
    {"run_command": "python3 train.py --time-budget 300"},
    {"run_command": "TIME_BUDGET=300 python3 train.py"},
    {"run_command": "python3 train.py --timeout 300"},
    {"run_command": "python3 train.py --max-time 300"},
    {"run_command": "python3 train.py --time 300"},
])
def test_time_keys_and_time_flags_are_refused(overrides):
    with pytest.raises(WallClockBudgetError):
        TrackGNodeSpec.from_mapping(spec_mapping(**overrides))


def test_step_flags_that_mention_time_words_are_not_refused():
    spec = TrackGNodeSpec.from_mapping(spec_mapping(run_command="python3 train.py --time-steps {budget}"))
    assert spec.budgeted_run_command().endswith("--time-steps 5")


# ---------------------------------------------------------------- 7. NaN loss detection


def test_loss_regex_is_required():
    metric = {"metric_regex": r"^val_accuracy:\s*(\S+)", "test_metric_regex": r"^test_accuracy:\s*(\S+)"}
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(metric=metric))


@pytest.mark.parametrize("token", ["tensor(nan)", "nan%", "tensor(inf, device='cuda:0')", "NaN,"])
def test_non_numeric_loss_tokens_with_nan_or_inf_are_divergent(token):
    log = f"step 1 loss: {token}\nval: 0.5\n"
    outcome = classify_run(log, ProcessExit(0), metric_regex=r"^val:\s*(\S+)", loss_regex=r"loss:\s*(.+)$")
    assert (outcome.kind, outcome.reason) == (DIVERGENT, "non_finite_loss")


def test_finite_tensor_loss_is_not_divergent():
    log = "step 1 loss: tensor(0.25, device='cuda:0')\nval: 0.5\n"
    assert classify_run(log, ProcessExit(0), metric_regex=r"^val:\s*(\S+)",
                        loss_regex=r"loss:\s*(.+)$").kind == OK


# ---------------------------------------------------------------- 8. shadowing modules


@pytest.mark.parametrize("path", ["model/numpy.py", "model/sitecustomize.py", "model/json/__init__.py",
                                  "model/os.py", "model/yaml.py"])
def test_new_files_that_could_shadow_imports_are_refused(tmp_path, path):
    node, base = make_node(tmp_path), base_spec()
    diff = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,1 @@\n+X = 1\n"
    result = apply_diff(node, diff, base, allow_new_files=True)
    assert result.provenance["refusal"] == "shadowing_module"
    assert not (node / path).exists()


def test_new_files_follow_the_node_flag(tmp_path):
    node = make_node(tmp_path)
    diff = "--- /dev/null\n+++ b/model/zz_helpers_trackg.py\n@@ -0,0 +1,1 @@\n+X = 1\n"
    assert apply_diff(node, diff, base_spec()).provenance["refusal"] == "new_file_not_allowed"
    spec = TrackGNodeSpec.from_mapping(spec_mapping(allow_new_files=True))
    result = WorkingCopyRunner(spec, node, tmp_path / "runs").run("r1", LIMITS, diff_text=diff)
    assert result.record["diff"]["status"] == "applied"


# ---------------------------------------------------------------- 9. smoke promotes code, not outputs


def test_smoke_promotes_the_pre_run_code_snapshot(tmp_path):
    node = make_node(tmp_path, mode="writeout")
    spec = TrackGNodeSpec.from_mapping(spec_mapping())
    lr = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
    records = run_smoke_campaign(spec, node, tmp_path / "runs", [lr], LIMITS)
    assert [r["decision"] for r in records] == ["baseline", "kept"]
    base_rec, kept = records[0]["record"], records[1]["record"]
    assert kept["incumbent_code_state_sha256"] == base_rec["code_state_sha256"]
    assert "resumed_from_checkpoint" not in (tmp_path / "runs" / "s001" / "run.log").read_text()
    assert not (tmp_path / "runs" / "s000_baseline" / "code" / "ckpt.txt").exists()
    assert (tmp_path / "runs" / "s000_baseline" / "work" / "ckpt.txt").exists()


# ---------------------------------------------------------------- 10-11. caps and launch failures


def test_caps_are_required_or_explicitly_waived():
    with pytest.raises(RunnerError):
        ResourceLimits(measured_run_seconds=1)
    with pytest.raises(RunnerError):
        ResourceLimits(measured_run_seconds=1, cpu_list=(0,))  # no memory cap
    with pytest.raises(RunnerError):
        ResourceLimits(measured_run_seconds=1, cpu_list=(0,), memory_method="external")  # no external_limit
    ok = ResourceLimits(measured_run_seconds=1, cpu_list=(0,), memory_method="external",
                        external_limit="slurm --mem=64G")
    assert ok.to_dict()["uncapped"] == []
    waived = ResourceLimits(measured_run_seconds=1, uncapped_reason="laptop test")
    assert waived.to_dict()["uncapped"] == ["CPU", "memory"]


def test_cpu_outside_the_process_cpu_set_is_refused_before_anything_is_written(tmp_path):
    node = make_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(spec_mapping()), node, tmp_path / "runs")
    bad = max(os.sched_getaffinity(0)) + 4096
    with pytest.raises(RunnerError):
        runner.run("r1", ResourceLimits(measured_run_seconds=1, cpu_list=(bad,), memory_bytes=2 ** 31))
    assert not (tmp_path / "runs" / "r1").exists()


def test_failure_inside_preexec_is_an_infrastructure_failure_with_a_record(tmp_path, monkeypatch):
    def broken(limits):
        def apply():
            raise OSError("cannot apply limits")
        return apply

    monkeypatch.setattr(runner_module, "_limits_preexec", broken)
    node = make_node(tmp_path)
    runner = WorkingCopyRunner(TrackGNodeSpec.from_mapping(spec_mapping()), node, tmp_path / "runs")
    result = runner.run("r1", LIMITS)
    assert (result.outcome.kind, result.outcome.reason) == (INFRASTRUCTURE_FAILURE, "launch_error")
    assert (result.run_dir / "run_record.json").exists()


# ---------------------------------------------------------------- notes 12c-15


def test_determinism_flags_must_be_real_booleans():
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(determinism={"cudnn_benchmark": "false"}))


def test_shebang_and_coding_cookie_changes_are_not_inert(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    (node / "model" / "run.sh").write_text("#!/usr/bin/env python3\necho hi\n")
    shebang = ("--- a/model/run.sh\n+++ b/model/run.sh\n@@ -1,1 +1,1 @@\n-#!/usr/bin/env python3\n"
               "+#!/usr/bin/env python2\n")
    assert apply_diff(node, shebang, base).status == "applied"
    (node / "model" / "enc.py").write_text("# -*- coding: utf-8 -*-\nX = 1\n")
    cookie = ("--- a/model/enc.py\n+++ b/model/enc.py\n@@ -1,1 +1,1 @@\n-# -*- coding: utf-8 -*-\n"
              "+# -*- coding: latin-1 -*-\n")
    assert apply_diff(node, cookie, base).status == "applied"


def test_new_file_mode_other_than_644_is_refused(tmp_path):
    node = make_node(tmp_path)
    diff = ("diff --git a/model/x.sh b/model/x.sh\nnew file mode 100755\n--- /dev/null\n+++ b/model/x.sh\n"
            "@@ -0,0 +1,1 @@\n+echo hi\n")
    result = apply_diff(node, diff, base_spec(), allow_new_files=True)
    assert result.provenance["refusal"] == "unsupported_diff_feature"


@pytest.mark.parametrize("editable, frozen", [
    (["model/"], ["model"]),
    (["model/net.py"], ["model/"]),
    (["."], ["evaluate.py"]),
    (["train.py"], ["."]),
])
def test_scope_overlap_is_checked_after_normalisation(editable, frozen):
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(editable_paths=editable, frozen_paths=frozen))
