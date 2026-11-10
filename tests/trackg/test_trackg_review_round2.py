"""Regression tests for the round-2 review of the Track G port kit (findings 1-7)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autoresearch.trackg.declared_edits import declare_edits
from autoresearch.trackg.diff_support import EvaluatedStates, apply_diff, fixed_import_path, shadowing_name
from autoresearch.trackg.node_template import TrackGNodeSpec, TrackGNodeSpecError, WallClockBudgetError
from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
from autoresearch.trackg.selection import SelectionPolicyError

from .fixtures import make_node, spec_mapping

ROOT = Path(__file__).resolve().parents[2]
LIMITS = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=10.0)


def base_spec(**overrides):
    return TrackGNodeSpec.from_mapping(spec_mapping(**overrides)).to_base_node_spec()


def register(evaluated, trial, result):
    p = result.provenance
    evaluated.register(trial, code_state_sha256=p["code_state_sha256"],
                       normalised_diff_sha256=p["normalised_diff_sha256"],
                       incumbent_code_state_sha256=p["incumbent_code_state_sha256"])


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


# ---------------------------------------------------------------- 1. code state decides duplicates

TWO = "def a():\n    x = 1\n    return x\n\n\ndef b():\n    x = 1\n    return x\n"
BLOCK = ("layers = [\n    Linear(64),\n    ReLU(),\n    Linear(64),\n    ReLU(),\n    Linear(64),\n    ReLU(),\n]\n")


def test_u0_diffs_at_two_places_are_not_duplicates(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    (node / "model" / "two.py").write_text(TWO)
    in_a = "--- a/model/two.py\n+++ b/model/two.py\n@@ -2 +2 @@\n-    x = 1\n+    x = 2\n"
    in_b = "--- a/model/two.py\n+++ b/model/two.py\n@@ -7 +7 @@\n-    x = 1\n+    x = 2\n"
    evaluated = EvaluatedStates()
    first = apply_diff(node, in_a, base, evaluated=evaluated, write=False)
    register(evaluated, "A", first)
    second = apply_diff(node, in_b, base, evaluated=evaluated, write=False)
    assert second.status == "applied" and second.provenance["duplicate"] is False
    assert second.provenance["hunks"][0]["applied_at"] == 7
    assert second.provenance["normalised_diff_sha256"] == first.provenance["normalised_diff_sha256"]
    assert second.provenance["same_diff_hash_other_state"] == "A"


def test_edit_to_first_vs_second_copy_of_a_repeated_block_is_not_a_duplicate(tmp_path):
    node, base = make_node(tmp_path), base_spec()
    (node / "model" / "block.py").write_text(BLOCK)
    first_copy = ("--- a/model/block.py\n+++ b/model/block.py\n@@ -1,3 +1,3 @@\n layers = [\n"
                  "-    Linear(64),\n+    Linear(128),\n     ReLU(),\n")
    second_copy = ("--- a/model/block.py\n+++ b/model/block.py\n@@ -3,3 +3,3 @@\n     ReLU(),\n"
                   "-    Linear(64),\n+    Linear(128),\n     ReLU(),\n")
    evaluated = EvaluatedStates()
    first = apply_diff(node, first_copy, base, evaluated=evaluated, write=False)
    register(evaluated, "first", first)
    second = apply_diff(node, second_copy, base, evaluated=evaluated, write=False)
    assert second.status == "applied" and second.provenance["duplicate"] is False
    assert second.provenance["code_state_sha256"] != first.provenance["code_state_sha256"]
    # applying the first copy's diff again is a code-state duplicate
    again = apply_diff(node, first_copy, base, evaluated=evaluated, write=False)
    assert (again.status, again.provenance["duplicate_kind"], again.provenance["duplicate_of"]) == (
        "duplicate", "code_state", "first")


# ---------------------------------------------------------------- 2. shadow check: new parts only, fixed path


def test_existing_directory_named_like_a_module_does_not_refuse_a_safe_new_file(tmp_path):
    node, base = make_node(tmp_path), base_spec(editable_paths=["train.py", "model/", "code/"])
    (node / "code").mkdir()
    (node / "code" / "existing.py").write_text("X = 1\n")
    diff = "--- /dev/null\n+++ b/code/zz_trackg_helper.py\n@@ -0,0 +1,1 @@\n+Y = 2\n"
    result = apply_diff(node, diff, base, allow_new_files=True)
    assert result.status == "applied", result.provenance["refusal_detail"]
    shadow = result.provenance["shadow_check"]
    assert shadow["basis"].startswith("interpreter:") or shadow["basis"] == "harness_sys_path"
    assert "" not in shadow["import_path"]
    # a new directory named like a stdlib module is still refused
    bad = "--- /dev/null\n+++ b/model/json/zz_x.py\n@@ -0,0 +1,1 @@\n+Y = 2\n"
    assert apply_diff(node, bad, base, allow_new_files=True).provenance["refusal"] == "shadowing_module"


@pytest.mark.parametrize("where", ["node", "harness_root", "tmp", "model"])
def test_shadow_check_does_not_depend_on_the_working_directory(tmp_path, monkeypatch, where):
    node = make_node(tmp_path)
    cwd = {"node": node, "harness_root": ROOT, "tmp": tmp_path, "model": node / "model"}[where]
    monkeypatch.chdir(cwd)
    monkeypatch.syspath_prepend("")  # as under python -m: the cwd is on sys.path
    import_path, _ = fixed_import_path(exclude=[node])
    assert os.path.realpath(str(cwd)) not in {os.path.realpath(p) for p in import_path}
    assert shadowing_name("model/zz_trackg_helper.py", node, import_path=import_path) is None
    assert shadowing_name("model/src/zz.py", node, import_path=import_path) is None  # repo dirs not on the path
    assert shadowing_name("model/numpy.py", node, import_path=import_path) == "numpy"
    assert shadowing_name("model/sitecustomize.py", node, import_path=import_path) == "sitecustomize"


def test_shadow_check_from_several_directories_in_subprocesses(tmp_path):
    node = make_node(tmp_path)
    code = ("import sys; sys.path.insert(0, %r); from autoresearch.trackg.diff_support import shadowing_name;"
            "print(shadowing_name('src/zz.py', %r), shadowing_name('model/zz.py', %r),"
            " shadowing_name('model/os.py', %r))") % (str(ROOT / "src"), str(node), str(node), str(node))
    outputs = set()
    for cwd in (ROOT, node, node / "model", tmp_path):
        out = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, check=True)
        outputs.add(out.stdout.strip())
    assert outputs == {"None None os"}


# ---------------------------------------------------------------- 3. declare_edits follows the node flag


def _edit(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_declare_edits_honours_allow_new_files_and_records_the_incumbent(tmp_path):
    node = make_node(tmp_path)
    d = tmp_path / "edits"
    d.mkdir()
    edits = [
        _edit(d / "E1.diff", "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"),
        _edit(d / "E2.diff", "--- a/model/net.py\n+++ b/model/net.py\n@@ -1,1 +1,1 @@\n-WIDTH = 64\n+WIDTH = 128\n"),
        _edit(d / "E3.diff", "--- /dev/null\n+++ b/model/zz_trackg_helpers.py\n@@ -0,0 +1,1 @@\n+X = 1\n"),
    ]
    from autoresearch.trackg.declared_edits import DeclaredEditsError
    with pytest.raises(DeclaredEditsError, match="refused"):
        declare_edits(TrackGNodeSpec.from_mapping(spec_mapping()), node, edits)
    manifest = declare_edits(TrackGNodeSpec.from_mapping(spec_mapping(allow_new_files=True)), node, edits)
    assert manifest["allow_new_files"] is True
    assert all(e["incumbent_code_state_sha256"] == manifest["baseline_code_state_sha256"] for e in manifest["edits"])
    # the runtime agrees with the declaration
    spec = TrackGNodeSpec.from_mapping(spec_mapping(allow_new_files=True))
    result = WorkingCopyRunner(spec, node, tmp_path / "runs").run("r1", LIMITS, diff_text=edits[2].read_text())
    assert result.record["diff"]["status"] == "applied"


# ---------------------------------------------------------------- 4. cleanup by session as well as group


def test_child_that_leaves_the_group_but_not_the_session_is_killed(tmp_path):
    node = make_node(tmp_path)
    (node / "job.py").write_text(
        "import os, subprocess, time\n"
        "p = subprocess.Popen(['sleep', '60'], process_group=0)\n"
        "open('child.pid', 'w').write(str(p.pid))\n"
        "for s in range(1, 6): print(f'step {s} loss: 0.5')\n"
        "print('val_accuracy: 0.7')\n")
    spec = TrackGNodeSpec.from_mapping(spec_mapping(run_command="python3 job.py"))
    result = WorkingCopyRunner(spec, node, tmp_path / "runs").run("r1", LIMITS)
    pid = int((result.work_dir / "child.pid").read_text())
    time.sleep(0.2)
    assert result.record["process"]["leftover_processes_killed"] is True
    assert not _alive(pid)


# ---------------------------------------------------------------- 5-7. spec checks


@pytest.mark.parametrize("command", ["TIMEOUT=3600 python3 train.py", "MAX_SECONDS=300 python3 train.py",
                                     "env RUN_TIME_LIMIT=60 python3 train.py"])
def test_time_budget_environment_prefixes_are_refused_with_the_word_named(command):
    with pytest.raises(WallClockBudgetError, match="contains"):
        TrackGNodeSpec.from_mapping(spec_mapping(run_command=command))


def test_key_error_message_names_the_time_word():
    with pytest.raises(WallClockBudgetError, match="'time'"):
        TrackGNodeSpec.from_mapping(spec_mapping(notes={"n_timesteps": 3}))


def test_loose_test_metric_regex_is_refused_at_load():
    metric = {"metric_regex": r"^val_accuracy:\s*(\S+)", "loss_regex": r"^step\s+\d+\s+loss:\s*(\S+)",
              "test_metric_regex": r"accuracy:\s*(\S+)"}
    with pytest.raises(SelectionPolicyError, match="test_metric_regex"):
        TrackGNodeSpec.from_mapping(spec_mapping(metric=metric))


def test_identical_train_and_split_seed_variables_are_refused():
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(seed_env={"train": "SEED", "split": "SEED"}))
