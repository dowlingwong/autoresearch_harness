"""M10 step 1 (G.2): diff application through the existing scope validator; inert and
duplicate detection; provenance."""
from __future__ import annotations

import os

import pytest

from autoresearch.trackg.diff_support import (
    EvaluatedStates,
    apply_diff,
    code_state_hashes,
    parse_unified_diff,
    DiffRefused,
)
from autoresearch.trackg.node_template import TrackGNodeSpec

from .fixtures import make_node, spec_mapping

LR = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"


@pytest.fixture()
def env(tmp_path):
    node = make_node(tmp_path)
    base = TrackGNodeSpec.from_mapping(spec_mapping()).to_base_node_spec()
    return node, base


def snapshot(node):
    return {p.relative_to(node).as_posix(): p.read_bytes() for p in sorted(node.rglob("*")) if p.is_file()}


def test_applied_diff_changes_the_copy_and_records_provenance(env):
    node, base = env
    incumbent = code_state_hashes(node, base)
    result = apply_diff(node, LR, base)
    assert result.status == "applied"
    p = result.provenance
    assert p["schema"] == "manager_proposal/v2-diff"
    assert p["diff_text"] == LR and len(p["diff_sha256"]) == 64 and len(p["normalised_diff_sha256"]) == 64
    assert p["files_touched"] == ["train.py"]
    assert p["incumbent_code_state_sha256"] == incumbent[0]
    assert p["code_state_sha256"] == code_state_hashes(node, base)[0] != incumbent[0]
    assert p["hunks"] == [{"file": "train.py", "old_start": 8, "applied_at": 8, "offset": 0}]
    assert "LEARNING_RATE = 0.2" in (node / "train.py").read_text()


@pytest.mark.parametrize("path, reason", [
    ("evaluate.py", "outside editable_paths"),
    ("config.yaml", "outside editable_paths"),
    ("model/metric.py", "matches frozen_paths"),
])
def test_hunk_outside_editable_or_inside_frozen_is_refused_and_nothing_is_written(env, path, reason):
    node, base = env
    old = (node / path).read_text().splitlines()[0]
    bad = f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n-{old}\n+# replaced\n"
    before = snapshot(node)
    # one good file plus one bad file: the whole diff is refused
    result = apply_diff(node, LR + bad, base)
    assert result.status == "refused" and result.provenance["refusal"] == "scope_violation"
    assert any(reason in v for v in result.provenance["scope"]["violations"])
    assert snapshot(node) == before


@pytest.mark.parametrize("diff, reason", [
    ("--- a/../x.py\n+++ b/../x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n", "unsafe_path"),
    ("--- /etc/passwd\n+++ /etc/passwd\n@@ -1,1 +1,1 @@\n-a\n+b\n", "unsafe_path"),
    ("--- a/train.py\n+++ b/model/net.py\n@@ -1,1 +1,1 @@\n-a\n+b\n", "unsupported_diff_feature"),
    ("diff --git a/train.py b/t2.py\nsimilarity index 90%\nrename from train.py\nrename to t2.py\n",
     "unsupported_diff_feature"),
    ("diff --git a/train.py b/train.py\nGIT binary patch\nliteral 0\n", "unsupported_diff_feature"),
    ("--- a/train.py\n+++ b/train.py\n@@ -8,2 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n",
     "malformed_hunk"),
    ("some prose before the diff\n" + LR, "malformed_diff"),
    ("--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.3\n+LEARNING_RATE = 0.2\n",
     "context_mismatch"),
    ("--- a/missing.py\n+++ b/missing.py\n@@ -1,1 +1,1 @@\n-a\n+b\n", "scope_violation"),
    ("--- a/model/new.py\n+++ b/model/new.py\n@@ -1,1 +1,1 @@\n-a\n+b\n", "context_mismatch"),
])
def test_malformed_unsafe_or_mismatched_diffs_are_refused(env, diff, reason):
    node, base = env
    before = snapshot(node)
    result = apply_diff(node, diff, base)
    assert result.status == "refused", result.provenance
    assert result.provenance["refusal"] == reason, result.provenance["refusal_detail"]
    assert snapshot(node) == before


def test_symlink_escape_is_refused(env, tmp_path):
    node, base = env
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text("a\n")
    os.symlink(outside, node / "model" / "link")
    diff = "--- a/model/link/x.py\n+++ b/model/link/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    result = apply_diff(node, diff, base)
    assert result.provenance["refusal"] == "unsafe_path"
    assert (outside / "x.py").read_text() == "a\n"


def test_wrong_line_number_applies_only_at_a_unique_exact_match(env):
    node, base = env
    moved = LR.replace("@@ -8,1 +8,1 @@", "@@ -20,1 +20,1 @@")
    result = apply_diff(node, moved, base)
    assert result.status == "applied" and result.provenance["hunks"][0]["offset"] == -12
    (node / "model" / "net.py").write_text("A = 1\nA = 1\n")
    amb = "--- a/model/net.py\n+++ b/model/net.py\n@@ -5,1 +5,1 @@\n-A = 1\n+A = 2\n"
    assert apply_diff(node, amb, base).provenance["refusal"] == "ambiguous_hunk"


@pytest.mark.parametrize("diff, reason", [
    # whitespace only
    ("--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE =   0.1   \n",
     "normalised_equal_ast_equal_to_incumbent"),
    # comment only (python)
    ("--- a/model/net.py\n+++ b/model/net.py\n@@ -1,3 +1,3 @@\n WIDTH = 64\n-# width of the hidden layer\n"
     "+# hidden width\n DEPTH = 2\n", "normalised_equal_ast_equal_to_incumbent"),
    # AST-equal rewrite
    ("--- a/model/net.py\n+++ b/model/net.py\n@@ -1,1 +1,1 @@\n-WIDTH = 64\n+WIDTH = (64)\n",
     "normalised_equal_ast_equal_to_incumbent"),
])
def test_inert_diffs_are_detected_and_not_written(env, diff, reason):
    node, base = env
    before = snapshot(node)
    result = apply_diff(node, diff, base)
    assert result.status == "inert" and result.provenance["inert"] is True
    assert result.provenance["inert_reason"] == reason
    assert result.provenance["code_state_sha256"] == result.provenance["incumbent_code_state_sha256"]
    assert snapshot(node) == before


def test_comment_only_change_in_a_non_python_editable_file_is_inert(tmp_path):
    node = make_node(tmp_path)
    (node / "model" / "cfg.yaml").write_text("width: 64\n# note\n")
    base = TrackGNodeSpec.from_mapping(spec_mapping()).to_base_node_spec()
    diff = "--- a/model/cfg.yaml\n+++ b/model/cfg.yaml\n@@ -1,2 +1,3 @@\n width: 64\n-# note\n+# other note\n+\n"
    assert apply_diff(node, diff, base).provenance["inert_reason"] == "normalised_equal"
    real = "--- a/model/cfg.yaml\n+++ b/model/cfg.yaml\n@@ -1,1 +1,1 @@\n-width: 64\n+width: 128\n"
    assert apply_diff(node, real, base).status == "applied"


def test_empty_diff_is_inert(env):
    node, base = env
    result = apply_diff(node, "", base)
    assert (result.status, result.provenance["inert_reason"]) == ("inert", "empty_diff")


def test_duplicates_by_normalised_diff_and_by_resulting_code_state(env, tmp_path):
    node, base = env
    evaluated = EvaluatedStates()
    evaluated.register("baseline", code_state_sha256=code_state_hashes(node, base)[0])
    # same diff text modulo trailing whitespace and line numbers -> same normalised diff hash
    first = apply_diff(node, LR, base, evaluated=evaluated, write=False)
    assert first.status == "applied"
    evaluated.register("t1", code_state_sha256=first.provenance["code_state_sha256"],
                       normalised_diff_sha256=first.provenance["normalised_diff_sha256"])
    variant = LR.replace("@@ -8,1 +8,1 @@", "@@ -9,1 +9,1 @@").replace("+LEARNING_RATE = 0.2", "+LEARNING_RATE = 0.2  ")
    dup = apply_diff(node, variant, base, evaluated=evaluated)
    assert (dup.status, dup.provenance["duplicate_kind"], dup.provenance["duplicate_of"]) == (
        "duplicate", "normalised_diff", "t1")
    # a different diff text that reaches the same code state (inline comment) -> code-state duplicate
    other = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2  # faster\n"
    dup2 = apply_diff(node, other, base, evaluated=evaluated)
    assert (dup2.status, dup2.provenance["duplicate_kind"], dup2.provenance["duplicate_of"]) == (
        "duplicate", "code_state", "t1")
    assert "LEARNING_RATE = 0.1" in (node / "train.py").read_text()
    # round trip of the registry
    assert EvaluatedStates.from_dict(evaluated.to_dict()).check(
        normalised_diff_sha256=None, code_state_sha256=first.provenance["code_state_sha256"])["duplicate"]


def test_create_and_delete_inside_editable_paths(env):
    node, base = env
    create = "--- /dev/null\n+++ b/model/extra.py\n@@ -0,0 +1,2 @@\n+def f():\n+    return 1\n"
    assert apply_diff(node, create, base).status == "applied"
    assert (node / "model" / "extra.py").read_text() == "def f():\n    return 1\n"
    delete = "--- a/model/extra.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-def f():\n-    return 1\n"
    assert apply_diff(node, delete, base).status == "applied"
    assert not (node / "model" / "extra.py").exists()
    frozen_create = "--- /dev/null\n+++ b/data/new.csv\n@@ -0,0 +1,1 @@\n+x\n"
    assert apply_diff(node, frozen_create, base).provenance["refusal"] == "scope_violation"


def test_no_newline_at_end_of_file_marker(env):
    node, base = env
    diff = ("--- a/model/net.py\n+++ b/model/net.py\n@@ -3,1 +3,1 @@\n-DEPTH = 2\n+DEPTH = 3\n"
            "\\ No newline at end of file\n")
    assert apply_diff(node, diff, base).status == "applied"
    assert (node / "model" / "net.py").read_text().endswith("DEPTH = 3")


def test_code_state_hash_covers_editable_files_only(env):
    node, base = env
    h0 = code_state_hashes(node, base)
    (node / "evaluate.py").write_text("changed\n")
    (node / "model" / "metric.py").write_text("changed\n")
    (node / "model" / "__pycache__").mkdir()
    (node / "model" / "__pycache__" / "net.cpython-313.pyc").write_bytes(b"\0")
    assert code_state_hashes(node, base) == h0
    (node / "model" / "net.py").write_text("WIDTH = 64\nDEPTH = 2\n")  # comment removed only
    h1 = code_state_hashes(node, base)
    assert h1[0] == h0[0] and h1[1] != h0[1]


def test_parse_rejects_a_file_listed_twice():
    with pytest.raises(DiffRefused):
        parse_unified_diff(LR + LR)


def test_symlinked_directory_inside_editable_paths_cannot_reach_a_frozen_path(env):
    node, base = env
    os.symlink(node / "data", node / "model" / "d")  # model/ is editable, data/ is frozen
    diff = "--- a/model/d/train.csv\n+++ b/model/d/train.csv\n@@ -1,2 +1,2 @@\n x,y\n-1,0\n+1,1\n"
    result = apply_diff(node, diff, base)
    assert result.provenance["refusal"] == "unsafe_path"
    assert (node / "data" / "train.csv").read_text() == "x,y\n1,0\n"
