"""G.1 step 2 support: 3-5 representative edits declared as hashed diffs before measurement."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch.trackg.declared_edits import DeclaredEditsError, declare_edits, write_manifest
from autoresearch.trackg.node_template import TrackGNodeSpec

from .fixtures import make_node, spec_mapping, write_spec

ROOT = Path(__file__).resolve().parents[2]


def edit(path: Path, old: str, new: str, line: int, file="train.py") -> Path:
    path.write_text(f"--- a/{file}\n+++ b/{file}\n@@ -{line},1 +{line},1 @@\n-{old}\n+{new}\n")
    return path


def three_edits(tmp_path):
    d = tmp_path / "edits"
    d.mkdir()
    return [edit(d / "E1.diff", "LEARNING_RATE = 0.1", "LEARNING_RATE = 0.2", 8),
            edit(d / "E2.diff", "LEARNING_RATE = 0.1", "LEARNING_RATE = 0.05", 8),
            edit(d / "E3.diff", "WIDTH = 64", "WIDTH = 128", 1, file="model/net.py")]


def test_manifest_hashes_each_edit_and_measures_nothing(tmp_path):
    node = make_node(tmp_path)
    spec = TrackGNodeSpec.from_mapping(spec_mapping())
    edits = three_edits(tmp_path)
    before = (node / "train.py").read_bytes()
    manifest = declare_edits(spec, node, edits)
    assert manifest["schema"] == "trackg_representative_edits/v1" and manifest["measured"] is False
    assert [e["id"] for e in manifest["edits"]] == ["E1", "E2", "E3"]
    assert manifest["edits"][0]["sha256"] == hashlib.sha256(edits[0].read_bytes()).hexdigest()
    assert (node / "train.py").read_bytes() == before
    out = tmp_path / "manifest.json"
    digest = write_manifest(manifest, out)
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_manifest(manifest, out)


def test_bad_edit_sets_are_refused(tmp_path):
    node = make_node(tmp_path)
    spec = TrackGNodeSpec.from_mapping(spec_mapping())
    edits = three_edits(tmp_path)
    with pytest.raises(DeclaredEditsError):
        declare_edits(spec, node, edits[:2])
    frozen = tmp_path / "edits" / "F.diff"
    frozen.write_text("--- a/model/metric.py\n+++ b/model/metric.py\n@@ -1,1 +1,1 @@\n-NAME = 'acc'\n+NAME = 'x'\n")
    with pytest.raises(DeclaredEditsError, match="refused"):
        declare_edits(spec, node, edits[:2] + [frozen])
    inert = edit(tmp_path / "edits" / "I.diff", "WIDTH = 64", "WIDTH = (64)", 1, file="model/net.py")
    with pytest.raises(DeclaredEditsError, match="inert"):
        declare_edits(spec, node, edits[:2] + [inert])
    dup = edit(tmp_path / "edits" / "D.diff", "LEARNING_RATE = 0.1", "LEARNING_RATE = 0.2  ", 8)
    with pytest.raises(DeclaredEditsError, match="duplicate"):
        declare_edits(spec, node, edits + [dup])


def test_script_end_to_end(tmp_path):
    node = make_node(tmp_path)
    spec_path = write_spec(tmp_path / "node.yaml")
    edits = three_edits(tmp_path)
    out = tmp_path / "manifest.json"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "trackg_declare_edits.py"), "--spec",
                           str(spec_path), "--node-dir", str(node), "--out", str(out), *map(str, edits)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["node"] == "fake_task"
    assert "manifest" in proc.stdout and "E3 E3.diff sha256" in proc.stdout
