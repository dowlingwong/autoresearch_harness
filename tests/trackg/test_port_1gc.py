"""Port kit for the Track G T1 expansion candidate 4: node gc7rc_forestcover (1GC-7RC Task 5, Forest Cover).

Static checks always run.  The synthetic smoke needs torch in the interpreter running the tests; it uses random
data of the real shape (581,012 rows x 54 features, seven classes), never the task data, and a reduced step
budget.  The 1GC-7RC suite repository is not public; the node ports the public dataset/task semantics the paper
states (see the node's README.md and LICENCES.md), not suite code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODES = ["gc7rc_forestcover"]


@pytest.mark.parametrize("node", NODES)
def test_static(node):
    ps.check_static(node, NODES)


@pytest.mark.parametrize("node", NODES)
def test_manifest_complete(node):
    ps.check_manifest(node)


@pytest.mark.parametrize("node", NODES)
def test_frozen_path_edit_refused(node):
    ps.check_scope(node)


# Pairs of declared edits that change different settings, compose, and both act after each other (S1.1
# interaction runs): E1/E2 (learning rate) x E3 (loss/clipping), and E1/E2 x E4 (architecture).
INTERACTING = {"gc7rc_forestcover": {"E1+E3", "E2+E3", "E1+E4", "E2+E4"}}


@pytest.mark.parametrize("node", NODES)
def test_interacting_edits_compose(node):
    pairs = ps.check_composable(node)
    for pair in INTERACTING[node]:
        a, b = pair.split("+")
        assert pairs[f"{a}+{b}"] == "applied" and pairs[f"{b}+{a}"] == "applied", (pair, pairs)


def test_task_semantics_pinned():
    """The 1GC-7RC Task 5 semantics the port implements are the pinned public ones: the UCI zip/member hashes,
    the 581,012 x 54 table with seven classes, the stratified 80/10/10 split, the fixed step budget and no task
    data in the branch."""
    node = ps.node_dir("gc7rc_forestcover")
    prepare = (node / "prepare_data.py").read_text()
    parent = (node / "run_fixed_budget.py").read_text()
    pin = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
    member = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
    assert pin in prepare and pin in parent
    assert member in prepare
    assert "https://archive.ics.uci.edu/static/public/31/covertype.zip" in prepare
    assert "10.24432/C50K5N" in prepare
    assert "N_ROWS = 581_012" in parent and "N_FEATURES = 54" in parent and "N_CLASSES = 7" in parent
    assert "SPLIT_SEED = 42" in parent and "VAL_FRACTION = 0.1" in parent and "TEST_FRACTION = 0.1" in parent
    assert "(x.astype(np.float64) - mean) / std" in parent  # StandardScaler semantics, training rows only
    spec = ps.load("gc7rc_forestcover")
    assert spec.run_budget.kind == "steps" and spec.run_budget.value == 8000
    assert spec.metric_name == "val_accuracy" and spec.metric_direction == "maximize"
    assert spec.selection.test_scored and spec.selection.test_metric_name == "test_accuracy"
    licences = (node / "LICENCES.md").read_text()
    assert "CC BY 4.0" in licences and "never redistributed" in licences.lower()
    assert "404" in licences  # the suite repository's availability, disclosed
    train = (node / "train.py").read_text()
    assert "HIDDEN = (128, 256, 128)" in train and "DROPOUT = 0.3" in train
    assert "nn.BatchNorm1d" in train and "batch_size" in train
    data_like = [p for p in node.rglob("*") if p.suffix in {".csv", ".npy", ".npz"}]
    assert not data_like, f"task data committed: {data_like}"


# --------------------------------------------------------------------------- synthetic data

def forest_data(d: Path) -> dict:
    rng = np.random.RandomState(0)
    # random features of the real shape (581,012 rows x 54: 10 continuous + 44 binary) and seven classes;
    # no task data
    x = rng.standard_normal((581_012, 54)).astype(np.float32)
    x[:, 10:] = (x[:, 10:] > 0).astype(np.float32)
    y = rng.randint(0, 7, size=581_012).astype(np.int8)
    np.save(d / "X.npy", x)
    np.save(d / "y.npy", y)
    return {"shapes": "random 581012 x 54 features + seven classes"}


SMOKE = {
    "gc7rc_forestcover": dict(
        make=forest_data, files=["X.npy", "y.npy"], budget=12, modules=("torch", "pandas"), divergent=set(),
        broken=('    return F.cross_entropy(model(batch["x"]), batch["y"])\n',
                '    return F.cross_entropy(model(batch["x"]), batch["y"]) * float("nan")\n')),
}


@pytest.mark.parametrize("node", NODES)
def test_synthetic_smoke(node, tmp_path, monkeypatch):
    cfg = SMOKE[node]
    if not ps.have(*cfg["modules"]):
        pytest.skip(f"needs {cfg['modules']} in the test interpreter")
    # The run command calls ``python3``: use the test interpreter's directory first on PATH.
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    work = ps.smoke_copy(node, tmp_path, cfg["make"], cfg["files"])
    records, per_edit = ps.run_smoke(node, tmp_path, work, cfg["budget"], ps.broken_diff(node, *cfg["broken"]))
    ps.check_smoke(records, per_edit, cfg["divergent"])


@pytest.mark.parametrize("node", NODES)
def test_worker_logits_refuse_tensor_subclass(node):
    """The worker-side evaluation (ar_fixed_loop.logits_array) returns plain float32 logits and refuses a
    tensor subclass from predict_logits (exit 4)."""
    if not ps.have("torch"):
        pytest.skip("needs torch")
    ps.check_logits_array_refuses_subclass(node)


@pytest.mark.parametrize("node", NODES)
def test_parent_metric_helpers(node):
    """R1-HARNESS-ISOLATION: the parent computes the metric (ar_parent), from the worker's arrays."""
    ps.check_parent_metric_helpers(node)
