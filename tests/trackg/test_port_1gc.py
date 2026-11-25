"""Port kit for the Track G T1 expansion candidate 2: node gc7rc_etth1 (1GC-7RC Task 6, ETTh1 forecasting).

Static checks always run.  The synthetic smoke needs torch in the interpreter running the tests; it uses random
data of the real shape (17,420 rows x 7 features), never the task data, and a reduced step budget.  The 1GC-7RC
suite repository is not public; the node ports the public dataset/task semantics the paper states (see the
node's README.md and LICENCES.md), not suite code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODES = ["gc7rc_etth1"]


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
# interaction runs): E1/E2 (learning rate) x E3 (loss), and E1/E2 x E4 (architecture).
INTERACTING = {"gc7rc_etth1": {"E1+E3", "E2+E3", "E1+E4", "E2+E4"}}


@pytest.mark.parametrize("node", NODES)
def test_interacting_edits_compose(node):
    pairs = ps.check_composable(node)
    for pair in INTERACTING[node]:
        a, b = pair.split("+")
        assert pairs[f"{a}+{b}"] == "applied" and pairs[f"{b}+{a}"] == "applied", (pair, pairs)


def test_task_semantics_pinned():
    """The 1GC-7RC Task 6 semantics the port implements are the pinned public ones: the ETTh1 source hash,
    the 336 -> 96 windows and the LTSF 12/4/4 split, the fixed step budget, and no task data in the branch."""
    node = ps.node_dir("gc7rc_etth1")
    prepare = (node / "prepare_data.py").read_text()
    parent = (node / "run_fixed_budget.py").read_text()
    pin = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
    assert pin in prepare and pin in parent
    assert "1d16c8f4f943005d613b5bc962e9eeb06058cf07" in prepare
    assert "SEQ_LEN = 336" in parent and "HORIZON = 96" in parent
    assert "NUM_TRAIN = 12 * 30 * 24" in parent and "NUM_VAL = 4 * 30 * 24" in parent
    spec = ps.load("gc7rc_etth1")
    assert spec.run_budget.kind == "steps" and spec.run_budget.value == 2000
    assert spec.metric_name == "val_mse" and spec.metric_direction == "minimize"
    assert spec.selection.test_scored and spec.selection.test_metric_name == "test_mse"
    licences = (node / "LICENCES.md").read_text()
    assert "CC BY-ND 4.0" in licences and "not redistributed" in licences.lower()
    assert "404" in licences  # the suite repository's availability, disclosed
    data_like = [p for p in node.rglob("*") if p.suffix in {".csv", ".npy", ".npz"}]
    assert not data_like, f"task data committed: {data_like}"


# --------------------------------------------------------------------------- synthetic data

def etth1_data(d: Path) -> dict:
    rng = np.random.RandomState(0)
    # random walks of the real shape (17,420 rows x 7 features); no task data
    series = np.cumsum(rng.randn(17_420, 7).astype(np.float64) * 0.1, axis=0) + 10.0
    np.save(d / "series.npy", series.astype(np.float32))
    return {"shapes": "random walk, 17420 x 7"}


SMOKE = {
    "gc7rc_etth1": dict(make=etth1_data, files=["series.npy"], budget=12, modules=("torch",), divergent=set(),
                        broken=('    return F.mse_loss(model(batch["x"]), batch["y"])\n',
                                '    return F.mse_loss(model(batch["x"]), batch["y"]) * float("nan")\n')),
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
def test_worker_predictions_refuse_tensor_subclass(node):
    """The worker-side evaluation (ar_fixed_loop.predictions_array) returns plain float32 predictions and
    refuses a tensor subclass from predict (exit 4)."""
    if not ps.have("torch"):
        pytest.skip("needs torch")
    import subprocess
    code = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(node_dir(node))!r})\n"
        "import ar_fixed_loop as L\n"
        "class Liar(torch.Tensor):\n"
        "    pass\n"
        "class T:\n"
        "    @staticmethod\n"
        "    def predict(model, batch):\n"
        "        return torch.zeros(len(batch['x']), 3, 2).as_subclass(Liar)\n"
        "L.predictions_array(T, torch.nn.Linear(1, 1), lambda idx: {'x': torch.zeros(len(idx), 3, 2)},"
        " 4, 3, 2, {}, torch.device('cpu'))\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert res.returncode == 4 and "plain torch.Tensor" in res.stdout, (res.returncode, res.stdout[-500:],
                                                                        res.stderr[-800:])
    ok = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(node_dir(node))!r})\n"
        "import ar_fixed_loop as L\n"
        "class T:\n"
        "    @staticmethod\n"
        "    def predict(model, batch):\n"
        "        return torch.zeros(len(batch['x']), 3, 2)\n"
        "a = L.predictions_array(T, torch.nn.Linear(1, 1), lambda idx: {'x': torch.zeros(len(idx), 3, 2)},"
        " 5, 3, 2, {}, torch.device('cpu'))\n"
        "print('SHAPE', a.shape, a.dtype)\n")
    res = subprocess.run([sys.executable, "-c", ok], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and "SHAPE (5, 6) float32" in res.stdout, (res.returncode, res.stdout[-500:],
                                                                          res.stderr[-800:])


@pytest.mark.parametrize("node", NODES)
def test_parent_metric_helpers(node):
    """R1-HARNESS-ISOLATION: the parent computes the metric (ar_parent), from the worker's arrays."""
    ps.check_parent_metric_helpers(node)


def node_dir(node: str) -> Path:
    return ps.node_dir(node)
