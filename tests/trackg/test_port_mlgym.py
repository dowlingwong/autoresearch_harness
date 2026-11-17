"""Port kit U3, suite MLGym: nodes mlgym_cifar10, mlgym_fmnist, mlgym_houseprice, mlgym_breakout_minatar (under
third_party/mlgym, CC BY-NC 4.0; isolated per R1-HARNESS-ISOLATION).

Static checks always run.  The synthetic smoke needs the node's runtime
packages in the interpreter running the tests (torch for the image nodes;
pandas and scikit-learn for mlgym_houseprice; jax, flax, optax, gymnax and
tensorflow-probability for mlgym_breakout_minatar).  It uses random data of the
real shapes (a random table with a few House Prices column names), never the
task data, and a reduced step budget.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODES = ["mlgym_cifar10", "mlgym_fmnist", "mlgym_houseprice", "mlgym_breakout_minatar"]


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
# interaction runs).  Edits that set the same line (two learning rates, two alphas) are alternatives and
# conflict; House Prices E1/E2 (Ridge alpha) are dead code after E4 (gradient boosting) and are left out.
INTERACTING = {
    "mlgym_cifar10": {"E1+E3", "E1+E4", "E2+E3", "E2+E4", "E3+E4"},
    "mlgym_fmnist": {"E1+E3", "E1+E4", "E2+E3", "E2+E4", "E3+E4"},
    "mlgym_houseprice": {"E1+E3", "E2+E3", "E3+E4"},
    "mlgym_breakout_minatar": {"E1+E3", "E2+E3", "E3+E4"},
}


@pytest.mark.parametrize("node", NODES)
def test_interacting_edits_compose(node):
    pairs = ps.check_composable(node)
    for pair in INTERACTING[node]:
        a, b = pair.split("+")
        assert pairs[f"{a}+{b}"] == "applied" and pairs[f"{b}+{a}"] == "applied", (pair, pairs)


def test_breakout_config_leaves_wrapper_keys_out():
    text = (ps.node_dir("mlgym_breakout_minatar") / "config.yaml").read_text()
    for key in ("num_train_steps", "evaluate_every_epochs", "num_test_rollouts"):
        assert f"  {key}:" not in text


# --------------------------------------------------------------------------- synthetic data

def image_data(shape_train, shape_test):
    def make(d: Path) -> dict:
        rng = np.random.RandomState(0)
        np.save(d / "train_x.npy", rng.randint(0, 256, shape_train).astype(np.uint8))
        np.save(d / "train_y.npy", rng.randint(0, 10, shape_train[0]).astype(np.int64))
        np.save(d / "test_x.npy", rng.randint(0, 256, shape_test).astype(np.uint8))
        np.save(d / "test_y.npy", rng.randint(0, 10, shape_test[0]).astype(np.int64))
        return {"shapes": "random uint8 images"}
    return make


def house_data(d: Path) -> dict:
    import pandas as pd

    rng = np.random.RandomState(0)
    n = 300
    area = rng.randint(500, 4000, n)
    frame = pd.DataFrame({
        "Id": np.arange(1, n + 1),
        "LotArea": rng.randint(1000, 20000, n),
        "OverallQual": rng.randint(1, 11, n),
        "GrLivArea": area,
        "YearBuilt": rng.randint(1900, 2010, n),
        "LotFrontage": np.where(rng.rand(n) < 0.2, np.nan, rng.randint(20, 150, n).astype(float)),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards"], n),
        "MSZoning": np.where(rng.rand(n) < 0.05, None, rng.choice(["RL", "RM", "FV"], n)),
        "SalePrice": (area * 80 + rng.randint(0, 50000, n)).astype(np.int64),
    })
    frame.to_csv(d / "houseprice.csv", index=False)
    return {"shapes": "random table with a few House Prices column names"}


SMOKE = {
    "mlgym_cifar10": dict(make=image_data((5400, 32, 32, 3), (300, 32, 32, 3)),
                          files=["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"], budget=30,
                          modules=("torch",), divergent=set(), torch_node=True,
                          broken=('    return F.cross_entropy(model(preprocess(batch["x"])), batch["y"])\n',
                                  '    return F.cross_entropy(model(preprocess(batch["x"])), batch["y"]) * float("nan")\n')),
    "mlgym_fmnist": dict(make=image_data((6400, 28, 28), (300, 28, 28)),
                         files=["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"], budget=30,
                         modules=("torch",), divergent=set(), torch_node=True,
                         broken=('    return F.cross_entropy(model(preprocess(batch["x"])), batch["y"])\n',
                                 '    return F.cross_entropy(model(preprocess(batch["x"])), batch["y"]) * float("nan")\n')),
    "mlgym_houseprice": dict(make=house_data, files=["houseprice.csv"], budget=1,
                             modules=("pandas", "sklearn"), divergent=set(), torch_node=False,
                             broken=("def inverse_transform_target(z):\n    return z\n",
                                     "def inverse_transform_target(z):\n    return z * float(\"nan\")\n")),
    "mlgym_breakout_minatar": dict(make=None, files=[], budget=17000,
                                   modules=("jax", "flax", "optax", "gymnax", "tensorflow_probability", "dotmap"),
                                   divergent=set(), torch_node=False,
                                   broken=("        pi = tfp.distributions.Categorical(logits=logits)\n",
                                           "        pi = tfp.distributions.Categorical(logits=logits * jnp.nan)\n",
                                           "networks.py")),
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
    ps.check_smoke(records, per_edit, cfg["divergent"], torch_node=cfg["torch_node"])


@pytest.mark.parametrize("node", ["mlgym_cifar10", "mlgym_fmnist"])
def test_worker_logits_refuse_tensor_subclass(node):
    """The worker-side evaluation returns plain float32 logits and refuses a tensor subclass from predict_logits."""
    if not ps.have("torch"):
        pytest.skip("needs torch")
    ps.check_logits_array_refuses_subclass(node)


@pytest.mark.parametrize("node", NODES)
def test_parent_metric_helpers(node):
    """R1-HARNESS-ISOLATION: the parent computes the metric (ar_parent), from the worker's arrays."""
    ps.check_parent_metric_helpers(node)


def test_mlgym_nodes_live_in_the_cc_by_nc_subtree():
    """R1-TRACKG-PORT-DECISIONS (1): the MLGym nodes and specs are under third_party/mlgym with its CC BY-NC 4.0
    LICENSE, and the root README and LICENSE state the scope."""
    sub = ps.ROOT / "third_party" / "mlgym"
    assert (sub / "LICENSE").read_text().startswith("Attribution-NonCommercial 4.0 International")
    for node in NODES:
        assert ps.node_dir(node) == sub / "nodes" / node and ps.spec_path(node).is_relative_to(sub)
        assert not (ps.ROOT / "nodes" / node).exists() and not (ps.ROOT / "configs" / "trackg" / "nodes" / f"{node}.yaml").exists()
    assert "third_party/mlgym/" in (ps.ROOT / "README.md").read_text().split("\n\n")[1]
    assert "third_party/mlgym/" in (ps.ROOT / "LICENSE").read_text()


def test_houseprice_data_only_from_the_users_kaggle_file(tmp_path, monkeypatch):
    """R1-TRACKG-PORT-DECISIONS (2): setup downloads nothing, needs --from-file, checks the table's shape and records
    its sha256; the spec's setup_command passes the user's file."""
    if not ps.have("pandas"):
        pytest.skip("needs pandas")
    import hashlib
    import json
    import subprocess

    node = ps.node_dir("mlgym_houseprice")
    text = (node / "prepare_data.py").read_text()
    assert "urllib" not in text and "raw.githubusercontent" not in text
    assert '--from-file "$AR_HOUSEPRICE_TRAIN_CSV"' in ps.load("mlgym_houseprice").setup_command
    work = tmp_path / "node"
    import shutil
    shutil.copytree(node, work, ignore=shutil.ignore_patterns("__pycache__", "data"))
    root = tmp_path / "root"
    root.mkdir()
    env = {**os.environ, "AR_TRACKG_DATA_ROOT": str(root)}
    run = lambda *a: subprocess.run([sys.executable, str(work / "prepare_data.py"), "--out", str(root / "hp"), *a],
                                    capture_output=True, text=True, env=env, timeout=120)
    res = run()
    assert res.returncode != 0 and "--from-file" in (res.stdout + res.stderr)
    bad = tmp_path / "bad.csv"
    bad.write_text("Id,SalePrice\n1,2\n")
    res = run("--from-file", str(bad))
    assert res.returncode != 0 and "not Kaggle" in (res.stdout + res.stderr) and not (root / "hp").exists()
    import numpy as np
    import pandas as pd
    cols = {f"c{i}": np.arange(1460) for i in range(79)}
    frame = pd.DataFrame({"Id": np.arange(1, 1461), **cols, "SalePrice": np.arange(1460) * 10 + 1000})
    good = tmp_path / "train.csv"
    frame.to_csv(good, index=False)
    res = run("--from-file", str(good))
    assert res.returncode == 0, res.stdout + res.stderr
    manifest = json.loads((root / "hp" / "data_manifest.json").read_text())
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    assert manifest["source"]["sha256"] == digest and manifest["files"]["houseprice.csv"]["sha256"] == digest
    assert "kaggle" in manifest["source"] and digest in res.stdout
