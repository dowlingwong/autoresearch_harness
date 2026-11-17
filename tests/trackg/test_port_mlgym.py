"""Port kit U3, suite MLGym: nodes mlgym_cifar10, mlgym_fmnist, mlgym_houseprice, mlgym_breakout_minatar.

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
