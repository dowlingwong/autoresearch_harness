"""Port kit U3, suite MLGym: nodes mlgym_cifar10, mlgym_fmnist, mlgym_houseprice, mlgym_breakout_minatar (under
third_party/mlgym, CC BY-NC 4.0; isolated per R1-HARNESS-ISOLATION), plus the R1-T1-NODE-EXPANSION candidate 1
node mlgym_titanic (MLGym titanicSurvival, CPU-only).

Static checks always run.  The synthetic smoke needs the node's runtime
packages in the interpreter running the tests (torch for the image nodes and
mlgym_titanic; pandas and scikit-learn for mlgym_houseprice; jax, flax, optax,
gymnax and tensorflow-probability for mlgym_breakout_minatar).  It uses random
data of the real shapes (a random table with a few House Prices column names; a
random table with the Titanic columns), never the task data, and a reduced step
budget.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODES = ["mlgym_cifar10", "mlgym_fmnist", "mlgym_houseprice", "mlgym_breakout_minatar", "mlgym_titanic"]


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
    "mlgym_titanic": {"E1+E3", "E1+E4", "E2+E3", "E2+E4", "E3+E4"},
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


def titanic_data(d: Path) -> dict:
    """Random tables with the Titanic task's columns (never the task data): train.csv with the Survived
    label, an unlabelled test.csv and a ground_truth.csv over the same PassengerIds."""
    import pandas as pd

    rng = np.random.RandomState(0)
    titles = ["Mr", "Mrs", "Miss", "Master", "Dr"]

    def table(n: int, first_id: int) -> pd.DataFrame:
        return pd.DataFrame({
            "PassengerId": np.arange(first_id, first_id + n),
            "Pclass": rng.randint(1, 4, n),
            "Name": [f"Test, {titles[i % len(titles)]}. Passenger {i}" for i in range(n)],
            "Sex": rng.choice(["male", "female"], n),
            "Age": np.where(rng.rand(n) < 0.2, np.nan, rng.uniform(1, 80, n)),
            "SibSp": rng.randint(0, 5, n),
            "Parch": rng.randint(0, 4, n),
            "Ticket": [f"T{i % 40}" for i in range(n)],
            "Fare": np.where(rng.rand(n) < 0.1, np.nan, rng.uniform(1, 200, n)),
            "Cabin": rng.choice([None, "C85", "B42"], n),
            "Embarked": rng.choice(["S", "C", "Q", None], n),
        })

    train = table(300, 1)
    train.insert(1, "Survived", rng.randint(0, 2, len(train)))
    test = table(120, 1001)
    truth = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": rng.randint(0, 2, len(test))})
    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    truth.to_csv(d / "ground_truth.csv", index=False)
    return {"shapes": "random Titanic-shaped tables (train 300, test/ground_truth 120)"}


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
    "mlgym_titanic": dict(make=titanic_data, files=["train.csv", "test.csv", "ground_truth.csv"], budget=30,
                          modules=("torch", "pandas"), divergent=set(), torch_node=True,
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
    assert manifest["source"]["md5"] == hashlib.md5(good.read_bytes()).hexdigest()
    # Kit review round 1, M1: a re-run accepts the same file only, and never a directory set up from another source
    assert run("--from-file", str(good)).returncode == 0
    other = tmp_path / "other.csv"
    frame.assign(SalePrice=frame["SalePrice"] + 1).to_csv(other, index=False)
    res = run("--from-file", str(other))
    assert res.returncode != 0 and "differs from the table already set up" in (res.stdout + res.stderr)
    ar_data = ps.load_node_module("mlgym_houseprice", "ar_data")
    old = root / "old"
    old.mkdir()
    shutil.copy(good, old / "houseprice.csv")
    ar_data.write_manifest(old, "mlgym_houseprice", ["houseprice.csv"],
                           {"mlgym": "data/regressionKaggleHousePrice/_helpers/train.csv", "sha256": digest})
    res = subprocess.run([sys.executable, str(work / "prepare_data.py"), "--out", str(old), "--from-file", str(bad)],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode != 0 and "not set up from the user's Kaggle download" in (res.stdout + res.stderr)


def test_houseprice_parent_requires_the_users_download_and_prints_its_hashes(tmp_path, monkeypatch):
    """Kit review round 1, M1 (R1-TRACKG-PORT-DECISIONS (2)): the frozen parent refuses a table whose manifest does
    not record the user's download (an MLGym copy set up by an older kit), and prints the sha256 and md5."""
    if not ps.have("pandas", "sklearn"):
        pytest.skip("needs pandas and scikit-learn")
    import hashlib
    work = ps.smoke_copy("mlgym_houseprice", tmp_path, house_data, ["houseprice.csv"])
    data = (work / "data").resolve()
    table = (data / "houseprice.csv").read_bytes()
    sha, md5 = hashlib.sha256(table).hexdigest(), hashlib.md5(table).hexdigest()
    ar_data = ps.load_node_module("mlgym_houseprice", "ar_data")
    env = dict(os.environ, AR_TRAIN_SEED="8001", AR_SPLIT_SEED="42")
    for source, want in (({"mlgym": "forwarded copy", "sha256": sha, "md5": md5}, 2),
                         ({"provided_by": "user download, passed with --from-file", "sha256": sha, "md5": md5}, 0)):
        (data / ar_data.MANIFEST_NAME).unlink()     # the smoke copy's own synthetic manifest, rewritten here
        ar_data.write_manifest(data, "mlgym_houseprice", ["houseprice.csv"], source)
        code, out, channel = ps.run_parent(work, ["run_fixed_budget.py", "--steps", "1"], env)
        assert code == want, (source, code, out[-2000:])
        if want == 0:
            assert f"AR_DATA houseprice_sha256={sha} md5={md5} " in out and "AR_METRIC val_rmse=" in channel
        else:
            assert "not set up from the user's Kaggle download" in out and not channel


def test_titanic_setup_pins_the_mlgym_commit_files():
    """Checklist F: the frozen setup downloads the three tables from MLGym's pinned commit (no user file, no
    --from-file), checks the three sha256 values pinned in the script before writing anything, and records them
    in the data manifest.  Nothing here downloads: the hashes are known from the pinned upstream commit."""
    if not ps.have("pandas"):
        pytest.skip("needs pandas")
    node = ps.node_dir("mlgym_titanic")
    sys.path.insert(0, str(node))
    try:
        prep = ps.load_node_module("mlgym_titanic", "prepare_data")
    finally:
        sys.path.pop(0)
    spec = ps.load("mlgym_titanic")
    assert 'python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_titanic"' in spec.setup_command
    assert "urllib" in (node / "prepare_data.py").read_text()
    assert prep.COMMIT == "9d40c1b5035202018cd7091fb4e83a9c68b377c0"
    assert prep.REPOSITORY == "https://github.com/facebookresearch/MLGym"
    assert prep.TASK == "titanicSurvival" and prep.DATASET == "titanic"
    assert sorted(prep.FILES) == ["ground_truth.csv", "test.csv", "train.csv"]
    assert set(prep.PINS) == set(prep.FILES) and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in prep.PINS.values())
    assert prep.BASE_URL.endswith(f"{prep.COMMIT}/data/titanic/data")
    wrong = {name: b"not the task table" for name in prep.FILES}
    with pytest.raises(SystemExit, match="differs from the pinned"):
        prep.check_files(wrong)

