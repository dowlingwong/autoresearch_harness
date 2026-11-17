"""Port kit U3, suite MLAgentBench: nodes mlab_cifar10, mlab_imdb, mlab_ogbn_arxiv.

Static checks always run.  The synthetic smoke needs torch (and transformers
for mlab_imdb) in the interpreter running the tests; it uses random data of
the real shapes, never the task data, and a reduced step budget.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODES = ["mlab_cifar10", "mlab_imdb", "mlab_ogbn_arxiv"]


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
# interaction runs).  Edits that set the same line (two learning rates) are alternatives and conflict; pairs
# where one edit makes the other dead code (cifar10 E3's Adam ignores E1's / E4's learning rate; ogbn E4's
# GraphSAGE ignores E3's sizes) compose but do not interact, and are left out.
INTERACTING = {
    "mlab_cifar10": {"E1+E2", "E2+E3", "E2+E4"},
    "mlab_imdb": {"E1+E3", "E1+E4", "E2+E3", "E2+E4", "E3+E4"},
    "mlab_ogbn_arxiv": {"E1+E3", "E2+E3"},
}


@pytest.mark.parametrize("node", NODES)
def test_interacting_edits_compose(node):
    pairs = ps.check_composable(node)
    for pair in INTERACTING[node]:
        a, b = pair.split("+")
        assert pairs[f"{a}+{b}"] == "applied" and pairs[f"{b}+{a}"] == "applied", (pair, pairs)


# --------------------------------------------------------------------------- synthetic data

def cifar_data(d: Path) -> dict:
    rng = np.random.RandomState(0)
    np.save(d / "train_x.npy", rng.randint(0, 256, (5400, 32, 32, 3)).astype(np.uint8))
    np.save(d / "train_y.npy", rng.randint(0, 10, 5400).astype(np.int64))
    np.save(d / "test_x.npy", rng.randint(0, 256, (300, 32, 32, 3)).astype(np.uint8))
    np.save(d / "test_y.npy", rng.randint(0, 10, 300).astype(np.int64))
    return {"shapes": "random uint8 images"}


def imdb_data(d: Path) -> dict:
    import torch
    from transformers import DistilBertConfig, DistilBertModel

    rng = np.random.RandomState(0)
    for split, n in (("train", 2600), ("test", 200)):
        np.save(d / f"{split}_ids.npy", rng.randint(0, 1000, (n, 512)).astype(np.int32))
        np.save(d / f"{split}_mask.npy", np.ones((n, 512), dtype=np.uint8))
        np.save(d / f"{split}_y.npy", rng.randint(0, 2, n).astype(np.int64))
    config = DistilBertConfig(vocab_size=1000, dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                              max_position_embeddings=512)
    torch.manual_seed(0)  # the tiny model's weights, so the smoke table reproduces
    DistilBertModel(config).save_pretrained(d / "pretrained")
    return {"shapes": "random token ids; tiny random DistilBERT"}


def ogbn_data(d: Path) -> dict:
    rng = np.random.RandomState(0)
    n = 2000
    np.save(d / "x.npy", rng.randn(n, 128).astype(np.float32))
    np.save(d / "y.npy", rng.randint(0, 40, n).astype(np.int64))
    np.save(d / "edge_index.npy", rng.randint(0, n, (2, 8000)).astype(np.int64))
    perm = rng.permutation(n)
    np.save(d / "train_idx.npy", np.sort(perm[:1200]).astype(np.int64))
    np.save(d / "valid_idx.npy", np.sort(perm[1200:1600]).astype(np.int64))
    np.save(d / "test_idx.npy", np.sort(perm[1600:]).astype(np.int64))
    return {"shapes": "random graph"}


SMOKE = {
    "mlab_cifar10": dict(make=cifar_data, files=["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"],
                         budget=30, modules=("torch",), divergent={"E4"},
                         broken=('    return F.cross_entropy(logits, batch["y"])\n',
                                 '    return F.cross_entropy(logits, batch["y"]) * float("nan")\n')),
    "mlab_imdb": dict(make=imdb_data,
                      files=["train_ids.npy", "train_mask.npy", "train_y.npy", "test_ids.npy", "test_mask.npy",
                             "test_y.npy", "pretrained/config.json", "pretrained/model.safetensors"],
                      budget=6, modules=("torch", "transformers"), divergent=set(),
                      broken=('    return F.cross_entropy(logits.float(), batch["y"])\n',
                              '    return F.cross_entropy(logits.float(), batch["y"]) * float("nan")\n')),
    "mlab_ogbn_arxiv": dict(make=ogbn_data,
                            files=["x.npy", "y.npy", "edge_index.npy", "train_idx.npy", "valid_idx.npy",
                                   "test_idx.npy"],
                            budget=20, modules=("torch",), divergent=set(),
                            broken=('    return F.nll_loss(model(batch["idx"]), batch["y"])\n',
                                    '    return F.nll_loss(model(batch["idx"]), batch["y"]) * float("nan")\n')),
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
    """The worker-side evaluation returns plain float32 logits and refuses a tensor subclass from predict_logits."""
    if not ps.have("torch"):
        pytest.skip("needs torch")
    ps.check_logits_array_refuses_subclass(node)


@pytest.mark.parametrize("node", NODES)
def test_parent_metric_helpers(node):
    """R1-HARNESS-ISOLATION: the parent computes the metric (ar_parent), from the worker's arrays."""
    ps.check_parent_metric_helpers(node)


def test_imdb_baseline_is_frozen_and_disclosed():
    """R1-TRACKG-PORT-DECISIONS (3): the port-written imdb baseline is frozen at the sha256 recorded in LICENCES.md
    (a change needs a new record and a review), and the disclosure wording is recorded."""
    import hashlib
    import re
    node = ps.node_dir("mlab_imdb")
    text = (node / "LICENCES.md").read_text()
    recorded = re.search(r"^BASELINE_SHA256 = ([0-9a-f]{64})$", text, re.MULTILINE).group(1)
    assert hashlib.sha256((node / "train.py").read_bytes()).hexdigest() == recorded
    assert '"baseline written by us"' in text and "census funnel" in text and "appendix" in text
    assert "R1-TRACKG-PORT-DECISIONS (3)" in (node / "run_fixed_budget.py").read_text()
