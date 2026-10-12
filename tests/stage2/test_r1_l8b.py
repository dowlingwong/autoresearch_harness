"""L8b tests: noisy CPU nodes (digits_mlp, tabular_mlp) and mlp_synthetic seed injection."""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]
NOISY_NODES = ("digits_mlp", "tabular_mlp")


def _baseline_proposal(spec) -> ManagerProposal:
    return ManagerProposal(
        manager_mode="test",
        proposal_summary="baseline",
        proposal_rationale="",
        target_files=spec.editable_paths,
        objective="Run baseline; no change.",
    )


def _run(node: str, tmp: Path, *, train_seed=None, split_seed=None, objective=None) -> float:
    spec = load_registered_node(node, repo_root=ROOT)
    node_dir = tmp / f"{node}_{train_seed}_{split_seed}_{abs(hash(objective)) % 10_000}"
    shutil.copytree(ROOT / "nodes" / node, node_dir)
    worker = LocalWorker(node_root=node_dir, artifact_dir=node_dir / "_art",
                         train_seed=train_seed, split_seed=split_seed)
    proposal = _baseline_proposal(spec)
    if objective is not None:
        proposal = ManagerProposal(manager_mode="test", proposal_summary="edit", proposal_rationale="",
                                   target_files=spec.editable_paths, objective=objective)
    result = worker.run_trial(proposal, spec, 1)
    assert result.success, result.failure_message
    return result.parsed_metrics["val_score"]


class TestNoisyNodes(unittest.TestCase):
    def test_registered_with_bounded_symbols(self) -> None:
        for node in NOISY_NODES:
            spec = load_registered_node(node, repo_root=ROOT)
            self.assertEqual(spec.metric_name, "val_score")
            self.assertEqual(spec.metric_direction, "maximize")
            self.assertEqual(spec.editable_paths, ("train.py",))
            self.assertEqual(
                set(spec.editable_symbols),
                {"HIDDEN_UNITS", "ALPHA", "LEARNING_RATE_INIT", "BATCH_SIZE", "MAX_ITER"},
            )
            for seed_var in ("AR_TRAIN_SEED", "AR_SPLIT_SEED", "TRAIN_SEED", "SPLIT_SEED"):
                self.assertNotIn(seed_var, spec.editable_symbols)

    def test_same_seeds_reproduce_and_each_seed_moves_the_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for node in NOISY_NODES:
                base = _run(node, tmp)
                self.assertEqual(base, _run(node, tmp, train_seed=42, split_seed=42))
                self.assertNotEqual(base, _run(node, tmp, train_seed=7))
                self.assertNotEqual(base, _run(node, tmp, split_seed=7))

    def test_worker_edit_changes_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = _run("digits_mlp", tmp)
            edited = _run("digits_mlp", tmp, objective="Change MAX_ITER from 40 to 10")
            self.assertNotEqual(base, edited)


class TestMlpSyntheticSeeds(unittest.TestCase):
    def _load(self):
        path = ROOT / "nodes" / "mlp_synthetic" / "benchmark.py"
        spec = importlib.util.spec_from_file_location("mlp_synthetic_benchmark_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_defaults_reproduce_previous_constants(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("AR_TRAIN_SEED", "AR_SPLIT_SEED")}
        with mock.patch.dict(os.environ, env, clear=True):
            module = self._load()
        self.assertEqual(module.SPLIT_SEED, 123)
        self.assertEqual(module.TRAIN_SEED, 124)

    def test_env_overrides_are_independent(self) -> None:
        with mock.patch.dict(os.environ, {"AR_TRAIN_SEED": "5"}):
            os.environ.pop("AR_SPLIT_SEED", None)
            module = self._load()
        self.assertEqual((module.SPLIT_SEED, module.TRAIN_SEED), (123, 5))


if __name__ == "__main__":
    unittest.main()
