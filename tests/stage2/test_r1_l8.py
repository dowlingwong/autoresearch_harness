"""L8 tests: OpenML template nodes, held_out flag, MLAgentBench seed injection."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.registry import load_registered_node
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]

NEW_OPENML = {
    "openml_breast_w": 15,
    "openml_credit_approval": 29,
    "openml_diabetes": 37,
    "openml_ionosphere": 59,
    "openml_wdbc": 1510,
}


class TestOpenmlPool(unittest.TestCase):
    def test_new_nodes_registered_and_configured(self) -> None:
        for name, dataset_id in NEW_OPENML.items():
            spec = load_registered_node(name, repo_root=ROOT)
            self.assertEqual(spec.metric_name, "val_auc")
            self.assertEqual(spec.metric_direction, "maximize")
            self.assertEqual(spec.editable_paths, ("config.yaml",))
            self.assertFalse(spec.held_out)
            self.assertIn("data_cache", spec.branch_state_paths)

    def test_dataset_json_matches_registry(self) -> None:
        for name, dataset_id in NEW_OPENML.items():
            meta = json.loads((ROOT / "nodes" / name / "dataset.json").read_text())
            self.assertEqual(meta["node_name"], name)
            self.assertEqual(int(meta["dataset_id"]), dataset_id)
            spec = load_registered_node(name, repo_root=ROOT)
            self.assertEqual(spec.name, name)

    def test_held_out_flag_round_trips(self) -> None:
        payload = {
            "name": "held_out_node",
            "description": "test",
            "editable_paths": ["config.yaml"],
            "frozen_paths": [],
            "setup_command": "",
            "run_command": "python3 train.py",
            "metric_name": "val_auc",
            "metric_direction": "maximize",
            "metric_parser": "x:y",
            "acceptance_rule": "candidate_metric > current_best_metric",
            "validity_checks": ["metric_present"],
            "default_budget": {"trials": 1},
            "held_out": True,
        }
        spec = NodeSpec.from_mapping(payload)
        self.assertTrue(spec.held_out)
        self.assertTrue(spec.to_dict()["held_out"])


class TestMLAgentBenchSeed(unittest.TestCase):
    def test_seed_env_reaches_node(self) -> None:
        spec = load_registered_node("mlagentbench_vectorization", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            node = tmp / "node"
            shutil.copytree(ROOT / "nodes" / "mlagentbench_vectorization", node)
            worker = LocalWorker(node_root=node, artifact_dir=tmp / "art", split_seed=123)
            proposal = ManagerProposal(
                manager_mode="test",
                proposal_summary="base",
                proposal_rationale="",
                target_files=spec.editable_paths,
                objective="Run baseline; no change.",
            )
            result = worker.run_trial(proposal, spec, 1)
            self.assertTrue(result.success, result.failure_message)
            metrics = json.loads((node / "artifacts" / "metrics.json").read_text())
        self.assertEqual(metrics["seed"], 123)


if __name__ == "__main__":
    unittest.main()
