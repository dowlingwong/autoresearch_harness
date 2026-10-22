"""L8 tests: OpenML template nodes, held_out flag, MLAgentBench seed injection."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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
# D21 pool extension (22 Oct 2026, paper repo PLAN §10 "D21 pool"): numeric-only
# OpenML datasets built from the same template.  AI-assisted (Claude); ENGINEERING_ONLY.
POOL_22OCT = {
    "openml_sonar": (40, "sonar", "Class"),
    "openml_blood_transfusion": (1464, "blood-transfusion-service-center", "Class"),
    "openml_qsar_biodeg": (1494, "qsar-biodeg", "Class"),
}
TEMPLATE_FILES = ("benchmark.py", "train.py", "config.yaml", "README.md")


def _load_benchmark(node_dir: Path):
    spec = importlib.util.spec_from_file_location(f"bench_{node_dir.name}", node_dir / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class TestOpenmlPool22Oct(unittest.TestCase):
    def test_nodes_registered_and_configured(self) -> None:
        for name, (dataset_id, dataset_name, target) in POOL_22OCT.items():
            with self.subTest(node=name):
                spec = load_registered_node(name, repo_root=ROOT)
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.metric_name, "val_auc")
                self.assertEqual(spec.metric_direction, "maximize")
                self.assertEqual(spec.editable_paths, ("config.yaml",))
                self.assertFalse(spec.held_out)
                self.assertIn("data_cache", spec.branch_state_paths)
                self.assertEqual(spec.metric_parser, "autoresearch.nodes.openml_credit_g.metric_parser:parse_val_auc")
                raw = json.loads((ROOT / "configs" / "nodes" / f"{name}.yaml").read_text())
                self.assertEqual(raw["dataset"]["dataset_id"], dataset_id)
                self.assertEqual(raw["dataset"]["dataset_name"], dataset_name)
                self.assertEqual(raw["dataset"]["target"], target)
                # numeric-only datasets: the categorical imputer never reaches the pipeline (D25)
                self.assertEqual(raw["active_keys"]["dataset_excluded"], ["imputer"])

    def test_dataset_json_and_template_files(self) -> None:
        for name, (dataset_id, dataset_name, target) in POOL_22OCT.items():
            with self.subTest(node=name):
                node_dir = ROOT / "nodes" / name
                meta = json.loads((node_dir / "dataset.json").read_text())
                self.assertEqual(meta, {"node_name": name, "dataset_id": dataset_id,
                                        "dataset_name": dataset_name, "target": target})
                for filename in TEMPLATE_FILES:
                    self.assertEqual((node_dir / filename).read_bytes(),
                                     (ROOT / "nodes" / "openml_template" / filename).read_bytes(), filename)

    def test_data_cache_is_kept_but_ignored(self) -> None:
        for name in (*POOL_22OCT, "openml_credit_g"):
            with self.subTest(node=name):
                cache = ROOT / "nodes" / name / "data_cache"
                self.assertTrue((cache / ".gitkeep").is_file())
                self.assertEqual((cache / ".gitignore").read_text().split(), ["*", "!.gitignore", "!.gitkeep"])
                probe = subprocess.run(
                    ["git", "check-ignore", "-q", f"nodes/{name}/data_cache/openml/openml.org/x.gz"],
                    cwd=ROOT, capture_output=True,
                )
                if probe.returncode == 128:
                    self.skipTest("not a git checkout")
                self.assertEqual(probe.returncode, 0, "cache content must be git-ignored")

    def test_positive_class_and_inactive_imputer_on_synthetic_frames(self) -> None:
        # No OpenML access: replace the loader with a synthetic frame shaped like
        # the dataset (numeric features, OpenML target labels) and check the
        # frozen pipeline.  LabelEncoder sorts the labels, so the positive class
        # (classes_[1], scored by val_auc) is 'Rock' for sonar and '2' for the
        # '1'/'2'-encoded targets.  Binary AUC is invariant to which label is
        # positive, so this is a naming convention, not a metric choice.
        labels = {"openml_sonar": ("Rock", "Mine"), "openml_blood_transfusion": ("1", "2"),
                  "openml_qsar_biodeg": ("1", "2")}
        rng = np.random.default_rng(0)
        for name, (a, b) in labels.items():
            with self.subTest(node=name), tempfile.TemporaryDirectory() as tmp:
                node_dir = Path(tmp) / name
                shutil.copytree(ROOT / "nodes" / name, node_dir, ignore=shutil.ignore_patterns("data_cache", "__pycache__"))
                module = _load_benchmark(node_dir)
                x = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"V{i}" for i in range(1, 5)])
                y = pd.Series(pd.Categorical(np.where(x["V1"] + rng.normal(size=200) > 0, a, b)), name="Class")
                module._load_frame = lambda: (x.copy(), y.copy())
                results = {}
                for imputer in ("most_frequent", "constant"):
                    cfg = (node_dir / "config.yaml").read_text().replace("imputer: most_frequent", f"imputer: {imputer}")
                    (node_dir / f"cfg_{imputer}.yaml").write_text(cfg)
                    results[imputer] = module.run_openml_benchmark(node_dir / f"cfg_{imputer}.yaml")
                self.assertEqual(results["most_frequent"]["positive_label"], sorted((a, b))[1])
                self.assertEqual(results["most_frequent"]["val_auc"], results["constant"]["val_auc"])
                self.assertGreater(results["most_frequent"]["val_auc"], 0.5)


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
