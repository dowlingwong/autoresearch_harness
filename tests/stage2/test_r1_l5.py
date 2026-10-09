"""L5: controller hooks, default-behaviour regression, and leak tests.

Written before the hooks were implemented, against the unmodified harness.
The regression test replays a dry campaign and a fixed-seed LocalWorker
campaign under the default ``ContinueController`` and compares every
non-excluded ``TrialRecord`` field with a golden fixture captured from the
unmodified code (BUILD_RUNBOOK §0 lists the exclusions).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.control_plane.controller import (
    ContinueController,
    ControlViewLeakError,
    build_control_view,
)
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "l5_default_records.json"

EXCLUDED_FIELDS = (
    "timestamp_start",
    "timestamp_end",
    "wall_clock_seconds",
    "patch_ref",
    "raw_log_ref",
    "git_commit_before",
    "git_commit_after",
    "provenance",
    "extra",
    "training_seed",
    "fast_config_hash",
    "node_state_hash",
    "patch_hash",
)


def _project(records) -> list[dict]:
    projected = []
    for record in records:
        payload = record.to_dict()
        for key in EXCLUDED_FIELDS:
            payload.pop(key, None)
        projected.append(payload)
    return projected


class TestDefaultControllerRegression(unittest.TestCase):
    def test_dry_and_local_default_records_match_fixture(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)

        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "dry_trials.jsonl"
            run_dry_campaign(
                node_spec=spec,
                campaign_id="l5dry",
                budget=3,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                controller=ContinueController(),
            )
            dry = _project(TrialAppendStore(records_path).read_all())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node_root = root / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
            records_path = root / "real_trials.jsonl"
            worker = LocalWorker(
                node_root=node_root,
                artifact_dir=root / "artifacts",
                train_seed=42,
                split_seed=42,
            )
            run_real_campaign(
                node_spec=spec,
                campaign_id="l5real",
                budget=2,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=worker,
                controller=ContinueController(),
            )
            real = _project(TrialAppendStore(records_path).read_all())

        self.assertEqual(dry, fixture["dry"])
        self.assertEqual(real, fixture["real"])


class TestControlViewLeak(unittest.TestCase):
    def _clean_kwargs(self) -> dict:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        return {
            "epoch": "E1",
            "budget_index": 1,
            "ledger_prefix": (),
            "current_best": None,
            "incumbent_stack_depth": 0,
            "proposal": None,
            "provisional": None,
            "budget": {"trials_remaining": 3},
            "node": spec,
        }

    def test_clean_view_builds(self) -> None:
        view = build_control_view(**self._clean_kwargs())
        self.assertEqual(view.epoch, "E1")
        self.assertTrue(view.input_hash())

    def test_rejects_audit_metrics(self) -> None:
        with self.assertRaises(ControlViewLeakError):
            build_control_view(**self._clean_kwargs(), audit_metrics={"val_score": 0.9})

    def test_rejects_future_records(self) -> None:
        with self.assertRaises(ControlViewLeakError):
            build_control_view(**self._clean_kwargs(), future_records=[{"trial_id": "later"}])

    def test_rejects_sibling_outcomes(self) -> None:
        with self.assertRaises(ControlViewLeakError):
            build_control_view(**self._clean_kwargs(), sibling_outcomes=[{"action": "skip"}])


if __name__ == "__main__":
    unittest.main()
