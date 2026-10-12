"""L12: LLM calls made by a controller (side query) are charged to the LLM-call cap."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.budget import ComputeBudget
from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.control_plane.controller import ControlDecision
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]


class ChargingController:
    """Continues everywhere but reports one LLM call per decision, like a side-query controller."""

    version = "charging/test"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, view, mask) -> ControlDecision:
        self.calls += 1
        return ControlDecision(action="continue", controller_version=self.version,
                               input_hash=view.input_hash(), llm_calls=1)


class TestControllerLlmCharging(unittest.TestCase):
    def _run(self, controller, cap):
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            node = tmp / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node)
            records = tmp / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="l12", budget=6, manager_mode="baseline_manager",
                memory_mode="none", records_path=records, worker=LocalWorker(node_root=node, artifact_dir=tmp / "art"),
                controller=controller, compute_budget=ComputeBudget(max_llm_calls=cap),
            )
            return len(TrialAppendStore(records).read_all())

    def test_side_query_calls_exhaust_the_llm_cap(self) -> None:
        # three decisions (E0, E1, E2) per trial, one charged call each: a cap of 3 allows one trial
        self.assertEqual(self._run(ChargingController(), cap=3), 1)

    def test_uncharged_decisions_leave_the_cap_alone(self) -> None:
        from autoresearch.control_plane.controller import ContinueController
        self.assertEqual(self._run(ContinueController(), cap=3), 6)

    def test_decision_rejects_negative_llm_calls(self) -> None:
        with self.assertRaises(ValueError):
            ControlDecision(action="continue", llm_calls=-1)


if __name__ == "__main__":
    unittest.main()
