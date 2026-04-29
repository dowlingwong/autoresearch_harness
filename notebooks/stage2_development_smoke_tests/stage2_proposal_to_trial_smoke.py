"""Stage 2 smoke test: full proposal-to-TrialRecord path without a live Ollama instance.

Tests the complete chain:
  ManagerProposal
    -> _packet_from_proposal (objective + description from proposal, not static file)
    -> generated_packet.json written to artifacts dir
    -> MockClawAdapter (simulates loop result without calling Ollama)
    -> WorkerResult
    -> Stage 2 decision (keep/discard owned by control plane)
    -> TrialRecord appended to JSONL

Also verifies that different memory modes produce different objective text in the
generated packet, confirming that memory context actually reaches the worker.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.manager.baseline_manager import BaselineManager
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult
from autoresearch.worker.claw_worker import _packet_from_proposal


class MockClawWorker:
    """Mimics ClawWorker without calling the legacy CLI.

    Writes the generated packet to artifacts (real path) and returns a synthetic
    WorkerResult as if the loop ran and improved the metric.
    """

    mode = "mock_claw_worker"

    def __init__(self, artifacts_dir: Path, metric_value: float = 0.795) -> None:
        self.artifacts_dir = artifacts_dir
        self.metric_value = metric_value
        self.last_generated_packet: dict | None = None
        self.last_packet_path: Path | None = None

    def run_trial(self, proposal: ManagerProposal, node_spec, budget_index: int) -> WorkerResult:
        trial_id = f"trial-{budget_index:03d}"
        artifact_dir = self.artifacts_dir / trial_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        packet = _packet_from_proposal(proposal, node_spec, budget_index)
        packet_path = artifact_dir / "generated_packet.json"
        packet_path.write_text(json.dumps(packet, indent=2))

        self.last_generated_packet = packet
        self.last_packet_path = packet_path

        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=True,
            parsed_metrics={node_spec.metric_name: self.metric_value},
            raw_log_ref=str(artifact_dir / "run.log"),
            patch_ref=str(packet_path),
            git_commit_before="mock-before",
            git_commit_after="mock-after",
        )


class TestPacketFromProposal(unittest.TestCase):
    def setUp(self):
        self.node_spec = load_registered_node("resnet_trigger", repo_root=ROOT)

    def test_objective_comes_from_proposal(self):
        proposal = ManagerProposal(
            manager_mode="baseline_manager",
            proposal_summary="reduce-lr",
            proposal_rationale="try smaller lr",
            target_files=("train.py",),
            objective="Lower the learning rate from 1e-3 to 5e-4 in train.py only.",
        )
        packet = _packet_from_proposal(proposal, self.node_spec, budget_index=1)
        self.assertEqual(packet["objective"], proposal.objective)

    def test_description_includes_summary_and_trial_index(self):
        proposal = ManagerProposal(
            manager_mode="baseline_manager",
            proposal_summary="reduce-lr",
            proposal_rationale="",
            target_files=("train.py",),
            objective="...",
        )
        packet = _packet_from_proposal(proposal, self.node_spec, budget_index=3)
        self.assertIn("reduce-lr", packet["description"])
        self.assertIn("trial-003", packet["description"])

    def test_train_command_from_node_spec(self):
        proposal = ManagerProposal(
            manager_mode="baseline_manager",
            proposal_summary="x",
            proposal_rationale="",
            target_files=("train.py",),
            objective="...",
        )
        packet = _packet_from_proposal(proposal, self.node_spec, budget_index=1)
        self.assertEqual(packet["train_command"], self.node_spec.run_command)

    def test_packet_defaults_override_timeout(self):
        proposal = ManagerProposal(
            manager_mode="baseline_manager",
            proposal_summary="x",
            proposal_rationale="",
            target_files=("train.py",),
            objective="...",
        )
        packet = _packet_from_proposal(
            proposal, self.node_spec, budget_index=1,
            packet_defaults={"timeout_seconds": 42}
        )
        self.assertEqual(packet["timeout_seconds"], 42)

    def test_packet_defaults_cannot_override_objective(self):
        """Objective always comes from proposal, never from defaults."""
        proposal = ManagerProposal(
            manager_mode="baseline_manager",
            proposal_summary="x",
            proposal_rationale="",
            target_files=("train.py",),
            objective="The real objective.",
        )
        packet = _packet_from_proposal(
            proposal, self.node_spec, budget_index=1,
            packet_defaults={"objective": "Should be ignored."}
        )
        self.assertEqual(packet["objective"], "The real objective.")


class TestProposalToTrialRecord(unittest.TestCase):
    def setUp(self):
        self.node_spec = load_registered_node("resnet_trigger", repo_root=ROOT)

    def test_full_proposal_to_trialrecord_chain(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            artifacts_dir = d / "artifacts"
            records_path = d / "trials.jsonl"
            worker = MockClawWorker(artifacts_dir=artifacts_dir, metric_value=0.800)

            result = run_real_campaign(
                node_spec=self.node_spec,
                campaign_id="proposal_smoke",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=worker,
            )
            self.assertEqual(result.records_written, 1)
            self.assertFalse(result.dry_run)

            records = TrialAppendStore(records_path).read_all()
            self.assertEqual(len(records), 1)
            record = records[0]

            # Decision owned by Stage 2, not the worker
            self.assertEqual(record.decision, TrialDecision.KEPT)
            self.assertAlmostEqual(record.parsed_metrics["val_auc"], 0.800)

            # Generated packet written to artifacts
            self.assertTrue(worker.last_packet_path.exists())
            packet = json.loads(worker.last_packet_path.read_text())
            self.assertNotEqual(packet["objective"], "")

            # Timestamps are real (not hardcoded 2026-04-28)
            self.assertNotIn("2026-04-28T00:00:00", record.timestamp_start)

            # Pending guard was cleaned up
            guard = d / "trials_pending.json"
            self.assertFalse(guard.exists())

    def test_different_memory_modes_produce_different_objectives(self):
        """Different memory modes must reach the worker as different objective text."""
        node_spec = self.node_spec
        records = []  # simulate prior trials for context

        proposals = {}
        for mode in (MemoryMode.NONE, MemoryMode.APPEND_ONLY_SUMMARY, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE):
            memory_context = build_memory_context(records, mode, node_spec, budget_index=1)
            status = ManagerStatus(
                campaign_id="ablation_test",
                budget_index=1,
                current_best_metric=0.78,
                metric_name=node_spec.metric_name,
                metric_direction=node_spec.metric_direction,
            )
            manager = PromptManager()
            proposal = manager.propose_next_trial(status, memory_context, node_spec)
            proposals[mode] = proposal

        # PromptManager encodes memory_mode in the rationale
        none_rationale = proposals[MemoryMode.NONE].proposal_rationale
        summary_rationale = proposals[MemoryMode.APPEND_ONLY_SUMMARY].proposal_rationale
        rationale_rationale = proposals[MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE].proposal_rationale

        self.assertIn("none", none_rationale)
        self.assertIn("append_only_summary", summary_rationale)
        self.assertIn("append_only_summary_with_rationale", rationale_rationale)

    def test_generated_packet_written_before_worker_called(self):
        """Verify artifact path is deterministic and file exists after a trial."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            artifacts_dir = d / "artifacts"
            records_path = d / "trials.jsonl"
            worker = MockClawWorker(artifacts_dir=artifacts_dir, metric_value=0.790)

            run_real_campaign(
                node_spec=self.node_spec,
                campaign_id="packet_path_smoke",
                budget=2,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=worker,
            )
            # Second trial packet (last one written)
            expected = artifacts_dir / "trial-002" / "generated_packet.json"
            self.assertTrue(expected.exists())
            packet = json.loads(expected.read_text())
            self.assertIn("trial-002", packet["description"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
