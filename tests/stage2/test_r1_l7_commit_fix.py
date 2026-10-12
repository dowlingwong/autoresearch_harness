"""L7 fix (found by E1b, 12 Oct): a resumed E2/E1 trial must commit like the live loop.

Before the fix, ``_complete_captured_trial``:
- left a discarded candidate's files in the workspace, so the continuation
  started from the discarded configuration;
- dropped verify's evidence and did not charge its 2r re-runs to the compute
  budget, so a verify branch got more continuation budget than its sibling;
- did not push a kept candidate onto the incumbent stack.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.budget import ComputeBudget
from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.snapshot import capture_snapshot, resume_campaign
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]


class AllE2:
    def __init__(self) -> None:
        self.snaps = []

    def __call__(self, context) -> None:
        if context.epoch == "E2" and context.worker_result is not None and context.worker_result.success:
            self.snaps.append(capture_snapshot(context))


class TestResumedCommitSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        spec = load_registered_node("digits_mlp", repo_root=ROOT)
        node = cls.tmp / "node"
        shutil.copytree(ROOT / "nodes" / "digits_mlp", node)
        cls.rec = AllE2()
        run_real_campaign(node_spec=spec, campaign_id="l7fix", budget=6, manager_mode="baseline_manager",
                          memory_mode="none", records_path=cls.tmp / "t.jsonl",
                          worker=LocalWorker(node_root=node, artifact_dir=cls.tmp / "art"),
                          compute_budget=ComputeBudget(max_worker_runs=30), snapshot_hook=cls.rec)
        assert cls.rec.snaps, "no E2 snapshot captured"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _files(self, root: Path, snap) -> dict:
        return {rel: (root / rel).read_bytes() for rel in snap.editable_paths}

    def test_continue_commits_like_the_live_loop(self) -> None:
        for k, snap in enumerate(self.rec.snaps):
            res = resume_campaign(snap, self.tmp / f"c{k}", first_action="continue", horizon=0)
            files = self._files(self.tmp / f"c{k}", snap)
            if res.first_record.decision == TrialDecision.KEPT:
                self.assertEqual(files, snap.editable_files)
                self.assertEqual(len(res.restored.incumbent_stack), len(snap.incumbent_stack) + 1)
            else:
                self.assertEqual(files, snap.incumbent_files)
                self.assertEqual(len(res.restored.incumbent_stack), len(snap.incumbent_stack))

    def test_verify_records_evidence_and_charges_runs(self) -> None:
        snap = self.rec.snaps[0]
        res = resume_campaign(snap, self.tmp / "v0", first_action="verify", horizon=0)
        control = res.first_record.extra["control"]
        self.assertEqual(control["action"], "verify")
        self.assertIn("verify", control)
        self.assertEqual(control["verify"]["r"], snap.verify_runs)
        before = int(snap.compute_budget_state.get("worker_runs", 0))
        self.assertEqual(res.restored.compute_budget.worker_runs, before + 2 * snap.verify_runs)
        files = self._files(self.tmp / "v0", snap)
        expected = snap.editable_files if res.first_record.decision == TrialDecision.KEPT else snap.incumbent_files
        self.assertEqual(files, expected)


if __name__ == "__main__":
    unittest.main()


class AllE1:
    def __init__(self) -> None:
        self.snaps = []

    def __call__(self, context) -> None:
        if context.epoch == "E1" and context.proposal is not None:
            self.snaps.append(capture_snapshot(context))


class TestResumedE1AndMask(unittest.TestCase):
    """L14 review fixes: resumed E1 charges its worker run; a forced verify respects the compute mask."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        spec = load_registered_node("digits_mlp", repo_root=ROOT)
        node = cls.tmp / "node"
        shutil.copytree(ROOT / "nodes" / "digits_mlp", node)
        cls.e1, cls.e2 = AllE1(), AllE2()

        def hook(ctx):
            cls.e1(ctx)
            cls.e2(ctx)

        run_real_campaign(node_spec=spec, campaign_id="l14", budget=4, manager_mode="baseline_manager",
                          memory_mode="none", records_path=cls.tmp / "t.jsonl",
                          worker=LocalWorker(node_root=node, artifact_dir=cls.tmp / "art"),
                          compute_budget=ComputeBudget(max_worker_runs=5), snapshot_hook=hook)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_resumed_e1_continue_charges_one_worker_run(self) -> None:
        snap = self.e1.snaps[0]
        res = resume_campaign(snap, self.tmp / "e1c", first_action="continue", horizon=0)
        before = int(snap.compute_budget_state.get("worker_runs", 0))
        self.assertEqual(res.restored.compute_budget.worker_runs, before + 1)

    def test_forced_verify_without_budget_falls_back_to_continue(self) -> None:
        late = [s for s in self.e2.snaps
                if 5 - int(s.compute_budget_state.get("worker_runs", 0)) < 2 * s.verify_runs]
        self.assertTrue(late, "need an E2 snapshot with fewer than 2r runs left")
        res = resume_campaign(late[-1], self.tmp / "late", first_action="verify", horizon=0)
        self.assertEqual(res.first_record.extra["control"]["action"], "continue")
        self.assertLessEqual(res.restored.compute_budget.worker_runs, 5)
