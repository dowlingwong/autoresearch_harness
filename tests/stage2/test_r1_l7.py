"""L7 tests: snapshot round-trip, restore equivalence, isolation, forced action, audit."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.snapshot import (
    SnapshotRecorder,
    capture_snapshot,
    read_snapshot,
    restore_snapshot,
    resume_campaign,
    snapshot_state_hashes,
    write_snapshot,
)
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]


def _copy_node(tmp: Path, name: str = "lr_synthetic") -> Path:
    node = tmp / name
    shutil.copytree(ROOT / "nodes" / name, node)
    return node


def _run_capture(tmp: Path, *, budget: int = 3, take_index: int = 1):
    spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    node = _copy_node(tmp)
    recorder = SnapshotRecorder(tmp / "snap", epochs=("E1",), take_index=take_index)
    worker = LocalWorker(node_root=node, artifact_dir=tmp / "art", train_seed=42, split_seed=42)
    run_real_campaign(
        node_spec=spec,
        campaign_id="l7",
        budget=budget,
        manager_mode="baseline_manager",
        memory_mode="none",
        records_path=tmp / "trials.jsonl",
        worker=worker,
        snapshot_hook=recorder,
    )
    records = TrialAppendStore(tmp / "trials.jsonl").read_all()
    return spec, node, recorder, records


class TestSnapshotRoundTrip(unittest.TestCase):
    def test_write_read_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, _ = _run_capture(tmp)
            self.assertEqual(len(recorder.snapshots), 1)
            original = recorder.snapshots[0]
            loaded = read_snapshot(tmp / "snap")
        self.assertEqual(loaded.epoch, "E1")
        self.assertEqual(loaded.budget_index, original.budget_index)
        self.assertEqual(loaded.layer_hashes, original.layer_hashes)
        self.assertEqual(snapshot_state_hashes(loaded), original.layer_hashes)
        self.assertIsNotNone(loaded.proposal)


class TestRestoreEquivalenceAndIsolation(unittest.TestCase):
    def test_restored_state_hashes_equal_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, _ = _run_capture(tmp)
            snapshot = recorder.snapshots[0]
            restored = restore_snapshot(snapshot, tmp / "restore_a")
            # Re-snapshot the restored workspace and compare editable-layer hash.
            content = {
                rel: (restored.into_dir / rel).read_bytes() for rel in snapshot.editable_paths
            }
            from autoresearch.control_plane.snapshot import _hash_file_map

            self.assertEqual(_hash_file_map(content), snapshot.layer_hashes["editable"])

    def test_siblings_do_not_share_writable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, _ = _run_capture(tmp)
            snapshot = recorder.snapshots[0]
            branch_a = restore_snapshot(snapshot, tmp / "branch_a")
            branch_b = restore_snapshot(snapshot, tmp / "branch_b")
            (branch_a.into_dir / "train.py").write_text("SIBLING_A\n", encoding="utf-8")
            self.assertNotEqual(
                (branch_b.into_dir / "train.py").read_text(encoding="utf-8"), "SIBLING_A\n"
            )
            self.assertNotEqual(branch_a.into_dir, branch_b.into_dir)


class TestForcedFirstAction(unittest.TestCase):
    def test_continue_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, records = _run_capture(tmp)
            result = resume_campaign(recorder.snapshots[0], tmp / "resume", first_action="continue")
        self.assertIsNotNone(result.first_record)
        self.assertEqual(result.first_record.budget_index, records[0].budget_index)
        self.assertEqual(
            result.first_record.parsed_metrics.get("val_score"),
            records[0].parsed_metrics.get("val_score"),
        )

    def test_masked_action_is_rejected_to_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, _ = _run_capture(tmp)
            # verify is not an E1 action; it must be rejected and continue.
            result = resume_campaign(recorder.snapshots[0], tmp / "resume2", first_action="verify")
        self.assertEqual(result.first_record.extra["control"]["action"], "continue")
        self.assertTrue(result.first_record.decision.value in {"kept", "discarded", "failed_invalid"})

    def test_horizon_runs_further_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, _ = _run_capture(tmp)
            result = resume_campaign(
                recorder.snapshots[0], tmp / "horizon", first_action="continue", horizon=2
            )
        self.assertIsNotNone(result.continuation)
        self.assertEqual(result.continuation.records_written, 2)


class TestRestoreAudit(unittest.TestCase):
    def test_replay_reproduces_metric_bit_for_bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, recorder, records = _run_capture(tmp)
            snapshot = recorder.snapshots[0]
            original_metric = records[0].parsed_metrics.get("val_score")
            result = resume_campaign(snapshot, tmp / "audit", first_action="continue")
            replay_metric = result.first_record.parsed_metrics.get("val_score")
        self.assertIsNotNone(original_metric)
        # Exact equality, never a tolerance.
        self.assertEqual(replay_metric, original_metric)


if __name__ == "__main__":
    unittest.main()
