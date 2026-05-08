"""Tests for autoresearch.memory.append_store — append-only JSONL ledger."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.memory.append_store import AppendOnlyStoreError, TrialAppendStore
from autoresearch.memory.schemas import (
    ExecutionStatus,
    FailureCategory,
    TrialDecision,
    TrialProvenance,
    TrialRecord,
    ValidityStatus,
)


def _provenance(n: int = 1) -> TrialProvenance:
    return TrialProvenance(
        proposal_id=f"prop-{n:03d}",
        patch_id=f"patch-{n:03d}",
        run_id=f"run-{n:03d}",
        metric_id=f"metric-{n:03d}",
        decision_id=f"decision-{n:03d}",
    )


def _record(n: int = 1, **overrides) -> TrialRecord:
    base = dict(
        trial_id=f"campaign-trial-{n:03d}",
        campaign_id="campaign",
        node_id="resnet_trigger",
        budget_index=n,
        timestamp_start="2026-01-01T00:00:00Z",
        timestamp_end="2026-01-01T00:10:00Z",
        manager_mode="baseline_manager",
        worker_mode="dry_run_worker",
        memory_mode="none",
        proposal_summary=f"change-{n}",
        proposal_rationale=f"reason {n}",
        targeted_files=("train.py",),
        patch_ref=f"experiments/artifacts/trial-{n:03d}/patch.diff",
        git_commit_before="abc",
        git_commit_after="def",
        execution_status=ExecutionStatus.SUCCESS,
        validity_status=ValidityStatus.VALID,
        failure_category=None,
        raw_log_ref=f"experiments/artifacts/trial-{n:03d}/run.log",
        parsed_metrics={"val_auc": 0.78 + n * 0.001},
        current_best_before=0.78,
        delta_vs_best=n * 0.001,
        decision=TrialDecision.KEPT,
        decision_rationale="Improved.",
        wall_clock_seconds=600.0,
        cumulative_budget_consumed=n,
        provenance=_provenance(n),
    )
    base.update(overrides)
    return TrialRecord(**base)


class TestTrialAppendStore(unittest.TestCase):
    def test_empty_store_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            self.assertEqual(store.read_all(), [])

    def test_append_and_read_single_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            r = _record(1)
            store.append(r)
            records = store.read_all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].trial_id, r.trial_id)

    def test_append_multiple_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            for i in range(1, 6):
                store.append(_record(i))
            records = store.read_all()
            self.assertEqual(len(records), 5)
            for i, record in enumerate(records, start=1):
                self.assertEqual(record.budget_index, i)

    def test_append_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            store.append(_record(1))
            store.append(_record(2))
            records = store.read_all()
            self.assertEqual(len(records), 2)

    def test_append_many(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            store.append_many([_record(i) for i in range(1, 4)])
            self.assertEqual(len(store.read_all()), 3)

    def test_file_is_valid_jsonl(self):
        """Each line must be valid JSON independently."""
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            for i in range(1, 4):
                store.append(_record(i))
            lines = store.path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                obj = json.loads(line)
                self.assertIn("trial_id", obj)

    def test_corrupt_line_raises_append_only_store_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            path.write_text("not valid json\n", encoding="utf-8")
            store = TrialAppendStore(path)
            with self.assertRaises(AppendOnlyStoreError):
                store.read_all()

    def test_parent_dir_created_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "deep" / "trials.jsonl"
            store = TrialAppendStore(path)
            store.append(_record(1))
            self.assertTrue(path.exists())

    def test_read_all_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            r = _record(1)
            path.write_text(
                json.dumps(r.to_dict(), sort_keys=True) + "\n\n",
                encoding="utf-8",
            )
            store = TrialAppendStore(path)
            records = store.read_all()
            self.assertEqual(len(records), 1)

    def test_returned_path_from_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TrialAppendStore(Path(tmp) / "trials.jsonl")
            returned = store.append(_record(1))
            self.assertEqual(returned, store.path)


if __name__ == "__main__":
    unittest.main()
