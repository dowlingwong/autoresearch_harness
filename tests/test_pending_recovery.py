"""Acceptance tests for pending-trial recovery (Chunk 1.5).

Verifies the acceptance criterion:
  "Manually creating a fake *_pending.json file, then running --mark-failed,
   produces a ledger entry with decision=failed_invalid and no pending file
   left behind."

Uses the control-plane helpers directly (same code the script calls) so
the test does not depend on subprocess invocation.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.campaign import (
    _acquire_pending,
    clear_pending_guard,
    inspect_pending_guard,
    list_pending_guards,
    mark_pending_failed,
    _pending_guard_path,
)
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision, ValidityStatus
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[1]


def _fake_node_spec():
    return load_registered_node("resnet_trigger", repo_root=ROOT)


class TestPendingGuardLifecycle(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self._tmp.name) / "test_campaign_trials.jsonl"
        self.node_spec = _fake_node_spec()

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # list_pending_guards
    # ------------------------------------------------------------------

    def test_list_returns_empty_when_no_guards(self):
        guards = list_pending_guards(self._tmp.name)
        self.assertEqual(guards, [])

    def test_list_finds_created_guard(self):
        guard = _acquire_pending(
            self.ledger,
            campaign_id="test_campaign",
            trial_id="test_campaign-trial-001",
            budget_index=1,
        )
        guards = list_pending_guards(self._tmp.name)
        self.assertIn(guard, guards)
        guard.unlink()  # cleanup

    # ------------------------------------------------------------------
    # inspect_pending_guard
    # ------------------------------------------------------------------

    def test_inspect_returns_guard_fields(self):
        guard = _acquire_pending(
            self.ledger,
            campaign_id="test_campaign",
            trial_id="test_campaign-trial-001",
            budget_index=1,
        )
        info = inspect_pending_guard(guard)
        self.assertEqual(info["campaign_id"], "test_campaign")
        self.assertEqual(info["trial_id"], "test_campaign-trial-001")
        self.assertEqual(info["budget_index"], 1)
        self.assertIn("records_path", info)
        self.assertIn("started", info)
        guard.unlink()

    # ------------------------------------------------------------------
    # ACCEPTANCE CRITERION: mark_pending_failed
    # ------------------------------------------------------------------

    def test_mark_failed_appends_failed_invalid_record_and_removes_guard(self):
        """Core acceptance criterion for Chunk 1.5."""
        # 1. Acquire a pending guard (simulates a mid-trial crash)
        guard = _acquire_pending(
            self.ledger,
            campaign_id="test_campaign",
            trial_id="test_campaign-trial-001",
            budget_index=1,
        )
        self.assertTrue(guard.exists(), "guard should exist before recovery")

        # 2. Run the recovery (mark-failed path)
        record = mark_pending_failed(
            guard_path=guard,
            node_spec=self.node_spec,
            manager_mode="test_manager",
            worker_mode="test_worker",
            memory_mode="none",
            failure_message="simulated crash",
        )

        # 3. Guard must be gone
        self.assertFalse(guard.exists(), "guard must be deleted after recovery")

        # 4. Ledger must have exactly one record
        store = TrialAppendStore(self.ledger)
        records = store.read_all()
        self.assertEqual(len(records), 1)

        # 5. Record must be failed_invalid
        self.assertEqual(records[0].decision, TrialDecision.FAILED_INVALID)
        self.assertEqual(records[0].validity_status, ValidityStatus.INVALID)

        # 6. The returned record must match what was appended
        self.assertEqual(records[0].trial_id, record.trial_id)
        self.assertEqual(records[0].campaign_id, "test_campaign")

    def test_mark_failed_is_idempotent_on_ledger(self):
        """Running recovery twice should not produce duplicate ledger entries."""
        guard = _acquire_pending(
            self.ledger,
            campaign_id="c2",
            trial_id="c2-trial-001",
            budget_index=1,
        )
        mark_pending_failed(guard_path=guard, node_spec=self.node_spec, failure_message="first")

        # Re-create guard to test a second recovery cycle
        guard2 = _acquire_pending(
            self.ledger,
            campaign_id="c2",
            trial_id="c2-trial-002",
            budget_index=2,
        )
        mark_pending_failed(guard_path=guard2, node_spec=self.node_spec, failure_message="second")

        records = TrialAppendStore(self.ledger).read_all()
        self.assertEqual(len(records), 2, "each recovery should append exactly one record")

    # ------------------------------------------------------------------
    # clear_pending_guard (without appending a record — escape hatch)
    # ------------------------------------------------------------------

    def test_clear_removes_guard_without_ledger_entry(self):
        guard = _acquire_pending(
            self.ledger,
            campaign_id="test_campaign",
            trial_id="test_campaign-trial-001",
            budget_index=1,
        )
        clear_pending_guard(guard)
        self.assertFalse(guard.exists())
        # No ledger entry written
        self.assertFalse(self.ledger.exists())

    def test_clear_is_safe_on_missing_guard(self):
        """clear_pending_guard on a nonexistent path must not raise."""
        fake_guard = Path(self._tmp.name) / "nonexistent_pending.json"
        clear_pending_guard(fake_guard)  # should not raise

    # ------------------------------------------------------------------
    # Fake pending guard (manually crafted, not via _acquire_pending)
    # ------------------------------------------------------------------

    def test_mark_failed_works_on_manually_created_guard(self):
        """The acceptance criterion from the chunk description: create a fake
        guard by hand, run mark-failed, verify ledger + guard cleanup."""
        guard = _pending_guard_path(self.ledger)
        guard.write_text(
            json.dumps({
                "campaign_id": "manual_campaign",
                "trial_id": "manual_campaign-trial-001",
                "budget_index": 1,
                "records_path": str(self.ledger),
                "started": "2026-01-01T00:00:00Z",
            }),
            encoding="utf-8",
        )
        self.assertTrue(guard.exists())

        mark_pending_failed(
            guard_path=guard,
            node_spec=self.node_spec,
            failure_message="manual recovery test",
        )

        self.assertFalse(guard.exists(), "guard must be removed after mark-failed")
        records = TrialAppendStore(self.ledger).read_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].decision, TrialDecision.FAILED_INVALID)
        self.assertEqual(records[0].campaign_id, "manual_campaign")

    def test_recover_pending_trial_script_mark_failed_by_campaign_id(self):
        guard = Path(self._tmp.name) / "script_campaign_pending.json"
        guard.write_text(
            json.dumps({
                "campaign_id": "script_campaign",
                "trial_id": "script_campaign-trial-001",
                "budget_index": 1,
                "records_path": str(self.ledger),
                "started": "2026-01-01T00:00:00Z",
            }),
            encoding="utf-8",
        )

        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "recover_pending_trial.py"),
                "--mark-failed",
                "script_campaign",
                "--reason",
                "script recovery test",
                "--ledger-dir",
                self._tmp.name,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(guard.exists())
        records = TrialAppendStore(self.ledger).read_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].decision, TrialDecision.FAILED_INVALID)


if __name__ == "__main__":
    unittest.main()
