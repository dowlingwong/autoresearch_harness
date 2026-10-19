"""Verify split default "vary" for new campaigns (PLAN §10, 18 Oct 2026).

New campaigns declare "vary"; manifests and snapshots without the field keep
"fixed" (D24 pattern).  ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.action_versions import (
    ActionVersions,
    declare_new_campaign,
    resolve_verify_split_policy,
)
from autoresearch.control_plane.snapshot import read_snapshot

from stage2.test_r1_d4_action_versions import ScriptedController, _campaign
from autoresearch.control_plane.snapshot import SnapshotRecorder


class TestSplitDefault(unittest.TestCase):
    def test_new_campaign_manifest_declares_vary(self):
        manifest = {"schema": "x"}
        versions = declare_new_campaign(manifest)
        self.assertEqual(manifest["verify_split_policy"], "vary")
        self.assertEqual(manifest["action_versions"]["verify_split_policy"], "vary")
        self.assertEqual(versions, ActionVersions.from_manifest(manifest))
        self.assertEqual(resolve_verify_split_policy(manifest), "vary")

    def test_historical_manifest_is_never_rewritten(self):
        with self.assertRaises(ValueError):
            declare_new_campaign({"verify_split_policy": "fixed"})
        explicit = {}
        declare_new_campaign(explicit, verify_split_policy="fixed")
        self.assertEqual(explicit["verify_split_policy"], "fixed")

    def test_manifest_without_the_field_means_fixed(self):
        self.assertEqual(resolve_verify_split_policy({"schema": "r1.e1b_headroom/1"}), "fixed")
        self.assertEqual(resolve_verify_split_policy({"verify_split_policy": "fixed (decided 9 Oct)"}), "fixed")
        with self.assertRaises(ValueError):
            resolve_verify_split_policy({"verify_split_policy": "random"})

    def test_snapshot_without_field_reads_fixed_and_new_campaign_verifies_with_vary(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SnapshotRecorder(Path(tmp) / "s", epochs=("E2",), take_index=2)
            records, _ = _campaign(tmp, [0.80, 0.81], snapshot_hook=recorder,
                                   controller=ScriptedController({(2, "E2"): "verify"}),
                                   action_versions=ActionVersions.new_campaign())
            manifest = json.loads(recorder.manifest_path.read_text())
            self.assertEqual(manifest["verify_split_policy"], "vary")
            self.assertEqual(records[1].extra["control"]["verify"]["split_policy"], "vary")
            del manifest["verify_split_policy"]
            del manifest["action_versions"]
            recorder.manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(read_snapshot(recorder.manifest_path.parent).verify_split_policy, "fixed")


if __name__ == "__main__":
    unittest.main()
