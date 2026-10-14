"""D27 (H4): record-only edit-novelty flags as a separate event.

Written against r1 at 9c4902d (before D27); unittest style follows the
repository convention.  AI-assisted (BUILD_BATCH3, H4); ENGINEERING_ONLY.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.memory.schemas import TrialRecord
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class _ConfigWritingWorker:
    """Writes a chosen LEARNING_RATE into train.py and returns a chosen metric."""

    mode = "config_writing_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        (self.node_root / "train.py").write_text(
            f"LEARNING_RATE = {spec['learning_rate']}\n", encoding="utf-8"
        )
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=True,
            parsed_metrics={node_spec.metric_name: spec["metric"]},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="b",
            git_commit_after="a",
        )


class _StaticBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, node_spec) -> None:
        self._proposal = ManagerProposal(
            manager_mode=self.mode,
            proposal_summary="static",
            proposal_rationale="r",
            target_files=node_spec.editable_paths,
            objective=f"Change {node_spec.editable_symbols[0]} from 1 to 2.",
            extra={},
        )

    def propose_next_trial(self, status, memory_context, node_spec):
        return self._proposal


class TestD27EditNoveltyFlags(unittest.TestCase):
    def _sequence(self):
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        trials = [
            {"learning_rate": "0.002", "metric": 0.90},  # T1: new config, discarded (below incumbent)
            {"learning_rate": "0.002", "metric": 0.90},  # T2: config seen before (T1)
            {"learning_rate": "0.001", "metric": 0.95},  # T3: equals incumbent, identical metric
            {"learning_rate": "0.003", "metric": 0.99},  # T4: new config, kept
            {"learning_rate": "0.003", "metric": 0.97},  # T5: equals (new) incumbent, seen before
        ]
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
            (node_root / "train.py").write_text("LEARNING_RATE = 0.001\n", encoding="utf-8")
            worker = _ConfigWritingWorker(node_root, trials)
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d27", budget=5, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=_StaticBackend(spec), event_store=CampaignEventStore(events_path),
                initial_current_best=0.95,
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            records = TrialAppendStore(records_path).read_all()
        return [e["payload"] for e in events if e["event_type"] == "edit_novelty"], records

    def test_flags_on_the_constructed_sequence(self) -> None:
        flags, _ = self._sequence()
        self.assertEqual(len(flags), 5)
        self.assertEqual(flags[0], {
            "config_equals_incumbent": False,
            "config_seen_before": False,
            "metric_identical_to_incumbent": False,
        })
        # T2 repeats T1's effective config (T1 discarded; workspace restored).
        self.assertEqual(flags[1], {
            "config_equals_incumbent": False,
            "config_seen_before": True,
            "metric_identical_to_incumbent": False,
        })
        # T3 is the incumbent's own config and metric.
        self.assertEqual(flags[2], {
            "config_equals_incumbent": True,
            "config_seen_before": False,
            "metric_identical_to_incumbent": True,
        })
        # T4 is new and kept.
        self.assertEqual(flags[3], {
            "config_equals_incumbent": False,
            "config_seen_before": False,
            "metric_identical_to_incumbent": False,
        })
        # T5 repeats the new incumbent's config and has been evaluated before.
        self.assertEqual(flags[4], {
            "config_equals_incumbent": True,
            "config_seen_before": True,
            "metric_identical_to_incumbent": False,
        })

    def test_records_are_unchanged(self) -> None:
        _, records = self._sequence()
        self.assertEqual([r.decision.value for r in records], ["discarded", "discarded", "discarded", "kept", "discarded"])
        for record in records:
            payload = record.to_dict()
            self.assertNotIn("edit_novelty", payload)
            self.assertNotIn("config_equals_incumbent", payload)
            self.assertNotIn("config_seen_before", payload)
            self.assertNotIn("metric_identical_to_incumbent", payload)
            self.assertNotIn("novelty", json.dumps(record.extra))
        # The record schema itself is untouched.
        self.assertEqual(
            set(records[0].to_dict()),
            set(TrialRecord.__dataclass_fields__),
        )


if __name__ == "__main__":
    unittest.main()
