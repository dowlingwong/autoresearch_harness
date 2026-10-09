"""R1 build-batch defects (BUILD_RUNBOOK L3).

Each test in this module is written to fail against the harness as it stood
before the corresponding fix and pass after it.  unittest style follows the
repository convention.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.claw_worker import _extract_worker_result

ROOT = Path(__file__).resolve().parents[2]


class TestPromptManagerIsDeterministic(unittest.TestCase):
    """L3.1: prompt_manager makes no LLM call and must say so."""

    def test_prompt_manager_declares_deterministic(self) -> None:
        self.assertTrue(getattr(PromptManager, "deterministic", False))

    def test_prompt_manager_holds_no_llm(self) -> None:
        manager = PromptManager()
        self.assertFalse(hasattr(manager, "_injected_llm"))
        self.assertFalse(hasattr(manager, "llm"))


class TestClawWorkerMetricSign(unittest.TestCase):
    """L3.2: the legacy path must not invert a minimise metric.

    The autoresearch node's metric is ``val_bpb`` (minimise); the legacy
    experiment payload already reports ``val_bpb`` directly, so it must be
    stored unmodified rather than as ``1 - val_bpb``.
    """

    def test_val_bpb_node_keeps_raw_sign(self) -> None:
        spec = load_registered_node("autoresearch_macos", repo_root=ROOT)
        self.assertEqual(spec.metric_name, "val_bpb")
        loop_result = {
            "history": [
                {
                    "run": {
                        "base_commit": "before",
                        "commit": "after",
                        "worker": {"last_result": {"changed_files": ["train.py"]}},
                        "experiment": {"success": True, "val_bpb": 0.2, "log_path": ""},
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = _extract_worker_result(
                loop_result,
                spec,
                packet_ref="packet.json",
                artifact_dir=Path(tmp) / "artifacts",
                node_root=Path(tmp),
            )
        self.assertEqual(result.parsed_metrics["val_bpb"], 0.2)


if __name__ == "__main__":
    unittest.main()
