"""R1 build-batch defects (BUILD_RUNBOOK L3).

Each test in this module is written to fail against the harness as it stood
before the corresponding fix and pass after it.  unittest style follows the
repository convention.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult
from autoresearch.worker.claw_worker import _extract_worker_result

ROOT = Path(__file__).resolve().parents[2]


class _FakeWorker:
    mode = "fake_worker"

    def __init__(self, results: list[WorkerResult]) -> None:
        self._results = list(results)

    def run_trial(self, proposal, node_spec, budget_index):
        return self._results.pop(0)


class _StaticBackend:
    mode = "static_backend"

    def __init__(self, proposal: ManagerProposal) -> None:
        self._proposal = proposal

    def propose_next_trial(self, status, memory_context, node_spec):
        return self._proposal


def _static_proposal(node_spec, symbol: str, old: str, new: str) -> ManagerProposal:
    return ManagerProposal(
        manager_mode="static_backend",
        proposal_summary="static-edit",
        proposal_rationale="test",
        target_files=node_spec.editable_paths,
        objective=f"Change {symbol} from {old} to {new}.",
        extra={
            "deterministic_patch": True,
            "structured_edit": {
                "type": "python_constant",
                "path": node_spec.editable_paths[0],
                "symbol": symbol,
                "old": old,
                "new": new,
            },
        },
    )


def _result(node_spec, *, metric=None, success=True) -> WorkerResult:
    parsed = {} if metric is None else {node_spec.metric_name: metric}
    return WorkerResult(
        worker_mode="fake_worker",
        changed_files=node_spec.editable_paths,
        success=success,
        parsed_metrics=parsed,
        raw_log_ref="fake.log",
        patch_ref="fake.diff",
        git_commit_before="before",
        git_commit_after="after",
    )


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


class TestCurrentBestOnResume(unittest.TestCase):
    """L3.3: _current_best must respect metric_direction and count kept only."""

    def test_resume_on_minimise_node_uses_minimum(self) -> None:
        spec = load_registered_node("autoresearch_macos", repo_root=ROOT)
        self.assertEqual(spec.metric_direction, "minimize")
        proposal = _static_proposal(spec, "DEPTH", "8", "9")
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "min_trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="min", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                worker=_FakeWorker([_result(spec, metric=0.5), _result(spec, metric=0.3)]),
                proposal_backend=_StaticBackend(proposal),
            )
            run_real_campaign(
                node_spec=spec, campaign_id="min", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                worker=_FakeWorker([_result(spec, metric=0.4)]),
                proposal_backend=_StaticBackend(proposal),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(
            [record.decision for record in records[:2]],
            [TrialDecision.KEPT, TrialDecision.KEPT],
        )
        self.assertEqual(records[2].current_best_before, 0.3)

    def test_resume_ignores_invalid_high_metric(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        self.assertEqual(spec.metric_direction, "maximize")
        proposal = _static_proposal(spec, "LEARNING_RATE", "5e-4", "4e-4")
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "max_trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="max", budget=1, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                worker=_FakeWorker([_result(spec, metric=0.99, success=False)]),
                proposal_backend=_StaticBackend(proposal),
            )
            run_real_campaign(
                node_spec=spec, campaign_id="max", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                worker=_FakeWorker([_result(spec, metric=0.5)]),
                proposal_backend=_StaticBackend(proposal),
            )
            run_real_campaign(
                node_spec=spec, campaign_id="max", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                worker=_FakeWorker([_result(spec, metric=0.6)]),
                proposal_backend=_StaticBackend(proposal),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(records[0].decision, TrialDecision.FAILED_INVALID)
        self.assertEqual(records[0].parsed_metrics[spec.metric_name], 0.99)
        self.assertEqual(records[2].current_best_before, 0.5)


if __name__ == "__main__":
    unittest.main()
