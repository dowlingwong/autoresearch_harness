from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.campaign import (
    clear_pending_guard,
    inspect_pending_guard,
    list_pending_guards,
    mark_pending_failed,
    run_dry_campaign,
    run_real_campaign,
)
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.baseline_manager import BaselineManager
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision, ValidityStatus
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult
from autoresearch.worker.claw_worker import _extract_worker_result

ROOT = Path(__file__).resolve().parents[2]


class FakeWorker:
    mode = "fake_worker"

    def __init__(self, results: list[WorkerResult]) -> None:
        self._results = list(results)

    def run_trial(self, proposal, node_spec, budget_index):
        if not self._results:
            raise AssertionError("fake worker has no queued result")
        return self._results.pop(0)


class RaisingWorker:
    mode = "raising_worker"

    def run_trial(self, proposal, node_spec, budget_index):
        raise RuntimeError("worker crashed")


def node_spec():
    return load_registered_node("resnet_trigger", repo_root=ROOT)


def worker_result(
    *,
    changed_files=("train.py",),
    success=True,
    metrics=None,
    failure_message=None,
):
    spec = node_spec()
    return WorkerResult(
        worker_mode="fake_worker",
        changed_files=tuple(changed_files),
        success=success,
        parsed_metrics={spec.metric_name: 0.81} if metrics is None else metrics,
        raw_log_ref="fake.log",
        patch_ref="fake.diff",
        git_commit_before="before",
        git_commit_after="after",
        failure_message=failure_message,
        extra={"parsed_metrics_ref": "parsed.json"},
    )


class TestManagers(unittest.TestCase):
    def test_baseline_manager_returns_bounded_proposal(self):
        spec = node_spec()
        status = ManagerStatus("c", 1, None, spec.metric_name, spec.metric_direction)
        ctx = build_memory_context([], MemoryMode.NONE, spec, 1)
        proposal = BaselineManager().propose_next_trial(status, ctx, spec)
        self.assertEqual(proposal.manager_mode, "baseline_manager")
        self.assertEqual(proposal.target_files, spec.editable_paths)
        self.assertTrue(proposal.objective)

    def test_prompt_manager_uses_memory_mode(self):
        spec = node_spec()
        status = ManagerStatus("c", 2, 0.8, spec.metric_name, spec.metric_direction)
        ctx = build_memory_context([], MemoryMode.APPEND_ONLY_SUMMARY, spec, 2)
        proposal = PromptManager().propose_next_trial(status, ctx, spec)
        self.assertEqual(proposal.manager_mode, "prompt_manager")
        self.assertIn("compressed prior-trial memory", proposal.objective)


@unittest.skipIf(
    importlib.util.find_spec("langgraph") is None or importlib.util.find_spec("langchain_core") is None,
    "langgraph/langchain-core not installed",
)
class TestLangGraphManager(unittest.TestCase):
    def test_fake_llm_proposal_records_hash_metadata(self):
        class FakeLLM:
            def invoke(self, messages):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "summary": "fake-lg",
                            "rationale": "fake rationale",
                            "objective": "Edit only train.py and make one bounded change.",
                        }
                    )
                )

        spec = node_spec()
        status = ManagerStatus("lg", 1, None, spec.metric_name, spec.metric_direction)
        ctx = build_memory_context([], MemoryMode.NONE, spec, 1)
        proposal = LangGraphManager(llm=FakeLLM()).propose_next_trial(status, ctx, spec)
        self.assertEqual(proposal.proposal_summary, "fake-lg")
        self.assertIn("context_sha256", proposal.extra)
        self.assertIn("raw_proposal_sha256", proposal.extra)


class TestCampaigns(unittest.TestCase):
    def test_dry_run_campaign_writes_records(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "dry_trials.jsonl"
            result = run_dry_campaign(
                node_spec=spec,
                campaign_id="dry",
                budget=2,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
            )
            self.assertEqual(result.records_written, 2)
            records = TrialAppendStore(records_path).read_all()
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.decision == TrialDecision.KEPT for record in records))

    def test_scope_violation_becomes_failed_invalid(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "scope_trials.jsonl"
            run_real_campaign(
                node_spec=spec,
                campaign_id="scope",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=FakeWorker([worker_result(changed_files=("prepare.py",))]),
            )
            record = TrialAppendStore(records_path).read_all()[0]
            self.assertEqual(record.validity_status, ValidityStatus.INVALID)
            self.assertEqual(record.failure_category, FailureCategory.INVALID_EDIT_SCOPE)
            self.assertEqual(record.decision, TrialDecision.FAILED_INVALID)

    def test_metric_missing_becomes_failed_invalid(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "metric_trials.jsonl"
            run_real_campaign(
                node_spec=spec,
                campaign_id="metric",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=FakeWorker([worker_result(metrics={})]),
            )
            record = TrialAppendStore(records_path).read_all()[0]
            self.assertEqual(record.failure_category, FailureCategory.METRIC_MISSING)
            self.assertEqual(record.decision, TrialDecision.FAILED_INVALID)

    def test_command_failure_with_metric_is_runtime_error(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "failed_trials.jsonl"
            run_real_campaign(
                node_spec=spec,
                campaign_id="failed",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=FakeWorker([worker_result(success=False, failure_message="command failed")]),
            )
            record = TrialAppendStore(records_path).read_all()[0]
            self.assertEqual(record.failure_category, FailureCategory.RUNTIME_ERROR)
            self.assertEqual(record.extra["worker_failure_message"], "command failed")

    def test_worker_exception_appends_failed_record(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "exception_trials.jsonl"
            run_real_campaign(
                node_spec=spec,
                campaign_id="exception",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=records_path,
                worker=RaisingWorker(),
            )
            record = TrialAppendStore(records_path).read_all()[0]
            self.assertEqual(record.worker_mode, "raising_worker")
            self.assertEqual(record.failure_category, FailureCategory.RUNTIME_ERROR)
            self.assertEqual(record.extra["worker_failure_message"], "worker crashed")


class TestPendingRecovery(unittest.TestCase):
    def test_list_inspect_mark_failed_and_clear_pending_guard(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records_path = root / "recover_trials.jsonl"
            guard = root / "recover_trials_pending.json"
            guard.write_text(
                json.dumps(
                    {
                        "campaign_id": "recover",
                        "trial_id": "recover-trial-001",
                        "budget_index": 1,
                        "records_path": str(records_path),
                        "started": "2026-01-01T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(list_pending_guards(root), [guard])
            self.assertEqual(inspect_pending_guard(guard)["trial_id"], "recover-trial-001")
            record = mark_pending_failed(
                guard_path=guard,
                node_spec=spec,
                manager_mode="baseline_manager",
                worker_mode="fake_worker",
                memory_mode="none",
                failure_message="test recovery",
            )
            self.assertFalse(guard.exists())
            self.assertEqual(record.failure_category, FailureCategory.RUNTIME_ERROR)
            self.assertEqual(len(TrialAppendStore(records_path).read_all()), 1)

            guard.write_text("{}", encoding="utf-8")
            clear_pending_guard(guard)
            self.assertFalse(guard.exists())


class TestClawWorkerExtraction(unittest.TestCase):
    def test_extract_worker_result_captures_artifacts(self):
        spec = node_spec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "node"
            artifacts = Path(tmp) / "artifacts"
            root.mkdir()
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "train.py").write_text("LR = 1e-3\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True, capture_output=True, text=True)
            before = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
            (root / "train.py").write_text("LR = 5e-4\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "candidate"], cwd=root, check=True, capture_output=True, text=True)
            after = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
            (root / "run.log").write_text("done\n", encoding="utf-8")

            loop_result = {
                "history": [
                    {
                        "run": {
                            "base_commit": before,
                            "commit": after,
                            "recommended_status": "keep",
                            "worker": {
                                "last_result": {
                                    "changed_files": ["train.py"],
                                    "stop_reason": "completed",
                                }
                            },
                            "experiment": {
                                "success": True,
                                "val_bpb": 0.2,
                                "log_path": "run.log",
                            },
                        }
                    }
                ]
            }
            result = _extract_worker_result(
                loop_result,
                spec,
                packet_ref=str(artifacts / "generated_packet.json"),
                artifact_dir=artifacts,
                node_root=root,
            )
            self.assertTrue(result.success)
            self.assertEqual(result.parsed_metrics[spec.metric_name], 0.8)
            self.assertTrue(Path(result.raw_log_ref).exists())
            self.assertTrue(Path(result.patch_ref).exists())
            self.assertTrue(Path(result.extra["parsed_metrics_ref"]).exists())
            self.assertTrue(Path(result.extra["legacy_loop_result_ref"]).exists())


if __name__ == "__main__":
    unittest.main()
