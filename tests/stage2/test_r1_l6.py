"""L6 action tests: skip, revise, verify (+demote), stop, masks, scripted run."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.budget import ComputeBudget
from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.controller import ControlDecision, ControlView, compute_mask
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import (
    ExecutionStatus,
    TrialDecision,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class ScriptedController:
    version = "scripted/v1"
    deterministic = False

    def __init__(self, script: dict[tuple[int, str], str]) -> None:
        self.script = dict(script)

    def decide(self, view: ControlView, mask: dict[str, bool]) -> ControlDecision:
        action = self.script.get((view.budget_index, view.epoch), "continue")
        return ControlDecision(
            action=action,
            controller_version=self.version,
            input_hash=view.input_hash(),
        )


class StaticBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, proposal: ManagerProposal) -> None:
        self.proposal = proposal

    def propose_next_trial(self, status, memory_context, node_spec):
        return self.proposal


class ScriptedWorker:
    """Writes chosen file content, returns chosen metrics, supports seeded runs."""

    mode = "scripted_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0
        self.seed_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        content = spec.get("content")
        if content is not None:
            (self.node_root / "train.py").write_text(content, encoding="utf-8")
        metric = spec.get("metric")
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=spec.get("success", True),
            parsed_metrics={} if metric is None else {node_spec.metric_name: metric},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="before",
            git_commit_after="after",
        )

    def run_trial_at_seed(self, proposal, node_spec, budget_index, *, train_seed, split_seed=None):
        self.seed_calls += 1
        text = (self.node_root / "train.py").read_text(encoding="utf-8")
        base = 0.95 if "CANDIDATE" in text else 0.90
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=(),
            success=True,
            parsed_metrics={node_spec.metric_name: base + (int(train_seed) % 5) * 0.0001},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="before",
            git_commit_after="after",
        )


class CountingLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.responses.pop(0))


def _proposal(node_spec) -> ManagerProposal:
    return ManagerProposal(
        manager_mode="static_backend",
        proposal_summary="static-edit",
        proposal_rationale="test",
        target_files=node_spec.editable_paths,
        objective=f"Change {node_spec.editable_symbols[0]} from 1 to 2.",
        extra={},
    )


def _copy_node(tmp: Path) -> Path:
    import shutil

    node_root = tmp / "node"
    shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
    return node_root


class TestSkipAction(unittest.TestCase):
    def test_skip_ledgers_skipped_record_and_uses_no_worker_run(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        proposal = _proposal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [{"metric": 0.9, "content": "K1\n"}])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="skip", budget=1, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(proposal),
                controller=ScriptedController({(1, "E1"): "skip"}),
                compute_budget=ComputeBudget(max_worker_runs=5),
            )
            record = TrialAppendStore(records_path).read_all()[0]
        self.assertEqual(record.decision, TrialDecision.SKIPPED)
        self.assertEqual(record.execution_status, ExecutionStatus.FAILED)
        self.assertEqual(record.validity_status, ValidityStatus.VALID)
        self.assertIsNone(record.failure_category)
        self.assertEqual(record.extra["control"]["action"], "skip")
        self.assertEqual(worker.run_calls, 0)

    def test_skipped_record_round_trips(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        record = TrialRecord(
            trial_id="s-trial-001", campaign_id="s", node_id="lr_synthetic", budget_index=1,
            timestamp_start="2026-01-01T00:00:00Z", timestamp_end="2026-01-01T00:00:00Z",
            manager_mode="static", worker_mode="control_plane", memory_mode="none",
            proposal_summary="s", proposal_rationale="r", targeted_files=("train.py",),
            patch_ref="", git_commit_before="", git_commit_after="",
            execution_status=ExecutionStatus.FAILED, validity_status=ValidityStatus.VALID,
            failure_category=None, raw_log_ref="", parsed_metrics={},
            current_best_before=None, delta_vs_best=None, decision=TrialDecision.SKIPPED,
            decision_rationale="skip", wall_clock_seconds=0.0, cumulative_budget_consumed=1,
            provenance=__import__(
                "autoresearch.memory.provenance", fromlist=["build_trial_provenance"]
            ).build_trial_provenance("s-trial-001"),
            extra={"control": {"action": "skip"}},
        )
        restored = TrialRecord.from_mapping(record.to_dict())
        self.assertEqual(restored.decision, TrialDecision.SKIPPED)
        self.assertEqual(restored.extra["control"]["action"], "skip")


class TestReviseAction(unittest.TestCase):
    def test_revise_calls_manager_twice_and_records_both(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        stub = CountingLLM([
            json.dumps({"summary": "first", "rationale": "a", "objective": "edit one"}),
            json.dumps({"summary": "second", "rationale": "b", "objective": "edit two"}),
        ])
        manager = LangGraphManager(llm=stub)
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [{"metric": 0.9, "content": "K1\n"}])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="revise", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager,
                controller=ScriptedController({(1, "E1"): "revise"}),
                compute_budget=ComputeBudget(max_llm_calls=10),
            )
            record = TrialAppendStore(records_path).read_all()[0]
        self.assertEqual(record.extra["control"]["action"], "revise")
        revision = record.extra["control"]["revision"]
        self.assertIn("original", revision)
        self.assertIn("revised", revision)
        self.assertNotEqual(revision["original"], revision["revised"] or {})
        self.assertEqual(stub.calls, 2)


class TestVerifyAction(unittest.TestCase):
    def test_verify_swap_hash_and_rescue(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        proposal = _proposal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_node(Path(tmp))
            worker = ScriptedWorker(node_root, [
                {"metric": 0.90, "content": "BASE1\n"},
                {"metric": 0.80, "content": "CANDIDATE2\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="verify", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(proposal),
                controller=ScriptedController({(2, "E2"): "verify"}),
            )
            records = TrialAppendStore(records_path).read_all()
            final_content = (node_root / "train.py").read_text(encoding="utf-8")
        verify = records[1].extra["control"]["verify"]
        self.assertEqual(verify["original_decision"], TrialDecision.DISCARDED.value)
        self.assertEqual(verify["response"], "keep")
        self.assertEqual(records[1].decision, TrialDecision.KEPT)
        self.assertEqual(verify["candidate_state_hash_before"], verify["candidate_state_hash_after"])
        self.assertEqual(final_content, "CANDIDATE2\n")


class TestDemoteAndStopActions(unittest.TestCase):
    def test_demote_reverts_last_unverified_keep(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        proposal = _proposal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "K1\n"},
                {"metric": 0.95, "content": "K2\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            events_path = Path(tmp) / "events.jsonl"
            event_store = CampaignEventStore(events_path)
            run_real_campaign(
                node_spec=spec, campaign_id="demote", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(proposal), event_store=event_store,
                controller=ScriptedController({(2, "E0"): "demote"}),
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertIn("incumbent_demoted", [event["event_type"] for event in events])

    def test_stop_ends_campaign(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        proposal = _proposal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "K1\n"},
                {"metric": 0.91, "content": "K2\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            events_path = Path(tmp) / "events.jsonl"
            event_store = CampaignEventStore(events_path)
            run_real_campaign(
                node_spec=spec, campaign_id="stop", budget=5, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(proposal), event_store=event_store,
                controller=ScriptedController({(3, "E0"): "stop"}),
            )
            records = TrialAppendStore(records_path).read_all()
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertIn("campaign_stopped_by_controller", [event["event_type"] for event in events])


class TestMasks(unittest.TestCase):
    def _view(self, epoch):
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        return ControlView(
            epoch=epoch, budget_index=3, ledger_prefix=(), current_best=None,
            incumbent_stack_depth=0, proposal=None, provisional=None, budget={}, node=spec,
        )

    def test_e0_stop_and_demote_masks(self) -> None:
        mask = compute_mask(self._view("E0"), t_min=3, top_of_stack_unverified=False)
        self.assertTrue(mask["stop"])
        self.assertFalse(mask["demote"])
        mask = compute_mask(self._view("E0"), t_min=5, top_of_stack_unverified=True)
        self.assertFalse(mask["stop"])
        self.assertTrue(mask["demote"])

    def test_e1_skip_and_revise_masks(self) -> None:
        mask = compute_mask(
            self._view("E1"), proposal_passed_precondition=True,
            manager_deterministic=False, revisions_this_trial=0, llm_cap_reached=False,
        )
        self.assertTrue(mask["skip"])
        self.assertTrue(mask["revise"])
        mask = compute_mask(
            self._view("E1"), proposal_passed_precondition=True,
            manager_deterministic=True, revisions_this_trial=0, llm_cap_reached=False,
        )
        self.assertFalse(mask["revise"])

    def test_e2_verify_mask(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        valid = TrialRecord(
            trial_id="v-trial-001", campaign_id="v", node_id="lr_synthetic", budget_index=1,
            timestamp_start="t", timestamp_end="t", manager_mode="m", worker_mode="w",
            memory_mode="none", proposal_summary="s", proposal_rationale="r",
            targeted_files=("train.py",), patch_ref="", git_commit_before="", git_commit_after="",
            execution_status=ExecutionStatus.SUCCESS, validity_status=ValidityStatus.VALID,
            failure_category=None, raw_log_ref="", parsed_metrics={"val_score": 0.9},
            current_best_before=None, delta_vs_best=None, decision=TrialDecision.KEPT,
            decision_rationale="r", wall_clock_seconds=0.0, cumulative_budget_consumed=1,
            provenance=__import__(
                "autoresearch.memory.provenance", fromlist=["build_trial_provenance"]
            ).build_trial_provenance("v-trial-001"),
        )
        view = ControlView(
            epoch="E2", budget_index=1, ledger_prefix=(), current_best=None,
            incumbent_stack_depth=0, proposal=None, provisional=valid, budget={}, node=spec,
        )
        self.assertTrue(compute_mask(view, verify_runs=2, worker_runs_remaining=None)["verify"])
        self.assertFalse(compute_mask(view, verify_runs=2, worker_runs_remaining=1)["verify"])


class TestScriptedControllerCampaign(unittest.TestCase):
    def test_each_action_once_across_six_trials(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        stub = CountingLLM([
            json.dumps({"summary": f"s{i}", "rationale": "r", "objective": "edit"})
            for i in range(20)
        ])
        manager = LangGraphManager(llm=stub)
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "BASE1\n"},
                {"metric": 0.92, "content": "BASE3\n"},
                {"metric": 0.93, "content": "BASE4\n"},
                {"metric": 0.91, "content": "CANDIDATE5\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            events_path = Path(tmp) / "events.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="scripted", budget=6, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager, event_store=CampaignEventStore(events_path),
                controller=ScriptedController({
                    (2, "E1"): "skip",
                    (3, "E1"): "revise",
                    (4, "E0"): "demote",
                    (5, "E2"): "verify",
                    (6, "E0"): "stop",
                }),
            )
            records = TrialAppendStore(records_path).read_all()
            events = [json.loads(line) for line in events_path.read_text().splitlines()]

        event_types = [event["event_type"] for event in events]
        self.assertIn("campaign_stopped_by_controller", event_types)
        self.assertIn("incumbent_demoted", event_types)
        self.assertEqual(records[1].decision, TrialDecision.SKIPPED)
        self.assertEqual(records[2].extra["control"]["action"], "revise")
        self.assertEqual(records[4].extra["control"]["verify"]["response"], "keep")
        seen = {
            records[1].extra["control"]["action"],
            records[2].extra["control"]["action"],
            "verify" if "verify" in records[4].extra["control"] else None,
        }
        self.assertEqual(seen, {"skip", "revise", "verify"})


if __name__ == "__main__":
    unittest.main()
