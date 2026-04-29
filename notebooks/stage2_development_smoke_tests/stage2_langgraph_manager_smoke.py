"""Stage 2 smoke test: LangGraphManager — no real Ollama required.

Tests the full path:
  LangGraphManager (with FakeListChatModel)
    -> graph: prepare_context -> generate_proposal -> validate_proposal
    -> ManagerProposal
    -> Stage 2 dry-run campaign (DryRunWorker)
    -> TrialRecord with manager_mode="langgraph_manager"

Guarantees:
  - LangGraph is scoped entirely to proposal generation.
  - The proposal passes through the same Stage 2 control-plane path as other managers.
  - No Ollama, no real worker, no real trial execution needed.
  - Graph validates its output and falls back gracefully on parse failure.

Run from repo root:
  .venv/bin/python notebooks/stage2_development_smoke_tests/stage2_langgraph_manager_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# LangGraph + langchain-core are in the project .venv
VENV_SITE = ROOT / ".venv" / "lib"
if VENV_SITE.exists():
    for p in sorted(VENV_SITE.iterdir()):
        site = p / "site-packages"
        if site.exists() and str(site) not in sys.path:
            sys.path.insert(0, str(site))

from langchain_core.language_models import FakeListChatModel

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import DryRunWorker


def _fake_llm(responses: list[str]) -> FakeListChatModel:
    return FakeListChatModel(responses=responses)


def _valid_json_response(summary: str, rationale: str, objective: str) -> str:
    import json
    return json.dumps({"summary": summary, "rationale": rationale, "objective": objective})


class TestLangGraphManagerPropose(unittest.TestCase):
    def setUp(self):
        self.node_spec = load_registered_node("resnet_trigger", repo_root=ROOT)
        self.status = ManagerStatus(
            campaign_id="lg_test",
            budget_index=1,
            current_best_metric=0.780,
            metric_name="val_auc",
            metric_direction="maximize",
        )

    def _context(self, mode: MemoryMode) -> object:
        return build_memory_context([], mode, self.node_spec, self.status.budget_index)

    def test_returns_manager_proposal_with_correct_mode(self):
        llm = _fake_llm([_valid_json_response("reduce-lr", "smaller lr helps", "In train.py change lr.")])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(proposal.manager_mode, "langgraph_manager")

    def test_objective_comes_from_llm_output(self):
        expected_obj = "In train.py, change learning_rate from 1e-3 to 5e-4 only."
        llm = _fake_llm([_valid_json_response("reduce-lr", "rationale", expected_obj)])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(proposal.objective, expected_obj)

    def test_summary_from_llm(self):
        llm = _fake_llm([_valid_json_response("add-dropout-0.3", "r", "o")])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(proposal.proposal_summary, "add-dropout-0.3")

    def test_target_files_locked_to_node_spec(self):
        llm = _fake_llm([_valid_json_response("s", "r", "o")])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(proposal.target_files, self.node_spec.editable_paths)

    def test_graph_is_compiled_once_and_reused(self):
        llm = _fake_llm(["invalid json response", _valid_json_response("s2", "r", "o2")])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        manager.propose_next_trial(self.status, ctx, self.node_spec)
        graph_id = id(manager._graph)
        manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(id(manager._graph), graph_id, "graph should not be rebuilt")

    def test_prepare_context_includes_node_name_and_metric(self):
        """The prompt fed to the LLM must contain the node name and metric."""
        captured_prompts = []

        class CapturingLLM(FakeListChatModel):
            def invoke(self, messages, **kwargs):
                captured_prompts.append(messages[0].content)
                return super().invoke(messages, **kwargs)

        llm = CapturingLLM(responses=[_valid_json_response("s", "r", "o")])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.APPEND_ONLY_SUMMARY)
        manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertTrue(captured_prompts, "LLM should have been called")
        prompt = captured_prompts[0]
        self.assertIn("resnet_trigger", prompt)
        self.assertIn("val_auc", prompt)
        self.assertIn("maximize", prompt)
        self.assertIn("train.py", prompt)

    def test_validate_proposal_falls_back_on_bad_json(self):
        llm = _fake_llm(["this is not json at all"])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        # Should not raise; fallback proposal must be valid
        self.assertEqual(proposal.manager_mode, "langgraph_manager")
        self.assertTrue(len(proposal.objective) > 0)

    def test_validate_proposal_strips_markdown_fences(self):
        wrapped = "```json\n" + _valid_json_response("fenced-s", "fenced-r", "fenced-o") + "\n```"
        llm = _fake_llm([wrapped])
        manager = LangGraphManager(llm=llm)
        ctx = self._context(MemoryMode.NONE)
        proposal = manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertEqual(proposal.proposal_summary, "fenced-s")
        self.assertEqual(proposal.objective, "fenced-o")

    def test_memory_context_text_appears_in_prompt(self):
        """With rationale mode, memory context text must reach the LLM prompt."""
        captured_prompts = []

        class CapturingLLM(FakeListChatModel):
            def invoke(self, messages, **kwargs):
                captured_prompts.append(messages[0].content)
                return super().invoke(messages, **kwargs)

        # Build a non-empty memory context by including a synthetic record
        from autoresearch.memory.schemas import (
            ExecutionStatus, TrialDecision, TrialRecord, ValidityStatus
        )
        from autoresearch.memory.provenance import build_trial_provenance
        fake_record = TrialRecord(
            trial_id="lg_test-trial-001",
            campaign_id="lg_test",
            node_id="resnet_trigger",
            budget_index=1,
            timestamp_start="2026-01-01T00:00:00Z",
            timestamp_end="2026-01-01T00:01:00Z",
            manager_mode="baseline_manager",
            worker_mode="dry_run_worker",
            memory_mode="none",
            proposal_summary="baseline-run",
            proposal_rationale="",
            targeted_files=("train.py",),
            patch_ref="",
            git_commit_before="",
            git_commit_after="",
            execution_status=ExecutionStatus.SUCCESS,
            validity_status=ValidityStatus.VALID,
            failure_category=None,
            raw_log_ref="",
            parsed_metrics={"val_auc": 0.780},
            current_best_before=None,
            delta_vs_best=None,
            decision=TrialDecision.KEPT,
            decision_rationale="first valid",
            wall_clock_seconds=10.0,
            cumulative_budget_consumed=1,
            provenance=build_trial_provenance("lg_test-trial-001"),
        )
        ctx = build_memory_context(
            [fake_record], MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE,
            self.node_spec, budget_index=1
        )
        self.assertGreater(len(ctx.context_text), 0)

        llm = CapturingLLM(responses=[_valid_json_response("s", "r", "o")])
        manager = LangGraphManager(llm=llm)
        manager.propose_next_trial(self.status, ctx, self.node_spec)
        self.assertIn("baseline-run", captured_prompts[0])


class TestLangGraphManagerInCampaign(unittest.TestCase):
    """Verify that LangGraphManager produces normal Stage 2 TrialRecords."""

    def setUp(self):
        self.node_spec = load_registered_node("resnet_trigger", repo_root=ROOT)

    def _patch_manager(self, fake_responses: list[str]) -> None:
        """Monkey-patch _manager() so the campaign uses our FakeListChatModel."""
        import autoresearch.control_plane.campaign as c_mod
        from autoresearch.manager.langgraph_manager import LangGraphManager
        self._original_manager = c_mod._manager

        def patched_manager(mode: str, llm=None):
            if mode == "langgraph_manager":
                return LangGraphManager(llm=_fake_llm(fake_responses))
            return self._original_manager(mode, llm=llm)

        c_mod._manager = patched_manager

    def _restore_manager(self) -> None:
        import autoresearch.control_plane.campaign as c_mod
        c_mod._manager = self._original_manager

    def test_dry_run_campaign_with_langgraph_manager(self):
        budget = 3
        self._patch_manager([
            _valid_json_response(f"proposal-{i}", f"rationale-{i}", f"Reduce lr step {i}.")
            for i in range(budget)
        ])
        try:
            with tempfile.TemporaryDirectory() as d:
                records_path = Path(d) / "lg_trials.jsonl"
                result = run_dry_campaign(
                    node_spec=self.node_spec,
                    campaign_id="langgraph_smoke",
                    budget=budget,
                    manager_mode="langgraph_manager",
                    memory_mode="append_only_summary_with_rationale",
                    records_path=records_path,
                )
                self.assertEqual(result.records_written, budget)
                records = TrialAppendStore(records_path).read_all()
                self.assertEqual(len(records), budget)
                for record in records:
                    self.assertEqual(record.manager_mode, "langgraph_manager")
                    self.assertIn(record.decision, (TrialDecision.KEPT, TrialDecision.DISCARDED, TrialDecision.FAILED_INVALID))
                # All decisions owned by Stage 2 — DryRunWorker always succeeds
                self.assertTrue(all(r.decision == TrialDecision.KEPT for r in records))
        finally:
            self._restore_manager()

    def test_proposal_summaries_appear_in_records(self):
        budget = 2
        self._patch_manager([
            _valid_json_response("add-dropout-lg", "r", "o"),
            _valid_json_response("reduce-wd-lg", "r", "o"),
        ])
        try:
            with tempfile.TemporaryDirectory() as d:
                records_path = Path(d) / "lg_trials.jsonl"
                run_dry_campaign(
                    node_spec=self.node_spec,
                    campaign_id="lg_summary_check",
                    budget=budget,
                    manager_mode="langgraph_manager",
                    memory_mode="none",
                    records_path=records_path,
                )
                records = TrialAppendStore(records_path).read_all()
                summaries = [r.proposal_summary for r in records]
                self.assertIn("add-dropout-lg", summaries)
                self.assertIn("reduce-wd-lg", summaries)
        finally:
            self._restore_manager()


if __name__ == "__main__":
    unittest.main(verbosity=2)
