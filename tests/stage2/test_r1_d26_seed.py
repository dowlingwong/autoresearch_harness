"""D26 (H3): deterministic per-call manager sampling seed.

Written against r1 at 9c4902d (before D26); unittest style follows the
repository convention.  AI-assisted (BUILD_BATCH3, H3); ENGINEERING_ONLY.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.campaign import run_dry_campaign
from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import LLMConfig
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager, manager_seed
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.summarizer import build_memory_context
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]


class _SeedRecordingLLM:
    """Stub chat model that records the seed kwarg passed to invoke."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.seeds: list[object] = []

    def invoke(self, messages, **kwargs):
        self.seeds.append(kwargs.get("seed"))
        return SimpleNamespace(
            content=self._content, usage_metadata={}, response_metadata={}
        )


def _status(spec, campaign_id: str = "c", budget_index: int = 3) -> ManagerStatus:
    return ManagerStatus(
        campaign_id=campaign_id,
        budget_index=budget_index,
        current_best_metric=None,
        metric_name=spec.metric_name,
        metric_direction=spec.metric_direction,
    )


def _response() -> str:
    return json.dumps({"summary": "s", "rationale": "r", "objective": "o"})


class TestD26SeedDerivation(unittest.TestCase):
    def test_derivation_is_stable(self) -> None:
        self.assertEqual(manager_seed("inst", 3, 1, 0), manager_seed("inst", 3, 1, 0))

    def test_seed_differs_across_trials_repetitions_and_revisions(self) -> None:
        seeds = {
            manager_seed("inst", 3, 0, 0),
            manager_seed("inst", 4, 0, 0),
            manager_seed("inst", 3, 1, 0),
            manager_seed("inst", 3, 0, 1),
            manager_seed("other", 3, 0, 0),
        }
        self.assertEqual(len(seeds), 5)

    def test_seed_is_a_non_negative_32_bit_int(self) -> None:
        seed = manager_seed("inst", 1, 0, 0)
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)


class TestD26SeedIsPassedAndLogged(unittest.TestCase):
    def test_seed_passed_to_the_llm_and_recorded_in_llm_call(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        llm = _SeedRecordingLLM(_response())
        manager = LangGraphManager(llm=llm, repetition=2)
        memory = build_memory_context([], "none", spec, 3)
        status = _status(spec, campaign_id="inst", budget_index=3)
        proposal = manager.propose_next_trial(status, memory, spec, revision_attempt=1)
        expected = manager_seed("inst", 3, 2, 1)
        self.assertEqual(llm.seeds, [expected])
        self.assertEqual(proposal.extra["llm_call"]["seed"], expected)

    def test_revision_attempt_changes_the_seed(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        llm = _SeedRecordingLLM(_response())
        manager = LangGraphManager(llm=llm, repetition=0)
        memory = build_memory_context([], "none", spec, 3)
        status = _status(spec, campaign_id="inst", budget_index=3)
        manager.propose_next_trial(status, memory, spec)
        manager.propose_next_trial(status, memory, spec, revision_feedback="f", revision_attempt=1)
        self.assertNotEqual(llm.seeds[0], llm.seeds[1])

    def test_llm_call_event_carries_the_seed(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        llm = _SeedRecordingLLM(_response())
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            run_dry_campaign(
                node_spec=spec, campaign_id="d26", budget=1,
                manager_mode="langgraph_manager", memory_mode="none",
                records_path=Path(tmp) / "trials.jsonl",
                manager_llm=llm, event_store=CampaignEventStore(events_path),
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
        llm_events = [e for e in events if e["event_type"] == "llm_call"]
        self.assertEqual(len(llm_events), 1)
        self.assertEqual(llm_events[0]["payload"]["seed"], llm.seeds[0])

    def test_seed_is_skipped_for_providers_without_one(self) -> None:
        # Review F6: D26 is an Ollama seed; an Anthropic-backed manager keeps
        # its pre-D26 unseeded behaviour instead of raising.
        import autoresearch.manager.langgraph_manager as lgm
        from autoresearch.llm.providers import LLMConfig

        captured = {}

        def fake_build(config, temperature, seed=None):
            captured["provider"] = config.provider
            captured["seed"] = seed
            return SimpleNamespace(invoke=lambda messages: None)

        self.assertTrue(hasattr(lgm, "_build_chat_model"))
        original_config, original_build = lgm.resolve_llm_config, lgm._build_chat_model
        try:
            lgm.resolve_llm_config = lambda model: LLMConfig(
                provider="anthropic", model_name="claude", base_url="", api_key="", model_id="claude",
            )
            lgm._build_chat_model = fake_build
            LangGraphManager(model="anthropic/claude")._resolve_llm(seed=12345)
        finally:
            lgm.resolve_llm_config, lgm._build_chat_model = original_config, original_build
        self.assertEqual(captured, {"provider": "anthropic", "seed": None})

    def test_pinned_ollama_backend_receives_the_seed_at_construction(self) -> None:
        config = LLMConfig(
            provider="ollama",
            model_name="qwen2.5-coder:7b",
            base_url="http://localhost:11434",
            api_key="",
            model_id="ollama/qwen2.5-coder:7b",
        )
        model = _build_chat_model(config, 0.2, seed=12345)
        self.assertEqual(model.seed, 12345)
        self.assertEqual(model.temperature, 0.2)
        unseeded = _build_chat_model(config, 0.2)
        self.assertIsNone(unseeded.seed)


if __name__ == "__main__":
    unittest.main()
