"""L4 telemetry tests: per-call latency/usage and parse-failure reporting."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.campaign import run_dry_campaign
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.manager.base import ManagerStatus
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]


class _StubLLM:
    def __init__(self, content: str, *, prompt_tokens=10, completion_tokens=20):
        self._content = content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def invoke(self, messages):
        return SimpleNamespace(
            content=self._content,
            usage_metadata={
                "input_tokens": self._prompt_tokens,
                "output_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            },
            response_metadata={},
        )


def _status(spec):
    return ManagerStatus("c", 1, None, spec.metric_name, spec.metric_direction)


class TestLlmCallTelemetry(unittest.TestCase):
    def test_proposal_carries_latency_and_tokens(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        ctx = build_memory_context([], MemoryMode.NONE, spec, 1)
        manager = LangGraphManager(
            llm=_StubLLM(json.dumps({"summary": "s", "rationale": "r", "objective": "o"}))
        )
        proposal = manager.propose_next_trial(_status(spec), ctx, spec)
        meta = proposal.extra["llm_call"]
        self.assertGreaterEqual(meta["latency_ms"], 0.0)
        self.assertEqual(meta["prompt_tokens"], 10)
        self.assertEqual(meta["completion_tokens"], 20)
        self.assertTrue(meta["parse_ok"])

    def test_parse_failure_is_recorded(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        ctx = build_memory_context([], MemoryMode.NONE, spec, 1)
        manager = LangGraphManager(llm=_StubLLM("not json at all"))
        proposal = manager.propose_next_trial(_status(spec), ctx, spec)
        self.assertFalse(proposal.extra["llm_call"]["parse_ok"])

    def test_campaign_emits_llm_call_event(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            run_dry_campaign(
                node_spec=spec,
                campaign_id="l4",
                budget=1,
                manager_mode="langgraph_manager",
                memory_mode="none",
                records_path=Path(tmp) / "trials.jsonl",
                manager_llm=_StubLLM(json.dumps({"summary": "s", "rationale": "r", "objective": "o"})),
                event_store=CampaignEventStore(events_path),
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
        llm_events = [event for event in events if event["event_type"] == "llm_call"]
        self.assertEqual(len(llm_events), 1)
        self.assertIn("latency_ms", llm_events[0]["payload"])

    def test_deterministic_manager_emits_no_llm_call_event(self) -> None:
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            run_dry_campaign(
                node_spec=spec,
                campaign_id="l4b",
                budget=1,
                manager_mode="baseline_manager",
                memory_mode="none",
                records_path=Path(tmp) / "trials.jsonl",
                event_store=CampaignEventStore(events_path),
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertNotIn("llm_call", [event["event_type"] for event in events])


if __name__ == "__main__":
    unittest.main()
