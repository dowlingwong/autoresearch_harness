"""Bounded LLM transport retry (19 Oct 2026; rerun-2 STOP diagnosis follow-up).

Up to 3 attempts, only for transport/server-loss errors; the same D26 seed on
every attempt; every attempt logged in the llm_call event; parse failures are
not retried.  ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from autoresearch.llm.transport_retry import (
    LEGACY_POLICY,
    POLICY_VERSION,
    TransportRetryPolicy,
    classify_transport_error,
    invoke_with_retry,
)
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager, manager_seed
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]
GOOD = json.dumps({"summary": "s", "rationale": "r", "objective": "o", "param": "x", "new_value": "1"})
FAST = TransportRetryPolicy(max_attempts=3, backoff_seconds=(0.0, 0.0))


def _runner_error():
    from ollama import ResponseError
    return ResponseError("llama runner process no longer running: -1 ")


class FlakyLLM:
    """Raises the queued errors first, then answers; records every seed it gets."""

    def __init__(self, errors, content=GOOD):
        self.errors = list(errors)
        self.content = content
        self.seeds = []

    def invoke(self, messages, seed=None):
        self.seeds.append(seed)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(content=self.content, usage_metadata={}, response_metadata={})


def _status(spec):
    return ManagerStatus(campaign_id="retry", budget_index=3, current_best_metric=0.5,
                         metric_name=spec.metric_name, metric_direction=spec.metric_direction)


def _propose(manager, spec):
    ctx = build_memory_context([], MemoryMode.NONE, spec, 3)
    return manager.propose_next_trial(_status(spec), ctx, spec)


class TestClassification(unittest.TestCase):
    def test_transport_errors(self):
        import httpx
        from ollama import ResponseError

        cases = {
            "runner_terminated": ResponseError("llama runner process no longer running: -1 "),
            "unexpected_eof": ResponseError("an error was encountered while running the model: unexpected EOF"),
            "server_5xx": ResponseError("internal", status_code=503),
            "connection_refused": ConnectionRefusedError(111, "Connection refused"),
            "connection_reset": ConnectionResetError(104, "Connection reset by peer"),
            "server_disconnected": httpx.RemoteProtocolError("Server disconnected without sending a response."),
        }
        for expected, exc in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_transport_error(exc), expected)
        self.assertIsNotNone(classify_transport_error(httpx.ConnectError("[Errno 111] Connection refused")))

    def test_non_transport_errors_are_not_retryable(self):
        import httpx
        from ollama import ResponseError

        for exc in (ValueError("bad json"), ResponseError("model not found", status_code=404),
                    httpx.ReadTimeout("timed out"), TimeoutError("slow"), KeyError("x")):
            with self.subTest(exc=exc):
                self.assertIsNone(classify_transport_error(exc))

    def test_cause_chain_is_followed(self):
        try:
            try:
                raise ConnectionResetError(104, "reset")
            except ConnectionResetError as inner:
                raise RuntimeError("wrapper") from inner
        except RuntimeError as outer:
            self.assertEqual(classify_transport_error(outer), "connection_reset")


class TestInvokeWithRetry(unittest.TestCase):
    def test_bounded_attempts_and_backoff(self):
        sleeps = []
        calls = []

        def call():
            calls.append(1)
            raise ConnectionResetError(104, "reset")

        with self.assertRaises(ConnectionResetError) as ctx:
            invoke_with_retry(call, TransportRetryPolicy(max_attempts=3, backoff_seconds=(2.0, 4.0)),
                              sleep=sleeps.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [2.0, 4.0])
        attempts = ctx.exception.llm_call_attempts
        self.assertEqual([a["attempt"] for a in attempts], [1, 2, 3])
        self.assertEqual({a["error_class"] for a in attempts}, {"connection_reset"})

    def test_non_retryable_raises_at_once(self):
        calls = []

        def call():
            calls.append(1)
            raise ValueError("nope")

        with self.assertRaises(ValueError) as ctx:
            invoke_with_retry(call, FAST, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.exception.llm_call_attempts[0]["error_class"], "non_retryable")


class TestManagerRetry(unittest.TestCase):
    def setUp(self):
        self.spec = load_registered_node("lr_synthetic", repo_root=ROOT)

    def test_retry_succeeds_with_the_same_seed_and_logs_every_attempt(self):
        llm = FlakyLLM([_runner_error()])
        proposal = _propose(LangGraphManager(llm=llm, transport_retry=FAST), self.spec)
        expected_seed = manager_seed("retry", 3, 0, 0)
        self.assertEqual(llm.seeds, [expected_seed, expected_seed])
        meta = proposal.extra["llm_call"]
        self.assertTrue(meta["parse_ok"])
        self.assertEqual(meta["seed"], expected_seed)
        self.assertEqual(meta["transport_retry_policy"], POLICY_VERSION)
        self.assertEqual([(a["attempt"], a["error_class"]) for a in meta["transport_attempts"]],
                         [(1, "runner_terminated"), (2, None)])

    def test_parse_failure_is_not_retried(self):
        llm = FlakyLLM([], content="not json at all")
        proposal = _propose(LangGraphManager(llm=llm, transport_retry=FAST), self.spec)
        self.assertEqual(len(llm.seeds), 1)
        self.assertFalse(proposal.extra["llm_call"]["parse_ok"])
        self.assertEqual(len(proposal.extra["llm_call"]["transport_attempts"]), 1)

    def test_permanent_failure_raises_after_three_attempts(self):
        llm = FlakyLLM([_runner_error() for _ in range(5)])
        with self.assertRaises(Exception) as ctx:
            _propose(LangGraphManager(llm=llm, transport_retry=FAST), self.spec)
        self.assertEqual(len(llm.seeds), 3)
        self.assertEqual(len(ctx.exception.llm_call_attempts), 3)
        self.assertEqual(ctx.exception.llm_call_seed, manager_seed("retry", 3, 0, 0))

    def test_legacy_policy_is_the_historical_behaviour(self):
        llm = FlakyLLM([])
        proposal = _propose(LangGraphManager(llm=llm, transport_retry=LEGACY_POLICY), self.spec)
        self.assertNotIn("transport_attempts", proposal.extra["llm_call"])
        self.assertNotIn("transport_retry_policy", proposal.extra["llm_call"])
        llm = FlakyLLM([_runner_error()])
        with self.assertRaises(Exception):
            _propose(LangGraphManager(llm=llm, transport_retry=None), self.spec)
        self.assertEqual(len(llm.seeds), 1)

    def test_campaign_logs_a_failed_call_before_raising(self):
        from autoresearch.control_plane.budget import ComputeBudget
        from autoresearch.control_plane.campaign import run_real_campaign
        from autoresearch.memory.event_store import CampaignEventStore

        class NeverWorker:
            mode = "never"

            def run_trial(self, *a, **k):  # pragma: no cover - must not be reached
                raise AssertionError("worker must not run after a failed LLM call")

        llm = FlakyLLM([_runner_error() for _ in range(3)])
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            with self.assertRaises(Exception):
                run_real_campaign(node_spec=self.spec, campaign_id="retry", budget=2,
                                  manager_mode="langgraph_manager", memory_mode="none",
                                  records_path=Path(tmp) / "t.jsonl", worker=NeverWorker(),
                                  proposal_backend=LangGraphManager(llm=llm, transport_retry=FAST),
                                  event_store=CampaignEventStore(events),
                                  compute_budget=ComputeBudget(max_worker_runs=2))
            payloads = [json.loads(l) for l in events.read_text().splitlines()]
        failed = [p for p in payloads if p["event_type"] == "llm_call_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(failed[0]["payload"]["transport_attempts"]), 3)
        self.assertEqual(failed[0]["payload"]["seed"], manager_seed("retry", 1, 0, 0))
        self.assertNotIn("llm_call", [p["event_type"] for p in payloads])


class _DropOnceHandler(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(body.get("options", {}).get("seed"))
        if len(type(self).calls) == 1:
            # Simulate a server that dies mid-request: close without any response.
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        payload = {"model": "qwen2.5-coder:7b", "created_at": "2026-10-19T00:00:00Z",
                   "message": {"role": "assistant", "content": GOOD}, "done": True, "done_reason": "stop",
                   "prompt_eval_count": 3, "eval_count": 4}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write((json.dumps(payload) + "\n").encode())


class TestRealChatOllamaAgainstDroppingServer(unittest.TestCase):
    def test_server_drop_is_retried_with_the_same_seed(self):
        try:
            import langchain_ollama  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("langchain-ollama not installed")
        _DropOnceHandler.calls = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DropOnceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            spec = load_registered_node("lr_synthetic", repo_root=ROOT)
            manager = LangGraphManager(model="qwen2.5-coder:7b",
                                       host=f"http://127.0.0.1:{server.server_address[1]}",
                                       transport_retry=FAST)
            proposal = _propose(manager, spec)
        finally:
            server.shutdown()
        seed = manager_seed("retry", 3, 0, 0)
        self.assertEqual(_DropOnceHandler.calls, [seed, seed])
        attempts = proposal.extra["llm_call"]["transport_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIsNotNone(attempts[0]["error_class"])
        self.assertIsNone(attempts[1]["error_class"])


if __name__ == "__main__":
    unittest.main()
