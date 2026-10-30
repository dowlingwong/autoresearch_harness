"""R1 A.10 (25 Oct 2026): the ``llamacpp`` provider against llama-server's OpenAI endpoint.

PLAN §10 R1-INFER-LLAMACPP: LLM inference uses llama.cpp (``llama-server``),
not Ollama.  These tests drive the real ``ChatOpenAI`` client (no inference)
against a small HTTP stand-in for llama-server and check what the C3 label
sweep relies on:

* the base URL reaches the OpenAI-compatible path ``/v1/chat/completions``
  whether or not ``LLAMACPP_BASE_URL`` ends in ``/v1``;
* no API key is needed for a local server (none is set in the environment);
* the D26 seed reaches the request body and is recorded in the ``llm_call``
  metadata;
* OpenAI-client connection errors are classified as transport errors like the
  Ollama ones, and the client makes no hidden retries of its own, so the
  bounded transport retry (19 Oct) is the only retry and logs every attempt.

ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import resolve_llm_config
from autoresearch.llm.transport_retry import TransportRetryPolicy, classify_transport_error
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager, manager_seed
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]
GOOD = json.dumps({"summary": "s", "rationale": "r", "objective": "o", "param": "x", "new_value": "1"})
FAST = TransportRetryPolicy(max_attempts=3, backoff_seconds=(0.0, 0.0))
# The client error's class name depends on the library versions: openai's APIConnectionError, or
# langchain-openai 1.7.x's OpenAIConnectionError wrapper (the Track S host pin, langchain-openai 1.7.3).
# The transport classification is what matters; either class name is accepted.
CONNECTION_ERROR_TYPES = {"APIConnectionError", "OpenAIConnectionError"}
NO_KEYS = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "LLAMACPP_API_KEY", "OPENAI_BASE_URL", "LLAMACPP_BASE_URL")}


def _status(spec):
    return ManagerStatus(campaign_id="llamacpp", budget_index=3, current_best_metric=0.5,
                         metric_name=spec.metric_name, metric_direction=spec.metric_direction)


def _propose(manager, spec):
    ctx = build_memory_context([], MemoryMode.NONE, spec, 3)
    return manager.propose_next_trial(_status(spec), ctx, spec)


class _LlamaServerStandIn(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/v1/chat/completions`` only; scripted per-request behaviour."""

    protocol_version = "HTTP/1.1"
    requests: list = []
    script: list = []   # per request: "ok" | "drop" | "503"

    def log_message(self, *args):
        pass

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        cls = type(self)
        cls.requests.append({"path": self.path, "body": body,
                             "authorization": self.headers.get("Authorization")})
        action = cls.script[len(cls.requests) - 1] if len(cls.requests) <= len(cls.script) else "ok"
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
            return
        if action == "drop":  # a server dying mid-request: no response at all
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        if action == "503":
            self._send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
            return
        self._send(200, {"id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                         "model": body.get("model"), "system_fingerprint": "b0000-test",
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": GOOD}}],
                         "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}})


class _Server:
    def __init__(self, script):
        _LlamaServerStandIn.requests = []
        _LlamaServerStandIn.script = list(script)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaServerStandIn)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class TestLlamaCppBaseUrl(unittest.TestCase):
    def test_base_url_gets_the_openai_v1_prefix(self):
        for given, expected in [("http://127.0.0.1:8085", "http://127.0.0.1:8085/v1"),
                                ("http://127.0.0.1:8085/", "http://127.0.0.1:8085/v1"),
                                ("http://127.0.0.1:8085/v1", "http://127.0.0.1:8085/v1"),
                                ("http://127.0.0.1:8085/v1/", "http://127.0.0.1:8085/v1")]:
            with self.subTest(given=given), mock.patch.dict(os.environ, {"LLAMACPP_BASE_URL": given}):
                self.assertEqual(resolve_llm_config("llamacpp/qwen").base_url, expected)
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            self.assertEqual(resolve_llm_config("llamacpp/qwen").base_url, "http://localhost:8080/v1")

    def test_chat_model_needs_no_api_key_and_makes_no_hidden_retries(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            llm = _build_chat_model(resolve_llm_config("llamacpp/qwen"), 0.2, seed=11)
        self.assertEqual(llm.max_retries, 0)
        self.assertEqual(llm.seed, 11)
        self.assertEqual(str(llm.openai_api_base), "http://localhost:8080/v1")


class TestLlamaCppManagerRequests(unittest.TestCase):
    def setUp(self):
        self.spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        self.seed = manager_seed("llamacpp", 3, 0, 0)

    def _manager(self, base_url, policy=FAST):
        env = dict(NO_KEYS, LLAMACPP_BASE_URL=base_url)
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return LangGraphManager(model="llamacpp/qwen-manager", transport_retry=policy)

    def test_seed_reaches_the_request_and_the_llm_call_record(self):
        with _Server(["ok"]) as url:
            proposal = _propose(self._manager(url), self.spec)  # no /v1 in the configured URL
        requests = _LlamaServerStandIn.requests
        self.assertEqual([r["path"] for r in requests], ["/v1/chat/completions"])
        self.assertEqual(requests[0]["body"]["seed"], self.seed)
        self.assertEqual(requests[0]["body"]["model"], "qwen-manager")
        meta = proposal.extra["llm_call"]
        self.assertTrue(meta["parse_ok"])
        self.assertEqual(meta["seed"], self.seed)
        self.assertEqual((meta["prompt_tokens"], meta["completion_tokens"]), (7, 5))

    def test_dropped_request_is_retried_once_with_the_same_seed(self):
        with _Server(["drop", "ok"]) as url:
            proposal = _propose(self._manager(url + "/v1"), self.spec)
        requests = _LlamaServerStandIn.requests
        # Exactly two requests: the harness retry, and no hidden retry inside the OpenAI client.
        self.assertEqual([r["body"]["seed"] for r in requests], [self.seed, self.seed])
        attempts = proposal.extra["llm_call"]["transport_attempts"]
        self.assertEqual([a["error_class"] for a in attempts], ["server_disconnected", None])
        self.assertIn(attempts[0]["error_type"], CONNECTION_ERROR_TYPES)

    def test_server_5xx_is_a_transport_error(self):
        with _Server(["503", "503", "503"]) as url:
            with self.assertRaises(Exception) as ctx:
                _propose(self._manager(url), self.spec)
        self.assertEqual(len(_LlamaServerStandIn.requests), 3)
        self.assertEqual([a["error_class"] for a in ctx.exception.llm_call_attempts], ["server_5xx"] * 3)
        self.assertEqual(ctx.exception.llm_call_seed, self.seed)

    def test_connection_refused_is_a_transport_error(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]  # closed again: nothing listens there
        with self.assertRaises(Exception) as ctx:
            _propose(self._manager(f"http://127.0.0.1:{port}"), self.spec)
        attempts = ctx.exception.llm_call_attempts
        self.assertEqual(len(attempts), 3)
        self.assertEqual({a["error_class"] for a in attempts}, {"connection_refused"})  # as for Ollama
        self.assertLessEqual({a["error_type"] for a in attempts}, CONNECTION_ERROR_TYPES)

    def test_openai_timeouts_and_client_errors_are_not_retried(self):
        import httpx
        import openai

        request = httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
        self.assertIsNone(classify_transport_error(openai.APITimeoutError(request=request)))
        response = httpx.Response(404, request=request)
        self.assertIsNone(classify_transport_error(
            openai.NotFoundError("not found", response=response, body=None)))
        response = httpx.Response(500, request=request)
        self.assertEqual(classify_transport_error(
            openai.InternalServerError("boom", response=response, body=None)), "server_5xx")


if __name__ == "__main__":
    unittest.main()
