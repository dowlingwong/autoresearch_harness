"""R1-CACHEPROMPT-FALSE (31 Oct 2026): ``cache_prompt`` is explicit on every llama-server request.

llama-server's prompt cache is on by default: with it, the slot reuses the previous request's KV prefix,
so identical (prompt, seed) calls are not bit-identical and an S0.1-style determinism check fails. The
harness therefore sends ``cache_prompt`` explicitly, defaulting to ``False``, and records the value used.

These tests drive the real ``ChatOpenAI`` client (no inference) against a stand-in for llama-server and
check every request path available at this commit:

* the manager proposal path (``LangGraphManager``), including a transport retry;
* the side-query / proposal-backend path (``LangChainProposalBackend._invoke``);
* the value reaching the ``llm_call`` record and the request payload the S0.1 probe inspects.

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

from autoresearch.llm.langchain_client import LangChainProposalBackend, _build_chat_model
from autoresearch.llm.providers import (
    LLAMACPP_CACHE_PROMPT_ENV,
    llamacpp_cache_prompt,
    manager_response_format,
    resolve_llm_config,
)
from autoresearch.llm.transport_retry import TransportRetryPolicy
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager, manager_seed
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[2]
GOOD = json.dumps({"summary": "s", "rationale": "r", "objective": "o", "param": "x", "new_value": "1"})
FAST = TransportRetryPolicy(max_attempts=3, backoff_seconds=(0.0, 0.0))
NO_KEYS = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "LLAMACPP_API_KEY", "OPENAI_BASE_URL", "LLAMACPP_BASE_URL",
                        LLAMACPP_CACHE_PROMPT_ENV)}


class _StandIn(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/v1/chat/completions``; records every request body."""

    protocol_version = "HTTP/1.1"
    requests: list = []
    script: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        cls = type(self)
        cls.requests.append({"path": self.path, "body": body})
        action = cls.script[len(cls.requests) - 1] if len(cls.requests) <= len(cls.script) else "ok"
        if action == "drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        data = json.dumps({"id": "x", "object": "chat.completion", "created": 1,
                           "model": body.get("model"),
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": GOOD}}],
                           "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server:
    def __init__(self, script=("ok",)):
        _StandIn.requests = []
        _StandIn.script = list(script)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class TestCachePromptConfig(unittest.TestCase):
    def test_llamacpp_default_is_false(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            self.assertIs(resolve_llm_config("llamacpp/qwen").cache_prompt, False)
            self.assertIs(llamacpp_cache_prompt(), False)

    def test_env_override_true(self):
        with mock.patch.dict(os.environ, dict(NO_KEYS, **{LLAMACPP_CACHE_PROMPT_ENV: "true"}), clear=True):
            self.assertIs(resolve_llm_config("llamacpp/qwen").cache_prompt, True)

    def test_env_rejects_a_non_boolean(self):
        with mock.patch.dict(os.environ, dict(NO_KEYS, **{LLAMACPP_CACHE_PROMPT_ENV: "yes-please"}), clear=True):
            with self.assertRaises(ValueError):
                resolve_llm_config("llamacpp/qwen")

    def test_other_providers_do_not_carry_the_field(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            for model in ("ollama/qwen", "vllm/qwen", "openai/gpt-4.1", "deepseek/deepseek-v4"):
                self.assertIsNone(resolve_llm_config(model).cache_prompt, model)

    def test_the_model_carries_the_field_in_extra_body(self):
        # ``extra_body`` is what the OpenAI SDK merges into the JSON body. It is deliberately NOT in
        # ``model_kwargs``: those become ``create()`` keyword arguments and an unknown name raises.
        from langchain_core.messages import HumanMessage
        with mock.patch.dict(os.environ, dict(NO_KEYS, LLAMACPP_BASE_URL="http://127.0.0.1:9/v1"), clear=True):
            llm = _build_chat_model(resolve_llm_config("llamacpp/qwen"), 0.2, seed=7,
                                    response_format=manager_response_format())
        payload = llm._get_request_payload([HumanMessage(content="hi")])
        self.assertIs(payload["extra_body"]["cache_prompt"], False)
        self.assertNotIn("cache_prompt", payload)      # it must not become a create() kwarg
        self.assertIn("response_format", payload)


class TestCachePromptOnEveryRequestPath(unittest.TestCase):
    def setUp(self):
        self.spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        self.seed = manager_seed("cacheprompt", 3, 0, 0)

    def _manager(self, base_url, policy=FAST, extra_env=None):
        env = dict(NO_KEYS, LLAMACPP_BASE_URL=base_url, **(extra_env or {}))
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return LangGraphManager(model="llamacpp/qwen-manager", transport_retry=policy)

    def _propose(self, manager):
        status = ManagerStatus(campaign_id="cacheprompt", budget_index=3, current_best_metric=0.5,
                               metric_name=self.spec.metric_name,
                               metric_direction=self.spec.metric_direction)
        ctx = build_memory_context([], MemoryMode.NONE, self.spec, 3)
        return manager.propose_next_trial(status, ctx, self.spec)

    def test_manager_proposal_carries_cache_prompt_false(self):
        with _Server() as url:
            proposal = self._propose(self._manager(url))
        requests = _StandIn.requests
        self.assertEqual([r["path"] for r in requests], ["/v1/chat/completions"])
        self.assertIs(requests[0]["body"]["cache_prompt"], False)
        # the value used lands in the llm_call record
        self.assertIs(proposal.extra["llm_call"]["cache_prompt"], False)

    def test_transport_retry_attempts_all_carry_it(self):
        with _Server(["drop", "ok"]) as url:
            proposal = self._propose(self._manager(url))
        self.assertEqual(len(_StandIn.requests), 2)
        self.assertEqual([r["body"]["cache_prompt"] for r in _StandIn.requests], [False, False])
        self.assertIs(proposal.extra["llm_call"]["cache_prompt"], False)

    def test_env_override_reaches_the_request_and_the_record(self):
        with _Server() as url:
            proposal = self._propose(self._manager(url, extra_env={LLAMACPP_CACHE_PROMPT_ENV: "true"}))
        self.assertIs(_StandIn.requests[0]["body"]["cache_prompt"], True)
        self.assertIs(proposal.extra["llm_call"]["cache_prompt"], True)

    def test_injected_llm_records_nothing(self):
        # An injected stub never sends the field, so the record must omit the key rather than claim a value.
        from types import SimpleNamespace
        manager = LangGraphManager(model="llamacpp/qwen-manager", transport_retry=FAST,
                                   llm=SimpleNamespace(invoke=lambda messages, **kw: SimpleNamespace(content=GOOD)))
        proposal = self._propose(manager)
        self.assertNotIn("cache_prompt", proposal.extra["llm_call"])

    def test_manager_proposal_schema_variant_carries_it_on_the_wire(self):
        # C3 uses the manager_proposal/v1 response_format; the field must ride along with it.
        from autoresearch.llm.providers import manager_response_format
        from types import SimpleNamespace
        schema_manager = LangGraphManager(model="llamacpp/qwen-manager", transport_retry=FAST,
                                          proposal_schema="manager_proposal/v1")
        with _Server() as url:
            env = dict(NO_KEYS, LLAMACPP_BASE_URL=url)
            with mock.patch.dict(os.environ, env, clear=True):
                self._propose(schema_manager)
        body = _StandIn.requests[0]["body"]
        self.assertIs(body["cache_prompt"], False)
        self.assertEqual(body["response_format"]["json_schema"]["name"], "manager_proposal_v1")

    def test_the_llm_call_event_persists_the_value(self):
        # R7: not just proposal.extra -- the emitted llm_call event payload must carry it.
        import json as _json
        from autoresearch.control_plane.campaign import _emit_llm_call
        from autoresearch.manager.base import ManagerProposal

        class _Store:
            def __init__(self):
                self.events = []

            def append(self, event):
                self.events.append(event)

        store = _Store()
        proposal = ManagerProposal(manager_mode="langgraph_manager", proposal_summary="s",
                                   proposal_rationale="r", target_files=[], objective="o",
                                   extra={"llm_call": {"seed": 1, "cache_prompt": False}})
        _emit_llm_call(store, campaign_id="c", trial_id="t", proposal=proposal)
        event = store.events[-1]
        self.assertEqual(event.event_type, "llm_call")
        self.assertIs(event.payload["cache_prompt"], False)

    def test_side_query_backend_carries_it(self):
        # The proposal-backend / side-query path builds its own model from the same config.
        with _Server() as url:
            with mock.patch.dict(os.environ, dict(NO_KEYS, LLAMACPP_BASE_URL=url), clear=True):
                backend = LangChainProposalBackend("llamacpp/qwen", temperature=0.2)
                text = backend._invoke("hello")
        self.assertIn("summary", text)
        self.assertIs(_StandIn.requests[0]["body"]["cache_prompt"], False)


if __name__ == "__main__":
    unittest.main()
