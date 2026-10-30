"""Track S (30 Oct 2026): the opt-in llama-server ``cache_prompt`` request field.

S0.1 found that repeated manager calls with the same prompt and D26 seed were not bit-identical
on a single llama-server slot (serial identity 0.93 / 0.87). The working hypothesis is that the
server's default prompt caching reuses the slot's KV prefix from the previous request, so the
computation depends on the predecessor. ``cache_prompt=False`` asks the server to re-process the
whole prompt on every call.

These tests drive the real ``ChatOpenAI`` client against the llama-server stand-in of
``test_r1_llamacpp_provider`` and check that:

* by default no ``cache_prompt`` field is sent, so the historical request body is unchanged;
* ``cache_prompt=False`` / ``True`` reach the HTTP body at the top level (not nested under
  ``extra_body``), alongside the seed and the ``manager_proposal/v1`` response format;
* the value is recorded in the ``llm_call`` metadata only when set;
* other providers and non-bool values are refused.

ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import manager_response_format, resolve_llm_config
from autoresearch.manager.langgraph_manager import LangGraphManager

from tests.stage2.test_r1_llamacpp_provider import (FAST, NO_KEYS, ROOT, _LlamaServerStandIn, _propose,
                                                    _Server)
from autoresearch.nodes.registry import load_registered_node


class TestCachePromptRequestBody(unittest.TestCase):
    def setUp(self):
        self.spec = load_registered_node("lr_synthetic", repo_root=ROOT)

    def _manager(self, base_url, **kwargs):
        patcher = mock.patch.dict(os.environ, dict(NO_KEYS, LLAMACPP_BASE_URL=base_url), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return LangGraphManager(model="llamacpp/qwen-manager", transport_retry=FAST, **kwargs)

    def test_default_sends_no_cache_prompt_field(self):
        with _Server(["ok"]) as url:
            proposal = _propose(self._manager(url), self.spec)
        body = _LlamaServerStandIn.requests[0]["body"]
        self.assertNotIn("cache_prompt", body)
        self.assertNotIn("extra_body", body)
        self.assertNotIn("cache_prompt", proposal.extra["llm_call"])

    def test_false_and_true_reach_the_top_level_of_the_body(self):
        for value in (False, True):
            with self.subTest(cache_prompt=value):
                with _Server(["ok"]) as url:
                    proposal = _propose(self._manager(url, cache_prompt=value,
                                                      proposal_schema="manager_proposal/v1"), self.spec)
                body = _LlamaServerStandIn.requests[0]["body"]
                self.assertIs(body["cache_prompt"], value)
                self.assertNotIn("extra_body", body)
                self.assertIn("seed", body)
                self.assertEqual(body["response_format"], manager_response_format("manager_proposal/v1"))
                self.assertIs(proposal.extra["llm_call"]["cache_prompt"], value)

    def test_retry_resends_the_same_cache_prompt(self):
        with _Server(["drop", "ok"]) as url:
            _propose(self._manager(url, cache_prompt=False), self.spec)
        bodies = [r["body"] for r in _LlamaServerStandIn.requests]
        self.assertEqual([b["cache_prompt"] for b in bodies], [False, False])
        self.assertEqual(bodies[0]["seed"], bodies[1]["seed"])

    def test_from_config_passes_cache_prompt(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w") as fh:
                json.dump({"model": "llamacpp/qwen-manager", "cache_prompt": False}, fh)
            self.assertIs(LangGraphManager.from_config(path)._cache_prompt, False)


class TestCachePromptValidation(unittest.TestCase):
    def test_build_chat_model_refuses_other_providers(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            with self.assertRaises(ValueError):
                _build_chat_model(resolve_llm_config("ollama/qwen2.5-coder:7b"), 0.2, seed=1, cache_prompt=False)

    def test_non_bool_values_are_refused(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            for bad in ("false", 0, 1):
                with self.subTest(value=bad):
                    with self.assertRaises(TypeError):
                        _build_chat_model(resolve_llm_config("llamacpp/qwen"), 0.2, seed=1, cache_prompt=bad)
                    with self.assertRaises(TypeError):
                        LangGraphManager(model="llamacpp/qwen", cache_prompt=bad)

    def test_manager_refuses_non_llamacpp_models_at_construction(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            for model in ("ollama/qwen2.5-coder:7b", "qwen2.5-coder:7b", "openai/gpt-4.1"):
                with self.subTest(model=model):
                    with self.assertRaises(ValueError):
                        LangGraphManager(model=model, cache_prompt=False)
            LangGraphManager(model="ollama/qwen2.5-coder:7b")   # unset: unchanged

    def test_injected_llm_records_no_cache_prompt(self):
        # An injected LLM (the --llm-stub path) is not built by _build_chat_model, so nothing is sent and
        # nothing may be recorded.
        from tests.stage2.test_r1_d26_seed import _SeedRecordingLLM, _response
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        manager = LangGraphManager(llm=_SeedRecordingLLM(_response()), cache_prompt=False)
        proposal = _propose(manager, spec)
        self.assertNotIn("cache_prompt", proposal.extra["llm_call"])

    def test_unset_leaves_the_client_without_extra_body(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            llm = _build_chat_model(resolve_llm_config("llamacpp/qwen"), 0.2, seed=1)
            self.assertFalse(getattr(llm, "extra_body", None))
            llm = _build_chat_model(resolve_llm_config("llamacpp/qwen"), 0.2, seed=1, cache_prompt=False)
            self.assertEqual(llm.extra_body, {"cache_prompt": False})


if __name__ == "__main__":
    unittest.main()
