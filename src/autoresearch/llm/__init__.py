"""LLM provider resolution and optional LangChain proposal backend."""

from autoresearch.llm.providers import LLMConfig, resolve_llm_config, resolve_worker_model_args

__all__ = ["LLMConfig", "resolve_llm_config", "resolve_worker_model_args"]
