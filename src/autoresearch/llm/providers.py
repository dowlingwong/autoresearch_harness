from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model_name: str
    base_url: str
    api_key: str
    model_id: str
    # R1-CACHEPROMPT-FALSE (31 Oct 2026): llama-server's prompt cache is on by default and makes
    # run-to-run output non-identical, so the value is now explicit. ``None`` = the provider does not
    # carry the setting (everything except ``llamacpp``).
    cache_prompt: bool | None = None


MANAGER_PROPOSAL_SCHEMA_VERSION = "manager_proposal/v1"


def manager_response_format(version: str = MANAGER_PROPOSAL_SCHEMA_VERSION) -> dict:
    """Structure only: D25 key/value validity remains the model's responsibility."""
    if version != MANAGER_PROPOSAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported proposal schema {version!r}")
    fields = ("rationale", "summary", "objective", "param", "old_value", "new_value")
    return {"type": "json_schema", "json_schema": {
        "name": "manager_proposal_v1", "strict": True,
        "schema": {"type": "object", "properties": {k: {"type": "string"} for k in fields},
                   "required": list(fields), "additionalProperties": False}}}


_LOCAL_PROVIDERS = {
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434", ""),
    "vllm": ("VLLM_BASE_URL", "http://localhost:8000", "VLLM_API_KEY"),
    "lm_studio": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1", "LMSTUDIO_API_KEY"),
    "llamacpp": ("LLAMACPP_BASE_URL", "http://localhost:8080/v1", "LLAMACPP_API_KEY"),
}

# Placeholder key for local OpenAI-compatible servers that need none (llama-server
# without ``--api-key``): the OpenAI client refuses to start without some key.
LOCAL_NO_KEY = "sk-no-key-required"

# R1-CACHEPROMPT-FALSE (31 Oct 2026): ``cache_prompt`` is sent explicitly on every llama-server
# request. The default is ``False`` because llama-server's prompt cache otherwise reuses the slot's KV
# prefix from the previous request, so identical (prompt, seed) calls are not bit-identical. Set
# ``LLAMACPP_CACHE_PROMPT=true`` to opt back into the server default.
LLAMACPP_CACHE_PROMPT_ENV = "LLAMACPP_CACHE_PROMPT"


def llamacpp_cache_prompt() -> bool:
    """The ``cache_prompt`` value for the llamacpp provider (default ``False``)."""
    raw = os.environ.get(LLAMACPP_CACHE_PROMPT_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{LLAMACPP_CACHE_PROMPT_ENV} must be a boolean (true/false), got {raw!r}")


def openai_v1_base_url(base_url: str) -> str:
    """llama-server's OpenAI-compatible routes live under ``/v1`` (R1 A.10, 25 Oct 2026).

    ``ChatOpenAI`` appends ``/chat/completions`` to its base URL, so a bare
    ``http://host:port`` gets ``/v1``; a URL already ending in ``/v1`` is kept.
    """
    url = base_url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"

_CLOUD_PROVIDERS = {
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_BASE_URL", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "deepseek": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


def resolve_llm_config(model_id: str) -> LLMConfig:
    """Resolve a provider-prefixed model id into client settings.

    Supported examples:
      - ``ollama/qwen2.5-coder:7b``
      - ``vllm/Qwen/Qwen2.5-Coder-7B``
      - ``lm_studio/local-model``
      - ``llamacpp/model.gguf``
      - ``openai/gpt-4.1``
      - ``anthropic/claude-3-5-sonnet-latest``
      - ``deepseek/deepseek-v4-flash``

    A bare model id is returned unchanged for backward compatibility.
    """
    raw = model_id.strip()
    if not raw:
        raise ValueError("model_id must not be empty")

    if "/" not in raw:
        return LLMConfig(
            provider="legacy",
            model_name=raw,
            base_url="",
            api_key="",
            model_id=raw,
        )

    prefix, model_name = raw.split("/", 1)
    if not model_name:
        raise ValueError(f"model id is missing provider-specific model name: {model_id!r}")

    if prefix in _LOCAL_PROVIDERS:
        base_env, default_base_url, key_env = _LOCAL_PROVIDERS[prefix]
        api_key = os.environ.get(key_env, "") if key_env else ""
        base_url = os.environ.get(base_env, default_base_url)
        if prefix == "llamacpp":
            base_url = openai_v1_base_url(base_url)
        return LLMConfig(
            provider=prefix,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model_id=raw,
            cache_prompt=llamacpp_cache_prompt() if prefix == "llamacpp" else None,
        )

    if prefix in _CLOUD_PROVIDERS:
        base_env, default_base_url, key_env = _CLOUD_PROVIDERS[prefix]
        return LLMConfig(
            provider=prefix,
            model_name=model_name,
            base_url=os.environ.get(base_env, default_base_url),
            api_key=os.environ.get(key_env, ""),
            model_id=raw,
        )

    raise ValueError(
        f"unsupported LLM provider prefix {prefix!r}; expected one of "
        "ollama, vllm, lm_studio, llamacpp, openai, anthropic, deepseek"
    )


def resolve_worker_model_args(model_id: str, legacy_host: str) -> tuple[str, str]:
    """Return ``(model_name, host)`` for current Ollama-compatible workers.

    Bare model ids preserve the old ``--model`` + ``--host`` behavior. Provider
    ids use the resolver base URL. The current worker backend is Ollama-style,
    so non-local cloud providers are intentionally rejected here.
    """
    cfg = resolve_llm_config(model_id)
    if cfg.provider == "legacy":
        return cfg.model_name, legacy_host
    if cfg.provider in {"ollama", "vllm", "lm_studio", "llamacpp"}:
        return cfg.model_name, cfg.base_url
    raise ValueError(f"{cfg.provider!r} is not supported by the current worker backend")
