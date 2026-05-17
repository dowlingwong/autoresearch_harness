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


_LOCAL_PROVIDERS = {
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434", ""),
    "vllm": ("VLLM_BASE_URL", "http://localhost:8000", "VLLM_API_KEY"),
    "lm_studio": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1", "LMSTUDIO_API_KEY"),
    "llamacpp": ("LLAMACPP_BASE_URL", "http://localhost:8080", "LLAMACPP_API_KEY"),
}

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
        return LLMConfig(
            provider=prefix,
            model_name=model_name,
            base_url=os.environ.get(base_env, default_base_url),
            api_key=api_key,
            model_id=raw,
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
