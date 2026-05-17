from __future__ import annotations

from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

from autoresearch.llm.langchain_client import LangChainProposalBackend
from autoresearch.llm.langchain_client import _build_chat_model
from autoresearch.llm.providers import LLMConfig
from autoresearch.llm.providers import resolve_llm_config, resolve_worker_model_args
from autoresearch.manager.base import ManagerStatus
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node


def test_resolve_ollama_prefix(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    cfg = resolve_llm_config("ollama/qwen2.5-coder:7b")
    assert cfg.provider == "ollama"
    assert cfg.model_name == "qwen2.5-coder:7b"
    assert cfg.base_url == "http://ollama.test:11434"
    assert cfg.api_key == ""


def test_resolve_local_provider_defaults(monkeypatch) -> None:
    for key in ("VLLM_BASE_URL", "LMSTUDIO_BASE_URL", "LLAMACPP_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert resolve_llm_config("vllm/Qwen/Qwen2.5").base_url == "http://localhost:8000"
    assert resolve_llm_config("lm_studio/local-model").base_url == "http://localhost:1234/v1"
    assert resolve_llm_config("llamacpp/model.gguf").base_url == "http://localhost:8080"


def test_resolve_cloud_provider_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    assert resolve_llm_config("openai/gpt-4.1").api_key == "openai-key"
    assert resolve_llm_config("anthropic/claude-sonnet").api_key == "anthropic-key"
    deepseek = resolve_llm_config("deepseek/deepseek-v4-flash")
    assert deepseek.provider == "deepseek"
    assert deepseek.model_name == "deepseek-v4-flash"
    assert deepseek.base_url == "https://api.deepseek.com"
    assert deepseek.api_key == "deepseek-key"


def test_resolve_deepseek_base_url_override(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/v1")
    cfg = resolve_llm_config("deepseek/deepseek-v4-pro")
    assert cfg.provider == "deepseek"
    assert cfg.base_url == "https://deepseek.example/v1"


def test_build_deepseek_rejects_invalid_thinking_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_THINKING", "maybe")
    cfg = LLMConfig(
        provider="deepseek",
        model_name="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_id="deepseek/deepseek-v4-flash",
    )
    try:
        _build_chat_model(cfg, temperature=0.2)
    except ValueError as exc:
        assert "DEEPSEEK_THINKING" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid DEEPSEEK_THINKING to raise")


def test_no_prefix_preserves_legacy_model_and_host() -> None:
    cfg = resolve_llm_config("qwen2.5-coder:7b")
    assert cfg.provider == "legacy"
    assert cfg.model_name == "qwen2.5-coder:7b"
    assert cfg.base_url == ""
    assert resolve_worker_model_args("qwen2.5-coder:7b", "http://legacy-host") == (
        "qwen2.5-coder:7b",
        "http://legacy-host",
    )


def test_provider_model_strips_worker_prefix(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    assert resolve_worker_model_args("ollama/qwen2.5-coder:7b", "http://ignored") == (
        "qwen2.5-coder:7b",
        "http://ollama.test:11434",
    )


def test_langchain_backend_validates_structured_response_and_writes_artifacts(tmp_path) -> None:
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    '{"summary": "reduce-lr", "rationale": "stabilize training", '
                    '"objective": "In train.py, reduce LR once. Edit only train.py."}'
                )
            )

    spec = load_registered_node("resnet_trigger")
    status = ManagerStatus("lc", 1, None, spec.metric_name, spec.metric_direction)
    memory = build_memory_context([], MemoryMode.NONE, spec, 1)
    backend = LangChainProposalBackend(
        "ollama/qwen2.5-coder:7b",
        artifacts_dir=tmp_path / "artifacts",
        llm=FakeLLM(),
    )
    proposal = backend.propose_next_trial(status, memory, spec)
    assert proposal.manager_mode == "langchain_backend"
    assert proposal.proposal_summary == "reduce-lr"
    assert proposal.target_files == spec.editable_paths
    assert "langchain_prompt_ref" in proposal.extra
    assert (tmp_path / "artifacts" / "trial-001" / "langchain" / "prompt.txt").exists()


def test_run_campaign_langchain_backend_writes_same_schema_as_native(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    native_records = tmp_path / "ledgers" / "native_trials.jsonl"
    langchain_records = tmp_path / "ledgers" / "langchain_trials.jsonl"

    base_cmd = [
        sys.executable,
        str(root / "scripts" / "run_campaign.py"),
        "--node",
        "resnet_trigger",
        "--budget",
        "1",
        "--memory-mode",
        "none",
        "--dry-run",
        "--tables-dir",
        str(tmp_path / "tables"),
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
    ]
    native = subprocess.run(
        [
            *base_cmd,
            "--campaign-id",
            "native_schema",
            "--manager",
            "baseline_manager",
            "--records",
            str(native_records),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    langchain = subprocess.run(
        [
            *base_cmd,
            "--campaign-id",
            "langchain_schema",
            "--manager",
            "baseline_manager",
            "--llm-backend",
            "langchain",
            "--llm-stub",
            "--model",
            "ollama/qwen2.5-coder:7b",
            "--records",
            str(langchain_records),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert langchain.returncode == 0, langchain.stderr

    native_row = json.loads(native_records.read_text(encoding="utf-8").splitlines()[0])
    langchain_row = json.loads(langchain_records.read_text(encoding="utf-8").splitlines()[0])
    assert set(native_row) == set(langchain_row)
    assert langchain_row["manager_mode"] == "langchain_backend"
    assert langchain_row["extra"]["manager"]["llm_backend"] == "langchain"
