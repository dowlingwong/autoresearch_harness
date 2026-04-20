---
name: External references
description: External sources, tools, or future systems that are relevant to the autoresearch memory and orchestration roadmap.
type: reference
---

## Upstream project

- **Karpathy/autoresearch** — the upstream repo this node is forked from. Tracks the original experiment loop design and the `val_bpb` metric convention.

## Models in use

- **qwen2.5-coder:7b** (Ollama) — default worker model for `train.py` edits. Reliable JSON tool-call protocol compliance.
- **claude-sonnet-4-6 / claude-opus-4-6** — manager model when running via the `claw` Rust CLI or Claude Code directly.

## Key papers / techniques in train.py

- **GQA (Grouped Query Attention)** — implemented in `CausalSelfAttention`; reduces KV cache size. Configurable via `n_kv_heads`.
- **RoPE (Rotary Position Embeddings)** — applied in `CausalSelfAttention.forward()`.
- **Muon optimizer** — the primary optimizer in the training loop; combined with AdamW for embedding parameters.
- **Sliding window attention** — controlled by the `"SSSL"` pattern in `train.py`.
- **MFU (Model FLOP Utilization)** — reported as `mfu_percent` in the run log summary block.

## Tooling

- **uv** — Python package manager used to run experiments (`uv run train.py`).
- **rustbpe / tiktoken** — BPE tokenizer libraries used by `prepare.py`.
- **Ollama** — local LLM server for the worker; native API at `:11434`.
- **llama.cpp server** — alternative OpenAI-compat worker server; default port 8080.
- **LM Studio** — alternative OpenAI-compat worker server; default port 1234.

## Harness

- **claw-code (rust/)** — Rust CLI agent backed by Anthropic/OpenAI API. Intended as the manager when a frontier model is needed. See `rust/USAGE.md` for build instructions.
- **autoresearch REST API** — `python3 -m src.main api-server` on port 7331. Allows any HTTP-capable LLM to drive the research loop.
