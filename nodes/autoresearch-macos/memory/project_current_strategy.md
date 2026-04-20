---
name: Current strategy
description: Current operating strategy for the autoresearch manager/worker system and how to use the control plane safely.
type: project
---

## Operating model

The control plane is a Python CLI (`python3 -m src.main autoresearch ...`) that manages the full experiment lifecycle. The manager LLM (Claude Code / claw) issues commands; the worker LLM (Ollama or any OpenAI-compat endpoint) edits `train.py`.

## Key invariants

- Only `train.py` is ever modified by the worker. `prepare.py` is read-only infrastructure.
- Every experiment runs for exactly 300 seconds (5-minute wall-clock budget via `TIME_BUDGET` in `prepare.py`).
- `val_bpb` (bits per byte) is the single comparison metric — it is vocabulary-size-independent.
- A candidate is `keep` if `val_bpb < best_bpb`, otherwise `discard`. Crash → `crash`.
- The branch `autoresearch/<tag>` isolates each experiment session from `master`.

## Control-plane command sequence

```
setup → isolate → baseline → [run → keep/discard] × N
```

Each command is idempotent where possible (setup, baseline skip if already done).

## Timeout budget (updated)

- Training wall clock: 300 s (hardcoded in `prepare.py`)
- Experiment subprocess timeout: 360 s (20% buffer above training budget)
- Acceptance test timeout: 300 s (syntax check only)

## Worker backend selection

The worker can use Ollama native API (`--backend ollama`) or any OpenAI-compatible endpoint (`--backend openai-compat`). Default worker is `qwen2.5-coder:7b` via Ollama.

## REST API

The control plane is also exposed as a local HTTP JSON API (`api-server` subcommand, port 7331 by default). This allows any LLM with HTTP tool-use to drive the loop without Python imports.

## Near-term priorities

1. Let the manager reason about `results.tsv` trend before generating the next experiment objective.
2. Accumulate failed-idea memory so the worker is not given the same dead-end hypothesis twice.
3. Evaluate larger worker models (34B+) for more substantive `train.py` edits.
