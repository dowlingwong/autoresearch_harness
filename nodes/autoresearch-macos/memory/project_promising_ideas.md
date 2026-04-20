---
name: Promising ideas
description: Promising next steps for memory, orchestration, and experiment quality in autoresearch.
type: project
---

## Trend-aware keep/discard

The current `_recommend_status()` is a pure threshold check (`val_bpb < best_bpb`). A moving-average or regression over the last N results could distinguish genuine improvements from noise, especially when `val_bpb` differences are in the 4th decimal place.

## Lane-aware memory

Partition `experiment_memory.jsonl` by experiment "lane" (e.g. attention variants, optimizer changes, data augmentation) so the manager can load only the relevant lane's history when generating the next objective, rather than the full event log.

## SQLite for structured experiment store

Replace the TSV + JSONL dual-file approach with a single SQLite database. Enables efficient queries like "show all keep decisions where peak_vram_mb < 8000" without scanning the full file. The current append-only files would become a SQLite WAL.

## Living research wiki

Extend the `memory/` topic files into a richer wiki that links specific `train.py` functions to the experiments that modified them. Each function (e.g. `CausalSelfAttention`, `MLP`) would have a page tracking: current implementation, what was tried, what the best variant was.

## REST API as the primary driver interface

Now that the `api-server` subcommand exists, the manager LLM can drive the full loop via HTTP tool-use without any Python installation. This opens up using the Rust `claw` binary (Claude-backed) as the manager while keeping the Python layer for all state bookkeeping.

## Larger worker models

`qwen2.5-coder:7b` is conservative for the complexity of `train.py` edits. Testing `qwen2.5-coder:32b` or `deepseek-coder-v2` via the `--backend openai-compat` flag (llama.cpp or LM Studio) could yield more substantive architectural changes.

## Parallel experiment lanes

The current architecture is strictly serial. With multiple MPS contexts or time-sliced execution, multiple branches could run experiments in parallel and merge only the winners back to the main experiment branch.
