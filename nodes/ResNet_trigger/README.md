# ResNet Trigger Worker Node

This directory converts the exploratory notebook [cnn_smoke_train_round1.ipynb](/Users/wongdowling/Documents/autoresearch_harness/nodes/ResNet_trigger/cnn_smoke_train_round1.ipynb) into an autoresearch-compatible worker node.

It follows the same control-plane shape as `nodes/autoresearch-macos`:

- `prepare.py` — fixed data-loading and split utilities
- `train.py` — the only file the worker should modify
- `program.md` — worker instructions
- `pyproject.toml` — node-local dependencies

## What changed from the notebook

The original notebook was a single manual experiment round:

- fixed config in notebook cells
- direct H5 loading inside the notebook
- repeated multi-run loop
- no control-plane integration
- no single-file editable training surface

This node turns that into a proper experiment target:

- reproducible train/val/test splits from `prepare.py`
- a single-run `train.py` entrypoint
- parseable output for the existing autoresearch harness
- artifacts written to `artifacts/`
- isolated git repo semantics for keep/discard later

## Objective

The true task metrics are:

- `best_val_auc` / `best_val_roc_auc` — primary, higher is better
- `best_val_pr_auc` — secondary, higher is better

The current manager layer expects a lower-is-better scalar called `val_bpb`, so this node reports:

```text
val_bpb = 1 - best_val_auc
```

This is a compatibility shim so the current claw/autoresearch control plane can rank candidate runs without deeper code changes.

## Quick start

From this directory:

```bash
uv sync
uv run train.py
```

If the run succeeds, `train.py` prints a summary block ending with:

```text
---
val_bpb:          ...
training_seconds: ...
total_seconds:    ...
peak_vram_mb:     ...
val_auc:          ...
val_pr_auc:       ...
val_roc_auc:      ...
best_model_path:  ...
```

Artifacts are written to `artifacts/`, including:

- `best_model.pt` — checkpoint from the highest validation AUC epoch
- `best_performance.json` — best-epoch metrics
- `history_latest.json`, `metrics_latest.json`, `timing_latest.json`

## Later integration with claw

Once this node is in a clean isolated git branch or nested repo, it can be used with the current Python control plane by pointing the API server at this directory:

```bash
python3 -m src.main api-server --root /path/to/nodes/ResNet_trigger
```

The important compatibility requirements are already satisfied:

- `prepare.py` exists
- `train.py` exists
- `program.md` exists
- `train.py` prints a parseable lower-is-better objective

## Current limitation

This node keeps the current harness assumption that only `train.py` is edited. If later research requires changing split logic or data transforms, the harness contract would need to widen beyond the current stage-one worker model.
