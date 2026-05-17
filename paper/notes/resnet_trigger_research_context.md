# Research Context: resnet_trigger

This file is a deterministic pre-campaign snapshot for manager and audit metadata.

## Node Spec

```json
{
  "acceptance_rule": "candidate_metric > current_best_metric",
  "default_budget": {
    "max_wall_clock_hours": 12.0,
    "trials": 50
  },
  "description": "Near-threshold detector waveform binary-classification benchmark using the ResNet_trigger node.",
  "editable_paths": [
    "train.py"
  ],
  "expected_runtime": "fast smoke: minutes on CPU; full campaign: overnight",
  "failure_categories": [
    "syntax_error",
    "runtime_error",
    "edit_failed",
    "proposal_precondition_failed",
    "effective_config_unchanged",
    "metric_missing",
    "invalid_edit_scope",
    "degraded_metric",
    "no_op_patch"
  ],
  "frozen_paths": [
    "prepare.py",
    "program.md",
    "pyproject.toml",
    "uv.lock",
    "resnet_1d.py",
    "signal_vacuum_sum_crop_4000x8000.h5",
    "noise_traces_4000x8000.h5",
    "artifacts/",
    "data/"
  ],
  "metric_direction": "maximize",
  "metric_name": "val_auc",
  "metric_parser": "autoresearch.nodes.resnet_trigger.metric_parser:parse_val_auc",
  "name": "resnet_trigger",
  "run_command": "uv run train.py > run.log 2>&1",
  "setup_command": "uv sync",
  "validity_checks": [
    "metric_present",
    "finite_metric",
    "editable_scope_only",
    "no_data_pipeline_modification",
    "command_exit_zero"
  ]
}
```

## Source Notes

### configs/nodes/resnet_trigger.yaml

```text
{
  "name": "resnet_trigger",
  "description": "Near-threshold detector waveform binary-classification benchmark using the ResNet_trigger node.",
  "editable_paths": [
    "train.py"
  ],
  "frozen_paths": [
    "prepare.py",
    "program.md",
    "pyproject.toml",
    "uv.lock",
    "resnet_1d.py",
    "signal_vacuum_sum_crop_4000x8000.h5",
    "noise_traces_4000x8000.h5",
    "artifacts/",
    "data/"
  ],
  "setup_command": "uv sync",
  "run_command": "uv run train.py > run.log 2>&1",
  "metric_name": "val_auc",
  "metric_direction": "maximize",
  "metric_parser": "autoresearch.nodes.resnet_trigger.metric_parser:parse_val_auc",
  "acceptance_rule": "candidate_metric > current_best_metric",
  "validity_checks": [
    "metric_present",
    "finite_metric",
    "editable_scope_only",
    "no_data_pipeline_modification",
    "command_exit_zero"
  ],
  "default_budget": {
    "trials": 50,
    "max_wall_clock_hours": 12
  },
  "expected_runtime": "fast smoke: minutes on CPU; full campaign: overnight",
  "failure_categories": [
    "syntax_error",
    "runtime_error",
    "edit_failed",
    "proposal_precondition_failed",
    "effective_config_unchanged",
    "metric_missing",
    "invalid_edit_scope",
    "degraded_metric",
    "no_op_patch"
  ]
}
```

### nodes/ResNet_trigger/program.md

```text
# ResNet Trigger Autoresearch Program

This node turns the exploratory notebook `cnn_smoke_train_round1.ipynb` into a governed autoresearch target.

## Scope

- Modify only `train.py`.
- Read-only files: `prepare.py`, `README.md`, `program.md`, the H5 source files.
- Do not add dependencies or new source files during the worker loop.

## Goal

Improve the trigger-classification model on the signal-vs-noise task.

The raw task metrics are:

- `val_auc` / `val_roc_auc` — primary task metric, higher is better
- `val_pr_auc` — secondary task metric, higher is better

The current autoresearch harness expects a lower-is-better scalar named `val_bpb`, so this node reports:

```text
val_bpb = 1 - best_val_auc
```

This is only a compatibility alias for the manager/control-plane layer. The real scientific interpretation should use AUC directly.

## Baseline workflow

1. Establish the baseline with the current `train.py`.
2. Propose one bounded `train.py` change.
3. Run the experiment.
4. Keep the change only if the compatibility objective improves:
   - lower `val_bpb`
   - equivalently higher `best_val_auc`
5. If the objective regresses or the run crashes, discard and revert.

During each run, save the checkpoint and metrics from the epoch with the highest validation AUC.

## Good experiment types

- learning-rate and weight-decay adjustments
- batch-size changes
- kernel-size, channel-width, and block-depth changes
- dropout changes
- residual block simplifications
- normalization and activation changes that stay inside `train.py`

## Bad experiment types

- changing the dataset files
- changing split logic in `prepare.py`
- adding complex new infrastructure
- editing anything outside `train.py`

## Output contract

The script prints a parseable summary block:

```text
---
val_bpb:          ...
training_seconds: ...
total_seconds:    ...
peak_vram_mb:     ...
...
val_auc:          ...
val_pr_auc:       ...
val_roc_auc:      ...
```

The manager uses `val_bpb` for keep/discard decisions. Humans should inspect `val_auc`, `val_roc_auc`, and the saved best checkpoint as the meaningful task outputs.

## Simplicity rule

Prefer smaller, cleaner changes. A tiny gain that adds a lot of complexity is usually not worth keeping. A similar result with simpler code is valuable.
```

### nodes/ResNet_trigger/README.md

```text
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
```
