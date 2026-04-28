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
