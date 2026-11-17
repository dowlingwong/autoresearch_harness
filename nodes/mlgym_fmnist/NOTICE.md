# Node mlgym_fmnist — origin, licence and changes

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates), task `imageClassificationFMnist`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below.

**Changes made by the port.**
- `train.py` (editable) is adapted from MLGym's `baseline.py`: the same CNN and the Hugging Face Trainer
  defaults the baseline relies on, written out as an explicit AdamW optimizer and linear schedule; the
  Trainer, its per-epoch test evaluation and its best-checkpoint reload are removed.
- Written by the port: `run_fixed_budget.py`, `ar_fixed_loop.py`, `ar_data.py`, `prepare_data.py`
  (fixed step budget, validation split carved from the training set, metric lines, hashed data manifest).
  MLGym's `evaluate.py` (test accuracy of a submission file) is replaced by the frozen test-accuracy line.
- The task data are not part of the node; `prepare_data.py` fetches them from their public source (its own
  terms apply) and records their hashes.
