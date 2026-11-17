# Node mlgym_breakout_minatar — origin, licence and changes

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates), task `rlBreakoutMinAtar`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below.

**Changes made by the port.**
- `networks.py` (editable) and `helpers.py` (frozen) are MLGym's files, unchanged apart from a docstring note.
- `config.yaml` (editable) is MLGym's file without `num_train_steps`, `evaluate_every_epochs` and
  `num_test_rollouts`, which the frozen wrapper owns.
- `policy.py` (frozen) is MLGym's file with three changes in `train_ppo` (listed in its docstring): an update
  callback for the loss log, in-training evaluation switchable off, and the step count returned.
- `evaluate_policy.py` (frozen) is MLGym's `evaluate.py` rollout as a function; written by the port:
  `run_fixed_budget.py` (one model per run instead of five checkpoints, fixed environment-step budget,
  validation keys from the split seed, MLGym's test keys 0-4 for the sealed test return).
- MLGym's PPO code is adapted from gymnax-blines and gymnax (R. T. Lange, Apache-2.0).
