# Node mlgym_breakout_minatar — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §3 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates; HEAD at U1 §3 and at the port), task `rlBreakoutMinAtar`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** Per the repository's `LICENSE` file (U1 §3), MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below. MLGym's README notes that SWE-Agent and Modded-NanoGPT are MIT-licensed and Gymnax and
gymnax-blines are Apache-2.0; CC BY-NC 4.0 grants no patent rights (U1 §3). Attribution: "Adapted from MLGym
(Nathani et al., 2025), CC BY-NC 4.0; modified as listed below".

**Scope within the harness repository.** R1-TRACKG-PORT-DECISIONS (1): this node lives in `third_party/mlgym/`, whose `LICENSE` is MLGym's CC BY-NC 4.0, and the root `LICENSE` and `README` state the scope. The repository's MIT licence does not cover this node's MLGym-derived files: they stay under CC BY-NC 4.0 (parts that MLGym took from gymnax or gymnax-blines under Apache-2.0). The files written by the port (below) are the repository's own, under MIT.

**Changes made by the port.**
- `networks.py` (editable) and `helpers.py` (frozen) are MLGym's files, unchanged apart from a docstring note.
- `config.yaml` (editable) is MLGym's file without `num_train_steps`, `evaluate_every_epochs` and
  `num_test_rollouts`, which the frozen wrapper owns.
- `policy.py` (frozen) is MLGym's file with three changes in `train_ppo` (listed in its docstring): an update
  callback for the loss log, in-training evaluation switchable off, and the step count returned.
- `evaluate_policy.py` (frozen) is MLGym's `evaluate.py` rollout as a frame loop in the frozen parent (only the
  policy logits come from the worker; R1-HARNESS-ISOLATION); written by the port: `ar_worker.py`, `ar_parent.py`,
  `ar_isolated.py`, `ar_data.py`,
  `run_fixed_budget.py` (one model per run instead of five checkpoints, fixed environment-step budget,
  validation keys from the split seed, MLGym's test keys 0-4 for the sealed test return).
- MLGym's PPO code is adapted from gymnax-blines and gymnax (R. T. Lange, Apache-2.0).

## Data

None: the environment is simulated by `gymnax` (`Breakout-MinAtar`). Gymnax and gymnax-blines are
**Apache-2.0** per MLGym's README (U1 §3). Attribution: "PPO and evaluation adapted by MLGym from gymnax-blines
and gymnax (R. T. Lange), Apache-2.0". A copy of the Apache License 2.0 ships as `third_party/mlgym/LICENSE-APACHE-2.0`
(Apache-2.0 §4(a)); it applies to those parts of `policy.py`, `helpers.py`, `networks.py` and `evaluate_policy.py`.

No container is used (native ports): no image digest applies. The node and its branch contain no task data; the
port's tests use synthetic data.
