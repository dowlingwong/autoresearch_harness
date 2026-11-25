# Node `mlgym_metamaze` — MLGym rlMetaMazeMisc (Track G T1 expansion candidate 3)

Frozen. This README is the node's design record (checklist F); the port kit branch is
`mac-trackg-mountaincar-20261125` on the harness repository, based on the accepted T1 TASL harness
`mac-trackg-t1-tasl-20261121` @ `c025f89a`.

## 1. Which candidate, and why MetaMaze

R1-T1-NODE-EXPANSION candidate 3 is *"MLGym MountainCarContinuous or MetaMaze"*.  The upstream MLGym task
`rlMountainCarContinuous` was checked first and **rejected for a noise floor**, then `rlMetaMazeMisc` was
ported.  The decision is recorded here before any measurement (ENGINEERING_ONLY; nothing below is a census
result).

**MountainCarContinuous is a solved/not-solved metric.**  An episode ends when the car reaches the goal, and
the +100 bonus is paid once, on that terminal step.  A solved episode returns ≈ +95 (100 minus the action
penalty over ~100–199 steps); an unsolved episode is truncated (gymnax `max_steps_in_episode = 999`, MLGym's
evaluation rolls 200 frames) and returns a negative value proportional to the sum of squared actions, with no
other signal.  There is no partial credit and no graded progress: the aggregate return is a two-mode mixture,
i.e. an affine transform of the solved fraction.  This matches the earlier U3 judgement
(`reports/TRACKG_U3_PORT_MLGYM_20261117.md`: "MountainCarContinuous tends to end solved or not solved, a poor
fit for a noise floor").  Two further checks at the fixed budget class this node must fit (R1-BX-D5: 1–2 min
runs; a probe at 2,000,000 environment steps is the same number of PPO updates a GPU run at that budget
takes):

- MLGym's own task record gives baseline `Reward Mean 33.7938`, `Reward Std 41.7073` over 6,400 individual
  evaluation episodes — the standard deviation exceeds the mean, the signature of the two-point mixture.
- A CPU probe of the MLGym PPO at 2,000,000 environment steps (30 PPO updates, MLGym's own configuration):
  every evaluation episode stayed in the unsolved mode — per-episode returns −25.5 to −217.3, mean −160.3,
  standard deviation 19.2, **no episode solved**.  The "progress" a seed could show at this budget is action
  magnitude (a penalty term), not task progress; a noise floor read there would be noise in an uninformative
  direction.  The port would have had to raise the budget by an order of magnitude to reach the solve
  transition, where the metric is again binary.

**MetaMaze's metric is a graded goal count.**  In `MetaMaze-misc` the goal is re-drawn each time the agent
reaches it and the episode continues to 200 steps (MLGym evaluates 199 frames), with +10 per goal, no step
penalty and no terminal bonus.  The return is therefore 10 × (number of goals reached) over the evaluation
window — partial progress is a graded value, not a pass/fail.  Same CPU probe method:

| probe (CPU, MLGym PPO, at 2,000,000 env steps unless noted) | return mean | std | min–max | episodes | distribution |
|---|---|---|---|---|---|
| random initialisation (0 updates, 2 eval seeds) | 1.41 | 3.90 | 0–30 | 512 | 447×0, 59×10, 5×20, 1×30 |
| trained, seed 42 (1 eval seed) | 8.13 | 12.30 | 0–50 | 256 | 151×0, 53×10, 18×20, 20×30, 11×40, 3×50 |
| trained, seed 43 (2 eval seeds) | 7.91 | 9.69 | 0–70 | 512 | 247×0, 167×10, 66×20, 25×30, 6×40, 1×70 |

The trained-at-budget policy is above the random start and below MLGym's published baseline
(mean 15.73, trained with 10,000,000 steps per checkpoint × 5 checkpoints), so the fixed budget leaves
headroom, and the graded count changes smoothly with policy quality.  That is the property the noise floor
needs.  ENGINEERING_ONLY: these are probe numbers from the port, not a noise floor or a census.

## 2. Design

| Item | Value |
|---|---|
| Node | `mlgym_metamaze` |
| Suite / domain | MLGym / RL |
| Upstream task | `rlMetaMazeMisc` (`configs/tasks/rlMetaMaze.yaml`, environment `gymnax` `MetaMaze-misc`), commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0` |
| Editable | `config.yaml` (MLGym's hyperparameters minus the three wrapper-owned keys), `networks.py` (MLGym's actor-critic network) |
| Frozen | `run_fixed_budget.py` (isolated parent), `ar_worker.py` (only process importing editable code), `ar_data.py`, `ar_parent.py`, `ar_isolated.py` (master `37d99ab0…`), `policy.py` (MLGym PPO with the reviewed port changes), `helpers.py`, `evaluate_policy.py` (MLGym rollout in the parent), `prepare_data.py`, `requirements.txt`, `requirements_gpu.txt`, `LICENCES.md`, `README.md`, `data/`, `representative_edits/` |
| Data | none (simulated); setup downloads the pinned MLGym task/package snapshot (9 files, sha256-pinned) and records the package pins in `data_manifest.json`, outside the repository |
| Selection metric | `val_return` (maximise), `AR_METRIC val_return=<float>`, validation keys derived from `AR_SPLIT_SEED` (5 keys) |
| Sealed test | `test_return`, `AR_TEST test_return=<float>`, MLGym's test keys `PRNGKey(0..4)`; never selected |
| Loss / steps lines | `step <env steps> loss <PPO total loss>`, `AR_STEPS env_steps=<n>` |
| Budget | `run_budget: {kind: steps, value: 2000000}` — environment steps (MLGym's rounding `(steps // num_train_envs + 1) * num_train_envs`), no wall-clock key anywhere (R1-BX-D5) |
| Workers / determinism | `dataloader_workers: 0`, `threads: 4`; determinism defaults (`cudnn_benchmark: false`, `deterministic_algorithms: true`, `:4096:8`, `PYTHONHASHSEED=0`); the JAX worker sets `XLA_FLAGS --xla_gpu_deterministic_ops=true`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `TF_CUDNN_DETERMINISTIC=1` |
| Isolation | `isolation.entry: run_fixed_budget.py`, `args: ["--steps", "{budget}"]`; the parent holds the runner's result channel and the worker imports the editable code (R1-HARNESS-ISOLATION) |
| Environment | `requirements.txt` = the port's local CPU check (jax[cpu] 0.4.35, flax 0.10.2, optax 0.2.4, chex 0.1.88, gymnax 0.0.9, tensorflow-probability 0.25.0, dotmap 1.3.30, tqdm 4.70.1, numpy 2.5.4); `requirements_gpu.txt` = the same pins with jax[cuda12] for the GPU host |

**One model per run.**  MLGym's `src/train.py` trains five checkpoints (five seeds) and its evaluator scores
all five.  The port trains one model, initialised exactly as MLGym's first checkpoint (the training seed's
`PRNGKey`, `rng_list[1]`), at the fixed budget; the evaluation keeps MLGym's procedure (5 keys × 256
environments; only the test keys are the sealed set).

**Budget semantics.**  The count is environment steps taken by the PPO loop, printed as `AR_STEPS env_steps=<n>` and
checked by the worker against MLGym's rounding; the parent cross-checks `metric.steps_regex` against the node's
budget.  An edit can change compute per step (network width/depth, batch composition) but not the step count;
the evaluation is frozen and does not count toward the budget.  The budget was proposed from a CPU measurement
(the baseline's 2,000,000 steps took 51 s of training on four cores) and must be measured by the U2 census on
the census GPU; **changing `run_budget.value` changes `node_spec_sha256` and the edits manifest**, so it needs
re-declared edits and a new review (checklist A).

**What is not enforced.**  As in the other MLGym nodes, the editable code runs inside the worker process: the
PPO loop is frozen, but an edit to `networks.py` can change compute per step (e.g. widen the network), and an
edit to `config.yaml` can change sampling/optimisation settings within the keys MLGym exposes.  The budget
bounds environment steps, not FLOPs.  The frozen parent computes the return from the worker's logits with the
environment, the keys and the sums in the parent, and the data check binds the run to the pinned snapshot.

## 3. Files

`config.yaml`, `networks.py` (editable); `run_fixed_budget.py`, `ar_worker.py`, `ar_parent.py`,
`ar_isolated.py`, `ar_data.py`, `policy.py`, `helpers.py`, `evaluate_policy.py`, `prepare_data.py`,
`requirements.txt`, `requirements_gpu.txt`, `LICENCES.md`, `README.md`, `representative_edits/` (four declared
diffs + manifest) — all frozen.  The node spec is
`third_party/mlgym/configs/trackg/nodes/mlgym_metamaze.yaml`.

## 4. Provenance

MLGym, task `rlMetaMazeMisc`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`; CC BY-NC 4.0, with the
PPO/rollout parts adapted by MLGym from gymnax-blines/gymnax (Apache-2.0).  The full licence, scope and change
record is in `LICENCES.md`.  ENGINEERING_ONLY: no GPU run, no census, no noise floor, no label.
