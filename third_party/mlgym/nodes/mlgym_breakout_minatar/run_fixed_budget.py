"""Frozen parent of node mlgym_breakout_minatar (R1-HARNESS-ISOLATION; started by the runner as ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code (networks.py, config.yaml).  Task: MLGym
rlBreakoutMinAtar (gymnax ``Breakout-MinAtar``).  The frozen worker ``ar_worker.py`` trains one model with the
frozen PPO (policy.py), the editable network and hyperparameters, from AR_TRAIN_SEED exactly as MLGym's
src/train.py does for its first checkpoint (MLGym trains five checkpoints per run; the port trains one), for
``--steps`` environment steps (MLGym's rounding: ``(steps // num_train_envs + 1) * num_train_envs``, printed
by the worker as ``AR_STEPS env_steps=``), with the loss log ``step <env steps> loss <PPO total loss>``.
Evaluation runs here: MLGym's rollout (evaluate_policy.py) with the environment, the keys, the action sampling
and the sums in this process and only the policy logits from the worker, frame by frame -- validation keys
derived from AR_SPLIT_SEED for the selection metric ``AR_METRIC val_return=``, MLGym's test keys
PRNGKey(0..4) for the sealed ``AR_TEST test_return=``, both written to the runner's result channel.  JAX runs
on the CPU in this process (the environment only); the worker keeps the GPU, with deterministic GPU ops and
without pre-allocating memory, and prints ``AR_INFRA …`` and exits 6 before importing the editable network when
a GPU was assigned but JAX's first device is not a GPU (the node's infrastructure pattern).
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import jax  # noqa: E402

jax.config.update("jax_platforms", "cpu")  # this process only steps the environment; the worker keeps the GPU

import numpy as np  # noqa: E402

import ar_parent as P  # noqa: E402
import evaluate_policy  # noqa: E402

ENV_NAME = "Breakout-MinAtar"
TEST_SEEDS = (0, 1, 2, 3, 4)
N_VAL_KEYS = 5


def validation_keys(split_seed: int):
    base = jax.random.PRNGKey(split_seed)
    keys = [jax.random.fold_in(base, 1 + k) for k in range(N_VAL_KEYS)]
    test = {tuple(np.asarray(jax.random.PRNGKey(s)).tolist()) for s in TEST_SEEDS}
    if any(tuple(np.asarray(k).tolist()) in test for k in keys):
        P.fail(P.EXIT_BAD_INVOCATION, "a validation key equals a test key")
    return keys


def main() -> None:
    import gymnax

    budget = P.read_budget("steps")
    train_seed, split_seed = P.read_seeds(default_train=42, split_note="")
    env, env_params = gymnax.make(ENV_NAME)
    n_actions = int(env.action_space(env_params).n)
    rollout = evaluate_policy.make_rollout(env, env_params)
    val_keys = validation_keys(split_seed)
    test_keys = [jax.random.PRNGKey(s) for s in TEST_SEEDS]

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed})

    def policy_logits(obs):
        _, arrays = P.call(worker, "act_logits", {}, {"obs": obs})
        return P.check_array(arrays, "logits", (evaluate_policy.BATCH_SIZE, n_actions)).astype(np.float32)

    try:
        val = evaluate_policy.mean_return(rollout, policy_logits, val_keys, n_actions)
        test = evaluate_policy.mean_return(rollout, policy_logits, test_keys, n_actions)
    except ValueError as error:
        P.fail(P.EXIT_BAD_RESULT, f"evaluation: {error}")
    P.stop_worker(worker)
    CHANNEL.write(f"AR_METRIC val_return={val:.6f}")
    CHANNEL.write(f"AR_TEST test_return={test:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
