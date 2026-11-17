"""Frozen fixed-step wrapper of node mlgym_breakout_minatar.

Run command: ``python3 run_fixed_budget.py --steps {budget}``.

FROZEN.  Task: MLGym rlBreakoutMinAtar (gymnax ``Breakout-MinAtar``), PPO from
the frozen policy.py with the editable network (networks.py) and
hyperparameters (config.yaml).  Budget: ``--steps`` environment steps, set as
``num_train_steps``; the frozen loop takes ``(steps // num_train_envs + 1) *
num_train_envs`` steps (MLGym's rounding) and the wrapper prints that number
as ``AR_STEPS env_steps=``.  Seeds: one model per run, initialised and trained
from AR_TRAIN_SEED exactly as MLGym's src/train.py does for its first
checkpoint (MLGym trains five checkpoints per run; the port trains one).
Evaluation: MLGym's rollout (evaluate_policy.py) on the final parameters --
validation keys derived from AR_SPLIT_SEED for the selection metric
``AR_METRIC val_return=``, MLGym's test keys PRNGKey(0..4) for the sealed
``AR_TEST test_return=``.  Loss log: ``step <env steps> loss <PPO total loss>``
after every update (policy.py's progress bar is disabled so these lines start
at the beginning of a line).  JAX runs with deterministic GPU ops and without
pre-allocating GPU memory (set below, before jax is imported).  When the runner
assigns a GPU (non-empty CUDA_VISIBLE_DEVICES) and JAX's first device is not a
GPU, the wrapper prints ``AR_INFRA …`` and exits 6 before the editable code is
imported (the node's infrastructure pattern).  The result lines (AR_STEPS,
AR_METRIC, AR_TEST) are written to a copy of the standard output descriptor taken
before the editable code is imported, and the run ends with ``os._exit(0)`` after
them.
"""
import argparse
import os
import sys

_OUT_FD = os.dup(1)  # the wrapper's result lines; taken before the editable code is imported

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import functools  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import tqdm  # noqa: E402

# policy.py draws a tqdm progress bar; its carriage returns would hide the loss lines from the
# outcome parser's line-anchored patterns, so every bar is created disabled.
tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True)

ENV_NAME = "Breakout-MinAtar"
WRAPPER_KEYS = ("num_train_steps", "evaluate_every_epochs", "num_test_rollouts")
TEST_SEEDS = (0, 1, 2, 3, 4)
N_VAL_KEYS = 5
EXIT_BAD_INVOCATION = 2
EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_INFRA = 6


def emit(line: str) -> None:
    """One of the wrapper's result lines, on the saved descriptor (editable code cannot rewrite it through sys.stdout)."""
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    os.write(_OUT_FD, (line + "\n").encode("utf-8"))


def seal() -> None:
    """Point descriptors 1 and 2 at /dev/null just before the final result lines (they use the saved copy), so
    threads of the editable code cannot print into the log while they are written."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def finish() -> None:
    """End the run after the last result line: no atexit handler or later output of the editable code runs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(0)


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def read_budget() -> int:
    parser = argparse.ArgumentParser(description="Track G fixed-step wrapper (environment steps)")
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        fail(EXIT_BAD_INVOCATION, f"--steps must be positive, got {args.steps}")
    if os.environ.get("AR_BUDGET_KIND", "steps") != "steps":
        fail(EXIT_BAD_INVOCATION, "AR_BUDGET_KIND must be steps")
    value = os.environ.get("AR_BUDGET_VALUE")
    if value is not None and int(value) != args.steps:
        fail(EXIT_BAD_INVOCATION, f"--steps {args.steps} disagrees with AR_BUDGET_VALUE={value}")
    return args.steps


def read_seeds() -> tuple[int, int]:
    raw_train, raw_split = os.environ.get("AR_TRAIN_SEED"), os.environ.get("AR_SPLIT_SEED")
    train = int(raw_train) if raw_train is not None else 42
    split = int(raw_split) if raw_split is not None else 0
    print(f"AR_SEEDS train={train}{'' if raw_train is not None else ' (default, MLGym seed)'} "
          f"split={split}{'' if raw_split is not None else ' (default)'}", flush=True)
    return train, split


def validation_keys(split_seed: int):
    base = jax.random.PRNGKey(split_seed)
    keys = [jax.random.fold_in(base, 1 + k) for k in range(N_VAL_KEYS)]
    test = {tuple(np.asarray(jax.random.PRNGKey(s)).tolist()) for s in TEST_SEEDS}
    if any(tuple(np.asarray(k).tolist()) in test for k in keys):
        fail(EXIT_BAD_INVOCATION, "a validation key equals a test key")
    return keys


def main() -> None:
    budget = read_budget()
    train_seed, split_seed = read_seeds()
    device = jax.devices()[0]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() and device.platform != "gpu":
        # A GPU was assigned but JAX runs elsewhere (wrong UUID, CPU-only jax): a silent CPU fallback is a
        # STOP for the census.  This runs before the editable network is imported, so no proposal causes it.
        print("AR_INFRA a GPU was assigned (CUDA_VISIBLE_DEVICES set) but CUDA is unavailable", flush=True)
        sys.exit(EXIT_INFRA)
    print(f"AR_DEVICE {device.platform} {getattr(device, 'device_kind', '')} jax={jax.__version__} "
          f"workers={os.environ.get('AR_NUM_WORKERS', '0')}", flush=True)

    import gymnax
    from helpers import load_config
    import evaluate_policy
    import policy
    import networks  # the editable network

    config = load_config("config.yaml", train_seed).train_config
    present = [k for k in WRAPPER_KEYS if k in config]
    if present:
        fail(EXIT_BAD_EDITABLE_API, f"config.yaml must not set {present} (owned by the wrapper)")
    config.num_train_steps = budget
    config.evaluate_every_epochs = 0
    envs = int(config.num_train_envs)
    expected_steps = (budget // envs + 1) * envs

    # MLGym src/train.py: rng_list = split(PRNGKey(seed), 6); training rng = rng_list[0];
    # the first checkpoint is initialised from rng_list[1].
    rng_list = jax.random.split(jax.random.PRNGKey(config["seed_id"]), 6)
    model, params = networks.get_model_ready(rng_list[1], config)

    def on_update(total_steps, metrics):
        print(f"step {total_steps} loss {float(np.asarray(metrics['total_loss'])):.6f}", flush=True)

    _, _, final_params, total_steps = policy.train_ppo(rng_list[0], config, model, params, None,
                                                       on_update=on_update)
    if total_steps != expected_steps or total_steps < budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"took {total_steps} environment steps, expected {expected_steps}")
    emit(f"AR_STEPS env_steps={total_steps}")

    env, env_params = gymnax.make(ENV_NAME)
    val = evaluate_policy.mean_return(env, env_params, model, final_params, validation_keys(split_seed))
    test = evaluate_policy.mean_return(env, env_params, model, final_params,
                                       [jax.random.PRNGKey(s) for s in TEST_SEEDS])
    # Both rollouts (editable network) run before the first result line; nothing editable runs after it.
    seal()  # stdout/stderr to /dev/null: threads of the editable code cannot print now
    emit(f"AR_METRIC val_return={val:.6f}")
    emit(f"AR_TEST test_return={test:.6f}")


if __name__ == "__main__":
    main()
    finish()  # no output of the editable code after the result lines
