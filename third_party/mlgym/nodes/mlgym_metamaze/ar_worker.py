"""Frozen worker entry of node mlgym_metamaze (R1-HARNESS-ISOLATION): the only process that imports the
editable network (networks.py) and reads config.yaml.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the runner's
result channel.  ``train``: check the device, load config.yaml (the keys the frozen code owns must be absent),
initialise and train one model with the frozen PPO (policy.py) for the budget, as MLGym's src/train.py does for
its first checkpoint; ``act_logits``: the policy's logits for a batch of observations (the parent samples the
actions and steps the environment).
"""
import os

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import ar_isolated as iso  # noqa: E402
CONN = iso.worker_connect()

import functools  # noqa: E402
import sys  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import tqdm  # noqa: E402

import ar_data  # noqa: E402

NODE = "mlgym_metamaze"
ASSET_FILES = ("config.yaml", "__init__.py", "evaluate.py", "helpers.py", "networks.py", "policy.py",
               "requirements.txt", "rlMetaMaze.yaml", "train.py")

# policy.py draws a tqdm progress bar; its carriage returns would hide the loss lines from the
# outcome parser's line-anchored patterns, so every bar is created disabled.
tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True)

WRAPPER_KEYS = ("num_train_steps", "evaluate_every_epochs", "num_test_rollouts")
EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_INFRA = 6
STATE = {}


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    device = jax.devices()[0]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() and device.platform != "gpu":
        # A GPU was assigned but JAX runs elsewhere (wrong UUID, CPU-only jax): a silent CPU fallback is a
        # STOP for the census.  This runs before the editable network is imported, so no proposal causes it.
        print("AR_INFRA a GPU was assigned (CUDA_VISIBLE_DEVICES set) but CUDA is unavailable", flush=True)
        sys.exit(EXIT_INFRA)
    print(f"AR_DEVICE {device.platform} {getattr(device, 'device_kind', '')} jax={jax.__version__} "
          f"workers={os.environ.get('AR_NUM_WORKERS', '0')}", flush=True)

    ar_data.verify(ar_data.data_dir_of(__file__), NODE, ASSET_FILES)

    from helpers import load_config
    import policy
    import networks  # the editable network

    config = load_config("config.yaml", train_seed).train_config
    present = [k for k in WRAPPER_KEYS if k in config]
    if present:
        fail(EXIT_BAD_EDITABLE_API, f"config.yaml must not set {present} (owned by the frozen code)")
    config.num_train_steps = budget
    config.evaluate_every_epochs = 0
    envs = int(config.num_train_envs)
    expected_steps = (budget // envs + 1) * envs

    # MLGym src/train.py: rng_list = split(PRNGKey(seed), 6); training rng = rng_list[0];
    # the first checkpoint is initialised from rng_list[1].
    rng_list = jax.random.split(jax.random.PRNGKey(config["seed_id"]), 6)
    model, params = networks.get_model_ready(rng_list[1], config)

    def on_update(total_steps, metrics):
        ar_data.emit(f"step {total_steps} loss {float(np.asarray(metrics['total_loss'])):.6f}")

    _, _, final_params, total_steps = policy.train_ppo(rng_list[0], config, model, params, None,
                                                       on_update=on_update)
    if total_steps != expected_steps or total_steps < budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"took {total_steps} environment steps, expected {expected_steps}")
    ar_data.emit(f"AR_STEPS env_steps={total_steps}")

    @jax.jit
    def logits_of(obs):
        _value, pi = model.apply(final_params, obs, rng=None)
        return pi.logits_parameter()

    STATE["logits_of"] = logits_of
    return {"env_steps": int(total_steps)}, {}


def act_logits(meta, arrays, fds):
    obs = np.asarray(arrays["obs"], dtype=np.float32)
    return {}, {"logits": np.asarray(STATE["logits_of"](obs), dtype=np.float32)}


iso.serve({"train": train, "act_logits": act_logits}, CONN)
