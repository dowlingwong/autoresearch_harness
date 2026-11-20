"""Frozen evaluation of node mlgym_breakout_minatar (runs in the frozen parent; R1-HARNESS-ISOLATION).

FROZEN.  :func:`rollout_batch` is MLGym's ``data/rlBreakoutMinAtar/evaluate.py`` rollout (CC BY-NC 4.0; adapted
from gymnax-blines) written as a frame loop: 256 environments reset from one PRNG key, actions sampled from a
categorical distribution over the policy's logits, rewards summed until each episode ends, exactly 999 frames
(``MAX_FRAMES``).  The random keys are split as in MLGym's ``scan_step``, and the action is drawn with
``tfp.distributions.Categorical(logits=...).sample`` as MLGym's baseline policy draws it.  Only the logits come
from the editable network: the parent asks the worker for them each frame (*policy_logits*), and the
environment, the keys, the sampling and the sum run here, in the process that never imports editable code.
:func:`mean_return` averages over a list of keys.  The parent calls it with MLGym's test keys ``PRNGKey(0..4)``
for the sealed test metric and with keys derived from the split seed for the validation metric.
"""
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

BATCH_SIZE = 256
MAX_FRAMES = 999


def make_rollout(env, env_params):
    """jitted reset / sample / step functions for :func:`rollout_batch`."""

    @jax.jit
    def reset(rng_key):
        keys = jax.random.split(rng_key, BATCH_SIZE)
        return jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

    @jax.jit
    def act_and_step(carry, logits):
        obs, env_state, rng, cumulative_rewards, valid_mask = carry
        rng, rng_act = jax.random.split(rng)
        actions = tfp.distributions.Categorical(logits=logits).sample(seed=rng_act)
        rng, rng_step = jax.random.split(rng)
        rng_steps = jax.random.split(rng_step, BATCH_SIZE)
        next_obs, next_env_state, reward, done, _info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_steps, env_state, actions, env_params)
        new_cumulative_rewards = cumulative_rewards + reward * valid_mask
        new_valid_mask = valid_mask * (1 - done)
        return next_obs, next_env_state, rng, new_cumulative_rewards, new_valid_mask

    return reset, act_and_step


def rollout_batch(rollout, policy_logits, rng_key, n_actions: int, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Returns of the 256 episodes started from *rng_key*; *policy_logits(obs)* -> (256, n_actions) array."""
    reset, act_and_step = rollout
    obs, env_state = reset(rng_key)
    carry = (obs, env_state, rng_key, jnp.zeros(BATCH_SIZE), jnp.ones(BATCH_SIZE))
    for _ in range(max_frames):
        logits = np.asarray(policy_logits(np.asarray(carry[0], dtype=np.float32)), dtype=np.float32)
        if logits.shape != (BATCH_SIZE, n_actions):
            raise ValueError(f"policy logits must have shape ({BATCH_SIZE}, {n_actions}), got {logits.shape}")
        if not np.all(np.isfinite(logits)):
            raise ValueError("policy logits are not finite")
        carry = act_and_step(carry, jnp.asarray(logits))
    return np.asarray(carry[3])


def mean_return(rollout, policy_logits, keys, n_actions: int) -> float:
    rewards = []
    for key in keys:
        rewards.extend(rollout_batch(rollout, policy_logits, key, n_actions).tolist())
    return float(np.mean(rewards))
