"""Frozen evaluation of node mlgym_breakout_minatar.

FROZEN.  ``rollout_batch`` is MLGym's ``data/rlBreakoutMinAtar/evaluate.py``
rollout (CC BY-NC 4.0; adapted from gymnax-blines): 256 environments reset
from one PRNG key, actions sampled from the policy, rewards summed until each
episode ends, at most 999 frames.  :func:`mean_return` averages it over a
list of keys.  The wrapper calls it with MLGym's test keys
``PRNGKey(0..4)`` for the sealed test metric and with keys derived from the
split seed for the validation metric.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

BATCH_SIZE = 256
MAX_FRAMES = 999


@partial(jax.jit, static_argnums=(0, 1, 2))
def rollout_batch(env, env_params, model, model_params, rng_key, max_frames=MAX_FRAMES):
    batch_size = BATCH_SIZE

    def scan_step(carry, _):
        obs, env_state, rng, cumulative_rewards, valid_mask = carry
        rng, rng_act = jax.random.split(rng)
        v_batch, pi_batch = model.apply(model_params, obs, rng=None)
        actions = pi_batch.sample(seed=rng_act)
        rng, rng_step = jax.random.split(rng)
        rng_steps = jax.random.split(rng_step, batch_size)
        next_obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_steps, env_state, actions, env_params
        )
        new_cumulative_rewards = cumulative_rewards + reward * valid_mask
        new_valid_mask = valid_mask * (1 - done)
        return [next_obs, next_env_state, rng, new_cumulative_rewards, new_valid_mask], [new_valid_mask]

    rng_reset_keys = jax.random.split(rng_key, batch_size)
    obs_batch, env_state_batch = jax.vmap(env.reset, in_axes=(0, None))(rng_reset_keys, env_params)
    rewards_batch = jnp.array(batch_size * [0.0])
    valid_mask = jnp.array(batch_size * [1.0])
    init_carry = [obs_batch, env_state_batch, rng_key, rewards_batch, valid_mask]
    carry_out, _ = jax.lax.scan(scan_step, init_carry, (), length=max_frames)
    return carry_out[-2].squeeze()


def mean_return(env, env_params, model, params, keys) -> float:
    rewards = []
    for key in keys:
        rewards.extend(np.asarray(rollout_batch(env, env_params, model, params, key)).tolist())
    return float(np.mean(rewards))
