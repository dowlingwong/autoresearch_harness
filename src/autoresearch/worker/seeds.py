"""Seed injection for workers that run experiment nodes (BUILD_RUNBOOK L3.4).

The control plane passes a per-run training seed and (where a node's seed also
fixes its data split) a separate split seed to the worker.  Workers forward
them to the node subprocess as the environment variables ``AR_TRAIN_SEED`` and
``AR_SPLIT_SEED``.  Node scripts read them with their current constants as
defaults, so behaviour is unchanged when the variables are unset.

These names are never editable symbols and never appear in a proposal.
"""
from __future__ import annotations

import os
from typing import Mapping

TRAIN_SEED_ENV = "AR_TRAIN_SEED"
SPLIT_SEED_ENV = "AR_SPLIT_SEED"
SEED_ENV_VARS = (TRAIN_SEED_ENV, SPLIT_SEED_ENV)


def seed_environment(
    train_seed: int | str | None = None,
    split_seed: int | str | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment with the seed variables set.

    *base* defaults to ``os.environ``.  Passing ``None`` for a seed leaves the
    corresponding variable at whatever the base environment already had, so an
    unset seed is identical to today's behaviour.
    """
    env: dict[str, str] = dict(os.environ if base is None else base)
    if train_seed is not None:
        env[TRAIN_SEED_ENV] = str(train_seed)
    if split_seed is not None:
        env[SPLIT_SEED_ENV] = str(split_seed)
    return env
