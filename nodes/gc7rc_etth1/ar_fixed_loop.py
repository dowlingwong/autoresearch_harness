"""Frozen fixed-step training loop for the Track G forecasting node gc7rc_etth1 (trackg_node/v1).

FROZEN: listed in the node's frozen_paths; a diff that touches it is refused.  It is the node's own copy of
the shared torch loop's design (``tests/trackg/port_support.py`` compares only the files listed in
``SHARED_FROZEN``; this forecasting variant is not that classification loop): the step budget, the seeds, the
order of the training windows, the loss log and the evaluation conversion are owned here, not by the editable
code.

The editable ``train.py`` provides

  CONFIG                       dict; ``batch_size`` (int) is required, optional
                               ``grad_clip`` (float or None) and ``amp_bf16`` (bool)
  build_model(info)            -> torch.nn.Module (info: device, seq_len, horizon, n_features)
  build_optimizer(model, total_steps)
                               -> (optimizer, scheduler or None); the scheduler is
                                  stepped once after every optimizer step
  training_step(model, batch, step)
                               -> scalar loss tensor for one batch ({"x": (B, L, C), "y": (B, H, C)})
  predict(model, batch)        -> plain tensor (B, horizon, n_features) for an evaluation batch
                                  (a dict {"x": (B, L, C)} without the targets)

This module owns everything a diff must not change in the training: exactly ``--steps`` optimizer steps
(counted on the optimizer itself), the seeds, the order of the training windows and the loss log the outcome
parser reads (``step <n> loss <mean over the last LOG_EVERY steps>``).

R1-HARNESS-ISOLATION: it runs in the node's **worker** process (``ar_worker.py``), next to the editable code;
:func:`predictions_array` returns the evaluation predictions as a NumPy array, which the worker sends to the
frozen **parent** (``run_fixed_budget.py``).  The parent, a separate process that never imports editable code,
computes the MSE from them with the targets it loaded itself and writes the result lines to the runner's
result channel.  So editable code cannot change the metric arithmetic or print a result line that counts;
what it can still do in the worker (extra updates by other means, reading the data files, answering the
parent with other predictions than its model's) is listed in the port report's threat model.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

import ar_data

LOG_EVERY = 10
EVAL_BATCH_SIZE = 256
EXIT_BAD_INVOCATION = 2
EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_INFRA = 6


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def read_budget(argv: list[str] | None = None) -> int:
    """The fixed step budget: ``--steps N`` from the run command; it must agree with
    ``AR_BUDGET_VALUE``/``AR_BUDGET_KIND`` when the runner sets them."""
    parser = argparse.ArgumentParser(description="Track G fixed-step forecasting wrapper")
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1:
        fail(EXIT_BAD_INVOCATION, f"--steps must be positive, got {args.steps}")
    kind = os.environ.get("AR_BUDGET_KIND")
    if kind is not None and kind != "steps":
        fail(EXIT_BAD_INVOCATION, f"AR_BUDGET_KIND={kind!r}, this wrapper runs a step budget")
    value = os.environ.get("AR_BUDGET_VALUE")
    if value is not None and int(value) != args.steps:
        fail(EXIT_BAD_INVOCATION, f"--steps {args.steps} disagrees with AR_BUDGET_VALUE={value}")
    return args.steps


def read_seeds(default_train: int = 0, default_split: int = 0) -> tuple[int, int]:
    """Training and split seeds from AR_TRAIN_SEED / AR_SPLIT_SEED (the node spec's seed_env)."""
    raw_train, raw_split = os.environ.get("AR_TRAIN_SEED"), os.environ.get("AR_SPLIT_SEED")
    train = int(raw_train) if raw_train is not None else default_train
    split = int(raw_split) if raw_split is not None else default_split
    print(f"AR_SEEDS train={train}{'' if raw_train is not None else ' (default)'} "
          f"split={split}{'' if raw_split is not None else ' (default)'}", flush=True)
    return train, split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def num_workers() -> int:
    """The fixed data-loader worker count the runner advertises (AR_NUM_WORKERS).
    This node keeps the series in memory and uses no worker processes; the count is read and printed so the
    run log shows it."""
    return int(os.environ.get("AR_NUM_WORKERS", "0"))


def pick_device() -> torch.device:
    """CUDA when available.  When the runner assigned a GPU (non-empty CUDA_VISIBLE_DEVICES) but CUDA
    is unavailable, stop with an ``AR_INFRA`` line (the node's infrastructure pattern): a silent CPU
    fallback is a STOP for the census.  This runs before the editable code is imported, so no
    proposal can cause it."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() and not torch.cuda.is_available():
        print("AR_INFRA a GPU was assigned (CUDA_VISIBLE_DEVICES set) but CUDA is unavailable", flush=True)
        sys.exit(EXIT_INFRA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"AR_DEVICE {device.type} {name} torch={torch.__version__} workers={num_workers()}", flush=True)
    return device


def check_api(train_mod) -> dict:
    for name in ("build_model", "build_optimizer", "training_step", "predict"):
        if not callable(getattr(train_mod, name, None)):
            fail(EXIT_BAD_EDITABLE_API, f"train.py must define {name}()")
    config = getattr(train_mod, "CONFIG", None)
    if not isinstance(config, dict):
        fail(EXIT_BAD_EDITABLE_API, "train.py must define CONFIG as a dict")
    batch_size = config.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        fail(EXIT_BAD_EDITABLE_API, f"CONFIG['batch_size'] must be a positive int, got {batch_size!r}")
    return config


class FixedStepSampler:
    """Exactly ``budget`` batches of training-window indices: epochs of a seeded permutation,
    the last incomplete batch of each epoch dropped.  The order depends only on the
    training seed, the number of training windows and the batch size."""

    def __init__(self, n_items: int, batch_size: int, budget: int, seed: int, max_batch_size: int):
        if batch_size > max_batch_size:
            fail(EXIT_BAD_EDITABLE_API, f"CONFIG['batch_size']={batch_size} exceeds this node's cap {max_batch_size}")
        if batch_size > n_items:
            fail(EXIT_BAD_EDITABLE_API, f"CONFIG['batch_size']={batch_size} exceeds the {n_items} training windows")
        self.n_items, self.batch_size, self.budget = n_items, batch_size, budget
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.served = 0

    def __iter__(self):
        per_epoch = self.n_items // self.batch_size
        while self.served < self.budget:
            # On the CPU whatever the default device (an edit may call torch.set_default_device): the order
            # depends only on the seed, and the CPU generator matches the device.
            order = torch.randperm(self.n_items, generator=self.generator, device="cpu")
            for k in range(per_epoch):
                if self.served >= self.budget:
                    return
                self.served += 1
                yield order[k * self.batch_size:(k + 1) * self.batch_size]


def train_fixed_steps(train_mod, model, optimizer, scheduler, sampler: FixedStepSampler, make_batch,
                      config: dict, device: torch.device) -> int:
    """Run exactly ``sampler.budget`` optimizer steps and print the loss log.

    The optimizer returned by build_optimizer must be a ``torch.optim.Optimizer``; a post-step hook
    counts its ``step()`` calls, and the run fails (exit 3) unless the count equals the budget, so
    editable code cannot add extra ``step()`` calls of that optimizer (for example inside
    training_step).  Known limits, not enforced: the hook counts calls, not updates, so an optimizer
    whose single ``step()`` makes several updates (Lookahead or SAM written inside ``step()``) passes;
    editable code runs in this process and could also add updates by other means (a second optimizer,
    manual updates, an EMA copy), tamper with the optimizer's hook registry or patch this module, print
    forged lines, or read the data files.  A subclass whose ``step()`` calls ``super().step()`` is
    counted twice if an instance of the parent class was created earlier in the process (torch's hook
    wrapping); the run then fails as divergent."""
    if not isinstance(optimizer, torch.optim.Optimizer):
        fail(EXIT_BAD_EDITABLE_API, "build_optimizer must return a torch.optim.Optimizer")
    optimizer_steps = [0]
    optimizer.register_step_post_hook(lambda opt, args, kwargs: optimizer_steps.__setitem__(0, optimizer_steps[0] + 1))
    amp = bool(config.get("amp_bf16", False)) and device.type == "cuda"
    clip = config.get("grad_clip")
    running = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    step = 0
    for idx in sampler:
        step += 1
        model.train()
        batch = make_batch(idx.to(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss = train_mod.training_step(model, batch, step)
        if type(loss) is not torch.Tensor or loss.ndim != 0:  # a plain tensor: no subclass code in the frozen loop
            fail(EXIT_BAD_EDITABLE_API, "training_step must return a scalar torch.Tensor (not a subclass)")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.detach().double()
        count += 1
        if step % LOG_EVERY == 0 or step == sampler.budget:
            ar_data.emit(f"step {step} loss {(running / count).item():.6f}")  # not through sys.stdout
            running.zero_()
            count = 0
    if step != sampler.budget or sampler.served != sampler.budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"ran {step} of {sampler.budget} steps")
    if optimizer_steps[0] != sampler.budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"the optimizer stepped {optimizer_steps[0]} times for a budget of {sampler.budget}")
    ar_data.emit(f"AR_STEPS completed={step}")
    return step


@torch.no_grad()
def predictions_array(train_mod, model, make_eval_batch, n_starts: int, horizon: int, n_features: int,
                      config: dict, device: torch.device) -> np.ndarray:
    """Predictions for the evaluation windows 0..n_starts-1, in order, as a float32 array
    (n_starts, horizon * n_features) for the parent: ``predict`` must return a plain tensor of shape
    (batch, horizon, n_features) (a subclass is refused: its methods would be editable code here)."""
    amp = bool(config.get("amp_bf16", False)) and device.type == "cuda"
    model.eval()
    out = []
    for start in range(0, n_starts, EVAL_BATCH_SIZE):
        idx = torch.arange(start, min(n_starts, start + EVAL_BATCH_SIZE), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = train_mod.predict(model, make_eval_batch(idx))
        if type(pred) is not torch.Tensor or tuple(pred.shape) != (len(idx), horizon, n_features):
            fail(EXIT_BAD_EDITABLE_API, f"predict must return a plain torch.Tensor of shape "
                                        f"({len(idx)}, {horizon}, {n_features})")
        with torch._C.DisableTorchFunction():
            out.append(pred.detach().float().cpu().numpy().reshape(len(idx), horizon * n_features))
    return np.concatenate(out, axis=0) if out else np.zeros((0, horizon * n_features), dtype=np.float32)
