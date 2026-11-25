"""Frozen parent helpers for isolated Track G nodes (R1-HARNESS-ISOLATION); NumPy only.

FROZEN: listed in the node's frozen_paths; copied byte-identically into every isolated node (the port tests
check this).  The node's parent script (``run_fixed_budget.py``) starts with ``ar_isolated.parent_start()``,
then uses these helpers: it reads the budget and the seeds, starts the frozen worker entry ``ar_worker.py``
(which alone imports the editable code), asks it for training and for predictions, computes the metric here
from the predictions and labels it loaded itself, and writes the result lines to the result channel.  Nothing
here imports editable code or torch, and nothing prints a result line to the run's output (the runner refuses
a run whose output holds a metric line).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import ar_isolated as iso

EXIT_BAD_INVOCATION = 2
EXIT_BAD_RESULT = 4
EXIT_WORKER = 12
WORKER_SCRIPT = "ar_worker.py"


def fail(code: int, message: str) -> None:
    """One ``AR_ERROR`` line (newlines removed, so text the worker sent cannot start a line) and exit."""
    text = " ".join(str(message).split())[:2000]
    print(f"AR_ERROR {text}", flush=True)
    sys.exit(code)


def read_budget(kind: str = "steps", exact: int | None = None) -> int:
    """The fixed budget from ``--steps N`` (or ``--tokens N``); it must agree with ``AR_BUDGET_KIND`` /
    ``AR_BUDGET_VALUE`` when the runner sets them, and equal *exact* when given."""
    flag = "--" + kind
    parser = argparse.ArgumentParser(description="Track G isolated parent")
    parser.add_argument(flag, type=int, required=True, dest="budget")
    args = parser.parse_args()
    if args.budget < 1:
        fail(EXIT_BAD_INVOCATION, f"{flag} must be positive, got {args.budget}")
    if exact is not None and args.budget != exact:
        fail(EXIT_BAD_INVOCATION, f"this node's budget is {flag} {exact}, got {args.budget}")
    env_kind = os.environ.get("AR_BUDGET_KIND")
    if env_kind is not None and env_kind != kind:
        fail(EXIT_BAD_INVOCATION, f"AR_BUDGET_KIND={env_kind!r}, this node runs a {kind} budget")
    value = os.environ.get("AR_BUDGET_VALUE")
    if value is not None and int(value) != args.budget:
        fail(EXIT_BAD_INVOCATION, f"{flag} {args.budget} disagrees with AR_BUDGET_VALUE={value}")
    return args.budget


def read_seeds(default_train: int = 0, default_split: int = 0, split_note: str = "") -> tuple[int, int]:
    """Training and split seeds from AR_TRAIN_SEED / AR_SPLIT_SEED (the node spec's seed_env)."""
    raw_train, raw_split = os.environ.get("AR_TRAIN_SEED"), os.environ.get("AR_SPLIT_SEED")
    train = int(raw_train) if raw_train is not None else default_train
    split = int(raw_split) if raw_split is not None else default_split
    print(f"AR_SEEDS train={train}{'' if raw_train is not None else ' (default)'} "
          f"split={split}{'' if raw_split is not None else ' (default' + split_note + ')'}", flush=True)
    return train, split


def start_worker() -> "iso.Worker":
    """Start the node's frozen worker entry (next to the parent script)."""
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    return iso.Worker(os.path.join(here, WORKER_SCRIPT), cwd=here)


def call(worker: "iso.Worker", op: str, meta: dict | None = None, arrays: dict | None = None) -> tuple[dict, dict]:
    """One worker request.  A worker that ended (a frozen check's exit, a crash) ends the run with the worker's
    exit code when it is positive, otherwise with exit 12; an error reply or a malformed reply ends it with 12."""
    try:
        return worker.call(op, meta or {}, arrays or {})
    except (iso.IsolationProtocolError, OSError) as error:
        try:
            code = worker.process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - still running: an error reply
            code = None
        fail(code if isinstance(code, int) and code > 0 else EXIT_WORKER,
             f"worker {op}: {error} (worker exit {code})")
    raise AssertionError("unreachable")


def stop_worker(worker: "iso.Worker") -> None:
    code = worker.close()
    print(f"AR_WORKER exit={code}", flush=True)


def check_array(arrays: dict, name: str, shape: tuple, kinds: str = "f") -> np.ndarray:
    """The worker's array *name*: present, of *shape*, with a dtype kind in *kinds*; returned as a private
    float64 (or int64) copy."""
    value = arrays.get(name)
    if not isinstance(value, np.ndarray) or tuple(value.shape) != tuple(shape) or value.dtype.kind not in kinds:
        got = None if not isinstance(value, np.ndarray) else (value.shape, value.dtype.name)
        fail(EXIT_BAD_RESULT, f"the worker's {name} must be an array of shape {tuple(shape)} "
                              f"(dtype kind {kinds}); got {got}")
    return np.array(value, dtype=np.float64 if value.dtype.kind == "f" else np.int64, copy=True)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy of argmax(logits) against *labels*; a row with a non-finite logit counts as wrong."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    finite = np.all(np.isfinite(logits), axis=1)
    pred = np.where(finite, np.argmax(np.where(np.isfinite(logits), logits, -np.inf), axis=1), -1)
    return float(np.mean(pred == labels)) if len(labels) else float("nan")


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error; ``nan`` if any prediction is not finite."""
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(math.sqrt(float(np.mean((np.asarray(y, dtype=np.float64) - pred) ** 2))))
