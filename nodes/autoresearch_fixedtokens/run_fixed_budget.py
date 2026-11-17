"""Frozen fixed-budget wrapper of the Track G node autoresearch_fixedtokens.

Run command: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 run_fixed_budget.py --tokens {budget}``
(the variable torch 2.6 reads; the fork's ``PYTORCH_ALLOC_CONF`` is read only from torch 2.9 on).

FROZEN.  Runs the editable train.py (the autoresearch training script) in
this process with a fixed budget of ``--tokens`` training tokens in place of
autoresearch's 5-minute wall-clock budget, then evaluates the trained
``model`` with autoresearch's own frozen metric, ``prepare.evaluate_bpb``
(bits per byte on the pinned validation shard), and prints
``AR_METRIC val_bpb=`` (minimize).  Data and tokenizer: the files prepared by
prepare_data.py, checked against data/data_manifest.json; prepare.py reads them
from ``data/`` (AUTORESEARCH_CACHE_DIR, set here before prepare is imported).
Seeds: torch is seeded from AR_TRAIN_SEED before train.py runs (autoresearch
itself seeds 42 in train.py; the port moved that here).  AR_SPLIT_SEED has no
effect: the validation shard is fixed by autoresearch.  The token budget is
enforced by ar_budget.py: the training loader serves exactly ``--tokens``
tokens and the wrapper stops (exit 3) unless train.py used all of them; a
request beyond the budget exits 7; a request for the validation split while
train.py runs exits 8; any other exception or ``SystemExit`` out of train.py
exits 4 (so a ``sys.exit(0)`` after a forged metric line cannot skip the budget
check).  Known in-process limits (the editable code runs in this process): it
can still print forged ``AR_`` lines before failing, patch this module's
functions at run time, introspect the budget closure, or read the parquet files
directly; the line order and the checks above narrow these, they do not close
them.
"""
import argparse
import os
import random
import runpy
import sys

EXIT_BAD_INVOCATION = 2
EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_OVER_BUDGET = 7
EXIT_VALIDATION_REQUESTED = 8
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet",
                  *[f"data/shard_{i:05d}.parquet" for i in range(10)]]
NODE = "autoresearch_fixedtokens"


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track G fixed-token wrapper")
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    budget = args.tokens
    if budget < 1:
        fail(EXIT_BAD_INVOCATION, f"--tokens must be positive, got {budget}")
    if os.environ.get("AR_BUDGET_KIND", "tokens") != "tokens":
        fail(EXIT_BAD_INVOCATION, "AR_BUDGET_KIND must be tokens")
    value = os.environ.get("AR_BUDGET_VALUE")
    if value is not None and int(value) != budget:
        fail(EXIT_BAD_INVOCATION, f"--tokens {budget} disagrees with AR_BUDGET_VALUE={value}")
    raw_train = os.environ.get("AR_TRAIN_SEED")
    train_seed = int(raw_train) if raw_train is not None else 42
    print(f"AR_SEEDS train={train_seed}{'' if raw_train is not None else ' (default, autoresearch seed)'} "
          f"split={os.environ.get('AR_SPLIT_SEED')} (no effect: fixed validation shard)", flush=True)

    import ar_data

    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, REQUIRED_FILES)
    os.environ["AUTORESEARCH_CACHE_DIR"] = str(data_dir.resolve())

    import numpy as np
    import torch

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    print(f"AR_DEVICE {'cuda ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"torch={torch.__version__} workers={os.environ.get('AR_NUM_WORKERS', '0')}", flush=True)

    import ar_budget
    import prepare

    ar_budget.TRAIN_TOKENS = budget  # read by train.py; the budget itself is enforced by the closure below
    make, served, open_validation = ar_budget.budgeted(prepare.make_dataloader, budget)
    prepare.make_dataloader = make
    try:
        namespace = runpy.run_path("train.py", run_name="__main__")  # the editable training script
    except ar_budget.BudgetExhausted as exc:
        fail(EXIT_OVER_BUDGET, f"train.py requested more than the budget: {exc}")
    except ar_budget.ValidationRequested as exc:
        fail(EXIT_VALIDATION_REQUESTED, f"train.py requested the validation split: {exc}")
    except BaseException as exc:  # noqa: BLE001 - SystemExit and KeyboardInterrupt included, on purpose
        fail(EXIT_BAD_EDITABLE_API, f"train.py ended with {type(exc).__name__}: {exc}")
    if served() != budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"train.py used {served()} of {budget} training tokens")
    print(f"AR_TOKENS consumed={served()}", flush=True)
    open_validation()

    model, tokenizer = namespace.get("model"), namespace.get("tokenizer")
    if not isinstance(model, torch.nn.Module) or tokenizer is None:
        fail(EXIT_BAD_EDITABLE_API, "train.py must leave the trained `model` and the `tokenizer` in its globals")
    batch_size = namespace.get("DEVICE_BATCH_SIZE")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 512:
        batch_size = 16
    device_type = next(model.parameters()).device.type
    model.eval()
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type in ("cuda", "cpu")):
        val_bpb = prepare.evaluate_bpb(model, tokenizer, batch_size)
    if torch.cuda.is_available():
        print(f"AR_INFO peak_vram_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    print(f"AR_METRIC val_bpb={val_bpb:.6f}", flush=True)


if __name__ == "__main__":
    main()
