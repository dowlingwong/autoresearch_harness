"""Frozen fixed-budget wrapper of the Track G node autoresearch_fixedtokens.

Run command: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 run_fixed_budget.py --tokens {budget}``
(the variable torch 2.6 reads; the fork's ``PYTORCH_ALLOC_CONF`` is read only from torch 2.9 on).

FROZEN.  Runs the editable train.py (the autoresearch training script) in
this process with a fixed budget of ``--tokens`` training tokens in place of
autoresearch's 5-minute wall-clock budget, then evaluates the trained
``model`` and prints ``AR_METRIC val_bpb=`` (minimize).  Data and tokenizer:
the files prepared by prepare_data.py, checked against data/data_manifest.json
(no unlisted file allowed); prepare.py reads them from ``data/``
(AUTORESEARCH_CACHE_DIR, set here before prepare is imported).  Seeds: torch is
seeded from AR_TRAIN_SEED before train.py runs (autoresearch itself seeds 42 in
train.py; the port moved that here).  AR_SPLIT_SEED has no effect: the
validation shard is fixed by autoresearch.

Budget (ar_budget.py): the training loader serves exactly ``--tokens`` tokens and
the wrapper stops (exit 3) unless train.py used all of them; a request beyond the
budget exits 7; a request for the validation split while train.py runs exits 8
(also when train.py catches the exception: the refusal is recorded); any other
exception or ``SystemExit`` out of train.py exits 4.

Metric (frozen here, not delegated to the editable model): autoresearch's bits
per byte on the pinned validation shard, as ``prepare.evaluate_bpb`` defines it
(``EVAL_TOKENS`` tokens in rows of ``MAX_SEQ_LEN``, special tokens of byte length
0 excluded), but the per-token cross-entropy is computed here from the model's
logits (``model(x)``, shape (B, T, vocab)), so an edited loss cannot change it.
Before that, a causality probe compares the model's logits on validation rows
with its logits on the same rows whose suffix after a cut position was replaced:
the logits before the cut must not change (exit 9 otherwise; tolerance
``CAUSAL_ATOL``), so a model that reads later tokens (the targets) is refused.

Output: the result lines (``AR_TOKENS``, ``AR_METRIC``) go to a copy of the
standard output descriptor taken before train.py runs, and the run ends with
``os._exit(0)`` right after the metric (ar_data.emit / finish), so train.py
cannot rewrite them through ``sys.stdout`` or add lines from an ``atexit``
handler.  Known in-process limits (the editable code runs in this process): it
can print a forged ``AR_METRIC`` line and end the process itself (``os._exit``)
before the wrapper prints, patch the wrapper's or prepare's functions at run
time, introspect the budget closure, read the parquet files directly (including
the validation shard), call prepare's internal readers, keep a background thread
running into the evaluation, or present a model that behaves causally only on
the probe; the checks above narrow these, they do not close them.  Such code is
visible in the diff.
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
EXIT_NOT_CAUSAL = 9
CAUSAL_ATOL = 0.05  # logits are soft-capped at 15; bf16 noise between two forwards is far below this
PROBE_CUTS = (0.25, 0.5, 0.75)  # fractions of the row length
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet",
                  *[f"data/shard_{i:05d}.parquet" for i in range(10)]]
NODE = "autoresearch_fixedtokens"


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def causality_probe(model, x, torch):
    """Logits before each cut must not depend on the tokens after it."""
    T = x.shape[1]
    with torch.no_grad():
        base = model(x)
        if not torch.is_tensor(base) or base.ndim != 3 or base.shape[:2] != x.shape:
            fail(EXIT_BAD_EDITABLE_API, "model(x) must return logits of shape (B, T, vocab)")
        if not torch.isfinite(base.float()).all():
            return  # a non-finite metric follows; nothing to compare
        for frac in PROBE_CUTS:
            cut = max(1, int(T * frac))
            altered = x.clone()
            altered[:, cut:] = torch.roll(x, shifts=1, dims=0)[:, cut:]
            if torch.equal(altered, x):
                altered[:, cut:] = (x[:, cut:] + 1) % base.shape[-1]
            diff = (model(altered).float()[:, :cut] - base.float()[:, :cut]).abs().max().item()
            if not diff <= CAUSAL_ATOL:
                fail(EXIT_NOT_CAUSAL, f"logits before position {cut} changed by {diff:.4g} when later tokens "
                                      f"changed (tolerance {CAUSAL_ATOL}): the model is not causal")


def frozen_bpb(model, tokenizer, batch_size, make_dataloader, get_token_bytes, max_seq_len, eval_tokens, torch):
    """prepare.evaluate_bpb with the cross-entropy computed here from the model's logits."""
    import math

    import torch.nn.functional as F

    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, max_seq_len, "val")
    steps = eval_tokens // (batch_size * max_seq_len)
    total_nats, total_bytes = 0.0, 0
    with torch.no_grad():
        for index in range(steps):
            x, y, _ = next(val_loader)
            if index == 0:
                causality_probe(model, x, torch)  # the evaluation batch shape: no recompilation
            logits = model(x)
            if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[:2] != x.shape:
                fail(EXIT_BAD_EDITABLE_API, "model(x) must return logits of shape (B, T, vocab)")
            nats = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
            nbytes = token_bytes[y.reshape(-1)]
            total_nats += (nats * (nbytes > 0)).sum().item()
            total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


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

    # Bound before train.py runs, so later patches of prepare's attributes do not reach the evaluation.
    get_token_bytes, max_seq_len, eval_tokens = prepare.get_token_bytes, prepare.MAX_SEQ_LEN, prepare.EVAL_TOKENS
    ar_budget.TRAIN_TOKENS = budget  # read by train.py; the budget itself is enforced by the closure below
    make, served, violations, open_validation = ar_budget.budgeted(prepare.make_dataloader, budget)
    prepare.make_dataloader = make
    try:
        namespace = runpy.run_path("train.py", run_name="__main__")  # the editable training script
    except ar_budget.BudgetExhausted as exc:
        fail(EXIT_OVER_BUDGET, f"train.py requested more than the budget: {exc}")
    except ar_budget.ValidationRequested as exc:
        fail(EXIT_VALIDATION_REQUESTED, f"train.py requested the validation split: {exc}")
    except BaseException as exc:  # noqa: BLE001 - SystemExit and KeyboardInterrupt included, on purpose
        fail(EXIT_BAD_EDITABLE_API, f"train.py ended with {type(exc).__name__}: {exc}")
    if "validation" in violations():
        fail(EXIT_VALIDATION_REQUESTED, "train.py requested the validation split (and caught the refusal)")
    if "budget" in violations():
        fail(EXIT_OVER_BUDGET, "train.py requested more than the budget (and caught the refusal)")
    if served() != budget:
        fail(EXIT_BUDGET_NOT_REACHED, f"train.py used {served()} of {budget} training tokens")
    ar_data.emit(f"AR_TOKENS consumed={served()}")
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
        val_bpb = frozen_bpb(model, tokenizer, batch_size, make, get_token_bytes, max_seq_len, eval_tokens, torch)
    if torch.cuda.is_available():
        print(f"AR_INFO peak_vram_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    ar_data.emit(f"AR_METRIC val_bpb={val_bpb:.6f}")
    ar_data.finish()  # no output of the editable code after the metric


if __name__ == "__main__":
    main()
