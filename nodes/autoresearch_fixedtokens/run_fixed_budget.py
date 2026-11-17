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
The tokenizer is the frozen one (``prepare.Tokenizer.from_directory()``), not an
object train.py left behind; ``torch.nn.functional.cross_entropy`` is bound before
train.py runs; prepare's module attributes are restored to their state before
train.py ran (undoing patches such as a changed ``VAL_FILENAME`` or a reload); the
data are checked again after train.py returns (a run that changed them fails itself,
with ``AR_ERROR train.py changed the data:``).  Before the sum, a causality probe
compares the model's logits on validation rows with its logits on the same rows
whose suffix after a cut position was replaced: the logits before the cut must not
change (exit 9 otherwise; tolerance ``CAUSAL_ATOL``; equal infinities count as
equal), so a model that reads later tokens (the targets) is refused; the largest
change is printed as ``AR_INFO causal_probe_max_change=`` for the census.  The
model's parameters and buffers must be unchanged by the evaluation (exit 10), so
a model that updates itself in ``forward`` cannot train on the validation shard.
``model(x)`` must return a plain ``torch.Tensor`` (a subclass is refused, exit 4)
and the frozen arithmetic runs with torch-function dispatch disabled, so no
editable ``__torch_function__`` subclass or mode runs inside it, and no torch
dispatch mode may be active then (exit 4); eval mode, the device and
the weight fingerprint read the modules' own ``_modules`` / ``_parameters`` /
``_buffers`` dicts, so no (overridable) method of the model is called; after the
evaluation, prepare's attributes (dunder names aside: ``warnings`` may add
``__warningregistry__``) and the data are checked once more (the model's own code
ran during it).  A patch of torch itself (e.g. ``torch.Tensor.__eq__``) is not
detected.

Output: the result lines (``AR_TOKENS``, ``AR_METRIC``) go to a copy of the
standard output descriptor taken before train.py runs; just before the metric the
process's descriptors 1 and 2 are pointed at /dev/null (ar_data.seal), so threads
train.py started cannot print into the log any more; and the run ends with
``os._exit(0)`` right after the metric (ar_data.emit / finish), so train.py
cannot rewrite the lines through ``sys.stdout`` or add lines from an ``atexit``
handler.  Known in-process limits (the editable code runs in this process): it
can print a forged ``AR_METRIC`` line and end the process itself (``os._exit``)
before the wrapper prints, write to the saved descriptor itself, patch the
wrapper's, ar_data's or torch's functions at run time (prepare's attributes are
restored before the evaluation), introspect the budget closure, read the parquet files directly (including
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

import ar_data  # takes the copy of the output descriptor for the result lines, before train.py runs

EXIT_BAD_INVOCATION = 2
EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_OVER_BUDGET = 7
EXIT_VALIDATION_REQUESTED = 8
EXIT_NOT_CAUSAL = 9
EXIT_WEIGHTS_CHANGED = 10
CAUSAL_ATOL = 0.05  # logits are soft-capped at 15; bf16 noise between two forwards is far below this
PROBE_CUTS = (0.25, 0.5, 0.75)  # fractions of the row length
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet",
                  *[f"data/shard_{i:05d}.parquet" for i in range(10)]]
NODE = "autoresearch_fixedtokens"


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def causality_probe(model, x, torch) -> float:
    """Logits before each cut must not depend on the tokens after it; returns the largest change seen."""
    T = x.shape[1]
    worst = 0.0
    with torch.no_grad():
        with torch._C.DisableTorchFunction():
            model_input = x.clone()
        base = plain_logits(model(model_input), x, torch)
        with torch._C.DisableTorchFunction():
            if torch.isnan(base).any():
                return float("nan")  # a non-finite metric follows; nothing to compare
        for frac in PROBE_CUTS:
            cut = max(1, int(T * frac))
            with torch._C.DisableTorchFunction():  # the probe's inputs are built without editable mode code
                altered = x.clone()
                altered[:, cut:] = torch.roll(x, shifts=1, dims=0)[:, cut:]
                if torch.equal(altered, x):
                    altered[:, cut:] = (x[:, cut:] + 1) % base.shape[-1]
                model_input = altered.clone()  # the model gets a copy, so it cannot change the probe's tensors
            after = plain_logits(model(model_input), x, torch)
            with torch._C.DisableTorchFunction():
                after, before = after[:, :cut], base[:, :cut]
                change = torch.where(after == before, torch.zeros_like(before), (after - before).abs())  # inf == inf
                diff = change.nan_to_num(nan=float("inf")).max().item()
            worst = max(worst, diff)
            if not diff <= CAUSAL_ATOL:
                fail(EXIT_NOT_CAUSAL, f"logits before position {cut} changed by {diff:.4g} when later tokens "
                                      f"changed (tolerance {CAUSAL_ATOL}): the model is not causal")
    return worst


def no_active_modes(torch) -> None:
    """Refuse a torch-dispatch mode left active by the editable code (DisableTorchFunction, used around the frozen
    arithmetic, does not reach dispatch modes; the stack is empty in a clean run).  Torch-function modes are kept out
    by DisableTorchFunction and not refused: an ordinary edit can leave one (``torch.set_default_device``)."""
    if torch._C._len_torch_dispatch_stack():
        fail(EXIT_BAD_EDITABLE_API, "a torch dispatch mode is active during the frozen evaluation")


def real_dict(module) -> dict:
    """The module's instance dict through nn.Module's own ``__dict__`` descriptor (``vars()`` would honour a
    ``__dict__`` property defined by the model's class)."""
    import torch
    return torch.nn.Module.__dict__["__dict__"].__get__(module)


def module_tree(model) -> list:
    """The model and its submodules, read from the modules' own ``_modules`` dicts: no method of the model (which
    could be overridden) is called."""
    seen, stack, out = set(), [model], []
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        out.append(module)
        stack.extend(child for child in real_dict(module).get("_modules", {}).values() if child is not None)
    return out


def module_tensors(model) -> list:
    """(name, tensor) of every parameter and buffer, read from the modules' ``_parameters`` / ``_buffers`` dicts."""
    out = []
    for index, module in enumerate(module_tree(model)):
        for kind in ("_parameters", "_buffers"):
            for name, tensor in real_dict(module).get(kind, {}).items():
                if tensor is not None:
                    out.append((f"{index}{kind}.{name}", tensor))
    return out


def plain_logits(logits, x, torch):
    """The model's output as plain float32 logits (B, T, vocab); a tensor subclass is refused, so no editable
    ``__torch_function__`` runs inside the frozen metric, and no mode may be active."""
    if type(logits) is not torch.Tensor or logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(x.shape):
        fail(EXIT_BAD_EDITABLE_API, "model(x) must return a plain torch.Tensor of logits of shape (B, T, vocab)")
    no_active_modes(torch)
    with torch._C.DisableTorchFunction():
        return logits.float()


def weights_fingerprint(model, torch) -> list:
    """Per tensor (sum, sum of squares) of the model's parameters and all buffers, to detect changes during the
    evaluation; read from the modules' own dicts, so no method of the model is consulted."""
    tensors = module_tensors(model)
    no_active_modes(torch)
    with torch.no_grad(), torch._C.DisableTorchFunction():
        return [(name, float(t.double().sum()).hex(), float(t.double().square().sum()).hex())  # hex: nan == nan
                for name, t in tensors]


def frozen_bpb(model, tokenizer, batch_size, make_dataloader, get_token_bytes, max_seq_len, eval_tokens, torch,
               cross_entropy):
    """prepare.evaluate_bpb with the cross-entropy computed here from the model's logits."""
    import math

    device = module_tensors(model)[0][1].device
    # The data path (prepare's loader and token byte lengths) runs with torch-function dispatch disabled, so a mode
    # the editable code left active cannot rewrite the targets or the byte counts; only model(x) runs outside it.
    with torch._C.DisableTorchFunction():
        token_bytes = get_token_bytes(device=device)
        val_loader = make_dataloader(tokenizer, batch_size, max_seq_len, "val")
    steps = eval_tokens // (batch_size * max_seq_len)
    total_nats, total_bytes = 0.0, 0
    with torch.no_grad():
        for index in range(steps):
            no_active_modes(torch)
            with torch._C.DisableTorchFunction():
                x, y, _ = next(val_loader)
                # Copies: x and y are views of one reused loader buffer, which the model could otherwise write into.
                x, y = x.clone(), y.clone()
                model_input = x.clone()
            if index == 0:
                worst = causality_probe(model, x, torch)  # the evaluation batch shape: no recompilation
                print(f"AR_INFO causal_probe_max_change={worst:.3g} (tolerance {CAUSAL_ATOL})", flush=True)
            logits = plain_logits(model(model_input), x, torch)
            with torch._C.DisableTorchFunction():  # no __torch_function__ subclass or mode of the editable code
                nats = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
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

    from torch.nn.functional import cross_entropy  # bound before train.py runs

    # Bound before train.py runs, so later patches of prepare's attributes do not reach the evaluation.
    get_token_bytes, max_seq_len, eval_tokens = prepare.get_token_bytes, prepare.MAX_SEQ_LEN, prepare.EVAL_TOKENS
    ar_budget.TRAIN_TOKENS = budget  # read by train.py; the budget itself is enforced by the closure below
    make, served, violations, open_validation = ar_budget.budgeted(prepare.make_dataloader, budget)
    prepare.make_dataloader = make
    vars(prepare).setdefault("__warningregistry__", {})  # warnings.warn may add it later; dunders are not compared
    prepare_snapshot = {k: v for k, v in vars(prepare).items() if not (k.startswith("__") and k.endswith("__"))}
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
    # The data are checked again: a run that changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, REQUIRED_FILES, prefix="AR_ERROR train.py changed the data:")
    for key in [k for k in vars(prepare) if k not in prepare_snapshot and not (k.startswith("__") and k.endswith("__"))]:
        delattr(prepare, key)
    for key, value in prepare_snapshot.items():
        setattr(prepare, key, value)
    open_validation()

    model = namespace.get("model")
    if not isinstance(model, torch.nn.Module):
        fail(EXIT_BAD_EDITABLE_API, "train.py must leave the trained `model` in its globals")
    tokenizer = prepare.Tokenizer.from_directory()  # the frozen tokenizer, not an object train.py left behind
    batch_size = namespace.get("DEVICE_BATCH_SIZE")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 512:
        batch_size = 16
    tensors = module_tensors(model)
    if not tensors:
        fail(EXIT_BAD_EDITABLE_API, "the model has no parameters")
    device_type = tensors[0][1].device.type
    for module in module_tree(model):  # eval mode without calling any method of the model
        real_dict(module)["training"] = False
    weights_before = weights_fingerprint(model, torch)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type in ("cuda", "cpu")):
        val_bpb = frozen_bpb(model, tokenizer, batch_size, make, get_token_bytes, max_seq_len, eval_tokens, torch,
                             cross_entropy)
    if weights_fingerprint(model, torch) != weights_before:
        fail(EXIT_WEIGHTS_CHANGED, "the model's weights or buffers changed during the evaluation")
    # The model's own code ran during the evaluation: prepare and the data are checked once more.
    current = {k for k in vars(prepare) if not (k.startswith("__") and k.endswith("__"))}
    if any(vars(prepare).get(k, object()) is not v for k, v in prepare_snapshot.items()) or \
            current != set(prepare_snapshot):
        fail(EXIT_BAD_EDITABLE_API, "prepare's attributes changed during the evaluation")
    ar_data.verify(data_dir, NODE, REQUIRED_FILES, prefix="AR_ERROR train.py changed the data:")
    if torch.cuda.is_available():
        print(f"AR_INFO peak_vram_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    ar_data.seal()  # stdout/stderr to /dev/null: threads of the editable code cannot print now
    ar_data.emit(f"AR_METRIC val_bpb={val_bpb:.6f}")
    ar_data.finish()  # no output of the editable code after the metric


if __name__ == "__main__":
    main()
    ar_data.finish()  # main() already ended the process after the metric; kept for the shared static check
