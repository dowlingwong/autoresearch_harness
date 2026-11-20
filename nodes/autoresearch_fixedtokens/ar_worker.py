"""Frozen worker entry of the Track G node autoresearch_fixedtokens (R1-HARNESS-ISOLATION): the only process
that runs the editable train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the runner's
result channel.  ``train``: seed from AR_TRAIN_SEED, replace ``prepare.make_dataloader`` by the budgeted
factory (ar_budget.py) and run train.py (``runpy``, as ``__main__``); exit 7 for a request beyond the budget,
8 for the validation split while train.py runs (also when train.py caught the refusal), 4 for any other
exception or ``SystemExit`` out of train.py, 3 unless train.py used the whole budget; print ``AR_TOKENS
consumed=``; check the data again; restore prepare's module attributes (undoing patches such as a changed
``VAL_FILENAME``); take the ``model`` train.py left, put it in eval mode without calling its methods, and
record its weights' fingerprint.  ``bind_cpu``: map the parent's CPU output buffer (a ``memfd`` handed over the
socket).  ``logits``: ``model(x)`` for the parent's input tokens (bf16 autocast), which must be a plain tensor
of shape (B, T, vocab), written as float32 into that CPU buffer -- on a GPU host too, so the result path carries
no CUDA IPC (R1-TRACKG-AR-COND7 option A).  ``finish``: the weights must be unchanged by the evaluation
(exit 10).  The metric is computed by the parent.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import os  # noqa: E402
import random  # noqa: E402
import runpy  # noqa: E402
import sys  # noqa: E402

import ar_data  # noqa: E402

EXIT_BUDGET_NOT_REACHED = 3
EXIT_BAD_EDITABLE_API = 4
EXIT_OVER_BUDGET = 7
EXIT_VALIDATION_REQUESTED = 8
EXIT_WEIGHTS_CHANGED = 10
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet",
                  *[f"data/shard_{i:05d}.parquet" for i in range(10)]]
NODE = "autoresearch_fixedtokens"
STATE = {}


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    sys.exit(code)


def real_dict(module) -> dict:
    """The module's instance dict through nn.Module's own ``__dict__`` descriptor (``vars()`` would honour a
    ``__dict__`` property defined by the model's class)."""
    import torch
    return torch.nn.Module.__dict__["__dict__"].__get__(module)


def module_tree(model) -> list:
    """The model and its submodules, read from the modules' own ``_modules`` dicts (no method of the model)."""
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
    out = []
    for index, module in enumerate(module_tree(model)):
        for kind in ("_parameters", "_buffers"):
            for name, tensor in real_dict(module).get(kind, {}).items():
                if tensor is not None:
                    out.append((f"{index}{kind}.{name}", tensor))
    return out


def weights_fingerprint(model, torch) -> list:
    with torch.no_grad(), torch._C.DisableTorchFunction():
        return [(name, float(t.double().sum()).hex(), float(t.double().square().sum()).hex())  # hex: nan == nan
                for name, t in module_tensors(model)]


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
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
    make, served, violations, open_validation = ar_budget.budgeted(prepare.make_dataloader, budget)
    prepare.make_dataloader = make
    vars(prepare).setdefault("__warningregistry__", {})  # warnings.warn may add it later; dunders are not compared
    snapshot = {k: v for k, v in vars(prepare).items() if not (k.startswith("__") and k.endswith("__"))}
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
    ar_data.verify(data_dir, NODE, REQUIRED_FILES, prefix="AR_ERROR train.py changed the data:")
    for key in [k for k in vars(prepare) if k not in snapshot and not (k.startswith("__") and k.endswith("__"))]:
        delattr(prepare, key)
    for key, value in snapshot.items():
        setattr(prepare, key, value)
    open_validation()

    model = namespace.get("model")
    if not isinstance(model, torch.nn.Module):
        fail(EXIT_BAD_EDITABLE_API, "train.py must leave the trained `model` in its globals")
    tensors = module_tensors(model)
    if not tensors:
        fail(EXIT_BAD_EDITABLE_API, "the model has no parameters")
    for module in module_tree(model):  # eval mode without calling any method of the model
        real_dict(module)["training"] = False
    batch_size = namespace.get("DEVICE_BATCH_SIZE")
    STATE.update(torch=torch, model=model, device=tensors[0][1].device, weights=weights_fingerprint(model, torch))
    if torch.cuda.is_available():
        print(f"AR_INFO peak_vram_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    return {"batch_size": batch_size if isinstance(batch_size, int) and not isinstance(batch_size, bool) else None,
            "tokens": served()}, {}


def bind_cpu(meta, arrays, fds):
    desc = meta["buffer"]
    STATE["shared"] = iso.SharedBuffer(tuple(desc["shape"]), desc["dtype"], fd=fds[0])
    STATE["out"] = STATE["torch"].from_numpy(STATE["shared"].array)
    return {}, {}


def logits(meta, arrays, fds):
    torch, model, device = STATE["torch"], STATE["model"], STATE["device"]
    out = STATE["out"]   # float32 CPU buffer: the parent's, mapped through the memfd (no CUDA IPC)
    x = torch.from_numpy(arrays["x"].astype("int64")).to(device)
    if tuple(x.shape) != tuple(out.shape[:2]):
        fail(EXIT_BAD_EDITABLE_API, f"input batch {tuple(x.shape)} does not fit the output buffer {tuple(out.shape)}")
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type in ("cuda", "cpu")):
        result = model(x)
    if type(result) is not torch.Tensor or tuple(result.shape) != tuple(out.shape):
        fail(EXIT_BAD_EDITABLE_API, f"model(x) must return a plain torch.Tensor of logits of shape {tuple(out.shape)}")
    with torch.no_grad(), torch._C.DisableTorchFunction():
        out.copy_(result.float())   # device -> host copy; synchronous, the parent's next request sees it
    return {}, {}


def finish(meta, arrays, fds):
    if weights_fingerprint(STATE["model"], STATE["torch"]) != STATE["weights"]:
        fail(EXIT_WEIGHTS_CHANGED, "the model's weights or buffers changed during the evaluation")
    return {}, {}


iso.serve({"train": train, "bind_cpu": bind_cpu, "logits": logits, "finish": finish}, CONN)
