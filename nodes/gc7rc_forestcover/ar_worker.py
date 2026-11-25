"""Frozen worker entry of node gc7rc_forestcover (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the
runner's result channel.  ``train``: take the parent's standardised feature matrix, labels and the training
rows, seed, import the editable train.py, build the model and the optimizer, and run exactly ``budget``
optimizer steps with the frozen loop (ar_fixed_loop: optimizer step hook, CPU batch order, loss lines).
``logits``: the validation or test logits as a float32 array (n_rows, 7), in row order.  The accuracies are
computed by the parent from those logits and the labels it loaded itself.

The device is picked by the frozen loop (CUDA when the census assigns a GPU, the CPU otherwise); the parent's
standardised features and the labels are moved there once, and every batch is gathered from that copy.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "gc7rc_forestcover"
N_CLASSES = 7
MAX_BATCH_SIZE = 4096  # the paper's Task 5 batch size; an edit may change the model, not the batch cap
FILES = ["X.npy", "y.npy"]
EXIT_BAD_EDITABLE_API = 4
STATE = {}


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    raise SystemExit(code)


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    n_features, n_classes = int(meta["n_features"]), int(meta["n_classes"])
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    x = np.asarray(arrays["X"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.int64)
    train_rows = np.asarray(arrays["train_rows"], dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != n_features or not np.all(np.isfinite(x)):
        fail(EXIT_BAD_EDITABLE_API, f"the parent's features must be finite (rows, {n_features}); got {x.shape}")
    if y.shape != (x.shape[0],) or len(train_rows) == 0:
        fail(EXIT_BAD_EDITABLE_API, f"the parent's labels/rows are invalid: y {y.shape}, {len(train_rows)} rows")
    x_t = torch.from_numpy(x).to(device)  # the parent's standardised features
    y_t = torch.from_numpy(y).to(device)
    rows = {"train": torch.from_numpy(train_rows).long().to(device),
            "val": torch.from_numpy(np.asarray(arrays["val_rows"], dtype=np.int64)).long().to(device),
            "test": torch.from_numpy(np.asarray(arrays["test_rows"], dtype=np.int64)).long().to(device)}

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "n_features": n_features, "num_classes": n_classes,
            "n_train": len(train_rows)}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(train_rows), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda idx: {"x": x_t[rows["train"][idx]], "y": y_t[rows["train"][idx]]},
                        config, device)
    STATE.update(T=T, model=model, config=config, device=device, x_t=x_t, rows=rows)
    return {}, {}


def logits(meta, arrays, fds):
    split = meta["split"]
    rows = STATE["rows"][split]
    out = L.logits_array(STATE["T"], STATE["model"], lambda idx: {"x": STATE["x_t"][rows[idx]]}, len(rows),
                         N_CLASSES, STATE["config"], STATE["device"])
    return {}, {"logits": out}


iso.serve({"train": train, "logits": logits}, CONN)
