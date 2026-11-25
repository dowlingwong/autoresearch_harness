"""Frozen worker entry of node mlgym_titanic (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the
runner's result channel.  ``train``: load the task tables, take the parent's training and validation rows,
seed, import the editable train.py, fit its feature encoder on the training rows only, build the model and
the optimizer, and run exactly ``budget`` optimizer steps with the frozen loop (ar_fixed_loop: optimizer step
hook, CPU batch order, loss lines).  ``logits``: the validation or test logits as a float32 array, in order.
The metric is computed by the parent.

The node is CPU-only: the worker pins every tensor and the model to the CPU and never touches CUDA (the
frozen loop's GPU device pick is for the GPU nodes; here the editable model is a small tabular MLP that the
R1-BX-D5 budget bounds by steps).
"""
import os

import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "mlgym_titanic"
N_CLASSES = 2
MAX_BATCH_SIZE = 4096
FILES = ["train.csv", "test.csv", "ground_truth.csv"]
EXIT_BAD_EDITABLE_API = 4
STATE = {}


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    raise SystemExit(code)


def encode_rows(encode, frame, name: str, n_rows: int) -> torch.Tensor:
    """Apply the editable encoder to *frame*; a float32 (n_rows, d) finite array or exit 4."""
    values = np.asarray(encode(frame.copy()), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != n_rows or values.shape[1] < 1:
        fail(EXIT_BAD_EDITABLE_API,
             f"build_features' encoder must return a 2-D array of shape ({n_rows}, d), got {values.shape}")
    if not np.all(np.isfinite(values)):
        fail(EXIT_BAD_EDITABLE_API, "build_features' encoder returned non-finite feature values")
    return torch.from_numpy(np.ascontiguousarray(values))


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    device = torch.device("cpu")
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    train_frame = pd.read_csv(data_dir / "train.csv")
    test_frame = pd.read_csv(data_dir / "test.csv")
    tr_rows = np.asarray(arrays["train_rows"], dtype=np.int64)
    val_rows = np.asarray(arrays["val_rows"], dtype=np.int64)
    labels = train_frame["Survived"].to_numpy(dtype=np.int64)
    print(f"AR_DEVICE cpu cpu torch={torch.__version__} workers={os.environ.get('AR_NUM_WORKERS', '0')}",
          flush=True)

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    if not callable(getattr(T, "build_features", None)):
        fail(EXIT_BAD_EDITABLE_API, "train.py must define build_features(train_frame)")
    encode = T.build_features(train_frame.iloc[tr_rows].reset_index(drop=True).copy())
    if not callable(encode):
        fail(EXIT_BAD_EDITABLE_API, "build_features must return a callable encoder")
    x_tr = encode_rows(encode, train_frame.iloc[tr_rows], "train", len(tr_rows))
    x_val = encode_rows(encode, train_frame.iloc[val_rows], "validation", len(val_rows))
    x_test = encode_rows(encode, test_frame, "test", len(test_frame))
    y_tr = torch.from_numpy(labels[tr_rows])
    STATE["eval"] = {"val": x_val, "test": x_test}

    info = {"device": device, "num_classes": N_CLASSES, "n_features": int(x_tr.shape[1]),
            "n_train": int(x_tr.shape[0])}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(x_tr), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda idx: {"x": x_tr[idx], "y": y_tr[idx]}, config, device)
    STATE.update(T=T, model=model, config=config, device=device)
    return {}, {}


def logits(meta, arrays, fds):
    x = STATE["eval"][meta["split"]]
    out = L.logits_array(STATE["T"], STATE["model"], lambda idx: {"x": x[idx]}, len(x), N_CLASSES,
                         STATE["config"], STATE["device"])
    return {}, {"logits": out}


iso.serve({"train": train, "logits": logits}, CONN)
