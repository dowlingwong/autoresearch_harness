"""Frozen worker entry of node gc7rc_etth1 (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the runner's
result channel.  ``train``: load the standardised series and the training-window starts the parent sent, seed,
import the editable train.py and run exactly ``budget`` optimizer steps with the frozen loop (ar_fixed_loop:
optimizer step hook, CPU window order, loss lines); ``predict``: the validation or test predictions as a float32
array of shape (n_windows, horizon * n_features), in order.  The MSE is computed by the parent.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "gc7rc_etth1"
FILES = ["series.npy"]
MAX_BATCH_SIZE = 4096
STATE = {}


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    raise SystemExit(code)


def windows(series: torch.Tensor, starts: torch.Tensor, seq_len: int, horizon: int,
            with_targets: bool = True) -> dict:
    """Look-back windows and (training only) their targets: {"x": (n, seq_len, C), "y": (n, horizon, C)}."""
    span = seq_len + horizon if with_targets else seq_len
    idx = starts.unsqueeze(1) + torch.arange(span, device=starts.device)
    block = series[idx]  # (n, seq_len [+ horizon], C)
    out = {"x": block[:, :seq_len, :]}
    if with_targets:
        out["y"] = block[:, seq_len:, :]
    return out


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    seq_len, horizon, n_features = int(meta["seq_len"]), int(meta["horizon"]), int(meta["n_features"])
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    series = np.asarray(arrays["series"], dtype=np.float32)
    starts = np.asarray(arrays["train_starts"], dtype=np.int64)
    if series.ndim != 2 or series.shape[1] != n_features or not np.all(np.isfinite(series)):
        fail(4, f"the parent's series must be finite (rows, {n_features}); got {series.shape}")
    series_t = torch.from_numpy(series).to(device)  # the parent's standardised series
    train_starts = torch.from_numpy(starts).long().to(device)
    STATE["eval_series"] = series_t
    STATE["seq_len"], STATE["horizon"], STATE["n_features"] = seq_len, horizon, n_features

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "seq_len": seq_len, "horizon": horizon, "n_features": n_features,
            "n_train": len(starts)}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(starts), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda idx: windows(series_t, train_starts[idx], seq_len, horizon), config, device)
    STATE.update(T=T, model=model, config=config, device=device)
    return {}, {}


def predict(meta, arrays, fds):
    starts = torch.from_numpy(np.asarray(arrays["starts"], dtype=np.int64)).long().to(STATE["device"])
    seq_len, horizon, n_features = STATE["seq_len"], STATE["horizon"], STATE["n_features"]
    out = L.predictions_array(STATE["T"], STATE["model"],
                              lambda idx: windows(STATE["eval_series"], starts[idx], seq_len, horizon,
                                                  with_targets=False),
                              len(starts), horizon, n_features, STATE["config"], STATE["device"])
    return {}, {"predictions": out}


iso.serve({"train": train, "predict": predict}, CONN)
