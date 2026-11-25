"""Frozen parent of node gc7rc_etth1 (R1-HARNESS-ISOLATION; started by the runner as
``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code.  Data: the public ETTh1 hourly series (Electricity
Transformer Temperature, 7 features, 17,420 rows) fetched by ``prepare_data.py`` from the pinned ETDataset
commit and stored as ``data/series.npy`` with a hashed data manifest; its upstream sha256 is pinned here and
checked against the manifest for the real (non-synthetic) data.  Split: the standard LTSF 12/4/4-month split
used by the 1GC-7RC Task 6 description (8,640 / 2,880 / 2,880 hourly rows; the remaining rows are not used),
standardised with the training rows' per-feature mean and standard deviation.  Input window 336 steps,
forecast horizon 96 steps, as stated for the task.  The frozen worker ``ar_worker.py`` imports train.py, runs
exactly ``--steps`` optimizer steps over the training windows and returns validation and test predictions; the
MSE values are computed here from the predictions and the targets this process loaded.

Budget: ``--steps`` optimizer steps (the spec's fixed step budget; R1-BX-D5).  Selection metric: validation
MSE (``AR_METRIC val_mse=``, minimize); the test MSE as ``AR_TEST test_mse=`` for the sealed record only, both
through the runner's result channel.  The loss lines ``step <n> loss <training MSE>`` are printed by the frozen
loop in the worker.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "gc7rc_etth1"
FILES = ["series.npy"]
SEQ_LEN = 336
HORIZON = 96
N_FEATURES = 7
NUM_TRAIN = 12 * 30 * 24  # 8,640 hourly rows, the LTSF 12-month training period
NUM_VAL = 4 * 30 * 24     # 2,880
NUM_TEST = 4 * 30 * 24    # 2,880
EXPECTED_ROWS = 17_420
# prepare_data.py's pinned download (ETDataset commit read by the U1 audit, 26 Nov 2026): the source file's
# sha256; checked here for every non-synthetic data manifest so a re-download of the source data is refused.
SOURCE_URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
              "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv")
SOURCE_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"


def window_starts(first_row: int, last_row: int) -> np.ndarray:
    """Absolute start rows of the look-back windows whose 96-step target ends before *last_row*."""
    return np.arange(first_row, last_row - SEQ_LEN - HORIZON + 1, dtype=np.int64)


def mse(target: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error on the standardised scale; ``nan`` if any prediction is not finite."""
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.mean((np.asarray(target, dtype=np.float64) - pred) ** 2))


def main() -> None:
    budget = P.read_budget("steps")
    train_seed, split_seed = P.read_seeds(default_split=0,
                                          split_note="; the split is chronological and does not use it")
    data_dir = ar_data.data_dir_of(__file__)
    manifest = ar_data.verify(data_dir, NODE, FILES)
    source = manifest.get("source", {})
    synthetic = source.get("synthetic") is True
    series_digest = manifest["files"]["series.npy"]["sha256"]
    if not synthetic and (source.get("sha256") != SOURCE_SHA256 or source.get("url") != SOURCE_URL):
        P.fail(P.EXIT_BAD_INVOCATION,
               f"the ETTh1 data were not set up from the pinned ETDataset download "
               f"(data manifest source: {source}); run prepare_data.py")
    print(f"AR_DATA series_sha256={series_digest} source_sha256={source.get('sha256')} "
          f"source_url={source.get('url')}"
          f"{' synthetic (port tests only; not the task data)' if synthetic else ''}", flush=True)
    series = np.load(data_dir / "series.npy")
    if series.shape != (EXPECTED_ROWS, N_FEATURES) or not np.all(np.isfinite(series)):
        P.fail(P.EXIT_BAD_INVOCATION, f"series.npy must be finite ({EXPECTED_ROWS}, {N_FEATURES}); "
                                      f"got {series.shape}")
    mean = series[:NUM_TRAIN].mean(axis=0)
    std = series[:NUM_TRAIN].std(axis=0)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        P.fail(P.EXIT_BAD_INVOCATION, "the training rows have a non-finite or zero standardisation statistic")
    scaled = ((series - mean) / std).astype(np.float32)

    starts = {
        "train": window_starts(0, NUM_TRAIN),
        "val": window_starts(NUM_TRAIN - SEQ_LEN, NUM_TRAIN + NUM_VAL),
        "test": window_starts(NUM_TRAIN + NUM_VAL - SEQ_LEN, NUM_TRAIN + NUM_VAL + NUM_TEST),
    }
    # The split is fixed and chronological: the split seed is recorded but changes nothing (as for ogbn_arxiv).
    print(f"AR_SPLIT train={len(starts['train'])} val={len(starts['val'])} test={len(starts['test'])} "
          f"seq_len={SEQ_LEN} horizon={HORIZON} features={N_FEATURES} "
          f"rows={NUM_TRAIN}+{NUM_VAL}+{NUM_TEST} split_seed={split_seed} (unused)", flush=True)

    def targets(name: str) -> np.ndarray:
        rows = starts[name]
        return np.stack([scaled[s + SEQ_LEN:s + SEQ_LEN + HORIZON].reshape(-1) for s in rows]).astype(np.float32)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed,
                             "seq_len": SEQ_LEN, "horizon": HORIZON, "n_features": N_FEATURES},
           {"series": scaled, "train_starts": starts["train"]})
    _, val_arrays = P.call(worker, "predict", {"split": "val"}, {"starts": starts["val"]})
    val_pred = P.check_array(val_arrays, "predictions", (len(starts["val"]), HORIZON * N_FEATURES))
    _, test_arrays = P.call(worker, "predict", {"split": "test"}, {"starts": starts["test"]})
    test_pred = P.check_array(test_arrays, "predictions", (len(starts["test"]), HORIZON * N_FEATURES))
    P.stop_worker(worker)
    # The data are checked again: a run whose worker changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")

    val_mse = mse(targets("val"), val_pred)
    test_mse = mse(targets("test"), test_pred)
    if np.isfinite(val_mse):
        y = targets("val")
        r2 = 1.0 - float(np.sum((y - val_pred) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        print(f"AR_INFO val_mse={val_mse:.6f} val_r2={r2:.6f}", flush=True)
    CHANNEL.write(f"AR_METRIC val_mse={val_mse:.6f}")
    CHANNEL.write(f"AR_TEST test_mse={test_mse:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
