"""Frozen parent of node gc7rc_forestcover (R1-HARNESS-ISOLATION; started by the runner as
``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code.  Data: the public UCI Covertype table (Forest Cover Type,
581,012 rows x 54 features: 10 continuous + 44 binary one-hot; the label is the UCI class 1-7) fetched by
``prepare_data.py`` from the pinned UCI zip and stored as ``data/X.npy`` (float32) and ``data/y.npy``
(int8 class 0-6) with a hashed data manifest; the source zip's sha256 and URL are pinned here and checked
against the manifest for the real (non-synthetic) data.  Split: the paper's Task 5 stratified 80/10/10 split
(464,810 / 58,101 / 58,101 rows), drawn with the fixed split seed 42 and a per-class seeded permutation, so
every run uses the same split.  Features are standardised with the training rows' per-feature mean and
population standard deviation (StandardScaler semantics) fitted on the training rows only.

The frozen worker ``ar_worker.py`` imports train.py, runs exactly ``--steps`` optimizer steps over the
training rows and returns the validation and test logits; the accuracies are computed here from the logits and
the labels this process loaded.

Budget: ``--steps`` optimizer steps (the spec's fixed step budget; R1-BX-D5).  Selection metric: validation
accuracy (``AR_METRIC val_accuracy=``, maximise); the test accuracy as ``AR_TEST test_accuracy=`` for the
sealed record only, both through the runner's result channel.  The loss lines ``step <n> loss <mean training
cross-entropy>`` are printed by the frozen loop in the worker.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "gc7rc_forestcover"
FILES = ["X.npy", "y.npy"]
N_ROWS = 581_012
N_FEATURES = 54
N_CLASSES = 7
SPLIT_SEED = 42          # fixed: the paper's Task 5 split is a single stratified 80/10/10 partition
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
# prepare_data.py's pinned download (UCI dataset 31, dataset page https://archive.ics.uci.edu/dataset/31/
# covertype, DOI 10.24432/C50K5N, CC BY 4.0): the source zip's sha256; checked here for every non-synthetic
# data manifest so a re-download from another source is refused.
SOURCE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
SOURCE_SHA256 = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"


def stratified_rows(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fixed stratified 80/10/10 split: per class, a seeded permutation, then 10 % validation and
    10 % test; the remaining 80 % trains.  Returns sorted row indices for (train, validation, test)."""
    rng = np.random.RandomState(SPLIT_SEED)
    train, validation, test = [], [], []
    for label in range(N_CLASSES):
        rows = np.flatnonzero(labels == label)
        rng.shuffle(rows)
        n_val = int(round(VAL_FRACTION * len(rows)))
        n_test = int(round(TEST_FRACTION * len(rows)))
        validation.append(rows[:n_val])
        test.append(rows[n_val:n_val + n_test])
        train.append(rows[n_val + n_test:])
    out = tuple(np.sort(np.concatenate(part)).astype(np.int64) for part in (train, validation, test))
    all_rows = np.concatenate(out)
    if len(all_rows) != N_ROWS or len(np.unique(all_rows)) != N_ROWS or not all(len(part) for part in out):
        P.fail(P.EXIT_BAD_INVOCATION, f"the stratified split is not a partition of {N_ROWS} rows: "
                                      f"{[len(part) for part in out]}")
    return out


def standardise(x: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler semantics: per-feature mean and population std from the training rows only."""
    mean = x[train_rows].mean(axis=0, dtype=np.float64)
    std = x[train_rows].std(axis=0, dtype=np.float64)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        P.fail(P.EXIT_BAD_INVOCATION, "the training rows have a non-finite or zero standardisation statistic")
    scaled = ((x.astype(np.float64) - mean) / std).astype(np.float32)
    return np.ascontiguousarray(scaled), mean, std


def main() -> None:
    budget = P.read_budget("steps")
    train_seed, split_seed = P.read_seeds(default_split=SPLIT_SEED,
                                          split_note="; the paper's stratified 80/10/10 split is fixed at seed 42 "
                                                     "and does not use it")
    data_dir = ar_data.data_dir_of(__file__)
    manifest = ar_data.verify(data_dir, NODE, FILES)
    source = manifest.get("source", {})
    synthetic = source.get("synthetic") is True
    if not synthetic and (source.get("zip_sha256") != SOURCE_SHA256 or source.get("url") != SOURCE_URL):
        P.fail(P.EXIT_BAD_INVOCATION,
               f"the Forest Cover data were not set up from the pinned UCI covertype.zip download "
               f"(data manifest source: {source}); run prepare_data.py")
    x = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    if x.shape != (N_ROWS, N_FEATURES) or x.dtype != np.float32 or not np.all(np.isfinite(x)):
        P.fail(P.EXIT_BAD_INVOCATION, f"X.npy must be finite float32 ({N_ROWS}, {N_FEATURES}); "
                                      f"got {x.shape} {x.dtype}")
    if y.shape != (N_ROWS,) or not np.issubdtype(y.dtype, np.integer) or int(y.min()) < 0 \
            or int(y.max()) >= N_CLASSES:
        P.fail(P.EXIT_BAD_INVOCATION, f"y.npy must be integer classes 0-{N_CLASSES - 1}, shape ({N_ROWS},); "
                                      f"got {y.shape} {y.dtype}")
    train_rows, val_rows, test_rows = stratified_rows(y)
    scaled, mean, std = standardise(x, train_rows)
    print(f"AR_SPLIT train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
          f"features={N_FEATURES} classes={N_CLASSES} split_seed={SPLIT_SEED} (fixed; "
          f"AR_SPLIT_SEED={split_seed} unused) standardised_on=train", flush=True)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed,
                             "n_features": N_FEATURES, "n_classes": N_CLASSES},
           {"X": scaled, "y": y.astype(np.int64),
            "train_rows": train_rows, "val_rows": val_rows, "test_rows": test_rows})
    _, val_arrays = P.call(worker, "logits", {"split": "val"})
    val_logits = P.check_array(val_arrays, "logits", (len(val_rows), N_CLASSES))
    _, test_arrays = P.call(worker, "logits", {"split": "test"})
    test_logits = P.check_array(test_arrays, "logits", (len(test_rows), N_CLASSES))
    P.stop_worker(worker)
    # The data are checked again: a run whose worker changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")

    val_accuracy = P.accuracy(val_logits, y[val_rows])
    test_accuracy = P.accuracy(test_logits, y[test_rows])
    # Only the selection metric is printed: the sealed test accuracy goes to the result channel only.
    print(f"AR_INFO val_accuracy={val_accuracy:.6f} "
          f"mean={np.round(mean[:3], 4).tolist()} std={np.round(std[:3], 4).tolist()}", flush=True)
    CHANNEL.write(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    CHANNEL.write(f"AR_TEST test_accuracy={test_accuracy:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
