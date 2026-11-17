"""Frozen parent of node mlab_ogbn_arxiv (R1-HARNESS-ISOLATION; started by the runner as ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code.  Data: ogbn-arxiv as stored by prepare_data.py and
checked against data/data_manifest.json.  Split: the dataset's official split (train 90,941 / valid 29,799 /
test 48,603 papers, by publication year); AR_SPLIT_SEED is read and logged but has no effect, because the
split is fixed by the benchmark.  The frozen worker ``ar_worker.py`` imports train.py, gives it the full
feature matrix and edge list (labels of validation and test nodes masked to -1), trains for exactly
``--steps`` optimizer steps (ar_fixed_loop; a training batch is ``{"idx": train node ids, "y": their labels}``)
and returns the validation and test logits (``predict_logits(model, {"idx": node ids})`` once per split).  The
accuracies are computed here with the labels this process loaded and written to the runner's result channel:
selection metric ``AR_METRIC val_accuracy=``; the test accuracy as ``AR_TEST test_accuracy=`` for the sealed
record only.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "mlab_ogbn_arxiv"
N_CLASSES = 40
FILES = ["x.npy", "y.npy", "edge_index.npy", "train_idx.npy", "valid_idx.npy", "test_idx.npy"]


def main() -> None:
    budget = P.read_budget("steps")
    train_seed, _split_seed = P.read_seeds()
    print("AR_SPLIT official ogbn-arxiv split; the split seed has no effect", flush=True)
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    y = np.load(data_dir / "y.npy").astype(np.int64).reshape(-1)
    va_idx = np.load(data_dir / "valid_idx.npy").astype(np.int64)
    te_idx = np.load(data_dir / "test_idx.npy").astype(np.int64)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed})
    _, val = P.call(worker, "logits", {"split": "val"})
    val_logits = P.check_array(val, "logits", (len(va_idx), N_CLASSES))
    _, test = P.call(worker, "logits", {"split": "test"})
    test_logits = P.check_array(test, "logits", (len(te_idx), N_CLASSES))
    P.stop_worker(worker)
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")
    val_accuracy = P.accuracy(val_logits, y[va_idx])
    test_accuracy = P.accuracy(test_logits, y[te_idx])
    CHANNEL.write(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    CHANNEL.write(f"AR_TEST test_accuracy={test_accuracy:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
