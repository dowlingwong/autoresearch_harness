"""Frozen parent of node mlgym_fmnist (R1-HARNESS-ISOLATION; started by the runner as ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code.  It reads the budget and the seeds, checks the data
(Fashion-MNIST from the Hugging Face dataset zalando-datasets/fashion_mnist, the dataset MLGym's task loads,
stored by prepare_data.py as uint8 arrays and checked against data/data_manifest.json), makes the split (the
60,000 training images split by AR_SPLIT_SEED into 54,000 for training and 6,000 for validation; the 10,000
test images are the suite's test set), and starts the frozen worker ``ar_worker.py``, which imports train.py,
trains for exactly ``--steps`` optimizer steps (ar_fixed_loop) and returns the validation and test logits.
The accuracies are computed here from those logits and the labels this process loaded, and written to the
runner's result channel: selection metric ``AR_METRIC val_accuracy=``; the test accuracy as
``AR_TEST test_accuracy=`` for the sealed record only.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "mlgym_fmnist"
N_CLASSES = 10
VAL_SIZE = 6000
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]


def main() -> None:
    budget = P.read_budget("steps")
    train_seed, split_seed = P.read_seeds()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    train_y = np.load(data_dir / "train_y.npy").astype(np.int64)
    test_y = np.load(data_dir / "test_y.npy").astype(np.int64)
    perm = np.random.RandomState(split_seed).permutation(len(train_y))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    print(f"AR_SPLIT train={len(tr_idx)} val={len(val_idx)} test={len(test_y)}", flush=True)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed},
           {"train_idx": tr_idx.astype(np.int64), "val_idx": val_idx.astype(np.int64)})
    _, val = P.call(worker, "logits", {"split": "val"})
    val_logits = P.check_array(val, "logits", (len(val_idx), N_CLASSES))
    _, test = P.call(worker, "logits", {"split": "test"})
    test_logits = P.check_array(test, "logits", (len(test_y), N_CLASSES))
    P.stop_worker(worker)
    # The data are checked again: a run whose worker changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")
    val_accuracy = P.accuracy(val_logits, train_y[val_idx])
    test_accuracy = P.accuracy(test_logits, test_y)
    CHANNEL.write(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    CHANNEL.write(f"AR_TEST test_accuracy={test_accuracy:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
