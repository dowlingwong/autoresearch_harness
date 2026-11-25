"""Frozen parent of node mlgym_titanic (R1-HARNESS-ISOLATION; started by the runner as ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports editable code.  It reads the budget and the seeds, checks the task data
(the Kaggle "Titanic - Machine Learning from Disaster" tables MLGym's ``data/titanic`` task ships:
``train.csv`` 891 rows with the ``Survived`` label, unlabelled ``test.csv`` 418 rows and ``ground_truth.csv``
with the test labels; downloaded by prepare_data.py from MLGym's pinned commit, hashes in
data/data_manifest.json), makes the split (20 % of ``train.csv`` as validation, drawn by AR_SPLIT_SEED; the
rest trains; the task's own test table is the sealed test set) and starts the frozen worker ``ar_worker.py``,
which imports train.py, fits the editable feature encoder on the training rows only, trains for exactly
``--steps`` optimizer steps (ar_fixed_loop) and returns the validation and test logits.  The accuracies are
computed here from those logits and the labels this process loaded (the test labels are aligned to
``test.csv`` by ``PassengerId``), and written to the runner's result channel: selection metric
``AR_METRIC val_accuracy=``; the test accuracy as ``AR_TEST test_accuracy=`` for the sealed record only.

Budget: exactly ``--steps`` optimizer steps (R1-BX-D5): the step count is the node's fixed budget, so an
edit can change the compute per step (a wider model) but not the number of steps.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "mlgym_titanic"
N_CLASSES = 2
VAL_FRACTION = 0.2
FILES = ["train.csv", "test.csv", "ground_truth.csv"]


def main() -> None:
    budget = P.read_budget("steps")
    train_seed, split_seed = P.read_seeds(default_split=42, split_note=", port validation split")
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    truth = pd.read_csv(data_dir / "ground_truth.csv")
    if "Survived" not in train.columns or len(test) != len(truth) \
            or set(test["PassengerId"]) != set(truth["PassengerId"]):
        P.fail(P.EXIT_BAD_RESULT, "the task tables are not MLGym's titanic data: train.csv needs the Survived "
                                  "label and test.csv / ground_truth.csv must list the same passengers")
    labels = truth.set_index("PassengerId").loc[test["PassengerId"], "Survived"].to_numpy(dtype=np.int64)
    targets = train["Survived"].to_numpy(dtype=np.int64)
    n_val = int(round(VAL_FRACTION * len(targets)))
    perm = np.random.RandomState(split_seed).permutation(len(targets))
    val_rows, tr_rows = perm[:n_val], perm[n_val:]
    print(f"AR_SPLIT train={len(tr_rows)} val={len(val_rows)} test={len(test)}", flush=True)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed},
           {"train_rows": tr_rows.astype(np.int64), "val_rows": val_rows.astype(np.int64)})
    _, val = P.call(worker, "logits", {"split": "val"})
    val_logits = P.check_array(val, "logits", (len(val_rows), N_CLASSES))
    _, test_out = P.call(worker, "logits", {"split": "test"})
    test_logits = P.check_array(test_out, "logits", (len(test), N_CLASSES))
    P.stop_worker(worker)
    # The data are checked again: a run whose worker changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")
    val_accuracy = P.accuracy(val_logits, targets[val_rows])
    test_accuracy = P.accuracy(test_logits, labels)
    CHANNEL.write(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    CHANNEL.write(f"AR_TEST test_accuracy={test_accuracy:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
