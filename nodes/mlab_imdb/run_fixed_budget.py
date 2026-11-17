"""Frozen fixed-step wrapper of node mlab_imdb (run command: ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  Data: IMDb reviews tokenized once by prepare_data.py (512 tokens,
distilbert-base-uncased WordPiece) and checked against data/data_manifest.json;
the pretrained model files are in data/pretrained/.  Split: the 25,000
training reviews are split by AR_SPLIT_SEED into 22,500 for training and 2,500
for validation; the 25,000 test reviews are the suite's test set.  Selection
metric: validation accuracy (``AR_METRIC val_accuracy=``); the test accuracy is
printed as ``AR_TEST test_accuracy=`` for the sealed record only.  Budget:
exactly ``--steps`` optimizer steps (ar_fixed_loop).
"""
import os

import numpy as np
import torch

import ar_data
import ar_fixed_loop as L

NODE = "mlab_imdb"
N_CLASSES = 2
VAL_SIZE = 2500
MAX_BATCH_SIZE = 128
FILES = ["train_ids.npy", "train_mask.npy", "train_y.npy", "test_ids.npy", "test_mask.npy", "test_y.npy"]


def main() -> None:
    budget = L.read_budget()
    train_seed, split_seed = L.read_seeds()
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES + ["pretrained/config.json", "pretrained/model.safetensors"])
    arr = {name[:-4]: torch.from_numpy(np.load(data_dir / name)) for name in FILES}
    n = len(arr["train_y"])
    perm = torch.from_numpy(np.random.RandomState(split_seed).permutation(n))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

    def part(prefix, idx=None):
        ids, mask, y = arr[f"{prefix}_ids"], arr[f"{prefix}_mask"], arr[f"{prefix}_y"].long()
        if idx is not None:
            ids, mask, y = ids[idx], mask[idx], y[idx]
        return ids.to(device), mask.to(device), y.to(device)

    tr, val, te = part("train", tr_idx), part("train", val_idx), part("test")
    print(f"AR_SPLIT train={len(tr[2])} val={len(val[2])} test={len(te[2])}", flush=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # runs never reach the network
    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "num_classes": N_CLASSES, "pretrained_dir": str(data_dir / "pretrained"),
            "max_tokens": tr[0].shape[1], "n_train": len(tr[2])}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(tr[2]), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda idx: {"input_ids": tr[0][idx], "attention_mask": tr[1][idx], "y": tr[2][idx]},
                        config, device)

    def evaluation(split):
        return lambda idx: {"input_ids": split[0][idx], "attention_mask": split[1][idx]}

    val_pred = L.predict_classes(T, model, evaluation(val), len(val[2]), N_CLASSES, config, device)
    test_pred = L.predict_classes(T, model, evaluation(te), len(te[2]), N_CLASSES, config, device)
    # Both evaluations (editable predict_logits) and the accuracies run before the first result line, which
    # only formats Python floats: nothing editable runs after it.
    val_accuracy, test_accuracy = L.accuracy(val_pred, val[2]), L.accuracy(test_pred, te[2])
    # The data are checked again: a run whose editable code changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES + ["pretrained/config.json", "pretrained/model.safetensors"], prefix="AR_ERROR the editable code changed the data:")
    ar_data.seal()  # stdout/stderr to /dev/null: threads of the editable code cannot print now
    ar_data.emit(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    ar_data.emit(f"AR_TEST test_accuracy={test_accuracy:.6f}")


if __name__ == "__main__":
    main()
    ar_data.finish()  # no output of the editable code after the result lines
