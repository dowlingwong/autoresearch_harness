"""Frozen fixed-step wrapper of node mlgym_cifar10 (run command: ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  Data: CIFAR-10 from the Hugging Face dataset uoft-cs/cifar10 (the
dataset MLGym's task loads), stored by prepare_data.py as uint8 arrays and
checked against data/data_manifest.json.
Split: the 50,000 training images are split by AR_SPLIT_SEED into 45,000 for
training and 5,000 for validation; the 10,000 test images are the suite's test
set.  Selection metric: validation accuracy (``AR_METRIC val_accuracy=``);
the test accuracy is printed as ``AR_TEST test_accuracy=`` for the sealed
record only.  Budget: exactly ``--steps`` optimizer steps (ar_fixed_loop).
"""
import numpy as np
import torch

import ar_data
import ar_fixed_loop as L

NODE = "mlgym_cifar10"
N_CLASSES = 10
VAL_SIZE = 5000
MAX_BATCH_SIZE = 4096
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]


def main() -> None:
    budget = L.read_budget()
    train_seed, split_seed = L.read_seeds()
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    arr = {name[:-4]: np.load(data_dir / name) for name in FILES}
    x_all = torch.from_numpy(arr["train_x"]).permute(0, 3, 1, 2).contiguous()  # (N, 3, 32, 32) uint8
    y_all = torch.from_numpy(arr["train_y"]).long()
    perm = torch.from_numpy(np.random.RandomState(split_seed).permutation(len(x_all)))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    x_tr, y_tr = x_all[tr_idx].to(device), y_all[tr_idx].to(device)
    x_val, y_val = x_all[val_idx].to(device), y_all[val_idx].to(device)
    x_te = torch.from_numpy(arr["test_x"]).permute(0, 3, 1, 2).contiguous().to(device)
    y_te = torch.from_numpy(arr["test_y"]).long().to(device)
    print(f"AR_SPLIT train={len(x_tr)} val={len(x_val)} test={len(x_te)}", flush=True)

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "num_classes": N_CLASSES, "input_shape": (3, 32, 32), "n_train": len(x_tr)}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(x_tr), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda idx: {"x": x_tr[idx], "y": y_tr[idx]}, config, device)

    val_pred = L.predict_classes(T, model, lambda idx: {"x": x_val[idx]}, len(x_val), N_CLASSES, config, device)
    ar_data.emit(f"AR_METRIC val_accuracy={L.accuracy(val_pred, y_val):.6f}")
    test_pred = L.predict_classes(T, model, lambda idx: {"x": x_te[idx]}, len(x_te), N_CLASSES, config, device)
    ar_data.emit(f"AR_TEST test_accuracy={L.accuracy(test_pred, y_te):.6f}")


if __name__ == "__main__":
    main()
    ar_data.finish()  # no output of the editable code after the result lines
