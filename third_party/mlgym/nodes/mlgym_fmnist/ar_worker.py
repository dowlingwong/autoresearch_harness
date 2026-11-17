"""Frozen worker entry of node mlgym_fmnist (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the
runner's result channel.  ``train``: load the data, take the parent's training and validation indices, seed,
import the editable train.py and run exactly ``budget`` optimizer steps with the frozen loop (ar_fixed_loop:
optimizer step hook, CPU batch order, loss lines); ``logits``: the validation or test logits as a float32
array, in order.  The metric is computed by the parent.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "mlgym_fmnist"
N_CLASSES = 10
MAX_BATCH_SIZE = 4096
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]
STATE = {}


def images(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(1).contiguous()  # (N, 1, 28, 28) uint8


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    arr = {name[:-4]: np.load(data_dir / name) for name in FILES}
    x_all = images(arr["train_x"])
    y_all = torch.from_numpy(arr["train_y"]).long()
    tr_idx = torch.from_numpy(np.asarray(arrays["train_idx"], dtype=np.int64))
    val_idx = torch.from_numpy(np.asarray(arrays["val_idx"], dtype=np.int64))
    x_tr, y_tr = x_all[tr_idx].to(device), y_all[tr_idx].to(device)
    STATE["eval"] = {"val": x_all[val_idx].to(device), "test": images(arr["test_x"]).to(device)}

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "num_classes": N_CLASSES, "input_shape": (1, 28, 28), "n_train": len(x_tr)}
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
