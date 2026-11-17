"""Frozen worker entry of node mlab_imdb (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the
runner's result channel.  ``train``: load the tokenized reviews, take the parent's training and validation
indices, seed, import the editable train.py (offline: ``HF_HUB_OFFLINE=1``) and run exactly ``budget``
optimizer steps with the frozen loop (ar_fixed_loop); ``logits``: the validation or test logits as a float32
array, in order.  The metric is computed by the parent.
"""
import os

import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "mlab_imdb"
N_CLASSES = 2
MAX_BATCH_SIZE = 128
FILES = ["train_ids.npy", "train_mask.npy", "train_y.npy", "test_ids.npy", "test_mask.npy", "test_y.npy"]
CHECKED = FILES + ["pretrained/config.json", "pretrained/model.safetensors"]
STATE = {}


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, CHECKED)
    arr = {name[:-4]: torch.from_numpy(np.load(data_dir / name)) for name in FILES}
    tr_idx = torch.from_numpy(np.asarray(arrays["train_idx"], dtype=np.int64))
    val_idx = torch.from_numpy(np.asarray(arrays["val_idx"], dtype=np.int64))

    def part(prefix, idx=None):
        ids, mask, y = arr[f"{prefix}_ids"], arr[f"{prefix}_mask"], arr[f"{prefix}_y"].long()
        if idx is not None:
            ids, mask, y = ids[idx], mask[idx], y[idx]
        return ids.to(device), mask.to(device), y.to(device)

    tr = part("train", tr_idx)
    STATE["eval"] = {"val": part("train", val_idx), "test": part("test")}

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
    STATE.update(T=T, model=model, config=config, device=device)
    return {}, {}


def logits(meta, arrays, fds):
    split = STATE["eval"][meta["split"]]
    out = L.logits_array(STATE["T"], STATE["model"],
                         lambda idx: {"input_ids": split[0][idx], "attention_mask": split[1][idx]},
                         len(split[2]), N_CLASSES, STATE["config"], STATE["device"])
    return {}, {"logits": out}


iso.serve({"train": train, "logits": logits}, CONN)
