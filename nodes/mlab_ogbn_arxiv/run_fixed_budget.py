"""Frozen fixed-step wrapper of node mlab_ogbn_arxiv (run command: ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  Data: ogbn-arxiv as stored by prepare_data.py and checked against
data/data_manifest.json.  Split: the dataset's official split (train 90,941 /
valid 29,799 / test 48,603 papers, by publication year); AR_SPLIT_SEED is read
and logged but has no effect, because the split is fixed by the benchmark.
The editable code gets the full feature matrix and edge list on the device in
``info`` (labels of validation and test nodes are masked to -1 there).  A
training batch is ``{"idx": train node ids, "y": their labels}``; evaluation
calls ``predict_logits(model, {"idx": node ids})`` once per split.  Selection
metric: validation accuracy (``AR_METRIC val_accuracy=``); the test accuracy is
printed as ``AR_TEST test_accuracy=`` for the sealed record only.  Budget:
exactly ``--steps`` optimizer steps (ar_fixed_loop).
"""
import numpy as np
import torch

import ar_data
import ar_fixed_loop as L

NODE = "mlab_ogbn_arxiv"
N_CLASSES = 40
MAX_BATCH_SIZE = 90941
FILES = ["x.npy", "y.npy", "edge_index.npy", "train_idx.npy", "valid_idx.npy", "test_idx.npy"]


@torch.no_grad()
def predict_split(T, model, idx, config, device):
    amp = bool(config.get("amp_bf16", False)) and device.type == "cuda"
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        logits = T.predict_logits(model, {"idx": idx})
    if type(logits) is not torch.Tensor or tuple(logits.shape) != (len(idx), N_CLASSES):
        L.fail(L.EXIT_BAD_EDITABLE_API, f"predict_logits must return a plain torch.Tensor of shape ({len(idx)}, {N_CLASSES})")
    with torch._C.DisableTorchFunction():  # no __torch_function__ subclass or mode of the editable code
        return logits.float().argmax(dim=1)


def main() -> None:
    budget = L.read_budget()
    train_seed, split_seed = L.read_seeds()
    print("AR_SPLIT official ogbn-arxiv split; the split seed has no effect", flush=True)
    device = L.pick_device()
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    arr = {name[:-4]: torch.from_numpy(np.load(data_dir / name)) for name in FILES}
    x = arr["x"].to(device)
    y = arr["y"].long().to(device)
    edge_index = arr["edge_index"].long().to(device)
    tr_idx, va_idx, te_idx = (arr[k].long().to(device) for k in ("train_idx", "valid_idx", "test_idx"))
    y_train_only = torch.full_like(y, -1)
    y_train_only[tr_idx] = y[tr_idx]
    print(f"AR_SPLIT train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} nodes={len(y)} "
          f"edges={edge_index.shape[1]}", flush=True)

    L.seed_everything(train_seed)
    import train as T  # the editable code, imported after seeding

    config = L.check_api(T)
    info = {"device": device, "num_classes": N_CLASSES, "x": x, "edge_index": edge_index, "num_nodes": len(y),
            "y_train_only": y_train_only, "n_train": len(tr_idx)}
    model = T.build_model(info)
    optimizer, scheduler = T.build_optimizer(model, budget)
    sampler = L.FixedStepSampler(len(tr_idx), config["batch_size"], budget, train_seed, MAX_BATCH_SIZE)
    L.train_fixed_steps(T, model, optimizer, scheduler, sampler,
                        lambda pos: {"idx": tr_idx[pos], "y": y[tr_idx[pos]]}, config, device)

    val_pred = predict_split(T, model, va_idx, config, device)
    test_pred = predict_split(T, model, te_idx, config, device)
    # Both evaluations (editable predict_logits) and the accuracies run before the first result line, which
    # only formats Python floats: nothing editable runs after it.
    val_accuracy, test_accuracy = L.accuracy(val_pred, y[va_idx]), L.accuracy(test_pred, y[te_idx])
    # The data are checked again: a run whose editable code changed them fails itself, not the runs after it.
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")
    ar_data.seal()  # stdout/stderr to /dev/null: threads of the editable code cannot print now
    ar_data.emit(f"AR_METRIC val_accuracy={val_accuracy:.6f}")
    ar_data.emit(f"AR_TEST test_accuracy={test_accuracy:.6f}")


if __name__ == "__main__":
    main()
    ar_data.finish()  # no output of the editable code after the result lines
