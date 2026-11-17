"""Frozen worker entry of node mlab_ogbn_arxiv (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the
runner's result channel.  ``train``: load the graph (official split), seed, import the editable train.py, give
it the features and edges with the labels of validation and test nodes masked to -1, and run exactly
``budget`` optimizer steps with the frozen loop (ar_fixed_loop); ``logits``: ``predict_logits(model,
{"idx": node ids})`` once for the validation or test nodes, as a float32 array.  The metric is computed by the
parent.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ar_data  # noqa: E402
import ar_fixed_loop as L  # noqa: E402

NODE = "mlab_ogbn_arxiv"
N_CLASSES = 40
MAX_BATCH_SIZE = 90941
FILES = ["x.npy", "y.npy", "edge_index.npy", "train_idx.npy", "valid_idx.npy", "test_idx.npy"]
STATE = {}


@torch.no_grad()
def predict_split(T, model, idx, config, device) -> np.ndarray:
    amp = bool(config.get("amp_bf16", False)) and device.type == "cuda"
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        logits = T.predict_logits(model, {"idx": idx})
    if type(logits) is not torch.Tensor or tuple(logits.shape) != (len(idx), N_CLASSES):
        L.fail(L.EXIT_BAD_EDITABLE_API, f"predict_logits must return a plain torch.Tensor of shape ({len(idx)}, {N_CLASSES})")
    with torch._C.DisableTorchFunction():
        return logits.detach().float().cpu().numpy()


def train(meta, arrays, fds):
    budget, train_seed = int(meta["budget"]), int(meta["train_seed"])
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
    STATE.update(T=T, model=model, config=config, device=device, eval={"val": va_idx, "test": te_idx})
    return {}, {}


def logits(meta, arrays, fds):
    idx = STATE["eval"][meta["split"]]
    return {}, {"logits": predict_split(STATE["T"], STATE["model"], idx, STATE["config"], STATE["device"])}


iso.serve({"train": train, "logits": logits}, CONN)
