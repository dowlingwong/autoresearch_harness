"""Frozen worker entry of node mlgym_houseprice (R1-HARNESS-ISOLATION): the only process that imports train.py.

FROZEN.  Started by the parent (run_fixed_budget.py) with one end of a socket pair; it never sees the runner's
result channel.  ``fit_predict``: load the table, take the parent's train / validation / test rows, import the
editable train.py, build the estimator from a copy of the training features and the training seed, fit it once
on the transformed training target, and return the predictions (inverse-transformed to the price scale) for
the three row sets.  The metric is computed by the parent.
"""
import ar_isolated as iso
CONN = iso.worker_connect()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ar_data  # noqa: E402

NODE = "mlgym_houseprice"
FILES = ["houseprice.csv"]
EXIT_BAD_EDITABLE_API = 4


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    raise SystemExit(code)


def fit_predict(meta, arrays, fds):
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, FILES)
    df = pd.read_csv(data_dir / "houseprice.csv")
    features = df.drop(columns=["Id", "SalePrice"])
    target = df["SalePrice"].to_numpy(dtype=np.float64)
    sets = {name: np.asarray(arrays[f"{name}_rows"], dtype=np.int64) for name in ("train", "val", "test")}
    x = {name: features.iloc[rows].reset_index(drop=True) for name, rows in sets.items()}

    import train as T  # the editable code

    for name in ("build_model", "transform_target", "inverse_transform_target"):
        if not callable(getattr(T, name, None)):
            fail(EXIT_BAD_EDITABLE_API, f"train.py must define {name}()")
    model = T.build_model(x["train"].copy(), int(meta["train_seed"]))
    model.fit(x["train"].copy(), np.asarray(T.transform_target(target[sets["train"]].copy()), dtype=np.float64))
    out = {}
    for name in ("train", "val", "test"):
        pred = np.asarray(T.inverse_transform_target(np.asarray(model.predict(x[name].copy()))), dtype=np.float64)
        if pred.shape != (len(x[name]),):
            fail(EXIT_BAD_EDITABLE_API, f"predictions must have shape ({len(x[name])},), got {pred.shape}")
        out[name] = pred
    return {}, out


iso.serve({"fit_predict": fit_predict}, CONN)
