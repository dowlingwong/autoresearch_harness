"""Frozen wrapper of node mlgym_houseprice (run command: ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  Data: MLGym's Kaggle House Prices table (1,460 rows) as stored by
prepare_data.py; its sha256 must be the one recorded there.  Split: MLGym's
own procedure (``_helpers/_generate_split.py``): 20 % test with
random_state 42 (fixed, the suite's test set), then 12.5 % of the rest as
validation with random_state = AR_SPLIT_SEED (split seed 42 reproduces MLGym's
files exactly).  The editable train.py builds an unfitted scikit-learn style
estimator from the training features and the training seed, and may
transform the target; this wrapper fits it once on the training rows and
scores predictions on the original price scale.

Budget: ``--steps`` must be 1 -- one ``fit`` call.  This task has no training
steps, so the budget does not bound the compute of an edit (a heavier
estimator only takes longer); the port report raises this for the author.
Selection metric: validation RMSE (``AR_METRIC val_rmse=``, minimize); the
test RMSE is printed as ``AR_TEST test_rmse=`` for the sealed record only.
The loss line ``step 1 loss <training RMSE>`` is what the outcome parser
checks for a non-finite fit.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import ar_data

NODE = "mlgym_houseprice"
FILES = ["houseprice.csv"]
PINNED_SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"
EXIT_BAD_INVOCATION = 2
EXIT_BAD_EDITABLE_API = 4


def fail(code: int, message: str) -> None:
    print(f"AR_ERROR {message}", flush=True)
    raise SystemExit(code)


def read_budget() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Track G wrapper (one fit)")
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.steps != 1:
        fail(EXIT_BAD_INVOCATION, f"this node's budget is one fit (--steps 1), got {args.steps}")
    if os.environ.get("AR_BUDGET_KIND", "steps") != "steps" or int(os.environ.get("AR_BUDGET_VALUE", "1")) != 1:
        fail(EXIT_BAD_INVOCATION, "AR_BUDGET_KIND/AR_BUDGET_VALUE disagree with --steps 1")
    return 1


def read_seeds() -> tuple[int, int]:
    import os

    raw_train, raw_split = os.environ.get("AR_TRAIN_SEED"), os.environ.get("AR_SPLIT_SEED")
    train = int(raw_train) if raw_train is not None else 0
    split = int(raw_split) if raw_split is not None else 42
    print(f"AR_SEEDS train={train}{'' if raw_train is not None else ' (default)'} "
          f"split={split}{'' if raw_split is not None else ' (default, MLGym split)'}", flush=True)
    return train, split


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def main() -> None:
    read_budget()
    train_seed, split_seed = read_seeds()
    data_dir = ar_data.data_dir_of(__file__)
    manifest = ar_data.verify(data_dir, NODE, FILES)
    if manifest["files"]["houseprice.csv"]["sha256"] != PINNED_SHA256:
        if manifest.get("source", {}).get("synthetic") is True:
            print("AR_DATA synthetic table (port tests only; not the task data)", flush=True)
        else:
            fail(ar_data.EXIT_DATA, "data/houseprice.csv is not MLGym's file")
    df = pd.read_csv(data_dir / "houseprice.csv")
    rows = np.arange(len(df))
    temp, test = train_test_split(rows, test_size=0.2, random_state=42)
    train, val = train_test_split(temp, test_size=0.125, random_state=split_seed)
    features = df.drop(columns=["Id", "SalePrice"])
    target = df["SalePrice"].to_numpy(dtype=np.float64)
    x_tr, x_val, x_te = (features.iloc[i].reset_index(drop=True) for i in (train, val, test))
    y_tr, y_val, y_te = target[train], target[val], target[test]
    print(f"AR_SPLIT train={len(train)} val={len(val)} test={len(test)}", flush=True)

    import train as T  # the editable code

    for name in ("build_model", "transform_target", "inverse_transform_target"):
        if not callable(getattr(T, name, None)):
            fail(EXIT_BAD_EDITABLE_API, f"train.py must define {name}()")
    model = T.build_model(x_tr.copy(), train_seed)
    model.fit(x_tr.copy(), np.asarray(T.transform_target(y_tr.copy()), dtype=np.float64))

    def predict(x):
        pred = np.asarray(T.inverse_transform_target(np.asarray(model.predict(x.copy()))), dtype=np.float64)
        if pred.shape != (len(x),):
            fail(EXIT_BAD_EDITABLE_API, f"predictions must have shape ({len(x)},), got {pred.shape}")
        return pred

    pred_tr = predict(x_tr)
    print(f"step 1 loss {rmse(y_tr, pred_tr) if np.all(np.isfinite(pred_tr)) else float('nan'):.6f}", flush=True)
    ar_data.emit("AR_STEPS completed=1")
    pred_val = predict(x_val)
    val_rmse = rmse(y_val, pred_val) if np.all(np.isfinite(pred_val)) else float("nan")
    print(f"AR_INFO val_r2={r2_score(y_val, pred_val) if np.isfinite(val_rmse) else float('nan'):.6f}", flush=True)
    ar_data.emit(f"AR_METRIC val_rmse={val_rmse:.6f}")
    pred_te = predict(x_te)
    test_rmse = rmse(y_te, pred_te) if np.all(np.isfinite(pred_te)) else float("nan")
    ar_data.emit(f"AR_TEST test_rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main()
    ar_data.finish()  # no output of the editable code after the result lines
