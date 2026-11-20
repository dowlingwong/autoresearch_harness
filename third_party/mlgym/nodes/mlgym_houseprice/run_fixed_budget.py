"""Frozen parent of node mlgym_houseprice (R1-HARNESS-ISOLATION; started by the runner as ``python3 run_fixed_budget.py --steps {budget}``).

FROZEN.  This process never imports the editable code.  Data: the Kaggle "House Prices - Advanced Regression
Techniques" training table (train.csv, 1,460 rows) that the user downloaded from Kaggle and passed to
prepare_data.py with ``--from-file`` (R1-TRACKG-PORT-DECISIONS (2)); its sha256 is in the data manifest and is
printed here (``AR_DATA houseprice_sha256= md5=``) for the census record, after checking that the manifest
records the user's download.  Split: MLGym's own procedure
(``_helpers/_generate_split.py``): 20 % test with random_state 42 (fixed, the suite's test set), then 12.5 % of
the rest as validation with random_state = AR_SPLIT_SEED (split seed 42 reproduces MLGym's split rows).  The
frozen worker ``ar_worker.py`` imports train.py, builds the unfitted estimator from the training features and
the training seed, fits it once on the training rows (with train.py's target transform) and returns
predictions on the original price scale; the RMSEs are computed here with the prices this process loaded.

Budget: ``--steps`` must be 1 -- one ``fit`` call.  This task has no training steps, so the budget does not
bound the compute of an edit (a heavier estimator only takes longer); the port report raises this for the
author.  Selection metric: validation RMSE (``AR_METRIC val_rmse=``, minimize); the test RMSE as
``AR_TEST test_rmse=`` for the sealed record only, both through the runner's result channel.  The loss line
``step 1 loss <training RMSE>`` (printed here) is what the outcome parser checks for a non-finite fit.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

NODE = "mlgym_houseprice"
FILES = ["houseprice.csv"]
# MLGym's copy of the table (data/regressionKaggleHousePrice/_helpers/train.csv at the commit the port read):
# compared for information only; the run uses the user's Kaggle download, whatever its hash.
MLGYM_COPY_SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"
USER_DOWNLOAD = "user download"   # prepare_data.py's source.provided_by prefix


def main() -> None:
    P.read_budget("steps", exact=1)
    train_seed, split_seed = P.read_seeds(default_split=42, split_note=", MLGym split")
    data_dir = ar_data.data_dir_of(__file__)
    manifest = ar_data.verify(data_dir, NODE, FILES)
    digest = manifest["files"]["houseprice.csv"]["sha256"]
    source = manifest.get("source", {})
    synthetic = source.get("synthetic") is True
    # Kit review round 1, M1 (R1-TRACKG-PORT-DECISIONS (2)): the table must be the user's Kaggle download, as
    # recorded by prepare_data.py with its hashes; anything else (MLGym's forwarded copy set up by an older kit)
    # is refused.  Synthetic tables exist only in the port tests and are marked as such.
    if not synthetic and (not str(source.get("provided_by", "")).startswith(USER_DOWNLOAD)
                          or source.get("sha256") != digest or not source.get("md5")):
        P.fail(P.EXIT_BAD_INVOCATION, f"the House Prices data were not set up from the user's Kaggle download "
                                      f"(data manifest source: {source}); run prepare_data.py --from-file")
    print(f"AR_DATA houseprice_sha256={digest} md5={source.get('md5')} same_as_mlgym_copy={digest == MLGYM_COPY_SHA256}"
          f"{' synthetic (port tests only; not the task data)' if synthetic else ''}", flush=True)
    df = pd.read_csv(data_dir / "houseprice.csv")
    rows = np.arange(len(df))
    temp, test = train_test_split(rows, test_size=0.2, random_state=42)
    train, val = train_test_split(temp, test_size=0.125, random_state=split_seed)
    target = df["SalePrice"].to_numpy(dtype=np.float64)
    print(f"AR_SPLIT train={len(train)} val={len(val)} test={len(test)}", flush=True)

    worker = P.start_worker()
    _, preds = P.call(worker, "fit_predict", {"train_seed": train_seed},
                      {"train_rows": train.astype(np.int64), "val_rows": val.astype(np.int64),
                       "test_rows": test.astype(np.int64)})
    pred_tr = P.check_array(preds, "train", (len(train),))
    pred_val = P.check_array(preds, "val", (len(val),))
    pred_te = P.check_array(preds, "test", (len(test),))
    P.stop_worker(worker)
    ar_data.verify(data_dir, NODE, FILES, prefix="AR_ERROR the editable code changed the data:")
    print(f"step 1 loss {P.rmse(target[train], pred_tr):.6f}", flush=True)
    print("AR_STEPS completed=1", flush=True)
    val_rmse = P.rmse(target[val], pred_val)
    test_rmse = P.rmse(target[test], pred_te)
    if np.isfinite(val_rmse):
        y = target[val]
        r2 = 1.0 - float(np.sum((y - pred_val) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        print(f"AR_INFO val_r2={r2:.6f}", flush=True)
    CHANNEL.write(f"AR_METRIC val_rmse={val_rmse:.6f}")
    CHANNEL.write(f"AR_TEST test_rmse={test_rmse:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
