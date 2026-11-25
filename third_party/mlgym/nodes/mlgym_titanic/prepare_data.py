"""Frozen one-off setup of node mlgym_titanic (the node's setup_command; never part of a run).

Fetches the three tables of MLGym's ``data/titanic`` task (configs/tasks/titanic.yaml, task id
``titanicSurvival``) from MLGym's **pinned commit** and nothing else:

  data/titanic/data/{train.csv,test.csv,ground_truth.csv}

They are the Kaggle "Titanic - Machine Learning from Disaster" competition tables that MLGym forwards in its
repository (train.csv 891 rows with the Survived label, unlabelled test.csv 418 rows, ground_truth.csv 418
rows with the test labels).  The port does **not** redistribute them: this script downloads them at setup
time, checks each file against the sha256 pinned here (the bytes of the pinned commit, so the download is
reproducible), stores them in --out, writes the hashed data manifest and links --out into the node as
``data``.  No Kaggle account or API is needed; see LICENCES.md for the data's terms.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_titanic"

Re-running with a complete --out only re-checks and re-links it.
--out must lie under ``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or new --out (a partial earlier
attempt is never reused); the manifest is written last and the directory is then made read-only, so
accidental writes by a run fail (the owner can restore the write bits; see ar_data).
"""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import ar_data

NODE = "mlgym_titanic"
REPOSITORY = "https://github.com/facebookresearch/MLGym"
COMMIT = "9d40c1b5035202018cd7091fb4e83a9c68b377c0"   # MLGym HEAD read by the U1 audit and the port
TASK = "titanicSurvival"
DATASET = "titanic"
BASE_URL = f"https://raw.githubusercontent.com/facebookresearch/MLGym/{COMMIT}/data/titanic/data"
FILES = ["train.csv", "test.csv", "ground_truth.csv"]
# The sha256 of each file inside the pinned upstream commit: the download is checked against these.
PINS = {
    "train.csv": "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7",
    "test.csv": "5c78320a80dc159b35c408363be46a5d93f078183d539db75ee42a796d6a4e95",
    "ground_truth.csv": "af6609cdf1dad0b699bd3414c13e999b2c2f651972f0938254a835985d584306",
}
SHAPES = {"train.csv": (891, 12), "test.csv": (418, 11), "ground_truth.csv": (418, 2)}
KAGGLE = "https://www.kaggle.com/competitions/titanic"
DOWNLOAD_READ_BYTES = 1 << 20


def fetch(url: str) -> bytes:
    """One HTTP GET of the pinned file (a fixed 120 s socket bound, not a training budget)."""
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - the pinned https URL
        return response.read()


def check_files(contents: dict[str, bytes]) -> None:
    """Every downloaded file's sha256 equals the pin; the tables' shapes are the task's."""
    for name in FILES:
        got = hashlib.sha256(contents[name]).hexdigest()
        if got != PINS[name]:
            raise SystemExit(f"{name} sha256 {got} differs from the pinned {PINS[name]}; nothing written")
    import io

    import pandas as pd

    train = pd.read_csv(io.BytesIO(contents["train.csv"]))
    test = pd.read_csv(io.BytesIO(contents["test.csv"]))
    truth = pd.read_csv(io.BytesIO(contents["ground_truth.csv"]))
    for name, frame in (("train.csv", train), ("test.csv", test), ("ground_truth.csv", truth)):
        if frame.shape != SHAPES[name]:
            raise SystemExit(f"{name} has shape {frame.shape}, expected {SHAPES[name]}; nothing written")
    if "Survived" not in train.columns or "Survived" not in truth.columns:
        raise SystemExit("train.csv / ground_truth.csv lack the Survived label; nothing written")
    if set(test["PassengerId"]) != set(truth["PassengerId"]):
        raise SystemExit("test.csv and ground_truth.csv list different PassengerIds; nothing written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    root = os.environ.get("AR_TRACKG_DATA_ROOT")
    if not root or not out.is_relative_to(Path(root).resolve()) or out == Path(root).resolve():
        raise SystemExit(f"--out must lie under $AR_TRACKG_DATA_ROOT (set: {bool(root)}); got {out}")
    node_dir = Path(__file__).resolve().parent
    if not (out / ar_data.MANIFEST_NAME).exists():
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty and has no data manifest (an earlier attempt?); move it aside "
                             "and run again into a new --out (nothing is reused)")
        contents = {}
        for name in FILES:
            url = f"{BASE_URL}/{name}"
            contents[name] = fetch(url)
            print(f"fetched {url} bytes={len(contents[name])}", flush=True)
        check_files(contents)
        out.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (out / name).write_bytes(contents[name])
            print(f"{name} sha256 {PINS[name]}", flush=True)
        source = {
            "mlgym": {"repository": REPOSITORY, "commit": COMMIT, "task": TASK, "dataset": DATASET},
            "urls": {name: f"{BASE_URL}/{name}" for name in FILES},
            "pinned_sha256": PINS,
            "rows": {name: SHAPES[name][0] for name in FILES},
            "kaggle_competition": KAGGLE,
            "license": "data: no explicit licence stated (Kaggle competition terms); MLGym's repository "
                       "LICENSE is CC-BY-NC-4.0 and does not relicense the data; never redistributed",
        }
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
