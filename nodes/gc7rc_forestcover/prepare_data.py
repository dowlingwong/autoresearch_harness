"""Frozen one-off setup of node gc7rc_forestcover (the node's setup_command; never part of a run).

Fetches the public **UCI Covertype** dataset -- the Forest Cover Type task of the 1GC-7RC paper's Task 5
(seven-class tabular prediction over 581,012 rows x 54 features: 10 continuous + 44 binary one-hot) -- and
nothing else:

    https://archive.ics.uci.edu/static/public/31/covertype.zip
    (UCI Machine Learning Repository, dataset 31 "Covertype", DOI 10.24432/C50K5N, CC BY 4.0)

The zip is checked against the sha256 pinned here, its member ``covtype.data.gz`` against a second pinned
sha256, and the member's 581,012 rows x 55 CSV columns (54 features + the class label 1-7) are parsed into
``X.npy`` (float32, 581,012 x 54) and ``y.npy`` (int8, the class mapped to 0-6).  The hashed data manifest
is written and ``--out`` is linked into the node as ``data``.  The port does **not** redistribute the data:
this script downloads them at setup time and the branch holds no task data.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/gc7rc_forestcover"

Re-running with a complete --out only re-checks and re-links it.  --out must lie under
``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or new --out (a partial earlier attempt is never
reused); the manifest is written last and the directory is then made read-only, so accidental writes by a
run fail (the owner can restore the write bits; see ar_data).  The legacy direct URL of the same file
(``https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz``) returned the same
11,240,707 bytes with the same sha256 on 25 Nov 2026; the zip is the dataset page's canonical download.
"""
import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import ar_data

NODE = "gc7rc_forestcover"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/31/covertype"
DOI = "10.24432/C50K5N"
DATASET_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
LEGACY_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
ZIP_SHA256 = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
ZIP_BYTES = 11_251_206
MEMBER = "covtype.data.gz"
GZ_SHA256 = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
GZ_BYTES = 11_240_707
ROWS, FEATURES, CLASSES = 581_012, 54, 7
FILES = ["X.npy", "y.npy"]
DOWNLOAD_TIMEOUT = 120  # a fixed socket bound for one GET, not a training budget


def fetch(url: str) -> bytes:
    """One HTTP GET of the pinned file (a fixed 120 s socket bound, not a training budget)."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310 - the pinned https URL
        return response.read()


def check_zip(zip_bytes: bytes) -> bytes:
    """The zip is the pinned one and its covtype.data.gz member is the pinned gzip; returns the member bytes."""
    if len(zip_bytes) != ZIP_BYTES or hashlib.sha256(zip_bytes).hexdigest() != ZIP_SHA256:
        raise SystemExit(f"covertype.zip is {len(zip_bytes)} bytes sha256 "
                         f"{hashlib.sha256(zip_bytes).hexdigest()}, not the pinned {ZIP_BYTES} / {ZIP_SHA256}; "
                         "nothing written")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        if MEMBER not in names:
            raise SystemExit(f"covertype.zip lacks {MEMBER} (members: {sorted(names)}); nothing written")
        member = archive.read(MEMBER)
    if len(member) != GZ_BYTES or hashlib.sha256(member).hexdigest() != GZ_SHA256:
        raise SystemExit(f"{MEMBER} is {len(member)} bytes sha256 {hashlib.sha256(member).hexdigest()}, "
                         f"not the pinned {GZ_BYTES} / {GZ_SHA256}; nothing written")
    return member


def parse(gz_bytes: bytes) -> tuple:
    """The member CSV -> (X float32 (ROWS, FEATURES), y int8 class 0-6); the UCI task's shapes checked."""
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(gz_bytes), header=None, compression="gzip")
    if frame.shape != (ROWS, FEATURES + 1):
        raise SystemExit(f"covtype.data.gz holds a {frame.shape} table, expected {(ROWS, FEATURES + 1)}; "
                         "nothing written")
    values = frame.to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise SystemExit("covtype.data.gz holds non-finite values; nothing written")
    labels = values[:, FEATURES]
    classes = sorted(int(v) for v in set(labels))
    if classes != list(range(1, CLASSES + 1)):
        raise SystemExit(f"the label column holds {classes}, expected the UCI classes 1-{CLASSES}; nothing written")
    x = values[:, :FEATURES].astype("float32")
    y = (labels - 1).astype("int8")
    return x, y, {int(c): int((y == c - 1).sum()) for c in classes}


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
        zip_bytes = fetch(DATASET_URL)
        print(f"fetched {DATASET_URL} bytes={len(zip_bytes)}", flush=True)
        gz_bytes = check_zip(zip_bytes)
        print(f"{MEMBER} sha256 {GZ_SHA256} bytes={len(gz_bytes)}", flush=True)
        x, y, counts = parse(gz_bytes)
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw").mkdir()
        (out / "raw" / "covertype.zip").write_bytes(zip_bytes)
        np.save(out / "X.npy", x)
        np.save(out / "y.npy", y)
        for name in FILES:
            print(f"{name} sha256 {ar_data.sha256_file(out / name)} bytes={(out / name).stat().st_size}",
                  flush=True)
        source = {
            "dataset": "UCI Covertype (Forest Cover Type)",
            "dataset_page": DATASET_PAGE,
            "doi": DOI,
            "url": DATASET_URL,
            "zip_sha256": ZIP_SHA256,
            "zip_bytes": ZIP_BYTES,
            "member": MEMBER,
            "member_sha256": GZ_SHA256,
            "member_bytes": GZ_BYTES,
            "same_file_legacy_url": LEGACY_DATA_URL,
            "rows": ROWS,
            "features": FEATURES,
            "classes": CLASSES,
            "class_counts": {f"class_{c}": n for c, n in counts.items()},
            "label": f"UCI covertype class 1-{CLASSES} mapped to 0-{CLASSES - 1} in y.npy",
            "license": "CC BY 4.0 (UCI dataset page); data downloaded by setup, never redistributed",
            "suite_note": "1GC-7RC Task 5 public dataset; the paper-named suite repository returned 404 "
                          "(U1 audit), so only the public dataset is used",
        }
        digest = ar_data.write_manifest(out, NODE, FILES, source, setup_only=("raw",))
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
