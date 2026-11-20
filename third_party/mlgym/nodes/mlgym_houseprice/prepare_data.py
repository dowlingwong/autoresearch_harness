"""Frozen one-off setup of node mlgym_houseprice (the node's setup_command; never part of a run).

R1-TRACKG-PORT-DECISIONS (2): the data are the Kaggle "House Prices - Advanced Regression Techniques" training
table (``train.csv``, 1,460 rows), which the user downloads from Kaggle (competition
house-prices-advanced-regression-techniques; Kaggle's terms apply) and passes with ``--from-file``; MLGym's
forwarded copy is not used.  Nothing is downloaded here.  The file is checked for its shape (1,460 rows, 81
columns including ``Id`` and ``SalePrice``), copied to --out, and its sha256 and md5 are recorded in the hashed
data manifest (and printed); the census records the manifest's sha256.  The split is made at run time by the
frozen parent (MLGym's own procedure), not here.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_houseprice" --from-file ~/Downloads/train.csv

Re-running with a complete --out re-checks it (its manifest must record the user's download; a --from-file
given again must be the same file) and re-links it.
--out must lie under ``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or
new --out (a partial earlier attempt is never reused); the manifest is written
last and the directory is then made read-only, so accidental writes by a run fail
(the owner can restore the write bits; see ar_data).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import ar_data

NODE = "mlgym_houseprice"
FILES = ["houseprice.csv"]
KAGGLE = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data (train.csv)"
ROWS, COLUMNS = 1460, 81
USER_DOWNLOAD = "user download"   # the manifest's source.provided_by starts with this (checked by every run too)


def check_table(data: bytes) -> None:
    """The Kaggle training table's shape: 1,460 rows, 81 columns, Id and SalePrice present."""
    import io

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(data))
    if frame.shape != (ROWS, COLUMNS) or not {"Id", "SalePrice"} <= set(frame.columns):
        raise SystemExit(f"--from-file is not Kaggle's House Prices train.csv (shape {frame.shape}, expected "
                         f"({ROWS}, {COLUMNS}) with Id and SalePrice); nothing written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--from-file", type=Path, default=None,
                        help="the Kaggle train.csv the user downloaded (required for a first setup)")
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
        if args.from_file is None:
            raise SystemExit("give --from-file <the Kaggle train.csv you downloaded> (R1-TRACKG-PORT-DECISIONS (2): "
                             "the user downloads the table from Kaggle; nothing is fetched here)")
        data = args.from_file.read_bytes()
        check_table(data)
        sha256, md5 = hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()
        out.mkdir(parents=True, exist_ok=True)
        (out / "houseprice.csv").write_bytes(data)
        source = {"kaggle": KAGGLE, "provided_by": f"{USER_DOWNLOAD}, passed with --from-file",
                  "from_file_name": args.from_file.name, "sha256": sha256, "md5": md5, "rows": ROWS}
        print(f"houseprice.csv sha256 {sha256} md5 {md5}")
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    # Kit review round 1, M1: a re-run must not accept a directory made by an earlier setup from another source
    # (for example MLGym's forwarded copy): the manifest must record the user's download, with its hashes, and a
    # --from-file given on the re-run must be the same file.
    source = manifest.get("source", {})
    digest = manifest["files"]["houseprice.csv"]["sha256"]
    if not str(source.get("provided_by", "")).startswith(USER_DOWNLOAD) or source.get("sha256") != digest \
            or not source.get("md5"):
        raise SystemExit(f"{out} was not set up from the user's Kaggle download (data manifest source: {source}); "
                         "move it aside and run again with --from-file into a new --out")
    if args.from_file is not None and hashlib.sha256(args.from_file.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"--from-file differs from the table already set up in {out} (sha256 {digest}); move the "
                         "directory aside to set up a different file")
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
