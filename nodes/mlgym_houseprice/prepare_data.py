"""Frozen one-off setup of node mlgym_houseprice (the node's setup_command; never part of a run).

Fetches MLGym's copy of the Kaggle "House Prices" training table
(``data/regressionKaggleHousePrice/_helpers/train.csv``, 1,460 rows) from the
MLGym repository at the commit the port read, checks it against the sha256 and
md5 the port recorded, copies it to --out, writes the hashed data manifest and
links --out into the node as ``data``.  The split is made at run time by the
frozen wrapper (MLGym's own procedure), not here.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_houseprice"

Instead of the download, ``--from-file <path>`` takes the same file from a local
MLGym checkout (it must have the same hashes).  Re-running with a complete --out
only re-checks and re-links it.
--out must lie under ``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or
new --out (a partial earlier attempt is never reused); the manifest is written
last and the directory is then made read-only, so accidental writes by a run fail
(the owner can restore the write bits; see ar_data).
"""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import ar_data

NODE = "mlgym_houseprice"
MLGYM_COMMIT = "9d40c1b5035202018cd7091fb4e83a9c68b377c0"
URL = (f"https://raw.githubusercontent.com/facebookresearch/MLGym/{MLGYM_COMMIT}/"
       "data/regressionKaggleHousePrice/_helpers/train.csv")
SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"
MD5 = "80ccab65fb115cbad143dbbd2bcd5577"
FILES = ["houseprice.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--from-file", type=Path, default=None)
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
        if args.from_file is not None:
            data = args.from_file.read_bytes()
        else:
            with urllib.request.urlopen(URL) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256 or hashlib.md5(data).hexdigest() != MD5:
            raise SystemExit("the fetched train.csv does not have the recorded sha256/md5; nothing written")
        out.mkdir(parents=True, exist_ok=True)
        (out / "houseprice.csv").write_bytes(data)
        source = {"repository": "https://github.com/facebookresearch/MLGym", "commit": MLGYM_COMMIT,
                  "path": "data/regressionKaggleHousePrice/_helpers/train.csv", "sha256": SHA256, "md5": MD5,
                  "rows": 1460}
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    if manifest["files"]["houseprice.csv"]["sha256"] != SHA256:
        raise SystemExit("data/houseprice.csv is not the recorded MLGym file")
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
