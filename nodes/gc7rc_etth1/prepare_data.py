"""Frozen one-off setup of node gc7rc_etth1 (the node's setup_command; never part of a run).

Downloads the public ETTh1 hourly series from the pinned ETDataset commit (the source the 1GC-7RC Task 6
description points at), checks the file's sha256 against the value pinned here, stores it as ``series.npy`` in
--out, writes the hashed data manifest and links --out into the node as ``data``.  The raw CSV is kept under a
setup-only subdirectory for audit; no run reads it.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/gc7rc_etth1"

The dataset is downloaded, never redistributed: the branch and the node carry this script and the hashes only.
Re-running with a complete --out only re-checks and re-links it.  --out must lie under
``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or new --out (a partial earlier attempt is never reused);
the manifest is written last and the directory is then made read-only, so accidental writes by a run fail (the
owner can restore the bits; see ar_data).
"""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

import ar_data

NODE = "gc7rc_etth1"
REPOSITORY = "https://github.com/zhouhaoyi/ETDataset"
COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"  # the ETDataset HEAD recorded by the U1 audit (17 Nov 2026)
SOURCE_URL = (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{COMMIT}/ETT-small/ETTh1.csv")
SOURCE_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
SOURCE_BYTES = 2_589_657
RAW_NAME = "ETTh1.csv"
FILES = ["series.npy"]
EXPECTED_SHAPE = (17_420, 7)  # date + 7 features on 17,420 hourly rows (1 July 2016 - 2 July 2018)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """One GET to *path* (the caller checks the hash); a short read is an error, not a partial file kept."""
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)


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
        out.mkdir(parents=True, exist_ok=True)
        raw = out / "raw"
        raw.mkdir()
        csv_path = raw / RAW_NAME
        download(SOURCE_URL, csv_path)
        if csv_path.stat().st_size != SOURCE_BYTES or sha256_file(csv_path) != SOURCE_SHA256:
            raise SystemExit(f"{csv_path} does not match the pinned ETTh1.csv "
                             f"(sha256 {sha256_file(csv_path)}, {csv_path.stat().st_size} bytes); move it aside "
                             "and run again")
        # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (the 7 task features).
        series = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=range(1, 8), dtype=np.float64)
        if series.shape != EXPECTED_SHAPE or not np.all(np.isfinite(series)):
            raise SystemExit(f"{csv_path} gave {series.shape}; expected finite {EXPECTED_SHAPE}")
        np.save(out / "series.npy", series.astype(np.float32))
        source = {"dataset": "ETTh1 (Electricity Transformer Temperature, hourly)",
                  "repository": REPOSITORY, "commit": COMMIT, "url": SOURCE_URL,
                  "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES, "csv": f"raw/{RAW_NAME}",
                  "shape": list(EXPECTED_SHAPE),
                  "license": "CC BY-ND 4.0 (the most restrictive of the three conflicting claims recorded by "
                             "the U1 audit; the 1GC-7RC paper's Appendix E also states CC BY-ND 4.0)",
                  "citation": "Zhou et al. 2021, Informer: Beyond Efficient Transformer for Long Sequence "
                              "Time-Series Forecasting (AAAI 2021)"}
        digest = ar_data.write_manifest(out, NODE, FILES, source, setup_only=("raw",))
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
