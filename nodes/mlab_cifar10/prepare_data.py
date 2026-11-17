"""Frozen one-off setup of node mlab_cifar10 (the node's setup_command; never part of a run).

Fetches CIFAR-10 with torchvision exactly as MLAgentBench's cifar10 task does
(``torchvision.datasets.CIFAR10(download=True)``, which checks the archive's
md5; the md5 torchvision checks must be the pinned ARCHIVE_MD5), stores the four arrays as .npy files in --out, writes the hashed
data manifest and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlab_cifar10"

Re-running with a complete --out only re-checks and re-links it.
--out must lie under ``$AR_TRACKG_DATA_ROOT``.  A first run needs an empty or
new --out (a partial earlier attempt is never reused); the manifest is written
last and the directory is then made read-only, so accidental writes by a run fail
(the owner can restore the write bits; see ar_data).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import ar_data

NODE = "mlab_cifar10"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"  # torchvision's published md5 of cifar-10-python.tar.gz
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]


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
        import torchvision
        if torchvision.datasets.CIFAR10.tgz_md5 != ARCHIVE_MD5:
            raise SystemExit("this torchvision checks a different CIFAR-10 archive md5; nothing downloaded")
        out.mkdir(parents=True, exist_ok=True)
        raw = out / "torchvision_raw"
        train = torchvision.datasets.CIFAR10(root=str(raw), train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=str(raw), train=False, download=True)
        np.save(out / "train_x.npy", np.asarray(train.data, dtype=np.uint8))
        np.save(out / "train_y.npy", np.asarray(train.targets, dtype=np.int64))
        np.save(out / "test_x.npy", np.asarray(test.data, dtype=np.uint8))
        np.save(out / "test_y.npy", np.asarray(test.targets, dtype=np.int64))
        source = {"loader": "torchvision.datasets.CIFAR10", "torchvision": torchvision.__version__,
                  "url": torchvision.datasets.CIFAR10.url, "archive_md5": torchvision.datasets.CIFAR10.tgz_md5,
                  "shapes": {"train_x": [50000, 32, 32, 3], "test_x": [10000, 32, 32, 3]}}
        digest = ar_data.write_manifest(out, NODE, FILES, source, setup_only=("torchvision_raw",))
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
