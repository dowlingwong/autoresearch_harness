"""Frozen one-off setup of node mlab_cifar10 (the node's setup_command; never part of a run).

Fetches CIFAR-10 with torchvision exactly as MLAgentBench's cifar10 task does
(``torchvision.datasets.CIFAR10(download=True)``, which checks the archive's
md5), stores the four arrays as .npy files in --out, writes the hashed
data manifest and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlab_cifar10"

Re-running with a complete --out only re-checks and re-links it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ar_data

NODE = "mlab_cifar10"
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    node_dir = Path(__file__).resolve().parent
    if not (out / ar_data.MANIFEST_NAME).exists():
        import torchvision
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
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
