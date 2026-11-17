"""Frozen one-off setup of node mlgym_fmnist (the node's setup_command; never part of a run).

Fetches Fashion-MNIST from the Hugging Face dataset ``zalando-datasets/fashion_mnist`` (the
dataset MLGym's imageClassificationFMnist task loads) at the hub revision
resolved now (it must start with U1's ``531be5e`` unless ``--accept-revision``
names another the author has accepted) and recorded in the manifest, stores images (N, 28, 28) and
labels as uint8/int64 .npy arrays in --out, writes the hashed data manifest
and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_fmnist"

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

NODE = "mlgym_fmnist"
DATASET = "zalando-datasets/fashion_mnist"
IMAGE_COLUMN = "image"
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]
U1_REVISION_PREFIX = "531be5e"  # dataset card revision observed by the U1 audit (17 Nov 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--accept-revision", default=None,
                        help="a dataset revision other than U1's, only after the author has accepted it")
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
        import datasets
        import huggingface_hub
        out.mkdir(parents=True, exist_ok=True)
        revision = huggingface_hub.HfApi().dataset_info(DATASET).sha
        expected = args.accept_revision or U1_REVISION_PREFIX
        if not revision.startswith(expected):
            raise SystemExit(f"{DATASET} is at revision {revision}, not the reviewed {expected}; nothing "
                             "downloaded (the author may accept it with --accept-revision)")
        ds = datasets.load_dataset(DATASET, revision=revision)
        shapes = {}
        for split in ("train", "test"):
            images = np.stack([np.asarray(img, dtype=np.uint8) for img in ds[split][IMAGE_COLUMN]])
            np.save(out / f"{split}_x.npy", images)
            np.save(out / f"{split}_y.npy", np.asarray(ds[split]["label"], dtype=np.int64))
            shapes[f"{split}_x"] = list(images.shape)
        source = {"dataset": DATASET, "revision": revision, "image_column": IMAGE_COLUMN,
                  "datasets": datasets.__version__, "shapes": shapes}
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
