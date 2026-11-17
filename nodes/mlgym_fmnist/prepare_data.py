"""Frozen one-off setup of node mlgym_fmnist (the node's setup_command; never part of a run).

Fetches Fashion-MNIST from the Hugging Face dataset ``zalando-datasets/fashion_mnist`` (the
dataset MLGym's imageClassificationFMnist task loads) at the hub revision
resolved now and recorded in the manifest, stores images (N, 28, 28) and
labels as uint8/int64 .npy arrays in --out, writes the hashed data manifest
and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_fmnist"

Re-running with a complete --out only re-checks and re-links it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ar_data

NODE = "mlgym_fmnist"
DATASET = "zalando-datasets/fashion_mnist"
IMAGE_COLUMN = "image"
FILES = ["train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    node_dir = Path(__file__).resolve().parent
    if not (out / ar_data.MANIFEST_NAME).exists():
        import datasets
        import huggingface_hub
        out.mkdir(parents=True, exist_ok=True)
        revision = huggingface_hub.HfApi().dataset_info(DATASET).sha
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
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
