"""Frozen one-off setup of node mlab_imdb (the node's setup_command; never part of a run).

Fetches the IMDb reviews (Hugging Face dataset ``stanfordnlp/imdb``, the
dataset MLAgentBench's imdb task loads as ``imdb``; its revision must be the one
the U1 audit observed, prefix ``e628166``, unless ``--accept-dataset-revision``
names another the author has accepted) and the pretrained
``distilbert-base-uncased`` model and tokenizer named by the task, both at the
hub revision resolved now and recorded in the manifest.  The reviews are
tokenized once (lower-cased WordPiece, truncated and padded to 512 tokens) and
stored as arrays; the model's original hub files (weights without a classification
head, config, tokenizer) are downloaded unchanged into ``pretrained/``.
Everything goes into --out, is hashed into the data manifest, and --out is
linked into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlab_imdb"

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

NODE = "mlab_imdb"
DATASET = "stanfordnlp/imdb"
MODEL = "distilbert-base-uncased"
MAX_TOKENS = 512
U1_DATASET_REVISION_PREFIX = "e628166"  # dataset card revision observed by the U1 audit (17 Nov 2026)
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
ARRAYS = ["train_ids.npy", "train_mask.npy", "train_y.npy", "test_ids.npy", "test_mask.npy", "test_y.npy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--accept-dataset-revision", default=None,
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
        import transformers
        out.mkdir(parents=True, exist_ok=True)
        api = huggingface_hub.HfApi()
        data_rev = api.dataset_info(DATASET).sha
        expected = args.accept_dataset_revision or U1_DATASET_REVISION_PREFIX
        if not data_rev.startswith(expected):
            raise SystemExit(f"{DATASET} is at revision {data_rev}, not the reviewed {expected}; nothing downloaded "
                             "(the author may accept it with --accept-dataset-revision)")
        model_rev = api.model_info(MODEL).sha
        ds = datasets.load_dataset(DATASET, revision=data_rev)
        pretrained = out / "pretrained"
        huggingface_hub.snapshot_download(MODEL, revision=model_rev, local_dir=pretrained,
                                          allow_patterns=MODEL_FILES)
        tok = transformers.AutoTokenizer.from_pretrained(pretrained)
        for split in ("train", "test"):
            enc = tok(ds[split]["text"], truncation=True, max_length=MAX_TOKENS, padding="max_length")
            np.save(out / f"{split}_ids.npy", np.asarray(enc["input_ids"], dtype=np.int32))
            np.save(out / f"{split}_mask.npy", np.asarray(enc["attention_mask"], dtype=np.uint8))
            np.save(out / f"{split}_y.npy", np.asarray(ds[split]["label"], dtype=np.int64))
        files = ARRAYS + [f"pretrained/{name}" for name in MODEL_FILES]
        source = {"dataset": DATASET, "dataset_revision": data_rev, "model": MODEL, "model_revision": model_rev,
                  "max_tokens": MAX_TOKENS, "datasets": datasets.__version__,
                  "transformers": transformers.__version__, "train_items": len(ds["train"]),
                  "test_items": len(ds["test"])}
        digest = ar_data.write_manifest(out, NODE, files, source, setup_only=("pretrained/.cache",))  # hub download metadata
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, ARRAYS + [f"pretrained/{name}" for name in MODEL_FILES])
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
