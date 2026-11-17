"""Frozen one-off setup of the Track G node autoresearch_fixedtokens (its setup_command; never part of a run).

Runs autoresearch's own preparation (prepare.py: download_data and
train_tokenizer) into --out, which becomes autoresearch's cache directory:
``--out/data/shard_*.parquet`` (the first 10 training shards of
karpathy/climbmix-400b-shuffle, autoresearch's default, plus the pinned
validation shard 06542) and ``--out/tokenizer/`` (the BPE tokenizer trained by
prepare.py on those shards).  Then it hashes every file into the data manifest
and links --out into the node as ``data``.  The dataset revision must be the
one the U1 audit observed (prefix ``915333b``) unless ``--accept-revision``
names another the author has accepted; the shards are downloaded from that
exact revision (``prepare.BASE_URL`` is pointed at it and the download workers
are forked so they inherit it; prepare.py itself is unchanged).  Data are never redistributed: they stay outside the node.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/autoresearch_fixedtokens"

The shard count is fixed here (10) because the tokenizer is trained on the
downloaded training shards.  Before the tokenizer is trained, ``--out/data`` must
hold exactly ``shard_00000``–``shard_00009`` and ``shard_06542`` (prepare.py's
download only prints failures, so a partial download would otherwise train a
different tokenizer); any missing or extra parquet file stops the setup without
writing a manifest.  An ``--out`` that already holds parquet files from
elsewhere is refused.  The manifest is written last, after prepare.py's
tokenizer sanity check, and the data directory is then made read-only.  ``--out`` must lie under
``$AR_TRACKG_DATA_ROOT``.
Re-running with a complete --out only re-checks and re-links it.
"""
import argparse
import json
import os
from pathlib import Path

import ar_data

NODE = "autoresearch_fixedtokens"
NUM_SHARDS = 10
REQUIRED_SHARDS = sorted([f"shard_{i:05d}.parquet" for i in range(NUM_SHARDS)] + ["shard_06542.parquet"])
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", *[f"data/{n}" for n in REQUIRED_SHARDS]]
DATASET = "karpathy/climbmix-400b-shuffle"
U1_REVISION_PREFIX = "915333b"  # dataset revision observed by the U1 audit (17 Nov 2026)


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
        if (out / "data").exists() and any((out / "data").glob("*.parquet")):
            raise SystemExit(f"{out / 'data'} already holds parquet files; use a new --out (nothing is reused)")
        out.mkdir(parents=True, exist_ok=True)
        os.environ["AUTORESEARCH_CACHE_DIR"] = str(out)
        import huggingface_hub

        import prepare  # reads AUTORESEARCH_CACHE_DIR at import

        revision = huggingface_hub.HfApi().dataset_info(DATASET).sha
        expected = args.accept_revision or U1_REVISION_PREFIX
        if not revision.startswith(expected):
            raise SystemExit(f"{DATASET} is at revision {revision}, not the reviewed {expected}; "
                             "nothing downloaded (the author may accept it with --accept-revision)")
        # prepare.py downloads from .../resolve/main; pin the download to the checked revision.  Its download
        # workers are a multiprocessing Pool: under "spawn"/"forkserver" they would re-import prepare and use
        # its own BASE_URL, so the workers are forked (they inherit the pinned BASE_URL).
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
        prepare.BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}"
        prepare.download_data(NUM_SHARDS)
        present = sorted(p.name for p in (out / "data").glob("*.parquet"))
        if present != REQUIRED_SHARDS:
            missing = sorted(set(REQUIRED_SHARDS) - set(present))
            extra = sorted(set(present) - set(REQUIRED_SHARDS))
            raise SystemExit(f"download incomplete or unexpected (missing {missing}, extra {extra}); no manifest "
                             "written: move this --out aside and run again into a new one")
        prepare.train_tokenizer()
        leftovers = sorted(str(p.relative_to(out)) for p in out.rglob("*.tmp"))
        if leftovers:
            raise SystemExit(f"temporary download files remain ({leftovers[:3]}); no manifest written: move this "
                             "--out aside and run again into a new one")
        files = sorted(str(p.relative_to(out)) for sub in ("data", "tokenizer")
                       for p in (out / sub).rglob("*") if p.is_file())
        source = {"preparation": "prepare.download_data(10) + prepare.train_tokenizer()",
                  "dataset": DATASET, "revision": revision, "base_url": prepare.BASE_URL, "num_shards": NUM_SHARDS, "val_shard": prepare.VAL_FILENAME,
                  "vocab_size": prepare.VOCAB_SIZE}
        digest = ar_data.write_manifest(out, NODE, files, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, REQUIRED_FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
