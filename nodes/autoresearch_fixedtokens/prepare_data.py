"""Frozen one-off setup of the Track G node autoresearch_fixedtokens (its setup_command; never part of a run).

Runs autoresearch's own preparation (prepare.py: download_data and
train_tokenizer) into --out, which becomes autoresearch's cache directory:
``--out/data/shard_*.parquet`` (the first 10 training shards of
karpathy/climbmix-400b-shuffle, autoresearch's default, plus the pinned
validation shard 06542) and ``--out/tokenizer/`` (the BPE tokenizer trained by
prepare.py on those shards).  Then it hashes every file into the data manifest
and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/autoresearch_fixedtokens"

The shard count is fixed here (10) because the tokenizer is trained on the
downloaded training shards.  Re-running with a complete --out only re-checks
and re-links it.
"""
import argparse
import json
import os
from pathlib import Path

import ar_data

NODE = "autoresearch_fixedtokens"
NUM_SHARDS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    node_dir = Path(__file__).resolve().parent
    if not (out / ar_data.MANIFEST_NAME).exists():
        out.mkdir(parents=True, exist_ok=True)
        os.environ["AUTORESEARCH_CACHE_DIR"] = str(out)
        import prepare  # reads AUTORESEARCH_CACHE_DIR at import

        prepare.download_data(NUM_SHARDS)
        prepare.train_tokenizer()
        files = sorted(str(p.relative_to(out)) for sub in ("data", "tokenizer")
                       for p in (out / sub).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        source = {"preparation": "prepare.download_data(10) + prepare.train_tokenizer()",
                  "base_url": prepare.BASE_URL, "num_shards": NUM_SHARDS, "val_shard": prepare.VAL_FILENAME,
                  "vocab_size": prepare.VOCAB_SIZE}
        digest = ar_data.write_manifest(out, NODE, files, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
    manifest = ar_data.verify(out, NODE, ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt",
                                          "data/shard_06542.parquet"])
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
