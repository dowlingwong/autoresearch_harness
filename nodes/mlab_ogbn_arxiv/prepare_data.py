"""Frozen one-off setup of node mlab_ogbn_arxiv (the node's setup_command; never part of a run).

Fetches ogbn-arxiv with the ``ogb`` package (``NodePropPredDataset``, the
library-agnostic loader of the same dataset MLAgentBench's task loads through
``PygNodePropPredDataset``), and stores node features, labels, the directed
edge list and the official train/valid/test split as .npy files in --out,
writes the hashed data manifest and links --out into the node as ``data``.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlab_ogbn_arxiv"

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

NODE = "mlab_ogbn_arxiv"
FILES = ["x.npy", "y.npy", "edge_index.npy", "train_idx.npy", "valid_idx.npy", "test_idx.npy"]


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
        import ogb
        from ogb.nodeproppred import NodePropPredDataset
        out.mkdir(parents=True, exist_ok=True)
        dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(out / "ogb_raw"))
        graph, labels = dataset[0]
        split = dataset.get_idx_split()
        np.save(out / "x.npy", np.asarray(graph["node_feat"], dtype=np.float32))
        np.save(out / "y.npy", np.asarray(labels, dtype=np.int64).reshape(-1))
        np.save(out / "edge_index.npy", np.asarray(graph["edge_index"], dtype=np.int64))
        for name, key in (("train_idx", "train"), ("valid_idx", "valid"), ("test_idx", "test")):
            np.save(out / f"{name}.npy", np.asarray(split[key], dtype=np.int64))
        source = {"loader": "ogb.nodeproppred.NodePropPredDataset", "name": "ogbn-arxiv", "ogb": ogb.__version__,
                  "num_nodes": int(graph["num_nodes"]), "num_edges": int(graph["edge_index"].shape[1]),
                  "split_sizes": {k: int(len(v)) for k, v in split.items()}}
        digest = ar_data.write_manifest(out, NODE, FILES, source, setup_only=("ogb_raw",))
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"])}))


if __name__ == "__main__":
    main()
