"""Frozen data helpers for Track G nodes (trackg_node/v1): a hashed data manifest.

FROZEN: listed in the node's frozen_paths.  The same file is copied
byte-identically into every node of the port (the port tests check this).

``prepare_data.py`` (the node's one-off setup step) writes the task data as
files into a directory outside the node, records every file's sha256 and the
data source in ``data_manifest.json``, and links that directory into the node
as ``data``.  The wrapper calls :func:`load_verified` at the start of every run,
so a run on changed or incomplete data stops before training.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA = "trackg_node_data/v1"
EXIT_DATA = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(data_dir: Path, node: str, files: list[str], source: dict) -> str:
    """Hash *files* (names relative to *data_dir*) and write the manifest once."""
    data_dir = Path(data_dir)
    entries = {name: {"sha256": sha256_file(data_dir / name), "bytes": (data_dir / name).stat().st_size}
               for name in sorted(files)}
    payload = {"schema": MANIFEST_SCHEMA, "node": node, "source": source, "files": entries}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with (data_dir / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def data_dir_of(script: str) -> Path:
    """The node's ``data`` link, next to the calling frozen script."""
    return Path(script).resolve().parent / "data"


def verify(data_dir: Path, node: str, names: list[str]) -> dict:
    """Check that the manifest exists, belongs to *node*, lists *names*, and that every
    listed file has the recorded size and sha256.  Exit with code 5 otherwise."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"AR_ERROR data manifest missing at {manifest_path}; run the node's setup_command", flush=True)
        sys.exit(EXIT_DATA)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("node") != node:
        print(f"AR_ERROR data manifest {manifest_path} is not a {MANIFEST_SCHEMA} manifest of {node}", flush=True)
        sys.exit(EXIT_DATA)
    files = manifest.get("files", {})
    for name in names:
        if name not in files:
            print(f"AR_ERROR data manifest lacks {name}", flush=True)
            sys.exit(EXIT_DATA)
    for name, entry in sorted(files.items()):
        path = data_dir / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            print(f"AR_ERROR data file {name} differs from the manifest", flush=True)
            sys.exit(EXIT_DATA)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"AR_DATA manifest_sha256={digest} files={len(files)}", flush=True)
    return manifest


def link_data(node_dir: Path, target: Path) -> None:
    """Create ``<node_dir>/data`` -> *target* (refuse to replace a different link or a directory)."""
    link = Path(node_dir) / "data"
    target = Path(target).resolve()
    if link.is_symlink():
        if Path(os.readlink(link)).resolve() == target:
            return
        raise SystemExit(f"{link} already links to {os.readlink(link)}; not replaced")
    if link.exists():
        raise SystemExit(f"{link} exists and is not a link; not replaced")
    link.symlink_to(target, target_is_directory=True)
