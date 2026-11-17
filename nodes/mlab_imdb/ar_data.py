"""Frozen data helpers for Track G nodes (trackg_node/v1): a hashed data manifest.

FROZEN: listed in the node's frozen_paths.  The same file is copied
byte-identically into every node of the port (the port tests check this).

``prepare_data.py`` (the node's one-off setup step) writes the task data as
files into a directory outside the node, records every file's sha256 and the
data source in ``data_manifest.json``, and links that directory into the node
as ``data``.  The wrapper calls :func:`verify` at the start of every run, so a
run on changed, incomplete or added data stops before training: every listed
file must have its recorded size and sha256, and no other file may exist in the
directory except under the setup-only subdirectories the manifest names (raw
downloads that no run reads).

Known limits (the editable code runs in the wrapper's process): the setup makes
the directory read-only, which makes accidental writes by a run fail, but code
running as the directory's owner can restore the write bits (and as root the bits
do not apply), and it could rewrite the manifest together with the data.  After
the census records the manifest's sha256, pin it and give the data another owner
or a read-only mount on the census host.

The wrapper's own result lines go through :func:`emit`, which writes to a copy of
the standard output descriptor taken when this module is imported (before the
editable code is), and the wrapper ends with :func:`finish` (``os._exit(0)``), so
editable code cannot rewrite those lines by replacing ``sys.stdout`` or add lines
after them from an ``atexit`` handler.  It can still print a forged line and exit
before the wrapper prints its own.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_OUT_FD = os.dup(1)  # the wrapper's result lines; taken at import, before the editable code runs
MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA = "trackg_node_data/v1"
EXIT_DATA = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(line: str) -> None:
    """Print one of the wrapper's result lines (AR_METRIC, AR_TEST, AR_STEPS, AR_TOKENS) on the saved descriptor."""
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 - the editable code may have replaced or closed sys.stdout
        pass
    os.write(_OUT_FD, (line.rstrip("\n") + "\n").encode("utf-8"))


def finish() -> None:
    """End the run after the wrapper's last line: no atexit handler or later output of the editable code runs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(0)


def write_manifest(data_dir: Path, node: str, files: list[str], source: dict,
                   setup_only: tuple[str, ...] = ()) -> str:
    """Hash *files* (names relative to *data_dir*) and write the manifest once.  *setup_only* names
    subdirectories that hold setup's raw downloads (never read by a run); every other file in
    *data_dir* must be listed in *files*."""
    data_dir = Path(data_dir)
    entries = {name: {"sha256": sha256_file(data_dir / name), "bytes": (data_dir / name).stat().st_size}
               for name in sorted(files)}
    payload = {"schema": MANIFEST_SCHEMA, "node": node, "source": source, "files": entries}
    if setup_only:
        payload["setup_only"] = sorted(str(d).strip("/") for d in setup_only)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with (data_dir / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_read_only(data_dir: Path) -> None:
    """Remove write permission from every file and directory under *data_dir* (after the manifest is
    written), so accidental writes by a run fail.  Code running as the owner can restore the bits."""
    data_dir = Path(data_dir)
    for path in sorted(data_dir.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode & ~0o222)
    data_dir.chmod(data_dir.stat().st_mode & ~0o222)


def data_dir_of(script: str) -> Path:
    """The node's ``data`` link, next to the calling frozen script."""
    return Path(script).resolve().parent / "data"


def unlisted_files(data_dir: Path, manifest: dict) -> list[str]:
    """Files under *data_dir* that are neither listed, the manifest itself, nor under a setup-only subdirectory."""
    listed = set(manifest.get("files", {})) | {MANIFEST_NAME}
    setup_only = tuple(f"{d}/" for d in manifest.get("setup_only", []))
    out = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(data_dir).as_posix()
        if rel not in listed and not rel.startswith(setup_only):
            out.append(rel)
    return out


def verify(data_dir: Path, node: str, names: list[str]) -> dict:
    """Check that the manifest exists, belongs to *node*, lists *names*, that every listed file
    has the recorded size and sha256, and that no unlisted file exists outside the setup-only
    subdirectories.  Exit with code 5 otherwise."""
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
    extra = unlisted_files(data_dir, manifest)
    if extra:
        print(f"AR_ERROR data directory holds files the manifest does not list: {extra[:5]}", flush=True)
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
