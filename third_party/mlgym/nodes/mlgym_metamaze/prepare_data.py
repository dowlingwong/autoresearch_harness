"""Frozen one-off setup of node mlgym_metamaze (the node's setup_command; never part of a run).

Downloads the MLGym task's asset bundle — the task config ``configs/tasks/rlMetaMaze.yaml`` and the task's
evaluation, requirements and ``src/`` files — from the pinned MLGym commit into --out, checks every file's
sha256 and size against the values pinned here (the bytes at commit
``9d40c1b5035202018cd7091fb4e83a9c68b377c0``), records the package pins in the hashed data manifest and links
--out into the node as ``data``.  The node has no task data: the environment is simulated by ``gymnax``
(``MetaMaze-misc``), so what is downloaded is the pinned upstream **package/task snapshot** the port was
adapted from; the setup makes that snapshot reproducible and the parent/worker data check binds every run to it.

    python3 prepare_data.py --out "$AR_TRACKG_DATA_ROOT/mlgym_metamaze"

The MLGym material is CC BY-NC 4.0 (see LICENCES.md); it is downloaded, never redistributed.  Re-running with
a complete --out only re-checks and re-links it.  --out must lie under ``$AR_TRACKG_DATA_ROOT``.  A first run
needs an empty or new --out (a partial earlier attempt is never reused); the manifest is written last and the
directory is then made read-only, so accidental writes by a run fail (the owner can restore the bits; see
ar_data).
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import urllib.request
from pathlib import Path

import ar_data

NODE = "mlgym_metamaze"
REPOSITORY = "https://github.com/facebookresearch/MLGym"
COMMIT = "9d40c1b5035202018cd7091fb4e83a9c68b377c0"
TASK = "rlMetaMazeMisc"
URL_BASE = f"https://raw.githubusercontent.com/facebookresearch/MLGym/{COMMIT}/"

# (upstream path at the pinned commit, flat file name written to --out, sha256, bytes)
ASSETS = (
    ("configs/tasks/rlMetaMaze.yaml", "rlMetaMaze.yaml",
     "45104e08cfcf81ad1de7567a9c2b0eaab2580adba4600a8e8603337da5884d82", 4_249),
    ("data/rlMetaMazeMisc/evaluate.py", "evaluate.py",
     "00ea0abc4223fb605b645ab7e73acec0634b18bc6198599bd0579f8f5ed07315", 4_111),
    ("data/rlMetaMazeMisc/requirements.txt", "requirements.txt",
     "6b90179a69d2bc321b83c8fa2c5539066b3affa648f6b81902062d945271f731", 70),
    ("data/rlMetaMazeMisc/src/__init__.py", "__init__.py",
     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
    ("data/rlMetaMazeMisc/src/config.yaml", "config.yaml",
     "f91f5586ff587cb72afb2c5add2e37c6c98e487ef2c82d20016b9cb545e420eb", 860),
    ("data/rlMetaMazeMisc/src/helpers.py", "helpers.py",
     "b9e0e07b8020323e892d08ba5ad231d95a2b8c3fcbcf35186bc0a5d8c1099859", 2_186),
    ("data/rlMetaMazeMisc/src/networks.py", "networks.py",
     "bc527dbaf530be2c573fd9b69ccbf3cbf9fd9a7b853fb791ef69cc73a4edd424", 3_238),
    ("data/rlMetaMazeMisc/src/policy.py", "policy.py",
     "755d8ec938c2baaa8163306de8ad84263091bcf30b4e9c99badc5cfe682f2ee6", 15_369),
    ("data/rlMetaMazeMisc/src/train.py", "train.py",
     "22f37c40c509a7583916c4fed9af01b414105793fb8f5a41d52f1ba3333a8bbf", 2_118),
)
FILES = ("__init__.py", "config.yaml", "evaluate.py", "helpers.py", "networks.py", "policy.py", "requirements.txt",
         "rlMetaMaze.yaml", "train.py")
assert sorted(FILES) == sorted(name for _upstream, name, _sha, _size in ASSETS)
# The package pins recorded in the manifest: requirements.txt is the port's local CPU check (the same pins as
# the other JAX node), requirements_gpu.txt the CUDA build the GPU host installs.  The resolved versions and the
# two requirement files' sha256 make the "package" half of this download-only node's provenance explicit.
DISTRIBUTIONS = ("jax", "jaxlib", "flax", "optax", "chex", "gymnax", "tensorflow-probability", "dotmap",
                 "PyYAML", "tqdm", "numpy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """One GET to *path* (the caller checks the hash); a short read is an error, not a partial file kept."""
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)


def package_record() -> dict:
    resolved = {}
    for dist in DISTRIBUTIONS:
        try:
            resolved[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            resolved[dist] = None
    return resolved


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
        out.mkdir(parents=True, exist_ok=True)
        for upstream, name, sha, size in ASSETS:
            path = out / name
            download(URL_BASE + upstream, path)
            got_sha, got_size = sha256_file(path), path.stat().st_size
            if got_sha != sha or got_size != size:
                raise SystemExit(f"{upstream} does not match the pinned MLGym asset (sha256 {got_sha}, "
                                 f"{got_size} bytes; expected {sha}, {size}); move {out} aside and run again")
        source = {
            "upstream": "MLGym", "repository": REPOSITORY, "commit": COMMIT, "task": TASK,
            "url_base": URL_BASE,
            "assets": [{"upstream": up, "file": name, "sha256": sha, "bytes": size}
                       for up, name, sha, size in ASSETS],
            "contents": "download-only MLGym task/package snapshot; no task data (gymnax MetaMaze-misc)",
            "license": "CC-BY-NC-4.0 (MLGym; its PPO code is adapted from gymnax-blines/gymnax, Apache-2.0)",
            "requirements_sha256": sha256_file(node_dir / "requirements.txt"),
            "requirements_gpu_sha256": sha256_file(node_dir / "requirements_gpu.txt"),
            "packages": package_record(),
        }
        digest = ar_data.write_manifest(out, NODE, FILES, source)
        print(f"wrote {out / ar_data.MANIFEST_NAME} sha256 {digest}")
        ar_data.make_read_only(out)  # accidental writes by a run fail; the owner can restore the bits
    manifest = ar_data.verify(out, NODE, FILES)
    ar_data.link_data(node_dir, out)
    print(json.dumps({"node": NODE, "data": str(out), "files": sorted(manifest["files"]),
                      "packages": manifest["source"].get("packages")}))


if __name__ == "__main__":
    main()
