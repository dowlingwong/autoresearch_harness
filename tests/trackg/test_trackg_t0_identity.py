"""M10 step 1: the T0 path (manager_proposal/v1) stays byte-identical.

The same probe script (``scripts/trackg_t0_identity_probe.py``) runs one small
``manager_proposal/v1`` campaign on the CPU node ``lr_synthetic`` against a
local fake llama-server, twice:

- with the harness ``src`` of the **base commit** (default ``a3fedbb``, the B1
  pin; override with ``TRACKG_T0_BASE_COMMIT``), exported with ``git archive``
  into a temporary directory (no worktree, no change to the repository).  If
  the commit cannot be exported the test fails, unless
  ``TRACKG_T0_ALLOW_SKIP=1`` asks for an explicit skip;
- with the harness ``src`` of the current checkout.

Both runs use the same working directory path (the first run's directory is
renamed before the second starts), so absolute paths in the records are
identical by construction.  The comparison is byte for byte:

- ``out/requests/<n>.json``: every request body the manager sent to the
  server, **unnormalised**;
- every other file of the run directory -- the worker artifacts
  (``artifacts/trial-*/run.log``, ``patch.diff``, …), the node copy (including
  ``node/.autoresearch_baseline/**`` and the final ``train.py``) and
  ``out/final_train.py`` -- **unnormalised** (``__pycache__`` excluded);
- ``out/trials.jsonl`` and ``out/events.jsonl``: byte for byte after replacing only the
  values of the documented run-to-run nondeterministic fields, which differ
  between two runs of the *same* commit: ``timestamp``, ``timestamp_start``,
  ``timestamp_end``, ``event_id`` (random), ``latency_ms``,
  ``wall_clock_seconds`` and ``elapsed_seconds`` (timings).
  The substitution is textual on each line, so key order, spacing and every
  other byte are still compared.

The node files and the node config come from each tree itself.

A second, structural test covers the T0 configurations the probe does not
exercise (keep rules, D27 skips, observers, ``seed_key``, transport retry):
no file that exists at the base commit differs at HEAD, and no module outside
``src/autoresearch/trackg/`` imports ``autoresearch.trackg``.

After a rebase onto a newer harness base, set ``TRACKG_T0_BASE_COMMIT`` to the
new merge base, or legitimate upstream T0 changes will fail these tests.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts" / "trackg_t0_identity_probe.py"
BASE_COMMIT = os.environ.get("TRACKG_T0_BASE_COMMIT", "a3fedbb")
NONDETERMINISTIC_STRINGS = ("timestamp", "timestamp_start", "timestamp_end", "event_id")
NONDETERMINISTIC_NUMBERS = ("latency_ms", "wall_clock_seconds", "elapsed_seconds")
_STR_RE = re.compile(rb'"(' + b"|".join(k.encode() for k in NONDETERMINISTIC_STRINGS) + rb')": "[^"]*"')
_NUM_RE = re.compile(rb'"(' + b"|".join(k.encode() for k in NONDETERMINISTIC_NUMBERS) + rb')": -?[0-9.eE+-]+')


def normalise(raw: bytes) -> bytes:
    return _NUM_RE.sub(rb'"\1": <N>', _STR_RE.sub(rb'"\1": "<S>"', raw))


def export_base_tree(target: Path) -> Path:
    try:
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", BASE_COMMIT,
             "src", "nodes/lr_synthetic", "configs/nodes/lr_synthetic.yaml"],
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        message = f"cannot export base commit {BASE_COMMIT}: {error}"
        if os.environ.get("TRACKG_T0_ALLOW_SKIP") == "1":
            pytest.skip(message)
        pytest.fail(message + " (set TRACKG_T0_ALLOW_SKIP=1 to skip explicitly)")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        try:
            tar.extractall(target, filter="data")
        except TypeError:  # pragma: no cover - Python without extraction filters
            tar.extractall(target)
    return target


def run_probe(tree: Path, work: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMACPP_", "PYTHONPATH"))}
    subprocess.run(
        [sys.executable, "-I", str(PROBE), "--harness-src", str(tree / "src"),
         "--node-config", str(tree / "configs" / "nodes" / "lr_synthetic.yaml"),
         "--node-dir", str(tree / "nodes" / "lr_synthetic"), "--work", str(work)],
        check=True, env=env, capture_output=True, timeout=300)


def files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


NORMALISED_FILES = ("out/trials.jsonl", "out/events.jsonl")


def test_manager_proposal_v1_campaign_is_byte_identical_to_the_base_commit(tmp_path):
    base_tree = export_base_tree(tmp_path / "base_tree")
    work = tmp_path / "run"
    run_probe(base_tree, work)
    work.rename(tmp_path / "base_run")
    run_probe(ROOT, work)
    work.rename(tmp_path / "head_run")
    base, head = files(tmp_path / "base_run"), files(tmp_path / "head_run")

    assert sorted(base) == sorted(head)
    requests = [k for k in base if k.startswith("out/requests/")]
    assert len(requests) == 5, "every proposal goes through the fake llama-server"
    artifacts = [k for k in base if k.startswith("artifacts/")]
    assert any(k.endswith("run.log") for k in artifacts) and any(k.endswith("patch.diff") for k in artifacts)
    for name in sorted(set(base) - set(NORMALISED_FILES)):
        assert base[name] == head[name], f"{name} differs (unnormalised comparison)"
    for name in NORMALISED_FILES:
        assert base[name].count(b"\n") == head[name].count(b"\n") > 0
        assert normalise(base[name]) == normalise(head[name]), f"{name} differs beyond the documented fields"
    # the fixture exercises the v1 schema on the wire and a malformed reply
    assert b'"manager_proposal_v1"' in base[requests[0]]
    assert b'"parse_failed": true' in base["out/trials.jsonl"] or b'"parse_failed":true' in base["out/trials.jsonl"]


def test_no_base_file_differs_and_no_t0_module_imports_trackg():
    try:
        changed = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--diff-filter=MDRT", "--name-only", BASE_COMMIT, "HEAD"],
            check=True, capture_output=True, text=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        if os.environ.get("TRACKG_T0_ALLOW_SKIP") == "1":
            pytest.skip(str(error))
        pytest.fail(f"cannot diff against {BASE_COMMIT}: {error}")
    assert changed == [], f"files of the base commit changed: {changed}"
    import ast

    offenders = []
    for path in sorted((ROOT / "src" / "autoresearch").rglob("*.py")):
        if "trackg" in path.relative_to(ROOT / "src" / "autoresearch").parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            if any(n == "autoresearch.trackg" or n.startswith("autoresearch.trackg.") for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_normalisation_touches_only_the_documented_fields():
    line = (b'{"event_id": "abc", "latency_ms": 1.5e-3, "payload": {"value": 0.9, "elapsed_seconds": 2}, '
            b'"timestamp": "2026-11-09T01:00:00Z", "wall_clock_seconds": 1.0, "seed": 7}\n')
    assert normalise(line) == (b'{"event_id": "<S>", "latency_ms": <N>, "payload": {"value": 0.9, '
                               b'"elapsed_seconds": <N>}, "timestamp": "<S>", "wall_clock_seconds": <N>, '
                               b'"seed": 7}\n')
