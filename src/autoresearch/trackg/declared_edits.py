"""Declaring a node's 3-5 representative edits as diffs before any measurement (G.1 step 2).

:func:`declare_edits` checks each declared diff against the node in a
throw-away copy -- it must apply through the scope validator, must not be
inert, and must not duplicate the baseline or another declared edit -- and
returns a manifest with the hashes.  :func:`write_manifest` writes it once
(it never overwrites).  Nothing is run or measured here; the porter commits
the diffs and the manifest, and only then measures (TRACKG_PORT_CHECKLIST.md
on the paper side).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from autoresearch.trackg.diff_support import EvaluatedStates, apply_diff, code_state_hashes
from autoresearch.trackg.node_template import TrackGNodeSpec

DECLARED_EDITS_SCHEMA = "trackg_representative_edits/v1"
MIN_EDITS, MAX_EDITS = 3, 5


class DeclaredEditsError(ValueError):
    """A declared edit set that cannot be accepted."""


def declare_edits(spec: TrackGNodeSpec, node_dir: str | Path, edit_paths: Sequence[str | Path]) -> dict[str, Any]:
    if not MIN_EDITS <= len(edit_paths) <= MAX_EDITS:
        raise DeclaredEditsError(f"declare {MIN_EDITS}-{MAX_EDITS} representative edits, got {len(edit_paths)}")
    names = [Path(p).name for p in edit_paths]
    if len(set(names)) != len(names):
        raise DeclaredEditsError("edit file names must be unique")
    base = spec.to_base_node_spec()
    node = Path(node_dir).resolve()
    baseline = code_state_hashes(node, base)
    evaluated = EvaluatedStates()
    evaluated.register("baseline", code_state_sha256=baseline[0])
    entries = []
    with tempfile.TemporaryDirectory(prefix="trackg_declare_") as tmp:
        for index, path in enumerate(edit_paths, start=1):
            raw = Path(path).read_bytes()
            text = raw.decode("utf-8")
            work = Path(tmp) / f"e{index}"
            shutil.copytree(node, work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            result = apply_diff(work, text, base, evaluated=evaluated, write=False)
            prov = result.provenance
            if result.status != "applied":
                detail = prov.get("refusal_detail") or prov.get("inert_reason") or prov.get("duplicate_of")
                raise DeclaredEditsError(f"{Path(path).name}: {result.status} ({detail})")
            evaluated.register(Path(path).name, code_state_sha256=prov["code_state_sha256"],
                               normalised_diff_sha256=prov["normalised_diff_sha256"])
            entries.append({
                "id": f"E{index}",
                "file": Path(path).name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "normalised_diff_sha256": prov["normalised_diff_sha256"],
                "files_touched": prov["files_touched"],
                "code_state_sha256": prov["code_state_sha256"],
            })
    return {
        "schema": DECLARED_EDITS_SCHEMA,
        "node": spec.name,
        "node_spec_sha256": spec.raw_sha256,
        "run_budget": {"kind": spec.run_budget.kind, "value": spec.run_budget.value},
        "baseline_code_state_sha256": baseline[0],
        "baseline_code_state_raw_sha256": baseline[1],
        "edits": entries,
        "measured": False,
    }


def write_manifest(manifest: dict[str, Any], out: str | Path) -> str:
    """Write the manifest (refusing to overwrite) and return its sha256."""
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with Path(out).open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()
