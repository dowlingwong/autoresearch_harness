#!/usr/bin/env python3
"""Check and hash a Track G node's 3-5 representative edits before any measurement.

Usage::

    python3 scripts/trackg_declare_edits.py --spec configs/trackg/nodes/<node>.yaml \
        --node-dir nodes/<node> --out nodes/<node>/representative_edits/manifest.json \
        nodes/<node>/representative_edits/E1.diff ... E5.diff

Each diff must apply through the scope validator, must not be inert and must
not duplicate the baseline or another declared edit.  The manifest is written
once (never overwritten); commit the diffs and the manifest, then measure.
Nothing is run here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoresearch.trackg.declared_edits import DeclaredEditsError, declare_edits, write_manifest  # noqa: E402
from autoresearch.trackg.node_template import load_trackg_node_spec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--node-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("edits", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = declare_edits(load_trackg_node_spec(args.spec), args.node_dir, args.edits)
    except DeclaredEditsError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    digest = write_manifest(manifest, args.out)
    print(f"manifest {args.out} sha256 {digest}")
    for edit in manifest["edits"]:
        print(f"{edit['id']} {edit['file']} sha256 {edit['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
