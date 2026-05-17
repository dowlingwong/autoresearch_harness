#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.research_context import write_node_research_context
from autoresearch.nodes.registry import load_registered_node


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic node research-context note.")
    parser.add_argument("--node", required=True, help="Registered node name.")
    parser.add_argument("--node-root", required=True, help="Node working directory.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "paper" / "notes"),
        help="Directory for <node>_research_context.md.",
    )
    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    ref = write_node_research_context(
        node_spec=node_spec,
        repo_root=ROOT,
        node_root=args.node_root,
        output_dir=args.output_dir,
    )
    print(f"wrote={ref.path}")
    print(f"sha256={ref.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
