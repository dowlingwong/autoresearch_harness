#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.nodes.registry import load_registered_node


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a registered Stage 2 node spec.")
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    spec = load_registered_node(args.node, repo_root=ROOT)
    print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

