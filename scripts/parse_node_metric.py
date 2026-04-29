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

from autoresearch.nodes.resnet_trigger.metric_parser import parse_val_auc_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a node run log into Stage 2 metrics.")
    parser.add_argument("--node", required=True, choices=("resnet_trigger",))
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    if args.node == "resnet_trigger":
        print(json.dumps(parse_val_auc_dict(args.log), indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported node: {args.node}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

