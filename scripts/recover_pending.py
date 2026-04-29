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

from autoresearch.control_plane.campaign import (
    clear_pending_guard,
    inspect_pending_guard,
    list_pending_guards,
    mark_pending_failed,
)
from autoresearch.nodes.registry import load_registered_node


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or recover Stage 2 pending-trial guards.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List pending guards below a directory.")
    list_parser.add_argument("--root", default=str(ROOT / "experiments" / "ledgers"))

    inspect_parser = subparsers.add_parser("inspect", help="Print one pending guard as JSON.")
    inspect_parser.add_argument("guard")

    fail_parser = subparsers.add_parser(
        "fail",
        help="Append a failed-invalid TrialRecord for the guard, then remove the guard.",
    )
    fail_parser.add_argument("guard")
    fail_parser.add_argument("--node", default="resnet_trigger")
    fail_parser.add_argument("--manager-mode", default="unknown_manager")
    fail_parser.add_argument("--worker-mode", default="unknown_worker")
    fail_parser.add_argument("--memory-mode", default="unknown")
    fail_parser.add_argument("--message", default="Recovered stale pending trial.")

    clear_parser = subparsers.add_parser("clear", help="Remove a pending guard without appending a record.")
    clear_parser.add_argument("guard")

    args = parser.parse_args()

    if args.command == "list":
        guards = [str(path) for path in list_pending_guards(args.root)]
        print(json.dumps({"pending_guards": guards}, indent=2, sort_keys=True))
        return 0

    if args.command == "inspect":
        print(json.dumps(inspect_pending_guard(args.guard), indent=2, sort_keys=True))
        return 0

    if args.command == "fail":
        node_spec = load_registered_node(args.node, repo_root=ROOT)
        record = mark_pending_failed(
            guard_path=args.guard,
            node_spec=node_spec,
            manager_mode=args.manager_mode,
            worker_mode=args.worker_mode,
            memory_mode=args.memory_mode,
            failure_message=args.message,
        )
        print(json.dumps({"record": record.to_dict()}, indent=2, sort_keys=True))
        return 0

    if args.command == "clear":
        path = clear_pending_guard(args.guard)
        print(json.dumps({"cleared": str(path)}, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
