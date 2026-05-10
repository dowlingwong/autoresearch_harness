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
    inspect_pending_guard,
    list_pending_guards,
    mark_pending_failed,
)
from autoresearch.nodes.registry import load_registered_node


def _guard_for_campaign(ledger_dir: Path, campaign_id: str) -> Path:
    candidates = (
        ledger_dir / f"{campaign_id}_pending.json",
        ledger_dir / f"{campaign_id}_trials_pending.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List, inspect, or safely fail stale pending-trial guards.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List all pending guards.")
    action.add_argument("--inspect", metavar="CAMPAIGN_ID", help="Show one pending guard.")
    action.add_argument("--mark-failed", metavar="CAMPAIGN_ID", help="Append failed_invalid, then delete guard.")
    parser.add_argument("--reason", help="Failure reason required with --mark-failed.")
    parser.add_argument("--node", default="resnet_trigger", help="Registered node name.")
    parser.add_argument(
        "--ledger-dir",
        default=str(ROOT / "experiments" / "ledgers"),
        help="Directory containing campaign ledgers and pending guards.",
    )
    parser.add_argument("--manager-mode", default="unknown_manager")
    parser.add_argument("--worker-mode", default="unknown_worker")
    parser.add_argument("--memory-mode", default="unknown")
    args = parser.parse_args()

    ledger_dir = Path(args.ledger_dir)

    if args.list:
        guards = [str(path) for path in list_pending_guards(ledger_dir)]
        print(json.dumps({"pending_guards": guards}, indent=2, sort_keys=True))
        return 0

    if args.inspect:
        guard = _guard_for_campaign(ledger_dir, args.inspect)
        if not guard.exists():
            print(f"[error] pending guard not found for campaign: {args.inspect}", file=sys.stderr)
            return 1
        print(json.dumps(inspect_pending_guard(guard), indent=2, sort_keys=True))
        return 0

    if args.mark_failed:
        if not args.reason:
            parser.error("--reason is required with --mark-failed")
        guard = _guard_for_campaign(ledger_dir, args.mark_failed)
        if not guard.exists():
            print(f"[error] pending guard not found for campaign: {args.mark_failed}", file=sys.stderr)
            return 1
        node_spec = load_registered_node(args.node, repo_root=ROOT)
        record = mark_pending_failed(
            guard_path=guard,
            node_spec=node_spec,
            manager_mode=args.manager_mode,
            worker_mode=args.worker_mode,
            memory_mode=args.memory_mode,
            failure_message=args.reason,
        )
        print(json.dumps({"record": record.to_dict()}, indent=2, sort_keys=True))
        return 0

    parser.error("unreachable")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
