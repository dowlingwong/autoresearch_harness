#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a promoted-master snapshot from a campaign ledger.")
    parser.add_argument("--ledger", required=True, help="Trial JSONL ledger.")
    parser.add_argument(
        "--metric-name",
        default="val_auc",
        help="Metric used to choose the promoted kept trial.",
    )
    parser.add_argument(
        "--metric-direction",
        default="maximize",
        choices=["maximize", "minimize"],
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper" / "tables" / "promoted_master_snapshot.json"),
    )
    args = parser.parse_args()

    records = TrialAppendStore(args.ledger).read_all()
    kept = [record for record in records if record.decision == TrialDecision.KEPT]
    reverse = args.metric_direction == "maximize"
    promoted = sorted(
        kept,
        key=lambda record: record.parsed_metrics.get(args.metric_name, float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )[0] if kept else None
    payload = {
        "ledger": str(Path(args.ledger)),
        "metric_name": args.metric_name,
        "metric_direction": args.metric_direction,
        "created_at": _iso_now(),
        "total_trials": len(records),
        "kept_trials": len(kept),
        "promoted_trial": promoted.trial_id if promoted else None,
        "promoted_metric": (
            promoted.parsed_metrics.get(args.metric_name) if promoted else None
        ),
        "patch_ref": promoted.patch_ref if promoted else None,
        "git_commit_after": promoted.git_commit_after if promoted else None,
        "node_state_hash": promoted.node_state_hash if promoted else None,
        "patch_hash": promoted.patch_hash if promoted else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={output}")
    return 0


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
