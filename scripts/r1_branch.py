#!/usr/bin/env python3
"""Run sibling branches from a campaign snapshot (R1 L7, ENGINEERING_ONLY).

Given a snapshot directory (written by ``SnapshotRecorder`` / ``write_snapshot``),
an action list and a horizon ``h``, restore the snapshot once per action into a
fresh isolated node directory, force that action as the first step, then run the
default policy for ``h`` further trials.  Writes ``branch_manifest.json``.

Usage:
  python3 scripts/r1_branch.py --snapshot <dir> --out <dir> \\
      --actions continue skip revise verify demote stop --horizon 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.snapshot import read_snapshot, resume_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Snapshot directory (with manifest.json).")
    parser.add_argument("--out", required=True, help="Output directory for branch dirs and manifest.")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["continue", "skip", "revise", "verify", "demote", "stop"],
        help="Actions to run as siblings.",
    )
    parser.add_argument("--horizon", type=int, default=0, help="Further trials after the forced action.")
    args = parser.parse_args()

    snapshot = read_snapshot(args.snapshot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "r1.branch_manifest/1",
        "label": "ENGINEERING_ONLY",
        "snapshot": str(Path(args.snapshot)),
        "node": snapshot.node_name,
        "epoch": snapshot.epoch,
        "budget_index": snapshot.budget_index,
        "horizon": args.horizon,
        "branches": [],
    }

    for action in args.actions:
        branch_dir = out / f"branch_{action}"
        entry = {"action": action, "branch_dir": str(branch_dir)}
        try:
            result = resume_campaign(snapshot, branch_dir, first_action=action, horizon=args.horizon)
            record = result.first_record
            control = dict(record.extra.get("control", {})) if record is not None else {}
            continuation = result.continuation
            entry.update(
                {
                    "effective_action": control.get("action"),
                    "decision": record.decision.value if record is not None else None,
                    "metric": (
                        record.parsed_metrics.get(result.restored.node_spec.metric_name)
                        if record is not None
                        else None
                    ),
                    "records_path": str(result.restored.records_path),
                    "events_path": str(result.restored.events_path),
                    "node_dir": str(result.restored.into_dir),
                    "continuation_records": (
                        continuation.records_written if continuation is not None else 0
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - record the blocker, never crash the sweep
            entry["error"] = f"{type(exc).__name__}: {exc}"
        manifest["branches"].append(entry)

    manifest_path = out / "branch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
