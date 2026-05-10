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

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.stress_worker import ScopeViolationWorker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the KDD forced-failure governance stress trial.",
    )
    parser.add_argument("--node", default="resnet_trigger")
    parser.add_argument("--campaign-id", default="kdd_stress_scope")
    parser.add_argument("--stress-mode", default="scope_violation", choices=("scope_violation",))
    parser.add_argument("--node-root", default=str(ROOT / "nodes" / "ResNet_trigger"))
    parser.add_argument("--records", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    records_path = (
        Path(args.records)
        if args.records
        else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    )
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else ROOT / "experiments" / "artifacts" / args.campaign_id
    )
    event_store = None if args.no_events else CampaignEventStore(
        Path(args.events) if args.events else _default_events_path(records_path, args.campaign_id)
    )
    worker = ScopeViolationWorker(node_root=args.node_root, artifacts_dir=artifacts_dir)
    result = run_real_campaign(
        node_spec=node_spec,
        campaign_id=args.campaign_id,
        budget=1,
        manager_mode="baseline_manager",
        memory_mode="append_only_summary_with_rationale",
        records_path=records_path,
        worker=worker,
        event_store=event_store,
    )
    records = TrialAppendStore(records_path).read_all()
    validation = _validate_stress_record(records, records_path, args.campaign_id)
    payload = {
        "campaign": result.to_dict(),
        "validation": validation,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(validation.values()) else 1


def _default_events_path(records_path: Path, campaign_id: str) -> Path:
    root = records_path.parent.parent if records_path.parent.name == "ledgers" else records_path.parent
    return root / "events" / f"{campaign_id}_events.jsonl"


def _validate_stress_record(records, records_path: Path, campaign_id: str) -> dict[str, bool]:
    if len(records) != 1:
        return {
            "one_record": False,
            "failed_invalid": False,
            "invalid_edit_scope": False,
            "patch_ref_present": False,
            "git_state_unchanged": False,
            "no_pending_guard": not (records_path.parent / f"{campaign_id}_pending.json").exists(),
            "provenance_complete": False,
        }
    record = records[0]
    provenance = record.provenance
    return {
        "one_record": True,
        "failed_invalid": record.decision == TrialDecision.FAILED_INVALID,
        "invalid_edit_scope": record.failure_category == FailureCategory.INVALID_EDIT_SCOPE,
        "patch_ref_present": bool(record.patch_ref and Path(record.patch_ref).exists()),
        "git_state_unchanged": record.git_commit_before == record.git_commit_after,
        "no_pending_guard": not (records_path.parent / f"{campaign_id}_pending.json").exists(),
        "provenance_complete": all(
            [
                provenance.proposal_id,
                provenance.patch_id,
                provenance.run_id,
                provenance.metric_id,
                provenance.decision_id,
            ]
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
