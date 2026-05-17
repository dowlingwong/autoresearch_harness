#!/usr/bin/env python3
"""Run a one-trial invalid-config stress campaign on an OpenML node."""
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
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker


class InvalidConfigManager:
    mode = "invalid_config_stress_manager"

    def __init__(self, old_max_depth: str) -> None:
        self._old_max_depth = old_max_depth

    def propose_next_trial(self, status, memory_context, node_spec):  # noqa: ANN001
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary="force-invalid-max-depth",
            proposal_rationale="Stress invalid-config rejection with an out-of-range tree depth.",
            target_files=node_spec.editable_paths,
            objective=(
                f"Change max_depth from {self._old_max_depth} to 999 in config.yaml "
                "and keep all other hyperparameters unchanged."
            ),
            extra={
                "deterministic_patch": True,
                "structured_edit": {
                    "type": "config_value",
                    "path": "config.yaml",
                    "symbol": "max_depth",
                    "old": self._old_max_depth,
                    "new": "999",
                    "effective_key": "max_depth",
                    "effective_before": self._old_max_depth,
                    "effective_after": "999",
                },
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="openml_bank_marketing",
                        choices=("openml_credit_g", "openml_bank_marketing"))
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    campaign_id = args.campaign_id or f"{args.node}_invalid_config_stress"
    node_spec = load_registered_node(args.node, repo_root=ROOT)
    old_max_depth = "5" if args.node == "openml_credit_g" else "8"
    records_path = ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"
    artifacts_dir = ROOT / "experiments" / "artifacts" / campaign_id
    event_store = None if args.no_events else CampaignEventStore(
        ROOT / "experiments" / "events" / f"{campaign_id}_events.jsonl"
    )
    worker = LocalWorker(
        node_root=ROOT / "nodes" / args.node,
        artifact_dir=artifacts_dir,
        timeout_seconds=120.0,
    )
    result = run_real_campaign(
        node_spec=node_spec,
        campaign_id=campaign_id,
        budget=1,
        manager_mode=InvalidConfigManager.mode,
        memory_mode="append_only_summary",
        records_path=records_path,
        worker=worker,
        proposal_backend=InvalidConfigManager(old_max_depth),
        event_store=event_store,
    )
    records = TrialAppendStore(records_path).read_all()
    record = records[0] if records else None
    validation = {
        "one_record": len(records) == 1,
        "failed_invalid": bool(record and record.decision == TrialDecision.FAILED_INVALID),
        "invalid_config": bool(record and record.failure_category == FailureCategory.INVALID_CONFIG),
        "patch_ref_present": bool(record and record.patch_ref and Path(record.patch_ref).exists()),
        "raw_log_ref_present": bool(record and record.raw_log_ref and Path(record.raw_log_ref).exists()),
        "provenance_complete": bool(
            record
            and record.provenance.proposal_id
            and record.provenance.patch_id
            and record.provenance.run_id
            and record.provenance.metric_id
            and record.provenance.decision_id
        ),
    }
    print(json.dumps({"campaign": result.to_dict(), "validation": validation}, indent=2, sort_keys=True))
    return 0 if all(validation.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
