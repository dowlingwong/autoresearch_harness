#!/usr/bin/env python3
"""Run a governed campaign on the synthetic lr_synthetic node using LocalWorker.

No Ollama required.  Uses BaselineManager with deterministic structured proposals
(LEARNING_RATE and REGULARIZATION sweeps) and LocalWorker for direct-edit execution.
This campaign demonstrates that the governance protocol generalises beyond the
ResNet-trigger node, addressing the single-node reviewer critique.

Usage (from repo root):
    python3 scripts/run_lr_synthetic_campaign.py \\
        --budget 5 --campaign-id lr_synth_baseline

Reset before running (idempotent):
    python3 scripts/reset_node_state.py --node lr_synthetic --campaign-id lr_synth_baseline

Post-run analysis:
    python3 scripts/analyze_lr_synthetic.py --campaign-id lr_synth_baseline
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

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Governed campaign on lr_synthetic node (LocalWorker, no Ollama).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--budget", type=int, default=5, help="Number of trials.")
    parser.add_argument("--campaign-id", default="lr_synth_baseline",
                        help="Campaign identifier.")
    parser.add_argument(
        "--memory-mode",
        default="append_only_summary",
        choices=("none", "append_only_summary", "append_only_summary_with_rationale"),
    )
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    node_spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    node_root = ROOT / "nodes" / "lr_synthetic"
    artifacts_dir = ROOT / "experiments" / "artifacts" / args.campaign_id
    records_path = ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"

    worker = LocalWorker(
        node_root=node_root,
        artifact_dir=artifacts_dir,
        timeout_seconds=120.0,
    )

    print(f"\nlr_synthetic Governed Campaign")
    print(f"  campaign_id : {args.campaign_id}")
    print(f"  budget      : {args.budget}")
    print(f"  memory_mode : {args.memory_mode}")
    print(f"  node_root   : {node_root}")
    print(f"  ledger      : {records_path}")
    print()

    event_store = None
    if not args.no_events:
        from autoresearch.memory.event_store import CampaignEventStore
        events_path = ROOT / "experiments" / "events" / f"{args.campaign_id}_events.jsonl"
        event_store = CampaignEventStore(events_path)

    result = run_real_campaign(
        node_spec=node_spec,
        campaign_id=args.campaign_id,
        budget=args.budget,
        manager_mode="baseline_manager",
        memory_mode=args.memory_mode,
        records_path=records_path,
        worker=worker,
        event_store=event_store,
    )

    # --- summary ---
    store = TrialAppendStore(records_path)
    records = store.read_all()
    decisions = [r.decision for r in records]
    kept = decisions.count(TrialDecision.KEPT)
    disc = decisions.count(TrialDecision.DISCARDED)
    fail = decisions.count(TrialDecision.FAILED_INVALID)
    best = max((r.parsed_metrics.get("val_score", 0.0) for r in records), default=0.0)
    prov_complete = all(
        r.provenance and all([
            r.provenance.proposal_id,
            r.provenance.patch_id,
            r.provenance.run_id,
            r.provenance.metric_id,
            r.provenance.decision_id,
        ])
        for r in records
    )

    print(f"\n{'='*50}")
    print(f"  lr_synthetic campaign: {args.campaign_id}")
    print(f"  trials    : {len(records)}")
    print(f"  kept      : {kept}  discarded: {disc}  failed_invalid: {fail}")
    print(f"  best val_score : {best:.6f}")
    print(f"  provenance : {'complete' if prov_complete else 'INCOMPLETE'}")
    print(f"{'='*50}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
