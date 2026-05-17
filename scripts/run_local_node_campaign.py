#!/usr/bin/env python3
"""Run a governed LocalWorker campaign on any registered local node.

This runner is intentionally generic. It is useful for public benchmark
adapters such as ``mlagentbench_vectorization`` and for smoke-testing bounded
config-edit nodes before spending API calls on larger campaigns.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker


def _manager_backend(manager: str, model: str, host: str, temperature: float):
    if manager != "langgraph_manager":
        return None
    from autoresearch.manager.langgraph_manager import LangGraphManager

    return LangGraphManager(model=model, host=host, temperature=temperature)


def _default_node_root(node_name: str) -> Path:
    exact = ROOT / "nodes" / node_name
    if exact.exists():
        return exact
    # Normalize hyphens and underscores so that e.g. "autoresearch-macos"
    # is found when the registered name is "autoresearch_macos".
    def _normalise(s: str) -> str:
        return s.lower().replace("-", "_")
    matches = [p for p in (ROOT / "nodes").glob("*") if _normalise(p.name) == _normalise(node_name)]
    if matches:
        return matches[0]
    return exact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="Registered node name.")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument(
        "--manager",
        choices=("baseline_manager", "prompt_manager", "langgraph_manager"),
        default="baseline_manager",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("none", "append_only_summary", "append_only_summary_with_rationale"),
        default="append_only_summary",
    )
    parser.add_argument("--node-root", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-reset", dest="reset", action="store_false")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    node_root = Path(args.node_root).resolve() if args.node_root else _default_node_root(args.node)
    campaign_id = args.campaign_id or f"{args.node}_{args.manager}_{args.budget}"
    records_path = ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"
    artifacts_dir = ROOT / "experiments" / "artifacts" / campaign_id
    events_path = ROOT / "experiments" / "events" / f"{campaign_id}_events.jsonl"

    if args.reset:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "reset_node_state.py"),
                "--node",
                args.node,
                "--campaign-id",
                campaign_id,
                "--node-root",
                str(node_root),
            ],
            check=True,
        )

    print()
    print(f"Local governed campaign: {args.node}")
    print(f"  campaign_id : {campaign_id}")
    print(f"  budget      : {args.budget}")
    print(f"  manager     : {args.manager}")
    print(f"  memory_mode : {args.memory_mode}")
    print(f"  metric      : {node_spec.metric_name} ({node_spec.metric_direction})")
    print(f"  node_root   : {node_root}")
    print(f"  ledger      : {records_path}")

    worker = LocalWorker(
        node_root=node_root,
        artifact_dir=artifacts_dir,
        timeout_seconds=args.timeout_seconds,
    )
    proposal_backend = _manager_backend(args.manager, args.model, args.host, args.temperature)
    event_store = None if args.no_events else CampaignEventStore(events_path)

    run_real_campaign(
        node_spec=node_spec,
        campaign_id=campaign_id,
        budget=args.budget,
        manager_mode=args.manager,
        memory_mode=args.memory_mode,
        records_path=records_path,
        worker=worker,
        proposal_backend=proposal_backend,
        manager_temperature=args.temperature,
        event_store=event_store,
    )

    records = TrialAppendStore(records_path).read_all()
    decisions = [record.decision for record in records]
    kept = decisions.count(TrialDecision.KEPT)
    discarded = decisions.count(TrialDecision.DISCARDED)
    failed = decisions.count(TrialDecision.FAILED_INVALID)
    best = max((record.parsed_metrics.get(node_spec.metric_name, 0.0) for record in records), default=0.0)
    provenance_complete = all(
        record.provenance
        and record.provenance.proposal_id
        and record.provenance.patch_id
        and record.provenance.run_id
        and record.provenance.metric_id
        and record.provenance.decision_id
        for record in records
    )

    print()
    print(f"Completed {campaign_id}")
    print(f"  trials         : {len(records)}")
    print(f"  kept/disc/fail : {kept}/{discarded}/{failed}")
    print(f"  best {node_spec.metric_name}: {best:.6f}")
    print(f"  provenance     : {'complete' if provenance_complete else 'incomplete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
