#!/usr/bin/env python3
"""Run governed LocalWorker campaigns on the OpenML tabular nodes.

Examples
--------
Run both public tabular campaigns with deterministic baseline proposals:

    uv run python3 scripts/run_openml_tabular_campaign.py --node all --budget 20

Run credit-g with LangGraph proposals:

    uv run python3 scripts/run_openml_tabular_campaign.py \\
        --node openml_credit_g --budget 20 --manager langgraph_manager

Run credit-g with a DeepSeek API manager (LocalWorker still runs locally):

    DEEPSEEK_API_KEY=... DEEPSEEK_THINKING=disabled \\
    uv run python3 scripts/run_openml_tabular_campaign.py \\
        --node openml_credit_g --budget 20 --manager langgraph_manager \\
        --model deepseek/deepseek-v4-flash

Summarize/export after the run:

    uv run python3 scripts/export_openml_paper_table.py
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


NODE_CHOICES = ("openml_credit_g", "openml_bank_marketing")


def _default_campaign_id(node_name: str, budget: int) -> str:
    return {
        "openml_credit_g": f"openml_credit_g_main_{budget}",
        "openml_bank_marketing": f"openml_bank_marketing_main_{budget}",
    }[node_name]


def _manager_backend(manager: str, model: str, host: str, temperature: float):
    if manager != "langgraph_manager":
        return None
    from autoresearch.manager.langgraph_manager import LangGraphManager

    return LangGraphManager(model=model, host=host, temperature=temperature)


def _run_one(args: argparse.Namespace, node_name: str, campaign_id: str) -> None:
    node_spec = load_registered_node(node_name, repo_root=ROOT)
    node_root = ROOT / "nodes" / node_name
    records_path = ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"
    artifacts_dir = ROOT / "experiments" / "artifacts" / campaign_id
    events_path = ROOT / "experiments" / "events" / f"{campaign_id}_events.jsonl"

    if args.reset:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "reset_node_state.py"),
                "--node",
                node_name,
                "--campaign-id",
                campaign_id,
            ],
            check=True,
        )

    print()
    print(f"OpenML governed campaign: {node_name}")
    print(f"  campaign_id : {campaign_id}")
    print(f"  budget      : {args.budget}")
    print(f"  manager     : {args.manager}")
    print(f"  memory_mode : {args.memory_mode}")
    print(f"  node_root   : {node_root}")
    print(f"  ledger      : {records_path}")

    worker = LocalWorker(
        node_root=node_root,
        artifact_dir=artifacts_dir,
        timeout_seconds=args.timeout_seconds,
    )
    event_store = None if args.no_events else CampaignEventStore(events_path)
    proposal_backend = _manager_backend(args.manager, args.model, args.host, args.temperature)

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenML tabular governed campaigns with LocalWorker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--node", choices=(*NODE_CHOICES, "all"), default="all")
    parser.add_argument("--budget", type=int, default=20)
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
    parser.add_argument("--campaign-id", help="Only valid when --node is not all.")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Manager model for langgraph_manager, e.g. qwen2.5-coder:7b or deepseek/deepseek-v4-flash.",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-reset", dest="reset", action="store_false")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    if args.node == "all" and args.campaign_id:
        parser.error("--campaign-id can only be used with a single --node")

    nodes = NODE_CHOICES if args.node == "all" else (args.node,)
    for node_name in nodes:
        campaign_id = args.campaign_id or _default_campaign_id(node_name, args.budget)
        _run_one(args, node_name, campaign_id)

    print()
    print("Next:")
    print("  uv run python3 scripts/export_openml_paper_table.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
