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

from autoresearch.evaluation.ablations import build_memory_ablation_plan, export_memory_ablation_summary
from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.nodes.spec import load_node_spec

MEMORY_MODES = ("none", "append_only_summary", "append_only_summary_with_rationale")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or plan the Stage 2 memory/governance ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plan only (no execution):
  python3 scripts/run_memory_ablation.py --node resnet_trigger --budget 3 --dry-run

Dry campaigns (mock DryRunWorker, all three memory modes):
  python3 scripts/run_memory_ablation.py --node resnet_trigger --budget 3 --execute-dry-campaigns

Real equal-budget campaigns (requires Ollama + node):
  python3 scripts/run_memory_ablation.py \\
      --node resnet_trigger --budget 5 \\
      --execute-real-campaigns \\
      --node-root nodes/ResNet_trigger \\
      --model qwen2.5-coder:7b --host http://localhost:11434
""",
    )
    parser.add_argument("--node", required=True, choices=("resnet_trigger",))
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--records", help="JSONL file to read pre-existing records from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan only — do not execute any campaigns.")
    parser.add_argument("--execute-dry-campaigns", action="store_true",
                        help="Run mock DryRunWorker campaigns for all three memory modes.")
    parser.add_argument("--execute-real-campaigns", action="store_true",
                        help="Run real ClawWorker campaigns for all three memory modes.")
    parser.add_argument("--ledger-dir", default=str(ROOT / "experiments" / "ledgers"))
    parser.add_argument("--artifacts-dir", default=str(ROOT / "experiments" / "artifacts"))
    parser.add_argument("--out", default=str(ROOT / "paper" / "tables" / "memory_ablation_summary.csv"))
    # Real-run args
    parser.add_argument("--node-root", help="Node working directory (required for --execute-real-campaigns)")
    parser.add_argument("--packet-defaults",
                        help="Optional JSON file for timeout/log_path overrides (objective always from proposal)")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    if args.budget < 1:
        parser.error("--budget must be >= 1")

    node_spec = load_node_spec(ROOT / "configs" / "nodes" / f"{args.node}.yaml")
    ledger_dir = Path(args.ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    campaign_outputs = []

    if args.execute_real_campaigns:
        if not args.node_root:
            parser.error("--node-root is required for --execute-real-campaigns")

        from autoresearch.worker.claw_worker import ClawWorker

        for mode in MEMORY_MODES:
            campaign_id = f"{args.node}_{mode}_ablation"
            records_path = ledger_dir / f"{campaign_id}_trials.jsonl"

            if args.packet_defaults:
                worker = ClawWorker.from_packet_defaults_file(
                    repo_root=ROOT,
                    node_root=args.node_root,
                    packet_defaults_path=args.packet_defaults,
                    artifacts_dir=args.artifacts_dir,
                    model=args.model,
                    host=args.host,
                )
            else:
                worker = ClawWorker(
                    repo_root=ROOT,
                    node_root=args.node_root,
                    artifacts_dir=args.artifacts_dir,
                    model=args.model,
                    host=args.host,
                )

            result = run_real_campaign(
                node_spec=node_spec,
                campaign_id=campaign_id,
                budget=args.budget,
                manager_mode="prompt_manager",
                memory_mode=mode,
                records_path=records_path,
                worker=worker,
            )
            campaign_outputs.append(result.to_dict())

    elif args.execute_dry_campaigns:
        for mode in MEMORY_MODES:
            campaign_id = f"{args.node}_{mode}_ablation"
            records_path = ledger_dir / f"{campaign_id}_trials.jsonl"
            result = run_dry_campaign(
                node_spec=node_spec,
                campaign_id=campaign_id,
                budget=args.budget,
                manager_mode="baseline_manager",
                memory_mode=mode,
                records_path=records_path,
            )
            campaign_outputs.append(result.to_dict())

    # Load records for summary
    records = []
    if args.records:
        records = TrialAppendStore(args.records).read_all()
    elif campaign_outputs:
        for output in campaign_outputs:
            records.extend(TrialAppendStore(str(output["records_path"])).read_all())

    rows = build_memory_ablation_plan(node_spec=node_spec, records=records, budget=args.budget)
    output = export_memory_ablation_summary(rows, args.out)
    print(
        json.dumps(
            {
                "node": args.node,
                "budget": args.budget,
                "dry_run": bool(args.dry_run),
                "execute_dry_campaigns": bool(args.execute_dry_campaigns),
                "execute_real_campaigns": bool(args.execute_real_campaigns),
                "output": str(output),
                "campaign_outputs": campaign_outputs,
                "rows": [row.to_dict() for row in rows],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
