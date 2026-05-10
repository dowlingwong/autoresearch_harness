#!/usr/bin/env python3
"""Compare manager strategies under equal budget on a single node.

This is Priority 13 from the Stage 2 plan — an optional secondary experiment.
It is only meaningful to run after the main campaign (Priority 9) and memory
ablation (Priority 12) are complete.

Usage
-----
Dry-run (no worker, no Ollama):

  python3 scripts/run_manager_comparison.py \\
      --node resnet_trigger \\
      --budget 5 \\
      --dry-run

Real run (requires Ollama at --host):

  python3 scripts/run_manager_comparison.py \\
      --node resnet_trigger \\
      --budget 5 \\
      --memory-mode append_only_summary_with_rationale \\
      --node-root nodes/ResNet_trigger \\
      --model qwen2.5-coder:7b \\
      --host http://localhost:11434

Each manager (baseline_manager, prompt_manager, and optionally langgraph_manager)
is run under identical conditions:
  - same node spec
  - same budget
  - same memory mode
  - same worker constraints

Results are written to:
  experiments/ledgers/<campaign-id>_<manager>_trials.jsonl

A summary CSV is written to:
  paper/tables/manager_comparison_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.nodes.registry import load_registered_node


MANAGERS = ["baseline_manager", "prompt_manager"]
OPTIONAL_MANAGERS = ["langgraph_manager"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare manager strategies under equal budget (Stage 2 Priority 13).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--node", required=True, help="Registered node name (e.g. resnet_trigger)")
    parser.add_argument("--budget", required=True, type=int, help="Number of trials per manager")
    parser.add_argument(
        "--campaign-id",
        default="manager_comparison",
        help="Base campaign ID; each manager gets <campaign-id>_<manager> (default: manager_comparison)",
    )
    parser.add_argument(
        "--memory-mode",
        default="append_only_summary_with_rationale",
        choices=["none", "append_only_summary", "append_only_summary_with_rationale"],
        help="Memory mode fed to all managers (default: append_only_summary_with_rationale)",
    )
    parser.add_argument(
        "--managers",
        nargs="+",
        default=MANAGERS,
        help=f"Managers to compare (default: {MANAGERS}). Add 'langgraph_manager' optionally.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use DryRunWorker; no Ollama required")
    # Real-run arguments (forwarded to run_real_campaign via ClawWorker)
    parser.add_argument("--node-root", help="Path to the node working directory (required for real runs)")
    parser.add_argument(
        "--artifacts-dir",
        default=str(ROOT / "experiments" / "artifacts"),
        help="Directory for trial artefacts",
    )
    parser.add_argument("--packet-defaults", help="Optional JSON packet-defaults file")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--allow-any-branch", action="store_true")
    parser.add_argument("--llm-stub", action="store_true", help="Inject a stub LLM for LangGraph manager")
    parser.add_argument(
        "--records-dir",
        default=str(ROOT / "experiments" / "ledgers"),
        help="Directory for JSONL ledger files",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(ROOT / "paper" / "tables"),
        help="Directory for output CSVs",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.node_root:
        parser.error("--node-root is required for real runs")

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    records_dir = Path(args.records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for manager_mode in args.managers:
        campaign_id = f"{args.campaign_id}_{manager_mode}"
        records_path = records_dir / f"{campaign_id}_trials.jsonl"

        print(f"\n{'='*60}")
        print(f"Running manager: {manager_mode}")
        print(f"  campaign_id : {campaign_id}")
        print(f"  budget      : {args.budget}")
        print(f"  memory_mode : {args.memory_mode}")
        print(f"  records     : {records_path}")
        print(f"{'='*60}")

        try:
            manager_llm = None
            if args.llm_stub and manager_mode == "langgraph_manager":
                from langchain_core.language_models import FakeListChatModel  # noqa: PLC0415
                import json as _json  # noqa: PLC0415

                _stub_response = _json.dumps({
                    "summary": "stub-proposal",
                    "rationale": "LLM stub for smoke testing",
                    "objective": (
                        f"In train.py, make one small bounded change to improve {args.node}. "
                        "Edit only train.py. Do not run the experiment."
                    ),
                })
                manager_llm = FakeListChatModel(responses=[_stub_response] * max(args.budget, 1))

            if args.dry_run:
                result = run_dry_campaign(
                    node_spec=node_spec,
                    campaign_id=campaign_id,
                    budget=args.budget,
                    manager_mode=manager_mode,
                    memory_mode=args.memory_mode,
                    records_path=records_path,
                    manager_llm=manager_llm,
                )
                print(f"  [dry-run] {result.records_written} records written")
            else:
                # Build the real ClawWorker.
                from autoresearch.worker.claw_worker import ClawWorker  # noqa: PLC0415

                packet_defaults = None
                if args.packet_defaults:
                    import json  # noqa: PLC0415
                    packet_defaults = json.loads(Path(args.packet_defaults).read_text())

                worker = ClawWorker(
                    repo_root=ROOT,
                    node_root=Path(args.node_root),
                    artifacts_dir=Path(args.artifacts_dir) / campaign_id,
                    model=args.model,
                    host=args.host,
                    allow_any_branch=args.allow_any_branch,
                    packet_defaults=packet_defaults or {},
                )
                run_real_campaign(
                    node_spec=node_spec,
                    campaign_id=campaign_id,
                    budget=args.budget,
                    manager_mode=manager_mode,
                    memory_mode=args.memory_mode,
                    records_path=records_path,
                    worker=worker,
                    manager_llm=manager_llm,
                )
                print(f"  [real] campaign complete → {records_path}")

            # Summarise
            if records_path.exists():
                summary = load_campaign_summary(
                    records_path,
                    metric_name=node_spec.metric_name,
                    metric_direction=node_spec.metric_direction,
                )
                m = summary.metrics
                results.append({
                    "manager_mode": manager_mode,
                    "campaign_id": campaign_id,
                    "memory_mode": args.memory_mode,
                    "budget": args.budget,
                    "total_trials": m.total_trials,
                    "kept_count": m.kept_count,
                    "discarded_count": m.discarded_count,
                    "failed_invalid_count": m.failed_invalid_count,
                    "acceptance_rate": m.acceptance_rate,
                    "invalid_rate": m.invalid_rate,
                    "initial_metric": m.initial_metric,
                    "best_metric": m.best_metric,
                    "net_gain": m.net_gain,
                    "gain_per_trial": m.gain_per_trial,
                    "complete_provenance_rate": m.complete_provenance_rate,
                    "artifact_capture_completeness": m.artifact_capture_completeness,
                    "total_wall_clock_seconds": m.total_wall_clock_seconds,
                })
                print(
                    f"  best={m.best_metric}  net_gain={m.net_gain}  "
                    f"kept={m.kept_count}/{m.total_trials}"
                )

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR in {manager_mode}: {exc}", file=sys.stderr)
            results.append({"manager_mode": manager_mode, "campaign_id": campaign_id, "error": str(exc)})

    # Write comparison CSV
    if results:
        out_path = tables_dir / "manager_comparison_summary.csv"
        all_keys: list[str] = []
        for row in results:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nComparison summary written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
