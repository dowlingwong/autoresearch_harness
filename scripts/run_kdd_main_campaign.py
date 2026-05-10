#!/usr/bin/env python3
"""Run the KDD AAE main 5-trial governed campaign.

This is the primary paper experiment (Chunk 2.2). It produces the main
lifecycle distribution result: kept / discarded / failed_invalid counts,
acceptance rate, provenance completeness, and AUC trajectory.

Always reset node state before running to ensure a reproducible start:
  python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id kdd_main_5trial

Dry-run (verify plumbing, no Ollama required):
  python3 scripts/run_kdd_main_campaign.py \\
      --node resnet_trigger --budget 5 --campaign-id kdd_main_5trial --dry-run

Real campaign (requires Ollama + node directory):
  python3 scripts/run_kdd_main_campaign.py \\
      --node resnet_trigger \\
      --budget 5 \\
      --campaign-id kdd_main_5trial \\
      --manager prompt_manager \\
      --memory-mode append_only_summary_with_rationale \\
      --node-root nodes/ResNet_trigger \\
      --model qwen2.5-coder:7b \\
      --host http://localhost:11434

Post-run export:
  python3 scripts/export_kdd_tables.py --campaign-id kdd_main_5trial

Post-run checks (acceptance criterion):
  - Ledger has exactly 5 records.
  - At least one 'kept' AND at least one 'discarded' or 'failed_invalid' record.
  - All records have provenance fully populated.
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

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.llm.langchain_client import LangChainProposalBackend
from autoresearch.llm.providers import resolve_worker_model_args
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.reporting.export_tables import export_campaign_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the KDD AAE primary governed campaign (Chunk 2.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--node", required=True, help="Registered node name.")
    parser.add_argument("--budget", required=True, type=int, help="Number of trials (5 for main campaign).")
    parser.add_argument("--campaign-id", required=True, help="Campaign identifier (e.g. kdd_main_5trial).")
    parser.add_argument(
        "--manager",
        default="prompt_manager",
        choices=("baseline_manager", "prompt_manager", "langgraph_manager"),
        help="Manager implementation.",
    )
    parser.add_argument(
        "--memory-mode",
        default="append_only_summary_with_rationale",
        choices=("none", "append_only_summary", "append_only_summary_with_rationale"),
        help="Memory mode for manager context.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunWorker (no Ollama or node-root needed). Validates plumbing only.",
    )
    parser.add_argument(
        "--dry-run-profile",
        choices=("mixed_lifecycle", "monotonic"),
        default="mixed_lifecycle",
        help=(
            "Synthetic dry-run profile. mixed_lifecycle exercises kept, discarded, "
            "and failed_invalid decisions; monotonic preserves the legacy all-kept profile."
        ),
    )
    # Real-run arguments
    parser.add_argument("--node-root", default=None,
                        help="Path to the node directory (required for real runs).")
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Model id. Supports provider/model form such as ollama/qwen2.5-coder:7b.",
    )
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument(
        "--llm-backend",
        default="native",
        choices=("native", "langchain"),
        help="Proposal backend. native uses the selected manager; langchain uses LangChainProposalBackend.",
    )
    parser.add_argument(
        "--require-isolated-branch",
        action="store_true",
        help="Require legacy autoresearch/<tag> branch isolation before running.",
    )
    parser.add_argument("--records", default=None,
                        help="Override ledger path.")
    parser.add_argument("--events", default=None, help="Override campaign event-stream JSONL path.")
    parser.add_argument("--no-events", action="store_true", help="Disable campaign event stream.")
    parser.add_argument("--tables-dir", default=str(ROOT / "paper" / "tables"))
    parser.add_argument("--no-export", action="store_true",
                        help="Skip table export after campaign completes.")
    args = parser.parse_args()

    if args.budget < 1:
        parser.error("--budget must be >= 1")

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    records_path = (
        Path(args.records) if args.records
        else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    event_store = None if args.no_events else CampaignEventStore(
        Path(args.events) if args.events else _default_events_path(records_path, args.campaign_id)
    )

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}KDD AAE Main Campaign")
    print(f"  node        : {args.node}")
    print(f"  campaign_id : {args.campaign_id}")
    print(f"  budget      : {args.budget}")
    print(f"  manager     : {args.manager}")
    print(f"  llm_backend : {args.llm_backend}")
    print(f"  memory_mode : {args.memory_mode}")
    print(f"  ledger      : {records_path}")
    if event_store is not None:
        print(f"  events      : {event_store.path}")
    print()

    proposal_backend = None
    if args.llm_backend == "langchain":
        proposal_backend = LangChainProposalBackend(
            args.model,
            artifacts_dir=ROOT / "experiments" / "artifacts" / args.campaign_id,
        )

    if args.dry_run:
        result = run_dry_campaign(
            node_spec=node_spec,
            campaign_id=args.campaign_id,
            budget=args.budget,
            manager_mode=args.manager,
            memory_mode=args.memory_mode,
            records_path=records_path,
            dry_run_profile=args.dry_run_profile,
            proposal_backend=proposal_backend,
            event_store=event_store,
        )
    else:
        if not args.node_root:
            parser.error("--node-root is required for real campaigns")
        from autoresearch.worker.claw_worker import ClawWorker
        artifacts_dir = ROOT / "experiments" / "artifacts" / args.campaign_id
        worker_model, worker_host = resolve_worker_model_args(args.model, args.host)
        worker = ClawWorker(
            repo_root=ROOT,
            node_root=Path(args.node_root),
            artifacts_dir=artifacts_dir,
            model=worker_model,
            host=worker_host,
            allow_any_branch=not args.require_isolated_branch,
        )
        result = run_real_campaign(
            node_spec=node_spec,
            campaign_id=args.campaign_id,
            budget=args.budget,
            manager_mode=args.manager,
            memory_mode=args.memory_mode,
            records_path=records_path,
            worker=worker,
            proposal_backend=proposal_backend,
            event_store=event_store,
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    # Post-run validation (acceptance criterion).
    records = TrialAppendStore(records_path).read_all()
    _validate_main_campaign(records, args.campaign_id, args.budget, records_path)

    if not args.no_export:
        summary = load_campaign_summary(records_path,
                                        metric_name=node_spec.metric_name,
                                        metric_direction=node_spec.metric_direction)
        outputs = export_campaign_tables(summary, args.tables_dir)
        print("\nExported tables:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")
        print(f"\nNext step: python3 scripts/export_kdd_tables.py --campaign-id {args.campaign_id}")

    return 0


def _validate_main_campaign(records, campaign_id: str, budget: int, ledger_path: Path) -> None:
    print("\nPost-run validation (Chunk 2.2 acceptance criterion):")

    ok_count = len(records) == budget
    print(f"  {'✅' if ok_count else '❌'} Ledger has {len(records)}/{budget} records")

    kept = [r for r in records if r.decision == TrialDecision.KEPT]
    non_kept = [r for r in records if r.decision != TrialDecision.KEPT]
    if budget < 2:
        ok_diversity = True
        print(
            f"  ℹ  Decision diversity skipped for budget={budget} smoke: "
            f"{len(kept)} kept, {len(non_kept)} non-kept"
        )
    else:
        ok_diversity = len(kept) >= 1 and len(non_kept) >= 1
        print(f"  {'✅' if ok_diversity else '⚠ '} Decision diversity: {len(kept)} kept, {len(non_kept)} non-kept")
        if not ok_diversity and len(kept) == budget:
            print("    → All trials kept. Fast-training config may be too easy. Consider --budget 10.")

    ok_provenance = all(
        all(v for v in [
            r.provenance.proposal_id, r.provenance.patch_id,
            r.provenance.run_id, r.provenance.metric_id, r.provenance.decision_id
        ]) for r in records
    )
    print(f"  {'✅' if ok_provenance else '❌'} All records have complete provenance")

    guard_alt = ledger_path.parent / f"{campaign_id}_pending.json"
    ok_guard = not guard_alt.exists()
    print(f"  {'✅' if ok_guard else '❌'} No pending guard remains")

    if ok_count and ok_diversity and ok_provenance and ok_guard:
        print(f"\n  Acceptance criterion met for {campaign_id}.")
    else:
        print(f"\n  ⚠  One or more checks failed — review before proceeding to Chunk 2.4.")


def _default_events_path(records_path: Path, campaign_id: str) -> Path:
    root = records_path.parent.parent if records_path.parent.name == "ledgers" else records_path.parent
    return root / "events" / f"{campaign_id}_events.jsonl"


if __name__ == "__main__":
    raise SystemExit(main())
