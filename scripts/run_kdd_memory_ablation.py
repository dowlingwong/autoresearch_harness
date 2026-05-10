#!/usr/bin/env python3
"""Run a single-mode KDD AAE memory-ablation campaign.

This script is the per-mode building block for the full ablation (Chunk 2.4).
Run it once per memory mode so each campaign gets its own ledger and can be
reset independently before starting.

Smoke test (1 trial, dry-run — no Ollama required):
  python3 scripts/run_kdd_memory_ablation.py \\
      --node resnet_trigger --budget 1 \\
      --memory-mode none \\
      --campaign-id smoke_none \\
      --dry-run

Real ablation campaign (requires Ollama + node directory):
  python3 scripts/run_kdd_memory_ablation.py \\
      --node resnet_trigger --budget 5 \\
      --memory-mode append_only_summary_with_rationale \\
      --campaign-id ablation_append_only_summary_with_rationale \\
      --manager prompt_manager \\
      --node-root nodes/ResNet_trigger \\
      --model qwen2.5-coder:7b --host http://localhost:11434

Full ablation loop (run after reset for each mode):
  for mode in none append_only_summary append_only_summary_with_rationale; do
    python3 scripts/reset_node_state.py --node resnet_trigger --campaign-id ablation_${mode}
    python3 scripts/run_kdd_memory_ablation.py \\
        --node resnet_trigger --budget 5 \\
        --memory-mode ${mode} \\
        --campaign-id ablation_${mode} \\
        --manager prompt_manager \\
        --node-root nodes/ResNet_trigger \\
        --model qwen2.5-coder:7b --host http://localhost:11434
  done
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

VALID_MEMORY_MODES = (
    "none",
    "append_only_summary",
    "append_only_summary_with_rationale",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one arm of the KDD AAE memory ablation (single memory mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--node", required=True, help="Registered node name (e.g. resnet_trigger).")
    parser.add_argument(
        "--memory-mode",
        required=True,
        choices=VALID_MEMORY_MODES,
        help="Memory mode for this ablation arm.",
    )
    parser.add_argument("--campaign-id", required=True, help="Unique campaign identifier for this arm.")
    parser.add_argument("--budget", required=True, type=int, help="Number of trials to run.")
    parser.add_argument("--manager", default="baseline_manager",
                        help="Manager to use (baseline_manager | prompt_manager | langgraph_manager).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunWorker with synthetic metrics. No Ollama or node-root required.",
    )
    parser.add_argument(
        "--dry-run-profile",
        default="auto",
        choices=(
            "auto",
            "monotonic",
            "mixed_lifecycle",
            "ablation_none",
            "ablation_append_only_summary",
            "ablation_append_only_summary_with_rationale",
        ),
        help=(
            "Synthetic profile for dry-run campaigns. auto maps each memory mode "
            "to a deterministic ablation profile."
        ),
    )
    # Real-run arguments
    parser.add_argument("--node-root", default=None,
                        help="Path to the node directory containing train.py (required for real runs).")
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
                        help="Override ledger path (default: experiments/ledgers/<campaign-id>_trials.jsonl).")
    parser.add_argument("--events", default=None, help="Override campaign event-stream JSONL path.")
    parser.add_argument("--no-events", action="store_true", help="Disable campaign event stream.")
    parser.add_argument("--tables-dir", default=str(ROOT / "paper" / "tables"),
                        help="Directory to write per-campaign CSV summary tables.")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip table export after the campaign completes.")
    args = parser.parse_args()

    if args.budget < 1:
        parser.error("--budget must be >= 1")

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    records_path = (
        Path(args.records)
        if args.records
        else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    event_store = None if args.no_events else CampaignEventStore(
        Path(args.events) if args.events else _default_events_path(records_path, args.campaign_id)
    )

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}KDD AAE Memory Ablation")
    print(f"  node         : {args.node}")
    print(f"  memory_mode  : {args.memory_mode}")
    print(f"  campaign_id  : {args.campaign_id}")
    print(f"  budget       : {args.budget}")
    print(f"  manager      : {args.manager}")
    print(f"  llm_backend  : {args.llm_backend}")
    print(f"  ledger       : {records_path}")
    if event_store is not None:
        print(f"  events       : {event_store.path}")
    print()

    proposal_backend = None
    if args.llm_backend == "langchain":
        proposal_backend = LangChainProposalBackend(
            args.model,
            artifacts_dir=ROOT / "experiments" / "artifacts" / args.campaign_id,
        )

    if args.dry_run:
        dry_profile = _resolve_dry_run_profile(args.memory_mode, args.dry_run_profile)
        result = run_dry_campaign(
            node_spec=node_spec,
            campaign_id=args.campaign_id,
            budget=args.budget,
            manager_mode=args.manager,
            memory_mode=args.memory_mode,
            records_path=records_path,
            dry_run_profile=dry_profile,
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

    # Post-run validation checks (acceptance criterion).
    records = TrialAppendStore(records_path).read_all()
    _validate_smoke(records, args.campaign_id, args.memory_mode, records_path, args.budget)

    # Export per-campaign summary tables.
    if not args.no_export:
        summary = load_campaign_summary(records_path,
                                        metric_name=node_spec.metric_name,
                                        metric_direction=node_spec.metric_direction)
        outputs = export_campaign_tables(summary, args.tables_dir)
        print("\nExported tables:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")

    return 0


def _validate_smoke(records, campaign_id: str, memory_mode: str, ledger_path: Path, expected_count: int) -> None:
    """Print acceptance-criterion check results to stdout."""
    print("\nPost-run validation:")

    ok_ledger = ledger_path.exists()
    print(f"  {'✅' if ok_ledger else '❌'} Ledger exists at {ledger_path}")

    ok_count = len(records) == expected_count
    print(f"  {'✅' if ok_count else '❌'} Ledger has {len(records)}/{expected_count} record(s)")

    mode_match = all(r.memory_mode == memory_mode for r in records)
    print(f"  {'✅' if mode_match else '❌'} All records have memory_mode={memory_mode!r}")

    # Check no pending guard remains
    guard = ledger_path.parent / (ledger_path.stem.replace("_trials", "") + "_pending.json")
    # Guard file follows the campaign_id pattern
    guard_alt = ledger_path.parent / f"{campaign_id}_pending.json"
    has_guard = guard.exists() or guard_alt.exists()
    print(f"  {'✅' if not has_guard else '❌'} No pending guard remains")

    repeated_present = all(
        "repeated_bad_count" in (
            ((getattr(r, "extra", {}) or {}).get("manager", {}) or {})
            .get("worker_repeated_bad_stats", {})
        )
        for r in records
    )
    print(f"  {'✅' if repeated_present else '❌'} repeated_bad_count present in every record")

    if not (ok_ledger and ok_count and mode_match and not has_guard and repeated_present):
        print("\n  ⚠  One or more checks failed — fix before running Chunk 2.2.")
    else:
        print(f"\n  All checks passed for {campaign_id}.")


def _resolve_dry_run_profile(memory_mode: str, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    return {
        "none": "ablation_none",
        "append_only_summary": "ablation_append_only_summary",
        "append_only_summary_with_rationale": "ablation_append_only_summary_with_rationale",
    }[memory_mode]


def _default_events_path(records_path: Path, campaign_id: str) -> Path:
    root = records_path.parent.parent if records_path.parent.name == "ledgers" else records_path.parent
    return root / "events" / f"{campaign_id}_events.jsonl"


if __name__ == "__main__":
    raise SystemExit(main())
