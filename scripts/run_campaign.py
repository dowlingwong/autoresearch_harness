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

# Add the project .venv site-packages so that langgraph / langchain-core are
# importable when this script is run with the system Python instead of
# .venv/bin/python.  This is a no-op when the venv is already active.
_VENV_SITE = ROOT / ".venv" / "lib"
if _VENV_SITE.exists():
    for _p in sorted(_VENV_SITE.iterdir()):
        _site = _p / "site-packages"
        if _site.exists() and str(_site) not in sys.path:
            sys.path.insert(0, str(_site))

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.nodes.registry import load_registered_node
from autoresearch.reporting.export_tables import export_campaign_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Stage 2 fixed-budget campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dry-run (no worker, mock metrics):
  python3 scripts/run_campaign.py \\
      --node resnet_trigger --campaign-id smoke --budget 3 --dry-run

Real campaign (ClawWorker drives the legacy loop with proposal-generated packets):
  python3 scripts/run_campaign.py \\
      --node resnet_trigger --campaign-id main \\
      --budget 15 --manager prompt_manager \\
      --memory-mode append_only_summary_with_rationale \\
      --node-root nodes/ResNet_trigger \\
      --model qwen2.5-coder:7b --host http://localhost:11434

Optionally supply a packet-defaults file to override timeout/log_path/syntax_check:
  --packet-defaults tests/stage_1_sprint_deliverable/loop_packet.json
""",
    )
    parser.add_argument("--node", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--manager", default="baseline_manager")
    parser.add_argument("--memory-mode", default="none")
    parser.add_argument("--records", help="Path to JSONL ledger (default: experiments/ledgers/<campaign-id>_trials.jsonl)")
    parser.add_argument("--tables-dir", default=str(ROOT / "paper" / "tables"))
    parser.add_argument("--dry-run", action="store_true", help="Use mock DryRunWorker; write synthetic TrialRecords.")
    # Real-run args
    parser.add_argument("--node-root", help="Path to the node working directory (required for real runs)")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "experiments" / "artifacts"),
                        help="Where generated packets and trial artifacts are written")
    parser.add_argument("--packet-defaults",
                        help="Optional JSON file whose timeout_seconds/log_path/results_tsv/syntax_check_command "
                             "override node-spec defaults. objective and description are always from the proposal.")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--llm-stub", action="store_true",
        help="Inject a FakeListChatModel into langgraph_manager instead of calling Ollama. "
             "For smoke-testing the LangGraph pipeline without a running LLM.",
    )
    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    records_path = (
        Path(args.records)
        if args.records
        else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the optional stub LLM for langgraph_manager smoke tests
    manager_llm = None
    if args.llm_stub:
        if args.manager != "langgraph_manager":
            parser.error("--llm-stub is only applicable when --manager langgraph_manager")
        from langchain_core.language_models import FakeListChatModel
        import json as _json
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
            campaign_id=args.campaign_id,
            budget=args.budget,
            manager_mode=args.manager,
            memory_mode=args.memory_mode,
            records_path=records_path,
            manager_llm=manager_llm,
        )
    else:
        if not args.node_root:
            parser.error("--node-root is required for real (non-dry-run) campaigns")

        from autoresearch.worker.claw_worker import ClawWorker

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
            campaign_id=args.campaign_id,
            budget=args.budget,
            manager_mode=args.manager,
            memory_mode=args.memory_mode,
            records_path=records_path,
            worker=worker,
            manager_llm=manager_llm,
        )

    summary = load_campaign_summary(records_path)
    table_outputs = export_campaign_tables(summary, args.tables_dir)
    print(
        json.dumps(
            {
                "campaign": result.to_dict(),
                "metrics": summary.metrics.to_dict(),
                "tables": {key: str(path) for key, path in table_outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
