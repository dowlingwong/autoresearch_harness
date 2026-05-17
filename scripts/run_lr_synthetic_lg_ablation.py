#!/usr/bin/env python3
"""LangGraph memory ablation on lr_synthetic node (LocalWorker, Ollama-backed manager).

Runs three arms (none / append_only_summary / append_only_summary_with_rationale)
with LangGraphManager + LocalWorker.  No coding agent required — LocalWorker applies
structured 'Change PARAM from X to Y' directives directly.

This is the cross-node validation for the memory ablation: same governance protocol,
different ML task, different worker type.

Usage
-----
# Dry-smoke (stub LLM, no Ollama, no node execution):
    python3 scripts/run_lr_synthetic_lg_ablation.py --budget 2 --llm-stub

# Real run with local Ollama manager:
    python3 scripts/run_lr_synthetic_lg_ablation.py --budget 10

# Single arm:
    python3 scripts/run_lr_synthetic_lg_ablation.py --budget 10 --arm summary

# With a non-default model or host:
    python3 scripts/run_lr_synthetic_lg_ablation.py \\
        --budget 10 --model qwen2.5-coder:7b --host http://localhost:11434

# With a DeepSeek API manager (LocalWorker still runs locally):
    DEEPSEEK_API_KEY=... DEEPSEEK_THINKING=disabled \\
    python3 scripts/run_lr_synthetic_lg_ablation.py \\
        --budget 10 --model deepseek/deepseek-v4-flash

Reset before re-running (per arm):
    python3 scripts/reset_node_state.py --node lr_synthetic \\
        --campaign-id lr_synth_lg_none
    python3 scripts/reset_node_state.py --node lr_synthetic \\
        --campaign-id lr_synth_lg_summary
    python3 scripts/reset_node_state.py --node lr_synthetic \\
        --campaign-id lr_synth_lg_rationale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.campaign import run_real_campaign, run_dry_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

MEMORY_MODES = {
    "none":     "none",
    "summary":  "append_only_summary",
    "rationale":"append_only_summary_with_rationale",
}

CAMPAIGN_IDS = {
    "none":     "lr_synth_lg_none",
    "summary":  "lr_synth_lg_summary",
    "rationale":"lr_synth_lg_rationale",
}


def _run_arm(
    arm_key: str,
    budget: int,
    node_spec,
    node_root: Path,
    ledger_dir: Path,
    artifacts_dir: Path,
    model: str,
    host: str,
    manager_llm=None,
    temperature: float = 0.7,
) -> None:
    memory_mode = MEMORY_MODES[arm_key]
    campaign_id = CAMPAIGN_IDS[arm_key]
    records_path = ledger_dir / f"{campaign_id}_trials.jsonl"

    print(f"\n{'='*60}")
    print(f"  ARM: {arm_key}  ({memory_mode})")
    print(f"  campaign_id : {campaign_id}")
    print(f"  budget      : {budget}")
    print(f"  ledger      : {records_path}")
    print(f"{'='*60}")

    worker = LocalWorker(
        node_root=node_root,
        artifact_dir=artifacts_dir / campaign_id,
        timeout_seconds=120.0,
    )

    events_path = ROOT / "experiments" / "events" / f"{campaign_id}_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    from autoresearch.memory.event_store import CampaignEventStore
    event_store = CampaignEventStore(events_path)

    from autoresearch.manager.langgraph_manager import LangGraphManager
    if manager_llm is not None:
        # stub mode — inject the fake LLM directly
        manager = LangGraphManager(llm=manager_llm, temperature=temperature)
    else:
        # real mode — pass model/host so LangGraphManager builds its own ChatOllama
        manager = LangGraphManager(model=model, host=host, temperature=temperature)

    run_real_campaign(
        node_spec=node_spec,
        campaign_id=campaign_id,
        budget=budget,
        manager_mode="langgraph_manager",
        memory_mode=memory_mode,
        records_path=records_path,
        worker=worker,
        proposal_backend=manager,
        manager_temperature=temperature,
        event_store=event_store,
    )

    # --- per-arm summary ---
    store = TrialAppendStore(records_path)
    records = store.read_all()
    decisions = [r.decision for r in records]
    kept  = decisions.count(TrialDecision.KEPT)
    disc  = decisions.count(TrialDecision.DISCARDED)
    fail  = decisions.count(TrialDecision.FAILED_INVALID)
    best  = max((r.parsed_metrics.get("val_score", 0.0) for r in records
                 if r.parsed_metrics), default=0.0)
    rbr_count = sum(
        1 for r in records
        if r.failure_category is not None
        and r.budget_index > 0
    )

    print(f"\n  Arm {arm_key}: kept={kept} disc={disc} fail={fail} "
          f"best_val_score={best:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph memory ablation on lr_synthetic (LocalWorker).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--budget", type=int, default=10,
                        help="Trials per arm (default 10).")
    parser.add_argument("--arm", choices=list(MEMORY_MODES), default=None,
                        help="Run a single arm instead of all three.")
    parser.add_argument("--model", default="qwen2.5-coder:7b",
                        help="Manager model for LangGraphManager, e.g. qwen2.5-coder:7b or deepseek/deepseek-v4-flash.")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host for local manager models; ignored by DeepSeek/OpenAI/Anthropic providers.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm-stub", action="store_true",
                        help="Inject a FakeListChatModel (no Ollama, for smoke testing).")
    parser.add_argument("--ledger-dir",
                        default=str(ROOT / "experiments" / "ledgers"))
    parser.add_argument("--artifacts-dir",
                        default=str(ROOT / "experiments" / "artifacts"))
    args = parser.parse_args()

    node_spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    node_root = ROOT / "nodes" / "lr_synthetic"
    ledger_dir = Path(args.ledger_dir)
    artifacts_dir = Path(args.artifacts_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    if not node_root.exists():
        print(f"ERROR: node_root not found: {node_root}", file=sys.stderr)
        return 1

    manager_llm = None
    if args.llm_stub:
        print("Using FakeListChatModel stub (no Ollama calls).")
        from langchain_core.language_models import FakeListChatModel
        stub_response = (
            '{"param": "LEARNING_RATE", "old_value": "0.01", '
            '"new_value": "0.05", "rationale": "Increase LR to explore faster."}'
        )
        manager_llm = FakeListChatModel(
            responses=[stub_response] * max(args.budget * 3, 6)
        )

    arms_to_run = [args.arm] if args.arm else list(MEMORY_MODES)

    print(f"\nlr_synthetic LangGraph Memory Ablation")
    print(f"  arms   : {arms_to_run}")
    print(f"  budget : {args.budget} trials/arm")
    print(f"  model  : {args.model}  host: {args.host}")
    print(f"  stub   : {args.llm_stub}")

    for arm_key in arms_to_run:
        # Reset node to baseline before each arm so edits don't accumulate.
        import subprocess as _sp
        campaign_id = CAMPAIGN_IDS[arm_key]
        print(f"\nResetting node state for arm '{arm_key}' (campaign {campaign_id}) ...")
        _sp.run(
            [sys.executable, str(ROOT / "scripts" / "reset_node_state.py"),
             "--node", "lr_synthetic", "--campaign-id", campaign_id],
            check=True,
        )
        _run_arm(
            arm_key=arm_key,
            budget=args.budget,
            node_spec=node_spec,
            node_root=node_root,
            ledger_dir=ledger_dir,
            artifacts_dir=artifacts_dir,
            model=args.model,
            host=args.host,
            manager_llm=manager_llm,
            temperature=args.temperature,
        )

    print("\n\nAll arms complete.")
    print("Next: run scripts/analyze_lr_synthetic_lg_ablation.py to compute RBR and update paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
