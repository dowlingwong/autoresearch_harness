#!/usr/bin/env python3
"""Run a matched governed vs. ungoverned counterfactual comparison.

Runs both arms back-to-back with identical settings and budgets, resets node
state between arms, then writes a summary JSON for analyze_counterfactual.py.

The ungoverned arm uses ungoverned=True in run_real_campaign(), which disables
exactly two governance mechanisms:
  1. Pending-trial guard  — NOT written/deleted; crashes are silent.
  2. Ledger append on failure — failed_invalid trials are NOT appended;
     from the ledger's perspective, those trials never happened.
A lightweight _ungoverned_obs.jsonl observation log IS written, recording every
trial that actually executed, so we can verify the trial count without relying
on the absent ledger entries.

────────────────────────────────────────────────────────────────────────────────
Quick start — openml_bank_marketing (LocalWorker, CPU-only, ~30 min for N=20):

  python3 scripts/run_counterfactual.py \\
      --node openml_bank_marketing \\
      --base kdd_cf_openml \\
      --budget 20 \\
      --node-root nodes/openml_bank_marketing \\
      --model deepseek/deepseek-v4-flash \\
      --host http://localhost:11434

Quick start — autoresearch_linux (ClawWorker, NVIDIA GPU, ~3 h for N=30):

  python3 scripts/run_counterfactual.py \\
      --node autoresearch_linux \\
      --base kdd_cf_arlinux \\
      --budget 30 \\
      --node-root /ceph/dwong/autoresearch_harness/nodes/autoresearch_linux \\
      --model deepseek/deepseek-v4-flash \\
      --host http://YOUR_HOST:11434 \\
      --use-claw-worker

────────────────────────────────────────────────────────────────────────────────
Output files (in experiments/ledgers/):
  {base}_gov_trials.jsonl         — governed ledger (all N records)
  {base}_ung_trials.jsonl         — ungoverned ledger (valid-only records)
  {base}_ung_ungoverned_obs.jsonl — observation log (all N entries)
  {base}_cf_summary.json          — summary for analyze_counterfactual.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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


# ── Node-state reset ──────────────────────────────────────────────────────────

def _reset_node_state(node_root: Path, editable_paths: tuple[str, ...]) -> None:
    """Reset every editable file to its clean baseline.

    Strategy:
      1. Copy from .autoresearch_baseline/<rel_path> if that file exists.
      2. Fall back to `git checkout -- <rel_path>` from node_root.
      3. Warn (but continue) if neither works.
    """
    baseline_dir = node_root / ".autoresearch_baseline"
    for rel_path in editable_paths:
        target = node_root / rel_path
        baseline_src = baseline_dir / rel_path
        if baseline_src.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_src, target)
            print(f"    reset  {rel_path}  ← .autoresearch_baseline/")
        else:
            result = subprocess.run(
                ["git", "checkout", "--", str(rel_path)],
                cwd=node_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"    reset  {rel_path}  ← git checkout")
            else:
                print(
                    f"    WARN   {rel_path}  — no baseline copy and git checkout failed: "
                    f"{result.stderr.strip() or 'no error message'}"
                )


# ── Worker / manager constructors ─────────────────────────────────────────────

def _build_worker(args, node_root: Path):
    if args.use_claw_worker:
        from autoresearch.worker.claw_worker import ClawWorker
        from autoresearch.llm.providers import resolve_worker_model_args
        worker_model_id = args.worker_model or "qwen2.5-coder:7b"
        worker_host_str = args.worker_host or args.host
        worker_model, worker_host = resolve_worker_model_args(worker_model_id, worker_host_str)
        return ClawWorker(
            repo_root=ROOT,
            node_root=str(node_root),
            artifacts_dir=str(ROOT / "experiments" / "artifacts"),
            model=worker_model,
            host=worker_host,
        )
    else:
        from autoresearch.worker.local_worker import LocalWorker
        return LocalWorker(node_root=str(node_root))


def _build_langgraph_manager(args):
    from autoresearch.manager.langgraph_manager import LangGraphManager
    return LangGraphManager(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
    )


# ── Ledger path helpers ───────────────────────────────────────────────────────

def _ledger_path(base: str, suffix: str) -> Path:
    return ROOT / "experiments" / "ledgers" / f"{base}_{suffix}_trials.jsonl"


def _obs_log_path(base: str) -> Path:
    # run_real_campaign names the obs log as: {campaign_id_stem}_ungoverned_obs.jsonl
    # campaign_id for ungoverned arm is f"{base}_ung"
    # ledger stem is f"{base}_ung_trials" → cid_stem = f"{base}_ung"
    # obs_path = f"{base}_ung_ungoverned_obs.jsonl"
    return ROOT / "experiments" / "ledgers" / f"{base}_ung_ungoverned_obs.jsonl"


# ── Per-arm summary ───────────────────────────────────────────────────────────

def _arm_summary(arm_label: str, ledger: Path, budget: int, obs_log: Path | None = None) -> dict:
    records = TrialAppendStore(ledger).read_all() if ledger.exists() else []
    ledger_n = len(records)

    obs_n = 0
    if obs_log and obs_log.exists():
        with open(obs_log, encoding="utf-8") as f:
            obs_n = sum(1 for line in f if line.strip())

    # "Trials that actually ran" — prefer observation log count (ground truth
    # for ungoverned arm), fall back to budget for governed arm.
    trials_ran = obs_n if obs_n > 0 else budget
    completeness = ledger_n / trials_ran if trials_ran > 0 else 0.0

    kept = sum(1 for r in records if r.decision == TrialDecision.KEPT)
    discarded = sum(1 for r in records if r.decision == TrialDecision.DISCARDED)
    failed_inv = sum(1 for r in records if r.decision == TrialDecision.FAILED_INVALID)

    print(f"\n  ── {arm_label.upper()} ──")
    print(f"     ledger records : {ledger_n:>4}  /  {trials_ran} trials ran")
    print(f"     completeness   : {completeness:.1%}")
    print(f"     kept / disc / fail-inv : {kept} / {discarded} / {failed_inv}")
    if obs_log and obs_log.exists():
        dropped = trials_ran - ledger_n
        print(f"     silent drops   : {dropped} trials ran but left no ledger record")

    return {
        "arm": arm_label,
        "budget": budget,
        "trials_ran": trials_ran,
        "ledger_n": ledger_n,
        "completeness": completeness,
        "kept": kept,
        "discarded": discarded,
        "failed_invalid": failed_inv,
        "obs_log_entries": obs_n,
        "silent_drops": trials_ran - ledger_n,
        "ledger_path": str(ledger),
        "obs_log_path": str(obs_log) if obs_log else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--node", required=True,
                    help="Registered node name (openml_bank_marketing | autoresearch_linux)")
    ap.add_argument("--base", required=True,
                    help="Campaign base name — arms get _gov / _ung suffix")
    ap.add_argument("--budget", type=int, required=True,
                    help="Trials per arm (recommend 20 for openml, 30 for autoresearch_linux)")
    ap.add_argument("--node-root", required=True,
                    help="Path to the node working directory")
    # LLM / proposal
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash",
                    help="Proposal manager model (default: deepseek/deepseek-v4-flash)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama-compatible LLM host")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--memory-mode", default="none",
                    choices=["none", "append_only_summary", "append_only_summary_with_rationale"],
                    help="Memory mode (default: none — matches autoresearch_linux none-arm baseline)")
    # Worker
    ap.add_argument("--use-claw-worker", action="store_true",
                    help="Use ClawWorker (required for autoresearch_linux); default is LocalWorker")
    ap.add_argument("--worker-model", default=None,
                    help="Worker model for ClawWorker (default: qwen2.5-coder:7b)")
    ap.add_argument("--worker-host", default=None,
                    help="Worker host for ClawWorker (default: same as --host)")
    # Control flow
    ap.add_argument("--governed-only", action="store_true",
                    help="Run only the governed arm")
    ap.add_argument("--ungoverned-only", action="store_true",
                    help="Run only the ungoverned arm")
    ap.add_argument("--skip-reset", action="store_true",
                    help="Skip node-state reset between arms (for debugging only)")
    args = ap.parse_args()

    if args.governed_only and args.ungoverned_only:
        print("ERROR: --governed-only and --ungoverned-only are mutually exclusive", file=sys.stderr)
        return 1

    node_root = Path(args.node_root).resolve()
    if not node_root.exists():
        print(f"ERROR: --node-root does not exist: {node_root}", file=sys.stderr)
        return 1

    node_spec = load_registered_node(args.node, repo_root=ROOT)

    gov_id = f"{args.base}_gov"
    ung_id = f"{args.base}_ung"
    gov_ledger = _ledger_path(args.base, "gov")
    ung_ledger = _ledger_path(args.base, "ung")
    obs_log = _obs_log_path(args.base)
    summary_out = ROOT / "experiments" / "ledgers" / f"{args.base}_cf_summary.json"

    gov_ledger.parent.mkdir(parents=True, exist_ok=True)

    worker = _build_worker(args, node_root)

    print(f"\n{'═' * 62}")
    print(f"  COUNTERFACTUAL COMPARISON  ·  {args.node}")
    print(f"  budget/arm={args.budget}  memory={args.memory_mode}  model={args.model}")
    print(f"{'═' * 62}")

    # ── Governed arm ──────────────────────────────────────────────────────────
    if not args.ungoverned_only:
        print(f"\n[1/2] Resetting node state → governed arm")
        if not args.skip_reset:
            _reset_node_state(node_root, node_spec.editable_paths)
        print(f"[1/2] Running governed arm  ({args.budget} trials, ungoverned=False) …")
        run_real_campaign(
            node_spec=node_spec,
            campaign_id=gov_id,
            budget=args.budget,
            manager_mode="langgraph_manager",
            memory_mode=args.memory_mode,
            records_path=gov_ledger,
            worker=worker,
            proposal_backend=_build_langgraph_manager(args),
            ungoverned=False,
        )
        print(f"[1/2] Done.  Ledger → {gov_ledger.name}")

    # ── Ungoverned arm ────────────────────────────────────────────────────────
    if not args.governed_only:
        print(f"\n[2/2] Resetting node state → ungoverned arm")
        if not args.skip_reset:
            _reset_node_state(node_root, node_spec.editable_paths)
        print(f"[2/2] Running ungoverned arm ({args.budget} trials, ungoverned=True) …")
        run_real_campaign(
            node_spec=node_spec,
            campaign_id=ung_id,
            budget=args.budget,
            manager_mode="langgraph_manager",
            memory_mode=args.memory_mode,
            records_path=ung_ledger,
            worker=worker,
            proposal_backend=_build_langgraph_manager(args),
            ungoverned=True,
        )
        print(f"[2/2] Done.  Ledger → {ung_ledger.name}")
        print(f"             Obs log → {obs_log.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  RESULTS — {args.node}")
    print(f"{'═' * 62}")

    arm_results: dict[str, dict] = {}
    if not args.ungoverned_only:
        arm_results["governed"] = _arm_summary("governed", gov_ledger, args.budget)
    if not args.governed_only:
        arm_results["ungoverned"] = _arm_summary("ungoverned", ung_ledger, args.budget, obs_log)

    meta = {
        "node": args.node,
        "base": args.base,
        "budget": args.budget,
        "memory_mode": args.memory_mode,
        "model": args.model,
        "use_claw_worker": args.use_claw_worker,
        "gov_ledger": str(gov_ledger),
        "ung_ledger": str(ung_ledger),
        "obs_log": str(obs_log),
        "arm_results": arm_results,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Summary JSON → {summary_out}")
    print(f"\nNext: python3 scripts/analyze_counterfactual.py \\")
    print(f"          --summary {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
