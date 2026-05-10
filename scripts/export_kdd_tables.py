#!/usr/bin/env python3
"""Export paper-ready CSV tables for the KDD AAE 2026 submission.

Supports three modes:

1. Single campaign (Chunk 2.2 post-run):
   python3 scripts/export_kdd_tables.py --campaign-id kdd_main_5trial

2. Memory ablation summary (Chunk 2.4 post-run):
   python3 scripts/export_kdd_tables.py --experiment memory_ablation

3. Full export (Chunk 3.1 — after all experiments):
   python3 scripts/export_kdd_tables.py \\
       --main-campaign kdd_main_5trial \\
       --ablation-campaigns ablation_none ablation_append_only_summary \\
                            ablation_append_only_summary_with_rationale \\
       --stress-campaign kdd_stress_scope \\
       --output-dir paper/tables/

Output files:
  main_campaign_summary.csv          — Table 1: lifecycle + AUC
  governance_metrics.csv             — governance columns
  memory_ablation_summary.csv        — Table 3: repeated_bad_rate per mode
  failure_taxonomy.csv               — Table 2: failure category counts
  provenance_chain.csv               — per-trial provenance completeness
  accepted_discarded_invalid_counts.csv — counts per campaign
  campaign_trajectory.csv            — trial-by-trial metric for Figure 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision
from autoresearch.memory.similarity import compute_repeated_bad_stats
from autoresearch.nodes.registry import load_registered_node
from autoresearch.reporting.export_tables import export_campaign_tables

ABLATION_CAMPAIGN_IDS = (
    "ablation_none",
    "ablation_append_only_summary",
    "ablation_append_only_summary_with_rationale",
)
ABLATION_MEMORY_MODES = (
    "none",
    "append_only_summary",
    "append_only_summary_with_rationale",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export KDD AAE paper tables (single campaign, ablation, or full).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--node", default="resnet_trigger",
                        help="Node name (for metric info).")
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "tables"),
                        help="Directory to write all CSVs.")
    parser.add_argument("--ledger-dir", default=str(ROOT / "experiments" / "ledgers"),
                        help="Directory containing *_trials.jsonl files.")

    # Mode A: single campaign
    parser.add_argument("--campaign-id", default=None,
                        help="Export tables for one campaign. Writes main_campaign_summary.csv "
                             "and governance_metrics.csv.")

    # Mode B: ablation experiment
    parser.add_argument("--experiment", choices=("memory_ablation",), default=None,
                        help="Export a named experiment summary (memory_ablation).")

    # Mode C: full paper export
    parser.add_argument("--main-campaign", default=None,
                        help="Main campaign ID for full export.")
    parser.add_argument("--ablation-campaigns", nargs="+", default=None,
                        help="Ablation campaign IDs (space-separated) for full export.")
    parser.add_argument("--stress-campaign", nargs="+", default=None,
                        help="Stress trial campaign ID(s) for full export.")

    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = Path(args.ledger_dir)

    outputs: dict[str, Path] = {}

    # Mode A: single campaign
    if args.campaign_id:
        ledger = ledger_dir / f"{args.campaign_id}_trials.jsonl"
        if not ledger.exists():
            print(f"[error] ledger not found: {ledger}", file=sys.stderr)
            return 1
        summary = load_campaign_summary(ledger, metric_name=node_spec.metric_name,
                                        metric_direction=node_spec.metric_direction)
        outputs.update(export_campaign_tables(summary, out_dir))
        # Trajectory CSV
        traj = _export_trajectory(ledger, out_dir, node_spec.metric_name)
        if traj:
            outputs["campaign_trajectory"] = traj
        # Failure taxonomy
        ftax = _export_failure_taxonomy([args.campaign_id], ledger_dir, out_dir)
        outputs["failure_taxonomy"] = ftax
        # Decision counts
        dc = _export_decision_counts([args.campaign_id], ledger_dir, out_dir)
        outputs["accepted_discarded_invalid_counts"] = dc
        prov = _export_provenance_chain([args.campaign_id], ledger_dir, out_dir)
        outputs["provenance_chain"] = prov

    # Mode B: memory ablation
    if args.experiment == "memory_ablation":
        campaign_ids = list(args.ablation_campaigns or ABLATION_CAMPAIGN_IDS)
        abl = _export_memory_ablation_summary(campaign_ids, ABLATION_MEMORY_MODES, ledger_dir, out_dir, node_spec)
        outputs["memory_ablation_summary"] = abl
        dc = _export_decision_counts(campaign_ids, ledger_dir, out_dir)
        outputs["accepted_discarded_invalid_counts"] = dc
        ftax = _export_failure_taxonomy(campaign_ids, ledger_dir, out_dir)
        outputs["failure_taxonomy"] = ftax
        prov = _export_provenance_chain(campaign_ids, ledger_dir, out_dir)
        outputs["provenance_chain"] = prov

    # Mode C: full export
    if args.main_campaign or args.ablation_campaigns or args.stress_campaign:
        all_campaign_ids: list[str] = []
        if args.main_campaign:
            ledger = ledger_dir / f"{args.main_campaign}_trials.jsonl"
            if ledger.exists():
                summary = load_campaign_summary(ledger, metric_name=node_spec.metric_name,
                                                metric_direction=node_spec.metric_direction)
                outputs.update(export_campaign_tables(summary, out_dir))
                traj = _export_trajectory(ledger, out_dir, node_spec.metric_name)
                if traj:
                    outputs["campaign_trajectory"] = traj
                all_campaign_ids.append(args.main_campaign)

        ablation_ids = list(args.ablation_campaigns or [])
        if ablation_ids:
            abl = _export_memory_ablation_summary(
                ablation_ids, ABLATION_MEMORY_MODES[:len(ablation_ids)],
                ledger_dir, out_dir, node_spec
            )
            outputs["memory_ablation_summary"] = abl
            all_campaign_ids.extend(ablation_ids)

        if args.stress_campaign:
            all_campaign_ids.extend(args.stress_campaign)

        if all_campaign_ids:
            dc = _export_decision_counts(all_campaign_ids, ledger_dir, out_dir)
            outputs["accepted_discarded_invalid_counts"] = dc
            ftax = _export_failure_taxonomy(all_campaign_ids, ledger_dir, out_dir)
            outputs["failure_taxonomy"] = ftax
            prov = _export_provenance_chain(all_campaign_ids, ledger_dir, out_dir)
            outputs["provenance_chain"] = prov

    if not outputs:
        parser.error(
            "Specify one of: --campaign-id, --experiment, or "
            "--main-campaign/--ablation-campaigns/--stress-campaign"
        )

    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2, sort_keys=True))
    _verify_outputs(outputs)
    return 0


# ---------------------------------------------------------------------------
# Per-table exporters
# ---------------------------------------------------------------------------

def _export_trajectory(ledger_path: Path, out_dir: Path, metric_name: str) -> Path | None:
    """Write trial-by-trial metric trajectory CSV (Figure 4 source data)."""
    records = TrialAppendStore(ledger_path).read_all()
    if not records:
        return None
    path = out_dir / "campaign_trajectory.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "budget_index", "trial_id", "decision", "metric_value",
            "current_best_before", "delta_vs_best", "memory_mode",
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "budget_index": r.budget_index,
                "trial_id": r.trial_id,
                "decision": r.decision,
                "metric_value": r.parsed_metrics.get(metric_name, ""),
                "current_best_before": r.current_best_before if r.current_best_before is not None else "",
                "delta_vs_best": r.delta_vs_best if r.delta_vs_best is not None else "",
                "memory_mode": r.memory_mode,
            })
    return path


def _export_failure_taxonomy(
    campaign_ids: list[str],
    ledger_dir: Path,
    out_dir: Path,
) -> Path:
    """Write failure category counts across all campaigns."""
    rows: list[dict] = []
    for cid in campaign_ids:
        ledger = ledger_dir / f"{cid}_trials.jsonl"
        if not ledger.exists():
            continue
        records = TrialAppendStore(ledger).read_all()
        category_counts: dict[str, int] = {c.value: 0 for c in FailureCategory}
        for r in records:
            if r.failure_category is not None:
                category_counts[r.failure_category.value] += 1
        total = len(records)
        for cat, count in category_counts.items():
            if count > 0:
                rows.append({
                    "campaign_id": cid,
                    "failure_category": cat,
                    "count": count,
                    "rate": round(count / total, 4) if total else 0.0,
                    "total_trials": total,
                })

    path = out_dir / "failure_taxonomy.csv"
    fieldnames = ["campaign_id", "failure_category", "count", "rate", "total_trials"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _export_decision_counts(
    campaign_ids: list[str],
    ledger_dir: Path,
    out_dir: Path,
) -> Path:
    """Write kept/discarded/failed_invalid counts per campaign."""
    rows: list[dict] = []
    for cid in campaign_ids:
        ledger = ledger_dir / f"{cid}_trials.jsonl"
        if not ledger.exists():
            continue
        records = TrialAppendStore(ledger).read_all()
        kept = sum(1 for r in records if r.decision == TrialDecision.KEPT)
        discarded = sum(1 for r in records if r.decision == TrialDecision.DISCARDED)
        failed = sum(1 for r in records if r.decision == TrialDecision.FAILED_INVALID)
        total = len(records)
        rows.append({
            "campaign_id": cid,
            "total_trials": total,
            "kept": kept,
            "discarded": discarded,
            "failed_invalid": failed,
            "acceptance_rate": round(kept / total, 4) if total else 0.0,
            "invalid_rate": round(failed / total, 4) if total else 0.0,
        })

    path = out_dir / "accepted_discarded_invalid_counts.csv"
    fieldnames = ["campaign_id", "total_trials", "kept", "discarded",
                  "failed_invalid", "acceptance_rate", "invalid_rate"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _export_memory_ablation_summary(
    campaign_ids: list[str],
    memory_modes: tuple[str, ...],
    ledger_dir: Path,
    out_dir: Path,
    node_spec,
) -> Path:
    """Write per-mode repeated-bad-rate summary CSV (Table 3)."""
    rows: list[dict] = []
    for cid, mode in zip(campaign_ids, memory_modes):
        ledger = ledger_dir / f"{cid}_trials.jsonl"
        if not ledger.exists():
            print(f"  [warn] ledger not found for {cid}, skipping", file=sys.stderr)
            continue
        records = TrialAppendStore(ledger).read_all()
        stats = compute_repeated_bad_stats(records)
        total = len(records)
        kept = sum(1 for r in records if r.decision == TrialDecision.KEPT)
        discarded = sum(1 for r in records if r.decision == TrialDecision.DISCARDED)
        failed = sum(1 for r in records if r.decision == TrialDecision.FAILED_INVALID)
        metrics_values = [r.parsed_metrics.get(node_spec.metric_name)
                          for r in records if node_spec.metric_name in r.parsed_metrics]
        best_metric = max(metrics_values) if metrics_values else None
        rows.append({
            "campaign_id": cid,
            "memory_mode": mode,
            "total_trials": total,
            "kept": kept,
            "discarded": discarded,
            "failed_invalid": failed,
            "acceptance_rate": round(kept / total, 4) if total else 0.0,
            "repeated_bad_count": stats.repeated_bad_count,
            "repeated_bad_rate": round(stats.repeated_bad_rate, 4),
            "repeated_invalid_count": stats.repeated_invalid_count,
            "repeated_degraded_count": stats.repeated_degraded_count,
            "best_metric": best_metric if best_metric is not None else "",
        })

    path = out_dir / "memory_ablation_summary.csv"
    fieldnames = [
        "campaign_id", "memory_mode", "total_trials",
        "kept", "discarded", "failed_invalid", "acceptance_rate",
        "repeated_bad_count", "repeated_bad_rate",
        "repeated_invalid_count", "repeated_degraded_count",
        "best_metric",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _export_provenance_chain(
    campaign_ids: list[str],
    ledger_dir: Path,
    out_dir: Path,
) -> Path:
    """Write one provenance-completeness row per trial."""
    rows: list[dict] = []
    for cid in campaign_ids:
        ledger = ledger_dir / f"{cid}_trials.jsonl"
        if not ledger.exists():
            continue
        records = TrialAppendStore(ledger).read_all()
        for record in records:
            provenance = record.provenance
            ids = {
                "proposal_id": provenance.proposal_id,
                "patch_id": provenance.patch_id,
                "run_id": provenance.run_id,
                "metric_id": provenance.metric_id,
                "decision_id": provenance.decision_id,
            }
            present = sum(1 for value in ids.values() if value)
            rows.append({
                "campaign_id": cid,
                "trial_id": record.trial_id,
                "budget_index": record.budget_index,
                "decision": record.decision.value,
                **ids,
                "complete": present == len(ids),
                "completeness_pct": round(100.0 * present / len(ids), 2),
            })

    path = out_dir / "provenance_chain.csv"
    fieldnames = [
        "campaign_id",
        "trial_id",
        "budget_index",
        "decision",
        "proposal_id",
        "patch_id",
        "run_id",
        "metric_id",
        "decision_id",
        "complete",
        "completeness_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _verify_outputs(outputs: dict[str, Path]) -> None:
    """Print a quick existence check for every output file."""
    print("\nFile verification:")
    all_ok = True
    for key, path in sorted(outputs.items()):
        ok = path.exists() and path.stat().st_size > 0
        if not ok:
            all_ok = False
        print(f"  {'✅' if ok else '❌'} {key}: {path}")
    if all_ok:
        print("\nAll table files written successfully.")
    else:
        print("\n⚠  Some files missing or empty.")


if __name__ == "__main__":
    raise SystemExit(main())
