#!/usr/bin/env python3
"""Export OpenML campaign summaries for the A-Governed paper."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node


PAPER_DIR = ROOT / "A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation"


def _row(campaign_id: str) -> dict[str, object]:
    ledger = ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"
    records = TrialAppendStore(ledger).read_all()
    if not records:
        raise FileNotFoundError(f"no records found for {campaign_id}: {ledger}")

    node_name = records[0].node_id
    spec = load_registered_node(node_name, repo_root=ROOT)
    decisions = [record.decision for record in records]
    kept = decisions.count(TrialDecision.KEPT)
    discarded = decisions.count(TrialDecision.DISCARDED)
    failed = decisions.count(TrialDecision.FAILED_INVALID)
    best = max((record.parsed_metrics.get(spec.metric_name, 0.0) for record in records), default=0.0)
    provenance = sum(
        1
        for record in records
        if record.provenance
        and record.provenance.proposal_id
        and record.provenance.patch_id
        and record.provenance.run_id
        and record.provenance.metric_id
        and record.provenance.decision_id
    ) / len(records)
    dataset = {
        "openml_credit_g": "credit-g",
        "openml_bank_marketing": "bank-marketing",
    }.get(node_name, node_name)

    return {
        "campaign_id": campaign_id,
        "node": node_name,
        "dataset": dataset,
        "trials": len(records),
        "kept": kept,
        "discarded": discarded,
        "failed_invalid": failed,
        "provenance": f"{provenance:.2f}",
        "metric_name": spec.metric_name,
        "best_metric": f"{best:.6f}",
    }


def _write_tex(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Node & Dataset & Trials & Kept & Discarded & Failed-invalid & Provenance & Best AUC \\",
        r"\midrule",
    ]
    for row in rows:
        node = str(row["node"]).replace("_", r"\_")
        dataset = str(row["dataset"]).replace("_", r"\_")
        lines.append(
            f"{node} & {dataset} & {row['trials']} & {row['kept']} & "
            f"{row['discarded']} & {row['failed_invalid']} & {row['provenance']} & "
            f"{row['best_metric']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenML campaign table for the paper.")
    parser.add_argument(
        "--campaign-id",
        action="append",
        dest="campaign_ids",
        help="Campaign id to include. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PAPER_DIR / "tables"),
        help="Output table directory.",
    )
    args = parser.parse_args()

    campaign_ids = args.campaign_ids or [
        "openml_credit_g_main_20",
        "openml_bank_marketing_main_20",
    ]
    rows = [_row(campaign_id) for campaign_id in campaign_ids]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "openml_campaign_summary.csv"
    tex_path = out_dir / "openml_campaign_summary.tex"

    fieldnames = (
        "campaign_id",
        "node",
        "dataset",
        "trials",
        "kept",
        "discarded",
        "failed_invalid",
        "provenance",
        "metric_name",
        "best_metric",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_tex(rows, tex_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
