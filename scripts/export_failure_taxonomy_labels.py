#!/usr/bin/env python3
"""Export failed-invalid trials for blind human taxonomy labeling.

The output CSV can be duplicated for two raters. Keep ``gold_failure_category``
hidden from raters if the goal is a blind Cohen's kappa estimate.
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

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision


def _ledger_path(campaign_id: str) -> Path:
    return ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, help="Campaign id to sample.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "plan" / "metric_validation" / "failure_taxonomy_sample.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for campaign_id in args.campaign:
        for record in TrialAppendStore(_ledger_path(campaign_id)).read_all():
            if record.decision != TrialDecision.FAILED_INVALID:
                continue
            rows.append(
                {
                    "sample_id": f"S{len(rows) + 1:03d}",
                    "campaign_id": record.campaign_id,
                    "trial_id": record.trial_id,
                    "node_id": record.node_id,
                    "manager_mode": record.manager_mode,
                    "proposal_summary": record.proposal_summary,
                    "proposal_rationale": record.proposal_rationale,
                    "decision_rationale": record.decision_rationale,
                    "worker_failure_message": str(record.extra.get("worker_failure_message") or ""),
                    "gold_failure_category": (
                        record.failure_category.value if record.failure_category else ""
                    ),
                    "rater_label": "",
                    "notes": "",
                }
            )
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "campaign_id",
        "trial_id",
        "node_id",
        "manager_mode",
        "proposal_summary",
        "proposal_rationale",
        "decision_rationale",
        "worker_failure_message",
    ]
    if args.include_gold:
        fieldnames.append("gold_failure_category")
    fieldnames.extend(["rater_label", "notes"])

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"wrote {len(rows)} samples to {output}")
    if not args.include_gold:
        print("gold labels omitted for blind rating; rerun with --include-gold for audit/debug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
