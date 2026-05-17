#!/usr/bin/env python3
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
from autoresearch.memory.similarity import compare_repetition_detectors


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare repeated-bad detector variants on campaign ledgers.")
    parser.add_argument("ledgers", nargs="+", help="Trial JSONL ledgers to compare.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper" / "tables" / "similarity_detector_comparison.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for ledger in args.ledgers:
        path = Path(ledger)
        records = TrialAppendStore(path).read_all()
        for result in compare_repetition_detectors(records, text_threshold=args.threshold):
            row = result.to_dict()
            row["ledger"] = str(path)
            row["total_trials"] = len(records)
            row["flagged_trial_ids"] = ",".join(result.flagged_trial_ids)
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ledger",
        "method",
        "threshold",
        "total_trials",
        "repeated_bad_count",
        "repeated_bad_rate",
        "repeated_invalid_count",
        "repeated_degraded_count",
        "flagged_trial_ids",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
