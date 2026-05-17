#!/usr/bin/env python3
"""Export metric-validation summaries for governed campaign ledgers."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import NodeRegistryError, load_registered_node


def _ledger_path(campaign_id: str) -> Path:
    return ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"


def _provenance_complete(record) -> bool:
    return bool(
        record.provenance
        and record.provenance.proposal_id
        and record.provenance.patch_id
        and record.provenance.run_id
        and record.provenance.metric_id
        and record.provenance.decision_id
    )


def _summarize(campaign_id: str) -> dict[str, str]:
    records = TrialAppendStore(_ledger_path(campaign_id)).read_all()
    decisions = Counter(record.decision.value for record in records)
    failures = Counter(
        record.failure_category.value
        for record in records
        if record.failure_category is not None
    )
    trials = len(records)
    valid = decisions[TrialDecision.KEPT.value] + decisions[TrialDecision.DISCARDED.value]
    repeated_bad = records[-1].repeated_bad_count if records else 0
    metric_name = ""
    if records:
        try:
            metric_name = load_registered_node(records[0].node_id, repo_root=ROOT).metric_name
        except NodeRegistryError:
            metric_name = ""
    if not metric_name:
        metric_name = next((next(iter(record.parsed_metrics)) for record in records if record.parsed_metrics), "")
    best_metric = ""
    if metric_name:
        values = [record.parsed_metrics[metric_name] for record in records if metric_name in record.parsed_metrics]
        if values:
            best_metric = f"{max(values):.9f}"
    return {
        "campaign_id": campaign_id,
        "node_id": records[0].node_id if records else "",
        "manager_mode": records[0].manager_mode if records else "",
        "memory_mode": records[0].memory_mode if records else "",
        "trials": str(trials),
        "kept": str(decisions[TrialDecision.KEPT.value]),
        "discarded": str(decisions[TrialDecision.DISCARDED.value]),
        "failed_invalid": str(decisions[TrialDecision.FAILED_INVALID.value]),
        "valid_fraction": f"{(valid / trials) if trials else 0.0:.6f}",
        "invalid_action_rate": f"{(decisions[TrialDecision.FAILED_INVALID.value] / trials) if trials else 0.0:.6f}",
        "repeated_bad_rate": f"{(repeated_bad / trials) if trials else 0.0:.6f}",
        "provenance_completeness": f"{(sum(_provenance_complete(record) for record in records) / trials) if trials else 0.0:.6f}",
        "metric_name": metric_name,
        "best_metric": best_metric,
        "failure_categories": "; ".join(f"{key}={value}" for key, value in sorted(failures.items())),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["campaign_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pair_summary(label: str, left_id: str, right_id: str) -> dict[str, str]:
    left_records = TrialAppendStore(_ledger_path(left_id)).read_all()
    right_records = TrialAppendStore(_ledger_path(right_id)).read_all()
    n = min(len(left_records), len(right_records))
    decision_matches = 0
    metric_deltas: list[float] = []
    for left, right in zip(left_records[:n], right_records[:n]):
        if left.decision == right.decision:
            decision_matches += 1
        shared = set(left.parsed_metrics) & set(right.parsed_metrics)
        for metric in shared:
            metric_deltas.append(abs(left.parsed_metrics[metric] - right.parsed_metrics[metric]))
            break
    return {
        "label": label,
        "left_campaign_id": left_id,
        "right_campaign_id": right_id,
        "paired_trials": str(n),
        "decision_agreement": f"{(decision_matches / n) if n else 0.0:.6f}",
        "mean_abs_metric_delta": f"{mean(metric_deltas) if metric_deltas else 0.0:.9f}",
        "max_abs_metric_delta": f"{max(metric_deltas) if metric_deltas else 0.0:.9f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional reliability pair as label:left_campaign:right_campaign.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "plan" / "metric_validation" / "governance_metric_summary.csv"),
    )
    parser.add_argument(
        "--pairs-out",
        default=str(ROOT / "plan" / "metric_validation" / "governance_metric_reliability_pairs.csv"),
    )
    args = parser.parse_args()

    rows = [_summarize(campaign_id) for campaign_id in args.campaign]
    _write_csv(Path(args.out), rows)
    print(f"wrote {len(rows)} campaign summaries to {args.out}")
    if args.pair:
        pair_rows = []
        for raw_pair in args.pair:
            parts = raw_pair.split(":")
            if len(parts) != 3:
                raise SystemExit(f"invalid --pair value {raw_pair!r}; expected label:left:right")
            pair_rows.append(_pair_summary(parts[0], parts[1], parts[2]))
        _write_csv(Path(args.pairs_out), pair_rows)
        print(f"wrote {len(pair_rows)} reliability pair summaries to {args.pairs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
