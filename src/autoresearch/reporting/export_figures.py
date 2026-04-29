from __future__ import annotations

import csv
from pathlib import Path

from autoresearch.evaluation.metrics import compute_campaign_metrics
from autoresearch.memory.schemas import TrialDecision, TrialRecord
from autoresearch.memory.similarity import compute_repeated_bad_stats


def export_figure_csvs(records: list[TrialRecord], output_dir: str | Path, metric_name: str = "val_auc") -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "campaign_trajectory": target_dir / "campaign_trajectory.csv",
        "accepted_discarded_invalid_counts": target_dir / "accepted_discarded_invalid_counts.csv",
        "repeated_bad_idea_rates": target_dir / "repeated_bad_idea_rates.csv",
        "gain_per_budget_unit": target_dir / "gain_per_budget_unit.csv",
    }
    _write_trajectory(outputs["campaign_trajectory"], records, metric_name)
    _write_counts(outputs["accepted_discarded_invalid_counts"], records)
    _write_repeated(outputs["repeated_bad_idea_rates"], records)
    _write_gain(outputs["gain_per_budget_unit"], records, metric_name)
    return outputs


def _write_trajectory(path: Path, records: list[TrialRecord], metric_name: str) -> None:
    best: float | None = None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("budget_index", "trial_id", "metric", "running_best", "decision"))
        writer.writeheader()
        for record in records:
            metric = record.parsed_metrics.get(metric_name)
            if metric is not None:
                best = metric if best is None else max(best, metric)
            writer.writerow(
                {
                    "budget_index": record.budget_index,
                    "trial_id": record.trial_id,
                    "metric": metric,
                    "running_best": best,
                    "decision": record.decision.value,
                }
            )


def _write_counts(path: Path, records: list[TrialRecord]) -> None:
    counts = {
        "kept": sum(1 for record in records if record.decision == TrialDecision.KEPT),
        "discarded": sum(1 for record in records if record.decision == TrialDecision.DISCARDED),
        "failed_invalid": sum(1 for record in records if record.decision == TrialDecision.FAILED_INVALID),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("decision", "count"))
        writer.writeheader()
        for decision, count in counts.items():
            writer.writerow({"decision": decision, "count": count})


def _write_repeated(path: Path, records: list[TrialRecord]) -> None:
    stats = compute_repeated_bad_stats(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(stats.to_dict().keys()))
        writer.writeheader()
        row = stats.to_dict()
        row["flagged_trial_ids"] = ",".join(stats.flagged_trial_ids)
        writer.writerow(row)


def _write_gain(path: Path, records: list[TrialRecord], metric_name: str) -> None:
    metrics = compute_campaign_metrics(records, metric_name=metric_name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("campaign_id", "gain_per_budget_unit", "total_trials"))
        writer.writeheader()
        writer.writerow(
            {
                "campaign_id": metrics.campaign_id,
                "gain_per_budget_unit": metrics.gain_per_budget_unit,
                "total_trials": metrics.total_trials,
            }
        )

