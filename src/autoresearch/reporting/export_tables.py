from __future__ import annotations

import csv
from pathlib import Path

from autoresearch.evaluation.campaign_summary import CampaignSummary


def export_campaign_tables(summary: CampaignSummary, output_dir: str | Path) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    main_path = target_dir / "main_campaign_summary.csv"
    governance_path = target_dir / "governance_metrics.csv"

    metrics = summary.metrics.to_dict()
    _write_single_row_csv(
        main_path,
        metrics,
        (
            "campaign_id",
            "node_id",
            "metric_name",
            "metric_direction",
            "initial_metric",
            "best_metric",
            "final_accepted_metric",
            "net_gain",
            "gain_per_trial",
            "gain_per_accepted_trial",
            "gain_per_budget_unit",
            "total_wall_clock_seconds",
            "gain_per_hour",
        ),
    )
    _write_single_row_csv(
        governance_path,
        metrics,
        (
            "campaign_id",
            "total_trials",
            "kept_count",
            "discarded_count",
            "failed_invalid_count",
            "acceptance_rate",
            "invalid_rate",
            "complete_provenance_rate",
            "editable_scope_violation_count",
            "trials_with_rationale",
            "command_failure_rate",
            "metric_parsing_failure_rate",
            "artifact_capture_completeness",
        ),
    )
    return {
        "main_campaign_summary": main_path,
        "governance_metrics": governance_path,
    }


def _write_single_row_csv(path: Path, row: dict[str, object], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field) for field in fieldnames})
