from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from autoresearch.evaluation.metrics import CampaignMetrics, compute_campaign_metrics
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialRecord


@dataclass(frozen=True)
class CampaignSummary:
    metrics: CampaignMetrics
    records: tuple[TrialRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "record_count": len(self.records),
        }


def load_campaign_summary(
    records_path: str | Path,
    metric_name: str = "val_auc",
    metric_direction: str = "maximize",
) -> CampaignSummary:
    records = TrialAppendStore(records_path).read_all()
    metrics = compute_campaign_metrics(records, metric_name=metric_name, metric_direction=metric_direction)  # type: ignore[arg-type]
    return CampaignSummary(metrics=metrics, records=tuple(records))

