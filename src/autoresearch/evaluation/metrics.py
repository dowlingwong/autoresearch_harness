from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from autoresearch.memory.schemas import FailureCategory, TrialDecision, TrialRecord, ValidityStatus

MetricDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class CampaignMetrics:
    campaign_id: str
    node_id: str
    metric_name: str
    metric_direction: str
    total_trials: int
    kept_count: int
    discarded_count: int
    failed_invalid_count: int
    acceptance_rate: float
    invalid_rate: float
    initial_metric: float | None
    best_metric: float | None
    final_accepted_metric: float | None
    net_gain: float | None
    gain_per_trial: float | None
    gain_per_accepted_trial: float | None
    total_wall_clock_seconds: float
    gain_per_hour: float | None
    gain_per_budget_unit: float | None
    complete_provenance_rate: float
    editable_scope_violation_count: int = 0
    trials_with_rationale: int = 0
    command_failure_rate: float = 0.0
    metric_parsing_failure_rate: float = 0.0
    artifact_capture_completeness: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_campaign_metrics(
    records: list[TrialRecord],
    metric_name: str = "val_auc",
    metric_direction: MetricDirection = "maximize",
) -> CampaignMetrics:
    if not records:
        return CampaignMetrics(
            campaign_id="",
            node_id="",
            metric_name=metric_name,
            metric_direction=metric_direction,
            total_trials=0,
            kept_count=0,
            discarded_count=0,
            failed_invalid_count=0,
            acceptance_rate=0.0,
            invalid_rate=0.0,
            initial_metric=None,
            best_metric=None,
            final_accepted_metric=None,
            net_gain=None,
            gain_per_trial=None,
            gain_per_accepted_trial=None,
            total_wall_clock_seconds=0.0,
            gain_per_hour=None,
            gain_per_budget_unit=None,
            complete_provenance_rate=0.0,
            editable_scope_violation_count=0,
            trials_with_rationale=0,
            command_failure_rate=0.0,
            metric_parsing_failure_rate=0.0,
            artifact_capture_completeness=0.0,
        )

    values = [record.parsed_metrics[metric_name] for record in records if metric_name in record.parsed_metrics]
    kept = [record for record in records if record.decision == TrialDecision.KEPT]
    discarded = [record for record in records if record.decision == TrialDecision.DISCARDED]
    invalid = [record for record in records if record.decision == TrialDecision.FAILED_INVALID]
    initial_metric = _initial_metric(records, metric_name)
    best_metric = _best(values, metric_direction)
    final_accepted_metric = _final_accepted_metric(kept, metric_name)
    net_gain = _gain(initial_metric, best_metric, metric_direction)
    total_wall_clock = sum(record.wall_clock_seconds for record in records)
    kept_count = len(kept)

    return CampaignMetrics(
        campaign_id=records[0].campaign_id,
        node_id=records[0].node_id,
        metric_name=metric_name,
        metric_direction=metric_direction,
        total_trials=len(records),
        kept_count=kept_count,
        discarded_count=len(discarded),
        failed_invalid_count=len(invalid),
        acceptance_rate=kept_count / len(records),
        invalid_rate=len(invalid) / len(records),
        initial_metric=initial_metric,
        best_metric=best_metric,
        final_accepted_metric=final_accepted_metric,
        net_gain=net_gain,
        gain_per_trial=(net_gain / len(records)) if net_gain is not None else None,
        gain_per_accepted_trial=(net_gain / kept_count) if net_gain is not None and kept_count else None,
        total_wall_clock_seconds=total_wall_clock,
        gain_per_hour=(net_gain / (total_wall_clock / 3600.0)) if net_gain is not None and total_wall_clock > 0 else None,
        gain_per_budget_unit=(net_gain / len(records)) if net_gain is not None else None,
        complete_provenance_rate=_complete_provenance_rate(records),
        editable_scope_violation_count=sum(1 for record in records if record.failure_category == FailureCategory.INVALID_EDIT_SCOPE),
        trials_with_rationale=sum(1 for record in records if bool(record.decision_rationale.strip())),
        command_failure_rate=_failure_rate(records, FailureCategory.RUNTIME_ERROR),
        metric_parsing_failure_rate=_failure_rate(records, FailureCategory.METRIC_MISSING),
        artifact_capture_completeness=_artifact_capture_completeness(records),
    )


def _initial_metric(records: list[TrialRecord], metric_name: str) -> float | None:
    for record in records:
        if record.current_best_before is not None:
            return record.current_best_before
    for record in records:
        if metric_name in record.parsed_metrics:
            return record.parsed_metrics[metric_name]
    return None


def _best(values: list[float], direction: MetricDirection) -> float | None:
    if not values:
        return None
    return max(values) if direction == "maximize" else min(values)


def _final_accepted_metric(records: list[TrialRecord], metric_name: str) -> float | None:
    for record in reversed(records):
        if metric_name in record.parsed_metrics:
            return record.parsed_metrics[metric_name]
    return None


def _gain(initial: float | None, best: float | None, direction: MetricDirection) -> float | None:
    if initial is None or best is None:
        return None
    return best - initial if direction == "maximize" else initial - best


def _complete_provenance_rate(records: list[TrialRecord]) -> float:
    if not records:
        return 0.0
    complete = 0
    for record in records:
        provenance = record.provenance
        if all(
            (
                provenance.proposal_id,
                provenance.patch_id,
                provenance.run_id,
                provenance.metric_id,
                provenance.decision_id,
            )
        ):
            complete += 1
    return complete / len(records)


def _failure_rate(records: list[TrialRecord], category: FailureCategory) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record.failure_category == category) / len(records)


def _artifact_capture_completeness(records: list[TrialRecord]) -> float:
    if not records:
        return 0.0
    complete = 0
    for record in records:
        if record.patch_ref and record.raw_log_ref:
            complete += 1
    return complete / len(records)
