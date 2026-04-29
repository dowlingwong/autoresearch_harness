from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from autoresearch.memory.schemas import FailureCategory, TrialDecision, ValidityStatus

MetricDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class DecisionResult:
    decision: TrialDecision
    rationale: str
    delta_vs_best: float | None
    failure_category: FailureCategory | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["failure_category"] = self.failure_category.value if self.failure_category else None
        return payload


def decide_trial(
    *,
    validity_status: ValidityStatus,
    candidate_metric: float | None,
    current_best_metric: float | None,
    metric_direction: MetricDirection,
    failure_category: FailureCategory | None = None,
) -> DecisionResult:
    if validity_status == ValidityStatus.INVALID:
        return DecisionResult(
            decision=TrialDecision.FAILED_INVALID,
            rationale=f"Invalid trial: {(failure_category or FailureCategory.RUNTIME_ERROR).value}.",
            delta_vs_best=None,
            failure_category=failure_category or FailureCategory.RUNTIME_ERROR,
        )
    if candidate_metric is None:
        return DecisionResult(
            decision=TrialDecision.FAILED_INVALID,
            rationale="Invalid trial: metric_missing.",
            delta_vs_best=None,
            failure_category=FailureCategory.METRIC_MISSING,
        )
    if current_best_metric is None:
        return DecisionResult(
            decision=TrialDecision.KEPT,
            rationale="First valid metric becomes the current best.",
            delta_vs_best=None,
        )

    delta = candidate_metric - current_best_metric if metric_direction == "maximize" else current_best_metric - candidate_metric
    if delta > 0:
        return DecisionResult(
            decision=TrialDecision.KEPT,
            rationale=f"Candidate improved {metric_direction} objective by {delta:.6f}.",
            delta_vs_best=delta,
        )
    return DecisionResult(
        decision=TrialDecision.DISCARDED,
        rationale=f"Candidate did not improve current best; delta={delta:.6f}.",
        delta_vs_best=delta,
        failure_category=FailureCategory.DEGRADED_METRIC,
    )

