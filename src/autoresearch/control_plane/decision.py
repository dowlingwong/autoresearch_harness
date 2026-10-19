from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from autoresearch.memory.schemas import FailureCategory, TrialDecision, ValidityStatus

if TYPE_CHECKING:  # pragma: no cover
    from autoresearch.control_plane.action_versions import KeepPrior

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



def decide_trial_bayes_threshold(
    *,
    validity_status: ValidityStatus,
    candidate_metric: float | None,
    current_best_metric: float | None,
    metric_direction: MetricDirection,
    prior: "KeepPrior",
    failure_category: FailureCategory | None = None,
) -> tuple[DecisionResult, dict[str, object]]:
    """Run-free Bayes threshold keep rule (``bayes_threshold/v1``, D4 amendment).

    Keep iff Δ̂ > t*, t* = −μσ²/τ² (T2 Proposition 1), with Δ̂ the
    direction-adjusted delta against the current best.  Invalid trials and the
    first valid metric are decided exactly as by :func:`decide_trial`.  Returns
    the decision and the record of the rule (version, prior, threshold, Δ̂, the
    default rule's decision) for the decision event.
    """
    from autoresearch.control_plane.action_versions import KEEP_RULE_BAYES, bayes_keep

    default = decide_trial(
        validity_status=validity_status,
        candidate_metric=candidate_metric,
        current_best_metric=current_best_metric,
        metric_direction=metric_direction,
        failure_category=failure_category,
    )
    meta: dict[str, object] = {
        "version": KEEP_RULE_BAYES,
        "prior": prior.to_dict(),
        "threshold": prior.threshold,
        "delta_hat": default.delta_vs_best,
        "default_decision": default.decision.value,
    }
    if default.delta_vs_best is None:
        # Invalid trial or first valid metric: the rule has no comparison to act on.
        meta["applied"] = False
        return default, meta
    meta["applied"] = True
    delta = default.delta_vs_best
    if bayes_keep(delta, prior):
        result = DecisionResult(
            decision=TrialDecision.KEPT,
            rationale=(
                f"Bayes threshold keep: delta {delta:.6f} exceeds t*={prior.threshold:.6f} "
                f"({metric_direction})."
            ),
            delta_vs_best=delta,
        )
    else:
        result = DecisionResult(
            decision=TrialDecision.DISCARDED,
            rationale=(
                f"Bayes threshold discard: delta {delta:.6f} does not exceed "
                f"t*={prior.threshold:.6f} ({metric_direction})."
            ),
            delta_vs_best=delta,
            failure_category=FailureCategory.DEGRADED_METRIC,
        )
    meta["decision"] = result.decision.value
    return result, meta
