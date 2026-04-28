from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class ValidityStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class TrialDecision(StrEnum):
    KEPT = "kept"
    DISCARDED = "discarded"
    FAILED_INVALID = "failed_invalid"


class FailureCategory(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    METRIC_MISSING = "metric_missing"
    INVALID_EDIT_SCOPE = "invalid_edit_scope"
    DEGRADED_METRIC = "degraded_metric"


@dataclass(frozen=True)
class TrialProvenance:
    proposal_id: str
    patch_id: str
    run_id: str
    metric_id: str
    decision_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    campaign_id: str
    node_id: str
    budget_index: int
    timestamp_start: str
    timestamp_end: str
    manager_mode: str
    worker_mode: str
    memory_mode: str
    proposal_summary: str
    proposal_rationale: str
    targeted_files: tuple[str, ...]
    patch_ref: str
    git_commit_before: str
    git_commit_after: str
    execution_status: ExecutionStatus
    validity_status: ValidityStatus
    failure_category: FailureCategory | None
    raw_log_ref: str
    parsed_metrics: dict[str, float]
    current_best_before: float | None
    delta_vs_best: float | None
    decision: TrialDecision
    decision_rationale: str
    wall_clock_seconds: float
    cumulative_budget_consumed: int
    provenance: TrialProvenance
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.trial_id:
            errors.append("trial_id must not be empty")
        if not self.campaign_id:
            errors.append("campaign_id must not be empty")
        if self.budget_index < 1:
            errors.append("budget_index must be >= 1")
        if self.cumulative_budget_consumed < self.budget_index:
            errors.append("cumulative_budget_consumed must be >= budget_index")
        if not self.targeted_files:
            errors.append("targeted_files must not be empty")
        if self.wall_clock_seconds < 0:
            errors.append("wall_clock_seconds must be >= 0")
        if self.validity_status == ValidityStatus.INVALID and self.failure_category is None:
            errors.append("invalid trials require failure_category")
        if self.decision == TrialDecision.FAILED_INVALID and self.validity_status != ValidityStatus.INVALID:
            errors.append("failed_invalid decisions require invalid validity_status")
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targeted_files"] = list(self.targeted_files)
        payload["execution_status"] = self.execution_status.value
        payload["validity_status"] = self.validity_status.value
        payload["failure_category"] = self.failure_category.value if self.failure_category else None
        payload["decision"] = self.decision.value
        payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=str(payload["trial_id"]),
            campaign_id=str(payload["campaign_id"]),
            node_id=str(payload["node_id"]),
            budget_index=int(payload["budget_index"]),
            timestamp_start=str(payload["timestamp_start"]),
            timestamp_end=str(payload["timestamp_end"]),
            manager_mode=str(payload["manager_mode"]),
            worker_mode=str(payload["worker_mode"]),
            memory_mode=str(payload["memory_mode"]),
            proposal_summary=str(payload["proposal_summary"]),
            proposal_rationale=str(payload.get("proposal_rationale", "")),
            targeted_files=tuple(str(path) for path in payload["targeted_files"]),
            patch_ref=str(payload["patch_ref"]),
            git_commit_before=str(payload["git_commit_before"]),
            git_commit_after=str(payload["git_commit_after"]),
            execution_status=ExecutionStatus(str(payload["execution_status"])),
            validity_status=ValidityStatus(str(payload["validity_status"])),
            failure_category=(
                FailureCategory(str(payload["failure_category"]))
                if payload.get("failure_category") is not None
                else None
            ),
            raw_log_ref=str(payload["raw_log_ref"]),
            parsed_metrics={str(k): float(v) for k, v in payload.get("parsed_metrics", {}).items()},
            current_best_before=(
                float(payload["current_best_before"])
                if payload.get("current_best_before") is not None
                else None
            ),
            delta_vs_best=(
                float(payload["delta_vs_best"])
                if payload.get("delta_vs_best") is not None
                else None
            ),
            decision=TrialDecision(str(payload["decision"])),
            decision_rationale=str(payload["decision_rationale"]),
            wall_clock_seconds=float(payload["wall_clock_seconds"]),
            cumulative_budget_consumed=int(payload["cumulative_budget_consumed"]),
            provenance=TrialProvenance(**dict(payload["provenance"])),
            extra=dict(payload.get("extra", {})),
        )

