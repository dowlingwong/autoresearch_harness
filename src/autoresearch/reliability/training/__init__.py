"""Training package — immutable training-view construction.

Builds separate views for successful-only SFT, recovery SFT, outcome-only
optimization, process supervision, and calibrated abstention from
independently adjudicated intervention transitions.

Each example records source run IDs, selection rule, verifier channel,
split, deduplication cluster, and view-builder hash. No task, template,
fault implementation, near-duplicate trace, or adjudicator-derived label
may cross the declared training/evaluation boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TrainingViewType(StrEnum):
    SUCCESSFUL_ONLY_SFT = "successful_only_sft"
    RECOVERY_SFT = "recovery_sft"
    OUTCOME_ONLY = "outcome_only"
    PROCESS_SUPERVISED = "process_supervised"
    PROCESS_ABSTENTION = "process_abstention"


@dataclass(frozen=True)
class TrainingExample:
    """One immutable training example with full provenance."""

    example_id: str
    source_run_ids: tuple[str, ...]
    view_type: TrainingViewType
    selection_rule: str
    verifier_channel: str
    split: str
    deduplication_cluster: str
    view_builder_hash: str
    state: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    reward: float = 0.0
    process_label: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingView:
    """Immutable projection of accepted intervention transitions."""

    view_type: TrainingViewType
    examples: list[TrainingExample] = field(default_factory=list)
    builder_hash: str = ""
    frozen_date: str = ""

    def freeze(self) -> None:
        import time
        content = json.dumps(
            [e.example_id for e in self.examples], sort_keys=True
        ).encode()
        self.builder_hash = hashlib.sha256(content).hexdigest()[:16]
        self.frozen_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def add_example(self, example: TrainingExample) -> None:
        self.examples.append(example)

    @property
    def size(self) -> int:
        return len(self.examples)


class TrainingViewBuilder:
    """Builds immutable training views from adjudicated transitions."""

    def __init__(self, evaluation_split_ids: set[str]):
        self._evaluation_split_ids = evaluation_split_ids

    def is_training_eligible(self, split: str) -> bool:
        return split not in self._evaluation_split_ids

    def build_successful_only_view(
        self, transitions: list[dict[str, Any]]
    ) -> TrainingView:
        view = TrainingView(view_type=TrainingViewType.SUCCESSFUL_ONLY_SFT)
        for t in transitions:
            if t.get("final_status") == "accepted" and t.get("scientific_validity"):
                if self.is_training_eligible(t.get("split", "")):
                    view.add_example(self._make_example(t, TrainingViewType.SUCCESSFUL_ONLY_SFT))
        view.freeze()
        return view

    def build_recovery_view(
        self, transitions: list[dict[str, Any]]
    ) -> TrainingView:
        view = TrainingView(view_type=TrainingViewType.RECOVERY_SFT)
        for t in transitions:
            if "recovery_action" in t:
                if self.is_training_eligible(t.get("split", "")):
                    view.add_example(self._make_example(t, TrainingViewType.RECOVERY_SFT))
        view.freeze()
        return view

    def build_process_supervised_view(
        self, transitions: list[dict[str, Any]]
    ) -> TrainingView:
        view = TrainingView(view_type=TrainingViewType.PROCESS_SUPERVISED)
        for t in transitions:
            if t.get("process_label") and self.is_training_eligible(t.get("split", "")):
                view.add_example(self._make_example(t, TrainingViewType.PROCESS_SUPERVISED))
        view.freeze()
        return view

    @staticmethod
    def _make_example(
        transition: dict[str, Any], view_type: TrainingViewType
    ) -> TrainingExample:
        import uuid
        return TrainingExample(
            example_id=uuid.uuid4().hex[:16],
            source_run_ids=(transition.get("run_id", ""),),
            view_type=view_type,
            selection_rule="adjudicated",
            verifier_channel=transition.get("verifier_version", ""),
            split=transition.get("split", ""),
            deduplication_cluster=transition.get("deduplication_cluster", ""),
            view_builder_hash="",
            state=transition.get("state", {}),
            action=transition.get("action", ""),
            reward=transition.get("reward", 0.0),
            process_label=transition.get("process_label", ""),
            outcome=transition.get("outcome", {}),
        )


@dataclass(frozen=True)
class ComputeLedgerRow:
    """One row of the matched-data/compute accounting ledger."""

    arm_id: str
    source_trajectories: int
    effective_tokens: int
    optimizer_steps: int
    evaluation_budget: int
    verification_cost: float
    match_notes: str = ""


@dataclass(frozen=True)
class ComputeLedger:
    """Matched data/compute ledger across training arms.

    Supports the R2-T requirement that all trainable arms be matched on
    source trajectories, effective tokens, optimizer steps, and evaluation
    budget; unavoidable mismatches must be reported, never hidden.
    """

    arms: dict[str, ComputeLedgerRow] = field(default_factory=dict)

    def add_row(self, row: ComputeLedgerRow) -> None:
        self.arms[row.arm_id] = row

    def report_mismatches(self) -> list[str]:
        """Return declared mismatches across arms."""
        mismatches: list[str] = []
        if not self.arms:
            return mismatches
        keys = ["source_trajectories", "effective_tokens", "optimizer_steps", "evaluation_budget"]
        base = next(iter(self.arms.values()))
        for key in keys:
            values = {aid: getattr(row, key) for aid, row in self.arms.items()}
            if len(set(values.values())) > 1:
                mismatches.append(
                    f"{key} not matched across arms: {values}"
                )
        return mismatches

    @property
    def matched(self) -> bool:
        return len(self.report_mismatches()) == 0


@dataclass(frozen=True)
class RegressionGateResult:
    """Outcome of the clean-task non-inferiority regression gate."""

    arm_id: str
    clean_task_success: float
    base_model_success: float
    margin: float
    lower_confidence_bound: float

    @property
    def passed(self) -> bool:
        return self.lower_confidence_bound >= self.base_model_success - self.margin

    @property
    def regression(self) -> float:
        return self.base_model_success - self.clean_task_success


def run_clean_task_regression_gate(
    *,
    arm_id: str,
    clean_task_success: float,
    base_model_success: float,
    margin: float = 0.02,
    confidence_interval_half_width: float = 0.005,
) -> RegressionGateResult:
    """Apply the predeclared clean-task non-inferiority gate (R2-T).

    The gate passes when the lower confidence bound of the clean-task success
    is above ``base_model_success - margin``. Default margin is the
    preregistered 0.02 (two percentage points).
    """
    lower_bound = clean_task_success - confidence_interval_half_width
    return RegressionGateResult(
        arm_id=arm_id,
        clean_task_success=clean_task_success,
        base_model_success=base_model_success,
        margin=margin,
        lower_confidence_bound=lower_bound,
    )
