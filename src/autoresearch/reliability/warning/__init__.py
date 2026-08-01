"""Warning package — calibrated hazard, attribution, and early-warning baselines.

Produces trajectory-indexed failure hazard predictions, root-cause attribution
distributions, and warning/abstention states from structured event histories.

Baselines: terminal verifier, runtime rules, agent self-confidence,
text-only trace critic, typed-event without state differences,
typed-event with state differences and evidence provenance, oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class WarningState(StrEnum):
    ACTIVE = "active"
    WARNING_ISSUED = "warning_issued"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class WarningPrediction:
    """Output of a warning model at a given trajectory step."""

    step_index: int
    failure_hazard: float
    cause_distribution: dict[str, float]
    warning_state: WarningState = WarningState.ACTIVE
    calibration_version: str = ""
    model_version: str = ""

    @property
    def most_likely_cause(self) -> str:
        if not self.cause_distribution:
            return "unknown"
        return max(self.cause_distribution, key=lambda k: self.cause_distribution[k])


@dataclass(frozen=True)
class WarningResult:
    """Aggregated warning evaluation for a trajectory."""

    run_id: str
    terminal_invalid: bool
    first_warning_step: int | None
    first_terminal_check_step: int | None
    normalized_lead_time: float | None
    false_alert: bool
    correctly_attributed: bool

    @property
    def useful_warning(self) -> bool:
        return (
            self.terminal_invalid
            and not self.false_alert
            and self.normalized_lead_time is not None
            and self.normalized_lead_time > 0
        )


class WarningModel(Protocol):
    """Protocol for calibrated early-warning models."""

    name: str

    def predict_at_step(
        self,
        run_id: str,
        step_index: int,
        history: list[dict[str, Any]],
    ) -> WarningPrediction: ...

    def calibration_version(self) -> str: ...
