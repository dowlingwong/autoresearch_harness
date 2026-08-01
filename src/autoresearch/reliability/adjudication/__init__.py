"""Adjudication package — independent execution and scientific outcome verification.

Adjudication separates execution success from scientific validity. Records
include deterministic checks, scientific checks, evidence references,
verifier versions, disagreements, human review, missing evidence, and a
final status from a closed vocabulary. Ambiguous cases remain ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AdjudicationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFIER_FAILURE = "verifier_failure"


@dataclass(frozen=True)
class AdjudicationRecord:
    """Independent adjudication of a trajectory or intervention step."""

    adjudication_id: str
    run_id: str
    step_index: int

    execution_success: bool
    scientific_validity: bool

    deterministic_checks: list[str] = field(default_factory=list)
    scientific_checks: list[str] = field(default_factory=list)
    evidence_references: list[str] = field(default_factory=list)
    verifier_versions: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    human_review: str = ""
    missing_evidence: list[str] = field(default_factory=list)

    final_status: AdjudicationStatus = AdjudicationStatus.ACCEPTED

    @classmethod
    def create(
        cls,
        run_id: str,
        step_index: int,
        execution_success: bool,
        scientific_validity: bool,
        **kwargs: Any,
    ) -> AdjudicationRecord:
        import uuid
        return cls(
            adjudication_id=uuid.uuid4().hex[:16],
            run_id=run_id,
            step_index=step_index,
            execution_success=execution_success,
            scientific_validity=scientific_validity,
            **kwargs,
        )

    def is_accepted(self) -> bool:
        return self.final_status == AdjudicationStatus.ACCEPTED

    def is_ambiguous(self) -> bool:
        return self.final_status in (
            AdjudicationStatus.AMBIGUOUS,
            AdjudicationStatus.INSUFFICIENT_EVIDENCE,
        )


def separate_channels(
    training_adjudicator: str,
    evaluation_adjudicator: str,
) -> bool:
    """Verify that training and evaluation adjudicators are distinct."""
    return training_adjudicator != evaluation_adjudicator
