"""Reliability event types — discriminated union event payloads.

Extends the harness CampaignEvent with reliability-specific payload types.
Events adhere to a stable envelope with parent links, visibility classes,
and integrity hashing.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReliabilityEventType(StrEnum):
    PROPOSAL = "proposal"
    ACTION_REQUEST = "action_request"
    ACTION_RESULT = "action_result"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_DIFF = "state_diff"
    EVIDENCE = "evidence"
    VERIFIER_REQUEST = "verifier_request"
    VERIFIER_RESULT = "verifier_result"
    FAULT_INJECTION = "fault_injection"
    BELIEF = "belief"
    RECOVERY_ACTION = "recovery_action"
    HUMAN_ESCALATION = "human_escalation"
    TERMINAL_OUTCOME = "terminal_outcome"
    ADJUDICATION = "adjudication"


class EventVisibility(StrEnum):
    POLICY_VISIBLE = "policy_visible"
    HIDDEN_FAULT_TRUTH = "hidden_fault_truth"
    SEALED_TEST_OUTCOME = "sealed_test_outcome"
    TRAINING_REWARD_ONLY = "training_reward_only"
    ADJUDICATION_ONLY = "adjudication_only"


SCHEMA_VERSION = "1.0"


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode("utf-8")


def _compute_integrity_hash(payload: dict[str, Any], parent_ids: list[str]) -> str:
    inner = {"payload": payload, "parent_event_ids": sorted(parent_ids)}
    return hashlib.sha256(_canonical_json(inner)).hexdigest()[:16]


@dataclass(frozen=True)
class ReliabilityEvent:
    """Stable event envelope for all reliability events."""

    event_id: str
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    parent_event_ids: tuple[str, ...] = ()
    event_type: ReliabilityEventType = ReliabilityEventType.PROPOSAL
    actor_type: str = ""
    actor_id: str = ""
    monotonic_step: int = 0
    wall_time_utc: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    visibility_class: EventVisibility = EventVisibility.POLICY_VISIBLE
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            object.__setattr__(
                self,
                "integrity_hash",
                _compute_integrity_hash(self.payload, list(self.parent_event_ids)),
            )
        if not self.wall_time_utc:
            object.__setattr__(
                self, "wall_time_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )

    @classmethod
    def create(
        cls,
        run_id: str,
        event_type: ReliabilityEventType,
        payload: dict[str, Any],
        *,
        parent_event_ids: tuple[str, ...] = (),
        actor_type: str = "",
        actor_id: str = "",
        monotonic_step: int = 0,
        artifact_refs: tuple[str, ...] = (),
        visibility_class: EventVisibility = EventVisibility.POLICY_VISIBLE,
    ) -> ReliabilityEvent:
        event_id = uuid.uuid4().hex[:16]
        return cls(
            event_id=event_id,
            run_id=run_id,
            parent_event_ids=parent_event_ids,
            event_type=event_type,
            actor_type=actor_type,
            actor_id=actor_id,
            monotonic_step=monotonic_step,
            payload=payload,
            artifact_refs=artifact_refs,
            visibility_class=visibility_class,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_event_ids": list(self.parent_event_ids),
            "event_type": self.event_type.value,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "monotonic_step": self.monotonic_step,
            "wall_time_utc": self.wall_time_utc,
            "payload": self.payload,
            "artifact_refs": list(self.artifact_refs),
            "visibility_class": self.visibility_class.value,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReliabilityEvent:
        return cls(
            event_id=d["event_id"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            run_id=d.get("run_id", ""),
            parent_event_ids=tuple(d.get("parent_event_ids", [])),
            event_type=ReliabilityEventType(d.get("event_type", "proposal")),
            actor_type=d.get("actor_type", ""),
            actor_id=d.get("actor_id", ""),
            monotonic_step=d.get("monotonic_step", 0),
            wall_time_utc=d.get("wall_time_utc", ""),
            payload=d.get("payload", {}),
            artifact_refs=tuple(d.get("artifact_refs", [])),
            visibility_class=EventVisibility(
                d.get("visibility_class", "policy_visible")
            ),
            integrity_hash=d.get("integrity_hash", ""),
        )


# Typed event factory functions


def ProposalEvent(
    run_id: str,
    proposed_action: str,
    declared_objective: str,
    expected_postcondition: str,
    dependency: str = "",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.PROPOSAL,
        payload={
            "proposed_action": proposed_action,
            "declared_objective": declared_objective,
            "expected_postcondition": expected_postcondition,
            "dependency": dependency,
        },
        **kwargs,
    )


def ActionRequestEvent(
    run_id: str,
    action_type: str,
    arguments: dict[str, Any],
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.ACTION_REQUEST,
        payload={"action_type": action_type, "arguments": arguments},
        **kwargs,
    )


def ActionResultEvent(
    run_id: str,
    exit_state: str,
    observed_state_difference: dict[str, Any] | None = None,
    side_effects: list[str] | None = None,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.ACTION_RESULT,
        payload={
            "exit_state": exit_state,
            "observed_state_difference": observed_state_difference or {},
            "side_effects": side_effects or [],
        },
        **kwargs,
    )


def StateSnapshotEvent(
    run_id: str,
    target: str,
    content_ref: str,
    snapshot_type: str = "filesystem",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.STATE_SNAPSHOT,
        payload={
            "target": target,
            "content_ref": content_ref,
            "snapshot_type": snapshot_type,
        },
        **kwargs,
    )


def StateDiffEvent(
    run_id: str,
    target: str,
    before_ref: str,
    after_ref: str,
    diff_type: str,
    expected_change: str = "",
    observed_change: str = "",
    reconciled: bool = False,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.STATE_DIFF,
        payload={
            "target": target,
            "before_ref": before_ref,
            "after_ref": after_ref,
            "diff_type": diff_type,
            "expected_change": expected_change,
            "observed_change": observed_change,
            "reconciled": reconciled,
        },
        **kwargs,
    )


def EvidenceEvent(
    run_id: str,
    artifact_hash: str,
    measurement: dict[str, Any],
    provenance: str = "",
    freshness: str = "",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.EVIDENCE,
        payload={
            "artifact_hash": artifact_hash,
            "measurement": measurement,
            "provenance": provenance,
            "freshness": freshness,
        },
        **kwargs,
    )


def VerifierRequestEvent(
    run_id: str,
    check_type: str,
    target: str,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.VERIFIER_REQUEST,
        payload={"check_type": check_type, "target": target},
        **kwargs,
    )


def VerifierResultEvent(
    run_id: str,
    check_type: str,
    passed: bool,
    evidence_refs: list[str] | None = None,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.VERIFIER_RESULT,
        payload={
            "check_type": check_type,
            "passed": passed,
            "evidence_refs": evidence_refs or [],
        },
        **kwargs,
    )


def FaultInjectionEvent(
    run_id: str,
    fault_manifest_hash: str,
    injection_trigger: str,
    severity: str,
    persistence: str,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.FAULT_INJECTION,
        payload={
            "fault_manifest_hash": fault_manifest_hash,
            "injection_trigger": injection_trigger,
            "severity": severity,
            "persistence": persistence,
        },
        visibility_class=EventVisibility.HIDDEN_FAULT_TRUTH,
        **kwargs,
    )


def BeliefEvent(
    run_id: str,
    failure_probability: float,
    cause_distribution: dict[str, float],
    calibration_version: str = "",
    abstention_state: str = "active",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.BELIEF,
        payload={
            "failure_probability": failure_probability,
            "cause_distribution": cause_distribution,
            "calibration_version": calibration_version,
            "abstention_state": abstention_state,
        },
        **kwargs,
    )


def RecoveryActionEvent(
    run_id: str,
    action: str,
    cost: dict[str, float],
    rollback_boundary: str = "",
    alternatives: list[str] | None = None,
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.RECOVERY_ACTION,
        payload={
            "action": action,
            "cost": cost,
            "rollback_boundary": rollback_boundary,
            "alternatives": alternatives or [],
        },
        **kwargs,
    )


def HumanEscalationEvent(
    run_id: str,
    reason: str,
    evidence_summary: str = "",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.HUMAN_ESCALATION,
        payload={"reason": reason, "evidence_summary": evidence_summary},
        **kwargs,
    )


def TerminalOutcomeEvent(
    run_id: str,
    task_success: bool,
    scientific_validity: bool,
    damage: dict[str, Any] | None = None,
    recovery_status: str = "",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.TERMINAL_OUTCOME,
        payload={
            "task_success": task_success,
            "scientific_validity": scientific_validity,
            "damage": damage or {},
            "recovery_status": recovery_status,
        },
        **kwargs,
    )


def AdjudicationEvent(
    run_id: str,
    execution_success: bool,
    scientific_validity: bool,
    verifier_version: str,
    evidence_references: list[str] | None = None,
    disagreements: list[str] | None = None,
    final_status: str = "accepted",
    **kwargs: Any,
) -> ReliabilityEvent:
    return ReliabilityEvent.create(
        run_id=run_id,
        event_type=ReliabilityEventType.ADJUDICATION,
        payload={
            "execution_success": execution_success,
            "scientific_validity": scientific_validity,
            "verifier_version": verifier_version,
            "evidence_references": evidence_references or [],
            "disagreements": disagreements or [],
            "final_status": final_status,
        },
        visibility_class=EventVisibility.ADJUDICATION_ONLY,
        **kwargs,
    )
