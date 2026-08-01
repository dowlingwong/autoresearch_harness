"""Control package — verification and recovery policies.

Represents the online decision state using R1's calibrated belief, cause
distribution, evidence freshness, reversibility, resource budget, and
estimated damage. Implements fixed, threshold, adaptive, and oracle policies.

Policies: never_verify, always_verify, fixed_periodic, random_matched_budget,
threshold_confidence, threshold_r1_probability, myopic_voi, learned, oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ControlAction(StrEnum):
    CONTINUE = "continue"
    VERIFY = "verify"
    RECOVER_RETRY = "recover_retry"
    RECOVER_REPAIR = "recover_repair"
    RECOVER_ROLLBACK = "recover_rollback"
    RECOVER_REPLAN = "recover_replan"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class ActionCost:
    """Cost model for a control action."""

    verification_cost: float = 0.0
    compute_cost: float = 0.0
    human_cost: float = 0.0
    damage_risk: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.verification_cost
            + self.compute_cost
            + self.human_cost
            + self.damage_risk
        )


@dataclass(frozen=True)
class DecisionState:
    """Observable state at a decision point."""

    run_id: str
    step_index: int
    failure_probability: float
    cause_distribution: dict[str, float]
    evidence_freshness: float
    budget_remaining: float
    reversibility: float
    estimated_damage: float
    available_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyDecision:
    """Output of a control policy at a decision point."""

    run_id: str
    step_index: int
    selected_action: ControlAction
    selection_propensity: float = 1.0
    eligibility_mask: dict[str, bool] = field(default_factory=dict)
    budget_state: dict[str, float] = field(default_factory=dict)
    action_cost: ActionCost = field(default_factory=ActionCost)
    postconditions: list[str] = field(default_factory=list)


class ControllerPolicy(Protocol):
    """Protocol for verification and recovery control policies."""

    name: str

    def decide(self, state: DecisionState) -> PolicyDecision: ...

    def action_costs(self) -> dict[ControlAction, ActionCost]: ...


# Baseline policy implementations


class NeverVerifyPolicy:
    """Always continue, never verify."""

    name = "never_verify"

    def decide(self, state: DecisionState) -> PolicyDecision:
        return PolicyDecision(
            run_id=state.run_id,
            step_index=state.step_index,
            selected_action=ControlAction.CONTINUE,
            selection_propensity=1.0,
            eligibility_mask={a: (a == "continue") for a in state.available_actions},
        )

    def action_costs(self) -> dict[ControlAction, ActionCost]:
        return {
            ControlAction.CONTINUE: ActionCost(),
            ControlAction.VERIFY: ActionCost(verification_cost=1.0),
        }


class AlwaysVerifyPolicy:
    """Verify every eligible step."""

    name = "always_verify"

    def decide(self, state: DecisionState) -> PolicyDecision:
        return PolicyDecision(
            run_id=state.run_id,
            step_index=state.step_index,
            selected_action=ControlAction.VERIFY,
            selection_propensity=1.0,
            eligibility_mask={a: (a == "verify") for a in state.available_actions},
        )

    def action_costs(self) -> dict[ControlAction, ActionCost]:
        return {
            ControlAction.CONTINUE: ActionCost(),
            ControlAction.VERIFY: ActionCost(verification_cost=1.0),
        }


class ThresholdPolicy:
    """Verify when failure probability exceeds a threshold."""

    name = "threshold_r1_probability"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, state: DecisionState) -> PolicyDecision:
        if state.failure_probability > self.threshold:
            action = ControlAction.VERIFY
        else:
            action = ControlAction.CONTINUE
        return PolicyDecision(
            run_id=state.run_id,
            step_index=state.step_index,
            selected_action=action,
            selection_propensity=1.0,
            eligibility_mask={
                a: (a in ("continue", "verify")) for a in state.available_actions
            },
        )

    def action_costs(self) -> dict[ControlAction, ActionCost]:
        return {
            ControlAction.CONTINUE: ActionCost(),
            ControlAction.VERIFY: ActionCost(verification_cost=1.0),
        }
