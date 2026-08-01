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


# Sentinel for unsupported counterfactual actions.
# Any counterfactual whose outcome was never observed under adequate
# propensity support must be represented as UNKNOWN, never imputed.
UNKNOWN_COUNTERFACTUAL = float("nan")


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


class FixedPeriodicPolicy:
    """Verify at a fixed step interval; otherwise continue."""

    name = "fixed_periodic"

    def __init__(self, interval: int = 5):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.interval = interval

    def decide(self, state: DecisionState) -> PolicyDecision:
        verify = (state.step_index % self.interval) == 0
        action = ControlAction.VERIFY if verify else ControlAction.CONTINUE
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


class RandomMatchedBudgetPolicy:
    """Verify with a fixed probability tuned to match a target budget.

    The verification probability is declared before evaluation and is not
    adapted from the test set.
    """

    name = "random_matched_budget"

    def __init__(self, verify_probability: float = 0.2, seed: int = 0):
        if not 0.0 <= verify_probability <= 1.0:
            raise ValueError("verify_probability must be in [0, 1]")
        self.verify_probability = verify_probability
        self._seed = seed

    def decide(self, state: DecisionState) -> PolicyDecision:
        import random

        rng = random.Random(self._seed * 1_000_003 + state.step_index)
        if rng.random() < self.verify_probability:
            action = ControlAction.VERIFY
            propensity = self.verify_probability
        else:
            action = ControlAction.CONTINUE
            propensity = 1.0 - self.verify_probability
        return PolicyDecision(
            run_id=state.run_id,
            step_index=state.step_index,
            selected_action=action,
            selection_propensity=propensity,
            eligibility_mask={
                a: (a in ("continue", "verify")) for a in state.available_actions
            },
        )

    def action_costs(self) -> dict[ControlAction, ActionCost]:
        return {
            ControlAction.CONTINUE: ActionCost(),
            ControlAction.VERIFY: ActionCost(verification_cost=1.0),
        }


class MyopicVoiPolicy:
    """Myopic value-of-information policy.

    Estimates the expected benefit of verification from the KL divergence
    between current and expected posterior belief. This is a fixture-grade
    deterministic policy used to exercise the contract, not a learned result.
    """

    name = "myopic_voi"
    is_fixture = True

    def __init__(self, vois_threshold: float = 0.1, verification_cost: float = 1.0):
        self.vois_threshold = vois_threshold
        self.verification_cost = verification_cost

    def decide(self, state: DecisionState) -> PolicyDecision:
        import math

        uncertainty = 0.0
        for prob in state.cause_distribution.values():
            if 0.0 < prob < 1.0:
                uncertainty += -prob * math.log(prob)
        expected_benefit = uncertainty * (1.0 - state.evidence_freshness)

        if expected_benefit - self.verification_cost > self.vois_threshold:
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
            action_cost=(
                ActionCost(verification_cost=self.verification_cost)
                if action is ControlAction.VERIFY
                else ActionCost()
            ),
        )

    def action_costs(self) -> dict[ControlAction, ActionCost]:
        return {
            ControlAction.CONTINUE: ActionCost(),
            ControlAction.VERIFY: ActionCost(verification_cost=self.verification_cost),
        }


class OraclePolicy:
    """Oracle controller using hidden fault truth.

    Reported only as an unattainable upper bound. It consumes hidden fault
    truth and is never included in the primary comparison.
    """

    name = "oracle_fault_aware"
    is_oracle = True

    def decide(self, state: DecisionState, hidden_fault_present: bool = False) -> PolicyDecision:
        action = ControlAction.VERIFY if hidden_fault_present else ControlAction.CONTINUE
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


def policy_registry() -> dict[str, ControllerPolicy]:
    """Return the declared controller policies by name."""
    return {
        "never_verify": NeverVerifyPolicy(),
        "always_verify": AlwaysVerifyPolicy(),
        "fixed_periodic": FixedPeriodicPolicy(),
        "random_matched_budget": RandomMatchedBudgetPolicy(),
        "threshold_r1_probability": ThresholdPolicy(),
        "myopic_voi": MyopicVoiPolicy(),
        "oracle_fault_aware": OraclePolicy(),
    }
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
