from __future__ import annotations

from enum import StrEnum


class TrialState(StrEnum):
    INITIALIZED = "initialized"
    PROPOSED = "proposed"
    PATCH_GENERATED = "patch_generated"
    SCOPE_VALIDATED = "scope_validated"
    EXECUTED = "executed"
    METRIC_PARSED = "metric_parsed"
    EVALUATED = "evaluated"
    KEPT = "kept"
    DISCARDED = "discarded"
    FAILED_INVALID = "failed_invalid"


TERMINAL_STATES = frozenset(
    {
        TrialState.KEPT,
        TrialState.DISCARDED,
        TrialState.FAILED_INVALID,
    }
)

ALLOWED_TRANSITIONS: dict[TrialState, frozenset[TrialState]] = {
    TrialState.INITIALIZED: frozenset({TrialState.PROPOSED}),
    TrialState.PROPOSED: frozenset({TrialState.PATCH_GENERATED, TrialState.FAILED_INVALID}),
    TrialState.PATCH_GENERATED: frozenset({TrialState.SCOPE_VALIDATED, TrialState.FAILED_INVALID}),
    TrialState.SCOPE_VALIDATED: frozenset({TrialState.EXECUTED, TrialState.FAILED_INVALID}),
    TrialState.EXECUTED: frozenset({TrialState.METRIC_PARSED, TrialState.FAILED_INVALID}),
    TrialState.METRIC_PARSED: frozenset({TrialState.EVALUATED, TrialState.FAILED_INVALID}),
    TrialState.EVALUATED: frozenset(
        {TrialState.KEPT, TrialState.DISCARDED, TrialState.FAILED_INVALID}
    ),
    TrialState.KEPT: frozenset(),
    TrialState.DISCARDED: frozenset(),
    TrialState.FAILED_INVALID: frozenset(),
}


class InvalidTransitionError(ValueError):
    """Raised when a trial lifecycle transition violates the state machine."""


def can_transition(current: TrialState, target: TrialState) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def assert_transition(current: TrialState, target: TrialState) -> None:
    if not can_transition(current, target):
        raise InvalidTransitionError(f"invalid trial transition: {current.value} -> {target.value}")


def transition(current: TrialState, target: TrialState) -> TrialState:
    assert_transition(current, target)
    return target

