"""Tests for autoresearch.control_plane.state_machine — trial lifecycle."""
from __future__ import annotations

import unittest

from autoresearch.control_plane.state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransitionError,
    TrialState,
    assert_transition,
    can_transition,
    transition,
)


class TestTrialStateEnum(unittest.TestCase):
    def test_all_states_are_strings(self):
        for state in TrialState:
            self.assertIsInstance(state.value, str)

    def test_terminal_states_have_no_successors(self):
        for terminal in TERMINAL_STATES:
            self.assertEqual(ALLOWED_TRANSITIONS[terminal], frozenset())

    def test_terminal_states_correct(self):
        self.assertIn(TrialState.KEPT, TERMINAL_STATES)
        self.assertIn(TrialState.DISCARDED, TERMINAL_STATES)
        self.assertIn(TrialState.FAILED_INVALID, TERMINAL_STATES)


class TestCanTransition(unittest.TestCase):
    def test_valid_forward_path(self):
        path = [
            (TrialState.INITIALIZED, TrialState.PROPOSED),
            (TrialState.PROPOSED, TrialState.PATCH_GENERATED),
            (TrialState.PATCH_GENERATED, TrialState.SCOPE_VALIDATED),
            (TrialState.SCOPE_VALIDATED, TrialState.EXECUTED),
            (TrialState.EXECUTED, TrialState.METRIC_PARSED),
            (TrialState.METRIC_PARSED, TrialState.EVALUATED),
            (TrialState.EVALUATED, TrialState.KEPT),
        ]
        for src, dst in path:
            self.assertTrue(can_transition(src, dst), f"expected {src} -> {dst} to be valid")

    def test_failed_invalid_reachable_from_most_states(self):
        for state in TrialState:
            if state in TERMINAL_STATES:
                continue
            if state == TrialState.INITIALIZED:
                continue  # only PROPOSED from INITIALIZED
            # FAILED_INVALID should be reachable from non-terminal non-initialized states.
            if TrialState.FAILED_INVALID in ALLOWED_TRANSITIONS.get(state, frozenset()):
                self.assertTrue(can_transition(state, TrialState.FAILED_INVALID))

    def test_invalid_backward_transition(self):
        self.assertFalse(can_transition(TrialState.EXECUTED, TrialState.INITIALIZED))
        self.assertFalse(can_transition(TrialState.KEPT, TrialState.INITIALIZED))

    def test_self_transition_invalid(self):
        for state in TrialState:
            self.assertFalse(can_transition(state, state))

    def test_skipping_states_invalid(self):
        # Cannot jump from INITIALIZED directly to EXECUTED.
        self.assertFalse(can_transition(TrialState.INITIALIZED, TrialState.EXECUTED))


class TestAssertTransition(unittest.TestCase):
    def test_valid_transition_does_not_raise(self):
        assert_transition(TrialState.INITIALIZED, TrialState.PROPOSED)

    def test_invalid_transition_raises(self):
        with self.assertRaises(InvalidTransitionError):
            assert_transition(TrialState.INITIALIZED, TrialState.EXECUTED)

    def test_terminal_to_anything_raises(self):
        with self.assertRaises(InvalidTransitionError):
            assert_transition(TrialState.KEPT, TrialState.INITIALIZED)


class TestTransition(unittest.TestCase):
    def test_returns_target_on_success(self):
        result = transition(TrialState.INITIALIZED, TrialState.PROPOSED)
        self.assertEqual(result, TrialState.PROPOSED)

    def test_raises_on_invalid(self):
        with self.assertRaises(InvalidTransitionError):
            transition(TrialState.PROPOSED, TrialState.KEPT)

    def test_happy_path_chain(self):
        states = [
            TrialState.INITIALIZED,
            TrialState.PROPOSED,
            TrialState.PATCH_GENERATED,
            TrialState.SCOPE_VALIDATED,
            TrialState.EXECUTED,
            TrialState.METRIC_PARSED,
            TrialState.EVALUATED,
            TrialState.KEPT,
        ]
        current = states[0]
        for next_state in states[1:]:
            current = transition(current, next_state)
        self.assertEqual(current, TrialState.KEPT)

    def test_failure_path_from_executed(self):
        result = transition(TrialState.EXECUTED, TrialState.FAILED_INVALID)
        self.assertEqual(result, TrialState.FAILED_INVALID)


if __name__ == "__main__":
    unittest.main()
