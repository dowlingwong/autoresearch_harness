"""Tests for autoresearch.control_plane.permissions — editable-scope validation."""
from __future__ import annotations

import unittest

from autoresearch.control_plane.permissions import (
    ScopeValidationResult,
    validate_edit_scope,
)
from autoresearch.nodes.spec import BudgetSpec, NodeSpec


def _spec(editable=("train.py",), frozen=("prepare.py", "data/")) -> NodeSpec:
    return NodeSpec(
        name="test_node",
        description="test",
        editable_paths=tuple(editable),
        frozen_paths=tuple(frozen),
        setup_command="python prepare.py",
        run_command="python train.py",
        metric_name="val_auc",
        metric_direction="maximize",
        metric_parser="metric_parser:parse_val_auc",
        acceptance_rule="candidate_metric > current_best_metric",
        validity_checks=("metric_present",),
        default_budget=BudgetSpec(trials=5),
    )


class TestValidEditScope(unittest.TestCase):
    def test_allowed_file_passes(self):
        result = validate_edit_scope(["train.py"], _spec())
        self.assertTrue(result.valid)
        self.assertEqual(result.violations, ())

    def test_frozen_file_fails(self):
        result = validate_edit_scope(["prepare.py"], _spec())
        self.assertFalse(result.valid)
        self.assertTrue(any("frozen_paths" in v for v in result.violations))

    def test_outside_editable_fails(self):
        result = validate_edit_scope(["config.yaml"], _spec())
        self.assertFalse(result.valid)
        self.assertTrue(any("outside editable_paths" in v for v in result.violations))

    def test_file_inside_frozen_directory_fails(self):
        result = validate_edit_scope(["data/train.csv"], _spec())
        self.assertFalse(result.valid)

    def test_multiple_files_all_valid(self):
        spec = _spec(editable=("train.py", "model.py"), frozen=("prepare.py",))
        result = validate_edit_scope(["train.py", "model.py"], spec)
        self.assertTrue(result.valid)

    def test_mixed_valid_and_invalid_fails(self):
        result = validate_edit_scope(["train.py", "prepare.py"], _spec())
        self.assertFalse(result.valid)

    def test_empty_changed_files(self):
        result = validate_edit_scope([], _spec())
        self.assertTrue(result.valid)
        self.assertEqual(result.changed_paths, ())

    def test_windows_style_path_normalised(self):
        result = validate_edit_scope(["train.py"], _spec())
        self.assertTrue(result.valid)

    def test_require_valid_raises_on_violation(self):
        result = validate_edit_scope(["prepare.py"], _spec())
        with self.assertRaises(PermissionError):
            result.require_valid()

    def test_require_valid_does_not_raise_on_success(self):
        result = validate_edit_scope(["train.py"], _spec())
        result.require_valid()  # should not raise

    def test_changed_paths_normalised_in_result(self):
        result = validate_edit_scope(["./train.py"], _spec())
        self.assertIn("train.py", result.changed_paths)

    def test_subdirectory_under_editable_prefix(self):
        spec = _spec(editable=("src/",), frozen=("prepare.py",))
        result = validate_edit_scope(["src/model.py"], spec)
        self.assertTrue(result.valid)


class TestScopeValidationResult(unittest.TestCase):
    def test_valid_result(self):
        r = ScopeValidationResult(valid=True, changed_paths=("train.py",), violations=())
        self.assertTrue(r.valid)
        self.assertEqual(r.changed_paths, ("train.py",))

    def test_invalid_result_has_violations(self):
        r = ScopeValidationResult(
            valid=False,
            changed_paths=("prepare.py",),
            violations=("prepare.py is outside editable_paths",),
        )
        self.assertFalse(r.valid)
        self.assertEqual(len(r.violations), 1)


if __name__ == "__main__":
    unittest.main()
