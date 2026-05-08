"""Tests for autoresearch.nodes.spec — NodeSpec loading and validation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.nodes.spec import BudgetSpec, NodeSpec, NodeSpecError, load_node_spec


def _minimal_payload(**overrides) -> dict:
    payload = {
        "name": "test_node",
        "description": "A test benchmark node.",
        "editable_paths": ["train.py"],
        "frozen_paths": ["prepare.py"],
        "setup_command": "python prepare.py",
        "run_command": "python train.py",
        "metric_name": "val_auc",
        "metric_direction": "maximize",
        "metric_parser": "metric_parser:parse_val_auc",
        "acceptance_rule": "candidate_metric > current_best_metric",
        "validity_checks": ["metric_present", "editable_scope_only"],
        "default_budget": {"trials": 10},
    }
    payload.update(overrides)
    return payload


class TestBudgetSpec(unittest.TestCase):
    def test_valid_budget(self):
        b = BudgetSpec.from_mapping({"trials": 5, "max_wall_clock_hours": 2.0})
        self.assertEqual(b.trials, 5)
        self.assertEqual(b.max_wall_clock_hours, 2.0)

    def test_optional_wall_clock(self):
        b = BudgetSpec.from_mapping({"trials": 3})
        self.assertIsNone(b.max_wall_clock_hours)

    def test_zero_trials_raises(self):
        with self.assertRaises(NodeSpecError):
            BudgetSpec.from_mapping({"trials": 0})

    def test_to_dict_round_trip(self):
        b = BudgetSpec(trials=7, max_wall_clock_hours=4.0)
        d = b.to_dict()
        self.assertEqual(d["trials"], 7)
        self.assertEqual(d["max_wall_clock_hours"], 4.0)


class TestNodeSpecFromMapping(unittest.TestCase):
    def test_minimal_valid_payload(self):
        spec = NodeSpec.from_mapping(_minimal_payload())
        self.assertEqual(spec.name, "test_node")
        self.assertEqual(spec.metric_name, "val_auc")
        self.assertEqual(spec.metric_direction, "maximize")
        self.assertIn("train.py", spec.editable_paths)

    def test_minimize_direction_accepted(self):
        spec = NodeSpec.from_mapping(_minimal_payload(metric_direction="minimize"))
        self.assertEqual(spec.metric_direction, "minimize")

    def test_invalid_direction_raises(self):
        with self.assertRaises(NodeSpecError):
            NodeSpec.from_mapping(_minimal_payload(metric_direction="sideways"))

    def test_missing_required_field_raises(self):
        payload = _minimal_payload()
        del payload["metric_name"]
        with self.assertRaises(NodeSpecError):
            NodeSpec.from_mapping(payload)

    def test_empty_editable_paths_raises(self):
        with self.assertRaises(NodeSpecError):
            NodeSpec.from_mapping(_minimal_payload(editable_paths=[]))

    def test_failure_categories_optional(self):
        spec = NodeSpec.from_mapping(_minimal_payload())
        self.assertEqual(spec.failure_categories, ())

    def test_failure_categories_parsed(self):
        spec = NodeSpec.from_mapping(
            _minimal_payload(failure_categories=["syntax_error", "runtime_error"])
        )
        self.assertIn("syntax_error", spec.failure_categories)

    def test_expected_runtime_optional(self):
        spec = NodeSpec.from_mapping(_minimal_payload())
        self.assertIsNone(spec.expected_runtime)

    def test_expected_runtime_parsed(self):
        spec = NodeSpec.from_mapping(_minimal_payload(expected_runtime="~15 minutes"))
        self.assertEqual(spec.expected_runtime, "~15 minutes")

    def test_to_dict_is_serialisable(self):
        spec = NodeSpec.from_mapping(_minimal_payload())
        d = spec.to_dict()
        # Should round-trip through JSON without error.
        dumped = json.dumps(d)
        self.assertIn("test_node", dumped)


class TestLoadNodeSpec(unittest.TestCase):
    def test_load_from_json_file(self):
        payload = _minimal_payload()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "test_node.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            spec = load_node_spec(config_path)
        self.assertEqual(spec.name, "test_node")

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_node_spec("/nonexistent/path/node.yaml")

    def test_load_resnet_trigger_real_config(self):
        """The real resnet_trigger.yaml must load without error."""
        root = Path(__file__).resolve().parents[1]
        config = root / "configs" / "nodes" / "resnet_trigger.yaml"
        if not config.exists():
            self.skipTest("resnet_trigger.yaml not found")
        spec = load_node_spec(config)
        self.assertEqual(spec.name, "resnet_trigger")
        self.assertIn("train.py", spec.editable_paths)
        self.assertEqual(spec.metric_direction, "maximize")


if __name__ == "__main__":
    unittest.main()
