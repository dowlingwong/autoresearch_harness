"""Tests for autoresearch.nodes.resnet_trigger.metric_parser."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from autoresearch.nodes.resnet_trigger.metric_parser import (
    MetricParseError,
    ParsedResNetMetrics,
    _extract_numeric_metrics,
    parse_val_auc,
    parse_val_auc_dict,
)


def _write_log(content: str) -> Path:
    """Write content to a temp log file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


class TestExtractNumericMetrics(unittest.TestCase):
    def test_simple_key_value(self):
        m = _extract_numeric_metrics("val_bpb: 0.22\n")
        self.assertAlmostEqual(m["val_bpb"], 0.22)

    def test_multiple_metrics(self):
        text = "train_loss: 0.45\nval_bpb: 0.19\nepochs: 10\n"
        m = _extract_numeric_metrics(text)
        self.assertIn("train_loss", m)
        self.assertIn("val_bpb", m)
        self.assertAlmostEqual(m["epochs"], 10.0)

    def test_no_match_returns_empty(self):
        self.assertEqual(_extract_numeric_metrics("nothing useful here"), {})

    def test_negative_value(self):
        m = _extract_numeric_metrics("delta: -0.003\n")
        self.assertAlmostEqual(m["delta"], -0.003)

    def test_scientific_notation(self):
        m = _extract_numeric_metrics("lr: 5e-4\n")
        self.assertAlmostEqual(m["lr"], 5e-4)

    def test_mid_line_not_matched(self):
        # Metric must start at column 0.
        m = _extract_numeric_metrics("  val_bpb: 0.22\n")
        self.assertNotIn("val_bpb", m)


class TestParseValAuc(unittest.TestCase):
    def test_parse_from_val_bpb(self):
        log = _write_log("val_bpb: 0.22\n")
        result = parse_val_auc(log)
        self.assertAlmostEqual(result.metric_value, 0.78)
        self.assertEqual(result.metric_name, "val_auc")
        self.assertEqual(result.metric_direction, "maximize")

    def test_parse_from_val_auc_directly(self):
        log = _write_log("val_auc: 0.81\n")
        result = parse_val_auc(log)
        self.assertAlmostEqual(result.metric_value, 0.81)

    def test_val_bpb_takes_priority_over_val_auc(self):
        """When both are present, val_bpb is the authoritative source."""
        log = _write_log("val_bpb: 0.20\nval_auc: 0.75\n")
        result = parse_val_auc(log)
        # val_bpb=0.20 → val_auc=0.80, not 0.75
        self.assertAlmostEqual(result.metric_value, 0.80)

    def test_missing_metric_raises(self):
        log = _write_log("train_loss: 0.4\n")
        with self.assertRaises(MetricParseError):
            parse_val_auc(log)

    def test_missing_file_raises(self):
        with self.assertRaises(MetricParseError):
            parse_val_auc("/tmp/nonexistent_run_log_abc123.log")

    def test_non_finite_raises(self):
        log = _write_log("val_bpb: inf\n")
        with self.assertRaises(MetricParseError):
            parse_val_auc(log)

    def test_result_is_frozen_dataclass(self):
        log = _write_log("val_bpb: 0.25\n")
        result = parse_val_auc(log)
        with self.assertRaises(Exception):
            result.metric_value = 0.0  # type: ignore[misc]

    def test_raw_metrics_included(self):
        log = _write_log("val_bpb: 0.22\ntrain_loss: 0.45\n")
        result = parse_val_auc(log)
        self.assertIn("val_bpb", result.raw_metrics)
        self.assertIn("val_auc", result.raw_metrics)
        self.assertIn("train_loss", result.raw_metrics)

    def test_source_log_recorded(self):
        log = _write_log("val_bpb: 0.22\n")
        result = parse_val_auc(log)
        self.assertEqual(result.source_log, str(log))

    def test_to_dict_serialisable(self):
        import json
        log = _write_log("val_bpb: 0.22\n")
        d = parse_val_auc(log).to_dict()
        json.dumps(d)  # must not raise

    def test_parse_val_auc_dict(self):
        log = _write_log("val_bpb: 0.30\n")
        d = parse_val_auc_dict(log)
        self.assertIn("metric_value", d)
        self.assertAlmostEqual(d["metric_value"], 0.70)


if __name__ == "__main__":
    unittest.main()
