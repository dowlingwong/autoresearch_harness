"""M10 step 0: validation-only selection for test-scored tasks."""
from __future__ import annotations

import pytest

from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import RunOutcome
from autoresearch.trackg.selection import (
    SelectionPolicyError,
    assert_no_test_metric,
    manager_view,
    parse_sealed_test_metric,
    select_best,
)

from .fixtures import spec_mapping


def spec(direction="maximize"):
    return TrackGNodeSpec.from_mapping(spec_mapping(metric_direction=direction))


def test_select_best_uses_the_validation_value_of_ok_runs_only():
    runs = [("a", RunOutcome.ok(0.7)), ("b", RunOutcome.divergent("non_finite_loss")),
            ("c", RunOutcome.ok(0.9)), ("d", RunOutcome.infrastructure_failure("timeout"))]
    assert select_best(runs, spec()) == "c"
    assert select_best(runs, spec("minimize")) == "a"
    assert select_best([("x", RunOutcome.divergent("no_metric_at_budget"))], spec()) is None


def test_select_best_refuses_raw_metric_dicts():
    with pytest.raises(SelectionPolicyError):
        select_best([("a", {"val_accuracy": 0.1, "test_accuracy": 0.9})], spec())


def test_test_metric_is_sealed_and_hidden_from_the_manager():
    sealed = parse_sealed_test_metric("val_accuracy: 0.6\ntest_accuracy: 0.55\n", spec())
    assert sealed == {"sealed": {"test_accuracy": 0.55}}
    record = {"outcome": {"kind": "ok", "value": 0.6}, **sealed}
    assert "sealed" not in manager_view(record)
    with pytest.raises(SelectionPolicyError):
        assert_no_test_metric(["val_accuracy", "test_accuracy"], spec())
    assert_no_test_metric(["val_accuracy"], spec())
