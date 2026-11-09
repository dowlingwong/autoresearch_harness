"""M10 step 0: the trackg_node/v1 template; a wall-clock budget is refused."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.trackg.node_template import (
    TrackGNodeSpec,
    TrackGNodeSpecError,
    WallClockBudgetError,
    load_trackg_node_spec,
)
from autoresearch.trackg.selection import SelectionPolicyError

from .fixtures import spec_mapping, write_spec

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_template_loads_and_maps_to_a_harness_node_spec():
    spec = load_trackg_node_spec(ROOT / "configs" / "trackg" / "node_template.yaml")
    assert spec.run_budget.kind == "steps" and spec.run_budget.value == 2000
    assert spec.budgeted_run_command().endswith("--steps 2000")
    base = spec.to_base_node_spec()
    assert base.default_budget.max_wall_clock_hours is None
    assert validate_edit_scope(["train.py"], base).valid
    assert not validate_edit_scope(["evaluate.py"], base).valid


def test_fixture_spec_round_trips_through_yaml(tmp_path):
    spec = load_trackg_node_spec(write_spec(tmp_path / "n.yaml"))
    assert spec.workers.threads == 1 and spec.selection.test_scored
    assert len(spec.raw_sha256) == 64


@pytest.mark.parametrize("kind", ["wall_clock", "seconds", "minutes", "time", "hours"])
def test_wall_clock_budget_kind_is_refused(kind):
    with pytest.raises(WallClockBudgetError):
        TrackGNodeSpec.from_mapping(spec_mapping(run_budget={"kind": kind, "value": 300}))


@pytest.mark.parametrize("overrides", [
    {"run_budget": {"kind": "steps", "value": 10, "max_wall_clock_seconds": 120}},
    {"default_budget": {"trials": 5, "max_wall_clock_hours": 2.0}},
    {"time_budget": 120},
    {"workers": {"dataloader_workers": 1, "threads": 1, "wall_clock_cap": 3}},
])
def test_wall_clock_key_anywhere_is_refused(overrides):
    with pytest.raises(WallClockBudgetError):
        TrackGNodeSpec.from_mapping(spec_mapping(**overrides))


@pytest.mark.parametrize("budget", [{"kind": "steps", "value": 0}, {"kind": "steps", "value": 1.5},
                                    {"kind": "steps", "value": True}, {"kind": "epochs", "value": 3},
                                    {"kind": "tokens"}])
def test_budget_must_be_a_positive_integer_of_steps_or_tokens(budget):
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(run_budget=budget))


def test_token_budget_is_accepted():
    spec = TrackGNodeSpec.from_mapping(spec_mapping(run_budget={"kind": "tokens", "value": 1_000_000}))
    assert spec.run_budget.kind == "tokens"


@pytest.mark.parametrize("overrides", [
    {"unknown_field": 1},
    {"frozen_paths": []},
    {"editable_paths": []},
    {"editable_paths": ["train.py"], "frozen_paths": ["train.py"]},
    {"editable_paths": ["../outside.py"]},
    {"editable_paths": ["/abs/train.py"]},
    {"workers": {"dataloader_workers": "auto", "threads": 4}},
    {"workers": {"dataloader_workers": 1, "threads": 0}},
    {"metric": {"metric_regex": "no capture group"}},
    {"metric_direction": "up"},
    {"schema": "trackg_node/v0"},
    {"determinism": {"cudnn_benchmark": True}},
    {"determinism": {"deterministic_algorithms": False}},
])
def test_malformed_specs_are_refused(overrides):
    with pytest.raises(TrackGNodeSpecError):
        TrackGNodeSpec.from_mapping(spec_mapping(**overrides))


@pytest.mark.parametrize("selection, metric_name", [
    ({"test_scored": True, "selection_split": "train", "test_metric_name": "test_accuracy"}, "val_accuracy"),
    ({"test_scored": True, "selection_split": "validation"}, "val_accuracy"),
    ({"test_scored": True, "selection_split": "validation", "test_metric_name": "val_accuracy"}, "val_accuracy"),
    ({"test_scored": True, "selection_split": "validation", "test_metric_name": "test_acc"}, "test_accuracy"),
])
def test_test_scored_spec_must_select_on_validation(selection, metric_name):
    with pytest.raises(SelectionPolicyError):
        TrackGNodeSpec.from_mapping(spec_mapping(selection=selection, metric_name=metric_name))
