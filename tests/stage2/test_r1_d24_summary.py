"""D24 (H1): memory-summary lines use the node metric, show 0.0, respect the direction.

AI-assisted (BUILD_BATCH3, H1); ENGINEERING_ONLY.
"""
from __future__ import annotations

from autoresearch.memory.schemas import (
    ExecutionStatus,
    TrialDecision,
    TrialProvenance,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.memory.summarizer import (
    MemoryMode,
    build_memory_context,
)
from autoresearch.nodes.spec import NodeSpec


def _spec(metric_name: str = "val_score", direction: str = "maximize") -> NodeSpec:
    return NodeSpec.from_mapping(
        {
            "name": "fixture_node",
            "description": "fixture",
            "editable_paths": ["train.py"],
            "frozen_paths": [],
            "setup_command": "",
            "run_command": "python3 train.py",
            "metric_name": metric_name,
            "metric_direction": direction,
            "metric_parser": "fixture:parse",
            "acceptance_rule": "candidate_metric > current_best_metric",
            "validity_checks": ["metric_present"],
            "default_budget": {"trials": 10},
        }
    )


def _record(n: int, metrics: dict, decision: TrialDecision = TrialDecision.DISCARDED) -> TrialRecord:
    return TrialRecord(
        trial_id=f"c-trial-{n:03d}",
        campaign_id="c",
        node_id="fixture_node",
        budget_index=n,
        timestamp_start="2026-01-01T00:00:00Z",
        timestamp_end="2026-01-01T00:01:00Z",
        manager_mode="langgraph_manager",
        worker_mode="local_worker",
        memory_mode="append_only_summary",
        proposal_summary=f"change-{n}",
        proposal_rationale="r",
        targeted_files=("train.py",),
        patch_ref="",
        git_commit_before="a",
        git_commit_after="b",
        execution_status=ExecutionStatus.SUCCESS,
        validity_status=ValidityStatus.VALID,
        failure_category=None,
        raw_log_ref="",
        parsed_metrics=metrics,
        current_best_before=None,
        delta_vs_best=None,
        decision=decision,
        decision_rationale="d",
        wall_clock_seconds=1.0,
        cumulative_budget_consumed=n,
        provenance=TrialProvenance("p", "p", "r", "m", "d"),
    )


def test_summary_line_shows_the_node_metric_not_the_first_parsed_metric():
    # train_loss comes first in insertion order; the node metric is val_score.
    records = [_record(1, {"train_loss": 0.5, "val_score": 0.91})]
    context = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY, _spec(), 2)
    assert "metric=0.910000" in context.context_text
    assert "train_loss" not in context.context_text


def test_summary_line_shows_a_metric_of_exactly_zero():
    # train_loss first and the node metric exactly 0.0: the old ``or`` fallback
    # returned the first truthy parsed metric instead of the target.
    records = [_record(1, {"train_loss": 0.3, "val_score": 0.0})]
    context = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY, _spec(), 2)
    assert "metric=0.000000" in context.context_text
    assert "metric=0.300000" not in context.context_text


def test_a_missing_node_metric_never_leaks_another_parsed_metric():
    records = [_record(1, {"train_loss": 0.3})]  # no val_score parsed
    context = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY, _spec(), 2)
    assert "metric=missing" in context.context_text
    assert "0.300000" not in context.context_text


def test_best_strategy_respects_minimise_direction():
    spec = _spec(metric_name="val_loss", direction="minimize")
    records = [
        _record(1, {"val_loss": 0.5}),
        _record(2, {"val_loss": 0.2}),
        _record(3, {"val_loss": 0.8}),
    ]
    context = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE, spec, 4)
    assert "best_strategy=change-2; metric=0.200000" in context.context_text


def test_best_strategy_respects_maximise_direction():
    spec = _spec(metric_name="val_score", direction="maximize")
    records = [
        _record(1, {"val_score": 0.5}),
        _record(2, {"val_score": 0.9}),
    ]
    context = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE, spec, 3)
    assert "best_strategy=change-2; metric=0.900000" in context.context_text


def test_none_mode_output_is_unchanged():
    spec = _spec(metric_name="val_score", direction="maximize")
    records = [_record(1, {"val_score": 0.5})]  # records are ignored under none
    context = build_memory_context(records, "none", spec, 7)
    assert context.context_text == (
        "node=fixture_node\n"
        "budget_index=7\n"
        "metric=val_score direction=maximize\n"
        "editable_paths=train.py"
    )
