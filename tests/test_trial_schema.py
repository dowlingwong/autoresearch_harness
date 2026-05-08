"""Tests for autoresearch.memory.schemas — TrialRecord and TrialProvenance."""
from __future__ import annotations

import json
import unittest

from autoresearch.memory.schemas import (
    ExecutionStatus,
    FailureCategory,
    TrialDecision,
    TrialProvenance,
    TrialRecord,
    ValidityStatus,
)


def _provenance(**overrides) -> TrialProvenance:
    base = {
        "proposal_id": "prop-001",
        "patch_id": "patch-001",
        "run_id": "run-001",
        "metric_id": "metric-001",
        "decision_id": "decision-001",
    }
    base.update(overrides)
    return TrialProvenance(**base)


def _valid_record(**overrides) -> TrialRecord:
    base = dict(
        trial_id="campaign-trial-001",
        campaign_id="campaign",
        node_id="resnet_trigger",
        budget_index=1,
        timestamp_start="2026-01-01T00:00:00Z",
        timestamp_end="2026-01-01T00:15:00Z",
        manager_mode="baseline_manager",
        worker_mode="dry_run_worker",
        memory_mode="none",
        proposal_summary="reduce-lr-5e-4",
        proposal_rationale="LR too high",
        targeted_files=("train.py",),
        patch_ref="experiments/artifacts/trial-001/patch.diff",
        git_commit_before="abc123",
        git_commit_after="def456",
        execution_status=ExecutionStatus.SUCCESS,
        validity_status=ValidityStatus.VALID,
        failure_category=None,
        raw_log_ref="experiments/artifacts/trial-001/run.log",
        parsed_metrics={"val_auc": 0.81},
        current_best_before=0.78,
        delta_vs_best=0.03,
        decision=TrialDecision.KEPT,
        decision_rationale="Improved by 0.03.",
        wall_clock_seconds=900.0,
        cumulative_budget_consumed=1,
        provenance=_provenance(),
    )
    base.update(overrides)
    return TrialRecord(**base)


class TestTrialProvenance(unittest.TestCase):
    def test_to_dict(self):
        p = _provenance()
        d = p.to_dict()
        self.assertEqual(d["proposal_id"], "prop-001")
        self.assertEqual(set(d.keys()), {"proposal_id", "patch_id", "run_id", "metric_id", "decision_id"})


class TestTrialRecord(unittest.TestCase):
    def test_valid_kept_record(self):
        r = _valid_record()
        self.assertEqual(r.decision, TrialDecision.KEPT)
        self.assertEqual(r.validity_status, ValidityStatus.VALID)
        self.assertIsNone(r.failure_category)

    def test_valid_discarded_record(self):
        r = _valid_record(
            decision=TrialDecision.DISCARDED,
            decision_rationale="Did not improve.",
            delta_vs_best=-0.01,
        )
        self.assertEqual(r.decision, TrialDecision.DISCARDED)

    def test_invalid_record_requires_failure_category(self):
        with self.assertRaises(ValueError):
            _valid_record(
                validity_status=ValidityStatus.INVALID,
                decision=TrialDecision.FAILED_INVALID,
                failure_category=None,  # missing — should raise
            )

    def test_failed_invalid_requires_invalid_validity(self):
        with self.assertRaises(ValueError):
            _valid_record(
                decision=TrialDecision.FAILED_INVALID,
                validity_status=ValidityStatus.VALID,  # mismatched
                failure_category=FailureCategory.RUNTIME_ERROR,
            )

    def test_empty_trial_id_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(trial_id="")

    def test_empty_campaign_id_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(campaign_id="")

    def test_budget_index_zero_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(budget_index=0, cumulative_budget_consumed=1)

    def test_cumulative_below_index_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(budget_index=5, cumulative_budget_consumed=3)

    def test_negative_wall_clock_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(wall_clock_seconds=-1.0)

    def test_empty_targeted_files_raises(self):
        with self.assertRaises(ValueError):
            _valid_record(targeted_files=())

    def test_to_dict_round_trip(self):
        r = _valid_record()
        d = r.to_dict()
        # Should be JSON serialisable.
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        r2 = TrialRecord.from_mapping(loaded)
        self.assertEqual(r2.trial_id, r.trial_id)
        self.assertEqual(r2.decision, r.decision)
        self.assertEqual(r2.parsed_metrics, r.parsed_metrics)

    def test_from_mapping_restores_enums(self):
        d = _valid_record().to_dict()
        d["execution_status"] = "success"
        d["validity_status"] = "valid"
        d["decision"] = "kept"
        r = TrialRecord.from_mapping(d)
        self.assertIsInstance(r.execution_status, ExecutionStatus)
        self.assertIsInstance(r.decision, TrialDecision)

    def test_from_mapping_with_failure_category(self):
        r = _valid_record(
            validity_status=ValidityStatus.INVALID,
            decision=TrialDecision.FAILED_INVALID,
            failure_category=FailureCategory.METRIC_MISSING,
            parsed_metrics={},
            delta_vs_best=None,
        )
        d = r.to_dict()
        r2 = TrialRecord.from_mapping(d)
        self.assertEqual(r2.failure_category, FailureCategory.METRIC_MISSING)

    def test_targeted_files_is_tuple(self):
        r = _valid_record()
        self.assertIsInstance(r.targeted_files, tuple)

    def test_extra_default_is_empty_dict(self):
        r = _valid_record()
        self.assertEqual(r.extra, {})


if __name__ == "__main__":
    unittest.main()
