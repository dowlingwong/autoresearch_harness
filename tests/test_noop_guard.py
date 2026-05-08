"""Tests for the no-op patch guard (Chunk 1.3).

Verifies that the control plane marks a successful-but-empty-patch worker
result as ``failed_invalid / no_op_patch`` rather than treating it as a
valid trial.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from autoresearch.control_plane.campaign import _patch_is_empty, _record_from_worker_result
from autoresearch.memory.schemas import FailureCategory, TrialDecision, ValidityStatus
from autoresearch.nodes.spec import NodeSpec
from autoresearch.worker.base import WorkerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node_spec(editable: tuple[str, ...] = ("train.py",)) -> NodeSpec:
    return NodeSpec(
        name="test_node",
        description="test",
        editable_paths=editable,
        frozen_paths=(),
        setup_command="",
        run_command="echo ok",
        metric_name="val_auc",
        metric_direction="maximize",
        metric_parser="",
        acceptance_rule="candidate_metric > current_best_metric",
        validity_checks=(),
        default_budget=MagicMock(trials=5),
        expected_runtime="",
        failure_categories=(),
    )


def _make_worker_result(patch_ref: str, success: bool = True, metric: float | None = 0.80) -> WorkerResult:
    metrics = {"val_auc": metric} if metric is not None else {}
    return WorkerResult(
        worker_mode="test_worker",
        changed_files=("train.py",),
        success=success,
        parsed_metrics=metrics,
        raw_log_ref="",
        patch_ref=patch_ref,
        git_commit_before="abc",
        git_commit_after="def",
    )


# ---------------------------------------------------------------------------
# _patch_is_empty unit tests
# ---------------------------------------------------------------------------

class TestPatchIsEmpty(unittest.TestCase):

    def test_blank_string_is_empty(self) -> None:
        self.assertTrue(_patch_is_empty(""))

    def test_whitespace_only_string_is_empty(self) -> None:
        self.assertTrue(_patch_is_empty("   "))

    def test_nonexistent_path_is_not_empty(self) -> None:
        """Synthetic dry-run paths that don't exist should return False."""
        self.assertFalse(_patch_is_empty("/nonexistent/path/patch.diff"))

    def test_existing_empty_file_is_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False) as f:
            path = Path(f.name)
        path.write_text("")
        try:
            self.assertTrue(_patch_is_empty(str(path)))
        finally:
            path.unlink(missing_ok=True)

    def test_existing_whitespace_file_is_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False) as f:
            path = Path(f.name)
        path.write_text("\n\n  \n")
        try:
            self.assertTrue(_patch_is_empty(str(path)))
        finally:
            path.unlink(missing_ok=True)

    def test_existing_real_patch_is_not_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False, mode="w") as f:
            f.write("--- a/train.py\n+++ b/train.py\n@@ -1,1 +1,1 @@\n-x=1\n+x=2\n")
            path = Path(f.name)
        try:
            self.assertFalse(_patch_is_empty(str(path)))
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Control-plane integration: no-op produces failed_invalid / no_op_patch
# ---------------------------------------------------------------------------

class TestNoOpGuardInControlPlane(unittest.TestCase):

    def _call_record(self, patch_ref: str, metric: float | None = 0.80) -> object:
        node_spec = _make_node_spec()
        worker_result = _make_worker_result(patch_ref=patch_ref, success=True, metric=metric)
        return _record_from_worker_result(
            campaign_id="test_campaign",
            budget_index=1,
            node_spec=node_spec,
            manager_mode="baseline_manager",
            memory_mode="none",
            proposal_summary="increase lr",
            proposal_rationale="test",
            worker_result=worker_result,
            current_best=None,
        )

    def test_empty_patch_ref_produces_noop_failure(self) -> None:
        record = self._call_record(patch_ref="")
        self.assertEqual(record.validity_status, ValidityStatus.INVALID)
        self.assertEqual(record.failure_category, FailureCategory.NO_OP_PATCH)
        self.assertEqual(record.decision, TrialDecision.FAILED_INVALID)

    def test_empty_file_produces_noop_failure(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False) as f:
            path = Path(f.name)
        path.write_text("")
        try:
            record = self._call_record(patch_ref=str(path))
            self.assertEqual(record.failure_category, FailureCategory.NO_OP_PATCH)
            self.assertEqual(record.decision, TrialDecision.FAILED_INVALID)
        finally:
            path.unlink(missing_ok=True)

    def test_real_patch_file_is_not_noop(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".diff", delete=False, mode="w") as f:
            f.write("--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-lr=0.01\n+lr=0.001\n")
            path = Path(f.name)
        try:
            record = self._call_record(patch_ref=str(path))
            # Should be VALID and KEPT (beats None baseline)
            self.assertEqual(record.validity_status, ValidityStatus.VALID)
            self.assertIsNone(record.failure_category)
            self.assertEqual(record.decision, TrialDecision.KEPT)
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_synthetic_path_is_not_noop(self) -> None:
        """Dry-run workers return non-existent path strings — must not trigger no-op guard."""
        record = self._call_record(patch_ref="experiments/artifacts/dry_run_trial_001/patch.diff")
        self.assertEqual(record.validity_status, ValidityStatus.VALID)
        self.assertIsNone(record.failure_category)

    def test_failed_worker_ignores_noop_check(self) -> None:
        """If the worker itself failed, the failure should be RUNTIME_ERROR, not NO_OP_PATCH."""
        node_spec = _make_node_spec()
        worker_result = _make_worker_result(patch_ref="", success=False, metric=None)
        record = _record_from_worker_result(
            campaign_id="test_campaign",
            budget_index=1,
            node_spec=node_spec,
            manager_mode="baseline_manager",
            memory_mode="none",
            proposal_summary="increase lr",
            proposal_rationale="test",
            worker_result=worker_result,
            current_best=None,
        )
        # success=False means no_op check short-circuits; worker failure takes precedence
        self.assertEqual(record.failure_category, FailureCategory.RUNTIME_ERROR)


if __name__ == "__main__":
    unittest.main()
