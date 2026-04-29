from __future__ import annotations

import sys
import unittest
from pathlib import Path

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "nodes" / "resnet_trigger.yaml").exists():
            return parent
    raise RuntimeError("could not locate autoresearch_harness repo root")


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.control_plane.state_machine import (
    InvalidTransitionError,
    TrialState,
    transition,
)
from autoresearch.memory.schemas import (
    ExecutionStatus,
    TrialDecision,
    TrialProvenance,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.nodes.spec import load_node_spec


class Stage2ContractTests(unittest.TestCase):
    def test_resnet_node_spec_loads(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")

        self.assertEqual(spec.name, "resnet_trigger")
        self.assertEqual(spec.metric_name, "val_auc")
        self.assertEqual(spec.metric_direction, "maximize")
        self.assertEqual(spec.editable_paths, ("train.py",))
        self.assertGreaterEqual(spec.default_budget.trials, 1)
        self.assertIn("invalid_edit_scope", spec.failure_categories)

    def test_edit_scope_accepts_only_train_py(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")

        valid = validate_edit_scope(["train.py"], spec)
        self.assertTrue(valid.valid)
        self.assertEqual(valid.violations, ())

        invalid = validate_edit_scope(["prepare.py", "artifacts/best_model.pt"], spec)
        self.assertFalse(invalid.valid)
        self.assertTrue(any("prepare.py" in violation for violation in invalid.violations))
        self.assertTrue(any("artifacts" in violation for violation in invalid.violations))

    def test_state_machine_allows_only_ordered_lifecycle(self) -> None:
        state = TrialState.INITIALIZED
        for target in (
            TrialState.PROPOSED,
            TrialState.PATCH_GENERATED,
            TrialState.SCOPE_VALIDATED,
            TrialState.EXECUTED,
            TrialState.METRIC_PARSED,
            TrialState.EVALUATED,
            TrialState.KEPT,
        ):
            state = transition(state, target)

        self.assertEqual(state, TrialState.KEPT)
        with self.assertRaises(InvalidTransitionError):
            transition(TrialState.PROPOSED, TrialState.EXECUTED)

    def test_trial_record_round_trips(self) -> None:
        record = TrialRecord(
            trial_id="trial-001",
            campaign_id="campaign-smoke",
            node_id="resnet_trigger",
            budget_index=1,
            timestamp_start="2026-04-28T00:00:00Z",
            timestamp_end="2026-04-28T00:01:00Z",
            manager_mode="baseline_manager",
            worker_mode="claw_style_worker",
            memory_mode="append_only_summary",
            proposal_summary="Lower learning rate",
            proposal_rationale="Prior run was unstable.",
            targeted_files=("train.py",),
            patch_ref="experiments/artifacts/trial-001/patch.diff",
            git_commit_before="abc123",
            git_commit_after="def456",
            execution_status=ExecutionStatus.SUCCESS,
            validity_status=ValidityStatus.VALID,
            failure_category=None,
            raw_log_ref="experiments/artifacts/trial-001/run.log",
            parsed_metrics={"val_auc": 0.7876},
            current_best_before=0.7799,
            delta_vs_best=0.0077,
            decision=TrialDecision.KEPT,
            decision_rationale="Validation AUC improved.",
            wall_clock_seconds=60.0,
            cumulative_budget_consumed=1,
            provenance=TrialProvenance(
                proposal_id="proposal-001",
                patch_id="patch-001",
                run_id="run-001",
                metric_id="metric-001",
                decision_id="decision-001",
            ),
        )

        payload = record.to_dict()
        loaded = TrialRecord.from_mapping(payload)
        self.assertEqual(loaded.trial_id, record.trial_id)
        self.assertEqual(loaded.parsed_metrics["val_auc"], 0.7876)
        self.assertEqual(loaded.decision, TrialDecision.KEPT)


if __name__ == "__main__":
    unittest.main()
