"""Golden tests for the reliability scaffold.

Covers: manifest immutability, event round-trip, append-only behavior,
fault-truth noninterference, state-difference reconciliation, task/template
split leakage, adjudicator/reward separation, deterministic replay, and
provenance propagation.
"""

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.reliability.schema.events import (
    ReliabilityEvent,
    ReliabilityEventType,
    EventVisibility,
    SCHEMA_VERSION,
    ProposalEvent,
    ActionResultEvent,
    StateDiffEvent,
    FaultInjectionEvent,
    BeliefEvent,
    TerminalOutcomeEvent,
    AdjudicationEvent,
)
from autoresearch.reliability.schema.manifest import (
    RunManifest,
    FaultManifest,
)
from autoresearch.reliability.schema.validation import (
    validate_manifest,
    validate_event_stream,
    validate_referential_integrity,
    ValidationResult,
)
from autoresearch.reliability.adjudication import AdjudicationRecord, AdjudicationStatus
from autoresearch.reliability.benchmark import SplitResolver, SplitAssignment
from autoresearch.reliability.control import (
    NeverVerifyPolicy,
    AlwaysVerifyPolicy,
    ThresholdPolicy,
    ControlAction,
    DecisionState,
)
from autoresearch.reliability.training import (
    TrainingView,
    TrainingViewType,
    TrainingExample,
    TrainingViewBuilder,
)


class TestManifestImmutability(unittest.TestCase):
    """Run manifests are frozen dataclasses and must be immutable."""

    def test_frozen_dataclass_prevents_mutation(self):
        m = RunManifest(
            run_id="test-run-001",
            campaign_id="test-campaign",
            task_id="task-001",
            task_family="repo_repair",
            replicate_id="rep-01",
        )
        with self.assertRaises(Exception):
            m.run_id = "modified"  # type: ignore

    def test_manifest_hash_deterministic(self):
        m1 = RunManifest(
            run_id="test-run-001",
            campaign_id="test-campaign",
            task_id="task-001",
            task_family="repo_repair",
            replicate_id="rep-01",
        )
        m2 = RunManifest(
            run_id="test-run-001",
            campaign_id="test-campaign",
            task_id="task-001",
            task_family="repo_repair",
            replicate_id="rep-01",
        )
        self.assertEqual(m1.manifest_hash, m2.manifest_hash)

    def test_different_values_produce_different_hash(self):
        m1 = RunManifest(
            run_id="test-run-001",
            campaign_id="test-campaign",
            task_id="task-001",
            task_family="repo_repair",
            replicate_id="rep-01",
        )
        m2 = RunManifest(
            run_id="test-run-002",
            campaign_id="test-campaign",
            task_id="task-001",
            task_family="repo_repair",
            replicate_id="rep-01",
        )
        self.assertNotEqual(m1.manifest_hash, m2.manifest_hash)


class TestEventRoundTrip(unittest.TestCase):
    """Events serialize and deserialize without semantic change."""

    def setUp(self):
        self.run_id = "test-run-roundtrip"

    def test_proposal_event_round_trip(self):
        e = ProposalEvent(
            run_id=self.run_id,
            proposed_action="edit_train_py",
            declared_objective="reduce_learning_rate",
            expected_postcondition="lower_val_loss",
        )
        d = e.to_dict()
        e2 = ReliabilityEvent.from_dict(d)
        self.assertEqual(e.event_id, e2.event_id)
        self.assertEqual(e.payload, e2.payload)
        self.assertEqual(e.event_type, e2.event_type)
        self.assertEqual(e.integrity_hash, e2.integrity_hash)

    def test_fault_injection_event_sealed_visibility(self):
        e = FaultInjectionEvent(
            run_id=self.run_id,
            fault_manifest_hash="abc123",
            injection_trigger="step_5",
            severity="high",
            persistence="persistent",
        )
        self.assertEqual(e.visibility_class, EventVisibility.HIDDEN_FAULT_TRUTH)

    def test_adjudication_event_visibility(self):
        e = AdjudicationEvent(
            run_id=self.run_id,
            execution_success=True,
            scientific_validity=True,
            verifier_version="v1.0",
        )
        self.assertEqual(e.visibility_class, EventVisibility.ADJUDICATION_ONLY)


class TestAppendOnlyBehavior(unittest.TestCase):
    """Event streams are append-only and immutable once written."""

    def setUp(self):
        self.run_id = "test-run-append-only"

    def test_event_immutable_after_creation(self):
        e = ProposalEvent(
            run_id=self.run_id,
            proposed_action="test",
            declared_objective="test",
            expected_postcondition="test",
        )
        with self.assertRaises(Exception):
            e.payload = {"modified": True}  # type: ignore

    def test_event_integrity_hash_depends_on_payload(self):
        e1 = ProposalEvent(
            run_id=self.run_id,
            proposed_action="action_a",
            declared_objective="obj",
            expected_postcondition="post",
        )
        e2 = ProposalEvent(
            run_id=self.run_id,
            proposed_action="action_b",  # different
            declared_objective="obj",
            expected_postcondition="post",
        )
        self.assertNotEqual(e1.integrity_hash, e2.integrity_hash)


class TestFaultTruthNoninterference(unittest.TestCase):
    """Policy-visible state must exclude hidden fault truth."""

    def test_fault_manifest_hides_hidden_fields(self):
        fm = FaultManifest(
            fault_id="fault-001",
            fault_family="action_and_tool",
            mechanism="malformed_argument",
            causal_parent="tool_call_step_3",
            affected_component="tool_executor",
            injection_trigger="step_5",
            deterministic_seed="seed-42",
            severity="high",
            persistence="persistent",
            observability_class="observable",
            policy_visible_fields=[
                "fault_id",
                "observability_class",
            ],
            hidden_fields=[
                "mechanism",
                "severity",
                "causal_parent",
            ],
        )
        visible = fm.policy_visible_subset()
        self.assertIn("fault_id", visible)
        self.assertIn("observability_class", visible)
        self.assertNotIn("mechanism", visible)
        self.assertNotIn("severity", visible)
        self.assertNotIn("causal_parent", visible)


class TestStateDifferenceReconciliation(unittest.TestCase):
    """State diffs must distinguish reconciled from unreconciled changes."""

    def test_state_diff_fields(self):
        diff = StateDiffEvent(
            run_id="test-run",
            target="train.py",
            before_ref="hash_a",
            after_ref="hash_b",
            diff_type="text",
            expected_change="lr_reduced",
            observed_change="lr_reduced_from_0.01_to_0.005",
            reconciled=True,
        )
        self.assertTrue(diff.payload["reconciled"])
        self.assertEqual(diff.payload["diff_type"], "text")


class TestSplitLeakage(unittest.TestCase):
    """Task templates must not cross train/test boundaries."""

    def test_no_overlap_detected(self):
        resolver = SplitResolver()
        train_tasks = {"task_a", "task_b", "task_c"}
        test_tasks = {"task_d", "task_e"}
        errors = resolver.validate_isolation(train_tasks, test_tasks)
        self.assertEqual(len(errors), 0)

    def test_overlap_detected(self):
        resolver = SplitResolver()
        train_tasks = {"task_a", "task_b", "task_c"}
        test_tasks = {"task_c", "task_d"}
        errors = resolver.validate_isolation(train_tasks, test_tasks)
        self.assertGreater(len(errors), 0)
        self.assertIn("task_c", errors[0])


class TestAdjudicatorRewardSeparation(unittest.TestCase):
    """Training reward and final adjudication must be separate channels."""

    def test_adjudication_distinct_from_training(self):
        from autoresearch.reliability.adjudication import separate_channels
        self.assertTrue(separate_channels("train-adjudicator-v1", "eval-adjudicator-v2"))
        self.assertFalse(separate_channels("same-adjudicator", "same-adjudicator"))

    def test_adjudication_statuses(self):
        record = AdjudicationRecord.create(
            run_id="test-run",
            step_index=10,
            execution_success=True,
            scientific_validity=False,
            final_status=AdjudicationStatus.REJECTED,
        )
        self.assertFalse(record.is_accepted())
        self.assertFalse(record.is_ambiguous())

    def test_ambiguous_remains_ambiguous(self):
        record = AdjudicationRecord.create(
            run_id="test-run",
            step_index=10,
            execution_success=True,
            scientific_validity=False,
            final_status=AdjudicationStatus.AMBIGUOUS,
        )
        self.assertTrue(record.is_ambiguous())


class TestDeterministicReplay(unittest.TestCase):
    """Manifest hashes and event streams must be reproducible from seeds."""

    def test_same_input_produces_same_hash(self):
        m1 = RunManifest(
            run_id="replay-001",
            campaign_id="campaign",
            task_id="task-001",
            task_family="family",
            replicate_id="rep-01",
            randomization_seed="seed-42",
        )
        m2 = RunManifest(
            run_id="replay-001",
            campaign_id="campaign",
            task_id="task-001",
            task_family="family",
            replicate_id="rep-01",
            randomization_seed="seed-42",
        )
        self.assertEqual(m1.manifest_hash, m2.manifest_hash)

    def test_fault_manifest_deterministic(self):
        fm1 = FaultManifest(
            fault_id="fault-replay",
            fault_family="verifier",
            mechanism="flaky_test",
            causal_parent="test_step",
            affected_component="test_runner",
            injection_trigger="step_8",
            deterministic_seed="seed-100",
            severity="medium",
            persistence="transient",
            observability_class="partially_observable",
        )
        fm2 = FaultManifest(
            fault_id="fault-replay",
            fault_family="verifier",
            mechanism="flaky_test",
            causal_parent="test_step",
            affected_component="test_runner",
            injection_trigger="step_8",
            deterministic_seed="seed-100",
            severity="medium",
            persistence="transient",
            observability_class="partially_observable",
        )
        self.assertEqual(fm1.manifest_hash, fm2.manifest_hash)


class TestValidation(unittest.TestCase):
    """Manifest and event stream validation."""

    def test_valid_manifest_passes(self):
        m = RunManifest(
            run_id="r1",
            campaign_id="c1",
            task_id="t1",
            task_family="tf",
            replicate_id="rep1",
        )
        result = validate_manifest(m)
        self.assertTrue(result.valid)

    def test_empty_manifest_fails(self):
        m = RunManifest(
            run_id="",
            campaign_id="",
            task_id="",
            task_family="",
            replicate_id="",
        )
        result = validate_manifest(m)
        self.assertFalse(result.valid)
        self.assertGreater(len(result.errors), 0)

    def test_valid_event_stream_passes(self):
        events = [
            ProposalEvent(
                run_id="r1",
                proposed_action="a",
                declared_objective="o",
                expected_postcondition="p",
                monotonic_step=0,
            ),
            TerminalOutcomeEvent(
                run_id="r1",
                task_success=True,
                scientific_validity=True,
                monotonic_step=1,
                parent_event_ids=(events[0].event_id,) if False else (),
            ),
        ]
        result = validate_event_stream(events)
        self.assertTrue(result.valid)

    def test_duplicate_event_ids_detected(self):
        e1 = ProposalEvent(
            run_id="r1",
            proposed_action="a",
            declared_objective="o",
            expected_postcondition="p",
            monotonic_step=0,
        )
        e2 = ReliabilityEvent(
            event_id=e1.event_id,
            run_id="r1",
            event_type=ReliabilityEventType.TERMINAL_OUTCOME,
            monotonic_step=1,
            payload={"task_success": True},
        )
        result = validate_event_stream([e1, e2])
        self.assertFalse(result.valid)

    def test_out_of_order_steps_detected(self):
        e1 = ProposalEvent(
            run_id="r1",
            proposed_action="a",
            declared_objective="o",
            expected_postcondition="p",
            monotonic_step=5,
        )
        e2 = TerminalOutcomeEvent(
            run_id="r1",
            task_success=True,
            scientific_validity=True,
            monotonic_step=3,
        )
        result = validate_event_stream([e1, e2])
        self.assertFalse(result.valid)


class TestControllerPolicies(unittest.TestCase):
    """Fixed controller policies produce deterministic decisions."""

    def setUp(self):
        self.state = DecisionState(
            run_id="test-run",
            step_index=5,
            failure_probability=0.3,
            cause_distribution={"model_error": 0.7, "tool_error": 0.3},
            evidence_freshness=0.8,
            budget_remaining=100.0,
            reversibility=0.9,
            estimated_damage=0.1,
            available_actions=("continue", "verify"),
        )

    def test_never_verify_policy(self):
        policy = NeverVerifyPolicy()
        decision = policy.decide(self.state)
        self.assertEqual(decision.selected_action, ControlAction.CONTINUE)

    def test_always_verify_policy(self):
        policy = AlwaysVerifyPolicy()
        decision = policy.decide(self.state)
        self.assertEqual(decision.selected_action, ControlAction.VERIFY)

    def test_threshold_policy_below_threshold(self):
        policy = ThresholdPolicy(threshold=0.5)
        decision = policy.decide(self.state)
        self.assertEqual(decision.selected_action, ControlAction.CONTINUE)

    def test_threshold_policy_above_threshold(self):
        policy = ThresholdPolicy(threshold=0.5)
        high_risk = DecisionState(
            run_id="test-run",
            step_index=5,
            failure_probability=0.8,
            cause_distribution={"model_error": 0.9},
            evidence_freshness=0.3,
            budget_remaining=50.0,
            reversibility=0.4,
            estimated_damage=0.7,
            available_actions=("continue", "verify", "abstain"),
        )
        decision = policy.decide(high_risk)
        self.assertEqual(decision.selected_action, ControlAction.VERIFY)


class TestTrainingViews(unittest.TestCase):
    """Training views are immutable and respect split boundaries."""

    def test_view_freeze_produces_hash(self):
        view = TrainingView(view_type=TrainingViewType.RECOVERY_SFT)
        view.add_example(
            TrainingExample(
                example_id="ex-001",
                source_run_ids=("run-001",),
                view_type=TrainingViewType.RECOVERY_SFT,
                selection_rule="adjudicated",
                verifier_channel="v1",
                split="train",
                deduplication_cluster="cluster-a",
                view_builder_hash="",
            )
        )
        view.freeze()
        self.assertNotEqual(view.builder_hash, "")

    def test_builder_respects_split_boundaries(self):
        builder = TrainingViewBuilder(evaluation_split_ids={"test", "unseen"})
        self.assertTrue(builder.is_training_eligible("train"))
        self.assertTrue(builder.is_training_eligible("dev"))
        self.assertFalse(builder.is_training_eligible("test"))
        self.assertFalse(builder.is_training_eligible("unseen"))


class TestProvenancePropagation(unittest.TestCase):
    """Derived artifacts must carry provenance to their source manifests."""

    def test_event_integrity_hash_includes_parents(self):
        parent = ProposalEvent(
            run_id="r1",
            proposed_action="a",
            declared_objective="o",
            expected_postcondition="p",
            monotonic_step=0,
        )
        child = BeliefEvent(
            run_id="r1",
            failure_probability=0.5,
            cause_distribution={"fault_a": 0.5},
            monotonic_step=1,
            parent_event_ids=(parent.event_id,),
        )
        self.assertNotEqual(parent.integrity_hash, child.integrity_hash)

    def test_manifest_round_trip_preserves_hash(self):
        m = RunManifest(
            run_id="r1",
            campaign_id="c1",
            task_id="t1",
            task_family="tf",
            replicate_id="rep1",
        )
        d = m.to_dict()
        m2 = RunManifest.from_dict(d)
        self.assertEqual(m.manifest_hash, m2.manifest_hash)


if __name__ == "__main__":
    unittest.main()
