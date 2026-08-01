"""Golden tests for the extended reliability vertical slice.

Covers the event-store adapter, integrity-hash validation, synthetic R1
endpoint derivation, manifest-bound LaTeX export, Phase 5 controller
policies, matched-compute ledger, clean-task regression gate, and
reward-channel/verifier-substitution audits.
"""

import tempfile
import unittest
from pathlib import Path

from autoresearch.reliability.schema.events import (
    ReliabilityEvent,
    ReliabilityEventType,
    ProposalEvent,
    BeliefEvent,
    StateDiffEvent,
    TerminalOutcomeEvent,
    AdjudicationEvent,
    FaultInjectionEvent,
)
from autoresearch.reliability.schema.manifest import RunManifest
from autoresearch.reliability.schema.validation import (
    validate_event_stream,
    ValidationResult,
)
from autoresearch.reliability.schema.adapter import (
    ReliabilityEventStoreAdapter,
)
from autoresearch.reliability.schema.export import (
    export_scalar_macro,
    export_latex_table,
)
from autoresearch.reliability.warning.endpoints import (
    derive_median_warning_lead_time,
)
from autoresearch.reliability.control import (
    FixedPeriodicPolicy,
    RandomMatchedBudgetPolicy,
    MyopicVoiPolicy,
    OraclePolicy,
    UNKNOWN_COUNTERFACTUAL,
    ThresholdPolicy,
    DecisionState,
    ControlAction,
    policy_registry,
)
from autoresearch.reliability.training import (
    ComputeLedger,
    ComputeLedgerRow,
    run_clean_task_regression_gate,
)
from autoresearch.reliability.adjudication import (
    audit_reward_channel,
    verifier_substitution_audit,
)


def make_manifest(run_id: str = "test-run-001") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        campaign_id="test-campaign",
        task_id="task-001",
        task_family="repo_repair",
        replicate_id="rep-01",
    )


def make_invalid_trajectory_events(run_id: str = "test-run-invalid") -> list[ReliabilityEvent]:
    proposal = ProposalEvent(
        run_id=run_id,
        proposed_action="edit_config",
        declared_objective="reduce_val_loss",
        expected_postcondition="lower_val_loss",
        monotonic_step=0,
    )
    diff = StateDiffEvent(
        run_id=run_id,
        target="config.yaml",
        before_ref="hash_before",
        after_ref="hash_after",
        diff_type="text",
        expected_change="lr_reduced",
        observed_change="lr_reduced",
        reconciled=True,
        monotonic_step=1,
    )
    belief_warning = BeliefEvent(
        run_id=run_id,
        failure_probability=0.8,
        cause_distribution={"model_error": 0.6, "tool_error": 0.4},
        abstention_state="warning_issued",
        monotonic_step=3,
    )
    belief_active = BeliefEvent(
        run_id=run_id,
        failure_probability=0.4,
        cause_distribution={"model_error": 0.5, "tool_error": 0.5},
        abstention_state="active",
        monotonic_step=4,
    )
    terminal = TerminalOutcomeEvent(
        run_id=run_id,
        task_success=True,
        scientific_validity=False,
        recovery_status="none",
        monotonic_step=5,
    )
    adjudication = AdjudicationEvent(
        run_id=run_id,
        execution_success=True,
        scientific_validity=False,
        verifier_version="v1.0",
        monotonic_step=6,
    )
    return [proposal, diff, belief_warning, belief_active, terminal, adjudication]


class TestEventStoreAdapter(unittest.TestCase):
    """Reliability events persist through the existing append-only store."""

    def test_append_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            adapter = ReliabilityEventStoreAdapter.open(path, "test-campaign")
            events = make_invalid_trajectory_events()
            for e in events:
                adapter.append(e)
            read_back = adapter.read_all()
            self.assertEqual(len(read_back), len(events))
            self.assertEqual(read_back[0].payload, events[0].payload)
            self.assertEqual(read_back[-1].event_type, ReliabilityEventType.ADJUDICATION)

    def test_append_only_preserves_all_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            adapter = ReliabilityEventStoreAdapter.open(path, "test-campaign")
            adapter.append(
                ProposalEvent(
                    run_id="r1",
                    proposed_action="a",
                    declared_objective="o",
                    expected_postcondition="p",
                    monotonic_step=0,
                )
            )
            adapter.append(
                BeliefEvent(
                    run_id="r1",
                    failure_probability=0.5,
                    cause_distribution={"x": 0.5},
                    monotonic_step=1,
                )
            )
            line_count = len(path.read_text().strip().split("\n"))
            self.assertEqual(line_count, 2)


class TestIntegrityValidation(unittest.TestCase):
    """Integrity hash, visibility class, and parent-link validation."""

    def test_valid_stream_passes(self):
        events = make_invalid_trajectory_events()
        result = validate_event_stream(events)
        self.assertTrue(result.valid)

    def test_integrity_hash_mismatch_detected(self):
        events = make_invalid_trajectory_events()
        first = events[0]
        corrupted = ReliabilityEvent(
            event_id=first.event_id,
            schema_version=first.schema_version,
            run_id=first.run_id,
            parent_event_ids=first.parent_event_ids,
            event_type=first.event_type,
            monotonic_step=first.monotonic_step,
            wall_time_utc=first.wall_time_utc,
            payload={"proposed_action": "TAMPERED"},
            visibility_class=first.visibility_class,
            integrity_hash=first.integrity_hash,
        )
        events[0] = corrupted
        result = validate_event_stream(events)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("integrity hash mismatch" in err for err in result.errors)
        )

    def test_fault_truth_visibility_enforced(self):
        events = [
            ProposalEvent(
                run_id="r1",
                proposed_action="a",
                declared_objective="o",
                expected_postcondition="p",
                monotonic_step=0,
            ),
            ReliabilityEvent.create(
                run_id="r1",
                event_type=ReliabilityEventType.FAULT_INJECTION,
                payload={
                    "fault_manifest_hash": "abc",
                    "injection_trigger": "step_5",
                    "severity": "high",
                    "persistence": "persistent",
                },
                monotonic_step=1,
                visibility_class="policy_visible",
            ),
        ]
        result = validate_event_stream(events)
        self.assertFalse(result.valid)


class TestEndpointDerivation(unittest.TestCase):
    """Synthetic R1 endpoint derived from raw events."""

    def test_median_warning_lead_time_positive(self):
        events = make_invalid_trajectory_events()
        manifest = make_manifest("test-run-invalid")
        endpoint = derive_median_warning_lead_time(
            events, manifest_hash=manifest.manifest_hash
        )
        self.assertEqual(endpoint.endpoint_id, "R1-E2-primary-warning-lead-time")
        self.assertEqual(endpoint.manifest_hash, manifest.manifest_hash)
        self.assertTrue(endpoint.value > 0)
        self.assertTrue(any("endpoint_uses_hidden_truth=False" in p for p in endpoint.provenance))

    def test_endpoint_rejects_hidden_truth(self):
        events = make_invalid_trajectory_events()
        # Endpoint must not read hidden fault truth even when a fault event is present
        events.append(
            FaultInjectionEvent(
                run_id="test-run-invalid",
                fault_manifest_hash="sealed",
                injection_trigger="x",
                severity="high",
                persistence="persistent",
                monotonic_step=2,
            )
        )
        manifest = make_manifest("test-run-invalid")
        endpoint = derive_median_warning_lead_time(
            events, manifest_hash=manifest.manifest_hash
        )
        self.assertTrue(endpoint.value > 0)


class TestManifestBoundExport(unittest.TestCase):
    """Manifest-bound LaTeX export with provenance."""

    def test_scalar_macro_export(self):
        events = make_invalid_trajectory_events()
        manifest = make_manifest("test-run-invalid")
        endpoint = derive_median_warning_lead_time(
            events, manifest_hash=manifest.manifest_hash
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated" / "endpoint_value.tex"
            provenance = export_scalar_macro(endpoint, out, manifest)
            content = out.read_text()
            self.assertIn("endpoint_id: R1-E2-primary-warning-lead-time", content)
            self.assertIn("manifest_hash:", content)
            self.assertEqual(provenance["type"], "scalar_macro")
            self.assertIn("manifest_hash", provenance)

    def test_latex_table_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated" / "results.tex"
            rows = [
                {
                    "Claim": "Failure dynamics",
                    "Effect": "0.42",
                    "Effect_artifact": "R1-E1-primary",
                },
                {
                    "Claim": "Early warning",
                    "Effect": "",
                    "Effect_artifact": "R1-E2-primary",
                },
            ]
            manifest = make_manifest("test-run-export")
            provenance = export_latex_table(
                rows,
                ["Claim", "Effect"],
                out,
                caption="Primary confirmatory results",
                label="tab:primary_results",
                manifest=manifest,
            )
            content = out.read_text()
            self.assertIn("R1-E1-primary", content)
            self.assertIn("[pending]", content)
            self.assertEqual(provenance["type"], "latex_table")
            self.assertEqual(provenance["rows"], 2)


class TestPhase5Policies(unittest.TestCase):
    """Phase 5 controller policies and the UNKNOWN counterfactual."""

    def setUp(self):
        self.state = DecisionState(
            run_id="test-run",
            step_index=7,
            failure_probability=0.3,
            cause_distribution={"model_error": 0.7, "tool_error": 0.3},
            evidence_freshness=0.5,
            budget_remaining=80.0,
            reversibility=0.9,
            estimated_damage=0.1,
            available_actions=("continue", "verify", "abstain"),
        )

    def test_fixed_periodic_policy(self):
        policy = FixedPeriodicPolicy(interval=5)
        self.assertEqual(policy.decide(self.state).selected_action, ControlAction.CONTINUE)
        step5 = DecisionState(**{**self.state.__dict__, "step_index": 10})
        self.assertEqual(policy.decide(step5).selected_action, ControlAction.VERIFY)

    def test_random_matched_budget_deterministic(self):
        p1 = RandomMatchedBudgetPolicy(verify_probability=0.5, seed=42)
        p2 = RandomMatchedBudgetPolicy(verify_probability=0.5, seed=42)
        self.assertEqual(
            p1.decide(self.state).selected_action,
            p2.decide(self.state).selected_action,
        )

    def test_myopic_voi_policy(self):
        policy = MyopicVoiPolicy(vois_threshold=10.0)
        decision = policy.decide(self.state)
        self.assertIn(
            decision.selected_action, (ControlAction.VERIFY, ControlAction.CONTINUE)
        )

    def test_oracle_is_separate_from_primary(self):
        oracle = OraclePolicy()
        self.assertTrue(oracle.is_oracle)

    def test_unknown_counterfactual_is_nan(self):
        self.assertTrue(UNKNOWN_COUNTERFACTUAL != UNKNOWN_COUNTERFACTUAL)

    def test_policy_registry_has_all_declared(self):
        registry = policy_registry()
        self.assertIn("never_verify", registry)
        self.assertIn("always_verify", registry)
        self.assertIn("fixed_periodic", registry)
        self.assertIn("random_matched_budget", registry)
        self.assertIn("threshold_r1_probability", registry)
        self.assertIn("myopic_voi", registry)
        self.assertIn("oracle_fault_aware", registry)


class TestMatchedLedgerAndRegressionGate(unittest.TestCase):
    """Matched-compute ledger and clean-task non-inferiority gate."""

    def test_matched_ledger_reports_no_mismatch(self):
        ledger = ComputeLedger()
        for arm in ["base", "recovery_sft", "process_supervised"]:
            ledger.add_row(
                ComputeLedgerRow(
                    arm_id=arm,
                    source_trajectories=1000,
                    effective_tokens=1_000_000,
                    optimizer_steps=200,
                    evaluation_budget=500,
                    verification_cost=10.0,
                )
            )
        self.assertTrue(ledger.matched)
        self.assertEqual(ledger.report_mismatches(), [])

    def test_matched_ledger_reports_mismatch(self):
        ledger = ComputeLedger()
        ledger.add_row(
            ComputeLedgerRow(arm_id="a", source_trajectories=1000, effective_tokens=1_000_000,
                             optimizer_steps=200, evaluation_budget=500, verification_cost=10.0)
        )
        ledger.add_row(
            ComputeLedgerRow(arm_id="b", source_trajectories=2000, effective_tokens=2_000_000,
                             optimizer_steps=200, evaluation_budget=500, verification_cost=10.0)
        )
        self.assertFalse(ledger.matched)
        self.assertTrue(any("source_trajectories" in m for m in ledger.report_mismatches()))

    def test_regression_gate_passes_within_margin(self):
        result = run_clean_task_regression_gate(
            arm_id="process_supervised",
            clean_task_success=0.95,
            base_model_success=0.96,
            margin=0.02,
        )
        self.assertTrue(result.passed)

    def test_regression_gate_fails_outside_margin(self):
        result = run_clean_task_regression_gate(
            arm_id="bad_arm",
            clean_task_success=0.85,
            base_model_success=0.96,
            margin=0.02,
        )
        self.assertFalse(result.passed)


class TestAuditHooks(unittest.TestCase):
    """Reward-channel and verifier-substitution audits."""

    def test_reward_channel_independent(self):
        audit = audit_reward_channel(
            training_verifier="v_train",
            evaluation_verifier="v_eval",
            reward_to_evidence_agreement=0.8,
        )
        self.assertTrue(audit.independent)

    def test_reward_channel_shared_impl_detected(self):
        audit = audit_reward_channel(
            training_verifier="v_shared",
            evaluation_verifier="v_shared",
            reward_to_evidence_agreement=0.7,
        )
        self.assertFalse(audit.independent)
        self.assertTrue(audit.shared_implementation_bug_detected)

    def test_verifier_substitution_within_tolerance(self):
        self.assertTrue(
            verifier_substitution_audit(
                primary_verifier_success=0.80,
                substitute_verifier_success=0.78,
                tolerance=0.05,
            )
        )
        self.assertFalse(
            verifier_substitution_audit(
                primary_verifier_success=0.80,
                substitute_verifier_success=0.70,
                tolerance=0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()
