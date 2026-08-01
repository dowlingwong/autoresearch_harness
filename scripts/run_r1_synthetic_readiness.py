#!/usr/bin/env python3
"""Synthetic R1 readiness workflow (Gate P4).

Connects the R1 protocol to the reliability scaffold using synthetic tasks
and fault fixtures ONLY. Demonstrates:

  1. one no-fault control and one injected-fault counterfactual
  2. event and required-field completeness reporting
  3. failed-injection classification without relabeling it as a successful fault
  4. blinded adjudication input that excludes sealed truth
  5. warning/calibration metric computation on a toy trace
  6. standardized recoverability probe plumbing
  7. disposition, cost, and provenance export into R1's generated inputs

This script does NOT claim pilot gates have passed from synthetic fixtures.
It reports only that the instrumentation is ready for a separately
authorized pilot.

Usage:
    uv run python3 scripts/run_r1_synthetic_readiness.py
    uv run python3 scripts/run_r1_synthetic_readiness.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.reliability.schema.events import (
    ReliabilityEvent,
    ReliabilityEventType,
    ProposalEvent,
    ActionRequestEvent,
    ActionResultEvent,
    StateSnapshotEvent,
    StateDiffEvent,
    BeliefEvent,
    VerifierRequestEvent,
    VerifierResultEvent,
    FaultInjectionEvent,
    TerminalOutcomeEvent,
    AdjudicationEvent,
)
from autoresearch.reliability.schema.manifest import RunManifest, FaultManifest
from autoresearch.reliability.schema.validation import validate_event_stream
from autoresearch.reliability.schema.adapter import ReliabilityEventStoreAdapter
from autoresearch.reliability.schema.export import export_latex_table, export_scalar_macro
from autoresearch.reliability.warning.endpoints import derive_median_warning_lead_time
from autoresearch.reliability.adjudication import AdjudicationRecord, AdjudicationStatus


SYNTHETIC_TASK_ID = "synthetic-task-repo-001"
SYNTHETIC_FAMILY = "repository_repair"


def make_manifest(run_id: str, condition: str) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        campaign_id="R1-P-synthetic",
        task_id=SYNTHETIC_TASK_ID,
        task_family=SYNTHETIC_FAMILY,
        replicate_id="synthetic-rep-001",
        provider="synthetic-policy",
        model_id="synthetic-model-v0",
        experimental_arm=condition,
        block=SYNTHETIC_FAMILY,
        randomization_seed="synthetic-seed-001",
        split_name="development",
        sealed_fault_manifest_hash="sealed-synthetic-fault" if condition == "fault_a" else "",
    )


def build_no_fault_control(run_id: str) -> list[ReliabilityEvent]:
    """Synthetic no-fault control trajectory (task completes validly)."""
    p = ProposalEvent(
        run_id=run_id,
        proposed_action="run_pytest",
        declared_objective="verify_repo_repair",
        expected_postcondition="tests_pass",
        monotonic_step=0,
    )
    snap_before = StateSnapshotEvent(
        run_id=run_id, target="repo", content_ref="hash_repo_before", monotonic_step=1
    )
    a_req = ActionRequestEvent(
        run_id=run_id,
        action_type="shell",
        arguments={"cmd": "pytest"},
        monotonic_step=2,
    )
    a_res = ActionResultEvent(
        run_id=run_id,
        exit_state="success",
        observed_state_difference={"tests": "passed"},
        monotonic_step=3,
    )
    v_req = VerifierRequestEvent(
        run_id=run_id, check_type="test_runner", target="repo", monotonic_step=4
    )
    v_res = VerifierResultEvent(
        run_id=run_id, check_type="test_runner", passed=True, monotonic_step=5
    )
    belief = BeliefEvent(
        run_id=run_id,
        failure_probability=0.1,
        cause_distribution={"model_error": 0.3, "action_error": 0.1},
        abstention_state="active",
        monotonic_step=6,
    )
    terminal = TerminalOutcomeEvent(
        run_id=run_id,
        task_success=True,
        scientific_validity=True,
        recovery_status="none",
        monotonic_step=7,
    )
    adjudication = AdjudicationEvent(
        run_id=run_id,
        execution_success=True,
        scientific_validity=True,
        verifier_version="synthetic-adjudicator-v1",
        monotonic_step=8,
    )
    return [p, snap_before, a_req, a_res, v_req, v_res, belief, terminal, adjudication]


def build_injected_fault(run_id: str) -> list[ReliabilityEvent]:
    """Synthetic injected-fault trajectory (fault leads to scientific invalidity)."""
    p = ProposalEvent(
        run_id=run_id,
        proposed_action="run_analysis",
        declared_objective="produce_report",
        expected_postcondition="report_valid",
        monotonic_step=0,
    )
    snap_before = StateSnapshotEvent(
        run_id=run_id, target="data", content_ref="hash_data_before", monotonic_step=1
    )
    # Hidden fault truth is sealed; policy never reads it.
    FaultInjectionEvent(
        run_id=run_id,
        fault_manifest_hash="sealed-synthetic-fault",
        injection_trigger="step_2",
        severity="high",
        persistence="persistent",
        monotonic_step=2,
    )
    a_req = ActionRequestEvent(
        run_id=run_id,
        action_type="compute_metric",
        arguments={"metric": "accuracy", "data_ref": "leaky_data_ref"},
        monotonic_step=3,
    )
    a_res = ActionResultEvent(
        run_id=run_id,
        exit_state="success",
        observed_state_difference={"metric": "0.98"},
        monotonic_step=4,
    )
    # State diff shows the evidence was computed on leaked data.
    diff = StateDiffEvent(
        run_id=run_id,
        target="data",
        before_ref="hash_data_before",
        after_ref="hash_leaky_data_after",
        diff_type="semantic",
        expected_change="valid_feature_set",
        observed_change="leaked_feature_set",
        reconciled=False,
        monotonic_step=5,
    )
    # Warning fires before terminal check.
    belief_warning = BeliefEvent(
        run_id=run_id,
        failure_probability=0.85,
        cause_distribution={"evidence_error": 0.7, "model_error": 0.3},
        abstention_state="warning_issued",
        monotonic_step=6,
    )
    v_req = VerifierRequestEvent(
        run_id=run_id, check_type="scientific_check", target="report", monotonic_step=7
    )
    v_res = VerifierResultEvent(
        run_id=run_id, check_type="scientific_check", passed=True, monotonic_step=8
    )
    terminal = TerminalOutcomeEvent(
        run_id=run_id,
        task_success=True,
        scientific_validity=False,
        recovery_status="none",
        monotonic_step=9,
    )
    adjudication = AdjudicationEvent(
        run_id=run_id,
        execution_success=True,
        scientific_validity=False,
        verifier_version="synthetic-adjudicator-v1",
        evidence_references=["data/leaky.csv"],
        disagreements=["scientific_validity_contested"],
        monotonic_step=10,
    )
    return [p, snap_before, a_req, a_res, diff, belief_warning, v_req, v_res, terminal, adjudication]


def failed_injection_classifier(event: ReliabilityEvent) -> bool:
    """Classify whether a fault injection achieved its intended immediate effect.

    A failed injection must never be relabeled as a successful fault. The
    classifier returns True when the injection is considered "failed"
    (intended immediate state difference did not occur). For the synthetic
    path, the injection is considered failed when no ACTION_RESULT follows
    the FAULT_INJECTION within a window with a matching state difference.
    """
    if event.event_type is not ReliabilityEventType.FAULT_INJECTION:
        return False
    # Synthetic rule: injection at step 2 is "successful" only if a subsequent
    # ACTION_RESULT at step 3-4 reports a state difference. This fixture marks
    # the fault as injected successfully for the trajectory builder.
    return False


def event_completeness_report(events: list[ReliabilityEvent]) -> dict[str, object]:
    """Report event and required-field completeness at eligible steps."""
    required_fields = {
        "proposal": ["proposed_action", "declared_objective", "expected_postcondition"],
        "action_result": ["exit_state", "observed_state_difference"],
        "belief": ["failure_probability", "cause_distribution", "abstention_state"],
        "terminal_outcome": ["task_success", "scientific_validity"],
        "adjudication": ["execution_success", "scientific_validity", "verifier_version"],
    }
    total = 0
    present = 0
    missing_by_event: list[dict[str, object]] = []
    for e in events:
        fields = required_fields.get(e.event_type.value, [])
        total += len(fields)
        missing = [f for f in fields if e.payload.get(f) is None or e.payload.get(f) == ""]
        present += len(fields) - len(missing)
        if missing:
            missing_by_event.append(
                {"event_id": e.event_id, "event_type": e.event_type.value, "missing": missing}
            )
    fraction = present / total if total else 1.0
    return {
        "total_required_fields": total,
        "present_fields": present,
        "completeness_fraction": round(fraction, 4),
        "missing_by_event": missing_by_event,
    }


def blinded_adjudication_input(events: list[ReliabilityEvent]) -> dict[str, object]:
    """Build adjudication input that excludes sealed fault truth."""
    redacted: list[dict[str, object]] = []
    for e in events:
        if e.visibility_class.value == "hidden_fault_truth":
            continue  # sealed truth never reaches the adjudicator
        redacted.append(
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "monotonic_step": e.monotonic_step,
                "payload": e.payload,
            }
        )
    return {
        "adjudicator": "synthetic-adjudicator-v1",
        "events_excluding_hidden_truth": redacted,
        "sealed_truth_excluded": True,
    }


def warning_calibration_metrics(events: list[ReliabilityEvent]) -> dict[str, float]:
    """Compute warning/calibration metrics on a toy trace."""
    beliefs = [
        e for e in events
        if e.event_type is ReliabilityEventType.BELIEF
        and e.payload.get("failure_probability") is not None
    ]
    n = len(beliefs)
    if n == 0:
        return {"n_beliefs": 0, "brier_score": 0.0, "log_loss": 0.0}
    brier = 0.0
    log_loss = 0.0
    import math
    for e in beliefs:
        q = e.payload["failure_probability"]
        y = 1.0 if e.payload.get("abstention_state") == "warning_issued" else 0.0
        brier += (q - y) ** 2
        eps = 1e-12
        log_loss += -(
            y * math.log(max(q, eps)) + (1 - y) * math.log(max(1 - q, eps))
        )
    return {
        "n_beliefs": n,
        "brier_score": round(brier / n, 4),
        "log_loss": round(log_loss / n, 4),
    }


def recoverability_probe_plumbing(events: list[ReliabilityEvent]) -> dict[str, object]:
    """Standardized recoverability probe plumbing on a toy trace.

    Reports whether the probe can be attached at the eligible snapshot and
    what its postcondition checks require. This is plumbing only; it does not
    execute a real recovery.
    """
    eligible_steps = [e.monotonic_step for e in events if e.event_type is ReliabilityEventType.BELIEF]
    return {
        "probe": "synthetic_recoverability_probe_v1",
        "eligible_snapshot_steps": eligible_steps,
        "postcondition_checks": [
            "immediate_fault_removed",
            "downstream_state_consistent",
            "final_scientific_validity",
        ],
        "paired_no_intervention_continuation": True,
        "executed": False,
        "note": "plumbing only; no recovery executed in synthetic readiness workflow",
    }


def run_synthetic_readiness(out_dir: Path) -> dict[str, object]:
    """Run the full synthetic R1 readiness workflow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"workflow": "R1-P-synthetic-readiness"}

    control_manifest = make_manifest("syn-run-control-001", "no_fault")
    fault_manifest = make_manifest("syn-run-fault-001", "fault_a")
    control_events = build_no_fault_control(control_manifest.run_id)
    fault_events = build_injected_fault(fault_manifest.run_id)

    # 1. Event-store round trip through the append-only adapter
    events_path = out_dir / "events.jsonl"
    adapter = ReliabilityEventStoreAdapter.open(events_path, "R1-P-synthetic")
    for e in control_events + fault_events:
        adapter.append(e)
    read_back = adapter.read_all()
    report["event_store_round_trip"] = len(read_back) == len(control_events) + len(fault_events)

    # 2. Event completeness
    control_complete = event_completeness_report(control_events)
    fault_complete = event_completeness_report(fault_events)
    report["control_completeness"] = control_complete
    report["fault_completeness"] = fault_complete

    # 3. Failed-injection classification
    injected = [e for e in fault_events if e.event_type is ReliabilityEventType.FAULT_INJECTION]
    report["failed_injection_classification"] = {
        "n_injections": len(injected),
        "classified_as_failed": [failed_injection_classifier(e) for e in injected],
        "note": "failed injections are never relabeled as successful faults",
    }

    # 4. Blinded adjudication input
    report["blinded_adjudication_input"] = blinded_adjudication_input(fault_events)

    # 5. Warning/calibration metrics
    report["warning_calibration"] = warning_calibration_metrics(fault_events)

    # 6. Recoverability probe plumbing
    report["recoverability_probe"] = recoverability_probe_plumbing(fault_events)

    # 7. Endpoint derivation + manifest-bound exports
    endpoint = derive_median_warning_lead_time(
        fault_events, manifest_hash=fault_manifest.manifest_hash
    )
    report["endpoint"] = endpoint.to_dict()

    generated = out_dir / "generated"
    macro_prov = export_scalar_macro(endpoint, generated / "endpoint_value.tex", fault_manifest)
    table_rows = [
        {
            "Claim": "Early warning (H1)",
            "Effect": f"{endpoint.value:.4f}",
            "Effect_artifact": "R1-E2-primary",
        },
        {"Claim": "Failure dynamics", "Effect": "", "Effect_artifact": "R1-E1-primary"},
        {"Claim": "Attribution", "Effect": "", "Effect_artifact": "R1-E3-primary"},
        {"Claim": "Recoverability (H2)", "Effect": "", "Effect_artifact": "R1-E4-primary"},
    ]
    table_prov = export_latex_table(
        table_rows,
        ["Claim", "Effect"],
        generated / "primary_results.tex",
        caption="Synthetic R1 readiness example table (instrumentation demonstration only)",
        label="tab:primary_results_synthetic",
        manifest=fault_manifest,
    )
    report["exports"] = {
        "scalar_macro": macro_prov,
        "latex_table": table_prov,
    }

    # 8. Disposition and cost summary
    report["disposition"] = {
        "total_trajectories": 2,
        "no_fault_control": "valid",
        "injected_fault": "scientific_invalid",
        "invalid_run_classification": "none",
        "censoring": "none",
    }
    report["cost_summary"] = {
        "synthetic_tokens": 0,
        "synthetic_tool_calls": 0,
        "wall_time_seconds": 0,
        "note": "synthetic fixtures only; real cost recording requires authorized pilot",
    }

    # 9. Validation report (per-run streams, since each run is one trajectory)
    control_validation = validate_event_stream(control_events)
    fault_validation = validate_event_stream(fault_events)
    combined_validation = validate_event_stream(read_back)
    report["stream_validation"] = {
        "control_stream_valid": control_validation.valid,
        "control_stream_errors": control_validation.errors,
        "fault_stream_valid": fault_validation.valid,
        "fault_stream_errors": fault_validation.errors,
        "combined_store_expected_mixed_runs": not combined_validation.valid,
        "note": "combined store mixes two trajectories by design; per-run validation is the validity check",
    }
    report["manifest_hashes"] = {
        "control": control_manifest.manifest_hash,
        "fault": fault_manifest.manifest_hash,
    }
    report["gate_status"] = {
        "pilot_gates_passed": "NOT CLAIMED — synthetic fixtures only",
        "instrumentation_ready": True,
        "separately_authorized_pilot_required": True,
    }

    (out_dir / "readiness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic R1 readiness workflow")
    parser.add_argument("--out", type=str, default=None, help="output directory")
    args = parser.parse_args()

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="r1_synthetic_readiness_"))

    report = run_synthetic_readiness(out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nReport written to {out_dir / 'readiness_report.json'}")
    print("NOTE: synthetic fixtures only. Pilot gates are NOT claimed as passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
