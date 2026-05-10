from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.noop_worker import NoOpPatchWorker

ROOT = Path(__file__).resolve().parents[1]


def test_noop_patch_worker_records_failed_invalid_without_training(tmp_path: Path) -> None:
    spec = load_registered_node("resnet_trigger", repo_root=ROOT)
    records_path = tmp_path / "noop_trials.jsonl"
    worker = NoOpPatchWorker(
        node_root=ROOT / "nodes" / "ResNet_trigger",
        artifacts_dir=tmp_path / "artifacts",
    )

    run_real_campaign(
        node_spec=spec,
        campaign_id="noop",
        budget=1,
        manager_mode="baseline_manager",
        memory_mode="append_only_summary_with_rationale",
        records_path=records_path,
        worker=worker,
    )

    record = TrialAppendStore(records_path).read_all()[0]
    assert record.decision == TrialDecision.FAILED_INVALID
    assert record.failure_category == FailureCategory.NO_OP_PATCH
    assert record.extra["worker"]["training_skipped"] is True
    assert Path(record.patch_ref).exists()


def test_run_kdd_noop_trial_script_validates_record(tmp_path: Path) -> None:
    records_path = tmp_path / "ledgers" / "kdd_stress_noop_trials.jsonl"
    artifacts_dir = tmp_path / "artifacts" / "kdd_stress_noop"
    events_path = tmp_path / "events" / "kdd_stress_noop_events.jsonl"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_kdd_noop_trial.py"),
            "--records",
            str(records_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--events",
            str(events_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["validation"]["failed_invalid"] is True
    assert payload["validation"]["no_op_patch"] is True
    assert payload["validation"]["training_skipped"] is True
