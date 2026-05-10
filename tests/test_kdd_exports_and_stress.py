from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import FailureCategory, TrialDecision
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.stress_worker import ScopeViolationWorker

ROOT = Path(__file__).resolve().parents[1]


def test_scope_violation_worker_records_invalid_edit_scope(tmp_path: Path) -> None:
    spec = load_registered_node("resnet_trigger", repo_root=ROOT)
    records_path = tmp_path / "kdd_stress_scope_trials.jsonl"
    worker = ScopeViolationWorker(
        node_root=ROOT / "nodes" / "ResNet_trigger",
        artifacts_dir=tmp_path / "artifacts" / "kdd_stress_scope",
    )
    run_real_campaign(
        node_spec=spec,
        campaign_id="kdd_stress_scope",
        budget=1,
        manager_mode="baseline_manager",
        memory_mode="append_only_summary_with_rationale",
        records_path=records_path,
        worker=worker,
    )
    record = TrialAppendStore(records_path).read_all()[0]
    assert record.decision == TrialDecision.FAILED_INVALID
    assert record.failure_category == FailureCategory.INVALID_EDIT_SCOPE
    assert Path(record.patch_ref).exists()
    assert record.git_commit_before == record.git_commit_after


def test_export_kdd_figures_writes_svg(tmp_path: Path) -> None:
    input_csv = tmp_path / "memory_ablation_summary.csv"
    input_csv.write_text(
        "campaign_id,memory_mode,total_trials,kept,discarded,failed_invalid,acceptance_rate,"
        "repeated_bad_count,repeated_bad_rate,repeated_invalid_count,repeated_degraded_count,best_metric\n"
        "a,none,5,1,2,2,0.2,2,0.4,1,1,0.781\n"
        "b,append_only_summary,5,3,2,0,0.6,1,0.2,0,1,0.783\n"
        "c,append_only_summary_with_rationale,5,4,1,0,0.8,0,0.0,0,0,0.784\n",
        encoding="utf-8",
    )
    output = tmp_path / "fig.svg"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_kdd_figures.py"),
            "--figure",
            "repeated_bad_rate",
            "--input",
            str(input_csv),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    text = output.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "Repeated-bad rate" in text


def test_export_kdd_tables_includes_provenance_chain(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledgers"
    output_dir = tmp_path / "tables"
    ledger_dir.mkdir()
    script = ROOT / "scripts" / "run_kdd_main_campaign.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--node",
            "resnet_trigger",
            "--budget",
            "5",
            "--campaign-id",
            "kdd_main_5trial",
            "--dry-run",
            "--records",
            str(ledger_dir / "kdd_main_5trial_trials.jsonl"),
            "--no-export",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    export = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_kdd_tables.py"),
            "--campaign-id",
            "kdd_main_5trial",
            "--ledger-dir",
            str(ledger_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export.returncode == 0, export.stderr
    provenance_path = output_dir / "provenance_chain.csv"
    rows = list(csv.DictReader(provenance_path.open("r", encoding="utf-8")))
    assert len(rows) == 5
    assert all(row["complete"] == "True" for row in rows)


def test_manager_comparison_summary_includes_artifact_capture_completeness(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledgers"
    tables_dir = tmp_path / "tables"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_manager_comparison.py"),
            "--node",
            "resnet_trigger",
            "--budget",
            "1",
            "--managers",
            "baseline_manager",
            "prompt_manager",
            "--dry-run",
            "--records-dir",
            str(ledger_dir),
            "--tables-dir",
            str(tables_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    rows = list(csv.DictReader((tables_dir / "manager_comparison_summary.csv").open()))
    assert {row["manager_mode"] for row in rows} == {"baseline_manager", "prompt_manager"}
    assert all(row["artifact_capture_completeness"] == "1.0" for row in rows)
