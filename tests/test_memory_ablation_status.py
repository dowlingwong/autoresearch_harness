from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_kdd_memory_ablation import build_rows, render_report

from autoresearch.control_plane.campaign import run_dry_campaign
from autoresearch.nodes.registry import load_registered_node

ROOT = Path(__file__).resolve().parents[1]


def test_memory_ablation_checker_labels_dry_run_evidence(tmp_path: Path) -> None:
    spec = load_registered_node("resnet_trigger", repo_root=ROOT)
    ledger_dir = tmp_path / "ledgers"
    campaigns = (
        "ablation_none",
        "ablation_append_only_summary",
        "ablation_append_only_summary_with_rationale",
    )
    modes = ("none", "append_only_summary", "append_only_summary_with_rationale")
    for campaign_id, mode in zip(campaigns, modes):
        run_dry_campaign(
            node_spec=spec,
            campaign_id=campaign_id,
            budget=1,
            manager_mode="baseline_manager",
            memory_mode=mode,
            records_path=ledger_dir / f"{campaign_id}_trials.jsonl",
        )

    rows = build_rows(ledger_dir, campaigns, modes)
    assert [row.evidence_type for row in rows] == ["dry_run", "dry_run", "dry_run"]
    report = render_report(rows, expected_trials=1, require_real=True)
    assert '"status": "failed_require_real"' in report


def test_memory_ablation_checker_cli_writes_report(tmp_path: Path) -> None:
    spec = load_registered_node("resnet_trigger", repo_root=ROOT)
    ledger_dir = tmp_path / "ledgers"
    output = tmp_path / "report.txt"
    for mode in ("none", "append_only_summary", "append_only_summary_with_rationale"):
        campaign_id = f"ablation_{mode}"
        run_dry_campaign(
            node_spec=spec,
            campaign_id=campaign_id,
            budget=1,
            manager_mode="baseline_manager",
            memory_mode=mode,
            records_path=ledger_dir / f"{campaign_id}_trials.jsonl",
        )

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_kdd_memory_ablation.py"),
            "--ledger-dir",
            str(ledger_dir),
            "--expected-trials",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert '"all_complete": true' in output.read_text(encoding="utf-8")


def test_memory_ablation_checker_flags_pending_guard(tmp_path: Path) -> None:
    spec = load_registered_node("resnet_trigger", repo_root=ROOT)
    ledger_dir = tmp_path / "ledgers"
    campaign_id = "ablation_none"
    run_dry_campaign(
        node_spec=spec,
        campaign_id=campaign_id,
        budget=1,
        manager_mode="baseline_manager",
        memory_mode="none",
        records_path=ledger_dir / f"{campaign_id}_trials.jsonl",
    )
    (ledger_dir / f"{campaign_id}_pending.json").write_text("{}", encoding="utf-8")

    rows = build_rows(ledger_dir, (campaign_id,), ("none",))
    assert rows[0].pending_guard_exists is True
    assert rows[0].complete is False
    report = render_report(rows, expected_trials=1, require_real=False)
    assert '"status": "failed_incomplete"' in report
