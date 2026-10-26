from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.nodes.mlgym_battle_of_sexes.metric_parser import adapt_evaluation, parse_score
from autoresearch.nodes.registry import load_registered_node


def test_node_scope_and_fake_evaluation():
    node = load_registered_node("mlgym_battle_of_sexes")
    assert node.metric_name == "Score" and node.metric_direction == "maximize"
    assert validate_edit_scope(["strategy.py"], node).valid
    assert not validate_edit_scope(["evaluate.py"], node).valid
    assert not validate_edit_scope(["target.py"], node).valid
    # Shape matches PythonSubmissionTasks.evaluate(), values are fabricated.
    assert adapt_evaluation(({"Score": 1.25}, "/workspace/target.py")) == {"Score": 1.25}


def test_missing_or_invalid_metric_is_not_success():
    for raw in ('{}', '{"Score": true}', '{"Score": NaN}', '{"Score": "1"}', 'noise', '[1]'):
        assert parse_score(raw) == {}
    assert adapt_evaluation(({"Score": 1}, "")) == {}
    assert adapt_evaluation(({"Score": float("inf")}, "target.py")) == {}
    assert adapt_evaluation(({}, "target.py")) == {}


def test_real_runner_fails_closed():
    runner = Path(__file__).resolve().parents[2] / "nodes/mlgym_battle_of_sexes/run_mlgym.py"
    run = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
    assert run.returncode != 0
    assert not parse_score(run.stdout)
    assert "not configured" in run.stderr


def test_three_trial_dry_campaign(tmp_path):
    root = Path(__file__).resolve().parents[2]
    ledger = tmp_path / "trials.jsonl"
    run = subprocess.run(
        [sys.executable, str(root / "scripts/run_campaign.py"), "--node", "mlgym_battle_of_sexes",
         "--campaign-id", "mlgym_stub", "--budget", "3", "--dry-run", "--no-events",
         "--records", str(ledger), "--tables-dir", str(tmp_path / "tables")],
        cwd=root, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(records) == 3
    assert all("Score" in record["parsed_metrics"] and record["validity_status"] == "valid"
               and record["worker_mode"] == "dry_run_worker" for record in records)
