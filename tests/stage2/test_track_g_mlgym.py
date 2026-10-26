from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.manager.base import ManagerProposal
from autoresearch.nodes.mlgym_battle_of_sexes.stub import FakeMLGymWorker
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


class FakeManager:
    deterministic = True

    def propose_next_trial(self, status, memory_context, node_spec):
        return ManagerProposal(
            manager_mode="fake_manager", proposal_summary=f"fake strategy {status.budget_index}",
            proposal_rationale="Synthetic scope and adapter exercise only.",
            target_files=("strategy.py",), objective="Illustrative strategy change; do not execute code.",
        )


def test_three_trial_fake_adapter_campaign(tmp_path):
    node = load_registered_node("mlgym_battle_of_sexes")
    calls = []

    def fake_evaluate(index):
        calls.append(index)
        return ({"Score": {1: 0.25, 2: 0.75, 3: 0.5}[index]}, "fake/target.py")

    worker = FakeMLGymWorker(artifact_dir=tmp_path / "artifacts", evaluate=fake_evaluate)
    ledger = tmp_path / "trials.jsonl"
    run_real_campaign(node_spec=node, campaign_id="mlgym_fake_only", budget=3,
                      manager_mode="fake_manager", memory_mode="none", records_path=ledger,
                      worker=worker, proposal_backend=FakeManager())
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert calls == [1, 2, 3]
    assert len(records) == 3
    assert [r["parsed_metrics"] for r in records] == [
        {"Score": 0.25}, {"Score": 0.75}, {"Score": 0.5}]
    assert [r["decision"] for r in records] == ["kept", "kept", "discarded"]
    assert all(r["targeted_files"] == ["strategy.py"] and r["validity_status"] == "valid"
               and r["extra"]["scope_validation"]["valid"] and r["extra"]["worker"]["synthetic_only"]
               and r["worker_mode"] == "mlgym_fake_only" for r in records)


def test_fake_worker_rejects_out_of_scope_without_evaluating(tmp_path):
    node = load_registered_node("mlgym_battle_of_sexes")
    calls = []
    worker = FakeMLGymWorker(artifact_dir=tmp_path, evaluate=lambda i: calls.append(i))
    proposal = ManagerProposal(manager_mode="fake", proposal_summary="bad", proposal_rationale="",
                               target_files=("evaluate.py",), objective="fake")
    import pytest
    with pytest.raises(PermissionError):
        worker.run_trial(proposal, node, 1)
    assert calls == []


def test_fake_worker_never_invents_invalid_score(tmp_path):
    node = load_registered_node("mlgym_battle_of_sexes")
    worker = FakeMLGymWorker(artifact_dir=tmp_path, evaluate=lambda i: ({"Score": float("nan")}, "fake/target.py"))
    proposal = ManagerProposal(manager_mode="fake", proposal_summary="fake", proposal_rationale="",
                               target_files=("strategy.py",), objective="fake")
    result = worker.run_trial(proposal, node, 1)
    assert not result.success and result.parsed_metrics == {}
