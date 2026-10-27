"""Track G adapter smoke: only stub Python executes, never model training."""
from __future__ import annotations

import dataclasses
import importlib.util
import shlex
import sys
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.autoresearch_macos.track_g_metric import parse_val_bpb
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.local_worker import LocalWorker

ROOT = Path(__file__).resolve().parents[2]


class StubProposals:
    mode = "stub"

    def propose_next_trial(self, status, memory_context, node_spec):
        index = status.budget_index
        old, new = [(4, 5), (5, 6), (6, 7)][index - 1]
        return ManagerProposal(
            manager_mode=self.mode, proposal_summary="stub edit", proposal_rationale="smoke",
            target_files=("train.py",), objective=f"Change DEPTH from {old} to {new}",
            extra={"deterministic_patch": True, "structured_edit": {
                "type": "python_constant", "path": "train.py", "symbol": "DEPTH",
                "old": str(old), "new": str(new),
            }},
        )


def test_node_contract_and_frozen_scope(tmp_path):
    spec = load_registered_node("track_g_autoresearch", repo_root=ROOT)
    assert spec.editable_paths == ("train.py",)
    assert validate_edit_scope(("train.py",), spec).valid
    assert not validate_edit_scope(("prepare.py",), spec).valid
    assert spec.metric_name == "val_bpb" and spec.metric_direction == "minimize"
    assert spec.run_command == "uv run train.py"
    assert "TIME_BUDGET = 300" in (ROOT / "nodes/autoresearch-macos/prepare.py").read_text()
    log = tmp_path / "log"
    log.write_text("val_bpb: 1.5\nval_bpb: 1.25\n")
    assert parse_val_bpb(log) == 1.25
    for bad in ("nan", "inf", "-1", "invalid"):
        log.write_text(f"val_bpb: 1.5\nval_bpb: {bad}\n")
        assert parse_val_bpb(log) is None


def test_three_trial_stub_campaign(tmp_path):
    spec = load_registered_node("track_g_autoresearch", repo_root=ROOT)
    # Only the command changes for the smoke: no uv, datasets, GPUs or training.
    spec = dataclasses.replace(spec, run_command=f'"{sys.executable}" train.py')
    node = tmp_path / "node"
    node.mkdir()
    (node / "train.py").write_text(
        'DEPTH = 4\nprint("val_bpb: %.2f" % {5: 1.0, 6: 0.9, 7: 1.1}[DEPTH])\n'
    )
    (node / "prepare.py").write_text("TIME_BUDGET = 300\n")
    records_path = tmp_path / "trials.jsonl"
    run_real_campaign(
        node_spec=spec, campaign_id="track-g-stub", budget=3,
        manager_mode="baseline_manager", memory_mode="none", records_path=records_path,
        worker=LocalWorker(node_root=node, artifact_dir=tmp_path / "artifacts"),
        proposal_backend=StubProposals(),
    )
    records = TrialAppendStore(records_path).read_all()
    assert len(records) == 3
    assert [r.parsed_metrics["val_bpb"] for r in records] == [1.0, 0.9, 1.1]
    assert [r.decision for r in records] == [TrialDecision.KEPT, TrialDecision.KEPT, TrialDecision.DISCARDED]
    assert all(r.extra["scope_validation"]["valid"] for r in records)
    assert (node / "prepare.py").read_text() == "TIME_BUDGET = 300\n"


def test_cli_local_worker_three_trial_stub(tmp_path, monkeypatch):
    """Exercise the CLI's local-worker route exclusively against a fake node."""
    module_spec = importlib.util.spec_from_file_location("track_g_cli_fixture", ROOT / "scripts/run_campaign.py")
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)

    node = tmp_path / "fake_node"
    node.mkdir()
    (node / "train.py").write_text(
        'DEPTH = 4\nprint("val_bpb: %.2f" % {5: 1.0, 6: 0.9, 7: 1.1}[DEPTH])\n'
    )
    (node / "prepare.py").write_text("TIME_BUDGET = 300\n")
    spec = load_registered_node("track_g_autoresearch", repo_root=ROOT)
    spec = dataclasses.replace(spec, run_command=f"{shlex.quote(sys.executable)} train.py")
    monkeypatch.setattr(cli, "load_registered_node", lambda name, repo_root: spec)

    def run_fake_campaign(**kwargs):
        assert isinstance(kwargs["worker"], LocalWorker)
        assert kwargs["node_spec"] is spec
        kwargs["proposal_backend"] = StubProposals()
        return run_real_campaign(**kwargs)

    monkeypatch.setattr(cli, "run_real_campaign", run_fake_campaign)
    records_path = tmp_path / "trials.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "run_campaign.py", "--node", "track_g_autoresearch", "--campaign-id", "fake-cli",
        "--budget", "3", "--node-root", str(node), "--worker", "local",
        "--records", str(records_path), "--artifacts-dir", str(tmp_path / "artifacts"),
        "--tables-dir", str(tmp_path / "tables"), "--no-events",
    ])
    assert cli.main() == 0
    records = TrialAppendStore(records_path).read_all()
    assert [r.parsed_metrics["val_bpb"] for r in records] == [1.0, 0.9, 1.1]
    assert [r.decision for r in records] == [TrialDecision.KEPT, TrialDecision.KEPT, TrialDecision.DISCARDED]
    assert all(r.extra["scope_validation"]["valid"] for r in records)
    assert (node / "prepare.py").read_text() == "TIME_BUDGET = 300\n"


def test_edit_real_bundled_assignment_without_running_training(tmp_path):
    import shutil
    from autoresearch.nodes.autoresearch_macos.local_edit import apply_track_g_edit
    from autoresearch.worker.local_worker import _AppliedEdit

    source = ROOT / "nodes/autoresearch-macos/train.py"
    target = tmp_path / "train.py"
    shutil.copyfile(source, target)
    result = apply_track_g_edit(target, "DEPTH", "4", "5")
    assert isinstance(result, _AppliedEdit)
    assert "DEPTH = 5               # number of transformer layers" in target.read_text()
    assert isinstance(apply_track_g_edit(target, "DEPTH", "4", "6"), str)
    assert "DEPTH = 5               # number of transformer layers" in target.read_text()
    assert isinstance(apply_track_g_edit(target, "DEPTH", "5", "("), str)
    compile(target.read_text(), str(target), "exec")
