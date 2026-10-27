"""Fixture-only checks for cumulative repeated-bad stamps across E1 and E2."""
import shutil
from pathlib import Path

import pytest

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.snapshot import capture_snapshot, resume_campaign
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.memory.similarity import compute_repeated_bad_stats
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class Manager:
    deterministic = True

    def propose_next_trial(self, status, memory_context, node_spec):
        return ManagerProposal(manager_mode="baseline_manager", proposal_summary="lower learning rate",
                               proposal_rationale="decrease learning rate", target_files=node_spec.editable_paths,
                               objective="edit", extra={})


class Worker:
    mode = "fixture_worker"

    def __init__(self, root, metrics, candidate_better=True):
        self.node_root = root
        self.metrics = list(metrics)
        self.candidate_better = candidate_better

    def result(self, spec, metric, changed=()):
        return WorkerResult(worker_mode=self.mode, changed_files=changed, success=True,
                            parsed_metrics={spec.metric_name: metric}, raw_log_ref="", patch_ref="fixture.diff",
                            git_commit_before="", git_commit_after="")

    def run_trial(self, proposal, spec, budget_index):
        (self.node_root / "train.py").write_text("CANDIDATE\n")
        return self.result(spec, self.metrics.pop(0), ("train.py",))

    def run_trial_at_seed(self, proposal, spec, budget_index, *, train_seed, split_seed=None):
        candidate = (self.node_root / "train.py").read_text() == "CANDIDATE\n"
        value = 0.95 if candidate == self.candidate_better else 0.90
        return self.result(spec, value)


class Controller:
    version = "fixture/v1"
    deterministic = True

    def __init__(self, verify_index):
        self.verify_index = verify_index

    def decide(self, view, mask):
        from autoresearch.control_plane.controller import ControlDecision
        action = "verify" if view.epoch == "E2" and view.budget_index == self.verify_index else "continue"
        return ControlDecision(action=action, controller_version=self.version, input_hash=view.input_hash())


def campaign(tmp_path, metrics, verify_index=None, candidate_better=True):
    spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    root = tmp_path / "node"
    shutil.copytree(ROOT / "nodes" / "lr_synthetic", root)
    (root / "train.py").write_text("INCUMBENT\n")
    snaps = {}

    def capture(ctx):
        if ctx.epoch in ("E1", "E2"):
            snaps[(ctx.epoch, ctx.budget_index)] = capture_snapshot(ctx)

    path = tmp_path / "trials.jsonl"
    run_real_campaign(node_spec=spec, campaign_id="stamp", budget=len(metrics),
                      manager_mode="baseline_manager", memory_mode="none", records_path=path,
                      worker=Worker(root, metrics, candidate_better), proposal_backend=Manager(),
                      controller=Controller(verify_index), initial_current_best=0.95, snapshot_hook=capture)
    return spec, snaps, TrialAppendStore(path).read_all()


def resume(tmp_path, snap, action, candidate_better=True):
    into = tmp_path / (snap.epoch + str(snap.budget_index) + action)
    result = resume_campaign(snap, into, first_action=action, horizon=0,
                             worker=Worker(into, [0.80], candidate_better), manager=Manager())
    records = TrialAppendStore(result.restored.records_path).read_all()
    assert records[-1].trial_id == result.first_record.trial_id
    assert records[-1].repeated_bad_count == result.first_record.repeated_bad_count
    assert records[-1].repeated_bad_count == compute_repeated_bad_stats(records).repeated_bad_count
    return records[-1]


def test_e1_evaluated_and_skip_include_prefix_and_zero(tmp_path):
    _, snaps, _ = campaign(tmp_path, [0.8, 0.8, 0.8])
    assert resume(tmp_path, snaps[("E1", 1)], "continue").repeated_bad_count == 0
    assert resume(tmp_path, snaps[("E1", 3)], "continue").repeated_bad_count == 2
    assert resume(tmp_path, snaps[("E1", 3)], "skip").repeated_bad_count == 1


@pytest.mark.parametrize("second_metric,better,provisional,final,expected", [
    (0.8, True, TrialDecision.DISCARDED, TrialDecision.KEPT, 0),
    (0.99, False, TrialDecision.KEPT, TrialDecision.DISCARDED, 1),
])
def test_e2_verify_recounts_final_decision_live_and_resumed(tmp_path, second_metric, better, provisional, final, expected):
    _, snaps, records = campaign(tmp_path, [0.8, second_metric], verify_index=2, candidate_better=better)
    snap = snaps[("E2", 2)]
    assert snap.provisional.decision == provisional
    assert snap.provisional.repeated_bad_count == (1 if provisional == TrialDecision.DISCARDED else 0)
    assert records[-1].decision == final
    assert records[-1].repeated_bad_count == expected
    assert records[-1].repeated_bad_count == compute_repeated_bad_stats(records).repeated_bad_count
    resumed = resume(tmp_path, snap, "verify", better)
    assert resumed.decision == final
    assert resumed.repeated_bad_count == expected
    continued = resume(tmp_path, snap, "continue", better)
    assert continued.decision == provisional
    assert continued.repeated_bad_count == snap.provisional.repeated_bad_count
