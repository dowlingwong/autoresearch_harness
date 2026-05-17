from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.reset_node_state as reset_node_state


def test_reset_node_state_clears_node_runtime_and_exact_campaign_artifacts(tmp_path, monkeypatch):
    node_root = tmp_path / "node"
    node_root.mkdir()
    (node_root / "train.py").write_text("print('ok')\n", encoding="utf-8")
    for name in (".autoresearch_state.json", "results.tsv", "experiment_memory.jsonl", "run.log"):
        (node_root / name).write_text("stale\n", encoding="utf-8")
    node_artifacts = node_root / "artifacts"
    node_artifacts.mkdir()
    (node_artifacts / "metrics_latest.json").write_text("{}", encoding="utf-8")

    experiments_dir = tmp_path / "experiments"
    ledgers_dir = experiments_dir / "ledgers"
    events_dir = experiments_dir / "events"
    artifacts_dir = experiments_dir / "artifacts"
    ledgers_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    campaign_id = "kdd_main_5trial"
    (ledgers_dir / f"{campaign_id}_trials.jsonl").write_text("{}\n", encoding="utf-8")
    (ledgers_dir / f"{campaign_id}_pending.json").write_text("{}", encoding="utf-8")
    (events_dir / f"{campaign_id}_events.jsonl").write_text("{}\n", encoding="utf-8")
    campaign_artifacts = artifacts_dir / campaign_id / "trial-001"
    campaign_artifacts.mkdir(parents=True)
    (campaign_artifacts / "run.log").write_text("stale\n", encoding="utf-8")
    sibling = artifacts_dir / f"{campaign_id}_not_this_campaign"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep\n", encoding="utf-8")

    monkeypatch.setattr(
        reset_node_state,
        "load_registered_node",
        lambda node_name, repo_root: SimpleNamespace(editable_paths=("train.py",)),
    )
    monkeypatch.setattr(reset_node_state, "LEDGERS_DIR", ledgers_dir)
    monkeypatch.setattr(reset_node_state, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(reset_node_state, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(reset_node_state, "_git_checkout", lambda *args, **kwargs: None)

    reset_node_state.reset_node(
        "resnet_trigger",
        node_root=node_root,
        campaign_id=campaign_id,
        dry_run=False,
    )
    reset_node_state.reset_node(
        "resnet_trigger",
        node_root=node_root,
        campaign_id=campaign_id,
        dry_run=False,
    )

    for name in (".autoresearch_state.json", "results.tsv", "experiment_memory.jsonl", "run.log"):
        assert not (node_root / name).exists()
    assert not node_artifacts.exists()
    assert not (ledgers_dir / f"{campaign_id}_trials.jsonl").exists()
    assert not (ledgers_dir / f"{campaign_id}_pending.json").exists()
    assert not (events_dir / f"{campaign_id}_events.jsonl").exists()
    assert not (artifacts_dir / campaign_id).exists()
    assert sibling.exists()


def test_git_checkout_restores_untracked_node_from_baseline_template(tmp_path, monkeypatch):
    node_root = tmp_path / "node"
    node_root.mkdir()
    (node_root / "train.py").write_text("MUTATED = True\n", encoding="utf-8")
    baseline_dir = node_root / ".autoresearch_baseline"
    baseline_dir.mkdir()
    (baseline_dir / "train.py").write_text("MUTATED = False\n", encoding="utf-8")

    monkeypatch.setattr(
        reset_node_state.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr="missing"),
    )

    reset_node_state._git_checkout(
        "train.py",
        node_root,
        dry_run=False,
        baseline_ref=None,
    )

    assert (node_root / "train.py").read_text(encoding="utf-8") == "MUTATED = False\n"
