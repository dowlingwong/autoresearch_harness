from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.control_plane import campaign
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.doom_loop import detect_doom_loop, reject_doom_loop
from autoresearch.memory.research_context import (
    load_node_research_context_ref,
    write_node_research_context,
)
from autoresearch.nodes.spec import BudgetSpec, NodeSpec


def _spec() -> NodeSpec:
    return NodeSpec(
        name="toy_node",
        description="toy",
        editable_paths=("train.py",),
        frozen_paths=("program.md",),
        setup_command="true",
        run_command="python train.py",
        metric_name="score",
        metric_direction="maximize",
        metric_parser="pkg:parse",
        acceptance_rule="higher",
        validity_checks=("metric_present",),
        default_budget=BudgetSpec(trials=1),
    )


def test_write_and_load_research_context_ref(tmp_path: Path):
    node_root = tmp_path / "node"
    node_root.mkdir()
    (node_root / "program.md").write_text("Task statement\n", encoding="utf-8")
    (node_root / "README.md").write_text("README note\n", encoding="utf-8")
    config_dir = tmp_path / "configs" / "nodes"
    config_dir.mkdir(parents=True)
    (config_dir / "toy_node.yaml").write_text('{"name": "toy_node"}\n', encoding="utf-8")

    ref = write_node_research_context(
        node_spec=_spec(),
        repo_root=tmp_path,
        node_root=node_root,
        output_dir=tmp_path / "notes",
    )
    loaded = load_node_research_context_ref(
        node_spec=_spec(),
        repo_root=tmp_path,
        output_dir=tmp_path / "notes",
    )

    assert loaded == ref
    assert Path(ref.path).read_text(encoding="utf-8").startswith("# Research Context: toy_node")
    assert len(ref.sha256) == 64


def test_campaign_metadata_attaches_research_context_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    node_root = tmp_path / "node"
    node_root.mkdir()
    (node_root / "program.md").write_text("Task statement\n", encoding="utf-8")
    (tmp_path / "configs" / "nodes").mkdir(parents=True)
    (tmp_path / "configs" / "nodes" / "toy_node.yaml").write_text('{"name": "toy_node"}\n', encoding="utf-8")
    ref = write_node_research_context(
        node_spec=_spec(),
        repo_root=tmp_path,
        node_root=node_root,
        output_dir=tmp_path / "paper" / "notes",
    )
    monkeypatch.setattr(campaign, "_repo_root", lambda: tmp_path)
    proposal = ManagerProposal(
        manager_mode="baseline_manager",
        proposal_summary="summary",
        proposal_rationale="rationale",
        target_files=("train.py",),
        objective="objective",
    )

    enriched = campaign._proposal_with_campaign_metadata(proposal, node_spec=_spec())

    assert enriched.extra["research_context"]["path"] == ref.path
    assert enriched.extra["research_context"]["sha256"] == ref.sha256


def test_doom_loop_detects_repeated_action():
    finding = detect_doom_loop(
        [
            "Lower learning rate to 3e-4",
            "Try dropout",
            "lower learning rate to 3e-4",
        ],
        max_repeats=2,
    )
    assert finding.repeated
    assert finding.count == 2


def test_reject_doom_loop_raises():
    with pytest.raises(ValueError, match="repeated proposal action"):
        reject_doom_loop(["lower lr", "lower lr"], max_repeats=2)
