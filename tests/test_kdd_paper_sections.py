from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "kdd_aae_2026" / "sections"


def _read(name: str) -> str:
    return (PAPER / name).read_text(encoding="utf-8")


def _positions(text: str, needles: list[str]) -> list[int]:
    positions = []
    for needle in needles:
        index = text.find(needle)
        assert index >= 0, f"Missing required text: {needle}"
        positions.append(index)
    return positions


def test_chunk_4_1_related_work_acceptance() -> None:
    text = _read("02_related_work.md")
    assert "AIDE (Jiang et al. 2025)" in text
    assert "| Dimension | AIDE | `autoresearch_harness` |" in text
    assert "AIDE optimises the ML metric; we govern the experimentation process." in text
    for source in ("Trivedy", "Böckeler", "OpenAI", "Anthropic"):
        assert source in text
    assert "MLflow" in text
    assert "Weights & Biases" in text
    assert "AutoML" in text
    assert "Mind2Web" in text
    assert "ml-intern" in text
    assert (
        "None of the above provides the full governed control plane: scope enforcement, "
        "append-only audit ledger, failure taxonomy, and memory ablation for evaluating "
        "autonomous experimentation."
    ) in text


def test_chunk_4_2_system_design_acceptance() -> None:
    text = _read("03_system_design.md")
    assert "managers cannot commit trial state" in text
    assert "Manager/Worker Separation" in text
    assert "Pending-Trial Guard" in text
    assert "Append-Only Ledger" in text
    assert "Memory Modes" in text
    assert (
        "We keep the keep/discard decision in the control plane, not delegated to the manager."
    ) in text
    assert "Rajasekaran et al. (2026)" in text
    assert "Table S1: Harness guides and sensors" in text
    assert "Guide (feedforward" in text
    assert "Sensor (feedback" in text
    assert "Table S2: Agent failure modes and harness controls" in text
    assert "Declares victory too early" in text


def test_chunk_4_3_experiments_acceptance() -> None:
    text = _read("04_experiments.md")
    assert "The hypothesis is pre-stated before results" in text
    assert "repeated_bad_rate(none) > repeated_bad_rate(append_only_summary)" in text
    assert "ResNet-trigger" in text
    assert "Fixed-Budget Campaign Protocol" in text
    assert "Stress Trial" in text
    assert "Governance Metrics" in text
    assert "Table 2: Failure taxonomy" in text
    for category in (
        "invalid_edit_scope",
        "syntax_error",
        "runtime_error",
        "metric_missing",
        "degraded_metric",
        "no_op_patch",
    ):
        assert category in text


def test_chunk_4_4_results_order_and_artifact_links() -> None:
    text = _read("05_results.md")
    ordered = [
        "Table 1: Main campaign",
        "Figure 2:",
        "Figure 3:",
        "Table 2 result artifact",
        "Table 4 result artifact",
        "Figure 4:",
        "The accepted edit also improved validation AUC by 0.002845",
    ]
    assert _positions(text, ordered) == sorted(_positions(text, ordered))
    for artifact in (
        "../../tables/main_campaign_summary.csv",
        "../../tables/governance_metrics.csv",
        "../../figures/fig2_repeated_bad_rate.svg",
        "../../figures/fig3_decision_breakdown.svg",
        "../../tables/failure_taxonomy.csv",
        "../../tables/provenance_chain.csv",
        "../../tables/artifact_completeness_report.txt",
        "../../../artifact_manifest.json",
        "../../figures/fig4_trajectory.svg",
    ):
        assert artifact in text


def test_chunk_4_5_discussion_acceptance() -> None:
    text = _read("06_discussion_limitations.md")
    non_claims = (
        "This work does not claim to build a general autonomous scientist, prove scientific "
        "discovery, introduce a universal optimization algorithm, or depend on a specific "
        "coding-agent backend. The ResNet-trigger task is used as a real scientific ML case "
        "study for evaluating governed autonomous experimentation; it is not claimed to "
        "represent all ML optimization tasks."
    )
    assert non_claims in text
    for paragraph in (
        "We evaluate on one real scientific ML node. This demonstrates real governed execution, but does not claim broad benchmark coverage.",
        "All experiments use the ResNet-trigger node. We cannot rule out that reported improvements overfit to this evaluation domain. Applying the harness hill-climbing methodology of Trivedy (2026) across multiple evaluation nodes with holdout splits would strengthen the governance claims.",
        "Dry-run tests validate control-plane contracts; reported empirical results use real worker campaigns unless explicitly marked as smoke tests.",
        "Orchestration backends, cloud execution, and UI layers are future extensions. This work focuses on the control-plane protocol and its audit metrics.",
    ):
        assert paragraph in text
    assert "holdout" in text
    assert (
        "The NodeSpec YAML pattern generalises the harness to new ML experiments without "
        "code changes; each spec is a harness template for a class of experiments."
    ) in text


def test_markdown_links_point_to_existing_repo_files() -> None:
    for section in PAPER.glob("*.md"):
        text = section.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (section.parent / target).resolve()
            assert path.exists(), f"{section.name} links to missing file: {target}"
