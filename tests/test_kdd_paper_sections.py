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
        "Table 2 result artifact",   # failure taxonomy — precedes memory ablation
        "Figure 2:",                  # repeated-bad rate figure — in memory ablation
        "Figure 3:",                  # decision breakdown
        "Table 4 result artifact",   # provenance chain
        "Figure 4:",                  # trajectory
        # Real evidence: net gain from the actual kdd_main_5trial run
        "+0.000933 over five trials",
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
        assert artifact in text, f"Section 05 missing artifact link: {artifact}"


def test_chunk_4_5_discussion_acceptance() -> None:
    text = _read("06_discussion_limitations.md")
    # Non-claims are expressed as a formal scope-and-non-claims bullet list.
    for non_claim in (
        "General autonomous scientist capability",
        "Scientific discovery",
        "Universal optimisation algorithm",
        "Backend-specific dependency",
        "Broad benchmark coverage",
        "Flagship-campaign lifecycle diversity",
        "Memory effect on repeated-bad rate",
    ):
        assert non_claim in text, f"Missing non-claim bullet: {non_claim}"
    # Memory ablation section must be present and honest
    assert "Memory Ablation Analysis" in text
    assert "edit_failed" in text
    assert "fail-safe" in text
    # Limitations section must name key gaps
    assert "holdout" in text
    assert "Single node" in text
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
