"""Markdown campaign report writer.

Generates a human-readable audit report from a ``CampaignSummary``.  The
report is designed to be committed alongside JSONL ledgers so reviewers can
quickly understand what happened in a campaign without loading the raw records.

Usage::

    from autoresearch.evaluation.campaign_summary import load_campaign_summary
    from autoresearch.reporting.write_report import write_campaign_report

    summary = load_campaign_summary("experiments/ledgers/my_campaign_trials.jsonl")
    path = write_campaign_report(summary, output_dir="paper/notes")
    print(f"Report written to {path}")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from autoresearch.evaluation.campaign_summary import CampaignSummary
from autoresearch.memory.schemas import TrialDecision, TrialRecord


def write_campaign_report(
    summary: CampaignSummary,
    output_dir: str | Path,
    filename: Optional[str] = None,
) -> Path:
    """Write a markdown report for *summary* into *output_dir*.

    Args:
        summary: A ``CampaignSummary`` produced by ``load_campaign_summary``.
        output_dir: Directory where the report file will be written.
        filename: Optional override for the output filename.  Defaults to
            ``<campaign_id>_report.md``.

    Returns:
        The ``Path`` of the written report file.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    m = summary.metrics
    campaign_id = m.campaign_id or "unknown"
    out_name = filename or f"{campaign_id}_report.md"
    out_path = target_dir / out_name

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Title and overview
    # ------------------------------------------------------------------
    lines.append(f"# Campaign Report: `{campaign_id}`\n")
    lines.append(
        f"**Node:** `{m.node_id}`  \n"
        f"**Metric:** `{m.metric_name}` ({m.metric_direction})  \n"
        f"**Total trials:** {m.total_trials}  \n"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Optimization summary table
    # ------------------------------------------------------------------
    lines.append("## Optimization Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Initial {m.metric_name} | {_fmt(m.initial_metric)} |")
    lines.append(f"| Best {m.metric_name} | {_fmt(m.best_metric)} |")
    lines.append(f"| Final accepted {m.metric_name} | {_fmt(m.final_accepted_metric)} |")
    lines.append(f"| Net gain | {_fmt(m.net_gain, sign=True)} |")
    lines.append(f"| Gain per trial | {_fmt(m.gain_per_trial)} |")
    lines.append(f"| Gain per accepted trial | {_fmt(m.gain_per_accepted_trial)} |")
    lines.append(f"| Gain per budget unit | {_fmt(m.gain_per_budget_unit)} |")
    lines.append(f"| Total wall-clock (s) | {m.total_wall_clock_seconds:.1f} |")
    lines.append(f"| Gain per hour | {_fmt(m.gain_per_hour)} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Governance summary table
    # ------------------------------------------------------------------
    lines.append("## Governance Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Kept | {m.kept_count} |")
    lines.append(f"| Discarded | {m.discarded_count} |")
    lines.append(f"| Failed / invalid | {m.failed_invalid_count} |")
    lines.append(f"| Acceptance rate | {m.acceptance_rate:.1%} |")
    lines.append(f"| Invalid rate | {m.invalid_rate:.1%} |")
    lines.append(f"| Complete provenance rate | {m.complete_provenance_rate:.1%} |")
    lines.append(f"| Editable-scope violations | {m.editable_scope_violation_count} |")
    lines.append(f"| Trials with rationale | {m.trials_with_rationale} |")
    lines.append(f"| Command failure rate | {m.command_failure_rate:.1%} |")
    lines.append(f"| Metric-parse failure rate | {m.metric_parsing_failure_rate:.1%} |")
    lines.append(f"| Artifact capture completeness | {m.artifact_capture_completeness:.1%} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Per-trial detail
    # ------------------------------------------------------------------
    lines.append("## Trial Log\n")
    lines.append("| # | ID | Proposal | Decision | Δ metric | Failure |")
    lines.append("|---|---|---|---|---:|---|")
    for record in summary.records:
        delta = _fmt(record.delta_vs_best, sign=True) if record.delta_vs_best is not None else "—"
        failure = record.failure_category.value if record.failure_category else ""
        decision_icon = _decision_icon(record.decision)
        lines.append(
            f"| {record.budget_index} "
            f"| `{record.trial_id}` "
            f"| {record.proposal_summary[:60]} "
            f"| {decision_icon} {record.decision.value} "
            f"| {delta} "
            f"| {failure} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Accepted trials detail
    # ------------------------------------------------------------------
    accepted = [r for r in summary.records if r.decision == TrialDecision.KEPT]
    if accepted:
        lines.append("## Accepted Trials\n")
        for record in accepted:
            lines.append(f"### Trial `{record.trial_id}` (budget index {record.budget_index})\n")
            lines.append(f"**Proposal:** {record.proposal_summary}  ")
            lines.append(f"**Rationale:** {record.proposal_rationale}  ")
            metric_val = record.parsed_metrics.get(m.metric_name)
            lines.append(f"**{m.metric_name}:** {_fmt(metric_val)}  ")
            lines.append(f"**Δ vs best before:** {_fmt(record.delta_vs_best, sign=True)}  ")
            if record.patch_ref:
                lines.append(f"**Patch:** `{record.patch_ref}`  ")
            if record.raw_log_ref:
                lines.append(f"**Log:** `{record.raw_log_ref}`  ")
            lines.append("")

    # ------------------------------------------------------------------
    # Provenance note
    # ------------------------------------------------------------------
    lines.append("## Provenance\n")
    lines.append(
        "Every trial record in this campaign was written to an append-only JSONL ledger.  "
        "Decisions are owned by the Stage 2 control plane, not the worker or the manager.  "
        "The provenance chain for each accepted result is: "
        "proposal → patch → scope check → run log → parsed metric → keep/discard decision.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def format_campaign_report(summary: CampaignSummary) -> str:
    """Return the report as a string without writing to disk."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_campaign_report(summary, output_dir=tmp, filename="report.md")
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], *, sign: bool = False, decimals: int = 6) -> str:
    if value is None:
        return "—"
    fmt = f"+.{decimals}f" if sign and value >= 0 else f".{decimals}f"
    return format(value, fmt)


def _decision_icon(decision: TrialDecision) -> str:
    return {
        TrialDecision.KEPT: "✅",
        TrialDecision.DISCARDED: "⬇️",
        TrialDecision.FAILED_INVALID: "❌",
    }.get(decision, "")
