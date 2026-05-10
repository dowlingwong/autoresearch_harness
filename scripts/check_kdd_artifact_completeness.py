#!/usr/bin/env python3
"""Chunk 3.3 — Verify artifact completeness across KDD campaigns.

For each trial in each campaign ledger this script checks:
  1. patch_ref file exists on disk (or is a recognised synthetic dry-run path)
  2. raw_log_ref file exists on disk (same caveat)
  3. parsed_metrics is non-empty for every valid (validity_status == "valid") trial
  4. provenance.decision_id is non-null

A gap is reported but NOT a fatal error so the script always produces a report
usable in the paper Limitations section.

Usage
-----
  python3 scripts/check_kdd_artifact_completeness.py \\
      --campaigns kdd_main_5trial ablation_none \\
                  ablation_append_only_summary \\
                  ablation_append_only_summary_with_rationale \\
                  kdd_stress_scope \\
      --output paper/tables/artifact_completeness_report.txt

Exit code
---------
  0  — all campaigns meet the ≥90 % completeness threshold
  1  — one or more campaigns are below threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LEDGERS_DIR = ROOT / "experiments" / "ledgers"
TABLES_DIR = ROOT / "paper" / "tables"

# Paths that DryRunWorker synthesises — they do not exist on disk.
_DRY_RUN_SUBSTRINGS = ("dry_run_trial",)


def _is_dry_run_path(path_str: str) -> bool:
    return any(s in path_str for s in _DRY_RUN_SUBSTRINGS)


def _resolve_path(path_str: str) -> Path:
    """Resolve an artifact path to an absolute Path.

    Paths stored by real runs may be absolute (using the original host's
    filesystem prefix). If the absolute path does not exist we try to rebase
    it relative to ROOT by finding the repo directory component in the stored
    path — this makes the script portable across machines and CI environments.

    Paths from dry-run trials are relative and won't exist on disk; callers
    guard against those before calling this function.
    """
    p = Path(path_str)
    if p.is_absolute():
        if p.exists():
            return p
        # Try to rebase: find the repo directory name inside the stored path
        # and reconstruct from ROOT.  e.g.:
        #   stored : /Users/alice/Documents/autoresearch_harness/experiments/artifacts/...
        #   ROOT   : /sessions/.../autoresearch_harness
        #   result : ROOT / experiments / artifacts / ...
        repo_name = ROOT.name  # e.g. "autoresearch_harness"
        parts = p.parts
        try:
            idx = parts.index(repo_name)
            relative_tail = Path(*parts[idx + 1:])
            rebased = ROOT / relative_tail
            if rebased.exists():
                return rebased
        except ValueError:
            pass
        # Give up — return the original (caller will report NOT FOUND)
        return p
    return ROOT / p


@dataclass
class TrialCheck:
    trial_id: str
    budget_index: int
    decision: str
    validity_status: str
    is_dry_run: bool
    # individual check results
    patch_ref_ok: bool = False
    patch_ref_note: str = ""
    raw_log_ref_ok: bool = False
    raw_log_ref_note: str = ""
    metrics_ok: bool = False
    metrics_note: str = ""
    decision_id_ok: bool = False
    decision_id_note: str = ""

    @property
    def all_ok(self) -> bool:
        return (
            self.patch_ref_ok
            and self.raw_log_ref_ok
            and self.metrics_ok
            and self.decision_id_ok
        )

    @property
    def check_count(self) -> int:
        return 4

    @property
    def pass_count(self) -> int:
        return sum([
            self.patch_ref_ok,
            self.raw_log_ref_ok,
            self.metrics_ok,
            self.decision_id_ok,
        ])


@dataclass
class CampaignReport:
    campaign_id: str
    ledger_path: Path
    trial_checks: list[TrialCheck] = field(default_factory=list)
    error: str = ""

    @property
    def total_trials(self) -> int:
        return len(self.trial_checks)

    @property
    def total_checks(self) -> int:
        return sum(t.check_count for t in self.trial_checks)

    @property
    def passed_checks(self) -> int:
        return sum(t.pass_count for t in self.trial_checks)

    @property
    def completeness_pct(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return 100.0 * self.passed_checks / self.total_checks

    @property
    def meets_threshold(self) -> bool:
        return self.completeness_pct >= 90.0


def _check_trial(record: dict, repo_root: Path) -> TrialCheck:
    trial_id = record.get("trial_id", "?")
    budget_index = int(record.get("budget_index") or 0)
    decision = record.get("decision", "")
    validity_status = record.get("validity_status", "")
    worker_mode = record.get("worker_mode", "")

    # Determine if this is a dry-run trial by worker_mode or path
    patch_ref_str = record.get("patch_ref") or ""
    raw_log_ref_str = record.get("raw_log_ref") or ""
    is_dry = (
        worker_mode == "dry_run"
        or _is_dry_run_path(patch_ref_str)
        or _is_dry_run_path(raw_log_ref_str)
    )

    tc = TrialCheck(
        trial_id=trial_id,
        budget_index=budget_index,
        decision=decision,
        validity_status=validity_status,
        is_dry_run=is_dry,
    )

    # --- Check 1: patch_ref ---
    if is_dry:
        tc.patch_ref_ok = True
        tc.patch_ref_note = "synthetic (dry-run)"
    elif not patch_ref_str:
        # Invalid trials may have empty patch_ref — acceptable
        if validity_status == "invalid":
            tc.patch_ref_ok = True
            tc.patch_ref_note = "empty (invalid trial)"
        else:
            tc.patch_ref_ok = False
            tc.patch_ref_note = "MISSING (no path set for valid trial)"
    else:
        resolved = _resolve_path(patch_ref_str)
        if resolved.exists():
            tc.patch_ref_ok = True
            tc.patch_ref_note = "exists"
        else:
            tc.patch_ref_ok = False
            tc.patch_ref_note = f"NOT FOUND: {patch_ref_str}"

    # --- Check 2: raw_log_ref ---
    if is_dry:
        tc.raw_log_ref_ok = True
        tc.raw_log_ref_note = "synthetic (dry-run)"
    elif not raw_log_ref_str:
        if validity_status == "invalid":
            tc.raw_log_ref_ok = True
            tc.raw_log_ref_note = "empty (invalid trial)"
        else:
            tc.raw_log_ref_ok = False
            tc.raw_log_ref_note = "MISSING (no path set for valid trial)"
    else:
        resolved = _resolve_path(raw_log_ref_str)
        if resolved.exists():
            tc.raw_log_ref_ok = True
            tc.raw_log_ref_note = "exists"
        else:
            tc.raw_log_ref_ok = False
            tc.raw_log_ref_note = f"NOT FOUND: {raw_log_ref_str}"

    # --- Check 3: parsed_metrics non-empty for valid trials ---
    parsed_metrics = record.get("parsed_metrics") or {}
    if validity_status != "valid":
        tc.metrics_ok = True
        tc.metrics_note = "n/a (invalid trial)"
    elif parsed_metrics:
        tc.metrics_ok = True
        tc.metrics_note = f"ok ({', '.join(f'{k}={v}' for k, v in parsed_metrics.items())})"
    else:
        tc.metrics_ok = False
        tc.metrics_note = "MISSING (valid trial has no parsed_metrics)"

    # --- Check 4: provenance.decision_id non-null ---
    provenance = record.get("provenance") or {}
    did = provenance.get("decision_id")
    if did:
        tc.decision_id_ok = True
        tc.decision_id_note = f"ok ({did})"
    else:
        tc.decision_id_ok = False
        tc.decision_id_note = "MISSING (provenance.decision_id is null/absent)"

    return tc


def check_campaign(campaign_id: str, ledger_path: Path, repo_root: Path) -> CampaignReport:
    report = CampaignReport(campaign_id=campaign_id, ledger_path=ledger_path)
    if not ledger_path.exists():
        report.error = f"Ledger not found: {ledger_path}"
        return report
    try:
        with ledger_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.error = f"JSON parse error at line {lineno}: {exc}"
                    return report
                tc = _check_trial(record, repo_root)
                report.trial_checks.append(tc)
    except OSError as exc:
        report.error = str(exc)
    return report


def format_report(reports: list[CampaignReport]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("KDD AAE 2026 — Artifact Completeness Report")
    lines.append("=" * 72)
    lines.append("")

    overall_checks = sum(r.total_checks for r in reports)
    overall_passed = sum(r.passed_checks for r in reports)
    overall_pct = 100.0 * overall_passed / overall_checks if overall_checks else 0.0
    lines.append(
        f"Overall: {overall_passed}/{overall_checks} checks passed "
        f"({overall_pct:.1f}%)"
    )
    lines.append("")

    for rep in reports:
        lines.append("-" * 72)
        lines.append(f"Campaign: {rep.campaign_id}")
        lines.append(f"  Ledger : {rep.ledger_path}")

        if rep.error:
            lines.append(f"  ERROR  : {rep.error}")
            lines.append("")
            continue

        dry_count = sum(1 for t in rep.trial_checks if t.is_dry_run)
        real_count = rep.total_trials - dry_count
        lines.append(
            f"  Trials : {rep.total_trials} "
            f"({real_count} real, {dry_count} dry-run/synthetic)"
        )
        lines.append(
            f"  Checks : {rep.passed_checks}/{rep.total_checks} passed "
            f"({rep.completeness_pct:.1f}%)  "
            f"{'PASS ✓' if rep.meets_threshold else 'BELOW THRESHOLD ✗'}"
        )
        lines.append("")

        # Per-trial detail
        for tc in rep.trial_checks:
            status = "✓" if tc.all_ok else "✗"
            dry_tag = " [dry-run]" if tc.is_dry_run else ""
            lines.append(
                f"  [{status}] T{tc.budget_index:02d} {tc.trial_id}{dry_tag}"
                f"  decision={tc.decision}  validity={tc.validity_status}"
            )
            if not tc.patch_ref_ok:
                lines.append(f"       patch_ref   : {tc.patch_ref_note}")
            if not tc.raw_log_ref_ok:
                lines.append(f"       raw_log_ref : {tc.raw_log_ref_note}")
            if not tc.metrics_ok:
                lines.append(f"       metrics     : {tc.metrics_note}")
            if not tc.decision_id_ok:
                lines.append(f"       decision_id : {tc.decision_id_note}")

        lines.append("")

    lines.append("=" * 72)
    lines.append("Notes")
    lines.append("-" * 72)
    lines.append(
        "  dry-run/synthetic: DryRunWorker produces placeholder artifact paths that"
    )
    lines.append(
        "    do not exist on disk. These are counted as PASS because they are"
    )
    lines.append(
        "    intentional — dry-run campaigns are for governance/smoke testing only."
    )
    lines.append(
        "  Any real-run gaps should be investigated and documented in Limitations."
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk 3.3: verify artifact completeness across KDD campaigns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--campaigns",
        nargs="+",
        default=[
            "kdd_main_5trial",
            "ablation_none",
            "ablation_append_only_summary",
            "ablation_append_only_summary_with_rationale",
            "kdd_stress_scope",
            "kdd_stress_noop",
        ],
        help="Campaign IDs to check (ledger files are looked up automatically)",
    )
    parser.add_argument(
        "--ledgers-dir",
        default=str(LEDGERS_DIR),
        help="Directory containing *_trials.jsonl ledger files",
    )
    parser.add_argument(
        "--output",
        default=str(TABLES_DIR / "artifact_completeness_report.txt"),
        help="Output report path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Minimum completeness %% required (default: 90.0)",
    )
    args = parser.parse_args()

    ledgers_dir = Path(args.ledgers_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reports: list[CampaignReport] = []
    for campaign_id in args.campaigns:
        ledger_path = ledgers_dir / f"{campaign_id}_trials.jsonl"
        print(f"Checking {campaign_id} ...", end=" ", flush=True)
        rep = check_campaign(campaign_id, ledger_path, ROOT)
        reports.append(rep)
        if rep.error:
            print(f"ERROR — {rep.error}")
        else:
            print(f"{rep.completeness_pct:.1f}% ({rep.passed_checks}/{rep.total_checks})")

    report_text = format_report(reports)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to: {output_path}")
    print(report_text)

    # Exit 1 if any campaign fails the threshold
    any_failed = any(
        (not rep.error and not rep.meets_threshold) for rep in reports
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
