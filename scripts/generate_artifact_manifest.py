#!/usr/bin/env python3
"""Chunk 3.4 — Generate artifact_manifest.json in the repo root.

The manifest is a machine-readable index of every campaign, ledger,
paper table, and figure used in the KDD AAE 2026 submission.  It lets
reviewers verify reproducibility by checking which files exist and what
commands were used to produce them.

Usage
-----
  python3 scripts/generate_artifact_manifest.py
  python3 scripts/generate_artifact_manifest.py --output artifact_manifest.json

The script always exits 0 on success.  If any referenced file is missing,
a warning is printed but the manifest is still written so gaps are visible.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LEDGERS_DIR = ROOT / "experiments" / "ledgers"
TABLES_DIR = ROOT / "paper" / "tables"
FIGURES_DIR = ROOT / "paper" / "figures"
DRY_RUN_ARTIFACT_MARKER = "dry_run_trial"


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _count_trials(ledger_path: Path) -> int:
    if not ledger_path.exists():
        return 0
    count = 0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_trial_records(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _resolve_artifact_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        if path.exists():
            return path
        parts = path.parts
        try:
            idx = parts.index(ROOT.name)
        except ValueError:
            return path
        return ROOT / Path(*parts[idx + 1:])
    return ROOT / path


def _manifest_path(path_str: str) -> str:
    if not path_str:
        return ""
    resolved = _resolve_artifact_path(path_str)
    return _relative(resolved)


def _is_synthetic_artifact(path_str: str, worker_mode: str) -> bool:
    return worker_mode == "dry_run_worker" or DRY_RUN_ARTIFACT_MARKER in path_str


def _artifact_ref(path_str: str, worker_mode: str) -> dict[str, Any]:
    synthetic = _is_synthetic_artifact(path_str, worker_mode)
    exists = False
    if path_str:
        exists = _resolve_artifact_path(path_str).exists()
    return {
        "path": _manifest_path(path_str),
        "exists": exists,
        "synthetic": synthetic,
    }


def _trial_artifacts_for_campaign(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    ledger_path = ROOT / str(campaign["ledger"])
    artifacts: list[dict[str, Any]] = []
    for record in _read_trial_records(ledger_path):
        worker_mode = str(record.get("worker_mode") or "")
        artifacts.append(
            {
                "trial_id": str(record.get("trial_id") or ""),
                "budget_index": int(record.get("budget_index") or 0),
                "decision": str(record.get("decision") or ""),
                "validity_status": str(record.get("validity_status") or ""),
                "worker_mode": worker_mode,
                "patch_ref": _artifact_ref(str(record.get("patch_ref") or ""), worker_mode),
                "raw_log_ref": _artifact_ref(str(record.get("raw_log_ref") or ""), worker_mode),
                "parsed_metrics_present": bool(record.get("parsed_metrics") or {}),
                "decision_id": str((record.get("provenance") or {}).get("decision_id") or ""),
            }
        )
    return artifacts


def _artifact_index(campaign_groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for campaigns in campaign_groups.values():
        for campaign in campaigns:
            index[str(campaign["id"])] = _trial_artifacts_for_campaign(campaign)
    return index


def build_manifest() -> dict:
    # ------------------------------------------------------------------ #
    # Campaigns                                                            #
    # ------------------------------------------------------------------ #
    kdd_campaigns = [
        {
            "id": "kdd_main_5trial",
            "description": "Main KDD campaign — 5 real trials, prompt_manager, claw_style_worker, append_only_summary_with_rationale; 2 kept, 3 failed_invalid (runtime_error)",
            "ledger": _relative(LEDGERS_DIR / "kdd_main_5trial_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "kdd_main_5trial_trials.jsonl"),
            "worker": "claw_style_worker",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
        {
            "id": "kdd_stress_scope",
            "description": "Stress trial — stress_scope_violation_worker, forces invalid_edit_scope failure; validates first-class audit object recording",
            "ledger": _relative(LEDGERS_DIR / "kdd_stress_scope_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "kdd_stress_scope_trials.jsonl"),
            "worker": "stress_scope_violation_worker",
            "manager": "baseline_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
        {
            "id": "kdd_stress_noop",
            "description": "Stress trial — stress_no_op_patch_worker, forces no_op_patch failure; validates no-op guard and first-class audit object recording",
            "ledger": _relative(LEDGERS_DIR / "kdd_stress_noop_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "kdd_stress_noop_trials.jsonl"),
            "worker": "stress_no_op_patch_worker",
            "manager": "baseline_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
    ]

    ablation_campaigns = [
        {
            "id": "ablation_none",
            "description": "Memory ablation arm — no memory context; 5 real trials, prompt_manager, claw_style_worker; 2 kept, 3 failed_invalid",
            "ledger": _relative(LEDGERS_DIR / "ablation_none_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "ablation_none_trials.jsonl"),
            "worker": "claw_style_worker",
            "manager": "prompt_manager",
            "memory_mode": "none",
        },
        {
            "id": "ablation_append_only_summary",
            "description": "Memory ablation arm — append-only summary (no rationale); 5 real trials, prompt_manager, claw_style_worker; 2 kept, 3 failed_invalid",
            "ledger": _relative(LEDGERS_DIR / "ablation_append_only_summary_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "ablation_append_only_summary_trials.jsonl"),
            "worker": "claw_style_worker",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary",
        },
        {
            "id": "ablation_append_only_summary_with_rationale",
            "description": "Memory ablation arm — append-only summary with rationale; 5 real trials, prompt_manager, claw_style_worker; 2 kept, 3 failed_invalid",
            "ledger": _relative(
                LEDGERS_DIR / "ablation_append_only_summary_with_rationale_trials.jsonl"
            ),
            "trials": _count_trials(
                LEDGERS_DIR / "ablation_append_only_summary_with_rationale_trials.jsonl"
            ),
            "worker": "claw_style_worker",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
    ]

    manager_comparison_campaigns = [
        {
            "id": "manager_comparison_baseline_manager",
            "description": "Manager comparison — baseline_manager, 5 real deterministic-patch trials; 2 kept, 3 discarded",
            "ledger": _relative(
                LEDGERS_DIR / "manager_comparison_baseline_manager_trials.jsonl"
            ),
            "trials": _count_trials(
                LEDGERS_DIR / "manager_comparison_baseline_manager_trials.jsonl"
            ),
            "worker": "claw_style_worker_deterministic_patch",
            "manager": "baseline_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
        {
            "id": "manager_comparison_prompt_manager",
            "description": "Manager comparison — prompt_manager, 5 real deterministic-patch trials; 2 kept, 3 discarded",
            "ledger": _relative(
                LEDGERS_DIR / "manager_comparison_prompt_manager_trials.jsonl"
            ),
            "trials": _count_trials(
                LEDGERS_DIR / "manager_comparison_prompt_manager_trials.jsonl"
            ),
            "worker": "claw_style_worker_deterministic_patch",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
    ]

    optional_p8_campaigns = [
        {
            "id": "p8_memory10_none",
            "description": "Optional Priority 8 memory extension — no memory, 10 real deterministic-patch trials; 3 kept, 7 discarded",
            "ledger": _relative(LEDGERS_DIR / "p8_memory10_none_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "p8_memory10_none_trials.jsonl"),
            "worker": "claw_style_worker_deterministic_patch",
            "manager": "prompt_manager",
            "memory_mode": "none",
        },
        {
            "id": "p8_memory10_append_only_summary",
            "description": "Optional Priority 8 memory extension — append-only summary, 10 real deterministic-patch trials; 4 kept, 5 discarded, 1 precondition failure",
            "ledger": _relative(LEDGERS_DIR / "p8_memory10_append_only_summary_trials.jsonl"),
            "trials": _count_trials(LEDGERS_DIR / "p8_memory10_append_only_summary_trials.jsonl"),
            "worker": "claw_style_worker_deterministic_patch",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary",
        },
        {
            "id": "p8_memory10_append_only_summary_with_rationale",
            "description": "Optional Priority 8 memory extension — append-only summary with rationale, 10 real deterministic-patch trials; 4 kept, 5 discarded, 1 precondition failure",
            "ledger": _relative(
                LEDGERS_DIR / "p8_memory10_append_only_summary_with_rationale_trials.jsonl"
            ),
            "trials": _count_trials(
                LEDGERS_DIR / "p8_memory10_append_only_summary_with_rationale_trials.jsonl"
            ),
            "worker": "claw_style_worker_deterministic_patch",
            "manager": "prompt_manager",
            "memory_mode": "append_only_summary_with_rationale",
        },
    ]

    # ------------------------------------------------------------------ #
    # Paper tables                                                         #
    # ------------------------------------------------------------------ #
    table_files = sorted(
        [_relative(p) for p in TABLES_DIR.glob("*.csv")]
        + [_relative(p) for p in TABLES_DIR.glob("*.txt")]
    )

    # ------------------------------------------------------------------ #
    # Paper figures — KDD submission figures only (fig1–fig4)             #
    # ------------------------------------------------------------------ #
    kdd_figures = [
        _relative(FIGURES_DIR / "fig1_architecture.svg"),
        _relative(FIGURES_DIR / "fig2_repeated_bad_rate.svg"),
        _relative(FIGURES_DIR / "fig3_decision_breakdown.svg"),
        _relative(FIGURES_DIR / "fig4_trajectory.svg"),
    ]
    supplementary_figures = sorted(
        _relative(p)
        for p in FIGURES_DIR.glob("*.svg")
        if p.name not in {Path(f).name for f in kdd_figures}
    )

    # ------------------------------------------------------------------ #
    # Environment                                                          #
    # ------------------------------------------------------------------ #
    environment = {
        "python": _python_version(),
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "platform": sys.platform,
        "default_model": "qwen2.5-coder:7b",
        "default_ollama_host": "http://localhost:11434",
        "node": "resnet_trigger",
        "metric": "val_auc (higher is better; derived from val_bpb = 1 - val_bpb)",
    }

    # ------------------------------------------------------------------ #
    # Canonical run commands                                               #
    # ------------------------------------------------------------------ #
    run_commands = [
        # Main 5-trial real campaign (requires Ollama + node directory)
        (
            "RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_N_SIGNAL=1000 "
            "RESNET_TRIGGER_FAST_N_NOISE=1000 RESNET_TRIGGER_FAST_TRACE_LEN=4096 "
            "RESNET_TRIGGER_FAST_BATCH_SIZE=64 RESNET_TRIGGER_FAST_EPOCHS=3 "
            "RESNET_TRIGGER_FAST_SKIP_TEST=1 RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 "
            "RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 RESNET_TRIGGER_DEVICE=cpu "
            "uv run --extra dev python scripts/run_kdd_main_campaign.py "
            "--node resnet_trigger --budget 5 --campaign-id kdd_main_5trial "
            "--manager prompt_manager --memory-mode append_only_summary_with_rationale "
            "--node-root nodes/ResNet_trigger --model ollama/qwen2.5-coder:7b --no-export"
        ),
        # Stress trials
        (
            "python3 scripts/run_kdd_stress_trial.py "
            "--node resnet_trigger --campaign-id kdd_stress_scope"
        ),
        (
            "python3 scripts/run_kdd_noop_trial.py "
            "--node resnet_trigger --campaign-id kdd_stress_noop "
            "--node-root nodes/ResNet_trigger"
        ),
        # Memory ablation (real campaigns — requires Ollama + node directory)
        (
            "RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_N_SIGNAL=1000 "
            "RESNET_TRIGGER_FAST_N_NOISE=1000 RESNET_TRIGGER_FAST_TRACE_LEN=4096 "
            "RESNET_TRIGGER_FAST_BATCH_SIZE=64 RESNET_TRIGGER_FAST_EPOCHS=3 "
            "RESNET_TRIGGER_FAST_SKIP_TEST=1 RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 "
            "RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 RESNET_TRIGGER_DEVICE=cpu "
            "uv run --extra dev python scripts/run_kdd_memory_ablation.py "
            "--node resnet_trigger --budget 5 --memory-mode none "
            "--campaign-id ablation_none --node-root nodes/ResNet_trigger "
            "--model ollama/qwen2.5-coder:7b"
        ),
        (
            "RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_N_SIGNAL=1000 "
            "RESNET_TRIGGER_FAST_N_NOISE=1000 RESNET_TRIGGER_FAST_TRACE_LEN=4096 "
            "RESNET_TRIGGER_FAST_BATCH_SIZE=64 RESNET_TRIGGER_FAST_EPOCHS=3 "
            "RESNET_TRIGGER_FAST_SKIP_TEST=1 RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 "
            "RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 RESNET_TRIGGER_DEVICE=cpu "
            "uv run --extra dev python scripts/run_kdd_memory_ablation.py "
            "--node resnet_trigger --budget 5 --memory-mode append_only_summary "
            "--campaign-id ablation_append_only_summary --node-root nodes/ResNet_trigger "
            "--model ollama/qwen2.5-coder:7b"
        ),
        (
            "RESNET_TRIGGER_FAST_SEARCH=1 RESNET_TRIGGER_FAST_N_SIGNAL=1000 "
            "RESNET_TRIGGER_FAST_N_NOISE=1000 RESNET_TRIGGER_FAST_TRACE_LEN=4096 "
            "RESNET_TRIGGER_FAST_BATCH_SIZE=64 RESNET_TRIGGER_FAST_EPOCHS=3 "
            "RESNET_TRIGGER_FAST_SKIP_TEST=1 RESNET_TRIGGER_EARLY_STOP_PATIENCE=2 "
            "RESNET_TRIGGER_EARLY_STOP_MIN_DELTA=0.002 RESNET_TRIGGER_DEVICE=cpu "
            "uv run --extra dev python scripts/run_kdd_memory_ablation.py "
            "--node resnet_trigger --budget 5 "
            "--memory-mode append_only_summary_with_rationale "
            "--campaign-id ablation_append_only_summary_with_rationale "
            "--node-root nodes/ResNet_trigger --model ollama/qwen2.5-coder:7b"
        ),
        # Export paper tables (full mode)
        (
            "python3 scripts/export_kdd_tables.py "
            "--main-campaign kdd_main_5trial "
            "--ablation-campaigns ablation_none ablation_append_only_summary "
            "ablation_append_only_summary_with_rationale "
            "--stress-campaign kdd_stress_scope kdd_stress_noop --output-dir paper/tables/"
        ),
        # Export KDD figures
        (
            "python3 scripts/export_kdd_figures.py "
            "--figure architecture --output paper/figures/fig1_architecture.svg"
        ),
        (
            "python3 scripts/export_kdd_figures.py --figure repeated_bad_rate "
            "--input paper/tables/memory_ablation_summary.csv "
            "--output paper/figures/fig2_repeated_bad_rate.svg"
        ),
        (
            "python3 scripts/export_kdd_figures.py --figure decision_breakdown "
            "--input paper/tables/accepted_discarded_invalid_counts.csv "
            "--output paper/figures/fig3_decision_breakdown.svg"
        ),
        (
            "python3 scripts/export_kdd_figures.py --figure trajectory "
            "--input paper/tables/campaign_trajectory.csv "
            "--output paper/figures/fig4_trajectory.svg"
        ),
        # Artifact completeness check
        (
            "python3 scripts/check_kdd_artifact_completeness.py "
            "--campaigns kdd_main_5trial ablation_none "
            "ablation_append_only_summary "
            "ablation_append_only_summary_with_rationale "
            "kdd_stress_scope kdd_stress_noop"
        ),
        # Artifact manifest
        "python3 scripts/generate_artifact_manifest.py --output artifact_manifest.json",
    ]

    campaign_groups = {
        "kdd": kdd_campaigns,
        "memory_ablation": ablation_campaigns,
        "manager_comparison": manager_comparison_campaigns,
        "optional_p8": optional_p8_campaigns,
    }

    manifest = {
        "paper": "KDD AAE 2026 — Governed Autonomous Experimentation",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_version": "1.0",
        "campaigns": campaign_groups,
        "artifacts": _artifact_index(campaign_groups),
        "tables": table_files,
        "figures": {
            "kdd_paper": kdd_figures,
            "supplementary": supplementary_figures,
        },
        "environment": environment,
        "run_commands": run_commands,
        "notes": (
            "All primary campaigns (kdd_main_5trial, ablation_none, "
            "ablation_append_only_summary, ablation_append_only_summary_with_rationale) "
            "use claw_style_worker with prompt_manager running real training on the "
            "ResNet-trigger node (CPU fast-search mode). Stress trials use synthetic "
            "workers (stress_scope_violation_worker, stress_no_op_patch_worker) to "
            "exercise forced failure paths. The governed control plane — append-only "
            "ledger, pending-trial guard, scope validation, state machine — is "
            "exercised identically in all modes."
        ),
    }
    return manifest


def _check_files(manifest: dict) -> list[str]:
    """Return a list of missing file warnings."""
    warnings: list[str] = []

    def _check(rel_path: str) -> None:
        p = ROOT / rel_path
        if not p.exists():
            warnings.append(f"  MISSING: {rel_path}")

    # Campaign ledgers
    for group in manifest["campaigns"].values():
        for camp in group:
            _check(camp["ledger"])

    # Tables
    for t in manifest["tables"]:
        _check(t)

    # Figures
    for f in manifest["figures"]["kdd_paper"]:
        _check(f)
    for f in manifest["figures"]["supplementary"]:
        _check(f)

    # Real artifact refs. Synthetic dry-run refs are placeholders, so they are
    # indexed but do not count as missing files.
    for trial_artifacts in manifest.get("artifacts", {}).values():
        for trial in trial_artifacts:
            for key in ("patch_ref", "raw_log_ref"):
                ref = trial.get(key) or {}
                if not ref.get("path") or ref.get("synthetic"):
                    continue
                _check(str(ref["path"]))

    # Canonical commands should not point reviewers at scripts that do not
    # exist.  These commands intentionally start with ``python scripts/...`` so
    # the check stays simple and visible.
    for command in manifest.get("run_commands", []):
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        for i, token in enumerate(tokens[:-1]):
            if token.startswith("python") and tokens[i + 1].startswith("scripts/"):
                _check(tokens[i + 1])

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk 3.4: generate artifact_manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifact_manifest.json"),
        help="Output path (default: artifact_manifest.json in repo root)",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    warnings = _check_files(manifest)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to: {out_path}")

    if warnings:
        print(f"\nWarnings ({len(warnings)} missing files):")
        for w in warnings:
            print(w)
    else:
        print("All referenced files exist. ✓")

    return 0


if __name__ == "__main__":
    sys.exit(main())
