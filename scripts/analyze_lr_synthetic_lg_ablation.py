#!/usr/bin/env python3
"""Summarize and validate the lr_synthetic LangGraph memory ablation."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.memory.similarity import compute_repeated_bad_stats
from autoresearch.nodes.registry import load_registered_node

CAMPAIGN_IDS = {
    "none": "lr_synth_lg_none",
    "summary": "lr_synth_lg_summary",
    "rationale": "lr_synth_lg_rationale",
}
MEMORY_MODES = {
    "none": "none",
    "summary": "append_only_summary",
    "rationale": "append_only_summary_with_rationale",
}
PAPER_DIR = ROOT / "A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation"


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    campaign_id: str
    expected_memory_mode: str
    memory_modes: tuple[str, ...]
    ledger_exists: bool
    pending_guard_exists: bool
    total_trials: int
    kept: int
    discarded: int
    failed_invalid: int
    acceptance_rate: float
    repeated_bad_count: int
    repeated_bad_rate: float
    repeated_invalid_count: int
    repeated_degraded_count: int
    best_val_score: float | None
    manager_modes: tuple[str, ...]
    worker_modes: tuple[str, ...]
    edited_symbols: tuple[str, ...]
    all_symbols_allowed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "campaign_id": self.campaign_id,
            "expected_memory_mode": self.expected_memory_mode,
            "memory_modes": ";".join(self.memory_modes),
            "ledger_exists": self.ledger_exists,
            "pending_guard_exists": self.pending_guard_exists,
            "total_trials": self.total_trials,
            "kept": self.kept,
            "discarded": self.discarded,
            "failed_invalid": self.failed_invalid,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "repeated_bad_count": self.repeated_bad_count,
            "repeated_bad_rate": round(self.repeated_bad_rate, 4),
            "repeated_invalid_count": self.repeated_invalid_count,
            "repeated_degraded_count": self.repeated_degraded_count,
            "best_val_score": "" if self.best_val_score is None else round(self.best_val_score, 6),
            "manager_modes": ";".join(self.manager_modes),
            "worker_modes": ";".join(self.worker_modes),
            "edited_symbols": ";".join(self.edited_symbols),
            "all_symbols_allowed": self.all_symbols_allowed,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze lr_synthetic LangGraph memory-ablation ledgers.",
    )
    parser.add_argument("--ledger-dir", default=str(ROOT / "experiments" / "ledgers"))
    parser.add_argument("--expected-trials", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-failed-invalid",
        action="store_true",
        help="Do not fail strict mode solely because an arm contains failed-invalid trials.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "paper" / "tables" / "lg_ablation" / "lr_synthetic_lg_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "paper" / "tables" / "lg_ablation" / "lr_synthetic_lg_ablation_report.json"),
    )
    args = parser.parse_args()

    spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    allowed_symbols = set(spec.editable_symbols)
    rows = [
        summarize_arm(
            arm=arm,
            campaign_id=campaign_id,
            expected_mode=MEMORY_MODES[arm],
            ledger_dir=Path(args.ledger_dir),
            metric_name=spec.metric_name,
            allowed_symbols=allowed_symbols,
        )
        for arm, campaign_id in CAMPAIGN_IDS.items()
    ]
    report = build_report(rows, expected_trials=args.expected_trials, allow_failed_invalid=args.allow_failed_invalid)
    write_outputs(rows, report, Path(args.output_csv), Path(args.output_json))
    print(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.strict and not report["strict_ok"]:
        return 1
    return 0


def summarize_arm(
    *,
    arm: str,
    campaign_id: str,
    expected_mode: str,
    ledger_dir: Path,
    metric_name: str,
    allowed_symbols: set[str],
) -> ArmSummary:
    ledger = ledger_dir / f"{campaign_id}_trials.jsonl"
    pending_guard_exists = any(
        path.exists()
        for path in (
            ledger_dir / f"{campaign_id}_pending.json",
            ledger_dir / f"{campaign_id}_trials_pending.json",
        )
    )
    if not ledger.exists():
        return ArmSummary(
            arm=arm,
            campaign_id=campaign_id,
            expected_memory_mode=expected_mode,
            memory_modes=(),
            ledger_exists=False,
            pending_guard_exists=pending_guard_exists,
            total_trials=0,
            kept=0,
            discarded=0,
            failed_invalid=0,
            acceptance_rate=0.0,
            repeated_bad_count=0,
            repeated_bad_rate=0.0,
            repeated_invalid_count=0,
            repeated_degraded_count=0,
            best_val_score=None,
            manager_modes=(),
            worker_modes=(),
            edited_symbols=(),
            all_symbols_allowed=False,
        )

    records = TrialAppendStore(ledger).read_all()
    stats = compute_repeated_bad_stats(records)
    metrics = [record.parsed_metrics[metric_name] for record in records if metric_name in record.parsed_metrics]
    edited_symbols = tuple(sorted({symbol for record in records for symbol in _edited_symbols(record)}))
    return ArmSummary(
        arm=arm,
        campaign_id=campaign_id,
        expected_memory_mode=expected_mode,
        memory_modes=tuple(sorted({record.memory_mode for record in records})),
        ledger_exists=True,
        pending_guard_exists=pending_guard_exists,
        total_trials=len(records),
        kept=sum(1 for record in records if record.decision == TrialDecision.KEPT),
        discarded=sum(1 for record in records if record.decision == TrialDecision.DISCARDED),
        failed_invalid=sum(1 for record in records if record.decision == TrialDecision.FAILED_INVALID),
        acceptance_rate=(
            sum(1 for record in records if record.decision == TrialDecision.KEPT) / len(records)
            if records else 0.0
        ),
        repeated_bad_count=stats.repeated_bad_count,
        repeated_bad_rate=stats.repeated_bad_rate,
        repeated_invalid_count=stats.repeated_invalid_count,
        repeated_degraded_count=stats.repeated_degraded_count,
        best_val_score=max(metrics) if metrics else None,
        manager_modes=tuple(sorted({record.manager_mode for record in records})),
        worker_modes=tuple(sorted({record.worker_mode for record in records})),
        edited_symbols=edited_symbols,
        all_symbols_allowed=bool(edited_symbols) and set(edited_symbols).issubset(allowed_symbols),
    )


def _edited_symbols(record) -> tuple[str, ...]:
    manager_extra = dict(record.extra.get("manager") or {})
    structured = dict(manager_extra.get("structured_edit") or {})
    symbol = str(structured.get("symbol") or "").strip()
    if symbol:
        return (symbol,)

    patch_ref = Path(record.patch_ref) if record.patch_ref else None
    if patch_ref is None or not patch_ref.exists():
        return ()
    symbols: list[str] = []
    pattern = re.compile(r"^\+([A-Z][A-Z0-9_]*)\s*=")
    for line in patch_ref.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            symbols.append(match.group(1))
    return tuple(symbols)


def build_report(
    rows: list[ArmSummary],
    *,
    expected_trials: int,
    allow_failed_invalid: bool,
) -> dict[str, object]:
    rbr_by_arm = {row.arm: row.repeated_bad_rate for row in rows if row.total_trials}
    summary_rbr = rbr_by_arm.get("summary")
    none_rbr = rbr_by_arm.get("none")
    rationale_rbr = rbr_by_arm.get("rationale")
    summary_ordering_holds = (
        summary_rbr is not None
        and none_rbr is not None
        and rationale_rbr is not None
        and summary_rbr < none_rbr
        and summary_rbr <= rationale_rbr
    )
    any_memory_improves = (
        summary_rbr is not None
        and rationale_rbr is not None
        and none_rbr is not None
        and min(summary_rbr, rationale_rbr) < none_rbr
    )
    strict_ok = all(
        row.ledger_exists
        and not row.pending_guard_exists
        and row.total_trials == expected_trials
        and row.memory_modes == (MEMORY_MODES[row.arm],)
        and row.manager_modes == ("langgraph_manager",)
        and row.worker_modes == ("local_worker",)
        and row.all_symbols_allowed
        and (allow_failed_invalid or row.failed_invalid == 0)
        for row in rows
    )
    return {
        "expected_trials_per_arm": expected_trials,
        "strict_ok": strict_ok,
        "summary_ordering_holds": summary_ordering_holds,
        "any_memory_arm_improves_over_none": any_memory_improves,
        "rows": [row.to_dict() for row in rows],
    }


def write_outputs(rows: list[ArmSummary], report: dict[str, object], csv_path: Path, json_path: Path) -> None:
    _write_csv(rows, csv_path)
    _write_json(report, json_path)

    if PAPER_DIR.exists():
        rel_csv = csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path.name
        rel_json = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path.name
        paper_csv = PAPER_DIR / "tables" / Path(rel_csv).name
        paper_json = PAPER_DIR / "tables" / Path(rel_json).name
        if "lg_ablation" in Path(rel_csv).parts:
            paper_csv = PAPER_DIR / "tables" / "lg_ablation" / Path(rel_csv).name
        if "lg_ablation" in Path(rel_json).parts:
            paper_json = PAPER_DIR / "tables" / "lg_ablation" / Path(rel_json).name
        _write_csv(rows, paper_csv)
        _write_json(report, paper_json)


def _write_csv(rows: list[ArmSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_dict()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def _write_json(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
