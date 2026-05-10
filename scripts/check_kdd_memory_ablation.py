#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

DEFAULT_CAMPAIGNS = (
    "ablation_none",
    "ablation_append_only_summary",
    "ablation_append_only_summary_with_rationale",
)
EXPECTED_MODES = (
    "none",
    "append_only_summary",
    "append_only_summary_with_rationale",
)


@dataclass(frozen=True)
class AblationCheckRow:
    campaign_id: str
    expected_memory_mode: str
    ledger_exists: bool
    pending_guard_exists: bool
    total_trials: int
    worker_modes: tuple[str, ...]
    memory_modes: tuple[str, ...]
    kept: int
    discarded: int
    failed_invalid: int
    repeated_bad_count: int
    repeated_bad_rate: float
    evidence_type: str

    @property
    def complete(self) -> bool:
        return (
            self.ledger_exists
            and not self.pending_guard_exists
            and self.total_trials > 0
            and self.memory_modes == (self.expected_memory_mode,)
        )

    @property
    def is_real(self) -> bool:
        return self.evidence_type == "real"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether KDD memory-ablation ledgers are complete and real-run ready.",
    )
    parser.add_argument("--ledger-dir", default=str(ROOT / "experiments" / "ledgers"))
    parser.add_argument("--campaigns", nargs="+", default=list(DEFAULT_CAMPAIGNS))
    parser.add_argument("--expected-modes", nargs="+", default=list(EXPECTED_MODES))
    parser.add_argument("--expected-trials", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "paper" / "tables" / "memory_ablation_run_report.txt"))
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Fail if any ablation arm was produced only by DryRunWorker.",
    )
    args = parser.parse_args()

    if len(args.campaigns) != len(args.expected_modes):
        parser.error("--campaigns and --expected-modes must have the same length")

    rows = build_rows(Path(args.ledger_dir), tuple(args.campaigns), tuple(args.expected_modes))
    report = render_report(rows, expected_trials=args.expected_trials, require_real=args.require_real)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)

    complete = all(row.complete and row.total_trials == args.expected_trials for row in rows)
    real_ok = all(row.is_real for row in rows) if args.require_real else True
    return 0 if complete and real_ok else 1


def build_rows(
    ledger_dir: Path,
    campaigns: tuple[str, ...],
    expected_modes: tuple[str, ...],
) -> list[AblationCheckRow]:
    rows: list[AblationCheckRow] = []
    for campaign_id, mode in zip(campaigns, expected_modes):
        ledger = ledger_dir / f"{campaign_id}_trials.jsonl"
        pending_guard_exists = any(
            path.exists()
            for path in (
                ledger_dir / f"{campaign_id}_pending.json",
                ledger_dir / f"{campaign_id}_trials_pending.json",
            )
        )
        if not ledger.exists():
            rows.append(
                AblationCheckRow(
                    campaign_id=campaign_id,
                    expected_memory_mode=mode,
                    ledger_exists=False,
                    pending_guard_exists=pending_guard_exists,
                    total_trials=0,
                    worker_modes=(),
                    memory_modes=(),
                    kept=0,
                    discarded=0,
                    failed_invalid=0,
                    repeated_bad_count=0,
                    repeated_bad_rate=0.0,
                    evidence_type="missing",
                )
            )
            continue
        records = TrialAppendStore(ledger).read_all()
        worker_modes = tuple(sorted({record.worker_mode for record in records}))
        memory_modes = tuple(sorted({record.memory_mode for record in records}))
        stats = compute_repeated_bad_stats(records)
        rows.append(
            AblationCheckRow(
                campaign_id=campaign_id,
                expected_memory_mode=mode,
                ledger_exists=True,
                pending_guard_exists=pending_guard_exists,
                total_trials=len(records),
                worker_modes=worker_modes,
                memory_modes=memory_modes,
                kept=sum(1 for record in records if record.decision == TrialDecision.KEPT),
                discarded=sum(1 for record in records if record.decision == TrialDecision.DISCARDED),
                failed_invalid=sum(1 for record in records if record.decision == TrialDecision.FAILED_INVALID),
                repeated_bad_count=stats.repeated_bad_count,
                repeated_bad_rate=stats.repeated_bad_rate,
                evidence_type=_evidence_type(worker_modes),
            )
        )
    return rows


def render_report(rows: list[AblationCheckRow], *, expected_trials: int, require_real: bool) -> str:
    payload = {
        "expected_trials_per_mode": expected_trials,
        "require_real": require_real,
        "all_complete": all(row.complete and row.total_trials == expected_trials for row in rows),
        "all_real": all(row.is_real for row in rows),
        "rows": [
            {
                "campaign_id": row.campaign_id,
                "expected_memory_mode": row.expected_memory_mode,
                "ledger_exists": row.ledger_exists,
                "pending_guard_exists": row.pending_guard_exists,
                "total_trials": row.total_trials,
                "worker_modes": list(row.worker_modes),
                "memory_modes": list(row.memory_modes),
                "kept": row.kept,
                "discarded": row.discarded,
                "failed_invalid": row.failed_invalid,
                "repeated_bad_count": row.repeated_bad_count,
                "repeated_bad_rate": round(row.repeated_bad_rate, 4),
                "evidence_type": row.evidence_type,
            }
            for row in rows
        ],
    }
    if require_real and not payload["all_real"]:
        payload["status"] = "failed_require_real"
        payload["message"] = "At least one memory-ablation arm is dry-run or missing; run real campaigns before making the paper claim."
    elif not payload["all_complete"]:
        payload["status"] = "failed_incomplete"
        payload["message"] = "At least one memory-ablation ledger is missing, short, has a pending guard, or has the wrong memory_mode."
    else:
        payload["status"] = "passed"
        payload["message"] = "Memory-ablation ledgers are complete for the requested criteria."
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _evidence_type(worker_modes: tuple[str, ...]) -> str:
    if not worker_modes:
        return "missing"
    if worker_modes == ("dry_run_worker",):
        return "dry_run"
    if "dry_run_worker" in worker_modes:
        return "mixed"
    return "real"


if __name__ == "__main__":
    raise SystemExit(main())
