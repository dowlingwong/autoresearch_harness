from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "nodes" / "resnet_trigger.yaml").exists():
            return parent
    raise RuntimeError("could not locate autoresearch_harness repo root")


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.evaluation.ablations import STAGE2_MEMORY_MODES, build_memory_ablation_plan
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.baseline_manager import BaselineManager
from autoresearch.manager.prompt_manager import PromptManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.provenance import build_trial_provenance
from autoresearch.memory.schemas import (
    ExecutionStatus,
    FailureCategory,
    TrialDecision,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.memory.similarity import compute_repeated_bad_stats
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.spec import load_node_spec


class Stage2PriorityFourManagerSmokeTests(unittest.TestCase):
    def test_memory_modes_create_expected_contexts(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")
        records = _records()

        none = build_memory_context(records, MemoryMode.NONE, spec, budget_index=3)
        summary = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY, spec, budget_index=3)
        rationale = build_memory_context(records, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE, spec, budget_index=3)

        self.assertIn("editable_paths=train.py", none.context_text)
        self.assertNotIn("AUC degraded", summary.context_text)
        self.assertIn("AUC degraded", rationale.context_text)
        self.assertLess(summary.compressed_chars, summary.raw_memory_chars)
        self.assertGreater(rationale.repeated_bad_stats.repeated_bad_count, 0)

    def test_repeated_bad_detector_flags_duplicate_bad_proposal(self) -> None:
        stats = compute_repeated_bad_stats(_records())

        self.assertEqual(stats.repeated_bad_count, 1)
        self.assertEqual(stats.repeated_degraded_count, 1)
        self.assertEqual(stats.flagged_trial_ids, ("trial-003",))

    def test_ablation_plan_has_three_equal_budget_modes(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")
        rows = build_memory_ablation_plan(spec, _records(), budget=3)

        self.assertEqual([row.memory_mode for row in rows], [mode.value for mode in STAGE2_MEMORY_MODES])
        self.assertEqual({row.planned_trials for row in rows}, {3})
        self.assertEqual(rows[-1].repeated_bad_count, 1)

    def test_run_memory_ablation_script_dry_run_writes_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger = tmp / "trials.jsonl"
            TrialAppendStore(ledger).append_many(_records())
            out = tmp / "memory_ablation_summary.csv"

            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_memory_ablation.py"),
                    "--node",
                    "resnet_trigger",
                    "--budget",
                    "3",
                    "--records",
                    str(ledger),
                    "--dry-run",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(len(payload["rows"]), 3)
            self.assertTrue(out.exists())
            with out.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["budget"], "3")

    def test_managers_return_structured_bounded_proposals(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")
        context = build_memory_context(_records(), MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE, spec, budget_index=3)
        status = ManagerStatus(
            campaign_id="campaign-smoke",
            budget_index=3,
            current_best_metric=0.7876,
            metric_name="val_auc",
            metric_direction="maximize",
        )

        baseline = BaselineManager().propose_next_trial(status, context, spec)
        prompt = PromptManager().propose_next_trial(status, context, spec)

        self.assertEqual(baseline.target_files, ("train.py",))
        self.assertIn("WEIGHT_DECAY", baseline.objective)
        self.assertEqual(prompt.target_files, ("train.py",))
        self.assertIn("Do not edit frozen paths", prompt.objective)
        self.assertIn("trial-003", prompt.objective)


def _records() -> list[TrialRecord]:
    return [
        _record(
            trial_id="trial-001",
            budget_index=1,
            proposal_summary="Lower learning rate",
            proposal_rationale="decrease lr from 1e-3 to 5e-4",
            decision=TrialDecision.KEPT,
            val_auc=0.7876,
            current_best_before=0.7799,
            decision_rationale="AUC improved",
        ),
        _record(
            trial_id="trial-002",
            budget_index=2,
            proposal_summary="Lower learning rate again",
            proposal_rationale="decrease lr from 5e-4 to 1e-4",
            decision=TrialDecision.DISCARDED,
            val_auc=0.7700,
            current_best_before=0.7876,
            decision_rationale="AUC degraded",
        ),
        _record(
            trial_id="trial-003",
            budget_index=3,
            proposal_summary="Lower learning rate again",
            proposal_rationale="decrease lr further",
            decision=TrialDecision.DISCARDED,
            val_auc=0.7600,
            current_best_before=0.7876,
            decision_rationale="AUC degraded",
        ),
    ]


def _record(
    *,
    trial_id: str,
    budget_index: int,
    proposal_summary: str,
    proposal_rationale: str,
    decision: TrialDecision,
    val_auc: float,
    current_best_before: float,
    decision_rationale: str,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        campaign_id="campaign-smoke",
        node_id="resnet_trigger",
        budget_index=budget_index,
        timestamp_start="2026-04-28T00:00:00Z",
        timestamp_end="2026-04-28T00:01:00Z",
        manager_mode="baseline_manager",
        worker_mode="claw_style_worker",
        memory_mode="append_only_summary_with_rationale",
        proposal_summary=proposal_summary,
        proposal_rationale=proposal_rationale,
        targeted_files=("train.py",),
        patch_ref=f"experiments/artifacts/{trial_id}/patch.diff",
        git_commit_before="before",
        git_commit_after="after",
        execution_status=ExecutionStatus.SUCCESS,
        validity_status=ValidityStatus.VALID,
        failure_category=FailureCategory.DEGRADED_METRIC if decision == TrialDecision.DISCARDED else None,
        raw_log_ref=f"experiments/artifacts/{trial_id}/run.log",
        parsed_metrics={"val_auc": val_auc},
        current_best_before=current_best_before,
        delta_vs_best=val_auc - current_best_before,
        decision=decision,
        decision_rationale=decision_rationale,
        wall_clock_seconds=60.0,
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
    )


if __name__ == "__main__":
    unittest.main()
