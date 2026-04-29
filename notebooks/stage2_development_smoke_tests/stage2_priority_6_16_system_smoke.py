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

from autoresearch.control_plane.budget import BudgetExceededError, BudgetState
from autoresearch.control_plane.decision import decide_trial
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision, ValidityStatus
from autoresearch.nodes.registry import load_registered_node


class Stage2PrioritySixToSixteenSmokeTests(unittest.TestCase):
    def test_decision_and_budget_modules(self) -> None:
        decision = decide_trial(
            validity_status=ValidityStatus.VALID,
            candidate_metric=0.8,
            current_best_metric=0.79,
            metric_direction="maximize",
        )
        self.assertEqual(decision.decision, TrialDecision.KEPT)

        budget = BudgetState(total_trials=2)
        budget = budget.consume_one()
        budget = budget.consume_one()
        self.assertTrue(budget.exhausted)
        with self.assertRaises(BudgetExceededError):
            budget.consume_one()

    def test_node_inspection_cli(self) -> None:
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "inspect_node.py"), "--node", "resnet_trigger"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["name"], "resnet_trigger")
        self.assertEqual(payload["editable_paths"], ["train.py"])

    def test_dry_campaign_and_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger = tmp / "smoke_trials.jsonl"
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_campaign.py"),
                    "--node",
                    "resnet_trigger",
                    "--campaign-id",
                    "smoke",
                    "--budget",
                    "3",
                    "--manager",
                    "baseline_manager",
                    "--memory-mode",
                    "none",
                    "--records",
                    str(ledger),
                    "--tables-dir",
                    str(tmp / "run_campaign_tables"),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            records = TrialAppendStore(ledger).read_all()
            self.assertEqual(len(records), 3)

            tables = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_paper_tables.py"),
                    "--campaign-id",
                    "smoke",
                    "--records",
                    str(ledger),
                    "--out-dir",
                    str(tmp / "tables"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(tables.returncode, 0, tables.stderr)
            self.assertTrue((tmp / "tables" / "main_campaign_summary.csv").exists())
            self.assertTrue((tmp / "tables" / "governance_metrics.csv").exists())

            figures = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_paper_figures.py"),
                    "--campaign-id",
                    "smoke",
                    "--records",
                    str(ledger),
                    "--out-dir",
                    str(tmp / "figures"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(figures.returncode, 0, figures.stderr)
            self.assertTrue((tmp / "figures" / "campaign_trajectory.csv").exists())
            with (tmp / "figures" / "accepted_discarded_invalid_counts.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["decision"] for row in rows}, {"kept", "discarded", "failed_invalid"})

    def test_memory_ablation_executes_dry_campaigns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "memory_ablation_summary.csv"
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_memory_ablation.py"),
                    "--node",
                    "resnet_trigger",
                    "--budget",
                    "2",
                    "--execute-dry-campaigns",
                    "--ledger-dir",
                    str(Path(tmpdir) / "ledgers"),
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
            self.assertEqual(len(payload["campaign_outputs"]), 3)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
