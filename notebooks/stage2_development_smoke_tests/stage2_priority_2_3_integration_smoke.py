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

from autoresearch.control_plane.lifecycle import (
    build_trial_record_from_legacy_result,
    build_trial_records_from_legacy_loop_result,
)
from autoresearch.evaluation.campaign_summary import load_campaign_summary
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.provenance import build_trial_provenance
from autoresearch.memory.schemas import (
    ExecutionStatus,
    TrialDecision,
    TrialRecord,
    ValidityStatus,
)
from autoresearch.nodes.resnet_trigger.metric_parser import MetricParseError, parse_val_auc
from autoresearch.nodes.spec import load_node_spec
from autoresearch.reporting.export_tables import export_campaign_tables


class Stage2PriorityTwoThreeTests(unittest.TestCase):
    def test_resnet_metric_parser_converts_val_bpb_to_auc(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "some setup",
                        "val_bpb:          0.212400",
                        "val_auc:          0.700000",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_val_auc(log_path)

            self.assertEqual(parsed.metric_name, "val_auc")
            self.assertAlmostEqual(parsed.metric_value, 0.7876)
            self.assertEqual(parsed.metric_direction, "maximize")
            self.assertEqual(parsed.raw_metrics["val_bpb"], 0.2124)

    def test_resnet_metric_parser_rejects_missing_metric(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text("training failed before metrics\n", encoding="utf-8")

            with self.assertRaises(MetricParseError):
                parse_val_auc(log_path)

    def test_append_store_preserves_records_in_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trials.jsonl"
            store = TrialAppendStore(path)
            first = _record("trial-001", 1, 0.78, TrialDecision.KEPT)
            second = _record("trial-002", 2, 0.77, TrialDecision.DISCARDED)

            store.append(first)
            store.append(second)
            loaded = store.read_all()

            self.assertEqual([record.trial_id for record in loaded], ["trial-001", "trial-002"])
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)
            self.assertEqual(loaded[0].provenance.proposal_id, first.provenance.proposal_id)

    def test_lifecycle_converts_legacy_result_to_stage2_record(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")
        legacy_result = {
            "run": {
                "commit": "after123",
                "base_commit": "before123",
                "recommended_status": "keep",
                "description": "lr tweak",
                "state": {
                    "best_bpb": 0.2200,
                    "pending_experiment": {
                        "packet": {
                            "description": "lr tweak",
                            "objective": "Lower learning rate from 1e-3 to 5e-4.",
                        }
                    },
                },
                "worker": {"last_result": {"changed_files": ["train.py"]}},
                "experiment": {
                    "success": True,
                    "val_bpb": 0.2124,
                    "log_path": "run.log",
                },
            },
            "decision": {"decision": "keep", "rationale": "AUC improved."},
        }

        record = build_trial_record_from_legacy_result(
            legacy_result=legacy_result,
            node_spec=spec,
            campaign_id="campaign-smoke",
            budget_index=1,
            manager_mode="prompt_manager",
            worker_mode="claw_style_worker",
            memory_mode="append_only_summary",
        )

        self.assertEqual(record.node_id, "resnet_trigger")
        self.assertEqual(record.decision, TrialDecision.KEPT)
        self.assertEqual(record.validity_status, ValidityStatus.VALID)
        self.assertAlmostEqual(record.parsed_metrics["val_auc"], 0.7876)
        self.assertAlmostEqual(record.current_best_before or 0.0, 0.78)
        self.assertAlmostEqual(record.delta_vs_best or 0.0, 0.0076)

    def test_lifecycle_converts_legacy_loop_history_to_records(self) -> None:
        spec = load_node_spec(ROOT / "configs" / "nodes" / "resnet_trigger.yaml")
        legacy_loop_result = {
            "history": [
                {
                    "run": {
                        "commit": "after123",
                        "base_commit": "before123",
                        "recommended_status": "discard",
                        "state": {"best_bpb": 0.2200, "pending_experiment": {"packet": {"description": "try dropout"}}},
                        "worker": {"last_result": {"changed_files": ["train.py"]}},
                        "experiment": {"success": True, "val_bpb": 0.2300, "log_path": "run.log"},
                    },
                    "decision": {"decision": "discard"},
                }
            ]
        }

        records = build_trial_records_from_legacy_loop_result(
            legacy_loop_result=legacy_loop_result,
            node_spec=spec,
            campaign_id="campaign-smoke",
            manager_mode="prompt_manager",
            worker_mode="claw_style_worker",
            memory_mode="append_only_summary",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].budget_index, 1)
        self.assertEqual(records[0].decision, TrialDecision.DISCARDED)

    def test_summary_exports_paper_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger = tmp / "trials.jsonl"
            store = TrialAppendStore(ledger)
            store.append(_record("trial-001", 1, 0.78, TrialDecision.KEPT, current_best_before=0.75))
            store.append(_record("trial-002", 2, 0.77, TrialDecision.DISCARDED, current_best_before=0.78))
            store.append(_record("trial-003", 3, 0.79, TrialDecision.KEPT, current_best_before=0.78))

            summary = load_campaign_summary(ledger)
            outputs = export_campaign_tables(summary, tmp / "tables")

            self.assertEqual(summary.metrics.total_trials, 3)
            self.assertEqual(summary.metrics.kept_count, 2)
            self.assertAlmostEqual(summary.metrics.best_metric or 0.0, 0.79)
            self.assertAlmostEqual(summary.metrics.net_gain or 0.0, 0.04)
            self.assertTrue(outputs["main_campaign_summary"].exists())
            self.assertTrue(outputs["governance_metrics"].exists())

            with outputs["governance_metrics"].open("r", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["total_trials"], "3")
            self.assertEqual(row["kept_count"], "2")

    def test_summarize_campaign_script_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger = tmp / "trials.jsonl"
            TrialAppendStore(ledger).append(_record("trial-001", 1, 0.78, TrialDecision.KEPT, current_best_before=0.75))

            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_campaign.py"),
                    "--campaign-id",
                    "campaign-smoke",
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

            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(payload["metrics"]["total_trials"], 1)
            self.assertTrue((tmp / "tables" / "main_campaign_summary.csv").exists())
            self.assertTrue((tmp / "tables" / "governance_metrics.csv").exists())


def _record(
    trial_id: str,
    budget_index: int,
    val_auc: float,
    decision: TrialDecision,
    current_best_before: float | None = 0.75,
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
        memory_mode="append_only_summary",
        proposal_summary=f"proposal {budget_index}",
        proposal_rationale="smoke test",
        targeted_files=("train.py",),
        patch_ref=f"experiments/artifacts/{trial_id}/patch.diff",
        git_commit_before="before",
        git_commit_after="after",
        execution_status=ExecutionStatus.SUCCESS,
        validity_status=ValidityStatus.VALID,
        failure_category=None,
        raw_log_ref=f"experiments/artifacts/{trial_id}/run.log",
        parsed_metrics={"val_auc": val_auc},
        current_best_before=current_best_before,
        delta_vs_best=(val_auc - current_best_before) if current_best_before is not None else None,
        decision=decision,
        decision_rationale="smoke test decision",
        wall_clock_seconds=60.0,
        cumulative_budget_consumed=budget_index,
        provenance=build_trial_provenance(trial_id),
    )


if __name__ == "__main__":
    unittest.main()
