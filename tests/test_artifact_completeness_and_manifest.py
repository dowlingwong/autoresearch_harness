"""Tests for chunk 3.3 (artifact completeness checker) and chunk 3.4 (artifact manifest).

Coverage:
  - _is_dry_run_path: recognises synthetic dry-run paths
  - _resolve_path: rebases absolute paths through repo root when env differs
  - _check_trial: correct results for dry-run, real-run-exists, real-run-missing,
                   invalid trial, missing metrics, missing decision_id
  - check_campaign: missing ledger returns error; valid ledger returns per-trial checks
  - format_report: plain-text report includes campaign names and overall stats
  - main() CLI: exits 0 when all pass, 1 when any campaign below threshold
  - generate_artifact_manifest.build_manifest: valid JSON, required top-level keys,
    all referenced paths exist, campaigns have required fields
  - generate_artifact_manifest._check_files: returns empty list for real repo
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------- #
# Import modules under test                                               #
# ---------------------------------------------------------------------- #
import importlib.util


def _load_script(name: str, rel_path: str):
    """Load a scripts/ module by path (they are not installed packages).

    We register the module in sys.modules first so that dataclass and other
    decorators can resolve ``cls.__module__`` correctly.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    mod.__name__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


completeness_mod = _load_script(
    "check_kdd_artifact_completeness",
    "scripts/check_kdd_artifact_completeness.py",
)
manifest_mod = _load_script(
    "generate_artifact_manifest",
    "scripts/generate_artifact_manifest.py",
)


# ====================================================================== #
#  Chunk 3.3 — artifact completeness checker                             #
# ====================================================================== #


class TestIsDryRunPath(unittest.TestCase):
    def test_recognises_dry_run_path(self):
        self.assertTrue(
            completeness_mod._is_dry_run_path(
                "experiments/artifacts/dry_run_trial_001/patch.diff"
            )
        )

    def test_does_not_flag_real_path(self):
        self.assertFalse(
            completeness_mod._is_dry_run_path(
                "/Users/dev/autoresearch_harness/experiments/artifacts/kdd_stress_scope/trial-001/patch.diff"
            )
        )

    def test_empty_string(self):
        self.assertFalse(completeness_mod._is_dry_run_path(""))


class TestResolvePath(unittest.TestCase):
    def test_relative_path_resolved_under_root(self):
        rel = "experiments/ledgers/kdd_main_5trial_trials.jsonl"
        resolved = completeness_mod._resolve_path(rel)
        self.assertEqual(resolved, ROOT / rel)

    def test_absolute_existing_path_returned_as_is(self, tmp_path=None):
        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = Path(f.name)
        try:
            resolved = completeness_mod._resolve_path(str(tmp))
            self.assertEqual(resolved, tmp)
        finally:
            os.unlink(tmp)

    def test_absolute_nonexistent_rebased_via_repo_name(self):
        # Simulate a path recorded on a different host that shares the same
        # repo directory name.
        repo_name = ROOT.name  # e.g. "autoresearch_harness"
        fake_prefix = f"/other/host/projects/{repo_name}"
        # Use a file we know exists in our repo.
        real_rel = "experiments/ledgers/kdd_main_5trial_trials.jsonl"
        stored_abs = f"{fake_prefix}/{real_rel}"
        resolved = completeness_mod._resolve_path(stored_abs)
        expected = ROOT / real_rel
        # Only assert the rebasing worked if the expected file actually exists.
        if expected.exists():
            self.assertEqual(resolved, expected)
        else:
            # File doesn't exist in test environment — rebasing attempted but
            # fallback to original is also acceptable.
            self.assertIn(
                str(resolved),
                [stored_abs, str(expected)],
            )


class TestCheckTrial(unittest.TestCase):
    """Unit tests for _check_trial with constructed record dicts."""

    def _good_provenance(self):
        return {
            "decision_id": "decision-abc123",
            "metric_id": "metric-abc123",
            "patch_id": "patch-abc123",
            "proposal_id": "proposal-abc123",
            "run_id": "run-abc123",
        }

    def test_dry_run_trial_all_pass(self):
        record = {
            "trial_id": "test-trial-001",
            "budget_index": 1,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "dry_run",
            "patch_ref": "experiments/artifacts/dry_run_trial_001/patch.diff",
            "raw_log_ref": "experiments/artifacts/dry_run_trial_001/run.log",
            "parsed_metrics": {"val_auc": 0.81},
            "provenance": self._good_provenance(),
        }
        tc = completeness_mod._check_trial(record, ROOT)
        self.assertTrue(tc.is_dry_run)
        self.assertTrue(tc.patch_ref_ok)
        self.assertTrue(tc.raw_log_ref_ok)
        self.assertTrue(tc.metrics_ok)
        self.assertTrue(tc.decision_id_ok)
        self.assertTrue(tc.all_ok)
        self.assertEqual(tc.pass_count, 4)

    def test_valid_trial_missing_metrics_fails(self):
        record = {
            "trial_id": "test-trial-002",
            "budget_index": 2,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "dry_run",
            "patch_ref": "experiments/artifacts/dry_run_trial_002/patch.diff",
            "raw_log_ref": "experiments/artifacts/dry_run_trial_002/run.log",
            "parsed_metrics": {},
            "provenance": self._good_provenance(),
        }
        tc = completeness_mod._check_trial(record, ROOT)
        self.assertFalse(tc.metrics_ok)
        self.assertIn("MISSING", tc.metrics_note)

    def test_invalid_trial_empty_metrics_passes(self):
        """Metrics are not expected for invalid trials."""
        record = {
            "trial_id": "test-trial-003",
            "budget_index": 3,
            "decision": "failed_invalid",
            "validity_status": "invalid",
            "worker_mode": "dry_run",
            "patch_ref": "",
            "raw_log_ref": "",
            "parsed_metrics": {},
            "provenance": self._good_provenance(),
        }
        tc = completeness_mod._check_trial(record, ROOT)
        self.assertTrue(tc.metrics_ok)
        self.assertIn("n/a", tc.metrics_note)

    def test_missing_decision_id_fails(self):
        record = {
            "trial_id": "test-trial-004",
            "budget_index": 4,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "dry_run",
            "patch_ref": "experiments/artifacts/dry_run_trial_004/patch.diff",
            "raw_log_ref": "experiments/artifacts/dry_run_trial_004/run.log",
            "parsed_metrics": {"val_auc": 0.80},
            "provenance": {"decision_id": None},
        }
        tc = completeness_mod._check_trial(record, ROOT)
        self.assertFalse(tc.decision_id_ok)
        self.assertIn("MISSING", tc.decision_id_note)

    def test_real_run_existing_artifacts_pass(self, tmp_path=None):
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            patch_file = td_path / "patch.diff"
            log_file = td_path / "run.log"
            patch_file.write_text("--- a/train.py\n+++ b/train.py\n", encoding="utf-8")
            log_file.write_text("Training complete\n", encoding="utf-8")

            record = {
                "trial_id": "real-trial-001",
                "budget_index": 1,
                "decision": "kept",
                "validity_status": "valid",
                "worker_mode": "claw_worker",
                "patch_ref": str(patch_file),
                "raw_log_ref": str(log_file),
                "parsed_metrics": {"val_auc": 0.82},
                "provenance": self._good_provenance(),
            }
            tc = completeness_mod._check_trial(record, ROOT)
            self.assertFalse(tc.is_dry_run)
            self.assertTrue(tc.patch_ref_ok)
            self.assertTrue(tc.raw_log_ref_ok)
            self.assertTrue(tc.all_ok)

    def test_real_run_missing_artifacts_fail(self):
        record = {
            "trial_id": "real-trial-002",
            "budget_index": 2,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "claw_worker",
            "patch_ref": "/nonexistent/path/patch.diff",
            "raw_log_ref": "/nonexistent/path/run.log",
            "parsed_metrics": {"val_auc": 0.80},
            "provenance": {
                "trial_id": "real-trial-002",
                "decision_id": "decision-xyz",
                "metric_id": "metric-xyz",
                "patch_id": "patch-xyz",
                "proposal_id": "prop-xyz",
                "run_id": "run-xyz",
            },
        }
        tc = completeness_mod._check_trial(record, ROOT)
        self.assertFalse(tc.is_dry_run)
        self.assertFalse(tc.patch_ref_ok)
        self.assertFalse(tc.raw_log_ref_ok)
        self.assertIn("NOT FOUND", tc.patch_ref_note)


class TestCheckCampaign(unittest.TestCase):
    def test_missing_ledger_returns_error(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nonexistent_trials.jsonl"
            rep = completeness_mod.check_campaign("missing_camp", missing, ROOT)
            self.assertTrue(rep.error)
            self.assertEqual(rep.total_trials, 0)

    def test_valid_ledger_produces_trial_checks(self):
        import tempfile, json as _json

        records = [
            {
                "trial_id": "c-trial-001",
                "budget_index": 1,
                "decision": "kept",
                "validity_status": "valid",
                "worker_mode": "dry_run",
                "patch_ref": "experiments/artifacts/dry_run_trial_001/patch.diff",
                "raw_log_ref": "experiments/artifacts/dry_run_trial_001/run.log",
                "parsed_metrics": {"val_auc": 0.78},
                "provenance": {"decision_id": "decision-abc"},
            },
            {
                "trial_id": "c-trial-002",
                "budget_index": 2,
                "decision": "failed_invalid",
                "validity_status": "invalid",
                "worker_mode": "dry_run",
                "patch_ref": "",
                "raw_log_ref": "",
                "parsed_metrics": {},
                "provenance": {"decision_id": "decision-def"},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "test_trials.jsonl"
            with ledger.open("w") as f:
                for r in records:
                    f.write(_json.dumps(r) + "\n")
            rep = completeness_mod.check_campaign("test_camp", ledger, ROOT)
        self.assertEqual(rep.error, "")
        self.assertEqual(rep.total_trials, 2)
        self.assertEqual(rep.completeness_pct, 100.0)
        self.assertTrue(rep.meets_threshold)

    def test_below_threshold_campaign_flagged(self):
        import tempfile, json as _json

        # Real-run record with nonexistent artifacts
        record = {
            "trial_id": "bad-trial-001",
            "budget_index": 1,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "claw_worker",
            "patch_ref": "/nonexistent/patch.diff",
            "raw_log_ref": "/nonexistent/run.log",
            "parsed_metrics": {},
            "provenance": {"decision_id": None},
        }
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "bad_trials.jsonl"
            ledger.write_text(_json.dumps(record) + "\n", encoding="utf-8")
            rep = completeness_mod.check_campaign("bad_camp", ledger, ROOT)
        self.assertFalse(rep.meets_threshold)
        self.assertLess(rep.completeness_pct, 90.0)


class TestFormatReport(unittest.TestCase):
    def _make_report(self, campaign_id: str, completeness: float):
        import tempfile, json as _json

        is_dry = completeness == 100.0
        record = {
            "trial_id": f"{campaign_id}-trial-001",
            "budget_index": 1,
            "decision": "kept",
            "validity_status": "valid",
            "worker_mode": "dry_run" if is_dry else "claw_worker",
            "patch_ref": (
                "experiments/artifacts/dry_run_trial_001/patch.diff"
                if is_dry
                else "/nonexistent/patch.diff"
            ),
            "raw_log_ref": (
                "experiments/artifacts/dry_run_trial_001/run.log"
                if is_dry
                else "/nonexistent/run.log"
            ),
            "parsed_metrics": {"val_auc": 0.80} if is_dry else {},
            "provenance": {"decision_id": "decision-abc" if is_dry else None},
        }
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / f"{campaign_id}_trials.jsonl"
            ledger.write_text(_json.dumps(record) + "\n", encoding="utf-8")
            rep = completeness_mod.check_campaign(campaign_id, ledger, ROOT)
        return rep

    def test_report_contains_overall_line(self):
        rep = self._make_report("test_camp", 100.0)
        text = completeness_mod.format_report([rep])
        self.assertIn("Overall:", text)
        self.assertIn("100.0%", text)

    def test_report_contains_campaign_name(self):
        rep = self._make_report("my_campaign", 100.0)
        text = completeness_mod.format_report([rep])
        self.assertIn("my_campaign", text)

    def test_report_flags_below_threshold(self):
        rep = self._make_report("broken_camp", 0.0)
        text = completeness_mod.format_report([rep])
        self.assertIn("BELOW THRESHOLD", text)


class TestCompletenessCLI(unittest.TestCase):
    """Integration test: run the CLI via subprocess on real repo ledgers."""

    def test_cli_passes_on_real_campaigns(self):
        import subprocess

        campaign_ids = [
            "kdd_main_5trial",
            "ablation_none",
            "ablation_append_only_summary",
            "ablation_append_only_summary_with_rationale",
            "kdd_stress_scope",
            "kdd_stress_noop",
        ]
        result = subprocess.run(
            [
                sys.executable,
                "scripts/check_kdd_artifact_completeness.py",
                "--campaigns",
                *campaign_ids,
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("100.0%", result.stdout)
        for campaign_id in campaign_ids:
            self.assertIn(campaign_id, result.stdout)

    def test_checked_in_report_contains_all_plan_campaigns(self):
        report_path = ROOT / "paper" / "tables" / "artifact_completeness_report.txt"
        self.assertTrue(report_path.exists())
        text = report_path.read_text(encoding="utf-8")
        for campaign_id in (
            "kdd_main_5trial",
            "ablation_none",
            "ablation_append_only_summary",
            "ablation_append_only_summary_with_rationale",
            "kdd_stress_scope",
            "kdd_stress_noop",
        ):
            self.assertIn(campaign_id, text)


# ====================================================================== #
#  Chunk 3.4 — artifact manifest                                          #
# ====================================================================== #


class TestBuildManifest(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest_mod.build_manifest()

    def test_required_top_level_keys(self):
        for key in ("paper", "generated", "manifest_version", "campaigns",
                    "artifacts", "tables", "figures", "environment", "run_commands", "notes"):
            self.assertIn(key, self.manifest, msg=f"Missing key: {key}")

    def test_campaigns_has_expected_groups(self):
        camps = self.manifest["campaigns"]
        for group in ("kdd", "memory_ablation", "manager_comparison"):
            self.assertIn(group, camps)

    def test_each_campaign_has_required_fields(self):
        required = {"id", "description", "ledger", "trials", "worker", "manager", "memory_mode"}
        for group in self.manifest["campaigns"].values():
            for camp in group:
                missing = required - set(camp.keys())
                self.assertFalse(
                    missing,
                    msg=f"Campaign {camp.get('id')} missing fields: {missing}",
                )

    def test_tables_is_nonempty_list_of_strings(self):
        tables = self.manifest["tables"]
        self.assertIsInstance(tables, list)
        self.assertGreater(len(tables), 0)
        for t in tables:
            self.assertIsInstance(t, str)

    def test_figures_has_kdd_paper_and_supplementary(self):
        figs = self.manifest["figures"]
        self.assertIn("kdd_paper", figs)
        self.assertEqual(len(figs["kdd_paper"]), 4)  # fig1–fig4

    def test_environment_has_python_and_model(self):
        env = self.manifest["environment"]
        self.assertIn("python", env)
        self.assertIn("default_model", env)

    def test_run_commands_nonempty(self):
        self.assertGreater(len(self.manifest["run_commands"]), 0)

    def test_artifacts_index_covers_campaign_trials(self):
        artifacts = self.manifest["artifacts"]
        for group in self.manifest["campaigns"].values():
            for camp in group:
                campaign_id = camp["id"]
                self.assertIn(campaign_id, artifacts)
                self.assertEqual(len(artifacts[campaign_id]), camp["trials"])
                for trial in artifacts[campaign_id]:
                    self.assertIn("patch_ref", trial)
                    self.assertIn("raw_log_ref", trial)
                    for key in ("patch_ref", "raw_log_ref"):
                        ref = trial[key]
                        self.assertIn("path", ref)
                        self.assertIn("exists", ref)
                        self.assertIn("synthetic", ref)

    def test_real_artifact_refs_exist(self):
        for campaign_artifacts in self.manifest["artifacts"].values():
            for trial in campaign_artifacts:
                for key in ("patch_ref", "raw_log_ref"):
                    ref = trial[key]
                    if ref["path"] and not ref["synthetic"]:
                        self.assertTrue(
                            (ROOT / ref["path"]).exists(),
                            msg=f"Missing real artifact ref: {ref['path']}",
                        )

    def test_run_command_script_paths_exist(self):
        import shlex

        checked = []
        for command in self.manifest["run_commands"]:
            tokens = shlex.split(command)
            for i, token in enumerate(tokens[:-1]):
                if token.startswith("python") and tokens[i + 1].startswith("scripts/"):
                    checked.append(tokens[i + 1])
                    self.assertTrue(
                        (ROOT / tokens[i + 1]).exists(),
                        msg=f"Missing command script: {tokens[i + 1]}",
                    )
        self.assertGreater(len(checked), 0)

    def test_manifest_serialisable_to_json(self):
        text = json.dumps(self.manifest)
        reparsed = json.loads(text)
        self.assertEqual(reparsed["manifest_version"], "1.0")

    def test_all_referenced_files_exist(self):
        warnings = manifest_mod._check_files(self.manifest)
        # In a full repo checkout all files should exist; surface any that don't.
        missing = [w for w in warnings if "MISSING" in w]
        self.assertEqual(
            missing,
            [],
            msg="Some files referenced in manifest are missing:\n" + "\n".join(missing),
        )


class TestGenerateManifestCLI(unittest.TestCase):
    def test_cli_writes_valid_json(self):
        import subprocess, tempfile, os

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "manifest.json")
            result = subprocess.run(
                [sys.executable, "scripts/generate_artifact_manifest.py", "--output", out],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("campaigns", data)
            self.assertEqual(data["manifest_version"], "1.0")

    def test_cli_reports_all_files_exist(self):
        import subprocess, tempfile, os

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "manifest.json")
            result = subprocess.run(
                [sys.executable, "scripts/generate_artifact_manifest.py", "--output", out],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                timeout=30,
            )
            self.assertIn("All referenced files exist", result.stdout)


if __name__ == "__main__":
    unittest.main()
