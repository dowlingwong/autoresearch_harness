"""D27 (H4 + batch-4 F3 fix): record-only edit-novelty flags as a separate event.

Written against r1 at 9c4902d (before D27) where applicable; the batch-4 tests
for the F3 fix fail at 0f2d9e9.  unittest style follows the repository
convention.  AI-assisted (BUILD_BATCH3 H4, BUILD_BATCH4 H6);
ENGINEERING_ONLY.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.campaign import (
    _active_config_sha256,
    _config_sha256,
    _effective_config_at,
    _novelty_config,
    _project_active_config,
    run_real_campaign,
)
from autoresearch.control_plane.snapshot import (
    capture_snapshot,
    read_snapshot,
    resume_campaign,
    write_snapshot,
)
from autoresearch.manager.base import ManagerProposal
from autoresearch.manager.hyperparam_edits import build_effective_config
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import TrialRecord
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class _ConfigWritingWorker:
    """Writes a chosen LEARNING_RATE into train.py and returns a chosen metric."""

    mode = "config_writing_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        (self.node_root / "train.py").write_text(
            f"LEARNING_RATE = {spec['learning_rate']}\n", encoding="utf-8"
        )
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=True,
            parsed_metrics={node_spec.metric_name: spec["metric"]},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="b",
            git_commit_after="a",
        )


class _HiddenUnitsWorker:
    """Edits HIDDEN_UNITS in digits_mlp's train.py (not covered by build_effective_config)."""

    mode = "hidden_units_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        text = (self.node_root / "train.py").read_text(encoding="utf-8")
        (self.node_root / "train.py").write_text(
            re.sub(r"HIDDEN_UNITS = \d+", f"HIDDEN_UNITS = {spec['hidden_units']}", text),
            encoding="utf-8",
        )
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=True,
            parsed_metrics={node_spec.metric_name: spec["metric"]},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="b",
            git_commit_after="a",
        )


class _YamlWritingWorker:
    """Rewrites config.yaml from a per-trial mapping and returns a chosen metric."""

    mode = "yaml_writing_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        text = "".join(f"{key}: {value}\n" for key, value in spec["config"].items())
        (self.node_root / "config.yaml").write_text(text, encoding="utf-8")
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("config.yaml",),
            success=True,
            parsed_metrics={node_spec.metric_name: spec["metric"]},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="b",
            git_commit_after="a",
        )


class _StaticBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, node_spec, objective: str | None = None) -> None:
        self._proposal = ManagerProposal(
            manager_mode=self.mode,
            proposal_summary="static",
            proposal_rationale="r",
            target_files=node_spec.editable_paths,
            objective=objective or f"Change {node_spec.editable_symbols[0]} from 1 to 2.",
            extra={},
        )

    def propose_next_trial(self, status, memory_context, node_spec):
        return self._proposal


class _E2Recorder:
    """Snapshot hook: capture at E2 (the shape used by the E1b branch collector)."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.snapshots = []

    def __call__(self, context) -> None:
        if context.epoch != "E2":
            return
        snapshot = capture_snapshot(context)
        write_snapshot(snapshot, self.out_dir)
        self.snapshots.append(snapshot)


class _E1Recorder:
    """Snapshot hook: capture at E1 (before the captured trial is evaluated)."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.snapshots = []

    def __call__(self, context) -> None:
        if context.epoch != "E1":
            return
        snapshot = capture_snapshot(context)
        write_snapshot(snapshot, self.out_dir)
        self.snapshots.append(snapshot)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _novelty_payloads(path: Path) -> list[dict]:
    return [e["payload"] for e in _events(path) if e["event_type"] == "edit_novelty"]


class TestD27EditNoveltyFlags(unittest.TestCase):
    def _sequence(self):
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        trials = [
            {"learning_rate": "0.002", "metric": 0.90},  # T1: new config, discarded (below incumbent)
            {"learning_rate": "0.002", "metric": 0.90},  # T2: config seen before (T1)
            {"learning_rate": "0.001", "metric": 0.95},  # T3: equals incumbent, identical metric
            {"learning_rate": "0.003", "metric": 0.99},  # T4: new config, kept
            {"learning_rate": "0.003", "metric": 0.97},  # T5: equals (new) incumbent, seen before
        ]
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
            (node_root / "train.py").write_text("LEARNING_RATE = 0.001\n", encoding="utf-8")
            worker = _ConfigWritingWorker(node_root, trials)
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d27", budget=5, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=_StaticBackend(spec), event_store=CampaignEventStore(events_path),
                initial_current_best=0.95,
            )
            events = _events(events_path)
            records = TrialAppendStore(records_path).read_all()
        return [e["payload"] for e in events if e["event_type"] == "edit_novelty"], records

    def _expected_payload(self, rate: str, metric_flag: bool, equals: bool, seen: bool) -> dict:
        # lr_synthetic is undeclared: the active projection is the full config,
        # so the active hash and the active flags mirror the full ones.  The
        # novelty configuration covers every editable symbol (F7 fix).
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        effective = _novelty_config({"LEARNING_RATE": rate}, spec)
        sha = _config_sha256(effective)
        return {
            "config_equals_incumbent": equals,
            "config_seen_before": seen,
            "metric_identical_to_incumbent": metric_flag,
            "effective_config_sha256": sha,
            "active_config_sha256": sha,
            "active_config_equals_incumbent": equals,
            "active_config_seen_before": seen,
        }

    def test_flags_on_the_constructed_sequence(self) -> None:
        flags, _ = self._sequence()
        self.assertEqual(len(flags), 5)
        self.assertEqual(flags[0], self._expected_payload("0.002", False, False, False))
        # T2 repeats T1's effective config (T1 discarded; workspace restored).
        self.assertEqual(flags[1], self._expected_payload("0.002", False, False, True))
        # T3 is the incumbent's own config and metric.
        self.assertEqual(flags[2], self._expected_payload("0.001", True, True, False))
        # T4 is new and kept.
        self.assertEqual(flags[3], self._expected_payload("0.003", False, False, False))
        # T5 repeats the new incumbent's config and has been evaluated before.
        self.assertEqual(flags[4], self._expected_payload("0.003", False, True, True))
        # A fresh campaign records no "unknown before resume" marker.
        for payload in flags:
            self.assertNotIn("seen_before_resume", payload)

    def test_records_are_unchanged(self) -> None:
        _, records = self._sequence()
        self.assertEqual([r.decision.value for r in records], ["discarded", "discarded", "discarded", "kept", "discarded"])
        for record in records:
            payload = record.to_dict()
            self.assertNotIn("edit_novelty", payload)
            self.assertNotIn("config_equals_incumbent", payload)
            self.assertNotIn("config_seen_before", payload)
            self.assertNotIn("metric_identical_to_incumbent", payload)
            self.assertNotIn("effective_config_sha256", payload)
            self.assertNotIn("active_config_sha256", payload)
            self.assertNotIn("novelty", json.dumps(record.extra))
        # The record schema itself is untouched.
        self.assertEqual(
            set(records[0].to_dict()),
            set(TrialRecord.__dataclass_fields__),
        )


class TestD27F3ResumeFix(unittest.TestCase):
    """Review F3: configs evaluated before a resume must set config_seen_before."""

    def _root_with_snapshot(self, tmp: Path, root_trials: list[dict]):
        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        root_dir = tmp / "root"
        node_root = root_dir / "node"
        shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
        (node_root / "train.py").write_text("LEARNING_RATE = 0.001\n", encoding="utf-8")
        worker = _ConfigWritingWorker(node_root, root_trials)
        recorder = _E2Recorder(root_dir / "snapshot")
        run_real_campaign(
            node_spec=spec, campaign_id="f3", budget=1, manager_mode="baseline_manager",
            memory_mode="none", records_path=root_dir / "root_trials.jsonl", worker=worker,
            proposal_backend=_StaticBackend(spec), event_store=CampaignEventStore(root_dir / "events.jsonl"),
            initial_current_best=0.95, snapshot_hook=recorder,
        )
        self.assertEqual(worker.run_calls, len(root_trials))
        return spec, root_dir / "snapshot"

    def _resume(self, tmp: Path, spec, snapshot_dir: Path, branch_trials: list[dict], horizon: int,
                mutate_manifest=None):
        snapshot = read_snapshot(snapshot_dir)
        if mutate_manifest is not None:
            manifest_path = snapshot_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            mutate_manifest(manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            snapshot = read_snapshot(snapshot_dir)
        into = tmp / "branch"
        worker = _ConfigWritingWorker(into, branch_trials)
        result = resume_campaign(
            snapshot, into, first_action="continue", horizon=horizon,
            manager=_StaticBackend(spec), worker=worker,
        )
        return result, worker

    def test_f3_reproduction_config_seen_before_true_after_resume(self) -> None:
        # Reviewer reproduction: evaluate LEARNING_RATE=0.002 (discarded),
        # resume, propose 0.002 again.  Before the fix the flag was False.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec, snapshot_dir = self._root_with_snapshot(tmp, [{"learning_rate": "0.002", "metric": 0.90}])
            result, worker = self._resume(tmp, spec, snapshot_dir, [{"learning_rate": "0.002", "metric": 0.90}], 1)
            self.assertEqual(worker.run_calls, 1)
            flags = _novelty_payloads(result.restored.events_path)
        self.assertEqual(len(flags), 1)
        self.assertIs(flags[0]["config_seen_before"], True)
        self.assertIs(flags[0]["active_config_seen_before"], True)
        self.assertIs(flags[0]["config_equals_incumbent"], False)
        # New-format snapshot: the seeds are known; no "unknown" marker.
        self.assertNotIn("seen_before_resume", flags[0])

    def test_legacy_snapshot_resumes_with_unknown_marker(self) -> None:
        def _drop_sets(manifest):
            manifest.pop("seen_effective_hashes", None)
            manifest.pop("seen_active_hashes", None)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = load_registered_node("lr_synthetic", repo_root=ROOT)
            root_dir = tmp / "root"
            node_root = root_dir / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
            (node_root / "train.py").write_text("LEARNING_RATE = 0.001\n", encoding="utf-8")
            # Two pre-resume trials; the snapshot captures the second one.
            worker = _ConfigWritingWorker(node_root, [
                {"learning_rate": "0.002", "metric": 0.90},
                {"learning_rate": "0.003", "metric": 0.91},
            ])
            recorder = _E2Recorder(root_dir / "snapshot")
            run_real_campaign(
                node_spec=spec, campaign_id="legacy", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=root_dir / "root_trials.jsonl", worker=worker,
                proposal_backend=_StaticBackend(spec),
                event_store=CampaignEventStore(root_dir / "events.jsonl"),
                initial_current_best=0.95, snapshot_hook=recorder,
            )
            # The E2 captured trial is committed from the provisional record,
            # so the branch worker only serves the two continuation trials.
            result, _ = self._resume(
                tmp, spec, root_dir / "snapshot",
                [{"learning_rate": "0.003", "metric": 0.95},   # duplicate of the captured trial
                 {"learning_rate": "0.002", "metric": 0.99}],  # trial 1's pre-resume config
                2, mutate_manifest=_drop_sets,
            )
            flags = _novelty_payloads(result.restored.events_path)
        self.assertEqual(len(flags), 2)
        # Old format: the pre-resume set is unknown, recorded on the first
        # event only.  The captured trial is evaluated inside the segment
        # (review F9), so its duplicate is seen; trial 1's genuine pre-resume
        # config is not, and the marker says the set was unknown rather than
        # silently empty.
        self.assertEqual(flags[0]["seen_before_resume"], "unknown")
        self.assertIs(flags[0]["config_seen_before"], True)  # captured trial
        self.assertNotIn("seen_before_resume", flags[1])
        self.assertIs(flags[1]["config_seen_before"], False)  # pre-resume 0.002

    def test_snapshot_carries_sorted_seen_hash_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, snapshot_dir = self._root_with_snapshot(tmp, [{"learning_rate": "0.002", "metric": 0.90}])
            snapshot = read_snapshot(snapshot_dir)
            self.assertEqual(len(snapshot.seen_effective_hashes), 1)
            self.assertEqual(snapshot.seen_active_hashes, snapshot.seen_effective_hashes)

    def test_e1_resume_counts_the_captured_trial_as_seen(self) -> None:
        # Review F9: the captured E1 trial is evaluated inside the resumed
        # segment (the snapshot predates it); a later duplicate of its
        # configuration must set config_seen_before.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = load_registered_node("lr_synthetic", repo_root=ROOT)
            root_dir = tmp / "root"
            node_root = root_dir / "node"
            shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
            (node_root / "train.py").write_text("LEARNING_RATE = 0.001\n", encoding="utf-8")
            recorder = _E1Recorder(root_dir / "snapshot")
            run_real_campaign(
                node_spec=spec, campaign_id="f9", budget=1, manager_mode="baseline_manager",
                memory_mode="none", records_path=root_dir / "root_trials.jsonl",
                worker=_ConfigWritingWorker(node_root, [{"learning_rate": "0.002", "metric": 0.90}]),
                proposal_backend=_StaticBackend(spec),
                event_store=CampaignEventStore(root_dir / "events.jsonl"),
                initial_current_best=0.95, snapshot_hook=recorder,
            )
            snapshot = read_snapshot(root_dir / "snapshot")
            into = tmp / "branch"
            branch_worker = _ConfigWritingWorker(into, [
                {"learning_rate": "0.002", "metric": 0.90},  # the captured trial, replayed
                {"learning_rate": "0.002", "metric": 0.95},  # duplicate proposal
            ])
            result = resume_campaign(snapshot, into, first_action="continue", horizon=1,
                                     manager=_StaticBackend(spec), worker=branch_worker)
            flags = _novelty_payloads(result.restored.events_path)
        self.assertEqual(branch_worker.run_calls, 2)
        self.assertEqual(len(flags), 1)
        self.assertIs(flags[0]["config_seen_before"], True)
        self.assertIs(flags[0]["active_config_seen_before"], True)


class TestD27ActiveVsFullHashing(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_registered_node("openml_ionosphere", repo_root=ROOT)

    def _projection(self, constants: dict) -> dict:
        effective = build_effective_config(constants)
        return _project_active_config(effective, self.spec)

    def test_projection_follows_the_configs_own_model_type(self) -> None:
        base = {"model_type": "logistic_regression", "C": "1.0", "class_weight": "balanced",
                "n_estimators": "100", "max_depth": "5", "learning_rate": "0.05",
                "scaler": "standard", "imputer": "most_frequent", "max_iter": "1000"}
        logreg = self._projection(base)
        self.assertIn("class_weight", logreg)
        self.assertNotIn("n_estimators", logreg)
        self.assertNotIn("imputer", logreg)  # dataset_excluded
        gb = self._projection({**base, "model_type": "gradient_boosting"})
        self.assertIn("n_estimators", gb)
        self.assertIn("learning_rate", gb)
        self.assertNotIn("class_weight", gb)

    def test_inert_key_change_same_active_hash_different_full_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "openml_ionosphere"
            shutil.copytree(ROOT / "nodes" / "openml_ionosphere", node_root,
                            ignore=shutil.ignore_patterns("__pycache__", "data_cache"))
            worker = SimpleNamespace(node_root=node_root)
            baseline = _effective_config_at(worker, self.spec)
            baseline_full = _config_sha256(baseline)
            baseline_active = _active_config_sha256(baseline, self.spec)
            self.assertNotEqual(baseline_active, baseline_full)  # declared node: projection bites

            text = (node_root / "config.yaml").read_text()
            # n_estimators is not active under logistic_regression.
            (node_root / "config.yaml").write_text(
                text.replace("n_estimators: 100", "n_estimators: 555")
            )
            inert = _effective_config_at(worker, self.spec)
            self.assertEqual(_active_config_sha256(inert, self.spec), baseline_active)
            self.assertNotEqual(_config_sha256(inert), baseline_full)

            # C is active under logistic_regression.
            (node_root / "config.yaml").write_text(
                text.replace("C: 0.15", "C: 0.9")
            )
            active = _effective_config_at(worker, self.spec)
            self.assertNotEqual(_active_config_sha256(active, self.spec), baseline_active)

    def test_event_level_projection_across_a_model_type_switch(self) -> None:
        gb_config = {
            "model_type": "gradient_boosting", "n_estimators": "100", "max_depth": "5",
            "learning_rate": "0.1", "class_weight": "balanced", "scaler": "standard",
            "imputer": "most_frequent", "max_iter": "1000", "C": "0.15",
        }
        inert_change = {**gb_config, "class_weight": "null"}
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            shutil.copytree(ROOT / "nodes" / "openml_ionosphere", node_root,
                            ignore=shutil.ignore_patterns("__pycache__", "data_cache"))
            worker = _YamlWritingWorker(node_root, [
                {"config": gb_config, "metric": 0.9},      # T1: switches LR -> GB, kept
                {"config": inert_change, "metric": 0.95},  # T2: class_weight inert under GB
            ])
            events_path = Path(tmp) / "events.jsonl"
            run_real_campaign(
                node_spec=self.spec, campaign_id="projection", budget=2,
                manager_mode="baseline_manager", memory_mode="none",
                records_path=Path(tmp) / "trials.jsonl", worker=worker,
                # model_type stays active under both declarations (D25 would
                # skip a proposal naming an inactive key before any worker run).
                proposal_backend=_StaticBackend(
                    self.spec,
                    objective="Change model_type from logistic_regression to gradient_boosting.",
                ),
                event_store=CampaignEventStore(events_path), initial_current_best=0.5,
            )
            flags = _novelty_payloads(events_path)
        self.assertEqual(len(flags), 2)
        # T1 projects under gradient_boosting (its own model_type), so the
        # active hash differs from the full hash.
        self.assertNotEqual(flags[0]["active_config_sha256"], flags[0]["effective_config_sha256"])
        # T2 changes only class_weight (inert under GB): same active hash as T1,
        # different full hash; the active flags see the incumbent / earlier
        # config while the full flags do not.
        self.assertEqual(flags[1]["active_config_sha256"], flags[0]["active_config_sha256"])
        self.assertNotEqual(flags[1]["effective_config_sha256"], flags[0]["effective_config_sha256"])
        self.assertIs(flags[1]["active_config_equals_incumbent"], True)
        self.assertIs(flags[1]["active_config_seen_before"], True)
        self.assertIs(flags[1]["config_equals_incumbent"], False)
        self.assertIs(flags[1]["config_seen_before"], False)


class TestD27EditableSymbolCoverage(unittest.TestCase):
    """Review F7: editable symbols outside build_effective_config's map must count.

    ``digits_mlp`` declares HIDDEN_UNITS, ALPHA and LEARNING_RATE_INIT among its
    editable symbols, none of which the narrow effective map covers.
    """

    def test_an_uncovered_editable_symbol_moves_the_config_hash(self) -> None:
        spec = load_registered_node("digits_mlp", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            shutil.copytree(ROOT / "nodes" / "digits_mlp", node_root)
            worker = SimpleNamespace(node_root=node_root)
            before = _effective_config_at(worker, spec)
            self.assertIn("HIDDEN_UNITS", before)
            self.assertEqual(before["HIDDEN_UNITS"], "32")
            text = (node_root / "train.py").read_text(encoding="utf-8")
            (node_root / "train.py").write_text(text.replace("HIDDEN_UNITS = 32", "HIDDEN_UNITS = 96"))
            after = _effective_config_at(worker, spec)
        self.assertEqual(after["HIDDEN_UNITS"], "96")
        self.assertNotEqual(_config_sha256(before), _config_sha256(after))

    def test_novelty_flags_see_an_uncovered_editable_symbol(self) -> None:
        spec = load_registered_node("digits_mlp", repo_root=ROOT)
        trials = [
            {"hidden_units": "64", "metric": 0.90},  # T1: new config, kept
            {"hidden_units": "96", "metric": 0.95},  # T2: a different new config
        ]
        with tempfile.TemporaryDirectory() as tmp:
            node_root = Path(tmp) / "node"
            shutil.copytree(ROOT / "nodes" / "digits_mlp", node_root)
            worker = _HiddenUnitsWorker(node_root, trials)
            events_path = Path(tmp) / "events.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f7", budget=2, manager_mode="baseline_manager",
                memory_mode="none", records_path=Path(tmp) / "trials.jsonl", worker=worker,
                proposal_backend=_StaticBackend(spec), event_store=CampaignEventStore(events_path),
                initial_current_best=0.5,
            )
            flags = _novelty_payloads(events_path)
        self.assertEqual(len(flags), 2)
        self.assertIs(flags[1]["config_equals_incumbent"], False)
        self.assertIs(flags[1]["config_seen_before"], False)
        self.assertIs(flags[1]["active_config_equals_incumbent"], False)
        self.assertIs(flags[1]["active_config_seen_before"], False)


if __name__ == "__main__":
    unittest.main()
