"""D25 (H2): node-declared active keys — prompt, validation, revisions, hashes.

Written against r1 at 9c4902d (before D25) where applicable; unittest style
follows the repository convention.  AI-assisted (BUILD_BATCH3, H2);
ENGINEERING_ONLY.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.budget import ComputeBudget
from autoresearch.control_plane.campaign import MAX_INACTIVE_KEY_REVISIONS, _manager_status, run_real_campaign
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.summarizer import build_memory_context
from autoresearch.nodes.active_keys import inactive_key_in_proposal, resolve_active_keys
from autoresearch.nodes.registry import load_registered_node, node_spec_path
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]

# sha256 of the LangGraphManager prompt for an undeclared node (lr_synthetic)
# with fixed status/memory, captured from r1 at 9c4902d (before D25).
UNDECLARED_PROMPT_SHA256 = "dacd99999c6b1b59c9e582c938a456276bd27964a4910c724623ce7f5ccec822"


def _proposal(node_spec, objective: str, summary: str = "p") -> ManagerProposal:
    return ManagerProposal(
        manager_mode="stub_manager",
        proposal_summary=summary,
        proposal_rationale="r",
        target_files=node_spec.editable_paths,
        objective=objective,
        extra={},
    )


class _StubManager:
    """Non-deterministic stub; accepts revision feedback and revision attempts."""

    deterministic = False

    def __init__(self, node_spec, proposals: list[ManagerProposal]) -> None:
        self._node_spec = node_spec
        self._proposals = list(proposals)
        self.calls: list[dict] = []

    def propose_next_trial(self, status, memory_context, node_spec, *, revision_feedback=None,
                           revision_attempt=None):
        self.calls.append({"feedback": revision_feedback, "attempt": revision_attempt})
        index = min(len(self.calls) - 1, len(self._proposals) - 1)
        return self._proposals[index]


class _CountingWorker:
    mode = "counting_worker"

    def __init__(self, node_root: Path, metric: float = 0.9) -> None:
        self.node_root = Path(node_root)
        self.metric = metric
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        self.run_calls += 1
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("config.yaml",),
            success=True,
            parsed_metrics={node_spec.metric_name: self.metric},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",  # nonexistent path: not an empty patch
            git_commit_before="b",
            git_commit_after="a",
        )


def _copy_openml(tmp: Path, node: str) -> Path:
    node_root = tmp / node
    shutil.copytree(ROOT / "nodes" / node, node_root, ignore=shutil.ignore_patterns("__pycache__", "data_cache"))
    return node_root


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestD25DeclarationResolution(unittest.TestCase):
    def test_ionosphere_logreg_shows_exactly_the_declared_active_keys(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        active = resolve_active_keys(spec, {"model_type": "logistic_regression"})
        self.assertEqual(active, ("model_type", "C", "class_weight", "max_iter", "scaler"))

    def test_ionosphere_gradient_boosting_has_its_own_keys(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        active = resolve_active_keys(spec, {"model_type": "gradient_boosting"})
        self.assertEqual(active, ("model_type", "n_estimators", "max_depth", "learning_rate", "scaler"))
        self.assertNotIn("class_weight", active)

    def test_credit_approval_keeps_imputer_because_it_has_categorical_columns(self) -> None:
        spec = load_registered_node("openml_credit_approval", repo_root=ROOT)
        active = resolve_active_keys(spec, {"model_type": "logistic_regression"})
        self.assertIn("imputer", active)

    def test_manager_status_keeps_the_full_editable_set(self) -> None:
        # Regression for review F1: the scripted stand-in reads this status, so
        # it must keep every editable constant (the D25 filter applies to the
        # LLM prompt only).  Filtering here silently changed the scripted
        # proposal sequence for the existing seeds.
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = SimpleNamespace(node_root=node_root)
            status = _manager_status(
                campaign_id="c", budget_index=1, current_best=None, node_spec=spec, worker=worker
            )
        self.assertIn("class_weight", status.current_constants)
        self.assertIn("imputer", status.current_constants)

    def test_prompt_active_set_follows_a_model_type_change_in_the_workspace(self) -> None:
        from autoresearch.manager.base import ManagerStatus
        from autoresearch.manager.langgraph_manager import _prepare_context

        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = SimpleNamespace(node_root=node_root)

            def prompt_for(status):
                memory = build_memory_context([], "none", spec, status.budget_index)
                state = {"status": status, "memory_context": memory, "node_spec": spec,
                         "context_text": "", "raw_proposal": "", "proposal": None,
                         "revision_feedback": None}
                return _prepare_context(state)["context_text"]

            before = _manager_status(
                campaign_id="c", budget_index=1, current_best=None, node_spec=spec, worker=worker
            )
            prompt = prompt_for(before)
            self.assertIn("class_weight = ", prompt)
            self.assertNotIn("imputer = ", prompt)

            config_text = (node_root / "config.yaml").read_text()
            (node_root / "config.yaml").write_text(
                config_text.replace("model_type: logistic_regression", "model_type: gradient_boosting")
            )
            after = _manager_status(
                campaign_id="c", budget_index=2, current_best=None, node_spec=spec, worker=worker
            )
            prompt_after = prompt_for(after)
        self.assertNotIn("class_weight = ", prompt_after)
        self.assertIn("learning_rate = ", prompt_after)
        self.assertNotIn("imputer = ", prompt_after)


class TestD25InactiveKeyValidation(unittest.TestCase):
    def test_inactive_key_is_rejected_before_any_worker_run(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = _CountingWorker(node_root)
            bad = _proposal(spec, "Change imputer from most_frequent to constant in config.yaml.")
            good = _proposal(spec, "Change C from 1.0 to 2.0 in config.yaml.", summary="good")
            manager = _StubManager(spec, [bad, good])
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d25a", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager, event_store=CampaignEventStore(events_path),
            )
            records = TrialAppendStore(records_path).read_all()
            events = _events(events_path)
        self.assertEqual(worker.run_calls, 1)
        self.assertEqual(records[0].decision.value, "kept")  # first valid metric
        self.assertEqual(manager.calls[0]["attempt"], None)
        self.assertEqual(manager.calls[1]["attempt"], 1)  # first revision seed differs
        rejected = [e for e in events if e["event_type"] == "proposal_rejected_inactive_key"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["payload"]["key"], "imputer")
        self.assertNotIn("skipped", rejected[0]["payload"])

    def test_exhausted_revisions_skip_with_reason_and_spend_no_run(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = _CountingWorker(node_root)
            bad = _proposal(spec, "Change imputer from most_frequent to constant in config.yaml.")
            manager = _StubManager(spec, [bad])
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d25b", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager, event_store=CampaignEventStore(events_path),
                compute_budget=ComputeBudget(max_llm_calls=10),
            )
            records = TrialAppendStore(records_path).read_all()
            events = _events(events_path)
        self.assertEqual(worker.run_calls, 0)
        record = records[0]
        self.assertEqual(record.decision.value, "skipped")
        self.assertEqual(record.extra["control"]["reason"], "inactive_key")
        self.assertEqual(record.extra["control"]["key"], "imputer")
        self.assertEqual(record.extra["control"]["revisions"], MAX_INACTIVE_KEY_REVISIONS)
        self.assertIn("inactive key", record.decision_rationale)
        # initial call + MAX revisions (each charged to the LLM-call cap; the
        # cap accounting itself is covered by test_llm_cap_stops_revisions...)
        self.assertEqual(manager.calls[0]["attempt"], None)
        self.assertEqual([c["attempt"] for c in manager.calls[1:]], [1, 2])
        rejected = [e for e in events if e["event_type"] == "proposal_rejected_inactive_key"]
        self.assertEqual(len(rejected), MAX_INACTIVE_KEY_REVISIONS + 1)
        self.assertTrue(rejected[-1]["payload"]["skipped"])

    def test_llm_cap_stops_revisions_before_the_limit(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = _CountingWorker(node_root)
            bad = _proposal(spec, "Change imputer from most_frequent to constant in config.yaml.")
            manager = _StubManager(spec, [bad])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d25c", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager, compute_budget=ComputeBudget(max_llm_calls=1),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(worker.run_calls, 0)
        self.assertEqual(records[0].decision.value, "skipped")
        self.assertEqual(len(manager.calls), 1)  # cap reached after the initial call
        self.assertEqual(records[0].extra["control"]["revisions"], 0)

    def _status_with_full_constants(self, spec, constants):
        from autoresearch.manager.base import ManagerStatus

        return ManagerStatus(
            campaign_id="c", budget_index=1, current_best_metric=None,
            metric_name=spec.metric_name, metric_direction=spec.metric_direction,
            current_constants=dict(constants),
        )

    def _langgraph_proposal(self, spec, status, response: dict):
        from autoresearch.manager.langgraph_manager import LangGraphManager

        class _LLM:
            def invoke(self, messages):
                return SimpleNamespace(content=json.dumps(response))

        return LangGraphManager(llm=_LLM()).propose_next_trial(
            status, build_memory_context([], "none", spec, 1), spec
        )

    def test_inactive_key_in_the_live_status_is_caught_via_the_structured_edit(self) -> None:
        # The campaign status keeps the full editable set (review F1), so an
        # inactive named key resolves to a structured edit and is caught there.
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        status = self._status_with_full_constants(spec, {
            "model_type": "logistic_regression", "C": "1.0", "class_weight": "null",
            "max_iter": "1000", "scaler": "none", "imputer": "most_frequent",
        })
        proposal = self._langgraph_proposal(spec, status, {
            "summary": "s", "rationale": "r",
            "objective": "Change imputer from most_frequent to constant in config.yaml.",
            "param": "imputer", "old_value": "most_frequent", "new_value": "constant",
        })
        self.assertEqual(
            inactive_key_in_proposal(proposal, spec, status.current_constants), "imputer"
        )

    def test_an_editable_symbol_missing_from_the_status_is_still_named(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        status = self._status_with_full_constants(spec, {
            "model_type": "logistic_regression", "C": "1.0", "max_iter": "1000",
        })
        proposal = self._langgraph_proposal(spec, status, {
            "summary": "s", "rationale": "r",
            "objective": "Change imputer in config.yaml.",
            "param": "imputer", "old_value": "most_frequent", "new_value": "constant",
        })
        self.assertEqual(proposal.extra.get("named_param"), "imputer")
        self.assertEqual(
            inactive_key_in_proposal(proposal, spec, status.current_constants), "imputer"
        )

    def test_a_manager_without_revision_attempt_is_not_passed_it(self) -> None:
        # Review F2: a manager accepting revision_feedback but not
        # revision_attempt must not crash the revision loop.
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)

        class _Manager:
            deterministic = False

            def __init__(self):
                self.calls = 0

            def propose_next_trial(self, status, memory_context, node_spec, *, revision_feedback=None):
                self.calls += 1
                if self.calls == 1:
                    return _proposal(spec, "Change imputer from most_frequent to constant in config.yaml.")
                return _proposal(spec, "Change C from 1.0 to 2.0 in config.yaml.", summary="good")

        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_openml(Path(tmp), "openml_ionosphere")
            worker = _CountingWorker(node_root)
            manager = _Manager()
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="d25d", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=manager,
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(manager.calls, 2)  # initial + one revision, no TypeError
        self.assertEqual(worker.run_calls, 1)
        self.assertEqual(records[0].decision.value, "kept")


class TestD25DefaultPathUnchanged(unittest.TestCase):
    def test_undeclared_prompt_is_byte_identical(self) -> None:
        from autoresearch.manager.base import ManagerStatus
        from autoresearch.manager.langgraph_manager import _prepare_context

        spec = load_registered_node("lr_synthetic", repo_root=ROOT)
        status = ManagerStatus(
            campaign_id="golden", budget_index=2, current_best_metric=0.75,
            metric_name=spec.metric_name, metric_direction=spec.metric_direction,
            current_constants={"LEARNING_RATE": "0.001", "REGULARIZATION": "0.0001",
                               "N_EPOCHS": "80", "BATCH_SIZE": "32"},
            effective_config={},
        )
        memory = build_memory_context([], "none", spec, 2)
        state = {"status": status, "memory_context": memory, "node_spec": spec,
                 "context_text": "", "raw_proposal": "", "proposal": None, "revision_feedback": None}
        prompt = _prepare_context(state)["context_text"]
        self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), UNDECLARED_PROMPT_SHA256)

    def test_undeclared_node_spec_dict_has_no_active_keys_field(self) -> None:
        registered = ("resnet_trigger", "lr_synthetic", "mlp_synthetic", "digits_mlp", "tabular_mlp",
                      "mlagentbench_vectorization", "autoresearch_macos", "autoresearch_linux")
        for name in registered:
            spec = load_registered_node(name, repo_root=ROOT)
            self.assertNotIn("active_keys", spec.to_dict(), name)

    def test_declared_node_spec_dict_carries_the_declaration(self) -> None:
        spec = load_registered_node("openml_ionosphere", repo_root=ROOT)
        self.assertIn("active_keys", spec.to_dict())

    def test_every_config_node_still_loads_and_resolves(self) -> None:
        for path in sorted((ROOT / "configs" / "nodes").glob("*.yaml")):
            spec = load_registered_node(path.stem, repo_root=ROOT)
            declaration = spec.active_keys
            if spec.name.startswith("openml_"):
                self.assertIsNotNone(declaration, spec.name)
                self.assertIn("logistic_regression", declaration["by_model_type"], spec.name)
            else:
                self.assertIsNone(declaration, spec.name)


if __name__ == "__main__":
    unittest.main()
