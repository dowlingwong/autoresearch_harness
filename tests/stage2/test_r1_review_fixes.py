"""R1 batch-1 review fixes (F1-F6).

Each test class targets one review finding and was written to fail against
``r1`` at 201e025 and pass after the corresponding "R1 review fix Fn" commit.
unittest style follows the repository convention.
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
from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.controller import ControlDecision, ControlView, compute_mask
from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.summarizer import build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]

# sha256 of the LangGraphManager prompt for ``_fixed_status`` with no memory,
# captured from r1 at 201e025 (before F1).  The default prompt must not change.
DEFAULT_PROMPT_SHA256 = "3e404e253ebc34a360f3ca851b604a2b17a12ea7c14cb0b3b55f8853be52e9ac"


class ScriptedController:
    version = "scripted/v1"
    deterministic = False

    def __init__(self, script: dict[tuple[int, str], str]) -> None:
        self.script = dict(script)

    def decide(self, view: ControlView, mask: dict[str, bool]) -> ControlDecision:
        action = self.script.get((view.budget_index, view.epoch), "continue")
        return ControlDecision(
            action=action, controller_version=self.version, input_hash=view.input_hash()
        )


class StaticBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, proposal: ManagerProposal) -> None:
        self.proposal = proposal
        self.calls = 0

    def propose_next_trial(self, status, memory_context, node_spec):
        self.calls += 1
        return self.proposal


class RecordingLLM:
    """Stub chat model that records every prompt it is given."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=self.responses.pop(0))


class ScriptedWorker:
    """Writes chosen file content and returns chosen metrics; optional seeded runs."""

    mode = "scripted_worker"

    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        self.node_root = Path(node_root)
        self._trials = list(trials)
        self.run_calls = 0

    def run_trial(self, proposal, node_spec, budget_index):
        spec = self._trials.pop(0)
        self.run_calls += 1
        content = spec.get("content")
        if content is not None:
            (self.node_root / "train.py").write_text(content, encoding="utf-8")
        metric = spec.get("metric")
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=("train.py",),
            success=spec.get("success", True),
            parsed_metrics={} if metric is None else {node_spec.metric_name: metric},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="before",
            git_commit_after="after",
        )


class SeededScriptedWorker(ScriptedWorker):
    def __init__(self, node_root: Path, trials: list[dict]) -> None:
        super().__init__(node_root, trials)
        self.seed_calls: list[tuple[object, object]] = []

    def run_trial_at_seed(self, proposal, node_spec, budget_index, *, train_seed, split_seed=None):
        self.seed_calls.append((train_seed, split_seed))
        text = (self.node_root / "train.py").read_text(encoding="utf-8")
        base = 0.95 if "CANDIDATE" in text else 0.90
        return WorkerResult(
            worker_mode=self.mode,
            changed_files=(),
            success=True,
            parsed_metrics={node_spec.metric_name: base + (int(train_seed) % 5) * 0.0001},
            raw_log_ref="fake.log",
            patch_ref="fake.diff",
            git_commit_before="before",
            git_commit_after="after",
        )


def _spec():
    return load_registered_node("lr_synthetic", repo_root=ROOT)


def _proposal(node_spec) -> ManagerProposal:
    return ManagerProposal(
        manager_mode="static_backend",
        proposal_summary="static-edit",
        proposal_rationale="test",
        target_files=node_spec.editable_paths,
        objective=f"Change {node_spec.editable_symbols[0]} from 1 to 2.",
        extra={},
    )


def _copy_node(tmp: Path) -> Path:
    node_root = tmp / "node"
    shutil.copytree(ROOT / "nodes" / "lr_synthetic", node_root)
    return node_root


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _fixed_status(spec) -> ManagerStatus:
    return ManagerStatus(
        campaign_id="c",
        budget_index=2,
        current_best_metric=0.9,
        metric_name=spec.metric_name,
        metric_direction=spec.metric_direction,
        current_constants={"LEARNING_RATE": "5e-4", "BATCH_SIZE": "4"},
    )


def _view(epoch: str, provisional=None) -> ControlView:
    return ControlView(
        epoch=epoch, budget_index=3, ledger_prefix=(), current_best=None,
        incumbent_stack_depth=0, proposal=None, provisional=provisional, budget={},
        node=_spec(),
    )


def _json(summary: str) -> str:
    return json.dumps({"summary": summary, "rationale": "r", "objective": "o"})


# ---------------------------------------------------------------------------
# F1 — revise passes its feedback to the manager
# ---------------------------------------------------------------------------


class TestF1ReviseFeedbackReachesManager(unittest.TestCase):
    def test_default_prompt_is_unchanged(self) -> None:
        spec = _spec()
        llm = RecordingLLM([_json("a")])
        LangGraphManager(llm=llm).propose_next_trial(
            _fixed_status(spec), build_memory_context([], "none", spec, 2), spec
        )
        self.assertEqual(
            hashlib.sha256(llm.prompts[0].encode("utf-8")).hexdigest(), DEFAULT_PROMPT_SHA256
        )

    def test_revision_feedback_enters_prompt_and_extra(self) -> None:
        spec = _spec()
        llm = RecordingLLM([_json("a"), _json("b")])
        manager = LangGraphManager(llm=llm)
        status = _fixed_status(spec)
        memory = build_memory_context([], "none", spec, 2)
        first = manager.propose_next_trial(status, memory, spec)
        second = manager.propose_next_trial(
            status, memory, spec, revision_feedback="step too large: propose a smaller step"
        )
        self.assertNotIn("step too large", llm.prompts[0])
        self.assertIn("step too large: propose a smaller step", llm.prompts[1])
        self.assertNotIn("revision_feedback", first.extra)
        self.assertEqual(
            second.extra["revision_feedback"], "step too large: propose a smaller step"
        )

    def test_campaign_revise_sends_feedback_on_second_call_only(self) -> None:
        spec = _spec()
        llm = RecordingLLM([_json("first"), _json("second")])
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [{"metric": 0.9, "content": "K1\n"}])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="revise", budget=1, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=LangGraphManager(llm=llm),
                controller=ScriptedController({(1, "E1"): "revise"}),
            )
            record = TrialAppendStore(records_path).read_all()[0]
        feedback = record.extra["control"]["revision"]["feedback"]
        self.assertEqual(len(llm.prompts), 2)
        self.assertNotIn(feedback, llm.prompts[0])
        self.assertIn(feedback, llm.prompts[1])
        # Apart from the feedback block the two prompts are identical.
        self.assertTrue(llm.prompts[1].startswith(llm.prompts[0].split("Propose exactly one")[0]))

    def test_revise_masked_for_manager_without_feedback_parameter(self) -> None:
        from autoresearch.control_plane.campaign import _accepts_revision_feedback

        class ResampleOnly:
            deterministic = False

            def propose_next_trial(self, status, memory_context, node_spec):
                raise AssertionError("not called")

        self.assertFalse(_accepts_revision_feedback(ResampleOnly()))
        self.assertTrue(_accepts_revision_feedback(LangGraphManager(llm=object())))
        view = _view("E1")
        mask = compute_mask(
            view, proposal_passed_precondition=True, manager_deterministic=False,
            revision_feedback_supported=False,
        )
        self.assertFalse(mask["revise"])


# ---------------------------------------------------------------------------
# F2 — verify needs real seed injection
# ---------------------------------------------------------------------------


class TestF2VerifyRequiresSeedInjection(unittest.TestCase):
    def _e2_mask(self, worker_cls) -> dict:
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = worker_cls(_copy_node(Path(tmp)), [{"metric": 0.9, "content": "K1\n"}])
            events_path = Path(tmp) / "events.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="seedmask", budget=1, manager_mode="baseline_manager",
                memory_mode="none", records_path=Path(tmp) / "trials.jsonl", worker=worker,
                proposal_backend=StaticBackend(_proposal(spec)),
                event_store=CampaignEventStore(events_path),
                controller=ScriptedController({}),
            )
            events = _events(events_path)
        (e2,) = [
            event for event in events
            if event["event_type"] == "control_decision" and event["payload"]["epoch"] == "E2"
        ]
        return e2["payload"]["mask"]

    def test_verify_masked_out_without_seed_support(self) -> None:
        self.assertFalse(self._e2_mask(ScriptedWorker)["verify"])

    def test_verify_feasible_with_seed_support(self) -> None:
        self.assertTrue(self._e2_mask(SeededScriptedWorker)["verify"])

    def test_seeded_run_raises_instead_of_running_unseeded(self) -> None:
        from autoresearch.control_plane import campaign

        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [{"metric": 0.9}])
            with self.assertRaises(campaign.SeedInjectionUnsupportedError):
                campaign._run_worker_at_seed(worker, _proposal(spec), spec, 900001, 1000)
            self.assertEqual(worker.run_calls, 0)


# ---------------------------------------------------------------------------
# F3 — demote only pops an unverified top-of-stack keep
# ---------------------------------------------------------------------------


class TestF3DemoteTopOfStackOnly(unittest.TestCase):
    def test_mask_false_when_top_keep_is_verified(self) -> None:
        # Stack after trial 2: [unverified keep K1, verified keep CANDIDATE2].
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = SeededScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "K1\n"},
                {"metric": 0.95, "content": "CANDIDATE2\n"},
                {"metric": 0.10, "content": "K3\n"},
            ])
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f3mask", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(_proposal(spec)),
                event_store=CampaignEventStore(events_path),
                controller=ScriptedController({(2, "E2"): "verify"}),
            )
            records = TrialAppendStore(records_path).read_all()
            events = _events(events_path)
        self.assertEqual(records[1].extra["control"]["verify"]["response"], "keep")
        e0_trial3 = [
            event for event in events
            if event["event_type"] == "control_decision"
            and event["payload"]["epoch"] == "E0"
            and event["trial_id"].endswith("trial-003")
        ]
        self.assertEqual(len(e0_trial3), 1)
        self.assertFalse(e0_trial3[0]["payload"]["mask"]["demote"])

    def test_perform_demote_refuses_verified_top_and_keeps_stack(self) -> None:
        from autoresearch.control_plane import campaign

        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_node(Path(tmp))
            worker = ScriptedWorker(node_root, [])
            (node_root / "train.py").write_text("V2\n", encoding="utf-8")
            stack = [
                campaign._IncumbentEntry(
                    snapshot=None, current_best_before=None, metric=0.90, verified=False
                ),
                campaign._IncumbentEntry(
                    snapshot=None, current_best_before=0.90, metric=0.95, verified=True
                ),
            ]
            with self.assertRaises(ValueError):
                campaign._perform_demote(
                    incumbent_stack=stack, worker=worker, node_spec=spec, current_best=0.95,
                    event_store=None, campaign_id="c", trial_id="t",
                )
            self.assertEqual(len(stack), 2)
            self.assertEqual((node_root / "train.py").read_text(encoding="utf-8"), "V2\n")

    def test_demote_reverts_only_unverified_top_above_verified_keep(self) -> None:
        # Stack after trial 2: [verified keep CANDIDATE1, unverified keep K2].
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            node_root = _copy_node(Path(tmp))
            worker = SeededScriptedWorker(node_root, [
                {"metric": 0.80, "content": "CANDIDATE1\n"},
                {"metric": 0.85, "content": "K2\n"},
                {"metric": 0.10, "content": "K3\n"},
            ])
            events_path = Path(tmp) / "events.jsonl"
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f3pop", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(_proposal(spec)),
                event_store=CampaignEventStore(events_path),
                controller=ScriptedController({(1, "E2"): "verify", (3, "E0"): "demote"}),
            )
            records = TrialAppendStore(records_path).read_all()
            events = _events(events_path)
            final_content = (node_root / "train.py").read_text(encoding="utf-8")
        self.assertEqual(records[0].extra["control"]["verify"]["response"], "keep")
        self.assertEqual(
            [event["event_type"] for event in events].count("incumbent_demoted"), 1
        )
        # Only K2 was reverted: the verified CANDIDATE1 incumbent and its metric remain.
        self.assertEqual(records[2].current_best_before, 0.80)
        self.assertEqual(final_content, "CANDIDATE1\n")


# ---------------------------------------------------------------------------
# F4 — training and split seeds are independent
# ---------------------------------------------------------------------------


def _load_node_module(path: Path, env: dict[str, str]):
    """Import a node script fresh with exactly *env*'s AR_* seed variables set."""
    import importlib.util
    import os
    from unittest import mock

    clean = {key: value for key, value in os.environ.items() if not key.startswith("AR_")}
    clean.update(env)
    with mock.patch.dict(os.environ, clean, clear=True):
        module_spec = importlib.util.spec_from_file_location(f"_node_{path.parent.name}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    return module


def _lr_synthetic_seeds(env: dict[str, str]) -> tuple[int, int, str]:
    """Run lr_synthetic's main() with training stubbed; return (dataset seed, train seed, data hash)."""
    import contextlib
    import io

    module = _load_node_module(ROOT / "nodes" / "lr_synthetic" / "train.py", env)
    seen: dict[str, object] = {}
    original_make = module._make_dataset

    def recording_make(*args, **kwargs):
        seen["dataset_seed"] = kwargs["seed"]
        X, y = original_make(*args, **kwargs)
        seen["hash"] = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
        return X, y

    def stub_train(*args, **kwargs):
        seen["train_seed"] = kwargs["seed"]
        return 0.5, 0.1

    module._make_dataset = recording_make
    module.train = stub_train
    with contextlib.redirect_stdout(io.StringIO()):
        module.main()
    return seen["dataset_seed"], seen["train_seed"], seen["hash"]


class TestF4IndependentSeeds(unittest.TestCase):
    def test_lr_synthetic_train_seed_does_not_move_dataset(self) -> None:
        default_split, default_train, default_hash = _lr_synthetic_seeds({})
        self.assertEqual((default_split, default_train), (42, 42))
        split, train, data_hash = _lr_synthetic_seeds({"AR_TRAIN_SEED": "7"})
        self.assertEqual((split, train), (42, 7))
        self.assertEqual(data_hash, default_hash)

    def test_lr_synthetic_split_seed_moves_only_dataset(self) -> None:
        _, _, default_hash = _lr_synthetic_seeds({})
        split, train, data_hash = _lr_synthetic_seeds({"AR_SPLIT_SEED": "9"})
        self.assertEqual((split, train), (9, 42))
        self.assertNotEqual(data_hash, default_hash)

    def test_openml_split_seed_defaults_independently(self) -> None:
        for node in ("openml_bank_marketing", "openml_credit_g"):
            path = ROOT / "nodes" / node / "benchmark.py"
            with self.subTest(node=node):
                default = _load_node_module(path, {})
                self.assertEqual((default.TRAIN_SEED, default.SPLIT_SEED, default.SEED), (42, 42, 42))
                train_only = _load_node_module(path, {"AR_TRAIN_SEED": "7"})
                self.assertEqual((train_only.TRAIN_SEED, train_only.SPLIT_SEED), (7, 42))
                split_only = _load_node_module(path, {"AR_SPLIT_SEED": "9"})
                self.assertEqual((split_only.TRAIN_SEED, split_only.SPLIT_SEED), (42, 9))

    def _verify_campaign(self, **kwargs):
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = SeededScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.80, "content": "CANDIDATE1\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f4verify", budget=1, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(_proposal(spec)),
                controller=ScriptedController({(1, "E2"): "verify"}),
                **kwargs,
            )
            record = TrialAppendStore(records_path).read_all()[0]
        return record.extra["control"]["verify"], worker.seed_calls

    def test_verify_default_policy_is_fixed_and_varies_only_training_seed(self) -> None:
        verify, calls = self._verify_campaign()
        self.assertEqual(verify["split_policy"], "fixed")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(split is None for _, split in calls))

    def test_verify_vary_policy_moves_split_with_training_seed(self) -> None:
        verify, calls = self._verify_campaign(verify_split_policy="vary")
        self.assertEqual(verify["split_policy"], "vary")
        self.assertTrue(all(split == train for train, split in calls))

    def test_local_worker_keeps_configured_split_seed_when_none(self) -> None:
        from autoresearch.worker.local_worker import LocalWorker

        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = LocalWorker(node_root=_copy_node(Path(tmp)), train_seed=3, split_seed=5)
            seen: list[tuple[object, object]] = []
            worker.run_trial = lambda *args: seen.append((worker._train_seed, worker._split_seed))
            worker.run_trial_at_seed(_proposal(spec), spec, 1, train_seed=11, split_seed=None)
            worker.run_trial_at_seed(_proposal(spec), spec, 2, train_seed=12, split_seed=13)
        self.assertEqual(seen, [(11, 5), (12, 13)])
        self.assertEqual((worker._train_seed, worker._split_seed), (3, 5))


# ---------------------------------------------------------------------------
# F5 — deterministic managers consume no LLM budget
# ---------------------------------------------------------------------------


class TestF5DeterministicManagersUseNoLLMCalls(unittest.TestCase):
    def test_real_campaign_with_llm_cap_runs_all_trials(self) -> None:
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "K1\n"},
                {"metric": 0.91, "content": "K2\n"},
                {"metric": 0.92, "content": "K3\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f5real", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=StaticBackend(_proposal(spec)),
                compute_budget=ComputeBudget(max_llm_calls=1),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(len(records), 3)

    def test_dry_campaign_with_llm_cap_runs_all_trials(self) -> None:
        from autoresearch.control_plane.campaign import run_dry_campaign

        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "trials.jsonl"
            run_dry_campaign(
                node_spec=spec, campaign_id="f5dry", budget=3, manager_mode="baseline_manager",
                memory_mode="none", records_path=records_path,
                compute_budget=ComputeBudget(max_llm_calls=1),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(len(records), 3)

    def test_llm_manager_still_counts_calls(self) -> None:
        spec = _spec()
        llm = RecordingLLM([_json(f"s{i}") for i in range(5)])
        with tempfile.TemporaryDirectory() as tmp:
            worker = ScriptedWorker(_copy_node(Path(tmp)), [
                {"metric": 0.90, "content": "K1\n"},
                {"metric": 0.91, "content": "K2\n"},
                {"metric": 0.92, "content": "K3\n"},
            ])
            records_path = Path(tmp) / "trials.jsonl"
            run_real_campaign(
                node_spec=spec, campaign_id="f5llm", budget=3, manager_mode="langgraph_manager",
                memory_mode="none", records_path=records_path, worker=worker,
                proposal_backend=LangGraphManager(llm=llm),
                compute_budget=ComputeBudget(max_llm_calls=1),
            )
            records = TrialAppendStore(records_path).read_all()
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
