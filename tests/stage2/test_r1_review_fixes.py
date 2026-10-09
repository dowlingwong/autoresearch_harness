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


if __name__ == "__main__":
    unittest.main()
