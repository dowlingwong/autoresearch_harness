"""Track S (29 Oct 2026): fork-arm overrides of ``resume_campaign`` and the skip displays.

Covers:

* the arm-override hook (``ArmOverride``): ``skip`` / ``execute`` / ``ghost`` /
  ``ctrl_only`` at E1 and forced ``keep`` / ``discard`` at E2;
* ``run_real_campaign(controller_ledger_view=...)`` (the ``ctrl_only`` view);
* the opt-in skip displays ``hidden/v1`` and ``reason/v1`` (``one_line/v1``
  and the historical display are unchanged);
* default paths: no arm and no view give the historical records.

The manager is the real ``LangGraphManager`` with an injected stub LLM that
records every prompt and D26 seed, so prompt identity is checked on the real
prompt path.  ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autoresearch.control_plane.action_versions import KEEP_RULE_BAYES, ActionVersions, KeepPrior
from autoresearch.control_plane.budget import ComputeBudget
from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.controller import ControlDecision
from autoresearch.control_plane.snapshot import (
    SHADOW_FILE,
    ArmOverride,
    capture_snapshot,
    resume_campaign,
)
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.memory.summarizer import (
    SKIP_DISPLAY_HIDDEN,
    SKIP_DISPLAY_ONE_LINE,
    SKIP_DISPLAY_REASON,
    SKIP_DISPLAY_TEXT,
    SKIP_REASON_TEXTS,
    SKIP_REASON_UNKNOWN_TEXT,
    MemoryMode,
    build_memory_context,
)
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]
NODE = "digits_mlp"
_ASSIGN = re.compile(r"^(ALPHA\s*=\s*)(.+?)\s*$", re.MULTILINE)


def _alpha_metric(text: str) -> float:
    """A deterministic metric of the configuration: peaked in log10(ALPHA) around -3."""
    alpha = float(_ASSIGN.search(text).group(2))
    return round(0.9 - 0.01 * (math.log10(alpha) + 3.0) ** 2, 9)


class EditWorker:
    """Applies the structured edit to ``train.py`` and returns ``_alpha_metric`` (no subprocess)."""

    mode = "track_s_test_worker"

    def __init__(self, node_root: Path) -> None:
        self.node_root, self.runs = Path(node_root), 0

    def run_trial(self, proposal, node_spec, budget_index):
        self.runs += 1
        path = self.node_root / "train.py"
        text = path.read_text()
        edit = (proposal.extra or {}).get("structured_edit")
        if edit:
            text = _ASSIGN.sub(lambda m: m.group(1) + str(edit["new"]), text) if edit["symbol"] == "ALPHA" else text
            path.write_text(text)
        return WorkerResult(worker_mode=self.mode, changed_files=("train.py",), success=True,
                            parsed_metrics={"val_score": _alpha_metric(text)}, raw_log_ref="", patch_ref="p",
                            git_commit_before="", git_commit_after="")


class PromptLLM:
    """Stub chat model: records (prompt, seed); proposes ALPHA x factor(seed, prompt), factor in {1/3, 3}."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def invoke(self, messages, **kwargs):
        prompt = messages[0].content
        seed = kwargs.get("seed")
        self.calls.append((prompt, seed))
        old = re.search(r"^\s*ALPHA = (\S+)\s*$", prompt, re.MULTILINE).group(1)
        h = int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest()[:8], 16)
        new = repr(float(old) * (3.0 if h % 2 else 1.0 / 3.0))
        body = {"summary": f"alpha-{new}", "rationale": "r", "objective": f"Change ALPHA from {old} to {new}.",
                "param": "ALPHA", "old_value": old, "new_value": new}
        return SimpleNamespace(content=json.dumps(body), usage_metadata={}, response_metadata={})


class ViewRecorder:
    """Controller: records every view's ledger digests; skips at E1 the budget indices in *skips*."""

    version = "view_recorder/v1"

    def __init__(self, skips=()) -> None:
        self.skips, self.views = set(skips), []

    def decide(self, view, mask):
        self.views.append((view.epoch, view.budget_index,
                           [(getattr(r, "trial_id"), r.decision.value, dict(r.parsed_metrics), r.delta_vs_best)
                            for r in view.ledger_prefix]))
        if view.epoch == "E1" and view.budget_index in self.skips and mask.get("skip"):
            return ControlDecision("skip", {}, self.version, view.input_hash(), detail={"skip_reason": "prior_no_gain"})
        return ControlDecision("continue", {}, self.version, view.input_hash())


class Capture:
    def __init__(self, epoch: str, index: int) -> None:
        self.epoch, self.index, self.snapshot = epoch, index, None

    def __call__(self, context) -> None:
        if context.epoch == self.epoch and context.budget_index == self.index and self.snapshot is None:
            self.snapshot = capture_snapshot(context)


def _versions():
    return ActionVersions.new_campaign(keep_rule=KEEP_RULE_BAYES)


PRIOR = KeepPrior(mu=0.0, tau=0.01, sigma=0.005)


def _root(tmp: Path, *, epoch: str = "E1", index: int = 3, budget: int = 6, skip_display=SKIP_DISPLAY_ONE_LINE):
    spec = load_registered_node(NODE, repo_root=ROOT)
    node = tmp / "node"
    shutil.copytree(ROOT / "nodes" / NODE, node, ignore=shutil.ignore_patterns("__pycache__"))
    capture = Capture(epoch, index)
    run_real_campaign(
        node_spec=spec, campaign_id="tracks", budget=budget, manager_mode="langgraph_manager",
        memory_mode="append_only_summary", records_path=tmp / "root.jsonl", worker=EditWorker(node),
        proposal_backend=LangGraphManager(llm=PromptLLM()), controller=ViewRecorder(),
        compute_budget=ComputeBudget(max_worker_runs=20, max_llm_calls=40), action_versions=_versions(),
        keep_prior=PRIOR, skip_display=skip_display, snapshot_hook=capture,
        event_store=CampaignEventStore(tmp / "root_events.jsonl"),
    )
    assert capture.snapshot is not None
    return capture.snapshot


def _leaf(snapshot, into: Path, arm, *, controller=None, horizon: int = 3):
    llm = PromptLLM()
    worker = EditWorker(into)
    controller = controller or ViewRecorder()
    result = resume_campaign(snapshot, into, arm=arm, horizon=horizon, worker=worker,
                             manager=LangGraphManager(llm=llm, repetition=101), policy=controller,
                             keep_prior=PRIOR)
    rows = TrialAppendStore(result.restored.records_path).read_all()
    return result, rows, llm, controller, worker


def _drop_timing(value):
    if isinstance(value, dict):
        return {k: _drop_timing(v) for k, v in value.items() if k not in ("latency_ms", "seconds")}
    if isinstance(value, list):
        return [_drop_timing(v) for v in value]
    return value


def _strip(record) -> dict:
    """A ledger row without wall-clock fields and without the arm label."""
    payload = record.to_dict()
    for key in ("timestamp_start", "timestamp_end", "wall_clock_seconds", "provenance"):
        payload.pop(key, None)
    payload.get("extra", {}).get("control", {}).pop("arm", None)
    return _drop_timing(payload)


class TestE1Arms(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.snapshot = _root(self.tmp / "root")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_skip_arm_records_a_skip_with_reason_and_spends_no_run(self) -> None:
        result, rows, _, _, worker = _leaf(self.snapshot, self.tmp / "skip", ArmOverride("skip"), horizon=0)
        first = result.first_record
        self.assertEqual(first.decision, TrialDecision.SKIPPED)
        self.assertEqual(first.extra["control"]["arm"], "skip")
        self.assertEqual(first.extra["control"]["skip_reason"], "prior_no_gain")
        self.assertEqual(worker.runs, 0)
        self.assertEqual(result.restored.compute_budget.worker_runs, self.snapshot.compute_budget_state["worker_runs"])
        self.assertIsNone(result.shadow_record)

    def test_execute_arm_equals_the_historical_continue(self) -> None:
        a, _, _, _, _ = _leaf(self.snapshot, self.tmp / "exec", "execute", horizon=0)
        b = resume_campaign(self.snapshot, self.tmp / "cont", first_action="continue", horizon=0,
                            worker=EditWorker(self.tmp / "cont"), manager=LangGraphManager(llm=PromptLLM()),
                            keep_prior=PRIOR)
        self.assertEqual(a.first_record.parsed_metrics, b.first_record.parsed_metrics)
        self.assertEqual(a.first_record.decision, b.first_record.decision)
        self.assertEqual(a.first_record.extra["control"]["arm"], "execute")
        self.assertNotIn("arm", b.first_record.extra["control"])

    def test_ghost_ledger_record_is_the_skip_record_and_leaks_nothing(self) -> None:
        skip, skip_rows, _, _, _ = _leaf(self.snapshot, self.tmp / "skip", "skip", horizon=0)
        ghost, ghost_rows, _, _, worker = _leaf(self.snapshot, self.tmp / "ghost", "ghost", horizon=0)
        self.assertEqual(worker.runs, 1)
        self.assertEqual(ghost.restored.compute_budget.worker_runs,
                         skip.restored.compute_budget.worker_runs + 1)
        g = ghost.first_record
        self.assertEqual(g.decision, TrialDecision.SKIPPED)
        self.assertEqual(g.parsed_metrics, {})
        self.assertIsNone(g.delta_vs_best)
        self.assertEqual(g.extra["control"]["arm"], "ghost")
        self.assertEqual(_strip(g), _strip(skip.first_record))
        # Nothing of the result anywhere in the ledger row.
        text = json.dumps(ghost_rows[-1].to_dict())
        metric = ghost.shadow_record.parsed_metrics["val_score"]
        self.assertNotIn(repr(metric), text)
        self.assertNotIn("keep_rule", text)
        # The shadow holds the executed, non-kept result outside the ledger.
        shadow = ghost.shadow_record
        self.assertIn(shadow.decision, (TrialDecision.DISCARDED, TrialDecision.FAILED_INVALID))
        self.assertEqual(shadow.extra["arm_override"]["arm"], "ghost")
        on_disk = TrialAppendStore(ghost.restored.into_dir / SHADOW_FILE).read_all()
        self.assertEqual([r.trial_id for r in on_disk], [shadow.trial_id])
        # The workspace is back at the incumbent (no keep).
        self.assertEqual((ghost.restored.into_dir / "train.py").read_bytes(), self.snapshot.incumbent_files["train.py"])

    def test_ghost_and_ctrl_only_prompts_and_seeds_equal_skip_until_controllers_differ(self) -> None:
        _, _, skip_llm, skip_ctl, _ = _leaf(self.snapshot, self.tmp / "skip", "skip")
        _, _, ghost_llm, ghost_ctl, _ = _leaf(self.snapshot, self.tmp / "ghost", "ghost")
        _, _, ctrl_llm, ctrl_ctl, _ = _leaf(self.snapshot, self.tmp / "ctrl", "ctrl_only")
        self.assertEqual(len(skip_llm.calls), 3)
        # The recording controller never diverges, so all prompts and seeds agree byte for byte.
        self.assertEqual(skip_llm.calls, ghost_llm.calls)
        self.assertEqual(skip_llm.calls, ctrl_llm.calls)
        # The controller of skip and ghost sees identical histories; ctrl_only sees the result.
        self.assertEqual(skip_ctl.views, ghost_ctl.views)
        captured = f"tracks-trial-{self.snapshot.budget_index:03d}"
        for epoch, index, prefix in ctrl_ctl.views:
            row = [p for p in prefix if p[0] == captured][0]
            self.assertIn(row[1], ("discarded", "failed_invalid"))
            self.assertIn("val_score", row[2])
        for epoch, index, prefix in ghost_ctl.views:
            row = [p for p in prefix if p[0] == captured][0]
            self.assertEqual((row[1], row[2], row[3]), ("skipped", {}, None))

    def test_novelty_set_gets_neither_ghost_nor_ctrl_only_config(self) -> None:
        for arm, expect_seen in (("execute", True), ("ghost", False), ("ctrl_only", False)):
            into = self.tmp / f"nov_{arm}"
            llm = PromptLLM()
            # Re-propose the captured configuration at the next trial: is it flagged as seen?
            captured_new = self.snapshot.proposal.extra["structured_edit"]["new"]

            class Repeat:
                mode, deterministic, n = "repeat", True, 0

                def propose_next_trial(self, status, memory_context, node_spec):
                    from autoresearch.manager.base import ManagerProposal
                    old = status.current_constants["ALPHA"]
                    return ManagerProposal(manager_mode="repeat", proposal_summary="repeat", proposal_rationale="",
                                           target_files=("train.py",), objective="Change ALPHA.",
                                           extra={"structured_edit": {"type": "python_constant", "path": "train.py",
                                                                      "symbol": "ALPHA", "old": old,
                                                                      "new": captured_new}})

            result = resume_campaign(self.snapshot, into, arm=ArmOverride(arm), horizon=1, worker=EditWorker(into),
                                     manager=Repeat(), keep_prior=PRIOR)
            if result.first_record.decision == TrialDecision.KEPT:
                continue  # the incumbent moved: the repeat is a no-op, not a novelty test
            events = [json.loads(l) for l in result.restored.events_path.read_text().splitlines()]
            novelty = [e for e in events if e["event_type"] == "edit_novelty"]
            self.assertEqual(len(novelty), 1, arm)
            self.assertEqual(novelty[0]["payload"]["config_seen_before"], expect_seen, arm)
            del llm

    def test_arm_epoch_and_first_action_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            resume_campaign(self.snapshot, self.tmp / "bad1", arm="keep", worker=EditWorker(self.tmp / "bad1"),
                            manager=LangGraphManager(llm=PromptLLM()), keep_prior=PRIOR)
        with self.assertRaises(ValueError):
            resume_campaign(self.snapshot, self.tmp / "bad2", arm="ghost", first_action="skip",
                            worker=EditWorker(self.tmp / "bad2"), manager=LangGraphManager(llm=PromptLLM()),
                            keep_prior=PRIOR)
        with self.assertRaises(ValueError):
            ArmOverride("verify")
        with self.assertRaises(ValueError):
            resume_campaign(self.snapshot, self.tmp / "bad3", worker=EditWorker(self.tmp / "bad3"),
                            manager=LangGraphManager(llm=PromptLLM()), keep_prior=PRIOR)


class TestE2ForcedKeep(unittest.TestCase):
    def test_forced_keep_and_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            snapshot = _root(tmp / "root", epoch="E2", index=3)
            original = snapshot.provisional.decision
            keep, _, _, _, _ = _leaf(snapshot, tmp / "keep", "keep", horizon=0)
            discard, _, _, _, _ = _leaf(snapshot, tmp / "discard", "discard", horizon=0)
            self.assertEqual(keep.first_record.decision, TrialDecision.KEPT)
            self.assertIsNone(keep.first_record.failure_category)
            self.assertEqual(discard.first_record.decision, TrialDecision.DISCARDED)
            for res, arm in ((keep, "keep"), (discard, "discard")):
                self.assertEqual(res.first_record.extra["arm_override"],
                                 {"arm": arm, "original_decision": original.value, "forced": True})
                self.assertEqual(res.first_record.extra["control"]["arm"], arm)
                self.assertEqual(res.first_record.parsed_metrics, snapshot.provisional.parsed_metrics)
            # Commit semantics follow the forced decision.
            self.assertEqual(len(keep.restored.incumbent_stack), len(snapshot.incumbent_stack) + 1)
            self.assertEqual(len(discard.restored.incumbent_stack), len(snapshot.incumbent_stack))
            self.assertEqual((discard.restored.into_dir / "train.py").read_bytes(), snapshot.incumbent_files["train.py"])
            self.assertEqual((keep.restored.into_dir / "train.py").read_bytes(), snapshot.editable_files["train.py"])


class TestDefaultPathsUnchanged(unittest.TestCase):
    def test_identity_controller_view_gives_the_same_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = load_registered_node(NODE, repo_root=ROOT)
            out = []
            for name, kwargs in (("a", {}), ("b", {"controller_ledger_view": lambda prefix: prefix})):
                node = tmp / name
                shutil.copytree(ROOT / "nodes" / NODE, node, ignore=shutil.ignore_patterns("__pycache__"))
                ctl = ViewRecorder(skips={3})
                run_real_campaign(node_spec=spec, campaign_id="d", budget=5, manager_mode="langgraph_manager",
                                  memory_mode="append_only_summary", records_path=tmp / f"{name}.jsonl",
                                  worker=EditWorker(node), proposal_backend=LangGraphManager(llm=PromptLLM()),
                                  controller=ctl, **kwargs)
                out.append(([_strip(r) for r in TrialAppendStore(tmp / f"{name}.jsonl").read_all()], ctl.views))
            self.assertEqual(out[0], out[1])


def _rec(trial: int, decision: str, *, control=None, metric=None):
    from autoresearch.memory.schemas import TrialRecord
    from autoresearch.memory.provenance import build_trial_provenance
    tid = f"c-trial-{trial:03d}"
    valid = decision != "failed_invalid"
    return TrialRecord.from_mapping({
        "trial_id": tid, "campaign_id": "c", "node_id": NODE, "budget_index": trial,
        "timestamp_start": "2026-10-29T00:00:00Z", "timestamp_end": "2026-10-29T00:00:01Z",
        "manager_mode": "m", "worker_mode": "w", "memory_mode": "append_only_summary",
        "proposal_summary": f"edit {trial}", "proposal_rationale": "why", "targeted_files": ["train.py"],
        "patch_ref": "", "git_commit_before": "", "git_commit_after": "",
        "execution_status": "success" if metric is not None else "failed",
        "validity_status": "valid" if valid else "invalid",
        "failure_category": None if valid and decision != "discarded" else ("degraded_metric" if valid else "runtime_error"),
        "raw_log_ref": "", "parsed_metrics": {} if metric is None else {"val_score": metric},
        "current_best_before": 0.9, "delta_vs_best": None if metric is None else metric - 0.9,
        "decision": decision, "decision_rationale": "r", "wall_clock_seconds": 1.0,
        "cumulative_budget_consumed": trial, "provenance": build_trial_provenance(tid).to_dict(),
        "extra": {} if control is None else {"control": control},
    })


class TestSkipDisplays(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_registered_node(NODE, repo_root=ROOT)
        self.records = [
            _rec(1, "kept", metric=0.9),
            _rec(2, "skipped", control={"epoch": "E1", "action": "skip", "skip_reason": "rule_duplicate"}),
            _rec(3, "skipped", control={"epoch": "E1", "action": "skip", "skip_reason": "rule_noop"}),
            _rec(4, "skipped", control={"epoch": "E1", "action": "skip", "skip_reason": "prior_no_gain"}),
            _rec(5, "skipped", control={"epoch": "E1", "action": "skip"}),
            _rec(6, "skipped", control={"epoch": "E1", "action": "skip", "reason": "inactive_key", "key": "X"}),
            _rec(7, "discarded", metric=0.89),
        ]

    def _ctx(self, display, mode=MemoryMode.APPEND_ONLY_SUMMARY):
        return build_memory_context(self.records, mode, self.spec, 8, skip_display=display).context_text

    def test_historical_and_one_line_unchanged(self) -> None:
        base = self._ctx(None).splitlines()
        self.assertEqual(len(base), 7)
        self.assertTrue(all("decision=" in line for line in base))
        one = self._ctx(SKIP_DISPLAY_ONE_LINE).splitlines()
        for i in (1, 2, 3, 4):
            self.assertEqual(one[i], f"c-trial-{i + 1:03d}: summary=edit {i + 1}; {SKIP_DISPLAY_TEXT}")
        self.assertEqual(one[5], base[5])  # D25 inactive-key skip keeps the generic line
        self.assertEqual(one[0], base[0])
        self.assertEqual(one[6], base[6])

    def test_hidden_omits_control_plane_skips_without_renumbering(self) -> None:
        for mode in (MemoryMode.APPEND_ONLY_SUMMARY, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE):
            lines = self._ctx(SKIP_DISPLAY_HIDDEN, mode).splitlines()
            ids = [line.split(":")[0] for line in lines if line.startswith("c-trial-")]
            self.assertEqual(ids, ["c-trial-001", "c-trial-006", "c-trial-007"])
            self.assertNotIn("skipped by the control plane", "\n".join(lines))
        self.assertEqual(self._ctx(SKIP_DISPLAY_HIDDEN, MemoryMode.NONE), self._ctx(None, MemoryMode.NONE))

    def test_reason_labels_each_skip_correctly(self) -> None:
        lines = self._ctx(SKIP_DISPLAY_REASON).splitlines()
        self.assertIn(SKIP_REASON_TEXTS["rule_duplicate"], lines[1])
        self.assertIn("duplicate", lines[1])
        self.assertIn(SKIP_REASON_TEXTS["rule_noop"], lines[2])
        self.assertIn("no-op", lines[2])
        self.assertTrue(lines[3].endswith(SKIP_DISPLAY_TEXT))
        self.assertTrue(lines[4].endswith("; " + SKIP_REASON_UNKNOWN_TEXT))
        self.assertNotIn("predicted no gain", lines[1] + lines[2] + lines[4])
        for i in (1, 2, 3, 4):
            self.assertNotIn("metric=", lines[i])
            self.assertTrue(lines[i].startswith(f"c-trial-{i + 1:03d}: summary=edit {i + 1}; "))
        self.assertEqual(lines[5], self._ctx(None).splitlines()[5])

    def test_unknown_display_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._ctx("hidden/v2")


if __name__ == "__main__":
    unittest.main()


class TestSeedKey(unittest.TestCase):
    def _seeds(self, campaign_id, **kwargs):
        from autoresearch.manager.base import ManagerStatus
        from autoresearch.manager.langgraph_manager import manager_seed
        spec = load_registered_node(NODE, repo_root=ROOT)
        llm = PromptLLM()
        manager = LangGraphManager(llm=llm, repetition=3, **kwargs)
        status = ManagerStatus(campaign_id=campaign_id, budget_index=5, current_best_metric=0.9,
                               metric_name=spec.metric_name, metric_direction=spec.metric_direction,
                               current_constants={"ALPHA": "1e-4"})
        context = build_memory_context([], MemoryMode.NONE, spec, 5)
        manager.propose_next_trial(status, context, spec)
        return llm.calls[0][1], manager_seed

    def test_default_seed_is_keyed_by_the_campaign_id(self) -> None:
        seed, manager_seed = self._seeds("policy-a")
        self.assertEqual(seed, manager_seed("policy-a", 5, 3, 0))

    def test_seed_key_overrides_the_campaign_id(self) -> None:
        a, manager_seed = self._seeds("policy-a", seed_key="instance-7")
        b, _ = self._seeds("policy-b", seed_key="instance-7")
        self.assertEqual(a, b)
        self.assertEqual(a, manager_seed("instance-7", 5, 3, 0))
        self.assertNotEqual(a, manager_seed("policy-a", 5, 3, 0))
