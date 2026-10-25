"""R1 Track H (25 Oct 2026): control-plane hooks for the learned-prior controller.

Covers the three minimal hooks the paper repository's ``RuleController`` needs:

* ``ControlDecision.detail`` — written to the ``control_decision`` event only
  when non-empty; an E1 ``skip_reason`` is copied into the skipped record;
* ``ControlDecision.events`` — whitelisted controller-side events
  (``prior_unavailable``) logged before the decision event;
* the R1-SKIP-DISPLAY memory display ``one_line/v1`` (author decision, 25 Oct):
  a control-plane skip is one line, "skipped by the control plane: predicted no
  gain", with no metric; recorded in ``campaign_started`` and in snapshots.

Backward compatibility: a controller that sets neither field and a campaign
without ``skip_display`` produce exactly the historical events, records and
memory text.  ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.controller import CONTROLLER_EVENT_TYPES, ControlDecision, ControlView
from autoresearch.control_plane.snapshot import SnapshotRecorder, read_snapshot
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.memory.summarizer import (
    SKIP_DISPLAY_ONE_LINE,
    SKIP_DISPLAY_TEXT,
    MemoryMode,
    _summary_line,
    build_memory_context,
    is_control_plane_skip,
)
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class ScriptedController:
    version = "scripted/v1"

    def __init__(self, script, detail=None, events=()):
        self.script, self.detail, self.events = dict(script), dict(detail or {}), tuple(events)

    def decide(self, view: ControlView, mask):
        action = self.script.get((view.budget_index, view.epoch), "continue")
        if action == "continue":
            return ControlDecision(action="continue", controller_version=self.version, input_hash=view.input_hash())
        return ControlDecision(action=action, controller_version=self.version, input_hash=view.input_hash(),
                               detail=self.detail, events=self.events)


class RecordingBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, spec):
        self.spec, self.contexts, self.n = spec, [], 0

    def propose_next_trial(self, status, memory_context, node_spec):
        self.contexts.append(memory_context.context_text)
        self.n += 1
        return ManagerProposal(manager_mode="static_backend", proposal_summary=f"edit number {self.n}",
                               proposal_rationale="t", target_files=self.spec.editable_paths,
                               objective="Change X.", extra={})


class Worker:
    mode = "scripted_worker"

    def __init__(self, node_root, metrics):
        self.node_root = Path(node_root)
        self.metrics = list(metrics)

    def run_trial(self, proposal, node_spec, budget_index):
        metric = self.metrics.pop(0)
        return WorkerResult(worker_mode=self.mode, changed_files=("train.py",), success=True,
                            parsed_metrics={"val_score": metric}, raw_log_ref="fake.log", patch_ref="fake.diff",
                            git_commit_before="b", git_commit_after="a")


def _spec():
    return load_registered_node("lr_synthetic", repo_root=ROOT)


def _campaign(tmp, metrics, *, controller=None, memory_mode="append_only_summary", budget=None, **kwargs):
    spec = _spec()
    node = Path(tmp) / "node"
    shutil.copytree(ROOT / "nodes" / "lr_synthetic", node)
    backend = RecordingBackend(spec)
    records_path, events_path = Path(tmp) / "trials.jsonl", Path(tmp) / "events.jsonl"
    run_real_campaign(node_spec=spec, campaign_id="h", budget=budget or len(metrics) + 1,
                      manager_mode="baseline_manager", memory_mode=memory_mode, records_path=records_path,
                      worker=Worker(node, metrics), proposal_backend=backend, controller=controller,
                      event_store=CampaignEventStore(events_path), **kwargs)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    return TrialAppendStore(records_path).read_all(), events, backend


def _strip_event(event):
    return {k: v for k, v in event.items() if k not in ("timestamp", "event_id", "created_at", "ts")}


SKIP2 = {(2, "E1"): "skip"}
DETAIL = {"skip_reason": "prior_no_gain", "probs": [0.5] + [0.05] * 10 + [0.0], "pi": 0.5, "t_star": 0.01}


class TestDecisionDetailAndEvents(unittest.TestCase):
    def test_validation(self):
        ControlDecision("continue", detail={"a": 1}, events=(("prior_unavailable", {"reason": "timeout"}),))
        with self.assertRaises(ValueError):
            ControlDecision("continue", events=(("decision_made", {}),))  # not a controller event type
        with self.assertRaises(ValueError):
            ControlDecision("continue", detail=[1])  # type: ignore[arg-type]
        self.assertEqual(CONTROLLER_EVENT_TYPES, ("prior_unavailable",))

    def test_detail_and_skip_reason_are_recorded(self):
        events_in = (("prior_unavailable", {"reason": "timeout", "fallback": "A1"}),)
        with tempfile.TemporaryDirectory() as tmp:
            records, events, _ = _campaign(tmp, [0.80, 0.85],
                                           controller=ScriptedController(SKIP2, DETAIL, events_in))
        self.assertEqual([r.decision.value for r in records], ["kept", "skipped", "kept"])
        self.assertEqual(records[1].extra["control"], {"epoch": "E1", "action": "skip",
                                                       "skip_reason": "prior_no_gain"})
        decisions = [e for e in events if e["event_type"] == "control_decision"]
        e1 = [e for e in decisions if e["payload"]["epoch"] == "E1" and e["payload"]["action"] == "skip"]
        self.assertEqual(len(e1), 1)
        self.assertEqual(e1[0]["payload"]["detail"], DETAIL)
        self.assertTrue(all("detail" not in e["payload"] for e in decisions if e is not e1[0]))
        unavailable = [e for e in events if e["event_type"] == "prior_unavailable"]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["payload"], {"epoch": "E1", "controller_version": "scripted/v1",
                                                     "reason": "timeout", "fallback": "A1"})
        # the controller event precedes its decision event
        order = [e["event_type"] for e in events]
        self.assertLess(order.index("prior_unavailable"), events.index(e1[0]))

    def test_without_detail_events_and_records_are_unchanged(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain, plain_events, _ = _campaign(a, [0.80, 0.85], controller=ScriptedController(SKIP2))
            with_detail, detail_events, _ = _campaign(b, [0.80, 0.85],
                                                      controller=ScriptedController(SKIP2, {"note": "x"}))
        self.assertEqual(plain[1].extra["control"], {"epoch": "E1", "action": "skip"})
        self.assertEqual(with_detail[1].extra["control"], {"epoch": "E1", "action": "skip"})  # no skip_reason
        payloads = [e["payload"] for e in plain_events if e["event_type"] == "control_decision"]
        self.assertTrue(all(set(p) == {"epoch", "action", "scores", "mask", "controller_version", "input_hash",
                                       "llm_calls"} for p in payloads))


def _record(trial_id, decision="kept", control=None, metric=0.9, summary="s"):
    from autoresearch.memory.schemas import TrialRecord
    with tempfile.TemporaryDirectory() as tmp:
        records, _, _ = _campaign(tmp, [metric])
    base = records[0]
    from autoresearch.memory.schemas import TrialDecision
    return replace(base, trial_id=trial_id, decision=TrialDecision(decision), proposal_summary=summary,
                   parsed_metrics={} if decision == "skipped" else {"val_score": metric},
                   delta_vs_best=None if decision == "skipped" else 0.01,
                   extra={"control": control} if control else {})


class TestSkipDisplay(unittest.TestCase):
    def setUp(self):
        self.spec = _spec()
        self.kept = _record("t-001", "kept", metric=0.9123, summary="first")
        self.cp_skip = _record("t-002", "skipped", {"epoch": "E1", "action": "skip",
                                                    "skip_reason": "prior_no_gain"}, summary="second")
        self.inactive = _record("t-003", "skipped", {"epoch": "E1", "action": "skip", "reason": "inactive_key",
                                                     "key": "class_weight", "revisions": 2}, summary="third")
        self.records = [self.kept, self.cp_skip, self.inactive]

    def test_classifier(self):
        self.assertFalse(is_control_plane_skip(self.kept))
        self.assertTrue(is_control_plane_skip(self.cp_skip))
        self.assertFalse(is_control_plane_skip(self.inactive))

    def test_default_display_is_historical(self):
        for mode in (MemoryMode.APPEND_ONLY_SUMMARY, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE, MemoryMode.NONE):
            old = build_memory_context(self.records, mode, self.spec, 4)
            new = build_memory_context(self.records, mode, self.spec, 4, skip_display=None)
            self.assertEqual(old, new)
        text = build_memory_context(self.records, MemoryMode.APPEND_ONLY_SUMMARY, self.spec, 4).context_text
        self.assertEqual(text, "\n".join(_summary_line(r, include_rationale=False, metric_name="val_score")
                                         for r in self.records))
        self.assertNotIn(SKIP_DISPLAY_TEXT, text)

    def test_one_line_display(self):
        for mode in (MemoryMode.APPEND_ONLY_SUMMARY, MemoryMode.APPEND_ONLY_SUMMARY_WITH_RATIONALE):
            ctx = build_memory_context(self.records, mode, self.spec, 4, skip_display=SKIP_DISPLAY_ONE_LINE)
            lines = ctx.context_text.splitlines()
            hits = [line for line in lines if SKIP_DISPLAY_TEXT in line]
            self.assertEqual(hits, [f"t-002: summary=second; {SKIP_DISPLAY_TEXT}"])
            for forbidden in ("metric=", "delta=", "rationale=", "failure=", "decision="):
                self.assertNotIn(forbidden, hits[0])
            # other records keep the historical line (the kept metric and the inactive-key skip)
            self.assertIn(_summary_line(self.kept, include_rationale=mode != MemoryMode.APPEND_ONLY_SUMMARY,
                                        metric_name="val_score"), lines)
            self.assertTrue(any(line.startswith("t-003: decision=skipped") for line in lines))
        none = build_memory_context(self.records, MemoryMode.NONE, self.spec, 4, skip_display=SKIP_DISPLAY_ONE_LINE)
        self.assertNotIn(SKIP_DISPLAY_TEXT, none.context_text)

    def test_unknown_display_is_refused(self):
        with self.assertRaises(ValueError):
            build_memory_context(self.records, MemoryMode.APPEND_ONLY_SUMMARY, self.spec, 4, skip_display="v2")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _campaign(tmp, [0.8], skip_display="two_lines/v1")

    def test_campaign_shows_the_line_to_the_manager_and_records_the_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, events, backend = _campaign(tmp, [0.80, 0.85], controller=ScriptedController(SKIP2, DETAIL),
                                                 skip_display=SKIP_DISPLAY_ONE_LINE)
        started = [e for e in events if e["event_type"] == "campaign_started"][0]
        self.assertEqual(started["payload"]["skip_display"], SKIP_DISPLAY_ONE_LINE)
        third = backend.contexts[2].splitlines()  # the manager's memory before trial 3
        self.assertEqual(third[1], f"{records[1].trial_id}: summary=edit number 2; {SKIP_DISPLAY_TEXT}")
        self.assertNotIn("metric", third[1])
        with tempfile.TemporaryDirectory() as tmp:
            _, events, backend = _campaign(tmp, [0.80, 0.85], controller=ScriptedController(SKIP2, DETAIL))
        started = [e for e in events if e["event_type"] == "campaign_started"][0]
        self.assertNotIn("skip_display", started["payload"])
        self.assertIn("decision=skipped", backend.contexts[2].splitlines()[1])

    def test_campaigns_without_skips_are_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain, plain_events, plain_backend = _campaign(a, [0.80, 0.85, 0.83])
            shown, shown_events, shown_backend = _campaign(b, [0.80, 0.85, 0.83], skip_display=SKIP_DISPLAY_ONE_LINE)
        strip = lambda r: {k: v for k, v in r.to_dict().items()  # noqa: E731
                           if k not in ("timestamp_start", "timestamp_end", "wall_clock_seconds", "provenance")}
        self.assertEqual([strip(r) for r in plain], [strip(r) for r in shown])
        self.assertEqual(plain_backend.contexts, shown_backend.contexts)
        self.assertEqual([e["event_type"] for e in plain_events], [e["event_type"] for e in shown_events])

    def test_snapshot_round_trip(self):
        for display in (None, SKIP_DISPLAY_ONE_LINE):
            with tempfile.TemporaryDirectory() as tmp:
                recorder = SnapshotRecorder(Path(tmp) / "snaps", epochs=("E2",), take_index=2)
                kwargs = {"skip_display": display} if display else {}
                _campaign(tmp, [0.80, 0.81], snapshot_hook=recorder, **kwargs)
                manifest = json.loads(recorder.manifest_path.read_text())
                snap = read_snapshot(recorder.manifest_path.parent)
            if display is None:
                self.assertNotIn("skip_display", manifest)
            else:
                self.assertEqual(manifest["skip_display"], display)
            self.assertEqual(snap.skip_display, display)


if __name__ == "__main__":
    unittest.main()
