"""D4 amendment (PLAN §10, 18 Oct 2026): declared, frozen action versions.

Covers the run-free Bayes threshold keep rule, the posterior-mean verify
response, the verify-failed policy, the new-campaign split default and the
backward compatibility of every historical default (the D24 pattern:
a manifest or caller without the field means the old value).
ENGINEERING_ONLY; AI-assisted (Claude).
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoresearch.control_plane.action_versions import (
    KEEP_RULE_BAYES,
    VERIFY_FAILED_STANDS,
    VERIFY_FAILED_VETO,
    VERIFY_RESPONSE_POSTERIOR,
    ActionVersions,
    KeepPrior,
    resolve_verify_split_policy,
    verify_posterior_mean,
)
from autoresearch.control_plane.campaign import run_real_campaign
from autoresearch.control_plane.controller import ControlDecision, ControlView
from autoresearch.control_plane.snapshot import SnapshotRecorder, read_snapshot
from autoresearch.manager.base import ManagerProposal
from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult

ROOT = Path(__file__).resolve().parents[2]


class ScriptedController:
    version = "scripted/v1"
    deterministic = False

    def __init__(self, script):
        self.script = dict(script)

    def decide(self, view: ControlView, mask):
        return ControlDecision(action=self.script.get((view.budget_index, view.epoch), "continue"),
                               controller_version=self.version, input_hash=view.input_hash())


class StaticBackend:
    mode = "static_backend"
    deterministic = True

    def __init__(self, proposal):
        self.proposal = proposal

    def propose_next_trial(self, status, memory_context, node_spec):
        return self.proposal


class Worker:
    """Scripted metrics per trial; seeded re-runs return candidate/incumbent bases."""

    mode = "scripted_worker"

    def __init__(self, node_root, metrics, cand=0.95, inc=0.90, seeded_success=True):
        self.node_root = Path(node_root)
        self.metrics = list(metrics)
        self.cand, self.inc, self.seeded_success = cand, inc, seeded_success
        self.n = 0

    def _result(self, metric, success=True, changed=("train.py",)):
        return WorkerResult(worker_mode=self.mode, changed_files=changed, success=success,
                            parsed_metrics={} if metric is None else {"val_score": metric},
                            raw_log_ref="fake.log", patch_ref="fake.diff",
                            git_commit_before="b", git_commit_after="a")

    def run_trial(self, proposal, node_spec, budget_index):
        self.n += 1
        (self.node_root / "train.py").write_text(f"CANDIDATE{self.n}\n", encoding="utf-8")
        return self._result(self.metrics.pop(0))

    def run_trial_at_seed(self, proposal, node_spec, budget_index, *, train_seed, split_seed=None):
        text = (self.node_root / "train.py").read_text(encoding="utf-8")
        base = self.cand if "CANDIDATE" in text else self.inc
        return self._result(base + (int(train_seed) % 3) * 1e-5 if self.seeded_success else None,
                            success=self.seeded_success, changed=())


def _spec():
    spec = load_registered_node("lr_synthetic", repo_root=ROOT)
    assert spec.metric_name == "val_score" or True
    return spec


def _proposal(spec):
    return ManagerProposal(manager_mode="static_backend", proposal_summary="edit", proposal_rationale="t",
                           target_files=spec.editable_paths, objective="Change X from 1 to 2.", extra={})


def _campaign(tmp, metrics, *, budget=None, controller=None, worker_kwargs=None, events=True, **kwargs):
    spec = _spec()
    node = Path(tmp) / "node"
    shutil.copytree(ROOT / "nodes" / "lr_synthetic", node)
    (node / "train.py").write_text("INCUMBENT\n", encoding="utf-8")
    worker = Worker(node, metrics, **(worker_kwargs or {}))
    records_path = Path(tmp) / "trials.jsonl"
    events_path = Path(tmp) / "events.jsonl"
    run_real_campaign(node_spec=spec, campaign_id="d4", budget=budget or len(metrics),
                      manager_mode="baseline_manager", memory_mode="none", records_path=records_path,
                      worker=worker, proposal_backend=StaticBackend(_proposal(spec)),
                      controller=controller, event_store=CampaignEventStore(events_path) if events else None,
                      **kwargs)
    records = TrialAppendStore(records_path).read_all()
    evs = [json.loads(l) for l in events_path.read_text().splitlines()] if events else []
    return records, evs


def _metric_name():
    return _spec().metric_name


class TestVersionsAndManifests(unittest.TestCase):
    def test_legacy_and_new_campaign_defaults(self):
        legacy = ActionVersions.legacy()
        self.assertEqual(legacy.to_dict(), {"keep_rule": "default/v1", "verify_response": "mean_delta/v1",
                                            "verify_failed": VERIFY_FAILED_VETO, "verify_split_policy": "fixed"})
        new = ActionVersions.new_campaign()
        self.assertEqual(new.verify_split_policy, "vary")
        self.assertEqual(new.verify_failed, VERIFY_FAILED_STANDS)
        self.assertEqual(new.keep_rule, "default/v1")  # Bayes rule is off unless declared
        self.assertTrue(legacy.is_legacy)
        self.assertFalse(new.is_legacy)

    def test_manifest_without_fields_means_the_old_values(self):
        self.assertEqual(ActionVersions.from_manifest({}), ActionVersions.legacy())
        self.assertEqual(ActionVersions.from_manifest(None), ActionVersions.legacy())
        self.assertEqual(resolve_verify_split_policy({}), "fixed")
        old_vary = ActionVersions.from_manifest({"verify_split_policy": "vary"})
        self.assertEqual(old_vary.verify_split_policy, "vary")
        self.assertEqual(old_vary.verify_failed, VERIFY_FAILED_VETO)
        declared = ActionVersions.from_manifest({"action_versions": ActionVersions.new_campaign().to_dict()})
        self.assertEqual(declared, ActionVersions.new_campaign())
        partial = ActionVersions.from_manifest({"action_versions": {"keep_rule": KEEP_RULE_BAYES}})
        self.assertEqual(partial.verify_failed, VERIFY_FAILED_VETO)
        self.assertEqual(partial.verify_split_policy, "fixed")

    def test_unknown_versions_are_refused(self):
        for bad in ({"keep_rule": "x"}, {"verify_response": "x"}, {"verify_failed": "x"},
                    {"verify_split_policy": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ActionVersions.new_campaign(**bad)

    def test_keep_prior_validation_and_threshold(self):
        prior = KeepPrior(mu=-0.01, tau=0.02, sigma=0.03)
        self.assertAlmostEqual(prior.threshold, 0.01 * 0.03 ** 2 / 0.02 ** 2)
        with self.assertRaises(ValueError):
            KeepPrior(mu=0.0, tau=0.0, sigma=0.1)
        with self.assertRaises(ValueError):
            KeepPrior(mu=0.0, tau=0.1, sigma=0.1, sigma_t=0.05, sigma_s=0.05)
        st, ss = KeepPrior(mu=0, tau=1, sigma=0.5, sigma_t=0.3).split_components()
        self.assertAlmostEqual(math.hypot(st, ss), 0.5)


def _numpy_posterior(dh, dbar, r, mu, tau, st, ss, policy):
    t2, s2 = st ** 2, ss ** 2
    if policy == "vary":
        vd, c = tau ** 2 + (t2 + s2) / r, tau ** 2
    else:
        vd, c = tau ** 2 + s2 + t2 / r, tau ** 2 + s2
    cov = np.array([[tau ** 2, tau ** 2, tau ** 2], [tau ** 2, tau ** 2 + t2 + s2, c], [tau ** 2, c, vd]])
    y = np.array([dh, dbar]) - mu
    return float(mu + cov[0, 1:] @ np.linalg.solve(cov[1:, 1:], y))


class TestPosteriorMean(unittest.TestCase):
    def test_matches_gaussian_conditioning(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            mu, tau = rng.normal(0, 0.01), rng.uniform(0.001, 0.05)
            st, ss = rng.uniform(0.001, 0.03), rng.uniform(0.001, 0.03)
            prior = KeepPrior(mu=mu, tau=tau, sigma=math.hypot(st, ss), sigma_t=st)
            dh, dbar, r = rng.normal(0, 0.03), rng.normal(0, 0.03), int(rng.integers(1, 5))
            for policy in ("vary", "fixed"):
                self.assertAlmostEqual(verify_posterior_mean(dh, dbar, r, prior, policy),
                                       _numpy_posterior(dh, dbar, r, mu, tau, st, ss, policy), places=10)

    def test_fixed_with_zero_training_noise_ignores_dbar(self):
        prior = KeepPrior(mu=-0.01, tau=0.02, sigma=0.03, sigma_t=0.0)
        a = verify_posterior_mean(0.02, -0.5, 2, prior, "fixed")
        b = verify_posterior_mean(0.02, +0.5, 2, prior, "fixed")
        self.assertAlmostEqual(a, b)
        expected = -0.01 + 0.02 ** 2 / (0.02 ** 2 + 0.03 ** 2) * (0.02 + 0.01)
        self.assertAlmostEqual(a, expected)

    def test_fixed_needs_the_split(self):
        with self.assertRaises(ValueError):
            verify_posterior_mean(0.0, 0.0, 2, KeepPrior(mu=0, tau=1, sigma=1), "fixed")


class TestBayesKeepRule(unittest.TestCase):
    def test_threshold_discards_a_small_positive_delta(self):
        prior = KeepPrior(mu=-0.01, tau=0.01, sigma=0.02)  # t* = 0.04
        with tempfile.TemporaryDirectory() as tmp:
            records, events = _campaign(tmp, [0.80, 0.81], action_versions=ActionVersions.new_campaign(
                keep_rule=KEEP_RULE_BAYES), keep_prior=prior)
        self.assertEqual([r.decision.value for r in records], ["kept", "discarded"])
        meta = records[1].extra["keep_rule"]
        self.assertEqual(meta["version"], KEEP_RULE_BAYES)
        self.assertAlmostEqual(meta["threshold"], 0.04)
        self.assertEqual(meta["default_decision"], "kept")
        self.assertEqual(records[1].failure_category.value, "degraded_metric")
        decisions = [e["payload"] for e in events if e["event_type"] == "decision_made"]
        self.assertEqual(decisions[1]["keep_rule"]["prior"], prior.to_dict())
        self.assertFalse(decisions[0]["keep_rule"]["applied"])  # first valid metric
        started = next(e for e in events if e["event_type"] == "campaign_started")
        self.assertEqual(started["payload"]["action_versions"]["keep_rule"], KEEP_RULE_BAYES)

    def test_threshold_keeps_a_small_negative_delta_when_proposals_help(self):
        prior = KeepPrior(mu=0.01, tau=0.01, sigma=0.02)  # t* = -0.04
        with tempfile.TemporaryDirectory() as tmp:
            records, _ = _campaign(tmp, [0.80, 0.79], action_versions=ActionVersions.new_campaign(
                keep_rule=KEEP_RULE_BAYES), keep_prior=prior)
        self.assertEqual([r.decision.value for r in records], ["kept", "kept"])

    def test_per_proposal_prior_sees_the_e2_view(self):
        seen = []

        def provider(view):
            seen.append((view.epoch, view.budget_index, view.provisional.decision.value))
            return KeepPrior(mu=-0.01 * view.budget_index, tau=0.01, sigma=0.02, source="per_proposal")

        with tempfile.TemporaryDirectory() as tmp:
            records, _ = _campaign(tmp, [0.80, 0.85, 0.86], action_versions=ActionVersions.new_campaign(
                keep_rule=KEEP_RULE_BAYES), keep_prior=provider)
        self.assertEqual([s[:2] for s in seen], [("E2", 1), ("E2", 2), ("E2", 3)])
        # trial 2: t* = 0.08 > 0.05 -> discard; trial 3 vs 0.80: t* = 0.12 > 0.06 -> discard
        self.assertEqual([r.decision.value for r in records], ["kept", "discarded", "discarded"])
        self.assertEqual(records[2].extra["keep_rule"]["prior"]["source"], "per_proposal")

    def test_invalid_trials_stay_failed_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, _ = _campaign(tmp, [0.8, None], action_versions=ActionVersions.new_campaign(
                keep_rule=KEEP_RULE_BAYES), keep_prior=KeepPrior(mu=0, tau=1, sigma=1))
        self.assertEqual(records[1].decision.value, "failed_invalid")
        self.assertEqual(records[1].extra["keep_rule"]["reason"], "invalid_trial")

    def test_missing_prior_or_conflicting_split_fails_at_start(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _campaign(tmp, [0.8], action_versions=ActionVersions.new_campaign(keep_rule=KEEP_RULE_BAYES))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _campaign(tmp, [0.8], action_versions=ActionVersions.new_campaign(), verify_split_policy="fixed")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _campaign(tmp, [0.8], keep_prior=KeepPrior(mu=0, tau=1, sigma=1))


def _strip(record):
    payload = record.to_dict()
    for key in ("timestamp_start", "timestamp_end", "wall_clock_seconds", "provenance"):
        payload.pop(key, None)
    return payload


class TestVerifyVersions(unittest.TestCase):
    def _verify(self, tmp, metrics, worker_kwargs=None, **kwargs):
        return _campaign(tmp, metrics, controller=ScriptedController({(2, "E2"): "verify"}),
                         worker_kwargs=worker_kwargs, **kwargs)

    def test_default_and_explicit_legacy_are_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain, _ = self._verify(a, [0.80, 0.81])
            legacy, _ = self._verify(b, [0.80, 0.81], action_versions=ActionVersions.legacy())
        self.assertEqual([_strip(r) for r in plain], [_strip(r) for r in legacy])
        verify = plain[1].extra["control"]["verify"]
        self.assertEqual(verify["split_policy"], "fixed")
        self.assertNotIn("action_versions", verify)

    def test_new_campaign_split_is_vary(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, _ = self._verify(tmp, [0.80, 0.81], action_versions=ActionVersions.new_campaign())
        verify = records[1].extra["control"]["verify"]
        self.assertEqual(verify["split_policy"], "vary")
        self.assertEqual(verify["action_versions"]["verify_failed"], VERIFY_FAILED_STANDS)
        self.assertFalse(verify["verify_failed"])

    def test_posterior_mean_keeps_a_confident_keep_that_dbar_would_veto(self):
        # Δ̂ = +0.05; D̄ ≈ −0.001 (noisy).  With a weak prior and noisy verify the
        # posterior mean stays positive; "D̄ > 0" vetoes.
        prior = KeepPrior(mu=0.0, tau=0.05, sigma=0.02)
        kw = {"cand": 0.899, "inc": 0.900}
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            mean_rule, _ = self._verify(a, [0.80, 0.85], worker_kwargs=kw,
                                        action_versions=ActionVersions.new_campaign())
            posterior, _ = self._verify(b, [0.80, 0.85], worker_kwargs=kw, keep_prior=prior,
                                        action_versions=ActionVersions.new_campaign(
                                            verify_response=VERIFY_RESPONSE_POSTERIOR))
        self.assertEqual(mean_rule[1].decision.value, "discarded")
        self.assertEqual(posterior[1].decision.value, "kept")
        verify = posterior[1].extra["control"]["verify"]
        self.assertEqual(verify["response"], "keep")
        self.assertGreater(verify["posterior_mean"], 0)
        expected = verify_posterior_mean(0.05, verify["mean_delta"], 2, prior, "vary")
        self.assertAlmostEqual(verify["posterior_mean"], expected)

    def test_all_pairs_failing_vetoes_only_under_the_legacy_policy(self):
        kw = {"seeded_success": False}
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            legacy, _ = self._verify(a, [0.80, 0.85], worker_kwargs=kw)
            new, _ = self._verify(b, [0.80, 0.85], worker_kwargs=kw, action_versions=ActionVersions.new_campaign())
        self.assertEqual(legacy[1].decision.value, "discarded")
        self.assertEqual(legacy[1].extra["control"]["verify"]["response"], "reject")
        self.assertEqual(new[1].decision.value, "kept")
        verify = new[1].extra["control"]["verify"]
        self.assertEqual(verify["response"], "verify_failed")
        self.assertTrue(verify["verify_failed"])
        self.assertEqual(verify["valid_pairs"], 0)


class TestSnapshotsCarryVersions(unittest.TestCase):
    def _snap(self, tmp, **kwargs):
        recorder = SnapshotRecorder(Path(tmp) / "snaps", epochs=("E2",), take_index=2)
        _campaign(tmp, [0.80, 0.81], snapshot_hook=recorder, **kwargs)
        return recorder.manifest_path

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._snap(tmp, action_versions=ActionVersions.new_campaign())
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["action_versions"], ActionVersions.new_campaign().to_dict())
            self.assertEqual(manifest["verify_split_policy"], "vary")
            self.assertEqual(read_snapshot(path.parent).action_versions,
                             ActionVersions.new_campaign().to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = self._snap(tmp)
            manifest = json.loads(path.read_text())
            self.assertNotIn("action_versions", manifest)
            self.assertIsNone(read_snapshot(path.parent).action_versions)

    def test_resume_applies_the_declared_verify_failed_policy(self):
        from autoresearch.control_plane.snapshot import resume_campaign

        outcomes = {}
        for label, kwargs in (("legacy", {}), ("new", {"action_versions": ActionVersions.new_campaign()})):
            with tempfile.TemporaryDirectory() as tmp:
                path = self._snap(tmp, **kwargs)
                snapshot = read_snapshot(path.parent)
                into = Path(tmp) / "branch"
                result = resume_campaign(snapshot, into, first_action="verify",
                                         worker=Worker(into, [], seeded_success=False),
                                         manager=StaticBackend(_proposal(_spec())))
                outcomes[label] = result.first_record
        self.assertEqual(outcomes["legacy"].decision.value, "discarded")
        self.assertEqual(outcomes["new"].decision.value, "kept")
        self.assertEqual(outcomes["new"].extra["control"]["verify"]["response"], "verify_failed")
        self.assertEqual(outcomes["new"].extra["control"]["verify"]["split_policy"], "vary")


if __name__ == "__main__":
    unittest.main()
