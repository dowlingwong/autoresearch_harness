from __future__ import annotations

import tempfile
from pathlib import Path

from autoresearch.control_plane.campaign import run_dry_campaign, run_real_campaign
from autoresearch.control_plane.events import CampaignEvent
from autoresearch.memory.event_store import CampaignEventStore
from autoresearch.nodes.registry import load_registered_node
from autoresearch.worker.base import WorkerResult


class InMemoryEventStore:
    def __init__(self) -> None:
        self.events: list[CampaignEvent] = []

    def append(self, event: CampaignEvent) -> None:
        self.events.append(event)


class FakeWorker:
    mode = "fake_worker"

    def __init__(self, result: WorkerResult) -> None:
        self.result = result

    def run_trial(self, proposal, node_spec, budget_index):
        return self.result


def _spec():
    return load_registered_node("resnet_trigger")


def _worker_result(*, changed_files=("train.py",), metrics=None) -> WorkerResult:
    spec = _spec()
    return WorkerResult(
        worker_mode="fake_worker",
        changed_files=tuple(changed_files),
        success=True,
        parsed_metrics={spec.metric_name: 0.81} if metrics is None else metrics,
        raw_log_ref="fake.log",
        patch_ref="fake.diff",
        git_commit_before="before",
        git_commit_after="after",
    )


def test_campaign_event_store_is_append_only(tmp_path: Path) -> None:
    store = CampaignEventStore(tmp_path / "events.jsonl")
    first = CampaignEvent.create(
        campaign_id="c",
        trial_id=None,
        event_type="campaign_started",
    )
    second = CampaignEvent.create(
        campaign_id="c",
        trial_id="c-trial-001",
        event_type="trial_started",
        payload={"budget_index": 1},
    )
    store.append(first)
    store.append(second)
    events = store.read_all()
    assert [event.event_type for event in events] == ["campaign_started", "trial_started"]
    assert events[1].payload["budget_index"] == 1


def test_dry_run_event_sequence_is_deterministic() -> None:
    spec = _spec()
    event_store = InMemoryEventStore()
    with tempfile.TemporaryDirectory() as tmp:
        run_dry_campaign(
            node_spec=spec,
            campaign_id="events",
            budget=2,
            manager_mode="baseline_manager",
            memory_mode="none",
            records_path=Path(tmp) / "events_trials.jsonl",
            event_store=event_store,
        )

    event_types = [event.event_type for event in event_store.events]
    per_trial = [
        "trial_started",
        "memory_context_built",
        "proposal_created",
        "pending_guard_acquired",
        "worker_started",
        "worker_finished",
        "scope_validated",
        "metric_parsed",
        "decision_made",
        "trial_record_appended",
        "pending_guard_released",
    ]
    assert event_types == ["campaign_started", *per_trial, *per_trial, "campaign_completed"]
    assert all(event.campaign_id == "events" for event in event_store.events)
    assert all(event.timestamp for event in event_store.events)


def test_failed_invalid_event_includes_failure_category() -> None:
    spec = _spec()
    event_store = InMemoryEventStore()
    with tempfile.TemporaryDirectory() as tmp:
        run_real_campaign(
            node_spec=spec,
            campaign_id="stress",
            budget=1,
            manager_mode="baseline_manager",
            memory_mode="none",
            records_path=Path(tmp) / "stress_trials.jsonl",
            worker=FakeWorker(_worker_result(changed_files=("prepare.py",))),
            event_store=event_store,
        )

    decisions = [event for event in event_store.events if event.event_type == "decision_made"]
    assert len(decisions) == 1
    assert decisions[0].payload["decision"] == "failed_invalid"
    assert decisions[0].payload["failure_category"] == "invalid_edit_scope"


def test_kdd_main_dry_run_script_writes_event_file(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    records = tmp_path / "ledgers" / "kdd_event_smoke_trials.jsonl"
    run = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_kdd_main_campaign.py"),
            "--node",
            "resnet_trigger",
            "--budget",
            "1",
            "--campaign-id",
            "kdd_event_smoke",
            "--dry-run",
            "--records",
            str(records),
            "--no-export",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    events_path = tmp_path / "events" / "kdd_event_smoke_events.jsonl"
    events = CampaignEventStore(events_path).read_all()
    assert events[0].event_type == "campaign_started"
    assert events[-1].event_type == "campaign_completed"
