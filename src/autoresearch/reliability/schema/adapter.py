"""Compatibility adapter: reliability events to the existing append-only store.

The reliability layer emits typed ReliabilityEvent records. This adapter maps
them into the harness CampaignEvent envelope and writes them to the existing
CampaignEventStore (append-only JSONL). It does NOT fork or duplicate the
control-plane event store; it extends it through a compatibility adapter.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.control_plane.events import CampaignEvent
from autoresearch.memory.event_store import CampaignEventStore

from .events import ReliabilityEvent, ReliabilityEventType, EventVisibility, SCHEMA_VERSION


class ReliabilityEventStoreAdapterError(RuntimeError):
    """Raised when a reliability event cannot be persisted or replayed."""


class ReliabilityEventStoreAdapter:
    """Writes ReliabilityEvent records into the existing append-only event store.

    Each ReliabilityEvent becomes one CampaignEvent line with a discriminator
    in the payload so the reliability event can be reconstructed on read.
    """

    def __init__(self, store: CampaignEventStore, campaign_id: str):
        self._store = store
        self._campaign_id = campaign_id

    @classmethod
    def open(cls, path: str | Path, campaign_id: str) -> "ReliabilityEventStoreAdapter":
        return cls(CampaignEventStore(path), campaign_id)

    def append(self, event: ReliabilityEvent) -> Path:
        d = event.to_dict()
        payload = {
            "reliability_event": True,
            "schema_version": d["schema_version"],
            "run_id": d["run_id"],
            "parent_event_ids": d["parent_event_ids"],
            "event_type": d["event_type"],
            "actor_type": d["actor_type"],
            "actor_id": d["actor_id"],
            "monotonic_step": d["monotonic_step"],
            "payload": d["payload"],
            "artifact_refs": d["artifact_refs"],
            "visibility_class": d["visibility_class"],
            "integrity_hash": d["integrity_hash"],
        }
        campaign = CampaignEvent.create(
            campaign_id=self._campaign_id,
            trial_id=event.run_id,
            event_type=event.event_type.value,
            payload=payload,
        )
        return self._store.append(campaign)

    def read_all(self) -> list[ReliabilityEvent]:
        events: list[ReliabilityEvent] = []
        for campaign in self._store.read_all():
            payload = campaign.payload
            if not payload.get("reliability_event"):
                continue
            try:
                events.append(
                    ReliabilityEvent.from_dict(
                        {
                            "event_id": campaign.event_id,
                            "schema_version": payload["schema_version"],
                            "run_id": payload["run_id"],
                            "parent_event_ids": payload["parent_event_ids"],
                            "event_type": payload["event_type"],
                            "actor_type": payload["actor_type"],
                            "actor_id": payload["actor_id"],
                            "monotonic_step": payload["monotonic_step"],
                            "wall_time_utc": campaign.timestamp,
                            "payload": payload["payload"],
                            "artifact_refs": payload["artifact_refs"],
                            "visibility_class": payload["visibility_class"],
                            "integrity_hash": payload["integrity_hash"],
                        }
                    )
                )
            except Exception as error:
                raise ReliabilityEventStoreAdapterError(
                    f"cannot reconstruct reliability event from {self._store.path}"
                ) from error
        return events
