from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class CampaignEvent:
    event_id: str
    campaign_id: str
    trial_id: str | None
    event_type: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        campaign_id: str,
        trial_id: str | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> "CampaignEvent":
        return cls(
            event_id=uuid4().hex[:16],
            campaign_id=campaign_id,
            trial_id=trial_id,
            event_type=event_type,
            timestamp=_iso_now(),
            payload=payload or {},
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CampaignEvent":
        return cls(
            event_id=str(payload["event_id"]),
            campaign_id=str(payload["campaign_id"]),
            trial_id=str(payload["trial_id"]) if payload.get("trial_id") is not None else None,
            event_type=str(payload["event_type"]),
            timestamp=str(payload["timestamp"]),
            payload=dict(payload.get("payload", {})),
        )


def emit(
    event_store: Any | None,
    *,
    campaign_id: str,
    trial_id: str | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> CampaignEvent | None:
    if event_store is None:
        return None
    event = CampaignEvent.create(
        campaign_id=campaign_id,
        trial_id=trial_id,
        event_type=event_type,
        payload=payload,
    )
    event_store.append(event)
    return event


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
