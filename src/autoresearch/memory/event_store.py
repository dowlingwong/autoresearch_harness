from __future__ import annotations

import json
from pathlib import Path

from autoresearch.control_plane.events import CampaignEvent


class CampaignEventStoreError(RuntimeError):
    """Raised when an append-only campaign event stream cannot be read."""


class CampaignEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: CampaignEvent) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return self.path

    def read_all(self) -> list[CampaignEvent]:
        if not self.path.exists():
            return []
        events: list[CampaignEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(CampaignEvent.from_mapping(json.loads(line)))
                except Exception as error:
                    raise CampaignEventStoreError(
                        f"invalid campaign event at {self.path}:{line_number}"
                    ) from error
        return events
