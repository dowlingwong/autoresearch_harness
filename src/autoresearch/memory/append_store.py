from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from autoresearch.memory.schemas import TrialRecord


class AppendOnlyStoreError(RuntimeError):
    """Raised when an append-only trial store cannot be read or written."""


class TrialAppendStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: TrialRecord) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return self.path

    def read_all(self) -> list[TrialRecord]:
        if not self.path.exists():
            return []
        records: list[TrialRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    records.append(TrialRecord.from_mapping(payload))
                except Exception as error:
                    raise AppendOnlyStoreError(f"invalid trial record at {self.path}:{line_number}") from error
        return records

    def append_many(self, records: Iterable[TrialRecord]) -> Path:
        for record in records:
            self.append(record)
        return self.path

