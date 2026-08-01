"""Observability package — state snapshots and semantic diffs.

Adapters for filesystem, git, process, environment, metrics, and
task-specific scientific evidence. Each adapter exposes snapshot(),
diff(before, after), reconcile(), and summarize_for_policy().

Policy summaries derive from the same raw snapshots used by independent
analysis but may redact hidden fault truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class DiffType(StrEnum):
    TEXT = "text"
    BINARY_HASH = "binary_hash"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    METRIC = "metric"
    SEMANTIC = "semantic"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable state capture at a trajectory step."""

    target: str
    snapshot_type: str
    content_hash: str
    timestamp_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, target: str, snapshot_type: str, content: bytes) -> StateSnapshot:
        import time
        return cls(
            target=target,
            snapshot_type=snapshot_type,
            content_hash=hashlib.sha256(content).hexdigest(),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


@dataclass(frozen=True)
class StateDifference:
    """Typed state change record between two snapshots."""

    target: str
    before_ref: str
    after_ref: str
    diff_type: DiffType = DiffType.COMPOSITE
    expected_change: str = ""
    observed_change: str = ""
    reconciled: bool = False
    confidence: float = 1.0


class StateObserver(Protocol):
    """Protocol for observing and diffing state at trajectory steps."""

    def snapshot(self, target: str) -> StateSnapshot: ...
    def diff(self, before: StateSnapshot, after: StateSnapshot) -> StateDifference: ...
    def reconcile(self, target: str, expected: str, actual: str) -> StateDifference: ...
    def summarize_for_policy(self, snapshot: StateSnapshot) -> dict[str, Any]: ...
