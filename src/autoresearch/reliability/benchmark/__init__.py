"""Benchmark package — task/fault manifests and split resolution.

Maps immutable task, model, scaffold, fault, and split manifests into
executable runs. Enforces group isolation at task-template, repository,
dataset, and injector-implementation level.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..schema.manifest import SplitManifest, TaskManifest


@dataclass(frozen=True)
class SplitAssignment:
    """Assignment of a task to a specific split regime."""

    task_id: str
    split_name: str
    assignment_hash: str = ""

    def __post_init__(self):
        if not self.assignment_hash:
            d = {"task_id": self.task_id, "split_name": self.split_name}
            object.__setattr__(
                self,
                "assignment_hash",
                hashlib.sha256(
                    json.dumps(d, sort_keys=True).encode()
                ).hexdigest()[:16],
            )


@dataclass
class SplitResolver:
    """Resolves group-isolated splits for task/fault/model assignment."""

    splits: dict[str, SplitManifest] = field(default_factory=dict)
    assignments: dict[str, SplitAssignment] = field(default_factory=dict)

    def register_split(self, split: SplitManifest) -> None:
        self.splits[split.split_id] = split

    def assign_task(self, task_id: str, split_id: str) -> SplitAssignment:
        if split_id not in self.splits:
            raise KeyError(f"unknown split: {split_id}")
        assignment = SplitAssignment(task_id=task_id, split_name=split_id)
        self.assignments[task_id] = assignment
        return assignment

    def get_split(self, task_id: str) -> str:
        if task_id not in self.assignments:
            raise KeyError(f"no split assignment for task: {task_id}")
        return self.assignments[task_id].split_name

    def validate_isolation(
        self, train_tasks: set[str], test_tasks: set[str]
    ) -> list[str]:
        """Check that no task template appears in both train and test."""
        errors: list[str] = []
        overlap = train_tasks & test_tasks
        if overlap:
            errors.append(f"task overlap between train and test: {overlap}")
        return errors
