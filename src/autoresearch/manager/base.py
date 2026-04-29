from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


@dataclass(frozen=True)
class ManagerStatus:
    campaign_id: str
    budget_index: int
    current_best_metric: float | None
    metric_name: str
    metric_direction: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManagerProposal:
    manager_mode: str
    proposal_summary: str
    proposal_rationale: str
    target_files: tuple[str, ...]
    objective: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_files"] = list(self.target_files)
        return payload


class Manager(Protocol):
    mode: str

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        ...
