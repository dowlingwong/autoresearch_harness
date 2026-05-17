from __future__ import annotations

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.manager.hyperparam_edits import select_structured_hyperparameter_edit
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


def select_bounded_objective(
    status: ManagerStatus,
    memory_context: MemoryContext,
    editable_path: str = "train.py",
) -> tuple[str, str, str, dict[str, object]]:
    return select_structured_hyperparameter_edit(status, memory_context, editable_path=editable_path)


class BaselineManager:
    mode = "baseline_manager"

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        editable_path = node_spec.editable_paths[0] if node_spec.editable_paths else "train.py"
        summary, objective, rationale, extra = self._select_objective(
            status,
            memory_context,
            editable_path=editable_path,
        )
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=summary,
            proposal_rationale=rationale,
            target_files=node_spec.editable_paths,
            objective=objective,
            extra=extra,
        )

    def _select_objective(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        editable_path: str = "train.py",
    ) -> tuple[str, str, str, dict[str, object]]:
        return select_bounded_objective(status, memory_context, editable_path=editable_path)
