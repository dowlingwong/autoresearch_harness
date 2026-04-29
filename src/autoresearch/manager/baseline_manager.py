from __future__ import annotations

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


class BaselineManager:
    mode = "baseline_manager"

    _OBJECTIVES = (
        (
            "lower-learning-rate",
            "Change LEARNING_RATE from 1e-3 to 5e-4 and keep all other hyperparameters unchanged.",
            "A conservative learning-rate reduction is a strong first baseline for stabilizing validation AUC.",
        ),
        (
            "small-dropout",
            "Set DROPOUT to 0.02 and keep all other hyperparameters unchanged.",
            "A small amount of regularization may improve near-threshold generalization.",
        ),
        (
            "lower-weight-decay",
            "Change WEIGHT_DECAY from 1e-4 to 3e-5 and keep all other hyperparameters unchanged.",
            "A lighter weight decay can improve fit if the current model is underfitting.",
        ),
    )

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        index = (status.budget_index - 1) % len(self._OBJECTIVES)
        summary, objective, rationale = self._OBJECTIVES[index]
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=summary,
            proposal_rationale=rationale,
            target_files=node_spec.editable_paths,
            objective=objective,
        )

