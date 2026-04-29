from __future__ import annotations

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext, MemoryMode
from autoresearch.nodes.spec import NodeSpec


class PromptManager:
    mode = "prompt_manager"

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        repeated_warning = ""
        if memory_context.repeated_bad_stats.repeated_bad_count:
            repeated_warning = (
                " Avoid proposals similar to repeated bad trials: "
                + ", ".join(memory_context.repeated_bad_stats.flagged_trial_ids)
                + "."
            )
        context_note = (
            "Use only current node/budget context."
            if memory_context.mode == MemoryMode.NONE
            else "Use the compressed prior-trial memory to avoid repeats."
        )
        objective = (
            f"Propose exactly one bounded change to {', '.join(node_spec.editable_paths)} "
            f"to improve {status.metric_name}. {context_note}{repeated_warning} "
            "Do not edit frozen paths or dependencies."
        )
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=f"memory-guided-proposal-{status.budget_index}",
            proposal_rationale=(
                f"Current best={status.current_best_metric}; "
                f"memory_mode={memory_context.mode.value}; "
                f"context_chars={memory_context.compressed_chars}."
            ),
            target_files=node_spec.editable_paths,
            objective=objective,
        )

