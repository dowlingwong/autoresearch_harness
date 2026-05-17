from __future__ import annotations

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.manager.baseline_manager import select_bounded_objective
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
        editable_path = node_spec.editable_paths[0] if node_spec.editable_paths else "train.py"
        summary, concrete_objective, base_rationale, extra = select_bounded_objective(
            status,
            memory_context,
            editable_path=editable_path,
        )
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
            f"In {', '.join(node_spec.editable_paths)}, {concrete_objective} "
            f"This is the manager-selected bounded edit to improve {status.metric_name}. "
            f"{context_note}{repeated_warning} "
            "Execute this edit; do not propose a different change. "
            "Do not edit frozen paths or dependencies."
        )
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=summary,
            proposal_rationale=(
                f"{base_rationale} Current best={status.current_best_metric}; "
                f"memory_mode={memory_context.mode.value}; context_chars={memory_context.compressed_chars}."
            ),
            target_files=node_spec.editable_paths,
            objective=objective,
            extra=extra,
        )
