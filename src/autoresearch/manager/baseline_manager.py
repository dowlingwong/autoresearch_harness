from __future__ import annotations

import re

from autoresearch.manager.base import ManagerProposal, ManagerStatus
from autoresearch.memory.summarizer import MemoryContext
from autoresearch.nodes.spec import NodeSpec


BOUNDED_OBJECTIVES = (
    (
        "lower-weight-decay",
        "Change WEIGHT_DECAY from 1e-4 to 5e-5 and keep all other hyperparameters unchanged.",
        "A moderate weight-decay reduction is a conservative first trial for the current baseline.",
    ),
    (
        "small-dropout",
        "Change DROPOUT from 0.0 to 0.02 and keep all other hyperparameters unchanged.",
        "A small amount of regularization may improve near-threshold generalization.",
    ),
    (
        "smaller-kernel",
        "Change KERNEL_SIZE from 7 to 5 and keep all other hyperparameters unchanged.",
        "A smaller temporal kernel can reduce smoothing and improve local feature sensitivity.",
    ),
    (
        "lower-grad-clip",
        "Change GRAD_CLIP_NORM from 1.0 to 0.5 and keep all other hyperparameters unchanged.",
        "A stricter gradient clip can stabilize near-threshold training.",
    ),
    (
        "larger-batch",
        "Change BATCH_SIZE from 32 to 64 and keep all other hyperparameters unchanged.",
        "A larger batch can reduce gradient noise while preserving the same model architecture.",
    ),
    (
        "slightly-more-epochs",
        "Change EPOCHS from 5 to 6 and keep all other hyperparameters unchanged.",
        "One additional epoch tests whether the baseline is mildly under-trained.",
    ),
    (
        "lower-learning-rate",
        "Change LEARNING_RATE from 5e-4 to 3e-4 and keep all other hyperparameters unchanged.",
        "A conservative learning-rate reduction tests whether the current baseline is still too aggressive.",
    ),
    (
        "aggressive-weight-decay-reduction",
        "Change WEIGHT_DECAY from 1e-4 to 3e-5 and keep all other hyperparameters unchanged.",
        "A lighter weight decay can improve fit if the current model is underfitting.",
    ),
    (
        "tiny-dropout",
        "Change DROPOUT from 0.0 to 0.01 and keep all other hyperparameters unchanged.",
        "A tiny dropout value tests regularization without heavily changing model capacity.",
    ),
    (
        "mild-weight-decay-reduction",
        "Change WEIGHT_DECAY from 1e-4 to 7e-5 and keep all other hyperparameters unchanged.",
        "A milder weight-decay reduction is distinct from the earlier 5e-5 and 3e-5 trials.",
    ),
    (
        "smaller-batch",
        "Change BATCH_SIZE from 32 to 16 and keep all other hyperparameters unchanged.",
        "A smaller batch can improve generalization if the baseline optimization is too smooth.",
    ),
    (
        "larger-grad-clip",
        "Change GRAD_CLIP_NORM from 1.0 to 2.0 and keep all other hyperparameters unchanged.",
        "A looser gradient clip tests whether the baseline is over-constraining updates.",
    ),
    (
        "larger-kernel",
        "Change KERNEL_SIZE from 7 to 9 and keep all other hyperparameters unchanged.",
        "A larger temporal kernel can capture wider waveform context.",
    ),
    (
        "fast-mode-more-data",
        "Change FAST_SEARCH_N_SIGNAL and FAST_SEARCH_N_NOISE from 1000 to 1500 and keep all other hyperparameters unchanged.",
        "Using more fast-search examples can reduce validation noise while keeping the loop practical.",
    ),
    (
        "fast-mode-longer-trace",
        "Change FAST_SEARCH_TRACE_LEN from 4096 to 5000 and keep all other hyperparameters unchanged.",
        "A slightly longer fast-search trace may preserve more waveform signal.",
    ),
)


def select_bounded_objective(status: ManagerStatus, memory_context: MemoryContext) -> tuple[str, str, str]:
    prior_summaries = set(re.findall(r"summary=([^;\n]+)", memory_context.context_text))
    candidates = list(BOUNDED_OBJECTIVES)
    start = (status.budget_index - 1) % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    for candidate in ordered:
        if candidate[0] not in prior_summaries:
            return candidate
    return ordered[0]


class BaselineManager:
    mode = "baseline_manager"

    def propose_next_trial(
        self,
        status: ManagerStatus,
        memory_context: MemoryContext,
        node_spec: NodeSpec,
    ) -> ManagerProposal:
        summary, objective, rationale = self._select_objective(status, memory_context)
        return ManagerProposal(
            manager_mode=self.mode,
            proposal_summary=summary,
            proposal_rationale=rationale,
            target_files=node_spec.editable_paths,
            objective=objective,
        )

    def _select_objective(self, status: ManagerStatus, memory_context: MemoryContext) -> tuple[str, str, str]:
        return select_bounded_objective(status, memory_context)
