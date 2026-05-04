from __future__ import annotations

import re

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
        (
            "tiny-dropout",
            "Set DROPOUT to 0.01 and keep all other hyperparameters unchanged.",
            "A tiny dropout value tests regularization without heavily changing model capacity.",
        ),
        (
            "moderate-weight-decay",
            "Change WEIGHT_DECAY from 1e-4 to 5e-5 and keep all other hyperparameters unchanged.",
            "A moderate weight-decay reduction is distinct from the more aggressive 3e-5 trial.",
        ),
        (
            "larger-batch",
            "Change BATCH_SIZE from 32 to 64 and keep all other hyperparameters unchanged.",
            "A larger batch can reduce gradient noise while preserving the same model architecture.",
        ),
        (
            "smaller-batch",
            "Change BATCH_SIZE from 32 to 16 and keep all other hyperparameters unchanged.",
            "A smaller batch can improve generalization if the baseline optimization is too smooth.",
        ),
        (
            "slightly-more-epochs",
            "Change EPOCHS from 5 to 6 and keep all other hyperparameters unchanged.",
            "One additional epoch tests whether the baseline is mildly under-trained.",
        ),
        (
            "lower-grad-clip",
            "Change GRAD_CLIP_NORM from 1.0 to 0.5 and keep all other hyperparameters unchanged.",
            "A stricter gradient clip can stabilize near-threshold training.",
        ),
        (
            "larger-grad-clip",
            "Change GRAD_CLIP_NORM from 1.0 to 2.0 and keep all other hyperparameters unchanged.",
            "A looser gradient clip tests whether the baseline is over-constraining updates.",
        ),
        (
            "smaller-kernel",
            "Change KERNEL_SIZE from 7 to 5 and keep all other hyperparameters unchanged.",
            "A smaller temporal kernel can reduce smoothing and improve local feature sensitivity.",
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
        prior_summaries = set(re.findall(r"summary=([^;\n]+)", memory_context.context_text))
        candidates = list(self._OBJECTIVES)
        start = (status.budget_index - 1) % len(candidates)
        ordered = candidates[start:] + candidates[:start]
        for candidate in ordered:
            if candidate[0] not in prior_summaries:
                return candidate
        return ordered[0]
