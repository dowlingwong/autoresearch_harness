from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from autoresearch.manager.base import ManagerStatus
from autoresearch.memory.summarizer import MemoryContext


@dataclass(frozen=True)
class HyperparameterCandidate:
    summary: str
    symbol: str
    target: str
    rationale: str


HYPERPARAMETER_CANDIDATES: tuple[HyperparameterCandidate, ...] = (
    HyperparameterCandidate(
        "lower-weight-decay",
        "WEIGHT_DECAY",
        "3e-5",
        "A lighter weight decay can improve fit if the current model is underfitting.",
    ),
    HyperparameterCandidate(
        "small-dropout",
        "DROPOUT",
        "0.022",
        "A small dropout adjustment tests regularization without changing model capacity.",
    ),
    HyperparameterCandidate(
        "smaller-kernel",
        "KERNEL_SIZE",
        "3",
        "A smaller temporal kernel can reduce smoothing and improve local feature sensitivity.",
    ),
    HyperparameterCandidate(
        "lower-grad-clip",
        "GRAD_CLIP_NORM",
        "0.25",
        "A stricter gradient clip can stabilize near-threshold training.",
    ),
    HyperparameterCandidate(
        "larger-batch",
        "BATCH_SIZE",
        "64",
        "A larger batch can reduce gradient noise while preserving the same model architecture.",
    ),
    HyperparameterCandidate(
        "slightly-more-epochs",
        "EPOCHS",
        "6",
        "One additional epoch tests whether the baseline is mildly under-trained.",
    ),
    HyperparameterCandidate(
        "lower-learning-rate",
        "LEARNING_RATE",
        "3e-4",
        "A conservative learning-rate reduction tests whether the current baseline is too aggressive.",
    ),
    HyperparameterCandidate(
        "higher-learning-rate",
        "LEARNING_RATE",
        "1e-3",
        "A higher learning rate tests whether the current baseline is too conservative.",
    ),
    HyperparameterCandidate(
        "much-lower-learning-rate",
        "LEARNING_RATE",
        "1e-4",
        "A strongly reduced learning rate tests whether fine-grained updates improve convergence.",
    ),
    HyperparameterCandidate(
        "very-high-learning-rate",
        "LEARNING_RATE",
        "2e-3",
        "An aggressive learning rate tests whether fast initial descent escapes local minima.",
    ),
    HyperparameterCandidate(
        "mild-weight-decay",
        "WEIGHT_DECAY",
        "7e-5",
        "A milder weight-decay increase tests regularization sensitivity around the current value.",
    ),
    HyperparameterCandidate(
        "very-low-weight-decay",
        "WEIGHT_DECAY",
        "1e-5",
        "Very light weight decay tests whether the model benefits from near-zero L2 regularization.",
    ),
    HyperparameterCandidate(
        "moderate-weight-decay",
        "WEIGHT_DECAY",
        "1e-4",
        "A moderate weight-decay increase tests whether stronger regularization reduces overfitting.",
    ),
    HyperparameterCandidate(
        "high-weight-decay",
        "WEIGHT_DECAY",
        "3e-4",
        "Strong weight decay tests whether the model is significantly overfit at lower regularization strengths.",
    ),
    HyperparameterCandidate(
        "tiny-dropout",
        "DROPOUT",
        "0.01",
        "A tiny dropout value tests regularization with minimal capacity change.",
    ),
    HyperparameterCandidate(
        "zero-dropout",
        "DROPOUT",
        "0.0",
        "Disabling dropout tests whether stochastic regularization is counter-productive on this task.",
    ),
    HyperparameterCandidate(
        "moderate-dropout",
        "DROPOUT",
        "0.05",
        "A moderate dropout rate tests whether increased stochastic regularization improves generalization.",
    ),
    HyperparameterCandidate(
        "high-dropout",
        "DROPOUT",
        "0.10",
        "A high dropout rate tests whether strong stochastic regularization helps on this waveform task.",
    ),
    HyperparameterCandidate(
        "smaller-batch",
        "BATCH_SIZE",
        "16",
        "A smaller batch can improve generalization if optimization is too smooth.",
    ),
    HyperparameterCandidate(
        "larger-grad-clip",
        "GRAD_CLIP_NORM",
        "1.0",
        "A looser gradient clip tests whether current clipping is over-constraining updates.",
    ),
    HyperparameterCandidate(
        "very-low-grad-clip",
        "GRAD_CLIP_NORM",
        "0.1",
        "A very strict gradient clip tests whether sharply bounded updates stabilize near-threshold training.",
    ),
    HyperparameterCandidate(
        "high-grad-clip",
        "GRAD_CLIP_NORM",
        "2.0",
        "A very loose gradient clip tests whether current clipping is over-constraining parameter updates.",
    ),
    HyperparameterCandidate(
        "larger-kernel",
        "KERNEL_SIZE",
        "7",
        "A larger temporal kernel can capture wider waveform context.",
    ),
    HyperparameterCandidate(
        "wide-kernel",
        "KERNEL_SIZE",
        "9",
        "A wide temporal kernel tests whether capturing longer waveform context improves trigger detection.",
    ),
    HyperparameterCandidate(
        "larger-hidden-dim",
        "HIDDEN_DIM",
        "16",
        "A larger hidden layer tests whether the synthetic MLP is capacity-limited.",
    ),
    HyperparameterCandidate(
        "smaller-hidden-dim",
        "HIDDEN_DIM",
        "4",
        "A smaller hidden layer tests whether the synthetic MLP is over-parameterized.",
    ),
    HyperparameterCandidate(
        "stronger-regularization",
        "REGULARIZATION",
        "0.005",
        "Stronger L2 regularization tests whether the model is overfitting the synthetic boundary.",
    ),
    HyperparameterCandidate(
        "weaker-regularization",
        "REGULARIZATION",
        "0.0002",
        "Weaker L2 regularization tests whether regularization is suppressing useful capacity.",
    ),
    HyperparameterCandidate(
        "more-n-epochs",
        "N_EPOCHS",
        "120",
        "More epochs test whether the NumPy model is under-trained at the current budget.",
    ),
    HyperparameterCandidate(
        "tabular-random-forest",
        "model_type",
        "random_forest",
        "A random forest tests whether the tabular task benefits from nonlinear feature interactions.",
    ),
    HyperparameterCandidate(
        "tabular-gradient-boosting",
        "model_type",
        "gradient_boosting",
        "Gradient boosting tests whether sequential tree ensembles improve the tabular validation AUC.",
    ),
    HyperparameterCandidate(
        "tabular-logistic-regression",
        "model_type",
        "logistic_regression",
        "Logistic regression tests whether the simpler linear model is sufficient for the tabular task.",
    ),
    HyperparameterCandidate(
        "tabular-weaker-logreg-regularization",
        "C",
        "2.0",
        "A larger logistic-regression C weakens regularization on the tabular pipeline.",
    ),
    HyperparameterCandidate(
        "tabular-stronger-logreg-regularization",
        "C",
        "0.5",
        "A smaller logistic-regression C strengthens regularization on the tabular pipeline.",
    ),
    HyperparameterCandidate(
        "tabular-deeper-trees",
        "max_depth",
        "12",
        "Deeper trees test whether the tabular model is capacity-limited.",
    ),
    HyperparameterCandidate(
        "tabular-shallower-trees",
        "max_depth",
        "4",
        "Shallower trees test whether the tabular model is overfitting.",
    ),
    HyperparameterCandidate(
        "tabular-more-estimators",
        "n_estimators",
        "250",
        "More trees can reduce ensemble variance on the public tabular task.",
    ),
    HyperparameterCandidate(
        "tabular-lower-learning-rate",
        "learning_rate",
        "0.02",
        "A lower boosting learning rate can improve validation AUC when more gradual updates help.",
    ),
    HyperparameterCandidate(
        "tabular-null-class-weight",
        "class_weight",
        "null",
        "Removing class weighting tests whether imbalance correction is hurting calibration.",
    ),
    HyperparameterCandidate(
        "tabular-balanced-class-weight",
        "class_weight",
        "balanced",
        "Balanced class weights test whether the public tabular task benefits from imbalance correction.",
    ),
    HyperparameterCandidate(
        "tabular-standard-scaler",
        "scaler",
        "standard",
        "Standard scaling tests whether normalized numeric features improve the selected model.",
    ),
    HyperparameterCandidate(
        "tabular-minmax-scaler",
        "scaler",
        "minmax",
        "Min-max scaling tests whether bounded numeric features improve the selected model.",
    ),
    HyperparameterCandidate(
        "tabular-no-scaler",
        "scaler",
        "none",
        "Disabling numeric scaling tests whether preprocessing is unnecessary for the selected model.",
    ),
    HyperparameterCandidate(
        "tabular-constant-imputer",
        "imputer",
        "constant",
        "Constant categorical imputation tests whether explicit missingness improves validation AUC.",
    ),
    HyperparameterCandidate(
        "tabular-more-max-iter",
        "max_iter",
        "1500",
        "More optimizer iterations test whether the sklearn logistic model is convergence-limited.",
    ),
    HyperparameterCandidate(
        "vectorized-im2col",
        "implementation",
        "im2col",
        "The im2col implementation tests whether replacing nested loops with matrix multiplication improves the vectorization task.",
    ),
    HyperparameterCandidate(
        "vectorized-einsum",
        "implementation",
        "einsum",
        "The einsum implementation tests whether a direct vectorized contraction improves the frozen convolution workload.",
    ),
    HyperparameterCandidate(
        "vectorization-larger-batch",
        "batch_size",
        "12",
        "A larger benchmark batch tests whether the selected implementation scales without changing the frozen correctness check.",
    ),
    HyperparameterCandidate(
        "vectorization-smaller-image",
        "image_size",
        "24",
        "A smaller image size tests whether workload scale changes the measured throughput score.",
    ),
    HyperparameterCandidate(
        "vectorization-more-filters",
        "num_filters",
        "12",
        "More filters test whether the implementation remains efficient as output-channel work increases.",
    ),
    HyperparameterCandidate(
        "vectorization-fewer-repeats",
        "repeat_count",
        "2",
        "Fewer timing repeats reduce benchmark cost while retaining the same metric parser and correctness guard.",
    ),
)


EFFECTIVE_KEYS: dict[str, str] = {
    "BATCH_SIZE": "batch_size",
    "EPOCHS": "epochs",
    "N_EPOCHS": "n_epochs",
    "LEARNING_RATE": "learning_rate",
    "WEIGHT_DECAY": "weight_decay",
    "KERNEL_SIZE": "kernel_size",
    "DROPOUT": "dropout",
    "GRAD_CLIP_NORM": "grad_clip_norm",
    "HIDDEN_DIM": "hidden_dim",
    "REGULARIZATION": "regularization",
    "C": "c",
    "MAX_ITER": "max_iter",
    "CLASS_WEIGHT": "class_weight",
    "MIN_FREQUENCY": "min_frequency",
    "model_type": "model_type",
    "max_depth": "max_depth",
    "n_estimators": "n_estimators",
    "learning_rate": "learning_rate",
    "class_weight": "class_weight",
    "scaler": "scaler",
    "imputer": "imputer",
    "max_iter": "max_iter",
    "implementation": "implementation",
    "batch_size": "batch_size",
    "image_size": "image_size",
    "num_filters": "num_filters",
    "kernel_size": "kernel_size",
    "stride": "stride",
    "padding": "padding",
    "repeat_count": "repeat_count",
}


def parse_train_constants(train_path: str | Path) -> dict[str, str]:
    """Return top-level ALL_CAPS constants from a Python training file."""
    path = Path(train_path)
    if not path.exists():
        return {}
    constants: dict[str, str] = {}
    pattern = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>[^#\n]+?)(?:\s*#.*)?$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            constants[match.group("name")] = match.group("value").strip()
    if constants:
        return constants
    if path.suffix.lower() in {".yaml", ".yml"}:
        yaml_pattern = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>[^#\n]*?)(?:\s*#.*)?$")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = yaml_pattern.match(line.strip())
            if match:
                constants[match.group("name")] = match.group("value").strip() or "null"
    return constants


def build_effective_config(
    constants: Mapping[str, str],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Approximate the training config that will actually be used.

    This is intentionally narrow: it covers the bounded manager constants used
    by the ResNet trigger node and the dependency-light synthetic nodes. In
    fast-search mode, BATCH_SIZE and EPOCHS are not effective for the ResNet
    trigger because train.py uses the fast-search environment overrides.
    """
    env = environ or os.environ
    fast_search = _env_bool(env.get("RESNET_TRIGGER_FAST_SEARCH", "0"))
    config = {
        "batch_size": str(constants.get("batch_size", constants.get("BATCH_SIZE", ""))),
        "epochs": str(constants.get("EPOCHS", "")),
        "n_epochs": str(constants.get("N_EPOCHS", "")),
        "learning_rate": str(constants.get("learning_rate", constants.get("LEARNING_RATE", ""))),
        "weight_decay": str(constants.get("WEIGHT_DECAY", "")),
        "kernel_size": str(constants.get("KERNEL_SIZE", "")),
        "dropout": str(constants.get("DROPOUT", "")),
        "grad_clip_norm": str(constants.get("GRAD_CLIP_NORM", "")),
        "hidden_dim": str(constants.get("HIDDEN_DIM", "")),
        "regularization": str(constants.get("REGULARIZATION", "")),
        "c": str(constants.get("C", "")),
        "min_frequency": str(constants.get("MIN_FREQUENCY", "")),
        "model_type": str(constants.get("model_type", "")),
        "max_depth": str(constants.get("max_depth", "")),
        "n_estimators": str(constants.get("n_estimators", "")),
        "class_weight": str(constants.get("class_weight", constants.get("CLASS_WEIGHT", ""))),
        "scaler": str(constants.get("scaler", "")),
        "imputer": str(constants.get("imputer", "")),
        "max_iter": str(constants.get("max_iter", constants.get("MAX_ITER", ""))),
        "implementation": str(constants.get("implementation", "")),
        "image_size": str(constants.get("image_size", "")),
        "num_filters": str(constants.get("num_filters", "")),
        "kernel_size": str(constants.get("kernel_size", "")),
        "stride": str(constants.get("stride", "")),
        "padding": str(constants.get("padding", "")),
        "repeat_count": str(constants.get("repeat_count", "")),
    }
    if fast_search:
        config["batch_size"] = str(env.get("RESNET_TRIGGER_FAST_BATCH_SIZE", "64"))
        config["epochs"] = str(env.get("RESNET_TRIGGER_FAST_EPOCHS", "3"))
    return config


def constants_after_edit(
    constants: Mapping[str, str],
    *,
    symbol: str,
    new_value: str,
) -> dict[str, str]:
    updated = dict(constants)
    updated[symbol] = new_value
    return updated


def select_structured_hyperparameter_edit(
    status: ManagerStatus,
    memory_context: MemoryContext,
    editable_path: str = "train.py",
) -> tuple[str, str, str, dict[str, object]]:
    prior_summaries = set(re.findall(r"summary=([^;\n]+)", memory_context.context_text))
    constants = dict(status.current_constants or {})
    effective_config = dict(status.effective_config or {})
    skipped: list[dict[str, str]] = []
    candidates = list(HYPERPARAMETER_CANDIDATES)
    start = (status.budget_index - 1) % len(candidates)
    ordered = candidates[start:] + candidates[:start]

    # Identify symbols pinned by environment overrides — editing them in train.py
    # has no effect on the actual training run, so exclude them upfront with a
    # distinct reason rather than letting them fall through to effective_config_unchanged.
    env_overridden_symbols: set[str] = set()
    if _env_bool(os.environ.get("RESNET_TRIGGER_FAST_SEARCH", "0")):
        env_overridden_symbols.update({"BATCH_SIZE", "EPOCHS"})

    if not constants:
        for candidate in ordered:
            if candidate.summary not in prior_summaries:
                return (
                    candidate.summary,
                    f"Change {candidate.symbol} to {candidate.target} in {editable_path} and keep all other hyperparameters unchanged.",
                    candidate.rationale,
                    {"state_aware": False, "reason": "current_train_constants_unavailable"},
                )
        candidate = ordered[0]
        return (
            candidate.summary,
            f"Change {candidate.symbol} to {candidate.target} in {editable_path} and keep all other hyperparameters unchanged.",
            candidate.rationale,
            {"state_aware": False, "reason": "current_train_constants_unavailable"},
        )

    for candidate in ordered:
        if candidate.summary in prior_summaries:
            skipped.append({"summary": candidate.summary, "reason": "already_attempted"})
            continue
        if candidate.symbol in env_overridden_symbols:
            skipped.append({"summary": candidate.summary, "reason": "env_overridden"})
            continue
        current = constants.get(candidate.symbol)
        if current is None:
            skipped.append({"summary": candidate.summary, "reason": "symbol_missing"})
            continue
        if values_equal(current, candidate.target):
            skipped.append({"summary": candidate.summary, "reason": "already_at_target"})
            continue
        effective_key = EFFECTIVE_KEYS.get(candidate.symbol)
        after_constants = constants_after_edit(
            constants,
            symbol=candidate.symbol,
            new_value=candidate.target,
        )
        after_effective = build_effective_config(after_constants)
        before_effective_value = effective_config.get(effective_key or "", "")
        after_effective_value = after_effective.get(effective_key or "", "")
        if effective_key and values_equal(before_effective_value, after_effective_value):
            skipped.append({"summary": candidate.summary, "reason": "effective_config_unchanged"})
            continue

        objective = (
            f"Change {candidate.symbol} from {current} to {candidate.target} in {editable_path} "
            "and keep all other hyperparameters unchanged."
        )
        structured_edit = {
            "type": "config_value" if editable_path.endswith((".yaml", ".yml")) else "python_constant",
            "path": editable_path,
            "symbol": candidate.symbol,
            "old": current,
            "new": candidate.target,
            "effective_key": effective_key or "",
            "effective_before": before_effective_value,
            "effective_after": after_effective_value,
        }
        return (
            candidate.summary,
            objective,
            candidate.rationale,
            {
                "structured_edit": structured_edit,
                "deterministic_patch": True,
                "current_constants": constants,
                "effective_config": effective_config,
                "skipped_candidates": skipped,
            },
        )

    fallback = ordered[0]
    objective = (
        f"No non-no-op structured hyperparameter edit is available for train.py. "
        f"First skipped candidate was {fallback.summary}."
    )
    return (
        "no-valid-hyperparameter-edit",
        objective,
        "The state-aware proposal selector could not find an edit that changes the current effective config.",
        {
            "proposal_precondition_failed": True,
            "failure_category": "proposal_precondition_failed",
            "current_constants": constants,
            "effective_config": effective_config,
            "skipped_candidates": skipped,
        },
    )


def values_equal(left: str | object, right: str | object) -> bool:
    left_s = str(left).strip()
    right_s = str(right).strip()
    if left_s == right_s:
        return True
    try:
        return float(left_s) == float(right_s)
    except ValueError:
        return left_s.replace(" ", "") == right_s.replace(" ", "")


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
