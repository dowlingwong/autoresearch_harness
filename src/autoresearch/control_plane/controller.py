"""Control-plane controller interface and decision epochs (BUILD_RUNBOOK L5).

A controller sees a leak-safe :class:`ControlView` and a feasibility ``mask``
and returns a :class:`ControlDecision`.  Controllers never write the ledger;
the control plane commits every decision.

Epochs inside one trial:

* **E0** — pre-proposal, before the memory context is built.
* **E1** — post-proposal, pre-execution, after the precondition check passes.
* **E2** — post-metric, pre-commit, after the provisional record is built.

``ContinueController`` (the default) always returns ``"continue"`` and is the
only way the unmodified trial records are reproduced exactly.

Feasibility (:func:`compute_mask`):

* E0 ``stop`` needs ``t >= t_min``; ``demote`` needs the *top* of the
  incumbent stack to be an unverified keep (it pops only that entry).
* E1 ``skip`` needs the proposal to have passed its precondition; ``revise``
  needs a non-deterministic manager that accepts ``revision_feedback``, no
  earlier revision this trial, and LLM budget left.
* E2 ``verify`` needs a valid provisional record, a worker with seed injection
  (``run_trial_at_seed``) and compute for ``2r`` runs.

``run_real_campaign`` never executes an infeasible choice: it logs a
``control_decision_rejected`` event and continues.  ``run_dry_campaign``
supports only ``continue``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Epoch = Literal["E0", "E1", "E2"]
ACTIONS: tuple[str, ...] = ("continue", "skip", "revise", "verify", "demote", "stop")

DEFAULT_T_MIN = 3
DEFAULT_VERIFY_RUNS = 2


class ControlViewLeakError(ValueError):
    """Raised when a forbidden input (audit, future or sibling data) is supplied."""


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def record_digest(record: Any) -> dict[str, Any] | None:
    """A stable, leak-safe projection of one ledger record."""
    if record is None:
        return None
    return {
        "trial_id": getattr(record, "trial_id", None),
        "budget_index": getattr(record, "budget_index", None),
        "decision": _enum_value(getattr(record, "decision", None)),
        "parsed_metrics": dict(getattr(record, "parsed_metrics", {}) or {}),
        "delta_vs_best": getattr(record, "delta_vs_best", None),
    }


def proposal_digest(proposal: Any) -> dict[str, Any] | None:
    if proposal is None:
        return None
    extra = getattr(proposal, "extra", {}) or {}
    return {
        "manager_mode": getattr(proposal, "manager_mode", None),
        "proposal_summary": getattr(proposal, "proposal_summary", None),
        "objective": getattr(proposal, "objective", None),
        "structured_edit": extra.get("structured_edit"),
    }


@dataclass(frozen=True)
class ControlView:
    """What a controller may see.  Built only by :func:`build_control_view`."""

    epoch: Epoch
    budget_index: int
    ledger_prefix: tuple[Any, ...]
    current_best: float | None
    incumbent_stack_depth: int
    proposal: Any | None
    provisional: Any | None
    budget: dict[str, Any]
    node: Any

    def canonical_payload(self) -> dict[str, Any]:
        node = self.node
        return {
            "epoch": self.epoch,
            "budget_index": self.budget_index,
            "current_best": self.current_best,
            "incumbent_stack_depth": self.incumbent_stack_depth,
            "budget": {key: self.budget[key] for key in sorted(self.budget)},
            "node": {
                "name": getattr(node, "name", None),
                "metric_name": getattr(node, "metric_name", None),
                "metric_direction": getattr(node, "metric_direction", None),
            },
            "ledger_prefix": [record_digest(record) for record in self.ledger_prefix],
            "proposal": proposal_digest(self.proposal),
            "provisional": record_digest(self.provisional),
        }

    def input_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControlDecision:
    action: str
    scores: dict[str, float] = field(default_factory=dict)
    controller_version: str = "unknown"
    input_hash: str = ""
    # LLM calls the controller itself made for this decision (R1 L12: an LLM
    # self-control side query).  The campaign charges them to the LLM-call cap.
    llm_calls: int = 0

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown control action: {self.action!r}")
        if not isinstance(self.llm_calls, int) or self.llm_calls < 0:
            raise ValueError("llm_calls must be a non-negative int")


class Controller(Protocol):
    version: str

    def decide(self, view: ControlView, mask: dict[str, bool]) -> ControlDecision: ...


class ContinueController:
    """Default controller: always continue; the regime is unchanged."""

    version = "continue/v1"
    deterministic = True

    def decide(self, view: ControlView, mask: dict[str, bool]) -> ControlDecision:
        return ControlDecision(
            action="continue",
            scores={},
            controller_version=self.version,
            input_hash=view.input_hash(),
        )


def build_control_view(
    *,
    epoch: Epoch,
    budget_index: int,
    ledger_prefix: Any = (),
    current_best: float | None = None,
    incumbent_stack_depth: int = 0,
    proposal: Any | None = None,
    provisional: Any | None = None,
    budget: dict[str, Any] | None = None,
    node: Any,
    audit_metrics: Any | None = None,
    future_records: Any | None = None,
    sibling_outcomes: Any | None = None,
) -> ControlView:
    """Build a :class:`ControlView`, refusing forbidden inputs.

    ``audit_metrics``, ``future_records`` and ``sibling_outcomes`` must never be
    provided; passing any non-empty value raises :class:`ControlViewLeakError`.
    The ledger prefix is additionally checked so it cannot contain a record at
    or beyond the current budget index.
    """
    if audit_metrics:
        raise ControlViewLeakError("audit metrics are forbidden in a ControlView")
    if future_records:
        raise ControlViewLeakError("future records are forbidden in a ControlView")
    if sibling_outcomes:
        raise ControlViewLeakError("sibling-branch outcomes are forbidden in a ControlView")

    prefix = tuple(ledger_prefix)
    for record in prefix:
        record_index = getattr(record, "budget_index", None)
        if record_index is not None and record_index >= budget_index:
            raise ControlViewLeakError(
                "ledger_prefix contains a record at or after the current budget index"
            )
    return ControlView(
        epoch=epoch,
        budget_index=budget_index,
        ledger_prefix=prefix,
        current_best=current_best,
        incumbent_stack_depth=incumbent_stack_depth,
        proposal=proposal,
        provisional=provisional,
        budget=dict(budget or {}),
        node=node,
    )


def compute_mask(
    view: ControlView,
    *,
    proposal_passed_precondition: bool = False,
    manager_deterministic: bool = True,
    revision_feedback_supported: bool = True,
    revisions_this_trial: int = 0,
    llm_cap_reached: bool = False,
    seed_injection_supported: bool = True,
    verify_runs: int = DEFAULT_VERIFY_RUNS,
    worker_runs_remaining: int | None = None,
    top_of_stack_unverified: bool = False,
    t_min: int = DEFAULT_T_MIN,
) -> dict[str, bool]:
    """Feasibility mask, following the problem spec §2.1 rules.

    ``revise`` additionally requires ``revision_feedback_supported``: the
    manager must accept the ``revision_feedback`` keyword, otherwise a revision
    would be a plain resample.  ``verify`` requires ``seed_injection_supported``;
    the campaign loops pass ``hasattr(worker, "run_trial_at_seed")``.
    """
    mask = {action: False for action in ACTIONS}
    mask["continue"] = True

    if view.epoch == "E0":
        mask["stop"] = view.budget_index >= t_min
        # demote pops exactly the top incumbent, so it is feasible only when
        # that top entry is an unverified keep (never reverts a verified keep).
        mask["demote"] = bool(top_of_stack_unverified)
    elif view.epoch == "E1":
        mask["skip"] = bool(proposal_passed_precondition)
        mask["revise"] = (
            (not manager_deterministic)
            and revision_feedback_supported
            and revisions_this_trial < 1
            and not llm_cap_reached
        )
    elif view.epoch == "E2":
        trial_valid = (
            view.provisional is not None
            and _enum_value(getattr(view.provisional, "validity_status", None)) == "valid"
        )
        enough_compute = worker_runs_remaining is None or worker_runs_remaining >= 2 * verify_runs
        mask["verify"] = bool(trial_valid and seed_injection_supported and enough_compute)
    return mask
