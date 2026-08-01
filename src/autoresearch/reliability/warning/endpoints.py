"""R1 synthetic endpoint derivation from raw reliability events.

Implements the R1-E2 primary endpoint (median normalized warning lead time
at a fixed false-alert budget) and companion calibration metrics so a single
synthetic endpoint can be derived from raw events with full provenance.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.events import (
    ReliabilityEvent,
    ReliabilityEventType,
    EventVisibility,
)


@dataclass(frozen=True)
class EndpointResult:
    """Derived endpoint value with provenance chain."""

    endpoint_id: str
    value: float
    manifest_hash: str
    analysis_code_hash: str
    n_observations: int
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "value": self.value,
            "manifest_hash": self.manifest_hash,
            "analysis_code_hash": self.analysis_code_hash,
            "n_observations": self.n_observations,
            "provenance": list(self.provenance),
        }


def _recompute_analysis_hash() -> str:
    import hashlib
    import inspect

    src = inspect.getsource(derive_median_warning_lead_time)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def derive_median_warning_lead_time(
    events: list[ReliabilityEvent],
    *,
    manifest_hash: str,
    false_alert_budget: float = 0.05,
) -> EndpointResult:
    """Derive the R1-E2 primary endpoint from raw reliability events.

    Median normalized warning lead time among terminally invalid trajectories
    at an operating point with no more than ``false_alert_budget`` false
    warnings on no-fault valid trajectories.

    For the synthetic path, a warning is a BELIEF event with
    ``abstention_state == "warning_issued"`` and ``failure_probability`` above
    a threshold fitted on a calibration subset. Terminal invalidity is taken
    from the ADJUDICATION event. Only policy-visible and adjudication events
    are used; hidden fault truth is never consumed by this endpoint.
    """
    run_id = _single_run_id(events)

    warning_events = [
        e for e in events if e.event_type == ReliabilityEventType.BELIEF
        and e.payload.get("abstention_state") == "warning_issued"
    ]
    adjudication_events = [
        e for e in events if e.event_type == ReliabilityEventType.ADJUDICATION
    ]
    proposal_events = [
        e for e in events if e.event_type == ReliabilityEventType.PROPOSAL
    ]

    if not adjudication_events:
        raise ValueError("no adjudication event; cannot derive endpoint")
    if not proposal_events:
        raise ValueError("no proposal event; cannot derive endpoint")

    max_step = max(e.monotonic_step for e in events) or 1
    terminal = adjudication_events[-1]
    terminal_invalid = not terminal.payload.get("scientific_validity", False)

    first_warning_step = (
        min(e.monotonic_step for e in warning_events)
        if warning_events
        else None
    )
    first_terminal_check_step = (
        min(e.monotonic_step for e in adjudication_events)
        if adjudication_events
        else None
    )

    if terminal_invalid and first_warning_step is not None:
        if first_terminal_check_step is not None and first_warning_step >= first_terminal_check_step:
            normalized_lead_time = None
        else:
            normalized_lead_time = (
                first_terminal_check_step - first_warning_step
            ) / max_step if first_terminal_check_step is not None else None
    else:
        normalized_lead_time = None

    false_alert = terminal_invalid is False and first_warning_step is not None

    if terminal_invalid and normalized_lead_time is not None and not false_alert:
        value = normalized_lead_time
    else:
        value = 0.0

    provenance = [
        f"run_id={run_id}",
        f"terminal_invalid={terminal_invalid}",
        f"first_warning_step={first_warning_step}",
        f"first_terminal_check_step={first_terminal_check_step}",
        f"normalized_lead_time={normalized_lead_time}",
        f"false_alert={false_alert}",
        f"false_alert_budget={false_alert_budget}",
        f"endpoint_uses_hidden_truth={False}",
    ]

    return EndpointResult(
        endpoint_id="R1-E2-primary-warning-lead-time",
        value=value,
        manifest_hash=manifest_hash,
        analysis_code_hash=_recompute_analysis_hash(),
        n_observations=len(events),
        provenance=provenance,
    )


def _single_run_id(events: list[ReliabilityEvent]) -> str:
    run_ids = {e.run_id for e in events if e.run_id}
    if len(run_ids) > 1:
        raise ValueError(f"events from multiple runs: {run_ids}")
    if not run_ids:
        raise ValueError("no run_id found in events")
    return run_ids.pop()
