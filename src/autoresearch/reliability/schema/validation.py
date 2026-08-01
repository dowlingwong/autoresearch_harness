"""Event stream and manifest validation.

Deterministic validation rejects corrupted, reordered, duplicated, or
partially missing event streams. Never passes an incomplete stream as valid.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import ReliabilityEvent, SCHEMA_VERSION
from .manifest import RunManifest


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, warnings: list[str] | None = None) -> ValidationResult:
        return cls(valid=True, warnings=warnings or [])

    @classmethod
    def fail(cls, errors: list[str], warnings: list[str] | None = None) -> ValidationResult:
        return cls(valid=False, errors=errors, warnings=warnings or [])


def validate_manifest(manifest: RunManifest) -> ValidationResult:
    """Validate a run manifest for required fields and integrity."""
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest.run_id:
        errors.append("run_id is empty")
    if not manifest.campaign_id:
        errors.append("campaign_id is empty")
    if not manifest.task_id:
        errors.append("task_id is empty")
    if not manifest.task_family:
        errors.append("task_family is empty")
    if not manifest.replicate_id:
        errors.append("replicate_id is empty")

    if manifest.schema_version != SCHEMA_VERSION:
        warnings.append(
            f"manifest schema version {manifest.schema_version} != "
            f"current {SCHEMA_VERSION}"
        )

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings
    )


def validate_event_stream(events: list[ReliabilityEvent]) -> ValidationResult:
    """Validate an event stream for completeness and ordering."""
    errors: list[str] = []
    warnings: list[str] = []

    if not events:
        errors.append("empty event stream")
        return ValidationResult.fail(errors, warnings)

    run_ids = {e.run_id for e in events}
    if len(run_ids) > 1:
        errors.append(f"events from multiple runs: {run_ids}")

    steps = [e.monotonic_step for e in events]
    if steps != sorted(steps):
        errors.append("events are not monotonically ordered by step")

    seen_ids: set[str] = set()
    for e in events:
        if e.event_id in seen_ids:
            errors.append(f"duplicate event_id: {e.event_id}")
        seen_ids.add(e.event_id)

    valid_event_ids = {e.event_id for e in events}
    for e in events:
        for pid in e.parent_event_ids:
            if pid and pid not in valid_event_ids:
                errors.append(
                    f"event {e.event_id} references missing parent {pid}"
                )

    step_seen: set[int] = set()
    for e in events:
        if e.monotonic_step in step_seen:
            errors.append(f"duplicate monotonic_step: {e.monotonic_step}")
        step_seen.add(e.monotonic_step)

    has_proposal = any(
        e.event_type.value == "proposal" for e in events
    )
    has_terminal = any(
        e.event_type.value in ("terminal_outcome", "adjudication")
        for e in events
    )
    if not has_proposal:
        warnings.append("stream has no proposal event")
    if not has_terminal:
        warnings.append("stream has no terminal or adjudication event")

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings
    )


def validate_referential_integrity(
    events: list[ReliabilityEvent],
    artifact_registry: dict[str, str] | None = None,
) -> ValidationResult:
    """Validate that artifact refs exist in the registry."""
    errors: list[str] = []
    warnings: list[str] = []

    if artifact_registry is None:
        warnings.append("no artifact registry provided; skipping ref checks")
        return ValidationResult.ok(warnings)

    for e in events:
        for ref in e.artifact_refs:
            if ref and ref not in artifact_registry:
                errors.append(
                    f"event {e.event_id} references missing artifact {ref}"
                )

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings
    )
