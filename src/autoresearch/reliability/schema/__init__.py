"""Reliability schema — event types and models for governed scientific-agent traces.

Extends the harness CampaignEvent with reliability-specific event payloads,
manifest schemas, and validation logic. Schema version 1.0.

Contract: forward-compatible readers for old minor versions; explicit migration
functions for major changes; rejection of unknown critical fields.
"""

from .events import (
    ReliabilityEvent,
    ReliabilityEventType,
    EventVisibility,
    SCHEMA_VERSION,
    ProposalEvent,
    ActionRequestEvent,
    ActionResultEvent,
    StateSnapshotEvent,
    StateDiffEvent,
    EvidenceEvent,
    VerifierRequestEvent,
    VerifierResultEvent,
    FaultInjectionEvent,
    BeliefEvent,
    RecoveryActionEvent,
    HumanEscalationEvent,
    TerminalOutcomeEvent,
    AdjudicationEvent,
)
from .manifest import (
    RunManifest,
    FaultManifest,
    SplitManifest,
    TaskManifest,
    ModelManifest,
)
from .validation import (
    validate_manifest,
    validate_event_stream,
    validate_referential_integrity,
    ValidationResult,
)
from .adapter import (
    ReliabilityEventStoreAdapter,
    ReliabilityEventStoreAdapterError,
)
from .export import (
    export_scalar_macro,
    export_latex_table,
    ExportError,
)

__all__ = [
    "ReliabilityEvent",
    "ReliabilityEventType",
    "EventVisibility",
    "SCHEMA_VERSION",
    "ProposalEvent",
    "ActionRequestEvent",
    "ActionResultEvent",
    "StateSnapshotEvent",
    "StateDiffEvent",
    "EvidenceEvent",
    "VerifierRequestEvent",
    "VerifierResultEvent",
    "FaultInjectionEvent",
    "BeliefEvent",
    "RecoveryActionEvent",
    "HumanEscalationEvent",
    "TerminalOutcomeEvent",
    "AdjudicationEvent",
    "RunManifest",
    "FaultManifest",
    "SplitManifest",
    "TaskManifest",
    "ModelManifest",
    "validate_manifest",
    "validate_event_stream",
    "validate_referential_integrity",
    "ValidationResult",
    "ReliabilityEventStoreAdapter",
    "ReliabilityEventStoreAdapterError",
    "export_scalar_macro",
    "export_latex_table",
    "ExportError",
]
