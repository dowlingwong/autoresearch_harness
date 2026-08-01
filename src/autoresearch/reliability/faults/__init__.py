"""Fault-injection package — deterministic injectors with causal metadata.

Injectors are registered with capability declarations. Every injector
requires a no-fault control and a replay test. Injection assignment is
block-randomized and recorded in a sealed manifest. The agent cannot
read the fault label, severity, intended symptom, or recovery key.

Initial injectors cover:
1. Malformed or redirected tool arguments
2. Timeout after a completed or partial side effect
3. Stale or corrupted memory retrieval
4. Evidence leakage or metric misuse
5. Incomplete or flaky verifier behavior
6. Conflicting worker ownership or duplicated action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..schema.manifest import FaultManifest


class FaultFamily(StrEnum):
    MODEL_AND_PLANNING = "model_and_planning"
    ACTION_AND_TOOL = "action_and_tool"
    RUNTIME_AND_ENVIRONMENT = "runtime_and_environment"
    MEMORY_AND_CONTEXT = "memory_and_context"
    EVIDENCE_AND_STATISTICS = "evidence_and_statistics"
    VERIFIER = "verifier"
    COORDINATION = "coordination"


class FaultSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InjectionBand(StrEnum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


@dataclass(frozen=True)
class InjectionResult:
    """Result of a fault injection attempt."""

    fault_id: str
    intended: bool
    immediate_state_difference_occurred: bool
    observed_symptoms: list[str] = field(default_factory=list)
    fidelity: float = 0.0


class FaultInjector(Protocol):
    """Protocol for deterministic fault injection."""

    family: FaultFamily

    def can_inject(self, target: str, context: dict[str, Any]) -> bool: ...

    def inject(
        self,
        manifest: FaultManifest,
        target_state: Any,
        seed: str,
    ) -> tuple[Any, InjectionResult]: ...

    def clean_counterfactual(
        self,
        target_state: Any,
        seed: str,
    ) -> Any: ...


@dataclass
class InjectorRegistry:
    """Registry of available fault injectors by family and mechanism."""

    injectors: dict[str, FaultInjector] = field(default_factory=dict)

    def register(self, name: str, injector: FaultInjector) -> None:
        self.injectors[name] = injector

    def get(self, name: str) -> FaultInjector:
        if name not in self.injectors:
            raise KeyError(f"unknown injector: {name}")
        return self.injectors[name]

    def list_families(self) -> list[str]:
        return sorted({inj.family.value for inj in self.injectors.values()})
