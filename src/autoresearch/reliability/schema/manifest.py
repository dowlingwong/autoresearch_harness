"""Run manifest and sealed fault manifest models.

Every trajectory begins with an immutable manifest. The manifest is signed
or hashed before execution. A run with a post-hoc edited manifest is invalid.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode("utf-8")


@dataclass(frozen=True)
class RunManifest:
    """Immutable manifest for a single trajectory run."""

    run_id: str
    campaign_id: str
    task_id: str
    task_family: str
    replicate_id: str

    # Version group
    git_commit: str = ""
    dirty_tree_digest: str = ""
    environment_lock_hash: str = ""
    schema_version: str = "1.0"

    # Agent group
    provider: str = ""
    model_id: str = ""
    model_revision: str = ""
    decoding_configuration: dict[str, Any] = field(default_factory=dict)
    scaffold_version: str = ""

    # Resources group
    wall_clock_budget_seconds: int = 0
    token_budget: int = 0
    tool_call_budget: int = 0
    compute_class: str = ""

    # Assignment group
    experimental_arm: str = ""
    block: str = ""
    randomization_seed: str = ""
    split_name: str = ""

    # Fault group
    sealed_fault_manifest_hash: str = ""

    # Evaluation group
    adjudicator_versions: tuple[str, ...] = ()
    primary_endpoint_version: str = ""
    analysis_plan_hash: str = ""

    # Governance group
    editable_paths: tuple[str, ...] = ()
    network_policy: str = ""
    secret_policy: str = ""
    approval_requirements: str = ""

    # Integrity
    manifest_hash: str = ""

    def __post_init__(self):
        if not self.manifest_hash:
            d = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ("manifest_hash",)
            }
            object.__setattr__(
                self,
                "manifest_hash",
                hashlib.sha256(_canonical_json(d)).hexdigest(),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "replicate_id": self.replicate_id,
            "git_commit": self.git_commit,
            "dirty_tree_digest": self.dirty_tree_digest,
            "environment_lock_hash": self.environment_lock_hash,
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "decoding_configuration": self.decoding_configuration,
            "scaffold_version": self.scaffold_version,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "token_budget": self.token_budget,
            "tool_call_budget": self.tool_call_budget,
            "compute_class": self.compute_class,
            "experimental_arm": self.experimental_arm,
            "block": self.block,
            "randomization_seed": self.randomization_seed,
            "split_name": self.split_name,
            "sealed_fault_manifest_hash": self.sealed_fault_manifest_hash,
            "adjudicator_versions": list(self.adjudicator_versions),
            "primary_endpoint_version": self.primary_endpoint_version,
            "analysis_plan_hash": self.analysis_plan_hash,
            "editable_paths": list(self.editable_paths),
            "network_policy": self.network_policy,
            "secret_policy": self.secret_policy,
            "approval_requirements": self.approval_requirements,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=d["run_id"],
            campaign_id=d.get("campaign_id", ""),
            task_id=d.get("task_id", ""),
            task_family=d.get("task_family", ""),
            replicate_id=d.get("replicate_id", ""),
            git_commit=d.get("git_commit", ""),
            dirty_tree_digest=d.get("dirty_tree_digest", ""),
            environment_lock_hash=d.get("environment_lock_hash", ""),
            schema_version=d.get("schema_version", "1.0"),
            provider=d.get("provider", ""),
            model_id=d.get("model_id", ""),
            model_revision=d.get("model_revision", ""),
            decoding_configuration=d.get("decoding_configuration", {}),
            scaffold_version=d.get("scaffold_version", ""),
            wall_clock_budget_seconds=d.get("wall_clock_budget_seconds", 0),
            token_budget=d.get("token_budget", 0),
            tool_call_budget=d.get("tool_call_budget", 0),
            compute_class=d.get("compute_class", ""),
            experimental_arm=d.get("experimental_arm", ""),
            block=d.get("block", ""),
            randomization_seed=d.get("randomization_seed", ""),
            split_name=d.get("split_name", ""),
            sealed_fault_manifest_hash=d.get("sealed_fault_manifest_hash", ""),
            adjudicator_versions=tuple(d.get("adjudicator_versions", [])),
            primary_endpoint_version=d.get("primary_endpoint_version", ""),
            analysis_plan_hash=d.get("analysis_plan_hash", ""),
            editable_paths=tuple(d.get("editable_paths", [])),
            network_policy=d.get("network_policy", ""),
            secret_policy=d.get("secret_policy", ""),
            approval_requirements=d.get("approval_requirements", ""),
            manifest_hash=d.get("manifest_hash", ""),
        )


@dataclass(frozen=True)
class FaultManifest:
    """Sealed fault assignment created before the run."""

    fault_id: str
    fault_family: str
    mechanism: str
    causal_parent: str
    affected_component: str
    injection_trigger: str
    deterministic_seed: str
    severity: str
    persistence: str
    observability_class: str
    expected_immediate_effects: list[str] = field(default_factory=list)
    expected_downstream_effects: list[str] = field(default_factory=list)
    reversibility: str = ""
    valid_recovery_classes: list[str] = field(default_factory=list)
    clean_counterfactual_run_id: str = ""
    policy_visible_fields: list[str] = field(default_factory=list)
    hidden_fields: list[str] = field(default_factory=list)
    injector_implementation_hash: str = ""
    manifest_hash: str = ""

    def __post_init__(self):
        if not self.manifest_hash:
            d = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ("manifest_hash",)
            }
            object.__setattr__(
                self,
                "manifest_hash",
                hashlib.sha256(_canonical_json(d)).hexdigest(),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_family": self.fault_family,
            "mechanism": self.mechanism,
            "causal_parent": self.causal_parent,
            "affected_component": self.affected_component,
            "injection_trigger": self.injection_trigger,
            "deterministic_seed": self.deterministic_seed,
            "severity": self.severity,
            "persistence": self.persistence,
            "observability_class": self.observability_class,
            "expected_immediate_effects": self.expected_immediate_effects,
            "expected_downstream_effects": self.expected_downstream_effects,
            "reversibility": self.reversibility,
            "valid_recovery_classes": self.valid_recovery_classes,
            "clean_counterfactual_run_id": self.clean_counterfactual_run_id,
            "policy_visible_fields": self.policy_visible_fields,
            "hidden_fields": self.hidden_fields,
            "injector_implementation_hash": self.injector_implementation_hash,
            "manifest_hash": self.manifest_hash,
        }

    def policy_visible_subset(self) -> dict[str, Any]:
        """Return only the fields the evaluated policy is permitted to observe."""
        d = self.to_dict()
        permitted = set(self.policy_visible_fields)
        return {k: v for k, v in d.items() if k in permitted}


@dataclass(frozen=True)
class SplitManifest:
    """Group-isolated split definition for task/fault/model separation."""

    split_id: str
    split_regime: str
    description: str
    task_template_ids: tuple[str, ...] = ()
    task_instance_ids: tuple[str, ...] = ()
    fault_family_ids: tuple[str, ...] = ()
    grouping_rules: list[str] = field(default_factory=list)
    leakage_audit_hash: str = ""


@dataclass(frozen=True)
class TaskManifest:
    """Scientific task definition."""

    task_id: str
    task_family: str
    workload_level: str
    description: str
    generator_hash: str = ""
    repository_url: str = ""
    dataset_hash: str = ""


@dataclass(frozen=True)
class ModelManifest:
    """Model configuration descriptor."""

    model_id: str
    provider: str
    model_family: str
    revision: str
    decoding_configuration: dict[str, Any] = field(default_factory=dict)
    scaffold_version: str = ""
