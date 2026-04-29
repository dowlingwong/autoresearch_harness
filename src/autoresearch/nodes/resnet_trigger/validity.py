from __future__ import annotations

from autoresearch.control_plane.permissions import ScopeValidationResult, validate_edit_scope
from autoresearch.nodes.spec import NodeSpec


def validate_resnet_scope(changed_paths: list[str] | tuple[str, ...], node_spec: NodeSpec) -> ScopeValidationResult:
    return validate_edit_scope(changed_paths, node_spec)

