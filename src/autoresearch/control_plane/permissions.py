from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from autoresearch.nodes.spec import NodeSpec


@dataclass(frozen=True)
class ScopeValidationResult:
    valid: bool
    changed_paths: tuple[str, ...]
    violations: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.valid:
            raise PermissionError("; ".join(self.violations))


def validate_edit_scope(changed_paths: list[str] | tuple[str, ...], node_spec: NodeSpec) -> ScopeValidationResult:
    normalized = tuple(_normalize_path(path) for path in changed_paths if path)
    violations: list[str] = []
    for path in normalized:
        if not _matches_any(path, node_spec.editable_paths):
            violations.append(f"{path} is outside editable_paths")
        if _matches_any(path, node_spec.frozen_paths):
            violations.append(f"{path} matches frozen_paths")
    return ScopeValidationResult(
        valid=not violations,
        changed_paths=normalized,
        violations=tuple(violations),
    )


def _normalize_path(path: str) -> str:
    normalized = str(PurePosixPath(path.replace("\\", "/")))
    if normalized == ".":
        return ""
    return normalized.lstrip("/")


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(path, pattern) for pattern in patterns)


def _matches(path: str, pattern: str) -> bool:
    normalized_pattern = _normalize_path(pattern)
    if normalized_pattern.endswith("/"):
        return path.startswith(normalized_pattern)
    if normalized_pattern.endswith("/*"):
        prefix = normalized_pattern[:-1]
        return path.startswith(prefix)
    return path == normalized_pattern or path.startswith(f"{normalized_pattern}/")

