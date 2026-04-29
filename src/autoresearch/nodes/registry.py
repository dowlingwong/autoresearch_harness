from __future__ import annotations

from pathlib import Path

from autoresearch.nodes.spec import NodeSpec, load_node_spec


class NodeRegistryError(KeyError):
    """Raised when a node name is not registered."""


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def node_spec_path(node_name: str, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root else repo_root_from_here()
    registry = {
        "resnet_trigger": root / "configs" / "nodes" / "resnet_trigger.yaml",
    }
    try:
        return registry[node_name]
    except KeyError as error:
        raise NodeRegistryError(f"unknown node: {node_name}") from error


def load_registered_node(node_name: str, repo_root: str | Path | None = None) -> NodeSpec:
    return load_node_spec(node_spec_path(node_name, repo_root=repo_root))

