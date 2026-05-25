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
        "lr_synthetic": root / "configs" / "nodes" / "lr_synthetic.yaml",
        "mlp_synthetic": root / "configs" / "nodes" / "mlp_synthetic.yaml",
        "mlagentbench_vectorization": root / "configs" / "nodes" / "mlagentbench_vectorization.yaml",
        "openml_credit_g": root / "configs" / "nodes" / "openml_credit_g.yaml",
        "openml_bank_marketing": root / "configs" / "nodes" / "openml_bank_marketing.yaml",
        "autoresearch_macos": root / "configs" / "nodes" / "autoresearch_macos.yaml",
        "autoresearch_linux": root / "configs" / "nodes" / "autoresearch_linux.yaml",
    }
    try:
        return registry[node_name]
    except KeyError as error:
        raise NodeRegistryError(f"unknown node: {node_name}") from error


def load_registered_node(node_name: str, repo_root: str | Path | None = None) -> NodeSpec:
    return load_node_spec(node_spec_path(node_name, repo_root=repo_root))
