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
        "digits_mlp": root / "configs" / "nodes" / "digits_mlp.yaml",
        "tabular_mlp": root / "configs" / "nodes" / "tabular_mlp.yaml",
        "mlagentbench_vectorization": root / "configs" / "nodes" / "mlagentbench_vectorization.yaml",
        "openml_credit_g": root / "configs" / "nodes" / "openml_credit_g.yaml",
        "openml_bank_marketing": root / "configs" / "nodes" / "openml_bank_marketing.yaml",
        "openml_breast_w": root / "configs" / "nodes" / "openml_breast_w.yaml",
        "openml_credit_approval": root / "configs" / "nodes" / "openml_credit_approval.yaml",
        "openml_diabetes": root / "configs" / "nodes" / "openml_diabetes.yaml",
        "openml_ionosphere": root / "configs" / "nodes" / "openml_ionosphere.yaml",
        "openml_wdbc": root / "configs" / "nodes" / "openml_wdbc.yaml",
        "openml_sonar": root / "configs" / "nodes" / "openml_sonar.yaml",
        "openml_blood_transfusion": root / "configs" / "nodes" / "openml_blood_transfusion.yaml",
        "openml_qsar_biodeg": root / "configs" / "nodes" / "openml_qsar_biodeg.yaml",
        "openml_climate": root / "configs" / "nodes" / "openml_climate.yaml",
        "autoresearch_macos": root / "configs" / "nodes" / "autoresearch_macos.yaml",
        "autoresearch_linux": root / "configs" / "nodes" / "autoresearch_linux.yaml",
        "track_g_autoresearch": root / "configs" / "nodes" / "track_g_autoresearch.yaml",
    }
    try:
        return registry[node_name]
    except KeyError as error:
        raise NodeRegistryError(f"unknown node: {node_name}") from error


def load_registered_node(node_name: str, repo_root: str | Path | None = None) -> NodeSpec:
    return load_node_spec(node_spec_path(node_name, repo_root=repo_root))
