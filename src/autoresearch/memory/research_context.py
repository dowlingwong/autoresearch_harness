from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from autoresearch.nodes.spec import NodeSpec


@dataclass(frozen=True)
class ResearchContextRef:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def write_node_research_context(
    *,
    node_spec: NodeSpec,
    repo_root: str | Path,
    node_root: str | Path,
    output_dir: str | Path | None = None,
) -> ResearchContextRef:
    """Write a deterministic pre-campaign research context for one node."""
    root = Path(repo_root).resolve()
    node_dir = Path(node_root).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else root / "paper" / "notes"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{node_spec.name}_research_context.md"
    content = build_node_research_context(node_spec=node_spec, repo_root=root, node_root=node_dir)
    path.write_text(content, encoding="utf-8")
    return ResearchContextRef(path=str(path), sha256=_sha256_bytes(content.encode("utf-8")))


def load_node_research_context_ref(
    *,
    node_spec: NodeSpec,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
) -> ResearchContextRef | None:
    root = Path(repo_root).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else root / "paper" / "notes"
    path = target_dir / f"{node_spec.name}_research_context.md"
    if not path.exists():
        return None
    return ResearchContextRef(path=str(path), sha256=_sha256_file(path))


def build_node_research_context(
    *,
    node_spec: NodeSpec,
    repo_root: str | Path,
    node_root: str | Path,
    max_chars_per_source: int = 8000,
) -> str:
    root = Path(repo_root).resolve()
    node_dir = Path(node_root).resolve()
    source_paths = [
        root / "configs" / "nodes" / f"{node_spec.name}.yaml",
        node_dir / "program.md",
        node_dir / "README.md",
    ]
    lines = [
        f"# Research Context: {node_spec.name}",
        "",
        "This file is a deterministic pre-campaign snapshot for manager and audit metadata.",
        "",
        "## Node Spec",
        "",
        "```json",
        json.dumps(node_spec.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
        "## Source Notes",
        "",
    ]
    for path in source_paths:
        if not path.exists() or not path.is_file():
            continue
        rel = _relative(path, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "\n\n[truncated]\n"
        lines.extend(
            [
                f"### {rel}",
                "",
                "```text",
                text.rstrip(),
                "```",
                "",
            ]
        )
    content = "\n".join(lines).rstrip() + "\n"
    return content


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
