"""Repository-relative path resolution for the autoresearch package.

All scripts and modules that need to reference well-known repo directories
should use these helpers rather than constructing paths ad-hoc.

The repo root is determined by walking up from this file until a directory
containing ``pyproject.toml`` is found.  This makes the helpers work
regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* until we find the directory that contains pyproject.toml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate repository root from {start}. "
        "Expected a pyproject.toml in an ancestor directory."
    )


# Resolve once at import time.
_THIS_FILE = Path(__file__)
REPO_ROOT: Path = _find_repo_root(_THIS_FILE)

# ---------------------------------------------------------------------------
# Well-known directories
# ---------------------------------------------------------------------------

SRC_DIR: Path = REPO_ROOT / "src"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"
PAPER_DIR: Path = REPO_ROOT / "paper"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
DOCS_DIR: Path = REPO_ROOT / "docs"

# Sub-directories under experiments/
LEDGERS_DIR: Path = EXPERIMENTS_DIR / "ledgers"
ARTIFACTS_DIR: Path = EXPERIMENTS_DIR / "artifacts"
SUMMARIES_DIR: Path = EXPERIMENTS_DIR / "summaries"
RUNS_DIR: Path = EXPERIMENTS_DIR / "runs"

# Sub-directories under paper/
PAPER_TABLES_DIR: Path = PAPER_DIR / "tables"
PAPER_FIGURES_DIR: Path = PAPER_DIR / "figures"
PAPER_NOTES_DIR: Path = PAPER_DIR / "notes"

# Sub-directories under configs/
NODES_CONFIG_DIR: Path = CONFIGS_DIR / "nodes"
CAMPAIGNS_CONFIG_DIR: Path = CONFIGS_DIR / "campaigns"
MANAGERS_CONFIG_DIR: Path = CONFIGS_DIR / "managers"
WORKERS_CONFIG_DIR: Path = CONFIGS_DIR / "workers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ledger_path(campaign_id: str) -> Path:
    """Return the canonical JSONL ledger path for a campaign."""
    return LEDGERS_DIR / f"{campaign_id}_trials.jsonl"


def artifacts_dir(trial_id: str) -> Path:
    """Return the canonical artifacts directory for a trial."""
    return ARTIFACTS_DIR / trial_id


def node_config_path(node_name: str) -> Path:
    """Return the canonical YAML config path for a node."""
    return NODES_CONFIG_DIR / f"{node_name}.yaml"


def ensure_dirs(*dirs: Path) -> None:
    """Create *dirs* (and any missing parents) if they do not already exist."""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
