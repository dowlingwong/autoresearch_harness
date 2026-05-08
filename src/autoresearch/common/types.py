"""Shared type aliases for the autoresearch package.

These types are defined here as the canonical source.  The per-module
definitions in nodes/spec.py and evaluation/metrics.py remain for backward
compatibility but new code should import from here.
"""
from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Metric direction
# ---------------------------------------------------------------------------

MetricDirection = Literal["maximize", "minimize"]

# ---------------------------------------------------------------------------
# Memory mode strings (canonical set)
# ---------------------------------------------------------------------------

MemoryModeStr = Literal[
    "none",
    "append_only_summary",
    "append_only_summary_with_rationale",
]

# ---------------------------------------------------------------------------
# Manager mode strings (canonical set)
# ---------------------------------------------------------------------------

ManagerModeStr = Literal[
    "baseline_manager",
    "prompt_manager",
    "langgraph_manager",
]

# ---------------------------------------------------------------------------
# Worker mode strings (canonical set)
# ---------------------------------------------------------------------------

WorkerModeStr = Literal[
    "dry_run_worker",
    "claw_style_worker",
    "local_worker",
    "fake_worker",
]
