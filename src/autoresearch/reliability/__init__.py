"""Reliability module — governed fault injection, verification, and recovery.

Extends the autoresearch harness with typed event contracts, sealed fault
injection, state-difference observation, independent adjudication, benchmark
split resolution, calibrated warning, cost-sensitive control, and immutable
training-view construction.

Package layout follows IMPLEMENTATION_PLAN.md.
"""

from . import schema

__all__ = ["schema"]
