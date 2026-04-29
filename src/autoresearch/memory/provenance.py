from __future__ import annotations

import hashlib

from autoresearch.memory.schemas import TrialProvenance


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return f"{prefix}-{digest.hexdigest()[:16]}"


def build_trial_provenance(trial_id: str) -> TrialProvenance:
    return TrialProvenance(
        proposal_id=stable_id("proposal", trial_id),
        patch_id=stable_id("patch", trial_id),
        run_id=stable_id("run", trial_id),
        metric_id=stable_id("metric", trial_id),
        decision_id=stable_id("decision", trial_id),
    )

