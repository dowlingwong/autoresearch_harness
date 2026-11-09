"""Stub-manager smoke campaign for a ported node (G.1 step 4; ENGINEERING_ONLY).

The "manager" is a fixed list of ``manager_proposal/v2-diff`` proposals (or
plain diffs).  The loop runs the baseline, then each diff against the current
incumbent through :class:`WorkingCopyRunner` (scope, inert and duplicate
checks; limits; outcome class), keeps a diff only if its run is ``ok`` and
strictly better on the validation metric, and appends one ledger line per
step to ``<runs_root>/smoke_ledger.jsonl``.

It is a smoke harness for ports, not the label-sweep or closed-loop code: no
noise-aware keep rule, no controller, no LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from autoresearch.trackg.diff_support import EvaluatedStates
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import OK
from autoresearch.trackg.proposal_v2_diff import DiffProposal
from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
from autoresearch.trackg.selection import manager_view

SMOKE_LEDGER_SCHEMA = "trackg_smoke_ledger/v1"


def run_smoke_campaign(
    spec: TrackGNodeSpec,
    node_dir: str | Path,
    runs_root: str | Path,
    proposals: Sequence[DiffProposal | str],
    limits: ResourceLimits,
    *,
    train_seed: int | None = None,
    split_seed: int | None = None,
) -> list[dict[str, Any]]:
    runs = Path(runs_root).resolve()
    runs.mkdir(parents=True, exist_ok=False)
    ledger = runs / "smoke_ledger.jsonl"
    ledger.touch(exist_ok=False)
    evaluated = EvaluatedStates()
    incumbent_dir = Path(node_dir).resolve()
    records: list[dict[str, Any]] = []

    def append(entry: dict[str, Any]) -> None:
        records.append(entry)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    baseline = WorkingCopyRunner(spec, incumbent_dir, runs).run(
        "s000_baseline", limits, train_seed=train_seed, split_seed=split_seed, evaluated=evaluated)
    best = baseline.outcome.value if baseline.outcome and baseline.outcome.kind == OK else None
    if best is not None:
        incumbent_dir = baseline.work_dir
    append({"schema": SMOKE_LEDGER_SCHEMA, "step": 0, "run_id": "s000_baseline", "decision": "baseline",
            "record": manager_view(baseline.record)})

    for index, proposal in enumerate(proposals, start=1):
        diff_text = proposal.diff if isinstance(proposal, DiffProposal) else str(proposal)
        run_id = f"s{index:03d}"
        result = WorkingCopyRunner(spec, incumbent_dir, runs).run(
            run_id, limits, train_seed=train_seed, split_seed=split_seed, diff_text=diff_text, evaluated=evaluated)
        if result.outcome is None:
            decision = f"not_run_{result.record['diff']['status']}"
        elif result.outcome.kind != OK:
            decision = f"discarded_{result.outcome.kind}"
        else:
            value = result.outcome.value
            better = best is None or (value > best if spec.metric_direction == "maximize" else value < best)
            decision = "kept" if better else "discarded"
            if better:
                best, incumbent_dir = value, result.work_dir
        entry = {"schema": SMOKE_LEDGER_SCHEMA, "step": index, "run_id": run_id, "decision": decision,
                 "incumbent_after": str(incumbent_dir), "record": manager_view(result.record)}
        if isinstance(proposal, DiffProposal):
            entry["proposal"] = {"schema_version": proposal.schema_version, "summary": proposal.summary,
                                 "rationale": proposal.rationale, "objective": proposal.objective}
        append(entry)
    return records
