"""A Track G campaign driven by the v2-diff LLM manager, behind an explicit flag (M10; AI-assisted draft).

``run_v2_diff_campaign(..., enable_v2_diff=True)`` is the campaign path for ``manager_proposal/v2-diff``.
Without ``enable_v2_diff=True`` it refuses to run: nothing in the T0 path calls it, and the v1 campaign
code (``autoresearch.control_plane``) is untouched, so the T0 ``manager_proposal/v1`` path stays
byte-identical (``tests/trackg/test_trackg_t0_identity.py``).

Per step (``budget`` manager calls, each charged):

1. the manager proposes one diff against the incumbent; a reply that does not parse is an
   ``invalid_proposal`` (charged, nothing runs);
2. the diff runs in a fresh working copy through :class:`WorkingCopyRunner` (scope, inert and duplicate
   checks, limits, outcome class). A diff that is refused, inert or a duplicate is not run;
3. outcome handling:
   * ``ok``: kept if strictly better than the incumbent on the validation metric (the next incumbent is
     the run's pre-run code snapshot), else discarded;
   * ``divergent``: discarded;
   * ``infrastructure_failure`` with reason ``timeout``: the same code state is rerun. **Repeated-timeout
     cap** (author's answer to M10 §5 question 8, option (a)): after ``timeout_cap`` (2) timeouts of the
     same code state, the proposal is ``divergent`` with reason ``timeout_repeated``, the code state is
     registered as evaluated, and it is never rerun;
   * any other ``infrastructure_failure`` (launch error, kill, an infrastructure log pattern such as Slurm's
     ``DUE TO TIME LIMIT``): rerun up to ``max_infra_reruns`` times, then recorded as
     ``infrastructure_failure_final`` (not registered as evaluated, so it can be proposed and run again later,
     bounded by the manager budget). These reruns never count toward the timeout cap.
   Reruns of the same proposal do not consume manager calls. The timeout count is per code state over the
   whole campaign: a code state that timed out once in an earlier step reaches the cap on its next timeout.
   A diff seen before against another incumbent reaches a new code state and is run
   (``seen_diff_other_incumbent`` is recorded; M10 §5 question 7, option (a)).

The baseline gets the same rerun and cap handling; if it still has no ``ok`` run, the campaign writes a
``baseline_failed`` ledger line and STOPs (nothing is compared against a missing incumbent value). A transport
failure of the manager (``ManagerTransportStop``) writes a ``stop_transport`` line and STOPs. The campaign has no
resume: an HTCondor eviction of the job that runs it ends the campaign, and a restart refuses the existing
``runs_root`` (Unit 2's resume-safe units are not wired here).

Every step appends one line to ``<runs_root>/campaign_ledger.jsonl`` (never overwritten): the proposal
fields, the manager call record (prompt hash, seed, raw reply and its hash, usage), every run record of
the step (``manager_view``: no sealed test metric), the decision and the incumbent after the step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoresearch.trackg.diff_support import EvaluatedStates
from autoresearch.trackg.manager_v2_diff import ManagerTransportStop, V2DiffManager
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK, RunOutcome
from autoresearch.trackg.runner import ResourceLimits, WorkingCopyRunner
from autoresearch.trackg.selection import manager_view

LEDGER_SCHEMA = "trackg_v2_diff_campaign_ledger/v1"
TIMEOUT_REASON = "timeout"
TIMEOUT_REPEATED = "timeout_repeated"


class V2DiffCampaignError(RuntimeError):
    pass


def run_v2_diff_campaign(
    spec: TrackGNodeSpec,
    node_dir: str | Path,
    runs_root: str | Path,
    manager: V2DiffManager,
    limits: ResourceLimits,
    budget: int,
    *,
    enable_v2_diff: bool = False,
    timeout_cap: int = 2,
    max_infra_reruns: int = 2,
    train_seed: int | None = None,
    split_seed: int | None = None,
    interpreter: str = "python3",
) -> list[dict[str, Any]]:
    if enable_v2_diff is not True:
        raise V2DiffCampaignError("the v2-diff campaign path is off: pass enable_v2_diff=True explicitly")
    if type(budget) is not int or budget < 1:
        raise V2DiffCampaignError("budget must be a positive integer")
    if type(timeout_cap) is not int or timeout_cap < 1 or type(max_infra_reruns) is not int or max_infra_reruns < 0:
        raise V2DiffCampaignError("timeout_cap must be >= 1 and max_infra_reruns >= 0")
    runs = Path(runs_root).resolve()
    runs.mkdir(parents=True, exist_ok=False)
    ledger = runs / "campaign_ledger.jsonl"
    ledger.touch(exist_ok=False)
    evaluated = EvaluatedStates()
    timeouts: dict[str, int] = {}
    incumbent_dir = Path(node_dir).resolve()
    history: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    def append(entry: dict[str, Any]) -> None:
        entries.append(entry)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            handle.flush()

    def better(value: float, best: float | None) -> bool:
        return best is None or (value > best if spec.metric_direction == "maximize" else value < best)

    def run_with_retries(step_tag: str, diff_text: str | None, entry: dict[str, Any]) -> Any:
        """Run (and rerun) one code state; returns the final outcome or None if the diff was not run.

        Sets entry["decision_hint"]: "not_run_<status>", "timeout_repeated", "infrastructure_failure_final" or "ran"."""
        attempt, reruns_other = 0, 0
        while True:
            run_id = step_tag + (f"_r{attempt}" if attempt else "")
            result = WorkingCopyRunner(spec, incumbent_dir, runs).run(
                run_id, limits, train_seed=train_seed, split_seed=split_seed, diff_text=diff_text,
                evaluated=evaluated, interpreter=interpreter)
            entry["runs"].append(manager_view(result.record))
            outcome = result.outcome
            if outcome is None:
                entry["decision_hint"] = f"not_run_{result.record['diff']['status']}"
                return None, result
            if outcome.kind == INFRASTRUCTURE_FAILURE and outcome.reason == TIMEOUT_REASON:
                state = result.record["code_state_sha256"]
                timeouts[state] = timeouts.get(state, 0) + 1
                if timeouts[state] >= timeout_cap:
                    final = RunOutcome.divergent(TIMEOUT_REPEATED, timeouts=timeouts[state], code_state_sha256=state)
                    diff = result.record.get("diff") or {}
                    evaluated.register(run_id, code_state_sha256=state,
                                       normalised_diff_sha256=diff.get("normalised_diff_sha256"),
                                       incumbent_code_state_sha256=result.record.get("incumbent_code_state_sha256"))
                    entry["decision_hint"] = "timeout_repeated"
                    return final, result
                attempt += 1
                continue
            if outcome.kind == INFRASTRUCTURE_FAILURE:
                if reruns_other < max_infra_reruns:
                    reruns_other += 1
                    attempt += 1
                    continue
                entry["decision_hint"] = "infrastructure_failure_final"
                return outcome, result
            entry["decision_hint"] = "ran"
            return outcome, result

    base_entry: dict[str, Any] = {"schema": LEDGER_SCHEMA, "step": 0, "runs": []}
    base_outcome, base_result = run_with_retries("s000_baseline", None, base_entry)
    base_entry.pop("decision_hint", None)
    if base_outcome is None or base_outcome.kind != OK:
        base_entry.update(decision="baseline_failed", final_outcome=base_outcome.to_dict() if base_outcome else None,
                          incumbent_after=str(incumbent_dir), best_after=None)
        append(base_entry)
        raise V2DiffCampaignError("STOP: the baseline did not give an ok run after reruns; nothing is compared "
                                  "against a missing incumbent value")
    best, incumbent_dir = base_outcome.value, base_result.code_dir
    base_entry.update(decision="baseline", final_outcome=base_outcome.to_dict(),
                      incumbent_after=str(incumbent_dir), best_after=best)
    append(base_entry)

    for step in range(1, budget + 1):
        try:
            proposal, call = manager.propose(spec, incumbent_dir, history, step)
        except ManagerTransportStop as stop:
            append({"schema": LEDGER_SCHEMA, "step": step, "decision": "stop_transport", "error": str(stop),
                    "manager_call": getattr(stop, "call", None), "runs": [],
                    "incumbent_after": str(incumbent_dir), "best_after": best})
            raise
        entry: dict[str, Any] = {"schema": LEDGER_SCHEMA, "step": step, "manager_call": call, "runs": []}
        if proposal is None:
            entry["decision"] = "invalid_proposal"
            summary = None
        else:
            summary = proposal.summary
            entry["proposal"] = {"schema_version": proposal.schema_version, "rationale": proposal.rationale,
                                 "summary": proposal.summary, "objective": proposal.objective,
                                 "diff": proposal.diff}
            outcome, result = run_with_retries(f"s{step:03d}", proposal.diff, entry)
            hint = entry.pop("decision_hint")
            if outcome is not None:
                entry["final_outcome"] = outcome.to_dict()
            if hint.startswith("not_run_") or hint == "infrastructure_failure_final":
                entry["decision"] = hint
            elif hint == "timeout_repeated" or outcome.kind == DIVERGENT:
                entry["decision"] = "discarded_divergent"
            elif better(outcome.value, best):
                best, incumbent_dir = outcome.value, result.code_dir
                entry["decision"] = "kept"
            else:
                entry["decision"] = "discarded"
        entry["incumbent_after"], entry["best_after"] = str(incumbent_dir), best
        metric = (entry.get("final_outcome") or {}).get("value")
        history.append({"step": step, "summary": summary, "decision": entry["decision"], "metric": metric})
        append(entry)
    return entries
