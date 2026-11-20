"""G.4 T1 pilot: v2-diff campaigns on ported nodes with the LLM manager, and the pilot measurements.

R1-TRACKG-G3-EARLY (20 Nov 2026): the G.4 pilot runs on three T1 nodes (``mlab_cifar10``, ``mlgym_fmnist`` and
``autoresearch_fixedtokens``, or ``mlgym_breakout_minatar`` while autoresearch is not ready) on NEMO2 H200
(R1-T1-HOST-H200).  The paper repository's job (``sweep2/g4/``) starts the pinned llama-server (main proposer
``gemma4_31b``, thinking off) and writes the pilot file; this module runs, for one node, a few independent
campaign instances of :func:`~autoresearch.trackg.campaign_v2_diff.run_v2_diff_campaign`
(``manager_proposal/v2-diff``), with:

* the unisolated-run gate on every executed run (:func:`autoresearch.trackg.noise_floor.check_effective`);
* ``cache_prompt: false`` in every request (each proposal is a fresh evaluation of its prompt);
* the timeout at 3 x the census run time (a timeout is the proposal's failure);
* a transport wrapper that records each call's wall time, so throughput can be split into LLM and run time.

:func:`summarise` reads the campaign ledgers and reports the quantities the pilot exists to measure:
throughput, diff liveness, the shares of invalid, refused, inert, duplicate, divergent and kept proposals, and
prompt and completion lengths.  ENGINEERING_ONLY: the pilot's proposals are not labels; no sealed test metric is
read (the ledgers hold manager views only).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from autoresearch.trackg import gpu_identity
from autoresearch.trackg.campaign_v2_diff import V2DiffCampaignError, run_v2_diff_campaign
from autoresearch.trackg.manager_v2_diff import Chat, ManagerTransportStop, V2DiffManager, openai_compatible_chat
from autoresearch.trackg.noise_floor import NoiseFloorStop, check_effective, check_node_pins
from autoresearch.trackg.node_template import load_trackg_node_spec
from autoresearch.trackg.runner import IntegrityError, ResourceLimits, RunnerError

PILOT_SCHEMA = "trackg_g4_pilot/v1"
PILOT_RECORD_SCHEMA = "trackg_g4_pilot_record/v1"
SUMMARY_SCHEMA = "trackg_g4_pilot_summary/v1"
PILOT_NODES = ("mlab_cifar10", "mlgym_fmnist", "autoresearch_fixedtokens", "mlgym_breakout_minatar")
RESERVED = (range(1001, 1031), range(2001, 2031), range(3001, 3031), range(7001, 7011), range(9001, 10000))
DECISIONS = ("kept", "discarded", "discarded_divergent", "invalid_proposal", "not_run_refused", "not_run_inert",
             "not_run_duplicate", "infrastructure_failure_final")


class PilotStop(RuntimeError):
    pass


def validate_pilot(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != PILOT_SCHEMA or data.get("label") != "ENGINEERING_ONLY":
        raise PilotStop(f"STOP: pilot file must be {PILOT_SCHEMA}, ENGINEERING_ONLY")
    if data.get("node") not in PILOT_NODES:
        raise PilotStop(f"STOP: {data.get('node')!r} is not a G.4 pilot node {PILOT_NODES}")
    if data.get("proposal_schema") != "manager_proposal/v2-diff":
        raise PilotStop("STOP: the pilot uses manager_proposal/v2-diff")
    if data.get("extra_payload") != {"cache_prompt": False}:
        raise PilotStop("STOP: every request carries cache_prompt: false and nothing else extra")
    budget = data.get("budget")
    if type(budget) is not int or not 1 <= budget <= 50:
        raise PilotStop("STOP: budget (manager calls per instance) must be an integer in [1, 50]")
    instances = data.get("instances")
    if not isinstance(instances, list) or not 1 <= len(instances) <= 10:
        raise PilotStop("STOP: 1-10 campaign instances")
    seen = set()
    for inst in instances:
        if set(inst) != {"instance", "train_seed", "split_seed", "manager_seed"}:
            raise PilotStop("STOP: an instance has instance, train_seed, split_seed, manager_seed")
        for key in ("train_seed", "split_seed", "manager_seed"):
            if type(inst[key]) is not int or any(inst[key] in r for r in RESERVED):
                raise PilotStop(f"STOP: {key} {inst[key]!r} is reserved (G.1b, audit or exam seeds) or not an integer")
        if inst["instance"] in seen:
            raise PilotStop("STOP: instance ids must be unique")
        seen.add(inst["instance"])
    seconds = data.get("measured_run_seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not 0 < seconds <= 300:
        raise PilotStop("STOP: measured_run_seconds (census) must be in (0, 300]")
    for key in ("node_spec_sha256", "committed_pin_sha256"):
        if not isinstance(data.get(key), str) or len(data[key]) != 64:
            raise PilotStop(f"STOP: {key} must be a sha256")
    dm = data.get("data_manifest_sha256", "missing")
    if dm is not None and (not isinstance(dm, str) or len(dm) != 64):
        raise PilotStop("STOP: data_manifest_sha256 must be a sha256, or null for a node without data")
    for key in ("spec_path", "node_dir", "base_url", "model", "cgroup_parent", "interpreter", "memory_limit"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise PilotStop(f"STOP: {key} must be a non-empty string")
    if data.get("isolation_waivers", {}) != {}:
        raise PilotStop("STOP: the pilot carries no isolation waiver")
    return data


def timed_chat(inner: Chat, clock: Callable[[], float] = time.monotonic) -> Chat:
    """Record each call's wall time in its usage (``wall_seconds``), so the ledger carries it."""
    def chat(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        start = clock()
        content, usage = inner(payload)
        usage = dict(usage)
        usage["wall_seconds"] = round(clock() - start, 3)
        return content, usage
    return chat


def run_pilot(data: dict[str, Any], harness_root: Path, out_dir: Path, *, gpu_uuids: tuple[str, ...],
              cpu_list: tuple[int, ...], chat: Chat | None = None, campaign=run_v2_diff_campaign,
              gate=check_effective) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {"schema": PILOT_RECORD_SCHEMA, "label": "ENGINEERING_ONLY", "node": data["node"],
                              "instances": [], "stop": None, "started_epoch": time.time()}
    try:
        spec = load_trackg_node_spec(harness_root / data["spec_path"])
        node_dir = (harness_root / data["node_dir"]).resolve()
        if spec.name != data["node"] or spec.raw_sha256 != data["node_spec_sha256"] or spec.isolation is None:
            raise PilotStop("STOP: node spec name/sha256 differ from the pilot pins, or the node is not isolated")
        check_node_pins(spec, node_dir, data["committed_pin_sha256"], data["data_manifest_sha256"])
        limits = ResourceLimits(gpu_uuids=gpu_uuids, cpu_list=cpu_list, memory_bytes=None, memory_method="external",
                                external_limit=data["memory_limit"],
                                measured_run_seconds=float(data["measured_run_seconds"]), timeout_factor=3.0,
                                cgroup_parent=data["cgroup_parent"], isolation_waivers={})
        transport = timed_chat(chat or openai_compatible_chat(data["base_url"]))
        for inst in data["instances"]:
            manager = V2DiffManager(chat=transport, model=data["model"], seed=inst["manager_seed"],
                                    max_tokens=int(data.get("max_tokens", 4096)), extra_payload={"cache_prompt": False})
            started = time.time()
            entry = {"instance": inst["instance"], **inst}
            record["instances"].append(entry)
            campaign(spec, node_dir, out_dir / f"instance_{inst['instance']}", manager, limits, data["budget"],
                     enable_v2_diff=True, train_seed=inst["train_seed"], split_seed=inst["split_seed"],
                     interpreter=data["interpreter"], record_gate=gate)
            entry["wall_seconds"] = round(time.time() - started, 1)
    except (PilotStop, NoiseFloorStop, V2DiffCampaignError, ManagerTransportStop, IntegrityError, RunnerError,
            gpu_identity.GpuIdentityError, OSError, ValueError) as exc:
        record["stop"] = f"{type(exc).__name__}: {exc}"
    record["finished_epoch"] = time.time()
    (out_dir / "pilot_record.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


def _q(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarise(out_dir: Path, *, census_divergence_rate: float | None = None) -> dict[str, Any]:
    """Pilot measurements from the campaign ledgers of one node (all instances)."""
    out_dir = Path(out_dir)
    entries = []
    for ledger in sorted(out_dir.glob("instance_*/campaign_ledger.jsonl")):
        entries += [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    steps = [e for e in entries if e.get("step", 0) >= 1 and e.get("decision") in DECISIONS]
    stops = [e["decision"] for e in entries if str(e.get("decision", "")).startswith("stop_")
             or e.get("decision") == "baseline_failed"]
    counts = {d: sum(1 for e in steps if e["decision"] == d) for d in DECISIONS}
    n = len(steps)
    valid = n - counts["invalid_proposal"]
    ran = counts["kept"] + counts["discarded"] + counts["discarded_divergent"]
    calls = [e["manager_call"] for e in steps if e.get("manager_call")]
    prompt_tokens = [float(c["usage"]["prompt_tokens"]) for c in calls if (c.get("usage") or {}).get("prompt_tokens")]
    completion = [float(c["usage"]["completion_tokens"]) for c in calls
                  if (c.get("usage") or {}).get("completion_tokens")]
    llm_seconds = [float(c["usage"]["wall_seconds"]) for c in calls if (c.get("usage") or {}).get("wall_seconds")]
    run_seconds = [float(r["elapsed_seconds_info_only"]) for e in steps for r in e.get("runs", [])
                   if r.get("executed") and r.get("elapsed_seconds_info_only") is not None]
    timeouts = sum(1 for e in steps if (e.get("final_outcome") or {}).get("reason") in ("timeout", "timeout_repeated"))
    rate = (lambda k, d: round(k / d, 4) if d else None)
    record = json.loads((out_dir / "pilot_record.json").read_text()) if (out_dir / "pilot_record.json").is_file() else {}
    wall = sum(float(i.get("wall_seconds") or 0) for i in record.get("instances", []))
    summary = {
        "schema": SUMMARY_SCHEMA, "label": "ENGINEERING_ONLY", "node": record.get("node"), "stops": stops,
        "pilot_stop": record.get("stop"), "proposals": n, "decisions": counts,
        "rates": {"invalid": rate(counts["invalid_proposal"], n), "refused": rate(counts["not_run_refused"], n),
                  "inert": rate(counts["not_run_inert"], n), "duplicate": rate(counts["not_run_duplicate"], n),
                  "divergent_of_ran": rate(counts["discarded_divergent"], ran), "kept_of_ran": rate(counts["kept"], ran),
                  "infrastructure_final": rate(counts["infrastructure_failure_final"], n),
                  "timeouts_of_ran": rate(timeouts, ran)},
        "diff_liveness": {"valid_of_proposals": rate(valid, n), "ran_of_valid": rate(ran, valid),
                          "ok_of_proposals": rate(counts["kept"] + counts["discarded"], n)},
        "prompt_tokens": {"median": _q(prompt_tokens, 0.5), "p95": _q(prompt_tokens, 0.95),
                          "max": max(prompt_tokens) if prompt_tokens else None, "n": len(prompt_tokens)},
        "completion_tokens": {"median": _q(completion, 0.5), "p95": _q(completion, 0.95),
                              "max": max(completion) if completion else None},
        "seconds": {"llm_median": _q(llm_seconds, 0.5), "llm_total": round(sum(llm_seconds), 1),
                    "run_median": _q(run_seconds, 0.5), "run_total": round(sum(run_seconds), 1),
                    "campaign_wall_total": round(wall, 1)},
        "throughput_proposals_per_hour": round(n / (wall / 3600.0), 2) if wall > 0 else None,
    }
    if census_divergence_rate is not None and summary["rates"]["divergent_of_ran"] is not None:
        summary["divergence_vs_census"] = {"census": census_divergence_rate,
                                           "ratio": round(summary["rates"]["divergent_of_ran"] / census_divergence_rate, 3)
                                           if census_divergence_rate > 0 else None}
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run (or summarise) one node of the G.4 pilot (ENGINEERING_ONLY).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--pilot", required=True, type=Path)
    r.add_argument("--harness-root", required=True, type=Path); r.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("summarise"); s.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "summarise":
        print(json.dumps(summarise(a.out), indent=1))
        return 0
    try:
        data = validate_pilot(json.loads(a.pilot.read_text()))
        import os
        gpus, _ = gpu_identity.resolve_from_job(os.environ)
    except (PilotStop, gpu_identity.GpuIdentityError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    import os
    rec = run_pilot(data, a.harness_root, a.out, gpu_uuids=gpus, cpu_list=tuple(sorted(os.sched_getaffinity(0))))
    summary = summarise(a.out)
    print(json.dumps({"node": data["node"], "stop": rec["stop"], "proposals": summary["proposals"]}))
    return 0 if rec["stop"] is None else 3


if __name__ == "__main__":
    sys.exit(main())
