#!/usr/bin/env python3
"""Re-classify the refused proposals of a sealed G.4 pilot attempt under the hunk-header recount, offline.

R1-G5-DIFF-RECOUNT.  ENGINEERING_ONLY.  For each sealed G.4 attempt (``sealed_g4/<run_id>/<node>/attempt-N/``, or
its bare ``out.tar``) this script

1. verifies the attempt against the committed captures of the G.4 report (``--pair <attempt> <captures/<node>>``):
   the sha256 of ``out.tar`` against the captures' ``SHA256SUMS`` (and ``sealed_SHA256SUMS``); where the attempt
   directory has them, its ``SHA256SUMS`` and ``SEAL_RECEIPT.json`` byte-identical to the captures' copies and its
   ``job_record.json`` against the receipt's sha256;
2. reads from ``out.tar`` only ``out/pilot.json`` (the pilot input) and ``out/pilot/instance_*/campaign_ledger.jsonl``
   (manager views; no sealed test metric).  It reads no run record, run log, result channel, summary, label,
   evaluation data, G.1b or G.5 file;
3. rebuilds each step's incumbent from the pinned task code: the node kit of ``--harness-root`` (whose node spec
   must have the pilot's ``node_spec_sha256``) plus the ledger's kept diffs in order, and checks every rebuilt
   incumbent against the code-state hashes the ledger recorded (normalised and raw).  The sealed ``out.tar`` also
   holds each refused run's working copy (``instance_*/<run_id>/work/``, the runner's layout), but this script does
   not read it;
4. for every ``not_run_refused`` proposal, re-runs only the parse/apply step (``diff_support.apply_diff`` with
   ``write=False``, against the rebuilt incumbent and the duplicate registry rebuilt from the ledger) under the G.4
   rule (the manager's text) and under the new rule (the text after
   ``campaign_v2_diff.recount_hunk_headers``).  Nothing is trained, run or sent to a model.  For every other
   proposal it checks that the recount leaves the text unchanged (the G.4 rule accepted it).  The counts are per
   step against G.4's actual incumbents (``basis`` in the output), not a G.4b trajectory.

``--harness-root`` must be the checkout this script runs from, at the same commit (``git rev-parse HEAD``); the
script refuses otherwise and records both commits and whether each checkout is clean.

Output: JSON per node and in total (before, G.4-rule replay, after, by reason), on stdout and, with ``--out``, in a
new file (never overwritten).

Usage (from any directory; ``--harness-root`` is a checkout of the harness commit that carries this script)::

    python -B scripts/trackg_g4_reclassify.py --harness-root <harness> \\
        --pair <sealed attempt dir or out.tar> <captures dir of that node> [--pair ...] [--out <new json file>]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoresearch.trackg.campaign_v2_diff import DIFF_RECOUNT_RULE, recount_hunk_headers  # noqa: E402
from autoresearch.trackg.diff_support import EvaluatedStates, apply_diff, code_state_hashes  # noqa: E402
from autoresearch.trackg.node_template import load_trackg_node_spec  # noqa: E402
from autoresearch.trackg.runner import _COPY_IGNORE  # noqa: E402

SCHEMA = "trackg_g4_reclassify/v1"
PILOT_SCHEMA = "trackg_g4_pilot/v1"
LEDGER_SCHEMA = "trackg_v2_diff_campaign_ledger/v1"
LEDGER_MEMBER = re.compile(r"^out/pilot/(instance_[A-Za-z0-9_.-]+)/campaign_ledger\.jsonl$")
PILOT_MEMBER = "out/pilot.json"
BASIS = ("per refused step, against the incumbent G.4 actually had at that step (rebuilt from the kit and G.4's kept "
         "diffs) and the states G.4 had evaluated by then: a static, per-step re-classification, not a G.4b "
         "trajectory (a proposal applied under the new rule would have run and could have changed later incumbents)")


class ReclassifyStop(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.lstrip("*").strip()] = digest
    return out


def verify_attempt(attempt: Path, captures: Path) -> tuple[Path, dict[str, Any]]:
    """The out.tar of *attempt* (a sealed attempt directory or the tar itself), verified against *captures*."""
    attempt, captures = Path(attempt), Path(captures)
    for name in ("SHA256SUMS", "SEAL_RECEIPT.json"):
        if not (captures / name).is_file():
            raise ReclassifyStop(f"STOP: the captures directory {captures} has no {name}")
    receipt = json.loads((captures / "SEAL_RECEIPT.json").read_text())
    expected = _sums(captures / "SHA256SUMS").get("out.tar")
    if not expected:
        raise ReclassifyStop(f"STOP: {captures}/SHA256SUMS lists no out.tar")
    if (captures / "sealed_SHA256SUMS").is_file() and _sums(captures / "sealed_SHA256SUMS").get("out.tar") != expected:
        raise ReclassifyStop(f"STOP: {captures}: SHA256SUMS and sealed_SHA256SUMS disagree on out.tar")
    tar = attempt / "out.tar" if attempt.is_dir() else attempt
    if not tar.is_file():
        raise ReclassifyStop(f"STOP: no out.tar at {attempt}")
    actual = _sha_file(tar)
    if actual != expected:
        raise ReclassifyStop(f"STOP: {tar} has sha256 {actual}, the captures say {expected}")
    checked = {"out_tar_sha256": actual}
    if attempt.is_dir():
        sealed_sums = captures / ("sealed_SHA256SUMS" if (captures / "sealed_SHA256SUMS").is_file() else "SHA256SUMS")
        if (attempt / "SHA256SUMS").is_file():
            if (attempt / "SHA256SUMS").read_bytes() != sealed_sums.read_bytes():
                raise ReclassifyStop(f"STOP: {attempt}/SHA256SUMS differs from the committed {sealed_sums.name}")
            for name, digest in _sums(attempt / "SHA256SUMS").items():
                if (attempt / name).is_file() and _sha_file(attempt / name) != digest:
                    raise ReclassifyStop(f"STOP: {attempt}/{name} does not match its SHA256SUMS line")
            checked["sha256sums"] = "identical to the captures; listed files verified"
        if (attempt / "SEAL_RECEIPT.json").is_file():
            if (attempt / "SEAL_RECEIPT.json").read_bytes() != (captures / "SEAL_RECEIPT.json").read_bytes():
                raise ReclassifyStop(f"STOP: {attempt}/SEAL_RECEIPT.json differs from the committed receipt")
            checked["seal_receipt"] = "identical to the captures"
        if (attempt / "job_record.json").is_file():
            if _sha_file(attempt / "job_record.json") != receipt.get("job_record_json_sha256"):
                raise ReclassifyStop(f"STOP: {attempt}/job_record.json does not match the receipt's sha256")
            checked["job_record_sha256"] = receipt.get("job_record_json_sha256")
    checked.update(receipt_node=receipt.get("node"), receipt_run_id=receipt.get("run_id"),
                   receipt_attempt=receipt.get("attempt"), receipt_stop=receipt.get("stop"))
    return tar, checked


def read_members(tar_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    """``out/pilot.json`` and the instance ledgers, and nothing else, from the tar (read in memory)."""
    pilot, ledgers, names = None, {}, []
    with tarfile.open(tar_path) as tar:
        for member in tar:
            name = member.name[2:] if member.name.startswith("./") else member.name
            match = LEDGER_MEMBER.match(name)
            if name != PILOT_MEMBER and not match:
                continue
            if not member.isreg():
                raise ReclassifyStop(f"STOP: {name} in {tar_path} is not a regular file")
            text = tar.extractfile(member).read().decode("utf-8")
            names.append(name)
            if name == PILOT_MEMBER:
                pilot = json.loads(text)
            else:
                ledgers[match.group(1)] = [json.loads(x) for x in text.splitlines() if x.strip()]
    if pilot is None or not ledgers:
        raise ReclassifyStop(f"STOP: {tar_path} lacks out/pilot.json or campaign ledgers")
    return pilot, ledgers, sorted(names)


def _status(app) -> str:
    return app.status if app.status != "refused" else f"refused:{app.provenance['refusal']}"


def _register_runs(evaluated: EvaluatedStates, entry: dict[str, Any]) -> None:
    """What the runner and campaign registered as evaluated for this ledger line."""
    for run in entry.get("runs") or []:
        outcome = run.get("outcome") or {}
        if run.get("executed") and outcome.get("kind") != "infrastructure_failure" and run.get("code_state_sha256"):
            evaluated.register(run.get("run_id") or "?", code_state_sha256=run["code_state_sha256"],
                               normalised_diff_sha256=(run.get("diff") or {}).get("normalised_diff_sha256"),
                               incumbent_code_state_sha256=run.get("incumbent_code_state_sha256"))
    final = entry.get("final_outcome") or {}
    if final.get("reason") == "timeout_repeated" and (entry.get("runs") or []):
        run = entry["runs"][-1]
        evaluated.register(run.get("run_id") or "?", code_state_sha256=(final.get("detail") or {}).get(
            "code_state_sha256") or run.get("code_state_sha256"),
            normalised_diff_sha256=(run.get("diff") or {}).get("normalised_diff_sha256"),
            incumbent_code_state_sha256=run.get("incumbent_code_state_sha256"))


def reclassify_instance(name: str, entries: list[dict[str, Any]], spec, node_dir: Path, work: Path,
                        interpreter: str) -> dict[str, Any]:
    base = spec.to_base_node_spec()
    out: dict[str, Any] = {"instance": name, "steps": [], "accepted_by_g4_rule": 0, "accepted_text_unchanged": 0,
                           "incumbent_unverified": 0, "stop": None}
    if not entries or entries[0].get("step") != 0 or entries[0].get("decision") != "baseline":
        out["stop"] = "no baseline line"
        return out
    inc = work / f"{name}_inc_0000"
    shutil.copytree(node_dir, inc, ignore=_COPY_IGNORE, symlinks=True)
    base_run = (entries[0].get("runs") or [{}])[-1]
    state = code_state_hashes(inc, base)
    if [base_run.get("code_state_sha256"), base_run.get("code_state_raw_sha256")] != list(state):
        out["stop"] = "the node kit's code state differs from the ledger's baseline (wrong harness checkout?)"
        return out
    evaluated = EvaluatedStates()
    _register_runs(evaluated, entries[0])
    for entry in entries[1:]:
        if entry.get("schema") != LEDGER_SCHEMA:
            out["stop"] = f"step {entry.get('step')}: not a {LEDGER_SCHEMA} line"
            break
        decision = entry.get("decision")
        if str(decision).startswith("stop_"):
            out["stop"] = decision
            break
        proposal = entry.get("proposal")
        if isinstance(proposal, dict) and isinstance(proposal.get("diff"), str):
            raw = proposal["diff"]
            rec = recount_hunk_headers(raw)
            if decision == "not_run_refused":
                run = (entry.get("runs") or [{}])[-1]
                row = {"instance": name, "step": entry.get("step"),
                       "before": (run.get("diff") or {}).get("refusal") or "unrecorded",
                       "hunks_recounted": len(rec.recounted), "left_to_strict_parser": rec.left}
                if [run.get("incumbent_code_state_sha256"), run.get("incumbent_code_state_raw_sha256")] != list(state):
                    row["old"] = row["new"] = "incumbent_unverified"
                    out["incumbent_unverified"] += 1
                else:
                    kw = dict(evaluated=copy.deepcopy(evaluated), write=False,
                              allow_new_files=spec.allow_new_files, interpreter=interpreter)
                    row["old"] = _status(apply_diff(inc, raw, base, **kw))
                    row["new"] = _status(apply_diff(inc, rec.text, base, **kw))
                out["steps"].append(row)
            else:
                out["accepted_by_g4_rule"] += 1
                out["accepted_text_unchanged"] += not rec.changed
            if decision == "kept":
                run = (entry.get("runs") or [{}])[-1]
                nxt = work / f"{name}_inc_{entry.get('step'):04d}"
                shutil.copytree(inc, nxt, ignore=_COPY_IGNORE, symlinks=True)
                app = apply_diff(nxt, rec.text, base, evaluated=None, allow_new_files=spec.allow_new_files,
                                 interpreter=interpreter)
                state = code_state_hashes(nxt, base)
                if not app.applied or [run.get("code_state_sha256"), run.get("code_state_raw_sha256")] != list(state):
                    out["stop"] = f"step {entry.get('step')}: the kept diff does not rebuild the ledger's code state"
                    break
                inc = nxt
        _register_runs(evaluated, entry)
    return out


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, dict[str, int]] = {}
    for r in rows:
        by_reason.setdefault(r["before"], {})
        by_reason[r["before"]][r["new"]] = by_reason[r["before"]].get(r["new"], 0) + 1
    mismatches = [{k: r[k] for k in ("instance", "step", "before", "old")} for r in rows
                  if r["old"] != f"refused:{r['before']}"]
    left = _count([r for r in rows if r["left_to_strict_parser"]], "left_to_strict_parser")
    return {"refused_before": _count(rows, "before"),
            "g4_rule_replay": {"matches_ledger": len(rows) - len(mismatches), "mismatches": mismatches},
            "new_rule": _count(rows, "new"),
            "new_rule_runnable": sum(1 for r in rows if r["new"] == "applied"),
            "by_reason": {k: dict(sorted(v.items())) for k, v in sorted(by_reason.items())},
            "with_recounted_headers": sum(1 for r in rows if r["hunks_recounted"]),
            "left_to_strict_parser": left}


def reclassify_node(attempt: Path, captures: Path, harness_root: Path) -> dict[str, Any]:
    tar, verification = verify_attempt(attempt, captures)
    pilot, ledgers, members = read_members(tar)
    if pilot.get("schema") != PILOT_SCHEMA or pilot.get("label") != "ENGINEERING_ONLY":
        raise ReclassifyStop(f"STOP: {tar}: out/pilot.json is not an ENGINEERING_ONLY {PILOT_SCHEMA} file")
    if pilot.get("node") != verification["receipt_node"]:
        raise ReclassifyStop(f"STOP: {tar}: the pilot names {pilot.get('node')}, the receipt "
                             f"{verification['receipt_node']}")
    root = Path(harness_root).resolve()
    spec_path, node_dir = (root / pilot["spec_path"]).resolve(), (root / pilot["node_dir"]).resolve()
    if root not in spec_path.parents or root not in node_dir.parents or not node_dir.is_dir():
        raise ReclassifyStop("STOP: the pilot's spec_path/node_dir are not inside the harness checkout")
    spec = load_trackg_node_spec(spec_path)
    if spec.name != pilot["node"] or spec.raw_sha256 != pilot.get("node_spec_sha256"):
        raise ReclassifyStop(f"STOP: {spec_path} is not the node spec the pilot pinned (sha256 differs)")
    interpreter = pilot.get("interpreter") if Path(str(pilot.get("interpreter"))).is_file() else "python3"
    instances = []
    with tempfile.TemporaryDirectory(prefix="g4_reclassify_") as tmp:
        for name in sorted(ledgers):
            instances.append(reclassify_instance(name, ledgers[name], spec, node_dir, Path(tmp), interpreter))
    rows = [r for inst in instances for r in inst["steps"]]
    decisions: dict[str, int] = {}
    for entries in ledgers.values():
        for e in entries[1:]:
            decisions[e.get("decision")] = decisions.get(e.get("decision"), 0) + 1
    return {"node": pilot["node"], "verification": verification, "members_read": members,
            "shadow_check_interpreter": interpreter, "proposals": sum(decisions.values()),
            "ledger_decisions": dict(sorted(decisions.items())),
            "accepted_by_g4_rule": sum(i["accepted_by_g4_rule"] for i in instances),
            "accepted_text_unchanged": sum(i["accepted_text_unchanged"] for i in instances),
            "incumbent_unverified": sum(i["incumbent_unverified"] for i in instances),
            "instance_stops": {i["instance"]: i["stop"] for i in instances if i["stop"]},
            **summarise_rows(rows), "steps": rows}


def _head(path: Path) -> str | None:
    """HEAD of the git checkout that holds *path*, or None."""
    res = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else None


def _dirty(path: Path) -> bool | None:
    res = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True)
    return bool(res.stdout.strip()) if res.returncode == 0 else None


def check_checkouts(harness_root: Path) -> dict[str, Any]:
    """The harness root must be a checkout at the commit this script runs from (fail closed: an unknown HEAD on
    either side is a mismatch).  Both commits and whether each checkout is clean go into the output."""
    script_dir = Path(__file__).resolve().parent
    script_commit, root_commit = _head(script_dir), _head(Path(harness_root))
    if script_commit is None or root_commit is None or script_commit != root_commit:
        raise ReclassifyStop(f"STOP: --harness-root HEAD {root_commit} is not the commit this script runs from "
                             f"({script_commit}); run the script of the harness checkout given as --harness-root")
    return {"script_commit": script_commit, "harness_root_commit": root_commit,
            "script_checkout_dirty": _dirty(script_dir), "harness_root_dirty": _dirty(Path(harness_root))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--harness-root", required=True, type=Path)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ATTEMPT", "CAPTURES"), type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.out is not None and a.out.exists():
        print(f"STOP: {a.out} exists; never overwritten", file=sys.stderr)
        return 2
    try:
        checkouts = check_checkouts(a.harness_root)
        nodes = [reclassify_node(attempt, caps, a.harness_root) for attempt, caps in a.pair]
    except (ReclassifyStop, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows = [r for n in nodes for r in n["steps"]]
    result = {"schema": SCHEMA, "label": "ENGINEERING_ONLY", "rule": DIFF_RECOUNT_RULE, **checkouts,
              "basis": BASIS, "nodes": nodes,
              "total": {"proposals": sum(n["proposals"] for n in nodes),
                        "accepted_by_g4_rule": sum(n["accepted_by_g4_rule"] for n in nodes),
                        "accepted_text_unchanged": sum(n["accepted_text_unchanged"] for n in nodes),
                        "incumbent_unverified": sum(n["incumbent_unverified"] for n in nodes),
                        **summarise_rows(rows)}}
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if a.out is not None:
        with open(a.out, "x", encoding="utf-8") as fh:
            fh.write(text)
    sys.stdout.write(text)
    stops = [n["instance_stops"] for n in nodes if n["instance_stops"]]
    return 3 if stops or result["total"]["incumbent_unverified"] or result["total"]["g4_rule_replay"]["mismatches"] \
        else 0


if __name__ == "__main__":
    sys.exit(main())
