#!/usr/bin/env python3
"""Re-execute kept LocalWorker trials from recorded patch artifacts."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.memory.schemas import TrialDecision
from autoresearch.nodes.registry import load_registered_node


def _ledger_path(campaign_id: str) -> Path:
    return ROOT / "experiments" / "ledgers" / f"{campaign_id}_trials.jsonl"


def _default_node_root(node_name: str) -> Path:
    exact = ROOT / "nodes" / node_name
    if exact.exists():
        return exact
    matches = [path for path in (ROOT / "nodes").glob("*") if path.name.lower() == node_name.lower()]
    return matches[0] if matches else exact


def _restore_baseline(node_copy: Path, node_source: Path, editable_paths: tuple[str, ...]) -> None:
    for rel_path in editable_paths:
        baseline = node_source / ".autoresearch_baseline" / rel_path
        target = node_copy / rel_path
        if baseline.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(baseline.read_bytes())


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in re.finditer(
        r"^([A-Za-z_][A-Za-z0-9_]*)(?::\s+|=)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$",
        text,
        flags=re.MULTILINE,
    ):
        key, value = match.groups()
        metrics[key.lower()] = float(value)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--node-root", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--out",
        default=None,
        help="CSV path; defaults to plan/metric_validation/reexecution_<campaign>.csv",
    )
    args = parser.parse_args()

    node_spec = load_registered_node(args.node, repo_root=ROOT)
    node_source = Path(args.node_root).resolve() if args.node_root else _default_node_root(args.node)
    output = Path(args.out or ROOT / "plan" / "metric_validation" / f"reexecution_{args.campaign}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    all_records = TrialAppendStore(_ledger_path(args.campaign)).read_all()
    records = [
        record for record in all_records
        if record.decision == TrialDecision.KEPT
    ][: args.limit]
    rows: list[dict[str, str]] = []
    for record in records:
        cumulative_kept = [
            prior for prior in all_records
            if prior.decision == TrialDecision.KEPT and prior.budget_index <= record.budget_index
        ]
        logged = record.parsed_metrics.get(node_spec.metric_name)
        row = {
            "campaign_id": record.campaign_id,
            "trial_id": record.trial_id,
            "node_id": record.node_id,
            "metric_name": node_spec.metric_name,
            "logged_metric": "" if logged is None else f"{logged:.9f}",
            "reexecuted_metric": "",
            "abs_delta": "",
            "within_tolerance": "false",
            "status": "",
            "message": "",
        }
        missing_patch = next((prior.patch_ref for prior in cumulative_kept if not Path(prior.patch_ref).exists()), "")
        if missing_patch:
            row["status"] = "skipped"
            row["message"] = f"patch_ref not found: {missing_patch}"
            rows.append(row)
            continue

        with tempfile.TemporaryDirectory(prefix="autoresearch_reexec_") as tmp:
            node_copy = Path(tmp) / node_source.name
            ignore = shutil.ignore_patterns(".autoresearch_artifacts", "__pycache__", "artifacts")
            shutil.copytree(node_source, node_copy, ignore=ignore)
            _restore_baseline(node_copy, node_source, node_spec.editable_paths)

            for prior in cumulative_kept:
                patch = subprocess.run(
                    ["patch", "-p1", "-i", str(prior.patch_ref)],
                    cwd=node_copy,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if patch.returncode != 0:
                    row["status"] = "patch_failed"
                    row["message"] = (
                        f"while applying {prior.trial_id}: "
                        + (patch.stdout + "\n" + patch.stderr).strip()
                    )
                    rows.append(row)
                    break
            if row["status"] == "patch_failed":
                continue

            run = subprocess.run(
                node_spec.run_command,
                cwd=node_copy,
                shell=True,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            combined = run.stdout + "\n" + run.stderr
            metrics = _parse_metrics(combined)
            observed = metrics.get(node_spec.metric_name)
            if run.returncode != 0 or observed is None or logged is None:
                row["status"] = "run_failed"
                row["message"] = combined[-500:]
                rows.append(row)
                continue

            delta = abs(observed - logged)
            row["reexecuted_metric"] = f"{observed:.9f}"
            row["abs_delta"] = f"{delta:.9f}"
            row["within_tolerance"] = str(delta <= args.tolerance).lower()
            row["status"] = "ok"
            rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "campaign_id",
            "trial_id",
            "node_id",
            "metric_name",
            "logged_metric",
            "reexecuted_metric",
            "abs_delta",
            "within_tolerance",
            "status",
            "message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reproduced = sum(row["within_tolerance"] == "true" for row in rows)
    print(f"wrote {len(rows)} reexecution rows to {output}")
    print(f"reproduced_within_tolerance={reproduced}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
