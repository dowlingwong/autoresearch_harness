"""Stub-manager smoke of a ported Track G node on the census host (checklist E), with the real data.

Run from the harness root, after the node's setup_command, with the node's Python environment:

    PYTHONPATH=src python3 -m tests.trackg.port_smoke_gpu --node mlab_cifar10 \
        --gpu-uuid GPU-xxxxxxxx --cpus 0-7 --memory-limit "<what enforces the memory cap>" \
        --runs-root /path/outside/the/repo/smoke_mlab_cifar10 --provisional-timeout 1800

It runs, at the node's full budget and with seeds 8001/8001: the baseline, a frozen-path diff, the
node's broken diff (from its suite test module's SMOKE table) and E1 through the stub manager, then
every declared edit once on the baseline, then a second baseline.  It checks checklist E (baseline ok,
frozen diff refused with scope_violation, broken diff divergent, complete run records, not_forced
only dataloader_workers for torch nodes, the patterns see the baseline log) and prints one JSON line
per run.  The timeout is provisional (first, measuring run): record it as such.  ENGINEERING_ONLY.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import port_support as ps

SUITE_MODULES = ("test_port_mlagentbench", "test_port_mlgym", "test_port_autoresearch")


def smoke_entry(node: str) -> dict:
    for name in SUITE_MODULES:
        try:
            module = importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError:
            continue
        table = getattr(module, "SMOKE", {})
        if node in table:
            return table[node]
    raise SystemExit(f"no SMOKE entry for {node} in {SUITE_MODULES}")


def cpu_list(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node", required=True)
    parser.add_argument("--gpu-uuid", default=None, help="GPU-... or MIG-...; omit only for a CPU-only node")
    parser.add_argument("--cpus", required=True, help="CPU list, e.g. 0-7 or 0,2,4")
    parser.add_argument("--memory-limit", required=True, help="what enforces the memory cap (recorded)")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--provisional-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    entry = smoke_entry(args.node)
    work = ps.node_dir(args.node)
    if not (work / "data").is_symlink() and entry.get("make") is not None:
        raise SystemExit(f"{work}/data is missing: run the node's setup_command first")
    lim = ps.limits(args.provisional_timeout, gpu_uuids=(args.gpu_uuid,) if args.gpu_uuid else (),
                    cpu_list=cpu_list(args.cpus), external_limit=args.memory_limit)
    broken = ps.broken_diff(args.node, *entry["broken"])
    records, per_edit = ps.run_smoke(args.node, args.runs_root, work, None, broken, run_limits=lim)
    for entry_ in records:
        print(json.dumps({"run": entry_["run_id"], "decision": entry_["decision"],
                          "outcome": entry_["record"].get("outcome")}, default=str))
    for record in per_edit:
        print(json.dumps({"run": record["run_id"], "outcome": record.get("outcome"),
                          "elapsed_seconds_info_only": record.get("elapsed_seconds_info_only")}, default=str))
    ps.check_smoke(records, per_edit, entry.get("divergent", set()), torch_node=entry.get("torch_node", True))
    print("CHECKLIST_E_PASSED", args.node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
