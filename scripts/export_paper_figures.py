#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.memory.append_store import TrialAppendStore
from autoresearch.reporting.export_figures import export_figure_csvs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export deterministic paper figure CSVs.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--records")
    parser.add_argument("--out-dir", default=str(ROOT / "paper" / "figures"))
    args = parser.parse_args()
    records_path = Path(args.records) if args.records else ROOT / "experiments" / "ledgers" / f"{args.campaign_id}_trials.jsonl"
    records = TrialAppendStore(records_path).read_all()
    outputs = export_figure_csvs(records, args.out_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

