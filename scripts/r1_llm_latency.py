#!/usr/bin/env python3
"""Summarise per-LLM-call telemetry from campaign event streams (R1 L4, PILOT).

Reads one or more ``*_events.jsonl`` files, extracts every ``llm_call`` event
(written by ``LangGraphManager`` via the control plane) and writes:

* ``llm_latency.csv`` — one row per call:
  ``campaign_id, trial_id, latency_ms, prompt_tokens, completion_tokens,
  response_chars, parse_ok``;
* ``llm_latency.json`` — a summary with counts, parse-failure rate and
  latency / token statistics.

This is PILOT instrument output, not evidence.

Usage:
  python3 scripts/r1_llm_latency.py \\
      --events experiments/r1/pilot/L4_.../real/events.jsonl \\
      --out experiments/r1/pilot/L4_...
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_calls(event_paths: list[Path]) -> list[dict]:
    calls: list[dict] = []
    for path in event_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event_type") != "llm_call":
                    continue
                payload = dict(event.get("payload") or {})
                calls.append(
                    {
                        "campaign_id": event.get("campaign_id"),
                        "trial_id": event.get("trial_id"),
                        "latency_ms": payload.get("latency_ms"),
                        "prompt_tokens": payload.get("prompt_tokens"),
                        "completion_tokens": payload.get("completion_tokens"),
                        "response_chars": payload.get("response_chars"),
                        "parse_ok": bool(payload.get("parse_ok", True)),
                    }
                )
    return calls


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", action="append", required=True, help="Event JSONL (repeatable).")
    parser.add_argument("--out", required=True, help="Output directory for the CSV and JSON.")
    args = parser.parse_args()

    event_paths = [Path(p) for p in args.events]
    missing = [str(p) for p in event_paths if not p.exists()]
    if missing:
        print(f"ERROR: event file(s) not found: {missing}", file=sys.stderr)
        return 1

    calls = _load_calls(event_paths)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "llm_latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "campaign_id",
                "trial_id",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "response_chars",
                "parse_ok",
            ],
        )
        writer.writeheader()
        writer.writerows(calls)

    latencies = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
    prompt_tokens = [c["prompt_tokens"] for c in calls if c["prompt_tokens"] is not None]
    completion_tokens = [c["completion_tokens"] for c in calls if c["completion_tokens"] is not None]
    parse_failures = sum(1 for c in calls if not c["parse_ok"])

    summary = {
        "label": "PILOT",
        "event_files": [str(p) for p in event_paths],
        "n_calls": len(calls),
        "n_parse_failures": parse_failures,
        "parse_failure_rate": (parse_failures / len(calls)) if calls else 0.0,
        "tokens_reported": bool(prompt_tokens or completion_tokens),
        "latency_ms": _stats(latencies),
        "prompt_tokens": _stats(prompt_tokens),
        "completion_tokens": _stats(completion_tokens),
    }
    json_path = out_dir / "llm_latency.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
