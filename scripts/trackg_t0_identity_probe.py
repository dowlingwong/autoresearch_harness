#!/usr/bin/env python3
"""T0 identity probe: one small ``manager_proposal/v1`` campaign on a CPU fixture.

Used by ``tests/trackg/test_trackg_t0_identity.py`` to show that the Track G
additions leave the T0 path byte-identical (M10 step 1).  The test runs this
same script twice, once with ``--harness-src`` pointing at the base commit's
``src`` and once at the current ``src``, in the same working directory, and
compares the outputs byte for byte.

What the probe does:

- copies the ``lr_synthetic`` node (from ``--node-dir``) into ``<work>/node``;
- starts a local fake llama-server (OpenAI-compatible ``/v1/chat/completions``)
  that returns a fixed sequence of ``manager_proposal/v1`` replies (valid
  proposals, one malformed reply) and stores every request body byte for byte;
- runs ``run_real_campaign`` with ``LangGraphManager(model="llamacpp/fixture",
  proposal_schema="manager_proposal/v1")``, the ``LocalWorker``, an event store
  and the default controller;
- writes ``requests/<n>.json`` (raw request bodies), ``trials.jsonl``,
  ``events.jsonl`` and a copy of the final ``train.py`` into ``<work>/out``.

The probe imports only from ``--harness-src``; nothing from Track G.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLIES = [
    {"rationale": "A larger step may converge faster.", "summary": "raise LR",
     "objective": "Change LEARNING_RATE from 5e-4 to 1e-3.", "param": "LEARNING_RATE",
     "old_value": "5e-4", "new_value": "1e-3"},
    {"rationale": "More epochs.", "summary": "more epochs",
     "objective": "Change N_EPOCHS from 80 to 120.", "param": "N_EPOCHS",
     "old_value": "80", "new_value": "120"},
    "this reply is not JSON",
    {"rationale": "Less regularisation.", "summary": "lower reg",
     "objective": "Change REGULARIZATION from 0.001 to 0.0005.", "param": "REGULARIZATION",
     "old_value": "0.001", "new_value": "0.0005"},
    {"rationale": "Bigger batches.", "summary": "batch 8",
     "objective": "Change BATCH_SIZE from 4 to 8.", "param": "BATCH_SIZE",
     "old_value": "4", "new_value": "8"},
]


def _serve(store: Path):
    counter = {"n": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                n = counter["n"]
                counter["n"] += 1
            (store / f"{n:03d}.json").write_bytes(raw)
            reply = REPLIES[n % len(REPLIES)]
            content = reply if isinstance(reply, str) else json.dumps(reply)
            payload = {"id": f"fixture-{n}", "object": "chat.completion", "created": 1, "model": "fixture",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    return http, thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-src", required=True, type=Path)
    parser.add_argument("--node-config", required=True, type=Path)
    parser.add_argument("--node-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.harness_src.resolve()))
    from autoresearch.control_plane.campaign import run_real_campaign
    from autoresearch.manager.langgraph_manager import LangGraphManager
    from autoresearch.memory.event_store import CampaignEventStore
    from autoresearch.nodes.spec import load_node_spec
    from autoresearch.worker.local_worker import LocalWorker

    work = args.work.resolve()
    if work.exists():
        raise SystemExit(f"{work} exists; the probe never reuses a work directory")
    node = work / "node"
    out = work / "out"
    requests = out / "requests"
    requests.mkdir(parents=True)
    shutil.copytree(args.node_dir, node, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    spec = load_node_spec(args.node_config)

    http, thread = _serve(requests)
    os.environ["LLAMACPP_BASE_URL"] = f"http://127.0.0.1:{http.server_address[1]}/v1"
    try:
        run_real_campaign(
            node_spec=spec, campaign_id="t0_identity", budget=args.budget,
            manager_mode="langgraph_manager", memory_mode="append_only_summary_with_rationale",
            records_path=out / "trials.jsonl",
            worker=LocalWorker(node_root=node, artifact_dir=work / "artifacts"),
            proposal_backend=LangGraphManager(model="llamacpp/fixture", proposal_schema="manager_proposal/v1",
                                              transport_retry=None),
            event_store=CampaignEventStore(out / "events.jsonl"),
        )
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
    shutil.copy2(node / "train.py", out / "final_train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
