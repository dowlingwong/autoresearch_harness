"""v2-diff manager and campaign path (M10 next step; synthetic fake node and fake LLM)."""
from __future__ import annotations

import http.server
import json
import threading

import pytest

from autoresearch.trackg.campaign_v2_diff import V2DiffCampaignError, run_v2_diff_campaign
from autoresearch.trackg.manager_v2_diff import V2DiffManager, build_prompt, openai_compatible_chat
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.runner import ResourceLimits

from .fixtures import make_node, spec_mapping

LIMITS = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=10)


def lr(old, new):
    return f"--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = {old}\n+LEARNING_RATE = {new}\n"


MODE_SLEEP = '--- a/train.py\n+++ b/train.py\n@@ -7,1 +7,1 @@\n-MODE = "ok"\n+MODE = "sleep"\n'
MODE_CRASH = '--- a/train.py\n+++ b/train.py\n@@ -7,1 +7,1 @@\n-MODE = "ok"\n+MODE = "crash"\n'


def reply(diff, summary="s"):
    return json.dumps({"rationale": "r", "summary": summary, "objective": "o", "diff": diff})


class ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


def spec():
    return TrackGNodeSpec.from_mapping(spec_mapping())


def test_flag_is_required(tmp_path):
    node = make_node(tmp_path)
    with pytest.raises(V2DiffCampaignError, match="enable_v2_diff=True"):
        run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(ScriptedChat([]), "m"), LIMITS, 1)
    assert not (tmp_path / "runs").exists()


def test_campaign_keeps_discards_and_records_invalid_and_not_run(tmp_path):
    node = make_node(tmp_path)
    chat = ScriptedChat([reply(lr("0.1", "0.2"), "lr up"), "not json", reply(lr("0.2", "0.15")),
                         reply(lr("0.2", "0.15")), reply(MODE_CRASH), reply(lr("0.2", "0.2  # same"))])
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(chat, "fixture", seed=7),
                                   LIMITS, 6, enable_v2_diff=True)
    assert [e["decision"] for e in entries] == ["baseline", "kept", "invalid_proposal", "discarded",
                                                "not_run_duplicate", "discarded_divergent", "not_run_inert"]
    assert entries[2]["manager_call"]["parse_error"] and entries[2]["runs"] == []
    assert entries[1]["proposal"]["schema_version"] == "manager_proposal/v2-diff"
    assert entries[-1]["best_after"] == pytest.approx(0.7)
    assert all("sealed" not in run for e in entries for run in e["runs"])
    ledger = [json.loads(s) for s in (tmp_path / "runs" / "campaign_ledger.jsonl").read_text().splitlines()]
    assert [e["decision"] for e in ledger] == [e["decision"] for e in entries]
    # every request asks for the v2-diff structure, temperature 0, a per-step seed
    assert all(p["response_format"]["json_schema"]["name"] == "manager_proposal_v2_diff" for p in chat.payloads)
    assert len({p["seed"] for p in chat.payloads}) == 6 and all(p["temperature"] == 0.0 for p in chat.payloads)
    # the history reaches later prompts, without test metrics
    last = chat.payloads[-1]["messages"][0]["content"]
    assert "step 1: lr up -> kept" in last and "test_accuracy" not in last.split("Earlier steps")[1]


def test_repeated_timeout_becomes_divergent_after_two_and_is_never_rerun(tmp_path):
    node = make_node(tmp_path)
    fast = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=0.5, timeout_factor=1.0,
                          kill_grace_seconds=0.2)
    chat = ScriptedChat([reply(MODE_SLEEP, "sleepy"), reply(MODE_SLEEP, "sleepy again")])
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(chat, "m"), fast, 2,
                                   enable_v2_diff=True)
    first = entries[1]
    assert [r["outcome"]["reason"] for r in first["runs"]] == ["timeout", "timeout"]
    assert first["final_outcome"]["kind"] == "divergent" and first["final_outcome"]["reason"] == "timeout_repeated"
    assert first["decision"] == "discarded_divergent"
    # the same code state proposed again is a duplicate of the evaluated (divergent) state: not run
    assert entries[2]["decision"] == "not_run_duplicate" and len(chat.payloads) == 2


def test_other_infrastructure_failures_rerun_and_do_not_count_toward_the_cap(tmp_path, monkeypatch):
    from autoresearch.trackg import campaign_v2_diff as camp
    from autoresearch.trackg.outcome import RunOutcome
    node = make_node(tmp_path)
    real = camp.WorkingCopyRunner.run

    def fake_run(self, run_id, limits, **kw):
        result = real(self, run_id, limits, **kw)
        if run_id.startswith("s001"):
            result.outcome = RunOutcome.infrastructure_failure("htcondor_eviction")
            result.record["outcome"] = result.outcome.to_dict()
        return result

    monkeypatch.setattr(camp.WorkingCopyRunner, "run", fake_run)
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(ScriptedChat([reply(lr("0.1", "0.3"))]), "m"),
                                   LIMITS, 1, enable_v2_diff=True, max_infra_reruns=2)
    step = entries[1]
    assert [r["run_id"] for r in step["runs"]] == ["s001", "s001_r1", "s001_r2"]
    assert step["decision"] == "infrastructure_failure_final" and step["final_outcome"]["reason"] == "htcondor_eviction"


def test_prompt_shows_editable_files_only():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        node = make_node(Path(d))
        text = build_prompt(spec(), node, [])
    assert "=== train.py ===" in text and "=== model/net.py ===" in text
    assert "evaluate.py" not in text.split("Editable files")[1] and "model/metric.py ===" not in text


def test_openai_compatible_transport_sends_the_payload():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": reply(lr("0.1", "0.2"))}}],
                               "usage": {"completion_tokens": 3}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        chat = openai_compatible_chat(f"http://127.0.0.1:{server.server_address[1]}/v1")
        content, usage = chat({"model": "m", "messages": []})
    finally:
        server.shutdown(); server.server_close()
    assert json.loads(content)["diff"].startswith("--- a/train.py") and usage == {"completion_tokens": 3}
    assert seen == [{"model": "m", "messages": []}]


def test_transport_failure_stops():
    def broken(payload):
        raise ConnectionError("down")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        node = make_node(Path(d))
        with pytest.raises(RuntimeError, match="STOP: v2-diff manager transport failed"):
            V2DiffManager(broken, "m").propose(spec(), node, [], 1)
