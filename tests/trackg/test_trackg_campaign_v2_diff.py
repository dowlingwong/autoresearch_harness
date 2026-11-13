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
    assert entries[0]["runs"][0]["outcome"]["kind"] == "ok"            # a real ok baseline under the short timeout
    first = entries[1]
    assert [r["outcome"]["reason"] for r in first["runs"]] == ["timeout", "timeout"]
    assert first["final_outcome"]["kind"] == "divergent" and first["final_outcome"]["reason"] == "timeout_repeated"
    assert first["decision"] == "discarded_divergent"
    # the same code state proposed again is a duplicate of the evaluated (divergent) state: not run
    assert entries[2]["decision"] == "not_run_duplicate" and len(chat.payloads) == 2


def _scripted_parse(monkeypatch, plan):
    """Script outcomes at the log-parsing step: plan maps a run-id prefix to a list of outcomes, used in turn."""
    from autoresearch.trackg import runner as runner_mod
    real = runner_mod.parse_run_log
    queues = {k: list(v) for k, v in plan.items()}

    def fake(log_path, process, spec):
        run_id = log_path.parent.name
        for prefix, queue in queues.items():
            if run_id.startswith(prefix) and queue:
                return queue.pop(0)
        return real(log_path, process, spec)

    monkeypatch.setattr(runner_mod, "parse_run_log", fake)


def test_other_infrastructure_failures_rerun_and_do_not_count_toward_the_cap(tmp_path, monkeypatch):
    from autoresearch.trackg.outcome import RunOutcome
    node = make_node(tmp_path)
    infra = RunOutcome.infrastructure_failure("infrastructure_log_pattern", pattern="DUE TO TIME LIMIT")
    _scripted_parse(monkeypatch, {"s001": [infra, infra, infra]})
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(ScriptedChat([reply(lr("0.1", "0.3"))]), "m"),
                                   LIMITS, 1, enable_v2_diff=True, max_infra_reruns=2)
    step = entries[1]
    assert [r["run_id"] for r in step["runs"]] == ["s001", "s001_r1", "s001_r2"]
    assert all(r["executed"] is True for r in step["runs"])                 # really rerun, not refused
    assert step["decision"] == "infrastructure_failure_final"


def test_timeouts_interleaved_with_infra_failures_and_across_steps(tmp_path, monkeypatch):
    """One timeout, then an infra failure, then a timeout of the same code state: capped (2 timeouts); the infra
    rerun does not count. The count carries across steps: a code state that timed out once in step 1 (then ran ok
    on its rerun) is capped by its next timeout in a later step only if it is run again (here it is a duplicate)."""
    from autoresearch.trackg.outcome import RunOutcome
    node = make_node(tmp_path)
    t_out = RunOutcome.infrastructure_failure("timeout")
    infra = RunOutcome.infrastructure_failure("killed", signal=9)
    _scripted_parse(monkeypatch, {"s001": [t_out, infra, t_out]})
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs",
                                   V2DiffManager(ScriptedChat([reply(lr("0.1", "0.3"))]), "m"), LIMITS, 1,
                                   enable_v2_diff=True)
    step = entries[1]
    assert [r["outcome"]["reason"] for r in step["runs"]] == ["timeout", "killed", "timeout"]
    assert step["final_outcome"]["reason"] == "timeout_repeated" and step["decision"] == "discarded_divergent"


def test_baseline_failure_is_rerun_then_stops_with_a_ledger_line(tmp_path, monkeypatch):
    from autoresearch.trackg.campaign_v2_diff import V2DiffCampaignError
    from autoresearch.trackg.outcome import RunOutcome
    node = make_node(tmp_path)
    t_out = RunOutcome.infrastructure_failure("timeout")
    _scripted_parse(monkeypatch, {"s000_baseline": [t_out, t_out]})
    with pytest.raises(V2DiffCampaignError, match="baseline"):
        run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(ScriptedChat([]), "m"), LIMITS, 1,
                             enable_v2_diff=True)
    ledger = [json.loads(s) for s in (tmp_path / "runs" / "campaign_ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["decision"] == "baseline_failed"
    assert [r["run_id"] for r in ledger[0]["runs"]] == ["s000_baseline", "s000_baseline_r1"]


def test_baseline_infra_failure_then_ok_is_used(tmp_path, monkeypatch):
    from autoresearch.trackg.outcome import RunOutcome
    node = make_node(tmp_path)
    _scripted_parse(monkeypatch, {"s000_baseline": [RunOutcome.infrastructure_failure("killed", signal=9)]})
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs",
                                   V2DiffManager(ScriptedChat([reply(lr("0.1", "0.05"))]), "m"), LIMITS, 1,
                                   enable_v2_diff=True)
    assert entries[0]["decision"] == "baseline" and entries[0]["best_after"] == pytest.approx(0.6)
    assert entries[1]["decision"] == "discarded"                            # 0.55 < 0.6: not kept unconditionally


def test_extra_payload_cannot_override_pinned_fields_and_requests_are_recorded(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        V2DiffManager(ScriptedChat([]), "m", extra_payload={"seed": 1})
    node = make_node(tmp_path)
    chat = ScriptedChat([reply(lr("0.1", "0.2"))])
    entries = run_v2_diff_campaign(spec(), node, tmp_path / "runs",
                                   V2DiffManager(chat, "fixture", seed=3, extra_payload={"cache_prompt": False}),
                                   LIMITS, 1, enable_v2_diff=True)
    req = entries[1]["manager_call"]["request"]
    assert req["model"] == "fixture" and req["cache_prompt"] is False and req["seed"] == entries[1]["manager_call"]["seed"]
    assert "messages" not in req and entries[1]["manager_call"]["base_seed"] == 3


def test_transport_stop_is_recorded_in_the_ledger(tmp_path):
    from autoresearch.trackg.manager_v2_diff import ManagerTransportStop

    def broken(payload):
        raise ConnectionError("down")

    node = make_node(tmp_path)
    with pytest.raises(ManagerTransportStop):
        run_v2_diff_campaign(spec(), node, tmp_path / "runs", V2DiffManager(broken, "m"), LIMITS, 2,
                             enable_v2_diff=True)
    ledger = [json.loads(s) for s in (tmp_path / "runs" / "campaign_ledger.jsonl").read_text().splitlines()]
    assert [e["decision"] for e in ledger] == ["baseline", "stop_transport"]
    assert ledger[1]["manager_call"]["prompt_sha256"]


def test_prompt_shows_editable_files_only(tmp_path):
    import os
    node = make_node(tmp_path)
    (node / "model" / "leak.txt").symlink_to(node / "data" / "train.csv")       # link to a frozen file
    text = build_prompt(spec(), node, [])
    assert "=== train.py ===" in text and "=== model/net.py ===" in text
    body = text.split("Editable files")[1]
    assert "evaluate.py" not in body and "model/metric.py ===" not in body and "model/leak.txt" not in body
    assert "x,y" not in body
    # a frozen entry written with a leading ./ is still frozen
    dotted = TrackGNodeSpec.from_mapping(spec_mapping(frozen_paths=["evaluate.py", "data/", "./model/metric.py"]))
    assert "model/metric.py ===" not in build_prompt(dotted, node, []).split("Editable files")[1]


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
