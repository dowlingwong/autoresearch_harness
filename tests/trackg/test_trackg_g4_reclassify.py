"""R1-G5-DIFF-RECOUNT: ``scripts/trackg_g4_reclassify.py`` on a synthetic sealed G.4 attempt.

The attempt is built the way the G.4 job builds one: a G.4-rule campaign (no recount, as on ``c025f89a``) writes
``out/pilot/instance_<k>/campaign_ledger.jsonl`` and its run directories; the job tars ``out`` into ``out.tar`` and
writes ``SHA256SUMS`` (out.tar, job_record.json, server_record.json); the seal copies them and writes
``SEAL_RECEIPT.json``; the report's captures hold copies of the sums and the receipt."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from autoresearch.trackg import campaign_v2_diff
from autoresearch.trackg.campaign_v2_diff import DiffRecount, run_v2_diff_campaign
from autoresearch.trackg.manager_v2_diff import V2DiffManager
from autoresearch.trackg.node_template import load_trackg_node_spec
from autoresearch.trackg.runner import ResourceLimits

from .fixtures import make_node, write_spec

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "trackg_g4_reclassify.py"
LIMITS = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=10)
H = "--- a/train.py\n+++ b/train.py\n"
REPLIES = [
    H + "@@ -8,3 +8,3 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.15\n",               # wrong counts -> applied
    H + "@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n",                # kept (the G.4 rule too)
    H + "@@ -8,4 +8,4 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.3\n",                # wrong context now
    H + "@@ -7,1 +7,1 @@\n MODE = \"ok\"\n-LEARNING_RATE = 0.2\n+LEARNING_RATE = 0.25\n",  # too small: malformed_diff
    H + "@@ -8,2 +8,2 @@\n-LEARNING_RATE = 0.2\n+LEARNING_RATE = 0.1\n",                # back to the baseline: dup
    H + "@@ -8,3 +8,3 @@\n-LEARNING_RATE = 0.2\n*LEARNING_RATE = 0.3\n",                # a bad tag: stays refused
]


def _load_script():
    spec = importlib.util.spec_from_file_location("trackg_g4_reclassify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScriptedChat:
    def __init__(self, replies):
        self.replies = [json.dumps({"rationale": "r", "summary": "s", "objective": "o", "diff": d}) for d in replies]

    def __call__(self, payload):
        return self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.fixture()
def sealed(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    (harness / "configs" / "trackg" / "nodes").mkdir(parents=True)
    node = make_node(harness / "nodes")                              # harness/nodes/node
    spec_path = write_spec(harness / "configs" / "trackg" / "nodes" / "fake_task.yaml")
    spec = load_trackg_node_spec(spec_path)
    job = tmp_path / "runs" / "g4-test" / "fake_task" / "attempt-1"
    out = job / "out"
    out.mkdir(parents=True)
    pilot = {"schema": "trackg_g4_pilot/v1", "label": "ENGINEERING_ONLY", "node": "fake_task",
             "spec_path": "configs/trackg/nodes/fake_task.yaml", "node_dir": "nodes/node",
             "node_spec_sha256": spec.raw_sha256, "interpreter": "python3"}
    (out / "pilot.json").write_text(json.dumps(pilot, indent=1) + "\n")
    # the G.4 rule: the campaign as on c025f89a passed the manager's text to the runner unchanged
    monkeypatch.setattr(campaign_v2_diff, "recount_hunk_headers", lambda text: DiffRecount(text=text))
    entries = run_v2_diff_campaign(spec, node, out / "pilot" / "instance_1", V2DiffManager(ScriptedChat(REPLIES), "m"),
                                   LIMITS, len(REPLIES), enable_v2_diff=True, unisolated_reason="test fixture")
    monkeypatch.undo()
    ledger = out / "pilot" / "instance_1" / "campaign_ledger.jsonl"
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    for e in lines:
        e.pop("diff_recount", None)                                   # G.4 ledgers have no such block
    ledger.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in lines))
    (out / "pilot" / "pilot_summary.json").write_text('{"note": "not read"}\n')
    assert [e["decision"] for e in entries] == ["baseline", "not_run_refused", "kept", "not_run_refused",
                                                "not_run_refused", "not_run_refused", "not_run_refused"]
    with tarfile.open(job / "out.tar", "w") as tar:                    # tar -cf out.tar out
        tar.add(out, arcname="out")
    (job / "job_record.json").write_text(json.dumps({"schema": "g4_tasl_job_record/1", "node": "fake_task"}) + "\n")
    (job / "server_record.json").write_text("{}\n")
    sums = "".join(f"{_sha(job / n)}  {n}\n" for n in ("out.tar", "job_record.json", "server_record.json"))
    (job / "SHA256SUMS").write_text(sums)
    target = tmp_path / "sealed_g4" / "g4-test" / "fake_task" / "attempt-1"
    target.mkdir(parents=True)
    for name in ("out.tar", "job_record.json", "SHA256SUMS", "server_record.json"):
        shutil.copy2(job / name, target / name)
    receipt = {"schema": "g4_tasl_seal_receipt/1", "run_id": "g4-test", "node": "fake_task", "attempt": 1,
               "stop": None, "job_record_json_sha256": _sha(target / "job_record.json")}
    (target / "SEAL_RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    captures = tmp_path / "captures" / "fake_task"
    captures.mkdir(parents=True)
    shutil.copy2(job / "SHA256SUMS", captures / "SHA256SUMS")
    shutil.copy2(target / "SHA256SUMS", captures / "sealed_SHA256SUMS")
    shutil.copy2(target / "SEAL_RECEIPT.json", captures / "SEAL_RECEIPT.json")
    return harness, target, captures


def test_reclassifies_the_refused_proposals_offline(sealed, tmp_path, capsys):
    harness, attempt, captures = sealed
    mod = _load_script()
    out = tmp_path / "result.json"
    rc = mod.main(["--harness-root", str(harness), "--pair", str(attempt), str(captures), "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    result = json.loads(out.read_text())
    assert result["schema"] == "trackg_g4_reclassify/v1" and result["label"] == "ENGINEERING_ONLY"
    node = result["nodes"][0]
    assert node["members_read"] == ["out/pilot.json", "out/pilot/instance_1/campaign_ledger.jsonl"]
    assert node["verification"]["out_tar_sha256"] == _sha(attempt / "out.tar")
    assert node["proposals"] == 6 and node["accepted_by_g4_rule"] == 1 and node["accepted_text_unchanged"] == 1
    assert node["refused_before"] == {"malformed_diff": 1, "malformed_hunk": 4}
    assert node["g4_rule_replay"] == {"matches_ledger": 5, "mismatches": []}
    assert [(r["step"], r["before"], r["new"]) for r in node["steps"]] == [
        (1, "malformed_hunk", "applied"),
        (3, "malformed_hunk", "refused:context_mismatch"),
        (4, "malformed_diff", "applied"),
        (5, "malformed_hunk", "duplicate"),
        (6, "malformed_hunk", "refused:malformed_hunk")]
    assert node["new_rule_runnable"] == 2 and node["with_recounted_headers"] == 4
    assert node["by_reason"]["malformed_hunk"] == {"applied": 1, "duplicate": 1, "refused:context_mismatch": 1,
                                                   "refused:malformed_hunk": 1}
    assert node["left_to_strict_parser"] == {"bad_line_tag": 1}
    assert result["total"]["refused_before"] == node["refused_before"]
    # no metric, no diff text in the output
    text = out.read_text()
    assert "LEARNING_RATE" not in text and "accuracy" not in text
    # never overwritten
    assert mod.main(["--harness-root", str(harness), "--pair", str(attempt), str(captures), "--out", str(out)]) == 2


def test_a_bare_out_tar_is_accepted(sealed, capsys):
    harness, attempt, captures = sealed
    assert _load_script().main(["--harness-root", str(harness), "--pair", str(attempt / "out.tar"),
                                str(captures)]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"][0]["new_rule_runnable"] == 2


def test_a_tampered_tar_or_receipt_stops(sealed, tmp_path, capsys):
    harness, attempt, captures = sealed
    mod = _load_script()
    bad = tmp_path / "bad"
    shutil.copytree(attempt, bad)
    (bad / "SEAL_RECEIPT.json").write_text("{}\n")
    assert mod.main(["--harness-root", str(harness), "--pair", str(bad), str(captures)]) == 2
    assert "SEAL_RECEIPT" in capsys.readouterr().err
    with open(bad / "out.tar", "ab") as fh:
        fh.write(b"\0" * 512)
    assert mod.main(["--harness-root", str(harness), "--pair", str(bad / "out.tar"), str(captures)]) == 2
    assert "sha256" in capsys.readouterr().err


def test_a_different_node_spec_stops(sealed, capsys):
    harness, attempt, captures = sealed
    write_spec(harness / "configs" / "trackg" / "nodes" / "fake_task.yaml", description="another description")
    assert _load_script().main(["--harness-root", str(harness), "--pair", str(attempt), str(captures)]) == 2
    assert "node spec" in capsys.readouterr().err


def test_a_changed_kit_is_reported_not_classified(sealed, capsys):
    harness, attempt, captures = sealed
    (harness / "nodes" / "node" / "train.py").write_text("print('other kit')\n")
    rc = _load_script().main(["--harness-root", str(harness), "--pair", str(attempt), str(captures)])
    result = json.loads(capsys.readouterr().out)
    assert rc == 3 and result["nodes"][0]["instance_stops"]["instance_1"].startswith("the node kit")
    assert result["nodes"][0]["steps"] == []
