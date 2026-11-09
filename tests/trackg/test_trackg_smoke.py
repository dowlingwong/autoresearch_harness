"""G.1 step 4 support: stub-manager smoke campaign on a fake node."""
from __future__ import annotations

import json

import pytest

from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.proposal_v2_diff import DiffProposal
from autoresearch.trackg.runner import ResourceLimits
from autoresearch.trackg.smoke import run_smoke_campaign

from .fixtures import make_node, spec_mapping


def lr(old, new):
    return f"--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = {old}\n+LEARNING_RATE = {new}\n"


def test_smoke_campaign_refuses_frozen_edits_detects_classes_and_keeps_improvements(tmp_path):
    node = make_node(tmp_path)
    spec = TrackGNodeSpec.from_mapping(spec_mapping())
    frozen = "--- a/evaluate.py\n+++ b/evaluate.py\n@@ -1,1 +1,1 @@\n-def evaluate(preds, labels):\n+def evaluate(p, l):\n"
    nan = ('--- a/train.py\n+++ b/train.py\n@@ -7,1 +7,1 @@\n-MODE = "ok"\n+MODE = "nan"\n')
    proposals = [
        frozen,                                   # refused
        DiffProposal("r", "lr up", "raise lr", lr("0.1", "0.2")),  # kept (0.7 > 0.6)
        lr("0.1", "0.2"),                         # context no longer matches the new incumbent
        lr("0.2", "0.15"),                        # worse -> discarded
        lr("0.2", "0.15"),                        # duplicate of an evaluated state
        nan,                                      # divergent
        lr("0.2", "0.2  # same"),                 # inert
    ]
    records = run_smoke_campaign(spec, node, tmp_path / "runs", proposals, ResourceLimits(measured_run_seconds=10))
    decisions = [r["decision"] for r in records]
    assert decisions == ["baseline", "not_run_refused", "kept", "not_run_refused", "discarded",
                         "not_run_duplicate", "discarded_divergent", "not_run_inert"]
    assert records[1]["record"]["diff"]["refusal"] == "scope_violation"
    assert records[3]["record"]["diff"]["refusal"] == "context_mismatch"
    assert records[2]["proposal"]["schema_version"] == "manager_proposal/v2-diff"
    assert all("sealed" not in r["record"] for r in records)
    ledger = [json.loads(line) for line in (tmp_path / "runs" / "smoke_ledger.jsonl").read_text().splitlines()]
    assert [e["decision"] for e in ledger] == decisions
    assert "LEARNING_RATE = 0.1" in (node / "train.py").read_text()
    with pytest.raises(FileExistsError):
        run_smoke_campaign(spec, node, tmp_path / "runs", [], ResourceLimits(measured_run_seconds=10))
