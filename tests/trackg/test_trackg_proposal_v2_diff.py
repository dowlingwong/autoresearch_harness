"""M10 step 1: the manager_proposal/v2-diff schema; v1 is untouched."""
from __future__ import annotations

import json

import pytest

from autoresearch.llm.providers import MANAGER_PROPOSAL_SCHEMA_VERSION, manager_response_format
from autoresearch.trackg.proposal_v2_diff import (
    MANAGER_PROPOSAL_V2_DIFF,
    ProposalParseError,
    manager_response_format_v2_diff,
    parse_v2_diff_reply,
)

GOOD = {"rationale": "r", "summary": "s", "objective": "o", "diff": "--- a/x.py\n+++ b/x.py\n"}


def test_response_format_is_structure_only_and_rationale_first():
    fmt = manager_response_format_v2_diff()
    schema = fmt["json_schema"]["schema"]
    assert fmt["json_schema"]["name"] == "manager_proposal_v2_diff"
    assert list(schema["properties"]) == ["rationale", "summary", "objective", "diff"]
    assert schema["additionalProperties"] is False
    assert all(v == {"type": "string"} for v in schema["properties"].values())


def test_strict_parse():
    proposal = parse_v2_diff_reply("<think>private</think>" + json.dumps(GOOD))
    assert proposal.diff == GOOD["diff"] and proposal.schema_version == MANAGER_PROPOSAL_V2_DIFF
    for bad in ("not json", json.dumps({**GOOD, "extra": "x"}), json.dumps({**GOOD, "diff": 3}),
                json.dumps({k: v for k, v in GOOD.items() if k != "diff"}), json.dumps([GOOD]),
                "```json\n" + json.dumps(GOOD) + "\n```"):
        with pytest.raises(ProposalParseError):
            parse_v2_diff_reply(bad)


def test_v1_schema_function_is_unchanged_and_still_refuses_other_versions():
    assert MANAGER_PROPOSAL_SCHEMA_VERSION == "manager_proposal/v1"
    assert manager_response_format()["json_schema"]["name"] == "manager_proposal_v1"
    with pytest.raises(ValueError):
        manager_response_format(MANAGER_PROPOSAL_V2_DIFF)
