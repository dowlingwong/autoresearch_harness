"""The ``manager_proposal/v2-diff`` schema (R1-BX-D4; R1-BX-PROPOSED-CONFIRM D4 (E3)).

GPU nodes take code-level diffs.  A v2-diff proposal is a JSON object with
exactly four string fields, in this order: ``rationale`` (first, as in v1, so
a causal model writes its reasoning before the edit), ``summary``,
``objective`` and ``diff`` (a unified diff against the incumbent).

This module is separate from ``autoresearch.llm.providers`` on purpose: the
v1 schema function there refuses every other version, and the T0 path that
uses it stays byte-identical.  Wiring v2-diff into a manager is not part of
this module.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

MANAGER_PROPOSAL_V2_DIFF = "manager_proposal/v2-diff"
V2_DIFF_FIELDS = ("rationale", "summary", "objective", "diff")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ProposalParseError(ValueError):
    """The reply does not match ``manager_proposal/v2-diff``; it is recorded, never repaired."""


@dataclass(frozen=True)
class DiffProposal:
    rationale: str
    summary: str
    objective: str
    diff: str
    schema_version: str = MANAGER_PROPOSAL_V2_DIFF


def manager_response_format_v2_diff() -> dict:
    """Structure-only ``response_format`` for an OpenAI-compatible server (like v1's)."""
    return {"type": "json_schema", "json_schema": {
        "name": "manager_proposal_v2_diff", "strict": True,
        "schema": {"type": "object", "properties": {k: {"type": "string"} for k in V2_DIFF_FIELDS},
                   "required": list(V2_DIFF_FIELDS), "additionalProperties": False}}}


def parse_v2_diff_reply(content: str) -> DiffProposal:
    """Parse a model reply strictly: one JSON object with exactly the four string fields.

    A leading ``<think>…</think>`` block is removed (as the v1 path does for
    reasoning models); nothing else is cleaned or guessed.
    """
    if not isinstance(content, str):
        raise ProposalParseError("reply must be text")
    text = _THINK_RE.sub("", content).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProposalParseError(f"reply is not one JSON object: {error}") from error
    if not isinstance(payload, dict) or set(payload) != set(V2_DIFF_FIELDS):
        raise ProposalParseError(f"{MANAGER_PROPOSAL_V2_DIFF} requires exactly the fields {V2_DIFF_FIELDS}")
    if any(not isinstance(payload[k], str) for k in V2_DIFF_FIELDS):
        raise ProposalParseError(f"{MANAGER_PROPOSAL_V2_DIFF} fields must all be strings")
    return DiffProposal(**{k: payload[k] for k in V2_DIFF_FIELDS})
