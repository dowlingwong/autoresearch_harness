"""An LLM manager that proposes ``manager_proposal/v2-diff`` diffs for Track G nodes (M10; AI-assisted draft).

This is the manager the v2-diff campaign (:mod:`autoresearch.trackg.campaign_v2_diff`) uses. It is new
code inside ``autoresearch.trackg``; the v1 managers (``autoresearch.manager``) and
``autoresearch.llm.providers`` are untouched, so the T0 ``manager_proposal/v1`` path stays byte-identical.

One call per proposal:

* the prompt shows the node's editable files of the current incumbent (text files only, each capped),
  the metric name and direction, the run budget, and a short history of earlier steps (summary, status
  and outcome class; never the sealed test metric);
* the request asks for the structure-only ``response_format`` of v2-diff, ``temperature`` 0 and a seed
  derived from the campaign seed and the step, so a run is repeatable on a deterministic server;
* the reply is parsed strictly (:func:`parse_v2_diff_reply`). A reply that does not parse is an
  **invalid proposal**: it is returned with the error, recorded and charged; it is never repaired.

The transport is injectable (``chat``): a callable taking the request payload and returning
``(content, usage)``. :func:`openai_compatible_chat` is the stdlib HTTP transport for an
OpenAI-compatible server such as llama-server.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from autoresearch.trackg.diff_support import editable_files
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.proposal_v2_diff import (MANAGER_PROPOSAL_V2_DIFF, DiffProposal, ProposalParseError,
                                                  manager_response_format_v2_diff, parse_v2_diff_reply)

PROMPT_VERSION = "trackg_v2_diff_prompt/v1"
MAX_FILE_CHARS = 20000
MAX_HISTORY = 12

Chat = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]
PINNED_FIELDS = ("model", "messages", "temperature", "seed", "max_tokens", "response_format")


class ManagerTransportStop(RuntimeError):
    """The transport failed (not model output): the campaign records it and STOPs."""


def openai_compatible_chat(base_url: str, *, timeout: float = 600.0) -> Chat:
    """POST the payload to ``<base_url>/chat/completions``; return the message content and usage."""
    url = base_url.rstrip("/") + "/chat/completions"

    def chat(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read().decode())
        return str(reply["choices"][0]["message"]["content"]), dict(reply.get("usage") or {})

    return chat


def _editable_files(spec: TrackGNodeSpec, incumbent_dir: Path) -> list[tuple[str, str]]:
    """The incumbent's editable text files, chosen exactly as the scope validator chooses them.

    ``diff_support.editable_files`` walks the copy without following directory links and keeps the paths that
    ``validate_edit_scope`` accepts (editable and not frozen, after path normalisation). Symlinked files are
    skipped, so a link cannot pull a frozen or outside file into the prompt; undecodable files are skipped."""
    root = Path(incumbent_dir)
    files = []
    for rel in editable_files(root, spec.to_base_node_spec()):
        path = root / rel
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files.append((rel, text if len(text) <= MAX_FILE_CHARS else text[:MAX_FILE_CHARS] + "\n[... truncated]\n"))
    return files


def build_prompt(spec: TrackGNodeSpec, incumbent_dir: Path, history: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"You improve the training code of the task '{spec.name}'.",
        f"Goal: {spec.metric_direction} {spec.metric_name} on the validation split, within a fixed budget of "
        f"{spec.run_budget.value} {spec.run_budget.kind}. Do not change the budget, the data, the evaluation or "
        "any frozen file.",
        "Propose ONE change as a unified diff against the files below (paths relative to the task root, "
        "'--- a/<path>' and '+++ b/<path>' headers, exact context lines).",
        'Reply with one JSON object with exactly the string fields "rationale", "summary", "objective", "diff".',
        "",
        "Editable files of the current best version:",
    ]
    for rel, text in _editable_files(spec, incumbent_dir):
        lines += [f"=== {rel} ===", text.rstrip("\n"), f"=== end {rel} ==="]
    lines.append("")
    if history:
        lines.append("Earlier steps (most recent last):")
        for h in list(history)[-MAX_HISTORY:]:
            lines.append(f"- step {h['step']}: {h.get('summary') or '(no summary)'} -> {h['decision']}"
                         + (f" ({h['metric']:.6g})" if isinstance(h.get("metric"), (int, float)) else ""))
    else:
        lines.append("No earlier steps.")
    return "\n".join(lines) + "\n"


@dataclass
class V2DiffManager:
    """Propose one v2-diff per call through an injectable chat transport."""

    chat: Chat
    model: str
    seed: int = 0
    max_tokens: int = 4096
    extra_payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MANAGER_PROPOSAL_V2_DIFF

    def __post_init__(self) -> None:
        clash = sorted(set(self.extra_payload) & set(PINNED_FIELDS))
        if clash:
            raise ValueError(f"extra_payload must not override the pinned request fields {clash}")

    def propose(self, spec: TrackGNodeSpec, incumbent_dir: Path, history: Sequence[dict[str, Any]],
                step: int) -> tuple[DiffProposal | None, dict[str, Any]]:
        prompt = build_prompt(spec, incumbent_dir, history)
        seed = int(hashlib.sha256(f"{self.seed}:{step}".encode()).hexdigest()[:8], 16)
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "seed": seed, "max_tokens": self.max_tokens,
                   "response_format": manager_response_format_v2_diff(), **self.extra_payload}
        call = {"schema_version": self.schema_version, "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "seed": seed, "base_seed": self.seed,
                "request": {k: v for k, v in payload.items() if k != "messages"}}
        try:
            content, usage = self.chat(payload)
        except Exception as error:  # any failure inside the transport is not model output: the campaign STOPs
            stop = ManagerTransportStop(f"STOP: v2-diff manager transport failed: {type(error).__name__}: {error}")
            stop.call = call
            raise stop from error
        call.update(usage=usage, raw_reply_sha256=hashlib.sha256(str(content).encode()).hexdigest(),
                    raw_reply=content)
        try:
            proposal = parse_v2_diff_reply(content)
        except ProposalParseError as error:
            call["parse_error"] = str(error)
            return None, call
        return proposal, call
